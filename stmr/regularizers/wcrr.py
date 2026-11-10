"""(Weakly) Convex Ridge Regularizer, a special case of a Field-of-Experts regularizer.

Vendored from https://github.com/johertrich/LearnedRegularizers and de-coupled from
``deepinv``: the original subclassed ``deepinv.optim.Prior`` purely for ``nn.Module``
behaviour, but only ``g``/``grad`` (both overridden here) are ever called, so it now
subclasses ``torch.nn.Module`` directly. Local changes: the pointwise potential is shared by
``g`` and ``grad`` (``_potential``), its weak-convexity term is skipped at weight 0, and
``freeze`` caches the Lipschitz constant and can torch.compile the potential.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P


class ZeroMean(nn.Module):
    """Enforcing zero mean on the filters improves performance."""

    def forward(self, x):
        return x - torch.mean(x, dim=(1, 2, 3), keepdim=True)


class WCRR(nn.Module):
    def __init__(
        self,
        sigma,  # scaling constant (we set it always to 0.1)
        weak_convexity,  # set to 0.0 for CRR and to 1.0 for WCRR
        tanh=False,  # whether to use tanh instead of smoothed l1 as potential
        nb_channels=[1, 4, 8, 64],  # number channels in the multiconv
        filter_sizes=[5, 5, 5],  # filter sizes of the convolution
        device="cuda" if torch.cuda.is_available() else "cpu",  # device
        pretrained=None,  # if a str is given here, the weights from the corresponding path will be loaded
    ):
        super().__init__()

        self.nb_filters = nb_channels[-1]
        self.filter_size = sum(filter_sizes) - len(filter_sizes) + 1
        self.filters = nn.Sequential(
            *[
                nn.Conv2d(
                    nb_channels[i],
                    nb_channels[i + 1],
                    filter_sizes[i],
                    padding=filter_sizes[i] // 2,
                    bias=False,
                )
                for i in range(len(filter_sizes))
            ]
        )
        P.register_parametrization(self.filters[0], "weight", ZeroMean())

        self.dirac = torch.zeros(
            1, 1, 2 * self.filter_size - 1, 2 * self.filter_size - 1
        )
        self.dirac[0, 0, self.filter_size - 1, self.filter_size - 1] = 1.0

        self.scaling = nn.Parameter(
            torch.log(torch.tensor(2.0) / sigma) * torch.ones(1, self.nb_filters, 1, 1)
        )
        self.beta = nn.Parameter(torch.tensor(4.0))
        self.weak_cvx = weak_convexity
        self.tanh = tanh
        self._frozen = False  # see freeze()
        self._lip = None
        self._pointwise = self._potential

        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained, map_location=device))

    def smooth_l1(self, x):
        if self.tanh:
            x_abs = torch.abs(x)
            return torch.log((torch.exp(x - x_abs) + torch.exp(-x - x_abs)) / 2) + x_abs
        return torch.clip(x**2, 0.0, 1.0) / 2 + torch.clip(torch.abs(x), 1.0) - 1.0

    def grad_smooth_l1(self, x):
        if self.tanh:
            return torch.tanh(x)
        else:
            return torch.clip(x, -1.0, 1.0)

    def get_conv_lip(self):
        impulse = self.filters(self.dirac)
        for filt in reversed(self.filters):
            impulse = F.conv_transpose2d(impulse, filt.weight, padding=filt.padding)
        return torch.fft.fft2(impulse, s=[256, 256]).abs().max()

    def freeze(self, compile_potential=False):
        """For fixed (pretrained) filters: stop their gradients, compute the Lipschitz constant
        once instead of in every conv / conv_transpose (filter stack + 256x256 FFT each), and
        optionally fuse the pointwise potential (_potential) into one kernel with torch.compile."""
        for p in self.parameters():
            p.requires_grad_(False)
        self._frozen = True
        if compile_potential:
            self._pointwise = torch.compile(self._potential)

    def _conv_lip(self):
        if not self._frozen:
            return self.get_conv_lip()
        if self._lip is None:  # lazily, in the current dtype / device (reset by _apply)
            self._lip = self.get_conv_lip().detach()
        return self._lip

    def conv(self, x):
        x = x / torch.sqrt(self._conv_lip())
        return self.filters(x)

    def conv_transpose(self, x):
        x = x / torch.sqrt(self._conv_lip())
        for filt in reversed(self.filters):
            x = F.conv_transpose2d(x, filt.weight, padding=filt.padding)
        return x

    def _potential(self, z, get_energy, get_grad):
        """At the scaled filter responses z = exp(scaling) * conv(x): the per-image energy and
        the derivative that conv_transpose maps back to x (each None unless asked for).
        The weak-convexity term is skipped when its weight is 0 (the plain CRR)."""
        reg = grad = None
        if get_energy:
            reg = self.smooth_l1(torch.exp(self.beta) * z) * torch.exp(-self.beta)
            if self.weak_cvx:
                reg = reg - self.smooth_l1(z) * self.weak_cvx
            reg = reg * torch.exp(-2 * self.scaling)
            reg = reg.sum(dim=(1, 2, 3))
        if get_grad:
            grad = self.grad_smooth_l1(torch.exp(self.beta) * z)
            if self.weak_cvx:
                grad = grad - self.grad_smooth_l1(z) * self.weak_cvx
            grad = grad * torch.exp(-self.scaling)
        return reg, grad

    def _potential_safe(self, z, get_energy, get_grad):
        if self._pointwise is not self._potential:  # compiled (freeze)
            try:
                return self._pointwise(z, get_energy, get_grad)
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as err:  # e.g. no working triton: eager computes the same
                warnings.warn(f"torch.compile of the WCRR potential failed, using eager: {err!r}")
                self._pointwise = self._potential
        return self._potential(z, get_energy, get_grad)

    def grad(self, x, get_energy=False):
        z = self.conv(x) * torch.exp(self.scaling)
        reg, grad = self._potential_safe(z, get_energy, True)
        grad = self.conv_transpose(grad)
        if get_energy:
            return reg, grad
        return grad

    def g(self, x):
        z = self.conv(x) * torch.exp(self.scaling)
        return self._potential_safe(z, True, False)[0]

    def _apply(self, fn):
        self.dirac = fn(self.dirac)
        self._lip = None  # recomputed in the new dtype / on the new device
        return super()._apply(fn)
