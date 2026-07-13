"""Adds a learnable regularization parameter and scaling to a base regularizer.

Vendored from https://github.com/johertrich/LearnedRegularizers and de-coupled from
``deepinv``: only ``g``/``grad`` are used, so this now subclasses ``torch.nn.Module``
directly instead of ``deepinv.optim.Prior``.
"""

import inspect

import torch


class ParameterLearningWrapper(torch.nn.Module):
    def __init__(
        self,
        regularizer,  # base regularizer R, equipped with a learnable parameter and scaling
        scale_init=0.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Define the regularizer

            R_tilde(x) = exp(alpha) / exp(s)**2 * R(exp(s) x)

        for a base regularizer R, where alpha and s are learnable parameters. The
        exponentials keep these scaling parameters positive.
        """
        super().__init__()
        self.regularizer = regularizer
        self.add_module("regularizer", self.regularizer)
        self.alpha = torch.nn.Parameter(
            torch.tensor(0.0, device=device, requires_grad=True)
        )
        self.scale = torch.nn.Parameter(
            torch.tensor(scale_init, device=device, requires_grad=True)
        )
        signature = inspect.signature(self.regularizer.grad)
        argument_names = [param.name for param in signature.parameters.values()]
        self.has_get_energy = "get_energy" in argument_names

    def g(self, x):
        return torch.exp(self.alpha - 2 * self.scale) * self.regularizer.g(
            torch.exp(self.scale) * x
        )

    def grad(self, x, get_energy=False):
        if not self.has_get_energy:
            if get_energy:
                return torch.exp(self.alpha - 2 * self.scale) * self.regularizer.g(
                    torch.exp(self.scale) * x
                ), torch.exp(self.alpha - self.scale) * self.regularizer.grad(
                    torch.exp(self.scale) * x
                )
            else:
                return torch.exp(self.alpha - self.scale) * self.regularizer.grad(
                    torch.exp(self.scale) * x
                )
        reg_out = self.regularizer.grad(torch.exp(self.scale) * x, get_energy=get_energy)
        if get_energy:
            return (
                torch.exp(self.alpha - 2 * self.scale) * reg_out[0],
                torch.exp(self.alpha - self.scale) * reg_out[1],
            )
        return torch.exp(self.alpha - self.scale) * reg_out
