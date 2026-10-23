"""SIREN velocity networks.

A ``Siren`` is a single sine-activated MLP (originally from IDIR,
https://github.com/MIAGroupUT/IDIR). ``GroupedSiren`` evaluates ``groups`` independent
SIRENs in parallel over a leading batch dimension, giving one stationary velocity field
per time interval (the non-stationary ensemble described in the theory).
"""

import math

import numpy as np
import torch
from torch import nn


def init_last_layer(weight, bias, std):
    # Near-zero initial velocity. std > 0: weights ~ N(0, std^2), e.g. std 1e-6 gives ~1e-3 px of
    # initial motion while the weights stay random. std = 0: the original U(+-1/600), whose random
    # field already moves ~1 px on average (SIREN features are O(1), so the output is ~weight scale).
    if std > 0:
        weight.normal_(0, std)
    else:
        val = 1 / 600
        weight.uniform_(-val, val)
    bias.zero_()


class Siren(nn.Module):
    """Dense MLP with sine activations.

    Args:
        layers: node counts per layer, e.g. ``[2, 256, 256, 2]``.
        weight_init: use the SIREN weight initialisation.
        last_init_zero: initialise the last layer to output near-zero velocity.
        omega: frequency scaling inside the activation.
        last_init_std: with last_init_zero, std of a normal last-layer init (0: U(+-1/600)).
    """

    def __init__(self, layers, weight_init=True, last_init_zero=False, omega=30, last_init_std=0.0):
        super().__init__()
        self.n_layers = len(layers) - 1
        self.omega = omega

        modules = []
        for i in range(self.n_layers):
            layer = nn.Linear(layers[i], layers[i + 1])
            if weight_init:
                with torch.no_grad():
                    if i == 0:
                        layer.weight.uniform_(-1 / layers[i], 1 / layers[i])
                    elif i == self.n_layers - 1 and last_init_zero:
                        init_last_layer(layer.weight, layer.bias, last_init_std)
                    else:
                        bound = np.sqrt(6 / layers[i]) / self.omega
                        layer.weight.uniform_(-bound, bound)
            modules.append(layer)
        self.layers = nn.Sequential(*modules)

    def forward(self, t, x):
        for layer in self.layers[:-1]:
            x = torch.sin(self.omega * layer(x))
        return self.layers[-1](x)


class GroupedLinear(nn.Module):
    """Batched linear layer with independent weights per group."""

    # The N points of each group are split into up to this many chunks (see forward).
    max_chunks = 64

    def __init__(self, groups, in_dim, out_dim):
        super().__init__()
        self.groups = groups
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.empty(groups, out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(groups, out_dim))

    def forward(self, x, scale=1.0):
        # x: (groups, N, in_dim); weight: (groups, out_dim, in_dim). Returns
        # scale * (x @ weight^T + bias); scale and bias are fused into the baddbmm.
        # The weight gradient reduces over all N points into a tiny (out_dim, in_dim) matrix,
        # which cuBLAS runs as ~one tile per batch entry: with only `groups` entries the GPU
        # idles (~45% of an epoch at 204x512). Splitting N into chunks that share their
        # group's weights fixes that; autograd sums the per-chunk weight grads through expand.
        G, N, _ = x.shape
        c = math.gcd(N, self.max_chunks)
        x = x.reshape(G * c, N // c, self.in_dim)
        weight = self.weight.unsqueeze(1).expand(-1, c, -1, -1).reshape(G * c, self.out_dim, self.in_dim)
        bias = self.bias.unsqueeze(1).expand(-1, c, -1).reshape(G * c, 1, self.out_dim)
        out = torch.baddbmm(bias, x, weight.transpose(1, 2), beta=scale, alpha=scale)
        return out.reshape(G, N, self.out_dim)


class GroupedSiren(nn.Module):
    """``groups`` independent SIRENs evaluated in parallel over the batch dimension.

    Args:
        groups: number of independent SIRENs (one per time interval).
        layers: layer sizes, e.g. ``[2, 256, 256, 2]``.
        weight_init: use the SIREN weight initialisation.
        last_init_zero: initialise the last layer to output near-zero velocity.
        omega: frequency scaling inside the activation.
        last_init_std: with last_init_zero, std of a normal last-layer init (0: U(+-1/600)).
    """

    def __init__(self, groups, layers, weight_init=True, last_init_zero=False, omega=30,
                 last_init_std=0.0):
        super().__init__()
        self.groups = groups
        self.n_layers = len(layers) - 1
        self.omega = omega

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            in_dim, out_dim = layers[i], layers[i + 1]
            layer = GroupedLinear(groups, in_dim, out_dim)
            if weight_init:
                with torch.no_grad():
                    if i == 0:
                        layer.weight.uniform_(-1 / in_dim, 1 / in_dim)
                    elif i == self.n_layers - 1 and last_init_zero:
                        init_last_layer(layer.weight, layer.bias, last_init_std)
                    else:
                        bound = np.sqrt(6 / in_dim) / self.omega
                        layer.weight.uniform_(-bound, bound)
            self.layers.append(layer)

    def forward(self, t, x):
        # x: (groups, N, input_dim)
        for layer in self.layers[:-1]:
            x = torch.sin(layer(x, self.omega))
        return self.layers[-1](x)
