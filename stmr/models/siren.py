"""SIREN velocity networks.

A ``Siren`` is a single sine-activated MLP (originally from IDIR,
https://github.com/MIAGroupUT/IDIR). ``GroupedSiren`` evaluates ``groups`` independent
SIRENs in parallel over a leading batch dimension, giving one stationary velocity field
per time interval (the non-stationary ensemble described in the theory).
"""

import numpy as np
import torch
from torch import nn


class Siren(nn.Module):
    """Dense MLP with sine activations.

    Args:
        layers: node counts per layer, e.g. ``[2, 256, 256, 2]``.
        weight_init: use the SIREN weight initialisation.
        last_init_zero: initialise the last layer to output near-zero velocity.
        omega: frequency scaling inside the activation.
    """

    def __init__(self, layers, weight_init=True, last_init_zero=False, omega=30):
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
                        val = 1 / 600
                        layer.weight.uniform_(-val, val)
                        layer.bias.zero_()
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

    def __init__(self, groups, in_dim, out_dim):
        super().__init__()
        self.groups = groups
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.empty(groups, out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(groups, out_dim))

    def forward(self, x):
        # x: (groups, N, in_dim); weight: (groups, out_dim, in_dim)
        return torch.bmm(x, self.weight.transpose(1, 2)) + self.bias.unsqueeze(1)


class GroupedSiren(nn.Module):
    """``groups`` independent SIRENs evaluated in parallel over the batch dimension.

    Args:
        groups: number of independent SIRENs (one per time interval).
        layers: layer sizes, e.g. ``[2, 256, 256, 2]``.
        weight_init: use the SIREN weight initialisation.
        last_init_zero: initialise the last layer to output near-zero velocity.
        omega: frequency scaling inside the activation.
    """

    def __init__(self, groups, layers, weight_init=True, last_init_zero=False, omega=30):
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
                        val = 1 / 600
                        layer.weight.uniform_(-val, val)
                        layer.bias.zero_()
                    else:
                        bound = np.sqrt(6 / in_dim) / self.omega
                        layer.weight.uniform_(-bound, bound)
            self.layers.append(layer)

    def forward(self, t, x):
        # x: (groups, N, input_dim)
        for layer in self.layers[:-1]:
            x = torch.sin(self.omega * layer(x))
        return self.layers[-1](x)
