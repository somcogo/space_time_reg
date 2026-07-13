import torch
from torch import nn
import numpy as np


class Siren(nn.Module):
    # from IDIR https://github.com/MIAGroupUT/IDIR/tree/main
    """This is a dense neural network with sine activation functions.

    The ODE time t is ignored, and all frame intervals are integrated with this same
    network, so choosing this model means one static velocity field shared across the
    whole time period. Use GroupedSiren for a per-interval (non-stationary) velocity.

    Arguments:
    layers -- ([*int]) amount of nodes in each layer of the network, e.g. [2, 16, 16, 1]
    gpu -- (boolean) use GPU when True, CPU when False
    weight_init -- (boolean) use special weight initialization if True
    omega -- (float) parameter used in the forward function
    """

    def __init__(self, layers, weight_init=True, last_init_zero=False, omega=30):
        """Initialize the network."""

        super(Siren, self).__init__()
        self.n_layers = len(layers) - 1
        self.omega = omega

        # Make the layers
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

            # Weight Initialization
            if weight_init:
                with torch.no_grad():
                    if i == 0:
                        self.layers[-1].weight.uniform_(-1 / layers[i], 1 / layers[i])
                    elif i == self.n_layers - 1 and last_init_zero:
                        val = 1/600
                        self.layers[-1].weight.uniform_(-val, val)
                        self.layers[-1].bias.zero_()
                    else:
                        self.layers[-1].weight.uniform_(
                            -np.sqrt(6 / layers[i]) / self.omega,
                            np.sqrt(6 / layers[i]) / self.omega,
                        )

        # Combine all layers to one model
        self.layers = nn.Sequential(*self.layers)

    def forward(self, t, x):
        """The forward function of the network."""

        # Perform relu on all layers except for the last one
        for layer in self.layers[:-1]:
            x = torch.sin(self.omega * layer(x))

        # Propagate through final layer and return the output
        return self.layers[-1](x)


class GroupedLinear(nn.Module):
    def __init__(self, groups, in_dim, out_dim):
        super().__init__()
        self.groups = groups
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weight = nn.Parameter(
            torch.empty(groups, out_dim, in_dim)
        )
        self.bias = nn.Parameter(
            torch.zeros(groups, out_dim)
        )

    def forward(self, x):
        # x: (T, N, in_dim)
        # weight: (T, out_dim, in_dim)
        out = torch.bmm(x, self.weight.transpose(1, 2))
        out = out + self.bias.unsqueeze(1)
        return out


class GroupedSiren(nn.Module):
    """
    T independent SIRENs evaluated in parallel.

    Arguments:
        groups -- number of independent SIRENs (T)
        layers -- list of layer sizes, e.g. [2, 16, 16, 1]
        weight_init -- use SIREN initialization
        last_init_zero -- apply small init to last layer
        omega -- frequency scaling
    """

    def __init__(self, groups, layers,
                 weight_init=True,
                 last_init_zero=False,
                 omega=30):

        super().__init__()
        self.groups = groups
        self.n_layers = len(layers) - 1
        self.omega = omega

        self.layers = nn.ModuleList()

        for i in range(self.n_layers):
            in_dim = layers[i]
            out_dim = layers[i + 1]

            layer = GroupedLinear(groups, in_dim, out_dim)
            self.layers.append(layer)

            if weight_init:
                with torch.no_grad():

                    if i == 0:
                        # First layer
                        layer.weight.uniform_(
                            -1 / in_dim,
                            1 / in_dim
                        )

                    elif i == self.n_layers - 1 and last_init_zero:
                        val = 1 / 600
                        layer.weight.uniform_(-val, val)
                        layer.bias.zero_()

                    else:
                        bound = np.sqrt(6 / in_dim) / self.omega
                        layer.weight.uniform_(-bound, bound)

    def forward(self, t, x):
        """
        x: (T, N, input_dim)
        """

        for layer in self.layers[:-1]:
            x = torch.sin(self.omega * layer(x))

        return self.layers[-1](x)