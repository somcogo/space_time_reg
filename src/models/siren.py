import torch
from torch import nn
import numpy as np


class Siren(nn.Module):
    # from IDIR https://github.com/MIAGroupUT/IDIR/tree/main
    """This is a dense neural network with sine activation functions.

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


class SirenT(nn.Module):
    # from IDIR https://github.com/MIAGroupUT/IDIR/tree/main
    """This is a dense neural network with sine activation functions.

    Arguments:
    layers -- ([*int]) amount of nodes in each layer of the network, e.g. [2, 16, 16, 1]
    gpu -- (boolean) use GPU when True, CPU when False
    weight_init -- (boolean) use special weight initialization if True
    omega -- (float) parameter used in the forward function
    """

    def __init__(self, layers, weight_init=True, omega=30):
        """Initialize the network."""

        super(SirenT, self).__init__()
        layers[0] = layers[0] + 1
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
                    else:
                        self.layers[-1].weight.uniform_(
                            -np.sqrt(6 / layers[i]) / self.omega,
                            np.sqrt(6 / layers[i]) / self.omega,
                        )

        # Combine all layers to one model
        self.layers = nn.Sequential(*self.layers)

    def forward(self, t, x):
        """The forward function of the network."""

        x = torch.concat([t.unsqueeze(0).expand(x.shape[0], 1), x], dim=-1)
        # Perform relu on all layers except for the last one
        for layer in self.layers[:-1]:
            x = torch.sin(self.omega * layer(x))

        # Propagate through final layer and return the output
        return self.layers[-1](x)
    
class SirenLateT(nn.Module):
    # from IDIR https://github.com/MIAGroupUT/IDIR/tree/main
    """This is a dense neural network with sine activation functions.

    Arguments:
    layers -- ([*int]) amount of nodes in each layer of the network, e.g. [2, 16, 16, 1]
    gpu -- (boolean) use GPU when True, CPU when False
    weight_init -- (boolean) use special weight initialization if True
    omega -- (float) parameter used in the forward function
    """

    def __init__(self, layers, weight_init=True, omega=30):
        """Initialize the network."""

        super().__init__()
        # layers[-2] = layers[-2] + 1
        self.n_layers = len(layers) - 1
        self.omega = omega

        # Make the layers
        self.layers = []
        for i in range(self.n_layers):
            if i == self.n_layers - 1:
                self.layers.append(nn.Linear(layers[i] + 1, layers[i + 1]))
            else:
                self.layers.append(nn.Linear(layers[i], layers[i + 1]))

            # Weight Initialization
            if weight_init:
                with torch.no_grad():
                    if i == 0:
                        self.layers[-1].weight.uniform_(-1 / layers[i], 1 / layers[i])
                    elif i == self.n_layers - 1:
                        self.layers[-1].weight.uniform_(
                            -np.sqrt(6 / (layers[i] + 1)) / self.omega,
                            np.sqrt(6 / (layers[i] + 1)) / self.omega,
                        )
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
        
        # Add time coordinate before last layer
        x = torch.concat([t.unsqueeze(0).expand(x.shape[0], 1), x], dim=-1)

        # Propagate through final layer and return the output
        return self.layers[-1](x)

class SirenEnsemble(nn.Module):
    def __init__(self, time_points, layers, weight_init=True, last_init_zero=False, omega=30):
        super().__init__()
        self.time_points = time_points
        self.sirens = nn.ModuleList([
            Siren(layers, weight_init, last_init_zero, omega) for i in range(len(time_points) - 1)
        ])

    def get_model_index(self, t):
        with torch.no_grad():
            if t < self.time_points[0]:
                return 0
            elif t >= self.time_points[-1]:
                return len(self.time_points) - 2
            else:
                return (self.time_points <= t ).sum() - 1
        
    def forward(self, t, x):
        model_index = self.get_model_index(t)
        sub_model = self.sirens[model_index]
        return sub_model(t, x)