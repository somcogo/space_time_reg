import torch
from torch import nn
import numpy as np

class WireReal(nn.Module):
    def __init__(self, layers, weight_init=True, omega=30., scale=10.):
        """Initialize the network."""
        super().__init__()
        self.n_layers = len(layers) - 1
        self.omega = omega
        self.scale = scale

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

        # Perform relu on all layers except for the last one
        for layer in self.layers[:-1]:
            lin = layer(x)
            omega = self.omega * lin
            scale = self.scale * lin
            x = torch.sin(omega) * torch.exp(-scale.abs().square())

        # Propagate through final layer and return the output
        return self.layers[-1](x)


class WireRealT(nn.Module):
    def __init__(self, layers, weight_init=True, omega=30., scale=10.):
        """Initialize the network."""

        super().__init__()
        layers[0] = layers[0] + 1
        self.n_layers = len(layers) - 1
        self.omega = omega
        self.scale = scale

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
            lin = layer(x)
            omega = self.omega * lin
            scale = self.scale * lin
            x = torch.sin(omega) * torch.exp(-scale.abs().square())

        # Propagate through final layer and return the output
        return self.layers[-1](x)
    
    
class WireRealLateT(nn.Module):
    def __init__(self, layers, weight_init=True, omega=30., scale=10.):
        """Initialize the network."""

        super().__init__()
        # layers[-2] = layers[-2] + 1
        self.n_layers = len(layers) - 1
        self.omega = omega
        self.scale = scale

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
            lin = layer(x)
            omega = self.omega * lin
            scale = self.scale * lin
            x = torch.sin(omega) * torch.exp(-scale.abs().square())
        
        # Add time coordinate before last layer
        x = torch.concat([t.unsqueeze(0).expand(x.shape[0], 1), x], dim=-1)

        # Propagate through final layer and return the output
        return self.layers[-1](x)
    