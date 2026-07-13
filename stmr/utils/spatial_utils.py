import numpy as np
import torch
from torch import nn


def generate_coord_tensor(dims, device, min_coord=-1, max_coord=1):
    coordinate_tensor = [torch.linspace(min_coord, max_coord, dims[i]) for i in range(len(dims))]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing='ij')
    coordinate_tensor = torch.stack(coordinate_tensor, dim=-1)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), len(dims)]).to(device)
    return coordinate_tensor


def get_relative_vel(config, func: nn.Module, coord_tensor: torch.Tensor,
                     time_points: torch.Tensor, keep_batch_dim) -> torch.Tensor | None:
    return func(time_points[-1], coord_tensor)
