import numpy as np
import torch
from torch import nn


def generate_coord_tensor(dims, device, min_coord=-1, max_coord=1):
    if dims[0] == 1:
        dims = dims[1:]
    coordinate_tensor = [torch.linspace(min_coord, max_coord, dims[i]) for i in range(len(dims))]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing='ij')
    coordinate_tensor = torch.stack(coordinate_tensor, dim=-1)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), len(dims)]).to(device)
    return coordinate_tensor


def motion_grid(shape) -> tuple:
    """The grid a deformation lives on for the spatial image shape ``shape``: a single slice
    (1, H, W) only moves in-plane, so its grid is (H, W) -- the squeeze generate_coord_tensor
    applies. Use it to reshape a flat phi [B, N, C] to [B, *grid, C]."""
    shape = tuple(shape)
    return shape[1:] if shape[0] == 1 else shape


def get_relative_vel(config, func: nn.Module, coord_tensor: torch.Tensor,
                     time_points: torch.Tensor, keep_batch_dim) -> torch.Tensor | None:
    return func(time_points[-1], coord_tensor)
