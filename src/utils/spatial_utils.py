from argparse import Namespace

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def generate_coord_tensor(dims, device, min_coord=-1, max_coord=1):
    coordinate_tensor = [torch.linspace(min_coord, max_coord, dims[i]) for i in range(len(dims))]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing='ij')
    coordinate_tensor = torch.stack(coordinate_tensor, dim=-1)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), len(dims)]).to(device)
    return coordinate_tensor

def upsample_img_seg(img, seg, config, epoch):
    ndx = config.schedule.index(epoch)
    downsample = config.downsamples[ndx]
    new_shape = [l // downsample for l in img.shape[1:]]
    mode = 'bilinear' if len(img.shape) == 3 else 'trilinear'
    antialias = mode == 'bilinear'
    new_img = F.interpolate(img.unsqueeze(1), size=new_shape, mode=mode, antialias=antialias).squeeze(1)
    new_seg = F.interpolate(seg.unsqueeze(1), size=new_shape, mode='nearest-exact').squeeze(1) if seg is not None else seg
    return new_img, new_seg, downsample

def get_relative_vel(config: Namespace, func: nn.Module, coord_tensor: torch.Tensor, time_points: torch.Tensor, keep_batch_dim) -> torch.Tensor|None:
    rel_vel = func(time_points[-1], coord_tensor)
    return rel_vel