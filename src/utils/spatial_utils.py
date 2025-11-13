import torch
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

def apply_grid_sample(input_img, phi, mode='bilinear'):
    grid = phi.reshape(phi.shape[0], *input_img.shape[2:], len(input_img.shape[2:]))
    grid = torch.stack([grid[..., i] for i in reversed(range(grid.shape[-1]))], dim=-1)
    moved = F.grid_sample(input_img, grid, align_corners=False, mode=mode)
    return moved

def get_relative_vel(func, config, time_points, coord_tensor, keep_batch_dim):
    if config.fin_diff_grad and config.lambda_grd + config.lambda_negJ + config.lambda_lap + config.lambda_hel > 0:
        if (config.func_name == 'siren' or config.func_name == 'wire'):
            rel_vel = func(time_points[1], coord_tensor).unsqueeze(0)
        elif ('siren' in config.func_name or 'wire' in config.func_name) and 't' in config.func_name:
            if keep_batch_dim:
                rel_vel = []
                for t in time_points:
                    rel_vel.append(func(t, coord_tensor))
                rel_vel = torch.stack(rel_vel)
            else:
                rel_vel = func(time_points[0], coord_tensor)
                for t in time_points[1:]:
                    rel_vel = rel_vel + func(t, coord_tensor)
                rel_vel = rel_vel.unsqueeze(0)
    else:
        rel_vel = None
    return rel_vel