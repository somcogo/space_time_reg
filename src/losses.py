import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.normalized_gradient_field import NormalizedGradientField2d, NormalizedGradientField3d, spatial_filter_nd, _grad_param

def calculate_losses(config, abs_phi, rel_vel, func, imgs, neural_reps, time_series, coord_tensor, losses, downsample):
    # abs_phi: [T, H*W, D]
    # rel_vel: [T or 1, H*W, D]
    imgs = imgs.to(config.device)
    neural_reps = [net.to(config.device) for net in neural_reps if net is not None]

    shape = [-1] + list(imgs.shape[1:]) + [len(imgs.shape) - 1]
    loss_sum = 0.
    moved_imgs = None
    for loss_type, loss_dict in losses.items():
        t1 = time.time()
        l, m = calc_single_loss(loss_type, imgs, neural_reps, abs_phi, rel_vel, func, coord_tensor, shape, config, time_series, downsample)
        t2 = time.time()

        moved_imgs = m if m is not None else moved_imgs
        mean = l.mean()
        loss_dict['mean'] = mean.detach().cpu()
        loss_dict['loss'] = l.detach()
        loss_dict['time'] += t2 - t1
        loss_sum  = loss_sum + loss_dict['lambda'] * mean

    return loss_sum, losses, moved_imgs

def calc_single_loss(loss_name, imgs, neural_reps, abs_phi, rel_vel, func, coord_tensor, shape, config, time_series, downsample):
    if loss_name == 'sim':
        return similarity_loss(imgs, neural_reps, abs_phi, config, downsample)
    elif loss_name == 'negJ':
        return negJ_loss(abs_phi, coord_tensor, shape, downsample)
    elif loss_name == 'grd':
        return vel_grad_loss(rel_vel, func, coord_tensor, shape, config, time_series, downsample)
    elif loss_name == 'lap':
        return vel_lap_loss(rel_vel, func, coord_tensor, shape, config, time_series, downsample)
    elif loss_name == 'pgr':
        return phi_grad_loss(abs_phi, coord_tensor, shape, downsample)

def similarity_loss(imgs, neural_reps, abs_phi, args, downsample):
    if args.loss == 'mse':
        loss_fn = nn.MSELoss(reduction='none')
    elif args.loss == 'ngf':
        if imgs.dim() == 3:
            loss_fn = NormalizedGradientField2d(mm_spacing=1, eps=1e-6, reduction='none')
        else:
            loss_fn = NormalizedGradientField3d(mm_spacing=1, eps=1e-6, reduction='none')
    loss_fn = loss_fn.to(args.device)

    if args.use_nreps:
        if args.dataset in ['rot_slow2_large', 'rot_slow2_64', 'mouse']:
            net = neural_reps[0]
            model_out = net(torch.tensor([], device=abs_phi.device), abs_phi) # T, H, W, 2
            moved = model_out.squeeze(2)
            moved = moved.reshape(imgs.shape)
            loss = loss_fn(imgs.unsqueeze(1), moved.unsqueeze(1))
        else:
            net = neural_reps[0].net
            model_out = net(abs_phi)
            moved = (model_out.squeeze(2) + 1) / 2
            moved = moved.reshape(imgs.shape)
            loss = loss_fn(imgs.unsqueeze(1), moved.unsqueeze(1))
    else:
        img = imgs[:1].unsqueeze(0).expand(abs_phi.shape[0], 1, *imgs.shape[1:])
        grid = abs_phi.reshape(abs_phi.shape[0], *imgs.shape[1:], len(imgs.shape[1:]))
        grid = torch.stack([grid[..., i] for i in reversed(range(grid.shape[-1]))], dim=-1)
        moved = F.grid_sample(img, grid, align_corners=False)
        loss = loss_fn(imgs.unsqueeze(1), moved)
        moved = moved.squeeze(1)
    
    return loss * downsample, moved.detach().cpu()

def negJ_loss(abs_phi, coord_tensor, shape, downsample):
    rel_phi = abs_phi - coord_tensor
    phi_reshaped = rel_phi.reshape(shape)
    phi_J = fin_diff_Jacobian(phi_reshaped)
    loss = neg_Jdet_loss(phi_J) / downsample**2
    return loss, None

def vel_grad_loss(rel_vel, func, coord_tensor, shape, config, time_series, downsample):
    if config.fin_diff_grad:
        vel_reshaped = rel_vel.reshape(shape)
        J = fin_diff_Jacobian(vel_reshaped)
    elif config.autograd_grid:
        J = get_Jacobian(func, time_series, dims=shape[-1], coord_tensor=coord_tensor)
    else:
        J = get_Jacobian(func, time_series, dims=shape[-1], coord_tensor=None)
    loss = torch.linalg.vector_norm(J, dim=-1) / downsample
    return loss, None

def vel_lap_loss(rel_vel, func, coord_tensor, shape, config, time_series, downsample):
    if config.fin_diff_grad:
        vel_reshaped = rel_vel.reshape(shape)
        J = fin_diff_Jacobian(vel_reshaped)
        lap = fin_Laplacian_from_Jac(J)
    elif config.autograd_grid:
        lap = get_Laplacian(func, time_series, dims=shape[-1], coord_tensor=coord_tensor)
    else:
        lap = get_Laplacian(func, time_series, dims=shape[-1], coord_tensor=None)
    loss = torch.linalg.vector_norm(lap, dim=-1) / downsample
    return loss, None

def phi_grad_loss(abs_phi, coord_tensor, shape, downsample):
    rel_phi = abs_phi - coord_tensor
    phi_reshaped = rel_phi.reshape(shape)
    phi_J = fin_diff_Jacobian(phi_reshaped)
    loss = torch.linalg.vector_norm(phi_J, dim=-1) / downsample
    return loss, None

def neg_Jdet_loss(J):
    Jdet = torch.det(J)
    # Jdet = JacboianDet(J)
    neg_Jdet = -1.0 * Jdet
    neg_Jdet = F.relu(neg_Jdet) + 1
    out = torch.log(neg_Jdet)

    # out = - torch.log(Jdet)

    # out = torch.log(Jdet)
    # # out = out ** 2
    # out = torch.abs(out)
    # out = torch.exp(out)
    # out = out - 1
    # out = torch.abs(out)

    return out

def fin_diff_Jacobian(f):
    partial_grads = [fin_diff_gradient(f, dim) for dim in range(len(f.shape)-2)]
    J = torch.stack(partial_grads, dim=-1)
    # J shape: [T or 1, H, W, f_dim, spatial_dim]
    return J

def fin_Laplacian_from_Jac(J):
    lap = sum([fin_diff_gradient(J[..., i], i) for i in range(J.shape[-1])])
    return lap

def fin_diff_gradient(f, axis):
    dims = len(f.shape) - 2
    if dims == 2:
        f = f.permute(0, 3, 1, 2)
    elif dims == 3:
        f = f.permute(0, 4, 1, 2, 3)
    b, c = f.shape[:2]
    spatial_shape = f.shape[2:]

    # [B*N, H, W]
    f = f.reshape(b * c, 1, *spatial_shape)
    grad_kernel = _grad_param(dims, 'default', axis=axis).to(f.device)
    grad = spatial_filter_nd(f, grad_kernel)
    grad = grad.view(b, c, *spatial_shape)
    if dims == 2:
        grad = grad.permute(0, 2, 3, 1)
    elif dims == 3:
        grad = grad.permute(0, 2, 3, 4, 1)
    return grad

def get_Jacobian(func, time_series, dims, coord_tensor=None):
    if coord_tensor is not None:
        coords = coord_tensor
    else:
        coords = torch.rand((1024, dims), requires_grad=True, device=time_series.device)*2 - 1
    rel_vel = []
    for t in time_series:
        vel = func(t, coords)
        rel_vel.append(vel)
    rel_vel = torch.concat(rel_vel)
    # J shape: [T or 1, N, f_dim, spatial_dim]
    return torch.stack([gradient(input=coords, output=rel_vel[:, i]) for i in range(dims)], dim=1)

def get_Laplacian(func, time_series, dims, coord_tensor=None):
    if coord_tensor is not None:
        coords = coord_tensor
    else:
        coords = torch.rand((1024, dims), requires_grad=True, device=time_series.device)*2 - 1
    rel_vel = []
    for t in time_series:
        vel = func(t, coords)
        rel_vel.append(vel)
    rel_vel = torch.concat(rel_vel) # [T or 1, N, f_dim]
    # J shape: [N, f_dim, spatial_dim]
    J = torch.stack([gradient(input=coords, output=rel_vel[:, i]) for i in range(dims)], dim=1)

    dxdxy = [gradient(input=coords, output=J[..., i, 0]) for i in range(dims)]
    ddxx = torch.stack([dxy[..., 0] for dxy in dxdxy], dim=-1)
    dydxy = [gradient(input=coords, output=J[..., i, 1]) for i in range(dims)]
    ddyy = torch.stack([dxy[..., 1] for dxy in dydxy], dim=-1)
    return ddxx + ddyy

def get_phi_Jacobian(phi, img_shape, coords):
    return torch.stack([gradient(input=coords, output=phi[-1][..., i]) for i in range(len(img_shape))], dim=1)

def gradient(input, output, grad_outputs=None):
    """Compute the gradient of the output wrt the input."""

    grad_outputs = torch.ones_like(output)
    grad = torch.autograd.grad(
        output, [input], grad_outputs=grad_outputs, create_graph=True
    )[0]
    return grad