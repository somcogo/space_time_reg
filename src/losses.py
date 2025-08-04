import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.utils import generate_coord_tensor
from src.normalized_gradient_field import NormalizedGradientField2d, NormalizedGradientField3d, spatial_filter_nd, _grad_param

def calculate_losses(config, abs_phi, data, rel_vel, func, time_series, coord_tensor, last_val=False):
    # abs_phi: [T, H*W, D]
    # rel_vel: [T or 1, H*W, D]
    imgs, neural_reps, _ = data
    imgs = imgs.to(config.device)
    neural_reps = [net.to(config.device) for net in neural_reps]
    times = np.zeros(10) if config.debug or last_val else np.zeros(5)
    vel_shape = [-1] + list(imgs.shape[1:]) + [len(imgs.shape) - 1]

    times[0] = time.time()
    sim_energy, moved_imgs = similarity_loss(imgs, neural_reps, abs_phi, config)
    loss_sim = config.lambda_st * sim_energy.mean()
    t_series_for_autograd = time_series if config.func_name == 'sirent' else time_series[:1]

    times[1] = time.time()

    if config.debug or last_val:
        vel_reshaped = rel_vel.reshape(vel_shape)
        rel_phi = abs_phi - coord_tensor
        phi_reshaped = rel_phi.reshape(vel_shape)
        findiff_J =  fin_diff_Jacobian(vel_reshaped)
        times[2] = time.time()
        autograd_grid_J = get_Jacobian(func, t_series_for_autograd, dims=len(imgs.shape)-1, img_shape=imgs.shape[1:], coord_tensor=coord_tensor)
        times[3] = time.time()
        autograd_rand_J = get_Jacobian(func, t_series_for_autograd, dims=len(imgs.shape)-1, img_shape=imgs.shape[1:], coord_tensor=None)
        times[4] = time.time()
        fin_laplacian = fin_Laplacian_from_Jac(findiff_J)
        times[5] = time.time()
        auto_grid_laplacian = get_Laplacian(func, t_series_for_autograd, dims=len(imgs.shape)-1, coord_tensor=coord_tensor)
        times[6] = time.time()
        auto_rand_laplacian = get_Laplacian(func, t_series_for_autograd, dims=len(imgs.shape)-1, coord_tensor=None)
        times[7] = time.time()
        phi_fin_J = fin_diff_Jacobian(phi_reshaped)
        times[8] = time.time()
        # phi_auto_grid_J = get_phi_Jacobian(rel_phi, imgs.shape[1:], coords=coord_tensor)
        times[9] = time.time()

        phi_J = phi_fin_J
        if config.fin_diff_grad:
            J = findiff_J
            lap = fin_laplacian
            # phi_J = phi_fin_J
        elif config.autograd_grid:
            J = autograd_grid_J
            lap = auto_grid_laplacian
            # phi_J = phi_auto_grid_J
        else:
            J = autograd_rand_J
            lap = auto_rand_laplacian
            # phi_J = phi_auto_grid_J
        # findiff_J: [T or 1, H, W, D, D]
        # autograd_J: [T or 1, H*W, D, D]
        neg_Jdet_energy = neg_Jdet_loss(J).mean()
        vel_grad_l2 = torch.linalg.vector_norm(J, dim=-1).mean()
        lap_l2 = torch.linalg.vector_norm(lap, dim=-1).mean()
        phi_grad_l2 = torch.linalg.vector_norm(phi_J, dim=-1).mean()

        energies = [sim_energy.detach(), findiff_J.detach(), autograd_grid_J.detach(), fin_laplacian.detach(), auto_grid_laplacian.detach(), phi_fin_J.detach()]
        loss_values = {'sim':sim_energy.detach().mean().cpu(),
                       
                       'fin_diff_vel_grad_norm':torch.linalg.vector_norm(findiff_J.detach(), dim=-1).mean().cpu(),
                       'auto_grid_vel_grad_norm':torch.linalg.vector_norm(autograd_grid_J.detach(), dim=-1).mean().cpu(),
                       'auto_rand_vel_grad_norm':torch.linalg.vector_norm(autograd_rand_J.detach(), dim=-1).mean().cpu(),

                       'fin_diff_vel_neg_det_J':neg_Jdet_loss(findiff_J.detach()).mean().cpu(),
                       'auto_grid_vel_neg_det_J':neg_Jdet_loss(autograd_grid_J.detach()).mean().cpu(),
                       'auto_rand_vel_neg_det_J':neg_Jdet_loss(autograd_rand_J.detach()).mean().cpu(),

                       'fin_diff_vel_lap_norm':torch.linalg.vector_norm(fin_laplacian.detach(), dim=-1).mean().cpu(),
                       'auto_grid_vel_lap_norm':torch.linalg.vector_norm(auto_grid_laplacian.detach(), dim=-1).mean().cpu(),
                       'auto_rand_vel_lap_norm':torch.linalg.vector_norm(auto_rand_laplacian.detach(), dim=-1).mean().cpu(),

                       'fin_diff_phi_grad_norm':torch.linalg.vector_norm(phi_fin_J.detach(), dim=-1).mean().cpu()}
    else:

        if config.lambda_negJ > 0 or config.lambda_grd > 0:
            if config.fin_diff_grad:
                vel_reshaped = rel_vel.reshape(vel_shape)
                J = fin_diff_Jacobian(vel_reshaped)
            elif config.autograd_grid:
                J = get_Jacobian(func, t_series_for_autograd, dims=len(imgs.shape)-1, img_shape=imgs.shape[1:], coord_tensor=coord_tensor)
            else:
                J = get_Jacobian(func, t_series_for_autograd, dims=len(imgs.shape)-1, img_shape=imgs.shape[1:], coord_tensor=None)
            neg_Jdet_energy = neg_Jdet_loss(J).mean()
            vel_grad_l2 = torch.linalg.vector_norm(J, dim=-1).mean()
        else:
            neg_Jdet_energy = torch.tensor(0)
            vel_grad_l2 = torch.tensor(0)
        times[2] = time.time()
        
        if config.lambda_lap > 0:
            if config.fin_diff_grad:
                vel_reshaped = rel_vel.reshape(vel_shape)
                J = fin_diff_Jacobian(vel_reshaped)
                lap = fin_Laplacian_from_Jac(J)
            elif config.autograd_grid:
                lap = get_Laplacian(func, t_series_for_autograd, dims=len(imgs.shape)-1, coord_tensor=coord_tensor)
            else:
                lap = get_Laplacian(func, t_series_for_autograd, dims=len(imgs.shape)-1, coord_tensor=None)
            lap_l2 = torch.linalg.vector_norm(lap, dim=-1).mean()
        else:
            lap_l2 = torch.tensor(0)
        times[3] = time.time()
        
        if config.lambda_pgr > 0:
            rel_phi = abs_phi - coord_tensor
            phi_reshaped = rel_phi.reshape(vel_shape)
            # if config.fin_diff_grad:
            phi_J = fin_diff_Jacobian(phi_reshaped)
            # else:
            #     phi_J = get_phi_Jacobian(rel_phi, imgs.shape[1:], coords=coord_tensor)
            phi_grad_l2 = torch.linalg.vector_norm(phi_J, dim=-1).mean()
        else:
            phi_grad_l2 = torch.tensor(0)
        times[4] = time.time()
        
        energies = [sim_energy.detach()]
        loss_values = {'sim':sim_energy.detach().mean().cpu(),
                       'vel_neg_det_J':neg_Jdet_energy.detach().cpu(),
                       'vel_grad_norm':vel_grad_l2.detach().cpu(),
                       'vel_lap_norm':lap_l2.detach().cpu(),
                       'phi_grad_norm':phi_grad_l2.detach().cpu()}

    loss_negJ = config.lambda_negJ * neg_Jdet_energy
    loss_grd = config.lambda_grd * vel_grad_l2
    loss_lap = config.lambda_lap * lap_l2
    loss_pgr = config.lambda_pgr * phi_grad_l2

    return [loss_sim, loss_negJ, loss_grd, loss_lap, loss_pgr], moved_imgs, energies, times, loss_values

def similarity_loss(imgs, neural_reps, abs_phi, args):
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
        img = imgs[:1].unsqueeze(0).expand(abs_phi.shape[0], 1, imgs.shape[1], imgs.shape[2])
        grid = abs_phi.reshape(abs_phi.shape[0], imgs.shape[1], imgs.shape[2], 2).permute(0, 2, 1, 3)
        moved = F.grid_sample(img, grid, align_corners=False)
        loss = loss_fn(imgs.unsqueeze(1), moved)
        moved = moved.squeeze(1)
    
    return loss, moved.detach().cpu()

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

def get_Jacobian(func, time_series, dims, img_shape, coord_tensor=None):
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