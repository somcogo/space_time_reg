import time

import torch
import torch.nn.functional as F

from src.losses.sim_loss import get_sim_loss_fn
from src.losses.grad_calc import fin_diff_Jacobian, get_Laplacian, get_autograd_Jacobian
from src.losses.recon_reg import get_recon_regularizer

def calculate_losses(config, moving, moving_inr, fixed, forw, abs_phi, rel_vel, func, time_series, coord_tensor, losses, downsample, ST):
    # abs_phi: [T, H*W, D]
    # rel_vel: [T or 1, H*W, D]
    moving = moving.to(config.device)
    moving_inr = moving_inr.to(config.device) if moving_inr is not None else None
    fixed = fixed.to(config.device)
    forw = forw.to(config.device)

    shape = [-1] + list(moving.shape[1:]) + [len(moving.shape) - 1]
    loss_sum = 0.
    moved_imgs = None
    moved_imgs_imspace = None
    for loss_type, loss_dict in losses.items():
        t1 = time.time()
        l, m, m_image = calc_single_loss(loss_type, moving, moving_inr, fixed, forw, abs_phi, rel_vel, func, coord_tensor, shape, config, time_series, downsample, ST)
        t2 = time.time()

        moved_imgs = m if m is not None else moved_imgs
        moved_imgs_imspace = m_image if m_image is not None else moved_imgs_imspace
        mean = l.mean()
        loss_dict['mean'] = mean.detach().cpu()
        loss_dict['loss'] = l.detach()
        loss_dict['time'] += t2 - t1
        loss_sum  = loss_sum + loss_dict['lambda'] * mean

    return loss_sum, losses, moved_imgs, moved_imgs_imspace

def calc_single_loss(loss_name, moving, moving_inr, fixed, forw, abs_phi, rel_vel, func, coord_tensor, shape, config, time_series, downsample, ST):
    if loss_name == 'sim':
        return similarity_loss(moving, moving_inr, fixed, forw, config, downsample, ST)
    elif loss_name == 'negJ':
        return negJ_loss(abs_phi, coord_tensor, shape, downsample)
    elif loss_name == 'grd':
        return vel_grad_loss(rel_vel, func, coord_tensor, shape, config, time_series, downsample)
    elif loss_name == 'lap':
        return vel_lap_loss(rel_vel, func, coord_tensor, shape, config, time_series, downsample)
    elif loss_name == 'pgr':
        return phi_grad_loss(abs_phi, coord_tensor, shape, downsample)
    elif loss_name == 'hyper_el':
        return compute_hyper_elastic_loss(abs_phi, rel_vel, func, coord_tensor, shape, config, time_series)
    elif loss_name == 'recon_reg':
        return compute_recon_reg_loss(moving, config)

def similarity_loss(moving, moving_inr, fixed, forw, config, downsample, ST):
    loss_fn = get_sim_loss_fn(config, moving)
    
    expanded_shape = [fixed.shape[0]] + list(moving.shape)
    moving = moving.unsqueeze(0).expand(expanded_shape)
    if config.use_nreps:
        moved_im = ST.apply(moving_inr)
    else:
        moved_im = ST.apply(moving)
    moved = forw(moved_im)
    loss = loss_fn(fixed, moved)
    
    return loss * (downsample ** (3 - len(moving.shape[2:]))), moved.detach().cpu(), moved_im.detach().cpu()

def negJ_loss(abs_phi, coord_tensor, shape, downsample):
    rel_phi = abs_phi - coord_tensor
    phi_reshaped = rel_phi.reshape(shape)
    phi_J = fin_diff_Jacobian(phi_reshaped)
    loss = neg_Jdet_loss(phi_J) / downsample**3
    return loss, None, None

def vel_grad_loss(rel_vel, func, coord_tensor, shape, config, time_series, downsample):
    J = get_Jacobian(rel_vel, func, coord_tensor, shape, config, time_series)
    loss = torch.linalg.vector_norm(J, dim=-1) / downsample
    return loss, None, None

def vel_lap_loss(rel_vel, func, coord_tensor, shape, config, time_series, downsample):
    lap = get_Laplacian(rel_vel, func, coord_tensor, shape, config, time_series)
    loss = torch.linalg.vector_norm(lap, dim=-1) / downsample
    return loss, None, None

def phi_grad_loss(abs_phi, coord_tensor, shape, downsample):
    phi_J = get_phi_Jacobian(abs_phi, coord_tensor, shape)
    loss = torch.linalg.vector_norm(phi_J, dim=-1) / downsample
    return loss, None, None

def compute_recon_reg_loss(moving, config):
    reg = get_recon_regularizer(config)
    loss = reg.g(moving.unsqueeze(1))
    return loss, None, None

def neg_Jdet_loss(J):
    Jdet = torch.det(J)
    neg_Jdet = -1.0 * Jdet
    neg_Jdet = F.relu(neg_Jdet) + 1
    out = torch.log(neg_Jdet)
    return out

def get_Jacobian(rel_vel, func, coord_tensor, shape, config, time_series):
    if config.fin_diff_grad:
        vel_reshaped = rel_vel.reshape(shape)
        J = fin_diff_Jacobian(vel_reshaped)
    elif config.autograd_grid:
        J = get_autograd_Jacobian(func, time_series, dims=shape[-1], coord_tensor=coord_tensor)
    else:
        J = get_autograd_Jacobian(func, time_series, dims=shape[-1], coord_tensor=None)
    return J

def get_phi_Jacobian(abs_phi, coord_tensor, shape):
    rel_phi = abs_phi - coord_tensor
    phi_reshaped = rel_phi.reshape(shape)
    phi_J = fin_diff_Jacobian(phi_reshaped)
    return phi_J

# hyperelastic loss implementation based on IDIR implementation https://github.com/MIAGroupUT/IDIR/blob/main/objectives/regularizers.py
def compute_hyper_elastic_loss(
    abs_phi, rel_vel, func, coord_tensor, shape, config, time_series, alpha_l=1, alpha_a=1, alpha_v=1
):
    """Compute the hyper-elastic regularization loss."""

    grad_u = get_Jacobian(rel_vel, func, coord_tensor, shape, config, time_series)
    grad_y = get_phi_Jacobian(abs_phi, coord_tensor, shape)
    # get_phi_Jacobian produces the grad of the relative displacement, want the grad of the absolut displacement
    for i in range(grad_y.shape[-1]):
        grad_y[..., i, i] = grad_y[..., i, i] + torch.ones_like(grad_y[..., i, i])

    # Compute length loss
    length_loss = torch.linalg.norm(grad_u, dim=(1, 2))
    length_loss = torch.pow(length_loss, 2)
    length_loss = torch.sum(length_loss)
    length_loss = 0.5 * alpha_l * length_loss

    # Compute cofactor matrices for the area loss
    cofactors = torch.zeros(*grad_y.shape[:-2], 3, 3)

    # Compute elements of cofactor matrices one by one (Ugliest solution ever?)
    cofactors[..., 0, 0] = torch.det(grad_y[..., 1:, 1:])
    cofactors[..., 0, 1] = torch.det(grad_y[..., 1:, 0::2])
    cofactors[..., 0, 2] = torch.det(grad_y[..., 1:, :2])
    cofactors[..., 1, 0] = torch.det(grad_y[..., 0::2, 1:])
    cofactors[..., 1, 1] = torch.det(grad_y[..., 0::2, 0::2])
    cofactors[..., 1, 2] = torch.det(grad_y[..., 0::2, :2])
    cofactors[..., 2, 0] = torch.det(grad_y[..., :2, 1:])
    cofactors[..., 2, 1] = torch.det(grad_y[..., :2, 0::2])
    cofactors[..., 2, 2] = torch.det(grad_y[..., :2, :2])

    # Compute area loss
    area_loss = torch.pow(cofactors, 2)
    area_loss = torch.sum(area_loss, dim=1)
    area_loss = area_loss - 1
    area_loss = torch.maximum(area_loss, torch.zeros_like(area_loss))
    area_loss = torch.pow(area_loss, 2)
    area_loss = torch.sum(area_loss)  # sum over dimension 1 and then 0
    area_loss = alpha_a * area_loss

    # Compute volume loss
    volume_loss = torch.det(grad_y)
    volume_loss = torch.mul(torch.pow(volume_loss - 1, 4), torch.pow(volume_loss, -2))
    volume_loss = torch.sum(volume_loss)
    volume_loss = alpha_v * volume_loss

    # Compute total loss
    loss = length_loss + area_loss + volume_loss

    return loss, None