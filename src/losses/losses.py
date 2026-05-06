import time

import torch
import torch.nn.functional as F
import fastmri

from src.losses.sim_loss import get_sim_loss_fn
from src.losses.grad_calc import get_Laplacian, get_Jacobian
from src.losses.recon_reg import get_recon_regularizer

def calculate_losses(config, inputs, model_outputs, coord_tensor, losses):
    # abs_phi: [T, H*W, D]
    # rel_vel: [T or 1, H*W, D]
    moving = inputs[0]
    shape = [-1] + list(moving.shape[1:]) + [len(moving.shape) - 1]
    loss_sum = 0.
    moved_imgs = None
    moved_imgs_imspace = None
    for loss_name, loss_dict in losses.items():
        t1 = time.time()
        l, m, m_image = calc_single_loss(config, loss_name, inputs, model_outputs, coord_tensor, shape)
        t2 = time.time()

        moved_imgs = m if m is not None else moved_imgs
        moved_imgs_imspace = m_image if m_image is not None else moved_imgs_imspace
        mean = l.mean()
        loss_dict['mean'] = mean.detach().cpu()
        loss_dict['loss'] = l.detach()
        loss_dict['time'] += t2 - t1
        loss_sum  = loss_sum + loss_dict['lambda'] * mean

    return loss_sum, [losses, moved_imgs, moved_imgs_imspace]

def calc_single_loss(config, loss_name, inputs, model_outputs, coord_tensor, shape):
    if loss_name == 'sim':
        return similarity_loss(config, inputs, model_outputs)
    elif loss_name == 'imdiff':
        return im_space_l2_loss(config, inputs, model_outputs)
    elif loss_name == 'negJ':
        return negJ_loss(model_outputs, coord_tensor, shape)
    elif loss_name == 'grd':
        return vel_grad_loss(model_outputs, shape)
    elif loss_name == 'lap':
        return vel_lap_loss(model_outputs, shape)
    elif loss_name == 'pgr':
        return phi_grad_loss(model_outputs, coord_tensor, shape)
    elif loss_name == 'hyper_el':
        return compute_hyper_elastic_loss(model_outputs, coord_tensor, shape)
    elif loss_name == 'recon_reg':
        return compute_recon_reg_loss(inputs, config)

def similarity_loss(config, inputs, model_outputs):
    moving, moving_inr, fixed, forw, _ = inputs
    ST = model_outputs[2]
    loss_fn = get_sim_loss_fn(config, moving)
    
    expanded_shape = [fixed.shape[0]] + list(moving.shape)
    moving = moving.unsqueeze(0).expand(expanded_shape)
    if config.use_nreps:
        moved_im = ST.apply(moving_inr)
    else:
        moved_im = ST.apply(moving)
    moved = forw(moved_im)
    if config.dataset == 'heart_gt_ft_abs' or 'cmr' in config.dataset:
        # fixed = fastmri.complex_abs(fixed.movedim(1,-1))
        moved = fastmri.complex_abs_sq(moved.movedim(1,-1)).unsqueeze(1)
        moved = (moved + 1e-8).sqrt()
    loss = loss_fn(fixed, moved)
    loss = config.sim_lambda.view(-1, 1, 1, 1) * loss
    
    return loss, moved.detach().cpu(), moved_im.detach().cpu()

def im_space_l2_loss(config, inputs, model_outputs):
    moving, moving_inr, fixed, _, _ = inputs
    ST = model_outputs[2]
    loss_fn = torch.nn.MSELoss(reduction='none')
    
    expanded_shape = [fixed.shape[0]] + list(moving.shape)
    moving = moving.unsqueeze(0).expand(expanded_shape)
    if config.use_nreps:
        moved_im = ST.apply(moving_inr)
    else:
        moved_im = ST.apply(moving)
    if config.dataset == 'heart_gt_ft_abs' or 'cmr' in config.dataset:
        moving = fastmri.complex_abs_sq(moving.movedim(1,-1)).unsqueeze(1)
        moving = (moving + 1e-8).sqrt()
        # fixed = fastmri.complex_abs(fixed.movedim(1,-1))
        moved_im = fastmri.complex_abs_sq(moved_im.movedim(1,-1)).unsqueeze(1)
        moved_im = (moved_im + 1e-8).sqrt()
    loss = loss_fn(moving, moved_im)
    
    return loss, None, None

def negJ_loss(model_outputs, coord_tensor, shape):
    abs_phi = model_outputs[1]
    rel_phi = abs_phi - coord_tensor
    phi_J = get_Jacobian(rel_phi, shape)
    loss = neg_Jdet_loss(phi_J)
    return loss, None, None

def vel_grad_loss(model_outputs, shape):
    rel_vel = model_outputs[0]
    J = get_Jacobian(rel_vel, shape)
    loss = torch.linalg.vector_norm(J, dim=-1)
    return loss, None, None

def vel_lap_loss(model_outputs, shape):
    rel_vel = model_outputs[0]
    lap = get_Laplacian(rel_vel, shape)
    loss = torch.linalg.vector_norm(lap, dim=-1)
    return loss, None, None

def phi_grad_loss(model_outputs, coord_tensor, shape):
    abs_phi = model_outputs[1]
    phi_J = get_phi_Jacobian(abs_phi, coord_tensor, shape)
    loss = torch.linalg.vector_norm(phi_J, dim=-1)
    return loss, None, None

def compute_recon_reg_loss(inputs, config):
    moving = inputs[0]
    reg = get_recon_regularizer(config)
    loss = reg.g(moving.unsqueeze(1))
    return loss, None, None

def neg_Jdet_loss(J):
    Jdet = torch.det(J)
    neg_Jdet = -1.0 * Jdet
    neg_Jdet = F.relu(neg_Jdet) + 1
    out = torch.log(neg_Jdet)
    return out

def get_phi_Jacobian(abs_phi, coord_tensor, shape):
    rel_phi = abs_phi - coord_tensor
    phi_J = get_Jacobian(rel_phi, shape)
    return phi_J

# hyperelastic loss implementation based on IDIR implementation https://github.com/MIAGroupUT/IDIR/blob/main/objectives/regularizers.py
def compute_hyper_elastic_loss(
    model_outputs, coord_tensor, shape, alpha_l=1, alpha_a=1, alpha_v=1
):
    """Compute the hyper-elastic regularization loss."""
    rel_vel, abs_phi, _ = model_outputs

    grad_u = get_Jacobian(rel_vel, shape)
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