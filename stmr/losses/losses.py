import time

import fastmri
import torch

from stmr.data.fft_utils import fft2c_new
from stmr.losses.grad_calc import get_Jacobian, get_Laplacian
from stmr.losses.sim_loss import get_sim_loss_fn


def calculate_losses(config, inputs, model_outputs, coord_tensor, losses, recon_regularizer=None):
    # abs_phi: [T, H*W, D]
    # rel_vel: [T or 1, H*W, D]
    moving = inputs[0]
    shape = [-1] + list(moving.shape[2:]) + [len(moving.shape) - 2]
    loss_sum = 0.
    moved_imgs_imspace = None
    for loss_name, loss_dict in losses.items():
        t1 = time.time()
        l, m_image = calc_single_loss(config, loss_name, inputs, model_outputs, coord_tensor, shape, recon_regularizer)
        t2 = time.time()

        moved_imgs_imspace = m_image if m_image is not None else moved_imgs_imspace
        mean = l.mean()
        loss_dict['mean'] = mean.detach().cpu()
        loss_dict['loss'] = l.detach()
        loss_dict['time'] += t2 - t1
        loss_sum  = loss_sum + loss_dict['lambda'] * mean

    return loss_sum, [losses, moved_imgs_imspace]

def calc_single_loss(config, loss_name, inputs, model_outputs, coord_tensor, shape, recon_regularizer=None):
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
        return compute_recon_reg_loss(inputs, recon_regularizer)
    elif loss_name == 'mcdc':
        return motion_comp_dc_loss(config, inputs, model_outputs)

def similarity_loss(config, inputs, model_outputs):
    moving, moving_inr, fixed, forw = inputs
    loss_fn = get_sim_loss_fn(config, moving)

    recon_kspace = forw(moving)
    # if config.dataset == 'heart_gt_ft_abs' or 'cmr' in config.dataset:
    #     recon_kspace = fastmri.complex_abs_sq(recon_kspace.movedim(1,-1)).unsqueeze(1)
    #     recon_kspace = (recon_kspace + 1e-8).sqrt()
    loss = loss_fn(fixed, recon_kspace)
    
    return loss, None

def im_space_l2_loss(config, inputs, model_outputs):
    moving, moving_inr, fixed, _ = inputs
    ST = model_outputs[2]
    loss_fn = torch.nn.MSELoss(reduction='none')
    is_complex_img = config.dataset == 'heart_gt_ft_abs' or 'cmr' in config.dataset

    if config.use_nreps:
        if moving_inr is None:
            raise ValueError(
                "use_nreps=True requires a neural-representation image (moving_inr), but it "
                "is None for this dataset. Set use_nreps=False (the supported cmr mode)."
            )
        moved_im = ST.apply(moving_inr)
    elif is_complex_img and getattr(config, 'imdiff_warp_mag', False):
        # Warp the magnitude image, not the complex channels: the phase is temporally
        # incoherent in this data (see motion_comp_dc_loss), so bilinearly interpolating
        # rotating phasors destructively interferes and leaves a residual floor that no
        # motion field can explain.
        moving = fastmri.complex_abs_sq(moving.movedim(1,-1)).unsqueeze(1)
        moving = (moving + 1e-8).sqrt()
        moved_im = ST.apply(moving[:-1])
        return loss_fn(moving[1:], moved_im), moved_im.detach().cpu()
    else:
        moved_im = ST.apply(moving[:-1])
    if is_complex_img:
        moving = fastmri.complex_abs_sq(moving.movedim(1,-1)).unsqueeze(1)
        moving = (moving + 1e-8).sqrt()
        moved_im = fastmri.complex_abs_sq(moved_im.movedim(1,-1)).unsqueeze(1)
        moved_im = (moved_im + 1e-8).sqrt()
    loss = loss_fn(moving[1:], moved_im)

    return loss, moved_im.detach().cpu()

def negJ_loss(model_outputs: list, coord_tensor: torch.Tensor, shape: list) -> list[torch.Tensor, None]:
    abs_phi = model_outputs[1]
    # rel_phi = abs_phi - coord_tensor
    phi_J = get_Jacobian(abs_phi, shape)
    I = get_Jacobian(coord_tensor, shape)
    loss = neg_Jdet_loss(phi_J, I)
    return loss, None

def vel_grad_loss(model_outputs, shape):
    rel_vel = model_outputs[0]
    J = get_Jacobian(rel_vel, shape)
    loss = torch.linalg.vector_norm(J, dim=-1)
    return loss, None

def vel_lap_loss(model_outputs, shape):
    rel_vel = model_outputs[0]
    lap = get_Laplacian(rel_vel, shape)
    loss = torch.linalg.vector_norm(lap, dim=-1)
    return loss, None

def phi_grad_loss(model_outputs, coord_tensor, shape):
    abs_phi = model_outputs[1]
    phi_J = get_phi_Jacobian(abs_phi, coord_tensor, shape)
    loss = torch.linalg.vector_norm(phi_J, dim=-1)
    return loss, None

def motion_comp_dc_loss(config, inputs, model_outputs):
    # Motion-compensated data consistency: frame t+1's *measured* k-space rows must be
    # explained by the warped frame t. This turns the neighbors' measurements into
    # (soft) data constraints instead of an image-space prior, which is how the k-space
    # information actually transfers between frames.
    #
    # The phase is temporally incoherent in this data (frame-to-frame complex copy has
    # ~6% relative residual vs ~0.8% for magnitude), so only the magnitude is routed
    # through the motion model; the phase is taken from the target frame's own current
    # estimate: pred = |warp(I_t)| * I_{t+1} / |I_{t+1}|. Warping the magnitude (not the
    # complex channels) also avoids destructive interpolation of rotating phasors.
    moving, _, fixed, forw = inputs
    ST = model_outputs[2]
    moving_mag = fastmri.complex_abs_sq(moving.movedim(1,-1)).unsqueeze(1)
    moving_mag = (moving_mag + 1e-12).sqrt()
    moved_mag = ST.apply(moving_mag[:-1])
    pred = moved_mag * moving[1:] / moving_mag[1:]
    pred_kspace = fft2c_new(pred.movedim(1,-1)).movedim(-1,1) * forw.mask[1:]
    loss_fn = torch.nn.MSELoss(reduction='none')
    loss = loss_fn(pred_kspace, fixed[1:])
    return loss, None

def compute_recon_reg_loss(inputs, recon_regularizer):
    moving = inputs[0]
    moving_abs = fastmri.complex_abs_sq(moving.movedim(1,-1)).unsqueeze(1)
    moving_abs = (moving_abs + 1e-8).sqrt()
    loss = recon_regularizer.g(moving_abs)
    return loss, None

def neg_Jdet_loss(J, I):
    # I = torch.eye(J.shape[-1], device=J.device)
    # Jdet = torch.det(J)
    # print(Jdet.mean(), J.mean(dim=(0,1,2)))
    # out = (Jdet - 1) ** 2
    out = torch.linalg.matrix_norm(J - I, dim=(-2, -1), ord='fro')
    # Jdet = torch.det(J)
    # neg_Jdet = -1.0 * Jdet
    # neg_Jdet = F.relu(neg_Jdet) + 1
    # out = torch.log(neg_Jdet)
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