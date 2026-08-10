import time

import fastmri
import torch

from stmr.data.data_utils import get_dataset_capabilities
from stmr.data.fft_utils import fft2c_new, ifft2c_new
from stmr.losses.grad_calc import get_Jacobian, get_Laplacian
from stmr.losses.sim_loss import get_sim_loss_fn
from stmr.state import Inputs, LossOutputs, ModelOutputs


def calculate_losses(config, inputs: Inputs, model_outputs: ModelOutputs, coord_tensor,
                     losses, recon_regularizer=None):
    # abs_phi: [T, H*W, D]
    # rel_vel: [T or 1, H*W, D]
    moving = inputs.moving
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

    return loss_sum, LossOutputs(losses=losses, moved_imspace=moved_imgs_imspace)

def calc_single_loss(config, loss_name, inputs, model_outputs, coord_tensor, shape, recon_regularizer=None):
    # Each entry adapts a single-loss function to the uniform call context. Replaces the
    # previous if/elif ladder; new losses register a lambda here.
    dispatch = {
        'sim': lambda: similarity_loss(config, inputs, model_outputs),
        'imdiff': lambda: im_space_l2_loss(config, inputs, model_outputs),
        'grad_phi': lambda: grad_phi_loss(model_outputs, coord_tensor, shape),
        'detJ': lambda: detJ_loss(model_outputs, coord_tensor, shape),
        'logdetJ': lambda: log_detJ_loss(model_outputs, coord_tensor, shape),
        'grd': lambda: vel_grad_loss(model_outputs, shape),
        'lap': lambda: vel_lap_loss(model_outputs, shape),
        'hyper_el': lambda: compute_hyper_elastic_loss(model_outputs, coord_tensor, shape),
        'recon_reg': lambda: compute_recon_reg_loss(inputs, recon_regularizer),
        'mcdc': lambda: motion_comp_dc_loss(config, inputs, model_outputs),
    }
    if loss_name not in dispatch:
        raise ValueError(f"Unknown loss {loss_name!r}; registered: {sorted(dispatch)}")
    return dispatch[loss_name]()

def similarity_loss(config, inputs, model_outputs):
    moving, forw = inputs.moving, inputs.forward
    loss_fn = get_sim_loss_fn(config, moving)

    if getattr(config, 'sim_domain', 'fourier') == 'image':
        # Image-space data consistency: compare the recon directly to the IFFT of the
        # measured k-space. On fully-sampled data this is identical (value AND gradient) to
        # the Fourier-space MSE below, because the FFT is unitary (Parseval). Running one
        # stage each way is a check that the FFT/IFFT data-consistency path is correct.
        target = ifft2c_new(inputs.fixed.movedim(1, -1)).movedim(-1, 1)
        loss = loss_fn(target, moving)
    else:
        recon_kspace = forw(moving)
        loss = loss_fn(inputs.fixed, recon_kspace)

    return loss, None

def im_space_l2_loss(config, inputs, model_outputs):
    moving, moving_inr = inputs.moving, inputs.moving_inr
    ST = model_outputs.transformer
    loss_fn = torch.nn.MSELoss(reduction='none')
    is_complex_img = (config.dataset == 'heart_gt_ft_abs'
                      or get_dataset_capabilities(config).is_complex_img)

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

def grad_phi_loss(model_outputs, coord_tensor: torch.Tensor, shape: list):
    # ||Dphi - Id||_fro: penalizes the deformation's Jacobian for departing from the
    # identity, i.e. penalizes the gradient of the displacement field phi - x. Finite
    # differencing is linear, so phi_J - I here is exactly the Jacobian of (phi - x); unlike
    # the det-based losses below, no inv(I) normalization is needed for this additive form.
    abs_phi = model_outputs.abs_phi
    phi_J = get_Jacobian(abs_phi, shape)
    I = get_Jacobian(coord_tensor, shape)
    loss = grad_phi_norm(phi_J, I)
    return loss, None

def _relative_phi_Jacobian(model_outputs, coord_tensor, shape):
    # get_Jacobian is a raw finite-difference operator in grid-index space: even the
    # identity map's own Jacobian I = get_Jacobian(coord_tensor, shape) isn't torch.eye
    # (grid spacing != 1, further perturbed near the boundary by replicate padding).
    # Normalizing by I -- as grad_phi_loss's J - I already does additively -- makes phi=identity
    # map to exactly J_rel=Identity (det=1) everywhere, including at the boundary.
    phi_J = get_Jacobian(model_outputs.abs_phi, shape)
    I = get_Jacobian(coord_tensor, shape)
    return phi_J @ torch.linalg.inv(I)

def detJ_loss(model_outputs, coord_tensor, shape):
    # (det(D phi) - 1)^2: penalizes local volume change of the deformation away from 1
    # (volume-preserving). Squared so over- and under-expansion both push the loss up and
    # the term stays non-negative (an unsquared det - 1 would be minimized by driving det
    # to -inf).
    detJ = torch.linalg.det(_relative_phi_Jacobian(model_outputs, coord_tensor, shape))
    loss = (detJ - 1) ** 2
    return loss, None

def log_detJ_loss(model_outputs, coord_tensor, shape, eps=1e-6):
    # log(det(D phi))^2: symmetric under det -> 1/det (unlike (det-1)^2), so compression and
    # expansion by the same factor are penalized equally. Squared for the same reason as
    # detJ_loss above. det is clamped away from 0 to avoid log(<=0) under a fold.
    detJ = torch.linalg.det(_relative_phi_Jacobian(model_outputs, coord_tensor, shape))
    loss = torch.log(detJ.clamp_min(eps)) ** 2
    return loss, None

def vel_grad_loss(model_outputs, shape):
    rel_vel = model_outputs.rel_vel
    J = get_Jacobian(rel_vel, shape)
    loss = torch.linalg.vector_norm(J, dim=-1)
    return loss, None

def vel_lap_loss(model_outputs, shape):
    rel_vel = model_outputs.rel_vel
    lap = get_Laplacian(rel_vel, shape)
    loss = torch.linalg.vector_norm(lap, dim=-1)
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
    moving, fixed, forw = inputs.moving, inputs.fixed, inputs.forward
    ST = model_outputs.transformer
    moving_mag = fastmri.complex_abs_sq(moving.movedim(1,-1)).unsqueeze(1)
    moving_mag = (moving_mag + 1e-12).sqrt()
    moved_mag = ST.apply(moving_mag[:-1])
    pred = moved_mag * moving[1:] / moving_mag[1:]
    pred_kspace = fft2c_new(pred.movedim(1,-1)).movedim(-1,1) * forw.mask[1:]
    loss_fn = torch.nn.MSELoss(reduction='none')
    loss = loss_fn(pred_kspace, fixed[1:])
    return loss, None

def compute_recon_reg_loss(inputs, recon_regularizer):
    moving = inputs.moving
    moving_abs = fastmri.complex_abs_sq(moving.movedim(1,-1)).unsqueeze(1)
    moving_abs = (moving_abs + 1e-8).sqrt()
    loss = recon_regularizer.g(moving_abs)
    return loss, None

def grad_phi_norm(J, I):
    return torch.linalg.matrix_norm(J - I, dim=(-2, -1), ord='fro')

def get_phi_Jacobian(abs_phi, coord_tensor, shape):
    rel_phi = abs_phi - coord_tensor
    phi_J = get_Jacobian(rel_phi, shape)
    return phi_J

# hyperelastic loss implementation based on IDIR implementation https://github.com/MIAGroupUT/IDIR/blob/main/objectives/regularizers.py
def compute_hyper_elastic_loss(
    model_outputs, coord_tensor, shape, alpha_l=1, alpha_a=1, alpha_v=1
):
    """Compute the hyper-elastic regularization loss."""
    rel_vel, abs_phi = model_outputs.rel_vel, model_outputs.abs_phi

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