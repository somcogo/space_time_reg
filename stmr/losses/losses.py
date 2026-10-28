import time
from argparse import Namespace

import fastmri
import torch

from stmr.data.data_utils import complex_abs, get_dataset_capabilities
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
    if config.joint_objective not in ('default', 'init'):
        raise ValueError(f"Unknown joint_objective {config.joint_objective!r}; use 'default' or 'init'")
    for loss_name, loss_dict in losses.items():
        t1 = time.time()
        l, m_image = calc_single_loss(config, loss_name, inputs, model_outputs, coord_tensor, shape, recon_regularizer)
        t2 = time.time()

        moved_imgs_imspace = m_image if m_image is not None else moved_imgs_imspace
        # joint_objective='init' reproduces the init's energy, which sums (not averages) over
        # frames and k-space entries.
        init_term = config.joint_objective == 'init' and loss_name in ('sim', 'recon_reg')
        mean = l.sum() if init_term else l.mean()
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
        'grad_phi': lambda: grad_phi_loss(config, model_outputs, coord_tensor, shape),
        'detJ': lambda: detJ_loss(model_outputs, coord_tensor, shape),
        'logdetJ': lambda: log_detJ_loss(model_outputs, coord_tensor, shape),
        'grd': lambda: vel_grad_loss(model_outputs, shape),
        'lap': lambda: vel_lap_loss(model_outputs, shape),
        'recon_reg': lambda: compute_recon_reg_loss(config, inputs, recon_regularizer),
        'mcdc': lambda: motion_comp_dc_loss(config, inputs, model_outputs),
    }
    if loss_name not in dispatch:
        raise ValueError(f"Unknown loss {loss_name!r}; registered: {sorted(dispatch)}")
    return dispatch[loss_name]()

def similarity_loss(config, inputs, model_outputs):
    moving, forw = inputs.moving, inputs.forward
    loss_fn = get_sim_loss_fn(config, moving)

    sim_domain = getattr(config, 'sim_domain', 'fourier')
    if getattr(config, 'joint_objective', 'default') == 'init':
        # The init's data term (recon_init.complex_l2), kept per entry for the sim-energy
        # visuals; calculate_losses sums it. forw is the masked FT and fixed is zero-filled.
        loss = 0.5 * (forw(moving) - inputs.fixed) ** 2
    elif sim_domain == 'image':
        # Image-space data consistency: compare the recon directly to the IFFT of the
        # measured k-space. On fully-sampled data this is identical (value AND gradient) to
        # the Fourier-space MSE below, because the FFT is unitary (Parseval). Running one
        # stage each way is a check that the FFT/IFFT data-consistency path is correct.
        target = ifft2c_new(inputs.fixed.movedim(1, -1)).movedim(-1, 1)
        loss = loss_fn(target, moving)
    elif sim_domain == 'image_mag':
        # Phase-insensitive image-space DC: compare the |magnitude| images of the recon and
        # the IFFT of the measured k-space. eps-guarded sqrt (not complex_abs) for safe grads.
        target = ifft2c_new(inputs.fixed.movedim(1, -1)).movedim(-1, 1)
        tm = (target.pow(2).sum(1, keepdim=True) + 1e-12).sqrt()   # |target| [T,1,H,W]
        mm = (moving.pow(2).sum(1, keepdim=True) + 1e-12).sqrt()   # |recon|  [T,1,H,W]
        loss = loss_fn(tm, mm)
    elif sim_domain == 'fourier_mag':
        # Phase-insensitive Fourier DC: compare the |magnitude| of the (masked) k-space on
        # both sides, discarding k-space phase. Used by the template-warp experiment so the
        # flow need not match the temporally-incoherent image phase. eps-guarded sqrt (not
        # complex_abs) so the masked-out zeros don't produce NaN gradients from d(sqrt)/d0.
        recon_kspace = forw(moving)
        rm = (recon_kspace.pow(2).sum(1, keepdim=True) + 1e-12).sqrt()   # |k-space| [T,1,H,W]
        fm = (inputs.fixed.pow(2).sum(1, keepdim=True) + 1e-12).sqrt()   # |target|  [T,1,H,W]
        loss = loss_fn(fm, rm)
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

def grad_phi_loss(config: Namespace, model_outputs, coord_tensor: torch.Tensor, shape: list):
    # ||Dphi - Id||_fro: penalizes the deformation's Jacobian for departing from the
    # identity, i.e. penalizes the gradient of the displacement field phi - x. Measured in
    # pixel units (see _pixel_phi_Jacobian) so the penalty is isotropic on non-square images.
    J = _pixel_phi_Jacobian(config, model_outputs, coord_tensor, shape)
    eye = torch.eye(J.shape[-1], device=J.device, dtype=J.dtype)
    loss = grad_phi_norm(J, eye)
    return loss, None

def _pixel_phi_Jacobian(config: Namespace, model_outputs, coord_tensor: torch.Tensor, shape: list):
    # phi and the grid live in [-1, 1] on every axis, so on a non-square image one normalized
    # unit spans a different number of pixels per axis ((N - 1) / 2). Differencing them as-is
    # weights each displacement component by 1 / that span (2.5x more for rows than columns
    # on 204x512), and right-multiplying by inv(I) alone (_relative_phi_Jacobian) only fixes
    # stretches -- shears stay anisotropic. Rescaling both to pixels first, then normalizing
    # by the identity's own finite-difference Jacobian, gives the Jacobian in pixel units:
    # rotation-invariant, and exactly Id for phi = identity, including at the boundary.
    span = torch.tensor([(n - 1) / 2 for n in shape[1:-1]], device=coord_tensor.device,
                        dtype=coord_tensor.dtype) * config.voxel_spacing
    phi_J = get_Jacobian(model_outputs.abs_phi * span, shape)
    I = get_Jacobian(coord_tensor * span, shape)
    return phi_J @ torch.linalg.inv(I)

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

def compute_recon_reg_loss(config, inputs, recon_regularizer):
    moving = inputs.moving
    T = moving.shape[0]
    moving = moving.movedim(2, 1).flatten(0, 1)
    if config.init_reg_abs:
        loss = recon_regularizer.g(complex_abs(moving)).reshape(-1)
    else:
        loss = recon_regularizer.g(moving.flatten(0, 1).unsqueeze(1)).reshape(moving.shape[0], -1).sum(1)
    loss = loss.view(T, -1).sum(1)
    return loss, None

def grad_phi_norm(J, I):
    return torch.linalg.matrix_norm(J - I, dim=(-2, -1), ord='fro')

def get_phi_Jacobian(abs_phi, coord_tensor, shape):
    rel_phi = abs_phi - coord_tensor
    phi_J = get_Jacobian(rel_phi, shape)
    return phi_J