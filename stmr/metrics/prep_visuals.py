import io
import math
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from fastmri import complex_abs
from flow_vis import flow_to_color
from matplotlib.axes import Axes
from PIL import Image
from torchvision.utils import make_grid

# TODO: refactor all the make_grid and ndarray -> torch.uint8 lines

def add_loss_specific_imgs(imgs_to_save: dict, losses: dict, reduce_dim: bool) -> dict:
    for loss_type, loss_dict in losses.items():
        if loss_type == 'grad_phi':
            loss = loss_dict['loss'][..., 0] if reduce_dim else loss_dict['loss']
            grad_phi = prep_detJ_vis(loss)
            imgs_to_save['vel_J_det/grad_phi'] = grad_phi
        elif loss_type == 'detJ':
            loss = loss_dict['loss'][..., 0] if reduce_dim else loss_dict['loss']
            imgs_to_save['vel_J_det/detJ'] = prep_detJ_vis(loss)
        elif loss_type == 'logdetJ':
            loss = loss_dict['loss'][..., 0] if reduce_dim else loss_dict['loss']
            imgs_to_save['vel_J_det/logdetJ'] = prep_detJ_vis(loss)
        elif loss_type == 'grd':
            loss = loss_dict['loss'][..., 0, :] if reduce_dim else loss_dict['loss']
            grad_norm = prep_vel_grad_vis(loss)
            imgs_to_save['vel_grad_norm/grad_norm'] = grad_norm
        elif loss_type == 'lap':
            loss = loss_dict['loss'][..., 0] if reduce_dim else loss_dict['loss']
            lap_norm = prep_vel_lap_vis(loss)
            imgs_to_save['laplacian/laplacian_norm'] = lap_norm

    return imgs_to_save

def prep_moved_img_vis(moved_im: torch.Tensor, moving: torch.Tensor, config: Namespace) -> list[torch.Tensor]:
    moving = moving.detach().cpu()[1:]
    if config.dataset == 'heart_gt_no_ft_no_abs':
        moved_im = moved_im.squeeze(1)
        moving = complex_abs(moving.movedim(1, -1))
    elif config.dataset == 'heart_gt_ft_abs' or 'cmr' in config.dataset:
        # moved_im = complex_abs(moved_im.movedim(1,-1))
        moving = complex_abs(moving.movedim(1, -1))
    elif config.dataset == 'toy_square':
        moved_im = complex_abs(moved_im.movedim(1,-1))
        moving = complex_abs(moving.movedim(1, -1))
    else:
        # Real single-channel (e.g. CBCT): drop the channel dim, matching moved_im.squeeze(1)
        # below -- this branch previously did squeeze(0), which is a no-op whenever there's
        # more than one frame and left a stray channel dim causing a stack-shape mismatch.
        moving = moving.squeeze(1)
    moved_im = moved_im.squeeze(1)

    reg_last = make_grid([torch.stack([moving[-1], moved_im[-1], moved_im[-1]])], nrow=2, normalize=True)
    reg_all = make_grid([torch.stack([im, m_im, m_im]) for im, m_im in zip(moving, moved_im, strict=False)], nrow=6, normalize=True)
    reg_imspace = make_grid(moved_im.unsqueeze(1), nrow=6, normalize=True)
    moving_im = make_grid(moving.unsqueeze(1), nrow=6, normalize=True)
    # moving = (moving - moving.min()) / (moving.max() - moving.min())

    reg_last = (reg_last*255).to(torch.uint8).permute(1, 2, 0)
    reg_all = (reg_all*255).to(torch.uint8).permute(1, 2, 0)
    reg_imspace = (reg_imspace*255).to(torch.uint8).permute(1, 2, 0)
    moving_im = (moving_im*255).to(torch.uint8).permute(1, 2, 0)
    # moving_im = (torch.stack([moving]*3, dim=1)*255).to(torch.uint8)

    return reg_last, reg_all, reg_imspace, moving_im

def prep_image_space_comp(moving: torch.Tensor, gt_im: torch.Tensor) -> torch.Tensor:
    fixed_im = complex_abs(gt_im.movedim(1, -1)).detach().cpu()
    moving = complex_abs(moving.movedim(1, -1)).detach().cpu()
    im_space_comp = make_grid([torch.stack([im, m_im, m_im]) for im, m_im in zip(fixed_im, moving, strict=False)], nrow=5, normalize=True)
    im_space_comp = (im_space_comp*255).to(torch.uint8).permute(1, 2, 0)
    return im_space_comp

def prep_init_recon(init_recon: torch.Tensor, is_complex: bool = True):
    init_recon = init_recon.detach().cpu()
    init_im = complex_abs(init_recon.movedim(1, -1)) if is_complex else init_recon.squeeze(1)
    init_grid = make_grid(init_im.unsqueeze(1), nrow=5, normalize=True)
    init_grid = (init_grid*255).to(torch.uint8).permute(1, 2, 0)
    return init_grid

def prep_vel_vis(rel_vel: torch.Tensor) -> list[torch.Tensor]:
    rel_act_velocity_color = []
    for time in range(rel_vel.shape[0]):
        rel_act_velocity_color.append(torch.from_numpy(flow_to_color(rel_vel[time].numpy(), convert_to_bgr=False)).permute(2, 0, 1))
    vel_color = make_grid(rel_act_velocity_color, nrow=5)
    vel_norm = make_grid([torch.linalg.norm(vel, ord=2, dim=-1).unsqueeze(0) for vel in rel_vel], nrow=5, normalize=True)
    vel_color = vel_color.permute(1, 2, 0)
    vel_norm = (vel_norm*255).to(torch.uint8).permute(1, 2, 0)

    return vel_color, vel_norm

def prep_flow_vis(rel_phi: torch.Tensor) -> torch.Tensor:
    rel_phi = rel_phi.numpy()
    rel_flow_colors = []
    for time in range(rel_phi.shape[0]):
        rel_flow_colors.append(torch.from_numpy(flow_to_color(rel_phi[time], convert_to_bgr=False)).permute(2, 0, 1))
    flow_col = make_grid(rel_flow_colors, nrow=5)
    flow_col = flow_col.permute(1, 2, 0)

    return flow_col

def prep_sim_meas_vis(sim_meas: torch.Tensor) -> list[torch.Tensor]:
    sim_meas = sim_meas.sum(dim=1, keepdim=True)
    sim_grid = make_grid([im for im in sim_meas], nrow=5, normalize=True)
    sim_grid = (sim_grid*255).cpu().to(torch.uint8).permute(1, 2, 0)

    log_sim_meas = torch.log(sim_meas)
    log_sim_grid = make_grid([im for im in log_sim_meas], nrow=5, normalize=True)
    log_sim_grid = (log_sim_grid*255).cpu().to(torch.uint8).permute(1, 2, 0)
    return sim_grid, log_sim_grid

def prep_imdiff_energy_vis(im_diff_loss: torch.Tensor) -> torch.Tensor:
    im_diff_loss = im_diff_loss.sum(dim=1, keepdim=True)
    diff_grid = make_grid([im for im in im_diff_loss], nrow=5, normalize=True)
    diff_grid = (diff_grid*255).cpu().to(torch.uint8).permute(1, 2, 0)
    return diff_grid

def prep_seg_vis(segs: torch.Tensor, pred_segs: torch.Tensor) -> list[torch.Tensor]:
    gt_mask = (segs > 0).float()
    pred_mask = (pred_segs > 0).float()
    seg_comb_last = make_grid([torch.stack([gt_mask[-1], torch.zeros_like(gt_mask[-1]), pred_mask[-1]])], nrow=2, normalize=True)
    seg_comb_all = make_grid([torch.stack([s, torch.zeros_like(s), pr]) for s, pr in zip(gt_mask, pred_mask, strict=False)], nrow=5, normalize=True)
    seg_comb_last = (seg_comb_last*255).to(torch.uint8).permute(1, 2, 0)
    seg_comb_all = (seg_comb_all*255).to(torch.uint8).permute(1, 2, 0)
    return seg_comb_all, seg_comb_last

def prep_detJ_vis(loss: torch.Tensor) -> torch.Tensor:
    J_det_grid = make_grid(loss.unsqueeze(1), nrow=5, normalize=True, pad_value=1)
    J_det_grid =(J_det_grid*255).cpu().to(torch.uint8).permute(1, 2, 0)
    return J_det_grid

def prep_vel_grad_vis(vel_grad: torch.Tensor) -> torch.Tensor:
    grad_norm = make_grid([torch.linalg.norm(im, dim=-1).unsqueeze(0) for im in vel_grad], nrow=5, normalize=True)    
    grad_norm = (grad_norm*255).cpu().to(torch.uint8).permute(1, 2, 0)
    return grad_norm

def prep_vel_lap_vis(lap: torch.Tensor) -> torch.Tensor:
    lap_grid = make_grid(lap.unsqueeze(1), nrow=5, normalize=True)
    lap_grid = (lap_grid*255).cpu().to(torch.uint8).permute(1, 2, 0)
    return lap_grid

def draw_deformed_grid(phi: torch.Tensor, ax: Axes) -> tuple[Axes]:
    # fig, ax = plt.subplots()
    for i in range(0, phi.shape[0], math.ceil(phi.shape[0]/32)):
        ax.plot(phi[i, :, 1], -phi[i, :, 0], 'r-', linewidth=0.5)
    for i in range(0, phi.shape[1], math.ceil(phi.shape[1]/32)):
        ax.plot(phi[:, i, 1], -phi[:, i, 0], 'r-', linewidth=0.5)
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_frame_on(False)
    ax.tick_params(tick1On=False)
    # ax.set_aspect('equal')
    # return ax


def prep_grid_def_vis(last_phi: torch.Tensor) -> torch.Tensor:
    fig, ax = plt.subplots()
    draw_deformed_grid(last_phi, ax)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    np_image = np.array(image).transpose(2, 0, 1)
    tens_image = torch.from_numpy(np_image).permute(1, 2, 0)
    plt.close(fig)
    return tens_image