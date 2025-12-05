import math
import io

import torch
from torchvision.utils import make_grid
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from flow_vis import flow_to_color
from fastmri import complex_abs, ifft2c

def add_loss_specific_imgs(imgs_to_save, losses, config, nr_time_frames, img_shape, reduce_dim):
    for loss_type, loss_dict in losses.items():
        if loss_type == 'negJ':
            loss = loss_dict['loss'][..., 0] if reduce_dim else loss_dict['loss']
            negJ = prep_detJ_vis(loss, config, nr_time_frames, img_shape)
            imgs_to_save['vel_J_det/negJ'] = negJ
        elif loss_type == 'grd':
            loss = loss_dict['loss'][..., 0, :] if reduce_dim else loss_dict['loss']
            grad_norm = prep_vel_grad_vis(loss, config, nr_time_frames, img_shape)
            imgs_to_save['vel_grad_norm/grad_norm'] = grad_norm
        elif loss_type == 'lap':
            loss = loss_dict['loss'][..., 0] if reduce_dim else loss_dict['loss']
            lap_norm = prep_vel_lap_vis(loss, config, img_shape)
            imgs_to_save['laplacian/laplacian_norm'] = lap_norm
        elif loss_type == 'pgr':
            loss = loss_dict['loss'][..., 0, :] if reduce_dim else loss_dict['loss']
            phi_grad_norm = prep_phi_grad_vis(loss)
            imgs_to_save['phi_grad_norm/phi_grad_norm'] = phi_grad_norm

    return imgs_to_save

def prep_moved_img_vis(imgs, moved, moved_im, moving, config):
    imgs = imgs.detach().cpu()

    if 'cmr' in config.dataset:
        imgs = complex_abs(imgs.movedim(1, -1))
        moved = complex_abs(moved.movedim(1, -1))
        moved_im = complex_abs(moved_im.movedim(1, -1))
        moving = complex_abs(moving.detach().cpu().movedim(0, -1))
    else:
        imgs = imgs.squeeze(1)
        moved = moved.squeeze(1)
        moved_im = moved_im.squeeze(1)
        moving = moving.squeeze(1)
    reg_last = make_grid([torch.stack([imgs[-1], moved[-1], moved[-1]])], nrow=2, normalize=True)
    reg_all = make_grid([torch.stack([im, m_im, m_im]) for im, m_im in zip(imgs, moved)], nrow=5, normalize=True)
    reg_imspace = make_grid(moved_im.unsqueeze(1), nrow=5, normalize=True)
    moving = (moving - moving.min()) / (moving.max() - moving.min())


    reg_last = (reg_last*255).to(torch.uint8).permute(1, 2, 0)
    reg_all = (reg_all*255).to(torch.uint8).permute(1, 2, 0)
    reg_imspace = (reg_imspace*255).to(torch.uint8).permute(1, 2, 0)
    moving_im = (torch.stack([moving]*3, dim=2)*255).to(torch.uint8)
    return reg_last, reg_all, reg_imspace, moving_im

def prep_image_space_comp(moved_im, gt_im):
    fixed_im = complex_abs(gt_im.movedim(1, -1))
    moved_im = complex_abs(moved_im.movedim(1, -1))
    im_space_comp = make_grid([torch.stack([im, m_im, m_im]) for im, m_im in zip(fixed_im, moved_im)], nrow=5, normalize=True)
    im_space_comp = (im_space_comp*255).to(torch.uint8).permute(1, 2, 0)
    return im_space_comp


def prep_vel_vis(rel_vel):
    rel_act_velocity_color = []
    for time in range(rel_vel.shape[0]):
        rel_act_velocity_color.append(torch.from_numpy(flow_to_color(rel_vel[time].numpy(), convert_to_bgr=False)).permute(2, 0, 1))
    vel_color = make_grid(rel_act_velocity_color, nrow=5)
    vel_norm = make_grid([torch.linalg.norm(vel, ord=2, dim=-1).unsqueeze(0) for vel in rel_vel], nrow=5, value_range=(0, 0.01))
    vel_color = vel_color.permute(1, 2, 0)
    vel_norm = (vel_norm*255).to(torch.uint8).permute(1, 2, 0)

    return vel_color, vel_norm

def prep_flow_vis(rel_phi):
    rel_flow_colors = []
    for time in range(rel_phi.shape[0]):
        rel_flow_colors.append(torch.from_numpy(flow_to_color(rel_phi[time], convert_to_bgr=False)).permute(2, 0, 1))
    flow_col = make_grid(rel_flow_colors, nrow=5)
    flow_col = flow_col.permute(1, 2, 0)

    return flow_col

def prep_sim_meas_vis(sim_meas, config):
    sim_meas = sim_meas.sum(dim=1, keepdim=True)
    sim_grid = make_grid([im for im in sim_meas], nrow=5, normalize=True)
    sim_grid = (sim_grid*255).cpu().to(torch.uint8).permute(1, 2, 0)

    log_sim_meas = torch.log(sim_meas)
    log_sim_grid = make_grid([im for im in log_sim_meas], nrow=5, value_range=(-32, -11), normalize=True)
    log_sim_grid = (log_sim_grid*255).cpu().to(torch.uint8).permute(1, 2, 0)
    return sim_grid, log_sim_grid

def prep_seg_vis(segs, pred_segs):
    gt_mask = (segs > 0).float()
    pred_mask = (pred_segs > 0).float()
    seg_comb_last = make_grid([torch.stack([gt_mask[-1], torch.zeros_like(gt_mask[-1]), pred_mask[-1]])], nrow=2, normalize=True)
    seg_comb_all = make_grid([torch.stack([s, torch.zeros_like(s), pr]) for s, pr in zip(gt_mask, pred_mask)], nrow=5, normalize=True)
    seg_comb_last = (seg_comb_last*255).to(torch.uint8).permute(1, 2, 0)
    seg_comb_all = (seg_comb_all*255).to(torch.uint8).permute(1, 2, 0)
    return seg_comb_all, seg_comb_last

def prep_detJ_vis(negJ, config, nr_time_frames, img_shape):
    if config.fin_diff_grad:
        J_det_grid = make_grid(negJ.unsqueeze(1), nrow=5, normalize=True, value_range=(0, 0.00001), pad_value=1)
    else:
        if len(negJ) < 4:
            negJ = negJ.unsqueeze(0)
        negJ = negJ / nr_time_frames
        negJ = [frame.reschape(img_shape) for frame in negJ]
        J_det_grid = make_grid(negJ, nrow=5, normalize=True, value_range=(0, 0.05))
    
    J_det_grid =(J_det_grid*255).cpu().to(torch.uint8).permute(1, 2, 0)
    return J_det_grid

def prep_vel_grad_vis(vel_grad, config, nr_time_frames, img_shape):
    if config.fin_diff_grad:
        grad_norm = make_grid([torch.linalg.norm(im, dim=-1).unsqueeze(0) for im in vel_grad], nrow=5, normalize=True, value_range=(0, 0.02))
    else:
        if len(vel_grad) < 4:
            vel_grad = vel_grad.unsqueeze(0)
        vel_grad = vel_grad / nr_time_frames
        grad_norm = make_grid([torch.linalg.norm(im, dim=-1).reshape(img_shape) for im in vel_grad], nrow=5, normalize=True, value_range=(0, 0.5))
    
    grad_norm = (grad_norm*255).cpu().to(torch.uint8).permute(1, 2, 0)
    return grad_norm

def prep_vel_lap_vis(lap, config, img_shape):
    if config.fin_diff_grad:
        # lap.shape T/1, H, W
        lap_grid = make_grid(lap.unsqueeze(1), nrow=5, normalize=True, value_range=(0, 0.09))
    else:
        # lap.shape H*W, 2
        lap_norm = torch.linalg.norm(lap, ord=2, dim=-1).reshape(img_shape).unsqueeze(0)
        lap_grid = make_grid(lap_norm, nrow=5, normalize=True, value_range=(0, 150))
    lap_grid = (lap_grid*255).cpu().to(torch.uint8).permute(1, 2, 0)
    return lap_grid

def prep_phi_grad_vis(phi_grad):
    phi_grad_norm = make_grid(torch.linalg.norm(phi_grad, dim=-1).unsqueeze(1), nrow=5, normalize=True, value_range=(0, 0.1))
    phi_grad_norm = (phi_grad_norm*255).cpu().to(torch.uint8).permute(1, 2, 0)
    return phi_grad_norm

def draw_deformed_grid(phi: torch.Tensor) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    for i in range(0, phi.shape[0], math.ceil(phi.shape[0]/64)):
        ax.plot(phi[i, :, 0], phi[i, :, 1], 'r-', linewidth=0.5)
    for i in range(0, phi.shape[1], math.ceil(phi.shape[1]/64)):
        ax.plot(phi[:, i, 0], phi[:, i, 1], 'r-', linewidth=0.5)
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_frame_on(False)
    ax.tick_params(tick10n=False)
    ax.set_aspect('equal')
    fig.tight_layout()
    return fig, ax


def prep_grid_def_vis(last_phi: torch.Tensor) -> torch.Tensor:
    fig, ax = draw_deformed_grid(last_phi)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    np_image = np.array(image).transpose(2, 0, 1)
    tens_image = torch.from_numpy(np_image).permute(1, 2, 0)
    plt.close(fig)
    return tens_image