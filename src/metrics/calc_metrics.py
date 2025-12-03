import torch
import torch.nn.functional as F
import numpy as np

from src.metrics.prep_visuals import prep_moved_img_vis, prep_sim_meas_vis, prep_flow_vis, prep_vel_vis, add_loss_specific_imgs, prep_seg_vis, prep_grid_def_vis, prep_image_space_comp
from src.metrics.metric_utils import add_grad_stats, add_vel_stats, add_losses, add_dices, reshape_phi_and_vel, reduce_dim, add_cmr_eval_metrics

def calculate_metrics(losses, config, abs_phi, rel_vel, fixed, moving, seg_mov, seg_fix, moved, moved_im, gt_im, inverse, func, ST, collect_imgs, last_val=False):
    abs_phi, rel_phi, rel_vel = reshape_phi_and_vel(abs_phi, rel_vel, moving)

    metrics = {}
    metrics = add_losses(losses, metrics)
    metrics = add_cmr_eval_metrics(moved_im, gt_im, metrics)
    if config.debug or last_val:
        metrics = add_grad_stats(func, metrics)
        metrics = add_vel_stats(rel_vel, metrics)
        metrics, pred_segs = add_dices(config, seg_fix, seg_mov, ST, metrics)

    if collect_imgs:
        reduce = len(fixed.shape) == 5
        fixed, moved, moved_im, rel_phi, rel_vel, seg_fix, pred_segs, abs_phi, sim_loss = reduce_dim(fixed, moved, moved_im, rel_phi, rel_vel, seg_fix, pred_segs, abs_phi, losses, reduce)

        reg_last, reg_all, reg_imspace, moving_im = prep_moved_img_vis(fixed, moved, moved_im, moving, config)
        flow_col = prep_flow_vis(rel_phi)
        sim_grid, log_sim_grid = prep_sim_meas_vis(sim_loss, config)
        def_grid = prep_grid_def_vis(abs_phi[-1])

        imgs_to_save = {
            'imgs/reg_last':reg_last,
            'imgs/reg_all':reg_all,
            'imgs/reg_imspace':reg_imspace,
            'imgs/moving':moving_im,
            'flows/flow':flow_col,
            'energies/sim_loss':sim_grid,
            'energies/log_sim_loss':log_sim_grid,
            'grid_deform/grid_def_last_step':def_grid
        }

        if 'cmr' in config.dataset:
            image_space_comp = prep_image_space_comp(moved_im, gt_im)
            imgs_to_save['imgs/comp_imspace'] = image_space_comp

        if rel_vel is not None:
            vel_color, vel_norm = prep_vel_vis(rel_vel)
            imgs_to_save['flows/vel_col'] = vel_color
            imgs_to_save['flows/vel_norm'] = vel_norm

        if seg_fix is not None:
            seg_last, seg_all = prep_seg_vis(seg_fix, pred_segs)
            imgs_to_save['segmentations/seg_last'] = seg_last
            imgs_to_save['segmentations/seg_all'] = seg_all

        imgs_to_save = add_loss_specific_imgs(imgs_to_save, losses, config, abs_phi.shape[0], fixed.shape[1:], reduce_dim=reduce)
    else:
        imgs_to_save = None

    return metrics, imgs_to_save

def get_relevant_loss_names(config):
    losses = {'sim':{'name':'Similarity loss',
                       'lambda':config.lambda_st,
                       'time':0.}}

    include_all = False
    # include_all = config.debug
    if config.lambda_negJ > 0 or config.lambda_grd > 0 or include_all:
        losses['negJ'] = {'name':'Vel negative det J',
                       'lambda':config.lambda_negJ,
                       'time':0.}
        losses['grd'] = {'name':'Vel gradient',
                       'lambda':config.lambda_grd,
                       'time':0.}
    if config.lambda_lap > 0 or include_all:
        losses['lap'] = {'name':'Vel Laplacian',
                       'lambda':config.lambda_lap,
                       'time':0.}
    if config.lambda_pgr > 0 or include_all:
        losses['pgr'] = {'name':'Phi gradient',
                       'lambda':config.lambda_pgr,
                       'time':0.}
    if config.lambda_hel > 0 or include_all:
        losses['hyper_el'] = {'name':'Hyper elasticity',
                       'lambda':config.lambda_hel,
                       'time':0.}
    if (config.lambda_recon > 0 or include_all) and 'cmr' in config.dataset:
        losses['recon_reg'] = {'name':'Reconstruction reg',
                       'lambda':config.lambda_recon,
                       'time':0.}

    return losses