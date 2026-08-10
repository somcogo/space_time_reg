from argparse import Namespace

from torch import nn

from stmr.metrics.metric_utils import (
    add_cmr_eval_metrics,
    add_dices,
    add_grad_stats,
    add_gt_error,
    add_losses,
    add_vel_stats,
    reduce_dim,
    reshape_phi_and_vel,
)
from stmr.metrics.prep_visuals import (
    add_loss_specific_imgs,
    prep_flow_vis,
    prep_grid_def_vis,
    prep_image_space_comp,
    prep_imdiff_energy_vis,
    prep_init_recon,
    prep_moved_img_vis,
    prep_seg_vis,
    prep_sim_meas_vis,
    prep_vel_vis,
)


def calc_init_metrics(config: Namespace, eval_inputs) -> list[dict]:
    init_recon = eval_inputs.init_recon
    gt_im = eval_inputs.gt_im
    is_complex = 'cmr' in config.dataset or 'heart' in config.dataset
    metrics = {}
    if is_complex:
        metrics = add_cmr_eval_metrics(init_recon, gt_im, metrics)
    imgs_to_save = {'imgs/init_recon':prep_init_recon(init_recon, is_complex)}
    return metrics, imgs_to_save

def calculate_metrics(config: Namespace, func: nn.Module, inputs, eval_inputs, model_outputs, loss_outputs, extended_log=False):
    rel_vel, abs_phi, ST = model_outputs.rel_vel, model_outputs.abs_phi, model_outputs.transformer
    losses, moved_im = loss_outputs.losses, loss_outputs.moved_imspace
    fixed = inputs.fixed
    moving = inputs.moving.detach().cpu()
    init_recon, gt_im, seg_moving, seg_fixed = (eval_inputs.init_recon, eval_inputs.gt_im,
                                                eval_inputs.seg_moving, eval_inputs.seg_fixed)
    abs_phi, rel_phi, rel_vel = reshape_phi_and_vel(abs_phi, rel_vel, moving)

    metrics = {}
    metrics = add_losses(losses, metrics)
    if 'cmr' in config.dataset or 'heart' in config.dataset:
        metrics = add_cmr_eval_metrics(moving, gt_im, metrics)
        metrics = add_gt_error(moving, gt_im, init_recon, rel_vel, metrics)
    if extended_log:
        metrics = add_grad_stats(func, metrics)
        metrics = add_vel_stats(rel_vel, metrics)
        metrics, pred_segs = add_dices(config, seg_fixed, seg_moving, ST, metrics)

        reduce = len(fixed.shape) == 5
        fixed, moved_im, rel_phi, rel_vel, seg_fixed, pred_segs, abs_phi, sim_loss, imdiff_loss = reduce_dim(fixed, moved_im, rel_phi, rel_vel, seg_fixed, pred_segs, abs_phi, losses, reduce)

        flow_col = prep_flow_vis(rel_phi)
        sim_grid, log_sim_grid = prep_sim_meas_vis(sim_loss)
        def_grid = prep_grid_def_vis(abs_phi[-1])

        imgs_to_save = {
            'flows/flow':flow_col,
            'energies/sim_loss':sim_grid,
            'energies/log_sim_loss':log_sim_grid,
            'grid_deform/grid_def_last_step':def_grid
        }

        # The warped-image panels need a moved image, which only the imdiff loss produces.
        # When that loss is off (lambda_rl2 == 0) skip them; the recon itself is still
        # shown via imgs/comp_imspace below.
        if moved_im is not None:
            reg_last, reg_all, reg_imspace, moving_im = prep_moved_img_vis(moved_im, moving, config)
            imgs_to_save['imgs/reg_last'] = reg_last
            imgs_to_save['imgs/reg_all'] = reg_all
            imgs_to_save['imgs/reg_imspace'] = reg_imspace
            imgs_to_save['imgs/moving'] = moving_im

        if imdiff_loss is not None:
            imdiff_grid = prep_imdiff_energy_vis(imdiff_loss)
            imgs_to_save['energies/im_diff_loss'] = imdiff_grid

        if 'cmr' in config.dataset or config.dataset == 'toy_square':
            image_space_comp = prep_image_space_comp(moving, gt_im)
            imgs_to_save['imgs/comp_imspace'] = image_space_comp

        if rel_vel is not None:
            vel_color, vel_norm = prep_vel_vis(rel_vel)
            imgs_to_save['flows/vel_col'] = vel_color
            imgs_to_save['flows/vel_norm'] = vel_norm

        if seg_fixed is not None:
            seg_last, seg_all = prep_seg_vis(seg_fixed, pred_segs)
            imgs_to_save['segmentations/seg_last'] = seg_last
            imgs_to_save['segmentations/seg_all'] = seg_all

        imgs_to_save = add_loss_specific_imgs(imgs_to_save, losses, reduce_dim=reduce)
    else:
        imgs_to_save = None

    return metrics, imgs_to_save

def get_relevant_loss_names(config):
    # Only losses with a strictly positive weight are computed: a zero-weight term
    # contributes nothing to loss_sum but still builds its full autograd graph every
    # epoch, which is pure wasted compute (and memory). Each term is gated by its own
    # lambda. 'sim' is always kept: it is the data-fidelity term and its per-pixel map is
    # required by the metrics/visualisation path (reduce_dim, prep_sim_meas_vis).
    is_cmr = 'cmr' in config.dataset
    candidates = [
        ('grad_phi', 'Grad phi (Dphi - Id)', config.lambda_grad_phi, True),
        ('detJ',     'Det(Dphi) - 1',        config.lambda_detJ, True),
        ('logdetJ',  'log det(Dphi)',        config.lambda_logdetJ, True),
        ('grd',      'Vel gradient',         config.lambda_grd,  True),
        ('lap',      'Vel Laplacian',        config.lambda_lap,  True),
        ('hyper_el', 'Hyper elasticity',     config.lambda_hel,  True),
        ('recon_reg', 'Reconstruction reg',  config.lambda_recon, is_cmr),
        ('mcdc',     'Motion comp DC',       config.lambda_mcdc, is_cmr),
        ('imdiff',   'Image space diff',     config.lambda_rl2,  True),
    ]

    losses = {'sim': {'name': 'Similarity loss', 'lambda': config.lambda_st, 'time': 0.}}
    for key, name, lam, enabled in candidates:
        if enabled and lam > 0:
            losses[key] = {'name': name, 'lambda': lam, 'time': 0.}

    return losses