import logging
import time
import copy

import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint

from src.models.factory import get_func
from src.losses.losses import calculate_losses
from src.metrics.calc_metrics import get_relevant_loss_names, calculate_metrics
from src.metrics.visualisation import prep_vis_summary_pdf
from src.utils.spatial_utils import generate_coord_tensor, upsample_img_seg, get_relative_vel
from src.utils.spatial_transformer import get_spatial_transformer
from src.utils.log_and_save import log_metrics
from src.data.data_load import prepare_inputs

def registration(config, writer, logger:logging.Logger):
    time_points = torch.linspace(0, 1, config.time_points, device=config.device)
    init_recon, moving_inr, fixed, gt_im, seg_mov, seg_fix, forw, inverse = prepare_inputs(logger, config)
    moving = nn.Parameter(init_recon[0].clone())
    dims = len(fixed.shape) - 2
    if 'siren' in config.func_name or 'wire' in config.func_name:
        config.func_kwargs['layers'][0] = dims
        config.func_kwargs['layers'][-1] = dims
    if 'ensemble' in config.func_name:
        config.func_kwargs['time_points'] = time_points
    func = get_func(config.func_name, config.func_kwargs)
    func = func.to(config.device)
    optimizer = torch.optim.Adam(func.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if 'cmr' in config.dataset:
        optimizer.add_param_group({'params': moving, 'lr':config.recon_lr, 'weight_decay':0.})
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [180, 500, 1000])

    moving.requires_grad_(True)
    logger.info(f'Set require_grad for recon to {moving.requires_grad}')

    best_loss = 1e8
    time_stamps = np.zeros((7, config.epochs))
    time_stamps[6, 0] = time.time()
    losses_to_calc = get_relevant_loss_names(config)
    downsample = 1
    all_metrics = []

    for epoch in range(1, config.epochs + 1):


        # TODO: reimplement downsampling
        # if len(config.schedule) > 1  and epoch in config.schedule:
        #     imgs, segs, downsample = upsample_img_seg(og_img, og_seg, config, epoch)

        coord_tensor = generate_coord_tensor(moving.shape[1:], config.device)
        coord_tensor.requires_grad = True
        log_epoch = epoch % config.log_cadence == 0
        optimizer.zero_grad()

        time_stamps[0, epoch-1] = time.time()
        rel_vel = get_relative_vel(func, config, time_points, coord_tensor, keep_batch_dim=True)
        # rel_vel = get_relative_vel(func, config, time_points, coord_tensor, keep_batch_dim=config.debug)
        abs_phi = odeint(func, coord_tensor, time_points, method=config.solver, atol=config.atol, rtol=config.rtol, options={'step_size':config.step_size})

        # TODO: don't forget to remove clip
        # abs_phi = torch.clamp(abs_phi, min=-1, max=1)

        ST = get_spatial_transformer(abs_phi, moving.shape, config)

        time_stamps[1, epoch-1] = time.time()
        loss_sum, losses, moved, moved_im = calculate_losses(config, moving, moving_inr, fixed, forw, abs_phi, rel_vel, func, time_points, coord_tensor, losses_to_calc, downsample, ST)

        time_stamps[2, epoch-1] = time.time()
        loss_sum.backward()
        if scheduler is not None:
            scheduler.step()

        time_stamps[3, epoch-1] = time.time()
        optimizer.step()

        time_stamps[4, epoch-1] = time.time()
        with torch.no_grad():
            collect_imgs = (epoch % 25 == 0 or epoch == 1 or loss_sum < best_loss) and config.debug
            metrics, imgs_to_save = calculate_metrics(losses, config, abs_phi, rel_vel, fixed, moving, seg_mov, seg_fix, moved, moved_im, gt_im, init_recon, func, ST, collect_imgs)
            log_metrics(config, metrics, writer, epoch, imgs_to_save)
            all_metrics.append({'metrics':copy.deepcopy(metrics), 'losses':copy.deepcopy(losses)})

        time_stamps[5, epoch-1] = time.time()
        if epoch == 1 or log_epoch:
            log_msg = f'Epoch {epoch:4d}/{config.epochs}, Losses '
            for loss_type, loss_dict in losses.items():
                log_msg += f'{loss_type}  {loss_dict['lambda'] * loss_dict['mean']:.5f}     '
            logger.info(log_msg)
        if loss_sum <= best_loss and epoch > config.schedule[-1]:
            best_loss = loss_sum
            best_phi = abs_phi
            best_vel = rel_vel
            best_moved = moved
            best_moved_im = moved_im
            best_moving = moving
            best_st_dict = func.state_dict()
            best_images = imgs_to_save
            best_epoch = epoch
            best_losses = losses

        if epoch < config.epochs:
            time_stamps[6, epoch] = time.time()

    with torch.no_grad():
        metrics, imgs_to_save = calculate_metrics(losses, config, abs_phi, rel_vel, fixed, moving, seg_mov, seg_fix, moved, moved_im, gt_im, init_recon, func, ST, collect_imgs=True, last_val=True)
        log_metrics(config, metrics, writer, epoch + 10, imgs_to_save, last_val=True)
        prep_vis_summary_pdf(config, gt_im, init_recon, best_moved_im, best_phi, best_vel, all_metrics)
    best_images = imgs_to_save

    logger.info('-------------------------------------------------')
    logger.info(f'Time spent (sec) over {config.epochs} iterations')
    logger.info('-------------------------------------------------')
    logger.info(f'Data loader:            {(time_stamps[0] - time_stamps[6]).sum():.4f}')
    logger.info(f'ODE solver:             {(time_stamps[1] - time_stamps[0]).sum():.4f}')
    logger.info(f'Loss calc:              {(time_stamps[2] - time_stamps[1]).sum():.4f}')
    logger.info(f'Backprop:               {(time_stamps[3] - time_stamps[2]).sum():.4f}')
    logger.info(f'Optim:                  {(time_stamps[4] - time_stamps[3]).sum():.4f}')
    logger.info(f'Metric calc:            {(time_stamps[5] - time_stamps[4]).sum():.4f}')
    logger.info(f'Total:                  {(time_stamps[5] - time_stamps[6]).sum():.4f}')

    logger.info('-------------------------------------------------')
    if config.fin_diff_grad:
        descr_str = 'Finite diff'
    elif config.autograd_grid:
        descr_str = 'Autograd grid'
    else:
        descr_str = 'Autograd rand'
    for loss_name, loss_dict in losses.items():
        name = loss_dict['name']
        if loss_name != 'sim':
            name = descr_str + name + ':'
        logger.info(f'{name:<30} {loss_dict['time']:.4f}')

    return best_phi, best_vel, coord_tensor, best_moved, best_st_dict, best_images, best_losses, time_stamps, best_epoch, best_moving, best_moved_im