import logging
import time
import math

import numpy as np
import torch
from torchdiffeq import odeint_adjoint as odeint

from src.networks import get_func
from src.utils import generate_grid_tensor, calculate_metrics, log_metrics, generate_coord_tensor
from src.losses import calculate_losses
from src.siren.dataio import get_mgrid

def registration(config, data, writer, logger:logging.Logger):
    img_shape = data[0].shape[1:]
    dims = len(img_shape)
    if config.func_name in ['siren', 'sirent']:
        config.func_kwargs['layers'][0] = dims
        config.func_kwargs['layers'][-1] = dims
    elif config.func_name in ['wire', 'wiret']:
        config.func_kwargs['in_features'] = dims
        config.func_kwargs['out_features'] = dims
    func = get_func(config.func_name, config.func_kwargs)
    func = func.to(config.device)
    coord_tensor = generate_coord_tensor(img_shape, config.device)
    coord_tensor.requires_grad = True
    time_points = torch.arange(config.time_points, device=config.device) / 19
    optimizer = torch.optim.Adam(func.parameters(), lr=config.lr)

    best_loss = 1e8
    time_stamps = np.zeros((7, config.epochs))
    time_stamps[6, 0] = time.time()
    loss_times = np.zeros((10, config.epochs)) if config.debug else np.zeros((5, config.epochs))

    for epoch in range(1, config.epochs + 1):
        log_epoch = epoch % config.log_cadence == 0
        optimizer.zero_grad()

        time_stamps[0, epoch-1] = time.time()
        if config.debug or (config.fin_diff_grad and config.lambda_grd + config.lambda_negJ + config.lambda_lap > 0):
            rel_vel = func(time_points[1], coord_tensor)
            if config.func_name == 'siren' and config.debug:
                rel_vel = rel_vel.unsqueeze(0)
            elif config.func_name == 'sirent':
                if config.debug:
                    rel_vel = []
                    for t in time_points:
                        rel_vel.append(func(t, coord_tensor))
                    rel_vel = torch.stack(rel_vel)
                else:
                    for t in time_points[2:]:
                        rel_vel = rel_vel + func(t, coord_tensor)
                    rel_vel = []
                    for t in time_points:
                        rel_vel.append(func(t, coord_tensor))
                    rel_vel = torch.stack(rel_vel)
        else:
            rel_vel = None
        abs_phi = odeint(func, coord_tensor, time_points, method=config.solver, atol=config.atol, rtol=config.rtol, options={'step_size':config.step_size})
        # abs_phi = torch.relu(abs_phi+1) - 1
        # abs_phi = -torch.relu(-abs_phi+1) + 1

        time_stamps[1, epoch-1] = time.time()
        losses, moved_imgs, visuals, loss_time, losses_to_log = calculate_losses(config, abs_phi, data, rel_vel, func, time_points, coord_tensor)
        loss = sum(losses)
        loss_times[:, epoch - 1] = loss_time

        time_stamps[2, epoch-1] = time.time()
        loss.backward()

        time_stamps[3, epoch-1] = time.time()
        optimizer.step()

        time_stamps[4, epoch-1] = time.time()
        with torch.no_grad():
            collect_imgs = epoch % 25 == 0 or epoch == 1 or losses[0] < best_loss
            metrics, imgs_to_save = calculate_metrics(losses, config, abs_phi, rel_vel, data, moved_imgs, func, visuals, collect_imgs)
            log_metrics(config, metrics, writer, epoch, losses_to_log, imgs_to_save)

        time_stamps[5, epoch-1] = time.time()
        if epoch == 1 or log_epoch:
            logger.info(f'Epoch {epoch:4d}/{config.epochs}, Losses Sim {losses[0]:.3f}    NegJ {losses[1]:.3f}    VGrad {losses[2]:.3f}    Lap {losses[3]:.3f}    PhiGrad {losses[4]:.3f}')
        if losses[0] < best_loss:
            best_loss = losses[0]
            best_phi = abs_phi
            best_vel = rel_vel
            best_moved = moved_imgs
            best_st_dict = func.state_dict()
            best_images = imgs_to_save if imgs_to_save is not None else best_images
            best_logged_losses = losses_to_log
            best_epoch = epoch

        if epoch < config.epochs:
            time_stamps[6, epoch] = time.time()

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
    logger.info(f'Similarity loss:        {(loss_times[1] - loss_times[0]).sum():.4f}')
    if config.debug:
        logger.info(f'Finite diff vel J:      {(loss_times[2] - loss_times[1]).sum():.4f}')
        logger.info(f'Autograd grid vel J:    {(loss_times[3] - loss_times[2]).sum():.4f}')
        logger.info(f'Autograd rand vel J:    {(loss_times[4] - loss_times[3]).sum():.4f}')
        logger.info(f'Finite diff vel Lap:    {(loss_times[5] - loss_times[4]).sum():.4f}')
        logger.info(f'Autograd grid vel Lap:  {(loss_times[6] - loss_times[5]).sum():.4f}')
        logger.info(f'Autograd rand vel Lap:  {(loss_times[7] - loss_times[6]).sum():.4f}')
        logger.info(f'Finite diff phi J:      {(loss_times[8] - loss_times[7]).sum():.4f}')
        logger.info(f'Autograd grid phi J:    {(loss_times[9] - loss_times[8]).sum():.4f}')
    else:
        if config.fin_diff_grad:
            descr_str = 'Finite diff'
        elif config.autograd_grid:
            descr_str = 'Autograd grid'
        else:
            descr_str = 'Autograd rand'
        logger.info(f'{descr_str} vel J:      {(loss_times[2] - loss_times[1]).sum():.4f}')
        logger.info(f'{descr_str} vel Lap:    {(loss_times[3] - loss_times[2]).sum():.4f}')
        logger.info(f'{descr_str} phi J:      {(loss_times[4] - loss_times[3]).sum():.4f}')

    return best_phi, best_vel, coord_tensor, best_moved, best_st_dict, best_images, best_logged_losses, time_stamps, loss_times, best_epoch