from logging import Logger
import time
import copy
from argparse import Namespace

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchdiffeq import odeint_adjoint as odeint

from src.losses.losses import calculate_losses
from src.metrics.calc_metrics import get_relevant_loss_names, calculate_metrics
from src.utils.spatial_utils import generate_coord_tensor, get_relative_vel
from src.utils.spatial_transformer import get_spatial_transformer
from src.utils.log_and_save import log_metrics

def get_model_outputs(config: Namespace, func: nn.Module, coord_tensor: torch.Tensor, time_points: torch.Tensor, inputs: list):
    moving = inputs[0]
    rel_vel = get_relative_vel(config, func, coord_tensor, time_points, keep_batch_dim=True)
    abs_phi = odeint(func, coord_tensor, time_points, method=config.solver, atol=config.atol, rtol=config.rtol, options={'step_size':config.step_size})
    ST = get_spatial_transformer(abs_phi, moving.shape, config)
    return rel_vel, abs_phi, ST

def registration(config: Namespace, writer: SummaryWriter, logger:Logger, inputs: list, eval_inputs: list, func: nn.Module):
    moving = inputs[0]
    time_points = inputs[-1]
    optimizer = torch.optim.Adam(func.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if 'cmr' in config.dataset:
        optimizer.add_param_group({'params': moving, 'lr':config.recon_lr, 'weight_decay':0.})
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [180, 500, 1000])

    moving.requires_grad_(True)
    if logger is not None:
        logger.info(f'Set require_grad for recon to {moving.requires_grad}')
    else:
        print(f'Set require_grad for recon to {moving.requires_grad}')

    best_loss = 1e8
    time_stamps = np.zeros((7, config.epochs))
    time_stamps[6, 0] = time.time()
    losses_to_calc = get_relevant_loss_names(config)
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
        model_outputs = get_model_outputs(config, func, coord_tensor, time_points, inputs)

        time_stamps[1, epoch-1] = time.time()
        loss_sum, loss_outputs = calculate_losses(config, inputs, model_outputs, coord_tensor, losses_to_calc)

        time_stamps[2, epoch-1] = time.time()
        loss_sum.backward()
        if scheduler is not None:
            scheduler.step()

        time_stamps[3, epoch-1] = time.time()
        optimizer.step()

        time_stamps[4, epoch-1] = time.time()
        with torch.no_grad():
            extended_log = (epoch % 25 == 0 or epoch == 1 or loss_sum < best_loss) and config.debug
            metrics, imgs_to_save = calculate_metrics(config, func, inputs, eval_inputs, model_outputs, loss_outputs, extended_log)
            log_metrics(config, metrics, writer, epoch, imgs_to_save)
            all_metrics.append({'metrics':copy.deepcopy(metrics), 'losses':copy.deepcopy(loss_outputs[0])})

        time_stamps[5, epoch-1] = time.time()
        if epoch == 1 or log_epoch:
            log_msg = f'Epoch {epoch:4d}/{config.epochs}, Losses '
            for loss_type, loss_dict in loss_outputs[0].items():
                log_msg += f'{loss_type}  {loss_dict['lambda'] * loss_dict['mean']:.5f}     '
            if logger is not None:
                logger.info(log_msg)
            else:
                print(log_msg)
        if loss_sum <= best_loss and epoch > config.schedule[-1]:
            best_model_out = model_outputs
            best_loss_out = loss_outputs
            best_moving = inputs[0]
            best_st_dict = func.state_dict()
            best_epoch = epoch
            coords = coord_tensor

        if epoch < config.epochs:
            time_stamps[6, epoch] = time.time()

    return best_model_out, best_loss_out, best_moving, best_st_dict, best_epoch, all_metrics, time_stamps, coords