from argparse import Namespace
from logging import Logger

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from src.metrics.calc_metrics import calculate_metrics
from src.utils.log_and_save import log_metrics
from src.metrics.visualisation import prep_vis_summary_pdf

def evaluate(config: Namespace, writer: SummaryWriter, logger: Logger, output: list, inputs: list, eval_inputs: list) -> dict[torch.Tensor]:
    model_outputs, loss_outputs, moving, _, epoch, all_metrics, time_stamps, _ = output
    init_recon, gt_im, _, _ = eval_inputs
    rel_vel, abs_phi, _ = model_outputs
    loss_dict, _ = loss_outputs
    
    with torch.no_grad():
        # inputs[0] is the live recon parameter, mutated by the final optimizer step (and
        # hard DC) after the best snapshot was taken. Evaluate the best-epoch recon instead,
        # so the reported metrics describe the same state as model_outputs and res.pt.
        best_inputs = [moving] + inputs[1:]
        metrics, imgs_to_save = calculate_metrics(config, None, best_inputs, eval_inputs, model_outputs, loss_outputs, extended_log=True)
        log_metrics(config, {}, writer, epoch, imgs_to_save, last_val=True)
        prep_vis_summary_pdf(config, gt_im, init_recon, moving, abs_phi, rel_vel, all_metrics)

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
    descr_str = 'Finite diff'
    for loss_name, loss_dict in loss_outputs[0].items():
        name = loss_dict['name']
        if loss_name != 'sim':
            name = descr_str + name + ':'
        logger.info(f'{name:<30} {loss_dict['time']:.4f}')
    return imgs_to_save