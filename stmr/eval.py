from argparse import Namespace
from logging import Logger

import torch
from torch.utils.tensorboard import SummaryWriter

from stmr.metrics.calc_metrics import calculate_metrics
from stmr.metrics.visualisation import prep_vis_summary_pdf
from stmr.state import Inputs, RegistrationResult
from stmr.utils.logging import log_metrics


def evaluate(config: Namespace, writer: SummaryWriter, logger: Logger,
             output: RegistrationResult, inputs: Inputs, eval_inputs) -> dict:
    model_outputs, loss_outputs = output.model_outputs, output.loss_outputs
    moving, epoch = output.best_moving, output.epoch
    all_metrics, time_stamps = output.all_metrics, output.time_stamps
    init_recon, gt_im = eval_inputs.init_recon, eval_inputs.gt_im
    rel_vel, abs_phi = model_outputs.rel_vel, model_outputs.abs_phi

    with torch.no_grad():
        # inputs.moving is the live recon parameter, mutated by the final optimizer step
        # after the best snapshot was taken. Evaluate the best-epoch recon instead, so the
        # reported metrics describe the same state as model_outputs/res.pt.
        best_inputs = Inputs(moving, inputs.moving_inr, inputs.fixed, inputs.forward)
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
    for loss_name, loss_dict in loss_outputs.losses.items():
        name = loss_dict['name']
        if loss_name != 'sim':
            name = descr_str + name + ':'
        logger.info(f'{name:<30} {loss_dict['time']:.4f}')
    return imgs_to_save