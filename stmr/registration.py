import copy
import time
from logging import Logger

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchdiffeq import odeint_adjoint as odeint

from stmr.config import Config
from stmr.data.fft_utils import apply_hard_data_consistency
from stmr.losses.losses import calculate_losses
from stmr.losses.recon_reg import get_recon_regularizer
from stmr.metrics.calc_metrics import calculate_metrics, get_relevant_loss_names
from stmr.state import EvalInputs, Inputs, LossOutputs, ModelOutputs, RegistrationResult
from stmr.utils.logging import log_metrics
from stmr.utils.spatial_transformer import get_spatial_transformer
from stmr.utils.spatial_utils import generate_coord_tensor, get_relative_vel


def get_model_outputs(config: Config, func: nn.Module, coord_tensor: torch.Tensor,
                      inputs: Inputs) -> ModelOutputs:
    time_points = torch.tensor([0., 1.], device=config.device)
    init_value = coord_tensor.expand((config.time_points - 1, -1, -1))

    rel_vel = get_relative_vel(config, func, init_value, time_points, keep_batch_dim=True)
    abs_phi = odeint(func, init_value, time_points, method=config.solver, atol=config.atol,
                     rtol=config.rtol, options={'step_size': config.step_size})
    abs_phi = abs_phi[1]  # ignore abs_phi[0], which is just init_value anyway
    ST = get_spatial_transformer(abs_phi, inputs.moving.shape[1:], config)
    return ModelOutputs(rel_vel=rel_vel, abs_phi=abs_phi, transformer=ST)


def registration(config: Config, writer: SummaryWriter, logger: Logger, inputs: Inputs,
                 eval_inputs: EvalInputs, func: nn.Module) -> RegistrationResult:
    moving = inputs.moving
    optimizer = torch.optim.Adam(func.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if 'cmr' in config.dataset:
        optimizer.add_param_group({'params': moving, 'lr': config.recon_lr,
                                   'weight_decay': 0., 'eps': config.recon_eps})

    moving.requires_grad_(config.learn_recon)
    logger.info(f'Set require_grad for recon to {moving.requires_grad}')

    if config.hard_dc and hasattr(inputs.forward, 'mask'):
        # Start from a measurement-consistent recon so every epoch (including the motion
        # warm-up, which never updates the recon) sees the projected images.
        with torch.no_grad():
            moving.copy_(apply_hard_data_consistency(moving, inputs.fixed, inputs.forward.mask))

    best_loss = 1e8
    time_stamps = np.zeros((7, config.epochs))
    time_stamps[6, 0] = time.time()
    losses_to_calc = get_relevant_loss_names(config)
    # Built once (it's a frozen, pretrained net) instead of re-loading it from disk every
    # epoch inside calculate_losses.
    recon_regularizer = get_recon_regularizer(config) if 'recon_reg' in losses_to_calc else None
    all_metrics = []

    for epoch in range(1, config.epochs + 1):
        if config.motion_warmup > 0 and epoch in (1, config.motion_warmup + 1):
            in_warmup = epoch == 1
            moving.requires_grad_(False if in_warmup else config.learn_recon)
            logger.info(
                f'Motion warm-up: recon frozen until epoch {config.motion_warmup}' if in_warmup
                else f'Motion warm-up over: set require_grad for recon to '
                     f'{moving.requires_grad} at epoch {epoch}')
        # interval == epochs means alternation is disabled; the guard also keeps the
        # final epoch from accidentally freezing func (epochs % (2*epochs) == epochs).
        if config.interval < config.epochs:
            if epoch % (2 * config.interval) == config.interval:
                moving.requires_grad_(True)
                for param in func.parameters():
                    param.requires_grad = False
                logger.info(f'Set require_grad for recon to {moving.requires_grad} and for '
                            f'func to {param.requires_grad} at epoch {epoch}')
            elif epoch % (2 * config.interval) == 0 or epoch == 1:
                moving.requires_grad_(False)
                for param in func.parameters():
                    param.requires_grad = True
                logger.info(f'Set require_grad for recon to {moving.requires_grad} and for '
                            f'func to {param.requires_grad} at epoch {epoch}')

        coord_tensor = generate_coord_tensor(moving.shape[2:], config.device)
        coord_tensor.requires_grad = True
        log_epoch = epoch % config.log_cadence == 0
        optimizer.zero_grad()

        time_stamps[0, epoch - 1] = time.time()
        model_outputs = get_model_outputs(config, func, coord_tensor, inputs)

        time_stamps[1, epoch - 1] = time.time()
        loss_sum, loss_outputs = calculate_losses(config, inputs, model_outputs, coord_tensor,
                                                  losses_to_calc, recon_regularizer)

        time_stamps[2, epoch - 1] = time.time()
        loss_sum.backward()

        # Snapshot before optimizer.step() and hard DC mutate moving/func, so the saved
        # state is exactly the one loss_sum was computed on.
        if loss_sum <= best_loss and epoch > config.schedule[-1]:
            best_loss = loss_sum.detach()
            best_model_out = ModelOutputs(model_outputs.rel_vel.clone(),
                                          model_outputs.abs_phi.clone(), None)
            best_loss_out = LossOutputs(
                copy.deepcopy(loss_outputs.losses),
                loss_outputs.moved_imspace.clone() if loss_outputs.moved_imspace is not None else None)
            best_moving = inputs.moving.detach().clone()
            best_st_dict = copy.deepcopy(func.state_dict())
            best_epoch = epoch
            coords = coord_tensor

        time_stamps[3, epoch - 1] = time.time()
        optimizer.step()

        time_stamps[4, epoch - 1] = time.time()
        if config.hard_dc and hasattr(inputs.forward, 'mask'):
            # Project the recon back onto the measurements: the regularizer terms may
            # only fill in the unmeasured k-space entries, never corrupt measured ones.
            with torch.no_grad():
                moving.copy_(apply_hard_data_consistency(moving, inputs.fixed, inputs.forward.mask))
        with torch.no_grad():
            extended_log = (epoch % 25 == 0 or epoch == 1 or loss_sum < best_loss) and config.debug
            metrics, imgs_to_save = calculate_metrics(config, func, inputs, eval_inputs,
                                                      model_outputs, loss_outputs, extended_log)
            log_metrics(config, metrics, writer, epoch, imgs_to_save)
            # Reduce the per-pixel 'loss' maps to per-frame sums on CPU before archiving:
            # keeping the full GPU maps in all_metrics leaks GPU memory linearly in epochs
            # and OOMs multi-thousand-epoch runs. The summary pdf only uses per-frame sums.
            slim_losses = {name: {**{k: v for k, v in d.items() if k != 'loss'},
                                  'loss': (d['loss'].sum(dim=tuple(range(1, d['loss'].dim())))
                                           if d['loss'].dim() > 1 else d['loss']).cpu()}
                           for name, d in loss_outputs.losses.items()}
            all_metrics.append({'metrics': copy.deepcopy(metrics), 'losses': slim_losses})

        time_stamps[5, epoch - 1] = time.time()
        if epoch == 1 or log_epoch:
            log_msg = f'Epoch {epoch:4d}/{config.epochs}, Losses '
            for loss_type, loss_dict in loss_outputs.losses.items():
                log_msg += f"{loss_type}  {loss_dict['lambda'] * loss_dict['mean']:.5f}     "
            logger.info(log_msg)
        if epoch < config.epochs:
            time_stamps[6, epoch] = time.time()

    return RegistrationResult(
        model_outputs=best_model_out,
        loss_outputs=best_loss_out,
        best_moving=best_moving,
        st_dict=best_st_dict,
        epoch=best_epoch,
        all_metrics=all_metrics,
        time_stamps=time_stamps,
        coords=coords,
    )
