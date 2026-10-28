import copy
import time
from logging import Logger

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchdiffeq import odeint, odeint_adjoint

from stmr.config import Config
from stmr.data.data_utils import get_dataset_capabilities
from stmr.data.recon_init import get_reg
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
    solve = odeint_adjoint if config.adjoint else odeint
    abs_phi = solve(func, init_value, time_points, method=config.solver, atol=config.atol,
                    rtol=config.rtol, options={'step_size': config.step_size})
    abs_phi = abs_phi[1]  # ignore abs_phi[0], which is just init_value anyway
    ST = get_spatial_transformer(abs_phi, inputs.moving.shape[1:], config)
    return ModelOutputs(rel_vel=rel_vel, abs_phi=abs_phi, transformer=ST)


def get_fixed_motion_outputs(config: Config, inputs: Inputs, eval_inputs: EvalInputs) -> ModelOutputs:
    """Motion held fixed instead of learned (oracle and control runs): the known motion of
    synthetic frames ('gt'), no motion ('identity'), or the phi saved in a res.pt."""
    T = inputs.moving.shape[0]
    H, W = inputs.moving.shape[2:]
    coord = generate_coord_tensor((H, W), config.device)
    if config.fixed_motion == 'identity':
        abs_phi = coord.expand(T - 1, -1, -1).contiguous()
    elif config.fixed_motion == 'gt':
        if eval_inputs.gt_phi is None:
            raise ValueError("fixed_motion='gt' needs frames with a known motion (set synth_flow)")
        abs_phi = eval_inputs.gt_phi.to(config.device)
    else:
        abs_phi = torch.load(config.fixed_motion, map_location=config.device,
                             weights_only=False)['phi'][:T - 1]
    abs_phi = abs_phi.to(inputs.moving.dtype)
    # The metrics/visuals expect a velocity; for a fixed motion the displacement stands in.
    ST = get_spatial_transformer(abs_phi, inputs.moving.shape[1:], config)
    return ModelOutputs(rel_vel=abs_phi - coord, abs_phi=abs_phi, transformer=ST)


def velocity_lr_factor(config: Config, step: int) -> float:
    """Multiplier on config.lr for optimizer step `step` (0-based): linear warm-up over
    lr_warmup steps, then constant or cosine-decayed to lr_min by the last epoch."""
    if step < config.lr_warmup:
        return (step + 1) / config.lr_warmup
    if config.lr_schedule == 'const':
        return 1.0
    if config.lr_schedule == 'cosine':
        progress = (step - config.lr_warmup) / max(1, config.epochs - 1 - config.lr_warmup)
        lr = config.lr_min + 0.5 * (config.lr - config.lr_min) * (1 + np.cos(np.pi * progress))
        return lr / config.lr
    raise ValueError(f"Unknown lr_schedule {config.lr_schedule!r}")


def registration(config: Config, writer: SummaryWriter, logger: Logger, inputs: Inputs,
                 eval_inputs: EvalInputs, func: nn.Module) -> RegistrationResult:
    moving = inputs.moving
    optimizer = torch.optim.Adam(func.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if get_dataset_capabilities(config).joint_recon and not config.template_warp_recon:
        optimizer.add_param_group({'params': moving, 'lr': config.recon_lr,
                                   'weight_decay': 0., 'eps': config.recon_eps})
    # Only the velocity net (group 0) is scheduled: the recon is in an early-stopping regime
    # where the result tracks recon_lr * epochs, so decaying it would change the recon.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, [lambda step: velocity_lr_factor(config, step)]
        + [lambda step: 1.0] * (len(optimizer.param_groups) - 1))

    # Template-warp recon: no learned image. Each epoch the recon is a differentiable warp of
    # the fully-sampled GT template (frame 0); only the motion net is optimised (above).
    template = eval_inputs.gt_im[:1].detach() if config.template_warp_recon else None

    moving.requires_grad_(config.learn_recon)
    logger.info(f'Set require_grad for recon to {moving.requires_grad}')

    # Fixed motion: built once and reused every epoch; the motion net gets no gradient.
    fixed_outputs = None
    if config.fixed_motion:
        fixed_outputs = get_fixed_motion_outputs(config, inputs, eval_inputs)
        logger.info(f'Motion fixed ({config.fixed_motion!r}), not learned')

    # Fully-sampled frozen template: pin the first `template_frames` recon frames to GT. Their
    # gradients are zeroed every step below so Adam never moves them off GT.
    n_template = config.template_frames
    if n_template > 0:
        with torch.no_grad():
            moving[:n_template].copy_(eval_inputs.gt_im[:n_template])
        logger.info(f'Froze first {n_template} recon frame(s) at GT (fully-sampled template)')

    best_loss = 1e8
    time_stamps = np.zeros((7, config.epochs))
    time_stamps[6, 0] = time.time()
    losses_to_calc = get_relevant_loss_names(config)
    # Built once (it's a frozen, pretrained net) instead of re-loading it from disk every
    # epoch inside calculate_losses. joint_objective='init' uses the init's regularizer choice.
    get_regularizer = get_reg if config.joint_objective == 'init' else get_recon_regularizer
    recon_regularizer = get_regularizer(config) if 'recon_reg' in losses_to_calc else None
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
        model_outputs = (fixed_outputs if fixed_outputs is not None
                         else get_model_outputs(config, func, coord_tensor, inputs))

        if config.template_warp_recon:
            # recon_t = warp(template, phi_{0->t}); recon_0 = template. Differentiable in phi,
            # so the k-space data term drives the motion net directly (no free recon).
            G = model_outputs.abs_phi.shape[0]
            warped = model_outputs.transformer.apply(
                template.expand(G, *template.shape[1:]).contiguous())
            inputs.moving = torch.cat([template, warped], dim=0)

        time_stamps[1, epoch - 1] = time.time()
        loss_sum, loss_outputs = calculate_losses(config, inputs, model_outputs, coord_tensor,
                                                  losses_to_calc, recon_regularizer)

        time_stamps[2, epoch - 1] = time.time()
        loss_sum.backward()

        if n_template > 0 and moving.grad is not None:
            moving.grad[:n_template].zero_()

        # Snapshot before optimizer.step() mutates moving/func, so the saved state is
        # exactly the one loss_sum was computed on.
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
        velocity_lr = optimizer.param_groups[0]['lr']
        optimizer.step()
        scheduler.step()

        time_stamps[4, epoch - 1] = time.time()
        metric_epoch = epoch == 1 or epoch % config.metric_cadence == 0 or epoch == config.epochs
        with torch.no_grad():
            if metric_epoch:
                extended_log = (epoch % 25 == 0 or epoch == 1 or loss_sum < best_loss) and config.debug
                metrics, imgs_to_save = calculate_metrics(config, func, inputs, eval_inputs,
                                                          model_outputs, loss_outputs, extended_log)
                log_metrics(config, metrics, writer, epoch, imgs_to_save)
                writer.add_scalar('optim/velocity lr', velocity_lr, epoch)
                # Reduce the per-pixel 'loss' maps to per-frame sums on CPU before archiving:
                # keeping the full GPU maps in all_metrics leaks GPU memory linearly in epochs
                # and OOMs multi-thousand-epoch runs. The summary pdf only uses per-frame sums.
                slim_losses = {name: {**{k: v for k, v in d.items() if k != 'loss'},
                                      'loss': (d['loss'].sum(dim=tuple(range(1, d['loss'].dim())))
                                               if d['loss'].dim() > 1 else d['loss']).cpu()}
                               for name, d in loss_outputs.losses.items()}
                all_metrics.append({'epoch': epoch, 'metrics': copy.deepcopy(metrics),
                                    'losses': slim_losses})

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
