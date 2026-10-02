from dataclasses import dataclass
from logging import Logger

import torch
from torch import nn

from stmr.config import Config
from stmr.state import EvalInputs, Inputs

from .data_utils import get_data, get_init, get_operators
from .noise import add_image_noise, add_kspace_noise
from .recon_init import init_using_nmAPG, init_with_grad_desc


@dataclass
class PreparedData:
    """Everything produced before the initial-reconstruction solve.

    Split out of prepare_inputs so an init-only runner (see stmr.initsweep) can reuse the exact
    same data path instead of duplicating it and drifting from the pipeline.
    """

    raw_kspace_data: torch.Tensor
    gt_kspace_data: torch.Tensor
    kspace_mask: torch.Tensor
    full_forw: nn.Module
    full_adj: nn.Module
    forw_subs: nn.Module
    forw_subs_adj: nn.Module
    fixed: torch.Tensor
    gt_im: torch.Tensor
    init: nn.Parameter


def prepare_data(config: Config) -> PreparedData:
    """Load k-space, build the forward operators, and produce the (un-solved) initial guess."""
    raw_kspace_data, gt_kspace_data, kspace_mask = get_data(config)
    # Cast loaded (float32) k-space to the active default dtype so the whole pipeline runs in
    # that precision (e.g. float64 when config.float64 is set). Mask stays bool.
    dt = torch.get_default_dtype()
    raw_kspace_data = raw_kspace_data.to(dt)
    gt_kspace_data = gt_kspace_data.to(dt)

    full_forw, full_adj, forw_subs, forw_subs_adj = get_operators(config, kspace_mask)
    forw_subs = forw_subs.to(config.device)
    forw_subs_adj = forw_subs_adj.to(config.device)

    # fixed keeps the full k-space shape, zero-filled at unmeasured entries. This supports
    # time-varying (k-t) masks where the number of measured rows differs between frames,
    # which the old row-subsampled representation could not express.
    fixed = raw_kspace_data.to(config.device)
    fixed = add_kspace_noise(fixed, kspace_mask.to(config.device), config.kspace_noise_sigma)
    gt_im = full_adj(gt_kspace_data).to(config.device)

    init = get_init(config, raw_kspace_data, gt_im,
                    forw_subs_adj(fixed).to(config.device), kspace_mask)
    return PreparedData(raw_kspace_data, gt_kspace_data, kspace_mask, full_forw, full_adj,
                        forw_subs, forw_subs_adj, fixed, gt_im, init)


def prepare_inputs(config: Config, logger: Logger, writer=None) -> tuple[Inputs, EvalInputs]:
    """Load k-space, build the forward operator, and produce the initial reconstruction.

    ``writer``: optional SummaryWriter. When given (and the nmAPG init is active), the init
    solve is logged to it under ``init/*`` via InitTBLogger -- this is the seam that lets the
    full pipeline record the init curves in the same event file as the registration.
    """
    data = prepare_data(config)
    raw_kspace_data, gt_kspace_data, kspace_mask = data.raw_kspace_data, data.gt_kspace_data, data.kspace_mask
    full_forw, forw_subs, forw_subs_adj = data.full_forw, data.forw_subs, data.forw_subs_adj
    fixed, gt_im, init = data.fixed, data.gt_im, data.init

    if config.init_skip:
        recon = init.detach()
    elif config.use_nmapg:
        # nmAPG optimizes frames as batch items and drops converged ones, indexing x[idx]
        # and y[idx] but not the forward operator. Packing the mask into y as extra
        # channels keeps each frame's mask aligned with its measurements under that
        # subsetting; the data-fit functions unpack it again.
        packed_measurements = torch.cat([fixed, kspace_mask.to(fixed)], dim=1)
        callback = None
        if writer is not None:
            # Built here (not by the caller) because it needs gt_im/kspace_mask, which only
            # exist inside this function.
            from stmr.metrics.init_logger import InitTBLogger
            callback = InitTBLogger(writer, gt_im, kspace_mask.to(config.device), fixed, config)
        recon, _ = init_using_nmAPG(config, init, packed_measurements, full_forw,
                                    forw_subs_adj, logger, callback=callback)
    else:
        recon, _ = init_with_grad_desc(config, init, fixed, forw_subs)

    recon = add_image_noise(recon.detach(), config.init_noise_sigma).detach()
    moving = nn.Parameter(recon.clone()).to(config.device)

    inputs = Inputs(moving=moving, moving_inr=None, fixed=fixed, forward=forw_subs)
    eval_inputs = EvalInputs(init_recon=recon, gt_im=gt_im, seg_moving=None, seg_fixed=None)
    return inputs, eval_inputs
