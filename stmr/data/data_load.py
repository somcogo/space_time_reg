from logging import Logger

import torch
from torch import nn

from stmr.config import Config
from stmr.state import EvalInputs, Inputs

from .data_utils import get_data, get_init, get_operators
from .recon_init import init_using_nmAPG, init_with_grad_desc


def prepare_inputs(config: Config, logger: Logger) -> tuple[Inputs, EvalInputs]:
    """Load k-space, build the forward operator, and produce the initial reconstruction."""
    raw_kspace_data, gt_kspace_data, kspace_mask = get_data(config)

    full_forw, full_adj, forw_subs, forw_subs_adj = get_operators(config, kspace_mask)
    forw_subs = forw_subs.to(config.device)
    forw_subs_adj = forw_subs_adj.to(config.device)

    # fixed keeps the full k-space shape, zero-filled at unmeasured entries. This supports
    # time-varying (k-t) masks where the number of measured rows differs between frames,
    # which the old row-subsampled representation could not express.
    fixed = raw_kspace_data.to(config.device)
    gt_im = full_adj(gt_kspace_data).to(config.device)

    init = get_init(config, raw_kspace_data, gt_im,
                    forw_subs_adj(fixed).to(config.device), kspace_mask)

    if config.init_skip:
        recon = init.detach()
    elif config.use_nmapg:
        # nmAPG optimizes frames as batch items and drops converged ones, indexing x[idx]
        # and y[idx] but not the forward operator. Packing the mask into y as extra
        # channels keeps each frame's mask aligned with its measurements under that
        # subsetting; the data-fit functions unpack it again.
        packed_measurements = torch.cat([fixed, kspace_mask.to(fixed)], dim=1)
        recon, _ = init_using_nmAPG(config, init, packed_measurements, full_forw,
                                    forw_subs_adj, logger)
    else:
        recon, _ = init_with_grad_desc(config, init, fixed, forw_subs)

    recon = recon.detach()
    moving = nn.Parameter(recon.clone()).to(config.device)

    inputs = Inputs(moving=moving, moving_inr=None, fixed=fixed, forward=forw_subs)
    eval_inputs = EvalInputs(init_recon=recon, gt_im=gt_im, seg_moving=None, seg_fixed=None)
    return inputs, eval_inputs
