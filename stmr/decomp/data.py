"""Load the GT cine and its magnitude for the decomposition experiment."""

from __future__ import annotations

import fastmri

from stmr.config import Config
from stmr.data.data_load import prepare_inputs
from stmr.utils.logging import get_logger

from .params import DecompParams


def load_gt(p: DecompParams):
    """Return (gt_im [T,2,H,W] complex, mag [T,1,H,W] magnitude, H, W) on p.device.

    Uses the pipeline's prepare_inputs only to fetch eval_inputs.gt_im (the fully-sampled
    adjoint); the init reconstruction it also builds is discarded — we register fixed GT.
    """
    cfg = Config(dataset=p.dataset, time_points=p.time_points,
                 slice_number=p.slice_number, start_frame=p.start_frame,
                 device=p.device).finalize()
    _, eval_inputs = prepare_inputs(cfg, get_logger())
    gt_im = eval_inputs.gt_im.to(p.device)                          # [T, 2, H, W]
    mag = fastmri.complex_abs(gt_im.movedim(1, -1)).unsqueeze(1)    # [T, 1, H, W]
    # Normalise magnitudes to unit peak so the data MSE is O(1e-4) rather than O(1e-7).
    # The regularity losses (grad_phi/log_detJ) act on the flow, not the image, so this makes
    # the reg lambdas portable/intuitive (O(1e-2)) instead of image-scale-dependent.
    mag = mag / (mag.max() + 1e-12)
    H, W = mag.shape[-2:]
    return gt_im, mag, H, W
