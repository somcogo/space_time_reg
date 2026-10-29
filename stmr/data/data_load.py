from dataclasses import dataclass
from logging import Logger

import torch
from torch import nn

from stmr.config import Config
from stmr.state import EvalInputs, Inputs

from .data_utils import get_data, get_init, get_operators
from .fft_utils import MaskedFT
from .known_motion import synthesize
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
    gt_phi: torch.Tensor | None = None  # the known motion, when the frames are synthetic (synth_flow)


def fold_slices(x: torch.Tensor) -> torch.Tensor:
    """[T, C, D, H, W] -> [T*D, C, H, W]: every (frame, slice) becomes its own 2D batch item, for
    the per-frame 2D init (nmAPG with the 2D WCRR prior). Inverse: unfold_slices."""
    return x.movedim(2, 1).flatten(0, 1)


def unfold_slices(x: torch.Tensor, n_slices: int) -> torch.Tensor:
    """[T*D, C, H, W] -> [T, C, D, H, W], the inverse of fold_slices."""
    return x.unflatten(0, (-1, n_slices)).movedim(1, 2).contiguous()


def prepare_data(config: Config) -> PreparedData:
    """Load k-space, build the forward operators, and produce the (un-solved) initial guess."""
    raw_kspace_data, gt_kspace_data, kspace_mask = get_data(config)
    if getattr(config, 'reverse_frames', False):
        raw_kspace_data, gt_kspace_data, kspace_mask = (
            x.flip(0) for x in (raw_kspace_data, gt_kspace_data, kspace_mask))
    # Cast loaded (float32) k-space to the active default dtype so the whole pipeline runs in
    # that precision (e.g. float64 when config.float64 is set). Mask stays bool.
    dt = torch.get_default_dtype()
    raw_kspace_data = raw_kspace_data.to(dt)
    gt_kspace_data = gt_kspace_data.to(dt)

    full_forw, full_adj, forw_subs, forw_subs_adj = get_operators(config, kspace_mask)
    forw_subs = forw_subs.to(config.device)
    forw_subs_adj = forw_subs_adj.to(config.device)

    # Normalise data: one scale for all loaded time points, the maximum of the GT magnitude image, so the
    # relative brightness between frames is kept (the FFT is linear, so scaling k-space scales the image
    # by the same factor). The magnitude, not |re|/|im|, so the scale does not depend on the image phase.
    if getattr(config, 'normalize_data', False):
        gt_scale = (full_adj(gt_kspace_data) ** 2).sum(dim=1).sqrt().max()
        raw_kspace_data = raw_kspace_data / gt_scale
        gt_kspace_data = gt_kspace_data / gt_scale

    # Known-motion test: replace the frames by frame 0 carried along a known motion (see
    # known_motion.py), and re-measure them with the same mask.
    gt_phi = None
    if getattr(config, 'synth_flow', ''):
        if not config.dataset.startswith('cmr'):
            raise ValueError("synth_flow builds the frames with the FFT; only cmr datasets are supported")
        gt_frames = full_adj(gt_kspace_data)
        if gt_frames.shape[2] != 1:
            raise ValueError(f"synth_flow builds a 2D known motion and needs a single slice; "
                             f"got {gt_frames.shape[2]} (slices={config.slices})")
        # The known motion lives on the slice's (H, W) grid: phi [T-1, H*W, 2].
        frames, gt_phi = synthesize(config, gt_frames[:, :, 0])
        gt_kspace_data = full_forw(frames.unsqueeze(2))
        raw_kspace_data = torch.where(kspace_mask, gt_kspace_data, torch.zeros_like(gt_kspace_data))

    # fixed keeps the full k-space shape, zero-filled at unmeasured entries. This supports
    # time-varying (k-t) masks where the number of measured rows differs between frames,
    # which the old row-subsampled representation could not express.
    fixed = raw_kspace_data.to(config.device)
    fixed = add_kspace_noise(fixed, kspace_mask.to(config.device), config.kspace_noise_sigma)
    gt_im = full_adj(gt_kspace_data).to(config.device)

    init = get_init(config, raw_kspace_data, gt_im,
                    forw_subs_adj(fixed).to(config.device), kspace_mask)
    return PreparedData(raw_kspace_data, gt_kspace_data, kspace_mask, full_forw, full_adj,
                        forw_subs, forw_subs_adj, fixed, gt_im, init, gt_phi)


def prepare_inputs(config: Config, logger: Logger, writer=None) -> tuple[Inputs, EvalInputs]:
    """Load k-space, build the forward operator, and produce the initial reconstruction.

    ``writer``: optional SummaryWriter. When given (and the nmAPG init is active), the init
    solve is logged to it under ``init/*`` via InitTBLogger -- this is the seam that lets the
    full pipeline record the init curves in the same event file as the registration.
    """
    data = prepare_data(config)
    raw_kspace_data, gt_kspace_data, kspace_mask = data.raw_kspace_data, data.gt_kspace_data, data.kspace_mask
    full_forw, full_adj, forw_subs, forw_subs_adj = data.full_forw, data.full_adj, data.forw_subs, data.forw_subs_adj
    fixed, gt_im, init = data.fixed, data.gt_im, data.init

    init_batched = fold_slices(init)
    fixed_batched = fold_slices(fixed)
    kspace_mask_batched = fold_slices(kspace_mask)
    gt_im_batched = fold_slices(gt_im)
    if config.init_skip:
        recon = init_batched.detach()
    elif config.use_nmapg:
        # nmAPG optimizes frames as batch items and drops converged ones, indexing x[idx]
        # and y[idx] but not the forward operator. Packing the mask into y as extra
        # channels keeps each frame's mask aligned with its measurements under that
        # subsetting; the data-fit functions unpack it again.
        packed_measurements = torch.cat([fixed_batched, kspace_mask_batched.to(fixed_batched)], dim=1)
        callback = None
        if writer is not None:
            # Built here (not by the caller) because it needs gt_im/kspace_mask, which only
            # exist inside this function.
            from stmr.metrics.init_logger import InitTBLogger
            callback = InitTBLogger(writer, gt_im_batched, kspace_mask_batched.to(config.device), fixed_batched, config)
        # init is an nn.Parameter; detach it, or nmAPG's in-place iterate updates build an
        # autograd graph that grows every iteration until the GPU runs out of memory.
        recon, _ = init_using_nmAPG(config, init_batched.detach(), packed_measurements, full_forw,
                                    full_adj, logger, callback=callback)
    else:
        init_batched = nn.Parameter(init_batched)
        forw_subs_batched = MaskedFT(kspace_mask_batched).to(config.device)
        recon, _ = init_with_grad_desc(config, init_batched, fixed_batched, forw_subs_batched)
    recon = unfold_slices(recon, fixed.shape[2])

    recon = add_image_noise(recon.detach(), config.init_noise_sigma).detach()
    moving = nn.Parameter(recon.clone()).to(config.device)

    inputs = Inputs(moving=moving, moving_inr=None, fixed=fixed, forward=forw_subs)
    eval_inputs = EvalInputs(init_recon=recon, gt_im=gt_im, seg_moving=None, seg_fixed=None,
                             gt_phi=data.gt_phi)
    return inputs, eval_inputs
