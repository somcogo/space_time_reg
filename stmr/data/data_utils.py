import argparse
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import spare_data
from .fft_utils import FastmriFT, FastmriIFT, MaskedFT, MaskedIFT

# Anchor data paths to the repo root (this file is <root>/stmr/data/data_utils.py), so loading
# works from any working directory -- e.g. a notebook in notebooks/, not just the repo root.
_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def _cmr(name: str) -> str:
    """Absolute path to a CMRxRecon test file, independent of the current working directory."""
    return str(_DATA_ROOT / "processed" / "cmrxrecon" / "test" / name)


CMR_VIEWS = ('sax', 'lax')


def _cmr_challenge_slice(patient: str, slices: list[int], view: str = 'sax') -> torch.Tensor:
    """Full single-coil k-space of slices straight from the challenge release, [T, 2, D, H, W].

    view 'sax': the short-axis stack (P001: 10 slices of 204 x 512). 'lax': the long-axis file,
    whose "slices" are separate views, not a stack (P001: 3 of 204 x 448).
    The processed files are this times one real scale per patient and a sign flip on some frames
    (fixed per patient, fitted on slice 0); the pipeline is invariant to both.
    """
    import h5py
    if view not in CMR_VIEWS:
        raise ValueError(f"Unknown view {view!r}; one of {CMR_VIEWS}")
    path = (_DATA_ROOT / "raw" / "cmrxrecon" / "ChallengeData" / "SingleCoil" / "Cine"
            / "TrainingSet" / "FullSample" / f"P{patient}" / f"cine_{view}.mat")
    with h5py.File(path, "r") as f:
        k = f["kspace_single_full"][:, slices]  # [T, D, H, W] of (real, imag) records
    return torch.from_numpy(np.stack([k["real"], k["imag"]], axis=1))


def complex_abs(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Modelled after fastmri.complex_abs but includes eps
    if not x.shape[1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")
    x = (x**2).sum(dim=1, keepdim=True)
    return (x + eps).sqrt()

def real_abs(x:torch.Tensor) -> torch.Tensor:
    return x.abs()


@dataclass
class DatasetSpec:
    """One dataset family's loader + forward-operator factory, plus capability flags that
    replace ad hoc `'cmr' in config.dataset` string checks elsewhere in the pipeline
    (registration.py's joint recon+motion optimizer, losses.py's complex-image handling)."""
    load: Callable[[argparse.Namespace], tuple]
    operators: Callable[[argparse.Namespace, torch.Tensor], tuple]
    is_complex_img: bool
    joint_recon: bool


def _cmr_load(config: argparse.Namespace) -> tuple:
    view = getattr(config, 'view', 'sax')
    # The processed _norm files (and so cmr_test2) exist for the SAX stack only.
    if view != 'sax' and (config.dataset == 'cmr_test2' or not getattr(config, 'cmr_raw', False)):
        raise ValueError(f"view={view!r} reads the challenge .mat: needs a cmr_P* dataset with "
                         f"cmr_raw=true (got dataset={config.dataset!r}, "
                         f"cmr_raw={getattr(config, 'cmr_raw', False)})")
    if config.dataset == 'cmr_test2':
        # Fully-sampled, unmasked "no k-space undersampling" mode used by the staged
        # ablation (stage S0-S2), on P001's slices and frames as the real cmr datasets.
        gt_kspace_data = torch.load(_cmr('training_p001_single_coil_full_cine_sax_norm.pt'))[:, config.slices].permute(0, 4, 1, 2, 3)
        gt_kspace_data = gt_kspace_data[config.start_frame:config.start_frame + config.time_points]
        kspace_mask = torch.ones_like(gt_kspace_data, dtype=bool)
        raw_kspace_data = gt_kspace_data
    else:
        patient = config.dataset.split('_')[1][1:]
        if getattr(config, 'cmr_raw', False):
            gt_kspace_data = _cmr_challenge_slice(patient, config.slices, view)
        else:
            gt_kspace_data = torch.load(_cmr(f'training_p{patient}_single_coil_full_cine_sax_norm.pt'))[:,config.slices].permute(0, 4, 1, 2, 3)
        gt_kspace_data = gt_kspace_data[config.start_frame:config.start_frame + config.time_points]
        kspace_mask = get_kspace_mask(config, gt_kspace_data, config.factor)
        raw_kspace_data = torch.zeros_like(gt_kspace_data)
        raw_kspace_data[kspace_mask] = gt_kspace_data[kspace_mask]
    # Keep the first `template_frames` frames fully sampled (frozen GT template): full k-space
    # for those frames feeds both the nmAPG init and the joint-opt data consistency. Only
    # applies to the generated-mask (undersampled) datasets; cmr_test2 is fully sampled anyway.
    n = getattr(config, 'template_frames', 0)
    if n > 0 and config.dataset != 'cmr_test2':
        kspace_mask[:n] = True
        raw_kspace_data = torch.zeros_like(gt_kspace_data)
        raw_kspace_data[kspace_mask] = gt_kspace_data[kspace_mask]
    return raw_kspace_data, gt_kspace_data, kspace_mask


def _cmr_operators(config: argparse.Namespace, mask: torch.Tensor) -> tuple:
    if config.dataset == 'cmr_test2':
        full_forw = FastmriFT()
        full_adj = FastmriIFT()
        forw_subs = FastmriFT()
        forw_subs_adj = FastmriIFT()
    else:
        full_forw = FastmriFT()
        full_adj = FastmriIFT()
        forw_subs = MaskedFT(mask)
        forw_subs_adj = MaskedIFT(mask)
    return full_forw, full_adj, forw_subs, forw_subs_adj


DATASET_FAMILIES: dict[str, DatasetSpec] = {
    'cmr': DatasetSpec(load=_cmr_load, operators=_cmr_operators,
                       is_complex_img=True, joint_recon=True),
    # 'cbct': DatasetSpec(load=spare_data.get_data, operators=spare_data.get_operators,
    #                     is_complex_img=False, joint_recon=True),
}


def _get_family(config: argparse.Namespace) -> DatasetSpec:
    name = config.dataset.split('_')[0]
    if name not in DATASET_FAMILIES:
        raise ValueError(f"Unknown dataset family {name!r} for dataset={config.dataset!r}; "
                         f"registered families: {sorted(DATASET_FAMILIES)}")
    return DATASET_FAMILIES[name]


def get_data(config: argparse.Namespace) -> list[torch.Tensor]:
    return _get_family(config).load(config)

def get_operators(config: argparse.Namespace, mask: torch.Tensor) -> list[nn.Module]:
    return _get_family(config).operators(config, mask)

def get_dataset_capabilities(config: argparse.Namespace) -> DatasetSpec:
    return _get_family(config)

def get_init(config: argparse.Namespace, raw_kspace_data: torch.Tensor, gt_im: torch.Tensor, fixed_adj: torch.Tensor, kspace_mask: torch.Tensor = None) -> nn.Parameter:
    if config.init == 'zero':
        # Shaped from gt_im (image domain), not raw_kspace_data: those coincide for MRI
        # (FFT preserves shape) but not for e.g. CBCT, where raw data lives in the
        # sinogram domain and has a different shape than the image being initialized.
        recon_init = nn.Parameter(torch.zeros_like(gt_im, device=config.device), requires_grad=True)
    elif config.init == 'gt':
        recon_init = nn.Parameter(gt_im.clone(), requires_grad=True)
    elif config.init == 'rand':
        recon_init = torch.nn.Parameter(torch.randn_like(gt_im, device=config.device)*1e-3, requires_grad=True)
    elif config.init == 'adj':
        recon_init = torch.nn.Parameter(fixed_adj.clone(), requires_grad=True)
    elif config.init == 'ktavg':
        # k-t shared init: fill each unmeasured k-space entry with the average of the
        # frames that measured it (motion-free temporal sharing). Requires a time-varying
        # (kt) mask to be useful; with a static mask this reduces to the zero-filled adj.
        counts = kspace_mask.sum(0, keepdim=True).clamp(min=1)
        avg = raw_kspace_data.sum(0, keepdim=True) / counts
        filled = torch.where(kspace_mask, raw_kspace_data, avg.expand_as(raw_kspace_data))
        recon_init = nn.Parameter(FastmriIFT()(filled).to(config.device), requires_grad=True)
    elif config.init == 'template':
        # Every frame starts as a copy of the template image (frame 0's GT). The frozen
        # template keeps its GT; non-template frames begin static and are moved by the flow.
        recon_init = nn.Parameter(gt_im[0:1].expand_as(gt_im).clone().to(config.device), requires_grad=True)
    return recon_init

    
def _acs_rows(h: int, acs: int) -> set[int]:
    """The `acs` fully-sampled centre rows (auto-calibration lines) around k = 0, which is row h//2.
    acs=24 is rows h//2-12 .. h//2+11, the challenge's ACS block."""
    start = h // 2 - acs // 2
    return set(range(start, start + acs))


def _rows_mask(shape: torch.Size, rows) -> torch.Tensor:
    """A [..., H, W] bool mask sampling the given rows (phase-encode lines) fully along W."""
    mask = torch.zeros(shape, dtype=bool)
    mask[..., list(rows), :] = 1
    return mask


def generate_standard_mask(shape: torch.Size, factor: int, acs: int = 24):
    # Equispaced rows 0, f, 2f, ... plus the ACS, the same in every frame. At acs=24 this is the
    # challenge's AccFactor04/08/10 mask (checked on P001 SAX and LAX): 69 / 47 / 42 of 204 rows,
    # so the effective acceleration is only 2.96 / 4.34 / 4.86.
    h = shape[-2]
    return _rows_mask(shape, _acs_rows(h, acs) | set(range(0, h, factor)))

def generate_random_mask(shape: torch.Size, factor: int, acs: int = 24, seed: int | None = None):
    # h//factor uniformly random rows (they may fall inside the ACS) plus the ACS, the same in
    # every frame. seed=None draws from the global `random` state.
    h = shape[-2]
    rng = random if seed is None else random.Random(seed)
    return _rows_mask(shape, _acs_rows(h, acs) | set(rng.sample(range(h), h // factor)))

def generate_random_mask2(shape: torch.Size, factor: int, acs: int = 24, seed: int | None = None):
    # Every row is kept with probability 1/factor, plus the ACS, the same in every frame.
    # seed=None draws from the global torch RNG.
    h = shape[-2]
    gen = None if seed is None else torch.Generator().manual_seed(seed)
    keep = torch.rand(h, generator=gen) < 1 / factor
    return _rows_mask(shape, _acs_rows(h, acs) | set(np.arange(h)[keep.numpy()].tolist()))

def generate_kt_mask(shape: torch.Size, factor: int, acs: int = 24):
    # Time-interleaved (k-t) sampling: every frame keeps the ACS center rows, but the
    # equispaced rows are shifted by one per frame. Consecutive frames then measure
    # complementary k-space rows, so temporal consistency + motion can actually fill in
    # the rows a frame is missing, and the aliasing ghosts are no longer static in time.
    t_dim, h = shape[0], shape[-2]
    acs_rows = _acs_rows(h, acs)
    mask = torch.zeros(shape, dtype=bool)
    for t in range(t_dim):
        mask[t, ..., list(acs_rows | set(range(t % factor, h, factor))), :] = 1
    return mask

def generate_kt_random_mask(shape: torch.Size, factor: int, acs: int = 24, seed: int | None = None):
    # Like generate_kt_mask but with an independent random row selection per frame.
    # seed=None draws from the global `random` state.
    t_dim, h = shape[0], shape[-2]
    rng = random if seed is None else random.Random(seed)
    acs_rows = _acs_rows(h, acs)
    mask = torch.zeros(shape, dtype=bool)
    for t in range(t_dim):
        mask[t, ..., list(acs_rows | set(rng.sample(range(h), h // factor))), :] = 1
    return mask


def _matched_row_count(h: int, factor: int, acs_rows: set[int]) -> int:
    """Rows outside the ACS that generate_standard_mask samples at this factor: a variable-density
    mask with this many extra rows measures exactly as many rows as the equispaced one, so the
    two compare at the same effective acceleration."""
    return len(set(range(0, h, factor)) - acs_rows)

def _vd_rows(h: int, n: int, acs_rows: set[int], sigma: float,
             gen: torch.Generator | None) -> set[int]:
    """n distinct rows outside the ACS, drawn with probability ~ exp(-k^2 / (2 (sigma h)^2)), k the
    row's distance from k = 0 (row h//2). Float64 weights, so a narrow sigma does not underflow
    the outer rows to probability 0."""
    free = torch.tensor(sorted(set(range(h)) - acs_rows))
    if n == 0 or free.numel() == 0:
        return set()
    weight = torch.exp(-0.5 * ((free.double() - h // 2) / (sigma * h)) ** 2)
    return set(free[torch.multinomial(weight, n, replacement=False, generator=gen)].tolist())

def generate_vd_mask(shape: torch.Size, factor: int, acs: int = 24, sigma: float = 0.25,
                     seed: int | None = None):
    # Variable-density random rows (Gaussian in k_y around k = 0) plus the ACS, the same in every
    # frame, with exactly as many rows as generate_standard_mask at the same factor and acs.
    # seed=None draws from the global torch RNG.
    h = shape[-2]
    gen = None if seed is None else torch.Generator().manual_seed(seed)
    acs_rows = _acs_rows(h, acs)
    n = _matched_row_count(h, factor, acs_rows)
    return _rows_mask(shape, acs_rows | _vd_rows(h, n, acs_rows, sigma, gen))

def generate_kt_vd_mask(shape: torch.Size, factor: int, acs: int = 24, sigma: float = 0.25,
                        seed: int | None = None):
    # generate_vd_mask with an independent draw per frame (k-t variable density): every frame
    # measures as many rows as the equispaced mask, but different ones away from the ACS.
    t_dim, h = shape[0], shape[-2]
    gen = None if seed is None else torch.Generator().manual_seed(seed)
    acs_rows = _acs_rows(h, acs)
    n = _matched_row_count(h, factor, acs_rows)
    mask = torch.zeros(shape, dtype=bool)
    for t in range(t_dim):
        mask[t, ..., list(acs_rows | _vd_rows(h, n, acs_rows, sigma, gen)), :] = 1
    return mask


MASKS = ('st', 'random', 'random2', 'kt', 'kt_random', 'vd', 'kt_vd')

def get_kspace_mask(config, kspace_data, factor):
    """The sampling mask config.mask (one of MASKS), [T, 2, D, H, W] bool; rows = H = phase encode.

    The random masks draw from their own RNG seeded with config.mask_seed, not from the global
    state. So the mask does not change with config.seed (repeats with other seeds stay paired on
    the same mask) or with whatever drew random numbers before, and every entry point (pipeline,
    initsweep, tune) builds the same mask from the same config.
    """
    shape, kind = kspace_data.shape, config.mask
    acs = getattr(config, 'acs_lines', 24)
    seed = getattr(config, 'mask_seed', 0)
    sigma = getattr(config, 'vd_sigma', 0.25)
    if kind == 'st':
        return generate_standard_mask(shape, factor, acs)
    if kind == 'kt':
        return generate_kt_mask(shape, factor, acs)
    if kind == 'random':
        return generate_random_mask(shape, factor, acs, seed)
    if kind == 'random2':
        return generate_random_mask2(shape, factor, acs, seed)
    if kind == 'kt_random':
        return generate_kt_random_mask(shape, factor, acs, seed)
    if kind == 'vd':
        return generate_vd_mask(shape, factor, acs, sigma, seed)
    if kind == 'kt_vd':
        return generate_kt_vd_mask(shape, factor, acs, sigma, seed)
    # An unknown name used to fall through to the standard mask silently.
    raise ValueError(f"Unknown mask {kind!r}; one of {MASKS}")
