import argparse
import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from . import spare_data
from .fft_utils import FastmriFT, FastmriIFT, MaskedFT, MaskedIFT


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
    gt_kspace_data = torch.load('data/processed/cmrxrecon/test/training_p001_single_coil_full_cine_sax_norm.pt')[:2,0].permute(0, 3, 1, 2)
    if config.dataset == 'cmr_test1':
        raw_kspace_data = torch.load('data/processed/cmrxrecon/test/training_p001_single_coil_acc_04_cine_sax_norm.pt')[:2,0].permute(0, 3, 1, 2)
        kspace_mask = (raw_kspace_data[:2] != 0)
    elif config.dataset == 'cmr_test2':
        # Fully-sampled, unmasked "no k-space undersampling" mode used by the staged
        # ablation (stage S0-S2). Reload from the full file honoring slice/frame selection
        # so it can run at the same time_points as the real cmr datasets; the module-level
        # gt_kspace_data above is pinned to [:2, 0] (kept for cmr_test1).
        gt_kspace_data = torch.load('data/processed/cmrxrecon/test/training_p001_single_coil_full_cine_sax_norm.pt')[:, config.slice_number].permute(0, 3, 1, 2)
        gt_kspace_data = gt_kspace_data[config.start_frame:config.start_frame + config.time_points]
        kspace_mask = torch.ones_like(gt_kspace_data, dtype=bool)
        raw_kspace_data = gt_kspace_data
    elif config.dataset == 'cmr_test3':
        kspace_mask = get_kspace_mask(config, gt_kspace_data, config.factor)
        raw_kspace_data = torch.zeros_like(gt_kspace_data)
        raw_kspace_data[kspace_mask] = gt_kspace_data[kspace_mask]
    else:
        patient = config.dataset.split('_')[1][1:]
        gt_kspace_data = torch.load(f'data/processed/cmrxrecon/test/training_p{patient}_single_coil_full_cine_sax_norm.pt')[:,config.slice_number].permute(0, 3, 1, 2)
        gt_kspace_data = gt_kspace_data[config.start_frame:config.start_frame + config.time_points]
        kspace_mask = get_kspace_mask(config, gt_kspace_data, config.factor)
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
    'cbct': DatasetSpec(load=spare_data.get_data, operators=spare_data.get_operators,
                        is_complex_img=False, joint_recon=True),
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
    return recon_init

    
def generate_standard_mask(shape: torch.Size, factor: int):
    h = shape[-2]
    start = h//2 - 12
    end = h//2 + 12

    set1 = set(range(start, end))
    set2 = set(range(0, h, factor))
    indices = list(set1.union(set2))

    mask = torch.zeros(shape, dtype=bool)
    mask[..., indices, :] = 1

    return mask

def generate_random_mask(shape: torch.Size, factor: int):
    h = shape[-2]
    start = h//2 - 12
    end = h//2 + 12

    set1 = set(range(start, end))
    set2 = set(random.sample(range(h), h//factor))
    indices = list(set1.union(set2))

    mask = torch.zeros(shape, dtype=bool)
    mask[..., indices, :] = 1

    return mask

def generate_random_mask2(shape: torch.Size, factor: int):
    rate = 1 / factor
    h = shape[-2]
    start = h//2 - 12
    end = h//2 + 12

    set1 = set(range(start, end))
    mask = torch.rand(h) < rate
    set2 = set(np.arange(h)[mask])
    indices = list(set1.union(set2))

    mask = torch.zeros(shape, dtype=bool)
    mask[..., indices, :] = 1

    return mask

def generate_kt_mask(shape: torch.Size, factor: int):
    # Time-interleaved (k-t) sampling: every frame keeps the ACS center rows, but the
    # equispaced rows are shifted by one per frame. Consecutive frames then measure
    # complementary k-space rows, so temporal consistency + motion can actually fill in
    # the rows a frame is missing, and the aliasing ghosts are no longer static in time.
    t_dim, h = shape[0], shape[-2]
    start = h//2 - 12
    end = h//2 + 12

    mask = torch.zeros(shape, dtype=bool)
    for t in range(t_dim):
        indices = list(set(range(start, end)).union(set(range(t % factor, h, factor))))
        mask[t, ..., indices, :] = 1

    return mask

def generate_kt_random_mask(shape: torch.Size, factor: int):
    # Like generate_kt_mask but with an independent random row selection per frame.
    t_dim, h = shape[0], shape[-2]
    start = h//2 - 12
    end = h//2 + 12

    mask = torch.zeros(shape, dtype=bool)
    for t in range(t_dim):
        indices = list(set(range(start, end)).union(set(random.sample(range(h), h//factor))))
        mask[t, ..., indices, :] = 1

    return mask

def get_kspace_mask(config, kspace_data, factor):
    if config.mask == 'random':
        mask = generate_random_mask(kspace_data.shape, factor)
    elif config.mask == 'random2':
        mask = generate_random_mask2(kspace_data.shape, factor)
    elif config.mask == 'kt':
        mask = generate_kt_mask(kspace_data.shape, factor)
    elif config.mask == 'kt_random':
        mask = generate_kt_random_mask(kspace_data.shape, factor)
    else:
        mask = generate_standard_mask(kspace_data.shape, factor)
    return mask