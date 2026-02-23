import random
import argparse

import numpy as np
import torch
from torch import nn

from .fft_utils import FastmriFT, FastmriIFT, FTAndSubsample, ZeroFillAndIFT


def get_data(config: argparse.Namespace) -> list[torch.Tensor]:
    # TODO: rewrite with option for other patients, downsamlpming factors, start frame and # of frames
    gt_kspace_data = torch.load(f'data/processed/cmrxrecon/test/training_p001_single_coil_full_cine_sax_norm.pt')[:2,0].permute(0, 3, 1, 2)
    if config.dataset == 'cmr_test1':
        raw_kspace_data = torch.load(f'data/processed/cmrxrecon/test/training_p001_single_coil_acc_04_cine_sax_norm.pt')[:2,0].permute(0, 3, 1, 2)
        kspace_mask = (raw_kspace_data[:2] != 0)
    elif config.dataset == 'cmr_test2':
        kspace_mask = torch.ones_like(gt_kspace_data, dtype=bool)
        raw_kspace_data = gt_kspace_data
    elif config.dataset == 'cmr_test3':
        kspace_mask = get_kspace_mask(config, gt_kspace_data, config.factor)
        raw_kspace_data = torch.zeros_like(gt_kspace_data)
        raw_kspace_data[kspace_mask] = gt_kspace_data[kspace_mask]
    elif 'cmr' in config.dataset:
        patient = config.dataset.split('_')[1][1:]        
        raw_kspace_data = torch.load(f'data/processed/cmrxrecon/test/training_p{patient}_single_coil_acc_04_cine_sax_norm.pt')[:,config.slice_number].permute(0, 3, 1, 2)
        raw_kspace_data = raw_kspace_data[config.start_frame:config.start_frame + config.time_points]
        gt_kspace_data = torch.load(f'data/processed/cmrxrecon/test/training_p{patient}_single_coil_full_cine_sax_norm.pt')[:,config.slice_number].permute(0, 3, 1, 2)
        gt_kspace_data = gt_kspace_data[config.start_frame:config.start_frame + config.time_points]
        kspace_mask = get_kspace_mask(config, raw_kspace_data, config.factor)
    return raw_kspace_data, gt_kspace_data, kspace_mask

def get_operators(config: argparse.Namespace, mask: torch.Tensor) -> list[nn.Module]:
    if config.dataset == 'cmr_test2':
        full_forw = FastmriFT()
        full_adj = FastmriIFT()
        forw_subs = FastmriFT()
        forw_subs_adj = FastmriIFT()
    else:
        full_forw = FastmriFT()
        full_adj = FastmriIFT()
        forw_subs = FTAndSubsample(mask)
        forw_subs_adj = ZeroFillAndIFT(mask)
    return full_forw, full_adj, forw_subs, forw_subs_adj

def get_init(config: argparse.Namespace, raw_kspace_data: torch.Tensor, gt_im: torch.Tensor) -> nn.Parameter:
    if config.init == 'zero':
        recon_init = nn.Parameter(torch.zeros_like(raw_kspace_data, device=config.device), requires_grad=True)
    elif config.init == 'gt':
        recon_init = nn.Parameter(gt_im.clone(), requires_grad=True)
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
    mask = torch.rand((h)) < rate
    set2 = set(np.arange(h)[mask])
    indices = list(set1.union(set2))

    mask = torch.zeros(shape, dtype=bool)
    mask[..., indices, :] = 1

    return mask

def get_kspace_mask(config, kspace_data, factor):
    if config.mask == 'random':
        mask = generate_random_mask(kspace_data.shape, factor)
    elif config.mask == 'random2':
        mask = generate_random_mask2(kspace_data.shape, factor)
    else:
        mask = generate_standard_mask(kspace_data.shape, factor)
    return mask