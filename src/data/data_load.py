from argparse import Namespace
from logging import Logger

import torch
import numpy as np
import nibabel as nib
from torch import nn
import h5py
import fastmri
from fastmri.data import transforms as T

from src.models.siren import Siren
from src.siren import modules
from .data_utils import get_data, get_operators, get_init
from .recon_init import init_using_nmAPG, init_with_grad_desc

def prepare_inverse_case(config: Namespace, logger: Logger) -> list[list]:
    if config.dataset == 'cmr_gt_test':
        config.dataset = 'cmr_P001'
        raw_kspace_data, gt_kspace_data, kspace_mask = get_data(config)
        full_forw, full_adj, forw_subs, forw_subs_adj = get_operators(config, kspace_mask)
        # smaller_shape = list(raw_kspace_data.shape[:2]) + [-1] + list(raw_kspace_data.shape[3:])
        # fixed = raw_kspace_data[kspace_mask.expand(raw_kspace_data.shape)].reshape(smaller_shape)
        # fixed = fixed.to(config.device)

        gt_im = full_adj(gt_kspace_data)
        gt_im = gt_im.to(config.device)
        full_forw, full_adj, forw_subs, forw_subs_adj = nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity()
        fixed = gt_im

        recon, init_metrics = gt_im, None
            
        # fixed = fastmri.complex_abs_sq(fixed.movedim(1,-1)).unsqueeze(1)
        # fixed = (fixed + 1e-8).sqrt()

        recon = recon.detach()
        moving = nn.Parameter(recon.clone())
    elif config.dataset == 'cmr_zerotest':
        fixed = torch.ones((2, 2, 204, 512), device=config.device, dtype=torch.float) * 0.09
        gt_im = torch.ones((2, 2, 204, 512), device=config.device, dtype=torch.float) * 0.09
        init = 2 * torch.ones((2, 2, 204, 512), device=config.device, dtype=torch.float) * 0.09
        full_forw, full_adj, forw_subs, forw_subs_adj = nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity()
        recon, init_metrics = init, None
        recon = recon.detach()
        moving = nn.Parameter(recon.clone())
    elif config.dataset == 'cmr_init_factor1':
        config.dataset = 'cmr_P001'
        raw_kspace_data, gt_kspace_data, kspace_mask = get_data(config)
        full_forw, full_adj, forw_subs, forw_subs_adj = get_operators(config, kspace_mask)
        smaller_shape = list(raw_kspace_data.shape[:2]) + [-1] + list(raw_kspace_data.shape[3:])
        fixed = raw_kspace_data[kspace_mask.expand(raw_kspace_data.shape)].reshape(smaller_shape)
        fixed = fixed.to(config.device)

        gt_im = full_adj(gt_kspace_data)
        gt_im = gt_im.to(config.device)
        init = torch.load('data/cmr_p001_init.pt')
        init = nn.Parameter(init).to(config.device)

        recon, init_metrics = init, None
            
        # fixed = fastmri.complex_abs_sq(fixed.movedim(1,-1)).unsqueeze(1)
        # fixed = (fixed + 1e-8).sqrt()

        recon = recon.detach()
        moving = nn.Parameter(recon.clone())
    elif config.dataset == 'cmr_init_noft':
        config.dataset = 'cmr_P001'
        raw_kspace_data, gt_kspace_data, kspace_mask = get_data(config)
        full_forw, full_adj, forw_subs, forw_subs_adj = get_operators(config, kspace_mask)

        gt_im = full_adj(gt_kspace_data)
        gt_im = gt_im.to(config.device)
        init = torch.load('data/cmr_p001_init.pt')
        init = nn.Parameter(init).to(config.device)
        fixed = gt_im

        recon, init_metrics = init, None
        full_forw, full_adj, forw_subs, forw_subs_adj = nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity()
            
        # fixed = fastmri.complex_abs_sq(fixed.movedim(1,-1)).unsqueeze(1)
        # fixed = (fixed + 1e-8).sqrt()

        recon = recon.detach()
        moving = nn.Parameter(recon.clone())
    else:
        raw_kspace_data, gt_kspace_data, kspace_mask = get_data(config)
        
        # TODO: if init debug is done, redo with generated masks to be able to switch
        # kspace_mask = get_kspace_mask(config, raw_kspace_data, factor=4)
        # kspace_mask = (raw_kspace_data[:1] != 0)
        full_forw, full_adj, forw_subs, forw_subs_adj = get_operators(config, kspace_mask)

        smaller_shape = list(raw_kspace_data.shape[:2]) + [-1] + list(raw_kspace_data.shape[3:])
        fixed = raw_kspace_data[kspace_mask.expand(raw_kspace_data.shape)].reshape(smaller_shape)
        fixed = fixed.to(config.device)

        gt_im = full_adj(gt_kspace_data)
        gt_im = gt_im.to(config.device)
        init = get_init(config, raw_kspace_data, gt_im, forw_subs_adj(fixed))

        if config.use_nmapg:
            recon, init_metrics = init_using_nmAPG(config, init, fixed, forw_subs, forw_subs_adj, logger)
            # recon = torch.zeros_like(init)
            # for t in range(init.shape[0]):
            #     r, init_metrics = init_using_nmAPG(config, init[t:t+1], fixed[t:t+1], forw_subs, forw_subs_adj, logger)
            #     recon[t] = r.detach()
        else:
            recon, init_metrics = init_with_grad_desc(config, init, fixed, forw_subs)
            
        # fixed = fastmri.complex_abs_sq(fixed.movedim(1,-1)).unsqueeze(1)
        # fixed = (fixed + 1e-8).sqrt()

        recon = recon.detach()
        moving = nn.Parameter(recon.clone())
    
    moving_inr = None
    seg_moving = None
    seg_fixed = None

    moving = moving.to(config.device)
    fixed = fixed.to(config.device)
    forw_subs = forw_subs.to(config.device)
    return [moving, moving_inr, fixed, forw_subs], [recon, gt_im, seg_moving, seg_fixed]

def prepare_inputs(config: Namespace, logger: Logger) -> list:
    inputs, eval_inputs = prepare_inverse_case(config, logger)
    
    return inputs, eval_inputs