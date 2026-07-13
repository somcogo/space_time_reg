from argparse import Namespace
from logging import Logger

import torch
from torch import nn

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
        fixed = raw_kspace_data.to(config.device)

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

        full_forw, full_adj, forw_subs, forw_subs_adj = get_operators(config, kspace_mask)
        forw_subs = forw_subs.to(config.device)
        forw_subs_adj = forw_subs_adj.to(config.device)

        # fixed keeps the full k-space shape, zero-filled at unmeasured entries. This
        # supports time-varying (k-t) masks where the number of measured rows differs
        # between frames, which the old row-subsampled representation could not express.
        fixed = raw_kspace_data.to(config.device)

        gt_im = full_adj(gt_kspace_data)
        gt_im = gt_im.to(config.device)
        init = get_init(config, raw_kspace_data, gt_im, forw_subs_adj(fixed).to(config.device), kspace_mask)

        if config.init_skip:
            recon, init_metrics = init.detach(), None
        elif config.use_nmapg:
            # nmAPG optimizes frames as batch items and drops converged ones, indexing
            # x[idx] and y[idx] but not the forward operator. Packing the mask into y as
            # extra channels keeps each frame's mask aligned with its measurements under
            # that subsetting; the data-fit functions unpack it again.
            packed_measurements = torch.cat([fixed, kspace_mask.to(fixed)], dim=1)
            recon, init_metrics = init_using_nmAPG(config, init, packed_measurements, full_forw, forw_subs_adj, logger)
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