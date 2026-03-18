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

def load_and_prepare_cmrxrecon(file_name):
    hf_m = h5py.File(file_name)
    newvalue = hf_m[list(hf_m.keys())[0]]
    fullmulti = newvalue["real"] + 1j*newvalue["imag"]
    fullmulti_t = T.to_tensor(fullmulti)
    return fullmulti_t

def get_syn_inputs(config):
    imgs = torch.from_numpy(np.load(f'data/syn/{config.dataset}/{config.dataset}.npy').transpose((2, 0, 1)))
    st_dicts = torch.load(f'data/syn/{config.dataset}/{config.dataset}_nrep_st_dicts.pt')
    segs = None
    models = []
    for i in range(len(st_dicts)):
        model = modules.SingleBVPNet(type='sine', mode='mlp', sidelength=imgs.shape[1:], device=config.device)
        model.load_state_dict(st_dicts[i])
        model.eval()
        models.append(model)
    return imgs, models, segs

def get_syn_test_inputs(config):
    imgs = torch.from_numpy(np.load(f'data/syn/rot_slow2/rot_slow2.npy').transpose((2, 0, 1)))
    st_dicts = torch.load(f'data/syn/rot_slow2/rot_slow2_nrep_st_dicts.pt')
    segs = None
    models = []
    for i in range(len(st_dicts)):
        model = modules.SingleBVPNet(type='sine', mode='mlp', sidelength=imgs.shape[1:], device=config.device)
        model.load_state_dict(st_dicts[i])
        model.eval()
        models.append(model)
    imgs = imgs[[0,5]]
    models = [models[0]]
    return imgs, models, segs

def get_large_rotslow_inputs():
    imgs = torch.from_numpy(np.load(f'data/syn/rot_slow2/rot_slow2.npy').transpose((2, 0, 1)))
    st_dicts = torch.load(f'data/syn/rot_slow2/rot_slow2_nrep_st_dicts_large.pt')
    segs = None
    models = []
    model = Siren([2, 256, 256, 256, 1])
    model.load_state_dict(st_dicts)
    model.eval()
    models = [model]
    return imgs, models, segs

def get_rot_slow2_64_inputs():
    imgs = torch.from_numpy(np.load(f'data/syn/rot_slow2/rot_slow2.npy').transpose((2, 0, 1)))
    st_dicts = torch.load(f'data/syn/rot_slow2/rot_slow2_nrep_st_dicts_64x64x64.pt')
    segs = None
    model = Siren([2, 64, 64, 64, 1])
    model.load_state_dict(st_dicts)
    model.eval()
    models = [model]
    return imgs, models, segs

def get_mouse_input():
    imgs = torch.from_numpy(np.load(f'data/cell_tracking/GFP-GOWT1_mouse_stem_small.npy').transpose((2, 0, 1))).float()
    st_dicts = torch.load(f'data/cell_tracking/GFP-GOWT1_mouse_stem_small_nrep.npy')[0]
    segs = torch.from_numpy(np.load(f'data/cell_tracking/GFP-GOWT1_mouse_stem_st_seg_small.npy').transpose((2, 0, 1)))
    model = Siren([2, 256, 256, 256, 1])
    model.load_state_dict(st_dicts)
    model.eval()
    models = [model]
    return imgs, models, segs

def get_mouse_large_input():
    imgs = torch.from_numpy(np.load(f'data/cell_tracking/GFP-GOWT1_mouse_stem.npy').transpose((2, 0, 1))).float()
    segs = torch.from_numpy(np.load(f'data/cell_tracking/GFP-GOWT1_mouse_stem_st_seg.npy').transpose((2, 0, 1)))
    models = [None]
    return imgs, models, segs

def get_mouse_corner_input():
    imgs = torch.from_numpy(np.load(f'data/cell_tracking/GFP-GOWT1_mouse_stem_corner.npy').transpose((2, 0, 1))).float()
    segs = torch.from_numpy(np.load(f'data/cell_tracking/GFP-GOWT1_mouse_stem_corner_seg.npy').transpose((2, 0, 1)))
    models = [None]
    return imgs, models, segs

def get_lung_test_input():
    imgs = torch.from_numpy(np.load(f'data/4D-Lung/first_try.npy'))[:, 25]
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
    segs = None
    models = [None]
    return imgs, models, segs

def get_lung_test_3d_input():
    imgs = torch.from_numpy(np.load(f'data/4D-Lung/first_try.npy'))[[0, 5], :48].permute(0, 2, 3, 1)
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
    segs = None
    models = [None]
    return imgs, models, segs

def get_oasis1_input():
    im1 = nib.load('data/oasis_example/OAS1_0001_MR1/brain.nii.gz').get_fdata()
    seg1 = nib.load('data/oasis_example/OAS1_0001_MR1/brain_aseg.nii.gz').get_fdata()
    im2 = nib.load('data/oasis_example/OAS1_0002_MR1/brain.nii.gz').get_fdata()
    seg2 = nib.load('data/oasis_example/OAS1_0002_MR1/brain_aseg.nii.gz').get_fdata()
    imgs = torch.from_numpy(np.stack([im1, im2], axis=0)).float()
    segs = torch.from_numpy(np.stack([seg1, seg2], axis=0))
    models = [None]
    return imgs, models, segs

def get_oasis2_input():
    im1 = nib.load('data/oasis_example/OAS1_0001_MR1/brain.nii.gz').get_fdata()
    seg1 = nib.load('data/oasis_example/OAS1_0001_MR1/brain_aseg.nii.gz').get_fdata()
    im2 = nib.load('data/oasis_example/OAS1_0001_MR1/brain.nii.gz').get_fdata()
    seg2 = nib.load('data/oasis_example/OAS1_0001_MR1/brain_aseg.nii.gz').get_fdata()
    imgs = torch.from_numpy(np.stack([im1, im2], axis=0)).float()
    segs = torch.from_numpy(np.stack([seg1, seg2], axis=0))
    models = [None]
    return imgs, models, segs

def get_oasis3_input():
    im1 = nib.load('data/oasis_example/OAS1_0002_MR1/brain.nii.gz').get_fdata()
    seg1 = nib.load('data/oasis_example/OAS1_0002_MR1/brain_aseg.nii.gz').get_fdata()
    im2 = nib.load('data/oasis_example/OAS1_0001_MR1/brain.nii.gz').get_fdata()
    seg2 = nib.load('data/oasis_example/OAS1_0001_MR1/brain_aseg.nii.gz').get_fdata()
    imgs = torch.from_numpy(np.stack([im1, im2], axis=0)).float()
    segs = torch.from_numpy(np.stack([seg1, seg2], axis=0))
    models = [None]
    return imgs, models, segs

def prepare_non_inverse_case(config):
    if config.dataset in ['easy', 'hard', 'rectri', 'rot', 'rot_slow', 'rot_slow2', 'rec']:
        imgs, models, segs = get_syn_inputs(config)
    elif config.dataset == 'syn_test':
        imgs, models, segs = get_syn_test_inputs(config)
    elif config.dataset == 'rot_slow2_large':
        imgs, models, segs = get_large_rotslow_inputs()
    elif config.dataset == 'rot_slow2_64':
        imgs, models, segs = get_rot_slow2_64_inputs()
    elif config.dataset == 'mouse':
        imgs, models, segs = get_mouse_input()
    elif config.dataset == 'mouse_large':
        imgs, models, segs = get_mouse_large_input()
    elif config.dataset == 'mouse_corner':
        imgs, models, segs = get_mouse_corner_input()
    elif config.dataset == 'lung_test':
        imgs, models, segs = get_lung_test_input()
    elif config.dataset == 'lung_test_3d':
        imgs, models, segs = get_lung_test_3d_input()
    elif config.dataset == 'oasis_examplev1':
        imgs, models, segs = get_oasis1_input()
    elif config.dataset == 'oasis_examplev2':
        imgs, models, segs = get_oasis3_input()
    elif config.dataset == 'oasis_examplev3':
        imgs, models, segs = get_oasis3_input()

    imgs = imgs[config.start_frame:config.start_frame + config.time_points]
    models = models[config.start_frame:config.start_frame + config.time_points]
    segs = segs[config.start_frame:config.start_frame + config.time_points] if segs is not None else None

    moving = imgs[config.tm].unsqueeze(0)
    moving_inr = models[config.tm]
    fixed = imgs.unsqueeze(1)
    seg_moving = segs[config.tm:config.tm+1].expand(segs.shape).unsqueeze(1).float() if segs is not None else None
    seg_fixed = segs
    forward_method = nn.Identity()
    gt_im = imgs.unsqueeze(1)
    recon = imgs.unsqueeze(1)

    moving = moving.to(config.device)
    moving_inr = moving_inr.to(config.device)
    fixed = fixed.to(config.device)
    forward_method = forward_method.to(config.device)
    return [moving, moving_inr, fixed, forward_method], [recon, gt_im, seg_moving, seg_fixed]

def prepare_inverse_case(config: Namespace, logger: Logger) -> list[list]:
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
        
    fixed = fastmri.complex_abs_sq(fixed.movedim(1,-1)).unsqueeze(1)
    fixed = (fixed + 1e-8).sqrt()

    recon = recon.detach()
    moving = nn.Parameter(recon[config.tm].clone())

    
    moving_inr = None
    seg_moving = None
    seg_fixed = None

    moving = moving.to(config.device)
    fixed = fixed.to(config.device)
    forw_subs = forw_subs.to(config.device)
    return [moving, moving_inr, fixed, forw_subs], [recon, gt_im, seg_moving, seg_fixed]

def prepare_inputs(config: Namespace, logger: Logger) -> list:
    if 'cmr' in config.dataset:
        inputs, eval_inputs = prepare_inverse_case(config, logger)
    else:
        inputs, eval_inputs = prepare_non_inverse_case(config)
    
    time_points = torch.linspace(0, 1, config.time_points, device=config.device)
    config.func_kwargs['time_points'] = time_points
    inputs.append(time_points)
    return inputs, eval_inputs