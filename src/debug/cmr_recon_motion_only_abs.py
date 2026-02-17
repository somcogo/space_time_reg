from argparse import Namespace
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import fastmri

from src.registration import registration
from src.models.factory import get_func
from src.data.data_load import prepare_inputs
from src.eval import evaluate
from src.metrics.calc_metrics import calculate_metrics
from src.utils.log_and_save import log_metrics
from src.metrics.visualisation import prep_vis_summary_pdf
from src.data.fft_utils import FastmriFT, FastmriIFT, FTAndSubsample, ZeroFillAndIFT
from src.data.data_utils import get_data, get_operators, get_init
from src.data.recon_init import init_using_nmAPG, init_with_grad_desc

torch.manual_seed(0)
random.seed(1)
np.random.seed(2)

def get_recon_init(config):
    gt_kspace_data = torch.load(f'data/processed/cmrxrecon/test/training_p001_single_coil_full_cine_sax_norm.pt')[:,0].permute(0, 3, 1, 2)
    raw_kspace_data = torch.load(f'data/processed/cmrxrecon/test/training_p001_single_coil_acc_04_cine_sax_norm.pt')[:,0].permute(0, 3, 1, 2)
    kspace_mask = (raw_kspace_data[:] != 0)
    
    full_forw = FastmriFT()
    full_adj = FastmriIFT()
    forw_subs = FTAndSubsample(kspace_mask)
    forw_subs_adj = ZeroFillAndIFT(kspace_mask)

    smaller_shape = list(raw_kspace_data.shape[:2]) + [-1] + list(raw_kspace_data.shape[3:])
    fixed = raw_kspace_data[kspace_mask.expand(raw_kspace_data.shape)].reshape(smaller_shape)
    fixed = fixed.to(config.device)

    gt_im = full_adj(gt_kspace_data)
    gt_im = gt_im.to(config.device)
    recon_init = torch.nn.Parameter(torch.zeros_like(raw_kspace_data, device=config.device), requires_grad=True)
    recon_init, metrics = init_with_grad_desc(config, recon_init, fixed, forw_subs)

    return recon_init.detach().cpu(), gt_im.detach().cpu(), metrics

def train(config):
    recon_init, _, _  = get_recon_init(config)
    gt_im = fastmri.complex_abs(recon_init.movedim(1, -1)).unsqueeze(1)

    forw = torch.nn.Identity()

    fixed = gt_im
    fixed = fixed.to(config.device)
    gt_im = gt_im.to(config.device)
    recon = gt_im
    moving = torch.nn.Parameter(gt_im[0].clone())

    
    moving_inr = None
    seg_moving = None
    seg_fixed = None

    moving = moving.to(config.device)
    fixed = fixed.to(config.device)
    forw = forw.to(config.device)

    inputs, eval_inputs = [moving, moving_inr, fixed, forw], [recon, gt_im, seg_moving, seg_fixed]
    time_points = torch.linspace(0, 1, config.time_points, device=config.device)
    config.func_kwargs['time_points'] = time_points
    inputs.append(time_points)


    func = get_func(config.func_name, config.func_kwargs)
    func = func.to(config.device)
    writer = SummaryWriter(log_dir=os.path.join(config.log_path, 'tensorboard'))
    logger = None
    


    out = registration(config=config,
                       writer=writer,
                       logger=logger,
                       inputs=inputs,
                       eval_inputs=eval_inputs,
                       func=func)

    model_outputs, loss_outputs, _, _, epoch, all_metrics, time_stamps, _ = out
    init_recon, gt_im, _, _ = eval_inputs
    abs_phi, rel_vel, _ = model_outputs
    loss_dict, _, moved_im = loss_outputs
    
    with torch.no_grad():
        metrics, imgs_to_save = calculate_metrics(config, None, inputs, eval_inputs, model_outputs, loss_outputs, extended_log=True)
        log_metrics(config, metrics, writer, epoch + 10, imgs_to_save, last_val=True)
        # prep_vis_summary_pdf(config, gt_im, init_recon, moved_im, abs_phi, rel_vel, all_metrics)

    print('-------------------------------------------------')
    print(f'Time spent (sec) over {config.epochs} iterations')
    print('-------------------------------------------------')
    print(f'Data loader:            {(time_stamps[0] - time_stamps[6]).sum():.4f}')
    print(f'ODE solver:             {(time_stamps[1] - time_stamps[0]).sum():.4f}')
    print(f'Loss calc:              {(time_stamps[2] - time_stamps[1]).sum():.4f}')
    print(f'Backprop:               {(time_stamps[3] - time_stamps[2]).sum():.4f}')
    print(f'Optim:                  {(time_stamps[4] - time_stamps[3]).sum():.4f}')
    print(f'Metric calc:            {(time_stamps[5] - time_stamps[4]).sum():.4f}')
    print(f'Total:                  {(time_stamps[5] - time_stamps[6]).sum():.4f}')

    print('-------------------------------------------------')
    descr_str = 'Finite diff'
    for loss_name, loss_dict in loss_outputs[0].items():
        name = loss_dict['name']
        if loss_name != 'sim':
            name = descr_str + name + ':'
        print(f'{name:<30} {loss_dict['time']:.4f}')

def main(**kwargs):
    config = Namespace(**kwargs)
    config.func_kwargs = {
        'layers': [2, 64, 64, 64, 2],
        'weight_init': True,
        'last_init_zero': False,
        'omega': 30,
        # 'img_sz': (128, 128),
        # 'smoothing_kernel': 'AK',
        # 'smoothing_win': 15,
        # 'smoothing_pass': 1,
        # 'ds': 2,
        # 'bs': 16,
        # 'use_t': False,
    }
    config.log_path = os.path.join('log/motion_test_cmr/heart_init_no_ft_abs', config.comment)
    train(config)

if __name__ == '__main__':
    debug = False

    epochs = 1000
    lr=1e-3
    solver = 'euler'
    step_size = 0.01
    func_name = 'sirenensemble'
    lambda_st = 1
    lambda_grd = 0.
    lambda_negJ = 0.
    lambda_hel = 0.
    lambda_pgr = 0.
    lambda_lap = 0.
    lambda_recon = 0.
    weight_decay = 0.
    start_frame = 0
    time_points = 12
    schedule = [1]

    # init_lr = 1e-2
    # lambda_init_rec = 1e-5
    scale = 10
    reg_alpha = 0
    recon_epochs = 200

    dataset = 'heart_init_no_ft_abs'
    device='cuda'
    for init_lr in [1e-2]:
        for lambda_init_rec in [1e-1]:
            comment = f'long_test-lr{lr}-grd{lambda_grd}-e{epochs}-a{reg_alpha}-irec{lambda_init_rec}-ilr{init_lr}-scale{scale}-re{recon_epochs}'
            main(
                log_cadence=50,
                epochs=epochs,
                lr=lr,
                schedule=schedule,
                solver=solver,
                use_nreps=False,
                loss='mse',
                debug=debug,
                atol=1e-8,
                rtol=1e-6,
                step_size=step_size,
                func_name=func_name,
                comment=comment,
                dataset=dataset,
                device=device,
                start_frame=start_frame,
                time_points=time_points,

                lambda_st=lambda_st,
                lambda_grd=lambda_grd,
                lambda_negJ=lambda_negJ,
                lambda_hel=lambda_hel,
                lambda_pgr=lambda_pgr,
                lambda_lap=lambda_lap,
                lambda_recon=lambda_recon,
                weight_decay=weight_decay,

                init_lr=init_lr,
                lambda_init_recon=lambda_init_rec,
                recon_scale=scale,
                reg_alpha=reg_alpha,
                recon_epochs=recon_epochs,
            )