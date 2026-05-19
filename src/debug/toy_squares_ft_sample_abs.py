from argparse import Namespace
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import random

import numpy as np
import torch
import torch.nn as nn
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
from src.data.data_utils import get_kspace_mask
from src.debug.generate_example import generate_sample

import matplotlib.pyplot as plt
plt.switch_backend('agg')

torch.manual_seed(0)
random.seed(1)
np.random.seed(2)

def train(config):
    cirs = generate_sample(circles_nr=config.circles_nr, direction=config.distance, img_size=256, seed=config.gen_seed)
    gt_im = torch.stack([cirs, torch.zeros_like(cirs)], dim=1)
    kspace_mask = get_kspace_mask(config, gt_im, factor=config.factor)
    
    full_forw = FastmriFT()
    full_adj = FastmriIFT()
    forw_subs = FTAndSubsample(kspace_mask)
    forw_subs_adj = ZeroFillAndIFT(kspace_mask)
    # full_adj = nn.Identity()
    # forw_subs = nn.Identity()

    gt_kspace_data = full_forw(gt_im)
    forw = forw_subs

    fixed = forw(gt_im)
    # fixed = fastmri.complex_abs_sq(fixed.to(config.device).movedim(1,-1)).unsqueeze(1)
    # fixed = (fixed + 1e-8).sqrt()
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
    

    # with torch.autograd.set_detect_anomaly(True):
    out = registration(config=config,
                    writer=writer,
                    logger=logger,
                    inputs=inputs,
                    eval_inputs=eval_inputs,
                    func=func)

    model_outputs, loss_outputs, _, _, epoch, all_metrics, time_stamps, _ = out
    init_recon, gt_im, _, _ = eval_inputs
    rel_vel, abs_phi, _ = model_outputs
    loss_dict, _, moved_im = loss_outputs
    
    # with torch.no_grad():
    #     # inputs[0] = fastmri.complex_abs(inputs[0].movedim(0,-1))
    #     # inputs[2] = inputs[2].transpose(0,1).reshape(24, 1, 204, 512)
    #     # loss_outputs[1] = loss_outputs[1].transpose(0,1).reshape(24, 1, 204, 512)
    #     # loss_outputs[2] = loss_outputs[2].transpose(0,1).reshape(24, 1, 204, 512)
    #     metrics, imgs_to_save = calculate_metrics(config, None, inputs, eval_inputs, model_outputs, loss_outputs, extended_log=True)
    #     log_metrics(config, metrics, writer, epoch + 10, imgs_to_save, last_val=True)
    #     prep_vis_summary_pdf(config, gt_im, init_recon, moved_im, abs_phi, rel_vel, all_metrics)

    os.makedirs(os.path.join(config.log_path, 'res'), exist_ok=True)
    torch.save(model_outputs[1].detach().cpu(), os.path.join(config.log_path, 'res', 'abs_phi.pt'))
    torch.save(model_outputs[0].detach().cpu(), os.path.join(config.log_path, 'res', 'rel_vel.pt'))
    torch.save(eval_inputs[1].detach().cpu(), os.path.join(config.log_path, 'res', 'gt_im.pt'))
    torch.save(inputs[2].detach().cpu(), os.path.join(config.log_path, 'res', 'fixed.pt'))
    torch.save(loss_outputs[2].detach().cpu(), os.path.join(config.log_path, 'res', 'moved_im.pt'))

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
        'last_init_zero': True,
        'omega': 30,
        # 'img_sz': (128, 128),
        # 'smoothing_kernel': 'AK',
        # 'smoothing_win': 15,
        # 'smoothing_pass': 1,
        # 'ds': 2,
        # 'bs': 16,
        # 'use_t': False,
    }
    config.log_path = os.path.join('log/motion_test_squares/toy_square_ft_sample_abs', config.comment)
    train(config)

if __name__ == '__main__':
    debug = True

    mask = 'st'

    epochs = 200
    lr=1e-4
    solver = 'euler'
    step_size = 0.01
    func_name = 'sirenensemble'
    lambda_st = 1
    lambda_grd = 1e-2
    lambda_negJ = 1e-10
    lambda_hel = 0.
    lambda_pgr = 0.
    lambda_lap = 0.
    lambda_recon = 0.
    lambda_rl2 = 0.
    weight_decay = 0.
    start_frame = 0
    time_points = 2
    schedule = [1]
    factor = 4

    dataset = 'toy_square'
    device='cuda'

    circles_nr = 1
    direction = 'nsame'
    dist = 20
    seed = 42
    for circles_nr in [7, 10]:
        for dist in [5, 10, 20, 50]:
            for seed in range(5):
                comment = f'ft-factor{factor}/no_abs-cirs{circles_nr}-dir{direction}-dist{dist}-seed{seed}-lr{lr}-grd{lambda_grd}-rl2{lambda_rl2}-e{epochs}-factor{factor}-tp{time_points}-lastinitzeroTrue'
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
                    tm=0,
                    step_size=step_size,
                    func_name=func_name,
                    comment=comment,
                    dataset=dataset,
                    device=device,
                    start_frame=start_frame,
                    time_points=time_points,
                    learn_recon=False,
                    interval=epochs,

                    lambda_st=lambda_st,
                    lambda_grd=lambda_grd,
                    lambda_negJ=lambda_negJ,
                    lambda_hel=lambda_hel,
                    lambda_pgr=lambda_pgr,
                    lambda_lap=lambda_lap,
                    lambda_rl2=lambda_rl2,
                    sim_lambda=torch.ones((2), device=device),
                    lambda_recon=lambda_recon,
                    weight_decay=weight_decay,

                    mask=mask,
                    factor=factor,

                    circles_nr=circles_nr,
                    direction=direction,
                    distance=dist,
                    gen_seed=seed,
                )