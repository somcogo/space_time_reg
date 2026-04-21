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
from src.data.fft_utils import FastmriFT, FastmriIFT

import matplotlib.pyplot as plt
plt.switch_backend('agg')

torch.manual_seed(0)
random.seed(1)
np.random.seed(2)

def train(config):
    gt_kspace_data = torch.load(f'data/processed/cmrxrecon/test/training_p001_single_coil_full_cine_sax_norm.pt')[:config.time_points,0].permute(0, 3, 1, 2)
    ift = FastmriIFT()
    gt_im = ift(gt_kspace_data)
    # gt_im = fastmri.complex_abs(gt_im.movedim(1, -1)).unsqueeze(1)

    forw = torch.nn.Identity()

    fixed = gt_im
    fixed = fastmri.complex_abs_sq(fixed.to(config.device).movedim(1,-1)).unsqueeze(1)
    fixed = (fixed + 1e-8).sqrt()
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

    with torch.no_grad():
        # inputs[0] = fastmri.complex_abs(inputs[0].movedim(0,-1))
        # inputs[2] = inputs[2].transpose(0,1).reshape(24, 1, 204, 512)
        # loss_outputs[1] = loss_outputs[1].transpose(0,1).reshape(24, 1, 204, 512)
        # loss_outputs[2] = loss_outputs[2].transpose(0,1).reshape(24, 1, 204, 512)
        metrics, imgs_to_save = calculate_metrics(config, None, inputs, eval_inputs, model_outputs, loss_outputs, extended_log=True)
        log_metrics(config, metrics, writer, epoch + 10, imgs_to_save, last_val=True)
        prep_vis_summary_pdf(config, gt_im, init_recon, moved_im, abs_phi, rel_vel, all_metrics)

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
        'layers': [2] + kwargs['depth']*[kwargs['dim']] + [2],
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
    config.log_path = os.path.join('log/motion_test_cmr/heart_gt_ft_abs_sirensearch', config.comment)
    train(config)

if __name__ == '__main__':
    debug = False

    # dim = 64
    # depth = 2

    epochs = 200
    # lr=1e-4
    solver = 'euler'
    step_size = 0.01
    func_name = 'sirenensemble'
    lambda_st = 1
    lambda_grd = 1e-4
    # lambda_negJ = 1e-10
    lambda_hel = 0.
    lambda_pgr = 0.
    lambda_lap = 0.
    lambda_recon = 0.
    weight_decay = 0.
    lambda_rl2 = 1e-2
    start_frame = 0
    tm = 0
    time_points = 2
    schedule = [1]

    dataset = 'heart_gt_ft_abs'
    device='cuda'
    for dim in [4, 2, 1]:
        for depth in [3, 2, 1]:
            for lambda_negJ in [1e-2]:
                for lr in [1e-3]:
                    comment = f'size_search/lr{lr}-grd{lambda_grd}-rl2{lambda_rl2}-negJ{lambda_negJ}-e{epochs}-tp{time_points}-lastinitzeroTrue-dim{dim}-depth{depth}'
                    # comment = f'hsearch2x64_new/lr{lr}-grd{lambda_grd}-rl2{lambda_rl2}-negJ{lambda_negJ}-e{epochs}-tp{time_points}-lastinitzeroTrue-dim{dim}-depth{depth}'
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
                        tm=tm,
                        time_points=time_points,

                        lambda_st=lambda_st,
                        lambda_grd=lambda_grd,
                        lambda_negJ=lambda_negJ,
                        lambda_hel=lambda_hel,
                        lambda_pgr=lambda_pgr,
                        lambda_lap=lambda_lap,
                        lambda_rl2=lambda_rl2,
                        lambda_recon=lambda_recon,
                        weight_decay=weight_decay,

                        dim=dim,
                        depth=depth,
                    )