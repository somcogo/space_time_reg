import argparse
import os
import time
from typing import Callable, List

import torch
from torch import nn
import fastmri

from learned_regularizers.evaluation.nmAPG import nmAPG
from src.data.data_utils import reconstruct_initial_frame_learned_reg
from src.data.fft_utils import FTAndSubsample, ZeroFillAndIFT, FastmriIFT, FastmriFT
from src.losses.recon_reg import get_recon_regularizer

def train(config: argparse.Namespace):
    raw_kspace_data, gt_kspace_data = get_data(config)
    
    kspace_mask = (raw_kspace_data[:1] != 0)
    full_forw, full_adj, forw_subs, forw_subs_adj = get_operators(config, kspace_mask)

    smaller_shape = list(raw_kspace_data.shape[:2]) + [-1] + list(raw_kspace_data.shape[3:])
    fixed = raw_kspace_data[kspace_mask.expand(raw_kspace_data.shape)].reshape(smaller_shape)

    gt_im = full_adj(gt_kspace_data)
    gt_im = gt_im.to(config.device)
    recon_init = get_init(config, raw_kspace_data, gt_im)

    reg = get_reg(config)
    data_fit, reg_eval, energy, energy_grad = get_functions(config, reg, forw_subs, forw_subs_adj)
    energy_and_grad = lambda val, y_in: (energy(val, y_in), energy_grad(val, y_in))
    
    t0 = time.time()
    x, L, i, converged = nmAPG(x0=recon_init,
                               y=fixed,
                               f=energy,
                               nabla=energy_grad,
                               f_and_nabla=energy_and_grad,
                               max_iter=config.recon_epochs,
                               verbose=config.debug,
                               tol=config.tol,
                               data_fit=data_fit,
                               reg=reg_eval)
    t1 = time.time()
    print(f'Finished initial reconstruction in {t1-t0:.4f} seconds')

    with torch.no_grad():
        print('gt max min', gt_im.abs().max(), gt_im.min(), 'recon init max min', recon_init.abs().max(), recon_init.min())
        gtabs = fastmri.complex_abs(gt_im.movedim(1, -1))
        reconabs = fastmri.complex_abs(recon_init.detach().cpu().movedim(1, -1))
        print('gt max min', gtabs.max(), gtabs.min(), 'recon init max min', reconabs.max(), reconabs.min())
        print(f'Max abs diff {(gtabs-reconabs).abs().max() / gtabs.max()}, max abs of diff value {(gt_im - recon_init.detach().cpu()).abs().max() / gt_im.abs().max()}')

def get_data(config: argparse.Namespace) -> list[torch.Tensor]:
    if config.data == 'cmr_test1':
        raw_kspace_data = torch.load(f'data/processed/cmrxrecon/test/training_p001_single_coil_acc_04_cine_sax_norm.pt')[:1,0].permute(0, 3, 1, 2)
        gt_kspace_data = torch.load(f'data/processed/cmrxrecon/test/training_p001_single_coil_full_cine_sax_norm.pt')[:1,0].permute(0, 3, 1, 2)
    elif config.data == 'cmr_test2':
        gt_kspace_data = torch.load(f'data/processed/cmrxrecon/test/training_p001_single_coil_full_cine_sax_norm.pt')[:1,0].permute(0, 3, 1, 2)
        raw_kspace_data = gt_kspace_data
    return raw_kspace_data, gt_kspace_data

def get_operators(config: argparse.Namespace, mask: torch.Tensor) -> list[nn.Module]:
    if config.data == 'cmr_test1':
        full_forw = FastmriFT()
        full_adj = FastmriIFT()
        forw_subs = FTAndSubsample(mask)
        forw_subs_adj = ZeroFillAndIFT(mask)
    elif config.data == 'cmr_test2':
        full_forw = FastmriFT()
        full_adj = FastmriIFT()
        forw_subs = FastmriFT()
        forw_subs_adj = FastmriIFT()
    return full_forw, full_adj, forw_subs, forw_subs_adj

def get_init(config: argparse.Namespace, raw_kspace_data: torch.Tensor, gt_im: torch.Tensor) -> nn.Parameter:
    if config.init == 'zero':
        recon_init = nn.Parameter(torch.zeros_like(raw_kspace_data, device=config.device), requires_grad=True)
    elif config.init == 'gt':
        recon_init = nn.Parameter(gt_im.clone(), requires_grad=True)
    return recon_init

def get_reg(config: argparse.Namespace) -> nn.Module:
    if config.reg == 'learned':
        reg = get_recon_regularizer(config)
    return reg
                  
def get_functions(config: argparse.Namespace, regularizer: nn.Module, forw: nn.Module, adj: nn.Module) -> list[Callable]:
    def data_fit(val: torch.Tensor, y_in: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            diff = forw(val) - y_in
            df = 0.5 * (diff ** 2).sum((1,2,3))
        if config.detach_grads:
            df = df.detach()
        return df.reshape(-1)
    
    def reg_eval(val: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            reg = config.lambda_init_recon * regularizer.g(
                val.flatten(0,1).unsqueeze(1)
            ).reshape(val.shape[0], -1).sum(1)
        if config.detach_grads:
            reg = reg.detach()
        return reg.reshape(-1)
    
    def energy(val: torch.Tensor, y_in: torch.Tensor) -> torch.Tensor:
        df = data_fit(val, y_in)
        reg = reg_eval(val)
        fun = df + reg
        if config.detach_grads:
            fun = fun.detach()
        return fun.reshape(-1)
    
    def energy_grad(val: torch.Tensor, y_in: torch.Tensor) -> torch.Tensor:
        diff = forw(val) - y_in
        df_grad = adj(diff)
        reg_grad = config.lambda_init_recon * regularizer.grad(
            val.flatten(0,1).unsqueeze(1)
        ).reshape(val.shape)
        return df_grad + reg_grad
    
    return data_fit, reg_eval, energy, energy_grad


config = argparse.Namespace(
    dataset='cmr_P001_Acc04',
    slice_number=0,
    start_frame=0,
    time_points=1,
    device='cuda',
    recon_scale=13,
    recon_epochs=20,
    debug=True,
    lambda_st=1,
    lambda_init_recon=10,
    detach_grads=True,
    init_lr=1e-4,
)

def main(**kwargs) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon_alpha", type=float, default=1)
    parser.add_argument("--recon_scale", type=float, default=1e-1)
    parser.add_argument("--recon_epochs", type=float, default=500)
    parser.add_argument("--lambda_st", type=float, default=1)
    parser.add_argument("--lambda_recon", type=float, default=1e-9)
    parser.add_argument("--log_path", type=str, default='test')

    config = parser.parse_args()
    d = vars(config)
    for (k, v) in kwargs.items():
        d[k] = v

    config.log_path = os.path.join('log/recon_init', config.log_path)
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(config)

if __name__ == '__main__':
    warnings = []
    for lr in [1e0, ]:
        for scale in [1e-1, 1e0, 1e1, 1e2, 1e3]: # 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3
            for lam in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
                try:
                    main(recon_lr=lr, recon_scale=scale, lambda_recon=lam, log_path=f'lr-{lr}-sc-{scale}-lam-{lam}')
                except:
                    warnings.append(f'lr-{lr}-sc-{scale}-lam-{lam}\n')
    print(*warnings)