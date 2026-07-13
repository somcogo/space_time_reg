import argparse
import time
from typing import Callable
import logging

import numpy as np
import torch
from torch import nn
import fastmri

from .nmapg import nmAPG
from src.losses.recon_reg import get_recon_regularizer
from src.losses.tv import get_tv
from .data_utils import complex_abs, real_abs


def init_using_nmAPG(config: argparse.Namespace,
                     recon_init: torch.Tensor,
                     fixed: torch.Tensor,
                     forw_subs: nn.Module,
                     forw_subs_adj: nn.Module,
                     logger: logging.Logger) -> tuple[torch.Tensor, np.ndarray]:
    reg = get_reg(config)
    data_fit, reg_eval, energy, energy_grad = get_functions(config, reg, forw_subs, forw_subs_adj)
    energy_and_grad = lambda val, y_in: (energy(val, y_in), energy_grad(val, y_in))
    
    weighted_data_fit = lambda val, y_in: config.lambda_st * data_fit(val, y_in)
    L_init = config.lambda_st

    t0 = time.time()
    x, L, i, converged, metrics = nmAPG(x0=recon_init,
                               y=fixed,
                               f=energy,
                               nabla=energy_grad,
                               f_and_nabla=energy_and_grad,
                               max_iter=config.recon_epochs,
                               L_init=L_init,
                               verbose=config.debug,
                               tol=config.tol,
                               data_fit=weighted_data_fit,
                               reg=reg_eval,
                               debug=config.debug,
                               logger=logger)
    t1 = time.time()
    if logger is None:
        print(f'Finished initial reconstruction in {t1-t0:.4f} seconds')
    else:
        logger.info(f'Finished initial reconstruction in {t1-t0:.4f} seconds')
    return x, metrics

def get_reg(config: argparse.Namespace) -> nn.Module:
    if config.reg == 'learned':
        reg = get_recon_regularizer(config)
    elif config.reg == 'tv':
        reg = get_tv(config)
    return reg

def unpack_measurements(y_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # y_in packs the zero-filled measured k-space (channels 0:2) together with the
    # per-frame k-space mask (channels 2:4), so that nmAPG's batch subsetting y[idx]
    # keeps each frame's mask aligned with its measurements (see prepare_inverse_case).
    return y_in[:, :2], y_in[:, 2:]

def complex_l2(val: torch.Tensor, y_in: torch.Tensor, forw: nn.Module, config: argparse.Namespace) -> torch.Tensor:
    kdata, mask = unpack_measurements(y_in)
    diff = mask * forw(val) - kdata
    df = 0.5 * (diff ** 2).sum((1,2,3))
    return df.reshape(-1)

def magnitude_l1(val: torch.Tensor, y_in: torch.Tensor, forw: nn.Module, config: argparse.Namespace) -> torch.Tensor:
    kdata, mask = unpack_measurements(y_in)
    val_abs = complex_abs(mask * forw(val))
    y_abs = complex_abs(kdata)
    df = real_abs(val_abs - y_abs).sum((1,2,3))
    return df.reshape(-1)

def log_magnitude(val: torch.Tensor, y_in: torch.Tensor, forw: nn.Module, config: argparse.Namespace) -> torch.Tensor:
    kdata, mask = unpack_measurements(y_in)
    val_abs = complex_abs(mask * forw(val))
    y_abs = complex_abs(kdata)
    diff = torch.log(val_abs + 1e-8) - torch.log(y_abs + 1e-8)
    df = 0.5 * (diff ** 2).sum((1,2,3))
    return df.reshape(-1)

def reg_on_abs(config: argparse.Namespace, val: torch.Tensor, regularizer: nn.Module) -> torch.Tensor:
    val_abs = complex_abs(val)
    reg = config.lambda_init_recon * regularizer.g(val_abs)
    return reg.reshape(-1)

def reg_without_abs(config: argparse.Namespace, val: torch.Tensor, regularizer: nn.Module) -> torch.Tensor:
    reg = config.lambda_init_recon * regularizer.g(
        val.flatten(0,1).unsqueeze(1)
    ).reshape(val.shape[0], -1).sum(1)
    return reg.reshape(-1)

                  
def get_functions(config: argparse.Namespace, regularizer: nn.Module, forw: nn.Module, adj: nn.Module) -> list[Callable]:
    def data_fit(val: torch.Tensor, y_in: torch.Tensor) -> torch.Tensor:
        if config.init_loss == 'mag_l1':
            df = magnitude_l1(val, y_in, forw, config)
        elif config.init_loss == 'log_mag':
            df = log_magnitude(val, y_in, forw, config)
        elif config.init_loss == 'l2':
            df = complex_l2(val, y_in, forw, config)
        return df
    
    def reg_eval(val: torch.Tensor) -> torch.Tensor:
        if config.init_reg_abs:
            reg = reg_on_abs(config, val, regularizer)
        else:
            reg = reg_without_abs(config, val, regularizer)
        return reg
    
    def energy(val: torch.Tensor, y_in: torch.Tensor) -> torch.Tensor:
        df = config.lambda_st * data_fit(val, y_in)
        reg = reg_eval(val)
        fun = df + reg
        if config.detach_grads:
            fun = fun.detach()
        return fun.reshape(-1)

    def energy_grad(val: torch.Tensor, y_in: torch.Tensor) -> torch.Tensor:
        val_req = val.detach().clone().requires_grad_(True)
        df = config.lambda_st * data_fit(val_req, y_in)
        reg = reg_eval(val_req)

        energy = df + reg
        energy.sum().backward()
        return val_req.grad
        # diff = forw(val) - y_in
        # df_grad = adj(diff)
        # reg_grad = config.lambda_init_recon * regularizer.grad(
        #     val.flatten(0,1).unsqueeze(1)
        # ).reshape(val.shape)
        # return df_grad + reg_grad
    
    return data_fit, reg_eval, energy, energy_grad

def init_with_grad_desc(config: argparse.Namespace,
                        recon: nn.Parameter,
                        gt: torch.Tensor,
                        forw: nn.Module,
                        sim_use_abs: bool = False,
                        reg_use_abs: bool = False):
    optimizer = torch.optim.Adam([recon], lr=config.init_lr)
    gt = gt.to(config.device)
    loss_fn = nn.MSELoss(reduction='mean')
    regularizer = get_recon_regularizer(config)

    metrics = np.zeros((config.recon_epochs+1, 2)) if config.debug else None

    if sim_use_abs:
        gt = complex_abs(gt)
    if config.debug:
        if sim_use_abs:
            pred = complex_abs(forw(recon))
        else:
            pred = forw(recon)
        sim_loss = config.lambda_st * loss_fn(pred, gt).detach().cpu()
        if reg_use_abs:
            pred = complex_abs(recon)
        else:
            pred = recon.flatten(0,1).unsqueeze(1)
        reg_loss = config.lambda_init_recon * regularizer.g(pred).mean().detach().cpu()
        metrics[0] = [sim_loss, reg_loss]
        print(f'Before opt energy {sim_loss+reg_loss}, data fit {sim_loss}, reg {reg_loss}')

    best_loss = 1e8
    t0 = time.time()
    for epoch in range(1, config.recon_epochs + 1):
        optimizer.zero_grad()
        
        if sim_use_abs:
            pred = complex_abs(forw(recon))
        else:
            pred = forw(recon)
        sim_loss = config.lambda_st * loss_fn(pred, gt)
        
        if reg_use_abs:
            pred = complex_abs(recon)
        else:
            pred = recon.flatten(0,1).unsqueeze(1)
        reg_loss = config.lambda_init_recon * regularizer.g(pred).mean()
        
        loss_sum = sim_loss + reg_loss
        loss_sum.backward()
        optimizer.step()
        if loss_sum <= best_loss:
            best_recon = recon.detach().clone()
            best_loss = loss_sum.detach().clone()
        if config.debug:
            sim_loss = sim_loss.detach().cpu().clone()
            reg_loss = reg_loss.detach().cpu().clone()
            metrics[epoch] = [sim_loss, reg_loss]
            print(f'Iter {epoch}/{config.recon_epochs} energy {sim_loss+reg_loss}, data fit {sim_loss}, reg {reg_loss}')
    t1 = time.time()
    print(f'Finished initial reconstruction in {t1-t0:.4f} seconds')
    return best_recon, metrics