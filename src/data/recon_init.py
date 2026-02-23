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


def init_using_nmAPG(config: argparse.Namespace,
                     recon_init: torch.Tensor,
                     fixed: torch.Tensor,
                     forw_subs: nn.Module,
                     forw_subs_adj: nn.Module,
                     logger: logging.Logger) -> tuple[torch.Tensor, np.ndarray]:
    reg = get_reg(config)
    data_fit, reg_eval, energy, energy_grad = get_functions(config, reg, forw_subs, forw_subs_adj)
    energy_and_grad = lambda val, y_in: (energy(val, y_in), energy_grad(val, y_in))
    
    t0 = time.time()
    x, L, i, converged, metrics = nmAPG(x0=recon_init,
                               y=fixed,
                               f=energy,
                               nabla=energy_grad,
                               f_and_nabla=energy_and_grad,
                               max_iter=config.recon_epochs,
                               verbose=config.debug,
                               tol=config.tol,
                               data_fit=data_fit,
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

def init_with_grad_desc(config: argparse.Namespace,
                        recon: nn.Parameter,
                        gt: torch.Tensor,
                        forw: nn.Module,):
    optimizer = torch.optim.Adam([recon], lr=config.init_lr)
    gt = gt.to(config.device)
    loss_fn = nn.MSELoss(reduction='mean')
    regularizer = get_recon_regularizer(config)

    metrics = np.zeros((config.recon_epochs+1, 2)) if config.debug else None
    if config.debug:
        pred = fastmri.complex_abs_sq(forw(recon).movedim(1,-1)).unsqueeze(1)
        pred = (pred + 1e-8).sqrt()
        sim_loss = config.lambda_st * loss_fn(pred, gt).detach().cpu()
        reg_loss = config.lambda_init_recon * regularizer.g(pred).mean().detach().cpu()
        metrics[0] = [sim_loss, reg_loss]
        print(f'Before opt energy {sim_loss+reg_loss}, data fit {sim_loss}, reg {reg_loss}')

    best_loss = 1e8
    t0 = time.time()
    for epoch in range(1, config.recon_epochs + 1):
        optimizer.zero_grad()
        pred = fastmri.complex_abs_sq(forw(recon).movedim(1,-1)).unsqueeze(1)
        pred = (pred + 1e-8).sqrt()
        sim_loss = config.lambda_st * loss_fn(pred, gt)
        reg_loss = config.lambda_init_recon * regularizer.g(pred).mean()
        loss_sum = sim_loss + reg_loss
        loss_sum.backward()
        print(recon.grad.abs().max(), sim_loss)
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