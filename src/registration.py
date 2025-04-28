import logging
import time
import math

import torch
from torchdiffeq import odeint_adjoint as odeint

from src.networks import get_func
from src.utils import generate_grid_tensor, calculate_metrics, log_metrics, generate_coord_tensor
from src.losses import calculate_losses
from src.siren.dataio import get_mgrid

def registration(config, data, writer, logger:logging.Logger):
    img_shape = data[0].shape[1:]
    dims = len(img_shape)
    config.func_kwargs['layers'][0] = dims
    config.func_kwargs['layers'][-1] = dims
    func = get_func(config.func_name, config.func_kwargs)
    func = func.to(config.device)
    coord_tensor = generate_coord_tensor(img_shape, config.device)
    time_points = torch.arange(config.time_points, device=config.device) * config.time_step
    optimizer = torch.optim.Adam(func.parameters(), lr=config.lr)

    siren_st_dict = torch.load('data/gt_state_dicts/rot_slow2_x_is_zero_siren_state_dict.pt')
    func.load_state_dict(siren_st_dict)

    best_loss = 1e8
    t7 = time.time()
    t17, t21, t32, t43, t54, t65, t67 = 0., 0., 0., 0., 0., 0., 0.
    for epoch in range(1, config.epochs + 1):
        func.load_state_dict(siren_st_dict)
        log_epoch = epoch % config.log_cadence == 0
        optimizer.zero_grad()
        t1 = time.time()
        phi = odeint(func, coord_tensor, time_points, method=config.solver)
        vel = func(time_points[1], coord_tensor)
        phi = torch.relu(phi+1) - 1
        phi = -torch.relu(-phi+1) + 1
        t2 = time.time()
        losses, moved_imgs, energies = calculate_losses(config, phi, data, vel)
        loss = sum(losses)
        t3 = time.time()
        loss.backward()
        t4 = time.time()
        optimizer.step()
        t5 = time.time()

        metrics = calculate_metrics(losses, phi, data)
        log_metrics(metrics, phi, data, writer, epoch, moved_imgs, vel, func)
        t6 = time.time()

        t17 += t1-t7
        t21 += t2-t1
        t32 += t3-t2
        t43 += t4-t3
        t54 += t5-t4
        t65 += t6-t5
        t67 += t6-t7
        if epoch == 1 or log_epoch:
            logger.info(f'Epoch {epoch:4d}/{config.epochs}, Losses Sim {losses[0]:.3f} NegJ {losses[1]:.3f} Smooth {losses[2]:.3f} Vmag {losses[3]:.3f}, Times DL/ODE/Loss/Back/Optim/Metr/Total {t17/100:.4f} {t21/100:.4f} {t32/100:.4f} {t43/100:.4f} {t54/100:.4f} {t65/100:.4f} {t67/100:.4f}')
            t17, t21, t32, t43, t54, t65, t67 = 0., 0., 0., 0., 0., 0., 0.
        

        if sum(losses) < best_loss:
            best_loss = sum(losses)
            best_phi = phi
            best_vel = vel
        t7 = time.time()
    
    return best_phi, best_vel, coord_tensor