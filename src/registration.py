import logging
import time

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
    # y0 = generate_grid_tensor(data[0].shape[1:]).to(config.device)
    # # y0 = get_mgrid(data[0].shape[1:]).view(data[0][:2].shape).unsqueeze(0).to(config.device)
    # y0 = y0[:, [1, 0], ...]
    coord_tensor = generate_coord_tensor(img_shape, config.device)
    time_points = torch.arange(config.time_points, device=config.device) * config.time_step
    optimizer = torch.optim.Adam(func.parameters(), lr=config.lr)

    best_loss = 1e8
    old_phi = 0
    t7 = time.time()
    t17, t21, t32, t43, t54, t65, t67 = 0., 0., 0., 0., 0., 0., 0.
    for epoch in range(1, config.epochs + 1):
        log_epoch = epoch % config.log_cadence == 0
        optimizer.zero_grad()
        t1 = time.time()
        phi = odeint(func, coord_tensor, time_points, method=config.solver)
        vel = func(time_points[-1], coord_tensor)
        phi = torch.relu(phi+1) - 1
        phi = -torch.relu(-phi+1) + 1
        t2 = time.time()
        losses, moved_imgs = calculate_losses(config, phi, data, vel)
        loss = sum(losses)
        t3 = time.time()
        loss.backward()
        t4 = time.time()
        optimizer.step()
        t5 = time.time()

        metrics = calculate_metrics(losses, phi, data)
        log_metrics(metrics, phi, data, writer, epoch, moved_imgs, vel, log_epoch)
        t6 = time.time()

        # new_phi = phi.detach()
        t17 += t1-t7
        t21 += t2-t1
        t32 += t3-t2
        t43 += t4-t3
        t54 += t5-t4
        t65 += t6-t5
        t67 += t6-t7
        if epoch == 1 or log_epoch:
            logger.info(f'Epoch {epoch}/{config.epochs}, Sim loss {losses[0]:.3f}, NegJ loss {losses[1]:.3f}, Smooth loss {losses[2]:.3f}, Vmag loss {losses[3]:.3f}')
            logger.info(f'Epoch {epoch}/{config.epochs}, Time: DL {t17/100:.4f}, ODE {t21/100:.4f}, Loss  {t32/100:.4f}, Backward {t43/100:.4f}, Optim  {t54/100:.4f}, Metric {t65/100:.4f}, Total {t67/100:.4f}')
            # last_layer_grad = sum([p.grad.abs() for p in func.parameters()][:-1])
            # grads = sum([p.grad.norm().cpu() for p in func.parameters()])
            # logger.info(f'Phi diff {(new_phi - old_phi).abs().max().cpu()}, Grad norm sum {grads}, phi nan {torch.isnan(phi).any()}')
            # old_phi = phi.detach()
            t17, t21, t32, t43, t54, t65, t67 = 0., 0., 0., 0., 0., 0., 0.
        

        if sum(losses) < best_loss:
            best_loss = sum(losses)
            best_phi = phi
            best_vel = vel
        t7 = time.time()
    
    return best_phi, best_vel