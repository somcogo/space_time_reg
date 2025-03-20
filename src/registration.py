import logging

import torch
from torchdiffeq import odeint_adjoint as odeint

from src.networks import get_func
from src.utils import generate_grid_tensor, calculate_metrics, log_metrics
from src.losses import calculate_losses
from src.siren.dataio import get_mgrid

def registration(config, data, writer, logger:logging.Logger):
    func = get_func(config.func_name, config.func_kwargs)
    func = func.to(config.device)
    y0 = generate_grid_tensor(data[0].shape[1:]).to(config.device)
    # y0 = get_mgrid(data[0].shape[1:]).view(data[0][:2].shape).unsqueeze(0).to(config.device)
    y0 = y0[:, [1, 0], ...]
    time_points = torch.arange(config.time_points, device=config.device) * config.time_step
    optimizer = torch.optim.Adam(func.parameters(), lr=config.lr)

    best_loss = 1e8
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad()
        phi = odeint(func, y0, time_points, method=config.solver)
        phi = torch.relu(phi+1) - 1
        phi = -torch.relu(-phi+1) + 1
        losses = calculate_losses(config, phi, data)
        loss = sum(losses)
        loss.backward()
        optimizer.step()

        metrics = calculate_metrics(loss, phi, data)
        log_metrics(metrics, phi, data, writer, epoch)

        if epoch == 1 or epoch % config.log_cadence == 0:
            logger.info(f'Epoch {epoch}/{config.epochs}, Sim loss {losses[0]:.3f}, NegJ loss {losses[1]:.3f}, Smooth loss {losses[2]:.3f}, Vmag loss {losses[3]:.3f}')
        
        if sum(losses) < best_loss:
            best_loss = sum(losses)
            best_phi = phi
    
    return best_phi        