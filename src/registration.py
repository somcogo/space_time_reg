import logging

import torch
from torchdiffeq import odeint_adjoint as odeint

from src.networks import get_func
from src.utils import generate_grid_tensor, calculate_metrics, log_metrics
from src.losses import calculate_losses

def registration(config, data, writer, logger:logging.Logger):
    func = get_func(config.func_name, config.func_kwargs)
    y0 = generate_grid_tensor(data[0].shape[1:])
    time_points = torch.arange(config.time_points) * config.time_step
    optimizer = torch.optim.Adam(func.parameters(), lr=config.lr)

    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad()
        phi = odeint(func, y0, time_points, method=config.solver)
        loss = calculate_losses(config, phi, data)
        loss.backward()
        optimizer.step()

        metrics = calculate_metrics(loss, phi, data)
        log_metrics(metrics, phi, data, writer, epoch)

        if epoch == 1 or epoch % config.log_cadence == 0:
            logger.info(f'Epoch {epoch}/{config.epochs}, Loss {loss:.2f}')
        