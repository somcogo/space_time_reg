import torch
from torchdiffeq import odeint_adjoint as odeint

from networks import get_func
from utils import generate_grid_tensor
from losses import calculate_losses

def registration(config, data, writer):
    func = get_func(config.func_name, config.func_kwargs)
    y0 = generate_grid_tensor(data['shape'])
    time_points = torch.arange(config.time_points) * config.time_steps
    optimizer = torch.optim.Adam(func.parameters(), lr=config.lr)

    for epoch in range(1, config.epochs + 1):
        phi = odeint(func, y0, time_points, method=config.solver)
        loss = calculate_losses(phi, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()