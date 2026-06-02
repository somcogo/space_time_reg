from argparse import Namespace
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random

import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint

from src.models.factory import get_func
from src.losses.losses import negJ_loss
from src.utils.spatial_utils import generate_coord_tensor

import matplotlib.pyplot as plt
plt.switch_backend('agg')

torch.manual_seed(0)
random.seed(1)
np.random.seed(2)

def get_model_output(config:Namespace, func: nn.Module, coord_tensor: torch.Tensor):
    time_points = torch.tensor([0., 1.], device=config.device)
    init_value = coord_tensor.expand((config.time_points - 1, -1, -1))

    abs_phi = odeint(func, init_value, time_points, method=config.solver, atol=config.atol, rtol=config.rtol, options={'step_size':config.step_size})
    abs_phi = abs_phi[1] # ignore abs_phi[0], which is just init_value anyway
    print(abs_phi.shape)
    return None, abs_phi, None

def train(config):
    func = get_func(config.func_name, config.func_kwargs)
    func = func.to(config.device)
    optimizer = torch.optim.Adam(func.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    coord_tensor = generate_coord_tensor(config.shape, config.device)
    shape = [-1] + list(config.shape) + [len(config.shape)]

    losses = torch.zeros((config.epochs))
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad()

        # mod_output = get_model_output(config, func, coord_tensor)
        init = torch.normal(0., config.std, size=(1, 16384, 2), device=config.device)
        abs_phi = nn.Parameter(init)
        mod_output = [None, abs_phi, None]
        loss = negJ_loss(mod_output, coord_tensor, shape)[0]
        loss_mean = loss.mean()

        loss_mean.backward()
        optimizer.step()

        losses[epoch-1] = loss_mean.clone().detach().cpu()
        if epoch % config.cadence == 0 or epoch == 1:
            print(f'Epoch {epoch}/{config.epochs}, loss {loss_mean}')
    
    # os.makedirs(os.path.join(config.log_path, 'res'), exist_ok=True)
    # torch.save(mod_output[1].detach().cpu(), os.path.join(config.log_path, 'res', 'abs_phi.pt'))
    # torch.save(mod_output[1].detach().cpu(), os.path.join(config.log_path, 'res', 'abs_phi.pt'))

def main(**kwargs):
    config = Namespace(**kwargs)
    config.func_kwargs = {
        'layers': [2, 64, 64, 64, 2],
        'weight_init': True,
        'last_init_zero': True,
        'omega': 30,
        'groups': config.time_points - 1
        # 'img_sz': (128, 128),
        # 'smoothing_kernel': 'AK',
        # 'smoothing_win': 15,
        # 'smoothing_pass': 1,
        # 'ds': 2,
        # 'bs': 16,
        # 'use_t': False,
    }
    config.log_path = os.path.join('log/phiJ_reg_test', config.comment)
    train(config)

if __name__ == '__main__':
    device = 'cuda'
    tp = 2
    ts = 0.1

    func_name = 'groupsiren'
    shape = [128, 128]
    std = 0.1

    epochs = 500
    lr = 1e-1
    wd = 0.

    comment = 'test'
    cadence = 10
    main(
        comment=comment,
        time_points=tp,
        step_size=ts,
        device=device,
        cadence=cadence,

        func_name=func_name,
        shape=shape,
        std=std,

        epochs=epochs,
        lr=lr,
        weight_decay=wd,

        solver='euler',
        atol=1e-8,
        rtol=1e-6,
    )