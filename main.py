import argparse
import os
import random
from typing import Union
from types import NoneType

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.utils import prepare_inputs, save_results, get_logger
from src.registration import registration
from src.eval import evaluate
torch.set_num_threads(8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--exp_name", type=str,
                        dest="exp_name", default='test/test',
                        help="name of experiment")
    parser.add_argument("--log_path", type=str,
                        dest="log_path", default='log',
                        help="path to save tensorboard logs")
    parser.add_argument("--dataset", type=str,
                        dest="dataset", default='easy',
                        help="dataset to use")
    parser.add_argument("--device", type=str,
                        dest="device", default='cuda',
                        help="device to use")
    parser.add_argument("--func_name", type=str,
                        dest="func_name", default='siren',
                        help="function to predict the velocity field")
    parser.add_argument("--time_points", type=int,
                        dest="time_points", default=20,
                        help="number of time points")
    parser.add_argument("--start_frame", type=int,
                        dest="start_frame", default=0,
                        help="Index of first time frame")
    parser.add_argument("--step_size", type=float,
                        dest="step_size", default=0.001,
                        help="size of time step")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=0.01,
                        help="learning rate")
    parser.add_argument("--epochs", type=int,
                        dest="epochs", default=100,
                        help="number of epochs")
    parser.add_argument("--solver", type=str,
                        dest="solver", default='rk4',
                        help="ode solver method")
    parser.add_argument("--log_level", type=str,
                        dest="log_level", default='info',
                        help="logging level")
    parser.add_argument("--log_cadence", type=int,
                        dest="log_cadence", default=10,
                        help="how ofter on log loss")
    
    parser.add_argument("--lambda_negJ", type=float,
                        dest="lambda_negJ", default=0.1,
                        help="loss weight for neg J")
    parser.add_argument("--lambda_smt", type=float,
                        dest="lambda_smt", default=1,
                        help="loss weight for flow gradient L2 norm")
    parser.add_argument("--lambda_mag", type=float,
                        dest="lambda_mag", default=1,
                        help="loss weight for v magnitude")
    parser.add_argument("--lambda_st", type=float,
                        dest="lambda_st", default=1,
                        help="loss weight for space-time loss")
    parser.add_argument("--lambda_grd", type=float,
                        dest="lambda_grd", default=1,
                        help="loss weight for velocity gradient L2 norm")
    parser.add_argument("--lambda_lap", type=float,
                        dest="lambda_lap", default=1,
                        help="loss weight for velocity laplacian L2 norm")
    parser.add_argument("--lambda_pgr", type=float,
                        dest="lambda_pgr", default=1,
                        help="loss weight for flow gradient L2 norm")
    
    parser.add_argument("--use_nreps", action=argparse.BooleanOptionalAction,
                        dest="use_nreps", default=True,
                        help="Whether to use neural representations to calculate the similarity losses")
    parser.add_argument("--use_t", action=argparse.BooleanOptionalAction,
                        dest="use_t", default=False,
                        help="Use NODER insead of NODEO")
    parser.add_argument("--const_phi", action=argparse.BooleanOptionalAction,
                        dest="const_phi", default=False,
                        help="Use constant phi between time points")
    parser.add_argument("--siren_depth", type=int,
                        dest="siren_depth", default=3,
                        help="Number of hidden layers in the siren network")
    parser.add_argument("--siren_dim", type=int,
                        dest="siren_dim", default=256,
                        help="Hidden dimension in the siren network")
    parser.add_argument("--siren_omega", type=int,
                        dest="siren_omega", default=30,
                        help="Omega used in siren network")
    parser.add_argument("--wire_scale", type=int,
                        dest="wire_scale", default=10,
                        help="Scale used in wire network")
    parser.add_argument("--loss", type=str,
                        dest="loss", default='ngf',
                        help="Loss function to use")
    parser.add_argument("--gpu_number", type=str,
                        dest="gpu_number", default="0",
                        help="Which gpu to use")
    parser.add_argument("--seed", type=int,
                        dest="seed", default="0",
                        help="Set manual seed")
    parser.add_argument("--fin_diff_grad", action=argparse.BooleanOptionalAction,
                        dest="fin_diff_grad", default=False,
                        help="Use finite differences to calculate grad loss instead of autograd")
    parser.add_argument("--use_grid", action=argparse.BooleanOptionalAction,
                        dest="use_grid", default=True,
                        help="Use grid to evaluate the similarity loss. If False, use random points")
    parser.add_argument("--autograd_grid", action=argparse.BooleanOptionalAction,
                        dest="autograd_grid", default=True,
                        help="Use grid to evaluate the grad loss. If False, use random points. Only for autograd")
    parser.add_argument("--atol", type=float,
                        dest="atol", default=1e-9,
                        help="Absolute tolerance for the ODE solver")
    parser.add_argument("--rtol", type=float,
                        dest="rtol", default=1e-7,
                        help="Relative tolerance for the ODE solver")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction,
                        dest="debug", default=False,
                        help="Use debug mode. Increases runtime significantly")
    
    config = parser.parse_args()

    torch.manual_seed(config.seed)
    random.seed(config.seed + 1)
    np.random.seed(config.seed + 2)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_number

    img_sz = (168, 168) if config.dataset in ['rot', 'rot_slow', 'rot_slow2'] else (128, 128)
    if config.func_name == 'nodeo':
        func_kwargs = {'img_sz':img_sz,
                       'smoothing_kernel':'GK',
                       'smoothing_win':15,
                       'smoothing_pass':1,
                       'ds':2,
                       'bs':16,
                       'use_t':config.use_t}
    elif 'siren' in config.func_name:
        layers = [3] + config.siren_depth * [config.siren_dim] + [3]
        func_kwargs = {'layers':layers,
                       'omega':config.siren_omega}
    elif 'wire' in config.func_name:
        layers = [3] + config.siren_depth * [config.siren_dim] + [3]
        func_kwargs = {'layers':layers,
                       'omega':config.siren_omega,
                       'scale':config.wire_scale}
    config.func_kwargs = func_kwargs
    config.log_path = os.path.join(config.log_path, config.exp_name)
    os.makedirs(config.log_path, exist_ok=True)
    config.step_size = None if config.step_size == 0 else config.step_size
    
    logger = get_logger(config.log_level)
    logger.info(f'Starting experiment with name {config.exp_name}')
    writer = SummaryWriter(os.path.join(config.log_path, 'tensorboard'))
    data = prepare_inputs(config)
    output = registration(config, data, writer, logger)
    # evaluate(config, output)
    save_results(config, output)