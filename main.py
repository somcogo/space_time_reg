import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from src.utils import prepare_inputs, save_results, get_logger
from src.registration import registration
from src.eval import evaluate
torch.set_num_threads(8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--exp_name", type=str,
                        dest="exp_name", default='test',
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
    parser.add_argument("--time_step", type=float,
                        dest="time_step", default=0.001,
                        help="length of time step")
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
    parser.add_argument("--use_nreps", type=bool,
                        dest="use_nreps", default=True,
                        help="Whether to use neural representations to calculate the similarity losses")
    parser.add_argument("--use_t", type=bool,
                        dest="use_t", default=False,
                        help="Use NODER insead of NODEO")
    parser.add_argument("--const_phi", type=bool,
                        dest="const_phi", default=False,
                        help="Use constant phi between time points")
    parser.add_argument("--siren_depth", type=int,
                        dest="siren_depth", default=3,
                        help="Number of hidden layers in the siren network")
    parser.add_argument("--siren_dim", type=int,
                        dest="siren_dim", default=256,
                        help="Hidden dimension in the siren network")
    parser.add_argument("--siren_omega", type=int,
                        dest="siren_omega", default=32,
                        help="Omega used in siren network")
    parser.add_argument("--loss", type=str,
                        dest="loss", default='ngf',
                        help="Loss function to use")
    parser.add_argument("--gpu_number", type=str,
                        dest="gpu_number", default="0",
                        help="Which gpu to use")
    
    config = parser.parse_args()

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
    else:
        layers = [3] + config.siren_depth * [config.siren_dim] + [3]
        func_kwargs = {'layers':layers,
                       'omega':config.siren_omega}
    config.func_kwargs = func_kwargs
    config.log_path = os.path.join(config.log_path, config.exp_name)
    os.makedirs(config.log_path, exist_ok=True)
    
    logger = get_logger(config.log_level)
    logger.info(f'Starting experiment with name {config.exp_name}')
    writer = SummaryWriter(os.path.join(config.log_path, 'tensorboard'))
    data = prepare_inputs(config)
    output = registration(config, data, writer, logger)
    # evaluate(config, output)
    save_results(config, output)