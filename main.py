import argparse
import os
import random

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.utils.log_and_save import save_results, log_metrics
from src.utils.logger import get_logger
from src.registration import registration
from src.eval import evaluate
from src.data.data_load import prepare_inputs
from src.models.factory import get_func
from src.metrics.calc_metrics import calc_init_metrics

torch.set_num_threads(8)

def main():
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
    parser.add_argument("--slice_number", type=int,
                        dest="slice_number", default=0,
                        help="Which slice of the CMRxRecon image to use")
    parser.add_argument("--step_size", type=float,
                        dest="step_size", default=0.001,
                        help="size of time step")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=0.01,
                        help="learning rate")
    parser.add_argument("--init_lr", type=float,
                        dest="init_lr", default=0.1,
                        help="reconstruction learning rate for initial recon")
    parser.add_argument("--recon_lr", type=float,
                        dest="recon_lr", default=0.1,
                        help="reconstruction learning rate")
    parser.add_argument("--weight_decay", type=float,
                        dest="weight_decay", default=0.1,
                        help="weight decay for velocity network")
    parser.add_argument("--epochs", type=int,
                        dest="epochs", default=100,
                        help="number of epochs")
    parser.add_argument("--recon_epochs", type=int,
                        dest="recon_epochs", default=200,
                        help="number of epochs for initial reconstruction")
    parser.add_argument("--solver", type=str,
                        dest="solver", default='rk4',
                        help="ode solver method")
    parser.add_argument("--log_level", type=str,
                        dest="log_level", default='info',
                        help="logging level")
    parser.add_argument("--log_cadence", type=int,
                        dest="log_cadence", default=10,
                        help="how often to log to terminal during registration")
    
    parser.add_argument("--lambda_st", type=float,
                        dest="lambda_st", default=1,
                        help="loss weight for similarity loss")
    parser.add_argument("--lambda_negJ", type=float,
                        dest="lambda_negJ", default=0.1,
                        help="loss weight for neg J")
    parser.add_argument("--lambda_grd", type=float,
                        dest="lambda_grd", default=1,
                        help="loss weight for velocity gradient L2 norm")
    parser.add_argument("--lambda_lap", type=float,
                        dest="lambda_lap", default=1,
                        help="loss weight for velocity laplacian L2 norm")
    parser.add_argument("--lambda_pgr", type=float,
                        dest="lambda_pgr", default=1,
                        help="loss weight for flow gradient L2 norm")
    parser.add_argument("--lambda_hel", type=float,
                        dest="lambda_hel", default=1,
                        help="loss weight for hyper elastic loss")
    parser.add_argument("--lambda_recon", type=float,
                        dest="lambda_recon", default=1,
                        help="loss weight for the reconstruction regularizer")
    parser.add_argument("--lambda_init_recon", type=float,
                        dest="lambda_init_recon", default=1,
                        help="loss weight for the reconstruction regularizer during initial recon")
    parser.add_argument("--recon_scale", type=float,
                        dest="recon_scale", default=0.1,
                        help="scale for the reconstruction regularizer")
    
    parser.add_argument("--use_nreps", action=argparse.BooleanOptionalAction,
                        dest="use_nreps", default=True,
                        help="Whether to use neural representations to calculate the similarity losses")
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
    parser.add_argument("--atol", type=float,
                        dest="atol", default=1e-9,
                        help="Absolute tolerance for the ODE solver")
    parser.add_argument("--rtol", type=float,
                        dest="rtol", default=1e-7,
                        help="Relative tolerance for the ODE solver")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction,
                        dest="debug", default=False,
                        help="Use debug mode. Increases runtime significantly")
    
    parser.add_argument("--schedule", type=list,
                        dest="schedule", default=None,
                        help="Epochs when downsample")
    parser.add_argument("--downsamples", type=list,
                        dest="downsamples", default=[8, 4, 2, 1],
                        help="Factor to downsample by")
    
    parser.add_argument("--detach_grads", action=argparse.BooleanOptionalAction,
                        dest="detach_grads", default=True,
                        help="Used in nmAPG for recon init")
    parser.add_argument("--use_nmapg", action=argparse.BooleanOptionalAction,
                        dest="use_nmapg", default=False,
                        help="Used nmAPG for initial recon")
    parser.add_argument("--random_mask", action=argparse.BooleanOptionalAction,
                        dest="random_mask", default=False,
                        help="Use randomized kspace mask")
    parser.add_argument("--init", type=str,
                        dest="init", default='zero',
                        help="How the value of the recon when starting the initial reconstruction")
    parser.add_argument("--reg", type=str,
                        dest="reg", default='learned',
                        help="Type of regularizer to use")
    parser.add_argument("--reg_alpha", type=float,
                        dest="reg_alpha", default=0.,
                        help="Alpha to use for the regularizer")
    parser.add_argument("--tol", type=float,
                        dest="tol", default=1e-4,
                        help="Tolerance to use in nmAPG")
    parser.add_argument("--factor", type=int,
                        dest="factor", default=4,
                        help="Downsampling factor for CMR kspace mask")
    parser.add_argument("--mask", type=str,
                        dest="mask", default='st',
                        help="How to generate kspace mask")
    parser.add_argument("--init_loss", type=str,
                        dest="init_loss", default='l2',
                        help="Loss used for initial recon")
    parser.add_argument("--init_reg_abs", action=argparse.BooleanOptionalAction,
                        dest="init_reg_abs", default=False,
                        help="Use the abs value of the image for the regularizer during initial recon")
    parser.add_argument("--tm", type=int,
                        dest="tm", default=0,
                        help="Time frame to use as moving image")
    
    config = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_number

    torch.manual_seed(config.seed)
    random.seed(config.seed + 1)
    np.random.seed(config.seed + 2)

    if 'siren' in config.func_name:
        layers = [3] + config.siren_depth * [config.siren_dim] + [3]
        func_kwargs = {'layers':layers,
                       'omega':config.siren_omega,
                    #    'last_init_zero':True}
                       'last_init_zero':'cmr' in config.dataset}
    elif 'wire' in config.func_name:
        layers = [3] + config.siren_depth * [config.siren_dim] + [3]
        func_kwargs = {'layers':layers,
                       'omega':config.siren_omega,
                       'scale':config.wire_scale}
    if config.recon_scale == 0:
        config.recon_scale = None
    
    config.func_kwargs = func_kwargs
    config.log_path = os.path.join(config.log_path, config.exp_name)
    os.makedirs(config.log_path, exist_ok=True)
    config.step_size = None if config.step_size == 0 else config.step_size
    config.schedule = [1] if config.schedule == None else config.schedule
    config.downsamples = [1]

    logger = get_logger(config.log_level)
    logger.info(f'Starting experiment with name {config.exp_name}')
    writer = SummaryWriter(os.path.join(config.log_path, 'tensorboard'))



    inputs, eval_inputs = prepare_inputs(config, logger)
    init_metrics, init_imgs = calc_init_metrics(eval_inputs=eval_inputs)
    log_metrics(config, init_metrics, writer, 0, init_imgs, True)

    fixed = inputs[2]
    dims = len(fixed.shape) - 2
    config.func_kwargs['layers'][0] = dims
    config.func_kwargs['layers'][-1] = dims
        
    func = get_func(config.func_name, config.func_kwargs)
    func = func.to(config.device)



    output = registration(config, writer, logger, inputs, eval_inputs, func)
    imgs_to_save = evaluate(config, writer, logger, output, inputs, eval_inputs)
    save_results(config, output, eval_inputs, imgs_to_save)

if __name__ == '__main__':
    main()