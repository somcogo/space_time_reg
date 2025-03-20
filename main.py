import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from src.utils import prepare_inputs, save_results, get_logger
from src.registration import registration
from src.eval import evaluate
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
torch.set_num_threads(8)

def main(config):
    logger = get_logger(config.log_level)
    writer = SummaryWriter(os.path.join(config.log_path, 'tensorboard'))
    data = prepare_inputs(config)
    output = registration(config, data, writer, logger)
    # evaluate(config, output)
    save_results(config, output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--exp_name", type=str,
                        dest="exp_name", default='test',
                        help="name of experiment")
    parser.add_argument("--log_path", type=str,
                        dest="log_path", default='log',
                        help="path to save tensorboard logs")
    parser.add_argument("--dataset", type=str,
                        dest="dataset", default='easysyn',
                        help="dataset to use")
    parser.add_argument("--device", type=str,
                        dest="device", default='cuda',
                        help="device to use")
    parser.add_argument("--func_name", type=str,
                        dest="func_name", default='nodeo',
                        help="function to predict the velocity field")
    parser.add_argument("--time_points", type=int,
                        dest="time_points", default=20,
                        help="number of time points")
    parser.add_argument("--time_step", type=float,
                        dest="time_step", default=0.001,
                        help="length of time step")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=0.005,
                        help="learning rate")
    parser.add_argument("--epochs", type=int,
                        dest="epochs", default=100,
                        help="number of epochs")
    parser.add_argument("--solver", type=str,
                        dest="solver", default='euler',
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
                        help="loss weight for gradient magnitude")
    parser.add_argument("--lambda_mag", type=float,
                        dest="lambda_mag", default=1,
                        help="loss weight for v magnitude")
    parser.add_argument("--lambda_st", type=float,
                        dest="lambda_st", default=1,
                        help="loss weight for space-time loss")
    parser.add_argument("--use_nreps", type=bool,
                        dest="use_nreps", default=True,
                        help="Whether to use neural representations to calculate the similarity losses")
    parser.add_argument("--use_t", type=bool,
                        dest="use_t", default=True,
                        help="Use NODER insead of NODEO")
    
    config = parser.parse_args()
    config.func_kwargs = {'img_sz':(128, 128),
                          'smoothing_kernel':'GK',
                          'smoothing_win':15,
                          'smoothing_pass':1,
                          'ds':2,
                          'bs':16,
                          'use_t':config.use_t}
    config.log_path = os.path.join(config.log_path, config.exp_name)
    os.makedirs(config.log_path, exist_ok=True)
    main(config)