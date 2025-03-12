import argparse
import os

from torch.utils.tensorboard import SummaryWriter

from src.utils import prepare_inputs, save_results
from src.registration import registration
from src.eval import evaluate
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def main(config):
    # writer = SummaryWriter(config.log_path)
    writer = None
    data = prepare_inputs(config)
    output = registration(config, data, writer)
    # evaluate(config, output)
    # save_results(config, output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--log_path", type=str,
                        dest="log_path", default='log',
                        help="path to save tensorboard logs")
    parser.add_argument("--dataset", type=str,
                        dest="dataset", default='easysyn',
                        help="dataset to use")
    parser.add_argument("--device", type=str,
                        dest="device", default='cpu',
                        help="device to use")
    parser.add_argument("--func_name", type=str,
                        dest="func_name", default='nodeo',
                        help="function to predict the velocity field")
    parser.add_argument("--time_points", type=int,
                        dest="time_points", default=19,
                        help="number of time points")
    parser.add_argument("--time_step", type=float,
                        dest="time_step", default=0.001,
                        help="length of time step")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=0.005,
                        help="learning rate")
    parser.add_argument("--epochs", type=int,
                        dest="epochs", default=1,
                        help="number of epochs")
    parser.add_argument("--solver", type=str,
                        dest="solver", default='euler',
                        help="ode solver method")
    
    config = parser.parse_args()
    config.func_kwargs = {'img_sz':(128, 128),
                          'smoothing_kernel':'GK',
                          'smoothing_win':15,
                          'smoothing_pass':1,
                          'ds':2,
                          'bs':16}
    main(config)