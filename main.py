import argparse
import os

from torch.utils.tensorboard import SummaryWriter

from src.utils import prepare_inputs, save_results
from src.registration import registration
from src.eval import evaluate
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def main(config):
    writer = SummaryWriter(config.log_path)
    data = prepare_inputs(config)
    output = registration(config, data, writer)
    evaluate(config, output)
    save_results(config, output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--savepath", type=str,
                        dest="savepath", default='./result',
                        help="path for saving results")
    
    config = parser.parse_args()
    main(config)