"""End-to-end reconstruction pipeline: data -> init recon -> registration -> eval -> save."""

import os
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from stmr.config import Config, build_velocity_kwargs
from stmr.data.data_load import prepare_inputs
from stmr.eval import evaluate
from stmr.metrics.calc_metrics import calc_init_metrics
from stmr.models.factory import get_func
from stmr.registration import registration
from stmr.utils.log_and_save import save_results
from stmr.utils.logging import get_logger, log_metrics


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed + 1)
    np.random.seed(seed + 2)


def run(config: Config):
    torch.set_num_threads(8)
    set_seed(config.seed)

    config.log_path = os.path.join(config.log_path, config.exp_name)
    os.makedirs(config.log_path, exist_ok=True)

    logger = get_logger(config.log_level)
    logger.info(f"Starting experiment with name {config.exp_name}")
    writer = SummaryWriter(os.path.join(config.log_path, "tensorboard"))

    inputs, eval_inputs = prepare_inputs(config, logger)
    init_metrics, init_imgs = calc_init_metrics(eval_inputs=eval_inputs)
    log_metrics(config, init_metrics, writer, 0, init_imgs, True)

    dims = len(inputs.fixed.shape) - 2
    config.func_kwargs = build_velocity_kwargs(config, dims)

    func = get_func(config.func_name, config.func_kwargs).to(config.device)

    output = registration(config, writer, logger, inputs, eval_inputs, func)
    imgs_to_save = evaluate(config, writer, logger, output, inputs, eval_inputs)
    save_results(config, output, inputs, eval_inputs, imgs_to_save)
    return output
