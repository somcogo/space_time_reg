"""Single place for run logging: a stream logger plus TensorBoard scalar/image logging."""

import logging

from torch.utils.tensorboard import SummaryWriter


def get_logger(level: str = "info") -> logging.Logger:
    logger = logging.getLogger("stmr")
    logger.handlers.clear()
    level = logging.INFO if level == "info" else logging.DEBUG

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    handler.setLevel(level)

    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def log_metrics(config, metrics: dict, writer: SummaryWriter, epoch: int,
                imgs_to_log: dict, last_val: bool = False) -> None:
    for k, v in metrics.items():
        writer.add_scalar(k, v, epoch)
    if imgs_to_log is not None and (config.debug or last_val):
        for k, v in imgs_to_log.items():
            writer.add_image(k, v, epoch, dataformats="HWC")
    writer.flush()
