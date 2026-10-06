"""Run the initial reconstruction (nmAPG) alone, with full logging and figures.

Reuses the pipeline's own data path (``prepare_data``) and its logging components
(``InitTBLogger``, ``log_metrics``, ``plot_loss_curves``, ``generate_tb_gifs``) so nothing here
is sweep-specific beyond the orchestration -- the same logging lands in a full run once
``prepare_inputs(config, logger, writer)`` is given a writer.
"""

from __future__ import annotations

import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from stmr.config import Config
from stmr.data.data_load import prepare_data
from stmr.data.recon_init import init_using_nmAPG
from stmr.metrics.init_logger import InitTBLogger
from stmr.metrics.init_metrics import init_quality_metrics, init_scalar_metrics
from stmr.utils.logging import get_logger
from stmr.viz.post_run import generate_tb_gifs

from .figures import make_init_figures


def run_init(config: Config) -> dict:
    """Solve the init for one config; write res.pt + figures + gifs; return a summary row."""
    config.log_path = os.path.join(config.log_path, config.exp_name)
    os.makedirs(config.log_path, exist_ok=True)
    logger = get_logger(config.log_level)
    logger.info(f"init run: {config.exp_name} "
                f"(lambda_init_recon={config.lambda_init_recon}, recon_scale={config.recon_scale})")
    writer = SummaryWriter(os.path.join(config.log_path, "tensorboard"))

    data = prepare_data(config)
    mask = data.kspace_mask.to(config.device)
    # nmAPG subsets the batch as frames converge, so the mask rides along inside y (see
    # prepare_inputs for the same packing).
    packed = torch.cat([data.fixed, mask.to(data.fixed)], dim=1)

    cb = InitTBLogger(writer, data.gt_im, mask, data.fixed, config,
                      log_every=config.log_cadence)
    t0 = time.time()
    recon, _ = init_using_nmAPG(config, data.init, packed, data.full_forw,
                                data.full_adj, logger, callback=cb)
    wall = time.time() - t0
    logger.info(f"init solved in {wall:.1f}s over {len(cb.steps)} logged iterations")

    # final metrics (full quality set, regardless of the logging cadence)
    with torch.no_grad():
        final = init_scalar_metrics(recon.detach(), {"data_fit": cb.history["energy/data_fit"][-1],
                                                     "reg": cb.history["energy/reg"][-1],
                                                     "res": cb.history["conv/residual"][-1],
                                                     "L": cb.history["conv/L"][-1],
                                                     "n_active": cb.history["conv/n_active"][-1]},
                                    data.fixed, mask)
        final.update(init_quality_metrics(recon.detach(), data.gt_im))

    torch.save({
        "recon": recon.detach().cpu(), "init": data.init.detach().cpu(),
        "gt_im": data.gt_im.detach().cpu(), "fixed": data.fixed.detach().cpu(),
        "mask": mask.detach().cpu(),
        "history": cb.history, "steps": cb.steps, "final": final,
        "wall_time": wall, "config": config.to_dict(),
    }, os.path.join(config.log_path, "res.pt"))

    make_init_figures(cb.history, cb.steps, recon.detach(), data.gt_im, data.fixed, mask,
                      os.path.join(config.log_path, "figs"))
    try:
        generate_tb_gifs(config.log_path, os.path.join(config.log_path, "gifs"), logger=logger)
    except Exception:
        logger.exception("tb gif generation failed (non-fatal)")

    row = {"exp_name": config.exp_name, "lambda_init_recon": config.lambda_init_recon,
           "recon_scale": config.recon_scale, "iters": len(cb.steps), "wall_time": wall}
    row.update({k: float(v) for k, v in final.items()})
    return row
