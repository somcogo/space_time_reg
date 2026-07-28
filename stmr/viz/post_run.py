"""Generate the tb_gif and deform-flow visualisations for a just-finished run.

Called once at the end of the main pipeline (see stmr.pipeline.run), after res.pt and the
tensorboard event files are on disk. Writes into ``<log_path>/gifs`` (per-tag
training-evolution GIFs, one per tensorboard image tag) and ``<log_path>/figs`` (the
learned-deformation sanity-check figures), next to the existing pdfs/ and imgs/ folders.

Failures here are logged and swallowed rather than raised: visualisation is a diagnostic
extra, and a plotting error should never take down a run whose actual results (res.pt,
metrics) are already saved.
"""

from __future__ import annotations

import logging
import os

from stmr.viz.deform import visualise as visualise_deform
from stmr.viz.tb_gif import list_image_tags, resolve_tb_dir, tag_to_gif


def generate_tb_gifs(logdir: str, out_dir: str, fps: float = 2.0,
                     logger: logging.Logger | None = None) -> list[str]:
    """Turn every image tag logged to TensorBoard under logdir into a GIF in out_dir."""
    log = logger or logging.getLogger(__name__)
    tb_dir = resolve_tb_dir(logdir)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for tag in list_image_tags(tb_dir):
        try:
            paths.append(tag_to_gif(tb_dir, tag, out_dir, fps, label=True))
        except Exception:
            log.exception(f"tb_gif failed for tag {tag!r} in {tb_dir}")
    return paths


def generate_deform_figs(res_path: str, out_dir: str, device: str = "cpu",
                         logger: logging.Logger | None = None) -> None:
    """Run the deform-flow sanity-check visualisation for a saved run's res.pt."""
    log = logger or logging.getLogger(__name__)
    try:
        visualise_deform(res_path, out_dir, device=device)
    except Exception:
        log.exception(f"deform visualisation failed for {res_path}")


def generate_run_visuals(config, logger: logging.Logger | None = None) -> None:
    """Populate <log_path>/gifs and <log_path>/figs for a just-finished run."""
    res_path = os.path.join(config.log_path, "res.pt")
    generate_tb_gifs(config.log_path, os.path.join(config.log_path, "gifs"), logger=logger)
    generate_deform_figs(res_path, os.path.join(config.log_path, "figs"),
                        device=config.device, logger=logger)
