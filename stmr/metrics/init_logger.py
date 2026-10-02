"""TensorBoard logging for the initial-reconstruction (nmAPG) solve.

``InitTBLogger`` *is* the nmAPG callback (see stmr.data.nmapg): construct it with a writer and
hand it to ``init_using_nmAPG(..., callback=...)``. It composes the modular dicts from
stmr.metrics.init_metrics and writes them with the pipeline's own ``log_metrics``.

Two properties make it reusable by the full pipeline as-is:
  * every tag is written under ``prefix`` (default ``init/``), so the init curves cannot collide
    with the registration's epoch-indexed tags and the TB layout is identical whether the init
    ran standalone or inside a full run;
  * it accumulates ``history`` ({tag: [values]} + steps), which feeds the figures
    (stmr.viz.curves.plot_loss_curves) and gets stored in res.pt.

It is strictly an observer: it never touches the iterate, so a run with and without it produces
a bit-identical reconstruction.
"""

from __future__ import annotations

import torch

from stmr.metrics.init_metrics import init_images, init_quality_metrics, init_scalar_metrics
from stmr.utils.logging import log_metrics


class InitTBLogger:
    def __init__(self, writer, gt_im: torch.Tensor, mask: torch.Tensor, fixed: torch.Tensor,
                 config, log_every: int = 5, prefix: str = "init/", log_images: bool = True):
        self.writer = writer
        self.gt_im = gt_im
        self.mask = mask
        self.fixed = fixed
        self.config = config
        self.log_every = max(1, int(log_every))
        self.prefix = prefix
        self.log_images = log_images
        self.history: dict[str, list[float]] = {}
        self.steps: list[int] = []

    def _record(self, step: int, metrics: dict) -> None:
        self.steps.append(step)
        for k, v in metrics.items():
            self.history.setdefault(k, []).append(float(v))

    def __call__(self, i: int, x: torch.Tensor, stats: dict) -> None:
        # i == -1 is the initial point; shift so TB steps start at 0.
        step = i + 1
        with torch.no_grad():
            xd = x.detach()
            metrics = init_scalar_metrics(xd, stats, self.fixed, self.mask)
            imgs = None
            if step % self.log_every == 0:
                metrics.update(init_quality_metrics(xd, self.gt_im))
                if self.log_images:
                    imgs = {f"{self.prefix}{k}": v
                            for k, v in init_images(xd, self.gt_im, self.fixed,
                                                    self.mask).items()}
        self._record(step, metrics)
        # log_metrics gates images on config.debug or last_val; force them through when present.
        log_metrics(self.config, {f"{self.prefix}{k}": v for k, v in metrics.items()},
                    self.writer, step, imgs, last_val=imgs is not None)
