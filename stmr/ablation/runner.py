"""Build a Config per stage, run the pipeline, and collect the metric rows."""

from __future__ import annotations

import os

from stmr.config import Config

from .report import extract_row
from .stages import BASE, STAGE_BY_ID, StageSpec


def build_config(spec: StageSpec, epochs: int, device: str, outdir: str,
                 overrides: dict | None = None) -> Config:
    """Materialise a stage's Config: BASE (or its YAML) overlaid with the stage overrides.

    ``overrides`` is an optional global override dict applied last (after the per-stage
    overrides), e.g. ``{"time_points": 12}`` to run the whole ladder at a different frame
    count. It is applied to YAML-backed stages too, so S6's full-pipeline config picks it up.

    The pipeline writes res.pt to ``{log_path}/{exp_name}/res.pt`` (run() joins the two), so
    log_path=outdir + exp_name=stage.id lands each stage's result at ``outdir/<id>/res.pt``.
    """
    if spec.yaml is not None:
        config = Config.from_yaml(spec.yaml)
        for key, val in spec.overrides.items():
            setattr(config, key, val)
    else:
        merged = {**BASE, **spec.overrides}
        config = Config.from_dict(merged)

    for key, val in (overrides or {}).items():
        setattr(config, key, val)

    config.device = device
    config.epochs = epochs
    config.exp_name = spec.id
    config.log_path = outdir
    return config.finalize()


def stage_res_path(spec: StageSpec, outdir: str) -> str:
    return os.path.join(outdir, spec.id, "res.pt")


def run_stage(spec: StageSpec, epochs: int, device: str, outdir: str,
              overrides: dict | None = None) -> dict:
    """Run one stage end-to-end and return its metric row (also persisted in res.pt)."""
    config = build_config(spec, epochs, device, outdir, overrides)
    from stmr.pipeline import run  # deferred: importing torch/pipeline is slow

    run(config)
    return extract_row(spec, stage_res_path(spec, outdir))


def run_ladder(stage_ids: list[str], epochs: int, device: str, outdir: str,
               overrides: dict | None = None) -> list[dict]:
    """Run the stages in order, threading each stage's row into the next stage's pass_fn."""
    rows, prev = [], None
    for sid in stage_ids:
        spec = STAGE_BY_ID[sid]
        row = run_stage(spec, epochs, device, outdir, overrides)
        row["passed"] = spec.pass_fn(row, prev) if spec.pass_fn is not None else None
        rows.append(row)
        prev = row
    return rows
