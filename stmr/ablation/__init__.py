"""Staged ablation suite: run the pipeline component-by-component to find where it breaks.

See stages.py for the ladder (S0..S5). Entry point: ``python -m stmr.ablation run``.
"""

from .runner import build_config, run_ladder, run_stage
from .stages import STAGE_BY_ID, STAGES, StageSpec, resolve_stage_ids

__all__ = [
    "STAGES", "STAGE_BY_ID", "StageSpec", "resolve_stage_ids",
    "build_config", "run_stage", "run_ladder",
]
