"""Staged ablation suite: run the pipeline component-by-component to find where it breaks.

See stages.py for the ladders (1/2: S0..S5 each; motion: S0, S0B, S0K, S0R). Entry point:
``python -m stmr.ablation run``.
"""

from .runner import build_config, collect_ladder, run_ladder, run_stage
from .stages import LADDERS, StageSpec, resolve_stage_ids

__all__ = [
    "LADDERS", "StageSpec", "resolve_stage_ids",
    "build_config", "run_stage", "run_ladder", "collect_ladder",
]
