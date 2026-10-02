"""Init-only experiments: run the nmAPG initial reconstruction alone, with full logging.

The logging/metric/figure components live outside this package on purpose
(stmr.metrics.init_metrics, stmr.metrics.init_logger, stmr.viz.curves) so the full pipeline can
adopt them later; this package holds only the sweep orchestration.

Entry point: ``python -m stmr.initsweep run|sweep|aggregate``.
"""

from .run import run_init

__all__ = ["run_init"]
