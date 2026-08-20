"""Two-stage coarse-to-fine motion decomposition.

Trains two SIREN velocity nets sequentially to register consecutive GT cine frames:
  stage 1: a small, regular phi1 captures the bulk motion  (warp(I_i, phi1) ~= I_{i+1});
  stage 2: a frozen phi1 + a larger phi2 capture the residual (warp(warp(I_i,phi1),phi2) ~= I_{i+1}).
Plus single-net baselines (large-alone, small-alone). Headline output: how much of the motion
phi1 vs phi2 is responsible for. Registration only -- images are fixed GT, nothing is
reconstructed. Kept separate from the main pipeline. Entry point: ``python -m stmr.decomp``.
"""

from .params import DecompParams

__all__ = ["DecompParams", "run_decomp", "analyze", "compute_responsibility", "make_figures"]


def __getattr__(name):
    # Lazy imports so `import stmr.decomp` (and DecompParams) stays cheap and avoids pulling
    # torch/odeint/matplotlib unless a driver is actually used.
    if name in ("run_decomp", "run_two_stage_core"):
        from .train import run_decomp, run_two_stage_core
        return {"run_decomp": run_decomp, "run_two_stage_core": run_two_stage_core}[name]
    if name == "compute_responsibility":
        from .metrics import compute_responsibility
        return compute_responsibility
    if name in ("make_figures", "analyze"):
        from .figures import make_figures
        if name == "make_figures":
            return make_figures
        from .analyze import analyze
        return analyze
    raise AttributeError(name)
