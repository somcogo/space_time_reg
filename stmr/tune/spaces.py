"""Search spaces: functions ``trial -> {config field: value}``.

Each tuning parameter is a config field sampled directly (no ratios or reparametrisations),
except ``use_logdetJ``: a log-uniform range cannot reach 0, so whether the logdetJ term is on
at all is its own (categorical) parameter, and lambda_logdetJ is only sampled when it is.

The stages tune in sequence (see configs/tune/): each later stage fixes the earlier stage's
pick in its YAML ``overrides`` and samples only its own space.

    lambdas  stage A: the loss weights and both learning rates, at the base config's SIREN
    siren    stage B: the velocity network, the ODE step and the velocity lr (which couples to
             omega and width), at stage A's loss weights and recon_lr

Everything is define-by-run; the function is also replayed on fixed params
(``optuna.trial.FixedTrial``) to rebuild a trial's overrides, so it must only use
``trial.suggest_*``. Add a space by writing a function and registering it in SPACES.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _lr(trial) -> float:
    return trial.suggest_float("lr", 3e-5, 2e-3, log=True)  # velocity net (cosine to lr_min 1e-5)


def lambdas(trial, **_) -> dict[str, Any]:
    """Stage A: lambda_rl2, lambda_grad_phi, lambda_logdetJ (or off), lambda_recon, lr, recon_lr.

    lambda_st (the data term) stays 1: Adam is invariant to a common scale of the loss, so only
    the other weights relative to it matter. lambda_recon also weighs the prior in the nmAPG
    init, so it changes the init too (scores are relative to the zero-filled recon for that).
    """
    out = {
        "lambda_rl2": trial.suggest_float("lambda_rl2", 1e1, 1e5, log=True),
        "lambda_grad_phi": trial.suggest_float("lambda_grad_phi", 1e-3, 1e1, log=True),
        "lambda_logdetJ": 0.0,
        "lambda_recon": trial.suggest_float("lambda_recon", 0.02, 0.5, log=True),
        "recon_lr": trial.suggest_float("recon_lr", 1e-4, 3e-2, log=True),
        "lr": _lr(trial),
    }
    if trial.suggest_categorical("use_logdetJ", [False, True]):
        out["lambda_logdetJ"] = trial.suggest_float("lambda_logdetJ", 1e-3, 1e1, log=True)
    return out


def siren(trial, **_) -> dict[str, Any]:
    """Stage B: SIREN depth / width / omega, the Euler step and the velocity lr.

    The ODE runs over t in [0, 1], so step_size 1 / 0.5 / 0.25 / 0.1 = 1 / 2 / 4 / 10 Euler
    steps (cost ~ steps; 1 step is a plain displacement field). Memory grows with
    steps x width x depth without the adjoint: the largest corner can OOM, which run.oom_policy
    'auto' scores as a crash when the GPU was otherwise free.
    """
    return {
        "siren_depth": trial.suggest_int("siren_depth", 2, 5),
        "siren_dim": trial.suggest_categorical("siren_dim", [32, 64, 128, 256]),
        "siren_omega": trial.suggest_int("siren_omega", 10, 60, step=5),
        "step_size": trial.suggest_categorical("step_size", [1.0, 0.5, 0.25, 0.1]),
        "lr": _lr(trial),
    }


def net(trial, **_) -> dict[str, Any]:
    """Architecture grid (configs/tune/arch_v1.yaml): depth / width / velocity lr only, omega and
    the Euler step fixed in the YAML. Wider bounds than `siren` (depth up to 12); meant to be run
    on enqueued grid points without pruning, so every point is fully evaluated."""
    return {
        "siren_depth": trial.suggest_int("siren_depth", 2, 12),
        "siren_dim": trial.suggest_categorical("siren_dim", [32, 64, 128, 256]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
    }


def refine(trial, center: dict[str, float], factor: float = 4.0,
           factors: dict[str, float] | None = None) -> dict[str, Any]:
    """Stage C: a narrow re-tune around earlier picks (``space_args`` in the YAML): every
    positive ``center`` value is sampled log-uniformly in [c / f, c * f] (f = factors[key] or
    factor); a 0 stays 0 (e.g. lambda_logdetJ switched off). Parameter names = config fields."""
    out: dict[str, Any] = {}
    for key, c in center.items():
        f = (factors or {}).get(key, factor)
        out[key] = trial.suggest_float(key, c / f, c * f, log=True) if c > 0 else c
    return out


SPACES: dict[str, Callable[..., dict[str, Any]]] = {
    "lambdas": lambdas,
    "siren": siren,
    "net": net,
    "refine": refine,
}


def overrides_from_params(space: str, params: dict[str, Any],
                          args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Rebuild the config overrides of a finished trial from its params."""
    import optuna

    return SPACES[space](optuna.trial.FixedTrial(params), **(args or {}))
