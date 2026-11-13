"""Study tables and plots: ``python -m stmr.tune report <spec>``.

Writes into <study_dir>/report/:
  trials.csv   one row per trial: state, pruned, value (mean score over the tuning cases; for
               pruned trials the paired estimate), mean gain over the init, params, overrides
  cases.csv    trial x case score matrix (blank = not run: pruned before it, or still running)
  *.png        optimization history, parameter importances, parallel coordinates, slices
"""

from __future__ import annotations

import csv
import os
import warnings
from typing import Any

import numpy as np
from optuna.trial import FrozenTrial, TrialState

from stmr.tune.spec import StudySpec


def case_attrs(t: FrozenTrial, spec: StudySpec) -> dict[str, dict[str, Any]]:
    return {c: t.user_attrs[f"case/{c}"] for c in spec.case_names("tune")
            if f"case/{c}" in t.user_attrs}


def trial_row(t: FrozenTrial, spec: StudySpec) -> dict[str, Any]:
    cases = case_attrs(t, spec)
    m = spec.metric
    gains = [c[f"{m}/final"] - c[f"{m}/init"] for c in cases.values()
             if f"{m}/final" in c and f"{m}/init" in c]
    runtime = sum(c.get("runtime_s", 0) for c in cases.values())
    return {
        "number": t.number,
        "state": t.state.name,
        "baseline": t.user_attrs.get("baseline", ""),
        "pruned": t.user_attrs.get("pruned", ""),
        "n_cases": len(cases),
        "value": t.value if t.state == TrialState.COMPLETE else "",
        "gain_over_init": float(np.mean(gains)) if gains else "",
        "above_init": f"{sum(g > 0 for g in gains)}/{len(gains)}" if gains else "",
        "n_crash": sum(c.get("status") != "ok" for c in cases.values()),
        "n_blowup": sum(bool(c.get("blowup")) for c in cases.values()),
        "gpu_h": round(runtime / 3600, 2),
        **{f"p:{k}": v for k, v in t.params.items()},
        **{f"cfg:{k}": v for k, v in (t.user_attrs.get("overrides") or {}).items()},
    }


def ranked(study, spec: StudySpec, full_only: bool = True) -> list[FrozenTrial]:
    """COMPLETE trials by value, best first; full_only drops pruned (estimated) ones."""
    ts = [t for t in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
          if not (full_only and t.user_attrs.get("pruned"))]
    return sorted(ts, key=lambda t: -t.value)


def write_report(spec: StudySpec, plots: bool = True, top: int = 10) -> str:
    from stmr.tune.objective import load

    study = load(spec)
    out = os.path.join(spec.study_dir, "report")
    os.makedirs(out, exist_ok=True)
    trials = study.get_trials(deepcopy=False)

    rows = [trial_row(t, spec) for t in trials]
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    _write_csv(os.path.join(out, "trials.csv"), rows, keys)

    names = spec.case_names("tune")
    case_rows = []
    for t in trials:
        ca = case_attrs(t, spec)
        case_rows.append({"number": t.number, **{n: ca[n]["score"] if n in ca else "" for n in names}})
    _write_csv(os.path.join(out, "cases.csv"), case_rows, ["number", *names])

    best = ranked(study, spec)
    print(f"study {spec.name}: {len(trials)} trials, {len(best)} fully evaluated; "
          f"score = {spec.metric}(final) - {spec.metric}(zero-filled), mean over "
          f"{len(names)} tuning cases")
    by_number = {r["number"]: r for r in rows}
    for t in best[:top]:
        r = by_number[t.number]
        params = ", ".join(f"{k}={_fmt(v)}" for k, v in t.params.items())
        tag = f" [{r['baseline']}]" if r["baseline"] else ""
        print(f"  #{t.number:<4}{t.value:+8.3f}  vs init {r['gain_over_init']:+.3f}  "
              f"above init {r['above_init']:>6}{tag}  {params}")

    if plots and len(best) >= 2:
        _plots(study, out)
    print(f"report in {out}")
    return out


def _plots(study, out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import optuna
    from optuna.visualization import matplotlib as ovm

    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    for name, fn in (("history", ovm.plot_optimization_history),
                     ("importances", ovm.plot_param_importances),
                     ("parallel", ovm.plot_parallel_coordinate),
                     ("slice", ovm.plot_slice)):
        try:
            ax = fn(study)
            fig = (ax.flat[0] if hasattr(ax, "flat") else ax).figure
            fig.savefig(os.path.join(out, f"{name}.png"), dpi=120, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:  # too few trials for fANOVA, etc.
            print(f"  plot {name} skipped: {e}")
    try:
        imp = optuna.importance.get_param_importances(study)
        print("  importances: " + ", ".join(f"{k} {v:.2f}" for k, v in imp.items()))
    except Exception as e:
        print(f"  importances skipped: {e}")


def _write_csv(path: str, rows: list[dict], keys: list[str]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _fmt(v: Any) -> str:
    return f"{v:.3g}" if isinstance(v, float) else str(v)
