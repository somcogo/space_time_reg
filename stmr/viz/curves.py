"""Generic training/solve curve plotting.

Deliberately generic over a ``{tag: [values]}`` history so the same function serves the init
solve (stmr.initsweep) and, later, the registration losses in the full pipeline -- the
registration keeps a list-of-dicts (``all_metrics``), which ``history_from_records`` converts.
Keeping this out of any experiment package is what makes that reuse a one-line change.
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def history_from_records(records: list[dict], key: str | None = None) -> dict[str, list[float]]:
    """Flatten a list of per-step metric dicts into {tag: [values]}.

    ``key`` optionally selects a sub-dict of each record (the registration stores its scalars
    under records[i]["metrics"]). Tags missing from a record are skipped for that step.
    """
    out: dict[str, list[float]] = {}
    for rec in records:
        d = rec.get(key, {}) if key else rec
        for k, v in d.items():
            try:
                out.setdefault(k, []).append(float(v))
            except (TypeError, ValueError):
                continue  # non-scalar entries (per-pixel maps) are not curves
    return out


def plot_loss_curves(history: dict[str, list[float]], out_path: str,
                     tags: list[str] | None = None, twin: str | None = None,
                     steps: list[int] | None = None, logy: bool = True,
                     title: str = "loss curves") -> str:
    """Plot selected tags against step, optionally with one tag on a twin y-axis.

    ``tags``  : which history keys to draw on the left axis (default: all present).
    ``twin``  : a tag drawn on a right-hand axis (e.g. PSNR against the energies).
    ``logy``  : log-scale the left axis (energies span orders of magnitude).
    """
    tags = [t for t in (tags or list(history)) if t in history and len(history[t])]
    if not tags:
        raise ValueError("no plottable tags in history")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for t in tags:
        y = np.asarray(history[t], dtype=float)
        x = np.asarray(steps[:len(y)] if steps is not None else range(len(y)))
        finite = np.isfinite(y)
        # log-scale needs strictly positive values; drop non-positive points rather than
        # letting matplotlib silently blank the whole series
        if logy:
            finite &= y > 0
        ax.plot(x[finite], y[finite], label=t, lw=1.3)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("iteration"); ax.set_ylabel("value"); ax.grid(alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if twin and twin in history and len(history[twin]):
        ax2 = ax.twinx()
        y = np.asarray(history[twin], dtype=float)
        x = np.asarray(steps[:len(y)] if steps is not None else range(len(y)))
        finite = np.isfinite(y)
        ax2.plot(x[finite], y[finite], color="tab:red", lw=1.6, ls="--", label=twin)
        ax2.set_ylabel(twin, color="tab:red"); ax2.tick_params(axis="y", labelcolor="tab:red")
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    ax.legend(handles, labels, fontsize=7, loc="best")
    ax.set_title(title, fontsize=10)
    fig.savefig(out_path, dpi=120); plt.close(fig)
    return out_path
