"""Per-axis robustness summary tables (markdown + csv)."""

from __future__ import annotations

import csv
import io
import os

import numpy as np


def _row(r):
    return {
        "scene": r.scene, "axis": r.axis, "key": r.key,
        "best_psnr": round(r.best_full_psnr, 3),
        "best_epoch": r.best_epoch,
        "imdiff_ratio": round(r.imdiff_ratio, 4),
        "foldover_frac": round(r.detJ["foldover_frac"], 5),
        "detJ_min": round(r.detJ["detJ_min"], 4),
        "vel_l2": round(r.vel_l2, 4),
        "vel_h1": round(r.vel_h1, 4),
    }


FIELDS = ["scene", "axis", "key", "best_psnr", "best_epoch", "imdiff_ratio",
          "foldover_frac", "detJ_min", "vel_l2", "vel_h1"]


def write_axis_table(records, out_dir, scene, axis):
    """Write <scene>_<axis>.md and .csv for one axis group. Returns the markdown string."""
    rows = [_row(r) for r in records]
    os.makedirs(out_dir, exist_ok=True)
    # csv
    with open(os.path.join(out_dir, f"{scene}_{axis}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    # markdown
    out = io.StringIO()
    out.write(f"### {scene} — {axis} axis\n\n")
    out.write("| " + " | ".join(FIELDS) + " |\n")
    out.write("|" + "|".join("---" for _ in FIELDS) + "|\n")
    for r in rows:
        out.write("| " + " | ".join(str(r[f]) for f in FIELDS) + " |\n")
    # spread summary
    def spread(key):
        vals = [r[key] for r in rows if isinstance(r[key], (int, float)) and np.isfinite(r[key])]
        return (np.mean(vals), np.std(vals), np.ptp(vals)) if vals else (np.nan,) * 3
    out.write("\nspread (mean / std / range) across this axis:\n")
    for key in ("best_psnr", "imdiff_ratio", "foldover_frac", "vel_l2"):
        m, s, rng = spread(key)
        out.write(f"- {key}: {m:.4g} / {s:.4g} / {rng:.4g}\n")
    md = out.getvalue()
    with open(os.path.join(out_dir, f"{scene}_{axis}.md"), "w") as fh:
        fh.write(md)
    return md
