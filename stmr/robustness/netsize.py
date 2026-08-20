"""Analyze the S5 network-size x seed sweep: is the deformation better determined (lower
cross-seed flow disagreement) at smaller SIREN sizes, and at what PSNR/NMSE cost?

Layout expected: <root>/S5/d<depth>h<dim>/seed<s>/S5/res.pt (built by netsize_launch).
"""

from __future__ import annotations

import os
import re

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from fastmri import complex_abs  # noqa: E402

from .collect import collect_sweep, disagreement_map, group_by_axis  # noqa: E402
from .figures import plot_disagreement  # noqa: E402

_SIZE_RE = re.compile(r"d(\d+)h(\d+)")


def _param_count(depth, dim, groups=5):
    """GroupedSiren param count: layers [2, dim*depth, 2], `groups` independent copies."""
    layers = [2] + [dim] * depth + [2]
    per = sum(layers[i] * layers[i + 1] + layers[i + 1] for i in range(len(layers) - 1))
    return groups * per


def _best_metrics(rec):
    """(best_psnr, best_nmse) from the run's curves (max PSNR / min NMSE over epochs)."""
    psnr = rec.curves.get("cmr evals all/full psnr", (None, np.array([np.nan])))[1]
    nmse = rec.curves.get("cmr evals all/full nsme", (None, np.array([np.nan])))[1]
    finite_p = psnr[np.isfinite(psnr)]
    finite_n = nmse[np.isfinite(nmse)]
    bp = float(finite_p.max()) if finite_p.size else float("nan")
    bn = float(finite_n.min()) if finite_n.size else float("nan")
    return bp, bn


def _gt_weight(recs):
    """Per-interval GT source-frame intensity weight [T-1, H, W], normalized to sum 1.

    We only care about motion where there is anatomy, so pixels are weighted by the GT
    image magnitude of the source frame (interval k uses GT frame k). All seeds share the
    same GT, so recs[0] suffices.
    """
    mag = complex_abs(recs[0].gt_im.movedim(1, -1))  # [T, H, W]
    w = mag[:-1]                                       # [T-1, H, W] source frames
    return w / (w.sum() + 1e-12)


def _flow_disagreement(recs):
    """Cross-seed disagreement, unweighted and GT-intensity-weighted.

    Returns (abs_px, rel, abs_px_w, rel_w, std_mag, mean_disp). abs_px = mean pixelwise
    displacement STD (px); rel = abs / mean flow magnitude. The _w variants replace the
    uniform pixel mean with a GT-intensity-weighted mean, so background pixels (no anatomy,
    no meaningful motion) don't dilute the metric.
    """
    std_mag, mean = disagreement_map(recs)          # [T-1,H,W], [T-1,H,W,2]; normalized units
    H = recs[0].H
    to_px = (H - 1) / 2.0
    flow_mag = mean.norm(dim=-1)                     # [T-1, H, W]

    abs_u = float(std_mag.mean())
    flow_u = float(flow_mag.mean())

    w = _gt_weight(recs)                              # [T-1, H, W], sums to 1
    abs_w = float((std_mag * w).sum())
    flow_w = float((flow_mag * w).sum())

    return (abs_u * to_px, abs_u / (flow_u + 1e-12),
            abs_w * to_px, abs_w / (flow_w + 1e-12), std_mag, mean)


def analyze(root, out):
    recs = collect_sweep(root)
    groups = group_by_axis(recs)                    # {(S5, "d3h64"): [seed recs]}
    rows = []
    for (scene, axis), rs in sorted(groups.items()):
        m = _SIZE_RE.match(axis)
        if not m:
            continue
        depth, dim = int(m.group(1)), int(m.group(2))
        abs_px, rel, abs_px_w, rel_w, std_mag, mean = _flow_disagreement(rs)
        bp = np.array([_best_metrics(r)[0] for r in rs])
        bn = np.array([_best_metrics(r)[1] for r in rs])
        rows.append({
            "size": axis, "depth": depth, "dim": dim,
            "params": _param_count(depth, dim),
            "nseed": len(rs),
            "disagree_px": abs_px, "rel_disagree": rel,
            "disagree_px_w": abs_px_w, "rel_disagree_w": rel_w,
            "psnr_mean": float(np.nanmean(bp)), "psnr_std": float(np.nanstd(bp)),
            "nmse_mean": float(np.nanmean(bn)), "nmse_std": float(np.nanstd(bn)),
            "vel_l2_mean": float(np.mean([r.vel_l2 for r in rs])),
            "vel_l2_std": float(np.std([r.vel_l2 for r in rs])),
            "detJmin_mean": float(np.mean([r.detJ["detJ_min"] for r in rs])),
            "fold_max": float(np.max([r.detJ["foldover_frac"] for r in rs])),
            "imdiff_mean": float(np.mean([r.imdiff_ratio for r in rs])),
        })
        # per-size disagreement map
        plot_disagreement(std_mag, mean, rs[0].gt_im,
                          os.path.join(out, "per-size", f"{axis}_disagreement.png"))

    rows.sort(key=lambda r: r["params"])
    _plot_summary(rows, os.path.join(out, "netsize_summary.png"))
    _write_table(rows, os.path.join(out, "netsize.md"), os.path.join(out, "netsize.csv"))
    return rows


def _plot_summary(rows, path):
    labels = [f"{r['size']}\n{r['params']/1e3:.0f}k" for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")

    # (A) determinacy: absolute + relative cross-seed disagreement
    a = ax[0]
    a.plot(x, [r["disagree_px"] for r in rows], "o-", color="#0072B2", label="abs disagreement (px)")
    a.set_ylabel("cross-seed disagreement (px)", color="#0072B2")
    a.tick_params(axis="y", labelcolor="#0072B2")
    a2 = a.twinx()
    a2.plot(x, [r["rel_disagree"] for r in rows], "s--", color="#D55E00", label="relative (/flow)")
    a2.set_ylabel("relative disagreement (dimensionless)", color="#D55E00")
    a2.tick_params(axis="y", labelcolor="#D55E00")
    a.set_title("Flow determinacy vs network size\n(lower = better determined)")
    a.set_xticks(x); a.set_xticklabels(labels, fontsize=8); a.grid(alpha=.25)

    # (B) performance: PSNR (higher better) + NMSE (lower better)
    b = ax[1]
    b.errorbar(x, [r["psnr_mean"] for r in rows], yerr=[r["psnr_std"] for r in rows],
               fmt="o-", color="#009E73", capsize=3, label="best PSNR")
    b.set_ylabel("best full PSNR (dB)", color="#009E73"); b.tick_params(axis="y", labelcolor="#009E73")
    b2 = b.twinx()
    b2.errorbar(x, [r["nmse_mean"] for r in rows], yerr=[r["nmse_std"] for r in rows],
                fmt="s--", color="#CC79A7", capsize=3, label="best NMSE")
    b2.set_ylabel("best NMSE", color="#CC79A7"); b2.tick_params(axis="y", labelcolor="#CC79A7")
    b.set_title("Reconstruction performance vs network size")
    b.set_xticks(x); b.set_xticklabels(labels, fontsize=8); b.grid(alpha=.25)

    fig.suptitle("S5 SIREN size sweep: does a smaller net give a better-determined flow?", fontsize=12)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120); plt.close(fig)


def _write_table(rows, md_path, csv_path):
    import csv
    cols = ["size", "params", "nseed", "psnr_mean", "psnr_std", "nmse_mean", "nmse_std",
            "disagree_px", "rel_disagree", "disagree_px_w", "rel_disagree_w",
            "vel_l2_mean", "detJmin_mean", "fold_max", "imdiff_mean"]
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    with open(md_path, "w") as fh:
        fh.write("| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n")
        for r in rows:
            fh.write("| " + " | ".join(
                (f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c])) for c in cols) + " |\n")
