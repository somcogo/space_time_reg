"""Figures for the motion-decomposition experiment (reused by run_decomp and analyze)."""

from __future__ import annotations

import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from flow_vis import flow_to_color  # noqa: E402

from stmr.robustness.jacobian import detJ_from_phi  # noqa: E402
from stmr.viz.deform import plot_displacement, to_mag  # noqa: E402

from .model import compose_fields, warp  # noqa: E402

_CLR1, _CLR2 = "#0072B2", "#D55E00"


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_detJ(abs_phi, H, W, out_path, title):
    detJ = detJ_from_phi(abs_phi, H, W).cpu().numpy()  # [T-1, H, W]
    T1 = detJ.shape[0]
    hi = float(np.percentile(np.abs(detJ - 1.0), 99)) + 1e-6
    fig, axes = plt.subplots(1, T1, figsize=(2.2 * T1 + 1.5, 2.7), layout="constrained")
    axes = np.atleast_1d(axes)
    im = None
    for k in range(T1):
        im = axes[k].imshow(detJ[k], cmap="RdBu_r", vmin=1 - hi, vmax=1 + hi)
        fold = detJ[k] <= 0
        if fold.any():
            axes[k].contour(fold, levels=[0.5], colors="lime", linewidths=0.7)
        axes[k].set_title(f"{k}->{k+1}\nfold {100*fold.mean():.2f}%", fontsize=8)
        axes[k].set_xticks([]); axes[k].set_yticks([])
    fig.colorbar(im, ax=list(axes), location="right", fraction=0.02, pad=0.01, aspect=30)
    fig.suptitle(f"{title}  (det J; 1=volume-preserving, <=0=fold)", fontsize=11)
    _save(fig, out_path)


def _plot_warp_grid(res, out_path):
    """Rows: source |I_i|, after phi1, after phi1+phi2, target |I_{i+1}|, |after-target|."""
    H, W = res["H"], res["W"]
    mag = res["mag"]                      # [T,1,H,W]
    src, tgt = mag[:-1], mag[1:]
    w1 = warp(res["abs_phi1"], src, H, W)
    w12 = warp(res["abs_phi2"], w1, H, W)
    rows = [("source I_i", src), ("after phi1", w1), ("after phi1+phi2", w12),
            ("target I_i+1", tgt), ("|resid|", (w12 - tgt).abs())]
    T1 = src.shape[0]
    vmax = float(tgt.max())
    dmax = float((w12 - tgt).abs().max()) + 1e-9
    fig, axes = plt.subplots(len(rows), T1, figsize=(1.6 * T1, 1.7 * len(rows)),
                             layout="constrained")
    axes = np.atleast_2d(axes)
    for r, (label, stack) in enumerate(rows):
        s = stack[:, 0].cpu().numpy()
        vm = dmax if label.startswith("|resid") else vmax
        cm = "magma" if label.startswith("|resid") else "gray"
        for k in range(T1):
            axes[r, k].imshow(s[k], cmap=cm, vmin=0, vmax=vm)
            axes[r, k].set_xticks([]); axes[r, k].set_yticks([])
            if r == 0:
                axes[r, k].set_title(f"{k}->{k+1}", fontsize=8)
        axes[r, 0].set_ylabel(label, fontsize=9)
    fig.suptitle("Registration: identity -> phi1 -> phi1+phi2 vs target", fontsize=11)
    _save(fig, out_path)


def _plot_responsibility(res, out_path):
    r = res["responsibility"]
    ds, rs = r["disp_split"], r["resid_split"]
    m1, m2 = np.array(ds["m1_per_interval"]), np.array(ds["m2_per_interval"])
    base = np.array(rs["base_per_interval"])
    r1 = np.array(rs["r1_per_interval"])
    r2 = np.array(rs["r2_per_interval"])
    ex1 = (base - r1) / (base + 1e-12)
    ex2 = (r1 - r2) / (base + 1e-12)
    un = r2 / (base + 1e-12)
    x = np.arange(len(m1))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
    # (1) displacement magnitude per interval
    ax[0].bar(x - 0.2, m1, 0.4, color=_CLR1, label="phi1 (coarse)")
    ax[0].bar(x + 0.2, m2, 0.4, color=_CLR2, label="phi2 (residual)")
    ax[0].set_xlabel("interval"); ax[0].set_ylabel("weighted displacement (px)")
    ax[0].set_title(f"Displacement magnitude\naggregate: phi1 {ds['frac1']*100:.0f}% / "
                    f"phi2 {ds['frac2']*100:.0f}%")
    ax[0].legend(fontsize=8); ax[0].set_xticks(x)
    # (2) residual explained, stacked
    ax[1].bar(x, ex1, color=_CLR1, label="explained by phi1")
    ax[1].bar(x, ex2, bottom=ex1, color=_CLR2, label="explained by phi2")
    ax[1].bar(x, un, bottom=ex1 + ex2, color="#999999", label="unexplained")
    ax[1].set_xlabel("interval"); ax[1].set_ylabel("fraction of base residual")
    ax[1].set_title(f"Residual reduction\nphi1 {rs['explained_phi1']*100:.0f}% / "
                    f"phi2 {rs['explained_phi2']*100:.0f}% / unexp {rs['unexplained']*100:.0f}%")
    ax[1].legend(fontsize=8); ax[1].set_xticks(x)
    # (3) capacity: two-stage vs baselines
    labels, vals, cols = ["two-stage\n(r2)"], [rs["r2"]], ["#009E73"]
    if "r_base_large" in rs:
        labels.append("large\nalone"); vals.append(rs["r_base_large"]); cols.append(_CLR2)
    if "r_base_small" in rs:
        labels.append("small\nalone"); vals.append(rs["r_base_small"]); cols.append(_CLR1)
    ax[2].bar(range(len(vals)), vals, color=cols)
    ax[2].set_xticks(range(len(vals))); ax[2].set_xticklabels(labels, fontsize=8)
    ax[2].set_ylabel("final residual (MSE)"); ax[2].set_title("Capacity check\n(lower = better)")
    fig.suptitle("Motion responsibility: phi1 (coarse) vs phi2 (residual)", fontsize=12)
    _save(fig, out_path)


def _plot_direction(abs_phi, coord, H, W, out_path, title):
    """Per-interval flow DIRECTION as an optical-flow colour wheel (hue=direction,
    brightness=magnitude, per-panel normalised)."""
    disp = (abs_phi - coord).reshape(-1, H, W, 2).cpu().numpy().astype("float32")  # (dy,dx)
    T1 = disp.shape[0]
    fig, axes = plt.subplots(1, T1, figsize=(2.2 * T1, 2.7), layout="constrained")
    axes = np.atleast_1d(axes)
    for k in range(T1):
        axes[k].imshow(flow_to_color(disp[k], convert_to_bgr=False))
        axes[k].set_title(f"{k}->{k+1}", fontsize=8)
        axes[k].set_xticks([]); axes[k].set_yticks([])
    fig.suptitle(f"{title} flow DIRECTION (hue = direction, brightness = magnitude)", fontsize=11)
    _save(fig, out_path)


def _plot_unexplained(res, out_path):
    """Residual map after the composed warp + what the unexplained residual is made of."""
    H, W = res["H"], res["W"]
    mag = res["mag"]
    src, tgt = mag[:-1], mag[1:]
    comp = compose_fields(res["abs_phi1"], res["abs_phi2"], H, W)
    resid = (warp(comp, src, H, W) - tgt).pow(2)[:, 0].cpu().numpy()   # [T-1,H,W]
    u = res["responsibility"]["unexplained"]
    T1 = resid.shape[0]
    vmax = float(np.percentile(resid, 99)) + 1e-12

    fig = plt.figure(figsize=(1.7 * T1 + 5, 3.3), layout="constrained")
    gs = fig.add_gridspec(1, T1 + 2)
    im = None
    for k in range(T1):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(resid[k], cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"{k}->{k+1}", fontsize=7); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=fig.axes, location="left", fraction=0.015, pad=0.01, aspect=30)
    axb = fig.add_subplot(gs[0, T1])
    axb.bar(["noise", "structured"], [u["noise_frac"], u["structured_frac"]],
            color=["#999999", "#D55E00"])
    axb.set_title(f"unexplained = {u['r2_over_base']*100:.0f}% of base", fontsize=8)
    axb.set_ylabel("fraction of base residual")
    axe = fig.add_subplot(gs[0, T1 + 1])
    axe.bar(["edge", "flat"], [u["resid_edge_frac"], u["resid_flat_frac"]],
            color=["#0072B2", "#009E73"])
    axe.set_title("residual location\nedge=through-plane / flat=intensity", fontsize=8)
    fig.suptitle("Unexplained residual: post-warp map, noise floor, and where it concentrates",
                 fontsize=11)
    _save(fig, out_path)


def make_figures(res: dict, out_dir: str):
    """Write all decomposition figures from a results dict (as saved in res.pt)."""
    os.makedirs(out_dir, exist_ok=True)
    H, W = res["H"], res["W"]
    gt, coord = res["gt_im"], res["coord"]
    plot_displacement(res["abs_phi1"], gt, H, W, 0, os.path.join(out_dir, "phi1_displacement.png"))
    plot_displacement(res["abs_phi2"], gt, H, W, 0,
                      os.path.join(out_dir, "phi2_residual_displacement.png"))
    _plot_direction(res["abs_phi1"], coord, H, W, os.path.join(out_dir, "phi1_direction.png"), "phi1")
    _plot_direction(res["abs_phi2"], coord, H, W,
                    os.path.join(out_dir, "phi2_direction.png"), "phi2 (residual)")
    _plot_detJ(res["abs_phi1"], H, W, os.path.join(out_dir, "detJ_phi1.png"), "phi1")
    _plot_detJ(res["abs_phi2"], H, W, os.path.join(out_dir, "detJ_phi2.png"), "phi2 (residual)")
    _plot_warp_grid(res, os.path.join(out_dir, "warp_vs_gt_grid.png"))
    _plot_unexplained(res, os.path.join(out_dir, "unexplained.png"))
    _plot_responsibility(res, os.path.join(out_dir, "responsibility.png"))
