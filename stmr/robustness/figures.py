"""Robustness figures: per-run deformation + Jacobian, and cross-run spread / curves.

Reuses stmr.viz.deform.plot_displacement for the flow itself and to_mag for magnitude images.
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from stmr.robustness.jacobian import detJ_from_phi  # noqa: E402
from stmr.viz.deform import plot_displacement, to_mag  # noqa: E402

_CLR = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#56B4E9", "#000000"]


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------- per run ----------

def plot_detJ(phi, gt_im, H, W, out_path, start=0):
    """Per-interval Jacobian-determinant heatmap (diverging around 1, folds flagged) + a
    pooled det(J) histogram. det<=0 pixels are folds (non-invertible)."""
    detJ = detJ_from_phi(phi, H, W).cpu().numpy()      # [T-1, H, W]
    gt = to_mag(gt_im)
    T1 = detJ.shape[0]
    hi = float(np.percentile(np.abs(detJ - 1.0), 99)) + 1e-6
    vmin, vmax = 1 - hi, 1 + hi

    fig = plt.figure(figsize=(2.3 * T1 + 2.5, 3.0), layout="constrained")
    gs = fig.add_gridspec(1, T1 + 1)
    im = None
    for k in range(T1):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(gt[start + k] if start + k < gt.shape[0] else gt[k], cmap="gray")
        im = ax.imshow(detJ[k], cmap="RdBu_r", vmin=vmin, vmax=vmax, alpha=0.75)
        fold = detJ[k] <= 0
        if fold.any():
            ax.contour(fold, levels=[0.5], colors="lime", linewidths=0.8)
        ax.set_title(f"det J  {start+k}->{start+k+1}\nfold {100*fold.mean():.2f}%", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=fig.axes, location="right", fraction=0.02, pad=0.01, aspect=30)
    axh = fig.add_subplot(gs[0, T1])
    axh.hist(detJ.reshape(-1), bins=60, color="#0072B2")
    axh.axvline(1.0, color="k", lw=0.8); axh.axvline(0.0, color="red", lw=1, ls="--")
    axh.set_title("det J histogram", fontsize=8); axh.set_yticks([])
    fig.suptitle("Deformation Jacobian determinant (1=volume-preserving, <=0=fold)", fontsize=11)
    _save(fig, out_path)


def plot_run_displacement(phi, gt_im, H, W, out_path, start=0):
    """Thin wrapper around the existing displacement+quiver figure."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plot_displacement(phi, gt_im, H, W, start, out_path)


# ---------- cross run ----------

def plot_disagreement(std_mag, mean_disp, gt_im, out_path, start=0):
    """Pixelwise displacement-STD 'disagreement map' + mean-flow quiver, per interval."""
    std = std_mag.cpu().numpy()                      # [T-1, H, W] normalized units
    H, W = std.shape[-2:]
    std_px = std * (H - 1) / 2.0                      # to pixels (approx, symmetric grid)
    dy = mean_disp[..., 0].cpu().numpy() * (H - 1) / 2.0
    dx = mean_disp[..., 1].cpu().numpy() * (W - 1) / 2.0
    gt = to_mag(gt_im)
    T1 = std.shape[0]
    vmax = float(np.percentile(std_px, 99)) + 1e-6
    step = max(1, H // 24)
    ys, xs = np.mgrid[0:H:step, 0:W:step]

    fig, axes = plt.subplots(1, T1, figsize=(2.4 * T1, 2.8), layout="constrained")
    axes = np.atleast_1d(axes)
    im = None
    for k in range(T1):
        axes[k].imshow(gt[start + k] if start + k < gt.shape[0] else gt[k], cmap="gray")
        im = axes[k].imshow(std_px[k], cmap="magma", vmin=0, vmax=vmax, alpha=0.6)
        axes[k].quiver(xs, ys, dx[k, ::step, ::step], dy[k, ::step, ::step],
                       color="cyan", angles="xy", scale_units="xy", scale=1, width=0.004)
        axes[k].set_title(f"interval {start+k}->{start+k+1}\nmax std {std_px[k].max():.2f}px",
                          fontsize=8)
        axes[k].set_xticks([]); axes[k].set_yticks([])
    fig.colorbar(im, ax=list(axes), location="right", fraction=0.02, pad=0.01, aspect=30)
    fig.suptitle("Cross-run flow disagreement (pixelwise displacement STD) + mean flow", fontsize=11)
    _save(fig, out_path)


def plot_curves(records, tb_key, out_path, title=None):
    """Overlay a per-epoch scalar across runs, with mean +/- std band. Runs whose curve is
    entirely non-finite (e.g. S0 PSNR: recon frozen at GT -> inf) are skipped and noted, so
    the figure isn't silently blank."""
    named = [(r.key, r.curves[tb_key]) for r in records if tb_key in r.curves]
    finite = [(k, (s, v)) for k, (s, v) in named if np.isfinite(v).any()]
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    if not finite:
        ax.text(0.5, 0.5, f"no finite '{tb_key}' values\n(recon = GT -> PSNR = inf)",
                transform=ax.transAxes, ha="center", va="center", color="crimson", fontsize=10)
        ax.set_title(title or tb_key); _save(fig, out_path); return
    for k, (s, v) in finite:
        vv = np.where(np.isfinite(v), v, np.nan)
        ax.plot(s, vv, lw=1, alpha=0.5, label=k)
    L = min(len(v) for _, (_, v) in finite)
    stack = np.stack([np.where(np.isfinite(v[:L]), v[:L], np.nan) for _, (_, v) in finite], 0)
    steps = finite[0][1][0][:L]
    ax.plot(steps, np.nanmean(stack, 0), color="k", lw=2, label="mean")
    ax.fill_between(steps, np.nanmean(stack, 0) - np.nanstd(stack, 0),
                    np.nanmean(stack, 0) + np.nanstd(stack, 0), color="k", alpha=0.15)
    ax.set_xlabel("epoch"); ax.set_ylabel(tb_key); ax.grid(alpha=.25)
    ax.legend(fontsize=7); ax.set_title(title or tb_key)
    _save(fig, out_path)


def plot_spread(records, valuefn, ylabel, out_path, title=None, floor=None):
    """Strip plot of a per-run scalar across the axis keys, with an optional determinism-floor
    reference line. Non-finite values (e.g. PSNR=inf when recon==GT) are not silently dropped:
    they are flagged with a red caret at the top of the frame and an annotation, so a missing
    marker never reads as a failed run."""
    keys = [r.key for r in records]
    raw = np.array([valuefn(r) for r in records], dtype=float)
    finite = np.isfinite(raw)
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(keys)), 4.4), layout="constrained")

    if finite.any():
        ax.plot(np.where(finite)[0], raw[finite], "o", color="#0072B2", ms=7, label="run")
    else:  # nothing finite to plot (e.g. S0 best-PSNR: recon frozen at GT -> all inf)
        ax.text(0.5, 0.5, "all values non-finite\n(recon = GT -> PSNR = inf)",
                transform=ax.transAxes, ha="center", va="center", color="crimson", fontsize=10)

    # flag inf / -inf points at the frame edge instead of dropping them
    if (~finite).any() and finite.any():
        top = ax.get_ylim()[1]
        for i in np.where(~finite)[0]:
            ax.plot([i], [top], marker="^", color="crimson", ms=9, clip_on=False)
            ax.annotate(f"{raw[i]:+.0f}".replace("+inf", "∞").replace("-inf", "-∞"),
                        xy=(i, top), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color="crimson")

    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.set_title(title or ylabel)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    _save(fig, out_path)


def plot_perturbation_sensitivity(records, baseline, out_path, scene=""):
    """||mean displacement change vs the sigma=0 baseline|| against sigma (pixels)."""
    base_disp = baseline.displacement()
    xs, ys = [], []
    for r in sorted(records, key=lambda r: float(r.key)):
        d = (r.displacement() - base_disp).norm(dim=-1).mean().item() * (r.H - 1) / 2.0
        xs.append(float(r.key)); ys.append(d)
    fig, ax = plt.subplots(figsize=(6, 4.2), layout="constrained")
    ax.plot([0] + xs, [0] + ys, "o-", color="#D55E00")
    ax.set_xlabel("noise sigma (relative)"); ax.set_ylabel("mean ||d phi|| vs baseline (px)")
    ax.grid(alpha=.25); ax.set_title(f"{scene}: perturbation sensitivity")
    _save(fig, out_path)


def plot_synthetic_determinacy(summ, out_path):
    """Cross-seed determinacy of the recovered synthetic flow (orig + weighted), vs motion
    amplitude (at zero noise) and vs image noise (at the middle amplitude)."""
    amps = sorted({r["amp_px"] for r in summ})
    sigs = sorted({r["sigma"] for r in summ})

    def get(amp, sig, k):
        for r in summ:
            if r["amp_px"] == amp and r["sigma"] == sig:
                return r[k]
        return np.nan

    fig, ax = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    s0 = sigs[0]
    ax[0].plot(amps, [get(a, s0, "rel_disagree") for a in amps], "o-", color="#0072B2", label="rel (orig)")
    ax[0].plot(amps, [get(a, s0, "rel_disagree_w") for a in amps], "s--", color="#D55E00", label="rel (weighted)")
    ax[0].set_xlabel("motion amplitude (px)"); ax[0].set_ylabel("cross-seed relative disagreement")
    ax[0].set_title(f"vs motion amplitude (noise={s0})")
    am = amps[len(amps) // 2]
    ax[1].plot(sigs, [get(am, s, "rel_disagree") for s in sigs], "o-", color="#0072B2", label="rel (orig)")
    ax[1].plot(sigs, [get(am, s, "rel_disagree_w") for s in sigs], "s--", color="#D55E00", label="rel (weighted)")
    ax[1].set_xlabel("image noise sigma"); ax[1].set_ylabel("cross-seed relative disagreement")
    ax[1].set_title(f"vs image noise (amp={am:g}px)")
    for a in ax:
        a.axhline(1.0, color="k", lw=.7, ls=":"); a.grid(alpha=.25); a.legend(fontsize=8)
    fig.suptitle("Synthetic flow: cross-seed determinacy of the recovered deformation", fontsize=12)
    _save(fig, out_path)


def plot_epe(rows, out_path):
    """Synthetic-GT EPE vs seed (per amplitude) and vs noise."""
    amps = sorted({r["amp_px"] for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for amp, c in zip(amps, _CLR):
        sub = [r for r in rows if r["amp_px"] == amp and r["sigma"] == 0.0]
        sub.sort(key=lambda r: r["seed"])
        ax[0].plot([r["seed"] for r in sub], [r["epe_px"] for r in sub], "o-",
                   color=c, label=f"amp {amp:g}px")
        sub2 = [r for r in rows if r["amp_px"] == amp and r["seed"] == 0]
        sub2.sort(key=lambda r: r["sigma"])
        ax[1].plot([r["sigma"] for r in sub2], [r["epe_px"] for r in sub2], "o-",
                   color=c, label=f"amp {amp:g}px")
    ax[0].set_xlabel("seed"); ax[0].set_ylabel("EPE (px)"); ax[0].set_title("EPE vs seed (sigma 0)")
    ax[1].set_xlabel("image noise sigma"); ax[1].set_ylabel("EPE (px)")
    ax[1].set_title("EPE vs noise (seed 0)")
    for a in ax:
        a.grid(alpha=.25); a.legend(fontsize=8)
    fig.suptitle("Synthetic ground-truth flow accuracy (endpoint error)", fontsize=12)
    _save(fig, out_path)
