"""Per-run figures for an init solve. Loss curves come from the shared stmr.viz.curves."""

from __future__ import annotations

import os

import matplotlib
import numpy as np
import torch
from fastmri import complex_abs

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from stmr.viz.curves import plot_loss_curves  # noqa: E402


def _mag(x: torch.Tensor) -> torch.Tensor:
    return complex_abs(x.detach().cpu().movedim(1, -1))


def make_init_figures(history: dict, steps: list, recon: torch.Tensor, gt_im: torch.Tensor,
                      fixed: torch.Tensor, mask: torch.Tensor, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    # 1. loss curves (shared plotter): energies on log-y, PSNR on a twin axis
    paths.append(plot_loss_curves(
        history, os.path.join(out_dir, "loss_curves.png"),
        tags=["energy/total", "energy/data_fit", "energy/reg"],
        twin="cmr evals all/full psnr", steps=steps, title="init energy + PSNR"))

    # 2. convergence: nmAPG's own stopping residual and its Lipschitz/step estimate
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for t, c in [("conv/residual", "tab:blue"), ("conv/L", "tab:green")]:
        if history.get(t):
            y = np.asarray(history[t], float); x = np.asarray(steps[:len(y)])
            f = np.isfinite(y) & (y > 0)
            ax.plot(x[f], y[f], label=t, color=c)
    ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.grid(alpha=.3)
    ax.legend(fontsize=8); ax.set_title("convergence", fontsize=10)
    p = os.path.join(out_dir, "convergence.png"); fig.savefig(p, dpi=120); plt.close(fig)
    paths.append(p)

    # 3. the data-consistency split: how much the iterate is pulled off the measurements vs
    #    how much the prior fills the unmeasured null space
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for t, c in [("dc/measured_residual", "tab:red"), ("dc/unmeasured_energy", "tab:purple")]:
        if history.get(t):
            y = np.asarray(history[t], float); x = np.asarray(steps[:len(y)])
            f = np.isfinite(y) & (y > 0)
            ax.plot(x[f], y[f], label=t, color=c)
    ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.grid(alpha=.3)
    ax.legend(fontsize=8); ax.set_title("k-space split: measured residual vs filled null space", fontsize=10)
    p = os.path.join(out_dir, "kspace_split.png"); fig.savefig(p, dpi=120); plt.close(fig)
    paths.append(p)

    # 4. final recon | GT | error
    rm, gm = _mag(recon), _mag(gt_im)
    err = (rm - gm).abs()
    t = 0
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.2), layout="constrained")
    vmax = float(gm[t].max())
    for a, (img, ttl, cm, vm) in zip(axes, [(rm[t], "init recon", "gray", vmax),
                                            (gm[t], "GT", "gray", vmax),
                                            (err[t], "|recon - GT|", "magma", None)]):
        im = a.imshow(img, cmap=cm, vmax=vm); a.set_title(ttl, fontsize=9); a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.03)
    fig.suptitle("final init reconstruction (frame 0)", fontsize=11)
    p = os.path.join(out_dir, "final_recon.png"); fig.savefig(p, dpi=120); plt.close(fig)
    paths.append(p)
    return paths
