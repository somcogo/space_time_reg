"""Synthetic ground-truth deformation recovery: accuracy, not just cross-run spread.

Fits the *real* motion stack (GroupedSiren -> neural-ODE -> GridSampleTransformer) to a known
analytic flow and measures endpoint error (EPE, in pixels) against the true field. Unlike the
data runs, here we have the exact GT displacement, so we can quantify how accurately -- and how
reproducibly across seeds/noise -- the estimator recovers a known motion.

The target is built with the *same* GridSampleTransformer the network applies, so the recovered
and GT flows live in one convention and the EPE is unambiguous. (This is the live-module,
self-consistent successor to the stale motion_findings/verify_4_motion_bug_audit.py.)
"""

import csv
import os
from collections import defaultdict

import numpy as np
import torch
from torchdiffeq import odeint

from stmr.models.siren import GroupedSiren
from stmr.utils.spatial_transformer import GridSampleTransformer
from stmr.utils.spatial_utils import generate_coord_tensor

from .jacobian import detJ_from_phi, detJ_stats


def _phantom(H, W, device):
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    img = (torch.exp(-((xs) ** 2 + (ys) ** 2) / 0.2)
           + 0.5 * torch.exp(-((xs - 0.3) ** 2 + (ys + 0.2) ** 2) / 0.02))
    return img[None, None].to(device)  # [1, 1, H, W]


def analytic_flow(H, W, amp, device):
    """Return (img [1,1,H,W], target [1,1,H,W], rel_gt [H*W,2], aphi_gt [1,H*W,2]).

    GT displacement in the network's (row=y, col=x) convention:
    rel = (amp*cos(3x), amp*sin(3y)). target = warp(img, aphi_gt) via GridSampleTransformer.
    """
    img = _phantom(H, W, device)
    coord = generate_coord_tensor((H, W), device)          # [N, 2] = (y, x)
    gy, gx = coord[:, 0], coord[:, 1]
    rel_gt = torch.stack([amp * torch.cos(3 * gx), amp * torch.sin(3 * gy)], dim=-1)
    aphi_gt = (coord + rel_gt).unsqueeze(0)                 # [1, N, 2]
    target = GridSampleTransformer(aphi_gt, (1, H, W)).apply(img)
    return img, target, rel_gt, aphi_gt


def fit_siren_to_flow(img, target, H, W, steps=300, lr=3e-4, seed=0,
                      img_noise=0.0, step_size=0.1, device="cpu"):
    """Fit a fresh GroupedSiren to warp img->target. Returns (rel_pred [N,2], residuals list).

    ``step_size`` is the euler ODE step (0.1 matches the pipeline; a coarser step is faster
    for CPU tests). Larger lr compensates when fewer steps are used.
    """
    torch.manual_seed(seed)
    src = img
    if img_noise:
        src = img + img_noise * img.std() * torch.randn_like(img)
    net = GroupedSiren(groups=1, layers=[2, 64, 64, 2], last_init_zero=True, omega=30).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    coord = generate_coord_tensor((H, W), device)
    t01 = torch.tensor([0.0, 1.0], device=device)

    def integrate():
        return odeint(net, coord.unsqueeze(0), t01, method="euler",
                      options={"step_size": step_size})[1]

    residuals = []
    for _ in range(steps):
        opt.zero_grad()
        loss = (GridSampleTransformer(integrate(), (1, H, W)).apply(src) - target).pow(2).mean()
        loss.backward()
        opt.step()
        residuals.append(loss.item())
    with torch.no_grad():
        rel_pred = (integrate()[0] - coord)
    return rel_pred, residuals


def epe(rel_pred, rel_gt, H, W):
    """Mean endpoint error in PIXELS between predicted and GT displacement fields."""
    scale = torch.tensor([(H - 1) / 2.0, (W - 1) / 2.0], device=rel_pred.device)
    d = (rel_pred - rel_gt) * scale
    return d.norm(dim=-1).mean().item()


def run_synthetic_sweep(seeds=(0, 1, 2, 3, 4), amps=(1.0, 3.0, 6.0),
                        img_sigmas=(0.0, 0.02, 0.05, 0.1), H=64, W=64,
                        steps=300, lr=3e-4, step_size=0.1, device="cpu"):
    """Grid over (seed, amp, noise). Returns (rows, img, H, W); each row carries the recovered
    flow ``rel_pred`` so the full metric set (determinacy, Jacobian, ...) can be computed. The
    phantom ``img`` is the same for every amp (only the target differs), so one copy suffices."""
    rows = []
    img = None
    for amp_px in amps:
        amp = amp_px / (W / 2)  # pixels -> normalized units
        img, target, rel_gt, _ = analytic_flow(H, W, amp, device)
        for seed in seeds:
            for sigma in img_sigmas:
                rel_pred, res = fit_siren_to_flow(img, target, H, W, steps=steps, lr=lr,
                                                  seed=seed, img_noise=sigma,
                                                  step_size=step_size, device=device)
                rows.append({
                    "seed": seed, "amp_px": amp_px, "sigma": sigma,
                    "epe_px": epe(rel_pred, rel_gt, H, W),
                    "resid_reduction": 1.0 - res[-1] / (res[0] + 1e-12),
                    "rel_pred": rel_pred.detach().cpu(),  # [H*W, 2], for downstream metrics
                })
    return rows, img.detach().cpu(), H, W


# ---------------------------------------------------------------------------
# Full metric logging (mirrors the netsize/data robustness checks)
# ---------------------------------------------------------------------------

def _run_detJ(rel_pred, H, W):
    """detJ stats of a recovered flow (rel_pred = displacement [H*W,2])."""
    coord = generate_coord_tensor((H, W), "cpu")
    aphi = (rel_pred + coord).unsqueeze(0)          # [1, H*W, 2] absolute deformation
    return detJ_stats(detJ_from_phi(aphi, H, W))


def _flow_mag_px(rel_pred, H, W):
    return float(rel_pred.reshape(H, W, 2).norm(dim=-1).mean()) * (H - 1) / 2.0


def _synthetic_disagreement(rel_preds, img, H, W):
    """Cross-seed disagreement of recovered flows, unweighted and phantom-intensity-weighted.
    Mirrors netsize._flow_disagreement. Returns (abs_px, rel, abs_px_w, rel_w)."""
    disps = torch.stack([rp.reshape(H, W, 2) for rp in rel_preds], 0)  # [n, H, W, 2]
    mean = disps.mean(0)
    std_mag = disps.std(0, unbiased=False).norm(dim=-1)  # [H, W]
    flow_mag = mean.norm(dim=-1)
    to_px = (H - 1) / 2.0
    abs_u, flow_u = float(std_mag.mean()), float(flow_mag.mean())
    w = img[0, 0].abs()
    w = w / (w.sum() + 1e-12)                        # phantom intensity weight
    abs_w = float((std_mag * w).sum())
    flow_w = float((flow_mag * w).sum())
    return abs_u * to_px, abs_u / (flow_u + 1e-12), abs_w * to_px, abs_w / (flow_w + 1e-12)


def analyze_synthetic(rows, img, H, W, out):
    """Write per-run + per-(amp,noise) summary CSV/MD with the full metric set, and figures."""
    os.makedirs(out, exist_ok=True)
    for r in rows:
        st = _run_detJ(r["rel_pred"], H, W)
        r["detJmin"] = st["detJ_min"]
        r["foldover_frac"] = st["foldover_frac"]
        r["flow_mag_px"] = _flow_mag_px(r["rel_pred"], H, W)

    run_cols = ["seed", "amp_px", "sigma", "epe_px", "resid_reduction",
                "flow_mag_px", "detJmin", "foldover_frac"]
    with open(os.path.join(out, "runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=run_cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["amp_px"], r["sigma"])].append(r)
    summ = []
    for (amp, sig), rs in sorted(groups.items()):
        ad, rd, adw, rdw = _synthetic_disagreement([r["rel_pred"] for r in rs], img, H, W)
        e = np.array([r["epe_px"] for r in rs])
        summ.append({
            "amp_px": amp, "sigma": sig, "nseed": len(rs),
            "epe_mean": float(e.mean()), "epe_std": float(e.std()),
            "resid_reduction": float(np.mean([r["resid_reduction"] for r in rs])),
            "disagree_px": ad, "rel_disagree": rd,
            "disagree_px_w": adw, "rel_disagree_w": rdw,
            "flow_mag_px": float(np.mean([r["flow_mag_px"] for r in rs])),
            "detJmin_mean": float(np.mean([r["detJmin"] for r in rs])),
            "fold_max": float(np.max([r["foldover_frac"] for r in rs])),
        })
    cols = ["amp_px", "sigma", "nseed", "epe_mean", "epe_std", "resid_reduction",
            "disagree_px", "rel_disagree", "disagree_px_w", "rel_disagree_w",
            "flow_mag_px", "detJmin_mean", "fold_max"]
    with open(os.path.join(out, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(summ)
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write("| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n")
        for r in summ:
            fh.write("| " + " | ".join(
                f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |\n")
    return summ
