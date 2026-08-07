"""Synthetic ground-truth deformation recovery: accuracy, not just cross-run spread.

Fits the *real* motion stack (GroupedSiren -> neural-ODE -> GridSampleTransformer) to a known
analytic flow and measures endpoint error (EPE, in pixels) against the true field. Unlike the
data runs, here we have the exact GT displacement, so we can quantify how accurately -- and how
reproducibly across seeds/noise -- the estimator recovers a known motion.

The target is built with the *same* GridSampleTransformer the network applies, so the recovered
and GT flows live in one convention and the EPE is unambiguous. (This is the live-module,
self-consistent successor to the stale motion_findings/verify_4_motion_bug_audit.py.)
"""

import numpy as np
import torch
from torchdiffeq import odeint

from stmr.models.siren import GroupedSiren
from stmr.utils.spatial_transformer import GridSampleTransformer
from stmr.utils.spatial_utils import generate_coord_tensor


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
    """Grid over (seed, amp, noise); return a list of {seed, amp_px, sigma, epe_px, resid_red}."""
    rows = []
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
                })
    return rows
