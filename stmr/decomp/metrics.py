"""Motion-responsibility metrics: how much of the motion phi1 vs phi2 accounts for.

Two complementary views, both focused on the anatomy via GT-intensity weighting:
  (a) displacement-magnitude split -- how big each field's motion is;
  (b) residual-reduction split     -- how much frame-alignment error each stage removes.
Plus per-net Jacobian/foldover stats and the single-net baseline residuals (capacity check).
"""

from __future__ import annotations

import torch
from fastmri import complex_abs

from stmr.robustness.jacobian import detJ_from_phi, detJ_stats
from stmr.utils.spatial_utils import generate_coord_tensor

from .model import compose_fields, warp

_EPS = 1e-12


def gt_weight(gt_im):
    """Per-interval GT source-frame intensity weight [T-1, H, W] (only anatomy matters)."""
    mag = complex_abs(gt_im.movedim(1, -1))  # [T, H, W]
    return mag[:-1]                           # [T-1, H, W] (un-normalised; callers normalise)


def disp_px(abs_phi, coord, H, W):
    """Per-pixel displacement magnitude in PIXELS -> [groups, H, W]."""
    disp = (abs_phi - coord).reshape(-1, H, W, 2)
    scale = disp.new_tensor([(H - 1) / 2.0, (W - 1) / 2.0])
    return (disp * scale).norm(dim=-1)


def _wmean(x, w):
    """Weighted mean of x [.,H,W] by w [.,H,W] over all elements."""
    return float((x * w).sum() / (w.sum() + _EPS))


def _per_interval_wmean(x, w):
    """Weighted mean per interval -> list[float] (weight renormalised within each interval)."""
    num = (x * w).flatten(1).sum(1)
    den = w.flatten(1).sum(1) + _EPS
    return (num / den).tolist()


def _mse_per_interval(a, b):
    return (a - b).pow(2).flatten(1).mean(1)   # [groups]


def compute_responsibility(abs_phi1, abs_phi2, mag, gt_im, coord, H, W,
                           abs_phi_base_large=None, abs_phi_base_small=None):
    """Return a nested dict with the displacement split, residual split, detJ stats, and
    baseline residuals. All tensors on the same device; returns plain Python floats/lists."""
    src, tgt = mag[:-1], mag[1:]
    w = gt_weight(gt_im).to(abs_phi1.device)

    # (a) displacement-magnitude split
    d1 = disp_px(abs_phi1, coord, H, W)
    d2 = disp_px(abs_phi2, coord, H, W)   # phi2's grid displacement = the residual step
    m1, m2 = _wmean(d1, w), _wmean(d2, w)
    disp_split = {
        "m1_px": m1, "m2_px": m2,
        "frac1": m1 / (m1 + m2 + _EPS), "frac2": m2 / (m1 + m2 + _EPS),
        "m1_per_interval": _per_interval_wmean(d1, w),
        "m2_per_interval": _per_interval_wmean(d2, w),
    }

    # (b) residual-reduction split (stage 2 via composed field -> single interpolation)
    w1 = warp(abs_phi1, src, H, W)
    comp = compose_fields(abs_phi1, abs_phi2, H, W)
    w12 = warp(comp, src, H, W)
    base = _mse_per_interval(src, tgt)
    r1 = _mse_per_interval(w1, tgt)
    r2 = _mse_per_interval(w12, tgt)
    bmean, r1mean, r2mean = float(base.mean()), float(r1.mean()), float(r2.mean())
    resid_split = {
        "base": bmean, "r1": r1mean, "r2": r2mean,
        "explained_phi1": (bmean - r1mean) / (bmean + _EPS),
        "explained_phi2": (r1mean - r2mean) / (bmean + _EPS),
        "unexplained": r2mean / (bmean + _EPS),
        "base_per_interval": base.tolist(),
        "r1_per_interval": r1.tolist(),
        "r2_per_interval": r2.tolist(),
    }
    if abs_phi_base_large is not None:
        resid_split["r_base_large"] = float(
            _mse_per_interval(warp(abs_phi_base_large, src, H, W), tgt).mean())
    if abs_phi_base_small is not None:
        resid_split["r_base_small"] = float(
            _mse_per_interval(warp(abs_phi_base_small, src, H, W), tgt).mean())

    # (c) Jacobian / foldover
    detJ = {"phi1": detJ_stats(detJ_from_phi(abs_phi1, H, W)),
            "phi2": detJ_stats(detJ_from_phi(abs_phi2, H, W))}

    # (d) what is the ~unexplained residual made of?
    unexplained = quantify_unexplained(w12, tgt, mag, base.mean())

    return {"disp_split": disp_split, "resid_split": resid_split, "detJ": detJ,
            "unexplained": unexplained}


def _bg_sigma(mag):
    """Noise std from the darkest corner patch across all frames (magnitude images)."""
    c = max(8, mag.shape[-2] // 8)
    corners = [mag[..., :c, :c], mag[..., :c, -c:], mag[..., -c:, :c], mag[..., -c:, -c:]]
    bg = min(corners, key=lambda p: float(p.mean()))
    return float(bg.std())


def quantify_unexplained(warped, target, mag, base):
    """Break the post-warp residual into interpretable parts.

    warped/target: [T-1,1,H,W]. Returns fractions of the base residual:
      - r2_over_base   : total unexplained (residual after the composed warp)
      - noise_frac     : thermal-noise floor 2*sigma^2 (two independent noisy frames)
      - structured_frac: unexplained beyond noise
    and, within the anatomy, where that residual sits:
      - resid_edge_frac: fraction of residual energy at high-gradient (edge) pixels
                         -> boundary mis-registration / through-plane motion
      - resid_flat_frac: fraction in flat regions -> intensity change / appearance
    """
    base = float(base)                                     # ensure scalar (not a cuda tensor)
    resid = (warped - target).pow(2)                       # [T-1,1,H,W]
    r2 = float(resid.mean())
    sigma = _bg_sigma(mag)
    noise_mse = 2.0 * sigma ** 2

    # edge vs flat inside the anatomy (exclude dark background)
    tgt = target[:, 0]                                     # [T-1,H,W]
    gy = torch.zeros_like(tgt); gx = torch.zeros_like(tgt)
    gy[:, 1:, :] = tgt[:, 1:, :] - tgt[:, :-1, :]
    gx[:, :, 1:] = tgt[:, :, 1:] - tgt[:, :, :-1]
    grad = (gy ** 2 + gx ** 2).sqrt()
    anat = tgt > 0.05 * float(tgt.max())                   # anatomy mask
    r = resid[:, 0]
    if anat.any():
        gvals = grad[anat]
        thr = float(gvals.median())
        edge = anat & (grad > thr)
        flat = anat & (grad <= thr)
        e = float(r[edge].sum()); f = float(r[flat].sum()); tot = e + f + _EPS
        edge_frac, flat_frac = e / tot, f / tot
    else:
        edge_frac = flat_frac = float("nan")

    return {
        "r2_over_base": r2 / (base + _EPS),
        "sigma": sigma,
        "noise_mse": noise_mse,
        "noise_frac": noise_mse / (base + _EPS),
        "structured_frac": max(r2 - noise_mse, 0.0) / (base + _EPS),
        "resid_edge_frac": edge_frac,
        "resid_flat_frac": flat_frac,
    }


def frame_match_psnr(warped_mag, target_mag):
    """Mean PSNR (dB) between a warped magnitude stack and the target magnitude stack."""
    mse = (warped_mag - target_mag).pow(2).mean()
    peak = target_mag.max()
    return float(10 * torch.log10(peak ** 2 / (mse + _EPS)))
