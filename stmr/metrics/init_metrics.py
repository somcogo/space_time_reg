"""Metrics and images for the initial-reconstruction (nmAPG) solve.

Deliberately modular: every function takes tensors and returns a plain dict, so a caller can
adopt any subset. That is what lets the full pipeline later cherry-pick these (e.g. just the
loss curves) without importing anything experiment-specific -- see stmr.metrics.init_logger
for the nmAPG callback that composes them, and stmr.initsweep for the sweep that drives it.

Shape convention throughout: complex images are [T, 2, H, W] (channel 1 = real/imag), k-space
likewise; ``mask`` is the boolean/0-1 k-space sampling mask broadcastable to that shape.
"""

from __future__ import annotations

import torch
from fastmri import complex_abs

from stmr.data.fft_utils import fft2c_new
from stmr.metrics.metric_utils import add_cmr_eval_metrics
from stmr.metrics.prep_visuals import prep_image_space_comp, prep_init_recon

_EPS = 1e-12


def _to_kspace(x: torch.Tensor) -> torch.Tensor:
    return fft2c_new(x.movedim(1, -1)).movedim(-1, 1)


def _mag(x: torch.Tensor) -> torch.Tensor:
    return complex_abs(x.movedim(1, -1))          # [T, H, W]


def _sharpness(x: torch.Tensor) -> float:
    """Mean finite-difference gradient magnitude of |x| -- a direct over-smoothing measure.

    A stronger prior blurs the reconstruction, which shows up here as a falling value; it is
    the cheapest way to see the reg weight trading detail for smoothness during a sweep.
    """
    m = _mag(x)
    gy = (m[:, 1:, :] - m[:, :-1, :]).abs().mean()
    gx = (m[:, :, 1:] - m[:, :, :-1]).abs().mean()
    return float(0.5 * (gy + gx))


def init_scalar_metrics(x: torch.Tensor, stats: dict, fixed: torch.Tensor,
                        mask: torch.Tensor) -> dict:
    """Cheap per-iteration scalars (safe to compute every nmAPG iteration).

    ``stats`` is the dict nmAPG hands the callback (data_fit, reg, res, L, n_active).

    The dc/* pair is the informative one for a regularisation sweep: measured_residual is how
    far the iterate has been pulled off the measurements, unmeasured_energy is how much the
    prior has invented in the k-space null space (zero when the prior is off).
    """
    k = _to_kspace(x)
    m = mask.to(dtype=k.dtype)
    measured_residual = float(((m * k - fixed) ** 2).sum())
    unmeasured_energy = float((((1.0 - m) * k) ** 2).sum())
    mag = _mag(x)
    return {
        "energy/data_fit": stats["data_fit"],
        "energy/reg": stats["reg"],
        "energy/total": stats["data_fit"] + stats["reg"],
        "conv/residual": stats["res"],
        "conv/L": stats["L"],
        "conv/step_size": 1.0 / (stats["L"] + _EPS),
        "conv/n_active": stats["n_active"],
        "dc/measured_residual": measured_residual,
        "dc/unmeasured_energy": unmeasured_energy,
        "recon/sharpness": _sharpness(x),
        "recon/mag_mean": float(mag.mean()),
        "recon/mag_max": float(mag.max()),
    }


def init_quality_metrics(x: torch.Tensor, gt_im: torch.Tensor) -> dict:
    """PSNR/SSIM/NMSE vs GT (full + cropped), plus the per-frame PSNR spread.

    Expensive -- SSIM is a per-(frame, channel) skimage call -- so drive this on a cadence.
    """
    metrics: dict = {}
    add_cmr_eval_metrics(x.detach().cpu(), gt_im.detach().cpu(), metrics)
    gm, xm = _mag(gt_im), _mag(x)
    peak = gm.amax(dim=(-2, -1))
    mse = ((xm - gm) ** 2).mean(dim=(-2, -1))
    per_frame = 10 * torch.log10(peak ** 2 / (mse + _EPS))
    metrics["recon/psnr_frame_min"] = float(per_frame.min())
    metrics["recon/psnr_frame_max"] = float(per_frame.max())
    return metrics


def init_images(x: torch.Tensor, gt_im: torch.Tensor, fixed: torch.Tensor,
                mask: torch.Tensor) -> dict:
    """TB image grids (HWC uint8). Logged on a cadence; tb_gif turns each tag into a GIF."""
    def _grid(t: torch.Tensor) -> torch.Tensor:
        return prep_init_recon(t.unsqueeze(1), is_complex=False)

    k = _to_kspace(x)
    m = mask.to(dtype=k.dtype)
    err = (_mag(x) - _mag(gt_im)).abs()
    # log-scaled so the (tiny) residual and the filled null space are both visible
    k_resid = torch.log1p(_mag(m * k - fixed))
    k_filled = torch.log1p(_mag((1.0 - m) * k))
    return {
        "imgs/recon": prep_init_recon(x),
        "imgs/recon_vs_gt": prep_image_space_comp(x, gt_im),
        "imgs/error": _grid(err),
        "imgs/kspace_residual": _grid(k_resid),
        "imgs/kspace_filled": _grid(k_filled),
    }
