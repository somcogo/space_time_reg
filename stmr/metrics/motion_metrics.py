"""Motion-accuracy metrics for saved deformations, in pixels and per image region.

``phi`` is the pipeline's deformation [T-1, H*W, 2] in normalised [-1, 1] coordinates with
channels (row/y, col/x), applied as a pull-back: warp(I_t, phi_t) ~ I_{t+1}. Displacements
u = phi - x are reported in pixels.

Residual reduction (how much warping lowers the frame-to-frame difference) is not a measure of
whether the motion is right: noise, non-motion intensity change and motion along edges or in
flat regions all decouple the two. These metrics measure the motion itself, either against a
known motion (endpoint error) or, without ground truth, by forward/backward consistency.
"""

import torch

from stmr.robustness.jacobian import detJ_from_phi
from stmr.utils.spatial_transformer import GridSampleTransformer
from stmr.utils.spatial_utils import generate_coord_tensor


def magnitude(x: torch.Tensor) -> torch.Tensor:
    """[T, 2, H, W] complex image -> [T, H, W] magnitude."""
    return x.pow(2).sum(1).sqrt()


def heart_crop(H: int, W: int, device=None) -> torch.Tensor:
    """The centre box the cropped CMR metrics use (metric_utils.add_cmr_eval_metrics)."""
    m = torch.zeros(H, W, dtype=torch.bool, device=device)
    m[round(H / 3):round(2 * H / 3), round(W / 4):round(3 * W / 4)] = True
    return m


def regions(gt_im: torch.Tensor) -> dict:
    """[H, W] masks: heart crop, tissue (mean magnitude > 10% of the max) and air (below 3% of
    the max in every frame, i.e. nothing to track)."""
    mag = magnitude(gt_im)
    H, W = mag.shape[-2:]
    return {
        "crop": heart_crop(H, W, mag.device),
        "tissue": mag.mean(0) > 0.10 * mag.max(),
        "air": mag.max(0).values < 0.03 * mag.max(),
    }


def displacement_px(phi: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """[B, H*W, 2] normalised deformation -> [B, H, W, 2] displacement in pixels (dy, dx)."""
    coord = generate_coord_tensor((H, W), phi.device).to(phi.dtype)
    span = torch.tensor([(H - 1) / 2, (W - 1) / 2], device=phi.device, dtype=phi.dtype)
    return ((phi - coord) * span).reshape(-1, H, W, 2)


def endpoint_error(phi: torch.Tensor, gt_phi: torch.Tensor, gt_im: torch.Tensor) -> dict:
    """Mean |u - u*| in pixels per region, next to the mean true motion |u*| (the error of
    predicting no motion). epe_rel_crop = epe_crop / gt_motion_crop: 0 is perfect, 1 is no better
    than zero motion."""
    H, W = gt_im.shape[-2:]
    u, u_gt = displacement_px(phi, H, W), displacement_px(gt_phi.to(phi), H, W)
    err, motion = (u - u_gt).norm(dim=-1), u_gt.norm(dim=-1)
    out = {}
    for name, m in regions(gt_im.to(phi.device)).items():
        out[f"epe_{name}"] = err[:, m].mean().item()
        out[f"gt_motion_{name}"] = motion[:, m].mean().item()
    out["epe_rel_crop"] = out["epe_crop"] / max(out["gt_motion_crop"], 1e-12)
    return out


def inverse_consistency(phi_fwd: torch.Tensor, phi_bwd: torch.Tensor, gt_im: torch.Tensor) -> dict:
    """|phi_fwd(phi_bwd(x)) - x| in pixels, between a run on the frames and a run on the same
    frames in reverse order. Forward interval k maps frame k+1 -> frame k; it pairs with interval
    T-2-k of the reversed run, which maps frame k -> frame k+1. Exact inverses give 0; unrelated
    fields give about the size of the motion. A large value shows at least one direction is wrong;
    a small one cannot rule out a bias both directions share."""
    T, _, H, W = gt_im.shape
    coord = generate_coord_tensor((H, W), phi_fwd.device).to(phi_fwd.dtype)
    span = torch.tensor([(H - 1) / 2, (W - 1) / 2], device=phi_fwd.device, dtype=phi_fwd.dtype)
    errs = []
    for k in range(T - 1):
        j = T - 2 - k
        u_fwd = (phi_fwd[k] - coord).reshape(H, W, 2).permute(2, 0, 1)[None]          # [1, 2, H, W]
        u_fwd_at = GridSampleTransformer(phi_bwd[j:j + 1], (2, H, W)).apply(u_fwd)[0]   # u_fwd(phi_bwd(x))
        u_bwd = (phi_bwd[j] - coord).reshape(H, W, 2).permute(2, 0, 1)
        errs.append(((u_bwd + u_fwd_at).permute(1, 2, 0) * span).norm(dim=-1))
    err = torch.stack(errs)                                                             # [T-1, H, W]
    motion = displacement_px(phi_fwd, H, W).norm(dim=-1)
    crop = heart_crop(H, W, err.device)
    out = {"fwd_bwd_crop": err[:, crop].mean().item(), "motion_crop": motion[:, crop].mean().item()}
    out["fwd_bwd_rel_crop"] = out["fwd_bwd_crop"] / max(out["motion_crop"], 1e-12)
    return out


def fold_stats(phi: torch.Tensor, H: int, W: int) -> dict:
    """Smallest Jacobian determinant and the fraction of folded pixels (det <= 0)."""
    detJ = detJ_from_phi(phi, H, W, device=phi.device)
    return {"detJ_min": detJ.min().item(), "fold_frac": (detJ <= 0).float().mean().item()}
