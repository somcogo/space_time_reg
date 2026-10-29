"""Extract metrics from a stage's ``res.pt`` and render the ladder summary.

All metrics are recomputed from the saved tensors (the best-loss recon ``moving``, the
deformation ``phi``, the init recon and GT), reusing the same helpers the live pipeline
uses -- so the runner needs no changes to ``stmr`` and stays robust to metric tweaks.
"""

from __future__ import annotations

import csv
import io
import os

import torch
from fastmri import complex_abs

from stmr.metrics.metric_utils import add_cmr_eval_metrics
from stmr.metrics.motion_metrics import endpoint_error, fold_stats, inverse_consistency
from stmr.utils.spatial_transformer import GridSampleTransformer


def _frame_residual(moving: torch.Tensor, phi: torch.Tensor):
    """(base, warp) magnitude MSE between consecutive frames.

    base = MSE(|I_t|, |I_{t+1}|) without motion; warp = MSE(|warp(I_t)|, |I_{t+1}|). Mirrors
    motion_findings/verify_4_motion_bug_audit.py. Returns (nan, nan) if phi can't be aligned.
    moving is [T, 2, H, W] or [T, 2, D, H, W]; a single slice (D = 1) moves in-plane, so its phi
    lives on the (H, W) grid and the slice axis is dropped.
    """
    T = moving.shape[0]
    if T < 2:
        return float("nan"), float("nan")
    if moving.dim() == 5 and moving.shape[2] == 1:
        moving = moving[:, :, 0]
    mag = complex_abs(moving.movedim(1, -1))  # [T, *S]
    abs_phi = phi.reshape(T - 1, -1, phi.shape[-1]) if phi.dim() > 3 else phi
    if abs_phi.shape[0] != T - 1 or abs_phi.shape[-1] != moving.dim() - 2:
        return float("nan"), float("nan")
    ST = GridSampleTransformer(abs_phi, moving.shape[1:])
    warped_mag = complex_abs(ST.apply(moving[:-1]).movedim(1, -1))
    mse = lambda a, b: (a - b).pow(2).mean().item()
    return mse(mag[:-1], mag[1:]), mse(warped_mag, mag[1:])


def extract_row(spec, res_path: str, outdir: str | None = None) -> dict:
    """Load a stage's res.pt and compute the metric row used by the summary table.

    Motion metrics: folds always; endpoint error when the run saved a known motion (gt_phi);
    forward/backward consistency when the stage is the reversed counterpart of another stage
    whose res.pt is in outdir.
    """
    d = torch.load(res_path, map_location="cpu", weights_only=False)
    moving, gt, init = d["moving"], d["gt_im"], d["recon_init"]
    phi, vel = d["phi"], d.get("vel")

    m = add_cmr_eval_metrics(moving, gt, {})
    m0 = add_cmr_eval_metrics(init, gt, {})
    base, warp = _frame_residual(moving, phi)

    motion = fold_stats(phi, *gt.shape[2:])
    if d.get("gt_phi") is not None:
        motion.update(endpoint_error(phi, d["gt_phi"], gt))
    partner = getattr(spec, "reverse_of", None)
    if partner and outdir:
        partner_path = os.path.join(outdir, partner, "res.pt")
        if os.path.exists(partner_path):
            fwd = torch.load(partner_path, map_location="cpu", weights_only=False)
            motion.update(inverse_consistency(fwd["phi"], phi, fwd["gt_im"]))

    return {
        "stage": spec.id,
        "title": spec.title,
        "init_full_psnr": float(m0["cmr evals all/full psnr"]),
        "full_psnr": float(m["cmr evals all/full psnr"]),
        "crop_psnr": float(m["cmr evals all/cropped psnr"]),
        "full_ssim": float(m["cmr evals all/full ssim"]),
        "d_vs_init": float(m["cmr evals all/full psnr"] - m0["cmr evals all/full psnr"]),
        "imdiff_base": base,
        "imdiff_warp": warp,
        "vel_rms": float(vel.pow(2).mean().sqrt()) if vel is not None else None,
        **motion,
    }


def _fmt(x, prec=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        if x != x:  # nan
            return "-"
        return f"{x:.{prec}g}" if abs(x) < 1e-2 else f"{x:.{prec}f}"
    return str(x)


def _passed_str(passed):
    if passed is None:
        return "report"
    return "PASS" if passed else "FAIL"


def render_markdown(rows: list[dict], specs_by_id: dict) -> str:
    cols = [
        ("stage", "Stage"), ("full_psnr", "PSNR"), ("crop_psnr", "crop PSNR"),
        ("init_full_psnr", "init PSNR"), ("d_vs_init", "d vs init"),
        ("imdiff_base", "imdiff base"), ("imdiff_warp", "imdiff warp"),
        ("vel_rms", "vel rms"), ("epe_crop", "EPE crop px"), ("epe_rel_crop", "EPE rel"),
        ("fwd_bwd_rel_crop", "fwd/bwd rel"), ("detJ_min", "min detJ"), ("passed", "result"),
    ]
    out = io.StringIO()
    out.write("| " + " | ".join(h for _, h in cols) + " |\n")
    out.write("|" + "|".join("---" for _ in cols) + "|\n")
    for r in rows:
        cells = []
        for key, _ in cols:
            if key == "passed":
                cells.append(_passed_str(r.get("passed")))
            elif key == "stage":
                cells.append(r["stage"])
            else:
                cells.append(_fmt(r.get(key)))
        out.write("| " + " | ".join(cells) + " |\n")
    out.write("\nCriteria:\n")
    for r in rows:
        spec = specs_by_id.get(r["stage"])
        if spec is not None:
            out.write(f"- {r['stage']} ({spec.title}): {spec.criterion}\n")
            out.write(f"    expectation: {spec.expectation}\n")
    return out.getvalue()


def render_csv(rows: list[dict]) -> str:
    fields = ["stage", "title", "full_psnr", "crop_psnr", "full_ssim", "init_full_psnr",
              "d_vs_init", "imdiff_base", "imdiff_warp", "vel_rms",
              "epe_crop", "epe_rel_crop", "gt_motion_crop", "epe_tissue", "epe_air",
              "fwd_bwd_crop", "fwd_bwd_rel_crop", "detJ_min", "fold_frac", "passed"]
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return out.getvalue()


def write_summary(rows: list[dict], specs_by_id: dict, outdir: str) -> str:
    md = render_markdown(rows, specs_by_id)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "summary.md"), "w") as fh:
        fh.write(md)
    with open(os.path.join(outdir, "summary.csv"), "w") as fh:
        fh.write(render_csv(rows))
    return md
