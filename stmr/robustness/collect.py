"""Gather robustness-sweep artifacts (res.pt + tensorboard) into RunRecords, group by axis.

No training is re-run: everything is computed from saved tensors and logged scalars.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from stmr.ablation.report import _frame_residual
from stmr.utils.spatial_utils import generate_coord_tensor
from stmr.viz.deform import load_run

from .jacobian import detJ_from_phi, detJ_stats

TB_KEYS = [
    "cmr evals all/full psnr",
    "cmr evals all/full nsme",
    "Main metrics/GT error",
    "Main metrics/vel H1 error",
    "Main metrics/vel L2 error",
    "Main metrics/vel grad L2 error",
]


def load_tb_scalars(run_dir, keys=TB_KEYS) -> dict:
    """Read scalar curves from run_dir/tensorboard. Returns {key: (steps, values)}."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tb = os.path.join(run_dir, "tensorboard")
    ea = EventAccumulator(tb, size_guidance={"scalars": 0})
    ea.Reload()
    avail = set(ea.Tags().get("scalars", []))
    out = {}
    for k in keys:
        if k in avail:
            ev = ea.Scalars(k)
            out[k] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    return out


@dataclass
class RunRecord:
    scene: str
    axis: str
    key: str
    run_dir: str
    phi: torch.Tensor
    H: int
    W: int
    gt_im: torch.Tensor
    detJ: dict
    imdiff_base: float
    imdiff_warp: float
    imdiff_ratio: float
    curves: dict = field(default_factory=dict)
    best_epoch: int = -1
    best_full_psnr: float = float("nan")
    vel_l2: float = float("nan")
    vel_h1: float = float("nan")

    def displacement(self) -> torch.Tensor:
        """[T-1, H, W, 2] displacement field (phi - identity) in normalized units."""
        coord = generate_coord_tensor((self.H, self.W), "cpu")
        return (self.phi - coord).reshape(-1, self.H, self.W, 2)


def _record_from_res(scene, axis, key, run_dir) -> RunRecord:
    d, _ = load_run(os.path.join(run_dir, "res.pt"), device="cpu")
    phi, gt = d["phi"], d["gt_im"]
    T, _, H, W = gt.shape
    base, warp = _frame_residual(d["moving"], phi)
    ratio = warp / base if base else float("nan")
    detJ = detJ_from_phi(phi, H, W)
    rec = RunRecord(scene, axis, key, run_dir, phi=phi, H=H, W=W, gt_im=gt,
                    detJ=detJ_stats(detJ), imdiff_base=base, imdiff_warp=warp,
                    imdiff_ratio=ratio)
    try:
        rec.curves = load_tb_scalars(run_dir)
        if "cmr evals all/full psnr" in rec.curves:
            s, v = rec.curves["cmr evals all/full psnr"]
            i = int(np.argmax(v))
            rec.best_epoch, rec.best_full_psnr = int(s[i]), float(v[i])
        for attr, k in (("vel_l2", "Main metrics/vel L2 error"),
                        ("vel_h1", "Main metrics/vel H1 error")):
            if k in rec.curves:
                setattr(rec, attr, float(rec.curves[k][1][-1]))
    except Exception as e:  # tensorboard optional; tensors are the source of truth
        print(f"  (tb scalars unavailable for {run_dir}: {e})")
    return rec


def collect_sweep(root) -> list[RunRecord]:
    """Walk root/<scene>/<axis>/<key>/<stage>/res.pt into RunRecords."""
    recs = []
    for res in sorted(glob.glob(os.path.join(root, "*", "*", "*", "*", "res.pt"))):
        run_dir = os.path.dirname(res)
        rel = os.path.relpath(res, root).split(os.sep)
        scene, axis, key = rel[0], rel[1], rel[2]
        try:
            recs.append(_record_from_res(scene, axis, key, run_dir))
        except Exception as e:
            print(f"  (skip {res}: {e})")
    return recs


def group_by_axis(records) -> dict:
    """{(scene, axis): [records]} sorted by key."""
    groups: dict = {}
    for r in records:
        groups.setdefault((r.scene, r.axis), []).append(r)
    for k in groups:
        groups[k].sort(key=lambda r: r.key)
    return groups


def disagreement_map(records):
    """Pixelwise displacement STD across a set of runs. Returns (std_mag [T-1,H,W],
    mean_disp [T-1,H,W,2]). Runs must share H, W, T."""
    disps = torch.stack([r.displacement() for r in records], dim=0)  # [n, T-1, H, W, 2]
    mean = disps.mean(0)
    # population std (unbiased=False) so a single-run group yields 0, not NaN
    std_mag = disps.std(0, unbiased=False).norm(dim=-1)  # magnitude of per-component STD
    return std_mag, mean
