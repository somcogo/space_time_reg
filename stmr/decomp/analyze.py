"""Regenerate figures + print the responsibility summary from a saved res.pt (no retrain)."""

from __future__ import annotations

import os

import torch

from .figures import make_figures


def analyze(root: str, out: str | None = None):
    d = torch.load(os.path.join(root, "res.pt"), map_location="cpu", weights_only=False)
    out = out or os.path.join(root, "figures")
    make_figures(d, out)
    r = d["responsibility"]
    ds, rs = r["disp_split"], r["resid_split"]
    print(f"displacement split : phi1 {ds['frac1']*100:.1f}% ({ds['m1_px']:.3f}px) | "
          f"phi2 {ds['frac2']*100:.1f}% ({ds['m2_px']:.3f}px)")
    print(f"residual explained : phi1 {rs['explained_phi1']*100:.1f}% | "
          f"phi2 {rs['explained_phi2']*100:.1f}% | unexplained {rs['unexplained']*100:.1f}%")
    if "r_base_large" in rs:
        print(f"capacity (MSE)     : two-stage r2={rs['r2']:.3e} | "
              f"large-alone={rs['r_base_large']:.3e} | small-alone={rs.get('r_base_small', float('nan')):.3e}")
    print(f"detJ min           : phi1 {r['detJ']['phi1']['detJ_min']:.3f} "
          f"(fold {r['detJ']['phi1']['foldover_frac']*100:.2f}%) | "
          f"phi2 {r['detJ']['phi2']['detJ_min']:.3f} (fold {r['detJ']['phi2']['foldover_frac']*100:.2f}%)")
    print(f"wrote figures -> {out}")
    return d
