"""Sweep phi1's regularisation x size (phi2 fixed large), launched as parallel subprocesses.

Each grid point is a full two-stage decomposition run (`python -m stmr.decomp run`) into
log/decomp/<root>/<label>/. `aggregate` reads each run's res.pt responsibility into a table +
a summary heatmap. Mirrors stmr/robustness/sweep.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

SIZES = [[2, 32, 32, 2], [2, 64, 64, 2], [2, 128, 128, 2]]  # phi1 (depth 2, widths 32/64/128)
REGS = [0.0, 1e-2, 1e-1]                                     # lam_grad_phi1 = lam_logdetJ1


def _label(layers, lam):
    return f"h{layers[1]}d{len(layers) - 2}_reg{lam:g}"


def build_grid():
    return [(layers, lam, _label(layers, lam)) for layers in SIZES for lam in REGS]


def _cli(layers, lam, label, gpu, steps, root):
    exp = f"{root}/{label}"
    return [sys.executable, "-m", "stmr.decomp", "run", "--exp-name", exp,
            "--device", f"cuda:{gpu}", "--set",
            f"phi1_layers={','.join(map(str, layers))}",
            f"lam_grad_phi1={lam}", f"lam_logdetJ1={lam}",
            f"steps1={steps}", f"steps2={steps}"]


def launch(root="sweep_phi1", gpus=(0, 1, 2, 3), steps=1000, dry_run=False):
    pts = build_grid()
    if dry_run:
        for i, (lay, lam, lb) in enumerate(pts):
            print(" ".join(_cli(lay, lam, lb, gpus[i % len(gpus)], steps, root)))
        return {}
    pending, running, free, results = list(pts), {}, list(gpus), {}

    def start(pt, gpu):
        lay, lam, lb = pt
        od = os.path.join("log/decomp", root, lb)
        os.makedirs(od, exist_ok=True)
        logf = open(os.path.join(od, "run.log"), "w")
        proc = subprocess.Popen(_cli(lay, lam, lb, gpu, steps, root), stdout=logf, stderr=logf)
        running[gpu] = (proc, lb, logf)
        print(f"[start] {lb} on cuda:{gpu}", flush=True)

    while pending or running:
        while pending and free:
            start(pending.pop(0), free.pop(0))
        time.sleep(5)
        for g, (proc, lb, logf) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                logf.close(); results[lb] = rc
                print(f"[done ] {lb} rc={rc}", flush=True)
                del running[g]; free.append(g)
    return results


def aggregate(root="sweep_phi1", out=None):
    import csv

    import matplotlib
    import numpy as np
    import torch
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = os.path.join("log/decomp", root)
    out = out or os.path.join(base, "summary")
    os.makedirs(out, exist_ok=True)
    rows = []
    for lb in sorted(os.listdir(base)):
        rp = os.path.join(base, lb, "res.pt")
        if not os.path.exists(rp):
            continue
        d = torch.load(rp, map_location="cpu", weights_only=False)
        r = d["responsibility"]; p = d["params"]
        rows.append({
            "label": lb, "dim": p["phi1_layers"][1], "reg": p["lam_grad_phi1"],
            "frac1": r["disp_split"]["frac1"], "frac2": r["disp_split"]["frac2"],
            "explained_phi1": r["resid_split"]["explained_phi1"],
            "explained_phi2": r["resid_split"]["explained_phi2"],
            "unexplained": r["unexplained"]["r2_over_base"],
            "r2": r["resid_split"]["r2"],
            "detJ_min_phi1": r["detJ"]["phi1"]["detJ_min"],
        })
    if not rows:
        print("no completed runs found"); return rows
    cols = ["label", "dim", "reg", "frac1", "explained_phi1", "explained_phi2",
            "unexplained", "r2", "detJ_min_phi1"]
    with open(os.path.join(out, "sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    dims = sorted({r["dim"] for r in rows}); regs = sorted({r["reg"] for r in rows})
    def grid(key):
        m = np.full((len(dims), len(regs)), np.nan)
        for r in rows:
            m[dims.index(r["dim"]), regs.index(r["reg"])] = r[key]
        return m
    panels = [("explained_phi1", "phi1 explains (frac base)"),
              ("explained_phi2", "phi2 adds (frac base)"),
              ("unexplained", "unexplained (frac base)"),
              ("frac1", "phi1 displacement share")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4), layout="constrained")
    for ax, (key, title) in zip(axes, panels):
        m = grid(key)
        im = ax.imshow(m, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(regs))); ax.set_xticklabels([f"{x:g}" for x in regs])
        ax.set_yticks(range(len(dims))); ax.set_yticklabels([f"h{x}" for x in dims])
        ax.set_xlabel("phi1 reg"); ax.set_ylabel("phi1 width")
        for i in range(len(dims)):
            for j in range(len(regs)):
                if not np.isnan(m[i, j]):
                    ax.text(j, i, f"{m[i,j]:.2f}", ha="center", va="center",
                            color="w", fontsize=8)
        ax.set_title(title, fontsize=9); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("phi1 reg x size sweep (phi2 fixed d3h256): motion split & unexplained residual",
                 fontsize=12)
    fig.savefig(os.path.join(out, "sweep_summary.png"), dpi=120); plt.close(fig)
    print(f"aggregated {len(rows)} runs -> {out}")
    return rows
