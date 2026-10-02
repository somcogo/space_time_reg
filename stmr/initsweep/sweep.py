"""Sweep the init's regularisation weight, launched as parallel subprocesses.

Each grid point is one full init solve (`python -m stmr.initsweep run`) into
log/initsweep/<root>/<label>/. `aggregate` reads each run's res.pt into a table plus the
summary and L-curve figures. Mirrors stmr/decomp/sweep.py.

The grid is (lambda_init_recon x recon_scale) so adding the scale axis later is just passing
more values -- with a single scale it collapses to the 1D lambda sweep.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

# log-spaced, with 0 as the no-reg control (prior off => nothing fills the k-space null space)
LAMBDAS = [0.0, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 3e-1, 1.0]
SCALES = [None]          # None = leave the config's recon_scale untouched (2nd axis later)


def _label(lam: float, scale: float | None) -> str:
    return f"lam{lam:g}" + ("" if scale is None else f"_scale{scale:g}")


def build_grid(lambdas=None, scales=None):
    lambdas = LAMBDAS if lambdas is None else lambdas
    scales = SCALES if scales is None else scales
    return [(lam, sc, _label(lam, sc)) for sc in scales for lam in lambdas]


def _cli(lam, scale, label, gpu, root, extra):
    exp = f"{root}/{label}"
    sets = [f"lambda_init_recon={lam}"]
    if scale is not None:
        sets.append(f"recon_scale={scale}")
    sets.extend(extra or [])
    return [sys.executable, "-m", "stmr.initsweep", "run", "--exp-name", exp,
            "--device", f"cuda:{gpu}", "--set", *sets]


def launch(root="lambda_sweep", gpus=(0, 1, 2, 3), dry_run=False, lambdas=None,
           scales=None, extra=None):
    pts = build_grid(lambdas, scales)
    if dry_run:
        for i, (lam, sc, lb) in enumerate(pts):
            print(" ".join(_cli(lam, sc, lb, gpus[i % len(gpus)], root, extra)))
        return {}
    pending, running, free, results = list(pts), {}, list(gpus), {}

    def start(pt, gpu):
        lam, sc, lb = pt
        od = os.path.join("log/initsweep", root, lb)
        os.makedirs(od, exist_ok=True)
        logf = open(os.path.join(od, "run.log"), "w")
        proc = subprocess.Popen(_cli(lam, sc, lb, gpu, root, extra), stdout=logf, stderr=logf)
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


def aggregate(root="lambda_sweep", out=None):
    import csv

    import matplotlib
    import numpy as np
    import torch
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = os.path.join("log/initsweep", root)
    out = out or os.path.join(base, "summary")
    os.makedirs(out, exist_ok=True)

    rows = []
    for lb in sorted(os.listdir(base)):
        rp = os.path.join(base, lb, "res.pt")
        if not os.path.exists(rp):
            continue
        d = torch.load(rp, map_location="cpu", weights_only=False)
        cfg, fin = d["config"], d["final"]
        rows.append({
            "label": lb, "lambda_init_recon": cfg["lambda_init_recon"],
            "recon_scale": cfg["recon_scale"], "iters": len(d["steps"]),
            "wall_time": d.get("wall_time", float("nan")),
            "psnr": fin["cmr evals all/full psnr"], "ssim": fin["cmr evals all/full ssim"],
            "nmse": fin["cmr evals all/full nsme"],
            "psnr_crop": fin["cmr evals all/cropped psnr"],
            "data_fit": fin["energy/data_fit"], "reg": fin["energy/reg"],
            "measured_residual": fin["dc/measured_residual"],
            "unmeasured_energy": fin["dc/unmeasured_energy"],
            "sharpness": fin["recon/sharpness"],
        })
    if not rows:
        print("no completed runs found"); return rows
    rows.sort(key=lambda r: (r["recon_scale"], r["lambda_init_recon"]))

    cols = list(rows[0])
    with open(os.path.join(out, "sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    lam = np.array([r["lambda_init_recon"] for r in rows])
    # log-x needs lambda>0; the 0 control is drawn separately as a horizontal reference
    pos = lam > 0
    zero = ~pos

    def _x(a):
        return lam[a]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), layout="constrained")
    panels = [("psnr", "final PSNR (dB)"), ("unmeasured_energy", "unmeasured k-space energy"),
              ("sharpness", "sharpness (mean |grad|)")]
    for ax, (key, ttl) in zip(axes, panels):
        v = np.array([r[key] for r in rows], dtype=float)
        ax.plot(_x(pos), v[pos], "o-", color="#0072B2")
        if zero.any():
            ax.axhline(v[zero][0], color="#D55E00", ls="--",
                       label=f"lambda=0 (no reg): {v[zero][0]:.4g}")
            ax.legend(fontsize=7)
        ax.set_xscale("log"); ax.set_xlabel("lambda_init_recon"); ax.set_ylabel(ttl)
        ax.grid(alpha=.3); ax.set_title(ttl, fontsize=9)
    fig.suptitle("init reg sweep", fontsize=12)
    fig.savefig(os.path.join(out, "sweep_summary.png"), dpi=120); plt.close(fig)

    # L-curve: the classic regularisation-parameter selection plot (data misfit vs reg energy,
    # parameterised by lambda); the corner is the usual pick.
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    df = np.array([r["data_fit"] for r in rows], float)
    rg = np.array([r["reg"] for r in rows], float)
    ok = (df > 0) & (rg > 0)
    ax.plot(df[ok], rg[ok], "o-", color="#0072B2")
    for r, x, y, k in zip(rows, df, rg, ok):
        if k:
            ax.annotate(f"{r['lambda_init_recon']:g}", (x, y), fontsize=7,
                        textcoords="offset points", xytext=(4, 4))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("data fit (weighted)"); ax.set_ylabel("reg energy (weighted)")
    ax.set_title("L-curve", fontsize=10); ax.grid(alpha=.3)
    fig.savefig(os.path.join(out, "l_curve.png"), dpi=120); plt.close(fig)

    print(f"aggregated {len(rows)} runs -> {out}")
    return rows
