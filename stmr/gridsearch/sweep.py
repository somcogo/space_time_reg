"""Hyperparameter grid on the full pipeline (configs/cmr_working_init.yaml), in stages.

    A      lr x lambda_rl2 x (lambda_grad_phi / lambda_rl2) x lambda_init_recon
    A2     A's extension: lambda_rl2 and lambda_init_recon trade off along a diagonal (only the
           balance matters), so extend it down, a lower grad_phi ratio, and 4000 epochs
    B      one run per DIAGONAL balance (its best of A+A2) x step_size x siren_dim x siren_omega
    seeds  the top-k of A+A2+B re-run with more seeds
    C      the final round on BEST (the winner of A+A2+B): longer training, higher omega (30 beat
           15 in B), and seeds of BEST at 4000 epochs (its run is in A2) for the noise floor

Every run uses the adjoint: direct backprop OOMs a 44 GB A40 at the largest stage-B point
(siren_dim 128, step 0.1), and the grid stays on one gradient method throughout. `probe` times
the largest config. Runs go to log/cmr/grid/<root>/<stage>/<label>/ as
`python -m stmr.cli` subprocesses, one per GPU slot; a run whose res.pt exists is skipped, so
re-launching a stage resumes it. Ranked by the best cropped (heart-region) PSNR over the metric
epochs; the cropped PSNR of the best-loss checkpoint (what res.pt holds) is reported alongside
as the validation number. Full-image PSNRs are kept as secondary columns.
"""

from __future__ import annotations

import csv
import itertools
import os
import subprocess
import sys
import time

BASE_YAML = "configs/cmr_working_init.yaml"
LOG_ROOT = "log/cmr/grid"
CROP = "cmr evals all/cropped psnr"  # the ranking metric (heart region)
FULL = "cmr evals all/full psnr"
COMMON = {"adjoint": True, "lr_schedule": "cosine", "lr_min": 1e-5}

# stage A
LRS = [1e-4, 2e-4]
RL2S = [3e4, 1e5, 3e5]
GP_RATIOS = [3e-5, 1e-4, 3e-4]  # lambda_grad_phi = ratio * lambda_rl2 (base: 1e1 / 1e5)
LIRS = [0.1, 0.2, 0.4]
# stage A2 (lr makes no difference in A; fixed at the best run's)
LR_A2 = 2e-4
DIAG_EXT = [(1e4, 0.05), (1e4, 0.1), (3e4, 0.05)]  # (lambda_rl2, lambda_init_recon)
GP_RATIOS_EXT = [3e-5, 1e-4]
GP_RATIO_LOW = 1e-5
EPOCHS_LONG = 4000
# stage B
DIAGONAL = [(3e4, 0.1), (1e5, 0.2), (3e5, 0.4)]  # the best (lambda_rl2, lambda_init_recon) of A
# stage C; B: siren_dim 128 / step_size 0.1 bought nothing, so BEST keeps A's net and step
BEST = {"lr": 2e-4, "lambda_rl2": 3e4, "lambda_grad_phi": 3.0, "lambda_init_recon": 0.1,
        "step_size": 0.5, "siren_dim": 64, "siren_omega": 30}
EPOCHS_C = [6000, 8000]
OMEGAS_C = [45, 60]
SEEDS_C = [1, 2]
STEPS = [0.1, 0.5]
DIMS = [64, 128]
OMEGAS = [15, 30]
BASE_B = {"step_size": 0.5, "siren_dim": 64, "siren_omega": 30}  # stage A's fixed values
# the probe: the most memory-hungry point of any stage (finest step, widest net)
PROBE = {"lr": 2e-4, "step_size": min(STEPS), "siren_dim": max(DIMS), "siren_omega": 30,
         "epochs": 30, "metric_cadence": 10, "log_cadence": 10}


def _label_a(lr, rl2, ratio, lir):
    return f"lr{lr:g}_rl2{rl2:g}_gp{ratio:g}_lir{lir:g}"


def grid_a():
    return [_point_a(*p) for p in itertools.product(LRS, RL2S, GP_RATIOS, LIRS)]


def _point_a(lr, rl2, ratio, lir):
    return (_label_a(lr, rl2, ratio, lir),
            {"lr": lr, "lambda_rl2": rl2, "lambda_grad_phi": ratio * rl2,
             "lambda_init_recon": lir})


def grid_a2(root):
    # the long runs first, so they don't straggle at the end of the stage
    pts = [(f"{row['label']}_ep{EPOCHS_LONG}", {**_overrides_of(row), "epochs": EPOCHS_LONG})
           for row in _best_per_balance(root, ["A"])]
    for (rl2, lir), ratio in itertools.product(DIAG_EXT, GP_RATIOS_EXT):
        pts.append(_point_a(LR_A2, rl2, ratio, lir))
    for rl2, lir in DIAGONAL:
        pts.append(_point_a(LR_A2, rl2, GP_RATIO_LOW, lir))
    return pts


def grid_b(root):
    pts = []
    for row in _best_per_balance(root, ["A", "A2"]):
        parent = _overrides_of(row)
        for step, dim, omega in itertools.product(STEPS, DIMS, OMEGAS):
            arch = {"step_size": step, "siren_dim": dim, "siren_omega": omega}
            if arch == BASE_B:
                continue  # already run in stage A
            pts.append((f"{row['label']}_ss{step:g}_d{dim}_om{omega}", {**parent, **arch}))
    return pts


def grid_c():
    pts = [(f"best_ep{ep}", {**BEST, "epochs": ep}) for ep in EPOCHS_C]
    pts += [(f"best_om{om}", {**BEST, "siren_omega": om}) for om in OMEGAS_C]
    pts += [(f"best_ep{EPOCHS_LONG}_seed{sd}", {**BEST, "epochs": EPOCHS_LONG, "seed": sd})
            for sd in SEEDS_C]
    return pts


def grid_seeds(root, top=3, seeds=(1, 2)):
    pts = []
    for row in _top(root, ["A", "A2", "B"], top):
        for seed in seeds:
            pts.append((f"{row['label']}_seed{seed}", {**_overrides_of(row), "seed": seed}))
    return pts


def _overrides_of(row):
    keys = ["lr", "lambda_rl2", "lambda_grad_phi", "lambda_init_recon", "step_size",
            "siren_dim", "siren_omega", "epochs"]
    return {k: row[k] for k in keys}


def _best_per_balance(root, stages):
    """For each DIAGONAL (lambda_rl2, lambda_init_recon), its best run at the base epochs."""
    rows = [r for r in collect(root) if r["stage"] in stages and r["epochs"] == 2000]
    best = []
    for rl2, lir in DIAGONAL:
        match = [r for r in rows if r["lambda_rl2"] == rl2 and r["lambda_init_recon"] == lir]
        if not match:
            raise SystemExit(f"no finished run with lambda_rl2={rl2}, lambda_init_recon={lir}")
        best.append(max(match, key=lambda r: r["best_crop"]))
    return best


def _top(root, stages, k):
    rows = [r for r in collect(root) if r["stage"] in stages]
    if not rows:
        raise SystemExit(f"no finished runs of stage(s) {stages} under {root}")
    return sorted(rows, key=lambda r: -r["best_crop"])[:k]


def _cli(overrides, exp, device):
    sets = [f"{k}={v}" for k, v in {**COMMON, **overrides}.items()]
    return [sys.executable, "-m", "stmr.cli", "--config", BASE_YAML, "--set",
            f"exp_name={exp}", f"device={device}", *sets]


def launch(pts, root, stage, gpus, per_gpu=1, dry_run=False):
    exp_base = os.path.join(LOG_ROOT.removeprefix("log/"), root, stage)
    todo = [(lb, ov) for lb, ov in pts
            if not os.path.exists(os.path.join("log", exp_base, lb, "res.pt"))]
    print(f"stage {stage}: {len(pts)} points, {len(pts) - len(todo)} already done", flush=True)
    slots = [g for g in gpus for _ in range(per_gpu)]
    if dry_run:
        for i, (lb, ov) in enumerate(todo):
            print(" ".join(_cli(ov, f"{exp_base}/{lb}", f"cuda:{slots[i % len(slots)]}")))
        return {}
    pending, running, free, results = list(todo), {}, list(range(len(slots))), {}

    def start(pt, slot):
        lb, ov = pt
        od = os.path.join("log", exp_base, lb)
        os.makedirs(od, exist_ok=True)
        logf = open(os.path.join(od, "run.log"), "w")
        proc = subprocess.Popen(_cli(ov, f"{exp_base}/{lb}", f"cuda:{slots[slot]}"),
                                stdout=logf, stderr=logf)
        running[slot] = (proc, lb, logf)
        print(f"[start] {lb} on cuda:{slots[slot]}", flush=True)

    while pending or running:
        while pending and free:
            start(pending.pop(0), free.pop(0))
        time.sleep(10)
        for s, (proc, lb, logf) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                logf.close()
                results[lb] = rc
                print(f"[done ] {lb} rc={rc}", flush=True)
                del running[s]
                free.append(s)
    return results


def probe(root, gpu):
    """Short run of the largest config; prints peak memory and s/epoch."""
    import torch
    exp = os.path.join(LOG_ROOT.removeprefix("log/"), root, "probe")
    od = os.path.join("log", exp)
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "run.log"), "w") as logf:
        rc = subprocess.call(_cli(PROBE, exp, f"cuda:{gpu}"), stdout=logf, stderr=logf)
    if rc != 0:
        print(f"probe failed rc={rc}, see {od}/run.log")
        return None
    d = torch.load(os.path.join(od, "res.pt"), map_location="cpu", weights_only=False)
    ts = d["time_stamps"]
    print(f"probe {PROBE}: peak {d['peak_mem_gb']:.2f} GB, "
          f"{(ts[5] - ts[0]).mean():.3f} s/epoch (incl. metric epochs)")
    return d["peak_mem_gb"]


def _eval_psnr(moving, gt_im):
    from stmr.metrics.metric_utils import add_cmr_eval_metrics
    return add_cmr_eval_metrics(moving, gt_im, {})


def collect(root):
    import torch
    rows = []
    base = os.path.join(LOG_ROOT, root)
    for stage in ("A", "A2", "B", "C", "seeds"):
        sd = os.path.join(base, stage)
        if not os.path.isdir(sd):
            continue
        for lb in sorted(os.listdir(sd)):
            rp = os.path.join(sd, lb, "res.pt")
            if not os.path.exists(rp):
                continue
            d = torch.load(rp, map_location="cpu", weights_only=False)
            cfg, curve = d["config"], d["metric_curve"]
            best = max(curve, key=lambda m: m[CROP])
            # res.pt's recon is the best-loss snapshot, not the last epoch
            at_loss = _eval_psnr(d["moving"], d["gt_im"])
            init = _eval_psnr(d["recon_init"], d["gt_im"])
            ts = d["time_stamps"]
            rows.append({
                "stage": stage, "label": lb,
                **{k: cfg[k] for k in ("lr", "lambda_rl2", "lambda_grad_phi", "lambda_init_recon",
                                       "step_size", "siren_dim", "siren_omega", "seed",
                                       "epochs", "adjoint")},
                "best_crop": best[CROP], "best_crop_epoch": best["epoch"],
                "best_crop_ssim": best["cmr evals all/cropped ssim"],
                "bestloss_crop": float(at_loss[CROP]), "bestloss_epoch": d["epoch"],
                "bestloss_crop_ssim": float(at_loss["cmr evals all/cropped ssim"]),
                "final_crop": curve[-1][CROP], "init_crop": float(init[CROP]),
                "best_full": max(m[FULL] for m in curve),
                "bestloss_full": float(at_loss[FULL]), "init_full": float(init[FULL]),
                "peak_mem_gb": d.get("peak_mem_gb"),
                "minutes": (ts[5, -1] - ts[6, 0]) / 60,
            })
    return rows


def aggregate(root, show=15):
    rows = sorted(collect(root), key=lambda r: -r["best_crop"])
    if not rows:
        print("no completed runs found")
        return rows
    out = os.path.join(LOG_ROOT, root, "summary.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'stage':<6}{'label':<48}{'bestCrop':>9}{'@ep':>6}{'lossCrop':>9}{'@ep':>6}"
          f"{'final':>8}{'init':>8}{'bestFull':>9}")
    for r in rows[:show]:
        print(f"{r['stage']:<6}{r['label']:<48}{r['best_crop']:9.3f}{r['best_crop_epoch']:6d}"
              f"{r['bestloss_crop']:9.3f}{r['bestloss_epoch']:6d}{r['final_crop']:8.3f}"
              f"{r['init_crop']:8.3f}{r['best_full']:9.3f}")
    print(f"{len(rows)} runs -> {out}")
    return rows
