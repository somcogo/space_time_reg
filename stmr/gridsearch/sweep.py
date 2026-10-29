"""Hyperparameter grid on the full pipeline (configs/cmr_working_init.yaml), in stages.

    A      lr x lambda_rl2 x (lambda_grad_phi / lambda_rl2) x lambda_init_recon
    A2     A's extension: lambda_rl2 and lambda_init_recon trade off along a diagonal (only the
           balance matters), so extend it down, a lower grad_phi ratio, and 4000 epochs
    B      one run per DIAGONAL balance (its best of A+A2) x step_size x siren_dim x siren_omega
    seeds  the top-k of A+A2+B re-run with more seeds
    C      the final round on BEST (the winner of A+A2+B): longer training, higher omega (30 beat
           15 in B), and seeds of BEST at 4000 epochs (its run is in A2) for the noise floor
    P      generalisation of the tuned BASE_YAML (= BEST at 4000 epochs), every slice x SEEDS_P:
           PATIENTS_PROC from the processed files (plus one raw run each, to confirm the processing
           changes nothing) and PATIENTS_RAW, which only exist as the challenge's raw files
    D      diagnosis of P's failures: the DIAG_N worst slices (blow-ups first) re-run with one
           change each (DIAG_VARIANTS), same seed and data as the P run
    R      every blow-up of P re-run with exact(er) gradients, RERUN_VARIANTS (D found the 2-step
           Euler adjoint at fault); all of one variant first

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
FAST_FAIL_S = 120
COOLDOWN_S = 900
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
# stage P (P001 slice 0 is the tuning slice)
PATIENTS_PROC = ["001", "002", "003", "004", "005"]
PATIENTS_RAW = [f"{i:03d}" for i in range(6, 21)]
SEEDS_P = [0, 1, 2]
RAW_SAX = "data/raw/cmrxrecon/ChallengeData/SingleCoil/Cine/TrainingSet/FullSample/P{}/cine_sax.mat"
# stage D. A blow-up: the total loss later exceeds BLOWUP x its minimum (the motion diverges).
BLOWUP = 3.0
DIAG_N = 16
DIAG_VARIANTS = {"bp": {"adjoint": False},  # exact gradients: is the 2-step Euler adjoint at fault?
                 "lr1e-4": {"lr": 1e-4},
                 "ss0.1": {"step_size": 0.1}}
RERUN_VARIANTS = {v: DIAG_VARIANTS[v] for v in ("bp", "ss0.1")}  # bp first; ss0.1 as the fallback
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


def _n_slices(p):
    import h5py
    with h5py.File(RAW_SAX.format(p), "r") as f:
        return f["kspace_single_full"].shape[1]


def grid_patients():
    # seed-major, so a partial run still covers every slice once
    n = {p: _n_slices(p) for p in PATIENTS_PROC + PATIENTS_RAW}
    pts = []
    for sd in SEEDS_P:
        for p in PATIENTS_PROC + PATIENTS_RAW:
            raw = p in PATIENTS_RAW
            for sl in range(n[p]):
                pts.append((f"P{p}_s{sl}_seed{sd}" + ("_raw" if raw else ""),
                            {"dataset": f"cmr_P{p}", "slices": [sl], "seed": sd,
                             "cmr_raw": raw}))
        if sd == SEEDS_P[0]:
            for p in PATIENTS_PROC:  # the raw check, on the mid slice
                sl = n[p] // 2
                pts.append((f"P{p}_s{sl}_seed{sd}_raw",
                            {"dataset": f"cmr_P{p}", "slices": [sl], "seed": sd,
                             "cmr_raw": True}))
    return pts


def grid_diag(src_root):
    """DIAG_VARIANTS on the DIAG_N worst distinct slices of stage P in src_root: the blow-ups
    (by final-PSNR loss), then the runs whose best-loss checkpoint is below the init."""
    rows = [r for r in collect(src_root) if r["stage"] == "P"]
    for r in rows:
        r["drop"] = r["init_crop"] - r["final_crop"]
    bad = sorted([r for r in rows if r["blowup"]], key=lambda r: -r["drop"])
    bad += sorted([r for r in rows if not r["blowup"] and r["bestloss_crop"] < r["init_crop"]],
                  key=lambda r: r["bestloss_crop"] - r["init_crop"])
    cases, seen = [], set()
    for r in bad:  # the worst seed of each slice
        if (r["dataset"], r["slice_number"]) not in seen and len(cases) < DIAG_N:
            seen.add((r["dataset"], r["slice_number"]))
            cases.append(r)
    pts = []
    for r in cases:
        base = {"dataset": r["dataset"], "slices": [r["slice_number"]], "seed": r["seed"],
                "cmr_raw": r["cmr_raw"]}
        pts += [(f"{r['label']}_{v}", {**base, **ov}) for v, ov in DIAG_VARIANTS.items()]
    return pts


def grid_rerun(src_root):
    """RERUN_VARIANTS of every stage-P blow-up in src_root, labelled as in stage D. Reads the
    aggregate's summary.csv when it has the blow-up flags (re-collecting P takes ~15 min)."""
    path = os.path.join(LOG_ROOT, src_root, "summary.csv")
    with open(path) as fh:
        cached = list(csv.DictReader(fh))
    if cached and "blowup" in cached[0]:
        rows = [{**r, "slice_number": int(r["slice_number"]), "seed": int(r["seed"]),
                 "cmr_raw": r["cmr_raw"] == "True"}
                for r in cached if r["stage"] == "P" and r["blowup"] == "True"]
    else:
        rows = [r for r in collect(src_root) if r["stage"] == "P" and r["blowup"]]
    pts = []
    for v, ov in RERUN_VARIANTS.items():
        for r in rows:
            pts.append((f"{r['label']}_{v}",
                        {"dataset": r["dataset"], "slices": [r["slice_number"]],
                         "seed": r["seed"], "cmr_raw": r["cmr_raw"], **ov}))
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


def idle_gpus():
    """GPUs with no compute process of anyone on them."""
    def query(q):
        out = subprocess.run(["nvidia-smi", f"--query-{q}", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout
        return [[x.strip() for x in line.split(",")] for line in out.splitlines() if line.strip()]
    index = {bus.lower(): int(i) for i, bus in query("gpu=index,pci.bus_id")}
    busy = {index[bus.lower()] for bus, in query("compute-apps=gpu_bus_id")}
    return sorted(set(index.values()) - busy)


def launch(pts, root, stage, gpus, per_gpu=1, dry_run=False):
    """gpus: GPU indices, or "auto": one run on every GPU that has no compute process of anyone,
    re-checked as runs finish, so GPUs other users free up are picked up (per_gpu is ignored)."""
    exp_base = os.path.join(LOG_ROOT.removeprefix("log/"), root, stage)
    todo = [(lb, ov) for lb, ov in pts
            if not os.path.exists(os.path.join("log", exp_base, lb, "res.pt"))]
    print(f"stage {stage}: {len(pts)} points, {len(pts) - len(todo)} already done", flush=True)
    auto = gpus == "auto"
    slots = [] if auto else [g for g in gpus for _ in range(per_gpu)]
    if dry_run:
        for i, (lb, ov) in enumerate(todo):
            gpu = "auto" if auto else slots[i % len(slots)]
            print(" ".join(_cli(ov, f"{exp_base}/{lb}", f"cuda:{gpu}")))
        return {}
    pending, running, free, results = list(todo), {}, list(range(len(slots))), {}
    retried, cooling = set(), {}  # cooling: slot -> time it may take runs again

    def start(pt, slot):
        lb, ov = pt
        od = os.path.join("log", exp_base, lb)
        os.makedirs(od, exist_ok=True)
        logf = open(os.path.join(od, "run.log"), "w")
        proc = subprocess.Popen(_cli(ov, f"{exp_base}/{lb}", f"cuda:{slots[slot]}"),
                                stdout=logf, stderr=logf)
        running[slot] = (proc, pt, logf, time.time())
        print(f"[start] {lb} on cuda:{slots[slot]}", flush=True)

    while pending or running:
        for s, until in list(cooling.items()):
            if time.time() >= until:
                del cooling[s]
                if not auto:
                    free.append(s)
        if auto and pending:  # a slot per idle GPU that none of our runs holds (or rests on)
            ours = {slots[s] for s in list(running) + list(cooling)}
            for g in idle_gpus():
                if g not in ours:
                    slots.append(g)
                    free.append(len(slots) - 1)
        while pending and free:
            start(pending.pop(0), free.pop(0))
        time.sleep(10)
        for s, (proc, pt, logf, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            logf.close()
            lb = pt[0]
            results[lb] = rc
            print(f"[done ] {lb} rc={rc}", flush=True)
            del running[s]
            if rc != 0 and lb not in retried:  # once more, at the end of the queue
                retried.add(lb)
                pending.append(pt)
            # A run that dies within minutes points at the GPU (e.g. CUDA init failing), not the
            # config: rest the slot instead of feeding it the rest of the queue.
            if rc != 0 and time.time() - t0 < FAST_FAIL_S:
                cooling[s] = time.time() + COOLDOWN_S
                print(f"[rest ] cuda:{slots[s]} for {COOLDOWN_S // 60} min", flush=True)
            elif not auto:  # auto: the GPU comes back via idle_gpus() if nobody else took it
                free.append(s)
        if auto:
            free.clear()
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


def _slice_of(cfg):
    """The slice a run used. The sweeps are per slice; runs from before the multi-slice
    refactor saved it as slice_number, newer ones as a one-element slices list."""
    if "slice_number" in cfg:
        return cfg["slice_number"]
    slices = cfg["slices"]
    return slices[0] if len(slices) == 1 else tuple(slices)


def collect(root):
    import torch
    rows = []
    base = os.path.join(LOG_ROOT, root)
    for stage in ("A", "A2", "B", "C", "P", "D", "R", "seeds"):
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
            total = [m["losses/total_loss"] for m in curve]
            i_min = min(range(len(total)), key=total.__getitem__)
            rows.append({
                "stage": stage, "label": lb, "slice_number": _slice_of(cfg),
                **{k: cfg.get(k) for k in ("dataset", "cmr_raw", "lr",
                                           "lambda_rl2", "lambda_grad_phi", "lambda_init_recon",
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
                "loss_rise": max(total[i_min:]) / total[i_min],
                "blowup": max(total[i_min:]) > BLOWUP * total[i_min],
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


def patient_summary(root):
    """Stage P per (patient, slice): init vs the seeds' best-loss checkpoint (mean +- std; the
    honest number), the final epoch and the best epoch (a ceiling: it includes epoch 1 ~ the
    init), the blow-up count, and raw - _norm at seed 0. A slice's seeds are its processed runs,
    or its raw runs for a raw-only patient."""
    import numpy as np
    rows = [r for r in collect(root) if r["stage"] == "P"]
    if not rows:
        print("no finished stage-P runs")
        return []
    out = []
    for key in sorted({(r["dataset"], r["slice_number"]) for r in rows}):
        norm = [r for r in rows if (r["dataset"], r["slice_number"]) == key and not r["cmr_raw"]]
        raw = [r for r in rows if (r["dataset"], r["slice_number"]) == key and r["cmr_raw"]]
        n0 = [r for r in norm if r["seed"] == 0]
        diff = raw[0]["bestloss_crop"] - n0[0]["bestloss_crop"] if raw and n0 else float("nan")
        source = "processed" if norm else "raw"
        norm = norm or raw
        loss = np.array([r["bestloss_crop"] for r in norm])
        init = norm[0]["init_crop"]
        out.append({
            "dataset": key[0], "slice": key[1], "source": source, "n_seeds": len(norm),
            "init_crop": init, "bestloss_crop_mean": loss.mean(),
            "bestloss_crop_std": loss.std(ddof=1) if len(loss) > 1 else float("nan"),
            "gain_bestloss": loss.mean() - init,
            "gain_final": float(np.mean([r["final_crop"] for r in norm])) - init,
            "gain_best": float(np.mean([r["best_crop"] for r in norm])) - init,
            "n_blowup": sum(r["blowup"] for r in norm),
            "init_full": norm[0]["init_full"],
            "best_full_mean": float(np.mean([r["best_full"] for r in norm])),
            "raw_minus_norm_seed0": diff,
        })
    with open(os.path.join(LOG_ROOT, root, "patient_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"{'patient':<10}{'slice':>6}{'n':>3}{'init':>8}{'lossCrop':>9}{'+-':>6}{'gain':>7}"
          f"{'final':>7}{'best':>7}{'blowup':>7}{'raw-norm':>9}")
    for o in out:
        print(f"{o['dataset']:<10}{o['slice']:>6}{o['n_seeds']:>3}{o['init_crop']:8.3f}"
              f"{o['bestloss_crop_mean']:9.3f}{o['bestloss_crop_std']:6.3f}{o['gain_bestloss']:+7.3f}"
              f"{o['gain_final']:+7.3f}{o['gain_best']:+7.3f}{o['n_blowup']:7d}"
              f"{o['raw_minus_norm_seed0']:+9.3f}")
    for k in ("gain_bestloss", "gain_final", "gain_best"):
        g = np.array([o[k] for o in out])
        print(f"{len(out)} slices, {k}: {g.mean():+.3f} dB mean, {np.median(g):+.3f} median, "
              f"{(g < 0).sum()} below init")
    print(f"blow-ups: {sum(o['n_blowup'] for o in out)} of {len(rows)} runs")
    return out


def diag_summary(root, src_root):
    """Stage D: each diagnosed case's P run next to its variants (cropped PSNR, init-relative)."""
    base = {r["label"]: r for r in collect(src_root) if r["stage"] == "P"}
    diag = [r for r in collect(root) if r["stage"] == "D"]
    if not diag:
        print("no finished stage-D runs")
        return []
    out = []
    for lb in sorted({r["label"].rsplit("_", 1)[0] for r in diag}):
        b = base[lb]
        row = {"case": lb, "init": b["init_crop"], "P_loss": b["bestloss_crop"] - b["init_crop"],
               "P_final": b["final_crop"] - b["init_crop"], "P_blowup": b["blowup"]}
        for v in DIAG_VARIANTS:
            r = next((r for r in diag if r["label"] == f"{lb}_{v}"), None)
            row[f"{v}_loss"] = r["bestloss_crop"] - r["init_crop"] if r else float("nan")
            row[f"{v}_final"] = r["final_crop"] - r["init_crop"] if r else float("nan")
            row[f"{v}_blowup"] = r["blowup"] if r else None
        out.append(row)
    with open(os.path.join(LOG_ROOT, root, "diag_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    cols = ["P"] + list(DIAG_VARIANTS)
    print(f"{'case':<26}{'init':>7}" + "".join(f"{c + ' loss/final':>22}" for c in cols))
    for o in out:
        print(f"{o['case']:<26}{o['init']:7.2f}" + "".join(
            f"{o[c + '_loss']:+9.2f}/{o[c + '_final']:+6.2f}{'*' if o[c + '_blowup'] else ' ':>6}"
            for c in cols))
    print("gain over init at the best-loss checkpoint / final epoch; * = blow-up")
    return out
