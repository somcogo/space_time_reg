"""Re-evaluate chosen trials: ``python -m stmr.tune reeval <spec> ...``.

The study's best value is optimistic (it is the max over many noisy trials), and it was
measured on the cases it was tuned on. Before choosing a config, run the top few and the
baselines again on the held-out patients, on fresh seeds, and with controls (e.g. the same
config with identity motion: does the learned motion help at the tuned parameters?).

Every (trial, variant, case, seed) is one run_case into
<study_dir>/reeval/t<n>[-<variant>]/<case>/seed<k>/, cached by its score.json, so an
interrupted re-evaluation resumes. Seed-<run_seed> runs of the tuning cases reuse the study's
own runs. Results: <study_dir>/reeval/<tag>.csv, and a summary per (trial, variant) with a
paired Wilcoxon test against the reference row.
"""

from __future__ import annotations

import csv
import os
import queue
import threading
from typing import Any

import numpy as np
from optuna.trial import TrialState

from stmr.tune import runner
from stmr.tune.objective import load, trial_dir
from stmr.tune.report import ranked
from stmr.tune.spec import StudySpec


def select_trials(study, spec: StudySpec, top: int, numbers: list[int] | None) -> list:
    if numbers:
        by_n = {t.number: t for t in study.get_trials(deepcopy=False)}
        return [by_n[n] for n in numbers]
    chosen = ranked(study, spec)[:top]
    seen = {t.number for t in chosen}
    names = set()
    for t in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
        b = t.user_attrs.get("baseline")
        if b and b not in names and t.number not in seen:
            names.add(b)
            chosen.append(t)
    return chosen


def reeval(spec: StudySpec, gpus: list[int], per_gpu: int = 1, top: int = 3,
           numbers: list[int] | None = None, which: str = "holdout",
           seeds: list[int] | None = None, controls: list[str] | None = None,
           tag: str | None = None, extra: dict[str, Any] | None = None) -> str:
    """extra: config overrides on top of every run (e.g. {'epochs': 1500}: do the picks hold
    at a longer budget?); they go into the run labels, so their runs are cached apart."""
    study = load(spec)
    trials = select_trials(study, spec, top, numbers)
    seeds = seeds or [spec.run_seed]
    controls = controls or []
    extra = extra or {}
    extra_label = "".join(f"_{k}{runner.format_value(v)}" for k, v in extra.items())
    unknown = set(controls) - set(spec.controls)
    if unknown:
        raise ValueError(f"unknown controls {sorted(unknown)}; the spec defines {sorted(spec.controls)}")
    variants = [("", {})] + [(c, spec.controls[c]) for c in controls]
    cases = spec.cases(which)
    tune_names = set(spec.case_names("tune"))

    jobs = []
    for t in trials:
        ov = t.user_attrs.get("overrides")
        if ov is None:
            raise ValueError(f"trial {t.number} has no stored overrides")
        for vname, vov in variants:
            for case in cases:
                for seed in seeds:
                    jobs.append((t, vname, {**ov, **vov, **extra}, case, seed))

    root = os.path.join(spec.study_dir, "reeval")
    tag = tag or (f"{which}_s{'-'.join(map(str, seeds))}"
                  + (f"_{'-'.join(controls)}" if controls else "") + extra_label)
    print(f"reeval {tag}: {len(trials)} trials x {len(variants)} variants x {len(cases)} cases x "
          f"{len(seeds)} seeds = {len(jobs)} runs on {len(gpus) * per_gpu} slots")

    results: list[dict[str, Any]] = []
    lock = threading.Lock()
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)

    def work(gpu: int):
        while True:
            try:
                t, vname, ov, case, seed = q.get_nowait()
            except queue.Empty:
                return
            label = f"t{t.number:04d}" + (f"-{vname}" if vname else "") + extra_label
            run_dir = os.path.join(root, label, case.name, f"seed{seed}")
            study_run = os.path.join(trial_dir(spec, t.number), case.name)
            if (not vname and not extra and seed == spec.run_seed and case.name in tune_names
                    and os.path.exists(os.path.join(study_run, "score.json"))):
                run_dir = study_run  # the study already ran exactly this
            try:
                res = runner.run_case(spec, ov, case, run_dir, gpu, seed=seed)
                row = {"trial": t.number, "variant": vname or "-", "case": case.name,
                       "seed": seed, "status": res.status, "score": res.score,
                       "baseline": t.user_attrs.get("baseline", ""), **res.metrics}
            except runner.OOMError as e:
                row = {"trial": t.number, "variant": vname or "-", "case": case.name,
                       "seed": seed, "status": "oom", "score": float("nan"), "error": str(e)}
            with lock:
                results.append(row)
                print(f"  [{len(results)}/{len(jobs)}] t{t.number} {vname or '-'} {case.name} "
                      f"seed{seed}: {row['status']} {row['score']:+.3f}", flush=True)

    threads = [threading.Thread(target=work, args=(g,), daemon=True)
               for g in gpus for _ in range(per_gpu)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"{tag}.csv")
    keys: list[str] = []
    for r in results:
        keys += [k for k in r if k not in keys]
    results.sort(key=lambda r: (r["trial"], r["variant"], r["case"], r["seed"]))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    summarize(results, spec.metric)
    print(f"results in {path}")
    return path


def summarize(rows: list[dict[str, Any]], metric: str) -> None:
    """Per (trial, variant): mean score over cases (each case averaged over its seeds), the seed
    spread, gain over the init, and a paired comparison with the first row (Wilcoxon over cases)."""
    from scipy.stats import wilcoxon

    groups: dict[tuple, dict[str, list[dict]]] = {}
    for r in rows:
        groups.setdefault((r["trial"], r["variant"]), {}).setdefault(r["case"], []).append(r)

    def per_case(g, key):
        return {c: float(np.nanmean([r.get(key, np.nan) for r in rs])) for c, rs in g.items()}

    keys = list(groups)  # rows are sorted by trial number: the first is the first baseline
    ref = per_case(groups[keys[0]], "score") if keys else {}
    print(f"\n{'trial':>6} {'variant':>10} {'score':>8} {'seed sd':>8} {'vs init':>8} "
          f"{'above':>7} {'vs ref':>8} {'p':>6}   (ref = trial {keys[0][0] if keys else '-'}; "
          f"p: paired Wilcoxon over cases, needs >= 5)")
    for k in keys:
        g = groups[k]
        sc = per_case(g, "score")
        gain = {c: float(np.nanmean([r.get(f"{metric}/final", np.nan) - r.get(f"{metric}/init", np.nan)
                                     for r in rs])) for c, rs in g.items()}
        sds = [np.std([r["score"] for r in rs], ddof=1) for rs in g.values() if len(rs) > 1]
        sd = f"{np.mean(sds):8.3f}" if sds else f"{'-':>8}"
        common = [c for c in sc if c in ref and np.isfinite(sc[c]) and np.isfinite(ref[c])]
        diff = [sc[c] - ref[c] for c in common]
        vs_ref, p = f"{'-':>8}", f"{'-':>6}"
        if k != keys[0] and diff:
            vs_ref = f"{np.mean(diff):+8.3f}"
            if len(diff) >= 5 and any(d != 0 for d in diff):
                p = f"{wilcoxon(diff).pvalue:6.3f}"
        above = sum(v > 0 for v in gain.values() if np.isfinite(v))
        print(f"{k[0]:>6} {k[1]:>10} {np.nanmean(list(sc.values())):+8.3f} {sd} "
              f"{np.nanmean(list(gain.values())):+8.3f} {above:>3}/{len(gain):<3} {vs_ref} {p}")
