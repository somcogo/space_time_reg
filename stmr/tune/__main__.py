"""CLI of the Optuna tuner (see stmr/tune/__init__.py for the design).

    python -m stmr.tune create  configs/tune/lambdas_v1.yaml          # make the study, enqueue baselines
    python -m stmr.tune launch  configs/tune/lambdas_v1.yaml --gpus 4,5,6,7 [--per-gpu 1]
    python -m stmr.tune launch  configs/tune/lambdas_v1.yaml --gpus auto   # every idle GPU
    python -m stmr.tune status  configs/tune/lambdas_v1.yaml [--recover]
    python -m stmr.tune stop    configs/tune/lambdas_v1.yaml          # SIGTERM; running trials re-enqueued
    python -m stmr.tune report  configs/tune/lambdas_v1.yaml
    python -m stmr.tune reeval  configs/tune/lambdas_v1.yaml --gpus 4,5,6,7 --top 3 \\
        --cases holdout --seeds 0,1 --controls identity [--set epochs=1500]
    python -m stmr.tune best    configs/tune/lambdas_v1.yaml [--trial N]  # values for the next stage
    python -m stmr.tune worker  configs/tune/lambdas_v1.yaml --gpu 6  # one worker, foreground

`launch` starts detached workers (they survive the terminal); their logs are in
<study_dir>/workers/. Add GPUs to a running study by launching again with other --gpus.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

from optuna.trial import TrialState

from stmr.tune import objective as obj
from stmr.tune.spec import REPO_ROOT, StudySpec


def idle_gpus(max_mem_mib: int = 1000, max_util: int = 10) -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         check=True).stdout
    gpus = []
    for line in out.strip().splitlines():
        i, mem, util = (int(x) for x in line.split(","))
        if mem < max_mem_mib and util < max_util:
            gpus.append(i)
    return gpus


def parse_gpus(s: str) -> list[int]:
    if s == "auto":
        g = idle_gpus()
        print(f"idle GPUs: {g}")
        return g
    return [int(x) for x in s.split(",") if x.strip()]


def _is_our_worker(pid: int, spec: StudySpec) -> bool:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            cmd = fh.read().decode(errors="replace")
    except FileNotFoundError:
        return False
    return "stmr.tune" in cmd and "worker" in cmd and os.path.basename(spec.spec_path) in cmd


def cmd_launch(spec: StudySpec, gpus: list[int], per_gpu: int) -> None:
    obj.create(spec)
    log_dir = os.path.join(spec.study_dir, "workers")
    os.makedirs(log_dir, exist_ok=True)
    workers = [w for w in obj.read_workers(spec) if _is_our_worker(w["pid"], spec)]
    busy = {(w["gpu"], w["slot"]) for w in workers}
    for g in gpus:
        for s in range(per_gpu):
            if (g, s) in busy:
                print(f"GPU {g} slot {s}: a worker is already running (pid "
                      f"{[w['pid'] for w in workers if (w['gpu'], w['slot']) == (g, s)][0]})")
                continue
            log_path = os.path.join(log_dir, f"gpu{g}_slot{s}.log")
            with open(log_path, "a") as fh:
                p = subprocess.Popen([sys.executable, "-m", "stmr.tune", "worker", spec.spec_path,
                                      "--gpu", str(g), "--slot", str(s)],
                                     cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            workers.append({"pid": p.pid, "gpu": g, "slot": s, "log": log_path,
                            "started": time.strftime("%Y-%m-%d %H:%M:%S")})
            print(f"GPU {g} slot {s}: worker pid {p.pid}, log {log_path}")
    with open(obj.worker_record(spec), "w") as fh:
        json.dump(workers, fh, indent=1)


def cmd_stop(spec: StudySpec) -> None:
    for w in obj.read_workers(spec):
        if _is_our_worker(w["pid"], spec):
            os.kill(w["pid"], signal.SIGTERM)
            print(f"SIGTERM -> worker {w['pid']} (GPU {w['gpu']})")


def cmd_status(spec: StudySpec, recover: bool) -> None:
    study = obj.load(spec)
    if recover:
        obj.recover_stale(study)
    trials = study.get_trials(deepcopy=False)
    by_state: dict[str, int] = {}
    for t in trials:
        by_state[t.state.name] = by_state.get(t.state.name, 0) + 1
    pruned = sum(bool(t.user_attrs.get("pruned")) for t in trials)
    print(f"study {spec.name}  ({spec.journal_path})")
    print(f"  trials: {by_state}  (of which pruned early: {pruned}); budget {spec.n_trials}")

    case_s = [c["runtime_s"] for t in trials for k, c in t.user_attrs.items()
              if k.startswith("case/") and c.get("status") == "ok"]
    if case_s:
        print(f"  case runs: {len(case_s)}, {sum(case_s) / 3600:.1f} GPU-h, "
              f"{sum(case_s) / len(case_s) / 60:.0f} min per case")
    full = [t for t in trials if t.state == TrialState.COMPLETE and not t.user_attrs.get("pruned")]
    if full:
        b = max(full, key=lambda t: t.value)
        print(f"  best fully-evaluated: #{b.number} {b.value:+.3f}  {b.params}")
    for t in trials:
        if t.state == TrialState.RUNNING:
            print(f"  running #{t.number}: {t.user_attrs.get('progress', '?')}  "
                  f"({t.user_attrs.get('worker', '?')})")
    alive = [w for w in obj.read_workers(spec) if _is_our_worker(w["pid"], spec)]
    print(f"  workers alive: {[(w['gpu'], w['slot'], w['pid']) for w in alive]}")


def cmd_best(spec: StudySpec, number: int | None) -> None:
    """A trial's config values as YAML, for the next stage's `overrides` (and its params, for an
    `enqueue` entry of a study with the same space)."""
    import yaml

    from stmr.tune.report import ranked, trial_row

    study = obj.load(spec)
    if number is None:
        best = ranked(study, spec)
        if not best:
            raise SystemExit("no fully evaluated trial yet")
        t = best[0]
    else:
        t = study.trials[number]
    r = trial_row(t, spec)
    print(f"# {spec.name} trial {t.number}: {spec.metric} score {t.value:+.4f} over "
          f"{r['n_cases']} cases (gain over init {r['gain_over_init']:+.4f}, above init "
          f"{r['above_init']}); check it on the holdout cases (reeval) before adopting it")
    print("# overrides:")
    print(yaml.safe_dump(t.user_attrs["overrides"], sort_keys=False).rstrip())
    print("# params (enqueue):")
    print(yaml.safe_dump({"params": t.params}, default_flow_style=None, sort_keys=False).rstrip())


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m stmr.tune")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("create", "status", "stop", "report", "best"):
        sp = sub.add_parser(name)
        sp.add_argument("spec")
        if name == "status":
            sp.add_argument("--recover", action="store_true",
                            help="fail + re-enqueue RUNNING trials whose worker died")
        if name == "report":
            sp.add_argument("--no-plots", action="store_true")
            sp.add_argument("--top", type=int, default=10)
        if name == "best":
            sp.add_argument("--trial", type=int, default=None, help="default: best fully evaluated")
    la = sub.add_parser("launch")
    la.add_argument("spec")
    la.add_argument("--gpus", required=True, help='e.g. "4,5,6,7", or "auto": every idle GPU')
    la.add_argument("--per-gpu", type=int, default=1)
    wk = sub.add_parser("worker")
    wk.add_argument("spec")
    wk.add_argument("--gpu", type=int, required=True)
    wk.add_argument("--slot", type=int, default=0)
    re = sub.add_parser("reeval")
    re.add_argument("spec")
    re.add_argument("--gpus", required=True)
    re.add_argument("--per-gpu", type=int, default=1)
    re.add_argument("--top", type=int, default=3, help="best fully-evaluated trials (+ baselines)")
    re.add_argument("--trials", default="", help="explicit trial numbers instead of --top")
    re.add_argument("--cases", default="holdout", choices=["tune", "holdout", "all"])
    re.add_argument("--seeds", default="", help="default: the study's run seed")
    re.add_argument("--controls", default="", help="names from the spec's controls")
    re.add_argument("--tag", default=None)
    re.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="config overrides on top of every run, e.g. epochs=1500")
    a = p.parse_args(argv)

    spec = StudySpec.load(a.spec)
    os.chdir(REPO_ROOT)  # relative paths in the spec (base_config, out_dir) are repo-relative
    if a.cmd == "create":
        study = obj.create(spec)
        print(f"study {spec.name}: {len(study.trials)} trials "
              f"({len(spec.enqueue)} baselines enqueued at creation), journal {spec.journal_path}")
    elif a.cmd == "launch":
        cmd_launch(spec, parse_gpus(a.gpus), a.per_gpu)
    elif a.cmd == "worker":
        obj.worker(spec, a.gpu, a.slot)
    elif a.cmd == "status":
        cmd_status(spec, a.recover)
    elif a.cmd == "stop":
        cmd_stop(spec)
    elif a.cmd == "report":
        from stmr.tune.report import write_report
        write_report(spec, plots=not a.no_plots, top=a.top)
    elif a.cmd == "best":
        cmd_best(spec, a.trial)
    elif a.cmd == "reeval":
        from stmr.tune.reeval import reeval
        extra = dict(kv.split("=", 1) for kv in a.set)  # strings; parse_cli types them
        reeval(spec, parse_gpus(a.gpus), a.per_gpu, a.top,
               [int(x) for x in a.trials.split(",") if x] or None, a.cases,
               [int(x) for x in a.seeds.split(",") if x] or None,
               [x for x in a.controls.split(",") if x] or None, a.tag, extra)


if __name__ == "__main__":
    main()
