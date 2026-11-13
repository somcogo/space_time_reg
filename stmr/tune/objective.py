"""The Optuna side: storage, sampler, pruner, the objective and the worker loop.

One worker = one GPU slot = one process running trials one after another. All workers share
a JournalStorage file (``<study_dir>/journal.log``, file-locked, safe for many processes on
one machine), so the sampler sees every finished trial whichever worker ran it.

A trial evaluates one sampled config on the tuning cases, one after another on its GPU:

* WilcoxonPruner (default): the cases run in a per-trial random order (so the study does not
  overfit the first few cases); after each, the per-case scores are compared, paired by case,
  with the best trial's by a Wilcoxon signed-rank test, and the trial stops once it is
  significantly worse. Its value is then an estimate of the mean over all cases: the best
  trial's mean plus the mean paired difference on the cases it did run. That can never exceed
  the best value, and it lets TPE learn from the pruned trial (Optuna's recommended pattern of
  returning a value instead of raising TrialPruned). ``user_attrs['pruned']`` marks such trials.
* MedianPruner: fixed case order, running mean reported per case, TrialPruned when below the
  median of earlier trials at the same case count.
"""

from __future__ import annotations

import json
import os
import random
import signal
import socket
import time
import warnings
from typing import Any

import numpy as np
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

from stmr.tune import runner
from stmr.tune.spaces import SPACES
from stmr.tune.spec import StudySpec

HOST = socket.gethostname()
# WilcoxonPruner and TPE's `group` are marked experimental; both are what this tuner relies on
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


# --- study ------------------------------------------------------------------------------------

def storage(spec: StudySpec) -> JournalStorage:
    os.makedirs(spec.study_dir, exist_ok=True)
    return JournalStorage(JournalFileBackend(spec.journal_path))


def make_sampler(spec: StudySpec, seed: int | None):
    cfg = dict(spec.sampler)
    name = cfg.pop("name", "tpe")
    cfg.pop("seed", None)
    if name == "tpe":
        # multivariate + group: model the parameters jointly (lambda_rl2 / lambda_recon trade
        # off along a diagonal) and handle the conditional ldj_ratio. constant_liar: running
        # trials count as bad guesses, so parallel workers do not all propose the same point.
        kw = {"multivariate": True, "group": True, "constant_liar": True, "n_startup_trials": 10}
        kw.update(cfg)
        return optuna.samplers.TPESampler(seed=seed, **kw)
    if name == "gp":
        return optuna.samplers.GPSampler(seed=seed, **cfg)
    return optuna.samplers.RandomSampler(seed=seed)


def make_pruner(spec: StudySpec):
    cfg = dict(spec.pruner)
    name = cfg.pop("name", "wilcoxon")
    if name == "wilcoxon":
        kw = {"p_threshold": 0.1, "n_startup_steps": 3}
        kw.update(cfg)
        return optuna.pruners.WilcoxonPruner(**kw)
    if name == "median":
        kw = {"n_startup_trials": 5, "n_warmup_steps": 2}
        kw.update(cfg)
        return optuna.pruners.MedianPruner(**kw)
    return optuna.pruners.NopPruner()


def create(spec: StudySpec) -> optuna.Study:
    """Create the study (idempotent). The first call enqueues spec.enqueue and records the case
    list; later calls check that the spec's tuning cases did not change under a running study."""
    spec.require_ready()
    study = optuna.create_study(study_name=spec.name, storage=storage(spec),
                                direction="maximize", load_if_exists=True)
    recorded = study.user_attrs.get("tune_cases")
    if recorded is None:
        study.set_user_attr("tune_cases", spec.case_names("tune"))
        study.set_user_attr("metric", spec.metric)
        study.set_user_attr("space", spec.space)
        study.set_user_attr("space_args", spec.space_args)
        study.set_user_attr("base_config", spec.base_config)
        study.set_user_attr("overrides", spec.overrides)
        study.set_user_attr("spec_yaml", spec.resolved_yaml)  # after `extends`
        for item in spec.enqueue:
            study.enqueue_trial(item["params"], user_attrs={"baseline": item.get("name", "")})
    else:
        if recorded != spec.case_names("tune"):
            raise ValueError("tune_cases differ from the ones this study was created with "
                             f"({recorded}); case ids would no longer pair. Use a new study name.")
        for key, default in (("metric", None), ("space", None), ("space_args", {})):
            if study.user_attrs.get(key, default) != getattr(spec, key):
                raise ValueError(f"{key} differs from the study's ({study.user_attrs.get(key)})")
    return study


def load(spec: StudySpec, sampler_seed: int | None = None) -> optuna.Study:
    return optuna.load_study(study_name=spec.name, storage=storage(spec),
                             sampler=make_sampler(spec, sampler_seed), pruner=make_pruner(spec))


def trial_dir(spec: StudySpec, number: int) -> str:
    return os.path.join(spec.study_dir, "trials", f"t{number:04d}")


# --- objective --------------------------------------------------------------------------------

class Objective:
    def __init__(self, spec: StudySpec, gpu: int, log=print):
        self.spec, self.gpu, self.log = spec, gpu, log
        self.cases = spec.cases("tune")
        self.pruner_name = spec.pruner.get("name", "wilcoxon")

    def __call__(self, trial: optuna.Trial) -> float:
        overrides = SPACES[self.spec.space](trial, **self.spec.space_args)
        trial.set_user_attr("worker", f"{HOST}:{os.getpid()}:gpu{self.gpu}")
        trial.set_user_attr("overrides", overrides)
        tdir = trial_dir(self.spec, trial.number)
        # fail fast (before any GPU time) on a key or value the pipeline would reject
        runner.validate(self.spec, runner.merge_overrides(self.spec, overrides, self.cases[0], tdir))

        order = list(range(len(self.cases)))
        if self.pruner_name == "wilcoxon":
            random.Random(trial.number).shuffle(order)
        scores: dict[int, float] = {}
        for k, idx in enumerate(order):
            case = self.cases[idx]
            trial.set_user_attr("progress", f"{k}/{len(order)} running {case.name}")
            res = runner.run_case(self.spec, overrides, case, os.path.join(tdir, case.name),
                                  self.gpu, seed=self.spec.run_seed, log=self.log)
            scores[idx] = res.score
            trial.set_user_attr(f"case/{case.name}", res.summary())
            self.log(f"[gpu {self.gpu}] trial {trial.number} {case.name}: {res.status} "
                     f"score {res.score:+.3f} ({res.runtime_s / 60:.0f} min)")

            if self.pruner_name == "wilcoxon":
                trial.report(res.score, step=idx)
            else:
                trial.report(float(np.mean(list(scores.values()))), step=k)
            if k + 1 < len(order) and trial.should_prune():
                trial.set_user_attr("pruned", True)
                trial.set_user_attr("n_cases", len(scores))
                trial.set_user_attr("progress", f"pruned after {len(scores)}/{len(order)}")
                if self.pruner_name != "wilcoxon":
                    raise optuna.TrialPruned()
                est = self._estimate(trial.study, scores)
                self.log(f"[gpu {self.gpu}] trial {trial.number} pruned after {len(scores)} "
                         f"cases, estimated mean {est:+.3f}")
                return est

        trial.set_user_attr("pruned", False)
        trial.set_user_attr("n_cases", len(scores))
        trial.set_user_attr("progress", "done")
        return float(np.mean(list(scores.values())))

    @staticmethod
    def _estimate(study: optuna.Study, scores: dict[int, float]) -> float:
        """Mean over all cases, estimated from the cases run: best mean + mean paired difference."""
        try:
            best = study.best_trial
        except ValueError:
            return float(np.mean(list(scores.values())))
        bv = best.intermediate_values
        common = [s for s in scores if s in bv]
        if not common:
            return float(np.mean(list(scores.values())))
        est = best.value + float(np.mean([scores[s] - bv[s] for s in common]))
        return min(est, best.value - 1e-6)  # pruned = judged worse than the best


# --- worker -----------------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def recover_stale(study: optuna.Study, log=print) -> int:
    """RUNNING trials whose worker process on this host is gone (killed hard, machine reboot):
    mark them FAIL and enqueue their params again. JournalStorage has no heartbeat, so this
    runs at every worker start and from `status --recover`."""
    n = 0
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        worker = t.user_attrs.get("worker", "")
        host, _, rest = worker.partition(":")
        if host != HOST or not rest:
            continue
        pid = int(rest.split(":")[0])
        if _pid_alive(pid):
            continue
        study._storage.set_trial_state_values(t._trial_id, TrialState.FAIL)
        study.enqueue_trial(t.params, user_attrs={"requeued_from": t.number,
                                                  **({"baseline": t.user_attrs["baseline"]}
                                                     if "baseline" in t.user_attrs else {})})
        log(f"recovered stale trial {t.number} (worker {worker} is gone), params re-enqueued")
        n += 1
    return n


def _budget_left(study: optuna.Study, n_trials: int) -> bool:
    trials = study.get_trials(deepcopy=False)
    waiting = sum(t.state == TrialState.WAITING for t in trials)
    used = sum(t.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING)
               for t in trials)
    return waiting > 0 or used < n_trials


def worker(spec: StudySpec, gpu: int, slot: int = 0) -> None:
    def log(msg):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)

    def on_term(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    spec.require_ready()
    signal.signal(signal.SIGTERM, on_term)
    base_seed = int(spec.sampler.get("seed", 0))
    # each worker its own sampler seed: identical seeds would make parallel workers -- and a
    # worker restarted on the same slot -- propose identical points in the random start-up phase
    n_prev = len(load(spec).get_trials(deepcopy=False))
    study = load(spec, sampler_seed=base_seed + 1000 * gpu + slot + 100_003 * n_prev)
    recover_stale(study, log)
    objective = Objective(spec, gpu, log)
    failures = 0
    log(f"worker {HOST}:{os.getpid()} on GPU {gpu} slot {slot}, study {spec.name}")
    while _budget_left(study, spec.n_trials):
        before = {t.number for t in study.get_trials(deepcopy=False)}
        try:
            study.optimize(objective, n_trials=1, catch=(Exception,), gc_after_trial=True)
        except KeyboardInterrupt:
            mine = [t for t in study.get_trials(deepcopy=False, states=(TrialState.FAIL,))
                    if t.number not in before and t.user_attrs.get("worker", "").endswith(
                        f":{os.getpid()}:gpu{gpu}")]
            for t in mine:  # the interrupted trial: run it again later
                study.enqueue_trial(t.params, user_attrs={"requeued_from": t.number})
            log("interrupted; the running trial was re-enqueued")
            return
        new = [t for t in study.get_trials(deepcopy=False)
               if t.number not in before
               and t.user_attrs.get("worker", "").endswith(f":{os.getpid()}:gpu{gpu}")]
        failed = any(t.state == TrialState.FAIL for t in new)
        failures = failures + 1 if failed else 0
        if failures >= spec.run.max_consecutive_failures:
            log(f"stopping: {failures} failed trials in a row (see the FAIL trials' logs)")
            return
    log("budget reached, worker done")


def worker_record(spec: StudySpec) -> str:
    return os.path.join(spec.study_dir, "workers.json")


def read_workers(spec: StudySpec) -> list[dict[str, Any]]:
    try:
        with open(worker_record(spec)) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return []
