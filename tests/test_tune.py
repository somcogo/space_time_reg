"""stmr.tune: spec + search space -> valid pipeline configs, scoring, the objective's paired
pruning, the OOM policy and the worker loop (with a fake case runner, no GPU)."""

import os
import signal
import subprocess
import sys

import numpy as np
import optuna
import pytest
import torch
import yaml
from optuna.trial import TrialState

from stmr.config import parse_cli
from stmr.data.fft_utils import FastmriFT
from stmr.tune import objective as obj
from stmr.tune import runner
from stmr.tune.runner import CaseResult
from stmr.tune.spaces import SPACES, overrides_from_params
from stmr.tune.spec import Case, StudySpec

SPEC_A = "configs/tune/lambdas_v1.yaml"
SPEC_B = "configs/tune/siren_v1.yaml"
SPEC_C = "configs/tune/refine_v1.yaml"
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _spec(tmp_path, n_cases=8, **kw):
    data = yaml.safe_load(open(SPEC_A))
    data.update(out_dir=str(tmp_path), name="t",
                tune_cases=[{"dataset": f"cmr_P{i + 1:03d}", "slices": [3]} for i in range(n_cases)],
                holdout_cases=[], **kw)
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(data))
    return StudySpec.load(p)


@pytest.mark.parametrize("path", [SPEC_A, SPEC_B, SPEC_C])
def test_spec_baselines_build_valid_configs(path):
    spec = StudySpec.load(path)
    assert len(spec.case_names("all")) == len(set(spec.case_names("all")))
    for item in spec.enqueue:
        ov = overrides_from_params(spec.space, item["params"], spec.space_args)
        for case in spec.cases("all"):
            cfg = runner.validate(spec, runner.merge_overrides(spec, ov, case, "log/x/y", seed=1))
            assert cfg.dataset == case.overrides["dataset"] and cfg.slices == case.overrides["slices"]
            assert cfg.seed == 1 and cfg.adjoint is False and cfg.epochs == 500
    if spec.space == "lambdas":
        wi = overrides_from_params(spec.space, spec.enqueue[0]["params"])
        assert wi == {"lambda_rl2": 2500.0, "lambda_grad_phi": 0.25, "lambda_logdetJ": 0.0,
                      "lambda_recon": 0.1, "recon_lr": 1e-3, "lr": 2e-4}


def test_later_stages_extend_earlier_ones(tmp_path):
    a, b, c = (StudySpec.load(p) for p in (SPEC_A, SPEC_B, SPEC_C))
    for later in (b, c):
        assert later.case_names("all") == a.case_names("all") and later.metric == a.metric
        assert later.overrides["epochs"] == a.overrides["epochs"] and "lambda_rl2" in later.overrides
        assert later.run.timeout_h != a.run.timeout_h and later.run.crash_score == a.run.crash_score
        assert "P031" in later.resolved_yaml
        later.require_ready()
    assert [e["name"] for e in b.enqueue] == ["base_siren"] and "siren_dim" in b.resolved_yaml
    assert [e["name"] for e in c.enqueue] == ["center"] and c.overrides["siren_dim"] == 128
    # stage C's centre = stage B's pick: B's net + C's centre params give the same config
    assert overrides_from_params(c.space, c.enqueue[0]["params"], c.space_args) == {
        **{k: b.overrides[k] for k in ("lambda_rl2", "lambda_grad_phi", "lambda_logdetJ",
                                       "lambda_recon", "recon_lr")},
        "lr": c.space_args["center"]["lr"]}
    # a spec with a `todo:` refuses to launch
    data = yaml.safe_load(open(SPEC_B))
    data.update(todo="fill me", extends=os.path.abspath(SPEC_A))
    p = tmp_path / "todo.yaml"
    p.write_text(yaml.safe_dump(data))
    with pytest.raises(SystemExit, match="not ready"):
        StudySpec.load(p).require_ready()


@pytest.mark.parametrize("space", sorted(SPACES))
def test_every_sampled_point_is_a_valid_config(space):
    spec = StudySpec.load(SPEC_A)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(20):
        t = study.ask()
        ov = SPACES[space](t, **({"center": {"lr": 1e-3}} if space == "refine" else {}))
        runner.validate(spec, runner.merge_overrides(spec, ov, spec.tune_cases[0], "log/x/y"))
        study.tell(t, 0.0)


def test_format_value_round_trips_through_parse_cli():
    ov = {"voxel_spacing": [8.0, 1.5, 1.5], "slices": [3, 4], "adjoint": False,
          "lambda_rl2": 1.234e-5, "fixed_motion": "identity", "epochs": 7}
    cfg = parse_cli(runner.cli_args("configs/cmr_working_init.yaml", ov))
    assert cfg.voxel_spacing == [8.0, 1.5, 1.5] and cfg.slices == [3, 4]
    assert cfg.adjoint is False and cfg.lambda_rl2 == 1.234e-5
    assert cfg.fixed_motion == "identity" and cfg.epochs == 7


def test_score_res_ranks_recons_and_flags_blowup(tmp_path):
    torch.manual_seed(0)
    T, H, W = 4, 32, 48
    gt = torch.randn(T, 2, 1, H, W)
    mask = torch.zeros(T, 2, 1, H, W, dtype=torch.bool)
    mask[..., ::4, :] = True
    mask[..., H // 2 - 2:H // 2 + 2, :] = True
    k = FastmriFT()(gt)
    res = {
        "gt_im": gt, "moving": gt + 0.05 * torch.randn_like(gt),
        "recon_init": gt + 0.2 * torch.randn_like(gt), "fixed": torch.where(mask, k, 0),
        "metric_curve": [{"losses/total_loss": v} for v in (5.0, 2.0, 1.0, 4.0)],
        "epoch": 3, "peak_mem_gb": 1.0,
    }
    torch.save(res, tmp_path / "res.pt")
    m = runner.score_res(str(tmp_path / "res.pt"))
    for metric in ("crop_psnr", "full_psnr", "official_ssim"):
        assert m[f"{metric}/final"] > m[f"{metric}/init"] > m[f"{metric}/zf"]
    assert m["blowup"] is True and m["best_epoch"] == 3


def test_oom_policy(tmp_path, monkeypatch):
    """auto: OOM on an otherwise empty GPU is scored (deterministic), on a shared one retried."""
    spec = _spec(tmp_path, run={"oom_retries": 1, "oom_wait_min": 0})
    calls = []

    def fake_subprocess(cmd, env, run_dir, timeout_s):
        calls.append(run_dir)
        return "crash", 1.0, "torch.OutOfMemoryError: CUDA out of memory."

    monkeypatch.setattr(runner, "_run_subprocess", fake_subprocess)
    case = Case("P001_s3", {"dataset": "cmr_P001", "slices": [3]})
    ov = overrides_from_params("lambdas", spec.enqueue[0]["params"])

    monkeypatch.setattr(runner, "gpu_memory_used_mib", lambda gpu: 17)
    res = runner.run_case(spec, ov, case, str(tmp_path / "a" / "P001_s3"), gpu=0, log=lambda *_: None)
    assert res.status == "oom" and res.score == spec.run.crash_score and len(calls) == 1

    monkeypatch.setattr(runner, "gpu_memory_used_mib", lambda gpu: 9000)
    with pytest.raises(runner.OOMError):
        runner.run_case(spec, ov, case, str(tmp_path / "b" / "P001_s3"), gpu=0, log=lambda *_: None)
    assert len(calls) == 3  # first try + 1 retry


def _fake_runner(offsets):
    """Score peaks at lambda_rl2 = 1e3; case difficulty offsets are large (as between patients)."""
    def run_case(spec, ov, case, run_dir, gpu, seed=None, log=print):
        idx = int(case.overrides["dataset"][-3:]) - 1
        q = -(np.log10(ov["lambda_rl2"]) - 3.0) ** 2
        return CaseResult(case.name, "ok", float(q + offsets[idx]),
                          {"official_ssim/final": q + offsets[idx] + 1, "official_ssim/init": 1.0},
                          run_dir, 60.0)
    return run_case


def test_wilcoxon_pruned_trials_never_beat_the_best(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    offsets = np.linspace(-3, 3, len(spec.tune_cases))
    monkeypatch.setattr(runner, "run_case", _fake_runner(offsets))
    study = optuna.create_study(direction="maximize", pruner=obj.make_pruner(spec),
                                sampler=optuna.samplers.RandomSampler(seed=0))
    study.optimize(obj.Objective(spec, gpu=0, log=lambda *_: None), n_trials=40)
    trials = study.get_trials(states=(TrialState.COMPLETE,))
    pruned = [t for t in trials if t.user_attrs["pruned"]]
    full = [t for t in trials if not t.user_attrs["pruned"]]
    assert pruned and full
    assert study.best_trial.user_attrs["pruned"] is False
    # a pruned trial's estimate is below the best at its time, hence below the final best
    assert max(t.value for t in pruned) < max(t.value for t in full)
    # the paired estimate recovers the trial's mean over all cases (case offsets cancel); it is
    # only ever clamped down, to just below the best trial of its time
    true = {t.number: -(np.log10(t.params["lambda_rl2"]) - 3.0) ** 2 + offsets.mean() for t in pruned}
    assert all(t.value <= true[t.number] + 1e-9 for t in pruned)
    assert np.median([t.value - true[t.number] for t in pruned]) == pytest.approx(0.0, abs=1e-9)
    assert all(t.user_attrs["n_cases"] < len(spec.tune_cases) for t in pruned)


def test_worker_runs_baselines_then_stops_at_budget(tmp_path, monkeypatch):
    spec = _spec(tmp_path, n_cases=4, n_trials=5)
    monkeypatch.setattr(runner, "run_case", _fake_runner(np.zeros(4)))
    old = signal.getsignal(signal.SIGTERM)
    try:
        obj.create(spec)
        obj.worker(spec, gpu=0)
    finally:
        signal.signal(signal.SIGTERM, old)
    study = obj.load(spec)
    trials = study.get_trials()
    assert [t.user_attrs.get("baseline") for t in trials[:2]] == ["working_init", "rl2_100"]
    assert trials[0].params["lambda_rl2"] == 2500.0
    assert 5 <= len(trials) and all(t.state == TrialState.COMPLETE for t in trials)
    params = [tuple(sorted(t.params.items())) for t in trials]
    assert len(set(params)) == len(params)  # no repeated points
    with pytest.raises(ValueError, match="tune_cases differ"):
        obj.create(_spec(tmp_path, n_cases=3))


def test_recover_stale_requeues_dead_workers_trial(tmp_path):
    spec = _spec(tmp_path, enqueue=[])
    study = obj.create(spec)
    t = study.ask()
    t.suggest_float("lambda_rl2", 1e1, 1e5, log=True)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    t.set_user_attr("worker", f"{obj.HOST}:{dead.pid}:gpu0")
    alive = study.ask()
    alive.set_user_attr("worker", f"{obj.HOST}:{os.getpid()}:gpu1")
    assert obj.recover_stale(study, log=lambda *_: None) == 1
    states = {tr.number: tr.state for tr in study.get_trials()}
    assert states[t.number] == TrialState.FAIL and states[alive.number] == TrialState.RUNNING
    waiting = study.get_trials(states=(TrialState.WAITING,))
    assert len(waiting) == 1 and waiting[0].user_attrs["requeued_from"] == t.number


def test_refine_space_samples_around_center_and_keeps_zeros():
    args = {"center": {"lambda_rl2": 400.0, "lr": 1e-3, "lambda_logdetJ": 0.0},
            "factor": 4.0, "factors": {"lr": 2.0}}
    assert overrides_from_params("refine", args["center"], args) == args["center"]
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(50):
        t = study.ask()
        ov = SPACES["refine"](t, **args)
        assert 100.0 <= ov["lambda_rl2"] <= 1600.0 and 5e-4 <= ov["lr"] <= 2e-3
        assert ov["lambda_logdetJ"] == 0.0 and "lambda_logdetJ" not in t.params
        study.tell(t, 0.0)
