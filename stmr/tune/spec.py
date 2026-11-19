"""A tuning study is described by one YAML file (see configs/tune/lambdas_v1.yaml).

The spec fixes everything that must not change while a study runs: the base pipeline config,
the overrides every run gets, the search space (by name, see spaces.py), the metric, and the
ordered list of tuning cases. The case order matters: a case's index is its id for the
WilcoxonPruner, which pairs each trial's per-case scores with the best trial's.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Score of a case = metric(best-loss recon) - metric(zero-filled recon), both against the GT.
# All three are higher-is-better, so every study maximizes.
METRICS = ("crop_psnr", "full_psnr", "official_ssim")
SAMPLERS = ("tpe", "gp", "random")
PRUNERS = ("wilcoxon", "median", "none")


@dataclass
class Case:
    """One evaluation instance: the config fields that select the data (dataset, slices, ...)."""
    name: str
    overrides: dict[str, Any]

    @classmethod
    def from_dict(cls, d: dict) -> Case:
        d = dict(d)
        name = d.pop("name", None) or _case_name(d)
        if "slices" in d and not isinstance(d["slices"], (list, tuple)):
            d["slices"] = [d["slices"]]
        return cls(name, d)


def _case_name(d: dict) -> str:
    ds = str(d["dataset"]).removeprefix("cmr_")
    sl = d.get("slices", [0])
    sl = sl if isinstance(sl, (list, tuple)) else [sl]
    # LAX view 0 of a patient is not its SAX slice 0; SAX names stay as they were.
    view = "" if d.get("view", "sax") == "sax" else f"_{d['view']}"
    return f"{ds}{view}_s{'-'.join(str(s) for s in sl)}"


@dataclass
class RunSettings:
    timeout_h: float = 6.0  # a case still running after this is killed and scored crash_score
    # CUDA OOM: 'retry' = someone else's memory (shared GPU): wait, retry, then fail the trial;
    # 'penalize' = the config needs too much: score crash_score; 'auto' = penalize when nothing
    # else held memory on the GPU when the run died, retry otherwise
    oom_policy: str = "auto"
    oom_retries: int = 3
    oom_wait_min: float = 10.0
    crash_score: float = -0.2  # score of a crashed / timed-out / NaN / OOM case, in metric units
    max_consecutive_failures: int = 3  # a worker stops after this many failed trials in a row
    keep_heavy_files: bool = False  # keep res.pt and np_imgs.npy (~130 MB per case)
    env: dict[str, str] = field(
        default_factory=lambda: {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})


@dataclass
class StudySpec:
    name: str
    base_config: str
    space: str
    tune_cases: list[Case]
    space_args: dict[str, Any] = field(default_factory=dict)  # keyword args of the space function
    holdout_cases: list[Case] = field(default_factory=list)
    overrides: dict[str, Any] = field(default_factory=dict)
    metric: str = "crop_psnr"
    n_trials: int = 80
    out_dir: str = "log/optuna"
    sampler: dict[str, Any] = field(default_factory=dict)
    pruner: dict[str, Any] = field(default_factory=dict)
    run: RunSettings = field(default_factory=RunSettings)
    enqueue: list[dict[str, Any]] = field(default_factory=list)  # [{name, params}] run first
    controls: dict[str, dict[str, Any]] = field(default_factory=dict)  # reeval variants
    todo: str = ""  # non-empty: the spec still has placeholders; create/launch/worker refuse
    spec_path: str = ""
    resolved_yaml: str = ""

    @property
    def study_dir(self) -> str:
        return os.path.join(self.out_dir, self.name)

    @property
    def journal_path(self) -> str:
        return os.path.join(self.study_dir, "journal.log")

    @property
    def run_seed(self) -> int:
        return int(self.overrides.get("seed", 0))

    def case_names(self, which: str = "tune") -> list[str]:
        return [c.name for c in self.cases(which)]

    def cases(self, which: str = "tune") -> list[Case]:
        if which == "tune":
            return list(self.tune_cases)
        if which == "holdout":
            return list(self.holdout_cases)
        if which == "all":
            return list(self.tune_cases) + list(self.holdout_cases)
        raise ValueError(f"cases must be tune | holdout | all, got {which!r}")

    @classmethod
    def load(cls, path: str | os.PathLike) -> StudySpec:
        data = _load_yaml(Path(path))
        resolved = yaml.safe_dump(data, sort_keys=False)  # after `extends`, stored in the study
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"{path}: unknown study keys {sorted(unknown)}")
        run_known = {f.name for f in fields(RunSettings)}
        run_unknown = set(data.get("run") or {}) - run_known
        if run_unknown:
            raise ValueError(f"{path}: unknown run keys {sorted(run_unknown)}")
        data["tune_cases"] = [Case.from_dict(c) for c in data.get("tune_cases") or []]
        data["holdout_cases"] = [Case.from_dict(c) for c in data.get("holdout_cases") or []]
        data["run"] = RunSettings(**(data.get("run") or {}))
        data["overrides"] = dict(data.get("overrides") or {})
        data["spec_path"] = str(Path(path).resolve())
        data["resolved_yaml"] = resolved
        spec = cls(**data)
        spec.validate()
        return spec

    def validate(self) -> None:
        from stmr.tune.spaces import SPACES

        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {self.metric!r}")
        if self.space not in SPACES:
            raise ValueError(f"space must be one of {sorted(SPACES)}, got {self.space!r}")
        if not self.tune_cases:
            raise ValueError("tune_cases is empty")
        names = self.case_names("all")
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(f"case names must be unique (tune + holdout): {sorted(dup)}")
        if self.sampler.get("name", "tpe") not in SAMPLERS:
            raise ValueError(f"sampler.name must be one of {SAMPLERS}")
        if self.pruner.get("name", "wilcoxon") not in PRUNERS:
            raise ValueError(f"pruner.name must be one of {PRUNERS}")
        if self.run.oom_policy not in ("auto", "retry", "penalize"):
            raise ValueError("run.oom_policy must be auto | retry | penalize")
        reserved = {"exp_name", "log_path", "device"}
        bad = reserved & (set(self.overrides) | {k for c in self.cases("all") for k in c.overrides})
        if bad:
            raise ValueError(f"{sorted(bad)} are set per run by the tuner; remove them")
        if not (REPO_ROOT / self.base_config).exists() and not Path(self.base_config).exists():
            raise ValueError(f"base_config {self.base_config!r} not found")

    def require_ready(self) -> None:
        if self.todo:
            raise SystemExit(f"{self.spec_path}: not ready to run (todo: {self.todo.strip()}). "
                             "Fill in the placeholders and delete the `todo` key.")


# keys merged entry by entry when a spec `extends` another; every other key is replaced
_MERGED = ("overrides", "run", "sampler", "pruner", "controls")


def _load_yaml(path: Path, _seen: tuple = ()) -> dict:
    """A spec's YAML; `extends: other.yaml` (relative to this file) starts from that spec, so a
    later stage keeps the earlier stage's cases and run settings by construction."""
    if path.resolve() in _seen:
        raise ValueError(f"circular extends: {path}")
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    parent = data.pop("extends", None)
    if parent is None:
        return data
    base = _load_yaml(path.parent / parent, (*_seen, path.resolve()))
    for key in ("todo", "enqueue"):  # never inherited: they describe one particular spec
        base.pop(key, None)
    for key, value in data.items():
        if key in _MERGED and isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base
