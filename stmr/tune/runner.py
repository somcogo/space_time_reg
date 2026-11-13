"""Run one (overrides, case) on one GPU and score it.

A run is a ``python -m stmr.cli --config <base> --set k=v ...`` subprocess (isolates CUDA memory,
default-dtype changes and crashes from the long-lived worker), writing into ``run_dir``. Its
score is read back from ``res.pt`` and cached in ``run_dir/score.json``, so a finished run is
never repeated (re-evaluation and resumed workers reuse it).

Score of a case = metric(best-loss recon) - metric(zero-filled recon), both against the GT.
The zero-filled recon depends only on the data and the mask, so the baseline is the same for
every trial: comparisons between trials stay paired, and patients of different difficulty
(P001 hard, P004 easy) land on a comparable scale. The best-loss recon is what res.pt holds
(selected without the GT), not the best-PSNR epoch.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from stmr.tune.spec import REPO_ROOT, Case, StudySpec

HEAVY_FILES = ("res.pt", "np_imgs.npy")
OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError", "CUBLAS_STATUS_ALLOC_FAILED")
BLOWUP = 3.0  # total loss later exceeding BLOWUP x its minimum = the motion diverged


class OOMError(RuntimeError):
    """The case kept running out of GPU memory: a resource problem, not a property of the config."""


@dataclass
class CaseResult:
    case: str
    status: str  # 'ok' | 'crash' | 'timeout' | 'nan' | 'oom'
    score: float
    metrics: dict[str, Any] = field(default_factory=dict)
    run_dir: str = ""
    runtime_s: float = 0.0
    error: str = ""

    def summary(self) -> dict[str, Any]:
        """The compact form stored in the trial's user attrs."""
        keep = ("blowup", "best_epoch", "peak_mem_gb")
        m = {k: v for k, v in self.metrics.items() if "/" in k or k in keep}
        return {"status": self.status, "score": self.score, "runtime_s": round(self.runtime_s),
                **m, **({"error": self.error[-300:]} if self.error else {})}


# --- command line -----------------------------------------------------------------------------

def format_value(v: Any) -> str:
    """A value as stmr.config.parse_cli's --set expects it."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "none"
    if isinstance(v, (list, tuple)):
        return ",".join(format_value(x) for x in v)
    if isinstance(v, float):
        return repr(v)
    return str(v)


def cli_args(base_config: str, overrides: dict[str, Any]) -> list[str]:
    return ["--config", base_config, "--set",
            *(f"{k}={format_value(v)}" for k, v in overrides.items())]


def merge_overrides(spec: StudySpec, params_overrides: dict[str, Any], case: Case,
                    run_dir: str, seed: int | None = None) -> dict[str, Any]:
    """base_config <- spec.overrides <- search space <- case <- bookkeeping (later wins)."""
    clash = set(params_overrides) & set(case.overrides)
    if clash:
        raise ValueError(f"search space and case {case.name} both set {sorted(clash)}")
    out = {**spec.overrides, **params_overrides, **case.overrides}
    if seed is not None:
        out["seed"] = seed
    out["log_path"] = os.path.dirname(run_dir)
    out["exp_name"] = os.path.basename(run_dir)
    out["device"] = "cuda"
    return out


def validate(spec: StudySpec, overrides: dict[str, Any]):
    """Parse the exact argv the subprocess will get, so bad keys/values fail before a GPU starts."""
    from stmr.config import parse_cli

    return parse_cli(cli_args(spec.base_config, overrides))


# --- running ----------------------------------------------------------------------------------

def run_case(spec: StudySpec, params_overrides: dict[str, Any], case: Case, run_dir: str,
             gpu: int, seed: int | None = None, log=print) -> CaseResult:
    """Run and score one case. Raises OOMError after spec.run.oom_retries OOMs in a row."""
    score_path = os.path.join(run_dir, "score.json")
    if os.path.exists(score_path):
        with open(score_path) as fh:
            return CaseResult(**json.load(fh))

    overrides = merge_overrides(spec, params_overrides, case, run_dir, seed)
    validate(spec, overrides)
    cmd = [sys.executable, "-m", "stmr.cli", *cli_args(spec.base_config, overrides)]
    # PCI_BUS_ID: CUDA's device index = nvidia-smi's (the OOM check and --gpus use the latter)
    env = {**os.environ, **spec.run.env, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "CUDA_VISIBLE_DEVICES": str(gpu)}

    for attempt in range(spec.run.oom_retries + 1):
        if os.path.isdir(run_dir):
            shutil.rmtree(run_dir)  # a previous attempt's partial output
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "cmd.txt"), "w") as fh:
            fh.write(" ".join(cmd) + "\n")
        log(f"[gpu {gpu}] {case.name} -> {run_dir}" + (f" (retry {attempt})" if attempt else ""))
        status, runtime, tail = _run_subprocess(cmd, env, run_dir, spec.run.timeout_h * 3600)
        if status == "ok":
            res_path = os.path.join(run_dir, "res.pt")
            if not os.path.exists(res_path):
                result = CaseResult(case.name, "crash", spec.run.crash_score, run_dir=run_dir,
                                    runtime_s=runtime, error="exit 0 but no res.pt\n" + tail)
                break
            metrics = score_res(res_path)
            score = metrics[f"{spec.metric}/final"] - metrics[f"{spec.metric}/zf"]
            if math.isfinite(score):
                result = CaseResult(case.name, "ok", score, metrics, run_dir, runtime)
            else:
                result = CaseResult(case.name, "nan", spec.run.crash_score, metrics, run_dir, runtime)
            break
        if status == "crash" and any(m in tail for m in OOM_MARKERS):
            if _oom_is_the_config(spec.run.oom_policy, gpu):
                log(f"[gpu {gpu}] {case.name}: CUDA OOM with the GPU otherwise free -> crash_score")
                result = CaseResult(case.name, "oom", spec.run.crash_score, run_dir=run_dir,
                                    runtime_s=runtime, error=tail)
                break
            if attempt < spec.run.oom_retries:
                log(f"[gpu {gpu}] {case.name}: CUDA OOM, retrying in {spec.run.oom_wait_min} min")
                time.sleep(spec.run.oom_wait_min * 60)
                continue
            raise OOMError(f"{case.name}: CUDA OOM {attempt + 1}x on GPU {gpu}")
        result = CaseResult(case.name, status, spec.run.crash_score, run_dir=run_dir,
                            runtime_s=runtime, error=tail)
        break

    if not spec.run.keep_heavy_files:
        for f in HEAVY_FILES:
            p = os.path.join(run_dir, f)
            if os.path.exists(p):
                os.remove(p)
    tmp = score_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(asdict(result), fh, indent=1)
    os.replace(tmp, score_path)
    return result


def _run_subprocess(cmd, env, run_dir, timeout_s) -> tuple[str, float, str]:
    """Run cmd into run_dir/run.log. Returns (ok|crash|timeout, seconds, last log lines).

    The child gets its own process group, which is killed if the worker is interrupted
    (SIGTERM is turned into KeyboardInterrupt by the worker), so no run outlives its worker.
    """
    log_path = os.path.join(run_dir, "run.log")
    t0 = time.time()
    with open(log_path, "w") as out:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout_s)
            status = "ok" if rc == 0 else "crash"
        except subprocess.TimeoutExpired:
            _kill(proc)
            status = "timeout"
        except BaseException:
            _kill(proc)
            raise
    return status, time.time() - t0, _tail(log_path)


def gpu_memory_used_mib(gpu: int) -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             check=True, timeout=30).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _oom_is_the_config(policy: str, gpu: int, free_below_mib: int = 1000) -> bool:
    """Whether an OOM is the config's own doing (score it) rather than a shared GPU (retry).
    'auto': the run has exited, so whatever memory is still in use belongs to someone else."""
    if policy in ("penalize", "retry"):
        return policy == "penalize"
    used = gpu_memory_used_mib(gpu)
    return used is not None and used < free_below_mib


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def _tail(path: str, n_bytes: int = 4000) -> str:
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        fh.seek(max(0, fh.tell() - n_bytes))
        return fh.read().decode(errors="replace")


# --- scoring ----------------------------------------------------------------------------------

def image_metrics(x, gt) -> dict[str, float]:
    """The pipeline's own CMR metrics (cmrx_crop box, gt.max() data range) + the official SSIM."""
    from stmr.metrics.metric_utils import add_cmr_eval_metrics, cmrx_official_ssim

    m = add_cmr_eval_metrics(x, gt, {})
    official = m.get("cmr evals official/ssim")
    return {
        "crop_psnr": float(m["cmr evals all/cropped psnr"]),
        "full_psnr": float(m["cmr evals all/full psnr"]),
        "crop_ssim": float(m["cmr evals all/cropped ssim"]),
        "official_ssim": float(official) if official is not None else cmrx_official_ssim(x, gt),
    }


def score_res(res_path: str) -> dict[str, Any]:
    """Metrics of the final (best-loss), init and zero-filled recons in a res.pt, plus run health."""
    import numpy as np
    import torch

    from stmr.data.fft_utils import FastmriIFT

    d = torch.load(res_path, map_location="cpu", weights_only=False)
    gt = d["gt_im"].float()
    recons = {
        "final": d["moving"].float(),
        "init": d["recon_init"].float(),
        "zf": FastmriIFT()(d["fixed"].float()),  # fixed: measured k-space, zero-filled
    }
    out: dict[str, Any] = {}
    for tag, x in recons.items():
        for k, v in image_metrics(x, gt).items():
            out[f"{k}/{tag}"] = v
    curve = [m.get("losses/total_loss", float("nan")) for m in d.get("metric_curve", [])]
    if curve and np.isfinite(curve).any():
        c = np.asarray(curve, dtype=float)
        i_min = int(np.nanargmin(c))
        out["blowup"] = bool(c[i_min] > 0 and np.nanmax(c[i_min:]) > BLOWUP * c[i_min])
    out["best_epoch"] = int(d.get("epoch", -1))
    out["peak_mem_gb"] = d.get("peak_mem_gb")
    return out
