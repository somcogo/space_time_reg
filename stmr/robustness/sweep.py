"""Launch grid points as `python -m stmr.ablation run` subprocesses, round-robin over GPUs.

Each point is a self-contained pipeline run; parallelism is one run per GPU at a time (the
runs are already multi-threaded and GPU-bound). Mirrors motion_findings/run_decomposition.sh
but in Python so it can consume the GridPoint matrix directly.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from .grid import GridPoint


def cli_for(gp: GridPoint, epochs: int, gpu: int, ladder: str = "1") -> list[str]:
    sets = [f"{k}={v}" for k, v in gp.overrides.items()]
    cmd = [sys.executable, "-m", "stmr.ablation", "run",
           "--ladder", ladder, "--stages", gp.scene,
           "--epochs", str(epochs), "--device", f"cuda:{gpu}",
           "--outdir", gp.outdir]
    if sets:
        cmd += ["--set", *sets]
    return cmd


def launch(points, gpus=(0, 1, 2, 3), epochs=500, ladder="1",
           dry_run=False, poll_s=5) -> dict:
    """Run all points, at most len(gpus) concurrently (one per GPU). Returns {outdir: rc}."""
    if dry_run:
        for i, gp in enumerate(points):
            print(" ".join(cli_for(gp, epochs, gpus[i % len(gpus)], ladder)))
        return {}

    pending = list(points)
    running: dict[int, tuple] = {}  # gpu -> (proc, gp, logfile)
    results: dict[str, int] = {}

    def start(gp, gpu):
        os.makedirs(gp.outdir, exist_ok=True)
        log = open(os.path.join(gp.outdir, "run.log"), "w")
        proc = subprocess.Popen(cli_for(gp, epochs, gpu, ladder), stdout=log, stderr=log)
        running[gpu] = (proc, gp, log)
        print(f"[start] {gp.scene}/{gp.axis}/{gp.key} on cuda:{gpu} (pid {proc.pid})")

    free = list(gpus)
    while pending or running:
        while pending and free:
            start(pending.pop(0), free.pop(0))
        time.sleep(poll_s)
        for gpu, (proc, gp, log) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                log.close()
                results[gp.outdir] = rc
                status = "ok" if rc == 0 else f"FAIL rc={rc}"
                print(f"[done ] {gp.scene}/{gp.axis}/{gp.key} ({status})")
                del running[gpu]
                free.append(gpu)
    return results
