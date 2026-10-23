"""CLI for the full-pipeline hyperparameter grid (see sweep.py for the stages).

    python -m stmr.gridsearch probe --root grid_v1 --gpu 0
    python -m stmr.gridsearch launch --stage A --root grid_v1 --gpus 0,1,2,3,5,6,7 [--dry-run]
    python -m stmr.gridsearch launch --stage A2 --root grid_v1 --gpus ...
    python -m stmr.gridsearch launch --stage B --root grid_v1 --gpus ...
    python -m stmr.gridsearch launch --stage C --root grid_v1 --gpus ...
    python -m stmr.gridsearch launch --stage seeds --root grid_v1 --gpus ... [--top 3 --seeds 1,2]
    python -m stmr.gridsearch aggregate --root grid_v1
"""

from __future__ import annotations

import argparse

from . import sweep


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m stmr.gridsearch")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("probe")
    pr.add_argument("--root", required=True)
    pr.add_argument("--gpu", type=int, default=0)
    la = sub.add_parser("launch")
    la.add_argument("--stage", choices=["A", "A2", "B", "C", "seeds"], required=True)
    la.add_argument("--root", required=True)
    la.add_argument("--gpus", default="0,1,2,3")
    la.add_argument("--per-gpu", type=int, default=1)
    la.add_argument("--top", type=int, default=3)
    la.add_argument("--seeds", default="1,2")
    la.add_argument("--dry-run", action="store_true")
    ag = sub.add_parser("aggregate")
    ag.add_argument("--root", required=True)
    a = p.parse_args(argv)

    if a.cmd == "probe":
        sweep.probe(a.root, a.gpu)
    elif a.cmd == "launch":
        if a.stage == "A":
            pts = sweep.grid_a()
        elif a.stage == "A2":
            pts = sweep.grid_a2(a.root)
        elif a.stage == "B":
            pts = sweep.grid_b(a.root)
        elif a.stage == "C":
            pts = sweep.grid_c()
        else:
            pts = sweep.grid_seeds(a.root, a.top, tuple(int(s) for s in a.seeds.split(",")))
        gpus = tuple(int(g) for g in a.gpus.split(",") if g != "")
        sweep.launch(pts, a.root, a.stage, gpus, a.per_gpu, a.dry_run)
        if not a.dry_run:
            sweep.aggregate(a.root)
    else:
        sweep.aggregate(a.root)


if __name__ == "__main__":
    main()
