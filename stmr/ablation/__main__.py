"""CLI for the staged ablation.

    python -m stmr.ablation run --stages S0-S5 --epochs 500 --device cuda
    python -m stmr.ablation run --stages S0,S3,S5 --full
    python -m stmr.ablation list
"""

from __future__ import annotations

import argparse
import datetime
import os

from .report import write_summary
from .runner import run_ladder
from .stages import STAGE_BY_ID, STAGES, resolve_stage_ids


def _default_outdir() -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("log", "ablation", stamp)


def _parse_overrides(args) -> dict:
    """Collect global Config overrides applied to every stage (incl. the YAML-backed S6)."""
    from stmr.config import Config, _coerce_to_type

    overrides = {}
    if args.time_points is not None:
        overrides["time_points"] = args.time_points
    known = {f.name: f.type for f in __import__("dataclasses").fields(Config)}
    for item in args.set:
        key, _, raw = item.partition("=")
        if key not in known:
            raise SystemExit(f"--set: unknown config key {key!r}")
        overrides[key] = _coerce_to_type(raw, known[key])
    return overrides


def cmd_run(args) -> None:
    stage_ids = resolve_stage_ids(args.stages)
    epochs = 2000 if args.full else args.epochs
    outdir = args.outdir or _default_outdir()
    overrides = _parse_overrides(args)
    os.makedirs(outdir, exist_ok=True)

    print(f"Ablation: stages={stage_ids} epochs={epochs} device={args.device}")
    if overrides:
        print(f"Overrides: {overrides}")
    print(f"Output:   {outdir}\n")

    rows = run_ladder(stage_ids, epochs, args.device, outdir, overrides)
    md = write_summary(rows, STAGE_BY_ID, outdir)

    print("\n" + "=" * 72)
    print(md)
    print(f"Wrote {os.path.join(outdir, 'summary.md')} and summary.csv")


def cmd_list(_args) -> None:
    for s in STAGES:
        print(f"{s.id}: {s.title}")
        print(f"     criterion:   {s.criterion}")
        print(f"     expectation: {s.expectation}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="stmr.ablation", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the staged ablation ladder")
    p_run.add_argument("--stages", default="all",
                       help="range 'S0-S5', comma list 'S0,S3,S5', or 'all' (default)")
    p_run.add_argument("--epochs", type=int, default=500, help="epochs per stage (default 500)")
    p_run.add_argument("--full", action="store_true", help="use 2000 epochs per stage")
    p_run.add_argument("--device", default="cuda")
    p_run.add_argument("--time-points", type=int, default=None, dest="time_points",
                       help="frames per stage (default 6; the data has 12 available)")
    p_run.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                       help="extra global Config overrides applied to every stage, "
                            "e.g. --set start_frame=0 slice_number=0")
    p_run.add_argument("--outdir", default=None, help="default: log/ablation/<timestamp>")
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser("list", help="list the stages and their criteria")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
