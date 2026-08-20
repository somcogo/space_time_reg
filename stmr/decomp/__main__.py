"""CLI for the two-stage motion-decomposition experiment.

    python -m stmr.decomp run --exp-name NAME --device cuda:N [--set steps1=3000 lam_grad_phi1=2.0 ...]
    python -m stmr.decomp analyze --root log/decomp/NAME [--out DIR]
    python -m stmr.decomp smoke [--device cpu]
"""

from __future__ import annotations

import argparse
from dataclasses import fields

from stmr.config import _coerce_to_type

from .params import DecompParams


def _apply_overrides(p: DecompParams, sets):
    types = {f.name: f.type for f in fields(DecompParams)}
    for item in sets:
        key, _, raw = item.partition("=")
        if key not in types:
            raise SystemExit(f"--set: unknown DecompParams key {key!r}")
        if "list" in str(types[key]):
            val = [int(x) for x in raw.split(",")]        # e.g. phi1_layers=2,32,32,2
        else:
            val = _coerce_to_type(raw, types[key])
        setattr(p, key, val)
    return p


def cmd_run(a):
    p = DecompParams(exp_name=a.exp_name, device=a.device)
    _apply_overrides(p, a.set)
    from .train import run_decomp
    run_decomp(p)


def cmd_analyze(a):
    from .analyze import analyze
    analyze(a.root, a.out)


def cmd_smoke(a):
    p = DecompParams().smoke()
    if a.device:
        p.device = a.device
    from .train import run_decomp
    run_decomp(p)


def cmd_sweep(a):
    from .sweep import aggregate, launch
    gpus = tuple(int(x) for x in a.gpus.split(",") if x != "")
    launch(root=a.root, gpus=gpus, steps=a.steps, dry_run=a.dry_run)
    if not a.dry_run:
        aggregate(root=a.root)


def cmd_sweep_analyze(a):
    from .sweep import aggregate
    aggregate(root=a.root, out=a.out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stmr.decomp", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="train the two-stage decomposition + baselines")
    r.add_argument("--exp-name", default="decomp_p001_s0", dest="exp_name")
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override any DecompParams field (lists as comma-sep, e.g. phi1_layers=2,32,32,2)")
    r.set_defaults(func=cmd_run)

    an = sub.add_parser("analyze", help="regenerate figures + summary from a run's res.pt")
    an.add_argument("--root", required=True); an.add_argument("--out", default=None)
    an.set_defaults(func=cmd_analyze)

    sm = sub.add_parser("smoke", help="tiny CPU run end-to-end")
    sm.add_argument("--device", default=None)
    sm.set_defaults(func=cmd_smoke)

    sw = sub.add_parser("sweep", help="sweep phi1 reg x size (phi2 fixed) across GPUs")
    sw.add_argument("--root", default="sweep_phi1")
    sw.add_argument("--gpus", default="0,1,2,3")
    sw.add_argument("--steps", type=int, default=1000)
    sw.add_argument("--dry-run", action="store_true")
    sw.set_defaults(func=cmd_sweep)

    sa = sub.add_parser("sweep-analyze", help="aggregate a finished sweep into table + figure")
    sa.add_argument("--root", default="sweep_phi1"); sa.add_argument("--out", default=None)
    sa.set_defaults(func=cmd_sweep_analyze)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
