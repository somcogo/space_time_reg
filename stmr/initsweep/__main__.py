"""CLI for the init-only experiments.

    python -m stmr.initsweep run --exp-name NAME --device cuda:N [--set lambda_init_recon=0.05 ...]
    python -m stmr.initsweep sweep --root lambda_sweep --gpus 0,1,2,3 [--dry-run]
    python -m stmr.initsweep aggregate --root lambda_sweep

Mirrors stmr/decomp/__main__.py. The sweep defaults (undersampled cmr_P001, factor 4, tp2,
nmAPG init) are applied on top of the base YAML and can be overridden with --set.
"""

from __future__ import annotations

import argparse
from dataclasses import fields

from stmr.config import Config, _coerce_to_type

BASE_YAML = "configs/cmr_soft_con.yaml"
# init-sweep defaults: the undersampled regime, where the prior actually does the work
SWEEP_DEFAULTS = dict(dataset="cmr_P001", factor=4, time_points=2, use_nmapg=True,
                      init="adj", init_skip=False, recon_epochs=150, lambda_st=1.0,
                      log_path="log/initsweep")


def _build_config(a) -> Config:
    config = Config.from_yaml(a.config)
    for k, v in SWEEP_DEFAULTS.items():
        setattr(config, k, v)
    types = {f.name: f.type for f in fields(Config)}
    for item in a.set:
        key, _, raw = item.partition("=")
        if key not in types:
            raise SystemExit(f"--set: unknown config key {key!r}")
        setattr(config, key, _coerce_to_type(raw, types[key]))
    config.exp_name = a.exp_name
    config.device = a.device
    return config.finalize()


def cmd_run(a):
    from .run import run_init
    row = run_init(_build_config(a))
    print("\n".join(f"{k}: {v}" for k, v in row.items()))


def cmd_sweep(a):
    from .sweep import aggregate, launch
    gpus = tuple(int(x) for x in a.gpus.split(",") if x != "")
    lambdas = [float(x) for x in a.lambdas.split(",")] if a.lambdas else None
    scales = [float(x) for x in a.scales.split(",")] if a.scales else None
    launch(root=a.root, gpus=gpus, dry_run=a.dry_run, lambdas=lambdas, scales=scales,
           extra=a.set)
    if not a.dry_run:
        aggregate(root=a.root)


def cmd_aggregate(a):
    from .sweep import aggregate
    aggregate(root=a.root, out=a.out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stmr.initsweep", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="solve one init end-to-end with logging + figures")
    r.add_argument("--exp-name", default="init_run", dest="exp_name")
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--config", default=BASE_YAML)
    r.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("sweep", help="sweep lambda_init_recon (x recon_scale) across GPUs")
    s.add_argument("--root", default="lambda_sweep")
    s.add_argument("--gpus", default="0,1,2,3")
    s.add_argument("--lambdas", default=None, help="comma-separated; default: the built-in grid")
    s.add_argument("--scales", default=None, help="comma-separated recon_scale values (2nd axis)")
    s.add_argument("--config", default=BASE_YAML)
    s.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="extra overrides forwarded to every run")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_sweep)

    g = sub.add_parser("aggregate", help="rebuild the table + figures from finished runs")
    g.add_argument("--root", default="lambda_sweep"); g.add_argument("--out", default=None)
    g.set_defaults(func=cmd_aggregate)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
