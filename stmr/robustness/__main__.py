"""Robustness harness CLI.

  python -m stmr.robustness sweep    --root DIR [--scenes S0,S5] --gpus 0,1,2,3 --epochs 500 [--dry-run]
  python -m stmr.robustness analyze  --root DIR --out FIGDIR
  python -m stmr.robustness synthetic --out FIGDIR [--quick]
  python -m stmr.robustness smoke    --root DIR --gpus 0 --epochs 3
"""

from __future__ import annotations

import argparse
import copy
import os

from .grid import build_matrix
from .sweep import launch


def _gpus(s):
    return tuple(int(x) for x in str(s).split(",") if x != "")


def cmd_sweep(a):
    scenes = tuple(a.scenes.split(","))
    pts = build_matrix(a.root, scenes=scenes)
    print(f"{len(pts)} grid points -> {a.root} on gpus {a.gpus}")
    launch(pts, gpus=_gpus(a.gpus), epochs=a.epochs, dry_run=a.dry_run)


def cmd_smoke(a):
    # one S0 + one S5 determinism point, tiny epochs
    pts = [p for p in build_matrix(a.root) if p.axis == "det" and p.key == "r0"]
    print(f"smoke: {[p.scene for p in pts]} at {a.epochs} epochs")
    launch(pts, gpus=_gpus(a.gpus), epochs=a.epochs)


def cmd_analyze(a):
    from .collect import collect_sweep, disagreement_map, group_by_axis
    from .figures import (plot_curves, plot_detJ, plot_disagreement,
                          plot_perturbation_sensitivity, plot_run_displacement, plot_spread)
    from .tables import write_axis_table

    recs = collect_sweep(a.root)
    print(f"collected {len(recs)} runs")
    figs, tabs = os.path.join(a.out, "cross"), os.path.join(a.out, "tables")
    # per-run figures
    for r in recs:
        base = os.path.join(a.out, "per-run", r.scene, r.axis, r.key)
        plot_detJ(r.phi, r.gt_im, r.H, r.W, os.path.join(base, "detJ.png"))
        plot_run_displacement(r.phi, r.gt_im, r.H, r.W, os.path.join(base, "displacement.png"))
    # cross-run per (scene, axis)
    groups = group_by_axis(recs)
    floors = {scene: [r.best_full_psnr for r in groups.get((scene, "det"), [])]
              for scene in {r.scene for r in recs}}

    def cross(rs, scene, axis):
        tag = f"{scene}_{axis}"
        std_mag, mean = disagreement_map(rs)
        plot_disagreement(std_mag, mean, rs[0].gt_im, os.path.join(figs, f"{tag}_disagreement.png"))
        plot_curves(rs, "cmr evals all/full psnr", os.path.join(figs, f"{tag}_curves.png"),
                    title=f"{tag}: full PSNR")
        plot_spread(rs, lambda r: r.best_full_psnr, "best full PSNR (dB)",
                    os.path.join(figs, f"{tag}_bestpsnr.png"), floor=floors.get(scene))
        plot_spread(rs, lambda r: r.imdiff_ratio, "imdiff warp/base (<1 helps)",
                    os.path.join(figs, f"{tag}_imdiff_ratio.png"))
        plot_spread(rs, lambda r: r.detJ["foldover_frac"], "foldover fraction",
                    os.path.join(figs, f"{tag}_foldover.png"))
        write_axis_table(rs, tabs, scene, axis)

    for (scene, axis), rs in groups.items():
        # The S5 init axis is curated below (ktavg == adj under a static mask, so it is
        # dropped; the nmAPG-refined init is pulled in from a det baseline instead).
        if (scene, axis) == ("S5", "init"):
            continue
        cross(rs, scene, axis)
        if axis == "noise":  # perturbation sensitivity vs the sigma=0 baseline (det/r0)
            base = next((r for r in groups.get((scene, "det"), []) if r.key == "r0"), None)
            if base is not None:
                plot_perturbation_sensitivity(rs, base,
                    os.path.join(figs, f"{scene}_perturbation_sensitivity.png"), scene)

    # Curated S5 init comparison: gt init vs adj init (no nmAPG) vs adj+nmAPG (a det baseline).
    byid = {(r.scene, r.axis, r.key): r for r in recs}

    def _relabel(k, label):
        r = byid.get(k)
        if r is None:
            return None
        r2 = copy.copy(r); r2.key = label; return r2

    init_cmp = [x for x in (
        _relabel(("S5", "init", "gt"), "gt"),
        _relabel(("S5", "init", "adj"), "adj"),
        _relabel(("S5", "det", "r0"), "adj+nmapg"),
    ) if x is not None]
    if len(init_cmp) >= 2:
        cross(init_cmp, "S5", "init")
    print(f"wrote figures to {a.out}")


def cmd_synthetic(a):
    from .figures import plot_epe, plot_synthetic_determinacy
    from .synthetic import analyze_synthetic, run_synthetic_sweep
    seeds = (0, 1, 2) if a.quick else (0, 1, 2, 3, 4)
    sig = (0.0, 0.05) if a.quick else (0.0, 0.02, 0.05, 0.1)
    kw = dict(seeds=seeds, img_sigmas=sig,
              amps=((1.0, 3.0) if a.quick else (1.0, 3.0, 6.0)),
              H=(28 if a.quick else 64), W=(28 if a.quick else 64),
              steps=(70 if a.quick else 300),
              lr=(1e-3 if a.quick else 3e-4),
              step_size=(0.5 if a.quick else 0.1), device="cpu")
    out = os.path.join(a.out, "synthetic")
    rows, img, H, W = run_synthetic_sweep(**kw)
    plot_epe(rows, os.path.join(out, "epe.png"))
    summ = analyze_synthetic(rows, img, H, W, out)          # runs.csv, summary.csv/md, metrics
    plot_synthetic_determinacy(summ, os.path.join(out, "determinacy.png"))
    print(f"wrote synthetic logs to {out} ({len(rows)} runs, {len(summ)} (amp,noise) groups)")


def cmd_netsize(a):
    from .netsize import analyze
    rows = analyze(a.root, a.out)
    print(f"netsize: {len(rows)} sizes analyzed -> {a.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="stmr.robustness", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sweep"); s.set_defaults(func=cmd_sweep)
    s.add_argument("--root", required=True); s.add_argument("--scenes", default="S0,S5")
    s.add_argument("--gpus", default="0,1,2,3"); s.add_argument("--epochs", type=int, default=500)
    s.add_argument("--dry-run", action="store_true")

    sm = sub.add_parser("smoke"); sm.set_defaults(func=cmd_smoke)
    sm.add_argument("--root", required=True); sm.add_argument("--gpus", default="0")
    sm.add_argument("--epochs", type=int, default=3)

    an = sub.add_parser("analyze"); an.set_defaults(func=cmd_analyze)
    an.add_argument("--root", required=True); an.add_argument("--out", required=True)

    sy = sub.add_parser("synthetic"); sy.set_defaults(func=cmd_synthetic)
    sy.add_argument("--out", required=True); sy.add_argument("--quick", action="store_true")

    ns = sub.add_parser("netsize"); ns.set_defaults(func=cmd_netsize)
    ns.add_argument("--root", required=True); ns.add_argument("--out", required=True)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
