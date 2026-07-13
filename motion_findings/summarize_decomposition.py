"""Read the static-mask decomposition runs and print the motion contribution per factor.
Points by default at the runs produced by run_decomposition.sh (log/cmr/soft_con/decomp_static).
Run:  python motion_findings/summarize_decomposition.py [logdir]
"""
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

base = sys.argv[1] if len(sys.argv) > 1 else 'log/cmr/soft_con/decomp_static'

def final(run, tag='cmr evals all/full psnr'):
    d = os.path.join(base, run, 'tensorboard')
    if not os.path.isdir(d): return None
    ea = EventAccumulator(d); ea.Reload()
    try: return ea.Scalars(tag)[-1].value
    except Exception: return None

factors = sorted({n.split('-')[0] for n in os.listdir(base)}, key=lambda s: int(s[1:])) if os.path.isdir(base) else []
print(f"{'factor':>7} | {'no-motion (CRR)':>15} {'+motion':>8} {'Δ motion':>9} | {'crop no-mot':>11} {'crop +mot':>9} {'crop Δ':>7}")
for f in factors:
    nm, mo = final(f'{f}-crronly'), final(f'{f}-motion')
    if nm is None or mo is None: continue
    cnm, cmo = final(f'{f}-crronly', 'cmr evals all/cropped psnr'), final(f'{f}-motion', 'cmr evals all/cropped psnr')
    cd = f'{cmo-cnm:+.2f}' if (cnm and cmo) else '  -'
    cnm_s = f'{cnm:11.2f}' if cnm else f'{"-":>11}'
    cmo_s = f'{cmo:9.2f}' if cmo else f'{"-":>9}'
    print(f"{f:>7} | {nm:15.2f} {mo:8.2f} {mo-nm:+9.2f} | {cnm_s} {cmo_s} {cd:>7}")
print("\nΔ motion = PSNR(with motion) - PSNR(CRR only, no motion) at matched init, STATIC mask.")
