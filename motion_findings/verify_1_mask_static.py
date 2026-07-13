"""VERIFY CLAIM 0: the CMRxRecon-provided undersampling mask is static across time.

If the provided (subsampled) mask is identical for every frame, then k-t (time-interleaved)
sampling cannot be reconstructed from challenge data alone, and temporal sharing adds no new
k-space information. Run:  python motion_findings/verify_1_mask_static.py [patient=001]
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

patient = sys.argv[1] if len(sys.argv) > 1 else '001'
path = f'data/processed/cmrxrecon/test/training_p{patient}_single_coil_acc_04_cine_sax_norm.pt'
acc = torch.load(path)          # [frames, slices, H, W, 2]
a = acc[:, 0]                   # slice 0, all frames -> [T, H, W, 2]
T, H, W, _ = a.shape

present = (a.abs().sum(-1) != 0)       # measured location = nonzero complex value
row_sampled = present.any(-1)          # [T, H]  (Cartesian phase-encode: whole rows)

print(f"patient p{patient}, acc_04, slice 0: shape [T={T}, H={H}, W={W}]")
print(f"sampled rows per frame: {row_sampled.sum(-1).tolist()}")
identical = all(torch.equal(row_sampled[t], row_sampled[0]) for t in range(T))
print(f"\nMASK IDENTICAL ACROSS ALL {T} FRAMES? -> {identical}")
if identical:
    idx = torch.nonzero(row_sampled[0]).flatten().tolist()
    print(f"  sampled rows: {idx[:10]} ...  (equispaced step {idx[1]-idx[0]} + ACS center block)")
    print("  => CONFIRMED static mask: k-t interleaving unavailable from subsampled-only data.")
else:
    union = torch.zeros(H, dtype=bool)
    for t in range(T): union |= row_sampled[t]
    print(f"  union over frames covers {union.sum().item()}/{H} rows (single frame {row_sampled[0].sum().item()})")
    print("  => mask IS time-varying: k-t sharing available directly from challenge data.")
