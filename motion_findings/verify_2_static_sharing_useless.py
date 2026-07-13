"""VERIFY CLAIM 1 (static/challenge setting): with a static mask, motion-free temporal sharing
gives NOTHING, so motion is the ONLY mechanism that can exploit temporal redundancy.

With a static mask every frame is missing the SAME k-space rows, so copying/averaging rows across
frames cannot fill any missing row -> a temporal "sliding window" recon is byte-for-byte the
zero-filled recon. This is why, in the challenge setting, the only way to use the other frames is
motion-compensated reconstruction (Claim 2).

Run:  python motion_findings/verify_2_static_sharing_useless.py [patient=001] [slice=0] [factor=4] [T=6]
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastmri import complex_abs
from src.data.data_utils import generate_standard_mask
from src.data.fft_utils import FastmriIFT
from src.metrics.metric_utils import calc_cmr_eval_metrics

patient = sys.argv[1] if len(sys.argv) > 1 else '001'
sl      = int(sys.argv[2]) if len(sys.argv) > 2 else 0
factor  = int(sys.argv[3]) if len(sys.argv) > 3 else 4
T       = int(sys.argv[4]) if len(sys.argv) > 4 else 6

gt_k = torch.load(f'data/processed/cmrxrecon/test/training_p{patient}_single_coil_full_cine_sax_norm.pt')[:, sl].permute(0, 3, 1, 2)[:T]
_, C, H, W = gt_k.shape
mask = generate_standard_mask(gt_k.shape, factor)     # STATIC: same rows every frame
ift = FastmriIFT()
gt_im = ift(gt_k)
row_measured = mask[:, 0, :, 0]

def report(name, recon_k):
    im_abs = complex_abs(ift(recon_k).movedim(1, -1)).unsqueeze(1)
    gt_abs = complex_abs(gt_im.movedim(1, -1)).unsqueeze(1)
    p, s, n = calc_cmr_eval_metrics(im_abs, gt_abs)
    print(f"{name:36s} psnr={p.mean():6.2f}  ssim={s.mean():.4f}  nmse={n.mean():.4f}")
    return p.mean()

zf = torch.where(mask, gt_k, torch.zeros(1))
p_zf = report('zero-filled', zf)

# "temporal sliding window": fill each unmeasured row from the nearest frame that measured it
sw = zf.clone()
for t in range(T):
    for r in range(H):
        if not row_measured[t, r]:
            srcs = [t2 for t2 in range(T) if row_measured[t2, r]]
            if srcs:
                t_src = min(srcs, key=lambda t2: abs(t2 - t))
                sw[t, :, r, :] = gt_k[t_src, :, r, :]
p_sw = report('temporal sliding window', sw)

print()
if abs(p_sw - p_zf) < 1e-4:
    print("=> IDENTICAL to zero-fill: with a static mask, temporal sharing recovers NOTHING.")
    print("   No frame measures a row that another frame is missing. Motion is the only temporal lever.")
else:
    print(f"=> sliding window differs from zero-fill by {p_sw - p_zf:+.3f} dB (mask is not fully static?).")
print(f"(factor {factor}, static mask, T={T}, patient p{patient} slice {sl})")
