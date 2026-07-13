"""VERIFY CLAIM 3 (supporting): the data phase is temporally incoherent, so complex-valued
temporal losses fight the data. This justifies (a) magnitude-based temporal terms and
(b) why motion-compensated data-consistency on complex values does not help.

Reports the relative L2 residual of predicting frame t+1 from frame t under several models.
Run:  python motion_findings/verify_3_phase.py [patient=001] [slice=0] [T=6]
"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.fft_utils import FastmriIFT

patient = sys.argv[1] if len(sys.argv) > 1 else '001'
sl      = int(sys.argv[2]) if len(sys.argv) > 2 else 0
T       = int(sys.argv[3]) if len(sys.argv) > 3 else 6

gt_k = torch.load(f'data/processed/cmrxrecon/test/training_p{patient}_single_coil_full_cine_sax_norm.pt')[:, sl].permute(0, 3, 1, 2)[:T]
im2 = FastmriIFT()(gt_k)
I = torch.view_as_complex(im2.movedim(1, -1).contiguous())   # [T,H,W]
m = I.abs()
rel = lambda pred, target: ((pred - target).abs().square().sum() / target.abs().square().sum()).item()

print(f"patient p{patient} slice {sl}, predicting frame t+1 from frame t (relative L2 residual, lower=better):")
print(f"  copy complex  I_t                     : {rel(I[:-1], I[1:]):.4f}")
print(f"  copy magnitude only                   : {rel(m[:-1]+0j, m[1:]+0j):.4f}")
print(f"  |I_t| with target frame's own phase   : {rel(m[:-1]*torch.exp(1j*torch.angle(I[1:])), I[1:]):.4f}")
print("\n=> If 'copy complex' >> 'copy magnitude', the phase varies fast in time: complex")
print("   temporal coupling is counterproductive; only magnitude should cross frames.")
