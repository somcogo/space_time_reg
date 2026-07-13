"""VERIFY CLAIM 2: the small motion gain is NOT a bug -- the motion path is correct.

(A) From saved runs: does the LEARNED motion reduce the real frame-to-frame residual, and in
    the right direction? (warp(frame_i) should match frame_i+1 better than frame_i does.)
(B) Synthetic control: feed frame pairs related by a KNOWN deformation through the real
    GroupedSiren -> odeint -> grid_sample stack and check it is recovered.

If (A) reduces the residual and (B) recovers known motion, the implementation is correct and
the small PSNR gain is a property of the data (consecutive frames are nearly identical after
k-space sharing), not a code defect.
Run:  python motion_findings/verify_4_motion_bug_audit.py
"""
import glob
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastmri import complex_abs
from src.models.siren import GroupedSiren
from src.utils.spatial_transformer import GridSampleTransformer
from src.utils.spatial_utils import generate_coord_tensor
from torchdiffeq import odeint_adjoint as odeint

torch.manual_seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

print("="*72)
print("(A) Does the SAVED learned motion reduce the real frame-to-frame residual?")
print("="*72)
runs = sorted(glob.glob('log/cmr/soft_con/sweep_dc/*/res.pt')
              + glob.glob('log/cmr/soft_con/decomp_static/*/res.pt')
              + glob.glob('log/cmr/soft_con/decomp/*/res.pt'))
shown = 0
for run in runs:
    try:
        d = torch.load(run, map_location='cpu', weights_only=False)
        moving, phi = d['moving'], d['phi']
        if d['config'].get('lambda_rl2', 0) == 0 and d['config'].get('lambda_mcdc', 0) == 0:
            continue  # skip runs with no motion-driving term
        T, C, H, W = moving.shape
        mag = complex_abs(moving.movedim(1, -1))
        abs_phi = phi.reshape(T-1, H*W, 2) if phi.dim() == 4 else phi
        if abs_phi.shape[0] != T-1:
            continue
        ST = GridSampleTransformer(abs_phi, moving.shape[1:])
        warped_mag = complex_abs(ST.apply(moving[:-1]).movedim(1, -1))
        mse = lambda a, b: (a-b).pow(2).mean().item()
        base, warp = mse(mag[:-1], mag[1:]), mse(warped_mag, mag[1:])
        tag = os.path.basename(os.path.dirname(run))
        arrow = 'REDUCES (ok)' if warp < base else 'INCREASES (BUG!)'
        print(f"  {tag:30s} frame resid {base:.2e} -> warp {warp:.2e}  {100*(base-warp)/base:+5.1f}%  {arrow}")
        shown += 1
    except Exception:
        continue
if shown == 0:
    print("  (no motion runs found under log/cmr/soft_con/sweep_dc/ -- run run_decomposition.sh first)")

print()
print("="*72)
print("(B) Synthetic recovery of a KNOWN deformation through the real motion path")
print("="*72)
H = W = 64
ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
img = (torch.exp(-((xs)**2+(ys)**2)/0.2) + 0.5*torch.exp(-((xs-0.3)**2+(ys+0.2)**2)/0.02))[None, None].to(dev)
gx, gy = xs.to(dev), ys.to(dev)
coord = generate_coord_tensor((H, W), dev)
t01 = torch.tensor([0., 1.], device=dev)
for px in [1.0, 3.0, 6.0]:
    amp = px/(W/2)
    grid = torch.stack([gx+amp*torch.sin(3*gy), gy+amp*torch.cos(3*gx)], -1)[None]
    target = F.grid_sample(img, grid, align_corners=True)
    net = GroupedSiren(groups=1, layers=[2, 64, 64, 2], last_init_zero=True, omega=30).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    init = (img-target).pow(2).mean().item()
    for _ in range(300):
        opt.zero_grad()
        aphi = odeint(net, coord.unsqueeze(0), t01, method='euler', options={'step_size': 0.1})[1]
        loss = (GridSampleTransformer(aphi, (1, H, W)).apply(img) - target).pow(2).mean()
        loss.backward(); opt.step()
    print(f"  known {px:.0f}px deformation: residual {init:.2e} -> {loss.item():.2e}  "
          f"({100*(init-loss.item())/init:.0f}% reduced, {'RECOVERED' if loss.item()<0.1*init else 'FAILED'})")
