"""Two-stage decomposition mechanism check on a synthetic large+small composite motion (CPU)."""

import torch

from stmr.decomp.train import run_two_stage_core
from stmr.robustness.synthetic import _phantom
from stmr.utils.spatial_transformer import GridSampleTransformer
from stmr.utils.spatial_utils import generate_coord_tensor


def test_two_stage_decomposition_cpu():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    H = W = 24
    coord = generate_coord_tensor((H, W), "cpu")
    src = _phantom(H, W, "cpu")                                  # [1,1,H,W]

    # composite GT motion: a large low-frequency deformation + a small high-frequency wobble
    big = torch.stack([0.15 * torch.cos(2 * coord[:, 1]), 0.15 * torch.sin(2 * coord[:, 0])], -1)
    small = torch.stack([0.02 * torch.cos(9 * coord[:, 1]), 0.02 * torch.sin(9 * coord[:, 0])], -1)
    aphi = (coord + big + small).unsqueeze(0)                    # [1, N, 2]
    tgt = GridSampleTransformer(aphi, (1, H, W)).apply(src)

    out = run_two_stage_core(src, tgt, H, W, layers1=[2, 32, 32, 2], layers2=[2, 64, 64, 2],
                             steps1=150, steps2=150, lr=1e-3, step_size=0.2, device="cpu",
                             lam_grad1=1e-2, lam_logdetJ1=1e-2)

    assert out["r1"] < out["base"]      # stage 1 reduces the frame-to-frame residual
    assert out["r2"] < out["r1"]        # stage 2 reduces it further (residual motion)
    assert out["m1_px"] > out["m2_px"]  # the coarse field carries more motion than the residual
