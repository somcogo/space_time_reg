"""Unit checks for the deformation-propagation math in stmr.viz.deform (CPU, no files)."""

import torch
from _phantom import moving_phantom

from stmr.config import Config
from stmr.utils.spatial_utils import generate_coord_tensor
from stmr.viz.deform import build_stepwise_trajectory, build_trajectory, flows_by_time


def _identity_phis(n_times, T, H, W):
    coord = generate_coord_tensor((H, W), "cpu")  # [H*W, 2]
    return coord.expand(n_times, T - 1, -1, -1).contiguous()


def test_identity_flow_leaves_source_unchanged():
    """Chaining identity deformations must return the source frame at every time point."""
    src, _ = moving_phantom(T=1, H=16, W=16)  # [1, 2, 16, 16]
    T = 3
    phis = _identity_phis(2, T, 16, 16)
    times = torch.tensor([0.0, 1.0])

    frames, labels = build_trajectory(src, phis, times, T, img_shape=(2, 16, 16), start=0)

    assert labels == [0.0, 1.0, 2.0]          # source + one image per interval
    assert len(frames) == T
    for f in frames:
        assert torch.allclose(f, src, atol=1e-5)


def test_nonzero_flow_moves_the_image():
    """A non-identity deformation must actually change the propagated image, and chaining
    two intervals must move it further than one."""
    src, _ = moving_phantom(T=1, H=32, W=32, shift_px=0.0)
    T = 3
    coord = generate_coord_tensor((32, 32), "cpu")
    shift = torch.zeros_like(coord)
    shift[:, 1] = 6.0 / (32 - 1) * 2.0  # sample ~6 px to the side each interval
    # phis[time, interval] : identity at t=0, the shifted sampling grid at t=1.
    phis = torch.stack([coord.expand(T - 1, -1, -1),
                        (coord + shift).expand(T - 1, -1, -1)]).contiguous()
    times = torch.tensor([0.0, 1.0])

    frames, _ = build_trajectory(src, phis, times, T, img_shape=(2, 32, 32), start=0)

    d1 = (frames[1] - src).abs().sum().item()
    d2 = (frames[2] - src).abs().sum().item()
    assert d1 > 0                # the flow changed the image
    assert d2 > d1              # chaining two intervals moved it further from the source


def test_stepwise_warps_each_gt_frame_once():
    """stepwise mode: frame 0 is GT_0, and each later frame is the corresponding *real* GT
    frame warped once (not the running image). With identity flows every warped frame equals
    its own GT source."""
    gt, _ = moving_phantom(T=4, H=16, W=16, shift_px=2.0)  # 4 real frames
    T = 4
    phis = _identity_phis(2, T, 16, 16)
    times = torch.tensor([0.0, 1.0])

    frames, labels = build_stepwise_trajectory(gt, phis, times, T, (2, 16, 16), start=0)

    assert labels == [0.0, 1.0, 2.0, 3.0]
    assert len(frames) == T
    # frame 0 is GT_0; frame k (k>=1) is warp(GT_{k-1}, identity) == GT_{k-1}.
    assert torch.allclose(frames[0], gt[0:1], atol=1e-5)
    for k in range(1, T):
        assert torch.allclose(frames[k], gt[k - 1:k], atol=1e-5)


def test_flows_by_time_uses_saved_phi_for_coarse():
    """substeps==1 must reuse the saved best-epoch phi without rebuilding the net."""
    torch.manual_seed(0)
    cfg = Config(time_points=3, device="cpu").finalize()
    d = {"phi": torch.randn(2, 16 * 16, 2) * 0.01}

    times, phis = flows_by_time(d, cfg, 16, 16, substeps=1, device="cpu")

    assert phis.shape == (2, 2, 16 * 16, 2)
    assert torch.allclose(phis[1], d["phi"])
