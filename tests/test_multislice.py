"""The multi-slice data path, [T, 2, D, H, W] with D = 1 for a single slice.

Config parsing of the list fields, the per-(frame, slice) init fold, the slice-aware motion
helpers, and the motion + loss inner loop on one and on several slices. CPU-only, tiny tensors;
the init test uses the git-tracked WCRR weights (as test_init_grad).
"""
from argparse import Namespace

import pytest
import torch
from torch import nn

from stmr.config import Config, build_velocity_kwargs, parse_cli
from stmr.data.data_load import fold_slices, unfold_slices
from stmr.data.fft_utils import FastmriFT, FastmriIFT
from stmr.data.recon_init import init_using_nmAPG
from stmr.losses.losses import calculate_losses
from stmr.metrics.calc_metrics import get_relevant_loss_names
from stmr.metrics.motion_metrics import displacement_px, endpoint_error, fold_stats
from stmr.models.factory import get_func
from stmr.registration import get_model_outputs
from stmr.robustness.jacobian import detJ_from_phi
from stmr.state import Inputs
from stmr.utils.spatial_utils import generate_coord_tensor


def test_list_fields_parse_from_yaml_and_cli(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("slices: [0, 2]\nvoxel_spacing: [8, 1.5, 1.5]\n")
    config = Config.from_yaml(cfg_file)
    assert config.slices == [0, 2]
    assert config.voxel_spacing == [8.0, 1.5, 1.5]
    assert all(isinstance(v, float) for v in config.voxel_spacing)
    assert Config.from_dict({"slices": 3}).slices == [3]

    config = parse_cli(["--set", "slices=1,4", "voxel_spacing=[8, 1.5, 1.5]"])
    assert config.slices == [1, 4]
    assert config.voxel_spacing == [8.0, 1.5, 1.5]


def test_fold_slices_makes_every_frame_slice_pair_a_batch_item():
    x = torch.randn(3, 2, 4, 5, 6)
    folded = fold_slices(x)
    assert folded.shape == (12, 2, 5, 6)
    assert torch.equal(folded[1 * 4 + 2], x[1, :, 2])  # item t * D + d is frame t, slice d
    assert torch.equal(unfold_slices(folded, 4), x)


def test_folded_init_matches_solving_each_slice_alone():
    # The init is a 2D solve per (frame, slice), so solving all slices folded into one batch
    # must give what solving each slice on its own gives.
    cfg = Namespace(device="cpu", reg="learned", reg_variant="wcrr", reg_alpha=1.0, recon_scale=3.0,
                    lambda_init_recon=0.05, lambda_st=1.0, init_loss="l2", init_reg_abs=False,
                    detach_grads=True, init_grad="autograd", recon_epochs=5, tol=0.0, debug=False)
    g = torch.Generator().manual_seed(0)
    T, D, H, W = 2, 3, 16, 24
    x0 = 0.05 * torch.randn(T, 2, D, H, W, generator=g)
    mask = torch.zeros(T, 2, D, H, W)
    mask[..., ::3, :] = 1
    kdata = mask * FastmriFT()(0.05 * torch.randn(T, 2, D, H, W, generator=g))

    def solve(x, k, m):
        recon, _ = init_using_nmAPG(cfg, x, torch.cat([k, m], dim=1), FastmriFT(), FastmriIFT(), None)
        return recon

    folded = unfold_slices(solve(fold_slices(x0), fold_slices(kdata), fold_slices(mask)), D)
    for d in range(D):
        alone = solve(x0[:, :, d], kdata[:, :, d], mask[:, :, d])
        assert (folded[:, :, d] - alone).norm() / alone.norm() < 1e-4


def test_single_slice_motion_lives_on_the_in_plane_grid():
    ident = generate_coord_tensor((1, 8, 12), "cpu")
    assert ident.shape == (8 * 12, 2)
    detJ = detJ_from_phi(ident.unsqueeze(0), 1, 8, 12)
    assert detJ.shape == (1, 8, 12)
    assert torch.allclose(detJ, torch.ones_like(detJ), atol=1e-5)


def test_multi_slice_identity_has_unit_jacobian_and_no_displacement():
    ident = generate_coord_tensor((3, 8, 12), "cpu").unsqueeze(0)
    detJ = detJ_from_phi(ident, 3, 8, 12)
    assert detJ.shape == (1, 3, 8, 12)
    assert torch.allclose(detJ, torch.ones_like(detJ), atol=1e-5)
    assert displacement_px(ident, 3, 8, 12).abs().max() < 1e-6
    assert fold_stats(ident, 3, 8, 12)["fold_frac"] == 0


def test_endpoint_error_takes_a_single_slice_volume():
    gt = torch.rand(3, 2, 1, 16, 16)
    coord = generate_coord_tensor((16, 16), "cpu")
    phi = coord + 0.05 * torch.randn(2, 16 * 16, 2)
    out = endpoint_error(phi, phi, gt)
    assert out["epe_crop"] == 0
    assert out["gt_motion_crop"] > 0


@pytest.mark.parametrize("n_slices", [1, 3])
def test_motion_and_loss_backward_on_a_volume(n_slices):
    """The registration inner loop (motion net + ODE flow + losses + backward) on [T, 2, D, H, W]
    input, as test_smoke does for the 2D case."""
    config = Config(
        device="cpu", dataset="cmr_P001", time_points=3,
        func_name="groupsiren", siren_depth=2, siren_dim=16, solver="euler",
        step_size=0.5, loss="mse", use_nreps=False,
        lambda_st=1.0, lambda_grad_phi=1e-2, lambda_grd=0.0, lambda_lap=0.0,
        lambda_recon=0.0, lambda_rl2=1.0, voxel_spacing=[8.0, 1.5, 1.5],
    ).finalize()
    # As pipeline.run: a single slice moves in-plane only.
    dims = 2 if n_slices == 1 else 3
    config.func_kwargs = build_velocity_kwargs(config, dims)
    func = get_func(config.func_name, config.func_kwargs)

    h = w = 8
    moving = nn.Parameter(torch.randn(config.time_points, 2, n_slices, h, w))
    forw = FastmriFT()
    fixed = forw(moving.detach())
    inputs = Inputs(moving=moving, moving_inr=None, fixed=fixed, forward=forw)

    coord = generate_coord_tensor(moving.shape[2:], "cpu")
    coord.requires_grad = True

    losses = get_relevant_loss_names(config)
    model_outputs = get_model_outputs(config, func, coord, inputs)
    assert model_outputs.abs_phi.shape == (config.time_points - 1, n_slices * h * w if dims == 3 else h * w, dims)
    loss_sum, loss_outputs = calculate_losses(config, inputs, model_outputs, coord, losses, None)

    assert torch.isfinite(loss_sum)
    loss_sum.backward()
    assert moving.grad is not None and torch.isfinite(moving.grad).all()
    assert any(p.grad is not None for p in func.parameters())
    assert set(loss_outputs.losses) >= {"sim", "grad_phi", "imdiff"}
