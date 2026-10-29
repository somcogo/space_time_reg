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
from stmr.data.fft_utils import FastmriFT, FastmriIFT, MaskedFT
from stmr.data.recon_init import get_reg, init_using_nmAPG
from stmr.losses.losses import calculate_losses, grad_phi_loss
from stmr.metrics.calc_metrics import get_relevant_loss_names
from stmr.metrics.motion_metrics import displacement_px, endpoint_error, fold_stats
from stmr.models.factory import get_func
from stmr.registration import get_model_outputs
from stmr.robustness.jacobian import detJ_from_phi
from stmr.state import Inputs, ModelOutputs
from stmr.utils.spatial_transformer import GridSampleTransformer, get_spatial_transformer
from stmr.utils.spatial_utils import generate_coord_tensor, get_phi_shape


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
                    lambda_recon=0.05, lambda_st=1.0, init_loss="l2", init_reg_abs=False,
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
    assert get_phi_shape((1, 8, 12)) == [-1, 8, 12, 2]
    assert get_phi_shape((3, 8, 12)) == [-1, 3, 8, 12, 3]
    ident = generate_coord_tensor((8, 12), "cpu")
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
        step_size=0.5, use_nreps=False,
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
    inputs = Inputs(moving=moving, moving_inr=None, fixed=fixed, forward=forw,
                    phi_shape=get_phi_shape(moving.shape[2:]))

    coord = generate_coord_tensor(inputs.phi_shape[1:-1], "cpu")
    coord.requires_grad = True

    losses = get_relevant_loss_names(config)
    model_outputs = get_model_outputs(config, func, coord, inputs.phi_shape)
    assert model_outputs.abs_phi.shape == (config.time_points - 1, n_slices * h * w if dims == 3 else h * w, dims)
    loss_sum, loss_outputs = calculate_losses(config, inputs, model_outputs, coord, losses, None)

    assert torch.isfinite(loss_sum)
    loss_sum.backward()
    assert moving.grad is not None and torch.isfinite(moving.grad).all()
    assert any(p.grad is not None for p in func.parameters())
    assert set(loss_outputs.losses) >= {"sim", "grad_phi", "imdiff"}


def test_multi_slice_needs_voxel_spacing():
    with pytest.raises(ValueError, match="voxel_spacing"):
        Config(slices=[0, 1]).finalize()
    with pytest.raises(ValueError, match="voxel_spacing"):
        Config(slices=[0, 1], voxel_spacing=[1.5, 1.5]).finalize()
    Config(slices=[0, 1], voxel_spacing=[8.0, 1.5, 1.5]).finalize()
    Config(slices=[0]).finalize()  # a single slice moves in-plane only


def test_3d_warp_pulls_along_z_y_x():
    # phi's channels are (z, y, x), applied as a pull-back: out(x) = img(phi(x)).
    D, H, W = 4, 6, 8
    shape = get_phi_shape((D, H, W))
    coord = generate_coord_tensor((D, H, W), "cpu")
    z = torch.arange(D, dtype=torch.float32).view(1, 1, D, 1, 1).expand(1, 2, D, H, W)
    phi = coord.clone()
    phi[:, 0] += 2 / (D - 1)  # +1 slice
    out = GridSampleTransformer(phi[None], shape).apply(z.contiguous())
    assert torch.allclose(out[0, 0, :, 2, 3], torch.tensor([1.0, 2.0, 3.0, 3.0]))  # edge repeated
    x = torch.arange(W, dtype=torch.float32).view(1, 1, 1, 1, W).expand(1, 2, D, H, W)
    phi = coord.clone()
    phi[:, 2] += 2 / (W - 1)  # +1 pixel along x
    out = GridSampleTransformer(phi[None], shape).apply(x.contiguous())
    assert torch.allclose(out[0, 0, 1, 2, :4], torch.tensor([1.0, 2.0, 3.0, 4.0]), atol=1e-5)


def test_through_plane_motion_keeps_the_edge_slices_bright():
    # A volume pads with its edge slice, not zeros: zeros would darken the whole first/last slice
    # under any through-plane motion. A single slice still pads with zeros (air) in-plane.
    D, H, W = 3, 8, 8
    coord = generate_coord_tensor((D, H, W), "cpu")
    phi = coord.clone()
    phi[:, 0] += 0.25 * 2 / (D - 1)  # a quarter slice out of the top of the stack
    out = GridSampleTransformer(phi[None], get_phi_shape((D, H, W))).apply(torch.ones(1, 2, D, H, W))
    assert torch.allclose(out, torch.ones_like(out))

    coord = generate_coord_tensor((H, W), "cpu")
    phi = coord.clone()
    phi[:, 1] += 0.5 * 2 / (W - 1)  # half a pixel out of the right edge
    out = GridSampleTransformer(phi[None], get_phi_shape((1, H, W))).apply(torch.ones(1, 2, 1, H, W))
    assert torch.allclose(out[..., -1], torch.full_like(out[..., -1], 0.5))


def test_grad_phi_is_isotropic_in_mm():
    # The same physical shear (du/dy = 0.05 mm/mm) costs the same on every displacement component,
    # with 8 mm slices and 1.4 mm pixels.
    D, H, W = 6, 20, 20
    spacing = [8.0, 1.4, 1.4]
    shape = get_phi_shape((D, H, W))
    coord = generate_coord_tensor((D, H, W), "cpu")
    span_mm = torch.tensor([(n - 1) / 2 * s for n, s in zip((D, H, W), spacing, strict=True)])
    for axis in range(3):
        u_mm = torch.zeros_like(coord)
        u_mm[:, axis] = 0.05 * coord[:, 1] * span_mm[1]
        mo = ModelOutputs(rel_vel=None, abs_phi=(coord + u_mm / span_mm)[None], transformer=None)
        loss, _ = grad_phi_loss(mo, coord, shape, spacing)
        assert torch.allclose(loss[0, 1:-1, 1:-1, 1:-1], torch.tensor(0.05), atol=1e-5)


def test_per_image_losses_dont_grow_with_frames_or_slices():
    # sim and recon_reg are averaged over the (frame, slice) images and the motion terms are
    # per-voxel means, so repeating the same images leaves every term (and their balance) as is.
    config = Config(device="cpu", dataset="cmr_P001", time_points=2, lambda_recon=0.05,
                    reg_variant="wcrr", recon_scale=3.0, lambda_rl2=1.0).finalize()
    reg = get_reg(config)
    g = torch.Generator().manual_seed(0)
    x = 0.05 * torch.randn(2, 2, 1, 16, 24, generator=g)
    mask = torch.zeros_like(x)
    mask[..., ::3, :] = 1
    kdata = mask * FastmriFT()(0.05 * torch.randn(x.shape, generator=g))

    def means(reps_t, reps_d):
        moving, k, m = (v.repeat(reps_t, 1, reps_d, 1, 1) for v in (x, kdata, mask))
        phi_shape = get_phi_shape(moving.shape[2:])
        coord = generate_coord_tensor(phi_shape[1:-1], "cpu")
        abs_phi = coord.expand(moving.shape[0] - 1, -1, -1)
        st = get_spatial_transformer(abs_phi, phi_shape, config)
        inputs = Inputs(moving=moving, moving_inr=None, fixed=k, forward=MaskedFT(m),
                        phi_shape=phi_shape)
        losses = {n: v for n, v in get_relevant_loss_names(config).items()
                  if n in ("sim", "recon_reg", "imdiff")}
        calculate_losses(config, inputs, ModelOutputs(None, abs_phi, st), coord, losses, reg)
        return {n: d["mean"] for n, d in losses.items()}

    ref = means(1, 1)
    for reps in [(3, 1), (1, 3), (2, 2)]:
        for name, value in means(*reps).items():
            assert torch.allclose(value, ref[name], rtol=1e-5), (reps, name)
