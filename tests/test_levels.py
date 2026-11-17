"""Multilevel (coarse-to-fine) registration: stmr/utils/multilevel.py + registration.start_level.

A coarse level f crops the central Nc = N // f rows/columns of k-space; coarse pixel j then
samples the band-limited image at native position N//2 + (j - Nc//2) * N / Nc (f*j + o when f
divides N, fractional otherwise), and the velocity net is evaluated there in native (f=1)
coordinates. These tests pin that geometry, the restriction / prolongation pair, the warp and
grad_phi at a coarse level, the per-level loss weights and lr, and that a single level is the
original pipeline.
"""

import math

import logging

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from stmr.config import Config, build_velocity_kwargs
from stmr.data.data_utils import generate_standard_mask
from stmr.data.fft_utils import FastmriFT, FastmriIFT, MaskedFT
from stmr.losses.losses import PER_IMAGE_LOSSES, grad_phi_loss
from stmr.metrics.calc_metrics import get_relevant_loss_names
from stmr.models.factory import get_func
from stmr.registration import registration, start_level, velocity_lr_factor
from stmr.state import EvalInputs, Inputs, ModelOutputs
from stmr.utils.multilevel import build_level, crop_kspace, level_coords, prolong, restrict
from stmr.utils.spatial_transformer import GridSampleTransformer
from stmr.utils.spatial_utils import generate_coord_tensor

# the CMRxRecon heights x readout, and small grids with every parity combination
REAL_GRIDS = [(162, 512), (204, 512), (246, 512)]
SMALL_GRIDS = [(20, 32), (21, 32), (24, 30), (25, 33)]
LOG = logging.getLogger("test_levels")


def coarse_positions(N: int, f: int) -> torch.Tensor:
    """Native (fractional) pixel positions of the Nc = N // f coarse pixels."""
    Nc = N // f
    return N // 2 + (torch.arange(Nc, dtype=torch.float64) - Nc // 2) * (N / Nc)


def bandlimited_at(x: torch.Tensor, pos_h: torch.Tensor, pos_w: torch.Tensor) -> torch.Tensor:
    """The band-limited (trigonometric) interpolant of x [1, 2, 1, H, W] at native positions
    pos_h x pos_w, evaluated from its spectrum in the centred-FFT convention (image index n is
    position n - N//2, k-space index k is frequency k - N//2)."""
    H, W = x.shape[-2:]
    X = FastmriFT()(x)
    X = torch.complex(X[:, 0], X[:, 1])  # [1, 1, H, W]

    def basis(pos, N):
        k = torch.arange(N, dtype=torch.float64) - N // 2
        return torch.exp(2j * math.pi * (pos[:, None] - N // 2) * k[None, :] / N) / math.sqrt(N)

    out = basis(pos_h, H) @ X @ basis(pos_w, W).T
    return torch.stack([out.real, out.imag], 1)


def bandlimited(H: int, W: int, f: int, seed: int = 0) -> torch.Tensor:
    """[1, 2, 1, H, W] float64 image whose spectrum lies inside the block crop_kspace keeps."""
    g = torch.Generator().manual_seed(seed)
    k = torch.zeros(1, 2, 1, H, W, dtype=torch.float64)
    Hc, Wc = H // f, W // f
    h0, w0 = H // 2 - Hc // 2, W // 2 - Wc // 2
    k[..., h0:h0 + Hc, w0:w0 + Wc] = torch.randn(1, 2, 1, Hc, Wc, generator=g, dtype=torch.float64)
    return FastmriIFT()(k)


def smooth_flow(u: torch.Tensor, amp: float) -> torch.Tensor:
    """A smooth displacement (normalised units) at native coordinates u [N, 2] = (y, x)."""
    y, x = u.unbind(-1)
    return amp * torch.stack([torch.sin(2 * x + 0.3), torch.cos(1.5 * y - 0.2)], -1)


# --- 11: restriction = sampling at the coarse positions ---------------------------------------

@pytest.mark.parametrize("grid", REAL_GRIDS + SMALL_GRIDS)
@pytest.mark.parametrize("f", [2, 3, 4, 8])
def test_restrict_samples_a_bandlimited_image_at_the_coarse_positions(grid, f):
    H, W = grid
    if H // f < 2 or W // f < 2:
        pytest.skip("level too coarse for this grid")
    x = bandlimited(H, W, f)
    expected = bandlimited_at(x, coarse_positions(H, f), coarse_positions(W, f))
    torch.testing.assert_close(restrict(x, f), expected, rtol=0, atol=1e-10)


def test_crop_kspace_keeps_the_mask_boolean():
    mask = generate_standard_mask(torch.Size([3, 2, 1, 204, 512]), 4)
    coarse = crop_kspace(mask, 4)
    assert coarse.dtype == torch.bool and coarse.shape[-2:] == (51, 128)
    # the ACS block survives the crop whole
    assert coarse[0, 0, 0, 51 // 2 - 12:51 // 2 + 12, 0].all()


# --- 12: level_coords ------------------------------------------------------------------------

@pytest.mark.parametrize("grid", REAL_GRIDS + SMALL_GRIDS + [(4, 204, 512)])
@pytest.mark.parametrize("f", [2, 3, 4, 8, 16])
def test_level_coords_sit_on_the_fine_grid_and_map_back_to_the_coarse_one(grid, f):
    if any(n // f < 2 for n in grid[-2:]):
        pytest.skip("level too coarse for this grid")
    u, a, b, coarse_grid = level_coords(grid, f, "cpu")
    axes = [torch.linspace(-1, 1, n, dtype=torch.float64) for n in grid[:-2]] + [
        -1 + 2 * coarse_positions(N, f) / (N - 1) for N in grid[-2:]]
    expected = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(grid))
    torch.testing.assert_close(u.double(), expected, rtol=0, atol=3e-7)
    torch.testing.assert_close((u - b) / a, generate_coord_tensor(coarse_grid, "cpu"),
                               rtol=0, atol=3e-7)
    if len(grid) == 3:  # the slice axis is not coarsened
        assert coarse_grid[0] == grid[0] and a[0] == 1 and b[0] == 0


@pytest.mark.parametrize("grid", REAL_GRIDS + [(4, 204, 512)])
def test_level_coords_at_f1_is_exactly_the_standard_grid(grid):
    u, a, b, coarse_grid = level_coords(grid, 1, "cpu")
    assert torch.equal(u, generate_coord_tensor(grid, "cpu"))
    assert list(coarse_grid) == list(grid) and (a == 1).all() and (b == 0).all()


def test_level_coords_rejects_a_single_pixel_axis():
    with pytest.raises(ValueError):
        level_coords((20, 512), 16, "cpu")


# --- 13: prolong is the right inverse of restrict ---------------------------------------------

@pytest.mark.parametrize("grid", REAL_GRIDS + SMALL_GRIDS)
@pytest.mark.parametrize("f", [2, 4])
def test_restrict_undoes_prolong_and_prolong_undoes_restrict_on_the_band(grid, f):
    H, W = grid
    d = torch.randn(1, 2, 1, H // f, W // f, dtype=torch.float64)
    torch.testing.assert_close(restrict(prolong(d, (H, W)), f), d, rtol=0, atol=1e-12)
    x = bandlimited(H, W, f, seed=1)
    torch.testing.assert_close(prolong(restrict(x, f), (H, W)), x, rtol=0, atol=1e-12)


# --- 14: a coarse warp samples the continuous warp --------------------------------------------

def _blobs_at(u: torch.Tensor) -> torch.Tensor:
    """Smooth complex test image (wide Gaussians) at native coordinates u [N, 2] -> [N, 2]."""
    y, x = u.unbind(-1)
    mag = sum(torch.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * 0.3 ** 2))
              for cy, cx in [(-0.3, -0.4), (0.25, 0.1), (-0.1, 0.55)])
    return torch.stack([mag, 0.3 * mag], -1)


@pytest.mark.parametrize("grid", [(64, 96), (66, 98), (246, 512)])  # f=4 divides both / neither
def test_coarse_warp_with_the_level_map_matches_warping_the_exact_coarse_samples(grid):
    """Warp restrict(image) at a coarse level, and warp the analytic image sampled exactly at the
    coarse pixels' native positions with the same phi and bilinear interpolation: equal up to
    restrict's band-limiting (<1%), while reading the coarse grid as spanning [-1, 1] is not."""
    H, W, f = *grid, 4
    img = _blobs_at(generate_coord_tensor(grid, "cpu").double()).T.reshape(1, 2, 1, H, W)
    u, a, b, coarse_grid = level_coords(grid, f, "cpu")
    u, a, b = u.double(), a.double(), b.double()
    exact = _blobs_at(u).T.reshape(1, 2, 1, *coarse_grid)
    phi, shape = (u + smooth_flow(u, 0.15))[None], [-1, *coarse_grid, 2]

    ref = GridSampleTransformer(phi, shape, (a, b)).apply(exact)
    coarse = GridSampleTransformer(phi, shape, (a, b)).apply(restrict(img, f))
    naive = GridSampleTransformer(phi, shape).apply(restrict(img, f))
    err = ((coarse - ref).norm() / ref.norm()).item()
    err_naive = ((naive - ref).norm() / ref.norm()).item()
    assert err < 0.01, err
    assert err_naive > 5 * err, (err, err_naive)


# --- 15: grad_phi does not depend on the level -------------------------------------------------

@pytest.mark.parametrize("grid", [(96, 128), (98, 130)])
@pytest.mark.parametrize("f", [2, 4])
def test_grad_phi_of_a_smooth_flow_is_level_invariant_with_the_native_span(grid, f):
    H, W = grid
    coord = generate_coord_tensor((H, W), "cpu").double()
    fine = ModelOutputs(None, (coord + smooth_flow(coord, 0.05))[None], None)
    gp_fine = grad_phi_loss(fine, coord, [-1, H, W, 2], grid=(H, W))[0].mean()

    u, _, _, coarse_grid = level_coords((H, W), f, "cpu")
    u = u.double()
    coarse = ModelOutputs(None, (u + smooth_flow(u, 0.05))[None], None)
    gp_coarse = grad_phi_loss(coarse, u, [-1, *coarse_grid, 2], grid=(H, W))[0].mean()
    assert gp_coarse.item() == pytest.approx(gp_fine.item(), rel=0.05)


# --- 10: a single level is the original pipeline ----------------------------------------------

def _tiny_problem(levels=None, starts=None, level_recon="reset", epochs=12, **overrides):
    """Synthetic 3-frame complex sequence (a moving blob + a static ring) under the challenge
    mask, the config and inputs registration() takes; CPU, no data files, no CRR weights."""
    torch.manual_seed(0)
    T, H, W = 3, 40, 64
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    frames = []
    for t in range(T):
        blob = torch.exp(-(yy ** 2 + (xx - 0.1 * t) ** 2) / 0.05)
        ring = torch.exp(-((yy ** 2 + xx ** 2).sqrt() - 0.7) ** 2 / 0.005)
        frames.append(torch.stack([blob + ring, 0.1 * blob], 0))
    gt_im = torch.stack(frames, 0).unsqueeze(2)
    kw = {} if levels is None else dict(levels=levels, level_starts=starts, level_recon=level_recon)
    config = Config(device="cpu", dataset="cmr_P001", time_points=T, func_name="groupsiren",
                    siren_depth=2, siren_dim=16, solver="euler", step_size=0.5, adjoint=False,
                    lambda_st=1.0, lambda_grad_phi=1e-2, lambda_grd=0.0, lambda_lap=0.0,
                    lambda_recon=0.0, lambda_rl2=100.0, epochs=epochs, lr=1e-3, recon_lr=1e-3,
                    lr_schedule="cosine", lr_min=1e-4, metric_cadence=4, log_cadence=4,
                    **kw, **overrides).finalize()
    config.func_kwargs = build_velocity_kwargs(config, dims=2)
    func = get_func(config.func_name, config.func_kwargs)
    mask = generate_standard_mask(gt_im.shape, 4)
    fixed = FastmriFT()(gt_im) * mask
    init = FastmriIFT()(fixed)
    inputs = Inputs(moving=nn.Parameter(init.clone()), moving_inr=None, fixed=fixed,
                    forward=MaskedFT(mask), phi_shape=[-1, H, W, 2], native_grid=(H, W))
    eval_inputs = EvalInputs(init_recon=init, gt_im=gt_im, seg_moving=None, seg_fixed=None)
    return config, func, inputs, eval_inputs


def test_level_f1_is_the_fine_inputs_with_the_standard_grid_and_an_exact_identity_warp():
    _, _, inputs, _ = _tiny_problem()
    level_inputs, coords, affine = build_level(inputs, 1)
    assert level_inputs is inputs
    assert torch.equal(coords, generate_coord_tensor((40, 64), "cpu"))
    phi = (coords + smooth_flow(coords, 0.05)).expand(2, -1, -1)
    img = torch.randn(2, 2, 1, 40, 64)
    # the pre-multilevel warp: no affine, channels reversed into grid_sample's (x, y)
    grid = phi.reshape(-1, 40, 64, 2).flip(-1)
    old = F.grid_sample(img.squeeze(2), grid, align_corners=True, padding_mode="zeros").unsqueeze(2)
    assert torch.equal(GridSampleTransformer(phi, inputs.phi_shape, affine).apply(img), old)


def test_a_single_level_keeps_the_lambdas_and_the_whole_run_lr_schedule():
    config, func, inputs, _ = _tiny_problem()
    _, _, _, optimizer, scheduler, losses = start_level(config, inputs, func, 0, LOG)
    assert losses == get_relevant_loss_names(config)
    for step in range(config.epochs):
        assert optimizer.param_groups[0]["lr"] == config.lr * velocity_lr_factor(config, step)
        optimizer.step()
        scheduler.step()


# --- per-level loss weights and lr ------------------------------------------------------------

def test_a_coarse_level_scales_the_mean_losses_by_its_pixel_ratio():
    config, func, inputs, _ = _tiny_problem(levels=[4, 1], starts=[1, 5])
    level_inputs, _, _, _, _, losses = start_level(config, inputs, func, 0, LOG)
    assert level_inputs.phi_shape[1:-1] == [10, 16]
    r = (10 * 16) / (40 * 64)
    full = get_relevant_loss_names(config)
    for name, loss in losses.items():
        expected = full[name]["lambda"] * (1 if name in PER_IMAGE_LOSSES else r)
        assert loss["lambda"] == pytest.approx(expected)
    assert {"imdiff", "grad_phi"} <= set(losses) and "sim" in losses


def test_lr_is_constant_after_level_warmup_and_the_last_level_runs_its_own_cosine():
    config = Config(lr=1e-4, lr_min=1e-6, lr_schedule="cosine", epochs=30, level_warmup=4,
                    levels=[4, 1], level_starts=[1, 11])
    coarse = [velocity_lr_factor(config, s, 10, final=False) for s in range(10)]
    assert coarse == [0.25, 0.5, 0.75] + [1.0] * 7
    final = [config.lr * velocity_lr_factor(config, s, 20, final=True) for s in range(20)]
    assert final[0] == pytest.approx(1e-4) and final[-1] == pytest.approx(1e-6)


@pytest.mark.parametrize("bad", [
    dict(levels=[4, 2], level_starts=[1, 5]),                       # last level not 1
    dict(levels=[4, 1], level_starts=[1]),                          # lengths differ
    dict(levels=[4, 1], level_starts=[2, 5]),                       # does not start at 1
    dict(levels=[4, 1], level_starts=[1, 1]),                       # not increasing
    dict(levels=[4, 1], level_starts=[1, 11]),                      # past the last epoch
    dict(levels=[4, 1], level_starts=[1, 5], lambda_grd=1.0),       # grid-index loss
    dict(levels=[1, 1], level_starts=[1, 5], motion_warmup=3),      # unsupported with levels
    dict(levels=[4, 1], level_starts=[1, 5], fixed_motion="identity"),
    dict(levels=[4, 1], level_starts=[1, 5], level_recon="keep"),
])
def test_bad_level_configs_are_rejected(bad):
    with pytest.raises(ValueError):
        Config(epochs=10, **{"lambda_grd": 0.0, "lambda_lap": 0.0, **bad}).finalize()


# --- end to end ----------------------------------------------------------------------------------

@pytest.mark.parametrize("level_recon", ["reset", "correct"])
def test_multilevel_registration_runs_every_level_and_returns_a_full_resolution_result(
        level_recon, tmp_path, caplog):
    config, func, inputs, eval_inputs = _tiny_problem(
        levels=[4, 2, 1], starts=[1, 4, 7], level_recon=level_recon, log_path=str(tmp_path))
    with caplog.at_level(logging.INFO, logger=LOG.name):
        res = registration(config, SummaryWriter(str(tmp_path)), LOG, inputs, eval_inputs, func)
    assert res.best_moving.shape == inputs.moving.shape
    assert res.model_outputs.abs_phi.shape == (2, 40 * 64, 2)
    assert res.epoch >= 7  # the best-loss snapshot only counts the full-resolution level
    assert sum("Level end" in r.message for r in caplog.records) == 2
    assert torch.isfinite(res.best_moving).all()
