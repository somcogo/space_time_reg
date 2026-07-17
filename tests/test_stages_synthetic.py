"""Synthetic, CPU-only mechanism checks mirroring the real-data ablation ladder.

These do NOT call the full pipeline (which needs data files / weights). They build the
Inputs + velocity net directly and drive a few optimisation steps on a tiny translating
phantom, checking the *mechanism* each stage relies on:

  S0       registration recovers a known translation (imdiff drops).
  S1       fully-sampled data consistency anchors the recon at GT under joint optimisation.
  S1 vs S2 image-space and Fourier-space sim losses are identical (FFT is unitary).
  S4       under an undersampling mask, hard DC preserves measured k-space and bounds drift.

The prior stages (S3 WCRR, S5 nmAPG init, S6 full pipeline) depend on pretrained weights /
real data and are covered only by the real-data runner (python -m stmr.ablation run). All
tests here use lambda_recon=0 (no weights), tiny tensors and <=40 CPU steps -- the whole file
runs in well under a minute.
"""

import pytest
import torch
from fastmri import complex_abs
from torch import nn

from stmr.ablation.stages import BASE, STAGES
from stmr.config import Config, build_velocity_kwargs
from stmr.data.data_utils import generate_standard_mask
from stmr.data.fft_utils import FastmriFT, MaskedFT, apply_hard_data_consistency
from stmr.losses.losses import calculate_losses, similarity_loss
from stmr.metrics.calc_metrics import get_relevant_loss_names
from stmr.models.factory import get_func
from stmr.registration import get_model_outputs
from stmr.state import Inputs
from stmr.utils.spatial_utils import generate_coord_tensor

from _phantom import moving_phantom


def _cpu_config(**over):
    base = dict(
        device="cpu", dataset="cmr_P001", func_name="groupsiren",
        siren_depth=2, siren_dim=16, solver="euler", step_size=0.5,
        loss="mse", use_nreps=False,
        lambda_negJ=1e-2, lambda_grd=0.0, lambda_lap=0.0, lambda_pgr=0.0,
        lambda_hel=0.0, lambda_recon=0.0, lambda_mcdc=0.0,
    )
    base.update(over)
    return Config(**base).finalize()


def _psnr(pred, gt):
    p = complex_abs(pred.movedim(1, -1))
    g = complex_abs(gt.movedim(1, -1))
    mse = (p - g).pow(2).mean()
    return (10 * torch.log10(g.max() ** 2 / mse)).item()


def _build(config, moving, forward):
    config.func_kwargs = build_velocity_kwargs(config, dims=2)
    func = get_func(config.func_name, config.func_kwargs)
    fixed = forward(moving.detach())
    inputs = Inputs(moving=moving, moving_inr=None, fixed=fixed, forward=forward)
    return func, inputs


def _optimize(config, func, inputs, steps, learn_recon):
    opt = torch.optim.Adam(func.parameters(), lr=1e-3)
    if learn_recon:
        opt.add_param_group({"params": inputs.moving, "lr": 1e-3})
    inputs.moving.requires_grad_(learn_recon)
    losses = get_relevant_loss_names(config)
    for _ in range(steps):
        coord = generate_coord_tensor(inputs.moving.shape[2:], "cpu")
        coord.requires_grad = True
        opt.zero_grad()
        model_outputs = get_model_outputs(config, func, coord, inputs)
        loss_sum, _ = calculate_losses(config, inputs, model_outputs, coord, losses, None)
        loss_sum.backward()
        opt.step()
        if config.hard_dc and hasattr(inputs.forward, "mask"):
            with torch.no_grad():
                inputs.moving.copy_(apply_hard_data_consistency(
                    inputs.moving, inputs.fixed, inputs.forward.mask))
    return func


def _imdiff_residual(moving, config, func):
    """Base (identity) vs warped consecutive-frame magnitude MSE for the current motion."""
    coord = generate_coord_tensor(moving.shape[2:], "cpu")
    inputs = Inputs(moving=moving, moving_inr=None, fixed=None, forward=None)
    with torch.no_grad():
        mo = get_model_outputs(config, func, coord, inputs)
        mag = complex_abs(moving.movedim(1, -1))
        warped = complex_abs(mo.transformer.apply(moving[:-1]).movedim(1, -1))
    base = (mag[:-1] - mag[1:]).pow(2).mean().item()
    warp = (warped - mag[1:]).pow(2).mean().item()
    return base, warp


def test_S0_registration_recovers_shift():
    """Frozen recon = GT; only imdiff+negJ drive the motion net. It should reduce the
    frame-to-frame residual well below identity (the S0 pass criterion)."""
    torch.manual_seed(0)
    config = _cpu_config(time_points=3, lambda_st=0.0, lambda_rl2=1.0)
    moving_gt, _shift = moving_phantom(T=3, H=32, W=32, shift_px=2.0)
    moving = nn.Parameter(moving_gt.clone())

    func, inputs = _build(config, moving, FastmriFT())
    base0, warp0 = _imdiff_residual(moving.detach(), config, func)
    func = _optimize(config, func, inputs, steps=40, learn_recon=False)
    base1, warp1 = _imdiff_residual(moving.detach(), config, func)

    # the motion net learned a real deformation ...
    assert warp1 < 0.5 * base1
    # ... and improved over the untrained (near-identity) starting point.
    assert warp1 < warp0
    # recon stayed exactly at GT (frozen).
    assert torch.equal(moving.detach(), moving_gt)


def _drift_after_optim(lambda_rl2, steps=40):
    """Start the recon at GT, optimize jointly under fully-sampled DC, return PSNR vs GT."""
    torch.manual_seed(0)
    config = _cpu_config(time_points=3, lambda_st=1.0, lambda_rl2=lambda_rl2, learn_recon=True)
    moving_gt, _ = moving_phantom(T=3, H=32, W=32, shift_px=2.0)
    moving = nn.Parameter(moving_gt.clone())
    func, inputs = _build(config, moving, FastmriFT())
    _optimize(config, func, inputs, steps=steps, learn_recon=True)
    return _psnr(moving.detach(), moving_gt)


def test_S1_fullysampled_dc_anchors_recon_at_gt():
    """The S1 mechanism, in two parts.

    (a) With fully-sampled DC and NO temporal coupling, GT is the exact global minimiser
        of the data term, so gradient descent started at GT never leaves it (PSNR stays
        ~infinite). This is the anchor.
    (b) Adding the temporal coupling is what drags the recon off GT -- exactly the failure
        the real-data S1 stage is built to detect. So the coupled run drifts measurably."""
    psnr_dc_only = _drift_after_optim(lambda_rl2=0.0)
    psnr_coupled = _drift_after_optim(lambda_rl2=1.0)

    # (a) DC alone holds the recon exactly at GT.
    assert psnr_dc_only > 80.0
    # (b) the coupling pulls it away.
    assert psnr_coupled < psnr_dc_only - 10.0


def test_fft_sim_equals_image_sim():
    """FFT works as intended: on fully-sampled data the Fourier-space similarity loss equals
    the image-space one -- same value AND same gradient (Parseval, the FFT is unitary). This
    is the exact mechanism the image-space S1 / Fourier-space S2 stages compare, verified
    here in one step instead of a full training run."""
    torch.manual_seed(0)
    moving_gt, _ = moving_phantom(T=3, H=16, W=16, shift_px=1.0)
    forw = FastmriFT()
    fixed = forw(moving_gt)
    # A single fixed perturbation, reused for both domains, so the only difference between
    # the two runs is where the sim loss is evaluated (image vs Fourier space).
    perturbed = moving_gt + 0.05 * torch.randn_like(moving_gt)

    def sim_loss(sim_domain):
        cfg = _cpu_config(time_points=3, lambda_st=1.0, lambda_rl2=0.0, sim_domain=sim_domain)
        moving = nn.Parameter(perturbed.clone())
        inputs = Inputs(moving=moving, moving_inr=None, fixed=fixed, forward=forw)
        loss = similarity_loss(cfg, inputs, None)[0].mean()
        loss.backward()
        return loss.item(), moving.grad.clone()

    l_img, g_img = sim_loss("image")
    l_fou, g_fou = sim_loss("fourier")
    assert abs(l_img - l_fou) < 1e-6
    assert torch.allclose(g_img, g_fou, atol=1e-6)


@pytest.mark.parametrize("hard_dc", [True, False])
def test_S4_masked_gt_init_bounded_drift(hard_dc):
    """Under an undersampling mask starting at GT: with hard DC the measured k-space rows
    are preserved exactly and PSNR drift is bounded; without it the recon drifts more."""
    torch.manual_seed(0)
    H = W = 64
    config = _cpu_config(time_points=3, lambda_st=1.0, lambda_rl2=1.0,
                         learn_recon=True, hard_dc=hard_dc)
    moving_gt, _ = moving_phantom(T=3, H=H, W=W, shift_px=2.0)
    moving = nn.Parameter(moving_gt.clone())

    mask = generate_standard_mask(moving_gt.shape, factor=2)
    forward = MaskedFT(mask)
    func, inputs = _build(config, moving, forward)
    measured_gt = forward(moving_gt)

    _optimize(config, func, inputs, steps=30, learn_recon=True)

    if hard_dc:
        # measured entries must be preserved to within numerical tolerance.
        measured_now = forward(moving.detach())
        assert torch.allclose(measured_now, measured_gt, atol=1e-4)
    # some structure retained either way; drift is bounded.
    assert _psnr(moving.detach(), moving_gt) > 20.0


def test_ladder_wiring():
    """Every StageSpec (except the YAML-backed S5) builds a valid Config with no unknown
    keys and finalizes cleanly."""
    for spec in STAGES:
        if spec.yaml is not None:
            continue
        merged = {**BASE, **spec.overrides}
        cfg = Config.from_dict(merged).finalize()
        assert cfg.dataset.startswith("cmr")
