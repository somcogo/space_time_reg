import torch
from torch import nn

from stmr.config import Config, build_velocity_kwargs
from stmr.data.fft_utils import FastmriFT
from stmr.losses.losses import calculate_losses
from stmr.metrics.calc_metrics import get_relevant_loss_names
from stmr.models.factory import get_func
from stmr.registration import get_model_outputs
from stmr.utils.spatial_utils import generate_coord_tensor


def test_end_to_end_motion_and_loss_backward_on_cpu():
    """Motion network + ODE flow + losses + backward run and produce gradients.

    Uses tiny synthetic k-space (no data files, no GPU) and lambda_recon=0 so the CRR
    weights are not needed. Guards the wiring of the whole registration inner loop.
    """
    config = Config(
        device="cpu", dataset="cmr_P001", time_points=3,
        func_name="groupsiren", siren_depth=2, siren_dim=16, solver="euler",
        step_size=0.5, loss="mse", use_nreps=False,
        lambda_st=1.0, lambda_negJ=1e-2, lambda_grd=0.0, lambda_lap=0.0,
        lambda_pgr=0.0, lambda_hel=0.0, lambda_recon=0.0, lambda_rl2=1.0,
    ).finalize()

    h = w = 8
    config.func_kwargs = build_velocity_kwargs(config, dims=2)
    func = get_func(config.func_name, config.func_kwargs)

    moving = nn.Parameter(torch.randn(config.time_points, 2, h, w))
    forw = FastmriFT()
    fixed = forw(moving.detach())
    inputs = [moving, None, fixed, forw]

    coord = generate_coord_tensor((h, w), "cpu")
    coord.requires_grad = True

    losses = get_relevant_loss_names(config)
    model_outputs = get_model_outputs(config, func, coord, inputs)
    loss_sum, loss_outputs = calculate_losses(config, inputs, model_outputs, coord, losses, None)

    assert torch.isfinite(loss_sum)
    loss_sum.backward()
    assert moving.grad is not None and torch.isfinite(moving.grad).all()
    assert any(p.grad is not None for p in func.parameters())
    # the expected active terms are present
    assert set(loss_outputs[0]) >= {"sim", "negJ", "imdiff"}
