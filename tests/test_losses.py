import pytest
import torch

from stmr.config import Config
from stmr.data.data_load import fold_slices, unfold_slices
from stmr.data.fft_utils import FastmriFT, FastmriIFT, MaskedFT
from stmr.data.recon_init import get_functions, get_reg
from stmr.losses.losses import (
    calculate_losses,
    detJ_loss,
    grad_phi_loss,
    im_space_l2_loss,
    log_detJ_loss,
    similarity_loss,
)
from stmr.metrics.calc_metrics import get_relevant_loss_names
from stmr.state import Inputs, ModelOutputs
from stmr.utils.spatial_transformer import get_spatial_transformer
from stmr.utils.spatial_utils import generate_coord_tensor, get_phi_shape


def _config():
    return Config(device="cpu", dataset="cmr_P001", use_nreps=False,
                  time_points=3).finalize()


def _identity_model_outputs(config, h=8, w=8, moving=None):
    coord = generate_coord_tensor((h, w), "cpu")
    abs_phi = coord.expand((config.time_points - 1, -1, -1)).contiguous()
    if moving is None:
        moving = torch.zeros(config.time_points, 2, 1, h, w)
    st = get_spatial_transformer(abs_phi, [-1, h, w, 2], config)
    return coord, ModelOutputs(rel_vel=None, abs_phi=abs_phi, transformer=st)


def test_grad_phi_is_zero_for_identity_deformation():
    config = _config()
    h = w = 8
    coord, model_outputs = _identity_model_outputs(config, h, w)
    shape = [-1, h, w, 2]
    loss, _ = grad_phi_loss(model_outputs, coord, shape)
    assert loss.abs().max() < 1e-4


def _affine_px_model_outputs(A, h, w):
    """Deformation X -> A X in pixel coordinates (about the centre), as normalized abs_phi."""
    coord = generate_coord_tensor((h, w), "cpu").double()
    span = torch.tensor([(h - 1) / 2, (w - 1) / 2], dtype=torch.float64)
    abs_phi = (((coord * span) @ A.T) / span)[None]
    return coord, ModelOutputs(rel_vel=None, abs_phi=abs_phi, transformer=None)


def test_grad_phi_is_isotropic_on_non_square_grid():
    """The same pixel-space stretch / shear costs the same along either axis, and the loss is
    invariant to rotating the deformation, even when H != W."""
    import math

    h, w = 12, 30
    shape = [-1, h, w, 2]
    e = 0.01

    def loss(A):
        coord, mo = _affine_px_model_outputs(torch.tensor(A, dtype=torch.float64), h, w)
        return grad_phi_loss(mo, coord, shape)[0]

    stretch_y, stretch_x = loss([[1 + e, 0], [0, 1]]), loss([[1, 0], [0, 1 + e]])
    shear_y, shear_x = loss([[1, e], [0, 1]]), loss([[1, 0], [e, 1]])
    assert torch.allclose(stretch_y, stretch_x)
    assert torch.allclose(shear_y, shear_x)

    th = math.radians(30)
    R = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]],
                     dtype=torch.float64)
    B = torch.tensor([[1.02, 0.01], [-0.005, 0.99]], dtype=torch.float64)
    assert torch.allclose(loss(B.tolist()), loss((R @ B @ R.T).tolist()))
    # same value at the (replicate-padded) border as in the interior for an affine map
    assert torch.allclose(loss(B.tolist()), loss(B.tolist())[0, h // 2, w // 2].expand(1, h, w))


def test_detJ_is_zero_for_identity_deformation():
    config = _config()
    h = w = 8
    coord, model_outputs = _identity_model_outputs(config, h, w)
    shape = [-1, h, w, 2]
    loss, _ = detJ_loss(model_outputs, coord, shape)
    assert loss.abs().max() < 1e-4


def test_log_detJ_is_zero_for_identity_deformation():
    config = _config()
    h = w = 8
    coord, model_outputs = _identity_model_outputs(config, h, w)
    shape = [-1, h, w, 2]
    loss, _ = log_detJ_loss(model_outputs, coord, shape)
    assert loss.abs().max() < 1e-4


def test_imdiff_is_zero_when_frames_match_and_transform_identity():
    config = _config()
    h = w = 8
    frame = torch.randn(1, 2, 1, h, w)
    moving = frame.repeat(config.time_points, 1, 1, 1, 1)  # identical frames
    _, model_outputs = _identity_model_outputs(config, h, w, moving)
    inputs = Inputs(moving=moving, moving_inr=None, fixed=None, forward=None, phi_shape=[-1, h, w, 2])
    loss, _ = im_space_l2_loss(config, inputs, model_outputs)
    assert loss.abs().max() < 1e-4


def test_data_fidelity_zero_when_recon_matches_measurements():
    config = _config()
    h = w = 8
    moving = torch.randn(config.time_points, 2, 1, h, w)
    forw = FastmriFT()
    fixed = forw(moving)  # measurements consistent with the recon
    inputs = Inputs(moving=moving, moving_inr=None, fixed=fixed, forward=forw, phi_shape=[-1, h, w, 2])
    loss, _ = similarity_loss(config, inputs)
    assert loss.abs().max() < 1e-5


@pytest.mark.parametrize("init_reg_abs", [True, False])
@pytest.mark.parametrize("n_slices", [1, 3])
def test_joint_recon_loss_is_the_init_energy(init_reg_abs, n_slices):
    # sim + recon_reg must be exactly the nmAPG init's energy (per 2D image) averaged over the
    # T * D images, with the gradient the init's analytic path uses.
    config = Config(device="cpu", dataset="cmr_P001", time_points=2,
                    reg_variant="wcrr", recon_scale=3.0, lambda_recon=0.05,
                    init_reg_abs=init_reg_abs, init_grad="analytic").finalize()
    g = torch.Generator().manual_seed(0)
    shape = (2, 2, n_slices, 32, 48)
    x = 0.05 * torch.randn(*shape, generator=g)
    mask = torch.zeros(*shape)
    mask[..., ::3, :] = 1
    kdata = mask * FastmriFT()(0.05 * torch.randn(*shape, generator=g))
    reg = get_reg(config)
    *_, energy_and_grad = get_functions(config, reg, FastmriFT(), FastmriIFT())
    # the init solves each (frame, slice) as a 2D batch item
    energy, grad = energy_and_grad(fold_slices(x), fold_slices(torch.cat([kdata, mask], dim=1)))

    moving = x.clone().requires_grad_(True)
    inputs = Inputs(moving=moving, moving_inr=None, fixed=kdata, forward=MaskedFT(mask),
                    phi_shape=get_phi_shape(shape[2:]))
    losses = {k: v for k, v in get_relevant_loss_names(config).items() if k in ("sim", "recon_reg")}
    loss_sum, _ = calculate_losses(config, inputs, None, None, losses, reg)
    loss_sum.backward()
    assert torch.allclose(loss_sum.detach(), energy.mean(), rtol=1e-6)
    expected_grad = unfold_slices(grad, n_slices) / energy.numel()
    assert (moving.grad - expected_grad).norm() / expected_grad.norm() < 1e-4


def test_use_nreps_without_inr_raises_clear_error():
    config = Config(device="cpu", dataset="cmr_P001", use_nreps=True, time_points=3).finalize()
    h = w = 8
    _, model_outputs = _identity_model_outputs(config, h, w)
    inputs = Inputs(moving=torch.zeros(3, 2, 1, h, w), moving_inr=None, fixed=None, forward=None,
                    phi_shape=[-1, h, w, 2])
    with pytest.raises(ValueError, match="use_nreps"):
        im_space_l2_loss(config, inputs, model_outputs)
