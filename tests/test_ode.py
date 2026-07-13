import torch
from torch import nn

from stmr.config import Config
from stmr.registration import get_model_outputs
from stmr.state import Inputs
from stmr.utils.spatial_utils import generate_coord_tensor


class _ConstVelocity(nn.Module):
    """Velocity field that is a spatial constant (needs a param for odeint_adjoint)."""

    def __init__(self, c):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(c), requires_grad=True)

    def forward(self, t, x):
        return self.c.expand_as(x)


def _config():
    return Config(device="cpu", dataset="cmr_P001", time_points=3, solver="euler",
                  step_size=0.1, use_nreps=False).finalize()


def _inputs(h=8, w=8):
    moving = torch.zeros(3, 2, h, w)  # only .shape[1:] is used by the transformer
    return Inputs(moving=moving, moving_inr=None, fixed=None, forward=None)


def test_zero_velocity_flow_is_identity():
    config = _config()
    coord = generate_coord_tensor((8, 8), "cpu")
    coord.requires_grad = True
    func = _ConstVelocity([0.0, 0.0])
    abs_phi = get_model_outputs(config, func, coord, _inputs()).abs_phi
    # flow of a zero field leaves the identity grid unchanged
    expected = coord.expand((config.time_points - 1, -1, -1))
    assert torch.allclose(abs_phi, expected, atol=1e-5)


def test_constant_velocity_translates_grid():
    config = _config()
    coord = generate_coord_tensor((8, 8), "cpu")
    coord.requires_grad = True
    shift = [0.1, -0.2]
    func = _ConstVelocity(shift)
    abs_phi = get_model_outputs(config, func, coord, _inputs()).abs_phi
    expected = coord.expand((config.time_points - 1, -1, -1)) + torch.tensor(shift)
    assert torch.allclose(abs_phi, expected, atol=1e-5)
