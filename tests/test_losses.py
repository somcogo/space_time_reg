import pytest
import torch

from stmr.config import Config
from stmr.data.fft_utils import FastmriFT
from stmr.losses.losses import im_space_l2_loss, negJ_loss, similarity_loss
from stmr.utils.spatial_transformer import get_spatial_transformer
from stmr.utils.spatial_utils import generate_coord_tensor


def _config():
    return Config(device="cpu", dataset="cmr_P001", loss="mse", use_nreps=False,
                  time_points=3).finalize()


def _identity_model_outputs(config, h=8, w=8, moving=None):
    coord = generate_coord_tensor((h, w), "cpu")
    abs_phi = coord.expand((config.time_points - 1, -1, -1)).contiguous()
    if moving is None:
        moving = torch.zeros(config.time_points, 2, h, w)
    st = get_spatial_transformer(abs_phi, moving.shape[1:], config)
    return coord, [None, abs_phi, st]


def test_negJ_is_zero_for_identity_deformation():
    config = _config()
    h = w = 8
    coord, model_outputs = _identity_model_outputs(config, h, w)
    shape = [-1, h, w, 2]
    loss, _ = negJ_loss(model_outputs, coord, shape)
    assert loss.abs().max() < 1e-4


def test_imdiff_is_zero_when_frames_match_and_transform_identity():
    config = _config()
    h = w = 8
    frame = torch.randn(1, 2, h, w)
    moving = frame.repeat(config.time_points, 1, 1, 1)  # identical frames
    _, model_outputs = _identity_model_outputs(config, h, w, moving)
    loss, _ = im_space_l2_loss(config, [moving, None, None, None], model_outputs)
    assert loss.abs().max() < 1e-4


def test_data_fidelity_zero_when_recon_matches_measurements():
    config = _config()
    h = w = 8
    moving = torch.randn(config.time_points, 2, h, w)
    forw = FastmriFT()
    fixed = forw(moving)  # measurements consistent with the recon
    loss, _ = similarity_loss(config, [moving, None, fixed, forw], None)
    assert loss.abs().max() < 1e-5


def test_use_nreps_without_inr_raises_clear_error():
    config = Config(device="cpu", dataset="cmr_P001", use_nreps=True, time_points=3).finalize()
    h = w = 8
    _, model_outputs = _identity_model_outputs(config, h, w)
    with pytest.raises(ValueError, match="use_nreps"):
        im_space_l2_loss(config, [torch.zeros(3, 2, h, w), None, None, None], model_outputs)
