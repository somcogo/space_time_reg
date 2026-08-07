"""det(J) recipe correctness: identity->1, uniform dilation->s^2, a fold-> negative det."""

import torch

from stmr.robustness.jacobian import detJ_from_phi, detJ_stats, foldover_fraction
from stmr.utils.spatial_utils import generate_coord_tensor

H = W = 16


def test_identity_has_unit_determinant():
    coord = generate_coord_tensor((H, W), "cpu")
    detJ = detJ_from_phi(coord.unsqueeze(0), H, W)
    assert torch.allclose(detJ, torch.ones_like(detJ), atol=1e-3)
    assert foldover_fraction(detJ) == 0.0


def test_uniform_dilation_determinant_is_s_squared():
    coord = generate_coord_tensor((H, W), "cpu")
    for s in (0.5, 1.5):
        detJ = detJ_from_phi((s * coord).unsqueeze(0), H, W)
        assert torch.allclose(detJ, torch.full_like(detJ, s * s), atol=1e-3)


def test_fold_gives_negative_determinant():
    coord = generate_coord_tensor((H, W), "cpu")
    folded = coord.clone()
    folded[:, 1] = -folded[:, 1]  # mirror the x coordinate -> orientation reversed
    detJ = detJ_from_phi(folded.unsqueeze(0), H, W)
    assert (detJ <= 0).any()
    assert foldover_fraction(detJ) > 0.5
    stats = detJ_stats(detJ)
    assert stats["detJ_min"] < 0
