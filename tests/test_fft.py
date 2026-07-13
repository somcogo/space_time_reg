import torch

from stmr.data.fft_utils import (
    MaskedFT,
    MaskedIFT,
    apply_hard_data_consistency,
    fft2c_new,
    ifft2c_new,
)


def _rand_img(t=2, h=16, w=16):
    # (T, 2, H, W): 2 = real/imag channels
    return torch.randn(t, 2, h, w)


def test_fft_ifft_roundtrip():
    x = _rand_img()
    xr = x.movedim(1, -1)
    back = ifft2c_new(fft2c_new(xr))
    assert torch.allclose(back, xr, atol=1e-5)


def test_masked_ft_is_adjoint_consistent():
    # <F x, y> == <x, F* y> for the masked operator (real inner product on 2-ch tensors).
    mask = torch.rand(2, 2, 16, 16) > 0.5
    ft, ift = MaskedFT(mask), MaskedIFT(mask)
    x, y = _rand_img(), _rand_img()
    lhs = (ft(x) * y).sum()
    rhs = (x * ift(y)).sum()
    assert torch.allclose(lhs, rhs, atol=1e-4)


def test_hard_data_consistency_pins_measured_entries():
    mask = torch.rand(2, 2, 16, 16) > 0.5
    measured = fft2c_new(_rand_img().movedim(1, -1)).movedim(-1, 1)
    img = _rand_img()
    projected = apply_hard_data_consistency(img, measured, mask)
    k = fft2c_new(projected.movedim(1, -1)).movedim(-1, 1)
    # measured entries must equal the measurements after projection
    assert torch.allclose(k[mask], measured[mask], atol=1e-4)
