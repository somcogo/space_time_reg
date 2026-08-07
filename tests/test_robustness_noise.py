"""The perturbation knobs are exact no-ops at sigma=0 and only touch the intended entries."""

import torch

from stmr.data.noise import add_image_noise, add_kspace_noise


def test_kspace_noise_sigma_zero_is_noop():
    torch.manual_seed(0)
    fixed = torch.randn(6, 2, 8, 8)
    mask = torch.zeros(6, 2, 8, 8, dtype=torch.bool)
    mask[..., ::2, :] = True
    assert torch.equal(add_kspace_noise(fixed, mask, 0.0), fixed)


def test_kspace_noise_only_touches_measured_entries():
    torch.manual_seed(0)
    fixed = torch.randn(6, 2, 8, 8)
    mask = torch.zeros(6, 2, 8, 8, dtype=torch.bool)
    mask[..., ::2, :] = True  # measure every other row
    out = add_kspace_noise(fixed, mask, sigma=0.1)

    unmeasured = ~mask
    assert torch.equal(out[unmeasured], fixed[unmeasured])  # untouched
    assert not torch.equal(out[mask], fixed[mask])          # perturbed


def test_image_noise_sigma_zero_is_noop():
    torch.manual_seed(0)
    recon = torch.randn(6, 2, 8, 8)
    assert torch.equal(add_image_noise(recon, 0.0), recon)


def test_image_noise_scales_with_sigma():
    torch.manual_seed(0)
    recon = torch.randn(6, 2, 16, 16)
    small = (add_image_noise(recon, 0.02) - recon).std()
    big = (add_image_noise(recon, 0.20) - recon).std()
    assert 0 < small < big
