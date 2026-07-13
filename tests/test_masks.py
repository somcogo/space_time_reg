import torch

from stmr.data.data_utils import generate_kt_mask, generate_standard_mask


def test_standard_mask_keeps_acs_center():
    shape = (4, 2, 64, 64)
    mask = generate_standard_mask(shape, factor=4)
    h = shape[-2]
    center = slice(h // 2 - 12, h // 2 + 12)
    assert mask[..., center, :].all()
    assert mask.dtype == torch.bool


def test_kt_mask_is_time_varying_but_shares_acs():
    shape = (6, 2, 64, 64)
    mask = generate_kt_mask(shape, factor=4)
    h = shape[-2]
    center = slice(h // 2 - 12, h // 2 + 12)
    # every frame keeps the ACS center rows
    assert mask[:, :, center, :].all()
    # but the sampled rows differ across consecutive frames (k-t interleaving)
    assert not torch.equal(mask[0], mask[1])


def test_kt_mask_undersamples():
    shape = (6, 2, 64, 64)
    mask = generate_kt_mask(shape, factor=4)
    # far fewer than all rows are sampled per frame
    frac = mask[0, 0, :, 0].float().mean().item()
    assert frac < 0.75
