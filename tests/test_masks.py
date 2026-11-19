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


# --- acs_lines, mask_seed, variable density (get_kspace_mask) ---

import random
from types import SimpleNamespace

import pytest

from stmr.data.data_utils import (MASKS, generate_kt_vd_mask, generate_random_mask,
                                  generate_vd_mask, get_kspace_mask)

_SHAPE = (12, 2, 1, 204, 8)  # [T, 2, D, H, W], H = the challenge's 204 phase-encode rows


def _rows(mask, t=0):
    """The sampled row indices of frame t (rows are full along W, equal over channels/slices)."""
    return set(torch.nonzero(mask[t, 0, 0, :, 0]).flatten().tolist())


def _mask(kind, **fields):
    cfg = SimpleNamespace(mask=kind, **fields)
    return get_kspace_mask(cfg, torch.zeros(_SHAPE), fields.pop("factor", 8))


@pytest.mark.parametrize("factor, n_rows", [(4, 69), (8, 47), (10, 42)])
def test_standard_mask_matches_the_challenge_masks(factor, n_rows):
    # The AccFactor04/08/10 cine masks of P001 (SAX and LAX) sample 69 / 47 / 42 of 204 rows:
    # rows 0, f, 2f, ... and the 24 ACS rows 90..113.
    rows = _rows(generate_standard_mask(_SHAPE, factor))
    assert len(rows) == n_rows
    assert set(range(90, 114)) <= rows and set(range(0, 204, factor)) <= rows


def test_acs_lines_sets_the_centre_block():
    for acs in (0, 7, 16, 24):
        rows = _rows(_mask("st", acs_lines=acs, factor=8))
        start = 204 // 2 - acs // 2
        assert set(range(start, start + acs)) <= rows
        assert rows == set(range(start, start + acs)) | set(range(0, 204, 8))
    # the default is the challenge's 24
    assert torch.equal(_mask("st", factor=8), _mask("st", acs_lines=24, factor=8))


@pytest.mark.parametrize("kind", ["random", "random2", "kt_random", "vd", "kt_vd"])
def test_random_masks_depend_on_mask_seed_only(kind):
    a = _mask(kind, mask_seed=3)
    # the global RNG state (what config.seed sets) does not change the mask
    random.seed(123); torch.manual_seed(123)
    b = _mask(kind, mask_seed=3)
    random.seed(7); torch.manual_seed(7)
    c = _mask(kind, mask_seed=3)
    assert torch.equal(a, b) and torch.equal(a, c)
    assert not torch.equal(a, _mask(kind, mask_seed=4))


def test_unseeded_generators_keep_drawing_from_the_global_rng():
    # direct callers without a seed get the old behaviour (global `random` state)
    random.seed(1); a = generate_random_mask(_SHAPE, 8)
    random.seed(1); b = generate_random_mask(_SHAPE, 8)
    assert torch.equal(a, b)


@pytest.mark.parametrize("factor", [4, 8, 10])
@pytest.mark.parametrize("acs", [24, 16, 0])
def test_vd_masks_sample_as_many_rows_as_the_equispaced_mask(factor, acs):
    n_st = len(_rows(generate_standard_mask(_SHAPE, factor, acs)))
    acs_rows = set(range(204 // 2 - acs // 2, 204 // 2 - acs // 2 + acs))
    vd = generate_vd_mask(_SHAPE, factor, acs, seed=0)
    kt = generate_kt_vd_mask(_SHAPE, factor, acs, seed=0)
    assert all(torch.equal(vd[0], vd[t]) for t in range(_SHAPE[0]))  # static
    assert len({frozenset(_rows(kt, t)) for t in range(_SHAPE[0])}) > 1  # per frame
    for m, t in [(vd, 0)] + [(kt, t) for t in range(_SHAPE[0])]:
        assert len(_rows(m, t)) == n_st
        assert acs_rows <= _rows(m, t)
    # full rows, the same over the complex channels and slices
    assert torch.equal(kt, kt[..., :1].expand_as(kt))
    assert torch.equal(kt[:, 0], kt[:, 1])


def test_vd_density_is_highest_near_k0():
    # Over many frames, rows just outside the ACS are sampled far more often than the edge rows.
    shape = (400, 2, 1, 204, 1)
    freq = generate_kt_vd_mask(shape, 8, 24, sigma=0.25, seed=0)[:, 0, 0, :, 0].float().mean(0)
    near = torch.cat([freq[80:90], freq[114:124]]).mean()
    edge = torch.cat([freq[:10], freq[-10:]]).mean()
    assert near > 3 * edge > 0
    # a narrower density concentrates further
    narrow = generate_kt_vd_mask(shape, 8, 24, sigma=0.1, seed=0)[:, 0, 0, :, 0].float().mean(0)
    assert torch.cat([narrow[:10], narrow[-10:]]).mean() < edge


def test_unknown_mask_raises_instead_of_falling_back_to_st():
    assert set(MASKS) == {"st", "random", "random2", "kt", "kt_random", "vd", "kt_vd"}
    with pytest.raises(ValueError, match="Unknown mask"):
        _mask("kt-vd")
