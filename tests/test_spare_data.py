import os
import types

import numpy as np
import pytest
import torch

from stmr.data.spare_data import (
    _dense_grid,
    _load_detector_row,
    _parse_dataset_id,
    _parse_geometry,
    _parse_resp_bin,
    _scan_dir,
    get_data,
    get_operators,
)

_RAW_DIR = os.path.join(_scan_dir("1", "01"), "Proj")

pytestmark = pytest.mark.skipif(not os.path.isdir(_RAW_DIR),
                                reason="SPARE data not present locally (data/raw/spare/...)")


def test_parse_dataset_id():
    assert _parse_dataset_id("cbct_P1T01") == ("1", "01")
    with pytest.raises(ValueError):
        _parse_dataset_id("cmr_P001")


def test_geometry_matches_known_p1_values():
    angles, src_iso, src_det = _parse_geometry(_RAW_DIR)
    assert src_iso == 1000
    assert src_det == 1536
    assert len(angles) == 340
    assert 0 <= angles.min() < angles.max() <= 360


def test_resp_bin_aligned_with_projections():
    angles, _, _ = _parse_geometry(_RAW_DIR)
    resp_bin = _parse_resp_bin(_RAW_DIR)
    assert len(resp_bin) == len(angles)
    assert resp_bin.min() >= 1
    assert resp_bin.max() <= 10


def test_load_detector_row_shape_and_range():
    angles, _, _ = _parse_geometry(_RAW_DIR)
    rows = _load_detector_row(_RAW_DIR, len(angles), row=256, det_size=512)
    assert rows.shape == (len(angles), 512)
    assert (rows >= 0).all()  # line-integral values are non-negative


def test_dense_grid_partitions_measurements_by_bin():
    angles, _, _ = _parse_geometry(_RAW_DIR)
    resp_bin = _parse_resp_bin(_RAW_DIR)
    rows = _load_detector_row(_RAW_DIR, len(angles), row=256, det_size=512)

    n_bins = 10
    dense_angles, dense_sino, mask = _dense_grid(angles, resp_bin, rows, n_bins, 512)

    assert dense_angles.shape == (len(angles),)
    assert np.all(np.diff(dense_angles) >= 0)  # angle-sorted
    assert mask.shape == (n_bins, len(angles), 512)
    # every measured projection lands in exactly one bin's mask, at its own angle row
    assert mask.any(dim=-1).sum().item() == len(angles)
    # unmasked entries are exactly zero
    assert (dense_sino[~mask] == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch-radon needs a GPU")
def test_get_data_and_get_operators_end_to_end():
    pytest.importorskip("torch_radon", reason="torch-radon not installed (see pyproject.toml)")
    config = types.SimpleNamespace(dataset="cbct_P1T01", time_points=10, start_frame=0,
                                   slice_number=0, device="cuda")

    raw, gt, mask = get_data(config)
    assert raw.shape == gt.shape == mask.shape
    assert raw.shape[0] == config.time_points
    assert raw.shape[1] == 1  # real-valued, single-channel
    assert mask.dtype == torch.bool
    assert (raw[~mask] == 0).all()  # raw is zero-filled at unmeasured entries

    full_forw, full_adj, forw_subs, forw_subs_adj = get_operators(config, mask)
    forw_subs, forw_subs_adj = forw_subs.to(config.device), forw_subs_adj.to(config.device)
    full_adj = full_adj.to(config.device)

    gt_im = full_adj(gt.to(config.device))
    assert gt_im.shape == (config.time_points, 1, 512, 512)
    assert torch.isfinite(gt_im).all()

    recon_sino = forw_subs(gt_im)
    assert recon_sino.shape == raw.shape
    assert torch.isfinite(recon_sino).all()
