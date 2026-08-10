import types

import pytest

from stmr.data.data_utils import get_dataset_capabilities


def test_cmr_capabilities():
    config = types.SimpleNamespace(dataset="cmr_P001")
    caps = get_dataset_capabilities(config)
    assert caps.is_complex_img is True
    assert caps.joint_recon is True


def test_cbct_capabilities():
    config = types.SimpleNamespace(dataset="cbct_P1T01")
    caps = get_dataset_capabilities(config)
    assert caps.is_complex_img is False
    assert caps.joint_recon is True


def test_unknown_family_raises():
    config = types.SimpleNamespace(dataset="unknown_family_xyz")
    with pytest.raises(ValueError, match="Unknown dataset family"):
        get_dataset_capabilities(config)
