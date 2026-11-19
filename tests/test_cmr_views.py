"""Loading the LAX views (and the SAX stack) from the challenge .mat (data/raw/cmrxrecon)."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from stmr.data.data_utils import _DATA_ROOT, _cmr_load
from stmr.tune.spec import Case

_FULL = _DATA_ROOT / "raw" / "cmrxrecon" / "ChallengeData" / "SingleCoil" / "Cine" / "TrainingSet" / "FullSample" / "P001"


def _cfg(**kw):
    base = dict(dataset="cmr_P001", cmr_raw=True, view="sax", slices=[0], start_frame=0,
                time_points=2, factor=4, mask="st", template_frames=0)
    return SimpleNamespace(**{**base, **kw})


@pytest.mark.skipif(not (Path(_FULL) / "cine_lax.mat").exists(), reason="challenge data not on disk")
def test_lax_view_loads_with_its_own_shape():
    raw, gt, mask = _cmr_load(_cfg(view="lax", slices=[1]))
    assert gt.shape == (2, 2, 1, 204, 448)  # [T, 2, D, H, W]; P001's LAX is 204 x 448
    assert mask.shape == gt.shape and torch.equal(raw[mask], gt[mask]) and (raw[~mask] == 0).all()
    sax = _cmr_load(_cfg())[1]
    assert sax.shape == (2, 2, 1, 204, 512)


def test_lax_needs_the_challenge_mat():
    with pytest.raises(ValueError, match="cmr_raw=true"):
        _cmr_load(_cfg(view="lax", cmr_raw=False))
    with pytest.raises(ValueError, match="cmr_raw=true"):
        _cmr_load(_cfg(view="lax", dataset="cmr_test2"))


def test_case_names_tell_views_apart():
    assert Case.from_dict({"dataset": "cmr_P001", "slices": 0}).name == "P001_s0"
    assert Case.from_dict({"dataset": "cmr_P001", "slices": 0, "view": "sax"}).name == "P001_s0"
    assert Case.from_dict({"dataset": "cmr_P001", "slices": 0, "view": "lax"}).name == "P001_lax_s0"
