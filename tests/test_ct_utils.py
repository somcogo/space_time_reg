import numpy as np
import pytest
import torch

pytest.importorskip("torch_radon", reason="torch-radon not installed (see pyproject.toml)")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="torch-radon's CUDA backend requires a GPU")

from stmr.data.ct_utils import (
    FanBeamBackprojector,
    FanBeamFBP,
    FanBeamProjector,
)

_DET_COUNT = 32
_N_ANGLES = 40
_ANGLES = np.linspace(0, 2 * np.pi, _N_ANGLES, endpoint=False)
_GEOM = dict(det_count=_DET_COUNT, angles=_ANGLES, src_dist=1000.0, det_dist=536.0,
            det_spacing=0.8 * (1000.0 / 1536.0))


def _rand_img(t=3):
    return torch.rand(t, 1, _DET_COUNT, _DET_COUNT, device="cuda")


def _rand_sino(t=3):
    return torch.rand(t, 1, _N_ANGLES, _DET_COUNT, device="cuda")


def test_forward_output_shape():
    proj = FanBeamProjector(**_GEOM).cuda()
    sino = proj(_rand_img())
    assert sino.shape == (3, 1, _N_ANGLES, _DET_COUNT)


def test_mask_zeros_unmeasured_entries():
    mask = torch.rand(3, 1, _N_ANGLES, _DET_COUNT, device="cuda") > 0.5
    proj = FanBeamProjector(mask=mask, **_GEOM).cuda()
    sino = proj(_rand_img())
    assert (sino[~mask] == 0).all()


def test_backprojector_is_adjoint_of_projector():
    # <A x, y> == <x, A* y> for the (masked) forward/backprojection pair.
    mask = torch.rand(3, 1, _N_ANGLES, _DET_COUNT, device="cuda") > 0.5
    proj = FanBeamProjector(mask=mask, **_GEOM).cuda()
    backproj = FanBeamBackprojector(mask=mask, **_GEOM).cuda()
    x, y = _rand_img(), _rand_sino()

    lhs = (proj(x) * y).sum()
    rhs = (x * backproj(y)).sum()
    # torch-radon's forward/backward are only approximately adjoint (bilinear
    # interpolation in the CUDA kernel isn't perfectly symmetric) -- a few percent
    # relative tolerance is the right bar, not near-exact equality.
    assert (lhs - rhs).abs() / rhs.abs() < 0.03


def test_fbp_recovers_a_simple_phantom():
    fbp = FanBeamFBP(**_GEOM).cuda()
    proj = FanBeamProjector(**_GEOM).cuda()

    img = torch.zeros(1, 1, _DET_COUNT, _DET_COUNT, device="cuda")
    img[0, 0, _DET_COUNT // 4:3 * _DET_COUNT // 4, _DET_COUNT // 4:3 * _DET_COUNT // 4] = 1.0

    sino = proj(img)
    recon = fbp(sino)
    # FBP is only an approximate inverse (unlike centered FFT/IFFT for MRI); a loose
    # correlation/shape check is the right bar here, not near-exact recovery.
    assert recon.shape == img.shape
    assert torch.isfinite(recon).all()
    fg = img > 0.5
    assert recon[fg].mean() > recon[~fg].mean()
