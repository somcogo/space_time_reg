"""Fan-beam CT forward/adjoint operators (2D mid-slice approximation of cone-beam CBCT).

Mirrors the nn.Module forward/adjoint pair pattern in fft_utils.py (FastmriFT/MaskedFT/
FastmriIFT/MaskedIFT) so the rest of the pipeline (data_load.py, losses.py) needs no
CBCT-specific branching -- only stmr/data/data_utils.py's dataset registry differs
between families. Requires torch-radon (carterbox fork, github.com/carterbox/torch-radon);
see pyproject.toml for install notes -- it has no PyPI package and must be built from
source against the local PyTorch/CUDA install.
"""

import torch
import torch_radon
from torch import nn


def _run_on_cuda(radon_fn, x: torch.Tensor) -> torch.Tensor:
    # torch-radon's CUDA backend requires CUDA input, unlike the FFT operators in
    # fft_utils.py these classes mirror (fft2c_new/ifft2c_new run identically on CPU or
    # GPU). data_load.py::prepare_inputs calls `full_adj`/`full_forw` on still-CPU tensors
    # before any `.to(config.device)` (only `forw_subs`/`forw_subs_adj` get moved first),
    # so these operators must be self-sufficient about landing their input on GPU rather
    # than relying on the caller -- moving back to the input's original device afterward
    # keeps them a drop-in match for the FFT operators' CPU-in/CPU-out-if-CPU-in behavior.
    original_device = x.device
    if not x.is_cuda:
        x = x.cuda()
    out = radon_fn(x)
    return out.to(original_device)


def _fanbeam(det_count: int, angles, src_dist: float, det_dist: float,
            det_spacing: float, image_size: int = None) -> torch_radon.FanBeam:
    # torch-radon infers the reconstruction volume shape from the last forward() call
    # unless a Volume2D with an explicit size is supplied -- required here since
    # FanBeamBackprojector/FanBeamFBP may call backward()/filter_sinogram() on an instance
    # that has never had forward() called on it.
    volume = torch_radon.Volume2D()
    volume.set_size(image_size or det_count, image_size or det_count)
    return torch_radon.FanBeam(det_count=det_count, angles=angles, src_dist=src_dist,
                               det_dist=det_dist, det_spacing=det_spacing, volume=volume)


class FanBeamProjector(nn.Module):
    """Forward operator: image -> sinogram over a fixed, dense angle grid.

    If `mask` is given, the output is zeroed at unmeasured (angle, detector-pixel) entries
    -- mirrors MaskedFT's mask-multiply pattern: the transform always covers the full dense
    grid, and masking simulates undersampling (or, for real acquisitions, encodes exactly
    which angles were actually measured per respiratory-phase frame).
    """

    def __init__(self, det_count: int, angles, src_dist: float, det_dist: float,
                det_spacing: float, mask: torch.Tensor = None, image_size: int = None):
        super().__init__()
        self.radon = _fanbeam(det_count, angles, src_dist, det_dist, det_spacing, image_size)
        if mask is not None:
            self.register_buffer("mask", mask)
        else:
            self.mask = None

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # image: [T, 1, H, W] real-valued -> sinogram: [T, 1, n_angles, det_count]
        sino = _run_on_cuda(self.radon.forward, image[:, 0]).unsqueeze(1)
        if self.mask is not None:
            sino = sino * self.mask
        return sino


class FanBeamBackprojector(nn.Module):
    """Adjoint of FanBeamProjector: sinogram -> image via raw (unfiltered) backprojection.

    This is the true adjoint of the forward projection (correct for gradient-based
    optimization, analogous to MaskedIFT) -- not a reconstruction filter. See FanBeamFBP
    for a filtered reconstruction used to build a genuine reference image.
    """

    def __init__(self, det_count: int, angles, src_dist: float, det_dist: float,
                det_spacing: float, mask: torch.Tensor = None, image_size: int = None):
        super().__init__()
        self.radon = _fanbeam(det_count, angles, src_dist, det_dist, det_spacing, image_size)
        if mask is not None:
            self.register_buffer("mask", mask)
        else:
            self.mask = None

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        if self.mask is not None:
            sinogram = sinogram * self.mask
        image = _run_on_cuda(self.radon.backward, sinogram[:, 0]).unsqueeze(1)
        return image


class FanBeamFBP(nn.Module):
    """Filtered backprojection over the full dense angle grid -- the CBCT analogue of
    FastmriIFT as the (approximate) inverse used to build a genuine reference image, e.g.
    `gt_im = full_adj(gt_sinogram)`. Not the adjoint of FanBeamProjector (see
    FanBeamBackprojector for that); ramp filtering makes this a reconstruction, not a
    transpose.
    """

    def __init__(self, det_count: int, angles, src_dist: float, det_dist: float,
                det_spacing: float, image_size: int = None):
        super().__init__()
        self.radon = _fanbeam(det_count, angles, src_dist, det_dist, det_spacing, image_size)

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        def _fbp(x):
            return self.radon.backward(self.radon.filter_sinogram(x))
        image = _run_on_cuda(_fbp, sinogram[:, 0]).unsqueeze(1)
        return image
