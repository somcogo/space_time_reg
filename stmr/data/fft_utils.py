import torch
from fastmri import fftshift, ifftshift
from torch import nn


def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.fft``.

    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data
    
def ifft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.ifft``.

    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data
    
class FastmriFT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image: torch.Tensor):
        spectrum = fft2c_new(image.movedim(1, -1)).movedim(-1, 1)
        return spectrum
    
class FastmriIFT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, spectrum: torch.Tensor):
        image = ifft2c_new(spectrum.movedim(1, -1)).movedim(-1, 1)
        return image

class MaskedFT(nn.Module):
    """Forward operator F: centered FFT followed by k-space masking.

    Unlike FTAndSubsample, the output keeps the full (T, 2, H, W) shape with zeros at
    unmeasured entries, and the full per-frame mask is kept. This supports time-varying
    (k-t) sampling masks, where the number of measured rows may differ between frames.
    """
    def __init__(self, kspace_mask):
        super().__init__()
        self.register_buffer("mask", kspace_mask)

    def forward(self, image: torch.Tensor):
        spectrum = fft2c_new(image.movedim(1, -1)).movedim(-1, 1)
        return spectrum * self.mask

class MaskedIFT(nn.Module):
    """Adjoint of MaskedFT: k-space masking followed by centered IFFT."""
    def __init__(self, kspace_mask):
        super().__init__()
        self.register_buffer("mask", kspace_mask)

    def forward(self, spectrum: torch.Tensor):
        spectrum = spectrum * self.mask
        return ifft2c_new(spectrum.movedim(1, -1)).movedim(-1, 1)

def apply_hard_data_consistency(image: torch.Tensor, measured_kspace: torch.Tensor, kspace_mask: torch.Tensor) -> torch.Tensor:
    """Project an image onto the set of images consistent with the measurements:
    replace the measured k-space entries with the measured values, keep the rest."""
    spectrum = fft2c_new(image.movedim(1, -1)).movedim(-1, 1)
    spectrum = torch.where(kspace_mask, measured_kspace, spectrum)
    return ifft2c_new(spectrum.movedim(1, -1)).movedim(-1, 1)

class FTAndSubsample(nn.Module):
    def __init__(self, kspace_mask):
        super().__init__()
        # kspace_mask is identical across the batch (time) dim by construction (see
        # generate_standard_mask/generate_random_mask*). Collapse that dim to 1 so
        # self.mask.expand(...) below broadcasts correctly for any batch size the caller
        # passes in - not just the original full batch. Without this, callers that operate
        # on a *subset* of the batch (e.g. nmAPG, which drops already-converged items from
        # its batch as optimization proceeds) crash with a shape-mismatch in expand().
        self.register_buffer("mask", kspace_mask[:1])

    def forward(self, image: torch.Tensor):
        spectrum = fft2c_new(image.movedim(1, -1)).movedim(-1, 1)
        new_shape = list(image.shape[:2]) + [-1] + list(image.shape[3:])
        return spectrum[self.mask.expand(spectrum.shape)].reshape(new_shape)

class ZeroFillAndIFT(nn.Module):
    def __init__(self, kspace_mask):
        super().__init__()
        # see FTAndSubsample above for why this is collapsed to a batch size of 1
        self.register_buffer("mask", kspace_mask[:1])

    def forward(self, spectrum: torch.Tensor):
        full_shape = [spectrum.shape[0]] + list(self.mask.shape[1:])
        full_spectrum = torch.zeros(full_shape, device=spectrum.device)
        full_spectrum[self.mask.expand(full_spectrum.shape)] = spectrum.flatten()
        image = ifft2c_new(full_spectrum.movedim(1, -1)).movedim(-1, 1)
        return image