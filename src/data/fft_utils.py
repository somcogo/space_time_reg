import torch
from torch import nn
from fastmri import fftshift, ifftshift
    
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

class FTAndSubsample(nn.Module):
    def __init__(self, kspace_mask):
        super().__init__()
        self.register_buffer("mask", kspace_mask)

    def forward(self, image: torch.Tensor):
        spectrum = fft2c_new(image.movedim(1, -1)).movedim(-1, 1)
        new_shape = list(image.shape[:2]) + [-1] + list(image.shape[3:])
        return spectrum[self.mask.expand(spectrum.shape)].reshape(new_shape)
    
class ZeroFillAndIFT(nn.Module):
    def __init__(self, kspace_mask):
        super().__init__()
        self.register_buffer("mask", kspace_mask)

    def forward(self, spectrum: torch.Tensor):
        full_shape = [spectrum.shape[0]] + list(self.mask.shape[1:])
        full_spectrum = torch.zeros(full_shape, device=spectrum.device)
        full_spectrum[self.mask.expand(full_spectrum.shape)] = spectrum.flatten()
        image = ifft2c_new(full_spectrum.movedim(1, -1)).movedim(-1, 1)
        return image