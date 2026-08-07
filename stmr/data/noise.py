"""Input-perturbation knobs for robustness testing.

Both functions are exact no-ops at ``sigma <= 0`` so they are safe to leave wired into the
pipeline permanently (defaults keep every existing run bit-identical). Noise realizations are
drawn from the global torch RNG, which ``pipeline.run`` seeds via ``set_seed(config.seed)``
before ``prepare_inputs`` -- so a given seed reproduces its noise, and different seeds vary it.
"""

import torch


def add_kspace_noise(fixed: torch.Tensor, kspace_mask: torch.Tensor,
                     sigma: float) -> torch.Tensor:
    """Additive Gaussian noise on the *measured* k-space entries only.

    ``sigma`` is relative to the RMS magnitude of the measured samples, so it acts like an
    inverse-SNR knob independent of the data's absolute scale. Unmeasured (zero-filled)
    entries are left untouched via the mask.
    """
    if not sigma or sigma <= 0:
        return fixed
    m = kspace_mask.to(fixed)
    measured = fixed[m.bool().expand_as(fixed)]
    scale = measured.pow(2).mean().sqrt()  # RMS of the measured k-space
    return fixed + sigma * scale * torch.randn_like(fixed) * m


def add_image_noise(recon: torch.Tensor, sigma: float) -> torch.Tensor:
    """Additive Gaussian noise on the init reconstruction, relative to its signal std.

    Perturbs the image the flow registers against (the effective perturbation for the
    registration-only S0 setting, where no k-space data term is active).
    """
    if not sigma or sigma <= 0:
        return recon
    scale = recon.detach().std()
    return recon + sigma * scale * torch.randn_like(recon)
