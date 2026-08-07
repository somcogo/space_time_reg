"""Jacobian-determinant / foldover diagnostics for a saved deformation field.

Reuses the exact operator the training losses use (``_relative_phi_Jacobian`` in
stmr/losses/losses.py), so the identity deformation maps to det=1 everywhere -- including at
the boundary, where the raw finite-difference Jacobian is not unit-scaled. A robust,
diffeomorphic flow has det(J) > 0 everywhere; det(J) <= 0 marks a fold (non-invertible).
"""

import torch

from stmr.losses.losses import _relative_phi_Jacobian
from stmr.state import ModelOutputs
from stmr.utils.spatial_utils import generate_coord_tensor


def detJ_from_phi(phi: torch.Tensor, H: int, W: int, device: str = "cpu") -> torch.Tensor:
    """Jacobian determinant of a saved deformation ``phi`` [B, H*W, 2] -> det map [B, H, W]."""
    with torch.no_grad():
        phi = phi.detach().to(device)
        coord = generate_coord_tensor((H, W), device)
        mo = ModelOutputs(rel_vel=None, abs_phi=phi, transformer=None)
        J = _relative_phi_Jacobian(mo, coord, shape=[-1, H, W, 2])  # [B, H, W, 2, 2]
        return torch.linalg.det(J)                                  # [B, H, W]


def foldover_fraction(detJ: torch.Tensor) -> float:
    """Fraction of pixels with non-positive Jacobian determinant (folds)."""
    return (detJ <= 0).float().mean().item()


def detJ_stats(detJ: torch.Tensor) -> dict:
    """Summary statistics of a det(J) map, for the robustness tables."""
    flat = detJ.reshape(-1)
    return {
        "detJ_min": flat.min().item(),
        "detJ_max": flat.max().item(),
        "detJ_mean": flat.mean().item(),
        "detJ_median": flat.median().item(),
        "detJ_p1": torch.quantile(flat, 0.01).item(),
        "detJ_p99": torch.quantile(flat, 0.99).item(),
        "foldover_frac": foldover_fraction(detJ),
    }
