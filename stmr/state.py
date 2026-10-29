"""Typed containers for the data threaded through the pipeline.

Replaces the positional ``inputs[0..3]`` / ``model_outputs[0..2]`` / 8-tuple ``output``
lists that used to be unpacked by index all over the codebase.

Images and k-space are always [T, 2, D, H, W] (D = 1 for a single slice). Deformations and
velocities are [T-1, N, C] on the motion grid: (H, W), N = H*W, C = 2 (y, x) for a single
slice, which only moves in-plane; (D, H, W), N = D*H*W, C = 3 (z, y, x) for several slices.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class Inputs:
    """Everything the optimisation reads each epoch."""

    moving: torch.Tensor          # recon parameter [T, 2, D, H, W]
    moving_inr: object | None     # neural-representation image (None for cmr)
    fixed: torch.Tensor           # measured k-space [T, 2, D, H, W], zero-filled
    forward: nn.Module            # forward operator (exposes .mask when masked)
    phi_shape: list               # shape of deformation [T, H, W, 2] if D=1 else [T, D, H, W, 3]


@dataclass
class EvalInputs:
    """Reference tensors used only for metrics / visualisation."""

    init_recon: torch.Tensor      # [T, 2, D, H, W]
    gt_im: torch.Tensor           # [T, 2, D, H, W]
    seg_moving: object | None
    seg_fixed: object | None
    gt_phi: torch.Tensor | None = None  # known motion [T-1, H*W, 2] (known-motion test, single slice only)


@dataclass
class ModelOutputs:
    """Output of the velocity network + ODE flow for one epoch."""

    rel_vel: torch.Tensor | None  # velocity field on the identity grid [T-1, N, C] (grouped net)
    abs_phi: torch.Tensor         # deformation phi_{t,t+1} applied to the grid [T-1, N, C]
    transformer: object | None    # spatial transformer that warps images with abs_phi


@dataclass
class LossOutputs:
    """Per-loss bookkeeping dict plus the last warped image (for visualisation)."""

    losses: dict
    moved_imspace: torch.Tensor | None


@dataclass
class RegistrationResult:
    """Best-epoch snapshot returned by :func:`registration`."""

    model_outputs: ModelOutputs
    loss_outputs: LossOutputs
    best_moving: torch.Tensor
    st_dict: dict
    epoch: int
    all_metrics: list
    time_stamps: np.ndarray
    coords: torch.Tensor
