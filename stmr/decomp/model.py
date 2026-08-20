"""SIREN velocity net construction, ODE integration, and image warping helpers.

Thin wrappers over the core primitives (stmr.models.siren.GroupedSiren, torchdiffeq.odeint,
stmr.utils.spatial_transformer.GridSampleTransformer), matching the standalone pattern of
stmr/robustness/synthetic.py::fit_siren_to_flow.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
# Adjoint integration: O(1) memory in the number of ODE steps (recomputes activations on the
# backward pass). Plain odeint retains the full graph across substeps and OOMs at full-res x
# many groups; the production pipeline (registration.py) uses the adjoint for the same reason.
from torchdiffeq import odeint_adjoint as odeint

from stmr.models.siren import GroupedSiren
from stmr.utils.spatial_transformer import GridSampleTransformer


def build_net(layers, groups, omega, device):
    """One near-identity GroupedSiren (one independent SIREN per frame interval)."""
    return GroupedSiren(groups=groups, layers=layers,
                        last_init_zero=True, omega=omega).to(device)


def make_integrator(net, coord, groups, solver, step_size):
    """Return a closure () -> abs_phi [groups, N, 2]: integrate the velocity net from the
    identity grid over t in [0,1]. coord: [N, 2]."""
    t01 = torch.tensor([0.0, 1.0], device=coord.device)
    init = coord.expand(groups, -1, -1).contiguous()
    opts = {"step_size": step_size} if step_size else None

    def integrate():
        return odeint(net, init, t01, method=solver, options=opts)[1]  # [groups, N, 2]

    return integrate


def warp(abs_phi, src, H, W):
    """Warp src [groups, C, H, W] by the deformation abs_phi [groups, H*W, 2]."""
    return GridSampleTransformer(abs_phi, (src.shape[1], H, W)).apply(src)


def compose_fields(abs_phi1, abs_phi2, H, W):
    """Compose two absolute deformations: phi1 o phi2 (apply phi2 then phi1, grid_sample
    pull-back semantics) = phi1 evaluated at phi2's sampling coordinates. This warps the
    original image ONCE (single interpolation) instead of chaining two image warps.

    Both fields are [groups, H*W, 2] absolute sampling coords in [-1,1], (y, x) order.
    Uses padding_mode='border' since we are resampling a coordinate field (zero-padding would
    snap out-of-range boundary samples to the image centre).
    """
    G = abs_phi1.shape[0]
    phi1_img = abs_phi1.reshape(G, H, W, 2).permute(0, 3, 1, 2)          # [G,2,H,W] (y,x)
    g = abs_phi2.reshape(G, H, W, 2)
    grid = torch.stack([g[..., 1], g[..., 0]], dim=-1)                   # -> (x,y) for grid_sample
    comp = F.grid_sample(phi1_img, grid, align_corners=True, padding_mode="border")
    return comp.permute(0, 2, 3, 1).reshape(G, H * W, 2)                 # [G,H*W,2] (y,x)
