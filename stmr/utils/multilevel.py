import math
from argparse import Namespace

import torch
import torch.nn.functional as F

from stmr.data.fft_utils import FastmriFT, FastmriIFT, MaskedFT
from stmr.utils.spatial_utils import generate_coord_tensor, get_phi_shape
from stmr.state import Inputs

def crop_kspace(x: torch.Tensor, f: int) -> torch.Tensor:
    H, W = x.shape[-2:]
    H_start = H//2 - H//(2*f)
    W_start = W//2 - W//(2*f)
    H_new = H // f
    W_new = W // f
    x_new = x[..., H_start: H_start + H_new, W_start: W_start + W_new]
    if x.dtype == torch.bool:
        return x_new
    scale = math.sqrt(H_new*W_new / (H*W))
    return x_new * scale

def restrict(x: torch.Tensor, f: int) -> torch.Tensor:
    fft = FastmriFT().to(x.device)
    ifft = FastmriIFT().to(x.device)
    return ifft(crop_kspace(fft(x), f))

def prolong(x: torch.Tensor, fine_shape: torch.Size) -> torch.Tensor:
    fft = FastmriFT().to(x.device)
    ifft = FastmriIFT().to(x.device)
    H, W = x.shape[-2:]
    H_new, W_new = fine_shape
    H1 = H_new//2 - H//2
    H2 = H_new - H - H1
    W1 = W_new//2 - W//2
    W2 = W_new - W - W1
    scale = math.sqrt(H_new*W_new / (H*W))

    return ifft(F.pad(fft(x), (W1, W2, H1, H2))) * scale

def level_coords(native_grid, f: int, device):
    """Motion grid of level f, in native (f=1) [-1, 1] coordinates.

    native_grid: the f=1 motion grid, (H, W) or (D, H, W). Only the last two (in-plane) axes
    are coarsened, like crop_kspace; the slice axis is kept.

    Coarse pixel j sits on native pixel f*j + o, o = N//2 - f*(Nc//2) (centred k-space crop),
    so the coarse grid does not span the native [-1, 1]: with align_corners=True its own
    coordinate g maps to the native u = a*g + b.

    Returns
      coords:      [prod(coarse_grid), C] coarse pixel centres in native coordinates; the
                   input to the velocity net and the ODE at this level
      a, b:        [C], in phi's channel order; the warp samples the coarse image at
                   g = (phi - b) / a
      coarse_grid: spatial shape of the coarse motion grid
    """
    a, b, coarse_grid = [], [], []
    for axis, N in enumerate(native_grid):
        if axis < len(native_grid) - 2:  # slice axis: not coarsened
            a.append(1.0); b.append(0.0); coarse_grid.append(N)
            continue
        Nc = N // f
        if Nc < 2:
            raise ValueError(f"f={f} leaves {Nc} pixel(s) on an axis of {N}")
        o = N // 2 - f * (Nc // 2)
        a_k = f * (Nc - 1) / (N - 1)
        a.append(a_k)
        b.append(a_k - 1 + 2 * o / (N - 1))
        coarse_grid.append(Nc)
    g = generate_coord_tensor(coarse_grid, device)
    a = torch.tensor(a, device=g.device, dtype=g.dtype)
    b = torch.tensor(b, device=g.device, dtype=g.dtype)
    return g * a + b, a, b, coarse_grid  # f=1: a=1, b=0 exactly -> the standard grid

def build_level(fine_inputs: Inputs, f: int) -> list[Inputs, torch.Tensor, tuple]:
    if f == 1:
        coords = generate_coord_tensor(fine_inputs.native_grid, fine_inputs.moving.device)
        return fine_inputs, coords, (1., 0.)
    moving = torch.nn.Parameter(restrict(fine_inputs.moving.detach(), f))
    fixed = crop_kspace(fine_inputs.fixed, f)
    forward = MaskedFT(crop_kspace(fine_inputs.forward.mask))
    coarse_coords, a, b, coarse_grid = level_coords(fine_inputs.native_grid, f, fine_inputs.moving.device)
    phi_shape = get_phi_shape(coarse_grid)
    coarse_inputs = Inputs(moving=moving, moving_inr=fine_inputs.moving_inr, fixed=fixed, forward=forward, phi_shape=phi_shape, native_grid=fine_inputs.native_grid)
    return coarse_inputs, coarse_coords, (a, b)