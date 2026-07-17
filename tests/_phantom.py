"""Tiny synthetic phantom for the staged-ablation mechanism tests.

A Gaussian blob translated by a known per-frame shift. Channel 0 holds the (real) image,
channel 1 the phase, which is left at 0 so consecutive frames are related by a pure
translation the motion path can recover.
"""

import torch


def moving_phantom(T=3, H=32, W=32, shift_px=2.0, sigma=0.18):
    """Return (moving [T, 2, H, W] real+phase, known_shift_px).

    Frame t is a Gaussian blob whose centre moves by ``shift_px`` pixels along the width
    per frame. The complex/phase channel is zero.
    """
    ys = torch.linspace(-1, 1, H)
    xs = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    moving = torch.zeros(T, 2, H, W)
    dx_norm = shift_px / (W / 2)  # pixel shift in the [-1, 1] normalised grid
    for t in range(T):
        cx = -0.3 + t * dx_norm
        cy = 0.0
        blob = torch.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / sigma)
        moving[t, 0] = blob
    return moving, shift_px
