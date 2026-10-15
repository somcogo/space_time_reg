"""Known-motion test frames: the real first frame carried along a known deformation.

The motion learned on real frames has no ground truth. Here the frames are replaced by frame 0
warped, frame after frame, by a *known* motion, so the learned motion can be scored against it
(stmr.metrics.motion_metrics.endpoint_error). Fresh noise at the level measured in the real
frames is added, and optionally a known non-motion brightness change of the bright
(blood-pool-like) region, to measure how much such a change biases the learned motion.
Used by the ablation stages S0K / S0R via ``Config.synth_flow``.
"""

import torch
import torch.nn.functional as F

from stmr.metrics.motion_metrics import heart_crop, magnitude
from stmr.utils.spatial_transformer import GridSampleTransformer
from stmr.utils.spatial_utils import generate_coord_tensor


def noise_sigma(gt_im: torch.Tensor) -> torch.Tensor:
    """Per-channel noise std from frame-to-frame differences in air (pixels below 3% of the max
    in every frame). Approximate: selecting pixels by darkness biases it slightly low."""
    mag = magnitude(gt_im)
    air = mag.max(0).values < 0.03 * mag.max()
    diff = gt_im[1:] - gt_im[:-1]
    return (diff[:, :, air].pow(2).mean() / 2).sqrt()


def known_flow(config, T: int, H: int, W: int, device=None) -> torch.Tensor:
    """The known deformation phi [T-1, H*W, 2], in the pipeline's convention
    (warp(I_t, phi_t) = I_{t+1})."""
    coord = generate_coord_tensor((H, W), device)
    if config.synth_flow == "analytic":
        # Smooth field with synth_flow_amp px per component; the sign flips every interval so
        # the frames oscillate instead of drifting out of the field of view.
        span = torch.tensor([(H - 1) / 2, (W - 1) / 2], device=device)
        y, x = coord.unbind(-1)
        u = config.synth_flow_amp * torch.stack([torch.cos(3 * x), torch.sin(3 * y)], -1) / span
        return coord + torch.stack([u if k % 2 == 0 else -u for k in range(T - 1)])
    phi = torch.load(config.synth_flow, map_location=device, weights_only=False)["phi"]
    if phi.shape[0] < T - 1 or phi.shape[1] != H * W:
        raise ValueError(f"synth_flow {config.synth_flow!r} holds phi of shape {tuple(phi.shape)}; "
                         f"need at least {T - 1} intervals on a {H}x{W} grid")
    return coord + (phi[:T - 1].to(coord) - coord) * config.synth_flow_amp


def bright_region(frame: torch.Tensor, sigma_px: float = 2.0) -> torch.Tensor:
    """Smooth [1, 1, H, W] weight in [0, 1] on the brightest 10% of the heart crop of a
    [2, H, W] frame -- the blood pool in bright-blood cine."""
    mag = magnitude(frame[None])[0]
    crop = heart_crop(*mag.shape, device=mag.device)
    m = ((mag > mag[crop].quantile(0.9)) & crop).to(frame.dtype)[None, None]
    r = int(3 * sigma_px)
    k = torch.exp(-torch.arange(-r, r + 1, device=frame.device, dtype=frame.dtype) ** 2
                  / (2 * sigma_px ** 2))
    k = k / k.sum()
    m = F.conv2d(F.conv2d(m, k.view(1, 1, 1, -1), padding=(0, r)), k.view(1, 1, -1, 1), padding=(r, 0))
    return m / m.max()


def synthesize(config, gt_im: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace the [T, 2, H, W] frames by frame 0 carried along the known motion.

    Returns (frames, phi). Frames are chained, frame_{t+1} = warp(frame_t, phi_t), so phi is
    exactly the frame-to-frame motion the pipeline estimates. Order: motion, then the optional
    brightness change (moving with the anatomy), then fresh noise.
    """
    T, _, H, W = gt_im.shape
    if T < 2:
        raise ValueError("the known-motion test needs at least 2 frames")
    sigma = noise_sigma(gt_im)  # measured on the real frames, before they are replaced
    phi = known_flow(config, T, H, W, gt_im.device).to(gt_im.dtype)

    def warp(img, k):
        return GridSampleTransformer(phi[k:k + 1], img.shape[1:]).apply(img)

    frames = [gt_im[:1]]
    for k in range(T - 1):
        frames.append(warp(frames[-1], k))
    frames = torch.cat(frames)

    if config.synth_gain:
        # Frame t's bright region is brighter by a factor 1 + synth_gain * t (complex scaling:
        # magnitude changes, phase does not), a brightness change no deformation can produce.
        weight = [bright_region(gt_im[0])]
        for k in range(T - 1):
            weight.append(warp(weight[-1], k))
        t = torch.arange(T, device=gt_im.device, dtype=gt_im.dtype).view(T, 1, 1, 1)
        frames = frames * (1 + config.synth_gain * t * torch.cat(weight))

    if config.synth_noise:
        gen = torch.Generator(device=gt_im.device).manual_seed(config.seed)
        frames = frames + config.synth_noise * sigma * torch.randn(
            frames.shape, generator=gen, device=gt_im.device, dtype=gt_im.dtype)
    return frames, phi
