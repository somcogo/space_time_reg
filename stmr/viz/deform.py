"""Visualise a learned deformation field by propagating one frame through it.

Takes the source frame (frame 0 by default) from a saved run and warps it forward through
the learned per-interval velocity flows, either:

  * to each of the other time points (``--substeps 1``), or
  * on a finer time grid inside every ODE interval (e.g. ``--substeps 10`` -> steps of 0.1),

then shows the propagated images next to the ground-truth frames so you can check the motion
is reasonable. Direction follows the training loss: ``warp(I_k, phi_k) ~= I_{k+1}``.

Two modes (``--mode``):
  * propagate (default): chain the running warped image through the whole cycle
    (warp(warp(warp(I_0, phi_0), phi_1), ...)); accumulates any per-interval warp error.
  * stepwise: warp each *real* GT frame once -- GT_0, warp(GT_0, phi_0), warp(GT_1, phi_1),
    ... -- re-anchored to the true frame every step, so there is no accumulated drift.

Outputs (into ``--out``):
  propagate_grid.png       propagated vs GT vs |difference| (coarse) or a contact sheet (fine)
  deform.gif               the propagated source animated through the cycle
  displacement_quiver.png  the per-interval displacement field (magnitude + arrows)

Usage:
  python -m stmr.viz.deform --res log/<run>/res.pt --out figs/<run> --substeps 1
  python -m stmr.viz.deform --res log/<run>/res.pt --out figs/<run> --substeps 10
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from fastmri import complex_abs  # noqa: E402
from torchdiffeq import odeint  # noqa: E402

from stmr.config import Config, build_velocity_kwargs  # noqa: E402
from stmr.models.factory import get_func  # noqa: E402
from stmr.utils.spatial_transformer import GridSampleTransformer  # noqa: E402
from stmr.utils.spatial_utils import generate_coord_tensor  # noqa: E402


def to_mag(img4: torch.Tensor) -> np.ndarray:
    """[B, 2, H, W] complex -> [B, H, W] magnitude as a numpy array."""
    return complex_abs(img4.movedim(1, -1)).cpu().numpy()


def load_run(res_path: str, device: str = "cpu"):
    d = torch.load(res_path, map_location=device, weights_only=False)
    cfg = Config.from_dict(d["config"]) if isinstance(d["config"], dict) else d["config"]
    cfg.device = device
    return d, cfg


def rebuild_func(cfg: Config, d: dict, device: str):
    """Reconstruct the velocity net from the saved weights."""
    func = get_func(cfg.func_name, build_velocity_kwargs(cfg, dims=2)).to(device)
    func.load_state_dict(d["st_dict"])
    func.eval()
    return func


def flows_by_time(d, cfg, H, W, substeps, device):
    """Return (times [n], phis [n, T-1, H*W, 2]) where phis[i, k] is interval k's flow at
    sub-time times[i]. substeps==1 reuses the saved best-epoch phi (no net rebuild); >1
    re-integrates the ODE on the finer grid."""
    coord = generate_coord_tensor((H, W), device)
    if substeps == 1:
        # times [0, 1]; index 0 is identity (unused by the trajectory builder), index 1 is
        # the saved full-interval deformation.
        times = torch.tensor([0.0, 1.0], device=device)
        phi = d["phi"].to(device)
        ident = coord.expand_as(phi)
        return times, torch.stack([ident, phi], dim=0)

    func = rebuild_func(cfg, d, device)
    times = torch.linspace(0.0, 1.0, substeps + 1, device=device)
    init = coord.expand((cfg.time_points - 1, -1, -1)).contiguous()
    opts = {"step_size": cfg.step_size} if cfg.step_size else None
    with torch.no_grad():
        phis = odeint(func, init, times, method=cfg.solver, atol=cfg.atol, rtol=cfg.rtol,
                      options=opts)
    return times, phis


def build_trajectory(source, phis, times, T, img_shape, start=0):
    """Chain-warp the source frame forward through the interval flows.

    Within interval k, each sub-time flow phi_k^s starts from frame k's identity, so warping
    the frame-k image by phi_k^s gives global time k+s; the interval's final flow (s=1)
    advances the running image to frame k+1. Returns (frames list of [1,2,H,W], global-time
    labels)."""
    substeps = len(times) - 1
    frames, labels = [source], [float(start)]
    img_k = source
    for k in range(start, T - 1):
        for i in range(1, substeps + 1):
            phi = phis[i, k:k + 1]  # [1, H*W, 2]
            warped = GridSampleTransformer(phi, img_shape).apply(img_k)
            frames.append(warped)
            labels.append(float(k + times[i].item()))
        img_k = frames[-1]  # reached frame k+1 (s=1)
    return frames, labels


def build_stepwise_trajectory(gt_im, phis, times, T, img_shape, start=0):
    """Warp each interval's OWN ground-truth source frame one step, instead of chaining.

    Sequence: GT_start, warp(GT_start, phi_start), warp(GT_{start+1}, phi_{start+1}), ...
    i.e. every frame k is the *actual* GT frame k warped by its interval flow (so it should
    look like GT_{k+1}). Unlike build_trajectory this re-anchors to the true frame each step,
    so there is no accumulated drift -- it mirrors the training loss warp(I_k, phi_k) ~=
    I_{k+1}. With substeps>1 the sub-steps morph GT_k continuously toward GT_{k+1}."""
    substeps = len(times) - 1
    frames, labels = [gt_im[start:start + 1]], [float(start)]
    for k in range(start, T - 1):
        src_k = gt_im[k:k + 1]  # the actual GT frame k, not the running warped image
        for i in range(1, substeps + 1):
            phi = phis[i, k:k + 1]
            frames.append(GridSampleTransformer(phi, img_shape).apply(src_k))
            labels.append(float(k + times[i].item()))
    return frames, labels


def _psnr(gt, pred):
    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(gt.max() ** 2 / mse)


def plot_coarse_grid(frames, labels, gt_im, start, out_path):
    """Propagated source (row 1) vs GT frames (row 2) vs |difference| (row 3).

    The diff row is scaled to the errors' own magnitude (not the GT brightness, which would
    wash small diffs out), on a single shared scale across all frames so panels are directly
    comparable, with one colorbar giving the absolute magnitude and each panel's own peak in
    its title."""
    warped = np.concatenate([to_mag(f) for f in frames], axis=0)  # [n, H, W]
    gt = to_mag(gt_im)[start:start + len(frames)]                 # aligned GT frames
    n = len(frames)
    peak = float(gt.max())  # peak signal: the reference the error is normalised against
    # Brighten the grayscale rows with a robust (99th-pct) window so the anatomy -- and the
    # visible error -- isn't crushed by a lone hot pixel.
    disp_vmax = max(float(np.percentile(gt, 99)), 1e-8)
    # Diff normalised to a percentage of peak signal, so "how bad" is read directly.
    diff_pct = np.abs(warped[:len(gt)] - gt[:len(warped)]) / peak * 100.0
    dvmax = max(float(np.percentile(diff_pct, 99.5)) if diff_pct.size else 1.0, 1e-8)

    fig, axes = plt.subplots(3, n, figsize=(2.1 * n, 6.8), layout="constrained")
    axes = np.atleast_2d(axes)
    diff_img = None
    for j in range(n):
        p = _psnr(gt[j], warped[j]) if j < len(gt) else float("nan")
        axes[0, j].imshow(warped[j], cmap="gray", vmin=0, vmax=disp_vmax)
        axes[0, j].set_title(f"t={labels[j]:.0f}\n{p:.1f} dB", fontsize=8)
        axes[1, j].imshow(gt[j] if j < len(gt) else np.zeros_like(warped[j]),
                          cmap="gray", vmin=0, vmax=disp_vmax)
        if j < len(gt):
            diff_img = axes[2, j].imshow(diff_pct[j], cmap="magma", vmin=0, vmax=dvmax)
            axes[2, j].set_title(f"p99 {np.percentile(diff_pct[j], 99):.1f}%", fontsize=8)
        else:
            axes[2, j].imshow(np.zeros_like(warped[j]), cmap="magma", vmin=0, vmax=dvmax)
        for r in range(3):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
    axes[0, 0].set_ylabel("warp(I_0)", fontsize=9)
    axes[1, 0].set_ylabel("GT", fontsize=9)
    axes[2, 0].set_ylabel("|diff|", fontsize=9)
    if diff_img is not None:
        cbar = fig.colorbar(diff_img, ax=axes[2, :].tolist(), location="right",
                            fraction=0.02, pad=0.02, aspect=30)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("|diff|  (% of peak signal, shared scale)", fontsize=7)
    fig.suptitle("Propagated source vs ground-truth frames", fontsize=11)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_contact_sheet(frames, labels, out_path, ncols=11):
    """A grid of every propagated frame (used for the fine-time-grid mode)."""
    mags = np.concatenate([to_mag(f) for f in frames], axis=0)
    vmax = float(mags.max())
    n = len(frames)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.5 * ncols, 1.6 * nrows))
    axes = np.array(axes).reshape(-1)
    for j in range(len(axes)):
        axes[j].set_xticks([]); axes[j].set_yticks([])
        if j < n:
            axes[j].imshow(mags[j], cmap="gray", vmin=0, vmax=vmax)
            axes[j].set_title(f"t={labels[j]:.2f}", fontsize=7)
        else:
            axes[j].axis("off")
    fig.suptitle("Source propagated through the learned flow (fine time grid)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def save_gif(frames, out_path, fps=6):
    """Write a magnitude GIF. ``frames`` is either a list of [1, 2, H, W] tensors or a single
    [T, 2, H, W] tensor (e.g. a GT cine straight from res.pt['gt_im'])."""
    import imageio.v2 as imageio

    if isinstance(frames, torch.Tensor):
        mags = to_mag(frames)  # [T, 2, H, W] -> [T, H, W]
    else:
        mags = np.concatenate([to_mag(f) for f in frames], axis=0)
    vmax = float(mags.max()) + 1e-8
    uint8 = np.clip(mags / vmax * 255, 0, 255).astype(np.uint8)
    imageio.mimsave(out_path, list(uint8), fps=fps, loop=0)


def plot_displacement(phi_full, gt_im, H, W, start, out_path, mag=5.0):
    """Per-interval displacement field: magnitude heatmap + (magnified) quiver arrows.

    phi_full: [T-1, H*W, 2] full-interval deformations (sampling coordinates). The
    displacement phi - identity is in normalised [-1, 1] units; converted to pixels for the
    quiver. Arrow lengths are multiplied by ``mag`` for visibility (noted in the title)."""
    coord = generate_coord_tensor((H, W), phi_full.device)
    disp = (phi_full - coord).reshape(-1, H, W, 2).cpu().numpy()  # [T-1, H, W, (dy,dx)]
    T1 = disp.shape[0]
    gt = to_mag(gt_im)
    vmax = float(gt.max())
    dy_px = disp[..., 0] * (H - 1) / 2.0
    dx_px = disp[..., 1] * (W - 1) / 2.0
    dmag = np.sqrt(dx_px ** 2 + dy_px ** 2)

    step = max(1, H // 24)
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    fig, axes = plt.subplots(1, T1, figsize=(2.3 * T1, 2.6))
    axes = np.atleast_1d(axes)
    for k in range(T1):
        axes[k].imshow(gt[start + k] if start + k < gt.shape[0] else gt[k],
                       cmap="gray", vmin=0, vmax=vmax)
        axes[k].imshow(dmag[k], cmap="magma", alpha=0.45)
        u = dx_px[k, ::step, ::step] * mag
        v = dy_px[k, ::step, ::step] * mag
        axes[k].quiver(xs, ys, u, v, color="cyan", angles="xy",
                       scale_units="xy", scale=1, width=0.004)
        axes[k].set_title(f"interval {start + k}->{start + k + 1}\nmax {dmag[k].max():.1f}px",
                          fontsize=8)
        axes[k].set_xticks([]); axes[k].set_yticks([])
    fig.suptitle(f"Learned displacement per interval (arrows x{mag:g})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def visualise(res_path, out_dir, substeps=1, source="gt", start=0, device="cpu",
              quiver_mag=5.0, mode="stepwise"):
    os.makedirs(out_dir, exist_ok=True)
    d, cfg = load_run(res_path, device)
    T = cfg.time_points
    src_key = "gt_im" if source == "gt" else "moving"
    imgs = d[src_key].to(device)
    _, _, H, W = imgs.shape
    img_shape = imgs.shape[1:]  # [2, H, W]

    times, phis = flows_by_time(d, cfg, H, W, substeps, device)
    if mode == "stepwise":
        # Each frame is the actual source frame k warped once by phi_k (re-anchored to GT
        # every step; no accumulated drift).
        frames, labels = build_stepwise_trajectory(imgs, phis, times, T, img_shape, start)
    else:
        frames, labels = build_trajectory(imgs[start:start + 1], phis, times, T,
                                          img_shape, start)

    grid_path = os.path.join(out_dir, "propagate_grid.png")
    if substeps == 1:
        plot_coarse_grid(frames, labels, d["gt_im"].to(device), start, grid_path)
    else:
        plot_contact_sheet(frames, labels, grid_path)

    gif_path = os.path.join(out_dir, "deform.gif")
    save_gif(frames, gif_path)

    quiver_path = os.path.join(out_dir, "displacement_quiver.png")
    plot_displacement(phis[-1], d["gt_im"].to(device), H, W, start, quiver_path, quiver_mag)

    # Console summary: how well each propagated frame matches its GT target.
    warped = np.concatenate([to_mag(f) for f in frames], axis=0)
    gt = to_mag(d["gt_im"].to(device))
    print(f"Loaded {res_path}: T={T}, source={source} frame {start}, "
          f"substeps={substeps}, mode={mode}")
    print(f"Built {len(frames)} frames over global time {labels[0]:.2f}..{labels[-1]:.2f} "
          f"({'each GT frame warped once' if mode == 'stepwise' else 'source propagated'})")
    if substeps == 1:
        print("frame  global_t  PSNR(warp vs GT)")
        for j, lab in enumerate(labels):
            gi = start + j
            if gi < gt.shape[0]:
                print(f"  {j:3d}   {lab:6.2f}    {_psnr(gt[gi], warped[j]):6.2f} dB")
    print(f"Wrote:\n  {grid_path}\n  {gif_path}\n  {quiver_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", required=True, help="path to a run's res.pt")
    p.add_argument("--out", required=True, help="output directory for the figures")
    p.add_argument("--substeps", type=int, default=1,
                   help="1 = deform to each frame; N>1 = N sub-steps per ODE interval "
                        "(e.g. 10 -> time steps of 0.1)")
    p.add_argument("--source", choices=["gt", "recon"], default="gt",
                   help="deform the GT image (default) or the reconstructed image")
    p.add_argument("--frame", type=int, default=0, help="source/start frame index")
    p.add_argument("--mode", choices=["propagate", "stepwise"], default="stepwise",
                   help="propagate = chain the running warped image through the cycle; "
                        "stepwise = warp each real GT frame once (GT0, warp(GT0), "
                        "warp(GT1), ...), re-anchored to GT every step")
    p.add_argument("--device", default="cpu")
    p.add_argument("--quiver-mag", type=float, default=5.0,
                   help="magnify displacement arrows for visibility")
    a = p.parse_args(argv)
    visualise(a.res, a.out, a.substeps, a.source, a.frame, a.device, a.quiver_mag, a.mode)


if __name__ == "__main__":
    main()
