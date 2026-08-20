"""Two-stage sequential training (+ single-net baselines) for motion decomposition.

Registration only: images are fixed GT magnitudes, only the SIREN velocity nets are optimised.
Structure follows stmr/robustness/synthetic.py::fit_siren_to_flow, run twice with a stage
hand-off (stage 2 registers the frozen-phi1-warped frames to the next frames).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from stmr.losses.losses import grad_phi_loss, log_detJ_loss
from stmr.robustness.jacobian import detJ_from_phi
from stmr.state import ModelOutputs
from stmr.utils.logging import get_logger
from stmr.utils.spatial_utils import generate_coord_tensor

from .metrics import compute_responsibility, disp_px, frame_match_psnr
from .model import build_net, compose_fields, make_integrator, warp
from .params import DecompParams


def _set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed + 1)
    np.random.seed(seed + 2)


def _reg(abs_phi, coord, H, W, lam_grad, lam_logdetJ):
    """Deformation-regularity penalty (grad_phi + log_detJ) built on a bare ModelOutputs."""
    if not lam_grad and not lam_logdetJ:
        return abs_phi.new_zeros(()), abs_phi.new_zeros(()), abs_phi.new_zeros(())
    mo = ModelOutputs(rel_vel=None, abs_phi=abs_phi, transformer=None)
    shape = [-1, H, W, 2]
    g = grad_phi_loss(mo, coord, shape)[0].mean() if lam_grad else abs_phi.new_zeros(())
    d = log_detJ_loss(mo, coord, shape)[0].mean() if lam_logdetJ else abs_phi.new_zeros(())
    return lam_grad * g + lam_logdetJ * d, g.detach(), d.detach()


def _effective_phi(abs_phi, pre_phi, H, W):
    """The deformation actually applied to the source: phi itself, or pre_phi o phi (fields
    composed) when a frozen earlier stage pre_phi is supplied -- a single interpolation."""
    return abs_phi if pre_phi is None else compose_fields(pre_phi, abs_phi, H, W)


def _log_step(writer, prefix, step, loss, data, g, d, abs_phi, eff_phi,
              source, target, coord, H, W):
    if writer is None:
        return
    with torch.no_grad():
        moved = warp(eff_phi, source, H, W)
        base = torch.nn.functional.mse_loss(source, target)
        detJ = detJ_from_phi(abs_phi, H, W)
        scalars = {
            f"{prefix}total_loss": float(loss),
            f"{prefix}data_loss": float(data),
            f"{prefix}reg/grad_phi": float(g),
            f"{prefix}reg/logdetJ": float(d),
            f"{prefix}residual/base": float(base),
            f"{prefix}residual/warp": float((moved - target).pow(2).mean()),
            f"{prefix}psnr_frame_match": frame_match_psnr(moved, target),
            f"{prefix}disp/mean_px": float(disp_px(abs_phi, coord, H, W).mean()),
            f"{prefix}detJ_min": float(detJ.min()),
            f"{prefix}foldover_frac": float((detJ <= 0).float().mean()),
        }
    for k, v in scalars.items():
        writer.add_scalar(k, v, step)


def _fit(net, integrate, coord, source_mag, target_mag, H, W, steps, lr,
         lam_grad, lam_logdetJ, writer=None, prefix="", log_every=50, pre_phi=None):
    """Train one net's velocity field; returns the final abs_phi (detached).

    ``pre_phi`` (a frozen earlier-stage deformation) enables field composition: the data loss
    warps ``source_mag`` by ``pre_phi o abs_phi`` (single interpolation), while the regularity
    penalty stays on this stage's own field ``abs_phi``. source_mag is always the ORIGINAL frame.
    """
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for step in range(steps):
        opt.zero_grad()
        abs_phi = integrate()
        eff_phi = _effective_phi(abs_phi, pre_phi, H, W)
        data = torch.nn.functional.mse_loss(warp(eff_phi, source_mag, H, W), target_mag)
        reg, g, d = _reg(abs_phi, coord, H, W, lam_grad, lam_logdetJ)
        loss = data + reg
        loss.backward()
        opt.step()
        if writer is not None and (step % log_every == 0 or step == steps - 1):
            _log_step(writer, prefix, step, loss, data, g, d, abs_phi, eff_phi,
                      source_mag, target_mag, coord, H, W)
    with torch.no_grad():
        return integrate().detach()


# --------------------------------------------------------------------------- #
# Hermetic core (used by the CPU test) -- fixed tensors, no data load / writer
# --------------------------------------------------------------------------- #
def run_two_stage_core(src, tgt, H, W, layers1=(2, 32, 32, 2), layers2=(2, 64, 64, 2),
                       steps1=150, steps2=150, lr=1e-3, step_size=0.2, device="cpu",
                       lam_grad1=1.0, lam_logdetJ1=1.0):
    """Two-stage fit on a single frame pair src->tgt (groups=1). Returns
    {base, r1, r2, m1_px, m2_px}. src/tgt: [1,1,H,W]."""
    coord = generate_coord_tensor((H, W), device)
    net1 = build_net(list(layers1), 1, 30.0, device)
    integ1 = make_integrator(net1, coord, 1, "euler", step_size)
    abs_phi1 = _fit(net1, integ1, coord, src, tgt, H, W, steps1, lr, lam_grad1, lam_logdetJ1)

    net1.eval()
    net2 = build_net(list(layers2), 1, 30.0, device)
    integ2 = make_integrator(net2, coord, 1, "euler", step_size)
    # stage 2 via field composition: warp the ORIGINAL source by phi1 o phi2 (single interp)
    abs_phi2 = _fit(net2, integ2, coord, src, tgt, H, W, steps2, lr, 0.0, 0.0,
                    pre_phi=abs_phi1)

    with torch.no_grad():
        comp = compose_fields(abs_phi1, abs_phi2, H, W)
        base = float((src - tgt).pow(2).mean())
        r1 = float((warp(abs_phi1, src, H, W) - tgt).pow(2).mean())
        r2 = float((warp(comp, src, H, W) - tgt).pow(2).mean())
        m1 = float(disp_px(abs_phi1, coord, H, W).mean())
        m2 = float(disp_px(abs_phi2, coord, H, W).mean())
    return {"base": base, "r1": r1, "r2": r2, "m1_px": m1, "m2_px": m2}


# --------------------------------------------------------------------------- #
# Full experiment
# --------------------------------------------------------------------------- #
def run_decomp(p: DecompParams) -> dict:
    """Run stage 1 -> stage 2 (+ baselines), compute responsibility, save res.pt + figures."""
    from .data import load_gt

    logger = get_logger()
    _set_seed(p.seed)
    out_dir = os.path.join(p.log_path, p.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(out_dir, "tensorboard"))

    gt_im, mag, H, W = load_gt(p)
    dev = p.device
    coord = generate_coord_tensor((H, W), dev)
    src, tgt = mag[:-1], mag[1:]
    logger.info(f"decomp: {p.dataset} s{p.slice_number} T={p.time_points} "
                f"({p.groups} intervals) {H}x{W} on {dev}")

    def _net(layers):
        return build_net(list(layers), p.groups, p.omega, dev)

    def _integ(net):
        return make_integrator(net, coord, p.groups, p.solver, p.step_size)

    # --- stage 1: small, regular phi1 ---
    net1 = _net(p.phi1_layers)
    abs_phi1 = _fit(net1, _integ(net1), coord, src, tgt, H, W, p.steps1, p.lr1,
                    p.lam_grad_phi1, p.lam_logdetJ1, writer, "stage1/", p.log_every)

    # --- stage 2: freeze phi1, train larger phi2 on the residual via field composition
    #     (warp the ORIGINAL source by phi1 o phi2 -> single interpolation) ---
    net1.eval()
    net2 = _net(p.phi2_layers)
    abs_phi2 = _fit(net2, _integ(net2), coord, src, tgt, H, W, p.steps2, p.lr2,
                    p.lam_grad_phi2, p.lam_logdetJ2, writer, "stage2/", p.log_every,
                    pre_phi=abs_phi1.detach())

    # --- baselines: single-net large-alone / small-alone ---
    abs_phi_bl = abs_phi_bs = None
    st_bl = st_bs = None
    if p.run_baselines:
        net_bl = _net(p.phi2_layers)
        abs_phi_bl = _fit(net_bl, _integ(net_bl), coord, src, tgt, H, W,
                          p.steps_base_large, p.lr_base_large, p.lam_grad_base_large,
                          p.lam_logdetJ_base_large, writer, "base_large/", p.log_every)
        st_bl = net_bl.state_dict()
        net_bs = _net(p.phi1_layers)
        abs_phi_bs = _fit(net_bs, _integ(net_bs), coord, src, tgt, H, W,
                          p.steps_base_small, p.lr_base_small, p.lam_grad_base_small,
                          p.lam_logdetJ_base_small, writer, "base_small/", p.log_every)
        st_bs = net_bs.state_dict()

    responsibility = compute_responsibility(abs_phi1, abs_phi2, mag, gt_im, coord, H, W,
                                            abs_phi_bl, abs_phi_bs)
    _log_responsibility(writer, responsibility)

    res = {
        "st_dict1": net1.state_dict(), "st_dict2": net2.state_dict(),
        "st_dict_base_large": st_bl, "st_dict_base_small": st_bs,
        "abs_phi1": abs_phi1.cpu(), "abs_phi2": abs_phi2.cpu(),
        "abs_phi_base_large": None if abs_phi_bl is None else abs_phi_bl.cpu(),
        "abs_phi_base_small": None if abs_phi_bs is None else abs_phi_bs.cpu(),
        "coord": coord.cpu(), "gt_im": gt_im.cpu(), "mag": mag.cpu(),
        "H": H, "W": W, "params": p.to_dict(), "responsibility": responsibility,
    }
    torch.save(res, os.path.join(out_dir, "res.pt"))
    writer.flush(); writer.close()

    from .figures import make_figures
    make_figures(res, os.path.join(out_dir, "figures"))
    _print_summary(logger, responsibility)
    logger.info(f"decomp done -> {out_dir}")
    return res


def _log_responsibility(writer, r):
    ds, rs = r["disp_split"], r["resid_split"]
    for k in ("m1_px", "m2_px", "frac1", "frac2"):
        writer.add_scalar(f"responsibility/disp_{k}", ds[k], 0)
    for k in ("explained_phi1", "explained_phi2", "unexplained", "r_base_large", "r_base_small"):
        if k in rs:
            writer.add_scalar(f"responsibility/resid_{k}", rs[k], 0)


def _print_summary(logger, r):
    ds, rs = r["disp_split"], r["resid_split"]
    logger.info(f"  displacement split: phi1 {ds['frac1']*100:.1f}% ({ds['m1_px']:.3f}px) | "
                f"phi2 {ds['frac2']*100:.1f}% ({ds['m2_px']:.3f}px)")
    logger.info(f"  residual explained: phi1 {rs['explained_phi1']*100:.1f}% | "
                f"phi2 {rs['explained_phi2']*100:.1f}% | unexplained {rs['unexplained']*100:.1f}%")
    if "r_base_large" in rs:
        logger.info(f"  capacity: two-stage r2={rs['r2']:.3e} | large-alone={rs['r_base_large']:.3e} "
                    f"| small-alone={rs.get('r_base_small', float('nan')):.3e}")
