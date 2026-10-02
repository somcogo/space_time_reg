# %% [markdown]
# # nmAPG initial reconstruction, condensed
# Everything the pipeline's init solve executes, in one file and in execution order:
# data -> operators -> regularizer -> objective -> nmAPG -> evaluation.
# Mirrors: stmr/data/data_utils.py (_cmr_load, generate_standard_mask, complex_abs),
# stmr/data/fft_utils.py, stmr/data/data_load.py (prepare_data), stmr/losses/recon_reg.py,
# stmr/regularizers/{wcrr,wrapper}.py, stmr/data/recon_init.py, stmr/data/nmapg.py,
# stmr/metrics/metric_utils.py (add_cmr_eval_metrics).
# No stmr import until the last cell, which checks this file is bit-identical to the pipeline.
# `# NOTE:` marks behaviour worth scrutinising.

# %% 0. Config -- every knob explicit (the pipeline falls back to Config defaults silently)
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as nnf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

C = dict(
    patient="001", slice=0, start_frame=0, T=2,   # data
    factor=4,                                     # mask: 24 centre rows + every factor-th row
    variant="wcrr",                               # 'crr' (weak_cvx 0) | 'wcrr' (weak_cvx 1)
    lam=0.02,                                     # lambda_init_recon
    scale=5.5,                                    # recon_scale (overrides pretrained s)
    alpha=1.0,                                    # reg_alpha   (overrides pretrained alpha)
    abs_reg=True,                                 # init_reg_abs: regularise |x| (True) or re/im
    lam_st=1.0,                                   # lambda_st (data weight; also L_init)
    max_iter=150,                                 # recon_epochs
    tol=1e-4,                                     # nmAPG stop: max_t ||x_k - x_{k-1}|| / ||x_k|| < tol
    device="cuda",                                # pick a GPU with no compute processes listed
    equiv_iters=50,                               # iterations for the pipeline-equivalence cell
)
C.update(json.loads(os.environ.get("INIT_CFG", "{}")))   # optional override for script runs
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "weights").is_dir())
print(C)

# %% 1. Data -- k-space, mask, measurements y, ground truth, zero-filled init
def load_data(C, device):
    fn = ROOT / f"data/processed/cmrxrecon/test/training_p{C['patient']}_single_coil_full_cine_sax_norm.pt"
    k_gt = torch.load(fn)[:, C["slice"]].permute(0, 3, 1, 2)                   # [frames,2,H,W]
    k_gt = k_gt[C["start_frame"]:C["start_frame"] + C["T"]].to(torch.get_default_dtype())
    H = k_gt.shape[-2]
    rows = list(set(range(H // 2 - 12, H // 2 + 12)) | set(range(0, H, C["factor"])))
    mask = torch.zeros(k_gt.shape, dtype=bool)
    mask[..., rows, :] = 1                                   # same rows for every frame and channel
    y = torch.zeros_like(k_gt)
    y[mask] = k_gt[mask]                                     # measured, zero-filled; noiseless
    y = y.to(device)
    gt = IFT(k_gt).to(device)                                # NOTE: GT FFT done on CPU, as in prepare_data
    mask = mask.to(device)
    x0 = IFT(y * mask)                                       # init='adj': zero-filled adjoint
    return k_gt, mask, y, gt, x0

# %% 2. Operators -- centred, unitary (ortho) 2D FFT over (H, W); tensors are [T,2,H,W] real
def FT(x):
    z = torch.view_as_complex(torch.fft.ifftshift(x.movedim(1, -1), dim=(-3, -2)).contiguous())
    return torch.fft.fftshift(torch.view_as_real(torch.fft.fftn(z, dim=(-2, -1), norm="ortho")), dim=(-3, -2)).movedim(-1, 1)

def IFT(k):
    z = torch.view_as_complex(torch.fft.ifftshift(k.movedim(1, -1), dim=(-3, -2)).contiguous())
    return torch.fft.fftshift(torch.view_as_real(torch.fft.ifftn(z, dim=(-2, -1), norm="ortho")), dim=(-3, -2)).movedim(-1, 1)

# %% 3. Regularizer -- WCRR rebuilt from the raw pretrained state dict, plus the alpha/scale wrapper
def load_reg(C, device):
    W = torch.load(ROOT / f"weights/bilevel_CT/{C['variant'].upper()}_bilevel_JFB_for_CT.pt", map_location=device)
    W = {k: v.to(torch.get_default_dtype()) for k, v in W.items()}   # no-op in float32 (stored dtype)
    w0_raw = W["regularizer.filters.0.parametrizations.weight.original"]    # [4,1,5,5]
    w1, w2 = W["regularizer.filters.1.weight"], W["regularizer.filters.2.weight"]   # [8,4,5,5], [64,8,5,5]
    scaling, beta = W["regularizer.scaling"], W["regularizer.beta"]         # [1,64,1,1], scalar
    wcvx = {"crr": 0.0, "wcrr": 1.0}[C["variant"]]
    # NOTE: alpha and scale REPLACE the pretrained values (WCRR: alpha 8.25, s 2.28; CRR: 7.36, 2.26).
    alpha = torch.tensor(C["alpha"], device=device) * torch.ones_like(W["alpha"])
    scale = torch.tensor(C["scale"], device=device) * torch.ones_like(W["scale"])
    dirac = torch.zeros(1, 1, 25, 25, device=device)
    dirac[0, 0, 12, 12] = 1.0                                # 25 = 2*(5+5+5-3+1)-1

    def filters():
        return [w0_raw - torch.mean(w0_raw, dim=(1, 2, 3), keepdim=True), w1, w2]   # ZeroMean on layer 0

    def conv(x):                                             # K x, normalised so ||K||_2 = 1
        ws = filters()
        h = dirac
        for w in ws:
            h = nnf.conv2d(h, w, padding=2)
        for w in reversed(ws):
            h = nnf.conv_transpose2d(h, w, padding=2)
        lip = torch.fft.fft2(h, s=[256, 256]).abs().max()    # spectral norm of K^T K from its impulse response
        x = x / torch.sqrt(lip)
        for w in ws:
            x = nnf.conv2d(x, w, padding=2)                  # NOTE: zero padding -> image borders act as edges
        return x

    def sl1(z):                                              # smoothed l1 (Huber-like, threshold 1)
        return torch.clip(z**2, 0.0, 1.0) / 2 + torch.clip(torch.abs(z), 1.0) - 1.0

    def R(x):                                                # WCRR.g: energy per image, [B]
        z = conv(x) * torch.exp(scaling)
        z = sl1(torch.exp(beta) * z) * torch.exp(-beta) - sl1(z) * wcvx
        z = z * torch.exp(-2 * scaling)
        return z.sum(dim=(1, 2, 3))

    def Rt(x):                                               # ParameterLearningWrapper.g
        return torch.exp(alpha - 2 * scale) * R(torch.exp(scale) * x)
    return Rt

# %% 4. Objective -- per-frame energy E_t(x) = lam_st * 0.5*sum|M F x - y|^2 + lam * Rt(input), shape [T]
def make_objective(C, Rt):
    def data_fit(x, yp):                                     # yp = cat([y, mask]) so it survives nmAPG's y[idx]
        k, m = yp[:, :2], yp[:, 2:]
        d = m * FT(x) - k
        return (0.5 * (d.abs() ** 2).sum((1, 2, 3))).reshape(-1)   # NOTE: a sum over pixels, not a mean

    def reg(x):
        if C["abs_reg"]:
            # NOTE: complex_abs adds eps=1e-6 INSIDE the sqrt -> |x| >= 1e-3, but median |gt| ~ 7e-4.
            mag = ((x**2).sum(dim=1, keepdim=True) + 1e-6).sqrt()          # [T,1,H,W]
            return (C["lam"] * Rt(mag)).reshape(-1)
        # re and im regularised as two independent real images (prior was trained on >=0 CT)
        return (C["lam"] * Rt(x.flatten(0, 1).unsqueeze(1)).reshape(x.shape[0], -1).sum(1)).reshape(-1)

    def energy(x, yp):
        return (C["lam_st"] * data_fit(x, yp) + reg(x)).detach().reshape(-1)   # detach_grads=True

    def energy_grad(x, yp):
        xr = x.detach().clone().requires_grad_(True)
        (C["lam_st"] * data_fit(xr, yp) + reg(xr)).sum().backward()   # frames are independent -> per-frame grads
        return xr.grad
    return data_fit, reg, energy, energy_grad

# %% 5. nmAPG -- line-for-line copy of stmr/data/nmapg.py with the logging removed
# Li & Lin, NeurIPS 2015, Alg. 4 (supplement). Each frame t is a batch item with its own L_t, and
# frames whose relative change drops below tol are removed from `idx` (they stop being updated).
def nmapg(x0, y, f, nabla, max_iter, L_init, tol, rho=0.9, delta=0.1, eta=0.8, track=None):
    f_and_nabla = lambda v, yy: (f(v, yy), nabla(v, yy))
    x = x0.clone(); x_old = x.clone(); z = x0.clone()        # x1, x0, z1
    t, t_old, q = 1.0, 0.0, 1.0
    c = f(x, y)                                              # c1: nonmonotone reference energy, [T]
    L = torch.full((x.shape[0], 1, 1, 1), L_init, dtype=torch.float32, device=x.device)
    res = (tol + 1) * torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    idx = torch.arange(0, x.shape[0], device=x.device)      # active frames
    grad = torch.zeros_like(x); x_bar = torch.zeros_like(x)
    x_bar_old = x_bar.clone(); grad_old = grad.clone()
    if track: track(-1, x, res, L, idx)
    for i in range(max_iter):
        assert not torch.any(torch.isnan(x)), "NaN in x"
        x_bar[idx] = x[idx] + t_old / t * (z[idx] - x[idx]) + (t_old - 1) / t * (x[idx] - x_old[idx])  # Eq 148
        x_old.copy_(x)
        energy, grad[idx] = f_and_nabla(x_bar[idx], y[idx])
        if i > 0:                                            # Barzilai-Borwein step: L = |r|^2 / |<r, s>|
            dx = grad[idx] - grad_old[idx]
            s = (dx * dx).sum((1, 2, 3), keepdim=True)
            L[idx] = torch.clip(s / (dx * (x_bar[idx] - x_bar_old[idx])).sum((1, 2, 3), keepdim=True).abs().clip(min=1e-12, max=None),
                                min=1.0, max=None)          # NOTE: L >= 1, so the step 1/L is <= 1
        # line search on z (Eq 151-152): shrink the step until sufficient decrease vs max(E(x_bar), c)
        idx_search = idx
        idx_sub = torch.arange(0, idx.shape[0], device=x.device)
        energy_new = energy.clone()
        dx = z[idx] - x_bar[idx]
        for ii in range(150):
            z[idx_search] = x_bar[idx_search] - grad[idx_search] / L[idx_search]   # Eq 151
            dx[idx_sub] = z[idx_search] - x_bar[idx_search]
            bound = torch.max(energy[idx_sub, None, None, None], c[idx_search, None, None, None]) \
                - delta * (dx[idx_sub] * dx[idx_sub]).sum((1, 2, 3), keepdim=True)
            if torch.all((energy_new_ := f(z[idx_search], y[idx_search])) <= bound.view(-1)):
                energy_new[idx_sub] = energy_new_
                break
            energy_new[idx_sub] = energy_new_
            idx_sub = idx_sub[energy_new_ > bound.view(-1)]
            idx_search = idx[idx_sub]
            L[idx_search] = L[idx_search] / rho
        # Eq 153-158: where z did not decrease enough vs c, also try a plain gradient step v from x
        idx2 = (energy_new[:] >= (c[idx] - delta * (dx * dx).sum((1, 2, 3)))).nonzero().view(-1)
        if idx2.nelement() > 0:
            idx_idx2 = idx[idx2]
            gradx = nabla(x[idx_idx2], y[idx_idx2])
            if i > 0:
                dx = gradx - grad_old[idx_idx2]
                s = (dx * dx).sum((1, 2, 3), keepdim=True)
                L[idx_idx2] = torch.clip(s / (dx * (x[idx_idx2] - x_bar_old[idx_idx2])).sum((1, 2, 3), keepdim=True).abs().clip(min=1e-12, max=None),
                                         min=1.0, max=None)
            for ii in range(150):
                v = x[idx_idx2] - gradx / L[idx_idx2]
                dx = v - x[idx_idx2]
                bound = c[idx_idx2, None, None, None] - delta * (dx * dx).sum((1, 2, 3), keepdim=True)
                if torch.all((energy_new2 := f(v, y[idx_idx2])) <= bound.view(-1) * (1 + 1e-4)):
                    break
                L[idx_idx2] = torch.where(energy_new2[:, None, None, None] <= bound, L[idx_idx2], L[idx_idx2] / rho)
            x[idx] = z[idx]
            idx3 = (energy_new2 <= energy_new[idx2]).nonzero().view(-1)
            x[idx_idx2[idx3]] = v[idx3]                      # keep whichever of z, v has lower energy
        else:
            x[idx] = z[idx]
        if i > 0:
            res[idx] = torch.norm(x[idx] - x_old[idx], p=2, dim=(1, 2, 3)) / torch.norm(x[idx], p=2, dim=(1, 2, 3))
        assert not torch.any(torch.isnan(res)), "NaN in res"
        idx = (res >= tol).nonzero().view(-1)                # NOTE: this relative-change test is the engagement cliff
        if track: track(i, x, res, L, idx)
        if torch.max(res) < tol:
            break
        t_old = t
        t = (np.sqrt(4.0 * t_old**2 + 1.0) + 1.0) / 2.0     # Eq 159
        q_old = q
        q = eta * q + 1.0                                    # Eq 160
        c[idx] = (eta * q_old * c[idx] + f(x[idx], y[idx])) / q   # Eq 161
        x_bar_old.copy_(x_bar)
        grad_old.copy_(grad)
    return x, L, i, res < tol

# %% Run
torch.manual_seed(0)
k_gt, mask, y, gt, x0 = load_data(C, C["device"])
Rt = load_reg(C, C["device"])
data_fit, reg, energy, energy_grad = make_objective(C, Rt)
yp = torch.cat([y, mask.to(y)], dim=1)                       # packed measurements [T,4,H,W]
hist = []
def track(i, x, res, L, idx):
    hist.append(dict(i=i + 1, df=float(C["lam_st"] * data_fit(x, yp).sum()), reg=float(reg(x).sum()),
                     res=float(res.max()), L=float(L.mean()), n_active=int(idx.numel())))
x, L, it, conv = nmapg(x0, yp, energy, energy_grad, C["max_iter"], C["lam_st"], C["tol"], track=track)
print(f"stopped after iteration {it + 1}, converged per frame: {conv.tolist()}, final L: {L.flatten().tolist()}")

# %% 6. Evaluate -- magnitude PSNR/SSIM/NMSE per frame (data_range = gt.max()), full frame + centre crop
def evaluate(xr, gt):
    xm = ((xr**2).sum(1)).sqrt().cpu().numpy()               # fastmri complex_abs: no eps here
    gm = ((gt**2).sum(1)).sqrt().cpu().numpy()
    H, W = gm.shape[-2:]
    out = {}
    for name, sl in (("full", np.s_[:, :, :]), ("crop", np.s_[:, round(H / 3):round(2 * H / 3), round(W / 4):round(3 * W / 4)])):
        p, g = xm[sl], gm[sl]
        out[name] = dict(
            psnr=np.mean([peak_signal_noise_ratio(g[t], p[t], data_range=g[t].max()) for t in range(len(g))]),
            ssim=np.mean([structural_similarity(g[t], p[t], data_range=g[t].max()) for t in range(len(g))]),
            nmse=np.mean([np.linalg.norm(g[t] - p[t]) ** 2 / np.linalg.norm(g[t]) ** 2 for t in range(len(g))]))
    return out

for name, v in (("zero-fill", x0), ("nmAPG", x)):
    e = evaluate(v, gt)
    print(f"{name:9}  full PSNR {e['full']['psnr']:.3f}  SSIM {e['full']['ssim']:.4f}  |  "
          f"crop PSNR {e['crop']['psnr']:.3f}  SSIM {e['crop']['ssim']:.4f}")

# %% 7. Diagnostic plots
h = {k: np.array([d[k] for d in hist]) for k in hist[0]}
fig, ax = plt.subplots(1, 3, figsize=(15, 3.5), layout="constrained")
ax[0].semilogy(h["i"], h["df"] + h["reg"], label="energy"); ax[0].semilogy(h["i"], h["df"], label="data fit")
ax[0].semilogy(h["i"], h["reg"], label="reg"); ax[0].legend(); ax[0].set_xlabel("iteration")
ax[1].semilogy(h["i"][1:], h["res"][1:]); ax[1].axhline(C["tol"], ls="--", c="k"); ax[1].set_title("max relative change (tol dashed)")
ax[2].plot(h["i"], h["L"]); ax[2].set_title("mean L (step = 1/L)")
fig, ax = plt.subplots(1, 4, figsize=(18, 3.5), layout="constrained")
m = lambda v: ((v[0] ** 2).sum(0)).sqrt().cpu()
for a, im, ttl in zip(ax, (m(x), m(gt), (m(x) - m(gt)).abs(), m(x0)), ("nmAPG", "GT", "|nmAPG - GT|", "zero-fill")):
    a.imshow(im, cmap="gray" if ttl != "|nmAPG - GT|" else "magma", vmax=None if ttl == "|nmAPG - GT|" else m(gt).max()); a.set_title(ttl); a.axis("off")
plt.show()

# %% 8. Sanity checks
def rel(a, b):
    return float((a - b).norm() / b.norm())

with torch.no_grad():
    v = torch.randn_like(x0, dtype=torch.float64); kk = torch.randn_like(v)
    print("FFT  | ||Fx||/||x|| - 1          :", float(FT(v).norm() / v.norm() - 1))
    print("FFT  | <Fx,k> - <x,F^H k> (rel)   :", float(((FT(v) * kk).sum() - (v * IFT(kk)).sum()) / (FT(v) * kk).sum().abs()))
    print("FFT  | ||F^H F x - x|| / ||x||    :", rel(IFT(FT(v)), v))
    print("data | y == M*F(gt) (rel err)     :", rel(mask * FT(gt), y), "| y exactly 0 off mask:", bool((y[~mask] == 0).all()))
    off = torch.load(ROOT / f"data/processed/cmrxrecon/test/training_p{C['patient']}_single_coil_acc_04_cine_sax_norm.pt")[:2, C["slice"]].permute(0, 3, 1, 2)
    moff = off != 0
    print(f"mask | ours: {int(mask[0, 0].any(1).sum())}/{mask.shape[-2]} rows (H) sampled, all columns (W) sampled: {bool(mask[0, 0].any(0).all())} "
          f"| official acc_04: {int(moff[0, 0].any(1).sum())}/{moff.shape[-2]} rows sampled, all columns sampled: {bool(moff[0, 0].any(0).all())}")
    print(f"mask | measured fraction ours {float(mask.float().mean()):.3f}, official {float(moff.float().mean()):.3f}, "
          f"same rows: {bool((mask[0, 0].any(1).cpu() == moff[0, 0].any(1)).all())}")
    gm_ = (gt**2).sum(1).sqrt(); ge_ = ((gt**2).sum(1) + 1e-6).sqrt()
    print(f"eps  | frac GT pixels with |x| < 1e-3: {float((gm_ < 1e-3).float().mean()):.2f}, mean bias {float((ge_ - gm_).mean() / gm_.mean()):+.1%}, "
          f"median gradient damping |x|/|x|_eps {float((gm_ / ge_).median()):.3f}")

# autograd vs central finite differences in float64, for the data term and the reg term separately,
# evaluated at the solver's output (not at x0, where the data-term gradient is exactly zero)
torch.set_default_dtype(torch.float64)
_, mask64, y64, _, _ = load_data(C, "cpu")
df64, reg64, _, _ = make_objective(C, load_reg(C, "cpu"))
torch.set_default_dtype(torch.float32)
yp64, x64 = torch.cat([y64, mask64.to(y64)], dim=1), x.detach().cpu().double()
g = torch.Generator().manual_seed(0)
for name, fn in (("data", lambda z: C["lam_st"] * df64(z, yp64)), ("reg ", reg64)):
    xg = x64.clone().requires_grad_(True)
    fn(xg).sum().backward()
    for trial in range(3):
        dirn = torch.randn(x64.shape, generator=g, dtype=torch.float64)
        dirn = dirn / dirn.norm() * x64.norm()
        hh = 1e-5
        fd = float((fn(x64 + hh * dirn).sum() - fn(x64 - hh * dirn).sum()) / (2 * hh))
        ad = float((xg.grad * dirn).sum())
        print(f"grad | {name} dir {trial}: autograd {ad:+.6e}  finite-diff {fd:+.6e}  rel diff {abs(ad - fd) / max(abs(fd), 1e-300):.1e}")

# %% 9. Equivalence vs the pipeline (CPU, deterministic). Imports stmr LAST: it switches matplotlib to Agg.
from stmr.config import Config                              # noqa: E402
from stmr.data.data_load import prepare_data                # noqa: E402
from stmr.data.recon_init import init_using_nmAPG           # noqa: E402

cfg = Config()
for k, val in dict(dataset=f"cmr_P{C['patient']}", slice_number=C["slice"], start_frame=C["start_frame"],
                   time_points=C["T"], factor=C["factor"], mask="st", template_frames=0, kspace_noise_sigma=0.0,
                   init="adj", use_nmapg=True, reg="learned", reg_variant=C["variant"], reg_alpha=C["alpha"],
                   recon_scale=C["scale"], lambda_init_recon=C["lam"], init_reg_abs=C["abs_reg"],
                   lambda_st=C["lam_st"], init_loss="l2", detach_grads=True, tol=C["tol"],
                   recon_epochs=C["equiv_iters"], debug=False, device="cpu").items():
    setattr(cfg, k, val)
d = prepare_data(cfg)
packed = torch.cat([d.fixed, d.kspace_mask.to(d.fixed)], dim=1)
x_pipe, _ = init_using_nmAPG(cfg, d.init, packed, d.full_forw, d.forw_subs_adj, logger=None)

k_c, mask_c, y_c, gt_c, x0_c = load_data(C, "cpu")
_, _, e_c, g_c = make_objective(C, load_reg(C, "cpu"))
x_c, L_c, it_c, _ = nmapg(x0_c, torch.cat([y_c, mask_c.to(y_c)], dim=1), e_c, g_c, C["equiv_iters"], C["lam_st"], C["tol"])
for name, a, b in (("measurements y", y_c, d.fixed), ("mask", mask_c, d.kspace_mask), ("GT image", gt_c, d.gt_im),
                   ("zero-fill init", x0_c, d.init.detach()), (f"recon after {it_c + 1} iters", x_c, x_pipe.detach())):
    print(f"equiv | {name:22}: bit-identical {torch.equal(a, b)}   max|diff| {float((a.float() - b.float()).abs().max()):.2e}")
print("equiv | pipeline x requires_grad:", x_pipe.requires_grad, "grad_fn:", type(x_pipe.grad_fn).__name__)
