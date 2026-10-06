# %% [markdown]
# # Checks for init_condensed.py
# Evaluation, diagnostic plots, sanity checks and the equivalence test against the pipeline
# (sections 6-9, moved out of init_condensed.py). Works on what init_condensed.py leaves behind:
# C, ROOT, FT, IFT, load_data, load_reg, make_objective, mask, y, yp, gt, x0, x, hist.

# %% Imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# %% Run the solve -- skip this cell if init_condensed.py was already run in this interactive session
from init_condensed import *  # noqa: E402,F403

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
df64, reg64, *_ = make_objective(C, *load_reg(C, "cpu"))
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

# closed-form gradient vs autograd, for both regularizer inputs (independent of which one C["grad"] selects)
for abs_reg in (True, False):
    out = {}
    for mode in ("autograd", "analytic"):
        Cm = {**C, "abs_reg": abs_reg, "grad": mode}
        out[mode] = make_objective(Cm, *load_reg(Cm, x.device))[4](x.detach(), yp)
    (Ea, Ga), (Eb, Gb) = out["autograd"], out["analytic"]
    print(f"grad | analytic vs autograd, abs_reg={abs_reg}: energy equal {torch.equal(Ea, Eb)}, gradient rel diff {rel(Gb, Ga):.1e}")

# %% 9. Equivalence vs the pipeline (CPU, deterministic)
# Both sides use the same solver (stmr/data/nmapg.py), so this checks the condensed data, FFT, regularizer
# and objective against the pipeline's prepare_data + init_using_nmAPG.
# Keep this cell last: these stmr imports can switch matplotlib back to the file-only Agg backend.
from stmr.config import Config                              # noqa: E402
from stmr.data.data_load import prepare_data                # noqa: E402
from stmr.data.nmapg import nmAPG                           # noqa: E402
from stmr.data.recon_init import init_using_nmAPG           # noqa: E402

cfg = Config()
for k, val in dict(dataset=f"cmr_P{C['patient']}", slice_number=C["slice"], start_frame=C["start_frame"],
                   time_points=C["T"], factor=C["factor"], mask="st", template_frames=0, kspace_noise_sigma=0.0,
                   init="adj", use_nmapg=True, reg="learned", reg_variant=C["variant"], reg_alpha=C["alpha"],
                   recon_scale=C["scale"], lambda_init_recon=C["lam"], init_reg_abs=C["abs_reg"], init_grad=C["grad"],
                   lambda_st=C["lam_st"], L_init=C["L_init"], init_loss="l2", detach_grads=True, tol=C["tol"],
                   recon_epochs=C["equiv_iters"], debug=False, device="cpu").items():
    setattr(cfg, k, val)
d = prepare_data(cfg)
packed = torch.cat([d.fixed, d.kspace_mask.to(d.fixed)], dim=1)
x_pipe, _ = init_using_nmAPG(cfg, d.init, packed, d.full_forw, d.full_adj, logger=None)

k_c, mask_c, y_c, gt_c, x0_c = load_data(C, "cpu")
_, _, e_c, g_c, eg_c = make_objective(C, *load_reg(C, "cpu"))
x_c, L_c, it_c, _, _ = nmAPG(x0=x0_c, y=torch.cat([y_c, mask_c.to(y_c)], dim=1), f=e_c, nabla=g_c, f_and_nabla=eg_c,
                             max_iter=C["equiv_iters"], L_init=C["L_init"], tol=C["tol"])
for name, a, b in (("measurements y", y_c, d.fixed), ("mask", mask_c, d.kspace_mask), ("GT image", gt_c, d.gt_im),
                   ("zero-fill init", x0_c, d.init.detach()), (f"recon after {it_c + 1} iters", x_c, x_pipe.detach())):
    print(f"equiv | {name:22}: bit-identical {torch.equal(a, b)}   max|diff| {float((a.float() - b.float()).abs().max()):.2e}")
print("equiv | pipeline x requires_grad:", x_pipe.requires_grad, "grad_fn:", type(x_pipe.grad_fn).__name__)
