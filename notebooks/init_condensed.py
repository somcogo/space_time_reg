# %% [markdown]
# # nmAPG initial reconstruction, condensed
# The pipeline's init solve in execution order: data -> operators -> regularizer -> objective -> solve.
# Mirrors stmr/data/{data_utils,fft_utils,data_load,recon_init}.py, stmr/losses/recon_reg.py and
# stmr/regularizers/{wcrr,wrapper}.py; the solver is the pipeline's own stmr/data/nmapg.py.
# Checks and plots are in init_condensed_checks.py. `# NOTE:` marks behaviour worth scrutinising.

# %% 0. Config
from pathlib import Path

import matplotlib
import torch
import torch.nn.functional as nnf

_backend = matplotlib.get_backend()
from stmr.data.nmapg import nmAPG                 # noqa: E402  the pipeline's solver
matplotlib.use(_backend)                          # importing stmr switches matplotlib to Agg; switch back so inline plots show

C = dict(
    patient="002", slice=3, start_frame=0, T=12,   # data
    factor=4,                                     # mask: 24 centre rows + every factor-th row
    variant="crr",                               # 'crr' (weak_cvx 0) | 'wcrr' (weak_cvx 1)
    lam=0.2,                                       # lambda_recon
    scale=3.55,                                    # recon_scale (overrides pretrained s)
    alpha=1.0,                                    # reg_alpha   (overrides pretrained alpha)
    abs_reg=False,                                 # init_reg_abs: regularise |x| (True) or re/im
    grad="analytic",                              # init_grad: 'autograd' | 'analytic' (closed-form gradient)
    lam_st=1.0,                                   # lambda_st (data weight)
    L_init=0.01,                                   # nmAPG's initial L (first step = 1/L_init)
    max_iter=1000,                                # recon_epochs
    tol=1e-4,                                      # nmAPG stop: max_t ||x_k - x_{k-1}|| / ||x_k|| < tol; 0 = never stop early
    device="cuda",                                # pick a GPU with no compute processes listed
)
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "weights").is_dir())
print(C)

#  1. Operators -- centred, unitary (ortho) 2D FFT over (H, W); tensors are [T,2,H,W] real
def _fft(x, fft2):
    z = torch.fft.ifftshift(torch.view_as_complex(x.movedim(1, -1).contiguous()), dim=(-2, -1))
    return torch.view_as_real(torch.fft.fftshift(fft2(z, norm="ortho"), dim=(-2, -1))).movedim(-1, 1)

def FT(x):
    return _fft(x, torch.fft.fft2)

def IFT(k):
    return _fft(k, torch.fft.ifft2)

#  2. Data -- k-space, mask, measurements y, ground truth, initial point
def load_data(C, device):
    fn = ROOT / f"data/processed/cmrxrecon/test/training_p{C['patient']}_single_coil_full_cine_sax_norm.pt"
    k_gt = torch.load(fn)[:, C["slice"]].permute(0, 3, 1, 2)                   # [frames,2,H,W]
    k_gt = k_gt[C["start_frame"]:C["start_frame"] + C["T"]].to(torch.get_default_dtype())

    gt = IFT(k_gt).to(device)                                # NOTE: GT FFT done on CPU, as in prepare_data
    gt_scale = gt.abs().max()-gt.abs().min()

    k_gt = k_gt / gt_scale.to(k_gt.device)



    H = k_gt.shape[-2]
    mask = torch.zeros(k_gt.shape, dtype=bool)               # same rows for every frame and channel
    mask[..., H // 2 - 12:H // 2 + 12, :] = True
    mask[..., ::C["factor"], :] = True
    y = (k_gt * mask).to(device)                             # measured, zero-filled; noiseless
    gt = IFT(k_gt).to(device)                                # NOTE: GT FFT done on CPU, as in prepare_data
    x0 = IFT(y)                                 # zero init; the pipeline's init='adj' is IFT(y)
    return k_gt, mask.to(device), y, gt, x0

#  3. Regularizer -- WCRR rebuilt from the raw pretrained state dict, plus the alpha/scale wrapper
def load_reg(C, device):
    W = torch.load(ROOT / f"weights/bilevel_CT/{C['variant'].upper()}_bilevel_JFB_for_CT.pt", map_location=device)
    W = {k: v.to(torch.get_default_dtype()) for k, v in W.items()}   # no-op in float32 (stored dtype)
    w0 = W["regularizer.filters.0.parametrizations.weight.original"]         # [4,1,5,5]
    filters = [w0 - w0.mean(dim=(1, 2, 3), keepdim=True),                    # ZeroMean on layer 0
               W["regularizer.filters.1.weight"], W["regularizer.filters.2.weight"]]   # [8,4,5,5], [64,8,5,5]
    # scaling, beta = W["regularizer.scaling"], W["regularizer.beta"]          # [1,64,1,1], scalar
    scaling, beta = W["regularizer.scaling"], torch.ones_like(W["regularizer.beta"])*3.0          # [1,64,1,1], scalar
    wcvx = {"crr": 0.0, "wcrr": 1.0}[C["variant"]]
    # NOTE: alpha and scale REPLACE the pretrained values (WCRR: alpha 8.25, s 2.28; CRR: 7.36, 2.26).
    alpha, scale = torch.tensor(C["alpha"], device=device), torch.tensor(C["scale"], device=device)

    h = torch.zeros(1, 1, 25, 25, device=device)             # impulse response of K^T K; 25 = 2*(5+5+5-3+1)-1
    h[0, 0, 12, 12] = 1.0
    for w in filters:
        h = nnf.conv2d(h, w, padding=2)
    for w in reversed(filters):
        h = nnf.conv_transpose2d(h, w, padding=2)
    norm = torch.fft.fft2(h, s=[256, 256]).abs().max().sqrt()   # ||K||_2 before normalisation

    def K(x):                                                # scaled filter responses, K normalised so ||K||_2 = 1
        x = x / norm
        for w in filters:
            x = nnf.conv2d(x, w, padding=2)                  # NOTE: zero padding -> image borders act as edges
        return x * torch.exp(scaling)

    def K_T(z):                                              # adjoint of K
        z = z * torch.exp(-scaling) / norm
        for w in reversed(filters):
            z = nnf.conv_transpose2d(z, w, padding=2)
        return z

    def sl1(z):                                              # smoothed l1 (Huber-like, threshold 1)
        return torch.clip(z**2, 0.0, 1.0) / 2 + torch.clip(torch.abs(z), 1.0) - 1.0

    def energy(z):                                           # WCRR.g on z = K(.), with the wrapper's weight: [B]
        e = sl1(torch.exp(beta) * z) * torch.exp(-beta) - sl1(z) * wcvx
        return torch.exp(alpha - 2 * scale) * (e * torch.exp(-2 * scaling)).sum(dim=(1, 2, 3))

    def Rt(x):                                               # ParameterLearningWrapper.g
        return energy(K(torch.exp(scale) * x))

    def Rt_and_grad(x):                                      # ParameterLearningWrapper.grad(get_energy=True)
        z = K(torch.exp(scale) * x)
        g = torch.clip(torch.exp(beta) * z, -1.0, 1.0) - torch.clip(z, -1.0, 1.0) * wcvx   # derivative of sl1 is a clip
        return energy(z), torch.exp(alpha - scale) * K_T(g)
    return Rt, Rt_and_grad

#  4. Objective -- per-frame energy E_t(x) = lam_st * 0.5*sum|M F x - y|^2 + lam * Rt(input), shape [T]
def make_objective(C, Rt, Rt_and_grad):
    def residual(x, yp):                                     # yp = cat([y, mask]) so it survives nmAPG's y[idx]
        return yp[:, 2:] * FT(x) - yp[:, :2]

    def data_fit(x, yp):
        return 0.5 * (residual(x, yp) ** 2).sum((1, 2, 3))   # NOTE: a sum over pixels, not a mean

    def reg_input(x):                                        # what the prior sees
        if C["abs_reg"]:
            # NOTE: complex_abs adds eps=1e-6 INSIDE the sqrt -> |x| >= 1e-3, but median |gt| ~ 7e-4.
            return ((x**2).sum(dim=1, keepdim=True) + 1e-6).sqrt()             # [T,1,H,W]
        return x.flatten(0, 1).unsqueeze(1)                  # re and im as two independent real images, [2T,1,H,W]

    def reg(x):
        return (C["lam"] * Rt(reg_input(x))).reshape(x.shape[0], -1).sum(1)

    def energy(x, yp):
        return (C["lam_st"] * data_fit(x, yp) + reg(x)).detach()

    def energy_and_grad(x, yp):                              # what nmAPG calls once per iteration
        if C["grad"] == "analytic":                          # closed form: same energy as energy(), no autograd
            with torch.no_grad():
                d, u = residual(x, yp), reg_input(x)
                e, g = Rt_and_grad(u)
                g = g * x / u if C["abs_reg"] else g.reshape(x.shape)   # chain rule: d|x|_eps/dx = x / |x|_eps
                E = C["lam_st"] * 0.5 * (d**2).sum((1, 2, 3)) + (C["lam"] * e).reshape(x.shape[0], -1).sum(1)
                return E, C["lam_st"] * IFT(yp[:, 2:] * d) + C["lam"] * g   # data gradient: F^H M (M F x - y)
        xr = x.detach().clone().requires_grad_(True)
        E = C["lam_st"] * data_fit(xr, yp) + reg(xr)
        E.sum().backward()                                   # frames are independent -> per-frame grads
        return E.detach(), xr.grad

    def energy_grad(x, yp):
        return energy_and_grad(x, yp)[1]
    return data_fit, reg, energy, energy_grad, energy_and_grad

#  5. Solve -- the pipeline's nmAPG (stmr/data/nmapg.py)
k_gt, mask, y, gt, x0 = load_data(C, C["device"])
print(x0.abs().max())
data_fit, reg, energy, energy_grad, energy_and_grad = make_objective(C, *load_reg(C, C["device"]))
yp = torch.cat([y, mask.to(y)], dim=1)                       # packed measurements [T,4,H,W]
hist = []                                                    # per-iteration stats; i = 0 is the initial point
x, L, it, conv, _ = nmAPG(x0=x0, y=yp, f=energy, nabla=energy_grad, f_and_nabla=energy_and_grad,
                          max_iter=C["max_iter"], L_init=C["L_init"], tol=C["tol"],
                          data_fit=lambda v, yy: C["lam_st"] * data_fit(v, yy), reg=reg,
                          callback=lambda i, x, s: hist.append(dict(s, i=i + 1, df=s["data_fit"])))
print(f"stopped after iteration {it + 1}, converged per frame: {conv.tolist()}, final L: {L.flatten().tolist()}")

#  7. Diagnostic plots
import matplotlib.pyplot as plt
import numpy as np
print(hist[0].keys())

h = {k: np.array([d[k] for d in hist]) for k in hist[0]}
fig, ax = plt.subplots(1, 4, figsize=(20, 3.5), layout="constrained")
ax[0].semilogy(h["i"], h["df"] + h["reg"], label="energy"); ax[0].semilogy(h["i"], h["df"], label="data fit")
ax[0].semilogy(h["i"], h["reg"], label="reg"); ax[0].legend(); ax[0].set_xlabel("iteration")
ax[1].semilogy(h["i"][1:], h["res"][1:]); ax[1].axhline(C["tol"], ls="--", c="k"); ax[1].set_title("max relative change (tol dashed)")
ax[2].plot(h["i"], h["L"]); ax[2].set_title("mean L (step = 1/L)")
ax[3].semilogy(h["i"], h["grad_sum"]); ax[3].set_title("L1 norm of grad")
fig, ax = plt.subplots(1, 4, figsize=(18, 3.5), layout="constrained")
m = lambda v: ((v[0] ** 2).sum(0)).sqrt().cpu()
for a, im, ttl in zip(ax, (m(x), m(gt), (m(x) - m(gt)).abs(), m(x0)), ("nmAPG", "GT", "|nmAPG - GT|", "zero-fill")):
    a.imshow(im, cmap="gray" if ttl != "|nmAPG - GT|" else "magma", vmax=None if ttl == "|nmAPG - GT|" else m(gt).max()); a.set_title(ttl); a.axis("off")
plt.show()
# %%
