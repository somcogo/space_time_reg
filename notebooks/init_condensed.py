# %% [markdown]
# # nmAPG initial reconstruction, condensed
# The pipeline's init solve in one file and in execution order:
# data -> operators -> regularizer -> objective -> solve.
# Mirrors: stmr/data/data_utils.py (_cmr_load, generate_standard_mask, complex_abs),
# stmr/data/fft_utils.py, stmr/data/data_load.py (prepare_data), stmr/losses/recon_reg.py,
# stmr/regularizers/{wcrr,wrapper}.py, stmr/data/recon_init.py.
# The solver is the pipeline's own stmr/data/nmapg.py (imported, not copied).
# Evaluation, plots, sanity checks and the equivalence test against the pipeline are in
# init_condensed_checks.py. `# NOTE:` marks behaviour worth scrutinising.

# %% 0. Config -- every knob explicit (the pipeline falls back to Config defaults silently)
import json
import os
from pathlib import Path

import matplotlib
import torch
import torch.nn.functional as nnf

_backend = matplotlib.get_backend()
from stmr.data.nmapg import nmAPG                 # noqa: E402  the pipeline's solver
matplotlib.use(_backend)                          # importing stmr switches matplotlib to Agg; switch back so inline plots show

C = dict(
    patient="001", slice=0, start_frame=0, T=2,   # data
    factor=4,                                     # mask: 24 centre rows + every factor-th row
    variant="wcrr",                               # 'crr' (weak_cvx 0) | 'wcrr' (weak_cvx 1)
    lam=0.,                                     # lambda_init_recon
    scale=5.5,                                    # recon_scale (overrides pretrained s)
    alpha=1.0,                                    # reg_alpha   (overrides pretrained alpha)
    abs_reg=True,                                 # init_reg_abs: regularise |x| (True) or re/im
    grad="analytic",                              # init_grad: 'autograd' | 'analytic' (closed-form gradient)
    lam_st=1.0,                                   # lambda_st (data weight)
    L_init=1.0,                                   # nmAPG's initial L (first step = 1/L_init)
    max_iter=1000,                                 # recon_epochs
    tol=1e-4,                                     # nmAPG stop: max_t ||x_k - x_{k-1}|| / ||x_k|| < tol
    device="cuda",                                # pick a GPU with no compute processes listed
    # equiv_iters=50,                               # iterations for the equivalence cell in init_condensed_checks.py
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

    def lip():                                               # spectral norm of K^T K from its impulse response
        h = dirac
        for w in filters():
            h = nnf.conv2d(h, w, padding=2)
        for w in reversed(filters()):
            h = nnf.conv_transpose2d(h, w, padding=2)
        return torch.fft.fft2(h, s=[256, 256]).abs().max()

    def conv(x):                                             # K x, normalised so ||K||_2 = 1
        x = x / torch.sqrt(lip())
        for w in filters():
            x = nnf.conv2d(x, w, padding=2)                  # NOTE: zero padding -> image borders act as edges
        return x

    def conv_T(z):                                           # K^T z, same normalisation
        z = z / torch.sqrt(lip())
        for w in reversed(filters()):
            z = nnf.conv_transpose2d(z, w, padding=2)
        return z

    def sl1(z):                                              # smoothed l1 (Huber-like, threshold 1)
        return torch.clip(z**2, 0.0, 1.0) / 2 + torch.clip(torch.abs(z), 1.0) - 1.0

    def R(x):                                                # WCRR.g: energy per image, [B]
        z = conv(x) * torch.exp(scaling)
        z = sl1(torch.exp(beta) * z) * torch.exp(-beta) - sl1(z) * wcvx
        z = z * torch.exp(-2 * scaling)
        return z.sum(dim=(1, 2, 3))

    def Rt(x):                                               # ParameterLearningWrapper.g
        return torch.exp(alpha - 2 * scale) * R(torch.exp(scale) * x)

    def R_and_grad(x):                                       # WCRR.grad(get_energy=True): energy [B] and dR/dx
        z = conv(x) * torch.exp(scaling)
        e = sl1(torch.exp(beta) * z) * torch.exp(-beta) - sl1(z) * wcvx
        e = (e * torch.exp(-2 * scaling)).sum(dim=(1, 2, 3))
        g = torch.clip(torch.exp(beta) * z, -1.0, 1.0) - torch.clip(z, -1.0, 1.0) * wcvx   # derivative of sl1 is a clip
        return e, conv_T(g * torch.exp(-scaling))

    def Rt_and_grad(x):                                      # ParameterLearningWrapper.grad(get_energy=True)
        e, g = R_and_grad(torch.exp(scale) * x)
        return torch.exp(alpha - 2 * scale) * e, torch.exp(alpha - scale) * g
    return Rt, Rt_and_grad

# %% 4. Objective -- per-frame energy E_t(x) = lam_st * 0.5*sum|M F x - y|^2 + lam * Rt(input), shape [T]
def make_objective(C, Rt, Rt_and_grad):
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

    def analytic_energy_and_grad(x, yp):                     # closed form: same energy as energy(), gradient without autograd
        with torch.no_grad():
            k, m = yp[:, :2], yp[:, 2:]
            d = m * FT(x) - k
            df = 0.5 * (d.abs() ** 2).sum((1, 2, 3))
            if C["abs_reg"]:
                mag = ((x**2).sum(dim=1, keepdim=True) + 1e-6).sqrt()
                e, g = Rt_and_grad(mag)
                e, g = C["lam"] * e, g * x / mag             # chain rule: d|x|_eps/dx = x / |x|_eps
            else:
                e, g = Rt_and_grad(x.flatten(0, 1).unsqueeze(1))
                e, g = (C["lam"] * e).reshape(x.shape[0], -1).sum(1), g.reshape(x.shape)
            E = C["lam_st"] * df.reshape(-1) + e.reshape(-1)
            G = C["lam_st"] * IFT(m * d) + C["lam"] * g      # data gradient: F^H M (M F x - y)
        return E.reshape(-1), G

    def energy_grad(x, yp):
        if C["grad"] == "analytic":
            return analytic_energy_and_grad(x, yp)[1]
        xr = x.detach().clone().requires_grad_(True)
        (C["lam_st"] * data_fit(xr, yp) + reg(xr)).sum().backward()   # frames are independent -> per-frame grads
        return xr.grad

    def energy_and_grad(x, yp):                              # what nmAPG calls once per iteration
        if C["grad"] == "analytic":
            return analytic_energy_and_grad(x, yp)
        return energy(x, yp), energy_grad(x, yp)
    return data_fit, reg, energy, energy_grad, energy_and_grad

# %% 5. Solve -- the pipeline's nmAPG (stmr/data/nmapg.py)
torch.manual_seed(0)
k_gt, mask, y, gt, x0 = load_data(C, C["device"])
Rt, Rt_and_grad = load_reg(C, C["device"])
data_fit, reg, energy, energy_grad, energy_and_grad = make_objective(C, Rt, Rt_and_grad)
yp = torch.cat([y, mask.to(y)], dim=1)                       # packed measurements [T,4,H,W]
hist = []                                                    # per-iteration stats, filled by the solver's callback
def callback(i, x, s):                                       # s = {data_fit, reg, res, L, n_active}; i = -1 is the initial point
    hist.append(dict(i=i + 1, df=s["data_fit"], reg=s["reg"], res=s["res"], L=s["L"], n_active=s["n_active"]))
x, L, it, conv, _ = nmAPG(x0=x0, y=yp, f=energy, nabla=energy_grad, f_and_nabla=energy_and_grad,
                          max_iter=C["max_iter"], L_init=C["L_init"], tol=C["tol"],
                          data_fit=lambda v, yy: C["lam_st"] * data_fit(v, yy), reg=reg, callback=callback)
print(f"stopped after iteration {it + 1}, converged per frame: {conv.tolist()}, final L: {L.flatten().tolist()}")
