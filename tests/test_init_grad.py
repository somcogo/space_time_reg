"""The nmAPG init's closed-form gradient (init_grad='analytic') must match autograd.

Small random complex image + row mask on CPU, with the pretrained (git-tracked) WCRR weights.
"""
from argparse import Namespace

import pytest
import torch

from stmr.data.fft_utils import FastmriFT, FastmriIFT
from stmr.data.recon_init import get_functions, get_reg


def _config(**kw):
    base = dict(device="cpu", reg="learned", reg_variant="wcrr", reg_alpha=1.0, recon_scale=3.0,
                lambda_init_recon=0.05, lambda_st=1.0, init_loss="l2", init_reg_abs=False,
                detach_grads=True, init_grad="autograd")
    return Namespace(**{**base, **kw})


def _problem(seed=0, T=2, H=32, W=48):
    g = torch.Generator().manual_seed(seed)
    x = 0.05 * torch.randn(T, 2, H, W, generator=g)
    mask = torch.zeros(T, 2, H, W)
    mask[:, :, ::3] = 1
    kdata = mask * FastmriFT()(0.05 * torch.randn(T, 2, H, W, generator=g))
    return x, torch.cat([kdata, mask], dim=1)


@pytest.mark.parametrize("init_reg_abs", [True, False])
def test_analytic_matches_autograd(init_reg_abs):
    x, packed = _problem()
    out = {}
    for mode in ("autograd", "analytic"):
        cfg = _config(init_reg_abs=init_reg_abs, init_grad=mode)
        *_, energy_grad, energy_and_grad = get_functions(cfg, get_reg(cfg), FastmriFT(), FastmriIFT())
        out[mode] = (*energy_and_grad(x, packed), energy_grad(x, packed))
    (e_auto, g_auto, _), (e_ana, g_ana, nabla_ana) = out["autograd"], out["analytic"]
    assert torch.allclose(e_ana, e_auto, rtol=1e-6)
    assert (g_ana - g_auto).norm() / g_auto.norm() < 1e-4
    assert torch.equal(nabla_ana, g_ana)
    assert not g_ana.requires_grad


@pytest.mark.parametrize("init_reg_abs", [True, False])
def test_analytic_handles_frame_subset(init_reg_abs):
    # nmAPG drops converged frames by indexing x[idx], y[idx]; each frame must stand alone.
    x, packed = _problem()
    cfg = _config(init_reg_abs=init_reg_abs, init_grad="analytic")
    *_, energy_and_grad = get_functions(cfg, get_reg(cfg), FastmriFT(), FastmriIFT())
    e_all, g_all = energy_and_grad(x, packed)
    e_one, g_one = energy_and_grad(x[1:], packed[1:])
    assert g_one.shape == x[1:].shape
    assert torch.allclose(e_one, e_all[1:], rtol=1e-6)
    assert torch.allclose(g_one, g_all[1:], rtol=1e-5, atol=1e-9)


@pytest.mark.parametrize("override", [dict(init_loss="mag_l1"), dict(reg="tv")])
def test_analytic_rejects_unsupported(override):
    cfg = _config(init_grad="analytic", **override)
    with pytest.raises(ValueError, match="analytic"):
        get_functions(cfg, get_reg(cfg), FastmriFT(), FastmriIFT())
