"""reg_fast (WCRR.freeze + the closed-form backward of recon_reg.reg_energy) must reproduce the
old path: autograd through ParameterLearningWrapper(WCRR).g, and the wrapper's own grad."""
from types import SimpleNamespace

import pytest
import torch

from stmr.losses.recon_reg import get_recon_regularizer, reg_energy


def _cfg(variant, device="cpu", **kw):
    return SimpleNamespace(device=device, reg_variant=variant, recon_scale=3.0, reg_alpha=1.0,
                           reg_beta=3.0, **kw)


@pytest.mark.parametrize("variant", ["crr", "wcrr"])
@pytest.mark.parametrize("amp", [0.01, 0.3])  # mostly quadratic zone / mostly linear zone
def test_matches_old_path(variant, amp):
    old_cfg, new_cfg = _cfg(variant, reg_fast=False), _cfg(variant, reg_fast=True)
    old = get_recon_regularizer(old_cfg).double()
    new = get_recon_regularizer(new_cfg).double()  # frozen: the Lipschitz cache is re-made in float64
    assert new.regularizer._frozen and not old.regularizer._frozen
    g = torch.Generator().manual_seed(0)
    x = amp * torch.randn(3, 1, 24, 40, generator=g, dtype=torch.float64)
    w = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)  # non-uniform upstream gradient
    xo, xn = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    eo, en = reg_energy(old_cfg, old, xo), reg_energy(new_cfg, new, xn)
    (eo * w).sum().backward()
    (en * w).sum().backward()
    assert en.shape == eo.shape
    assert torch.allclose(en, eo, rtol=1e-10, atol=0)
    assert (xn.grad - xo.grad).norm() / xo.grad.norm() < 1e-10
    e1, g1 = old.grad(x, get_energy=True)
    e2, g2 = new.grad(x, get_energy=True)
    assert torch.allclose(e2, e1, rtol=1e-10, atol=0)
    assert (g2 - g1).norm() / g1.norm() < 1e-10
    with torch.no_grad():  # energy-only path (nmAPG line search)
        assert torch.allclose(new.g(x), eo.detach(), rtol=1e-10, atol=0)


def test_old_path_is_unchanged_by_default():
    reg = get_recon_regularizer(_cfg("crr"))  # Namespace without reg_fast
    assert not reg.regularizer._frozen and reg.regularizer._pointwise == reg.regularizer._potential
    x = torch.randn(2, 1, 16, 16, requires_grad=True)
    assert reg_energy(_cfg("crr"), reg, x).grad_fn.name() != "_RegEnergyBackward"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled path needs CUDA")
def test_compiled_cuda_matches():
    old_cfg, new_cfg = _cfg("crr", "cuda", reg_fast=False), _cfg("crr", "cuda", reg_fast=True)
    old, new = get_recon_regularizer(old_cfg), get_recon_regularizer(new_cfg)
    assert new.regularizer._pointwise is not new.regularizer._potential
    x = 0.2 * torch.randn(4, 1, 64, 96, device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    xo, xn = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    eo, en = reg_energy(old_cfg, old, xo), reg_energy(new_cfg, new, xn)
    eo.sum().backward()
    en.sum().backward()
    assert torch.allclose(en, eo, rtol=1e-5)
    assert (xn.grad - xo.grad).norm() / xo.grad.norm() < 1e-3  # TF32 convs, other cuDNN algorithms
