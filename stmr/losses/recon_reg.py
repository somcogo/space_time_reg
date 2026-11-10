from pathlib import Path

import torch

from stmr.regularizers import WCRR, ParameterLearningWrapper

# Anchor data paths to the repo root (this file is <root>/stmr/data/data_utils.py), so loading
# works from any working directory -- e.g. a notebook in notebooks/, not just the repo root.
_WEIGHTS_ROOT = Path(__file__).resolve().parents[2] / "weights"

# weak_convexity=0.0 is the plain CRR; 1.0 is WCRR. Empirically (see notes on the soft_con
# branch) WCRR reconstructs cardiac cine MRI noticeably better than CRR at a comparable
# lambda/scale operating point, even though both ship pretrained on CT rather than MRI.
REGULARIZER_VARIANTS = {
    'crr': {
        'weak_convexity': 0.0,
        'weight_path': _WEIGHTS_ROOT / 'bilevel_CT' / 'CRR_bilevel_JFB_for_CT.pt',
    },
    'wcrr': {
        'weak_convexity': 1.0,
        'weight_path': _WEIGHTS_ROOT / 'bilevel_CT' / 'WCRR_bilevel_JFB_for_CT.pt',
    },
}

def get_recon_regularizer(config):
    # getattr with a default so standalone debug scripts/notebooks that build their own
    # config Namespace (without --reg_variant) keep the original CRR behavior unchanged.
    variant = REGULARIZER_VARIANTS[getattr(config, 'reg_variant', 'crr')]
    reg = WCRR(sigma=0.1, weak_convexity=variant['weak_convexity']).to(config.device)
    wrapped_reg = ParameterLearningWrapper(reg, device=config.device)
    weights = torch.load(variant['weight_path'], map_location=config.device)
    if config.recon_scale is not None:
        weights['scale'] = torch.tensor(config.recon_scale, device=config.device) * torch.ones_like(wrapped_reg.scale, device=config.device)
    if config.reg_alpha is not None:
        weights['alpha'] = torch.tensor(config.reg_alpha, device=config.device) * torch.ones_like(wrapped_reg.alpha, device=config.device)
    # getattr with a default so configs built without reg_beta (tests, notebooks) keep the pretrained beta.
    if getattr(config, 'reg_beta', None) is not None:
        weights['regularizer.beta'] = torch.tensor(config.reg_beta, device=config.device) * torch.ones_like(reg.beta, device=config.device)
    wrapped_reg.load_state_dict(weights)
    wrapped_reg.eval()
    for p in wrapped_reg.parameters():
        p.requires_grad_(False)
    # getattr default False: notebooks/tests that build their own Namespace keep the old path.
    if getattr(config, 'reg_fast', False):
        reg.freeze(compile_potential=str(config.device).startswith('cuda'))
    return wrapped_reg


class _RegEnergy(torch.autograd.Function):
    """reg.g(x) whose backward is the regularizer's own closed-form gradient (reg.grad, computed
    with the energy in the forward): no autograd graph through the 64-channel filter responses,
    whose backward costs ~5x the forward."""

    @staticmethod
    def forward(ctx, x, reg):
        energy, grad = reg.grad(x, get_energy=True)
        ctx.save_for_backward(grad)
        return energy

    @staticmethod
    def backward(ctx, grad_energy):
        (grad,) = ctx.saved_tensors
        return grad_energy.view(-1, *[1] * (grad.dim() - 1)) * grad, None


def reg_energy(config, reg, x):
    """Per-image reg.g(x), differentiable in x; via _RegEnergy when config.reg_fast."""
    if (getattr(config, 'reg_fast', False) and getattr(reg, 'has_get_energy', False)
            and torch.is_grad_enabled() and x.requires_grad):
        return _RegEnergy.apply(x, reg)
    return reg.g(x)