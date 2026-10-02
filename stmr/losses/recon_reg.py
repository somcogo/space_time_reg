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
    wrapped_reg.load_state_dict(weights)
    wrapped_reg.eval()
    for p in wrapped_reg.parameters():
        p.requires_grad_(False)
    return wrapped_reg