import torch

from learned_regularizers.priors import WCRR, ParameterLearningWrapper

def get_recon_regularizer(config):
    reg = WCRR(sigma=0.1, weak_convexity=0.0).to(config.device)
    wrapped_reg = ParameterLearningWrapper(reg, device=config.device)
    weights = torch.load('learned_regularizers/weights/score_for_CT/CRR_score_training_for_CT.pt', map_location=config.device)
    # weights = torch.load('learned_regularizers/weights/bilevel_CT/CRR_bilevel_JFB_for_CT.pt', map_location=config.device)
    if config.recon_scale is not None:
        weights['regularizer.scaling'] = torch.tensor(config.recon_scale, device=config.device) * torch.ones_like(reg.scaling, device=config.device)
    wrapped_reg.load_state_dict(weights)
    wrapped_reg.eval()
    for p in wrapped_reg.parameters():
        p.requires_grad_(False)
    return wrapped_reg