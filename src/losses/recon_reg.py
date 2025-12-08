import torch

from learned_regularizers.priors import WCRR, ParameterLearningWrapper

def get_recon_regularizer(config):
    reg = WCRR(sigma=config.recon_scale, weak_convexity=0.0).to(config.device)
    regu = ParameterLearningWrapper(reg, device=config.device)
    weights = torch.load('learned_regularizers/weights/bilevel_CT/CRR_bilevel_JFB_for_CT.pt', map_location=config.device)
    regu.load_state_dict(weights)
    regu.eval()
    for p in regu.parameters():
        p.requires_grad_(False)
    return regu