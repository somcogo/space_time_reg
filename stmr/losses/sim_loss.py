from torch import nn

from stmr.losses.ncc import NODEO_NCC
from stmr.losses.normalized_gradient_field import NormalizedGradientField2d


def get_sim_loss_fn(config, imgs):
    if config.loss == 'mse':
        loss_fn = nn.MSELoss(reduction='none')
    elif config.loss == 'ngf':
        if imgs.dim() == 3:
            loss_fn = NormalizedGradientField2d(mm_spacing=1, eps=1e-6, reduction='none')
        else:
            loss_fn = NODEO_NCC()
    return loss_fn.to(config.device)