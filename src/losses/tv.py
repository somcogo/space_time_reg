import torch
from torch import nn

def get_tv(config):
    return TV()

class TV(nn.Module):
    def __init__(self, eps=1e-6):
        super(TV, self).__init__()
        self.eps = eps

    def grad_forward(self, x):
        """
        Forward finite differences
        x: (..., H, W)
        returns gx, gy same shape
        """
        gx = torch.zeros_like(x)
        gy = torch.zeros_like(x)

        gx[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
        gy[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]

        return gx, gy
    
    def div_backward(self, px, py):
        """
        Divergence (adjoint of grad_forward)
        px, py: (..., H, W)
        """
        div = torch.zeros_like(px)

        div[..., 1:, :] += px[..., 1:, :]
        div[..., :-1, :] -= px[..., :-1, :]

        div[..., :, 1:] += py[..., :, 1:]
        div[..., :, :-1] -= py[..., :, :-1]

        return div
    
    def g(self, x):
        gx, gy = self.grad_forward(x)
        mag = torch.sqrt(gx**2 + gy**2 + self.eps)
        return mag.sum()
    
    def grad(self, x):
        gx, gy = self.grad_forward(x)

        denom = torch.sqrt(gx**2 + gy**2 + self.eps)

        px = gx / denom
        py = gy / denom

        return -self.div_backward(px, py)
