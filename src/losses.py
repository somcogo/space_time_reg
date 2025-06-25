import torch
from torch import nn
import torch.nn.functional as F

from src.utils import generate_coord_tensor
from src.normalized_gradient_field import NormalizedGradientField2d, NormalizedGradientField3d, spatial_filter_nd, _grad_param

def calculate_losses(config, abs_phi, data, rel_vel, func, time_series):
    imgs, neural_reps = data
    imgs = imgs.to(config.device)
    neural_reps = [net.to(config.device) for net in neural_reps]

    sim_energy, moved_imgs = similarity_loss(imgs, neural_reps, abs_phi, config)
    loss_sim = config.lambda_st * sim_energy.mean()

    vel_shape = [-1] + list(imgs.shape[1:]) + [len(imgs.shape) - 1]
    vel_reshaped = rel_vel.reshape(vel_shape)
    findiff_J =  fin_diff_Jacobian(vel_reshaped)
    if config.func_name == 'sirent':
        autograd_grid_J = get_Jacobian(func, time_series, dims=len(imgs.shape)-1, img_shape=imgs.shape[1:], autograd_grid=True)
    elif config.func_name == 'siren':
        autograd_grid_J = get_Jacobian(func, time_series[:1], dims=len(imgs.shape)-1, img_shape=imgs.shape[1:], autograd_grid=True)
    if config.fin_diff_grad:
        J = findiff_J
    elif config.autograd_grid:
        J = autograd_grid_J
    else:
        if config.func_name == 'sirent':
            J = get_Jacobian(func, time_series, dims=len(imgs.shape)-1, img_shape=imgs.shape[1:], autograd_grid=False)
        elif config.func_name == 'siren':
            J = get_Jacobian(func, time_series[:1], dims=len(imgs.shape)-1, img_shape=imgs.shape[1:], autograd_grid=False)
    
    
    neg_Jdet_energy = neg_Jdet_loss(J)
    loss_negJ = config.lambda_negJ * neg_Jdet_energy.mean()

    vel_grad_l2 = torch.linalg.vector_norm(J, dim=-1)
    loss_grd = config.lambda_grd * vel_grad_l2.mean()

    return [loss_sim, loss_negJ, loss_grd], moved_imgs, [sim_energy.detach(), findiff_J.detach(), autograd_grid_J.detach()]

def similarity_loss(imgs, neural_reps, abs_phi, args):
    if args.loss == 'mse':
        loss_fn = nn.MSELoss(reduction='none')
    elif args.loss == 'ngf':
        if imgs.dim() == 3:
            loss_fn = NormalizedGradientField2d(mm_spacing=1, eps=1e-6, reduction='none')
        else:
            loss_fn = NormalizedGradientField3d(mm_spacing=1, eps=1e-6, reduction='none')
    loss_fn = loss_fn.to(args.device)
    net = neural_reps[0].net

    model_out = net(abs_phi)
    moved = (model_out.squeeze(2) + 1) / 2
    moved = moved.reshape(imgs.shape)
    loss = loss_fn(imgs.unsqueeze(1), moved.unsqueeze(1))
    return loss, moved.detach().cpu()

def neg_Jdet_loss(J):
    Jdet = torch.det(J)
    # Jdet = JacboianDet(J)
    neg_Jdet = -1.0 * Jdet
    neg_Jdet = F.relu(neg_Jdet) + 1
    out = torch.log(neg_Jdet)

    # out = - torch.log(Jdet)

    # out = torch.log(Jdet)
    # # out = out ** 2
    # out = torch.abs(out)
    # out = torch.exp(out)
    # out = out - 1
    # out = torch.abs(out)

    return out

def fin_diff_Jacobian(f):
    partial_grads = [fin_diff_gradient(f, dim) for dim in range(len(f.shape)-2)]
    J = torch.stack(partial_grads, dim=-1)
    return J

def fin_diff_gradient(f, axis):
    dims = len(f.shape) - 2
    if dims == 2:
        f = f.permute(0, 3, 1, 2)
    elif dims == 3:
        f = f.permute(0, 4, 1, 2, 3)
    b, c = f.shape[:2]
    spatial_shape = f.shape[2:]

    # [B*N, H, W]
    f = f.reshape(b * c, 1, *spatial_shape)
    grad_kernel = _grad_param(dims, 'default', axis=axis).to(f.device)
    grad = spatial_filter_nd(f, grad_kernel)
    grad = grad.view(b, c, *spatial_shape)
    if dims == 2:
        grad = grad.permute(0, 2, 3, 1)
    elif dims == 3:
        grad = grad.permute(0, 2, 3, 4, 1)
    return grad

def get_Jacobian(func, time_series, dims, img_shape, autograd_grid):
    if autograd_grid:
        coords = generate_coord_tensor(img_shape, time_series.device)
        coords.requires_grad = True
    else:
        coords = torch.rand((1024, dims), requires_grad=True, device=time_series.device)*2 - 1
    rel_vel = []
    for t in time_series:
        vel = func(t, coords)
        rel_vel.append(vel)
    rel_vel = torch.concat(rel_vel)
    return torch.stack([gradient(input=coords, output=rel_vel[:,i]) for i in range(dims)], dim=1)

def gradient(input, output, grad_outputs=None):
    """Compute the gradient of the output wrt the input."""

    grad_outputs = torch.ones_like(output)
    grad = torch.autograd.grad(
        output, [input], grad_outputs=grad_outputs, create_graph=True
    )[0]
    return grad