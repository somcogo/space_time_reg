import torch
from torch import nn
import torch.nn.functional as F

from src.utils import generate_coord_tensor
from src.normalized_gradient_field import NormalizedGradientField2d, NormalizedGradientField3d, spatial_filter_nd, _grad_param

def calculate_losses(config, abs_phi, data, rel_vel, func, time_series):
    imgs, neural_reps = data
    imgs = imgs.to(config.device)
    neural_reps = [net.to(config.device) for net in neural_reps]
    coord_tensor = generate_coord_tensor(imgs.shape[1:], config.device)
    rel_phi = abs_phi - coord_tensor

    # reshape phi from (-1, 2) to (1, x, y, z, 2)
    phi_shape = [-1] + list(imgs.shape[1:]) + [len(imgs.shape)-1]
    sim_energy, moved_imgs = similarity_loss(imgs, neural_reps, abs_phi, config)
    loss_sim = config.lambda_st * sim_energy.mean()

    J = get_Jacobian(func, time_series, dims=len(imgs.shape)-1, img_shape=imgs.shape[1:])

    neg_Jdet_energy = neg_Jdet_loss(J)
    # neg_Jdet_energy = neg_Jdet_loss(abs_phi[-1].reshape(phi_shape))
    loss_negJ = config.lambda_negJ * neg_Jdet_energy.mean()

    # phi_grad_energy = grad_loss(rel_phi[-1].reshape(phi_shape))
    # loss_smt = config.lambda_smt * phi_grad_energy.mean()

    vel_shape = [-1] + list(imgs.shape[1:]) + [len(imgs.shape) - 1]
    vel_reshaped = rel_vel.reshape(vel_shape)
    vel_grad_energy = grad_loss(vel_reshaped)
    loss_grd = config.lambda_grd * vel_grad_energy.mean()
    # vel_grad_l2 = torch.linalg.vector_norm(J, dim=-1)
    # loss_grd = config.lambda_grd * vel_grad_l2.mean()

    return [loss_sim, loss_negJ, loss_grd], moved_imgs, sim_energy

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

# def JacboianDet(phi):
#     if len(phi.shape) == 4:
#         dx = phi[:, 1:, :-1, :] - phi[:, :-1, :-1, :]
#         dy = phi[:, :-1, 1:, :] - phi[:, :-1, :-1, :]

#         det = dx[:, :, :, 0] * dy[:, :, :, 1] - dx[:, :, :, 1] * dy[:, :, :, 0]
#     else:
#         dx = phi[:, 1:, 1:, 1:, :] - phi[:, :-1, 1:, 1:, :]
#         dy = phi[:, 1:, 1:, 1:, :] - phi[:, 1:, :-1, 1:, :]
#         dz = phi[:, 1:, 1:, 1:, :] - phi[:, 1:, 1:, :-1, :]

#         det0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
#         det1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 0])
#         det2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy[:, :, :, :, 1] * dz[:, :, :, :, 0])

#         det = det0 - det1 + det2
#     return det

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

def grad_loss(f):
    f = f.transpose(1, -1)
    b, c = f.shape[:2]
    spatial_shape = f.shape[2:]

    # [B*N, H, W]
    f = f.reshape(b * c, 1, *spatial_shape)
    grad_u_kernel = _grad_param(2, 'default', axis=0).to(f.device)
    grad_v_kernel = _grad_param(2, 'default', axis=1).to(f.device)
    grad_u = spatial_filter_nd(f, grad_u_kernel)
    grad_v = spatial_filter_nd(f, grad_v_kernel)
    grad = grad_u ** 2 + grad_v ** 2
    if len(f.shape) == 5:
        grad_w_kernel = _grad_param(2, 'default', axis=2)
        grad_w = spatial_filter_nd(f, grad_w_kernel)
        grad = grad + grad_w ** 2
    grad = grad.view(b, c, *spatial_shape).transpose(1, -1)
    return grad
        
    if len(f.shape) == 5:
    #     gradient_magnitude = (((f[:, 1:, :, :, :] - f[:, :-1, :, :, :]) ** 2).mean() + \
    #  ((f[:, :, 1:, :, :] - f[:, :, :-1, :, :]) ** 2).mean() + \
    #  ((f[:, :, :, 1:, :] - f[:, :, :, :-1, :]) ** 2).mean())
        x_grad_magnitude = ((f[:, 1:, :, :, :] - f[:, :-1, :, :, :]) ** 2)
        y_grad_magnitude = ((f[:, :, 1:, :, :] - f[:, :, :-1, :, :]) ** 2)
        z_grad_magnitude = ((f[:, :, :, 1:, :] - f[:, :, :, :-1, :]) ** 2)
        grad_magnitude = x_grad_magnitude + y_grad_magnitude + z_grad_magnitude
    else:
    #     gradient_magnitude = (((f[:, 1:, :, :] - f[:, :-1, :, :]) ** 2).mean() + \
    #  ((f[:, :, 1:, :] - f[:, :, :-1, :]) ** 2).mean())
        x_grad_magnitude = ((f[:, 1:, 1:, :] - f[:, :-1, 1:, :]) ** 2)
        y_grad_magnitude = ((f[:, 1:, 1:, :] - f[:, 1:, :-1, :]) ** 2)
        grad_magnitude = x_grad_magnitude + y_grad_magnitude
    return grad_magnitude

# def magnitude_loss(all_v):
#     if len(all_v.shape) == 5:
#         all_v_x_2 = all_v[:, 0, :, :, :] * all_v[:, 0, :, :, :]
#         all_v_y_2 = all_v[:, 1, :, :, :] * all_v[:, 1, :, :, :]
#         all_v_z_2 = all_v[:, 2, :, :, :] * all_v[:, 2, :, :, :]
#         # all_v_magnitude = torch.mean(all_v_x_2 + all_v_y_2 + all_v_z_2)
#         magnitude = all_v_x_2 + all_v_y_2 + all_v_z_2
#     else:
#         all_v_x_2 = all_v[:, 0, :, :] * all_v[:, 0, :, :]
#         all_v_y_2 = all_v[:, 1, :, :] * all_v[:, 1, :, :]
#         # all_v_magnitude = torch.mean(all_v_x_2 + all_v_y_2)
#         magnitude = all_v_x_2 + all_v_y_2
#     return magnitude

def get_Jacobian(func, time_series, dims, img_shape):
    # coords = generate_coord_tensor(img_shape, time_series.device)
    # coords.requires_grad = True
    coords = torch.rand((1024, dims), requires_grad=True, device=time_series.device)*2 - 1
    rel_vel = []
    for t in time_series:
        vel = func(t, coords)
        rel_vel.append(vel)
    rel_vel = torch.concat(rel_vel)
    return torch.stack([gradient(input=coords, output=rel_vel[i]) for i in range(dims)], dim=1)

def gradient(input, output, grad_outputs=None):
    """Compute the gradient of the output wrt the input."""

    grad_outputs = torch.ones_like(output)
    grad = torch.autograd.grad(
        output, [input], grad_outputs=grad_outputs, create_graph=True
    )[0]
    return grad