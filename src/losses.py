import torch
from torch import nn
import torch.nn.functional as F

from src.utils import generate_coord_tensor
from src.normalized_gradient_field import NormalizedGradientField2d, NormalizedGradientField3d

def calculate_losses(config, phi, data, vel):
    imgs, neural_reps = data
    imgs = imgs.to(config.device)
    neural_reps = [net.to(config.device) for net in neural_reps]
    # grid = get_mgrid(imgs[0].shape, dim=len(imgs.shape) - 1).view(phi.shape[1:]).to(config.device)
    # grid = generate_grid_tensor(data[0].shape[1:]).to(config.device)
    # grid = grid[:, [1, 0], ...]
    coord_tensor = generate_coord_tensor(imgs.shape[1:], config.device)

    # reshape phi from (-1, 2) to (1, x, y, z, 2)
    phi_shape = [-1] + list(imgs.shape[1:]) + [len(imgs.shape)-1]
    loss_sim, moved_imgs = similarity_loss(imgs, neural_reps, phi, config)
    neg_Jdet_energy = neg_Jdet_loss((phi[-1]-coord_tensor).reshape(phi_shape))
    loss_negJ = config.lambda_negJ * neg_Jdet_energy.mean()
    phi_grad_energy = grad_loss(phi[-1].reshape(phi_shape))
    loss_smt = config.lambda_smt * phi_grad_energy.mean()

    vel_shape = list(imgs.shape[1:]) + [len(imgs.shape) - 1]
    vel_reshaped = vel.reshape(vel_shape).permute(2, 0, 1).unsqueeze(0)
    # loss_mag = config.lambda_mag * magnitude_loss((phi[1:] - phi[:-1]).reshape(phi_shape))
    # magnitude_energy = magnitude_loss(vel_reshaped)
    # loss_mag = config.lambda_mag * magnitude_energy.mean()
    vel_grad_energy = grad_loss(vel_reshaped)
    loss_grd = config.lambda_grd * vel_grad_energy.mean()
    # loss_negJ = 0.
    # loss_smt = 0.
    # loss_mag = 0.

    return [config.lambda_st * loss_sim, loss_negJ, loss_smt, loss_grd], moved_imgs, [neg_Jdet_energy, phi_grad_energy, vel_grad_energy]

def similarity_loss(imgs, neural_reps, phi, args):
    if args.loss == 'mse':
        loss_fn = nn.MSELoss()
    elif args.loss == 'ngf':
        if imgs.dim() == 3:
            loss_fn = NormalizedGradientField2d(mm_spacing=1)
        else:
            loss_fn = NormalizedGradientField3d(mm_spacing=1)
    loss_fn = loss_fn.to(args.device)
    if args.use_nreps:
        # losses = []
        # moving = imgs[0]
        # for time in range(imgs.shape[0]):
        #     net = neural_reps[time].net
        #     coords = phi[time]
        #     model_out = net(coords)
        #     fixed_preimage = (model_out.squeeze(1) + 1) / 2
        #     fixed_preimage = fixed_preimage.reshape(moving.shape)
        #     losses.append(loss_fn(fixed_preimage, moving))
        # loss = sum(losses) / len(losses)

        net = neural_reps[0].net
        fixed = imgs

        if args.const_phi:
            coord_tensor = coord_tensor = generate_coord_tensor(imgs.shape[1:], args.device)
            rel_phi = (phi-coord_tensor)
            phi1 = rel_phi[1:2].expand(phi.shape[0], phi.shape[1], 2) * torch.arange(phi.shape[0], device=args.device).unsqueeze(1).unsqueeze(1)
            coords = phi1 + coord_tensor
        else:
            coords = phi

        model_out = net(coords)
        moved = (model_out.squeeze(2) + 1) / 2
        moved = moved.reshape(fixed.shape)
        loss = loss_fn(fixed.unsqueeze(1), moved.unsqueeze(1))
    else:
        moving = torch.stack([imgs[0]]*phi.shape[0])
        phi_grid_sample = torch.stack([phi[:, 1], phi[:, 0]], dim=1).squeeze(1).permute(0, 2, 3, 1)
        moved = F.grid_sample(moving, phi_grid_sample, align_corners=True, mode='bilinear')
        fixed = imgs
        loss = loss_fn(fixed, moved)
    return loss, moved.detach().cpu()



# def grid_sample_similarity_loss(moving, fixed, phi):
#     phi_grid_sample = torch.stack([phi[:, 1], phi[:, 0]], dim=1)
#     loss_fn = nn.MSELoss()
#     fixed_preimage = F.grid_sample(fixed.unsqueeze(0).unsqueeze(0), phi_grid_sample, align_corners=True, mode='bilinear')
#     l2loss = loss_fn(moving, fixed_preimage.squeeze())
#     return l2loss

# def n_rep_similarity_loss(moving, fixed_n_rep, coords):
#     loss_fn = nn.MSELoss()
#     model_out = fixed_n_rep.net(coords.squeeze(0).permute(2, 1, 0))
#     fixed = model_out.squeeze(2)
#     fixed = (fixed + 1) / 2
#     l2loss = loss_fn(fixed, moving.squeeze())
#     return l2loss

# def space_time_loss(imgs, n_reps, phi):
#     loss_fn = nn.MSELoss()
#     moving = torch.stack([imgs[0]]*phi.shape[0])
#     grid = phi.squeeze(1).permute(0, 2, 3, 1)
#     fixed = imgs
#     moved = F.grid_sample(moving.unsqueeze(1), grid, align_corners=True, mode='bilinear')
#     l2loss = loss_fn(fixed, moved.squeeze())
#     return l2loss

class NCC(torch.nn.Module):
    """
    NCC with cumulative sum implementation for acceleration. local (over window) normalized cross correlation.
    """

    def __init__(self, win=21, eps=1e-5):
        super(NCC, self).__init__()
        self.eps = eps
        self.win = win
        self.win_raw = win

    def window_sum_cs3D(self, I, win_size):
        half_win = int(win_size / 2)
        pad = [half_win + 1, half_win] * 3

        I_padded = F.pad(I, pad=pad, mode='constant', value=0)  # [x+pad, y+pad, z+pad]

        # Run the cumulative sum across all 3 dimensions
        I_cs_x = torch.cumsum(I_padded, dim=2)
        I_cs_xy = torch.cumsum(I_cs_x, dim=3)
        I_cs_xyz = torch.cumsum(I_cs_xy, dim=4)

        x, y, z = I.shape[2:]

        # Use subtraction to calculate the window sum
        I_win = I_cs_xyz[:, :, win_size:, win_size:, win_size:] \
                - I_cs_xyz[:, :, win_size:, win_size:, :z] \
                - I_cs_xyz[:, :, win_size:, :y, win_size:] \
                - I_cs_xyz[:, :, :x, win_size:, win_size:] \
                + I_cs_xyz[:, :, win_size:, :y, :z] \
                + I_cs_xyz[:, :, :x, win_size:, :z] \
                + I_cs_xyz[:, :, :x, :y, win_size:] \
                - I_cs_xyz[:, :, :x, :y, :z]

        return I_win

    def forward(self, I, J):
        # compute CC squares
        I = I.double()
        J = J.double()

        I2 = I * I
        J2 = J * J
        IJ = I * J

        # compute local sums via cumsum trick
        I_sum_cs = self.window_sum_cs3D(I, self.win)
        J_sum_cs = self.window_sum_cs3D(J, self.win)
        I2_sum_cs = self.window_sum_cs3D(I2, self.win)
        J2_sum_cs = self.window_sum_cs3D(J2, self.win)
        IJ_sum_cs = self.window_sum_cs3D(IJ, self.win)

        win_size_cs = (self.win * 1.) ** 3

        u_I_cs = I_sum_cs / win_size_cs
        u_J_cs = J_sum_cs / win_size_cs

        cross_cs = IJ_sum_cs - u_J_cs * I_sum_cs - u_I_cs * J_sum_cs + u_I_cs * u_J_cs * win_size_cs
        I_var_cs = I2_sum_cs - 2 * u_I_cs * I_sum_cs + u_I_cs * u_I_cs * win_size_cs
        J_var_cs = J2_sum_cs - 2 * u_J_cs * J_sum_cs + u_J_cs * u_J_cs * win_size_cs

        cc_cs = cross_cs * cross_cs / (I_var_cs * J_var_cs + self.eps)
        cc2 = cc_cs  # cross correlation squared

        # return negative cc.
        return 1. - torch.mean(cc2).float()

def JacboianDet(phi):
    if len(phi.shape) == 4:
        dx = phi[:, 1:, 1:, :] - phi[:, :-1, 1:, :]
        dy = phi[:, 1:, 1:, :] - phi[:, 1:, :-1, :]

        det = dx[:, :, :, 0] * dy[:, :, :, 1] - dx[:, :, :, 1] * dy[:, :, :, 0]
    else:
        dx = phi[:, 1:, 1:, 1:, :] - phi[:, :-1, 1:, 1:, :]
        dy = phi[:, 1:, 1:, 1:, :] - phi[:, 1:, :-1, 1:, :]
        dz = phi[:, 1:, 1:, 1:, :] - phi[:, 1:, 1:, :-1, :]

        det0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
        det1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 0])
        det2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy[:, :, :, :, 1] * dz[:, :, :, :, 0])

        det = det0 - det1 + det2
    return det

def neg_Jdet_loss(J):
    Jdet = JacboianDet(J)
    neg_Jdet = -1.0 * Jdet
    selected_neg_Jdet = F.relu(neg_Jdet)
    return selected_neg_Jdet ** 2

def grad_loss(f):
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

def magnitude_loss(all_v):
    if len(all_v.shape) == 5:
        all_v_x_2 = all_v[:, 0, :, :, :] * all_v[:, 0, :, :, :]
        all_v_y_2 = all_v[:, 1, :, :, :] * all_v[:, 1, :, :, :]
        all_v_z_2 = all_v[:, 2, :, :, :] * all_v[:, 2, :, :, :]
        # all_v_magnitude = torch.mean(all_v_x_2 + all_v_y_2 + all_v_z_2)
        magnitude = all_v_x_2 + all_v_y_2 + all_v_z_2
    else:
        all_v_x_2 = all_v[:, 0, :, :] * all_v[:, 0, :, :]
        all_v_y_2 = all_v[:, 1, :, :] * all_v[:, 1, :, :]
        # all_v_magnitude = torch.mean(all_v_x_2 + all_v_y_2)
        magnitude = all_v_x_2 + all_v_y_2
    return magnitude