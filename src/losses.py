import torch
from torch import nn
import torch.nn.functional as F

from src.siren.dataio import get_mgrid

def calculate_losses(config, phi, data):
    imgs, neural_reps = data
    grid = get_mgrid(imgs[0].shape, dim=len(imgs.shape) - 1).view(phi.shape[1:])
    coords = phi + grid

    moving = imgs[0]
    fixed_n_rep = neural_reps[-1]
    coord = coords[1]
    loss_sim = similarity_loss(moving, fixed_n_rep, coord)
    loss_negJ = config.lambda_negJ * neg_Jdet_loss(coords[-1])
    loss_smt = config.lambda_smt * smoothloss_loss(phi[-1])
    loss_mag = config.lambda_mag * magnitude_loss(phi[1:] - phi[:-1])
    loss = loss_sim + loss_negJ + loss_smt + loss_mag

    return loss

def similarity_loss(moving, fixed_n_rep, coords):
    loss_fn = nn.MSELoss()
    model_out = fixed_n_rep.net(coords.squeeze(0).permute(1, 2, 0))
    fixed = model_out.squeeze(2)
    l2loss = loss_fn(fixed, moving)
    return l2loss


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
        dx = phi[:, 1:, :-1, :] - phi[:, :-1, :-1, :]
        dy = phi[:, :-1, 1:, :] - phi[:, :-1, :-1, :]

        det = dx[:, :, :, 0] * dy[:, :, :, 1] - dx[:, :, :, 1] * dy[:, :, :, 0]
    else:
        dx = phi[:, 1:, :-1, :-1, :] - phi[:, :-1, :-1, :-1, :]
        dy = phi[:, :-1, 1:, :-1, :] - phi[:, :-1, :-1, :-1, :]
        dz = phi[:, :-1, :-1, 1:, :] - phi[:, :-1, :-1, :-1, :]

        det0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
        det1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 0])
        det2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy[:, :, :, :, 1] * dz[:, :, :, :, 0])

        det = det0 - det1 + det2
    return det

def neg_Jdet_loss(J):
    Jdet = JacboianDet(J)
    neg_Jdet = -1.0 * (Jdet - 0.5)
    selected_neg_Jdet = F.relu(neg_Jdet)
    return torch.mean(selected_neg_Jdet ** 2)

def smoothloss_loss(df):
    if len(df.shape) == 5:
        gradient_magnitude = (((df[:, :, 1:, :, :] - df[:, :, :-1, :, :]) ** 2).mean() + \
     ((df[:, :, :, 1:, :] - df[:, :, :, :-1, :]) ** 2).mean() + \
     ((df[:, :, :, :, 1:] - df[:, :, :, :, :-1]) ** 2).mean())
    else:
        gradient_magnitude = (((df[:, :, 1:, :] - df[:, :, :-1, :]) ** 2).mean() + \
     ((df[:, :, :, 1:] - df[:, :, :, :-1]) ** 2).mean())
    return gradient_magnitude

def magnitude_loss(all_v):
    if len(all_v.shape) == 6:
        all_v_x_2 = all_v[:, :, 0, :, :, :] * all_v[:, :, 0, :, :, :]
        all_v_y_2 = all_v[:, :, 1, :, :, :] * all_v[:, :, 1, :, :, :]
        all_v_z_2 = all_v[:, :, 2, :, :, :] * all_v[:, :, 2, :, :, :]
        all_v_magnitude = torch.mean(all_v_x_2 + all_v_y_2 + all_v_z_2)
    else:
        all_v_x_2 = all_v[:, :, 0, :, :] * all_v[:, :, 0, :, :]
        all_v_y_2 = all_v[:, :, 1, :, :] * all_v[:, :, 1, :, :]
        all_v_magnitude = torch.mean(all_v_x_2 + all_v_y_2)
    return all_v_magnitude