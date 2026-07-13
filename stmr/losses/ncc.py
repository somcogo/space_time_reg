import torch
import torch.nn.functional as F


class NODEO_NCC(torch.nn.Module):
    """
    NCC with cumulative sum implementation for acceleration. local (over window) normalized cross correlation.
    """

    def __init__(self, win=21, eps=1e-5):
        super().__init__()
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
        return 1. - cc2.float()