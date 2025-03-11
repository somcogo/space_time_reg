import math

import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F
from torch import nn

class GaussianKernel(torch.nn.Module):
    def __init__(self, win=11, nsig=0.1, dim=3):
        super(GaussianKernel, self).__init__()
        self.win = win
        self.nsig = nsig
        self.dim = dim
        if dim == 3:
            kernel_x, kernel_y, kernel_z = self.gkern1D_xyz(self.win, self.nsig)
            kernel = kernel_x * kernel_y * kernel_z
        else:
            kernel_x, kernel_y= self.gkern1D_xy(self.win, self.nsig)
            kernel = kernel_x * kernel_y

        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)
        if dim == 3:
            self.register_buffer("kernel_z", kernel_z)
        self.register_buffer("kernel", kernel)

    def gkern1D(self, kernlen=None, nsig=None):
        '''
        :param nsig: large nsig gives more freedom(pixels as agents), small nsig is more fluid.
        :return: Returns a 1D Gaussian kernel.
        '''
        x = np.linspace(-nsig, nsig, kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kern1d = kern1d / kern1d.sum()
        return torch.tensor(kern1d, requires_grad=False).float()

    def gkern1D_xyz(self, kernlen=None, nsig=None):
        """Returns 3 1D Gaussian kernel on xyz direction."""
        kernel_1d = self.gkern1D(kernlen, nsig)
        kernel_x = kernel_1d.view(1, 1, -1, 1, 1)
        kernel_y = kernel_1d.view(1, 1, 1, -1, 1)
        kernel_z = kernel_1d.view(1, 1, 1, 1, -1)
        return kernel_x, kernel_y, kernel_z

    def gkern1D_xy(self, kernlen=None, nsig=None):
        """Returns 2 1D Gaussian kernel on xy direction."""
        kernel_1d = self.gkern1D(kernlen, nsig)
        kernel_x = kernel_1d.view(1, 1, -1, 1)
        kernel_y = kernel_1d.view(1, 1, 1, -1)
        return kernel_x, kernel_y

    def forward(self, x):
        pad = int((self.win - 1) / 2)
        # Apply Gaussian by 3D kernel
        if self.dim == 3:
            x = F.conv3d(x, self.kernel, padding=pad)
        else:
            x = F.conv2d(x, self.kernel, padding=pad)
        return x


class AveragingKernel(torch.nn.Module):
    def __init__(self, win=11, dim=3):
        super(AveragingKernel, self).__init__()
        self.win = win
        self.dim = dim

    def window_averaging(self, v):
        win_size = self.win
        v = v.double()

        half_win = int(win_size / 2)
        pad = [half_win + 1, half_win] * self.dim

        v_padded = F.pad(v, pad=pad, mode='constant', value=0)  # [x+pad, y+pad, z+pad]

        # Run the cumulative sum across all 3 dimensions
        v_cs_x = torch.cumsum(v_padded, dim=2)
        v_cs_xy = torch.cumsum(v_cs_x, dim=3)
        if self.dim == 3:
            v_cs_xyz = torch.cumsum(v_cs_xy, dim=4)
            x, y, z = v.shape[2:]
            v_win = v_cs_xyz[:, :, win_size:, win_size:, win_size:] \
                    - v_cs_xyz[:, :, win_size:, win_size:, :z] \
                    - v_cs_xyz[:, :, win_size:, :y, win_size:] \
                    - v_cs_xyz[:, :, :x, win_size:, win_size:] \
                    + v_cs_xyz[:, :, win_size:, :y, :z] \
                    + v_cs_xyz[:, :, :x, win_size:, :z] \
                    + v_cs_xyz[:, :, :x, :y, win_size:] \
                    - v_cs_xyz[:, :, :x, :y, :z]
            v_win = v_win / (win_size ** 3)
        else:
            x, y = v.shape[2:]
            v_win = v_cs_xy[:, :, win_size:, win_size:] \
                    - v_cs_xy[:, :, win_size:, :y] \
                    - v_cs_xy[:, :, :x, win_size:] \
                    + v_cs_xy[:, :, :x, :y]
            v_win = v_win / (win_size ** 2)

        v_win = v_win.float()
        return v_win

    def forward(self, v):
        return self.window_averaging(v)


class BrainNet(nn.Module):
    def __init__(self, img_sz, smoothing_kernel, smoothing_win, smoothing_pass, ds, bs):
        super(BrainNet, self).__init__()
        padding_mode = 'replicate'
        bias = True
        self.ds = ds
        self.bs = bs
        self.img_sz = img_sz
        self.dim = len(img_sz)
        self.smoothing_kernel = smoothing_kernel
        self.smoothing_pass = smoothing_pass

        conv_module = nn.Conv3d if self.dim == 3 else nn.Conv2d
        # self.enc_conv1 = conv_module(3, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv2 = conv_module(self.dim, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv3 = conv_module(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv4 = conv_module(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv5 = conv_module(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv6 = conv_module(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)

        if self.dim == 3:
            self.bottleneck_sz = int(
                math.ceil(img_sz[0] / pow(2, self.ds)) * math.ceil(img_sz[1] / pow(2, self.ds)) * math.ceil(
                    img_sz[2] / pow(2, self.ds)))
        else:
            self.bottleneck_sz = int(
                math.ceil(img_sz[0] / pow(2, self.ds)) * math.ceil(img_sz[1] / pow(2, self.ds)))
        
        if self.dim == 3:
            self.downs_sz = 32 * math.ceil(self.img_sz[0] / 64) * math.ceil(self.img_sz[1] / 64) * math.ceil(self.img_sz[2] / 64)
        else:
            self.downs_sz = 32 * math.ceil(self.img_sz[0] / 64) * math.ceil(self.img_sz[1] / 64)
        
        self.lin1 = nn.Linear(self.downs_sz, self.bs, bias=bias)
        self.lin2 = nn.Linear(self.bs, self.bottleneck_sz * self.dim, bias=bias)
        self.relu = nn.ReLU()

        # Create smoothing kernels
        if self.smoothing_kernel == 'AK':
            self.sk = AveragingKernel(win=smoothing_win, dim=self.dim)
        else:
            self.sk = GaussianKernel(win=smoothing_win, nsig=0.1, dim=self.dim)

    def forward(self, t, x):

        imgx = self.img_sz[0]
        imgy = self.img_sz[1]
        if self.dim == 3:
            imgz = self.img_sz[2]
        # x = self.relu(self.enc_conv1(x))
        interp_mode = 'trilinear' if self.dim == 3 else 'bilinear'
        x = F.interpolate(x, scale_factor=0.5, mode=interp_mode)  # Optional to downsample the image
        x = self.relu(self.enc_conv2(x))
        x = self.relu(self.enc_conv3(x))
        x = self.relu(self.enc_conv4(x))
        x = self.relu(self.enc_conv5(x))
        x = self.enc_conv6(x)
        x = x.view(-1)
        x = self.relu(self.lin1(x))
        x = self.lin2(x)

        if self.dim == 3:
            x = x.view(1, 3, int(math.ceil(imgx / pow(2, self.ds))), int(math.ceil(imgy / pow(2, self.ds))),
                   int(math.ceil(imgz / pow(2, self.ds))))
        else:
            x = x.view(1, 2, int(math.ceil(imgx / pow(2, self.ds))), int(math.ceil(imgy / pow(2, self.ds))))
        for _ in range(self.ds):
            x = F.upsample(x, scale_factor=2, mode=interp_mode)
        # Apply Gaussian/Averaging smoothing
        for _ in range(self.smoothing_pass):
            if self.smoothing_kernel == 'AK':
                x = self.sk(x)
            else:
                if self.dim == 3:
                    x_x = self.sk(x[:, 0, :, :, :].unsqueeze(1))
                    x_y = self.sk(x[:, 1, :, :, :].unsqueeze(1))
                    x_z = self.sk(x[:, 2, :, :, :].unsqueeze(1))
                    x_xyz = [x_x, x_y, x_z]
                else:
                    x_x = self.sk(x[:, 0, :, :].unsqueeze(1))
                    x_y = self.sk(x[:, 1, :, :].unsqueeze(1))
                    x_xyz = [x_x, x_y]
                x = torch.cat(x_xyz, 1)
        return x
    
def get_func(func_name, network_kwargs):
    if func_name == 'nodeo':
        func = BrainNet(**network_kwargs)

    return func