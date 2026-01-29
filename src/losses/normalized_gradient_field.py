from __future__ import absolute_import
from typing import List, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.nn.parameter import Parameter
from monai.utils.enums import LossReduction

_func_conv_nd_table = {
    1: F.conv1d,
    2: F.conv2d,
    3: F.conv3d
}
# https://github.com/BailiangJ/normalized_gradient_field_pytorch/blob/master/normalized_gradient_field.py

def spatial_filter_nd(x: torch.Tensor, kernel: torch.Tensor, mode: str = 'replicate') -> torch.Tensor:
    """ N-dimensional spatial filter with padding.

    Args:
        x: the shape should be BNH[WD]
        kernel: Weight tensor (e.g., Gaussain kernel, Gradient kernel).
        mode (str, optional): Padding mode. Defaults to 'replicate'.

    Returns:
        torch.Tensor: Output tensor
    """

    n_dim = x.dim() - 2
    if n_dim <= 0 or n_dim > 3:
        raise AssertionError(f"the spatial dims of input should be 1, 2 or 3, get{n_dim}")
    conv = _func_conv_nd_table[n_dim]

    pad = [None, None] * n_dim
    pad[0::2] = kernel.shape[2:]
    pad[1::2] = kernel.shape[2:]
    pad = [k // 2 for k in pad]

    return conv(F.pad(x, pad=pad, mode=mode), kernel)


# NOTE: Gaussian kernel
def _gauss_1d(x, mu, sigma):
    return 1. / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))


def gauss_kernel_1d(sigma, truncate=4.0):
    sd = float(sigma)
    lw = int(truncate * sd + 0.5)
    x = np.arange(-lw, lw + 1)
    kernel_1d = _gauss_1d(x, 0., sigma)
    return kernel_1d / kernel_1d.sum()


def gauss_kernel_2d(sigma, truncate=4.0):
    sd = float(sigma)
    lw = int(truncate * sd + 0.5)
    x = y = np.arange(-lw, lw + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    kernel_2d = _gauss_1d(X, 0., sigma) \
                * _gauss_1d(Y, 0., sigma)
    return kernel_2d / kernel_2d.sum()


def gauss_kernel_3d(sigma, truncate=4.0):
    sd = float(sigma)
    lw = int(truncate * sd + 0.5)
    x = y = z = np.arange(-lw, lw + 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    kernel_3d = _gauss_1d(X, 0., sigma) \
                * _gauss_1d(Y, 0., sigma) \
                * _gauss_1d(Z, 0., sigma)
    return kernel_3d / kernel_3d.sum()


# NOTE: Average kernel
def _average_kernel_nd(ndim, kernel_size):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * ndim

    kernel_nd = np.ones(kernel_size)
    kernel_nd /= np.sum(kernel_nd)

    return kernel_nd


def average_kernel_1d(kernel_size):
    return _average_kernel_nd(1, kernel_size)


def average_kernel_2d(kernel_size):
    return _average_kernel_nd(2, kernel_size)


def average_kernel_3d(kernel_size):
    return _average_kernel_nd(3, kernel_size)


# NOTE: Gradient kernel
def gradient_kernel_1d(method='default'):
    if method == 'default':
        kernel_1d = np.array([-1, 0, +1])
    else:
        raise ValueError('unsupported method (got {})'.format(method))

    return kernel_1d


def gradient_kernel_2d(method='default', axis=0):
    if method == 'default':
        kernel_2d = np.array([[0, -1, 0],
                              [0, 0, 0],
                              [0, +1, 0]])
    elif method == 'sobel':
        kernel_2d = np.array([[-1, -2, -1],
                              [0, 0, 0],
                              [+1, +2, +1]])
    elif method == 'prewitt':
        kernel_2d = np.array([[-1, -1, -1],
                              [0, 0, 0],
                              [+1, +1, +1]])
    elif method == 'isotropic':
        kernel_2d = np.array([[-1, -np.sqrt(2), -1],
                              [0, 0, 0],
                              [+1, +np.sqrt(2), +1]])
    else:
        raise ValueError('unsupported method (got {})'.format(method))

    return np.moveaxis(kernel_2d, 0, axis)


def gradient_kernel_3d(method='default', axis=0):
    if method == 'default':
        kernel_3d = np.array([[[0, 0, 0],
                               [0, -1, 0],
                               [0, 0, 0]],
                              [[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]],
                              [[0, 0, 0],
                               [0, +1, 0],
                               [0, 0, 0]]])
    elif method == 'sobel':
        kernel_3d = np.array([[[-1, -3, -1],
                               [-3, -6, -3],
                               [-1, -3, -1]],
                              [[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]],
                              [[+1, +3, +1],
                               [+3, +6, +3],
                               [+1, +3, +1]]])
    elif method == 'prewitt':
        kernel_3d = np.array([[[-1, -1, -1],
                               [-1, -1, -1],
                               [-1, -1, -1]],
                              [[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]],
                              [[+1, +1, +1],
                               [+1, +1, +1],
                               [+1, +1, +1]]])
    elif method == 'isotropic':
        kernel_3d = np.array([[[-1, -1, -1],
                               [-1, -np.sqrt(2), -1],
                               [-1, -1, -1]],
                              [[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]],
                              [[+1, +1, +1],
                               [+1, +np.sqrt(2), +1],
                               [+1, +1, +1]]])
    else:
        raise ValueError('unsupported method (got {})'.format(method))

    return np.moveaxis(kernel_3d, 0, axis)

def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return [x, x]


def _grad_param(ndim, method, axis):
    if ndim == 1:
        kernel = gradient_kernel_1d(method)
    elif ndim == 2:
        kernel = gradient_kernel_2d(method, axis)
    elif ndim == 3:
        kernel = gradient_kernel_3d(method, axis)
    else:
        raise NotImplementedError

    kernel = kernel.reshape(1, 1, *kernel.shape)
    return Parameter(torch.Tensor(kernel).float())


def _gauss_param(ndim, sigma, truncate):
    if ndim == 1:
        kernel = gauss_kernel_1d(sigma, truncate)
    elif ndim == 2:
        kernel = gauss_kernel_2d(sigma, truncate)
    elif ndim == 3:
        kernel = gauss_kernel_3d(sigma, truncate)
    else:
        raise NotImplementedError

    kernel = kernel.reshape(1, 1, *kernel.shape)
    return Parameter(torch.Tensor(kernel).float())


class NormalizedGradientField2d(_Loss):
    """
    Compute the normalized gradient fields defined in:
    Haber, Eldad, and Jan Modersitzki. "Intensity gradient based registration and fusion of multi-modal images."
    In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 726-733. Springer,
    Berlin, Heidelberg, 2006.

    Häger, Stephanie, et al. "Variable Fraunhofer MEVIS RegLib Comprehensively Applied to Learn2Reg Challenge."
    International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2020.

    Adopted from:
    https://github.com/yuta-hi/pytorch_similarity
    https://github.com/visva89/pTVreg/blob/master/mutils/My/image_metrics/metric_ngf.m
    """

    def __init__(self,
                 grad_method: str = 'default',
                 gauss_sigma: float = None,
                 gauss_truncate: float = 4.0,
                 eps: Optional[float] = 1e-5,
                 mm_spacing: Optional[Union[int, float, Tuple[int, ...], List[int]]] = None,
                 reduction: Union[LossReduction, str] = LossReduction.MEAN) -> None:
        """

        Args:
            grad_method: {'default', 'sobel', 'prewitt', 'isotropic'}
            type of gradient kernel. Defaults to 'default' (finite difference).
            gauss_sigma: standard deviation from Gaussian kernel. Defaults to None.
            gauss_truncate: trunncate the Gaussian kernel at this number of sd. Defaults to 4.0.
            eps_src: smooth constant for denominator in computing norm of source/moving gradient
            eps_tar: smooth constant for denominator in computing norm of target/fixed gradient
            mm_spacing: pixel spacing of input images
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
        """
        super().__init__(reduction=LossReduction(reduction).value)

        self.eps = eps

        if isinstance(mm_spacing, (int, float)):
            # self.dvol = mm_spacing ** 2
            self.mm_spacing = [mm_spacing] * 2
        if isinstance(mm_spacing, (list, tuple)):
            if len(mm_spacing) == 2:
                # self.dvol = np.prod(mm_spacing)
                self.mm_spacing = mm_spacing
            else:
                raise ValueError(f'expected length 2 spacing, got {mm_spacing}')

        self.grad_method = grad_method
        self.gauss_sigma = _pair(gauss_sigma)
        self.gauss_truncate = gauss_truncate

        self.grad_u_kernel = None
        self.grad_v_kernel = None

        self.gauss_kernel_x = None
        self.gauss_kernel_y = None

        self._initialize_params()
        self._freeze_params()

    def _initialize_params(self):
        self._initialize_grad_kernel()
        self._initialize_gauss_kernel()

    def _initialize_grad_kernel(self):
        self.grad_u_kernel = _grad_param(2, self.grad_method, axis=0)
        self.grad_v_kernel = _grad_param(2, self.grad_method, axis=1)

    def _initialize_gauss_kernel(self):
        if self.gauss_sigma[0] is not None:
            self.gauss_kernel_x = _gauss_param(2, self.gauss_sigma[0], self.gauss_truncate)
        if self.gauss_sigma[1] is not None:
            self.gauss_kernel_y = _gauss_param(2, self.gauss_sigma[1], self.gauss_truncate)

    def _check_type_forward(self, x: torch.Tensor):
        if x.dim() != 4:
            raise ValueError(f'expected 4D input (BCHW), (got {x.dim()}D input)')

    def _freeze_params(self):
        self.grad_u_kernel.requires_grad = False
        self.grad_v_kernel.requires_grad = False
        if self.gauss_kernel_x is not None:
            self.gauss_kernel_x.requires_grad = False
        if self.gauss_kernel_y is not None:
            self.gauss_kernel_y.requires_grad = False

    def forward(self, source, target) -> torch.Tensor:
        """

        Args:
            source: source/moving image, shape should be BCHW
            target: target/fixed image, shape should be BCHW

        Returns:
            ngf: normalized gradient field between source and target
        """

        self._check_type_forward(source)
        self._check_type_forward(target)
        self._freeze_params()

        # if source.shape[1] != target.shape[1]:
        #     source = torch.mean(source, dim=1, keepdim=True)
        #     target = torch.mean(target, dim=1, keepdim=True)

        # reshape
        b, c = source.shape[:2]
        spatial_shape = source.shape[2:]

        # [B*N, H, W]
        source = source.view(b * c, 1, *spatial_shape)
        target = target.view(b * c, 1, *spatial_shape)

        # smoothing
        if self.gauss_kernel_x is not None:
            source = spatial_filter_nd(source, self.gauss_kernel_x)
        if self.gauss_kernel_y is not None:
            target = spatial_filter_nd(target, self.gauss_kernel_y)

        # gradient
        src_grad_u = spatial_filter_nd(source, self.grad_u_kernel) * self.mm_spacing[0]
        src_grad_v = spatial_filter_nd(source, self.grad_v_kernel) * self.mm_spacing[1]

        tar_grad_u = spatial_filter_nd(target, self.grad_u_kernel) * self.mm_spacing[0]
        tar_grad_v = spatial_filter_nd(target, self.grad_v_kernel) * self.mm_spacing[1]

        if self.eps is None:
            with torch.no_grad():
                self.eps = torch.mean(torch.abs(tar_grad_u) + torch.abs(tar_grad_v))
        # print(self.eps, torch.abs(tar_grad_u)[torch.abs(tar_grad_u)<0.5].mean(), torch.abs(tar_grad_v)[torch.abs(tar_grad_v)<0.5].mean())

        # gradient norm
        src_grad_norm = src_grad_u ** 2 + src_grad_v ** 2 + self.eps ** 2
        tar_grad_norm = tar_grad_u ** 2 + tar_grad_v ** 2 + self.eps ** 2

        # nominator
        product = src_grad_u * tar_grad_u + src_grad_v * tar_grad_v + self.eps ** 2

        # denominator
        denom = src_grad_norm * tar_grad_norm

        # integrator
        ngf = 0.5 *  (1 - (product ** 2 / denom))
        # ngf = 1.0 - product ** 2 / denom
        # ngf = product**2 / denom

        # reshape back
        ngf = ngf.view(b, c, *spatial_shape)

        # integration
        # ngf = 0.5 * self.dvol * ngf
        # ngf = 0.5 * torch.sum(ngf, dim=(2, 3)) * self.dvol
        # ngf = 0.5 * torch.mean(ngf, dim=(2, 3)) * self.dvol

        # reduction
        if self.reduction == LossReduction.MEAN.value:
            ngf = torch.mean(ngf)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            ngf = torch.sum(ngf)  # sum over batch and channel dims
        elif self.reduction != LossReduction.NONE.value:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return  ngf


class NormalizedGradientField3d(_Loss):
    """
    Compute the normalized gradient fields defined in:
    Haber, Eldad, and Jan Modersitzki. "Intensity gradient based registration and fusion of multi-modal images."
    In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 726-733. Springer,
    Berlin, Heidelberg, 2006.

    Häger, Stephanie, et al. "Variable Fraunhofer MEVIS RegLib Comprehensively Applied to Learn2Reg Challenge."
    International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2020.

    Adopted from:
    https://github.com/yuta-hi/pytorch_similarity
    https://github.com/visva89/pTVreg/blob/master/mutils/My/image_metrics/metric_ngf.m
    """

    def __init__(self,
                 grad_method: str = 'default',
                 gauss_sigma: float = None,
                 gauss_truncate: float = 4.0,
                 eps: Optional[float] = 1e-5,
                 mm_spacing: Optional[Union[int, float, Tuple[int, ...], List[int]]] = None,
                 reduction: Union[LossReduction, str] = LossReduction.MEAN) -> None:
        """

        Args:
            grad_method: {'default', 'sobel', 'prewitt', 'isotropic'}
            type of gradient kernel. Defaults to 'default' (finite difference).
            gauss_sigma: standard deviation from Gaussian kernel. Defaults to None.
            gauss_truncate: trunncate the Gaussian kernel at this number of sd. Defaults to 4.0.
            eps_src: smooth constant for denominator in computing norm of source/moving gradient
            eps_tar: smooth constant for denominator in computing norm of target/fixed gradient
            mm_spacing: pixel spacing of input images
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
        """
        super().__init__(reduction=LossReduction(reduction).value)

        self.eps = eps

        if isinstance(mm_spacing, (int, float)):
            # self.dvol = mm_spacing ** 3
            self.mm_spacing = [mm_spacing] * 3
        if isinstance(mm_spacing, (list, tuple)):
            if len(mm_spacing) == 3:
                # self.dvol = np.prod(mm_spacing)
                self.mm_spacing = mm_spacing
            else:
                raise ValueError(f'expected length 2 spacing, got {mm_spacing}')

        self.grad_method = grad_method
        self.gauss_sigma = _pair(gauss_sigma)
        self.gauss_truncate = gauss_truncate

        self.grad_u_kernel = None
        self.grad_v_kernel = None
        self.grad_w_kernel = None

        self.gauss_kernel_x = None
        self.gauss_kernel_y = None

        self._initialize_params()
        self._freeze_params()

    def _initialize_params(self):
        self._initialize_grad_kernel()
        self._initialize_gauss_kernel()

    def _initialize_grad_kernel(self):
        self.grad_u_kernel = _grad_param(3, self.grad_method, axis=0)
        self.grad_v_kernel = _grad_param(3, self.grad_method, axis=1)
        self.grad_w_kernel = _grad_param(3, self.grad_method, axis=2)

    def _initialize_gauss_kernel(self):
        if self.gauss_sigma[0] is not None:
            self.gauss_kernel_x = _gauss_param(3, self.gauss_sigma[0], self.gauss_truncate)
        if self.gauss_sigma[1] is not None:
            self.gauss_kernel_y = _gauss_param(3, self.gauss_sigma[1], self.gauss_truncate)

    def _check_type_forward(self, x: torch.Tensor):
        if x.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(x.dim()))

    def _freeze_params(self):
        self.grad_u_kernel.requires_grad = False
        self.grad_v_kernel.requires_grad = False
        self.grad_w_kernel.requires_grad = False
        if self.gauss_kernel_x is not None:
            self.gauss_kernel_x.requires_grad = False
        if self.gauss_kernel_y is not None:
            self.gauss_kernel_y.requires_grad = False

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        self._check_type_forward(source)
        self._check_type_forward(target)
        self._freeze_params()

        # if source.shape[1] != target.shape[1]:
        #     source = torch.mean(source, dim=1, keepdim=True)
        #     target = torch.mean(target, dim=1, keepdim=True)

        # reshape
        b, c = source.shape[:2]
        spatial_shape = source.shape[2:]

        source = source.view(b * c, 1, *spatial_shape)
        target = target.view(b * c, 1, *spatial_shape)

        # smoothing
        if self.gauss_kernel_x is not None:
            source = spatial_filter_nd(source, self.gauss_kernel_x)
        if self.gauss_kernel_y is not None:
            target = spatial_filter_nd(target, self.gauss_kernel_y)

        # gradient
        src_grad_u = spatial_filter_nd(source, self.grad_u_kernel) * self.mm_spacing[0]
        src_grad_v = spatial_filter_nd(source, self.grad_v_kernel) * self.mm_spacing[1]
        src_grad_w = spatial_filter_nd(source, self.grad_w_kernel) * self.mm_spacing[2]

        tar_grad_u = spatial_filter_nd(target, self.grad_u_kernel) * self.mm_spacing[0]
        tar_grad_v = spatial_filter_nd(target, self.grad_v_kernel) * self.mm_spacing[1]
        tar_grad_w = spatial_filter_nd(target, self.grad_w_kernel) * self.mm_spacing[2]

        if self.eps is None:
            with torch.no_grad():
                self.eps = torch.mean(torch.abs(src_grad_u) + torch.abs(src_grad_v) + torch.abs(src_grad_w))

        # gradient norm
        src_grad_norm = src_grad_u ** 2 + src_grad_v ** 2 + src_grad_w ** 2 + self.eps ** 2
        tar_grad_norm = tar_grad_u ** 2 + tar_grad_v ** 2 + tar_grad_w ** 2 + self.eps ** 2

        # nominator
        product = src_grad_u * tar_grad_u + src_grad_v * tar_grad_v + src_grad_w * tar_grad_w + self.eps ** 2

        # denominator
        denom = src_grad_norm * tar_grad_norm

        # integrator
        ngf = 0.5 *  (1 - (product ** 2 / denom))
        # ngf = 1.0 - product ** 2 / denom
        # ngf = product**2 / denom

        # reshape back
        ngf = ngf.view(b, c, *spatial_shape)

        # integration
        # ngf = 0.5 * self.dvol * ngf
        # ngf = 0.5 * torch.sum(ngf, dim=(2, 3)) * self.dvol
        # ngf = 0.5 * torch.mean(ngf, dim=(2, 3)) * self.dvol

        # reduction
        if self.reduction == LossReduction.MEAN.value:
            ngf = torch.mean(ngf)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            ngf = torch.sum(ngf)  # sum over batch and channel dims
        elif self.reduction != LossReduction.NONE.value:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return ngf
    