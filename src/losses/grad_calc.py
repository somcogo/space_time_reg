import torch

from src.losses.normalized_gradient_field import spatial_filter_nd, _grad_param

def get_Laplacian(rel_vel, func, coord_tensor, shape, config, time_series):
    if config.fin_diff_grad:
        vel_reshaped = rel_vel.reshape(shape)
        J = fin_diff_Jacobian(vel_reshaped)
        lap = fin_Laplacian_from_Jac(J)
    elif config.autograd_grid:
        lap = get_autograd_Laplacian(func, time_series, dims=shape[-1], coord_tensor=coord_tensor)
    else:
        lap = get_autograd_Laplacian(func, time_series, dims=shape[-1], coord_tensor=None)
    return lap

def fin_diff_Jacobian(f):
    partial_grads = [fin_diff_gradient(f, dim) for dim in range(len(f.shape)-2)]
    J = torch.stack(partial_grads, dim=-1)
    # J shape: [T or 1, H, W, f_dim, spatial_dim]
    return J

def fin_Laplacian_from_Jac(J):
    lap = sum([fin_diff_gradient(J[..., i], i) for i in range(J.shape[-1])])
    return lap

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

def get_autograd_Jacobian(func, time_series, dims, coord_tensor=None):
    if coord_tensor is not None:
        coords = coord_tensor
    else:
        coords = torch.rand((1024, dims), requires_grad=True, device=time_series.device)*2 - 1
    rel_vel = []
    for t in time_series:
        vel = func(t, coords)
        rel_vel.append(vel)
    rel_vel = torch.concat(rel_vel)
    # J shape: [T or 1, N, f_dim, spatial_dim]
    return torch.stack([gradient(input=coords, output=rel_vel[:, i]) for i in range(dims)], dim=1)

def get_autograd_Laplacian(func, time_series, dims, coord_tensor=None):
    if coord_tensor is not None:
        coords = coord_tensor
    else:
        coords = torch.rand((1024, dims), requires_grad=True, device=time_series.device)*2 - 1
    rel_vel = []
    for t in time_series:
        vel = func(t, coords)
        rel_vel.append(vel)
    rel_vel = torch.concat(rel_vel) # [T or 1, N, f_dim]
    # J shape: [N, f_dim, spatial_dim]
    J = torch.stack([gradient(input=coords, output=rel_vel[:, i]) for i in range(dims)], dim=1)

    dxdxy = [gradient(input=coords, output=J[..., i, 0]) for i in range(dims)]
    ddxx = torch.stack([dxy[..., 0] for dxy in dxdxy], dim=-1)
    dydxy = [gradient(input=coords, output=J[..., i, 1]) for i in range(dims)]
    ddyy = torch.stack([dxy[..., 1] for dxy in dydxy], dim=-1)
    return ddxx + ddyy

def get_autograd_phi_Jacobian(phi, img_shape, coords):
    return torch.stack([gradient(input=coords, output=phi[-1][..., i]) for i in range(len(img_shape))], dim=1)

def gradient(input, output, grad_outputs=None):
    """Compute the gradient of the output wrt the input."""

    grad_outputs = torch.ones_like(output)
    grad = torch.autograd.grad(
        output, [input], grad_outputs=grad_outputs, create_graph=True
    )[0]
    return grad