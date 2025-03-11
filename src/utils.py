import torch

def generate_grid_tensor(shape):
    if len(shape) == 3:
        x_grid = torch.linspace(-1., 1., shape[0])
        y_grid = torch.linspace(-1., 1., shape[1])
        z_grid = torch.linspace(-1., 1., shape[2])
        x_grid, y_grid, z_grid = torch.meshgrid(x_grid, y_grid, z_grid)

        # Note that default the dimension in the grid is reversed:
        # z, y, x
        grid = torch.stack([z_grid, y_grid, x_grid], dim=0).unsqueeze(0)
    else:
        x_grid = torch.linspace(-1., 1., shape[0])
        y_grid = torch.linspace(-1., 1., shape[1])
        x_grid, y_grid = torch.meshgrid(x_grid, y_grid)

        # Note that default the dimension in the grid is reversed:
        # y, x
        grid = torch.stack([y_grid, x_grid], dim=0).unsqueeze(0)

    return grid

def prepare_inputs(config, output):
    pass

def save_results(config, output):
    pass