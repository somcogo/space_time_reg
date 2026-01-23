import copy
from functools import partial
import logging
import time
import random

import torch
from torch import nn
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from fastmri import fft2c, ifft2c
from argparse import Namespace
from deepinv.optim.data_fidelity import L2Distance

from src.models.siren import Siren
from src.siren import training, dataio, modules, loss_functions
from src.utils.spatial_utils import generate_coord_tensor
from src.losses.recon_reg import get_recon_regularizer
from learned_regularizers.evaluation.nmAPG import nmAPG

# Based on reconstruct_nmAPG from learned_regularizers https://github.com/johertrich/LearnedRegularizers/blob/main/evaluation/nmAPG.py
def reconstruct_initial_frame_learned_reg(config: Namespace, recon: torch.Tensor, gt: torch.Tensor, forw: nn.Module):
    x0 = recon
    gt = gt.to(config.device)
    loss_fn = L2Distance()
    regularizer = get_recon_regularizer(config)
    def energy(val, y_in):
        with torch.no_grad():
            sim_loss = config.lambda_st * loss_fn(forw(val), y_in)
            reg_loss = config.lambda_init_recon * regularizer.g(val.flatten(0,1).unsqueeze(1)).reshape(val.shape[0], -1).sum(1)
            fun = sim_loss + reg_loss
        if config.detach_grads:
            fun = fun.detach()
        return fun.reshape(-1)
    
    def energy_grad(val, y_in):
        sim_grad = config.lambda_st * calc_sim_grad(loss_fn, forw, val, y_in)
        reg_grad = config.lambda_init_recon * regularizer.grad(val.flatten(0,1).unsqueeze(1)).reshape(val.shape)
        grad = sim_grad + reg_grad
        if config.detach_grads:
            grad = grad.detach()
        return grad

    energy_and_grad = lambda val, y_in: (energy(val, y_in), energy_grad(val, y_in))
    
    t0 = time.time()
    x, L, i, converged = nmAPG(x0=x0,
                               y=gt,
                               f=energy,
                               nabla=energy_grad,
                               f_and_nabla=energy_and_grad,
                               max_iter=config.recon_epochs,
                               verbose=config.debug,
                               tol=1e-6)
    t1 = time.time()
    print(f'Finished initial reconstruction in {t1-t0:.4f} seconds')
    return x

def calc_sim_grad(loss_fn, forw, val, y_in):
    with torch.enable_grad():
        val = val.requires_grad_()
        h = loss_fn(forw(val), y_in)
        grad = torch.autograd.grad(
            h, val, torch.ones_like(h), create_graph=True
        )[0]
    return grad


def reconstruct_initial_frame(logger: logging.Logger, config: Namespace, recon: nn.Parameter, gt: torch.Tensor, forw: nn.Module):
    optimizer = torch.optim.Adam([recon], lr=config.init_lr)
    # recon = recon.unsqueeze(0).to(config.device)
    gt = gt.to(config.device)
    loss_fn = nn.MSELoss(reduction='mean')
    regularizer = get_recon_regularizer(config)
    best_loss = 1e8
    t0 = time.time()
    for epoch in range(1, config.recon_epochs + 1):
        optimizer.zero_grad()
        sim_loss = config.lambda_st * loss_fn(forw(recon), gt)
        reg_loss = config.lambda_init_recon * regularizer.g(recon.flatten(0,1).unsqueeze(1)).mean()
        loss_sum = sim_loss + reg_loss
        loss_sum.backward()
        optimizer.step()
        if loss_sum <= best_loss:
            best_recon = recon.detach().clone()
            best_loss = loss_sum.detach().clone()
    t1 = time.time()
    if logger is not None:
        logger.info(f'Finished initial reconstruction in {t1-t0:.4f} seconds')
    return best_recon

def fit_neural_reps(data, args):
    n_reps = []
    for time_point in range(data.shape[0]):
        lr = 1e-4
        num_epochs = 10000
        steps_til_summary = 1000
        dset = SingleImgDataset(Image.fromarray(data[time_point]))
        if len(data.shape) == 3:
            coord_dataset = dataio.Implicit2DWrapper(dset, sidelength=data.shape[1:], compute_diff='all')
        else:
            coord_dataset = dataio.Implicit3DWrapper(dset, sidelength=data.shape[1:], compute_diff='all')

        dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

        model = modules.SingleBVPNet(type='sine', mode='mlp', sidelength=data.shape[1:], device=args.device)
        model.cuda()

        loss_fn = partial(loss_functions.image_mse, None)

        n_rep = training.train(model=model, train_dataloader=dataloader, epochs=num_epochs, lr=lr,
                    steps_til_summary=steps_til_summary, loss_fn=loss_fn, device=args.device)
        n_reps.append(n_rep)
    return n_reps

def fit_siren_to_img(img, num_epochs=2000, lr=1e-4, device='cuda', layers=[2, 256, 256, 256, 1], min_coord=-1, max_coord=1, omega=30):
    img_shape = img.shape
    rep = Siren(layers=layers, omega=omega)
    rep.to(device)
    optim = Adam(params=rep.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    coord_tensor = generate_coord_tensor(img_shape, device, min_coord=min_coord, max_coord=max_coord)
    gt = torch.from_numpy(img).to(device)

    min_loss = 1e10
    best_epoch = 0
    for epoch in range(num_epochs):
        pred = rep(torch.tensor([], device=device), coord_tensor)
        pred = pred.reshape(img_shape)
        loss = loss_fn(pred, gt.float())
        optim.zero_grad()
        loss.backward()
        optim.step()
        if loss < min_loss:
            min_loss = loss
            best_st_dict = rep.state_dict()
            best_epoch = epoch
        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, curr loss {loss}, best loss {min_loss} from epoch {best_epoch}')
    best_nrep = copy.deepcopy(rep)
    best_nrep.load_state_dict(best_st_dict)

    return best_nrep, best_st_dict

def fit_siren_to_flow(flow, num_epochs=2000, lr=1e-4, device='cuda', layers=[3, 256, 256, 256, 3], min_coord=-1, max_coord=1, omega=30):
    flow_shape = flow.shape
    rep = Siren(layers=layers, omega=omega)
    rep.to(device)
    optim = Adam(params=rep.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    coord_tensor = generate_coord_tensor(flow_shape[:-1], device, min_coord=min_coord, max_coord=max_coord)
    gt = torch.from_numpy(flow).to(device)

    min_loss = 1e10
    best_epoch = 0
    for epoch in range(num_epochs):
        pred = rep(torch.tensor([], device=device), coord_tensor)
        pred = pred.reshape(flow_shape)
        loss = loss_fn(pred, gt.float())
        optim.zero_grad()
        loss.backward()
        optim.step()
        if loss < min_loss:
            min_loss = loss
            best_st_dict = rep.state_dict()
            best_epoch = epoch
        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, curr loss {loss}, best loss {min_loss} from epoch {best_epoch}')
    best_nrep = copy.deepcopy(rep)
    best_nrep.load_state_dict(best_st_dict)

    return best_nrep, best_st_dict
        
class SingleImgDataset(Dataset):
    def __init__(self, img):
        super().__init__()
        self.img = img
        self.img_channels = 1

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img
    
class FastmriFT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image: torch.Tensor):
        spectrum = fft2c(image.movedim(1, -1)).movedim(-1, 1)
        return spectrum
    
class FastmriIFT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, spectrum: torch.Tensor):
        image = ifft2c(spectrum.movedim(1, -1)).movedim(-1, 1)
        return image
    
class FTAndSubsample(nn.Module):
    def __init__(self, kspace_mask):
        super().__init__()
        self.register_buffer("mask", kspace_mask)

    def forward(self, image: torch.Tensor):
        spectrum = fft2c(image.movedim(1, -1)).movedim(-1, 1)
        new_shape = list(image.shape[:2]) + [-1] + list(image.shape[3:])
        return spectrum[self.mask].reshape(new_shape)
    
class ZeroFillAndIFT(nn.Module):
    def __init__(self, kspace_mask):
        super().__init__()
        self.register_buffer("mask", kspace_mask)

    def forward(self, spectrum: torch.Tensor):
        full_spectrum = torch.zeros(self.mask.shape, device=spectrum.device)
        full_spectrum[self.mask] = spectrum.flatten()
        image = ifft2c(full_spectrum.movedim(1, -1)).movedim(-1, 1)
        return image
    
def generate_standard_mask(shape: torch.Size, factor: int):
    h = shape[-2]
    start = h//2 - 12
    end = h//2 + 12

    set1 = set(range(start, end))
    set2 = set(range(0, h, factor))
    indices = list(set1.union(set2))

    mask = torch.zeros(shape, dtype=bool)
    mask[..., indices, :] = 1

    return mask

def generate_random_mask(shape: torch.Size, factor: int):
    h = shape[-2]
    start = h//2 - 12
    end = h//2 + 12

    set1 = set(range(start, end))
    set2 = set(random.sample(range(h), h//factor))
    indices = list(set1.union(set2))

    mask = torch.zeros(shape, dtype=bool)
    mask[..., indices, :] = 1

    return mask

def get_kspace_mask(config, kspace_data, factor):
    if config.random_mask:
        mask = generate_random_mask(kspace_data.shape, factor)
    else:
        mask = generate_standard_mask(kspace_data.shape, factor)
    return mask