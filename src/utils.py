from functools import partial
import logging
import logging.handlers
import os

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
import numpy as np
from flow_vis import flow_to_color

from src.siren import training, dataio, modules, loss_functions


def get_logger(level):
    logger = logging.getLogger()
    if level == 'info':
        level = logging.INFO
    else:
        level = logging.DEBUG

    logfmt_str = "%(asctime)s %(levelname)s %(message)s"
    formatter = logging.Formatter(logfmt_str)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    streamHandler.setLevel(level)

    logger.addHandler(streamHandler)
    logger.setLevel(level)
    return logger

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
        
class SingleImgDataset(Dataset):
    def __init__(self, img):
        super().__init__()
        self.img = img
        self.img_channels = 1

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img

def generate_grid_tensor(shape):
    if len(shape) == 3:
        x_grid = torch.linspace(-1., 1., shape[0])
        y_grid = torch.linspace(-1., 1., shape[1])
        z_grid = torch.linspace(-1., 1., shape[2])
        x_grid, y_grid, z_grid = torch.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

        # Note that default the dimension in the grid is reversed:
        # z, y, x
        grid = torch.stack([z_grid, y_grid, x_grid], dim=0).unsqueeze(0)
    else:
        x_grid = torch.linspace(-1., 1., shape[0])
        y_grid = torch.linspace(-1., 1., shape[1])
        x_grid, y_grid = torch.meshgrid(x_grid, y_grid, indexing='ij')

        # Note that default the dimension in the grid is reversed:
        # y, x
        grid = torch.stack([y_grid, x_grid], dim=0).unsqueeze(0)

    return grid

def generate_coord_tensor(dims, device):
    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(len(dims))]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=-1)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), len(dims)]).to(device)
    return coordinate_tensor

def prepare_inputs(config):
    if config.dataset == 'easysyn':
        imgs = torch.from_numpy(np.load('data/syn/easy.npy').transpose((2, 0, 1)))
        st_dicts = torch.load('data/syn/easy_nrep_st_dicts.pt')
    elif config.dataset == 'rectri':
        imgs = torch.from_numpy(np.load('data/syn/rectri.npy').transpose((2, 0, 1)))
        st_dicts = torch.load('data/syn/rectri_nrep_st_dicts.pt')
    elif config.dataset == 'hard':
        imgs = torch.from_numpy(np.load('data/syn/hard.npy').transpose((2, 0, 1)))
        st_dicts = torch.load('data/syn/hard_nrep_st_dicts.pt')
    elif config.dataset == 'rot':
        imgs = torch.from_numpy(np.load('data/syn/rot.npy').transpose((2, 0, 1)))
        st_dicts = torch.load('data/syn/rot_nrep_st_dicts.pt')
    elif config.dataset == 'rot_slow':
        imgs = torch.from_numpy(np.load('data/syn/rot_slow.npy').transpose((2, 0, 1)))
        st_dicts = torch.load('data/syn/rot_slow_nrep_st_dicts.pt')
    elif config.dataset == 'rot_slow2':
        imgs = torch.from_numpy(np.load('data/syn/rot_slow2.npy').transpose((2, 0, 1)))
        st_dicts = torch.load('data/syn/rot_slow2_nrep_st_dicts.pt')
    elif config.dataset == 'rec':
        imgs = torch.from_numpy(np.load('data/syn/rec.npy').transpose((2, 0, 1)))
        st_dicts = torch.load('data/syn/rec_nrep_st_dicts.pt')


    models = []
    for i in range(len(st_dicts)):
        model = modules.SingleBVPNet(type='sine', mode='mlp', sidelength=imgs.shape[1:], device=config.device)
        model.load_state_dict(st_dicts[i])
        model.eval()
        models.append(model)

    # imgs = torch.stack([imgs[0], imgs[5]])
    # models = [models[0], models[5]]
    # imgs = imgs[:5]
    # models = models[:5]
    imgs = imgs[:config.time_points]
    models = models[:config.time_points]
    return imgs, models

def calculate_metrics(losses, phi, data):
    metrics = {}
    metrics['losses/total_loss'] = sum(losses)
    metrics['losses/sim_loss'] = losses[0]
    metrics['losses/negJ_loss'] = losses[1]
    metrics['losses/smooth_loss'] = losses[2]
    metrics['losses/magnitude_loss'] = losses[3]
    return metrics

def log_metrics(metrics, phi, data, writer, epoch, moved_imgs, vel):
    for k, v in metrics.items():
        writer.add_scalar(k, v, epoch)

    imgs, neural_reps = data
    imgs = imgs.detach().cpu()

    imgs_to_display_nrep = [imgs[0].unsqueeze(0), imgs[-1].unsqueeze(0), moved_imgs[0].unsqueeze(0), moved_imgs[-1].unsqueeze(0)]
    img_grid_nrep = make_grid(imgs_to_display_nrep, nrow=2, normalize=True)
    moved_grid_nrep = make_grid([im.unsqueeze(0) for im in moved_imgs], nrow=5, normalize=True)

    writer.add_image('imgs/comparison', img_grid_nrep, epoch, dataformats='CHW', )
    writer.add_image('imgs/all_time', moved_grid_nrep, epoch, dataformats='CHW')

    points_st_dict = torch.load('data/syn/pts128_nrep_st_dicts.pt')[0]
    points_nrep = modules.SingleBVPNet(type='sine', mode='mlp', sidelength=imgs.shape[1:], device=phi.device)
    points_nrep.load_state_dict(points_st_dict)
    points_nrep = points_nrep.to(phi.device)
    points_out = points_nrep.net(phi)
    points_out = (points_out.permute(0, 2, 1).reshape(phi.shape[0], 1, imgs.shape[1], imgs.shape[2]) + 1) / 2
    points_grid = make_grid(points_out, nrow=5)

    phi = phi.detach().cpu()
    rel_phi = phi.detach().cpu()
    coord_tensor = generate_coord_tensor(imgs.shape[1:], device='cpu')
    rel_phi_shape = list(imgs.shape) + [len(imgs.shape) - 1]
    rel_phi = (rel_phi-coord_tensor).reshape(rel_phi_shape).numpy()
    abs_phi = phi.reshape(rel_phi_shape).numpy()
    rel_flow_colors = []
    abs_flow_colors = []
    for time in range(phi.shape[0]):
        rel_flow_colors.append(torch.from_numpy(flow_to_color(rel_phi[time], convert_to_bgr=False)).permute(2, 0, 1))
        abs_flow_colors.append(torch.from_numpy(flow_to_color(abs_phi[time], convert_to_bgr=False)).permute(2, 0, 1))
    rel_flow_grid = make_grid(rel_flow_colors, nrow=5)
    abs_flow_grid = make_grid(abs_flow_colors, nrow=5)

    vel_shape = rel_phi_shape[1:]
    vel = vel.detach().cpu().reshape(vel_shape).numpy()
    act_velocity_color = torch.from_numpy(flow_to_color(vel, convert_to_bgr=False)).permute(2, 0, 1)
    diff_v = rel_phi[-1] - rel_phi[-2]
    diff_velocity_color = torch.from_numpy(flow_to_color(diff_v, convert_to_bgr=False)).permute(2, 0, 1)

    writer.add_image('imgs/rel_flow_all', rel_flow_grid, epoch, dataformats='CHW', )
    writer.add_image('imgs/abs_flow_all', abs_flow_grid, epoch, dataformats='CHW', )
    writer.add_image('imgs/velocity', act_velocity_color, epoch, dataformats='CHW', )
    writer.add_image('imgs/phi_diff_vel', diff_velocity_color, epoch, dataformats='CHW', )
    writer.add_image('imgs/points_moved', points_grid.detach().cpu().numpy(), epoch, dataformats='CHW', )


    
def save_results(config, output):
    save_path = os.path.join(config.log_path, 'res.pt')
    save_dict = {'phi':output[0].detach().cpu(),
                 'vel':output[1].detach().cpu(),
                 'config':vars(config)}
    torch.save(save_dict, save_path)
