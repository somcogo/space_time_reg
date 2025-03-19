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


    models = []
    for i in range(imgs.shape[0]):
        model = modules.SingleBVPNet(type='sine', mode='mlp', sidelength=imgs.shape[1:], device=config.device)
        model.load_state_dict(st_dicts[i])
        model.eval()
        models.append(model)

    # imgs = torch.stack([imgs[0], imgs[5]])
    # models = [models[0], models[5]]
    imgs = imgs[:5]
    models = models[:5]
    return imgs, models

def calculate_metrics(loss, phi, data):
    metrics = {}
    metrics['total_loss'] = loss
    return metrics

def log_metrics(metrics, phi, data, writer, epoch):
    for k, v in metrics.items():
        writer.add_scalar(k, v, epoch)

    imgs, neural_reps = data
    imgs = imgs.detach().cpu()
    # scale_factor = torch.tensor(imgs[0].shape).view(1, 1, 2, 1, 1)
    # new_locs = (phi + 1.) / 2. * scale_factor
    new_locs = phi.detach().cpu()

    moved_imgs = []
    for time in range(phi.shape[0]):
        grid = new_locs[time].permute(0, 2, 3, 1)
        # channel reversal for grid_sample
        # if grid.shape[-1] == 2:
        #     grid = grid[..., [1, 0]]
        # else:
        #     grid = grid[..., [2, 1, 0]]
        input_img = imgs[0].unsqueeze(0).unsqueeze(0)
        moved_imgs.append(F.grid_sample(input_img, grid, align_corners=True, mode='bilinear'))

    imgs_to_display = [imgs[0].unsqueeze(0), imgs[-1].unsqueeze(0), moved_imgs[0].squeeze(0), moved_imgs[-1].squeeze(0)]
    img_grid = make_grid(imgs_to_display, nrow=2, normalize=True)
    moved_grid = make_grid([im.squeeze(0) for im in moved_imgs], normalize=True)

    writer.add_image('imgs/imgs', img_grid, epoch, dataformats='CHW', )
    writer.add_image('imgs/flow', moved_grid, epoch, dataformats='CHW')


    moved_nrep_imgs = []
    net = neural_reps[-1].cpu()
    for time in range(phi.shape[0]):
        grid = new_locs[time].squeeze(0).permute(2, 1, 0)
        moved_nrep_imgs.append(net.net(grid).squeeze(2).detach().cpu())

    imgs_to_display_nrep = [imgs[0].unsqueeze(0), imgs[-1].unsqueeze(0), moved_nrep_imgs[0].unsqueeze(0), moved_nrep_imgs[-1].unsqueeze(0)]
    img_grid_nrep = make_grid(imgs_to_display_nrep, nrow=2, normalize=True)
    moved_grid_nrep = make_grid([im.unsqueeze(0) for im in moved_nrep_imgs], normalize=True)

    writer.add_image('nrep/imgs', img_grid_nrep, epoch, dataformats='CHW', )
    writer.add_image('nrep/flow', moved_grid_nrep, epoch, dataformats='CHW')
    
    

    

def save_results(config, phi):
    save_path = os.path.join(config.log_path, 'res.pt')
    save_dict = {'phi':phi.detach().cpu()}
    torch.save(save_dict, save_path)
