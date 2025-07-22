from functools import partial
import logging
import os
import math
import io

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
import numpy as np
from flow_vis import flow_to_color
import matplotlib.pyplot as plt

from src.siren import training, dataio, modules, loss_functions
from src.networks import Siren


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

def generate_coord_tensor(dims, device, min_coord=-1, max_coord=1):
    coordinate_tensor = [torch.linspace(min_coord, max_coord, dims[i]) for i in range(len(dims))]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing='ij')
    coordinate_tensor = torch.stack(coordinate_tensor, dim=-1)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), len(dims)]).to(device)
    return coordinate_tensor

def prepare_inputs(config):
    if config.dataset in ['easy', 'hard', 'rectri', 'rot', 'rot_slow', 'rot_slow2', 'rec']:
        imgs = torch.from_numpy(np.load(f'data/syn/{config.dataset}/{config.dataset}.npy').transpose((2, 0, 1)))
        st_dicts = torch.load(f'data/syn/{config.dataset}/{config.dataset}_nrep_st_dicts.pt')
    elif config.dataset == 'test':
        imgs = torch.from_numpy(np.load(f'data/syn/rec/rec.npy').transpose((2, 0, 1)))
        st_dicts = torch.load(f'data/syn/rec/rec_nrep_st_dicts.pt')
        imgs = imgs[[0, 3]]
        st_dicts = [st_dicts[0], st_dicts[3]]
        config.time_points = 2
    elif config.dataset == 'const':
        imgs = torch.from_numpy(np.load(f'data/syn/rec/rec.npy').transpose((2, 0, 1)))
        st_dicts = torch.load(f'data/syn/rec/rec_nrep_st_dicts.pt')
        imgs = imgs[[0, 0]]
        st_dicts = [st_dicts[0], st_dicts[0]]
        config.time_points = 2
    elif config.dataset == 'rot_slow2_large':
        imgs = torch.from_numpy(np.load(f'data/syn/rot_slow2/rot_slow2.npy').transpose((2, 0, 1)))
        st_dicts = torch.load(f'data/syn/rot_slow2/rot_slow2_nrep_st_dicts_large.pt')
    elif config.dataset == 'rot_slow2_64':
        imgs = torch.from_numpy(np.load(f'data/syn/rot_slow2/rot_slow2.npy').transpose((2, 0, 1)))
        st_dicts = torch.load(f'data/syn/rot_slow2/rot_slow2_nrep_st_dicts_64x64x64.pt')

    models = []
    if config.dataset == 'rot_slow2_large':
        model = Siren([2, 256, 256, 256, 1])
        model.load_state_dict(st_dicts)
        model.eval()
        models = [model]
    elif config.dataset == 'rot_slow2_64':
        model = Siren([2, 64, 64, 64, 1])
        model.load_state_dict(st_dicts)
        model.eval()
        models = [model]
    else:
        for i in range(len(st_dicts)):
            model = modules.SingleBVPNet(type='sine', mode='mlp', sidelength=imgs.shape[1:], device=config.device)
            model.load_state_dict(st_dicts[i])
            model.eval()
            models.append(model)

    imgs = imgs[:config.time_points]
    models = models[:config.time_points]
    return imgs, models

def calculate_metrics(losses, config, abs_phi, rel_vel, data, moved_imgs, func, energies, collect_imgs, last_val=False):
    metrics = {}
    metrics['losses/total_loss'] = sum(losses)
    metrics['losses/sim_loss'] = losses[0]
    metrics['losses/negJ_loss'] = losses[1]
    # metrics['losses/smooth_loss'] = losses[2]
    metrics['losses/gradient_loss'] = losses[2]
    metrics['losses/laplacian_loss'] = losses[3]
    metrics['losses/phi_grad_loss'] = losses[4]

    if config.debug or last_val:
        grads = torch.tensor([p.grad.norm() for p in func.parameters()])
        names = [n for n, p in func.named_parameters()]
        metrics['grad_stats/mean_grad'] = grads.mean()
        metrics['grad_stats/min_grad'] = grads.min()
        metrics['grad_stats/max_grad'] = grads.max()

        for i in range(len(names)):
            metrics[f'all_grads/{names[i]}'] = grads[i]

        metrics['phi_stats/abs_max'] = abs_phi.max()
        metrics['phi_stats/abs_min'] = abs_phi.min()
        phi_11 = (abs_phi == torch.tensor([1, 1], device=abs_phi.device)).sum()
        phi_m1m1 = (abs_phi == torch.tensor([-1, -1], device=abs_phi.device)).sum()
        phi_1m1 = (abs_phi == torch.tensor([1, -1], device=abs_phi.device)).sum()
        phi_m11 = (abs_phi == torch.tensor([-1, 1], device=abs_phi.device)).sum()
        phi_boundary = phi_11 + phi_m1m1 + phi_1m1 + phi_m11
        all_phi = math.prod(list(abs_phi.shape))
        metrics['phi_stats/boundary_ratio'] = phi_boundary/all_phi
        metrics['phi_stats/boundary_absolute'] = phi_boundary
        metrics['vel_stats/rel_max'] = rel_vel.max()
        metrics['vel_stats/rel_min'] = rel_vel.min()
        metrics['vel_stats/rel_mean'] = rel_vel.mean()


    if collect_imgs:
        imgs_to_save = {}
        imgs, neural_reps = data
        imgs = imgs.detach().cpu()

        registration_last = make_grid([torch.stack([imgs[-1], torch.zeros_like(imgs[-1]), moved_imgs[-1]])], nrow=2, normalize=True)
        registration_all = make_grid([torch.stack([im, torch.zeros_like(im), m_im]) for im, m_im in zip(imgs, moved_imgs)], nrow=5, normalize=True)

        abs_phi = abs_phi.detach().cpu()
        coord_tensor = generate_coord_tensor(imgs.shape[1:], device='cpu')
        phi_shape = list(imgs.shape) + [len(imgs.shape) - 1]
        rel_phi = (abs_phi - coord_tensor).reshape(phi_shape).numpy()
        rel_flow_colors = []
        for time in range(abs_phi.shape[0]):
            rel_flow_colors.append(torch.from_numpy(flow_to_color(rel_phi[time], convert_to_bgr=False)).permute(2, 0, 1))
        flow_col = make_grid(rel_flow_colors, nrow=5)

        sim_grid = make_grid([im for im in energies[0]], nrow=5, normalize=True)

        imgs_to_save = {'imgs/reg_last':(registration_last*255).to(torch.uint8).permute(1, 2, 0),
                        'imgs/reg_all':(registration_all*255).to(torch.uint8).permute(1, 2, 0),
                        'flows/flow':flow_col.permute(1, 2, 0),
                        'energies/sim_loss':(sim_grid*255).cpu().to(torch.uint8).permute(1, 2, 0)
                        }


        if config.debug or last_val:
            vel_shape = [rel_vel.shape[0]] + phi_shape[1:]
            rel_vel = (rel_vel.detach().cpu()).reshape(vel_shape)
            rel_act_velocity_color = []
            for time in range(rel_vel.shape[0]):
                rel_act_velocity_color.append(torch.from_numpy(flow_to_color(rel_vel[time].numpy(), convert_to_bgr=False)).permute(2, 0, 1))
            vel_color = make_grid(rel_act_velocity_color, nrow=5)
            vel_norm = make_grid([torch.linalg.norm(vel, ord=2, dim=-1).unsqueeze(0) for vel in rel_vel], nrow=5, value_range=(0, 2))
            imgs_to_save['flows/vel_col'] = vel_color.permute(1, 2, 0)
            imgs_to_save['flows/vel_norm'] = (vel_norm*255).to(torch.uint8).permute(1, 2, 0)

            fin_J_det = torch.det(energies[1])
            pos_fin_J = torch.relu(fin_J_det)
            neg_fin_J = torch.relu(-fin_J_det)
            rgb_fin_J = [torch.stack([p, torch.zeros_like(p), n]) for p, n in zip(pos_fin_J, neg_fin_J)]
            fin_J_det_grid = make_grid(rgb_fin_J, nrow=5, normalize=True, value_range=(0, 0.001), pad_value=1)
            fin_grad_norm = make_grid([torch.linalg.norm(im, dim=(-2, -1)).unsqueeze(0) for im in energies[1]], nrow=5, normalize=True, value_range=(0, 0.1))

            if len(energies[2].shape) < 4:
                energies[2] = energies[2].unsqueeze(0)
            energies[2] = energies[2] / abs_phi.shape[0]
            auto_J_det = torch.det(energies[2])
            pos_auto_J = torch.relu(auto_J_det)
            neg_auto_J = torch.relu(-auto_J_det)
            rgb_auto_J = [torch.stack([p.reshape(imgs.shape[1:]), torch.zeros_like(p.reshape(imgs.shape[1:])), n.reshape(imgs.shape[1:])]) for p, n in zip(pos_auto_J, neg_auto_J)]
            auto_J_det_grid = make_grid(rgb_auto_J, nrow=5, normalize=True, value_range=(0, 0.005))
            auto_grad_norm = make_grid([torch.linalg.norm(im, dim=(-2, -1)).reshape(imgs.shape[1:]) for im in energies[2]], nrow=5, normalize=True, value_range=(0, 0.5))

            fin_lap = energies[3] # T/1, H, W, 2
            fin_lap_norm = torch.linalg.norm(fin_lap, ord=2, dim=-1)
            fin_lap_grid = make_grid(fin_lap_norm, nrow=5, normalize=True, value_range=(0, 0.09))
            auto_lap = energies[4] # H*W, 2
            auto_lap_norm = torch.linalg.norm(auto_lap, ord=2, dim=-1).reshape(imgs.shape[1:]).unsqueeze(0)
            auto_lap_grid = make_grid(auto_lap_norm, nrow=5, normalize=True, value_range=(0, 150))

            fin_phi_J_det = torch.det(energies[5]) # T, H, W, 2, 2
            pos_fin_phi_J = torch.relu(fin_phi_J_det)
            neg_fin_phi_J = torch.relu(-fin_phi_J_det)
            rgb_fin_phi_J = [torch.stack([p, torch.zeros_like(p), n]) for p, n in zip(pos_fin_phi_J, neg_fin_phi_J)]
            fin_phi_J_det_grid = make_grid(rgb_fin_phi_J, nrow=5, normalize=True, value_range=(0, 0.001), pad_value=1)
            fin_phi_grad_norm = make_grid([torch.linalg.norm(im, dim=(-2, -1)).unsqueeze(0) for im in energies[1]], nrow=5, normalize=True, value_range=(0, 0.1))

            # if len(energies[6].shape) < 4:
            #     energies[6] = energies[6].unsqueeze(0)
            # energies[6] = energies[6] / abs_phi.shape[0]
            # auto_phi_J_det = torch.det(energies[6]) # H*W, 2, 2
            # pos_auto_phi_J = torch.relu(auto_phi_J_det)
            # neg_auto_phi_J = torch.relu(-auto_phi_J_det)
            # rgb_auto_phi_J = [torch.stack([p.reshape(imgs.shape[1:]), torch.zeros_like(p.reshape(imgs.shape[1:])), n.reshape(imgs.shape[1:])]) for p, n in zip(pos_auto_phi_J, neg_auto_phi_J)]
            # auto_phi_J_det_grid = make_grid(rgb_auto_phi_J, nrow=5, normalize=True, value_range=(0, 0.005))
            # auto_phi_grad_norm = make_grid([torch.linalg.norm(im, dim=(-2, -1)).reshape(imgs.shape[1:]) for im in energies[6]], nrow=5, normalize=True, value_range=(0, 0.5))

            phi = abs_phi.reshape(phi_shape)[-1]
            fig, ax = plt.subplots()
            for i in range(0, phi.shape[0], math.ceil(phi.shape[0]/64)):
                ax.plot(phi[i, :, 0], phi[i, :, 1], 'r-', linewidth=0.5)
            for i in range(0, phi.shape[1], math.ceil(phi.shape[1]/64)):
                ax.plot(phi[:, i, 0], phi[:, i, 1], 'r-', linewidth=0.5)
            ax.axis('off')
            ax.set_aspect('equal')
            fig.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf)
            np_image = np.array(image).transpose(2, 0, 1)
            plt.close(fig)

            imgs_to_save.update({
                'J_det/fin_J_det':(fin_J_det_grid*255).cpu().to(torch.uint8).permute(1, 2, 0),
                'J_det/auto_J_det':(auto_J_det_grid*255).cpu().to(torch.uint8).permute(1, 2, 0),
                'grad_norm/fin_grad_norm':(fin_grad_norm*255).cpu().to(torch.uint8).permute(1, 2, 0),
                'grad_norm/auto_grad_norm':(auto_grad_norm*255).cpu().to(torch.uint8).permute(1, 2, 0),
                'grid_deform/grid_def_last_step':torch.from_numpy(np_image).permute(1, 2, 0),
                'laplacian/fin_lap':(fin_lap_grid*255).cpu().to(torch.uint8).permute(1, 2, 0),
                'laplacian/auto_lap':(auto_lap_grid*255).cpu().to(torch.uint8).permute(1, 2, 0),
                'phi_J_det/fin_phi_J_det':(fin_phi_J_det_grid*255).cpu().to(torch.uint8).permute(1, 2, 0),
                # 'phi_J_det/auto_phi_J_det':(auto_phi_J_det_grid*255).cpu().to(torch.uint8).permute(1, 2, 0),
                'phi_grad_norm/fin_phi_grad_norm':(fin_phi_grad_norm*255).cpu().to(torch.uint8).permute(1, 2, 0),
                # 'phi_grad_norm/auto_phi_grad_norm':(auto_phi_grad_norm*255).cpu().to(torch.uint8).permute(1, 2, 0),
            })
    else:
        imgs_to_save = None


    return metrics, imgs_to_save

def log_metrics(config, metrics, writer, epoch, losses_to_log, imgs_to_log, last_val=False):
    for k, v in metrics.items():
        writer.add_scalar(k, v, epoch)

    for k, v in losses_to_log.items():
        writer.add_scalar(f'debug_losses/{k}', v, epoch)
    
    if imgs_to_log is not None and (config.debug or last_val):
        for k, v in imgs_to_log.items():
            writer.add_image(k, v, epoch, dataformats='HWC', )

    # if config.debug:
    #     with torch.no_grad():
    #         grads = torch.tensor([p.grad.norm() for p in func.parameters()])
    #         names = [n for n, p in func.named_parameters()]
    #         writer.add_scalar('grad_stats/mean_grad', grads.mean(), epoch)
    #         writer.add_scalar('grad_stats/min_grad', grads.min(), epoch)
    #         writer.add_scalar('grad_stats/max_grad', grads.max(), epoch)

    #         for i in range(len(names)):
    #             writer.add_scalar(f'all_grads/{names[i]}', grads[i], epoch)

    #         writer.add_scalar('phi_stats/abs_max', abs_phi.max(), epoch)
    #         writer.add_scalar('phi_stats/abs_min', abs_phi.min(), epoch)
    #         phi_11 = (abs_phi == torch.tensor([1, 1], device=abs_phi.device)).sum()
    #         phi_m1m1 = (abs_phi == torch.tensor([-1, -1], device=abs_phi.device)).sum()
    #         phi_1m1 = (abs_phi == torch.tensor([1, -1], device=abs_phi.device)).sum()
    #         phi_m11 = (abs_phi == torch.tensor([-1, 1], device=abs_phi.device)).sum()
    #         phi_boundary = phi_11 + phi_m1m1 + phi_1m1 + phi_m11
    #         all_phi = math.prod(list(abs_phi.shape))
    #         writer.add_scalar('phi_stats/boundary_ratio', phi_boundary/all_phi, epoch)
    #         writer.add_scalar('phi_stats/boundary_absolute', phi_boundary, epoch)


    #         writer.add_scalar('vel_stats/rel_max', rel_vel.max(), epoch)
    #         writer.add_scalar('vel_stats/rel_min', rel_vel.min(), epoch)
    #         writer.add_scalar('vel_stats/rel_mean', rel_vel.mean(), epoch)

    # if epoch % 25 == 0 or epoch == 1:
    #     imgs, neural_reps = data
    #     imgs = imgs.detach().cpu()

    #     imgs_to_display = [torch.stack([imgs[-1], torch.zeros_like(imgs[-1]), moved_imgs[-1]])]
    #     img_grid = make_grid(imgs_to_display, nrow=2, normalize=True)
    #     moved_grid = make_grid([torch.stack([im, torch.zeros_like(im), m_im]) for im, m_im in zip(imgs, moved_imgs)], nrow=5, normalize=True)

    #     writer.add_image('imgs/comparison', img_grid, epoch, dataformats='CHW', )
    #     writer.add_image('imgs/all_time', moved_grid, epoch, dataformats='CHW')

    #     abs_phi = abs_phi.detach().cpu()
    #     coord_tensor = generate_coord_tensor(imgs.shape[1:], device='cpu')
    #     phi_shape = list(imgs.shape) + [len(imgs.shape) - 1]
    #     rel_phi = (abs_phi - coord_tensor).reshape(phi_shape).numpy()
    #     rel_flow_colors = []
    #     for time in range(abs_phi.shape[0]):
    #         rel_flow_colors.append(torch.from_numpy(flow_to_color(rel_phi[time], convert_to_bgr=False)).permute(2, 0, 1))
    #     rel_flow_grid = make_grid(rel_flow_colors, nrow=5)

    #     vel_shape = [rel_vel.shape[0]] + phi_shape[1:]
    #     rel_vel = (rel_vel.detach().cpu()).reshape(vel_shape)
    #     rel_act_velocity_color = []
    #     rel_vel_norm = []
    #     for time in range(rel_vel.shape[0]):
    #         rel_act_velocity_color.append(torch.from_numpy(flow_to_color(rel_vel[time].numpy(), convert_to_bgr=False)).permute(2, 0, 1))
    #         rel_vel_norm.append(torch.linalg.norm(rel_vel[time], ord=2, dim=-1).unsqueeze(0))
    #     rel_vel_grid = make_grid(rel_act_velocity_color, nrow=5)
    #     rel_vel_norm_grid = make_grid(rel_vel_norm, nrow=5, value_range=(0, 2))

    #     writer.add_image('flows/rel_flow_all', rel_flow_grid, epoch, dataformats='CHW', )
    #     writer.add_image('flows/rel_velocity', rel_vel_grid, epoch, dataformats='CHW', )
    #     writer.add_image('flows/rel_vel_norm', rel_vel_norm_grid, epoch, dataformats='CHW', )

    #     sim_grid = make_grid([im for im in energies[0]], nrow=5, normalize=True)
    #     writer.add_image('energies/sim', sim_grid, epoch, dataformats='CHW', )

    #     imgs_to_save = {'reg_last':(img_grid*255).to(torch.uint8).permute(1, 2, 0),
    #                     'reg_all':(moved_grid*255).to(torch.uint8).permute(1, 2, 0),
    #                     'flow':rel_flow_grid.permute(1, 2, 0),
    #                     'vel_col':rel_vel_grid.permute(1, 2, 0),
    #                     'vel_norm':(rel_vel_norm_grid*255).to(torch.uint8).permute(1, 2, 0),
    #                     'sim_loss':(sim_grid*255).cpu().to(torch.uint8).permute(1, 2, 0)}

    #     if config.debug:
    #         fin_J_det = torch.det(energies[1])
    #         pos_fin_J = torch.relu(fin_J_det)
    #         neg_fin_J = torch.relu(-fin_J_det)
    #         rgb_fin_J = [torch.stack([p, torch.zeros_like(p), n]) for p, n in zip(pos_fin_J, neg_fin_J)]
    #         fin_J_det_grid = make_grid(rgb_fin_J, nrow=5, normalize=True, value_range=(0, 0.001), pad_value=1)
    #         fin_grad_norm = make_grid([torch.linalg.norm(im, dim=(-2, -1)).unsqueeze(0) for im in energies[1]], nrow=5, normalize=True, value_range=(0, 0.1))

    #         if len(energies[2].shape) < 4:
    #             energies[2] = energies[2].unsqueeze(0)
    #         energies[2] = energies[2] / abs_phi.shape[0]
    #         auto_J_det = torch.det(energies[2])
    #         pos_auto_J = torch.relu(auto_J_det)
    #         neg_auto_J = torch.relu(-auto_J_det)
    #         rgb_auto_J = [torch.stack([p.reshape(imgs.shape[1:]), torch.zeros_like(p.reshape(imgs.shape[1:])), n.reshape(imgs.shape[1:])]) for p, n in zip(pos_auto_J, neg_auto_J)]
    #         auto_J_det_grid = make_grid(rgb_auto_J, nrow=5, normalize=True, value_range=(0, 0.005))
    #         auto_grad_norm = make_grid([torch.linalg.norm(im, dim=(-2, -1)).reshape(imgs.shape[1:]) for im in energies[2]], nrow=5, normalize=True, value_range=(0, 0.5))

    #         writer.add_image('J_det/fin_diff', fin_J_det_grid, epoch, dataformats='CHW', )
    #         writer.add_image('J_det/auto_grad', auto_J_det_grid, epoch, dataformats='CHW', )
    #         writer.add_image('grad_norm/fin_diff', fin_grad_norm, epoch, dataformats='CHW', )
    #         writer.add_image('grad_norm/auto_grad', auto_grad_norm, epoch, dataformats='CHW', )

    #         fin_lap = energies[3] # T/1, H, W, 2
    #         fin_lap_norm = torch.linalg.norm(fin_lap, ord=2, dim=-1)
    #         fin_lap_grid = make_grid(fin_lap_norm, nrow=5, normalize=True, value_range=(0, 0.09))
    #         auto_lap = energies[4] # H*W, 2
    #         auto_lap_norm = torch.linalg.norm(auto_lap, ord=2, dim=-1).reshape(imgs.shape[1:]).unsqueeze(0)
    #         auto_lap_grid = make_grid(auto_lap_norm, nrow=5, normalize=True, value_range=(0, 150))
    #         writer.add_image('laplacian/fin_diff', fin_lap_grid, epoch, dataformats='CHW', )
    #         writer.add_image('laplacian/auto_grad', auto_lap_grid, epoch, dataformats='CHW', )

    #         fin_phi_J_det = torch.det(energies[5]) # T, H, W, 2, 2
    #         pos_fin_phi_J = torch.relu(fin_phi_J_det)
    #         neg_fin_phi_J = torch.relu(-fin_phi_J_det)
    #         rgb_fin_phi_J = [torch.stack([p, torch.zeros_like(p), n]) for p, n in zip(pos_fin_phi_J, neg_fin_phi_J)]
    #         fin_phi_J_det_grid = make_grid(rgb_fin_phi_J, nrow=5, normalize=True, value_range=(0, 0.001), pad_value=1)
    #         fin_phi_grad_norm = make_grid([torch.linalg.norm(im, dim=(-2, -1)).unsqueeze(0) for im in energies[1]], nrow=5, normalize=True, value_range=(0, 0.1))

    #         # if len(energies[6].shape) < 4:
    #         #     energies[6] = energies[6].unsqueeze(0)
    #         # energies[6] = energies[6] / abs_phi.shape[0]
    #         # auto_phi_J_det = torch.det(energies[6]) # H*W, 2, 2
    #         # pos_auto_phi_J = torch.relu(auto_phi_J_det)
    #         # neg_auto_phi_J = torch.relu(-auto_phi_J_det)
    #         # rgb_auto_phi_J = [torch.stack([p.reshape(imgs.shape[1:]), torch.zeros_like(p.reshape(imgs.shape[1:])), n.reshape(imgs.shape[1:])]) for p, n in zip(pos_auto_phi_J, neg_auto_phi_J)]
    #         # auto_phi_J_det_grid = make_grid(rgb_auto_phi_J, nrow=5, normalize=True, value_range=(0, 0.005))
    #         # auto_phi_grad_norm = make_grid([torch.linalg.norm(im, dim=(-2, -1)).reshape(imgs.shape[1:]) for im in energies[6]], nrow=5, normalize=True, value_range=(0, 0.5))

    #         writer.add_image('phi_J_det/fin_diff', fin_phi_J_det_grid, epoch, dataformats='CHW', )
    #         # writer.add_image('phi_J_det/auto_grad', auto_phi_J_det_grid, epoch, dataformats='CHW', )
    #         writer.add_image('phi_grad_norm/fin_diff', fin_phi_grad_norm, epoch, dataformats='CHW', )
    #         # writer.add_image('phi_grad_norm/auto_grad', auto_phi_grad_norm, epoch, dataformats='CHW', )

    #         phi = abs_phi.reshape(phi_shape)[-1]
    #         fig, ax = plt.subplots()
    #         for i in range(0, phi.shape[0], math.ceil(phi.shape[0]/64)):
    #             ax.plot(phi[i, :, 0], phi[i, :, 1], 'r-', linewidth=0.5)
    #         for i in range(0, phi.shape[1], math.ceil(phi.shape[1]/64)):
    #             ax.plot(phi[:, i, 0], phi[:, i, 1], 'r-', linewidth=0.5)
    #         ax.axis('off')
    #         ax.set_aspect('equal')
    #         fig.tight_layout()
            
    #         buf = io.BytesIO()
    #         fig.savefig(buf, format='png')
    #         buf.seek(0)
    #         image = Image.open(buf)
    #         np_image = np.array(image).transpose(2, 0, 1)
    #         writer.add_image('grid_deform/last_step', np_image, epoch, dataformats='CHW', )
    #         plt.close(fig)

    #         imgs_to_save.update({
    #             'fin_J_det':(fin_J_det_grid*255).cpu().to(torch.uint8).permute(1, 2, 0),
    #             'auto_J_det':(auto_J_det_grid*255).cpu().to(torch.uint8).permute(1, 2, 0),
    #             'fin_grad_norm':(fin_grad_norm*255).cpu().to(torch.uint8).permute(1, 2, 0),
    #             'auto_grad_norm':(auto_grad_norm*255).cpu().to(torch.uint8).permute(1, 2, 0),
    #             'grid_def':np_image.transpose(1, 2, 0),
    #             'fin_lap':(fin_lap_grid*255).cpu().to(torch.uint8).permute(1, 2, 0),
    #             'auto_lap':(auto_lap_grid*255).cpu().to(torch.uint8).permute(1, 2, 0),
    #             'fin_phi_J_det':(fin_phi_J_det_grid*255).cpu().to(torch.uint8).permute(1, 2, 0),
    #             # 'auto_phi_J_det':(auto_phi_J_det_grid*255).cpu().to(torch.uint8).permute(1, 2, 0),
    #             'fin_phi_grad_norm':(fin_phi_grad_norm*255).cpu().to(torch.uint8).permute(1, 2, 0),
    #             # 'auto_phi_grad_norm':(auto_phi_grad_norm*255).cpu().to(torch.uint8).permute(1, 2, 0),
    #         })
    #     else:
    #         imgs_to_save = {}
    
def save_results(config, output):
    save_path = os.path.join(config.log_path, 'res.pt')
    save_dict = {'phi':output[0].detach().cpu(),
                 'vel':output[1].detach().cpu() if output[1] is not None else None,
                 'coord_tensor':output[2].detach().cpu(),
                 'moved_imgs':output[3].detach().cpu(),
                 'st_dict':output[4],
                 'losses':output[6],
                 'time_stamps':output[7],
                 'loss_times':output[8],
                 'epoch':output[9],
                 'config':vars(config)}
    np_save_path = os.path.join(config.log_path, 'np_imgs.npy')
    np_save_dict = {'images':output[5]}
    torch.save(save_dict, save_path)
    np.save(np_save_path, np_save_dict, allow_pickle=True)
    # if config.debug:
    img_dict = os.path.join(config.log_path, 'imgs')
    os.makedirs(img_dict, exist_ok=True)
    for k, v in output[5].items():
        Image.fromarray(v.numpy()).save(os.path.join(img_dict, f'{k.split('/')[1]}.png'))
