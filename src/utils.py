import copy
from functools import partial
import logging
import os
import math
import io

from PIL import Image
import torch
import torch.nn.functional as F
from torch.optim import Adam
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
        segs = None
    elif config.dataset == 'test':
        imgs = torch.from_numpy(np.load(f'data/syn/rec/rec.npy').transpose((2, 0, 1)))
        st_dicts = torch.load(f'data/syn/rec/rec_nrep_st_dicts.pt')
        imgs = imgs[[0, 3]]
        st_dicts = [st_dicts[0], st_dicts[3]]
        config.time_points = 2
        segs = None
    elif config.dataset == 'const':
        imgs = torch.from_numpy(np.load(f'data/syn/rec/rec.npy').transpose((2, 0, 1)))
        st_dicts = torch.load(f'data/syn/rec/rec_nrep_st_dicts.pt')
        imgs = imgs[[0, 0]]
        st_dicts = [st_dicts[0], st_dicts[0]]
        config.time_points = 2
        segs = None
    elif config.dataset == 'rot_slow2_large':
        imgs = torch.from_numpy(np.load(f'data/syn/rot_slow2/rot_slow2.npy').transpose((2, 0, 1)))
        st_dicts = torch.load(f'data/syn/rot_slow2/rot_slow2_nrep_st_dicts_large.pt')
        segs = None
    elif config.dataset == 'rot_slow2_64':
        imgs = torch.from_numpy(np.load(f'data/syn/rot_slow2/rot_slow2.npy').transpose((2, 0, 1)))
        st_dicts = torch.load(f'data/syn/rot_slow2/rot_slow2_nrep_st_dicts_64x64x64.pt')
        segs = None
    elif config.dataset == 'mouse':
        imgs = torch.from_numpy(np.load(f'data/cell_tracking/GFP-GOWT1_mouse_stem_small.npy').transpose((2, 0, 1))).float()
        st_dicts = torch.load(f'data/cell_tracking/GFP-GOWT1_mouse_stem_small_nrep.npy')[0]
        segs = torch.from_numpy(np.load(f'data/cell_tracking/GFP-GOWT1_mouse_stem_st_seg_small.npy').transpose((2, 0, 1)))
    elif config.dataset == 'mouse_large':
        imgs = torch.from_numpy(np.load(f'data/cell_tracking/GFP-GOWT1_mouse_stem.npy').transpose((2, 0, 1))).float()
        st_dicts = None
        segs = torch.from_numpy(np.load(f'data/cell_tracking/GFP-GOWT1_mouse_stem_st_seg.npy').transpose((2, 0, 1)))
    elif config.dataset == 'mouse_corner':
        imgs = torch.from_numpy(np.load(f'data/cell_tracking/GFP-GOWT1_mouse_stem_corner.npy').transpose((2, 0, 1))).float()
        st_dicts = None
        segs = torch.from_numpy(np.load(f'data/cell_tracking/GFP-GOWT1_mouse_stem_corner_seg.npy').transpose((2, 0, 1)))

    models = []
    if config.dataset == 'rot_slow2_large' or config.dataset == 'mouse':
        model = Siren([2, 256, 256, 256, 1])
        model.load_state_dict(st_dicts)
        model.eval()
        models = [model]
    elif config.dataset == 'mouse_large' or config.dataset == 'mouse_corner':
        models = [None]
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

    imgs = imgs[config.start_frame:config.start_frame + config.time_points]
    models = models[config.start_frame:config.start_frame + config.time_points]
    segs = segs[config.start_frame:config.start_frame + config.time_points] if segs is not None else None
    return imgs, models, segs

def calculate_metrics(losses, config, abs_phi, rel_vel, imgs, segs, moved_imgs, func, collect_imgs, last_val=False):
    metrics = {}
    total = 0.
    for loss_type, loss_dict in losses.items():
        metrics[f'losses/{loss_dict['name']}'] = loss_dict['lambda'] * loss_dict['mean']
        metrics[f'debug_losses/{loss_dict['name']}'] = loss_dict['mean']
        total += loss_dict['lambda'] * loss_dict['mean']
    metrics['losses/total_loss'] = total

    phi_shape = list(imgs.shape) + [len(imgs.shape) - 1]
    abs_phi = abs_phi.detach().cpu()
    coord_tensor = generate_coord_tensor(imgs.shape[1:], device='cpu')
    rel_phi = (abs_phi - coord_tensor).reshape(phi_shape).numpy()
    vel_shape = [rel_vel.shape[0]] + phi_shape[1:]
    rel_vel = (rel_vel.detach().cpu()).reshape(vel_shape)
    if config.debug or last_val:
        if not last_val:
            grads = torch.tensor([p.grad.norm() for p in func.parameters()])
            names = [n for n, p in func.named_parameters()]
            metrics['grad_stats/mean_grad'] = grads.mean()
            metrics['grad_stats/min_grad'] = grads.min()
            metrics['grad_stats/max_grad'] = grads.max()

            for i in range(len(names)):
                metrics[f'all_grads/{names[i]}'] = grads[i]
                
        metrics['vel_stats/rel_max'] = rel_vel.max()
        metrics['vel_stats/rel_min'] = rel_vel.min()
        metrics['vel_stats/rel_mean'] = rel_vel.mean()

        if segs is not None:
            present_classes = [i for i in range(1, int(segs.max()) + 1) if (segs == i).sum() > 0]
            input_segs = segs[:1].expand(segs.shape).unsqueeze(1).float()

            grid = abs_phi.reshape(imgs.shape[0], imgs.shape[1], imgs.shape[2], 2)
            grid = torch.stack([grid[..., 1], grid[..., 0]], dim=-1)
            pred_segs = F.grid_sample(input_segs, grid, mode='nearest', align_corners=False).squeeze()

            dices = np.zeros(len(present_classes))
            for i, cls in enumerate(present_classes):
                gt_mask = segs == cls
                pred_mask = pred_segs == cls
                intersect = (gt_mask * pred_mask).sum()
                union = (gt_mask.sum() + pred_mask.sum())
                dices[i] = 2*intersect/union if union > 0 else 1
            metrics['dices/mean dice'] = dices.mean()


    if collect_imgs:
        reg_last, reg_all = prep_moved_img_vis(imgs, moved_imgs)
        flow_col, vel_color, vel_norm = prep_flow_vis(rel_phi, rel_vel)
        sim_grid = prep_sim_meas_vis(losses['sim']['loss'])
        def_grid = prep_grid_def_vis(abs_phi.reshape(phi_shape)[-1])

        imgs_to_save = {
            'imgs/reg_last':reg_last,
            'imgs/reg_all':reg_all,
            'flows/flow':flow_col,
            'flows/vel_col':vel_color,
            'flows/vel_norm':vel_norm,
            'energies/sim_loss':sim_grid,
            'grid_deform/grid_def_last_step':def_grid
        }

        if segs is not None:
            seg_last, seg_all = prep_seg_vis(segs, pred_segs)
            imgs_to_save['segmentations/seg_last'] = seg_last
            imgs_to_save['segmentations/seg_all'] = seg_all

        imgs_to_save = add_loss_specific_imgs(imgs_to_save, losses, config, abs_phi.shape[0], imgs.shape[1:])
    else:
        imgs_to_save = None

    return metrics, imgs_to_save

def add_loss_specific_imgs(imgs_to_save, losses, config, nr_time_frames, img_shape):
    for loss_type, loss_dict in losses.items():
        if loss_type == 'negJ':
            negJ = prep_detJ_vis(loss_dict['loss'], config, nr_time_frames, img_shape)
            imgs_to_save['vel_J_det/negJ'] = negJ
        elif loss_type == 'grd':
            grad_norm = prep_vel_grad_vis(loss_dict['loss'], config, nr_time_frames, img_shape)
            imgs_to_save['vel_grad_norm/grad_norm'] = grad_norm
        elif loss_type == 'lap':
            lap_norm = prep_vel_lap_vis(loss_dict['loss'], config, img_shape)
            imgs_to_save['laplacian/laplacian_norm'] = lap_norm
        elif loss_type == 'pgr':
            phi_grad_norm = prep_phi_grad_vis(loss_dict['loss'])
            imgs_to_save['phi_grad_norm/phi_grad_norm'] = phi_grad_norm

    return imgs_to_save

def prep_moved_img_vis(imgs, moved_imgs):
    imgs = imgs.detach().cpu()
    reg_last = make_grid([torch.stack([imgs[-1], torch.zeros_like(imgs[-1]), moved_imgs[-1]])], nrow=2, normalize=True)
    reg_all = make_grid([torch.stack([im, torch.zeros_like(im), m_im]) for im, m_im in zip(imgs, moved_imgs)], nrow=5, normalize=True)

    reg_last = (reg_last*255).to(torch.uint8).permute(1, 2, 0)
    reg_all = (reg_all*255).to(torch.uint8).permute(1, 2, 0)
    return reg_last, reg_all

def prep_flow_vis(rel_phi, rel_vel):
    rel_flow_colors = []
    for time in range(rel_phi.shape[0]):
        rel_flow_colors.append(torch.from_numpy(flow_to_color(rel_phi[time], convert_to_bgr=False)).permute(2, 0, 1))
    flow_col = make_grid(rel_flow_colors, nrow=5)
    flow_col = flow_col.permute(1, 2, 0)

    rel_act_velocity_color = []
    for time in range(rel_vel.shape[0]):
        rel_act_velocity_color.append(torch.from_numpy(flow_to_color(rel_vel[time].numpy(), convert_to_bgr=False)).permute(2, 0, 1))
    vel_color = make_grid(rel_act_velocity_color, nrow=5)
    vel_norm = make_grid([torch.linalg.norm(vel, ord=2, dim=-1).unsqueeze(0) for vel in rel_vel], nrow=5, value_range=(0, 2))
    vel_color = vel_color.permute(1, 2, 0)
    vel_norm = (vel_norm*255).to(torch.uint8).permute(1, 2, 0)

    return flow_col, vel_color, vel_norm

def prep_sim_meas_vis(sim_meas):
    sim_grid = make_grid([im for im in sim_meas], nrow=5, normalize=True)
    sim_grid = (sim_grid*255).cpu().to(torch.uint8).permute(1, 2, 0)
    return sim_grid

def prep_seg_vis(segs, pred_segs):
    gt_mask = (segs > 0).float()
    pred_mask = (pred_segs > 0).float()
    seg_comb_last = make_grid([torch.stack([gt_mask[-1], torch.zeros_like(gt_mask[-1]), pred_mask[-1]])], nrow=2, normalize=True)
    seg_comb_all = make_grid([torch.stack([s, torch.zeros_like(s), pr]) for s, pr in zip(gt_mask, pred_mask)], nrow=5, normalize=True)
    seg_comb_last = (seg_comb_last*255).to(torch.uint8).permute(1, 2, 0)
    seg_comb_all = (seg_comb_all*255).to(torch.uint8).permute(1, 2, 0)
    return seg_comb_all, seg_comb_last

def prep_detJ_vis(negJ, config, nr_time_frames, img_shape):
    if config.fin_diff_grad:
        J_det_grid = make_grid(negJ.unsqueeze(1), nrow=5, normalize=True, value_range=(0, 0.00001), pad_value=1)
    else:
        if len(negJ) < 4:
            negJ = negJ.unsqueeze(0)
        negJ = negJ / nr_time_frames
        negJ = [frame.reschape(img_shape) for frame in negJ]
        J_det_grid = make_grid(negJ, nrow=5, normalize=True, value_range=(0, 0.05))
    
    J_det_grid =(J_det_grid*255).cpu().to(torch.uint8).permute(1, 2, 0)
    return J_det_grid

def prep_vel_grad_vis(vel_grad, config, nr_time_frames, img_shape):
    if config.fin_diff_grad:
        grad_norm = make_grid([torch.linalg.norm(im, dim=-1).unsqueeze(0) for im in vel_grad], nrow=5, normalize=True, value_range=(0, 0.02))
    else:
        if len(vel_grad) < 4:
            vel_grad = vel_grad.unsqueeze(0)
        vel_grad = vel_grad / nr_time_frames
        grad_norm = make_grid([torch.linalg.norm(im, dim=-1).reshape(img_shape) for im in vel_grad], nrow=5, normalize=True, value_range=(0, 0.5))
    
    grad_norm = (grad_norm*255).cpu().to(torch.uint8).permute(1, 2, 0)
    return grad_norm

def prep_vel_lap_vis(lap, config, img_shape):
    if config.fin_diff_grad:
        # lap.shape T/1, H, W
        lap_grid = make_grid(lap.unsqueeze(1), nrow=5, normalize=True, value_range=(0, 0.09))
    else:
        # lap.shape H*W, 2
        lap_norm = torch.linalg.norm(lap, ord=2, dim=-1).reshape(img_shape).unsqueeze(0)
        lap_grid = make_grid(lap_norm, nrow=5, normalize=True, value_range=(0, 150))
    lap_grid = (lap_grid*255).cpu().to(torch.uint8).permute(1, 2, 0)
    return lap_grid

def prep_phi_grad_vis(phi_grad):
    phi_grad_norm = make_grid(torch.linalg.norm(phi_grad, dim=-1).unsqueeze(1), nrow=5, normalize=True, value_range=(0, 0.1))
    phi_grad_norm = (phi_grad_norm*255).cpu().to(torch.uint8).permute(1, 2, 0)
    return phi_grad_norm

def prep_grid_def_vis(last_phi):
    fig, ax = plt.subplots()
    for i in range(0, last_phi.shape[0], math.ceil(last_phi.shape[0]/64)):
        ax.plot(last_phi[i, :, 0], last_phi[i, :, 1], 'r-', linewidth=0.5)
    for i in range(0, last_phi.shape[1], math.ceil(last_phi.shape[1]/64)):
        ax.plot(last_phi[:, i, 0], last_phi[:, i, 1], 'r-', linewidth=0.5)
    ax.axis('off')
    ax.set_aspect('equal')
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    np_image = np.array(image).transpose(2, 0, 1)
    tens_image = torch.from_numpy(np_image).permute(1, 2, 0)
    plt.close(fig)
    return tens_image
    

def log_metrics(config, metrics, writer, epoch, imgs_to_log, last_val=False):
    for k, v in metrics.items():
        writer.add_scalar(k, v, epoch)

    if imgs_to_log is not None and (config.debug or last_val):
        for k, v in imgs_to_log.items():
            writer.add_image(k, v, epoch, dataformats='HWC', )
    
def save_results(config, output):
    save_path = os.path.join(config.log_path, 'res.pt')
    save_dict = {'phi':output[0].detach().cpu(),
                 'vel':output[1].detach().cpu() if output[1] is not None else None,
                 'coord_tensor':output[2].detach().cpu(),
                 'moved_imgs':output[3].detach().cpu(),
                 'st_dict':output[4],
                #  'losses':output[6],
                 'time_stamps':output[7],
                 'epoch':output[8],
                 'config':vars(config)}
    np_save_path = os.path.join(config.log_path, 'np_imgs.npy')
    np_save_dict = {'images':output[5]}
    torch.save(save_dict, save_path)
    np.save(np_save_path, np_save_dict, allow_pickle=True)
    img_dict = os.path.join(config.log_path, 'imgs')
    os.makedirs(img_dict, exist_ok=True)
    for k, v in output[5].items():
        Image.fromarray(v.numpy()).save(os.path.join(img_dict, f'{k.split('/')[1]}.png'))

def upsample_img_seg(img, seg, config, epoch):
    ndx = config.schedule.index(epoch)
    downsample = config.downsamples[ndx]
    print(downsample)
    new_shape = [l // downsample for l in img.shape[1:]]
    mode = 'bilinear' if len(img.shape) == 3 else 'trilinear'
    new_img = F.interpolate(img.unsqueeze(1), size=new_shape, mode=mode, antialias=True).squeeze(1)
    new_seg = F.interpolate(seg.unsqueeze(1), size=new_shape, mode='nearest-exact').squeeze(1)
    return new_img, new_seg, downsample

def get_relevant_loss_names(config):
    losses = {'sim':{'name':'Similarity loss',
                       'lambda':config.lambda_st,
                       'time':0.}}

    if config.lambda_negJ > 0 or config.lambda_grd > 0 or config.debug:
        losses['negJ'] = {'name':'Vel negative det J',
                       'lambda':config.lambda_negJ,
                       'time':0.}
        losses['grd'] = {'name':'Vel gradient',
                       'lambda':config.lambda_grd,
                       'time':0.}
    if config.lambda_lap > 0 or config.debug:
        losses['lap'] = {'name':'Vel Laplacian',
                       'lambda':config.lambda_lap,
                       'time':0.}
    if config.lambda_pgr > 0 or config.debug:
        losses['pgr'] = {'name':'Phi gradient',
                       'lambda':config.lambda_pgr,
                       'time':0.}

    return losses

def get_relative_vel(func, config, time_points, coord_tensor, keep_batch_dim):
    if config.fin_diff_grad and config.lambda_grd + config.lambda_negJ + config.lambda_lap > 0:
        if (config.func_name == 'siren' or config.func_name == 'wire'):
            rel_vel = func(time_points[1], coord_tensor).unsqueeze(0)
        elif ('siren' in config.func_name or 'wire' in config.func_name) and 't' in config.func_name:
            if keep_batch_dim:
                rel_vel = []
                for t in time_points:
                    rel_vel.append(func(t, coord_tensor))
                rel_vel = torch.stack(rel_vel)
            else:
                rel_vel = func(time_points[0], coord_tensor)
                for t in time_points[1:]:
                    rel_vel = rel_vel + func(t, coord_tensor)
                rel_vel = rel_vel.unsqueeze(0)
    return rel_vel