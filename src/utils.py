import logging
import os
import math
import io

from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import make_grid
import numpy as np
from flow_vis import flow_to_color
import matplotlib.pyplot as plt

from src.normalized_gradient_field import NormalizedGradientField2d, NormalizedGradientField3d, NODEO_NCC

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

def generate_coord_tensor(dims, device, min_coord=-1, max_coord=1):
    coordinate_tensor = [torch.linspace(min_coord, max_coord, dims[i]) for i in range(len(dims))]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing='ij')
    coordinate_tensor = torch.stack(coordinate_tensor, dim=-1)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), len(dims)]).to(device)
    return coordinate_tensor


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
    if rel_vel is not None:
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
                
        if rel_vel is not None:
            metrics['vel_stats/rel_max'] = rel_vel.max()
            metrics['vel_stats/rel_min'] = rel_vel.min()
            metrics['vel_stats/rel_mean'] = rel_vel.mean()

        if segs is not None:
            if 'oasis' in config.dataset:
                dice, pred_segs = calc_oasis_dice(segs=segs, abs_phi=abs_phi)
                metrics['dices/mean dice'] = dice.mean()
            else:
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
        reduce_dim = len(imgs.shape) == 4
        slice_ndx = imgs.shape[-1] // 2
        if reduce_dim:
            imgs = imgs[..., slice_ndx]
            moved_imgs = moved_imgs[..., slice_ndx]
            rel_phi = rel_phi[..., slice_ndx, :-1]
            if rel_vel is not None:
                rel_vel = rel_vel[..., slice_ndx, :-1]
            if segs is not None:
                segs = segs[..., slice_ndx]
                pred_segs = pred_segs[..., slice_ndx]
        sim_loss = losses['sim']['loss'][..., slice_ndx] if reduce_dim else losses['sim']['loss']
        abs_phi = abs_phi.reshape(phi_shape)[..., slice_ndx, :-1] if reduce_dim else abs_phi.reshape(phi_shape)

        reg_last, reg_all = prep_moved_img_vis(imgs, moved_imgs)
        flow_col = prep_flow_vis(rel_phi)
        sim_grid = prep_sim_meas_vis(sim_loss)
        def_grid = prep_grid_def_vis(abs_phi[-1])

        imgs_to_save = {
            'imgs/reg_last':reg_last,
            'imgs/reg_all':reg_all,
            'flows/flow':flow_col,
            'energies/sim_loss':sim_grid,
            'grid_deform/grid_def_last_step':def_grid
        }

        if rel_vel is not None:
            vel_color, vel_norm = prep_vel_vis(rel_phi, rel_vel)
            imgs_to_save['flows/vel_col'] = vel_color
            imgs_to_save['flows/vel_norm'] = vel_norm

        if segs is not None:
            seg_last, seg_all = prep_seg_vis(segs, pred_segs)
            imgs_to_save['segmentations/seg_last'] = seg_last
            imgs_to_save['segmentations/seg_all'] = seg_all

        imgs_to_save = add_loss_specific_imgs(imgs_to_save, losses, config, abs_phi.shape[0], imgs.shape[1:], reduce_dim=reduce_dim)
    else:
        imgs_to_save = None

    return metrics, imgs_to_save

def add_loss_specific_imgs(imgs_to_save, losses, config, nr_time_frames, img_shape, reduce_dim):
    for loss_type, loss_dict in losses.items():
        if loss_type == 'negJ':
            loss = loss_dict['loss'][..., 0] if reduce_dim else loss_dict['loss']
            negJ = prep_detJ_vis(loss, config, nr_time_frames, img_shape)
            imgs_to_save['vel_J_det/negJ'] = negJ
        elif loss_type == 'grd':
            loss = loss_dict['loss'][..., 0, :] if reduce_dim else loss_dict['loss']
            grad_norm = prep_vel_grad_vis(loss, config, nr_time_frames, img_shape)
            imgs_to_save['vel_grad_norm/grad_norm'] = grad_norm
        elif loss_type == 'lap':
            loss = loss_dict['loss'][..., 0] if reduce_dim else loss_dict['loss']
            lap_norm = prep_vel_lap_vis(loss, config, img_shape)
            imgs_to_save['laplacian/laplacian_norm'] = lap_norm
        elif loss_type == 'pgr':
            loss = loss_dict['loss'][..., 0, :] if reduce_dim else loss_dict['loss']
            phi_grad_norm = prep_phi_grad_vis(loss)
            imgs_to_save['phi_grad_norm/phi_grad_norm'] = phi_grad_norm

    return imgs_to_save

def prep_moved_img_vis(imgs, moved_imgs):
    imgs = imgs.detach().cpu()
    reg_last = make_grid([torch.stack([imgs[-1], torch.zeros_like(imgs[-1]), moved_imgs[-1]])], nrow=2, normalize=True)
    reg_all = make_grid([torch.stack([im, torch.zeros_like(im), m_im]) for im, m_im in zip(imgs, moved_imgs)], nrow=5, normalize=True)

    reg_last = (reg_last*255).to(torch.uint8).permute(1, 2, 0)
    reg_all = (reg_all*255).to(torch.uint8).permute(1, 2, 0)
    return reg_last, reg_all

def prep_vel_vis(rel_vel):
    rel_act_velocity_color = []
    for time in range(rel_vel.shape[0]):
        rel_act_velocity_color.append(torch.from_numpy(flow_to_color(rel_vel[time].numpy(), convert_to_bgr=False)).permute(2, 0, 1))
    vel_color = make_grid(rel_act_velocity_color, nrow=5)
    vel_norm = make_grid([torch.linalg.norm(vel, ord=2, dim=-1).unsqueeze(0) for vel in rel_vel], nrow=5, value_range=(0, 0.1))
    vel_color = vel_color.permute(1, 2, 0)
    vel_norm = (vel_norm*255).to(torch.uint8).permute(1, 2, 0)

    return vel_color, vel_norm

def prep_flow_vis(rel_phi):
    rel_flow_colors = []
    for time in range(rel_phi.shape[0]):
        rel_flow_colors.append(torch.from_numpy(flow_to_color(rel_phi[time], convert_to_bgr=False)).permute(2, 0, 1))
    flow_col = make_grid(rel_flow_colors, nrow=5)
    flow_col = flow_col.permute(1, 2, 0)

    return flow_col

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
    new_shape = [l // downsample for l in img.shape[1:]]
    mode = 'bilinear' if len(img.shape) == 3 else 'trilinear'
    antialias = mode == 'bilinear'
    new_img = F.interpolate(img.unsqueeze(1), size=new_shape, mode=mode, antialias=antialias).squeeze(1)
    new_seg = F.interpolate(seg.unsqueeze(1), size=new_shape, mode='nearest-exact').squeeze(1) if seg is not None else seg
    return new_img, new_seg, downsample

def get_relevant_loss_names(config):
    losses = {'sim':{'name':'Similarity loss',
                       'lambda':config.lambda_st,
                       'time':0.}}

    include_all = False
    # include_all = config.debug
    if config.lambda_negJ > 0 or config.lambda_grd > 0 or include_all:
        losses['negJ'] = {'name':'Vel negative det J',
                       'lambda':config.lambda_negJ,
                       'time':0.}
        losses['grd'] = {'name':'Vel gradient',
                       'lambda':config.lambda_grd,
                       'time':0.}
    if config.lambda_lap > 0 or include_all:
        losses['lap'] = {'name':'Vel Laplacian',
                       'lambda':config.lambda_lap,
                       'time':0.}
    if config.lambda_pgr > 0 or include_all:
        losses['pgr'] = {'name':'Phi gradient',
                       'lambda':config.lambda_pgr,
                       'time':0.}
    if config.lambda_hel > 0 or include_all:
        losses['hyper_el'] = {'name':'Hyper elasticity',
                       'lambda':config.lambda_hel,
                       'time':0.}

    return losses

def get_relative_vel(func, config, time_points, coord_tensor, keep_batch_dim):
    if config.fin_diff_grad and config.lambda_grd + config.lambda_negJ + config.lambda_lap + config.lambda_hel > 0:
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
    else:
        rel_vel = None
    return rel_vel

def calc_oasis_dice(segs, abs_phi):
    input_seg = segs[:1].expand(segs.shape).unsqueeze(1).float()
    grid = abs_phi.reshape(abs_phi.shape[0], *segs.shape[1:], len(segs.shape[1:]))
    grid = torch.stack([grid[..., i] for i in reversed(range(grid.shape[-1]))], dim=-1)
    pred_seg = F.grid_sample(input_seg, grid, mode='nearest', align_corners=False)
    label = [2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 28, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 60]
    dice = calc_dice(segs[-1].numpy(), pred_seg[-1].numpy(), labels=label)
    return dice, pred_seg.squeeze(1)

def calc_dice(array1, array2, labels):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    """
    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem

def apply_grid_sample(input_img, phi, mode='bilinear'):
    grid = phi.reshape(phi.shape[0], *input_img.shape[2:], len(input_img.shape[2:]))
    grid = torch.stack([grid[..., i] for i in reversed(range(grid.shape[-1]))], dim=-1)
    moved = F.grid_sample(input_img, grid, align_corners=False, mode=mode)
    return moved

def get_sim_loss_fn(config, imgs):
    if config.loss == 'mse':
        loss_fn = nn.MSELoss(reduction='none')
    elif config.loss == 'ngf':
        if imgs.dim() == 3:
            loss_fn = NormalizedGradientField2d(mm_spacing=1, eps=1e-6, reduction='none')
        else:
            loss_fn = NODEO_NCC()
    return loss_fn.to(config.device)

class GridSampleTransformer():
    def __init__(self, abs_phi, img_shape):
        grid = abs_phi.reshape(abs_phi.shape[0], *img_shape[1:], len(img_shape[1:]))
        self.grid = torch.stack([grid[..., i] for i in reversed(range(grid.shape[-1]))], dim=-1)

    def apply(self, input_img, mode='bilinear'):
        return F.grid_sample(input_img, self.grid, align_corners=False, mode=mode)
    
class NeuralRepTransformer():
    def __init__(self, abs_phi, img_shape, use_old_nrep):
        if use_old_nrep:
            self.nrep_applier = OldNeuralRepApplier(abs_phi)
        else:
            self.nrep_applier = NewNeuralRepApplier(abs_phi)
        self.grid_applier = GridSampleTransformer(abs_phi, img_shape)
        self.img_shape = img_shape

    def apply(self, input_img, input_is_seg=False):
        if input_is_seg:
            out = self.grid_applier.apply(input_img, mode='nearest')
        else:
            out = self.nrep_applier.apply(input_img)
            out = out.reshape(self.img_shape)
        return out

class OldNeuralRepApplier():
    def __init__(self, abs_phi):
        self.input = abs_phi

    def apply(self, nrep):
        model_out = nrep.net(self.input)
        return (model_out.squeeze(2) + 1) / 2
    
class NewNeuralRepApplier():
    def __init__(self, abs_phi):
        self.input = abs_phi

    def apply(self, nrep):
        model_out = nrep(torch.tensor([], device=self.input.device), self.input)
        return model_out.squeeze(2)

def get_spatial_transformer(abs_phi, img_shape, config):
    if config.use_nreps:
        use_old_nrep = config.dataset in ['easy', 'hard', 'rectri', 'rot', 'rot_slow', 'rot_slow2', 'rec', 'syn_test']
        return NeuralRepTransformer(abs_phi, img_shape, use_old_nrep)
    else:
        return GridSampleTransformer(abs_phi, img_shape)

def get_lr_scheduler(config, optimizer):
    pass