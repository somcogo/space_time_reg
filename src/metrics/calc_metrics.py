import torch
import torch.nn.functional as F
import numpy as np

from src.metrics.prep_visuals import prep_moved_img_vis, prep_sim_meas_vis, prep_flow_vis, prep_vel_vis, add_loss_specific_imgs, prep_seg_vis, prep_grid_def_vis
from src.utils.spatial_utils import generate_coord_tensor
from src.metrics.dice import calc_oasis_dice

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
            vel_color, vel_norm = prep_vel_vis(rel_vel)
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