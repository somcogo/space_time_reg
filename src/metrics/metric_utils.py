import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from fastmri import complex_abs

from src.utils.spatial_utils import generate_coord_tensor

def calc_oasis_dice(segs, ST):
    input_seg = segs[:1].expand(segs.shape).unsqueeze(1).float()
    pred_seg = ST.apply(input_seg, mode='nearest')
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

def add_losses(losses, metrics):
    total = 0.
    for loss_type, loss_dict in losses.items():
        metrics[f'losses/{loss_dict['name']}'] = loss_dict['lambda'] * loss_dict['mean']
        metrics[f'debug_losses/{loss_dict['name']}'] = loss_dict['mean']
        total += loss_dict['lambda'] * loss_dict['mean']
    metrics['losses/total_loss'] = total
    return metrics

def add_grad_stats(network, metrics):
    grads = [p.grad for p in network.parameters()]
    if grads[0] is not None:
        grads = torch.tensor([g.norm() for g in grads])
        names = [n for n, p in network.named_parameters()]
        metrics['grad_stats/mean_grad'] = grads.mean()
        metrics['grad_stats/min_grad'] = grads.min()
        metrics['grad_stats/max_grad'] = grads.max()

        for i in range(len(names)):
            metrics[f'all_grads/{names[i]}'] = grads[i]
    return metrics
    
def add_vel_stats(rel_vel, metrics):
    if rel_vel is not None:
        metrics['vel_stats/rel_max'] = rel_vel.max()
        metrics['vel_stats/rel_min'] = rel_vel.min()
        metrics['vel_stats/rel_mean'] = rel_vel.mean()
    return metrics
    
def add_dices(config, seg_fix, seg_mov, ST, metrics):
    if seg_fix is not None:
        if 'oasis' in config.dataset:
            dice, pred_segs = calc_oasis_dice(segs=seg_mov, ST=ST)
            metrics['dices/mean dice'] = dice.mean()
        else:
            present_classes = [i for i in range(1, int(seg_fix.max()) + 1) if (seg_fix == i).sum() > 0]
            pred_segs = ST.apply(seg_mov, mode='nearest')
            dices = np.zeros(len(present_classes))
            for i, cls in enumerate(present_classes):
                gt_mask = seg_fix == cls
                pred_mask = pred_segs == cls
                intersect = (gt_mask * pred_mask).sum()
                union = (gt_mask.sum() + pred_mask.sum())
                dices[i] = 2*intersect/union if union > 0 else 1
            metrics['dices/mean dice'] = dices.mean()
    else:
        pred_segs = None
    return metrics, pred_segs

def reshape_phi_and_vel(abs_phi, rel_vel, moving):
    phi_shape = [-1] + list(moving.shape)[1:] + [len(moving.shape) - 1]
    abs_phi = abs_phi.detach().cpu()
    coord_tensor = generate_coord_tensor(abs_phi.reshape(phi_shape).shape[1:-1], device='cpu')
    rel_phi = (abs_phi - coord_tensor).reshape(phi_shape).numpy()
    abs_phi = abs_phi.reshape(phi_shape)
    if rel_vel is not None:
        vel_shape = [rel_vel.shape[0]] + phi_shape[1:]
        rel_vel = (rel_vel.detach().cpu()).reshape(vel_shape)
    return abs_phi, rel_phi, rel_vel

def reduce_dim(fixed, moved, moved_im, rel_phi, rel_vel, seg_fix, pred_segs, abs_phi, losses, reduce):
    slice_ndx = fixed.shape[-1] // 2
    if reduce:
        fixed = fixed[..., slice_ndx]
        moved = moved[..., slice_ndx]
        moved_im = moved_im[..., slice_ndx]
        rel_phi = rel_phi[..., slice_ndx, :-1]
        abs_phi = abs_phi[..., slice_ndx, :-1]
        if rel_vel is not None:
            rel_vel = rel_vel[..., slice_ndx, :-1]
        if seg_fix is not None:
            seg_fix = seg_fix[..., slice_ndx]
            pred_segs = pred_segs[..., slice_ndx]
    sim_loss = losses['sim']['loss'][..., slice_ndx] if reduce else losses['sim']['loss']
    return fixed, moved, moved_im, rel_phi, rel_vel, seg_fix, pred_segs, abs_phi, sim_loss

# add_cmr_eval_metrics, psnr, ssim and nmse function implementations are based on the official CMRxRecon evaluation code https://github.com/CmrxRecon/CMRxRecon/blob/main/Evaluation/Evaluation.py
def psnr(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)

def ssim(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    maxval = gt.max()
    return structural_similarity(gt, pred, data_range=maxval)

def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.array(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)

def calc_cmr_eval_metrics(pred_recon: torch.Tensor, gt_recon: torch.Tensor):
    gt_recon = gt_recon.cpu().numpy()
    pred_recon = pred_recon.cpu().numpy()
    psnr_array = np.zeros((gt_recon.shape[0], gt_recon.shape[1]))
    ssim_array = np.zeros((gt_recon.shape[0], gt_recon.shape[1]))
    nmse_array = np.zeros((gt_recon.shape[0], gt_recon.shape[1]))
    for t in range(gt_recon.shape[0]):
        for c in range(gt_recon.shape[1]):
            pred, gt = pred_recon[t, c], gt_recon[t, c]
            psnr_array[t, c] = psnr(gt / gt.max(), pred / pred.max())
            ssim_array[t, c] = ssim(gt / gt.max(), pred / pred.max())
            nmse_array[t, c] = nmse(gt / gt.max(), pred / pred.max())
    return psnr_array, ssim_array, nmse_array

def add_cmr_eval_metrics(moved_im: torch.Tensor, gt_im: torch.Tensor, metrics: dict):
    moved_im_img = complex_abs(moved_im.movedim(1, -1)).unsqueeze(1)
    gt_im_img = complex_abs(gt_im.movedim(1, -1)).unsqueeze(1)
    full_psnr, full_ssim, full_nmse = calc_cmr_eval_metrics(moved_im_img, gt_im_img)
    
    T, N, H, W = moved_im_img.shape
    h_from, h_to, w_from, w_to = round(H / 3), round(2 * H / 3), round(W / 4), round(3 * W / 4)
    crop_moved_img = moved_im_img[:, :, h_from:h_to, w_from:w_to]
    crop_gt_img = gt_im_img[:, :, h_from:h_to, w_from:w_to]
    crop_psnr, crop_ssim, crop_nmse = calc_cmr_eval_metrics(crop_moved_img, crop_gt_img)

    metrics['cmr evals all/full psnr'] = full_psnr.mean()
    metrics['cmr evals all/full ssim'] = full_ssim.mean()
    metrics['cmr evals all/full nsme'] = full_nmse.mean()
    
    metrics['cmr evals first/full psnr'] = full_psnr[0].mean()
    metrics['cmr evals first/full ssim'] = full_ssim[0].mean()
    metrics['cmr evals first/full nsme'] = full_nmse[0].mean()
    
    metrics['cmr evals all/cropped psnr'] = crop_psnr.mean()
    metrics['cmr evals all/cropped ssim'] = crop_ssim.mean()
    metrics['cmr evals all/cropped nmse'] = crop_nmse.mean()
    
    metrics['cmr evals first/cropped psnr'] = crop_psnr[0].mean()
    metrics['cmr evals first/cropped ssim'] = crop_ssim[0].mean()
    metrics['cmr evals first/cropped nmse'] = crop_nmse[0].mean()
    return metrics