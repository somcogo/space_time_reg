from argparse import Namespace

import numpy as np
import torch
from fastmri import complex_abs
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import nn

from stmr.data.data_utils import complex_abs as my_complex_abs
from stmr.losses.grad_calc import get_Jacobian
from stmr.utils.spatial_transformer import GridSampleTransformer
from stmr.utils.spatial_utils import generate_coord_tensor


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

def add_losses(losses: dict, metrics: dict) -> dict:
    total = 0.
    for loss_dict in losses.values():
        metrics[f'losses/{loss_dict['name']}'] = loss_dict['lambda'] * loss_dict['mean']
        metrics[f'debug_losses/{loss_dict['name']}'] = loss_dict['mean']
        total += loss_dict['lambda'] * loss_dict['mean']
    metrics['losses/total_loss'] = total
    return metrics

def add_grad_stats(network: nn.Module, metrics: dict) -> dict:
    if network is not None:
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
    
def add_vel_stats(rel_vel: torch.Tensor, metrics: dict) -> dict:
    if rel_vel is not None:
        metrics['vel_stats/rel_max'] = rel_vel.max()
        metrics['vel_stats/rel_min'] = rel_vel.min()
        metrics['vel_stats/rel_mean'] = rel_vel.mean()
    return metrics
    
def add_dices(config: Namespace, seg_fix: torch.Tensor, seg_mov: torch.Tensor, ST: GridSampleTransformer, metrics: dict) -> list[dict, torch.Tensor]:
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

def add_gt_error(moving: torch.Tensor, gt_im: torch.Tensor, init_recon: torch.Tensor, rel_vel: torch.Tensor, metrics: dict) -> dict:
    # print(moved_im.min(), moved_im.max(), gt_im.min(), gt_im.max(), init_recon.min(), init_recon.max())
    moving_abs = my_complex_abs(moving)
    gt_abs = my_complex_abs(gt_im).cpu()
    in_abs = my_complex_abs(init_recon).cpu()
    mx, mn = gt_abs.max(), gt_abs.min()
    scale = 1 if (mx - mn) < 1e-10 else mx - mn

    moving_abs_rescaled = moving_abs / scale
    gt_abs_rescaled = gt_abs / scale
    in_abs_rescaled = in_abs / scale
    # print(moved_abs.min(), moved_abs.max(), gt_abs.min(), gt_abs.max(), in_abs.min(), in_abs.max())
    gt_im_error_rescaled = (moving_abs_rescaled - gt_abs_rescaled).abs()
    in_im_error_rescaled = (in_abs_rescaled - gt_abs_rescaled).abs()
    gt_im_error = (moving_abs - gt_abs).abs()
    for t in range(gt_im_error_rescaled.shape[0]):
        metrics[f'GT error sum/frame moved {t}'] = gt_im_error_rescaled[t].sum()
        metrics[f'GT error max/frame moved {t}'] = gt_im_error_rescaled[t].max()

        metrics[f'GT error sum/frame init {t}'] = in_im_error_rescaled[t].sum()
        metrics[f'GT error max/frame init {t}'] = in_im_error_rescaled[t].max()


        # metrics[f'GT error sum/frame moved {t} no cx abs'] = gt_err_nosq[t].sum()
        # metrics[f'GT error max/frame moved {t} no cx abs'] = gt_err_nosq[t].max()

        # metrics[f'GT error sum/frame moved {t}'] = gt_im_error[t].sum()
        # metrics[f'GT error max/frame moved {t}'] = gt_im_error[t].max()

        # metrics[f'GT error sum/frame init {t}'] = in_im_error[t].sum()
        # metrics[f'GT error max/frame init {t}'] = in_im_error[t].max()

    vel_J = get_Jacobian(rel_vel, rel_vel.shape)
    vel_norm = torch.linalg.vector_norm(rel_vel, dim=(-1))
    grad_norm = torch.linalg.vector_norm(vel_J, dim=(-2, -1))
    h1_norm = vel_norm[:gt_im_error.shape[0]] + grad_norm[:gt_im_error.shape[0]]

    metrics['Main metrics/GT error'] = gt_im_error.sum()
    metrics['Main metrics/vel H1 error'] = h1_norm.sum()
    metrics['Main metrics/vel L2 error'] = vel_norm[:gt_im_error.shape[0]].sum()
    metrics['Main metrics/vel grad L2 error'] = grad_norm[:gt_im_error.shape[0]].sum()
    metrics['Main metrics/Combined'] = gt_im_error.sum() + h1_norm.sum()


    # moving_abs = my_complex_abs(moving)
    # gt_abs = my_complex_abs(gt_im).cpu()
    # in_abs = my_complex_abs(init_recon).cpu()
    # sqabs_err = (moving_abs - gt_abs).square().sum()
    # abs_err = (moving - gt_im.cpu()).square().sum()

    # loss_fn = nn.MSELoss(reduction='none')
    # sim_loss = loss_fn(moving, gt_im.cpu()).sum()

    # print(f'MSELoss sum is {sim_loss}, abs error {abs_err}, squared abs error {sqabs_err}')

    return metrics

def reshape_phi_and_vel(abs_phi: torch.Tensor, rel_vel: torch.Tensor, phi_shape: list) -> list[torch.Tensor]:
    abs_phi = abs_phi.detach().cpu()
    coord_tensor = generate_coord_tensor(phi_shape[1:-1], device='cpu')
    rel_phi = (abs_phi - coord_tensor).reshape(phi_shape)
    abs_phi = abs_phi.reshape(phi_shape)
    if rel_vel is not None:
        vel_shape = [rel_vel.shape[0]] + phi_shape[1:]
        rel_vel = (rel_vel.detach().cpu()).reshape(vel_shape)
    return abs_phi, rel_phi, rel_vel

def reduce_dim(fixed: torch.Tensor, moved_im: torch.Tensor, moving: torch.Tensor, gt_im: torch.Tensor, rel_phi: torch.Tensor, rel_vel: torch.Tensor, seg_fix: torch.Tensor, pred_segs: torch.Tensor, abs_phi: torch.Tensor, losses: dict, reduce: bool, slice_ndx: int) -> list[torch.Tensor]:
    fixed = fixed[:, :, slice_ndx]
    moving = moving[:, :, slice_ndx]
    gt_im = gt_im[:, :, slice_ndx]
    if seg_fix is not None:
        seg_fix = seg_fix[:, :, slice_ndx]
        pred_segs = pred_segs[:, :, slice_ndx]
    if moved_im is not None:
        moved_im = moved_im[:, :, slice_ndx]
    if reduce:
        rel_phi = rel_phi[:, slice_ndx, :, :, 1:]
        abs_phi = abs_phi[:, slice_ndx, :, :, 1:]
        if rel_vel is not None:
            rel_vel = rel_vel[:, slice_ndx, :, :, 1:]
    sim_loss = losses['sim']['loss'][:, :, slice_ndx]
    if 'imdiff' in losses.keys():
        imdiff_loss = losses['imdiff']['loss'][:, :, slice_ndx]
    else:
        imdiff_loss = None
    return fixed, moved_im, moving, gt_im, rel_phi, rel_vel, seg_fix, pred_segs, abs_phi, sim_loss, imdiff_loss

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

def cmrx_crop(H: int, W: int) -> tuple[int, int, int, int]:
    """(h_from, h_to, w_from, w_to) of the CMRxRecon ranking crop. Use this for every heart/eval
    crop. run4Ranking.m takes crop(img, [round(sx/3), round(sy/2)]) with MATLAB sx = readout = our
    W and sy = phase = our H, i.e. a centred round(H/2) x round(W/3) box placed like crop.m
    (0-based start floor(m/2) + ceil(-s/2)): rows 51:153, cols 171:342 for 204 x 512."""
    h, w = int(np.floor(H / 2 + 0.5)), int(np.floor(W / 3 + 0.5))  # MATLAB round
    h_from, w_from = H // 2 + int(np.ceil(-h / 2)), W // 2 + int(np.ceil(-w / 2))
    return h_from, h_from + h, w_from, w_from + w

def calc_cmr_eval_metrics(pred_recon: torch.Tensor, gt_recon: torch.Tensor) -> list[np.ndarray]:
    # backward compatibility with 2D runs
    if len(pred_recon.shape) == 4:
        pred_recon = pred_recon.unsqueeze(2)
        gt_recon = gt_recon.unsqueeze(2)
    gt_recon = gt_recon.cpu().numpy()
    pred_recon = pred_recon.cpu().numpy()
    psnr_array = np.zeros(gt_recon.shape[:3])
    ssim_array = np.zeros(gt_recon.shape[:3])
    nmse_array = np.zeros(gt_recon.shape[:3])
    for t in range(gt_recon.shape[0]):
        for c in range(gt_recon.shape[1]):
            for d in range(gt_recon.shape[2]):
                pred, gt = pred_recon[t, c, d], gt_recon[t, c, d]
                # No per-image self-normalization: pred/pred.max() would hide global scale
                # errors and let a single bright artifact pixel shift the score. psnr/ssim/nmse
                # already use gt.max() as the data range, as in the official CMRxRecon eval.
                psnr_array[t, c, d] = psnr(gt, pred)
                ssim_array[t, c, d] = ssim(gt, pred)
                nmse_array[t, c, d] = nmse(gt, pred)
    return psnr_array, ssim_array, nmse_array

def add_cmr_eval_metrics(moving: torch.Tensor, gt_im: torch.Tensor, metrics: dict) -> dict:
    moving_im_img = complex_abs(moving.movedim(1, -1)).unsqueeze(1)
    gt_im_img = complex_abs(gt_im.movedim(1, -1)).unsqueeze(1)
    full_psnr, full_ssim, full_nmse = calc_cmr_eval_metrics(moving_im_img, gt_im_img)

    metrics['cmr evals all/full psnr'] = full_psnr.mean()
    metrics['cmr evals all/full ssim'] = full_ssim.mean()
    metrics['cmr evals all/full nsme'] = full_nmse.mean()
    
    metrics['cmr evals first/full psnr'] = full_psnr[0].mean()
    metrics['cmr evals first/full ssim'] = full_ssim[0].mean()
    metrics['cmr evals first/full nsme'] = full_nmse[0].mean()
    
    
    H, W = moving_im_img.shape[-2:]
    h_from, h_to, w_from, w_to = cmrx_crop(H, W)
    crop_moved_img = moving_im_img[..., h_from:h_to, w_from:w_to]
    crop_gt_img = gt_im_img[..., h_from:h_to, w_from:w_to]
    crop_psnr, crop_ssim, crop_nmse = calc_cmr_eval_metrics(crop_moved_img, crop_gt_img)
    
    metrics['cmr evals all/cropped psnr'] = crop_psnr.mean()
    metrics['cmr evals all/cropped ssim'] = crop_ssim.mean()
    metrics['cmr evals all/cropped nmse'] = crop_nmse.mean()
    
    metrics['cmr evals first/cropped psnr'] = crop_psnr[0].mean()
    metrics['cmr evals first/cropped ssim'] = crop_ssim[0].mean()
    metrics['cmr evals first/cropped nmse'] = crop_nmse[0].mean()

    metrics['cmr evals official/ssim'] = cmrx_official_ssim(moving, gt_im)
    return metrics
def cmrx_official_ssim(pred_recon: torch.Tensor, gt_recon: torch.Tensor) -> float:
    """SSIM as ranked in the CMRxRecon 2023 challenge (CMRxReconDemo/run4Ranking.m and
    Evaluation/Evaluation.py calmetric): magnitude, central 2 slices (all if fewer than 3) x first
    3 frames, centre crop round(H/2) x round(W/3), each 2D image divided by its own max (gt AND
    pred), SSIM with data range 1, mean over the images. Inputs [T, 2, D, H, W].

    The slice choice is only the official one when D is the patient's full stack. A single-slice
    run (D=1) is scored on that slice, so average the runs of the central slices (0-based
    round(sz/2)-2 and round(sz/2)-1, i.e. s3 and s4 for 10 slices). MATLAB's sx (readout) is our
    W and sy (phase) our H; our 2D images are its transpose, which leaves SSIM unchanged.
    """
    mag = lambda x: complex_abs(x[:3].detach().movedim(1, -1)).cpu().numpy().astype(np.float32)
    pred, gt = mag(pred_recon), mag(gt_recon)
    D, H, W = gt.shape[1:]
    if D >= 3:
        c = int(np.floor(D / 2 + 0.5))  # MATLAB round
        pred, gt = pred[:, c - 2:c], gt[:, c - 2:c]
    h_from, h_to, w_from, w_to = cmrx_crop(H, W)
    pred, gt = pred[..., h_from:h_to, w_from:w_to], gt[..., h_from:h_to, w_from:w_to]
    hw = gt.shape[-2:]
    return float(np.mean([ssim(g / g.max(), p / p.max())
                          for p, g in zip(pred.reshape(-1, *hw), gt.reshape(-1, *hw))]))
