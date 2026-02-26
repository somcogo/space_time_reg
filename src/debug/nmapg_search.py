import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import argparse
from operator import itemgetter

import torch
from torch.utils.tensorboard import SummaryWriter
import fastmri

from src.data.recon_init import init_using_nmAPG, init_with_grad_desc
from src.data.data_utils import get_data, get_operators, get_init, get_kspace_mask
from src.metrics.metric_utils import calc_cmr_eval_metrics

def train(config: argparse.Namespace):
    raw_kspace_data, gt_kspace_data, kspace_mask = get_data(config)
    
    # kspace_mask = get_kspace_mask(config, raw_kspace_data, config.factor)
    # kspace_mask = (raw_kspace_data[:1] != 0)
    full_forw, full_adj, forw_subs, forw_subs_adj = get_operators(config, kspace_mask)

    smaller_shape = list(raw_kspace_data.shape[:2]) + [-1] + list(raw_kspace_data.shape[3:])
    fixed = raw_kspace_data[kspace_mask.expand(raw_kspace_data.shape)].reshape(smaller_shape)
    fixed = fixed.to(config.device)

    gt_im = full_adj(gt_kspace_data)
    gt_im = gt_im.to(config.device)
    recon_init = get_init(config, raw_kspace_data, gt_im)

    if config.method == 'nmapg':
        recon_init, metrics = init_using_nmAPG(config, recon_init, fixed, forw_subs, forw_subs_adj, logger=None)
    elif config.method == 'graddes':
        recon_init, metrics = init_with_grad_desc(config, recon_init, fixed, forw_subs)
    return recon_init.detach().cpu(), gt_im.detach().cpu(), metrics

def log_metrics(config, rec, gt_im, metrics):
    writer = SummaryWriter(os.path.join(config.log_path, 'tensorboard'))

    if metrics is not None:
        for epoch in range(metrics.shape[0]):
            writer.add_scalar('losses/data_fit', metrics[epoch, 0], global_step=epoch)
            writer.add_scalar('losses/reg_value', metrics[epoch, 1], global_step=epoch)
            writer.add_scalar('losses/total_energy', metrics[epoch].sum(), global_step=epoch)
    else:
        epoch = config.recon_epochs

    rec_abs = fastmri.complex_abs(rec.movedim(1, -1))
    gt_abs = fastmri.complex_abs(gt_im.movedim(1, -1))

    a = max(gt_im.max()-gt_im.min(), rec.max()-rec.min())
    rec = rec / a
    gt_im = gt_im / a

    b = max(gt_abs.max(), rec_abs.max())
    rec_abs = rec_abs / b
    gt_abs = gt_abs / b

    writer.add_image('image/gt_real', gt_im[0,0].abs(), global_step=epoch, dataformats='HW')
    writer.add_image('image/gt_imag', gt_im[0,1].abs(), global_step=epoch, dataformats='HW')
    writer.add_image('image/rec_real', rec[0,0].abs(), global_step=epoch, dataformats='HW')
    writer.add_image('image/rec_imag', rec[0,1].abs(), global_step=epoch, dataformats='HW')
    writer.add_image('image/error_real', (gt_im[0,0]-rec[0,0]).abs()/(gt_im[0,0]-rec[0,0]).abs().max(), global_step=epoch, dataformats='HW')
    writer.add_image('image/error_imag', (gt_im[0,1]-rec[0,1]).abs()/(gt_im[0,1]-rec[0,1]).abs().max(), global_step=epoch, dataformats='HW')

    real_scaled = torch.zeros_like(gt_im[0,0])
    real_mask = gt_im[0,0] > 0.0
    real_scaled[real_mask] = (gt_im[0,0]-rec[0,0]).abs()[real_mask] / gt_im[0,0][real_mask]
    imag_scaled = torch.zeros_like(gt_im[0,1])
    imag_mask = gt_im[0,1] > 0.0
    imag_scaled[imag_mask] = (gt_im[0,1]-rec[0,1]).abs()[imag_mask] / gt_im[0,1][imag_mask]
    writer.add_image('image/error_real_scaled', real_scaled, global_step=epoch, dataformats='HW')
    writer.add_image('image/error_imag_scaled', imag_scaled, global_step=epoch, dataformats='HW')

    writer.add_image('comp_abs/gt', gt_abs, global_step=epoch, dataformats='CHW')
    writer.add_image('comp_abs/rec', rec_abs, global_step=epoch, dataformats='CHW')
    writer.add_image('comp_abs/error', (gt_abs-rec_abs).abs()/(gt_abs-rec_abs).abs().max(), global_step=epoch, dataformats='CHW')

    abs_scaled = torch.zeros_like(gt_abs)
    abs_mask = gt_abs > 0.0
    abs_scaled[abs_mask] = (gt_abs-rec_abs).abs()[abs_mask] / gt_abs[abs_mask]
    writer.add_image('comp_abs/error_scaled', abs_scaled, global_step=epoch, dataformats='CHW')

    writer.add_scalar('error/max', (gt_im-rec).abs().max(), global_step=epoch)
    writer.add_scalar('error/mean', (gt_im-rec).abs().mean(), global_step=epoch)
    writer.add_scalar('error/cabs_max', (gt_abs-rec_abs).abs().max(), global_step=epoch)
    writer.add_scalar('error/cabs_mean', (gt_abs-rec_abs).abs().mean(), global_step=epoch)

    psnr_array, ssim_array, nmse_array = calc_cmr_eval_metrics(rec_abs, gt_abs)
    writer.add_scalar('cmr metrics/psnr', psnr_array.mean(), global_step=epoch)
    writer.add_scalar('cmr metrics/ssim', ssim_array.mean(), global_step=epoch)
    writer.add_scalar('cmr metrics/nmse', nmse_array.mean(), global_step=epoch)
    

    return (gt_abs-rec_abs).abs().mean(), (gt_abs-rec_abs).abs().max()

def main(**kwargs) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cmr_test1')
    parser.add_argument("--init", type=str, default='zero')
    parser.add_argument("--reg", type=str, default='learned')
    parser.add_argument("--method", type=str, default='graddes')
    parser.add_argument("--mask", type=str, default='random2')
    # parser.add_argument("--slice_number", type=int, default=0)
    # parser.add_argument("--start_frame", type=int, default=0)
    # parser.add_argument("--time_points", type=int, default=1)
    parser.add_argument("--init_lr", type=float, default=1e-2)
    parser.add_argument("--recon_scale", type=float, default=1e-1)
    parser.add_argument("--reg_alpha", type=float, default=1)
    parser.add_argument("--recon_epochs", type=float, default=50)
    parser.add_argument("--lambda_st", type=float, default=1)
    parser.add_argument("--lambda_init_recon", type=float, default=1e-9)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--factor", type=float, default=4)
    parser.add_argument("--log_path", type=str, default='test')

    config = parser.parse_args()
    d = vars(config)
    for (k, v) in kwargs.items():
        d[k] = v
    
    if config.recon_scale == 0:
        config.recon_scale = None
    if config.reg_alpha == 0:
        config.reg_alpha = None
    config.debug = True
    config.detach_grads = True
    config.log_path = os.path.join('log/nmapg_debug_folder/loss_fn_search', config.log_path)
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    rec, gt_im, metrics = train(config)
    mean_error, max_error = log_metrics(config, rec, gt_im, metrics)
    os.makedirs(os.path.join(config.log_path, 'imgs'), exist_ok=True)
    torch.save(rec.detach().cpu(), os.path.join(config.log_path, 'imgs', 'rec.pt'))
    return mean_error, max_error

if __name__ == '__main__':
    warnings = []
    results = []
    alpha = 6
    scale = 10
    reg = 'learned'
    method = 'nmapg'
    # for lam in [1e-5]:
    #     for lr in [1e-2, ]:
    lam = 1e-5
    lr = 1e-2
    factor = 4
    dset = 'cmr_P001'
    mask = 'st'
    init_loss = 'log_mag'
    init_reg_abs = True
    for scale in [0, 10, 15, 5, -5]:
        for lam in [1e4, 1e2, 1e0, 1e-2, 1e-4]:
            try:
                name = f'{dset}-{method}-{reg}-alpha-{alpha}-sc-{scale}-lam-{lam}-lr{lr}-factor{factor}-mask{mask}-loss-{init_loss}-rabs{init_reg_abs}'
                mean_err, max_err = main(reg_alpha=alpha, recon_scale=scale, lambda_init_recon=lam, log_path=name, reg=reg, method=method, init_lr=lr, factor=factor, mask=mask, dataset=dset, slice_number=0, time_points=1, start_frame=0, init_loss=init_loss, init_reg_abs=init_reg_abs)
                results.append([name, mean_err, max_err])
            except Exception as e:
                warnings.append(name + ' error ' + str(e) + '\n')
    mean_sorted = sorted(results, key=itemgetter(1))
    max_sorted = sorted(results, key=itemgetter(2))
    print('--------------------------')
    print('Warnings:')
    print(*warnings)
    print('--------------------------')
    print(f'Best mean result: name {mean_sorted[0][0]} error {mean_sorted[0][1]}')
    print(f'Best max result: name {max_sorted[0][0]} error {max_sorted[0][1]}')
    print('--------------------------')
    print('All res')
    for res in max_sorted:
        print(res)