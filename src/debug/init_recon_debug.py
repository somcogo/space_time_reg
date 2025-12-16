import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import time

import torch
from torch import nn
import argparse
from fastmri import complex_abs
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from src.data.data_utils import FTAndSubsample, ZeroFillAndIFT, FastmriIFT
from src.losses.recon_reg import get_recon_regularizer
from src.metrics.metric_utils import calc_cmr_eval_metrics

def train(config: argparse.Namespace):
    writer = SummaryWriter(os.path.join(config.log_path, 'tensorboard'))
    patient = '001'
    raw_kspace_data = torch.load(f'data/processed/cmrxrecon/test/training_p{patient}_single_coil_acc_04_cine_sax.pt')[:,0].permute(0, 3, 1, 2)
    raw_kspace_data = raw_kspace_data[:10]
    kspace_mask = (raw_kspace_data[:1] != 0)

    gt_kspace_data = torch.load(f'data/processed/cmrxrecon/test/training_p{patient}_single_coil_full_cine_sax.pt')[:,0].permute(0, 3, 1, 2)
    gt_kspace_data = gt_kspace_data[:10]

    smaller_shape = list(raw_kspace_data.shape[:2]) + [-1] + list(raw_kspace_data.shape[3:])
    gt = raw_kspace_data[kspace_mask.expand(raw_kspace_data.shape)].reshape(smaller_shape)

    forw = FTAndSubsample(kspace_mask)
    inverse_method = ZeroFillAndIFT(kspace_mask)
    recon_init = nn.Parameter(torch.zeros_like(raw_kspace_data, device=config.device), requires_grad=True)
    inverse_FT = FastmriIFT()
    gt_im = inverse_FT(gt_kspace_data)


    optimizer = torch.optim.Adam([recon_init], lr=config.recon_lr)
    gt = gt.to(config.device)
    loss_fn = nn.MSELoss(reduction='mean')
    regularizer = get_recon_regularizer(config)
    best_loss = 1e8
    t1 = time.time()
    for epoch in range(1, config.recon_epochs + 1):
        optimizer.zero_grad()
        sim_loss = config.lambda_st * loss_fn(forw(recon_init), gt)
        reg_loss = config.lambda_recon * regularizer.g(recon_init.flatten(0,1).unsqueeze(1)).mean()
        loss_sum = sim_loss + reg_loss
        loss_sum.backward()
        optimizer.step()

        writer.add_scalar('loss/sum', loss_sum.detach().cpu(), global_step=epoch)
        writer.add_scalar('loss/sim', sim_loss.detach().cpu(), global_step=epoch)
        writer.add_scalar('loss/reg', reg_loss.detach().cpu(), global_step=epoch)
        
        psnr, ssim, nmse = calc_cmr_eval_metrics(complex_abs(recon_init.detach().cpu().movedim(1, -1)).unsqueeze(0),
                                                 complex_abs(gt_im.detach().cpu().movedim(1, -1)).unsqueeze(0))
        writer.add_scalar('metrics/nmse', nmse.mean(), global_step=epoch)
        writer.add_scalar('metrics/psnr', psnr.mean(), global_step=epoch)
        writer.add_scalar('metrics/ssim', ssim.mean(), global_step=epoch)
        img = complex_abs(recon_init.detach().cpu().movedim(1, -1))[:1]
        img = (img - img.min()) / (img.max() - img.min())
        writer.add_image('imgs/recon_0', img, global_step=epoch, dataformats='CHW')

        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{config.recon_epochs}, loss {loss_sum.detach().cpu().item():.9f}, nmse {nmse.mean():.2f}, psnr {psnr.mean():.2f}, ssim {ssim.mean():.2f}")
        if loss_sum <= best_loss:
            best_recon = recon_init.detach().clone()
            best_loss = loss_sum.detach().clone()
            best_epoch = epoch
    t2 = time.time()

    best_image = complex_abs(best_recon.movedim(1, -1)).unsqueeze(1)
    gt_image = complex_abs(gt_im.movedim(1, -1)).unsqueeze(1)
    best_psnr, best_ssim, best_nmse = calc_cmr_eval_metrics(best_image, gt_image)
    print(f"Finished training in {t2-t1:.0f} seconds. Best results from epoch {epoch}, loss {best_loss.detach().cpu().item():.9f}, nmse {best_nmse.mean():.2f}, psnr {best_psnr.mean():.2f}, ssim {best_ssim.mean():.2f}")

    os.makedirs(os.path.join(config.log_path, 'imgs'), exist_ok=True)
    best_image = (best_image - best_image.min()) / (best_image.max() - best_image.min())
    gt_image = (gt_image - gt_image.min()) / (gt_image.max() - gt_image.min())
    save_image(best_image[0], os.path.join(config.log_path, 'imgs', 'recon.png'))
    save_image(gt_image[0], os.path.join(config.log_path, 'imgs', 'gt.png'))
    torch.save({'nmse':best_nmse.mean(),
                'psnr':best_psnr.mean(),
                'ssim':best_ssim.mean(),
                'loss':best_loss.mean(),
                'epoch':best_epoch,
                'config':config},
                os.path.join(config.log_path, 'res.pt'))

def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon_lr", type=float, default=1e-3)
    parser.add_argument("--recon_scale", type=float, default=1e-1)
    parser.add_argument("--recon_epochs", type=float, default=500)
    parser.add_argument("--lambda_st", type=float, default=1)
    parser.add_argument("--lambda_recon", type=float, default=1e-9)
    parser.add_argument("--log_path", type=str, default='test')

    config = parser.parse_args()
    d = vars(config)
    for (k, v) in kwargs.items():
        d[k] = v

    config.log_path = os.path.join('log/recon_init', config.log_path)
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(config)

if __name__ == '__main__':
    warnings = []
    for lr in [1e0, ]:
        for scale in [1e-1, 1e0, 1e1, 1e2, 1e3]: # 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3
            for lam in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
                try:
                    main(recon_lr=lr, recon_scale=scale, lambda_recon=lam, log_path=f'lr-{lr}-sc-{scale}-lam-{lam}')
                except:
                    warnings.append(f'lr-{lr}-sc-{scale}-lam-{lam}\n')
    print(*warnings)