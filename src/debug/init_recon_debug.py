import os

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
    writer = SummaryWriter(os.path.join(config.log_dir, 'tensorboard'))
    patient = '001'
    raw_kspace_data = torch.load(f'data/processed/cmrxrecon/test/training_p{patient}_single_coil_acc_04_cine_sax.pt')[:,0].permute(0, 3, 1, 2)
    raw_kspace_data = raw_kspace_data[:10]
    kspace_mask = (raw_kspace_data[:1] != 0)

    gt_kspace_data = torch.load(f'data/processed/cmrxrecon/test/training_p{patient}_single_coil_full_cine_sax.pt')[:,0].permute(0, 3, 1, 2)
    gt_kspace_data = gt_kspace_data[:10]

    smaller_shape = list(raw_kspace_data.shape[:2]) + [-1] + list(raw_kspace_data.shape[3:])
    fixed = raw_kspace_data[kspace_mask.expand(raw_kspace_data.shape)].reshape(smaller_shape)

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

        nmse, psnr, ssim = calc_cmr_eval_metrics(complex_abs(recon_init.detach().cpu().movedim(1, -1)).unsqueeze(0),
                                                 complex_abs(gt.detach().cpu().movedim(1, -1)).unsqueeze(0))
        writer.add_scalar('metrics/nmse', nmse.mean(), global_step=epoch)
        writer.add_scalar('metrics/psnr', psnr.mean(), global_step=epoch)
        writer.add_scalar('metrics/ssim', ssim.mean(), global_step=epoch)

        print(f"Epoch {epoch}/{config.recon_epochs+1}, loss {loss_sum.detach().cpu().item():.8f}, nmse {nmse.mean():.8f}, psnr {psnr.mean():.8f}, ssim {ssim.mean():.8f}")
        if loss_sum <= best_loss:
            best_recon = recon_init.detach().clone()
            best_loss = loss_sum.detach().clone()
            best_epoch = epoch

    best_image = complex_abs(best_recon.movedim(1, -1)).unsqueeze(0)
    gt_image = complex_abs(gt_im.movedim(1, -1)).unsqueeze(0)
    best_nmse, best_psnr, best_ssim = calc_cmr_eval_metrics(best_image, gt_image)

    save_image(best_recon, os.path.join(config.log_path, 'imgs', 'recon.png'))
    save_image(gt_image, os.path.join(config.log_path, 'imgs', 'gt.png'))
    torch.save({'nmse':torch.from_numpy(best_nmse.mean()),
                'psnr':torch.from_numpy(best_psnr.mean()),
                'ssim':torch.from_numpy(best_ssim.mean()),
                'loss':best_loss.mean(),
                'epoch':best_epoch,
                'config':config},
                os.path.join(config.log_path, 'res.pt'))

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon_lr", type=float, default=1e-3)
    parser.add_argument("--recon_scale", type=float, default=1e-1)
    parser.add_argument("--recon_epochs", type=float, default=500)
    parser.add_argument("--lambda_st", type=float, default=1)
    parser.add_argument("--lambda_recon", type=float, default=1)
    parser.add_argument("--log_path", type=str, default='test')

    config = parser.parse_args()

    config.log_path = os.path.join('log/recon_init', config.log_path)
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(config)

if __name__ == '__main__':
    main()
    