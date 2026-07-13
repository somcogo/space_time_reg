import os

import numpy as np
import torch
from PIL import Image


def save_results(config, output: list, inputs: list, eval_inputs: list,
                 images: dict) -> None:
    model_outputs, loss_outputs, moving, st_dict, epoch, all_metrics, time_stamps, coords = output
    rel_vel, abs_phi, _ = model_outputs
    _, moved_im = loss_outputs
    recon_init, gt_im, _, _ = eval_inputs
    _, _, fixed, _ = inputs
    save_path = os.path.join(config.log_path, 'res.pt')
    save_dict = {
        'phi': abs_phi.detach().cpu(),
        'vel': rel_vel.detach().cpu() if rel_vel is not None else None,
        'coord_tensor': coords.detach().cpu(),
        'st_dict': st_dict,
        'time_stamps': time_stamps,
        'epoch': epoch,
        'moving': moving.detach().cpu(),
        'moved_img_space': moved_im.detach().cpu(),
        'recon_init': recon_init.detach().cpu(),
        'config': config.to_dict() if hasattr(config, 'to_dict') else vars(config),
        'fixed': fixed.detach().cpu(),
        'gt_im': gt_im.detach().cpu(),
    }
    torch.save(save_dict, save_path)
    if config.debug:
        torch.save(all_metrics, os.path.join(config.log_path, 'metrics.pt'))
    np.save(os.path.join(config.log_path, 'np_imgs.npy'), {'images': images}, allow_pickle=True)
    img_dir = os.path.join(config.log_path, 'imgs')
    os.makedirs(img_dir, exist_ok=True)
    for k, v in images.items():
        Image.fromarray(v.numpy()).save(os.path.join(img_dir, f"{k.split('/')[1]}.png"))
