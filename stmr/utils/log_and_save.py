import os

import numpy as np
import torch
from PIL import Image

from stmr.state import EvalInputs, Inputs, RegistrationResult


def save_results(config, output: RegistrationResult, inputs: Inputs,
                 eval_inputs: EvalInputs, images: dict) -> None:
    model_outputs, loss_outputs = output.model_outputs, output.loss_outputs
    moving, st_dict, epoch = output.best_moving, output.st_dict, output.epoch
    all_metrics, time_stamps, coords = output.all_metrics, output.time_stamps, output.coords
    rel_vel, abs_phi = model_outputs.rel_vel, model_outputs.abs_phi
    moved_im = loss_outputs.moved_imspace
    recon_init, gt_im = eval_inputs.init_recon, eval_inputs.gt_im
    fixed = inputs.fixed
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
