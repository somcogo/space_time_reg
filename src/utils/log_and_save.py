import os
from argparse import Namespace

import torch
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

def log_metrics(config: Namespace, metrics: dict, writer: SummaryWriter, epoch: int, imgs_to_log: dict, last_val=False) -> None:
    for k, v in metrics.items():
        writer.add_scalar(k, v, epoch)

    if imgs_to_log is not None and (config.debug or last_val):
        for k, v in imgs_to_log.items():
            writer.add_image(k, v, epoch, dataformats='HWC', )
    writer.flush()
    
def save_results(config: Namespace, output: list, eval_inputs: list, images: dict[torch.Tensor]) -> None:
    model_outputs, loss_outputs, moving, st_dict, epoch, _, time_stamps, coords = output
    abs_phi, rel_vel, _ = model_outputs
    _, moved_im = loss_outputs
    recon_init = eval_inputs[0]
    save_path = os.path.join(config.log_path, 'res.pt')
    save_dict = {'phi':abs_phi.detach().cpu(),
                 'vel':rel_vel.detach().cpu() if rel_vel is not None else None,
                 'coord_tensor':coords.detach().cpu(),
                #  'moved_imgs':moved.detach().cpu(),
                 'st_dict':st_dict,
                #  'losses':output[6],
                 'time_stamps':time_stamps,
                 'epoch':epoch,
                 'moving':moving.detach().cpu(),
                 'moved_img_space':moved_im.detach().cpu(),
                 'recon_init':recon_init.detach().cpu(),
                 'config':vars(config)}
    np_save_path = os.path.join(config.log_path, 'np_imgs.npy')
    np_save_dict = {'images':images}
    torch.save(save_dict, save_path)
    np.save(np_save_path, np_save_dict, allow_pickle=True)
    img_dict = os.path.join(config.log_path, 'imgs')
    os.makedirs(img_dict, exist_ok=True)
    for k, v in images.items():
        Image.fromarray(v.numpy()).save(os.path.join(img_dict, f'{k.split('/')[1]}.png'))