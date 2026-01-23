import os

import torch
import numpy as np
from PIL import Image

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
                 'moving':output[9],
                 'moved_img_space':output[10],
                 'recon_init':output[11].detach().cpu(),
                 'config':vars(config)}
    np_save_path = os.path.join(config.log_path, 'np_imgs.npy')
    np_save_dict = {'images':output[5]}
    torch.save(save_dict, save_path)
    np.save(np_save_path, np_save_dict, allow_pickle=True)
    img_dict = os.path.join(config.log_path, 'imgs')
    os.makedirs(img_dict, exist_ok=True)
    for k, v in output[5].items():
        Image.fromarray(v.numpy()).save(os.path.join(img_dict, f'{k.split('/')[1]}.png'))