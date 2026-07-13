import os
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from fastmri import complex_abs
from flow_vis import flow_to_color
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import AxesImage

from stmr.utils.spatial_utils import generate_coord_tensor

from .prep_visuals import draw_deformed_grid


def prep_vis_summary_pdf(config: Namespace,
                         gt: torch.Tensor,
                         init_recon: torch.Tensor,
                         final_recon: torch.Tensor,
                         abs_phi: torch.Tensor,
                         rel_vel: torch.Tensor,
                         all_metrics: list):
    pdf_dir_path = os.path.join(config.log_path, 'pdfs')
    os.makedirs(pdf_dir_path, exist_ok=True)
    img_pdf_path = os.path.join(pdf_dir_path, 'images.pdf')
    phi_pdf_path = os.path.join(pdf_dir_path, 'deformation.pdf')
    vel_pdf_path = os.path.join(pdf_dir_path, 'velocity.pdf')
    T, C, H, W = gt.shape
    gt = prep_tensor(config, gt)
    init = prep_tensor(config, init_recon)
    final = prep_tensor(config, final_recon)
    gt, init, final = norm_arrays(gt, init, final)
    abs_phi = prep_phi(abs_phi, gt)
    rel_vel = prep_vel(rel_vel, gt)
    coord_tensor = generate_coord_tensor(abs_phi.shape[1:-1], 'cpu').numpy().reshape(abs_phi.shape[1:])
    rel_phi = abs_phi - coord_tensor

    losses, debug_losses = prep_metrics(all_metrics)

    # Create pdf with images visualised
    with PdfPages(img_pdf_path) as pdf:
        for t in range(T):
            create_figure_for_time_point(pdf, gt[t], init[t], final[t], t, losses)
        plt.close('all')

    # Create pdf with deformation visualised
    with PdfPages(phi_pdf_path) as pdf:
        create_figure_for_deformation(pdf, gt, final, abs_phi, rel_phi, losses, debug_losses)
        plt.close('all')

    # Create pdf with velocity visualised
    with PdfPages(vel_pdf_path) as pdf:
        create_figure_for_velocity(pdf, final, rel_vel)
        plt.close('all')

def add_cbar(img: AxesImage, fontsize: int, ticks: list):
    cbar = plt.colorbar(img, aspect=5, shrink=0.5, location='right') # , fraction=0.046, pad=0.04, aspect=1, shrink=0.5, 
    cbar.ax.tick_params(labelsize=fontsize)
    if ticks is not None:
        cbar.set_ticks(ticks)

def use_imshow_on_axes(axes: Axes, image: np.ndarray, axis_off: bool, title: str, fontsize: int, use_cbar: bool, ticks: list=None, cmap: str='gray'):
    plt_img = axes.imshow(image, cmap=cmap)
    if axis_off:
        axes.axis("off")
    axes.set_title(title, fontsize=fontsize)
    if use_cbar:
        add_cbar(plt_img, fontsize * 2 // 3, ticks)

def prep_tensor(config: Namespace, tensor: torch.Tensor) -> np.ndarray:
    if 'cmr' in config.dataset or config.dataset == 'heart_gt_ft_abs':
        out = complex_abs(tensor.detach().cpu().movedim(1, -1)).numpy()
    else:
        out = tensor.squeeze(1).detach().cpu().numpy()
    return out

def norm_arrays(gt: np.ndarray, init: np.ndarray, final: np.ndarray):
    mx = gt.max()
    mn = gt.min()
    scale = 1 if mx == mn else mx -mn

    gt = gt / scale
    init = init / scale
    final = final / scale

    return gt, init, final


def create_figure_for_time_point(pdf: PdfPages, gt: np.ndarray, init: np.ndarray, final: np.ndarray, t: int, losses: dict):
    fontsize = 12
    f, a = plt.subplots(3, 3)
    f.set_size_inches(9, 6)
    ticks = [min(gt.min(), init.min(), final.min(), np.abs(init - gt).min(), np.abs(final - gt).min()),
             max(gt.max(), init.max(), final.max(), np.abs(init - gt).max(), np.abs(final - gt).max())]

    use_imshow_on_axes(a[0,0], gt, True, f"GT time {t}", fontsize, True, ticks=ticks)
    use_imshow_on_axes(a[1,0], init, True, f"Initial recon time {t}", fontsize, True, ticks=ticks)
    use_imshow_on_axes(a[2,0], final, True, f"Final recon time {t}", fontsize, True, ticks=ticks)
    
    a[0,1].plot(losses['GT error sum'][:,t], 'r', label='learned')
    a[0,1].plot(losses['GT error sum init'][:,t], 'b', label='init')
    # a[0,1].axhline(y=np.abs(init -gt).sum(), color='b', linestyle='-', label='init')
    a[0,1].set_title('Error sum', fontsize=fontsize)
    a[0,1].legend()

    use_imshow_on_axes(a[1,1], np.abs(init - gt), True, "Error", fontsize, True, ticks=[np.abs(init - gt).min(), np.abs(init - gt).max()])
    use_imshow_on_axes(a[2,1], np.abs(final - gt), True, "Error", fontsize, True, ticks=[np.abs(final - gt).min(), np.abs(final - gt).max()])

    a[0,2].plot(losses['GT error max'][:,t], 'r', label='learned')
    a[0,2].plot(losses['GT error max init'][:,t], 'b', label='init')
    # a[0,2].axhline(y=np.abs(init -gt).max(), color='b', linestyle='-', label='init')
    a[0,2].set_title('Error max', fontsize=fontsize)
    a[0,2].legend()

    a[1,2].axis("off")
    a[2,2].axis("off")

    f.tight_layout()
    pdf.savefig(f)
    plt.close(f)

def prep_phi(phi: torch.Tensor, gt: np.ndarray) -> np.ndarray:
    phi = phi.detach().cpu().numpy()
    shape = list(gt[1:].shape) + [len(gt.shape) - 1]
    phi = phi.reshape(shape)
    return phi

def create_figure_for_deformation(pdf: PdfPages, gt: np.ndarray, final: np.ndarray, abs_phi: np.ndarray, rel_phi: np.ndarray, losses: dict, debug_losses: dict):
    fontsize = 12
    f, a = plt.subplots(abs_phi.shape[0] + 1, 12, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.2, 1, 1, 1, 1, 1, 1, 1]})
    f.set_size_inches(36, (abs_phi.shape[0] + 1)*2)

    use_imshow_on_axes(a[0,0], gt[0], True, "GT at time 0", fontsize, False)
    use_imshow_on_axes(a[0,1], final[0], True, "Final recon at time 0", fontsize, False)
    a[0,2].axis("off")
    a[0,3].axis("off")
    a[0,4].axis("off")
    a[0,5].plot(losses['Similarity loss'][:,0])
    a[0,5].set_title('Sim loss', fontsize=fontsize)
    a[0,6].plot(losses['Reconstruction reg'][:,0])
    a[0,6].set_title('CRR weighted', fontsize=fontsize)
    a[0,7].axis("off")
    a[0,8].axis("off")
    a[0,9].plot(debug_losses['Reconstruction reg'][:,0])
    a[0,9].set_title('CRR original value', fontsize=fontsize)
    a[0,10].axis("off")
    a[0,11].axis("off")

    for t in range(1, abs_phi.shape[0] + 1):
        use_imshow_on_axes(a[t,0], gt[t], True, f"GT at time {t}", fontsize, False)
        use_imshow_on_axes(a[t,1], final[t], True, f"Final recon at time {t}", fontsize, False)

        deform_dir = flow_to_color(rel_phi[t - 1])        
        use_imshow_on_axes(a[t,2], deform_dir, True, f"Deformation at time {t}", fontsize, False, cmap=None)
        draw_deformed_grid(torch.from_numpy(abs_phi[t - 1]), a[t, 3])
        a[t,3].set_title(f"Deformed grid at time {t}", fontsize=fontsize)
        a[t,3].set_box_aspect(abs_phi.shape[1]/abs_phi.shape[2])

        phi_norm = np.linalg.norm(rel_phi[t - 1], axis=-1)
        use_imshow_on_axes(a[t,4], phi_norm, True, f"Deformation norm at time {t}", fontsize, True, ticks=[phi_norm.min(), phi_norm.max()])

        a[t,5].plot(losses['Similarity loss'][:,t])
        a[t,5].set_title('Sim loss', fontsize=fontsize)

        if 'Reconstruction reg' in losses.keys():
            a[t,6].plot(losses['Reconstruction reg'][:,t])
            a[t,6].set_title('CRR weighted', fontsize=fontsize)
        else:
            a[t,6].axis("off")
        
        if 'Phi negative det J' in losses.keys():
            a[t,7].plot(losses['Phi negative det J'][:,t-1])
            a[t,7].set_title('Deform grad loss weighted', fontsize=fontsize)
        else:
            a[t,7].axis("off")
        
        if 'Image space diff' in losses.keys():
            a[t,8].plot(losses['Image space diff'][:,t-1])
            a[t,8].set_title('Temp cons loss weighted', fontsize=fontsize)
        else:
            a[t,7].axis("off")
        
        if 'Reconstruction reg' in losses.keys():
            a[t,9].plot(debug_losses['Reconstruction reg'][:,t])
            a[t,9].set_title('CRR OG value', fontsize=fontsize)
        else:
            a[t,8].axis("off")
        
        if 'Phi negative det J' in losses.keys():
            a[t,10].plot(debug_losses['Phi negative det J'][:,t-1])
            a[t,10].set_title('Deform grad loss OG value', fontsize=fontsize)
        else:
            a[t,10].axis("off")
        
        if 'Image space diff' in losses.keys():
            a[t,11].plot(debug_losses['Image space diff'][:,t-1])
            a[t,11].set_title('Temp cons loss OG value', fontsize=fontsize)
        else:
            a[t,11].axis("off")
        
    f.tight_layout()
    pdf.savefig(f)
    plt.close(f)

def prep_vel(vel: torch.Tensor, gt: np.ndarray) -> np.ndarray:
    vel = vel.detach().cpu().numpy()
    shape = [-1] + list(gt.shape)[1:] + [len(gt.shape) - 1]
    vel = vel.reshape(shape)
    return vel

def create_figure_for_velocity(pdf: PdfPages, final: np.ndarray, rel_vel: np.ndarray):
    fontsize = 10
    f, a = plt.subplots(rel_vel.shape[0] + 1, 3)
    f.set_size_inches(9, (rel_vel.shape[0] + 1)*2)
    ticks = [final.min(), final.max()]
    # ticks_vel = [np.linalg.norm(rel_vel, axis=-1).min(), np.linalg.norm(rel_vel, axis=-1).max()]

    use_imshow_on_axes(a[0,0], final[0], True, "Final recon at time 0", fontsize, True, ticks=ticks)
    a[0,1].axis("off")
    a[0,2].axis("off")

    for t in range(1, rel_vel.shape[0] + 1):
        use_imshow_on_axes(a[t,0], final[t], True, f"Final recon at time {t}", fontsize, False)
        vel_dir = flow_to_color(rel_vel[t-1])
        use_imshow_on_axes(a[t,1], vel_dir, True, f"Velocity direction at time {t}", fontsize, False, cmap=None)
        normed_vel = np.linalg.norm(rel_vel[t-1], axis=-1)
        use_imshow_on_axes(a[t,2], normed_vel, True, f"Velocity norm at time {t}", fontsize, True, ticks=[normed_vel.min(), normed_vel.max()])
    
    f.tight_layout()
    pdf.savefig(f)
    plt.close(f)

def prep_metrics(all_metrics):
    loss_names = [l['name'] for l in all_metrics[0]['losses'].values() if l['name'] != 'Reconstruction reg']
    T = all_metrics[0]['losses']['sim']['loss'].shape[0]
    losses = {}
    debug_losses = {}
    
    for name in loss_names:
        losses[name] = np.zeros((len(all_metrics), T)) if name == 'Similarity loss' else np.zeros((len(all_metrics), T-1))
        debug_losses[name] = np.zeros((len(all_metrics), T)) if name == 'Similarity loss' else np.zeros((len(all_metrics), T-1))
        for i, m in enumerate(all_metrics):
            for l in m['losses'].values():
                if l['name'] == name:
                    # all_metrics may already store per-frame sums (1-D); only reduce
                    # spatial dims when the full per-pixel map is present.
                    per_frame = l['loss'].sum(dim=tuple(range(1, l['loss'].dim()))) if l['loss'].dim() > 1 else l['loss']
                    losses[name][i] = l['lambda'] * per_frame.cpu().numpy()
                    debug_losses[name][i] = per_frame.cpu().numpy()
    
    name = 'Reconstruction reg'
    losses[name] = np.zeros((len(all_metrics), T))
    debug_losses[name] = np.zeros((len(all_metrics), T))
    for i, m in enumerate(all_metrics):
        for l in m['losses'].values():
            if l['name'] == name:
                losses[name][i] = l['lambda'] * l['loss'].cpu().numpy()
                debug_losses[name][i] = l['loss'].cpu().numpy()

    name = 'GT error sum'
    losses[name] = np.zeros((len(all_metrics), T))
    for i, m in enumerate(all_metrics):
        for k, v in m['metrics'].items():
            if name in k and 'moved' in k:
                t = int(k[-1])
                losses[name][i][t] = v.cpu().numpy()

    name = 'GT error max'
    losses[name] = np.zeros((len(all_metrics), T))
    for i, m in enumerate(all_metrics):
        for k, v in m['metrics'].items():
            if name in k and 'moved' in k:
                t = int(k[-1])
                losses[name][i][t] = v.cpu().numpy()
    

    name = 'GT error sum init'
    losses[name] = np.zeros((len(all_metrics), T))
    for i, m in enumerate(all_metrics):
        for k, v in m['metrics'].items():
            if 'GT error sum' in k and 'init' in k:
                t = int(k[-1])
                losses[name][i][t] = v.cpu().numpy()

    name = 'GT error max init'
    losses[name] = np.zeros((len(all_metrics), T))
    for i, m in enumerate(all_metrics):
        for k, v in m['metrics'].items():
            if 'GT error max' in k and 'init' in k:
                t = int(k[-1])
                losses[name][i][t] = v.cpu().numpy()
    
    return losses, debug_losses