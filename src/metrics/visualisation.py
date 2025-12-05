import math
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
from fastmri import complex_abs
from argparse import Namespace
from flow_vis import flow_to_color

from .prep_visuals import draw_deformed_grid

def prep_vis_summary_pdf(config: Namespace,
                         gt: torch.Tensor,
                         init_recon: torch.Tensor,
                         final_recon: torch.Tensor,
                         rel_phi: torch.Tensor,
                         rel_vel: torch.Tensor):
    pdf_dir_path = os.path.join(config.log_path, 'pdfs')
    os.makedirs(pdf_dir_path, exist_ok=True)
    img_pdf_path = os.path.join(pdf_dir_path, 'images.pdf')
    phi_pdf_path = os.path.join(pdf_dir_path, 'deformation.pdf')
    vel_pdf_path = os.path.join(pdf_dir_path, 'velocity.pdf')
    T, C, H, W = gt.shape
    gt = prep_tensor(gt)
    init = prep_tensor(init_recon)
    final = prep_tensor(final_recon)
    rel_phi = prep_phi(rel_phi, gt)
    rel_vel = prep_vel(rel_vel, gt)

    # Create pdf with images visualised
    with PdfPages(img_pdf_path) as pdf:
        for t in range(T):
            create_figure_for_time_point(pdf, gt[t], init[t], final[t], t)
        plt.close('all')

    # Create pdf with deformation visualised
    with PdfPages(phi_pdf_path) as pdf:
        create_figure_for_deformation(pdf, gt, final, rel_phi)
        plt.close('all')

    # Create pdf with velocity visualised
    with PdfPages(vel_pdf_path) as pdf:
        create_figure_for_velocity(pdf, final, rel_vel)
        plt.close('all')

def prep_tensor(tensor: torch.Tensor) -> np.ndarray:
    return complex_abs(tensor.detach().cpu().movedim(1, -1)).numpy()

def create_figure_for_time_point(pdf: PdfPages, gt: np.ndarray, init: np.ndarray, final: np.ndarray, t: int):
    f, a = plt.subplots(3, 2)
    f.set_size_inches(6, 9)

    a[0,0].imshow(gt, cmap='gray')
    a[0,0].axis("off")
    a[0,0].set_title(f"GT time {t}")

    a[1,0].imshow(init, cmap='gray')
    a[1,0].axis("off")
    a[1,0].set_title(f"Initial recon time {t}")

    a[2,0].imshow(final, cmap='gray')
    a[2,0].axis("off")
    a[2,0].set_title(f"Final recon time {t}")
    
    a[0,1].imshow(np.abs(gt - gt), cmap='gray')
    a[0,1].axis("off")

    a[1,1].imshow(np.abs(init - gt), cmap='gray')
    a[1,1].axis("off")
    a[1,1].set_title(f"Initial recon error time {t}")

    a[2,1].imshow(np.abs(final - gt), cmap='gray')
    a[2,1].axis("off")
    a[2,1].set_title(f"Final recon error time {t}")
    pdf.savefig(f)
    plt.close(f)

def prep_phi(phi: torch.Tensor, gt: np.ndarray) -> np.ndarray:
    phi = phi.detach().cpu().numpy()
    shape = list(gt.shape) + [len(gt.shape) - 1]
    phi = phi.reshape(shape)
    return phi

def create_figure_for_deformation(pdf: PdfPages, gt: np.ndarray, final: np.ndarray, rel_phi: np.ndarray):
    for page in range(math.ceil((rel_phi.shape[0]-1) / 4)):
        start_ndx = 4*page + 1
        end_ndx = 4*(page + 1) + 1
        f, a = plt.subplots(5, 4, gridspec_kw={'width_ratios': [1, 1, 1, 1]})
        f.set_size_inches(8, 10)

        a[0,0].imshow(gt[0], cmap='gray')
        a[0,0].axis("off")
        a[0,0].set_title("GT at time 0")

        a[0,1].imshow(final[0], cmap='gray')
        a[0,1].axis("off")
        a[0,1].set_title("Final recon at time 0")

        a[0,2].axis("off")
        a[0,3].axis("off")

        for i, t in enumerate(range(start_ndx, end_ndx)):
            a[i,0].imshow(gt[t], cmap='gray')
            a[i,0].axis("off")
            a[i,0].set_title(f"GT at time {t}")

            a[i,1].imshow(final[t], cmap='gray')
            a[i,1].axis("off")
            a[i,1].set_title(f"Final recon at time {t}")

            deform_dir = flow_to_color(rel_phi[t])
            a[i,2].imshow(deform_dir)
            a[i,2].axis("off")
            a[i,2].set_title(f"Deformation direction at time {t}")

            grid_f, grid_a = draw_deformed_grid(rel_phi[t])
            a[i,3] = grid_a
            a[i,3].set_title(f"Deformed grid at time {t}")
            plt.close(grid_f)
        
        pdf.savefig(f)
        plt.close(f)

def prep_vel(vel: torch.Tensor, gt: np.ndarray) -> np.ndarray:
    vel = vel.detach().cpu().numpy()
    shape = [-1] + list(gt[1:].shape) + [len(gt.shape) - 1]
    vel = vel.reshape(shape)
    return vel

def create_figure_for_velocity(pdf: PdfPages, final: np.ndarray, rel_vel: np.ndarray):
    for page in range(math.ceil((rel_vel.shape[0]-1) / 4)):
        start_ndx = 4*page + 1
        end_ndx = 4*(page + 1) + 1
        f, a = plt.subplots(5, 3)
        f.set_size_inches(6, 10)

        a[0,0].imshow(final[0], cmap='gray')
        a[0,0].axis("off")
        a[0,0].set_title("Final recon at time 0")

        a[0,1].axis("off")
        a[0,2].axis("off")

        for i, t in enumerate(range(start_ndx, end_ndx)):
            a[i,0].imshow(final[t], cmap='gray')
            a[i,0].axis("off")
            a[i,0].set_title(f"Final recon at time {t}")

            vel_dir = flow_to_color(rel_vel[t])
            a[i,1].imshow(vel_dir)
            a[i,1].axis("off")
            a[i,1].set_title(f"Velocity direction at time {t}")

            a[i,2].imshow(np.linalg.norm(rel_vel[t], axis=-1), cmap="gray")
            a[i,2].axis("off")
            a[i,2].set_title(f"Velocity norm at time {t}")
        
        pdf.savefig(f)
        plt.close(f)