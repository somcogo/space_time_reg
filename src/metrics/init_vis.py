import os
from argparse import Namespace

import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
from fastmri import complex_abs

def prep_vis_summary_pdf(config: Namespace,
                         gt: torch.Tensor,
                         init_recon: torch.Tensor,):
    pdf_dir_path = os.path.join(config.log_path, 'pdfs')
    os.makedirs(pdf_dir_path, exist_ok=True)
    img_pdf_path = os.path.join(pdf_dir_path, 'images.pdf')
    T, C, H, W = gt.shape
    gt = prep_tensor(gt)
    init = prep_tensor(init_recon)
    gt, init= norm_arrays(gt, init)

    # Create pdf with images visualised
    with PdfPages(img_pdf_path) as pdf:
        for t in range(T):
            create_figure_for_time_point(pdf, gt[t], init[t], t)
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

def prep_tensor(tensor: torch.Tensor) -> np.ndarray:
    out = complex_abs(tensor.detach().cpu().movedim(1, -1)).numpy()
    return out

def norm_arrays(gt: np.ndarray, init: np.ndarray):
    mx = max(gt.max(), init.max())
    mn = min(gt.min(), init.min())

    gt = gt / (mx - mn)
    init = init / (mx - mn)

    return gt, init

def create_figure_for_time_point(pdf: PdfPages, gt: np.ndarray, init: np.ndarray, t: int):
    fontsize = 10
    f, a = plt.subplots(2, 2)
    f.set_size_inches(8, 4)
    ticks = [min(gt.min(), init.min(), np.abs(init - gt).min()),
             max(gt.max(), init.max(), np.abs(init - gt).max())]

    use_imshow_on_axes(a[0,0], gt, True, f"GT time {t}", fontsize, True, ticks=ticks)
    use_imshow_on_axes(a[1,0], init, True, f"Initial recon time {t}", fontsize, True, ticks=ticks)
    
    use_imshow_on_axes(a[1,1], np.abs(init - gt), True, "Error", fontsize, True, ticks=[np.abs(init - gt).min(), np.abs(init - gt).max()])

    
    img = np.abs(init - gt)/(gt+1e-10)
    img[np.abs(init - gt)<0.05] = 0
    img = np.clip(img, max=1)
    use_imshow_on_axes(a[0,1], img, True, "Error scaled by GT value pixelwise", fontsize, True, ticks=[img.min(), img.max()])

    # f.tight_layout()
    pdf.savefig(f)
    plt.close(f)