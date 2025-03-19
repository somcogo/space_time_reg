import os
import math

from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def create_animation(img_to_animate):
    animation.embed_limit = 10
    fig = plt.figure()
    ims = []
    for image in range(0,img_to_animate.shape[-1]):
        im = plt.imshow(img_to_animate[:,:,image], 
                        animated=True)
        plt.axis("off")
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,
                                    repeat_delay=1000)
    plt.close()
    html = HTML(ani.to_jshtml())
    return html, ani

def save_animation(ani, path):
    FFwriter = animation.FFMpegWriter(fps=10)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ani.save(path, writer = FFwriter)

def load_shape(name):
    img = Image.open(f'data/syn/{name}.png')
    img = np.array(img, dtype=np.int32)[:,:,0]
    img = 255 - img
    img[img==255] = 3
    img[img==60] = 2
    img[img==128] = 1
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(torch.float)
    return img

def create_syn_data(file_name, case):
    rec = load_shape('rec')
    tri = load_shape('tri')
    cir = load_shape('cir')

    if case == 'hard':
        rec_angles = math.pi*np.linspace(0, 0.5, 20)
        rec_xs = np.linspace(0, -0.75, 20)
        rec_ys = np.linspace(0, -0.03, 20)

        tri_angles = np.linspace(0, 0., 20)
        tri_xs = np.cos(np.linspace(0, math.pi*8)) * 0.1
        tri_ys = np.linspace(-0.3, 0.3, 20)

        cir_angles = np.linspace(0, 0., 20)
        cir_xs = np.linspace(0.2, -0.8, 20)
        cir_ys = np.linspace(-0.2, 0.1, 5)
        cir_ys = np.concatenate([cir_ys, -cir_ys-0.1, cir_ys, -cir_ys-0.1])
    elif case == 'easy':
        rec_angles = math.pi*np.linspace(0, 0.5, 20)
        rec_xs = np.linspace(0, -0.75, 20)
        rec_ys = np.linspace(0, -0.03, 20)

        tri_angles = np.linspace(0, 0., 20)
        tri_xs = np.linspace(-0.1, 0.1, 20)
        tri_ys = np.linspace(-0.3, 0.3, 20)

        cir_angles = np.linspace(0, 0., 20)
        cir_xs = np.linspace(0.2, -0.8, 20)
        cir_ys = np.linspace(0.1, -0.2, 20)
    elif case == 'rectri':
        rec_angles = math.pi*np.linspace(0, 0.5, 20)
        rec_xs = np.linspace(0, -0.75, 20)
        rec_ys = np.linspace(0, -0.03, 20)

        tri_angles = np.linspace(0, 0., 20)
        tri_xs = np.linspace(-0.1, 0.1, 20)
        tri_ys = np.linspace(-0.3, 0.3, 20)

    recs = torch.zeros((128, 128, 20))
    for i in range(20):
        phi = rec_angles[i]
        rot_m = torch.tensor([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
        tr = torch.tensor([[rec_xs[i]], [rec_ys[i]]])
        theta = torch.concat([rot_m, tr], dim=-1).unsqueeze(0).to(torch.float)
        aff_gird = torch.nn.functional.affine_grid(theta, size=torch.Size([1, 1, 128, 128]), align_corners=False)
        out = torch.nn.functional.grid_sample(rec, aff_gird, align_corners=False, mode='nearest')
        recs[:,:,i] = out.squeeze()

    tris = torch.zeros((128, 128, 20))
    for i in range(20):
        phi = tri_angles[i]
        rot_m = torch.tensor([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
        tr = torch.tensor([[tri_xs[i]], [tri_ys[i]]])
        theta = torch.concat([rot_m, tr], dim=-1).unsqueeze(0).to(torch.float)
        aff_gird = torch.nn.functional.affine_grid(theta, size=torch.Size([1, 1, 128, 128]), align_corners=False)
        out = torch.nn.functional.grid_sample(tri, aff_gird, align_corners=False, mode='nearest')
        tris[:,:,i] = out.squeeze()

    cirs = torch.zeros((128, 128, 20))
    if case is not 'rectri':
        for i in range(20):
            phi = cir_angles[i]
            rot_m = torch.tensor([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
            tr = torch.tensor([[cir_xs[i]], [cir_ys[i]]])
            theta = torch.concat([rot_m, tr], dim=-1).unsqueeze(0).to(torch.float)
            aff_gird = torch.nn.functional.affine_grid(theta, size=torch.Size([1, 1, 128, 128]), align_corners=False)
            out = torch.nn.functional.grid_sample(cir, aff_gird, align_corners=False, mode='nearest')
            cirs[:,:,i] = out.squeeze()

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    np.save(file_name, recs+tris+cirs)