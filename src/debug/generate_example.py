import random

import torch

def generate_sample(circles_nr, direction, max_dist=10, img_size=128, seed=42):
    random.seed(seed)
    out = torch.zeros((2, img_size , img_size))
    for i in range(circles_nr):
        overlap = True
        while overlap:
            x, y, x_off, y_off = generate_new_coords(img_size, max_dist)
            if i == 0 or direction != 'same':
                x_offset = x_off
                y_offset = y_off
            x_offset = min(max(x_offset, -x), img_size-33-x)
            y_offset = min(max(y_offset, -y), img_size-33-y)
            overlap = out[:, x:x+32, y:y+32].sum() > 0 or out[:, x+x_offset:x+x_offset+32, y+y_offset:y+y_offset+32].sum() > 0
        out[0, x:x+32, y:y+32] = 1
        out[1, x+x_offset:x+x_offset+32, y+y_offset:y+y_offset+32] = 1
    return out

def generate_new_coords(img_size, max_dist):
    x = random.randint(0, img_size - 32)
    y = random.randint(0, img_size - 32)
    x_offset = random.randint(-max_dist, max_dist)
    y_offset = random.randint(-max_dist, max_dist)
    return x, y, x_offset, y_offset

# def load_shape(name):
#     img = Image.open(f'data/syn/{name}.png')
#     img = np.array(img, dtype=np.int32)[:,:,0]
#     img = 255 - img
#     img[img==255] = 3
#     img[img==60] = 2
#     img[img==128] = 1
#     img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(torch.float)
#     return img

# def get_circle(x, y, img):
#     # cir = load_shape('cir')[:,:,75:107, 24:56].squeeze()
#     # img[x:x+32, y:y+32] = cir
#     img[x:x+32, y:y+32] = 1
#     return img