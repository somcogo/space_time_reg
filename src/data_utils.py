import copy
from functools import partial

import torch
import numpy as np
import nibabel as nib
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from src.networks import Siren
from src.siren import training, dataio, modules, loss_functions
from src.utils import generate_coord_tensor

def get_syn_inputs(config):
    imgs = torch.from_numpy(np.load(f'data/syn/{config.dataset}/{config.dataset}.npy').transpose((2, 0, 1)))
    st_dicts = torch.load(f'data/syn/{config.dataset}/{config.dataset}_nrep_st_dicts.pt')
    segs = None
    models = []
    for i in range(len(st_dicts)):
        model = modules.SingleBVPNet(type='sine', mode='mlp', sidelength=imgs.shape[1:], device=config.device)
        model.load_state_dict(st_dicts[i])
        model.eval()
        models.append(model)
    return imgs, models, segs

def get_syn_test_inputs(config):
    imgs = torch.from_numpy(np.load(f'data/syn/rot_slow2/rot_slow2.npy').transpose((2, 0, 1)))
    st_dicts = torch.load(f'data/syn/rot_slow2/rot_slow2_nrep_st_dicts.pt')
    segs = None
    models = []
    for i in range(len(st_dicts)):
        model = modules.SingleBVPNet(type='sine', mode='mlp', sidelength=imgs.shape[1:], device=config.device)
        model.load_state_dict(st_dicts[i])
        model.eval()
        models.append(model)
    imgs = imgs[[0,5]]
    models = [models[0]]
    return imgs, models, segs

def get_large_rotslow_inputs():
    imgs = torch.from_numpy(np.load(f'data/syn/rot_slow2/rot_slow2.npy').transpose((2, 0, 1)))
    st_dicts = torch.load(f'data/syn/rot_slow2/rot_slow2_nrep_st_dicts_large.pt')
    segs = None
    models = []
    model = Siren([2, 256, 256, 256, 1])
    model.load_state_dict(st_dicts)
    model.eval()
    models = [model]
    return imgs, models, segs

def get_rot_slow2_64_inputs():
    imgs = torch.from_numpy(np.load(f'data/syn/rot_slow2/rot_slow2.npy').transpose((2, 0, 1)))
    st_dicts = torch.load(f'data/syn/rot_slow2/rot_slow2_nrep_st_dicts_64x64x64.pt')
    segs = None
    model = Siren([2, 64, 64, 64, 1])
    model.load_state_dict(st_dicts)
    model.eval()
    models = [model]
    return imgs, models, segs

def get_mouse_input():
    imgs = torch.from_numpy(np.load(f'data/cell_tracking/GFP-GOWT1_mouse_stem_small.npy').transpose((2, 0, 1))).float()
    st_dicts = torch.load(f'data/cell_tracking/GFP-GOWT1_mouse_stem_small_nrep.npy')[0]
    segs = torch.from_numpy(np.load(f'data/cell_tracking/GFP-GOWT1_mouse_stem_st_seg_small.npy').transpose((2, 0, 1)))
    model = Siren([2, 256, 256, 256, 1])
    model.load_state_dict(st_dicts)
    model.eval()
    models = [model]
    return imgs, models, segs

def get_mouse_large_input():
    imgs = torch.from_numpy(np.load(f'data/cell_tracking/GFP-GOWT1_mouse_stem.npy').transpose((2, 0, 1))).float()
    segs = torch.from_numpy(np.load(f'data/cell_tracking/GFP-GOWT1_mouse_stem_st_seg.npy').transpose((2, 0, 1)))
    models = [None]
    return imgs, models, segs

def get_mouse_corner_input():
    imgs = torch.from_numpy(np.load(f'data/cell_tracking/GFP-GOWT1_mouse_stem_corner.npy').transpose((2, 0, 1))).float()
    segs = torch.from_numpy(np.load(f'data/cell_tracking/GFP-GOWT1_mouse_stem_corner_seg.npy').transpose((2, 0, 1)))
    models = [None]
    return imgs, models, segs

def get_lung_test_input():
    imgs = torch.from_numpy(np.load(f'data/4D-Lung/first_try.npy'))[:, 25]
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
    segs = None
    models = [None]
    return imgs, models, segs

def get_lung_test_3d_input():
    imgs = torch.from_numpy(np.load(f'data/4D-Lung/first_try.npy'))[[0, 5], :48].permute(0, 2, 3, 1)
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
    segs = None
    models = [None]
    return imgs, models, segs

def get_oasis1_input():
    im1 = nib.load('data/oasis_example/OAS1_0001_MR1/brain.nii.gz').get_fdata()
    seg1 = nib.load('data/oasis_example/OAS1_0001_MR1/brain_aseg.nii.gz').get_fdata()
    im2 = nib.load('data/oasis_example/OAS1_0002_MR1/brain.nii.gz').get_fdata()
    seg2 = nib.load('data/oasis_example/OAS1_0002_MR1/brain_aseg.nii.gz').get_fdata()
    imgs = torch.from_numpy(np.stack([im1, im2], axis=0)).float()
    segs = torch.from_numpy(np.stack([seg1, seg2], axis=0))
    models = [None]
    return imgs, models, segs

def get_oasis2_input():
    im1 = nib.load('data/oasis_example/OAS1_0001_MR1/brain.nii.gz').get_fdata()
    seg1 = nib.load('data/oasis_example/OAS1_0001_MR1/brain_aseg.nii.gz').get_fdata()
    im2 = nib.load('data/oasis_example/OAS1_0001_MR1/brain.nii.gz').get_fdata()
    seg2 = nib.load('data/oasis_example/OAS1_0001_MR1/brain_aseg.nii.gz').get_fdata()
    imgs = torch.from_numpy(np.stack([im1, im2], axis=0)).float()
    segs = torch.from_numpy(np.stack([seg1, seg2], axis=0))
    models = [None]
    return imgs, models, segs

def get_oasis3_input():
    im1 = nib.load('data/oasis_example/OAS1_0002_MR1/brain.nii.gz').get_fdata()
    seg1 = nib.load('data/oasis_example/OAS1_0002_MR1/brain_aseg.nii.gz').get_fdata()
    im2 = nib.load('data/oasis_example/OAS1_0001_MR1/brain.nii.gz').get_fdata()
    seg2 = nib.load('data/oasis_example/OAS1_0001_MR1/brain_aseg.nii.gz').get_fdata()
    imgs = torch.from_numpy(np.stack([im1, im2], axis=0)).float()
    segs = torch.from_numpy(np.stack([seg1, seg2], axis=0))
    models = [None]
    return imgs, models, segs

def prepare_inputs(config):
    if config.dataset in ['easy', 'hard', 'rectri', 'rot', 'rot_slow', 'rot_slow2', 'rec']:
        imgs, models, segs = get_syn_inputs(config)
    elif config.dataset == 'syn_test':
        imgs, models, segs = get_syn_test_inputs(config)
    elif config.dataset == 'rot_slow2_large':
        imgs, models, segs = get_large_rotslow_inputs()
    elif config.dataset == 'rot_slow2_64':
        imgs, models, segs = get_rot_slow2_64_inputs()
    elif config.dataset == 'mouse':
        imgs, models, segs = get_mouse_input()
    elif config.dataset == 'mouse_large':
        imgs, models, segs = get_mouse_large_input()
    elif config.dataset == 'mouse_corner':
        imgs, models, segs = get_mouse_corner_input()
    elif config.dataset == 'lung_test':
        imgs, models, segs = get_lung_test_input()
    elif config.dataset == 'lung_test_3d':
        imgs, models, segs = get_lung_test_3d_input()
    elif config.dataset == 'oasis_examplev1':
        imgs, models, segs = get_oasis1_input()
    elif config.dataset == 'oasis_examplev2':
        imgs, models, segs = get_oasis3_input()
    elif config.dataset == 'oasis_examplev3':
        imgs, models, segs = get_oasis3_input()

    imgs = imgs[config.start_frame:config.start_frame + config.time_points]
    models = models[config.start_frame:config.start_frame + config.time_points]
    segs = segs[config.start_frame:config.start_frame + config.time_points] if segs is not None else None
    return imgs, models, segs



def fit_neural_reps(data, args):
    n_reps = []
    for time_point in range(data.shape[0]):
        lr = 1e-4
        num_epochs = 10000
        steps_til_summary = 1000
        dset = SingleImgDataset(Image.fromarray(data[time_point]))
        if len(data.shape) == 3:
            coord_dataset = dataio.Implicit2DWrapper(dset, sidelength=data.shape[1:], compute_diff='all')
        else:
            coord_dataset = dataio.Implicit3DWrapper(dset, sidelength=data.shape[1:], compute_diff='all')

        dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

        model = modules.SingleBVPNet(type='sine', mode='mlp', sidelength=data.shape[1:], device=args.device)
        model.cuda()

        loss_fn = partial(loss_functions.image_mse, None)

        n_rep = training.train(model=model, train_dataloader=dataloader, epochs=num_epochs, lr=lr,
                    steps_til_summary=steps_til_summary, loss_fn=loss_fn, device=args.device)
        n_reps.append(n_rep)
    return n_reps

def fit_siren_to_img(img, num_epochs=2000, lr=1e-4, device='cuda', layers=[2, 256, 256, 256, 1], min_coord=-1, max_coord=1, omega=30):
    img_shape = img.shape
    rep = Siren(layers=layers, omega=omega)
    rep.to(device)
    optim = Adam(params=rep.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    coord_tensor = generate_coord_tensor(img_shape, device, min_coord=min_coord, max_coord=max_coord)
    gt = torch.from_numpy(img).to(device)

    min_loss = 1e10
    best_epoch = 0
    for epoch in range(num_epochs):
        pred = rep(torch.tensor([], device=device), coord_tensor)
        pred = pred.reshape(img_shape)
        loss = loss_fn(pred, gt.float())
        optim.zero_grad()
        loss.backward()
        optim.step()
        if loss < min_loss:
            min_loss = loss
            best_st_dict = rep.state_dict()
            best_epoch = epoch
        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, curr loss {loss}, best loss {min_loss} from epoch {best_epoch}')
    best_nrep = copy.deepcopy(rep)
    best_nrep.load_state_dict(best_st_dict)

    return best_nrep, best_st_dict

def fit_siren_to_img(img, num_epochs=2000, lr=1e-4, device='cuda', layers=[2, 256, 256, 256, 1], min_coord=-1, max_coord=1, omega=30):
    img_shape = img.shape
    rep = Siren(layers=layers, omega=omega)
    rep.to(device)
    optim = Adam(params=rep.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    coord_tensor = generate_coord_tensor(img_shape, device, min_coord=min_coord, max_coord=max_coord)
    gt = torch.from_numpy(img).to(device)

    min_loss = 1e10
    best_epoch = 0
    for epoch in range(num_epochs):
        pred = rep(torch.tensor([], device=device), coord_tensor)
        pred = pred.reshape(img_shape)
        loss = loss_fn(pred, gt.float())
        optim.zero_grad()
        loss.backward()
        optim.step()
        if loss < min_loss:
            min_loss = loss
            best_st_dict = rep.state_dict()
            best_epoch = epoch
        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, curr loss {loss}, best loss {min_loss} from epoch {best_epoch}')
    best_nrep = copy.deepcopy(rep)
    best_nrep.load_state_dict(best_st_dict)

    return best_nrep, best_st_dict
        
class SingleImgDataset(Dataset):
    def __init__(self, img):
        super().__init__()
        self.img = img
        self.img_channels = 1

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img