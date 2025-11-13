import torch
from flow_vis import flow_to_color

from src.models.siren import Siren
from src.utils.spatial_utils import generate_coord_tensor

def get_gt_model_and_coord_tensor_and_t(device, img_sz, st_pick):
    if st_pick == 'gt':
        state_dict = torch.load('data/gt_state_dicts/rot_slow2_siren_state_dict.pt')
    elif st_pick == 'x_zero':
        state_dict = torch.load('data/gt_state_dicts/rot_slow2_x_is_zero_siren_state_dict.pt')
    elif st_pick == 'y_zero':
        state_dict = torch.load('data/gt_state_dicts/rot_slow2_y_is_zero_siren_state_dict.pt')

    model = Siren(layers=[2, 32, 32, 2])
    model.load_state_dict(state_dict)
    model.to(device)

    coord_tensor = generate_coord_tensor((img_sz, img_sz), device)
    t = torch.tensor(0, device=device)
    
    with torch.no_grad():
        vel = model(t, coord_tensor)
        vel_img = vel.view(img_sz, img_sz, 2).detach().cpu().numpy()
        vel_color = torch.from_numpy(flow_to_color(vel_img, convert_to_bgr=False)).permute(2, 0, 1)

    return model, coord_tensor, t, vel, vel_img, vel_color