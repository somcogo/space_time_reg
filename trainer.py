

import torch
import torch.nn as nn

# TODO: from omnimotion
def gen_grid(h, w, device):
    lin_y = torch.arange(0, h, device=device)
    lin_x = torch.arange(0, w, device=device)
    grid_y, grid_x = torch.meshgrid((lin_y, lin_x))
    grid = torch.stack((grid_x, grid_y), -1)
    return grid  # [h, w, 2]

# TODO: from omnimotion
def normalize_coords(coords, h, w, no_shift=False):
    assert coords.shape[-1] == 2
    return coords / torch.tensor([w-1., h-1.], device=coords.device) * 2 - 1.

def apply_flow_to_frame(flow, frame, device):
    #TODO: cite omnimotion
    h, w = frame.shape[-2:]
    grid = gen_grid(h, w, device=device)
    new_coords = flow + grid
    new_coords_normed = normalize_coords(new_coords, h, w)  # [h, w, 2]
    new_frame = nn.functional.grid_sample(frame, new_coords_normed, align_corners=True)
    return new_frame

class SegmentationMaskTrainer():
    def __init__(self, init_data, flows, **kwargs):
        self.init_frame = init_data[0]
        self.frame_number = init_data[1]
        self.flows = flows
        self.device = kwargs['device']

        self.init_seg_masks(init_data)

    def init_seg_masks(self, init_data):
        orig_mask = torch.from_numpy(init_data[2]).to(self.device)
        seg_masks = torch.empty((self.frame_number, *orig_mask.shape), device=self.device, dtype=torch.float)
        seg_masks[self.init_frame] = orig_mask

        mask = orig_mask
        for frame in range(self.init_frame - 1, -1, -1):
            mask = apply_flow_to_frame(mask, self.flows[frame + 1][frame])
            seg_masks[frame] = mask

        mask = orig_mask
        for frame in range(self.init_frame + 1, self.frame_number):
            mask = apply_flow_to_frame(mask, self.flows[frame-1][frame])
            seg_masks[frame - 1] = mask

        self.seg_masks = []
        for frame in range(self.frame_number):
            if frame == self.init_frame:
                self.seg_masks.append(orig_mask)
            else:
                self.seg_masks.append(nn.Parameter(seg_masks[frame]))

    def init_optimizer(self, ):
        pass

    def train(self, ):
        pass

    # def calculate_loss(self, ):
    #     pass

    def calculate_gradient(self, ):
        pass

    def calculate_space_time_reg(self, ):
        pass
    
