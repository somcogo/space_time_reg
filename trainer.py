import torch
import torch.nn as nn
from torch.optim import Adam

# TODO: from omnimotion
def gen_grid(h, w, device):
    lin_y = torch.arange(0, h, device=device)
    lin_x = torch.arange(0, w, device=device)
    grid_y, grid_x = torch.meshgrid((lin_y, lin_x))
    grid = torch.stack((grid_x, grid_y), -1)
    return grid  # [h, w, 2]

# TODO: from omnimotion
def normalize_coords(coords, h, w):
    assert coords.shape[-1] == 2
    return coords / torch.tensor([w-1., h-1.], device=coords.device) * 2 - 1.

def apply_flow_to_frame(flow, frame, device):
    #TODO: cite omnimotion
    # flow: (N, H, W, 2), frame: (N, C, H, W)
    h, w = frame.shape[-2:]
    grid = gen_grid(h, w, device=device)
    new_coords = flow + grid
    new_coords_normed = normalize_coords(new_coords, h, w)  # [h, w, 2]
    # print(frame.shape)
    new_frame = nn.functional.grid_sample(frame, new_coords_normed, align_corners=True)
    return new_frame

class SegmentationMaskTrainer():
    def __init__(self, init_data, flows, masks, **kwargs):
        self.init_frame = init_data[0]
        self.frame_number = init_data[1]
        self.flows = flows
        self.consistency_masks = masks[..., 0]
        self.occlusion_masks = masks[..., 1]
        self.device = kwargs['device']
        self.time_window = kwargs['time_window']
        self.config = kwargs

        self.init_seg_masks(init_data)
        self.init_optimizer()

    def init_seg_masks(self, init_data):
        orig_mask = torch.from_numpy(init_data[2]).to(self.device)
        seg_masks = torch.empty((self.frame_number, *orig_mask.shape), device=self.device, dtype=torch.float)
        seg_masks[self.init_frame] = orig_mask

        mask = orig_mask.unsqueeze(0).unsqueeze(0)
        for frame in range(self.init_frame - 1, -1, -1):
            mask = apply_flow_to_frame(self.flows[frame + 1][frame].unsqueeze(0), mask, self.device) > 0.5
            mask = mask.to(dtype=torch.double)
            seg_masks[frame] = mask

        mask = orig_mask.unsqueeze(0).unsqueeze(0)
        for frame in range(self.init_frame + 1, self.frame_number):
            mask = apply_flow_to_frame(self.flows[frame - 1][frame].unsqueeze(0), mask, self.device) > 0.5
            mask = mask.to(dtype=torch.double)
            seg_masks[frame] = mask

        self.seg_masks = []
        for frame in range(self.frame_number):
            if frame == self.init_frame:
                self.seg_masks.append(orig_mask)
            else:
                self.seg_masks.append(nn.Parameter(seg_masks[frame]))

    def init_optimizer(self, ):
        optim = Adam(params=[self.seg_masks[i] for i in range(self.frame_number) if i != self.init_frame],
                     lr=self.config['lr'] if 'lr' in self.config.keys() else 0.001,
                     betas=self.config['betas'] if 'betas' in self.config.keys() else (0.9, 0.999),
                     weight_decay=self.config['weight_decay'] if 'weight_decay' in self.config.keys() else 0)
        self.optim = optim

    def train(self, ):
        pass

    # def calculate_loss(self, ):
    #     pass

    def calculate_gradient(self,):
        stacked_masks = torch.concat([*self.seg_masks], dim=0)
        g_x = stacked_masks[:, 1:] - stacked_masks[:, :-1]
        g_y = stacked_masks[1:, :] - stacked_masks[:-1, :]
        return (g_x**2).sum() + (g_y ** 2).sum()

    def calculate_space_time_reg(self, ):
        # create batched frames and flows to feed into apply_flow_to_frame
        batched_frames = torch.zeros(2*self.time_window, self.frame_number, *self.seg_masks[0].shape, device=self.seg_masks.device, dtype=self.seg_masks.dtype)
        batched_flows = torch.zeros(2*self.time_window, self.frame_number, *self.flows[0].shape, device=self.flows.device, dtype=self.flows.dtype)
        for i in range(self.frame_number):
            frames_before = i - max(0, i - self.time_window)
            frames_after = min(self.frame_number, i + self.time_window) - i
            batched_frames[:frames_before][i] = self.seg_masks[i - frames_before:i]
            batched_frames[-frames_after:][i] = self.seg_masks[i:i + frames_after]
            batched_flows[:frames_before][i] = self.flows[i - frames_before:i]
            batched_flows[-frames_after:][i] = self.flows[i:i + frames_after]

        # calculate masks predicted by flows
        batched_pred = apply_flow_to_frame(batched_flows, torch.broadcast_to(self.seg_masks, batched_frames.shape), batched_frames.device)

        # masked difference
        reg_sum = (batched_pred - batched_frames)**2[self.consistency_masks].sum()


    def compute_metrics(self, ):
        pass

    def log_metrics(self, ):
        pass

    def save_masks(self, ):
        pass
