import torch
import torch.nn.functional as F


class GridSampleTransformer:
    def __init__(self, abs_phi, phi_shape):
        self.in_2d = len(phi_shape) == 4
        grid = abs_phi.reshape(phi_shape)
        self.grid = torch.stack([grid[..., i] for i in reversed(range(grid.shape[-1]))], dim=-1)

    def apply(self, input_img, mode='bilinear'):
        squeeze = self.in_2d and input_img.dim() == 5
        if squeeze:
            input_img = input_img.squeeze(2)
        output_img = F.grid_sample(input_img, self.grid, align_corners=True, mode=mode)
        if squeeze:
            output_img = output_img.unsqueeze(2)
        return output_img
    
class NeuralRepTransformer:
    def __init__(self, abs_phi, phi_shape, use_old_nrep):
        if use_old_nrep:
            self.nrep_applier = OldNeuralRepApplier(abs_phi)
        else:
            self.nrep_applier = NewNeuralRepApplier(abs_phi)
        self.grid_applier = GridSampleTransformer(abs_phi, phi_shape)
        # The nrep is a one-channel image sampled on the motion grid: [T-1, 1, *grid].
        self.img_shape = [-1, 1, *phi_shape[1:-1]]

    def apply(self, input_img, input_is_seg=False):
        if input_is_seg:
            out = self.grid_applier.apply(input_img, mode='nearest')
        else:
            out = self.nrep_applier.apply(input_img)
            out = out.reshape(self.img_shape)
            if self.grid_applier.squeeze_img:
                out = out.unsqueeze(2)  # put the slice axis back, as GridSampleTransformer does
        return out

class OldNeuralRepApplier:
    def __init__(self, abs_phi):
        self.input = abs_phi

    def apply(self, nrep):
        model_out = nrep.net(self.input)
        return (model_out.squeeze(2) + 1) / 2
    
class NewNeuralRepApplier:
    def __init__(self, abs_phi):
        self.input = abs_phi

    def apply(self, nrep):
        model_out = nrep(torch.tensor([], device=self.input.device), self.input)
        return model_out.squeeze(2)

def get_spatial_transformer(abs_phi, phi_shape, config):
    if config.use_nreps:
        use_old_nrep = config.dataset in ['easy', 'hard', 'rectri', 'rot', 'rot_slow', 'rot_slow2', 'rec', 'syn_test']
        return NeuralRepTransformer(abs_phi, phi_shape, use_old_nrep)
    else:
        return GridSampleTransformer(abs_phi, phi_shape)
