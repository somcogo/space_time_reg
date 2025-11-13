import torch
import torch.nn.functional as F
import numpy as np

def calc_oasis_dice(segs, abs_phi):
    input_seg = segs[:1].expand(segs.shape).unsqueeze(1).float()
    grid = abs_phi.reshape(abs_phi.shape[0], *segs.shape[1:], len(segs.shape[1:]))
    grid = torch.stack([grid[..., i] for i in reversed(range(grid.shape[-1]))], dim=-1)
    pred_seg = F.grid_sample(input_seg, grid, mode='nearest', align_corners=False)
    label = [2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 28, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 60]
    dice = calc_dice(segs[-1].numpy(), pred_seg[-1].numpy(), labels=label)
    return dice, pred_seg.squeeze(1)

def calc_dice(array1, array2, labels):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    """
    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem