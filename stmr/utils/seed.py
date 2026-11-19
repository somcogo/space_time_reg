import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed the global torch / random / numpy RNGs (network init, k-space and init noise).
    The random sampling masks do not use these; they have their own RNG (config.mask_seed)."""
    torch.manual_seed(seed)
    random.seed(seed + 1)
    np.random.seed(seed + 2)
