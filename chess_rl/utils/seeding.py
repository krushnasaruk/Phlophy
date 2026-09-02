"""
Deterministic pseudo-random number generator management.
Ensures reproducible training, search, and environment initializations.
"""

import os
import random
from typing import Optional
import numpy as np
import torch


def seed_everything(seed: int, deterministic_cudnn: bool = True) -> int:
    """
    Seed all random number generators across Python, NumPy, and PyTorch.

    Args:
        seed: The integer seed to set.
        deterministic_cudnn: Whether to enforce deterministic CUDA convolution kernels.

    Returns:
        The seed integer set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return seed


def get_sub_seed(base_seed: int, offset: int) -> int:
    """
    Derive a deterministic sub-seed from a master seed and integer offset.
    Useful for independent worker threads or environment instances.
    """
    return (base_seed * 1000003 + offset) % (2**31 - 1)
