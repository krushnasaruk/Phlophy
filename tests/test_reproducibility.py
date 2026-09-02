"""
Unit tests for deterministic seeding and random number generation reproducibility.
"""

import random
import numpy as np
import torch

from chess_rl.utils.seeding import seed_everything, get_sub_seed


def test_seed_everything_determinism():
    seed = 42

    # Run 1
    seed_everything(seed)
    py_rand_1 = [random.random() for _ in range(5)]
    np_rand_1 = np.random.rand(5)
    torch_rand_1 = torch.rand(5)

    # Run 2
    seed_everything(seed)
    py_rand_2 = [random.random() for _ in range(5)]
    np_rand_2 = np.random.rand(5)
    torch_rand_2 = torch.rand(5)

    assert py_rand_1 == py_rand_2
    assert np.allclose(np_rand_1, np_rand_2)
    assert torch.allclose(torch_rand_1, torch_rand_2)


def test_get_sub_seed():
    base = 1234
    s1 = get_sub_seed(base, 0)
    s2 = get_sub_seed(base, 1)
    s1_repeat = get_sub_seed(base, 0)

    assert s1 != s2
    assert s1 == s1_repeat
