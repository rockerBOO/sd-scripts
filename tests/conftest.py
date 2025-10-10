"""
Global pytest configuration and fixtures for reproducible testing.
"""

import numpy as np
import torch
import pytest
import random


def set_global_seed(seed=42):
    """
    Set a global random seed for all tests to ensure reproducibility.

    Args:
        seed (int, optional): Random seed to use. Defaults to 42.
    """
    # Set Python's built-in random seed
    random.seed(seed)

    # Set NumPy seed
    np.random.seed(seed)

    # Set PyTorch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture(scope="session", autouse=True)
def set_reproducibility():
    """
    Global fixture to set reproducible random seeds for all tests.
    Automatically used for every test session.
    """
    set_global_seed()


