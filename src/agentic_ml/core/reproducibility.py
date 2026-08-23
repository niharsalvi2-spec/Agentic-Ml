"""
Centralized reproducibility context and seeding utilities.
"""
import os
import random
import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Seed all standard RNGs for deterministic execution."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
