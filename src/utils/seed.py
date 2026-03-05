"""
seed.py
─────────────────────────────────────────────────────────────────────────────
Deterministic seeding for reproducible experiments.

Call set_seed() once at the top of each notebook.

Sources of randomness fixed:
    - Python's built-in random module
    - NumPy
    - PyTorch CPU operations
    - PyTorch CUDA operations (GPU)

Usage:
    from src.utils.seed import set_seed

    set_seed(42)
─────────────────────────────────────────────────────────────────────────────
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Fix all sources of randomness for fully reproducible experiments.

    Args:
        seed: integer seed value. Use the same value across all notebooks
              to ensure CNN and Transformer experiments are comparable.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)   # for multi-GPU setups

    print(f"Seed set to {seed}")
