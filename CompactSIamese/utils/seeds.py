import random

import numpy as np
import torch


def set_seed(seed: int = 32, deterministic_torch: bool = False) -> None:
    """Set seeds for reproducibility across random, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
