import random

import numpy as np
import torch


def ensure_reproducibility(
    *, seed: int = 42, device: torch.device | int | str | None = None
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is None:
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
    else:
        torch.set_default_device(device)
