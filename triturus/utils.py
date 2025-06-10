import os
import random

import numpy as np
import torch


def ensure_reproducibility(
    *,
    determinism: bool = False,
    seed: int = 42,
    device: torch.device | int | str | None = None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is None:
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
    else:
        torch.set_default_device(device)
    if determinism:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
