import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _setup_global_state():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_device("cuda")
