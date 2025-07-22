import pytest
import torch

from triturus.mm import mm
from triturus.utils import set_tf32_enabled


@pytest.mark.parametrize("allow_tf32", [False, True])
def test_mm(allow_tf32: bool):
    set_tf32_enabled(allow_tf32)
    m, k, n = 150, 100, 200
    a = torch.rand(m, k)
    b = torch.rand(k, n)
    gt = torch.mm(a, b)
    c = mm(a, b)
    assert torch.allclose(gt, c)
