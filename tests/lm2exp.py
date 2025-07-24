import pytest
import torch

from baselines.lm2exp import lm2exp as torch_lm2exp
from baselines.lm2exp import lm2exp_jit as torch_jit_lm2exp
from triturus.lm2exp import lm2exp
from triturus.utils import set_tf32_enabled


@pytest.mark.parametrize("allow_tf32", [False, True])
def test_lm2exp(allow_tf32: bool):
    set_tf32_enabled(allow_tf32)
    batch, m, k, n = 24, 150, 100, 200
    a = torch.rand(batch, m, k)
    b = torch.randn(batch, k, n)
    gt = torch_lm2exp(a, b)
    gt_jit = torch_jit_lm2exp(a, b)
    assert torch.allclose(gt, gt_jit)
    c = lm2exp(a, b)
    if allow_tf32:
        assert torch.allclose(gt, c, rtol=5e-4)
    else:
        assert torch.allclose(gt, c)
