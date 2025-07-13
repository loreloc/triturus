import torch

from baselines.logmm2exp import logmm2exp as torch_logmm2exp
from baselines.logmm2exp import logmm2exp_jit as torch_logmm2exp_jit
from triturus.logmm2exp import logmm2exp


def test_logmm2exp():
    m, k, n = 150, 100, 200
    a = torch.rand(m, k)
    b = torch.randn(k, n)
    gt = torch_logmm2exp(a, b)
    gt_jit = torch_logmm2exp_jit(a, b)
    c = logmm2exp(a, b)
    assert torch.allclose(gt, gt_jit)
    assert torch.allclose(gt, c)
