import torch

from baselines.logmm2exp import logmm2exp as torch_logmm2exp
from triturus.logmm2exp import logmm2exp, logmm2exp_fused


def test_logmm2exp():
    m, k, n = 150, 100, 200
    a = torch.rand(m, k)
    b = torch.randn(k, n)
    gt = torch_logmm2exp(a, b)
    c = logmm2exp(a, b)
    assert torch.allclose(gt, c)


def test_logmm2exp_fused():
    m, k, n = 150, 100, 200
    a = torch.rand(m, k)
    b = torch.randn(k, n)
    gt = torch_logmm2exp(a, b)
    c = logmm2exp_fused(a, b)
    assert torch.allclose(gt, c)
