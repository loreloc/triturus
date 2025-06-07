import torch

from triturus.vadd import vadd


def test_vadd():
    n = 1800
    x = torch.rand(n)
    y = torch.rand(n)
    gt = x + y
    r = vadd(x, y)
    assert torch.allclose(gt, r)
