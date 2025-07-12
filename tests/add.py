import torch

from triturus.add import vadd


def test_vadd():
    n = 2000
    x = torch.rand(n)
    y = torch.rand(n)
    gt = x + y
    r = vadd(x, y)
    assert torch.allclose(gt, r)
