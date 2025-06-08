import torch

from triturus.mm import mm


def test_mm():
    m, k, n = 150, 100, 200
    a = torch.rand(m, k)
    b = torch.rand(k, n)
    gt = torch.mm(a, b)
    c = mm(a, b)
    assert torch.allclose(gt, c)
