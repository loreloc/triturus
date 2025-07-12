import torch

from triturus.max import vamax, vmax


def test_vamax():
    n = 2000
    x = torch.randn(n)
    gt_m = torch.amax(x, dim=0)
    m = vamax(x)
    assert torch.allclose(gt_m, m)


def test_vmax():
    n = 2000
    x = torch.randn(n)
    gt_m, gt_i = torch.max(x, dim=0)
    m, i = vmax(x)
    assert torch.allclose(gt_m, m)
    assert torch.allclose(gt_i, i)
