import torch

from triturus.max import vmax


def test_vmax():
    n = 1800
    x = torch.rand(n)
    gt_m, gt_p = torch.max(x, dim=0)
    m, p = vmax(x)
    assert torch.allclose(gt_m, m)
    assert torch.allclose(gt_p, p)
