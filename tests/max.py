import itertools

import pytest
import torch

from triturus.max import matmax, vamax, vmax


@pytest.mark.parametrize("block_size", [2, 32, 128])
def test_max_vamax(block_size: int):
    n = 2000
    x = torch.randn(n)
    gt_m = torch.amax(x, dim=0)
    m = vamax(x, block_size=block_size)
    assert torch.all(gt_m == m)


@pytest.mark.parametrize("block_size", [2, 32, 128])
def test_max_vmax(block_size: int):
    n = 2000
    x = torch.randn(n)
    gt_m, gt_i = torch.max(x, dim=0)
    m, i = vmax(x, block_size=block_size)
    assert torch.all(gt_m == m)
    assert torch.all(gt_i == i)


@pytest.mark.parametrize("block_size,axis", itertools.product([2, 32, 128], [0, 1]))
def test_max_matmax(block_size: int, axis: int):
    m, n = 200, 150
    x = torch.randn(m, n)
    gt_m = torch.amax(x, dim=axis)
    m = matmax(x, axis=axis, block_size=block_size)
    assert torch.all(gt_m == m)
