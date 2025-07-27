import itertools

import pytest
import torch

from baselines.lt2exp import lt2exp as torch_lt2exp
from baselines.lt2exp import lt2exp_einsum as torch_lt2exp_einsum
from baselines.lt2exp import lt2exp_einsum_jit as torch_jit_lt2exp_einsum
from baselines.lt2exp import lt2exp_jit as torch_jit_lt2exp
from triturus.lt2exp import lt2exp, lt2exp_split
from triturus.utils import set_tf32_enabled


@pytest.mark.parametrize("use_split,allow_tf32", itertools.product([False, True], [False, True]))
def test_lt2exp(use_split: bool, allow_tf32: bool):
    set_tf32_enabled(allow_tf32)
    batch, m, j, k, n = 3, 75, 50, 50, 100
    w = torch.rand(batch, m, j, k)
    a = torch.randn(batch, j, n)
    b = torch.randn(batch, k, n)
    gt = torch_lt2exp(w, a, b)
    gt_jit = torch_jit_lt2exp(w, a, b)
    gt_einsum = torch_lt2exp_einsum(w, a, b)
    gt_einsum_jit = torch_jit_lt2exp_einsum(w, a, b)
    assert torch.allclose(gt, gt_jit)
    if allow_tf32:
        assert torch.allclose(gt_einsum, gt_einsum_jit, rtol=5e-5)
        assert torch.allclose(gt, gt_einsum, rtol=5e-5)
        assert torch.allclose(gt_jit, gt_einsum_jit, rtol=5e-5)
    else:
        assert torch.allclose(gt_einsum, gt_einsum_jit)
        assert torch.allclose(gt, gt_einsum)
        assert torch.allclose(gt_jit, gt_einsum_jit)
    if use_split:
        c = lt2exp_split(w, a, b)
    else:
        c = lt2exp(w, a, b)
    if allow_tf32:
        assert torch.allclose(gt, c, rtol=5e-4)
    else:
        assert torch.allclose(gt, c)
