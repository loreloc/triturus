import torch
import triton
import triton.language as tl


@triton.jit
def _ker_vadd(
    x_ptr,  # x: N-dimensional vector
    y_ptr,  # y: N-dimensional vector
    r_ptr,  # r: N-dimensional vector
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    r = x + y
    tl.store(r_ptr + offs, r, mask=mask)


def vadd(x: torch.Tensor, y: torch.Tensor, *, block_size: int = 32) -> torch.Tensor:
    assert len(x.shape) == len(y.shape) == 1 and x.shape == y.shape
    assert x.device == y.device
    n = x.shape[0]
    r = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _ker_vadd[grid](x, y, r, n, BLOCK_SIZE=block_size)
    return r
