import math

import torch
import triton
import triton.language as tl


@triton.jit
def _ker_vamax(
    x_ptr,  # A pointer to a 1-dimensional vector
    m_ptr,  # A pointer to the maximums vector
    n: int,  # The size of the input vector
    BLOCK_SIZE: tl.constexpr,  # The block size
    STRIDE: tl.constexpr,  # The stride to be used, it is a power of BLOCK_SIZE
):
    # Retrieve the program id on a 1D grid
    pid = tl.program_id(axis=0)
    # Compute the offsets of each block
    BLOCK_STRIDE: tl.constexpr = STRIDE // BLOCK_SIZE
    x_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if BLOCK_STRIDE > 0:
        x_offs = x_offs * BLOCK_STRIDE
        x_offs = tl.multiple_of(x_offs, BLOCK_STRIDE)
    else:
        x_offs = tl.max_contiguous(x_offs, BLOCK_SIZE)
    # Load a block
    mask = x_offs < n
    x = tl.load(x_ptr + x_offs, mask=mask, other=-float("inf"))
    # Compute the maximum in the block
    m = tl.max(x, axis=0)
    # Store the maximum
    tl.store(m_ptr + pid * STRIDE, m)


def vamax(x: torch.Tensor, *, block_size: int = 256) -> torch.Tensor:
    assert len(x.shape) == 1
    n = x.shape[0]
    assert n > 0
    # Compute the number of programs
    num_programs = triton.cdiv(n, block_size)
    # Allocate the maximum working tensor, on the same device
    stage_max = torch.empty(num_programs, dtype=x.dtype, device=x.device)
    # Launch the first reduce kernel and use the given block size
    grid = (num_programs,)
    _ker_vamax[grid](x, stage_max, n, BLOCK_SIZE=block_size, STRIDE=1)
    # Compute the number of reduction iterations
    num_iters = math.ceil(math.log(n, block_size)) - 1
    m, n = len(stage_max), num_programs
    stride = block_size
    for _ in range(num_iters):
        # Compute the number of programs
        num_programs = triton.cdiv(n, block_size)
        # Launch the reduce kernel
        grid = (num_programs,)
        _ker_vamax[grid](stage_max, stage_max, m, BLOCK_SIZE=block_size, STRIDE=stride)
        n = num_programs
        stride *= block_size
    return stage_max[0]


@triton.jit
def _ker_vmax(
    x_ptr,  # A pointer to a 1-dimensional vector
    m_ptr,  # A pointer to the maximums vector
    i_ptr,  # A pointer to the max indices vector
    n: int,  # The size of the input vector
    BLOCK_SIZE: tl.constexpr,  # The block size
    STRIDE: tl.constexpr,  # The stride to be used, it is a power of BLOCK_SIZE
):
    # Retrieve the program id on a 1D grid
    pid = tl.program_id(axis=0)
    # Compute the offsets of each block
    BLOCK_STRIDE: tl.constexpr = STRIDE // BLOCK_SIZE
    x_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if BLOCK_STRIDE > 0:
        x_offs = x_offs * BLOCK_STRIDE
        x_offs = tl.multiple_of(x_offs, BLOCK_STRIDE)
    else:
        x_offs = tl.max_contiguous(x_offs, BLOCK_SIZE)
    # Load a block
    mask = x_offs < n
    x = tl.load(x_ptr + x_offs, mask=mask, other=-float("inf"))
    # Compute the maximum in the block
    m, j = tl.max(x, axis=0, return_indices=True)
    # Store the maximum
    tl.store(m_ptr + pid * STRIDE, m)
    # Store the index
    if BLOCK_STRIDE > 0:
        i = tl.load(i_ptr + (pid * BLOCK_SIZE + j) * BLOCK_STRIDE)
        tl.store(i_ptr + pid * STRIDE, i)
    else:
        tl.store(i_ptr + pid * STRIDE, pid * BLOCK_SIZE + j)


def vmax(
    x: torch.Tensor, *, block_size: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(x.shape) == 1
    n = x.shape[0]
    assert n > 0
    # Compute the number of programs
    num_programs = triton.cdiv(n, block_size)
    # Allocate the maximum working tensor, on the same device
    stage_max = torch.empty(num_programs, dtype=x.dtype, device=x.device)
    # Allocate the maximum indices working tensor, on the same device
    stage_idx = torch.empty(num_programs, dtype=torch.int64, device=x.device)
    # Launch the first reduce kernel and use the given block size
    grid = (num_programs,)
    _ker_vmax[grid](x, stage_max, stage_idx, n, BLOCK_SIZE=block_size, STRIDE=1)
    # Compute the number of reduction iterations
    num_iters = math.ceil(math.log(n, block_size)) - 1
    m, n = len(stage_max), num_programs
    stride = block_size
    for _ in range(num_iters):
        # Compute the number of programs
        num_programs = triton.cdiv(n, block_size)
        # Launch the reduce kernel
        grid = (num_programs,)
        _ker_vmax[grid](
            stage_max, stage_max, stage_idx, m, BLOCK_SIZE=block_size, STRIDE=stride
        )
        n = num_programs
        stride *= block_size
    return stage_max[0], stage_idx[0]
