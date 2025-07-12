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
):
    # Retrieve the program id on a 1D grid
    pid = tl.program_id(axis=0)
    # Compute the offsets of each block. E.g., if BLOCK_SIZE = 32 then
    # pid  =  0       1       2       3       ...
    # offs = [0..32][32..64][64..96][96..128] ...
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Compute the mask corresponding to valid data entries
    mask = offs < n
    # Load a block of the input vectors to the DRAM
    x = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))
    # Compute the maximum in the block
    m = tl.max(x, axis=0)
    # Store the maximum in the maximums vector
    tl.store(m_ptr + pid, m)


@triton.jit
def _ker_vamax_reduce(
    m_ptr,  # A pointer to the maximums vector
    n: int,  # The size of the maximums vector
    BLOCK_SIZE: tl.constexpr,  # The block size
    STRIDE: tl.constexpr,  # The stride, which increases after each reduction stage
):
    # Retrieve the program id on a 1D grid
    pid = tl.program_id(axis=0)
    # Compute the offsets of each block
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Compute the mask corresponding to valid data entries
    mask = offs * STRIDE < n
    # Load a block of the input vectors to the DRAM
    x = tl.load(m_ptr + offs * STRIDE, mask=mask, other=-float("inf"))
    # Compute the maximum in the block
    m = tl.max(x, axis=0)
    # Store the maximum in the maximums vector
    store_offs = pid * STRIDE * BLOCK_SIZE
    tl.store(m_ptr + store_offs, m)


def vamax(x: torch.Tensor, *, block_size: int = 128) -> torch.Tensor:
    assert len(x.shape) == 1
    n = x.shape[0]
    assert n > 0
    # Compute the number of programs
    num_programs = triton.cdiv(n, block_size)
    # Allocate the maximum working tensor, on the same device
    stage_max = torch.empty(num_programs, dtype=x.dtype, device=x.device)
    # Launch the first reduce kernel and use the given block size
    grid = (num_programs,)
    _ker_vamax[grid](x, stage_max, n, BLOCK_SIZE=block_size)
    # Compute the number of reduction iterations
    num_iters = math.ceil(math.log(n, block_size)) - 1
    m, n = len(stage_max), num_programs
    stride = 1
    for _ in range(num_iters):
        # Compute the number of programs
        num_programs = triton.cdiv(n, block_size)
        # Launch the reduce kernel
        grid = (num_programs,)
        _ker_vamax_reduce[grid](stage_max, m, BLOCK_SIZE=block_size, STRIDE=stride)
        n = num_programs
        stride *= block_size
    return stage_max[0]


@triton.jit
def _ker_vmax(
    x_ptr,  # A pointer to a 1-dimensional vector
    m_ptr,  # A pointer to the maximums vector
    i_ptr,  # A pointer to the max indices vector
    n: int,  # The size of the maximums vector
    BLOCK_SIZE: tl.constexpr,  # The block size
):
    # Retrieve the program id on a 1D grid
    pid = tl.program_id(axis=0)
    # Compute the offsets of each block. E.g., if BLOCK_SIZE = 32 then
    # pid  =  0       1       2       3       ...
    # offs = [0..32][32..64][64..96][96..128] ...
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Compute the mask corresponding to valid data entries
    mask = offs < n
    # Load a block of the input vectors to the DRAM
    x = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))
    # Compute the maximum and its index in the block
    m, i = tl.max(x, axis=0, return_indices=True)
    # Store the maximum and the index in the working arrays
    tl.store(m_ptr + pid, m)
    tl.store(i_ptr + pid, pid * BLOCK_SIZE + i)


@triton.jit
def _ker_vmax_reduce(
    m_ptr,  # A pointer to the maximums vector
    i_ptr,  # A pointer to the maximums indices vector
    n: int,  # The size of the maximums vector
    BLOCK_SIZE: tl.constexpr,  # The block size
    STRIDE: tl.constexpr,  # The stride, which increases after each reduction stage
):
    # Retrieve the program id on a 1D grid
    pid = tl.program_id(axis=0)
    # Compute the offsets of each block. E.g., if BLOCK_SIZE = 32 then
    # pid  =  0       1       2       3       ...
    # offs = [0..32][32..64][64..96][96..128] ...
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Compute the mask corresponding to valid data entries
    mask = offs * STRIDE < n
    # Load a block of the input vectors to the DRAM
    x = tl.load(m_ptr + offs * STRIDE, mask=mask, other=-float("inf"))
    # Compute the maximum and its index in the block
    m, j = tl.max(x, axis=0, return_indices=True)
    store_offs = pid * STRIDE * BLOCK_SIZE
    tl.store(m_ptr + store_offs, m)
    i = tl.load(i_ptr + (pid * BLOCK_SIZE + j) * STRIDE)
    tl.store(i_ptr + store_offs, i)


def vmax(
    x: torch.Tensor, *, block_size: int = 128
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
    _ker_vmax[grid](x, stage_max, stage_idx, n, BLOCK_SIZE=block_size)
    # Compute the number of reduction iterations
    num_iters = math.ceil(math.log(n, block_size)) - 1
    m, n = len(stage_max), num_programs
    stride = 1
    for _ in range(num_iters):
        # Compute the number of programs
        num_programs = triton.cdiv(n, block_size)
        # Launch the reduce kernel
        grid = (num_programs,)
        _ker_vmax_reduce[grid](
            stage_max, stage_idx, m, BLOCK_SIZE=block_size, STRIDE=stride
        )
        n = num_programs
        stride *= block_size
    return stage_max[0], stage_idx[0]
