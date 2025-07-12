import itertools
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
    num_programs = tl.num_programs(axis=0)
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
    # Store the maximum in the working arrays
    tl.store(m_ptr + pid, m)


def vamax(x: torch.Tensor, *, block_size: int = 128) -> torch.Tensor:
    assert len(x.shape) == 1
    # Compute the number of reduction iterations
    n = x.shape[0]
    assert n > 0
    num_iters = math.ceil(math.log(n, block_size))
    for _ in range(num_iters):
        # Compute the number of programs
        num_programs = triton.cdiv(n, block_size)
        # Allocate the maximum working tensor, on the same device
        stage_maximums = torch.empty(num_programs, dtype=x.dtype, device=x.device)
        # Launch the kernel and use the given block size
        grid = (num_programs,)
        _ker_vamax[grid](x, stage_maximums, n, BLOCK_SIZE=block_size)
        x = stage_maximums
        n = num_programs
    return x[0]


@triton.jit
def _ker_vmax(
    x_ptr,  # A pointer to a 1-dimensional vector
    i_ptr,  # A pointer to the indices of elements in x vector
    m_ptr,  # A pointer to the maximums vector
    j_ptr,  # A pointer to the max indices vector
    n: int,  # The size of the input vector
    BLOCK_SIZE: tl.constexpr,  # The block size
    FIRST_REDUCTION: tl.constexpr,  # Whether this is the first reduction
):
    # Retrieve the program id on a 1D grid
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    # Compute the offsets of each block. E.g., if BLOCK_SIZE = 32 then
    # pid  =  0       1       2       3       ...
    # offs = [0..32][32..64][64..96][96..128] ...
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Compute the mask corresponding to valid data entries
    mask = offs < n
    # Load a block of the input vectors to the DRAM
    x = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))
    # Compute the maximum and its index in the block
    m, j = tl.max(x, axis=0, return_indices=True)
    # Store the maximum and the index in the working arrays
    tl.store(m_ptr + pid, m)
    if FIRST_REDUCTION:
        tl.store(j_ptr + pid, pid * BLOCK_SIZE + j)
    else:
        j = tl.load(i_ptr + pid * BLOCK_SIZE + j)
        tl.store(j_ptr + pid, j)


def vmax(
    x: torch.Tensor, *, block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(x.shape) == 1
    # Compute the number of reduction iterations
    n = x.shape[0]
    assert n > 0
    num_iters = math.ceil(math.log(n, block_size))
    # The maximum indices tensor is not set in the first reduction
    i = None
    for it in range(num_iters):
        # Compute the number of programs
        num_programs = triton.cdiv(n, block_size)
        # Allocate the maximum working tensor, on the same device
        stage_maximums = torch.empty(num_programs, dtype=x.dtype, device=x.device)
        # Allocate the maximum indices working tensor, on the same device
        stage_indices = torch.empty(num_programs, dtype=torch.int64, device=x.device)
        # Launch the kernel and use the given block size
        grid = (num_programs,)
        _ker_vmax[grid](
            x,
            i,
            stage_maximums,
            stage_indices,
            n,
            BLOCK_SIZE=block_size,
            FIRST_REDUCTION=it == 0,
        )
        x = stage_maximums
        i = stage_indices
        n = num_programs
    return x[0], i[0]
