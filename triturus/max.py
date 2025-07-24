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
    x = tl.load(x_ptr + x_offs, mask=mask, other=float("-inf"))
    # Compute the maximum in the block
    m = tl.max(x, axis=0)
    # Store the maximum
    tl.store(m_ptr + pid * STRIDE, m)


def vamax(x: torch.Tensor, *, block_size: int = 256) -> torch.Tensor:
    assert len(x.shape) == 1
    assert x.is_floating_point()
    assert x.is_contiguous()
    n = x.shape[0]
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
    x = tl.load(x_ptr + x_offs, mask=mask, other=float("-inf"))
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


def vmax(x: torch.Tensor, *, block_size: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(x.shape) == 1
    assert x.is_floating_point()
    assert x.is_contiguous()
    n = x.shape[0]
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
        _ker_vmax[grid](stage_max, stage_max, stage_idx, m, BLOCK_SIZE=block_size, STRIDE=stride)
        n = num_programs
        stride *= block_size
    return stage_max[0], stage_idx[0]


@triton.jit
def _ker_matmax(
    x_ptr,  # A pointer to a M x N matrix
    y_ptr,  # A pointer to the maximums matrix
    x_str0,  # The stride of X along axis=0
    x_str1,  # The stride of X along axis=1
    y_str0,  # The stride of Y along axis=0
    y_str1,  # The stride of Y along axis=1
    m: int,
    n: int,
    BLOCK_SIZE: tl.constexpr,  # The block size
    STRIDE: tl.constexpr,  # The stride to be used, it is a power of BLOCK_SIZE
):
    # Retrieve the program id on a 2D grid
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    num_programs0 = tl.cdiv(m, BLOCK_SIZE)
    num_programs1 = num_programs // num_programs0
    pid_i = pid // num_programs1
    pid_j = pid % num_programs1
    # Compute the offsets of each block
    BLOCK_STRIDE: tl.constexpr = STRIDE // BLOCK_SIZE
    block_idx = tl.arange(0, BLOCK_SIZE)
    x_offs0 = pid_i * BLOCK_SIZE + block_idx
    x_offs1 = pid_j * BLOCK_SIZE + block_idx
    if BLOCK_STRIDE > 0:
        x_offs1 = x_offs1 * BLOCK_STRIDE
        x_offs1 = tl.multiple_of(x_offs1, BLOCK_STRIDE)
        x_offs0 = tl.max_contiguous(x_offs0, BLOCK_SIZE)
        tl.assume(x_str0 == y_str0)
        tl.assume(x_str1 == y_str1)
    else:
        x_offs0 = tl.max_contiguous(x_offs0, BLOCK_SIZE)
        x_offs1 = tl.max_contiguous(x_offs1, BLOCK_SIZE)
    # Load a block
    mask0 = x_offs0 < m
    mask1 = x_offs1 < n
    x_ptrs = x_ptr + x_offs0[:, None] * x_str0 + x_offs1[None, :] * x_str1
    x = tl.load(x_ptrs, mask=mask0[:, None] & mask1[None, :], other=float("-inf"))
    # Compute the maximum in the block
    y = tl.max(x, axis=1)
    # Store the maximum
    y_ptrs = y_ptr + x_offs0 * y_str0 + pid_j * STRIDE * y_str1
    tl.store(y_ptrs, y, mask=mask0)


def matmax(
    x: torch.Tensor,
    axis: int = 0,
    *,
    block_size: int = 32,
) -> torch.Tensor:
    assert len(x.shape) == 2
    assert x.is_floating_point()
    assert x.is_contiguous()
    assert axis in {0, 1}
    if axis == 0:
        x = x.T
    m, n = x.shape
    # Compute the number of programs
    num_programs_batch = triton.cdiv(m, block_size)
    num_programs_redux = triton.cdiv(n, block_size)
    # Allocate the maximum working tensor, on the same device
    stage_max = torch.empty((m, num_programs_redux), dtype=x.dtype, device=x.device)
    # Launch the first reduce kernel and use the given block size
    grid = (num_programs_batch * num_programs_redux,)
    _ker_matmax[grid](
        x,
        stage_max,
        x.stride(0),
        x.stride(1),
        stage_max.stride(0),
        stage_max.stride(1),
        m,
        n,
        BLOCK_SIZE=block_size,
        STRIDE=1,
    )
    # Compute the number of reduction iterations
    num_iters = math.ceil(math.log(n, block_size)) - 1
    k, n = stage_max.shape[1], num_programs_redux
    stride = block_size
    for _ in range(num_iters):
        # Compute the number of programs
        num_programs_redux = triton.cdiv(n, block_size)
        # Launch the reduce kernel
        grid = (num_programs_batch * num_programs_redux,)
        _ker_matmax[grid](
            stage_max,
            stage_max,
            stage_max.stride(0),
            stage_max.stride(1),
            stage_max.stride(0),
            stage_max.stride(1),
            m,
            k,
            BLOCK_SIZE=block_size,
            STRIDE=stride,
        )
        n = num_programs_redux
        stride *= block_size
    return stage_max[:, 0]
