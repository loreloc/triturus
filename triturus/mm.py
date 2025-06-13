import itertools

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
        for bs, nw, ns in itertools.product([32, 64, 128], [2, 4, 8], [3, 4, 5])
    ],
    key=["m", "k", "n"],
)
@triton.jit
def _ker_mm(
    a_ptr,  # A pointer to a M x K matrix (A)
    b_ptr,  # A pointer to a K x N matrix (B)
    c_ptr,  # A pointer to a M x N matrix (C)
    a_str0,  # The stride of A along axis=0
    a_str1,  # The stride of A along axis=1
    b_str0,  # The stride of B along axis=0
    b_str1,  # The stride of B along axis=1
    c_str0,  # The stride of C along axis=0
    c_str1,  # The stride of C along axis=1
    m: int,
    k: int,
    n: int,
    BLOCK_SIZE: tl.constexpr,  # The block size
):
    # Retrieve the program ids on a 2D grid
    pid_i = tl.program_id(axis=0)
    pid_j = tl.program_id(axis=1)
    # Compute the number of blocks to multiply and accumulate, and the block indices
    # E.g., block_idx is [0..32] in the case BLOCK_SIZE=32
    num_blocks = tl.cdiv(k, BLOCK_SIZE)
    block_idx = tl.arange(0, BLOCK_SIZE)
    # Compute the pointers to the starting blocks of A (along axis=0) and B (along axis=1)
    # pid_i, pid_j = (0,0)    (0,1)     (1,0)     (1,1)    ...
    # a_offs0      = [0..32]  [ 0..32]  [32..64]  [32..64] ...
    # b_offs1      = [0..32]  [32..64]  [ 0..32]  [32..64] ...
    a_offs0 = pid_i * BLOCK_SIZE + block_idx
    b_offs1 = pid_j * BLOCK_SIZE + block_idx
    # Multiply by the strides for A and B along axes 0 and 1 as to retrieve the pointers
    a_ptrs = a_ptr + a_offs0[:, None] * a_str0 + block_idx[None, :] * a_str1
    b_ptrs = b_ptr + block_idx[:, None] * b_str0 + b_offs1[None, :] * b_str1
    # Instantiate the accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    # Compute and acucmulate the dot products of block matrices
    for h in range(num_blocks):
        # Handle out of bounds using masks
        mask = (h * BLOCK_SIZE + block_idx) < k
        a_mask = (a_offs0[:, None] < m) & mask[None, :]
        b_mask = (b_offs1[None, :] < n) & mask[:, None]
        # Since the accumulator has fixed size, we load 0.0 whenever we are out of bounds
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        # Compute the dot product of blocks
        acc = tl.dot(a, b, acc=acc, input_precision="ieee")
        # Move the pointers for A along axis=1 by the block size
        # Move the pointers for B along axis=0 by the block size
        a_ptrs += BLOCK_SIZE * a_str1
        b_ptrs += BLOCK_SIZE * b_str0
    # Compute the pointers where to store the accumulator values
    c_ptrs = c_ptr + a_offs0[:, None] * c_str0 + b_offs1[None, :] * c_str1
    # Handle out of bounds using a mask
    c_mask = (a_offs0[:, None] < m) & (b_offs1[None, :] < n)
    # Store the block accumulator
    tl.store(c_ptrs, acc, mask=c_mask)


def mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert len(a.shape) == len(b.shape) == 2
    assert a.shape[1] == b.shape[0]
    assert a.dtype == b.dtype
    assert a.device == b.device
    # Allocate the result tensor, on the same device
    c = torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)
    # The number of kernel instances for each axis
    # In this case it is a tuple of two elements:
    #   the ceiling division of the number of rows in A;
    #   the ceiling division of the number of columns in B.
    grid = lambda meta: (
        triton.cdiv(a.shape[0], meta["BLOCK_SIZE"]),
        triton.cdiv(b.shape[1], meta["BLOCK_SIZE"]),
    )
    # Launch the kernel and use the given block size
    _ker_mm[grid](
        a,
        b,
        c,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        a.shape[0],
        a.shape[1],
        b.shape[1],
    )
    return c
