import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE": 64, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE": 64, "BLOCK_SIZE_K": 64}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE": 128, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE": 128, "BLOCK_SIZE_K": 64}, num_stages=4, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE": 128, "BLOCK_SIZE_K": 128}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE": 256, "BLOCK_SIZE_K": 64}, num_stages=2, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE": 512, "BLOCK_SIZE_K": 128}, num_stages=2, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE": 512, "BLOCK_SIZE_K": 256}, num_stages=2, num_warps=8
        ),
    ],
    key=["m", "k", "n"],
)
@triton.jit
def _ker_colsmax(
    b_ptr,  # A pointer to a K x N matrix (B)
    m_ptr,  # A pointer to a N-dimensional vector, storing the maximums of B along axis=0
    b_str0,  # The stride of B along axis=0
    b_str1,  # The stride of B along axis=1
    k: int,
    n: int,
    BLOCK_SIZE: tl.constexpr,  # The block size
    BLOCK_SIZE_K: tl.constexpr,  # The block size along the dimension to contract
):
    # Retrieve the program ids on a 1D grid
    pid_i = tl.program_id(axis=0)
    # Compute the maximum values of B along axis=0
    block_idx = tl.arange(0, BLOCK_SIZE)
    block_idx_k = tl.arange(0, BLOCK_SIZE_K)
    num_blocks = tl.cdiv(k, BLOCK_SIZE_K)
    # Compute the pointers to the starting blocks of B (along axis=1)
    b_offs1 = (pid_i * BLOCK_SIZE + block_idx) % n
    b_ptrs = b_ptr + block_idx_k[:, None] * b_str0 + b_offs1[None, :] * b_str1
    # Compute the number of blocks along axis=0 and reduce to the maximum values
    max_block = tl.full((BLOCK_SIZE,), value=float("-inf"), dtype=tl.float32)
    for _ in range(num_blocks - 1):
        # Read the block columns and reduce to the maximum values
        block = tl.load(b_ptrs)
        max_block = tl.maximum(max_block, tl.max(block, axis=0))
        b_ptrs += BLOCK_SIZE_K * b_str0
    # Read the block columns and reduce to the maximum values
    mask = block_idx_k + (num_blocks - 1) * BLOCK_SIZE_K < k
    block = tl.load(b_ptrs, mask=mask[:, None], other=float("-inf"))
    max_block = tl.maximum(max_block, tl.max(block, axis=0))
    # Store the maximum values
    m_ptrs = m_ptr + b_offs1
    block_mask1 = b_offs1 < n
    tl.store(m_ptrs, max_block, mask=block_mask1)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE": 16, "BLOCK_SIZE_K": 16, "GROUP_SIZE": 8},
            num_stages=5,
            num_warps=1,
        ),
        triton.Config(
            {"BLOCK_SIZE": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE": 8},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE": 8},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE": 8},
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE": 8},
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE": 8},
            num_stages=2,
            num_warps=8,
        ),
    ],
    key=["m", "k", "n"],
)
@triton.jit
def _ker_logmm2exp(
    a_ptr,  # A pointer to a M x K matrix (A)
    b_ptr,  # A pointer to a K x N matrix (B)
    c_ptr,  # A pointer to a M x N matrix (C)
    m_ptr,  # A pointer to the maximum of the columns of B
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
    BLOCK_SIZE_K: tl.constexpr,  # The block size along the dimension to contract
    GROUP_SIZE: tl.constexpr,  # The group size used for swizzling
    PRECISION: tl.constexpr,  # It can be either 'tf32' or 'ieee'
):
    # Retrieve the program ids on a 2D grid
    pid = tl.program_id(axis=0)
    num_programs0 = tl.cdiv(m, BLOCK_SIZE)
    num_programs1 = tl.cdiv(n, BLOCK_SIZE)
    pid_i = pid // num_programs1
    pid_j = pid % num_programs1
    pid_i, pid_j = tl.swizzle2d(pid_i, pid_j, num_programs0, num_programs1, GROUP_SIZE)
    # Compute the number of blocks to multiply and accumulate, and the block indices
    block_idx = tl.arange(0, BLOCK_SIZE)
    block_idx_k = tl.arange(0, BLOCK_SIZE_K)
    num_blocks = tl.cdiv(k, BLOCK_SIZE_K)
    # Compute the pointers to the starting blocks of A (along axis=0) and B (along axis=1)
    a_offs0 = (pid_i * BLOCK_SIZE + block_idx) % m
    b_offs1 = (pid_j * BLOCK_SIZE + block_idx) % n
    # Multiply by the strides for A and B along axes 0 and 1 as to retrieve the pointers
    a_ptrs = a_ptr + a_offs0[:, None] * a_str0 + block_idx_k[None, :] * a_str1
    b_ptrs = b_ptr + block_idx_k[:, None] * b_str0 + b_offs1[None, :] * b_str1
    # Load the maximum values
    max_block = tl.load(m_ptr + b_offs1)
    # Instantiate the accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    # Compute and accumulate the dot products of block matrices
    for _ in range(num_blocks - 1):
        # Load the blocks
        a_block = tl.load(a_ptrs)
        b_block = tl.load(b_ptrs)
        # Exponentiate the values in the B matrix ...
        # ... but first subtract the maximum values for numerical stability
        exp_b_block = tl.exp(b_block - max_block)
        # Compute the dot product of blocks
        acc = tl.dot(
            a_block,
            exp_b_block,
            acc=acc,
            input_precision=PRECISION,
        )
        # Move the pointers for A along axis=1 by the block size
        # Move the pointers for B along axis=0 by the block size
        a_ptrs += BLOCK_SIZE_K * a_str1
        b_ptrs += BLOCK_SIZE_K * b_str0
    # Handle out of bounds using masks
    mask = block_idx_k + (num_blocks - 1) * BLOCK_SIZE_K < k
    # Since the accumulator has fixed size, we load 0.0 whenever we are out of bounds
    a_block = tl.load(a_ptrs, mask=mask[None, :], other=0.0)
    b_block = tl.load(b_ptrs, mask=mask[:, None], other=0.0)
    # Exponentiate the values in the B matrix
    exp_b_block = tl.exp(b_block - max_block)
    # Compute the dot product of blocks
    acc = tl.dot(
        a_block,
        exp_b_block,
        acc=acc,
        input_precision=PRECISION,
    )
    # Compute the logarithm of the accumulator, and add the maximum values back
    log_acc = max_block + tl.log(acc)
    # Compute the pointers where to store the accumulator values
    c_ptrs = c_ptr + a_offs0[:, None] * c_str0 + b_offs1[None, :] * c_str1
    # Store the block accumulator, and use masks
    block_mask0 = a_offs0 < m
    block_mask1 = b_offs1 < n
    tl.store(c_ptrs, log_acc, mask=block_mask0[:, None] & block_mask1[None, :])


def logmm2exp(
    a: torch.Tensor, b: torch.Tensor, *, use_tf32: bool = False
) -> torch.Tensor:
    assert len(a.shape) == len(b.shape) == 2
    assert a.shape[1] == b.shape[0]
    assert a.dtype == b.dtype == torch.float32
    assert a.device == b.device
    assert a.is_contiguous() and b.is_contiguous()

    # Allocate the maximum values and the result tensors, on the same device
    m = torch.empty((b.shape[1]), dtype=a.dtype, device=a.device)
    c = torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)
    # Launch the kernel to compute the maximum values of the columns of B first
    grid_colsmax = lambda meta: (triton.cdiv(b.shape[1], meta["BLOCK_SIZE"]),)
    _ker_colsmax[grid_colsmax](
        b,
        m,
        b.stride(0),
        b.stride(1),
        b.shape[0],
        b.shape[1],
    )
    # Launch the kernel
    grid_logmm2exp = lambda meta: (
        triton.cdiv(a.shape[0], meta["BLOCK_SIZE"])
        * triton.cdiv(b.shape[1], meta["BLOCK_SIZE"]),
    )
    _ker_logmm2exp[grid_logmm2exp](
        a,
        b,
        c,
        m,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        a.shape[0],
        a.shape[1],
        b.shape[1],
        PRECISION="tf32" if use_tf32 else "ieee",
    )
    return c
