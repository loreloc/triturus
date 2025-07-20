import torch
import triton
import triton.language as tl

CONFIGS = [
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16, "GROUP_SIZE": 4},
        num_stages=6,
        num_warps=1,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 16, "GROUP_SIZE": 4},
        num_stages=6,
        num_warps=1,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE": 4},
        num_stages=2,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE": 4},
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 16, "GROUP_SIZE": 4},
        num_stages=2,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 16, "GROUP_SIZE": 4},
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 16, "GROUP_SIZE": 4},
        num_stages=2,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 16, "GROUP_SIZE": 4},
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE": 4},
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16, "GROUP_SIZE": 4},
        num_stages=3,
        num_warps=2,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 16, "GROUP_SIZE": 4},
        num_stages=3,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 16, "GROUP_SIZE": 4},
        num_stages=3,
        num_warps=8,
    ),
]


@triton.autotune(
    configs=CONFIGS,
    key=["m", "k", "n"],
)
@triton.jit
def _ker_logmm2exp(
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
    BLOCK_SIZE_M: tl.constexpr,  # The block size along axis=0 of A
    BLOCK_SIZE_N: tl.constexpr,  # The block size along axis=1 of B
    BLOCK_SIZE_K: tl.constexpr,  # The block size along the dimension to contract
    GROUP_SIZE: tl.constexpr,  # The group size used for swizzling
    TILE_SIZE: tl.constexpr,  # The size of a tile column of B, i.e., next_power_of_2(k)
    PRECISION: tl.constexpr,  # It can be either 'tf32' or 'ieee'
):
    # Retrieve the program ids on a 2D grid
    pid = tl.program_id(axis=0)
    num_programs0 = tl.cdiv(m, BLOCK_SIZE_M)
    num_programs1 = tl.cdiv(n, BLOCK_SIZE_N)
    pid_i = pid // num_programs1
    pid_j = pid % num_programs1
    pid_i, pid_j = tl.swizzle2d(pid_i, pid_j, num_programs0, num_programs1, GROUP_SIZE)
    # Hint some information to the compiler
    tl.assume(pid_i >= 0)
    tl.assume(pid_j >= 0)
    tl.assume(m > 0)
    tl.assume(n > 0)
    tl.assume(k > 0)
    tl.assume(a_str0 > 0)
    tl.assume(a_str1 > 0)
    tl.assume(b_str0 > 0)
    tl.assume(b_str1 > 0)
    tl.assume(c_str0 > 0)
    tl.assume(c_str1 > 0)
    # Compute the number of blocks to multiply and accumulate, and the block indices
    num_blocks = tl.cdiv(k, BLOCK_SIZE_K)
    block_idx_k = tl.arange(0, BLOCK_SIZE_K)
    # Compute the pointers to the starting blocks of A (along axis=0) and B (along axis=1)
    a_offs0 = (pid_i * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % m
    b_offs1 = (pid_j * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % n
    # Compute the maximum values
    tile_idx = tl.arange(0, TILE_SIZE) % n
    b_tile_ptrs = b_ptr + tile_idx[:, None] * b_str0 + b_offs1[None, :] * b_str1
    b_tile = tl.load(b_tile_ptrs)
    max_block = tl.clamp(tl.max(b_tile, axis=0), -1e38, 1e38)
    # Multiply by the strides for A and B along axes 0 and 1 as to retrieve the pointers
    a_ptrs = a_ptr + a_offs0[:, None] * a_str0 + block_idx_k[None, :] * a_str1
    b_ptrs = b_ptr + block_idx_k[:, None] * b_str0 + b_offs1[None, :] * b_str1
    # Instantiate the accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Compute and accumulate the dot products of block matrices
    for _ in range(num_blocks - 1):
        # Load the blocks
        a_block = tl.load(a_ptrs)
        b_block = tl.load(b_ptrs)
        # Exponentiate the values in the B matrix ...
        # ... but first subtract the maximum values for numerical stability
        # Compute the dot product of blocks
        acc = tl.dot(
            a_block,
            tl.exp(b_block - max_block),
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
    # Compute the dot product of blocks
    acc = tl.dot(
        a_block,
        tl.exp(b_block - max_block),
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

    # Allocate the result tensor, on the same device
    c = torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)
    # Compute the tile size, which will be used to compute the maximum along multiple block columns in B
    tile_size = triton.next_power_of_2(b.shape[0])
    # Launch the kernel
    grid = lambda meta: (
        triton.cdiv(a.shape[0], meta["BLOCK_SIZE_M"])
        * triton.cdiv(b.shape[1], meta["BLOCK_SIZE_N"]),
    )
    _ker_logmm2exp[grid](
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
        TILE_SIZE=tile_size,
        PRECISION="tf32" if use_tf32 else "ieee",
    )
    return c
