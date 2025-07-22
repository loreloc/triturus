import torch
import triton
import triton.language as tl

from triturus.utils import cast_fp32_to_tf32, is_triturus_tf32_enabled

CONFIGS = [
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 16, "GROUP_SIZE": 8},
        num_stages=6,
        num_warps=1,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE": 8},
        num_stages=6,
        num_warps=2,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 16, "GROUP_SIZE": 8},
        num_stages=5,
        num_warps=2,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE": 8},
        num_stages=5,
        num_warps=2,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE": 8},
        num_stages=5,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE": 8},
        num_stages=5,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE": 8},
        num_stages=4,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE": 8},
        num_stages=4,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE": 8},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE": 8},
        num_stages=3,
        num_warps=8,
    ),
]


@triton.autotune(configs=CONFIGS, key=["batch", "m", "k", "n"])
@triton.jit
def _ker_lm2exp(
    a_ptr,  # A pointer to a batch x M x K matrix (A)
    b_ptr,  # A pointer to a batch x K x N matrix (B)
    c_ptr,  # A pointer to a batch x M x N matrix (C)
    a_str0,  # The stride of A along axis=0
    a_str1,  # The stride of A along axis=1
    a_str2,  # The stride of A along axis=2
    b_str0,  # The stride of B along axis=0
    b_str1,  # The stride of B along axis=1
    b_str2,  # The stride of B along axis=2
    c_str0,  # The stride of C along axis=0
    c_str1,  # The stride of C along axis=1
    c_str2,  # The stride of C along axis=2
    batch: int,  # The batch dimension
    m: int,
    k: int,
    n: int,
    BLOCK_SIZE_M: tl.constexpr,  # The block size along axis=1 of A
    BLOCK_SIZE_N: tl.constexpr,  # The block size along axis=2 of B
    BLOCK_SIZE_K: tl.constexpr,  # The block size along the dimension to contract
    GROUP_SIZE: tl.constexpr,  # The group size used for swizzling
    TILE_SIZE: tl.constexpr,  # The size of a tile column of B, i.e., next_power_of_2(k)
    ALLOW_TF32: tl.constexpr,  # Whether to allow tf32
):
    PRECISION: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"
    # Retrieve the program ids on a 3D grid
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    num_programs12 = num_programs // batch
    num_programs1 = tl.cdiv(m, BLOCK_SIZE_M)
    num_programs2 = num_programs12 // num_programs1
    pid_batch = pid // num_programs12
    pid_i = (pid % num_programs12) // num_programs2
    pid_j = (pid % num_programs12) % num_programs2
    pid_i, pid_j = tl.swizzle2d(pid_i, pid_j, num_programs1, num_programs2, GROUP_SIZE)
    # Hint some information to the compiler
    tl.assume(pid_batch >= 0)
    tl.assume(pid_i >= 0)
    tl.assume(pid_j >= 0)
    tl.assume(m > 0)
    tl.assume(k > 0)
    tl.assume(n > 0)
    tl.assume(a_str0 > 0)
    tl.assume(a_str1 > 0)
    tl.assume(a_str2 > 0)
    tl.assume(b_str0 > 0)
    tl.assume(b_str1 > 0)
    tl.assume(b_str2 > 0)
    tl.assume(c_str0 > 0)
    tl.assume(c_str1 > 0)
    tl.assume(c_str2 > 0)
    # Move the pointers based on the batch program id
    a_ptr += pid_batch * a_str0
    b_ptr += pid_batch * b_str0
    c_ptr += pid_batch * c_str0
    # Compute the number of blocks to multiply and accumulate, and the block indices
    num_blocks = tl.cdiv(k, BLOCK_SIZE_K)
    block_idx_k = tl.arange(0, BLOCK_SIZE_K)
    # Compute the pointers to the starting blocks of A (along axis=0) and B (along axis=1)
    a_offs1 = (pid_i * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % m
    b_offs2 = (pid_j * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % n
    # Compute the maximum values
    tile_idx = tl.arange(0, TILE_SIZE) % k
    b_tile_ptrs = b_ptr + tile_idx[:, None] * b_str1 + b_offs2[None, :] * b_str2
    b_tile = tl.load(b_tile_ptrs)
    max_block = tl.clamp(tl.max(b_tile, axis=0), -1e38, 1e38)
    # Multiply by the strides for A and B along axes 1 and 2 as to retrieve the pointers
    a_ptrs = a_ptr + a_offs1[:, None] * a_str1 + block_idx_k[None, :] * a_str2
    b_ptrs = b_ptr + block_idx_k[:, None] * b_str1 + b_offs2[None, :] * b_str2
    # Instantiate the accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Compute and accumulate the dot products of block matrices
    for h in range(num_blocks):
        # Load the blocks
        if h < num_blocks - 1:
            a_block = tl.load(a_ptrs)
            b_block = tl.load(b_ptrs)
        else:
            # Handle out of bounds using masks
            mask = block_idx_k + h * BLOCK_SIZE_K < k
            # Since the accumulator has fixed size, we load 0.0 whenever we are out of bounds
            a_block = tl.load(a_ptrs, mask=mask[None, :], other=0.0)
            b_block = tl.load(b_ptrs, mask=mask[:, None], other=0.0)
        # Exponentiate the values in the B matrix ...
        # ... but first subtract the maximum values for numerical stability
        b_block_exp = tl.exp(b_block - max_block)
        # Compute the dot product of blocks
        if ALLOW_TF32:
            a_block = cast_fp32_to_tf32(a_block)
            b_block_exp = cast_fp32_to_tf32(b_block_exp)
        acc = tl.dot(
            a_block,
            b_block_exp,
            acc=acc,
            input_precision=PRECISION,
        )
        # Move the pointers for A along axis=2 by the block size
        # Move the pointers for B along axis=1 by the block size
        a_ptrs += BLOCK_SIZE_K * a_str2
        b_ptrs += BLOCK_SIZE_K * b_str1
    # Compute the logarithm of the accumulator, and add the maximum values back
    log_acc = max_block + tl.log(acc)
    # Compute the pointers where to store the accumulator values
    c_ptrs = c_ptr + a_offs1[:, None] * c_str1 + b_offs2[None, :] * c_str2
    # Store the block accumulator, and use masks
    block_mask1 = a_offs1 < m
    block_mask2 = b_offs2 < n
    tl.store(c_ptrs, log_acc, mask=block_mask1[:, None] & block_mask2[None, :])


def lm2exp(
    a: torch.Tensor, b: torch.Tensor, *, allow_tf32: bool = False
) -> torch.Tensor:
    assert len(a.shape) == len(b.shape) == 3
    assert a.shape[0] == b.shape[0]
    assert a.shape[2] == b.shape[1]
    assert a.dtype == b.dtype == torch.float32
    assert a.device == b.device

    # Allocate the result tensor, on the same device
    c = torch.empty(
        (a.shape[0], a.shape[1], b.shape[2]), dtype=a.dtype, device=a.device
    )
    # Specify how to compute the number of programs required in the grid
    grid = lambda meta: (
        a.shape[0]
        * triton.cdiv(a.shape[1], meta["BLOCK_SIZE_M"])
        * triton.cdiv(b.shape[2], meta["BLOCK_SIZE_N"]),
    )
    # Compute the tile size, which will be used to compute the maximum along multiple block columns in B
    tile_size = triton.next_power_of_2(b.shape[0])
    # Launch the kernel
    allow_tf32 = is_triturus_tf32_enabled()
    _ker_lm2exp[grid](
        a,
        b,
        c,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        a.shape[0],
        a.shape[1],
        a.shape[2],
        b.shape[2],
        TILE_SIZE=tile_size,
        ALLOW_TF32=allow_tf32,
    )
    return c
