import torch
import triton
import triton.language as tl

from triturus.utils import cast_fp32_to_tf32, is_triturus_tf32_enabled

# Automatically tune the block size and the number of warps,
# for different combinations of the matrix dimensions m, k, n.
#
# In triton, we typically break our kernel into thread blocks, and each of them
# will be run on a streaming multiprocessor (SM). The execution of our kernel is
# complete when all thread blocks have been executed. Since the number of
# thread blocks can be smaller or larger than the number of SMs, a single SM
# executes zero, one, or more than one thread blocks.
#
# In particular, a thread block is a collection of warps, each of them
# consisting of a fixed number of threads. For instance, if each warp consists
# of 32 threads and the computation to be done within a thread block requires 10
# threads, then one warp is launched. Similarly, if we need 40 threads, then two
# warps are launched. The results of the threads in warp that are not needed are
# simply thrown away.
#
# In general, a larger block size means a higher number of warps. The maximum
# number of warps is here automatically tuned together with the block size.
# If the maximum number of warps we set is less than the number of warps that
# would be required, then the computation done by a single thread must increase,
# i.e., by adding loops in a thread during compilation.
#
# Crucially, different combinations of block size and number of warps might
# mean different memory access orderings, which in turn has an impact on
# the resulting performances. More specifically, multiple warps can be executed
# concurrently in a SM. However, context switches of threads across different
# warps typically occur whenever a certain thread is waiting to access the
# memory. While this is useful to reduce time wasted on waiting the memory, it
# can significantly increase the amount of cache required as to save the context
# before switching to a different thread. As the amount of cache is limited,
# the number of warps need to be carefully tuned.
#
# We keep the group size for program id swizzling constant,
# as to reduce the total number of configurations to try. The program id
# swizzling reorders the program ids as to better leverage the cache.
# That is, program ids that would use the same data should be executed
# together. In this case, the same data would be the rows (columns) of
# the matrix A (B), as the same rows and columns are used for filling
# multiple entries of the output tensor.
#
CONFIGS = [
    triton.Config(
        {"BLOCK_SIZE": 32, "BLOCK_SIZE_K": 16, "GROUP_SIZE": 8},
        num_stages=6,
        num_warps=1,
    ),
    triton.Config(
        {"BLOCK_SIZE": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE": 8},
        num_stages=6,
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
        num_stages=4,
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
]


@triton.autotune(configs=CONFIGS, key=["m", "k", "n", "ALLOW_TF32"])
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
    BLOCK_SIZE_K: tl.constexpr,  # The block size along the dimension to contract
    GROUP_SIZE: tl.constexpr,  # The group size used for swizzling
    ALLOW_TF32: tl.constexpr,  # Whether to allow tf32
):
    PRECISION: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"
    # Retrieve the program ids on a 2D grid
    pid = tl.program_id(axis=0)
    num_programs0 = tl.cdiv(m, BLOCK_SIZE)
    num_programs1 = tl.cdiv(n, BLOCK_SIZE)
    pid_i = pid // num_programs1
    pid_j = pid % num_programs1
    pid_i, pid_j = tl.swizzle2d(pid_i, pid_j, num_programs0, num_programs1, GROUP_SIZE)
    # Hint some information to the compiler
    tl.assume(pid_i >= 0)
    tl.assume(pid_j >= 0)
    tl.assume(m > 0)
    tl.assume(k > 0)
    tl.assume(n > 0)
    tl.assume(a_str0 > 0)
    tl.assume(a_str1 > 0)
    tl.assume(b_str0 > 0)
    tl.assume(b_str1 > 0)
    tl.assume(c_str0 > 0)
    tl.assume(c_str1 > 0)
    # Compute the number of blocks to multiply and accumulate, and the block indices
    block_idx = tl.arange(0, BLOCK_SIZE)
    block_idx_k = tl.arange(0, BLOCK_SIZE_K)
    num_blocks = tl.cdiv(k, BLOCK_SIZE_K)
    # Compute the pointers to the starting blocks of A (along axis=0) and B (along axis=1)
    # pid_i, pid_j = (0,0)    (0,1)     (1,0)     (1,1)    ...
    # a_offs0      = [0..32]  [ 0..32]  [32..64]  [32..64] ...
    # b_offs1      = [0..32]  [32..64]  [ 0..32]  [32..64] ...
    a_offs0 = (pid_i * BLOCK_SIZE + block_idx) % m
    b_offs1 = (pid_j * BLOCK_SIZE + block_idx) % n
    # Multiply by the strides for A and B along axes 0 and 1 as to retrieve the pointers
    a_ptrs = a_ptr + a_offs0[:, None] * a_str0 + block_idx_k[None, :] * a_str1
    b_ptrs = b_ptr + block_idx_k[:, None] * b_str0 + b_offs1[None, :] * b_str1
    # Instantiate the accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    # Compute and accumulate the dot products of block matrices
    for _ in range(num_blocks - 1):
        # Load the blocks
        a_block = tl.load(a_ptrs)
        b_block = tl.load(b_ptrs)
        # Compute the dot product of blocks
        if ALLOW_TF32:
            a_block = cast_fp32_to_tf32(a_block)
            b_block = cast_fp32_to_tf32(b_block)
        acc = tl.dot(a_block, b_block, acc=acc, input_precision=PRECISION)
        # Move the pointers for A along axis=1 by the block size
        # Move the pointers for B along axis=0 by the block size
        a_ptrs += BLOCK_SIZE_K * a_str1
        b_ptrs += BLOCK_SIZE_K * b_str0
    # Perform the last dot product operation
    # Handle out of bounds using masks
    mask = block_idx_k + (num_blocks - 1) * BLOCK_SIZE_K < k
    # Since the accumulator has fixed size, we load 0.0 whenever we are out of bounds
    a_block = tl.load(a_ptrs, mask=mask[None, :], other=0.0)
    b_block = tl.load(b_ptrs, mask=mask[:, None], other=0.0)
    # Compute the dot product of blocks
    if ALLOW_TF32:
        a_block = cast_fp32_to_tf32(a_block)
        b_block = cast_fp32_to_tf32(b_block)
    acc = tl.dot(a_block, b_block, acc=acc, input_precision=PRECISION)
    # Store the block accumulator, and use masks
    c_offs0 = pid_i * BLOCK_SIZE + block_idx
    c_offs1 = pid_j * BLOCK_SIZE + block_idx
    c_ptrs = c_ptr + c_offs0[:, None] * c_str0 + c_offs1[None, :] * c_str1
    c_mask = (c_offs0 < m)[:, None] & (c_offs1 < n)[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


def mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert len(a.shape) == len(b.shape) == 2
    assert a.shape[1] == b.shape[0]
    assert a.dtype == b.dtype == torch.float32
    assert a.device == b.device
    assert a.is_contiguous() and b.is_contiguous()

    # Allocate the result tensor, on the same device
    c = torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)
    # The number of kernel instances for each axis
    # In this case it is a tuple of two elements:
    #   the ceiling division of the number of rows in A and the block size;
    #   the ceiling division of the number of columns in B and the block size.
    grid = lambda meta: (
        triton.cdiv(a.shape[0], meta["BLOCK_SIZE"]) * triton.cdiv(b.shape[1], meta["BLOCK_SIZE"]),
    )
    # Launch the kernel and use the given grid settings
    allow_tf32 = is_triturus_tf32_enabled()
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
        ALLOW_TF32=allow_tf32,
    )
    return c
