import itertools

import torch
import triton
import triton.language as tl


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
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs, nw in itertools.product([32, 64, 128], [2, 4, 8])
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
    a_block_mask0 = a_offs0[:, None] < m
    b_block_mask1 = b_offs1[None, :] < n
    # Multiply by the strides for A and B along axes 0 and 1 as to retrieve the pointers
    a_ptrs = a_ptr + a_offs0[:, None] * a_str0 + block_idx[None, :] * a_str1
    b_ptrs = b_ptr + block_idx[:, None] * b_str0 + b_offs1[None, :] * b_str1
    # Instantiate the accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    # Compute and acucmulate the dot products of block matrices
    for h in range(num_blocks):
        # Handle out of bounds using masks
        mask = (h * BLOCK_SIZE + block_idx) < k
        # Since the accumulator has fixed size, we load 0.0 whenever we are out of bounds
        a = tl.load(a_ptrs, mask=a_block_mask0 & mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=b_block_mask1 & mask[:, None], other=0.0)
        # Compute the dot product of blocks
        acc = tl.dot(a, b, acc=acc, input_precision="ieee")
        # Move the pointers for A along axis=1 by the block size
        # Move the pointers for B along axis=0 by the block size
        a_ptrs += BLOCK_SIZE * a_str1
        b_ptrs += BLOCK_SIZE * b_str0
    # Compute the pointers where to store the accumulator values
    c_ptrs = c_ptr + a_offs0[:, None] * c_str0 + b_offs1[None, :] * c_str1
    # Store the block accumulator
    tl.store(c_ptrs, acc, mask=a_block_mask0 & b_block_mask1)


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
