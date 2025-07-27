from typing import Any, cast

import torch
import triton
import triton.language as tl

from triturus.utils import is_triturus_tf32_enabled

CONFIGS = [
    triton.Config(
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_INNER": 32, "GROUP_SIZE": 4},
        num_stages=5,
        num_warps=1,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_INNER": 32, "GROUP_SIZE": 4},
        num_stages=5,
        num_warps=1,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_INNER": 32, "GROUP_SIZE": 4},
        num_stages=5,
        num_warps=1,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_INNER": 16, "GROUP_SIZE": 4},
        num_stages=6,
        num_warps=1,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_INNER": 32, "GROUP_SIZE": 4},
        num_stages=6,
        num_warps=1,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_INNER": 16, "GROUP_SIZE": 4},
        num_stages=5,
        num_warps=2,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_INNER": 32, "GROUP_SIZE": 4},
        num_stages=4,
        num_warps=2,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_INNER": 32, "GROUP_SIZE": 4},
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_INNER": 32, "GROUP_SIZE": 4},
        num_stages=5,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_INNER": 32, "GROUP_SIZE": 4},
        num_stages=4,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_INNER": 64, "GROUP_SIZE": 4},
        num_stages=4,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_INNER": 32, "GROUP_SIZE": 4},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_INNER": 64, "GROUP_SIZE": 4},
        num_stages=3,
        num_warps=8,
    ),
]


@triton.autotune(configs=CONFIGS, key=["batch", "m", "j", "k", "n", "ALLOW_TF32"])
@triton.jit
def _ker_lt2exp(
    w_ptr,  # A pointer to a batch x M x (J x K) tensor (W)
    a_ptr,  # A pointer to a batch x J x N tensor (A)
    b_ptr,  # A pointer to a batch x K x N tensor (B)
    c_ptr,  # A pointer to a batch x M x N tensor (C)
    w_str0,  # The stride of W along axis=0
    w_str1,  # The stride of W along axis=1
    w_str2,  # The stride of W along axis=2
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
    j: int,
    k: int,
    n: int,
    BLOCK_SIZE_M: tl.constexpr,  # The block size along axis=1 of W
    BLOCK_SIZE_N: tl.constexpr,  # The block size along axis=2 of A (or B)
    BLOCK_SIZE_INNER: tl.constexpr,  # The block size along the dimension to contract
    GROUP_SIZE: tl.constexpr,  # The group size used for swizzling
    A_TILE_SIZE: tl.constexpr,  # The size of a tile column of A, i.e., next_power_of_2(j)
    B_TILE_SIZE: tl.constexpr,  # The size of a tile column of B, i.e., next_power_of_2(k)
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
    tl.assume(pid_batch >= 0)
    tl.assume(pid_i >= 0)
    tl.assume(pid_j >= 0)
    tl.assume(m > 0)
    tl.assume(j > 0)
    tl.assume(k > 0)
    tl.assume(n > 0)
    tl.assume(w_str0 > 0)
    tl.assume(w_str1 > 0)
    tl.assume(w_str2 > 0)
    tl.assume(a_str0 > 0)
    tl.assume(a_str1 > 0)
    tl.assume(a_str2 > 0)
    tl.assume(b_str0 > 0)
    tl.assume(b_str1 > 0)
    tl.assume(b_str2 > 0)
    tl.assume(c_str0 > 0)
    tl.assume(c_str1 > 0)
    tl.assume(c_str2 > 0)
    # Compute the number of blocks to multiply and accumulate, and the block indices
    num_blocks = tl.cdiv(j * k, BLOCK_SIZE_INNER)
    block_idx = tl.arange(0, BLOCK_SIZE_INNER)
    # Compute the pointers to the starting blocks of W (along axis=1)
    w_offs1 = (pid_i * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % m
    # Compute the pointers to the starting blocks of A and B (along axis=2 for both)
    ab_offs2 = (pid_j * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % n
    # Move the pointers based on the batch program id,
    # and based on the offsets for W, A, B, C
    w_ptrs = w_ptr + pid_batch * w_str0 + w_offs1[:, None] * w_str1 + block_idx[None, :] * w_str2
    a_ptrs = a_ptr + pid_batch * a_str0 + ab_offs2[None, :] * a_str2
    b_ptrs = b_ptr + pid_batch * b_str0 + ab_offs2[None, :] * b_str2
    # Compute the maximum values of the column tiles of A
    a_tile_idx = tl.arange(0, A_TILE_SIZE) % j
    a_tile = tl.load(a_ptrs + a_tile_idx[:, None] * a_str1)
    a_max_block = tl.clamp(tl.max(a_tile, axis=0), -1e38, 1e38)
    # Compute the maximum values of the column tiles of B
    b_tile_idx = tl.arange(0, B_TILE_SIZE) % k
    b_tile = tl.load(b_ptrs + b_tile_idx[:, None] * b_str1)
    b_max_block = tl.clamp(tl.max(b_tile, axis=0), -1e38, 1e38)
    # Instantiate the accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Compute and accumulate the dot products of block matrices
    for h in range(num_blocks):
        block_idx_h = block_idx + h * BLOCK_SIZE_INNER
        block_idx_h = tl.max_contiguous(block_idx_h, BLOCK_SIZE_INNER)
        a_block_idx = block_idx_h // k
        b_block_idx = block_idx_h % k
        a_offs1 = a_block_idx[:, None] * a_str1
        b_offs1 = b_block_idx[:, None] * b_str1
        if h < num_blocks - 1:
            # Load the weights
            w = tl.load(w_ptrs)
            # Load the values of A
            a_block = tl.load(a_ptrs + a_offs1)
        else:
            # Load the weights, and handle out of bounds using masks
            mask = block_idx_h < j * k
            w = tl.load(w_ptrs, mask=mask[None, :], other=0.0)
            # Load the values of A, and handle out of bounds using masks
            a_block = tl.load(a_ptrs + a_offs1, mask=mask[:, None], other=float("-inf"))
        # Load the values of B
        b_block = tl.load(b_ptrs + b_offs1)
        # Exponentiate the values in the A and B matrices ...
        # ... but first subtract the maximum values for numerical stability
        a_block_exp = tl.exp(a_block - a_max_block)
        b_block_exp = tl.exp(b_block - b_max_block)
        # Compute the dot product of blocks
        acc = tl.dot(
            w,
            a_block_exp * b_block_exp,
            acc=acc,
            input_precision=PRECISION,
        )
        # Move the pointers for B along axis=2 by the block size
        w_ptrs += BLOCK_SIZE_INNER * w_str2
    # Compute the logarithm of the accumulator, and add the maximum values back
    log_acc = a_max_block + b_max_block + tl.log(acc)
    # Store the block accumulator, and use masks
    c_offs1 = pid_i * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_offs2 = pid_j * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_offs1 = tl.max_contiguous(c_offs1, BLOCK_SIZE_M)
    c_offs2 = tl.max_contiguous(c_offs2, BLOCK_SIZE_N)
    c_ptrs = c_ptr + pid_batch * c_str0 + c_offs1[:, None] * c_str1 + c_offs2[None, :] * c_str2
    c_mask = (c_offs1 < m)[:, None] & (c_offs2 < n)[None, :]
    tl.store(c_ptrs, log_acc, mask=c_mask)


def lt2exp(w: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert len(w.shape) == 4
    assert len(a.shape) == len(b.shape) == 3
    assert w.shape[0] == a.shape[0] == b.shape[0]
    assert w.shape[2] == a.shape[1]
    assert w.shape[3] == b.shape[1]
    assert a.shape[2] == b.shape[2]
    assert w.dtype == a.dtype == b.dtype == torch.float32
    assert w.device == a.device == b.device
    batch, m, j, k, n = w.shape[0], w.shape[1], w.shape[2], w.shape[3], a.shape[2]

    # Allocate the result tensor, on the same device
    c = torch.empty((batch, m, n), dtype=a.dtype, device=a.device)
    # Specify how to compute the number of programs required in the grid
    grid = lambda meta: (
        batch * triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
    )
    # Compute the tile sizes, which will be used to compute the maximum along multiple block columns in A and B
    a_tile_size = triton.next_power_of_2(j)
    b_tile_size = triton.next_power_of_2(k)
    # Reshape the weights tensor as to flatten the last two dimensions
    w = w.view(batch, m, -1)
    # Launch the kernel
    allow_tf32 = is_triturus_tf32_enabled()
    _ker_lt2exp[grid](
        w,
        a,
        b,
        c,
        w.stride(0),
        w.stride(1),
        w.stride(2),
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        batch,
        m,
        j,
        k,
        n,
        A_TILE_SIZE=a_tile_size,
        B_TILE_SIZE=b_tile_size,
        ALLOW_TF32=allow_tf32,
    )
    return c


CONFIGS_SPLIT = [
    *tuple(
        triton.Config(
            {
                "BLOCK_SIZE_M": bs,
                "BLOCK_SIZE_N": bs,
                "BLOCK_SIZE_INNER": bsi,
                "GROUP_SIZE": 4,
                "NUM_SPLITS": ns,
            },
            num_stages=4,
            num_warps=4,
        )
        for ns in [4, 8, 16]
        for bsi in [16, 32]
        for bs in [16, 32]
    )
]


@triton.autotune(
    configs=CONFIGS_SPLIT,
    key=["batch", "m", "j", "k", "n", "NUM_SPLITS", "ALLOW_TF32"],
)
@triton.jit
def _ker_lt2exp_split(
    w_ptr,  # A pointer to a batch x M x (J x K) tensor (W)
    a_ptr,  # A pointer to a batch x J x N tensor (A)
    b_ptr,  # A pointer to a batch x K x N tensor (B)
    c_ptr,  # A pointer to a batch x M x N tensor (C)
    lock_ptr,  # A pointer to a batch x R x S tensor containing lock entries
    l_str0,  # The stride of the lock along axis=0
    l_str1,  # The stride of the lock along axis=1
    l_str2,  # The stride of the lock along axis=2
    w_str0,  # The stride of W along axis=0
    w_str1,  # The stride of W along axis=1
    w_str2,  # The stride of W along axis=2
    a_str0,  # The stride of A along axis=0
    a_str1,  # The stride of A along axis=1
    a_str2,  # The stride of A along axis=2
    b_str0,  # The stride of B along axis=0
    b_str1,  # The stride of B along axis=1
    b_str2,  # The stride of B along axis=2
    c_str0,  # The stride of C along axis=0
    c_str1,  # The stride of C along axis=1
    c_str2,  # The stride of C along axis=2
    batch: int,
    m: int,
    j: int,
    k: int,
    n: int,
    BLOCK_SIZE_M: tl.constexpr,  # The block size along axis=1 of W
    BLOCK_SIZE_N: tl.constexpr,  # The block size along axis=2 of A (or B)
    BLOCK_SIZE_INNER: tl.constexpr,  # The block size along the dimension to contract
    GROUP_SIZE: tl.constexpr,  # The group size used for swizzling
    NUM_SPLITS: tl.constexpr,  # The number of splits
    A_TILE_SIZE: tl.constexpr,  # The size of a tile column of A, i.e., next_power_of_2(j)
    B_TILE_SIZE: tl.constexpr,  # The size of a tile column of B, i.e., next_power_of_2(k)
    ALLOW_TF32: tl.constexpr,  # Whether to allow tf32
):
    PRECISION: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"
    INTERLEAVED_TILE_SIZE: tl.constexpr = NUM_SPLITS * BLOCK_SIZE_INNER
    # Retrieve the program ids on a 4D grid
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    num_programs12 = num_programs // batch
    num_programs1 = tl.cdiv(m, BLOCK_SIZE_M)
    num_programs2 = num_programs12 // num_programs1
    pid_batch = pid // num_programs12
    pid_i = (pid % num_programs12) // num_programs2
    pid_j = (pid % num_programs12) % num_programs2
    pid_i, pid_j = tl.swizzle2d(pid_i, pid_j, num_programs1, num_programs2, GROUP_SIZE)
    pid_inn = tl.program_id(axis=1)
    # Some hints to the compiler
    tl.assume(pid_batch >= 0)
    tl.assume(pid_i >= 0)
    tl.assume(pid_j >= 0)
    tl.assume(pid_inn >= 0)
    tl.assume(m > 0)
    tl.assume(j > 0)
    tl.assume(k > 0)
    tl.assume(n > 0)
    tl.assume(l_str0 > 0)
    tl.assume(l_str1 > 0)
    tl.assume(l_str2 > 0)
    tl.assume(w_str0 > 0)
    tl.assume(w_str1 > 0)
    tl.assume(w_str2 > 0)
    tl.assume(a_str0 > 0)
    tl.assume(a_str1 > 0)
    tl.assume(a_str2 > 0)
    tl.assume(b_str0 > 0)
    tl.assume(b_str1 > 0)
    tl.assume(b_str2 > 0)
    tl.assume(c_str0 > 0)
    tl.assume(c_str1 > 0)
    tl.assume(c_str2 > 0)
    # Move the pointers based on the batch program id,
    # and based on the offsets for W, A, B, C
    block_idx1 = tl.arange(0, BLOCK_SIZE_M)
    block_idx2 = tl.arange(0, BLOCK_SIZE_N)
    block_idx = pid_inn * BLOCK_SIZE_INNER + tl.arange(0, BLOCK_SIZE_INNER)
    block_idx = tl.max_contiguous(block_idx, BLOCK_SIZE_INNER)
    w_offs1 = (pid_i * BLOCK_SIZE_M + block_idx1) % m
    ab_offs2 = (pid_j * BLOCK_SIZE_N + block_idx2) % n
    w_ptrs = w_ptr + pid_batch * w_str0 + w_offs1[:, None] * w_str1 + block_idx[None, :] * w_str2
    a_ptrs = a_ptr + pid_batch * a_str0 + ab_offs2[None, :] * a_str2
    b_ptrs = b_ptr + pid_batch * b_str0 + ab_offs2[None, :] * b_str2
    # Compute the maximum values of the column tiles of A
    a_tile_idx = tl.arange(0, A_TILE_SIZE) % j
    a_tile = tl.load(a_ptrs + a_tile_idx[:, None] * a_str1)
    a_max_block = tl.clamp(tl.max(a_tile, axis=0), -1e38, 1e38)
    # Compute the maximum values of the column tiles of B
    b_tile_idx = tl.arange(0, B_TILE_SIZE) % k
    b_tile = tl.load(b_ptrs + b_tile_idx[:, None] * b_str1)
    b_max_block = tl.clamp(tl.max(b_tile, axis=0), -1e38, 1e38)
    # Instantiate the accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Compute and accumulate the dot products of block matrices
    num_blocks = tl.cdiv(j * k, INTERLEAVED_TILE_SIZE)
    for h in range(num_blocks):
        block_idx_h = block_idx + h * INTERLEAVED_TILE_SIZE
        block_idx_h = tl.max_contiguous(block_idx_h, BLOCK_SIZE_INNER)
        # Load the weights, and handle out of bounds using masks
        # Load the values of A and B, and handle out of bounds using masks
        a_block_idx = block_idx_h // k
        b_block_idx = block_idx_h % k
        a_offs1 = a_block_idx[:, None] * a_str1
        b_offs1 = b_block_idx[:, None] * b_str1
        if h < num_blocks - 1:
            w = tl.load(w_ptrs)
            a_block = tl.load(a_ptrs + a_offs1)
        else:
            mask = block_idx_h < j * k
            w = tl.load(w_ptrs, mask=mask[None, :], other=0.0)
            a_block = tl.load(a_ptrs + a_offs1, mask=mask[:, None], other=float("-inf"))
        b_block = tl.load(b_ptrs + b_offs1)
        # Exponentiate the values in the A and B matrices ...
        # ... but first subtract the maximum values for numerical stability
        a_block_exp = tl.exp(a_block - a_max_block)
        b_block_exp = tl.exp(b_block - b_max_block)
        # Compute the dot product of blocks
        acc = tl.dot(
            w,
            a_block_exp * b_block_exp,
            acc=acc,
            input_precision=PRECISION,
        )
        # Move the pointers for B along axis=2 by the block size
        w_ptrs += INTERLEAVED_TILE_SIZE * w_str2
    # Atomically add the accumulated results
    c_offs1 = pid_i * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_offs2 = pid_j * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = (c_offs1 < m)[:, None] & (c_offs2 < n)[None, :]
    c_ptrs = c_ptr + pid_batch * c_str0 + c_offs1[:, None] * c_str1 + c_offs2[None, :] * c_str2
    lock_ptr += pid_batch * l_str0 + pid_i * l_str1 + pid_j * l_str2
    while tl.atomic_cas(lock_ptr, 0, 1, sem="acquire") == 1:
        pass
    cur_log_c = tl.load(c_ptrs, mask=c_mask, other=float("-inf"))
    log_c = a_max_block + b_max_block + tl.log(acc + tl.exp(cur_log_c - a_max_block - b_max_block))
    tl.store(c_ptrs, log_c, mask=c_mask)
    # Ensure all threads have finished before releasing the lock
    tl.debug_barrier()
    tl.atomic_xchg(lock_ptr, 0, sem="release")


def _ker_lt2exp_split_prepare(*args: Any, **_: Any) -> None:
    c = cast(torch.Tensor, args[3])
    lock = cast(torch.Tensor, args[4])
    c.fill_(float("-inf"))
    lock.zero_()


_ker_lt2exp_split.fn.add_pre_run_hook(_ker_lt2exp_split_prepare)


def lt2exp_split(w: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert len(w.shape) == 4
    assert len(a.shape) == len(b.shape) == 3
    assert w.shape[0] == a.shape[0] == b.shape[0]
    assert w.shape[2] == a.shape[1]
    assert w.shape[3] == b.shape[1]
    assert a.shape[2] == b.shape[2]
    assert w.dtype == a.dtype == b.dtype == torch.float32
    assert w.device == a.device == b.device
    batch, m, j, k, n = w.shape[0], w.shape[1], w.shape[2], w.shape[3], a.shape[2]

    # Allocate the lock
    MIN_BLOCK_SIZE = 32
    lock = torch.empty(
        batch,
        triton.cdiv(m, MIN_BLOCK_SIZE),
        triton.cdiv(n, MIN_BLOCK_SIZE),
        dtype=torch.int32,
        device=a.device,
    )
    # Allocate the result tensor, on the same device
    c = torch.empty(batch, m, n, dtype=a.dtype, device=a.device)
    grid = lambda meta: (
        batch * triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
        meta["NUM_SPLITS"],
    )
    # Compute the tile sizes, which will be used to compute the maximum along multiple block columns in A and B
    a_tile_size = triton.next_power_of_2(j)
    b_tile_size = triton.next_power_of_2(k)
    # Reshape the weights tensor as to flatten the last two dimensions
    w = w.view(batch, m, -1)
    # Launch the kernel to compute the exponentiated dot products
    allow_tf32 = is_triturus_tf32_enabled()
    _ker_lt2exp_split[grid](
        w,
        a,
        b,
        c,
        lock,
        lock.stride(0),
        lock.stride(1),
        lock.stride(2),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        batch,
        m,
        j,
        k,
        n,
        A_TILE_SIZE=a_tile_size,
        B_TILE_SIZE=b_tile_size,
        ALLOW_TF32=allow_tf32,
    )
    return c
