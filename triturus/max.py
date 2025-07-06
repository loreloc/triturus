import math

import torch
import triton
import triton.language as tl


@triton.jit
def _ker_vmax(
    x_ptr,  # A pointer to a 1-dimensional vector
    m_ptr,  # A pointer to the maximums vector
    p_ptr,  # A pointer to the index of the maximums
    out_m_ptr,  # A pointer to the maximum output
    out_p_ptr,  # A pointer to the index of the maximum output
    n: int,  # The size of the input vector
    BLOCK_SIZE: tl.constexpr,  # The block size
    NUM_ITERS: tl.constexpr,  # The number of iterations, i.e., proportional to ceil(log2(n))
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
    m, p = tl.max(x, axis=0, return_indices=True)
    # Store the maximum and the index in the working arrays
    tl.store(m_ptr + pid, m)
    tl.store(p_ptr + pid, pid * BLOCK_SIZE + p)
    # SYNC threads
    tl.debug_barrier()
    # Proceed repeatedly compute maximum and index in the working ararys
    for _ in range(NUM_ITERS):
        n = tl.cdiv(n, BLOCK_SIZE)
        mask = offs < n
        # Load from the working arary and compute the maximum and its index in the block
        x = tl.load(m_ptr + offs, mask=mask, other=-float("inf"))
        m, s = tl.max(x, axis=0, return_indices=True)
        # Retrieve the index within the the input vector
        pid_mask = pid < n
        p = tl.load(p_ptr + s, mask=pid_mask)
        # Store the maximum and the index in the working arrays
        tl.store(m_ptr + pid, m, mask=pid_mask)
        tl.store(p_ptr + pid, p, mask=pid_mask)
        # SYNC threads
        tl.debug_barrier()
    # Load and store the results at the first position of the maximums and indices working arrays
    m = tl.load(m_ptr)
    p = tl.load(p_ptr)
    tl.store(out_m_ptr, m)
    tl.store(out_p_ptr, p)


def vmax(x: torch.Tensor, *, block_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(x.shape) == 1
    assert x.shape[0] > 0
    n = x.shape[0]
    # The number of kernel instances for each axis
    # In this case is the ceiling division of the vector size (n) and the block size
    num_programs = triton.cdiv(n, block_size)
    # Allocate the maximum and its index working tensors, on the same device
    m = torch.empty(num_programs, dtype=x.dtype, device=x.device)
    p = torch.empty(num_programs, dtype=torch.int64, device=x.device)
    # Allocate the maximum and index results
    out_m = torch.empty(1, dtype=x.dtype, device=x.device)
    out_p = torch.empty(1, dtype=torch.int64, device=x.device)
    # Compute the number of iterations
    num_iters = math.ceil(math.log(n, block_size)) - 1
    # Launch the kernel and use the given block size
    grid = (num_programs,)
    _ker_vmax[grid](
        x, m, p, out_m, out_p, n, BLOCK_SIZE=block_size, NUM_ITERS=num_iters
    )
    return out_m, out_p
