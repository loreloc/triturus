import torch
import triton
import triton.language as tl


@triton.jit
def _ker_vmax(
    x_ptr,  # A pointer to a 1-dimensional vector
    m_ptr,  # A pointer to the maximum
    p_ptr,  # A pointer to the index of the maximum
    lock_ptr,  # A pointer to a lock
    n: int,  # The size of the input vector
    BLOCK_SIZE: tl.constexpr,  # The block size
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
    x = tl.load(x_ptr + offs, mask=mask)
    m, p = tl.max(x, axis=0, return_indices=True)
    # Acquire the lock
    while tl.atomic_cas(lock_ptr, 0, 1) == 1:
        pass
    # Read and conditionally update the maximum and its index
    m_cur = tl.load(m_ptr)
    if m > m_cur:
        tl.store(m_ptr, m)
        tl.store(p_ptr, pid * BLOCK_SIZE + p)
    # Release the lock
    tl.store(lock_ptr, 0)


def vmax(
    x: torch.Tensor, *, block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(x.shape) == 1
    assert x.shape[0] > 0
    n = x.shape[0]
    # Allocate the lock
    lock = torch.zeros(1, dtype=torch.int32, device=x.device)
    # Allocate the maximum and its index tensors, on the same device
    m = x[0]
    p = torch.zeros(1, dtype=torch.int64, device=x.device)
    # The number of kernel instances for each axis
    # In this case is the ceiling division of the vector size (n) and the block size
    grid = (triton.cdiv(n, block_size),)
    # Launch the kernel and use the given block size
    _ker_vmax[grid](x, m, p, lock, n, BLOCK_SIZE=block_size)
    return m, p
