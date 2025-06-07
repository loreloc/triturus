import torch
import triton
import triton.language as tl


@triton.jit
def _ker_vadd(
    x_ptr: triton.language.pointer_type,  # A pointer to n-dimensional vector
    y_ptr: triton.language.pointer_type,
    r_ptr: triton.language.pointer_type,
    n: int,  # The size of the input and output vectors
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
    y = tl.load(y_ptr + offs, mask=mask)
    # Compute the element-wise sum of blocks
    r = x + y
    # Store the block
    tl.store(r_ptr + offs, r, mask=mask)


def vadd(x: torch.Tensor, y: torch.Tensor, *, block_size: int = 32) -> torch.Tensor:
    assert len(x.shape) == len(y.shape) == 1 and x.shape == y.shape
    assert x.device == y.device
    n = x.shape[0]
    # Allocate the result tensor, on the same device
    r = torch.empty_like(x)
    # The callable returning the number of kernel instances for each axis
    # In this case is the ceiling division of the vector sizes (n) and the block size
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    # Launch the kernel and use the given block size
    _ker_vadd[grid](x, y, r, n, BLOCK_SIZE=block_size)
    return r
