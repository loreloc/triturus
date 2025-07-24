import os
import random

import numpy as np
import torch
import triton
import triton.language as tl

_TRITURUS_TF32_ENABLED: bool = False


def ensure_reproducibility(
    *,
    seed: int = 42,
    determinism: bool = False,
    device: torch.device | int | str | None = None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is None:
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
    else:
        torch.set_default_device(device)
    if determinism:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def set_tf32_enabled(allow_tf32: bool):
    global _TRITURUS_TF32_ENABLED
    if allow_tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        _TRITURUS_TF32_ENABLED = True
    else:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        _TRITURUS_TF32_ENABLED = False


def is_torch_tf32_enabled() -> bool:
    return torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32


def is_triturus_tf32_enabled() -> bool:
    global _TRITURUS_TF32_ENABLED
    return _TRITURUS_TF32_ENABLED


@triton.jit
def cast_fp32_to_tf32(x):
    # Useful to get the same precision of torch matmul with tf32 enabled
    # ref: https://github.com/triton-lang/triton/issues/4574#issuecomment-2311136914
    ASM_CAST_TF32: tl.constexpr = "cvt.rna.tf32.f32 $0, $1;"
    return tl.inline_asm_elementwise(
        ASM_CAST_TF32, "=r, r", [x], dtype=tl.float32, is_pure=True, pack=1
    )
