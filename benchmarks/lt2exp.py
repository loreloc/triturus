from collections.abc import Callable

import torch
import triton

from baselines.lt2exp import lt2exp as torch_lt2exp
from baselines.lt2exp import lt2exp_einsum as torch_lt2exp_einsum
from baselines.lt2exp import lt2exp_einsum_jit as torch_jit_lt2exp_einsum
from baselines.lt2exp import lt2exp_jit as torch_jit_lt2exp
from benchmarks.utils import QUANTILES, eval_gbps
from triturus.lt2exp import lt2exp
from triturus.utils import ensure_reproducibility, set_tf32_enabled


class Providers:
    TORCH = "torch"
    TORCH_EINSUM = "torch (einsum)"
    TORCH_JIT = "torch (jit)"
    TORCH_JIT_EINSUM = "torch (jit, einsum)"
    TRITURUS = "triturus"


PROVIDERS = [
    Providers.TORCH,
    Providers.TORCH_EINSUM,
    Providers.TORCH_JIT,
    Providers.TORCH_JIT_EINSUM,
    Providers.TRITURUS,
]


CONFIGS = [
    *tuple(
        triton.testing.Benchmark(
            x_names=["m", "j", "k", "n"],
            x_vals=[64, 128, 256, 512],
            x_log=True,
            line_arg="provider",
            line_vals=PROVIDERS,
            line_names=PROVIDERS,
            ylabel="GB/s",
            plot_name=f"lt2exp (square matrices, batch={1} allow_tf32={allow_tf32})",
            args={"batch": 1, "allow_tf32": allow_tf32},
        )
        for allow_tf32 in [False, True]
    ),
    *tuple(
        triton.testing.Benchmark(
            x_names=["m", "j", "k"],
            x_vals=[64, 128, 256],
            x_log=True,
            line_arg="provider",
            line_vals=PROVIDERS,
            line_names=PROVIDERS,
            ylabel="GB/s",
            plot_name=f"lt2exp (rectangular matrices, n={n} batch={batch} allow_tf32={allow_tf32})",
            args={"n": n, "batch": batch, "allow_tf32": allow_tf32},
        )
        for batch in [16, 32, 64, 128]
        for n in [256]
        for allow_tf32 in [False, True]
    ),
]


@triton.testing.perf_report(CONFIGS)
def benchmark_lt2exp(
    batch, m, j, k, n, provider, *, allow_tf32: bool = False
) -> tuple[float, float, float]:
    ensure_reproducibility()
    set_tf32_enabled(allow_tf32)
    w = torch.rand(batch, m, j, k)
    a = torch.randn(batch, j, n)
    b = torch.randn(batch, k, n)
    fn: Callable[[], torch.Tensor]
    match provider:
        case Providers.TORCH:
            fn = lambda: torch_lt2exp(w, a, b)
        case Providers.TORCH_EINSUM:
            fn = lambda: torch_lt2exp_einsum(w, a, b)
        case Providers.TORCH_JIT:
            fn = lambda: torch_jit_lt2exp(w, a, b)
        case Providers.TORCH_JIT_EINSUM:
            fn = lambda: torch_jit_lt2exp_einsum(w, a, b)
        case Providers.TRITURUS:
            fn = lambda: lt2exp(w, a, b)
        case _:
            assert False, provider
    ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=200, rep=1500, quantiles=QUANTILES)
    size = batch * (m * j * k + j * n + k * n)
    return (
        eval_gbps(size, ms),
        eval_gbps(size, max_ms),
        eval_gbps(size, min_ms),
    )


if __name__ == "__main__":
    benchmark_lt2exp.run(print_data=True)
