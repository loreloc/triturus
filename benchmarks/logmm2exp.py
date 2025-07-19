from collections.abc import Callable

import torch
import triton

from baselines.logmm2exp import logmm2exp as torch_logmm2exp
from baselines.logmm2exp import logmm2exp_jit as torch_logmm2exp_jit
from benchmarks.utils import QUANTILES, eval_gbps
from triturus.logmm2exp import logmm2exp, logmm2exp_fused
from triturus.utils import ensure_reproducibility


class Providers:
    TORCH = "torch"
    TORCH_JIT = "torch (jit)"
    TRITURUS = "triturus"
    TRITURUS_MAXFUSED = "triturus (max-fused)"


CONFIGS = [
    triton.testing.Benchmark(
        x_names=["m", "k", "n"],
        x_vals=[32, 48, 128, 192, 512, 768, 2048, 3072, 8192],
        x_log=True,
        line_arg="provider",
        line_vals=[
            Providers.TORCH,
            Providers.TORCH_JIT,
            Providers.TRITURUS,
            Providers.TRITURUS_MAXFUSED,
        ],
        line_names=[
            Providers.TORCH,
            Providers.TORCH_JIT,
            Providers.TRITURUS,
            Providers.TRITURUS_MAXFUSED,
        ],
        ylabel="GiB/s",
        plot_name="logmm2exp performance",
        args={},
    )
]


@triton.testing.perf_report(CONFIGS)
def benchmark_logmm2exp(m, k, n, provider) -> tuple[float, float, float]:
    ensure_reproducibility()
    a = torch.rand(m, k)
    b = torch.randn(k, n)
    fn: Callable[[], torch.Tensor]
    match provider:
        case Providers.TORCH:
            fn = lambda: torch_logmm2exp(a, b)
        case Providers.TORCH_JIT:
            fn = lambda: torch_logmm2exp_jit(a, b)
        case Providers.TRITURUS:
            fn = lambda: logmm2exp(a, b)
        case Providers.TRITURUS_MAXFUSED:
            fn = lambda: logmm2exp_fused(a, b)
        case _:
            assert False, provider
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=QUANTILES)
    size = m * k + k * n
    return (
        eval_gbps(size, ms),
        eval_gbps(size, min_ms),
        eval_gbps(size, max_ms),
    )


benchmark_logmm2exp.run(print_data=True)
