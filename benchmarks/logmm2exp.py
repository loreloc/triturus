from collections.abc import Callable

import torch
import triton

from baselines.logmm2exp import logmm2exp as torch_logmm2exp
from baselines.logmm2exp import logmm2exp_jit as torch_logmm2exp_jit
from benchmarks.utils import QUANTILES, eval_gbps
from triturus.logmm2exp import logmm2exp
from triturus.utils import ensure_reproducibility


class Providers:
    TORCH = "torch"
    TORCH_JIT = "torch (jit)"
    TRITURUS = "triturus"


CONFIGS = [
    triton.testing.Benchmark(
        x_names=["m", "k", "n"],
        x_vals=[48, 128, 192, 512, 768, 2048, 3072, 8192],
        x_log=True,
        line_arg="provider",
        line_vals=[
            Providers.TORCH,
            Providers.TORCH_JIT,
            Providers.TRITURUS,
        ],
        line_names=[
            Providers.TORCH,
            Providers.TORCH_JIT,
            Providers.TRITURUS,
        ],
        ylabel="GiB/s",
        plot_name="logmm2exp performance (squared matrices)",
        args={},
    ),
    *tuple(
        triton.testing.Benchmark(
            x_names=["m"],
            x_vals=[128, 192, 512, 768, 2048, 3072, 8192, 16384, 32768, 98304],
            x_log=True,
            line_arg="provider",
            line_vals=[
                Providers.TORCH,
                Providers.TORCH_JIT,
                Providers.TRITURUS,
            ],
            line_names=[
                Providers.TORCH,
                Providers.TORCH_JIT,
                Providers.TRITURUS,
            ],
            ylabel="GiB/s",
            plot_name=f"logmm2exp performance (rectangular matrices k={k}, n={n})",
            args={"k": k, "n": n},
        )
        for k in [64, 128, 256, 512]
        for n in [256]
    ),
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
        case _:
            assert False, provider
    ms, min_ms, max_ms = triton.testing.do_bench(
        fn, warmup=50, rep=300, quantiles=QUANTILES
    )
    size = m * k + k * n
    return (
        eval_gbps(size, ms),
        eval_gbps(size, min_ms),
        eval_gbps(size, max_ms),
    )


benchmark_logmm2exp.run(print_data=True)
