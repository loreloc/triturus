from collections.abc import Callable

import torch
import triton

from baselines.lm2exp import lm2exp as torch_lm2exp
from baselines.lm2exp import lm2exp_jit as torch_lm2exp_jit
from benchmarks.utils import QUANTILES, eval_gbps
from triturus.lm2exp import lm2exp
from triturus.utils import ensure_reproducibility


class Providers:
    TORCH = "torch"
    TORCH_JIT = "torch (jit)"
    TRITURUS = "triturus"


CONFIGS = [
    triton.testing.Benchmark(
        x_names=["m", "k", "n"],
        x_vals=[48, 128, 192, 512, 768, 1536, 2048],
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
        plot_name="logmm2exp performance (square matrices)",
        args={"batch": 1},
    ),
    *tuple(
        triton.testing.Benchmark(
            x_names=["m", "k"],
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
            plot_name=f"logmm2exp performance (rectangular matrices n={n} batch={batch})",
            args={"n": n},
        )
        for batch in [6, 24, 96, 384]
        for n in [256]
    ),
]


@triton.testing.perf_report(CONFIGS)
def benchmark_lm2exp(batch, m, k, n, provider) -> tuple[float, float, float]:
    ensure_reproducibility()
    a = torch.rand(batch, m, k)
    b = torch.randn(batch, k, n)
    fn: Callable[[], torch.Tensor]
    match provider:
        case Providers.TORCH:
            fn = lambda: torch_lm2exp(a, b)
        case Providers.TORCH_JIT:
            fn = lambda: torch_lm2exp_jit(a, b)
        case Providers.TRITURUS:
            fn = lambda: lm2exp(a, b)
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


benchmark_lm2exp.run(print_data=True)
