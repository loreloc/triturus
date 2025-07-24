from collections.abc import Callable

import torch
import triton

from baselines.lm2exp import lm2exp as torch_lm2exp
from baselines.lm2exp import lm2exp_jit as torch_lm2exp_jit
from benchmarks.utils import QUANTILES, eval_gbps
from triturus.lm2exp import lm2exp
from triturus.utils import ensure_reproducibility, set_tf32_enabled


class Providers:
    TORCH = "torch"
    TORCH_JIT = "torch (jit)"
    TRITURUS = "triturus"


PROVIDERS = [
    Providers.TORCH,
    Providers.TORCH_JIT,
    Providers.TRITURUS,
]


CONFIGS = [
    *tuple(
        triton.testing.Benchmark(
            x_names=["m", "k", "n"],
            x_vals=[128, 192, 256, 512, 768, 1536, 2048],
            x_log=True,
            line_arg="provider",
            line_vals=PROVIDERS,
            line_names=PROVIDERS,
            ylabel="GB/s",
            plot_name=f"lm2exp (square matrices, allow_tf32={allow_tf32})",
            args={"batch": 1, "allow_tf32": allow_tf32},
        )
        for allow_tf32 in [False, True]
    ),
    *tuple(
        triton.testing.Benchmark(
            x_names=["m", "k"],
            x_vals=[64, 96, 128, 192, 256, 512, 768, 1024],
            x_log=True,
            line_arg="provider",
            line_vals=PROVIDERS,
            line_names=PROVIDERS,
            ylabel="GB/s",
            plot_name=f"lm2exp (rectangular matrices, n={n} batch={batch} allow_tf32={allow_tf32})",
            args={"n": n, "batch": batch, "allow_tf32": allow_tf32},
        )
        for batch in [64, 384]
        for n in [256]
        for allow_tf32 in [False, True]
    ),
    # *tuple(
    #     triton.testing.Benchmark(
    #         x_names=["n", "k"],
    #         x_vals=[64, 96, 128, 192, 256, 512, 768, 1024, 2048],
    #         x_log=True,
    #         line_arg="provider",
    #         line_vals=PROVIDERS,
    #         line_names=PROVIDERS,
    #         ylabel="GB/s",
    #         plot_name=f"logmm2exp performance (rectangular matrices m={m} batch={batch}, allow_tf32={allow_tf32})",
    #         args={"m": m, "batch": batch, "allow_tf32": allow_tf32},
    #     )
    #     for batch in [64, 384]
    #     for m in [256]
    #     for allow_tf32 in [False, True]
    # ),
]


@triton.testing.perf_report(CONFIGS)
def benchmark_lm2exp(
    batch, m, k, n, provider, *, allow_tf32: bool = False
) -> tuple[float, float, float]:
    ensure_reproducibility()
    set_tf32_enabled(allow_tf32)
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
    ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=150, rep=1000, quantiles=QUANTILES)
    size = batch * (m * k + k * n)
    return (
        eval_gbps(size, ms),
        eval_gbps(size, min_ms),
        eval_gbps(size, max_ms),
    )


if __name__ == "__main__":
    benchmark_lm2exp.run(print_data=True)
