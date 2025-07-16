import torch
import triton

from baselines.logmm2exp import logmm2exp as torch_logmm2exp
from baselines.logmm2exp import logmm2exp_jit as torch_logmm2exp_jit
from benchmarks.utils import QUANTILES, Providers, eval_gbps
from triturus.logmm2exp import logmm2exp
from triturus.utils import ensure_reproducibility

CONFIGS = [
    triton.testing.Benchmark(
        x_names=["m", "k", "n"],
        x_vals=[48 + 2**i for i in list(range(4, 15))],
        x_log=True,
        line_arg="provider",
        line_vals=[Providers.CUBLAS, Providers.TRITURUS],
        line_names=[Providers.CUBLAS, Providers.TRITURUS],
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
    match provider:
        case Providers.CUBLAS:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch_logmm2exp_jit(a, b), quantiles=QUANTILES
            )
        case Providers.TRITURUS:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: logmm2exp(a, b), quantiles=QUANTILES
            )
        case _:
            assert False, provider

    size = m * k + k * n
    return (
        eval_gbps(size, ms),
        eval_gbps(size, min_ms),
        eval_gbps(size, max_ms),
    )


benchmark_logmm2exp.run(show_plots=True, print_data=True)
