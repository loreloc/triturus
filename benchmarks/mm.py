import torch
import triton

from benchmarks.utils import QUANTILES, Providers, eval_tflops
from triturus.mm import mm
from triturus.utils import ensure_reproducibility

CONFIGS = [
    triton.testing.Benchmark(
        x_names=["m", "k", "n"],
        x_vals=[72 * i for i in range(1, 65)],
        line_arg="provider",
        line_vals=[Providers.CUBLAS, Providers.TRITURUS],
        line_names=[Providers.CUBLAS, Providers.TRITURUS],
        ylabel="TFLOPS",
        plot_name="mm performance",
        args={},
    )
]


@triton.testing.perf_report(CONFIGS)
def benchmark_vadd(m, k, n, provider) -> tuple[float, float, float]:
    ensure_reproducibility()
    a = torch.rand(m, k)
    b = torch.rand(k, n)
    match provider:
        case Providers.CUBLAS:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.mm(a, b), quantiles=QUANTILES
            )
        case Providers.TRITURUS:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: mm(a, b), quantiles=QUANTILES
            )
        case _:
            assert False, provider

    nflops = m * n * k
    return (
        eval_tflops(nflops, ms),
        eval_tflops(nflops, min_ms),
        eval_tflops(nflops, max_ms),
    )


benchmark_vadd.run(show_plots=True, print_data=True)
