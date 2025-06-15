import torch
import triton

from benchmarks.utils import QUANTILES, Providers, eval_tflops
from triturus.mm import mm
from triturus.utils import ensure_reproducibility

CONFIGS = [
    triton.testing.Benchmark(
        x_names=["m", "k", "n"],
        x_vals=[256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240],
        x_log=True,
        line_arg="provider",
        line_vals=[Providers.CUBLAS, Providers.TRITURUS],
        line_names=[Providers.CUBLAS, Providers.TRITURUS],
        ylabel="TFLOPS",
        plot_name="mm performance",
        args={},
    )
]


@triton.testing.perf_report(CONFIGS)
def benchmark_mm(m, k, n, provider) -> tuple[float, float, float]:
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

    nflops = m * n * (2 * k - 1)
    return (
        eval_tflops(nflops, ms),
        eval_tflops(nflops, min_ms),
        eval_tflops(nflops, max_ms),
    )


benchmark_mm.run(show_plots=True, print_data=True)
