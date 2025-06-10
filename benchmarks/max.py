import torch
import triton

from benchmarks.utils import QUANTILES, Providers, eval_tflops
from triturus.max import vmax
from triturus.utils import ensure_reproducibility

CONFIGS = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[400 * i for i in range(2, 17)],
        line_arg="provider",
        line_vals=[Providers.CUBLAS, Providers.TRITURUS],
        line_names=[Providers.CUBLAS, Providers.TRITURUS],
        ylabel="TFLOPS",
        plot_name="vadd performance",
        args={},
    )
]


@triton.testing.perf_report(CONFIGS)
def benchmark_vmax(n, provider) -> tuple[float, float, float]:
    ensure_reproducibility()
    x = torch.rand(n)
    match provider:
        case Providers.CUBLAS:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.max(x, dim=0), quantiles=QUANTILES
            )
        case Providers.TRITURUS:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: vmax(x), quantiles=QUANTILES
            )
        case _:
            assert False, provider

    return eval_tflops(n, ms), eval_tflops(n, min_ms), eval_tflops(n, max_ms)


benchmark_vmax.run(show_plots=True, print_data=True)
