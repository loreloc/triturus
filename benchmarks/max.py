import torch
import triton

from benchmarks.utils import QUANTILES, Providers, eval_tflops
from triturus.max import matmax, vamax, vmax
from triturus.utils import ensure_reproducibility

CONFIGS_VMAX = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[48 + 2**i for i in list(range(4, 23, 2))],
        line_arg="provider",
        line_vals=[Providers.CUBLAS, Providers.TRITURUS],
        line_names=[Providers.CUBLAS, Providers.TRITURUS],
        ylabel="TFLOPS",
        plot_name="vmax performance",
        args={},
    )
]


CONFIGS_MATMAX = [
    triton.testing.Benchmark(
        x_names=["m", "n"],
        x_vals=[48 + 2**i for i in list(range(4, 15))],
        line_arg="provider",
        line_vals=[Providers.CUBLAS, Providers.TRITURUS],
        line_names=[Providers.CUBLAS, Providers.TRITURUS],
        ylabel="TFLOPS",
        plot_name=f"matmax performance (axis={axis})",
        args={"axis": axis},
    )
    for axis in [0, 1]
]


@triton.testing.perf_report(CONFIGS_VMAX)
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


@triton.testing.perf_report(CONFIGS_MATMAX)
def benchmark_matmax(m, n, provider, axis) -> tuple[float, float, float]:
    ensure_reproducibility()
    x = torch.rand(m, n)
    match provider:
        case Providers.CUBLAS:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.amax(x, dim=0), quantiles=QUANTILES
            )
        case Providers.TRITURUS:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: matmax(x, axis=axis), quantiles=QUANTILES
            )
        case _:
            assert False, provider

    nflops = m * n
    return (
        eval_tflops(nflops, ms),
        eval_tflops(nflops, min_ms),
        eval_tflops(nflops, max_ms),
    )


benchmark_vmax.run(show_plots=True, print_data=True)
benchmark_matmax.run(show_plots=True, print_data=True)
