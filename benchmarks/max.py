from collections.abc import Callable

import torch
import triton

from benchmarks.utils import QUANTILES, eval_tflops
from triturus.max import matmax, vamax, vmax
from triturus.utils import ensure_reproducibility


class Providers:
    TORCH = "torch"
    TRITURUS = "triturus"


CONFIGS_VMAX = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[32, 96, 256, 768, 2048, 6144, 16384, 49152, 131072, 393216, 1048576],
        line_arg="provider",
        line_vals=[Providers.TORCH, Providers.TRITURUS],
        line_names=[Providers.TORCH, Providers.TRITURUS],
        ylabel="TFLOPS",
        plot_name=f"vmax performance (return_indices={return_indices})",
        args={"return_indices": return_indices},
    )
    for return_indices in [False, True]
]


@triton.testing.perf_report(CONFIGS_VMAX)
def benchmark_vmax(n, provider, return_indices) -> tuple[float, float, float]:
    ensure_reproducibility()
    fn: Callable[[], torch.Tensor]
    x = torch.rand(n)
    match provider:
        case Providers.TORCH:
            if return_indices:
                fn = lambda: torch.max(x, dim=0)
            else:
                fn = lambda: torch.amax(x, dim=0)
        case Providers.TRITURUS:
            if return_indices:
                fn = lambda: vmax(x)
            else:
                fn = lambda: vamax(x)
        case _:
            assert False, provider
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=QUANTILES)
    return eval_tflops(n, ms), eval_tflops(n, min_ms), eval_tflops(n, max_ms)


CONFIGS_MATMAX = [
    triton.testing.Benchmark(
        x_names=["m", "n"],
        x_vals=[32, 48, 128, 192, 512, 768, 2048, 3072, 8192],
        line_arg="provider",
        line_vals=[Providers.TORCH, Providers.TRITURUS],
        line_names=[Providers.TORCH, Providers.TRITURUS],
        ylabel="TFLOPS",
        plot_name=f"matmax performance (axis={axis})",
        args={"axis": axis},
    )
    for axis in [0, 1]
]


@triton.testing.perf_report(CONFIGS_MATMAX)
def benchmark_matmax(m, n, provider, axis) -> tuple[float, float, float]:
    ensure_reproducibility()
    x = torch.rand(m, n)
    fn: Callable[[], torch.Tensor]
    match provider:
        case Providers.TORCH:
            fn = lambda: torch.amax(x, dim=axis)
        case Providers.TRITURUS:
            fn = lambda: matmax(x, axis=axis)
        case _:
            assert False, provider
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=QUANTILES)
    nflops = m * n
    return (
        eval_tflops(nflops, ms),
        eval_tflops(nflops, min_ms),
        eval_tflops(nflops, max_ms),
    )


benchmark_vmax.run(print_data=True)
benchmark_matmax.run(print_data=True)
