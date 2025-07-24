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
        plot_name=f"vmax (return_idx={return_idx})",
        args={"return_idx": return_idx},
    )
    for return_idx in [False, True]
]


@triton.testing.perf_report(CONFIGS_VMAX)
def benchmark_vmax(n, provider, return_idx) -> tuple[float, float, float]:
    ensure_reproducibility()
    fn: Callable[[], torch.Tensor]
    x = torch.rand(n)
    match provider:
        case Providers.TORCH:
            if return_idx:
                fn = lambda: torch.max(x, dim=0)
            else:
                fn = lambda: torch.amax(x, dim=0)
        case Providers.TRITURUS:
            if return_idx:
                fn = lambda: vmax(x)
            else:
                fn = lambda: vamax(x)
        case _:
            assert False, provider
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=QUANTILES)
    nflops = n - 1
    return eval_tflops(nflops, ms), eval_tflops(nflops, min_ms), eval_tflops(nflops, max_ms)


CONFIGS_MATMAX = [
    triton.testing.Benchmark(
        x_names=["m", "n"],
        x_vals=[32, 48, 128, 192, 512, 768, 2048, 3072, 8192],
        line_arg="provider",
        line_vals=[Providers.TORCH, Providers.TRITURUS],
        line_names=[Providers.TORCH, Providers.TRITURUS],
        ylabel="TFLOPS",
        plot_name=f"matmax (axis={axis})",
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
    nflops = n * (m - 1) if axis == 0 else m * (n - 1)
    return (
        eval_tflops(nflops, ms),
        eval_tflops(nflops, min_ms),
        eval_tflops(nflops, max_ms),
    )


if __name__ == "__main__":
    benchmark_vmax.run(print_data=True)
    benchmark_matmax.run(print_data=True)
