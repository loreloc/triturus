from collections.abc import Callable

import torch
import triton

from benchmarks.utils import QUANTILES, eval_tflops
from triturus.mm import mm
from triturus.utils import ensure_reproducibility, set_tf32_enabled


class Providers:
    TORCH = "torch"
    TRITURUS = "triturus"


CONFIGS = [
    triton.testing.Benchmark(
        x_names=["m", "k", "n"],
        x_vals=[48, 128, 192, 512, 768, 2048, 3072, 8192, 12288, 16384],
        x_log=True,
        line_arg="provider",
        line_vals=[Providers.TORCH, Providers.TRITURUS],
        line_names=[Providers.TORCH, Providers.TRITURUS],
        ylabel="TFLOPS",
        plot_name=f"mm (allow_tf32={allow_tf32})",
        args={"allow_tf32": allow_tf32},
    )
    for allow_tf32 in [False, True]
]


@triton.testing.perf_report(CONFIGS)
def benchmark_mm(m, k, n, provider, *, allow_tf32: bool = False) -> tuple[float, float, float]:
    ensure_reproducibility()
    set_tf32_enabled(allow_tf32)
    a = torch.rand(m, k)
    b = torch.rand(k, n)
    fn: Callable[[], torch.Tensor]
    match provider:
        case Providers.TORCH:
            fn = lambda: torch.mm(a, b)
        case Providers.TRITURUS:
            fn = lambda: mm(a, b)
        case _:
            assert False, provider
    ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=125, rep=750, quantiles=QUANTILES)
    nflops = m * n * (2 * k - 1)
    return (
        eval_tflops(nflops, ms),
        eval_tflops(nflops, max_ms),
        eval_tflops(nflops, min_ms),
    )


if __name__ == "__main__":
    benchmark_mm.run(print_data=True)
