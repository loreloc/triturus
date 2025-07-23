from collections.abc import Callable

import torch
import triton

from benchmarks.utils import QUANTILES, eval_tflops
from triturus.add import vadd
from triturus.utils import ensure_reproducibility


class Providers:
    TORCH = "torch"
    TRITURUS = "triturus"


CONFIGS = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[32, 48, 128, 192, 512, 768, 2048, 3072, 8192, 12288, 32768, 49152],
        line_arg="provider",
        line_vals=[Providers.TORCH, Providers.TRITURUS],
        line_names=[Providers.TORCH, Providers.TRITURUS],
        ylabel="TFLOPS",
        plot_name="vadd",
        args={},
    )
]


@triton.testing.perf_report(CONFIGS)
def benchmark_vadd(n, provider) -> tuple[float, float, float]:
    ensure_reproducibility()
    x = torch.rand(n)
    y = torch.rand(n)
    fn: Callable[[], torch.Tensor]
    match provider:
        case Providers.TORCH:
            fn = lambda: x + y
        case Providers.TRITURUS:
            fn = lambda: vadd(x, y)
        case _:
            assert False, provider
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=QUANTILES)
    return eval_tflops(n, ms), eval_tflops(n, min_ms), eval_tflops(n, max_ms)


if __name__ == "__main__":
    benchmark_vadd.run(print_data=True)
