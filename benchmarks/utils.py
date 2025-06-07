from enum import Enum


class Providers(Enum):
    CUBLAS = "cuBLAS"
    TRITURUS = "triturus"


def eval_tflops(nflops: int, ms: float) -> float:
    return (nflops * 1e-12) / (ms * 1e-3)


def eval_gbps(size: int, ms: float) -> float:
    return (size * 1e-9) / (ms * 1e-3)
