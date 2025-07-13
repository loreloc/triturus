import torch


def logmm2exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m = torch.amax(b, dim=0, keepdim=True)
    r = torch.mm(a, torch.exp(b - m))
    return m + torch.log(r)


logmm2exp_jit = torch.compile(logmm2exp)
