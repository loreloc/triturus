import torch


def logmm2exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m = torch.amax(b, dim=0, keepdim=True)
    m = torch.clamp(m, -1e38, 1e38)
    r = torch.mm(a, torch.exp(b - m))
    return m + torch.log(r)


logmm2exp_jit = torch.compile(logmm2exp)
