import torch


def lm2exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert len(a.shape) == len(b.shape) == 3
    assert a.shape[0] == b.shape[0]
    assert a.shape[2] == b.shape[1]
    assert a.dtype == b.dtype == torch.float32
    assert a.device == b.device
    assert a.is_contiguous() and b.is_contiguous()
    #
    m = torch.amax(b, dim=1, keepdim=True)
    m = torch.clamp(m, -1e38, 1e38)
    r = torch.bmm(a, torch.exp(b - m))
    return m + torch.log(r)


lm2exp_jit = torch.compile(lm2exp)
