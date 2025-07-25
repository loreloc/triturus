import torch


def lt2exp(w: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert len(w.shape) == 4
    assert len(a.shape) == len(b.shape) == 3
    assert w.shape[0] == a.shape[0] == b.shape[0]
    assert w.shape[2] * w.shape[3] == a.shape[1] * b.shape[1]
    assert a.shape[2] == b.shape[2]
    assert w.dtype == a.dtype == b.dtype == torch.float32
    assert w.device == a.device == b.device
    #
    ma = torch.amax(a, dim=1, keepdim=True)
    ma = torch.clamp(ma, -1e38, 1e38)
    exp_a = torch.exp(a - ma)
    mb = torch.amax(b, dim=1, keepdim=True)
    mb = torch.clamp(mb, -1e38, 1e38)
    exp_b = torch.exp(b - mb)
    s = exp_a.unsqueeze(dim=2) * exp_b.unsqueeze(dim=1)
    s = s.view(s.shape[0], -1, s.shape[3])
    r = torch.bmm(w.view(w.shape[0], w.shape[1], -1), s)
    return ma + mb + torch.log(r)


def lt2exp_einsum(w: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert len(w.shape) == 4
    assert len(a.shape) == len(b.shape) == 3
    assert w.shape[0] == a.shape[0] == b.shape[0]
    assert w.shape[2] * w.shape[3] == a.shape[1] * b.shape[1]
    assert a.shape[2] == b.shape[2]
    assert w.dtype == a.dtype == b.dtype == torch.float32
    assert w.device == a.device == b.device
    #
    ma = torch.amax(a, dim=1, keepdim=True)
    ma = torch.clamp(ma, -1e38, 1e38)
    exp_a = torch.exp(a - ma)
    mb = torch.amax(b, dim=1, keepdim=True)
    mb = torch.clamp(mb, -1e38, 1e38)
    exp_b = torch.exp(b - mb)
    r = torch.einsum("brij,bis,bjs->brs", w, exp_a, exp_b)
    return ma + mb + torch.log(r)


lt2exp_jit = torch.compile(lt2exp)

lt2exp_einsum_jit = torch.compile(lt2exp_einsum)
