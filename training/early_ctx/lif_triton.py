"""Fused PLIF time loop (Triton) for training: same math as riverops PLIFNode + speedpatch (hard reset, ATan
surrogate alpha=2, zero initial state), but one kernel launch for the whole sequence instead of ~6 per time step.

  v_t = u_{t-1} * a + x_t        a = 1 - 1/tau
  s_t = H(v_t - th)              backward: ATan surrogate  sg = (alpha/2) / (1 + (pi*alpha/2 * (v_t - th))^2)
  u_t = (1 - s_t) * v_t          gradient flows through s_t into the reset, as in riverops

fast_forward(m, x) == m.head(m.backbone(x)).flatten(2) (checked by check() against the riverops path).
"""
import math

import torch
import triton
import triton.language as tl

ALPHA = 2.0


@triton.jit
def _fwd(X, S, V, a, th, T, N, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    base = b.to(tl.int64) * T * N
    u = tl.zeros([BLOCK], dtype=tl.float32)
    for t in range(T):
        p = base + t * N + offs
        x = tl.load(X + p, mask=mask, other=0.0)
        v = u * a + x
        s = (v >= th).to(tl.float32)
        tl.store(S + p, s, mask=mask)
        tl.store(V + p, v, mask=mask)
        u = (1.0 - s) * v


@triton.jit
def _bwd(GS, V, DX, GA, a, th, T, N, scale, half_alpha, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    pn = tl.program_id(1)
    offs = pn * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    base = b.to(tl.int64) * T * N
    gu = tl.zeros([BLOCK], dtype=tl.float32)
    ga = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(T):
        t = T - 1 - i
        p = base + t * N + offs
        v = tl.load(V + p, mask=mask, other=0.0)
        gs = tl.load(GS + p, mask=mask, other=0.0)
        s = (v >= th).to(tl.float32)
        z = scale * (v - th)
        sg = half_alpha / (1.0 + z * z)
        dv = gs * sg + gu * ((1.0 - s) - v * sg)
        tl.store(DX + p, dv, mask=mask)
        vp = tl.load(V + p - N, mask=mask & (t > 0), other=0.0)
        up = tl.where(vp >= th, 0.0, vp)
        ga += dv * up
        gu = dv * a
    tl.store(GA + (b * tl.num_programs(1) + pn) * BLOCK + tl.arange(0, BLOCK), tl.where(mask, ga, 0.0))


class LIFScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, th):
        B, T = x.shape[:2]
        xf = x.contiguous().view(B, T, -1).float()
        N = xf.shape[2]
        S = torch.empty_like(xf)
        V = torch.empty_like(xf)
        BLOCK = 256
        grid = (B, triton.cdiv(N, BLOCK))
        _fwd[grid](xf, S, V, float(a), float(th), T, N, BLOCK=BLOCK)
        ctx.save_for_backward(V)
        ctx.a, ctx.th, ctx.shape, ctx.BLOCK, ctx.grid = float(a), float(th), x.shape, BLOCK, grid
        return S.view(x.shape)

    @staticmethod
    def backward(ctx, gs):
        (V,) = ctx.saved_tensors
        B, T, N = V.shape
        gsf = gs.contiguous().view(B, T, N).float()
        DX = torch.empty_like(V)
        GA = torch.empty(ctx.grid[0] * ctx.grid[1] * ctx.BLOCK, device=V.device, dtype=torch.float32)
        _bwd[ctx.grid](gsf, V, DX, GA, ctx.a, ctx.th, T, N, math.pi * ALPHA / 2.0, ALPHA / 2.0, BLOCK=ctx.BLOCK)
        return DX.view(ctx.shape), GA.sum().view(()), None


def lif(L, cur):
    """cur [B, T, ...] -> spikes, with gradient to the leak parameter through a = 1 - 1/tau."""
    tau = L.lif.tau
    a = 1.0 - 1.0 / tau if torch.is_tensor(tau) else torch.tensor(1.0 - 1.0 / tau, device=cur.device)
    return LIFScan.apply(cur, a, float(L.lif.v_threshold))


def fast_forward(m, x):
    h = x
    for L in list(m.backbone) + [m.head]:
        cur = L.bn(L.conv(h)) if getattr(L, "bn", None) is not None else L.conv(h)
        h = lif(L, cur)
    return h.flatten(2)


def check(m, B=4, T=120, dev="cuda"):
    torch.manual_seed(0)
    return check_batch(m, (torch.rand(B, T, 2, 80, 80, device=dev) < 0.15).float())


def check_batch(m, x):
    """Compare output spikes and parameter gradients with the riverops path
    (train mode, same batch; model state restored afterwards)."""
    B, T = x.shape[:2]
    w = torch.rand(B, T, device=x.device)
    m.train()
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    m.zero_grad()
    y1 = m.head(m.backbone(x)).flatten(2)[..., 0]
    (w * y1).sum().backward()
    g1 = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.load_state_dict(sd)
    m.zero_grad()
    y2 = fast_forward(m, x)[..., 0]
    (w * y2).sum().backward()
    g2 = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.load_state_dict(sd)
    m.zero_grad()
    out = {"spike_mismatch": int((y1 != y2).sum()), "spikes": int(y1.sum())}
    # relative error per parameter, with a floor of 1e-3 x the largest parameter-gradient norm, so that a
    # near-zero gradient (e.g. a leak when the layer barely fires) is not judged by float summation order
    floor = 1e-3 * max(float(g.norm()) for g in g1.values())
    err = {k: float((g1[k] - g2[k]).norm()) / (float(g1[k].norm()) + floor + 1e-12) for k in g1}
    out["max_rel_grad_err"] = max(err.values())
    out["worst"] = max(err, key=err.get)
    out["worst_norms"] = [float(g1[out["worst"]].norm()), float(g2[out["worst"]].norm()), floor]
    return out
