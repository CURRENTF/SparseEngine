"""Paged varlen MLA prefill without materializing expanded per-head KV."""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout, DotOperandLayout, NVMMADistributedLayout
from triton.experimental.gluon.language.nvidia.ampere import mma_v2


@g.jit(do_not_specialize=["T", "q0", "q1", "r0", "r1", "p0"])
def _attention(
    Q, QR, C, KR, PAGES, ROWS, LENS, CQ, O, L,
    q0, q1, r0, r1,
    c0: gl.constexpr, k0: gl.constexpr, p0,
    T, H: gl.constexpr, D: gl.constexpr, R: gl.constexpr,
    S: gl.constexpr, SCALE: gl.constexpr, M: gl.constexpr, N: gl.constexpr,
):
    tile, split, batch = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    begin, end = gl.load(CQ + batch), gl.load(CQ + batch + 1)
    length, row = gl.load(LENS + batch), gl.load(ROWS + batch)
    mem: gl.constexpr = BlockedLayout([1, 8], [4, 8], [8, 1], [1, 0])
    mma: gl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 8], instr_shape=[16, 8])
    x = tile * M + gl.arange(0, M, layout=SliceLayout(1, mem))
    token, head = begin.to(gl.int64) + x // H, x % H
    valid_q = x // H < end - begin
    d = gl.arange(0, D, layout=SliceLayout(0, mem))
    r = gl.arange(0, R, layout=SliceLayout(0, mem))
    q = gl.load(Q + token[:, None] * q0 + head[:, None] * q1 + d[None, :], valid_q[:, None], 0)
    qr = gl.load(QR + token[:, None] * r0 + head[:, None] * r1 + r[None, :], valid_q[:, None], 0)
    q = gl.convert_layout(q, DotOperandLayout(0, mma, 2))
    qr = gl.convert_layout(qr, DotOperandLayout(0, mma, 2))
    qi = tile * M + gl.arange(0, M, layout=SliceLayout(1, mma))
    qi = qi // H
    maximum = gl.full((M,), -float("inf"), gl.float32, SliceLayout(1, mma))
    denominator = gl.zeros((M,), gl.float32, SliceLayout(1, mma))
    acc = gl.zeros((M, D), gl.float32, mma)
    span = gl.cdiv(length, S * N) * N
    stop = gl.minimum((split + 1) * span, length)
    stop = gl.minimum(stop, length - (end - begin) + (tile * M + M - 1) // H + 1)
    stop = gl.where(tile * M < (end - begin) * H, stop, 0)
    for start in range(split * span, stop, N):
        ki = start + gl.arange(0, N, layout=SliceLayout(1, mem))
        slots = gl.load(PAGES + row.to(gl.int64) * p0 + ki, ki < length, 0).to(gl.int64)
        # MMA v2 distributes the 512-wide value accumulator across all warps;
        # the short query tile stays in registers throughout the KV scan.
        z = gl.zeros((M, N), gl.float32, mma)
        c = gl.load(C + slots[:, None] * c0 + d[None, :], ki[:, None] < length, 0)
        z = mma_v2(q, gl.convert_layout(c.T, DotOperandLayout(1, mma, 2)), z)
        kr = gl.load(KR + slots[:, None] * k0 + r[None, :], ki[:, None] < length, 0)
        z = mma_v2(qr, gl.convert_layout(kr.T, DotOperandLayout(1, mma, 2)), z)
        z *= SCALE * 1.4426950408889634
        ka = start + gl.arange(0, N, layout=SliceLayout(0, mma))
        visible = (ka[None, :] < length) & (ka[None, :] <= length - (end - begin) + qi[:, None])
        z = gl.where(visible, z, -float("inf"))
        updated = gl.maximum(maximum, gl.max(z, 1))
        safe = gl.where(updated == -float("inf"), 0., updated)
        p = gl.exp2(z - safe[:, None])
        alpha = gl.exp2(maximum - safe)
        acc = acc * alpha[:, None]
        v = gl.load(C + slots[:, None] * c0 + d[None, :], ki[:, None] < length, 0)
        acc = mma_v2(gl.convert_layout(p.to(v.dtype), DotOperandLayout(0, mma, 2)),
                     gl.convert_layout(v, DotOperandLayout(1, mma, 2)), acc)
        denominator = denominator * alpha + gl.sum(p, 1)
        maximum = updated
    value = acc / gl.where(denominator > 0, denominator, 1.)[:, None]
    lse = (maximum + gl.log2(denominator)) * 0.6931471805599453
    x = tile * M + gl.arange(0, M, layout=SliceLayout(1, mma))
    token, head = begin.to(gl.int64) + x // H, x % H
    valid_q = x // H < end - begin
    d = gl.arange(0, D, layout=SliceLayout(0, mma))
    if S == 1:
        offsets = (token * H + head)[:, None] * D + d[None, :]
    else:
        offsets = ((split * H + head).to(gl.int64) * T + token)[:, None] * D + d[None, :]
    gl.store(O + offsets, value, valid_q[:, None])
    gl.store(L + (split * H + head).to(gl.int64) * T + token, lse, valid_q)


@triton.jit(do_not_specialize=["T"])
def _merge(P, PL, O, L, T, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr):
    token, head = tl.program_id(0), tl.program_id(1)
    s, d = tl.arange(0, S), tl.arange(0, D)
    offsets = (s.to(tl.int64) * H + head) * T + token
    lse = tl.load(PL + offsets)
    maximum = tl.max(lse, 0)
    weights = tl.exp(lse - maximum)
    denominator = tl.sum(weights, 0)
    values = tl.load(P + offsets[:, None] * D + d[None, :])
    value = tl.sum(values * weights[:, None], 0) / denominator
    tl.store(O + (token.to(tl.int64) * H + head) * D + d, value)
    tl.store(L + head * T + token, maximum + tl.log(denominator))


def attention_latent(q, q_rope, latent, rope, slots, rows, lengths, cu_q,
                     *, max_q, scale, splits):
    """Metadata values are validated by ChunkedMlaPrefill.prepare, not read back."""
    if (q.ndim != 3 or q_rope.shape != (*q.shape[:2], 64)
            or q.shape[-1] != 512 or latent.ndim != 3 or latent.shape[1:] != (1, 512)
            or rope.shape != (latent.shape[0], 1, 64)):
        raise ValueError("Latent prefill requires Q[...,512/64] and cache [slots,1,512/64]")
    if q.dtype not in (torch.float16, torch.bfloat16) or any(
        t.dtype != q.dtype or t.device != q.device or t.stride(-1) != 1
        for t in (q, q_rope, latent, rope)
    ):
        raise ValueError("Latent prefill requires matching FP16/BF16 contiguous feature dimensions")
    if (q.device.type != "cuda" or slots.ndim != 2 or slots.stride(1) != 1
            or rows.ndim != 1 or lengths.shape != rows.shape
            or cu_q.shape != (rows.numel() + 1,) or max_q <= 0
            or splits < 1 or splits > 32 or splits & (splits - 1)):
        raise ValueError("Invalid latent prefill metadata or split count")
    if any(t.device != q.device or t.dtype not in (torch.int32, torch.int64)
           for t in (slots, rows, lengths, cu_q)) or any(
        t.stride(0) != 1 for t in (rows, lengths, cu_q)
    ):
        raise ValueError("Latent prefill metadata must use contiguous integer vectors on the query device")
    tokens, heads, dim = q.shape
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    lse = torch.empty((heads, tokens), dtype=torch.float32, device=q.device)
    if splits == 1:
        partial, partial_lse = output, lse
    else:
        partial = torch.empty((splits, heads, tokens, dim), dtype=torch.float32, device=q.device)
        partial_lse = torch.empty((splits, heads, tokens), dtype=torch.float32, device=q.device)
    _attention[(triton.cdiv(max_q * heads, 16), splits, rows.numel())](
        q, q_rope, latent, rope, slots, rows, lengths, cu_q, partial, partial_lse,
        *q.stride()[:2], *q_rope.stride()[:2], latent.stride(0), rope.stride(0), slots.stride(0),
        tokens, heads, dim, 64, splits, scale, 16, 64, num_warps=8, num_stages=1,
    )
    if splits > 1:
        _merge[(tokens, heads)](partial, partial_lse, output, lse, tokens, heads, splits, dim)
    return output, lse
