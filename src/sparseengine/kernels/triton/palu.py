"""Grouped low-rank KV storage and fused Palu attention.

Decode reconstructs K only inside a tile; V remains in latent coordinates.
All lengths and positions are device inputs, including during graph replay.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _store(XK, XV, POS, SLOTS, CK, CV, CP, SK: tl.constexpr, SV: tl.constexpr,
           WK: tl.constexpr, WV: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    slot = tl.load(SLOTS + t)
    if slot >= 0:
        x = tl.arange(0, BLOCK)
        k = tl.load(XK + t * SK + x, x < WK, 0)
        v = tl.load(XV + t * SV + x, x < WV, 0)
        tl.store(CK + slot * WK + x, k, x < WK)
        tl.store(CV + slot * WV + x, v, x < WV)
        tl.store(CP + slot, tl.load(POS + t))


def store_latent(write, slots, cache):
    wk, wv = write.key_latent.shape[1] * write.key_latent.shape[2], write.value_latent.shape[1] * write.value_latent.shape[2]
    _store[(slots.numel(),)](write.key_latent, write.value_latent, write.positions, slots,
                           cache.key_latent, cache.value_latent, cache.positions,
                           write.key_latent.stride(0), write.value_latent.stride(0),
                           wk, wv, triton.next_power_of_2(max(wk, wv)))


@triton.jit
def _rope_q(Q, POS, ROPE, OUT, SQ0: tl.constexpr, SQ1: tl.constexpr,
            H: tl.constexpr, D: tl.constexpr):
    t, h = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, D // 2)
    pos = tl.load(POS + t)
    c = tl.load(ROPE + pos * D + d)
    s = tl.load(ROPE + pos * D + D // 2 + d)
    a = tl.load(Q + t * SQ0 + h * SQ1 + d).to(tl.float32)
    b = tl.load(Q + t * SQ0 + h * SQ1 + D // 2 + d).to(tl.float32)
    tl.store(OUT + (t * H + h) * D + d, a * c - b * s)
    tl.store(OUT + (t * H + h) * D + D // 2 + d, b * c + a * s)


def rotate_query(q, positions, rope):
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _rope_q[q.shape[:2]](q, positions, rope, out, q.stride(0), q.stride(1), q.shape[1], q.shape[2])
    return out


@triton.jit
def _keys(CK, CP, BK, NORM, ROPE, slots, valid, kv_head,
          H: tl.constexpr, G: tl.constexpr, RK: tl.constexpr,
          KR: tl.constexpr, D: tl.constexpr, N: tl.constexpr,
          NORMALIZE: tl.constexpr, EPS: tl.constexpr):
    r = tl.arange(0, KR)
    d = tl.arange(0, D)
    k = tl.full((N, D), 0., tl.float32)
    # Bound shared memory independently of the latent rank and head width.
    for offset in range(tl.cdiv(RK, KR)):
        indices = offset * KR + r
        z = tl.load(CK + slots[:, None] * (H // G * RK) + (kv_head // G) * RK + indices[None, :],
                    valid[:, None] & (indices[None, :] < RK), 0)
        w = tl.load(BK + (kv_head * RK + indices[:, None]) * D + d[None, :], indices[:, None] < RK, 0)
        k = tl.dot(z, w, k)
    k = k.to(CK.dtype.element_ty).to(tl.float32)
    if NORMALIZE:
        norm = tl.load(NORM + d)
        k = (k * tl.rsqrt(tl.sum(k * k, 1)[:, None] / D + EPS) * norm[None, :]).to(CK.dtype.element_ty).to(tl.float32)
    positions = tl.load(CP + slots, valid, 0)
    c = tl.load(ROPE + positions[:, None] * D + (d[None, :] % (D // 2)))
    s = tl.load(ROPE + positions[:, None] * D + D // 2 + (d[None, :] % (D // 2)))
    partner = tl.gather(k, tl.broadcast_to(((d + D // 2) % D)[None, :], (N, D)), axis=1)
    return (k * c + partner * s * tl.where(d[None, :] < D // 2, -1., 1.)).to(CK.dtype.element_ty)


@triton.jit(do_not_specialize=["ROW", "LENGTH"])
def _materialize(CK, CV, CP, BK, NORM, ROPE, TABLE, KOUT, VOUT,
                 ROW, LENGTH, TS: tl.constexpr,
                 H: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                 RK: tl.constexpr, RV: tl.constexpr, KR: tl.constexpr,
                 VR: tl.constexpr, NORMALIZE: tl.constexpr, EPS: tl.constexpr,
                 N: tl.constexpr):
    h = tl.program_id(1)
    t = tl.program_id(0) * N + tl.arange(0, N)
    valid = t < LENGTH
    slots = tl.load(TABLE + ROW * TS + t, valid, 0)
    k = _keys(CK, CP, BK, NORM, ROPE, slots, valid, h, H, G, RK, KR, D, N, NORMALIZE, EPS)
    d = tl.arange(0, D)
    tl.store(KOUT + (t[:, None] * H + h) * D + d[None, :], k, valid[:, None])
    r = tl.arange(0, VR)
    v = tl.load(CV + slots[:, None] * (H // G * RV) + (h // G) * RV + r[None, :], valid[:, None] & (r[None, :] < RV), 0)
    tl.store(VOUT + (t[:, None] * H + h) * VR + r[None, :], v, valid[:, None])


def materialize_prefill(payload, key_up, norm, rope, table, row, length, group_size, eps):
    h, rk, d = key_up.shape
    rv = payload.value_latent.shape[-1]
    vr = max(64, triton.next_power_of_2(rv))
    k = torch.empty(length, h, d, dtype=payload.key_latent.dtype, device=table.device)
    v = torch.empty(length, h, vr, dtype=k.dtype, device=k.device)
    _materialize[(triton.cdiv(length, 32), h)](
        payload.key_latent, payload.value_latent, payload.positions, key_up,
        norm if norm is not None else key_up, rope, table, k, v,
        row, length, table.stride(0), h, group_size, d, rk, rv,
        min(64 if d == 256 else 128, triton.next_power_of_2(rk)), vr, norm is not None, eps, 32,
        num_warps=4,
    )
    return k, v


@triton.jit
def _decode(Q, CK, CV, CP, BK, NORM, ROPE, TABLE, ROWS, LENS, MO, ML,
            SQ: tl.constexpr, TS: tl.constexpr, H: tl.constexpr, QH: tl.constexpr,
            G: tl.constexpr, D: tl.constexpr, RK: tl.constexpr, RV: tl.constexpr,
            KR: tl.constexpr, VR: tl.constexpr, QPAD: tl.constexpr,
            SPLITS: tl.constexpr, NORMALIZE: tl.constexpr, EPS: tl.constexpr,
            SCALE: tl.constexpr, N: tl.constexpr):
    b, h, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(LENS + b)
    row = tl.load(ROWS + b)
    qh = h * (QH // H) + tl.arange(0, QPAD)
    qm = tl.arange(0, QPAD) < QH // H
    d = tl.arange(0, D)
    r = tl.arange(0, VR)
    span = tl.cdiv(length, SPLITS * N) * N
    start, end = split * span, tl.minimum((split + 1) * span, length)
    acc = tl.full((QPAD, VR), 0., tl.float32)
    m = tl.full((QPAD,), -float("inf"), tl.float32)
    denom = tl.full((QPAD,), 0., tl.float32)
    if start < end:
        q = tl.load(Q + b * SQ + qh[:, None] * D + d[None, :], qm[:, None], 0)
        for off in range(start, end, N):
            t = off + tl.arange(0, N)
            valid = t < end
            slots = tl.load(TABLE + row * TS + t, valid, 0)
            k = _keys(CK, CP, BK, NORM, ROPE, slots, valid, h, H, G, RK, KR, D, N, NORMALIZE, EPS)
            scores = tl.dot(q, tl.trans(k)) * (SCALE * 1.4426950408889634)
            scores = tl.where(valid[None, :], scores, -float("inf"))
            new_m = tl.maximum(m, tl.max(scores, 1))
            alpha = tl.exp2(m - new_m)
            p = tl.exp2(scores - new_m[:, None])
            v = tl.load(CV + slots[:, None] * (H // G * RV) + (h // G) * RV + r[None, :], valid[:, None] & (r[None, :] < RV), 0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            denom = denom * alpha + tl.sum(p, 1)
            m = new_m
    out = acc / tl.where(denom[:, None] > 0, denom[:, None], 1.)
    lse = tl.where(denom > 0, m + tl.log2(denom), -float("inf"))
    tl.store(MO + ((b * QH + qh[:, None]) * SPLITS + split) * RV + r[None, :], out, qm[:, None] & (r[None, :] < RV))
    tl.store(ML + (b * QH + qh) * SPLITS + split, lse, qm)


@triton.jit
def _merge(MO, ML, OUT, QH: tl.constexpr, RV: tl.constexpr,
           VR: tl.constexpr, SPLITS: tl.constexpr, SP: tl.constexpr):
    bh = tl.program_id(0)
    s, r = tl.arange(0, SP), tl.arange(0, VR)
    l = tl.load(ML + bh * SPLITS + s, s < SPLITS, -float("inf"))
    maximum = tl.max(l, 0)
    maximum = tl.where(maximum == -float("inf"), 0., maximum)
    w = tl.exp2(l - maximum)
    denom = tl.sum(w, 0)
    o = tl.load(MO + (bh * SPLITS + s[:, None]) * RV + r[None, :], (s[:, None] < SPLITS) & (r[None, :] < RV), 0)
    value = tl.sum(w[:, None] * o, 0) / tl.where(denom > 0, denom, 1.)
    tl.store(OUT + bh * RV + r, value, r < RV)


def decode(q, view, key_up, norm, rope, group_size, eps, mid_o, mid_lse, output, scale):
    p, meta = view.payload, view.meta
    h, rk, d = key_up.shape
    b, qh = q.shape[:2]
    rv = p.value_latent.shape[-1]
    splits = mid_lse.shape[-1]
    _decode[(b, h, splits)](
        q, p.key_latent, p.value_latent, p.positions, key_up,
        norm if norm is not None else key_up, rope,
        meta.active_slots, meta.req_indices, meta.context_lens, mid_o, mid_lse,
        q.stride(0), meta.active_slots.stride(0), h, qh, group_size, d, rk, rv,
        min(64 if d == 256 else 128, triton.next_power_of_2(rk)), triton.next_power_of_2(rv),
        max(16, triton.next_power_of_2(qh // h)), splits, norm is not None, eps, scale, 32,
        num_warps=4, num_stages=2,
    )
    _merge[(b * qh,)](mid_o, mid_lse, output, qh, rv, triton.next_power_of_2(rv),
                       splits, triton.next_power_of_2(splits))
    return output[:b]
