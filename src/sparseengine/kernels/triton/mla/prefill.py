"""Bounded MLA attention partials and online output merging."""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from sparseengine.platforms import device_runtime


@lru_cache(maxsize=None)
def _launch_config(device_index, head_dim, short_query):
    from sparseengine.kernels.triton.context_flashattention_nopad import (
        _device_max_shared_memory,
        select_context_attention_launch_config,
    )

    # Reuse the existing prefill tiles and shared-memory compatibility bound.
    max_shared_memory = _device_max_shared_memory(device_index)
    config = select_context_attention_launch_config(
        head_dim,
        max_shared_memory=max_shared_memory,
        is_tesla="Tesla" in device_runtime.optional_device_name(device_index),
    )
    if short_query:
        # The HD256 32x64 three-stage partial uses 151,552 shared bytes
        # with Triton 3.6; SM120 allows only 101,376 bytes per block.
        stages = 1 if head_dim == 256 and max_shared_memory < 151552 else 3
        return 32, 64, 4, stages
    return config


@triton.jit
def _attention(
    Q,
    K,
    V,
    O,
    L,
    CQ,
    CK,
    q0: tl.constexpr,
    q1: tl.constexpr,
    k0: tl.constexpr,
    k1: tl.constexpr,
    v0: tl.constexpr,
    v1: tl.constexpr,
    TOTAL_Q,
    H: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    block, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qs, qe = tl.load(CQ + batch), tl.load(CQ + batch + 1)
    ks, ke = tl.load(CK + batch), tl.load(CK + batch + 1)
    qi = block * M + tl.arange(0, M)
    di = tl.arange(0, D)
    # Packed ragged tensors can exceed 2**31 elements even with int32 cu_seqlens.
    q = tl.load(
        Q + (qs.to(tl.int64) + qi[:, None]) * q0 + head * q1 + di[None, :], qi[:, None] < qe - qs, 0
    )
    maximum = tl.full((M,), -float("inf"), tl.float32)
    denominator = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, D), tl.float32)
    # Match the causal traversal of context_flashattention_nopad: later keys
    # cannot contribute to this query tile. Ragged-batch padding does no work.
    end = ke - ks
    if CAUSAL:
        end = tl.minimum(end, (block + 1) * M + (ke - ks) - (qe - qs))
    end = tl.where(block * M < qe - qs, tl.maximum(end, 0), 0)
    for start in range(0, end, N):
        ki = start + tl.arange(0, N)
        k = tl.load(
            K + (ks.to(tl.int64) + ki[None, :]) * k0 + head * k1 + di[:, None],
            ki[None, :] < ke - ks,
            0,
        )
        z = tl.dot(q, k) * (SCALE * 1.4426950408889634)
        valid = ki[None, :] < ke - ks
        if CAUSAL:
            valid = valid & (ki[None, :] <= qi[:, None] + (ke - ks) - (qe - qs))
        z = tl.where(valid, z, -float("inf"))
        updated = tl.maximum(maximum, tl.max(z, 1))
        safe = tl.where(updated == -float("inf"), 0.0, updated)
        p = tl.exp2(z - safe[:, None])
        alpha = tl.exp2(maximum - safe)
        v = tl.load(
            V + (ks.to(tl.int64) + ki[:, None]) * v0 + head * v1 + di[None, :],
            ki[:, None] < ke - ks,
            0,
        )
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)
        denominator = denominator * alpha + tl.sum(p, 1)
        maximum = updated
    out = acc / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(
        O + ((qs.to(tl.int64) + qi[:, None]) * H + head) * D + di[None, :],
        out,
        qi[:, None] < qe - qs,
    )
    # The merge and score APIs consume natural-log LSE, despite exp2 internally.
    lse = (maximum + tl.log2(denominator)) * 0.6931471805599453
    tl.store(L + head * TOTAL_Q + qs + qi, lse, qi < qe - qs)


def attention_partial(q, k, v, cu_q, cu_k, max_q, max_k, *, scale, causal):
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    lse = torch.empty((q.shape[1], q.shape[0]), device=q.device, dtype=torch.float32)
    # Observation-window chunks have few queries but can scan long history.
    # Resolve the narrow tile within the same device-resource bound.
    block_m, block_n, num_warps, num_stages = _launch_config(
        q.device.index, q.shape[2], max_q < 1024
    )
    _attention[(triton.cdiv(max_q, block_m), q.shape[1], cu_q.numel() - 1)](
        q,
        k,
        v,
        output,
        lse,
        cu_q,
        cu_k,
        *q.stride()[:2],
        *k.stride()[:2],
        *v.stride()[:2],
        q.shape[0],
        q.shape[1],
        q.shape[2],
        scale,
        causal,
        block_m,
        block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output, lse


@triton.jit
def _merge(
    O,
    L,
    P,
    PL,
    o0: tl.constexpr,
    o1: tl.constexpr,
    l0,
    l1,
    p0: tl.constexpr,
    p1: tl.constexpr,
    pl0,
    pl1,
    QN,
    H: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    ROWS: tl.constexpr = 8,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    qi, head = row // H, row % H
    valid = qi < QN
    a = tl.load(L + head * l0 + qi * l1, valid, -float("inf"))
    b = tl.load(PL + head * pl0 + qi * pl1, valid, -float("inf"))
    m = tl.maximum(a, b)
    safe = tl.where(m == -float("inf"), 0.0, m)
    wa, wb = tl.exp(a - safe), tl.exp(b - safe)
    total = wa + wb
    denom = tl.where(total > 0, total, 1.0)
    d = tl.arange(0, BD)
    old = tl.load(
        O + qi[:, None] * o0 + head[:, None] * o1 + d[None, :],
        valid[:, None] & (d[None, :] < D),
        0,
    ).to(tl.float32)
    partial = tl.load(
        P + qi[:, None] * p0 + head[:, None] * p1 + d[None, :],
        valid[:, None] & (d[None, :] < D),
        0,
    ).to(tl.float32)
    result = old * (wa / denom)[:, None] + partial * (wb / denom)[:, None]
    tl.store(
        O + qi[:, None] * o0 + head[:, None] * o1 + d[None, :],
        result,
        valid[:, None] & (d[None, :] < D),
    )
    tl.store(L + head * l0 + qi * l1, safe + tl.log(total), valid)


def merge_partial(output, lse, partial, partial_lse):
    _merge[(triton.cdiv(output.shape[0] * output.shape[1], 8),)](
        output,
        lse,
        partial,
        partial_lse,
        *output.stride()[:2],
        *lse.stride(),
        *partial.stride()[:2],
        *partial_lse.stride(),
        output.shape[0],
        output.shape[1],
        output.shape[2],
        triton.next_power_of_2(output.shape[2]),
    )
