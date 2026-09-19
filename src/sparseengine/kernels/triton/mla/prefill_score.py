"""Sparse observation scores over one bounded expanded or latent key block."""

import torch
import triton
import triton.language as tl


# Absorbed Q is a transposed [head, query, latent] BMM output: its head
# stride changes with the observation length, even for one fixed model.
@triton.jit(do_not_specialize=["q1"])
def _score(
    Q,
    K,
    QR,
    KR,
    LSE,
    OUT,
    STATS,
    q0: tl.constexpr,
    q1,
    k0: tl.constexpr,
    k1: tl.constexpr,
    qr0: tl.constexpr,
    qr1: tl.constexpr,
    kr0: tl.constexpr,
    QN,
    KN,
    H: tl.constexpr,
    D: tl.constexpr,
    QSTART,
    KSTART,
    CSTART,
    CEND,
    SCALE: tl.constexpr,
    MODE: tl.constexpr,
    NK,
    M: tl.constexpr = 32,
    N: tl.constexpr = 64,
):
    head, qb, kb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qi = qb * M + tl.arange(0, M)
    ki = kb * N + tl.arange(0, N)
    d = tl.arange(0, D)
    q = tl.load(Q + qi[:, None] * q0 + head * q1 + d[None, :], qi[:, None] < QN, 0)
    kh = head if MODE == 0 else 0
    k = tl.load(K + ki[None, :] * k0 + kh * k1 + d[:, None], ki[None, :] < KN, 0)
    z = tl.dot(q, k)
    if MODE != 0:
        r = tl.arange(0, 64)
        qr = tl.load(
            QR + qi[:, None] * qr0 + head * qr1 + r[None, :], qi[:, None] < QN, 0
        )
        kr = tl.load(KR + ki[None, :] * kr0 + r[:, None], ki[None, :] < KN, 0)
        z += tl.dot(qr, kr)
    key_valid = (ki < KN) & (KSTART + ki >= CSTART) & (KSTART + ki < CEND)
    valid = (
        (qi[:, None] < QN)
        & key_valid[None, :]
        & (QSTART + qi[:, None] >= KSTART + ki[None, :])
    )
    if MODE == 0:
        maximum = tl.max(tl.where(valid, z, -float("inf")), 0)
        tl.atomic_max(OUT + ki, maximum, key_valid)
    elif MODE == 1:
        z = tl.where(valid, z * SCALE, -float("inf"))
        maximum = tl.max(z, 1)
        safe = tl.where(maximum == -float("inf"), 0.0, maximum)
        total = tl.sum(tl.exp(z - safe[:, None]), 1)
        tl.store(STATS + (head * NK + kb) * QN + qi, maximum + tl.log(total), qi < QN)
    else:
        lse = tl.load(LSE + head * QN + qi, qi < QN, float("inf"))
        p = tl.where(
            valid & (lse[:, None] != -float("inf")),
            tl.exp(z * SCALE - lse[:, None]),
            0.0,
        )
        mass = tl.sum(p, 0) / QN
        tl.atomic_add(OUT + head * KN + ki, mass, key_valid)


@triton.jit
def _merge_stats(STATS, LSE, QN, NK, BK: tl.constexpr, BQ: tl.constexpr = 16):
    head, qb = tl.program_id(0), tl.program_id(1)
    qi = qb * BQ + tl.arange(0, BQ)
    ki = tl.arange(0, BK)
    values = tl.load(
        STATS + (head * NK + ki[None, :]) * QN + qi[:, None],
        (qi[:, None] < QN) & (ki[None, :] < NK),
        -float("inf"),
    )
    old = tl.load(LSE + head * QN + qi, qi < QN, -float("inf"))
    maximum = tl.maximum(tl.max(values, 1), old)
    safe = tl.where(maximum == -float("inf"), 0.0, maximum)
    total = tl.sum(tl.exp(values - safe[:, None]), 1) + tl.exp(old - safe)
    tl.store(LSE + head * QN + qi, safe + tl.log(total), qi < QN)


@triton.jit
def _reduce_heads(
    HEAD, OUT, H: tl.constexpr, KN, BH: tl.constexpr, BN: tl.constexpr = 128
):
    k = tl.program_id(0) * BN + tl.arange(0, BN)
    h = tl.arange(0, BH)
    values = tl.load(
        HEAD + h[:, None] * KN + k[None, :], (h[:, None] < H) & (k[None, :] < KN), 0.0
    )
    tl.store(OUT + k, tl.max(values, 0), k < KN)


def score_block(
    q,
    k,
    output,
    lse,
    *,
    query_start,
    key_start,
    candidate_start,
    candidate_end,
    scale,
    mode,
    rope_q=None,
    rope_k=None,
):
    """Score expanded K logits, or latent probabilities with accumulated LSE."""
    qn, heads, dim = q.shape
    kn = k.shape[0]
    if not qn or not kn:
        return
    latent = mode != "logits"
    if latent and (
        dim != 512
        or rope_q is None
        or rope_k is None
        or rope_q.shape[-1] != 64
        or rope_k.shape[-1] != 64
    ):
        raise ValueError(
            "MLA latent scoring requires 512 latent and 64 RoPE dimensions."
        )
    nk = triton.cdiv(kn, 64)
    stats = (
        torch.empty((heads, nk, qn), dtype=torch.float32, device=q.device)
        if mode == "stats"
        else output
    )
    target = (
        torch.zeros((heads, kn), dtype=torch.float32, device=q.device)
        if mode == "probability"
        else output
    )
    qr, kr = (rope_q, rope_k) if latent else (q, k)
    _score[(heads, triton.cdiv(qn, 32), nk)](
        q,
        k,
        qr,
        kr,
        lse,
        target,
        stats,
        *q.stride()[:2],
        *k.stride()[:2],
        *qr.stride()[:2],
        kr.stride(0),
        qn,
        kn,
        heads,
        dim,
        query_start,
        key_start,
        candidate_start,
        candidate_end,
        scale,
        {"logits": 0, "stats": 1, "probability": 2}[mode],
        nk,
        num_warps=4,
    )
    if mode == "stats":
        _merge_stats[(heads, triton.cdiv(qn, 16))](
            stats, lse, qn, nk, triton.next_power_of_2(nk)
        )
    elif mode == "probability":
        _reduce_heads[(triton.cdiv(kn, 128),)](
            target, output, heads, kn, triton.next_power_of_2(heads)
        )
