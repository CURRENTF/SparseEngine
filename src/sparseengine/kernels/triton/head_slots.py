"""Address/metadata adapters for head slots. Attention and scoring reuse upstream kernels."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _publish(Table, PosTable, Writes, Columns, Rows, Positions,
             N, H: tl.constexpr, R: tl.constexpr, T: tl.constexpr,
             BLOCK: tl.constexpr):
    l = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tok, h = i // H, i % H
    row = tl.load(Rows + tok, tok < N, other=-1)
    col = tl.load(Columns + l * N * H + i, i < N * H, other=0)
    slot = tl.load(Writes + l * N * H + i, i < N * H, other=-1)
    pos = tl.load(Positions + tok, tok < N, other=0)
    valid = (i < N * H) & (row >= 0) & (slot >= 0)
    dst = ((l * R + row) * H + h) * T + col
    tl.store(Table + dst, slot, valid)
    tl.store(PosTable + dst, pos, valid)


def publish_head_slots(table, positions_table, writes, columns, rows, positions):
    layers, n, heads = writes.shape
    _publish[(layers, triton.cdiv(n * heads, 256))](
        table, positions_table, writes, columns, rows, positions,
        n, heads, table.shape[1], table.shape[3], 256)


@triton.jit(do_not_specialize=["N"])
def _store(K, V, KP, VP, Slots, N, H: tl.constexpr, D: tl.constexpr,
           KT: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
           VT: tl.constexpr, VH: tl.constexpr, VD: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    t, h, d = i // (H * D), (i // D) % H, i % D
    slot = tl.load(Slots + t * H + h, t < N, other=-1)
    valid = (t < N) & (slot >= 0)
    k = tl.load(K + t * KT + h * KH + d * KD, valid, other=0)
    v = tl.load(V + t * VT + h * VH + d * VD, valid, other=0)
    tl.store(KP + slot * D + d, k, valid)
    tl.store(VP + slot * D + d, v, valid)


def store_head_slots(k, v, kp, vp, slots):
    n, h, d = k.shape
    _store[(triton.cdiv(n * h * d, 256),)](
        k, v, kp, vp, slots, n, h, d, *k.stride(), *v.stride(), 256)


@triton.jit(do_not_specialize=["N"])
def _queries(Q, Ring, Rows, Positions, Keep, N, QH: tl.constexpr,
             D: tl.constexpr, W: tl.constexpr, QT: tl.constexpr,
             QHS: tl.constexpr, QD: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    t, h, d = i // (QH * D), (i // D) % QH, i % D
    row = tl.load(Rows + t, t < N, other=-1)
    pos = tl.load(Positions + t, t < N, other=0)
    keep = tl.load(Keep + t, t < N, other=False)
    valid = (t < N) & (row >= 0) & keep
    q = tl.load(Q + t * QT + h * QHS + d * QD, valid, other=0)
    tl.store(Ring + ((row * W + pos % W) * QH + h) * D + d, q, valid)


def record_head_queries(q, ring, rows, positions, keep):
    n, h, d = q.shape
    _queries[(triton.cdiv(n * h * d, 256),)](
        q, ring, rows, positions, keep, n, h, d, ring.shape[1], *q.stride(), 256)


@triton.jit
def _reorder_q(Source, Dest, Cu, H: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
               ST: tl.constexpr, SH: tl.constexpr, SD: tl.constexpr,
               INVERSE: tl.constexpr, BLOCK: tl.constexpr):
    b, h = tl.program_id(0), tl.program_id(1)
    begin, end = tl.load(Cu + b), tl.load(Cu + b + 1)
    i = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    t, g, d = i // (G * D), (i // D) % G, i % D
    normal_t, normal_h = begin + t, h * G + g
    virtual_t = begin * H + h * (end - begin) + t
    if INVERSE:
        src = virtual_t * ST + g * SH + d * SD
        dst = (normal_t * H * G + normal_h) * D + d
    else:
        src = normal_t * ST + normal_h * SH + d * SD
        dst = (virtual_t * G + g) * D + d
    value = tl.load(Source + src, t < end - begin, other=0)
    tl.store(Dest + dst, value, t < end - begin)


def reorder_head_queries(q, cu, heads, max_query_len, *, inverse=False):
    groups = q.shape[1] if inverse else q.shape[1] // heads
    n = q.shape[0] // heads if inverse else q.shape[0] * heads
    out_heads = heads * groups if inverse else groups
    out = torch.empty((n, out_heads, q.shape[2]), dtype=q.dtype, device=q.device)
    _reorder_q[(cu.numel() - 1, heads, triton.cdiv(max_query_len * groups * q.shape[2], 256))](
        q, out, cu, heads, groups, q.shape[2], *q.stride(), inverse, 256)
    return out
