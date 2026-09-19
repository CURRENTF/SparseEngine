"""Page-tiled QuEST bounds with vector or matrix-product reductions."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sparseengine.kernels.triton.quest_decode_view import (
    _validate_quest_page_score_inputs,
)


@triton.jit
def _near_rounding_midpoint(value, error, DTYPE: tl.constexpr):
    magnitude = tl.abs(value)
    exponent = ((magnitude.to(tl.int32, bitcast=True) >> 23) & 255) - 127
    MANTISSA: tl.constexpr = 7 if DTYPE == tl.bfloat16 else 10
    MIN_EXPONENT: tl.constexpr = -126 if DTYPE == tl.bfloat16 else -14
    spacing = tl.exp2((tl.maximum(exponent, MIN_EXPONENT) - MANTISSA).to(tl.float32))
    lower = tl.floor(magnitude / spacing) * spacing
    midpoint = lower + spacing * 0.5
    rounded = value.to(DTYPE).to(tl.float32)
    close = (tl.abs(magnitude - midpoint) <= error) | (tl.abs(rounded) == float("inf"))
    return close, spacing


@triton.jit(do_not_specialize=["NUM_PAGES"])
def _gqa_page_bounds(
    query, page_max, page_min, page_table, partial, repair,
    NUM_PAGES, QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr, HEAD_TILES: tl.constexpr,
    NUM_PARTS: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row, tile, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kv_head = part // HEAD_TILES
    pages = tile * BLOCK_P + tl.arange(0, BLOCK_P)
    physical = tl.maximum(tl.load(page_table + row * NUM_PAGES + pages, pages < NUM_PAGES, 0), 0)
    heads = (part % HEAD_TILES) * 16 + tl.arange(0, 16)
    positive = tl.full((16, BLOCK_P), 0, tl.float32)
    negative = tl.full((16, BLOCK_P), 0, tl.float32)
    positive_abs = tl.full((16, BLOCK_P), 0, tl.float32)
    negative_abs = tl.full((16, BLOCK_P), 0, tl.float32)
    for block in range(tl.cdiv(HEAD_DIM, BLOCK_D)):
        dims = block * BLOCK_D + tl.arange(0, BLOCK_D)
        q = tl.load(
            query + row * QUERY_HEADS * HEAD_DIM
            + (kv_head * GROUP_SIZE + heads[:, None]) * HEAD_DIM + dims[None, :],
            (heads[:, None] < GROUP_SIZE) & (dims[None, :] < HEAD_DIM), 0,
        )
        offsets = physical[None, :] * KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM + dims[:, None]
        mask = (pages[None, :] < NUM_PAGES) & (dims[:, None] < HEAD_DIM)
        high = tl.load(page_max + offsets, mask, 0)
        low = tl.load(page_min + offsets, mask, 0)
        positive = tl.dot(tl.where(q >= 0, q, 0).to(q.dtype), high, positive)
        negative = tl.dot(tl.where(q < 0, q, 0).to(q.dtype), low, negative)
        positive_abs = tl.dot(tl.where(q >= 0, q, 0).to(q.dtype), tl.abs(high), positive_abs)
        negative_abs = tl.dot(tl.where(q < 0, -q, 0).to(q.dtype), tl.abs(low), negative_abs)
    # Round once after the complete dimension reduction, separately for each
    # sign. Rounding individual K tiles would change the scoring formula.
    bounds = (positive.to(query.dtype.element_ty) + negative.to(query.dtype.element_ty)).to(query.dtype.element_ty)
    best = tl.max(tl.where(heads[:, None] < GROUP_SIZE, bounds.to(tl.float32), -float("inf")), 0)
    # A changed FP32 reduction tree can cross a low-precision rounding
    # midpoint. Only heads close enough to win can change this page's max.
    # Bound both accumulation errors using gamma_n and sum(abs(products)).
    gamma: tl.constexpr = (2.0 * HEAD_DIM * 2.0**-24) / (1.0 - 2.0 * HEAD_DIM * 2.0**-24)
    pos_error = gamma * positive_abs
    neg_error = gamma * negative_abs
    pos_close, pos_spacing = _near_rounding_midpoint(positive, pos_error, query.dtype.element_ty)
    neg_close, neg_spacing = _near_rounding_midpoint(negative, neg_error, query.dtype.element_ty)
    margin = 2 * (pos_spacing + neg_spacing + pos_error + neg_error)
    contender = (heads[:, None] < GROUP_SIZE) & (bounds.to(tl.float32) >= best[None, :] - margin)
    flagged = (pos_close | neg_close) & contender
    needs_repair = tl.sum(tl.where(flagged, 1 << tl.arange(0, 16)[:, None], 0), 0)
    safe_best = tl.max(tl.where((heads[:, None] < GROUP_SIZE) & ~flagged, bounds.to(tl.float32), -float("inf")), 0)
    tl.store(partial + (row * NUM_PARTS + part) * NUM_PAGES + pages, safe_best, pages < NUM_PAGES)
    tl.store(repair + (row * NUM_PARTS + part) * NUM_PAGES + pages, needs_repair, pages < NUM_PAGES)


@triton.jit(do_not_specialize=["NUM_PAGES"])
def _vector_page_bounds(
    query, page_max, page_min, page_table, partial,
    NUM_PAGES, QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row, tile, kv_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    pages = tile * BLOCK_P + tl.arange(0, BLOCK_P)
    physical = tl.maximum(tl.load(page_table + row * NUM_PAGES + pages, pages < NUM_PAGES, 0), 0)
    dims = tl.arange(0, BLOCK_D)
    offsets = physical[:, None] * KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM + dims[None, :]
    mask = (pages[:, None] < NUM_PAGES) & (dims[None, :] < HEAD_DIM)
    high = tl.load(page_max + offsets, mask, 0).to(tl.float32)
    low = tl.load(page_min + offsets, mask, 0).to(tl.float32)
    best = tl.full((BLOCK_P,), -float("inf"), tl.float32)
    for head in range(GROUP_SIZE):
        q = tl.load(
            query + (row * QUERY_HEADS + kv_head * GROUP_SIZE + head) * HEAD_DIM + dims,
            dims < HEAD_DIM, 0,
        )
        q32 = q.to(tl.float32)
        positive = tl.sum(tl.where(q32[None, :] >= 0, q32[None, :] * high, 0), 1).to(q.dtype)
        negative = tl.sum(tl.where(q32[None, :] < 0, q32[None, :] * low, 0), 1).to(q.dtype)
        bound = (positive + negative).to(q.dtype)
        best = tl.maximum(best, bound.to(tl.float32))
    tl.store(partial + (row * KV_HEADS + kv_head) * NUM_PAGES + pages, best, pages < NUM_PAGES)


@triton.jit(do_not_specialize=["NUM_PAGES"])
def _max_head_bounds(
    partial, output, NUM_PAGES,
    NUM_PARTS: tl.constexpr, BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    pages = tl.program_id(1) * 256 + tl.arange(0, 256)
    heads = tl.arange(0, BLOCK_H)
    scores = tl.load(
        partial + (row * NUM_PARTS + heads[:, None]) * NUM_PAGES + pages[None, :],
        (heads[:, None] < NUM_PARTS) & (pages[None, :] < NUM_PAGES), -float("inf"),
    )
    tl.store(output + row * NUM_PAGES + pages, tl.max(scores.to(tl.float32), 0), pages < NUM_PAGES)


@triton.jit(do_not_specialize=["PAGES"])
def _repair_and_merge_page_bounds(
    query, high, low, slots, repair, partial, output,
    PAGES, HEADS: tl.constexpr, KV_HEADS: tl.constexpr,
    DIM: tl.constexpr, PARTS: tl.constexpr, HEAD_TILES: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_H: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    pages = tile * BLOCK_P + tl.arange(0, BLOCK_P)
    parts = tl.arange(0, BLOCK_H)
    offsets = (row * PARTS + parts[:, None]) * PAGES + pages[None, :]
    mask = (parts[:, None] < PARTS) & (pages[None, :] < PAGES)
    flags = tl.load(repair + offsets, mask, 0)
    scores = tl.load(partial + offsets, mask, -float("inf")).to(tl.float32)
    best = tl.max(scores, 0)
    # Unflagged bounds are already final. Merge them before loading metadata
    # for the rare flagged entries, avoiding a separate partial-write/merge pass.
    pending = flags != 0
    indices = parts[:, None] * BLOCK_P + tl.arange(0, BLOCK_P)[None, :]
    next_index = tl.min(tl.min(tl.where(pending, indices, 2147483647), 1), 0)
    dims = tl.arange(0, BLOCK_D)
    while next_index < PARTS * BLOCK_P:
        part = next_index // BLOCK_P
        page = tile * BLOCK_P + next_index % BLOCK_P
        bits = tl.sum(tl.sum(tl.where(indices == next_index, flags, 0), 1), 0)
        physical = tl.maximum(tl.load(slots + row * PAGES + page), 0)
        page_best = tl.max(tl.where(pages == page, best, -float("inf")), 0)
        while bits != 0:
            head = libdevice.ffs(bits) - 1
            kh = part // HEAD_TILES
            qh = kh * (HEADS // KV_HEADS) + part % HEAD_TILES * 16 + head
            q = tl.load(query + (row * HEADS + qh) * DIM + dims, dims < DIM, 0)
            offset = (physical * KV_HEADS + kh) * DIM + dims
            hi = tl.load(high + offset, dims < DIM, 0).to(tl.float32)
            lo = tl.load(low + offset, dims < DIM, 0).to(tl.float32)
            pos = tl.sum(tl.where(q >= 0, q.to(tl.float32) * hi, 0), 0).to(q.dtype)
            neg = tl.sum(tl.where(q < 0, q.to(tl.float32) * lo, 0), 0).to(q.dtype)
            page_best = tl.maximum(page_best, (pos + neg).to(q.dtype).to(tl.float32))
            bits = bits & ~(1 << head)
        best = tl.where(pages == page, page_best, best)
        pending = pending & (indices != next_index)
        next_index = tl.min(tl.min(tl.where(pending, indices, 2147483647), 1), 0)
    tl.store(output + row * PAGES + pages, best, pages < PAGES)


def _score_buffers(query, batch, parts, pages):
    output = torch.empty((batch, pages), device=query.device, dtype=query.dtype)
    partial = output if parts == 1 else torch.empty(
        (batch, parts, pages), device=query.device, dtype=query.dtype,
    )
    return partial, output


def _merge(partial, output, batch, parts, pages):
    if parts > 1:
        _max_head_bounds[(batch, triton.cdiv(pages, 256))](
            partial, output, pages, parts, triton.next_power_of_2(parts), num_warps=4,
        )
    return output


def score_quest_pages_tensorcore(
    query, page_max, page_min, page_table, *, block_pages=64, block_dim=128, num_warps=4,
):
    batch, heads, dim, kv_heads = _validate_quest_page_score_inputs(query, page_max, page_min, page_table)
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Tensor-core QuEST scoring requires BF16 or FP16.")
    if block_pages not in (16, 32, 64, 128) or block_dim not in (32, 64, 128):
        raise ValueError("Unsupported QuEST matrix-product tile.")
    pages = int(page_table.shape[1])
    head_tiles = triton.cdiv(heads // kv_heads, 16)
    parts = kv_heads * head_tiles
    partial, output = _score_buffers(query, batch, parts, pages)
    if not batch or not pages:
        return output
    repair = torch.empty((batch, parts, pages), device=query.device, dtype=torch.int32)
    _gqa_page_bounds[(batch, triton.cdiv(pages, block_pages), parts)](
        query, page_max, page_min, page_table, partial, repair,
        pages, heads, kv_heads, dim, heads // kv_heads, head_tiles, parts,
        block_pages, block_dim, num_warps=num_warps, num_stages=2,
    )
    _repair_and_merge_page_bounds[(batch, triton.cdiv(pages, 4))](
        query, page_max, page_min, page_table, repair, partial, output,
        pages, heads, kv_heads, dim, parts, head_tiles,
        triton.next_power_of_2(dim), 4, triton.next_power_of_2(parts), num_warps=(
            1 if heads == kv_heads == 1 else min(max(triton.next_power_of_2(dim) // 256, 1), 8)
        ),
    )
    return output


def score_quest_pages_vector(
    query, page_max, page_min, page_table, *, block_pages=8, num_warps=4,
):
    batch, heads, dim, kv_heads = _validate_quest_page_score_inputs(query, page_max, page_min, page_table)
    if block_pages not in (1, 2, 4, 8, 16, 32):
        raise ValueError("Unsupported QuEST vector tile.")
    pages = int(page_table.shape[1])
    partial, output = _score_buffers(query, batch, kv_heads, pages)
    if not batch or not pages:
        return output
    _vector_page_bounds[(batch, triton.cdiv(pages, block_pages), kv_heads)](
        query, page_max, page_min, page_table, partial,
        pages, heads, kv_heads, dim, heads // kv_heads,
        block_pages, triton.next_power_of_2(dim), num_warps=num_warps, num_stages=1,
    )
    return _merge(partial, output, batch, kv_heads, pages)
