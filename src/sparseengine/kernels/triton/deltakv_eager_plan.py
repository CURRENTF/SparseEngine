"""Compact eager reconstruction metadata before exact host slot allocation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare(
    RAW,
    LATENT,
    ROWS,
    LENS,
    CANDIDATES,
    TOP,
    POS,
    LAT,
    STATS,
    RAW_S0: tl.constexpr,
    RAW_S1: tl.constexpr,
    LAT_S0: tl.constexpr,
    LAT_S1: tl.constexpr,
    C_S0: tl.constexpr,
    C_S1: tl.constexpr,
    K: tl.constexpr,
    SINK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    row = tl.load(ROWS + b)
    total = tl.load(LENS + b * 2)
    compressed = tl.load(LENS + b * 2 + 1)
    sink = tl.minimum(SINK, total)
    compressed = tl.minimum(compressed, tl.maximum(0, total - sink))
    tail_start = sink + compressed
    i = tl.arange(0, BLOCK)
    candidate = tl.load(CANDIDATES + b * C_S0 + i * C_S1, i < K, -1)
    valid = (i < K) & (candidate >= 0) & (candidate < compressed)
    position = candidate + SINK
    raw = tl.load(RAW + row * RAW_S0 + position * RAW_S1, valid, -1)
    latent = tl.load(LATENT + row * LAT_S0 + position * LAT_S1, valid, -1)
    need = valid & (latent >= 0)
    selected_rank = tl.cumsum(valid.to(tl.int32)) - 1
    need_rank = tl.cumsum(need.to(tl.int32)) - 1
    # Negative values encode a row-local reconstruction slot until allocation.
    tl.store(TOP + b * K + selected_rank, tl.where(need, -2 - need_rank, raw), valid)
    tl.store(POS + b * K + need_rank, position, need)
    tl.store(LAT + b * K + need_rank, latent, need)
    bad_sink = tl.full((), 0, tl.int32)
    bad_tail = tl.full((), 0, tl.int32)
    offsets = tl.arange(0, 1024)
    raw_count = sink + total - tail_start
    for block in range(tl.cdiv(raw_count, 1024)):
        offset = block * 1024 + offsets
        p = tl.where(offset < sink, offset, tail_start + offset - sink)
        slot = tl.load(RAW + row * RAW_S0 + p * RAW_S1, offset < raw_count, 0)
        bad_sink |= tl.sum(((offset < sink) & (slot < 0)).to(tl.int32)) > 0
        bad_tail |= (
            tl.sum(((offset >= sink) & (offset < raw_count) & (slot < 0)).to(tl.int32))
            > 0
        )
    tl.store(STATS + b * 4, tl.sum(valid.to(tl.int32)))
    tl.store(STATS + b * 4 + 1, tl.sum(need.to(tl.int32)))
    tl.store(STATS + b * 4 + 2, bad_sink)
    tl.store(STATS + b * 4 + 3, bad_tail)


@triton.jit
def _finish(
    RAW,
    ROWS,
    LENS,
    TOP,
    POS,
    LAT,
    STATS,
    TEMP,
    ACTIVE,
    RECON_POS,
    RECON_LAT,
    CONTEXT,
    RAW_S0: tl.constexpr,
    RAW_S1: tl.constexpr,
    K: tl.constexpr,
    SINK: tl.constexpr,
    WIDTH: tl.constexpr,
    BATCH: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b, block = tl.program_id(0), tl.program_id(1)
    row = tl.load(ROWS + b)
    total = tl.load(LENS + b * 2)
    sink = tl.minimum(SINK, total)
    compressed = tl.minimum(tl.load(LENS + b * 2 + 1), tl.maximum(0, total - sink))
    kept = tl.load(STATS + b * 4)
    needed = tl.load(STATS + b * 4 + 1)
    batches = tl.arange(0, BATCH_BLOCK)
    previous = tl.load(STATS + batches * 4 + 1, (batches < b) & (batches < BATCH), 0)
    temp_start = tl.sum(previous)
    length = sink + kept + total - sink - compressed
    offset = block * BLOCK + tl.arange(0, BLOCK)
    in_top = (offset >= sink) & (offset < sink + kept)
    raw_pos = tl.where(offset < sink, offset, offset - kept + compressed)
    raw = tl.load(RAW + row * RAW_S0 + raw_pos * RAW_S1, (offset < length) & ~in_top, 0)
    top = tl.load(TOP + b * K + offset - sink, in_top, 0)
    temp = tl.load(TEMP + temp_start - top - 2, in_top & (top < -1), 0)
    value = tl.where(in_top, tl.where(top < -1, temp, top), raw)
    tl.store(ACTIVE + b * WIDTH + offset, value, offset < WIDTH)
    pos = tl.load(POS + b * K + offset, offset < needed, 0)
    latent = tl.load(LAT + b * K + offset, offset < needed, 0)
    tl.store(RECON_POS + temp_start + offset, pos, offset < needed)
    tl.store(RECON_LAT + temp_start + offset, latent, offset < needed)
    if block == 0:
        tl.store(CONTEXT + b, length)


def prepare_eager_plan(raw, latent, rows, lengths, candidates, sink):
    batch, k = candidates.shape
    top = torch.empty((batch, k), dtype=torch.int32, device=rows.device)
    positions, latent_slots = torch.empty_like(top), torch.empty_like(top)
    stats = torch.empty((batch, 4), dtype=torch.int32, device=rows.device)
    _prepare[(batch,)](
        raw,
        latent,
        rows,
        lengths,
        candidates,
        top,
        positions,
        latent_slots,
        stats,
        *raw.stride(),
        *latent.stride(),
        *candidates.stride(),
        k,
        sink,
        max(1, triton.next_power_of_2(k)),
    )
    return top, positions, latent_slots, stats


def finish_eager_plan(
    raw, rows, lengths, plan, temp, active, recon_pos, recon_lat, context, sink
):
    top, positions, latent_slots, stats = plan
    batch, k = top.shape
    width = active.shape[1]
    _finish[(batch, triton.cdiv(max(width, k, 1), 256))](
        raw,
        rows,
        lengths,
        top,
        positions,
        latent_slots,
        stats,
        temp,
        active,
        recon_pos,
        recon_lat,
        context,
        *raw.stride(),
        k,
        sink,
        width,
        batch,
        triton.next_power_of_2(batch),
        256,
    )
