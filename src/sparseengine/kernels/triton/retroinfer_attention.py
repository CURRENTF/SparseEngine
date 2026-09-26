"""Direct-slot RetroInfer decode with exact and estimated wave zones."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _retroinfer_paged_decode(
    Q, K, V, SLOT_TABLE, CENTROIDS, VALUE_SUMS, SIZES, OFFSETS,
    SORTED_POSITIONS, RANKED, OUT,
    H_KV: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
    D_BLOCK: tl.constexpr, CLUSTERS, TOP,
    INDEX_TOKENS, SINK_END,
    RECENT_START, RECENT_END,
    RETRIEVED, ESTIMATED,
    K_STRIDE: tl.constexpr, V_STRIDE: tl.constexpr,
    SCALE: tl.constexpr, BLOCK_K: tl.constexpr,
):
    head = tl.program_id(0)
    kv_head = head // GROUP
    dim = tl.arange(0, D_BLOCK)
    tokens = tl.arange(0, BLOCK_K)
    query = tl.load(Q + head * D + dim, dim < D, other=0).to(tl.float32)
    maximum = -float("inf")
    denominator = tl.full((), 0, tl.float32)
    numerator = tl.full((D_BLOCK,), 0, tl.float32)

    for start in range(tl.cdiv(SINK_END, BLOCK_K)):
        positions = start * BLOCK_K + tokens
        valid = positions < SINK_END
        slots = tl.load(SLOT_TABLE + positions, valid, other=0)
        key_offsets = slots[:, None] * K_STRIDE + kv_head * D + dim[None, :]
        value_offsets = slots[:, None] * V_STRIDE + kv_head * D + dim[None, :]
        keys = tl.load(K + key_offsets, valid[:, None] & (dim[None, :] < D), other=0).to(tl.float32)
        values = tl.load(V + value_offsets, valid[:, None] & (dim[None, :] < D), other=0).to(tl.float32)
        scores = tl.where(valid, tl.sum(keys * query[None, :], axis=1) * SCALE, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
        weights = tl.exp(scores - next_maximum)
        old_scale = tl.exp(maximum - next_maximum)
        denominator = denominator * old_scale + tl.sum(weights, axis=0)
        numerator = numerator * old_scale + tl.sum(values * weights[:, None], axis=0)
        maximum = next_maximum

    for start in range(tl.cdiv(RECENT_END - RECENT_START, BLOCK_K)):
        positions = RECENT_START + start * BLOCK_K + tokens
        valid = positions < RECENT_END
        slots = tl.load(SLOT_TABLE + positions, valid, other=0)
        key_offsets = slots[:, None] * K_STRIDE + kv_head * D + dim[None, :]
        value_offsets = slots[:, None] * V_STRIDE + kv_head * D + dim[None, :]
        keys = tl.load(K + key_offsets, valid[:, None] & (dim[None, :] < D), other=0).to(tl.float32)
        values = tl.load(V + value_offsets, valid[:, None] & (dim[None, :] < D), other=0).to(tl.float32)
        scores = tl.where(valid, tl.sum(keys * query[None, :], axis=1) * SCALE, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
        weights = tl.exp(scores - next_maximum)
        old_scale = tl.exp(maximum - next_maximum)
        denominator = denominator * old_scale + tl.sum(weights, axis=0)
        numerator = numerator * old_scale + tl.sum(values * weights[:, None], axis=0)
        maximum = next_maximum

    for rank in range(RETRIEVED):
        cluster = tl.load(RANKED + kv_head * TOP + rank)
        begin = tl.load(OFFSETS + kv_head * (CLUSTERS + 1) + cluster)
        end = tl.load(OFFSETS + kv_head * (CLUSTERS + 1) + cluster + 1)
        for start in range(tl.cdiv(end - begin, BLOCK_K)):
            locations = begin + start * BLOCK_K + tokens
            valid = locations < end
            positions = tl.load(
                SORTED_POSITIONS + kv_head * INDEX_TOKENS + locations,
                valid, other=0,
            )
            slots = tl.load(SLOT_TABLE + positions, valid, other=0)
            key_offsets = slots[:, None] * K_STRIDE + kv_head * D + dim[None, :]
            value_offsets = slots[:, None] * V_STRIDE + kv_head * D + dim[None, :]
            keys = tl.load(K + key_offsets, valid[:, None] & (dim[None, :] < D), other=0).to(tl.float32)
            values = tl.load(V + value_offsets, valid[:, None] & (dim[None, :] < D), other=0).to(tl.float32)
            scores = tl.where(valid, tl.sum(keys * query[None, :], axis=1) * SCALE, -float("inf"))
            next_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
            weights = tl.exp(scores - next_maximum)
            old_scale = tl.exp(maximum - next_maximum)
            denominator = denominator * old_scale + tl.sum(weights, axis=0)
            numerator = numerator * old_scale + tl.sum(values * weights[:, None], axis=0)
            maximum = next_maximum

    for rank in range(RETRIEVED, RETRIEVED + ESTIMATED):
        cluster = tl.load(RANKED + kv_head * TOP + rank)
        count = tl.load(SIZES + kv_head * CLUSTERS + cluster).to(tl.float32)
        if count > 0:
            center = tl.load(
                CENTROIDS + (kv_head * CLUSTERS + cluster) * D + dim,
                dim < D, other=0,
            ).to(tl.float32)
            value_sum = tl.load(
                VALUE_SUMS + (kv_head * CLUSTERS + cluster) * D + dim,
                dim < D, other=0,
            ).to(tl.float32)
            score = tl.sum(center * query, axis=0) * SCALE
            next_maximum = tl.maximum(maximum, score)
            weight = tl.exp(score - next_maximum)
            old_scale = tl.exp(maximum - next_maximum)
            denominator = denominator * old_scale + count * weight
            numerator = numerator * old_scale + value_sum * weight
            maximum = next_maximum
    tl.store(OUT + head * D + dim, numerator / denominator, dim < D)


def retroinfer_paged_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_table: torch.Tensor,
    centroids: torch.Tensor,
    value_sums: torch.Tensor,
    sizes: torch.Tensor,
    offsets: torch.Tensor,
    sorted_positions: torch.Tensor,
    ranked: torch.Tensor,
    *,
    sink_end: int,
    recent_start: int,
    recent_end: int,
    retrieval_clusters: int,
    estimation_clusters: int,
    output: torch.Tensor,
) -> None:
    """Run one request's exact sink/recent/retrieval plus estimated clusters."""
    heads, dim = query.shape
    kv_heads, clusters, _ = centroids.shape
    if (
        heads % kv_heads or query.shape[1] != key_cache.shape[-1]
        or key_cache.shape != value_cache.shape or key_cache.shape[1] != kv_heads
        or not 0 < sink_end <= recent_start <= recent_end <= slot_table.numel()
        or retrieval_clusters < 0 or estimation_clusters < 0
        or retrieval_clusters + estimation_clusters > clusters
        or output.shape != query.shape
    ):
        raise ValueError("RetroInfer paged decode has incompatible shape or zone bounds.")
    if any(not tensor.is_cuda or tensor.device != query.device for tensor in (
        key_cache, value_cache, slot_table, centroids, value_sums,
        sizes, offsets, sorted_positions, ranked, output,
    )):
        raise ValueError("RetroInfer paged decode requires one CUDA device.")
    if any(not tensor.is_contiguous() for tensor in (
        query, slot_table, centroids, value_sums, sizes, offsets,
        sorted_positions, ranked, output,
    )):
        raise ValueError("RetroInfer paged decode requires contiguous metadata.")
    if query.dtype not in (torch.float16, torch.bfloat16) or any(
        tensor.dtype != query.dtype for tensor in (key_cache, value_cache, centroids)
    ):
        raise TypeError("RetroInfer paged decode requires uniform FP16/BF16 QKV.")
    if value_sums.dtype != torch.float32 or any(
        tensor.dtype != torch.int32 for tensor in (slot_table, sizes, offsets, sorted_positions, ranked)
    ):
        raise TypeError("RetroInfer index metadata has incorrect dtype.")
    top = ranked.shape[1]
    if (
        ranked.shape != (kv_heads, top) or top < max(retrieval_clusters + estimation_clusters, 1)
        or sizes.shape != (kv_heads, clusters)
        or offsets.shape != (kv_heads, clusters + 1)
        or sorted_positions.shape[0] != kv_heads
        or value_sums.shape != centroids.shape
    ):
        raise ValueError("RetroInfer index metadata shapes are incompatible.")
    _retroinfer_paged_decode[(heads,)](
        query, key_cache, value_cache, slot_table, centroids, value_sums,
        sizes, offsets, sorted_positions, ranked, output,
        H_KV=kv_heads, GROUP=heads // kv_heads, D=dim,
        D_BLOCK=triton.next_power_of_2(dim), CLUSTERS=clusters, TOP=top,
        INDEX_TOKENS=sorted_positions.shape[1], SINK_END=sink_end,
        RECENT_START=recent_start, RECENT_END=recent_end,
        RETRIEVED=retrieval_clusters, ESTIMATED=estimation_clusters,
        K_STRIDE=key_cache.stride(0), V_STRIDE=value_cache.stride(0),
        SCALE=dim ** -0.5, BLOCK_K=32,
    )
