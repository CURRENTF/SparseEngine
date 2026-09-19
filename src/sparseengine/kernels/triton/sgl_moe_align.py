# SPDX-License-Identifier: Apache-2.0
# Copyright 2023-2026 SGLang Team
"""Small-batch MoE alignment adapted from SGLang.

Source: sgl-project/sglang@24d625698d44c78f6e8ab8b7c19f96f45bbaa90a
``python/sglang/kernels/ops/moe/moe_align_small_numel.py``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SMALL_NUMEL_LIMIT = 64


@triton.jit
def _moe_align_small_numel_kernel(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    num_experts,
    block_size,
    numel,
    NP: tl.constexpr,
    NB: tl.constexpr,
):
    pair_offsets = tl.arange(0, NP)
    pair_mask = pair_offsets < numel
    expert_ids = tl.load(topk_ids_ptr + pair_offsets, mask=pair_mask, other=-2)
    buckets = tl.where(pair_mask, (expert_ids + 1).to(tl.int32), num_experts)

    same_bucket = (
        (buckets[None, :] == buckets[:, None])
        & pair_mask[None, :]
        & pair_mask[:, None]
    )
    earlier = pair_offsets[None, :] < pair_offsets[:, None]
    rank = tl.sum((same_bucket & earlier).to(tl.int32), axis=1)
    count = tl.sum(same_bucket.to(tl.int32), axis=1)
    padded_count = ((count + block_size - 1) // block_size) * block_size
    representative = (rank == 0) & pair_mask

    smaller_representative = (
        (buckets[None, :] < buckets[:, None]) & representative[None, :]
    )
    exclusive_offset = tl.sum(
        smaller_representative.to(tl.int32) * padded_count[None, :],
        axis=1,
    )
    total = tl.sum(tl.where(representative, padded_count, 0), axis=0)
    tl.store(num_tokens_post_pad_ptr, total.to(tl.int32))

    block_offsets = tl.arange(0, NB)
    block_starts = block_offsets * block_size
    block_owned = (
        (block_starts[:, None] >= exclusive_offset[None, :])
        & (block_starts[:, None] < (exclusive_offset + padded_count)[None, :])
        & representative[None, :]
    )
    block_expert_ids = tl.sum(
        block_owned.to(tl.int32) * (buckets[None, :] - 1),
        axis=1,
    )
    tl.store(
        expert_ids_ptr + block_offsets,
        block_expert_ids.to(tl.int32),
        mask=block_starts < total,
    )

    fill_iterations = (total + NP - 1) // NP
    for iteration in range(fill_iterations):
        fill_offsets = iteration * NP + pair_offsets
        tl.store(
            sorted_token_ids_ptr + fill_offsets,
            tl.full([NP], 0, tl.int32) + numel,
            mask=fill_offsets < total,
        )
    tl.debug_barrier()
    positions = exclusive_offset + rank
    tl.store(
        sorted_token_ids_ptr + positions,
        pair_offsets.to(tl.int32),
        mask=pair_mask,
    )


def sgl_moe_align_small_numel(
    topk_ids: torch.Tensor,
    *,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
) -> None:
    """Write SGL-compatible alignment metadata in one Triton launch."""

    numel = int(topk_ids.numel())
    if not 0 < numel <= SMALL_NUMEL_LIMIT:
        raise ValueError(
            f"small-numel alignment requires 1..{SMALL_NUMEL_LIMIT} IDs, got {numel}."
        )
    _moe_align_small_numel_kernel[(1,)](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        int(num_experts) + 1,
        int(block_size),
        numel,
        NP=triton.next_power_of_2(max(numel, 2)),
        NB=triton.next_power_of_2(max(int(expert_ids.numel()), 2)),
        num_warps=4,
    )


__all__ = ["SMALL_NUMEL_LIMIT", "sgl_moe_align_small_numel"]
