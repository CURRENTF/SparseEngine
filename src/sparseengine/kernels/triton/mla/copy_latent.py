# SPDX-License-Identifier: Apache-2.0
# Derived from ModelTC/lightllm at commit
# 65c174ee95ac6a6fd36b18b63d0b33d97e76b770:
# lightllm/common/basemodel/triton_kernel/kv_copy/mla_copy_kv.py
# Local changes: mask padded slot_mapping=-1, guard cache bounds, enforce the
# GLM latent layout, and offer explicit duplicate/range validation.

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .decode_stage1 import MLA_LATENT_DIM, MLA_ROPE_DIM


@triton.jit
def _copy_latent_kernel(
    latent,
    rope,
    slot_mapping,
    latent_cache,
    rope_cache,
    page_max,
    page_min,
    stride_latent_token,
    stride_latent_head,
    stride_latent_dim,
    stride_rope_token,
    stride_rope_head,
    stride_rope_dim,
    stride_cache_latent_slot,
    stride_cache_latent_head,
    stride_cache_latent_dim,
    stride_cache_rope_slot,
    stride_cache_rope_head,
    stride_cache_rope_dim,
    cache_slot_count,
    LATENT_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    UPDATE_QUEST_METADATA: tl.constexpr,
):
    token_index = tl.program_id(0)
    latent_offsets = tl.arange(0, LATENT_DIM)
    rope_offsets = tl.arange(0, ROPE_DIM)
    destination_slot = tl.load(slot_mapping + token_index).to(tl.int64)
    valid_slot = (destination_slot >= 0) & (
        destination_slot < cache_slot_count
    )
    safe_slot = tl.where(valid_slot, destination_slot, 0)

    latent_offsets_in = (
        token_index * stride_latent_token
        + latent_offsets * stride_latent_dim
    )
    rope_offsets_in = (
        token_index * stride_rope_token + rope_offsets * stride_rope_dim
    )
    latent_values = tl.load(
        latent + latent_offsets_in,
        mask=valid_slot,
        other=0.0,
    )
    rope_values = tl.load(
        rope + rope_offsets_in,
        mask=valid_slot,
        other=0.0,
    )

    latent_offsets_out = (
        safe_slot * stride_cache_latent_slot
        + latent_offsets * stride_cache_latent_dim
    )
    rope_offsets_out = (
        safe_slot * stride_cache_rope_slot
        + rope_offsets * stride_cache_rope_dim
    )
    tl.store(
        latent_cache + latent_offsets_out,
        latent_values,
        mask=valid_slot,
    )
    tl.store(
        rope_cache + rope_offsets_out,
        rope_values,
        mask=valid_slot,
    )
    if UPDATE_QUEST_METADATA:
        metadata_dim = LATENT_DIM + ROPE_DIM
        page_slot = safe_slot // PAGE_SIZE
        page_offset = safe_slot % PAGE_SIZE
        metadata_base = page_slot * metadata_dim
        update_mask = valid_slot & (page_offset != 0)
        old_latent_max = tl.load(
            page_max + metadata_base + latent_offsets,
            mask=update_mask,
            other=-float("inf"),
        )
        old_latent_min = tl.load(
            page_min + metadata_base + latent_offsets,
            mask=update_mask,
            other=float("inf"),
        )
        rope_metadata_offsets = metadata_base + LATENT_DIM + rope_offsets
        old_rope_max = tl.load(
            page_max + rope_metadata_offsets,
            mask=update_mask,
            other=-float("inf"),
        )
        old_rope_min = tl.load(
            page_min + rope_metadata_offsets,
            mask=update_mask,
            other=float("inf"),
        )
        tl.store(
            page_max + metadata_base + latent_offsets,
            tl.maximum(old_latent_max, latent_values),
            mask=valid_slot,
        )
        tl.store(
            page_min + metadata_base + latent_offsets,
            tl.minimum(old_latent_min, latent_values),
            mask=valid_slot,
        )
        tl.store(
            page_max + rope_metadata_offsets,
            tl.maximum(old_rope_max, rope_values),
            mask=valid_slot,
        )
        tl.store(
            page_min + rope_metadata_offsets,
            tl.minimum(old_rope_min, rope_values),
            mask=valid_slot,
        )


def _validate_copy_tensors(
    latent: torch.Tensor,
    rope: torch.Tensor,
    slot_mapping: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
) -> None:
    tensors = {
        "latent": latent,
        "rope": rope,
        "slot_mapping": slot_mapping,
        "latent_cache": latent_cache,
        "rope_cache": rope_cache,
    }
    for name, tensor in tensors.items():
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA tensor, got {tensor.device}")
        if tensor.device != latent.device:
            raise ValueError(
                f"{name} is on {tensor.device}, expected {latent.device}"
            )
    for name in ("latent", "rope", "latent_cache", "rope_cache"):
        if tensors[name].dtype != torch.bfloat16:
            raise TypeError(
                f"{name} must use {torch.bfloat16}, got {tensors[name].dtype}"
            )
    if slot_mapping.dtype != torch.int32:
        raise TypeError(
            f"slot_mapping must use {torch.int32}, got {slot_mapping.dtype}"
        )

    if latent.ndim != 3 or latent.shape[1:] != (1, MLA_LATENT_DIM):
        raise ValueError(
            "latent must have shape [tokens, 1, 512], got "
            f"{tuple(latent.shape)}"
        )
    if rope.ndim != 3 or rope.shape[1:] != (1, MLA_ROPE_DIM):
        raise ValueError(
            "rope must have shape [tokens, 1, 64], got "
            f"{tuple(rope.shape)}"
        )
    if latent_cache.ndim != 3 or latent_cache.shape[1:] != (
        1,
        MLA_LATENT_DIM,
    ):
        raise ValueError(
            "latent_cache must have shape [slots, 1, 512], got "
            f"{tuple(latent_cache.shape)}"
        )
    if rope_cache.ndim != 3 or rope_cache.shape[1:] != (1, MLA_ROPE_DIM):
        raise ValueError(
            "rope_cache must have shape [slots, 1, 64], got "
            f"{tuple(rope_cache.shape)}"
        )
    token_count = latent.shape[0]
    if rope.shape[0] != token_count or slot_mapping.shape != (token_count,):
        raise ValueError(
            "latent, rope, and slot_mapping token dimensions must match"
        )
    if latent_cache.shape[0] != rope_cache.shape[0]:
        raise ValueError("latent_cache and rope_cache must have equal slots")


def validate_copy_slot_mapping(
    slot_mapping: torch.Tensor,
    *,
    cache_slot_count: int,
) -> None:
    """Synchronously validate slots once at an owning runtime boundary."""

    if cache_slot_count <= 0:
        raise ValueError("cache_slot_count must be positive")
    valid_slots = slot_mapping[slot_mapping >= 0]
    if valid_slots.numel() == 0:
        return
    if bool(torch.any(valid_slots >= cache_slot_count).item()):
        raise ValueError(
            f"slot_mapping contains a slot outside [0, {cache_slot_count})"
        )
    if valid_slots.unique().numel() != valid_slots.numel():
        raise ValueError("slot_mapping contains duplicate non-padding slots")


def validate_copy_slot_mappings(
    slot_mappings: torch.Tensor,
    *,
    cache_slot_count: int,
) -> None:
    """Validate equally-sized layer mappings with one device synchronization."""

    if cache_slot_count <= 0:
        raise ValueError("cache_slot_count must be positive")
    if slot_mappings.ndim != 2:
        raise ValueError(
            "slot_mappings must have shape [layers, tokens], got "
            f"{tuple(slot_mappings.shape)}"
        )
    if slot_mappings.numel() == 0:
        return

    valid = slot_mappings >= 0
    out_of_bounds = torch.any(valid & (slot_mappings >= cache_slot_count))
    if int(slot_mappings.shape[1]) > 1:
        padding = torch.full_like(slot_mappings, cache_slot_count)
        sorted_slots = torch.sort(
            torch.where(valid, slot_mappings, padding),
            dim=1,
        ).values
        duplicate = torch.any(
            (sorted_slots[:, 1:] == sorted_slots[:, :-1])
            & (sorted_slots[:, 1:] != cache_slot_count)
        )
    else:
        duplicate = torch.zeros((), dtype=torch.bool, device=slot_mappings.device)

    # Materialize both flags together. The old per-layer path performed up to
    # one host synchronization per layer; decode mappings all share a width, so
    # this keeps exact ValueError diagnostics with one synchronization total.
    flags = torch.stack((out_of_bounds, duplicate)).to(device="cpu").tolist()
    if bool(flags[0]):
        raise ValueError(
            f"slot_mapping contains a slot outside [0, {cache_slot_count})"
        )
    if bool(flags[1]):
        raise ValueError("slot_mapping contains duplicate non-padding slots")


@torch.no_grad()
def copy_latent_to_cache(
    latent: torch.Tensor,
    rope: torch.Tensor,
    slot_mapping: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    *,
    validate_slots: bool = True,
) -> None:
    """Copy one token batch into latent MLA caches.

    Negative slots are padding and are never read or written. Set
    ``validate_slots=False`` when the owning runtime trusts its allocator
    invariants or has already validated this exact mapping. The kernel still
    masks padding and out-of-range slots.
    """

    _validate_copy_tensors(
        latent,
        rope,
        slot_mapping,
        latent_cache,
        rope_cache,
    )
    token_count = slot_mapping.numel()
    if token_count == 0:
        return
    cache_slot_count = latent_cache.shape[0]
    if validate_slots:
        validate_copy_slot_mapping(
            slot_mapping,
            cache_slot_count=cache_slot_count,
        )

    _copy_latent_kernel[(token_count,)](
        latent,
        rope,
        slot_mapping,
        latent_cache,
        rope_cache,
        latent_cache,
        latent_cache,
        *latent.stride(),
        *rope.stride(),
        *latent_cache.stride(),
        *rope_cache.stride(),
        cache_slot_count=cache_slot_count,
        LATENT_DIM=MLA_LATENT_DIM,
        ROPE_DIM=MLA_ROPE_DIM,
        PAGE_SIZE=1,
        UPDATE_QUEST_METADATA=False,
        num_warps=1,
        num_stages=1,
    )


@torch.no_grad()
def copy_latent_to_cache_with_quest_metadata(
    latent: torch.Tensor,
    rope: torch.Tensor,
    slot_mapping: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    page_max: torch.Tensor,
    page_min: torch.Tensor,
    *,
    page_size: int,
    validate_slots: bool = True,
) -> None:
    """Copy decode MLA state and update its fused QuEST page bounds."""

    _validate_copy_tensors(
        latent,
        rope,
        slot_mapping,
        latent_cache,
        rope_cache,
    )
    metadata_dim = MLA_LATENT_DIM + MLA_ROPE_DIM
    expected_tail = (1, metadata_dim)
    if (
        page_max.shape != page_min.shape
        or page_max.ndim != 3
        or tuple(page_max.shape[1:]) != expected_tail
    ):
        raise ValueError(
            "MLA QuEST metadata must have matching [pages, 1, 576] shapes"
        )
    for name, tensor in {"page_max": page_max, "page_min": page_min}.items():
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must use torch.bfloat16, got {tensor.dtype}")
        if tensor.device != latent.device:
            raise ValueError(f"{name} must share the MLA cache device")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    page_size = int(page_size)
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    expected_pages = (int(latent_cache.shape[0]) + page_size - 1) // page_size
    if int(page_max.shape[0]) != expected_pages:
        raise ValueError(
            "MLA QuEST metadata capacity must cover the cache exactly: "
            f"expected_pages={expected_pages} got={int(page_max.shape[0])}"
        )
    token_count = slot_mapping.numel()
    if token_count == 0:
        return
    cache_slot_count = latent_cache.shape[0]
    if validate_slots:
        validate_copy_slot_mapping(
            slot_mapping,
            cache_slot_count=cache_slot_count,
        )

    _copy_latent_kernel[(token_count,)](
        latent,
        rope,
        slot_mapping,
        latent_cache,
        rope_cache,
        page_max,
        page_min,
        *latent.stride(),
        *rope.stride(),
        *latent_cache.stride(),
        *rope_cache.stride(),
        cache_slot_count=cache_slot_count,
        LATENT_DIM=MLA_LATENT_DIM,
        ROPE_DIM=MLA_ROPE_DIM,
        PAGE_SIZE=page_size,
        UPDATE_QUEST_METADATA=True,
        num_warps=1,
        num_stages=1,
    )
