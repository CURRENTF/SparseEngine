from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


@triton.jit
def store_kvcache_with_quest_metadata_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    page_max_ptr,
    page_min_ptr,
    D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    offsets = tl.arange(0, D)
    key = tl.load(key_ptr + idx * key_stride + offsets)
    value = tl.load(value_ptr + idx * value_stride + offsets)
    cache_offsets = slot * D + offsets
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

    page_slot = slot // PAGE_SIZE
    page_offset = slot % PAGE_SIZE
    metadata_offsets = page_slot * D + offsets
    old_max = tl.load(
        page_max_ptr + metadata_offsets,
        mask=page_offset != 0,
        other=-float("inf"),
    )
    old_min = tl.load(
        page_min_ptr + metadata_offsets,
        mask=page_offset != 0,
        other=float("inf"),
    )
    tl.store(page_max_ptr + metadata_offsets, tl.maximum(old_max, key))
    tl.store(page_min_ptr + metadata_offsets, tl.minimum(old_min, key))


@triton.jit
def store_prefill_kvcache_with_quest_metadata_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    page_segments_ptr,
    page_max_ptr,
    page_min_ptr,
    D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    segment_idx = tl.program_id(0)
    feature_block = tl.program_id(1)
    token_start = tl.load(page_segments_ptr + segment_idx * 2)
    token_count = tl.load(page_segments_ptr + segment_idx * 2 + 1)

    token_offsets = tl.arange(0, PAGE_SIZE)
    feature_offsets = feature_block * BLOCK_D + tl.arange(0, BLOCK_D)
    token_mask = token_offsets < token_count
    feature_mask = feature_offsets < D
    io_mask = token_mask[:, None] & feature_mask[None, :]

    input_offsets = (
        (token_start + token_offsets[:, None]) * key_stride
        + feature_offsets[None, :]
    )
    keys = tl.load(key_ptr + input_offsets, mask=io_mask, other=0.0)
    values = tl.load(
        value_ptr
        + (token_start + token_offsets[:, None]) * value_stride
        + feature_offsets[None, :],
        mask=io_mask,
        other=0.0,
    )
    slots = tl.load(
        slot_mapping_ptr + token_start + token_offsets,
        mask=token_mask,
        other=0,
    )
    cache_offsets = slots[:, None] * D + feature_offsets[None, :]
    tl.store(k_cache_ptr + cache_offsets, keys, mask=io_mask)
    tl.store(v_cache_ptr + cache_offsets, values, mask=io_mask)

    first_slot = tl.load(slot_mapping_ptr + token_start)
    page_slot = first_slot // PAGE_SIZE
    page_offset = first_slot % PAGE_SIZE
    metadata_offsets = page_slot * D + feature_offsets
    new_max = tl.max(
        tl.where(token_mask[:, None], keys, -float("inf")), axis=0
    )
    new_min = tl.min(
        tl.where(token_mask[:, None], keys, float("inf")), axis=0
    )
    old_max = tl.load(
        page_max_ptr + metadata_offsets,
        mask=(page_offset != 0) & feature_mask,
        other=-float("inf"),
    )
    old_min = tl.load(
        page_min_ptr + metadata_offsets,
        mask=(page_offset != 0) & feature_mask,
        other=float("inf"),
    )
    tl.store(
        page_max_ptr + metadata_offsets,
        tl.maximum(old_max, new_max),
        mask=feature_mask,
    )
    tl.store(
        page_min_ptr + metadata_offsets,
        tl.minimum(old_min, new_min),
        mask=feature_mask,
    )


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    n_tokens, num_heads, head_dim = key.shape
    d_model = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(-1) == 1
    assert slot_mapping.numel() == n_tokens
    max_launch_tokens = int(os.getenv("SPARSEENGINE_STORE_KVCACHE_CHUNK_TOKENS", "524288") or 0)
    if max_launch_tokens <= 0 or n_tokens <= max_launch_tokens:
        store_kvcache_kernel[(n_tokens,)](
            key,
            key.stride(0),
            value,
            value.stride(0),
            k_cache,
            v_cache,
            slot_mapping,
            d_model,
        )
        return

    for start in range(0, n_tokens, max_launch_tokens):
        end = min(n_tokens, start + max_launch_tokens)
        store_kvcache_kernel[(end - start,)](
            key[start:end],
            key.stride(0),
            value[start:end],
            value.stride(0),
            k_cache,
            v_cache,
            slot_mapping[start:end],
            d_model,
        )


def store_kvcache_with_quest_metadata(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_max: torch.Tensor,
    page_min: torch.Tensor,
    *,
    page_size: int,
) -> None:
    """Store decode KV and update the owning QuEST page bounds atomically."""

    n_tokens, num_heads, head_dim = key.shape
    d_model = num_heads * head_dim
    if page_max.shape != page_min.shape or page_max.shape[1:] != key.shape[1:]:
        raise ValueError(
            "QuEST page metadata must have matching [pages, heads, dim] shapes"
        )
    if page_max.dtype != key.dtype or page_min.dtype != key.dtype:
        raise TypeError("QuEST page metadata must use the key dtype")
    if page_max.device != key.device or page_min.device != key.device:
        raise ValueError("QuEST page metadata must share the key device")
    if not page_max.is_contiguous() or not page_min.is_contiguous():
        raise ValueError("QuEST page metadata must be contiguous")
    if key.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError("KV innermost dimensions must be contiguous")
    if key.stride(1) != head_dim or value.stride(1) != head_dim:
        raise ValueError("KV heads must be densely packed")
    if k_cache.stride(-1) != 1 or slot_mapping.numel() != n_tokens:
        raise ValueError("KV cache layout or slot_mapping shape is invalid")
    page_size = int(page_size)
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    expected_pages = (int(k_cache.shape[0]) + page_size - 1) // page_size
    if int(page_max.shape[0]) != expected_pages:
        raise ValueError(
            "QuEST page metadata capacity must cover the KV cache exactly: "
            f"expected_pages={expected_pages} got={int(page_max.shape[0])}"
        )
    if n_tokens == 0:
        return
    store_kvcache_with_quest_metadata_kernel[(n_tokens,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        page_max,
        page_min,
        D=d_model,
        PAGE_SIZE=page_size,
    )


def store_prefill_kvcache_with_quest_metadata(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_segments: torch.Tensor,
    page_max: torch.Tensor,
    page_min: torch.Tensor,
    *,
    page_size: int,
) -> None:
    """Store prefill KV and reduce QuEST bounds with one owner per touched page."""

    n_tokens, num_heads, head_dim = key.shape
    d_model = num_heads * head_dim
    if key.shape != value.shape:
        raise ValueError("Prefill K/V tensors must have matching shapes")
    if page_segments.ndim != 2 or tuple(page_segments.shape[1:]) != (2,):
        raise ValueError("QuEST prefill page_segments must have shape [pages, 2]")
    if page_segments.dtype != torch.int32:
        raise TypeError("QuEST prefill page_segments must use torch.int32")
    if page_segments.device != key.device:
        raise ValueError("QuEST prefill page_segments must share the key device")
    if not page_segments.is_contiguous():
        raise ValueError("QuEST prefill page_segments must be contiguous")
    if page_max.shape != page_min.shape or page_max.shape[1:] != key.shape[1:]:
        raise ValueError(
            "QuEST page metadata must have matching [pages, heads, dim] shapes"
        )
    if page_max.dtype != key.dtype or page_min.dtype != key.dtype:
        raise TypeError("QuEST page metadata must use the key dtype")
    if page_max.device != key.device or page_min.device != key.device:
        raise ValueError("QuEST page metadata must share the key device")
    if not page_max.is_contiguous() or not page_min.is_contiguous():
        raise ValueError("QuEST page metadata must be contiguous")
    if key.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError("KV innermost dimensions must be contiguous")
    if key.stride(1) != head_dim or value.stride(1) != head_dim:
        raise ValueError("KV heads must be densely packed")
    if k_cache.stride(-1) != 1 or slot_mapping.shape != (n_tokens,):
        raise ValueError("KV cache layout or slot_mapping shape is invalid")
    if slot_mapping.dtype != torch.int32 or slot_mapping.device != key.device:
        raise TypeError("slot_mapping must be an int32 tensor on the key device")
    page_size = int(page_size)
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    expected_pages = (int(k_cache.shape[0]) + page_size - 1) // page_size
    if int(page_max.shape[0]) != expected_pages:
        raise ValueError(
            "QuEST page metadata capacity must cover the KV cache exactly: "
            f"expected_pages={expected_pages} got={int(page_max.shape[0])}"
        )
    num_segments = int(page_segments.shape[0])
    if n_tokens == 0:
        if num_segments != 0:
            raise ValueError("Empty prefill input requires an empty page plan")
        return
    if num_segments == 0:
        raise ValueError("Non-empty prefill input requires a non-empty page plan")

    # A wide feature tile amortizes page-plan and slot loads for QuEST's small
    # physical pages.
    block_d = min(1024, triton.next_power_of_2(d_model))
    store_prefill_kvcache_with_quest_metadata_kernel[
        (num_segments, triton.cdiv(d_model, block_d))
    ](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        page_segments,
        page_max,
        page_min,
        D=d_model,
        PAGE_SIZE=page_size,
        BLOCK_D=block_d,
        num_warps=4,
    )
