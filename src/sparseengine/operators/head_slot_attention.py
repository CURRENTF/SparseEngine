"""Virtual MQA adaptation; only queries and index metadata change layout."""

from dataclasses import replace

import torch

from sparseengine.engine.cache_manager.base import (
    AttentionViewMeta, DecodeComputeView, ExplicitKVPayload, HeadSlotViewMeta,
    PrefillComputeView,
)
from sparseengine.kernels.triton.head_slots import reorder_head_queries


def virtual_head_view(view):
    meta, payload = view.meta, view.payload
    if not isinstance(meta, HeadSlotViewMeta) or payload.backend != "head_slots":
        raise TypeError("Independent head attention requires typed HeadSlotViewMeta and payload.")
    rows, heads, width = meta.active_slots.shape
    if meta.context_lens.shape != (meta.req_indices.numel(), heads):
        raise ValueError("Head-slot lengths must have shape [batch, KV heads].")
    # The final resident row is a dedicated empty padding row. Inactive requests
    # never address a live row, including through the upstream page-table reader.
    req = torch.where(meta.req_indices >= 0, meta.req_indices, rows - 1)
    virtual_rows = (req[:, None] * heads + torch.arange(heads, device=req.device,
                                                       dtype=torch.int32)).flatten()
    return AttentionViewMeta(
        active_slots=meta.active_slots.view(rows * heads, width),
        req_indices=virtual_rows,
        context_lens=meta.context_lens.flatten(),
        max_context_len=meta.max_context_len,
    ), ExplicitKVPayload(payload.k_cache.unsqueeze(1), payload.v_cache.unsqueeze(1))


def run_head_slot_decode(provider, spec, q, view, **kwargs):
    meta, payload = virtual_head_view(view)
    heads, groups = spec.num_kv_heads, spec.num_query_heads // spec.num_kv_heads
    virtual_spec = replace(spec, kv_cache_layout="token", num_query_heads=groups,
                           num_kv_heads=1, max_batch_size=spec.max_batch_size * heads)
    output = provider.run(virtual_spec, q.reshape(-1, groups, spec.head_dim),
                          DecodeComputeView(meta, payload), **kwargs)
    return output.reshape(q.shape)


def run_head_slot_prefill(provider, spec, q, view, *, qo_indptr, chunk_lens,
                          max_context_len, layer_idx):
    meta, payload = virtual_head_view(view)
    heads, groups = spec.num_kv_heads, spec.num_query_heads // spec.num_kv_heads
    from sparseengine.utils.context import get_context

    scope = get_context().attention_validation_scope
    if provider._query_plan_scope is not scope:
        provider._max_query_len = int(chunk_lens.max().item())
        provider._query_plan_scope = scope
    max_query_len = provider._max_query_len
    packed_q = reorder_head_queries(q, qo_indptr, heads, max_query_len)
    offsets = torch.arange(heads, device=q.device, dtype=torch.int32)
    virtual_cu = torch.cat(((qo_indptr[:-1, None] * heads
                            + chunk_lens[:, None] * offsets).flatten(),
                            qo_indptr[-1:] * heads))
    virtual_spec = replace(spec, kv_cache_layout="token", num_query_heads=groups,
                           num_kv_heads=1)
    output = provider.run(virtual_spec, packed_q, PrefillComputeView(meta, payload),
                          qo_indptr=virtual_cu, chunk_lens=chunk_lens.repeat_interleave(heads),
                          max_context_len=max_context_len, layer_idx=layer_idx)
    return reorder_head_queries(output, qo_indptr, heads, max_query_len, inverse=True)
