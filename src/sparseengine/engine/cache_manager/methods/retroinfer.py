"""Request-owned GPU wave indices over Standard's full explicit KV cache."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sparseengine.engine.sequence import Sequence

from ..base import AttentionViewMeta, DecodeComputeView, SparseSelection
from ..standard import StandardCacheManager
from .retroinfer_index import (
    RetroInferIndex,
    append_retroinfer_index,
    build_retroinfer_segment,
)


@dataclass(frozen=True)
class RetroInferDecodeRow:
    row_index: int
    sink_end: int
    recent_start: int
    recent_end: int
    index: RetroInferIndex | None


@dataclass(frozen=True)
class RetroInferDecodeViewMeta(AttentionViewMeta):
    rows: tuple[RetroInferDecodeRow, ...] = ()


class RetroInferGPUCacheManager(StandardCacheManager):
    """Keep all KV on GPU and add per-request, per-layer clustered views."""

    def __init__(self, config, parallel_context, *, allocation_budget_bytes=None):
        super().__init__(
            config, parallel_context,
            allocation_budget_bytes=allocation_budget_bytes,
        )
        from ..storage import ExplicitKVStorage

        if not isinstance(self.attention_cache_storage, ExplicitKVStorage):
            raise TypeError("RetroInfer GPU-only requires uniform explicit KV storage.")
        self._indices: dict[int, tuple[RetroInferIndex, ...]] = {}
        self._decode_seq_ids: tuple[int, ...] = ()

    def _get_available_slots_info(self) -> tuple[int, int]:
        available, layer_slot_bytes = super()._get_available_slots_info()
        if getattr(self.config, "startup_cache_phase", "production") == "profiling":
            return available, layer_slot_bytes
        # Reserve index memory for every allocatable KV slot, plus one full
        # context's clustering workspace. The builder processes layers in turn.
        element_bytes = self._cache_slot_dtype_size()
        heads = self.num_kv_heads
        dim = self.head_dim
        layers = self.num_kv_layers
        avg = self.config.retroinfer_avg_cluster_size
        index_bytes_per_token_per_layer = (
            4 * heads + (heads * dim * (element_bytes + 4) + 8 * heads) / avg
        )
        kv_bytes_per_token = layers * layer_slot_bytes
        indexed_bytes_per_token = layers * index_bytes_per_token_per_layer
        workspace_bytes = int(self.max_model_len * layer_slot_bytes * 4)
        if available <= workspace_bytes + kv_bytes_per_token:
            raise RuntimeError(
                "RetroInfer GPU-only cannot reserve its index build workspace: "
                f"available={available} required>{workspace_bytes + kv_bytes_per_token}."
            )
        remaining = available - workspace_bytes
        slots = int(remaining // (kv_bytes_per_token + 2 * indexed_bytes_per_token))
        if slots <= 0:
            raise RuntimeError("RetroInfer GPU-only has no KV slots after index reservation.")
        return slots * kv_bytes_per_token, layer_slot_bytes

    def _prepare_decode(self, seqs: list[Sequence]):
        result = super()._prepare_decode(seqs)
        self._decode_seq_ids = tuple(seq.seq_id for seq in seqs)
        return result

    def prepare_decode_graph_step(self, seqs, state):
        # Eager decode also uses the static graph-compatible input path.
        result = super().prepare_decode_graph_step(seqs, state)
        self._decode_seq_ids = tuple(seq.seq_id for seq in seqs)
        return result

    def prepare_decode_static(
        self, seqs, input_ids, positions, slot_mapping, context_lens, req_indices,
    ):
        result = super().prepare_decode_static(
            seqs, input_ids, positions, slot_mapping, context_lens, req_indices,
        )
        self._decode_seq_ids = tuple(seq.seq_id for seq in seqs)
        return result

    def _build_segment_for_layer(
        self,
        layer_idx: int,
        row: int,
        start: int,
        end: int,
        *,
        segments: int,
    ) -> RetroInferIndex:
        slots = self.buffer_req_to_token_slots[row, start:end].to(torch.int64)
        key_cache, value_cache = self.get_layer_kv_cache(layer_idx)
        keys = key_cache.index_select(0, slots).transpose(0, 1).contiguous()
        values = value_cache.index_select(0, slots).transpose(0, 1).contiguous()
        return build_retroinfer_segment(
            keys, values,
            start_position=start,
            avg_cluster_size=self.config.retroinfer_avg_cluster_size,
            num_segments=segments,
            retrieval_ratio=self.config.retroinfer_retrieval_ratio,
            estimation_ratio=self.config.retroinfer_estimation_ratio,
        )

    def _update_index(self, seq_id: int) -> None:
        if getattr(self.config, "startup_cache_phase", "production") == "profiling":
            return
        row = self.seq_id_to_row.get(seq_id)
        if row is None:
            return
        length = int(self.row_seq_lens[row])
        sink = self.config.retroinfer_sink_tokens
        recent = self.config.retroinfer_recent_tokens
        layers = tuple(int(x) for x in self.runtime_layout.kv_idx_to_layer_idx)
        previous = self._indices.get(seq_id)
        if previous is None:
            start, end = sink, length - recent
            if end - start < self.config.retroinfer_min_index_tokens:
                return
            segments = max(1, (end - start) // 8192)
            built = tuple(
                self._build_segment_for_layer(layer, row, start, end, segments=segments)
                for layer in layers
            )
            self._indices[seq_id] = built
            return
        indexed_end = previous[0].indexed_end
        update = self.config.retroinfer_update_tokens
        while length - recent - indexed_end >= update:
            next_end = indexed_end + update
            built = tuple(
                append_retroinfer_index(
                    old,
                    self._build_segment_for_layer(
                        layer, row, indexed_end, next_end, segments=1,
                    ),
                )
                for layer, old in zip(layers, previous)
            )
            self._indices[seq_id] = built
            previous = built
            indexed_end = next_end

    def on_forward_end(self, seqs: list[Sequence], is_prefill: bool):
        result = super().on_forward_end(seqs, is_prefill)
        for seq in seqs:
            if not is_prefill or seq.is_last_chunk_prefill:
                self._update_index(seq.seq_id)
        return result

    def free_seq(self, seq_id: int):
        self._indices.pop(seq_id, None)
        return super().free_seq(seq_id)

    def build_decode_compute_view(
        self,
        layer_idx: int,
        q: torch.Tensor,
        selection: SparseSelection,
        *,
        num_heads: int,
        num_kv_heads: int,
    ) -> DecodeComputeView:
        view = super().build_decode_compute_view(
            layer_idx, q, selection,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
        )
        if not self._decode_seq_ids or len(self._decode_seq_ids) > q.shape[0]:
            raise RuntimeError("RetroInfer decode sequence identities do not match the query batch.")
        kv_layer = self.kv_layer_index(layer_idx)
        rows = []
        # Static eager execution may pad to a graph bucket. Its inactive rows
        # mirror one real request and their outputs are discarded by the runner.
        batch_seq_ids = self._decode_seq_ids + (
            self._decode_seq_ids[0],
        ) * (q.shape[0] - len(self._decode_seq_ids))
        for seq_id in batch_seq_ids:
            row = self.seq_id_to_row[seq_id]
            length = int(self.row_seq_lens[row])
            sink_end = min(self.config.retroinfer_sink_tokens, length)
            indices = self._indices.get(seq_id)
            index = None if indices is None else indices[kv_layer]
            recent_start = sink_end if index is None else index.indexed_end
            if not 0 < sink_end <= recent_start <= length:
                raise RuntimeError("RetroInfer index and live KV range are inconsistent.")
            rows.append(RetroInferDecodeRow(row, sink_end, recent_start, length, index))
        meta = view.meta
        return DecodeComputeView(
            meta=RetroInferDecodeViewMeta(
                active_slots=meta.active_slots,
                req_indices=meta.req_indices,
                context_lens=meta.context_lens,
                max_context_len=meta.max_context_len,
                attn_score=meta.attn_score,
                temp_slots=meta.temp_slots,
                is_sparse=any(row.index is not None for row in rows),
                rows=tuple(rows),
            ),
            payload=view.payload,
        )
