from __future__ import annotations

from dataclasses import dataclass

import torch

from sparseengine.engine.cache_manager.base import SparseSelection
from sparseengine.operators.compressed_index import CompressedIndexOpSpec, resolve_compressed_index_provider
from .base import SparseMethodRuntime


@dataclass(frozen=True)
class SharedKVSelectionQuery:
    """Semantic learned index query plus the cache owner's physical candidates."""
    candidates: object
    index_query: torch.Tensor | None = None
    index_head_weights: torch.Tensor | None = None


class DeepSeekV4Runtime(SparseMethodRuntime):
    """Causal window and compressed-history selection for native V4 attention."""

    def __init__(self, config, cache_manager):
        super().__init__(config, cache_manager)
        self.ratios = cache_manager.ratios
        self.window_size = int(config.hf_config.sliding_window)
        self.index_topk = int(config.hf_config.index_topk)
        self.index_provider = resolve_compressed_index_provider(
            CompressedIndexOpSpec(64, 128, 64, self.index_topk, cache_manager.page_size,
                                  bool(config.decode_graph)), device_index=cache_manager.device.index or 0,
        ) if 4 in self.ratios else None

    def needs_attention_score(self, layer_idx, step):
        return False

    def _select(self, layer_idx, selection_query):
        if not isinstance(selection_query, SharedKVSelectionQuery):
            raise TypeError("Native shared KV requires a typed selection query")
        candidates = selection_query.candidates
        ratio = self.ratios[layer_idx]
        positions = candidates.query_positions.to(torch.int32)
        window = torch.where(candidates.window_logical_positions <= positions[:, None],
                             candidates.window_slots, -1)
        compressed_length = (positions+1)//ratio if ratio else torch.zeros_like(positions)
        if ratio == 4:
            if selection_query.index_query is None or selection_query.index_head_weights is None:
                raise ValueError("Ratio-4 attention requires learned index query and head weights")
            width = candidates.index_slots.shape[1]
            if width <= self.index_topk:
                column = torch.arange(width, device=self.device)
                selected = torch.where(column[None, :] < compressed_length[:, None],
                                       candidates.compressed_slots, -1)
                compressed = torch.nn.functional.pad(selected, (0, self.index_topk-width), value=-1)
            else:
                scores, selected = self.cache_manager.index_selection_workspace(len(positions), width)
                self.index_provider.score(selection_query.index_query, selection_query.index_head_weights,
                                          candidates.index_pages, candidates.index_slots.contiguous(),
                                          compressed_length, out=scores)
                self.index_provider.select(scores, candidates.compressed_slots.contiguous(),
                                           compressed_length, out=selected)
                compressed = selected
            compressed_length = compressed_length.clamp_max(self.index_topk)
        elif ratio:
            column = torch.arange(candidates.compressed_slots.shape[1], device=self.device)
            compressed = torch.where(column[None, :] < compressed_length[:, None],
                                     candidates.compressed_slots, -1)
        else:
            compressed = candidates.compressed_slots
        indices = torch.cat((window, compressed), dim=1)
        capacity = ((indices.shape[1]+63)//64)*64
        if capacity != indices.shape[1]:
            indices = torch.nn.functional.pad(indices, (0, capacity-indices.shape[1]), value=-1)
        # The window occupies a fixed span; -1 entries inside that span are masked
        # by sparse FlashMLA. Length is the occupied span, not a count of live keys.
        lengths = (self.window_size+compressed_length).to(torch.int32)
        return SparseSelection(kind="shared_kv", req_indices=candidates.query_rows,
                               context_lens=lengths, max_context_len=capacity,
                               active_slots=indices[:, None].contiguous())

    def build_prefill_selection(self, request):
        return self._select(request.layer_idx, request.selection_query)

    def build_decode_selection(self, request):
        return self._select(request.layer_idx, request.selection_query)
