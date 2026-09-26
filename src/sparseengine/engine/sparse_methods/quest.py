"""QuEST's per-decode query-aware page selection policy."""

from __future__ import annotations

import torch

from sparseengine.engine.cache_manager import MlaLatentSelectionQuery, PageSelectionPlan
from sparseengine.engine.cache_manager.methods.quest import QuestCacheManager
from sparseengine.engine.cache_manager.storage import MlaLatentStorage
from sparseengine.kernels.triton.quest_decode_view import fuse_mla_quest_selection_query
from sparseengine.operators.quest_scoring import (
    QuestPageScoreSpec,
    resolve_quest_page_score_provider,
    score_quest_pages_batched,
)

from .base import DecodeSelectionRequest
from .passthrough import PassThroughRuntime


class QuestRuntime(PassThroughRuntime):
    """Own the logical decision; borrow page metadata from the cache manager."""

    def __init__(self, config, cache_manager):
        super().__init__(config, cache_manager)
        if not isinstance(cache_manager, QuestCacheManager):
            raise TypeError("QuestRuntime requires QuestCacheManager.")
        self.quest_cache = cache_manager
        self.quest_page_scorer = None
        if cache_manager.platform.is_cuda_alike():
            self.quest_page_scorer = resolve_quest_page_score_provider(
                QuestPageScoreSpec(
                    dtype=cache_manager.hf_config.dtype,
                    query_heads=(
                        1 if isinstance(cache_manager.attention_cache_storage, MlaLatentStorage)
                        else int(cache_manager.hf_config.num_attention_heads) // cache_manager.tp_size
                    ),
                    kv_heads=cache_manager.metadata_num_heads,
                    head_dim=cache_manager.metadata_head_dim,
                    cuda_graph=bool(config.decode_graph),
                ),
                device_index=cache_manager.device.index or 0,
            )

    def _score_query(self, request: DecodeSelectionRequest) -> torch.Tensor:
        cache = self.quest_cache
        if isinstance(cache.attention_cache_storage, MlaLatentStorage):
            q = request.selection_query
            if not isinstance(q, MlaLatentSelectionQuery):
                raise TypeError("MLA QuEST requires an absorbed latent and RoPE selection query.")
            query = (
                fuse_mla_quest_selection_query(q.latent, q.rope)
                if q.latent.is_cuda else q.fused().mean(dim=1, keepdim=True)
            )
        else:
            query = request.query
            if not isinstance(query, torch.Tensor):
                raise TypeError("Explicit-KV QuEST requires a tensor selection query.")
        if query.ndim != 3 or int(query.shape[-1]) != cache.metadata_head_dim:
            raise ValueError(
                "QuEST selection query does not match page metadata: "
                f"query={tuple(query.shape)} metadata_dim={cache.metadata_head_dim}."
            )
        return query

    @torch.no_grad()
    def build_decode_selection(self, request: DecodeSelectionRequest):
        selection = self._full_selection(request.layer_idx)
        cache = self.quest_cache
        max_context_len = selection.max_context_len
        if max_context_len is None:
            raise RuntimeError("QuEST decode requires a pinned max_context_len.")
        token_budget = int(self.config.quest_token_budget)
        page_budget_base = max(3, token_budget // cache.page_size)
        max_keep = max(token_budget, page_budget_base * cache.page_size, cache.page_size)
        if (request.layer_idx < self.config.quest_skip_layers
                or token_budget <= 0 or int(max_context_len) <= max_keep):
            return selection

        max_pages = min(
            cache.max_pages_per_row,
            (int(max_context_len) + cache.page_size - 1) // cache.page_size,
        )
        prev_budget = min(page_budget_base - 1, max_pages - 1)
        if prev_budget <= 0:
            return selection
        score_query = self._score_query(request)
        row_page_slots, num_pages, previous_page_counts = (
            cache.quest_decode_page_inputs(
                selection.req_indices,
                selection.context_lens,
                max_pages=max_pages,
            )
        )
        kv_idx = cache.kv_layer_index(request.layer_idx)
        if self.quest_page_scorer is not None:
            scores = self.quest_page_scorer.score(
                score_query.contiguous(),
                cache.metadata_cache[0, kv_idx],
                cache.metadata_cache[1, kv_idx],
                row_page_slots,
            )
        else:
            safe_slots = row_page_slots.to(torch.long).clamp_min_(0)
            shape = (
                int(score_query.shape[0]), max_pages,
                cache.metadata_num_heads, cache.metadata_head_dim,
            )
            page_max = cache.metadata_cache[0, kv_idx].index_select(
                0, safe_slots.reshape(-1)
            ).view(shape).permute(0, 2, 1, 3)
            page_min = cache.metadata_cache[1, kv_idx].index_select(
                0, safe_slots.reshape(-1)
            ).view(shape).permute(0, 2, 1, 3)
            scores = score_quest_pages_batched(
                score_query, page_max, page_min, cache.metadata_num_heads,
            )
        selection.page_selection = PageSelectionPlan(
            scores=scores.contiguous(),
            row_page_slots=row_page_slots,
            num_pages=num_pages,
            previous_page_counts=previous_page_counts,
            previous_page_budget=prev_budget,
            token_budget=token_budget,
            max_keep_tokens=max_keep,
        )
        return selection
