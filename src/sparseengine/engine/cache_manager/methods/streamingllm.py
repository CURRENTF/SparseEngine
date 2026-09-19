from __future__ import annotations

from sparseengine.config import Config
from sparseengine.distributed import ParallelContext
from sparseengine.engine.sequence import Sequence

from .snapkv import SnapKVCacheManager


class StreamingLLMCacheManager(SnapKVCacheManager):
    """Attention-sink / StreamingLLM cache manager.

    Reuse the standard per-layer physical slot bookkeeping from SnapKV, but keep
    scheduling headroom aligned with the fixed recent-window policy instead of
    score-based top-k eviction.
    """

    def __init__(
        self,
        config: Config,
        parallel_context: ParallelContext,
        *,
        allocation_budget_bytes: int | None = None,
    ):
        super().__init__(
            config,
            parallel_context,
            allocation_budget_bytes=allocation_budget_bytes,
        )
        # StreamingLLM applies the same prefix/recent window on every layer, so
        # decode metadata stays layer-uniform even after compaction.
        self._uniform_decode_metadata = True

    def prefill_batched_tokens_margin(self) -> int:
        return int(self.config.recent_keep_tokens)

    def decode_window_costs(self, seq: Sequence, tokens: int) -> dict[str, int]:
        tokens = max(0, int(tokens))
        budget = int(self.config.sink_keep_tokens) + int(self.config.recent_keep_tokens)
        costs = {}
        for layer, resident in zip(
            self.kv_transformer_layer_indices(),
            self.chain_physical_residency(seq.seq_id),
        ):
            # Decode appends before compacting at twice the retained budget.
            # An already oversized row still needs one writable append slot.
            costs[f"layer_{layer}"] = (
                min(tokens, max(1, 2 * budget - int(resident)))
                if budget > 0 else tokens
            )
        return costs

    def _decode_graph_metadata_always_uniform(self) -> bool:
        return True

    def remaining_prefill_tokens(self, seq: Sequence) -> int:
        remaining = int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
        recent = int(self.config.recent_keep_tokens)
        if recent > 0 and remaining > recent:
            return remaining - recent
        return remaining

    def free_prefix_recent_slots_batch_layers(
        self,
        layer_indices: list[int],
        seqs: list[Sequence],
        *,
        kv_len: int,
        sink_keep_tokens: int,
        recent_keep_tokens: int,
    ):
        super().free_prefix_recent_slots_batch_layers(
            layer_indices,
            seqs,
            kv_len=kv_len,
            sink_keep_tokens=sink_keep_tokens,
            recent_keep_tokens=recent_keep_tokens,
        )
        if layer_indices and len(layer_indices) == self.num_layers:
            self._uniform_decode_metadata = True
