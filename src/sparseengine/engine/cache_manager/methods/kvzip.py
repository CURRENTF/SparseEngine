from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from sparseengine.engine.cache_manager.base import ExplicitKVPayload
from sparseengine.kernels.triton.prefill_score import prefill_score_fwd

from .snapkv import SnapKVCacheManager
from ..head_slots import HeadSlotCacheManager
from sparseengine.kernels.triton.prefill_score import PrefillScoreWorkspace


def kvzip_reconstruction_slot_reserve(config) -> int:
    if config.max_model_len <= config.kvzip_token_budget:
        return 0
    return int(config.engine_prefill_chunk_size)


@dataclass
class ReconstructionChunk:
    seq_id: int
    prefix_len: int
    replay_len: int
    scores: torch.Tensor
    query_start: torch.Tensor
    query_end: torch.Tensor
    sink_end: int
    chunk_start: int
    chunk_end: int
    visited_layers: set[int] = field(default_factory=set)


class KVzipReconstructionMixin:
    """Common replay lifecycle for shared-token and independent-head storage."""

    def __init__(self, config, parallel_context, *, allocation_budget_bytes=None):
        self.reconstruction_prompt_ids: dict[str, tuple[int, ...]] = {}
        self._reconstruction: ReconstructionChunk | None = None
        self._reconstruction_source: dict[int, list[int]] = {}
        self.context_compaction_count = 0
        self._prefill_score_workspace = PrefillScoreWorkspace()
        super().__init__(config, parallel_context, allocation_budget_bytes=allocation_budget_bytes)

    def debug_state_summary(self):
        summary = super().debug_state_summary()
        summary["kvzip_context_compactions"] = self.context_compaction_count
        return summary

    def _score_workspace_shape(self) -> tuple[int, int, int]:
        heads = int(self.hf_config.num_attention_heads) // self.tp_size
        groups = heads // self.num_kv_heads
        padded_heads = self.num_kv_heads * (1 << (groups - 1).bit_length())
        queries = int(self.config.engine_prefill_chunk_size)
        padded_queries = ((queries + 31) // 32 + 4) * 32
        block_n = 64 if self.head_dim >= 128 else 128
        blocks = (int(self.config.max_model_len) + block_n - 1) // block_n
        return padded_heads * padded_queries, blocks, heads

    def _get_available_slots_info(self):
        available, slot_bytes = super()._get_available_slots_info()
        if (self.config.max_model_len <= self.config.kvzip_token_budget
                or getattr(self.config, "startup_cache_phase", "") == "profiling"):
            return available, slot_bytes
        # Bound the existing tiled scorer's partial/global LSE, head reduction,
        # aggregate scores and stable-sort/compaction intermediates. One replay
        # request and one layer score are live at a time.
        padded_rows, blocks, heads = self._score_workspace_shape()
        length = int(self.config.max_model_len)
        workspace = 4 * padded_rows * (blocks + 1)
        # Global thresholding gathers all TP head shards on the leader. Bound
        # local scores, gathered populations, sort values/indices and selections.
        workspace += 64 * self.num_kv_layers * int(self.hf_config.num_key_value_heads) * length
        if workspace >= available:
            raise ValueError("Insufficient KV allocation budget for KVzip scoring workspace.")
        return available - workspace, slot_bytes

    @property
    def reconstruction_slot_reserve(self) -> int:
        return kvzip_reconstruction_slot_reserve(self.config)

    def set_reconstruction_prompt(self, token_ids: dict[str, list[int]]) -> None:
        if set(token_ids) != {"first", "continuation", "suffix"} or not all(token_ids.values()):
            raise ValueError("KVzip needs nonempty first/continuation/assistant-suffix token IDs.")
        peak = (max(len(token_ids["first"]), len(token_ids["continuation"]))
                + len(token_ids["suffix"]) + self.config.kvzip_score_chunk_size
                + self.config.kvzip_prev_postfix_size)
        if peak > self.config.engine_prefill_chunk_size:
            raise ValueError(
                "KVzip reconstruction prompt + kvzip_score_chunk_size + "
                "kvzip_prev_postfix_size must fit engine_prefill_chunk_size: "
                f"replay={peak} capacity={self.config.engine_prefill_chunk_size}."
            )
        self.reconstruction_prompt_ids = {name: tuple(int(x) for x in ids)
                                          for name, ids in token_ids.items()}

    def prompt_admission_costs(self, seq):
        length = seq.context_length
        if length is None:
            if (seq.num_prompt_tokens > self.config.kvzip_token_budget
                    and getattr(self.config, "startup_cache_phase", "") != "profiling"):
                raise ValueError("Compressible KVzip requests require context_length and "
                                 "context_prefix_length; question KV must be processed after eviction.")
            return super().prompt_admission_costs(seq)
        context_size = int(length) - int(seq.context_prefix_length)
        if context_size > self.config.kvzip_token_budget:
            if seq.context_prefix_length <= 0:
                raise ValueError("KVzip requires a nonempty protected system prefix via context_prefix_length.")
            if not self.reconstruction_prompt_ids:
                raise RuntimeError("KVzip reconstruction prompt has not been initialized.")
            template = self.reconstruction_prompt_ids
            chunk = self.config.kvzip_score_chunk_size
            replay = len(template["suffix"]) + len(template["first"]) + min(context_size, chunk)
            if context_size > chunk:
                # The largest continuation is the second chunk. Later chunks
                # have the same instruction/postfix and no larger source span.
                replay = max(replay, len(template["suffix"]) + len(template["continuation"])
                             + min(chunk, self.config.kvzip_prev_postfix_size)
                             + min(context_size - chunk, chunk))
            if length + replay > self.max_model_len:
                raise ValueError(
                    "KVzip prompt plus reconstruction replay exceeds max_model_len: "
                    f"prompt={length} replay={replay} capacity={self.max_model_len}. "
                    "Shorten the prompt or reduce kvzip_score_chunk_size."
                )
        return super().prompt_admission_costs(seq)

    def prepare_step(self, seqs, is_prefill):
        if is_prefill and self._reconstruction is None:
            for seq in seqs:
                if (seq.context_length is None
                        or seq.context_length - seq.context_prefix_length <= self.config.kvzip_token_budget
                        or seq.num_prefilled_tokens >= seq.context_length):
                    continue
                start, count = int(seq.num_prefilled_tokens), int(seq.current_chunk_size)
                if start + count > seq.context_length:
                    raise RuntimeError("KVzip context prefill crossed the question boundary.")
                # TP followers receive only this chunk, while the leader keeps
                # the complete business token history. Persist the same source.
                tokens = (seq.token_ids[start:start + count]
                          if len(seq.token_ids) > count else seq.token_ids)
                if start == 0:
                    self._reconstruction_source[int(seq.seq_id)] = []
                source = self._reconstruction_source.get(int(seq.seq_id))
                if source is None or len(source) != start or len(tokens) != count:
                    raise RuntimeError("KVzip prompt chunks must arrive in logical order.")
                source.extend(int(token) for token in tokens)
        return super().prepare_step(seqs, is_prefill)

    def _prefill_uses_retained_rows(self, seq) -> bool:
        return (self._reconstruction is None and seq.context_length is not None
                and seq.num_prefilled_tokens >= seq.context_length
                and seq.context_length - seq.context_prefix_length > self.config.kvzip_token_budget)

    def reconstruction_source(self, seq) -> list[int]:
        source = self._reconstruction_source.get(int(seq.seq_id))
        if source is None or len(source) != int(seq.context_length):
            raise RuntimeError("KVzip reconstruction requires the complete original context.")
        return source

    @contextmanager
    def reconstruction_chunk(self, seq, replay_len: int, scores: torch.Tensor, *,
                             chunk_start: int, chunk_end: int):
        if self._reconstruction is not None:
            raise RuntimeError("KVzip reconstruction forwards must run serially.")
        length = int(seq.context_length)
        if not 0 < seq.context_prefix_length <= chunk_start < chunk_end <= length:
            raise ValueError("KVzip score chunk must lie inside the unprotected context.")
        if tuple(scores.shape) != (self.num_kv_layers, self.num_kv_heads, length):
            raise ValueError("KVzip score accumulator must retain every layer and KV head independently.")
        if length + replay_len > self.max_model_len:
            raise ValueError("KVzip reconstruction exceeds max_model_len.")
        state = ReconstructionChunk(
            int(seq.seq_id), length, replay_len, scores,
            torch.tensor([length], dtype=torch.int32, device=self.device),
            torch.tensor([length + replay_len], dtype=torch.int32, device=self.device),
            int(seq.context_prefix_length), int(chunk_start), int(chunk_end),
        )
        self._reconstruction = state
        try:
            yield
            if state.visited_layers != set(self.kv_transformer_layer_indices()):
                raise RuntimeError("KVzip reconstruction did not score every KV layer.")
        finally:
            try:
                self._release_reconstruction_suffix(seq, length)
            finally:
                self._reconstruction = None

    def prefill_score_request(self, layer_idx, seqs):
        # This method owns scoring after ordinary attention. No extra output or
        # alternate provider is required from the main attention operation.
        return None

    def _prefill_score_dtype(self):
        return torch.float32

    def free_seq(self, seq_id):
        self._reconstruction_source.pop(int(seq_id), None)
        return super().free_seq(seq_id)


class KVzipCacheManager(KVzipReconstructionMixin, SnapKVCacheManager):
    """Shared head unions with independent layer slots."""

    reconstruction_heads_shared = True

    def __init__(self, config, parallel_context, *, allocation_budget_bytes=None):
        super().__init__(config, parallel_context, allocation_budget_bytes=allocation_budget_bytes)
        if self.reconstruction_slot_reserve and getattr(config, "startup_cache_phase", "") != "profiling":
            padded_rows, blocks, heads = self._score_workspace_shape()
            self._prefill_score_workspace.probability_lse_buffers(
                group_count=padded_rows, candidate_blocks=blocks, block_rows=1, device=self.device)
            self._prefill_step_score_buffer(batch_size=1, max_context_len=self.max_model_len, device=self.device)

    @property
    def num_free_slots(self) -> int:
        # A single shared scratch reserve survives admission, partial prefill
        # and decode reservations. Auxiliary allocations use the raw free lists.
        return max(0, super().num_free_slots - self.reconstruction_slot_reserve)

    def decode_window_budgets(self):
        return {name: max(0, count - self.reconstruction_slot_reserve)
                for name, count in super().decode_window_budgets().items()}

    def _release_reconstruction_suffix(self, seq, length: int) -> None:
        for layer in self.kv_transformer_layer_indices():
            row = self.seq_id_to_row[layer].get(int(seq.seq_id))
            if row is None or int(self.row_seq_lens[layer][row]) < length:
                raise RuntimeError("KVzip reconstruction lost its original cache row.")
            end = int(self.row_seq_lens[layer][row])
            if end > length:
                slots = self.buffer_req_to_token_slots[layer][row, length:end]
                self._append_compaction_free_slots(layer, slots.clone())
                slots.zero_()
                self.row_seq_lens[layer][row] = length

    @torch.no_grad()
    def collect_prefill_attention_score(
        self, layer_idx, q, view, *, b_start_loc, chunk_lens, attention_lse=None,
    ):
        state = self._reconstruction
        if state is None:
            return
        if not isinstance(view.payload, ExplicitKVPayload):
            raise TypeError("KVzip scoring requires explicit KV.")
        if chunk_lens.numel() != 1 or q.shape[0] != state.replay_len:
            raise RuntimeError("KVzip reconstruction query batch or length mismatch.")
        if layer_idx in state.visited_layers:
            raise RuntimeError("KVzip reconstruction scored a layer more than once.")
        row = self.seq_id_to_row[layer_idx][state.seq_id]
        length = state.prefix_len + state.replay_len
        if int(self.row_seq_lens[layer_idx][row]) != length:
            raise RuntimeError("KVzip scoring requires the original dense prompt plus replay.")
        score = self._prefill_step_score_buffer(
            batch_size=1, max_context_len=length, device=q.device,
        )
        layer_scores = state.scores[self.kv_layer_index(layer_idx)]
        prefill_score_fwd(
            q, view.payload.k_cache, score, view.meta.req_indices, b_start_loc,
            view.meta.context_lens, state.query_start, state.replay_len,
            view.meta.active_slots, state.query_start, state.query_end,
            reconstruction_range=(state.sink_end, state.chunk_start, state.chunk_end),
            reconstruction_head_scores=layer_scores.unsqueeze(0),
            score_mode="probability", workspace=self._prefill_score_workspace,
        )
        state.visited_layers.add(int(layer_idx))

    def compact_reconstruction(self, seq, keep: list[torch.Tensor]) -> None:
        self._uniform_decode_metadata = False
        for layer in self.kv_transformer_layer_indices():
            self.free_part_slots(layer, seq, keep[self.kv_layer_index(layer)], keep_indices_sorted=True)
        self._reconstruction_source.pop(int(seq.seq_id), None)
        self.context_compaction_count += 1


class KVzipHeadSlotCacheManager(KVzipReconstructionMixin, HeadSlotCacheManager):
    """Original headwise threshold selections backed by independently reclaimed KV."""

    reconstruction_heads_shared = False

    @property
    def scratch_slot_reserve(self):
        # The auxiliary forward may consume its own reserved scratch. Business
        # admission/append/decode must leave that reserve available for replay.
        return 0 if self._reconstruction is not None else self.reconstruction_slot_reserve

    def _release_reconstruction_suffix(self, seq, length):
        import numpy as np

        row = self.allocator.seq_rows[int(seq.seq_id)]
        if np.any(self.allocator.lengths[:, row] < length):
            raise RuntimeError("KVzip replay lost original head-slot history.")
        keep = [[np.arange(length) for _ in range(self.num_kv_heads)] for _ in range(self.num_kv_layers)]
        self.allocator.compact(int(seq.seq_id), keep)
        self.publish_seq_metadata(int(seq.seq_id))

    @torch.no_grad()
    def collect_prefill_attention_score(self, layer_idx, q, view, *, b_start_loc,
                                        chunk_lens, attention_lse=None):
        state = self._reconstruction
        if state is None:
            return
        if not isinstance(view.payload, ExplicitKVPayload) or view.payload.backend != "head_slots":
            raise TypeError("KVzip independent-head scoring requires head-slot KV.")
        if chunk_lens.numel() != 1 or q.shape[0] != state.replay_len:
            raise RuntimeError("KVzip reconstruction query batch or length mismatch.")
        if layer_idx in state.visited_layers:
            raise RuntimeError("KVzip reconstruction scored a layer more than once.")
        layer, row = self.kv_layer_index(layer_idx), self.allocator.seq_rows[state.seq_id]
        length = state.prefix_len + state.replay_len
        if (self.allocator.lengths[layer, row] != length).any():
            raise RuntimeError("KVzip scoring requires original dense head histories plus replay.")
        from sparseengine.kernels.triton.head_slots import reorder_head_queries

        heads = self.num_kv_heads
        cu = torch.tensor([0, state.replay_len], dtype=torch.int32, device=q.device)
        packed_q = reorder_head_queries(q, cu, heads, state.replay_len)
        reqs = torch.arange(row * heads, (row + 1) * heads, dtype=torch.int32, device=q.device)
        starts = torch.arange(heads, dtype=torch.int32, device=q.device) * state.replay_len
        cached = torch.full((heads,), state.prefix_len, dtype=torch.int32, device=q.device)
        ends = cached + state.replay_len
        table = view.meta.active_slots.flatten(0, 1)
        score = torch.empty((heads, length), dtype=torch.float32, device=q.device)
        prefill_score_fwd(packed_q, view.payload.k_cache.unsqueeze(1), score, reqs, starts,
            ends, cached, state.replay_len, table, cached, ends,
            reconstruction_range=(state.sink_end, state.chunk_start, state.chunk_end),
            reconstruction_head_scores=state.scores[layer].unsqueeze(1),
            score_mode="probability", workspace=self._prefill_score_workspace)
        state.visited_layers.add(int(layer_idx))

    def compact_reconstruction(self, seq, keep):
        from ..base import HeadSlotSelection

        if not all(isinstance(selection, HeadSlotSelection) for selection in keep):
            raise TypeError("Independent-head KVzip requires typed head selections.")
        cpu = [[indices.cpu().numpy() for indices in selection.indices] for selection in keep]
        self.allocator.compact(int(seq.seq_id), cpu)
        self.publish_seq_metadata(int(seq.seq_id))
        self._reconstruction_source.pop(int(seq.seq_id), None)
        self.context_compaction_count += 1
