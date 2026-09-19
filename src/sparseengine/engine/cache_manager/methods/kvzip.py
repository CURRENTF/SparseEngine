from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from sparseengine.engine.cache_manager.base import ExplicitKVPayload
from sparseengine.kernels.triton.prefill_score import prefill_score_fwd

from .snapkv import SnapKVCacheManager


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
    visited_layers: set[int] = field(default_factory=set)


class KVzipCacheManager(SnapKVCacheManager):
    """Token-shared reconstruction compression using independent layer slots."""

    def __init__(self, config, parallel_context, *, allocation_budget_bytes=None):
        self.reconstruction_prompt_ids: tuple[int, ...] = ()
        self._reconstruction: ReconstructionChunk | None = None
        self._reconstruction_source: dict[int, list[int]] = {}
        super().__init__(config, parallel_context,
                         allocation_budget_bytes=allocation_budget_bytes)
        if self.reconstruction_slot_reserve and getattr(config, "startup_cache_phase", "") != "profiling":
            padded_rows, blocks, heads = self._score_workspace_shape()
            self._prefill_score_workspace.probability_lse_buffers(
                group_count=padded_rows, candidate_blocks=blocks, block_rows=1,
                device=self.device,
            )
            self._prefill_score_workspace.probability_head_score_buffer(
                batch_size=1, query_heads=heads, score_width=self.max_model_len,
                device=self.device,
            )
            self._prefill_step_score_buffer(
                batch_size=1, max_context_len=self.max_model_len, device=self.device,
            )

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
        workspace += 4 * heads * length + 64 * self.num_kv_layers * length
        if workspace >= available:
            raise ValueError("Insufficient KV allocation budget for KVzip scoring workspace.")
        return available - workspace, slot_bytes

    @property
    def reconstruction_slot_reserve(self) -> int:
        return kvzip_reconstruction_slot_reserve(self.config)

    @property
    def num_free_slots(self) -> int:
        # A single shared scratch reserve survives admission, partial prefill
        # and decode reservations. Auxiliary allocations use the raw free lists.
        return max(0, super().num_free_slots - self.reconstruction_slot_reserve)

    def decode_window_budgets(self):
        return {name: max(0, count - self.reconstruction_slot_reserve)
                for name, count in super().decode_window_budgets().items()}

    def set_reconstruction_prompt(self, token_ids: list[int]) -> None:
        if not token_ids:
            raise ValueError("KVzip reconstruction prompt must tokenize to non-empty IDs.")
        peak = (len(token_ids) + self.config.kvzip_score_chunk_size
                + self.config.kvzip_prev_postfix_size)
        if peak > self.config.engine_prefill_chunk_size:
            raise ValueError(
                "KVzip reconstruction prompt + kvzip_score_chunk_size + "
                "kvzip_prev_postfix_size must fit engine_prefill_chunk_size: "
                f"replay={peak} capacity={self.config.engine_prefill_chunk_size}."
            )
        self.reconstruction_prompt_ids = tuple(int(x) for x in token_ids)

    def prompt_admission_costs(self, seq):
        length = int(seq.num_prompt_tokens)
        if length > self.config.kvzip_token_budget:
            if not self.reconstruction_prompt_ids:
                raise RuntimeError("KVzip reconstruction prompt has not been initialized.")
            replay = (len(self.reconstruction_prompt_ids)
                      + min(length, self.config.kvzip_score_chunk_size)
                      + min(max(0, length - self.config.kvzip_score_chunk_size),
                            self.config.kvzip_prev_postfix_size))
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
                if seq.num_prompt_tokens <= self.config.kvzip_token_budget:
                    continue
                start, count = int(seq.num_prefilled_tokens), int(seq.current_chunk_size)
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

    def reconstruction_source(self, seq) -> list[int]:
        source = self._reconstruction_source.get(int(seq.seq_id))
        if source is None or len(source) != int(seq.num_prompt_tokens):
            raise RuntimeError("KVzip reconstruction requires the complete original prompt.")
        return source

    @contextmanager
    def reconstruction_chunk(self, seq, replay_len: int, scores: torch.Tensor):
        if self._reconstruction is not None:
            raise RuntimeError("KVzip reconstruction forwards must run serially.")
        length = int(seq.num_prompt_tokens)
        if length + replay_len > self.max_model_len:
            raise ValueError("KVzip reconstruction exceeds max_model_len.")
        state = ReconstructionChunk(
            int(seq.seq_id), length, replay_len, scores,
            torch.tensor([length], dtype=torch.int32, device=self.device),
            torch.tensor([length + replay_len], dtype=torch.int32, device=self.device),
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

    def prefill_score_request(self, layer_idx, seqs):
        # This method owns scoring after ordinary attention. No extra output or
        # alternate provider is required from the main attention operation.
        return None

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
        prefill_score_fwd(
            q, view.payload.k_cache, score, view.meta.req_indices, b_start_loc,
            view.meta.context_lens, state.query_start, state.replay_len,
            view.meta.active_slots, state.query_start, state.query_end,
            candidate_start=0, recent_keep_tokens=state.replay_len,
            score_mode="probability", workspace=self._prefill_score_workspace,
        )
        torch.maximum(state.scores, score[0, :state.prefix_len], out=state.scores)
        state.visited_layers.add(int(layer_idx))

    def _prefill_score_dtype(self):
        return torch.float32

    def compact_reconstruction(self, seq, keep: torch.Tensor) -> None:
        self._uniform_decode_metadata = False
        for layer in self.kv_transformer_layer_indices():
            self.free_part_slots(layer, seq, keep, keep_indices_sorted=True)
        self._reconstruction_source.pop(int(seq.seq_id), None)

    def free_seq(self, seq_id):
        self._reconstruction_source.pop(int(seq_id), None)
        return super().free_seq(seq_id)
