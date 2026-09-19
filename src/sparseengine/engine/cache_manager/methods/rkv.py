from __future__ import annotations

import torch

from sparseengine.config import Config
from sparseengine.distributed import ParallelContext
from sparseengine.engine.sequence import Sequence
from sparseengine.operators.rkv_similarity import prepare_rkv_similarity_provider
from .rkv_scoring import rkv_head_scores, rkv_score_tiles

from .snapkv import SnapKVCacheManager


class RKVCacheManager(SnapKVCacheManager):
    """SnapKV-style physical cache with R-KV decode-time joint eviction scoring."""

    def __init__(
        self,
        config: Config,
        parallel_context: ParallelContext,
        *,
        allocation_budget_bytes: int | None = None,
    ):
        self._rkv_query_cache_enabled = self._query_cache_needed_for_config(config)
        super().__init__(
            config,
            parallel_context,
            allocation_budget_bytes=allocation_budget_bytes,
        )
        self._rkv_observation_tokens = int(config.rkv_observation_tokens)
        self._rkv_query_cache = []
        self._rkv_query_positions = []
        self._rkv_similarity_provider = (
            prepare_rkv_similarity_provider(self._rkv_query_cache_dtype(), device=self.device)
            if self._rkv_query_cache_enabled else None
        )
        if self._rkv_query_cache_enabled:
            kv_layer_set = set(self.kv_transformer_layer_indices())
            self._rkv_query_cache = [
                (
                    torch.empty(
                        (
                            self.max_buffer_rows,
                            self._rkv_observation_tokens,
                            self._rkv_num_query_heads(),
                            self.head_dim,
                        ),
                        dtype=self._rkv_query_cache_dtype(),
                        device=self.device,
                    )
                    if layer_idx in kv_layer_set
                    else None
                )
                for layer_idx in range(self.num_layers)
            ]
            self._rkv_query_positions = [
                (
                    torch.full(
                        (self.max_buffer_rows, self._rkv_observation_tokens),
                        -1,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    if layer_idx in kv_layer_set
                    else None
                )
                for layer_idx in range(self.num_layers)
            ]

    @staticmethod
    def _query_cache_needed_for_config(config: Config) -> bool:
        obs = int(getattr(config, "rkv_observation_tokens", 0) or 0)
        if obs <= 0:
            return False
        budget = (
            int(getattr(config, "sink_keep_tokens", 0) or 0)
            + int(getattr(config, "decode_keep_tokens", 0) or 0)
            + int(getattr(config, "recent_keep_tokens", 0) or 0)
        )
        trigger_len = budget + int(getattr(config, "rkv_compression_interval", 0) or 0)
        return int(getattr(config, "max_model_len", 0) or 0) >= trigger_len

    def _is_rkv_query_cache_enabled(self) -> bool:
        return bool(getattr(self, "_rkv_query_cache_enabled", True))

    def _rkv_query_cache_dtype(self) -> torch.dtype:
        dtype = self.hf_config.dtype
        return dtype if isinstance(dtype, torch.dtype) else torch.float16

    def _rkv_num_query_heads(self) -> int:
        return int(self.hf_config.num_attention_heads) // int(self.tp_size)

    def _rkv_query_cache_bytes(self) -> int:
        if not self._is_rkv_query_cache_enabled():
            return 0
        obs = int(getattr(self.config, "rkv_observation_tokens", 0) or 0)
        if obs <= 0:
            return 0
        dtype_size = torch.tensor([], dtype=self._rkv_query_cache_dtype()).element_size()
        num_query_cache_layers = int(getattr(self, "num_kv_layers", self.num_layers))
        query_elems = (
            num_query_cache_layers
            * int(self.max_buffer_rows)
            * obs
            * self._rkv_num_query_heads()
            * int(self.head_dim)
        )
        position_elems = num_query_cache_layers * int(self.max_buffer_rows) * obs
        position_dtype_size = torch.tensor([], dtype=torch.int32).element_size()
        return int(query_elems * dtype_size + position_elems * position_dtype_size)

    def _get_available_slots_info(self) -> tuple[int, int]:
        scoring_enabled = (self._is_rkv_query_cache_enabled()
                           and getattr(self.config, "startup_cache_phase", "") != "profiling")
        if scoring_enabled:
            # A long prompt is still uncompressed at the first decode eviction.
            # Reserving the configured bytes alone does not prove it can score
            # that domain. Runtime scores one request at a time, not C requests.
            rkv_score_tiles(
                batch=1, heads=self.num_kv_heads, length=int(self.config.max_model_len),
                dim=self.head_dim, groups=self._rkv_num_query_heads() // self.num_kv_heads,
                window=int(self.config.rkv_observation_tokens),
                element_size=torch.tensor([], dtype=self._rkv_query_cache_dtype()).element_size(),
                workspace_bytes=int(self.config.rkv_score_chunk_mb) * 1024**2,
            )
        available_memory, slot_bytes_per_layer = super()._get_available_slots_info()
        query_cache_bytes = self._rkv_query_cache_bytes()
        if scoring_enabled:
            # Startup profiling builds a small prefill-only cache. Scoring is
            # post-decode work and its reserve belongs to the serving pool.
            query_cache_bytes += int(self.config.rkv_score_chunk_mb) * 1024**2
        if query_cache_bytes >= available_memory:
            raise RuntimeError(
                "Not enough GPU memory for R-KV query cache and scoring workspace. "
                f"query_cache={query_cache_bytes / 1024**3:.2f}GiB "
                f"available={available_memory / 1024**3:.2f}GiB. "
                "Reduce rkv_observation_tokens, rkv_score_chunk_mb or max_num_seqs_in_batch."
            )
        return int(available_memory - query_cache_bytes), int(slot_bytes_per_layer)

    def _rkv_layer_query_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        layer_idx = int(layer_idx)
        self.kv_layer_index(layer_idx)
        cache = self._rkv_query_cache[layer_idx]
        positions = self._rkv_query_positions[layer_idx]
        if cache is None or positions is None:
            raise RuntimeError(f"R-KV query cache is not allocated for layer={layer_idx}.")
        return cache, positions

    def _clear_rkv_query_cache_row(self, layer_idx: int, row_idx: int):
        if not self._is_rkv_query_cache_enabled():
            return
        _, positions = self._rkv_layer_query_cache(layer_idx)
        positions[int(row_idx)].fill_(-1)

    def _clear_rkv_query_cache_rows(self, layer_idx: int, row_indices: list[int | None]):
        if not self._is_rkv_query_cache_enabled():
            return
        rows = [int(row_idx) for row_idx in row_indices if row_idx is not None]
        if not rows:
            return
        _, positions = self._rkv_layer_query_cache(layer_idx)
        rows_tensor = torch.tensor(rows, dtype=torch.long, device=self.device)
        positions.index_fill_(0, rows_tensor, -1)

    def decode_window_costs(self, seq: Sequence, tokens: int) -> dict[str, int]:
        if seq.is_recompute_replay:
            # Recompute is not genuine decode and cannot release slots at a
            # compression boundary. Reserve its entire requested decode window.
            return {f"layer_{layer}": int(tokens) for layer in self.kv_transformer_layer_indices()}
        return super().decode_window_costs(seq, tokens)

    def snapshot_chain_method_state(self, seq_id: int):
        state = super().snapshot_chain_method_state(seq_id)
        if self._is_rkv_query_cache_enabled():
            for layer in self.kv_transformer_layer_indices():
                row = self.seq_id_to_row[layer][seq_id]
                cache, positions = self._rkv_layer_query_cache(layer)
                state.tensors[f"queries/{layer}"] = cache[row]
                state.tensors[f"positions/{layer}"] = positions[row]
        return state

    def restore_chain_method_state(self, seq_id: int, state) -> None:
        if self._is_rkv_query_cache_enabled():
            for layer in self.kv_transformer_layer_indices():
                row = self.seq_id_to_row[layer][seq_id]
                cache, positions = self._rkv_layer_query_cache(layer)
                cache[row].copy_(state.tensors[f"queries/{layer}"], non_blocking=True)
                positions[row].copy_(state.tensors[f"positions/{layer}"], non_blocking=True)

    def free_seq(self, seq_id: int):
        row_by_layer = [
            self.seq_id_to_row[layer_idx].get(int(seq_id))
            for layer_idx in self.kv_transformer_layer_indices()
        ]
        for layer_idx, row_idx in zip(self.kv_transformer_layer_indices(), row_by_layer):
            if row_idx is not None:
                self._clear_rkv_query_cache_row(layer_idx, row_idx)
        return super().free_seq(seq_id)

    def free_part_slots(
        self,
        layer_idx: int,
        seq: Sequence,
        keep_indices: torch.Tensor,
        *,
        keep_indices_sorted: bool = False,
    ):
        self.kv_layer_index(layer_idx)
        row_idx = self.seq_id_to_row[int(layer_idx)].get(seq.seq_id)
        result = super().free_part_slots(
            layer_idx,
            seq,
            keep_indices,
            keep_indices_sorted=keep_indices_sorted,
        )
        if row_idx is not None:
            self._clear_rkv_query_cache_row(layer_idx, row_idx)
        return result

    def free_part_slots_batch(
        self,
        layer_idx: int,
        seqs: list[Sequence],
        keep_indices: torch.Tensor,
        *,
        keep_indices_sorted: bool = False,
    ):
        self.kv_layer_index(layer_idx)
        row_indices = [
            self.seq_id_to_row[int(layer_idx)].get(seq.seq_id)
            for seq in seqs
        ]
        result = super().free_part_slots_batch(
            layer_idx,
            seqs,
            keep_indices,
            keep_indices_sorted=keep_indices_sorted,
        )
        self._clear_rkv_query_cache_rows(layer_idx, row_indices)
        return result

    def free_part_slots_batch_layers(
        self,
        layer_indices: list[int],
        seqs: list[Sequence],
        keep_indices: torch.Tensor,
        *,
        keep_indices_sorted: bool = False,
    ):
        for layer_idx in layer_indices:
            self.kv_layer_index(int(layer_idx))
        row_indices_by_layer = [
            [
                self.seq_id_to_row[int(layer_idx)].get(seq.seq_id)
                for seq in seqs
            ]
            for layer_idx in layer_indices
        ]
        result = super().free_part_slots_batch_layers(
            layer_indices,
            seqs,
            keep_indices,
            keep_indices_sorted=keep_indices_sorted,
        )
        for layer_idx, row_indices in zip(layer_indices, row_indices_by_layer):
            self._clear_rkv_query_cache_rows(int(layer_idx), row_indices)
        return result

    def decode_graph_keepalive_tensors(self) -> list[torch.Tensor]:
        if not self._is_rkv_query_cache_enabled():
            return super().decode_graph_keepalive_tensors()
        return super().decode_graph_keepalive_tensors() + list(self._rkv_query_cache) + list(self._rkv_query_positions)

    @torch.no_grad()
    def record_prefill_query(self, layer_idx, q, view, *, b_start_loc, chunk_lens):
        # A prompt/chain append is not an observation of generated reasoning.
        if self._is_rkv_query_cache_enabled():
            _, positions = self._rkv_layer_query_cache(layer_idx)
            positions.index_fill_(0, view.meta.req_indices.to(torch.long), -1)

    @torch.no_grad()
    def record_decode_query(self, layer_idx: int, q: torch.Tensor):
        if not self._is_rkv_query_cache_enabled() or q.numel() == 0:
            return
        cache, positions_cache = self._rkv_layer_query_cache(layer_idx)
        state = self.get_layer_batch_states(layer_idx)
        rows = state.req_indices.to(torch.long)
        positions = state.context_lens.to(torch.long) - 1
        cols = positions.remainder(self._rkv_observation_tokens)
        cache[rows, cols] = q
        positions_cache[rows, cols] = positions.to(torch.int32)

    def rkv_observation_ready(self, layer_idx: int, seqs, kv_len: int) -> torch.Tensor:
        _, positions = self._rkv_layer_query_cache(layer_idx)
        rows = torch.tensor([self.seq_id_to_row[layer_idx][s.seq_id] for s in seqs],
                            device=self.device, dtype=torch.long)
        expected = torch.arange(kv_len - self._rkv_observation_tokens, kv_len,
                                device=self.device)
        return (positions[rows[:, None], expected.remainder(self._rkv_observation_tokens)]
                == expected).all(dim=-1)

    def rkv_joint_scores(self, layer_idx: int, seqs, kv_len: int) -> torch.Tensor:
        cache, _ = self._rkv_layer_query_cache(layer_idx)
        rows = torch.tensor([self.seq_id_to_row[layer_idx][s.seq_id] for s in seqs],
                            device=self.device, dtype=torch.long)
        slots = self.buffer_req_to_token_slots[layer_idx][rows, :kv_len].to(torch.long)
        keys = self.materialize_attention_keys(layer_idx, slots).transpose(1, 2)
        cols = torch.arange(kv_len - self._rkv_observation_tokens, kv_len,
                            device=self.device).remainder(self._rkv_observation_tokens)
        queries = cache[rows[:, None], cols].transpose(1, 2).contiguous()
        return rkv_head_scores(
            keys, queries, window=self._rkv_observation_tokens,
            kernel_size=int(self.config.rkv_kernel_size), alpha=float(self.config.rkv_alpha),
            workspace_bytes=int(self.config.rkv_score_chunk_mb) * 1024**2,
            similarity_provider=getattr(self, "_rkv_similarity_provider", None),
        ).mean(dim=1)
