from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from sparseengine.engine.sequence import Sequence
from sparseengine.method_registry import (
    normalize_sparse_method,
    resolve_prefill_sparse_method,
)
from sparseengine.platforms import device_runtime
from sparseengine.utils.context import get_context
from sparseengine.utils.profiler import cpu_timing, profiler

from ..base import ExplicitKVPayload, PrefillComputeView, PrefillScoreRequest
from .snapkv import SnapKVCacheManager
from ..storage import ExplicitKVStorage


class _H2ORowRef(NamedTuple):
    """Minimal owner handle for an unscheduled resident decode row."""

    seq_id: int


class H2OCacheManager(SnapKVCacheManager):
    """H2O physical KV eviction with one score vector per layer and sequence.

    Sparse-Engine owns one physical token row shared by all KV heads, so this v1
    implementation maintains one cumulative normalized token-importance vector
    aligned with that row. The probability prefill path accumulates normalized
    attention mass. The logits path max-reduces raw QK over the observation
    queries and query heads, normalizes that token vector, and then accumulates
    it on the same per-query mass scale.
    """

    def __init__(
        self,
        config,
        parallel_context,
        *,
        allocation_budget_bytes: int | None = None,
    ):
        super().__init__(
            config,
            parallel_context,
            allocation_budget_bytes=allocation_budget_bytes,
        )
        self._h2o_scores: dict[tuple[int, int], torch.Tensor] = {}
        self._h2o_decode_score_rows: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._h2o_decode_score_metadata = None
        # Decode rows remain reclaimable while temporarily absent from a
        # scheduled batch. Keep only ids here: caching full Sequence objects
        # would retain their logical token histories on every worker.
        self._h2o_active_decode_seq_ids: set[int] = set()
        self._h2o_counters = {
            "intermediate_prefill_evictions": 0,
            "final_prefill_evictions": 0,
            # One burst is one sequence compacted by either the periodic
            # trigger or cache pressure. Row and drop counters below count
            # physical layer rows/tokens instead.
            "decode_eviction_bursts": 0,
            "decode_evictions": 0,
            "dropped_tokens": 0,
        }
        self._h2o_final_prefill_workspace: torch.Tensor | None = None
        self._h2o_decode_score_workspace: torch.Tensor | None = None
        self._h2o_decode_score_signature: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        self._h2o_decode_score_length = 0

    @property
    def h2o_decode_budget(self) -> int:
        return int(self.config.h2o_decode_budget)

    @property
    def h2o_decode_eviction_interval(self) -> int:
        return int(self.config.h2o_decode_eviction_interval)

    @property
    def h2o_prefill_budget(self) -> int:
        return int(self.config.h2o_prefill_budget)

    @property
    def h2o_prefill_enabled(self) -> bool:
        return (
            resolve_prefill_sparse_method(
                getattr(self.config, "prefill_sparse_method", None),
                sparse_method=getattr(self.config, "sparse_method", None),
            )
            == "h2o_prefill"
        )

    @property
    def h2o_decode_enabled(self) -> bool:
        return normalize_sparse_method(
            getattr(self.config, "sparse_method", None)
        ) == "h2o"

    def _prefill_append_peak(
        self,
        resident_len: int,
        remaining_tokens: int,
        engine_prefill_chunk_size: int,
    ) -> int:
        """Reserve the append-before-evict peak for a resident H2O row."""
        resident_len = int(resident_len)
        remaining_tokens = max(0, int(remaining_tokens))
        if not self.h2o_prefill_enabled:
            return resident_len + remaining_tokens
        chunk_size = max(1, int(engine_prefill_chunk_size))
        return min(
            resident_len + remaining_tokens,
            max(resident_len, self.h2o_prefill_budget) + chunk_size,
        )

    def _prompt_prefill_peak(self, seq: Sequence, engine_prefill_chunk_size: int) -> int:
        """Peak physical row size needed to make chunked prefill progress."""
        return self._prefill_append_peak(
            0,
            int(seq.num_prompt_tokens),
            engine_prefill_chunk_size,
        )

    def prompt_admission_cost(self, seq: Sequence) -> int:
        return self._prompt_prefill_peak(seq, int(self.config.engine_prefill_chunk_size))

    def prompt_admission_free_slots(self) -> int:
        return int(self.num_free_slots)

    def prompt_admission_budgets(
        self,
        waiting_seqs: deque[Sequence],
        engine_prefill_chunk_size: int,
    ) -> dict[str, int]:
        reserved = self.reserved_prefill_slots(waiting_seqs, engine_prefill_chunk_size)
        return {"slots": max(0, int(self.num_free_slots) - int(reserved))}

    def prompt_admission_costs(self, seq: Sequence) -> dict[str, int]:
        return {"slots": self.prompt_admission_cost(seq)}

    def prompt_logical_reservation_cost(self, seq: Sequence) -> int:
        return self.prompt_admission_cost(seq)

    def reserved_prefill_slots(self, waiting_seqs: deque[Sequence], engine_prefill_chunk_size: int) -> int:
        reserved = 0
        first_layer = int(self.kv_transformer_layer_indices()[0])
        for seq in waiting_seqs:
            if not 0 < int(seq.num_prefilled_tokens) < int(seq.num_prompt_tokens):
                continue
            remaining = int(seq.num_prompt_tokens) - int(seq.num_prefilled_tokens)
            physical_len = self._physical_row_len(first_layer, seq)
            peak_len = self._prefill_append_peak(
                physical_len,
                remaining,
                engine_prefill_chunk_size,
            )
            reserved += max(0, peak_len - physical_len)
        return int(reserved)

    def prefill_step_free_slots(self) -> int:
        return int(self.num_free_slots)

    def prefill_step_free_slots_for(self, seq: Sequence) -> int:
        del seq
        return int(self.num_free_slots)

    def prefill_step_reservation_cost(self, seq: Sequence, scheduled_tokens: int) -> int:
        del seq
        return int(scheduled_tokens)

    def decode_step_free_slots(self) -> int:
        return int(self.num_free_slots)

    def decode_step_free_slots_for(self, seq: Sequence) -> int:
        del seq
        return int(self.num_free_slots)

    def decode_step_reservation_cost(self, seq: Sequence) -> int:
        del seq
        return 1

    def _decode_append_peak(self, resident: int, tokens: int) -> int:
        peak = resident + tokens
        if getattr(self.config, "h2o_decode_eviction", False) and tokens:
            trigger = self.h2o_decode_budget + self.h2o_decode_eviction_interval
            peak = min(peak, max(resident + 1, trigger))
        return peak

    def decode_window_costs(self, seq: Sequence, tokens: int) -> dict[str, int]:
        # Decode only needs append headroom, not chain admission deficits/rows.
        tokens = max(0, int(tokens))
        return {
            f"layer_{layer}": self._decode_append_peak(resident, tokens) - resident
            for layer, resident in zip(
                self.kv_transformer_layer_indices(), self.chain_physical_residency(seq.seq_id),
            )
        }

    def chain_capacity_deficits(
        self,
        *,
        suffix_tokens: int,
        generation_tokens: int = 0,
        existing_slots_by_layer: tuple[int, ...] = (),
        outstanding_reserved_slots_by_layer: tuple[int, ...] = (),
        outstanding_reserved_rows: int = 0,
        needs_resident_row: bool,
    ) -> tuple[tuple[int, ...], int, tuple[int, ...], int]:
        """Reserve prefill and decode peaks under the selected retention policy."""
        suffix_tokens = max(0, int(suffix_tokens))
        generated_kv_tokens = max(0, int(generation_tokens) - 1)
        chunk_size = max(1, int(self.config.engine_prefill_chunk_size))
        layer_ids = self.kv_transformer_layer_indices()
        required_by_layer = []
        for local_layer, _layer_idx in enumerate(layer_ids):
            existing = (
                int(existing_slots_by_layer[local_layer])
                if local_layer < len(existing_slots_by_layer)
                else 0
            )
            prefill_peak = self._prefill_append_peak(
                existing,
                suffix_tokens,
                chunk_size,
            )
            # The final prompt boundary belongs to the decode method. Without
            # H2O decode, the last prefill chunk remains resident after any
            # earlier h2o_prefill compactions.
            if suffix_tokens <= 0:
                resident_after_prefill = existing
            elif self.h2o_decode_enabled:
                resident_after_prefill = min(
                    existing + suffix_tokens,
                    self.h2o_decode_budget,
                )
            else:
                resident_after_prefill = prefill_peak
            decode_peak = self._decode_append_peak(resident_after_prefill, generated_kv_tokens)
            required_by_layer.append(
                max(0, max(prefill_peak, decode_peak) - (0 if needs_resident_row else existing))
            )

        slot_deficits = tuple(
            max(
                0,
                int(required)
                - max(
                    0,
                    int(self._num_free_slots[layer_idx])
                    - (
                        int(
                            outstanding_reserved_slots_by_layer[local_layer]
                        )
                        if local_layer
                        < len(outstanding_reserved_slots_by_layer)
                        else 0
                    ),
                ),
            )
            for local_layer, (layer_idx, required) in enumerate(
                zip(layer_ids, required_by_layer)
            )
        )
        required_rows = 1 if needs_resident_row else 0
        available_rows = max(
            0,
            min(
                (len(self.free_rows[layer_idx]) for layer_idx in layer_ids),
                default=0,
            )
            - max(0, int(outstanding_reserved_rows)),
        )
        row_deficit = max(0, required_rows - available_rows)
        return (
            tuple(int(value) for value in required_by_layer),
            required_rows,
            slot_deficits,
            row_deficit,
        )

    @cpu_timing.timed
    @torch.no_grad()
    def prepare_decode_static(
        self,
        seqs: list[Sequence],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        req_indices: torch.Tensor,
    ):
        """Prepare H2O decode metadata through its uniform-row invariant.

        H2O appends and evicts one token for every active KV layer, so request
        rows, row lengths, and free-stack pointers stay aligned across layers.
        Physical slot ids may differ; gather those once from the shared
        layer-major free stack and broadcast only the common row metadata.
        """
        if self.free_slots_stack_tensor is None:
            return super().prepare_decode_static(
                seqs,
                input_ids,
                positions,
                slot_mapping,
                context_lens,
                req_indices,
            )

        with profiler.record("cache_prepare_decode"):
            real_batch_size = len(seqs)
            graph_batch_size = int(input_ids.numel())
            if real_batch_size <= 0:
                raise ValueError("Static decode requires a non-empty real decode batch.")
            if any(
                int(tensor.numel()) != graph_batch_size
                for tensor in (positions, slot_mapping, context_lens, req_indices)
            ):
                raise ValueError("Static decode input and metadata buffers must share one graph batch size.")
            if real_batch_size > graph_batch_size:
                raise ValueError(
                    "Static decode graph batch is smaller than the real decode batch: "
                    f"graph={graph_batch_size}, real={real_batch_size}."
                )

            layer_ids = tuple(int(layer_id) for layer_id in self.kv_transformer_layer_indices())
            if not layer_ids:
                raise RuntimeError("H2O static decode requires at least one KV layer.")
            first_layer = layer_ids[0]
            seq_ids = tuple(int(seq.seq_id) for seq in seqs)
            row_indices = tuple(
                int(self.seq_id_to_row[first_layer][seq_id]) for seq_id in seq_ids
            )
            if self.validate_runtime_invariants:
                first_row_lens = tuple(
                    int(self.row_seq_lens[first_layer][row_idx])
                    for row_idx in row_indices
                )
                for layer_id in layer_ids[1:]:
                    layer_rows = tuple(
                        int(self.seq_id_to_row[layer_id].get(seq_id, -1))
                        for seq_id in seq_ids
                    )
                    if layer_rows != row_indices:
                        raise RuntimeError(
                            "H2O static decode requires uniform request rows across KV layers: "
                            f"first_layer={first_layer} rows={row_indices} "
                            f"layer={layer_id} layer_rows={layer_rows}."
                        )
                    layer_row_lens = tuple(
                        int(self.row_seq_lens[layer_id][row_idx])
                        for row_idx in layer_rows
                    )
                    if layer_row_lens != first_row_lens:
                        raise RuntimeError(
                            "H2O static decode requires uniform row lengths across KV layers: "
                            f"first_layer={first_layer} lengths={first_row_lens} "
                            f"layer={layer_id} layer_lengths={layer_row_lens}."
                        )
            row_cache_key = (seq_ids, row_indices, layer_ids)
            cached_rows = getattr(self, "_h2o_decode_static_rows", None)
            if cached_rows is None or cached_rows[0] != row_cache_key:
                rows_gpu = torch.tensor(row_indices, dtype=torch.long, device=self.device)
                rows_i32_gpu = rows_gpu.to(torch.int32)
                batch_cols_gpu = torch.arange(
                    real_batch_size,
                    dtype=torch.long,
                    device=self.device,
                )
                self._h2o_decode_static_rows = (
                    row_cache_key,
                    rows_gpu,
                    rows_i32_gpu,
                    batch_cols_gpu,
                )
            else:
                _, rows_gpu, rows_i32_gpu, batch_cols_gpu = cached_rows

            topology_key = layer_ids
            cached_topology = getattr(self, "_h2o_decode_static_topology", None)
            if cached_topology is None or cached_topology[0] != topology_key:
                layers_gpu = torch.tensor(layer_ids, dtype=torch.long, device=self.device)
                kv_layers_gpu = torch.tensor(
                    [self.kv_layer_index(layer_id) for layer_id in layer_ids],
                    dtype=torch.long,
                    device=self.device,
                )
                self._h2o_decode_static_topology = (
                    topology_key,
                    layers_gpu,
                    kv_layers_gpu,
                )
            else:
                _, layers_gpu, kv_layers_gpu = cached_topology

            cur_lens = self.row_seq_lens[first_layer][list(row_indices)].copy()
            max_cur_len = int(cur_lens.max()) if cur_lens.size else 0
            if max_cur_len + 1 > int(self.max_model_len):
                raise RuntimeError(
                    "KV row length exceeds max_model_len in H2O static decode: "
                    f"max_cur_len={max_cur_len} max_model_len={int(self.max_model_len)}."
                )
            static_cap = getattr(self, "_decode_static_max_context_len", None)
            if static_cap is not None and max_cur_len + 1 > int(static_cap):
                raise RuntimeError(
                    "H2O static decode context exceeds the captured graph capacity: "
                    f"next_len={max_cur_len + 1} static_cap={int(static_cap)}."
                )

            free_ptrs = tuple(int(self._num_free_slots[layer_id]) for layer_id in layer_ids)
            if self.validate_runtime_invariants and any(
                ptr != free_ptrs[0] for ptr in free_ptrs[1:]
            ):
                raise RuntimeError(
                    "H2O static decode requires aligned per-layer free-stack pointers: "
                    f"layers={layer_ids} ptrs={free_ptrs}."
                )
            free_ptr = free_ptrs[0]
            min_free_ptr = min(free_ptrs)
            if min_free_ptr < real_batch_size:
                raise RuntimeError(
                    "Out of KV cache slots in H2O static decode: "
                    f"need={real_batch_size} free={min_free_ptr}."
                )
            new_slots = self.free_slots_stack_tensor[
                kv_layers_gpu,
                free_ptr - real_batch_size : free_ptr,
            ]
            for layer_id in layer_ids:
                self._num_free_slots[layer_id] -= real_batch_size

            next_lens = cur_lens + 1
            next_lens_gpu = torch.as_tensor(
                next_lens,
                dtype=torch.int32,
                device=self.device,
            )
            self.buffer_req_to_token_slots_tensor[
                kv_layers_gpu[:, None],
                rows_gpu[None, :],
                torch.as_tensor(cur_lens, dtype=torch.long, device=self.device)[None, :],
            ] = new_slots.to(torch.int32)
            for layer_id in layer_ids:
                self.row_seq_lens[layer_id][list(row_indices)] = next_lens

            layers_slot_mapping, layers_context_lens, layers_req_indices = (
                self._get_decode_static_buffers(graph_batch_size)
            )
            layers_slot_mapping.index_fill_(0, layers_gpu, -1)
            layers_slot_mapping[
                layers_gpu[:, None],
                batch_cols_gpu[None, :],
            ] = new_slots.to(torch.int32)

            layers_context_lens.index_fill_(0, layers_gpu, int(next_lens[0]))
            layers_req_indices.index_fill_(0, layers_gpu, int(row_indices[0]))
            layers_context_lens[
                layers_gpu[:, None],
                batch_cols_gpu[None, :],
            ] = next_lens_gpu[None, :]
            layers_req_indices[
                layers_gpu[:, None],
                batch_cols_gpu[None, :],
            ] = rows_i32_gpu[None, :]

            input_ids_list = [seq.decode_input_token for seq in seqs]
            positions_list = [seq.decode_input_position for seq in seqs]
            input_ids[:real_batch_size].copy_(
                torch.tensor(input_ids_list, dtype=torch.int64, device=self.device)
            )
            positions[:real_batch_size].copy_(
                torch.tensor(positions_list, dtype=torch.int64, device=self.device)
            )
            if graph_batch_size > real_batch_size:
                input_ids[real_batch_size:].fill_(input_ids_list[0])
                positions[real_batch_size:].fill_(positions_list[0])

            binding_key = (
                graph_batch_size,
                int(layers_slot_mapping.data_ptr()),
                int(layers_context_lens.data_ptr()),
                int(layers_req_indices.data_ptr()),
            )
            if self._decode_static_state_binding_key != binding_key:
                max_context_len = (
                    int(static_cap)
                    if static_cap is not None
                    else int(next_lens.max())
                )
                for layer_id in layer_ids:
                    state = self.layer_batch_states[layer_id]
                    state.slot_mapping = layers_slot_mapping[layer_id]
                    state.context_lens = layers_context_lens[layer_id]
                    state.max_context_len = max_context_len
                    state.req_indices = layers_req_indices[layer_id]
                self._decode_static_state_binding_key = binding_key

            self.validate_decode_cuda_graph_slot_mappings()

            slot_mapping.copy_(layers_slot_mapping[first_layer])
            context_lens.copy_(layers_context_lens[first_layer])
            req_indices.copy_(layers_req_indices[first_layer])
            return input_ids, positions, None

    @staticmethod
    def select_h2o_indices(
        scores: torch.Tensor,
        *,
        budget: int,
        recent_ratio: float,
    ) -> torch.Tensor:
        """Select heavy hitters plus a fixed recent suffix, in logical order."""
        if scores.dim() != 1:
            raise ValueError(f"H2O scores must be 1D, got shape={tuple(scores.shape)}.")
        kv_len = int(scores.numel())
        budget = int(budget)
        if budget <= 0:
            raise ValueError(f"H2O budget must be positive, got {budget}.")
        if not 0.0 < float(recent_ratio) < 1.0:
            raise ValueError(f"H2O recent_ratio must be in (0, 1), got {recent_ratio}.")
        if kv_len <= budget:
            return torch.arange(kv_len, dtype=torch.long, device=scores.device)

        recent_count = max(1, int(budget * float(recent_ratio)))
        recent_count = min(recent_count, budget, kv_len)
        heavy_count = budget - recent_count
        recent_start = kv_len - recent_count
        recent = torch.arange(recent_start, kv_len, dtype=torch.long, device=scores.device)
        if heavy_count == 0:
            return recent
        heavy = torch.argsort(
            scores[:recent_start],
            dim=0,
            descending=True,
            stable=True,
        )[: min(heavy_count, recent_start)]
        keep = torch.cat((heavy, recent))
        if int(keep.numel()) != budget:
            raise RuntimeError(
                "H2O selection did not fill the requested budget: "
                f"kv_len={kv_len} budget={budget} selected={int(keep.numel())}."
            )
        return torch.sort(keep).values

    @staticmethod
    def select_h2o_indices_batch(
        scores: torch.Tensor,
        *,
        budget: int,
        recent_ratio: float,
    ) -> torch.Tensor:
        """Batched H2O selection for scores shaped [batch, kv_len]."""
        if scores.dim() != 2:
            raise ValueError(
                "Batched H2O scores must have shape [batch, kv_len], "
                f"got {tuple(scores.shape)}."
            )
        kv_len = int(scores.shape[-1])
        budget = int(budget)
        if budget <= 0:
            raise ValueError(f"H2O budget must be positive, got {budget}.")
        if not 0.0 < float(recent_ratio) < 1.0:
            raise ValueError(f"H2O recent_ratio must be in (0, 1), got {recent_ratio}.")
        if kv_len <= budget:
            return torch.arange(
                kv_len, dtype=torch.long, device=scores.device
            ).expand(int(scores.shape[0]), -1)

        recent_count = max(1, int(budget * float(recent_ratio)))
        recent_count = min(recent_count, budget, kv_len)
        heavy_count = budget - recent_count
        recent_start = kv_len - recent_count
        recent = torch.arange(
            recent_start, kv_len, dtype=torch.long, device=scores.device
        ).expand(int(scores.shape[0]), -1)
        if heavy_count == 0:
            return recent
        heavy = torch.argsort(
            scores[:, :recent_start],
            dim=1,
            descending=True,
            stable=True,
        )[:, : min(heavy_count, recent_start)]
        keep = torch.cat((heavy, recent), dim=1)
        if int(keep.shape[1]) != budget:
            raise RuntimeError(
                "Batched H2O selection did not fill the requested budget: "
                f"kv_len={kv_len} budget={budget} selected={int(keep.shape[1])}."
            )
        return torch.sort(keep, dim=1).values

    def _score_key(self, layer_idx: int, seq_id: int) -> tuple[int, int]:
        return int(layer_idx), int(seq_id)

    def h2o_score(self, layer_idx: int, seq_id: int) -> torch.Tensor | None:
        return self._h2o_scores.get(self._score_key(layer_idx, seq_id))

    def _require_score_length(
        self,
        layer_idx: int,
        seq: Sequence,
        expected_len: int,
    ) -> torch.Tensor:
        score = self.h2o_score(layer_idx, seq.seq_id)
        if score is None:
            raise RuntimeError(
                f"H2O score vector is missing: layer={layer_idx} seq_id={seq.seq_id}."
            )
        if int(score.numel()) != int(expected_len):
            raise RuntimeError(
                "H2O score vector is not aligned with the physical KV row: "
                f"layer={layer_idx} seq_id={seq.seq_id} scores={int(score.numel())} "
                f"physical_len={int(expected_len)}."
            )
        return score

    @staticmethod
    def _expand_score(
        score: torch.Tensor | None,
        new_len: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        new_len = int(new_len)
        if new_len < 0:
            raise ValueError(f"H2O score length must be non-negative, got {new_len}.")
        old_len = 0 if score is None else int(score.numel())
        if old_len > new_len:
            raise RuntimeError(
                f"H2O score vector cannot shrink without keep_indices: old={old_len} new={new_len}."
            )
        expanded = torch.zeros((new_len,), dtype=torch.float32, device=device)
        if score is not None and old_len > 0:
            expanded[:old_len].copy_(score.to(device=device, dtype=torch.float32))
        return expanded

    @classmethod
    def _accumulate_score(
        cls,
        previous: torch.Tensor | None,
        step_score: torch.Tensor,
        *,
        new_len: int,
        weight: float,
    ) -> torch.Tensor:
        if step_score.dim() != 1 or int(step_score.numel()) < int(new_len):
            raise ValueError(
                "H2O step score must be a 1D vector covering new_len: "
                f"shape={tuple(step_score.shape)} new_len={int(new_len)}."
            )
        cumulative = cls._expand_score(previous, new_len, device=step_score.device)
        cumulative.add_(step_score[:new_len].float(), alpha=float(weight))
        return cumulative

    @staticmethod
    def _normalize_logit_prefill_score(
        step_score: torch.Tensor,
        *,
        new_len: int,
    ) -> torch.Tensor:
        """Normalize max-reduced prefill logits to step token probabilities.

        Unscored positions retain -inf and map to zero probability under
        softmax.
        """
        if step_score.dim() != 1 or int(step_score.numel()) < int(new_len):
            raise ValueError(
                "H2O logit prefill score must be a 1D vector covering new_len: "
                f"shape={tuple(step_score.shape)} new_len={int(new_len)}."
            )
        logits = step_score[: int(new_len)].float()
        has_invalid = (torch.isnan(logits) | (logits == torch.inf)).any()
        has_finite = torch.isfinite(logits).any()
        valid = (~has_invalid) & has_finite
        if valid.is_cuda:
            torch._assert_async(valid)
        elif not bool(valid.item()):
            raise RuntimeError(
                "H2O logit prefill score contains invalid non-finite values (NaN or +inf) "
                "or lacks any finite score."
            )
        return torch.softmax(logits, dim=0)

    def _physical_row_len(self, layer_idx: int, seq: Sequence) -> int:
        row_idx = self.seq_id_to_row[layer_idx].get(int(seq.seq_id))
        if row_idx is None:
            raise RuntimeError(
                f"H2O physical row is missing: layer={layer_idx} seq_id={seq.seq_id}."
            )
        return int(self.row_seq_lens[layer_idx][row_idx])

    def _invalidate_h2o_decode_score_workspace(self) -> None:
        """Invalidate row views when cache ownership or layout changes."""
        self._h2o_decode_score_signature = None
        self._h2o_decode_score_length = 0

    def _prepare_prefill(self, seqs: list[Sequence]):
        """Append a logical prompt chunk to each compressed physical KV row."""
        with profiler.record("cache_prepare_prefill"):
            self._invalidate_h2o_decode_score_workspace()
            self._decode_static_state_binding_key = None
            self._prefill_context_lens_cpu_by_layer = {}
            self._prefill_score_metadata_cache = {}
            layer_ids = self.kv_transformer_layer_indices()
            score_layer_ids = set(self.kv_transformer_layer_indices())
            total_chunk_tokens = sum(int(seq.current_chunk_size) for seq in seqs)
            input_ids_np = np.empty(total_chunk_tokens, dtype=np.int64)
            positions_np = np.empty(total_chunk_tokens, dtype=np.int64)
            cu_seqlens_q = [0]
            layers_slot_mapping = torch.full(
                (self.num_layers, total_chunk_tokens),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            layer_context_lens: dict[int, list[int]] = {int(layer): [] for layer in layer_ids}

            token_offset = 0
            for seq in seqs:
                chunk_size = int(seq.current_chunk_size)
                logical_start = int(seq.num_prefilled_tokens)
                logical_end = logical_start + chunk_size
                if logical_start == 0:
                    for layer_idx in score_layer_ids:
                        self._h2o_scores.pop(self._score_key(layer_idx, seq.seq_id), None)

                for layer_idx in layer_ids:
                    row_idx = self._get_free_row(layer_idx, int(seq.seq_id))
                    physical_start = int(self.row_seq_lens[layer_idx][row_idx])
                    if logical_start == 0 and physical_start != 0:
                        raise RuntimeError(
                            "H2O first prefill chunk found a non-empty physical row: "
                            f"layer={layer_idx} seq_id={seq.seq_id} physical_len={physical_start}."
                        )
                    if logical_start > 0 and layer_idx in score_layer_ids:
                        self._require_score_length(layer_idx, seq, physical_start)
                    self._allocate(layer_idx, int(seq.seq_id), chunk_size)
                    physical_end = physical_start + chunk_size
                    layers_slot_mapping[
                        layer_idx, token_offset : token_offset + chunk_size
                    ] = self.buffer_req_to_token_slots[layer_idx][
                        row_idx, physical_start:physical_end
                    ]
                    layer_context_lens[int(layer_idx)].append(physical_end)

                chunk_tokens = seq.token_ids
                if len(chunk_tokens) > chunk_size:
                    chunk_tokens = chunk_tokens[logical_start:logical_end]
                if len(chunk_tokens) != chunk_size:
                    raise RuntimeError(
                        "H2O prefill token slice length mismatch: "
                        f"seq_id={seq.seq_id} expected={chunk_size} got={len(chunk_tokens)}."
                    )
                input_ids_np[token_offset : token_offset + chunk_size] = chunk_tokens
                positions_np[token_offset : token_offset + chunk_size] = np.arange(
                    logical_start, logical_end
                )
                token_offset += chunk_size
                cu_seqlens_q.append(token_offset)

            for layer_idx in layer_ids:
                context_lens = layer_context_lens[int(layer_idx)]
                self._prefill_context_lens_cpu_by_layer[int(layer_idx)] = tuple(
                    int(value) for value in context_lens
                )
                state = self.layer_batch_states[layer_idx]
                state.slot_mapping = layers_slot_mapping[layer_idx]
                state.context_lens = torch.tensor(
                    context_lens, dtype=torch.int32, device=self.device
                )
                state.max_context_len = max(context_lens) if context_lens else 0
                state.req_indices = torch.tensor(
                    [self.seq_id_to_row[layer_idx][int(seq.seq_id)] for seq in seqs],
                    dtype=torch.int32,
                    device=self.device,
                )

            return (
                torch.from_numpy(input_ids_np).to(self.device),
                torch.from_numpy(positions_np).to(self.device),
                torch.tensor(cu_seqlens_q, dtype=torch.int32, device=self.device),
            )

    def prefill_score_ranges(
        self,
        layer_idx: int,
        seqs: list[Sequence],
    ) -> list[tuple[int, Sequence, int, int, int]]:
        """Return (batch, seq, prompt_cache_len, score_start, score_end)."""
        window = int(self.config.h2o_prefill_score_window)
        ranges = []
        for batch_idx, seq in enumerate(seqs):
            chunk_len = int(seq.current_chunk_size)
            context_len = self._physical_row_len(layer_idx, seq)
            prompt_cache_len = context_len - chunk_len
            if prompt_cache_len < 0:
                raise RuntimeError(
                    "H2O current chunk exceeds its physical row: "
                    f"layer={layer_idx} seq_id={seq.seq_id} "
                    f"context={context_len} chunk={chunk_len}."
                )
            score_end = context_len
            score_start = (
                max(prompt_cache_len, context_len - window)
                if window > 0
                else prompt_cache_len
            )
            ranges.append((batch_idx, seq, prompt_cache_len, score_start, score_end))
        return ranges

    def prefill_score_request(self, layer_idx, seqs):
        rows = self.prefill_score_ranges(layer_idx, seqs)
        if not rows:
            return None
        ranges = [(0, 0)] * len(seqs)
        for batch_idx, _seq, _cached, start, end in rows:
            ranges[batch_idx] = (start, end)
        return PrefillScoreRequest(tuple(ranges), self.config.sparse_prefill_score_mode)

    @torch.no_grad()
    def collect_prefill_attention_score(
        self,
        layer_idx: int,
        q: torch.Tensor,
        view: PrefillComputeView,
        *,
        b_start_loc: torch.Tensor,
        chunk_lens: torch.Tensor,
        attention_lse: torch.Tensor | None = None,
    ):
        ctx = get_context()
        if not ctx.is_prefill:
            raise RuntimeError("H2O prefill score collection was called outside prefill.")
        seqs = getattr(ctx, "seqs", None)
        if seqs is None:
            raise RuntimeError("H2O prefill score collection requires current seqs in context.")
        if int(chunk_lens.ndim) != 1 or int(chunk_lens.shape[0]) != len(seqs):
            raise RuntimeError(
                "H2O prefill scoring chunk-length batch mismatch: "
                f"shape={tuple(chunk_lens.shape)} seqs={len(seqs)}."
            )
        ranges = self.prefill_score_ranges(layer_idx, seqs)
        if not ranges:
            return None
        if view.token_scores is None and not isinstance(view.payload, ExplicitKVPayload):
            raise TypeError(
                "H2O prefill scoring requires ExplicitKVPayload, got "
                f"{type(view.payload).__name__}."
            )
        meta = view.meta
        payload = view.payload

        context_lens = tuple(int(item[4]) for item in ranges)
        prepared_context_lens = getattr(
            self,
            "_prefill_context_lens_cpu_by_layer",
            {},
        ).get(int(layer_idx))
        if prepared_context_lens is None and meta.context_lens.device.type == "cpu":
            prepared_context_lens = tuple(
                int(value) for value in meta.context_lens.tolist()
            )
        if prepared_context_lens is None:
            raise RuntimeError(
                "H2O prefill scoring requires CPU context lengths prepared "
                f"for layer={layer_idx}."
            )
        if tuple(prepared_context_lens) != context_lens:
            raise RuntimeError(
                "H2O prefill score view is not in compressed physical coordinates: "
                f"layer={layer_idx} view={tuple(prepared_context_lens)} "
                f"physical={context_lens}."
            )
        prompt_cache_lens_cpu = tuple(int(item[2]) for item in ranges)
        score_starts_cpu = tuple(int(item[3]) for item in ranges)
        score_ends_cpu = tuple(int(item[4]) for item in ranges)
        (
            prompt_cache_lens,
            _batch_indices,
            score_starts,
            score_ends,
        ) = self._cached_prefill_score_metadata_tensors(
            device=q.device,
            context_lens=context_lens,
            prompt_cache_lens=prompt_cache_lens_cpu,
            batch_indices=tuple(range(len(ranges))),
            score_starts=score_starts_cpu,
            score_ends=score_ends_cpu,
        )
        max_context_len = max(context_lens)
        if view.token_scores is not None:
            step_score = view.token_scores.to(self._prefill_score_dtype())
        elif meta.attn_score is None:
            step_score = self._prefill_step_score_buffer(
                batch_size=len(seqs),
                max_context_len=max_context_len,
                device=q.device,
            )
            max_score_len = max(item[4] - item[3] for item in ranges)
            if attention_lse is None:
                self._run_prefill_score(
                    q,
                    payload.k_cache,
                    step_score,
                    meta,
                    b_start_loc,
                    prompt_cache_lens,
                    max_score_len,
                    score_starts,
                    score_ends,
                    candidate_start=0,
                    recent_keep_tokens=0,
                )
            else:
                if self.config.sparse_prefill_score_mode != "probability":
                    raise RuntimeError(
                        "FA3 softmax LSE is only valid for probability H2O scoring."
                    )
                # TODO(h2o-probability-efficiency): optimize full-chunk scoring;
                # matched H100/Qwen3-4B runs show substantial TTFT overhead versus
                # logits. Preserve normalized-attention score semantics.
                from sparseengine.kernels.triton.prefill_score import (
                    prefill_score_from_lse_fwd,
                )

                prefill_score_from_lse_fwd(
                    q,
                    payload.k_cache,
                    attention_lse,
                    step_score,
                    meta.req_indices,
                    b_start_loc,
                    meta.context_lens,
                    prompt_cache_lens,
                    max_score_len,
                    meta.active_slots,
                    score_starts,
                    score_ends,
                    workspace=getattr(self, "_prefill_score_workspace", None),
                )
        else:
            if (
                self.config.sparse_prefill_score_mode != "logits"
                or int(self.config.h2o_prefill_score_window) != 0
            ):
                raise RuntimeError(
                    "H2O main-attention prefill scores require logits mode with "
                    "h2o_prefill_score_window=0."
                )
            step_score = meta.attn_score
            if (
                step_score.ndim != 2
                or int(step_score.shape[0]) != len(seqs)
                or int(step_score.shape[1]) < max_context_len
            ):
                raise ValueError(
                    "H2O fused prefill scores must have shape [batch, context], "
                    f"got {tuple(step_score.shape)} for batch={len(seqs)} "
                    f"max_context_len={max_context_len}."
                )
        for batch_idx, seq, prompt_cache_len, score_start, score_end in ranges:
            key = self._score_key(layer_idx, seq.seq_id)
            previous = self._h2o_scores.get(key)
            if prompt_cache_len > 0 and previous is None:
                raise RuntimeError(
                    "H2O prefill score vector is missing for an existing physical prefix: "
                    f"layer={layer_idx} seq_id={seq.seq_id} "
                    f"prompt_cache_len={prompt_cache_len}."
                )
            if previous is not None and int(previous.numel()) != prompt_cache_len:
                raise RuntimeError(
                    "H2O prefill score vector lost physical-row alignment before append: "
                    f"layer={layer_idx} seq_id={seq.seq_id} scores={int(previous.numel())} "
                    f"prompt_cache_len={prompt_cache_len}."
                )
            effective_queries = score_end - score_start
            score_row = step_score[batch_idx]
            if self.config.sparse_prefill_score_mode == "logits":
                score_row = self._normalize_logit_prefill_score(
                    score_row,
                    new_len=score_end,
                )
            cumulative = self._accumulate_score(
                previous,
                score_row,
                new_len=score_end,
                weight=float(effective_queries),
            )
            self._h2o_scores[key] = cumulative
        return None

    @torch.no_grad()
    def update_decode_attention_scores(
        self,
        layer_idx: int,
        seqs: list[Sequence],
        normalized_scores: torch.Tensor,
    ):
        if normalized_scores.dim() != 2 or int(normalized_scores.shape[0]) < len(seqs):
            raise ValueError(
                "H2O decode scores must have shape [batch, length], "
                f"got {tuple(normalized_scores.shape)} for batch={len(seqs)}."
            )
        kv_lens = [self._physical_row_len(layer_idx, seq) for seq in seqs]
        if kv_lens and int(normalized_scores.shape[1]) < max(kv_lens):
            raise ValueError(
                "H2O decode scores do not cover the longest physical KV row: "
                f"score_width={int(normalized_scores.shape[1])} "
                f"max_kv_len={max(kv_lens)}."
            )
        if kv_lens and all(kv_len == kv_lens[0] for kv_len in kv_lens[1:]):
            kv_len = int(kv_lens[0])
            previous_rows = []
            for seq in seqs:
                previous = self._h2o_scores.get(self._score_key(layer_idx, seq.seq_id))
                expected_previous_len = kv_len - 1
                if previous is None or int(previous.numel()) != expected_previous_len:
                    raise RuntimeError(
                        "H2O decode score vector must align before appending the new token: "
                        f"layer={layer_idx} seq_id={seq.seq_id} "
                        f"scores={None if previous is None else int(previous.numel())} "
                        f"expected={expected_previous_len} current_kv_len={kv_len}."
                    )
                previous_rows.append(previous)
            previous_scores = torch.stack(previous_rows, dim=0)
            cumulative = F.pad(previous_scores, (0, 1))
            cumulative.add_(normalized_scores[: len(seqs), :kv_len].float())
            for batch_idx, seq in enumerate(seqs):
                self._h2o_scores[self._score_key(layer_idx, seq.seq_id)] = cumulative[
                    batch_idx
                ]
            return

        for batch_idx, (seq, kv_len) in enumerate(zip(seqs, kv_lens)):
            key = self._score_key(layer_idx, seq.seq_id)
            previous = self._h2o_scores.get(key)
            expected_previous_len = kv_len - 1
            if previous is None or int(previous.numel()) != expected_previous_len:
                raise RuntimeError(
                    "H2O decode score vector must align before appending the new token: "
                    f"layer={layer_idx} seq_id={seq.seq_id} "
                    f"scores={None if previous is None else int(previous.numel())} "
                    f"expected={expected_previous_len} current_kv_len={kv_len}."
                )
            cumulative = self._accumulate_score(
                previous,
                normalized_scores[batch_idx],
                new_len=kv_len,
                weight=1.0,
            )
            self._h2o_scores[key] = cumulative

    @cpu_timing.timed
    @torch.no_grad()
    def accumulate_decode_headwise_logits(
        self,
        layer_indices: list[int],
        seqs: list[Sequence],
        raw_logits: torch.Tensor,
        *,
        softmax_scale: float,
        normalize_logits: bool = True,
    ) -> None:
        """Accumulate into request-owned rows, independent of the scheduled batch.

        Only a replaced/compacted row or an exhausted capacity copies history.
        No model graph owns these rows; finish_step publishes the updated views.
        """
        from sparseengine.kernels.triton.h2o_score import h2o_headwise_softmax_accumulate_rows

        if raw_logits.ndim != 4 or tuple(raw_logits.shape[:2]) != (len(layer_indices), len(seqs)):
            raise ValueError("H2O raw decode logits must be [layers, batch, heads, width]")
        if not layer_indices or not seqs:
            return
        if raw_logits.dtype != torch.float32 or not 0 < softmax_scale < float("inf"):
            raise ValueError("H2O raw decode requires FP32 logits and finite positive scale")
        if len(set(layer_indices)) != len(layer_indices) or len({seq.seq_id for seq in seqs}) != len(seqs):
            raise ValueError("H2O decode cannot update the same destination row twice")
        keys = [(layer, int(seq.seq_id)) for layer in layer_indices for seq in seqs]
        lengths = [self._physical_row_len(layer, seq) for layer in layer_indices for seq in seqs]
        if min(lengths) <= 0 or max(lengths) > raw_logits.shape[-1]:
            raise ValueError("H2O raw decode logits do not cover the physical rows")
        previous_rows = [self._h2o_scores.get(key) for key in keys]
        # Validate the entire batch before changing ownership or publishing views.
        for key, previous, length in zip(keys, previous_rows, lengths):
            if previous is None or previous.numel() != length - 1:
                raise RuntimeError(
                    "H2O headwise score vector must align before appending: "
                    f"layer={key[0]} seq_id={key[1]} expected={length - 1}"
                )
            if previous.device != raw_logits.device or previous.dtype != torch.float32:
                raise TypeError("H2O history and logits must share FP32 dtype and device")
        rows = getattr(self, "_h2o_decode_score_rows", None)
        if rows is None:
            rows = self._h2o_decode_score_rows = {}
        destinations = []
        for key, previous, length in zip(keys, previous_rows, lengths):
            owned = rows.get(key)
            if owned is None or owned[1] is not previous or owned[0].numel() < length:
                capacity = max(length, self.h2o_decode_budget + self.h2o_decode_eviction_interval)
                if owned is not None and owned[1] is previous:
                    capacity = max(capacity, 2 * owned[0].numel())
                buffer = torch.empty(capacity, dtype=torch.float32, device=raw_logits.device)
                if length > 1:
                    buffer[:length - 1].copy_(previous)
            else:
                buffer = owned[0]
            destinations.append(buffer)

        # Reuse pinned host metadata only after its preceding DMA has completed.
        # Device copies are ordered on the same stream as score accumulation.
        metadata = getattr(self, "_h2o_decode_score_metadata", None)
        if metadata is not None:
            device_runtime.synchronize_event(metadata[2])
        count = len(keys)
        pin_memory = raw_logits.is_cuda and device_runtime.supports_pin_memory()
        if metadata is None or metadata[0].shape[0] < count or metadata[1].device != raw_logits.device:
            capacity = 1 << (count - 1).bit_length()
            host = torch.empty((capacity, 2), dtype=torch.int64, pin_memory=pin_memory)
            device = torch.empty_like(host, device=raw_logits.device)
            ready = device_runtime.new_event(device=raw_logits.device)
            metadata = self._h2o_decode_score_metadata = (host, device, ready)
        host, device, ready = metadata
        host.numpy()[:count, 0] = [buffer.data_ptr() for buffer in destinations]
        host.numpy()[:count, 1] = lengths
        device[:count].copy_(host[:count], non_blocking=pin_memory)
        device_runtime.record_event(ready, device=raw_logits.device)
        h2o_headwise_softmax_accumulate_rows(
            raw_logits[..., :max(lengths)],
            device[:count].view(len(layer_indices), len(seqs), 2),
            softmax_scale=softmax_scale,
            normalize_logits=normalize_logits,
        )
        for key, buffer, length in zip(keys, destinations, lengths):
            view = buffer[:length]
            self._h2o_scores[key] = view
            rows[key] = (buffer, view)

    @cpu_timing.timed
    @torch.no_grad()
    def update_decode_attention_scores_all_layers(
        self,
        layer_indices: list[int],
        seqs: list[Sequence],
        reduced_scores: torch.Tensor,
        *,
        normalize_logits: bool = False,
        softmax_scale: float | None = None,
    ) -> bool:
        """Accumulate one reduced [layers, batch, width] decode score tensor.

        Returns True when the cross-layer path was used. CUDA rows keep their
        ownership through batch churn for raw QK and already-normalized scores.
        CPU probability updates retain their reference tensor implementation.
        """
        if reduced_scores.dim() != 3:
            raise ValueError(
                "H2O all-layer decode scores must have shape [layers, batch, width], "
                f"got {tuple(reduced_scores.shape)}."
            )
        if tuple(reduced_scores.shape[:2]) != (len(layer_indices), len(seqs)):
            raise ValueError(
                "H2O all-layer decode score shape does not match layers and batch: "
                f"layers={len(layer_indices)} batch={len(seqs)} "
                f"shape={tuple(reduced_scores.shape)}."
            )
        if not layer_indices or not seqs:
            return True
        if normalize_logits and softmax_scale is None:
            raise ValueError(
                "H2O raw-logit accumulation requires an explicit softmax_scale."
            )

        if normalize_logits or reduced_scores.is_cuda:
            scores = reduced_scores if normalize_logits else reduced_scores.float()
            self.accumulate_decode_headwise_logits(
                layer_indices, seqs, scores.unsqueeze(2),
                softmax_scale=float(softmax_scale) if normalize_logits else 1.0,
                normalize_logits=normalize_logits,
            )
            return True

        physical_lens = [
            self._physical_row_len(layer_indices[0], seq) for seq in seqs
        ]
        if self.validate_runtime_invariants:
            physical_lens.extend(
                self._physical_row_len(layer_idx, seq)
                for layer_idx in layer_indices[1:]
                for seq in seqs
            )
        kv_len = int(physical_lens[0])
        if any(int(length) != kv_len for length in physical_lens[1:]):
            for local_layer, layer_idx in enumerate(layer_indices):
                self.update_decode_attention_scores(
                    layer_idx,
                    seqs,
                    reduced_scores[local_layer],
                )
            return False
        if kv_len <= 0:
            raise RuntimeError(
                f"H2O all-layer decode score update requires positive KV length, got {kv_len}."
            )
        if int(reduced_scores.shape[2]) < kv_len:
            raise ValueError(
                "H2O all-layer decode scores do not cover the physical KV rows: "
                f"score_width={int(reduced_scores.shape[2])} kv_len={kv_len}."
            )

        previous_len = kv_len - 1
        signature = (
            tuple(int(layer_idx) for layer_idx in layer_indices),
            tuple(int(seq.seq_id) for seq in seqs),
        )
        workspace = getattr(self, "_h2o_decode_score_workspace", None)
        can_reuse = (
            workspace is not None
            and getattr(self, "_h2o_decode_score_signature", None) == signature
            and getattr(self, "_h2o_decode_score_length", 0) == previous_len
            and tuple(workspace.shape[:2]) == (len(layer_indices), len(seqs))
            and int(workspace.shape[2]) >= kv_len
        )
        previous_rows = []
        if not can_reuse or self.validate_runtime_invariants:
            for layer_idx in layer_indices:
                for seq in seqs:
                    previous = self._h2o_scores.get(
                        self._score_key(layer_idx, seq.seq_id)
                    )
                    if previous is None or int(previous.numel()) != previous_len:
                        raise RuntimeError(
                            "H2O all-layer score vector must align before appending: "
                            f"layer={layer_idx} seq_id={seq.seq_id} "
                            f"scores={None if previous is None else int(previous.numel())} "
                            f"expected={previous_len} current_kv_len={kv_len}."
                        )
                    previous_rows.append(previous)

        if can_reuse and self.validate_runtime_invariants:
            storage_ptr = workspace.untyped_storage().data_ptr()
            row_stride = int(workspace.stride(1))
            for row_idx, previous in enumerate(previous_rows):
                if (
                    previous.untyped_storage().data_ptr() != storage_ptr
                    or int(previous.storage_offset()) != row_idx * row_stride
                    or int(previous.numel()) != previous_len
                ):
                    can_reuse = False
                    break

        if not can_reuse:
            capacity = max(
                kv_len,
                self.h2o_decode_budget + self.h2o_decode_eviction_interval,
            )
            workspace = torch.empty(
                (len(layer_indices), len(seqs), capacity),
                dtype=torch.float32,
                device=reduced_scores.device,
            )
            if previous_len:
                previous_scores = torch.stack(previous_rows, dim=0).view(
                    len(layer_indices), len(seqs), previous_len
                )
                workspace[:, :, :previous_len].copy_(previous_scores)

        score_update = reduced_scores[:, :, :kv_len].float()
        if previous_len:
            workspace[:, :, :previous_len].add_(score_update[:, :, :previous_len])
        workspace[:, :, previous_len:kv_len].copy_(score_update[:, :, previous_len:kv_len])
        self._h2o_decode_score_workspace = workspace
        self._h2o_decode_score_signature = signature
        self._h2o_decode_score_length = kv_len
        for local_layer, layer_idx in enumerate(layer_indices):
            for batch_idx, seq in enumerate(seqs):
                self._h2o_scores[
                    self._score_key(layer_idx, seq.seq_id)
                ] = workspace[local_layer, batch_idx, :kv_len]
        return True

    def free_part_slots(
        self,
        layer_idx: int,
        seq: Sequence,
        keep_indices: torch.Tensor,
        *,
        keep_indices_sorted: bool = False,
    ):
        if keep_indices is None:
            return
        row_idx, kv_len, keep_indices = self._prepare_free_part_slots(
            layer_idx,
            seq,
            keep_indices,
            keep_indices_sorted=keep_indices_sorted,
            synchronize_validation=True,
        )
        score = self._require_score_length(layer_idx, seq, kv_len)
        kept_score = score.index_select(0, keep_indices).contiguous()
        self._invalidate_h2o_decode_score_workspace()
        self._apply_free_part_slots(
            layer_idx,
            seq,
            row_idx,
            kv_len,
            keep_indices,
        )
        self._h2o_scores[self._score_key(layer_idx, seq.seq_id)] = kept_score

    def _get_final_prefill_workspace(
        self,
        *,
        batch_size: int,
        budget: int,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        if batch_size <= 0 or budget <= 0:
            raise RuntimeError(
                "H2O final-prefill workspace requires positive batch and budget: "
                f"batch={batch_size} budget={budget}."
            )
        if not isinstance(k_cache, torch.Tensor) or not isinstance(v_cache, torch.Tensor):
            raise TypeError("H2O final-prefill dense compaction requires tensor K/V caches.")
        if k_cache.dim() != 3 or v_cache.dim() != 3 or tuple(k_cache.shape) != tuple(v_cache.shape):
            raise RuntimeError(
                "H2O final-prefill dense compaction requires matching [slots, heads, dim] caches: "
                f"k_shape={tuple(k_cache.shape)} v_shape={tuple(v_cache.shape)}."
            )
        if k_cache.dtype != v_cache.dtype or k_cache.device != v_cache.device:
            raise RuntimeError(
                "H2O final-prefill dense compaction requires matching K/V dtype and device: "
                f"k={k_cache.dtype}/{k_cache.device} v={v_cache.dtype}/{v_cache.device}."
            )

        required_shape = (
            2,
            int(batch_size),
            int(budget),
            int(k_cache.shape[1]),
            int(k_cache.shape[2]),
        )
        workspace = getattr(self, "_h2o_final_prefill_workspace", None)
        needs_allocation = (
            workspace is None
            or workspace.dtype != k_cache.dtype
            or workspace.device != k_cache.device
            or int(workspace.shape[1]) < int(batch_size)
            or tuple(workspace.shape[2:]) != required_shape[2:]
        )
        if needs_allocation:
            workspace = torch.empty(
                required_shape,
                dtype=k_cache.dtype,
                device=k_cache.device,
            )
            self._h2o_final_prefill_workspace = workspace
        return workspace[:, :batch_size]

    @staticmethod
    def _assert_final_prefill_tensor(condition: torch.Tensor, message: str) -> None:
        if condition.numel() != 1:
            raise RuntimeError(
                "H2O final-prefill validation must reduce to one boolean: "
                f"shape={tuple(condition.shape)} message={message}."
            )
        if condition.is_cuda:
            torch._assert_async(condition)
        elif not bool(condition.item()):
            raise RuntimeError(message)

    def _preflight_final_prefill_dense_capacity(
        self,
        seqs: list[Sequence],
    ) -> None:
        """Validate every final-prefill free-stack update before moving any KV."""
        final_seqs = [seq for seq in seqs if bool(seq.is_last_chunk_prefill)]
        if not final_seqs:
            return
        seq_ids = [int(seq.seq_id) for seq in final_seqs]
        if len(seq_ids) != len(set(seq_ids)):
            raise RuntimeError(
                "H2O final-prefill capacity preflight received duplicate seq ids: "
                f"{seq_ids}."
            )

        budget = self.h2o_decode_budget
        for layer_idx in self.kv_transformer_layer_indices():
            row_indices = []
            release_count = 0
            for seq in final_seqs:
                row_idx = self.seq_id_to_row[layer_idx].get(int(seq.seq_id))
                if row_idx is None:
                    raise RuntimeError(
                        "H2O final-prefill capacity preflight is missing a physical row: "
                        f"layer={layer_idx} seq_id={int(seq.seq_id)}."
                    )
                row_indices.append(int(row_idx))
                release_count += max(
                    0,
                    int(self.row_seq_lens[layer_idx][row_idx]) - budget,
                )
            if len(row_indices) != len(set(row_indices)):
                raise RuntimeError(
                    "H2O final-prefill capacity preflight received duplicate physical rows: "
                    f"layer={layer_idx} rows={row_indices}."
                )
            if release_count == 0:
                continue

            free_stack = self.free_slots_stack[layer_idx]
            if free_stack is None or free_stack.dim() != 1:
                raise RuntimeError(
                    "H2O final-prefill dense compaction requires a one-dimensional "
                    f"free-slot stack: layer={layer_idx}."
                )
            free_ptr = int(self._num_free_slots[layer_idx])
            if free_ptr < 0 or free_ptr + release_count > int(free_stack.numel()):
                raise RuntimeError(
                    "H2O final-prefill dense compaction would overflow the free-slot "
                    f"stack: layer={layer_idx} ptr={free_ptr} release={release_count} "
                    f"capacity={int(free_stack.numel())}."
                )

    @torch.no_grad()
    def _compact_final_prefill_dense_batch(
        self,
        layer_idx: int,
        seqs: list[Sequence],
        keep_indices: torch.Tensor,
    ) -> None:
        """Move final H2O selections into ascending physical destination slots."""
        if not seqs:
            raise RuntimeError("H2O final-prefill dense compaction requires sequences.")
        kv_idx = self.kv_layer_index(layer_idx)
        budget = self.h2o_decode_budget
        batch_size = len(seqs)
        keep_indices = keep_indices.to(
            device=self.device,
            dtype=torch.long,
        ).contiguous()
        if keep_indices.dim() != 2 or tuple(keep_indices.shape) != (batch_size, budget):
            raise RuntimeError(
                "H2O final-prefill keep indices must have shape [batch, decode_budget]: "
                f"expected={(batch_size, budget)} got={tuple(keep_indices.shape)}."
            )

        seq_ids = [int(seq.seq_id) for seq in seqs]
        if len(seq_ids) != len(set(seq_ids)):
            raise RuntimeError(
                f"H2O final-prefill dense compaction received duplicate seq ids: {seq_ids}."
            )
        row_indices = []
        cur_lens = []
        for seq_id in seq_ids:
            row_idx = self.seq_id_to_row[layer_idx].get(seq_id)
            if row_idx is None:
                raise RuntimeError(
                    "H2O final-prefill dense compaction is missing a physical row: "
                    f"layer={layer_idx} seq_id={seq_id}."
                )
            row_indices.append(int(row_idx))
            cur_lens.append(int(self.row_seq_lens[layer_idx][row_idx]))
        if len(row_indices) != len(set(row_indices)):
            raise RuntimeError(
                "H2O final-prefill dense compaction received duplicate physical rows: "
                f"layer={layer_idx} rows={row_indices}."
            )
        kv_len = int(cur_lens[0])
        if any(int(length) != kv_len for length in cur_lens[1:]):
            raise RuntimeError(
                "H2O final-prefill dense batch requires uniform physical lengths; "
                "nonuniform callers must use batch-one compaction: "
                f"layer={layer_idx} lengths={cur_lens}."
            )
        if kv_len <= budget:
            raise RuntimeError(
                "H2O final-prefill dense compaction requires an over-budget row: "
                f"layer={layer_idx} kv_len={kv_len} budget={budget}."
            )

        free_count = (kv_len - budget) * batch_size
        free_ptr = int(self._num_free_slots[layer_idx])
        free_stack = self.free_slots_stack[layer_idx]
        if free_stack is None or free_stack.dim() != 1:
            raise RuntimeError(
                "H2O final-prefill dense compaction requires a one-dimensional free-slot stack: "
                f"layer={layer_idx}."
            )
        if free_ptr < 0 or free_ptr + free_count > int(free_stack.numel()):
            raise RuntimeError(
                "H2O final-prefill dense compaction would overflow the free-slot stack: "
                f"layer={layer_idx} ptr={free_ptr} release={free_count} "
                f"capacity={int(free_stack.numel())}."
            )

        storage = getattr(self, "attention_cache_storage", None)
        uses_explicit_kv = storage is None or isinstance(storage, ExplicitKVStorage)
        if uses_explicit_kv:
            k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
            workspace = self._get_final_prefill_workspace(
                batch_size=batch_size,
                budget=budget,
                k_cache=k_cache,
                v_cache=v_cache,
            )
            slot_capacity = int(k_cache.shape[0])
        else:
            slot_capacity = storage.slot_capacity()
        rows_gpu = torch.tensor(row_indices, dtype=torch.long, device=self.device)
        old_slots = self.buffer_req_to_token_slots[layer_idx][
            rows_gpu, :kv_len
        ].clone()
        self._assert_final_prefill_tensor(
            ((keep_indices >= 0) & (keep_indices < kv_len)).all(),
            "H2O final-prefill keep indices are out of bounds: "
            f"layer={layer_idx} kv_len={kv_len}.",
        )
        if budget > 1:
            self._assert_final_prefill_tensor(
                (keep_indices[:, 1:] > keep_indices[:, :-1]).all(),
                "H2O final-prefill keep indices must be strictly increasing in logical order: "
                f"layer={layer_idx}.",
            )
        self._assert_final_prefill_tensor(
            ((old_slots >= 0) & (old_slots < slot_capacity)).all(),
            "H2O final-prefill slot map contains an out-of-range physical slot: "
            f"layer={layer_idx} num_slots={slot_capacity}.",
        )

        globally_sorted_slots = torch.sort(old_slots.reshape(-1)).values
        if int(globally_sorted_slots.numel()) > 1:
            self._assert_final_prefill_tensor(
                (globally_sorted_slots[1:] != globally_sorted_slots[:-1]).all(),
                "H2O final-prefill active physical slots must be unique across the batch: "
                f"layer={layer_idx} rows={row_indices}.",
            )

        selected_slots = old_slots.gather(1, keep_indices).to(torch.long)
        sorted_old_slots = torch.sort(old_slots, dim=1).values
        destination_slots = sorted_old_slots[:, :budget].contiguous()
        released_slots = sorted_old_slots[:, budget:].reshape(-1).contiguous()

        selected_flat = selected_slots.reshape(-1)
        destination_flat = destination_slots.reshape(-1).to(torch.long)
        if uses_explicit_kv:
            workspace[0].copy_(
                k_cache.index_select(0, selected_flat).view(
                    batch_size,
                    budget,
                    int(k_cache.shape[1]),
                    int(k_cache.shape[2]),
                )
            )
            workspace[1].copy_(
                v_cache.index_select(0, selected_flat).view(
                    batch_size,
                    budget,
                    int(v_cache.shape[1]),
                    int(v_cache.shape[2]),
                )
            )
            k_cache.index_copy_(
                0,
                destination_flat,
                workspace[0].reshape(
                    batch_size * budget,
                    int(k_cache.shape[1]),
                    int(k_cache.shape[2]),
                ),
            )
            v_cache.index_copy_(
                0,
                destination_flat,
                workspace[1].reshape(
                    batch_size * budget,
                    int(v_cache.shape[1]),
                    int(v_cache.shape[2]),
                ),
            )
        else:
            storage.copy_slots(kv_idx, selected_flat, destination_flat)

        free_stack[free_ptr : free_ptr + free_count] = released_slots.to(
            dtype=free_stack.dtype,
            device=free_stack.device,
        )
        self._num_free_slots[layer_idx] = free_ptr + free_count
        self.buffer_req_to_token_slots[layer_idx][
            rows_gpu, :budget
        ] = destination_slots.to(torch.int32)
        self.buffer_req_to_token_slots[layer_idx][rows_gpu, budget:kv_len] = 0
        self.row_seq_lens[layer_idx][row_indices] = budget
        self._uniform_decode_metadata = False

    def _evict(self, seqs: list[Sequence], *, is_prefill: bool):
        if is_prefill:
            self._preflight_final_prefill_dense_capacity(seqs)
        if self._try_batched_evict(seqs, is_prefill=is_prefill):
            return
        ratio = float(self.config.h2o_recent_ratio)
        for layer_idx in self.kv_transformer_layer_indices():
            for seq in seqs:
                is_final_prefill = bool(is_prefill and seq.is_last_chunk_prefill)
                budget = (
                    self.h2o_decode_budget
                    if not is_prefill or is_final_prefill
                    else self.h2o_prefill_budget
                )
                kv_len = self._physical_row_len(layer_idx, seq)
                if kv_len <= budget:
                    continue
                score = self._require_score_length(layer_idx, seq, kv_len)
                keep_indices = self.select_h2o_indices(
                    score,
                    budget=budget,
                    recent_ratio=ratio,
                )
                dropped = kv_len - int(keep_indices.numel())
                if is_final_prefill:
                    kept_score = score.index_select(0, keep_indices).contiguous()
                    self._compact_final_prefill_dense_batch(
                        layer_idx,
                        [seq],
                        keep_indices.unsqueeze(0),
                    )
                    self._h2o_scores[self._score_key(layer_idx, seq.seq_id)] = kept_score
                else:
                    self.free_part_slots(
                        layer_idx,
                        seq,
                        keep_indices,
                        keep_indices_sorted=True,
                    )
                if is_prefill:
                    counter = (
                        "final_prefill_evictions"
                        if is_final_prefill
                        else "intermediate_prefill_evictions"
                    )
                else:
                    counter = "decode_evictions"
                self._h2o_counters[counter] += 1
                self._h2o_counters["dropped_tokens"] += int(dropped)

    def _try_batched_evict(self, seqs: list[Sequence], *, is_prefill: bool) -> bool:
        """Compact each layer across uniform sequence rows with one batch op."""
        if not seqs:
            return False
        layer_indices = list(self.kv_transformer_layer_indices())
        if not layer_indices:
            return False

        final_flags = [bool(seq.is_last_chunk_prefill) for seq in seqs] if is_prefill else []
        if is_prefill and any(flag != final_flags[0] for flag in final_flags[1:]):
            return False
        budget = (
            self.h2o_decode_budget
            if not is_prefill or final_flags[0]
            else self.h2o_prefill_budget
        )

        layer_rows: list[tuple[int, int, list[torch.Tensor]]] = []
        for layer_idx in layer_indices:
            physical_lens = [
                self._physical_row_len(layer_idx, seq) for seq in seqs
            ]
            kv_len = int(physical_lens[0])
            if any(int(length) != kv_len for length in physical_lens[1:]):
                return False
            if kv_len <= budget:
                continue
            score_rows = [
                self._require_score_length(layer_idx, seq, kv_len) for seq in seqs
            ]
            layer_rows.append((int(layer_idx), kv_len, score_rows))

        compact_layers = []
        compact_indices = []
        compact_scores = []
        compact_lengths = []
        if layer_rows:
            self._invalidate_h2o_decode_score_workspace()
        for layer_idx, kv_len, score_rows in layer_rows:
            with profiler.record("h2o_evict_score_stack"):
                scores = torch.stack(score_rows, dim=0)
            with profiler.record("h2o_evict_select"):
                keep_indices = self.select_h2o_indices_batch(
                    scores,
                    budget=budget,
                    recent_ratio=float(self.config.h2o_recent_ratio),
                )
                kept_scores = scores.gather(1, keep_indices)
            compact_layers.append(layer_idx)
            compact_indices.append(keep_indices)
            compact_scores.append(kept_scores)
            compact_lengths.append(kv_len)

        if compact_layers:
            # Each layer supplies its own keep indices. Page-table compaction
            # avoids copying retained K/V rows into a dense destination range.
            with profiler.record("h2o_evict_compact"):
                SnapKVCacheManager.free_part_slots_batch_layers(
                    self,
                    compact_layers,
                    seqs,
                    torch.stack(compact_indices, dim=0),
                    keep_indices_sorted=True,
                )
            for local_layer, layer_idx in enumerate(compact_layers):
                for batch_idx, seq in enumerate(seqs):
                    self._h2o_scores[
                        self._score_key(layer_idx, seq.seq_id)
                    ] = compact_scores[local_layer][batch_idx]

        num_evicted_rows = len(compact_layers) * len(seqs)
        dropped_tokens = sum(
            (kv_len - budget) * len(seqs) for kv_len in compact_lengths
        )

        if is_prefill:
            counter = (
                "final_prefill_evictions"
                if final_flags[0]
                else "intermediate_prefill_evictions"
            )
        else:
            counter = "decode_evictions"
        self._h2o_counters[counter] += int(num_evicted_rows)
        self._h2o_counters["dropped_tokens"] += int(dropped_tokens)
        return True

    def evict_after_intermediate_prefill(self, seqs: list[Sequence]) -> None:
        """Apply the prefill method only between prompt chunks."""

        intermediate = [seq for seq in seqs if not seq.is_last_chunk_prefill]
        if intermediate:
            self._evict(intermediate, is_prefill=True)

    def compact_final_prefill_for_decode(self, seqs: list[Sequence]) -> None:
        """Apply the decode method at the final prompt boundary."""

        final = [seq for seq in seqs if seq.is_last_chunk_prefill]
        if not final:
            return
        self._evict(final, is_prefill=True)
        for layer_idx in self.kv_transformer_layer_indices():
            for seq in final:
                kv_len = self._physical_row_len(layer_idx, seq)
                if kv_len > self.h2o_decode_budget:
                    raise RuntimeError(
                        "H2O final prefill did not compact to the decode budget: "
                        f"layer={layer_idx} seq_id={seq.seq_id} "
                        f"kv_len={kv_len} budget={self.h2o_decode_budget}."
                    )
        if self.num_free_slots <= 0:
            self._evict_decode_rows([])

    def evict_after_prefill(self, seqs: list[Sequence]) -> None:
        """Compatibility wrapper for the legacy combined H2O lifecycle."""

        self._evict(seqs, is_prefill=True)
        for layer_idx in self.kv_transformer_layer_indices():
            for seq in seqs:
                if not seq.is_last_chunk_prefill:
                    continue
                kv_len = self._physical_row_len(layer_idx, seq)
                if kv_len > self.h2o_decode_budget:
                    raise RuntimeError(
                        "H2O final prefill did not compact to the decode budget: "
                        f"layer={layer_idx} seq_id={seq.seq_id} "
                        f"kv_len={kv_len} budget={self.h2o_decode_budget}."
                    )
        if self.num_free_slots <= 0:
            self._evict_decode_rows([])

    def _decode_eviction_groups(
        self,
        seqs: list[Sequence],
    ) -> tuple[list[int], dict[int, list[Sequence | _H2ORowRef]]]:
        seq_ids = [int(seq.seq_id) for seq in seqs]
        if len(seq_ids) != len(set(seq_ids)):
            raise RuntimeError(
                f"H2O decode eviction received duplicate sequence ids: {seq_ids}."
            )
        self._h2o_active_decode_seq_ids.update(seq_ids)
        layer_indices = [int(layer) for layer in self.kv_transformer_layer_indices()]
        budget = self.h2o_decode_budget
        periodic_trigger = budget + self.h2o_decode_eviction_interval
        # The scheduler reads physical free capacity before the next forward.
        # If this step consumed the final slot, every active decode row that is
        # already over budget is reclaimable, including temporarily
        # unscheduled rows. Idle chain and prefill-only rows are not active.
        under_pressure = self.num_free_slots <= 0
        trigger = budget + 1 if under_pressure else periodic_trigger
        seq_by_id = {int(seq.seq_id): seq for seq in seqs}
        candidate_ids = list(seq_ids)
        if under_pressure:
            candidate_ids.extend(
                sorted(self._h2o_active_decode_seq_ids.difference(seq_ids))
            )
        groups: dict[int, list[Sequence | _H2ORowRef]] = {}
        for seq_id in candidate_ids:
            seq = seq_by_id.get(seq_id, _H2ORowRef(seq_id))
            kv_len = self._physical_row_len(layer_indices[0], seq)
            if self.validate_runtime_invariants:
                physical_lens = [kv_len]
                physical_lens.extend(
                    self._physical_row_len(layer_idx, seq)
                    for layer_idx in layer_indices[1:]
                )
                if any(length != kv_len for length in physical_lens[1:]):
                    raise RuntimeError(
                        "H2O decode eviction requires aligned KV-layer row lengths: "
                        f"seq_id={seq.seq_id} lengths={physical_lens}."
                    )
            if kv_len >= trigger:
                groups.setdefault(kv_len, []).append(seq)
        return layer_indices, groups

    def _preflight_decode_eviction_capacity(
        self,
        layer_indices: list[int],
        groups: dict[int, list[Sequence | _H2ORowRef]],
    ) -> None:
        dropped_per_layer = sum(
            (kv_len - self.h2o_decode_budget) * len(group_seqs)
            for kv_len, group_seqs in groups.items()
        )
        for layer_idx in layer_indices:
            free_stack = self.free_slots_stack[layer_idx]
            end = int(self._num_free_slots[layer_idx]) + int(dropped_per_layer)
            if end > int(free_stack.numel()):
                raise RuntimeError(
                    "H2O decode eviction would overflow the free-slot stack: "
                    f"layer={layer_idx} end={end} capacity={int(free_stack.numel())}."
                )

    def _evict_decode_rows(self, seqs: list[Sequence]) -> None:
        layer_indices, groups = self._decode_eviction_groups(seqs)
        if not groups:
            return
        self._invalidate_h2o_decode_score_workspace()
        self._preflight_decode_eviction_capacity(layer_indices, groups)

        budget = self.h2o_decode_budget
        ratio = float(self.config.h2o_recent_ratio)
        fast_path = self.buffer_req_to_token_slots_tensor is not None
        evicted_rows = 0
        dropped_tokens = 0
        burst_count = 0
        for kv_len, group_seqs in groups.items():
            burst_count += len(group_seqs)
            if fast_path:
                with profiler.record("h2o_decode_burst_select"):
                    scores = torch.stack(
                        [
                            self._require_score_length(layer_idx, seq, kv_len)
                            for layer_idx in layer_indices
                            for seq in group_seqs
                        ],
                        dim=0,
                    ).view(len(layer_indices), len(group_seqs), kv_len)
                    keep_indices = self.select_h2o_indices_batch(
                        scores.view(-1, kv_len),
                        budget=budget,
                        recent_ratio=ratio,
                    ).view(len(layer_indices), len(group_seqs), budget)
                    kept_scores = scores.gather(2, keep_indices)
                with profiler.record("h2o_decode_burst_compact_layers"):
                    SnapKVCacheManager.free_part_slots_batch_layers(
                        self,
                        layer_indices,
                        group_seqs,
                        keep_indices,
                        keep_indices_sorted=True,
                    )
                for local_layer, layer_idx in enumerate(layer_indices):
                    for batch_idx, seq in enumerate(group_seqs):
                        self._h2o_scores[
                            self._score_key(layer_idx, seq.seq_id)
                        ] = kept_scores[local_layer, batch_idx]
            else:
                with profiler.record("h2o_decode_burst_fallback"):
                    for layer_idx in layer_indices:
                        for seq in group_seqs:
                            score = self._require_score_length(
                                layer_idx, seq, kv_len
                            )
                            keep_indices = self.select_h2o_indices(
                                score,
                                budget=budget,
                                recent_ratio=ratio,
                            )
                            self.free_part_slots(
                                layer_idx,
                                seq,
                                keep_indices,
                                keep_indices_sorted=True,
                            )
            rows = len(layer_indices) * len(group_seqs)
            evicted_rows += rows
            dropped_tokens += (kv_len - budget) * rows

        self._h2o_counters["decode_eviction_bursts"] += int(burst_count)
        self._h2o_counters["decode_evictions"] += int(evicted_rows)
        self._h2o_counters["dropped_tokens"] += int(dropped_tokens)

    @cpu_timing.timed
    def evict_after_decode(self, seqs: list[Sequence]):
        if not seqs:
            return
        self._evict_decode_rows(seqs)

    def snapshot_chain_method_state(self, seq_id: int):
        state = super().snapshot_chain_method_state(seq_id)
        for layer in self.kv_transformer_layer_indices():
            state.tensors[str(layer)] = self._h2o_scores[self._score_key(layer, seq_id)]
        return state

    def restore_chain_method_state(self, seq_id: int, state) -> None:
        for layer in self.kv_transformer_layer_indices():
            self._h2o_scores[self._score_key(layer, seq_id)] = state.tensors[str(layer)].to(
                self.device, non_blocking=True,
            )

    def free_seq(self, seq_id: int):
        seq_id = int(seq_id)
        self._invalidate_h2o_decode_score_workspace()
        self._h2o_active_decode_seq_ids.discard(seq_id)
        for key in list(self._h2o_scores):
            if key[1] == seq_id:
                self._h2o_scores.pop(key, None)
        rows = getattr(self, "_h2o_decode_score_rows", {})
        for key in list(rows):
            if key[1] == seq_id:
                rows.pop(key)
        super().free_seq(seq_id)

    def on_chain_turn_finished(
        self,
        seq_id: int,
        processed_token_count: int,
    ) -> None:
        seq_id = int(seq_id)
        self._h2o_active_decode_seq_ids.discard(seq_id)
        for layer_idx in self.kv_transformer_layer_indices():
            row_idx = self.seq_id_to_row[layer_idx].get(seq_id)
            if row_idx is None:
                continue
            physical_len = int(self.row_seq_lens[layer_idx][row_idx])
            key = self._score_key(layer_idx, seq_id)
            score = self._h2o_scores.get(key)
            if score is None:
                raise RuntimeError(
                    "H2O chain turn finished without a score vector for a resident "
                    f"KV row: layer={layer_idx} seq_id={seq_id} "
                    f"physical_len={physical_len}."
                )
            self._h2o_scores[key] = self._expand_score(
                score,
                physical_len,
                device=score.device,
            )
        super().on_chain_turn_finished(seq_id, processed_token_count)

    def reset_after_warmup(self) -> None:
        self._h2o_scores.clear()
        getattr(self, "_h2o_decode_score_rows", {}).clear()
        self._h2o_decode_score_metadata = None
        self._h2o_active_decode_seq_ids.clear()
        self._h2o_decode_score_workspace = None
        self._h2o_decode_score_signature = None
        self._h2o_decode_score_length = 0
        for counter in self._h2o_counters:
            self._h2o_counters[counter] = 0

    def debug_state_summary(self) -> dict[str, object]:
        summary = super().debug_state_summary()
        workspace = getattr(self, "_h2o_final_prefill_workspace", None)
        summary["h2o"] = {
            "counters": dict(self._h2o_counters),
            "score_lengths": {
                f"{layer_idx}:{seq_id}": int(score.numel())
                for (layer_idx, seq_id), score in sorted(self._h2o_scores.items())
            },
            "final_prefill_workspace": (
                None
                if workspace is None
                else {
                    "shape": list(workspace.shape),
                    "dtype": str(workspace.dtype),
                    "device": str(workspace.device),
                    "nbytes": int(workspace.untyped_storage().nbytes()),
                }
            ),
        }
        return summary
