from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from sparseengine.config import Config
from sparseengine.distributed import ParallelContext
from sparseengine.engine.sequence import Sequence
from sparseengine.engine.prefill import (
    PREFILL_EXECUTION_CHUNKED,
    PREFILL_EXECUTION_FULL,
    PREFILL_EXECUTION_RAW_OFFLOAD,
)
from sparseengine.method_registry import (
    PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
)
from sparseengine.utils.context import get_context
from sparseengine.utils.log import logger, log_level
from sparseengine.utils.profiler import profiler
from sparseengine.layers.rotary_embedding import get_rope, apply_rotary_emb
from sparseengine.platforms import device_runtime

from ..base import (
    AttentionViewMeta,
    CacheManager,
    DecodeComputeView,
    ExplicitKVPayload,
    LayerBatchStates,
    PrefillComputeView,
    SparseSelection,
)
from ..raw_kv_offload import RawKVOffloadBuffer


@dataclass(frozen=True)
class DeltaKVFullPrefillPlanCPU:
    total_len: int
    sink_len: int
    evict_start: int
    evict_end: int
    evict_len: int
    keep_positions: tuple[int, ...]
    center_positions: tuple[int, ...]
    latent_positions: tuple[int, ...]


class DeltaKVCacheManager(CacheManager):
    @staticmethod
    def _get_rope_theta(hf_config) -> float:
        if hasattr(hf_config, "rope_theta"):
            return float(hf_config.rope_theta)
        rope_parameters = getattr(hf_config, "rope_parameters", None)
        if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
            return float(rope_parameters["rope_theta"])
        return 10000.0

    @staticmethod
    def _normalize_rope_scaling(hf_config) -> tuple[tuple[str, object], ...] | None:
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if rope_scaling is None:
            return None
        if isinstance(rope_scaling, dict):
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
            is_default_rope = rope_type in (None, "default")
            allowed_default_keys = {"rope_type", "type", "rope_theta"}
            if is_default_rope and set(rope_scaling).issubset(allowed_default_keys):
                return None
            if rope_type == "llama3":
                required = {
                    "factor",
                    "low_freq_factor",
                    "high_freq_factor",
                    "original_max_position_embeddings",
                }
                missing = sorted(required.difference(rope_scaling))
                if missing:
                    raise ValueError(f"Llama3 rope_scaling missing required keys: {missing}.")
                return tuple(sorted(rope_scaling.items()))
        raise NotImplementedError(f"Unsupported DeltaKV cache RoPE scaling: {rope_scaling!r}.")

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
        assert self.tp_size == 1, "DeltaKVCacheManager currently does not support TP compressors"

        self.full_attention_layers = config.full_attention_layers
        if not isinstance(self.full_attention_layers, list) or not all(
            isinstance(layer_idx, int) for layer_idx in self.full_attention_layers
        ):
            raise ValueError("DeltaKV full_attention_layers must be a list of transformer layer indices.")
        if not self.full_attention_layers:
            raise ValueError("DeltaKV requires at least one configured full_attention_layers entry.")
        kv_layer_ids = list(self.kv_transformer_layer_indices())
        kv_layer_set = set(kv_layer_ids)
        invalid_full_layers = sorted(set(self.full_attention_layers) - kv_layer_set)
        if invalid_full_layers:
            raise ValueError(
                "DeltaKV full_attention_layers must refer to full-attention KV layers only; "
                f"invalid_or_linear_layers={invalid_full_layers} kv_layers={kv_layer_ids}."
            )
        self.deltakv_layer_ids = [i for i in kv_layer_ids if i not in self.full_attention_layers]
        self.full_layer_ids = [i for i in kv_layer_ids if i in self.full_attention_layers]
        self.deltakv_layer_to_idx = {l: i for i, l in enumerate(self.deltakv_layer_ids)}
        self.full_layer_to_idx = {l: i for i, l in enumerate(self.full_layer_ids)}


        self.full_num_slots = 0
        self.deltakv_latent_num_slots = 0
        self.deltakv_full_num_slots = 0
        self.deltakv_prefill_staging_num_slots = 0
        self.full_kv_cache = None
        self.deltakv_full_kv_cache = None
        self.deltakv_prefill_staging_kv_cache = None
        self.deltakv_latent_cache = None
        self.deltakv_latent_to_full_slots = None
        self.deltakv_slot_to_pos = None
        # Reserved decode-reconstruction scratch slots in the sparse full-KV pool.
        # This is set in allocate_kv_cache() and used to provide backpressure to Scheduler
        # (so requests wait instead of crashing inside attention).
        self._deltakv_decode_reconstruct_full_reserve = 0
        self._deltakv_temp_full_reserve = 0
        self._deltakv_static_temp_slots_reserved_total = 0
        # Budgeting for centers: we reserve "future center slots" at admission time to avoid
        # admitting more long prompts than the sparse full-KV pool can sustain.
        self._deltakv_centers_capacity = 0
        self._deltakv_centers_reserved_total = 0
        self._deltakv_centers_reserved_by_seq: dict[int, int] = {}
        self._deltakv_latent_reserved_total = 0
        self._deltakv_latent_reserved_by_seq: dict[int, int] = {}
        self._full_layers_reserved_total = 0
        self._full_layers_reserved_by_seq: dict[int, int] = {}
        self._full_layer_kivi_reserved_total = 0
        self._full_layer_kivi_reserved_by_seq: dict[int, int] = {}

        self.allocate_kv_cache()

        self.free_slots_stack_full = torch.arange(self.full_num_slots, dtype=torch.int32, device=self.device)
        self._num_free_slots_full = self.full_num_slots

        self.free_slots_stack_deltakv_full = torch.arange(self.deltakv_full_num_slots, dtype=torch.int32, device=self.device)
        self._num_free_slots_deltakv_full = self.deltakv_full_num_slots

        self.free_slots_stack_deltakv_latent = torch.arange(self.deltakv_latent_num_slots, dtype=torch.int32, device=self.device)
        self._num_free_slots_deltakv_latent = self.deltakv_latent_num_slots

        self.full_layer_slots_map = torch.zeros(
            (self.max_buffer_rows, self.max_model_len), dtype=torch.int32, device=self.device
        )
        self.sparse_layer_raw_slots_map = torch.full(
            (self.max_buffer_rows, self.max_model_len), -1, dtype=torch.int32, device=self.device
        )
        self.sparse_layer_latent_slots_map = torch.full(
            (self.max_buffer_rows, self.max_model_len), -1, dtype=torch.int32, device=self.device
        )

        self.seq_id_to_row: dict[int, int] = {}
        self.free_rows = deque(range(self.max_buffer_rows))
        self.row_seq_lens = np.zeros((self.max_buffer_rows,), dtype=np.int32)
        self.row_deltakv_compressed_lens = np.zeros((self.max_buffer_rows,), dtype=np.int32)
        self.row_deltakv_compressed_lens_gpu = torch.zeros(
            (self.max_buffer_rows,), dtype=torch.int32, device=self.device
        )
        self.row_deltakv_center_slots = [[None for _ in range(self.num_layers)] for _ in range(self.max_buffer_rows)]

        self.full_layer_batch_states = LayerBatchStates()
        self.deltakv_layer_batch_states = LayerBatchStates()

        num_deltakv_layers = len(self.deltakv_layer_ids)
        self._init_compressor_modules(config, num_deltakv_layers)


        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_model_len,
            base=self._get_rope_theta(self.hf_config),
            rope_scaling=self._normalize_rope_scaling(self.hf_config),
        ).to(device=self.device)

        self.cos_sin_cache = self.rotary_emb.cos_sin_cache

        # Per-step/per-segment cache for DeltaKV view planning (shared across layers).
        self._deltakv_view_cache_key: tuple[int, int, int, int, int] | None = None
        self._deltakv_view_cache_value = None
        self._deltakv_prefill_staging_active = False
        self._deltakv_prefill_staging_slot_mapping = None
        self._deltakv_prefill_staging_active_slots = None
        self._deltakv_prefill_staging_req_indices = None
        self._deltakv_prefill_staging_context_lens = None
        self._deltakv_full_prefill_plans: dict[int, dict[str, torch.Tensor | int]] = {}
        self._deltakv_full_prefill_compressed_layers: set[int] = set()
        self.raw_kv_offload_buffer = RawKVOffloadBuffer(pin_memory=device_runtime.supports_pin_memory())
        self._deltakv_long_prefill_offload_step_active = False
        self._deltakv_long_prefill_offload_row_idx: int | None = None
        self._deltakv_long_prefill_offload_start = 0
        self._deltakv_long_prefill_offload_end = 0
        self._deltakv_long_prefill_offload_total_len = 0
        self._deltakv_long_prefill_offload_is_last_chunk = False

    def _init_compressor_modules(self, config: Config, num_deltakv_layers: int):
        from sparseengine.utils.compressor import create_compressor

        self.compress_down = []
        self.compress_up = []
        for _ in range(num_deltakv_layers):
            self.compress_down.append(create_compressor(is_down=True, config=config).to(device=self.device))
            self.compress_up.append(create_compressor(is_down=False, config=config).to(device=self.device))

    def _deltakv_reset_view_cache(self):
        self._deltakv_view_cache_key = None
        self._deltakv_view_cache_value = None

    def _deltakv_reset_full_prefill_staging(self, *, clear_plans: bool = True):
        self._deltakv_prefill_staging_active = False
        self._deltakv_prefill_staging_slot_mapping = None
        self._deltakv_prefill_staging_active_slots = None
        self._deltakv_prefill_staging_req_indices = None
        self._deltakv_prefill_staging_context_lens = None
        if clear_plans:
            self._deltakv_full_prefill_plans = {}
        self._deltakv_full_prefill_compressed_layers = set()
        self._deltakv_long_prefill_offload_row_idx = None
        self._deltakv_long_prefill_offload_start = 0
        self._deltakv_long_prefill_offload_end = 0
        self._deltakv_long_prefill_offload_total_len = 0
        self._deltakv_long_prefill_offload_is_last_chunk = False

    def _use_decode_static_paths(self) -> bool:
        if get_context().is_prefill:
            return False
        config = getattr(self, "config", None)
        if config is None:
            return True
        return bool(getattr(config, "decode_graph", False)) or (
            self._is_stream_capturing()
        )

    def _deltakv_prefill_staging_capacity(self) -> int:
        return int(self.config.max_model_len)

    def _max_decode_scratch_seqs(self) -> int:
        return self.decode_scratch_sequence_capacity(self.config)

    @staticmethod
    def decode_scratch_sequence_capacity(config) -> int:
        max_seqs = max(int(config.max_num_seqs_in_batch), int(config.max_decoding_seqs))
        if bool(getattr(config, "decode_graph", False)):
            capture_sizes = getattr(config, "decode_graph_capture_sizes", None) or []
            if capture_sizes:
                max_seqs = max(max_seqs, max(int(size) for size in capture_sizes))
        return max_seqs

    def _deltakv_full_prefill_recent_tokens(self) -> int:
        return int(self.config.recent_keep_tokens)

    def _deltakv_base_cluster_step(self) -> int:
        deltakv_center_ratio = float(self.config.deltakv_center_ratio or 0.0)
        if deltakv_center_ratio <= 0.0:
            raise ValueError(f"DeltaKV deltakv_center_ratio must be > 0, got {deltakv_center_ratio}.")
        return max(1, int(1.0 / max(1e-6, deltakv_center_ratio)))

    @staticmethod
    def _deltakv_center_positions_cpu(
        *,
        start: int,
        end: int,
        base_step: int,
    ) -> tuple[int, ...]:
        start = int(start)
        end = int(end)
        base_step = max(1, int(base_step))
        if end <= start:
            return ()
        return tuple(range(start, end, base_step))

    def _deltakv_center_positions_for_block_cpu(
        self,
        row_idx: int,
        *,
        start: int,
        end: int,
        update_state: bool,
    ) -> tuple[int, ...]:
        positions = self._deltakv_center_positions_cpu(
            start=start,
            end=end,
            base_step=self._deltakv_base_cluster_step(),
        )
        return positions

    def _deltakv_center_rel_for_block(
        self,
        row_idx: int,
        *,
        start: int,
        end: int,
        update_state: bool,
    ) -> torch.Tensor:
        positions = self._deltakv_center_positions_for_block_cpu(
            row_idx,
            start=start,
            end=end,
            update_state=update_state,
        )
        if not positions:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        rel = [pos - int(start) for pos in positions]
        return torch.tensor(rel, dtype=torch.long, device=self.device)

    def _should_use_full_prefill_staging(self, seqs: list[Sequence]) -> bool:
        # Base DeltaKV has no middle full-prefill region: prompts above the
        # chunk boundary use RawKV offload. Subclasses may stage short prefills
        # for method-specific storage formats.
        return False

    def _long_prefill_offload_threshold(self) -> int:
        return int(self.config.long_prefill_offload_threshold)

    @staticmethod
    def _has_attached_prefill_prefix(seq: Sequence) -> bool:
        if bool(getattr(seq, "is_recompute_replay", False)):
            return False
        return (
            getattr(seq, "chain_status", "") == "resumed"
            or int(getattr(seq, "chain_reused_tokens", 0) or 0) > 0
            or int(getattr(seq, "prefix_cache_hit_len", 0) or 0) > 0
        )

    def prefill_execution_mode(self, seq: Sequence) -> str:
        if (
            getattr(self.config, "prefill_schedule_policy", None)
            != PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH
        ):
            return PREFILL_EXECUTION_CHUNKED
        if self._has_attached_prefill_prefix(seq):
            attached = max(
                int(getattr(seq, "chain_reused_tokens", 0) or 0),
                int(getattr(seq, "prefix_cache_hit_len", 0) or 0),
            )
            raise RuntimeError(
                "DeltaKV does not support attached-prefix prefill. Its compressed "
                "and quantized row state has no chain/radix residency contract, so "
                "routing an attached prefix through full or raw_offload staging "
                "would overwrite resident cache metadata. "
                f"seq_id={seq.seq_id} attached_prefix_tokens={attached}."
            )
        residual = int(self.remaining_prefill_tokens(seq))
        if residual <= 0:
            raise ValueError(
                "DeltaKV prefill execution mode requires a positive residual: "
                f"seq_id={seq.seq_id} residual={residual}."
            )
        initial_mode = (
            PREFILL_EXECUTION_FULL
            if residual <= self._long_prefill_offload_threshold()
            else PREFILL_EXECUTION_RAW_OFFLOAD
        )
        return self._apply_sticky_raw_offload_mode(seq, initial_mode)

    def requires_long_prefill_offload(self, seq: Sequence) -> bool:
        return self.prefill_execution_mode(seq) == PREFILL_EXECUTION_RAW_OFFLOAD

    def _should_use_long_prefill_offload_staging(self, seqs: list[Sequence]) -> bool:
        if not self.deltakv_layer_ids or len(seqs) != 1:
            return False
        seq = seqs[0]
        return self.requires_long_prefill_offload(seq) and int(seq.current_chunk_size or 0) > 0

    def requires_full_prefill_step(self, seq: Sequence) -> bool:
        return self.prefill_execution_mode(seq) == PREFILL_EXECUTION_FULL

    def defer_prefill_eviction(self) -> bool:
        return bool(getattr(self, "_deltakv_long_prefill_offload_step_active", False))

    @staticmethod
    def _deltakv_full_prefill_plan_cpu(
        total_len: int,
        *,
        sink: int,
        recent: int,
        cluster_step: int,
    ) -> DeltaKVFullPrefillPlanCPU:
        total_len = max(0, int(total_len))
        sink = max(0, int(sink))
        recent = max(0, int(recent))
        cluster_step = max(1, int(cluster_step))
        sink_len = min(sink, total_len)
        buffer_start = sink_len
        buffer_len = total_len - buffer_start
        if buffer_len <= recent:
            keep = tuple(range(total_len))
            return DeltaKVFullPrefillPlanCPU(
                total_len=total_len,
                sink_len=sink_len,
                evict_start=buffer_start,
                evict_end=buffer_start,
                evict_len=0,
                keep_positions=keep,
                center_positions=(),
                latent_positions=(),
            )

        evict_len = ((buffer_len - recent) // max(1, recent)) * max(1, recent) if recent > 0 else buffer_len
        evict_start = buffer_start
        evict_end = evict_start + evict_len
        center_positions = DeltaKVCacheManager._deltakv_center_positions_cpu(
            start=evict_start,
            end=evict_end,
            base_step=cluster_step,
        )
        # DeltaKV stores compressed payloads for the entire finalized
        # history block, including center tokens. Centers also keep raw slots as
        # future references, but attention should reconstruct them from latent
        # payloads when they are selected from compressed history.
        latent_positions = tuple(range(evict_start, evict_end))
        keep_positions = (
            tuple(range(sink_len))
            + center_positions
            + tuple(range(evict_end, total_len))
        )
        return DeltaKVFullPrefillPlanCPU(
            total_len=total_len,
            sink_len=sink_len,
            evict_start=evict_start,
            evict_end=evict_end,
            evict_len=evict_len,
            keep_positions=keep_positions,
            center_positions=center_positions,
            latent_positions=latent_positions,
        )

    def _deltakv_latent_payload_dim(self) -> int:
        payload_dim = int(getattr(self.config, "deltakv_latent_dim", 0) or 0)
        if payload_dim <= 0:
            raise ValueError(f"DeltaKV deltakv_latent_dim must be positive, got {payload_dim}.")
        return payload_dim

    def _store_deltakv_latent(self, l_idx: int, latent_slots: torch.Tensor, latent: torch.Tensor):
        if int(latent_slots.numel()) != int(latent.shape[0]):
            raise RuntimeError(
                "DeltaKV latent store shape mismatch: "
                f"slots={int(latent_slots.numel())}, latent_rows={int(latent.shape[0])}."
            )
        capturing = self._is_stream_capturing()
        if not capturing:
            latent_cap = int(self.deltakv_latent_cache.shape[1])
            bad_latent = (latent_slots < 0) | (latent_slots >= latent_cap)
            if bool(bad_latent.any()):
                bad = latent_slots[bad_latent][:16].detach().cpu().tolist()
                raise RuntimeError(
                    "DeltaKV latent store got slot outside cache: "
                    f"cap={latent_cap}, bad={bad}."
                )

        self.deltakv_latent_cache[l_idx, latent_slots] = latent.to(self.deltakv_latent_cache.dtype)

    def _load_deltakv_latent(self, l_idx: int, latent_slots: torch.Tensor) -> torch.Tensor:
        return self.deltakv_latent_cache[l_idx, latent_slots]

    def prepare_step(self, seqs: list[Sequence], is_prefill: bool):
        # Reset per-step cache to avoid stale views across steps.
        self._deltakv_reset_view_cache()
        use_long_prefill_offload = bool(is_prefill and self._should_use_long_prefill_offload_staging(seqs))
        self._deltakv_long_prefill_offload_step_active = use_long_prefill_offload
        self._deltakv_reset_full_prefill_staging(clear_plans=not use_long_prefill_offload)
        self._deltakv_long_prefill_offload_step_active = use_long_prefill_offload
        return super().prepare_step(seqs, is_prefill)

    def allocate_kv_cache(self):
        available_memory, slot_bytes_per_layer = self._get_available_slots_info()
        config = self.config
        dtype_size = torch.tensor([], dtype=self.hf_config.dtype).element_size()
        self.deltakv_prefill_staging_num_slots = self._deltakv_prefill_staging_capacity()
        prefill_staging_bytes = int(self.deltakv_prefill_staging_num_slots) * int(slot_bytes_per_layer)
        persistent_memory = int(available_memory) - int(prefill_staging_bytes)
        if persistent_memory <= 0:
            raise RuntimeError(
                "Not enough GPU memory for DeltaKV prefill staging KV. "
                f"staging_slots={self.deltakv_prefill_staging_num_slots} "
                f"required={prefill_staging_bytes / 1024**3:.2f}GiB "
                f"available={available_memory / 1024**3:.2f}GiB."
            )

        num_full_layers = len(self.full_layer_ids)
        num_deltakv_layers = len(self.deltakv_layer_ids)
        assert num_full_layers > 0, "DeltaKV requires at least one full-attention layer."
        assert num_deltakv_layers > 0, "DeltaKV requires at least one sparse layer."

        # Full layers store all tokens. Sparse layers store:
        # - latent for all tokens (for reconstruction)
        # - a bounded full-KV pool: centers + uncompressed buffer (+ current chunk) + reconstructed top tokens.
        latent_payload_dim = self._deltakv_latent_payload_dim()
        latent_bytes = latent_payload_dim * dtype_size
        deltakv_center_ratio = max(0.0, float(config.deltakv_center_ratio))

        per_token_bytes = (
            num_full_layers * slot_bytes_per_layer
            + num_deltakv_layers * (deltakv_center_ratio * slot_bytes_per_layer + latent_bytes)
        )
        if per_token_bytes <= 0:
            raise ValueError("Invalid KV cache allocation configuration.")

        max_tokens = max(1, int(persistent_memory / per_token_bytes))

        # Reserve some headroom for the sparse full-KV pool (centers/buffer/temp).
        # This is important for large batch sizes, where the required temp slots can spike.
        reserve_ratio = float(config.deltakv_full_pool_reserve_ratio)
        if reserve_ratio > 0:
            reserve_ratio = max(0.0, min(0.5, reserve_ratio))
            max_tokens = max(1, int(max_tokens * (1.0 - reserve_ratio)))
        self.full_num_slots = max_tokens
        self.deltakv_latent_num_slots = max_tokens

        # Now decide the sparse full-KV pool size from remaining bytes.
        bytes_full_layers = self.full_num_slots * num_full_layers * slot_bytes_per_layer
        bytes_latent = self.deltakv_latent_num_slots * num_deltakv_layers * latent_bytes
        bytes_misc = 0  # small tensors (slot maps) are negligible vs KV
        bytes_left = persistent_memory - bytes_full_layers - bytes_latent - bytes_misc
        if bytes_left <= 0:
            raise RuntimeError(
                "Not enough GPU memory left for DeltaKV full-KV pool after allocating full layers + latent cache. "
                "Try reducing max_model_len / gpu_memory_utilization / deltakv_latent_dim."
            )
        max_deltakv_full_slots = max(1, int(bytes_left // (num_deltakv_layers * slot_bytes_per_layer)))

        sink = int(config.sink_keep_tokens)
        recent = int(config.recent_keep_tokens)
        # Sparse full-KV pool must cover:
        # - per-seq resident tokens: sink + (<=2*recent) uncompressed buffer
        # - current prefill step's chunk tokens across the whole batch
        # - temp reconstructed top tokens (per seq, per sparse layer attention)
        #
        # If we under-estimate this, the system should backpressure at scheduling time
        # (queue) rather than crashing in _allocate_temp_deltakv_full().
        max_seqs = self._max_decode_scratch_seqs()
        max_admission_seqs = int(config.max_num_seqs_in_gpu)
        top_tokens = int(config.decode_keep_tokens)
        # Worst-case total reconstructed top tokens within a single attention call:
        #   num_seqs_in_batch * top_k_per_seq
        # For prefill, num_seqs_in_batch is also bounded by (max_num_batched_tokens / chunk_size)
        # when chunks are full; cap by max_seqs to avoid over-estimation.
        max_prefill_seqs_by_tokens = max(
            1,
            int(config.max_num_batched_tokens) // int(config.engine_prefill_chunk_size),
        )
        max_prefill_seqs = min(max_admission_seqs, max_prefill_seqs_by_tokens)
        total_top_slots = max(max_seqs * top_tokens, max_prefill_seqs * top_tokens)
        max_step_chunk = int(min(int(config.max_num_batched_tokens), max_prefill_seqs * int(config.engine_prefill_chunk_size)))
        overhead_slots = max_admission_seqs * (sink + 2 * recent) + total_top_slots + max_step_chunk
        if max_deltakv_full_slots <= overhead_slots:
            raise RuntimeError(
                f"DeltaKV full-KV pool too small: max={max_deltakv_full_slots}, required>={overhead_slots + 1}. "
                "Reduce engine_prefill_chunk_size/decode_keep_tokens/recent_keep_tokens or increase gpu_memory_utilization."
            )

        desired_centers = max(1, int(deltakv_center_ratio * self.full_num_slots * 1.5))
        centers_capacity = min(desired_centers, max_deltakv_full_slots - overhead_slots)
        self.deltakv_full_num_slots = overhead_slots + centers_capacity
        self._deltakv_centers_capacity = int(centers_capacity)
        # Reserve scratch capacity for reconstruction. Scheduler-visible free slots will exclude this
        # so requests wait instead of triggering temp-slot OOM mid-forward.
        self._deltakv_decode_reconstruct_full_reserve = min(self.deltakv_full_num_slots, int(total_top_slots))
        self._deltakv_temp_full_reserve = self._deltakv_decode_reconstruct_full_reserve

        logger.info(
            f"DeltaKV allocation: full_layers_slots={self.full_num_slots}; "
            f"deltakv_full_slots={self.deltakv_full_num_slots} (overhead={overhead_slots}, centers={centers_capacity}); "
            f"deltakv_latent_slots={self.deltakv_latent_num_slots} "
            f"(full_layers={num_full_layers}, deltakv_layers={num_deltakv_layers}, "
            f"deltakv_full_pool_reserve_ratio={reserve_ratio:.3f}, "
            f"deltakv_decode_reconstruct_full_reserve={self._deltakv_temp_full_reserve}, "
            f"deltakv_prefill_staging_slots={self.deltakv_prefill_staging_num_slots}, "
            f"latent_payload_dim={latent_payload_dim})."
        )

        self.full_kv_cache = torch.empty(
            2,
            num_full_layers,
            self.full_num_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.dtype,
            device=self.device,
        )

        self.deltakv_full_kv_cache = torch.empty(
            2,
            num_deltakv_layers,
            self.deltakv_full_num_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.dtype,
            device=self.device,
        )
        self._deltakv_postrope_slot_marker = torch.zeros(
            num_deltakv_layers,
            self.deltakv_full_num_slots,
            dtype=torch.int32,
            device=self.device,
        )
        self.deltakv_prefill_staging_kv_cache = torch.empty(
            2,
            self.deltakv_prefill_staging_num_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.dtype,
            device=self.device,
        )
        self.deltakv_latent_cache = torch.empty(
            num_deltakv_layers,
            self.deltakv_latent_num_slots,
            latent_payload_dim,
            dtype=self.hf_config.dtype,
            device=self.device,
        )
        self.deltakv_latent_to_full_slots = torch.full(
            (num_deltakv_layers, self.deltakv_latent_num_slots, config.deltakv_neighbor_count),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.deltakv_slot_to_pos = torch.full(
            (self.deltakv_full_num_slots,),
            -1,
            dtype=torch.int32,
            device=self.device,
        )

    def get_layer_batch_states(self, layer_idx: int) -> LayerBatchStates:
        if layer_idx in self.full_attention_layers:
            return self.full_layer_batch_states
        else:
            return self.deltakv_layer_batch_states

    def get_layer_kv_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx in self.full_layer_to_idx:
            idx = self.full_layer_to_idx[layer_idx]
            return self.full_kv_cache[0, idx], self.full_kv_cache[1, idx]
        else:
            idx = self.deltakv_layer_to_idx[layer_idx]
            return self.deltakv_full_kv_cache[0, idx], self.deltakv_full_kv_cache[1, idx]

    def get_layer_store_view(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.has_prefill_staging_view(layer_idx):
            return (
                self.deltakv_prefill_staging_kv_cache[0],
                self.deltakv_prefill_staging_kv_cache[1],
                self._deltakv_prefill_staging_slot_mapping,
            )
        k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        state = self.get_layer_batch_states(layer_idx)
        return k_cache, v_cache, state.slot_mapping

    def _stores_sparse_raw_kv(self, layer_idx: int) -> bool:
        return layer_idx in self.deltakv_layer_to_idx

    def _collect_k_norm_weights(self, layers, layer_ids: list[int]):
        weights = []
        eps = 1e-6
        for layer_idx in layer_ids:
            attn = getattr(layers[layer_idx], "self_attn", None)
            k_norm = getattr(attn, "k_norm", None)
            if k_norm is None:
                return None, None
            weights.append(
                k_norm.weight.detach()
                .to(device=self.device, dtype=self.hf_config.dtype)
                .clone()
            )
            eps = float(getattr(k_norm, "eps", 1e-6))
        return torch.stack(weights, dim=0) if weights else None, eps

    def set_model_layers(self, layers):
        self.deltakv_k_norm_weight, self.deltakv_k_norm_eps = self._collect_k_norm_weights(
            layers,
            self.deltakv_layer_ids,
        )

    def _apply_sparse_k_norm_if_needed(self, l_idx: int, key: torch.Tensor) -> torch.Tensor:
        weight = getattr(self, "deltakv_k_norm_weight", None)
        if weight is None:
            return key
        orig_dtype = key.dtype
        x = key.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + float(getattr(self, "deltakv_k_norm_eps", 1e-6) or 1e-6))
        return x.to(orig_dtype) * weight[int(l_idx)].to(dtype=orig_dtype)

    def _apply_sparse_rope_to_key(self, positions: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        cos_sin = self.rotary_emb.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        return apply_rotary_emb(key, cos, sin)

    def get_layer_store_tensors(
        self,
        layer_idx: int,
        *,
        k_post_rope: torch.Tensor,
        v: torch.Tensor,
        pre_rope_k: torch.Tensor | None = None,
        pre_rope_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._stores_sparse_raw_kv(layer_idx):
            return k_post_rope, v
        source_k = pre_rope_k
        if source_k is None:
            raise RuntimeError("DeltaKV sparse raw storage requires pre-RoPE key states.")
        source_v = pre_rope_v if pre_rope_v is not None else v
        if int(source_k.shape[0]) != int(k_post_rope.shape[0]) or int(source_v.shape[0]) != int(v.shape[0]):
            raise RuntimeError(
                "DeltaKV sparse raw storage shape mismatch: "
                f"k_raw={tuple(source_k.shape)} k_post_rope={tuple(k_post_rope.shape)} "
                f"v_raw={tuple(source_v.shape)} v={tuple(v.shape)}."
            )
        return source_k, source_v

    def save_raw_kv_if_needed(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        if self._stores_sparse_raw_kv(layer_idx):
            self._store_layer_kv(layer_idx, k, v)

    def save_rope_kv_if_needed(
        self,
        layer_idx: int,
        k_post_rope: torch.Tensor,
        v: torch.Tensor,
    ):
        if self._stores_sparse_raw_kv(layer_idx):
            return None
        return super().save_rope_kv_if_needed(
            layer_idx,
            k_post_rope,
            v,
        )

    def _already_postrope_mask(
        self,
        layer_idx: int,
        slots: torch.Tensor,
        already_postrope_slots: torch.Tensor,
    ) -> torch.Tensor:
        valid = slots >= 0
        if already_postrope_slots.numel() == 0:
            return torch.zeros_like(valid, dtype=torch.bool)
        markers = getattr(self, "_deltakv_postrope_slot_marker", None)
        if markers is None:
            raise RuntimeError("DeltaKV post-RoPE slot marker is not initialized.")
        l_idx = self.deltakv_layer_to_idx[int(layer_idx)]
        marker = markers[l_idx]
        marker.zero_()

        cap = int(marker.shape[0])
        already = already_postrope_slots.to(device=marker.device, dtype=torch.int32).flatten()
        valid_already = (already >= 0) & (already < cap)
        safe_already = already.clamp(0, cap - 1).to(torch.long)
        marker.scatter_reduce_(
            0,
            safe_already,
            valid_already.to(torch.int32),
            reduce="amax",
            include_self=True,
        )

        safe_slots = slots.to(device=marker.device, dtype=torch.long).clamp(0, cap - 1)
        return valid & (marker[safe_slots] > 0)

    def _materialize_deltakv_active_postrope_view(
        self,
        layer_idx: int,
        active_slots: torch.Tensor,
        new_context_lens: torch.Tensor,
        already_postrope_slots: torch.Tensor,
        active_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize raw sparse-layer K slots into post-RoPE scratch slots for attention."""
        if active_slots.numel() == 0:
            return active_slots, torch.empty((0,), device=active_slots.device, dtype=torch.int32)
        if self._use_decode_static_paths():
            return self._materialize_deltakv_active_postrope_view_static(
                layer_idx,
                active_slots,
                new_context_lens,
                already_postrope_slots,
                active_pos,
            )

        l_idx = self.deltakv_layer_to_idx[layer_idx]
        k_cache = self.deltakv_full_kv_cache[0, l_idx]
        v_cache = self.deltakv_full_kv_cache[1, l_idx]
        materialized: list[torch.Tensor] = []
        for b in range(int(active_slots.shape[0])):
            view_len = int(new_context_lens[b].item())
            if view_len <= 0:
                continue
            src_slots = active_slots[b, :view_len].to(torch.int32)
            if (src_slots < 0).any():
                raise RuntimeError(f"DeltaKV sparse view contains a negative slot, layer={layer_idx}.")
            already = self._already_postrope_mask(layer_idx, src_slots, already_postrope_slots)
            raw_idx = torch.nonzero(~already, as_tuple=False).flatten()
            if raw_idx.numel() == 0:
                continue

            raw_slots = src_slots.index_select(0, raw_idx)
            pos = self.deltakv_slot_to_pos[raw_slots.to(torch.long)].to(torch.long)
            if (pos < 0).any():
                raise RuntimeError(f"DeltaKV sparse raw slot has unknown position, layer={layer_idx}.")

            out_slots = self._allocate_temp_deltakv_full(int(raw_idx.numel())).to(torch.int32)
            cos_sin = self.cos_sin_cache[pos]
            cos, sin = cos_sin.chunk(2, dim=-1)
            k_normed = self._apply_sparse_k_norm_if_needed(l_idx, k_cache[raw_slots.to(torch.long)])
            k_postrope = apply_rotary_emb(k_normed, cos, sin)
            out_i64 = out_slots.to(torch.long)
            k_cache[out_i64] = k_postrope.to(k_cache.dtype)
            v_cache[out_i64] = v_cache[raw_slots.to(torch.long)].to(v_cache.dtype)
            self.deltakv_slot_to_pos[out_i64] = pos.to(torch.int32)
            active_slots[b, raw_idx] = out_slots.to(active_slots.dtype)
            materialized.append(out_slots)

        if not materialized:
            return active_slots, torch.empty((0,), device=active_slots.device, dtype=torch.int32)
        return active_slots, torch.cat(materialized, dim=0).to(torch.int32)

    def _ensure_decode_static_materialized_slots(self, active_slots: torch.Tensor) -> torch.Tensor:
        key = (tuple(active_slots.shape), str(active_slots.device))
        cache = getattr(self, "_deltakv_decode_static_materialized_slots_by_shape", None)
        if cache is None:
            cache = {}
            self._deltakv_decode_static_materialized_slots_by_shape = cache
        slots = cache.get(key)
        if slots is not None:
            return slots
        total = int(active_slots.numel())
        if total == 0:
            slots = torch.empty((0,), device=active_slots.device, dtype=torch.int32)
        else:
            slots = self._allocate_temp_deltakv_full(total).to(torch.int32)
            self._deltakv_static_temp_slots_reserved_total = int(
                getattr(self, "_deltakv_static_temp_slots_reserved_total", 0) or 0
            ) + total
        cache[key] = slots
        return slots

    def _materialize_deltakv_active_postrope_view_static(
        self,
        layer_idx: int,
        active_slots: torch.Tensor,
        new_context_lens: torch.Tensor,
        already_postrope_slots: torch.Tensor,
        active_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        l_idx = self.deltakv_layer_to_idx[layer_idx]
        k_cache = self.deltakv_full_kv_cache[0, l_idx]
        v_cache = self.deltakv_full_kv_cache[1, l_idx]

        flat_active = active_slots.reshape(-1).to(torch.int32)
        out_slots = self._ensure_decode_static_materialized_slots(active_slots).to(torch.int32)
        if flat_active.numel() == 0:
            return active_slots, torch.empty((0,), device=active_slots.device, dtype=torch.int32)

        if active_slots.dim() == 2:
            cols = torch.arange(active_slots.shape[1], device=active_slots.device, dtype=new_context_lens.dtype)
            visible = (cols.unsqueeze(0) < new_context_lens.to(device=active_slots.device).unsqueeze(1)).reshape(-1)
        else:
            visible = torch.ones_like(flat_active, dtype=torch.bool)
        valid = visible & (flat_active >= 0)
        already = self._already_postrope_mask(layer_idx, flat_active, already_postrope_slots)
        raw_mask = valid & ~already
        safe_raw_slots = torch.where(raw_mask, flat_active, torch.zeros_like(flat_active)).to(torch.long)
        if active_pos is None:
            raw_pos = self.deltakv_slot_to_pos[safe_raw_slots].to(torch.long)
        else:
            raw_pos = active_pos.reshape(-1).to(device=active_slots.device, dtype=torch.long)
        is_capturing = self._is_stream_capturing()
        if not is_capturing and ((raw_pos < 0) & raw_mask).any():
            raise RuntimeError(f"DeltaKV static sparse raw slot has unknown position, layer={layer_idx}.")
        safe_pos = torch.where(raw_mask, raw_pos, torch.zeros_like(raw_pos))

        cos_sin = self.cos_sin_cache[safe_pos]
        cos, sin = cos_sin.chunk(2, dim=-1)
        k_normed = self._apply_sparse_k_norm_if_needed(l_idx, k_cache[safe_raw_slots])
        k_postrope = apply_rotary_emb(k_normed, cos, sin)
        out_i64 = out_slots.to(torch.long)
        prev_k = k_cache[out_i64]
        prev_v = v_cache[out_i64]
        write_mask = raw_mask[:, None, None]
        k_cache[out_i64] = torch.where(write_mask, k_postrope.to(k_cache.dtype), prev_k)
        v_cache[out_i64] = torch.where(write_mask, v_cache[safe_raw_slots].to(v_cache.dtype), prev_v)
        prev_pos = self.deltakv_slot_to_pos[out_i64]
        self.deltakv_slot_to_pos[out_i64] = torch.where(raw_mask, safe_pos.to(torch.int32), prev_pos)

        flat_out = torch.where(raw_mask, out_slots, flat_active).to(torch.int32)
        active_slots.copy_(flat_out.view_as(active_slots))
        return active_slots, torch.empty((0,), device=active_slots.device, dtype=torch.int32)

    def get_layer_compute_tensors(self, layer_idx: int, selection: SparseSelection | None = None):
        del selection
        if self.has_prefill_staging_view(layer_idx):
            return self.deltakv_prefill_staging_kv_cache[0], self.deltakv_prefill_staging_kv_cache[1]
        raise NotImplementedError

    def get_prefill_compute_view(
        self,
        layer_idx: int,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        selection: SparseSelection,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.has_prefill_staging_view(layer_idx):
            if bool(getattr(self, "_deltakv_long_prefill_offload_step_active", False)):
                visible_len = int(context_lens.max().item()) if context_lens.numel() > 0 else 0
                if visible_len > int(k_current.shape[0]):
                    return (
                        self.deltakv_prefill_staging_kv_cache[0],
                        self.deltakv_prefill_staging_kv_cache[1],
                        active_slots,
                        req_indices,
                        context_lens,
                    )
            return k_current, v_current, active_slots, req_indices, context_lens
        return super().get_prefill_compute_view(
            layer_idx,
            k_current,
            v_current,
            selection,
            active_slots,
            req_indices,
            context_lens,
        )

    def build_prefill_compute_view(
        self,
        layer_idx: int,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        selection: SparseSelection,
    ) -> PrefillComputeView:
        if self.has_prefill_staging_view(layer_idx) or selection.kind != "deltakv":
            return super().build_prefill_compute_view(layer_idx, k_current, v_current, selection)

        active_slots, local_req, context_lens, temp_slots = self.deltakv_reconstruct(
            layer_idx=layer_idx,
            active_compressed_indices=selection.active_compressed_indices,
            context_lens=selection.context_lens,
            req_indices=selection.req_indices,
            chunk_lens=selection.chunk_lens,
            return_reconstruct_temp_slots=selection.release_temp_slots,
        )
        k_cache, v_cache, active_slots, req_indices, context_lens = self.get_prefill_compute_view(
            layer_idx,
            k_current,
            v_current,
            selection,
            active_slots,
            local_req,
            context_lens,
        )
        return PrefillComputeView(
            meta=AttentionViewMeta(
                active_slots=active_slots,
                req_indices=req_indices,
                context_lens=context_lens,
                attn_score=selection.attn_score,
                max_context_len=selection.max_context_len,
                temp_slots=temp_slots,
            ),
            payload=ExplicitKVPayload(k_cache=k_cache, v_cache=v_cache),
        )

    def build_decode_compute_view(
        self,
        layer_idx: int,
        q: torch.Tensor,
        selection: SparseSelection,
        *,
        num_heads: int,
        num_kv_heads: int,
    ) -> DecodeComputeView:
        if selection.kind != "deltakv":
            return super().build_decode_compute_view(
                layer_idx,
                q,
                selection,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )

        active_slots, local_req, context_lens, temp_slots = self.deltakv_reconstruct(
            layer_idx=layer_idx,
            active_compressed_indices=selection.active_compressed_indices,
            context_lens=selection.context_lens,
            req_indices=selection.req_indices,
            chunk_lens=selection.chunk_lens,
            return_reconstruct_temp_slots=selection.release_temp_slots,
        )
        k_cache, v_cache, active_slots, req_indices, context_lens = self.get_layer_compute_view(
            layer_idx,
            active_slots,
            local_req,
            context_lens,
            selection,
        )
        return DecodeComputeView(
            meta=AttentionViewMeta(
                active_slots=active_slots,
                req_indices=req_indices,
                context_lens=context_lens,
                attn_score=selection.attn_score,
                max_context_len=selection.max_context_len,
                temp_slots=temp_slots,
            ),
            payload=ExplicitKVPayload(k_cache=k_cache, v_cache=v_cache),
        )

    def has_prefill_staging_view(self, layer_idx: int) -> bool:
        return bool(
            getattr(self, "_deltakv_prefill_staging_active", False)
            and layer_idx in getattr(self, "deltakv_layer_to_idx", {})
        )

    def get_prefill_staging_view(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not self.has_prefill_staging_view(layer_idx):
            raise NotImplementedError("DeltaKV prefill staging view is not active for this layer.")
        return (
            self._deltakv_prefill_staging_active_slots,
            self._deltakv_prefill_staging_req_indices,
            self._deltakv_prefill_staging_context_lens,
            None,
        )

    def get_layer_buffer_req_to_token_slots(self, layer_idx: int) -> torch.Tensor:
        if layer_idx in self.full_layer_to_idx:
            return self.full_layer_slots_map
        else:
            # DeltaKV sparse layers never directly expose a dense Req->slots table because
            # most historical tokens are either compressed or reconstructed on-the-fly.
            raise NotImplementedError("DeltaKV sparse layers should use build_*_compute_view().")

    def _debug_token_slots_for_mapping(
        self,
        layer_idx: int | None,
    ) -> torch.Tensor:
        del layer_idx
        return self.full_layer_slots_map

    @property
    def num_free_slots(self) -> int:
        # Scheduling should be conservative: we must be able to allocate both
        # full-layer KV slots and DeltaKV full-KV slots for new tokens.
        deltakv_usable = self._num_free_slots_deltakv_full - self._deltakv_unallocated_temp_full_reserve()
        return min(self._num_free_slots_full, max(0, deltakv_usable))

    def _deltakv_unallocated_temp_full_reserve(self) -> int:
        configured = int(getattr(self, "_deltakv_temp_full_reserve", 0) or 0)
        static_reserved = int(getattr(self, "_deltakv_static_temp_slots_reserved_total", 0) or 0)
        return max(0, configured - static_reserved)

    def num_free_slots_full_layers(self) -> int:
        return int(self._num_free_slots_full)

    def prefill_step_free_slots(self) -> int:
        # Current-step persistent capacity only. Full-prefill staging capacity is
        # checked in _prepare_prefill() and decode reconstruction reserve is excluded.
        return int(self.num_free_slots)

    def prefill_step_free_slots_for(self, seq: Sequence) -> int:
        if self.requires_long_prefill_offload(seq):
            staging_slots = int(getattr(self, "deltakv_prefill_staging_num_slots", 0) or 0)
            return staging_slots
        return super().prefill_step_free_slots_for(seq)

    def prefill_step_reservation_cost(self, seq: Sequence, scheduled_tokens: int) -> int:
        if self.requires_long_prefill_offload(seq):
            return 0
        return super().prefill_step_reservation_cost(seq, scheduled_tokens)

    def reserved_prefill_slots(self, waiting_seqs: deque[Sequence], engine_prefill_chunk_size: int) -> int:
        # DeltaKV can evict sparse-layer KV during long prefill; reserving the entire remaining
        # prompt is overly conservative and causes decode thrashing. Reserve at most one chunk
        # per in-progress prefill sequence.
        reserved = 0
        for seq in waiting_seqs:
            if 0 < seq.num_prefilled_tokens < seq.num_prompt_tokens:
                remaining = int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
                reserved += min(remaining, int(engine_prefill_chunk_size))
        return reserved

    def decode_window_budgets(self) -> dict[str, int]:
        return {
            "full_layers": int(self._num_free_slots_full),
            "deltakv_raw": max(0, int(self._num_free_slots_deltakv_full)
                               - self._deltakv_unallocated_temp_full_reserve()),
            "deltakv_latent": int(self._num_free_slots_deltakv_latent),
        }

    def decode_window_costs(self, seq: Sequence, tokens: int) -> dict[str, int]:
        row = self.seq_id_to_row[seq.seq_id]
        end = int(self.row_seq_lens[row]) + int(tokens)
        # Include history that can be compressed in one indivisible eviction;
        # the horizon does not truncate or reschedule that eviction.
        latent = max(0, self._estimate_deltakv_latent_slots_for_total_len(end)
                     - int(self.row_deltakv_compressed_lens[row]))
        return {"full_layers": int(tokens), "deltakv_raw": int(tokens),
                "deltakv_latent": latent}

    def prefill_capacity_after_decode_reservations(
        self, free_slots: int, reserved: dict[str, int], *, admission: bool,
    ) -> int:
        if admission:
            return int(free_slots) - reserved.get("full_layers", 0)
        # Prefill appends to both raw pools; latent slots and quantized blocks
        # retain their separate admission budgets and cannot consume raw space.
        raw_free = max(
            0, self._num_free_slots_deltakv_full - self._deltakv_unallocated_temp_full_reserve()
        )
        return min(
            int(free_slots),
            self._num_free_slots_full - reserved.get("full_layers", 0),
            raw_free - reserved.get("deltakv_raw", 0),
        )

    def prompt_admission_free_slots(self) -> int:
        # Full-attention layers store every token and cannot be evicted, so gate admission by that pool.
        return self.num_free_slots_full_layers()

    def prompt_admission_cost(self, seq: Sequence) -> int:
        # Prompt capacity only; future decode growth uses the shared window ledger.
        return int(seq.num_prompt_tokens)

    def prompt_logical_reservation_cost(self, seq: Sequence) -> int:
        # DeltaKV does not need to reserve the full prompt in sparse layers.
        return 0

    def prompt_admission_failure_action(self) -> str:
        # Defer admission until other sequences finish and free full-layer slots.
        return "defer"

    def prompt_admission_budgets(
        self,
        waiting_seqs: deque[Sequence],
        engine_prefill_chunk_size: int,
    ) -> dict[str, int]:
        # Gate on full-attention pool, future centers, and the sparse raw pool's
        # final keep-position representation. Decode-reconstruction scratch slots
        # live in the same sparse raw pool, so admission must not spend them.
        reserved = int(self.reserved_prefill_slots(waiting_seqs, engine_prefill_chunk_size))
        centers_free = max(0, int(self._deltakv_centers_capacity) - int(self._deltakv_centers_reserved_total))
        temp_reserve = self._deltakv_unallocated_temp_full_reserve()
        raw_free = max(0, int(self._num_free_slots_deltakv_full) - temp_reserve - reserved)
        full_layers_reserved = int(getattr(self, "_full_layers_reserved_total", 0) or 0)
        return {
            "full_layers": max(0, int(self.num_free_slots_full_layers()) - full_layers_reserved),
            "deltakv_centers": centers_free,
            "deltakv_raw": raw_free,
        }

    def _deltakv_plan_for_total_len_cpu(self, total_len: int) -> DeltaKVFullPrefillPlanCPU:
        return self._deltakv_full_prefill_plan_cpu(
            int(total_len),
            sink=int(self.config.sink_keep_tokens),
            recent=self._deltakv_full_prefill_recent_tokens(),
            cluster_step=self._deltakv_base_cluster_step(),
        )

    def _estimate_deltakv_centers_for_total_len_exact(self, total_len: int) -> int:
        plan = self._deltakv_plan_for_total_len_cpu(total_len)
        return len(plan.center_positions)

    def _estimate_deltakv_latent_slots_for_total_len(self, total_len: int) -> int:
        # Same finalized-history extent as the prefill plan, without building
        # token-position tuples on the decode scheduling path.
        buffer_len = max(0, int(total_len) - int(self.config.sink_keep_tokens))
        recent = int(self._deltakv_full_prefill_recent_tokens())
        if buffer_len <= recent:
            return 0
        return ((buffer_len - recent) // recent) * recent if recent > 0 else buffer_len

    def _estimate_deltakv_raw_slots_for_total_len(self, total_len: int) -> int:
        plan = self._deltakv_plan_for_total_len_cpu(total_len)
        return len(plan.keep_positions)

    def _max_len_for_deltakv_center_budget(self, center_budget: int) -> int:
        center_budget = max(0, int(center_budget))
        lo, hi = 0, int(self.config.max_model_len)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._estimate_deltakv_centers_for_total_len_exact(mid) <= center_budget:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _max_resource_under_deltakv_center_budget(self, center_budget: int, resource_fn) -> int:
        """
        Estimate max total resource reachable under center budget and max active seqs.
        Greedy long-sequence packing estimates fixed-stride center allocation
        under the configured cluster ratio.
        Admission still enforces per-resource budgets before a prompt is scheduled.
        """
        remaining = max(0, int(center_budget))
        total = 0
        max_seqs = int(self.config.max_num_seqs_in_gpu)
        for _ in range(max_seqs):
            if remaining <= 0:
                break
            length = self._max_len_for_deltakv_center_budget(remaining)
            centers = self._estimate_deltakv_centers_for_total_len_exact(length)
            if centers <= 0:
                break
            total += int(resource_fn(length))
            remaining -= int(centers)
        return int(total)

    def _estimate_centers_for_total_len(self, total_len: int) -> int:
        return self._estimate_deltakv_centers_for_total_len_exact(total_len)

    def prompt_admission_costs(self, seq: Sequence) -> dict[str, int]:
        prompt_len = int(seq.num_prompt_tokens)
        total_len = int(seq.num_prompt_tokens)
        return {
            "full_layers": int(seq.num_prompt_tokens),
            "deltakv_centers": self._estimate_centers_for_total_len(total_len),
            "deltakv_raw": self._estimate_deltakv_raw_slots_for_total_len(prompt_len),
        }

    def on_prompt_admitted(self, seq: Sequence, costs: dict[str, int]):
        # Reserve future centers budget to prevent admitting too many long prompts.
        seq_id = int(seq.seq_id)
        if seq_id in self._deltakv_centers_reserved_by_seq:
            return
        full_layers = int(costs.get("full_layers", 0) or 0)
        self._full_layers_reserved_by_seq[seq_id] = full_layers
        self._full_layers_reserved_total += full_layers
        centers = int(costs.get("deltakv_centers", 0) or 0)
        self._deltakv_centers_reserved_by_seq[seq_id] = centers
        self._deltakv_centers_reserved_total += centers

    def _release_prompt_admission_reservations(self, seq_id: int):
        seq_id = int(seq_id)
        full_layers = getattr(self, "_full_layers_reserved_by_seq", {}).pop(seq_id, 0)
        if full_layers:
            total_reserved = int(getattr(self, "_full_layers_reserved_total", 0) or 0)
            if total_reserved < int(full_layers):
                raise RuntimeError(
                    "DeltaKV full-layer reservation release underflow: "
                    f"seq_id={seq_id} release={int(full_layers)} total_reserved={total_reserved}."
                )
            self._full_layers_reserved_total -= int(full_layers)
        centers = self._deltakv_centers_reserved_by_seq.pop(seq_id, 0)
        if centers:
            self._deltakv_centers_reserved_total -= int(centers)
        latent = getattr(self, "_deltakv_latent_reserved_by_seq", {}).pop(seq_id, 0)
        if latent:
            self._deltakv_latent_reserved_total -= int(latent)
        kivi = getattr(self, "_full_layer_kivi_reserved_by_seq", {}).pop(seq_id, 0)
        if kivi:
            self._full_layer_kivi_reserved_total -= int(kivi)

    def _consume_full_layer_reservation(self, seq_id: int, size: int):
        size = int(size)
        if size <= 0:
            return
        reserved_by_seq = getattr(self, "_full_layers_reserved_by_seq", None)
        if not reserved_by_seq:
            return
        seq_id = int(seq_id)
        remaining = int(reserved_by_seq.get(seq_id, 0) or 0)
        if remaining <= 0:
            return
        consumed = min(size, remaining)
        next_remaining = remaining - consumed
        if next_remaining:
            reserved_by_seq[seq_id] = next_remaining
        else:
            reserved_by_seq.pop(seq_id, None)
        total_reserved = int(getattr(self, "_full_layers_reserved_total", 0) or 0)
        if total_reserved < consumed:
            raise RuntimeError(
                "DeltaKV full-layer reservation accounting underflow: "
                f"seq_id={seq_id} consume={consumed} total_reserved={total_reserved}."
            )
        self._full_layers_reserved_total = total_reserved - consumed

    @torch.no_grad()
    def _allocate_temp_deltakv_full(self, size: int) -> torch.Tensor:
        """Allocate DeltaKV full-KV slots without touching per-seq slot maps (scratch for reconstruction)."""
        if self._num_free_slots_deltakv_full < size:
            raise RuntimeError(
                "Out of DeltaKV full cache slots (temp). "
                f"need={size} free={self._num_free_slots_deltakv_full} "
                f"(reserved_for_temp={int(getattr(self, '_deltakv_temp_full_reserve', 0) or 0)}). "
                "Try reducing batch_size/decode_keep_tokens, or increase deltakv_full_pool_reserve_ratio."
            )
        ptr = self._num_free_slots_deltakv_full
        # Static decode caches temp slots across steps; keep them independent
        # from the mutable free stack backing storage.
        select_index = self.free_slots_stack_deltakv_full[ptr - size: ptr].clone().to(torch.int32)
        self._num_free_slots_deltakv_full -= size
        return select_index

    def _allocate_persistent_deltakv_full_slots(self, size: int, temp_reserve: int) -> torch.Tensor:
        size = int(size)
        temp_reserve = int(temp_reserve)
        if size == 0:
            return torch.empty((0,), dtype=torch.int32, device=self.device)
        free_count = int(self._num_free_slots_deltakv_full)
        usable = free_count - temp_reserve
        if usable < size:
            raise RuntimeError(
                "Out of DeltaKV full cache slots (persistent). "
                f"need={size} free_total={free_count} free_usable={max(0, usable)} "
                f"(reserved_for_temp={temp_reserve}, configured_temp_reserve="
                f"{int(getattr(self, '_deltakv_temp_full_reserve', 0) or 0)}, "
                f"static_temp_reserved={int(getattr(self, '_deltakv_static_temp_slots_reserved_total', 0) or 0)}). "
                "Reduce concurrency/chunk size, or increase deltakv_full_pool_reserve_ratio."
            )

        end = usable
        start = end - size
        select_index = self.free_slots_stack_deltakv_full[start:end].clone().to(torch.int32)
        if temp_reserve > 0:
            self.free_slots_stack_deltakv_full[start:start + temp_reserve] = (
                self.free_slots_stack_deltakv_full[end:free_count].clone()
            )
        self._num_free_slots_deltakv_full -= size
        return select_index

    @torch.no_grad()
    def free_temp_deltakv_full(self, slots: torch.Tensor | None):
        """Return scratch slots allocated by _allocate_temp_deltakv_full()."""
        if slots is None or slots.numel() == 0:
            return
        if self._is_stream_capturing():
            return
        slots = slots.to(torch.int32)
        reset_view_cache = False
        cached = self._deltakv_view_cache_value
        if cached is not None:
            cached_temp_slots = cached[3]
            if cached_temp_slots is not None and cached_temp_slots.numel() > 0:
                reset_view_cache = bool(
                    (slots[:, None] == cached_temp_slots.to(slots.device)[None, :]).any().item()
                )
        ptr = self._num_free_slots_deltakv_full
        self.free_slots_stack_deltakv_full[ptr: ptr + slots.numel()] = slots
        self._num_free_slots_deltakv_full += slots.numel()
        # Scratch slots have no stable position.
        self.deltakv_slot_to_pos[slots] = -1
        # Keep cached reconstruct slots alive when freeing unrelated materialized views.
        if reset_view_cache:
            self._deltakv_reset_view_cache()

    def release_layer_temp_slots(self, layer_idx: int, temp_slots: torch.Tensor | None):
        del layer_idx
        self.free_temp_deltakv_full(temp_slots)

    @staticmethod
    def _append_tensor_refs(out: list[torch.Tensor], value):
        if isinstance(value, torch.Tensor):
            out.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                DeltaKVCacheManager._append_tensor_refs(out, item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                DeltaKVCacheManager._append_tensor_refs(out, item)

    def decode_graph_keepalive_tensors(self) -> list[torch.Tensor]:
        refs: list[torch.Tensor] = []
        for attr_name in (
            "_deltakv_decode_static_slot_mapping",
            "_deltakv_decode_static_compressed_lens",
            "_deltakv_decode_static_active_pos",
            "_deltakv_materialized_active_slots",
            "_deltakv_materialized_flat_slots",
            "_deltakv_materialized_local_req",
        ):
            self._append_tensor_refs(refs, getattr(self, attr_name, None))
        for attr_name in (
            "_deltakv_decode_static_slot_mapping_by_shape",
            "_deltakv_decode_static_compressed_lens_by_shape",
            "_deltakv_decode_static_temp_slots_by_shape",
            "_deltakv_decode_static_plan_buffers_by_shape",
            "_deltakv_decode_static_materialized_slots_by_shape",
        ):
            self._append_tensor_refs(refs, getattr(self, attr_name, None))
        return refs

    def _get_free_row(self, seq_id: int) -> int:
        if seq_id in self.seq_id_to_row:
            return self.seq_id_to_row[seq_id]
        if not self.free_rows:
            raise RuntimeError("No free rows in cache manager buffer!")
        row_idx = self.free_rows.popleft()
        self.seq_id_to_row[seq_id] = row_idx
        return row_idx

    @torch.no_grad()
    def _allocate_full(self, seq_id: int, size: int) -> torch.Tensor:
        row_idx = self.seq_id_to_row.get(seq_id)
        cur_len = 0 if row_idx is None else int(self.row_seq_lens[row_idx])
        if cur_len + size > self.full_layer_slots_map.shape[1]:
            raise RuntimeError("KV row length exceeds max_model_len in DeltaKV _allocate_full.")
        assert self._num_free_slots_full >= size, (
            f"Out of full KV cache slots: need {size}, free {self._num_free_slots_full}"
        )
        row_idx = self._get_free_row(seq_id)

        ptr = self._num_free_slots_full
        select_index = self.free_slots_stack_full[ptr - size: ptr]
        self._num_free_slots_full -= size

        self.full_layer_slots_map[row_idx, cur_len: cur_len + size] = select_index
        full_slot_to_pos = getattr(self, "full_layer_slot_to_pos", None)
        if full_slot_to_pos is not None:
            full_slot_to_pos[select_index] = torch.arange(cur_len, cur_len + size, device=self.device, dtype=torch.int32)
        self._consume_full_layer_reservation(seq_id, size)
        return select_index

    @torch.no_grad()
    def _allocate_deltakv_full(self, seq_id: int, size: int) -> torch.Tensor:
        row_idx = self.seq_id_to_row.get(seq_id)
        cur_len = 0 if row_idx is None else int(self.row_seq_lens[row_idx])
        if cur_len + size > self.sparse_layer_raw_slots_map.shape[1]:
            raise RuntimeError("KV row length exceeds max_model_len in DeltaKV _allocate_deltakv_full.")
        temp_reserve = self._deltakv_unallocated_temp_full_reserve()
        usable = self._num_free_slots_deltakv_full - temp_reserve
        if usable < size:
            raise RuntimeError(
                "Out of DeltaKV full cache slots (persistent). "
                f"need={size} free_total={self._num_free_slots_deltakv_full} free_usable={usable} "
                f"(reserved_for_temp={temp_reserve}, configured_temp_reserve="
                f"{int(getattr(self, '_deltakv_temp_full_reserve', 0) or 0)}, "
                f"static_temp_reserved={int(getattr(self, '_deltakv_static_temp_slots_reserved_total', 0) or 0)}). "
                "Reduce concurrency/chunk size, or increase deltakv_full_pool_reserve_ratio."
            )
        row_idx = self._get_free_row(seq_id)

        select_index = self._allocate_persistent_deltakv_full_slots(size, temp_reserve)

        self.sparse_layer_raw_slots_map[row_idx, cur_len: cur_len + size] = select_index
        self.deltakv_slot_to_pos[select_index] = torch.arange(cur_len, cur_len + size, device=self.device, dtype=torch.int32)
        return select_index

    @torch.no_grad()
    def _allocate_deltakv_full_positions(self, seq_id: int, positions: torch.Tensor) -> torch.Tensor:
        positions = positions.to(device=self.device, dtype=torch.int32).contiguous()
        size = int(positions.numel())
        if size == 0:
            return torch.empty((0,), dtype=torch.int32, device=self.device)
        temp_reserve = self._deltakv_unallocated_temp_full_reserve()
        usable = self._num_free_slots_deltakv_full - temp_reserve
        if usable < size:
            raise RuntimeError(
                "Out of DeltaKV persistent raw slots for full-prefill final representation. "
                f"need={size} free_total={self._num_free_slots_deltakv_full} free_usable={usable} "
                f"(decode_reconstruct_reserve={temp_reserve}, configured_decode_reconstruct_reserve="
                f"{int(getattr(self, '_deltakv_temp_full_reserve', 0) or 0)}, "
                f"static_temp_reserved={int(getattr(self, '_deltakv_static_temp_slots_reserved_total', 0) or 0)})."
            )
        row_idx = self._get_free_row(seq_id)
        select_index = self._allocate_persistent_deltakv_full_slots(size, temp_reserve)
        self.sparse_layer_raw_slots_map[row_idx, positions.to(torch.long)] = select_index
        self.deltakv_slot_to_pos[select_index.to(torch.long)] = positions
        return select_index

    @torch.no_grad()
    def _allocate_deltakv_latent(self, size: int) -> torch.Tensor:
        assert self._num_free_slots_deltakv_latent >= size, (
            f"Out of DeltaKV latent cache slots: need {size}, free {self._num_free_slots_deltakv_latent}"
        )
        ptr = self._num_free_slots_deltakv_latent
        select_index = self.free_slots_stack_deltakv_latent[ptr - size: ptr]
        self._num_free_slots_deltakv_latent -= size
        return select_index

    @torch.no_grad()
    def _allocate_batch_full(self, seq_ids: list[int], size: int) -> torch.Tensor:
        assert size == 1, "Batch allocation currently only supports size=1 (Decode)"
        batch_size = len(seq_ids)
        assert self._num_free_slots_full >= batch_size, (
            f"Out of full KV cache slots: need {batch_size}, free {self._num_free_slots_full}"
        )
        row_indices = self._get_decode_rows(seq_ids, self.full_layer_slots_map.shape[1])
        cur_lens = self.row_seq_lens[row_indices]

        ptr = self._num_free_slots_full
        select_indices = self.free_slots_stack_full[ptr - batch_size: ptr]
        self._num_free_slots_full -= batch_size

        rows_gpu = torch.tensor(row_indices, dtype=torch.long, device=self.device)
        cols_gpu = torch.tensor(cur_lens, dtype=torch.long, device=self.device)
        self.full_layer_slots_map[rows_gpu, cols_gpu] = select_indices
        full_slot_to_pos = getattr(self, "full_layer_slot_to_pos", None)
        if full_slot_to_pos is not None:
            full_slot_to_pos[select_indices] = cols_gpu.to(torch.int32)
        for seq_id in seq_ids:
            self._consume_full_layer_reservation(seq_id, 1)
        return select_indices

    @torch.no_grad()
    def _allocate_batch_deltakv_full(self, seq_ids: list[int], size: int) -> torch.Tensor:
        assert size == 1, "Batch allocation currently only supports size=1 (Decode)"
        batch_size = len(seq_ids)
        temp_reserve = self._deltakv_unallocated_temp_full_reserve()
        usable = self._num_free_slots_deltakv_full - temp_reserve
        if usable < batch_size:
            raise RuntimeError(
                "Out of DeltaKV full cache slots (persistent batch). "
                f"need={batch_size} free_total={self._num_free_slots_deltakv_full} free_usable={usable} "
                f"(reserved_for_temp={temp_reserve}, configured_temp_reserve="
                f"{int(getattr(self, '_deltakv_temp_full_reserve', 0) or 0)}, "
                f"static_temp_reserved={int(getattr(self, '_deltakv_static_temp_slots_reserved_total', 0) or 0)}). "
                "Reduce concurrency, or increase deltakv_full_pool_reserve_ratio."
            )
        row_indices = self._get_decode_rows(seq_ids, self.sparse_layer_raw_slots_map.shape[1])
        cur_lens = self.row_seq_lens[row_indices]

        select_indices = self._allocate_persistent_deltakv_full_slots(batch_size, temp_reserve)

        rows_gpu = torch.tensor(row_indices, dtype=torch.long, device=self.device)
        cols_gpu = torch.tensor(cur_lens, dtype=torch.long, device=self.device)
        self.sparse_layer_raw_slots_map[rows_gpu, cols_gpu] = select_indices
        self.deltakv_slot_to_pos[select_indices] = cols_gpu.to(torch.int32)
        return select_indices

    def _get_decode_rows(self, seq_ids: list[int], capacity: int) -> list[int]:
        needed_rows = len(set(seq_ids).difference(self.seq_id_to_row))
        if needed_rows > len(self.free_rows):
            raise RuntimeError(
                f"No free rows for DeltaKV decode batch: need={needed_rows} free={len(self.free_rows)}."
            )
        for seq_id in seq_ids:
            row = self.seq_id_to_row.get(seq_id)
            cur_len = 0 if row is None else int(self.row_seq_lens[row])
            if cur_len + 1 > capacity:
                raise RuntimeError("KV row length exceeds max_model_len in DeltaKV decode batch.")
        return [self._get_free_row(seq_id) for seq_id in seq_ids]

    def _active_deltakv_raw_slots_for_free(self, row_idx: int, cur_len: int) -> torch.Tensor:
        parts = []
        raw_slots = self.sparse_layer_raw_slots_map[row_idx, :cur_len]
        if raw_slots.numel() > 0:
            parts.append(raw_slots.to(torch.int32))

        for center_slots in self.row_deltakv_center_slots[row_idx]:
            if center_slots is not None and center_slots.numel() > 0:
                parts.append(center_slots.to(device=raw_slots.device, dtype=torch.int32).flatten())

        if not parts:
            return torch.empty((0,), dtype=torch.int32, device=self.sparse_layer_raw_slots_map.device)

        slots = torch.cat(parts)
        slots = slots[slots >= 0]
        if slots.numel() == 0:
            return slots.to(torch.int32)

        slots = torch.unique(slots.to(torch.int32))
        if self.deltakv_slot_to_pos is not None:
            active = self.deltakv_slot_to_pos[slots.to(torch.long)] >= 0
            slots = slots[active]
        return slots.to(torch.int32)

    def _filter_deltakv_center_slots_for_evict_free(
        self,
        row_idx: int,
        slots: torch.Tensor,
        extra_center_slots: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if slots.numel() == 0:
            return slots.to(torch.int32)

        parts = []
        for center_slots in self.row_deltakv_center_slots[int(row_idx)]:
            if center_slots is not None and center_slots.numel() > 0:
                parts.append(center_slots.to(device=slots.device, dtype=torch.int32).flatten())
        if extra_center_slots is not None and extra_center_slots.numel() > 0:
            parts.append(extra_center_slots.to(device=slots.device, dtype=torch.int32).flatten())
        if not parts:
            return slots.to(torch.int32)

        centers = torch.unique(torch.cat(parts))
        centers = centers[centers >= 0]
        if centers.numel() == 0:
            return slots.to(torch.int32)

        slots_i32 = slots.to(torch.int32)
        is_center = (slots_i32[:, None] == centers[None, :]).any(dim=1)
        return slots_i32[~is_center]

    def free_seq(self, seq_id: int):
        with profiler.record("cache_free_seq"):
            self.reset_prefill_execution_state(seq_id)
            self._release_prompt_admission_reservations(seq_id)
            row_idx = self.seq_id_to_row.pop(seq_id, None)
            if row_idx is None:
                raise ValueError
            self.raw_kv_offload_buffer.release_row(int(row_idx))

            cur_len = self.row_seq_lens[row_idx]
            assert cur_len > 0


            full_slots = self.full_layer_slots_map[row_idx, :cur_len]
            ptr = self._num_free_slots_full
            self.free_slots_stack_full[ptr: ptr + cur_len] = full_slots
            self._num_free_slots_full += cur_len


            slots = self._active_deltakv_raw_slots_for_free(row_idx, int(cur_len))
            ptr = self._num_free_slots_deltakv_full

            self.free_slots_stack_deltakv_full[ptr: ptr + slots.numel()] = slots
            self._num_free_slots_deltakv_full += slots.numel()
            self.deltakv_slot_to_pos[slots] = -1

            latent_slots = self.sparse_layer_latent_slots_map[row_idx, :cur_len]
            mask_latent = latent_slots >= 0
            if mask_latent.any():
                slots = latent_slots[mask_latent]
                ptr = self._num_free_slots_deltakv_latent
                self.free_slots_stack_deltakv_latent[ptr: ptr + slots.numel()] = slots
                self._num_free_slots_deltakv_latent += slots.numel()

            self.full_layer_slots_map[row_idx, :] = 0
            self.sparse_layer_raw_slots_map[row_idx, :] = -1
            self.sparse_layer_latent_slots_map[row_idx, :] = -1
            self.row_seq_lens[row_idx] = 0
            self.row_deltakv_compressed_lens[row_idx] = 0
            self.row_deltakv_compressed_lens_gpu[row_idx] = 0
            self.row_deltakv_center_slots[row_idx] = [None for _ in range(self.num_layers)]
            self.free_rows.append(row_idx)

    def free_slot_stats(self) -> dict[str, int]:
        full_free = int(getattr(self, "_num_free_slots_full", 0) or 0)
        deltakv_full_free_total = int(getattr(self, "_num_free_slots_deltakv_full", 0) or 0)
        deltakv_latent_free = int(getattr(self, "_num_free_slots_deltakv_latent", 0) or 0)
        configured_temp_reserve = int(getattr(self, "_deltakv_temp_full_reserve", 0) or 0)
        static_temp_reserved = int(getattr(self, "_deltakv_static_temp_slots_reserved_total", 0) or 0)
        temp_reserve = self._deltakv_unallocated_temp_full_reserve()
        deltakv_full_free_usable = max(0, deltakv_full_free_total - temp_reserve)
        centers_cap = int(getattr(self, "_deltakv_centers_capacity", 0) or 0)
        centers_reserved = int(getattr(self, "_deltakv_centers_reserved_total", 0) or 0)
        centers_free = max(0, centers_cap - centers_reserved)
        full_layers_reserved = int(getattr(self, "_full_layers_reserved_total", 0) or 0)
        active = int(len(getattr(self, "seq_id_to_row", {}) or {}))
        return {
            "free_slots": int(self.num_free_slots),
            "full_free": full_free,
            "full_reserved": full_layers_reserved,
            "full_free_after_reserved": max(0, full_free - full_layers_reserved),
            "deltakv_full_free_total": deltakv_full_free_total,
            "deltakv_full_free_usable": deltakv_full_free_usable,
            "deltakv_decode_reconstruct_reserve": temp_reserve,
            "deltakv_decode_reconstruct_reserve_configured": configured_temp_reserve,
            "deltakv_decode_static_temp_reserved": static_temp_reserved,
            "deltakv_prefill_staging_capacity": int(getattr(self, "deltakv_prefill_staging_num_slots", 0) or 0),
            "deltakv_prefill_staging_active": int(bool(getattr(self, "_deltakv_prefill_staging_active", False))),
            "deltakv_latent_free": deltakv_latent_free,
            "centers_capacity": centers_cap,
            "centers_reserved": centers_reserved,
            "centers_free": centers_free,
            "active_seqs": active,
        }

    def free_part_slots(self, layer_idx: int, seq: Sequence, keep_indices: torch.Tensor):
        raise ValueError("DeltaKV does not support partial slot freeing via this method.")

    def _tensor_from_positions(self, positions: tuple[int, ...]) -> torch.Tensor:
        if not positions:
            return torch.empty((0,), dtype=torch.int32, device=self.device)
        return torch.tensor(positions, dtype=torch.int32, device=self.device)

    def _prepare_full_prefill_staging_plan(
        self,
        seq: Sequence,
        row_idx: int,
        total_len: int,
        *,
        staging_start: int = 0,
    ):
        if total_len > int(self.deltakv_prefill_staging_num_slots):
            raise RuntimeError(
                "DeltaKV full-prefill staging capacity is too small for prompt. "
                f"prompt_len={total_len} staging_slots={self.deltakv_prefill_staging_num_slots}."
            )
        plan_cpu = self._deltakv_full_prefill_plan_cpu(
            total_len,
            sink=int(self.config.sink_keep_tokens),
            recent=self._deltakv_full_prefill_recent_tokens(),
            cluster_step=self._deltakv_base_cluster_step(),
        )

        keep_pos = self._tensor_from_positions(plan_cpu.keep_positions)
        center_pos = self._tensor_from_positions(plan_cpu.center_positions)
        latent_pos = self._tensor_from_positions(plan_cpu.latent_positions)
        raw_slots = self._allocate_deltakv_full_positions(seq.seq_id, keep_pos)
        raw_by_pos = {
            int(pos): raw_slots[i]
            for i, pos in enumerate(plan_cpu.keep_positions)
        }
        center_slots = (
            torch.stack([raw_by_pos[int(pos)] for pos in plan_cpu.center_positions]).to(torch.int32)
            if plan_cpu.center_positions
            else torch.empty((0,), dtype=torch.int32, device=self.device)
        )
        sink_slots = (
            self.sparse_layer_raw_slots_map[row_idx, : plan_cpu.sink_len].to(torch.int32)
            if plan_cpu.sink_len > 0
            else torch.empty((0,), dtype=torch.int32, device=self.device)
        )

        if latent_pos.numel() > 0:
            latent_slots = self._allocate_deltakv_latent(int(latent_pos.numel())).to(torch.int32)
            self.sparse_layer_latent_slots_map[row_idx, latent_pos.to(torch.long)] = latent_slots
        else:
            latent_slots = torch.empty((0,), dtype=torch.int32, device=self.device)

        if plan_cpu.evict_len > 0:
            evict_pos = torch.arange(plan_cpu.evict_start, plan_cpu.evict_end, device=self.device, dtype=torch.int32)
            center_rel = (center_pos - int(plan_cpu.evict_start)).to(torch.long)
            to_compress_mask = torch.ones((plan_cpu.evict_len,), device=self.device, dtype=torch.bool)
            latent_store_mask = to_compress_mask
            latent_store_indices = torch.arange(plan_cpu.evict_len, device=self.device, dtype=torch.long)
        else:
            evict_pos = torch.empty((0,), dtype=torch.int32, device=self.device)
            to_compress_mask = torch.empty((0,), dtype=torch.bool, device=self.device)
            latent_store_mask = torch.empty((0,), dtype=torch.bool, device=self.device)
            latent_store_indices = torch.empty((0,), dtype=torch.long, device=self.device)
        if latent_slots.numel() != int(latent_store_indices.numel()):
            raise RuntimeError(
                "DeltaKV full-prefill latent slot count mismatch: "
                f"latent_slots={latent_slots.numel()} "
                f"latent_store_tokens={int(latent_store_indices.numel())}."
            )

        self.row_deltakv_compressed_lens[row_idx] = int(plan_cpu.evict_len)
        self.row_deltakv_compressed_lens_gpu[row_idx] = int(plan_cpu.evict_len)
        self._after_full_prefill_plan_prepared(
            row_idx=row_idx,
            evict_start=int(plan_cpu.evict_start),
            evict_end=int(plan_cpu.evict_end),
            evict_positions=evict_pos,
        )
        self._deltakv_full_prefill_plans[row_idx] = {
            "row_idx": int(row_idx),
            "total_len": int(total_len),
            "staging_start": int(staging_start),
            "evict_start": int(plan_cpu.evict_start),
            "sink_slots": sink_slots,
            "center_pos": center_pos,
            "center_slots": center_slots,
            "keep_pos": keep_pos,
            "keep_slots": raw_slots,
            "evict_pos": evict_pos,
            "latent_slots": latent_slots,
            "to_compress_mask": to_compress_mask,
            "latent_store_mask": latent_store_mask,
            "latent_store_indices": latent_store_indices,
            "latent_store_indices_contiguous": bool(plan_cpu.evict_len > 0),
        }

    def _after_full_prefill_plan_prepared(
        self,
        *,
        row_idx: int,
        evict_start: int,
        evict_end: int,
        evict_positions: torch.Tensor,
    ):
        del row_idx, evict_start, evict_end, evict_positions

    def _deltakv_gather_raw_kv_from_cache(
        self,
        *,
        slots: torch.Tensor,
        pos: torch.Tensor | None,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        slots = slots.to(torch.int32).contiguous()
        del pos
        if slots.numel() == 0:
            return torch.empty(
                (0, 2 * self.num_kv_heads * self.head_dim),
                dtype=self.hf_config.dtype,
                device=self.device,
            )
        slots_i64 = slots.to(torch.long)
        k_raw = k_cache[slots_i64]
        v = v_cache[slots_i64]
        kv_dim_half = self.num_kv_heads * self.head_dim
        return torch.cat(
            [
                k_raw.reshape(-1, kv_dim_half),
                v.reshape(-1, kv_dim_half),
            ],
            dim=-1,
        )

    def _deltakv_store_layer_latent(
        self,
        *,
        l_idx: int,
        latent_slots: torch.Tensor,
        kv_block: torch.Tensor,
        base_kv: torch.Tensor,
        to_compress_indices: torch.Tensor,
    ):
        down = self.compress_down[l_idx]
        kv_down = down(kv_block).squeeze(0)
        base_down = down(base_kv).squeeze(0)
        latent_all = (kv_down - base_down).index_select(0, to_compress_indices)
        self._store_deltakv_latent(l_idx, latent_slots, latent_all)

    @torch.no_grad()
    def _deltakv_compress_full_prefill_layer(self, layer_idx: int):
        if layer_idx in self._deltakv_full_prefill_compressed_layers:
            raise RuntimeError(f"DeltaKV full-prefill layer compressed twice: layer={layer_idx}.")
        if layer_idx not in self.deltakv_layer_to_idx:
            return

        l_idx = self.deltakv_layer_to_idx[layer_idx]
        k_stage = self.deltakv_prefill_staging_kv_cache[0]
        v_stage = self.deltakv_prefill_staging_kv_cache[1]
        k_persist = self.deltakv_full_kv_cache[0, l_idx]
        v_persist = self.deltakv_full_kv_cache[1, l_idx]

        for plan in self._deltakv_full_prefill_plans.values():
            staging_start = int(plan.get("staging_start", 0) or 0)
            keep_pos = plan["keep_pos"]
            keep_slots = plan["keep_slots"]
            if keep_slots.numel() > 0:
                keep_pos_i64 = keep_pos.to(torch.long) + staging_start
                keep_slots_i64 = keep_slots.to(torch.long)
                k_persist[keep_slots_i64] = k_stage[keep_pos_i64]
                v_persist[keep_slots_i64] = v_stage[keep_pos_i64]

            sink_slots = plan["sink_slots"].to(torch.int32)
            center_pos = plan["center_pos"].to(torch.int32)
            center_slots = plan["center_slots"].to(torch.int32)
            self.row_deltakv_center_slots[int(plan["row_idx"])][layer_idx] = torch.cat(
                [sink_slots, center_slots],
                dim=0,
            )

            latent_slots = plan["latent_slots"].to(torch.int32)
            evict_pos = plan["evict_pos"].to(torch.int32)
            if latent_slots.numel() == 0:
                continue

            latent_store_indices = plan["latent_store_indices"].to(torch.long)
            kv_block = self._deltakv_gather_raw_kv_from_cache(
                slots=evict_pos + staging_start,
                pos=evict_pos,
                k_cache=k_stage,
                v_cache=v_stage,
            ).unsqueeze(0)
            topk_center_indices, base_kv = self._cluster_compress(
                layer_idx=layer_idx,
                kv_states=kv_block,
                existing_center_slots=sink_slots,
                cluster_step=self._deltakv_base_cluster_step(),
                new_center_rel=(center_pos - int(plan["evict_start"])).to(torch.long),
            )
            all_center_slots = torch.cat([sink_slots, center_slots], dim=0)
            father_slots_full = all_center_slots[topk_center_indices.to(torch.long)]
            father_slots = father_slots_full.index_select(0, latent_store_indices)
            k_neighbors = self.deltakv_latent_to_full_slots.shape[-1]
            k_eff = father_slots.shape[1]
            if k_eff < k_neighbors:
                pad = father_slots[:, :1].expand(-1, k_neighbors - k_eff)
                father_slots = torch.cat([father_slots, pad], dim=1)
            elif k_eff > k_neighbors:
                father_slots = father_slots[:, :k_neighbors]
            self.deltakv_latent_to_full_slots[l_idx, latent_slots] = father_slots.to(torch.int32)
            self._deltakv_store_layer_latent(
                l_idx=l_idx,
                latent_slots=latent_slots,
                kv_block=kv_block,
                base_kv=base_kv,
                to_compress_indices=latent_store_indices,
            )

        self._deltakv_full_prefill_compressed_layers.add(layer_idx)

    def _deltakv_finish_full_prefill_staging(self):
        for plan in self._deltakv_full_prefill_plans.values():
            self.raw_kv_offload_buffer.release_row(int(plan["row_idx"]))
            keep_slots = plan["keep_slots"].to(torch.long)
            if keep_slots.numel() == 0:
                continue
            keep_pos = plan["keep_pos"].to(device=keep_slots.device, dtype=torch.int32)
            self.deltakv_slot_to_pos[keep_slots] = keep_pos
        self._deltakv_prefill_staging_active = False
        self._deltakv_full_prefill_plans = {}
        self._deltakv_full_prefill_compressed_layers = set()

    def on_layer_attention_end(self, layer_idx: int):
        if not self.has_prefill_staging_view(layer_idx):
            return
        self._deltakv_compress_full_prefill_layer(layer_idx)
        if len(self._deltakv_full_prefill_compressed_layers) == len(self.deltakv_layer_ids):
            self._deltakv_finish_full_prefill_staging()

    def _prepare_prefill(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_prefill"):
            use_long_prefill_offload_staging = self._should_use_long_prefill_offload_staging(seqs)
            use_full_prefill_staging = self._should_use_full_prefill_staging(seqs) or use_long_prefill_offload_staging
            total_chunk_tokens = sum(seq.current_chunk_size for seq in seqs)
            if use_full_prefill_staging and total_chunk_tokens > int(self.deltakv_prefill_staging_num_slots):
                raise RuntimeError(
                    "DeltaKV full-prefill staging capacity is too small for this step. "
                    f"tokens={total_chunk_tokens} staging_slots={self.deltakv_prefill_staging_num_slots}."
                )

            input_ids_np = np.empty(total_chunk_tokens, dtype=np.int64)
            positions_np = np.empty(total_chunk_tokens, dtype=np.int64)
            cu_seqlens_q = [0]

            full_slot_mapping = torch.empty(total_chunk_tokens, dtype=torch.int32, device=self.device)
            if use_full_prefill_staging:
                deltakv_slot_mapping = torch.empty(total_chunk_tokens, dtype=torch.int32, device=self.device)
            else:
                deltakv_slot_mapping = torch.empty(total_chunk_tokens, dtype=torch.int32, device=self.device)
            context_lens_list = []
            req_indices = []

            token_offset = 0
            for seq in seqs:
                chunk_size = seq.current_chunk_size
                start_idx = seq.num_prefilled_tokens
                end_idx = start_idx + chunk_size

                if seq.seq_id in self.seq_id_to_row:
                    row_idx = self.seq_id_to_row[seq.seq_id]
                    if self.row_seq_lens[row_idx] != start_idx:
                        raise ValueError(
                            "KV cache row length mismatch in prefill: "
                            f"seq_id={seq.seq_id} row_seq_len={self.row_seq_lens[row_idx]} "
                            f"start_idx={start_idx}"
                        )

                full_slots = self._allocate_full(seq.seq_id, chunk_size)
                row_idx = self.seq_id_to_row[seq.seq_id]
                full_slot_mapping[token_offset: token_offset + chunk_size] = full_slots
                if use_full_prefill_staging:
                    if not use_long_prefill_offload_staging and start_idx != 0:
                        raise RuntimeError("DeltaKV full-prefill staging only supports first-prefill prompts.")
                    staging_range = torch.arange(
                        start_idx if use_long_prefill_offload_staging else token_offset,
                        end_idx if use_long_prefill_offload_staging else token_offset + chunk_size,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    deltakv_slot_mapping[token_offset: token_offset + chunk_size] = staging_range
                    if use_long_prefill_offload_staging:
                        if end_idx > int(self.deltakv_prefill_staging_num_slots):
                            raise RuntimeError(
                                "DeltaKV long-prefill offload staging capacity is too small for this chunk. "
                                f"context_len={end_idx} staging_slots={self.deltakv_prefill_staging_num_slots}."
                            )
                        if start_idx == 0:
                            self._prepare_full_prefill_staging_plan(
                                seq,
                                row_idx,
                                seq.num_prompt_tokens,
                                staging_start=0,
                            )
                        elif row_idx not in self._deltakv_full_prefill_plans:
                            raise RuntimeError(
                                "DeltaKV long-prefill offload lost its full-prefill plan between chunks. "
                                f"seq_id={seq.seq_id} row={row_idx} start={start_idx}."
                            )
                    else:
                        self._prepare_full_prefill_staging_plan(
                            seq,
                            row_idx,
                            end_idx,
                            staging_start=token_offset,
                        )
                else:
                    self._allocate_deltakv_full(seq.seq_id, chunk_size)
                    deltakv_slot_mapping[token_offset: token_offset + chunk_size] = \
                        self.sparse_layer_raw_slots_map[row_idx, start_idx:end_idx]

                self.row_seq_lens[row_idx] += chunk_size
                if use_long_prefill_offload_staging:
                    self._deltakv_long_prefill_offload_row_idx = int(row_idx)
                    self._deltakv_long_prefill_offload_start = int(start_idx)
                    self._deltakv_long_prefill_offload_end = int(end_idx)
                    self._deltakv_long_prefill_offload_total_len = int(seq.num_prompt_tokens)
                    self._deltakv_long_prefill_offload_is_last_chunk = bool(seq.is_last_chunk_prefill)
                context_lens_list.append(end_idx)
                req_indices.append(row_idx)

                chunk_tokens = seq.token_ids
                if len(chunk_tokens) > chunk_size:
                    chunk_tokens = chunk_tokens[start_idx:end_idx]

                input_ids_np[token_offset: token_offset + chunk_size] = chunk_tokens
                positions_np[token_offset: token_offset + chunk_size] = np.arange(start_idx, end_idx)

                cu_seqlens_q.append(cu_seqlens_q[-1] + chunk_size)
                token_offset += chunk_size

            context_lens = torch.tensor(context_lens_list, dtype=torch.int32, device=self.device)
            req_indices_tensor = torch.tensor(req_indices, dtype=torch.int32, device=self.device)

            full_state = self.full_layer_batch_states
            full_state.slot_mapping = full_slot_mapping
            full_state.context_lens = context_lens
            full_state.max_context_len = int(max(context_lens_list)) if context_lens_list else 0
            full_state.req_indices = req_indices_tensor

            deltakv_state = self.deltakv_layer_batch_states
            deltakv_state.slot_mapping = deltakv_slot_mapping
            deltakv_state.context_lens = context_lens
            deltakv_state.max_context_len = int(max(context_lens_list)) if context_lens_list else 0
            deltakv_state.req_indices = req_indices_tensor

            if use_full_prefill_staging:
                self._deltakv_prefill_staging_active = True
                self._deltakv_prefill_staging_slot_mapping = deltakv_slot_mapping
                max_context_len = int(max(context_lens_list)) if context_lens_list else 0
                active_slots = torch.full(
                    (len(seqs), max_context_len),
                    -1,
                    dtype=torch.int32,
                    device=self.device,
                )
                offset = 0
                for b_idx, seq in enumerate(seqs):
                    chunk_size = int(seq.current_chunk_size)
                    visible_len = int(seq.num_prefilled_tokens) + chunk_size if use_long_prefill_offload_staging else chunk_size
                    slot_start = 0 if use_long_prefill_offload_staging else offset
                    active_slots[b_idx, :visible_len] = torch.arange(
                        slot_start,
                        slot_start + visible_len,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    offset += chunk_size
                self._deltakv_prefill_staging_active_slots = active_slots
                self._deltakv_prefill_staging_req_indices = torch.arange(
                    len(seqs),
                    dtype=torch.int32,
                    device=self.device,
                )
                self._deltakv_prefill_staging_context_lens = context_lens

            input_ids = torch.from_numpy(input_ids_np).to(device=self.device)
            positions = torch.from_numpy(positions_np).to(device=self.device)
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=self.device)
            return input_ids, positions, cu_seqlens_q

    def _prepare_decode(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_decode"):
            batch_size = len(seqs)
            input_ids_list = [seq.decode_input_token for seq in seqs]
            positions_list = [seq.decode_input_position for seq in seqs]
            seq_ids = [seq.seq_id for seq in seqs]

            full_slot_mapping = torch.empty((batch_size,), dtype=torch.int32, device=self.device)
            deltakv_slot_mapping = torch.empty((batch_size,), dtype=torch.int32, device=self.device)

            full_slots = self._allocate_batch_full(seq_ids, 1)
            deltakv_slots = self._allocate_batch_deltakv_full(seq_ids, 1)
            full_slot_mapping[:] = full_slots
            deltakv_slot_mapping[:] = deltakv_slots

            row_indices = [self.seq_id_to_row[sid] for sid in seq_ids]
            self.row_seq_lens[row_indices] += 1
            context_lens = torch.tensor(
                self.row_seq_lens[row_indices],
                dtype=torch.int32,
                device=self.device,
            )
            req_indices = torch.tensor(row_indices, dtype=torch.int32, device=self.device)

            full_state = self.full_layer_batch_states
            full_state.slot_mapping = full_slot_mapping
            full_state.context_lens = context_lens
            full_state.req_indices = req_indices

            deltakv_state = self.deltakv_layer_batch_states
            deltakv_state.slot_mapping = deltakv_slot_mapping
            deltakv_state.context_lens = context_lens
            deltakv_state.req_indices = req_indices

            input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device=self.device)
            positions = torch.tensor(positions_list, dtype=torch.int64, device=self.device)
            return input_ids, positions, None

    def prepare_decode_static(
        self,
        seqs: list[Sequence],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        req_indices: torch.Tensor,
    ):
        """Prepare DeltaKV decode metadata into graph-stable caller-owned buffers."""
        with profiler.record("cache_prepare_decode"):
            # Static decode bypasses prepare_step(), so reset per-step DeltaKV
            # view planning here.  The cache key uses stable tensor addresses;
            # graph/static decode updates tensor contents in place each step.
            self._deltakv_reset_view_cache()
            real_batch_size = len(seqs)
            graph_batch_size = int(input_ids.numel())
            if real_batch_size <= 0:
                raise ValueError("Static DeltaKV decode requires a non-empty real decode batch.")
            if positions.numel() != graph_batch_size:
                raise ValueError("Static DeltaKV decode input buffers must have the same graph batch size.")
            if (
                slot_mapping.numel() != graph_batch_size
                or context_lens.numel() != graph_batch_size
                or req_indices.numel() != graph_batch_size
            ):
                raise ValueError("Static DeltaKV decode metadata buffers must have the same graph batch size.")
            if real_batch_size > graph_batch_size:
                raise ValueError(
                    "Static DeltaKV decode graph batch is smaller than the real decode batch: "
                    f"graph={graph_batch_size}, real={real_batch_size}."
                )

            input_ids_list = [seq.decode_input_token for seq in seqs]
            positions_list = [seq.decode_input_position for seq in seqs]
            seq_ids = [seq.seq_id for seq in seqs]

            full_slots = self._allocate_batch_full(seq_ids, 1)
            deltakv_slots = self._allocate_batch_deltakv_full(seq_ids, 1)

            row_indices = [self.seq_id_to_row[sid] for sid in seq_ids]
            self.row_seq_lens[row_indices] += 1
            real_context_lens = self.row_seq_lens[row_indices]

            input_ids[:real_batch_size].copy_(torch.tensor(input_ids_list, dtype=torch.int64, device=self.device))
            positions[:real_batch_size].copy_(torch.tensor(positions_list, dtype=torch.int64, device=self.device))
            context_lens[:real_batch_size].copy_(torch.tensor(real_context_lens, dtype=torch.int32, device=self.device))
            req_indices[:real_batch_size].copy_(torch.tensor(row_indices, dtype=torch.int32, device=self.device))

            if graph_batch_size > real_batch_size:
                first_context_len = int(real_context_lens[0])
                first_row_idx = int(row_indices[0])
                input_ids[real_batch_size:].fill_(int(input_ids_list[0]))
                positions[real_batch_size:].fill_(int(positions_list[0]))
                context_lens[real_batch_size:].fill_(first_context_len)
                req_indices[real_batch_size:].fill_(first_row_idx)

            full_slot_mapping = slot_mapping
            full_slot_mapping[:real_batch_size].copy_(full_slots)
            if graph_batch_size > real_batch_size:
                full_slot_mapping[real_batch_size:].fill_(-1)

            shape_key = tuple(int(x) for x in slot_mapping.shape)
            slot_cache = getattr(self, "_deltakv_decode_static_slot_mapping_by_shape", None)
            if slot_cache is None:
                slot_cache = {}
                self._deltakv_decode_static_slot_mapping_by_shape = slot_cache
            deltakv_slot_mapping = slot_cache.get(shape_key)
            if deltakv_slot_mapping is None or deltakv_slot_mapping.device != slot_mapping.device:
                deltakv_slot_mapping = torch.empty_like(slot_mapping)
                slot_cache[shape_key] = deltakv_slot_mapping
            self._deltakv_decode_static_slot_mapping = deltakv_slot_mapping
            deltakv_slot_mapping[:real_batch_size].copy_(deltakv_slots)
            if graph_batch_size > real_batch_size:
                deltakv_slot_mapping[real_batch_size:].fill_(-1)

            lens_key = tuple(int(x) for x in context_lens.shape)
            lens_cache = getattr(self, "_deltakv_decode_static_compressed_lens_by_shape", None)
            if lens_cache is None:
                lens_cache = {}
                self._deltakv_decode_static_compressed_lens_by_shape = lens_cache
            compressed_lens = lens_cache.get(lens_key)
            if compressed_lens is None or compressed_lens.device != context_lens.device:
                compressed_lens = torch.empty_like(context_lens)
                lens_cache[lens_key] = compressed_lens
            self._deltakv_decode_static_compressed_lens = compressed_lens
            real_compressed_lens = self.row_deltakv_compressed_lens[row_indices]
            compressed_lens[:real_batch_size].copy_(
                torch.tensor(real_compressed_lens, dtype=torch.int32, device=self.device)
            )
            if graph_batch_size > real_batch_size:
                compressed_lens[real_batch_size:].fill_(int(real_compressed_lens[0]))

            # CUDA Graph replay requires all decode-side score/view tensors to keep the
            # shape captured during warmup/capture.  The actual per-row lengths still
            # live in context_lens; max_context_len is only a capacity hint for static
            # buffers such as sparse-controller attn_score.
            static_context_cap = getattr(self, "_decode_static_max_context_len", None)
            state_max_context_len = (
                int(static_context_cap)
                if static_context_cap is not None
                else (int(max(real_context_lens)) if row_indices else 0)
            )

            full_state = self.full_layer_batch_states
            full_state.slot_mapping = full_slot_mapping
            full_state.context_lens = context_lens
            full_state.max_context_len = state_max_context_len
            full_state.req_indices = req_indices

            deltakv_state = self.deltakv_layer_batch_states
            deltakv_state.slot_mapping = deltakv_slot_mapping
            deltakv_state.context_lens = context_lens
            deltakv_state.max_context_len = state_max_context_len
            deltakv_state.req_indices = req_indices

            return input_ids, positions, None

    def get_compressed_lens(self, req_indices: torch.Tensor) -> torch.Tensor:
        if self._use_decode_static_paths():
            compressed_lens = getattr(self, "_deltakv_decode_static_compressed_lens", None)
            if compressed_lens is not None:
                return compressed_lens[: req_indices.numel()]
            return self.row_deltakv_compressed_lens_gpu[req_indices.to(torch.long)].to(torch.int32)
        if not get_context().is_prefill:
            return self.row_deltakv_compressed_lens_gpu[req_indices.to(torch.long)].to(torch.int32)
        compressed = self.row_deltakv_compressed_lens[req_indices.cpu().numpy()]
        return torch.tensor(compressed, dtype=torch.int32, device=self.device)

    @staticmethod
    def _metric_l2(kv_states, all_centers):
        """Compute an L2-equivalent *ranking* score for top-k selection.

        For squared L2 distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*dot(a,b).
        For a fixed `a`, ||a||^2 is constant across all `b`, so argmin(||a-b||^2)
        is equivalent to argmax(2*dot(a,b) - ||b||^2).

        We return `scores = 2*dot(a,b) - ||b||^2` (higher is better), keeping the
        large (N, M) score matrix in bf16/fp16 to avoid fp32 bandwidth overhead.
        """
        # kv_states: (1, N, D), all_centers: (1, M, D) for eviction.
        a = kv_states[0]
        b = all_centers[0]
        if a.numel() == 0 or b.numel() == 0:
            return kv_states.new_empty((kv_states.shape[0], kv_states.shape[1], all_centers.shape[1]))

        # GEMM via cuBLAS / tensorcores; output stays in low precision.
        dot = torch.matmul(a, b.transpose(0, 1))  # (N, M)

        # Keep norm computation small; cast down for the broadcast combine.
        b_norm = (b * b).sum(dim=1, dtype=torch.float32).to(dot.dtype)  # (M,)
        scores = dot.mul(2.0).sub_(b_norm.unsqueeze(0))
        return scores.unsqueeze(0)

    @staticmethod
    def _metric_dot(kv_states, all_centers):
        # Used only for top-k ranking; keep low precision for speed.
        return torch.matmul(kv_states, all_centers.transpose(-1, -2))

    @staticmethod
    def _metric_cosine(kv_states, all_centers, eps: float = 1e-6):
        # Keep normalization in fp32; matmul in fp32 since inputs are normalized anyway.
        kv_states_f = kv_states.float()
        all_centers_f = all_centers.float()
        kv_norm = kv_states_f / (kv_states_f.norm(p=2, dim=-1, keepdim=True) + eps)
        c_norm = all_centers_f / (all_centers_f.norm(p=2, dim=-1, keepdim=True) + eps)
        return torch.matmul(kv_norm, c_norm.transpose(-1, -2))

    def _gather_raw_kv_by_slots(
        self,
        layer_idx: int,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """Gather sparse-layer KV (concat) in the raw pre-RoPE K space.

        Returns: (N, kv_dim) on CUDA.
        """
        assert layer_idx in self.deltakv_layer_to_idx
        l_idx = self.deltakv_layer_to_idx[layer_idx]
        k_cache = self.deltakv_full_kv_cache[0, l_idx]
        v_cache = self.deltakv_full_kv_cache[1, l_idx]

        slots_i64 = slots.to(torch.long)
        k_raw = k_cache[slots_i64]  # (N, kv_heads, head_dim)
        v = v_cache[slots_i64]

        pos = self.deltakv_slot_to_pos[slots.to(torch.int32)].to(torch.long)
        if (pos < 0).any():
            raise RuntimeError("DeltaKV: center slot has unknown position (deltakv_slot_to_pos == -1).")

        kv_dim_half = self.num_kv_heads * self.head_dim
        k_flat = k_raw.reshape(-1, kv_dim_half)
        v_flat = v.reshape(-1, kv_dim_half)
        return torch.cat([k_flat, v_flat], dim=-1)

    def _cluster_compress(
        self,
        layer_idx: int,
        kv_states: torch.Tensor,  # (1, N, kv_dim), de-RoPE already applied on K
        existing_center_slots: torch.Tensor,  # (M0,)
        cluster_step: int,
        new_center_rel: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute top-k father *slots* and per-token base KV mean for a contiguous block.

        Returns:
          father_slots: (N, K) int32 physical slots
          base_kv: (1, N, kv_dim) float/bf16 mean of father KVs (de-RoPE space)
        """
        assert kv_states.dim() == 3 and kv_states.shape[0] == 1
        _, n, kv_dim = kv_states.shape
        k_neighbors = int(self.config.deltakv_neighbor_count)

        # Existing centers are always visible (from earlier blocks). New centers come from this block.
        with profiler.record("deltakv_cluster_existing_centers"):
            existing_centers = (
                self._gather_raw_kv_by_slots(layer_idx, existing_center_slots).unsqueeze(0)
                if existing_center_slots.numel() > 0
                else kv_states.new_zeros((1, 0, kv_dim))
            )
        if new_center_rel is None:
            new_center_rel = torch.arange(0, n, max(1, int(cluster_step)), device=kv_states.device, dtype=torch.long)
        else:
            new_center_rel = new_center_rel.to(device=kv_states.device, dtype=torch.long)
            if new_center_rel.numel() > 0 and ((new_center_rel < 0).any() or (new_center_rel >= n).any()):
                raise RuntimeError("DeltaKV center positions are outside the current block.")
        new_centers = kv_states.index_select(1, new_center_rel) if new_center_rel.numel() else kv_states[:, :0, :]
        all_centers = torch.cat([existing_centers, new_centers], dim=1)  # (1, M, kv_dim)
        m0 = existing_centers.shape[1]
        m_new = new_centers.shape[1]

        metric_type = self.config.cluster_metric
        with profiler.record("deltakv_cluster_metric"):
            if metric_type == "l2":
                scores = self._metric_l2(kv_states, all_centers)
            elif metric_type == "dot":
                scores = self._metric_dot(kv_states, all_centers)
            elif metric_type == "cosine":
                scores = self._metric_cosine(kv_states, all_centers)
            elif metric_type == "fastdot":
                # Fast approximate metric: pure dot-product in low precision, no fp32 casts/norms.
                # Only used for top-k selection; accuracy is intentionally relaxed for speed.
                scores = torch.bmm(kv_states, all_centers.transpose(1, 2))
            else:
                raise ValueError(f"Unknown cluster_metric: {metric_type}")

        # Causal mask: within the current block, a token can only use new centers sampled at <= its index.
        if m_new > 0:
            with profiler.record("deltakv_cluster_causal_mask"):
                rows = torch.arange(n, device=kv_states.device).view(n, 1)
                cols = new_center_rel.view(1, m_new)
                mask_new = cols <= rows  # (N, m_new)
                mask_existing = torch.ones((n, m0), device=kv_states.device, dtype=torch.bool)
                full_mask = torch.cat([mask_existing, mask_new], dim=1)  # (N, M)
                scores = scores.masked_fill(~full_mask.unsqueeze(0), float("-inf"))

        k_eff = min(k_neighbors, all_centers.shape[1])
        if k_eff <= 0:
            raise RuntimeError("DeltaKV: no available centers to assign.")
        with profiler.record("deltakv_cluster_topk"):
            topk_indices = scores.topk(k=k_eff, dim=-1).indices  # (1, N, K)

        # Base KV mean in de-RoPE space.
        with profiler.record("deltakv_cluster_gather_mean"):
            gather_idx = topk_indices.view(1, -1)[:, :, None].expand(-1, -1, kv_dim)
            gathered = all_centers.gather(1, gather_idx).view(1, n, k_eff, kv_dim).mean(dim=2)
        return topk_indices.squeeze(0).to(torch.int32), gathered

    @torch.no_grad()
    def deltakv_evict(self, seqs: list[Sequence]):
        # Called from SparseController.post_forward(), which runs outside model forward.
        # Must be no-grad to avoid building enormous autograd graphs.
        with profiler.record("deltakv_evict_total"):
            self._deltakv_evict_impl(seqs)

    def _deltakv_evict_impl(self, seqs: list[Sequence]):
        if not self.deltakv_layer_ids:
            return
        sink = int(self.config.sink_keep_tokens)
        recent = int(self.config.recent_keep_tokens)
        cluster_step = self._deltakv_base_cluster_step()

        # Compress per sequence (long-text batches are typically small).
        for seq in seqs:
            with profiler.record("deltakv_evict_seq"):
                self._deltakv_evict_one_seq(seq, sink=sink, recent=recent, cluster_step=cluster_step)

    def _deltakv_evict_one_seq(self, seq: Sequence, *, sink: int, recent: int, cluster_step: int):
        row_idx = self.seq_id_to_row.get(seq.seq_id, None)
        if row_idx is None:
            return

        total_len = int(self.row_seq_lens[row_idx])
        compressed_len = int(self.row_deltakv_compressed_lens[row_idx])  # length of finalized history (excluding sink)
        buffer_start = sink + compressed_len
        buffer_len = total_len - buffer_start
        if buffer_len <= recent:
            return

        # Evict as much as possible, but keep at least `recent` tokens in the uncompressed buffer.
        # Match the reference logic: compress in multiples of `recent` (tail_token_size).
        evict_len = ((buffer_len - recent) // recent) * recent
        if evict_len <= 0:
            return

        evict_start = buffer_start
        evict_end = evict_start + evict_len

        # Raw slots exist for the evicted block before we start.
        with profiler.record("deltakv_evict_read_slots"):
            raw_slots_block = self.sparse_layer_raw_slots_map[row_idx, evict_start:evict_end].clone()
        if (raw_slots_block < 0).any():
            raise RuntimeError("DeltaKV eviction expects raw slots for the buffer block.")

        # Select new centers (prototypes) by fixed stride within the evicted block.
        with profiler.record("deltakv_evict_select_centers"):
            center_rel = self._deltakv_center_rel_for_block(
                row_idx,
                start=evict_start,
                end=evict_end,
                update_state=True,
            )
            new_center_slots = raw_slots_block[center_rel].to(torch.int32)

        # Initialize per-layer center slots (previous centers, without current block).
        with profiler.record("deltakv_evict_prev_centers"):
            sink_slots = self.sparse_layer_raw_slots_map[row_idx, :sink].to(torch.int32)
            prev_center_slots_by_layer: dict[int, torch.Tensor] = {}
            for layer_idx in self.deltakv_layer_ids:
                existing = self.row_deltakv_center_slots[row_idx][layer_idx]
                prev_center_slots_by_layer[layer_idx] = (sink_slots if existing is None else existing.to(torch.int32))

        # Raw-K KV for the whole evicted block per layer, compute assignments,
        # and store latents for every finalized token. Center tokens also stay
        # as full KV references and remain mapped.
        with profiler.record("deltakv_evict_build_masks"):
            is_center = torch.zeros((evict_len,), device=self.device, dtype=torch.bool)
            is_center[center_rel] = True
            to_store_mask = torch.ones((evict_len,), device=self.device, dtype=torch.bool)
            to_free_mask = ~is_center

        # Allocate shared latent slots for this block (shared index across layers).
        with profiler.record("deltakv_evict_alloc_latent"):
            latent_slots = self._allocate_deltakv_latent(evict_len).to(torch.int32)
            pos_all = torch.arange(evict_start, evict_end, device=self.device, dtype=torch.long)
            pos_to_free = pos_all[to_free_mask]
            # Map every finalized history position, including centers, to a
            # latent payload. Raw slots for centers remain mapped separately.
            self.sparse_layer_latent_slots_map[row_idx, pos_all] = latent_slots

        for layer_idx in self.deltakv_layer_ids:
            l_idx = self.deltakv_layer_to_idx[layer_idx]
            k_cache = self.deltakv_full_kv_cache[0, l_idx]
            v_cache = self.deltakv_full_kv_cache[1, l_idx]

            # Gather KV for block tokens from raw slots.
            with profiler.record("deltakv_evict_gather_kv"):
                slots_i64 = raw_slots_block.to(torch.long)
                k_raw = k_cache[slots_i64]
                v = v_cache[slots_i64]

            kv_dim_half = self.num_kv_heads * self.head_dim
            with profiler.record("deltakv_evict_build_kv_block"):
                kv_block = torch.cat(
                    [k_raw.reshape(evict_len, kv_dim_half), v.reshape(evict_len, kv_dim_half)],
                    dim=-1,
                ).unsqueeze(0)  # (1, N, kv_dim)

            existing_center_slots = prev_center_slots_by_layer[layer_idx]
            # Compute top-k father indices (into all_centers) + base KV mean.
            with profiler.record("deltakv_evict_cluster"):
                topk_center_indices, base_kv = self._cluster_compress(
                    layer_idx=layer_idx,
                    kv_states=kv_block,
                    existing_center_slots=existing_center_slots,
                    cluster_step=cluster_step,
                    new_center_rel=center_rel,
                )

            # Remap center indices -> physical slots.
            # all_center_slots = [existing centers..., new centers...]
            with profiler.record("deltakv_evict_remap_fathers"):
                all_center_slots = torch.cat([existing_center_slots, new_center_slots], dim=0)  # (M,)
                father_slots_full = all_center_slots[topk_center_indices.to(torch.long)]  # (N, K)
                father_slots = father_slots_full  # (N, K)

            # Store father slots for reconstruction.
            with profiler.record("deltakv_evict_store_fathers"):
                K = self.deltakv_latent_to_full_slots.shape[-1]
                k_eff = father_slots.shape[1]
                if k_eff < K:
                    pad = father_slots[:, :1].expand(-1, K - k_eff)
                    father_slots = torch.cat([father_slots, pad], dim=1)
                elif k_eff > K:
                    father_slots = father_slots[:, :K]
                self.deltakv_latent_to_full_slots[l_idx, latent_slots] = father_slots.to(torch.int32)

            # Latent residual in compressed space.
            down = self.compress_down[l_idx]
            with profiler.record("deltakv_evict_compress_down"):
                kv_down = down(kv_block).squeeze(0)  # (N, latent_dim)
                base_down = down(base_kv).squeeze(0)
                latent_all = (kv_down - base_down)[to_store_mask]  # (N, latent_dim)
            with profiler.record("deltakv_evict_store_latent"):
                self._store_deltakv_latent(l_idx, latent_slots, latent_all)

        # Append new centers after this block is processed (so "existing" for next blocks).
        with profiler.record("deltakv_evict_append_centers"):
            for layer_idx in self.deltakv_layer_ids:
                self.row_deltakv_center_slots[row_idx][layer_idx] = torch.cat(
                    [prev_center_slots_by_layer[layer_idx], new_center_slots], dim=0
                )

        # Free full-KV slots for non-center tokens in the evicted block.
        with profiler.record("deltakv_evict_free_full_slots"):
            free_slots = self._filter_deltakv_center_slots_for_evict_free(
                row_idx,
                raw_slots_block[to_free_mask].to(torch.int32),
                extra_center_slots=new_center_slots,
            )
            ptr = self._num_free_slots_deltakv_full
            self.free_slots_stack_deltakv_full[ptr: ptr + free_slots.numel()] = free_slots
            self._num_free_slots_deltakv_full += free_slots.numel()
            self.deltakv_slot_to_pos[free_slots] = -1

        # Mark compressed tokens as not having full KV anymore.
        with profiler.record("deltakv_evict_update_maps"):
            self.sparse_layer_raw_slots_map[row_idx, pos_to_free] = -1
            # Finalized history grows by the whole evicted block (centers are also part of history).
            self.row_deltakv_compressed_lens[row_idx] += evict_len
            self.row_deltakv_compressed_lens_gpu[row_idx] += int(evict_len)

    def deltakv_reconstruct(
        self,
        layer_idx: int,
        active_compressed_indices: torch.Tensor | None,
        context_lens: torch.Tensor,
        req_indices: torch.Tensor,
        chunk_lens: torch.Tensor | None,
        return_reconstruct_temp_slots: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build DeltaKV reading view for a sparse layer and reconstruct missing KV into scratch slots.

        Args:
          layer_idx: actual model layer id
          active_compressed_indices: (B, Kmax) compressed-candidate indices after the sink window,
            padded with -1 if needed; may be None
          context_lens: (B,) desired view lengths (sink + topk + buffer_len_total)
          req_indices: (B,) global row indices (into slot maps)
          chunk_lens: (B,) length of current chunk (prefill) or None for decode

        Returns:
          active_slots: (B, max_s) int32, Req->slots for kernels (local indexing by batch row)
          local_req_indices: (B,) int32 = arange(B)
          new_context_lens: (B,) int32 actual view lengths (sink + topk + buffer_len_total)
          temp_slots: (Nt,) int32 scratch slots to be freed after attention
        """
        with profiler.record("deltakv_reconstruct_total"):
            active_slots, local_req, new_context_lens, temp_slots, recon_pos, recon_latent, recon_out_slot = (
                self._deltakv_build_view_and_plan_reconstruct(layer_idx, active_compressed_indices, req_indices)
            )
            active_pos = None if get_context().is_prefill else getattr(self, "_deltakv_decode_static_active_pos", None)

        l_idx = self.deltakv_layer_to_idx[layer_idx]
        k_cache = self.deltakv_full_kv_cache[0, l_idx]
        v_cache = self.deltakv_full_kv_cache[1, l_idx]

        if recon_latent.numel() > 0:
            kv_dim_half = self.num_kv_heads * self.head_dim
            with profiler.record("deltakv_reconstruct_compress_up"):
                latent = self._load_deltakv_latent(l_idx, recon_latent)  # (Nt, latent_dim)
                kv_delta = self.compress_up[l_idx](latent)  # (Nt, kv_dim)

            with profiler.record("deltakv_reconstruct_read_fathers"):
                father_slots = self.deltakv_latent_to_full_slots[l_idx, recon_latent].to(torch.int32)  # (Nt, K)
            if (father_slots < 0).any():
                raise RuntimeError("DeltaKV: missing father slots for reconstruction.")

            # Torch reconstruction baseline. Father center slots are stored as raw pre-RoPE K.
            with profiler.record("deltakv_reconstruct_gather_raw_fathers"):
                k_father_raw = k_cache[father_slots.to(torch.long)]  # (Nt, K, kv_heads, head_dim)
                v_father = v_cache[father_slots.to(torch.long)]
                father_pos = self.deltakv_slot_to_pos[father_slots].to(torch.long)  # (Nt, K)
            if (father_pos < 0).any():
                raise RuntimeError("DeltaKV: father center slot has unknown position.")
            with profiler.record("deltakv_reconstruct_mean_fathers"):
                kv_father = torch.cat(
                    [
                        k_father_raw.reshape(k_father_raw.shape[0], k_father_raw.shape[1], kv_dim_half),
                        v_father.reshape(v_father.shape[0], v_father.shape[1], kv_dim_half),
                    ],
                    dim=-1,
                ).mean(dim=1)  # (Nt, kv_dim)

            with profiler.record("deltakv_reconstruct_apply_delta_and_rope"):
                kv_raw = kv_delta + kv_father  # (Nt, kv_dim)
                k_raw = kv_raw[:, :kv_dim_half].reshape(-1, self.num_kv_heads, self.head_dim)
                v_out = kv_raw[:, kv_dim_half:].reshape(-1, self.num_kv_heads, self.head_dim)
                k_normed = self._apply_sparse_k_norm_if_needed(l_idx, k_raw)

                cos_sin_t = self.cos_sin_cache[recon_pos]  # (Nt, 1, head_dim)
                cos_t, sin_t = cos_sin_t.chunk(2, dim=-1)
                k_out = apply_rotary_emb(k_normed, cos_t, sin_t)

            with profiler.record("deltakv_reconstruct_writeback"):
                out_i64 = recon_out_slot.to(torch.long)
                k_cache[out_i64] = k_out.to(k_cache.dtype)
                v_cache[out_i64] = v_out.to(v_cache.dtype)
                self.deltakv_slot_to_pos[out_i64] = recon_pos.to(torch.int32)

        attn_active_slots = active_slots.clone()
        attn_active_slots, materialized_temp_slots = self._materialize_deltakv_active_postrope_view(
            layer_idx,
            attn_active_slots,
            new_context_lens,
            recon_out_slot,
            active_pos,
        )

        debug_layers = os.getenv("SPARSEENGINE_DEBUG_RECONSTRUCT_LAYERS")
        if debug_layers:
            wanted = {int(part) for part in debug_layers.split(",") if part.strip()}
            if int(layer_idx) in wanted and active_slots.shape[0] > 0:
                active_len = int(new_context_lens[0].item())
                slots = attn_active_slots[0, :active_len].to(torch.long)
                if (slots < 0).any():
                    raise RuntimeError(f"DeltaKV reconstruct debug saw negative active slot: layer={layer_idx}.")
                active_pos = self.deltakv_slot_to_pos[slots].to(torch.int32)
                if (active_pos < 0).any():
                    raise RuntimeError(f"DeltaKV reconstruct debug saw unknown slot position: layer={layer_idx}.")
                debug = getattr(self, "debug_last_reconstruct", {})
                debug[int(layer_idx)] = {
                    "positions": active_pos.detach().cpu(),
                    "k": k_cache[slots].permute(1, 0, 2).unsqueeze(0).detach().float().cpu(),
                    "v": v_cache[slots].permute(1, 0, 2).unsqueeze(0).detach().float().cpu(),
                }
                self.debug_last_reconstruct = debug

        returned_temp_slots = []
        if materialized_temp_slots.numel() > 0:
            returned_temp_slots.append(materialized_temp_slots.to(active_slots.device))
        if self._should_return_reconstruct_temp_slots(
            static_decode=static_decode,
            return_reconstruct_temp_slots=return_reconstruct_temp_slots,
            temp_slots=temp_slots,
        ):
            returned_temp_slots.append(temp_slots)
        if returned_temp_slots:
            temp_slots = torch.cat(returned_temp_slots, dim=0).to(torch.int32)
        else:
            temp_slots = torch.empty((0,), device=active_slots.device, dtype=torch.int32)
        return attn_active_slots, local_req, new_context_lens, temp_slots

    def _deltakv_build_view_and_plan_reconstruct(
        self,
        layer_idx: int,
        active_compressed_indices: torch.Tensor | None,
        req_indices: torch.Tensor,
    ) -> tuple[
        torch.Tensor,  # active_slots
        torch.Tensor,  # local_req_indices
        torch.Tensor,  # new_context_lens
        torch.Tensor,  # temp_slots
        torch.Tensor,  # recon_pos (int32)
        torch.Tensor,  # recon_latent (int32)
        torch.Tensor,  # recon_out_slot (int32)
    ]:
        ctx = get_context()
        if self._use_decode_static_paths():
            return self._deltakv_build_view_and_plan_reconstruct_static(
                layer_idx,
                active_compressed_indices,
                req_indices,
            )
        if self._is_stream_capturing():
            capture_view = self._deltakv_build_prefill_capture_view(layer_idx, active_compressed_indices, req_indices)
            if capture_view is not None:
                return capture_view
            if not getattr(ctx, "is_prefill", False):
                return self._deltakv_build_view_and_plan_reconstruct_static(
                    layer_idx,
                    active_compressed_indices,
                    req_indices,
                )

        req_ptr = int(req_indices.data_ptr()) if req_indices is not None and req_indices.numel() > 0 else 0
        req_n = int(req_indices.numel()) if req_indices is not None else 0
        if active_compressed_indices is None:
            act_ptr = 0
            act_b = req_n
            act_k = 0
        else:
            act_ptr = int(active_compressed_indices.data_ptr()) if active_compressed_indices.numel() > 0 else int(active_compressed_indices.data_ptr())
            act_b = int(active_compressed_indices.shape[0])
            act_k = int(active_compressed_indices.shape[1])

        key = (req_ptr, req_n, act_ptr, act_b, act_k)
        if self._deltakv_view_cache_key == key and self._deltakv_view_cache_value is not None:
            with profiler.record("deltakv_build_view_cache_hit"):
                return self._deltakv_view_cache_value

        with profiler.record("deltakv_build_view_total"):
            out = self._deltakv_build_view_and_plan_reconstruct_impl(layer_idx, active_compressed_indices, req_indices)
        self._deltakv_view_cache_key = key
        self._deltakv_view_cache_value = out
        return out


    @staticmethod
    def _should_return_reconstruct_temp_slots(
        *,
        static_decode: bool,
        return_reconstruct_temp_slots: bool,
        temp_slots: torch.Tensor,
    ) -> bool:
        """Return True only for one-shot DeltaKV scratch slots.

        Static decode slots are resident graph/eager-static workspaces owned by
        _ensure_decode_static_temp_slots(). Releasing them through
        Attention.forward(finally) would put graph replay output slots back on
        the general raw-slot free stack.
        """
        return (
            (not bool(static_decode))
            and bool(return_reconstruct_temp_slots)
            and temp_slots is not None
            and temp_slots.numel() > 0
        )

    def _ensure_decode_static_temp_slots(self, batch_size: int, k_max: int) -> torch.Tensor:
        batch_size = int(batch_size)
        k_max = int(k_max)
        if batch_size < 0 or k_max < 0:
            raise ValueError(f"Invalid static DeltaKV temp shape: batch={batch_size}, k={k_max}.")
        cache = getattr(self, "_deltakv_decode_static_temp_slots_by_shape", None)
        if cache is None:
            cache = {}
            self._deltakv_decode_static_temp_slots_by_shape = cache
        key = (batch_size, k_max)
        slots = cache.get(key)
        if slots is not None:
            return slots
        if batch_size == 0 or k_max == 0:
            slots = torch.empty((batch_size, k_max), device=self.device, dtype=torch.int32)
        else:
            slots = self._allocate_temp_deltakv_full(batch_size * k_max).to(torch.int32).view(batch_size, k_max)
            self._deltakv_static_temp_slots_reserved_total = int(
                getattr(self, "_deltakv_static_temp_slots_reserved_total", 0) or 0
            ) + int(batch_size * k_max)
        cache[key] = slots
        return slots

    def _ensure_decode_static_plan_buffers(self, batch_size: int, k_max: int, max_s: int, device: torch.device):
        batch_size = int(batch_size)
        k_max = int(k_max)
        max_s = int(max_s)
        if batch_size < 0 or k_max < 0 or max_s < 0:
            raise ValueError(f"Invalid static DeltaKV plan shape: batch={batch_size}, k={k_max}, max_s={max_s}.")
        cache = getattr(self, "_deltakv_decode_static_plan_buffers_by_shape", None)
        if cache is None:
            cache = {}
            self._deltakv_decode_static_plan_buffers_by_shape = cache
        key = (batch_size, k_max, max_s, str(device))
        buffers = cache.get(key)
        if buffers is not None:
            return buffers

        active_slots = torch.empty((batch_size, max_s), device=device, dtype=torch.int32)
        active_pos = torch.empty((batch_size, max_s), device=device, dtype=torch.int32)
        local_req = torch.arange(batch_size, device=device, dtype=torch.int32)
        new_context_lens = torch.empty((batch_size,), device=device, dtype=torch.int32)
        recon_size = batch_size * k_max
        recon_pos = torch.empty((recon_size,), device=device, dtype=torch.int32)
        recon_latent = torch.empty((recon_size,), device=device, dtype=torch.int32)
        recon_out_slot = torch.empty((recon_size,), device=device, dtype=torch.int32)
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        buffers = (active_slots, active_pos, local_req, new_context_lens, empty, recon_pos, recon_latent, recon_out_slot)
        cache[key] = buffers
        return buffers

    def _deltakv_decode_static_max_buffer(self) -> int:
        recent = int(self.config.recent_keep_tokens)
        # Decode runs before post-forward eviction. DeltaKV evicts raw tail
        # tokens in recent-sized chunks, so the visible uncompressed tail can
        # include one recent window plus the next remainder/current token.
        return max(recent + 1, 2 * recent)

    def _deltakv_build_view_and_plan_reconstruct_static(
        self,
        layer_idx: int,
        active_compressed_indices: torch.Tensor | None,
        req_indices: torch.Tensor,
    ):
        if layer_idx in self.full_layer_to_idx:
            raise ValueError("deltakv_reconstruct should only be called for sparse layers.")

        bsz = int(req_indices.shape[0])
        if active_compressed_indices is None:
            active_compressed_indices = torch.empty((bsz, 0), device=req_indices.device, dtype=torch.int32)
        k_max = int(active_compressed_indices.shape[1])
        if bsz == 0:
            empty = torch.empty((0,), device=req_indices.device, dtype=torch.int32)
            return torch.empty((0, 0), device=req_indices.device, dtype=torch.int32), empty, empty, empty, empty, empty, empty

        context_lens = self.deltakv_layer_batch_states.context_lens
        if context_lens is None:
            raise RuntimeError("DeltaKV static decode context_lens buffer was not initialized.")
        context_lens = context_lens[:bsz].to(torch.int32)
        compressed_lens = self.get_compressed_lens(req_indices).to(torch.int32)

        sink = int(self.config.sink_keep_tokens)
        max_buffer = self._deltakv_decode_static_max_buffer()
        max_s = sink + k_max + max_buffer
        temp_slots = self._ensure_decode_static_temp_slots(bsz, k_max)
        buffers = self._ensure_decode_static_plan_buffers(bsz, k_max, max_s, req_indices.device)
        active_slots, active_pos, local_req, new_context_lens, no_free_temp_slots, recon_pos, recon_latent, recon_out_slot = buffers

        from sparseengine.kernels.triton.deltakv_kernels import deltakv_static_decode_plan

        deltakv_static_decode_plan(
            raw_slots_map=self.sparse_layer_raw_slots_map,
            latent_slots_map=self.sparse_layer_latent_slots_map,
            active_compressed_indices=active_compressed_indices,
            req_indices=req_indices,
            context_lens=context_lens,
            compressed_lens=compressed_lens,
            temp_slots=temp_slots,
            active_slots_out=active_slots,
            active_pos_out=active_pos,
            new_context_lens_out=new_context_lens,
            recon_pos_out=recon_pos,
            recon_latent_out=recon_latent,
            recon_out_slot_out=recon_out_slot,
            sink=sink,
            max_buffer=max_buffer,
        )
        self._deltakv_decode_static_active_pos = active_pos
        return active_slots, local_req, new_context_lens, no_free_temp_slots, recon_pos, recon_latent, recon_out_slot

    def _deltakv_build_prefill_capture_view(
        self,
        layer_idx: int,
        active_compressed_indices: torch.Tensor | None,
        req_indices: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ] | None:
        from sparseengine.utils.context import get_context

        ctx = get_context()
        if not getattr(ctx, "is_prefill", False) or layer_idx in self.full_layer_to_idx:
            return None
        active_k = 0 if active_compressed_indices is None else int(active_compressed_indices.shape[-1])
        if active_k != 0:
            return None

        empty = torch.empty((0,), device=self.device, dtype=torch.int32)
        if self.has_prefill_staging_view(layer_idx):
            return (
                self._deltakv_prefill_staging_active_slots,
                self._deltakv_prefill_staging_req_indices,
                self._deltakv_prefill_staging_context_lens,
                empty,
                empty,
                empty,
                empty,
            )

        seqs = getattr(ctx, "seqs", None)
        if seqs is None or len(seqs) != int(req_indices.numel()):
            return None
        if any(int(seq.num_prefilled_tokens) != 0 for seq in seqs):
            return None

        max_context_len = self.deltakv_layer_batch_states.max_context_len
        if max_context_len is None:
            return None
        max_context_len = int(max_context_len)
        rows = req_indices.to(torch.long)
        active_slots = self.sparse_layer_raw_slots_map.index_select(0, rows)[:, :max_context_len].to(torch.int32)
        local_req = torch.arange(int(req_indices.numel()), device=self.device, dtype=torch.int32)
        return (
            active_slots,
            local_req,
            self.deltakv_layer_batch_states.context_lens,
            empty,
            empty,
            empty,
            empty,
        )

    def _deltakv_build_view_and_plan_reconstruct_impl(
        self,
        layer_idx: int,
        active_compressed_indices: torch.Tensor | None,
        req_indices: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if layer_idx in self.full_layer_to_idx:
            raise ValueError("deltakv_reconstruct should only be called for sparse layers.")
        if active_compressed_indices is None:
            active_compressed_indices = torch.empty((req_indices.numel(), 0), device=self.device, dtype=torch.int32)

        bsz = int(req_indices.numel())
        if bsz == 0:
            empty0 = torch.empty((0,), device=self.device, dtype=torch.int32)
            return torch.empty((0, 0), device=self.device, dtype=torch.int32), empty0, empty0, empty0, empty0, empty0, empty0

        from sparseengine.kernels.triton.deltakv_eager_plan import (
            finish_eager_plan,
            prepare_eager_plan,
        )

        sink = int(self.config.sink_keep_tokens)
        req_indices = req_indices.contiguous()
        req_indices_cpu = req_indices.cpu().numpy()
        total_lens = self.row_seq_lens[req_indices_cpu]
        compressed_lens = self.row_deltakv_compressed_lens[req_indices_cpu]
        lengths = torch.tensor(
            np.stack((total_lens, compressed_lens), axis=1),
            dtype=torch.int32,
            device=self.device,
        )
        plan = prepare_eager_plan(
            self.sparse_layer_raw_slots_map,
            self.sparse_layer_latent_slots_map,
            req_indices,
            lengths,
            active_compressed_indices.to(torch.int32),
            sink,
        )
        # Exact allocation remains host-owned. Read all counts and validation
        # flags together instead of synchronizing for every selected request.
        stats = plan[-1].cpu().tolist()
        context_lens = []
        for total, compressed, (kept, needed, bad_sink, bad_tail) in zip(
            total_lens, compressed_lens, stats
        ):
            if bad_sink:
                raise RuntimeError("DeltaKV: missing full slots in sink window.")
            if bad_tail:
                raise RuntimeError("DeltaKV: buffer contains missing full slots.")
            sink_len = min(sink, int(total))
            compressed = min(int(compressed), max(0, int(total) - sink_len))
            context_lens.append(int(total) - compressed + kept)
        needed_total = sum(row[1] for row in stats)
        if needed_total > self._num_free_slots_deltakv_full:
            raise RuntimeError(
                "Out of DeltaKV full cache slots (temp). "
                f"need={needed_total} free={self._num_free_slots_deltakv_full}."
            )
        allocated = [
            self._allocate_temp_deltakv_full(row[1]) for row in stats if row[1]
        ]
        empty = torch.empty(0, dtype=torch.int32, device=self.device)
        temp_slots = torch.cat(allocated) if allocated else empty
        active_slots = torch.empty(
            (bsz, max(context_lens)), dtype=torch.int32, device=self.device
        )
        new_context_lens = torch.empty((bsz,), dtype=torch.int32, device=self.device)
        recon_pos = torch.empty_like(temp_slots)
        recon_latent = torch.empty_like(temp_slots)
        finish_eager_plan(
            self.sparse_layer_raw_slots_map,
            req_indices,
            lengths,
            plan,
            temp_slots,
            active_slots,
            recon_pos,
            recon_latent,
            new_context_lens,
            sink,
        )
        local_req = torch.arange(bsz, device=self.device, dtype=torch.int32)
        return (
            active_slots,
            local_req,
            new_context_lens,
            temp_slots,
            recon_pos,
            recon_latent,
            temp_slots.clone(),
        )


class DeltaKVCacheTritonManagerV4(DeltaKVCacheManager):
    """Single maintained Triton DeltaKV runtime.

    Historical method names (`deltakv`, `v2`, `v3`) are kept as config
    aliases, but the implementation is consolidated here so new CUDA-graph and
    kernel work does not have to support slower legacy class variants.
    """

    def _deltakv_gather_raw_kv(
        self,
        *,
        slots: torch.Tensor,
        pos: torch.Tensor,
        cos_sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        from sparseengine.kernels.triton.deltakv_kernels import deltakv_gather_raw_kv_grouped_heads

        hp = int(getattr(self.config, "deltakv_triton_gather_heads_per_program", 4) or 1)
        hp = max(1, min(hp, int(self.num_kv_heads)))
        return deltakv_gather_raw_kv_grouped_heads(
            slots=slots,
            pos=pos,
            cos_sin=cos_sin,
            k_cache=k_cache,
            v_cache=v_cache,
            heads_per_program=hp,
        )

    def _deltakv_reconstruct_writeback(
        self,
        *,
        kv_delta: torch.Tensor,
        father_slots: torch.Tensor,
        slot_to_pos: torch.Tensor,
        out_slots: torch.Tensor,
        out_pos: torch.Tensor,
        cos_sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        l_idx: int | None = None,
    ):
        from sparseengine.kernels.triton.deltakv_kernels import deltakv_reconstruct_writeback_grouped_heads

        hp = int(getattr(self.config, "deltakv_triton_reconstruct_heads_per_program", 4) or 1)
        hp = max(1, min(hp, int(self.num_kv_heads)))
        k_norm_weight = None
        if l_idx is not None and getattr(self, "deltakv_k_norm_weight", None) is not None:
            k_norm_weight = self.deltakv_k_norm_weight[int(l_idx)]
        return deltakv_reconstruct_writeback_grouped_heads(
            kv_delta=kv_delta,
            father_slots=father_slots,
            slot_to_pos=slot_to_pos,
            out_slots=out_slots,
            out_pos=out_pos,
            cos_sin=cos_sin,
            k_cache=k_cache,
            v_cache=v_cache,
            heads_per_program=hp,
            k_norm_weight=k_norm_weight,
            k_norm_eps=float(getattr(self, "deltakv_k_norm_eps", 1e-6) or 1e-6),
            raw_k_cache=True,
            store_raw_k=False,
        )

    @torch.no_grad()
    def deltakv_reconstruct(
        self,
        layer_idx: int,
        active_compressed_indices: torch.Tensor | None,
        context_lens: torch.Tensor,
        req_indices: torch.Tensor,
        chunk_lens: torch.Tensor | None,
        return_reconstruct_temp_slots: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with profiler.record("deltakv_reconstruct_triton_total"):
            active_slots, local_req, new_context_lens, temp_slots, recon_pos, recon_latent, recon_out_slot = (
                self._deltakv_build_view_and_plan_reconstruct(layer_idx, active_compressed_indices, req_indices)
            )
            active_pos = None if get_context().is_prefill else getattr(self, "_deltakv_decode_static_active_pos", None)

            l_idx = self.deltakv_layer_to_idx[layer_idx]
            k_cache = self.deltakv_full_kv_cache[0, l_idx]
            v_cache = self.deltakv_full_kv_cache[1, l_idx]

            if recon_latent.numel() > 0:
                static_decode = not get_context().is_prefill
                safe_recon_latent = recon_latent.clamp_min(0) if static_decode else recon_latent
                with profiler.record("deltakv_reconstruct_triton_compress_up"):
                    latent = self._load_deltakv_latent(l_idx, safe_recon_latent)  # (Nt, latent_dim)
                    kv_delta = self.compress_up[l_idx](latent)  # (Nt, kv_dim) in raw-K space

                with profiler.record("deltakv_reconstruct_triton_read_fathers"):
                    father_slots = self.deltakv_latent_to_full_slots[l_idx, safe_recon_latent].to(torch.int32)  # (Nt, K)
                if static_decode:
                    father_slots = father_slots.clamp_min(0)
                elif (father_slots < 0).any():
                    raise RuntimeError("DeltaKV: missing father slots for reconstruction.")

                # cos_sin_cache: (max_pos, 1, head_dim) -> (max_pos, head_dim)
                cos_sin = self.cos_sin_cache[:, 0, :]

                with profiler.record("deltakv_reconstruct_triton_kernel"):
                    self._deltakv_reconstruct_writeback(
                        kv_delta=kv_delta,
                        father_slots=father_slots,
                        slot_to_pos=self.deltakv_slot_to_pos,
                        out_slots=recon_out_slot,
                        out_pos=recon_pos,
                        cos_sin=cos_sin,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        l_idx=l_idx,
                    )
                    out_i64 = recon_out_slot.clamp_min(0).to(torch.long)
                    prev_pos = self.deltakv_slot_to_pos[out_i64]
                    valid_out = recon_out_slot >= 0
                    self.deltakv_slot_to_pos[out_i64] = torch.where(
                        valid_out,
                        recon_pos.to(torch.int32),
                        prev_pos,
                    )

            attn_active_slots = active_slots.clone()
            attn_active_slots, materialized_temp_slots = self._materialize_deltakv_active_postrope_view(
                layer_idx,
                attn_active_slots,
                new_context_lens,
                recon_out_slot,
                active_pos,
            )

            returned_temp_slots = []
            if materialized_temp_slots.numel() > 0:
                returned_temp_slots.append(materialized_temp_slots.to(active_slots.device))
            if self._should_return_reconstruct_temp_slots(
                static_decode=static_decode,
                return_reconstruct_temp_slots=return_reconstruct_temp_slots,
                temp_slots=temp_slots,
            ):
                returned_temp_slots.append(temp_slots)
            if returned_temp_slots:
                temp_slots = torch.cat(returned_temp_slots, dim=0).to(torch.int32)
            else:
                temp_slots = torch.empty((0,), device=active_slots.device, dtype=torch.int32)
            return attn_active_slots, local_req, new_context_lens, temp_slots

    def _gather_raw_kv_by_slots(
        self,
        layer_idx: int,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        with profiler.record("deltakv_gather_raw_total"):
            if slots.numel() == 0:
                return torch.empty(
                    (0, 2 * self.num_kv_heads * self.head_dim),
                    device=self.device,
                    dtype=self.hf_config.dtype,
                )

            assert layer_idx in self.deltakv_layer_to_idx
            l_idx = self.deltakv_layer_to_idx[layer_idx]
            k_cache = self.deltakv_full_kv_cache[0, l_idx]
            v_cache = self.deltakv_full_kv_cache[1, l_idx]

            slots_i32 = slots.to(torch.int32)
            pos = self.deltakv_slot_to_pos[slots_i32].to(torch.int32)
            is_capturing = self._is_stream_capturing()
            if not is_capturing and (pos < 0).any():
                raise RuntimeError("DeltaKV: center slot has unknown position (deltakv_slot_to_pos == -1).")

            return self._deltakv_gather_raw_kv_from_cache(
                slots=slots_i32,
                pos=pos,
                k_cache=k_cache,
                v_cache=v_cache,
            )

    @torch.no_grad()
    def deltakv_evict(self, seqs: list[Sequence]):
        with profiler.record("deltakv_evict_triton_total"):
            if not self.deltakv_layer_ids:
                return
            sink = int(self.config.sink_keep_tokens)
            recent = int(self.config.recent_keep_tokens)
            cluster_step = self._deltakv_base_cluster_step()

            for seq in seqs:
                with profiler.record("deltakv_evict_triton_seq"):
                    row_idx = self.seq_id_to_row.get(seq.seq_id, None)
                    if row_idx is None:
                        continue

                    total_len = int(self.row_seq_lens[row_idx])
                    compressed_len = int(self.row_deltakv_compressed_lens[row_idx])
                    buffer_start = sink + compressed_len
                    buffer_len = total_len - buffer_start
                    if buffer_len <= recent:
                        continue

                    evict_len = ((buffer_len - recent) // recent) * recent
                    if evict_len <= 0:
                        continue

                    evict_start = buffer_start
                    evict_end = evict_start + evict_len

                    with profiler.record("deltakv_evict_triton_read_slots"):
                        raw_slots_block = self.sparse_layer_raw_slots_map[row_idx, evict_start:evict_end].clone()
                    if (raw_slots_block < 0).any():
                        raise RuntimeError("DeltaKV eviction expects raw slots for the buffer block.")

                    with profiler.record("deltakv_evict_triton_select_centers"):
                        center_rel = self._deltakv_center_rel_for_block(
                            row_idx,
                            start=evict_start,
                            end=evict_end,
                            update_state=True,
                        )
                        new_center_slots = raw_slots_block[center_rel].to(torch.int32)

                    with profiler.record("deltakv_evict_triton_prev_centers"):
                        sink_slots = self.sparse_layer_raw_slots_map[row_idx, :sink].to(torch.int32)
                        prev_center_slots_by_layer: dict[int, torch.Tensor] = {}
                        for layer_idx in self.deltakv_layer_ids:
                            existing = self.row_deltakv_center_slots[row_idx][layer_idx]
                            prev_center_slots_by_layer[layer_idx] = (
                                sink_slots if existing is None else existing.to(torch.int32)
                            )

                    with profiler.record("deltakv_evict_triton_build_masks"):
                        is_center = torch.zeros((evict_len,), device=self.device, dtype=torch.bool)
                        is_center[center_rel] = True
                        to_free_mask = ~is_center

                    with profiler.record("deltakv_evict_triton_alloc_latent"):
                        latent_slots = self._allocate_deltakv_latent(evict_len).to(torch.int32)
                        pos_all = torch.arange(evict_start, evict_end, device=self.device, dtype=torch.int32)
                        pos_to_free = pos_all[to_free_mask]
                        self.sparse_layer_latent_slots_map[row_idx, pos_all.to(torch.long)] = latent_slots

                    raw_slots_block_i32 = raw_slots_block.to(torch.int32)

                    for layer_idx in self.deltakv_layer_ids:
                        l_idx = self.deltakv_layer_to_idx[layer_idx]
                        k_cache = self.deltakv_full_kv_cache[0, l_idx]
                        v_cache = self.deltakv_full_kv_cache[1, l_idx]

                        with profiler.record("deltakv_evict_triton_gather_raw"):
                            kv_block = self._deltakv_gather_raw_kv_from_cache(
                                slots=raw_slots_block_i32,
                                pos=pos_all,
                                k_cache=k_cache,
                                v_cache=v_cache,
                            ).unsqueeze(0)  # (1, N, kv_dim)

                        existing_center_slots = prev_center_slots_by_layer[layer_idx]
                        with profiler.record("deltakv_evict_triton_cluster"):
                            topk_center_indices, base_kv = self._cluster_compress(
                                layer_idx=layer_idx,
                                kv_states=kv_block,
                                existing_center_slots=existing_center_slots,
                                cluster_step=cluster_step,
                                new_center_rel=center_rel,
                            )

                        with profiler.record("deltakv_evict_triton_remap_fathers"):
                            all_center_slots = torch.cat([existing_center_slots, new_center_slots], dim=0)  # (M,)
                            father_slots_full = all_center_slots[topk_center_indices.to(torch.long)]  # (N, K)
                            father_slots = father_slots_full

                        with profiler.record("deltakv_evict_triton_store_fathers"):
                            K = self.deltakv_latent_to_full_slots.shape[-1]
                            k_eff = father_slots.shape[1]
                            if k_eff < K:
                                pad = father_slots[:, :1].expand(-1, K - k_eff)
                                father_slots = torch.cat([father_slots, pad], dim=1)
                            elif k_eff > K:
                                father_slots = father_slots[:, :K]
                            self.deltakv_latent_to_full_slots[l_idx, latent_slots] = father_slots.to(torch.int32)

                        down = self.compress_down[l_idx]
                        with profiler.record("deltakv_evict_triton_compress_down"):
                            kv_down = down(kv_block).squeeze(0)  # (N, latent_dim)
                            base_down = down(base_kv).squeeze(0)
                            latent_all = kv_down - base_down
                        with profiler.record("deltakv_evict_triton_store_latent"):
                            self._store_deltakv_latent(l_idx, latent_slots, latent_all)

                    with profiler.record("deltakv_evict_triton_append_centers"):
                        for layer_idx in self.deltakv_layer_ids:
                            self.row_deltakv_center_slots[row_idx][layer_idx] = torch.cat(
                                [prev_center_slots_by_layer[layer_idx], new_center_slots], dim=0
                            )

                    with profiler.record("deltakv_evict_triton_free_full_slots"):
                        free_slots = self._filter_deltakv_center_slots_for_evict_free(
                            row_idx,
                            raw_slots_block_i32[to_free_mask],
                            extra_center_slots=new_center_slots,
                        )
                        ptr = self._num_free_slots_deltakv_full
                        self.free_slots_stack_deltakv_full[ptr: ptr + free_slots.numel()] = free_slots
                        self._num_free_slots_deltakv_full += free_slots.numel()
                        self.deltakv_slot_to_pos[free_slots] = -1

                    with profiler.record("deltakv_evict_triton_update_maps"):
                        self.sparse_layer_raw_slots_map[row_idx, pos_to_free.to(torch.long)] = -1
                        self.row_deltakv_compressed_lens[row_idx] += evict_len
                        self.row_deltakv_compressed_lens_gpu[row_idx] += int(evict_len)

    def _cluster_compress(
        self,
        layer_idx: int,
        kv_states: torch.Tensor,  # (1, N, kv_dim), de-RoPE already applied on K
        existing_center_slots: torch.Tensor,  # (M0,)
        cluster_step: int,
        new_center_rel: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """V4: reuse V3's blockwise L2-topk, but fuse gather+mean in Triton."""
        if os.getenv("SPARSEENGINE_DELTAKV_DISABLE_TRITON_CLUSTER") == "1":
            return super()._cluster_compress(
                layer_idx,
                kv_states,
                existing_center_slots,
                cluster_step,
                new_center_rel=new_center_rel,
            )
        metric_type = self.config.cluster_metric
        if metric_type != "l2":
            return super()._cluster_compress(
                layer_idx,
                kv_states,
                existing_center_slots,
                cluster_step,
                new_center_rel=new_center_rel,
            )

        assert kv_states.dim() == 3 and kv_states.shape[0] == 1
        _, n, kv_dim = kv_states.shape
        k_neighbors = int(self.config.deltakv_neighbor_count)
        cluster_step = max(1, int(cluster_step))
        is_capturing = self._is_stream_capturing()

        if new_center_rel is not None:
            new_center_rel = new_center_rel.to(device=kv_states.device, dtype=torch.long)
            if (
                not is_capturing
                and new_center_rel.numel() > 0
                and ((new_center_rel < 0).any() or (new_center_rel >= n).any())
            ):
                return super()._cluster_compress(
                    layer_idx,
                    kv_states,
                    existing_center_slots,
                    cluster_step,
                    new_center_rel=new_center_rel,
                )
            new_centers = kv_states.index_select(1, new_center_rel) if new_center_rel.numel() else kv_states[:, :0, :]
        else:
            new_centers = kv_states[:, ::cluster_step, :]

        with profiler.record("deltakv_cluster_existing_centers"):
            existing_centers = (
                self._gather_raw_kv_by_slots(layer_idx, existing_center_slots).unsqueeze(0)
                if existing_center_slots.numel() > 0
                else kv_states.new_zeros((1, 0, kv_dim))
            )
        all_centers = torch.cat([existing_centers, new_centers], dim=1)  # (1, M, kv_dim)
        m0 = int(existing_centers.shape[1])
        M = int(all_centers.shape[1])

        k_eff = min(k_neighbors, M)
        if k_eff <= 0:
            raise RuntimeError("DeltaKV: no available centers to assign.")
        # Small M: torch is fine.
        if M < 128 or kv_states.dtype not in (torch.float16, torch.bfloat16):
            return super()._cluster_compress(
                layer_idx,
                kv_states,
                existing_center_slots,
                cluster_step,
                new_center_rel=new_center_rel,
            )

        from sparseengine.kernels.triton.deltakv_kernels import batch_gather_mean, deltakv_l2_topk_blockwise

        with profiler.record("deltakv_cluster_metric"):
            partial_scores, partial_idx = deltakv_l2_topk_blockwise(
                tokens=kv_states[0],
                centers=all_centers[0],
                m0=m0,
                cluster_step=int(cluster_step),
                k=k_eff,
                new_center_rel=new_center_rel,
            )

        # Merge candidates across blocks: (N, MB*K) -> topk(K).
        with profiler.record("deltakv_cluster_topk"):
            NB, MB, BN, KK = partial_scores.shape
            cand_scores = partial_scores.permute(0, 2, 1, 3).reshape(NB * BN, MB * KK)[:n]
            cand_idx = partial_idx.permute(0, 2, 1, 3).reshape(NB * BN, MB * KK)[:n]
            merge_pos = cand_scores.topk(k=k_eff, dim=1).indices
            topk_indices_i32 = cand_idx.gather(1, merge_pos)  # (N, K) int32

        with profiler.record("deltakv_cluster_validate_topk"):
            if not is_capturing:
                invalid_topk = (topk_indices_i32 < 0) | (topk_indices_i32 >= M)
                if invalid_topk.any():
                    return super()._cluster_compress(
                        layer_idx,
                        kv_states,
                        existing_center_slots,
                        cluster_step,
                        new_center_rel=new_center_rel,
                    )

        with profiler.record("deltakv_cluster_gather_mean"):
            gather_chunk_size = int(getattr(self.config, "deltakv_cluster_gather_chunk_size", 16384))
            if gather_chunk_size <= 0:
                raise ValueError(
                    "deltakv_cluster_gather_chunk_size must be > 0, "
                    f"got {gather_chunk_size}."
                )
            if n <= gather_chunk_size:
                gathered = batch_gather_mean(all_centers[0], topk_indices_i32.unsqueeze(0))  # (1, N, kv_dim)
            else:
                gathered = kv_states.new_empty((1, n, kv_dim))
                for start in range(0, n, gather_chunk_size):
                    end = min(start + gather_chunk_size, n)
                    gathered[:, start:end, :].copy_(
                        batch_gather_mean(
                            all_centers[0],
                            topk_indices_i32[start:end].unsqueeze(0).contiguous(),
                        )
                    )

        return topk_indices_i32.to(torch.int32), gathered
