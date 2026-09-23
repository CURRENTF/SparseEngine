from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from sparseengine.config import Config
from sparseengine.distributed import ParallelContext
from sparseengine.engine.decode_graph_contract import CacheDecodeGraphState
from sparseengine.engine.sequence import Sequence
from sparseengine.engine.prefill import (
    PREFILL_EXECUTION_CHUNKED,
    PREFILL_EXECUTION_FULL,
    PREFILL_EXECUTION_RAW_OFFLOAD,
)
from sparseengine.method_registry import PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH
from sparseengine.platforms import device_runtime
from sparseengine.kernels.triton.prefill_score import (
    PrefillScoreWorkspace,
    prefill_score_fwd,
)
from sparseengine.utils.context import get_context
from sparseengine.utils.log import logger, log_level
from sparseengine.utils.profiler import profiler

from ..base import (
    AttentionCacheWrite,
    CacheManager,
    ExplicitKVPayload,
    MlaLatentPayload,
    LayerBatchStates,
    PrefillComputeView,
    PrefillScoreRequest,
    SparseSelection,
)
from ..raw_kv_offload import RawKVOffloadBuffer
from ..chain_offload import ChainMethodState, ChainOffloadController
from ..storage import ExplicitKVStorage, create_attention_cache_storage


_INT32_BYTES = 4
_PAGE_TABLE_COMPACTION_TILE_ELEMENTS = 256 * 1024


@dataclass
class SnapKVDecodeGraphState(CacheDecodeGraphState):
    layer_slot_mapping: torch.Tensor
    layer_free_starts: torch.Tensor
    active_count: torch.Tensor
    layer_fact_storage: torch.Tensor
    host_layer_free_starts: torch.Tensor
    host_active_count: torch.Tensor
    host_layer_fact_storage: torch.Tensor
    transformer_layer_ids: torch.Tensor


def resolve_snapkv_cache_capacity(
    *,
    available_bytes: int,
    slot_bytes_per_layer: int,
    num_kv_layers: int,
    max_buffer_rows: int,
    max_model_len: int,
    layer_ratios: list[float] | None = None,
) -> tuple[int, tuple[int, ...], int]:
    """Size KV storage together with its persistent slot metadata."""
    available_bytes = int(available_bytes)
    slot_bytes_per_layer = int(slot_bytes_per_layer)
    num_kv_layers = int(num_kv_layers)
    row_slot_map_bytes = (
        num_kv_layers
        * int(max_buffer_rows)
        * int(max_model_len)
        * _INT32_BYTES
    )
    slot_pool_bytes = available_bytes - row_slot_map_bytes
    if slot_pool_bytes <= 0:
        raise RuntimeError(
            "Not enough GPU memory for SnapKV row-slot metadata. "
            f"row_slot_map_bytes={row_slot_map_bytes} "
            f"available_bytes={available_bytes}."
        )

    persistent_bytes_per_slot = slot_bytes_per_layer + _INT32_BYTES
    if layer_ratios is None:
        slots_per_layer = slot_pool_bytes // (
            num_kv_layers * persistent_bytes_per_slot
        )
        layer_slots = (int(slots_per_layer),) * num_kv_layers
        base_slots = int(slots_per_layer)
    else:
        ratios = tuple(float(ratio) for ratio in layer_ratios)
        if len(ratios) != num_kv_layers or any(ratio <= 0 for ratio in ratios):
            raise ValueError(
                "PyramidKV layer ratios must contain one positive value per KV layer: "
                f"ratios={ratios}, num_kv_layers={num_kv_layers}."
            )
        base_slots = int(
            slot_pool_bytes
            // (persistent_bytes_per_slot * sum(ratios))
        )
        layer_slots = tuple(int(base_slots * ratio) for ratio in ratios)

    if base_slots <= 0 or any(num_slots <= 0 for num_slots in layer_slots):
        raise RuntimeError(
            "Not enough GPU memory for SnapKV KV slots and free-slot metadata. "
            f"slot_pool_bytes={slot_pool_bytes} "
            f"persistent_bytes_per_slot={persistent_bytes_per_slot} "
            f"num_kv_layers={num_kv_layers}."
        )
    return base_slots, layer_slots, row_slot_map_bytes


class SnapKVCacheManager(CacheManager):
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
        self.attention_cache_storage = (
            create_attention_cache_storage(
                config,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            )
            if config.pyramid_layer_ratios is None
            else None
        )
        self.pyramidkv_prefill_staging_num_slots = 0
        self.pyramidkv_prefill_staging_kv_cache = None
        self._pyramidkv_prefill_staging_active = False
        self._pyramidkv_prefill_staging_was_active = False
        self._pyramidkv_prefill_staging_slot_mapping = None
        self._pyramidkv_prefill_staging_slot_mapping_by_layer = {}
        self._pyramidkv_prefill_staging_active_slots = None
        self._pyramidkv_prefill_staging_active_slots_by_layer = {}
        self._pyramidkv_prefill_staging_req_indices = None
        self._pyramidkv_prefill_staging_context_lens = None
        self._pyramidkv_prefill_staging_context_lens_by_layer = {}
        self._pyramidkv_prefill_staging_context_lens_cpu_by_layer = {}
        self._pyramidkv_prefill_staging_seq_offsets: dict[int, int] = {}
        self._pyramidkv_prefill_staging_materialized_layers: set[tuple[int, int]] = set()
        self.raw_kv_offload_buffer = RawKVOffloadBuffer(pin_memory=device_runtime.supports_pin_memory())
        self._pyramidkv_long_prefill_offload_step_active = False
        self._pyramidkv_long_prefill_offload_seq_id: int | None = None
        self._pyramidkv_long_prefill_offload_start = 0
        self._pyramidkv_long_prefill_offload_end = 0
        self._pyramidkv_long_prefill_offload_total_len = 0
        self._pyramidkv_long_prefill_offload_residual_start = 0
        self._pyramidkv_long_prefill_offload_resident_prefix_lens: dict[int, int] = {}
        self._pyramidkv_long_prefill_offload_is_last_chunk = False
        self._pyramidkv_long_prefill_offload_prefetch_stream = None
        self._pyramidkv_long_prefill_offload_prefetch_states: dict[tuple[int, int, str, int], dict] = {}
        self.allocate_kv_cache()

        self.layer_num_slots = []
        self.free_slots_stack_tensor = None
        self.free_slots_stack = []
        self._num_free_slots = []
        self.buffer_req_to_token_slots = []
        self.seq_id_to_row = []
        self.free_rows = []
        self.row_seq_lens = []
        self.layer_batch_states = [LayerBatchStates() for _ in range(self.num_layers)]
        self._decode_static_buffers: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._decode_static_index_buffers: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._decode_static_state_binding_key: tuple[int, int, int, int] | None = None
        self._prefill_attn_score_accumulators: dict[tuple[int, int], torch.Tensor] = {}
        self._prefill_context_lens_cpu_by_layer: dict[int, tuple[int, ...]] = {}
        self._prefill_score_metadata_cache: dict[
            tuple,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._prefill_step_score_buffers: dict[
            tuple[torch.device, torch.dtype], torch.Tensor
        ] = {}
        self._prefill_score_workspace = PrefillScoreWorkspace()
        self._uniform_decode_metadata = self._sparse_eviction_never_triggers()
        self.buffer_req_to_token_slots_tensor = torch.zeros(
            (self.num_kv_layers, self.max_buffer_rows, self.max_model_len),
            dtype=torch.int32,
            device=self.device,
        )
        if not isinstance(config.num_kvcache_slots, list):
            num_slots = int(config.num_kvcache_slots)
            self.free_slots_stack_tensor = torch.arange(
                num_slots,
                dtype=torch.int32,
                device=self.device,
            ).expand(self.num_kv_layers, -1).clone()

        for layer_id in range(self.num_layers):
            if not self.is_full_attention_layer(layer_id):
                self.layer_num_slots.append(0)
                self.free_slots_stack.append(None)
                self._num_free_slots.append(0)
                self.buffer_req_to_token_slots.append(None)
                self.seq_id_to_row.append({})
                self.free_rows.append(deque(range(self.max_buffer_rows)))
                self.row_seq_lens.append(np.zeros((self.max_buffer_rows,), dtype=np.int32))
                continue
            kv_idx = self.kv_layer_index(layer_id)
            num_slots = (
                config.num_kvcache_slots[layer_id]
                if isinstance(config.num_kvcache_slots, list)
                else config.num_kvcache_slots
            )
            self.layer_num_slots.append(num_slots)
            if self.free_slots_stack_tensor is not None:
                self.free_slots_stack.append(self.free_slots_stack_tensor[kv_idx])
            else:
                self.free_slots_stack.append(
                    torch.arange(num_slots, dtype=torch.int32, device=self.device)
                )
            self._num_free_slots.append(num_slots)
            self.buffer_req_to_token_slots.append(self.buffer_req_to_token_slots_tensor[kv_idx])
            self.seq_id_to_row.append({})
            self.free_rows.append(deque(range(self.max_buffer_rows)))
            self.row_seq_lens.append(np.zeros((self.max_buffer_rows,), dtype=np.int32))

    def _sparse_eviction_never_triggers(self) -> bool:
        method = str(getattr(self.config, "sparse_method", "") or "")
        max_model_len = int(getattr(self.config, "max_model_len", 0) or 0)
        sink = int(getattr(self.config, "sink_keep_tokens", 0) or 0)
        recent = int(getattr(self.config, "recent_keep_tokens", 0) or 0)
        decode_keep = int(getattr(self.config, "decode_keep_tokens", 0) or 0)
        if method in {"snapkv", "rkv", "skipkv"}:
            return max_model_len <= sink + decode_keep + recent
        if method == "pyramidkv" and self.config.pyramid_layer_ratios is None:
            return max_model_len <= sink + decode_keep + recent
        return False

    def _decode_graph_metadata_always_uniform(self) -> bool:
        """Whether every reachable decode state keeps rows aligned by layer."""
        return self._sparse_eviction_never_triggers()

    def _pyramidkv_can_use_full_prefill_staging(self) -> bool:
        return (
            self.config.sparse_method == "pyramidkv"
            and self.config.pyramid_layer_ratios is not None
            and self.config.prefill_schedule_policy == PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH
        )

    def _pyramidkv_reset_full_prefill_staging(self):
        self._pyramidkv_clear_long_prefill_offload_prefetch()
        self._pyramidkv_prefill_staging_active = False
        self._pyramidkv_prefill_staging_was_active = False
        self._pyramidkv_prefill_staging_slot_mapping = None
        self._pyramidkv_prefill_staging_slot_mapping_by_layer = {}
        self._pyramidkv_prefill_staging_active_slots = None
        self._pyramidkv_prefill_staging_active_slots_by_layer = {}
        self._pyramidkv_prefill_staging_req_indices = None
        self._pyramidkv_prefill_staging_context_lens = None
        self._pyramidkv_prefill_staging_context_lens_by_layer = {}
        self._pyramidkv_prefill_staging_context_lens_cpu_by_layer = {}
        self._pyramidkv_prefill_staging_seq_offsets = {}
        self._pyramidkv_prefill_staging_materialized_layers = set()
        self._pyramidkv_long_prefill_offload_seq_id = None
        self._pyramidkv_long_prefill_offload_start = 0
        self._pyramidkv_long_prefill_offload_end = 0
        self._pyramidkv_long_prefill_offload_total_len = 0
        self._pyramidkv_long_prefill_offload_residual_start = 0
        self._pyramidkv_long_prefill_offload_resident_prefix_lens = {}
        self._pyramidkv_long_prefill_offload_is_last_chunk = False

    def _long_prefill_offload_threshold(self) -> int:
        return int(self.config.long_prefill_offload_threshold)

    def prefill_execution_mode(self, seq: Sequence) -> str:
        if not self._pyramidkv_can_use_full_prefill_staging():
            return PREFILL_EXECUTION_CHUNKED
        residual = int(self.remaining_prefill_tokens(seq))
        if residual <= 0:
            raise ValueError(
                "PyramidKV prefill execution mode requires a positive residual: "
                f"seq_id={seq.seq_id} residual={residual}."
            )
        initial_mode = (
            PREFILL_EXECUTION_FULL
            if residual <= self._long_prefill_offload_threshold()
            else PREFILL_EXECUTION_RAW_OFFLOAD
        )
        mode = self._apply_sticky_raw_offload_mode(seq, initial_mode)
        if (
            mode == PREFILL_EXECUTION_RAW_OFFLOAD
            and self.pyramidkv_prefill_staging_kv_cache is None
        ):
            raise RuntimeError(
                "PyramidKV raw_offload mode requires prefill staging KV. "
                f"seq_id={seq.seq_id} residual={residual}."
            )
        return mode

    def prefill_batch_compatibility_key(self, seq: Sequence) -> object:
        if (
            self._pyramidkv_can_use_full_prefill_staging()
            and self.prefill_execution_mode(seq) == PREFILL_EXECUTION_FULL
        ):
            resumed_resident = (
                getattr(seq, "chain_status", "") == "resumed"
                and not bool(getattr(seq, "is_recompute_replay", False))
                and int(seq.num_prefilled_tokens) > 0
            )
            return (
                "pyramidkv_resumed_resident_full"
                if resumed_resident
                else "pyramidkv_fresh_staged_full"
            )
        return None

    def requires_long_prefill_offload(self, seq: Sequence) -> bool:
        return self.prefill_execution_mode(seq) == PREFILL_EXECUTION_RAW_OFFLOAD

    def _should_use_pyramidkv_long_prefill_offload_staging(self, seqs: list[Sequence]) -> bool:
        if not self._pyramidkv_can_use_full_prefill_staging():
            return False
        if self.pyramidkv_prefill_staging_kv_cache is None or len(seqs) != 1:
            return False
        seq = seqs[0]
        return self.requires_long_prefill_offload(seq) and int(seq.current_chunk_size or 0) > 0

    def _should_use_pyramidkv_full_prefill_staging(self, seqs: list[Sequence]) -> bool:
        if not self._pyramidkv_can_use_full_prefill_staging():
            return False
        if self.pyramidkv_prefill_staging_kv_cache is None or not seqs:
            return False
        total_chunk_tokens = 0
        for seq in seqs:
            if self.requires_long_prefill_offload(seq):
                return False
            remaining = int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
            if int(seq.num_prefilled_tokens) != 0 or int(seq.current_chunk_size) != remaining:
                return False
            total_chunk_tokens += int(seq.current_chunk_size)
        return total_chunk_tokens <= int(self.pyramidkv_prefill_staging_num_slots)

    def requires_full_prefill_step(self, seq: Sequence) -> bool:
        return self.prefill_execution_mode(seq) == PREFILL_EXECUTION_FULL

    def allocate_kv_cache(self):
        available_memory, slot_bytes_per_layer = self._get_available_slots_info()
        config = self.config
        num_layers = self.num_kv_layers

        if config.pyramid_layer_ratios is not None:
            staging_bytes = 0
            if self._pyramidkv_can_use_full_prefill_staging():
                self.pyramidkv_prefill_staging_num_slots = max(
                    int(config.max_model_len),
                    int(config.max_num_batched_tokens),
                )
                staging_bytes = int(self.pyramidkv_prefill_staging_num_slots) * int(slot_bytes_per_layer)
                available_memory = int(available_memory) - staging_bytes
                if available_memory <= 0:
                    raise RuntimeError(
                        "Not enough GPU memory for PyramidKV full-prefill staging KV. "
                        f"staging_slots={self.pyramidkv_prefill_staging_num_slots} "
                        f"required={staging_bytes / 1024**3:.2f}GiB."
                    )
            # PyramidKV: 根据比例分配每层不同大小的 cache
            kv_layer_ids = list(self.runtime_layout.kv_idx_to_layer_idx)
            base_slots, resolved_layer_slots, row_slot_map_bytes = (
                resolve_snapkv_cache_capacity(
                    available_bytes=available_memory,
                    slot_bytes_per_layer=slot_bytes_per_layer,
                    num_kv_layers=num_layers,
                    max_buffer_rows=self.max_buffer_rows,
                    max_model_len=self.max_model_len,
                    layer_ratios=config.pyramid_layer_ratios,
                )
            )
            kv_layer_slots = list(resolved_layer_slots)
            assert kv_layer_slots[0] == max(kv_layer_slots), (
                "The first KV layer must have the largest PyramidKV allocation, but "
                f"first={kv_layer_slots[0]}, max={max(kv_layer_slots)}."
            )
            layer_slots = [0] * self.num_layers

            if staging_bytes:
                self.pyramidkv_prefill_staging_kv_cache = torch.empty(
                    2,
                    self.pyramidkv_prefill_staging_num_slots,
                    self.num_kv_heads,
                    self.head_dim,
                    dtype=self.hf_config.dtype,
                    device=self.device,
                )

            self.kv_cache = []
            for kv_idx, layer_idx in enumerate(kv_layer_ids):
                num_slots = kv_layer_slots[kv_idx]
                layer_slots[layer_idx] = num_slots
                k_cache = torch.empty(
                    num_slots, self.num_kv_heads, self.head_dim,
                    dtype=self.hf_config.dtype, device=self.device
                )
                v_cache = torch.empty(
                    num_slots, self.num_kv_heads, self.head_dim,
                    dtype=self.hf_config.dtype, device=self.device
                )
                self.kv_cache.append((k_cache, v_cache))

            config.num_kvcache_slots = layer_slots
            logger.info(
                f"PyramidKV: KV layer slots = {list(zip(kv_layer_ids, kv_layer_slots))}, "
                f"base_slots = {base_slots}, "
                f"prefill_staging_slots={self.pyramidkv_prefill_staging_num_slots}, "
                f"row_slot_map_bytes={row_slot_map_bytes}"
            )
        else:
            # 标准模式：所有层使用相同大小
            num_slots, _, row_slot_map_bytes = resolve_snapkv_cache_capacity(
                available_bytes=available_memory,
                slot_bytes_per_layer=slot_bytes_per_layer,
                num_kv_layers=num_layers,
                max_buffer_rows=self.max_buffer_rows,
                max_model_len=self.max_model_len,
            )
            config.num_kvcache_slots = num_slots

            logger.info(
                "Standard Mode (SnapKV): Each layer can accommodate "
                f"{config.num_kvcache_slots} tokens, "
                f"row_slot_map_bytes={row_slot_map_bytes}."
            )
            storage = self.attention_cache_storage
            if storage is None:
                raise RuntimeError(
                    "Uniform SnapKV requires an attention cache storage."
                )
            storage.allocate(
                num_layers=num_layers,
                num_slots=config.num_kvcache_slots,
                device=self.device,
            )
            self.kv_cache = (
                storage.cache
                if isinstance(storage, ExplicitKVStorage)
                else None
            )

    def attention_cache_bytes_per_slot_per_layer(self) -> int:
        storage = getattr(self, "attention_cache_storage", None)
        if storage is None:
            return super().attention_cache_bytes_per_slot_per_layer()
        return int(storage.bytes_per_slot_per_layer())

    def get_layer_batch_states(self, layer_idx: int) -> LayerBatchStates:
        self.kv_layer_index(layer_idx)
        return self.layer_batch_states[layer_idx]

    def get_layer_kv_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        kv_idx = self.kv_layer_index(layer_idx)
        if isinstance(self.kv_cache, list):
            return self.kv_cache[kv_idx]
        elif isinstance(self.kv_cache, torch.Tensor):
            return self.kv_cache[0, kv_idx], self.kv_cache[1, kv_idx]
        else:
            raise ValueError

    def store_attention_payload(
        self,
        layer_idx: int,
        payload: AttentionCacheWrite,
    ) -> torch.Tensor:
        storage = getattr(self, "attention_cache_storage", None)
        if storage is None or isinstance(storage, ExplicitKVStorage):
            return super().store_attention_payload(layer_idx, payload)
        slot_mapping = self.layer_batch_states[layer_idx].slot_mapping
        if slot_mapping is None:
            raise RuntimeError(
                f"Attention cache store requires slot_mapping at layer={layer_idx}."
            )
        storage.store(
            self.kv_layer_index(layer_idx),
            slot_mapping,
            payload,
        )
        return slot_mapping

    def get_layer_compute_payload(
        self,
        layer_idx: int,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        selection: SparseSelection | None = None,
    ):
        storage = getattr(self, "attention_cache_storage", None)
        if storage is None or isinstance(storage, ExplicitKVStorage):
            return super().get_layer_compute_payload(
                layer_idx,
                active_slots,
                req_indices,
                context_lens,
                selection,
            )
        return (
            storage.layer_payload(self.kv_layer_index(layer_idx)),
            active_slots,
            req_indices,
            context_lens,
        )

    def get_prefill_compute_payload(
        self,
        layer_idx: int,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        selection: SparseSelection,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
    ):
        storage = getattr(self, "attention_cache_storage", None)
        if storage is None or isinstance(storage, ExplicitKVStorage):
            return super().get_prefill_compute_payload(
                layer_idx,
                k_current,
                v_current,
                selection,
                active_slots,
                req_indices,
                context_lens,
            )
        return self.get_layer_compute_payload(
            layer_idx,
            active_slots,
            req_indices,
            context_lens,
            selection,
        )

    def get_layer_store_view(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.has_prefill_staging_view(layer_idx):
            return (
                self.pyramidkv_prefill_staging_kv_cache[0],
                self.pyramidkv_prefill_staging_kv_cache[1],
                self._pyramidkv_prefill_staging_slot_mapping_by_layer.get(
                    int(layer_idx),
                    self._pyramidkv_prefill_staging_slot_mapping,
                ),
            )
        k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        return k_cache, v_cache, self.layer_batch_states[layer_idx].slot_mapping

    def get_layer_compute_tensors(self, layer_idx: int, selection: SparseSelection | None = None):
        del selection
        if self.has_prefill_staging_view(layer_idx):
            return self.pyramidkv_prefill_staging_kv_cache[0], self.pyramidkv_prefill_staging_kv_cache[1]
        raise NotImplementedError

    def has_prefill_staging_view(self, layer_idx: int) -> bool:
        return bool(
            self._pyramidkv_prefill_staging_active
            and self.config.sparse_method == "pyramidkv"
            and 0 <= int(layer_idx) < int(self.num_layers)
            and self.is_full_attention_layer(layer_idx)
        )

    def get_prefill_staging_view(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not self.has_prefill_staging_view(layer_idx):
            raise NotImplementedError("PyramidKV prefill staging view is not active for this layer.")
        return (
            self._pyramidkv_prefill_staging_active_slots_by_layer.get(
                int(layer_idx),
                self._pyramidkv_prefill_staging_active_slots,
            ),
            self._pyramidkv_prefill_staging_req_indices,
            self._pyramidkv_prefill_staging_context_lens_by_layer.get(
                int(layer_idx),
                self._pyramidkv_prefill_staging_context_lens,
            ),
            None,
        )

    def prefill_staging_was_active(self) -> bool:
        return bool(self._pyramidkv_prefill_staging_was_active)

    def get_layer_buffer_req_to_token_slots(self, layer_idx: int) -> torch.Tensor:
        self.kv_layer_index(layer_idx)
        return self.buffer_req_to_token_slots[layer_idx]

    @property
    def num_free_slots(self) -> int:
        return min(self._num_free_slots[layer_idx] for layer_idx in self.kv_transformer_layer_indices())

    def decode_window_budgets(self) -> dict[str, int]:
        return {f"layer_{layer}": int(self._num_free_slots[layer])
                for layer in self.kv_transformer_layer_indices()}

    def decode_window_costs(self, seq: Sequence, tokens: int) -> dict[str, int]:
        required, _, _, _ = self.chain_capacity_deficits(
            suffix_tokens=0, generation_tokens=int(tokens) + 1,
            existing_slots_by_layer=self.chain_physical_residency(seq.seq_id),
            needs_resident_row=False,
        )
        return {f"layer_{layer}": int(cost) for layer, cost in
                zip(self.kv_transformer_layer_indices(), required)}

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
        suffix_tokens = max(0, int(suffix_tokens))
        generated_kv_tokens = max(0, int(generation_tokens) - 1)
        layer_ids = self.kv_transformer_layer_indices()
        method = str(self.config.sparse_method)
        use_new_pyramid_staging = (
            needs_resident_row
            and not any(existing_slots_by_layer)
            and method == "pyramidkv"
            and self._pyramidkv_can_use_full_prefill_staging()
        )
        required_by_layer = []
        for local_layer, layer_idx in enumerate(layer_ids):
            existing = (
                int(existing_slots_by_layer[local_layer])
                if local_layer < len(existing_slots_by_layer)
                else 0
            )
            kv_layer_idx = self.kv_layer_index(layer_idx)
            is_full_layer = (
                method in ("snapkv", "pyramidkv")
                and kv_layer_idx
                < int(getattr(self.config, "snapkv_num_full_layers", 0))
            )
            budget = None
            trigger_len = None
            if method == "pyramidkv" and not is_full_layer:
                budget = self._pyramidkv_layer_budget(layer_idx)
                top_budget = (
                    int(budget)
                    - int(self.config.sink_keep_tokens)
                    - int(self.config.recent_keep_tokens)
                )
                trigger_len = max(
                    int(budget) + 1,
                    int(budget) + max(0, int(top_budget)),
                )
            elif method == "snapkv" and not is_full_layer:
                budget = (
                    int(self.config.sink_keep_tokens)
                    + int(self.config.decode_keep_tokens)
                    + int(self.config.recent_keep_tokens)
                )
            elif method in ("rkv", "skipkv"):
                budget = (
                    int(self.config.sink_keep_tokens)
                    + int(self.config.decode_keep_tokens)
                    + int(self.config.recent_keep_tokens)
                )
                interval = int(
                    getattr(
                        self.config,
                        (
                            "rkv_compression_interval"
                            if method == "rkv"
                            else "skipkv_compression_interval"
                        ),
                        0,
                    )
                )
                trigger_len = max(int(budget) + 1, int(budget) + interval)

            prefill_physical_peak = existing + suffix_tokens
            if use_new_pyramid_staging and budget is not None:
                prefill_physical_peak = min(suffix_tokens, int(budget))

            if method == "snapkv" and budget is not None:
                # SnapKV currently compacts only at final prefill. Decode is
                # score-free and grows monotonically, so chain admission must
                # reserve the full requested decode growth rather than the
                # legacy periodic-eviction trigger.
                resident_after_prefill = (
                    min(existing + suffix_tokens, int(budget))
                    if suffix_tokens > 0
                    else existing
                )
                decode_physical_peak = (
                    resident_after_prefill + generated_kv_tokens
                )
            elif budget is None or trigger_len is None:
                decode_physical_peak = (
                    prefill_physical_peak + generated_kv_tokens
                )
            else:
                # Decode-only reservations start from the live row; only an
                # actual suffix prefill can compact PyramidKV to its budget.
                resident_after_prefill = (
                    existing + suffix_tokens
                    if suffix_tokens == 0 or method in ("rkv", "skipkv")
                    else min(existing + suffix_tokens, int(budget))
                )
                if use_new_pyramid_staging:
                    resident_after_prefill = min(
                        suffix_tokens, int(budget)
                    )
                if generated_kv_tokens <= 0:
                    decode_physical_peak = resident_after_prefill
                elif method == "rkv":
                    # Global R-KV is armed only at decode-buffer boundaries. A
                    # long prompt can grow by a full interval before eviction.
                    interval = int(self.config.rkv_compression_interval)
                    decode_physical_peak = min(
                        resident_after_prefill + generated_kv_tokens,
                        max(resident_after_prefill, int(trigger_len) - 1) + interval,
                    )
                elif resident_after_prefill >= int(trigger_len):
                    decode_physical_peak = resident_after_prefill + 1
                else:
                    decode_physical_peak = resident_after_prefill + min(
                        generated_kv_tokens,
                        int(trigger_len) - resident_after_prefill,
                    )
            required_by_layer.append(
                max(prefill_physical_peak, decode_physical_peak)
                - (0 if needs_resident_row else existing)
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

    def create_chain_offload(self, capacity_bytes: int) -> ChainOffloadController:
        return ChainOffloadController(self, capacity_bytes)

    def chain_token_slots(self, layer: int, seq_id: int) -> torch.Tensor:
        row = self.seq_id_to_row[layer][int(seq_id)]
        length = int(self.row_seq_lens[layer][row])
        return self.buffer_req_to_token_slots[layer][row, :length]

    def allocate_chain_restore(
        self, seq_id: int, lengths_by_layer: tuple[int, ...],
    ) -> dict[int, torch.Tensor]:
        """All-layer restore allocation; payload transfer belongs to offload.

        Preflight uses CPU mirrors only. A partial allocation failure is cleaned
        by the ordinary method-specific free_seq, including sidecars and rows.
        The six chain methods inherit this physical allocator, not an algorithm.
        """
        seq_id = int(seq_id)
        layers = tuple(self.kv_transformer_layer_indices())
        if len(lengths_by_layer) != len(layers):
            raise ValueError("Chain restore lengths must cover every KV layer.")
        if self.chain_has_residency(seq_id):
            raise RuntimeError("Cannot restore over a resident chain.")
        for layer, length in zip(layers, lengths_by_layer):
            if type(length) is not int or length < 0:
                raise ValueError("Chain restore lengths must be non-negative integers.")
            if length > self.buffer_req_to_token_slots[layer].shape[1]:
                raise ValueError("Chain restore exceeds the cache row capacity.")
            if not self.free_rows[layer] or self._num_free_slots[layer] < length:
                raise RuntimeError("Chain restore has insufficient reserved GPU rows/slots.")
        try:
            return {
                layer: self._allocate(layer, seq_id, length)
                for layer, length in zip(layers, lengths_by_layer)
            }
        except BaseException:
            self.free_seq(seq_id)
            raise

    def chain_storage_tensors(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        storage = getattr(self, "attention_cache_storage", None)
        if storage is not None:
            payload = storage.layer_payload(self.kv_layer_index(layer))
            if isinstance(payload, MlaLatentPayload):
                return payload.latent_cache, payload.rope_cache
            if isinstance(payload, ExplicitKVPayload):
                return payload.k_cache, payload.v_cache
            raise TypeError("Unsupported chain offload storage payload.")
        return self.get_layer_kv_cache(layer)

    def snapshot_chain_method_state(self, seq_id: int) -> ChainMethodState:
        return ChainMethodState()

    def restore_chain_method_state(self, seq_id: int, state: ChainMethodState) -> None:
        if state.tensors or state.metadata is not None:
            raise RuntimeError("Unexpected auxiliary state in a SnapKV chain snapshot.")

    def chain_physical_residency(self, seq_id: int) -> tuple[int, ...]:
        seq_id = int(seq_id)
        residency = []
        for layer_idx in self.kv_transformer_layer_indices():
            row_idx = self.seq_id_to_row[layer_idx].get(seq_id)
            if row_idx is None:
                raise RuntimeError(
                    f"Missing chain row for seq_id={seq_id} layer={layer_idx}."
                )
            residency.append(int(self.row_seq_lens[layer_idx][row_idx]))
        return tuple(residency)

    def chain_has_residency(self, seq_id: int) -> bool:
        seq_id = int(seq_id)
        return any(
            seq_id in self.seq_id_to_row[layer_idx]
            for layer_idx in self.kv_transformer_layer_indices()
        )

    def reset_after_warmup(self) -> None:
        # TP workers do not run the rank-0 Scheduler, so their RuntimeState
        # resident-sequence set cannot be the source of truth for warmup rows.
        # Reclaim directly from each rank-local cache-manager ledger before
        # serving the first real chain request.
        resident_seq_ids = sorted(
            {
                int(seq_id)
                for layer_idx in self.kv_transformer_layer_indices()
                for seq_id in self.seq_id_to_row[layer_idx]
            }
        )
        for seq_id in resident_seq_ids:
            self.free_seq(seq_id)

    def chain_physical_kv_len(self, layer_idx: int, seq_id: int) -> int:
        row_idx = self.seq_id_to_row[int(layer_idx)].get(int(seq_id))
        if row_idx is None:
            raise RuntimeError(
                f"Missing chain row for seq_id={seq_id} layer={layer_idx}."
            )
        return int(self.row_seq_lens[int(layer_idx)][row_idx])

    def _pyramidkv_layer_budget(self, layer_idx: int) -> int:
        decode_keep = int(self.config.decode_keep_tokens)
        ratio = float(self.config.pyramid_layer_ratios[self.kv_layer_index(layer_idx)])
        base_ratio = float(self.config.pyramid_layer_ratios[0])
        scaled_top_tokens = int(decode_keep * ratio / base_ratio)
        return int(self.config.sink_keep_tokens) + scaled_top_tokens + int(self.config.recent_keep_tokens)

    def _pyramidkv_prompt_admission_cost(self, seq: Sequence) -> int:
        prompt_len = int(seq.num_prompt_tokens)
        if prompt_len <= 0:
            return 0
        return max(
            min(prompt_len, self._pyramidkv_layer_budget(layer_idx))
            for layer_idx in self.kv_transformer_layer_indices()
        )

    def prompt_admission_cost(self, seq: Sequence) -> int:
        if self._pyramidkv_can_use_full_prefill_staging():
            return self._pyramidkv_prompt_admission_cost(seq)
        return super().prompt_admission_cost(seq)

    def prompt_logical_reservation_cost(self, seq: Sequence) -> int:
        if self._pyramidkv_can_use_full_prefill_staging():
            return self._pyramidkv_prompt_admission_cost(seq)
        return super().prompt_logical_reservation_cost(seq)

    def prompt_admission_free_slots(self) -> int:
        if self._pyramidkv_can_use_full_prefill_staging():
            return max(
                int(self._num_free_slots[layer_idx])
                for layer_idx in self.kv_transformer_layer_indices()
            )
        return super().prompt_admission_free_slots()

    def prompt_admission_budgets(self, waiting_seqs, engine_prefill_chunk_size: int) -> dict[str, int]:
        if not self._pyramidkv_can_use_full_prefill_staging():
            return super().prompt_admission_budgets(waiting_seqs, engine_prefill_chunk_size)
        return {
            f"layer_{layer_idx}": int(self._num_free_slots[layer_idx])
            for layer_idx in self.kv_transformer_layer_indices()
        }

    def prompt_admission_costs(self, seq: Sequence) -> dict[str, int]:
        if not self._pyramidkv_can_use_full_prefill_staging():
            return super().prompt_admission_costs(seq)
        prompt_len = int(seq.num_prompt_tokens)
        return {
            f"layer_{layer_idx}": min(prompt_len, self._pyramidkv_layer_budget(layer_idx))
            for layer_idx in self.kv_transformer_layer_indices()
        }

    def prefill_step_free_slots(self) -> int:
        if self._pyramidkv_can_use_full_prefill_staging():
            return int(self.pyramidkv_prefill_staging_num_slots)
        return super().prefill_step_free_slots()

    def prefill_step_free_slots_for(self, seq: Sequence) -> int:
        if self.requires_long_prefill_offload(seq):
            return int(self.pyramidkv_prefill_staging_num_slots)
        return super().prefill_step_free_slots_for(seq)

    def prefill_capacity_after_decode_reservations(
        self, free_slots: int, reserved: dict[str, int], *, admission: bool,
    ) -> int:
        if self._pyramidkv_can_use_full_prefill_staging() and not admission:
            # The staging tensor is a separate temporary pool. Decode windows
            # reserve the persistent per-layer pools and cannot consume it.
            return int(free_slots)
        return super().prefill_capacity_after_decode_reservations(
            free_slots, reserved, admission=admission,
        )

    def min_final_prefill_chunk_size(self, seq: Sequence) -> int:
        method = self.config.sparse_method
        if method not in {"snapkv", "pyramidkv"}:
            return 0
        window = int(getattr(self.config, "snapkv_window_size", 0) or 0)
        if window <= 0:
            return 0
        if (
            int(getattr(self.config, "snapkv_num_full_layers", 0) or 0)
            >= int(self.num_kv_layers)
        ):
            return 0
        is_chain_resume = (
            getattr(seq, "chain_status", "") == "resumed"
            and not bool(getattr(seq, "is_recompute_replay", False))
        )
        residual = int(seq.num_prompt_tokens) - int(seq.num_prefilled_tokens)
        raw_offload_turn_tokens = None
        if (
            method == "pyramidkv"
            and is_chain_resume
            and self.requires_long_prefill_offload(seq)
        ):
            reused_tokens = max(
                int(getattr(seq, "chain_reused_tokens", 0) or 0),
                int(getattr(seq, "prefix_cache_hit_len", 0) or 0),
            )
            raw_offload_turn_tokens = max(
                0,
                int(seq.num_prompt_tokens) - reused_tokens,
            )
        if method == "snapkv":
            budget = (
                int(self.config.sink_keep_tokens)
                + int(self.config.decode_keep_tokens)
                + int(self.config.recent_keep_tokens)
            )
            if is_chain_resume:
                first_layer = int(self.kv_transformer_layer_indices()[0])
                final_physical_len = (
                    self.chain_physical_kv_len(first_layer, int(seq.seq_id))
                    + residual
                )
            else:
                final_physical_len = int(seq.num_prompt_tokens)
            needs_score = final_physical_len > budget
        else:
            needs_score = False
            for layer_idx in self.kv_transformer_layer_indices():
                if self.kv_layer_index(layer_idx) < int(
                    getattr(self.config, "snapkv_num_full_layers", 0) or 0
                ):
                    continue
                final_physical_len = (
                    self.chain_physical_kv_len(layer_idx, int(seq.seq_id))
                    + (
                        raw_offload_turn_tokens
                        if raw_offload_turn_tokens is not None
                        else residual
                    )
                    if is_chain_resume
                    else int(seq.num_prompt_tokens)
                )
                if final_physical_len > self._pyramidkv_layer_budget(layer_idx):
                    needs_score = True
                    break
        if not needs_score:
            return 0
        return min(window, int(self.remaining_prefill_tokens(seq)))

    def prefill_staging_context_lens_cpu(
        self,
        layer_idx: int,
    ) -> tuple[int, ...] | None:
        return getattr(
            self,
            "_pyramidkv_prefill_staging_context_lens_cpu_by_layer",
            {},
        ).get(int(layer_idx))

    def prefill_step_reservation_cost(self, seq: Sequence, scheduled_tokens: int) -> int:
        if self.requires_long_prefill_offload(seq):
            return 0
        return super().prefill_step_reservation_cost(seq, scheduled_tokens)

    def reserved_prefill_slots(self, waiting_seqs, engine_prefill_chunk_size: int) -> int:
        if not self._pyramidkv_can_use_full_prefill_staging():
            return super().reserved_prefill_slots(waiting_seqs, engine_prefill_chunk_size)
        reserved = 0
        for seq in waiting_seqs:
            if 0 < seq.num_prefilled_tokens < seq.num_prompt_tokens:
                reserved += self._pyramidkv_prompt_admission_cost(seq)
        return int(reserved)

    def prefill_batched_tokens_margin(self) -> int:
        return 0

    def remaining_prefill_tokens(self, seq: Sequence) -> int:
        return int(seq.num_prompt_tokens - seq.num_prefilled_tokens)

    def _prefill_score_dtype(self) -> torch.dtype:
        score_dtype_name = str(getattr(self.config, "sparse_attn_score_dtype", "float32") or "float32").lower()
        try:
            return {
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
            }[score_dtype_name]
        except KeyError as exc:
            raise ValueError(
                "sparse_attn_score_dtype must be 'float32', 'bfloat16', or 'float16', "
                f"got {score_dtype_name!r}."
            ) from exc

    def _prefill_score_layer_budget(self, layer_idx: int) -> int | None:
        if self.kv_layer_index(layer_idx) < int(getattr(self.config, "snapkv_num_full_layers", 0) or 0):
            return None
        if self.config.sparse_method == "pyramidkv":
            if self.config.pyramid_layer_ratios is None:
                return None
            return self._pyramidkv_layer_budget(layer_idx)
        if self.config.sparse_method == "snapkv":
            return (
                int(self.config.sink_keep_tokens)
                + int(self.config.decode_keep_tokens)
                + int(self.config.recent_keep_tokens)
            )
        return None

    def _prefill_score_rows(
        self,
        layer_idx: int,
        seqs: list[Sequence],
    ) -> list[tuple[int, Sequence, int, int]]:
        budget = self._prefill_score_layer_budget(layer_idx)
        if budget is None:
            return []
        window = int(getattr(self.config, "snapkv_window_size", 0) or 0)
        if window <= 0:
            return []

        rows = []
        for b_idx, seq in enumerate(seqs):
            if seq.current_chunk_size is None:
                raise RuntimeError(
                    "Prefill score collection requires current_chunk_size. "
                    f"layer={layer_idx} seq_id={seq.seq_id}"
                )
            prompt_len = int(seq.num_prompt_tokens)
            is_chain_resume = (
                getattr(seq, "chain_status", "") == "resumed"
                and not bool(getattr(seq, "is_recompute_replay", False))
            )
            reused_tokens = (
                0
                if bool(getattr(seq, "is_recompute_replay", False))
                else int(getattr(seq, "chain_reused_tokens", 0) or 0)
            )
            appended_delta_len = max(0, prompt_len - reused_tokens)
            if is_chain_resume:
                staging_context_lens = self.prefill_staging_context_lens_cpu(
                    layer_idx
                )
                physical_context_len = (
                    int(staging_context_lens[b_idx])
                    if staging_context_lens is not None
                    else self.chain_physical_kv_len(
                        layer_idx,
                        int(seq.seq_id),
                    )
                )
            else:
                physical_context_len = prompt_len
            if physical_context_len <= int(budget):
                continue
            score_window = min(window, appended_delta_len) if appended_delta_len else window
            logical_score_end = prompt_len
            logical_score_start = max(0, logical_score_end - score_window)
            chunk_start = int(seq.num_prefilled_tokens)
            chunk_end = chunk_start + int(seq.current_chunk_size)
            if (
                chunk_start <= logical_score_start
                and chunk_end >= logical_score_end
            ):
                if is_chain_resume:
                    score_end = physical_context_len
                    score_start = max(0, score_end - score_window)
                else:
                    score_start = logical_score_start
                    score_end = logical_score_end
                rows.append((b_idx, seq, score_start, score_end))
            elif (
                seq.is_last_chunk_prefill
                and chunk_start < logical_score_end
                and chunk_end > logical_score_start
            ):
                raise RuntimeError(
                    "SnapKV/PyramidKV prefill score requires the score query window to fit in "
                    "the final prefill chunk. "
                    f"layer={layer_idx} seq_id={seq.seq_id} "
                    f"score_range=[{logical_score_start}, {logical_score_end}) "
                    f"chunk_range=[{chunk_start}, {chunk_end})."
                )
        return rows

    def _clear_prefill_attention_scores(self, seq_id: int):
        seq_id = int(seq_id)
        for key in list(self._prefill_attn_score_accumulators):
            if key[1] == seq_id:
                self._prefill_attn_score_accumulators.pop(key, None)

    def _get_prefill_attention_score_accumulator(
        self,
        layer_idx: int,
        seq: Sequence,
        *,
        prompt_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        key = (int(layer_idx), int(seq.seq_id))
        if int(seq.num_prefilled_tokens) == 0:
            self._prefill_attn_score_accumulators.pop(key, None)
        acc = self._prefill_attn_score_accumulators.get(key)
        if acc is not None and int(acc.numel()) != int(prompt_len):
            raise RuntimeError(
                "Prefill attention-score accumulator length changed without "
                "a lifecycle reset: "
                f"layer={layer_idx} seq_id={seq.seq_id} "
                f"existing={int(acc.numel())} requested={int(prompt_len)}."
            )
        if acc is None:
            acc = torch.full(
                (int(prompt_len),),
                self._prefill_score_initial_value(),
                dtype=self._prefill_score_dtype(),
                device=device,
            )
        self._prefill_attn_score_accumulators[key] = acc
        return acc

    def _prefill_score_initial_value(self) -> float:
        mode = self.config.sparse_prefill_score_mode
        return -torch.inf if mode == "logits" else 0.0

    def _run_prefill_score(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        step_score: torch.Tensor,
        meta,
        b_start_loc: torch.Tensor,
        b_prompt_cache_len: torch.Tensor,
        max_query_len: int,
        score_starts: torch.Tensor,
        score_ends: torch.Tensor,
        *,
        candidate_start: int,
        recent_keep_tokens: int,
        batch_indices: torch.Tensor | None = None,
    ) -> None:
        mode = self.config.sparse_prefill_score_mode
        with profiler.record("prefill_token_score"):
            prefill_score_fwd(
                q,
                k_cache,
                step_score,
                meta.req_indices,
                b_start_loc,
                meta.context_lens,
                b_prompt_cache_len,
                max_query_len,
                meta.active_slots,
                score_starts,
                score_ends,
                candidate_start=candidate_start,
                recent_keep_tokens=recent_keep_tokens,
                score_mode=mode,
                workspace=getattr(self, "_prefill_score_workspace", None),
                batch_indices=batch_indices,
            )

    def _prefill_score_metadata_tensors(
        self,
        layer_idx: int,
        seqs: list[Sequence],
        rows: list[tuple[int, Sequence, int, int]],
        *,
        device: torch.device,
    ) -> tuple[
        tuple[int, ...],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        context_lens_by_layer = getattr(
            self,
            "_prefill_context_lens_cpu_by_layer",
            None,
        )
        context_lens = (
            None
            if context_lens_by_layer is None
            else context_lens_by_layer.get(int(layer_idx))
        )
        if context_lens is None:
            raise RuntimeError(
                "SnapKV prefill scoring requires CPU context lengths prepared "
                f"for layer={layer_idx}."
            )
        if len(context_lens) != len(seqs):
            raise RuntimeError(
                "SnapKV prefill scoring context-length batch mismatch: "
                f"layer={layer_idx} lengths={len(context_lens)} seqs={len(seqs)}."
            )

        prompt_cache_lens = tuple(
            int(context_len) - int(seq.current_chunk_size)
            for context_len, seq in zip(context_lens, seqs)
        )
        if any(value < 0 for value in prompt_cache_lens):
            raise RuntimeError(
                "SnapKV prefill scoring received a context shorter than its chunk: "
                f"layer={layer_idx} context_lens={context_lens} "
                f"prompt_cache_lens={prompt_cache_lens}."
            )
        batch_indices = []
        score_starts = []
        score_ends = []
        for b_idx, _seq, score_start, score_end in rows:
            if not 0 <= int(b_idx) < len(seqs):
                raise RuntimeError(
                    "SnapKV prefill score row is outside the current batch: "
                    f"layer={layer_idx} row={b_idx} batch={len(seqs)}."
                )
            batch_indices.append(int(b_idx))
            score_starts.append(int(score_start))
            score_ends.append(int(score_end))

        tensors = self._cached_prefill_score_metadata_tensors(
            device=device,
            context_lens=context_lens,
            prompt_cache_lens=prompt_cache_lens,
            batch_indices=tuple(batch_indices),
            score_starts=tuple(score_starts),
            score_ends=tuple(score_ends),
        )
        return context_lens, *tensors

    def _cached_prefill_score_metadata_tensors(
        self,
        *,
        device: torch.device,
        context_lens: tuple[int, ...],
        prompt_cache_lens: tuple[int, ...],
        batch_indices: tuple[int, ...],
        score_starts: tuple[int, ...],
        score_ends: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        signature = (
            device,
            context_lens,
            prompt_cache_lens,
            batch_indices,
            score_starts,
            score_ends,
        )
        cache = getattr(self, "_prefill_score_metadata_cache", None)
        if cache is None:
            cache = {}
            self._prefill_score_metadata_cache = cache
        tensors = cache.get(signature)
        if tensors is None:
            tensors = (
                torch.tensor(prompt_cache_lens, dtype=torch.int32, device=device),
                torch.tensor(batch_indices, dtype=torch.int32, device=device),
                torch.tensor(score_starts, dtype=torch.int32, device=device),
                torch.tensor(score_ends, dtype=torch.int32, device=device),
            )
            cache[signature] = tensors
        return tensors

    def _prefill_step_score_buffer(
        self,
        *,
        batch_size: int,
        max_context_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        dtype = self._prefill_score_dtype()
        key = (device, dtype)
        buffers = getattr(self, "_prefill_step_score_buffers", None)
        if buffers is None:
            buffers = {}
            self._prefill_step_score_buffers = buffers
        buffer = buffers.get(key)
        if (
            buffer is None
            or int(buffer.shape[0]) < int(batch_size)
            or int(buffer.shape[1]) < int(max_context_len)
        ):
            rows = max(int(batch_size), 0 if buffer is None else int(buffer.shape[0]))
            columns = max(
                int(max_context_len),
                0 if buffer is None else int(buffer.shape[1]),
            )
            buffer = torch.empty((rows, columns), dtype=dtype, device=device)
            buffers[key] = buffer
        return buffer[:batch_size, :max_context_len]

    def prefill_score_request(self, layer_idx, seqs):
        rows = self._prefill_score_rows(layer_idx, seqs)
        if not rows:
            return None
        ranges = [(0, 0)] * len(seqs)
        for batch_idx, _seq, start, end in rows:
            ranges[batch_idx] = (start, end)
        return PrefillScoreRequest(
            tuple(ranges), self.config.sparse_prefill_score_mode,
            int(self.config.sink_keep_tokens), int(self.config.recent_keep_tokens),
        )

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
        del attention_lse
        ctx = get_context()
        if not ctx.is_prefill:
            return None
        if self.config.sparse_method not in ("snapkv", "pyramidkv"):
            return None
        seqs = getattr(ctx, "seqs", None)
        if seqs is None:
            raise RuntimeError("Prefill score collection requires current seqs in context.")

        rows = self._prefill_score_rows(layer_idx, seqs)
        if not rows:
            return None
        if view.token_scores is None and not isinstance(view.payload, ExplicitKVPayload):
            raise TypeError(
                "SnapKV prefill scoring requires ExplicitKVPayload, got "
                f"{type(view.payload).__name__}."
            )
        meta = view.meta
        payload = view.payload

        if int(chunk_lens.ndim) != 1 or int(chunk_lens.shape[0]) != len(seqs):
            raise RuntimeError(
                "SnapKV prefill scoring chunk-length batch mismatch: "
                f"shape={tuple(chunk_lens.shape)} seqs={len(seqs)}."
            )
        max_score_len = max(
            int(score_end) - int(score_start)
            for _b_idx, _seq, score_start, score_end in rows
        )
        (
            context_lens,
            b_prompt_cache_len,
            score_batch_indices,
            score_starts,
            score_ends,
        ) = self._prefill_score_metadata_tensors(
            layer_idx,
            seqs,
            rows,
            device=q.device,
        )

        max_context_len = max(
            int(context_lens[b_idx]) for b_idx, _seq, _start, _end in rows
        )
        step_score = self._prefill_step_score_buffer(
            batch_size=len(rows),
            max_context_len=max_context_len,
            device=q.device,
        )
        if view.token_scores is not None:
            step_score.copy_(view.token_scores[:, :max_context_len].index_select(
                0, score_batch_indices.long(),
            ))
        else:
            self._run_prefill_score(
                q,
                payload.k_cache,
                step_score,
                meta,
                b_start_loc,
                b_prompt_cache_len,
                max_score_len,
                score_starts,
                score_ends,
                candidate_start=int(self.config.sink_keep_tokens),
                recent_keep_tokens=int(self.config.recent_keep_tokens),
                batch_indices=score_batch_indices,
            )

        for score_row_idx, (b_idx, seq, _score_start, _score_end) in enumerate(rows):
            context_len = int(context_lens[b_idx])
            acc = self._get_prefill_attention_score_accumulator(
                layer_idx,
                seq,
                prompt_len=context_len,
                device=q.device,
            )
            torch.maximum(
                acc[:context_len],
                step_score[score_row_idx, :context_len],
                out=acc[:context_len],
            )
        return None

    def pop_prefill_attention_score(self, layer_idx: int, seq: Sequence) -> torch.Tensor | None:
        return self._prefill_attn_score_accumulators.pop((int(layer_idx), int(seq.seq_id)), None)

    def _get_free_row(self, layer_idx: int, seq_id: int) -> int:
        if seq_id in self.seq_id_to_row[layer_idx]:
            return self.seq_id_to_row[layer_idx][seq_id]
        if not self.free_rows[layer_idx]:
            raise RuntimeError("No free rows in cache manager buffer!")
        row_idx = self.free_rows[layer_idx].popleft()
        self.seq_id_to_row[layer_idx][seq_id] = row_idx
        return row_idx

    @torch.no_grad()
    def _allocate(self, layer_idx: int, seq_id: int, size: int) -> torch.Tensor:
        with profiler.record("cache_allocate"):
            if type(size) is not int or size < 0:
                raise ValueError("KV allocation size must be a non-negative integer.")
            if self._num_free_slots[layer_idx] < size:
                raise RuntimeError(
                    f"Out of KV cache slots: need {size}, free {self._num_free_slots[layer_idx]}"
                )
            existing_row = self.seq_id_to_row[layer_idx].get(seq_id)
            cur_len = 0 if existing_row is None else int(self.row_seq_lens[layer_idx][existing_row])
            limit = min(int(self.max_model_len), self.buffer_req_to_token_slots[layer_idx].shape[1])
            if cur_len + size > limit:
                raise RuntimeError(
                    "KV row length exceeds max_model_len in _allocate: "
                    f"layer={layer_idx} seq_id={seq_id} row={existing_row} "
                    f"cur_len={cur_len} size={size} max_model_len={limit}"
                )
            row_idx = self._get_free_row(layer_idx, seq_id)
            ptr = self._num_free_slots[layer_idx]
            select_index = self.free_slots_stack[layer_idx][ptr - size: ptr]
            try:
                # Publish ownership only after the row write succeeds. The ID
                # view is not copied; ordinary prefill keeps its zero-copy path.
                self.buffer_req_to_token_slots[layer_idx][row_idx, cur_len:cur_len + size] = select_index
                self.row_seq_lens[layer_idx][row_idx] = cur_len + size
                self._num_free_slots[layer_idx] = ptr - size
            except BaseException:
                self._num_free_slots[layer_idx] = ptr
                self.row_seq_lens[layer_idx][row_idx] = cur_len
                if existing_row is None:
                    self.seq_id_to_row[layer_idx].pop(seq_id, None)
                    self.free_rows[layer_idx].appendleft(row_idx)
                raise
            return select_index

    def _ensure_decode_buffers(self, batch_size: int):
        if not hasattr(self, "_decode_buf_capacity") or self._decode_buf_capacity < batch_size:
            cap = max(batch_size, getattr(self, "_decode_buf_capacity", 0) * 2, 64)
            self._decode_buf_capacity = cap
            pin_memory = device_runtime.supports_pin_memory()
            self._pinned_input_ids = torch.empty(cap, dtype=torch.int64, pin_memory=pin_memory)
            self._pinned_positions = torch.empty(cap, dtype=torch.int64, pin_memory=pin_memory)
            self._cuda_input_ids = torch.empty(cap, dtype=torch.int64, device=self.device)
            self._cuda_positions = torch.empty(cap, dtype=torch.int64, device=self.device)

            self._pinned_layers_context_lens = torch.empty((self.num_layers, cap), dtype=torch.int32, pin_memory=pin_memory)
            self._cuda_layers_context_lens = torch.empty((self.num_layers, cap), dtype=torch.int32, device=self.device)
            self._cuda_layers_slot_mapping = torch.empty((self.num_layers, cap), dtype=torch.int32, device=self.device)
            self._pinned_layers_req_indices = torch.empty((self.num_layers, cap), dtype=torch.int32, pin_memory=pin_memory)
            self._cuda_layers_req_indices = torch.empty((self.num_layers, cap), dtype=torch.int32, device=self.device)

            self._static_rows_gpu = torch.empty(cap, dtype=torch.long, device=self.device)
            self._static_cols_gpu = torch.empty(cap, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def _allocate_batch(self, layer_idx: int, seq_ids: list[int], size: int) -> torch.Tensor:
        assert size == 1, "Batch allocation currently only supports size=1 (Decode)"
        batch_size = len(seq_ids)
        assert self._num_free_slots[layer_idx] >= batch_size, (
            f"Out of KV cache slots: need {batch_size}, free {self._num_free_slots[layer_idx]}"
        )
        needed_rows = len(set(seq_ids).difference(self.seq_id_to_row[layer_idx]))
        if needed_rows > len(self.free_rows[layer_idx]):
            raise RuntimeError(
                f"No free rows for decode batch: need={needed_rows} "
                f"free={len(self.free_rows[layer_idx])}."
            )
        existing = [self.seq_id_to_row[layer_idx][sid] for sid in seq_ids
                    if sid in self.seq_id_to_row[layer_idx]]
        if any(int(self.row_seq_lens[layer_idx][row]) + size > self.max_model_len for row in existing):
            raise RuntimeError("KV row length exceeds max_model_len in _allocate_batch.")
        self._ensure_decode_buffers(batch_size)

        new_seq_ids = list(dict.fromkeys(
            sid for sid in seq_ids if sid not in self.seq_id_to_row[layer_idx]
        ))
        row_indices = [self._get_free_row(layer_idx, sid) for sid in seq_ids]
        cur_lens = self.row_seq_lens[layer_idx][row_indices]
        ptr = self._num_free_slots[layer_idx]
        select_indices = self.free_slots_stack[layer_idx][ptr - batch_size: ptr]
        self._num_free_slots[layer_idx] -= batch_size
        try:
            rows_gpu = self._static_rows_gpu[:batch_size]
            cols_gpu = self._static_cols_gpu[:batch_size]
            rows_gpu.copy_(torch.as_tensor(row_indices, dtype=torch.long), non_blocking=True)
            cols_gpu.copy_(torch.as_tensor(cur_lens, dtype=torch.long), non_blocking=True)
            self.buffer_req_to_token_slots[layer_idx][rows_gpu, cols_gpu] = select_indices.to(torch.int32)
            self.row_seq_lens[layer_idx][row_indices] += 1
        except BaseException as error:
            try:
                for row_idx, cur_len in zip(row_indices, cur_lens, strict=True):
                    self.buffer_req_to_token_slots[layer_idx][int(row_idx), int(cur_len)] = 0
                self.row_seq_lens[layer_idx][row_indices] = cur_lens
                if device_runtime.supports_streams(self.device):
                    device_runtime.synchronize()
                for seq_id in reversed(new_seq_ids):
                    row_idx = self.seq_id_to_row[layer_idx].pop(seq_id)
                    self.free_rows[layer_idx].appendleft(row_idx)
                self._num_free_slots[layer_idx] = ptr
            except BaseException as rollback_error:
                quarantined = getattr(self, "_quarantined_decode_allocations", None)
                if quarantined is None:
                    quarantined = []
                    self._quarantined_decode_allocations = quarantined
                quarantined.append(
                    (int(layer_idx), tuple(int(seq_id) for seq_id in seq_ids), select_indices)
                )
                error.add_note(
                    "Decode allocation rollback could not fence and restore metadata; "
                    "slot ownership was quarantined instead of being recycled."
                )
                raise error from rollback_error
            raise

        return select_indices

    @torch.no_grad()
    def _allocate_prefill_batch_same_size_all_layers(
        self,
        seqs: list[Sequence],
        layers_slot_mapping: torch.Tensor,
    ) -> bool:
        if self.free_slots_stack_tensor is None or not seqs:
            return False
        chunk_size = int(seqs[0].current_chunk_size)
        if chunk_size <= 0 or any(int(seq.current_chunk_size) != chunk_size for seq in seqs):
            return False

        batch_size = len(seqs)
        total_size = batch_size * chunk_size
        layer_ids = self.kv_transformer_layer_indices()
        if not layer_ids:
            return False
        with profiler.record("cache_allocate"):
            min_free = min(int(self._num_free_slots[layer_id]) for layer_id in layer_ids)
            if min_free < total_size:
                raise RuntimeError(
                    "Out of KV cache slots in batched prefill allocation: "
                    f"need={total_size} free={min_free}"
                )

            row_indices = np.empty((len(layer_ids), batch_size), dtype=np.int64)
            start_lens = np.empty((len(layer_ids), batch_size), dtype=np.int64)
            for local_layer, layer_id in enumerate(layer_ids):
                for seq_idx, seq in enumerate(seqs):
                    row_idx = self._get_free_row(layer_id, int(seq.seq_id))
                    row_len = int(self.row_seq_lens[layer_id][row_idx])
                    is_chain_resume = bool(
                        getattr(seq, "chain_status", "") == "resumed"
                        and not bool(getattr(seq, "is_recompute_replay", False))
                    )
                    expected_start = int(seq.num_prefilled_tokens)
                    if not is_chain_resume and row_len != expected_start:
                        raise ValueError(
                            "KV cache row length mismatch in batched prefill allocation: "
                            f"layer={layer_id} seq_id={seq.seq_id} row_seq_len={row_len} "
                            f"start_idx={expected_start}"
                        )
                    if row_len + chunk_size > int(self.max_model_len):
                        raise RuntimeError(
                            "KV row length exceeds max_model_len in batched prefill allocation: "
                            f"layer={layer_id} seq_id={seq.seq_id} row={row_idx} "
                            f"cur_len={row_len} size={chunk_size} max_model_len={int(self.max_model_len)}"
                        )
                    row_indices[local_layer, seq_idx] = int(row_idx)
                    start_lens[local_layer, seq_idx] = row_len

            layers_gpu = torch.tensor(layer_ids, dtype=torch.long, device=self.device)
            kv_layers_gpu = torch.tensor(
                [self.kv_layer_index(layer_id) for layer_id in layer_ids],
                dtype=torch.long,
                device=self.device,
            )
            first_layer = int(layer_ids[0])
            if all(int(self._num_free_slots[layer_id]) == int(self._num_free_slots[first_layer]) for layer_id in layer_ids):
                ptr = int(self._num_free_slots[first_layer])
                selected_slots = self.free_slots_stack_tensor[kv_layers_gpu, ptr - total_size: ptr].view(
                    len(layer_ids),
                    batch_size,
                    chunk_size,
                ).flip(1)
            else:
                ptrs = np.asarray([self._num_free_slots[layer_id] for layer_id in layer_ids], dtype=np.int64)
                seq_offsets = (batch_size - 1 - np.arange(batch_size, dtype=np.int64)) * chunk_size
                token_offsets = np.arange(chunk_size, dtype=np.int64)
                slot_offsets = (
                    ptrs[:, None, None]
                    - total_size
                    + seq_offsets[None, :, None]
                    + token_offsets[None, None, :]
                )
                slot_offsets_gpu = torch.from_numpy(slot_offsets).to(device=self.device, dtype=torch.long)
                selected_slots = self.free_slots_stack_tensor[
                    kv_layers_gpu[:, None, None],
                    slot_offsets_gpu,
                ]

            rows_gpu = torch.from_numpy(row_indices).to(device=self.device, dtype=torch.long)
            cols_gpu = (
                torch.from_numpy(start_lens).to(device=self.device, dtype=torch.long)[:, :, None]
                + torch.arange(chunk_size, dtype=torch.long, device=self.device)[None, None, :]
            )
            self.buffer_req_to_token_slots_tensor[
                kv_layers_gpu[:, None, None],
                rows_gpu[:, :, None],
                cols_gpu,
            ] = selected_slots.to(torch.int32)
            layers_slot_mapping[layers_gpu, :total_size] = selected_slots.reshape(len(layer_ids), total_size)
            for local_layer, layer_id in enumerate(layer_ids):
                self._num_free_slots[layer_id] -= total_size
                self.row_seq_lens[layer_id][row_indices[local_layer]] += chunk_size
            return True

    def free_seq(self, seq_id: int):
        with profiler.record("cache_free_seq"):
            self.reset_prefill_execution_state(seq_id)
            self._clear_prefill_attention_scores(seq_id)
            self._pyramidkv_clear_long_prefill_offload_prefetch()
            for layer_idx in self.kv_transformer_layer_indices():
                row_idx = self.seq_id_to_row[layer_idx].pop(seq_id, None)
                if row_idx is None:
                    # A failed prepare/restore can own only a subset of layers.
                    # Repeated terminal cleanup must still visit the other layers.
                    continue
                self.raw_kv_offload_buffer.release_layer(
                    layer_idx=layer_idx,
                    row_idx=int(row_idx),
                    kind=self._pyramidkv_long_prefill_offload_kind(),
                )

                cur_len = self.row_seq_lens[layer_idx][row_idx]
                slots = self.buffer_req_to_token_slots[layer_idx][row_idx, :cur_len]

                if cur_len > 0:
                    ptr = self._num_free_slots[layer_idx]
                    self.free_slots_stack[layer_idx][ptr: ptr + cur_len] = slots
                    self._num_free_slots[layer_idx] += cur_len

                self.buffer_req_to_token_slots[layer_idx][row_idx, :] = 0
                self.row_seq_lens[layer_idx][row_idx] = 0
                self.free_rows[layer_idx].append(row_idx)

    def decode_kv_lens_for_layer(self, layer_idx: int, seqs: list[Sequence]) -> list[int]:
        self.kv_layer_index(layer_idx)
        kv_lens = []
        for seq in seqs:
            row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
            if row_idx is None:
                raise RuntimeError(
                    f"Missing decode row for seq_id={seq.seq_id} on layer={layer_idx}."
                )
            kv_lens.append(int(self.row_seq_lens[layer_idx][row_idx]))
        return kv_lens

    @staticmethod
    def _assert_compaction_indices(
        condition: torch.Tensor,
        message: str,
        *,
        synchronize: bool = False,
    ) -> None:
        if condition.numel() != 1:
            raise RuntimeError(
                "KV page-table compaction validation must reduce to one boolean: "
                f"shape={tuple(condition.shape)}."
            )
        if condition.is_cuda and not synchronize:
            torch._assert_async(condition)
        elif not bool(condition.item()):
            raise RuntimeError(message)

    def _prepare_compaction_keep_indices(
        self,
        keep_indices: torch.Tensor,
        row_lens: torch.Tensor,
        *,
        sort_dim: int,
        keep_indices_sorted: bool,
        operation: str,
        synchronize_validation: bool = False,
    ) -> torch.Tensor:
        keep_indices = keep_indices.to(
            device=self.device,
            dtype=torch.long,
        ).contiguous()
        if not keep_indices_sorted:
            keep_indices = torch.sort(keep_indices, dim=sort_dim).values
        row_lens = row_lens.to(
            device=self.device,
            dtype=torch.long,
        )
        self._assert_compaction_indices(
            ((keep_indices >= 0) & (keep_indices < row_lens.unsqueeze(-1))).all(),
            f"{operation} keep_indices out of bounds.",
            synchronize=synchronize_validation,
        )
        if int(keep_indices.shape[sort_dim]) > 1:
            self._assert_compaction_indices(
                (
                    keep_indices.narrow(sort_dim, 1, keep_indices.shape[sort_dim] - 1)
                    > keep_indices.narrow(sort_dim, 0, keep_indices.shape[sort_dim] - 1)
                ).all(),
                f"{operation} keep_indices must be unique.",
                synchronize=synchronize_validation,
            )
        return keep_indices

    def _preflight_compaction_free_capacity(
        self,
        layer_idx: int,
        release_count: int,
        *,
        operation: str,
    ) -> None:
        layer_idx = int(layer_idx)
        release_count = int(release_count)
        free_stack = self.free_slots_stack[layer_idx]
        if free_stack is None or free_stack.dim() != 1:
            raise RuntimeError(
                f"{operation} requires a one-dimensional free-slot stack: "
                f"layer={layer_idx}."
            )
        ptr = int(self._num_free_slots[layer_idx])
        if release_count < 0 or ptr < 0 or ptr + release_count > int(free_stack.numel()):
            raise RuntimeError(
                f"{operation} would overflow the free-slot stack: "
                f"layer={layer_idx} ptr={ptr} release={release_count} "
                f"capacity={int(free_stack.numel())}."
            )

    def _prepare_free_part_slots(
        self,
        layer_idx: int,
        seq: Sequence,
        keep_indices: torch.Tensor,
        *,
        keep_indices_sorted: bool,
        synchronize_validation: bool = False,
    ) -> tuple[int, int, torch.Tensor]:
        self.kv_layer_index(layer_idx)
        row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
        if row_idx is None:
            raise ValueError
        cur_len = int(self.row_seq_lens[layer_idx][row_idx])
        keep_indices = keep_indices.to(
            device=self.device,
            dtype=torch.long,
        ).contiguous()
        if keep_indices.dim() != 1:
            raise RuntimeError(
                "free_part_slots expected one-dimensional keep_indices: "
                f"layer={layer_idx} seq_id={seq.seq_id} "
                f"keep_shape={tuple(keep_indices.shape)}"
            )
        if keep_indices.numel() <= 0:
            raise RuntimeError(
                f"free_part_slots got empty keep_indices: layer={layer_idx} seq_id={seq.seq_id}"
            )
        keep_indices = self._prepare_compaction_keep_indices(
            keep_indices,
            torch.tensor(cur_len, dtype=torch.long, device=self.device),
            sort_dim=0,
            keep_indices_sorted=keep_indices_sorted,
            operation="free_part_slots",
            synchronize_validation=synchronize_validation,
        )
        self._preflight_compaction_free_capacity(
            layer_idx,
            cur_len - int(keep_indices.numel()),
            operation="free_part_slots",
        )
        return int(row_idx), cur_len, keep_indices

    def _append_compaction_free_slots(
        self,
        layer_idx: int,
        dropped_slots: torch.Tensor,
    ) -> None:
        count = int(dropped_slots.numel())
        if count <= 0:
            return
        ptr = int(self._num_free_slots[layer_idx])
        self.free_slots_stack[layer_idx][ptr: ptr + count] = dropped_slots
        self._num_free_slots[layer_idx] = ptr + count

    def _apply_free_part_slots(
        self,
        layer_idx: int,
        seq: Sequence,
        row_idx: int,
        cur_len: int,
        keep_indices: torch.Tensor,
    ) -> None:
        self._uniform_decode_metadata = False
        old_slots = self.buffer_req_to_token_slots[layer_idx][row_idx, :cur_len].clone()
        new_slots = old_slots[keep_indices]

        mask = torch.ones_like(old_slots, dtype=torch.bool)
        mask[keep_indices] = False
        dropped_slots = old_slots[mask]
        self._append_compaction_free_slots(layer_idx, dropped_slots)
        if dropped_slots.numel() <= 0:
            logger.warning(
                f"[SnapKV] dropped 0 tokens? layer={layer_idx} "
                f"seq_id={seq.seq_id} row={row_idx} cur_len={cur_len}"
            )

        self.buffer_req_to_token_slots[layer_idx][row_idx, :] = 0
        self.buffer_req_to_token_slots[layer_idx][row_idx, :new_slots.numel()] = new_slots
        self.row_seq_lens[layer_idx][row_idx] = new_slots.numel()
        if log_level == 'DEBUG':
            logger.debug(
                "[SnapKV] free_part_slots(after): "
                f"layer={layer_idx} seq_id={seq.seq_id} row={row_idx} "
                f"context_len={cur_len} -> {int(new_slots.numel())}"
            )

    def _page_table_compaction_tile_elements(self) -> int:
        tile_elements = int(
            getattr(
                self,
                "_page_table_compaction_tile_elements_override",
                _PAGE_TABLE_COMPACTION_TILE_ELEMENTS,
            )
        )
        if tile_elements <= 0:
            raise RuntimeError(
                "Page-table compaction tile size must be positive: "
                f"tile_elements={tile_elements}."
            )
        return tile_elements

    def _compact_uniform_row_tile(
        self,
        layer_idx: int,
        row_indices: list[int],
        cur_len: int,
        keep_indices: torch.Tensor,
    ) -> int:
        rows_gpu = torch.tensor(row_indices, dtype=torch.long, device=self.device)
        old_slots = self.buffer_req_to_token_slots[layer_idx][rows_gpu, :cur_len]
        new_slots = old_slots.gather(1, keep_indices)
        mask = torch.ones_like(old_slots, dtype=torch.bool)
        mask.scatter_(1, keep_indices, False)
        dropped_slots = old_slots[mask]
        self._append_compaction_free_slots(layer_idx, dropped_slots)

        new_len = int(new_slots.shape[1])
        self.buffer_req_to_token_slots[layer_idx][rows_gpu, :new_len] = new_slots
        self.buffer_req_to_token_slots[layer_idx][rows_gpu, new_len:] = 0
        self.row_seq_lens[layer_idx][row_indices] = new_len
        return int(dropped_slots.numel())

    def _compact_single_row_column_tiles(
        self,
        layer_idx: int,
        row_idx: int,
        cur_len: int,
        keep_indices: torch.Tensor,
        tile_elements: int,
    ) -> int:
        slot_row = self.buffer_req_to_token_slots[layer_idx][row_idx]
        dropped_count = 0

        # Complete the dropped-slot read before rewriting any source entry.
        # Each old-slot, mask, dropped-slot, and selected-slot temporary is
        # capped at tile_elements entries.
        for start in range(0, cur_len, tile_elements):
            end = min(cur_len, start + tile_elements)
            old_slots = slot_row[start:end].clone()
            mask = torch.ones_like(old_slots, dtype=torch.bool)
            left = int(torch.searchsorted(keep_indices, start).item())
            right = int(torch.searchsorted(keep_indices, end).item())
            if right > left:
                mask[keep_indices[left:right] - start] = False
            dropped_slots = old_slots[mask]
            self._append_compaction_free_slots(layer_idx, dropped_slots)
            dropped_count += int(dropped_slots.numel())

        new_len = int(keep_indices.numel())
        for start in range(0, new_len, tile_elements):
            end = min(new_len, start + tile_elements)
            selected_slots = slot_row.index_select(
                0, keep_indices[start:end]
            )
            # Sorted unique keep indices satisfy keep[i] >= i. Therefore an
            # earlier destination tile cannot overwrite a later source tile.
            slot_row[start:end] = selected_slots
        slot_row[new_len:] = 0
        self.row_seq_lens[layer_idx][row_idx] = new_len
        return dropped_count

    def _compact_uniform_rows_bounded(
        self,
        layer_idx: int,
        row_indices: list[int],
        cur_len: int,
        keep_indices: torch.Tensor,
    ) -> None:
        """Compact rows without a layer/batch/full-context temporary."""
        tile_elements = self._page_table_compaction_tile_elements()
        dropped_count = 0
        if cur_len <= tile_elements:
            rows_per_tile = max(1, tile_elements // max(1, cur_len))
            for start in range(0, len(row_indices), rows_per_tile):
                end = min(len(row_indices), start + rows_per_tile)
                dropped_count += self._compact_uniform_row_tile(
                    layer_idx,
                    row_indices[start:end],
                    cur_len,
                    keep_indices[start:end],
                )
        else:
            for row_idx, row_keep_indices in zip(row_indices, keep_indices):
                dropped_count += self._compact_single_row_column_tiles(
                    layer_idx,
                    int(row_idx),
                    cur_len,
                    row_keep_indices,
                    tile_elements,
                )
        if dropped_count <= 0:
            logger.warning(
                f"[SnapKV] dropped 0 tokens in bounded batch? layer={layer_idx} "
                f"rows={row_indices} cur_len={cur_len}"
            )

    def _compact_uniform_layer_tile(
        self,
        layer_indices: list[int],
        kv_layer_indices: list[int],
        row_indices: np.ndarray,
        cur_len: int,
        keep_indices: torch.Tensor,
        common_slot_table: torch.Tensor,
    ) -> None:
        """Compact a bounded tile of compatible layers in shared Torch ops."""
        kv_layers_gpu = torch.tensor(
            kv_layer_indices,
            dtype=torch.long,
            device=self.device,
        )
        rows_gpu = torch.from_numpy(row_indices).to(
            device=self.device,
            dtype=torch.long,
        )
        old_slots = common_slot_table[
            kv_layers_gpu[:, None],
            rows_gpu,
            :cur_len,
        ]
        new_slots = old_slots.gather(2, keep_indices)
        mask = torch.ones_like(old_slots, dtype=torch.bool)
        mask.scatter_(2, keep_indices, False)
        dropped_per_layer = old_slots[mask].view(
            len(layer_indices),
            int(row_indices.shape[1]) * (cur_len - int(keep_indices.shape[2])),
        )
        for local_layer, layer_idx in enumerate(layer_indices):
            self._append_compaction_free_slots(
                int(layer_idx),
                dropped_per_layer[local_layer],
            )

        new_len = int(new_slots.shape[2])
        common_slot_table[
            kv_layers_gpu[:, None],
            rows_gpu,
            :new_len,
        ] = new_slots
        common_slot_table[
            kv_layers_gpu[:, None],
            rows_gpu,
            new_len:,
        ] = 0
        for local_layer, layer_idx in enumerate(layer_indices):
            self.row_seq_lens[int(layer_idx)][row_indices[local_layer]] = new_len
        if dropped_per_layer.numel() <= 0:
            logger.warning(
                "[SnapKV] dropped 0 tokens in bounded layer batch? "
                f"layers={layer_indices} rows={row_indices.tolist()} "
                f"cur_len={cur_len}"
            )

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
        row_idx, cur_len, keep_indices = self._prepare_free_part_slots(
            layer_idx,
            seq,
            keep_indices,
            keep_indices_sorted=keep_indices_sorted,
            synchronize_validation=self.validate_runtime_invariants,
        )
        if log_level == 'DEBUG':
            keep_cnt = int(keep_indices.numel())
            logger.debug(
                "[SnapKV] free_part_slots(before): "
                f"layer={layer_idx} seq_id={seq.seq_id} row={row_idx} "
                f"context_len={int(cur_len)} keep={keep_cnt} drop={max(0, int(cur_len) - keep_cnt)}"
            )
        self._apply_free_part_slots(
            layer_idx,
            seq,
            row_idx,
            cur_len,
            keep_indices,
        )

    def free_part_slots_batch(
        self,
        layer_idx: int,
        seqs: list[Sequence],
        keep_indices: torch.Tensor,
        *,
        keep_indices_sorted: bool = False,
    ):
        if keep_indices is None:
            return
        if not seqs:
            return
        self.kv_layer_index(layer_idx)
        keep_indices = keep_indices.to(device=self.device, dtype=torch.long).contiguous()
        if keep_indices.dim() != 2 or int(keep_indices.shape[0]) != len(seqs):
            raise RuntimeError(
                "free_part_slots_batch expected keep_indices with shape [batch, keep]: "
                f"batch={len(seqs)} keep_shape={tuple(keep_indices.shape)}"
            )
        if int(keep_indices.shape[1]) <= 0:
            raise RuntimeError(f"free_part_slots_batch got empty keep_indices: layer={layer_idx}")
        if len(seqs) == 1:
            SnapKVCacheManager.free_part_slots(
                self,
                layer_idx,
                seqs[0],
                keep_indices[0],
                keep_indices_sorted=keep_indices_sorted,
            )
            return

        row_indices = []
        cur_lens = []
        for seq in seqs:
            row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
            if row_idx is None:
                raise ValueError
            row_indices.append(int(row_idx))
            cur_lens.append(int(self.row_seq_lens[layer_idx][row_idx]))

        if len(row_indices) != len(set(row_indices)):
            raise RuntimeError(
                "free_part_slots_batch received duplicate physical rows: "
                f"layer={layer_idx} rows={row_indices}."
            )
        keep_indices = self._prepare_compaction_keep_indices(
            keep_indices,
            torch.tensor(cur_lens, dtype=torch.long, device=self.device),
            sort_dim=1,
            keep_indices_sorted=keep_indices_sorted,
            operation="free_part_slots_batch",
            synchronize_validation=self.validate_runtime_invariants,
        )
        total_release_count = sum(
            int(cur_len) - int(keep_indices.shape[1]) for cur_len in cur_lens
        )
        self._preflight_compaction_free_capacity(
            layer_idx,
            total_release_count,
            operation="free_part_slots_batch",
        )
        self._uniform_decode_metadata = False

        first_len = cur_lens[0]
        if any(cur_len != first_len for cur_len in cur_lens):
            for seq, seq_keep_indices in zip(seqs, keep_indices):
                SnapKVCacheManager.free_part_slots(
                    self,
                    layer_idx,
                    seq,
                    seq_keep_indices,
                    keep_indices_sorted=True,
                )
            return
        cur_len = int(first_len)
        self._compact_uniform_rows_bounded(
            layer_idx,
            row_indices,
            cur_len,
            keep_indices,
        )

    def free_part_slots_batch_layers(
        self,
        layer_indices: list[int],
        seqs: list[Sequence],
        keep_indices: torch.Tensor,
        *,
        keep_indices_sorted: bool = False,
    ):
        if keep_indices is None:
            return
        if not layer_indices or not seqs:
            return
        for layer_idx in layer_indices:
            self.kv_layer_index(int(layer_idx))
        keep_indices = keep_indices.to(device=self.device, dtype=torch.long).contiguous()
        num_layers = len(layer_indices)
        batch_size = len(seqs)
        if keep_indices.dim() != 3 or tuple(keep_indices.shape[:2]) != (num_layers, batch_size):
            raise RuntimeError(
                "free_part_slots_batch_layers expected keep_indices with shape [layers, batch, keep]: "
                f"layers={num_layers} batch={batch_size} keep_shape={tuple(keep_indices.shape)}"
            )
        if int(keep_indices.shape[2]) <= 0:
            raise RuntimeError("free_part_slots_batch_layers got empty keep_indices.")
        normalized_layers = [int(layer_idx) for layer_idx in layer_indices]
        if len(normalized_layers) != len(set(normalized_layers)):
            raise RuntimeError(
                "free_part_slots_batch_layers received duplicate layers: "
                f"layers={normalized_layers}."
            )
        if len(layer_indices) == 1:
            SnapKVCacheManager.free_part_slots_batch(
                self,
                normalized_layers[0],
                seqs,
                keep_indices[0],
                keep_indices_sorted=keep_indices_sorted,
            )
            return

        row_indices = np.empty((num_layers, batch_size), dtype=np.int64)
        cur_lens = np.empty((num_layers, batch_size), dtype=np.int64)
        for local_layer, layer_idx in enumerate(layer_indices):
            layer_idx = int(layer_idx)
            for seq_idx, seq in enumerate(seqs):
                row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
                if row_idx is None:
                    raise ValueError
                row_indices[local_layer, seq_idx] = int(row_idx)
                cur_lens[local_layer, seq_idx] = int(self.row_seq_lens[layer_idx][row_idx])

        for local_layer, layer_idx in enumerate(layer_indices):
            rows = row_indices[local_layer].tolist()
            if len(rows) != len(set(rows)):
                raise RuntimeError(
                    "free_part_slots_batch_layers received duplicate physical rows: "
                    f"layer={int(layer_idx)} rows={rows}."
                )
        keep_indices = self._prepare_compaction_keep_indices(
            keep_indices,
            torch.from_numpy(cur_lens).to(device=self.device, dtype=torch.long),
            sort_dim=2,
            keep_indices_sorted=keep_indices_sorted,
            operation="free_part_slots_batch_layers",
            synchronize_validation=self.validate_runtime_invariants,
        )
        for local_layer, layer_idx in enumerate(layer_indices):
            release_count = int(
                np.sum(
                    cur_lens[local_layer]
                    - int(keep_indices.shape[2])
                )
            )
            self._preflight_compaction_free_capacity(
                int(layer_idx),
                release_count,
                operation="free_part_slots_batch_layers",
            )
        self._uniform_decode_metadata = False

        tile_elements = self._page_table_compaction_tile_elements()
        cur_len = int(cur_lens[0, 0])
        elements_per_layer = batch_size * cur_len
        common_slot_table = getattr(
            self,
            "buffer_req_to_token_slots_tensor",
            None,
        )
        kv_layer_indices = [
            self.kv_layer_index(layer_idx) for layer_idx in normalized_layers
        ]
        common_views_match = (
            common_slot_table is not None
            and common_slot_table.dim() == 3
            and len(kv_layer_indices) == len(set(kv_layer_indices))
            and all(
                self.buffer_req_to_token_slots[layer_idx] is not None
                and tuple(self.buffer_req_to_token_slots[layer_idx].shape)
                == tuple(common_slot_table[kv_idx].shape)
                and self.buffer_req_to_token_slots[layer_idx].stride()
                == common_slot_table[kv_idx].stride()
                and self.buffer_req_to_token_slots[layer_idx].data_ptr()
                == common_slot_table[kv_idx].data_ptr()
                for layer_idx, kv_idx in zip(
                    normalized_layers,
                    kv_layer_indices,
                )
            )
        )
        can_compact_layer_tiles = (
            common_views_match
            and np.all(cur_lens == cur_len)
            and elements_per_layer <= tile_elements
        )
        if can_compact_layer_tiles:
            layers_per_tile = max(1, tile_elements // max(1, elements_per_layer))
            for start in range(0, num_layers, layers_per_tile):
                end = min(num_layers, start + layers_per_tile)
                self._compact_uniform_layer_tile(
                    normalized_layers[start:end],
                    kv_layer_indices[start:end],
                    row_indices[start:end],
                    cur_len,
                    keep_indices[start:end],
                    common_slot_table,
                )
            return

        for local_layer, layer_idx in enumerate(layer_indices):
            layer_idx = int(layer_idx)
            layer_cur_lens = cur_lens[local_layer]
            cur_len = int(layer_cur_lens[0])
            if np.all(layer_cur_lens == cur_len):
                self._compact_uniform_rows_bounded(
                    layer_idx,
                    row_indices[local_layer].tolist(),
                    cur_len,
                    keep_indices[local_layer],
                )
                continue
            for seq_idx, seq in enumerate(seqs):
                SnapKVCacheManager.free_part_slots(
                    self,
                    layer_idx,
                    seq,
                    keep_indices[local_layer, seq_idx],
                    keep_indices_sorted=True,
                )

    def free_prefix_recent_slots_batch_layers(
        self,
        layer_indices: list[int],
        seqs: list[Sequence],
        *,
        kv_len: int,
        sink_keep_tokens: int,
        recent_keep_tokens: int,
    ):
        if not layer_indices or not seqs:
            return

        for layer_idx in layer_indices:
            self.kv_layer_index(int(layer_idx))
        self._uniform_decode_metadata = False
        kv_len = int(kv_len)
        sink_end = min(int(sink_keep_tokens), kv_len)
        recent_start = max(sink_end, kv_len - int(recent_keep_tokens))
        new_len = sink_end + (kv_len - recent_start)
        if new_len <= 0:
            raise RuntimeError("prefix/recent compaction cannot keep zero tokens.")
        if new_len >= kv_len:
            return

        num_layers = len(layer_indices)
        batch_size = len(seqs)
        row_indices = np.empty((num_layers, batch_size), dtype=np.int64)
        cur_lens = np.empty((num_layers, batch_size), dtype=np.int64)
        for local_layer, layer_idx in enumerate(layer_indices):
            layer_idx = int(layer_idx)
            for seq_idx, seq in enumerate(seqs):
                row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
                if row_idx is None:
                    raise ValueError
                row_indices[local_layer, seq_idx] = int(row_idx)
                cur_lens[local_layer, seq_idx] = int(self.row_seq_lens[layer_idx][row_idx])
        if not np.all(cur_lens == kv_len):
            raise RuntimeError(
                "prefix/recent compaction expected uniform row lengths: "
                f"kv_len={kv_len} observed={cur_lens.tolist()}"
            )

        kv_layers_gpu = torch.tensor(
            [self.kv_layer_index(int(layer_idx)) for layer_idx in layer_indices],
            dtype=torch.long,
            device=self.device,
        )
        rows_gpu = torch.from_numpy(row_indices).to(device=self.device, dtype=torch.long)
        drop_cols = torch.arange(sink_end, recent_start, dtype=torch.long, device=self.device)
        dropped_per_layer = self.buffer_req_to_token_slots_tensor[
            kv_layers_gpu[:, None, None],
            rows_gpu[:, :, None],
            drop_cols[None, None, :],
        ].reshape(num_layers, -1)
        drop_count = int(dropped_per_layer.shape[1])
        if drop_count > 0:
            if self.free_slots_stack_tensor is not None:
                ptrs = np.asarray([self._num_free_slots[int(layer_idx)] for layer_idx in layer_indices], dtype=np.int64)
                offsets = ptrs[:, None] + np.arange(drop_count, dtype=np.int64)[None, :]
                self.free_slots_stack_tensor[
                    kv_layers_gpu[:, None],
                    torch.from_numpy(offsets).to(device=self.device, dtype=torch.long),
                ] = dropped_per_layer.to(torch.int32)
            else:
                for local_layer, layer_idx in enumerate(layer_indices):
                    layer_idx = int(layer_idx)
                    ptr = self._num_free_slots[layer_idx]
                    self.free_slots_stack[layer_idx][ptr: ptr + drop_count] = dropped_per_layer[local_layer]
            for layer_idx in layer_indices:
                self._num_free_slots[int(layer_idx)] += drop_count

        if recent_start < kv_len:
            recent_cols = torch.arange(recent_start, kv_len, dtype=torch.long, device=self.device)
            dst_cols = torch.arange(sink_end, new_len, dtype=torch.long, device=self.device)
            recent_slots = self.buffer_req_to_token_slots_tensor[
                kv_layers_gpu[:, None, None],
                rows_gpu[:, :, None],
                recent_cols[None, None, :],
            ]
            self.buffer_req_to_token_slots_tensor[
                kv_layers_gpu[:, None, None],
                rows_gpu[:, :, None],
                dst_cols[None, None, :],
            ] = recent_slots
        tail_cols = torch.arange(new_len, kv_len, dtype=torch.long, device=self.device)
        self.buffer_req_to_token_slots_tensor[
            kv_layers_gpu[:, None, None],
            rows_gpu[:, :, None],
            tail_cols[None, None, :],
        ] = 0
        for local_layer, layer_idx in enumerate(layer_indices):
            self.row_seq_lens[int(layer_idx)][row_indices[local_layer]] = new_len

    def materialize_prefill_staging_layer(self, layer_idx: int, seq: Sequence, keep_indices: torch.Tensor):
        self.kv_layer_index(layer_idx)
        if not self.has_prefill_staging_view(layer_idx):
            raise RuntimeError("PyramidKV prefill staging is not active.")
        materialized_key = (int(layer_idx), int(seq.seq_id))
        if materialized_key in self._pyramidkv_prefill_staging_materialized_layers:
            raise RuntimeError(
                f"PyramidKV prefill staging layer materialized twice: "
                f"layer={layer_idx} seq_id={seq.seq_id}."
            )

        row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
        if row_idx is None:
            raise RuntimeError(f"PyramidKV staging row is missing: layer={layer_idx} seq_id={seq.seq_id}.")
        resident_prefix_len = int(
            getattr(
                self,
                "_pyramidkv_long_prefill_offload_resident_prefix_lens",
                {},
            ).get(int(layer_idx), 0)
        )
        if resident_prefix_len > 0:
            if int(self.row_seq_lens[layer_idx][row_idx]) != resident_prefix_len:
                raise RuntimeError(
                    "PyramidKV raw_offload resident prefix changed before final "
                    "materialization: "
                    f"layer={layer_idx} seq_id={seq.seq_id} "
                    f"expected={resident_prefix_len} "
                    f"observed={int(self.row_seq_lens[layer_idx][row_idx])}."
                )
            old_slots = self.buffer_req_to_token_slots[layer_idx][
                row_idx,
                :resident_prefix_len,
            ].clone()
            ptr = int(self._num_free_slots[layer_idx])
            self.free_slots_stack[layer_idx][ptr : ptr + resident_prefix_len] = old_slots
            self._num_free_slots[layer_idx] += resident_prefix_len
            self.buffer_req_to_token_slots[layer_idx][row_idx, :resident_prefix_len] = 0
            self.row_seq_lens[layer_idx][row_idx] = 0
        if int(self.row_seq_lens[layer_idx][row_idx]) != 0:
            raise RuntimeError(
                "PyramidKV full-prefill staging expects an empty persistent row before materialization. "
                f"layer={layer_idx} seq_id={seq.seq_id} row_len={int(self.row_seq_lens[layer_idx][row_idx])}."
            )

        keep_indices = keep_indices.to(device=self.device, dtype=torch.long).contiguous()
        num_keep = int(keep_indices.numel())
        if num_keep <= 0:
            raise RuntimeError("PyramidKV staging materialization cannot keep zero tokens.")

        slots = self._allocate(layer_idx, seq.seq_id, num_keep).to(torch.long)
        k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        k_stage = self.pyramidkv_prefill_staging_kv_cache[0]
        v_stage = self.pyramidkv_prefill_staging_kv_cache[1]
        staging_offset = int(self._pyramidkv_prefill_staging_seq_offsets[int(seq.seq_id)])
        staging_indices = keep_indices + staging_offset
        k_cache[slots] = k_stage[staging_indices]
        v_cache[slots] = v_stage[staging_indices]

        self._pyramidkv_prefill_staging_materialized_layers.add(materialized_key)
        expected_materializations = int(self.num_kv_layers) * len(self._pyramidkv_prefill_staging_seq_offsets)
        if len(self._pyramidkv_prefill_staging_materialized_layers) == expected_materializations:
            self._pyramidkv_prefill_staging_active = False
            self._release_pyramidkv_long_prefill_offload_rows()

    def materialize_prefill_staging_layer_batch(
        self,
        layer_idx: int,
        seq_keep_indices: list[tuple[Sequence, torch.Tensor]],
    ):
        if not seq_keep_indices:
            return
        if len(seq_keep_indices) == 1:
            seq, keep_indices = seq_keep_indices[0]
            self.materialize_prefill_staging_layer(layer_idx, seq, keep_indices)
            return
        self.kv_layer_index(layer_idx)
        if not self.has_prefill_staging_view(layer_idx):
            raise RuntimeError("PyramidKV prefill staging is not active.")

        seq_ids = []
        keep_tensors = []
        keep_sizes = []
        all_staging_indices = []
        materialized_keys = []
        for seq, keep_indices in seq_keep_indices:
            materialized_key = (int(layer_idx), int(seq.seq_id))
            if materialized_key in self._pyramidkv_prefill_staging_materialized_layers:
                raise RuntimeError(
                    f"PyramidKV prefill staging layer materialized twice: "
                    f"layer={layer_idx} seq_id={seq.seq_id}."
                )

            row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
            if row_idx is None:
                raise RuntimeError(f"PyramidKV staging row is missing: layer={layer_idx} seq_id={seq.seq_id}.")
            if int(self.row_seq_lens[layer_idx][row_idx]) != 0:
                raise RuntimeError(
                    "PyramidKV full-prefill staging expects an empty persistent row before materialization. "
                    f"layer={layer_idx} seq_id={seq.seq_id} row_len={int(self.row_seq_lens[layer_idx][row_idx])}."
                )

            keep_indices = keep_indices.to(device=self.device, dtype=torch.long).contiguous()
            num_keep = int(keep_indices.numel())
            if num_keep <= 0:
                raise RuntimeError("PyramidKV staging materialization cannot keep zero tokens.")

            staging_offset = int(self._pyramidkv_prefill_staging_seq_offsets[int(seq.seq_id)])
            seq_ids.append(int(seq.seq_id))
            keep_tensors.append(keep_indices)
            keep_sizes.append(num_keep)
            all_staging_indices.append(keep_indices + staging_offset)
            materialized_keys.append(materialized_key)

        slots = torch.cat(
            [
                self._allocate(layer_idx, seq_id, num_keep).to(torch.long)
                for seq_id, num_keep in zip(seq_ids, keep_sizes)
            ],
            dim=0,
        )
        staging_indices = torch.cat(all_staging_indices, dim=0)
        k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        k_stage = self.pyramidkv_prefill_staging_kv_cache[0]
        v_stage = self.pyramidkv_prefill_staging_kv_cache[1]
        k_cache[slots] = k_stage[staging_indices]
        v_cache[slots] = v_stage[staging_indices]

        self._pyramidkv_prefill_staging_materialized_layers.update(materialized_keys)
        expected_materializations = int(self.num_kv_layers) * len(self._pyramidkv_prefill_staging_seq_offsets)
        if len(self._pyramidkv_prefill_staging_materialized_layers) == expected_materializations:
            self._pyramidkv_prefill_staging_active = False
            self._release_pyramidkv_long_prefill_offload_rows()

    def _pyramidkv_long_prefill_offload_kind(self) -> str:
        return "pyramidkv_post_rope"

    def _release_pyramidkv_long_prefill_offload_rows(self):
        if getattr(self, "_pyramidkv_long_prefill_offload_seq_id", None) is None:
            return
        self._pyramidkv_clear_long_prefill_offload_prefetch()
        seq_id = int(self._pyramidkv_long_prefill_offload_seq_id)
        seen_rows = set()
        for layer_idx in self.kv_transformer_layer_indices():
            row_idx = self.seq_id_to_row[layer_idx].get(seq_id)
            if row_idx is None:
                continue
            row_idx = int(row_idx)
            if row_idx in seen_rows:
                continue
            self.raw_kv_offload_buffer.release_row(row_idx)
            seen_rows.add(row_idx)
        self._pyramidkv_long_prefill_offload_seq_id = None

    def _pyramidkv_long_prefill_offload_row(self, layer_idx: int) -> int:
        seq_id = self._pyramidkv_long_prefill_offload_seq_id
        if seq_id is None:
            raise RuntimeError("PyramidKV long-prefill offload has no active seq_id.")
        row_idx = self.seq_id_to_row[int(layer_idx)].get(int(seq_id))
        if row_idx is None:
            raise RuntimeError(
                "PyramidKV long-prefill offload row is missing: "
                f"layer={layer_idx} seq_id={seq_id}."
            )
        return int(row_idx)

    def _pyramidkv_long_prefill_offload_prefetch_enabled(self) -> bool:
        return device_runtime.supports_streams(self.device)

    def _pyramidkv_clear_long_prefill_offload_prefetch(self):
        states = getattr(self, "_pyramidkv_long_prefill_offload_prefetch_states", None) or {}
        for state in list(states.values()):
            event = state.get("event")
            if event is not None:
                device_runtime.wait_event(event, device=self.device)
        self._pyramidkv_long_prefill_offload_prefetch_states = {}

    def _pyramidkv_drop_long_prefill_offload_prefetch(self, key: tuple[int, int, str, int]):
        states = getattr(self, "_pyramidkv_long_prefill_offload_prefetch_states", None) or {}
        state = states.pop(key, None)
        if state is not None:
            event = state.get("event")
            if event is not None:
                device_runtime.wait_event(event, device=self.device)
        self._pyramidkv_long_prefill_offload_prefetch_states = states

    def _pyramidkv_consume_long_prefill_offload_staged_prefetch(
        self,
        *,
        layer_idx: int,
        row_idx: int,
        end: int,
    ) -> bool:
        kind = self._pyramidkv_long_prefill_offload_kind()
        key = (int(layer_idx), int(row_idx), kind, int(end))
        states = getattr(self, "_pyramidkv_long_prefill_offload_prefetch_states", None) or {}
        state = states.pop(key, None)
        if state is None:
            self._pyramidkv_long_prefill_offload_prefetch_states = states
            return False
        with profiler.record("pyramidkv_long_prefill_offload_prefetch_wait"):
            device_runtime.wait_event(state["event"], device=self.device)
        self._pyramidkv_long_prefill_offload_prefetch_states = states
        return True

    def _pyramidkv_schedule_next_long_prefill_offload_prefetch(self, *, layer_idx: int, end: int):
        if int(end) <= 0 or not self._pyramidkv_long_prefill_offload_prefetch_enabled():
            return
        next_layers = [
            candidate
            for candidate in self.kv_transformer_layer_indices()
            if int(candidate) > int(layer_idx)
        ]
        if not next_layers:
            return
        next_layer = int(next_layers[0])
        row_idx = self._pyramidkv_long_prefill_offload_row(next_layer)
        resident_prefix_len = int(
            getattr(
                self,
                "_pyramidkv_long_prefill_offload_resident_prefix_lens",
                {},
            ).get(next_layer, 0)
        )
        kind = self._pyramidkv_long_prefill_offload_kind()
        key = (next_layer, int(row_idx), kind, int(end))
        states = getattr(self, "_pyramidkv_long_prefill_offload_prefetch_states", None) or {}
        keep_keys = {key}
        for old_key in list(states):
            if old_key not in keep_keys:
                self._pyramidkv_drop_long_prefill_offload_prefetch(old_key)
                states = getattr(self, "_pyramidkv_long_prefill_offload_prefetch_states", None) or {}
        if key in states:
            return

        stream = getattr(self, "_pyramidkv_long_prefill_offload_prefetch_stream", None)
        if stream is None:
            stream = device_runtime.new_stream(device=self.device)
            if stream is None:
                raise RuntimeError(
                    "PyramidKV long-prefill offload prefetch is enabled, but the active platform "
                    f"does not support streams for device={self.device}."
                )
            self._pyramidkv_long_prefill_offload_prefetch_stream = stream

        with profiler.record("pyramidkv_long_prefill_offload_prefetch_schedule"):
            staging_available_event = device_runtime.new_event(device=self.device)
            if staging_available_event is None:
                raise RuntimeError(
                    "PyramidKV long-prefill offload prefetch could not create a staging event "
                    f"for device={self.device}."
                )
            device_runtime.record_event(staging_available_event, device=self.device)
            with device_runtime.stream_context(stream):
                device_runtime.stream_wait_event(stream, staging_available_event)
                self.raw_kv_offload_buffer.copy_prefix_to(
                    layer_idx=next_layer,
                    row_idx=row_idx,
                    kind=kind,
                    end=end,
                    k_out=self.pyramidkv_prefill_staging_kv_cache[
                        0,
                        resident_prefix_len : resident_prefix_len + end,
                    ],
                    v_out=self.pyramidkv_prefill_staging_kv_cache[
                        1,
                        resident_prefix_len : resident_prefix_len + end,
                    ],
                )
                event = device_runtime.new_event(device=self.device)
                if event is None:
                    raise RuntimeError(
                        "PyramidKV long-prefill offload prefetch could not create a completion event "
                        f"for device={self.device}."
                    )
                device_runtime.record_event(event, device=self.device)
        states[key] = {
            "layer_idx": next_layer,
            "row_idx": int(row_idx),
            "kind": kind,
            "end": int(end),
            "staging_available_event": staging_available_event,
            "event": event,
        }
        self._pyramidkv_long_prefill_offload_prefetch_states = states

    def _pyramidkv_schedule_post_layer_long_prefill_offload_prefetch(self, layer_idx: int):
        if not bool(getattr(self, "_pyramidkv_long_prefill_offload_step_active", False)):
            return
        start = int(getattr(self, "_pyramidkv_long_prefill_offload_start", 0) or 0)
        residual_start = int(
            getattr(self, "_pyramidkv_long_prefill_offload_residual_start", 0) or 0
        )
        restored_residual = start - residual_start
        if restored_residual <= 0:
            return
        with profiler.record("pyramidkv_long_prefill_offload_after_attention_prefetch"):
            self._pyramidkv_schedule_next_long_prefill_offload_prefetch(
                layer_idx=layer_idx,
                end=restored_residual,
            )

    @torch.no_grad()
    def before_prefill_layer_attention(self, layer_idx: int, selection: SparseSelection):
        del selection
        if not bool(getattr(self, "_pyramidkv_long_prefill_offload_step_active", False)):
            return None
        if not self.has_prefill_staging_view(layer_idx):
            return None
        start = int(getattr(self, "_pyramidkv_long_prefill_offload_start", 0) or 0)
        residual_start = int(
            getattr(self, "_pyramidkv_long_prefill_offload_residual_start", 0) or 0
        )
        restored_residual = start - residual_start
        row_idx = self._pyramidkv_long_prefill_offload_row(layer_idx)
        resident_prefix_len = int(
            getattr(
                self,
                "_pyramidkv_long_prefill_offload_resident_prefix_lens",
                {},
            ).get(int(layer_idx), 0)
        )
        if resident_prefix_len > 0:
            slots = self.buffer_req_to_token_slots[int(layer_idx)][
                row_idx,
                :resident_prefix_len,
            ].to(torch.long)
            k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
            self.pyramidkv_prefill_staging_kv_cache[
                0,
                :resident_prefix_len,
            ].copy_(k_cache[slots])
            self.pyramidkv_prefill_staging_kv_cache[
                1,
                :resident_prefix_len,
            ].copy_(v_cache[slots])
        if restored_residual <= 0:
            return None
        with profiler.record("pyramidkv_long_prefill_offload_wait_or_restore"):
            staged = self._pyramidkv_consume_long_prefill_offload_staged_prefetch(
                layer_idx=layer_idx,
                row_idx=row_idx,
                end=restored_residual,
            )
            if staged:
                return None
        with profiler.record("pyramidkv_long_prefill_offload_restore_prefix"):
            self.raw_kv_offload_buffer.copy_prefix_to(
                layer_idx=layer_idx,
                row_idx=row_idx,
                kind=self._pyramidkv_long_prefill_offload_kind(),
                end=restored_residual,
                k_out=self.pyramidkv_prefill_staging_kv_cache[
                    0,
                    resident_prefix_len : resident_prefix_len + restored_residual,
                ],
                v_out=self.pyramidkv_prefill_staging_kv_cache[
                    1,
                    resident_prefix_len : resident_prefix_len + restored_residual,
                ],
            )
        return None

    @torch.no_grad()
    def _offload_pyramidkv_long_prefill_layer(self, layer_idx: int):
        start = int(getattr(self, "_pyramidkv_long_prefill_offload_start", 0) or 0)
        end = int(getattr(self, "_pyramidkv_long_prefill_offload_end", 0) or 0)
        total_len = int(getattr(self, "_pyramidkv_long_prefill_offload_total_len", 0) or 0)
        residual_start = int(
            getattr(self, "_pyramidkv_long_prefill_offload_residual_start", 0) or 0
        )
        if end <= start:
            raise RuntimeError(
                "PyramidKV long-prefill offload has invalid range: "
                f"layer={layer_idx} start={start} end={end}."
            )
        row_idx = self._pyramidkv_long_prefill_offload_row(layer_idx)
        resident_prefix_len = int(
            getattr(
                self,
                "_pyramidkv_long_prefill_offload_resident_prefix_lens",
                {},
            ).get(int(layer_idx), 0)
        )
        offload_start = start - residual_start
        offload_end = end - residual_start
        staging_start = resident_prefix_len + offload_start
        staging_end = resident_prefix_len + offload_end
        k = self.pyramidkv_prefill_staging_kv_cache[0, staging_start:staging_end]
        v = self.pyramidkv_prefill_staging_kv_cache[1, staging_start:staging_end]
        kind = self._pyramidkv_long_prefill_offload_kind()
        with profiler.record("pyramidkv_long_prefill_offload_ensure_entry"):
            self.raw_kv_offload_buffer.ensure_entry(
                layer_idx=layer_idx,
                row_idx=row_idx,
                kind=kind,
                total_len=total_len - residual_start,
                k_shape_tail=tuple(k.shape[1:]),
                v_shape_tail=tuple(v.shape[1:]),
                dtype=k.dtype,
            )
        with profiler.record("pyramidkv_long_prefill_offload_put_range"):
            self.raw_kv_offload_buffer.put_range(
                layer_idx=layer_idx,
                row_idx=row_idx,
                kind=kind,
                start=offload_start,
                k=k,
                v=v,
            )

    def on_layer_attention_end(self, layer_idx: int):
        if not self.has_prefill_staging_view(layer_idx):
            return
        if not bool(getattr(self, "_pyramidkv_long_prefill_offload_step_active", False)):
            return
        if bool(getattr(self, "_pyramidkv_long_prefill_offload_is_last_chunk", False)):
            if not bool(getattr(self, "_pyramidkv_prefill_staging_active", False)):
                self._release_pyramidkv_long_prefill_offload_rows()
            return
        self._offload_pyramidkv_long_prefill_layer(layer_idx)
        self._pyramidkv_schedule_post_layer_long_prefill_offload_prefetch(layer_idx)

    def prepare_step(self, seqs: list[Sequence], is_prefill: bool):
        self._pyramidkv_reset_full_prefill_staging()
        self._pyramidkv_long_prefill_offload_step_active = bool(
            is_prefill and self._should_use_pyramidkv_long_prefill_offload_staging(seqs)
        )
        return super().prepare_step(seqs, is_prefill)

    def _prepare_prefill(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_prefill"):
            self._decode_static_state_binding_key = None
            layer_ids = self.kv_transformer_layer_indices()
            self._pyramidkv_prefill_staging_slot_mapping_by_layer = {}
            self._pyramidkv_prefill_staging_active_slots_by_layer = {}
            self._pyramidkv_prefill_staging_context_lens_by_layer = {}
            self._pyramidkv_prefill_staging_context_lens_cpu_by_layer = {}
            self._prefill_context_lens_cpu_by_layer = {}
            self._prefill_score_metadata_cache = {}
            self._pyramidkv_long_prefill_offload_resident_prefix_lens = {}
            for seq in seqs:
                starts_resumed_turn = (
                    getattr(seq, "chain_status", "") == "resumed"
                    and not bool(getattr(seq, "is_recompute_replay", False))
                    and int(seq.num_prefilled_tokens)
                    == int(getattr(seq, "chain_reused_tokens", 0) or 0)
                )
                if int(seq.num_prefilled_tokens) == 0 or starts_resumed_turn:
                    self._clear_prefill_attention_scores(seq.seq_id)

            use_long_prefill_offload_staging = self._should_use_pyramidkv_long_prefill_offload_staging(seqs)
            use_full_prefill_staging = (
                self._should_use_pyramidkv_full_prefill_staging(seqs)
                or use_long_prefill_offload_staging
            )
            total_chunk_tokens = sum(seq.current_chunk_size for seq in seqs)
            if use_full_prefill_staging and total_chunk_tokens > int(self.pyramidkv_prefill_staging_num_slots):
                raise RuntimeError(
                    "PyramidKV full-prefill staging capacity is too small for this step. "
                    f"tokens={total_chunk_tokens} staging_slots={self.pyramidkv_prefill_staging_num_slots}."
                )

            input_ids_np = np.empty(total_chunk_tokens, dtype=np.int64)
            positions_np = np.empty(total_chunk_tokens, dtype=np.int64)
            cu_seqlens_q = [0]

            if use_full_prefill_staging:
                layers_slot_mapping_cuda = torch.empty(
                    (self.num_layers, total_chunk_tokens),
                    dtype=torch.int32,
                    device=self.device,
                )
            else:
                layers_slot_mapping_cuda = torch.empty(
                    (self.num_layers, total_chunk_tokens), dtype=torch.int32, device=self.device
                )
            context_lens_list = [[] for _ in range(self.num_layers)]

            use_batched_prefill_alloc = (
                not use_full_prefill_staging
                and self._allocate_prefill_batch_same_size_all_layers(seqs, layers_slot_mapping_cuda)
            )
            if use_batched_prefill_alloc:
                for layer_id in layer_ids:
                    context_lens_list[layer_id] = [
                        int(
                            self.row_seq_lens[layer_id][
                                self.seq_id_to_row[layer_id][
                                    int(seq.seq_id)
                                ]
                            ]
                            if getattr(seq, "chain_status", "") == "resumed"
                            and not bool(getattr(seq, "is_recompute_replay", False))
                            else int(seq.num_prefilled_tokens)
                            + int(seq.current_chunk_size)
                        )
                        for seq in seqs
                    ]

            token_offset = 0
            for seq in seqs:
                chunk_size = seq.current_chunk_size
                start_idx = seq.num_prefilled_tokens
                end_idx = start_idx + chunk_size
                residual_start = 0
                if use_long_prefill_offload_staging and not bool(
                    getattr(seq, "is_recompute_replay", False)
                ):
                    residual_start = max(
                        int(getattr(seq, "chain_reused_tokens", 0) or 0),
                        int(getattr(seq, "prefix_cache_hit_len", 0) or 0),
                    )
                residual_progress = int(start_idx) - int(residual_start)
                if residual_progress < 0:
                    raise RuntimeError(
                        "PyramidKV raw_offload residual progress is negative: "
                        f"seq_id={seq.seq_id} start={start_idx} "
                        f"residual_start={residual_start}."
                    )

                if not use_batched_prefill_alloc:
                    for layer_id in layer_ids:
                        resident_prefix_len = 0
                        if use_long_prefill_offload_staging and residual_start > 0:
                            row_idx = self.seq_id_to_row[layer_id].get(seq.seq_id)
                            if row_idx is None:
                                raise RuntimeError(
                                    "PyramidKV raw_offload resumed prefix row is missing: "
                                    f"layer={layer_id} seq_id={seq.seq_id}."
                                )
                            resident_prefix_len = int(
                                self.row_seq_lens[layer_id][int(row_idx)]
                            )
                        if seq.seq_id in self.seq_id_to_row[layer_id]:
                            row_idx = self.seq_id_to_row[layer_id][seq.seq_id]
                            is_chain_resume = (
                                getattr(seq, "chain_status", "") == "resumed"
                                and not bool(
                                    getattr(seq, "is_recompute_replay", False)
                                )
                            )
                            expected_row_len = (
                                resident_prefix_len
                                if use_long_prefill_offload_staging
                                else start_idx
                            )
                            if (
                                not is_chain_resume
                                and self.row_seq_lens[layer_id][row_idx]
                                != expected_row_len
                            ):
                                raise ValueError(
                                    "KV cache row length mismatch in prefill: "
                                    f"layer={layer_id} seq_id={seq.seq_id} "
                                    f"row_seq_len={self.row_seq_lens[layer_id][row_idx]} "
                                    f"expected={expected_row_len} start_idx={start_idx}"
                                )
                        if use_full_prefill_staging:
                            if start_idx != 0 and not use_long_prefill_offload_staging:
                                raise RuntimeError("PyramidKV full-prefill staging only supports first-prefill prompts.")
                            self._get_free_row(layer_id, seq.seq_id)
                        else:
                            self._allocate(layer_id, seq.seq_id, chunk_size)
                        row_idx = self.seq_id_to_row[layer_id][seq.seq_id]
                        if use_full_prefill_staging:
                            staging_start = (
                                resident_prefix_len + residual_progress
                                if use_long_prefill_offload_staging
                                else token_offset
                            )
                            staging_end = staging_start + int(chunk_size)
                            if staging_end > int(self.pyramidkv_prefill_staging_num_slots):
                                raise RuntimeError(
                                    "PyramidKV raw_offload staging capacity is too small "
                                    "for the attached prefix and residual chunk: "
                                    f"layer={layer_id} seq_id={seq.seq_id} "
                                    f"required={staging_end} "
                                    f"staging_slots={self.pyramidkv_prefill_staging_num_slots}."
                                )
                            layers_slot_mapping_cuda[
                                layer_id,
                                token_offset : token_offset + chunk_size,
                            ] = torch.arange(
                                staging_start,
                                staging_end,
                                dtype=torch.int32,
                                device=self.device,
                            )
                            context_len = (
                                resident_prefix_len + residual_progress + chunk_size
                                if use_long_prefill_offload_staging
                                else chunk_size
                            )
                            if use_long_prefill_offload_staging:
                                self._pyramidkv_long_prefill_offload_resident_prefix_lens[
                                    int(layer_id)
                                ] = resident_prefix_len
                        else:
                            physical_end = int(
                                self.row_seq_lens[layer_id][row_idx]
                            )
                            physical_start = physical_end - int(chunk_size)
                            layers_slot_mapping_cuda[layer_id, token_offset: token_offset + chunk_size] = \
                                self.buffer_req_to_token_slots[layer_id][
                                    row_idx, physical_start:physical_end
                                ]
                            context_len = physical_end
                        context_lens_list[layer_id].append(context_len)

                chunk_tokens = seq.token_ids
                if len(chunk_tokens) > chunk_size:
                    chunk_tokens = chunk_tokens[start_idx:end_idx]

                input_ids_np[token_offset: token_offset + chunk_size] = chunk_tokens
                positions_np[token_offset: token_offset + chunk_size] = np.arange(start_idx, end_idx)

                cu_seqlens_q.append(cu_seqlens_q[-1] + chunk_size)
                token_offset += chunk_size

            layers_context_lens_np = np.zeros((self.num_layers, len(seqs)), dtype=np.int32)
            for layer_id in layer_ids:
                layers_context_lens_np[layer_id] = context_lens_list[layer_id]
                self._prefill_context_lens_cpu_by_layer[int(layer_id)] = tuple(
                    int(value) for value in context_lens_list[layer_id]
                )
            layers_context_lens_cuda = torch.from_numpy(layers_context_lens_np).to(
                device=self.device,
                dtype=torch.int32,
            )

            for layer_id in layer_ids:
                state = self.layer_batch_states[layer_id]
                state.slot_mapping = layers_slot_mapping_cuda[layer_id]
                state.context_lens = layers_context_lens_cuda[layer_id]
                state.max_context_len = int(max(context_lens_list[layer_id])) if context_lens_list[layer_id] else 0
                req_ids = [self.seq_id_to_row[layer_id][seq.seq_id] for seq in seqs]
                state.req_indices = torch.tensor(req_ids, dtype=torch.int32, device=self.device)

            if use_full_prefill_staging:
                self._pyramidkv_prefill_staging_active = True
                self._pyramidkv_prefill_staging_was_active = True
                self._pyramidkv_prefill_staging_slot_mapping = layers_slot_mapping_cuda[0]
                self._pyramidkv_prefill_staging_slot_mapping_by_layer = {
                    int(layer_id): layers_slot_mapping_cuda[int(layer_id)]
                    for layer_id in layer_ids
                }
                self._pyramidkv_prefill_staging_req_indices = torch.arange(
                    len(seqs),
                    dtype=torch.int32,
                    device=self.device,
                )
                for layer_id in layer_ids:
                    layer_context_lens = [int(value) for value in context_lens_list[layer_id]]
                    max_context_len = max(layer_context_lens)
                    active_slots = torch.full(
                        (len(seqs), max_context_len),
                        -1,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    offset = 0
                    for b_idx, (seq, visible_len) in enumerate(
                        zip(seqs, layer_context_lens)
                    ):
                        slot_start = 0 if use_long_prefill_offload_staging else offset
                        self._pyramidkv_prefill_staging_seq_offsets[
                            int(seq.seq_id)
                        ] = int(slot_start)
                        active_slots[b_idx, :visible_len] = torch.arange(
                            slot_start,
                            slot_start + visible_len,
                            dtype=torch.int32,
                            device=self.device,
                        )
                        offset += int(seq.current_chunk_size)
                    context_lens = torch.tensor(
                        layer_context_lens,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    self._pyramidkv_prefill_staging_active_slots_by_layer[
                        int(layer_id)
                    ] = active_slots
                    self._pyramidkv_prefill_staging_context_lens_by_layer[
                        int(layer_id)
                    ] = context_lens
                    self._pyramidkv_prefill_staging_context_lens_cpu_by_layer[
                        int(layer_id)
                    ] = tuple(layer_context_lens)
                first_layer = int(layer_ids[0])
                self._pyramidkv_prefill_staging_active_slots = (
                    self._pyramidkv_prefill_staging_active_slots_by_layer[first_layer]
                )
                self._pyramidkv_prefill_staging_context_lens = (
                    self._pyramidkv_prefill_staging_context_lens_by_layer[first_layer]
                )
                if use_long_prefill_offload_staging:
                    seq = seqs[0]
                    self._pyramidkv_long_prefill_offload_seq_id = int(seq.seq_id)
                    self._pyramidkv_long_prefill_offload_start = int(seq.num_prefilled_tokens)
                    self._pyramidkv_long_prefill_offload_end = int(seq.num_prefilled_tokens + seq.current_chunk_size)
                    self._pyramidkv_long_prefill_offload_total_len = int(seq.num_prompt_tokens)
                    self._pyramidkv_long_prefill_offload_residual_start = max(
                        0
                        if bool(getattr(seq, "is_recompute_replay", False))
                        else int(getattr(seq, "chain_reused_tokens", 0) or 0),
                        0
                        if bool(getattr(seq, "is_recompute_replay", False))
                        else int(getattr(seq, "prefix_cache_hit_len", 0) or 0),
                    )
                    self._pyramidkv_long_prefill_offload_is_last_chunk = bool(seq.is_last_chunk_prefill)

            input_ids = torch.from_numpy(input_ids_np).to(self.device)
            positions = torch.from_numpy(positions_np).to(self.device)
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=self.device)
            return input_ids, positions, cu_seqlens_q

    def _prepare_decode(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_decode"):
            self._decode_static_state_binding_key = None
            layer_ids = self.kv_transformer_layer_indices()
            batch_size = len(seqs)
            self._ensure_decode_buffers(batch_size)

            input_ids_list = [seq.decode_input_token for seq in seqs]
            positions_list = [seq.decode_input_position for seq in seqs]
            seq_ids = [seq.seq_id for seq in seqs]

            layers_slot_mapping_cuda = self._cuda_layers_slot_mapping[:, :batch_size]

            rows_gpu = self._static_rows_gpu[:batch_size]
            cols_gpu = self._static_cols_gpu[:batch_size]

            can_use_uniform_tensor_fast_path = (
                self._uniform_decode_metadata
                and self.free_slots_stack_tensor is not None
                and self.buffer_req_to_token_slots_tensor is not None
                and int(self.num_layers) == int(self.num_kv_layers)
                and tuple(layer_ids) == tuple(range(int(self.num_layers)))
            )
            if can_use_uniform_tensor_fast_path:
                first_layer = layer_ids[0]
                row_indices = [self._get_free_row(first_layer, sid) for sid in seq_ids]
                cur_lens = self.row_seq_lens[first_layer][row_indices]
                ptr = self._num_free_slots[first_layer]
                assert ptr >= batch_size, f"Out of KV slots: need {batch_size}, free {ptr}"

                select_indices_3d = self.free_slots_stack_tensor[:, ptr - batch_size: ptr]
                for l in layer_ids:
                    self._num_free_slots[l] -= batch_size

                rows_gpu.copy_(torch.as_tensor(row_indices, dtype=torch.long), non_blocking=True)
                cols_gpu.copy_(torch.as_tensor(cur_lens, dtype=torch.long), non_blocking=True)

                self.buffer_req_to_token_slots_tensor[:, rows_gpu, cols_gpu] = select_indices_3d
                for l in layer_ids:
                    self.row_seq_lens[l][row_indices] += 1

                layers_slot_mapping_cuda.copy_(select_indices_3d, non_blocking=True)
                for l in layer_ids:
                    self._pinned_layers_context_lens[l, :batch_size] = torch.as_tensor(self.row_seq_lens[l][row_indices], dtype=torch.int32)
                    self._pinned_layers_req_indices[l, :batch_size] = torch.as_tensor(row_indices, dtype=torch.int32)
            else:
                for layer_id in layer_ids:
                    assert self._num_free_slots[layer_id] >= batch_size
                    row_indices = [self._get_free_row(layer_id, sid) for sid in seq_ids]
                    cur_lens = self.row_seq_lens[layer_id][row_indices]

                    ptr = self._num_free_slots[layer_id]
                    select_indices = self.free_slots_stack[layer_id][ptr - batch_size: ptr]
                    self._num_free_slots[layer_id] -= batch_size

                    rows_gpu.copy_(torch.as_tensor(row_indices, dtype=torch.long), non_blocking=True)
                    cols_gpu.copy_(torch.as_tensor(cur_lens, dtype=torch.long), non_blocking=True)
                    self.buffer_req_to_token_slots[layer_id][rows_gpu, cols_gpu] = select_indices.to(torch.int32)
                    self.row_seq_lens[layer_id][row_indices] += 1

                    layers_slot_mapping_cuda[layer_id].copy_(select_indices, non_blocking=True)
                    self._pinned_layers_context_lens[layer_id, :batch_size] = torch.as_tensor(self.row_seq_lens[layer_id][row_indices], dtype=torch.int32)
                    self._pinned_layers_req_indices[layer_id, :batch_size] = torch.as_tensor(row_indices, dtype=torch.int32)

            self._cuda_layers_context_lens[:, :batch_size].copy_(self._pinned_layers_context_lens[:, :batch_size], non_blocking=True)
            self._cuda_layers_req_indices[:, :batch_size].copy_(self._pinned_layers_req_indices[:, :batch_size], non_blocking=True)

            for layer_id in layer_ids:
                state = self.layer_batch_states[layer_id]
                state.slot_mapping = layers_slot_mapping_cuda[layer_id]
                state.context_lens = self._cuda_layers_context_lens[layer_id, :batch_size]
                lens_layer = self._pinned_layers_context_lens[layer_id, :batch_size]
                state.max_context_len = int(lens_layer.max().item()) if batch_size > 0 else 0
                state.req_indices = self._cuda_layers_req_indices[layer_id, :batch_size]

            self._pinned_input_ids[:batch_size].copy_(torch.as_tensor(input_ids_list, dtype=torch.int64))
            self._pinned_positions[:batch_size].copy_(torch.as_tensor(positions_list, dtype=torch.int64))
            self._cuda_input_ids[:batch_size].copy_(self._pinned_input_ids[:batch_size], non_blocking=True)
            self._cuda_positions[:batch_size].copy_(self._pinned_positions[:batch_size], non_blocking=True)

            return self._cuda_input_ids[:batch_size], self._cuda_positions[:batch_size], None

    def _get_decode_static_buffers(
        self,
        graph_batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        graph_batch_size = int(graph_batch_size)
        buffers = self._decode_static_buffers.get(graph_batch_size)
        if buffers is None:
            buffers = (
                torch.empty((self.num_layers, graph_batch_size), dtype=torch.int32, device=self.device),
                torch.empty((self.num_layers, graph_batch_size), dtype=torch.int32, device=self.device),
                torch.empty((self.num_layers, graph_batch_size), dtype=torch.int32, device=self.device),
            )
            self._decode_static_buffers[graph_batch_size] = buffers
        return buffers

    def _get_decode_static_index_buffers(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(batch_size)
        buffers = self._decode_static_index_buffers.get(batch_size)
        if buffers is None:
            buffers = (
                torch.empty((self.num_layers, batch_size), dtype=torch.long, device=self.device),
                torch.empty((self.num_layers, batch_size), dtype=torch.long, device=self.device),
                torch.empty((self.num_layers, batch_size), dtype=torch.long, device=self.device),
            )
            self._decode_static_index_buffers[batch_size] = buffers
        return buffers

    def _bind_decode_static_layer_states(
        self,
        graph_batch_size: int,
        layers_slot_mapping: torch.Tensor,
        layers_context_lens: torch.Tensor,
        layers_req_indices: torch.Tensor,
        max_context_lens: np.ndarray,
    ) -> None:
        binding_key = (
            int(graph_batch_size),
            int(layers_slot_mapping.data_ptr()),
            int(layers_context_lens.data_ptr()),
            int(layers_req_indices.data_ptr()),
        )
        if self._decode_static_state_binding_key != binding_key:
            for layer_id in self.kv_transformer_layer_indices():
                state = self.layer_batch_states[layer_id]
                state.slot_mapping = layers_slot_mapping[layer_id]
                state.context_lens = layers_context_lens[layer_id]
                state.req_indices = layers_req_indices[layer_id]
            self._decode_static_state_binding_key = binding_key
        for layer_id in self.kv_transformer_layer_indices():
            self.layer_batch_states[layer_id].max_context_len = int(max_context_lens[layer_id])

    def _prepare_decode_static_uniform(
        self,
        seqs: list[Sequence],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        req_indices: torch.Tensor,
    ):
        real_batch_size = len(seqs)
        graph_batch_size = int(input_ids.numel())
        input_ids_list = [seq.decode_input_token for seq in seqs]
        positions_list = [seq.decode_input_position for seq in seqs]
        seq_ids = [seq.seq_id for seq in seqs]
        layer_ids = self.kv_transformer_layer_indices()
        first_layer = int(layer_ids[0])
        row_indices = [self.seq_id_to_row[first_layer][sid] for sid in seq_ids]
        if self.validate_runtime_invariants:
            first_row_lens = tuple(
                int(self.row_seq_lens[first_layer][row_idx])
                for row_idx in row_indices
            )
            for layer_id in layer_ids[1:]:
                layer_rows = tuple(
                    int(self.seq_id_to_row[layer_id].get(sid, -1))
                    for sid in seq_ids
                )
                if layer_rows != tuple(row_indices):
                    raise RuntimeError(
                        "Uniform static decode requires identical request rows across KV layers: "
                        f"first_layer={first_layer} rows={tuple(row_indices)} "
                        f"layer={layer_id} layer_rows={layer_rows}."
                    )
                layer_row_lens = tuple(
                    int(self.row_seq_lens[layer_id][row_idx])
                    for row_idx in layer_rows
                )
                if layer_row_lens != first_row_lens:
                    raise RuntimeError(
                        "Uniform static decode requires identical row lengths across KV layers: "
                        f"first_layer={first_layer} lengths={first_row_lens} "
                        f"layer={layer_id} layer_lengths={layer_row_lens}."
                    )

        cur_lens = self.row_seq_lens[first_layer][row_indices]
        if len(cur_lens) > 0 and int(max(cur_lens)) + 1 > int(self.max_model_len):
            raise RuntimeError(
                "KV row length exceeds max_model_len in uniform prepare_decode_static: "
                f"max_cur_len={int(max(cur_lens))} max_model_len={int(self.max_model_len)}"
            )
        if any(int(self._num_free_slots[layer_id]) < real_batch_size for layer_id in layer_ids):
            raise RuntimeError(
                "Out of KV cache slots in uniform prepare_decode_static: "
                f"need={real_batch_size} free={min(int(self._num_free_slots[layer_id]) for layer_id in layer_ids)}"
            )

        if self.validate_runtime_invariants:
            free_ptrs = tuple(
                int(self._num_free_slots[layer_id]) for layer_id in layer_ids
            )
            if any(ptr != free_ptrs[0] for ptr in free_ptrs[1:]):
                raise RuntimeError(
                    "Uniform static decode requires aligned per-layer free-stack pointers: "
                    f"layers={layer_ids} ptrs={free_ptrs}."
                )

        ptr = int(self._num_free_slots[first_layer])
        new_slots_batch = self.free_slots_stack[first_layer][ptr - real_batch_size : ptr]
        for layer_id in layer_ids:
            self._num_free_slots[layer_id] -= real_batch_size

        rows_gpu = torch.tensor(row_indices, dtype=torch.long, device=self.device)
        cols_gpu = torch.tensor(cur_lens, dtype=torch.long, device=self.device)
        kv_layers_gpu = torch.tensor(
            [self.kv_layer_index(layer_id) for layer_id in layer_ids],
            dtype=torch.long,
            device=self.device,
        )
        self.buffer_req_to_token_slots_tensor[
            kv_layers_gpu[:, None],
            rows_gpu[None, :],
            cols_gpu[None, :],
        ] = new_slots_batch.to(torch.int32).unsqueeze(0)
        for layer_id in layer_ids:
            self.row_seq_lens[layer_id][row_indices] += 1
        real_context_lens = self.row_seq_lens[first_layer][row_indices]
        real_max_context_len = int(max(real_context_lens)) if row_indices else 0

        input_ids[:real_batch_size].copy_(torch.tensor(input_ids_list, dtype=torch.int64, device=self.device))
        positions[:real_batch_size].copy_(torch.tensor(positions_list, dtype=torch.int64, device=self.device))
        slot_mapping[:real_batch_size].copy_(new_slots_batch)
        context_lens[:real_batch_size].copy_(torch.tensor(real_context_lens, dtype=torch.int32, device=self.device))
        req_indices[:real_batch_size].copy_(torch.tensor(row_indices, dtype=torch.int32, device=self.device))
        if graph_batch_size > real_batch_size:
            input_ids[real_batch_size:].fill_(int(input_ids_list[0]))
            positions[real_batch_size:].fill_(int(positions_list[0]))
            slot_mapping[real_batch_size:].fill_(-1)
            context_lens[real_batch_size:].fill_(int(real_context_lens[0]))
            req_indices[real_batch_size:].fill_(int(row_indices[0]))

        for layer_id in layer_ids:
            state = self.layer_batch_states[layer_id]
            state.slot_mapping = slot_mapping
            state.context_lens = context_lens
            state.max_context_len = real_max_context_len
            state.req_indices = req_indices
        self._decode_static_state_binding_key = None
        self.validate_decode_cuda_graph_slot_mappings()
        return input_ids, positions, None

    def _allocate_decode_batch_all_layers(
        self,
        seq_ids: list[int],
        *,
        static_cap: int | None = None,
        slot_output: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray, bool]:
        batch_size = len(seq_ids)
        layer_ids = self.kv_transformer_layer_indices()
        row_indices = np.zeros((self.num_layers, batch_size), dtype=np.int64)
        cur_lens = np.zeros((self.num_layers, batch_size), dtype=np.int64)
        for layer_id in layer_ids:
            rows = [self._get_free_row(layer_id, sid) for sid in seq_ids]
            row_indices[layer_id] = rows
            cur_lens[layer_id] = self.row_seq_lens[layer_id][rows]

        active_cur_lens = cur_lens[list(layer_ids)] if layer_ids else cur_lens
        max_cur_len = int(active_cur_lens.max()) if active_cur_lens.size else 0
        if max_cur_len + 1 > int(self.max_model_len):
            raise RuntimeError(
                "KV row length exceeds max_model_len in batched static decode allocation: "
                f"max_cur_len={max_cur_len} max_model_len={int(self.max_model_len)}"
            )
        next_lens = cur_lens + 1
        if static_cap is not None:
            max_context_lens = next_lens.max(axis=1) if next_lens.size else np.zeros(self.num_layers)
            too_long = np.nonzero(max_context_lens > int(static_cap))[0]
            too_long = np.asarray([layer_id for layer_id in too_long if layer_id in set(layer_ids)], dtype=np.int64)
            if too_long.size > 0:
                layer_id = int(too_long[0])
                raise RuntimeError(
                    "static decode context length exceeds captured graph max_context_len: "
                    f"layer={layer_id} real_max_context_len={int(max_context_lens[layer_id])} "
                    f"static_cap={int(static_cap)}"
                )
        min_free = min(int(self._num_free_slots[layer_id]) for layer_id in layer_ids)
        if min_free < batch_size:
            raise RuntimeError(
                "Out of KV cache slots in batched static decode allocation: "
                f"need={batch_size} free={min_free}"
            )

        layer_index = torch.tensor(layer_ids, dtype=torch.long, device=self.device)
        kv_layer_indices = torch.tensor(
            [self.kv_layer_index(layer_id) for layer_id in layer_ids],
            dtype=torch.long,
            device=self.device,
        )
        if self.free_slots_stack_tensor is not None:
            ptrs = np.asarray([self._num_free_slots[layer_id] for layer_id in layer_ids], dtype=np.int64)
            slot_offsets = ptrs[:, None] - batch_size + np.arange(batch_size, dtype=np.int64)[None, :]
            slot_offsets_active = torch.from_numpy(slot_offsets).to(device=self.device, dtype=torch.long)
            # Index both dimensions together: selecting layers first copies the
            # entire cache-capacity stack to retrieve only one slot per request.
            selected_active = self.free_slots_stack_tensor[
                kv_layer_indices[:, None], slot_offsets_active
            ]
            if slot_output is not None:
                selected_slots = slot_output
                active_rows = selected_slots.index_select(0, layer_index)
                active_rows[:, :batch_size] = selected_active
                selected_slots.index_copy_(0, layer_index, active_rows)
                wrote_slot_output = True
            else:
                selected_slots = torch.empty((self.num_layers, batch_size), dtype=torch.int32, device=self.device)
                selected_slots.fill_(-1)
                selected_slots.index_copy_(0, layer_index, selected_active)
                wrote_slot_output = False
            for layer_id in layer_ids:
                self._num_free_slots[layer_id] -= batch_size
        else:
            _slot_offsets_gpu, rows_gpu, cols_gpu = self._get_decode_static_index_buffers(batch_size)
            if slot_output is not None:
                selected_slots = slot_output
                wrote_slot_output = True
            else:
                selected_slots = torch.empty((self.num_layers, batch_size), dtype=torch.int32, device=self.device)
                selected_slots.fill_(-1)
                wrote_slot_output = False
            for layer_id in layer_ids:
                ptr = int(self._num_free_slots[layer_id])
                selected_slots[layer_id, :batch_size].copy_(self.free_slots_stack[layer_id][ptr - batch_size: ptr])
                self._num_free_slots[layer_id] -= batch_size

        rows_active = torch.from_numpy(row_indices[list(layer_ids)]).to(device=self.device, dtype=torch.long)
        cols_active = torch.from_numpy(cur_lens[list(layer_ids)]).to(device=self.device, dtype=torch.long)
        selected_active = selected_slots.index_select(0, layer_index)[:, :batch_size]
        self.buffer_req_to_token_slots_tensor[
            kv_layer_indices[:, None],
            rows_active,
            cols_active,
        ] = selected_active

        for layer_id in layer_ids:
            self.row_seq_lens[layer_id][row_indices[layer_id]] += 1
        return selected_slots, next_lens, row_indices, wrote_slot_output

    def init_decode_graph_state(
        self,
        contract,
        inputs,
    ) -> CacheDecodeGraphState:
        inputs.validate(contract)
        if (
            self.free_slots_stack_tensor is None
            or not self._decode_graph_metadata_always_uniform()
        ):
            return CacheDecodeGraphState(contract=contract, inputs=inputs)

        num_layers = int(self.num_layers)
        batch_capacity = int(contract.batch_capacity)
        storage_elements = num_layers + 1
        pin_memory = bool(inputs.host.input_ids.is_pinned())
        host_storage = torch.empty(
            storage_elements,
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        device_storage = torch.empty(
            storage_elements,
            dtype=torch.int32,
            device=self.device,
        )

        layer_free_starts = device_storage[:num_layers]
        active_count = device_storage[-1:]
        host_layer_free_starts = host_storage[:num_layers]
        host_active_count = host_storage[-1:]
        return SnapKVDecodeGraphState(
            contract=contract,
            inputs=inputs,
            layer_slot_mapping=torch.empty(
                (num_layers, batch_capacity),
                dtype=torch.int32,
                device=self.device,
            ),
            layer_free_starts=layer_free_starts,
            active_count=active_count,
            layer_fact_storage=device_storage,
            host_layer_free_starts=host_layer_free_starts,
            host_active_count=host_active_count,
            host_layer_fact_storage=host_storage,
            transformer_layer_ids=torch.tensor(
                self.kv_transformer_layer_indices(),
                dtype=torch.int32,
                device=self.device,
            ),
        )

    def _prepare_uniform_decode_graph_step(
        self,
        seqs: list[Sequence],
        state: SnapKVDecodeGraphState,
        seq_ids: np.ndarray,
    ):
        inputs = state.inputs
        host = inputs.host
        host_facts = self.acquire_step_host_buffer(state.host_layer_fact_storage)
        host_free_starts, host_active_count = host_facts[:-1], host_facts[-1:]
        real_batch_size = len(seqs)
        graph_batch_size = int(inputs.batch_capacity)
        layer_ids = tuple(self.kv_transformer_layer_indices())
        first_layer = int(layer_ids[0])
        rows = np.asarray(
            [self.seq_id_to_row[first_layer][int(seq_id)] for seq_id in seq_ids],
            dtype=np.int32,
        )
        current = self.row_seq_lens[first_layer][rows]
        next_lens = current + 1
        if self.validate_runtime_invariants:
            for layer_id in layer_ids[1:]:
                layer_rows = np.asarray(
                    [self.seq_id_to_row[layer_id].get(int(seq_id), -1) for seq_id in seq_ids],
                    dtype=np.int32,
                )
                if not np.array_equal(layer_rows, rows):
                    raise RuntimeError(
                        "Uniform sparse graph decode requires identical request rows "
                        f"across layers: first={rows.tolist()} "
                        f"layer={layer_id} rows={layer_rows.tolist()}."
                    )
                if not np.array_equal(self.row_seq_lens[layer_id][layer_rows], current):
                    raise RuntimeError(
                        "Uniform sparse graph decode requires identical row lengths "
                        f"across layers: first={current.tolist()} "
                        f"layer={layer_id}."
                    )

        max_context_len = int(next_lens.max())
        if max_context_len > int(state.contract.context_capacity):
            raise RuntimeError(
                "Sparse decode context exceeds captured graph capacity: "
                f"real={max_context_len} capacity={state.contract.context_capacity}."
            )
        if max_context_len > int(self.max_model_len):
            raise RuntimeError(
                "Sparse decode context exceeds max_model_len: "
                f"real={max_context_len} max_model_len={self.max_model_len}."
            )
        free_ptrs = tuple(int(self._num_free_slots[layer_id]) for layer_id in layer_ids)
        if min(free_ptrs) < real_batch_size:
            raise RuntimeError(
                "Out of sparse KV cache slots during graph reservation: "
                f"need={real_batch_size} free={min(free_ptrs)}."
            )
        if self.validate_runtime_invariants and any(
            free_ptr != free_ptrs[0] for free_ptr in free_ptrs[1:]
        ):
            raise RuntimeError(
                "Uniform sparse graph decode requires aligned free-stack pointers: "
                f"layers={layer_ids} pointers={free_ptrs}."
            )

        host_active_count[0] = real_batch_size
        for layer_id, free_ptr in zip(layer_ids, free_ptrs):
            host_free_starts[layer_id] = free_ptr - real_batch_size
            self._num_free_slots[layer_id] -= real_batch_size
            self.row_seq_lens[layer_id][rows] += 1

        host.pack_cache_facts(
            context_lens=next_lens,
            request_indices=rows,
            real_batch_size=real_batch_size,
            padding_active=bool(state.contract.padding.active),
        )
        non_blocking = bool(host.input_ids.is_pinned())
        inputs.input_ids.copy_(host.input_ids, non_blocking=non_blocking)
        inputs.positions.copy_(host.positions, non_blocking=non_blocking)
        inputs.context_lens.copy_(host.context_lens, non_blocking=non_blocking)
        inputs.request_indices.copy_(host.request_indices, non_blocking=non_blocking)
        inputs.active_mask.copy_(host.active_mask, non_blocking=non_blocking)
        private_non_blocking = bool(host_facts.is_pinned())
        state.layer_fact_storage.copy_(
            host_facts,
            non_blocking=private_non_blocking,
        )

        binding_key = (
            graph_batch_size,
            int(state.layer_slot_mapping.data_ptr()),
            int(inputs.context_lens.data_ptr()),
            int(inputs.request_indices.data_ptr()),
        )
        if self._decode_static_state_binding_key != binding_key:
            for layer_id in layer_ids:
                layer_state = self.layer_batch_states[layer_id]
                layer_state.slot_mapping = state.layer_slot_mapping[layer_id]
                layer_state.context_lens = inputs.context_lens
                layer_state.req_indices = inputs.request_indices
            self._decode_static_state_binding_key = binding_key
        for layer_id in layer_ids:
            self.layer_batch_states[layer_id].max_context_len = max_context_len
        return inputs.input_ids, inputs.positions, None

    @torch.no_grad()
    def prepare_decode_graph_step(
        self,
        seqs: list[Sequence],
        state: CacheDecodeGraphState,
    ):
        if not isinstance(state, SnapKVDecodeGraphState):
            result = super().prepare_decode_graph_step(seqs, state)
            inputs = state.inputs
            host = inputs.host
            real_batch_size = len(seqs)
            graph_batch_size = int(inputs.batch_capacity)
            first_layer = int(self.kv_transformer_layer_indices()[0])
            row_indices = np.asarray(
                [self.seq_id_to_row[first_layer][int(seq.seq_id)] for seq in seqs],
                dtype=np.int32,
            )
            real_context_lens = self.row_seq_lens[first_layer][row_indices].astype(
                np.int32,
                copy=False,
            )
            host.input_ids.numpy()[:real_batch_size] = [
                int(seq.decode_input_token) for seq in seqs
            ]
            host.positions.numpy()[:real_batch_size] = [
                int(seq.decode_input_position) for seq in seqs
            ]
            host.context_lens.numpy()[:real_batch_size] = real_context_lens
            host.request_indices.numpy()[:real_batch_size] = row_indices
            host.active_mask[:real_batch_size].fill_(True)
            if graph_batch_size > real_batch_size:
                host.input_ids[real_batch_size:].fill_(int(seqs[0].decode_input_token))
                host.positions[real_batch_size:].fill_(
                    int(seqs[0].decode_input_position)
                )
                host.context_lens[real_batch_size:].fill_(int(real_context_lens[0]))
                host.request_indices[real_batch_size:].fill_(int(row_indices[0]))
                host.active_mask[real_batch_size:].fill_(state.contract.padding.active)
            return result

        inputs = state.inputs
        host = inputs.host
        real_batch_size = len(seqs)
        graph_batch_size = int(inputs.batch_capacity)
        if real_batch_size <= 0 or real_batch_size > graph_batch_size:
            raise ValueError(
                "Sparse decode graph requires a non-empty active batch within capacity: "
                f"real={real_batch_size} capacity={graph_batch_size}."
            )
        with profiler.record("cache_prepare_decode"):
            seq_ids = host.pack_requests(seqs)
            if not self._uniform_decode_metadata:
                raise RuntimeError(
                    "A layer-uniform decode graph cannot publish per-layer metadata."
                )
            return self._prepare_uniform_decode_graph_step(seqs, state, seq_ids)

    def prepare_decode_graph_in(self, state: CacheDecodeGraphState) -> None:
        if not isinstance(state, SnapKVDecodeGraphState):
            return super().prepare_decode_graph_in(state)
        from sparseengine.kernels.triton.sparse_decode_graph_metadata import (
            publish_uniform_sparse_decode_graph_slots,
        )

        publish_uniform_sparse_decode_graph_slots(
            self.free_slots_stack_tensor,
            self.buffer_req_to_token_slots_tensor,
            state.layer_slot_mapping,
            state.inputs.write_slot_mapping,
            state.inputs.context_lens,
            state.inputs.request_indices,
            state.layer_free_starts,
            state.active_count,
            state.transformer_layer_ids,
        )

    def decode_graph_state_keepalive_tensors(
        self,
        state: CacheDecodeGraphState,
    ) -> list[torch.Tensor]:
        tensors = super().decode_graph_state_keepalive_tensors(state)
        if isinstance(state, SnapKVDecodeGraphState):
            tensors.extend(
                (
                    state.layer_slot_mapping,
                    state.layer_fact_storage,
                    state.host_layer_fact_storage,
                    state.transformer_layer_ids,
                )
            )
        return tensors

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
        """Prepare per-layer decode metadata into graph-stable CUDA buffers."""
        with profiler.record("cache_prepare_decode"):
            real_batch_size = len(seqs)
            graph_batch_size = int(input_ids.numel())
            if real_batch_size <= 0:
                raise ValueError("Static decode requires a non-empty real decode batch.")
            if positions.numel() != graph_batch_size:
                raise ValueError("Static decode input buffers must have the same graph batch size.")
            if (
                slot_mapping.numel() != graph_batch_size
                or context_lens.numel() != graph_batch_size
                or req_indices.numel() != graph_batch_size
            ):
                raise ValueError("Static decode metadata buffers must have the same graph batch size.")
            if real_batch_size > graph_batch_size:
                raise ValueError(
                    "Static decode graph batch is smaller than the real decode batch: "
                    f"graph={graph_batch_size}, real={real_batch_size}."
                )

            input_ids_list = [seq.decode_input_token for seq in seqs]
            positions_list = [seq.decode_input_position for seq in seqs]
            seq_ids = [seq.seq_id for seq in seqs]

            if self._uniform_decode_metadata:
                result = self._prepare_decode_static_uniform(
                    seqs,
                    input_ids,
                    positions,
                    slot_mapping,
                    context_lens,
                    req_indices,
                )
                if result is not None:
                    return result

            input_ids[:real_batch_size].copy_(torch.tensor(input_ids_list, dtype=torch.int64, device=self.device))
            positions[:real_batch_size].copy_(torch.tensor(positions_list, dtype=torch.int64, device=self.device))
            if graph_batch_size > real_batch_size:
                input_ids[real_batch_size:].fill_(int(input_ids_list[0]))
                positions[real_batch_size:].fill_(int(positions_list[0]))

            layers_slot_mapping, layers_context_lens, layers_req_indices = self._get_decode_static_buffers(
                graph_batch_size
            )

            static_cap = getattr(self, "_decode_static_max_context_len", None)
            new_slots, context_lens_np, req_indices_np, wrote_slot_output = self._allocate_decode_batch_all_layers(
                seq_ids,
                static_cap=None if static_cap is None else int(static_cap),
                slot_output=layers_slot_mapping,
            )
            max_context_lens = context_lens_np.max(axis=1) if context_lens_np.size else np.zeros(self.num_layers)

            if not wrote_slot_output:
                layers_slot_mapping[:, :real_batch_size].copy_(new_slots)
            layers_context_lens[:, :real_batch_size].copy_(
                torch.from_numpy(context_lens_np).to(device=self.device, dtype=torch.int32)
            )
            layers_req_indices[:, :real_batch_size].copy_(
                torch.from_numpy(req_indices_np).to(device=self.device, dtype=torch.int32)
            )
            if graph_batch_size > real_batch_size:
                layers_slot_mapping[:, real_batch_size:].fill_(-1)
                layers_context_lens[:, real_batch_size:].copy_(
                    torch.from_numpy(context_lens_np[:, :1]).to(device=self.device, dtype=torch.int32)
                )
                layers_req_indices[:, real_batch_size:].copy_(
                    torch.from_numpy(req_indices_np[:, :1]).to(device=self.device, dtype=torch.int32)
                )

            self._bind_decode_static_layer_states(
                graph_batch_size,
                layers_slot_mapping,
                layers_context_lens,
                layers_req_indices,
                max_context_lens,
            )
            self.validate_decode_cuda_graph_slot_mappings()

            first_layer = int(self.kv_transformer_layer_indices()[0])
            slot_mapping.copy_(layers_slot_mapping[first_layer])
            context_lens.copy_(layers_context_lens[first_layer])
            req_indices.copy_(layers_req_indices[first_layer])

            return input_ids, positions, None
