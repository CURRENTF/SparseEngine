from __future__ import annotations

from collections import deque
from itertools import islice
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np
import torch

from sparseengine.config import Config
from sparseengine.distributed import ParallelContext
from sparseengine.engine.decode_graph_contract import (
    CacheDecodeGraphState,
    DecodeGraphHostInputs,
)
from sparseengine.engine.prefix_cache import PrefixCacheBlock
from sparseengine.engine.sequence import Sequence
from sparseengine.env import debug_slots_enabled
from sparseengine.platforms import device_runtime
from sparseengine.utils.log import log_level, logger
from sparseengine.utils.profiler import cpu_timing, profiler

from .base import (
    AttentionCacheWrite,
    AttentionPayload,
    CacheManager,
    LayerBatchStates,
    MlaLatentPayload,
    MlaLatentWrite,
    PrefillComputeView,
    SparseSelection,
)
from .prefix_cache_mixin import PrefixCacheMixin
from .prefix_prune_scoring import PrefixPruneScoringMixin
from .prefix_offload import (
    PinnedPrefixKVPool,
    StandardPrefixOffloadController,
)
from .storage import (
    ExplicitKVStorage,
    HeterogeneousExplicitKVStorage,
    MlaLatentStorage,
    create_attention_cache_storage,
)


@dataclass
class StandardPrefixBlockPayload:
    token_slots: torch.Tensor | None
    block_start: int = 0
    block_end: int = 0
    host_block_index: int | None = None
    retained_offsets: tuple[int, ...] | None = None

    def resident_tokens(self, block_size: int) -> int:
        if self.retained_offsets is None:
            return int(block_size)
        return len(self.retained_offsets)


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    merged: list[tuple[int, int]] = []
    for start, end in sorted((int(s), int(e)) for s, e in ranges if int(e) > int(s)):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _complement_ranges(start: int, end: int, ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cur = int(start)
    result: list[tuple[int, int]] = []
    for range_start, range_end in _merge_ranges(ranges):
        if cur < range_start:
            result.append((cur, range_start))
        cur = max(cur, range_end)
    if cur < int(end):
        result.append((cur, int(end)))
    return result


class StandardCacheManager(PrefixPruneScoringMixin, PrefixCacheMixin, CacheManager):
    _prefix_payload_type = StandardPrefixBlockPayload

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
        self.attention_cache_storage = create_attention_cache_storage(
            config,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )
        self.allocate_kv_cache()

        num_slots = config.num_kvcache_slots
        self.free_slots_stack = torch.arange(num_slots, dtype=torch.int32, device=self.device)
        self._num_free_slots = num_slots

        self.buffer_req_to_token_slots = torch.zeros(
            (self.max_buffer_rows, self.max_model_len), dtype=torch.int32, device=self.device
        )

        self.seq_id_to_row: dict[int, int] = {}
        self.free_rows = deque(range(self.max_buffer_rows))
        self.row_seq_lens = np.zeros((self.max_buffer_rows,), dtype=np.int32)
        # Physical KV slots may be shorter than logical positions after an
        # idle prefix tree has been pruned.
        self.row_logical_lens = np.zeros((self.max_buffer_rows,), dtype=np.int32)
        self.layer_batch_state = LayerBatchStates()

        self.enable_prefix_caching = bool(
            config.enable_prefix_caching and config.sparse_method in ("", "omnikv", "fp8_kv")
            and getattr(config, "resolved_prefix_cache_mode", "radix") == "radix"
            and not getattr(getattr(config, "runtime_layout", None), "linear_attention_layer_indices", ())
        )
        self.prefix_cache_block_size = int(config.prefix_cache_block_size)
        self.seq_id_to_cached_ranges: dict[int, list[tuple[int, int]]] = {}
        self._prefix_prune_scoring: dict[str, object] | None = None
        self._init_radix_prefix_cache()

    def _prefix_offload_device_blocks(self) -> int:
        return int(self.config.num_kvcache_slots) // self.prefix_cache_block_size

    def _build_explicit_prefix_offload_controller(self, host_capacity_blocks: int):
        storage = self.attention_cache_storage
        host_pool = PinnedPrefixKVPool(
            capacity_blocks=host_capacity_blocks,
            num_layers=self.num_kv_layers,
            block_size=self.prefix_cache_block_size,
            num_kv_heads=storage.num_kv_heads,
            head_dim=storage.head_dim,
            dtype=storage.dtype,
        )
        return StandardPrefixOffloadController(
            prefix_cache=self.prefix_cache,
            kv_cache=storage.cache,
            host_pool=host_pool,
            block_size=self.prefix_cache_block_size,
            device=self.device,
        )

    def allocate_kv_cache(self):
        available_memory, slot_bytes_per_layer = self._get_available_slots_info()
        num_layers = self.num_kv_layers

        storage = self.attention_cache_storage
        slot_bytes = (
            storage.bytes_per_slot()
            if isinstance(storage, HeterogeneousExplicitKVStorage)
            else num_layers * slot_bytes_per_layer
        )
        if getattr(self, "max_buffer_rows", None) is not None and hasattr(self.config, "limit_auto_max_model_len"):
            slot_bytes += torch.tensor([], dtype=torch.int32).element_size()
            row_bytes_per_token = self.max_buffer_rows * torch.tensor([], dtype=torch.int32).element_size()
            self.config.limit_auto_max_model_len(
                available_memory // (slot_bytes + row_bytes_per_token)
            )
            self.max_model_len = self.config.max_model_len
            available_memory -= self.max_model_len * row_bytes_per_token

        self.config.num_kvcache_slots = available_memory // slot_bytes
        if (
            getattr(self.config, "startup_cache_phase", "production")
            != "profiling"
            and getattr(self, "max_model_len", None) is not None
            and self.config.num_kvcache_slots < self.max_model_len
        ):
            raise RuntimeError(
                "KV cache capacity is smaller than max_model_len after reserving runtime metadata: "
                f"capacity={self.config.num_kvcache_slots} max_model_len={self.max_model_len}."
            )
        # Host-resident prefixes remain live after their GPU slots are reclaimed.
        if (
            getattr(self.config, "prefix_cache_max_blocks", None) is not None
            and not getattr(self.config, "enable_prefix_cache_offload", False)
        ):
            self.config.prefix_cache_max_blocks = min(
                self.config.prefix_cache_max_blocks,
                self.config.num_kvcache_slots // getattr(self.config, "prefix_cache_block_size", 16),
            )

        self.attention_cache_storage.allocate(
            num_layers=num_layers,
            num_slots=self.config.num_kvcache_slots,
            device=self.device,
        )
        self.kv_cache = getattr(self.attention_cache_storage, "kv_cache", None)

    def attention_cache_bytes_per_slot_per_layer(self) -> int:
        storage = getattr(self, "attention_cache_storage", None)
        if storage is None:
            return super().attention_cache_bytes_per_slot_per_layer()
        return int(storage.bytes_per_slot_per_layer())

    def _logical_live_kv_bytes(self) -> int:
        storage = getattr(self, "attention_cache_storage", None)
        if not isinstance(storage, HeterogeneousExplicitKVStorage):
            return super()._logical_live_kv_bytes()
        return int(self.row_seq_lens.sum()) * storage.bytes_per_slot()

    def _require_explicit_storage(
        self, operation: str
    ) -> ExplicitKVStorage | HeterogeneousExplicitKVStorage:
        storage = self.attention_cache_storage
        if not isinstance(storage, (ExplicitKVStorage, HeterogeneousExplicitKVStorage)):
            raise TypeError(
                f"{operation} requires ExplicitKVStorage, got "
                f"{type(storage).__name__}."
            )
        return storage

    def get_layer_batch_states(self, layer_idx: int) -> LayerBatchStates:
        return self.layer_batch_state

    def get_layer_kv_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        kv_idx = self.kv_layer_index(layer_idx)
        payload = self._require_explicit_storage("get_layer_kv_cache").layer_payload(kv_idx)
        return payload.k_cache, payload.v_cache

    def get_layer_store_view(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        return k_cache, v_cache, self.layer_batch_state.slot_mapping

    def store_attention_payload(
        self,
        layer_idx: int,
        payload: AttentionCacheWrite,
    ) -> torch.Tensor:
        slot_mapping = self.layer_batch_state.slot_mapping
        if slot_mapping is None:
            raise RuntimeError(
                f"Attention cache store requires slot_mapping at layer={layer_idx}."
            )
        self.attention_cache_storage.store(
            self.kv_layer_index(layer_idx),
            slot_mapping,
            payload,
        )
        return slot_mapping

    def _validate_attention_slot_mapping(self, slot_mapping: torch.Tensor) -> None:
        storage = getattr(self, "attention_cache_storage", None)
        if storage is not None:
            storage.validate_slot_mapping(slot_mapping)

    def get_layer_compute_payload(
        self,
        layer_idx: int,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        selection: SparseSelection | None = None,
    ) -> tuple[AttentionPayload, torch.Tensor, torch.Tensor, torch.Tensor]:
        del selection
        return (
            self.attention_cache_storage.layer_payload(
                self.kv_layer_index(layer_idx)
            ),
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
    ) -> tuple[AttentionPayload, torch.Tensor, torch.Tensor, torch.Tensor]:
        del k_current, v_current, selection
        return self.get_layer_compute_payload(
            layer_idx,
            active_slots,
            req_indices,
            context_lens,
        )

    def build_prefill_compute_view(
        self, layer_idx: int, k_current: torch.Tensor, v_current: torch.Tensor,
        selection: SparseSelection,
    ) -> PrefillComputeView:
        view = super().build_prefill_compute_view(
            layer_idx, k_current, v_current, selection,
        )
        storage = self.attention_cache_storage
        meta, payload = view.meta, view.payload
        state = self.layer_batch_state
        # _prepare_prefill appends current tokens in packed query order to the
        # tail of these exact physical rows. Staged, selected, transformed, or
        # differently typed views must continue reading their cache payload.
        if (
            not isinstance(storage, MlaLatentStorage)
            or not isinstance(payload, MlaLatentPayload)
            or meta.temp_slots is not None
            or meta.active_slots is not self.buffer_req_to_token_slots
            or meta.req_indices is not state.req_indices
            or meta.context_lens is not state.context_lens
            or k_current.dtype != payload.latent_cache.dtype
            or v_current.dtype != payload.rope_cache.dtype
            or k_current.device != payload.latent_cache.device
            or v_current.device != payload.rope_cache.device
        ):
            return view
        physical = storage.layer_payload(self.kv_layer_index(layer_idx))
        if (
            payload.latent_cache.data_ptr() != physical.latent_cache.data_ptr()
            or payload.rope_cache.data_ptr() != physical.rope_cache.data_ptr()
            or payload.latent_cache.shape != physical.latent_cache.shape
            or payload.rope_cache.shape != physical.rope_cache.shape
            or payload.latent_cache.stride() != physical.latent_cache.stride()
            or payload.rope_cache.stride() != physical.rope_cache.stride()
        ):
            return view
        return replace(view, current_mla=MlaLatentWrite(
            latent=k_current.unsqueeze(1), rope=v_current.unsqueeze(1),
        ), host_request_layout=getattr(self, "prefill_plan", None))

    def get_layer_compute_tensors(self, layer_idx: int, selection: SparseSelection | None = None):
        del selection
        raise NotImplementedError

    def get_layer_buffer_req_to_token_slots(self, layer_idx: int) -> torch.Tensor:
        self.kv_layer_index(layer_idx)
        return self.buffer_req_to_token_slots

    @property
    def num_free_slots(self) -> int:
        return self._num_free_slots

    @property
    def num_free_rows(self) -> int:
        return len(self.free_rows)

    @cpu_timing.timed
    def prompt_admission_budgets(
        self,
        waiting_seqs: deque[Sequence],
        engine_prefill_chunk_size: int,
    ) -> dict[str, int]:
        budgets = super().prompt_admission_budgets(waiting_seqs, engine_prefill_chunk_size)
        budgets["rows"] = int(self.num_free_rows)
        return budgets

    def prompt_admission_costs(self, seq: Sequence) -> dict[str, int]:
        costs = super().prompt_admission_costs(seq)
        costs["rows"] = 1
        return costs

    def free_slot_stats(self) -> dict[str, int]:
        self._poll_prefix_offload()
        stats = super().free_slot_stats()
        stats["free_rows"] = int(self.num_free_rows)
        if getattr(self, "prefix_cache", None) is not None:
            stats.update(self.prefix_cache.stats())
            stats["prefix_cache_evictable_slots"] = int(self._prefix_evictable_slots())
        controller = getattr(self, "prefix_offload_controller", None)
        if controller is not None:
            stats.update(controller.stats())
        return stats

    def _prefix_resident_slots_for_ids(
        self, block_ids: frozenset[bytes], *, view: str = "explicit"
    ) -> int:
        return self._prefix_resident_weight_for_ids(block_ids, view=view)

    def _prefix_immediately_evictable_slots(self) -> int:
        if (
            getattr(self, "prefix_cache", None) is None
            or self._prefix_offload_enabled()
        ):
            return 0
        return self._prefix_resident_slots_for_ids(
            self.prefix_cache.evictable_block_ids(), view="immediate"
        )

    def prefill_step_free_slots(self) -> int:
        physical_free = int(self.num_free_slots)
        max_step_tokens = int(
            getattr(self.config, "max_num_batched_tokens", 0) or 0
        )
        if max_step_tokens > 0 and physical_free >= max_step_tokens:
            return physical_free
        return int(self.num_free_slots + self._prefix_step_reclaimable_slots())

    def decode_step_free_slots(self) -> int:
        physical_free = int(self.num_free_slots)
        max_step_seqs = int(
            getattr(self.config, "max_decoding_seqs", 0) or 0
        )
        if max_step_seqs > 0 and physical_free >= max_step_seqs:
            return physical_free
        immediately_evictable = self._prefix_immediately_evictable_slots()
        if (
            max_step_seqs > 0
            and physical_free + immediately_evictable >= max_step_seqs
        ):
            return int(physical_free + immediately_evictable)
        return int(self.num_free_slots + self._prefix_step_reclaimable_slots())

    def prompt_admission_cost(self, seq: Sequence) -> int:
        hit_len = int(getattr(seq, "prefix_cache_hit_len", 0) or 0)
        suffix_len = int(seq.num_prompt_tokens - hit_len)
        if hit_len <= 0:
            return suffix_len
        reclaimable_slots, promotion_slots = self._prefix_hit_capacity_slots(seq)
        return suffix_len + reclaimable_slots + promotion_slots

    def prompt_admission_shared_costs(
        self, seq: Sequence
    ) -> dict[str, dict[object, int]]:
        if self.prefix_cache is None or int(getattr(seq, "prefix_cache_hit_len", 0) or 0) <= 0:
            return {}
        if "slots" not in self.prompt_admission_costs(seq):
            return {}
        chain = self._prefix_hit_chain(seq)
        reclaimable_ids = (
            self.prefix_cache.device_reclaimable_block_ids()
            if self._prefix_offload_enabled()
            else self.prefix_cache.freeable_block_ids()
        )
        costs: dict[object, int] = {}
        for block in chain:
            block_id = block.stable_block_id
            if block_id in reclaimable_ids or (
                self._prefix_offload_enabled() and not block.residency.device_present
            ):
                costs[block_id] = self._block_resident_tokens_or_full(block)
        return {"slots": costs} if costs else {}

    def _block_resident_tokens_or_full(self, block: PrefixCacheBlock) -> int:
        payload = block.payload
        if isinstance(payload, StandardPrefixBlockPayload):
            return payload.resident_tokens(self.prefix_cache_block_size)
        # Tests and external index-only users may intentionally use metadata-only payloads.
        return int(self.prefix_cache_block_size)

    def _prefix_block_resident_tokens(self, block: PrefixCacheBlock) -> int:
        return self._block_resident_tokens_or_full(block)

    @cpu_timing.timed
    def _prefix_hit_capacity_slots(self, seq: Sequence) -> tuple[int, int]:
        return self._prefix_hit_capacity_units(seq)

    def _standard_payload(self, block: PrefixCacheBlock) -> StandardPrefixBlockPayload:
        payload = block.payload
        if not isinstance(payload, StandardPrefixBlockPayload):
            raise RuntimeError("Standard prefix cache block has an invalid payload.")
        return payload

    _prefix_block_payload = _standard_payload

    def _free_device_prefix_blocks(self, blocks: list[PrefixCacheBlock]) -> None:
        # Validate the whole ownership return before publishing any free slots.
        parts = []
        count = 0
        for block in blocks:
            payload = block.payload
            if not isinstance(payload, StandardPrefixBlockPayload):
                raise RuntimeError("Standard prefix cache block is missing its device payload.")
            slots = payload.token_slots
            expected = payload.resident_tokens(self.prefix_cache_block_size)
            if not isinstance(slots, torch.Tensor) or int(slots.numel()) != expected:
                raise RuntimeError(
                    "Standard prefix cache block has invalid device slots: "
                    f"block={block.stable_block_id.hex()[:16]}."
                )
            if expected:
                parts.append(slots)
                count += expected
        ptr = self._num_free_slots
        if ptr + count > int(self.free_slots_stack.numel()):
            raise RuntimeError("Standard prefix slot free stack overflow.")
        if parts:
            slots = parts[0] if len(parts) == 1 else torch.cat(parts)
            slots = slots.to(device=self.device, dtype=torch.int32)
            self.free_slots_stack[ptr: ptr + count].copy_(slots)
        self._num_free_slots += count
        for block in blocks:
            block.payload.token_slots = None

    def _make_prefix_block_payload(self, slots: torch.Tensor) -> StandardPrefixBlockPayload:
        return StandardPrefixBlockPayload(
            token_slots=slots,
            block_start=0,
            block_end=int(slots.numel()),
            retained_offsets=None,
        )

    def _mark_materialized_prefix_block(self, seq: Sequence, block: PrefixCacheBlock) -> None:
        cached_ranges = self.seq_id_to_cached_ranges.setdefault(seq.seq_id, [])
        # Appending tokens advances physical and logical row lengths equally.
        # Their difference is the fixed prefix-pruning offset while attached
        # blocks are referenced; avoid rescanning the entire prefix per token.
        hit_len = int(getattr(seq, "prefix_cache_hit_len", 0) or 0)
        offset = 0
        if hit_len:
            row_idx = self.seq_id_to_row[seq.seq_id]
            offset = int(self.row_seq_lens[row_idx]) - int(self.row_logical_lens[row_idx])
        start = int(block.logical_block_idx) * self.prefix_cache_block_size + offset
        if start < 0:
            raise RuntimeError(
                "materialized prefix block resolved to a negative physical row offset: "
                f"seq_id={seq.seq_id} logical_block={block.logical_block_idx} "
                f"physical_offset={offset} hit_len={hit_len}."
            )
        cached_ranges.append((start, start + self.prefix_cache_block_size))

    def _build_mixed_prefix_payload(
        self, slots: torch.Tensor, block_start: int, block_end: int
    ) -> StandardPrefixBlockPayload:
        return StandardPrefixBlockPayload(
            token_slots=slots,
            block_start=block_start,
            block_end=block_end,
        )

    def _mixed_prefix_attach_units(
        self, seq_id: int, start: int, payload: StandardPrefixBlockPayload, slots: torch.Tensor
    ) -> None:
        count = int(slots.numel())
        if count <= 0:
            raise RuntimeError("Standard mixed prefix KV payload is empty.")
        if count % int(self.config.prefix_cache_block_size) != 0:
            raise RuntimeError(
                f"Standard mixed prefix KV payload size must be block-aligned, got {count}."
            )
        if int(payload.block_start) != start:
            raise RuntimeError(
                "Standard mixed prefix KV payload attach must be contiguous: "
                f"seq_id={seq_id} block_start={int(payload.block_start)} row_len={start}."
            )
        end = start + count
        if int(payload.block_end) not in {0, end}:
            raise RuntimeError(
                "Standard mixed prefix KV payload has inconsistent block_end: "
                f"payload_end={int(payload.block_end)} expected={end}."
            )

    def free_prefix_kv_payload(self, payload: object) -> None:
        if not isinstance(payload, StandardPrefixBlockPayload):
            raise RuntimeError("Standard mixed prefix KV payload is missing token slots.")
        if not isinstance(payload.token_slots, torch.Tensor):
            raise RuntimeError("Standard mixed prefix KV payload has no device slots.")
        slots = payload.token_slots.to(device=self.device, dtype=torch.int32).reshape(-1)
        count = int(slots.numel())
        ptr = self._num_free_slots
        if ptr + count > int(self.free_slots_stack.numel()):
            raise RuntimeError(
                "Freeing mixed prefix KV payload would overflow the slot pool: "
                f"free={ptr} returning={count} capacity={self.free_slots_stack.numel()}."
            )
        self.free_slots_stack[ptr: ptr + count].copy_(slots)
        self._num_free_slots += count
        payload.token_slots = None

    def _stage_mixed_prefix_payloads(
        self, payloads: list[StandardPrefixBlockPayload], counts: list[int], slots: torch.Tensor
    ) -> None:
        offset = 0
        for payload, count in zip(payloads, counts):
            payload.token_slots = slots[offset : offset + count]
            offset += count

    def prefix_kv_payload_nbytes(self, payload: object) -> int:
        if not isinstance(payload, StandardPrefixBlockPayload):
            raise RuntimeError("Standard mixed prefix KV payload is missing token slots.")
        if not isinstance(payload.token_slots, torch.Tensor):
            raise RuntimeError("Standard mixed prefix KV payload has no device slots.")
        storage = self.attention_cache_storage
        if isinstance(storage, HeterogeneousExplicitKVStorage):
            return int(payload.token_slots.numel()) * storage.bytes_per_slot()
        dtype_size = self._cache_slot_dtype_size()
        return int(
            payload.token_slots.numel()
            * self.num_kv_layers
            * 2
            * self.num_kv_heads
            * self.head_dim
            * dtype_size
        )

    def _compact_prefix_prune_payloads(
        self, plans: list[tuple[PrefixCacheBlock, tuple[int, ...]]]
    ) -> dict[str, object]:
        block_size = self.prefix_cache_block_size
        resident_slots = []
        keep_positions: list[int] = []
        drop_positions: list[int] = []
        slot_cursor = 0
        for block, new_offsets in plans:
            payload = self._standard_payload(block)
            old_offsets = (
                tuple(range(block_size)) if payload.retained_offsets is None
                else tuple(int(offset) for offset in payload.retained_offsets)
            )
            if block.residency.device_present:
                old_slots = payload.token_slots
                if not isinstance(old_slots, torch.Tensor) or old_slots.numel() != len(old_offsets):
                    raise RuntimeError(
                        "prefix prune found inconsistent Standard device payload: "
                        f"block={block.stable_block_id.hex()[:16]} offsets={len(old_offsets)}."
                    )
                new_set = set(new_offsets)
                keep_positions.extend(
                    slot_cursor + i for i, offset in enumerate(old_offsets) if offset in new_set
                )
                drop_positions.extend(
                    slot_cursor + i for i, offset in enumerate(old_offsets) if offset not in new_set
                )
                resident_slots.append(old_slots)
                slot_cursor += len(old_offsets)
        # Prepare all tensors before mutation. Gather/release once even for B=1.
        kept_slots = dropped_slots = None
        if resident_slots:
            all_slots = torch.cat(resident_slots)
            kept_slots = all_slots[
                torch.tensor(keep_positions, dtype=torch.long, device=all_slots.device)
            ]
            dropped_slots = all_slots[
                torch.tensor(drop_positions, dtype=torch.long, device=all_slots.device)
            ]
        # Failures in validation or tensor preparation leave every interval intact.
        kept_cursor = 0
        for block, new_offsets in plans:
            payload = self._standard_payload(block)
            payload.retained_offsets = new_offsets
            if block.residency.device_present:
                assert kept_slots is not None
                payload.token_slots = kept_slots[kept_cursor : kept_cursor + len(new_offsets)]
                kept_cursor += len(new_offsets)
        if dropped_slots is not None:
            self._return_prefix_device_slots(dropped_slots)
        return {"freed_device_slots": len(drop_positions)}

    def mark_materialized_prefix_kv_payload(self, seq: Sequence, payload: object) -> None:
        if not isinstance(payload, StandardPrefixBlockPayload):
            raise RuntimeError("Standard mixed prefix KV payload is missing token slots.")
        row_idx = self.seq_id_to_row.get(int(seq.seq_id))
        if row_idx is None:
            raise RuntimeError(f"Cannot mark mixed prefix payload for unknown seq_id={seq.seq_id}.")
        start = int(payload.block_start)
        end = int(payload.block_end)
        row_len = int(self.row_seq_lens[row_idx])
        if start < 0 or end <= start or end > row_len:
            raise RuntimeError(
                "Cannot mark mixed prefix payload: "
                f"seq_id={seq.seq_id} range={start}:{end} row_len={row_len}."
            )
        self.seq_id_to_cached_ranges.setdefault(int(seq.seq_id), []).append((start, end))

    def rollback_materialized_prefix_kv_payload(
        self,
        seq: Sequence,
        payload: object,
    ) -> None:
        if not isinstance(payload, StandardPrefixBlockPayload):
            raise RuntimeError("Standard mixed prefix KV payload is missing token slots.")
        seq_id = int(seq.seq_id)
        target = (int(payload.block_start), int(payload.block_end))
        cached_ranges = self.seq_id_to_cached_ranges.get(seq_id)
        if not cached_ranges:
            return
        for idx in range(len(cached_ranges) - 1, -1, -1):
            if cached_ranges[idx] == target:
                cached_ranges.pop(idx)
                break
        if not cached_ranges:
            self.seq_id_to_cached_ranges.pop(seq_id, None)

    def _reset_prefix_cache_allocator_after_clear(self) -> None:
        if self.seq_id_to_row:
            raise RuntimeError("Cannot reset prefix cache while Standard sequences are active.")
        controller = getattr(self, "prefix_offload_controller", None)
        if controller is not None:
            controller.reset()
        num_slots = int(self.config.num_kvcache_slots)
        self.free_slots_stack[:num_slots] = torch.arange(num_slots, dtype=torch.int32, device=self.device)
        self._num_free_slots = num_slots
        self.seq_id_to_cached_ranges.clear()
        getattr(self, "_prefix_write_through_candidates", {}).clear()

    def reset_after_warmup(self) -> None:
        if getattr(self.config, "resolved_prefix_cache_mode", "disabled") == "chain":
            # TP workers do not own the scheduler's resident-sequence ledger.
            # Reclaim their retained warmup chains from the local allocator.
            for seq_id in sorted(self.seq_id_to_row):
                self.free_seq(seq_id)
        if self.enable_prefix_caching and self.prefix_cache is not None:
            self.reset_prefix_cache()
            return
        self._reset_prefix_cache_allocator_after_clear()

    @cpu_timing.timed
    def on_forward_end(self, seqs: list[Sequence], is_prefill: bool):
        self._poll_prefix_offload()
        return super().on_forward_end(seqs, is_prefill)

    @property
    def _prefix_cached_rows(self) -> dict[int, list[tuple[int, int]]]:
        return self.seq_id_to_cached_ranges

    def _prefix_cached_entries(self, ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return ranges

    def _mark_mixed_prefix_ranges(self, seq_id: int, ranges: list[tuple[int, int]]) -> None:
        self.seq_id_to_cached_ranges.setdefault(seq_id, []).extend(ranges)

    def _take_prefix_device_units(self, count: int) -> torch.Tensor:
        return self._take_prefix_device_slots(count)

    def _return_prefix_device_units(self, units: torch.Tensor) -> None:
        self._return_prefix_device_slots(units)

    def _validate_resident_radix_payload(
        self, seq_id: int, block: PrefixCacheBlock, expected_tokens: int
    ) -> None:
        slots = block.payload.token_slots
        if not isinstance(slots, torch.Tensor) or int(slots.numel()) != expected_tokens:
            raise RuntimeError(
                f"Invalid Standard prefix cache block slots for seq_id={seq_id}: "
                f"logical_block_idx={block.logical_block_idx}."
            )

    def _stage_radix_prefix_blocks(
        self, blocks: list[PrefixCacheBlock], units: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        offset = 0
        for block in blocks:
            payload = block.payload
            end = offset + payload.resident_tokens(self._radix_block_size)
            payload.token_slots = units[offset:end]
            offset = end
        return (units if units.numel() else None), None

    def _mixed_prefix_allocation_count(self, tokens: int) -> int:
        if tokens <= 0:
            raise RuntimeError("Mixed prefix KV payload has an invalid token range.")
        return tokens

    def _release_mixed_prefix_payload_device(self, payload: StandardPrefixBlockPayload) -> None:
        self._return_prefix_device_units(payload.token_slots)
        payload.token_slots = None


    def _validate_prefix_row_capacity(self, seq_id: int, end: int, *, mixed: bool) -> None:
        if mixed:
            if end > int(self.max_model_len):
                raise RuntimeError(
                    "Attaching mixed prefix KV payload exceeds max_model_len: "
                    f"seq_id={seq_id} end={end} max_model_len={self.max_model_len}."
                )
        elif end > self.buffer_req_to_token_slots.shape[1]:
            raise ValueError("Prefix attachment exceeds the physical row capacity.")

    def _take_device_slots(self, count: int, *, copy: bool = False) -> torch.Tensor:
        """Reclaim eligible cache entries, then transfer slots out of the free pool."""
        count = int(count)
        if count < 0:
            raise ValueError("Device slot count must be non-negative.")
        self._evict_prefix_cache_until_free(count)
        if self._num_free_slots < count:
            raise RuntimeError(
                "Out of KV cache slots after prefix eviction: "
                f"need={count} free={self._num_free_slots}."
            )
        ptr = self._num_free_slots
        slots = self.free_slots_stack[ptr - count:ptr]
        if copy:
            slots = slots.clone()
        self._num_free_slots -= count
        return slots

    def _take_prefix_device_slots(self, count: int) -> torch.Tensor:
        return self._take_device_slots(count, copy=True)

    @contextmanager
    def reserve_prefill_slots(
        self,
        requests: list[tuple[int, int]],
        *,
        prefix_blocks: dict[int, list[PrefixCacheBlock]] | None = None,
    ):
        """Reserve an ordered prefix of (seq_id, query_tokens) before execution.

        Callers pin any cached prefixes they need before admission. Eviction uses
        the ordinary allocator policy and respects those refs. Unconsumed slots
        and newly claimed rows are returned on exit, including failed forwards.
        Consumed slots follow the normal row ownership/free_seq lifecycle.
        CPU-only prefix blocks need promotion headroom, counted once per shared
        block. Leave that headroom in the pool for the ordinary attach path.
        """
        if getattr(self, "_prefill_slot_reservations", None) is not None:
            raise RuntimeError("A prefill slot reservation is already active.")
        if not requests or len({sid for sid, _ in requests}) != len(requests):
            raise ValueError("Prefill reservation requires distinct sequence IDs.")
        if any(size <= 0 for _, size in requests):
            raise ValueError("Prefill reservation sizes must be positive.")
        candidates = []
        required_slots = []
        promotion_blocks = set()
        query_slots = promotion_slots = 0
        available_rows = len(self.free_rows)
        for sid, size in requests:
            if sid not in self.seq_id_to_row:
                if not available_rows:
                    break
                available_rows -= 1
            candidates.append((sid, size))
            query_slots += size
            for block in (() if prefix_blocks is None else prefix_blocks[sid]):
                if not block.residency.device_present and block.stable_block_id not in promotion_blocks:
                    promotion_blocks.add(block.stable_block_id)
                    promotion_slots += self._standard_payload(block).resident_tokens(
                        self.prefix_cache_block_size
                    )
            required_slots.append(query_slots + promotion_slots)
        if not candidates:
            raise RuntimeError("Prefill reservation requires an idle cache row.")
        self._evict_prefix_cache_until_free(required_slots[-1])
        admitted, total = [], 0
        for (sid, size), required in zip(candidates, required_slots, strict=True):
            if required > self._num_free_slots:
                break
            admitted.append((sid, size))
            total += size
        if not admitted:
            raise RuntimeError(
                "Out of KV cache slots after prefix eviction for prefill reservation: "
                f"need={required_slots[0]} free={self._num_free_slots}."
            )
        # Clone once: intervening frees may overwrite the allocator's stack.
        slots = self._take_device_slots(total, copy=True)
        reserved, offset = {}, 0
        for sid, size in admitted:
            reserved[sid] = slots[offset:offset + size]
            offset += size
        new_rows = [sid for sid, _ in admitted if sid not in self.seq_id_to_row]
        self._prefill_slot_reservations = reserved
        try:
            for sid in new_rows:
                self._get_free_row(sid)
            yield len(admitted)
        finally:
            self._prefill_slot_reservations = None
            if reserved:
                self._return_prefix_device_slots(torch.cat(list(reserved.values())))
            for sid in new_rows:
                if sid in self.seq_id_to_row:
                    self.free_seq(sid)

    def _return_prefix_device_slots(self, slots: torch.Tensor) -> None:
        slots = slots.to(device=self.device, dtype=torch.int32).reshape(-1)
        count = int(slots.numel())
        ptr = self._num_free_slots
        self.free_slots_stack[ptr:ptr + count] = slots
        self._num_free_slots += count

    def _get_free_row(self, seq_id: int) -> int:
        if not hasattr(self, "row_logical_lens"):
            self.row_logical_lens = self.row_seq_lens.copy()
        if seq_id in self.seq_id_to_row:
            return self.seq_id_to_row[seq_id]
        if not self.free_rows:
            raise RuntimeError("No free rows in cache manager buffer!")
        row_idx = self.free_rows.popleft()
        self.seq_id_to_row[seq_id] = row_idx
        return row_idx

    @torch.no_grad()
    def _allocate(self, seq_id: int, size: int) -> torch.Tensor:
        with profiler.record("cache_allocate"):
            if type(size) is not int or size < 0:
                raise ValueError("Prefill allocation size must be a non-negative integer.")
            existing_row = self.seq_id_to_row.get(seq_id)
            if existing_row is None and not self.free_rows:
                raise RuntimeError("No free rows in cache manager buffer!")
            cur_len = 0 if existing_row is None else int(self.row_seq_lens[existing_row])
            if cur_len + size > self.buffer_req_to_token_slots.shape[1]:
                raise ValueError("Prefill allocation exceeds the cache row capacity.")
            old_logical_len = (
                cur_len if existing_row is None or not hasattr(self, "row_logical_lens")
                else int(self.row_logical_lens[existing_row])
            )
            reservations = getattr(self, "_prefill_slot_reservations", None)
            reserved = None if reservations is None else reservations.get(seq_id)
            if reserved is not None and reserved.numel() != size:
                raise RuntimeError("Prefill allocation differs from its slot reservation.")
            select_index = reserved if reserved is not None else self._take_device_slots(size)
            try:
                row_idx = self._get_free_row(seq_id)
                self.buffer_req_to_token_slots[row_idx, cur_len:cur_len + size] = select_index
                self.row_seq_lens[row_idx] = cur_len + size
                self.row_logical_lens[row_idx] = old_logical_len + size
            except BaseException:
                row_idx = self.seq_id_to_row.get(seq_id)
                if row_idx is not None:
                    self.row_seq_lens[row_idx] = cur_len
                    self.row_logical_lens[row_idx] = old_logical_len
                    self.buffer_req_to_token_slots[row_idx, cur_len:cur_len + size] = 0
                if reserved is None:
                    self._return_prefix_device_slots(select_index)
                if existing_row is None and seq_id in self.seq_id_to_row:
                    self.free_rows.appendleft(self.seq_id_to_row.pop(seq_id))
                raise
            if reserved is not None:
                del reservations[seq_id]
            return select_index

    def _ensure_decode_buffers(self, batch_size: int):
        if not hasattr(self, "_decode_buf_capacity") or self._decode_buf_capacity < batch_size:
            cap = max(batch_size, getattr(self, "_decode_buf_capacity", 0) * 2, 64)
            self._decode_buf_capacity = cap
            pin_memory = device_runtime.supports_pin_memory()
            self._pinned_input_ids = torch.empty(cap, dtype=torch.int64, pin_memory=pin_memory)
            self._pinned_positions = torch.empty(cap, dtype=torch.int64, pin_memory=pin_memory)
            self._pinned_context_lens = torch.empty(cap, dtype=torch.int32, pin_memory=pin_memory)
            self._pinned_req_indices = torch.empty(cap, dtype=torch.int32, pin_memory=pin_memory)
            self._cuda_input_ids = torch.empty(cap, dtype=torch.int64, device=self.device)
            self._cuda_positions = torch.empty(cap, dtype=torch.int64, device=self.device)
            self._cuda_context_lens = torch.empty(cap, dtype=torch.int32, device=self.device)
            self._cuda_req_indices = torch.empty(cap, dtype=torch.int32, device=self.device)
            self._cuda_slot_mapping = torch.empty(cap, dtype=torch.int32, device=self.device)
            self._static_rows_gpu = torch.empty(cap, dtype=torch.long, device=self.device)
            self._static_cols_gpu = torch.empty(cap, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def _allocate_batch(self, seq_ids: list[int], size: int) -> torch.Tensor:
        assert size == 1, "Batch allocation currently only supports size=1 (Decode)"
        batch_size = len(seq_ids)
        row_indices, pending_rows = self._plan_decode_rows(seq_ids)
        self._ensure_decode_buffers(batch_size)
        cur_lens = self.row_seq_lens[row_indices]
        logical_lens = self.row_logical_lens[row_indices]
        if np.any(cur_lens >= self.buffer_req_to_token_slots.shape[1]):
            raise ValueError("Decode allocation exceeds the cache row capacity.")
        select_indices = self._take_device_slots(batch_size)
        rows_committed = False
        try:
            self._commit_decode_rows(pending_rows)
            rows_committed = True
            rows_gpu = self._static_rows_gpu[:batch_size]
            cols_gpu = self._static_cols_gpu[:batch_size]
            rows_gpu.copy_(torch.as_tensor(row_indices, dtype=torch.long), non_blocking=True)
            cols_gpu.copy_(torch.as_tensor(cur_lens, dtype=torch.long), non_blocking=True)
            self.buffer_req_to_token_slots[rows_gpu, cols_gpu] = select_indices
            self.row_seq_lens[row_indices] += 1
            self.row_logical_lens[row_indices] += 1
        except BaseException as error:
            try:
                if rows_committed or not pending_rows:
                    for row_idx, cur_len in zip(row_indices, cur_lens, strict=True):
                        self.buffer_req_to_token_slots[int(row_idx), int(cur_len)] = 0
                self.row_seq_lens[row_indices] = cur_lens
                self.row_logical_lens[row_indices] = logical_lens
                if device_runtime.supports_streams(self.device):
                    device_runtime.synchronize()
                if rows_committed:
                    self._rollback_decode_rows(pending_rows)
                elif any(seq_id in self.seq_id_to_row for seq_id, _ in pending_rows):
                    raise RuntimeError(
                        "Static decode allocation partially committed its row plan."
                    )
                self._return_prefix_device_slots(select_indices)
            except BaseException as rollback_error:
                quarantined = getattr(self, "_quarantined_decode_allocations", None)
                if quarantined is None:
                    quarantined = []
                    self._quarantined_decode_allocations = quarantined
                quarantined.append((tuple(int(seq_id) for seq_id in seq_ids), select_indices))
                error.add_note(
                    "Decode allocation rollback could not fence and restore metadata; "
                    "slot ownership was quarantined instead of being recycled."
                )
                raise error from rollback_error
            raise

        return select_indices

    def _plan_decode_rows(
        self,
        seq_ids: list[int] | np.ndarray,
    ) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
        try:
            rows = np.asarray(
                [self.seq_id_to_row[seq_id] for seq_id in seq_ids],
                dtype=np.int64,
            )
            return rows, ()
        except KeyError:
            pass

        rows = np.empty(len(seq_ids), dtype=np.int64)
        pending: list[tuple[int, int]] = []
        free_rows = iter(self.free_rows)
        for index, seq_id in enumerate(seq_ids):
            row = self.seq_id_to_row.get(seq_id)
            if row is None:
                try:
                    row = next(free_rows)
                except StopIteration as error:
                    raise RuntimeError(
                        "No free rows for static decode batch: "
                        f"need={len(pending) + 1} free={len(self.free_rows)}."
                    ) from error
                pending.append((int(seq_id), row))
            rows[index] = row
        return rows, tuple(pending)

    def _commit_decode_rows(
        self,
        pending: tuple[tuple[int, int], ...],
    ) -> None:
        if not pending:
            return
        expected = [row for _, row in pending]
        actual = list(islice(self.free_rows, len(pending)))
        if actual != expected:
            raise RuntimeError(
                "Static decode row plan changed before commit: "
                f"expected={expected} actual={actual}."
            )
        for seq_id, _ in pending:
            self.seq_id_to_row[seq_id] = self.free_rows.popleft()

    def _rollback_decode_rows(
        self,
        pending: tuple[tuple[int, int], ...],
    ) -> None:
        for seq_id, expected_row in reversed(pending):
            actual_row = self.seq_id_to_row.get(seq_id)
            if actual_row != expected_row:
                raise RuntimeError(
                    "Static decode row rollback found unexpected ownership: "
                    f"seq_id={seq_id} expected={expected_row} actual={actual_row}."
                )
            self.seq_id_to_row.pop(seq_id)
            self.free_rows.appendleft(expected_row)

    @torch.no_grad()
    def _allocate_decode_batch_static(
        self,
        seq_ids: list[int],
        *,
        row_indices: np.ndarray,
        pending_rows: tuple[tuple[int, int], ...],
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        batch_size = len(seq_ids)
        if not hasattr(self, "row_logical_lens"):
            self.row_logical_lens = self.row_seq_lens.copy()
        if row_indices.shape != (batch_size,):
            raise ValueError(
                "Static decode reservation rows must match the active batch: "
                f"shape={row_indices.shape} batch={batch_size}."
            )
        if np.any(self.row_seq_lens[row_indices] >= self.buffer_req_to_token_slots.shape[1]):
            raise ValueError("Decode allocation exceeds the cache row capacity.")
        select_indices = self._take_device_slots(batch_size)
        try:
            self._commit_decode_rows(pending_rows)
        except BaseException:
            self._return_prefix_device_slots(select_indices)
            raise
        self.row_seq_lens[row_indices] += 1
        self.row_logical_lens[row_indices] += 1

        return select_indices, self.row_seq_lens[row_indices], row_indices

    @cpu_timing.timed
    def free_seq(self, seq_id: int):
        with profiler.record("cache_free_seq"):
            self._poll_prefix_offload()
            debug_slots = debug_slots_enabled()
            row_idx = self.seq_id_to_row.pop(seq_id, None)
            if row_idx is None:
                self.seq_id_to_cached_ranges.pop(seq_id, None)
                released = self._release_prefix_request(seq_id)
                if released:
                    self._schedule_write_through_prefix_blocks(released)
                return

            cur_len = self.row_seq_lens[row_idx]
            cached_ranges = _merge_ranges(self.seq_id_to_cached_ranges.pop(seq_id, []))

            if cur_len < 0:
                raise RuntimeError(
                    f"KV cache row length became negative for seq_id={seq_id}: {cur_len}."
                )
            before_free = self._num_free_slots
            freed_tokens = 0
            for start, end in _complement_ranges(0, int(cur_len), cached_ranges):
                slots = self.buffer_req_to_token_slots[row_idx, start:end]
                count = int(end - start)
                ptr = self._num_free_slots
                self.free_slots_stack[ptr: ptr + count] = slots
                self._num_free_slots += count
                freed_tokens += count
            released_prefix_blocks = self._release_prefix_request(seq_id)
            after_free = self._num_free_slots

            self.buffer_req_to_token_slots[row_idx, :] = 0
            self.row_seq_lens[row_idx] = 0
            self.row_logical_lens[row_idx] = 0
            self.free_rows.append(row_idx)
            # Host pressure/submission can fail. The request must already be
            # fully detached, so a retry cannot leak its row or double-free KV.
            self._schedule_write_through_prefix_blocks(released_prefix_blocks)

            if debug_slots:
                logger.info(
                    "free_seq seq_id={} row_idx={} freed_tokens={} free_slots_before={} free_slots_after={}",
                    seq_id,
                    row_idx,
                    int(freed_tokens),
                    int(before_free),
                    int(after_free),
                )
            if log_level == 'DEBUG': logger.debug(f'free seq {row_idx} with {cur_len} tokens')

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
        # All KV layers share one slot allocator. Repeat its accounting along
        # the chain's layer axis; never sum it as independent physical pools.
        required = max(0, int(suffix_tokens)) + max(0, int(generation_tokens) - 1)
        if needs_resident_row:
            required += max(existing_slots_by_layer, default=0)
        available = max(
            0, self._num_free_slots - max(outstanding_reserved_slots_by_layer, default=0)
        )
        required_rows = int(needs_resident_row)
        available_rows = max(0, len(self.free_rows) - outstanding_reserved_rows)
        return (
            (required,) * self.num_kv_layers,
            required_rows,
            (max(0, required - available),) * self.num_kv_layers,
            max(0, required_rows - available_rows),
        )

    def chain_physical_residency(self, seq_id: int) -> tuple[int, ...]:
        row = self.seq_id_to_row.get(int(seq_id))
        if row is None:
            raise RuntimeError(f"Missing chain row for seq_id={seq_id}.")
        return (int(self.row_seq_lens[row]),) * self.num_kv_layers

    def chain_has_residency(self, seq_id: int) -> bool:
        return int(seq_id) in self.seq_id_to_row

    def chain_physical_kv_len(self, layer_idx: int, seq_id: int) -> int:
        return self.chain_physical_residency(seq_id)[self.kv_layer_index(layer_idx)]

    def debug_live_seq_slots(self) -> dict[int, int]:
        return {
            int(seq_id): int(self.row_seq_lens[row_idx])
            for seq_id, row_idx in self.seq_id_to_row.items()
            if int(self.row_seq_lens[row_idx]) > 0
        }

    def free_part_slots(self, layer_idx: int, seq: Sequence, keep_indices: torch.Tensor):
        raise ValueError('不需要实现该方法')

    def _attach_prefill_prefixes(self, seqs):
        # A promotion for an earlier request must not evict a later admitted hit.
        # Maintenance snapshots already have their complete chains pinned.
        protected = {block.stable_block_id: block for seq in seqs
                     if seq.seq_id not in self.seq_id_to_prefix_blocks
                     and getattr(seq, "prefix_prune_prefix", None) is None
                     for block in self._prefix_hit_chain(seq)}
        for block in protected.values():
            self.prefix_cache.acquire_block_ref(block)
        try:
            for seq in seqs:
                self._attach_prefix_cache_if_needed(seq)
        finally:
            self._release_prefix_blocks(list(protected.values()))

    def _prepare_prefill(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_prefill"):
            self._poll_prefix_offload()
            self._prefix_offload_step_h2d_operations = {}
            self._attach_prefill_prefixes(seqs)

            total_chunk_tokens = sum(seq.current_chunk_size for seq in seqs)

            input_ids_np = np.empty(total_chunk_tokens, dtype=np.int64)
            positions_np = np.empty(total_chunk_tokens, dtype=np.int64)
            cu_seqlens_q = [0]

            slot_mapping = torch.empty(total_chunk_tokens, dtype=torch.int32, device=self.device)
            context_lens_list = []
            req_indices = []

            token_offset = 0
            for seq in seqs:
                chunk_size = seq.current_chunk_size
                start_idx = seq.num_prefilled_tokens
                end_idx = start_idx + chunk_size

                if seq.seq_id in self.seq_id_to_row:
                    row_idx = self.seq_id_to_row[seq.seq_id]
                    if self.row_logical_lens[row_idx] != start_idx:
                        raise ValueError(
                            "KV cache logical row length mismatch in prefill: "
                            f"seq_id={seq.seq_id} row_logical_len={self.row_logical_lens[row_idx]} "
                            f"start_idx={start_idx}"
                        )

                resident_start = (
                    0
                    if seq.seq_id not in self.seq_id_to_row
                    else int(self.row_seq_lens[self.seq_id_to_row[seq.seq_id]])
                )
                allocated_slots = self._allocate(seq.seq_id, chunk_size)
                row_idx = self.seq_id_to_row[seq.seq_id]
                resident_end = resident_start + chunk_size
                slot_mapping[token_offset: token_offset + chunk_size] = self.buffer_req_to_token_slots[
                    row_idx, resident_start:resident_end
                ]
                context_lens_list.append(resident_end)
                req_indices.append(row_idx)

                chunk_tokens = seq.token_ids
                if len(chunk_tokens) > chunk_size:
                    chunk_tokens = chunk_tokens[start_idx:end_idx]
                chunk_tokens = list(chunk_tokens)

                input_ids_np[token_offset: token_offset + chunk_size] = chunk_tokens
                positions_np[token_offset: token_offset + chunk_size] = (
                    np.arange(start_idx, end_idx) - int(getattr(seq, "position_offset", 0))
                )
                self._record_prefix_materialization(seq, chunk_tokens, allocated_slots)

                cu_seqlens_q.append(cu_seqlens_q[-1] + chunk_size)
                token_offset += chunk_size

            # Pack metadata into two owned staging allocations. CUDA copies use
            # pinned storage and do not wait for the preceding model step. Fresh
            # allocations avoid overwriting inputs still used by an async step;
            # PyTorch's pinned allocator tracks the copy stream before reuse.
            pin = self.device.type == "cuda" and device_runtime.supports_pin_memory()
            host_tokens = torch.empty((2, total_chunk_tokens), dtype=torch.int64, pin_memory=pin)
            host_tokens[0].numpy()[:] = input_ids_np
            host_tokens[1].numpy()[:] = positions_np
            host_metadata = torch.empty(3 * len(seqs) + 1, dtype=torch.int32, pin_memory=pin)
            host_metadata.numpy()[:] = context_lens_list + req_indices + cu_seqlens_q
            device_tokens = host_tokens.to(self.device, non_blocking=pin)
            device_metadata = host_metadata.to(self.device, non_blocking=pin)
            context_lens = device_metadata[:len(seqs)]
            req_indices_tensor = device_metadata[len(seqs):2 * len(seqs)]

            self.layer_batch_state.slot_mapping = slot_mapping
            self.layer_batch_state.context_lens = context_lens
            self.layer_batch_state.max_context_len = max(context_lens_list) if context_lens_list else 0
            self.layer_batch_state.req_indices = req_indices_tensor
            self._validate_attention_slot_mapping(slot_mapping)

            # Host metadata for prepared per-request prefill providers; no D2H reads.
            self.prefill_plan = tuple(
                (cu_seqlens_q[i], seq.current_chunk_size, req_indices[i], context_lens_list[i])
                for i, seq in enumerate(seqs)
            )

            if log_level == 'DEBUG':
                logger.debug(f'{context_lens_list=}   {req_indices=}  {slot_mapping[:10].tolist()=}  {slot_mapping[-10:].tolist()=}')

            input_ids, positions = device_tokens.unbind(0)
            cu_seqlens_q = device_metadata[2 * len(seqs):]
            return input_ids, positions, cu_seqlens_q

    def _prepare_decode(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_decode"):
            self._poll_prefix_offload()
            self._prefix_offload_step_h2d_operations = {}
            batch_size = len(seqs)
            self._ensure_decode_buffers(batch_size)

            input_ids_list = [seq.decode_input_token for seq in seqs]
            positions_list = [seq.decode_input_position for seq in seqs]
            seq_ids = [seq.seq_id for seq in seqs]

            new_slots_batch = self._allocate_batch(seq_ids, 1)
            row_indices = [self.seq_id_to_row[sid] for sid in seq_ids]
            for seq, slot in zip(seqs, new_slots_batch):
                self._record_prefix_materialization(seq, [seq.decode_input_token], slot.reshape(1))

            self._pinned_context_lens[:batch_size].copy_(
                torch.as_tensor(self.row_seq_lens[row_indices], dtype=torch.int32)
            )
            self._pinned_req_indices[:batch_size].copy_(
                torch.as_tensor(row_indices, dtype=torch.int32)
            )
            self._pinned_input_ids[:batch_size].copy_(
                torch.as_tensor(input_ids_list, dtype=torch.int64)
            )
            self._pinned_positions[:batch_size].copy_(
                torch.as_tensor(positions_list, dtype=torch.int64)
            )

            context_lens = self._cuda_context_lens[:batch_size]
            context_lens.copy_(self._pinned_context_lens[:batch_size], non_blocking=True)
            req_indices = self._cuda_req_indices[:batch_size]
            req_indices.copy_(self._pinned_req_indices[:batch_size], non_blocking=True)

            slot_mapping = self._cuda_slot_mapping[:batch_size]
            slot_mapping.copy_(new_slots_batch, non_blocking=True)

            self.layer_batch_state.slot_mapping = slot_mapping
            self.layer_batch_state.context_lens = context_lens
            self.layer_batch_state.max_context_len = int(self._pinned_context_lens[:batch_size].max().item()) if row_indices else 0
            self.layer_batch_state.req_indices = req_indices
            self._validate_attention_slot_mapping(slot_mapping)

            if log_level == 'DEBUG':
                logger.debug(f'{slot_mapping=}   {context_lens.tolist()=}  {slot_mapping[:10]=}  {slot_mapping[-10:]=}')

            input_ids = self._cuda_input_ids[:batch_size]
            input_ids.copy_(self._pinned_input_ids[:batch_size], non_blocking=True)
            positions = self._cuda_positions[:batch_size]
            positions.copy_(self._pinned_positions[:batch_size], non_blocking=True)
            return input_ids, positions, None

    def before_prefill_layer_attention(
        self,
        layer_idx: int,
        selection: SparseSelection,
    ):
        controller = getattr(self, "prefix_offload_controller", None)
        if controller is not None and self._prefix_offload_step_h2d_operations:
            if device_runtime.is_stream_capturing():
                raise RuntimeError("Prefix H2D waits are forbidden during graph capture.")
            kv_layer_index = self.kv_layer_index(layer_idx)
            with profiler.record("prefix_cache_h2d_layer_wait"):
                for operation in self._prefix_offload_step_h2d_operations.values():
                    controller.wait_for_layer(operation, kv_layer_index)
        return super().before_prefill_layer_attention(layer_idx, selection)

    def prepare_decode_static(
        self,
        seqs: list[Sequence],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        req_indices: torch.Tensor,
    ):
        """Prepare decode metadata into caller-owned static CUDA buffers.

        Used by CUDA Graph decode replay: tensor addresses must stay stable, so
        this avoids the ordinary per-step metadata tensor allocation path.
        """
        return self._prepare_decode_graph_buffers(
            seqs,
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            req_indices=req_indices,
            publish_slots_outside_graph=True,
        )

    def prepare_decode_graph_step(
        self,
        seqs: list[Sequence],
        state: CacheDecodeGraphState,
    ):
        inputs = state.inputs
        return self._prepare_decode_graph_buffers(
            seqs,
            input_ids=inputs.input_ids,
            positions=inputs.positions,
            slot_mapping=inputs.write_slot_mapping,
            context_lens=inputs.context_lens,
            req_indices=inputs.request_indices,
            active_mask=inputs.active_mask,
            host_inputs=inputs.host,
            padding_write_slot=int(state.contract.padding.write_slot),
            padding_active=bool(state.contract.padding.active),
            mirror_first_real_row_for_reads=bool(
                state.contract.padding.mirror_first_real_row_for_reads
            ),
            context_capacity=int(state.contract.context_capacity),
        )

    def prepare_decode_graph_in(self, state: CacheDecodeGraphState) -> None:
        """Publish reservations before provider graph-in preparation consumes them."""

        from sparseengine.kernels.triton.decode_graph_metadata import (
            publish_decode_graph_slots,
        )

        inputs = state.inputs
        publish_decode_graph_slots(
            self.buffer_req_to_token_slots,
            inputs.request_indices,
            inputs.context_lens,
            inputs.write_slot_mapping,
            inputs.active_mask,
        )

    def _prepare_decode_graph_buffers(
        self,
        seqs: list[Sequence],
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        req_indices: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        host_inputs: DecodeGraphHostInputs | None = None,
        padding_write_slot: int = -1,
        padding_active: bool = False,
        mirror_first_real_row_for_reads: bool = True,
        context_capacity: int | None = None,
        publish_slots_outside_graph: bool = False,
    ):
        with profiler.record("cache_prepare_decode"):
            self._poll_prefix_offload()
            self._prefix_offload_step_h2d_operations = {}
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
            if active_mask is not None and active_mask.numel() != graph_batch_size:
                raise ValueError(
                    "Static decode active_mask must match the graph batch size."
                )
            if not mirror_first_real_row_for_reads:
                raise ValueError(
                    "StandardCacheManager requires padded read rows to mirror the "
                    "first real request."
                )

            if host_inputs is None:
                input_ids_list = [seq.decode_input_token for seq in seqs]
                positions_list = [seq.decode_input_position for seq in seqs]
                seq_ids = [seq.seq_id for seq in seqs]
            else:
                seq_ids = host_inputs.pack_requests(seqs)
                input_ids_list = None
                positions_list = None

            prospective_rows, pending_rows = self._plan_decode_rows(seq_ids)
            if context_capacity is not None:
                max_requested_context_len = int(
                    (self.row_seq_lens[prospective_rows] + 1).max()
                )
                if max_requested_context_len > context_capacity:
                    raise ValueError(
                        "Decode request exceeded the captured graph context capacity: "
                        f"requested={max_requested_context_len} "
                        f"captured={context_capacity}."
                    )

            new_slots_batch, real_context_lens, row_indices = self._allocate_decode_batch_static(
                seq_ids,
                row_indices=prospective_rows,
                pending_rows=pending_rows,
            )
            for seq, slot in zip(seqs, new_slots_batch):
                self._record_prefix_materialization(seq, [seq.decode_input_token], slot.reshape(1))

            slot_mapping[:real_batch_size].copy_(new_slots_batch)
            if host_inputs is None:
                assert input_ids_list is not None
                assert positions_list is not None
                input_ids[:real_batch_size].copy_(
                    torch.tensor(input_ids_list, dtype=torch.int64)
                )
                positions[:real_batch_size].copy_(
                    torch.tensor(positions_list, dtype=torch.int64)
                )
                context_lens[:real_batch_size].copy_(
                    torch.from_numpy(real_context_lens.astype(np.int32, copy=False))
                )
                req_indices[:real_batch_size].copy_(
                    torch.from_numpy(row_indices.astype(np.int32, copy=False))
                )
            else:
                host_inputs.pack_cache_facts(
                    context_lens=real_context_lens,
                    request_indices=row_indices,
                    real_batch_size=real_batch_size,
                    padding_active=padding_active,
                )
                non_blocking = bool(host_inputs.input_ids.is_pinned())
                input_ids[:real_batch_size].copy_(
                    host_inputs.input_ids[:real_batch_size],
                    non_blocking=non_blocking,
                )
                positions[:real_batch_size].copy_(
                    host_inputs.positions[:real_batch_size],
                    non_blocking=non_blocking,
                )
                context_lens[:real_batch_size].copy_(
                    host_inputs.context_lens[:real_batch_size],
                    non_blocking=non_blocking,
                )
                req_indices[:real_batch_size].copy_(
                    host_inputs.request_indices[:real_batch_size],
                    non_blocking=non_blocking,
                )
                assert active_mask is not None
                active_mask[:real_batch_size].copy_(
                    host_inputs.active_mask[:real_batch_size],
                    non_blocking=non_blocking,
                )

            if graph_batch_size > real_batch_size:
                # CUDA Graph replay is shape-static. Padded rows mirror the first
                # real request for read-only work, but use the contract's safe
                # write sentinel so they never consume persistent cache capacity.
                first_context_len = int(real_context_lens[0])
                first_row_idx = int(row_indices[0])
                if host_inputs is None:
                    assert input_ids_list is not None
                    assert positions_list is not None
                    first_input_id = int(input_ids_list[0])
                    first_position = int(positions_list[0])
                else:
                    first_input_id = int(host_inputs.input_ids[0])
                    first_position = int(host_inputs.positions[0])
                input_ids[real_batch_size:].fill_(first_input_id)
                positions[real_batch_size:].fill_(first_position)
                slot_mapping[real_batch_size:].fill_(padding_write_slot)
                context_lens[real_batch_size:].fill_(first_context_len)
                req_indices[real_batch_size:].fill_(first_row_idx)
                if active_mask is not None:
                    active_mask[real_batch_size:].fill_(padding_active)

            if publish_slots_outside_graph:
                from sparseengine.kernels.triton.decode_graph_metadata import (
                    publish_decode_graph_slots,
                )

                publish_decode_graph_slots(
                    self.buffer_req_to_token_slots,
                    req_indices[:real_batch_size],
                    context_lens[:real_batch_size],
                    slot_mapping[:real_batch_size],
                )

            self.layer_batch_state.slot_mapping = slot_mapping
            self.layer_batch_state.context_lens = context_lens
            self.layer_batch_state.max_context_len = int(real_context_lens.max()) if real_batch_size > 0 else 0
            self.layer_batch_state.req_indices = req_indices
            self.validate_decode_cuda_graph_slot_mappings()

            return input_ids, positions, None
