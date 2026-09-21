from __future__ import annotations

import os
import time
from bisect import bisect_left
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
from sparseengine.engine.prefix_cache import (
    PrefixCacheBlock,
    PrefixTransferKind,
    RadixPrefixIndex,
    build_prefix_cache_fingerprint,
    select_write_through_candidates,
    usable_prefix_cache_tokens,
)
from sparseengine.engine.prefix_prune import (
    PrefixPruneRecord,
    normalize_prefix_prune_ranges,
)
from sparseengine.engine.sequence import Sequence
from sparseengine.kernels.triton.prefill_score import prefill_score_fwd
from sparseengine.platforms import device_runtime
from sparseengine.utils.log import log_level, logger
from sparseengine.utils.profiler import cpu_timing, profiler

from .base import (
    AttentionCacheWrite,
    AttentionPayload,
    CacheManager,
    ExplicitKVPayload,
    LayerBatchStates,
    PrefillComputeView,
    PrefillScoreRequest,
    SparseSelection,
)
from .offload.prefix_components import (
    ComponentPrefixOffloadController,
    ComponentPrefixPool,
    prefix_block_bytes,
    storage_prefix_components,
)
from .prefix_cache_mixin import PrefixCacheMixin
from .prefix_offload import (
    PinnedPrefixKVPool,
    PrefixH2DOperation,
    PrefixOffloadController,
    StandardPrefixOffloadController,
)
from .storage import (
    ExplicitKVStorage,
    HeterogeneousExplicitKVStorage,
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


class StandardCacheManager(PrefixCacheMixin, CacheManager):

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
            config.enable_prefix_caching and config.sparse_method in ("", "omnikv")
            and getattr(config, "resolved_prefix_cache_mode", "radix") == "radix"
            and not getattr(getattr(config, "runtime_layout", None), "linear_attention_layer_indices", ())
        )
        self.prefix_cache_block_size = int(config.prefix_cache_block_size)
        self.prefix_cache: RadixPrefixIndex | None = None
        if self.enable_prefix_caching:
            self.prefix_cache = RadixPrefixIndex(
                block_size=self.prefix_cache_block_size,
                fingerprint=build_prefix_cache_fingerprint(config, self.prefix_cache_block_size),
                max_blocks=config.prefix_cache_max_blocks,
            )
        self.seq_id_to_prefix_blocks: dict[int, list[PrefixCacheBlock]] = {}
        self.seq_id_to_cached_ranges: dict[int, list[tuple[int, int]]] = {}
        self._scheduler_capacity_snapshot_depth = 0
        self._scheduler_freeable_block_ids: frozenset[bytes] | None = None
        self._prefix_resident_slots_cache: tuple[
            RadixPrefixIndex, int, dict[int, tuple[frozenset[bytes], int]]
        ] | None = None
        self._scheduler_reclaimable_slots: int | None = None
        self._init_prefix_cache_runtime()
        self.prefix_offload_controller: PrefixOffloadController | None = None
        self._prefix_offload_step_h2d_operations: dict[int, PrefixH2DOperation] = {}
        self._prefix_write_through_candidates: dict[bytes, PrefixCacheBlock] = {}
        self._prefix_prune_scoring: dict[str, object] | None = None
        has_linear_layers = bool(
            getattr(getattr(config, "runtime_layout", None), "linear_attention_layer_indices", ())
        )
        if bool(getattr(config, "enable_prefix_cache_offload", False)) and not has_linear_layers:
            self._init_prefix_offload()

    def _init_prefix_offload(self) -> None:
        if not self.enable_prefix_caching or self.prefix_cache is None:
            raise RuntimeError("Prefix cache offload requires the Standard prefix cache.")
        if self.tp_size not in (1, 2):
            raise RuntimeError("Prefix cache offload currently supports only TP=1 or TP=2.")
        if not device_runtime.supports_pin_memory():
            raise RuntimeError("Prefix cache offload requires pinned host memory support.")
        if not device_runtime.supports_streams(self.device):
            raise RuntimeError("Prefix cache offload requires asynchronous device streams.")
        host_size_gb = getattr(self.config, "prefix_cache_host_size_gb", None)
        if host_size_gb is None:
            raise RuntimeError("Prefix cache offload requires prefix_cache_host_size_gb.")
        storage = self.attention_cache_storage
        components = storage_prefix_components(storage, self.num_kv_layers)
        bytes_per_block = prefix_block_bytes(components, self.prefix_cache_block_size)
        host_bytes = int(float(host_size_gb) * (1024**3))
        host_capacity_blocks = host_bytes // bytes_per_block
        gpu_capacity_blocks = int(self.config.num_kvcache_slots) // self.prefix_cache_block_size
        required_blocks = gpu_capacity_blocks
        if self.prefix_cache.max_blocks is not None:
            required_blocks = min(required_blocks, int(self.prefix_cache.max_blocks))
        if host_capacity_blocks < required_blocks:
            raise RuntimeError(
                "Prefix host tier is too small for write-through safety: "
                f"host_blocks={host_capacity_blocks} required_blocks={required_blocks} "
                f"bytes_per_block={bytes_per_block} host_size_gb={host_size_gb}."
            )
        if not isinstance(storage, ExplicitKVStorage):
            host_pool = ComponentPrefixPool(
                components=components,
                capacity_blocks=host_capacity_blocks,
                block_size=self.prefix_cache_block_size,
            )
            self.prefix_offload_controller = ComponentPrefixOffloadController(
                components=components,
                prefix_cache=self.prefix_cache,
                host_pool=host_pool,
                block_size=self.prefix_cache_block_size,
                device=self.device,
            )
            return
        host_pool = PinnedPrefixKVPool(
            capacity_blocks=host_capacity_blocks,
            num_layers=self.num_kv_layers,
            block_size=self.prefix_cache_block_size,
            num_kv_heads=storage.num_kv_heads,
            head_dim=storage.head_dim,
            dtype=storage.dtype,
        )
        self.prefix_offload_controller = StandardPrefixOffloadController(
            prefix_cache=self.prefix_cache,
            kv_cache=storage.cache,
            host_pool=host_pool,
            block_size=self.prefix_cache_block_size,
            device=self.device,
        )

    def _prefix_offload_enabled(self) -> bool:
        return getattr(self, "prefix_offload_controller", None) is not None

    def _poll_prefix_offload(self) -> None:
        controller = getattr(self, "prefix_offload_controller", None)
        if controller is not None:
            controller.poll()

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

    def _require_uniform_explicit_storage(self, operation: str) -> ExplicitKVStorage:
        storage = self.attention_cache_storage
        if not isinstance(storage, ExplicitKVStorage):
            raise NotImplementedError(
                f"{operation} does not support heterogeneous per-layer KV shapes."
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

    def _require_prefix_cache(self) -> RadixPrefixIndex:
        if getattr(self, "prefix_cache", None) is None:
            raise RuntimeError("prefix cache is not enabled for this cache manager.")
        return self.prefix_cache

    def prefix_cache_inspect(
        self,
        token_ids: list[int],
        *,
        include_subtree: bool = False,
    ) -> dict[str, object]:
        self._poll_prefix_offload()
        return self._require_prefix_cache().inspect_prefix(
            [int(token_id) for token_id in token_ids],
            include_subtree=include_subtree,
        )

    def prefix_cache_match(self, token_ids: list[int]) -> dict[str, object]:
        self._poll_prefix_offload()
        if getattr(self, "prefix_cache", None) is None:
            return {
                "supported": True,
                "enabled": False,
                "method": str(getattr(self.config, "sparse_method", "") or ""),
                "matched_tokens": 0,
                "matched_blocks": 0,
                "match_ratio": 0.0,
                "reason": "prefix cache is not enabled for this cache manager.",
            }
        token_ids = [int(token_id) for token_id in token_ids]
        usable_tokens = usable_prefix_cache_tokens(len(token_ids), self.prefix_cache_block_size)
        hit_len, hit_last_block_id, hit_blocks = self.prefix_cache.match_longest_prefix(
            token_ids,
            max_usable_tokens=usable_tokens,
        )
        resident_kv_tokens = 0
        if hit_last_block_id is not None and hit_blocks > 0:
            resident_kv_tokens = sum(
                self._block_resident_tokens_or_full(block)
                for block in self.prefix_cache.get_chain(
                    hit_last_block_id, hit_blocks
                )
            )
        return {
            "supported": True,
            "enabled": True,
            "method": str(getattr(self.config, "sparse_method", "") or ""),
            "block_size": int(self.prefix_cache_block_size),
            "prompt_tokens": int(len(token_ids)),
            "usable_tokens": int(usable_tokens),
            "matched_tokens": int(hit_len),
            "matched_blocks": int(hit_blocks),
            "resident_kv_tokens": int(resident_kv_tokens),
            "match_ratio": 0.0 if usable_tokens <= 0 else float(hit_len) / float(usable_tokens),
            "last_block_id": None if hit_last_block_id is None else hit_last_block_id.hex(),
            "live_blocks": int(len(self.prefix_cache)),
        }

    def prefix_cache_delete_subtree(self, token_ids: list[int]) -> dict[str, object]:
        controller = getattr(self, "prefix_offload_controller", None)
        if controller is not None:
            controller.synchronize_all()
        normalized = [int(token_id) for token_id in token_ids]
        prefix_cache = self._require_prefix_cache()
        plan = prefix_cache.preview_delete_subtree(normalized)
        self.synchronize_prefix_cache_delete_plan(plan.to_dict())
        result = prefix_cache.safe_delete_subtree(normalized)
        self._free_prefix_cache_blocks(result.deleted_blocks)
        return result.to_dict()

    def prefix_cache_set_eviction_priority(
        self,
        token_ids: list[int],
        *,
        priority: int,
    ) -> dict[str, object]:
        return self._require_prefix_cache().set_subtree_eviction_priority(
            [int(token_id) for token_id in token_ids],
            int(priority),
        )

    def _prefix_evictable_slots(self) -> int:
        if getattr(self, "prefix_cache", None) is None:
            return 0
        block_ids = (
            self.prefix_cache.device_freeable_block_ids()
            if self._prefix_offload_enabled()
            else self._prefix_freeable_block_ids_for_capacity()
        )
        return self._prefix_resident_slots_for_ids(block_ids)

    def _prefix_resident_slots_for_ids(self, block_ids: frozenset[bytes]) -> int:
        if self.prefix_cache is None:
            return 0
        # Index capacity epochs cover membership, references, priority,
        # transfers and payload compaction. Keep weighted totals across
        # scheduler passes as well as the index's immutable ID sets.
        cache = getattr(self, "_prefix_resident_slots_cache", None)
        epoch = self.prefix_cache.capacity_epoch
        if cache is None or cache[0] is not self.prefix_cache or cache[1] != epoch:
            cache = (self.prefix_cache, epoch, {})
            self._prefix_resident_slots_cache = cache
        totals = cache[2]
        # The index reuses these immutable sets within an epoch. Identity keys
        # avoid linear equality checks between equal offload capacity views;
        # retaining each set prevents object-ID reuse while it is cached.
        key = id(block_ids)
        if key in totals:
            return totals[key][1]
        total = 0
        for block_id in block_ids:
            block = self.prefix_cache.get_block(block_id)
            if block is None or not block.residency.device_present:
                continue
            total += self._block_resident_tokens_or_full(block)
        totals[key] = (block_ids, int(total))
        return int(total)

    def _prefix_freeable_block_ids_for_capacity(self) -> frozenset[bytes]:
        if self.prefix_cache is None:
            return frozenset()
        if self._scheduler_capacity_snapshot_depth <= 0:
            return self.prefix_cache.freeable_block_ids()
        if self._scheduler_freeable_block_ids is None:
            self._scheduler_freeable_block_ids = (
                self.prefix_cache.freeable_block_ids()
            )
        return self._scheduler_freeable_block_ids

    @contextmanager
    def scheduler_capacity_snapshot(self):
        """Reuse one immutable prefix-capacity view within a scheduler pass."""
        self._scheduler_capacity_snapshot_depth += 1
        if self._scheduler_capacity_snapshot_depth == 1:
            self._scheduler_freeable_block_ids = None
            self._scheduler_reclaimable_slots = None
        try:
            yield
        finally:
            self._scheduler_capacity_snapshot_depth -= 1
            if self._scheduler_capacity_snapshot_depth == 0:
                self._scheduler_freeable_block_ids = None
                self._scheduler_reclaimable_slots = None

    def _prefix_step_reclaimable_slots(self) -> int:
        if getattr(self, "prefix_cache", None) is None:
            return 0
        if (
            self._scheduler_capacity_snapshot_depth > 0
            and self._scheduler_reclaimable_slots is not None
        ):
            return self._scheduler_reclaimable_slots
        block_ids = (
            self.prefix_cache.device_reclaimable_block_ids()
            if self._prefix_offload_enabled()
            else self._prefix_freeable_block_ids_for_capacity()
        )
        reclaimable_slots = self._prefix_resident_slots_for_ids(block_ids)
        if self._scheduler_capacity_snapshot_depth > 0:
            self._scheduler_reclaimable_slots = reclaimable_slots
        return reclaimable_slots

    def _prefix_immediately_evictable_slots(self) -> int:
        if (
            getattr(self, "prefix_cache", None) is None
            or self._prefix_offload_enabled()
        ):
            return 0
        return self._prefix_resident_slots_for_ids(
            self.prefix_cache.evictable_block_ids()
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

    def prompt_admission_free_slots(self) -> int:
        return int(self.num_free_slots + self._prefix_step_reclaimable_slots())

    def prompt_admission_cost(self, seq: Sequence) -> int:
        hit_len = int(getattr(seq, "prefix_cache_hit_len", 0) or 0)
        suffix_len = int(seq.num_prompt_tokens - hit_len)
        if hit_len <= 0:
            return suffix_len
        reclaimable_slots, promotion_slots = self._prefix_hit_capacity_slots(seq)
        return suffix_len + reclaimable_slots + promotion_slots

    def _block_resident_tokens_or_full(self, block: PrefixCacheBlock) -> int:
        payload = block.payload
        if isinstance(payload, StandardPrefixBlockPayload):
            return payload.resident_tokens(self.prefix_cache_block_size)
        # Tests and external index-only users may intentionally use metadata-only payloads.
        return int(self.prefix_cache_block_size)

    @cpu_timing.timed
    def _prefix_hit_capacity_slots(self, seq: Sequence) -> tuple[int, int]:
        if self.prefix_cache is None:
            return 0, 0
        self._prefix_hit_capacity_counts(seq)
        entry = self.prefix_hit_capacity_cache.get(seq)
        if entry is not None and entry.weighted_slots is not None:
            return entry.weighted_slots
        chain = tuple(self._prefix_hit_chain(seq)) if entry is None else entry.chain
        freeable_ids = (
            self.prefix_cache.device_reclaimable_block_ids()
            if self._prefix_offload_enabled()
            else self.prefix_cache.freeable_block_ids()
        )
        reclaimable = sum(
            self._block_resident_tokens_or_full(block)
            for block in chain
            if block.stable_block_id in freeable_ids
        )
        promotion = (
            sum(
                self._block_resident_tokens_or_full(block)
                for block in chain
                if not block.residency.device_present
            )
            if self._prefix_offload_enabled()
            else 0
        )
        result = int(reclaimable), int(promotion)
        if entry is not None:
            self.prefix_hit_capacity_cache[seq] = replace(entry, weighted_slots=result)
        return result

    def _standard_payload(self, block: PrefixCacheBlock) -> StandardPrefixBlockPayload:
        payload = block.payload
        if not isinstance(payload, StandardPrefixBlockPayload):
            raise RuntimeError("Standard prefix cache block has an invalid payload.")
        return payload

    def _prefix_hit_chain(self, seq: Sequence) -> list[PrefixCacheBlock]:
        if self.prefix_cache is None or seq.prefix_cache_hit_last_block_id is None:
            return []
        return self.prefix_cache.get_chain(
            seq.prefix_cache_hit_last_block_id,
            int(seq.prefix_cache_hit_block_count),
        )

    def prompt_logical_reservation_cost(self, seq: Sequence) -> int:
        return int(self.prompt_admission_cost(seq))

    @cpu_timing.timed
    def refresh_prefix_cache_hit(self, seq: Sequence) -> None:
        self._poll_prefix_offload()
        self.clear_prefix_cache_hit(seq)
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        if seq.num_prefilled_tokens != 0 or seq.num_completion_tokens != 0:
            return
        usable_tokens = usable_prefix_cache_tokens(seq.num_prompt_tokens, self.prefix_cache_block_size)
        if usable_tokens <= 0:
            return
        with profiler.record("prefix_cache_lookup"):
            hit_len, last_block_id, hit_blocks = self._lookup_prefix_cache_hit(
                seq,
                usable_tokens,
            )
        if hit_len <= 0:
            return
        if last_block_id is None or hit_blocks <= 0:
            raise RuntimeError("Prefix cache lookup returned an invalid hit.")
        if hit_len >= seq.num_prompt_tokens or hit_len % self.prefix_cache_block_size != 0:
            raise RuntimeError(
                "Prefix cache lookup returned an unusable hit length: "
                f"seq_id={seq.seq_id} hit_len={hit_len} prompt_len={seq.num_prompt_tokens} "
                f"block_size={self.prefix_cache_block_size}."
            )
        seq.prefix_cache_enabled = True
        seq.prefix_cache_hit_len = int(hit_len)
        seq.prefix_cache_hit_block_count = int(hit_blocks)
        seq.prefix_cache_hit_last_block_id = last_block_id
        seq.prefix_cache_block_size = self.prefix_cache_block_size
        seq.prefix_cache_method = str(self.config.sparse_method or "")

    def _free_prefix_cache_blocks(self, blocks: list[PrefixCacheBlock]) -> None:
        pending = getattr(self, "_prefix_write_through_candidates", None)
        host_blocks: list[PrefixCacheBlock] = []
        device_blocks: list[PrefixCacheBlock] = []
        for block in blocks:
            if pending is not None:
                pending.pop(block.stable_block_id, None)
            payload = block.payload
            if not isinstance(payload, StandardPrefixBlockPayload):
                raise RuntimeError("Standard prefix cache block is missing token slots.")
            if block.residency.device_present:
                device_blocks.append(block)
            if block.residency.host_present:
                host_blocks.append(block)
        self._free_device_prefix_blocks(device_blocks)
        controller = getattr(self, "prefix_offload_controller", None)
        if host_blocks:
            if controller is None:
                raise RuntimeError(
                    "Prefix blocks have host payloads but no offload controller is active."
                )
            controller.free_host_payloads(host_blocks)

    def _free_device_prefix_block(self, block: PrefixCacheBlock) -> None:
        self._free_device_prefix_blocks([block])

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

    def build_prefix_kv_payload(self, seq: Sequence, block_start: int, block_end: int) -> StandardPrefixBlockPayload:
        block_start = int(block_start)
        block_end = int(block_end)
        if block_end <= block_start:
            raise ValueError(f"Invalid prefix KV payload range: {block_start}:{block_end}.")
        row_idx = self.seq_id_to_row.get(int(seq.seq_id))
        if row_idx is None:
            raise RuntimeError(f"Cannot build prefix KV payload for unknown seq_id={seq.seq_id}.")
        row_len = int(self.row_seq_lens[row_idx])
        if block_end > row_len:
            raise RuntimeError(
                "Cannot build prefix KV payload beyond materialized row length: "
                f"seq_id={seq.seq_id} block={block_start}:{block_end} row_len={row_len}."
            )
        slots = self.buffer_req_to_token_slots[row_idx, block_start:block_end].detach().to(
            dtype=torch.int32,
        ).clone()
        return StandardPrefixBlockPayload(
            token_slots=slots,
            block_start=block_start,
            block_end=block_end,
        )

    def attach_prefix_kv_payloads(self, seq: Sequence, payloads: list[object]) -> None:
        if not payloads:
            return
        seq_id = int(seq.seq_id)
        previous_row = self.seq_id_to_row.get(seq_id)
        cur_len = 0 if previous_row is None else int(self.row_seq_lens[previous_row])
        start = cur_len
        slot_parts = []
        ranges = []
        for payload in payloads:
            if not isinstance(payload, StandardPrefixBlockPayload):
                raise RuntimeError("Standard mixed prefix KV payload is missing token slots.")
            if not isinstance(payload.token_slots, torch.Tensor):
                raise RuntimeError("Standard mixed prefix KV payload has no device slots.")
            slots = payload.token_slots.to(device=self.device, dtype=torch.int32)
            if slots.ndim != 1:
                slots = slots.reshape(-1)
            count = int(slots.numel())
            if count <= 0:
                raise RuntimeError("Standard mixed prefix KV payload is empty.")
            if count % int(self.config.prefix_cache_block_size) != 0:
                raise RuntimeError(
                    f"Standard mixed prefix KV payload size must be block-aligned, got {count}."
                )
            if int(payload.block_start) != cur_len:
                raise RuntimeError(
                    "Standard mixed prefix KV payload attach must be contiguous: "
                    f"seq_id={seq_id} block_start={int(payload.block_start)} row_len={cur_len}."
                )
            end = cur_len + count
            if int(payload.block_end) not in {0, end}:
                raise RuntimeError(
                    "Standard mixed prefix KV payload has inconsistent block_end: "
                    f"payload_end={int(payload.block_end)} expected={end}."
                )
            slot_parts.append(slots)
            ranges.append((cur_len, end))
            cur_len = end
        if end > int(self.max_model_len):
            raise RuntimeError(
                "Attaching mixed prefix KV payload exceeds max_model_len: "
                f"seq_id={seq.seq_id} end={end} max_model_len={self.max_model_len}."
            )
        slots = slot_parts[0] if len(slot_parts) == 1 else torch.cat(slot_parts)
        row_idx = self._get_free_row(seq_id)
        try:
            self.buffer_req_to_token_slots[row_idx, start:end].copy_(slots)
        except BaseException:
            self.buffer_req_to_token_slots[row_idx, start:end].zero_()
            if previous_row is None:
                self.seq_id_to_row.pop(seq_id)
                self.free_rows.appendleft(row_idx)
            raise
        self.row_seq_lens[row_idx] = end
        self.row_logical_lens[row_idx] = end
        self.seq_id_to_cached_ranges.setdefault(seq_id, []).extend(ranges)

    def validate_prefix_kv_attach(self, seq: Sequence) -> bool:
        row_idx = self.seq_id_to_row.get(int(seq.seq_id))
        if row_idx is not None and int(self.row_seq_lens[row_idx]) != 0:
            raise RuntimeError(
                "Cannot attach mixed prefix KV to a non-empty row: "
                f"seq_id={seq.seq_id} row_idx={row_idx} "
                f"row_len={int(self.row_seq_lens[row_idx])}."
            )
        if row_idx is None and not self.free_rows:
            raise RuntimeError("No free rows in cache manager buffer!")
        return row_idx is not None

    def rollback_prefix_kv_attach(
        self,
        seq: Sequence,
        payloads: list[object],
        *,
        row_preexisted: bool,
    ) -> None:
        normalized: list[StandardPrefixBlockPayload] = []
        expected_start = 0
        for payload in payloads:
            if not isinstance(payload, StandardPrefixBlockPayload):
                raise RuntimeError("Standard mixed prefix rollback received an invalid payload.")
            if int(payload.block_start) != expected_start or int(payload.block_end) <= expected_start:
                raise RuntimeError(
                    "Standard mixed prefix rollback payloads are not contiguous: "
                    f"expected_start={expected_start} "
                    f"range={int(payload.block_start)}:{int(payload.block_end)}."
                )
            normalized.append(payload)
            expected_start = int(payload.block_end)
        if not normalized:
            return

        seq_id = int(seq.seq_id)
        row_idx = self.seq_id_to_row.get(seq_id)
        if row_idx is None or int(self.row_seq_lens[row_idx]) != expected_start:
            raise RuntimeError(
                "Standard mixed prefix rollback row state is inconsistent: "
                f"seq_id={seq_id} row_idx={row_idx} "
                f"row_len={None if row_idx is None else int(self.row_seq_lens[row_idx])} "
                f"expected={expected_start}."
            )
        expected_ranges = [
            (int(payload.block_start), int(payload.block_end))
            for payload in normalized
        ]
        if self.seq_id_to_cached_ranges.get(seq_id) != expected_ranges:
            raise RuntimeError(
                "Standard mixed prefix rollback cached ranges are inconsistent: "
                f"seq_id={seq_id} expected={expected_ranges} "
                f"got={self.seq_id_to_cached_ranges.get(seq_id)}."
            )

        self.buffer_req_to_token_slots[row_idx, :expected_start] = 0
        self.row_seq_lens[row_idx] = 0
        self.row_logical_lens[row_idx] = 0
        self.seq_id_to_cached_ranges.pop(seq_id, None)
        if not row_preexisted:
            owner = self.seq_id_to_row.pop(seq_id, None)
            if owner != row_idx:
                raise RuntimeError(
                    "Standard mixed prefix rollback row ownership changed unexpectedly: "
                    f"seq_id={seq_id} expected_row={row_idx} owner={owner}."
                )
            self.free_rows.appendleft(row_idx)

    def free_prefix_kv_payload(self, payload: object) -> None:
        if not isinstance(payload, StandardPrefixBlockPayload):
            raise RuntimeError("Standard mixed prefix KV payload is missing token slots.")
        if not isinstance(payload.token_slots, torch.Tensor):
            raise RuntimeError("Standard mixed prefix KV payload has no device slots.")
        slots = payload.token_slots.to(device=self.device, dtype=torch.int32).reshape(-1)
        count = int(slots.numel())
        ptr = self._num_free_slots
        self.free_slots_stack[ptr: ptr + count] = slots
        self._num_free_slots += count

    def allocate_prefix_kv_payload_device(self, payload: object) -> None:
        self.allocate_prefix_kv_payloads_device([payload])

    def allocate_prefix_kv_payloads_device(self, payloads: list[object]) -> None:
        normalized: list[StandardPrefixBlockPayload] = []
        counts: list[int] = []
        for payload in payloads:
            if not isinstance(payload, StandardPrefixBlockPayload):
                raise RuntimeError("Standard mixed prefix KV payload is missing token slots.")
            if isinstance(payload.token_slots, torch.Tensor):
                raise RuntimeError("Mixed prefix KV payload is already device-resident.")
            count = int(payload.block_end) - int(payload.block_start)
            if count <= 0:
                raise RuntimeError("Mixed prefix KV payload has an invalid token range.")
            normalized.append(payload)
            counts.append(count)
        if not normalized:
            return
        slots = self._take_prefix_device_slots(sum(counts))
        offset = 0
        for payload, count in zip(normalized, counts):
            payload.token_slots = slots[offset : offset + count]
            offset += count

    def free_prefix_kv_payload_device(self, payload: object) -> None:
        if not isinstance(payload, StandardPrefixBlockPayload):
            raise RuntimeError("Standard mixed prefix KV payload is missing token slots.")
        if not isinstance(payload.token_slots, torch.Tensor):
            raise RuntimeError("Mixed prefix KV payload has no device slots to demote.")
        self._return_prefix_device_slots(payload.token_slots)
        payload.token_slots = None

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

    def validate_prefix_cache_prune_target(
        self,
        token_ids: list[int],
        *,
        range_start: int | None = None,
        range_end: int | None = None,
        ranges: list[tuple[int, int]] | None = None,
        allow_recompress: bool = False,
    ) -> list[PrefixCacheBlock]:
        intervals = normalize_prefix_prune_ranges(
            token_count=len(token_ids), block_size=self.prefix_cache_block_size,
            range_start=range_start, range_end=range_end, ranges=ranges,
        )
        if allow_recompress:
            raise RuntimeError(
                "allow_recompress is not implemented because dropped KV cannot be "
                "rescored without rebuilding the original dense prefix."
            )
        controller = getattr(self, "prefix_offload_controller", None)
        if controller is not None:
            controller.synchronize_all()
        prefix_cache = self._require_prefix_cache()
        block_size = int(self.prefix_cache_block_size)
        range_end = intervals[-1][1]
        block_ids = prefix_cache.block_ids_for_tokens(
            [int(token_id) for token_id in token_ids[:range_end]],
            max_tokens=range_end,
        )
        expected_blocks = range_end // block_size
        if len(block_ids) != expected_blocks:
            raise RuntimeError(
                "prefix prune selector does not cover a complete block-aligned range."
            )
        hit_len, last_block_id, hit_blocks = prefix_cache.match_longest_block_ids(block_ids)
        if hit_len != range_end or hit_blocks != expected_blocks or last_block_id is None:
            raise RuntimeError(
                "prefix prune target is not fully present in the radix tree: "
                f"requested_end={range_end} matched_tokens={hit_len}."
            )
        chain = prefix_cache.get_chain(last_block_id, expected_blocks)
        affected = [
            block for left, right in intervals
            for block in chain[left // block_size : right // block_size]
        ]
        blocked = [
            block.stable_block_id.hex()[:16]
            for block in affected
            if int(block.ref_count) != 0 or block.residency.transfer is not None
        ]
        if blocked:
            raise RuntimeError(
                "prefix prune requires an idle, transfer-free subtree interval: "
                f"blocked_blocks={blocked}."
            )
        if any(self._standard_payload(block).retained_offsets is not None for block in affected):
            raise RuntimeError(
                "prefix prune target overlaps an already pruned block; "
                "repeated compression of the same interval is not implemented."
            )
        return affected

    @torch.no_grad()
    def prefix_cache_prune(
        self,
        token_ids: list[int],
        *,
        range_start: int | None = None,
        range_end: int | None = None,
        ranges: list[tuple[int, int]] | None = None,
        keep_indices: torch.Tensor,
        policy: str,
        prune_id: str,
        allow_recompress: bool = False,
    ) -> dict[str, object]:
        """Commit a shared mask indexed into the sorted, packed interval union."""
        intervals = normalize_prefix_prune_ranges(
            token_count=len(token_ids), block_size=self.prefix_cache_block_size,
            range_start=range_start, range_end=range_end, ranges=ranges,
        )
        affected = self.validate_prefix_cache_prune_target(
            token_ids, ranges=intervals, allow_recompress=allow_recompress,
        )
        prefix_cache = self._require_prefix_cache()
        block_size = int(self.prefix_cache_block_size)
        width = sum(right - left for left, right in intervals)
        # One host transfer for all interval selections, not one per block/range.
        selected = sorted({int(index) for index in keep_indices.detach().cpu().tolist()})
        if any(index < 0 or index >= width for index in selected):
            raise ValueError("prefix prune keep mask contains an out-of-range token index.")
        if len(selected) != int(keep_indices.numel()):
            raise ValueError("prefix prune keep mask contains duplicate token indices.")
        selected_set = set(selected)

        plans = []
        resident_slots = []
        keep_positions: list[int] = []
        drop_positions: list[int] = []
        slot_cursor = 0
        for relative_block_idx, block in enumerate(affected):
            payload = self._standard_payload(block)
            old_offsets = (
                tuple(range(block_size)) if payload.retained_offsets is None
                else tuple(int(offset) for offset in payload.retained_offsets)
            )
            block_base = relative_block_idx * block_size
            new_offsets = tuple(
                offset for offset in old_offsets if block_base + offset in selected_set
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
            plans.append((block, new_offsets))

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
        records = []
        block_cursor = 0
        created_at = time.time()
        for left, right in intervals:
            size = right - left
            retained = sum(
                len(offsets)
                for _, offsets in plans[block_cursor : block_cursor + size // block_size]
            )
            records.append((affected[block_cursor], PrefixPruneRecord(
                prune_id=prune_id, policy=policy, range_start=left, range_end=right,
                original_tokens=size, retained_tokens=retained, created_at=created_at,
            )))
            block_cursor += size // block_size

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
        for block, record in records:
            block.prune_record = record
        prefix_cache.mark_payload_compacted(affected)
        return {
            "prune_id": str(prune_id),
            "policy": str(policy),
            "range": [intervals[0][0], intervals[-1][1]],
            "ranges": [list(span) for span in intervals],
            "logical_tokens": width,
            "retained_tokens": len(selected),
            "freed_device_slots": len(drop_positions),
            "affected_blocks": len(affected),
            "quality_degraded": True,
        }

    def begin_prefix_prune_scoring(
        self,
        *,
        seq_id: int,
        candidate_start: int,
        query_start: int,
        query_end: int,
    ) -> None:
        if self._prefix_prune_scoring is not None:
            raise RuntimeError("another prefix-prune scoring forward is already active.")
        if not (0 <= candidate_start < query_start < query_end):
            raise ValueError(
                "invalid prefix-prune scoring ranges: "
                f"candidate_start={candidate_start} query=[{query_start}, {query_end})."
            )
        self._prefix_prune_scoring = {
            "seq_id": int(seq_id),
            "candidate_start": int(candidate_start),
            "query_start": int(query_start),
            "query_end": int(query_end),
            "score": None,
        }

    def begin_prefix_prune_scoring_batch(self, requests) -> None:
        if self._prefix_prune_scoring is not None:
            raise RuntimeError("another prefix-prune scoring forward is already active.")
        states = []
        for request in requests:
            self.begin_prefix_prune_scoring(**request)
            states.append(self._prefix_prune_scoring)
            self._prefix_prune_scoring = None
        self._prefix_prune_scoring = {"batch": states}

    def finish_prefix_prune_scoring_batch(self) -> list[torch.Tensor]:
        state = self._prefix_prune_scoring
        self._prefix_prune_scoring = None
        if state is None or "batch" not in state:
            raise RuntimeError("No prefix-prune scoring batch is active.")
        return [self._finish_prefix_prune_score(item) for item in state["batch"]]

    def abort_prefix_prune_scoring(self) -> None:
        self._prefix_prune_scoring = None

    def finish_prefix_prune_scoring(self) -> torch.Tensor:
        state = self._prefix_prune_scoring
        self._prefix_prune_scoring = None
        return self._finish_prefix_prune_score(state)

    @staticmethod
    def _finish_prefix_prune_score(state) -> torch.Tensor:
        if state is None or not isinstance(state.get("score"), torch.Tensor):
            raise RuntimeError("prefix-prune scoring forward produced no attention scores.")
        score = state["score"]
        positions = state.get("logical_positions")
        if positions is not None:
            logical_score = score.new_zeros(int(state["query_end"]))
            logical_score.index_copy_(0, positions, score)
            return logical_score
        return score

    def _prefix_prune_physical_score_window(self, state=None):
        if state is None:
            state = self._prefix_prune_scoring
        if "physical_window" in state:
            return state["physical_window"]
        row = self.seq_id_to_row[int(state["seq_id"])]
        end = int(self.row_seq_lens[row])
        query_len = int(state["query_end"]) - int(state["query_start"])
        start = end - query_len
        candidate = int(state["candidate_start"])
        if end != int(state["query_end"]):
            # Build this mapping once per maintenance forward, not per layer.
            positions = []
            for block in self.seq_id_to_prefix_blocks[int(state["seq_id"])]:
                payload = self._standard_payload(block)
                base = int(block.logical_block_idx) * self.prefix_cache_block_size
                offsets = payload.retained_offsets
                if offsets is None:
                    positions.extend(range(base, base + self.prefix_cache_block_size))
                else:
                    positions.extend(base + offset for offset in offsets)
            if len(positions) != start:
                raise RuntimeError("Prefix scoring logical/physical mapping length mismatch.")
            candidate = bisect_left(positions, candidate)
            positions.extend(range(int(state["query_start"]), int(state["query_end"])))
            state["logical_positions"] = torch.tensor(positions, dtype=torch.long, device=self.device)
        state["physical_window"] = (start, end, candidate)
        return state["physical_window"]

    def prefill_score_request(self, layer_idx, seqs):
        state = self._prefix_prune_scoring
        if state is None:
            return None
        if "batch" in state:
            states = state["batch"]
            if [s.seq_id for s in seqs] != [s["seq_id"] for s in states]:
                raise RuntimeError("Prefix scoring batch order differs from model inputs.")
            windows = [self._prefix_prune_physical_score_window(s) for s in states]
            return PrefillScoreRequest(
                tuple((start, end) for start, end, _ in windows), "probability",
                candidate_ranges=tuple((candidate, start) for start, _, candidate in windows),
            )
        start, end, candidate = self._prefix_prune_physical_score_window()
        return PrefillScoreRequest(((start, end),), "probability",
                                  candidate, end - start)

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
        del layer_idx, attention_lse
        state = self._prefix_prune_scoring
        if state is None:
            return None
        states = state.get("batch", [state])
        if int(chunk_lens.numel()) != len(states):
            raise RuntimeError("Prefix scoring metadata does not cover the maintenance batch.")
        if view.token_scores is None and not isinstance(view.payload, ExplicitKVPayload):
            raise TypeError("Prefix pruning requires explicit KV storage or fused token scores.")
        offset = 0
        for i, item in enumerate(states):
            start, end, candidate = self._prefix_prune_physical_score_window(item)
            length = end - start
            if view.token_scores is not None:
                if view.token_scores.shape[0] != len(states) or view.token_scores.shape[1] < end:
                    raise RuntimeError("Prefix-prune score shape does not match physical context.")
                score = view.token_scores[i, :end]
            else:
                # Explicit-KV scorer has scalar candidate bounds. Model projections
                # still run as one batch; score each row with its own normalizer.
                step_score = torch.zeros((1, end), dtype=torch.float32, device=q.device)
                starts = torch.tensor([start], dtype=torch.int32, device=q.device)
                prefill_score_fwd(
                    q[offset:offset + length], view.payload.k_cache, step_score,
                    view.meta.req_indices[i:i + 1], torch.zeros_like(starts),
                    view.meta.context_lens[i:i + 1], starts, length,
                    view.meta.active_slots, starts,
                    torch.tensor([end], dtype=torch.int32, device=q.device),
                    candidate_start=candidate, recent_keep_tokens=length,
                    score_mode="probability",
                )
                score = step_score[0]
            accumulated = item.get("score")
            if accumulated is None:
                item["score"] = score.clone()
            else:
                torch.maximum(accumulated, score, out=accumulated)
            offset += length
        if offset != q.shape[0]:
            raise RuntimeError("Prefix-prune query window length mismatch.")
        return None

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

    def _on_prefix_cache_reset(self) -> None:
        self._prefix_resident_slots_cache = None
        controller = getattr(self, "prefix_offload_controller", None)
        if controller is not None:
            controller.prefix_cache = self._require_prefix_cache()

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
    def _evict_prefix_cache_until_free(self, needed_slots: int) -> None:
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        needed_slots = int(needed_slots)
        if self._num_free_slots >= needed_slots:
            return
        if self._prefix_offload_enabled():
            controller = self.prefix_offload_controller
            assert controller is not None
            while self._num_free_slots < needed_slots:
                self._poll_prefix_offload()
                missing_slots = needed_slots - int(self._num_free_slots)
                with profiler.record("prefix_cache_device_demote"):
                    demoted = self.prefix_cache.demote_device_until_weight(
                        missing_slots,
                        self._block_resident_tokens_or_full,
                    )
                self._free_device_prefix_blocks(demoted)
                if self._num_free_slots >= needed_slots:
                    return
                if not controller.wait_oldest_d2h():
                    break
            return

        missing_slots = needed_slots - int(self._num_free_slots)
        with profiler.record("prefix_cache_evict"):
            evicted = self.prefix_cache.evict_until_weight(
                missing_slots,
                self._block_resident_tokens_or_full,
            )
        self._free_prefix_cache_blocks(evicted)

    def _evict_prefix_cache_for_insert(self, needed_blocks: int = 1) -> None:
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        if self._prefix_offload_enabled():
            max_blocks = self.prefix_cache.max_blocks
            if max_blocks is None:
                return
            over_capacity = len(self.prefix_cache) + int(needed_blocks) - int(max_blocks)
            if over_capacity <= 0:
                return
            controller = self.prefix_offload_controller
            assert controller is not None
            evicted: list[PrefixCacheBlock] = []
            while len(evicted) < over_capacity:
                self._poll_prefix_offload()
                remaining = over_capacity - len(evicted)
                with profiler.record("prefix_cache_host_evict"):
                    host_evicted = self.prefix_cache.evict_host_until_freeable(remaining)
                self._free_prefix_cache_blocks(host_evicted)
                evicted.extend(host_evicted)
                if len(evicted) >= over_capacity:
                    break

                remaining = over_capacity - len(evicted)
                with profiler.record("prefix_cache_device_demote"):
                    demoted = self.prefix_cache.demote_device_until_freeable(remaining)
                self._free_device_prefix_blocks(demoted)
                with profiler.record("prefix_cache_host_evict"):
                    newly_evicted = self.prefix_cache.evict_host_until_freeable(remaining)
                self._free_prefix_cache_blocks(newly_evicted)
                evicted.extend(newly_evicted)
                if len(evicted) >= over_capacity:
                    break

                # A successful wait retires one queued D2H operation.
                # Queue exhaustion bounds retries without scanning the radix.
                if not controller.wait_oldest_d2h():
                    break
            if len(evicted) != over_capacity:
                raise RuntimeError(
                    "Prefix cache logical capacity exceeded and not enough CPU-only leaves "
                    "are evictable: "
                    f"live_blocks={len(self.prefix_cache)} max_blocks={max_blocks} "
                    f"needed_blocks={needed_blocks} evicted_blocks={len(evicted)}."
                )
            return
        with profiler.record("prefix_cache_evict"):
            evicted = self.prefix_cache.ensure_insert_capacity(needed_blocks)
        self._free_prefix_cache_blocks(evicted)

    def _ensure_prefix_host_capacity(self, needed_blocks: int) -> None:
        controller = self.prefix_offload_controller
        if controller is None:
            raise RuntimeError("Prefix host capacity requested without an offload controller.")
        needed_blocks = int(needed_blocks)
        if controller.host_pool.free_blocks >= needed_blocks:
            return
        missing = needed_blocks - controller.host_pool.free_blocks
        with profiler.record("prefix_cache_host_evict"):
            evicted = self._require_prefix_cache().evict_host_until_freeable(missing)
        self._free_prefix_cache_blocks(evicted)
        if controller.host_pool.free_blocks < needed_blocks:
            raise RuntimeError(
                "Prefix host pool cannot preserve write-through residency: "
                f"need={needed_blocks} free={controller.host_pool.free_blocks} "
                f"evicted={len(evicted)} capacity={controller.host_pool.capacity_blocks}."
            )

    def _schedule_write_through_prefix_blocks(
        self,
        newly_unreferenced: list[PrefixCacheBlock] | None = None,
    ) -> None:
        if not self._prefix_offload_enabled():
            return
        if device_runtime.is_stream_capturing():
            raise RuntimeError("Prefix D2H scheduling is forbidden during graph capture.")
        self._poll_prefix_offload()
        prefix_cache = self._require_prefix_cache()
        pending = getattr(self, "_prefix_write_through_candidates", None)
        if pending is None:
            pending = {}
            self._prefix_write_through_candidates = pending
        selected = select_write_through_candidates(
            prefix_cache,
            pending,
            newly_unreferenced,
        )
        if not selected:
            return
        self._ensure_prefix_host_capacity(len(selected))
        controller = self.prefix_offload_controller
        assert controller is not None
        with profiler.record("prefix_cache_d2h_submit"):
            controller.submit_d2h(selected)
        for block in selected:
            pending.pop(block.stable_block_id, None)

    @cpu_timing.timed
    def on_forward_end(self, seqs: list[Sequence], is_prefill: bool):
        self._poll_prefix_offload()
        return super().on_forward_end(seqs, is_prefill)

    def _attach_prefix_cache_if_needed(self, seq: Sequence) -> None:
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        hit_len = int(getattr(seq, "prefix_cache_hit_len", 0) or 0)
        if hit_len <= 0:
            return
        if seq.seq_id in self.seq_id_to_prefix_blocks:
            return
        self._poll_prefix_offload()
        with profiler.record("prefix_cache_attach"):
            if seq.prefix_cache_hit_last_block_id is None:
                raise RuntimeError(f"seq_id={seq.seq_id} has prefix hit length but no last block id.")
            if hit_len % self.prefix_cache_block_size != 0:
                raise RuntimeError(
                    f"seq_id={seq.seq_id} prefix hit length is not block aligned: "
                    f"hit_len={hit_len} block_size={self.prefix_cache_block_size}."
                )
            chain = self.prefix_cache.get_chain(
                seq.prefix_cache_hit_last_block_id,
                int(seq.prefix_cache_hit_block_count),
            )
            if len(chain) * self.prefix_cache_block_size != hit_len:
                raise RuntimeError(
                    "Prefix cache chain length does not match scheduler metadata: "
                    f"seq_id={seq.seq_id} hit_len={hit_len} blocks={len(chain)} "
                    f"block_size={self.prefix_cache_block_size}."
                )
            cpu_only_blocks: list[PrefixCacheBlock] = []
            existing_h2d_operations: dict[int, PrefixH2DOperation] = {}
            resident_size = 0
            promotion_slot_count = 0
            saw_cpu_only = False
            for block in chain:
                payload = block.payload
                if not isinstance(payload, StandardPrefixBlockPayload):
                    raise RuntimeError(
                        f"Invalid Standard prefix cache payload for seq_id={seq.seq_id}: "
                        f"logical_block_idx={block.logical_block_idx}."
                    )
                residency = block.residency
                residency.validate()
                expected_slots = payload.resident_tokens(self.prefix_cache_block_size)
                resident_size += expected_slots
                if not residency.device_present:
                    saw_cpu_only = True
                    if not self._prefix_offload_enabled() or not residency.host_present:
                        raise RuntimeError(
                            "Prefix lookup returned a non-device block that cannot be promoted: "
                            f"seq_id={seq.seq_id} block={block.stable_block_id.hex()[:16]}."
                        )
                    if residency.transfer is not None:
                        raise RuntimeError(
                            "CPU-only prefix block has an unexpected in-flight transfer: "
                            f"seq_id={seq.seq_id} transfer={residency.transfer.value}."
                        )
                    if payload.host_block_index is None:
                        raise RuntimeError(
                            "CPU-only prefix block is missing its host allocation: "
                            f"seq_id={seq.seq_id} block={block.stable_block_id.hex()[:16]}."
                        )
                    cpu_only_blocks.append(block)
                    promotion_slot_count += expected_slots
                    continue
                if saw_cpu_only:
                    raise RuntimeError(
                        "Prefix device residency is not root-contiguous: "
                        f"seq_id={seq.seq_id} block={block.stable_block_id.hex()[:16]}."
                    )
                if (
                    not isinstance(payload.token_slots, torch.Tensor)
                    or int(payload.token_slots.numel()) != expected_slots
                ):
                    raise RuntimeError(
                        f"Invalid Standard prefix cache block slots for seq_id={seq.seq_id}: "
                        f"logical_block_idx={block.logical_block_idx}."
                    )
                if residency.transfer == PrefixTransferKind.H2D:
                    controller = self.prefix_offload_controller
                    assert controller is not None
                    operation = controller.h2d_operation_for_block(block)
                    if operation is None:
                        raise RuntimeError(
                            "Prefix block is promoting without a tracked H2D operation: "
                            f"block={block.stable_block_id.hex()[:16]}."
                        )
                    existing_h2d_operations[id(operation)] = operation

            existing_row_idx = self.seq_id_to_row.get(seq.seq_id)
            if existing_row_idx is not None and int(self.row_seq_lens[existing_row_idx]) != 0:
                raise RuntimeError(
                    f"Cannot attach prefix cache to non-empty row: seq_id={seq.seq_id} "
                    f"row_idx={existing_row_idx} "
                    f"row_len={int(self.row_seq_lens[existing_row_idx])}."
                )
            if existing_row_idx is None and not self.free_rows:
                raise RuntimeError("No free rows in cache manager buffer!")

            if resident_size > self.buffer_req_to_token_slots.shape[1]:
                raise ValueError("Prefix attachment exceeds the physical row capacity.")
            old_logical_len = (
                0 if existing_row_idx is None else int(self.row_logical_lens[existing_row_idx])
            )
            acquired = 0
            old_ranges = self.seq_id_to_cached_ranges.get(seq.seq_id)
            allocated_promotion_slots: torch.Tensor | None = None
            submitted_operation: PrefixH2DOperation | None = None
            promotion_committed = False
            try:
                for block in chain:
                    self.prefix_cache.acquire_block_ref(block)
                    acquired += 1
                if cpu_only_blocks:
                    if not self._prefix_offload_enabled():
                        raise RuntimeError(
                            "CPU prefix hit requires prefix cache offload to be enabled."
                        )
                    if device_runtime.is_stream_capturing():
                        raise RuntimeError("Prefix H2D promotion is forbidden during graph capture.")
                    allocated_promotion_slots = self._take_prefix_device_slots(promotion_slot_count)
                    offset = 0
                    for block in cpu_only_blocks:
                        payload = self._standard_payload(block)
                        end = offset + payload.resident_tokens(self.prefix_cache_block_size)
                        payload.token_slots = allocated_promotion_slots[offset:end]
                        offset = end
                    controller = self.prefix_offload_controller
                    assert controller is not None
                    with profiler.record("prefix_cache_h2d_submit"):
                        submitted_operation = controller.submit_h2d(cpu_only_blocks)
                        promotion_committed = True

                row_idx = self._get_free_row(seq.seq_id)
                if submitted_operation is not None:
                    existing_h2d_operations[id(submitted_operation)] = submitted_operation
                self._prefix_offload_step_h2d_operations.update(existing_h2d_operations)

                resident_slots = []
                for block in chain:
                    payload = self._standard_payload(block)
                    if not isinstance(payload.token_slots, torch.Tensor):
                        raise RuntimeError("Prefix promotion completed without device slots.")
                    if payload.token_slots.numel():
                        resident_slots.append(payload.token_slots)
                if resident_slots:
                    # One packed row copy, as before; no per-block copy kernels.
                    self.buffer_req_to_token_slots[row_idx, :resident_size] = torch.cat(resident_slots)
                self.row_seq_lens[row_idx] = resident_size
                self.row_logical_lens[row_idx] = hit_len
                self.seq_id_to_cached_ranges[seq.seq_id] = (
                    [(0, resident_size)] if resident_size else []
                )
                self.seq_id_to_prefix_blocks[seq.seq_id] = chain
                self.prefix_cache.touch_chain(chain)
            except BaseException:
                # These are aliases, not private allocations: never free them
                # via free_seq before the ownership markers have been committed.
                row_idx = self.seq_id_to_row.get(seq.seq_id)
                if row_idx is not None:
                    self.buffer_req_to_token_slots[row_idx, :] = 0
                    self.row_seq_lens[row_idx] = 0
                    self.row_logical_lens[row_idx] = old_logical_len
                    if existing_row_idx is None:
                        self.seq_id_to_row.pop(seq.seq_id, None)
                        self.free_rows.appendleft(row_idx)
                self.seq_id_to_prefix_blocks.pop(seq.seq_id, None)
                if old_ranges is None:
                    self.seq_id_to_cached_ranges.pop(seq.seq_id, None)
                else:
                    self.seq_id_to_cached_ranges[seq.seq_id] = old_ranges
                for block in chain[:acquired]:
                    self.prefix_cache.release_block_ref(block)
                if allocated_promotion_slots is not None and not promotion_committed:
                    # submit_h2d owns fencing/rollback if submission itself fails.
                    self._return_prefix_device_slots(allocated_promotion_slots)
                    for block in cpu_only_blocks:
                        self._standard_payload(block).token_slots = None
                # A submitted promotion remains index-owned and tracked, even
                # without this request. Do not synchronize or recycle its slots.
                raise

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
        if np.any(cur_lens >= self.buffer_req_to_token_slots.shape[1]):
            raise ValueError("Decode allocation exceeds the cache row capacity.")
        select_indices = self._take_device_slots(batch_size)
        try:
            self._commit_decode_rows(pending_rows)
        except BaseException:
            self._return_prefix_device_slots(select_indices)
            raise

        rows_gpu = self._static_rows_gpu[:batch_size]
        cols_gpu = self._static_cols_gpu[:batch_size]
        rows_gpu.copy_(torch.as_tensor(row_indices, dtype=torch.long), non_blocking=True)
        cols_gpu.copy_(torch.as_tensor(cur_lens, dtype=torch.long), non_blocking=True)
        self.buffer_req_to_token_slots[rows_gpu, cols_gpu] = select_indices
        self.row_seq_lens[row_indices] += 1
        self.row_logical_lens[row_indices] += 1

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
            debug_slots = os.getenv("SPARSEENGINE_DEBUG_SLOTS", "0") == "1"
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

    def _prepare_prefill(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_prefill"):
            self._poll_prefix_offload()
            self._prefix_offload_step_h2d_operations = {}
            for seq in seqs:
                self._attach_prefix_cache_if_needed(seq)

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
                positions_np[token_offset: token_offset + chunk_size] = np.arange(start_idx, end_idx)
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
