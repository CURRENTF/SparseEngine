from __future__ import annotations

from dataclasses import dataclass, field
from weakref import WeakKeyDictionary

import torch

from sparseengine.config import Config
from sparseengine.engine.prefix_cache import (
    PrefixCacheBlock,
    PrefixBlockPayload,
    PrefixTransferKind,
    RadixPrefixIndex,
    build_prefix_cache_fingerprint,
    select_write_through_candidates,
    usable_prefix_cache_tokens,
)
from sparseengine.engine.mixed_prefix_offload import (
    MixedQuestPrefixOffloadController,
    MixedPrefixOffloadController,
    PinnedMixedPrefixPool,
    PinnedMixedQuestPrefixPool,
)
from sparseengine.engine.recurrent_state_manager import RecurrentPrefixPayload
from sparseengine.platforms import device_runtime
from sparseengine.engine.sequence import Sequence
from sparseengine.engine.cache_manager.prefix_cache_mixin import (
    PrefixHitCapacityCacheEntry,
    PrefixLookupCache,
    lookup_prefix_cache_hit,
    prefix_hit_capacity_counts,
)
from sparseengine.utils.profiler import profiler


@dataclass
class MixedPrefixBlockPayload:
    kv_payload: object
    recurrent_payload: object
    token_count: int
    accounting_bytes: int
    recurrent_bytes: int
    host_block_index: int | None = None


@dataclass
class _PendingMixedPrefixBlock:
    stable_block_id: bytes
    parent_block_id: bytes | None
    logical_block_idx: int
    token_ids: list[int]
    payload: MixedPrefixBlockPayload


@dataclass
class _MixedPrefixRuntimeState:
    parent_block_id: bytes | None
    next_logical_block_idx: int
    pending_tokens: list[int] = field(default_factory=list)


class PrefixCacheCoordinator:
    """Owns mixed KV+recurrent prefix radix state for mixed runtimes."""

    def __init__(self, config: Config, cache_manager, recurrent_state_manager):
        self.config = config
        self.cache_manager = cache_manager
        self.recurrent_state_manager = recurrent_state_manager
        self.enabled = bool(config.enable_prefix_caching)
        self.block_size = int(config.prefix_cache_block_size or 0)
        self.max_recurrent_bytes = int(config.prefix_recurrent_capacity_bytes)
        self.prefix_cache = None
        if self.enabled:
            self.prefix_cache = RadixPrefixIndex(
                block_size=self.block_size,
                fingerprint=build_prefix_cache_fingerprint(config, self.block_size),
                max_blocks=config.prefix_cache_max_blocks,
            )
        self.seq_id_to_prefix_blocks: dict[int, list[PrefixCacheBlock]] = {}
        self.seq_id_to_materialized_blocks: dict[int, list[PrefixCacheBlock]] = {}
        self.runtime_states: dict[int, _MixedPrefixRuntimeState] = {}
        self.pending_blocks: dict[int, list[_PendingMixedPrefixBlock]] = {}
        self.pending_duplicate_refs: dict[int, list[bytes]] = {}
        self.pending_block_ids: set[bytes] = set()
        self.pending_recurrent_bytes = 0
        self.prefix_lookup_cache = PrefixLookupCache()
        self.prefix_hit_capacity_cache: WeakKeyDictionary[
            Sequence, PrefixHitCapacityCacheEntry
        ] = WeakKeyDictionary()
        self.capacity_limited_seq_ids: set[int] = set()
        self.skipped_capacity_blocks = 0
        self.offload_controller: MixedPrefixOffloadController | None = None
        self._step_h2d_operations = []
        self._write_through_candidates: dict[bytes, PrefixCacheBlock] = {}
        if bool(getattr(config, "enable_prefix_cache_offload", False)):
            self._init_offload()

    def _init_offload(self) -> None:
        if self.prefix_cache is None:
            raise RuntimeError("Mixed prefix offload requires an enabled coordinator radix.")
        if int(getattr(self.config, "tensor_parallel_size", 1)) not in (1, 2):
            raise RuntimeError("Mixed prefix offload currently supports only TP=1 or TP=2.")
        is_quest = str(getattr(self.config, "sparse_method", "") or "") == "quest"
        if not device_runtime.supports_pin_memory() or not device_runtime.supports_streams(
            self.cache_manager.device
        ):
            raise RuntimeError("Mixed prefix offload requires CUDA streams and pinned host memory.")
        host_size_gb = getattr(self.config, "prefix_cache_host_size_gb", None)
        if host_size_gb is None:
            raise RuntimeError("Mixed prefix offload requires prefix_cache_host_size_gb.")
        state_spec = self.recurrent_state_manager.state_spec
        recurrent_layers = tuple(
            int(layer_idx)
            for layer_idx in self.config.runtime_layout.linear_attention_layer_indices
        )
        kv_items_per_block = self.block_size
        if is_quest:
            page_size = int(self.cache_manager.page_size)
            if self.block_size % page_size != 0:
                raise RuntimeError("Mixed QuEST prefix block must contain whole pages.")
            kv_items_per_block += self.block_size // page_size
        kv_bytes = int(
            kv_items_per_block
            * self.cache_manager.num_kv_layers
            * 2
            * self.cache_manager.num_kv_heads
            * self.cache_manager.head_dim
            * self.cache_manager.kv_cache.element_size()
        )
        recurrent_bytes = state_spec.bytes_for_layers(len(recurrent_layers))
        bytes_per_block = kv_bytes + recurrent_bytes
        host_capacity_blocks = int(float(host_size_gb) * (1024**3)) // bytes_per_block
        required_blocks = int(self.prefix_cache.max_blocks or 0)
        if required_blocks <= 0:
            raise RuntimeError("Mixed prefix offload requires a finite prefix_cache_max_blocks.")
        if host_capacity_blocks < required_blocks:
            raise RuntimeError(
                "Mixed prefix host tier is too small for write-through safety: "
                f"host_blocks={host_capacity_blocks} required_blocks={required_blocks} "
                f"bytes_per_block={bytes_per_block}."
            )
        pool_type = PinnedMixedQuestPrefixPool if is_quest else PinnedMixedPrefixPool
        pool_kwargs = {}
        if is_quest:
            pool_kwargs["page_size"] = int(self.cache_manager.page_size)
        host_pool = pool_type(
            capacity_blocks=host_capacity_blocks,
            num_layers=self.cache_manager.num_kv_layers,
            block_size=self.block_size,
            num_kv_heads=self.cache_manager.num_kv_heads,
            head_dim=self.cache_manager.head_dim,
            dtype=self.cache_manager.kv_cache.dtype,
            state_spec=state_spec,
            recurrent_layer_indices=recurrent_layers,
            **pool_kwargs,
        )
        controller_type = (
            MixedQuestPrefixOffloadController if is_quest else MixedPrefixOffloadController
        )
        controller_kwargs = {}
        if is_quest:
            controller_kwargs["device_metadata_cache"] = self.cache_manager.metadata_cache
        self.offload_controller = controller_type(
            prefix_cache=self.prefix_cache,
            kv_cache=self.cache_manager.kv_cache,
            host_pool=host_pool,
            block_size=self.block_size,
            device=self.cache_manager.device,
            kv_transformer_layer_indices=tuple(
                int(layer_idx) for layer_idx in self.config.runtime_layout.kv_idx_to_layer_idx
            ),
            **controller_kwargs,
        )

    def _offload_enabled(self) -> bool:
        return getattr(self, "offload_controller", None) is not None

    def _poll_offload(self) -> None:
        controller = getattr(self, "offload_controller", None)
        if controller is not None:
            controller.poll()

    def _require_prefix_cache(self) -> RadixPrefixIndex:
        if self.prefix_cache is None:
            raise RuntimeError("prefix cache is not enabled for this runtime.")
        return self.prefix_cache

    def inspect(self, token_ids: list[int], *, include_subtree: bool = False) -> dict[str, object]:
        self._poll_offload()
        return self._require_prefix_cache().inspect_prefix(
            [int(token_id) for token_id in token_ids],
            include_subtree=include_subtree,
        )

    def match(self, token_ids: list[int]) -> dict[str, object]:
        self._poll_offload()
        if self.prefix_cache is None:
            return {
                "supported": True,
                "enabled": False,
                "method": str(getattr(self.config, "sparse_method", "") or ""),
                "matched_tokens": 0,
                "matched_blocks": 0,
                "match_ratio": 0.0,
                "reason": "prefix cache is not enabled for this runtime.",
            }
        token_ids = [int(token_id) for token_id in token_ids]
        usable_tokens = usable_prefix_cache_tokens(len(token_ids), self.block_size)
        hit_len, hit_last_block_id, hit_blocks = self.prefix_cache.match_longest_prefix(
            token_ids,
            max_usable_tokens=usable_tokens,
        )
        return {
            "supported": True,
            "enabled": True,
            "method": str(getattr(self.config, "sparse_method", "") or ""),
            "block_size": int(self.block_size),
            "prompt_tokens": int(len(token_ids)),
            "usable_tokens": int(usable_tokens),
            "matched_tokens": int(hit_len),
            "matched_blocks": int(hit_blocks),
            "match_ratio": 0.0 if usable_tokens <= 0 else float(hit_len) / float(usable_tokens),
            "last_block_id": None if hit_last_block_id is None else hit_last_block_id.hex(),
            "live_blocks": int(len(self.prefix_cache)),
        }

    def delete_subtree(self, token_ids: list[int]) -> dict[str, object]:
        controller = getattr(self, "offload_controller", None)
        if controller is not None:
            controller.synchronize_all()
        normalized = [int(token_id) for token_id in token_ids]
        prefix_cache = self._require_prefix_cache()
        plan = prefix_cache.preview_delete_subtree(normalized)
        self.cache_manager.synchronize_prefix_cache_delete_plan(plan.to_dict())
        result = prefix_cache.safe_delete_subtree(normalized)
        self._free_blocks(result.deleted_blocks)
        return result.to_dict()

    def set_eviction_priority(self, token_ids: list[int], *, priority: int) -> dict[str, object]:
        return self._require_prefix_cache().set_subtree_eviction_priority(
            [int(token_id) for token_id in token_ids],
            int(priority),
        )

    def stats(self) -> dict[str, int]:
        self._poll_offload()
        if self.prefix_cache is None:
            return {}
        accounting_bytes = 0
        recurrent_bytes = 0
        for block in self.prefix_cache.blocks.values():
            payload = block.payload
            if isinstance(payload, MixedPrefixBlockPayload):
                accounting_bytes += int(payload.accounting_bytes)
                recurrent_bytes += int(payload.recurrent_bytes)
        stats = self.prefix_cache.stats()
        stats["mixed_prefix_cache_accounting_bytes"] = int(accounting_bytes)
        stats["mixed_prefix_cache_recurrent_bytes"] = int(recurrent_bytes)
        stats["mixed_prefix_cache_pending_recurrent_bytes"] = int(
            self.pending_recurrent_bytes
        )
        stats["mixed_prefix_cache_max_recurrent_bytes"] = int(self.max_recurrent_bytes)
        stats["mixed_prefix_cache_skipped_capacity_blocks"] = int(self.skipped_capacity_blocks)
        stats["mixed_prefix_cache_evictable_slots"] = int(self.evictable_slots())
        controller = getattr(self, "offload_controller", None)
        if controller is not None:
            stats.update(controller.stats())
        return stats

    def debug_state_summary(self) -> dict[str, object] | None:
        """Return the coordinator-owned radix state for synchronized TP tests."""
        self._poll_offload()
        if self.prefix_cache is None:
            return None
        blocks = []
        for block_id, block in sorted(self.prefix_cache.blocks.items()):
            blocks.append(
                {
                    "stable_block_id": block_id.hex(),
                    "parent_block_id": (
                        None
                        if block.parent_block_id is None
                        else block.parent_block_id.hex()
                    ),
                    "logical_block_idx": int(block.logical_block_idx),
                    "token_ids": [int(token_id) for token_id in block.token_ids],
                    "ref_count": int(block.ref_count),
                    "last_access": int(block.last_access),
                    "eviction_priority": int(block.eviction_priority),
                    "device_present": bool(block.residency.device_present),
                    "host_present": bool(block.residency.host_present),
                    "transfer": (
                        None
                        if block.residency.transfer is None
                        else block.residency.transfer.value
                    ),
                }
            )
        return {
            "fingerprint": self.prefix_cache.fingerprint.hex(),
            "stats": {
                str(key): int(value) for key, value in self.stats().items()
            },
            "blocks": blocks,
        }

    def evictable_slots(self) -> int:
        if self.prefix_cache is None:
            return 0
        freeable = (
            self.prefix_cache.device_freeable_blocks()
            if self._offload_enabled()
            else self.prefix_cache.freeable_blocks()
        )
        return int(freeable * self.block_size)

    def step_reclaimable_slots(self) -> int:
        if self.prefix_cache is None:
            return 0
        reclaimable = (
            self.prefix_cache.device_reclaimable_blocks()
            if self._offload_enabled()
            else self.prefix_cache.freeable_blocks()
        )
        return int(reclaimable * self.block_size)

    def admission_reclaimable_slots(self) -> int:
        if self.prefix_cache is None:
            return 0
        reclaimable = (
            self.prefix_cache.device_reclaimable_blocks()
            if self._offload_enabled()
            else self.prefix_cache.freeable_blocks()
        )
        return int(reclaimable * self.block_size)

    def prefix_hit_evictable_slots(self, seq: Sequence) -> int:
        if self.prefix_cache is None or int(getattr(seq, "prefix_cache_hit_len", 0) or 0) <= 0:
            return 0
        cache = getattr(self, "prefix_hit_capacity_cache", None)
        if cache is None:
            cache = WeakKeyDictionary()
            self.prefix_hit_capacity_cache = cache
        reclaimable_blocks, promotion_blocks = prefix_hit_capacity_counts(
            self.prefix_cache,
            cache,
            seq,
            offload_enabled=self._offload_enabled(),
        )
        return int((reclaimable_blocks + promotion_blocks) * self.block_size)

    def refresh_prefix_cache_hit(self, seq: Sequence) -> None:
        self._poll_offload()
        seq.clear_prefix_cache_hit()
        if self.prefix_cache is None:
            return
        if seq.num_prefilled_tokens != 0 or seq.num_completion_tokens != 0:
            return
        usable_tokens = usable_prefix_cache_tokens(seq.num_prompt_tokens, self.block_size)
        if usable_tokens <= 0:
            return
        with profiler.record("mixed_prefix_cache_lookup"):
            hit_len, last_block_id, hit_blocks = lookup_prefix_cache_hit(
                self.prefix_cache,
                self.prefix_lookup_cache,
                seq,
                usable_tokens,
            )
        if hit_len <= 0:
            return
        if last_block_id is None or hit_blocks <= 0:
            raise RuntimeError("Mixed prefix cache lookup returned an invalid hit.")
        if hit_len >= seq.num_prompt_tokens or hit_len % self.block_size != 0:
            raise RuntimeError(
                "Mixed prefix cache lookup returned an unusable hit length: "
                f"seq_id={seq.seq_id} hit_len={hit_len} prompt_len={seq.num_prompt_tokens} "
                f"block_size={self.block_size}."
            )
        seq.prefix_cache_enabled = True
        seq.prefix_cache_hit_len = int(hit_len)
        seq.prefix_cache_hit_block_count = int(hit_blocks)
        seq.prefix_cache_hit_last_block_id = last_block_id
        seq.prefix_cache_block_size = self.block_size
        seq.prefix_cache_method = str(self.config.sparse_method or "")

    def attach_prefix_cache_hits(self, seqs: list[Sequence]) -> None:
        if self.prefix_cache is None:
            return
        self._poll_offload()
        self._step_h2d_operations = []
        for seq in seqs:
            self._attach_seq(seq)

    def before_prefill_layer_attention(self, layer_idx: int) -> None:
        controller = getattr(self, "offload_controller", None)
        if controller is None or not self._step_h2d_operations:
            return
        if device_runtime.is_stream_capturing():
            raise RuntimeError("Mixed prefix KV H2D waits are forbidden during graph capture.")
        kv_layer_index = self.cache_manager.kv_layer_index(layer_idx)
        for operation in self._step_h2d_operations:
            controller.wait_for_layer(operation, kv_layer_index)

    def _attach_seq(self, seq: Sequence) -> None:
        hit_len = int(getattr(seq, "prefix_cache_hit_len", 0) or 0)
        if hit_len <= 0 or seq.seq_id in self.seq_id_to_prefix_blocks:
            return
        if seq.prefix_cache_hit_last_block_id is None:
            raise RuntimeError(f"seq_id={seq.seq_id} has mixed prefix hit length but no last block id.")
        if hit_len % self.block_size != 0:
            raise RuntimeError(
                f"seq_id={seq.seq_id} mixed prefix hit length is not block aligned: "
                f"hit_len={hit_len} block_size={self.block_size}."
            )
        chain = self._require_prefix_cache().get_chain(
            seq.prefix_cache_hit_last_block_id,
            int(seq.prefix_cache_hit_block_count),
        )
        if len(chain) * self.block_size != hit_len:
            raise RuntimeError(
                "Mixed prefix cache chain length does not match scheduler metadata: "
                f"seq_id={seq.seq_id} hit_len={hit_len} blocks={len(chain)} block_size={self.block_size}."
            )

        with profiler.record("mixed_prefix_cache_attach"):
            cpu_only_blocks: list[PrefixCacheBlock] = []
            existing_operations = []
            saw_cpu_only = False
            for block in chain:
                payload = block.payload
                if not isinstance(payload, MixedPrefixBlockPayload):
                    raise RuntimeError("Mixed prefix cache block has an invalid payload.")
                block.residency.validate()
                if not block.residency.device_present:
                    saw_cpu_only = True
                    if not self._offload_enabled() or not block.residency.host_present:
                        raise RuntimeError(
                            "Mixed prefix lookup returned a non-device block that cannot be promoted."
                        )
                    if block.residency.transfer is not None or payload.host_block_index is None:
                        raise RuntimeError("Host-only mixed prefix block has invalid transfer state.")
                    cpu_only_blocks.append(block)
                    continue
                if saw_cpu_only:
                    raise RuntimeError("Mixed prefix device residency is not root-contiguous.")
                if block.residency.transfer == PrefixTransferKind.H2D:
                    controller = self.offload_controller
                    assert controller is not None
                    operation = controller.h2d_operation_for_block(block)
                    if operation is None:
                        raise RuntimeError("Mixed prefix H2D block has no tracked operation.")
                    if all(operation is not current for current in existing_operations):
                        existing_operations.append(operation)

            row_preexisted = bool(self.cache_manager.validate_prefix_kv_attach(seq))
            for block in chain:
                self.prefix_cache.acquire_block_ref(block)
            submitted_operation = None
            attached_kv_payloads: list[object] = []
            try:
                if cpu_only_blocks:
                    if device_runtime.is_stream_capturing():
                        raise RuntimeError("Mixed prefix H2D is forbidden during graph capture.")
                    missing_slots = max(
                        0,
                        len(cpu_only_blocks) * self.block_size
                        - int(self.cache_manager.num_free_slots),
                    )
                    if missing_slots:
                        self.evict_for_slots(missing_slots)
                    kv_payloads = []
                    for block in cpu_only_blocks:
                        payload = block.payload
                        assert isinstance(payload, MixedPrefixBlockPayload)
                        kv_payloads.append(payload.kv_payload)
                    self.cache_manager.allocate_prefix_kv_payloads_device(kv_payloads)
                    controller = self.offload_controller
                    assert controller is not None
                    controller.allocate_device_recurrent(cpu_only_blocks)
                    submitted_operation = controller.submit_h2d(cpu_only_blocks)
                for block in chain:
                    payload = block.payload
                    assert isinstance(payload, MixedPrefixBlockPayload)
                    self.cache_manager.attach_prefix_kv_payload(seq, payload.kv_payload)
                    attached_kv_payloads.append(payload.kv_payload)
                last_payload = chain[-1].payload
                if not isinstance(last_payload, MixedPrefixBlockPayload):
                    raise RuntimeError("Mixed prefix cache block has an invalid recurrent payload.")
                last_operation = None
                controller = getattr(self, "offload_controller", None)
                if controller is not None:
                    last_operation = controller.h2d_operation_for_block(chain[-1])
                self.recurrent_state_manager.attach_prefix_recurrent_payload(
                    seq,
                    last_payload.recurrent_payload,
                    readiness_events=(
                        None
                        if last_operation is None
                        else last_operation.auxiliary_layer_events
                    ),
                )
            except BaseException as attach_error:
                rollback_error = None
                if attached_kv_payloads:
                    try:
                        self.cache_manager.rollback_prefix_kv_attach(
                            seq,
                            attached_kv_payloads,
                            row_preexisted=row_preexisted,
                        )
                    except BaseException as exc:
                        rollback_error = exc
                for block in chain:
                    self.prefix_cache.release_block_ref(block)
                if cpu_only_blocks and submitted_operation is None:
                    controller = self.offload_controller
                    assert controller is not None
                    for block in cpu_only_blocks:
                        payload = block.payload
                        assert isinstance(payload, MixedPrefixBlockPayload)
                        recurrent = payload.recurrent_payload
                        if isinstance(recurrent, RecurrentPrefixPayload) and recurrent.layer_states:
                            controller.free_device_recurrent(block)
                        kv_payload = payload.kv_payload
                        if isinstance(getattr(kv_payload, "token_slots", None), torch.Tensor):
                            self.cache_manager.free_prefix_kv_payload_device(kv_payload)
                if rollback_error is not None:
                    raise RuntimeError(
                        "Mixed prefix attach rollback failed after "
                        f"{type(attach_error).__name__}: {attach_error}"
                    ) from rollback_error
                raise

            operations = list(existing_operations)
            if submitted_operation is not None:
                operations.append(submitted_operation)
            for operation in operations:
                if all(operation is not current for current in self._step_h2d_operations):
                    self._step_h2d_operations.append(operation)
            self.seq_id_to_prefix_blocks[int(seq.seq_id)] = chain
            self.prefix_cache.touch_chain(chain)

    def record_step_tokens(self, seqs: list[Sequence], is_prefill: bool) -> None:
        if self.prefix_cache is None:
            return
        for seq in seqs:
            if int(seq.seq_id) in self.capacity_limited_seq_ids:
                continue
            if is_prefill:
                chunk_size = int(seq.current_chunk_size or 0)
                if chunk_size <= 0:
                    continue
                start = int(seq.num_prefilled_tokens)
                end = start + chunk_size
                boundary = start + (self.block_size - (start % self.block_size))
                if start % self.block_size == 0:
                    boundary = start + self.block_size
                if end > boundary:
                    raise RuntimeError(
                        "Mixed prefix prefill chunks must not cross recurrent snapshot boundaries: "
                        f"seq_id={seq.seq_id} start={start} end={end} block_size={self.block_size}. "
                        "Schedule the prefill suffix up to the next prefix block boundary first."
                    )
                token_ids = seq.token_ids
                if len(token_ids) > chunk_size:
                    token_ids = token_ids[start:end]
                self._record_tokens(seq, [int(token_id) for token_id in token_ids])
            else:
                self._record_tokens(seq, [seq.decode_input_token])

    def finish_step(self) -> None:
        self._step_h2d_operations.clear()

    def _record_tokens(self, seq: Sequence, token_ids: list[int]) -> None:
        if getattr(seq, "multimodal_digest", None) is not None:
            return
        if not token_ids:
            return
        state = self.runtime_states.get(int(seq.seq_id))
        if state is None:
            hit_blocks = int(getattr(seq, "prefix_cache_hit_block_count", 0) or 0)
            state = _MixedPrefixRuntimeState(
                parent_block_id=getattr(seq, "prefix_cache_hit_last_block_id", None),
                next_logical_block_idx=hit_blocks,
            )
            self.runtime_states[int(seq.seq_id)] = state

        pending = self.pending_blocks.setdefault(int(seq.seq_id), [])

        def add_block(block_tokens: list[int]) -> None:
            block_start = int(state.next_logical_block_idx) * self.block_size
            block_end = block_start + self.block_size
            stable_block_id = self.prefix_cache.stable_block_id(block_tokens, state.parent_block_id)
            existing = self.prefix_cache.get_block(stable_block_id)
            if existing is not None:
                self._hold_materialized_ref(seq, existing)
                state.parent_block_id = stable_block_id
                state.next_logical_block_idx += 1
                return
            if stable_block_id in self.pending_block_ids:
                duplicate_refs = self.pending_duplicate_refs.setdefault(
                    int(seq.seq_id),
                    [],
                )
                if stable_block_id not in duplicate_refs:
                    duplicate_refs.append(stable_block_id)
                state.parent_block_id = stable_block_id
                state.next_logical_block_idx += 1
                return

            recurrent_bytes = int(
                self.recurrent_state_manager.prefix_recurrent_snapshot_nbytes()
            )
            if not self._reserve_pending_block(stable_block_id, recurrent_bytes):
                self.capacity_limited_seq_ids.add(int(seq.seq_id))
                self.skipped_capacity_blocks += 1
                return
            recurrent_payload = None
            try:
                kv_payload = self.cache_manager.build_prefix_kv_payload(
                    seq,
                    block_start,
                    block_end,
                )
                recurrent_payload = self.recurrent_state_manager.build_prefix_recurrent_payload(
                    seq,
                    block_end,
                )
                actual_recurrent_bytes = int(
                    self.recurrent_state_manager.prefix_recurrent_payload_nbytes(
                        recurrent_payload
                    )
                )
                if actual_recurrent_bytes != recurrent_bytes:
                    raise RuntimeError(
                        "Mixed prefix recurrent snapshot bytes differ from the model declaration: "
                        f"declared={recurrent_bytes} actual={actual_recurrent_bytes}."
                    )
                accounting_bytes = int(
                    self.cache_manager.prefix_kv_payload_nbytes(kv_payload)
                ) + recurrent_bytes
                pending.append(
                    _PendingMixedPrefixBlock(
                        stable_block_id=stable_block_id,
                        parent_block_id=state.parent_block_id,
                        logical_block_idx=state.next_logical_block_idx,
                        token_ids=block_tokens,
                        payload=MixedPrefixBlockPayload(
                            kv_payload=kv_payload,
                            recurrent_payload=recurrent_payload,
                            token_count=self.block_size,
                            accounting_bytes=accounting_bytes,
                            recurrent_bytes=recurrent_bytes,
                        ),
                    )
                )
            except BaseException:
                if recurrent_payload is not None:
                    self.recurrent_state_manager.free_prefix_recurrent_payload(
                        recurrent_payload
                    )
                self._release_pending_reservation(stable_block_id, recurrent_bytes)
                raise
            state.parent_block_id = stable_block_id
            state.next_logical_block_idx += 1

        offset = 0
        if state.pending_tokens:
            need = self.block_size - len(state.pending_tokens)
            take = min(need, len(token_ids))
            state.pending_tokens.extend(token_ids[:take])
            offset = take
            if len(state.pending_tokens) == self.block_size:
                add_block(list(state.pending_tokens))
                state.pending_tokens = []
            else:
                return

        full_tokens = ((len(token_ids) - offset) // self.block_size) * self.block_size
        end_full = offset + full_tokens
        for block_start in range(offset, end_full, self.block_size):
            add_block(token_ids[block_start : block_start + self.block_size])
        state.pending_tokens = token_ids[end_full:] if end_full < len(token_ids) else []

    def commit_pending_blocks(self, seqs: list[Sequence]) -> None:
        if self.prefix_cache is None:
            return
        with profiler.record("mixed_prefix_cache_commit"):
            for seq in seqs:
                pending_blocks = self.pending_blocks.pop(int(seq.seq_id), [])
                materialized = self.seq_id_to_materialized_blocks.setdefault(int(seq.seq_id), [])
                for pending_idx, pending in enumerate(pending_blocks):
                    inserted = None
                    inserted_new = False
                    recurrent_released = False
                    try:
                        if self.prefix_cache.has_block(pending.stable_block_id):
                            raise RuntimeError(
                                "Mixed prefix block became duplicate after unique pending reservation."
                            )
                        block = PrefixCacheBlock(
                            stable_block_id=pending.stable_block_id,
                            parent_block_id=pending.parent_block_id,
                            block_size=self.block_size,
                            logical_block_idx=pending.logical_block_idx,
                            payload=pending.payload,
                            token_ids=tuple(pending.token_ids),
                            ref_count=1,
                        )
                        inserted = self.prefix_cache.insert_block(block)
                        if inserted is not block:
                            raise RuntimeError(
                                "Mixed prefix insertion returned an unexpected duplicate block."
                            )
                        inserted_new = True
                        materialized.append(inserted)
                        try:
                            self.cache_manager.mark_materialized_prefix_kv_payload(
                                seq,
                                pending.payload.kv_payload,
                            )
                        except BaseException:
                            self.cache_manager.rollback_materialized_prefix_kv_payload(
                                seq,
                                pending.payload.kv_payload,
                            )
                            materialized.remove(inserted)
                            self.prefix_cache.set_block_ref_count(inserted, 0)
                            self.prefix_cache.rollback_inserted_leaf(inserted)
                            self.recurrent_state_manager.free_prefix_recurrent_payload(
                                pending.payload.recurrent_payload
                            )
                            recurrent_released = True
                            inserted_new = False
                            raise
                    except BaseException:
                        if not inserted_new and not recurrent_released:
                            self.recurrent_state_manager.free_prefix_recurrent_payload(
                                pending.payload.recurrent_payload
                            )
                        for unprocessed in pending_blocks[pending_idx + 1 :]:
                            self.recurrent_state_manager.free_prefix_recurrent_payload(
                                unprocessed.payload.recurrent_payload
                            )
                            self._release_pending_reservation(
                                unprocessed.stable_block_id,
                                int(unprocessed.payload.recurrent_bytes),
                            )
                        self.pending_duplicate_refs.pop(int(seq.seq_id), None)
                        if not materialized:
                            self.seq_id_to_materialized_blocks.pop(
                                int(seq.seq_id),
                                None,
                            )
                        raise
                    finally:
                        self._release_pending_reservation(
                            pending.stable_block_id,
                            int(pending.payload.recurrent_bytes),
                        )
                for stable_block_id in self.pending_duplicate_refs.pop(
                    int(seq.seq_id),
                    [],
                ):
                    block = self.prefix_cache.get_block(stable_block_id)
                    if block is None:
                        raise RuntimeError(
                            "Mixed prefix duplicate reservation was not committed by its owner."
                        )
                    self._hold_materialized_ref(seq, block)

    def _hold_materialized_ref(
        self,
        seq: Sequence,
        block: PrefixCacheBlock,
    ) -> None:
        seq_id = int(seq.seq_id)
        held = [
            *self.seq_id_to_prefix_blocks.get(seq_id, []),
            *self.seq_id_to_materialized_blocks.get(seq_id, []),
        ]
        if any(existing.stable_block_id == block.stable_block_id for existing in held):
            return
        self._require_prefix_cache().acquire_block_ref(block)
        self.seq_id_to_materialized_blocks.setdefault(seq_id, []).append(block)

    def _live_recurrent_bytes(self) -> int:
        total = 0
        for block in self._require_prefix_cache().blocks.values():
            payload = block.payload
            if not isinstance(payload, MixedPrefixBlockPayload):
                raise RuntimeError("Mixed prefix cache block has an invalid payload.")
            total += int(payload.recurrent_bytes)
        return int(total)

    def _reserve_pending_block(
        self,
        stable_block_id: bytes,
        recurrent_bytes: int,
    ) -> bool:
        recurrent_bytes = int(recurrent_bytes)
        if stable_block_id in self.pending_block_ids:
            return False
        if not self._evict_for_insert(
            1,
            incoming_recurrent_bytes=recurrent_bytes,
        ):
            return False
        self.pending_block_ids.add(stable_block_id)
        self.pending_recurrent_bytes += recurrent_bytes
        return True

    def _release_pending_reservation(
        self,
        stable_block_id: bytes,
        recurrent_bytes: int,
    ) -> None:
        if stable_block_id not in self.pending_block_ids:
            raise RuntimeError("Mixed prefix pending reservation is missing during release.")
        self.pending_block_ids.remove(stable_block_id)
        self.pending_recurrent_bytes -= int(recurrent_bytes)
        if self.pending_recurrent_bytes < 0:
            raise RuntimeError("Mixed prefix pending recurrent byte count became negative.")

    def _evict_for_insert(self, needed_blocks: int, *, incoming_recurrent_bytes: int) -> bool:
        incoming_recurrent_bytes = int(incoming_recurrent_bytes)
        if incoming_recurrent_bytes < 0:
            raise ValueError(
                f"incoming_recurrent_bytes must be >= 0, got {incoming_recurrent_bytes}."
            )
        if incoming_recurrent_bytes > self.max_recurrent_bytes:
            raise RuntimeError(
                "Mixed prefix recurrent payload exceeds the configured byte budget: "
                f"payload_bytes={incoming_recurrent_bytes} "
                f"max_bytes={self.max_recurrent_bytes}."
            )
        prefix_cache = self._require_prefix_cache()
        needed_blocks = int(needed_blocks)
        if prefix_cache.max_blocks is not None:
            over_capacity = (
                len(prefix_cache.blocks)
                + len(getattr(self, "pending_block_ids", ()))
                + needed_blocks
                - int(prefix_cache.max_blocks)
            )
            if over_capacity > 0:
                if self._offload_enabled():
                    if not self._evict_host_blocks(over_capacity):
                        return False
                else:
                    evicted = prefix_cache.evict_until_freeable(over_capacity)
                    self._free_blocks(evicted)
                    if len(evicted) != over_capacity:
                        return False
        while (
            self._live_recurrent_bytes()
            + int(getattr(self, "pending_recurrent_bytes", 0))
            + incoming_recurrent_bytes
            > self.max_recurrent_bytes
        ):
            if self._offload_enabled():
                if not self._evict_host_blocks(1):
                    return False
            else:
                byte_evicted = prefix_cache.evict_until_freeable(1)
                if not byte_evicted:
                    return False
                self._free_blocks(byte_evicted)
        return True

    def _evict_host_blocks(self, needed_blocks: int) -> bool:
        controller = getattr(self, "offload_controller", None)
        if controller is None:
            raise RuntimeError("Mixed host eviction requested without an offload controller.")
        evicted: list[PrefixCacheBlock] = []
        while len(evicted) < int(needed_blocks):
            self._poll_offload()
            remaining = int(needed_blocks) - len(evicted)
            host_evicted = self._require_prefix_cache().evict_host_until_freeable(remaining)
            self._free_blocks(host_evicted)
            evicted.extend(host_evicted)
            if len(evicted) >= int(needed_blocks):
                break
            demoted = self._require_prefix_cache().demote_device_until_freeable(remaining)
            for block in demoted:
                self._free_device_block(block)
            newly_evicted = self._require_prefix_cache().evict_host_until_freeable(remaining)
            self._free_blocks(newly_evicted)
            evicted.extend(newly_evicted)
            if len(evicted) >= int(needed_blocks):
                break
            inflight_before = sum(
                1
                for block in self._require_prefix_cache().blocks.values()
                if block.residency.transfer == PrefixTransferKind.D2H
            )
            if inflight_before <= 0 or not controller.wait_oldest_d2h():
                break
            self._poll_offload()
            inflight_after = sum(
                1
                for block in self._require_prefix_cache().blocks.values()
                if block.residency.transfer == PrefixTransferKind.D2H
            )
            if inflight_after >= inflight_before:
                break
        return len(evicted) == int(needed_blocks)

    def evict_for_slots(self, needed_slots: int) -> None:
        if self.prefix_cache is None:
            return
        needed_slots = int(needed_slots)
        if needed_slots <= 0:
            return
        needed_blocks = (needed_slots + self.block_size - 1) // self.block_size
        if self._offload_enabled():
            controller = self.offload_controller
            assert controller is not None
            freed_blocks = 0
            while freed_blocks < needed_blocks:
                self._poll_offload()
                demoted = self.prefix_cache.demote_device_until_freeable(
                    needed_blocks - freed_blocks
                )
                for block in demoted:
                    self._free_device_block(block)
                freed_blocks += len(demoted)
                if freed_blocks >= needed_blocks:
                    break
                if not controller.wait_oldest_d2h():
                    break
            success = freed_blocks == needed_blocks
        else:
            evicted = self.prefix_cache.evict_until_freeable(needed_blocks)
            self._free_blocks(evicted)
            freed_blocks = len(evicted)
            success = freed_blocks == needed_blocks
        if not success:
            raise RuntimeError(
                "Mixed prefix cache could not evict enough blocks for KV allocation: "
                f"needed_slots={needed_slots} block_size={self.block_size} "
                f"needed_blocks={needed_blocks} freed_blocks={freed_blocks}."
            )

    def _free_device_block(self, block: PrefixCacheBlock) -> None:
        payload = block.payload
        if not isinstance(payload, MixedPrefixBlockPayload):
            raise RuntimeError("Mixed prefix cache block has an invalid payload.")
        controller = self.offload_controller
        if controller is None:
            raise RuntimeError("Mixed device demotion requires an offload controller.")
        self.cache_manager.free_prefix_kv_payload_device(payload.kv_payload)
        controller.free_device_recurrent(block)

    def _free_blocks(self, blocks: list[PrefixCacheBlock]) -> None:
        pending = getattr(self, "_write_through_candidates", None)
        if pending is not None:
            for block in blocks:
                pending.pop(block.stable_block_id, None)
        payloads = [block.payload for block in blocks]
        if any(
            not isinstance(payload, MixedPrefixBlockPayload)
            for payload in payloads
        ):
            raise RuntimeError("Mixed prefix cache block has an invalid payload.")
        if self._offload_enabled():
            host_blocks = []
            for block, payload in zip(blocks, payloads):
                if block.residency.device_present:
                    self._free_device_block(block)
                if block.residency.host_present:
                    host_blocks.append(block)
                recurrent = payload.recurrent_payload
                if isinstance(recurrent, RecurrentPrefixPayload):
                    recurrent.layer_states = {}
            if host_blocks:
                controller = self.offload_controller
                assert controller is not None
                controller.free_host_payloads(host_blocks)
            return
        for payload in payloads:
            self.cache_manager.free_prefix_kv_payload(payload.kv_payload)
            self.recurrent_state_manager.free_prefix_recurrent_payload(payload.recurrent_payload)

    def _ensure_host_capacity(self, needed_blocks: int) -> None:
        controller = self.offload_controller
        if controller is None:
            raise RuntimeError("Mixed host capacity requested without an offload controller.")
        if controller.host_pool.free_blocks >= int(needed_blocks):
            return
        missing = int(needed_blocks) - controller.host_pool.free_blocks
        if not self._evict_host_blocks(missing):
            raise RuntimeError(
                "Mixed prefix host pool cannot preserve write-through residency: "
                f"need={needed_blocks} free={controller.host_pool.free_blocks}."
            )

    def _schedule_write_through(
        self,
        newly_unreferenced: list[PrefixCacheBlock] | None = None,
    ) -> None:
        if not self._offload_enabled():
            return
        if device_runtime.is_stream_capturing():
            raise RuntimeError("Mixed prefix D2H scheduling is forbidden during graph capture.")
        self._poll_offload()
        prefix_cache = self._require_prefix_cache()
        pending = getattr(self, "_write_through_candidates", None)
        if pending is None:
            pending = {}
            self._write_through_candidates = pending
        selected = select_write_through_candidates(
            prefix_cache,
            pending,
            newly_unreferenced,
        )
        if not selected:
            return
        self._ensure_host_capacity(len(selected))
        controller = self.offload_controller
        assert controller is not None
        controller.submit_d2h(selected)
        for block in selected:
            pending.pop(block.stable_block_id, None)

    def release_seq(self, seq_id: int) -> None:
        seq_id = int(seq_id)
        self.prefix_lookup_cache.discard(seq_id)
        released_blocks = self.seq_id_to_prefix_blocks.pop(seq_id, [])
        released_blocks.extend(self.seq_id_to_materialized_blocks.pop(seq_id, []))
        prefix_cache = self._require_prefix_cache()
        for block in released_blocks:
            prefix_cache.release_block_ref(
                block,
                negative_error="Mixed prefix cache block ref_count became negative.",
            )
        self.runtime_states.pop(seq_id, None)
        for pending in self.pending_blocks.pop(seq_id, []):
            self.recurrent_state_manager.free_prefix_recurrent_payload(
                pending.payload.recurrent_payload
            )
            self._release_pending_reservation(
                pending.stable_block_id,
                int(pending.payload.recurrent_bytes),
            )
        self.pending_duplicate_refs.pop(seq_id, None)
        self.capacity_limited_seq_ids.discard(seq_id)
        self._schedule_write_through(released_blocks)

    def reset_after_warmup(self) -> None:
        if self.prefix_cache is None:
            return
        if self.seq_id_to_prefix_blocks or self.seq_id_to_materialized_blocks:
            raise RuntimeError("Cannot reset mixed prefix cache while sequences still reference blocks.")
        for pending_blocks in self.pending_blocks.values():
            for pending in pending_blocks:
                self.recurrent_state_manager.free_prefix_recurrent_payload(
                    pending.payload.recurrent_payload
                )
                self._release_pending_reservation(
                    pending.stable_block_id,
                    int(pending.payload.recurrent_bytes),
                )
        blocks = list(self.prefix_cache.blocks.values())
        referenced = [block for block in blocks if int(block.ref_count) != 0]
        if referenced:
            raise RuntimeError(
                "Cannot reset mixed prefix cache while blocks are referenced: "
                f"referenced_blocks={len(referenced)}."
            )
        controller = getattr(self, "offload_controller", None)
        if controller is not None:
            controller.synchronize_all()
        self._free_blocks(blocks)
        if controller is not None:
            controller.reset()
        self.prefix_cache = RadixPrefixIndex(
            block_size=self.block_size,
            fingerprint=build_prefix_cache_fingerprint(self.config, self.block_size),
            max_blocks=self.config.prefix_cache_max_blocks,
        )
        self.seq_id_to_prefix_blocks.clear()
        self.seq_id_to_materialized_blocks.clear()
        self.runtime_states.clear()
        self.pending_blocks.clear()
        self.pending_duplicate_refs.clear()
        self.pending_block_ids.clear()
        self.pending_recurrent_bytes = 0
        self.prefix_lookup_cache = PrefixLookupCache()
        self.prefix_hit_capacity_cache = WeakKeyDictionary()
        self.capacity_limited_seq_ids.clear()
        self.skipped_capacity_blocks = 0
        getattr(self, "_step_h2d_operations", []).clear()
        getattr(self, "_write_through_candidates", {}).clear()
        if controller is not None:
            controller.prefix_cache = self.prefix_cache
