from __future__ import annotations

from dataclasses import dataclass

import torch

from sparseengine.config import Config
from sparseengine.engine.prefix_cache import PrefixCacheBlock
from sparseengine.engine.mixed_prefix_offload import (
    MixedPrefixOffloadController,
    PinnedMixedPrefixPool,
)
from sparseengine.engine.recurrent_state_manager import RecurrentPrefixPayload
from sparseengine.platforms import device_runtime
from sparseengine.engine.sequence import Sequence
from sparseengine.engine.cache_manager.prefix_cache_mixin import PendingPrefixBlock, RadixPrefixRuntime
from sparseengine.utils.profiler import profiler


@dataclass
class MixedPrefixBlockPayload:
    kv_payload: object
    recurrent_payload: object
    token_count: int
    accounting_bytes: int
    recurrent_bytes: int
    host_block_index: int | None = None

    def resident_tokens(self, block_size: int) -> int:
        return block_size


class PrefixCacheCoordinator(RadixPrefixRuntime):
    """Owns mixed KV+recurrent prefix radix state for mixed runtimes."""

    _prefix_label = "Mixed prefix"
    _prefix_profile = "mixed_prefix_cache"
    _prefix_wait_label = "Mixed prefix KV"
    _prefix_disabled_reason = "prefix cache is not enabled for this runtime."
    _prefix_payload_type = MixedPrefixBlockPayload

    def _prefix_cache_negative_refcount_message(self) -> str:
        return "Mixed prefix cache block ref_count became negative."

    def clear_prefix_cache_hit(self, seq: Sequence) -> None:
        seq.clear_prefix_cache_hit()

    def synchronize_prefix_cache_delete_plan(self, plan) -> None:
        self.cache_manager.synchronize_prefix_cache_delete_plan(plan)

    def __init__(self, config: Config, cache_manager, recurrent_state_manager):
        self.config = config
        self.cache_manager = cache_manager
        self.recurrent_state_manager = recurrent_state_manager
        self.enable_prefix_caching = bool(config.enable_prefix_caching)
        self.prefix_cache_block_size = int(config.prefix_cache_block_size or 0)
        self.max_recurrent_bytes = int(config.prefix_recurrent_capacity_bytes)
        self._init_prefix_index()
        self._init_prefix_cache_runtime()
        self.pending_duplicate_refs: dict[int, list[bytes]] = {}
        self.pending_block_ids: set[bytes] = set()
        self.pending_recurrent_bytes = 0
        self._recurrent_bytes_cache = None
        self.capacity_limited_seq_ids: set[int] = set()
        self.skipped_capacity_blocks = 0
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
        kv_items_per_block = self.prefix_cache_block_size
        if is_quest:
            page_size = int(self.cache_manager.page_size)
            if self.prefix_cache_block_size % page_size != 0:
                raise RuntimeError("Mixed QuEST prefix block must contain whole pages.")
            kv_items_per_block += self.prefix_cache_block_size // page_size
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
        host_pool = PinnedMixedPrefixPool(
            capacity_blocks=host_capacity_blocks,
            num_layers=self.cache_manager.num_kv_layers,
            block_size=self.prefix_cache_block_size,
            num_kv_heads=self.cache_manager.num_kv_heads,
            head_dim=self.cache_manager.head_dim,
            dtype=self.cache_manager.kv_cache.dtype,
            state_spec=state_spec,
            recurrent_layer_indices=recurrent_layers,
            page_size=page_size if is_quest else None,
        )
        self.prefix_offload_controller = MixedPrefixOffloadController(
            prefix_cache=self.prefix_cache,
            kv_cache=self.cache_manager.kv_cache,
            host_pool=host_pool,
            block_size=self.prefix_cache_block_size,
            device=self.cache_manager.device,
            kv_transformer_layer_indices=tuple(
                int(layer_idx) for layer_idx in self.config.runtime_layout.kv_idx_to_layer_idx
            ),
            device_metadata_cache=self.cache_manager.metadata_cache if is_quest else None,
        )

    def stats(self) -> dict[str, int]:
        self._poll_prefix_offload()
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
        controller = getattr(self, "prefix_offload_controller", None)
        if controller is not None:
            stats.update(controller.stats())
        return stats

    def debug_state_summary(self) -> dict[str, object] | None:
        """Return the coordinator-owned radix state for synchronized TP tests."""
        self._poll_prefix_offload()
        if self.prefix_cache is None:
            return None
        return self.prefix_cache.debug_state_summary(stats=self.stats)

    def evictable_slots(self) -> int:
        if self.prefix_cache is None:
            return 0
        freeable = (
            self.prefix_cache.device_freeable_blocks()
            if self._prefix_offload_enabled()
            else self.prefix_cache.freeable_blocks()
        )
        return int(freeable * self.prefix_cache_block_size)

    def step_reclaimable_slots(self) -> int:
        if self.prefix_cache is None:
            return 0
        reclaimable = (
            self.prefix_cache.device_reclaimable_blocks()
            if self._prefix_offload_enabled()
            else self.prefix_cache.freeable_blocks()
        )
        return int(reclaimable * self.prefix_cache_block_size)

    def prefix_hit_evictable_slots(self, seq: Sequence) -> int:
        if int(getattr(seq, "prefix_cache_hit_len", 0) or 0) <= 0:
            return 0
        reclaimable, promotion = self._prefix_hit_capacity_counts(seq)
        return (reclaimable + promotion) * self.prefix_cache_block_size

    def prefix_hit_shared_costs(self, seq: Sequence) -> dict[object, int]:
        if self.prefix_cache is None or int(getattr(seq, "prefix_cache_hit_len", 0) or 0) <= 0:
            return {}
        hit_blocks = int(getattr(seq, "prefix_cache_hit_block_count", 0) or 0)
        last_block_id = getattr(seq, "prefix_cache_hit_last_block_id", None)
        if hit_blocks <= 0:
            return {}
        if last_block_id is None:
            raise RuntimeError(
                f"seq_id={seq.seq_id} has mixed prefix hit blocks but no last block id."
            )
        reclaimable_ids = (
            self.prefix_cache.device_reclaimable_block_ids()
            if self._prefix_offload_enabled()
            else self.prefix_cache.freeable_block_ids()
        )
        costs: dict[object, int] = {}
        for block in self.prefix_cache.get_chain(last_block_id, hit_blocks):
            if block.stable_block_id in reclaimable_ids or (
                self._prefix_offload_enabled() and not block.residency.device_present
            ):
                costs[block.stable_block_id] = int(self.prefix_cache_block_size)
        return costs

    def attach_prefix_cache_hits(self, seqs: list[Sequence]) -> None:
        if self.prefix_cache is None:
            return
        self._poll_prefix_offload()
        self._prefix_offload_step_h2d_operations = {}
        for seq in seqs:
            self._attach_seq(seq)

    def before_prefill_layer_attention(self, layer_idx: int) -> None:
        self._wait_for_prefix_layer(layer_idx, self.cache_manager)

    def _attach_seq(self, seq: Sequence) -> None:
        hit_len = int(getattr(seq, "prefix_cache_hit_len", 0) or 0)
        if hit_len <= 0 or seq.seq_id in self.seq_id_to_prefix_blocks:
            return
        with profiler.record("mixed_prefix_cache_attach"):
            chain, _, cpu_only_blocks, existing_operations, _, promotion_tokens = (
                self._prefix_chain_for_attachment(seq, hit_len)
            )
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
                        promotion_tokens
                        - int(self.cache_manager.num_free_slots),
                    )
                    if missing_slots:
                        self.evict_for_slots(missing_slots)
                    kv_payloads = [block.payload.kv_payload for block in cpu_only_blocks]
                    self.cache_manager.allocate_prefix_kv_payloads_device(kv_payloads)
                    controller = self.prefix_offload_controller
                    assert controller is not None
                    controller.allocate_device_recurrent(cpu_only_blocks)
                    submitted_operation = controller.submit_h2d(cpu_only_blocks)
                kv_payloads = [block.payload.kv_payload for block in chain]
                self.cache_manager.attach_prefix_kv_payloads(seq, kv_payloads)
                attached_kv_payloads = kv_payloads
                controller = self.prefix_offload_controller
                last_operation = (
                    controller.h2d_operation_for_block(chain[-1]) if controller is not None else None
                )
                self.recurrent_state_manager.attach_prefix_recurrent_payload(
                    seq, chain[-1].payload.recurrent_payload,
                    readiness_events=None if last_operation is None else last_operation.auxiliary_layer_events,
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
                    controller = self.prefix_offload_controller
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

            if submitted_operation is not None:
                existing_operations[id(submitted_operation)] = submitted_operation
            self._prefix_offload_step_h2d_operations.update(existing_operations)
            self._hold_prefix_chain(int(seq.seq_id), chain)

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
                boundary = start + (self.prefix_cache_block_size - (start % self.prefix_cache_block_size))
                if end > boundary:
                    raise RuntimeError(
                        "Mixed prefix prefill chunks must not cross recurrent snapshot boundaries: "
                        f"seq_id={seq.seq_id} start={start} end={end} block_size={self.prefix_cache_block_size}. "
                        "Schedule the prefill suffix up to the next prefix block boundary first."
                    )
                token_ids = seq.token_ids
                if len(token_ids) > chunk_size:
                    token_ids = token_ids[start:end]
                self._record_tokens(seq, [int(token_id) for token_id in token_ids])
            else:
                self._record_tokens(seq, [seq.decode_input_token])

    def finish_step(self) -> None:
        self._prefix_offload_step_h2d_operations.clear()

    def _record_tokens(self, seq: Sequence, token_ids: list[int]) -> None:
        if getattr(seq, "multimodal_digest", None) is not None:
            return
        if not token_ids:
            return
        state, pending = self._prefix_record_state(seq)
        for block_tokens, _ in self._iter_completed_prefix_blocks(state, token_ids):
            block_start = int(state.next_logical_block_idx) * self.prefix_cache_block_size
            block_end = block_start + self.prefix_cache_block_size
            stable_block_id = self.prefix_cache.stable_block_id(block_tokens, state.parent_block_id)
            existing = self.prefix_cache.get_block(stable_block_id)
            if existing is not None:
                self._hold_materialized_prefix_block_ref(seq, existing)
                state.parent_block_id = stable_block_id
                state.next_logical_block_idx += 1
                continue
            if stable_block_id in self.pending_block_ids:
                duplicate_refs = self.pending_duplicate_refs.setdefault(
                    int(seq.seq_id),
                    [],
                )
                if stable_block_id not in duplicate_refs:
                    duplicate_refs.append(stable_block_id)
                state.parent_block_id = stable_block_id
                state.next_logical_block_idx += 1
                continue

            recurrent_bytes = int(
                self.recurrent_state_manager.prefix_recurrent_snapshot_nbytes()
            )
            if not self._reserve_pending_block(stable_block_id, recurrent_bytes):
                self.capacity_limited_seq_ids.add(int(seq.seq_id))
                self.skipped_capacity_blocks += 1
                continue
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
                    PendingPrefixBlock(
                        stable_block_id=stable_block_id,
                        parent_block_id=state.parent_block_id,
                        logical_block_idx=state.next_logical_block_idx,
                        token_ids=block_tokens,
                        payload=MixedPrefixBlockPayload(
                            kv_payload=kv_payload,
                            recurrent_payload=recurrent_payload,
                            token_count=self.prefix_cache_block_size,
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


    def commit_pending_blocks(self, seqs: list[Sequence]) -> None:
        if self.prefix_cache is None:
            return
        with profiler.record("mixed_prefix_cache_commit"):
            for seq in seqs:
                pending_prefix_blocks = self.pending_prefix_blocks.pop(int(seq.seq_id), [])
                materialized = self.seq_id_to_materialized_blocks.setdefault(int(seq.seq_id), {})
                for pending_idx, pending in enumerate(pending_prefix_blocks):
                    inserted_new = False
                    recurrent_released = False
                    try:
                        if self.prefix_cache.has_block(pending.stable_block_id):
                            raise RuntimeError(
                                "Mixed prefix block became duplicate after unique pending reservation."
                            )
                        inserted, inserted_new = self._insert_pending_prefix_block(pending)
                        if not inserted_new:
                            raise RuntimeError(
                                "Mixed prefix insertion returned an unexpected duplicate block."
                            )
                        materialized[inserted.stable_block_id] = inserted
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
                            materialized.pop(inserted.stable_block_id)
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
                        self._release_pending_blocks(pending_prefix_blocks[pending_idx + 1 :])
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
                    self._hold_materialized_prefix_block_ref(seq, block)

    def _live_recurrent_bytes(self) -> int:
        prefix_cache = self._require_prefix_cache()
        cached = getattr(self, "_recurrent_bytes_cache", None)
        # recurrent_bytes is fixed when a block is built, including across
        # offload. Only insertion/removal changes this logical byte total.
        if (cached is not None and cached[0] is prefix_cache
                and cached[1] == prefix_cache.insert_epoch
                and cached[2] == prefix_cache.remove_epoch):
            return cached[3]
        total = 0
        for block in prefix_cache.blocks.values():
            payload = block.payload
            if not isinstance(payload, MixedPrefixBlockPayload):
                raise RuntimeError("Mixed prefix cache block has an invalid payload.")
            total += int(payload.recurrent_bytes)
        self._recurrent_bytes_cache = (
            prefix_cache, prefix_cache.insert_epoch, prefix_cache.remove_epoch, total,
        )
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
                if self._prefix_offload_enabled():
                    if not self._evict_host_blocks(over_capacity):
                        return False
                else:
                    evicted = prefix_cache.evict_until_freeable(over_capacity)
                    self._free_prefix_cache_blocks(evicted)
                    if len(evicted) != over_capacity:
                        return False
        missing_bytes = (
            self._live_recurrent_bytes()
            + int(getattr(self, "pending_recurrent_bytes", 0))
            + incoming_recurrent_bytes
            - self.max_recurrent_bytes
        )
        if missing_bytes > 0:
            if self._prefix_offload_enabled():
                return self._evict_host_blocks(0, needed_recurrent_bytes=missing_bytes)
            byte_evicted = prefix_cache.evict_until_weight(
                missing_bytes, lambda block: int(block.payload.recurrent_bytes),
            )
            freed_bytes = sum(int(block.payload.recurrent_bytes) for block in byte_evicted)
            self._free_prefix_cache_blocks(byte_evicted)
            return freed_bytes >= missing_bytes
        return True

    def _evict_host_blocks(self, needed_blocks: int, *, needed_recurrent_bytes: int = 0) -> bool:
        controller = getattr(self, "prefix_offload_controller", None)
        if controller is None:
            raise RuntimeError("Mixed host eviction requested without an offload controller.")
        needed_weight = int(needed_recurrent_bytes or needed_blocks)
        block_weight = (
            (lambda block: int(block.payload.recurrent_bytes))
            if needed_recurrent_bytes else (lambda _block: 1)
        )
        return self._reclaim_host_prefix_weight(needed_weight, block_weight) >= needed_weight

    def evict_for_slots(self, needed_slots: int) -> None:
        if self.prefix_cache is None:
            return
        needed_slots = int(needed_slots)
        if needed_slots <= 0:
            return
        needed_blocks = (needed_slots + self.prefix_cache_block_size - 1) // self.prefix_cache_block_size
        freed_blocks = self._reclaim_device_prefix_weight(needed_blocks, lambda _block: 1)
        if freed_blocks != needed_blocks:
            raise RuntimeError(
                "Mixed prefix cache could not evict enough blocks for KV allocation: "
                f"needed_slots={needed_slots} block_size={self.prefix_cache_block_size} "
                f"needed_blocks={needed_blocks} freed_blocks={freed_blocks}."
            )

    def _free_device_prefix_blocks(self, blocks: list[PrefixCacheBlock]) -> None:
        for block in blocks:
            self._free_device_block(block)

    def _free_device_block(self, block: PrefixCacheBlock) -> None:
        payload = block.payload
        if not isinstance(payload, MixedPrefixBlockPayload):
            raise RuntimeError("Mixed prefix cache block has an invalid payload.")
        controller = self.prefix_offload_controller
        if controller is None:
            raise RuntimeError("Mixed device demotion requires an offload controller.")
        self.cache_manager.free_prefix_kv_payload_device(payload.kv_payload)
        controller.free_device_recurrent(block)

    def _free_prefix_cache_blocks(self, blocks: list[PrefixCacheBlock]) -> None:
        pending = getattr(self, "_prefix_write_through_candidates", None)
        if pending is not None:
            for block in blocks:
                pending.pop(block.stable_block_id, None)
        payloads = [block.payload for block in blocks]
        if any(
            not isinstance(payload, MixedPrefixBlockPayload)
            for payload in payloads
        ):
            raise RuntimeError("Mixed prefix cache block has an invalid payload.")
        cached = getattr(self, "_recurrent_bytes_cache", None)
        prefix_cache = self._require_prefix_cache()
        # Keep the byte snapshot valid when these removals account for every
        # index mutation since it was computed. Other mutations force a rescan.
        if (cached is not None and cached[0] is prefix_cache
                and cached[1] == prefix_cache.insert_epoch
                and cached[2] + len(blocks) == prefix_cache.remove_epoch):
            self._recurrent_bytes_cache = (
                prefix_cache, prefix_cache.insert_epoch, prefix_cache.remove_epoch,
                cached[3] - sum(int(payload.recurrent_bytes) for payload in payloads),
            )
        if self._prefix_offload_enabled():
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
                controller = self.prefix_offload_controller
                assert controller is not None
                controller.free_host_payloads(host_blocks)
            return
        for payload in payloads:
            self.cache_manager.free_prefix_kv_payload(payload.kv_payload)
            self.recurrent_state_manager.free_prefix_recurrent_payload(payload.recurrent_payload)

    def _ensure_prefix_host_capacity(self, needed_blocks: int) -> None:
        controller = self.prefix_offload_controller
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

    def release_seq(self, seq_id: int) -> None:
        released = self._release_prefix_request(int(seq_id))
        self._schedule_write_through_prefix_blocks(released)

    def _discard_pending_prefix_blocks(self, seq_id: int) -> None:
        self._release_pending_blocks(self.pending_prefix_blocks.pop(seq_id, []))
        self.pending_duplicate_refs.pop(seq_id, None)
        self.capacity_limited_seq_ids.discard(seq_id)

    def _release_pending_blocks(self, pending_blocks: list[PendingPrefixBlock]) -> None:
        for pending in pending_blocks:
            self.recurrent_state_manager.free_prefix_recurrent_payload(
                pending.payload.recurrent_payload
            )
            self._release_pending_reservation(
                pending.stable_block_id,
                int(pending.payload.recurrent_bytes),
            )

    def reset_after_warmup(self) -> None:
        if self.prefix_cache is None:
            return
        if self.seq_id_to_prefix_blocks or self.seq_id_to_materialized_blocks:
            raise RuntimeError("Cannot reset mixed prefix cache while sequences still reference blocks.")
        for pending_blocks in self.pending_prefix_blocks.values():
            self._release_pending_blocks(pending_blocks)
        blocks = list(self.prefix_cache.blocks.values())
        referenced = [block for block in blocks if int(block.ref_count) != 0]
        if referenced:
            raise RuntimeError(
                "Cannot reset mixed prefix cache while blocks are referenced: "
                f"referenced_blocks={len(referenced)}."
            )
        controller = getattr(self, "prefix_offload_controller", None)
        if controller is not None:
            controller.synchronize_all()
        self._free_prefix_cache_blocks(blocks)
        if controller is not None:
            controller.reset()
        self._reset_prefix_index()
        self.pending_duplicate_refs.clear()
        self.pending_block_ids.clear()
        self.pending_recurrent_bytes = 0
        self._recurrent_bytes_cache = None
        self.capacity_limited_seq_ids.clear()
        self.skipped_capacity_blocks = 0
