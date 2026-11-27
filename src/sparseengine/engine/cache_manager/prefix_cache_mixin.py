from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from weakref import ReferenceType, WeakKeyDictionary, ref

import torch

from sparseengine.engine.prefix_cache import (
    PrefixBlockPayload,
    PrefixCacheBlock,
    PrefixTransferKind,
    RadixPrefixIndex,
    build_prefix_cache_fingerprint,
    select_write_through_candidates,
    usable_prefix_cache_tokens,
)
from sparseengine.engine.prefix_prune import PrefixPruneRecord, normalize_prefix_prune_ranges
from sparseengine.engine.sequence import Sequence
from sparseengine.platforms import device_runtime
from sparseengine.utils.profiler import cpu_timing, profiler

from .prefix_offload import PrefixH2DOperation, PrefixOffloadController


@dataclass
class PrefixRuntimeState:
    parent_block_id: bytes | None
    next_logical_block_idx: int
    pending_tokens: list[int]
    pending_slots: list[torch.Tensor]


@dataclass
class PendingPrefixBlock:
    stable_block_id: bytes
    parent_block_id: bytes | None
    logical_block_idx: int
    payload: PrefixBlockPayload
    slots: torch.Tensor
    token_ids: list[int]


@dataclass(frozen=True)
class PrefixLookupCacheEntry:
    token_ids: tuple[int, ...]
    usable_tokens: int
    block_ids: tuple[bytes, ...]
    result: tuple[int, bytes | None, int]


class _ResidentWeightCache:
    def __init__(self, index: RadixPrefixIndex) -> None:
        self.index = index
        self.payload_epoch = index.resident_payload_epoch
        self.totals: dict[str, tuple[frozenset[bytes], int]] = {}
        self.weights: dict[bytes, int] = {}

    def discard(self, removed_ids: set[bytes]) -> None:
        for view, (block_ids, total) in self.totals.items():
            removed = block_ids & removed_ids
            if removed:
                self.totals[view] = (
                    block_ids - removed,
                    total - sum(self.weights[block_id] for block_id in removed),
                )
        for block_id in removed_ids:
            self.weights.pop(block_id, None)


class PrefixLookupCache:
    """Live requests, the last lookup batch, and bounded individual TP lookups."""

    def __init__(self, max_entries: int = 128):
        if max_entries <= 0:
            raise ValueError("Prefix lookup cache capacity must be positive.")
        self.max_entries = max_entries
        self.entries: OrderedDict[int, PrefixLookupCacheEntry] = OrderedDict()
        self._live_entries: dict[int, tuple[ReferenceType[Sequence], PrefixLookupCacheEntry]] = {}
        self._batch_entries: OrderedDict[int, PrefixLookupCacheEntry] | None = None

    @contextmanager
    def batch(self, *, first: bool, last: bool):
        # Keep the previous batch readable until every RPC fragment completes.
        # Evicting during a fragment can evict later requests in the same pass.
        if first:
            self._batch_entries = OrderedDict()
        elif self._batch_entries is None:
            raise RuntimeError("Prefix lookup batch continuation has no first fragment.")
        try:
            yield
        except BaseException:
            self._batch_entries = None
            raise
        else:
            if last:
                assert self._batch_entries is not None
                self.entries = self._batch_entries
                self._batch_entries = None

    def _remember_live(self, seq: Sequence, entry: PrefixLookupCacheEntry) -> None:
        seq_id = seq.seq_id
        previous = self._live_entries.get(seq_id)
        if previous is not None and previous[0]() is seq:
            self._live_entries[seq_id] = previous[0], entry
            return
        cache_ref = ref(self)

        def discard_dead(owner: ReferenceType[Sequence]) -> None:
            cache = cache_ref()
            if cache is not None:
                current = cache._live_entries.get(seq_id)
                # A deserialized replacement may already own the same ID.
                if current is not None and current[0] is owner:
                    cache._live_entries.pop(seq_id)

        self._live_entries[seq_id] = ref(seq, discard_dead), entry

    def get(self, seq: Sequence) -> PrefixLookupCacheEntry | None:
        live = self._live_entries.get(seq.seq_id)
        if live is not None and live[0]() is seq:
            entry = live[1]
        else:
            entry = (self._batch_entries.get(seq.seq_id)
                     if self._batch_entries is not None else None)
            if entry is None:
                entry = self.entries.get(seq.seq_id)
                if entry is not None:
                    self.entries.move_to_end(seq.seq_id)
            if entry is not None:
                self._remember_live(seq, entry)
        if entry is not None and self._batch_entries is not None:
            self._batch_entries[seq.seq_id] = entry
        return entry

    def __setitem__(self, seq: Sequence, entry: PrefixLookupCacheEntry) -> None:
        self._remember_live(seq, entry)
        if self._batch_entries is not None:
            self._batch_entries[seq.seq_id] = entry
            return
        self.entries[seq.seq_id] = entry
        self.entries.move_to_end(seq.seq_id)
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)

    def discard(self, seq_id: int) -> None:
        self._live_entries.pop(seq_id, None)
        self.entries.pop(seq_id, None)
        if self._batch_entries is not None:
            self._batch_entries.pop(seq_id, None)


@dataclass(frozen=True)
class PrefixHitCapacityCacheEntry:
    prefix_cache: RadixPrefixIndex
    last_block_id: bytes
    hit_blocks: int
    remove_epoch: int
    capacity_epoch: int
    offload_enabled: bool
    chain: tuple[PrefixCacheBlock, ...]
    reclaimable_blocks: int
    promotion_blocks: int
    weighted_slots: tuple[int, int] | None = None


def lookup_prefix_cache_hit(
    prefix_cache: RadixPrefixIndex,
    cache: PrefixLookupCache,
    seq: Sequence,
    usable_tokens: int,
) -> tuple[int, bytes | None, int]:
    prompt_token_ids = seq.prompt_token_ids
    entry = cache.get(seq)
    if (
        entry is not None
        and (entry.token_ids is prompt_token_ids or entry.token_ids == prompt_token_ids)
        and entry.usable_tokens == int(usable_tokens)
    ):
        hit_blocks = int(entry.result[2])
        # Indexed paths are root-contiguous and deletion is leaf-only. If the
        # last hit survives, all its ancestors survive too, even if unrelated
        # branches were evicted. Reinserting the same logical ID is also safe:
        # this memo contains IDs/counts, never payload objects.
        lookup_is_current = (
            (hit_blocks == 0 or prefix_cache.has_block(entry.block_ids[hit_blocks - 1]))
            and (
                hit_blocks == len(entry.block_ids)
                or not prefix_cache.has_block(entry.block_ids[hit_blocks])
            )
        )
        if lookup_is_current:
            return entry.result
        block_ids = entry.block_ids
    else:
        if not isinstance(prompt_token_ids, tuple):
            prompt_token_ids = tuple(int(token_id) for token_id in prompt_token_ids)
        block_ids = tuple(
            prefix_cache.block_ids_for_tokens(
                prompt_token_ids,
                max_tokens=int(usable_tokens),
            )
        )

    result = prefix_cache.lookup_longest_block_ids(block_ids)
    cache[seq] = PrefixLookupCacheEntry(
        token_ids=prompt_token_ids,
        usable_tokens=int(usable_tokens),
        block_ids=block_ids,
        result=result,
    )
    return result


def prefix_hit_capacity_counts(
    prefix_cache: RadixPrefixIndex,
    cache: WeakKeyDictionary[Sequence, PrefixHitCapacityCacheEntry],
    seq: Sequence,
    *,
    offload_enabled: bool,
) -> tuple[int, int]:
    hit_blocks = int(getattr(seq, "prefix_cache_hit_block_count", 0) or 0)
    last_block_id = getattr(seq, "prefix_cache_hit_last_block_id", None)
    if hit_blocks <= 0:
        return 0, 0
    if last_block_id is None:
        raise RuntimeError(
            f"seq_id={seq.seq_id} has prefix hit blocks but no last block id."
        )

    try:
        entry = cache.get(seq)
        cacheable = True
    except TypeError:
        entry = None
        cacheable = False
    same_chain = (
        entry is not None
        and entry.prefix_cache is prefix_cache
        and entry.last_block_id == last_block_id
        and entry.hit_blocks == hit_blocks
        and entry.remove_epoch == prefix_cache.remove_epoch
    )
    if (
        same_chain
        and entry is not None
        and entry.capacity_epoch == prefix_cache.capacity_epoch
        and entry.offload_enabled == bool(offload_enabled)
    ):
        return entry.reclaimable_blocks, entry.promotion_blocks

    chain = (
        entry.chain
        if same_chain and entry is not None
        else tuple(prefix_cache.get_chain(last_block_id, hit_blocks))
    )
    freeable_block_ids = (
        prefix_cache.device_reclaimable_block_ids()
        if offload_enabled
        else prefix_cache.freeable_block_ids()
    )
    reclaimable_blocks = sum(
        1 for block in chain if block.stable_block_id in freeable_block_ids
    )
    promotion_blocks = (
        sum(1 for block in chain if not block.residency.device_present)
        if offload_enabled
        else 0
    )
    if cacheable:
        cache[seq] = PrefixHitCapacityCacheEntry(
            prefix_cache=prefix_cache,
            last_block_id=last_block_id,
            hit_blocks=hit_blocks,
            remove_epoch=prefix_cache.remove_epoch,
            capacity_epoch=prefix_cache.capacity_epoch,
            offload_enabled=bool(offload_enabled),
            chain=chain,
            reclaimable_blocks=reclaimable_blocks,
            promotion_blocks=promotion_blocks,
        )
    return reclaimable_blocks, promotion_blocks


class PrefixCacheMixin:
    """Own Radix transactions over manager-owned token slots or pages.

    Storage hooks validate and pack payloads, bind and clear row storage, and
    return device units to the allocator. The mixin owns radix refs and transfer
    submission. A submitted H2D destination remains owned by the index.
    """

    _prefix_label = "Standard prefix"
    _prefix_profile = "prefix_cache"
    _prefix_device_unit_tokens = 1
    _prefix_unit_name = "block"
    _prefix_hit_capacity_device_resident_only = False

    @property
    def _radix_block_size(self) -> int:
        return self.prefix_cache_block_size

    def _prefix_hit_method(self) -> str:
        return str(self.config.sparse_method or "")

    def _init_radix_prefix_cache(self) -> None:
        config = self.config
        self.prefix_cache: RadixPrefixIndex | None = None
        if self.enable_prefix_caching:
            block_size = self.prefix_cache_block_size
            self.prefix_cache = RadixPrefixIndex(
                block_size=block_size,
                fingerprint=build_prefix_cache_fingerprint(config, block_size),
                max_blocks=config.prefix_cache_max_blocks,
            )
        self.seq_id_to_prefix_blocks: dict[int, list[PrefixCacheBlock]] = {}
        self._scheduler_capacity_snapshot_depth = 0
        self._scheduler_freeable_block_ids: frozenset[bytes] | None = None
        self._scheduler_reclaimable_units: int | None = None
        self._init_prefix_cache_runtime()
        self.prefix_offload_controller: PrefixOffloadController | None = None
        self._prefix_offload_step_h2d_operations: dict[int, PrefixH2DOperation] = {}
        self._prefix_write_through_candidates: dict[bytes, PrefixCacheBlock] = {}
        layout = getattr(config, "runtime_layout", None)
        if (
            getattr(config, "enable_prefix_cache_offload", False)
            and not getattr(layout, "linear_attention_layer_indices", ())
        ):
            self._init_prefix_offload()

    def _prefix_storage_components(self):
        from .offload.prefix_components import storage_prefix_components

        return storage_prefix_components(self.attention_cache_storage, self.num_kv_layers)

    def _prefix_offload_enabled(self) -> bool:
        return getattr(self, "prefix_offload_controller", None) is not None

    def _poll_prefix_offload(self) -> None:
        controller = getattr(self, "prefix_offload_controller", None)
        if controller is not None:
            controller.poll()

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

    def _prefix_hit_chain(self, seq: Sequence) -> list[PrefixCacheBlock]:
        if self.prefix_cache is None or seq.prefix_cache_hit_last_block_id is None:
            return []
        return self.prefix_cache.get_chain(
            seq.prefix_cache_hit_last_block_id,
            int(seq.prefix_cache_hit_block_count),
        )

    def _free_device_prefix_block(self, block: PrefixCacheBlock) -> None:
        self._free_device_prefix_blocks([block])

    def _on_prefix_cache_reset(self) -> None:
        self._prefix_resident_weight_cache = None
        controller = getattr(self, "prefix_offload_controller", None)
        if controller is not None:
            controller.prefix_cache = self._require_prefix_cache()

    def allocate_prefix_kv_payload_device(self, payload: object) -> None:
        self.allocate_prefix_kv_payloads_device([payload])

    def prompt_admission_free_slots(self) -> int:
        return int(self.num_free_slots + self._prefix_step_reclaimable_slots())

    def prompt_logical_reservation_cost(self, seq: Sequence) -> int:
        return int(self.prompt_admission_cost(seq))

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
                self._prefix_block_resident_tokens(block)
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

    @cpu_timing.timed
    def refresh_prefix_cache_hit(self, seq: Sequence) -> None:
        self._poll_prefix_offload()
        self.clear_prefix_cache_hit(seq)
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        if seq.num_prefilled_tokens != 0 or seq.num_completion_tokens != 0:
            return
        block_size = self._radix_block_size
        usable_tokens = usable_prefix_cache_tokens(seq.num_prompt_tokens, block_size)
        if usable_tokens <= 0:
            return
        with profiler.record(f"{self._prefix_profile}_lookup"):
            hit_len, last_block_id, hit_blocks = self._lookup_prefix_cache_hit(
                seq,
                usable_tokens,
            )
        if hit_len <= 0:
            return
        if last_block_id is None or hit_blocks <= 0:
            raise RuntimeError(f"{self._prefix_label} cache lookup returned an invalid hit.")
        if hit_len >= seq.num_prompt_tokens or hit_len % block_size != 0:
            raise RuntimeError(
                f"{self._prefix_label} cache lookup returned an unusable hit length: "
                f"seq_id={seq.seq_id} hit_len={hit_len} prompt_len={seq.num_prompt_tokens} "
                f"{self._prefix_unit_name}_size={block_size}."
            )
        seq.prefix_cache_enabled = True
        seq.prefix_cache_hit_len = int(hit_len)
        seq.prefix_cache_hit_block_count = int(hit_blocks)
        seq.prefix_cache_hit_last_block_id = last_block_id
        seq.prefix_cache_block_size = block_size
        seq.prefix_cache_method = self._prefix_hit_method()

    def _prefix_evictable_slots(self) -> int:
        if getattr(self, "prefix_cache", None) is None:
            return 0
        block_ids = (
            self.prefix_cache.device_freeable_block_ids()
            if self._prefix_offload_enabled()
            else self._prefix_freeable_block_ids_for_capacity()
        )
        units = self._prefix_resident_units_for_ids(block_ids, view="evictable")
        return units * self._prefix_device_unit_tokens

    @contextmanager
    def scheduler_capacity_snapshot(self):
        """Reuse one immutable prefix-capacity view within a scheduler pass."""
        self._scheduler_capacity_snapshot_depth += 1
        if self._scheduler_capacity_snapshot_depth == 1:
            self._scheduler_freeable_block_ids = None
            self._scheduler_reclaimable_units = None
        try:
            yield
        finally:
            self._scheduler_capacity_snapshot_depth -= 1
            if self._scheduler_capacity_snapshot_depth == 0:
                self._scheduler_freeable_block_ids = None
                self._scheduler_reclaimable_units = None

    def _prefix_step_reclaimable_slots(self) -> int:
        return self._prefix_step_reclaimable_units() * self._prefix_device_unit_tokens

    def _prefix_step_reclaimable_units(self) -> int:
        if getattr(self, "prefix_cache", None) is None:
            return 0
        if (
            self._scheduler_capacity_snapshot_depth > 0
            and self._scheduler_reclaimable_units is not None
        ):
            return self._scheduler_reclaimable_units
        block_ids = (
            self.prefix_cache.device_reclaimable_block_ids()
            if self._prefix_offload_enabled()
            else self._prefix_freeable_block_ids_for_capacity()
        )
        reclaimable_units = self._prefix_resident_units_for_ids(block_ids, view="reclaimable")
        if self._scheduler_capacity_snapshot_depth > 0:
            self._scheduler_reclaimable_units = reclaimable_units
        return reclaimable_units

    def _prefix_resident_units_for_ids(
        self, block_ids: frozenset[bytes], *, view: str
    ) -> int:
        return self._prefix_resident_slots_for_ids(block_ids, view=view)

    def _prefix_block_capacity_weight(self, block: PrefixCacheBlock) -> int:
        return self._prefix_block_resident_tokens(block) // self._prefix_device_unit_tokens

    def _prefix_hit_capacity_units(self, seq: Sequence) -> tuple[int, int]:
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
        device_resident_only = self._prefix_hit_capacity_device_resident_only
        reclaimable = sum(
            self._prefix_block_capacity_weight(block)
            for block in chain
            if block.stable_block_id in freeable_ids
            and (not device_resident_only or block.residency.device_present)
        )
        promotion = (
            sum(
                self._prefix_block_capacity_weight(block)
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

    def _free_prefix_cache_blocks(self, blocks: list[PrefixCacheBlock]) -> None:
        pending = getattr(self, "_prefix_write_through_candidates", None)
        host_blocks: list[PrefixCacheBlock] = []
        device_blocks: list[PrefixCacheBlock] = []
        for block in blocks:
            if pending is not None:
                pending.pop(block.stable_block_id, None)
            if not isinstance(block.payload, self._prefix_payload_type):
                raise RuntimeError(
                    f"{self._prefix_label} cache block is missing its payload."
                )
            if block.residency.device_present:
                device_blocks.append(block)
            if block.residency.host_present:
                host_blocks.append(block)
        self._free_device_prefix_blocks(device_blocks)
        controller = getattr(self, "prefix_offload_controller", None)
        if host_blocks:
            if controller is None:
                raise RuntimeError(
                    f"{self._prefix_label} blocks have host payloads but no offload controller is active."
                )
            controller.free_host_payloads(host_blocks)

    @cpu_timing.timed
    def _evict_prefix_cache_until_free(self, needed_slots: int) -> None:
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        needed_slots = int(needed_slots)
        if self.num_free_slots >= needed_slots:
            return
        if self._prefix_offload_enabled():
            controller = self.prefix_offload_controller
            assert controller is not None
            while self.num_free_slots < needed_slots:
                self._poll_prefix_offload()
                missing_slots = needed_slots - int(self.num_free_slots)
                with profiler.record(f"{self._prefix_profile}_device_demote"):
                    demoted = self.prefix_cache.demote_device_until_weight(
                        self._prefix_device_units_for_tokens(missing_slots),
                        self._prefix_block_capacity_weight,
                    )
                self._free_device_prefix_blocks(demoted)
                if self.num_free_slots >= needed_slots:
                    return
                if not controller.wait_oldest_d2h():
                    break
            return

        missing_slots = needed_slots - int(self.num_free_slots)
        with profiler.record(f"{self._prefix_profile}_evict"):
            evicted = self.prefix_cache.evict_until_weight(
                self._prefix_device_units_for_tokens(missing_slots),
                self._prefix_block_capacity_weight,
            )
        self._free_prefix_cache_blocks(evicted)

    def _prefix_device_units_for_tokens(self, token_count: int) -> int:
        unit = self._prefix_device_unit_tokens
        return (token_count + unit - 1) // unit

    @cpu_timing.timed
    def _evict_prefix_cache_for_insert(self, needed_blocks: int = 1) -> None:
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        if not self._prefix_offload_enabled():
            with profiler.record(f"{self._prefix_profile}_evict"):
                evicted = self.prefix_cache.ensure_insert_capacity(needed_blocks)
            self._free_prefix_cache_blocks(evicted)
            return
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
            with profiler.record(f"{self._prefix_profile}_host_evict"):
                host_evicted = self.prefix_cache.evict_host_until_freeable(remaining)
            self._free_prefix_cache_blocks(host_evicted)
            evicted.extend(host_evicted)
            if len(evicted) >= over_capacity:
                break
            with profiler.record(f"{self._prefix_profile}_device_demote"):
                demoted = self.prefix_cache.demote_device_until_freeable(remaining)
            self._free_device_prefix_blocks(demoted)
            with profiler.record(f"{self._prefix_profile}_host_evict"):
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
                f"{self._prefix_label} logical capacity exceeded and not enough CPU-only leaves "
                "are evictable: "
                f"live_blocks={len(self.prefix_cache)} max_blocks={max_blocks} "
                f"needed_blocks={needed_blocks} evicted_blocks={len(evicted)}."
            )

    def _ensure_prefix_host_capacity(self, needed_blocks: int) -> None:
        controller = self.prefix_offload_controller
        if controller is None:
            raise RuntimeError(f"{self._prefix_label} host capacity requested without an offload controller.")
        needed_blocks = int(needed_blocks)
        if controller.host_pool.free_blocks >= needed_blocks:
            return
        missing = needed_blocks - controller.host_pool.free_blocks
        with profiler.record(f"{self._prefix_profile}_host_evict"):
            evicted = self._require_prefix_cache().evict_host_until_freeable(missing)
        self._free_prefix_cache_blocks(evicted)
        if controller.host_pool.free_blocks < needed_blocks:
            raise RuntimeError(
                f"{self._prefix_label} host pool cannot preserve write-through residency: "
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
            raise RuntimeError(f"{self._prefix_label} D2H scheduling is forbidden during graph capture.")
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
        with profiler.record(f"{self._prefix_profile}_d2h_submit"):
            controller.submit_d2h(selected)
        for block in selected:
            pending.pop(block.stable_block_id, None)

    def _init_prefix_offload(self) -> None:
        if not self.enable_prefix_caching or self.prefix_cache is None:
            raise RuntimeError(f"{self._prefix_label} cache offload requires an active {self._prefix_label} cache.")
        if self.tp_size not in (1, 2):
            raise RuntimeError("Prefix cache offload currently supports only TP=1 or TP=2.")
        if not device_runtime.supports_pin_memory():
            raise RuntimeError("Prefix cache offload requires pinned host memory support.")
        if not device_runtime.supports_streams(self.device):
            raise RuntimeError("Prefix cache offload requires asynchronous device streams.")
        host_size_gb = getattr(self.config, "prefix_cache_host_size_gb", None)
        if host_size_gb is None:
            raise RuntimeError("Prefix cache offload requires prefix_cache_host_size_gb.")
        from .offload.prefix_components import (
            ComponentPrefixOffloadController,
            ComponentPrefixPool,
            prefix_block_bytes,
        )
        from .storage import ExplicitKVStorage

        components = self._prefix_storage_components()
        block_size = self._radix_block_size
        bytes_per_block = prefix_block_bytes(components, block_size)
        host_bytes = int(float(host_size_gb) * (1024**3))
        host_capacity_blocks = host_bytes // bytes_per_block
        required_blocks = self._prefix_offload_device_blocks()
        if self.prefix_cache.max_blocks is not None:
            required_blocks = min(required_blocks, int(self.prefix_cache.max_blocks))
        if host_capacity_blocks < required_blocks:
            raise RuntimeError(
                f"{self._prefix_label} host tier is too small for write-through safety: "
                f"host_blocks={host_capacity_blocks} required_blocks={required_blocks} "
                f"bytes_per_block={bytes_per_block} host_size_gb={host_size_gb}."
            )
        if not isinstance(self.attention_cache_storage, ExplicitKVStorage):
            host_pool = ComponentPrefixPool(
                components=components,
                capacity_blocks=host_capacity_blocks,
                block_size=block_size,
            )
            self.prefix_offload_controller = ComponentPrefixOffloadController(
                components=components,
                prefix_cache=self.prefix_cache,
                host_pool=host_pool,
                block_size=block_size,
                device=self.device,
            )
            return
        self.prefix_offload_controller = self._build_explicit_prefix_offload_controller(
            host_capacity_blocks,
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
            token_count=len(token_ids), block_size=self._radix_block_size,
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
        block_size = int(self._radix_block_size)
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
        if any(self._prefix_block_payload(block).retained_offsets is not None for block in affected):
            raise RuntimeError(
                "prefix prune target overlaps an already pruned block; "
                "repeated compression of the same interval is not implemented."
            )
        return affected

    def _prefix_prune_prefix_logical_positions(self, seq_id: int) -> list[int]:
        positions: list[int] = []
        for block in self.seq_id_to_prefix_blocks[int(seq_id)]:
            payload = self._prefix_block_payload(block)
            base = int(block.logical_block_idx) * self._radix_block_size
            offsets = payload.retained_offsets
            if offsets is None:
                positions.extend(range(base, base + self._radix_block_size))
            else:
                positions.extend(base + offset for offset in offsets)
        return positions

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
            token_count=len(token_ids), block_size=self._radix_block_size,
            range_start=range_start, range_end=range_end, ranges=ranges,
        )
        affected = self.validate_prefix_cache_prune_target(
            token_ids, ranges=intervals, allow_recompress=allow_recompress,
        )
        prefix_cache = self._require_prefix_cache()
        block_size = int(self._radix_block_size)
        width = sum(right - left for left, right in intervals)
        # One host transfer for all interval selections, not one per block/range.
        selected = sorted({int(index) for index in keep_indices.detach().cpu().tolist()})
        if any(index < 0 or index >= width for index in selected):
            raise ValueError("prefix prune keep mask contains an out-of-range token index.")
        if len(selected) != int(keep_indices.numel()):
            raise ValueError("prefix prune keep mask contains duplicate token indices.")
        self._validate_prefix_prune_selection(selected)
        selected_set = set(selected)

        plans = []
        for relative_block_idx, block in enumerate(affected):
            payload = self._prefix_block_payload(block)
            old_offsets = (
                tuple(range(block_size)) if payload.retained_offsets is None
                else tuple(int(offset) for offset in payload.retained_offsets)
            )
            block_base = relative_block_idx * block_size
            new_offsets = tuple(
                offset for offset in old_offsets if block_base + offset in selected_set
            )
            plans.append((block, new_offsets))

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

        compaction = self._compact_prefix_prune_payloads(plans)
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
            **compaction,
            "affected_blocks": len(affected),
            "quality_degraded": True,
        }

    def _validate_prefix_prune_selection(self, selected: list[int]) -> None:
        pass

    def validate_prefix_kv_attach(self, seq: Sequence) -> bool:
        return self._validate_prefix_attach_row(seq, mixed=True) is not None

    def _validate_prefix_attach_row(self, seq: Sequence, *, mixed: bool) -> int | None:
        row_idx = self.seq_id_to_row.get(int(seq.seq_id))
        if row_idx is not None and int(self.row_seq_lens[row_idx]) != 0:
            subject = f"mixed {self._prefix_label.lower()} KV" if mixed else f"{self._prefix_label.lower()} cache"
            raise RuntimeError(
                f"Cannot attach {subject} to a non-empty row: "
                f"seq_id={seq.seq_id} row_idx={row_idx} "
                f"row_len={int(self.row_seq_lens[row_idx])}."
            )
        if row_idx is None and not self.free_rows:
            raise RuntimeError("No free rows in cache manager buffer!")
        return row_idx

    def _attach_prefix_cache_if_needed(self, seq: Sequence) -> None:
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        hit_len = int(getattr(seq, "prefix_cache_hit_len", 0) or 0)
        if hit_len <= 0 or seq.seq_id in self.seq_id_to_prefix_blocks:
            return
        prepared = getattr(seq, "prefix_prune_prefix", None)
        if prepared is not None and prepared.token_slots is not None:
            self.attach_prefix_prune_prefix(seq, prepared)
            return

        self._poll_prefix_offload()
        with profiler.record(f"{self._prefix_profile}_attach"):
            self._attach_radix_prefix_chain(seq, hit_len)

    def _attach_radix_prefix_chain(self, seq: Sequence, hit_len: int) -> None:
        prefix_cache = self._require_prefix_cache()
        seq_id = seq.seq_id
        block_size = self._radix_block_size
        label = self._prefix_label.lower()
        kind = self._prefix_unit_name
        if seq.prefix_cache_hit_last_block_id is None:
            raise RuntimeError(f"seq_id={seq_id} has {label} hit length but no last block id.")
        if hit_len % block_size != 0:
            raise RuntimeError(
                f"seq_id={seq_id} {label} hit length is not {kind} aligned: "
                f"hit_len={hit_len} {kind}_size={block_size}."
            )
        chain = prefix_cache.get_chain(
            seq.prefix_cache_hit_last_block_id, int(seq.prefix_cache_hit_block_count)
        )
        if len(chain) * block_size != hit_len:
            raise RuntimeError(
                f"{self._prefix_label} cache chain length does not match scheduler metadata: "
                f"seq_id={seq_id} hit_len={hit_len} blocks={len(chain)} {kind}_size={block_size}."
            )

        resident_blocks: list[PrefixCacheBlock] = []
        host_blocks: list[PrefixCacheBlock] = []
        operations: dict[int, PrefixH2DOperation] = {}
        resident_tokens = promotion_tokens = 0
        saw_cpu_only = False
        controller = getattr(self, "prefix_offload_controller", None)
        for block_index, block in enumerate(chain):
            payload = block.payload
            if not isinstance(payload, self._prefix_payload_type):
                raise RuntimeError(
                    f"Invalid {label} payload for seq_id={seq_id}: "
                    f"logical_block_idx={block.logical_block_idx}."
                )
            self._validate_radix_block_index(block, block_index)
            residency = block.residency
            residency.validate()
            expected_tokens = payload.resident_tokens(block_size)
            resident_tokens += expected_tokens
            if self._prepare_empty_radix_payload(block, expected_tokens):
                if not residency.device_present:
                    host_blocks.append(block)
                continue
            if not residency.device_present:
                saw_cpu_only = True
                if controller is None or not residency.host_present:
                    raise RuntimeError(
                        f"{self._prefix_label} lookup returned a non-device block that cannot be promoted: "
                        f"seq_id={seq_id} block={block.stable_block_id.hex()[:16]}."
                    )
                if residency.transfer is not None:
                    raise RuntimeError(
                        f"CPU-only {label} block has an unexpected in-flight transfer: "
                        f"seq_id={seq_id} transfer={residency.transfer.value}."
                    )
                if payload.host_block_index is None:
                    raise RuntimeError(
                        f"CPU-only {label} block is missing its host allocation: "
                        f"seq_id={seq_id} block={block.stable_block_id.hex()[:16]}."
                    )
                host_blocks.append(block)
                promotion_tokens += expected_tokens
                continue
            if saw_cpu_only:
                raise RuntimeError(
                    f"{self._prefix_label} device residency is not root-contiguous: "
                    f"seq_id={seq_id} block={block.stable_block_id.hex()[:16]}."
                )
            self._validate_resident_radix_payload(seq_id, block, expected_tokens)
            if expected_tokens:
                resident_blocks.append(block)
            if residency.transfer == PrefixTransferKind.H2D:
                assert controller is not None
                operation = controller.h2d_operation_for_block(block)
                if operation is None:
                    raise RuntimeError(
                        f"{self._prefix_label} block is promoting without a tracked H2D operation: "
                        f"block={block.stable_block_id.hex()[:16]}."
                    )
                operations[id(operation)] = operation

        # Pack physical IDs before refs or allocations change. QuEST validates
        # page geometry in one device read for the entire resident prefix.
        slot_parts, device_units = self._pack_radix_prefix_blocks(resident_blocks)
        previous_row = self._validate_prefix_attach_row(seq, mixed=False)
        self._validate_prefix_row_capacity(seq_id, resident_tokens, mixed=False)
        old_logical_len = self._prefix_attach_rollback_logical_len(previous_row)
        cached_rows = self._prefix_cached_rows
        old_cached = cached_rows.get(seq_id)
        allocated_units = None
        promotion_committed = False
        acquired = 0
        try:
            for block in chain:
                prefix_cache.acquire_block_ref(block)
                acquired += 1
            if host_blocks:
                if device_runtime.is_stream_capturing():
                    raise RuntimeError(
                        f"{self._prefix_label} H2D promotion is forbidden during graph capture."
                    )
                assert controller is not None
                allocated_units = self._take_prefix_device_units(
                    self._prefix_device_units_for_tokens(promotion_tokens)
                )
                promoted_slots, promoted_units = self._stage_radix_prefix_blocks(
                    host_blocks, allocated_units
                )
                with profiler.record(f"{self._prefix_profile}_h2d_submit"):
                    operation = controller.submit_h2d(host_blocks)
                    # Empty-only promotions can commit without a copy ticket.
                    promotion_committed = True
                if operation is not None:
                    operations[id(operation)] = operation
                if promoted_slots is not None:
                    slot_parts.append(promoted_slots)
                if promoted_units is not None:
                    device_units = (
                        promoted_units if device_units is None
                        else torch.cat((device_units, promoted_units))
                    )

            row_idx = self._get_free_row(seq_id)
            self._prefix_offload_step_h2d_operations.update(operations)
            if slot_parts:
                token_slots = slot_parts[0] if len(slot_parts) == 1 else torch.cat(slot_parts)
                self._write_prefix_row_slots(row_idx, 0, resident_tokens, token_slots, device_units)
            self.row_seq_lens[row_idx] = resident_tokens
            self.row_logical_lens[row_idx] = hit_len
            cached_rows[seq_id] = self._prefix_cached_entries(
                [(0, resident_tokens)] if resident_tokens else []
            )
            self.seq_id_to_prefix_blocks[seq_id] = chain
            prefix_cache.touch_chain(chain)
        except BaseException:
            # These are cache aliases. Never release them via the private-row
            # allocator, even if the failure occurred after row publication.
            row_idx = self.seq_id_to_row.get(seq_id)
            if row_idx is not None:
                self._clear_prefix_row_slots(row_idx, 0)
                self.row_seq_lens[row_idx] = 0
                self.row_logical_lens[row_idx] = old_logical_len
                if previous_row is None:
                    self.seq_id_to_row.pop(seq_id, None)
                    self.free_rows.appendleft(row_idx)
            self.seq_id_to_prefix_blocks.pop(seq_id, None)
            if old_cached is None:
                cached_rows.pop(seq_id, None)
            else:
                cached_rows[seq_id] = old_cached
            for block in chain[:acquired]:
                prefix_cache.release_block_ref(block)
            if allocated_units is not None and not promotion_committed:
                self._return_prefix_device_units(allocated_units)
                self._clear_radix_promotion_payloads(host_blocks)
            # Submitted destinations stay index-owned until H2D retirement.
            raise

    def _validate_radix_block_index(self, block: PrefixCacheBlock, block_index: int) -> None:
        pass

    def _prepare_empty_radix_payload(self, block: PrefixCacheBlock, expected_tokens: int) -> bool:
        return False

    def _prefix_attach_rollback_logical_len(self, previous_row: int | None) -> int:
        return 0 if previous_row is None else int(self.row_logical_lens[previous_row])

    def _pack_radix_prefix_blocks(
        self, blocks: list[PrefixCacheBlock],
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        return [block.payload.token_slots for block in blocks], None

    def _clear_radix_promotion_payloads(self, blocks: list[PrefixCacheBlock]) -> None:
        for block in blocks:
            block.payload.token_slots = None

    def _write_prefix_row_slots(
        self, row_idx: int, start: int, end: int,
        slots: torch.Tensor, device_units: torch.Tensor | None,
    ) -> None:
        self.buffer_req_to_token_slots[row_idx, start:end].copy_(slots)

    def _clear_prefix_row_slots(
        self, row_idx: int, start: int, end: int | None = None
    ) -> None:
        self.buffer_req_to_token_slots[row_idx, start:end].zero_()

    def attach_prefix_kv_payloads(self, seq: Sequence, payloads: list[object]) -> None:
        if not payloads:
            return
        seq_id = int(seq.seq_id)
        previous_row = self.seq_id_to_row.get(seq_id)
        start = 0 if previous_row is None else int(self.row_seq_lens[previous_row])
        end = start
        slot_parts = []
        unit_parts = []
        ranges = []
        for payload in payloads:
            self._validate_mixed_prefix_payload_type(payload)
            if not isinstance(payload.token_slots, torch.Tensor):
                raise RuntimeError(f"{self._prefix_label} mixed KV payload has no device slots.")
            slots = payload.token_slots.to(device=self.device, dtype=torch.int32).reshape(-1)
            units = self._mixed_prefix_attach_units(seq_id, end, payload, slots)
            next_end = end + int(slots.numel())
            if units is not None:
                unit_parts.append(units)
            slot_parts.append(slots)
            ranges.append((end, next_end))
            end = next_end
        self._validate_prefix_row_capacity(seq_id, end, mixed=True)
        slots = slot_parts[0] if len(slot_parts) == 1 else torch.cat(slot_parts)
        units = self._pack_mixed_prefix_units(start, slots, unit_parts)
        try:
            row_idx = self._get_free_row(seq_id)
            self._write_prefix_row_slots(row_idx, start, end, slots, units)
        except BaseException:
            row_idx = self.seq_id_to_row.get(seq_id)
            if row_idx is not None:
                self._clear_prefix_row_slots(row_idx, start, end)
                if previous_row is None:
                    self.seq_id_to_row.pop(seq_id)
                    self.free_rows.appendleft(row_idx)
            raise
        self._mark_mixed_prefix_ranges(seq_id, ranges)
        self.row_seq_lens[row_idx] = end
        self.row_logical_lens[row_idx] = end

    def _pack_mixed_prefix_units(
        self, start: int, slots: torch.Tensor, unit_parts: list[torch.Tensor]
    ) -> torch.Tensor | None:
        return None

    def _validate_mixed_prefix_payload_type(self, payload: object) -> None:
        if not isinstance(payload, self._prefix_payload_type):
            raise RuntimeError(f"{self._prefix_label} mixed KV payload is missing token slots.")

    def rollback_prefix_kv_attach(
        self,
        seq: Sequence,
        payloads: list[object],
        *,
        row_preexisted: bool,
    ) -> None:
        ranges: list[tuple[int, int]] = []
        expected_start = 0
        for payload in payloads:
            if not isinstance(payload, self._prefix_payload_type):
                raise RuntimeError(f"{self._prefix_label} mixed rollback received an invalid payload.")
            if int(payload.block_start) != expected_start or int(payload.block_end) <= expected_start:
                raise RuntimeError(
                    f"{self._prefix_label} mixed rollback payloads are not contiguous: "
                    f"expected_start={expected_start} "
                    f"range={int(payload.block_start)}:{int(payload.block_end)}."
                )
            ranges.append((int(payload.block_start), int(payload.block_end)))
            expected_start = int(payload.block_end)
        if not ranges:
            return

        seq_id = int(seq.seq_id)
        row_idx = self.seq_id_to_row.get(seq_id)
        if row_idx is None or int(self.row_seq_lens[row_idx]) != expected_start:
            raise RuntimeError(
                f"{self._prefix_label} mixed rollback row state is inconsistent: "
                f"seq_id={seq_id} row_idx={row_idx} "
                f"row_len={None if row_idx is None else int(self.row_seq_lens[row_idx])} "
                f"expected={expected_start}."
            )
        cached_rows = self._prefix_cached_rows
        expected = self._prefix_cached_entries(ranges)
        if cached_rows.get(seq_id) != expected:
            raise RuntimeError(
                f"{self._prefix_label} mixed rollback cached entries are inconsistent: "
                f"seq_id={seq_id} expected={expected} got={cached_rows.get(seq_id)}."
            )

        self._clear_prefix_row_slots(row_idx, 0, expected_start)
        self.row_seq_lens[row_idx] = 0
        self.row_logical_lens[row_idx] = 0
        cached_rows.pop(seq_id, None)
        if not row_preexisted:
            owner = self.seq_id_to_row.pop(seq_id, None)
            if owner != row_idx:
                raise RuntimeError(
                    f"{self._prefix_label} mixed rollback row ownership changed unexpectedly: "
                    f"seq_id={seq_id} expected_row={row_idx} owner={owner}."
                )
            self.free_rows.appendleft(row_idx)

    def build_prefix_kv_payload(self, seq: Sequence, block_start: int, block_end: int) -> PrefixBlockPayload:
        block_start = int(block_start)
        block_end = int(block_end)
        if block_end <= block_start:
            raise ValueError(f"Invalid {self._prefix_label} KV payload range: {block_start}:{block_end}.")
        self._validate_mixed_prefix_range(block_start, block_end)
        row_idx = self.seq_id_to_row.get(int(seq.seq_id))
        if row_idx is None:
            raise RuntimeError(f"Cannot build {self._prefix_label} KV payload for unknown seq_id={seq.seq_id}.")
        row_len = int(self.row_seq_lens[row_idx])
        if block_end > row_len:
            raise RuntimeError(
                f"Cannot build {self._prefix_label} KV payload beyond materialized row length: "
                f"seq_id={seq.seq_id} block={block_start}:{block_end} row_len={row_len}."
            )
        slots = self.buffer_req_to_token_slots[row_idx, block_start:block_end].detach().to(
            dtype=torch.int32, copy=True,
        )
        return self._build_mixed_prefix_payload(slots, block_start, block_end)

    def _validate_mixed_prefix_range(self, start: int, end: int) -> None:
        pass

    def allocate_prefix_kv_payloads_device(self, payloads: list[object]) -> None:
        if not payloads:
            return
        counts = []
        for payload in payloads:
            self._validate_mixed_prefix_payload_type(payload)
            if isinstance(payload.token_slots, torch.Tensor):
                raise RuntimeError(f"{self._prefix_label} mixed KV payload is already device-resident.")
            tokens = int(payload.block_end) - int(payload.block_start)
            counts.append(self._mixed_prefix_allocation_count(tokens))
        units = self._take_prefix_device_units(sum(counts))
        self._stage_mixed_prefix_payloads(payloads, counts, units)

    def free_prefix_kv_payload_device(self, payload: object) -> None:
        self._validate_mixed_prefix_payload_type(payload)
        if not isinstance(payload.token_slots, torch.Tensor):
            raise RuntimeError(f"{self._prefix_label} mixed KV payload has no device slots to demote.")
        self._release_mixed_prefix_payload_device(payload)

    def _prefix_resident_weight_for_ids(
        self, block_ids: frozenset[bytes], *, view: str = "explicit"
    ) -> int:
        index = self.prefix_cache
        if index is None:
            return 0
        # Reference/priority changes alter membership, not physical weights.
        # Reuse unchanged weights and adjust only the changed part of each view.
        # Deletions invalidate only their own weights, including when an ID is
        # reinserted with different storage before the next capacity query.
        removed_ids = index.take_removed_block_ids_for_capacity()
        cache = self._prefix_resident_weight_cache
        if (cache is None or cache.index is not index
                or cache.payload_epoch != index.resident_payload_epoch
                or removed_ids is None):
            cache = _ResidentWeightCache(index)
            self._prefix_resident_weight_cache = cache
        elif removed_ids:
            cache.discard(removed_ids)
        totals, weights = cache.totals, cache.weights
        previous_ids, total = totals.get(view, (frozenset(), 0))
        if previous_ids is block_ids:
            return total
        for block_id in previous_ids - block_ids:
            total -= weights[block_id]
        missing_block = False
        for block_id in block_ids - previous_ids:
            weight = weights.get(block_id)
            if weight is None:
                block = index.get_block(block_id)
                if block is None:
                    # An older immutable view can outlive a deletion. Do not
                    # memoize absence across a later insertion of the same ID.
                    missing_block = True
                    continue
                weight = (
                    self._prefix_block_capacity_weight(block)
                    if block.residency.device_present else 0
                )
                weights[block_id] = weight
            total += weight
        if missing_block:
            totals.pop(view, None)
        else:
            totals[view] = (block_ids, int(total))
        return int(total)

    def _init_prefix_cache_runtime(self) -> None:
        self._prefix_resident_weight_cache: _ResidentWeightCache | None = None
        self.seq_id_to_materialized_blocks: dict[int, dict[bytes, PrefixCacheBlock]] = {}
        self.prefix_runtime_states: dict[int, PrefixRuntimeState] = {}
        self.pending_prefix_blocks: dict[int, list[PendingPrefixBlock]] = {}
        self.prefix_lookup_cache = PrefixLookupCache()
        self.prefix_hit_capacity_cache: WeakKeyDictionary[
            Sequence, PrefixHitCapacityCacheEntry
        ] = WeakKeyDictionary()

    def _prefix_cache_materialization_subject(self) -> str:
        return "Prefix materialization"

    def _prefix_cache_negative_refcount_message(self) -> str:
        return "Prefix cache block ref_count became negative."

    def _prefix_cache_materialize_profile_name(self) -> str:
        return "prefix_cache_materialize"

    def _make_prefix_block_payload(self, slots: torch.Tensor) -> PrefixBlockPayload:
        raise NotImplementedError

    def _make_prefix_block_payloads(self, slots: list[torch.Tensor]) -> list[PrefixBlockPayload]:
        return [self._make_prefix_block_payload(block_slots) for block_slots in slots]

    def _mark_materialized_prefix_block(self, seq: Sequence, block: PrefixCacheBlock) -> None:
        raise NotImplementedError

    def _reset_prefix_cache_allocator_after_clear(self) -> None:
        raise NotImplementedError

    def _release_prefix_blocks(self, blocks: list[PrefixCacheBlock]) -> None:
        prefix_cache = self.prefix_cache
        if prefix_cache is None and blocks:
            raise RuntimeError("Cannot release prefix blocks without a prefix cache.")
        for block in blocks:
            assert prefix_cache is not None
            prefix_cache.release_block_ref(
                block,
                negative_error=self._prefix_cache_negative_refcount_message(),
            )

    def _hold_materialized_prefix_block_ref(
        self,
        seq: Sequence,
        block: PrefixCacheBlock,
    ) -> None:
        attached = self.seq_id_to_prefix_blocks.get(seq.seq_id, ())
        logical_idx = int(block.logical_block_idx)
        # Attached chains start at the radix root, so logical index is also the
        # list position. Materialized/replayed refs have a single keyed owner.
        if (0 <= logical_idx < len(attached)
                and attached[logical_idx].stable_block_id == block.stable_block_id):
            return
        materialized = self.seq_id_to_materialized_blocks.get(seq.seq_id)
        if materialized is not None and block.stable_block_id in materialized:
            return
        prefix_cache = self.prefix_cache
        if prefix_cache is None:
            raise RuntimeError("Cannot hold a prefix block without a prefix cache.")
        prefix_cache.acquire_block_ref(block)
        if materialized is None:
            materialized = self.seq_id_to_materialized_blocks[seq.seq_id] = {}
        materialized[block.stable_block_id] = block

    def _lookup_prefix_cache_hit(
        self,
        seq: Sequence,
        usable_tokens: int,
    ) -> tuple[int, bytes | None, int]:
        prefix_cache = self.prefix_cache
        if prefix_cache is None:
            return 0, None, 0
        return lookup_prefix_cache_hit(
            prefix_cache,
            self.prefix_lookup_cache,
            seq,
            usable_tokens,
        )

    def _prefix_hit_capacity_counts(
        self,
        seq: Sequence,
    ) -> tuple[int, int]:
        prefix_cache = getattr(self, "prefix_cache", None)
        if prefix_cache is None:
            return 0, 0
        cache = getattr(self, "prefix_hit_capacity_cache", None)
        if cache is None:
            cache = WeakKeyDictionary()
            self.prefix_hit_capacity_cache = cache
        return prefix_hit_capacity_counts(
            prefix_cache,
            cache,
            seq,
            offload_enabled=self._prefix_offload_enabled(),
        )

    def reset_prefix_cache(self) -> None:
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        referenced = [
            block.stable_block_id.hex()
            for block in self.prefix_cache.blocks.values()
            if int(block.ref_count) != 0
        ]
        if referenced:
            raise RuntimeError(
                "Cannot reset prefix cache while blocks are still referenced: "
                f"referenced_blocks={referenced[:5]}."
            )

        self._reset_prefix_cache_allocator_after_clear()
        self.prefix_cache = RadixPrefixIndex(
            block_size=int(self.prefix_cache_block_size),
            fingerprint=build_prefix_cache_fingerprint(self.config, int(self.prefix_cache_block_size)),
            max_blocks=getattr(self.config, "prefix_cache_max_blocks", None),
        )
        self._on_prefix_cache_reset()
        self.seq_id_to_prefix_blocks.clear()
        self._init_prefix_cache_runtime()

    def _record_prefix_materialization(
        self,
        seq: Sequence,
        token_ids: list[int],
        slots: torch.Tensor,
    ) -> None:
        if getattr(seq, "multimodal_digest", None) is not None:
            return
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        if len(token_ids) != int(slots.numel()):
            raise RuntimeError(
                f"{self._prefix_cache_materialization_subject()} token/slot mismatch: "
                f"seq_id={seq.seq_id} tokens={len(token_ids)} slots={int(slots.numel())}."
            )
        if not token_ids:
            return

        # Freeze allocator/row aliases once, on their producer stream. Async
        # collection owns this snapshot and must not clone it a second time.
        slots = slots.detach().to(dtype=torch.int32, copy=True).reshape(-1)
        token_ids = [int(token_id) for token_id in token_ids]
        pending = getattr(self, "_async_prefix_records", None)
        if pending is not None:
            pending.append((seq, token_ids, slots))
            return
        self._record_frozen_prefix_materialization(seq, token_ids, slots)

    def _record_frozen_prefix_materialization(
        self, seq: Sequence, token_ids: list[int], slots: torch.Tensor,
    ) -> None:
        """Consume an owned slot snapshot after its execution result retires."""

        state = self.prefix_runtime_states.get(seq.seq_id)
        if state is None:
            hit_blocks = int(getattr(seq, "prefix_cache_hit_block_count", 0) or 0)
            parent_block_id = getattr(seq, "prefix_cache_hit_last_block_id", None)
            state = PrefixRuntimeState(
                parent_block_id=parent_block_id,
                next_logical_block_idx=hit_blocks,
                pending_tokens=[],
                pending_slots=[],
            )
            self.prefix_runtime_states[seq.seq_id] = state

        pending_blocks = self.pending_prefix_blocks.setdefault(seq.seq_id, [])
        block_size = int(self.prefix_cache_block_size)
        completed: list[tuple[list[int], torch.Tensor]] = []

        offset = 0
        if state.pending_tokens:
            need = block_size - len(state.pending_tokens)
            take = min(need, len(token_ids))
            state.pending_tokens.extend(token_ids[:take])
            state.pending_slots.append(slots[:take])
            offset = take
            if len(state.pending_tokens) == block_size:
                block_tokens = list(state.pending_tokens)
                block_slots = torch.cat(state.pending_slots, dim=0)
                completed.append((block_tokens, block_slots))
                state.pending_tokens = []
                state.pending_slots = []
            else:
                return

        full_tokens = ((len(token_ids) - offset) // block_size) * block_size
        end_full = offset + full_tokens
        for block_start in range(offset, end_full, block_size):
            block_end = block_start + block_size
            completed.append((
                token_ids[block_start:block_end],
                slots[block_start:block_end],
            ))

        if end_full < len(token_ids):
            state.pending_tokens = token_ids[end_full:]
            state.pending_slots = [slots[end_full:]]
        else:
            state.pending_tokens = []
            state.pending_slots = []

        if completed:
            payloads = self._make_prefix_block_payloads([block_slots for _, block_slots in completed])
            for (block_tokens, block_slots), payload in zip(completed, payloads, strict=True):
                stable_block_id = self.prefix_cache.stable_block_id(block_tokens, state.parent_block_id)
                pending_blocks.append(PendingPrefixBlock(
                    stable_block_id=stable_block_id, parent_block_id=state.parent_block_id,
                    logical_block_idx=state.next_logical_block_idx, payload=payload,
                    slots=block_slots, token_ids=block_tokens,
                ))
                state.parent_block_id = stable_block_id
                state.next_logical_block_idx += 1

    def _release_prefix_request(self, seq_id: int) -> list[PrefixCacheBlock]:
        """Release request refs/records, not physical storage or host transfers.

        Works before any row was allocated and after a previous cleanup. Callers
        return their private slots/pages first; the index keeps shared payloads.
        """
        released = self.seq_id_to_prefix_blocks.pop(seq_id, [])
        borrowed = getattr(self, "_prefix_prune_borrowed_seqs", None)
        if borrowed is not None and seq_id in borrowed:
            borrowed.remove(seq_id)
            released = []
        materialized = self.seq_id_to_materialized_blocks.pop(seq_id, None)
        if materialized:
            released.extend(materialized.values())
        self._release_prefix_blocks(released)
        self.prefix_runtime_states.pop(seq_id, None)
        self.pending_prefix_blocks.pop(seq_id, None)
        self.prefix_lookup_cache.discard(seq_id)
        return released

    def on_forward_end(self, seqs: list[Sequence], is_prefill: bool):
        self.publish_pending_prefix_blocks(seqs)

    def publish_pending_prefix_blocks(self, seqs: list[Sequence]):
        """Publish materialized records without repeating forward lifecycle hooks."""
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        with profiler.record(self._prefix_cache_materialize_profile_name()):
            for seq in seqs:
                pending_blocks = self.pending_prefix_blocks.pop(seq.seq_id, [])
                if not pending_blocks:
                    continue
                materialized = self.seq_id_to_materialized_blocks.setdefault(seq.seq_id, {})
                protected: list[PrefixCacheBlock] = []
                protected_block_ids = {
                    block_id
                    for pending in pending_blocks
                    for block_id in (pending.parent_block_id, pending.stable_block_id)
                    if block_id is not None and self.prefix_cache.has_block(block_id)
                }
                for block_id in protected_block_ids:
                    block = self.prefix_cache.get_block(block_id)
                    if block is None:
                        continue
                    self.prefix_cache.acquire_block_ref(block)
                    protected.append(block)
                try:
                    # Reserve once for distinct new blocks, with existing
                    # parents/duplicates pinned above. Per-block eviction
                    # rebuilds the candidate heap for every published block.
                    new_block_ids = {
                        pending.stable_block_id for pending in pending_blocks
                        if not self.prefix_cache.has_block(pending.stable_block_id)
                    }
                    if new_block_ids:
                        self._evict_prefix_cache_for_insert(len(new_block_ids))
                    for pending in pending_blocks:
                        block = PrefixCacheBlock(
                            stable_block_id=pending.stable_block_id,
                            parent_block_id=pending.parent_block_id,
                            block_size=int(self.prefix_cache_block_size),
                            logical_block_idx=pending.logical_block_idx,
                            payload=pending.payload,
                            token_ids=tuple(pending.token_ids),
                            ref_count=1,
                        )
                        inserted = self.prefix_cache.insert_block(block)
                        if inserted is not block:
                            # Recompute replay deliberately rebuilds the exact
                            # token stream instead of attaching a prefix hit.
                            # Its blocks can therefore duplicate blocks left by
                            # the first pass. Keep those parents referenced for
                            # the replay lifetime, but do not mark the replay's
                            # newly allocated slots as prefix-owned: the
                            # existing block payload owns different slots.
                            self._hold_materialized_prefix_block_ref(seq, inserted)
                            continue
                        materialized[inserted.stable_block_id] = inserted
                        self._mark_materialized_prefix_block(seq, inserted)
                finally:
                    self._release_prefix_blocks(protected)
