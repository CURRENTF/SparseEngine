from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from weakref import WeakKeyDictionary

import torch

from sparseengine.engine.prefix_cache import (
    PrefixBlockPayload,
    PrefixCacheBlock,
    RadixPrefixIndex,
    build_prefix_cache_fingerprint,
)
from sparseengine.engine.sequence import Sequence
from sparseengine.utils.profiler import profiler


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


class PrefixLookupCache:
    """Bounded request-ID memoization, including deserialized TP requests."""

    def __init__(self, max_entries: int = 128):
        if max_entries <= 0:
            raise ValueError("Prefix lookup cache capacity must be positive.")
        self.max_entries = max_entries
        self.entries: OrderedDict[int, PrefixLookupCacheEntry] = OrderedDict()

    def get(self, seq: Sequence) -> PrefixLookupCacheEntry | None:
        entry = self.entries.get(seq.seq_id)
        if entry is not None:
            self.entries.move_to_end(seq.seq_id)
        return entry

    def __setitem__(self, seq: Sequence, entry: PrefixLookupCacheEntry) -> None:
        self.entries[seq.seq_id] = entry
        self.entries.move_to_end(seq.seq_id)
        if len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)

    def discard(self, seq_id: int) -> None:
        self.entries.pop(seq_id, None)


@dataclass(frozen=True)
class PrefixHitCapacityCacheEntry:
    last_block_id: bytes
    hit_blocks: int
    remove_epoch: int
    capacity_epoch: int
    offload_enabled: bool
    chain: tuple[PrefixCacheBlock, ...]
    reclaimable_blocks: int
    promotion_blocks: int


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
    """Shared prefix-cache block materialization for cache managers."""

    def _init_prefix_cache_runtime(self) -> None:
        self.seq_id_to_materialized_blocks: dict[int, list[PrefixCacheBlock]] = {}
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

    def _mark_materialized_prefix_block(self, seq: Sequence, block: PrefixCacheBlock) -> None:
        raise NotImplementedError

    def _reset_prefix_cache_allocator_after_clear(self) -> None:
        raise NotImplementedError

    def _on_prefix_cache_reset(self) -> None:
        return None

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
        held = [
            *self.seq_id_to_prefix_blocks.get(seq.seq_id, []),
            *self.seq_id_to_materialized_blocks.get(seq.seq_id, []),
        ]
        if any(
            existing.stable_block_id == block.stable_block_id
            for existing in held
        ):
            return
        prefix_cache = self.prefix_cache
        if prefix_cache is None:
            raise RuntimeError("Cannot hold a prefix block without a prefix cache.")
        prefix_cache.acquire_block_ref(block)
        self.seq_id_to_materialized_blocks.setdefault(seq.seq_id, []).append(block)

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
        pending = getattr(self, "_async_prefix_records", None)
        if pending is not None:
            pending.append((seq, list(token_ids), slots.clone()))
            return
        if len(token_ids) != int(slots.numel()):
            raise RuntimeError(
                f"{self._prefix_cache_materialization_subject()} token/slot mismatch: "
                f"seq_id={seq.seq_id} tokens={len(token_ids)} slots={int(slots.numel())}."
            )
        if not token_ids:
            return

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
        token_ids = [int(token_id) for token_id in token_ids]
        slots = slots.detach().to(dtype=torch.int32).reshape(-1).clone()

        def add_block(block_tokens: list[int], block_slots: torch.Tensor) -> None:
            stable_block_id = self.prefix_cache.stable_block_id(block_tokens, state.parent_block_id)
            pending_blocks.append(
                PendingPrefixBlock(
                    stable_block_id=stable_block_id,
                    parent_block_id=state.parent_block_id,
                    logical_block_idx=state.next_logical_block_idx,
                    payload=self._make_prefix_block_payload(block_slots),
                    slots=block_slots,
                    token_ids=block_tokens,
                )
            )
            state.parent_block_id = stable_block_id
            state.next_logical_block_idx += 1

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
                add_block(block_tokens, block_slots)
                state.pending_tokens = []
                state.pending_slots = []
            else:
                return

        full_tokens = ((len(token_ids) - offset) // block_size) * block_size
        end_full = offset + full_tokens
        for block_start in range(offset, end_full, block_size):
            block_end = block_start + block_size
            add_block(
                token_ids[block_start:block_end],
                slots[block_start:block_end],
            )

        if end_full < len(token_ids):
            state.pending_tokens = token_ids[end_full:]
            state.pending_slots = [slots[end_full:]]
        else:
            state.pending_tokens = []
            state.pending_slots = []

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
                materialized = self.seq_id_to_materialized_blocks.setdefault(seq.seq_id, [])
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
                    for pending in pending_blocks:
                        if not self.prefix_cache.has_block(pending.stable_block_id):
                            self._evict_prefix_cache_for_insert(1)
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
                        materialized.append(inserted)
                        self._mark_materialized_prefix_block(seq, inserted)
                finally:
                    self._release_prefix_blocks(protected)
