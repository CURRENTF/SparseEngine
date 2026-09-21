"""Ownership and invalidation contracts across complete cache call paths."""
from types import SimpleNamespace

import pytest
import torch

from cache_contracts.cases import RADIX_CASES, make_radix
from cache_contracts.ownership import RadixHarness
from sparseengine.engine.async_scheduling.execution import AsyncExecution, DeviceResult
from sparseengine.platforms import device_runtime
from test_radix_cache_contract import materialize, request


@pytest.mark.parametrize('method', RADIX_CASES, ids=lambda m: m or 'vanilla')
def test_async_slot_snapshot_survives_allocator_reuse_and_discards_cancelled_records(method, monkeypatch):
    """Deferred publication owns frozen IDs even after the borrowed stack changes."""
    m = make_radix(method)
    live, cancelled = request([1, 2, 3, 4]), request([5, 6])
    expected = None
    records = []
    m._async_prefix_records = records
    for seq in (live, cancelled):
        slots = m._allocate(seq.seq_id, len(seq.token_ids))
        tokens = list(seq.token_ids)
        m._record_prefix_materialization(seq, tokens, slots)
        if seq is live:
            expected = slots.clone()
        # The allocator's borrowed ID range may be reused before collection;
        # the request row itself already contains the original IDs.
        slots.fill_(10000)
        tokens[:] = [99] * len(tokens)
    m._async_prefix_records = None
    assert len(m.prefix_cache) == 0
    a = object.__new__(AsyncExecution)
    a.runner = SimpleNamespace(cache_manager=m)
    a.completed = 0
    a.results = {1: DeviceResult([live, cancelled], True, object(), torch.tensor([7, 8]),
                                None, None, records, [])}
    monkeypatch.setattr(device_runtime, 'synchronize_event', lambda event: None)
    assert a.collect(1, discarded=(cancelled.seq_id,))[0] == [7, 8]
    blocks = list(m.prefix_cache.blocks.values())
    assert [block.token_ids for block in blocks] == [(1, 2), (3, 4)]
    torch.testing.assert_close(torch.cat([block.payload.token_slots for block in blocks]), expected)
    h = RadixHarness(m, paged=method == 'quest')
    h.observe()
    for seq in (live, cancelled):
        m.free_seq(seq.seq_id)
        h.assert_released(seq.seq_id)
    m.reset_prefix_cache()
    assert len(h.observe().free) == h.observe().capacity


def test_quest_batch_materialization_rejects_a_malformed_later_page_before_publication():
    """Batch validation must check every page, not only its first page ID."""
    m = make_radix('quest')
    seq = request([1, 2, 3, 4])
    slots = m._allocate(seq.seq_id, 4).clone()
    slots[-1] += 1
    with pytest.raises(RuntimeError, match='non-contiguous'):
        m._record_prefix_materialization(seq, list(seq.token_ids), slots)
    assert len(m.prefix_cache) == 0
    m.free_seq(seq.seq_id)
    h = RadixHarness(m, paged=True)
    assert len(h.observe().free) == h.observe().capacity


def test_quest_bulk_page_release_rejects_a_corrupt_id_without_returning_any_pages():
    """Validate the complete return batch before publishing allocator ownership."""
    m = make_radix('quest')
    seq = materialize(m, [1, 2, 3, 4])
    m.free_seq(seq.seq_id)
    blocks = m.prefix_cache.evict_until_freeable(2)
    before = m._num_free_pages
    saved = blocks[-1].payload.block_slot
    blocks[-1].payload.block_slot = blocks[0].payload.block_slot
    with pytest.raises(RuntimeError, match='mismatched'):
        m._free_prefix_cache_blocks(blocks)
    assert m._num_free_pages == before
    assert all(block.payload.token_slots is not None for block in blocks)
    blocks[-1].payload.block_slot = saved
    m._free_prefix_cache_blocks(blocks)
    h = RadixHarness(m, paged=True)
    assert len(h.observe().free) == h.observe().capacity


def test_weighted_hit_capacity_invalidates_after_pins_compaction_and_index_replacement():
    """A memoized slot cost must not reuse weights from a different physical view."""
    from sparseengine.engine.cache_manager.standard import StandardPrefixBlockPayload
    from sparseengine.engine.prefix_cache import PrefixCacheBlock, RadixPrefixIndex
    m = make_radix('')
    producer = materialize(m, [1, 2, 3, 4])
    m.free_seq(producer.seq_id)
    seq = request([1, 2, 3, 4, 5])
    m.refresh_prefix_cache_hit(seq)
    assert m.prompt_admission_cost(seq) == 5
    index = m.prefix_cache
    leaf = list(index.blocks.values())[-1]
    index.acquire_block_ref(leaf)
    assert m.prompt_admission_cost(seq) == 1
    index.release_block_ref(leaf)
    assert m.prompt_admission_cost(seq) == 5
    m.prefix_cache_prune([1, 2, 3, 4], range_start=0, range_end=4,
                         keep_indices=torch.tensor([0, 3]), policy='snapkv_global', prune_id='weights')
    assert m.prompt_admission_cost(seq) == 3
    # A replacement with the same IDs and epochs owns different payloads.
    replacement = RadixPrefixIndex(block_size=2, fingerprint=index.fingerprint)
    for block in index.blocks.values():
        replacement.insert_block(PrefixCacheBlock(
            block.stable_block_id, block.parent_block_id, 2, block.logical_block_idx,
            StandardPrefixBlockPayload(torch.empty(0, dtype=torch.int32), retained_offsets=()),
            token_ids=block.token_ids,
        ))
    replacement._capacity_epoch = index.capacity_epoch
    m.prefix_cache = replacement
    assert m.prompt_admission_cost(seq) == 1


def test_chain_routing_snapshots_follow_all_logical_transitions_and_remain_immutable():
    """Cached routing must refresh for state changes, tombstone expiry and reset."""
    from sparseengine.engine.chain_cache import ChainCacheIndex, ChainState
    index = ChainCacheIndex(max_tombstones=1)
    empty = index.routing_snapshot()
    for i, name in enumerate(('a', 'b')):
        plan = index.plan_admission(chain_id=name, seq_id=i, token_ids=[1], fingerprint=b'route')
        record = index.apply_admission(plan, fingerprint=b'route')
        active = index.routing_snapshot()
        assert active.match(name)['state'] == 'active'
        index.finish(name, token_ids=[1], processed_token_count=1, physical_slots_by_layer=(1,))
        idle = index.routing_snapshot()
        assert idle.match(name)['state'] == 'idle'
        plan = index.plan_admission(chain_id=name, seq_id=i, token_ids=[1, 2], fingerprint=b'route')
        index.apply_admission(plan, fingerprint=b'route')
        assert index.routing_snapshot().match(name)['state'] == 'active'
        # Runtime uses this transition if restoring a resumed chain fails.
        index._set_record_state(record, ChainState.IDLE)
        assert index.routing_snapshot().match(name)['state'] == 'idle'
        index.invalidate(name)
        assert index.routing_snapshot().match(name)['tombstone']
        assert active.match(name)['state'] == 'active' and idle.match(name)['state'] == 'idle'
    assert not index.routing_snapshot().match('a')['tombstone']
    index.reset()
    assert index.routing_snapshot() == empty


@pytest.mark.parametrize('method', ('', 'omnikv'), ids=('vanilla', 'omnikv'))
@pytest.mark.parametrize('pruned', (False, True), ids=('full', 'including-empty-block'))
def test_standard_bulk_return_preserves_slot_ownership_for_full_and_pruned_blocks(method, pruned):
    """Returning unequal and empty payloads must not lose or double-charge slots."""
    m = make_radix(method)
    seq = materialize(m, [1, 2, 3, 4, 5, 6])
    m.free_seq(seq.seq_id)
    if pruned:
        m.prefix_cache_prune([1, 2, 3, 4, 5, 6], range_start=0, range_end=6,
                             keep_indices=torch.tensor([0, 5]), policy='snapkv_global', prune_id='return')
    blocks = m.prefix_cache.evict_until_freeable(3)
    m._free_prefix_cache_blocks(blocks)
    h = RadixHarness(m, paged=False)
    assert len(h.observe().free) == h.observe().capacity
    assert all(block.payload.token_slots is None for block in blocks)


@pytest.mark.parametrize('method', ('', 'omnikv'), ids=('vanilla', 'omnikv'))
def test_standard_bulk_return_validates_later_payload_and_capacity_before_mutation(method):
    """A failed return must leave every source payload and allocator count intact."""
    m = make_radix(method)
    seq = materialize(m, [1, 2, 3, 4])
    m.free_seq(seq.seq_id)
    blocks = m.prefix_cache.evict_until_freeable(2)
    before = m._num_free_slots
    saved = blocks[-1].payload.token_slots
    blocks[-1].payload.token_slots = saved[:1]
    with pytest.raises(RuntimeError, match='invalid device slots'):
        m._free_prefix_cache_blocks(blocks)
    assert m._num_free_slots == before
    assert all(block.payload.token_slots is not None for block in blocks)
    blocks[-1].payload.token_slots = saved
    m._num_free_slots = m.free_slots_stack.numel()
    with pytest.raises(RuntimeError, match='overflow'):
        m._free_prefix_cache_blocks(blocks)
    assert all(block.payload.token_slots is not None for block in blocks)
    m._num_free_slots = before
    m._free_prefix_cache_blocks(blocks)
    h = RadixHarness(m, paged=False)
    assert len(h.observe().free) == h.observe().capacity


@pytest.mark.parametrize('kept', ([None, None, None], [(6, 1), None, ()], [(), (), ()], []))
def test_host_transfer_indices_preserve_block_and_retained_offset_order(kept):
    """Packing must preserve nonadjacent host blocks and empty/mixed payloads."""
    from sparseengine.engine.cache_manager.prefix_offload import PinnedPrefixBlockPool, PrefixOffloadController
    from sparseengine.engine.cache_manager.standard import StandardPrefixBlockPayload
    pool = PinnedPrefixBlockPool(capacity_blocks=5, num_layers=2, block_size=7)
    c = object.__new__(PrefixOffloadController)
    c.host_pool, c.block_size, c.device = pool, 7, torch.device('cpu')
    ids = [4, 0, 2][:len(kept)]
    blocks = [SimpleNamespace(payload=SimpleNamespace(kv_payload=StandardPrefixBlockPayload(
        None, retained_offsets=offsets))) for offsets in kept]
    actual = c._host_token_indices(blocks, ids)
    storage = torch.arange(35).reshape(5, 7)
    expected_parts = [storage[bid] if offsets is None else storage[bid, list(offsets)]
                      for bid, offsets in zip(ids, kept)]
    expected = torch.cat(expected_parts) if expected_parts else torch.empty(0, dtype=torch.long)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous()
    with pytest.raises(ValueError, match='equal length'):
        pool.retained_token_indices([0], [], c.device)


def test_host_transfer_indices_reject_invalid_pruned_offsets():
    from sparseengine.engine.cache_manager.prefix_offload import PinnedPrefixBlockPool, PrefixOffloadController
    from sparseengine.engine.cache_manager.standard import StandardPrefixBlockPayload
    c = object.__new__(PrefixOffloadController)
    c.host_pool = PinnedPrefixBlockPool(capacity_blocks=2, num_layers=1, block_size=4)
    c.block_size, c.device = 4, torch.device('cpu')
    for offsets in ((-1,), (4,)):
        blocks = [SimpleNamespace(payload=StandardPrefixBlockPayload(None)),
                  SimpleNamespace(payload=StandardPrefixBlockPayload(None, retained_offsets=offsets))]
        with pytest.raises(RuntimeError, match='invalid retained token offsets'):
            c._host_token_indices(blocks, [0, 1])


@pytest.mark.parametrize('method', RADIX_CASES, ids=lambda m: m or 'vanilla')
def test_live_waiting_requests_keep_lookup_memos_without_retaining_cancelled_owners(method):
    """A queue wider than retired-memo capacity must not rehash every prompt."""
    import gc
    from weakref import ref
    from sparseengine.engine.cache_manager.prefix_cache_mixin import PrefixLookupCache
    m = make_radix(method)
    m.prefix_lookup_cache = PrefixLookupCache(max_entries=2)
    producer = materialize(m, [1, 2, 3, 4, 5, 6])
    m.free_seq(producer.seq_id)
    seqs = [request([1, 2, 3, 4, 5, 6, 10 + i]) for i in range(7)]
    for seq in seqs:
        m.refresh_prefix_cache_hit(seq)
    generated = m.prefix_cache.block_id_generation_requests
    for _ in range(3):
        for seq in seqs:
            m.refresh_prefix_cache_hit(seq)
            assert seq.prefix_cache_hit_len == 6
    assert m.prefix_cache.block_id_generation_requests == generated
    removed = m.prefix_cache.safe_delete_subtree([1, 2, 3, 4, 5, 6]).deleted_blocks
    m._free_prefix_cache_blocks(removed)
    generated = m.prefix_cache.block_id_generation_requests
    for seq in seqs:
        m.refresh_prefix_cache_hit(seq)
        assert seq.prefix_cache_hit_len == 4
    assert m.prefix_cache.block_id_generation_requests == generated
    owners = [ref(seq) for seq in seqs]
    del seq
    seqs.clear()
    gc.collect()
    assert all(owner() is None for owner in owners)
    assert len(m.prefix_lookup_cache.entries) <= m.prefix_lookup_cache.max_entries


def test_lookup_memo_tracks_deserialized_owner_replacement_and_explicit_release():
    import gc
    import pickle
    from weakref import ref
    from sparseengine.engine.cache_manager.prefix_cache_mixin import PrefixLookupCache
    m = make_radix('')
    m.prefix_lookup_cache = PrefixLookupCache(max_entries=1)
    original = request([1, 2, 3])
    m.refresh_prefix_cache_hit(original)
    restored = pickle.loads(pickle.dumps(original))
    m.refresh_prefix_cache_hit(restored)
    original_ref = ref(original)
    del original
    gc.collect()
    assert original_ref() is None
    # Evict the bounded ID memo while the restored request remains live.
    m.refresh_prefix_cache_hit(request([7, 8, 9]))
    generated = m.prefix_cache.block_id_generation_requests
    m.refresh_prefix_cache_hit(restored)
    assert m.prefix_cache.block_id_generation_requests == generated
    m.prefix_lookup_cache.discard(restored.seq_id)
    assert m.prefix_lookup_cache.get(restored) is None
    restored._prompt_token_ids = (7, 8, 9)
    m.refresh_prefix_cache_hit(restored)
    assert m.prefix_cache.block_id_generation_requests == generated + 1


def test_batched_prefix_hash_packing_preserves_signed_little_endian_wire_ids():
    """Faster packing must preserve cache IDs shared across requests and TP ranks."""
    import hashlib
    import struct
    from sparseengine.engine.prefix_cache import RadixPrefixIndex
    values = [-2**63, -1, 0, 2**63 - 1]
    index = RadixPrefixIndex(block_size=len(values), fingerprint=b'wire-contract')
    parent = None
    for tokens in (values, list(reversed(values))):
        wire = b'wire-contract' + (b'\x00' if parent is None else b'\x01' + parent)
        wire += b''.join(value.to_bytes(8, 'little', signed=True) for value in tokens)
        expected = hashlib.sha256(wire).digest()
        assert index.stable_block_id(tokens, parent) == expected
        parent = expected
    for invalid in (-2**63 - 1, 2**63):
        with pytest.raises(struct.error):
            index.stable_block_id([0, 1, 2, invalid], parent)
