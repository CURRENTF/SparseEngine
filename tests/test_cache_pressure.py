"""Capacity-pressure ordering, shared ownership, and failure cleanup."""
import random
from types import SimpleNamespace

import pytest

from cache_contracts.cases import RADIX_CASES, make_radix
from cache_contracts.ownership import RadixHarness
from sparseengine.engine.prefix_cache import (
    PrefixBlockResidency, PrefixCacheBlock, PrefixTransferKind, RadixPrefixIndex,
)
from test_radix_cache_contract import materialize, request


@pytest.mark.parametrize('shape', ['wide', 'deep', 'branched'])
def test_weighted_demotion_matches_leaf_oracle_across_partial_batches(shape):
    """Parent readiness must follow resident children, including pinned siblings."""
    rng = random.Random(71)
    index = RadixPrefixIndex(block_size=1, fingerprint=b'demotion-oracle')
    blocks = []
    parents = {}
    for n in range(48):
        parent = None if n == 0 else (0 if shape == 'wide' else
                                      n - 1 if shape == 'deep' else rng.randrange(n))
        parents[n] = parent
        block = PrefixCacheBlock(
            stable_block_id=n.to_bytes(2, 'big'),
            parent_block_id=None if parent is None else blocks[parent].stable_block_id,
            block_size=1, logical_block_idx=0 if parent is None else blocks[parent].logical_block_idx + 1,
            payload=SimpleNamespace(weight=n % 4), token_ids=(n,),
            ref_count=int(n in (17, 31)), eviction_priority=-1 if n == 23 else n % 3,
            residency=PrefixBlockResidency(host_present=True,
                transfer=PrefixTransferKind.H2D if n == 37 else None),
        )
        index.insert_block(block)
        blocks.append(block)
    # Independent virtual-leaf oracle; its full child scans are intentional.
    resident = set(range(len(blocks)))
    for needed in (0, 1, 13, 1000):
        expected = []
        total = 0
        while total < needed:
            eligible = [n for n in resident
                        if n not in (17, 23, 31, 37)
                        and not any(parents[child] == n for child in resident)]
            if not eligible:
                break
            chosen = min(eligible, key=lambda n: (-(n % 3), blocks[n].last_access,
                                                  blocks[n].stable_block_id))
            expected.append(blocks[chosen])
            resident.remove(chosen)
            total += chosen % 4
        actual = index.demote_device_until_weight(needed, lambda block: block.payload.weight)
        assert actual == expected
        assert {n for n, block in enumerate(blocks) if block.residency.device_present} == resident
        assert len(index) == len(blocks)
        assert all(block.residency.host_present for block in blocks)


@pytest.mark.parametrize('method', RADIX_CASES, ids=lambda m: m or 'vanilla')
def test_batch_publication_under_pressure_preserves_replayed_parent_and_ownership(method):
    """A batch must charge only new IDs and keep the replayed parent pinned."""
    m = make_radix(method)
    m.prefix_cache.max_blocks = 3
    h = RadixHarness(m, paged=method == 'quest')
    for tokens in ([1, 2], [11, 12], [21, 22]):
        seq = materialize(m, tokens)
        m.free_seq(seq.seq_id)
    parent_id = m.prefix_cache.stable_block_id([1, 2], None)
    parent_payload = m.prefix_cache.get_block(parent_id).payload
    seq = materialize(m, [1, 2, 3, 4, 5, 6])
    h.observe()
    blocks = list(m.prefix_cache.blocks.values())
    assert len(blocks) == 3
    assert m.prefix_cache.get_block(parent_id).payload is parent_payload
    assert [block.token_ids for block in blocks] == [(1, 2), (3, 4), (5, 6)]
    assert all(block.ref_count == 1 for block in blocks)
    m.free_seq(seq.seq_id)
    h.assert_released(seq.seq_id)
    m.reset_prefix_cache()
    assert len(h.observe().free) == h.observe().capacity


@pytest.mark.parametrize('method', RADIX_CASES, ids=lambda m: m or 'vanilla')
def test_batch_capacity_failure_releases_temporary_parent_protection(method):
    """Failed batch reservation must leave recompute storage request-owned."""
    m = make_radix(method)
    m.prefix_cache.max_blocks = 1
    parent_seq = materialize(m, [1, 2])
    m.free_seq(parent_seq.seq_id)
    seq = request([1, 2, 3, 4])
    slots = m._allocate(seq.seq_id, 4)
    m._record_prefix_materialization(seq, list(seq.token_ids), slots)
    with pytest.raises(RuntimeError, match='capacity exceeded'):
        m.publish_pending_prefix_blocks([seq])
    h = RadixHarness(m, paged=method == 'quest')
    h.observe()
    m.free_seq(seq.seq_id)
    h.assert_released(seq.seq_id)
    assert all(block.ref_count == 0 for block in m.prefix_cache.blocks.values())
    m.reset_prefix_cache()
    assert len(h.observe().free) == h.observe().capacity


def test_offload_plan_partitions_mixed_victims_without_reordering():
    """Backed and unbacked victims must each retain the driver's LRU order."""
    from sparseengine.engine.chain_cache import (
        ChainAdmissionPlan, ChainCacheCoordinator, ChainCacheIndex, ChainRecord, ChainState,
    )
    c = object.__new__(ChainCacheCoordinator)
    c.index = ChainCacheIndex()
    ordered = ('old-backed', 'old-unbacked', 'new-backed', 'new-unbacked')
    for n, name in enumerate(ordered):
        c.index.records[name] = ChainRecord(name, n, b'partition', ChainState.IDLE)
    c.offload = SimpleNamespace(snapshots={0: object(), 2: object()})
    plan = ChainAdmissionPlan('incoming', 9, 'created', 0, victim_chain_ids=ordered,
                              reserved_slots_by_layer=(5, 7), reserved_rows=1)
    actual = c._offload_plan(plan)
    assert actual.demote_chain_ids == ('old-backed', 'new-backed')
    assert actual.victim_chain_ids == ('old-unbacked', 'new-unbacked')
    assert actual.reserved_slots_by_layer == (5, 7) and actual.reserved_rows == 1
    assert plan.victim_chain_ids == ordered and plan.demote_chain_ids == ()


def test_idle_lru_iterator_preserves_ties_while_consumed_victims_are_removed():
    """Reclaim loops mutate the index between consecutive LRU selections."""
    from sparseengine.engine.chain_cache import ChainCacheIndex, ChainRecord, ChainState
    index = ChainCacheIndex()
    for name, age, rows, state in [
        ('z', 2, 1, ChainState.IDLE), ('b', 1, 1, ChainState.IDLE),
        ('a', 1, 1, ChainState.IDLE), ('host', 0, 0, ChainState.IDLE),
        ('writer', 0, 1, ChainState.ACTIVE),
    ]:
        record = ChainRecord(name, len(index.records), b'lru', state, resident_rows=rows, last_access=age)
        index.records[name] = record
    selected = []
    for record in index.idle_resident_lru():
        selected.append(record.chain_id)
        index.evict(record.chain_id)
    assert selected == ['a', 'b', 'z']
    assert set(index.records) == {'host', 'writer'}


@pytest.mark.parametrize('residency', ['device', 'host', 'dual'])
@pytest.mark.parametrize('blocked', [False, True])
def test_mixed_byte_pressure_respects_pins_zero_sizes_and_pending_bytes(residency, blocked):
    """Variable-size eviction must meet bytes, not a guessed block count."""
    from sparseengine.engine.prefix_cache_coordinator import PrefixCacheCoordinator, MixedPrefixBlockPayload
    c = object.__new__(PrefixCacheCoordinator)
    c.prefix_cache = RadixPrefixIndex(block_size=1, fingerprint=b'bytes')
    sizes = [0, 3, 5, 7, 11]
    c.max_recurrent_bytes = sum(sizes)
    c.pending_recurrent_bytes = 2
    freed = []
    c.cache_manager = SimpleNamespace(
        free_prefix_kv_payload=lambda payload: freed.append(payload),
        free_prefix_kv_payload_device=lambda payload: None,
    )
    c.recurrent_state_manager = SimpleNamespace(free_prefix_recurrent_payload=lambda payload: None)
    c.offload_controller = None if residency == 'device' else SimpleNamespace(
        poll=lambda: None, wait_oldest_d2h=lambda: False,
        free_host_payloads=lambda blocks: freed.extend(block.payload.kv_payload for block in blocks),
        free_device_recurrent=lambda block: None,
    )
    for n, size in enumerate(sizes):
        c.prefix_cache.insert_block(PrefixCacheBlock(
            n.to_bytes(2, 'big'), None, 1, 0,
            MixedPrefixBlockPayload(n, object(), 1, size, size), token_ids=(n,),
            ref_count=int(n == 2 or (blocked and n > 0)),
            residency=PrefixBlockResidency(device_present=residency != 'host',
                                           host_present=residency != 'device'),
        ))
    assert c._live_recurrent_bytes() == sum(sizes)
    success = c._evict_for_insert(1, incoming_recurrent_bytes=10)
    assert success is not blocked
    assert freed == ([0] if blocked else [0, 1, 3, 4])
    remaining = [n for n in range(len(sizes)) if n not in freed]
    assert c._live_recurrent_bytes() == sum(sizes[n] for n in remaining)
    assert set(c.prefix_cache.blocks) == {n.to_bytes(2, 'big') for n in remaining}
