"""Shared-block protocol: exercise actual managers and independent identities."""
from __future__ import annotations

from collections import deque
import pytest
from cache_contracts.cases import RADIX_CASES, make_radix
from cache_contracts.ownership import RadixHarness, ContractViolation


def request(tokens):
    from sparseengine.engine.sequence import Sequence
    from sparseengine.sampling_params import SamplingParams
    return Sequence(tokens, SamplingParams(max_tokens=4))


def materialize(m, tokens=(1, 2, 3)):
    seq = request(list(tokens))
    slots = m._allocate(seq.seq_id, len(tokens))
    m._record_prefix_materialization(seq, list(tokens), slots)
    m.publish_pending_prefix_blocks([seq])
    return seq


def reader(m):
    seq = request([1, 2, 3, 4])
    m.refresh_prefix_cache_hit(seq)
    assert seq.prefix_cache_hit_len == 2
    return seq


@pytest.mark.parametrize('method', RADIX_CASES, ids=lambda m: m or 'vanilla')
def test_shared_prefix_two_readers_and_idempotent_cleanup(method):
    from sparseengine.engine.cache_manager.base import CacheManager
    from sparseengine.engine.cache_manager.radix_contract import validate_radix_manager
    m = make_radix(method)
    validate_radix_manager(m, CacheManager, mixed=False)
    h = RadixHarness(m, paged=method == 'quest')
    original = h.observe()
    producer = materialize(m)
    h.observe()
    m.free_seq(producer.seq_id)
    h.assert_released(producer.seq_id)
    a, b = reader(m), reader(m)
    before_lookup = h.observe()
    # Lookup is a hint. No refs or rows have been acquired yet.
    assert all(block.ref_count == 0 for block in m.prefix_cache.blocks.values())
    m._attach_prefix_cache_if_needed(a)
    h.observe()
    m._attach_prefix_cache_if_needed(a)
    h.observe()
    m._attach_prefix_cache_if_needed(b)
    h.observe()
    assert len(h.observe().free) == len(before_lookup.free)
    assert all(block.ref_count == 2 for block in m.prefix_cache.blocks.values())
    m.free_seq(a.seq_id)
    m.free_seq(a.seq_id)
    h.assert_released(a.seq_id)
    assert all(block.ref_count == 1 for block in m.prefix_cache.blocks.values())
    m.free_seq(b.seq_id)
    h.assert_released(b.seq_id)
    m.reset_prefix_cache()
    assert set(h.observe().free) == set(original.free)


@pytest.mark.parametrize('method', RADIX_CASES, ids=lambda m: m or 'vanilla')
def test_duplicate_recompute_slots_remain_private(method):
    m = make_radix(method)
    h = RadixHarness(m, paged=method == 'quest')
    first = materialize(m)
    m.free_seq(first.seq_id)
    original_payloads = {bid: block.payload for bid, block in m.prefix_cache.blocks.items()}
    # Deliberately recompute instead of attaching: same tokens, distinct storage.
    replay = materialize(m)
    h.observe()
    assert all(m.prefix_cache.blocks[bid].payload is payload
               for bid, payload in original_payloads.items())
    m.free_seq(replay.seq_id)
    h.assert_released(replay.seq_id)
    m.reset_prefix_cache()
    assert len(h.observe().free) == h.observe().capacity


@pytest.mark.parametrize('method', RADIX_CASES, ids=lambda m: m or 'vanilla')
@pytest.mark.parametrize('size', [-1, 17])
def test_invalid_allocation_is_non_mutating(method, size):
    m = make_radix(method)
    h = RadixHarness(m, paged=method == 'quest')
    before = h.observe()
    with pytest.raises((ValueError, RuntimeError)):
        m._allocate(77, size)
    assert h.observe() == before


@pytest.mark.parametrize('method', RADIX_CASES, ids=lambda m: m or 'vanilla')
def test_row_exhaustion_does_not_pin_speculative_prefix(method):
    m = make_radix(method)
    h = RadixHarness(m, paged=method == 'quest')
    producer = materialize(m)
    m.free_seq(producer.seq_id)
    a, b, blocked = reader(m), reader(m), reader(m)
    m._attach_prefix_cache_if_needed(a)
    m._attach_prefix_cache_if_needed(b)
    before = h.observe()
    with pytest.raises(RuntimeError, match='rows'):
        m._attach_prefix_cache_if_needed(blocked)
    assert h.observe() == before
    m.free_seq(blocked.seq_id)
    h.assert_released(blocked.seq_id)
    m.free_seq(a.seq_id)
    m.free_seq(b.seq_id)
    h.observe()


@pytest.mark.parametrize('method', RADIX_CASES, ids=lambda m: m or 'vanilla')
@pytest.mark.parametrize('boundary', ['claim_row', 'publish_row'])
def test_attach_failure_releases_aliases_but_not_index_storage(method, boundary, monkeypatch):
    m = make_radix(method)
    producer = materialize(m)
    m.free_seq(producer.seq_id)
    seq = reader(m)
    h = RadixHarness(m, paged=method == 'quest')
    before = h.observe()
    def fail(*args, **kwargs):
        raise RuntimeError('injected attach failure')
    if boundary == 'claim_row':
        original = m._get_free_row
        def claim_then_fail(sid):
            original(sid)
            fail()
        target, name, replacement = m, '_get_free_row', claim_then_fail
    else:
        target, name, replacement = m.prefix_cache, 'touch_chain', fail
    with monkeypatch.context() as patch:
        patch.setattr(target, name, replacement)
        with pytest.raises(RuntimeError, match='injected'):
            m._attach_prefix_cache_if_needed(seq)
    assert h.observe() == before
    h.assert_released(seq.seq_id)
    # The failed attach must not poison a retry with stale ownership metadata.
    m._attach_prefix_cache_if_needed(seq)
    h.observe()
    m.free_seq(seq.seq_id)
    h.assert_released(seq.seq_id)


@pytest.mark.parametrize('method', RADIX_CASES, ids=lambda m: m or 'vanilla')
def test_post_detach_offload_failure_cannot_strand_request_row(method, monkeypatch):
    m = make_radix(method)
    producer = materialize(m)
    def fail(*args):
        raise RuntimeError('injected host-pressure failure')
    with monkeypatch.context() as patch:
        patch.setattr(m, '_schedule_write_through_prefix_blocks', fail)
        with pytest.raises(RuntimeError, match='injected'):
            m.free_seq(producer.seq_id)
    h = RadixHarness(m, paged=method == 'quest')
    h.assert_released(producer.seq_id)
    m.free_seq(producer.seq_id)
    h.assert_released(producer.seq_id)


@pytest.mark.parametrize('method', RADIX_CASES, ids=lambda m: m or 'vanilla')
def test_partial_publication_keeps_only_committed_block_ownership(method, monkeypatch):
    m = make_radix(method)
    seq = request([1, 2, 3, 4, 5])
    slots = m._allocate(seq.seq_id, 5)
    m._record_prefix_materialization(seq, list(seq.token_ids), slots)
    original = m.prefix_cache.insert_block
    calls = 0
    def insert(block):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('injected second-block publication failure')
        return original(block)
    with monkeypatch.context() as patch:
        patch.setattr(m.prefix_cache, 'insert_block', insert)
        with pytest.raises(RuntimeError, match='injected'):
            m.publish_pending_prefix_blocks([seq])
    h = RadixHarness(m, paged=method == 'quest')
    h.observe()
    m.free_seq(seq.seq_id)
    h.assert_released(seq.seq_id)
    assert len(m.prefix_cache.blocks) == 1
    m.reset_prefix_cache()
    assert len(h.observe().free) == h.observe().capacity


@pytest.mark.parametrize('method', RADIX_CASES, ids=lambda m: m or 'vanilla')
def test_physical_oracle_detects_mutated_refcount(method):
    m = make_radix(method)
    seq = materialize(m)
    block = next(iter(m.prefix_cache.blocks.values()))
    block.ref_count += 1
    with pytest.raises(ContractViolation, match='reference count'):
        RadixHarness(m, paged=method == 'quest').observe()
    block.ref_count -= 1
    m.free_seq(seq.seq_id)


def host_chain_controller(monkeypatch, retained_counts, *, on_host=True):
    """Real radix residency transitions with only the device transport stubbed."""
    from contextlib import nullcontext
    import torch
    from sparseengine.engine.cache_manager.prefix_offload import (
        PinnedPrefixBlockPool, PrefixOffloadController,
    )
    from sparseengine.engine.cache_manager.standard import StandardPrefixBlockPayload
    from sparseengine.engine.prefix_cache import PrefixCacheBlock, RadixPrefixIndex
    from sparseengine.platforms import device_runtime

    index = RadixPrefixIndex(block_size=2, fingerprint=b'pruned-h2d')
    blocks = []
    parent_id = None
    for i, count in enumerate(retained_counts):
        tokens = (2 * i, 2 * i + 1)
        block_id = index.stable_block_id(list(tokens), parent_id)
        block = PrefixCacheBlock(
            block_id, parent_id, 2, i,
            StandardPrefixBlockPayload(
                token_slots=torch.arange(2 * i, 2 * i + count, dtype=torch.int32),
                host_block_index=i if on_host else None, retained_offsets=tuple(range(count)),
            ), token_ids=tokens,
        )
        index.insert_block(block)
        if on_host:
            index.begin_d2h(block)
            index.finish_d2h(block)
        blocks.append(block)
        parent_id = block_id
    if on_host:
        assert len(index.demote_device_until_freeable(len(blocks))) == len(blocks)
    controller = object.__new__(PrefixOffloadController)
    controller.prefix_cache = index
    controller.block_size = 2
    controller.device = torch.device('cpu')
    controller.host_pool = PinnedPrefixBlockPool(
        capacity_blocks=len(blocks), num_layers=1, block_size=2,
    )
    controller.h2d_stream = object()
    controller.d2h_stream = object()
    controller._new_event = lambda *args: object()
    controller._submit_h2d_layer = lambda *args: None
    controller._submit_d2h_payload = lambda *args: None
    controller._h2d_token_byte_count = lambda n: n * 4
    controller._transfer_token_byte_count = lambda n: n * 4
    controller.h2d_operations = deque()
    controller.d2h_operations = deque()
    controller._h2d_by_block_id = {}
    controller.h2d_bytes = controller.h2d_submitted_operations = controller.h2d_merged_blocks = 0
    controller.d2h_bytes = controller.d2h_submitted_operations = controller.d2h_merged_blocks = 0
    controller.h2d_completed_operations = controller.d2h_completed_operations = 0
    for name, impl in {
        'is_stream_capturing': lambda: False,
        'record_event': lambda *args, **kwargs: None,
        'stream_context': lambda stream: nullcontext(),
        'stream_wait_event': lambda *args: None,
        'synchronize_stream': lambda *args: None,
    }.items():
        monkeypatch.setattr(device_runtime, name, impl)
    return controller, blocks


@pytest.mark.parametrize('counts', [(0, 2), (2, 0, 2), (0, 0)])
def test_pruned_h2d_preserves_root_to_leaf_residency(monkeypatch, counts):
    from sparseengine.engine.prefix_cache import PrefixTransferKind
    controller, blocks = host_chain_controller(monkeypatch, counts)
    operation = controller.submit_h2d(blocks)
    for block, count in zip(blocks, counts):
        assert block.residency.device_present and block.residency.host_present
        assert block.residency.transfer == (PrefixTransferKind.H2D if count else None)
    if any(counts):
        assert operation.blocks == [block for block, count in zip(blocks, counts) if count]
        assert operation.device_token_indices.tolist() == [
            slot for i, count in enumerate(counts) for slot in range(2 * i, 2 * i + count)
        ]
        assert controller.h2d_bytes == sum(counts) * 4
    else:
        assert operation is None
        assert not controller.h2d_operations


@pytest.mark.parametrize('counts,boundary', [
    ((0, 2), 'begin'), ((0, 0), 'begin'),
    ((0, 2), 'transfer'), ((2, 0, 2), 'transfer'),
])
def test_pruned_h2d_failure_rolls_back_empty_and_nonempty_blocks(monkeypatch, counts, boundary):
    controller, blocks = host_chain_controller(monkeypatch, counts)
    def fail(*args):
        raise RuntimeError('injected H2D failure')
    if boundary == 'begin':
        begin = controller.prefix_cache.begin_h2d
        def begin_until_last(block):
            if block is blocks[-1]:
                fail()
            begin(block)
        monkeypatch.setattr(controller.prefix_cache, 'begin_h2d', begin_until_last)
    else:
        monkeypatch.setattr(controller, '_submit_h2d_layer', fail)
    with pytest.raises(RuntimeError, match='injected H2D'):
        controller.submit_h2d(blocks)
    assert all(block.residency.host_present and not block.residency.device_present
               and block.residency.transfer is None for block in blocks)
    assert not controller.h2d_operations and not controller._h2d_by_block_id
    assert controller.h2d_bytes == controller.h2d_submitted_operations == 0


@pytest.mark.parametrize('method', ['', 'quest'], ids=['standard', 'quest'])
def test_shared_h2d_operations_wait_once_per_layer_across_readers(method):
    from cache_contracts.cases import make_inflight_radix
    m, first = make_inflight_radix(method, block_count=6, blocks_per_operation=2)
    second = request(list(first.token_ids))
    second.prefix_cache_hit_len = first.prefix_cache_hit_len
    second.prefix_cache_hit_block_count = first.prefix_cache_hit_block_count
    second.prefix_cache_hit_last_block_id = first.prefix_cache_hit_last_block_id
    h = RadixHarness(m, paged=method == 'quest')
    before = h.observe()
    for seq in (first, second):
        m._attach_prefix_cache_if_needed(seq)
        h.observe()
    controller = m.prefix_offload_controller
    for layer in (0, 1):
        m.before_prefill_layer_attention(layer, None)
    assert [(id(operation), layer) for operation, layer in controller.waited_layers] == [
        (id(operation), layer) for layer in (0, 1) for operation in controller.submitted_h2d
    ]
    for seq in (first, second):
        m.free_seq(seq.seq_id)
    assert h.observe() == before


@pytest.mark.parametrize('corruption', ['offset', 'page', 'out_of_range', 'short'])
def test_quest_attach_rejects_invalid_page_geometry_before_mutation(corruption):
    from cache_contracts.cases import make_inflight_radix
    m, seq = make_inflight_radix('quest', block_count=3, blocks_per_operation=2)
    block = m.prefix_cache.blocks[seq.prefix_cache_hit_last_block_id]
    payload = block.payload
    if corruption == 'offset':
        payload.token_slots = payload.token_slots.flip(0)
    elif corruption == 'page':
        payload.token_slots = payload.token_slots + m.page_size
    elif corruption == 'out_of_range':
        payload.block_slot = m.num_pages
    else:
        payload.token_slots = payload.token_slots[:1]
    free_rows = list(m.free_rows)
    free_pages = m._num_free_pages
    with pytest.raises(RuntimeError, match='page'):
        m._attach_prefix_cache_if_needed(seq)
    assert not m.seq_id_to_row and not m.seq_id_to_prefix_blocks
    assert list(m.free_rows) == free_rows and m._num_free_pages == free_pages
    assert all(block.ref_count == 0 for block in m.prefix_cache.blocks.values())


def test_quest_attach_combines_resident_prefix_and_promoted_suffix():
    from cache_contracts.cases import make_inflight_radix
    m, seq = make_inflight_radix('quest', block_count=3, blocks_per_operation=2)
    for block in m.prefix_cache.blocks.values():
        m.prefix_cache.finish_h2d(block)
    demoted = m.prefix_cache.demote_device_until_freeable(1)
    assert len(demoted) == 1
    m._free_device_prefix_block(demoted[0])
    h = RadixHarness(m, paged=True)
    before_free = len(h.observe().free)
    m._attach_prefix_cache_if_needed(seq)
    assert len(h.observe().free) == before_free - 1
    row = m.seq_id_to_row[seq.seq_id]
    assert m.buffer_req_to_page_slots[row, :2].tolist() == [0, 1]
    assert m.buffer_req_to_token_slots[row, :4].tolist() == [0, 1, 2, 3]
    assert m.buffer_req_to_page_slots[row, 2].item() == demoted[0].payload.block_slot
    m.free_seq(seq.seq_id)
    h.assert_released(seq.seq_id)


@pytest.mark.parametrize('counts', [(0, 2), (2, 0, 2), (0, 0)])
def test_pruned_d2h_publishes_parent_before_empty_child(monkeypatch, counts):
    from sparseengine.platforms import device_runtime
    c, blocks = host_chain_controller(monkeypatch, counts, on_host=False)
    c.submit_d2h(blocks)
    if any(counts):
        assert not any(block.residency.host_present for block in blocks)
        monkeypatch.setattr(device_runtime, 'is_event_complete', lambda event: True)
        assert c.poll_d2h() == 1
    assert all(block.residency.host_present and block.residency.device_present
               and block.residency.transfer is None for block in blocks)
    assert c.host_pool.used_blocks == len(blocks)
    assert c.d2h_bytes == 4 * sum(counts)


@pytest.mark.parametrize('counts,boundary', [
    ((0, 0), 'begin'), ((2, 0, 2), 'begin'), ((2, 0, 2), 'transfer'),
])
def test_pruned_d2h_failure_returns_every_host_allocation(monkeypatch, counts, boundary):
    c, blocks = host_chain_controller(monkeypatch, counts, on_host=False)
    def fail(*args):
        raise RuntimeError('injected D2H failure')
    if boundary == 'begin':
        begin = c.prefix_cache.begin_d2h
        def begin_until_last(block):
            if block is blocks[-1]:
                fail()
            begin(block)
        monkeypatch.setattr(c.prefix_cache, 'begin_d2h', begin_until_last)
    else:
        monkeypatch.setattr(c, '_submit_d2h_payload', fail)
    with pytest.raises(RuntimeError, match='injected D2H'):
        c.submit_d2h(blocks)
    assert c.host_pool.used_blocks == 0
    assert not c.d2h_operations
    assert all(block.residency.device_present and not block.residency.host_present
               and block.residency.transfer is None and block.payload.host_block_index is None
               for block in blocks)


def test_h2d_poll_retires_only_the_completed_stream_prefix(monkeypatch):
    from sparseengine.platforms import device_runtime
    c, blocks = host_chain_controller(monkeypatch, (2, 2, 2))
    operations = [c.submit_h2d([block]) for block in blocks]
    completed = set()
    queries = []
    def ready(event):
        queries.append(event)
        return event in completed
    monkeypatch.setattr(device_runtime, 'is_event_complete', ready)
    assert c.poll_h2d() == 0
    assert queries == [operations[0].completion_event]
    assert all(block.residency.transfer is not None for block in blocks)
    completed.add(operations[0].completion_event)
    assert c.poll_h2d() == 1
    assert blocks[0].residency.transfer is None
    assert all(block.residency.transfer is not None for block in blocks[1:])
    completed.update(operation.completion_event for operation in operations)
    assert c.poll_h2d() == 2
    assert not c.h2d_operations and not c._h2d_by_block_id
    assert all(block.residency.transfer is None for block in blocks)
    queries.clear()
    assert c.poll_h2d() == 0 and not queries


@pytest.mark.parametrize('kind', ['radix', 'mixed'])
def test_replay_ref_is_unique_across_attached_and_materialized_blocks(monkeypatch, kind):
    from types import SimpleNamespace
    from sparseengine.engine.cache_manager.prefix_cache_mixin import PrefixCacheMixin
    from sparseengine.engine.prefix_cache_coordinator import PrefixCacheCoordinator
    c, blocks = host_chain_controller(monkeypatch, (2, 2, 2), on_host=False)
    cls = PrefixCacheMixin if kind == 'radix' else PrefixCacheCoordinator
    m = object.__new__(cls)
    m.prefix_cache = c.prefix_cache
    m.seq_id_to_prefix_blocks = {71: blocks[:1]}
    m.seq_id_to_materialized_blocks = {}
    c.prefix_cache.acquire_block_ref(blocks[0])
    hold = (m._hold_materialized_prefix_block_ref if kind == 'radix'
            else m._hold_materialized_ref)
    seq = SimpleNamespace(seq_id=71)
    for block in (*blocks, *reversed(blocks), *blocks):
        hold(seq, block)
    assert [block.ref_count for block in blocks] == [1, 1, 1]
    assert list(m.seq_id_to_materialized_blocks[71].values()) == blocks[1:]


def test_mixed_attach_waits_once_for_shared_operations_across_requests():
    from copy import copy
    from types import SimpleNamespace
    from cache_contracts.cases import make_inflight_radix
    from sparseengine.engine.prefix_cache_coordinator import PrefixCacheCoordinator, MixedPrefixBlockPayload
    m, first = make_inflight_radix('', block_count=4, blocks_per_operation=2)
    for block in m.prefix_cache.blocks.values():
        block.payload = MixedPrefixBlockPayload(
            kv_payload=block.payload, recurrent_payload=object(), token_count=2,
            accounting_bytes=0, recurrent_bytes=0,
        )
    controller = m.prefix_offload_controller
    for operation in controller.submitted_h2d:
        operation.auxiliary_layer_events = {}
    c = object.__new__(PrefixCacheCoordinator)
    c.prefix_cache = m.prefix_cache
    c.block_size = 2
    c.offload_controller = controller
    c.seq_id_to_prefix_blocks = {}
    c._step_h2d_operations = {}
    c.cache_manager = SimpleNamespace(
        validate_prefix_kv_attach=lambda seq: False,
        attach_prefix_kv_payloads=lambda seq, payloads: None,
        kv_layer_index=lambda layer: layer,
    )
    c.recurrent_state_manager = SimpleNamespace(attach_prefix_recurrent_payload=lambda *args, **kwargs: None)
    second = copy(first)
    second.seq_id = 72
    c.attach_prefix_cache_hits([first, second])
    for layer in (0, 1):
        c.before_prefill_layer_attention(layer)
    assert [(id(operation), layer) for operation, layer in controller.waited_layers] == [
        (id(operation), layer) for layer in (0, 1) for operation in controller.submitted_h2d
    ]
    assert all(block.ref_count == 2 for block in m.prefix_cache.blocks.values())
    c.finish_step()
    c.before_prefill_layer_attention(0)
    assert len(controller.waited_layers) == 4


def test_mixed_recurrent_bytes_follow_index_insert_remove_and_reset():
    from sparseengine.engine.prefix_cache import PrefixCacheBlock, RadixPrefixIndex
    from sparseengine.engine.prefix_cache_coordinator import PrefixCacheCoordinator, MixedPrefixBlockPayload
    c = object.__new__(PrefixCacheCoordinator)
    c.prefix_cache = RadixPrefixIndex(block_size=1, fingerprint=b'byte-accounting')
    assert c._live_recurrent_bytes() == 0
    blocks = []
    for token, size in [(1, 7), (2, 11)]:
        block_id = c.prefix_cache.stable_block_id([token], None)
        payload = MixedPrefixBlockPayload(object(), object(), 1, size, size)
        block = PrefixCacheBlock(block_id, None, 1, 0, payload, token_ids=(token,))
        c.prefix_cache.insert_block(block)
        blocks.append(block)
        assert c._live_recurrent_bytes() == sum(b.payload.recurrent_bytes for b in blocks)
    c.prefix_cache.acquire_block_ref(blocks[0])
    removed = c.prefix_cache.evict_until_freeable(1)
    assert removed == [blocks[1]]
    assert c._live_recurrent_bytes() == 7
    c.prefix_cache.release_block_ref(blocks[0])
    assert c._live_recurrent_bytes() == 7
    c.prefix_cache = RadixPrefixIndex(block_size=1, fingerprint=b'replacement')
    assert c._live_recurrent_bytes() == 0
