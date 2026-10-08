from types import SimpleNamespace

import torch
import pytest

from sparseengine.engine.cache_manager.methods.deepseek_v4 import (
    DeepSeekV4CacheManager, NativeSharedKVRequest,
)
from sparseengine.engine.cache_manager.storage.compressed_family import CompressedKVFamily
from sparseengine.engine.cache_manager.storage.shared_kv_state import SharedKVStateRows
from sparseengine.engine.prefix_cache import RadixPrefixIndex
from sparseengine.engine.runtime_state import RuntimeState
from sparseengine.engine.sequence import Sequence
from test_prefill_schedule_policy import make_scheduler


def native_prefix_owner():
    # Exercise physical ownership and retirement independently of CUDA operators.
    cache = object.__new__(DeepSeekV4CacheManager)
    cache.config = SimpleNamespace(hf_config=SimpleNamespace(dtype=torch.bfloat16),
                                   sparse_method="deepseek_v4", decode_reservation_tokens=1)
    cache.state_rows = SharedKVStateRows(num_rows=4, reserved_rows=1,
        compress_ratios=(4,), device=torch.device("cpu"))
    family = CompressedKVFamily(layer_ids=(0,), with_index=True, num_pages=5,
        reserved_pages=1, page_size=64, device=torch.device("cpu"))
    lease = family.new_lease()
    family.reserve(lease, 4)
    family.mark_materialized(lease, 4)
    row = cache.state_rows.allocate()
    cache.requests = {1: NativeSharedKVRequest(row, 16, {4: lease})}
    cache.families = {4: family}
    cache.max_buffer_rows = 1
    cache.max_model_len = 256
    cache.device = torch.device("cpu")
    cache.family_slots = {4: torch.zeros(4, 256, dtype=torch.int32)}
    cache.prefix_rows = 2
    cache.prefix_cache_block_size = 16
    cache.prefix_cache = RadixPrefixIndex(block_size=16, fingerprint=b"native", max_blocks=2)
    cache.pending_prefix = {}
    cache.private_prefix_records = {}
    cache._async_prefix_records = []
    seq = SimpleNamespace(seq_id=1, num_prompt_tokens=32, token_ids=list(range(32)))
    return cache, seq


def publish_snapshot(cache, seq):
    cache._freeze_prefix_snapshot(seq)
    retired, tokens, record = cache._async_prefix_records.pop(0)
    cache._record_frozen_prefix_materialization(retired, tokens, record)
    cache.publish_pending_prefix_blocks([seq])


def test_prefix_snapshot_is_private_until_retirement_and_survives_append():
    cache, seq = native_prefix_owner()
    request = cache.requests[1]
    for tensor in cache.state_rows.accounting_tensors():
        tensor[request.row].fill_(7)
    family, lease = cache.families[4], request.leases[4]
    for tensor in family.accounting_tensors():
        tensor[lease.pages[0]].fill_(9)
    cache._freeze_prefix_snapshot(seq)
    assert len(cache.prefix_cache) == 0
    frozen_seq, tokens, record = cache._async_prefix_records[0]
    for tensor in cache.state_rows.accounting_tensors():
        tensor[request.row].zero_()
        assert bool((tensor[record.payload.row] == 7).all())
    family.reserve(lease, 5)
    for tensor in family.accounting_tensors():
        tensor[lease.pages[0]].zero_()
        assert bool((tensor[record.payload.leases[4].pages[0]] == 9).all())
    cache._record_frozen_prefix_materialization(frozen_seq, tokens, record)
    cache.publish_pending_prefix_blocks([seq])
    assert len(cache.prefix_cache) == 1
    assert cache.prefix_cache.match_longest_prefix(seq.token_ids, max_usable_tokens=16)[0] == 16
    cache.free_seq(1)
    cache.reset_prefix_cache()
    assert cache.state_rows.num_free_rows == 3
    assert family.allocator.num_free_pages == 4


def test_discarded_async_prefix_releases_unpublished_physical_state():
    cache, seq = native_prefix_owner()
    cache._freeze_prefix_snapshot(seq)
    cache.free_seq(seq.seq_id)
    assert len(cache.prefix_cache) == 0
    assert cache.state_rows.num_free_rows == 3
    assert cache.families[4].allocator.num_free_pages == 4


def test_decode_reservation_counts_shared_tail_copy_only_when_group_publishes():
    cache, seq = native_prefix_owner()
    seq.num_prefilled_tokens, seq.prefix_cache_hit_len = 16, 0
    assert cache.decode_window_costs(seq, 4) == {"ratio_4": 0}
    cache._freeze_prefix_snapshot(seq)
    assert cache.decode_window_costs(seq, 3) == {"ratio_4": 0}
    assert cache.decode_window_costs(seq, 4) == {"ratio_4": 1}
    assert cache.prefill_private_slots_for(seq) == 3
    cache.free_seq(seq.seq_id)


def test_native_prefix_match_reports_only_published_reusable_tokens():
    cache, seq = native_prefix_owner()
    assert cache.prefix_cache_match(seq.token_ids)["matched_tokens"] == 0
    cache._freeze_prefix_snapshot(seq)
    assert cache.prefix_cache_match(seq.token_ids)["matched_tokens"] == 0
    retired_seq, tokens, record = cache._async_prefix_records[0]
    cache._record_frozen_prefix_materialization(retired_seq, tokens, record)
    cache.publish_pending_prefix_blocks([seq])
    result = cache.prefix_cache_match(seq.token_ids)
    assert result["supported"] and result["enabled"]
    assert result["matched_tokens"] == 16
    assert result["matched_blocks"] == 1
    assert result["match_ratio"] == 1.0
    cache.free_seq(seq.seq_id)
    cache.reset_prefix_cache()
    assert cache.prefix_cache_match(seq.token_ids)["matched_tokens"] == 0


@pytest.mark.parametrize("prefix_hit", [False, True])
def test_scheduler_admits_prompt_using_evictable_native_prefix_pages(prefix_hit):
    # Previously cached pages made an otherwise fitting prompt fail admission
    # before _reserve could evict them; allocator-only tests miss that boundary.
    cache, seq = native_prefix_owner()
    cache.enable_prefix_caching = True
    publish_snapshot(cache, seq)
    cache.free_seq(seq.seq_id)
    cache.max_model_len = 1024
    cache.config.max_num_seqs_in_gpu = 1
    runtime = RuntimeState(cache.config, cache)
    scheduler = make_scheduler("all_chunked", chunk=128, max_tokens=128, oracle=runtime)
    tokens = seq.token_ids[:16]+[999]*1008 if prefix_hit else [999]*1024
    request = Sequence(tokens)
    scheduler.add(request)
    batch, prefill, preempted = scheduler.schedule()
    assert batch == [request] and prefill and not preempted
    assert request.prefix_cache_hit_len == 0
    assert cache.families[4].allocator.num_free_pages == 3
    owner = cache._new_request(request.seq_id)
    cache._reserve(owner, request.num_prompt_tokens)
    assert len(cache.prefix_cache) == 0
    assert cache.families[4].allocator.num_free_pages == 0
    cache.free_seq(request.seq_id)
    assert cache.families[4].allocator.num_free_pages == 4


def test_reclaimable_pages_count_all_prefix_owners_and_exclude_live_leases():
    # Two snapshots share a page: it can be counted only once, and only after
    # the request and any unpublished snapshots have released their references.
    cache, seq = native_prefix_owner()
    publish_snapshot(cache, seq)
    cache.requests[1].length = 32
    cache.families[4].mark_materialized(cache.requests[1].leases[4], 8)
    publish_snapshot(cache, seq)
    assert cache.decode_window_budgets() == {"ratio_4": 3}
    cache.free_seq(1)
    assert cache._reclaimable_prefix_capacity() == ({4: 1}, 2)
    assert cache.decode_window_budgets() == {"ratio_4": 4}
    block = next(iter(cache.prefix_cache.blocks.values()))
    cache.prefix_cache.acquire_block_ref(block)
    assert cache.decode_window_budgets() == {"ratio_4": 3}
    cache.prefix_cache.release_block_ref(block)
    assert cache.decode_window_budgets() == {"ratio_4": 4}
    # Ownership changes without a free-page change must invalidate capacity.
    family = cache.families[4]
    retained = family.snapshot(block.payload.leases[4])
    assert cache.decode_window_budgets() == {"ratio_4": 3}
    family.release(retained)
    assert cache.decode_window_budgets() == {"ratio_4": 4}
    cache.reset_prefix_cache()


def test_attached_prefix_initializes_new_row_mapping_before_tail_copy():
    # Restoring a prefix has a logical length but no addresses in its new row.
    # Incremental append must initialize that row and preserve the snapshot.
    cache, seq = native_prefix_owner()
    cache.enable_prefix_caching = True
    publish_snapshot(cache, seq)
    cache.free_seq(1)
    request = Sequence(seq.token_ids[:16]+[999]*64)
    cache.refresh_prefix_cache_hit(request)
    assert request.prefix_cache_hit_len == 16
    addresses = cache.family_slots[4].data_ptr()
    cache._attach_prefix_cache_if_needed(request)
    owner = cache.requests[request.seq_id]
    family = cache.families[4]
    frozen = owner.prefix_blocks[-1].payload.leases[4]
    old_page = frozen.pages[0]
    torch.testing.assert_close(cache.family_slots[4][owner.row, :4],
                               torch.tensor(family.physical_slots(frozen, 0, 4), dtype=torch.int32))
    cache._reserve(owner, 20)
    assert owner.leases[4].pages[0] != old_page
    assert frozen.pages == [old_page]
    expected = torch.tensor(family.physical_slots(owner.leases[4], 0, 5), dtype=torch.int32)
    torch.testing.assert_close(cache.family_slots[4][owner.row, :5], expected)
    assert cache.family_slots[4].data_ptr() == addresses
    cache.free_seq(request.seq_id)
    cache.reset_prefix_cache()
    assert family.allocator.num_free_pages == 4


def test_prefill_pins_every_batch_prefix_before_any_append_can_evict():
    # Earlier append allocations used to evict a later speculative hit after
    # scheduling had already advanced that request's prefill position.
    cache, _ = native_prefix_owner()
    cache.free_seq(1)
    cache.enable_prefix_caching = True
    cache.max_model_len, cache.max_buffer_rows = 1024, 2
    cache.config.max_num_seqs_in_gpu = 2
    cache.config.max_num_batched_tokens = 512
    cache.state_rows = SharedKVStateRows(num_rows=8, reserved_rows=2,
                                       compress_ratios=(4,), device=cache.device)
    cache.family_slots = {4: torch.zeros(8, 256, dtype=torch.int32)}
    cache.prefix_rows, cache.prefix_cache_block_size = 4, 256
    cache.prefix_cache = RadixPrefixIndex(block_size=256, fingerprint=b"native", max_blocks=4)
    cache.compression_planners = {4: lambda *args, **kwargs: None}
    for tag in (2, 1, 3, 4):
        seq = Sequence([tag]*257)
        owner = cache._new_request(seq.seq_id)
        cache._reserve(owner, 256)
        owner.length = 256
        cache.families[4].mark_materialized(owner.leases[4], 64)
        publish_snapshot(cache, seq)
        cache.free_seq(seq.seq_id)
    seqs = [Sequence([1]*256+[11]*256), Sequence([2]*256+[22]*4)]
    scheduler = make_scheduler("all_chunked", chunk=256, max_tokens=512,
                               oracle=RuntimeState(cache.config, cache))
    for seq in seqs:
        scheduler.add(seq)
    batch, prefill, preempted = scheduler.schedule()
    assert batch == seqs and prefill and not preempted
    assert all(seq.num_prefilled_tokens == 256 for seq in seqs)
    inputs, positions, boundaries = cache._prepare_prefill(batch)
    torch.testing.assert_close(inputs, torch.tensor([11]*256+[22]*4))
    torch.testing.assert_close(positions, torch.tensor(list(range(256, 512))+list(range(256, 260))))
    torch.testing.assert_close(boundaries, torch.tensor([0, 256, 260], dtype=torch.int32))
    for seq in seqs:
        owner = cache.requests[seq.seq_id]
        assert owner.length == seq.num_prompt_tokens
        block = owner.prefix_blocks[-1]
        assert cache.prefix_cache.get_block(block.stable_block_id) is block
        assert block.ref_count == 1
        cache.free_seq(seq.seq_id)
    cache.reset_prefix_cache()
    assert cache.families[4].allocator.num_free_pages == 4
    assert cache.state_rows.num_free_rows == 6


def test_prefill_keeps_admitted_hits_when_another_attach_pins_the_same_chain():
    # Historical partial-page snapshots can fill the pool. Pinning the chain
    # removes reclaimable capacity but does not invalidate another admitted hit
    # whose chunk stays within the same compressed group.
    cache, _ = native_prefix_owner()
    cache.free_seq(1)
    cache.enable_prefix_caching = True
    cache.max_model_len, cache.max_buffer_rows = 1024, 2
    cache.config.max_num_seqs_in_gpu, cache.config.max_num_batched_tokens = 2, 32
    cache.state_rows = SharedKVStateRows(num_rows=20, reserved_rows=2,
                                       compress_ratios=(4,), device=cache.device)
    cache.family_slots = {4: torch.zeros(20, 256, dtype=torch.int32)}
    cache.families = {4: CompressedKVFamily(
        layer_ids=(0,), with_index=True, num_pages=17, reserved_pages=1,
        page_size=64, device=cache.device,
    )}
    cache.prefix_rows = 16
    cache.prefix_cache = RadixPrefixIndex(block_size=16, fingerprint=b"native", max_blocks=16)
    cache.compression_planners = {4: lambda *args, **kwargs: None}
    original = Sequence(list(range(256)))
    owner = cache._new_request(original.seq_id)
    for end in range(16, 257, 16):
        cache._reserve(owner, end)
        owner.length = end
        cache.families[4].mark_materialized(owner.leases[4], end//4)
        publish_snapshot(cache, original)
    cache.free_seq(original.seq_id)
    seqs = [Sequence(original.token_ids+[tag]) for tag in (1000, 1001)]
    scheduler = make_scheduler("all_chunked", chunk=16, max_tokens=32,
                               oracle=RuntimeState(cache.config, cache))
    for seq in seqs:
        scheduler.add(seq)
    batch, prefill, preempted = scheduler.schedule()
    assert batch == seqs and prefill and not preempted
    assert all(seq.prefix_cache_hit_len == 256 for seq in seqs)
    cache._prepare_prefill(batch)
    assert all(cache.requests[seq.seq_id].length == seq.num_prompt_tokens for seq in seqs)
    assert cache.families[4].allocator.num_free_pages == 0
    assert cache.requests[seqs[0].seq_id].prefix_blocks == cache.requests[seqs[1].seq_id].prefix_blocks
    for seq in seqs:
        cache.free_seq(seq.seq_id)
    cache.reset_prefix_cache()
    assert cache.families[4].allocator.num_free_pages == 16
    assert cache.state_rows.num_free_rows == 18


@pytest.mark.parametrize("snapshot", [False, True])
def test_incremental_address_updates_preserve_mapping_and_cow(snapshot, monkeypatch):
    # Compare every populated entry to the physical lease, while checking that
    # incomplete groups do no mapping work and a shared tail remains immutable.
    cache, seq = native_prefix_owner()
    cache.max_model_len = 1024
    owner = cache.requests[1]
    family = cache.families[4]
    cache._reserve(owner, 16)
    frozen = family.snapshot(owner.leases[4]) if snapshot else None
    old_page = owner.leases[4].pages[0]
    original = family.physical_slots
    ranges = []

    def mapped(lease, start, stop):
        ranges.append((start, stop))
        return original(lease, start, stop)

    monkeypatch.setattr(family, "physical_slots", mapped)
    for length in (17, 18, 19):
        cache._reserve(owner, length)
    assert not ranges
    for length in (20, 256, 260):
        cache._reserve(owner, length)
        target = length//4
        expected = torch.tensor(original(owner.leases[4], 0, target), dtype=torch.int32)
        torch.testing.assert_close(cache.family_slots[4][owner.row, :target], expected)
        family.mark_materialized(owner.leases[4], target)
        owner.length = length
    assert ranges == ([(0, 5), (5, 64), (64, 65)] if snapshot
                      else [(4, 5), (5, 64), (64, 65)])
    if frozen is not None:
        assert frozen.pages == [old_page]
        assert owner.leases[4].pages[0] != old_page
        family.release(frozen)
    cache.free_seq(1)
    assert family.allocator.num_free_pages == 4
