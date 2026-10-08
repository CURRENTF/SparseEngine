from math import ceil
from types import SimpleNamespace

import pytest
import torch

from sparseengine.engine.cache_manager.methods.deepseek_v4 import DeepSeekV4CacheManager
from sparseengine.engine.cache_manager.storage.compressed_family import CompressedKVFamily
from sparseengine.engine.cache_manager.storage.shared_kv_state import SharedKVStateRows
from sparseengine.engine.prefix_cache import RadixPrefixIndex
from sparseengine.engine.runtime_state import RuntimeState
from sparseengine.engine.sequence import Sequence, SequenceStatus
from sparseengine.sampling_params import SamplingParams
from test_prefill_schedule_policy import make_scheduler


def native_capacity_owner(*, prefix_block_size=None):
    cache = object.__new__(DeepSeekV4CacheManager)
    cache.config = SimpleNamespace(
        hf_config=SimpleNamespace(dtype=torch.bfloat16), sparse_method="deepseek_v4",
        engine_prefill_chunk_size=128, max_num_seqs_in_gpu=2,
        max_num_batched_tokens=512, decode_reservation_tokens=1,
    )
    cache.ratios = (4, 128)
    cache.max_model_len = 8192
    cache.max_buffer_rows = 2
    cache.device = torch.device("cpu")
    cache.enable_prefix_caching = False
    cache.prefix_cache = None
    cache.requests = {}
    cache.pending_prefix = {}
    cache.private_prefix_records = {}
    cache.state_rows = SharedKVStateRows(num_rows=4, reserved_rows=2,
                                        compress_ratios=cache.ratios, device=cache.device)
    cache.families = {
        ratio: CompressedKVFamily(layer_ids=(layer,), with_index=ratio == 4,
                                 num_pages=pages+1, reserved_pages=1,
                                 page_size=cache.page_size, device=cache.device)
        for layer, (ratio, pages) in enumerate(((4, 32), (128, 1)))
    }
    cache.family_slots = {
        ratio: torch.zeros(cache.state_rows.num_rows, ceil(cache.max_model_len/ratio),
                           dtype=torch.int32)
        for ratio in cache.families
    }
    if prefix_block_size is not None:
        cache.enable_prefix_caching = True
        cache.prefix_rows = 32
        cache.prefix_cache_block_size = prefix_block_size
        cache.state_rows = SharedKVStateRows(
            num_rows=cache.max_buffer_rows*2+cache.prefix_rows,
            reserved_rows=cache.max_buffer_rows, compress_ratios=cache.ratios,
            device=cache.device,
        )
        cache.family_slots = {
            ratio: torch.zeros(cache.state_rows.num_rows, ceil(cache.max_model_len/ratio),
                               dtype=torch.int32)
            for ratio in cache.families
        }
        cache.prefix_cache = RadixPrefixIndex(
            block_size=prefix_block_size, fingerprint=b"native", max_blocks=cache.prefix_rows,
        )
        # These lifecycle tests retire synthetic writes; no CUDA plan is executed.
        cache.compression_planners = {ratio: lambda *args, **kwargs: None
                                      for ratio in cache.families}
    return cache


def resident_request(cache, prompt_length, cached_length, *, decoding=False):
    seq = Sequence([1]*prompt_length, SamplingParams(max_tokens=2, temperature=0))
    seq.num_prefilled_tokens = cached_length
    seq.status = SequenceStatus.RUNNING
    request = cache._new_request(seq.seq_id)
    cache._reserve(request, cached_length)
    for ratio, lease in request.leases.items():
        cache.families[ratio].mark_materialized(lease, cached_length//ratio)
    request.length = cached_length
    if decoding:
        seq.append_token(2)
    return seq


@pytest.mark.parametrize("length", [127, 256, 8191])
def test_scheduler_decode_matches_independent_family_capacity(length):
    # Previously a ratio-128 page cost was compared with ratio-4 token capacity;
    # once ratio-128 was full, appends into its private tail also stopped.
    cache = native_capacity_owner()
    seq = resident_request(cache, length, length, decoding=True)
    runtime = RuntimeState(cache.config, cache)
    scheduler = make_scheduler("all_chunked", oracle=runtime)
    scheduler.decoding.append(seq)
    assert runtime.startup_decode_batch_fits([seq])
    batch, prefill, preempted = scheduler.schedule()
    assert batch == [seq] and not prefill and not preempted
    cache._reserve(cache.requests[seq.seq_id], length+1)


def test_decode_batch_cannot_spend_one_family_page_twice():
    # Per-request fits do not prove a whole batch fits in independent pools.
    cache = native_capacity_owner()
    seqs = [resident_request(cache, 127, 127, decoding=True) for _ in range(2)]
    runtime = RuntimeState(cache.config, cache)
    before = {r: f.allocator.num_free_pages for r, f in cache.families.items()}
    assert all(runtime.startup_decode_batch_fits([seq]) for seq in seqs)
    assert not runtime.startup_decode_batch_fits(seqs)
    scheduler = make_scheduler("all_chunked", oracle=runtime)
    scheduler.decoding.extend(seqs)
    assert scheduler._eligible_decode_batch() == seqs[:1]
    assert list(scheduler.decoding) == seqs
    assert before == {r: f.allocator.num_free_pages for r, f in cache.families.items()}


def test_prefill_can_grow_one_family_while_another_has_only_private_capacity():
    cache = native_capacity_owner()
    seq = resident_request(cache, 257, 256)
    assert cache.families[128].allocator.num_free_pages == 0
    runtime = RuntimeState(cache.config, cache)
    scheduler = make_scheduler("all_chunked", oracle=runtime)
    scheduler.waiting.append(seq)
    batch, prefill, preempted = scheduler.schedule()
    assert batch == [seq] and prefill and not preempted
    assert seq.current_chunk_size == 1
    cache._reserve(cache.requests[seq.seq_id], seq.num_prefilled_tokens+seq.current_chunk_size)


def test_prefill_chunk_stops_before_an_unavailable_family_page():
    # Available private slots can fund part of a chunk, but never a new page.
    cache = native_capacity_owner()
    seq = resident_request(cache, 260, 255)
    family = cache.families[4]
    held = family.allocator.allocate_pages(family.allocator.num_free_pages)
    scheduler = make_scheduler("all_chunked", oracle=RuntimeState(cache.config, cache))
    scheduler.waiting.append(seq)
    batch, prefill, preempted = scheduler.schedule()
    assert batch == [seq] and prefill and not preempted
    assert seq.current_chunk_size == 4
    cache._reserve(cache.requests[seq.seq_id], 259)
    with pytest.raises(MemoryError, match="family exhausted"):
        cache._reserve(cache.requests[seq.seq_id], 260)
    family.allocator.release_pages(held)


def test_prefill_reports_exhausted_family_instead_of_spinning_without_progress():
    # Independent budgets must retain the scheduler's explicit deadlock failure
    # when an admitted partial prompt cannot append even one token.
    cache = native_capacity_owner()
    seq = resident_request(cache, 260, 259)
    family = cache.families[4]
    held = family.allocator.allocate_pages(family.allocator.num_free_pages)
    scheduler = make_scheduler("all_chunked", oracle=RuntimeState(cache.config, cache))
    scheduler.waiting.append(seq)
    with pytest.raises(RuntimeError, match="No prefill candidate can use"):
        scheduler.schedule()
    assert list(scheduler.waiting) == [seq]
    assert cache.requests[seq.seq_id].length == 259
    family.allocator.release_pages(held)


def test_prefill_batch_charges_independent_page_budgets():
    cache = native_capacity_owner()
    seqs = [resident_request(cache, 128, 127) for _ in range(2)]
    scheduler = make_scheduler("all_chunked", oracle=RuntimeState(cache.config, cache))
    scheduler.waiting.extend(seqs)
    batch, prefill, preempted = scheduler.schedule()
    assert batch == seqs[:1] and prefill and not preempted
    assert list(scheduler.waiting) == seqs[1:]
    cache._reserve(cache.requests[batch[0].seq_id], 128)


def test_prefill_preserves_another_requests_decode_page_reservation():
    cache = native_capacity_owner()
    decode = resident_request(cache, 127, 127, decoding=True)
    prefill = resident_request(cache, 128, 127)
    runtime = RuntimeState(cache.config, cache)
    assert runtime.decode_reservations.acquire(decode)
    budgets = runtime.step_allocation_budgets(is_prefill=True)
    assert runtime.step_allocation_costs(prefill, 1)["ratio_128"] > budgets["ratio_128"]
    scheduler = make_scheduler("all_chunked", oracle=runtime)
    scheduler.waiting.append(prefill)
    scheduler.decoding.append(decode)
    batch, is_prefill, preempted = scheduler.schedule()
    assert batch == [decode] and not is_prefill and not preempted


@pytest.mark.parametrize("block_size", [16, 128])
@pytest.mark.parametrize("prompt_length", [128, 512])
def test_native_snapshots_cannot_stall_an_admitted_prompt_or_its_private_tail(
    block_size, prompt_length,
):
    # A snapshot of the sole ratio-128 page previously forced a COW allocation
    # that could never fit. Exercise scheduler progress across compression groups.
    cache = native_capacity_owner(prefix_block_size=block_size)
    runtime = RuntimeState(cache.config, cache)
    scheduler = make_scheduler("all_chunked", chunk=128, max_tokens=128, oracle=runtime)
    seq = Sequence([1]*prompt_length)
    scheduler.add(seq)
    for _ in range(prompt_length):
        batch, prefill, preempted = scheduler.schedule()
        assert batch == [seq] and prefill and not preempted
        cache._prepare_prefill(batch)
        cache.on_forward_end(batch, is_prefill=True)
        seq.num_prefilled_tokens = cache.requests[seq.seq_id].length
        if seq.num_prefilled_tokens == seq.num_prompt_tokens:
            break
        scheduler.waiting.append(seq)
    else:
        pytest.fail("Admitted native prompt did not finish")
    owner = cache.requests[seq.seq_id]
    cache._reserve(owner, prompt_length+128)
    assert cache.families[128].allocator.reference_count(owner.leases[128].pages[-1]) == 1
    if block_size < 128:
        assert cache.prefix_cache_match(seq.token_ids)["matched_tokens"] > 0
    cache.free_seq(seq.seq_id)
    cache.reset_prefix_cache()
    assert cache.state_rows.num_free_rows == cache.state_rows.num_rows-cache.max_buffer_rows
    assert {r: f.allocator.num_free_pages for r, f in cache.families.items()} == {4: 32, 128: 1}


@pytest.mark.parametrize("other_is_decoding", [False, True])
def test_native_snapshot_preserves_other_prompt_and_decode_commitments(other_is_decoding):
    # A snapshot's new tail copy must not consume a page promised to another
    # partial prompt or an acquired decode window.
    cache = native_capacity_owner(prefix_block_size=16)
    cache.config.decode_reservation_tokens = 4
    cache.families = {
        ratio: CompressedKVFamily(layer_ids=(layer,), with_index=ratio == 4,
                                 num_pages=pages+1, reserved_pages=1,
                                 page_size=cache.page_size, device=cache.device)
        for layer, (ratio, pages) in enumerate(((4, 3), (128, 2)))
    }
    current = Sequence([1]*16)
    other = Sequence([2]*(256 if other_is_decoding else 512))
    current.current_chunk_size, other.current_chunk_size = 16, 256
    cache._prepare_prefill([current, other])
    for owner in cache.requests.values():
        for ratio, lease in owner.leases.items():
            cache.families[ratio].mark_materialized(lease, owner.length//ratio)
    runtime = RuntimeState(cache.config, cache)
    if other_is_decoding:
        other.num_prefilled_tokens = other.num_prompt_tokens
        other.append_token(3)
        assert runtime.decode_reservations.acquire(other)
        assert runtime.decode_reservations.outstanding()["ratio_4"] == 1
    cache._freeze_prefix_snapshot(current)
    cache.publish_pending_prefix_blocks([current])
    assert len(cache.prefix_cache) == 0
    cache._reserve(cache.requests[other.seq_id], 260 if other_is_decoding else 512)
    cache._reserve(cache.requests[current.seq_id], 20)
    for seq in (current, other):
        cache.free_seq(seq.seq_id)
    assert {r: f.allocator.num_free_pages for r, f in cache.families.items()} == {4: 3, 128: 2}


def allocation_owner(budget):
    cache = object.__new__(DeepSeekV4CacheManager)
    cache.config = SimpleNamespace()
    cache.ratios = (4, 128)
    cache.max_buffer_rows = 1
    cache.max_model_len = 8192
    cache.device = torch.device("cpu")
    cache.enable_prefix_caching = False
    cache.allocation_budget_bytes = budget
    return cache


@pytest.mark.parametrize("slack", [-1, 0, 1])
def test_total_persistent_bytes_respect_budget_before_family_allocation(slack, monkeypatch):
    # Derive the minimum from real tensor storage, independently of the planner.
    # Previous checks admitted two families that each fit but whose sum did not.
    measured = allocation_owner(4*1024**2)
    measured.allocate_kv_cache()
    actual = sum(t.numel()*t.element_size() for _, t in measured._iter_accounting_tensors())
    minimum, allocated = 0, 0
    for family in measured.families.values():
        page_bytes = sum(t[0].numel()*t.element_size() for t in family.accounting_tensors())
        minimum += page_bytes
        allocated += family.allocator.num_free_pages*page_bytes
    budget = actual-allocated+minimum+slack
    cache = allocation_owner(budget)
    if slack < 0:
        def unexpected_allocation(**_):
            pytest.fail("Physical families allocated before checking the total byte budget")
        monkeypatch.setattr(
            "sparseengine.engine.cache_manager.methods.deepseek_v4.CompressedKVFamily",
            unexpected_allocation,
        )
        with pytest.raises(MemoryError, match="families need"):
            cache.allocate_kv_cache()
    else:
        cache.allocate_kv_cache()
        persistent = sum(t.numel()*t.element_size() for _, t in cache._iter_accounting_tensors())
        assert persistent <= budget
