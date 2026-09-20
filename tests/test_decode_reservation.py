"""Independent capacity/accounting oracles for window renewal and eviction."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sparseengine.engine.cache_manager.decode_reservation import DecodeReservations
from sparseengine.engine.sequence import Sequence
from sparseengine.sampling_params import SamplingParams


class Pools:
    def __init__(self, **free):
        self.free = free

    def decode_window_budgets(self):
        return dict(self.free)

    def decode_window_costs(self, seq, tokens):
        return {name: tokens for name in self.free}


@pytest.mark.parametrize("resident,tokens,budget", [
    (2, 0, 4), (2, 3, 4), (4, 1024, 4), (7, 9, 4),
    (8, 9, 4), (11, 9, 4), (4, 9, 0),
])
def test_streamingllm_window_matches_append_and_compact_peak(resident, tokens, budget):
    # Catch linear-growth reservations for a cache that repeatedly compacts.
    from sparseengine.engine.cache_manager.methods.streamingllm import StreamingLLMCacheManager
    manager = object.__new__(StreamingLLMCacheManager)
    manager.config = SimpleNamespace(sink_keep_tokens=0, recent_keep_tokens=budget)
    manager.kv_transformer_layer_indices = lambda: [0, 2]
    manager.chain_physical_residency = lambda seq_id: (resident, resident + 1)
    expected = {}
    for layer, initial in zip([0, 2], [resident, resident + 1]):
        length = peak = initial
        for _ in range(tokens):
            length += 1
            peak = max(peak, length)
            if budget > 0 and length >= 2 * budget:
                length = budget
        expected[f"layer_{layer}"] = peak - initial
    assert manager.decode_window_costs(request(), tokens) == expected


def request(limit=20):
    seq = Sequence([1, 2], SamplingParams(max_tokens=limit))
    seq.num_prefilled_tokens = 2
    seq.append_token(3)
    return seq


def test_failed_multi_pool_renewal_does_not_partially_commit():
    pools = Pools(raw=12, latent=4)
    ledger = DecodeReservations(pools, 4)
    a, b = request(), request()
    assert ledger.acquire(a)
    assert not ledger.acquire(b)
    assert list(ledger.requests) == [a.seq_id]
    assert pools.free == {'raw': 12, 'latent': 4}
    ledger.release(a.seq_id)
    assert ledger.acquire(b)


def test_generated_slots_replace_reservation_and_renewal_is_bounded():
    pools = Pools(slots=9)
    ledger = DecodeReservations(pools, 4)
    seq = request(limit=7)
    assert ledger.acquire(seq)
    for _ in range(4):
        pools.free['slots'] -= 1
        seq.append_token(4)
    assert ledger.outstanding() == {}
    assert ledger.acquire(seq)
    assert ledger.outstanding() == {'slots': 2}
    ledger.release(seq.seq_id)
    ledger.release(seq.seq_id)
    assert ledger.outstanding() == {}


def test_final_pending_output_does_not_block_another_decode():
    # The last submitted output has consumed its slot; keeping its lease alive
    # until collection used to reject B and trigger unnecessary IDLE eviction.
    pools = Pools(slots=2)
    ledger = DecodeReservations(pools, 1)
    a, b = request(limit=2), request()
    assert ledger.acquire(a)
    pools.free['slots'] -= 1
    a.num_pending_outputs = 1
    assert not ledger.needs_acquisition(a)
    assert ledger.outstanding() == {}
    assert ledger.acquire_many([a, b]) is None
    before = ledger.outstanding()
    a.num_pending_outputs -= 1
    a.append_token(4)
    assert ledger.outstanding() == before == {'slots': 1}
    assert pools.free == {'slots': 1}


def test_pending_outputs_renew_window_before_collection_and_respect_output_limit():
    # A fix to outstanding() alone would still miss renewal at the submitted
    # boundary, or create a lease ending before the newly promised window.
    pools = Pools(slots=5)
    ledger = DecodeReservations(pools, 2)
    seq = request(limit=6)
    assert ledger.acquire(seq)
    seq.num_pending_outputs = 2
    pools.free['slots'] -= 2
    assert ledger.needs_acquisition(seq)
    assert ledger.acquire(seq)
    assert ledger.outstanding() == {'slots': 2}
    for _ in range(2):
        seq.num_pending_outputs -= 1
        seq.append_token(4)
        assert ledger.outstanding() == {'slots': 2}
    seq.num_pending_outputs = 2
    pools.free['slots'] -= 2
    assert ledger.acquire(seq)
    assert ledger.outstanding() == {'slots': 1}
    seq.num_pending_outputs += 1
    pools.free['slots'] -= 1
    assert not ledger.needs_acquisition(seq)
    assert ledger.outstanding() == {}


def test_sole_request_can_make_progress_without_a_whole_window():
    ledger = DecodeReservations(Pools(slots=3), 100)
    seq = request()
    assert not ledger.acquire(seq)
    assert ledger.acquire(seq, allow_short=True)
    assert ledger.requests[seq.seq_id].end - seq.num_completion_tokens == 3


def test_pending_prefill_capacity_is_not_spent_by_decode():
    ledger = DecodeReservations(Pools(slots=6), 4)
    assert not ledger.acquire(request(), prefill_reserve={"slots": 3})
    assert ledger.requests == {}


def test_batch_renewal_accounts_for_live_windows_and_keeps_successful_prefix():
    # A batch-local total must include existing windows and every earlier
    # acquisition, including when a later request fails in a different pool.
    pools = Pools(raw=12, latent=10)
    ledger = DecodeReservations(pools, 4)
    a, b, c = request(), request(), request()
    assert ledger.acquire(a)
    a.append_token(4)
    pools.free = {name: free - 1 for name, free in pools.free.items()}
    reserved = dict(raw=2, latent=1)
    assert ledger.acquire_many([a, b, c], prefill_reserve=reserved) is c
    assert set(ledger.requests) == {a.seq_id, b.seq_id}
    assert ledger.outstanding() == dict(raw=7, latent=7)
    assert all(7 + reserved[name] <= free for name, free in pools.free.items())
    ledger.release(a.seq_id)
    assert ledger.acquire_many([b, c], prefill_reserve=reserved) is None
    assert ledger.outstanding() == dict(raw=8, latent=8)


def test_batch_renewal_refreshes_eviction_headroom_between_calls():
    # Eviction lowers residency but increases the outstanding append headroom
    # of a live window. A persistent cached total would admit B unsafely.
    class EvictingPool(Pools):
        def __init__(self):
            super().__init__(slots=3)
            self.lengths = {}

        def decode_window_costs(self, seq, tokens):
            length = peak = self.lengths[seq.seq_id]
            for _ in range(tokens):
                length += 1
                peak = max(peak, length)
                if length >= 7:
                    length = 4
            return {"slots": peak - self.lengths[seq.seq_id]}

    pool = EvictingPool()
    ledger = DecodeReservations(pool, 4)
    a, b = request(), request()
    pool.lengths = {a.seq_id: 6, b.seq_id: 4}
    assert ledger.acquire_many([a]) is None
    a.append_token(4)
    pool.lengths[a.seq_id] = 4
    pool.free['slots'] += 2
    assert ledger.acquire_many([a, b]) is b  # 3 + 3 cannot fit in five slots.
    assert set(ledger.requests) == {a.seq_id}
    # The expired window must no longer be counted when A itself renews.
    for _ in range(3):
        a.append_token(4)
    assert ledger.acquire_many([a, b]) is b
    assert ledger.requests[a.seq_id].end == a.num_completion_tokens + 4
    ledger.release(a.seq_id)
    assert ledger.acquire_many([b]) is None


def test_batch_failure_propagates_without_committing_failing_request():
    pools = Pools(slots=12)
    ledger = DecodeReservations(pools, 4)
    a, b = request(), request()
    original = pools.decode_window_costs

    def costs(seq, tokens):
        if seq is b:
            raise RuntimeError('missing physical row')
        return original(seq, tokens)

    pools.decode_window_costs = costs
    with pytest.raises(RuntimeError, match='missing physical row'):
        ledger.acquire_many([a, b])
    assert set(ledger.requests) == {a.seq_id}


def test_runtime_reuses_live_windows_without_querying_prefix_capacity():
    # The ledger's early return alone did not prevent its caller from scanning
    # the radix tree first. Exercise that caller and renewal safety together.
    from sparseengine.engine.runtime_state import RuntimeState

    pools = Pools(slots=12)
    pools.prompt_admission_budgets = Mock(return_value={"slots": 9})
    pools.reserved_prefill_slots = Mock(return_value=3)
    pools.decode_window_budgets = Mock(wraps=pools.decode_window_budgets)
    runtime = RuntimeState(SimpleNamespace(engine_prefill_chunk_size=4,
                                          decode_reservation_tokens=4), pools)
    a, b = request(), request()
    waiting = [Sequence([1, 2, 3])]
    assert runtime.reserve_decode_windows([a, b], waiting) is None
    windows = dict(runtime.decode_reservations.requests)
    pools.decode_window_budgets.reset_mock()
    pools.prompt_admission_budgets.reset_mock()
    pools.reserved_prefill_slots.reset_mock()
    for _ in range(4):
        assert runtime.reserve_decode_windows([a, b], waiting) is None
    pools.decode_window_budgets.assert_not_called()
    pools.prompt_admission_budgets.assert_not_called()
    pools.reserved_prefill_slots.assert_not_called()
    assert runtime.decode_reservations.requests == windows

    # A's completed tokens consume real capacity. B's four-token reservation
    # plus the waiting prompt leave insufficient headroom to renew A.
    for _ in range(4):
        a.append_token(4)
        pools.free["slots"] -= 1
    pools.prompt_admission_budgets.return_value = {"slots": 5}
    assert runtime.reserve_decode_windows([a, b], waiting) is a
    pools.decode_window_budgets.assert_called_once()
    assert runtime.decode_reservations.requests == windows
    pools.prompt_admission_budgets.return_value = {"slots": 8}
    pools.reserved_prefill_slots.return_value = 0
    assert runtime.reserve_decode_windows([a, b], []) is None
    assert runtime.decode_reservations.requests[a.seq_id].end == a.num_completion_tokens + 4
    assert runtime.decode_reservations.outstanding() == {"slots": 8}


def test_runtime_finished_and_replay_rows_need_no_new_window_budget():
    from sparseengine.engine.runtime_state import RuntimeState

    pools = Pools(slots=0)
    pools.decode_window_budgets = Mock(side_effect=AssertionError("unused capacity query"))
    runtime = RuntimeState(SimpleNamespace(decode_reservation_tokens=4), pools)
    done = request(limit=1)
    replay = request()
    replay.start_recompute_replay()
    assert runtime.reserve_decode_windows([done, replay], []) is None
    assert runtime.decode_reservations.requests == {}
    pools.decode_window_budgets.assert_not_called()


@pytest.mark.parametrize('window', [1, 3, 7, 32])
def test_window_does_not_schedule_or_reset_eviction(window):
    # Independent per-step simulation: append, then evict at resident length 8.
    class EvictingPool(Pools):
        def __init__(self):
            super().__init__(slots=100)
            self.resident = 4
            self.events = []

        def decode_window_costs(self, seq, tokens):
            length = self.resident
            peak = length
            for _ in range(tokens):
                length += 1
                peak = max(peak, length)
                if length == 8:
                    length = 4
            return {'slots': peak - self.resident}

    pool = EvictingPool()
    ledger = DecodeReservations(pool, window)
    seq = request(limit=21)
    for step in range(20):
        assert ledger.acquire(seq)
        pool.resident += 1
        if pool.resident == 8:
            pool.resident = 4
            pool.events.append(step)
        seq.append_token(5)
    assert pool.events == [3, 7, 11, 15, 19]


@pytest.mark.parametrize('tokens', [1, 3, 16, 257])
def test_snapkv_window_cost_ignores_output_limit(tokens):
    from sparseengine.engine.cache_manager.methods.snapkv import SnapKVCacheManager
    manager = object.__new__(SnapKVCacheManager)
    manager.config = SimpleNamespace(sparse_method='snapkv', sink_keep_tokens=2,
                                     decode_keep_tokens=4, recent_keep_tokens=2,
                                     snapkv_num_full_layers=0)
    manager.kv_transformer_layer_indices = lambda: [0]
    manager.kv_layer_index = lambda i: i
    manager.chain_physical_residency = lambda seq_id: (8,)
    manager._num_free_slots = [1000]
    manager.free_rows = [[]]
    seq = request(limit=8192)
    assert manager.decode_window_costs(seq, tokens) == {'layer_0': tokens}
    assert manager.chain_physical_residency(seq.seq_id) == (8,)


@pytest.mark.parametrize('tokens,eviction', [(0, True), (1, True), (2, True),
                                          (7, True), (31, True), (7, False)])
def test_h2o_window_covers_append_before_eviction(tokens, eviction):
    # Include rows before/at/past the eviction trigger, uneven residency, and
    # non-contiguous layer IDs: the direct cost path must retain physical peaks.
    from sparseengine.engine.cache_manager.methods.h2o import H2OCacheManager
    manager = object.__new__(H2OCacheManager)
    manager.config = SimpleNamespace(sparse_method='h2o', h2o_decode_budget=4,
                                     h2o_decode_eviction_interval=3, h2o_prefill_budget=4,
                                     h2o_decode_eviction=eviction,
                                     engine_prefill_chunk_size=4)
    layers, residents = [0, 2, 3, 5], (2, 6, 7, 9)
    manager.kv_transformer_layer_indices = lambda: layers
    manager.chain_physical_residency = lambda seq_id: residents
    expected = {}
    for layer, resident in zip(layers, residents):
        length = peak = resident
        for _ in range(tokens):
            length += 1
            peak = max(peak, length)
            if eviction and length >= 7:
                length = 4
        expected[f'layer_{layer}'] = peak - resident
    assert manager.decode_window_costs(request(), tokens) == expected


@pytest.mark.parametrize('resident,tokens', [
    (2, 3), (4, 0), (4, 1), (5, 1), (5, 9), (6, 1), (8, 9),
])
def test_pyramidkv_window_starts_from_actual_residency(resident, tokens):
    # A decode window does not perform final-prefill compaction. Existing
    # SnapKV/H2O tests use different retention policies and miss this regression.
    from sparseengine.engine.cache_manager.methods.snapkv import SnapKVCacheManager

    manager = object.__new__(SnapKVCacheManager)
    manager.config = SimpleNamespace(
        sparse_method='pyramidkv', sink_keep_tokens=1, recent_keep_tokens=1,
        decode_keep_tokens=2, pyramid_layer_ratios=[1.0], snapkv_num_full_layers=0,
    )
    manager.kv_transformer_layer_indices = lambda: [0]
    manager.kv_layer_index = lambda layer: layer
    manager.chain_physical_residency = lambda seq_id: (resident,)
    manager._num_free_slots = [100]
    manager.free_rows = [[]]
    length = peak = resident
    for _ in range(tokens):
        length += 1
        peak = max(peak, length)
        if length >= 6:
            length = 4
    assert manager.decode_window_costs(request(), tokens) == {'layer_0': peak - resident}


def make_kivi_manager(seqs, *, sink=8, residual=32, group=32):
    from sparseengine.engine.cache_manager.methods.deltakv_less_memory import DeltaKVLessMemoryCacheManager

    manager = object.__new__(DeltaKVLessMemoryCacheManager)
    manager.config = SimpleNamespace(
        sink_keep_tokens=sink, recent_keep_tokens=32,
        full_layer_kv_quant_bits=4, full_layer_kivi_group_size=group,
        full_layer_kivi_residual_length=residual,
    )
    manager.head_dim = 128
    manager.seq_id_to_row = {seq.seq_id: row for row, seq in enumerate(seqs)}
    manager.row_seq_lens = [40] * len(seqs)
    manager.row_deltakv_compressed_lens = [0] * len(seqs)
    manager.row_full_layer_kivi_quantized_lens = [sink] * len(seqs)
    manager._num_free_slots_deltakv_full = 100_000
    manager._num_free_slots_deltakv_latent = 100_000
    manager._num_free_slots_full_layer_kivi = 100_000
    return manager


@pytest.mark.parametrize('length,quantized,sink,residual,group,tokens', [
    (3, 0, 8, 32, 32, 1),       # Prompt has not filled the sink yet.
    (40, 8, 8, 32, 32, 0),
    (40, 8, 8, 32, 32, 7),      # Window ends before any quantization.
    (71, 8, 8, 32, 32, 1),      # Append must fit before freeing a group.
    (40, 8, 8, 32, 32, 1024),   # Many quantization cycles in one window.
    (104, 72, 8, 32, 32, 1024), # Previously quantized history is not raw KV.
    (104, 8, 8, 32, 32, 1024),  # An overdue eviction still needs an append slot.
    (93, 16, 16, 48, 16, 41),   # Residual and group lengths can differ.
    (32, 0, 0, 32, 32, 65),
])
def test_kivi_window_matches_append_then_quantize_oracle(
    length, quantized, sink, residual, group, tokens,
):
    # StreamingLLM's whole-row compaction tests do not cover KIVI group release.
    seq = request()
    manager = make_kivi_manager([seq], sink=sink, residual=residual, group=group)
    manager.row_seq_lens[0] = length
    manager.row_full_layer_kivi_quantized_lens[0] = quantized
    raw = list(range(sink, length))
    raw = [position for position in raw if position >= quantized]
    initial = peak = min(sink, length) + len(raw)
    blocks = 0
    for position in range(length, length + tokens):
        if position >= sink:
            raw.append(position)
        peak = max(peak, min(sink, position + 1) + len(raw))
        while len(raw) >= residual + group:
            del raw[:group]
            blocks += 1
    costs = manager.decode_window_costs(seq, tokens)
    assert costs['full_layers'] == peak - initial
    assert costs['full_layer_kivi_blocks'] == blocks


def test_kivi_bounded_raw_pool_keeps_two_decode_windows_reserved():
    # Reproduce the review's unnecessary preemption: the pool can hold both
    # append-before-quantize peaks, but cannot hold 1024 raw tokens per request.
    seqs = [request(limit=2050), request(limit=2050)]
    manager = make_kivi_manager(seqs)
    capacity = manager._resident_full_layer_raw_overhead_slots(2, 8, 32)
    raw = [list(range(8, 40)) for _ in seqs]
    manager._num_free_slots_full = capacity - sum(8 + len(row) for row in raw)
    ledger = DecodeReservations(manager, 1024)
    for _ in range(2049):
        for seq in seqs:
            assert ledger.acquire(seq)
        assert ledger.outstanding()['full_layers'] <= manager._num_free_slots_full
        for row, seq in enumerate(seqs):
            raw[row].append(manager.row_seq_lens[row])
            manager.row_seq_lens[row] += 1
        assert sum(8 + len(row) for row in raw) <= capacity
        for row, seq in enumerate(seqs):
            while len(raw[row]) >= 64:
                del raw[row][:32]
                manager.row_full_layer_kivi_quantized_lens[row] += 32
                manager._num_free_slots_full_layer_kivi -= 1
            seq.append_token(4)
        manager._num_free_slots_full = capacity - sum(8 + len(row) for row in raw)
    assert ledger.outstanding() == {}


def test_deltakv_latent_reservation_does_not_consume_prefill_raw_slots():
    # A block compression needs many latent slots but only one raw append.
    # Ledger-only tests miss RuntimeState collapsing these distinct pools.
    from sparseengine.engine.cache_manager.methods.deltakv_base import DeltaKVCacheManager
    from sparseengine.engine.runtime_state import RuntimeState

    seq = Sequence([1] * 263, SamplingParams(max_tokens=10))
    seq.num_prefilled_tokens = seq.num_prompt_tokens
    seq.append_token(2)
    manager = object.__new__(DeltaKVCacheManager)
    manager.config = SimpleNamespace(
        sink_keep_tokens=8, recent_keep_tokens=128, decode_reservation_tokens=1,
        max_num_seqs_in_gpu=2,
    )
    manager.seq_id_to_row = {seq.seq_id: 0}
    manager.row_seq_lens = [seq.num_prompt_tokens]
    manager.row_deltakv_compressed_lens = [0]
    manager._num_free_slots_full = 64
    manager._num_free_slots_deltakv_full = 80
    manager._deltakv_temp_full_reserve = 16
    manager._num_free_slots_deltakv_latent = 1024
    manager._deltakv_centers_capacity = 128
    manager._deltakv_centers_reserved_total = 0
    runtime = RuntimeState(manager.config, manager)

    assert runtime.decode_reservations.acquire(seq)
    reserved = runtime.decode_reservations.outstanding()
    assert reserved['deltakv_latent'] > manager.prefill_step_free_slots()
    # Only the next raw append competes with prefill in either raw pool.
    expected = manager.prefill_step_free_slots() - 1
    assert runtime.prefill_step_free_slots() == expected
    assert runtime.prompt_admission_free_slots() == expected
    budgets = runtime.prompt_admission_budgets([], 128)
    assert budgets['full_layers'] == budgets['deltakv_raw'] == expected
    assert runtime.decode_reservations.outstanding() == reserved
    runtime.decode_reservations.release(seq.seq_id)
    assert runtime.prefill_step_free_slots() == manager.prefill_step_free_slots()
    assert runtime.prompt_admission_free_slots() == manager.prompt_admission_free_slots()


def test_kivi_prefill_subtracts_each_raw_pool_before_taking_minimum():
    # KIVI bounds full-layer raw growth while sparse raw growth uses a different
    # pool. Subtracting the largest raw reservation from the smallest pool fails.
    from sparseengine.engine.runtime_state import RuntimeState

    seq = request(limit=2050)
    manager = make_kivi_manager([seq])
    manager.config.decode_reservation_tokens = 1024
    manager._num_free_slots_full = 96
    manager._num_free_slots_deltakv_full = 4096
    runtime = RuntimeState(manager.config, manager)
    assert runtime.decode_reservations.acquire(seq)
    reserved = runtime.decode_reservations.outstanding()
    full_available = manager._num_free_slots_full - reserved['full_layers']
    raw_available = manager._num_free_slots_deltakv_full - reserved['deltakv_raw']
    assert 0 < full_available < raw_available
    assert runtime.prefill_step_free_slots() == full_available
    assert runtime.prompt_admission_free_slots() == full_available

    # Exhausting the other raw pool must block prefill without borrowing latent
    # space or reducing the independent full-layer admission budget.
    manager._num_free_slots_deltakv_full = reserved['deltakv_raw']
    assert runtime.prefill_step_free_slots() == 0
    assert runtime.prompt_admission_free_slots() == full_available
