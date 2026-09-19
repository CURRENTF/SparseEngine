"""Protect snapshot ownership, admission accounting, and actual CUDA transfers."""

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sparseengine.engine.cache_manager import chain_offload
from sparseengine.engine.cache_manager.chain_offload import ChainOffloadController
from sparseengine.engine.cache_manager.methods.h2o import H2OCacheManager
from sparseengine.engine.cache_manager.methods.rkv import RKVCacheManager
from sparseengine.engine.cache_manager.methods.skipkv import (
    SkipKVCacheManager, SkipKVSentence, SkipKVSequenceState,
)
from sparseengine.engine.cache_manager.methods.snapkv import SnapKVCacheManager
from sparseengine.engine.cache_manager.raw_kv_offload import RawKVOffloadBuffer
from sparseengine.engine.cache_manager.storage.mla_latent import MlaLatentStorage
from sparseengine.engine.chain_cache import ChainCacheCoordinator, ChainCapacityError, ChainGoneError, ChainState
from sparseengine.engine.runtime_state import RuntimeState


def make_manager(cls=SnapKVCacheManager, device="cpu", rows=2, capacity=16):
    """Use real slot allocation/free and method hooks with small owned storage."""
    m = object.__new__(cls)
    m.device = torch.device(device)
    m.num_layers = m.num_kv_layers = 2
    m.max_model_len = 32
    m.runtime_layout = SimpleNamespace(kv_idx_to_layer_idx=(0, 1), kv_layer_index=lambda i: i)
    m.config = SimpleNamespace(
        sparse_method="snapkv", max_model_len=32, max_num_seqs_in_gpu=rows,
        sink_keep_tokens=1, recent_keep_tokens=1, decode_keep_tokens=8,
        snapkv_num_full_layers=0, model="test-model", tensor_parallel_size=1,
        hf_config=SimpleNamespace(model_type="test", dtype=torch.float16),
    )
    m.kv_cache = torch.arange(2 * 2 * capacity * 8, dtype=torch.float16, device=device).reshape(2, 2, capacity, 1, 8)
    m.free_rows = [deque(range(rows)) for _ in range(2)]
    m.seq_id_to_row = [{}, {}]
    m.row_seq_lens = [np.zeros(rows, dtype=np.int32) for _ in range(2)]
    m._num_free_slots = [capacity, capacity]
    m.free_slots_stack = [torch.arange(capacity, dtype=torch.int32, device=device) for _ in range(2)]
    m.buffer_req_to_token_slots = [torch.zeros((rows, 32), dtype=torch.int32, device=device) for _ in range(2)]
    m._prefill_attn_score_accumulators = {}
    m.raw_kv_offload_buffer = RawKVOffloadBuffer(pin_memory=False)
    if issubclass(cls, H2OCacheManager):
        m._h2o_scores = {}
        m._h2o_active_decode_seq_ids = set()
    if issubclass(cls, RKVCacheManager):
        m._rkv_query_cache_enabled = True
        m._rkv_query_cache = [torch.arange(rows * 3 * 8, dtype=torch.float16, device=device).reshape(rows, 3, 1, 8) for _ in range(2)]
        m._rkv_query_positions = [torch.arange(rows * 3, dtype=torch.int32, device=device).reshape(rows, 3) for _ in range(2)]
    if issubclass(cls, SkipKVCacheManager):
        m._skipkv_seq_states = {}
        m._skipkv_row_gen_indices = [{}, {}]
    return m


def populate(m, seq_id=1, lengths=(3, 5)):
    expected = []
    for layer, length in enumerate(lengths):
        slots = m._allocate(layer, seq_id, length).long()
        k, v = m.chain_storage_tensors(layer)
        expected.append((k[slots].clone(), v[slots].clone()))
        if isinstance(m, H2OCacheManager):
            m._h2o_scores[layer, seq_id] = torch.arange(length, dtype=torch.float32, device=m.device) + 0.5
        if isinstance(m, SkipKVCacheManager):
            row = m.seq_id_to_row[layer][seq_id]
            m._skipkv_row_gen_indices[layer][row] = list(range(length))
    if isinstance(m, SkipKVCacheManager):
        m._skipkv_seq_states[seq_id] = SkipKVSequenceState(
            num_prompt_tokens=10,
            sentences=[SkipKVSentence(10, 12, torch.ones(8, device=m.device), cache_ranges={0: (1, 3)})],
        )
    return expected


@pytest.fixture
def cpu_transfers(monkeypatch):
    """Only transport is simulated; these tests establish CPU lifecycle contracts."""
    runtime = chain_offload.device_runtime
    monkeypatch.setattr(runtime, "supports_pin_memory", lambda: True)
    monkeypatch.setattr(runtime, "supports_streams", lambda d: True)
    monkeypatch.setattr(runtime, "new_stream", lambda d: object())
    monkeypatch.setattr(runtime, "new_event", lambda d: object())
    monkeypatch.setattr(runtime, "is_stream_capturing", lambda: False)
    monkeypatch.setattr(runtime, "is_event_complete", lambda e: False)
    monkeypatch.setattr(runtime, "stream_context", lambda s: nullcontext())
    waits = []
    monkeypatch.setattr(runtime, "record_event", lambda *args: None)
    monkeypatch.setattr(runtime, "stream_wait_event", lambda *args: None)
    monkeypatch.setattr(runtime, "synchronize_event", lambda e: waits.append(e))
    monkeypatch.setattr(runtime, "synchronize_stream", lambda s: None)
    monkeypatch.setattr(chain_offload, "HostTensorPool", lambda shapes, dtype: SimpleNamespace(
        tensors=tuple(torch.empty(shape, dtype=dtype) for shape in shapes)))
    def transfer(sk, dk, sv, dv, si, di, item_size):
        dk[di] = sk[si]
        dv[di] = sv[si]
    monkeypatch.setattr(chain_offload, "_load_kvcache_transfer_ops", lambda: (None, transfer))
    def single_transfer(src, dst, si, di, item_size):
        dst[di] = src[si]
    monkeypatch.setattr(chain_offload, "_load_single_transfer", lambda: single_transfer)
    return waits


def test_resume_waits_for_copy_then_invalidates_snapshot(cpu_transfers):
    m = make_manager()
    populate(m)
    c = ChainOffloadController(m, 4096)
    c.save(1, m.snapshot_chain_method_state(1))
    assert c.snapshots[1].completion is not None
    assert not c.snapshots[1].valid
    c.invalidate(1)
    assert len(cpu_transfers) == 1
    assert not c.snapshots[1].valid
    assert m.chain_has_residency(1)
    m.free_seq(1)
    with pytest.raises(RuntimeError, match="invalid"):
        c.restore(1)
    c.reset()
    assert c.used_bytes == 0


def round_trip(m):
    expected = populate(m)
    original = m.snapshot_chain_method_state(1)
    expected_tensors = {name: t.clone() for name, t in original.tensors.items()}
    c = ChainOffloadController(m, 16384)
    c.save(1, original)
    c.wait(1)
    m.free_seq(1)
    # Force a different row and physical slots; stale slot IDs cannot pass.
    populate(m, 2, (2, 2))
    c.restore(1)
    for layer, (expected_k, expected_v) in enumerate(expected):
        row = m.seq_id_to_row[layer][1]
        slots = m.buffer_req_to_token_slots[layer][row, :len(expected_k)].long()
        k, v = m.chain_storage_tensors(layer)
        torch.testing.assert_close(k[slots], expected_k, rtol=0, atol=0)
        torch.testing.assert_close(v[slots], expected_v, rtol=0, atol=0)
    restored = m.snapshot_chain_method_state(1)
    for name, tensor in expected_tensors.items():
        torch.testing.assert_close(restored.tensors[name], tensor, rtol=0, atol=0)
    if isinstance(m, SkipKVCacheManager):
        assert m._skipkv_row_gen_indices[1][m.seq_id_to_row[1][1]] == list(range(5))
        assert m._skipkv_seq_states[1].sentences[0].cache_ranges == {0: (1, 3)}
    # Rewrite existing tokens too: a suffix-only copy would preserve old values.
    c.invalidate(1)
    for layer in range(2):
        for tensor in m.chain_storage_tensors(layer):
            tensor.add_(7)
    c.save(1, m.snapshot_chain_method_state(1))
    c.wait(1)
    torch.testing.assert_close(c.snapshots[1].kv[0][0], expected[0][0].cpu() + 7, rtol=0, atol=0)
    c.reset()
    m.free_seq(1)
    m.free_seq(2)
    assert m._num_free_slots == [16, 16]


@pytest.mark.parametrize("cls", [SnapKVCacheManager, H2OCacheManager, RKVCacheManager, SkipKVCacheManager])
def test_method_state_and_slot_remapping(cpu_transfers, cls):
    round_trip(make_manager(cls))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual pinned-memory transfer requires CUDA")
@pytest.mark.parametrize("cls", [SnapKVCacheManager, H2OCacheManager, RKVCacheManager, SkipKVCacheManager])
def test_cuda_whole_chain_round_trip(cls):
    round_trip(make_manager(cls, "cuda:0"))


def mla_manager(device):
    m = make_manager(H2OCacheManager, device)
    storage = MlaLatentStorage(kv_lora_rank=512, rope_dim=64, dtype=torch.bfloat16)
    storage.allocate(num_layers=2, num_slots=16, device=torch.device(device))
    storage.latent_cache.fill_(1)
    storage.rope_cache.fill_(2)
    m.attention_cache_storage = storage
    m.kv_cache = None
    return m


def test_mla_latent_and_rope_use_their_own_token_width(cpu_transfers):
    round_trip(mla_manager("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual latent transfer requires CUDA")
def test_cuda_mla_whole_chain_round_trip():
    round_trip(mla_manager("cuda:0"))


def make_runtime(cpu_transfers, rows=1):
    m = make_manager(rows=rows)
    m.config.enable_prefix_cache_offload = True
    m.config.prefix_cache_host_size_gb = 4096 / 1024**3
    coordinator = ChainCacheCoordinator(m.config, m)
    runtime = RuntimeState(m.config, m, chain_cache_coordinator=coordinator)
    return m, coordinator, runtime


def finish_chain(m, c, r, seq_id, name, lengths=(3, 5)):
    p = c.plan_admission(chain_id=name, seq_id=seq_id, token_ids=[1, 2, 3, 4, 5])
    r.chain_apply_admission(p)
    populate(m, seq_id, lengths)
    r._resident_seq_ids.add(seq_id)
    record = c.index.finish(name, token_ids=[1, 2, 3, 4, 5], processed_token_count=5,
                            physical_slots_by_layer=lengths)
    c.save_finished_chain(record)


def test_cpu_only_chain_survives_demote_and_reserves_restore(cpu_transfers):
    m, c, r = make_runtime(cpu_transfers)
    finish_chain(m, c, r, 1, "a")
    finish_chain(m, c, r, 2, "b")
    assert c.index.lookup("a").resident_rows == 0
    assert not m.chain_has_residency(1)
    assert r._resident_seq_ids == {2}
    p = c.plan_admission(chain_id="a", seq_id=1, token_ids=[1, 2, 3, 4, 5, 6])
    assert p.reserved_slots_by_layer == (4, 6)
    assert p.reserved_rows == 1
    assert p.demote_chain_ids == ("b",)
    assert p.victim_chain_ids == ()
    c.validate_admission_plan(p, input_token_count=6,
        input_prefix_digest=c.index.lookup("a").processed_token_digest)
    r.chain_apply_admission(p)
    assert m.chain_physical_residency(1) == (3, 5)
    assert c.index.lookup("a").reserved_slots_by_layer == (1, 1)
    assert c._outstanding_active_reservations() == ((1, 1), 0)
    assert r._resident_seq_ids == {1}
    assert c.index.lookup("b").state is ChainState.IDLE
    assert not c.offload.snapshots[1].valid
    r.chain_invalidate("b")
    assert 2 not in c.offload.snapshots
    with pytest.raises(ChainGoneError):
        c.index.lookup("b")


def test_host_pressure_reclaims_only_cpu_copy_when_gpu_survives(cpu_transfers):
    m, c, r = make_runtime(cpu_transfers, rows=2)
    c.offload.capacity_bytes = 256  # exactly one 3+5-token FP16 K/V snapshot
    finish_chain(m, c, r, 1, "a")
    finish_chain(m, c, r, 2, "b")
    assert m.chain_has_residency(1)
    assert c.index.lookup("a").state is ChainState.IDLE
    assert 1 not in c.offload.snapshots
    assert c.offload.used_bytes == 256


def test_host_pressure_tombstones_cpu_only_chain(cpu_transfers):
    m, c, r = make_runtime(cpu_transfers)
    c.offload.capacity_bytes = 256
    finish_chain(m, c, r, 1, "a")
    finish_chain(m, c, r, 2, "b")
    with pytest.raises(ChainGoneError):
        c.index.lookup("a")
    assert c.offload.used_bytes == 256


def test_oversized_snapshot_fails_without_evicting_other_chains(cpu_transfers):
    m, c, r = make_runtime(cpu_transfers, rows=2)
    finish_chain(m, c, r, 1, "a")
    c.offload.capacity_bytes = 255
    with pytest.raises(ChainCapacityError, match="host budget"):
        c.save_finished_chain(c.index.lookup("a"))
    assert c.index.lookup("a").resident_rows == 1
    assert 1 in c.offload.snapshots


def test_restore_capacity_failure_does_not_partially_allocate(cpu_transfers):
    m = make_manager()
    populate(m)
    c = ChainOffloadController(m, 4096)
    c.save(1, m.snapshot_chain_method_state(1))
    c.wait(1)
    m.free_seq(1)
    m._num_free_slots[1] = 0
    before = [list(rows) for rows in m.free_rows]
    with pytest.raises(RuntimeError, match="insufficient"):
        c.restore(1)
    assert not m.chain_has_residency(1)
    assert before == [list(rows) for rows in m.free_rows]
    assert c.snapshots[1].valid


def test_restore_transfer_failure_releases_rows_and_allows_retry(cpu_transfers, monkeypatch):
    m, c, r = make_runtime(cpu_transfers)
    finish_chain(m, c, r, 1, "a")
    finish_chain(m, c, r, 2, "b")
    p = c.plan_admission(chain_id="a", seq_id=1, token_ids=[1, 2, 3, 4, 5, 6])
    transfer = c.offload.transfer
    def fail(*args):
        raise RuntimeError("transfer failed")
    monkeypatch.setattr(c.offload, "transfer", fail)
    with pytest.raises(RuntimeError, match="transfer failed"):
        r.chain_apply_admission(p)
    assert c.index.lookup("a").state is ChainState.IDLE
    assert c.index.lookup("a").resident_rows == 0
    assert not m.chain_has_residency(1)
    assert m._num_free_slots == [16, 16]
    assert c._outstanding_active_reservations() == ((), 0)
    assert c.offload.snapshots[1].valid
    monkeypatch.setattr(c.offload, "transfer", transfer)
    retry = c.plan_admission(chain_id="a", seq_id=1, token_ids=[1, 2, 3, 4, 5, 6])
    r.chain_apply_admission(retry)
    assert m.chain_physical_residency(1) == (3, 5)


def test_warmup_reset_drains_transfers_before_freeing_device(cpu_transfers):
    m, c, r = make_runtime(cpu_transfers)
    finish_chain(m, c, r, 1, "a")
    finish_chain(m, c, r, 2, "b")
    r.reset_after_warmup()
    assert c.offload.used_bytes == 0
    assert not c.offload.snapshots
    assert not c.index.records
    assert m._num_free_slots == [16, 16]
    assert not r._resident_seq_ids


def test_tp_plan_is_independent_of_local_transfer_completion(cpu_transfers):
    ranks = [make_runtime(cpu_transfers) for _ in range(2)]
    for m, c, r in ranks:
        finish_chain(m, c, r, 1, "a")
        finish_chain(m, c, r, 2, "b")
    ranks[0][1].offload.wait(2)
    driver = ranks[0][1]
    p = driver.plan_admission(chain_id="a", seq_id=1, token_ids=[1, 2, 3, 4, 5, 6])
    for m, c, r in ranks:
        c.validate_admission_plan(p, input_token_count=6,
            input_prefix_digest=driver.index.lookup("a").processed_token_digest)
        r.chain_apply_admission(p)
        assert m.chain_physical_residency(1) == (3, 5)
        assert c.index.lookup("b").resident_rows == 0


def test_decode_reclaim_keeps_snapshot_and_restores_exact_payload(cpu_transfers):
    m, c, r = make_runtime(cpu_transfers, rows=2)
    finish_chain(m, c, r, 1, "a")
    finish_chain(m, c, r, 2, "b")
    expected = []
    for layer in range(2):
        row = m.seq_id_to_row[layer][1]
        slots = m.buffer_req_to_token_slots[layer][row, :m.row_seq_lens[layer][row]].long()
        expected.append(tuple(t[slots].clone() for t in m.chain_storage_tensors(layer)))
    r.chain_reclaim_idle("a", 1, True)
    assert c.index.lookup("a").resident_rows == 0
    assert c.offload.snapshots[1].valid
    assert r._resident_seq_ids == {2}
    plan = c.plan_admission(chain_id="a", seq_id=1, token_ids=[1, 2, 3, 4, 5, 6])
    r.chain_apply_admission(plan)
    for layer in range(2):
        row = m.seq_id_to_row[layer][1]
        slots = m.buffer_req_to_token_slots[layer][row, :m.row_seq_lens[layer][row]].long()
        for tensor, reference in zip(m.chain_storage_tensors(layer), expected[layer]):
            torch.testing.assert_close(tensor[slots], reference, rtol=0, atol=0)


def test_chain_offload_rejects_unhandled_recurrent_state(cpu_transfers):
    m = make_manager()
    m.num_layers = 3
    with pytest.raises(ValueError, match="recurrent/linear"):
        ChainOffloadController(m, 4096)


def test_snapshot_reuses_allocation_but_refreshes_all_values(cpu_transfers):
    m = make_manager()
    populate(m)
    c = ChainOffloadController(m, 4096)
    c.save(1, m.snapshot_chain_method_state(1))
    c.wait(1)
    address = c.snapshots[1].kv[0][0].data_ptr()
    c.invalidate(1)
    m.kv_cache.zero_()
    c.save(1, m.snapshot_chain_method_state(1))
    c.wait(1)
    assert c.snapshots[1].kv[0][0].data_ptr() == address
    assert torch.count_nonzero(c.snapshots[1].kv[0][0]) == 0
    assert c.used_bytes == 256


def test_failed_turn_snapshot_cannot_publish_a_stale_logical_history(cpu_transfers):
    m, c, r = make_runtime(cpu_transfers)
    finish_chain(m, c, r, 1, "a")
    record = c.index.lookup("a")
    record.state = ChainState.ACTIVE
    c.offload.capacity_bytes = 255
    with pytest.raises(ChainCapacityError, match="host budget"):
        r.chain_finish("a", 1, record.processed_token_digest, 5)
    with pytest.raises(ChainGoneError):
        c.index.lookup("a")
    assert not m.chain_has_residency(1)
    assert not c.offload.snapshots
    assert not r._resident_seq_ids


def test_host_pressure_discards_obsolete_active_snapshot_first(cpu_transfers):
    m, c, r = make_runtime(cpu_transfers, rows=2)
    finish_chain(m, c, r, 1, "a")
    finish_chain(m, c, r, 2, "b")
    c.offload.capacity_bytes = 512
    p = c.plan_admission(chain_id="b", seq_id=2, token_ids=[1, 2, 3, 4, 5, 6])
    r.chain_apply_admission(p)
    # A is older and CPU-only. B's obsolete allocation must not evict A.
    c.offload.wait(1)
    m.free_seq(1)
    c.index.lookup("a").resident_rows = 0
    r._resident_seq_ids.discard(1)
    finish_chain(m, c, r, 3, "c")
    assert c.index.lookup("a").resident_rows == 0
    assert 1 in c.offload.snapshots and 2 not in c.offload.snapshots
