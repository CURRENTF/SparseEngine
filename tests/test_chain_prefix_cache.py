from __future__ import annotations

import pickle
import json
from types import SimpleNamespace

import pytest

from sparseengine.configs.sparse import _normalize_h2o
from sparseengine.engine.cache_manager.methods.h2o import H2OCacheManager
from sparseengine.engine.cache_manager.methods.snapkv import SnapKVCacheManager
from sparseengine.engine.chain_cache import (
    ChainAdmissionPlan,
    ChainBusyError,
    ChainCacheIndex,
    ChainCapacityError,
    ChainFingerprintMismatchError,
    ChainGoneError,
    ChainNotFoundError,
    ChainOwnerMismatchError,
    ChainPrefixMismatchError,
    ChainState,
    build_chain_cache_fingerprint,
    normalize_prefix_cache_mode,
    stable_token_digest,
)
from sparseengine.engine.chain_cache import ChainCacheCoordinator
from sparseengine.engine.llm_engine import LLMEngine
from sparseengine.engine.runtime_state import RuntimeState
from sparseengine.engine.sequence import Sequence
from sparseengine.sampling_params import SamplingParams


FINGERPRINT = b"chain-test-fingerprint"


@pytest.mark.parametrize("pressure", ["kv_slots", "resident_rows", "rejected", "demoted"])
def test_chain_admission_logs_reservations_and_preserves_failed_state(pressure):
    """A positive physical free count must not hide reservation-driven eviction."""
    from sparseengine.utils.log import logger

    config = _h2o_fingerprint_config(
        engine_prefill_chunk_size=4, max_num_seqs_in_gpu=3,
    )
    manager = object.__new__(H2OCacheManager)
    manager.config = config
    manager.kv_transformer_layer_indices = lambda: [0]
    manager._num_free_slots = [100 if pressure == "resident_rows" else 16]
    manager.free_rows = [[] if pressure == "resident_rows" else [0, 1]]
    manager.chain_has_residency = lambda seq_id: False
    coordinator = ChainCacheCoordinator(config, manager)
    coordinator.decode_reservations = SimpleNamespace(
        outstanding=lambda: {"layer_0": 0 if pressure == "resident_rows" else 4},
    )
    if pressure != "resident_rows":
        active = coordinator.index.plan_admission(
            chain_id="active", seq_id=1, token_ids=[1],
            fingerprint=coordinator.fingerprint, reserved_slots_by_layer=(8,),
            reserved_rows=1,
        )
        coordinator.index.apply_admission(active, fingerprint=coordinator.fingerprint)
    if pressure != "rejected":
        idle = coordinator.index.plan_admission(
            chain_id="idle", seq_id=2, token_ids=[2], fingerprint=coordinator.fingerprint,
        )
        coordinator.index.apply_admission(idle, fingerprint=coordinator.fingerprint)
        coordinator.index.finish("idle", token_ids=[2], processed_token_count=1,
                                 physical_slots_by_layer=(8,))
    if pressure == "demoted":
        coordinator.offload = SimpleNamespace(poll=lambda: None, snapshots={2: object()})
    before = pickle.dumps(coordinator.index)
    messages = []
    sink = logger.add(lambda msg: messages.append(msg.record["message"]), level="DEBUG")
    try:
        if pressure == "rejected":
            with pytest.raises(ChainCapacityError):
                coordinator.plan_admission(chain_id="incoming", seq_id=3, token_ids=list(range(20)))
        else:
            plan = coordinator.plan_admission(chain_id="incoming", seq_id=3, token_ids=list(range(20)))
            assert plan.victim_chain_ids == (() if pressure == "demoted" else ("idle",))
    finally:
        logger.remove(sink)
    # Planning/logging cannot free or mutate the chains, even on rejection.
    assert pickle.dumps(coordinator.index) == before
    events = [json.loads(m.removeprefix("chain_admission ")) for m in messages
              if m.startswith("chain_admission ")]
    assert len(events) == 1
    event = events[0]
    assert event["required_slots_by_layer"] == [12]  # retain 8 then append chunk of 4
    if pressure == "resident_rows":
        assert event["pressure"] == ["resident_rows"]
        assert event["row_deficit"] == 1
        assert event["slot_deficits_by_layer"] == [0]
    else:
        assert event["cache_free_slot_budgets"] == {"layer_0": 16}
        assert event["chain_reserved_slots_by_layer"] == [8]
        assert event["decode_reserved_slots_by_layer"] == [4]
        assert event["outstanding_reserved_slots_by_layer"] == [12]
        assert event["slot_deficits_by_layer"] == [8]  # 12 needed - (16 free - 12 reserved)
        assert event["row_deficit"] == 0
        assert event["pressure"] == ["kv_slots"]
    assert event["outcome"] == ("rejected" if pressure == "rejected" else "planned")
    if pressure != "rejected":
        assert event["victim_chain_ids"] == ([] if pressure == "demoted" else ["idle"])
        assert event["demote_chain_ids"] == (["idle"] if pressure == "demoted" else [])


def test_score_free_h2o_accepts_non_kernel_aligned_decode_budget():
    config = SimpleNamespace(
        h2o_decode_budget=16,
        h2o_decode_eviction_interval=1,
        h2o_prefill_budget=32,
        h2o_recent_ratio=0.5,
        h2o_prefill_score_window=0,
        sparse_prefill_score_mode="probability",
    )

    _normalize_h2o(config)

    assert config.h2o_decode_budget == 16
    assert config.h2o_decode_eviction_interval == 1


def _create_and_finish(
    index: ChainCacheIndex,
    chain_id: str,
    seq_id: int,
    token_ids: list[int],
    *,
    physical_slots: tuple[int, ...] = (3, 3),
):
    plan = index.plan_admission(
        chain_id=chain_id,
        seq_id=seq_id,
        token_ids=token_ids,
        fingerprint=FINGERPRINT,
    )
    index.apply_admission(plan, fingerprint=FINGERPRINT)
    return index.finish(
        chain_id,
        token_ids=token_ids,
        processed_token_count=len(token_ids),
        physical_slots_by_layer=physical_slots,
    )


def test_chain_create_finish_resume_and_processed_digest():
    index = ChainCacheIndex()
    created = index.plan_admission(
        chain_id="chain-a",
        seq_id=7,
        token_ids=[1, 2, 3],
        fingerprint=FINGERPRINT,
    )
    assert created.status == "created"
    assert created.reused_tokens == 0
    record = index.apply_admission(created, fingerprint=FINGERPRINT)
    assert record.state is ChainState.ACTIVE

    with pytest.raises(ChainBusyError):
        index.plan_admission(
            chain_id="chain-a",
            seq_id=7,
            token_ids=[1, 2, 3, 4],
            fingerprint=FINGERPRINT,
        )

    index.finish(
        "chain-a",
        token_ids=[1, 2, 3],
        processed_token_count=3,
        physical_slots_by_layer=(2, 3),
    )
    resumed = index.plan_admission(
        chain_id="chain-a",
        seq_id=7,
        token_ids=[1, 2, 3, 4, 5],
        fingerprint=FINGERPRINT,
    )
    assert resumed.status == "resumed"
    assert resumed.reused_tokens == 3
    assert resumed.seq_id == 7


def test_chain_keeps_full_tokens_independent_of_sparse_physical_slots():
    index = ChainCacheIndex()
    plan = index.plan_admission(
        chain_id="chain-a",
        seq_id=7,
        token_ids=[10, 11],
        fingerprint=FINGERPRINT,
    )
    index.apply_admission(plan, fingerprint=FINGERPRINT)
    record = index.finish(
        "chain-a",
        token_ids=[10, 11, 20, 21, 99],
        processed_token_count=4,
        physical_slots_by_layer=(2, 3),
    )

    assert list(record.token_ids) == [10, 11, 20, 21, 99]
    assert record.processed_token_count == 4
    assert record.physical_slots_by_layer == (2, 3)

    resumed = index.plan_admission(
        chain_id="chain-a",
        seq_id=7,
        token_ids=[10, 11, 20, 21, 99, 30],
        fingerprint=FINGERPRINT,
    )
    assert resumed.status == "resumed"
    assert resumed.reused_tokens == 4


def test_discard_chain_targets_expected_resident_sequence():
    class Scheduler:
        def __init__(self):
            self.aborted = []

        def abort(self, seq_id):
            self.aborted.append(int(seq_id))

    class ModelRunner:
        def __init__(self, coordinator):
            self.runtime_state = SimpleNamespace(
                chain_cache_coordinator=coordinator,
            )
            self.calls = []

        def call(self, method, *args):
            self.calls.append((method, args))

    record = SimpleNamespace(seq_id=7)
    coordinator = SimpleNamespace(
        index=SimpleNamespace(records={"chain-a": record}),
    )
    engine = object.__new__(LLMEngine)
    engine.scheduler = Scheduler()
    engine.model_runner = ModelRunner(coordinator)
    engine._active_chain_sequences = {7: SimpleNamespace()}

    assert engine.discard_chain("missing", expected_seq_id=8) is False
    with pytest.raises(ChainOwnerMismatchError):
        engine.discard_chain("chain-a", expected_seq_id=8)
    assert engine.discard_chain("chain-a", expected_seq_id=7) is True
    assert engine.scheduler.aborted == [7]
    assert engine._active_chain_sequences == {}
    assert engine.model_runner.calls == [
        ("chain_invalidate", ("chain-a", 7)),
    ]


def test_chain_resume_rejects_digest_and_fingerprint_mismatch():
    index = ChainCacheIndex()
    _create_and_finish(index, "chain-a", 1, [10, 20, 30])

    with pytest.raises(ChainPrefixMismatchError):
        index.plan_admission(
            chain_id="chain-a",
            seq_id=1,
            token_ids=[10, 99, 30, 40],
            fingerprint=FINGERPRINT,
        )
    with pytest.raises(ChainPrefixMismatchError) as shorter:
        index.plan_admission(
            chain_id="chain-a",
            seq_id=1,
            token_ids=[10, 20],
            fingerprint=FINGERPRINT,
        )
    assert shorter.value.chain_id == "chain-a"
    with pytest.raises(ChainFingerprintMismatchError):
        index.plan_admission(
            chain_id="chain-a",
            seq_id=1,
            token_ids=[10, 20, 30, 40],
            fingerprint=b"different",
        )


def test_chain_compact_validation_matches_full_token_validation():
    index = ChainCacheIndex()
    _create_and_finish(index, "chain-a", 1, [10, 20, 30])

    compact = index.plan_admission_digest(
        chain_id="chain-a",
        seq_id=1,
        input_token_count=5,
        input_prefix_digest=stable_token_digest([10, 20, 30]),
        fingerprint=FINGERPRINT,
    )

    assert compact.status == "resumed"
    assert compact.reused_tokens == 3
    with pytest.raises(ChainPrefixMismatchError):
        index.plan_admission_digest(
            chain_id="chain-a",
            seq_id=1,
            input_token_count=2,
            input_prefix_digest=stable_token_digest([10, 20]),
            fingerprint=FINGERPRINT,
        )
    with pytest.raises(ChainPrefixMismatchError):
        index.plan_admission_digest(
            chain_id="chain-a",
            seq_id=1,
            input_token_count=4,
            input_prefix_digest=stable_token_digest([10, 99, 30]),
            fingerprint=FINGERPRINT,
        )


def test_chain_unknown_and_tombstone_are_distinct():
    index = ChainCacheIndex()
    with pytest.raises(ChainNotFoundError):
        index.lookup("unknown")
    _create_and_finish(index, "chain-a", 1, [1])
    index.evict("chain-a")
    with pytest.raises(ChainGoneError):
        index.lookup("chain-a")


def test_chain_lru_is_strict_idle_only_and_accumulates_layer_deficits():
    index = ChainCacheIndex()
    old = _create_and_finish(
        index, "old", 1, [1], physical_slots=(2, 1)
    )
    newer = _create_and_finish(
        index, "newer", 2, [2], physical_slots=(1, 4)
    )
    active_plan = index.plan_admission(
        chain_id="active",
        seq_id=3,
        token_ids=[3],
        fingerprint=FINGERPRINT,
    )
    index.apply_admission(active_plan, fingerprint=FINGERPRINT)
    assert old.last_access < newer.last_access

    plan = index.plan_admission(
        chain_id="incoming",
        seq_id=4,
        token_ids=[4],
        fingerprint=FINGERPRINT,
        required_slots_by_layer=(3, 4),
        row_deficit=2,
    )
    assert plan.victim_chain_ids == ("old", "newer")
    assert "active" not in plan.victim_chain_ids


def test_chain_lru_tie_break_is_chain_id_and_active_capacity_fails():
    index = ChainCacheIndex()
    first = _create_and_finish(index, "b", 1, [1], physical_slots=(1,))
    second = _create_and_finish(index, "a", 2, [2], physical_slots=(1,))
    first.last_access = second.last_access = 9
    plan = index.plan_admission(
        chain_id="incoming",
        seq_id=3,
        token_ids=[3],
        fingerprint=FINGERPRINT,
        required_slots_by_layer=(1,),
    )
    assert plan.victim_chain_ids == ("a",)

    index.apply_admission(plan, fingerprint=FINGERPRINT)
    for chain_id in list(index.records):
        record = index.records[chain_id]
        index._set_record_state(record, ChainState.ACTIVE)
    with pytest.raises(ChainCapacityError):
        index.plan_admission(
            chain_id="another",
            seq_id=4,
            token_ids=[4],
            fingerprint=FINGERPRINT,
            required_slots_by_layer=(1,),
        )


def test_chain_tombstones_are_bounded():
    index = ChainCacheIndex(max_tombstones=2)
    for seq_id, chain_id in enumerate(("a", "b", "c"), start=1):
        _create_and_finish(index, chain_id, seq_id, [seq_id])
        index.evict(chain_id)
    assert tuple(index.tombstones) == ("b", "c")
    with pytest.raises(ChainNotFoundError):
        index.lookup("a")
    with pytest.raises(ChainGoneError):
        index.lookup("b")


def test_chain_token_history_is_compact_bounded_and_reclaimed():
    index = ChainCacheIndex(max_token_history_tokens=4)
    first = _create_and_finish(index, "first", 1, [1, 2, 3])

    assert first.token_ids.typecode == "I"
    assert list(first.token_ids) == [1, 2, 3]
    assert index.stats()["chain_cache_token_history_tokens"] == 3
    assert index.stats()["chain_cache_token_history_capacity"] == 4
    assert (
        index.stats()["chain_cache_token_history_bytes"]
        == 3 * first.token_ids.itemsize
    )
    assert (
        index.stats()["chain_cache_token_history_byte_capacity"]
        == 4 * first.token_ids.itemsize
    )

    second_plan = index.plan_admission(
        chain_id="second",
        seq_id=2,
        token_ids=[4, 5],
        fingerprint=FINGERPRINT,
    )
    index.apply_admission(second_plan, fingerprint=FINGERPRINT)
    with pytest.raises(ChainCapacityError, match="token history"):
        index.finish(
            "second",
            token_ids=[4, 5],
            processed_token_count=2,
            physical_slots_by_layer=(2,),
        )
    assert index.lookup("second").state is ChainState.ACTIVE

    evicted = index.evict("first")
    assert list(evicted.token_ids) == []
    index.finish(
        "second",
        token_ids=[4, 5],
        processed_token_count=2,
        physical_slots_by_layer=(2,),
    )
    assert index.stats()["chain_cache_token_history_tokens"] == 2


def test_chain_apply_plan_rejects_duplicate_resident_seq_owner():
    index = ChainCacheIndex()
    _create_and_finish(index, "chain-a", 1, [1])
    plan = ChainAdmissionPlan(
        chain_id="chain-b",
        seq_id=1,
        status="created",
        reused_tokens=0,
    )
    with pytest.raises(Exception, match="already owns chain"):
        index.apply_admission(plan, fingerprint=FINGERPRINT)


@pytest.mark.parametrize(
    ("method", "requested", "expected"),
    [
        ("", "auto", "radix"),
        ("omnikv", "auto", "radix"),
        ("quest", "radix", "radix"),
        ("streamingllm", "auto", "chain"),
        ("snapkv", "auto", "chain"),
        ("h2o", "auto", "chain"),
        ("pyramidkv", "chain", "chain"),
        ("rkv", "auto", "chain"),
        ("skipkv", "chain", "chain"),
    ],
)
def test_prefix_cache_mode_resolution(method, requested, expected):
    assert (
        normalize_prefix_cache_mode(
            requested, enabled=True, method=method
        )
        == expected
    )


def test_prefix_cache_mode_rejects_incompatible_mode():
    with pytest.raises(ValueError, match="incompatible"):
        normalize_prefix_cache_mode("radix", enabled=True, method="snapkv")
    with pytest.raises(ValueError, match="incompatible"):
        normalize_prefix_cache_mode("radix", enabled=True, method="h2o")
    with pytest.raises(ValueError, match="incompatible"):
        normalize_prefix_cache_mode("chain", enabled=True, method="quest")
    assert (
        normalize_prefix_cache_mode("chain", enabled=False, method="snapkv")
        == "disabled"
    )


def test_sequence_ipc_preserves_chain_admission_fields():
    seq = Sequence(
        [1, 2, 3],
        SamplingParams(max_tokens=2, temperature=0.0),
    )
    seq.current_chunk_size = 1
    seq.chain_id = "chain-a"
    seq.chain_status = "resumed"
    seq.chain_reused_tokens = 2
    restored = pickle.loads(pickle.dumps(seq))
    assert restored.chain_id == "chain-a"
    assert restored.chain_status == "resumed"
    assert restored.chain_reused_tokens == 2


def test_chain_routing_snapshot_is_read_only_state_copy():
    index = ChainCacheIndex()
    _create_and_finish(index, "idle", 1, [1])
    plan = index.plan_admission(
        chain_id="active",
        seq_id=2,
        token_ids=[2],
        fingerprint=FINGERPRINT,
    )
    index.apply_admission(plan, fingerprint=FINGERPRINT)
    snapshot = index.routing_snapshot()
    index.invalidate("active")
    assert snapshot.match("idle")["state"] == "idle"
    assert snapshot.match("active")["state"] == "active"
    assert snapshot.match("unknown") == {
        "enabled": True,
        "present": False,
        "state": None,
        "tombstone": False,
    }


def test_runtime_chain_lru_reclaims_payload_before_reusing_capacity():
    class CacheManager:
        def __init__(self):
            self.freed = []

        def free_seq(self, seq_id):
            self.freed.append(int(seq_id))

    config = SimpleNamespace(
        sparse_method="snapkv",
        model="/models/test",
        hf_config=SimpleNamespace(model_type="test", dtype="float16"),
        tensor_parallel_size=1,
        max_model_len=128,
        prefix_cache_salt="",
        chain_cache_max_tombstones=8,
        full_attention_layers=[],
        sink_keep_tokens=1,
        recent_keep_tokens=1,
        decode_keep_tokens=4,
        snapkv_window_size=2,
        snapkv_num_full_layers=0,
        sparse_attn_score_dtype="float32",
        pool_kernel_size=1,
    )
    cache_manager = CacheManager()
    coordinator = ChainCacheCoordinator(config, cache_manager)
    runtime_state = RuntimeState(
        config,
        cache_manager,
        chain_cache_coordinator=coordinator,
    )
    for seq_id, chain_id in ((1, "old"), (2, "newer")):
        plan = coordinator.index.plan_admission(
            chain_id=chain_id,
            seq_id=seq_id,
            token_ids=[seq_id],
            fingerprint=coordinator.fingerprint,
        )
        coordinator.index.apply_admission(
            plan,
            fingerprint=coordinator.fingerprint,
        )
        coordinator.index.finish(
            chain_id,
            token_ids=[seq_id],
            processed_token_count=1,
            physical_slots_by_layer=(2,),
        )
        runtime_state._resident_seq_ids.add(seq_id)

    incoming = coordinator.index.plan_admission(
        chain_id="incoming",
        seq_id=3,
        token_ids=[3],
        fingerprint=coordinator.fingerprint,
        required_slots_by_layer=(2,),
        row_deficit=1,
    )
    result = runtime_state.chain_apply_admission(incoming)

    assert result["victim_chain_ids"] == ["old"]
    assert cache_manager.freed == [1]
    assert 1 not in runtime_state._resident_seq_ids
    with pytest.raises(ChainGoneError):
        coordinator.index.lookup("old")
    assert coordinator.index.lookup("newer").state is ChainState.IDLE


def test_runtime_warmup_reset_reclaims_chain_payload_before_metadata():
    class CacheManager:
        def __init__(self):
            self.freed = []

        def free_seq(self, seq_id):
            self.freed.append(int(seq_id))

    config = SimpleNamespace(
        sparse_method="snapkv",
        model="/models/test",
        hf_config=SimpleNamespace(model_type="test", dtype="float16"),
        tensor_parallel_size=1,
        max_model_len=128,
        prefix_cache_salt="",
        chain_cache_max_tombstones=8,
        full_attention_layers=[],
        sink_keep_tokens=1,
        recent_keep_tokens=1,
        decode_keep_tokens=4,
        snapkv_window_size=2,
        snapkv_num_full_layers=0,
        sparse_attn_score_dtype="float32",
        pool_kernel_size=1,
    )
    cache_manager = CacheManager()
    coordinator = ChainCacheCoordinator(config, cache_manager)
    runtime_state = RuntimeState(
        config,
        cache_manager,
        chain_cache_coordinator=coordinator,
    )
    for seq_id, chain_id in ((11, "warmup-a"), (12, "warmup-b")):
        plan = coordinator.index.plan_admission(
            chain_id=chain_id,
            seq_id=seq_id,
            token_ids=[seq_id],
            fingerprint=coordinator.fingerprint,
        )
        coordinator.index.apply_admission(
            plan,
            fingerprint=coordinator.fingerprint,
        )
        coordinator.index.finish(
            chain_id,
            token_ids=[seq_id],
            processed_token_count=1,
            physical_slots_by_layer=(1,),
        )
        runtime_state._resident_seq_ids.add(seq_id)

    runtime_state.reset_after_warmup()

    assert cache_manager.freed == [11, 12]
    assert coordinator.index.records == {}
    assert runtime_state._resident_seq_ids == set()


def test_snapkv_resumed_prefill_score_uses_physical_coordinates():
    manager = object.__new__(SnapKVCacheManager)
    manager.config = SimpleNamespace(
        sparse_method="snapkv",
        snapkv_num_full_layers=0,
        snapkv_window_size=8,
        sink_keep_tokens=2,
        decode_keep_tokens=16,
        recent_keep_tokens=4,
    )
    manager.kv_layer_index = lambda layer_idx: int(layer_idx)
    manager.chain_physical_kv_len = lambda layer_idx, seq_id: 30
    seq = Sequence(
        list(range(100)),
        SamplingParams(max_tokens=2, temperature=0.0),
    )
    seq.chain_id = "chain-a"
    seq.chain_status = "resumed"
    seq.chain_reused_tokens = 90
    seq.num_prefilled_tokens = 90
    seq.current_chunk_size = 10

    rows = manager._prefill_score_rows(0, [seq])

    assert rows == [(0, seq, 22, 30)]


def test_snapkv_reset_after_warmup_releases_rank_local_rows_without_scheduler_state():
    manager = object.__new__(SnapKVCacheManager)
    manager.kv_transformer_layer_indices = lambda: [0, 1]
    manager.seq_id_to_row = [{12: 0, 11: 1}, {11: 0, 12: 1}]
    freed = []
    manager.free_seq = lambda seq_id: freed.append(int(seq_id))

    manager.reset_after_warmup()

    assert freed == [11, 12]


def test_snapkv_resumed_prefill_skips_score_below_physical_budget():
    manager = object.__new__(SnapKVCacheManager)
    manager.config = SimpleNamespace(
        sparse_method="snapkv",
        snapkv_num_full_layers=0,
        snapkv_window_size=8,
        sink_keep_tokens=2,
        decode_keep_tokens=16,
        recent_keep_tokens=4,
    )
    manager.kv_layer_index = lambda layer_idx: int(layer_idx)
    manager.chain_physical_kv_len = lambda layer_idx, seq_id: 20
    seq = Sequence(
        list(range(100)),
        SamplingParams(max_tokens=2, temperature=0.0),
    )
    seq.chain_status = "resumed"
    seq.chain_reused_tokens = 90
    seq.num_prefilled_tokens = 90
    seq.current_chunk_size = 10

    assert manager._prefill_score_rows(0, [seq]) == []


def test_snapkv_recompute_replay_scores_the_full_prompt_window():
    manager = object.__new__(SnapKVCacheManager)
    manager.config = SimpleNamespace(
        sparse_method="snapkv",
        snapkv_num_full_layers=0,
        snapkv_window_size=8,
        sink_keep_tokens=2,
        decode_keep_tokens=16,
        recent_keep_tokens=4,
    )
    manager.kv_layer_index = lambda layer_idx: int(layer_idx)
    seq = Sequence(
        list(range(100)),
        SamplingParams(max_tokens=2, temperature=0.0),
    )
    seq.chain_status = "resumed"
    seq.chain_reused_tokens = 96
    seq.append_token(100)
    seq.start_recompute_replay()
    seq.num_prefilled_tokens = 92
    seq.current_chunk_size = 8

    rows = manager._prefill_score_rows(0, [seq])

    assert rows == [(0, seq, 92, 100)]


def test_pyramid_chain_resume_uses_residual_for_raw_offload():
    manager = object.__new__(SnapKVCacheManager)
    manager.config = SimpleNamespace(
        sparse_method="pyramidkv",
        long_prefill_offload_threshold=5,
    )
    manager._pyramidkv_can_use_full_prefill_staging = lambda: True
    manager.pyramidkv_prefill_staging_kv_cache = object()
    seq = Sequence(
        list(range(100)),
        SamplingParams(max_tokens=2, temperature=0.0),
    )
    seq.chain_status = "resumed"
    seq.num_prefilled_tokens = 90
    seq.current_chunk_size = 10

    assert manager.requires_long_prefill_offload(seq) is True


def test_new_pyramid_chain_capacity_uses_materialized_layer_budgets():
    manager = object.__new__(SnapKVCacheManager)
    manager.config = SimpleNamespace(
        sparse_method="pyramidkv",
        sink_keep_tokens=1,
        recent_keep_tokens=1,
    )
    manager.kv_transformer_layer_indices = lambda: [0, 1]
    manager.kv_layer_index = lambda layer_idx: int(layer_idx)
    manager._num_free_slots = [2, 2]
    manager.free_rows = [[0], [0]]
    manager._pyramidkv_can_use_full_prefill_staging = lambda: True
    manager._pyramidkv_layer_budget = lambda layer_idx: (10, 5)[layer_idx]

    required, required_rows, deficits, row_deficit = (
        manager.chain_capacity_deficits(
            suffix_tokens=100,
            generation_tokens=1,
            needs_resident_row=True,
        )
    )

    assert required == (10, 5)
    assert required_rows == 1
    assert deficits == (8, 3)
    assert row_deficit == 0

    _, _, decode_deficits, _ = manager.chain_capacity_deficits(
        suffix_tokens=100,
        generation_tokens=5,
        needs_resident_row=True,
    )
    assert decode_deficits == (12, 6)


def test_resumed_snapkv_capacity_reserves_score_free_decode_growth():
    manager = object.__new__(SnapKVCacheManager)
    manager.config = SimpleNamespace(
        sparse_method="snapkv",
        snapkv_num_full_layers=0,
        sink_keep_tokens=1,
        decode_keep_tokens=8,
        recent_keep_tokens=1,
    )
    manager.kv_transformer_layer_indices = lambda: [0]
    manager.kv_layer_index = lambda layer_idx: int(layer_idx)
    manager._num_free_slots = [3]
    manager.free_rows = [[]]

    required, required_rows, deficits, row_deficit = (
        manager.chain_capacity_deficits(
            suffix_tokens=2,
            generation_tokens=100,
            existing_slots_by_layer=(10,),
            needs_resident_row=False,
        )
    )

    # Final prefill compacts to ten slots, but score-free decode never evicts.
    # generation_tokens includes the already-materialized decode input, so the
    # resident row must reserve 99 additional slots.
    assert required == (99,)
    assert required_rows == 0
    assert deficits == (96,)
    assert row_deficit == 0


def test_resumed_snapkv_without_suffix_reserves_growth_from_existing_row():
    manager = object.__new__(SnapKVCacheManager)
    manager.config = SimpleNamespace(
        sparse_method="snapkv",
        snapkv_num_full_layers=0,
        sink_keep_tokens=1,
        decode_keep_tokens=8,
        recent_keep_tokens=1,
    )
    manager.kv_transformer_layer_indices = lambda: [0]
    manager.kv_layer_index = lambda layer_idx: int(layer_idx)
    manager._num_free_slots = [0]
    manager.free_rows = [[]]

    required, required_rows, deficits, row_deficit = (
        manager.chain_capacity_deficits(
            suffix_tokens=0,
            generation_tokens=4,
            existing_slots_by_layer=(12,),
            needs_resident_row=False,
        )
    )

    assert required == (3,)
    assert required_rows == 0
    assert deficits == (3,)
    assert row_deficit == 0


def test_resumed_h2o_capacity_uses_chunked_physical_peak():
    manager = object.__new__(H2OCacheManager)
    manager.config = SimpleNamespace(
        sparse_method="h2o",
        h2o_decode_budget=4,
        h2o_decode_eviction_interval=3,
        h2o_prefill_budget=8,
        engine_prefill_chunk_size=4,
    )
    manager.kv_transformer_layer_indices = lambda: [0]
    manager._num_free_slots = [3]
    manager.free_rows = [[]]

    required, required_rows, deficits, row_deficit = (
        manager.chain_capacity_deficits(
            suffix_tokens=100,
            generation_tokens=10,
            existing_slots_by_layer=(4,),
            needs_resident_row=False,
        )
    )

    # Intermediate prefill peaks at 12, then score-free decode grows from the
    # four-token final-prefill row to 13. The resident row needs nine more slots.
    assert required == (9,)
    assert required_rows == 0
    assert deficits == (6,)
    assert row_deficit == 0


def test_h2o_flashprefill_capacity_does_not_assume_h2o_prefill_compaction():
    manager = object.__new__(H2OCacheManager)
    manager.config = SimpleNamespace(
        sparse_method="h2o",
        prefill_sparse_method="flashprefill_v2",
        h2o_decode_budget=4,
        h2o_decode_eviction_interval=3,
        h2o_prefill_budget=8,
        engine_prefill_chunk_size=4,
    )
    manager.kv_transformer_layer_indices = lambda: [0]
    manager._num_free_slots = [3]
    manager.free_rows = [[]]

    required, required_rows, deficits, row_deficit = (
        manager.chain_capacity_deficits(
            suffix_tokens=100,
            generation_tokens=10,
            existing_slots_by_layer=(4,),
            needs_resident_row=False,
        )
    )

    # FlashPrefill owns the prefill axis in this combination. H2O still
    # compacts the final prompt for decode, but the full suffix must fit first.
    assert required == (100,)
    assert required_rows == 0
    assert deficits == (97,)
    assert row_deficit == 0


def test_resumed_h2o_capacity_reserves_over_budget_first_prefill_chunk():
    manager = object.__new__(H2OCacheManager)
    manager.config = SimpleNamespace(
        sparse_method="h2o",
        h2o_decode_budget=4,
        h2o_decode_eviction_interval=3,
        h2o_prefill_budget=4,
        engine_prefill_chunk_size=4,
    )
    manager.kv_transformer_layer_indices = lambda: [0]
    manager._num_free_slots = [2]
    manager.free_rows = [[]]

    required, required_rows, deficits, row_deficit = (
        manager.chain_capacity_deficits(
            suffix_tokens=100,
            generation_tokens=0,
            existing_slots_by_layer=(6,),
            needs_resident_row=False,
        )
    )

    assert required == (4,)
    assert required_rows == 0
    assert deficits == (2,)
    assert row_deficit == 0


def test_resumed_h2o_capacity_handles_small_suffix_and_outstanding_by_layer():
    manager = object.__new__(H2OCacheManager)
    manager.config = SimpleNamespace(
        sparse_method="h2o",
        h2o_decode_budget=4,
        h2o_decode_eviction_interval=3,
        h2o_prefill_budget=4,
        engine_prefill_chunk_size=4,
    )
    manager.kv_transformer_layer_indices = lambda: [0, 2]
    manager._num_free_slots = [3, 0, 4]
    manager.free_rows = [[], [], []]

    required, required_rows, deficits, row_deficit = (
        manager.chain_capacity_deficits(
            suffix_tokens=2,
            generation_tokens=0,
            existing_slots_by_layer=(6, 4),
            outstanding_reserved_slots_by_layer=(2, 3),
            needs_resident_row=False,
        )
    )

    assert required == (2, 2)
    assert required_rows == 0
    assert deficits == (1, 1)
    assert row_deficit == 0


def test_resumed_h2o_capacity_without_suffix_starts_decode_from_existing_row():
    manager = object.__new__(H2OCacheManager)
    manager.config = SimpleNamespace(
        sparse_method="h2o",
        h2o_decode_budget=4,
        h2o_decode_eviction_interval=3,
        h2o_prefill_budget=4,
        engine_prefill_chunk_size=4,
    )
    manager.kv_transformer_layer_indices = lambda: [0]
    manager._num_free_slots = [0]
    manager.free_rows = [[]]

    required, required_rows, deficits, row_deficit = (
        manager.chain_capacity_deficits(
            suffix_tokens=0,
            generation_tokens=2,
            existing_slots_by_layer=(6,),
            needs_resident_row=False,
        )
    )

    assert required == (1,)
    assert required_rows == 0
    assert deficits == (1,)
    assert row_deficit == 0


def test_new_h2o_chain_capacity_reserves_row_and_outstanding_slots():
    manager = object.__new__(H2OCacheManager)
    manager.config = SimpleNamespace(
        sparse_method="h2o",
        h2o_decode_budget=4,
        h2o_decode_eviction_interval=3,
        h2o_prefill_budget=8,
        engine_prefill_chunk_size=4,
    )
    manager.kv_transformer_layer_indices = lambda: [0]
    manager._num_free_slots = [16]
    manager.free_rows = [[0, 1]]

    required, required_rows, deficits, row_deficit = (
        manager.chain_capacity_deficits(
            suffix_tokens=100,
            generation_tokens=10,
            outstanding_reserved_slots_by_layer=(8,),
            outstanding_reserved_rows=1,
            needs_resident_row=True,
        )
    )

    assert required == (13,)
    assert required_rows == 1
    assert deficits == (5,)
    assert row_deficit == 0


def test_h2o_chain_capacity_reserves_score_free_decode_growth():
    manager = object.__new__(H2OCacheManager)
    manager.config = SimpleNamespace(
        sparse_method="h2o",
        h2o_decode_budget=4,
        h2o_decode_eviction_interval=3,
        h2o_prefill_budget=8,
        engine_prefill_chunk_size=4,
    )
    manager.kv_transformer_layer_indices = lambda: [0]
    manager._num_free_slots = [1]
    manager.free_rows = [[]]

    required, required_rows, deficits, row_deficit = (
        manager.chain_capacity_deficits(
            suffix_tokens=0,
            generation_tokens=10,
            existing_slots_by_layer=(4,),
            needs_resident_row=False,
        )
    )

    assert required == (9,)
    assert required_rows == 0
    assert deficits == (8,)
    assert row_deficit == 0


def _h2o_fingerprint_config(**overrides):
    values = {
        "sparse_method": "h2o",
        "prefill_sparse_method": "h2o_prefill",
        "model": "/models/test",
        "hf_config": SimpleNamespace(
            model_type="qwen2",
            dtype="float16",
        ),
        "tensor_parallel_size": 1,
        "max_model_len": 128,
        "full_attention_layers": [],
        "prefix_cache_salt": "",
        "h2o_decode_budget": 4,
        "h2o_decode_eviction_interval": 3,
        "h2o_prefill_budget": 8,
        "h2o_recent_ratio": 0.5,
        "h2o_prefill_score_window": 4,
        "sparse_attn_score_dtype": "float32",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("prefill_sparse_method", "flashprefill_v2"),
        ("h2o_decode_budget", 5),
        ("h2o_decode_eviction", True),
        ("h2o_decode_eviction_interval", 2),
        ("h2o_prefill_budget", 9),
        ("h2o_recent_ratio", 0.25),
        ("h2o_prefill_score_window", 8),
        ("sparse_attn_score_dtype", "float16"),
    ],
)
def test_h2o_chain_fingerprint_covers_physical_state_config(
    field_name,
    changed_value,
):
    baseline = build_chain_cache_fingerprint(_h2o_fingerprint_config())
    changed = build_chain_cache_fingerprint(
        _h2o_fingerprint_config(**{field_name: changed_value})
    )
    assert changed != baseline


def test_chain_admission_reserves_capacity_before_prefill_allocation():
    config = SimpleNamespace(
        sparse_method="snapkv",
        model="/models/test",
        hf_config=SimpleNamespace(model_type="test", dtype="float16"),
        tensor_parallel_size=1,
        max_model_len=128,
        prefix_cache_salt="",
        chain_cache_max_tombstones=8,
        full_attention_layers=[],
        snapkv_num_full_layers=0,
        sink_keep_tokens=1,
        decode_keep_tokens=8,
        recent_keep_tokens=1,
    )
    manager = object.__new__(SnapKVCacheManager)
    manager.config = config
    manager.kv_transformer_layer_indices = lambda: [0]
    manager.kv_layer_index = lambda layer_idx: int(layer_idx)
    manager._num_free_slots = [5]
    manager.free_rows = [[0, 1]]
    manager.seq_id_to_row = [{}]
    coordinator = ChainCacheCoordinator(config, manager)

    first = coordinator.plan_admission(
        chain_id="first",
        seq_id=1,
        token_ids=[1, 2, 3, 4],
    )
    coordinator.apply_admission(first)

    assert first.reserved_slots_by_layer == (4,)
    assert first.reserved_rows == 1
    with pytest.raises(ChainCapacityError):
        coordinator.plan_admission(
            chain_id="second",
            seq_id=2,
            token_ids=[5, 6, 7, 8],
        )


def test_engine_chain_admission_reuses_resident_seq_and_logical_boundary():
    class CacheManager:
        def __init__(self):
            self.finished_turns = []
            self.freed_seqs = []

        def free_seq(self, seq_id):
            self.freed_seqs.append(seq_id)

        def chain_capacity_deficits(self, **_kwargs):
            return (), 0, (), 0

        def kv_transformer_layer_indices(self):
            return [0, 1]

        def chain_physical_residency(self, _seq_id):
            return (3, 4)

        def on_chain_turn_finished(self, seq_id, processed_token_count):
            self.finished_turns.append(
                (int(seq_id), int(processed_token_count))
            )

    config = SimpleNamespace(
        resolved_prefix_cache_mode="chain",
        max_model_len=128,
        sparse_method="snapkv",
        model="/models/test",
        hf_config=SimpleNamespace(model_type="test", dtype="float16"),
        tensor_parallel_size=1,
        prefix_cache_salt="",
        chain_cache_max_tombstones=8,
        full_attention_layers=[],
        sink_keep_tokens=1,
        recent_keep_tokens=1,
        decode_keep_tokens=4,
        snapkv_window_size=2,
        snapkv_num_full_layers=0,
        sparse_attn_score_dtype="float32",
        pool_kernel_size=1,
    )
    cache_manager = CacheManager()
    coordinator = ChainCacheCoordinator(config, cache_manager)
    runtime_state = RuntimeState(
        config,
        cache_manager,
        chain_cache_coordinator=coordinator,
    )

    class ModelRunner:
        def __init__(self):
            self.runtime_state = runtime_state
            self.calls = []

        def call(self, method, *args):
            self.calls.append((method, args))
            if method == "chain_invalidate":
                return self.runtime_state.chain_invalidate(
                    args[0],
                    expected_seq_id=args[1],
                )
            return getattr(self.runtime_state, method)(*args)

    class Scheduler:
        def __init__(self):
            self.added = []

        def add(self, seq):
            self.added.append(seq)

        def abort(self, _seq_id):
            return False

    engine = object.__new__(LLMEngine)
    engine.config = config
    engine.model_runner = ModelRunner()
    engine.scheduler = Scheduler()
    engine._active_chain_sequences = {}
    params = SamplingParams(max_tokens=2, temperature=0.0)

    engine._startup_warmup_active = True
    warmup = engine.admit_request([7], params)
    assert warmup.chain_status == "disabled"
    assert not coordinator.index.records
    assert not engine._active_chain_sequences
    engine._startup_warmup_active = False

    first = engine.admit_request([1, 2, 3], params)
    assert first.chain_status == "created"
    validation_args = next(
        args
        for method, args in engine.model_runner.calls
        if method == "chain_validate_admission_plan"
    )
    assert validation_args[1:] == (
        3,
        stable_token_digest([], count=0),
    )
    with pytest.raises(ChainBusyError):
        engine.admit_request([1, 2, 3, 4], params, chain_id=first.chain_id)

    runtime_state.chain_finish(
        first.chain_id,
        first.seq_id,
        stable_token_digest([1, 2, 3, 9], count=3),
        3,
    )
    coordinator.remember_processed_tokens(
        chain_id=first.chain_id,
        seq_id=first.seq_id,
        token_ids=[1, 2, 3, 9],
        processed_token_count=3,
    )
    assert cache_manager.finished_turns == [(first.seq_id, 3)]
    engine._active_chain_sequences.clear()
    resumed = engine.admit_request(
        [5],
        params,
        chain_id=first.chain_id,
        chain_append_only=True,
    )

    assert resumed.seq_id == first.seq_id
    assert resumed.chain_status == "resumed"
    assert resumed.reused_tokens == 3
    assert resumed.prefilled_tokens == 2
    resumed_seq = engine.scheduler.added[-1]
    assert resumed_seq.num_prefilled_tokens == 3
    assert resumed_seq.token_ids[3:] == [9, 5]

    with pytest.raises(ChainNotFoundError):
        engine.admit_request([1, 2], params, chain_id="unknown")

    runtime_state.chain_finish(
        resumed.chain_id,
        resumed.seq_id,
        stable_token_digest([1, 2, 3, 9, 5], count=4),
        4,
    )
    assert cache_manager.finished_turns == [
        (first.seq_id, 3),
        (resumed.seq_id, 4),
    ]
    engine._active_chain_sequences.clear()

    recreated = engine.admit_request(
        [1, 2, 99],
        params,
        chain_id=resumed.chain_id,
    )

    assert recreated.chain_id != resumed.chain_id
    assert recreated.seq_id == resumed.seq_id
    assert recreated.chain_status == "recreated"
    assert recreated.reused_tokens == 0
    assert recreated.prefilled_tokens == 3
    assert cache_manager.freed_seqs == [resumed.seq_id]
    recreated_seq = engine.scheduler.added[-1]
    assert recreated_seq.token_ids == [1, 2, 99]
    assert recreated_seq.num_prefilled_tokens == 0
    with pytest.raises(ChainGoneError):
        coordinator.index.lookup(resumed.chain_id)

    engine.abort_request(recreated.seq_id)
    engine.abort_request(recreated.seq_id)
    assert cache_manager.freed_seqs == [resumed.seq_id, recreated.seq_id]
    with pytest.raises(ChainGoneError):
        coordinator.index.lookup(recreated.chain_id)


def test_engine_abort_rejects_unsafe_chain_retain_disposition():
    engine = object.__new__(LLMEngine)
    engine.scheduler = SimpleNamespace(abort=lambda _seq_id: False)
    engine._active_chain_sequences = {}

    with pytest.raises(ValueError, match="only supports 'invalidate'"):
        engine.abort_request(7, disposition="retain")


def test_chain_rank_validation_protects_existing_decode_window():
    config = _h2o_fingerprint_config(engine_prefill_chunk_size=4)

    def coordinator(free):
        manager = object.__new__(H2OCacheManager)
        manager.config = config
        manager.kv_transformer_layer_indices = lambda: [0]
        manager._num_free_slots = [free]
        manager.free_rows = [[0, 1]]
        manager.seq_id_to_row = [{}]
        return ChainCacheCoordinator(config, manager)

    driver = coordinator(10)
    driver.decode_reservations = SimpleNamespace(outstanding=lambda: {'layer_0': 4})
    plan = driver.plan_admission(chain_id='new', seq_id=10, token_ids=[1, 2, 3, 4])
    assert plan.decode_reserved_slots_by_layer == (4,)
    worker = coordinator(10)
    assert worker.validate_admission_plan(
        plan, input_token_count=4, input_prefix_digest=stable_token_digest([], count=0),
    ) == plan
    # The worker has no scheduler ledger, but must still protect the driver's
    # promise rather than admit into seven free slots with four already owed.
    worker.cache_manager._num_free_slots = [7]
    with pytest.raises(ChainCapacityError):
        worker.validate_admission_plan(
            plan, input_token_count=4, input_prefix_digest=stable_token_digest([], count=0),
        )


@pytest.mark.parametrize("count", [None, 0, 1, 5])
def test_bulk_token_digest_preserves_wire_format(count):
    import hashlib
    import struct

    tokens = [-2**63, -1, 0, 2**32 - 1, 2**63 - 1]
    expected = hashlib.sha256(b"".join(struct.pack("<q", t) for t in tokens[:count])).digest()
    assert stable_token_digest(iter(tokens), count=count) == expected
    with pytest.raises(ChainPrefixMismatchError):
        stable_token_digest(tokens, count=len(tokens) + 1)


def test_prepared_chain_history_is_owned_validated_and_immutable():
    from sparseengine.engine.chain_cache import ChainRecord

    coordinator = object.__new__(ChainCacheCoordinator)
    coordinator.index = ChainCacheIndex(max_token_history_tokens=8)
    record = ChainRecord("prepared", 7, FINGERPRINT, ChainState.ACTIVE)
    coordinator.index.records[record.chain_id] = record
    coordinator.index._set_record_state(record, ChainState.ACTIVE)
    tokens = [1, 2, 3, 9]
    prepared = coordinator.prepare_processed_tokens(
        chain_id=record.chain_id, seq_id=7, token_ids=tokens, processed_token_count=3,
    )
    tokens[0] = 99  # callers cannot mutate the snapshot between prepare and commit
    with pytest.raises(ChainOwnerMismatchError):
        coordinator.remember_prepared_tokens(prepared)
    for invalid in ([1, -1, 3], [1, 2**32, 3], [1, 2]):
        with pytest.raises(ChainOwnerMismatchError):
            coordinator.prepare_processed_tokens(
                chain_id=record.chain_id, seq_id=7, token_ids=invalid, processed_token_count=3,
            )
    assert not record.token_ids and coordinator.index._token_history_tokens == 0
    coordinator.index.finish_digest(
        record.chain_id, processed_token_digest=prepared.processed_token_digest,
        processed_token_count=3, physical_slots_by_layer=(3,),
    )
    coordinator.remember_prepared_tokens(prepared)
    assert list(record.token_ids) == [1, 2, 3, 9]
    assert coordinator.index._token_history_tokens == 4
    # A different owner/boundary must not accept a stale prepared completion.
    record.seq_id = 8
    with pytest.raises(ChainOwnerMismatchError):
        coordinator.remember_prepared_tokens(prepared)
    record.seq_id = 7
    record.processed_token_count = 4
    with pytest.raises(ChainOwnerMismatchError):
        coordinator.remember_prepared_tokens(prepared)
    record.processed_token_count = 3
    # Capacity can change between preparation and completion of the RPC.
    coordinator.index.max_token_history_tokens = 3
    with pytest.raises(ChainCapacityError):
        coordinator.remember_prepared_tokens(prepared)
    assert list(record.token_ids) == [1, 2, 3, 9]
    assert coordinator.index._token_history_tokens == 4
