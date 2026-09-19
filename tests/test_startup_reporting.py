import pytest

from sparseengine.engine.startup import (
    CacheRuntimeBuildMeasurement,
    DeviceMemorySnapshot,
    MemoryProfileMeasurement,
    build_startup_capacity_decision,
    log_startup_completion,
    validate_production_kv_records,
)


def _measurement(*, consumed: int = 0, transient: int = 0):
    return MemoryProfileMeasurement(
        consumed_bytes=consumed,
        transient_peak_bytes=transient,
    )


def test_capacity_decision_uses_each_rank_profile_and_global_minimum_budget():
    persistent = [
        {
            "world_rank": 0,
            "snapshot": DeviceMemorySnapshot(700, 1000, 300, 300),
            "pre_graph_release_snapshot": DeviceMemorySnapshot(600, 1000, 400, 400),
            "post_graph_release_snapshot": DeviceMemorySnapshot(650, 1000, 350, 350),
            "profiling_kv_budget_bytes": 50,
            "runtime_build": CacheRuntimeBuildMeasurement(50, 0, 0),
        },
        {
            "world_rank": 1,
            "snapshot": DeviceMemorySnapshot(650, 1000, 350, 350),
            "pre_graph_release_snapshot": DeviceMemorySnapshot(580, 1000, 420, 420),
            "post_graph_release_snapshot": DeviceMemorySnapshot(640, 1000, 360, 360),
            "profiling_kv_budget_bytes": 10,
            "runtime_build": CacheRuntimeBuildMeasurement(10, 0, 0),
        },
    ]
    prefill = [
        {"world_rank": 1, "measurement": _measurement(transient=80)},
        {"world_rank": 0, "measurement": _measurement(transient=160)},
    ]
    graph = [
        {
            "world_rank": 0,
            "after": DeviceMemorySnapshot(600, 1000, 400, 400),
            "measurement": _measurement(consumed=50),
        },
        {
            "world_rank": 1,
            "after": DeviceMemorySnapshot(580, 1000, 420, 420),
            "measurement": _measurement(consumed=60),
        },
    ]
    decode = [
        {"world_rank": 0, "measurement": _measurement(transient=70)},
        {"world_rank": 1, "measurement": _measurement(transient=120)},
    ]

    decision = build_startup_capacity_decision(
        prefill_records=prefill,
        graph_records=graph,
        decode_records=decode,
        persistent_records=persistent,
        gpu_memory_utilization=0.9,
    )

    assert [plan.profile.prefill_transient_bytes for plan in decision.rank_plans] == [160, 80]
    assert [plan.capacity.local_kv_budget_bytes for plan in decision.rank_plans] == [390, 370]
    assert decision.selected_kv_budget_bytes == 370
    assert decision.limiting_rank == 1


def test_capacity_decision_reserves_runtime_build_and_profile_persistent_growth():
    persistent = [{
        "world_rank": 0,
        "snapshot": DeviceMemorySnapshot(700, 1000, 300, 300),
        "pre_graph_release_snapshot": DeviceMemorySnapshot(430, 1000, 570, 570),
        "post_graph_release_snapshot": DeviceMemorySnapshot(480, 1000, 520, 520),
        "profiling_kv_budget_bytes": 200,
        "runtime_build": CacheRuntimeBuildMeasurement(200, 0, 20),
    }]
    prefill = [{"world_rank": 0, "measurement": _measurement(consumed=10, transient=100)}]
    graph = [{
        "world_rank": 0,
        "after": DeviceMemorySnapshot(450, 1000, 550, 550),
        "measurement": _measurement(consumed=70, transient=20),
    }]
    decode = [{"world_rank": 0, "measurement": _measurement(consumed=15, transient=80)}]

    decision = build_startup_capacity_decision(
        prefill_records=prefill,
        graph_records=graph,
        decode_records=decode,
        persistent_records=persistent,
        gpu_memory_utilization=0.9,
    )

    profile = decision.rank_plans[0].profile
    assert profile.cuda_graph_bytes == 50
    assert profile.profile_persistent_growth_bytes == 45
    assert profile.runtime_persistent_bytes == 20
    assert decision.selected_kv_budget_bytes == 430


def test_capacity_decision_does_not_double_count_runtime_state_that_survives_release():
    decision = build_startup_capacity_decision(
        prefill_records=[{
            "world_rank": 0,
            "measurement": _measurement(transient=100),
        }],
        graph_records=[{
            "world_rank": 0,
            "after": DeviceMemorySnapshot(430, 1000, 570, 570),
            "measurement": _measurement(consumed=50),
        }],
        decode_records=[{
            "world_rank": 0,
            "measurement": _measurement(transient=80),
        }],
        persistent_records=[{
            "world_rank": 0,
            "snapshot": DeviceMemorySnapshot(680, 1000, 320, 320),
            "pre_graph_release_snapshot": DeviceMemorySnapshot(430, 1000, 570, 570),
            "post_graph_release_snapshot": DeviceMemorySnapshot(480, 1000, 520, 520),
            "profiling_kv_budget_bytes": 200,
            "runtime_build": CacheRuntimeBuildMeasurement(200, 0, 20),
        }],
        gpu_memory_utilization=0.9,
    )

    assert decision.rank_plans[0].profile.runtime_persistent_bytes == 0
    assert decision.selected_kv_budget_bytes == 430


def test_production_kv_records_must_match_across_ranks():
    records = [
        {"world_rank": 0, "num_kvcache_slots": 10},
        {"world_rank": 1, "num_kvcache_slots": 9},
    ]

    try:
        validate_production_kv_records(records)
    except RuntimeError as exc:
        assert "differs across ranks" in str(exc)
    else:  # pragma: no cover - contract failure path.
        raise AssertionError("mismatched production capacities were accepted")


def test_layer_varying_production_capacities_compare_structurally():
    records = [
        {"world_rank": 0, "num_kvcache_slots": [10, 8]},
        {"world_rank": 1, "num_kvcache_slots": [10, 8]},
    ]

    assert validate_production_kv_records(records) == (10, 8)


def test_final_startup_state_rejects_projected_peak_above_physical_capacity():
    persistent = [{
        "world_rank": 0,
        "snapshot": DeviceMemorySnapshot(700, 1000, 300, 300),
        "pre_graph_release_snapshot": DeviceMemorySnapshot(600, 1000, 400, 400),
        "post_graph_release_snapshot": DeviceMemorySnapshot(650, 1000, 350, 350),
        "profiling_kv_budget_bytes": 50,
        "runtime_build": CacheRuntimeBuildMeasurement(50, 0, 0),
    }]
    prefill = [{"world_rank": 0, "measurement": _measurement(transient=100)}]
    graph = [{
        "world_rank": 0,
        "after": DeviceMemorySnapshot(600, 1000, 400, 400),
        "measurement": _measurement(consumed=50),
    }]
    decode = [{"world_rank": 0, "measurement": _measurement(transient=80)}]
    decision = build_startup_capacity_decision(
        prefill_records=prefill,
        graph_records=graph,
        decode_records=decode,
        persistent_records=persistent,
        gpu_memory_utilization=0.9,
    )

    with pytest.raises(RuntimeError, match="exceeds physical GPU memory capacity"):
        log_startup_completion(
            [{"world_rank": 0, "num_kvcache_slots": 10}],
            [{
                "world_rank": 0,
                "snapshot": DeviceMemorySnapshot(0, 1000, 1000, 1000),
            }],
            decision,
        )


def test_final_startup_state_may_use_memory_above_utilization_target():
    persistent = [{
        "world_rank": 0,
        "snapshot": DeviceMemorySnapshot(700, 1000, 300, 300),
        "pre_graph_release_snapshot": DeviceMemorySnapshot(600, 1000, 400, 400),
        "post_graph_release_snapshot": DeviceMemorySnapshot(650, 1000, 350, 350),
        "profiling_kv_budget_bytes": 50,
        "runtime_build": CacheRuntimeBuildMeasurement(50, 0, 0),
    }]
    decision = build_startup_capacity_decision(
        prefill_records=[{
            "world_rank": 0,
            "measurement": _measurement(transient=100),
        }],
        graph_records=[{
            "world_rank": 0,
            "after": DeviceMemorySnapshot(600, 1000, 400, 400),
            "measurement": _measurement(consumed=50),
        }],
        decode_records=[{
            "world_rank": 0,
            "measurement": _measurement(transient=80),
        }],
        persistent_records=persistent,
        gpu_memory_utilization=0.9,
    )

    log_startup_completion(
        [{"world_rank": 0, "num_kvcache_slots": 10}],
        [{
            "world_rank": 0,
            "snapshot": DeviceMemorySnapshot(100, 1000, 900, 900),
        }],
        decision,
    )
