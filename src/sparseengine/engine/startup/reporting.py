from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sparseengine.utils.log import logger

from .capacity import KVCapacityPlan, StartupMemoryProfile
from .memory import DeviceMemorySnapshot


@dataclass(frozen=True)
class RankStartupMemoryPlan:
    world_rank: int
    profile: StartupMemoryProfile
    capacity: KVCapacityPlan


@dataclass(frozen=True)
class StartupCapacityDecision:
    rank_plans: tuple[RankStartupMemoryPlan, ...]
    selected_kv_budget_bytes: int
    limiting_rank: int


def _records_by_rank(records: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for record in records:
        rank = int(record["world_rank"])
        if rank in indexed:
            raise RuntimeError(f"Duplicate startup {label} record for rank {rank}.")
        indexed[rank] = record
    return indexed


def build_startup_capacity_decision(
    *,
    prefill_records: list[dict[str, Any]],
    graph_records: list[dict[str, Any]],
    decode_records: list[dict[str, Any]],
    persistent_records: list[dict[str, Any]],
    gpu_memory_utilization: float,
) -> StartupCapacityDecision:
    phases = {
        "prefill": _records_by_rank(prefill_records, "prefill"),
        "graph": _records_by_rank(graph_records, "graph"),
        "decode": _records_by_rank(decode_records, "decode"),
        "persistent": _records_by_rank(persistent_records, "persistent"),
    }
    rank_sets = {label: set(records) for label, records in phases.items()}
    expected_ranks = rank_sets["persistent"]
    mismatched = {
        label: sorted(ranks)
        for label, ranks in rank_sets.items()
        if ranks != expected_ranks
    }
    if not expected_ranks or mismatched:
        raise RuntimeError(
            "Startup memory profiling records do not cover the same ranks: "
            f"expected={sorted(expected_ranks)} observed={mismatched}."
        )

    rank_plans = []
    for rank in sorted(expected_ranks):
        snapshot: DeviceMemorySnapshot = phases["persistent"][rank]["snapshot"]
        pre_graph_release: DeviceMemorySnapshot = phases["persistent"][rank][
            "pre_graph_release_snapshot"
        ]
        post_graph_release: DeviceMemorySnapshot = phases["persistent"][rank][
            "post_graph_release_snapshot"
        ]
        graph_bytes = max(
            0,
            int(post_graph_release.free_bytes - pre_graph_release.free_bytes),
        )
        build = phases["persistent"][rank]["runtime_build"]
        profile_persistent_growth = (
            int(phases["prefill"][rank]["measurement"].consumed_bytes)
            + int(phases["decode"][rank]["measurement"].consumed_bytes)
            + max(
                0,
                int(phases["graph"][rank]["measurement"].consumed_bytes)
                - graph_bytes,
            )
        )
        temporary_runtime_bytes = max(
            0,
            int(snapshot.free_bytes) - int(post_graph_release.free_bytes),
        )
        runtime_persistent_bytes = max(
            0,
            temporary_runtime_bytes
            - min(
                int(build.manager_consumed_bytes),
                int(phases["persistent"][rank]["profiling_kv_budget_bytes"]),
            ),
        )
        profile = StartupMemoryProfile(
            total_bytes=int(snapshot.total_bytes),
            persistent_bytes=int(snapshot.total_bytes - snapshot.free_bytes),
            runtime_persistent_bytes=runtime_persistent_bytes,
            profile_persistent_growth_bytes=profile_persistent_growth,
            prefill_transient_bytes=int(
                phases["prefill"][rank]["measurement"].transient_peak_bytes
            ),
            decode_transient_bytes=int(
                phases["decode"][rank]["measurement"].transient_peak_bytes
            ),
            cuda_graph_bytes=graph_bytes,
        )
        rank_plans.append(
            RankStartupMemoryPlan(
                world_rank=rank,
                profile=profile,
                capacity=KVCapacityPlan.from_profile(
                    profile,
                    gpu_memory_utilization,
                ),
            )
        )

    limiting = min(
        rank_plans,
        key=lambda plan: plan.capacity.local_kv_budget_bytes,
    )
    return StartupCapacityDecision(
        rank_plans=tuple(rank_plans),
        selected_kv_budget_bytes=int(limiting.capacity.local_kv_budget_bytes),
        limiting_rank=int(limiting.world_rank),
    )


def _gib(value: int) -> float:
    return int(value) / (1024**3)


def log_startup_capacity_decision(decision: StartupCapacityDecision) -> None:
    logger.info("Startup memory profile:")
    for rank_plan in decision.rank_plans:
        profile = rank_plan.profile
        capacity = rank_plan.capacity
        logger.info(
            "  rank={}: total={:.2f} GiB target={:.2f} GiB "
            "persistent={:.2f} GiB prefill_peak={:.2f} GiB "
            "decode_peak={:.2f} GiB runtime_peak={:.2f} GiB "
            "runtime_rebuild_reserve={:.2f} GiB "
            "observed_profile_growth={:.2f} GiB "
            "cuda_graph={:.2f} GiB safety_headroom={:.2f} GiB "
            "local_kv_budget={:.2f} GiB",
            rank_plan.world_rank,
            _gib(profile.total_bytes),
            _gib(capacity.target_bytes),
            _gib(profile.persistent_bytes),
            _gib(profile.prefill_transient_bytes),
            _gib(profile.decode_transient_bytes),
            _gib(profile.runtime_transient_bytes),
            _gib(profile.runtime_persistent_bytes),
            _gib(profile.profile_persistent_growth_bytes),
            _gib(profile.cuda_graph_bytes),
            _gib(capacity.safety_headroom_bytes),
            _gib(capacity.local_kv_budget_bytes),
        )
    logger.info(
        "Startup KV decision: selected_budget={:.2f} GiB limiting_rank={}.",
        _gib(decision.selected_kv_budget_bytes),
        decision.limiting_rank,
    )


def log_startup_completion(
    production_records: list[dict[str, Any]],
    final_records: list[dict[str, Any]],
    decision: StartupCapacityDecision,
) -> None:
    production = _records_by_rank(production_records, "production")
    final = _records_by_rank(final_records, "final")
    if set(production) != set(final):
        raise RuntimeError(
            "Production and final startup records cover different ranks: "
            f"production={sorted(production)} final={sorted(final)}."
        )
    slot_count = validate_production_kv_records(production_records)
    plans = {plan.world_rank: plan for plan in decision.rank_plans}
    for rank in sorted(final):
        snapshot: DeviceMemorySnapshot = final[rank]["snapshot"]
        rank_plan = plans.get(rank)
        if rank_plan is None:
            raise RuntimeError(f"Missing startup capacity plan for rank={rank}.")
        used_bytes = int(snapshot.total_bytes - snapshot.free_bytes)
        projected_peak_bytes = (
            used_bytes + int(rank_plan.profile.runtime_transient_bytes)
        )
        target_bytes = int(rank_plan.capacity.target_bytes)
        target_overage_bytes = max(0, projected_peak_bytes - target_bytes)
        physical_headroom_bytes = int(snapshot.total_bytes) - projected_peak_bytes
        if physical_headroom_bytes < 0:
            raise RuntimeError(
                "Production runtime exceeds physical GPU memory capacity: "
                f"rank={rank} persistent_used={used_bytes} "
                f"runtime_transient={rank_plan.profile.runtime_transient_bytes} "
                f"projected_peak={projected_peak_bytes} target={target_bytes} "
                f"physical_capacity={snapshot.total_bytes} "
                f"physical_overage={-physical_headroom_bytes}."
            )
        log = logger.warning if target_overage_bytes > 0 else logger.info
        log(
            "Startup final state: rank={} kv_slots={} free={:.2f} GiB "
            "used={:.2f} GiB projected_peak={:.2f} GiB "
            "target={:.2f} GiB target_overage={:.2f} MiB "
            "physical_headroom={:.2f} GiB post_capture_warmup=passed.",
            rank,
            slot_count,
            _gib(snapshot.free_bytes),
            _gib(used_bytes),
            _gib(projected_peak_bytes),
            _gib(target_bytes),
            target_overage_bytes / (1024**2),
            _gib(physical_headroom_bytes),
        )


def validate_production_kv_records(
    production_records: list[dict[str, Any]],
) -> int | tuple[int, ...]:
    production = _records_by_rank(production_records, "production")
    slot_counts = {
        (
            tuple(int(value) for value in record["num_kvcache_slots"])
            if isinstance(record["num_kvcache_slots"], (list, tuple))
            else int(record["num_kvcache_slots"])
        )
        for record in production.values()
    }
    if len(slot_counts) != 1:
        raise RuntimeError(
            "Production KV capacity differs across ranks: "
            f"records={production_records!r}."
        )
    return next(iter(slot_counts))


__all__ = [
    "RankStartupMemoryPlan",
    "StartupCapacityDecision",
    "build_startup_capacity_decision",
    "log_startup_capacity_decision",
    "log_startup_completion",
    "validate_production_kv_records",
]
