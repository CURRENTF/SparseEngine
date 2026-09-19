from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


GRADE_ORDER = {"A": 3, "B": 2, "C": 1, "D": 0, "N/A": -1}


@dataclass(frozen=True)
class GateGrade:
    name: str
    grade: str
    status: str
    metrics: dict[str, Any]
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "grade": self.grade,
            "status": self.status,
            "metrics": self.metrics,
            "reason": self.reason,
        }


def _require_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}.")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return resolved


def grade_quality(
    vanilla_score: float,
    sparse_score: float,
    *,
    minimum_vanilla_score: float,
) -> GateGrade:
    vanilla = _require_number(vanilla_score, "vanilla_score")
    sparse = _require_number(sparse_score, "sparse_score")
    minimum_vanilla = _require_number(minimum_vanilla_score, "minimum_vanilla_score")
    if vanilla < minimum_vanilla:
        return GateGrade(
            name="quality",
            grade="D",
            status="failed",
            metrics={
                "vanilla_score": vanilla,
                "sparse_score": sparse,
                "minimum_vanilla_score": minimum_vanilla,
            },
            reason=(
                "Vanilla score is below minimum required baseline: "
                f"vanilla_score={vanilla} minimum_vanilla_score={minimum_vanilla}."
            ),
        )
    score_loss = max(0.0, vanilla - sparse)
    if score_loss < 0.1:
        grade = "A"
    elif score_loss <= 0.5:
        grade = "B"
    elif score_loss <= 1.0:
        grade = "C"
    else:
        grade = "D"
    return GateGrade(
        name="quality",
        grade=grade,
        status="success" if grade != "D" else "failed",
        metrics={
            "vanilla_score": vanilla,
            "sparse_score": sparse,
            "score_loss": score_loss,
            "minimum_vanilla_score": minimum_vanilla,
        },
    )


def grade_ruler_quality(
    vanilla_score: float,
    sparse_score: float,
    *,
    minimum_vanilla_score: float,
    maximum_score_loss: float,
) -> GateGrade:
    vanilla = _require_number(vanilla_score, "vanilla_score")
    sparse = _require_number(sparse_score, "sparse_score")
    minimum_vanilla = _require_number(
        minimum_vanilla_score, "minimum_vanilla_score"
    )
    maximum_loss = _require_number(maximum_score_loss, "maximum_score_loss")
    if maximum_loss <= 0.0:
        raise ValueError(f"maximum_score_loss must be positive, got {maximum_loss}.")
    metrics = {
        "vanilla_score": vanilla,
        "sparse_score": sparse,
        "score_loss": max(0.0, vanilla - sparse),
        "minimum_vanilla_score": minimum_vanilla,
        "maximum_score_loss": maximum_loss,
    }
    if vanilla < minimum_vanilla:
        return GateGrade(
            name="ruler_quality",
            grade="D",
            status="failed",
            metrics=metrics,
            reason=(
                "Vanilla RULER score is below the required baseline floor: "
                f"vanilla_score={vanilla} minimum_vanilla_score={minimum_vanilla}."
            ),
        )
    score_loss = metrics["score_loss"]
    if score_loss == 0.0:
        grade = "A"
    elif score_loss <= maximum_loss / 2.0:
        grade = "B"
    elif score_loss <= maximum_loss:
        grade = "C"
    else:
        grade = "D"
    return GateGrade(
        name="ruler_quality",
        grade=grade,
        status="success" if grade != "D" else "failed",
        metrics=metrics,
        reason=(
            "Sparse RULER score loss exceeds the configured per-context limit."
            if grade == "D"
            else ""
        ),
    )


def grade_longbench_v2_quality(
    vanilla_score: float,
    sparse_score: float,
    *,
    minimum_vanilla_score: float,
    maximum_score_loss: float,
) -> GateGrade:
    vanilla = _require_number(vanilla_score, "vanilla_score")
    sparse = _require_number(sparse_score, "sparse_score")
    minimum_vanilla = _require_number(
        minimum_vanilla_score, "minimum_vanilla_score"
    )
    maximum_loss = _require_number(maximum_score_loss, "maximum_score_loss")
    if maximum_loss <= 0.0:
        raise ValueError(f"maximum_score_loss must be positive, got {maximum_loss}.")
    score_loss = max(0.0, vanilla - sparse)
    metrics = {
        "vanilla_score": vanilla,
        "sparse_score": sparse,
        "score_loss": score_loss,
        "minimum_vanilla_score": minimum_vanilla,
        "maximum_score_loss": maximum_loss,
    }
    if vanilla < minimum_vanilla:
        return GateGrade(
            name="longbench_v2_quality",
            grade="D",
            status="failed",
            metrics=metrics,
            reason=(
                "Vanilla LongBench v2 score is below the required baseline floor: "
                f"vanilla_score={vanilla} minimum_vanilla_score={minimum_vanilla}."
            ),
        )
    if score_loss == 0.0:
        grade = "A"
    elif score_loss <= maximum_loss / 2.0:
        grade = "B"
    elif score_loss <= maximum_loss:
        grade = "C"
    else:
        grade = "D"
    return GateGrade(
        name="longbench_v2_quality",
        grade=grade,
        status="success" if grade != "D" else "failed",
        metrics=metrics,
        reason=(
            "Sparse LongBench v2 score loss exceeds the configured limit."
            if grade == "D"
            else ""
        ),
    )


def grade_perf(
    speedup: float,
    *,
    graph_expected: bool = True,
    graph_active: bool = True,
    require_speedup: bool = True,
    prefill_speedup: float | None = None,
    minimum_prefill_speedup: float | None = None,
) -> GateGrade:
    speedup = _require_number(speedup, "speedup")
    resolved_prefill_speedup = (
        None
        if prefill_speedup is None
        else _require_number(prefill_speedup, "prefill_speedup")
    )
    resolved_minimum_prefill_speedup = (
        None
        if minimum_prefill_speedup is None
        else _require_number(minimum_prefill_speedup, "minimum_prefill_speedup")
    )
    if (
        resolved_minimum_prefill_speedup is not None
        and resolved_minimum_prefill_speedup <= 0.0
    ):
        raise ValueError(
            "minimum_prefill_speedup must be positive, "
            f"got {resolved_minimum_prefill_speedup}."
        )
    metrics = {
        "speedup": speedup,
        "decode_speedup": speedup,
        "prefill_speedup": resolved_prefill_speedup,
        "minimum_prefill_speedup": resolved_minimum_prefill_speedup,
        "graph_expected": graph_expected,
        "graph_active": graph_active,
        "require_speedup": require_speedup,
    }
    if graph_expected and not graph_active:
        return GateGrade(
            "performance",
            "D",
            "failed",
            metrics,
            "decode CUDA graph was expected but not active.",
        )
    if resolved_minimum_prefill_speedup is not None:
        if resolved_prefill_speedup is None:
            return GateGrade(
                "performance",
                "D",
                "failed",
                metrics,
                "prefill speedup is required by the method performance policy.",
            )
        if resolved_prefill_speedup < resolved_minimum_prefill_speedup:
            return GateGrade(
                "performance",
                "D",
                "failed",
                metrics,
                "prefill speedup is below the required minimum.",
            )
    if not require_speedup:
        return GateGrade(
            "performance",
            "A",
            "success",
            metrics,
            "Speedup is recorded but not required by this performance gate.",
        )
    if speedup >= 2.0:
        grade = "A"
    elif speedup >= 1.5:
        grade = "B"
    elif speedup > 1.0:
        grade = "C"
    else:
        grade = "D"
    return GateGrade(
        "performance",
        grade,
        "success" if grade != "D" else "failed",
        metrics,
    )


def grade_memory(*, expected_savings: float | None, observed_savings: float | None) -> GateGrade:
    if expected_savings is None or observed_savings is None:
        return GateGrade(
            "memory",
            "D",
            "failed",
            {"expected_savings": expected_savings, "observed_savings": observed_savings},
            "Memory accounting is incomplete.",
        )
    expected = _require_number(expected_savings, "expected_savings")
    observed = _require_number(observed_savings, "observed_savings")
    error = abs(expected - observed)
    if observed <= 0:
        grade = "D"
    elif error <= 0.05:
        grade = "A"
    elif error <= 0.10:
        grade = "B"
    elif error <= 0.20:
        grade = "C"
    else:
        grade = "D"
    return GateGrade(
        "memory",
        grade,
        "success" if grade != "D" else "failed",
        {"expected_savings": expected, "observed_savings": observed, "abs_error": error},
    )


def grade_stress(
    *,
    completed: bool,
    crashed: bool,
    preemptions: int,
    full_admission_window: bool,
    utilization_ok: bool,
) -> GateGrade:
    metrics = {
        "completed": bool(completed),
        "crashed": bool(crashed),
        "preemptions": int(preemptions),
        "full_admission_window": bool(full_admission_window),
        "utilization_ok": bool(utilization_ok),
    }
    if not completed or crashed:
        return GateGrade("stress", "D", "failed", metrics, "Run crashed, stuck, or did not finish.")
    if preemptions == 0 and full_admission_window and utilization_ok:
        grade = "A"
    elif preemptions == 0:
        grade = "B"
    else:
        grade = "C"
    return GateGrade("stress", grade, "success", metrics)


def grade_stress_v2(summary: dict[str, Any] | None) -> GateGrade:
    if not isinstance(summary, dict):
        return GateGrade("stress_v2", "D", "failed", {}, "Missing stress_v2 aggregate summary.")
    cases = summary.get("cases")
    if not isinstance(cases, list) or not cases:
        return GateGrade("stress_v2", "D", "failed", summary, "No stress_v2 cases were recorded.")

    failed_cases = [case for case in cases if case.get("status") != "success"]
    cache_cases = [case for case in cases if bool(case.get("enable_prefix_caching", False))]
    cache_hit_failures = [
        case.get("case", "")
        for case in cache_cases
        if int(case.get("hit_requests", 0) or 0) <= 0 or int(case.get("total_cached_tokens", 0) or 0) <= 0
    ]
    variable_length_cases = [
        case
        for case in cases
        if int(case.get("unique_prompt_lengths", 0) or 0) > 1
        or int(case.get("max_prompt_tokens", 0) or 0) > int(case.get("min_prompt_tokens", 0) or 0)
    ]
    eligible_hit_rates = [
        float(case.get("eligible_cache_hit_rate", 0.0) or 0.0)
        for case in cache_cases
        if float(case.get("total_eligible_cache_tokens", 0.0) or 0.0) > 0.0
    ]
    min_eligible_hit_rate = min(eligible_hit_rates) if eligible_hit_rates else 0.0
    metrics = {
        "completed": summary.get("status") == "success" and not failed_cases,
        "case_count": len(cases),
        "failed_cases": [case.get("case", "") for case in failed_cases],
        "cache_case_count": len(cache_cases),
        "cache_hit_failures": cache_hit_failures,
        "variable_length_case_count": len(variable_length_cases),
        "min_eligible_cache_hit_rate": min_eligible_hit_rate,
    }
    if summary.get("status") != "success" or failed_cases:
        return GateGrade("stress_v2", "D", "failed", metrics, "One or more serving-trace cases failed.")
    if not cache_cases:
        return GateGrade("stress_v2", "D", "failed", metrics, "No prefix-cache-enabled cases were run.")
    if cache_hit_failures:
        return GateGrade("stress_v2", "D", "failed", metrics, "Prefix-cache cases did not observe cache hits.")
    if not variable_length_cases:
        return GateGrade("stress_v2", "D", "failed", metrics, "Serving trace did not vary prompt lengths.")

    if min_eligible_hit_rate >= 0.80:
        grade = "A"
    elif min_eligible_hit_rate >= 0.50:
        grade = "B"
    else:
        grade = "C"
    return GateGrade("stress_v2", grade, "success", metrics)


def worst_required_grade(grades: list[GateGrade]) -> str:
    required = [grade.grade for grade in grades if grade.grade != "N/A"]
    if not required:
        return "N/A"
    return min(required, key=lambda grade: GRADE_ORDER[grade])
