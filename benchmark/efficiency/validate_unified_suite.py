# SPDX-License-Identifier: Apache-2.0
"""Validate unified benchmark stage status, artifacts, and matched coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.efficiency.metrics import REQUEST_METRIC_CONTRACT

SYSTEM_PROTOCOLS = {
    "sengine-vanilla": ("sparseengine", "vanilla"),
    "sengine-snapkv": ("sparseengine", "snapkv"),
    "sengine-h2o": ("sparseengine", "h2o"),
    "sengine-omnikv": ("sparseengine", "omnikv"),
    "sengine-deltakv": ("sparseengine", "deltakv"),
    "vllm-vanilla": ("vllm", "vanilla"),
    "vllm": ("vllm", "vanilla"),
}


def _layer_ids(value: Any) -> list[int] | None:
    if isinstance(value, str):
        try:
            return [int(part.strip()) for part in value.split(",") if part.strip()]
        except ValueError:
            return None
    if isinstance(value, list) and not any(isinstance(item, bool) for item in value):
        try:
            return [int(item) for item in value]
        except (TypeError, ValueError):
            return None
    return None


def _validate_omnikv_runtime_config(
    config: dict[str, Any],
    *,
    system: str,
    errors: list[str],
) -> None:
    effective = config.get("effective_runtime")
    benchmark_config = (
        effective.get("benchmark_config") if isinstance(effective, dict) else None
    )
    if not isinstance(benchmark_config, dict):
        errors.append(
            f"missing effective OmniKV runtime config for {system}; "
            "resolved_config.json must record worker_info benchmark_config"
        )
        return
    full_layers = benchmark_config.get("full_attention_layers")
    obs_layers = benchmark_config.get("obs_layer_ids")
    effective_layers = _layer_ids(full_layers)
    requested = config.get("requested_runtime")
    requested_config = (
        requested.get("config") if isinstance(requested, dict) else None
    )
    requested_layers = _layer_ids(
        requested_config.get("full_attention_layers")
        if isinstance(requested_config, dict)
        else None
    )
    if effective_layers is None or len(effective_layers) <= 1:
        errors.append(
            f"invalid OmniKV full_attention_layers for {system}: {full_layers!r}"
        )
    if requested_layers is None:
        errors.append(
            f"missing requested OmniKV full_attention_layers for {system}: "
            f"{requested!r}"
        )
    elif effective_layers is not None and requested_layers != effective_layers:
        errors.append(
            f"OmniKV full-attention layer mismatch for {system}: "
            f"requested={requested_layers} effective={effective_layers}"
        )
    if not isinstance(obs_layers, list) or not obs_layers:
        errors.append(f"invalid OmniKV obs_layer_ids for {system}: {obs_layers!r}")


def _validate_sparse_operator_stats(
    system_dir: Path,
    *,
    system: str,
    errors: list[str],
) -> None:
    stats = _read_json(system_dir / "operator_runtime_stats.json", errors)
    if stats is None:
        return
    if stats.get("status") != "success":
        errors.append(f"failed operator runtime stats for {system}: {stats}")
    ranks = stats.get("world_ranks")
    if not isinstance(ranks, list) or not ranks:
        errors.append(f"empty operator runtime stats for {system}")
        return
    for index, rank in enumerate(ranks):
        bindings = rank.get("bindings") if isinstance(rank, dict) else None
        if not isinstance(bindings, list) or not bindings:
            errors.append(
                f"missing provider bindings in operator runtime stats "
                f"for {system} rank[{index}]"
            )
            continue
        invalid = [
            binding
            for binding in bindings
            if not isinstance(binding, dict) or not binding.get("selected_provider")
        ]
        if invalid:
            errors.append(
                f"invalid provider binding records for {system} rank[{index}]: "
                f"{invalid}"
            )


def _read_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        errors.append(f"missing artifact: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid JSON {path}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"expected JSON object in {path}, got {type(value).__name__}")
        return None
    return value


def _read_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]] | None:
    if not path.is_file():
        errors.append(f"missing artifact: {path}")
        return None
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(
                    f"line {line_number} is {type(row).__name__}, expected object"
                )
            rows.append(row)
    except Exception as exc:
        errors.append(f"invalid JSONL {path}: {exc}")
        return None
    if not rows:
        errors.append(f"empty artifact: {path}")
        return None
    return rows


def _longbench_identity_map(
    rows: list[dict[str, Any]],
    *,
    artifact: str,
    default_dataset: str | None,
    errors: list[str],
) -> dict[tuple[str, int], str]:
    identities: dict[tuple[str, int], str] = {}
    for index, row in enumerate(rows):
        dataset = default_dataset if default_dataset is not None else row.get("dataset")
        source_idx = row.get("source_idx")
        status = row.get("status")
        if not isinstance(dataset, str) or not isinstance(source_idx, int):
            errors.append(
                f"invalid LongBench identity in {artifact}[{index}]: {row}"
            )
            continue
        identity = (dataset, source_idx)
        if identity in identities:
            errors.append(f"duplicate LongBench identity in {artifact}: {identity}")
            continue
        if status != "success":
            errors.append(
                f"non-success LongBench sample in {artifact} for {identity}: {status!r}"
            )
        identities[identity] = str(status)
    return identities


def _validate_longbench_artifacts(
    system_dir: Path,
    *,
    system: str,
    tasks: list[str],
    expected_count: int,
    require_operator_stats: bool,
    errors: list[str],
) -> None:
    run_status = _read_json(system_dir / "run_status.json", errors)
    if run_status is not None and run_status.get("status") != "success":
        errors.append(f"non-terminal LongBench run status for {system}: {run_status}")

    expected_total = len(tasks) * int(expected_count)
    identity_maps: dict[str, dict[tuple[str, int], str]] = {}
    for artifact in (
        "raw_outputs.jsonl",
        "parsed_outputs.jsonl",
        "sample_results.jsonl",
    ):
        rows = _read_jsonl(system_dir / artifact, errors)
        if rows is None:
            continue
        if len(rows) != expected_total:
            errors.append(
                f"LongBench artifact coverage mismatch {system}/{artifact}: "
                f"rows={len(rows)} expected={expected_total}"
            )
        identity_maps[artifact] = _longbench_identity_map(
            rows,
            artifact=f"{system}/{artifact}",
            default_dataset=None,
            errors=errors,
        )

    task_identities: dict[tuple[str, int], str] = {}
    for task in tasks:
        rows = _read_jsonl(system_dir / f"{task}.jsonl", errors)
        if rows is None:
            continue
        current = _longbench_identity_map(
            rows,
            artifact=f"{system}/{task}.jsonl",
            default_dataset=task,
            errors=errors,
        )
        overlap = set(task_identities).intersection(current)
        if overlap:
            errors.append(
                f"duplicate LongBench task identities for {system}: {sorted(overlap)}"
            )
        task_identities.update(current)

    reference_name = next(iter(identity_maps), None)
    if reference_name is not None:
        reference = identity_maps[reference_name]
        for artifact, identities in identity_maps.items():
            if identities != reference:
                errors.append(
                    f"LongBench structured identities/statuses differ for {system}: "
                    f"{reference_name} vs {artifact}"
                )
        if task_identities != reference:
            errors.append(
                f"LongBench task and structured identities/statuses differ for {system}"
            )

    if require_operator_stats:
        _validate_sparse_operator_stats(
            system_dir,
            system=system,
            errors=errors,
        )


def _validate_synthetic_rows(
    rows: list[dict[str, Any]],
    *,
    system: str,
    errors: list[str],
) -> None:
    scenarios = {row.get("scenario") for row in rows}
    expected_scenarios = {"fixed_batch", "oversubscribed_churn"}
    if scenarios != expected_scenarios:
        errors.append(
            f"synthetic scenarios mismatch for {system}: "
            f"actual={sorted(str(item) for item in scenarios)} "
            f"expected={sorted(expected_scenarios)}"
        )
    for index, row in enumerate(rows):
        if row.get("request_metric_contract") != REQUEST_METRIC_CONTRACT:
            errors.append(
                f"request metric contract mismatch in {system}[{index}]; "
                "reaggregate request artifacts or rerun before comparing"
            )
        if row.get("status") != "success":
            errors.append(f"failed synthetic row {system}[{index}]: {row}")
            continue
        for field in ("request_throughput_rps", "prefill_token_throughput_tps"):
            if not isinstance(row.get(field), (int, float)) or float(row[field]) <= 0:
                errors.append(f"invalid {field} in synthetic row {system}[{index}]")
        decode_status = row.get("decode_metric_status")
        decode_tps = row.get("decode_token_throughput_tps")
        batch_decode_tps = row.get("batch_decode_token_throughput_tps")
        tpot_ms = row.get("tpot_ms_mean")
        if decode_status == "success":
            if not isinstance(decode_tps, (int, float)) or float(decode_tps) <= 0:
                errors.append(
                    f"invalid decode_token_throughput_tps in synthetic row {system}[{index}]"
                )
            if not isinstance(tpot_ms, (int, float)) or float(tpot_ms) <= 0:
                errors.append(f"invalid tpot_ms_mean in synthetic row {system}[{index}]")
            if not isinstance(batch_decode_tps, (int, float)) or float(batch_decode_tps) <= 0:
                errors.append(
                    f"invalid batch_decode_token_throughput_tps in synthetic row "
                    f"{system}[{index}]"
                )
            elif isinstance(decode_tps, (int, float)) and abs(
                float(batch_decode_tps) - float(decode_tps)
            ) > 1e-9 * max(1.0, abs(float(decode_tps))):
                errors.append(
                    f"decode compatibility alias differs from batch window in "
                    f"{system}[{index}]"
                )
            if row.get("tpot_timing_scope") != "mean_per_request_first_token_to_finish_v2":
                errors.append(f"invalid TPOT timing scope in {system}[{index}]")
            if row.get("scenario") == "fixed_batch" and isinstance(tpot_ms, (int, float)):
                proxy = row.get("tpot_concurrency_proxy_tps")
                expected_proxy = float(row["concurrency"]) * 1000.0 / float(tpot_ms)
                if not isinstance(proxy, (int, float)) or abs(
                    float(proxy) - expected_proxy
                ) > 1e-9 * max(1.0, abs(expected_proxy)):
                    errors.append(
                        f"invalid TPOT concurrency proxy in {system}[{index}]"
                    )
        elif decode_status == "skipped_by_policy":
            if decode_tps is not None:
                errors.append(
                    f"decode throughput must be null when skipped in {system}[{index}]"
                )
            if tpot_ms is not None:
                errors.append(f"TPOT must be null when skipped in {system}[{index}]")
            if batch_decode_tps is not None:
                errors.append(
                    f"batch decode throughput must be null when skipped in "
                    f"{system}[{index}]"
                )
        else:
            errors.append(f"invalid decode metric status in {system}[{index}]")
        actual = row.get("actual_hardware_metrics")
        if not isinstance(actual, dict) or actual.get("metric_source") != "nvidia-smi sampled activity":
            errors.append(f"missing directly sampled hardware metrics in {system}[{index}]")
        forbidden = {"prefill_mfu_pct_mean", "decode_mbu_pct_mean"} & set(row)
        if forbidden:
            errors.append(
                f"theoretical efficiency metrics remain in {system}[{index}]: {sorted(forbidden)}"
            )
        concurrency = row.get("concurrency")
        if not isinstance(concurrency, int) or concurrency <= 0:
            errors.append(f"invalid concurrency in synthetic row {system}[{index}]")
        elif concurrency > 1 and row.get("prompt_len_min") == row.get("prompt_len_max"):
            errors.append(f"non-variable prompt lengths in synthetic row {system}[{index}]")
        if row.get("scenario") == "oversubscribed_churn":
            if int(row.get("request_count", 0)) <= int(concurrency or 0):
                errors.append(f"churn row is not oversubscribed in {system}[{index}]")
            if row.get("fixed_batch_comparison_status") != "success":
                errors.append(f"churn row lacks fixed-batch comparison in {system}[{index}]")
            expected_decode_comparison = (
                "skipped_by_policy" if decode_tps is None else "success"
            )
            if row.get("churn_decode_tps_comparison_status") != expected_decode_comparison:
                errors.append(
                    f"invalid churn decode comparison status in {system}[{index}]"
                )


def _synthetic_trace_map(
    rows: list[dict[str, Any]],
    *,
    system: str,
    errors: list[str],
) -> dict[str, dict[str, list[Any]]]:
    traces: dict[str, dict[str, list[Any]]] = {}
    observed_prompt_digests: set[str] = set()
    for index, row in enumerate(rows):
        trace = row.get("trace")
        if row.get("status") != "success" or not isinstance(trace, dict):
            errors.append(f"invalid raw synthetic record {system}[{index}]")
            continue
        digests = trace.get("prompt_digests")
        prompt_lengths = trace.get("prompt_lengths")
        output_lengths = trace.get("output_lengths")
        if not isinstance(digests, list) or not digests or len(digests) != len(set(digests)):
            errors.append(f"duplicate or missing prompt digests in {system}[{index}]")
            continue
        if not isinstance(prompt_lengths, list) or len(prompt_lengths) != len(digests):
            errors.append(f"prompt length coverage mismatch in {system}[{index}]")
            continue
        if not isinstance(output_lengths, list) or len(output_lengths) != len(digests):
            errors.append(f"output length coverage mismatch in {system}[{index}]")
            continue
        repeated = observed_prompt_digests.intersection(str(digest) for digest in digests)
        if repeated:
            errors.append(
                f"prompts repeat across synthetic iterations in {system}[{index}]: "
                f"{sorted(repeated)}"
            )
        observed_prompt_digests.update(str(digest) for digest in digests)
        concurrency = int(row.get("concurrency", 0))
        if concurrency > 1 and len(set(prompt_lengths)) < 2:
            errors.append(f"raw batch prompt lengths are not variable in {system}[{index}]")
        key = "|".join(
            str(row.get(field))
            for field in (
                "scenario",
                "prompt_len",
                "output_len",
                "concurrency",
                "iteration",
            )
        )
        if key in traces:
            errors.append(f"duplicate synthetic case key for {system}: {key}")
        traces[key] = {
            "prompt_digests": [str(digest) for digest in digests],
            "prompt_lengths": [int(length) for length in prompt_lengths],
            "output_lengths": [int(length) for length in output_lengths],
        }
    return traces


def validate_suite(
    root: Path,
    systems: list[str],
    tasks: list[str],
    expected_count: int,
) -> dict[str, Any]:
    errors: list[str] = []
    scenario_a_rows: dict[str, list[dict[str, Any]]] = {}
    scenario_a_traces: dict[str, dict[str, dict[str, list[Any]]]] = {}
    scenario_b_hardware: dict[str, dict[str, Any]] = {}
    scenario_b_seeds: dict[str, int] = {}

    unknown_systems = [system for system in systems if system not in SYSTEM_PROTOCOLS]
    if unknown_systems:
        errors.append(f"unknown systems: {unknown_systems}")

    for scenario in ("scenario_a_synthetic", "scenario_b_longbench"):
        for system in systems:
            system_dir = root / scenario / system
            stage = _read_json(system_dir / "stage_status.json", errors)
            if stage is not None and stage.get("status") != "success":
                errors.append(f"failed stage {scenario}/{system}: {stage}")

            if scenario == "scenario_b_longbench":
                hardware = _read_json(
                    system_dir / "gpu_timeline_summary.json", errors
                )
                if hardware is not None:
                    if hardware.get("status") != "success":
                        errors.append(
                            f"failed hardware metrics {scenario}/{system}: {hardware}"
                        )
                    scenario_b_hardware[system] = hardware.get("aggregate", {})

            result_name = "summary.json" if scenario == "scenario_a_synthetic" else "result.json"
            result = _read_json(system_dir / result_name, errors)
            if result is not None:
                if result.get("status") != "success":
                    errors.append(f"failed result {scenario}/{system}: {result}")
                if scenario == "scenario_a_synthetic":
                    rows = result.get("summary")
                    if not isinstance(rows, list) or not rows:
                        errors.append(f"missing summary rows for {scenario}/{system}")
                    else:
                        scenario_a_rows[system] = rows
                        _validate_synthetic_rows(rows, system=system, errors=errors)
                    raw_rows = _read_jsonl(system_dir / "raw_samples.jsonl", errors)
                    if raw_rows is not None:
                        scenario_a_traces[system] = _synthetic_trace_map(
                            raw_rows, system=system, errors=errors
                        )
                    request_rows = _read_jsonl(
                        system_dir / "request_samples.jsonl", errors
                    )
                    if request_rows is not None:
                        invalid_statuses = [
                            row.get("status")
                            for row in request_rows
                            if row.get("status") != "success"
                        ]
                        if invalid_statuses:
                            errors.append(
                                f"failed synthetic request samples for {system}: "
                                f"{invalid_statuses}"
                            )

            if system not in SYSTEM_PROTOCOLS:
                continue
            expected_engine, expected_method = SYSTEM_PROTOCOLS[system]
            if scenario == "scenario_a_synthetic":
                provenance = _read_json(system_dir / "run_manifest.json", errors)
                config = provenance.get("args", {}) if provenance is not None else {}
                workload = provenance.get("workload", {}) if provenance is not None else {}
                if workload.get("prefix_caching_enabled") is not False:
                    errors.append(
                        f"synthetic prefix-caching contract is missing or enabled for {system}"
                    )
                if workload.get("iteration_prompt_reuse_allowed") is not False:
                    errors.append(
                        f"synthetic prompt-reuse contract is missing or enabled for {system}"
                    )
                actual_engine = config.get("engine")
                actual_method = config.get("sparse_method")
                if actual_engine == "sparseengine":
                    _validate_sparse_operator_stats(
                        system_dir,
                        system=system,
                        errors=errors,
                    )
            else:
                provenance = _read_json(system_dir / "resolved_config.json", errors)
                config = provenance if provenance is not None else {}
                actual_engine = config.get("backend")
                actual_method = (
                    config.get("sparse_method")
                    if actual_engine == "sparseengine"
                    else "vanilla"
                )
                args = config.get("args")
                seed = args.get("seed") if isinstance(args, dict) else None
                if not isinstance(seed, int):
                    errors.append(f"missing LongBench seed for {system}: {seed!r}")
                else:
                    scenario_b_seeds[system] = seed
                if actual_engine == "vllm":
                    prefix_enabled = (
                        args.get("enable_prefix_caching")
                        if isinstance(args, dict)
                        else None
                    )
                else:
                    effective = config.get("effective_runtime")
                    prefix_enabled = (
                        effective.get("prefix_cache_enabled")
                        if isinstance(effective, dict)
                        else None
                    )
                if prefix_enabled is not False:
                    errors.append(
                        f"LongBench prefix caching is missing or enabled for {system}: "
                        f"{prefix_enabled!r}"
                    )
                _validate_longbench_artifacts(
                    system_dir,
                    system=system,
                    tasks=tasks,
                    expected_count=expected_count,
                    require_operator_stats=(expected_engine == "sparseengine"),
                    errors=errors,
                )
            if provenance is not None and (
                actual_engine != expected_engine or actual_method != expected_method
            ):
                errors.append(
                    f"protocol mismatch {scenario}/{system}: "
                    f"actual={actual_engine}/{actual_method} "
                    f"expected={expected_engine}/{expected_method}"
                )
            if (
                scenario == "scenario_b_longbench"
                and actual_engine == "sparseengine"
                and actual_method == "omnikv"
            ):
                _validate_omnikv_runtime_config(
                    config,
                    system=system,
                    errors=errors,
                )

    coverage = validate_longbench_coverage(
        root / "scenario_b_longbench", systems, tasks, expected_count
    )
    errors.extend(coverage["errors"])
    reference_traces: dict[str, dict[str, list[Any]]] | None = None
    for system in systems:
        traces = scenario_a_traces.get(system)
        if traces is None:
            continue
        if reference_traces is None:
            reference_traces = traces
        elif traces != reference_traces:
            errors.append(f"synthetic random traces differ for system={system}")
    if scenario_b_seeds and len(set(scenario_b_seeds.values())) != 1:
        errors.append(f"LongBench seeds differ across systems: {scenario_b_seeds}")

    return {
        "status": "failed" if errors else "success",
        "systems": systems,
        "expected_samples_per_task": expected_count,
        "scenario_a_summary": scenario_a_rows,
        "scenario_a_trace_keys": sorted((reference_traces or {}).keys()),
        "scenario_b_hardware": scenario_b_hardware,
        "longbench_source_ids": coverage["source_ids"],
        "errors": errors,
    }


def validate_longbench_coverage(
    root: Path,
    systems: list[str],
    tasks: list[str],
    expected_count: int,
) -> dict[str, Any]:
    errors: list[str] = []
    reference_ids: dict[str, list[int]] | None = None
    for system in systems:
        task_ids: dict[str, list[int]] = {}
        for task in tasks:
            path = root / system / f"{task}.jsonl"
            if not path.is_file():
                errors.append(f"missing LongBench output: {path}")
                continue
            try:
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except Exception as exc:
                errors.append(f"invalid LongBench output {path}: {exc}")
                continue
            successful_ids = [
                int(row["source_idx"])
                for row in rows
                if row.get("status") == "success" and row.get("source_idx") is not None
            ]
            if len(rows) != expected_count or len(successful_ids) != expected_count:
                errors.append(
                    f"coverage mismatch {system}/{task}: rows={len(rows)} "
                    f"success={len(successful_ids)} expected={expected_count}"
                )
            task_ids[task] = successful_ids
        if reference_ids is None:
            reference_ids = task_ids
        elif task_ids != reference_ids:
            errors.append(f"LongBench source IDs differ for system={system}")

    return {
        "status": "failed" if errors else "success",
        "systems": systems,
        "expected_samples_per_task": expected_count,
        "source_ids": reference_ids or {},
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--systems", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--longbench-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_count <= 0:
        raise ValueError(f"--expected-count must be positive, got {args.expected_count}.")
    systems = [item.strip() for item in args.systems.split(",") if item.strip()]
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    if not systems or not tasks:
        raise ValueError("--systems and --tasks must be non-empty.")
    if args.longbench_only:
        report = validate_longbench_coverage(args.root, systems, tasks, args.expected_count)
        output = args.root / "coverage_status.json"
    else:
        report = validate_suite(args.root, systems, tasks, args.expected_count)
        output = args.root / "suite_status.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
