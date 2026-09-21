# SPDX-License-Identifier: Apache-2.0
"""Standardized Synthetic Length Sweep & Micro-Efficiency Benchmark for SparseEngine & vLLM."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
src_path = str(REPO_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from benchmark.efficiency.hardware_monitor import GPUHardwareMonitor
from benchmark.efficiency.metrics_calculator import ModelArchitectureSpecs
from benchmark.efficiency.metrics import (
    BATCH_DECODE_WINDOW_SCOPE,
    REQUEST_METRIC_CONTRACT,
    REQUEST_TPOT_SCOPE,
    TPOT_CONCURRENCY_PROXY_SCOPE,
    event_window_metrics as _phase_throughput_metrics,
    mean_event_window_metrics as _mean_phase_throughput_metrics,
    percentile as _percentile,
    request_metrics,
    request_summary,
    request_timeline_metrics as _request_phase_metrics_from_timestamps,
    tpot_concurrency_proxy_tps as _tpot_concurrency_proxy_tps,
)
from benchmark.efficiency.workload import (
    TRACE_GENERATOR_VERSION,
    build_request_trace,
    derive_trace_seed,
    trace_metadata,
)
from benchmark.sparseengine_regression.manifest import (
    validate_omnikv_benchmark_config,
)


class HardwareMetricError(RuntimeError):
    """Raised when directly sampled GPU metrics are unavailable or incomplete."""



def _installed_distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _physical_gpu_metadata(physical_gpu_ids: list[int]) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            ",".join(str(gpu_id) for gpu_id in physical_gpu_ids),
            "--query-gpu=index,name,compute_cap,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    devices = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise RuntimeError(f"Unexpected nvidia-smi GPU metadata row: {line!r}.")
        gpu_id, name, compute_capability, total_memory_mib, driver_version = fields
        capability_parts = compute_capability.split(".")
        if len(capability_parts) != 2:
            raise RuntimeError(
                f"Unexpected nvidia-smi compute capability {compute_capability!r}."
            )
        devices.append(
            {
                "physical_device_index": int(gpu_id),
                "name": name,
                "compute_capability": [int(part) for part in capability_parts],
                "total_memory_mib": int(total_memory_mib),
                "driver_version": driver_version,
            }
        )
    actual_ids = {device["physical_device_index"] for device in devices}
    if actual_ids != set(physical_gpu_ids):
        raise RuntimeError(
            "nvidia-smi GPU metadata did not cover the selected devices: "
            f"expected={sorted(physical_gpu_ids)} actual={sorted(actual_ids)}."
        )
    return devices


def _runtime_environment_metadata(physical_gpu_ids: list[int]) -> dict[str, Any]:
    import torch

    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "packages": {
            name: _installed_distribution_version(name)
            for name in (
                "vllm",
                "triton",
                "flashinfer-python",
                "sglang-kernel",
                "tilelang",
            )
        },
        "physical_cuda_devices": _physical_gpu_metadata(physical_gpu_ids),
    }


def _git_metadata() -> dict[str, Any]:
    def _run_git(*args: str) -> str | None:
        try:
            res = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            return res.stdout.strip() or None
        except Exception:
            return None

    return {
        "git_commit": _run_git("rev-parse", "HEAD"),
        "git_branch": _run_git("branch", "--show-current"),
        "git_dirty": bool(_run_git("status", "--porcelain")),
    }


def _parse_ints(val: str) -> list[int]:
    return [int(x.strip()) for x in val.split(",") if x.strip()]


def _parse_json_arg(val: str | None) -> dict[str, Any]:
    if not val:
        return {}
    val = val.strip()
    if val.startswith("@"):
        path = Path(val[1:]).expanduser()
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(val)


_DECODE_GRAPH_COUNTERS = (
    "capture_count",
    "replay_count",
    "eager_static_count",
    "force_eager_count",
    "eviction_count",
    "recapture_count",
)


def _decode_graph_counter_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, int]:
    return {
        name: int(after.get(name, 0)) - int(before.get(name, 0))
        for name in _DECODE_GRAPH_COUNTERS
    }


def _monitor_gpu_ids(explicit: str | None) -> list[int]:
    value = explicit or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not value:
        raise ValueError(
            "Actual GPU metric collection requires --monitor-gpus or an integer-only "
            "CUDA_VISIBLE_DEVICES list."
        )
    try:
        gpu_ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(
            "Actual GPU metric collection currently requires physical integer GPU IDs, "
            f"got {value!r}."
        ) from exc
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"GPU metric IDs must be non-empty and unique, got {gpu_ids}.")
    return gpu_ids


def _case_monitor(
    args: argparse.Namespace,
    case_name: str,
) -> GPUHardwareMonitor:
    output_file = Path(args.output_dir) / "case_hardware" / f"{case_name}.json"
    monitor = GPUHardwareMonitor(
        _monitor_gpu_ids(args.monitor_gpus),
        interval_ms=args.hardware_sampling_interval_ms,
        output_file=output_file,
    )
    monitor.start()
    return monitor


def _actual_hardware_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("status") != "success":
        raise HardwareMetricError(f"Actual GPU metric collection failed: {summary}")
    aggregate = summary.get("aggregate")
    gpus = summary.get("gpus")
    if not isinstance(aggregate, dict) or not isinstance(gpus, dict) or not gpus:
        raise HardwareMetricError(f"Actual GPU metric summary is incomplete: {summary}")
    return {
        "metric_source": "nvidia-smi sampled activity",
        "sample_count": int(summary["total_samples"]),
        "sampling_interval_ms": int(summary["sampling_interval_ms"]),
        "gpu_compute_activity_pct_mean": float(aggregate["mean_compute_util_pct"]),
        "gpu_memory_io_activity_pct_mean": float(
            aggregate["mean_memory_io_activity_pct"]
        ),
        "gpu_active_duty_pct_mean": float(
            aggregate["mean_coarse_gpu_active_duty_pct"]
        ),
        "gpu_power_w_mean_total": float(aggregate["avg_total_power_w"]),
        "peak_vram_gb_max": max(float(gpu["peak_vram_gb"]) for gpu in gpus.values()),
        "per_gpu": gpus,
    }


def _trace_for_iteration(
    args: argparse.Namespace,
    model_specs: ModelArchitectureSpecs,
    *,
    scenario: str,
    phase: str,
    prompt_len: int,
    output_len: int,
    concurrency: int,
    iteration: int,
    request_count: int,
    vary_output_lengths: bool,
):
    seed = derive_trace_seed(
        args.seed,
        scenario=scenario,
        phase=phase,
        nominal_prompt_len=prompt_len,
        nominal_output_len=output_len,
        concurrency=concurrency,
        iteration=iteration,
    )
    trace = build_request_trace(
        seed=seed,
        request_count=request_count,
        nominal_prompt_len=prompt_len,
        nominal_output_len=output_len,
        vocab_size=model_specs.vocab_size,
        prompt_jitter_fraction=args.prompt_length_jitter,
        output_jitter_fraction=args.output_length_jitter,
        vary_output_lengths=vary_output_lengths,
    )
    if getattr(args, "shared_prompt", False):
        first = trace[0]
        trace = [replace(request, prompt_token_ids=list(first.prompt_token_ids),
                         prompt_digest=first.prompt_digest, trace_seed=first.trace_seed)
                 for request in trace]
    return trace


def _attach_churn_comparisons(rows: list[dict[str, Any]]) -> None:
    fixed_by_case: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("engine"),
            row.get("sparse_method"),
            row.get("protocol_label"),
            row.get("prompt_len"),
            row.get("output_len"),
            row.get("concurrency"),
        )
        if row.get("scenario") == "fixed_batch":
            fixed_by_case[key] = row
    for row in rows:
        if row.get("scenario") != "oversubscribed_churn":
            continue
        key = (
            row.get("engine"),
            row.get("sparse_method"),
            row.get("protocol_label"),
            row.get("prompt_len"),
            row.get("output_len"),
            row.get("concurrency"),
        )
        fixed = fixed_by_case.get(key)
        if fixed is None:
            row["fixed_batch_comparison_status"] = "skipped_by_policy"
            continue
        fixed_decode_tps = fixed.get("decode_token_throughput_tps")
        fixed_request_rps = float(fixed["request_throughput_rps"])
        if fixed_request_rps <= 0:
            raise ValueError(f"Fixed-batch throughput must be positive for comparison: {fixed}")
        row["fixed_batch_comparison_status"] = "success"
        row_decode_tps = row.get("decode_token_throughput_tps")
        if fixed_decode_tps is None or row_decode_tps is None:
            row["churn_decode_tps_comparison_status"] = "skipped_by_policy"
            row["churn_decode_tps_ratio_vs_fixed_batch"] = None
        else:
            if float(fixed_decode_tps) <= 0 or float(row_decode_tps) <= 0:
                raise ValueError(
                    f"Decode throughput must be positive for comparison: fixed={fixed}, row={row}"
                )
            row["churn_decode_tps_comparison_status"] = "success"
            row["churn_decode_tps_ratio_vs_fixed_batch"] = (
                float(row_decode_tps) / float(fixed_decode_tps)
            )
        row["churn_request_rps_ratio_vs_fixed_batch"] = (
            float(row["request_throughput_rps"]) / fixed_request_rps
        )
        row["churn_ttft_p99_delta_ms_vs_fixed_batch"] = (
            float(row["ttft_ms_p99"]) - float(fixed["ttft_ms_p99"])
        )


def _append_request_samples(
    output_dir: Path,
    *,
    engine: str,
    sparse_method: str,
    scenario: str,
    prompt_len: int,
    output_len: int,
    concurrency: int,
    iteration: int,
    requests: list[dict[str, Any]],
) -> None:
    path = output_dir / "request_samples.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for request in requests:
            row = {
                "engine": engine,
                "sparse_method": sparse_method,
                "scenario": scenario,
                "nominal_prompt_len": prompt_len,
                "nominal_output_len": output_len,
                "concurrency": concurrency,
                "iteration": iteration,
                **request,
            }
            if row.get("status") != "success":
                raise ValueError(f"Benchmark request sample is not successful: {row}")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _vllm_request_phase_seconds(metrics: Any) -> tuple[float, float, str]:
    """Return TTFT, decode duration, and the explicit vLLM timing contract."""
    legacy_first = getattr(metrics, "first_token_time", None)
    legacy_finished = getattr(metrics, "finished_time", None)
    arrival = getattr(metrics, "arrival_time", None)
    if legacy_first is not None and legacy_finished is not None and arrival is not None:
        ttft_s = float(legacy_first) - float(arrival)
        decode_s = float(legacy_finished) - float(legacy_first)
        source = "vllm_legacy_wall_timestamps"
    else:
        first_token_latency = getattr(metrics, "first_token_latency", None)
        first_token_ts = getattr(metrics, "first_token_ts", None)
        last_token_ts = getattr(metrics, "last_token_ts", None)
        if (
            first_token_latency is None
            or first_token_ts is None
            or last_token_ts is None
            or float(first_token_latency) <= 0
            or float(first_token_ts) <= 0
            or float(last_token_ts) <= 0
        ):
            raise RuntimeError(
                "vLLM request metrics expose neither complete legacy timestamps nor "
                "complete V1 latency/timestamps."
            )
        ttft_s = float(first_token_latency)
        decode_s = float(last_token_ts) - float(first_token_ts)
        source = "vllm_v1_first_token_latency_and_monotonic_decode"
    if ttft_s <= 0 or decode_s < 0:
        raise RuntimeError(
            f"vLLM reported invalid phase timing: ttft_s={ttft_s}, decode_s={decode_s}."
        )
    return ttft_s, decode_s, source


def _vllm_phase_metrics(outputs: list[Any], expected_output_len: int) -> tuple[float, float | None]:
    """Return mean request TTFT and TPOT from the same requests."""
    if not outputs:
        raise RuntimeError("vLLM returned no request outputs.")
    ttft_values = []
    tpot_values = []
    for output in outputs:
        metrics = getattr(output, "metrics", None)
        if metrics is None:
            raise RuntimeError(
                "vLLM request metrics are unavailable; TTFT/TPOT require request timing."
            )
        token_count = len(output.outputs[0].token_ids) if output.outputs else 0
        if token_count != expected_output_len:
            raise RuntimeError(
                f"vLLM generated {token_count} tokens, expected output_len={expected_output_len}."
            )
        ttft_s, decode_s, _source = _vllm_request_phase_seconds(metrics)
        row = request_metrics(ttft_s, decode_s, token_count)
        ttft_values.append(row["ttft_ms"])
        if token_count > 1:
            decode_ms = decode_s * 1000.0
            if decode_ms <= 0:
                raise RuntimeError(f"vLLM reported non-positive decode duration {decode_ms} ms.")
            tpot_values.append(row["tpot_ms"])
    return statistics.fmean(ttft_values), (sum(tpot_values) / len(tpot_values) if tpot_values else None)


def _vllm_batch_phase_seconds(outputs: list[Any]) -> tuple[float, float]:
    """Return prefill and decode wall-time windows for one submitted workload."""
    if not outputs:
        raise RuntimeError("vLLM returned no request outputs.")

    ttft_values = []
    decode_starts = []
    decode_finishes = []
    timing_sources = set()
    for output in outputs:
        metrics = getattr(output, "metrics", None)
        if metrics is None:
            raise RuntimeError(
                "vLLM request metrics are unavailable; phase throughput requires request timing."
            )
        ttft_s, decode_s, source = _vllm_request_phase_seconds(metrics)
        timing_sources.add(source)
        decode_start = float(
            metrics.first_token_time
            if source == "vllm_legacy_wall_timestamps"
            else metrics.first_token_ts
        )
        ttft_values.append(ttft_s)
        decode_starts.append(decode_start)
        decode_finishes.append(decode_start + decode_s)

    if len(timing_sources) != 1:
        raise RuntimeError(
            f"vLLM returned mixed request timing contracts: {sorted(timing_sources)}."
        )
    return max(ttft_values), max(decode_finishes) - min(decode_starts)


def _record_batch_first_tokens(
    expected_seq_ids: set[int],
    observed_seq_ids: set[int],
    step_outputs: list[tuple[int, list[int]]],
    finished_outputs: list[tuple[int, list[int], Any, Any]],
) -> bool:
    """Record first-token publication and report when the whole batch is covered."""
    for seq_id, token_ids in step_outputs:
        if token_ids and int(seq_id) in expected_seq_ids:
            observed_seq_ids.add(int(seq_id))
    for seq_id, token_ids, _token_logprobs, _top_logprobs in finished_outputs:
        if token_ids and int(seq_id) in expected_seq_ids:
            observed_seq_ids.add(int(seq_id))
    return observed_seq_ids == expected_seq_ids


def _sparse_probe_max_model_len(args, hyper_params):
    required = max(args.prompt_lens) + max(args.output_lens) + 128
    capacity = int(hyper_params.get("max_model_len", required))
    if capacity < required:
        raise ValueError(
            f"max_model_len={capacity} cannot cover the probe workload and margin ({required})"
        )
    return capacity


def _resolve_sparse_probe_protocol(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int | None, dict[str, Any], str]:
    hyper_params = _parse_json_arg(args.hyper_params)
    hyper_params.setdefault("tensor_parallel_size", args.tensor_parallel_size)
    hyper_params.setdefault("expert_parallel_size", getattr(args, "expert_parallel_size", 1))
    hyper_params.setdefault("gpu_memory_utilization", args.gpu_memory_utilization)
    hyper_params.setdefault("decode_graph", True)
    from sparseengine.config import Config
    hyper_params.setdefault("decode_reservation_tokens", Config.decode_reservation_tokens)
    hyper_params.setdefault("max_num_batched_tokens", args.max_num_batched_tokens)
    hyper_params.setdefault("engine_prefill_chunk_size", 8192)
    if args.sparse_method == "snapkv":
        hyper_params.setdefault("snapkv_window_size", 64)
        hyper_params.setdefault("sink_keep_tokens", 64)
        hyper_params.setdefault("decode_keep_tokens", 2048)
        hyper_params.setdefault("recent_keep_tokens", 64)
        hyper_params.setdefault("pool_kernel_size", 7)
    elif args.sparse_method == "h2o":
        hyper_params.setdefault("h2o_decode_budget", 4096)
        hyper_params.setdefault("h2o_prefill_budget", 8192)
        hyper_params.setdefault("h2o_recent_ratio", 0.5)
        hyper_params.setdefault("h2o_prefill_score_window", 128)
    elif args.sparse_method == "omnikv":
        validate_omnikv_benchmark_config(
            hyper_params,
            allow_single_full_layer=args.allow_single_omnikv_full_layer,
        )

    if args.sparse_method in {"snapkv", "h2o"}:
        configured_mode = hyper_params.get("sparse_prefill_score_mode")
        if args.sparse_prefill_score_mode is None and configured_mode is None:
            raise ValueError(
                f"{args.sparse_method} efficiency probes require an explicit "
                "--sparse-prefill-score-mode "
                "(probability or logits)."
            )
        if (
            args.sparse_prefill_score_mode is not None
            and configured_mode is not None
            and str(configured_mode) != args.sparse_prefill_score_mode
        ):
            raise ValueError(
                "Conflicting sparse prefill score modes in --hyper-params and "
                "--sparse-prefill-score-mode."
            )
        hyper_params["sparse_prefill_score_mode"] = (
            args.sparse_prefill_score_mode
            if args.sparse_prefill_score_mode is not None
            else str(configured_mode)
        )

    sparse_budget = (
        int(hyper_params.get("sink_keep_tokens", 64))
        + int(hyper_params.get("decode_keep_tokens", 2048))
        + int(hyper_params.get("recent_keep_tokens", 64))
        if args.sparse_method == "snapkv"
        else None
    )
    protocol = {
        "score_mode": hyper_params.get("sparse_prefill_score_mode"),
        "score_window": hyper_params.get("snapkv_window_size"),
        "sparse_budget": sparse_budget,
        "max_num_batched_tokens": int(hyper_params["max_num_batched_tokens"]),
    }
    protocol_label = f"sparseengine-{args.sparse_method}"
    if args.sparse_method == "snapkv":
        protocol_label += (
            f"-{protocol['score_mode']}-budget{protocol['sparse_budget']}"
            f"-window{protocol['score_window']}"
        )
    elif args.sparse_method == "h2o":
        protocol.update(
            {
                "score_window": int(hyper_params["h2o_prefill_score_window"]),
                "decode_budget": int(hyper_params["h2o_decode_budget"]),
                "prefill_budget": int(hyper_params["h2o_prefill_budget"]),
                "recent_ratio": float(hyper_params["h2o_recent_ratio"]),
            }
        )
        protocol_label += (
            f"-{protocol['score_mode']}-decode{protocol['decode_budget']}"
            f"-prefill{protocol['prefill_budget']}"
            f"-window{protocol['score_window']}"
        )
    return hyper_params, sparse_budget, protocol, protocol_label


def _format_markdown_report(
    summary_rows: list[dict[str, Any]],
    model_name: str,
    tp_size: int,
    ep_size: int,
) -> str:
    lines = [
        "# Standardized LLM Efficiency Benchmark Report",
        "",
        f"- **Model**: `{model_name}`",
        f"- **Tensor parallel size**: `{tp_size}`",
        f"- **Expert parallel size**: `{ep_size}`",
        "- **GPU metrics**: directly sampled activity; no theoretical MFU/MBU estimates",
        f"- **Timestamp**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "| System / Method | Scenario | Prompt Range | Output Range | Concurrency | Req/s | First-token window tok/s | Batch decode-window tok/s | TTFT p50/p99 (ms) | Request TPOT mean (ms) | TPOT-equivalent concurrent tok/s | GPU compute activity | GPU memory I/O activity | Peak VRAM (GB) | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    def _number(value: Any, precision: int) -> str:
        return "n/a" if value is None else f"{float(value):.{precision}f}"

    for r in summary_rows:
        fallback_label = f"{r['engine']}-{r.get('sparse_method', 'vanilla')}"
        sys_label = f"`{r.get('protocol_label', fallback_label)}`"
        prompt_range = f"{r['prompt_len_min']}-{r['prompt_len_max']}"
        output_range = f"{r['output_len_min']}-{r['output_len_max']}"
        lines.append(
            f"| {sys_label} | {r['scenario']} | {prompt_range} | {output_range} "
            f"| {r['concurrency']} | {_number(r.get('request_throughput_rps'), 2)} "
            f"| {_number(r.get('prefill_token_throughput_tps'), 2)} "
            f"| {_number(r.get('decode_token_throughput_tps'), 2)} "
            f"| {_number(r.get('ttft_ms_p50'), 2)}/{_number(r.get('ttft_ms_p99'), 2)} "
            f"| {_number(r.get('tpot_ms_mean'), 2)} "
            f"| {_number(r.get('tpot_concurrency_proxy_tps'), 2)} "
            f"| {_number(r.get('gpu_compute_activity_pct_mean'), 1)}% "
            f"| {_number(r.get('gpu_memory_io_activity_pct_mean'), 1)}% "
            f"| {_number(r.get('peak_vram_gb_max'), 2)} | {r.get('status', 'success')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _replica_concurrency(global_concurrency: int, hyper_params: dict[str, Any]) -> int:
    replicas = int(hyper_params.get("data_parallel_size", 1))
    if replicas <= 0:
        raise ValueError("data_parallel_size must be positive")
    return (global_concurrency + replicas - 1) // replicas


def _resident_sequence_rows(
    global_concurrency: int, hyper_params: dict[str, Any]
) -> int:
    return int(
        hyper_params.get(
            "max_num_seqs_in_gpu",
            _replica_concurrency(global_concurrency, hyper_params),
        )
    )


def run_sparseengine_probe(
    args: argparse.Namespace,
    model_specs: ModelArchitectureSpecs,
) -> list[dict[str, Any]]:
    import torch
    from sparseengine import LLM, SamplingParams
    from sparseengine.utils.profiler import profiler

    if args.scenario == "churn":
        return run_sparseengine_churn(args, model_specs)

    results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_samples_file = output_dir / "raw_samples.jsonl"

    hyper_params, sparse_budget, protocol, protocol_label = _resolve_sparse_probe_protocol(args)
    sparse_kwargs: dict[str, Any] = {"sparse_method": args.sparse_method}

    max_len_needed = _sparse_probe_max_model_len(args, hyper_params)
    max_concurrency = max(args.batch_sizes)
    wave_size = int(getattr(args, "prefill_wave_size", 0))
    if wave_size < 0 or (wave_size and (args.sparse_method != "h2o" or args.scenario != "fixed")):
        raise ValueError("--prefill-wave-size requires native H2O with --scenario fixed")
    if wave_size:
        protocol = {**protocol, "prefill_wave_size": wave_size,
                    "wave_barrier": "previous wave first-token publication after H2O final-prefill compaction",
                    "arrival_scope": "all requests available at workload start; wave admission wait included"}
    engine_kwargs = {
        **hyper_params,
        "max_model_len": max_len_needed,
        "enable_prefix_caching": getattr(args, "enable_prefix_caching", False),
        **sparse_kwargs,
        "max_num_seqs_in_batch": _replica_concurrency(
            min(wave_size, max_concurrency) if wave_size else max_concurrency, hyper_params
        ),
        "max_decoding_seqs": _replica_concurrency(max_concurrency, hyper_params),
        "max_num_seqs_in_gpu": _resident_sequence_rows(
            max_concurrency, hyper_params
        ),
    }

    print(f"[SparseEngine Probe] Initializing LLM with method={args.sparse_method}, max_model_len={max_len_needed}...")
    llm = LLM(args.model_path, **engine_kwargs)

    for p_len in args.prompt_lens:
        for o_len in args.output_lens:
            for bs in args.batch_sizes:
                print(f"\n---> Benchmarking Sweep: PromptLen={p_len}, OutputLen={o_len}, BatchSize={bs} <---")

                # 1. Warmup
                for w_idx in range(args.num_warmups):
                    print(f"  [Warmup {w_idx + 1}/{args.num_warmups}]...")
                    warmup_trace = _trace_for_iteration(
                        args,
                        model_specs,
                        scenario="fixed_batch",
                        phase="warmup",
                        prompt_len=p_len,
                        output_len=o_len,
                        concurrency=bs,
                        iteration=w_idx,
                        request_count=bs,
                        vary_output_lengths=False,
                    )
                    width = wave_size or len(warmup_trace)
                    for offset in range(0, len(warmup_trace), width):
                        waiting_for_first = set()
                        for request in warmup_trace[offset:offset + width]:
                            waiting_for_first.add(int(llm.add_request(
                                request.prompt_token_ids,
                                SamplingParams(temperature=0.0, top_p=1.0, top_k=1,
                                               ignore_eos=True, max_tokens=request.output_len),
                            )))
                        if wave_size:
                            while waiting_for_first:
                                finished, _ = llm.step()
                                for seq_id, tokens in getattr(llm, "last_step_token_outputs", []):
                                    if tokens:
                                        waiting_for_first.discard(int(seq_id))
                                for seq_id, tokens, *_ in finished:
                                    if tokens:
                                        waiting_for_first.discard(int(seq_id))
                    while not llm.is_finished():
                        llm.step()

                torch.cuda.synchronize()

                # 2. Benchmark Iterations
                iter_records = []
                case_name = f"fixed-p{p_len}-o{o_len}-c{bs}"
                monitor = _case_monitor(args, case_name)
                try:
                    for it in range(args.num_iters):
                        profiler.reset()
                        trace = _trace_for_iteration(
                            args,
                            model_specs,
                            scenario="fixed_batch",
                            phase="measure",
                            prompt_len=p_len,
                            output_len=o_len,
                            concurrency=bs,
                            iteration=it,
                            request_count=bs,
                            vary_output_lengths=False,
                        )
                        graph_before = llm.debug_sparse_state_summaries()[0]["decode_graph"]
                        t_start = time.perf_counter()

                        seq_to_request: dict[int, Any] = {}
                        arrival_times: dict[int, float] = {}
                        first_token_times: dict[int, float] = {}
                        finished_times: dict[int, float] = {}
                        generated_counts: dict[int, int] = {}
                        cached_tokens: dict[int, int] = {}
                        next_request = 0
                        wave_events = []
                        current_wave = set()
                        peak_first_token_live_requests = 0
                        peak_scheduler_decoding_requests = 0
                        peak_decode_free_slot_stats = None
                        def admit_wave():
                            nonlocal next_request, current_wave
                            current_wave = set()
                            width = wave_size or len(trace)
                            for request in trace[next_request:next_request + width]:
                                arrival = t_start if wave_size else time.perf_counter()
                                seq_id = int(llm.add_request(
                                    request.prompt_token_ids,
                                    SamplingParams(temperature=0.0, top_p=1.0, top_k=1,
                                                   ignore_eos=True, max_tokens=request.output_len),
                                ))
                                if seq_id in seq_to_request:
                                    raise RuntimeError(f"Duplicate request sequence ID: {seq_id}")
                                seq_to_request[seq_id] = request
                                arrival_times[seq_id] = arrival
                                current_wave.add(seq_id)
                            next_request += len(current_wave)
                            if wave_size:
                                wave_events.append({"admitted_seq_ids": sorted(current_wave),
                                                    "time_since_start_s": time.perf_counter() - t_start,
                                                    "previous_first_tokens": len(first_token_times),
                                                    "free_slot_stats": llm.worker_load()["cache"]})
                        admit_wave()

                        while not llm.is_finished():
                            finished_outputs, _num_tokens = llm.step()
                            now = time.perf_counter()
                            for seq_id, hit_tokens in llm.last_step_prompt_cache_hits:
                                cached_tokens[int(seq_id)] = max(
                                    cached_tokens.get(int(seq_id), 0), int(hit_tokens))
                            for seq_id, token_ids in getattr(
                                llm, "last_step_token_outputs", []
                            ):
                                seq_id = int(seq_id)
                                if token_ids and seq_id in seq_to_request:
                                    first_token_times.setdefault(seq_id, now)
                            for seq_id, token_ids, _token_logprobs, _top_logprobs in finished_outputs:
                                seq_id = int(seq_id)
                                if token_ids and seq_id in seq_to_request:
                                    first_token_times.setdefault(seq_id, now)
                                finished_times[seq_id] = now
                                generated_counts[seq_id] = len(token_ids)

                            peak_first_token_live_requests = max(
                                peak_first_token_live_requests,
                                len(first_token_times.keys() - finished_times.keys()),
                            )
                            decoding_requests = llm.worker_routing_load()["decoding_requests"]
                            if decoding_requests > peak_scheduler_decoding_requests:
                                peak_scheduler_decoding_requests = decoding_requests
                                if wave_size:
                                    peak_decode_free_slot_stats = llm.worker_load()["cache"]
                            if next_request < len(trace) and current_wave <= first_token_times.keys():
                                admit_wave()
                        if len(seq_to_request) != len(trace):
                            raise RuntimeError("Wave workload ended before all requests were admitted")

                        elapsed_s = time.perf_counter() - t_start
                        graph_after = llm.debug_sparse_state_summaries()[0]["decode_graph"]
                        if getattr(args, "enable_prefix_caching", False):
                            if set(cached_tokens) != set(seq_to_request):
                                raise RuntimeError("Missing native prefix-cache hit observations")
                            for seq_id, hits in cached_tokens.items():
                                if not 0 <= hits < seq_to_request[seq_id].prompt_len:
                                    raise RuntimeError(f"Invalid native prefix-cache hit count: {hits}")
                        timing_metrics = _request_phase_metrics_from_timestamps(
                            arrival_times=arrival_times,
                            first_token_times=first_token_times,
                            finished_times=finished_times,
                            generated_counts=generated_counts,
                        )
                        for seq_id, generated in generated_counts.items():
                            expected = int(seq_to_request[seq_id].output_len)
                            if generated != expected:
                                raise RuntimeError(
                                    f"SparseEngine generated {generated} tokens for seq_id={seq_id}, "
                                    f"expected {expected}."
                                )
                        ttft_ms = float(timing_metrics["ttft_ms"])
                        tpot_ms = timing_metrics["tpot_ms"]
                        total_input = sum(request.prompt_len for request in trace)
                        total_output = sum(request.output_len for request in trace)
                        phase_metrics = _phase_throughput_metrics(
                            total_input_tokens=total_input,
                            total_output_tokens=total_output,
                            request_count=bs,
                            prefill_elapsed_s=float(timing_metrics["prefill_elapsed_s"]),
                            decode_elapsed_s=float(timing_metrics["decode_elapsed_s"]),
                        )
                        request_results = []
                        for timing in timing_metrics["request_timings"]:
                            seq_id = int(timing["request_id"])
                            request_results.append(
                                {
                                    **seq_to_request[seq_id].metadata(),
                                    **timing,
                                    "seq_id": seq_id,
                                    "num_cached_tokens": cached_tokens.get(seq_id, 0),
                                    "timing_source": (
                                        "sparseengine_wave_workload_arrival_step_publication_v1" if wave_size
                                        else "sparseengine_step_token_publication_no_extra_sync_v1"),
                                }
                            )
                        profiler_snap = profiler.snapshot()

                        rec = {
                            "engine": "sparseengine",
                            "sparse_method": args.sparse_method,
                            "scenario": "fixed_batch",
                            "prompt_len": p_len,
                            "output_len": o_len,
                            "concurrency": bs,
                            "iteration": it,
                            "status": "success",
                            "elapsed_s": elapsed_s,
                            "ttft_ms": ttft_ms,
                            "batch_max_ttft_ms": max(row["ttft_ms"] for row in request_results),
                            **request_summary(request_results),
                            "tpot_ms": None if tpot_ms is None else round(tpot_ms, 2),
                            "tpot_timing_scope": REQUEST_TPOT_SCOPE,
                            "tpot_concurrency_proxy_tps": _tpot_concurrency_proxy_tps(
                                concurrency=bs,
                                tpot_ms=tpot_ms,
                            ),
                            "tpot_concurrency_proxy_scope": TPOT_CONCURRENCY_PROXY_SCOPE,
                            "request_throughput_rps": bs / elapsed_s,
                            "output_token_throughput_tps": total_output / elapsed_s,
                            **phase_metrics,
                            "profiler_breakdown": profiler_snap,
                            "profiler_status": "success" if profiler_snap else "skipped_by_policy",
                            "protocol": protocol,
                            "prefill_wave_events": wave_events,
                            "peak_first_token_live_requests": peak_first_token_live_requests,
                            "peak_scheduler_decoding_requests": peak_scheduler_decoding_requests,
                            "peak_decode_free_slot_stats": peak_decode_free_slot_stats,
                            "protocol_label": protocol_label,
                            "decode_metric_status": "success" if tpot_ms is not None else "skipped_by_policy",
                            "trace": trace_metadata(trace, allow_duplicate_prompts=getattr(args, "shared_prompt", False)),
                            "request_results": request_results,
                            "decode_cuda_graph_before": graph_before,
                            "decode_cuda_graph_after": graph_after,
                            "decode_cuda_graph_counter_delta": _decode_graph_counter_delta(
                                graph_before, graph_after
                            ),
                        }
                        iter_records.append(rec)
                        with open(raw_samples_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        _append_request_samples(
                            output_dir,
                            engine="sparseengine",
                            sparse_method=args.sparse_method,
                            scenario="fixed_batch",
                            prompt_len=p_len,
                            output_len=o_len,
                            concurrency=bs,
                            iteration=it,
                            requests=request_results,
                        )

                        print(
                            f"  Iter {it + 1}/{args.num_iters}: TTFT={ttft_ms:.1f}ms | "
                            f"TPOT={tpot_ms if tpot_ms is not None else 'n/a'}ms | "
                            f"FirstTokenWindow={phase_metrics['prefill_token_throughput_tps']:.1f} tok/s | "
                            f"BatchDecodeWindow={phase_metrics['decode_token_throughput_tps'] or 0.0:.1f} tok/s"
                        )
                finally:
                    hardware_summary = monitor.stop()
                hardware = _actual_hardware_metrics(hardware_summary)

                # Aggregated summary
                request_stats = request_summary([
                    request for record in iter_records for request in record["request_results"]
                ])
                request_rates = [r["request_throughput_rps"] for r in iter_records]
                prompt_lengths = [
                    length for record in iter_records for length in record["trace"]["prompt_lengths"]
                ]
                output_lengths = [
                    length for record in iter_records for length in record["trace"]["output_lengths"]
                ]

                summary_row = {
                    "engine": "sparseengine",
                    "sparse_method": args.sparse_method,
                    "scenario": "fixed_batch",
                    "prompt_len": p_len,
                    "output_len": o_len,
                    "prompt_len_min": min(prompt_lengths),
                    "prompt_len_max": max(prompt_lengths),
                    "output_len_min": min(output_lengths),
                    "output_len_max": max(output_lengths),
                    "concurrency": bs,
                    "request_count": bs,
                    **request_stats,
                    "batch_max_ttft_ms_mean": statistics.fmean(r["batch_max_ttft_ms"] for r in iter_records),
                    "tpot_timing_scope": REQUEST_TPOT_SCOPE,
                    "tpot_concurrency_proxy_tps": _tpot_concurrency_proxy_tps(
                        concurrency=bs,
                        tpot_ms=(
                            request_stats["tpot_ms_mean"]
                        ),
                    ),
                    "tpot_concurrency_proxy_scope": TPOT_CONCURRENCY_PROXY_SCOPE,
                    "request_throughput_rps": statistics.fmean(request_rates),
                    **_mean_phase_throughput_metrics(iter_records),
                    "sequence_replacements": 0,
                    "status": "success",
                    "protocol": protocol,
                    "protocol_label": protocol_label,
                    "decode_metric_status": "success" if request_stats["tpot_request_count"] else "skipped_by_policy",
                    "actual_hardware_metrics": hardware,
                    **{key: value for key, value in hardware.items() if key != "per_gpu"},
                }
                results.append(summary_row)

    operator_runtime_stats = llm.operator_runtime_stats()
    with (output_dir / "operator_runtime_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"status": "success", "world_ranks": operator_runtime_stats},
            handle,
            indent=2,
            ensure_ascii=False,
        )

    if hasattr(llm, "exit"):
        llm.exit()
    return results


def run_sparseengine_churn(
    args: argparse.Namespace,
    model_specs: ModelArchitectureSpecs,
) -> list[dict[str, Any]]:
    import torch
    from sparseengine import LLM, SamplingParams
    from sparseengine.utils.profiler import profiler
    from benchmark.efficiency.metrics import scheduler_phase_metrics

    results: list[dict[str, Any]] = []
    output_dir = Path(args.output_dir)
    raw_samples_file = output_dir / "raw_samples.jsonl"
    base_hyper_params, _sparse_budget, protocol, protocol_label = (
        _resolve_sparse_probe_protocol(args)
    )
    max_len_needed = _sparse_probe_max_model_len(args, base_hyper_params)

    for concurrency in args.batch_sizes:
        engine_kwargs = {
            **base_hyper_params,
            "max_model_len": max_len_needed,
            "sparse_method": args.sparse_method,
            "enable_prefix_caching": False,
            "max_num_seqs_in_batch": _replica_concurrency(concurrency, base_hyper_params),
            "max_decoding_seqs": _replica_concurrency(concurrency, base_hyper_params),
            "max_num_seqs_in_gpu": _resident_sequence_rows(
                concurrency, base_hyper_params
            ),
        }
        print(
            "[SparseEngine Churn] Initializing "
            f"method={args.sparse_method}, max_concurrency={concurrency}..."
        )
        engine_init_started = time.perf_counter()
        llm = LLM(args.model_path, **engine_kwargs)
        engine_init_s = time.perf_counter() - engine_init_started
        startup_graph_summary = llm.debug_sparse_state_summaries()[0][
            "decode_graph"
        ]
        try:
            request_count = concurrency * args.churn_request_multiplier
            for p_len in args.prompt_lens:
                for o_len in args.output_lens:
                    for warmup_index in range(args.num_warmups):
                        warmup_count = min(request_count, max(concurrency, 2 * concurrency))
                        trace = _trace_for_iteration(
                            args,
                            model_specs,
                            scenario="oversubscribed_churn",
                            phase="warmup",
                            prompt_len=p_len,
                            output_len=o_len,
                            concurrency=concurrency,
                            iteration=warmup_index,
                            request_count=warmup_count,
                            vary_output_lengths=True,
                        )
                        for request in trace:
                            llm.add_request(
                                request.prompt_token_ids,
                                SamplingParams(
                                    temperature=0.0,
                                    top_p=1.0,
                                    top_k=1,
                                    ignore_eos=True,
                                    max_tokens=request.output_len,
                                ),
                            )
                        while not llm.is_finished():
                            llm.step()
                    torch.cuda.synchronize()

                    case_name = f"churn-p{p_len}-o{o_len}-c{concurrency}"
                    monitor = _case_monitor(args, case_name)
                    iter_records: list[dict[str, Any]] = []
                    try:
                        for iteration in range(args.num_iters):
                            profiler.reset()
                            graph_before = llm.debug_sparse_state_summaries()[0][
                                "decode_graph"
                            ]
                            trace = _trace_for_iteration(
                                args,
                                model_specs,
                                scenario="oversubscribed_churn",
                                phase="measure",
                                prompt_len=p_len,
                                output_len=o_len,
                                concurrency=concurrency,
                                iteration=iteration,
                                request_count=request_count,
                                vary_output_lengths=True,
                            )
                            seq_to_request: dict[int, Any] = {}
                            arrival_times: dict[int, float] = {}
                            first_token_times: dict[int, float] = {}
                            finished_times: dict[int, float] = {}
                            generated_counts: dict[int, int] = {}
                            started = time.perf_counter()
                            for request in trace:
                                seq_id = int(
                                    llm.add_request(
                                        request.prompt_token_ids,
                                        SamplingParams(
                                            temperature=0.0,
                                            top_p=1.0,
                                            top_k=1,
                                            ignore_eos=True,
                                            max_tokens=request.output_len,
                                        ),
                                    )
                                )
                                if seq_id in seq_to_request:
                                    raise RuntimeError(
                                        f"Duplicate SparseEngine churn sequence ID: {seq_id}."
                                    )
                                seq_to_request[seq_id] = request
                                arrival_times[seq_id] = time.perf_counter()

                            step_count = 0
                            scheduler_steps = []
                            while not llm.is_finished():
                                finished_outputs, _num_tokens = llm.step()
                                now = time.perf_counter()
                                step_count += 1
                                scheduler = getattr(llm, "scheduler", None)
                                if scheduler is not None:
                                    scheduler_steps.append({
                                        "step": step_count,
                                        "elapsed_s": now - started,
                                        **scheduler.last_phase_decision,
                                    })
                                for seq_id, token_ids in getattr(
                                    llm, "last_step_token_outputs", []
                                ):
                                    seq_id = int(seq_id)
                                    if token_ids and seq_id in seq_to_request:
                                        first_token_times.setdefault(seq_id, now)
                                for (
                                    seq_id,
                                    token_ids,
                                    _token_logprobs,
                                    _top_logprobs,
                                ) in finished_outputs:
                                    seq_id = int(seq_id)
                                    if token_ids and seq_id in seq_to_request:
                                        first_token_times.setdefault(seq_id, now)
                                    finished_times[seq_id] = now
                                    generated_counts[seq_id] = len(token_ids)
                            elapsed_s = time.perf_counter() - started
                            graph_after = llm.debug_sparse_state_summaries()[0][
                                "decode_graph"
                            ]

                            expected_seq_ids = set(seq_to_request)
                            for name, observed in (
                                ("first-token", set(first_token_times)),
                                ("completion", set(finished_times)),
                                ("generated-count", set(generated_counts)),
                            ):
                                if observed != expected_seq_ids:
                                    raise RuntimeError(
                                        f"SparseEngine churn {name} coverage mismatch: "
                                        f"missing={sorted(expected_seq_ids - observed)}, "
                                        f"unexpected={sorted(observed - expected_seq_ids)}."
                                    )

                            request_results = []
                            for seq_id, request in seq_to_request.items():
                                generated = generated_counts[seq_id]
                                if generated != request.output_len:
                                    raise RuntimeError(
                                        f"SparseEngine churn seq_id={seq_id} generated "
                                        f"{generated} tokens, expected {request.output_len}."
                                    )
                                first = first_token_times[seq_id]
                                finished = finished_times[seq_id]
                                request_results.append(
                                    {
                                        **request.metadata(),
                                        "seq_id": seq_id,
                                        **request_metrics(
                                            first - arrival_times[seq_id], finished - first, generated
                                        ),
                                        "timing_source": "sparseengine_step_token_publication_no_extra_sync_v1",
                                    }
                                )

                            total_input = sum(request.prompt_len for request in trace)
                            total_output = sum(request.output_len for request in trace)
                            phase_metrics = _phase_throughput_metrics(
                                total_input_tokens=total_input,
                                total_output_tokens=total_output,
                                request_count=request_count,
                                prefill_elapsed_s=max(first_token_times.values()) - started,
                                decode_elapsed_s=(
                                    max(finished_times.values()) - min(first_token_times.values())
                                ),
                            )
                            profiler_snap = profiler.snapshot()
                            record = {
                                "engine": "sparseengine",
                                "sparse_method": args.sparse_method,
                                "scenario": "oversubscribed_churn",
                                "prompt_len": p_len,
                                "output_len": o_len,
                                "concurrency": concurrency,
                                "request_count": request_count,
                                "queued_request_count": request_count - concurrency,
                                "iteration": iteration,
                                "status": "success",
                                "elapsed_s": elapsed_s,
                                "step_count": step_count,
                                "scheduler_steps": scheduler_steps,
                                "scheduler_metrics": scheduler_phase_metrics(scheduler_steps),
                                "engine_init_s": engine_init_s,
                                "startup_decode_cuda_graph": startup_graph_summary,
                                "decode_cuda_graph_before": graph_before,
                                "decode_cuda_graph_after": graph_after,
                                "decode_cuda_graph_counter_delta": (
                                    _decode_graph_counter_delta(graph_before, graph_after)
                                ),
                                "request_throughput_rps": request_count / elapsed_s,
                                "output_token_throughput_tps": total_output / elapsed_s,
                                **phase_metrics,
                                "decode_metric_status": (
                                    "success"
                                    if phase_metrics["decode_token_throughput_tps"] is not None
                                    else "skipped_by_policy"
                                ),
                                "profiler_breakdown": profiler_snap,
                                "profiler_status": "success" if profiler_snap else "skipped_by_policy",
                                "protocol": protocol,
                                "protocol_label": protocol_label,
                                "trace": trace_metadata(trace),
                                "request_results": request_results,
                            }
                            iter_records.append(record)
                            with raw_samples_file.open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            _append_request_samples(
                                output_dir,
                                engine="sparseengine",
                                sparse_method=args.sparse_method,
                                scenario="oversubscribed_churn",
                                prompt_len=p_len,
                                output_len=o_len,
                                concurrency=concurrency,
                                iteration=iteration,
                                requests=request_results,
                            )
                    finally:
                        hardware_summary = monitor.stop()
                    hardware = _actual_hardware_metrics(hardware_summary)

                    request_results = [
                        request
                        for record in iter_records
                        for request in record["request_results"]
                    ]
                    request_stats = request_summary(request_results)
                    prompt_lengths = [
                        length
                        for record in iter_records
                        for length in record["trace"]["prompt_lengths"]
                    ]
                    output_lengths = [
                        length
                        for record in iter_records
                        for length in record["trace"]["output_lengths"]
                    ]
                    results.append(
                        {
                            "engine": "sparseengine",
                            "sparse_method": args.sparse_method,
                            "scenario": "oversubscribed_churn",
                            "prompt_len": p_len,
                            "output_len": o_len,
                            "prompt_len_min": min(prompt_lengths),
                            "prompt_len_max": max(prompt_lengths),
                            "output_len_min": min(output_lengths),
                            "output_len_max": max(output_lengths),
                            "concurrency": concurrency,
                            "request_count": request_count,
                            "sequence_replacements": request_count - concurrency,
                            "request_throughput_rps": statistics.fmean(
                                record["request_throughput_rps"] for record in iter_records
                            ),
                            **_mean_phase_throughput_metrics(iter_records),
                            **request_stats,
                            "tpot_timing_scope": REQUEST_TPOT_SCOPE,
                            "decode_metric_status": (
                                "success" if request_stats["tpot_request_count"] else "skipped_by_policy"
                            ),
                            "status": "success",
                            "protocol": protocol,
                            "protocol_label": protocol_label,
                            "actual_hardware_metrics": hardware,
                            **{key: value for key, value in hardware.items() if key != "per_gpu"},
                        }
                    )
        finally:
            if hasattr(llm, "exit"):
                llm.exit()
    del llm
    return results


def _vllm_engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Allow fork-specific options without overriding the matched workload."""
    matched = {
        "model": args.model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": max(args.prompt_lens) + max(args.output_lens) + 128,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": max(args.batch_sizes),
        "enable_prefix_caching": getattr(args, "enable_prefix_caching", False),
        "disable_log_stats": False,
        "trust_remote_code": True,
        "seed": args.seed,
    }
    extra = _parse_json_arg(getattr(args, "engine_kwargs", "{}"))
    if not isinstance(extra, dict):
        raise ValueError("--engine-kwargs must be a JSON object")
    conflicts = sorted(key for key in extra if key in matched)
    if conflicts:
        raise ValueError(f"--engine-kwargs cannot override matched probe options: {conflicts}")
    scorer = extra.get("compression_scorer")
    budget = extra.get("compression_budget_tokens")
    if not (
        (args.sparse_method == "vanilla" and scorer in (None, "none"))
        or (
            args.sparse_method == "snapkv"
            and scorer == "snapkv"
            and type(budget) is int
            and budget > 0
        )
    ):
        raise ValueError(
            "vLLM method label does not match its explicit compression configuration: "
            "vanilla requires no compression scorer; snapkv requires "
            "compression_scorer='snapkv' and a positive integer compression_budget_tokens."
        )
    return {**matched, **extra}


def run_vllm_probe(
    args: argparse.Namespace,
    model_specs: ModelArchitectureSpecs,
) -> list[dict[str, Any]]:
    from vllm import LLM

    if int(_parse_json_arg(args.engine_kwargs).get("data_parallel_size", 1)) > 1 and args.scenario != "fixed":
        raise ValueError("vLLM DP currently supports fixed request-mode probes only")
    if args.scenario == "churn":
        return run_vllm_churn(args, model_specs)

    kwargs = _vllm_engine_kwargs(args)
    dp_size = int(kwargs.get("data_parallel_size", 1))
    print(f"[vLLM Probe] Initializing vLLM (TP={args.tensor_parallel_size}, DP={dp_size})...")
    if dp_size > 1:
        from benchmark.efficiency.vllm_dp import VLLMDPBatch

        if kwargs.get("data_parallel_size_local", dp_size) != dp_size:
            raise ValueError("The DP request probe currently requires single-node local DP")
        kwargs["max_num_seqs"] = (max(args.batch_sizes) + dp_size - 1) // dp_size
        (Path(args.output_dir) / "dp_engine_config.json").write_text(json.dumps(kwargs, indent=2))
        llm = VLLMDPBatch(kwargs)
    else:
        llm = LLM(**kwargs)
    try:
        return _run_vllm_fixed_probe(args, model_specs, llm, dp_size)
    finally:
        if dp_size > 1:
            llm.close()


def _run_vllm_fixed_probe(args, model_specs, llm, dp_size):
    import torch
    from vllm import SamplingParams

    results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_samples_file = output_dir / "raw_samples.jsonl"

    for p_len in args.prompt_lens:
        for o_len in args.output_lens:
            for bs in args.batch_sizes:
                print(f"\n---> Benchmarking vLLM Sweep: PromptLen={p_len}, OutputLen={o_len}, BatchSize={bs} <---")
                sampling_params = SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    top_k=1,
                    ignore_eos=True,
                    max_tokens=o_len,
                    detokenize=False,
                )

                # Warmup with unique token IDs
                for w_idx in range(args.num_warmups):
                    print(f"  [Warmup {w_idx + 1}/{args.num_warmups}]...")
                    warmup_trace = _trace_for_iteration(
                        args,
                        model_specs,
                        scenario="fixed_batch",
                        phase="warmup",
                        prompt_len=p_len,
                        output_len=o_len,
                        concurrency=bs,
                        iteration=w_idx,
                        request_count=bs,
                        vary_output_lengths=False,
                    )
                    warmup_prompts = [
                        {"prompt_token_ids": request.prompt_token_ids}
                        for request in warmup_trace
                    ]
                    llm.generate(warmup_prompts, sampling_params=sampling_params, use_tqdm=False)

                torch.cuda.synchronize()

                iter_records = []
                case_name = f"fixed-p{p_len}-o{o_len}-c{bs}"
                monitor = _case_monitor(args, case_name)
                try:
                    for it in range(args.num_iters):
                        trace = _trace_for_iteration(
                            args,
                            model_specs,
                            scenario="fixed_batch",
                            phase="measure",
                            prompt_len=p_len,
                            output_len=o_len,
                            concurrency=bs,
                            iteration=it,
                            request_count=bs,
                            vary_output_lengths=False,
                        )
                        prompt_tokens = [
                            {"prompt_token_ids": request.prompt_token_ids}
                            for request in trace
                        ]
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        outputs = llm.generate(
                            prompt_tokens,
                            sampling_params=sampling_params,
                            use_tqdm=False,
                        )
                        torch.cuda.synchronize()
                        elapsed_s = time.perf_counter() - started
                        if dp_size > 1:
                            with (output_dir / "generated_outputs.jsonl").open("a") as f:
                                for index, output in enumerate(outputs):
                                    f.write(json.dumps({
                                        "prompt_len": p_len, "output_len": o_len,
                                        "concurrency": bs, "iteration": it,
                                        "request_id": output.request_id,
                                        "dp_rank": index % dp_size,
                                        "token_ids": [candidate.token_ids for candidate in output.outputs],
                                    }) + "\n")
                        if len(outputs) != bs:
                            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for batch_size={bs}.")

                        ttft_ms, tpot_ms = _vllm_phase_metrics(outputs, o_len)
                        prefill_elapsed_s, decode_elapsed_s = _vllm_batch_phase_seconds(outputs)
                        request_results = []
                        for request, output in zip(trace, outputs):
                            metrics = getattr(output, "metrics", None)
                            if metrics is None or len(output.outputs) != 1:
                                raise RuntimeError(
                                    "vLLM fixed-batch request timing requires one timed candidate."
                                )
                            generated = len(output.outputs[0].token_ids)
                            cached = getattr(output, "num_cached_tokens", None)
                            if args.enable_prefix_caching and cached is None:
                                raise RuntimeError("vLLM did not report prefix-cache hit tokens")
                            if cached is not None and not 0 <= cached < request.prompt_len:
                                raise RuntimeError(f"Invalid vLLM prefix-cache hit count: {cached}")
                            ttft_s, decode_s, timing_source = _vllm_request_phase_seconds(
                                metrics
                            )
                            request_results.append(
                                {
                                    **request.metadata(),
                                    "request_id": str(output.request_id),
                                    "num_cached_tokens": cached,
                                    **request_metrics(ttft_s, decode_s, generated),
                                    "timing_source": timing_source,
                                }
                            )
                        total_input = sum(request.prompt_len for request in trace)
                        total_output = sum(request.output_len for request in trace)
                        phase_metrics = _phase_throughput_metrics(
                            total_input_tokens=total_input,
                            total_output_tokens=total_output,
                            request_count=bs,
                            prefill_elapsed_s=prefill_elapsed_s,
                            decode_elapsed_s=decode_elapsed_s,
                        )
                        rec = {
                            "engine": "vllm",
                            "data_parallel_size": dp_size,
                            "request_adapter": "async_dp_round_robin" if dp_size > 1 else "offline_llm",
                            "sparse_method": args.sparse_method,
                            "scenario": "fixed_batch",
                            "prompt_len": p_len,
                            "output_len": o_len,
                            "concurrency": bs,
                            "iteration": it,
                            "status": "success",
                            "elapsed_s": elapsed_s,
                            "ttft_ms": ttft_ms,
                            "batch_max_ttft_ms": max(row["ttft_ms"] for row in request_results),
                            **request_summary(request_results),
                            "tpot_ms": None if tpot_ms is None else round(tpot_ms, 2),
                            "tpot_timing_scope": REQUEST_TPOT_SCOPE,
                            "tpot_concurrency_proxy_tps": _tpot_concurrency_proxy_tps(
                                concurrency=bs,
                                tpot_ms=tpot_ms,
                            ),
                            "tpot_concurrency_proxy_scope": TPOT_CONCURRENCY_PROXY_SCOPE,
                            "request_throughput_rps": bs / elapsed_s,
                            "output_token_throughput_tps": total_output / elapsed_s,
                            **phase_metrics,
                            "decode_metric_status": "success" if tpot_ms is not None else "skipped_by_policy",
                            "protocol_label": getattr(args, "backend_label", None) or f"vllm-{args.sparse_method}",
                            "trace": trace_metadata(trace, allow_duplicate_prompts=getattr(args, "shared_prompt", False)),
                            "request_results": request_results,
                        }
                        iter_records.append(rec)
                        with open(raw_samples_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        _append_request_samples(
                            output_dir,
                            engine="vllm",
                            sparse_method=args.sparse_method,
                            scenario="fixed_batch",
                            prompt_len=p_len,
                            output_len=o_len,
                            concurrency=bs,
                            iteration=it,
                            requests=request_results,
                        )

                        print(
                            f"  Iter {it + 1}/{args.num_iters}: TTFT={ttft_ms:.1f}ms | "
                            f"TPOT={tpot_ms if tpot_ms is not None else 'n/a'}ms | "
                            f"FirstTokenWindow={phase_metrics['prefill_token_throughput_tps']:.1f} tok/s | "
                            f"BatchDecodeWindow={phase_metrics['decode_token_throughput_tps'] or 0.0:.1f} tok/s"
                        )
                finally:
                    hardware_summary = monitor.stop()
                hardware = _actual_hardware_metrics(hardware_summary)

                request_stats = request_summary([
                    request for record in iter_records for request in record["request_results"]
                ])
                request_rates = [r["request_throughput_rps"] for r in iter_records]
                prompt_lengths = [
                    length for record in iter_records for length in record["trace"]["prompt_lengths"]
                ]
                output_lengths = [
                    length for record in iter_records for length in record["trace"]["output_lengths"]
                ]

                summary_row = {
                    "engine": "vllm",
                    "data_parallel_size": dp_size,
                    "request_adapter": "async_dp_round_robin" if dp_size > 1 else "offline_llm",
                    "sparse_method": args.sparse_method,
                    "scenario": "fixed_batch",
                    "prompt_len": p_len,
                    "output_len": o_len,
                    "prompt_len_min": min(prompt_lengths),
                    "prompt_len_max": max(prompt_lengths),
                    "output_len_min": min(output_lengths),
                    "output_len_max": max(output_lengths),
                    "concurrency": bs,
                    "request_count": bs,
                    **request_stats,
                    "batch_max_ttft_ms_mean": statistics.fmean(r["batch_max_ttft_ms"] for r in iter_records),
                    "tpot_timing_scope": REQUEST_TPOT_SCOPE,
                    "tpot_concurrency_proxy_tps": _tpot_concurrency_proxy_tps(
                        concurrency=bs,
                        tpot_ms=(
                            request_stats["tpot_ms_mean"]
                        ),
                    ),
                    "tpot_concurrency_proxy_scope": TPOT_CONCURRENCY_PROXY_SCOPE,
                    "request_throughput_rps": statistics.fmean(request_rates),
                    **_mean_phase_throughput_metrics(iter_records),
                    "sequence_replacements": 0,
                    "status": "success",
                    "decode_metric_status": "success" if request_stats["tpot_request_count"] else "skipped_by_policy",
                    "protocol_label": getattr(args, "backend_label", None) or f"vllm-{args.sparse_method}",
                    "actual_hardware_metrics": hardware,
                    **{key: value for key, value in hardware.items() if key != "per_gpu"},
                }
                results.append(summary_row)

    del llm
    return results


def run_vllm_churn(
    args: argparse.Namespace,
    model_specs: ModelArchitectureSpecs,
) -> list[dict[str, Any]]:
    import torch
    from vllm import LLM, SamplingParams

    results: list[dict[str, Any]] = []
    output_dir = Path(args.output_dir)
    raw_samples_file = output_dir / "raw_samples.jsonl"
    max_len_needed = max(args.prompt_lens) + max(args.output_lens) + 128

    for concurrency in args.batch_sizes:
        print(f"[vLLM Churn] Initializing max_concurrency={concurrency}...")
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=max_len_needed,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=concurrency,
            enable_prefix_caching=False,
            disable_log_stats=False,
            trust_remote_code=True,
            seed=args.seed,
        )
        try:
            request_count = concurrency * args.churn_request_multiplier
            for p_len in args.prompt_lens:
                for o_len in args.output_lens:
                    for warmup_index in range(args.num_warmups):
                        warmup_count = min(request_count, 2 * concurrency)
                        trace = _trace_for_iteration(
                            args,
                            model_specs,
                            scenario="oversubscribed_churn",
                            phase="warmup",
                            prompt_len=p_len,
                            output_len=o_len,
                            concurrency=concurrency,
                            iteration=warmup_index,
                            request_count=warmup_count,
                            vary_output_lengths=True,
                        )
                        prompts = [
                            {"prompt_token_ids": request.prompt_token_ids}
                            for request in trace
                        ]
                        sampling = [
                            SamplingParams(
                                temperature=0.0,
                                top_p=1.0,
                                top_k=1,
                                ignore_eos=True,
                                max_tokens=request.output_len,
                                detokenize=False,
                            )
                            for request in trace
                        ]
                        outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
                        if len(outputs) != warmup_count:
                            raise RuntimeError(
                                f"vLLM churn warmup returned {len(outputs)} outputs, "
                                f"expected {warmup_count}."
                            )
                    torch.cuda.synchronize()

                    case_name = f"churn-p{p_len}-o{o_len}-c{concurrency}"
                    monitor = _case_monitor(args, case_name)
                    iter_records: list[dict[str, Any]] = []
                    try:
                        for iteration in range(args.num_iters):
                            trace = _trace_for_iteration(
                                args,
                                model_specs,
                                scenario="oversubscribed_churn",
                                phase="measure",
                                prompt_len=p_len,
                                output_len=o_len,
                                concurrency=concurrency,
                                iteration=iteration,
                                request_count=request_count,
                                vary_output_lengths=True,
                            )
                            prompts = [
                                {"prompt_token_ids": request.prompt_token_ids}
                                for request in trace
                            ]
                            sampling = [
                                SamplingParams(
                                    temperature=0.0,
                                    top_p=1.0,
                                    top_k=1,
                                    ignore_eos=True,
                                    max_tokens=request.output_len,
                                    detokenize=False,
                                )
                                for request in trace
                            ]
                            torch.cuda.synchronize()
                            started = time.perf_counter()
                            outputs = llm.generate(
                                prompts,
                                sampling_params=sampling,
                                use_tqdm=False,
                            )
                            torch.cuda.synchronize()
                            elapsed_s = time.perf_counter() - started
                            if len(outputs) != request_count:
                                raise RuntimeError(
                                    f"vLLM churn returned {len(outputs)} outputs, "
                                    f"expected {request_count}."
                                )

                            request_results = []
                            for request, output in zip(trace, outputs):
                                metrics = getattr(output, "metrics", None)
                                if metrics is None:
                                    raise RuntimeError(
                                        "vLLM churn requires timing metrics for every request."
                                    )
                                if len(output.outputs) != 1:
                                    raise RuntimeError(
                                        f"vLLM churn request {request.request_index} returned "
                                        f"{len(output.outputs)} candidates, expected one."
                                    )
                                generated = len(output.outputs[0].token_ids)
                                if generated != request.output_len:
                                    raise RuntimeError(
                                        f"vLLM churn request {request.request_index} generated "
                                        f"{generated} tokens, expected {request.output_len}."
                                    )
                                ttft_s, decode_s, timing_source = (
                                    _vllm_request_phase_seconds(metrics)
                                )
                                request_results.append(
                                    {
                                        **request.metadata(),
                                        "request_id": str(output.request_id),
                                        **request_metrics(ttft_s, decode_s, generated),
                                        "timing_source": timing_source,
                                        "generated_tokens": generated,
                                    }
                                )

                            total_input = sum(request.prompt_len for request in trace)
                            total_output = sum(request.output_len for request in trace)
                            prefill_elapsed_s, decode_elapsed_s = _vllm_batch_phase_seconds(
                                outputs
                            )
                            phase_metrics = _phase_throughput_metrics(
                                total_input_tokens=total_input,
                                total_output_tokens=total_output,
                                request_count=request_count,
                                prefill_elapsed_s=prefill_elapsed_s,
                                decode_elapsed_s=decode_elapsed_s,
                            )
                            record = {
                                "engine": "vllm",
                                "sparse_method": "vanilla",
                                "scenario": "oversubscribed_churn",
                                "prompt_len": p_len,
                                "output_len": o_len,
                                "concurrency": concurrency,
                                "request_count": request_count,
                                "queued_request_count": request_count - concurrency,
                                "iteration": iteration,
                                "status": "success",
                                "elapsed_s": elapsed_s,
                                "request_throughput_rps": request_count / elapsed_s,
                                "output_token_throughput_tps": total_output / elapsed_s,
                                **phase_metrics,
                                "decode_metric_status": (
                                    "success"
                                    if phase_metrics["decode_token_throughput_tps"] is not None
                                    else "skipped_by_policy"
                                ),
                                "protocol_label": "vllm-vanilla",
                                "trace": trace_metadata(trace),
                                "request_results": request_results,
                            }
                            iter_records.append(record)
                            with raw_samples_file.open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            _append_request_samples(
                                output_dir,
                                engine="vllm",
                                sparse_method="vanilla",
                                scenario="oversubscribed_churn",
                                prompt_len=p_len,
                                output_len=o_len,
                                concurrency=concurrency,
                                iteration=iteration,
                                requests=request_results,
                            )
                    finally:
                        hardware_summary = monitor.stop()
                    hardware = _actual_hardware_metrics(hardware_summary)

                    request_results = [
                        request
                        for record in iter_records
                        for request in record["request_results"]
                    ]
                    request_stats = request_summary(request_results)
                    prompt_lengths = [
                        length
                        for record in iter_records
                        for length in record["trace"]["prompt_lengths"]
                    ]
                    output_lengths = [
                        length
                        for record in iter_records
                        for length in record["trace"]["output_lengths"]
                    ]
                    results.append(
                        {
                            "engine": "vllm",
                            "sparse_method": "vanilla",
                            "scenario": "oversubscribed_churn",
                            "prompt_len": p_len,
                            "output_len": o_len,
                            "prompt_len_min": min(prompt_lengths),
                            "prompt_len_max": max(prompt_lengths),
                            "output_len_min": min(output_lengths),
                            "output_len_max": max(output_lengths),
                            "concurrency": concurrency,
                            "request_count": request_count,
                            "sequence_replacements": request_count - concurrency,
                            "request_throughput_rps": statistics.fmean(
                                record["request_throughput_rps"] for record in iter_records
                            ),
                            **_mean_phase_throughput_metrics(iter_records),
                            **request_stats,
                            "tpot_timing_scope": REQUEST_TPOT_SCOPE,
                            "decode_metric_status": (
                                "success" if request_stats["tpot_request_count"] else "skipped_by_policy"
                            ),
                            "status": "success",
                            "protocol_label": "vllm-vanilla",
                            "actual_hardware_metrics": hardware,
                            **{key: value for key, value in hardware.items() if key != "per_gpu"},
                        }
                    )
        finally:
            del llm
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Standardized Synthetic Length Sweep & Efficiency Probe.")
    parser.add_argument("--engine", type=str, choices=["sparseengine", "vllm", "hisparse", "vortex"], default="sparseengine")
    parser.add_argument("--model-path", type=str, required=True, help="Model path or HF name")
    parser.add_argument("--sparse-method", type=str, default="vanilla",
                        help="Sparse method name; fixed-batch vLLM accepts vanilla or explicitly configured snapkv.")
    parser.add_argument("--prompt-lens", type=_parse_ints, default=[8192, 16384, 32768])
    parser.add_argument("--output-lens", type=_parse_ints, default=[128])
    parser.add_argument("--batch-sizes", type=_parse_ints, default=[1])
    parser.add_argument(
        "--scenario",
        choices=["fixed", "churn", "all"],
        default="all",
        help="fixed measures one variable-length batch; churn oversubscribes the scheduler.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-prefix-caching", action="store_true",
                        help="Enable prefix caching in fixed request-mode probes.")
    parser.add_argument("--shared-prompt", action="store_true",
                        help="Use identical prompts within each fixed workload; fresh prompts each iteration.")
    parser.add_argument(
        "--prompt-length-jitter",
        type=float,
        default=0.10,
        help="Fraction below each requested prompt length used for variable-length batches.",
    )
    parser.add_argument(
        "--output-length-jitter",
        type=float,
        default=0.25,
        help="Fraction below each requested output length used by churn traces.",
    )
    parser.add_argument(
        "--churn-request-multiplier",
        type=int,
        default=4,
        help="Churn request count is max concurrency times this multiplier.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument(
        "--expert-parallel-size",
        type=int,
        default=1,
        help="SparseEngine expert parallel size.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=65536,
        help="Matched scheduler token budget used by SparseEngine and vLLM.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--prefill-wave-size", type=int, default=0,
                        help="Native fixed H2O only: admit next wave after prior first tokens/final-prefill pruning; retain workload-start arrival timestamps.")
    parser.add_argument("--num-warmups", type=int, default=1)
    parser.add_argument("--decode-only-steps", type=int, default=0,
                        help="Paper decode window: shared native/vLLM/HiSparse adapters, CUDA sync only at drained window edges.")
    parser.add_argument("--wave-decode-gap-steps", type=int, default=0,
                        help="Native paper-window wave admission decode gap; not used by request probes.")
    parser.add_argument("--decode-only-warmup-steps", type=int, default=8)
    parser.add_argument("--num-iters", type=int, default=3)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--hyper-params", type=str, default="{}")
    parser.add_argument("--engine-kwargs", type=str, default="{}",
                        help="vLLM/fork constructor options as JSON or @file; matched workload options cannot be overridden.")
    parser.add_argument("--backend-label", default=None,
                        help="Display label for a fixed-batch vLLM fork; does not select an engine or algorithm.")
    parser.add_argument(
        "--allow-single-omnikv-full-layer",
        action="store_true",
        help="Allow an explicit single-full-layer OmniKV ablation.",
    )
    parser.add_argument(
        "--sparse-prefill-score-mode",
        choices=["probability", "logits"],
        default=None,
        help="Required for SnapKV probes so probability and logits results cannot be conflated.",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default=None,
        help="Deprecated compatibility label; theoretical hardware profiles are no longer used.",
    )
    parser.add_argument(
        "--monitor-gpus",
        default=None,
        help="Physical GPU IDs for directly sampled metrics; defaults to CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--hardware-sampling-interval-ms",
        type=int,
        default=100,
        help="nvidia-smi activity sampling interval for each measured case.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.enable_prefix_caching or args.shared_prompt:
        if args.scenario != "fixed" or args.decode_only_steps or args.engine not in {"sparseengine", "vllm"}:
            raise ValueError("Prefix reuse probes require native/vLLM fixed request mode")
        if args.prompt_length_jitter or args.prefill_wave_size:
            raise ValueError("Prefix reuse probes require zero prompt jitter and no prefill waves")
        if args.engine == "vllm" and int(_parse_json_arg(args.engine_kwargs).get("data_parallel_size", 1)) > 1:
            raise ValueError("Prefix reuse probes currently require DP1")
    if args.decode_only_steps:
        if args.engine == "vllm" and int(
            _parse_json_arg(args.engine_kwargs).get("data_parallel_size", 1)
        ) > 1:
            raise ValueError("vLLM DP currently supports fixed request-mode probes only")
        if args.engine == "sparseengine" and int(
            _parse_json_arg(args.hyper_params).get("data_parallel_size", 1)
        ) > 1:
            raise ValueError("DP attention currently supports request-mode probes only")
        from benchmark.efficiency.paper import run_paper_decode
        run_paper_decode(args)
        return
    if args.engine in ("hisparse", "vortex") or args.wave_decode_gap_steps:
        raise ValueError("HiSparse and wave-decode-gap-steps currently require explicit decode-only mode")
    if args.engine != "vllm" and _parse_json_arg(args.engine_kwargs):
        raise ValueError("--engine-kwargs is supported only by the vLLM adapter")
    if args.engine == "vllm" and args.scenario != "fixed" and _parse_json_arg(args.engine_kwargs):
        raise ValueError("Fork-specific --engine-kwargs currently require --scenario fixed")
    if args.backend_label and (args.engine != "vllm" or args.scenario != "fixed"):
        raise ValueError("--backend-label requires fixed-batch vLLM")
    if args.decode_only_steps < 0 or args.decode_only_warmup_steps < 1:
        raise ValueError("Invalid decode-only window length or warmup")
    if args.prefill_wave_size < 0 or (
        args.prefill_wave_size
        and (args.engine != "sparseengine" or args.sparse_method != "h2o" or args.scenario != "fixed")
    ):
        raise ValueError("--prefill-wave-size requires native H2O with --scenario fixed")
    for name in ("prompt_lens", "output_lens", "batch_sizes"):
        values = getattr(args, name)
        if not values or any(int(value) <= 0 for value in values):
            raise ValueError(f"--{name.replace('_', '-')} requires positive integers, got {values}.")
    if (
        args.tensor_parallel_size <= 0
        or args.expert_parallel_size <= 0
        or args.max_num_batched_tokens <= 0
        or args.num_warmups < 0
        or args.num_iters <= 0
    ):
        raise ValueError(
            "tensor parallel size, expert parallel size, max batched tokens, "
            "and iterations must be positive, "
            "and warmups must be non-negative."
        )
    if not 0.0 <= args.prompt_length_jitter < 1.0:
        raise ValueError("--prompt-length-jitter must be in [0, 1).")
    if not 0.0 <= args.output_length_jitter < 1.0:
        raise ValueError("--output-length-jitter must be in [0, 1).")
    if args.churn_request_multiplier < 2:
        raise ValueError("--churn-request-multiplier must be at least 2.")
    if args.hardware_sampling_interval_ms < 50:
        raise ValueError("--hardware-sampling-interval-ms must be at least 50.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_names = (
        "run_manifest.json",
        "raw_samples.jsonl",
        "request_samples.jsonl",
        "summary.json",
        "comparison_report.md",
        "run_status.json",
        "operator_runtime_stats.json",
    )
    collisions = [name for name in artifact_names if (output_dir / name).exists()]
    if collisions:
        raise FileExistsError(
            f"Refusing to mix benchmark runs in {output_dir}: existing artifacts={collisions}. "
            "Use a unique --output-dir."
        )
    (output_dir / "raw_samples.jsonl").write_text("", encoding="utf-8")
    (output_dir / "request_samples.jsonl").write_text("", encoding="utf-8")

    try:
        model_specs = ModelArchitectureSpecs.from_model_path_or_name(args.model_path)
        monitor_gpus = _monitor_gpu_ids(args.monitor_gpus)
        runtime_environment = _runtime_environment_metadata(monitor_gpus)
    except Exception as exc:
        failure = {
            "status": "metric_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "manifest_version": "1.0",
                    "timestamp": datetime.now().isoformat(),
                    "command": [sys.executable, *sys.argv],
                    "git": _git_metadata(),
                    "args": vars(args),
                    **failure,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump({"status": "metric_failed", "summary": [], "error": repr(exc)}, f, indent=2)
        with open(output_dir / "run_status.json", "w", encoding="utf-8") as f:
            json.dump(failure, f, indent=2)
        raise

    # 1. Save Run Manifest
    manifest = {
        "manifest_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "command": [sys.executable, *sys.argv],
        "git": _git_metadata(),
        "args": vars(args),
        "environment": {
            key: os.environ[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "NCCL_DEBUG",
                "VLLM_USE_V1",
                "PROFILER_SENGINE",
                "SPARSEENGINE_SYNC_DEVICE",
            )
            if key in os.environ
        },
        "runtime_environment": runtime_environment,
        "hardware_metrics": {
            "source": "nvidia-smi sampled activity",
            "physical_gpu_ids": monitor_gpus,
            "sampling_interval_ms": args.hardware_sampling_interval_ms,
            "deprecated_hardware_profile_label": args.hardware,
            "theoretical_mfu_mbu_enabled": False,
        },
        "workload": {
            "request_metric_contract": REQUEST_METRIC_CONTRACT,
            "stage_metrics_status": "not_measured",
            "trace_generator_version": TRACE_GENERATOR_VERSION,
            "prefix_caching_enabled": args.enable_prefix_caching,
            "shared_prompt_within_workload": args.shared_prompt,
            "cross_engine_trace_contract": "same seed, token IDs, and per-request lengths",
            "iteration_prompt_reuse_allowed": False,
            "phase_throughput_contract": (
                "diagnostic windows only, not execution stages; "
                "first-token window: prompt tokens / maximum arrival-to-first-token "
                "duration (SparseEngine churn: submission-to-last-first-token); "
                "batch decode window: generated tokens after each first token / "
                "first-first-token-to-last-completion window; request TPOT: mean "
                "per-request (finish-first)/(generated-1); TPOT-equivalent concurrent "
                "proxy: concurrency*1000/request_tpot_ms"
            ),
        },
        "model_specs": {
            "hidden_size": model_specs.hidden_size,
            "num_hidden_layers": model_specs.num_hidden_layers,
            "is_moe": model_specs.is_moe,
            "num_experts": model_specs.num_experts,
            "num_experts_per_tok": model_specs.num_experts_per_tok,
            "num_attention_heads": model_specs.num_attention_heads,
            "num_key_value_heads": model_specs.num_key_value_heads,
            "head_dim": model_specs.head_dim,
            "vocab_size": model_specs.vocab_size,
        },
        "status": "running",
    }
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 2. Run Benchmarks
    try:
        scenarios = ["fixed", "churn"] if args.scenario == "all" else [args.scenario]
        summary_rows = []
        requested_scenario = args.scenario
        for scenario in scenarios:
            args.scenario = scenario
            if args.engine == "sparseengine":
                summary_rows.extend(run_sparseengine_probe(args, model_specs))
            else:
                summary_rows.extend(run_vllm_probe(args, model_specs))
        args.scenario = requested_scenario
        _attach_churn_comparisons(summary_rows)
    except Exception as exc:
        failure_status = "metric_failed" if isinstance(exc, HardwareMetricError) else "model_failed"
        failure = {
            "status": failure_status,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        manifest.update(failure)
        with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump({"status": failure_status, "summary": [], "error": repr(exc)}, f, indent=2)
        with open(output_dir / "run_status.json", "w", encoding="utf-8") as f:
            json.dump(failure, f, indent=2)
        raise

    # 3. Save Summary & Markdown Report
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"status": "success", "summary": summary_rows}, f, indent=2, ensure_ascii=False)

    md_report = _format_markdown_report(
        summary_rows,
        Path(args.model_path).name,
        args.tensor_parallel_size,
        args.expert_parallel_size,
    )
    with open(output_dir / "comparison_report.md", "w", encoding="utf-8") as f:
        f.write(md_report)

    manifest["status"] = "success"
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(output_dir / "run_status.json", "w", encoding="utf-8") as f:
        json.dump({"status": "success"}, f, indent=2)

    print("\n" + "=" * 100)
    print(md_report)
    print("=" * 100)
    print(f"\n[Bench Probe] Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
