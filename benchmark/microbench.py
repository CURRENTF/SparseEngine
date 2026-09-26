import os
import torch
import argparse
import multiprocessing as mp
import traceback
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
src_path = str(REPO_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from benchmark.efficiency.metrics import PipelinedDecodeWindow, decode_window_fields, stage_throughput
from sparseengine.method_registry import (
    CANONICAL_SPARSE_METHODS,
    PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    get_default_prefill_schedule_policy,
    is_decode_cuda_graph_supported,
    is_tp_decode_cuda_graph_supported,
    normalize_sparse_method,
)


DEFAULT_ALL_CHUNKED_PREFILL_SIZE = 96 * 1024
DEFAULT_LONG_PREFILL_OFFLOAD_THRESHOLD = 64 * 1024


def get_peak_memory():
    return torch.cuda.max_memory_allocated() / (1024 ** 3) # GB


def _load_json_arg(value: str) -> dict[str, Any]:
    """Load a JSON object from a CLI arg.

    Supports:
      - Inline JSON: '{"gpu_memory_utilization": 0.9}'
      - File JSON: '@config.json'
    """
    if value is None:
        return {}
    value = str(value).strip()
    if value.startswith("@"):
        path = Path(value[1:]).expanduser()
        value = path.read_text(encoding="utf-8")

    try:
        parsed = json.loads(value)
    except Exception as e:
        raise ValueError(f"Invalid JSON for --hyper_params: {e}") from e

    if not isinstance(parsed, dict):
        raise ValueError("--hyper_params must be a JSON object (dict).")
    return parsed


def _build_engine_hyper_params(args) -> dict[str, Any]:
    # Keep benchmark defaults stable (do not rely on sparseengine.Config defaults).
    hyper_params: dict[str, Any] = {
        "decode_graph": True,
        "gpu_memory_utilization": 0.8,
        "tensor_parallel_size": 1,
    }

    hyper_params.update(_load_json_arg(args.hyper_params))
    if getattr(args, "engine", "sparseengine") == "sparseengine":
        from sparseengine.config import Config
        hyper_params.setdefault("decode_reservation_tokens", Config.decode_reservation_tokens)

    return hyper_params


def _write_jsonl(path: str | None, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False, sort_keys=True, default=_json_default)
            handle.write("\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False, sort_keys=True, default=_json_default)
            handle.write("\n")


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def _git_metadata() -> dict[str, Any]:
    return {
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
    }


def _selected_env_snapshot() -> dict[str, str]:
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "PYTHONPATH",
        "SENGINE_BENCHMARK_OUTPUT_DIR",
        "SENGINE_BENCHMARK_DATA_DIR",
        "SPARSEENGINE_OUTPUT_DIR",
        "SPARSEENGINE_DATA_DIR",
        "SPARSEENGINE_FAKE_ATTENTION",
        "SPARSEENGINE_FAKE_PREFILL_ATTENTION",
        "SPARSEENGINE_FAKE_DECODE_ATTENTION",
        "SPARSEENGINE_FAKE_ATTENTION_MODE",
        "SPARSEENGINE_ALLOW_FAKE_ATTENTION",
        "SPARSEENGINE_LONG_PREFILL_OFFLOAD_MIN_TOKENS",
        "SPARSEENGINE_RAWKV_BUFFER_MODE",
        "SPARSEENGINE_RAWKV_PREFETCH",
    ]
    return {key: os.environ[key] for key in keys if key in os.environ}


def _standard_status(status: Any) -> str:
    normalized = str(status or "unknown").strip().upper()
    if normalized == "SUCCESS":
        return "success"
    if normalized == "SKIPPED_BY_POLICY":
        return "skipped_by_policy"
    if normalized in {"FAILED", "OOM"}:
        return "model_failed"
    return "model_failed"


def _benchmark_sparse_method(method: str) -> str:
    """Resolve a benchmark label to the runtime sparse method without fallback."""
    runtime_method = str(method).strip().lower()
    normalized = normalize_sparse_method(runtime_method)
    if normalized not in CANONICAL_SPARSE_METHODS:
        raise ValueError(f"Unsupported benchmark sparse method: {method!r}.")
    if runtime_method in {
        "deltakv-less-memory-cudagraph",
        "deltakv_less_memory_cudagraph",
    }:
        return runtime_method
    return "vanilla" if normalized == "" else normalized


def _apply_prefill_policy_defaults(
    hyper_params: dict[str, Any],
    method: str,
) -> None:
    runtime_method = _benchmark_sparse_method(method)
    policy = get_default_prefill_schedule_policy(runtime_method)
    if policy == PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH:
        hyper_params.setdefault(
            "long_prefill_offload_threshold",
            DEFAULT_LONG_PREFILL_OFFLOAD_THRESHOLD,
        )
        boundary = hyper_params["long_prefill_offload_threshold"]
        chunk_size = hyper_params.get("engine_prefill_chunk_size", boundary)
        if (
            not isinstance(boundary, int)
            or isinstance(boundary, bool)
            or boundary <= 0
        ):
            raise ValueError(
                "long_prefill_offload_threshold must be a positive integer."
            )
        if (
            not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or chunk_size <= 0
        ):
            raise ValueError("engine_prefill_chunk_size must be a positive integer.")
        if chunk_size > boundary:
            raise ValueError(
                f"sparse_method={runtime_method!r} uses {policy}; require "
                "engine_prefill_chunk_size <= long_prefill_offload_threshold, "
                f"got {chunk_size} > {boundary}."
            )
    else:
        hyper_params.setdefault(
            "engine_prefill_chunk_size",
            DEFAULT_ALL_CHUNKED_PREFILL_SIZE,
        )
        hyper_params.pop("long_prefill_offload_threshold", None)
        chunk_size = hyper_params["engine_prefill_chunk_size"]
        if (
            not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or chunk_size <= 0
        ):
            raise ValueError("engine_prefill_chunk_size must be a positive integer.")


def _record_child_exit_failure(
    results_dict,
    *,
    method: str,
    length: int,
    batch_size: int,
    exitcode: int | None,
    synchronize_step_timing: bool,
) -> None:
    if exitcode in (None, 0):
        return
    case_key = (method, length, batch_size)
    row = dict(results_dict.get(case_key) or {})
    partial_status = row.get("status")
    row.setdefault("method", method)
    row.setdefault("sparse_method", normalize_sparse_method(method))
    row.setdefault("length", int(length))
    row.setdefault("batch_size", int(batch_size))
    row.setdefault("synchronize_step_timing", bool(synchronize_step_timing))
    row.setdefault("error", f"benchmark child exited with code {exitcode}")
    row["status"] = "FAILED"
    row["child_exitcode"] = int(exitcode)
    if partial_status is not None:
        row["child_partial_status"] = str(partial_status)
    results_dict[case_key] = row


def _artifact_records(args, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        record = dict(row)
        record["case_id"] = idx
        record["benchmark"] = "microbench"
        record["raw_status"] = row.get("status", "UNKNOWN")
        record["status"] = _standard_status(row.get("status"))
        record.setdefault("prompt_tokens", row.get("length"))
        record.setdefault("max_new_tokens", int(args.output_len))
        record.setdefault("sampling_temperature", float(args.temperature))
        record.setdefault("sampling_top_p", float(args.top_p))
        record.setdefault(
            "synchronize_step_timing",
            bool(getattr(args, "synchronize_step_timing", False)),
        )
        if "prefill_tp" in row:
            record.setdefault("prefill_tok_s", row["prefill_tp"])
        if "decode_tp" in row:
            record.setdefault("decode_tok_s", row["decode_tp"])
        if "ttft" in row:
            record.setdefault("ttft_s", row["ttft"])
        if "itl" in row:
            record.setdefault("itl_ms", row["itl"])
        if "mem" in row:
            record.setdefault("peak_memory_gb", row["mem"])
        records.append(record)
    return records


def _write_output_dir(args, rows: list[dict[str, Any]]) -> None:
    if not args.output_dir:
        return

    output_dir = Path(args.output_dir).expanduser()
    records = _artifact_records(args, rows)
    success_records = [row for row in records if row["status"] == "success"]
    skipped_records = [row for row in records if row["status"] == "skipped_by_policy"]
    failed_records = [row for row in records if row["status"] not in {"success", "skipped_by_policy"}]

    run_info = {
        **_git_metadata(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "model_path": args.model_path,
        "methods": [part.strip() for part in args.methods.split(",") if part.strip()],
        "lengths": [int(part) for part in args.lengths.split(",") if part.strip()],
        "batch_sizes": [int(part) for part in args.batch_sizes.split(",") if part.strip()],
        "output_len": int(args.output_len),
        "decode_window_steps": int(getattr(args, "decode_window_steps", 0)),
        "decode_warmup_steps_after_full": int(getattr(args, "decode_warmup_steps_after_full", 0)),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "synchronize_step_timing": bool(
            getattr(args, "synchronize_step_timing", False)
        ),
        "hyper_params": args.hyper_params_dict,
        "env": _selected_env_snapshot(),
    }
    aggregate_status = "success"
    if failed_records:
        aggregate_status = "model_failed"
    elif skipped_records:
        aggregate_status = "skipped_by_policy"

    aggregate = {
        "benchmark": "microbench",
        "status": aggregate_status,
        "num_cases": len(records),
        "success_cases": len(success_records),
        "skipped_cases": len(skipped_records),
        "failed_cases": len(failed_records),
        "synchronize_step_timing": bool(
            getattr(args, "synchronize_step_timing", False)
        ),
        "records": records,
    }

    report_lines = [
        "# SparseEngine Microbenchmark",
        "",
        f"- Model: `{args.model_path}`",
        f"- Methods: `{args.methods}`",
        f"- Lengths: `{args.lengths}`",
        f"- Batch sizes: `{args.batch_sizes}`",
        f"- Output length: `{args.output_len}`",
        "",
        "| Method | Prompt tokens | Batch | Status | First observed token s | Prefill step tok/s | Decode "
        + ("window" if getattr(args, "decode_window_steps", 0) else "step")
        + " tok/s | Peak GB | Decode speedup |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        ok = record["status"] == "success"
        speedup = record.get("speedup_vs_vanilla_decode")
        report_lines.append(
            "| {method} | {prompt} | {batch} | {status} | {ttft} | {prefill} | {decode} | {mem} | {speedup} |".format(
                method=record.get("method", ""),
                prompt=record.get("prompt_tokens", ""),
                batch=record.get("batch_size", ""),
                status=record["status"],
                ttft=_format_metric(record.get('ttft_s'), 3) if ok else "",
                prefill=_format_metric(record.get('prefill_tok_s'), 1) if ok else "",
                decode=_format_metric(record.get('decode_tok_s'), 1) if ok else "",
                mem=_format_metric(record.get('peak_memory_gb'), 2) if ok else "",
                speedup=("n/a" if speedup is None else f"{float(speedup):.2f}") if ok else "",
            )
        )

    _write_json(output_dir / "run_info.json", run_info)
    _write_jsonl_rows(output_dir / "performance.jsonl", records)
    _write_jsonl_rows(output_dir / "per_sample_results.jsonl", records)
    _write_json(output_dir / "aggregate_metrics.json", aggregate)
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def _format_metric(value, precision):
    return "n/a" if value is None else f"{value:.{precision}f}"


def _decode_cuda_graph_status(llm) -> dict[str, Any]:
    runner = getattr(getattr(llm, "model_runner", None), "decode_graph_runner", None)
    states = getattr(runner, "_graphs", {}) if runner is not None else {}
    graph_count = sum(
        1
        for state in states.values()
        if getattr(state, "graph", None) is not None
    )
    configured = bool(getattr(getattr(llm, "config", None), "decode_graph", False))
    return {
        "decode_graph_configured": configured,
        "decode_graph_runner_initialized": runner is not None,
        "decode_graph_state_count": int(len(states)),
        "decode_graph_graph_count": int(graph_count),
        "decode_graph_capture_count": int(
            getattr(runner, "capture_count", 0)
        ),
        "decode_graph_replay_count": int(
            getattr(runner, "replay_count", 0)
        ),
        "decode_graph_eager_static_count": int(
            getattr(runner, "eager_static_count", 0)
        ),
        "decode_graph_force_eager_count": int(
            getattr(runner, "force_eager_count", 0)
        ),
        "decode_graph_last_state_key": str(getattr(runner, "last_state_key", None)) if runner is not None else None,
        "decode_graph_active": bool(configured and graph_count > 0),
    }


def _jsonable_config_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable_config_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_config_value(item) for key, item in value.items()}
    return _json_default(value)


def _resolved_engine_config(llm) -> dict[str, Any]:
    config = getattr(llm, "config", None)
    if config is None:
        return {}
    keys = (
        "sparse_method",
        "prefill_schedule_policy",
        "engine_prefill_chunk_size",
        "long_prefill_offload_threshold",
        "decode_graph",
        "decode_graph_capture_sampling",
        "deltakv_sparse_decode_backend",
        "deltakv_triton_materialize_block_tokens",
        "deltakv_triton_gather_heads_per_program",
        "deltakv_triton_reconstruct_heads_per_program",
        "full_layer_kv_quant_bits",
        "deltakv_latent_quant_bits",
        "deltakv_latent_quant_group_size",
        "full_attention_layers",
        "obs_layer_ids",
        "h2o_decode_budget",
        "h2o_prefill_budget",
        "h2o_recent_ratio",
        "h2o_prefill_score_window",
    )
    return {
        key: _jsonable_config_value(getattr(config, key))
        for key in keys
        if hasattr(config, key)
    }


def _cache_stats(llm) -> dict[str, int]:
    cache_manager = getattr(getattr(llm, "model_runner", None), "cache_manager", None)
    if cache_manager is None or not hasattr(cache_manager, "free_slot_stats"):
        return {}
    raw_stats = cache_manager.free_slot_stats()
    return {
        str(key): int(value)
        for key, value in raw_stats.items()
        if isinstance(value, (int, float, bool))
    }


def _numeric_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in sorted(set(before) | set(after))
    }


def _observe_prefix_cache_hits(llm, hits_by_seq_id: dict[int, int]) -> None:
    scheduler = getattr(llm, "scheduler", None)
    if scheduler is None:
        return
    for queue_name in ("waiting", "decoding"):
        for seq in getattr(scheduler, queue_name, []):
            seq_id = int(getattr(seq, "seq_id", -1))
            hit_len = int(getattr(seq, "prefix_cache_hit_len", 0) or 0)
            if seq_id >= 0 and hit_len > 0:
                hits_by_seq_id[seq_id] = max(hit_len, int(hits_by_seq_id.get(seq_id, 0)))


def _finished_outputs_have_tokens(finished_outputs) -> bool:
    for output in finished_outputs:
        if len(output) >= 2 and output[1]:
            return True
    return False


def _completed_output_records(outputs, output_len):
    return [
        {"request_id": row[0], "token_ids": row[1],
         "status": "success" if len(row[1]) == output_len else "model_failed"}
        for row in outputs
    ]


def benchmark_task(method, length, bs, args, results_dict):
    if getattr(args, "engine", "sparseengine") in ("hisparse", "vortex"):
        from benchmark.hisparse_microbench import benchmark_decode_stage
        benchmark_decode_stage(method, length, bs, args, results_dict)
        return
    if getattr(args, "engine", "sparseengine") == "vllm":
        from benchmark.vllm_microbench import benchmark_decode_stage
        benchmark_decode_stage(method, length, bs, args, results_dict)
        return

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print(f"\n>>> Starting: {method.upper()} | Context: {length} | Batch: {bs}...")

    base_hyper_params = args.hyper_params_dict
    synchronize_step_timing = bool(
        getattr(args, "synchronize_step_timing", False)
    )
    sparse_kwargs: dict[str, Any] = {
        "sparse_method": _benchmark_sparse_method(method),
    }

    normalized_method = normalize_sparse_method(sparse_kwargs["sparse_method"])
    tensor_parallel_size = int(base_hyper_params.get("tensor_parallel_size", 1) or 1)
    graph_supported = (
        is_tp_decode_cuda_graph_supported(normalized_method)
        if tensor_parallel_size > 1
        else is_decode_cuda_graph_supported(normalized_method)
    )
    if bool(base_hyper_params.get("decode_graph")) and not graph_supported:
        results_dict[(method, length, bs)] = {
            "method": method,
            "sparse_method": normalized_method,
            "length": int(length),
            "batch_size": int(bs),
            "status": "SKIPPED_BY_POLICY",
            "reason": (
                "decode_graph is not supported for "
                f"sparse_method={normalized_method!r}, tensor_parallel_size={tensor_parallel_size}."
            ),
            "decode_graph_expected": True,
            "decode_graph_active": False,
            "synchronize_step_timing": synchronize_step_timing,
        }
        print(
            f"[{method.upper()}] SKIPPED_BY_POLICY: decode_graph is not supported "
            f"for sparse_method={normalized_method!r}, tensor_parallel_size={tensor_parallel_size}."
        )
        return

    llm = None
    resolved_engine_config: dict[str, Any] = {}
    step_rows = []
    completed_outputs = []
    completed_requests = 0
    actual_decode_peak = 0
    window = None
    try:
        m_len = length + args.output_len + (0 if getattr(args, "require_full_decode_batch", False) else 100)
        if args.max_model_len_override is not None:
            if args.max_model_len_override < length + args.output_len:
                raise ValueError(
                    "max_model_len_override must be >= length + output_len; "
                    f"got {args.max_model_len_override} < {length + args.output_len}."
                )
            m_len = int(args.max_model_len_override)
        # max_model_len is derived from the workload geometry.
        # They can be passed in --hyper_params, but will be overwritten here to keep the benchmark consistent.
        hyper_params = dict(base_hyper_params)
        _apply_prefill_policy_defaults(hyper_params, method)
        hyper_params.pop("max_model_len", None)
        hyper_params.setdefault("max_num_seqs_in_batch", int(bs))
        hyper_params.setdefault("max_decoding_seqs", int(bs))

        from sparseengine import LLM, SamplingParams
        engine_kwargs = {
            **hyper_params,
            "max_model_len": m_len,
            **sparse_kwargs,
        }
        llm = LLM(args.model_path, **engine_kwargs)
        resolved_engine_config = _resolved_engine_config(llm)
        prefix_cache_stats_before = _cache_stats(llm)

        prompt_token_ids = [[100] * length for _ in range(bs)]
        sampling_params = [
            SamplingParams(
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                ignore_eos=True,
                max_tokens=args.output_len,
            )
            for _ in range(bs)
        ]
        admission_wave_size = int(getattr(args, "admission_wave_size", 0) or 0)
        staged_admission = 0 < admission_wave_size < bs
        strict_batch = bool(getattr(args, "require_full_decode_batch", False))
        step_rows = []
        completed_outputs = []
        actual_decode_peak = 0
        wave_decode_gap_steps = int(getattr(args, "wave_decode_gap_steps", 0) or 0)


        from sparseengine.utils.profiler import profiler
        profiler.reset()

        torch.cuda.synchronize()

        prefill_tokens = 0
        decode_tokens = 0
        prefill_times = []
        decode_times = []
        ttft = None
        decode_tokens_after_full = 0
        decode_times_after_full = []
        decode_bs_after_full = []
        full_admission_reached = not (staged_admission or strict_batch)
        impossible_full_admission = False
        decode_steps_after_full = 0
        decode_warmup_steps_after_full = int(getattr(args, "decode_warmup_steps_after_full", 0) or 0)
        max_decode_steps_after_full = int(getattr(args, "max_decode_steps_after_full", 0) or 0)
        if decode_warmup_steps_after_full < 0:
            raise ValueError("decode_warmup_steps_after_full must be non-negative.")
        if max_decode_steps_after_full > 0 and max_decode_steps_after_full <= decode_warmup_steps_after_full:
            raise ValueError(
                "max_decode_steps_after_full must exceed decode_warmup_steps_after_full when both are set."
            )

        t_start = perf_counter()
        decode_started = False

        # Manually run the generation loop to get detailed stats
        next_request_idx = 0
        decode_steps_since_last_wave = 0
        request_seq_ids: list[int] = []
        prefix_hits_by_seq_id: dict[int, int] = {}

        def add_wave(max_new_requests: int):
            nonlocal next_request_idx, decode_steps_since_last_wave
            end_idx = min(bs, next_request_idx + max_new_requests)
            for req_idx in range(next_request_idx, end_idx):
                seq_id = llm.add_request(prompt_token_ids[req_idx], sampling_params[req_idx])
                request_seq_ids.append(int(seq_id))
            added = end_idx - next_request_idx
            next_request_idx = end_idx
            decode_steps_since_last_wave = 0
            return added

        add_wave(admission_wave_size if staged_admission else bs)

        window = None
        window_graphs = []
        if getattr(args, "decode_window_steps", 0):
            def window_sync():
                window_graphs[:] = [
                    {"capture_count": r["decode_graph"]["capture_count"],
                     "replay_count": r["decode_graph"]["replay_count"],
                     "eager_decode_count": r["decode_graph"]["eager_static_count"]
                                           + r["decode_graph"]["force_eager_count"]}
                    for r in llm.debug_sparse_state_summaries(synchronize=True)]
            window = PipelinedDecodeWindow(bs, args.decode_window_steps,
                decode_warmup_steps_after_full, synchronize=window_sync,
                clock=perf_counter, graph_stats=lambda: [dict(r) for r in window_graphs])

        has_queued = False
        zero_steps = 0
        while next_request_idx < bs or not llm.is_finished():
            if (
                staged_admission
                and next_request_idx < bs
                and len(llm.scheduler.waiting) == 0
                and (
                    (
                        len(llm.scheduler.decoding) > 0
                        and decode_steps_since_last_wave >= wave_decode_gap_steps
                    )
                    or (len(llm.scheduler.decoding) == 0 and llm.is_finished())
                )
            ):
                add_wave(admission_wave_size)

            if window is not None:
                window.boundary()
                window_seqs = list(llm.scheduler.decoding)
                window_ids = [seq.seq_id for seq in window_seqs]
                window_contexts = [len(seq) for seq in window_seqs]
                window_admitted = next_request_idx == bs and not llm.scheduler.waiting
            step_start = perf_counter()
            finished_outputs, num_tokens = llm.step()
            if synchronize_step_timing:
                torch.cuda.synchronize()
            step_dt = perf_counter() - step_start
            if strict_batch and int(getattr(llm.scheduler, "total_preemptions", 0)):
                raise RuntimeError("Full decode batch capacity exceeded: scheduler preemption")
            if window is not None:
                ticket = window.submit(is_decode=num_tokens < 0, request_ids=window_ids,
                    tokens=abs(num_tokens), admission_complete=window_admitted,
                    context_lengths=window_contexts,
                    preemptions=int(getattr(llm.scheduler, "total_preemptions", 0)))
                actual_tokens = sum(len(tokens) for _, tokens in llm.last_step_token_outputs) if num_tokens < 0 else None
                window.complete(ticket, decode_tokens=actual_tokens)
            completed_requests += len(finished_outputs)
            actual_decode_peak = max(actual_decode_peak, -num_tokens)
            if strict_batch:
                completed_outputs.extend(finished_outputs)
                step_rows.append({"tokens": num_tokens, "elapsed_s": step_dt, "measured": False})
            _observe_prefix_cache_hits(llm, prefix_hits_by_seq_id)

            if num_tokens > 0:
                prefill_tokens += num_tokens
                prefill_times.append(step_dt)
                if decode_started:
                    has_queued = True
                zero_steps = 0
                # In this engine, the first completion token is sampled during the *last* prefill
                # step of each sequence (Scheduler.postprocess appends it in the prefill branch).
                # So TTFT should be captured on a prefill step, not on the first decode step.
                if ttft is None and (
                    llm.scheduler.decoding
                    or _finished_outputs_have_tokens(finished_outputs)
                ):
                    ttft = perf_counter() - t_start
            elif num_tokens < 0:
                decode_started = True
                decode_steps_since_last_wave += 1
                decode_times.append(step_dt)
                decode_tokens += (-num_tokens)
                if full_admission_reached and (not strict_batch or -num_tokens == bs):
                    decode_steps_after_full += 1
                    if decode_steps_after_full > decode_warmup_steps_after_full:
                        if strict_batch:
                            step_rows[-1]["measured"] = True
                        decode_times_after_full.append(step_dt)
                        decode_tokens_after_full += (-num_tokens)
                        decode_bs_after_full.append(-num_tokens)
                zero_steps = 0
                # Fallback: if output_len==0/1 or internal behavior changes, ensure TTFT is set.
                if ttft is None:
                    ttft = perf_counter() - t_start
            else:
                zero_steps += 1
                if zero_steps >= 50:
                    raise RuntimeError("llm.step() returned 0 tokens repeatedly; scheduler may be stuck.")

            if (staged_admission or strict_batch) and not full_admission_reached and next_request_idx == bs:
                if len(llm.scheduler.waiting) == 0 and len(llm.scheduler.decoding) == bs:
                    full_admission_reached = True
                elif finished_outputs:
                    impossible_full_admission = True
                    break

            if strict_batch and finished_outputs and not full_admission_reached:
                raise RuntimeError("Full decode batch capacity exceeded: request finished before all admission waves completed")

            if full_admission_reached and max_decode_steps_after_full > 0 and decode_steps_after_full >= max_decode_steps_after_full:
                break

        # Admission failure is capacity evidence, not an incomplete timing window.
        # Check it before require_result() so the capacity sweep can refine its bound.
        if strict_batch and (not full_admission_reached or actual_decode_peak != bs):
            raise RuntimeError("Full decode batch capacity exceeded: not all requests entered decode together")
        if window is not None:
            window.boundary()
            window_result = window.require_result()
            case_path = Path(args.output_dir) / f"{method}-{length}-{bs}"
            (case_path / "window.json").parent.mkdir(parents=True, exist_ok=True)
            (case_path / "window.json").write_text(json.dumps(window_result, indent=2) + "\n")
        if strict_batch:
            if len(completed_outputs) != bs or any(len(row[1]) != args.output_len for row in completed_outputs):
                raise RuntimeError("Incomplete output: every request must finish the requested output length")
            if args.output_dir:
                case_path = Path(args.output_dir) / f"{method}-{length}-{bs}"
                _write_jsonl_rows(case_path / "steps.jsonl", step_rows)
                _write_jsonl_rows(case_path / "raw_outputs.jsonl",
                                  _completed_output_records(completed_outputs, args.output_len))
        if decode_warmup_steps_after_full > 0 and not decode_times_after_full:
            raise RuntimeError(
                "No measured decode steps remained after discarding "
                f"{decode_warmup_steps_after_full} warmup steps."
            )

        torch.cuda.synchronize()
        peak_mem = get_peak_memory()
        graph_status = _decode_cuda_graph_status(llm)
        operator_runtime_stats = llm.operator_runtime_stats()
        prefix_cache_stats_after = _cache_stats(llm)
        prefix_cache_stats_delta = _numeric_delta(prefix_cache_stats_before, prefix_cache_stats_after)
        observed_prefix_hit_tokens = int(sum(prefix_hits_by_seq_id.values()))
        observed_prefix_hit_requests = int(sum(1 for value in prefix_hits_by_seq_id.values() if int(value) > 0))
        stats_prefix_hit_tokens = int(prefix_cache_stats_delta.get("prefix_cache_hit_tokens", 0))
        stats_prefix_hit_requests = int(prefix_cache_stats_delta.get("prefix_cache_hit_requests", 0))
        if bool(getattr(args, "require_prefix_cache_hit", False)) and max(
            observed_prefix_hit_tokens,
            stats_prefix_hit_tokens,
        ) <= 0:
            raise RuntimeError(
                "Prefix-cache stress did not observe any prefix cache hit: "
                f"observed_prefix_hit_tokens={observed_prefix_hit_tokens}, "
                f"stats_prefix_hit_tokens={stats_prefix_hit_tokens}, "
                f"request_seq_ids={request_seq_ids[:8]}."
            )
        preemptions = int(getattr(getattr(llm, "scheduler", None), "total_preemptions", 0) or 0)
        recompute_replays = int(
            getattr(
                getattr(llm, "scheduler", None),
                "total_recompute_replays",
                0,
            )
            or 0
        )
        cache_manager = getattr(getattr(llm, "model_runner", None), "cache_manager", None)
        if cache_manager is None or not hasattr(cache_manager, "memory_accounting"):
            raise RuntimeError("Sparse-VLLM cache manager does not expose memory_accounting().")
        memory_accounting = cache_manager.memory_accounting()

        ttft = float(ttft or 0.0)
        prefill_s = sum(prefill_times)
        decode_s = sum(decode_times)

        prefill_tp = stage_throughput(prefill_tokens, prefill_s)
        used_full_admission_window = bool(decode_times_after_full)
        decode_s_effective = sum(decode_times_after_full) if used_full_admission_window else decode_s
        decode_tokens_effective = decode_tokens_after_full if used_full_admission_window else decode_tokens
        decode_tp = stage_throughput(decode_tokens_effective, decode_s_effective)
        # Legacy ITL alias: a batch-derived execution proxy, not request TPOT.
        avg_itl = (decode_s_effective / (decode_tokens_effective / bs) * 1000) if decode_tokens_effective > 0 else 0
        avg_active_bs = (
            sum(decode_bs_after_full) / len(decode_bs_after_full)
            if decode_bs_after_full
            else (decode_tokens / len(decode_times) if decode_times else 0)
        )

        stage_mode = (
            f" | AdmissionWave: {admission_wave_size}"
            f" | WaveGapSteps: {wave_decode_gap_steps}"
            f" | FullAdmit: {'yes' if full_admission_reached else 'no'}"
            f" | DecodeScope: {'full' if used_full_admission_window else 'fallback'}"
            if staged_admission
            else ""
        )
        if window is None:
            print(f"[{method.upper()}] First observed token: {ttft:.2f}s | Prefill step: {prefill_tp or 0.0:.2f} tok/s | Decode step: {decode_tp or 0.0:.2f} tok/s | Decode execution proxy: {avg_itl:.2f}ms | AvgBS: {avg_active_bs:.1f} | Mem: {peak_mem:.2f} GB{stage_mode}")

        results_dict[(method, length, bs)] = {
            "method": method,
            "sparse_method": normalized_method,
            "length": int(length),
            "batch_size": int(bs),
            "prefill_tp": prefill_tp or 0.0,
            "engine": "sparseengine",
            "actual_decode_peak": actual_decode_peak,
            "completed_requests": completed_requests,
            "output_len": int(args.output_len),
            "require_full_decode_batch": strict_batch,
            "measurement_scope": "full_batch_pure_decode_steps" if strict_batch else "legacy_stage_scope",
            "decode_tp": decode_tp or 0.0,
            "stage_metrics_status": "success" if synchronize_step_timing else "not_measured",
            "stage_timing_scope": "sum_synchronized_llm_steps" if synchronize_step_timing else "unsynchronized_host_steps",
            "prefill_stage_elapsed_s": prefill_s if synchronize_step_timing else None,
            "decode_stage_elapsed_s": decode_s_effective if synchronize_step_timing else None,
            "prefill_computed_tokens": prefill_tokens,
            "logical_input_tokens": bs * length,
            "decode_stage_tokens": decode_tokens_effective,
            "prefill_stage_throughput_tps": prefill_tp if synchronize_step_timing else None,
            "decode_stage_throughput_tps": decode_tp if synchronize_step_timing else None,
            "request_metrics_status": "not_measured",
            "ttft_timing_scope": "batch_start_to_first_observed_token",
            "itl_timing_scope": "batch_scaled_decode_execution_proxy",
            "ttft": ttft,
            "itl": avg_itl,
            "avg_bs": avg_active_bs,
            "mem": peak_mem,
            "has_queued": has_queued,
            "full_admission_reached": full_admission_reached,
            "impossible_full_admission": impossible_full_admission,
            "decode_scope": "full" if used_full_admission_window else "fallback",
            "staged_admission": staged_admission,
            "admission_wave_size": admission_wave_size if staged_admission else None,
            "wave_decode_gap_steps": wave_decode_gap_steps if staged_admission else None,
            "max_decode_steps_after_full": int(getattr(args, "max_decode_steps_after_full", 0) or 0),
            "decode_warmup_steps_after_full": decode_warmup_steps_after_full,
            "decode_steps_after_full": int(decode_steps_after_full),
            "measured_decode_steps_after_full": len(decode_times_after_full),
            "synchronize_step_timing": synchronize_step_timing,
            "scheduler_preemptions": preemptions,
            "scheduler_recompute_replays": recompute_replays,
            "decode_graph_expected": bool(base_hyper_params.get("decode_graph")),
            **graph_status,
            "operator_runtime_stats": operator_runtime_stats,
            "prefix_cache_required": bool(getattr(args, "require_prefix_cache_hit", False)),
            "prefix_cache_stats_before": prefix_cache_stats_before,
            "prefix_cache_stats_after": prefix_cache_stats_after,
            "prefix_cache_stats_delta": prefix_cache_stats_delta,
            "prefix_cache_hit_tokens": stats_prefix_hit_tokens,
            "prefix_cache_hit_requests": stats_prefix_hit_requests,
            "observed_prefix_cache_hit_tokens": observed_prefix_hit_tokens,
            "observed_prefix_cache_hit_requests": observed_prefix_hit_requests,
            "observed_prefix_cache_hit_by_seq": {
                str(seq_id): int(hit_len)
                for seq_id, hit_len in sorted(prefix_hits_by_seq_id.items())
            },
            "memory_accounting": memory_accounting,
            "engine_hyper_params": engine_kwargs,
            "resolved_engine_config": resolved_engine_config,
            "status": "SUCCESS"
        }
        if window is not None:
            result = dict(results_dict[(method, length, bs)])
            result.update(decode_window_fields(window_result))
            results_dict[(method, length, bs)] = result
            _write_jsonl_rows(case_path / "window_steps.jsonl", window.records)

    except Exception as e:
        print(f"Error at {method}/{length}/{bs}: {e}")
        traceback.print_exc()
        if getattr(args, "require_full_decode_batch", False) and args.output_dir:
            case_path = Path(args.output_dir) / f"{method}-{length}-{bs}"
            _write_jsonl_rows(case_path / "steps.jsonl", step_rows)
            _write_jsonl_rows(case_path / "raw_outputs.jsonl",
                              _completed_output_records(completed_outputs, args.output_len))
            if window is not None:
                _write_jsonl_rows(case_path / "window_steps.jsonl", window.records)
        results_dict[(method, length, bs)] = {
            "method": method,
            "sparse_method": normalized_method,
            "length": int(length),
            "batch_size": int(bs),
            "actual_decode_peak": actual_decode_peak,
            "completed_requests": completed_requests,
            "status": "FAILED",
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "synchronize_step_timing": synchronize_step_timing,
            "resolved_engine_config": _resolved_engine_config(llm) if llm is not None else resolved_engine_config,
        }
    finally:
        if llm is not None and hasattr(llm, "exit"):
            llm.exit()


def main():
    parser = argparse.ArgumentParser(description="Professional benchmark for sparseengine.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--engine", choices=("sparseengine", "vllm", "hisparse", "vortex"), default="sparseengine")
    parser.add_argument("--engine_kwargs", default="{}", help="External-engine constructor options, JSON or @file; workload/timing keys remain protected.")
    parser.add_argument("--backend_label", default=None)
    parser.add_argument("--require_full_decode_batch", action="store_true",
                        help="Measure only pure decode steps with all requested sequences resident; require complete outputs and no preemption.")
    parser.add_argument("--decode_window_steps", type=int, default=0,
                        help="Opt-in continuous decode window with boundary-only synchronization; mutually exclusive with synchronized steps.")
    parser.add_argument("--lengths", type=str, default="16000,32000,64000", help="Context lengths to test")
    parser.add_argument("--batch_sizes", type=str, default="4", help="Batch sizes to test")
    parser.add_argument(
        "--methods",
        type=str,
        default="vanilla,snapkv,omnikv",
        help=(
            "Methods to test (vanilla, streamingllm, attention-sink, snapkv, h2o, pyramidkv, "
            "omnikv, quest, rkv, skipkv, deltakv; deltakv-less-memory* are legacy aliases)"
        ),
    )
    parser.add_argument("--output_len", type=int, default=512, help="Output tokens per request")
    parser.add_argument(
        "--max_model_len_override",
        type=int,
        default=None,
        help="Optional fixed max_model_len for diagnostic runs; defaults to length + output_len + 100.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for generation. Default 0.0 (greedy) for throughput benchmarking.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling top-p. Only used when temperature > 0.",
    )
    parser.add_argument(
        "--admission_wave_size",
        type=int,
        default=0,
        help="If >0 and < batch size, only admit this many sequences at a time. Decode throughput is then measured after the final wave has fully entered decode.",
    )
    parser.add_argument(
        "--max_decode_steps_after_full",
        type=int,
        default=0,
        help="If >0, stop after this many decode steps after full admission is reached.",
    )
    parser.add_argument(
        "--decode_warmup_steps_after_full",
        type=int,
        default=0,
        help="Discard this many initial full-admission decode steps from stage throughput and execution-proxy metrics.",
    )
    parser.add_argument(
        "--synchronize_step_timing",
        action="store_true",
        help=(
            "Opt-in stage diagnostic: synchronize CUDA after each llm.step() before measuring its duration, "
            "so post-sparse work is attributed to the step that launched it."
        ),
    )
    parser.add_argument(
        "--wave_decode_gap_steps",
        type=int,
        default=0,
        help="In staged admission mode, require this many decode steps before admitting the next wave.",
    )
    parser.add_argument(
        "--require_prefix_cache_hit",
        action="store_true",
        help="Fail the benchmark case unless at least one prefix-cache hit is observed.",
    )
    parser.add_argument(
        "--hyper_params",
        type=str,
        default="{}",
        help=(
            "LLMEngine/Config hyper-params as JSON (string or @file.json). "
            "engine_prefill_chunk_size controls all_chunked steps and RawKV "
            "offload chunks; long_prefill_offload_threshold selects the full "
            "versus RawKV mode for long_bs1full_short_batch methods."
        ),
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Optional machine-readable JSONL output path with one row per method/length/batch.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional directory for run_info.json, performance.jsonl, aggregate_metrics.json, and report.md.",
    )

    args = parser.parse_args()
    args.engine_kwargs_dict = _load_json_arg(args.engine_kwargs)
    if args.engine == "sparseengine" and (args.engine_kwargs_dict or args.backend_label):
        parser.error("engine_kwargs/backend_label are for external stage adapters only")
    if args.engine in ("hisparse", "vortex") and not args.output_dir:
        parser.error("HiSparse stage adapter requires --output_dir for worker-side raw steps")
    if args.decode_window_steps and (args.decode_window_steps < 1 or args.synchronize_step_timing
            or not args.require_full_decode_batch or not args.output_dir
            or args.decode_warmup_steps_after_full < 1 or args.max_decode_steps_after_full):
        parser.error("Decode windows require full batch, positive warmup, output_dir, no step sync and complete outputs")
    if args.require_full_decode_batch and (not (args.synchronize_step_timing or args.decode_window_steps) or args.max_decode_steps_after_full):
        parser.error("Full decode batch measurement requires synchronized timing and untruncated outputs")
    if args.engine == "vllm" and not args.require_full_decode_batch:
        parser.error("vLLM stage adapter requires --require_full_decode_batch")
    try:
        args.hyper_params_dict = _build_engine_hyper_params(args)
    except ValueError as e:
        parser.error(str(e))

    test_lengths = [int(x) for x in args.lengths.split(",")]
    test_methods = args.methods.split(",")
    test_batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    manager = mp.Manager()
    results_dict = manager.dict()

    for method in test_methods:
        for length in test_lengths:
            for bs in test_batch_sizes:
                p = mp.Process(target=benchmark_task, args=(method, length, bs, args, results_dict))
                p.start()
                p.join()
                _record_child_exit_failure(
                    results_dict,
                    method=method,
                    length=length,
                    batch_size=bs,
                    exitcode=p.exitcode,
                    synchronize_step_timing=bool(args.synchronize_step_timing),
                )


    print(f"\n\n{'='*140}")
    decode_column = "DecWindowTP" if args.decode_window_steps else "DecStepTP"
    print(f"{ 'Method':<12} {'Len':<8} {'BS':<4} {'First(s)':<10} {'PreStepTP':<12} {decode_column:<12} {'Proxy(ms)':<10} {'AvgBS':<8} {'Mem(GB)':<10} {'Speedup'}")
    print("-" * 140)


    vanilla_stats = {}
    for length in test_lengths:
        for bs in test_batch_sizes:
            v_res = results_dict.get(("vanilla", length, bs))
            if v_res and v_res["status"] == "SUCCESS":
                vanilla_stats[(length, bs)] = v_res["decode_tp"]

    jsonl_rows: list[dict[str, Any]] = []
    for method in test_methods:
        for length in test_lengths:
            for bs in test_batch_sizes:
                res = results_dict.get((method, length, bs))
                if not res or res["status"] in ["FAILED", "OOM", "SKIPPED_BY_POLICY"]:
                    status_str = res["status"] if res else "UNKNOWN"
                    print(f"{method:<12} {length:<8} {bs:<4} {status_str:<10} {'-':<12} {'-':<12} {'-':<10} {'-':<8} {'-':<10} {'-'}")
                    row = dict(res or {})
                    row.setdefault("method", method)
                    row.setdefault("length", int(length))
                    row.setdefault("batch_size", int(bs))
                    row.setdefault("status", status_str)
                    row.setdefault(
                        "synchronize_step_timing",
                        bool(getattr(args, "synchronize_step_timing", False)),
                    )
                    jsonl_rows.append(row)
                    continue

                ttft = res["ttft"]
                pre_tp = res["prefill_tp"]
                dec_tp = res["decode_tp"]
                itl = res["itl"]
                avg_bs = res["avg_bs"]
                mem = res["mem"]
                has_queued = res.get("has_queued", False)

                bs_str = f"{bs}*" if has_queued else f"{bs}"

                speedup = None if args.decode_window_steps else 1.0
                if (length, bs) in vanilla_stats:
                    vanilla_decode_tp = float(vanilla_stats[(length, bs)])
                    speedup = dec_tp / vanilla_decode_tp if vanilla_decode_tp > 0 else None

                speedup_str = "n/a" if speedup is None else f"{speedup:.2f}x"
                print(f"{method:<12} {length:<8} {bs_str:<4} {_format_metric(ttft, 2):<10} {_format_metric(pre_tp, 1):<12} {_format_metric(dec_tp, 1):<12} {_format_metric(itl, 2):<10} {_format_metric(avg_bs, 1):<8} {_format_metric(mem, 2):<10} {speedup_str}")
                row = dict(res)
                row["speedup_vs_vanilla_decode"] = None if speedup is None else float(speedup)
                jsonl_rows.append(row)
    print(f"{ '='*140}\n")
    _write_jsonl(args.output_jsonl, jsonl_rows)
    _write_output_dir(args, jsonl_rows)
    failed_rows = [
        row
        for row in jsonl_rows
        if str(row.get("status", "")).upper() in {"FAILED", "OOM", "UNKNOWN"}
    ]
    if failed_rows:
        print(f"[microbench] {len(failed_rows)} benchmark case(s) failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    raise SystemExit(main())
