#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT_FOR_IMPORT))
sys.path.insert(0, str(REPO_ROOT_FOR_IMPORT / "src"))

from benchmark.efficiency.metrics import percentile

CASE_PRESETS: dict[str, dict[str, Any]] = {
    "baseline_full": {
        "method": "vanilla",
        "enable_prefix_caching": False,
        "label": "full attention, prefix cache off",
    },
    "prefix_full": {
        "method": "vanilla",
        "enable_prefix_caching": True,
        "label": "full attention, prefix cache on",
    },
    "prefix_omnikv": {
        "method": "omnikv",
        "enable_prefix_caching": True,
        "label": "OmniKV, prefix cache on",
    },
    "prefix_quest": {
        "method": "quest",
        "enable_prefix_caching": True,
        "label": "QuEST, prefix cache on",
    },
    "chain_streamingllm": {
        "method": "streamingllm",
        "enable_prefix_caching": True,
        "label": "StreamingLLM, linear chain prefix cache",
    },
    "chain_snapkv": {
        "method": "snapkv",
        "enable_prefix_caching": True,
        "label": "SnapKV, linear chain prefix cache",
    },
    "chain_h2o": {
        "method": "h2o",
        "enable_prefix_caching": True,
        "label": "H2O, linear chain prefix cache",
    },
    "chain_pyramidkv": {
        "method": "pyramidkv",
        "enable_prefix_caching": True,
        "label": "PyramidKV, linear chain prefix cache",
    },
    "chain_rkv": {
        "method": "rkv",
        "enable_prefix_caching": True,
        "label": "R-KV, linear chain prefix cache",
    },
    "chain_skipkv": {
        "method": "skipkv",
        "enable_prefix_caching": True,
        "label": "SkipKV, linear chain prefix cache",
    },
}

CASE_ALIASES = {
    "baseline": "baseline_full",
    "vanilla": "baseline_full",
    "full": "baseline_full",
    "prefix": "prefix_full",
    "prefix_vanilla": "prefix_full",
    "omnikv": "prefix_omnikv",
    "quest": "prefix_quest",
    "streamingllm": "chain_streamingllm",
    "attention-sink": "chain_streamingllm",
    "attention_sink": "chain_streamingllm",
    "snapkv": "chain_snapkv",
    "h2o": "chain_h2o",
    "pyramidkv": "chain_pyramidkv",
    "rkv": "chain_rkv",
    "skipkv": "chain_skipkv",
}


@dataclass
class RequestSpec:
    request_key: str
    workload: str
    phase: str
    session_id: int
    turn: int
    prompt_token_ids: list[int]
    output_len: int
    eligible_cache_tokens: int
    expected_reuse_tokens: int


@dataclass
class RequestState:
    spec: RequestSpec
    seq_id: int
    add_s: float
    first_token_s: float | None = None
    finish_s: float | None = None
    generated_token_ids: list[int] = field(default_factory=list)
    prefix_cache_hit_len: int = 0
    prefix_cache_hit_blocks: int = 0
    status: str = "success"
    error_message: str = ""
    chain_expected: bool = False
    chain_id: str | None = None
    chain_status: str = "disabled"
    reused_tokens: int = 0
    prefilled_tokens: int = 0
    physical_residency: list[int] = field(default_factory=list)


def _cache_hit_metric_error(
    spec: RequestSpec,
    *,
    cached_tokens: int,
    cached_blocks: int,
    block_size: int,
) -> str:
    if cached_tokens < 0:
        return f"runtime reported negative cached_tokens={cached_tokens}."
    if cached_blocks < 0:
        return f"runtime reported negative cached_blocks={cached_blocks}."
    if cached_tokens == 0 and cached_blocks == 0:
        return ""
    if block_size <= 0:
        return f"invalid prefix cache block_size={block_size}."
    if cached_tokens % block_size != 0:
        return (
            f"cached_tokens={cached_tokens} is not aligned to prefix cache block_size={block_size}."
        )
    if cached_blocks * block_size != cached_tokens:
        return (
            f"cached_blocks={cached_blocks} does not match cached_tokens={cached_tokens} "
            f"with block_size={block_size}."
        )
    if cached_tokens > int(spec.eligible_cache_tokens):
        return (
            f"cached_tokens={cached_tokens} exceeds planned_eligible_cache_tokens="
            f"{int(spec.eligible_cache_tokens)}."
        )
    if cached_tokens > int(spec.expected_reuse_tokens):
        return (
            f"cached_tokens={cached_tokens} exceeds expected_reuse_tokens="
            f"{int(spec.expected_reuse_tokens)}."
        )
    return ""


def _chain_cache_metric_error(
    spec: RequestSpec,
    state: RequestState,
) -> str:
    if not state.chain_id:
        return "chain request completed without a chain_id."
    expected_reuse = int(spec.expected_reuse_tokens)
    if expected_reuse == 0 and state.chain_status != "created":
        return (
            "first chain turn must report chain_status='created', got "
            f"{state.chain_status!r}."
        )
    if expected_reuse > 0 and state.chain_status != "resumed":
        return (
            "continuing chain turn must report chain_status='resumed', got "
            f"{state.chain_status!r}."
        )
    if int(state.reused_tokens) != expected_reuse:
        return (
            f"reused_logical_tokens={int(state.reused_tokens)} does not match "
            f"expected_reuse_tokens={expected_reuse}."
        )
    expected_prefill = len(spec.prompt_token_ids) - expected_reuse
    if int(state.prefilled_tokens) != expected_prefill:
        return (
            f"prefilled_delta_tokens={int(state.prefilled_tokens)} does not "
            f"match expected suffix={expected_prefill}."
        )
    if not state.physical_residency:
        return "chain request finished without per-layer physical residency."
    return ""


def _write_request_records(
    *,
    states: dict[int, RequestState],
    tokenizer: Any,
    per_turn_path: Path,
    raw_output_path: Path,
    batch_start_s: float,
    block_size: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with per_turn_path.open("a", encoding="utf-8") as per_turn, raw_output_path.open("a", encoding="utf-8") as raw_output:
        for seq_id in sorted(states):
            state = states[seq_id]
            spec = state.spec
            first_token_s = state.first_token_s or state.finish_s or time.perf_counter()
            finish_s = state.finish_s or first_token_s
            generated = state.generated_token_ids[: int(spec.output_len)]
            cached_tokens = int(state.prefix_cache_hit_len)
            cached_blocks = int(state.prefix_cache_hit_blocks)
            planned_eligible_tokens = int(spec.eligible_cache_tokens)
            status = state.status
            error_message = state.error_message
            metric_error = (
                _chain_cache_metric_error(spec, state)
                if state.chain_expected
                else _cache_hit_metric_error(
                    spec,
                    cached_tokens=cached_tokens,
                    cached_blocks=cached_blocks,
                    block_size=block_size,
                )
            )
            if metric_error:
                status = "metric_failed"
                error_message = metric_error if not error_message else f"{error_message}; {metric_error}"
            record = {
                "request_key": spec.request_key,
                "seq_id": seq_id,
                "workload": spec.workload,
                "phase": spec.phase,
                "session_id": spec.session_id,
                "turn": spec.turn,
                "status": status,
                "prompt_tokens": len(spec.prompt_token_ids),
                "max_new_tokens": int(spec.output_len),
                "generated_tokens": len(generated),
                "planned_eligible_cache_tokens": planned_eligible_tokens,
                "eligible_cache_tokens": planned_eligible_tokens,
                "expected_reuse_tokens": int(spec.expected_reuse_tokens),
                "cached_tokens": cached_tokens,
                "cached_blocks": cached_blocks,
                "ttft_s": float(first_token_s - state.add_s),
                "latency_s": float(finish_s - state.add_s),
                "batch_elapsed_s": float(finish_s - batch_start_s),
                "error_message": error_message,
                "chain_id": state.chain_id,
                "chain_status": state.chain_status,
                "reused_logical_tokens": state.reused_tokens,
                "prefilled_delta_tokens": state.prefilled_tokens,
                "physical_residency_by_layer": state.physical_residency,
            }
            raw = {
                **record,
                "prompt_token_ids": spec.prompt_token_ids,
                "generated_token_ids": generated,
                "generated_text": tokenizer.decode(generated, skip_special_tokens=True) if generated else "",
            }
            per_turn.write(json.dumps(record, ensure_ascii=False) + "\n")
            raw_output.write(json.dumps(raw, ensure_ascii=False) + "\n")
            records.append(record)
    return records


def _load_json_arg(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    raw = str(value).strip()
    if raw.startswith("@"):
        raw = Path(raw[1:]).expanduser().read_text(encoding="utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("JSON argument must decode to an object.")
    return parsed


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _canonical_cases(value: str) -> list[str]:
    cases: list[str] = []
    for raw_case in _split_csv(value):
        case = CASE_ALIASES.get(raw_case, raw_case)
        if case not in CASE_PRESETS:
            supported = ", ".join(sorted(CASE_PRESETS))
            raise ValueError(f"Unknown benchmark case {raw_case!r}. Supported cases: {supported}.")
        if case not in cases:
            cases.append(case)
    return cases


def _usable_prefix_cache_tokens(prompt_len: int, block_size: int) -> int:
    prompt_len = int(prompt_len)
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"prefix cache block size must be > 0, got {block_size}.")
    if prompt_len <= 1:
        return 0
    return ((prompt_len - 1) // block_size) * block_size


def _eligible_cache_tokens(reusable_prefix_len: int, current_prompt_len: int, block_size: int) -> int:
    reusable_blocks = (int(reusable_prefix_len) // int(block_size)) * int(block_size)
    return min(reusable_blocks, _usable_prefix_cache_tokens(current_prompt_len, block_size))


def _multiturn_reusable_prefix_len(
    *,
    turn: int,
    session_id: int,
    shared_system_len: int,
    history_len: int,
) -> int:
    if int(turn) > 0:
        return int(history_len)
    if int(session_id) > 0:
        return int(shared_system_len)
    return 0


def _percentile(values: list[float], pct: float) -> float:
    # Failed/empty cache cases retain their legacy empty-summary representation.
    if not values:
        return 0.0
    return percentile(values, pct)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _shell_command() -> str:
    return "python " + " ".join(sys.argv)


def _run_git(repo_root: Path, args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def git_metadata(repo_root: Path) -> dict[str, Any]:
    commit = _run_git(repo_root, ["rev-parse", "HEAD"]) or "unknown"
    status = _run_git(repo_root, ["status", "--porcelain"])
    return {
        "git_commit": commit,
        "git_commit_short": commit[:12] if commit != "unknown" else "unknown",
        "git_branch": _run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown",
        "git_dirty": bool(status),
        "git_status": status,
    }


def selected_env_snapshot() -> dict[str, str]:
    keys = {
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "NCCL_DEBUG",
        "PYTHONPATH",
        "SPARSEENGINE_MASTER_PORT",
        "TOKENIZERS_PARALLELISM",
        "TRANSFORMERS_CACHE",
        "VLLM_ATTENTION_BACKEND",
    }
    prefixes = ("CUDA_", "NCCL_", "SPARSEENGINE_")
    selected = {
        key
        for key in os.environ
        if key in keys or any(key.startswith(prefix) for prefix in prefixes)
    }
    return {key: os.environ[key] for key in sorted(selected)}


def benchmark_output_root() -> Path:
    env_root = os.getenv("SPARSEENGINE_PREFIX_CACHE_BENCH_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return REPO_ROOT_FOR_IMPORT / "outputs" / "prefix_cache_benchmarks"


def default_ledger_paths(feature: str, output_root: Path) -> tuple[Path, Path]:
    safe_feature = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in feature)
    ledger_dir = output_root / "_ledgers"
    return ledger_dir / f"{safe_feature}.jsonl", ledger_dir / f"{safe_feature}.csv"


def _ledger_csv_fields(record: dict[str, Any]) -> list[str]:
    preferred = [
        "run_id",
        "timestamp",
        "feature",
        "objective",
        "git_commit",
        "git_branch",
        "git_dirty",
        "benchmark",
        "benchmark_source",
        "script",
        "model_path",
        "method",
        "dataset",
        "status",
        "output_dir",
        "speedup",
        "memory_delta",
        "failure_summary",
        "decision",
        "notes",
    ]
    return [field for field in preferred if field in record]


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def append_ledger_record(record: dict[str, Any], *, jsonl_path: Path, csv_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    fields = _ledger_csv_fields(record)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({field: _csv_value(record.get(field)) for field in fields})


def _case_block_size(args: argparse.Namespace, case_name: str) -> int:
    method = CASE_PRESETS[case_name]["method"]
    if method == "quest":
        return int(args.quest_chunk_size)
    return int(args.prefix_cache_block_size)


def _case_engine_kwargs(args: argparse.Namespace, case_name: str, max_prompt_len: int) -> dict[str, Any]:
    preset = CASE_PRESETS[case_name]
    hyper_params = {
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "expert_parallel_size": int(
            getattr(args, "expert_parallel_size", 1)
        ),
        "decode_graph": bool(
            getattr(args, "decode_graph", False)
        ),
        "max_num_seqs_in_batch": int(args.max_active_requests),
        "max_decoding_seqs": int(args.max_active_requests),
        "max_num_batched_tokens": int(args.max_num_batched_tokens),
        "engine_prefill_chunk_size": int(args.engine_prefill_chunk_size),
        "throughput_log_interval_s": 0.0,
        "sink_keep_tokens": int(args.sink_keep_tokens),
        "recent_keep_tokens": int(args.recent_keep_tokens),
        "decode_keep_tokens": int(args.decode_keep_tokens),
        "full_attention_layers": args.full_attention_layers,
        "quest_chunk_size": int(args.quest_chunk_size),
        "prefix_cache_block_size": _case_block_size(args, case_name),
        "prefix_cache_max_blocks": args.prefix_cache_max_blocks,
        "prefix_cache_salt": args.prefix_cache_salt,
        "enable_prefix_caching": bool(preset["enable_prefix_caching"]),
        "prefix_cache_mode": (
            "chain" if case_name.startswith("chain_") else "auto"
        ),
        "sparse_method": preset["method"],
        "max_model_len": int(max_prompt_len + args.output_len + args.max_model_len_margin),
    }
    hyper_params.update(_load_json_arg(args.hyper_params))

    # Keep run geometry comparable across methods even when --hyper_params includes an older value.
    hyper_params["enable_prefix_caching"] = bool(preset["enable_prefix_caching"])
    hyper_params["sparse_method"] = preset["method"]
    hyper_params["quest_chunk_size"] = int(args.quest_chunk_size)
    hyper_params["full_attention_layers"] = args.full_attention_layers
    hyper_params["prefix_cache_block_size"] = _case_block_size(args, case_name)
    hyper_params["max_model_len"] = int(max_prompt_len + args.output_len + args.max_model_len_margin)
    hyper_params["max_num_seqs_in_batch"] = int(args.max_active_requests)
    hyper_params["max_decoding_seqs"] = int(args.max_active_requests)
    return {key: value for key, value in hyper_params.items() if value is not None}


def _decode_graph_rank_summaries(llm: Any) -> list[dict[str, Any]]:
    summaries = llm.debug_sparse_state_summaries()
    graph_summaries: list[dict[str, Any]] = []
    for summary in summaries:
        graph = summary.get("decode_graph")
        if not isinstance(graph, dict):
            raise RuntimeError(
                "Worker debug summary is missing decode_graph evidence: "
                f"world_rank={summary.get('world_rank')}."
            )
        graph_summaries.append(
            {
                "world_rank": int(summary["world_rank"]),
                **graph,
            }
        )
    return graph_summaries


def _decode_graph_rank_deltas(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    before_by_rank = {int(item["world_rank"]): item for item in before}
    after_by_rank = {int(item["world_rank"]): item for item in after}
    if set(before_by_rank) != set(after_by_rank):
        raise RuntimeError(
            "Decode CUDA Graph evidence changed world ranks: "
            f"before={sorted(before_by_rank)} after={sorted(after_by_rank)}."
        )
    counter_names = (
        "capture_count",
        "replay_count",
        "eager_static_count",
        "force_eager_count",
    )
    return [
        {
            "world_rank": rank,
            **{
                name: int(after_by_rank[rank][name])
                - int(before_by_rank[rank][name])
                for name in counter_names
            },
        }
        for rank in sorted(before_by_rank)
    ]


def _token_vocab(tokenizer: Any) -> list[int]:
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    vocab_values = sorted(set(int(token_id) for token_id in tokenizer.get_vocab().values()))
    ids = [token_id for token_id in vocab_values if token_id not in special_ids and token_id >= 0]
    if not ids:
        raise RuntimeError("Tokenizer vocabulary has no usable non-special token ids.")
    return ids


def _sample_tokens(vocab_ids: list[int], rng: random.Random, length: int) -> list[int]:
    length = int(length)
    if length <= 0:
        return []
    return [int(token_id) for token_id in rng.choices(vocab_ids, k=length)]


def _sample_length(rng: random.Random, max_len: int, min_len: int | None = None) -> int:
    max_value = int(max_len)
    min_value = max_value if min_len is None else int(min_len)
    if min_value < 0 or max_value < 0:
        raise ValueError(f"Length bounds must be non-negative, got min={min_value}, max={max_value}.")
    if min_value > max_value:
        raise ValueError(f"Length min must be <= max, got min={min_value}, max={max_value}.")
    return int(rng.randint(min_value, max_value))


def _length_bounds(args: argparse.Namespace, max_name: str, min_name: str) -> tuple[int, int]:
    max_value = int(getattr(args, max_name))
    min_raw = getattr(args, min_name, None)
    min_value = max_value if min_raw is None else int(min_raw)
    if min_value < 0 or max_value < 0:
        raise ValueError(f"{min_name}/{max_name} must be non-negative, got {min_value}/{max_value}.")
    if min_value > max_value:
        raise ValueError(f"{min_name} must be <= {max_name}, got {min_value} > {max_value}.")
    return min_value, max_value


def _token_count_plan(args: argparse.Namespace) -> dict[str, int]:
    session_prefix_min, session_prefix_max = _length_bounds(args, "session_prefix_len", "session_prefix_min_len")
    user_min, user_max = _length_bounds(args, "user_len", "user_min_len")
    shared_suffix_min, shared_suffix_max = _length_bounds(args, "shared_suffix_len", "shared_suffix_min_len")
    multiturn_first_prompt = (
        int(args.system_prompt_len)
        + session_prefix_max
        + user_max
    )
    multiturn_max_prompt = (
        int(args.system_prompt_len)
        + session_prefix_max
        + int(args.turns) * (user_max + int(args.output_len))
    )
    shared_prefix_max_prompt = int(args.shared_prefix_len) + shared_suffix_max
    return {
        "session_prefix_min": session_prefix_min,
        "session_prefix_max": session_prefix_max,
        "user_min": user_min,
        "user_max": user_max,
        "shared_suffix_min": shared_suffix_min,
        "shared_suffix_max": shared_suffix_max,
        "multiturn_first_prompt": multiturn_first_prompt,
        "multiturn_max_prompt": multiturn_max_prompt,
        "shared_prefix_max_prompt": shared_prefix_max_prompt,
        "max_prompt_len": max(multiturn_max_prompt, shared_prefix_max_prompt),
    }


def _long_text_threshold(
    args: argparse.Namespace,
    *,
    is_prefill: bool,
    engine_kwargs: dict[str, Any] | None = None,
) -> int:
    resolved = engine_kwargs or {}
    base = (
        int(resolved.get("sink_keep_tokens", args.sink_keep_tokens))
        + int(resolved.get("decode_keep_tokens", args.decode_keep_tokens))
        + int(resolved.get("recent_keep_tokens", args.recent_keep_tokens))
    )
    chunk_size = int(
        resolved.get("engine_prefill_chunk_size", args.engine_prefill_chunk_size)
    )
    return base + (chunk_size if is_prefill else 0)


def _trace_sparse_path_summary(
    args: argparse.Namespace,
    engine_kwargs: dict[str, Any] | None = None,
) -> dict[str, int | bool | str]:
    plan = _token_count_plan(args)
    prefill_threshold = _long_text_threshold(
        args,
        is_prefill=True,
        engine_kwargs=engine_kwargs,
    )
    decode_threshold = _long_text_threshold(
        args,
        is_prefill=False,
        engine_kwargs=engine_kwargs,
    )
    multiturn_base_prefix = int(args.system_prompt_len) + int(args.session_prefix_len)
    return {
        "history_update": str(args.history_update),
        "require_omnikv_prefill_path": bool(args.require_omnikv_prefill_path),
        "omnikv_prefill_long_text_threshold": prefill_threshold,
        "omnikv_decode_long_text_threshold": decode_threshold,
        "quest_sparse_decode_threshold": decode_threshold,
        "min_performance_prompt_len": int(args.min_performance_prompt_len),
        "min_cacheable_prefix_len": int(args.min_cacheable_prefix_len),
        "multiturn_base_prefix": int(multiturn_base_prefix),
        "shared_prefix_len": int(args.shared_prefix_len),
        "max_prompt_len": int(plan["max_prompt_len"]),
        "multiturn_first_prompt": int(plan["multiturn_first_prompt"]),
        "multiturn_max_prompt": int(plan["multiturn_max_prompt"]),
        "shared_prefix_max_prompt": int(plan["shared_prefix_max_prompt"]),
    }


def _validate_sparse_path_requirements(args: argparse.Namespace, cases: list[str], workloads: set[str]) -> None:
    chain_cases = [case for case in cases if case.startswith("chain_")]
    chain_errors: list[str] = []
    if chain_cases and args.history_update != "generated":
        chain_errors.append(
            "linear chain cases require --history_update generated so the next "
            "turn contains the exact prior model tokens."
        )
    if chain_cases and "shared_prefix" in workloads:
        chain_errors.append(
            "linear chain cases do not support the branching shared_prefix "
            "workload; use --workloads multiturn."
        )
    if chain_errors:
        raise ValueError("Invalid linear chain benchmark:\n- " + "\n- ".join(chain_errors))
    if args.allow_short_trace:
        return

    errors: list[str] = []
    token_plan = _token_count_plan(args)
    resolved_kwargs = _case_engine_kwargs(
        args,
        cases[0],
        token_plan["max_prompt_len"],
    )
    summary = _trace_sparse_path_summary(args, resolved_kwargs)
    prefill_threshold = int(summary["omnikv_prefill_long_text_threshold"])
    quest_threshold = int(summary["quest_sparse_decode_threshold"])
    shared_prompt = int(summary["shared_prefix_max_prompt"])
    multiturn_first = int(summary["multiturn_first_prompt"])
    multiturn_max = int(summary["multiturn_max_prompt"])
    min_prompt = int(args.min_performance_prompt_len)
    min_cacheable_prefix = int(args.min_cacheable_prefix_len)

    if min_prompt > 0:
        if "shared_prefix" in workloads and shared_prompt < min_prompt:
            errors.append(
                "shared_prefix prompt is too short for a stable performance benchmark: "
                f"shared_prefix_len + shared_suffix_len = {shared_prompt}, "
                f"min_performance_prompt_len = {min_prompt}."
            )
        if "multiturn" in workloads and multiturn_first < min_prompt:
            errors.append(
                "multiturn first prompt is too short for a stable performance benchmark: "
                f"system + session + user = {multiturn_first}, "
                f"min_performance_prompt_len = {min_prompt}."
            )

    if min_cacheable_prefix > 0 and any(case.startswith("prefix_") for case in cases):
        multiturn_base_prefix = int(args.system_prompt_len) + int(args.session_prefix_len)
        if "shared_prefix" in workloads and int(args.shared_prefix_len) < min_cacheable_prefix:
            errors.append(
                "shared_prefix cacheable prefix is too short to dominate noise: "
                f"shared_prefix_len = {args.shared_prefix_len}, "
                f"min_cacheable_prefix_len = {min_cacheable_prefix}."
            )
        if "multiturn" in workloads and multiturn_base_prefix < min_cacheable_prefix:
            errors.append(
                "multiturn reusable base prefix is too short to dominate noise: "
                f"system_prompt_len + session_prefix_len = {multiturn_base_prefix}, "
                f"min_cacheable_prefix_len = {min_cacheable_prefix}."
            )

    if "prefix_omnikv" in cases:
        if not args.require_omnikv_prefill_path:
            errors.append(
                "prefix_omnikv requires --require_omnikv_prefill_path for a TTFT/prefill sparse-path benchmark."
            )
        if "shared_prefix" in workloads and shared_prompt <= prefill_threshold:
            errors.append(
                "shared_prefix prompt is too short for OmniKV prefill sparse path: "
                f"shared_prefix_len + shared_suffix_len = {shared_prompt}, "
                f"threshold = sink + top + recent + chunk = {prefill_threshold}."
            )
        if "multiturn" in workloads and multiturn_first <= prefill_threshold:
            errors.append(
                "multiturn first prompt is too short for OmniKV prefill sparse path: "
                f"system + session + user = {multiturn_first}, "
                f"threshold = sink + top + recent + chunk = {prefill_threshold}."
            )

    if "prefix_quest" in cases:
        if "shared_prefix" in workloads and shared_prompt <= quest_threshold:
            errors.append(
                "shared_prefix prompt is too short for QuEST decode sparse path: "
                f"shared_prefix_len + shared_suffix_len = {shared_prompt}, "
                f"derived QuEST token budget = {quest_threshold}."
            )
        if "multiturn" in workloads and multiturn_max <= quest_threshold:
            errors.append(
                "multiturn trace is too short for QuEST decode sparse path: "
                f"max multiturn prompt = {multiturn_max}, "
                f"derived QuEST token budget = {quest_threshold}."
            )

    if errors:
        raise ValueError(
            "Prefix-cache performance benchmark trace is not valid for stable sparse/cache measurement. "
            "Increase prompt lengths and reusable prefix lengths, or lower sparse budgets; pass "
            "--allow_short_trace only for cache-lifecycle smoke tests.\n- " + "\n- ".join(errors)
        )


def _find_live_seq(llm: Any, seq_id: int) -> Any | None:
    for queue in (llm.scheduler.waiting, llm.scheduler.decoding):
        for seq in queue:
            if int(seq.seq_id) == int(seq_id):
                return seq
    return None


def _cache_stats(llm: Any) -> dict[str, int]:
    stats = llm.model_runner.runtime_state.free_slot_stats()
    return {str(key): int(value) for key, value in stats.items() if isinstance(value, (int, float))}


def _run_request_batch(
    *,
    llm: Any,
    specs: list[RequestSpec],
    tokenizer: Any,
    per_turn_path: Path,
    raw_output_path: Path,
    block_size: int,
    max_steps: int,
    session_chain_ids: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    from sparseengine import SamplingParams
    import torch

    states: dict[int, RequestState] = {}
    batch_start_s = time.perf_counter()
    active = set(states)
    step_count = 0
    zero_progress_steps = 0
    failure: Exception | None = None
    try:
        for spec in specs:
            sampling_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=int(spec.output_len),
            )
            requested_chain_id = (
                None
                if session_chain_ids is None
                else session_chain_ids.get(int(spec.session_id))
            )
            if hasattr(llm, "admit_request"):
                admission = llm.admit_request(
                    spec.prompt_token_ids,
                    sampling_params,
                    chain_id=requested_chain_id,
                )
                seq_id = int(admission.seq_id)
                if session_chain_ids is not None and admission.chain_id:
                    session_chain_ids[int(spec.session_id)] = str(
                        admission.chain_id
                    )
                state = RequestState(
                    spec=spec,
                    seq_id=seq_id,
                    add_s=time.perf_counter(),
                    chain_expected=session_chain_ids is not None,
                    chain_id=admission.chain_id,
                    chain_status=str(admission.chain_status),
                    reused_tokens=int(admission.reused_tokens),
                    prefilled_tokens=int(admission.prefilled_tokens),
                    prefix_cache_hit_len=int(admission.reused_tokens),
                )
            else:
                seq_id = llm.add_request(
                    spec.prompt_token_ids, sampling_params
                )
                state = RequestState(
                    spec=spec,
                    seq_id=int(seq_id),
                    add_s=time.perf_counter(),
                )
            states[int(seq_id)] = state
            active.add(int(seq_id))

        while active:
            if step_count >= max_steps:
                raise RuntimeError(f"Exceeded max_steps={max_steps} while running active requests.")
            step_count += 1
            _finished_outputs, num_tokens = llm.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            now_s = time.perf_counter()

            if num_tokens == 0:
                zero_progress_steps += 1
                if zero_progress_steps >= 50:
                    raise RuntimeError("llm.step() returned 0 repeatedly; scheduler may be stuck.")
            else:
                zero_progress_steps = 0

            for seq_id, token_ids in llm.last_step_token_outputs:
                seq_id = int(seq_id)
                if seq_id not in states:
                    continue
                state = states[seq_id]
                if state.first_token_s is None:
                    state.first_token_s = now_s
                    seq = _find_live_seq(llm, seq_id)
                    if seq is not None:
                        state.prefix_cache_hit_len = int(getattr(seq, "prefix_cache_hit_len", 0) or 0)
                        state.prefix_cache_hit_blocks = int(getattr(seq, "prefix_cache_hit_block_count", 0) or 0)
                state.generated_token_ids.extend(int(token_id) for token_id in token_ids)

            for seq_id in list(active):
                state = states[seq_id]
                if len(state.generated_token_ids) >= int(state.spec.output_len):
                    state.finish_s = now_s
                    if state.chain_id:
                        coordinator = getattr(
                            llm.model_runner.runtime_state,
                            "chain_cache_coordinator",
                            None,
                        )
                        if coordinator is None:
                            raise RuntimeError(
                                "Chain admission succeeded without a chain coordinator."
                            )
                        record = coordinator.index.lookup(state.chain_id)
                        state.physical_residency = list(
                            record.physical_slots_by_layer
                        )
                    active.discard(seq_id)
    except Exception as exc:
        failure = exc
        now_s = time.perf_counter()
        existing_keys = {state.spec.request_key for state in states.values()}
        for seq_id in active:
            state = states[seq_id]
            state.status = "model_failed"
            state.error_message = repr(exc)
            state.finish_s = state.finish_s or now_s
        next_failed_seq_id = -1
        for spec in specs:
            if spec.request_key in existing_keys:
                continue
            states[next_failed_seq_id] = RequestState(
                spec=spec,
                seq_id=next_failed_seq_id,
                add_s=batch_start_s,
                finish_s=now_s,
                status="model_failed",
                error_message=repr(exc),
            )
            next_failed_seq_id -= 1

    records = _write_request_records(
        states=states,
        tokenizer=tokenizer,
        per_turn_path=per_turn_path,
        raw_output_path=raw_output_path,
        batch_start_s=batch_start_s,
        block_size=int(block_size),
    )
    if failure is not None:
        raise failure
    return records


def _run_multiturn_workload(
    *,
    llm: Any,
    tokenizer: Any,
    vocab_ids: list[int],
    args: argparse.Namespace,
    rng: random.Random,
    block_size: int,
    per_turn_path: Path,
    raw_output_path: Path,
) -> list[dict[str, Any]]:
    shared_system = _sample_tokens(vocab_ids, rng, args.system_prompt_len)
    histories: dict[int, list[int]] = {}
    for session_id in range(int(args.sessions)):
        session_prefix_len = _sample_length(rng, args.session_prefix_len, args.session_prefix_min_len)
        histories[session_id] = shared_system + _sample_tokens(vocab_ids, rng, session_prefix_len)

    records: list[dict[str, Any]] = []
    session_chain_ids: dict[int, str] | None = (
        {}
        if str(
            getattr(llm.config, "resolved_prefix_cache_mode", "disabled")
        )
        == "chain"
        else None
    )
    for turn in range(int(args.turns)):
        specs: list[RequestSpec] = []
        prompt_by_session: dict[int, list[int]] = {}
        for session_id in range(int(args.sessions)):
            reusable_prefix_len = _multiturn_reusable_prefix_len(
                turn=turn,
                session_id=session_id,
                shared_system_len=len(shared_system),
                history_len=len(histories[session_id]),
            )
            user_len = _sample_length(rng, args.user_len, args.user_min_len)
            user_tokens = _sample_tokens(vocab_ids, rng, user_len)
            prompt = histories[session_id] + user_tokens
            prompt_by_session[session_id] = prompt
            if session_chain_ids is None:
                eligible_cache_tokens = _eligible_cache_tokens(
                    reusable_prefix_len, len(prompt), block_size
                )
                expected_reuse_tokens = reusable_prefix_len
            else:
                # The final sampled token has not run a forward pass and is
                # deliberately part of the next turn's prefill suffix.
                expected_reuse_tokens = (
                    max(0, reusable_prefix_len - 1) if turn > 0 else 0
                )
                eligible_cache_tokens = expected_reuse_tokens
            specs.append(
                RequestSpec(
                    request_key=f"mt_s{session_id:04d}_t{turn:03d}",
                    workload="multiturn",
                    phase="turn",
                    session_id=session_id,
                    turn=turn,
                    prompt_token_ids=prompt,
                    output_len=int(args.output_len),
                    eligible_cache_tokens=eligible_cache_tokens,
                    expected_reuse_tokens=expected_reuse_tokens,
                )
            )
        round_records = _run_request_batch(
            llm=llm,
            specs=specs,
            tokenizer=tokenizer,
            per_turn_path=per_turn_path,
            raw_output_path=raw_output_path,
            block_size=block_size,
            max_steps=int(args.max_steps_per_round),
            session_chain_ids=session_chain_ids,
        )
        records.extend(round_records)
        by_key = {record["request_key"]: record for record in round_records}
        raw_generated: dict[str, list[int]] = {}
        with raw_output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                if payload.get("workload") == "multiturn" and payload.get("turn") == turn:
                    raw_generated[payload["request_key"]] = [int(x) for x in payload["generated_token_ids"]]
        synthetic_history_tokens: dict[str, list[int]] = {}
        if args.history_update == "synthetic":
            for session_id in range(int(args.sessions)):
                request_key = f"mt_s{session_id:04d}_t{turn:03d}"
                synthetic_history_tokens[request_key] = _sample_tokens(vocab_ids, rng, int(args.output_len))
        for session_id in range(int(args.sessions)):
            request_key = f"mt_s{session_id:04d}_t{turn:03d}"
            generated = raw_generated.get(request_key, [])
            if by_key.get(request_key, {}).get("status") == "success":
                if args.history_update == "generated":
                    history_tokens = generated
                else:
                    history_tokens = synthetic_history_tokens[request_key]
                histories[session_id] = prompt_by_session[session_id] + history_tokens
    return records


def _run_shared_prefix_workload(
    *,
    llm: Any,
    tokenizer: Any,
    vocab_ids: list[int],
    args: argparse.Namespace,
    rng: random.Random,
    block_size: int,
    per_turn_path: Path,
    raw_output_path: Path,
) -> list[dict[str, Any]]:
    shared_prefix = _sample_tokens(vocab_ids, rng, args.shared_prefix_len)
    records: list[dict[str, Any]] = []

    if shared_prefix:
        warm_spec = RequestSpec(
            request_key="sp_warmup",
            workload="shared_prefix",
            phase="warmup",
            session_id=-1,
            turn=-1,
            prompt_token_ids=shared_prefix,
            output_len=max(2, int(args.output_len)),
            eligible_cache_tokens=0,
            expected_reuse_tokens=0,
        )
        records.extend(
            _run_request_batch(
                llm=llm,
                specs=[warm_spec],
                tokenizer=tokenizer,
                per_turn_path=per_turn_path,
                raw_output_path=raw_output_path,
                block_size=block_size,
                max_steps=int(args.max_steps_per_round),
            )
        )

    specs: list[RequestSpec] = []
    for req_idx in range(int(args.shared_prompts)):
        suffix_len = _sample_length(rng, args.shared_suffix_len, args.shared_suffix_min_len)
        suffix = _sample_tokens(vocab_ids, rng, suffix_len)
        prompt = shared_prefix + suffix
        specs.append(
            RequestSpec(
                request_key=f"sp_req{req_idx:04d}",
                workload="shared_prefix",
                phase="bench",
                session_id=req_idx,
                turn=0,
                prompt_token_ids=prompt,
                output_len=int(args.output_len),
                eligible_cache_tokens=_eligible_cache_tokens(len(shared_prefix), len(prompt), block_size),
                expected_reuse_tokens=len(shared_prefix),
            )
        )
    records.extend(
        _run_request_batch(
            llm=llm,
            specs=specs,
            tokenizer=tokenizer,
            per_turn_path=per_turn_path,
            raw_output_path=raw_output_path,
            block_size=block_size,
            max_steps=int(args.max_steps_per_round),
        )
    )
    return records


def _summarize_records(
    *,
    case_name: str,
    case_config: dict[str, Any],
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    engine_kwargs: dict[str, Any],
    cache_stats_before: dict[str, int],
    cache_stats_after: dict[str, int],
    peak_memory_gb: float,
    elapsed_s: float,
    decode_graph_before: list[dict[str, Any]] | None = None,
    decode_graph_after: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    success = [record for record in records if record.get("status") == "success"]
    failures = [record for record in records if record.get("status") != "success"]
    failure_status_counts: dict[str, int] = {}
    for record in failures:
        status = str(record.get("status") or "metric_failed")
        failure_status_counts[status] = failure_status_counts.get(status, 0) + 1
    if not failure_status_counts:
        summary_status = "success"
    elif len(failure_status_counts) == 1:
        summary_status = next(iter(failure_status_counts))
    else:
        summary_status = "mixed_failed"
    bench_success = [record for record in success if record.get("phase") != "warmup"]
    ttfts = [float(record["ttft_s"]) for record in bench_success]
    latencies = [float(record["latency_s"]) for record in bench_success]
    prompt_tokens = [int(record["prompt_tokens"]) for record in bench_success]
    generated_tokens = [int(record["generated_tokens"]) for record in bench_success]
    cached_tokens = [int(record["cached_tokens"]) for record in bench_success]
    eligible_tokens = [int(record["eligible_cache_tokens"]) for record in bench_success]
    total_prompt_tokens = sum(prompt_tokens)
    total_generated_tokens = sum(generated_tokens)
    total_cached_tokens = sum(cached_tokens)
    total_eligible_tokens = sum(eligible_tokens)
    trace_summary = _trace_sparse_path_summary(args, engine_kwargs)
    prefill_threshold = int(trace_summary["omnikv_prefill_long_text_threshold"])
    decode_threshold = int(trace_summary["omnikv_decode_long_text_threshold"])
    quest_threshold = int(trace_summary["quest_sparse_decode_threshold"])
    stats_delta = {
        key: int(cache_stats_after.get(key, 0)) - int(cache_stats_before.get(key, 0))
        for key in sorted(set(cache_stats_before) | set(cache_stats_after))
    }
    graph_required = bool(getattr(args, "decode_graph", False))
    graph_deltas = _decode_graph_rank_deltas(
        decode_graph_before or [],
        decode_graph_after or [],
    )
    graph_failures: list[str] = []
    if graph_required:
        if not graph_deltas:
            graph_failures.append("missing per-rank CUDA Graph counters")
        after_by_rank = {
            int(item["world_rank"]): item
            for item in (decode_graph_after or [])
        }
        for delta in graph_deltas:
            rank = int(delta["world_rank"])
            after = after_by_rank[rank]
            if int(after["capture_count"]) <= 0:
                graph_failures.append(
                    f"world rank {rank} did not capture a decode graph"
                )
            if int(delta["replay_count"]) <= 0:
                graph_failures.append(
                    f"world rank {rank} did not replay a decode graph"
                )
            if int(delta["eager_static_count"]) != 0:
                graph_failures.append(
                    f"world rank {rank} used eager static decode"
                )
            if int(delta["force_eager_count"]) != 0:
                graph_failures.append(
                    f"world rank {rank} used forced eager fallback"
                )
        if graph_failures:
            summary_status = "metric_failed"
            failure_status_counts["metric_failed"] = (
                failure_status_counts.get("metric_failed", 0) + 1
            )

    by_turn: dict[str, dict[str, Any]] = {}
    for record in bench_success:
        if record.get("workload") != "multiturn":
            continue
        key = f"turn_{int(record['turn'])}"
        bucket = by_turn.setdefault(key, {"count": 0, "ttft_s": [], "cached_tokens": 0, "eligible_cache_tokens": 0, "prompt_tokens": 0})
        bucket["count"] += 1
        bucket["ttft_s"].append(float(record["ttft_s"]))
        bucket["cached_tokens"] += int(record["cached_tokens"])
        bucket["eligible_cache_tokens"] += int(record["eligible_cache_tokens"])
        bucket["prompt_tokens"] += int(record["prompt_tokens"])
    per_turn = {
        key: {
            "count": value["count"],
            "mean_ttft_ms": _mean(value["ttft_s"]) * 1000.0,
            "p90_ttft_ms": _percentile(value["ttft_s"], 0.90) * 1000.0,
            "cache_hit_rate": (
                value["cached_tokens"] / value["prompt_tokens"] if value["prompt_tokens"] else 0.0
            ),
            "eligible_hit_rate": (
                value["cached_tokens"] / value["eligible_cache_tokens"] if value["eligible_cache_tokens"] else 0.0
            ),
        }
        for key, value in sorted(by_turn.items())
    }

    summary = {
        "case": case_name,
        "case_label": case_config["label"],
        "method": case_config["method"],
        "enable_prefix_caching": bool(case_config["enable_prefix_caching"]),
        "status": summary_status,
        "failure_status_counts": failure_status_counts,
        "requests": len(records),
        "bench_requests": len(bench_success),
        "success_requests": len(success),
        "failed_requests": len(failures),
        "elapsed_s": elapsed_s,
        "request_throughput": len(bench_success) / elapsed_s if elapsed_s > 0 else 0.0,
        "input_token_throughput": total_prompt_tokens / elapsed_s if elapsed_s > 0 else 0.0,
        "output_token_throughput": total_generated_tokens / elapsed_s if elapsed_s > 0 else 0.0,
        "total_prompt_tokens": total_prompt_tokens,
        "total_generated_tokens": total_generated_tokens,
        "total_cached_tokens": total_cached_tokens,
        "total_eligible_cache_tokens": total_eligible_tokens,
        "min_prompt_tokens": min(prompt_tokens) if prompt_tokens else 0,
        "max_prompt_tokens": max(prompt_tokens) if prompt_tokens else 0,
        "unique_prompt_lengths": len(set(prompt_tokens)),
        "cache_hit_rate": total_cached_tokens / total_prompt_tokens if total_prompt_tokens else 0.0,
        "eligible_cache_hit_rate": total_cached_tokens / total_eligible_tokens if total_eligible_tokens else 0.0,
        "logical_token_reuse_rate": total_cached_tokens / total_prompt_tokens if total_prompt_tokens else 0.0,
        "recomputed_prompt_tokens": total_prompt_tokens - total_cached_tokens,
        "hit_requests": sum(1 for value in cached_tokens if value > 0),
        "long_prefill_requests": sum(1 for record in bench_success if int(record["prompt_tokens"]) > prefill_threshold),
        "long_decode_requests": sum(1 for record in bench_success if int(record["prompt_tokens"]) > decode_threshold),
        "quest_sparse_decode_eligible_requests": sum(
            1 for record in bench_success if int(record["prompt_tokens"]) > quest_threshold
        ),
        "trace_sparse_path_summary": trace_summary,
        "mean_ttft_ms": _mean(ttfts) * 1000.0,
        "median_ttft_ms": _percentile(ttfts, 0.50) * 1000.0,
        "p90_ttft_ms": _percentile(ttfts, 0.90) * 1000.0,
        "p95_ttft_ms": _percentile(ttfts, 0.95) * 1000.0,
        "p99_ttft_ms": _percentile(ttfts, 0.99) * 1000.0,
        "mean_latency_ms": _mean(latencies) * 1000.0,
        "p90_latency_ms": _percentile(latencies, 0.90) * 1000.0,
        "peak_memory_gb": peak_memory_gb,
        "prefix_cache_stats_before": cache_stats_before,
        "prefix_cache_stats_after": cache_stats_after,
        "prefix_cache_stats_delta": stats_delta,
        "decode_graph_required": graph_required,
        "decode_graph_failures": graph_failures,
        "decode_graph_before": decode_graph_before or [],
        "decode_graph_after": decode_graph_after or [],
        "decode_graph_delta": graph_deltas,
        "per_turn": per_turn,
    }
    return summary


def _run_case_worker(case_name: str, args_dict: dict[str, Any], case_dir: str) -> None:
    args = argparse.Namespace(**args_dict)
    case_dir_path = Path(case_dir)
    case_dir_path.mkdir(parents=True, exist_ok=True)
    per_turn_path = case_dir_path / "per_turn_results.jsonl"
    raw_output_path = case_dir_path / "raw_outputs.jsonl"
    per_turn_path.write_text("", encoding="utf-8")
    raw_output_path.write_text("", encoding="utf-8")

    llm = None
    try:
        import torch
        from transformers import AutoTokenizer
        from sparseengine import LLM

        case_index = sorted(CASE_PRESETS).index(case_name)
        os.environ["SPARSEENGINE_MASTER_PORT"] = str(int(args.master_port_base) + case_index)
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        rng = random.Random(int(args.seed))
        block_size = _case_block_size(args, case_name)
        token_plan = _token_count_plan(args)
        engine_kwargs = _case_engine_kwargs(args, case_name, token_plan["max_prompt_len"])

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
        vocab_ids = _token_vocab(tokenizer)

        started_s = time.perf_counter()
        llm = LLM(args.model_path, **engine_kwargs)
        cache_stats_before = _cache_stats(llm)
        decode_graph_before = _decode_graph_rank_summaries(llm)

        records: list[dict[str, Any]] = []
        workloads = set(_split_csv(args.workloads))
        if "shared_prefix" in workloads:
            records.extend(
                _run_shared_prefix_workload(
                    llm=llm,
                    tokenizer=tokenizer,
                    vocab_ids=vocab_ids,
                    args=args,
                    rng=rng,
                    block_size=block_size,
                    per_turn_path=per_turn_path,
                    raw_output_path=raw_output_path,
                )
            )
        if "multiturn" in workloads:
            records.extend(
                _run_multiturn_workload(
                    llm=llm,
                    tokenizer=tokenizer,
                    vocab_ids=vocab_ids,
                    args=args,
                    rng=rng,
                    block_size=block_size,
                    per_turn_path=per_turn_path,
                    raw_output_path=raw_output_path,
                )
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - started_s
        peak_memory_gb = (
            torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        )
        cache_stats_after = _cache_stats(llm)
        decode_graph_after = _decode_graph_rank_summaries(llm)
        summary = _summarize_records(
            case_name=case_name,
            case_config=CASE_PRESETS[case_name],
            records=records,
            args=args,
            engine_kwargs=engine_kwargs,
            cache_stats_before=cache_stats_before,
            cache_stats_after=cache_stats_after,
            peak_memory_gb=peak_memory_gb,
            elapsed_s=elapsed_s,
            decode_graph_before=decode_graph_before,
            decode_graph_after=decode_graph_after,
        )
        (case_dir_path / "aggregate_metrics.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        error = {
            "case": case_name,
            "method": CASE_PRESETS[case_name]["method"],
            "enable_prefix_caching": bool(CASE_PRESETS[case_name]["enable_prefix_caching"]),
            "status": "oom" if "out of memory" in str(exc).lower() else "model_failed",
            "error_message": repr(exc),
            "traceback": traceback.format_exc(),
        }
        (case_dir_path / "aggregate_metrics.json").write_text(
            json.dumps(error, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        if llm is not None and hasattr(llm, "exit"):
            llm.exit()


def _read_case_summary(case_dir: Path, case_name: str) -> dict[str, Any]:
    path = case_dir / "aggregate_metrics.json"
    if not path.exists():
        return {"case": case_name, "status": "model_failed", "error_message": "missing aggregate_metrics.json"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {"case": case_name, "status": "metric_failed"}
    except Exception as exc:
        return {"case": case_name, "status": "metric_failed", "error_message": repr(exc)}


def _write_report(output_dir: Path, summaries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    sparse_path_thresholds = {
        str(summary.get("case", "")): summary.get("trace_sparse_path_summary", {})
        for summary in summaries
    }
    lines = [
        "# Prefix Cache Benchmark",
        "",
        f"- Model: `{args.model_path}`",
        f"- Workloads: `{args.workloads}`",
        f"- Sessions/turns: `{args.sessions}/{args.turns}`",
        f"- Shared prefix prompts: `{args.shared_prompts}`",
        f"- History update: `{args.history_update}`",
        f"- Sparse-path thresholds: `{sparse_path_thresholds}`",
        "",
        "| Case | Status | Method | Prefix cache | Requests | Long prefill reqs | QuEST sparse-decode reqs | Mean TTFT ms | P90 TTFT ms | Cache hit rate | Eligible hit rate | Recomputed prompt tokens | Peak GB |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            "| {case} | {status} | {method} | {prefix} | {requests} | {long_prefill} | {quest_decode} | {mean_ttft:.2f} | {p90_ttft:.2f} | {hit:.4f} | {eligible:.4f} | {recomputed} | {mem:.2f} |".format(
                case=summary.get("case", ""),
                status=summary.get("status", ""),
                method=summary.get("method", ""),
                prefix=str(summary.get("enable_prefix_caching", "")),
                requests=int(summary.get("bench_requests", 0) or 0),
                long_prefill=int(summary.get("long_prefill_requests", 0) or 0),
                quest_decode=int(summary.get("quest_sparse_decode_eligible_requests", 0) or 0),
                mean_ttft=float(summary.get("mean_ttft_ms", 0.0) or 0.0),
                p90_ttft=float(summary.get("p90_ttft_ms", 0.0) or 0.0),
                hit=float(summary.get("cache_hit_rate", 0.0) or 0.0),
                eligible=float(summary.get("eligible_cache_hit_rate", 0.0) or 0.0),
                recomputed=int(summary.get("recomputed_prompt_tokens", 0) or 0),
                mem=float(summary.get("peak_memory_gb", 0.0) or 0.0),
            )
        )
    output_dir.joinpath("report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_ledger(output_dir: Path, summaries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    ledger_jsonl, ledger_csv = default_ledger_paths(args.feature, benchmark_output_root())
    git = git_metadata(REPO_ROOT_FOR_IMPORT)
    for idx, summary in enumerate(summaries, start=1):
        status = summary.get("status", "model_failed")
        if status not in {"success", "invalid_run", "invalid_input", "model_failed", "parse_failed", "metric_failed", "mixed_failed", "skipped_by_policy", "oom", "timeout"}:
            status = "metric_failed"
        case_name = str(summary.get("case", f"case_{idx}"))
        record = {
            "run_id": f"{args.feature}_{_now_id()}_{git['git_commit']}_{idx:03d}_{case_name}",
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "feature": args.feature,
            "objective": args.objective,
            **git,
            "benchmark": "prefix_cache_trace",
            "benchmark_tier": "standalone",
            "benchmark_source": "repo_script",
            "script": "scripts/benchmarks/bench_prefix_cache.py",
            "command": _shell_command(),
            "model_path": args.model_path,
            "tokenizer_path": args.model_path,
            "method": summary.get("method"),
            "method_config": {
                "case": case_name,
                "enable_prefix_caching": summary.get("enable_prefix_caching"),
                "workloads": args.workloads,
                "sessions": args.sessions,
                "turns": args.turns,
                "system_prompt_len": args.system_prompt_len,
                "session_prefix_min_len": args.session_prefix_min_len,
                "session_prefix_len": args.session_prefix_len,
                "user_min_len": args.user_min_len,
                "user_len": args.user_len,
                "shared_prefix_len": args.shared_prefix_len,
                "shared_suffix_min_len": args.shared_suffix_min_len,
                "shared_suffix_len": args.shared_suffix_len,
                "output_len": args.output_len,
                "history_update": args.history_update,
            },
            "dataset": "synthetic_token_trace",
            "split": "synthetic",
            "sample_policy": "seeded_dynamic_multiturn",
            "lengths": _token_count_plan(args),
            "max_new_tokens": args.output_len,
            "decode_config": {"temperature": 0.0, "top_p": 1.0, "ignore_eos": True},
            "gpu": os.getenv("CUDA_VISIBLE_DEVICES", f"cuda_device={args.cuda_device}"),
            "env": selected_env_snapshot(),
            "output_dir": str(output_dir / case_name),
            "status": status,
            "primary_metrics": summary,
            "speedup": None,
            "memory_delta": None,
            "failure_summary": summary.get("error_message", "") if status != "success" else "",
            "decision": "keep" if status == "success" else "investigate",
            "notes": "prefix cache synthetic shared-prefix and dynamic multi-turn trace",
        }
        append_ledger_record(record, jsonl_path=ledger_jsonl, csv_path=ledger_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SparseEngine prefix cache on shared-prefix and dynamic multi-turn traces.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--cases", default="baseline_full,prefix_full,prefix_omnikv,prefix_quest")
    parser.add_argument("--workloads", default="shared_prefix,multiturn", help="Comma-separated: shared_prefix,multiturn")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--feature", default="prefix_cache")
    parser.add_argument("--objective", default="evaluate SparseEngine prefix cache on realistic multi-turn traces")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--continue_on_failure", action="store_true")
    parser.add_argument("--allow_short_trace", action="store_true", help="Allow cache-lifecycle smoke traces that do not enter sparse paths.")
    parser.add_argument("--min_performance_prompt_len", type=int, default=8192)
    parser.add_argument("--min_cacheable_prefix_len", type=int, default=8192)
    parser.add_argument("--cuda_device", default=None, help="Physical GPU id to expose through CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--case_timeout_s", type=float, default=0.0)
    parser.add_argument("--master_port_base", type=int, default=24000)

    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--history_update", choices=("synthetic", "generated"), default="synthetic")
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--system_prompt_len", type=int, default=16384)
    parser.add_argument("--session_prefix_len", type=int, default=2048)
    parser.add_argument("--session_prefix_min_len", type=int, default=None)
    parser.add_argument("--user_len", type=int, default=256)
    parser.add_argument("--user_min_len", type=int, default=None)
    parser.add_argument("--output_len", type=int, default=128)
    parser.add_argument("--shared_prompts", type=int, default=4)
    parser.add_argument("--shared_prefix_len", type=int, default=16384)
    parser.add_argument("--shared_suffix_len", type=int, default=2048)
    parser.add_argument("--shared_suffix_min_len", type=int, default=None)

    parser.add_argument("--gpu_memory_utilization", type=float, default=0.65)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--expert_parallel_size", type=int, default=1)
    parser.add_argument(
        "--decode_graph",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max_active_requests", type=int, default=4)
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192)
    parser.add_argument("--engine_prefill_chunk_size", type=int, default=4096)
    parser.add_argument("--max_model_len_margin", type=int, default=64)
    parser.add_argument("--hyper_params", default="{}")

    parser.add_argument("--prefix_cache_block_size", type=int, default=16)
    parser.add_argument("--prefix_cache_max_blocks", type=int, default=None)
    parser.add_argument("--prefix_cache_salt", default="prefix-cache-bench-v1")
    parser.add_argument("--quest_chunk_size", type=int, default=16)
    parser.add_argument("--sink_keep_tokens", type=int, default=8)
    parser.add_argument("--recent_keep_tokens", type=int, default=256)
    parser.add_argument("--decode_keep_tokens", type=int, default=2048)
    parser.add_argument("--require_omnikv_prefill_path", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--full_attention_layers",
        "--full_attention_layers",
        dest="full_attention_layers",
        default="0,1,2,4,7,14",
    )
    parser.add_argument("--max_steps_per_round", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = _canonical_cases(args.cases)
    workloads = set(_split_csv(args.workloads))
    unsupported_workloads = workloads - {"shared_prefix", "multiturn"}
    if unsupported_workloads:
        raise ValueError(f"Unsupported workloads: {sorted(unsupported_workloads)}")
    if args.output_len < 2:
        raise ValueError("--output_len must be >= 2 so per-request prefix-hit metadata remains observable.")
    _validate_sparse_path_requirements(args, cases, workloads)
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    time_tag = _now_id()
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else benchmark_output_root() / "prefix_cache" / time_tag
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    token_plan = _token_count_plan(args)
    case_engine_kwargs = {
        case: _case_engine_kwargs(args, case, token_plan["max_prompt_len"])
        for case in cases
    }
    plan = {
        "command": _shell_command(),
        "output_dir": str(output_dir),
        "cases": cases,
        "workloads": sorted(workloads),
        "token_count_plan": token_plan,
        "trace_sparse_path_summary": _trace_sparse_path_summary(
            args,
            case_engine_kwargs[cases[0]],
        ),
        "case_engine_kwargs": case_engine_kwargs,
        "git": git_metadata(REPO_ROOT_FOR_IMPORT),
        "env": selected_env_snapshot(),
    }
    (output_dir / "benchmark_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    run_info = {
        **git_metadata(REPO_ROOT_FOR_IMPORT),
        "command": _shell_command(),
        "model_path": args.model_path,
        "cases": cases,
        "workloads": sorted(workloads),
        "args": vars(args),
        "env": selected_env_snapshot(),
    }
    (output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plan, indent=2, ensure_ascii=False))

    if args.dry_run:
        return

    ctx = mp.get_context("spawn")
    summaries: list[dict[str, Any]] = []
    for case_name in cases:
        case_dir = output_dir / case_name
        process = ctx.Process(target=_run_case_worker, args=(case_name, vars(args), str(case_dir)))
        started_s = time.perf_counter()
        process.start()
        process.join(timeout=float(args.case_timeout_s) if args.case_timeout_s > 0 else None)
        elapsed_s = time.perf_counter() - started_s
        if process.is_alive():
            process.terminate()
            process.join()
            timeout_summary = {
                "case": case_name,
                "method": CASE_PRESETS[case_name]["method"],
                "enable_prefix_caching": bool(CASE_PRESETS[case_name]["enable_prefix_caching"]),
                "status": "timeout",
                "error_message": f"case exceeded timeout_s={args.case_timeout_s}",
                "elapsed_s": elapsed_s,
            }
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "aggregate_metrics.json").write_text(
                json.dumps(timeout_summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        elif process.exitcode != 0 and not args.continue_on_failure:
            summaries.append(_read_case_summary(case_dir, case_name))
            break
        summaries.append(_read_case_summary(case_dir, case_name))

    case_failure_status_counts: dict[str, int] = {}
    for summary in summaries:
        status = str(summary.get("status") or "metric_failed")
        if status == "success":
            continue
        case_failure_status_counts[status] = (
            case_failure_status_counts.get(status, 0) + 1
        )
    if not case_failure_status_counts:
        aggregate_status = "success"
    elif len(case_failure_status_counts) == 1:
        aggregate_status = next(iter(case_failure_status_counts))
    else:
        aggregate_status = "mixed_failed"
    aggregate = {
        "benchmark": "prefix_cache_trace",
        "status": aggregate_status,
        "failure_status_counts": case_failure_status_counts,
        "output_dir": str(output_dir),
        "cases": summaries,
    }
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "performance.jsonl").open("w", encoding="utf-8") as handle:
        for summary in summaries:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    _write_report(output_dir, summaries, args)
    _append_ledger(output_dir, summaries, args)

    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    if aggregate["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
