#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
SCBENCH_DIR = REPO_ROOT / "benchmark" / "scbench"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(SCBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(SCBENCH_DIR))

from benchmark.sparseengine_regression.manifest import (  # noqa: E402
    deltakv_checkpoint_path_for,
    load_manifest,
    missing_runtime_inputs,
    resolve_manifest_paths,
)
from eval_utils import (  # noqa: E402
    DATA_NAME_TO_MAX_NEW_TOKENS,
    create_multiturn_prompt,
)

DEFAULT_MAX_SEQ_LENGTH = 131_072


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def numeric_stats_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in sorted(set(before) | set(after))
    }


def usable_prefix_cache_tokens(prompt_len: int, block_size: int) -> int:
    prompt_len = int(prompt_len)
    block_size = int(block_size)
    if block_size <= 0 or prompt_len <= 1:
        return 0
    return ((prompt_len - 1) // block_size) * block_size


def common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if int(left_token) != int(right_token):
            break
        count += 1
    return count


def list_field(example: dict[str, Any], field: str) -> list[Any]:
    value = example[field]
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError(f"SCBench field {field!r} must be a list, got {type(value).__name__}.")
    return value


def encode_prompt_fragment(fragment: Any, tokenizer: Any) -> list[int]:
    if isinstance(fragment, list):
        return [int(token_id) for token_id in fragment]
    return [int(token_id) for token_id in tokenizer.encode(str(fragment), add_special_tokens=False)]


def eligible_cache_tokens(reusable_prefix_tokens: int, current_prompt_len: int, block_size: int) -> int:
    reusable = (int(reusable_prefix_tokens) // int(block_size)) * int(block_size)
    return min(reusable, usable_prefix_cache_tokens(current_prompt_len, block_size))


def cache_reuse_accounting(
    *,
    cached_tokens: int,
    planned_session_eligible_tokens: int,
) -> dict[str, int | str]:
    cached_tokens = int(cached_tokens)
    planned_session_eligible_tokens = int(planned_session_eligible_tokens)
    covered_tokens = min(cached_tokens, planned_session_eligible_tokens)
    beyond_tokens = max(0, cached_tokens - planned_session_eligible_tokens)
    if cached_tokens == 0:
        scope = "none"
    elif beyond_tokens > 0:
        scope = "beyond_planned_session_prefix"
    else:
        scope = "within_planned_session_prefix"
    return {
        "planned_session_prefix_covered_tokens": covered_tokens,
        "cached_tokens_beyond_planned_session_prefix": beyond_tokens,
        "cache_hit_scope": scope,
    }


def cache_reuse_totals(records: Sequence[dict[str, Any]]) -> tuple[int, int, int, int]:
    total_planned = 0
    total_covered = 0
    total_beyond = 0
    beyond_requests = 0
    for record in records:
        planned = int(
            record.get(
                "planned_session_eligible_cache_tokens",
                record.get("eligible_cache_tokens", 0),
            )
            or 0
        )
        cached = int(record.get("cached_tokens", 0) or 0)
        accounting = cache_reuse_accounting(
            cached_tokens=cached,
            planned_session_eligible_tokens=planned,
        )
        covered = int(accounting["planned_session_prefix_covered_tokens"])
        beyond = int(accounting["cached_tokens_beyond_planned_session_prefix"])
        total_planned += planned
        total_covered += covered
        total_beyond += beyond
        beyond_requests += int(beyond > 0)
    return total_planned, total_covered, total_beyond, beyond_requests


def truncate_token_ids(token_ids: list[int], max_length: int, manner: str = "middle") -> list[int]:
    if max_length < 0 or len(token_ids) <= max_length:
        return token_ids
    if manner != "middle":
        raise ValueError(f"Unsupported truncate manner: {manner!r}.")
    split = max_length // 2
    return token_ids[:split] + token_ids[-split:]


def truncate_by_tokens(text: str, tokenizer: Any, max_tokens: int) -> list[int]:
    token_ids = [int(token_id) for token_id in tokenizer.encode(text)]
    return truncate_token_ids(token_ids, max_tokens, manner="middle")


def load_scbench_dataset(data_name: str):
    local_data_dir = os.environ.get("SCBENCH_LOCAL_DATA_DIR")
    if not local_data_dir:
        raise FileNotFoundError(
            "SCBENCH_LOCAL_DATA_DIR is required for the SparseVLLM SCBench regression subset."
        )

    root = Path(local_data_dir)
    candidates = [
        (root / data_name / "test-00000-of-00001.parquet", "parquet"),
        (root / f"{data_name}.parquet", "parquet"),
        (root / "data" / f"{data_name}.jsonl", "json"),
        (root / f"{data_name}.jsonl", "json"),
    ]
    for path, loader in candidates:
        if path.exists():
            return load_dataset(loader, data_files=str(path), split="train")
    raise FileNotFoundError(
        f"SCBENCH_LOCAL_DATA_DIR={local_data_dir!r} does not contain a standard file "
        f"for task {data_name!r}."
    )


def select_examples(
    examples: Any,
    *,
    tokenizer: Any,
    num_eval_examples: int,
    start_example_id: int,
    context_min_tokens: int,
    context_max_tokens: int,
) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    for idx in range(len(examples)):
        if idx < int(start_example_id):
            continue
        eg = examples[idx]
        if isinstance(eg, str):
            eg = json.loads(eg)
        eg = dict(eg)
        context = eg.get("context", eg.get("input", ""))
        if not context and "prompts" in eg:
            prompts = list_field(eg, "prompts")
            context = prompts[0] if prompts else ""
        n_tokens = len(encode_prompt_fragment(context, tokenizer))
        if context_min_tokens >= 0 and n_tokens < context_min_tokens:
            continue
        if context_max_tokens >= 0 and n_tokens >= context_max_tokens:
            continue
        selected.append((idx, eg))
        if num_eval_examples != -1 and len(selected) >= int(num_eval_examples):
            break
    if not selected:
        raise ValueError(
            "SCBench selection produced zero examples. "
            f"num_eval_examples={num_eval_examples}, start_example_id={start_example_id}, "
            f"context_min_tokens={context_min_tokens}, context_max_tokens={context_max_tokens}."
        )
    return selected


def method_runtime_config(
    manifest: dict[str, Any],
    *,
    model_id: str,
    method_id: str,
    batch_size: int,
    max_seq_length: int,
    tensor_parallel_size: int,
    prefix_cache_block_size: int,
    gpu_memory_utilization: float | None,
    scbench_max_steps: int,
    prefix_cache_salt: str,
    decode_graph: bool = False,
) -> dict[str, Any]:
    method = manifest["methods"][method_id]
    cfg = dict(method.get("config") or {})
    cfg.update((method.get("model_configs") or {}).get(model_id, {}))
    cfg["sparse_method"] = method["sparse_method"]
    checkpoint_path = deltakv_checkpoint_path_for(manifest["models"][model_id], method)
    if checkpoint_path:
        cfg.setdefault("deltakv_checkpoint_path", checkpoint_path)
    cfg["enable_prefix_caching"] = method["sparse_method"] in {"vanilla", "omnikv", "quest"}
    cfg["decode_graph"] = bool(decode_graph)
    if decode_graph:
        cfg["decode_graph_capture_sampling"] = False
    cfg["max_model_len"] = int(max_seq_length)
    cfg["tensor_parallel_size"] = int(tensor_parallel_size)
    cfg["max_num_seqs_in_batch"] = int(batch_size)
    cfg["max_decoding_seqs"] = int(batch_size)
    cfg["max_num_batched_tokens"] = max(int(max_seq_length) * int(batch_size), int(max_seq_length))
    cfg["throughput_log_interval_s"] = 0.0
    cfg["scbench_max_steps"] = int(scbench_max_steps)
    cfg["prefix_cache_salt"] = str(prefix_cache_salt)

    if gpu_memory_utilization is not None:
        cfg["gpu_memory_utilization"] = float(gpu_memory_utilization)

    if method["sparse_method"] == "quest":
        quest_chunk_size = int(cfg.get("quest_chunk_size", prefix_cache_block_size))
        cfg["quest_chunk_size"] = quest_chunk_size
        if bool(manifest["models"][model_id].get("mixed_attention", False)):
            block_size = int(prefix_cache_block_size)
            if block_size <= 0 or block_size % quest_chunk_size != 0:
                raise ValueError(
                    "Mixed-attention Quest prefix cache block size must be a positive multiple "
                    f"of quest_chunk_size: block_size={block_size}, quest_chunk_size={quest_chunk_size}."
                )
            cfg["prefix_cache_block_size"] = block_size
        else:
            cfg["prefix_cache_block_size"] = quest_chunk_size
    else:
        cfg["prefix_cache_block_size"] = int(prefix_cache_block_size)
    return cfg


@dataclass
class ExampleState:
    source_idx: int
    example: dict[str, Any]
    encoded: dict[str, Any]
    input_ids: list[int] | None = None
    cache_prefix_token_ids: list[int] | None = None
    answers: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RequestSpec:
    data_name: str
    example_idx: int
    local_idx: int
    turn_idx: int
    prompt_token_ids: tuple[int, ...]
    reusable_prefix_tokens: int
    max_tokens: int


class BatchedSparseVLLMGenerator:
    def __init__(self, llm: Any, tokenizer: Any, *, max_steps: int):
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_steps = int(max_steps)
        self.request_counter = 0

    def cache_stats(self) -> dict[str, int]:
        model_runner = getattr(self.llm, "model_runner", None)
        cache_manager = getattr(model_runner, "cache_manager", None)
        if cache_manager is None or not hasattr(cache_manager, "free_slot_stats"):
            return {}
        raw_stats = cache_manager.free_slot_stats()
        return {
            str(key): int(value)
            for key, value in raw_stats.items()
            if isinstance(value, (int, float, bool))
        }

    def prefix_cache_block_size(self) -> int:
        config = getattr(self.llm, "config", None)
        return int(getattr(config, "prefix_cache_block_size", 16) or 16)

    def find_live_seq(self, seq_id: int) -> Any | None:
        scheduler = getattr(self.llm, "scheduler", None)
        if scheduler is None:
            return None
        for queue_name in ("waiting", "decoding"):
            for seq in getattr(scheduler, queue_name, []):
                if int(getattr(seq, "seq_id", -1)) == int(seq_id):
                    return seq
        return None

    @staticmethod
    def trace_metric_error(
        *,
        cached_tokens: int,
        cached_blocks: int,
        max_usable_tokens: int,
        block_size: int,
    ) -> str:
        if cached_tokens < 0:
            return f"cached_tokens={cached_tokens} is negative."
        if cached_blocks < 0:
            return f"cached_blocks={cached_blocks} is negative."
        if cached_tokens == 0 and cached_blocks == 0:
            return ""
        if block_size <= 0:
            return f"invalid prefix_cache_block_size={block_size}."
        if cached_tokens % block_size != 0:
            return f"cached_tokens={cached_tokens} is not block-aligned to {block_size}."
        if cached_blocks * block_size != cached_tokens:
            return f"cached_blocks={cached_blocks} does not match cached_tokens={cached_tokens}."
        if cached_tokens > max_usable_tokens:
            return (
                f"cached_tokens={cached_tokens} exceeds max_usable_cache_tokens="
                f"{max_usable_tokens}."
            )
        return ""

    def run_batch(self, specs: list[RequestSpec]) -> tuple[dict[int, str], list[dict[str, Any]]]:
        from sparseengine import SamplingParams as SparseSamplingParams

        block_size = self.prefix_cache_block_size()
        stats_before = self.cache_stats()
        seq_to_runtime: dict[int, dict[str, Any]] = {}
        seq_to_spec: dict[int, RequestSpec] = {}
        start_s = time.perf_counter()

        for spec in specs:
            prompt_token_ids = [int(token_id) for token_id in spec.prompt_token_ids]
            seq_id = self.llm.add_request(
                prompt_token_ids,
                SparseSamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    top_k=1,
                    max_tokens=int(spec.max_tokens),
                ),
            )
            self.request_counter += 1
            seq_to_spec[int(seq_id)] = spec
            seq_to_runtime[int(seq_id)] = {
                "request_idx": self.request_counter,
                "seq_id": int(seq_id),
                "first_token_s": None,
                "finish_s": None,
                "generated_token_ids": [],
                "cached_tokens": 0,
                "cached_blocks": 0,
                "status": "success",
                "error_message": "",
                "planned_session_eligible_tokens": eligible_cache_tokens(
                    spec.reusable_prefix_tokens,
                    len(spec.prompt_token_ids),
                    block_size,
                ),
                "max_usable_tokens": usable_prefix_cache_tokens(
                    len(spec.prompt_token_ids),
                    block_size,
                ),
            }

        step_count = 0
        zero_progress_steps = 0
        while not self.llm.is_finished():
            if step_count >= self.max_steps:
                raise RuntimeError(f"SparseVLLM SCBench batch exceeded max_steps={self.max_steps}.")
            step_count += 1
            finished_outputs, num_tokens = self.llm.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            now_s = time.perf_counter()

            if num_tokens == 0:
                zero_progress_steps += 1
                if zero_progress_steps >= 50:
                    raise RuntimeError("SparseVLLM scheduler made no progress for 50 steps.")
            else:
                zero_progress_steps = 0

            for output_seq_id, _token_ids in getattr(self.llm, "last_step_token_outputs", []):
                output_seq_id = int(output_seq_id)
                runtime = seq_to_runtime.get(output_seq_id)
                if runtime is None or runtime["first_token_s"] is not None:
                    continue
                runtime["first_token_s"] = now_s
                seq = self.find_live_seq(output_seq_id)
                if seq is not None:
                    runtime["cached_tokens"] = int(getattr(seq, "prefix_cache_hit_len", 0) or 0)
                    runtime["cached_blocks"] = int(getattr(seq, "prefix_cache_hit_block_count", 0) or 0)

            for output_seq_id, output_token_ids, _logprobs, _top_logprobs in finished_outputs:
                output_seq_id = int(output_seq_id)
                runtime = seq_to_runtime.get(output_seq_id)
                if runtime is None:
                    continue
                runtime["generated_token_ids"] = [int(token_id) for token_id in output_token_ids]
                runtime["finish_s"] = now_s

        stats_after = self.cache_stats()
        outputs: dict[int, str] = {}
        traces: list[dict[str, Any]] = []
        batch_end_s = time.perf_counter()
        for seq_id, spec in seq_to_spec.items():
            runtime = seq_to_runtime[seq_id]
            first_token_s = runtime["first_token_s"] or runtime["finish_s"] or batch_end_s
            finish_s = runtime["finish_s"] or batch_end_s
            generated_token_ids = runtime["generated_token_ids"]
            cached_tokens = int(runtime["cached_tokens"])
            cached_blocks = int(runtime["cached_blocks"])
            metric_error = self.trace_metric_error(
                cached_tokens=cached_tokens,
                cached_blocks=cached_blocks,
                max_usable_tokens=int(runtime["max_usable_tokens"]),
                block_size=block_size,
            )
            reuse_accounting = cache_reuse_accounting(
                cached_tokens=cached_tokens,
                planned_session_eligible_tokens=int(runtime["planned_session_eligible_tokens"]),
            )
            status = str(runtime["status"])
            error_message = str(runtime["error_message"])
            if metric_error:
                status = "metric_failed"
                error_message = metric_error if not error_message else f"{error_message}; {metric_error}"
            text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            outputs[seq_id] = text
            traces.append(
                {
                    "request_idx": int(runtime["request_idx"]),
                    "seq_id": int(seq_id),
                    "mode": "multi_turn",
                    "data_name": spec.data_name,
                    "example_id": int(spec.example_idx),
                    "turn_idx": int(spec.turn_idx),
                    "status": status,
                    "prompt_tokens": len(spec.prompt_token_ids),
                    "max_new_tokens": int(spec.max_tokens),
                    "generated_tokens": len(generated_token_ids),
                    "generated_token_ids": generated_token_ids,
                    "planned_reusable_prefix_tokens": int(spec.reusable_prefix_tokens),
                    "planned_session_reusable_prefix_tokens": int(spec.reusable_prefix_tokens),
                    "eligible_cache_tokens": int(runtime["planned_session_eligible_tokens"]),
                    "planned_session_eligible_cache_tokens": int(
                        runtime["planned_session_eligible_tokens"]
                    ),
                    "max_usable_cache_tokens": int(runtime["max_usable_tokens"]),
                    "cached_tokens": cached_tokens,
                    "cached_blocks": cached_blocks,
                    **reuse_accounting,
                    "ttft_s": float(first_token_s - start_s),
                    "latency_s": float(finish_s - start_s),
                    "error_message": error_message,
                    "prefix_cache_stats_before": stats_before,
                    "prefix_cache_stats_after": stats_after,
                    "prefix_cache_stats_delta": numeric_stats_delta(stats_before, stats_after),
                }
            )
        return outputs, traces


def prefix_summary(trace_path: Path, summary_path: Path) -> dict[str, Any]:
    records = read_jsonl(trace_path)
    success = [record for record in records if record.get("status") == "success"]
    failures = [record for record in records if record.get("status") != "success"]
    total_prompt_tokens = sum(int(record.get("prompt_tokens", 0) or 0) for record in success)
    total_generated_tokens = sum(int(record.get("generated_tokens", 0) or 0) for record in success)
    total_cached_tokens = sum(int(record.get("cached_tokens", 0) or 0) for record in success)
    total_cached_blocks = sum(int(record.get("cached_blocks", 0) or 0) for record in success)
    (
        total_planned_session_eligible_tokens,
        total_planned_session_covered_tokens,
        total_beyond_planned_session_tokens,
        beyond_planned_session_requests,
    ) = cache_reuse_totals(success)
    request_elapsed_s = sum(float(record.get("latency_s", 0.0) or 0.0) for record in success)
    status_counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "metric_failed"))
        status_counts[status] = status_counts.get(status, 0) + 1

    final_stats = records[-1].get("prefix_cache_stats_after", {}) if records else {}
    first_before = records[0].get("prefix_cache_stats_before", {}) if records else {}
    summary = {
        "status": "success" if records and not failures else ("skipped_by_policy" if not records else "metric_failed"),
        "trace_path": str(trace_path),
        "request_count": len(records),
        "success_requests": len(success),
        "failed_requests": len(failures),
        "status_counts": status_counts,
        "total_prompt_tokens": total_prompt_tokens,
        "total_generated_tokens": total_generated_tokens,
        "total_cached_tokens": total_cached_tokens,
        "total_cached_blocks": total_cached_blocks,
        "total_eligible_cache_tokens": total_planned_session_eligible_tokens,
        "total_planned_session_eligible_cache_tokens": total_planned_session_eligible_tokens,
        "total_planned_session_prefix_covered_tokens": total_planned_session_covered_tokens,
        "total_cached_tokens_beyond_planned_session_prefix": total_beyond_planned_session_tokens,
        "hit_requests": sum(1 for record in success if int(record.get("cached_tokens", 0) or 0) > 0),
        "beyond_planned_session_prefix_hit_requests": beyond_planned_session_requests,
        "cache_hit_rate": total_cached_tokens / total_prompt_tokens if total_prompt_tokens else 0.0,
        "eligible_cache_hit_rate": (
            total_planned_session_covered_tokens / total_planned_session_eligible_tokens
            if total_planned_session_eligible_tokens
            else 0.0
        ),
        "planned_session_prefix_coverage_rate": (
            total_planned_session_covered_tokens / total_planned_session_eligible_tokens
            if total_planned_session_eligible_tokens
            else 0.0
        ),
        "request_elapsed_s": request_elapsed_s,
        "request_throughput": len(success) / request_elapsed_s if request_elapsed_s > 0 else 0.0,
        "input_token_throughput": total_prompt_tokens / request_elapsed_s if request_elapsed_s > 0 else 0.0,
        "recomputed_prompt_tokens": total_prompt_tokens - total_cached_tokens,
        "prefix_cache_stats_final": final_stats,
        "prefix_cache_stats_delta": numeric_stats_delta(first_before, final_stats),
    }
    write_json(summary_path, summary)
    return summary


def decode_graph_status(llm: Any) -> dict[str, Any]:
    runner = getattr(getattr(llm, "model_runner", None), "decode_graph_runner", None)
    states = getattr(runner, "_graphs", {}) if runner is not None else {}
    graph_count = sum(
        1
        for state in getattr(states, "values", lambda: [])()
        if getattr(state, "graph", None) is not None
    )
    configured = bool(getattr(getattr(llm, "config", None), "decode_graph", False))
    return {
        "decode_graph_configured": configured,
        "decode_graph_runner_initialized": runner is not None,
        "decode_graph_state_count": int(len(states)) if states is not None else 0,
        "decode_graph_graph_count": int(graph_count),
        "decode_graph_last_state_key": str(getattr(runner, "last_state_key", None)) if runner is not None else None,
        "decode_graph_active": bool(configured and graph_count > 0),
    }


def prepare_states(
    *,
    data_name: str,
    examples: list[tuple[int, dict[str, Any]]],
    tokenizer: Any,
    use_chat_template: bool,
    disable_golden_context: bool,
    max_input_length: int,
    max_turns: int,
) -> list[ExampleState]:
    states: list[ExampleState] = []
    for source_idx, eg in examples:
        if "multi_turns" in eg:
            turns = list_field(eg, "multi_turns")
            if max_turns > 0 and len(turns) > max_turns:
                eg = {**eg, "multi_turns": turns[:max_turns]}
            encoded = create_multiturn_prompt(
                eg,
                data_name=data_name,
                tok=tokenizer,
                use_chat_template=use_chat_template,
                use_vllm=False,
                disable_golden_context=disable_golden_context,
            )
            encoded["prompts"][0] = truncate_by_tokens(encoded["prompts"][0], tokenizer, max_input_length)
        elif "prompts" in eg and "ground_truth" in eg:
            prompts = list_field(eg, "prompts")
            ground_truth = list_field(eg, "ground_truth")
            if len(prompts) == len(ground_truth) + 1:
                context_tokens = encode_prompt_fragment(
                    truncate_token_ids(encode_prompt_fragment(prompts[0], tokenizer), max_input_length),
                    tokenizer,
                )
                turn_prompts = prompts[1:]
                if max_turns > 0:
                    turn_prompts = turn_prompts[:max_turns]
                    ground_truth = ground_truth[:max_turns]
                encoded = {
                    "prompts": turn_prompts,
                    "ground_truth": ground_truth,
                    "context_token_ids": context_tokens,
                    "prompt_format": "preprocessed_scdq",
                }
            elif len(prompts) == len(ground_truth):
                if max_turns > 0:
                    prompts = prompts[:max_turns]
                    ground_truth = ground_truth[:max_turns]
                prompts = list(prompts)
                prompts[0] = truncate_token_ids(encode_prompt_fragment(prompts[0], tokenizer), max_input_length)
                encoded = {
                    "prompts": prompts,
                    "ground_truth": ground_truth,
                    "prompt_format": "preprocessed_multiturn",
                }
            else:
                raise ValueError(
                    "Preprocessed SCBench prompts must either match ground_truth length or contain "
                    f"one context prompt plus one prompt per answer: prompts={len(prompts)}, "
                    f"ground_truth={len(ground_truth)}."
                )
            if "task" in eg:
                task = list_field(eg, "task") if isinstance(eg["task"], (str, list)) else eg["task"]
                encoded["task"] = task[: len(encoded["ground_truth"])] if isinstance(task, list) else task
        else:
            raise KeyError("SCBench example must contain either 'multi_turns' or 'prompts'/'ground_truth'.")
        states.append(ExampleState(source_idx=int(source_idx), example=eg, encoded=encoded))
    return states


def build_turn_specs(
    states: list[ExampleState],
    *,
    data_name: str,
    tokenizer: Any,
    turn_idx: int,
    max_new_tokens: int | dict[str, int],
    disable_golden_context: bool,
) -> list[RequestSpec]:
    specs: list[RequestSpec] = []
    for local_idx, state in enumerate(states):
        encoded = state.encoded
        if isinstance(max_new_tokens, dict):
            max_tokens = int(max_new_tokens[encoded["task"][turn_idx]])
        else:
            max_tokens = int(max_new_tokens)

        if turn_idx == 0:
            if "context_token_ids" in encoded:
                prompt_token_ids = [int(token_id) for token_id in encoded["context_token_ids"]]
                prompt_token_ids += encode_prompt_fragment(encoded["prompts"][0], tokenizer)
            else:
                prompt_token_ids = encode_prompt_fragment(encoded["prompts"][0], tokenizer)
            reusable_prefix_tokens = 0
        elif "context_token_ids" in encoded:
            context_ids = [int(token_id) for token_id in encoded["context_token_ids"]]
            current_ids = encode_prompt_fragment(encoded["prompts"][turn_idx], tokenizer)
            prompt_token_ids = context_ids + current_ids
            prior_cache_prefix = (
                state.cache_prefix_token_ids
                if state.cache_prefix_token_ids is not None
                else context_ids
            )
            reusable_prefix_tokens = common_prefix_len(prior_cache_prefix, prompt_token_ids)
        else:
            if state.input_ids is None:
                state.input_ids = []
            if disable_golden_context and state.answers:
                state.input_ids = (
                    state.input_ids
                    + tokenizer.encode(state.answers[-1], add_special_tokens=False)
                    + [int(tokenizer.eos_token_id)]
                )
            current_ids = encode_prompt_fragment(encoded["prompts"][turn_idx], tokenizer)
            prompt_token_ids = [int(token_id) for token_id in state.input_ids + current_ids]
            prior_cache_prefix = (
                state.cache_prefix_token_ids
                if state.cache_prefix_token_ids is not None
                else state.input_ids
            )
            reusable_prefix_tokens = common_prefix_len(prior_cache_prefix, prompt_token_ids)

        specs.append(
            RequestSpec(
                data_name=data_name,
                example_idx=state.source_idx,
                local_idx=local_idx,
                turn_idx=int(turn_idx),
                prompt_token_ids=tuple(prompt_token_ids),
                reusable_prefix_tokens=int(reusable_prefix_tokens),
                max_tokens=max_tokens,
            )
        )
    return specs


def score_predictions(
    *,
    pred_path: Path,
    data_name: str,
    model_name_tag: str,
    max_seq_length: int,
) -> str:
    compute_scores = importlib.import_module("compute_scores").compute_scores
    return str(
        compute_scores(
            pred_path,
            data_name,
            model_name_tag,
            max_seq_length=max_seq_length,
            scdq_mode=False,
        )
    )


def run_task(
    *,
    llm: Any,
    tokenizer: Any,
    generator: BatchedSparseVLLMGenerator,
    method_id: str,
    data_name: str,
    output_dir: Path,
    num_eval_examples: int,
    start_example_id: int,
    max_turns: int,
    max_seq_length: int,
    use_chat_template: bool,
    disable_golden_context: bool,
    context_min_tokens: int,
    context_max_tokens: int,
    batch_size: int,
    model_name_tag: str,
) -> dict[str, Any]:
    max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
    if isinstance(max_new_tokens, dict):
        tokens_to_reserve = sum(int(value) for value in max_new_tokens.values())
    else:
        tokens_to_reserve = int(max_new_tokens) * int(max_turns)
    max_input_length = int(max_seq_length) - tokens_to_reserve
    if max_input_length <= 0:
        raise ValueError(
            f"max_seq_length={max_seq_length} leaves no room for input after reserving "
            f"{tokens_to_reserve} generated tokens for {data_name}."
        )

    dataset = load_scbench_dataset(data_name)
    examples = select_examples(
        dataset,
        tokenizer=tokenizer,
        num_eval_examples=num_eval_examples,
        start_example_id=start_example_id,
        context_min_tokens=context_min_tokens,
        context_max_tokens=context_max_tokens,
    )
    states = prepare_states(
        data_name=data_name,
        examples=examples,
        tokenizer=tokenizer,
        use_chat_template=use_chat_template,
        disable_golden_context=disable_golden_context,
        max_input_length=max_input_length,
        max_turns=max_turns,
    )
    effective_turns = min(len(state.encoded["prompts"]) for state in states)
    if max_turns > 0:
        effective_turns = min(effective_turns, max_turns)

    pred_path = output_dir / f"prediction_{data_name}_multi_turn.jsonl"
    sample_path = output_dir / f"sample_results_{data_name}_multi_turn.jsonl"
    trace_path = output_dir / f"prefix_cache_trace_{data_name}_multi_turn.jsonl"
    summary_path = output_dir / f"prefix_cache_summary_{data_name}_multi_turn.json"
    for path in (pred_path, sample_path, trace_path):
        path.write_text("", encoding="utf-8")

    for turn_idx in tqdm(range(effective_turns), desc=f"{method_id}/{data_name} turns"):
        specs = build_turn_specs(
            states,
            data_name=data_name,
            tokenizer=tokenizer,
            turn_idx=turn_idx,
            max_new_tokens=max_new_tokens,
            disable_golden_context=disable_golden_context,
        )
        for batch_start in range(0, len(specs), int(batch_size)):
            batch_specs = specs[batch_start : batch_start + int(batch_size)]
            seq_outputs, traces = generator.run_batch(batch_specs)
            seq_ids = [int(trace["seq_id"]) for trace in traces]
            by_seq = dict(zip(seq_ids, batch_specs))
            for trace in traces:
                append_jsonl(trace_path, trace)
                spec = by_seq[int(trace["seq_id"])]
                state = states[spec.local_idx]
                answer = seq_outputs[int(trace["seq_id"])]
                state.input_ids = list(spec.prompt_token_ids)
                state.cache_prefix_token_ids = list(spec.prompt_token_ids) + [
                    int(token_id) for token_id in trace.get("generated_token_ids", [])
                ]
                state.answers.append(answer)
                ground_truth = state.encoded["ground_truth"][spec.turn_idx]
                row = {
                    "id": int(state.source_idx),
                    "turn_idx": int(spec.turn_idx),
                    "prediction": answer,
                    "ground_truth": ground_truth,
                }
                if "task" in state.encoded:
                    row["task"] = state.encoded["task"][spec.turn_idx]
                append_jsonl(pred_path, row)
                append_jsonl(
                    sample_path,
                    {
                        "benchmark": "scbench",
                        "method": method_id,
                        "data_name": data_name,
                        "id": int(state.source_idx),
                        "turn_idx": int(spec.turn_idx),
                        "status": trace["status"],
                        "prompt_tokens": int(trace["prompt_tokens"]),
                        "max_new_tokens": int(trace["max_new_tokens"]),
                        "generated_tokens": int(trace["generated_tokens"]),
                        "cached_tokens": int(trace["cached_tokens"]),
                        "cached_blocks": int(trace["cached_blocks"]),
                        "prediction_path": str(pred_path),
                    },
                )
            if any(trace["status"] != "success" for trace in traces):
                raise RuntimeError(f"Metric failure detected in {trace_path}.")

    score = score_predictions(
        pred_path=pred_path,
        data_name=data_name,
        model_name_tag=model_name_tag,
        max_seq_length=max_seq_length,
    )
    pfx_summary = prefix_summary(trace_path, summary_path)
    return {
        "status": "success",
        "data_name": data_name,
        "score": score,
        "prediction_path": str(pred_path),
        "sample_results_path": str(sample_path),
        "prefix_summary_path": str(summary_path),
        "prefix_summary": pfx_summary,
        "num_examples": len(states),
        "num_turns": effective_turns,
    }


def run_method(
    *,
    manifest: dict[str, Any],
    model_id: str,
    method_id: str,
    tasks: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from sparseengine import LLM as SparseLLM

    model = manifest["models"][model_id]
    method_dir = Path(args.output_dir) / model_id / method_id
    method_dir.mkdir(parents=True, exist_ok=True)
    runtime_cfg = method_runtime_config(
        manifest,
        model_id=model_id,
        method_id=method_id,
        batch_size=int(args.batch_size),
        max_seq_length=int(args.max_seq_length),
        tensor_parallel_size=int(args.tensor_parallel_size),
        prefix_cache_block_size=int(args.prefix_cache_block_size),
        gpu_memory_utilization=args.gpu_memory_utilization,
        scbench_max_steps=int(args.scbench_max_steps),
        prefix_cache_salt=f"{args.prefix_cache_salt}:{model_id}:{method_id}",
        decode_graph=bool(args.decode_graph),
    )
    write_json(method_dir / "runtime_config.json", runtime_cfg)

    tokenizer_path = model.get("tokenizer_path") or model["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    scbench_max_steps = int(runtime_cfg.pop("scbench_max_steps"))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    llm = SparseLLM(model["model_path"], **runtime_cfg)
    generator = BatchedSparseVLLMGenerator(llm, tokenizer, max_steps=scbench_max_steps)
    model_name_tag = f"{Path(model['model_path']).name}_{method_id}_sparseengine_multi_turn_prefix"
    task_results: dict[str, Any] = {}
    graph_status: dict[str, Any] = {}
    started_s = time.perf_counter()
    try:
        for data_name in tasks:
            task_results[data_name] = run_task(
                llm=llm,
                tokenizer=tokenizer,
                generator=generator,
                method_id=method_id,
                data_name=data_name,
                output_dir=method_dir,
                num_eval_examples=int(args.num_eval_examples),
                start_example_id=int(args.start_example_id),
                max_turns=int(args.max_turns),
                max_seq_length=int(args.max_seq_length),
                use_chat_template=bool(args.use_chat_template),
                disable_golden_context=bool(args.disable_golden_context),
                context_min_tokens=int(args.context_min_tokens),
                context_max_tokens=int(args.context_max_tokens),
                batch_size=int(args.batch_size),
                model_name_tag=model_name_tag,
            )
        graph_status = decode_graph_status(llm)
        if bool(runtime_cfg.get("decode_graph")) and not bool(graph_status["decode_graph_active"]):
            raise RuntimeError(
                "decode_graph=True was configured for SCBench, but no active decode CUDA graph "
                f"was captured. graph_status={graph_status}."
            )
    finally:
        exit_fn = getattr(llm, "exit", None)
        if exit_fn is not None:
            exit_fn()
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    result = {
        "status": "success",
        "model_id": model_id,
        "method": method_id,
        "tasks": task_results,
        "elapsed_s": float(time.perf_counter() - started_s),
        "runtime_config_path": str(method_dir / "runtime_config.json"),
        "decode_graph_status": graph_status,
    }
    write_json(method_dir / "method_summary.json", result)
    return result


def child_command(args: argparse.Namespace, method_id: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--manifest",
        str(args.manifest),
        "--model_id",
        str(args.model_id),
        "--methods",
        method_id,
        "--tasks",
        str(args.tasks),
        "--output_dir",
        str(args.output_dir),
        "--num_eval_examples",
        str(int(args.num_eval_examples)),
        "--start_example_id",
        str(int(args.start_example_id)),
        "--max_turns",
        str(int(args.max_turns)),
        "--max_seq_length",
        str(int(args.max_seq_length)),
        "--batch_size",
        str(int(args.batch_size)),
        "--tensor_parallel_size",
        str(int(args.tensor_parallel_size)),
        "--prefix_cache_block_size",
        str(int(args.prefix_cache_block_size)),
        "--prefix_cache_salt",
        str(args.prefix_cache_salt),
        "--scbench_max_steps",
        str(int(args.scbench_max_steps)),
        "--context_min_tokens",
        str(int(args.context_min_tokens)),
        "--context_max_tokens",
        str(int(args.context_max_tokens)),
        "--single_method_child",
    ]
    if args.decode_graph:
        cmd.append("--decode_graph")
    if args.gpu_memory_utilization is not None:
        cmd.extend(["--gpu_memory_utilization", str(float(args.gpu_memory_utilization))])
    if args.trust_remote_code:
        cmd.append("--trust_remote_code")
    if args.use_chat_template:
        cmd.append("--use_chat_template")
    if args.disable_golden_context:
        cmd.append("--disable_golden_context")
    return cmd


def run_methods_in_subprocesses(
    *,
    args: argparse.Namespace,
    methods: list[str],
    tasks: list[str],
) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_info = {
        "status": "running",
        "model_id": args.model_id,
        "methods": methods,
        "tasks": tasks,
        "num_eval_examples": int(args.num_eval_examples),
        "max_turns": int(args.max_turns),
        "max_seq_length": int(args.max_seq_length),
        "batch_size": int(args.batch_size),
        "decode_graph": bool(args.decode_graph),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scbench_local_data_dir": os.environ.get("SCBENCH_LOCAL_DATA_DIR"),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method_commands": [],
    }
    write_json(output_dir / "run_info.json", run_info)

    results: dict[str, Any] = {}
    for method_id in methods:
        cmd = child_command(args, method_id)
        command_record = {
            "method": method_id,
            "cmd": cmd,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        run_info["method_commands"].append(command_record)
        write_json(output_dir / "run_info.json", run_info)
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True)
        command_record["returncode"] = int(proc.returncode)
        command_record["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if proc.returncode != 0:
            run_info["status"] = "model_failed"
            write_json(output_dir / "run_info.json", run_info)
            raise RuntimeError(f"SCBench child method {method_id!r} failed with exit code {proc.returncode}.")
        child_summary_path = output_dir / "scbench_methods_summary.json"
        with child_summary_path.open("r", encoding="utf-8") as handle:
            child_summary = json.load(handle)
        results.update(child_summary["results"])

    summary = {
        **run_info,
        "status": "success",
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    write_json(output_dir / "scbench_methods_summary.json", summary)
    summary_jsonl = output_dir / "scbench_methods_summary.jsonl"
    summary_jsonl.write_text("", encoding="utf-8")
    for method_id, method_result in results.items():
        for data_name, task_result in method_result["tasks"].items():
            append_jsonl(
                summary_jsonl,
                {
                    "model_id": args.model_id,
                    "method": method_id,
                    "data_name": data_name,
                    "score": task_result["score"],
                    "prefix_summary": task_result["prefix_summary"],
                    "decode_graph_status": method_result.get("decode_graph_status", {}),
                    "elapsed_s": method_result["elapsed_s"],
                },
            )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small batched SparseVLLM SCBench multi-turn subset.")
    parser.add_argument("--manifest", default=str(REPO_ROOT / "benchmark" / "sparseengine_regression" / "manifest.json"))
    parser.add_argument("--model_id", default="qwen3_4b")
    parser.add_argument("--methods", default="vanilla,omnikv,quest")
    parser.add_argument("--tasks", default="scbench_kv,scbench_qa_eng")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_eval_examples", type=int, default=4)
    parser.add_argument("--start_example_id", type=int, default=0)
    parser.add_argument("--max_turns", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--prefix_cache_block_size", type=int, default=16)
    parser.add_argument("--prefix_cache_salt", default="scbench-regression")
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--scbench_max_steps", type=int, default=200_000)
    parser.add_argument("--decode_graph", action="store_true")
    parser.add_argument("--context_min_tokens", type=int, default=-1)
    parser.add_argument("--context_max_tokens", type=int, default=-1)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--disable_golden_context", action="store_true")
    parser.add_argument("--single_method_child", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = parse_csv(args.tasks)
    methods = parse_csv(args.methods)
    unknown_tasks = sorted(set(tasks) - set(DATA_NAME_TO_MAX_NEW_TOKENS))
    if unknown_tasks:
        raise ValueError(f"Unknown SCBench tasks: {unknown_tasks}")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch_size must be > 0.")
    if int(args.max_turns) <= 0:
        raise ValueError("--max_turns must be > 0 for this multi-turn subset.")
    if len(methods) > 1 and not args.single_method_child:
        return run_methods_in_subprocesses(args=args, methods=methods, tasks=tasks)
    if args.single_method_child and len(methods) != 1:
        raise ValueError("--single_method_child requires exactly one method.")

    manifest = resolve_manifest_paths(load_manifest(args.manifest))
    missing_methods = sorted(set(methods) - set(manifest["methods"]))
    if missing_methods:
        raise ValueError(f"Unknown manifest methods: {missing_methods}")
    missing = []
    for method_id in methods:
        missing.extend(missing_runtime_inputs(manifest, args.model_id, method_id))
    if missing:
        raise FileNotFoundError(f"Missing runtime inputs for SCBench subset: {sorted(set(missing))}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_info = {
        "status": "running",
        "model_id": args.model_id,
        "methods": methods,
        "tasks": tasks,
        "num_eval_examples": int(args.num_eval_examples),
        "max_turns": int(args.max_turns),
        "max_seq_length": int(args.max_seq_length),
        "batch_size": int(args.batch_size),
        "decode_graph": bool(args.decode_graph),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scbench_local_data_dir": os.environ.get("SCBENCH_LOCAL_DATA_DIR"),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(output_dir / "run_info.json", run_info)

    results: dict[str, Any] = {}
    for method_id in methods:
        results[method_id] = run_method(
            manifest=manifest,
            model_id=args.model_id,
            method_id=method_id,
            tasks=tasks,
            args=args,
        )

    summary = {
        **run_info,
        "status": "success",
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    write_json(output_dir / "scbench_methods_summary.json", summary)
    for method_id, method_result in results.items():
        for data_name, task_result in method_result["tasks"].items():
            append_jsonl(
                output_dir / "scbench_methods_summary.jsonl",
                {
                    "model_id": args.model_id,
                    "method": method_id,
                    "data_name": data_name,
                    "score": task_result["score"],
                    "prefix_summary": task_result["prefix_summary"],
                    "decode_graph_status": method_result.get("decode_graph_status", {}),
                    "elapsed_s": method_result["elapsed_s"],
                },
            )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
