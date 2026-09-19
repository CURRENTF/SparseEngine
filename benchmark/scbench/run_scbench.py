# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import json
import os
import sys
import time
import copy
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Any, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Where to append scbench_eval.log. Default to repo-local outputs unless overridden.
BASE_PATH = os.environ.get(
    "SPARSEENGINE_OUTPUT_DIR",
    str(REPO_ROOT / "outputs"),
)

import torch
from benchmark.model_adapters.quantization import build_model_load_kwargs, restore_modules_to_dtype
from args import parse_args
from compute_scores import compute_scores
from datasets import load_dataset
from eval_utils import (
    DATA_NAME_TO_MAX_NEW_TOKENS,
    GreedySearch,
    GreedySearch_InfLLM,
    GreedySearch_Mamba2,
    GreedySearch_RetrAttn,
    GreedySearch_RetrAttn_Legacy,
    GreedySearch_vLLM,
    check_benchmark_availability,
    create_multiturn_prompt,
    create_scdq_prompt,
    dump_jsonl,
    get_compressed_examples,
    get_ground_truth,
    load_data,
)
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    MambaForCausalLM,
    Qwen2ForCausalLM,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils.import_utils import _is_package_available

SPARSEENGINE_ATTN_TYPE = "sparseengine"

LLM = None
SamplingParams = None
if _is_package_available("vllm"):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        LLM = None
        SamplingParams = None

LMCacheLLM = None
if _is_package_available("lmcache_vllm"):
    try:
        from lmcache_vllm.vllm import LLM as LMCacheLLM
        import lmcache_vllm
    except ImportError:
        LMCacheLLM = None

import random

try:
    from minference import MInference
except ImportError:
    MInference = None


def _numeric_stats_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in sorted(set(before) | set(after))
    }


def _usable_prefix_cache_tokens(prompt_len: int, block_size: int) -> int:
    prompt_len = int(prompt_len)
    block_size = int(block_size)
    if block_size <= 0 or prompt_len <= 1:
        return 0
    return ((prompt_len - 1) // block_size) * block_size


def _common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if int(left_token) != int(right_token):
            break
        count += 1
    return count


def _eligible_cache_tokens(reusable_prefix_tokens: int, current_prompt_len: int, block_size: int) -> int:
    reusable = (int(reusable_prefix_tokens) // int(block_size)) * int(block_size)
    return min(reusable, _usable_prefix_cache_tokens(current_prompt_len, block_size))


def _cache_reuse_accounting(
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


def _cache_reuse_totals(records: Sequence[dict[str, Any]]) -> tuple[int, int, int, int]:
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
        accounting = _cache_reuse_accounting(
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


def _prefix_trace_path(
    result_dir: Path,
    data_name: str,
    use_scdq: str,
    use_llmlingua: str,
    *,
    rank: int | None = None,
) -> Path:
    suffix = f".rank{rank}" if rank is not None else ""
    return result_dir / f"prefix_cache_trace_{data_name}{use_scdq}{use_llmlingua}{suffix}.jsonl"


def _prefix_summary_path(
    result_dir: Path,
    data_name: str,
    use_scdq: str,
    use_llmlingua: str,
) -> Path:
    return result_dir / f"prefix_cache_summary_{data_name}{use_scdq}{use_llmlingua}.json"


def _write_jsonl(records: list[dict[str, Any]], path: Path, *, append: bool = True) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _flush_sparseengine_prefix_trace(
    model: Any,
    path: Path,
    *,
    data_name: str,
    example_id: int,
) -> list[dict[str, Any]]:
    pop_records = getattr(model, "pop_prefix_cache_trace_records", None)
    if pop_records is None:
        return []
    records = pop_records()
    for record in records:
        record["data_name"] = data_name
        record["example_id"] = int(example_id)
    _write_jsonl(records, path, append=True)
    return records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_sparseengine_prefix_summary(trace_path: Path, summary_path: Path) -> dict[str, Any]:
    records = _read_jsonl(trace_path)
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
    ) = _cache_reuse_totals(success)
    request_elapsed_s = sum(float(record.get("latency_s", 0.0) or 0.0) for record in success)

    status_counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "metric_failed"))
        status_counts[status] = status_counts.get(status, 0) + 1

    final_stats = records[-1].get("prefix_cache_stats_after", {}) if records else {}
    stats_delta: dict[str, int] = {}
    if records:
        first_before = records[0].get("prefix_cache_stats_before", {}) or {}
        stats_delta = _numeric_stats_delta(first_before, final_stats)

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
        "prefix_cache_stats_delta": stats_delta,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def _maybe_write_sparseengine_prefix_summary(
    model: Any,
    trace_path: Path,
    summary_path: Path,
) -> dict[str, Any] | None:
    if not hasattr(model, "pop_prefix_cache_trace_records"):
        return None
    return _write_sparseengine_prefix_summary(trace_path, summary_path)


class SparseVLLMSCBenchSearch:
    def __init__(self, llm: Any, tokenizer: Any, *, max_steps: int = 200_000):
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_steps = int(max_steps)
        self._trace_records: list[dict[str, Any]] = []
        self._request_counter = 0
        self._last_cache_prefix_token_ids: list[int] | None = None
        self._closed = False

    def _cache_stats(self) -> dict[str, int]:
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

    def _prefix_cache_block_size(self) -> int:
        config = getattr(self.llm, "config", None)
        return int(getattr(config, "prefix_cache_block_size", 16) or 16)

    def _find_live_seq(self, seq_id: int) -> Any | None:
        scheduler = getattr(self.llm, "scheduler", None)
        if scheduler is None:
            return None
        for queue_name in ("waiting", "decoding"):
            for seq in getattr(scheduler, queue_name, []):
                if int(getattr(seq, "seq_id", -1)) == int(seq_id):
                    return seq
        return None

    @staticmethod
    def _trace_metric_error(
        *,
        cached_tokens: int,
        cached_blocks: int,
        max_usable_cache_tokens: int,
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
        if cached_tokens > max_usable_cache_tokens:
            return (
                f"cached_tokens={cached_tokens} exceeds max_usable_cache_tokens="
                f"{max_usable_cache_tokens}."
            )
        return ""

    def _run_one_request(
        self,
        prompt_token_ids: list[int],
        *,
        max_tokens: int,
        mode: str,
        turn_idx: int,
        reusable_prefix_tokens: int,
    ) -> str:
        from sparseengine import SamplingParams as SparseSamplingParams

        prompt_token_ids = [int(token_id) for token_id in prompt_token_ids]
        max_tokens = int(max_tokens)
        block_size = self._prefix_cache_block_size()
        eligible_tokens = _eligible_cache_tokens(
            reusable_prefix_tokens,
            len(prompt_token_ids),
            block_size,
        )
        max_usable_tokens = _usable_prefix_cache_tokens(len(prompt_token_ids), block_size)
        stats_before = self._cache_stats()
        start_s = time.perf_counter()
        seq_id = self.llm.add_request(
            prompt_token_ids,
            SparseSamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
            ),
        )
        self._request_counter += 1
        first_token_s: float | None = None
        finish_s: float | None = None
        generated_token_ids: list[int] = []
        cached_tokens = 0
        cached_blocks = 0
        status = "success"
        error_message = ""

        try:
            step_count = 0
            zero_progress_steps = 0
            while not self.llm.is_finished():
                if step_count >= self.max_steps:
                    raise RuntimeError(
                        f"Sparse-Engine SCBench request exceeded max_steps={self.max_steps}."
                    )
                step_count += 1
                finished_outputs, num_tokens = self.llm.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                now_s = time.perf_counter()

                if num_tokens == 0:
                    zero_progress_steps += 1
                    if zero_progress_steps >= 50:
                        raise RuntimeError("Sparse-Engine scheduler made no progress for 50 steps.")
                else:
                    zero_progress_steps = 0

                for output_seq_id, _token_ids in getattr(self.llm, "last_step_token_outputs", []):
                    if int(output_seq_id) != int(seq_id) or first_token_s is not None:
                        continue
                    first_token_s = now_s
                    seq = self._find_live_seq(seq_id)
                    if seq is not None:
                        cached_tokens = int(getattr(seq, "prefix_cache_hit_len", 0) or 0)
                        cached_blocks = int(getattr(seq, "prefix_cache_hit_block_count", 0) or 0)

                for output_seq_id, output_token_ids, _logprobs, _top_logprobs in finished_outputs:
                    if int(output_seq_id) == int(seq_id):
                        generated_token_ids = [int(token_id) for token_id in output_token_ids]
                        finish_s = now_s
        except Exception as exc:
            status = "model_failed"
            error_message = repr(exc)
            finish_s = time.perf_counter()
            raise
        finally:
            end_s = finish_s or time.perf_counter()
            if first_token_s is None:
                first_token_s = end_s
            metric_error = self._trace_metric_error(
                cached_tokens=cached_tokens,
                cached_blocks=cached_blocks,
                max_usable_cache_tokens=max_usable_tokens,
                block_size=block_size,
            )
            reuse_accounting = _cache_reuse_accounting(
                cached_tokens=cached_tokens,
                planned_session_eligible_tokens=eligible_tokens,
            )
            if metric_error:
                status = "metric_failed"
                error_message = metric_error if not error_message else f"{error_message}; {metric_error}"
            stats_after = self._cache_stats()
            record = {
                "request_idx": self._request_counter,
                "seq_id": int(seq_id),
                "mode": mode,
                "turn_idx": int(turn_idx),
                "status": status,
                "prompt_tokens": len(prompt_token_ids),
                "max_new_tokens": max_tokens,
                "generated_tokens": len(generated_token_ids),
                "generated_token_ids": generated_token_ids,
                "planned_reusable_prefix_tokens": int(reusable_prefix_tokens),
                "planned_session_reusable_prefix_tokens": int(reusable_prefix_tokens),
                "eligible_cache_tokens": int(eligible_tokens),
                "planned_session_eligible_cache_tokens": int(eligible_tokens),
                "max_usable_cache_tokens": int(max_usable_tokens),
                "cached_tokens": int(cached_tokens),
                "cached_blocks": int(cached_blocks),
                **reuse_accounting,
                "ttft_s": float(first_token_s - start_s),
                "latency_s": float(end_s - start_s),
                "error_message": error_message,
                "prefix_cache_stats_before": stats_before,
                "prefix_cache_stats_after": stats_after,
                "prefix_cache_stats_delta": _numeric_stats_delta(stats_before, stats_after),
            }
            self._trace_records.append(record)
            if status == "success":
                self._last_cache_prefix_token_ids = prompt_token_ids + generated_token_ids

        return self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    def pop_prefix_cache_trace_records(self) -> list[dict[str, Any]]:
        records = self._trace_records
        self._trace_records = []
        return records

    def test_scdq(self, example, max_length=100):
        results = []
        init_prompt_ids: list[int] | None = None
        cache_prefix_token_ids: list[list[int]] = []
        for idx, prompt in enumerate(example["prompts"]):
            if idx == 0:
                init_prompt_ids = [int(token_id) for token_id in prompt]
                continue
            if isinstance(max_length, dict):
                max_length_per_turn = max_length[example["task"][idx - 1]]
            else:
                max_length_per_turn = max_length
            if init_prompt_ids is None:
                raise RuntimeError("SCDQ prompt is missing the shared context prompt.")
            current_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = init_prompt_ids + current_ids
            reusable_prefix_tokens = len(init_prompt_ids)
            if cache_prefix_token_ids:
                reusable_prefix_tokens = max(
                    _common_prefix_len(prefix, input_ids) for prefix in cache_prefix_token_ids
                )
            results.append(
                self._run_one_request(
                    input_ids,
                    max_tokens=int(max_length_per_turn),
                    mode="scdq",
                    turn_idx=idx - 1,
                    reusable_prefix_tokens=reusable_prefix_tokens,
                )
            )
            last_cache_prefix = getattr(self, "_last_cache_prefix_token_ids", None)
            if last_cache_prefix is not None:
                cache_prefix_token_ids.append(list(last_cache_prefix))
        output = {"answers": results, "gt": example["ground_truth"]}
        if isinstance(max_length, dict):
            output["task"] = example["task"]
        return output

    def test(self, example, max_length=100, disable_golden_context=False):
        results = []
        input_ids: list[int] | None = None
        cache_prefix_token_ids: list[int] | None = None
        for idx, prompt in enumerate(example["prompts"]):
            if isinstance(max_length, dict):
                max_length_per_turn = max_length[example["task"][idx]]
            else:
                max_length_per_turn = max_length

            if idx == 0:
                input_ids = [int(token_id) for token_id in prompt]
                reusable_prefix_tokens = 0
            else:
                current_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                if input_ids is None:
                    input_ids = []
                prior_input_ids = input_ids
                if disable_golden_context and results:
                    input_ids = (
                        input_ids
                        + self.tokenizer.encode(results[-1], add_special_tokens=False)
                        + [self.tokenizer.eos_token_id]
                    )
                    prior_input_ids = input_ids
                input_ids = input_ids + current_ids
                reusable_prefix_tokens = _common_prefix_len(
                    cache_prefix_token_ids if cache_prefix_token_ids is not None else prior_input_ids,
                    input_ids,
                )

            answer = self._run_one_request(
                input_ids,
                max_tokens=int(max_length_per_turn),
                mode="multi_turn",
                turn_idx=idx,
                reusable_prefix_tokens=reusable_prefix_tokens,
            )
            results.append(answer)
            last_cache_prefix = getattr(self, "_last_cache_prefix_token_ids", None)
            if last_cache_prefix is not None:
                cache_prefix_token_ids = list(last_cache_prefix)
        output = {"answers": results, "gt": example["ground_truth"]}
        if isinstance(max_length, dict):
            output["task"] = example["task"]
        return output

    def clear(self):
        if self._closed:
            return
        self._closed = True
        exit_fn = getattr(self.llm, "exit", None)
        if exit_fn is not None:
            exit_fn()


# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
def truncate_input(input: list, max_length: int, manner="middle"):
    if max_length < 0:
        return input
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens or max_tokens < 0
    return tokens


def _shorten_val(v: Any) -> str:
    v = str(v)
    if "/" in v:
        v = os.path.basename(v.rstrip("/"))
    if len(v) > 40:
        v = v[:20] + ".." + v[-15:]
    return v


def _build_result_dir(args, real_model_name: str, scdq_mode: bool) -> tuple[Path, str, str, str]:
    disable_golden_context = "_disable_golden_context" if args.disable_golden_context else ""
    use_scdq = "_scdq" if scdq_mode else "_multi_turn"
    use_llmlingua = "_lingua" if args.use_llmlingua else ""

    verbalize_hyper_param = (
        f"_{'-'.join([f'{k}={v}' for k, v in args.hyper_param.items() if k != 'best_pattern'])}"
        if args.hyper_param
        else ""
    )
    verbalize_hyper_param = _shorten_val(verbalize_hyper_param)

    result_dir = Path(
        args.output_dir,
        f"{real_model_name}_{args.attn_type}{disable_golden_context}_{args.kv_type}{verbalize_hyper_param}",
    )
    real_model_name_tag = (
        f"{real_model_name}_{args.attn_type}{use_scdq}{disable_golden_context}_{args.kv_type}{verbalize_hyper_param}"
    )
    return result_dir, real_model_name_tag, use_scdq, use_llmlingua


def _merge_rank_jsonl(dst_path: Path, src_paths: list[Path]):
    rows = []
    for p in src_paths:
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

    rows.sort(key=lambda x: (x.get("id", -1), x.get("turn_idx", -1)))
    dump_jsonl(rows, dst_path)


def _merge_prefix_trace_jsonl(dst_path: Path, src_paths: list[Path]):
    rows = []
    for p in src_paths:
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

    rows.sort(
        key=lambda x: (
            x.get("example_id", -1),
            x.get("turn_idx", -1),
            x.get("request_idx", -1),
            x.get("seq_id", -1),
        )
    )
    dump_jsonl(rows, dst_path)


def _load_subset_indices(path: str | None) -> list[int] | None:
    if not path:
        return None

    subset_path = Path(path)
    if subset_path.suffix.lower() == ".json":
        with open(subset_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = payload.get("indices", [])
        if not isinstance(payload, list):
            raise ValueError(f"Invalid subset index file: {path}")
        return [int(x) for x in payload]

    indices = []
    with open(subset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            indices.append(int(line))
    return indices


def _filter_examples(
    examples,
    tok,
    subset_indices: list[int] | None = None,
    context_min_tokens: int = -1,
    context_max_tokens: int = -1,
):
    selected_indices = list(range(len(examples)))

    if subset_indices is not None:
        subset_set = set(int(i) for i in subset_indices)
        selected_indices = [i for i in selected_indices if i in subset_set]

    if context_min_tokens >= 0 or context_max_tokens >= 0:
        filtered = []
        lengths = []
        for i in selected_indices:
            context = examples[i]["context"]
            n_tokens = len(tok.encode(context, add_special_tokens=False))
            if context_min_tokens >= 0 and n_tokens < context_min_tokens:
                continue
            if context_max_tokens >= 0 and n_tokens >= context_max_tokens:
                continue
            filtered.append(i)
            lengths.append(n_tokens)
        selected_indices = filtered
        if lengths:
            print(
                "[SCBench] Context length filter kept "
                f"{len(lengths)} examples | min={min(lengths)} avg={sum(lengths)/len(lengths):.1f} max={max(lengths)}"
            )
        else:
            print("[SCBench] Context length filter kept 0 examples")

    if isinstance(examples, list):
        return [examples[i] for i in selected_indices]
    return examples.select(selected_indices)


def _load_scbench_dataset(data_name: str):
    local_data_dir = os.environ.get("SCBENCH_LOCAL_DATA_DIR")
    if local_data_dir:
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
            f"SCBENCH_LOCAL_DATA_DIR={local_data_dir!r} does not contain "
            f"a standard SCBench file for task {data_name!r}."
        )

    return load_dataset("microsoft/SCBench", data_name, split="test")


def _select_worker_cuda_device(rank: int, world_size: int) -> int:
    visible_devices = [
        device.strip()
        for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if device.strip()
    ]

    if world_size > 1 and visible_devices:
        if rank >= len(visible_devices):
            raise ValueError(
                f"Rank {rank} is out of range for CUDA_VISIBLE_DEVICES={visible_devices}"
            )
        target_visible_device = visible_devices[rank]
        print(
            f"[Rank {rank}] Using CUDA visible index {rank} (physical {target_visible_device})"
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    return rank


def _run_scbench_worker(
    rank: int,
    world_size: int,
    args,
    data_names: list[str],
    max_seq_length: int,
    scdq_mode: bool,
):
    local_cuda_device = _select_worker_cuda_device(rank, world_size)

    hyper_param = args.hyper_param.copy() if args.hyper_param else {}
    # Worker-local device index within the current visible device list.
    hyper_param["cuda_device"] = local_cuda_device

    model_name = args.model_name_or_path
    real_model_name = model_name.split("/")[-1]
    result_dir, _, use_scdq, use_llmlingua = _build_result_dir(args, real_model_name, scdq_mode)
    result_dir.mkdir(exist_ok=True, parents=True)

    model, tok = load_model(
        model_name,
        args.topk,
        args.starting_layer,
        args.topk_dims_file_path,
        args.use_sparq,
        attn_type=args.attn_type,
        max_seq_length=max_seq_length,
        is_search=args.is_search,
        kv_type=args.kv_type,
        trust_remote_code=args.trust_remote_code,
        kv_cache_cpu=args.kv_cache_cpu,
        kv_cache_cpu_device=args.kv_cache_cpu_device,
        tensor_parallel_size=args.tensor_parallel_size,
        hyper_param=hyper_param,
        copy_on_gpu=args.copy_on_gpu,
    )
    subset_indices = _load_subset_indices(args.subset_indices_file)

    for data_name in data_names:
        max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
        if isinstance(max_new_tokens, dict):
            assert (
                max(max_new_tokens.values()) <= max_seq_length
            ), "max_new_tokens must be less than max_seq_length"
        elif max_new_tokens >= max_seq_length:
            max_new_tokens = 500

        output_path = result_dir / f"prediction_{data_name}{use_scdq}{use_llmlingua}.rank{rank}.jsonl"
        trace_path = _prefix_trace_path(
            result_dir,
            data_name,
            use_scdq,
            use_llmlingua,
            rank=rank,
        )
        if hasattr(model, "pop_prefix_cache_trace_records"):
            trace_path.write_text("", encoding="utf-8")
        examples = _load_scbench_dataset(data_name)

        if args.use_llmlingua:
            compression_ratio = hyper_param.get("llmlingua_ratio", 3) if hyper_param else 3
            examples = get_compressed_examples(examples, data_name, args.data_dir, rate=1 / compression_ratio)

        examples = _filter_examples(
            examples,
            tok=tok,
            subset_indices=subset_indices,
            context_min_tokens=args.context_min_tokens,
            context_max_tokens=args.context_max_tokens,
        )

        max_turn_size = len(examples[0]["multi_turns"])
        if args.max_turns > 0 and args.max_turns < max_turn_size:
            examples = [{**eg, "multi_turns": eg["multi_turns"][: args.max_turns]} for eg in examples]
            max_turn_size = args.max_turns

        if args.num_eval_examples != -1:
            num_eval_examples = min(args.num_eval_examples, len(examples))
            if isinstance(examples, list):
                examples = examples[:num_eval_examples]
            else:
                examples = examples.select(range(num_eval_examples))

        preds = []
        for i in tqdm(range(len(examples)), desc=f"[Rank {rank}] {data_name}"):
            if i < args.start_example_id:
                continue
            if i % world_size != rank:
                continue

            eg = examples[i]

            if isinstance(eg, str):
                try:
                    eg = json.loads(eg)
                except:
                    pass

            if data_name in ["scbench_summary_with_needles", "scbench_repoqa_and_kv"]:
                tokens_to_sum = sum(list(max_new_tokens.values())) if isinstance(max_new_tokens, dict) else max_new_tokens
                max_input_length = max_seq_length - (tokens_to_sum * max_turn_size // 2)
            else:
                max_input_length = max_seq_length - max_new_tokens * max_turn_size

            pred = get_pred(
                model,
                eg,
                data_name,
                max_new_tokens,
                max_input_length=max_input_length,
                attn_type=args.attn_type,
                tok=tok,
                use_chat_template=args.use_chat_template,
                scdq_mode=scdq_mode,
                disable_golden_context=args.disable_golden_context,
            )
            gts = get_ground_truth(eg, data_name)
            for turn_idx, (ans, gt, turn) in enumerate(zip(pred["answers"], gts, eg["multi_turns"])):
                case = {
                    "id": i,
                    "turn_idx": turn_idx,
                    "prediction": ans,
                    "ground_truth": gt,
                }
                if "task" in pred:
                    case["task"] = pred["task"][turn_idx]
                if data_name == "scbench_repoqa":
                    case["lang"] = eg["lang"]
                    case["repo"] = eg["repo"]
                    case["func_name"] = turn["name"]
                if data_name == "scbench_repoqa_and_kv":
                    case["lang"] = eg["lang"]
                    case["repo"] = eg["repo"]
                    if turn["task"] == "scbench_repoqa":
                        case["func_name"] = turn["name"]
                if data_name == "scbench_kv_compressible":
                    case["task"] = eg["task"]
                preds.append(case)
            dump_jsonl(preds, output_path)
            _flush_sparseengine_prefix_trace(
                model,
                trace_path,
                data_name=data_name,
                example_id=i,
            )
            torch.cuda.empty_cache()

    try:
        model.clear()
    except Exception:
        pass


def get_pred(
    model,
    eg,
    data_name,
    max_new_tokens,
    max_input_length: int,
    attn_type: str = "vllm",
    tok=None,
    use_chat_template=False,
    scdq_mode=False,
    disable_golden_context=False,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    if scdq_mode:
        encoded_eg = create_scdq_prompt(
            eg,
            data_name=data_name,
            tok=tok,
            use_chat_template=use_chat_template,
            use_vllm=("vllm" in attn_type),
        )
    else:
        # multi-turn mode
        encoded_eg = create_multiturn_prompt(
            eg,
            data_name=data_name,
            tok=tok,
            use_chat_template=use_chat_template,
            use_vllm=("vllm" in attn_type),
            disable_golden_context=disable_golden_context,
        )
    context = truncate_by_tokens(
        encoded_eg["prompts"][0], model.tokenizer, max_input_length
    )
    encoded_eg["prompts"][0] = context
    if scdq_mode:
        # scdq mode has no action for disable_golden_context
        outputs = model.test_scdq(encoded_eg, max_length=max_new_tokens)
    else:
        # multi-turn mode test
        outputs = model.test(
            encoded_eg,
            max_length=max_new_tokens,
            disable_golden_context=disable_golden_context,
        )

    print("Chunked generation:", json.dumps(outputs, indent=2, ensure_ascii=False))
    return outputs


def load_model(
    model_name: str,
    topk: int = -1,
    starting_layer: int = -1,
    topk_dims_file_path: str = "",
    use_sparq: bool = False,
    attn_type: str = "vllm",
    max_seq_length: int = None,
    is_search: bool = False,
    kv_type: str = "",
    trust_remote_code: bool = False,
    kv_cache_cpu: bool = False,
    kv_cache_cpu_device: str = "cpu",
    tensor_parallel_size: int = 1,
    hyper_param: dict = None,
    copy_on_gpu: bool = False,
):
    hyper_param = hyper_param.copy() if hyper_param else {}
    if model_name == "THUDM/glm-4-9b-chat-1m":
        tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, revision="refs/pr/19"
        )
    else:
        tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
    # tok.pad_token = tok.eos_token

    if attn_type == SPARSEENGINE_ATTN_TYPE:
        from sparseengine import LLM as SparseLLM

        sparse_hyper_param = hyper_param.copy()
        scbench_max_steps = int(sparse_hyper_param.pop("scbench_max_steps", 200_000))
        cuda_device = sparse_hyper_param.pop("cuda_device", "auto")
        if cuda_device != "auto" and torch.cuda.is_available():
            torch.cuda.set_device(int(cuda_device))

        if max_seq_length is not None:
            sparse_hyper_param.setdefault("max_model_len", int(max_seq_length))
        sparse_hyper_param.setdefault("tensor_parallel_size", int(tensor_parallel_size))
        sparse_hyper_param.setdefault("throughput_log_interval_s", 0.0)

        llm = SparseLLM(model_name, **sparse_hyper_param)
        llm = SparseVLLMSCBenchSearch(llm, tok, max_steps=scbench_max_steps)
        print("Sparse-Engine model and tokenizer loaded.")
        return llm, tok

    if attn_type == "vllm_blend":
        if LMCacheLLM is None:
            raise ImportError("lmcache_vllm is required for attn_type='vllm_blend'.")
        llm = LMCacheLLM(
            model=model_name,
            enable_prefix_caching=True,
            max_model_len=max_seq_length,
            tensor_parallel_size=tensor_parallel_size,
            enable_chunked_prefill=False,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=0.5,
            swap_space=64,
        )
        llm = GreedySearch_vLLM(llm, tok)
    elif attn_type == "vllm_kv":
        if LLM is None:
            raise ImportError("vllm is required for attn_type='vllm_kv'.")
        llm = LLM(
            model=model_name,
            max_model_len=max_seq_length,
            tensor_parallel_size=tensor_parallel_size,
            enable_chunked_prefill=False,
            trust_remote_code=True,
            swap_space=64,
            enable_kvcompress=True,
            block_size=16,
            kv_head_bias_path=None,
            kv_head_bias_weight=0,
            disable_log_stats=True,
            prefill_metric_collection_window_size=32,
            prefill_metric_collection_block_size=4096,
            max_kv_per_compression=50_000_000,
            metric_aggregation="L2-sum",
            maxpool_metrics=True,
        )
        llm = GreedySearch_vLLM(
            llm,
            tok,
            is_kv_compress=True,
        )
    elif "vllm" in attn_type:
        if LLM is None:
            raise ImportError(f"vllm is required for attn_type={attn_type!r}.")
        # num_gpus
        llm = LLM(
            model=model_name,
            enable_prefix_caching="Jamba" not in model_name,
            max_model_len=max_seq_length,
            tensor_parallel_size=tensor_parallel_size,
            enable_chunked_prefill=False,
            trust_remote_code=trust_remote_code,
            swap_space=64,
        )
        if attn_type != "vllm":
            if MInference is not None:
                minference_patch = MInference(
                    attn_type,
                    model_name,
                    config_path=topk_dims_file_path,
                    starting_layer=starting_layer,
                    attn_kwargs=hyper_param,
                )
                llm = minference_patch(llm)
            else:
                print(f"Warning: minference is not installed. Skipping patch for {attn_type}")
        llm = GreedySearch_vLLM(llm, tok)
    else:
        runtime_hyper_param, model_load_kwargs, target_torch_dtype = build_model_load_kwargs(
            hyper_param,
            default_torch_dtype=torch.bfloat16,
        )
        worker_cuda_device = runtime_hyper_param.pop(
            "cuda_device",
            hyper_param.get("cuda_device", "auto"),
        )
        hf_device_map = "auto"
        if worker_cuda_device != "auto":
            hf_device_map = {"": int(worker_cuda_device)}
        explicit_torch_dtype = "torch_dtype" in hyper_param
        model_torch_dtype = target_torch_dtype if (model_load_kwargs or explicit_torch_dtype) else "auto"
        if MInference is not None:
            minference_patch = MInference(
                attn_type.replace("_sink", ""),
                model_name,
                config_path=topk_dims_file_path,
                starting_layer=starting_layer,
                kv_type=kv_type,
                is_search=is_search,
                kv_cache_cpu=kv_cache_cpu,
                kv_cache_cpu_device=kv_cache_cpu_device,
                attn_kwargs=runtime_hyper_param,
            )
        else:
            minference_patch = None
            print(f"Warning: minference is not installed. Skipping patch for {attn_type}")

        if "mamba" in model_name.lower() or "recurrentgemma" in model_name.lower():
            llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=model_torch_dtype,
                device_map=hf_device_map,
                resume_download=None,
                trust_remote_code=trust_remote_code,
                **model_load_kwargs,
            )
            if model_load_kwargs:
                restore_modules_to_dtype(llm, target_torch_dtype)
            llm = GreedySearch_Mamba2(llm, tok)

            return llm, tok
        else:
            llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=model_torch_dtype,
                device_map=hf_device_map,
                trust_remote_code=trust_remote_code,
                attn_implementation="flash_attention_2",
                **model_load_kwargs,
            )
            if model_load_kwargs:
                restore_modules_to_dtype(llm, target_torch_dtype)
            if minference_patch is not None:
                llm = minference_patch(llm)

        if attn_type == "inf_llm":
            llm = GreedySearch_InfLLM(llm.model, tok)
            return llm, tok
        elif kv_type in ["retr_attn", "kivi"]:
            llm = GreedySearch_RetrAttn(
                llm,
                tok,
            )
            return llm, tok

        llm = GreedySearch(
            llm,
            tok,
        )

    print("Model and tokenizer loaded.")
    return llm, tok


if __name__ == "__main__":
    args = parse_args()
    mp.set_start_method("spawn", force=True)
    args.hyper_param = args.hyper_param.copy() if args.hyper_param else {}
    if args.load_in_4bit:
        args.hyper_param["load_in_4bit"] = True
    if args.load_in_8bit:
        args.hyper_param["load_in_8bit"] = True
    if args.model_torch_dtype:
        args.hyper_param["torch_dtype"] = args.model_torch_dtype

    # check_benchmark_availability(args.data_dir)
    model_name = args.model_name_or_path
    max_seq_length = args.max_seq_length
    real_model_name = model_name.split("/")[-1]
    data_name = args.task
    scdq_mode = args.same_context_different_query

    if "," in data_name:
        data_names = data_name.split(",")
    else:
        data_names = [data_name]

    if max_seq_length == -1:
        max_seq_length = 160_000

    result_dir, real_model_name_tag, use_scdq, use_llmlingua = _build_result_dir(args, real_model_name, scdq_mode)
    result_dir.mkdir(exist_ok=True, parents=True)

    if args.ws > 1:
        procs = []
        for rank in range(args.ws):
            p = mp.Process(
                target=_run_scbench_worker,
                args=(rank, args.ws, args, data_names, max_seq_length, scdq_mode),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        for p in procs:
            if p.exitcode != 0:
                raise RuntimeError(f"SCBench worker exited with code {p.exitcode}")

        results = {}
        for data_name in data_names:
            merged_path = result_dir / f"prediction_{data_name}{use_scdq}{use_llmlingua}.jsonl"
            shard_paths = [
                result_dir / f"prediction_{data_name}{use_scdq}{use_llmlingua}.rank{rank}.jsonl"
                for rank in range(args.ws)
            ]
            _merge_rank_jsonl(merged_path, shard_paths)
            if args.attn_type == SPARSEENGINE_ATTN_TYPE:
                trace_path = _prefix_trace_path(result_dir, data_name, use_scdq, use_llmlingua)
                trace_shard_paths = [
                    _prefix_trace_path(
                        result_dir,
                        data_name,
                        use_scdq,
                        use_llmlingua,
                        rank=rank,
                    )
                    for rank in range(args.ws)
                ]
                _merge_prefix_trace_jsonl(trace_path, trace_shard_paths)
                _write_sparseengine_prefix_summary(
                    trace_path,
                    _prefix_summary_path(result_dir, data_name, use_scdq, use_llmlingua),
                )
            score = compute_scores(
                merged_path,
                data_name,
                real_model_name_tag,
                max_seq_length=max_seq_length,
                scdq_mode=scdq_mode,
            )
            results[data_name] = score
    else:
        # Model
        model, tok = load_model(
            model_name,
            args.topk,
            args.starting_layer,
            args.topk_dims_file_path,
            args.use_sparq,
            attn_type=args.attn_type,
            max_seq_length=max_seq_length,
            is_search=args.is_search,
            kv_type=args.kv_type,
            trust_remote_code=args.trust_remote_code,
            kv_cache_cpu=args.kv_cache_cpu,
            kv_cache_cpu_device=args.kv_cache_cpu_device,
            tensor_parallel_size=args.tensor_parallel_size,
            hyper_param=args.hyper_param.copy(),
            copy_on_gpu=args.copy_on_gpu,
        )
        subset_indices = _load_subset_indices(args.subset_indices_file)

        results = {}
        for data_name in data_names:
            max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
            if isinstance(max_new_tokens, dict):
                assert (
                    max(max_new_tokens.values()) <= max_seq_length
                ), "max_new_tokens must be less than max_seq_length"
            elif max_new_tokens >= max_seq_length:
                max_new_tokens = 500

            output_path = result_dir / f"prediction_{data_name}{use_scdq}{use_llmlingua}.jsonl"
            trace_path = _prefix_trace_path(result_dir, data_name, use_scdq, use_llmlingua)
            trace_summary_path = _prefix_summary_path(result_dir, data_name, use_scdq, use_llmlingua)
            if hasattr(model, "pop_prefix_cache_trace_records"):
                trace_path.write_text("", encoding="utf-8")
            examples = _load_scbench_dataset(data_name)

            if args.use_llmlingua:
                compression_ratio = args.hyper_param.get("llmlingua_ratio", 3) if args.hyper_param else 3
                examples = get_compressed_examples(examples, data_name, args.data_dir, rate=1 / compression_ratio)

            examples = _filter_examples(
                examples,
                tok=tok,
                subset_indices=subset_indices,
                context_min_tokens=args.context_min_tokens,
                context_max_tokens=args.context_max_tokens,
            )
            max_turn_size = len(examples[0]["multi_turns"])
            if args.max_turns > 0 and args.max_turns < max_turn_size:
                examples = [{**eg, "multi_turns": eg["multi_turns"][: args.max_turns]} for eg in examples]
                max_turn_size = args.max_turns

            if args.num_eval_examples != -1:
                num_eval_examples = min(args.num_eval_examples, len(examples))
                if isinstance(examples, list):
                    examples = examples[:num_eval_examples]
                else:
                    examples = examples.select(range(num_eval_examples))

            preds = []
            print(f"==== Evaluation {data_name}====")
            print(f"# examples: {len(examples)}")
            print(f"Num eval examples: {args.num_eval_examples}")
            print(f"Verbose: {args.verbose}")
            print(f"Max new tokens: {max_new_tokens}")
            print(f"Num of turns: {max_turn_size}")

            for i in tqdm(range(len(examples))):
                if i < args.start_example_id:
                    continue

                eg = examples[i]

                if isinstance(eg, str):
                    try:
                        eg = json.loads(eg)
                    except:
                        pass

                if data_name in ["scbench_summary_with_needles", "scbench_repoqa_and_kv"]:
                    tokens_to_sum = sum(list(max_new_tokens.values())) if isinstance(max_new_tokens, dict) else max_new_tokens
                    max_input_length = max_seq_length - (tokens_to_sum * max_turn_size // 2)
                else:
                    max_input_length = max_seq_length - max_new_tokens * max_turn_size

                pred = get_pred(
                    model,
                    eg,
                    data_name,
                    max_new_tokens,
                    max_input_length=max_input_length,
                    attn_type=args.attn_type,
                    tok=tok,
                    use_chat_template=args.use_chat_template,
                    scdq_mode=scdq_mode,
                    disable_golden_context=args.disable_golden_context,
                )
                gts = get_ground_truth(eg, data_name)
                for turn_idx, (ans, gt, turn) in enumerate(zip(pred["answers"], gts, eg["multi_turns"])):
                    case = {
                        "id": i,
                        "turn_idx": turn_idx,
                        "prediction": ans,
                        "ground_truth": gt,
                    }
                    if "task" in pred:
                        case["task"] = pred["task"][turn_idx]
                    if data_name == "scbench_repoqa":
                        case["lang"] = eg["lang"]
                        case["repo"] = eg["repo"]
                        case["func_name"] = turn["name"]
                    if data_name == "scbench_repoqa_and_kv":
                        case["lang"] = eg["lang"]
                        case["repo"] = eg["repo"]
                        if turn["task"] == "scbench_repoqa":
                            case["func_name"] = turn["name"]
                    if data_name == "scbench_kv_compressible":
                        case["task"] = eg["task"]
                    preds.append(case)
                dump_jsonl(preds, output_path)
                _flush_sparseengine_prefix_trace(
                    model,
                    trace_path,
                    data_name=data_name,
                    example_id=i,
                )
                torch.cuda.empty_cache()

            score = compute_scores(
                output_path,
                data_name,
                real_model_name_tag,
                max_seq_length=max_seq_length,
                scdq_mode=scdq_mode,
            )
            _maybe_write_sparseengine_prefix_summary(model, trace_path, trace_summary_path)
            results[data_name] = score

    print("==== Results ====")
    print(json.dumps(results, indent=2))

    # 记录评测信息到日志文件
    os.makedirs(BASE_PATH, exist_ok=True)
    log_path = os.path.join(BASE_PATH, "scbench_eval.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command: python {' '.join(sys.argv)}\n")
        f.write(f"Args: {json.dumps(vars(args), indent=2)}\n")
        f.write("-" * 80 + "\n")

    # 记录结果到日志
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"Evaluation Results:\n")
        f.write(json.dumps(results, indent=4, ensure_ascii=False))
        f.write("\n" + "="*80 + "\n\n")

    try:
        lmcache_vllm.close_lmcache_engine()
    except:
        pass
