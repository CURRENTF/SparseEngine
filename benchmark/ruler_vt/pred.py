#!/usr/bin/env python3
"""Evaluate SparseEngine methods on the self-contained RULER core task set.

The runner covers retrieval, multi-hop tracing, and aggregation without
requiring downloaded essays or QA corpora.  It uses this repo's native
SparseEngine inference path and RULER's string-match-all scoring contract.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from benchmark.model_adapters.sparseengine import get_sparseengine_generate_api
from benchmark.ruler_vt.tasks import (
    RulerSample,
    SUPPORTED_TASKS,
    canonical_task,
    generate_non_vt_samples,
    resolve_task_config,
)


TASK_TEMPLATE = (
    "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
    "{context}\n"
    "Question: Find all variables that are assigned the value {query} in the text above."
)
ANSWER_PREFIX = (
    " Answer: According to the chain(s) of variable assignment in the text above, "
    "{num_v} variables are assigned the value {query}, they are: "
)
HAYSTACK_SENTENCE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def normalize_prediction(text: str) -> str:
    text = text.strip()
    return re.sub(r"[\x00-\x1f]", "\n", text).strip()


def string_match_all(prediction: str, references: list[str]) -> float:
    pred = normalize_prediction(prediction).lower()
    return sum(1.0 if ref.lower() in pred else 0.0 for ref in references) / len(references)


VTSample = RulerSample


class VariableTrackingGenerator:
    def __init__(
        self,
        tokenizer,
        *,
        num_chains: int = 1,
        num_hops: int = 4,
        tokens_to_generate: int = 30,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_chains = num_chains
        self.num_hops = num_hops
        self.tokens_to_generate = tokens_to_generate

    def generate_chains(self, *, is_icl: bool = False) -> tuple[list[list[str]], list[list[str]]]:
        var_len = 3 if is_icl else 5
        total_vars = (self.num_hops + 1) * self.num_chains
        vars_all = [
            "".join(random.choices(string.ascii_uppercase, k=var_len)).upper()
            for _ in range(total_vars)
        ]
        while len(set(vars_all)) < total_vars:
            vars_all.append("".join(random.choices(string.ascii_uppercase, k=var_len)).upper())

        vars_ret: list[list[str]] = []
        chains_ret: list[list[str]] = []
        for start in range(0, len(vars_all), self.num_hops + 1):
            this_vars = vars_all[start : start + self.num_hops + 1]
            vars_ret.append(this_vars)
            first_value = "12345" if is_icl else str(np.random.randint(10000, 99999))
            this_chain = [f"VAR {this_vars[0]} = {first_value}"]
            for hop in range(self.num_hops):
                this_chain.append(f"VAR {this_vars[hop + 1]} = VAR {this_vars[hop]} ")
            chains_ret.append(this_chain)
        return vars_ret, chains_ret

    @staticmethod
    def shuffle_sublists(chains: list[list[str]]) -> list[str]:
        heap: list[tuple[float, int, int]] = []
        import heapq

        for chain_idx in range(len(chains)):
            heapq.heappush(heap, (random.random(), chain_idx, 0))

        shuffled: list[str] = []
        while heap:
            _, chain_idx, elem_idx = heapq.heappop(heap)
            shuffled.append(chains[chain_idx][elem_idx])
            if elem_idx + 1 < len(chains[chain_idx]):
                heapq.heappush(heap, (random.random(), chain_idx, elem_idx + 1))
        return shuffled

    def generate_input_output(self, num_noises: int, *, is_icl: bool = False) -> tuple[str, list[str], str]:
        variables, chains = self.generate_chains(is_icl=is_icl)
        value = chains[0][0].split("=")[-1].strip()
        sentences = [HAYSTACK_SENTENCE] * num_noises
        for chain in chains:
            positions = sorted(random.sample(range(len(sentences)), len(chain)))
            for insert_pos, hop_idx in zip(positions, range(len(chain))):
                sentences.insert(insert_pos + hop_idx, chain[hop_idx])
        context = "\n".join(sentences).replace(". \n", ".\n")
        text = (
            TASK_TEMPLATE.format(context=context, query=value)
            + ANSWER_PREFIX.format(num_v=self.num_hops + 1, query=value)
        )
        return text, variables[0], value

    def randomize_icl(self, icl_example: dict[str, Any]) -> str:
        icl = icl_example["input"] + " " + " ".join(icl_example["outputs"]) + "\n"
        for item in icl_example["outputs"]:
            icl = icl.replace(item, "".join(random.choices(string.ascii_uppercase, k=len(item))).upper())
        return icl.replace("12345", str(np.random.randint(10000, 99999)))

    def make_icl_example(self) -> dict[str, Any]:
        text, answer, _query = self.generate_input_output(5, is_icl=True)
        return {"input": text, "outputs": answer}

    def optimal_num_noises(self, max_seq_length: int, icl_example: dict[str, Any]) -> int:
        incremental = 10
        icl_tokens = count_tokens(
            self.tokenizer,
            icl_example["input"] + " " + " ".join(icl_example["outputs"]) + "\n",
        )
        sample_text, _answer, _query = self.generate_input_output(incremental, is_icl=False)
        tokens_per_haystack = count_tokens(self.tokenizer, sample_text) / incremental
        estimated_max_noises = int((max_seq_length / tokens_per_haystack) * 3)

        lower = incremental
        upper = max(estimated_max_noises, incremental * 2)
        optimal = incremental
        while lower <= upper:
            mid = (lower + upper) // 2
            text, _answer, _query = self.generate_input_output(mid, is_icl=False)
            total = count_tokens(self.tokenizer, text) + icl_tokens + self.tokens_to_generate
            if total <= max_seq_length:
                optimal = mid
                lower = mid + 1
            else:
                upper = mid - 1
        return optimal

    def generate_samples(
        self,
        *,
        context_lengths: list[int],
        samples_per_length: int,
    ) -> list[VTSample]:
        samples: list[VTSample] = []
        sample_idx = 0
        for context_length in context_lengths:
            icl_example = self.make_icl_example()
            num_noises = self.optimal_num_noises(context_length, icl_example)
            for _ in range(samples_per_length):
                used_noises = num_noises
                while True:
                    text, answer, query = self.generate_input_output(used_noises, is_icl=False)
                    cutoff = text.index(TASK_TEMPLATE[:20])
                    text = text[:cutoff] + self.randomize_icl(icl_example) + "\n" + text[cutoff:]
                    length = count_tokens(self.tokenizer, text) + self.tokens_to_generate
                    if length <= context_length or used_noises <= 10:
                        break
                    used_noises -= 10

                prefix_idx = text.rfind(ANSWER_PREFIX[:10])
                if prefix_idx < 0:
                    raise ValueError("Generated VT sample is missing the answer prefix.")
                answer_prefix = text[prefix_idx:]
                prompt = text[:prefix_idx]
                samples.append(
                    VTSample(
                        index=sample_idx,
                        context_length=context_length,
                        input=prompt,
                        outputs=answer,
                        length=length,
                        answer_prefix=answer_prefix,
                        query=query,
                        task="vt",
                        metadata={"category": "multi_hop_tracing"},
                    )
                )
                sample_idx += 1
        return samples


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_from_row(row: dict[str, Any]) -> VTSample:
    others = row.get("others") or {}
    return VTSample(
        index=int(row["index"]),
        context_length=int(row["context_length"]),
        input=str(row["input"]),
        outputs=list(row["outputs"]),
        length=int(row["length"]),
        answer_prefix=str(row["answer_prefix"]),
        query=str(others.get("query", "")),
        task=canonical_task(str(others.get("task", "vt"))),
        metadata=dict(others.get("metadata") or {}),
    )


def parse_context_lengths(value: str) -> list[int]:
    lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("context lengths must be a non-empty list of positive integers.")
    if len(set(lengths)) != len(lengths):
        raise ValueError(f"context lengths must be unique, got {lengths}.")
    return lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hyper-param", default=None)
    parser.add_argument("--hyper-param-json", default=None)
    parser.add_argument("--task", default="vt", choices=SUPPORTED_TASKS)
    parser.add_argument("--task-config-json", default=None)
    parser.add_argument("--sparse-method", default=None)
    parser.add_argument("--deltakv-checkpoint-path", default=None)
    parser.add_argument("--context-lengths", default="4096,8192,16384,32768,65536,98304")
    parser.add_argument("--samples-per-length", type=int, default=20)
    parser.add_argument("--num-chains", type=int, default=None)
    parser.add_argument("--num-hops", type=int, default=None)
    parser.add_argument("--tokens-to-generate", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--minimum-context-utilization", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ws", type=int, default=1, help="Number of single-GPU worker processes.")
    parser.add_argument("--worker-rank", type=int, default=-1)
    parser.add_argument("--worker-world-size", type=int, default=1)
    parser.add_argument(
        "--no-answer-prefix",
        action="store_true",
        help="Do not append RULER's answer_prefix to the prompt before generation.",
    )
    parser.add_argument(
        "--allow-prefix-caching",
        action="store_true",
        help="Explicitly allow enable_prefix_caching in the runtime config.",
    )
    parser.add_argument(
        "--prefix-cache-replay",
        action="store_true",
        help=(
            "Immediately replay every deterministic batch and save independent "
            "outputs plus cache-stat deltas."
        ),
    )
    parser.add_argument(
        "--require-prefix-cache-hit",
        action="store_true",
        help="Fail unless the replay pass observes prefix-cache hit requests and tokens.",
    )
    args = parser.parse_args()
    task_overrides: dict[str, Any] = {}
    if args.task_config_json:
        payload = json.loads(args.task_config_json)
        if not isinstance(payload, dict):
            raise ValueError("--task-config-json must decode to a JSON object.")
        task_overrides.update(payload)
    if args.num_chains is not None:
        task_overrides["num_chains"] = int(args.num_chains)
    if args.num_hops is not None:
        task_overrides["num_hops"] = int(args.num_hops)
    if args.tokens_to_generate is not None:
        task_overrides["tokens_to_generate"] = int(args.tokens_to_generate)
    if args.max_new_tokens is not None:
        task_overrides["max_new_tokens"] = int(args.max_new_tokens)
    args.task = canonical_task(args.task)
    args.task_config = resolve_task_config(args.task, task_overrides)
    args.num_chains = int(args.task_config.get("num_chains", 1))
    args.num_hops = int(args.task_config.get("num_hops", 1))
    args.tokens_to_generate = int(args.task_config["tokens_to_generate"])
    args.max_new_tokens = int(args.task_config["max_new_tokens"])
    return args


def generate_dataset(args: argparse.Namespace, output_dir: Path, tokenizer) -> list[VTSample]:
    context_lengths = parse_context_lengths(args.context_lengths)
    if args.task == "vt":
        generator = VariableTrackingGenerator(
            tokenizer,
            num_chains=args.num_chains,
            num_hops=args.num_hops,
            tokens_to_generate=args.tokens_to_generate,
        )
        samples = generator.generate_samples(
            context_lengths=context_lengths,
            samples_per_length=args.samples_per_length,
        )
    else:
        samples = generate_non_vt_samples(
            task=args.task,
            tokenizer=tokenizer,
            context_lengths=context_lengths,
            samples_per_length=args.samples_per_length,
            seed=args.seed,
            config=args.task_config,
        )
    dataset_rows = [
        {
            "index": sample.index,
            "input": sample.input,
            "outputs": sample.outputs,
            "length": sample.length,
            "context_length": sample.context_length,
            "answer_prefix": sample.answer_prefix,
            "others": {
                "query": sample.query,
                "task": sample.task,
                "metadata": sample.metadata,
            },
        }
        for sample in samples
    ]
    write_jsonl(output_dir / "dataset.jsonl", dataset_rows)
    return samples


def build_infer_config(args: argparse.Namespace, context_lengths: list[int]) -> dict[str, Any]:
    infer_config: dict[str, Any] = {}
    if int(args.tokens_to_generate) != int(args.max_new_tokens):
        raise ValueError(
            "--tokens-to-generate must equal --max-new-tokens so generated dataset "
            "lengths match the inference contract."
        )
    if args.max_model_len is not None and int(args.max_model_len) < max(context_lengths):
        raise ValueError(
            "--max-model-len must cover the largest RULER context length: "
            f"max_model_len={args.max_model_len} largest_context={max(context_lengths)}."
        )
    if args.hyper_param and args.hyper_param_json:
        raise ValueError("Pass only one of --hyper-param and --hyper-param-json.")
    if args.hyper_param:
        with open(args.hyper_param, "r", encoding="utf-8") as f:
            infer_config.update(json.load(f))
    if args.hyper_param_json:
        payload = json.loads(args.hyper_param_json)
        if not isinstance(payload, dict):
            raise ValueError("--hyper-param-json must decode to a JSON object.")
        infer_config.update(payload)
    tensor_parallel_size = int(infer_config.get("tensor_parallel_size", 1))
    if tensor_parallel_size <= 0:
        raise ValueError(
            f"tensor_parallel_size must be > 0, got {tensor_parallel_size}."
        )
    if int(args.ws) > 1 and tensor_parallel_size > 1:
        raise ValueError(
            "RULER does not support data-parallel --ws > 1 together with "
            "tensor_parallel_size > 1 because each data-parallel worker is "
            "assigned exactly one visible GPU; got "
            f"ws={args.ws}, tensor_parallel_size={tensor_parallel_size}."
        )
    prefix_caching = bool(infer_config.get("enable_prefix_caching", False))
    if prefix_caching and not args.allow_prefix_caching:
        raise ValueError(
            "RULER quality regression requires enable_prefix_caching=False unless "
            "--allow-prefix-caching is passed explicitly."
        )
    if args.prefix_cache_replay and not prefix_caching:
        raise ValueError("--prefix-cache-replay requires enable_prefix_caching=true.")
    if args.require_prefix_cache_hit and not args.prefix_cache_replay:
        raise ValueError("--require-prefix-cache-hit requires --prefix-cache-replay.")
    if args.prefix_cache_replay and args.temperature != 0.0:
        raise ValueError("--prefix-cache-replay requires deterministic temperature=0.")
    if not 0.0 < float(args.minimum_context_utilization) <= 1.0:
        raise ValueError(
            "--minimum-context-utilization must be in (0, 1], got "
            f"{args.minimum_context_utilization}."
        )
    infer_config["max_model_len"] = args.max_model_len or (
        max(context_lengths) + args.tokens_to_generate + 1024
    )
    return infer_config


def write_run_info(
    args: argparse.Namespace,
    output_dir: Path,
    tokenizer_path: str,
    infer_config: dict[str, Any],
) -> None:
    context_lengths = parse_context_lengths(args.context_lengths)

    run_info = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "command": "python " + " ".join(sys.argv),
        "cwd": os.getcwd(),
        "model_path": args.model_path,
        "tokenizer_path": tokenizer_path,
        "hyper_param": args.hyper_param,
        "infer_config": infer_config,
        "task": args.task,
        "task_config": args.task_config,
        "context_lengths": context_lengths,
        "samples_per_length": args.samples_per_length,
        "minimum_context_utilization": float(args.minimum_context_utilization),
        "num_chains": args.num_chains,
        "num_hops": args.num_hops,
        "append_answer_prefix": not args.no_answer_prefix,
        "prefix_cache_replay": bool(args.prefix_cache_replay),
        "require_prefix_cache_hit": bool(args.require_prefix_cache_hit),
        "seed": args.seed,
        "cuda_device": args.cuda_device,
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
        },
    }
    with (output_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)


def _cache_stats(generate_fn) -> dict[str, int]:
    llm = getattr(generate_fn, "_sparseengine_llm", None)
    if llm is None:
        raise RuntimeError(
            "SparseVLLM RULER generation did not expose _sparseengine_llm; "
            "cannot validate prefix-cache execution."
        )
    stats = llm.model_runner.runtime_state.free_slot_stats()
    return {
        str(key): int(value)
        for key, value in stats.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _stats_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in sorted(set(before) | set(after))
    }


def _generate_batch(generate, args: argparse.Namespace, prompts: list[str]) -> tuple[list[str], str, str | None]:
    try:
        predictions = generate(
            prompts if len(prompts) > 1 else prompts[0],
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
        )
        if isinstance(predictions, str):
            predictions = [predictions]
        else:
            predictions = list(predictions)
        if len(predictions) != len(prompts):
            raise RuntimeError(
                "SparseVLLM RULER generation returned the wrong batch size: "
                f"expected={len(prompts)} actual={len(predictions)}."
            )
        return predictions, "success", None
    except Exception as exc:  # Preserve one explicit failure row per requested sample.
        return ["" for _ in prompts], "model_failed", repr(exc)


def _append_batch_records(
    *,
    batch: list[VTSample],
    prompts: list[str],
    predictions: list[str],
    status: str,
    error: str | None,
    raw_path: Path,
    parsed_path: Path,
    result_path: Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for sample, prompt, prediction in zip(batch, prompts, predictions):
        parsed = normalize_prediction(prediction)
        score = string_match_all(parsed, sample.outputs) if status == "success" else 0.0
        result = {
            "index": sample.index,
            "task": sample.task,
            "context_length": sample.context_length,
            "length": sample.length,
            "prediction": parsed,
            "outputs": sample.outputs,
            "score": score,
            "correct": score == 1.0,
            "status": status,
            "error": error,
        }
        with raw_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "index": sample.index,
                        "task": sample.task,
                        "context_length": sample.context_length,
                        "length": sample.length,
                        "prompt": prompt,
                        "raw_output": prediction,
                        "status": status,
                        "error": error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        with parsed_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "index": sample.index,
                        "task": sample.task,
                        "context_length": sample.context_length,
                        "prediction": parsed,
                        "outputs": sample.outputs,
                        "status": status,
                        "error": error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        with result_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        results.append(result)
    return results


def evaluate_samples(
    args: argparse.Namespace,
    samples: list[VTSample],
    output_dir: Path,
    infer_config: dict[str, Any],
) -> None:
    rank_suffix = "" if args.worker_rank < 0 else f"_rank{args.worker_rank}"
    raw_path = output_dir / f"raw_outputs{rank_suffix}.jsonl"
    parsed_path = output_dir / f"parsed_outputs{rank_suffix}.jsonl"
    result_path = output_dir / f"per_sample_results{rank_suffix}.jsonl"
    replay_raw_path = output_dir / f"raw_outputs_prefix_cache_replay{rank_suffix}.jsonl"
    replay_parsed_path = output_dir / f"parsed_outputs_prefix_cache_replay{rank_suffix}.jsonl"
    replay_result_path = output_dir / f"per_sample_results_prefix_cache_replay{rank_suffix}.jsonl"
    paths = [raw_path, parsed_path, result_path]
    if args.prefix_cache_replay:
        paths.extend([replay_raw_path, replay_parsed_path, replay_result_path])
    for path in paths:
        path.write_text("", encoding="utf-8")

    tokenizer_path = args.tokenizer_path or args.model_path
    generate = get_sparseengine_generate_api(
        model_path=args.model_path,
        infer_config=infer_config,
        deltakv_checkpoint_path=args.deltakv_checkpoint_path,
        sparse_method=args.sparse_method,
    )

    rank = max(int(args.worker_rank), 0)
    prefix_trace_path = output_dir / f"prefix_cache_trace_rank{rank}.jsonl"
    if args.prefix_cache_replay:
        prefix_trace_path.write_text("", encoding="utf-8")
    stats_before = _cache_stats(generate) if args.prefix_cache_replay else {}
    replay_hit_requests = 0
    replay_hit_tokens = 0
    replay_mismatches: list[int] = []
    for start in tqdm(
        range(0, len(samples), args.batch_size),
        desc=f"RULER-{getattr(args, 'task', 'vt')}",
    ):
        batch = samples[start : start + args.batch_size]
        prompts = [
            sample.input if args.no_answer_prefix else sample.input + sample.answer_prefix
            for sample in batch
        ]
        predictions, status, error = _generate_batch(generate, args, prompts)
        primary_results = _append_batch_records(
            batch=batch,
            prompts=prompts,
            predictions=predictions,
            status=status,
            error=error,
            raw_path=raw_path,
            parsed_path=parsed_path,
            result_path=result_path,
        )
        if not args.prefix_cache_replay:
            continue

        stats_before_replay = _cache_stats(generate)
        replay_predictions, replay_status, replay_error = _generate_batch(generate, args, prompts)
        replay_results = _append_batch_records(
            batch=batch,
            prompts=prompts,
            predictions=replay_predictions,
            status=replay_status,
            error=replay_error,
            raw_path=replay_raw_path,
            parsed_path=replay_parsed_path,
            result_path=replay_result_path,
        )
        stats_after_replay = _cache_stats(generate)
        delta = _stats_delta(stats_before_replay, stats_after_replay)
        replay_hit_requests += int(delta.get("prefix_cache_hit_requests", 0))
        replay_hit_tokens += int(delta.get("prefix_cache_hit_tokens", 0))
        mismatches = [
            int(primary["index"])
            for primary, replay in zip(primary_results, replay_results)
            if primary["status"] != replay["status"]
            or primary["prediction"] != replay["prediction"]
            or primary["score"] != replay["score"]
        ]
        replay_mismatches.extend(mismatches)
        with prefix_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "sample_indices": [int(sample.index) for sample in batch],
                        "stats_before_replay": stats_before_replay,
                        "stats_after_replay": stats_after_replay,
                        "stats_delta": delta,
                        "output_mismatch_indices": mismatches,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    if args.prefix_cache_replay:
        stats_after = _cache_stats(generate)
        with (output_dir / f"prefix_cache_stats_rank{rank}.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "status": "success",
                    "rank": rank,
                    "stats_before": stats_before,
                    "stats_after": stats_after,
                    "stats_delta": _stats_delta(stats_before, stats_after),
                    "replay_hit_requests": int(replay_hit_requests),
                    "replay_hit_tokens": int(replay_hit_tokens),
                    "output_mismatch_indices": replay_mismatches,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")


def write_aggregate(
    output_dir: Path,
    context_lengths: list[int],
    elapsed_seconds: float,
    *,
    task: str = "vt",
) -> dict[str, Any]:
    result_rows = read_jsonl(output_dir / "per_sample_results.jsonl")
    by_length: dict[str, dict[str, Any]] = {}
    for context_length in context_lengths:
        rows = [row for row in result_rows if row["context_length"] == context_length]
        if not rows:
            continue
        by_length[str(context_length)] = {
            "score": round(100 * float(np.mean([row["score"] for row in rows])), 2),
            "exact_match": round(100 * float(np.mean([row["correct"] for row in rows])), 2),
            "num_samples": len(rows),
            "num_success": sum(row["status"] == "success" for row in rows),
            "mean_input_tokens": round(float(np.mean([row["length"] for row in rows])), 2),
            "minimum_context_utilization": round(
                min(float(row["length"]) / context_length for row in rows), 6
            ),
        }

    aggregate = {
        "task": canonical_task(task),
        "metric": "string_match_all",
        "score_by_context_length": by_length,
        "overall_score": (
            round(100 * float(np.mean([row["score"] for row in result_rows])), 2)
            if result_rows
            else None
        ),
        "num_samples": len(result_rows),
        "num_success": sum(row.get("status") == "success" for row in result_rows),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    prefix_summary_path = output_dir / "prefix_cache_summary.json"
    if prefix_summary_path.is_file():
        aggregate["prefix_cache"] = json.loads(
            prefix_summary_path.read_text(encoding="utf-8")
        )
    with (output_dir / "aggregate_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return aggregate


def merge_worker_outputs(
    output_dir: Path,
    world_size: int,
    *,
    prefix_cache_replay: bool,
) -> None:
    stems = ["raw_outputs", "parsed_outputs", "per_sample_results"]
    if prefix_cache_replay:
        stems.extend(
            [
                "raw_outputs_prefix_cache_replay",
                "parsed_outputs_prefix_cache_replay",
                "per_sample_results_prefix_cache_replay",
            ]
        )
    for stem in stems:
        rows: list[dict[str, Any]] = []
        for rank in range(world_size):
            path = output_dir / f"{stem}_rank{rank}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"Missing worker output: {path}")
            rows.extend(read_jsonl(path))
        rows.sort(key=lambda row: int(row["index"]))
        write_jsonl(output_dir / f"{stem}.jsonl", rows)


def write_prefix_cache_summary(output_dir: Path, world_size: int) -> dict[str, Any]:
    rank_records: list[dict[str, Any]] = []
    for rank in range(world_size):
        path = output_dir / f"prefix_cache_stats_rank{rank}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing RULER prefix-cache stats for rank={rank}: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid RULER prefix-cache stats for rank={rank}: {payload!r}")
        rank_records.append(payload)
    summary = {
        "status": "success",
        "world_size": int(world_size),
        "replay_hit_requests": sum(
            int(row.get("replay_hit_requests", 0)) for row in rank_records
        ),
        "replay_hit_tokens": sum(
            int(row.get("replay_hit_tokens", 0)) for row in rank_records
        ),
        "output_mismatch_indices": sorted(
            int(index)
            for row in rank_records
            for index in row.get("output_mismatch_indices", [])
        ),
        "ranks": rank_records,
    }
    with (output_dir / "prefix_cache_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary


def validate_run(
    args: argparse.Namespace,
    output_dir: Path,
    context_lengths: list[int],
) -> None:
    rows = read_jsonl(output_dir / "per_sample_results.jsonl")
    task = canonical_task(getattr(args, "task", "vt"))
    expected_count = len(context_lengths) * int(args.samples_per_length)
    if len(rows) != expected_count:
        raise RuntimeError(
            "RULER result count mismatch: "
            f"expected={expected_count} actual={len(rows)}."
        )
    expected_indices = list(range(expected_count))
    observed_indices = [int(row["index"]) for row in rows]
    if observed_indices != expected_indices:
        raise RuntimeError(
            "RULER sample identities are incomplete or reordered: "
            f"expected={expected_indices} actual={observed_indices}."
        )
    failed = [row for row in rows if row.get("status") != "success"]
    if failed:
        raise RuntimeError(
            f"RULER generation failed for {len(failed)}/{expected_count} samples."
        )
    wrong_task = [row for row in rows if canonical_task(row.get("task", "vt")) != task]
    if wrong_task:
        raise RuntimeError(
            f"RULER artifact contains {len(wrong_task)} rows for a task other than {task}."
        )
    for context_length in context_lengths:
        length_rows = [
            row for row in rows if int(row["context_length"]) == context_length
        ]
        if len(length_rows) != int(args.samples_per_length):
            raise RuntimeError(
                "RULER context-length coverage mismatch: "
                f"context_length={context_length} expected={args.samples_per_length} "
                f"actual={len(length_rows)}."
            )
        underfilled = [
            row
            for row in length_rows
            if float(row["length"]) / context_length
            < float(args.minimum_context_utilization)
        ]
        if underfilled:
            minimum_observed = min(
                float(row["length"]) / context_length for row in length_rows
            )
            raise RuntimeError(
                "RULER samples did not reach the configured target-length utilization: "
                f"context_length={context_length} minimum_observed={minimum_observed:.6f} "
                f"required={float(args.minimum_context_utilization):.6f}."
            )

    if not args.prefix_cache_replay:
        return
    replay_rows = read_jsonl(output_dir / "per_sample_results_prefix_cache_replay.jsonl")
    if len(replay_rows) != expected_count:
        raise RuntimeError(
            "RULER prefix-cache replay result count mismatch: "
            f"expected={expected_count} actual={len(replay_rows)}."
        )
    replay_failed = [row for row in replay_rows if row.get("status") != "success"]
    if replay_failed:
        raise RuntimeError(
            f"RULER prefix-cache replay failed for {len(replay_failed)}/{expected_count} samples."
        )
    summary = json.loads(
        (output_dir / "prefix_cache_summary.json").read_text(encoding="utf-8")
    )
    mismatches = summary.get("output_mismatch_indices") or []
    if mismatches:
        raise RuntimeError(
            "RULER prefix-cache replay changed deterministic outputs for sample indices "
            f"{mismatches}."
        )
    if args.require_prefix_cache_hit and (
        int(summary.get("replay_hit_requests", 0)) <= 0
        or int(summary.get("replay_hit_tokens", 0)) <= 0
    ):
        raise RuntimeError(
            "RULER prefix-cache replay observed no cache hit requests/tokens: "
            f"{summary}."
        )


def launch_workers(args: argparse.Namespace, output_dir: Path) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        gpu_ids = [gpu.strip() for gpu in visible.split(",") if gpu.strip()]
    else:
        gpu_ids = [str(i) for i in range(torch.cuda.device_count())]
    if len(gpu_ids) < args.ws:
        raise ValueError(f"Requested ws={args.ws}, but only {len(gpu_ids)} visible GPUs are available: {gpu_ids}")

    script_path = Path(__file__).resolve()
    child_base = sys.argv[1:]
    procs: list[subprocess.Popen] = []
    for rank in range(args.ws):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[rank]
        cmd = [
            sys.executable,
            "-u",
            str(script_path),
            *child_base,
            "--worker-rank",
            str(rank),
            "--worker-world-size",
            str(args.ws),
            "--cuda-device",
            "0",
        ]
        print(f"[Parent] launch rank={rank} gpu={gpu_ids[rank]} cmd={' '.join(cmd)}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env, cwd=str(script_path.parent.parent.parent)))

    failed: list[tuple[int, int]] = []
    for rank, proc in enumerate(procs):
        ret = proc.wait()
        if ret != 0:
            failed.append((rank, ret))
    if failed:
        raise RuntimeError("RULER worker failed: " + ", ".join(f"rank={r}, exitcode={c}" for r, c in failed))


def main() -> None:
    args = parse_args()
    random.seed(args.seed + max(args.worker_rank, 0))
    np.random.seed(args.seed + max(args.worker_rank, 0))
    torch.manual_seed(args.seed + max(args.worker_rank, 0))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context_lengths = parse_context_lengths(args.context_lengths)
    tokenizer_path = args.tokenizer_path or args.model_path

    if args.worker_rank < 0:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        generate_dataset(args, output_dir, tokenizer)
        infer_config = build_infer_config(args, context_lengths)
        write_run_info(args, output_dir, tokenizer_path, infer_config)

        if args.ws > 1:
            start = time.time()
            launch_workers(args, output_dir)
            merge_worker_outputs(
                output_dir,
                args.ws,
                prefix_cache_replay=bool(args.prefix_cache_replay),
            )
            if args.prefix_cache_replay:
                write_prefix_cache_summary(output_dir, args.ws)
            write_aggregate(
                output_dir,
                context_lengths,
                time.time() - start,
                task=args.task,
            )
            validate_run(args, output_dir, context_lengths)
            return

        samples = [sample_from_row(row) for row in read_jsonl(output_dir / "dataset.jsonl")]
        start = time.time()
        evaluate_samples(args, samples, output_dir, infer_config)
        if args.prefix_cache_replay:
            write_prefix_cache_summary(output_dir, 1)
        write_aggregate(
            output_dir,
            context_lengths,
            time.time() - start,
            task=args.task,
        )
        validate_run(args, output_dir, context_lengths)
        return

    samples_all = [sample_from_row(row) for row in read_jsonl(output_dir / "dataset.jsonl")]
    samples = samples_all[args.worker_rank :: args.worker_world_size]
    infer_config = build_infer_config(args, context_lengths)
    evaluate_samples(args, samples, output_dir, infer_config)


if __name__ == "__main__":
    main()
