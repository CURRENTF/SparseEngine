# SPDX-License-Identifier: Apache-2.0
"""LongBench prediction runner using upstream vLLM with strict coverage checks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.long_bench.prompt_budget import encode_prompt_with_generation_budget

DATA_PREFIX_PATH = (
    os.getenv("SPARSEENGINE_LONGBENCH_DATA_DIR")
    or os.getenv("SPARSEENGINE_DATA_DIR")
    or str(REPO_ROOT / "data" / "LongBench")
)
NO_CHAT_TEMPLATE_DATASETS = {
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "lcc",
    "repobench-p",
}


def get_longbench_data_path(dataset: str, use_longbench_e: bool = False) -> str:
    suffix = "_e" if use_longbench_e else ""
    return os.path.join(DATA_PREFIX_PATH, "data", f"{dataset}{suffix}.jsonl")


def build_chat_prompt(tokenizer: Any, formatted_prompt: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template") or not tokenizer.chat_template:
        raise ValueError("Tokenizer has no chat template; use a no-template LongBench task instead.")
    messages = [{"role": "user", "content": formatted_prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if prompt.endswith("<think>\n"):
        prompt += "</think>\n"
    return prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LongBench evaluation on upstream vLLM.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument(
        "--task",
        type=str,
        default="qasper,hotpotqa,multi_news,trec,passage_retrieval_en,lcc",
    )
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument(
        "--samples_per_task",
        type=int,
        default=-1,
        help="Deprecated compatibility alias; prefer --num_samples.",
    )
    parser.add_argument("--min_required_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_prefix_caching", action="store_true")
    parser.add_argument("--e", action="store_true")
    return parser.parse_args()


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_record(output_root: Path, task_path: Path, record: dict[str, Any]) -> None:
    raw_record = {
        key: record.get(key)
        for key in (
            "dataset",
            "sample_idx",
            "source_idx",
            "sample_id",
            "status",
            "prompt_tokens",
            "raw_pred",
            "error",
            "traceback",
        )
        if key in record
    }
    parsed_record = {
        key: record.get(key)
        for key in (
            "dataset",
            "sample_idx",
            "source_idx",
            "sample_id",
            "status",
            "prompt_tokens",
            "pred",
            "error",
        )
        if key in record
    }
    task_record = {
        "status": record["status"],
        "pred": record.get("pred", ""),
        "raw_pred": record.get("raw_pred", ""),
        "answers": record.get("answers"),
        "all_classes": record.get("all_classes"),
        "length": record.get("length"),
        "prompt_tokens": record.get("prompt_tokens"),
        "source_idx": record.get("source_idx"),
        "sample_id": record.get("sample_id"),
    }
    if "error" in record:
        task_record["error"] = record["error"]
    _append_jsonl(output_root / "raw_outputs.jsonl", raw_record)
    _append_jsonl(output_root / "parsed_outputs.jsonl", parsed_record)
    _append_jsonl(output_root / "sample_results.jsonl", record)
    _append_jsonl(task_path, task_record)


def _effective_num_samples(args: argparse.Namespace) -> int | None:
    if args.num_samples is not None:
        if args.num_samples <= 0:
            raise ValueError(f"--num_samples must be > 0 when set, got {args.num_samples}.")
        if args.samples_per_task > 0 and args.samples_per_task != args.num_samples:
            raise ValueError("--num_samples and --samples_per_task disagree.")
        return int(args.num_samples)
    return int(args.samples_per_task) if args.samples_per_task > 0 else None


def _write_run_status(output_root: Path, status: str, **details: Any) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_status.json").write_text(
        json.dumps({"status": status, **details}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.max_model_len <= 0:
        raise ValueError(f"--max_model_len must be > 0, got {args.max_model_len}.")
    if args.batch_size == 0:
        raise ValueError("--batch_size must be positive or negative for full-task batching.")
    num_samples = _effective_num_samples(args)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_run_status(output_root, "running")
    for artifact in ("raw_outputs.jsonl", "parsed_outputs.jsonl", "sample_results.jsonl"):
        (output_root / artifact).write_text("", encoding="utf-8")

    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
        enable_prefix_caching=args.enable_prefix_caching,
        trust_remote_code=True,
    )

    with (REPO_ROOT / "benchmark/long_bench/config/dataset2prompt.json").open(
        "r", encoding="utf-8"
    ) as handle:
        dataset2prompt = json.load(handle)
    with (REPO_ROOT / "benchmark/long_bench/config/dataset2maxlen.json").open(
        "r", encoding="utf-8"
    ) as handle:
        dataset2maxlen = json.load(handle)
    tasks = [task.strip() for task in args.task.split(",") if task.strip()]
    if not tasks:
        raise ValueError("--task must name at least one LongBench dataset.")

    resolved = {
        "timestamp": datetime.now().isoformat(),
        "command": [sys.executable, *sys.argv],
        "backend": "vllm",
        "model_path": args.model_path,
        "tokenizer_path": tokenizer_path,
        "datasets": tasks,
        "longbench_data_root": DATA_PREFIX_PATH,
        "num_samples": num_samples,
        "args": vars(args),
    }
    (output_root / "resolved_config.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    coverage: dict[str, Any] = {"status": "success", "tasks": {}}
    for dataset in tasks:
        if dataset not in dataset2prompt or dataset not in dataset2maxlen:
            raise ValueError(f"Unknown LongBench dataset: {dataset}")
        data_path = Path(get_longbench_data_path(dataset, args.e))
        if not data_path.is_file():
            raise FileNotFoundError(f"LongBench dataset file not found: {data_path}")
        with data_path.open("r", encoding="utf-8") as handle:
            all_lines = [json.loads(line) for line in handle if line.strip()]
        if num_samples is not None and len(all_lines) < num_samples:
            raise RuntimeError(
                f"LongBench dataset={dataset} has {len(all_lines)} rows, fewer than "
                f"requested num_samples={num_samples}."
            )
        lines = all_lines if num_samples is None else all_lines[:num_samples]
        if not lines:
            raise RuntimeError(f"LongBench dataset={dataset} contains no samples.")
        if args.min_required_samples > 0 and len(lines) < args.min_required_samples:
            raise RuntimeError(
                f"LongBench dataset={dataset} selected {len(lines)} rows, fewer than "
                f"min_required_samples={args.min_required_samples}."
            )

        task_path = output_root / f"{dataset}.jsonl"
        task_path.write_text("", encoding="utf-8")
        max_gen = int(dataset2maxlen[dataset])
        prepared: list[tuple[dict[str, Any], list[int], int]] = []
        for source_idx, item in enumerate(lines):
            base = {
                "dataset": dataset,
                "sample_idx": source_idx,
                "source_idx": source_idx,
                "sample_id": item.get("_id", ""),
                "answers": item.get("answers"),
                "all_classes": item.get("all_classes"),
                "length": item.get("length"),
            }
            try:
                if "answers" not in item or "all_classes" not in item:
                    raise ValueError("LongBench sample must contain answers and all_classes fields.")
                formatted = dataset2prompt[dataset].format(**item)
                prompt = (
                    formatted
                    if dataset in NO_CHAT_TEMPLATE_DATASETS
                    else build_chat_prompt(tokenizer, formatted)
                )
                prompt_token_ids = encode_prompt_with_generation_budget(
                    tokenizer,
                    prompt,
                    max_model_len=args.max_model_len,
                    max_gen=max_gen,
                )
                prompt_tokens = len(prompt_token_ids)
                prepared.append((base, prompt_token_ids, prompt_tokens))
            except Exception as exc:
                failed = {
                    **base,
                    "status": "invalid_input",
                    "pred": "",
                    "raw_pred": "",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                _write_record(output_root, task_path, failed)
                raise RuntimeError(
                    f"LongBench prompt preparation failed for dataset={dataset}, "
                    f"source_idx={source_idx}: {exc}"
                ) from exc

        batch_size = len(prepared) if args.batch_size < 0 else args.batch_size
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=max_gen,
        )
        completed_source_indices: list[int] = []
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            prompts = [{"prompt_token_ids": entry[1]} for entry in batch]
            try:
                outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
            except Exception as exc:
                for base, _prompt, prompt_tokens in batch:
                    _write_record(
                        output_root,
                        task_path,
                        {
                            **base,
                            "prompt_tokens": prompt_tokens,
                            "status": "model_failed",
                            "pred": "",
                            "raw_pred": "",
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                raise
            if len(outputs) != len(batch):
                error = (
                    f"vLLM returned {len(outputs)} outputs for {len(batch)} prompts "
                    f"in dataset={dataset}."
                )
                for base, _prompt, prompt_tokens in batch:
                    _write_record(
                        output_root,
                        task_path,
                        {
                            **base,
                            "prompt_tokens": prompt_tokens,
                            "status": "parse_failed",
                            "pred": "",
                            "raw_pred": "",
                            "error": error,
                        },
                    )
                raise RuntimeError(error)
            for (base, _prompt, prompt_tokens), output in zip(batch, outputs):
                try:
                    if not output.outputs:
                        raise RuntimeError("vLLM returned no completion.")
                    pred_text = output.outputs[0].text
                    if not isinstance(pred_text, str):
                        raise TypeError(
                            f"vLLM prediction must be str, got {type(pred_text).__name__}."
                        )
                except Exception as exc:
                    _write_record(
                        output_root,
                        task_path,
                        {
                            **base,
                            "prompt_tokens": prompt_tokens,
                            "status": "parse_failed",
                            "pred": "",
                            "raw_pred": "",
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    raise RuntimeError(
                        f"LongBench output parsing failed for dataset={dataset}, "
                        f"source_idx={base['source_idx']}: {exc}"
                    ) from exc
                _write_record(
                    output_root,
                    task_path,
                    {
                        **base,
                        "prompt_tokens": prompt_tokens,
                        "status": "success",
                        "pred": pred_text,
                        "raw_pred": pred_text,
                    },
                )
                completed_source_indices.append(int(base["source_idx"]))

        expected_source_indices = list(range(len(lines)))
        if completed_source_indices != expected_source_indices:
            raise RuntimeError(
                f"LongBench coverage mismatch for dataset={dataset}: "
                f"expected={expected_source_indices}, actual={completed_source_indices}."
            )
        coverage["tasks"][dataset] = {
            "status": "success",
            "expected_count": len(lines),
            "actual_count": len(completed_source_indices),
            "source_indices": completed_source_indices,
        }

    (output_root / "coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    eval_cmd = [sys.executable, str(REPO_ROOT / "benchmark/long_bench/eval.py"), "--path", str(output_root)]
    if args.e:
        eval_cmd.append("--e")
    subprocess.run(eval_cmd, check=True, cwd=REPO_ROOT)
    _write_run_status(output_root, "success")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            failed_args = parse_args()
            status = "model_failed"
            if isinstance(exc, subprocess.CalledProcessError) and any(
                str(part).endswith("benchmark/long_bench/eval.py")
                for part in (exc.cmd if isinstance(exc.cmd, (list, tuple)) else [exc.cmd])
            ):
                status = "metric_failed"
            elif isinstance(exc, (FileNotFoundError, TypeError, ValueError)):
                status = "invalid_input"
            _write_run_status(
                Path(failed_args.output_root),
                status,
                error=repr(exc),
                traceback=traceback.format_exc(),
            )
        except Exception:
            pass
        raise
