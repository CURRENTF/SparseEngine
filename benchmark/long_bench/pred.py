import os
import json
import sys
import subprocess
import re
import traceback
from typing import Any, Union
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.multiprocessing as mp
import torch
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
import torch.distributed as dist
from benchmark.model_adapters.sparseengine import get_sparseengine_generate_api
from benchmark.long_bench.prompt_budget import encode_prompt_with_generation_budget
from benchmark.sparseengine_regression.manifest import (
    validate_omnikv_benchmark_config,
)
from datetime import datetime

BASE_PATH = os.getenv("SPARSEENGINE_OUTPUT_DIR", str(REPO_ROOT / "outputs"))
DATA_PREFIX_PATH = os.getenv("SPARSEENGINE_LONGBENCH_DATA_DIR") or os.getenv("SPARSEENGINE_DATA_DIR")
DEFAULT_MAX_MODEL_LEN = 121_000
NO_CHAT_TEMPLATE_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
SAMPLE_STATUSES = {
    "success",
    "invalid_input",
    "model_failed",
    "parse_failed",
    "metric_failed",
    "skipped_by_policy",
}


def get_longbench_data_path(dataset, use_longbench_e):
    if not DATA_PREFIX_PATH:
        raise FileNotFoundError(
            "LongBench data root is not configured.\n"
            "Set SPARSEENGINE_LONGBENCH_DATA_DIR or SPARSEENGINE_DATA_DIR to the LongBench "
            "root directory that contains data/*.jsonl."
        )
    suffix = "_e" if use_longbench_e else ""
    return os.path.join(DATA_PREFIX_PATH, "data", f"{dataset}{suffix}.jsonl")


def validate_longbench_data_paths(datasets, use_longbench_e):
    if not DATA_PREFIX_PATH:
        raise FileNotFoundError(
            "LongBench data root is not configured.\n"
            "Set SPARSEENGINE_LONGBENCH_DATA_DIR or SPARSEENGINE_DATA_DIR to the LongBench "
            "root directory that contains data/*.jsonl."
        )
    if not os.path.isdir(DATA_PREFIX_PATH):
        raise FileNotFoundError(
            "LongBench data root does not exist: "
            f"{DATA_PREFIX_PATH}\n"
            "Set SPARSEENGINE_LONGBENCH_DATA_DIR or SPARSEENGINE_DATA_DIR to the LongBench root "
            "directory that contains data/*.jsonl."
        )

    data_dir = os.path.join(DATA_PREFIX_PATH, "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            "LongBench data directory does not exist: "
            f"{data_dir}\n"
            "Set SPARSEENGINE_LONGBENCH_DATA_DIR or SPARSEENGINE_DATA_DIR to the LongBench root "
            "directory that contains a data/ subdirectory."
        )

    missing_paths = [
        get_longbench_data_path(dataset, use_longbench_e)
        for dataset in datasets
        if not os.path.isfile(get_longbench_data_path(dataset, use_longbench_e))
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Missing LongBench dataset files:\n"
            + "\n".join(missing_paths)
            + "\nCheck SPARSEENGINE_LONGBENCH_DATA_DIR / SPARSEENGINE_DATA_DIR."
        )

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def should_use_chat_template(dataset, no_chat_template=False, thinking_mode="off"):
    return not no_chat_template and dataset not in NO_CHAT_TEMPLATE_DATASETS


def build_chat(tokenizer, prompt, dataset, no_chat_template=False, thinking_mode="off"):
    if not should_use_chat_template(dataset, no_chat_template, thinking_mode):
        return prompt
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        msgs = [
            # {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt},
        ]
        enable_thinking = thinking_mode != "off"
        prompt = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        # Some local Qwen3 tokenizer templates still end with an open `<think>` block
        # even when `enable_thinking=False`. Close it explicitly to force empty-thinking mode.
        if thinking_mode == "off" and prompt.endswith("<think>\n"):
            prompt += "</think>\n"
    if os.getenv('DEBUG'):
        print('input prompt:', prompt)
    return prompt


def strip_thinking_content(text: str) -> str:
    closing_tag = "</think>"
    if closing_tag not in text:
        raise ValueError(
            "Thinking output ended before </think>; increase max_new_tokens instead "
            "of scoring truncated reasoning."
        )
    return text.split(closing_tag, 1)[1].lstrip()


def _load_hyper_params(value: str | None) -> dict[str, Any]:
    if value is None:
        return {}
    if os.path.exists(value):
        with open(value, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    else:
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse --hyper_param {value!r}; it is neither an existing "
                f"JSON file nor a valid JSON object: {exc}"
            ) from exc
    if not isinstance(loaded, dict):
        raise ValueError(
            f"--hyper_param must contain a JSON object, got {type(loaded).__name__}."
        )
    return loaded


def _build_infer_config(args: argparse.Namespace) -> dict[str, Any]:
    extra_config = _load_hyper_params(args.hyper_param)
    configured_method = extra_config.get("sparse_method")
    if configured_method is not None and configured_method != args.sparse_method:
        raise ValueError(
            f"Conflicting sparse_method values: --sparse_method={args.sparse_method!r}, "
            f"--hyper_param sparse_method={configured_method!r}."
        )
    configured_max_model_len = extra_config.get("max_model_len")
    if (
        configured_max_model_len is not None
        and int(configured_max_model_len) != int(args.max_model_len)
    ):
        raise ValueError(
            "Conflicting max_model_len values: "
            f"--max_model_len={args.max_model_len}, "
            f"--hyper_param max_model_len={configured_max_model_len}."
        )
    prefix_caching = bool(extra_config.get("enable_prefix_caching", False))
    if prefix_caching and not bool(getattr(args, "allow_prefix_caching", False)):
        raise ValueError(
            "LongBench quality regression requires enable_prefix_caching=False "
            "unless --allow_prefix_caching is passed explicitly."
        )

    if args.sparse_method == "omnikv":
        validate_omnikv_benchmark_config(
            extra_config,
            allow_single_full_layer=args.allow_single_omnikv_full_layer,
        )

    infer_config = dict(extra_config)
    infer_config["max_model_len"] = int(args.max_model_len)
    infer_config["enable_prefix_caching"] = prefix_caching
    return infer_config


def _requested_runtime_config(
    args: argparse.Namespace,
    infer_config: dict[str, Any],
) -> dict[str, Any]:
    public = dict(infer_config)
    public["sparse_method"] = args.sparse_method
    if args.deltakv_checkpoint_path is not None:
        public["deltakv_checkpoint_path"] = args.deltakv_checkpoint_path
    return {"config": public}


def _record_effective_runtime_config(
    *,
    generate_fn,
    out_root: str,
) -> None:
    llm = getattr(generate_fn, "_sparseengine_llm", None)
    if llm is None:
        raise RuntimeError(
            "SparseVLLM LongBench generation did not expose _sparseengine_llm; "
            "cannot record the effective runtime config."
        )
    runtime_info = llm.worker_info(tags=["longbench-quality"])
    resolved_path = Path(out_root) / "resolved_config.json"
    resolved = (
        json.loads(resolved_path.read_text(encoding="utf-8"))
        if resolved_path.is_file()
        else {"backend": "sparseengine"}
    )
    resolved["effective_runtime"] = runtime_info
    temporary_path = resolved_path.with_name(
        f".{resolved_path.name}.{os.getpid()}.tmp"
    )
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(resolved, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_path, resolved_path)


def _append_jsonl(path: str | os.PathLike[str], record: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


def _write_json(path: str | os.PathLike[str], value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, destination)


def _artifact_paths(out_root: str) -> dict[str, str]:
    return {
        "raw": os.path.join(out_root, "raw_outputs.jsonl"),
        "parsed": os.path.join(out_root, "parsed_outputs.jsonl"),
        "sample": os.path.join(out_root, "sample_results.jsonl"),
    }


def _write_run_status(
    out_root: str,
    status: str,
    *,
    error: BaseException | None = None,
    traceback_text: str | None = None,
) -> None:
    record = {"status": status}
    if error is not None:
        record["error"] = repr(error)
        record["traceback"] = traceback_text

    _write_json(Path(out_root) / "run_status.json", record)


def _worker_output_root(out_root: str, rank: int) -> str:
    return os.path.join(out_root, f".worker_rank{int(rank)}")


def _read_worker_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Invalid worker JSONL {path}:{line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise RuntimeError(
                f"Worker JSONL {path}:{line_number} must contain an object."
            )
        rows.append(row)
    return rows


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")
    os.replace(temporary, path)


def _merge_worker_outputs(
    out_root: str,
    *,
    datasets: list[str],
    world_size: int,
) -> None:
    artifact_names = [
        *(f"{dataset}.jsonl" for dataset in datasets),
        "raw_outputs.jsonl",
        "parsed_outputs.jsonl",
        "sample_results.jsonl",
    ]
    for artifact_name in artifact_names:
        merged: list[dict[str, Any]] = []
        observed: set[tuple[object, ...]] = set()
        is_task = artifact_name not in {
            "raw_outputs.jsonl",
            "parsed_outputs.jsonl",
            "sample_results.jsonl",
        }
        for rank in range(int(world_size)):
            worker_path = Path(_worker_output_root(out_root, rank)) / artifact_name
            for row in _read_worker_jsonl(worker_path):
                source_idx = row.get("source_idx")
                dataset = artifact_name.removesuffix(".jsonl") if is_task else row.get("dataset")
                if not isinstance(source_idx, int) or not isinstance(dataset, str):
                    raise RuntimeError(
                        f"Worker artifact {worker_path} lacks dataset/source_idx identity: "
                        f"{row}"
                    )
                key = (dataset, source_idx)
                if key in observed:
                    raise RuntimeError(
                        f"Duplicate LongBench sample identity {key} while merging "
                        f"{artifact_name}."
                    )
                observed.add(key)
                merged.append(row)
        merged.sort(
            key=lambda row: (
                artifact_name.removesuffix(".jsonl")
                if is_task
                else str(row["dataset"]),
                int(row["source_idx"]),
                int(row.get("sample_idx", -1)),
            )
        )
        _write_jsonl_atomic(Path(out_root) / artifact_name, merged)


def _write_operator_runtime_stats(*, generate_fn, out_root: str, rank: int) -> None:
    llm = getattr(generate_fn, "_sparseengine_llm", None)
    if llm is None:
        raise RuntimeError(
            "SparseVLLM LongBench generation did not expose _sparseengine_llm; "
            "cannot record operator runtime stats."
        )
    _write_json(
        Path(out_root) / f"operator_runtime_stats_rank{rank}.json",
        {
            "status": "success",
            "launcher_rank": int(rank),
            "world_ranks": llm.operator_runtime_stats(),
        },
    )


def _write_worker_load_stats(*, generate_fn, out_root: str, rank: int) -> None:
    llm = getattr(generate_fn, "_sparseengine_llm", None)
    if llm is None:
        raise RuntimeError(
            "SparseVLLM LongBench generation did not expose _sparseengine_llm; "
            "cannot record final cache statistics."
        )
    _write_json(
        Path(out_root) / f"worker_load_stats_rank{rank}.json",
        {
            "status": "success",
            "launcher_rank": int(rank),
            "worker_load": llm.worker_load(),
            "sparse_state": llm.debug_sparse_state_summaries(synchronize=True),
        },
    )


def _merge_operator_runtime_stats(out_root: str, *, world_size: int) -> None:
    world_ranks: list[dict[str, Any]] = []
    for rank in range(int(world_size)):
        path = Path(out_root) / f"operator_runtime_stats_rank{rank}.json"
        if not path.is_file():
            raise RuntimeError(f"Missing operator runtime stats for rank={rank}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "success" or not isinstance(
            payload.get("world_ranks"), list
        ):
            raise RuntimeError(f"Invalid operator runtime stats for rank={rank}: {payload}")
        for item in payload["world_ranks"]:
            if not isinstance(item, dict):
                raise RuntimeError(
                    f"Invalid operator runtime rank record for launcher rank={rank}: {item!r}"
                )
            world_ranks.append({"launcher_rank": rank, **item})
    _write_json(
        Path(out_root) / "operator_runtime_stats.json",
        {"status": "success", "world_ranks": world_ranks},
    )


def _try_write_failed_run_status(
    out_root: str | None,
    status: str,
    error: BaseException,
    traceback_text: str,
) -> None:
    if out_root is None:
        return
    try:
        os.makedirs(out_root, exist_ok=True)
        _write_run_status(
            out_root,
            status,
            error=error,
            traceback_text=traceback_text,
        )
    except Exception as status_error:
        print(
            f"[Warning] Failed to write run_status.json while handling "
            f"{error!r}: {status_error!r}",
            file=sys.stderr,
        )


def _decode_cuda_graph_status(
    *,
    generate_fn,
    rank: int,
) -> dict[str, Any]:
    llm = getattr(generate_fn, "_sparseengine_llm", None)
    if llm is None:
        raise RuntimeError(
            "SparseVLLM LongBench generation did not expose _sparseengine_llm; "
            "cannot verify decode CUDA graph execution."
        )

    runner = getattr(llm, "model_runner", None)
    graph_runner = getattr(runner, "decode_graph_runner", None)
    graph_states = (
        getattr(graph_runner, "_graphs", {})
        if graph_runner is not None
        else {}
    )
    graph_count = sum(
        getattr(state, "graph", None) is not None
        for state in graph_states.values()
    )
    graph_status = {
        "rank": int(rank),
        "configured": bool(
            getattr(getattr(llm, "config", None), "decode_graph", False)
        ),
        "runner_initialized": graph_runner is not None,
        "state_count": int(len(graph_states)),
        "graph_count": int(graph_count),
        "active": bool(graph_count > 0),
        "capture_count": int(getattr(graph_runner, "capture_count", 0)),
        "replay_count": int(getattr(graph_runner, "replay_count", 0)),
        "eager_static_count": int(getattr(graph_runner, "eager_static_count", 0)),
        "force_eager_count": int(getattr(graph_runner, "force_eager_count", 0)),
        "last_state_key": str(getattr(graph_runner, "last_state_key", None)),
        "state_keys": [str(key) for key in graph_states],
    }
    return graph_status


def _write_decode_cuda_graph_status(
    *,
    generate_fn,
    out_root: str,
    rank: int,
    before: dict[str, Any] | None = None,
) -> dict[str, Any]:
    graph_status = _decode_cuda_graph_status(generate_fn=generate_fn, rank=rank)
    if before is not None:
        counter_keys = (
            "capture_count",
            "replay_count",
            "eager_static_count",
            "force_eager_count",
        )
        graph_status["before"] = before
        graph_status["counter_delta"] = {
            key: int(graph_status[key]) - int(before[key]) for key in counter_keys
        }
    status_path = os.path.join(
        out_root,
        f"decode_graph_status_rank{rank}.json",
    )
    with open(status_path, "w", encoding="utf-8") as handle:
        json.dump(graph_status, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return graph_status


def _sample_base_record(
    *,
    dataset: str,
    batch_offset: int,
    json_obj: dict[str, Any],
    prompt_tokens: int | None = None,
) -> dict[str, Any]:
    source_idx = json_obj.get("_longbench_source_idx")
    if source_idx is None:
        source_idx = json_obj.get("_source_idx", batch_offset)
    return {
        "dataset": dataset,
        "sample_idx": int(batch_offset),
        "source_idx": int(source_idx),
        "prompt_tokens": None if prompt_tokens is None else int(prompt_tokens),
        "answers": json_obj.get("answers"),
        "all_classes": json_obj.get("all_classes"),
        "length": json_obj.get("length"),
    }


def _write_sample_record(
    *,
    out_root: str,
    task_out_path: str,
    record: dict[str, Any],
) -> None:
    status = record.get("status")
    if status not in SAMPLE_STATUSES:
        raise ValueError(f"Invalid sample status {status!r}; expected one of {sorted(SAMPLE_STATUSES)}.")

    paths = _artifact_paths(out_root)
    raw_record = {
        key: record.get(key)
        for key in (
            "dataset",
            "sample_idx",
            "source_idx",
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
            "status",
            "prompt_tokens",
            "pred",
            "error",
        )
        if key in record
    }
    _append_jsonl(paths["raw"], raw_record)
    _append_jsonl(paths["parsed"], parsed_record)
    _append_jsonl(paths["sample"], record)

    # Keep the historical per-task files for benchmark/long_bench/eval.py.
    task_record = {
        "status": record["status"],
        "pred": record.get("pred", ""),
        "raw_pred": record.get("raw_pred", record.get("pred", "")),
        "answers": record.get("answers"),
        "all_classes": record.get("all_classes"),
        "length": record.get("length"),
        "prompt_tokens": record.get("prompt_tokens"),
        "source_idx": record.get("source_idx"),
    }
    if "error" in record:
        task_record["error"] = record["error"]
    _append_jsonl(task_out_path, task_record)


def load_model_and_tokenizer(rank, args, infer_config):

    generate_fn = get_sparseengine_generate_api(
        model_path=args.model_path,
        infer_config=infer_config,
        deltakv_checkpoint_path=args.deltakv_checkpoint_path,
        sparse_method=args.sparse_method,
    )

    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    if os.path.isdir(args.model_path) and not os.path.exists(os.path.join(args.model_path, "generation_config.json")):
        # HF generation metadata is optional; malformed existing files still fail.
        generation_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    else:
        generation_config = GenerationConfig.from_pretrained(args.model_path, trust_remote_code=True)
    eos_token_ids = generation_config.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = []
    elif isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    else:
        eos_token_ids = list(eos_token_ids)
    if tokenizer.eos_token_id is not None:
        eos_token_ids.append(int(tokenizer.eos_token_id))
    if getattr(tokenizer, "eot_token_id", None) is not None:
        eos_token_ids.append(int(tokenizer.eot_token_id))
    eos_token_ids = list(dict.fromkeys(int(token_id) for token_id in eos_token_ids))

    return generate_fn, tokenizer, int(args.max_model_len), eos_token_ids


def get_pred(rank, data, dataset_info, args, model, tokenizer, model_max_length, eos_token_ids):
    dataset = dataset_info['dataset']
    prompt_format = dataset_info['prompt_format']
    max_gen = args.max_new_tokens_override if args.max_new_tokens_override is not None else dataset_info['max_gen']
    runtime_max_model_len = int(model_max_length)
    out_path = dataset_info['out_path']
    out_root = dataset_info['out_root']

    batch_size = len(data) if args.batch_size <= 0 else args.batch_size
    failures: list[dict[str, Any]] = []
    for i in tqdm(range(0, len(data), batch_size), desc=f'[Rank {rank}] {dataset}'):
        batch_data = data[i:i + batch_size]
        prompts = []
        prepared_records: list[dict[str, Any]] = []
        for json_obj in batch_data:
            selected_idx = int(json_obj.get("_longbench_selected_idx", i + len(prepared_records)))
            prompt_tokens = None
            try:
                if "answers" not in json_obj or "all_classes" not in json_obj:
                    raise ValueError("LongBench sample must contain answers and all_classes fields.")

                prompt = prompt_format.format(**json_obj)
                prompt = build_chat(tokenizer, prompt, dataset, args.no_chat_template, args.thinking_mode)
                prompt_token_ids = encode_prompt_with_generation_budget(
                    tokenizer,
                    prompt,
                    max_model_len=runtime_max_model_len,
                    max_gen=max_gen,
                )
                prompt_tokens = len(prompt_token_ids)
                prompts.append(prompt_token_ids)
                prepared_records.append(
                    _sample_base_record(
                        dataset=dataset,
                        batch_offset=selected_idx,
                        json_obj=json_obj,
                        prompt_tokens=prompt_tokens,
                    )
                )
            except Exception as exc:
                record = _sample_base_record(
                    dataset=dataset,
                    batch_offset=selected_idx,
                    json_obj=json_obj,
                    prompt_tokens=prompt_tokens,
                )
                record.update(
                    {
                        "status": "invalid_input",
                        "pred": "",
                        "raw_pred": "",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                _write_sample_record(out_root=out_root, task_out_path=out_path, record=record)
                failures.append(record)

        if failures:
            break

        try:
            preds = model(
                prompts,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                eos_token_id=eos_token_ids,
            )
        except Exception as exc:
            for record in prepared_records:
                failed = dict(record)
                failed.update(
                    {
                        "status": "model_failed",
                        "pred": "",
                        "raw_pred": "",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                _write_sample_record(out_root=out_root, task_out_path=out_path, record=failed)
                failures.append(failed)
            break

        if isinstance(preds, str): preds = [preds]
        if len(preds) != len(prepared_records):
            error = (
                f"Model returned {len(preds)} predictions for "
                f"{len(prepared_records)} prompts in dataset={dataset}."
            )
            for record in prepared_records:
                failed = dict(record)
                failed.update(
                    {
                        "status": "parse_failed",
                        "pred": "",
                        "raw_pred": "",
                        "error": error,
                    }
                )
                _write_sample_record(out_root=out_root, task_out_path=out_path, record=failed)
                failures.append(failed)
            break

        for record, pred in zip(prepared_records, preds):
            raw_pred = pred
            try:
                if not isinstance(raw_pred, str):
                    raise TypeError(f"Model prediction must be str, got {type(raw_pred).__name__}.")
                should_strip_thinking = (
                    args.thinking_mode == "on_strip"
                    and not args.no_chat_template
                    and dataset not in NO_CHAT_TEMPLATE_DATASETS
                )
                parsed_pred = strip_thinking_content(raw_pred) if should_strip_thinking else raw_pred
            except Exception as exc:
                failed = dict(record)
                failed.update(
                    {
                        "status": "parse_failed",
                        "pred": "",
                        "raw_pred": raw_pred if isinstance(raw_pred, str) else repr(raw_pred),
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                _write_sample_record(out_root=out_root, task_out_path=out_path, record=failed)
                failures.append(failed)
                continue

            ok = dict(record)
            ok.update(
                {
                    "status": "success",
                    "pred": parsed_pred,
                    "raw_pred": raw_pred,
                }
            )
            _write_sample_record(out_root=out_root, task_out_path=out_path, record=ok)

        if failures:
            break

    if failures:
        first = failures[0]
        raise RuntimeError(
            f"LongBench prediction failed for dataset={dataset}, rank={rank}, "
            f"status={first.get('status')}, source_idx={first.get('source_idx')}: {first.get('error')}"
        )


def worker(
    rank,
    world_size,
    datasets,
    dataset2prompt,
    dataset2maxlen,
    args,
    out_root,
    max_length_limit,
    infer_config,
):
    worker_out_root = (
        _worker_output_root(out_root, rank) if world_size > 1 else out_root
    )
    os.makedirs(worker_out_root, exist_ok=True)
    if world_size > 1:
        for dataset in datasets:
            Path(worker_out_root, f"{dataset}.jsonl").write_text(
                "", encoding="utf-8"
            )
        for artifact in _artifact_paths(worker_out_root).values():
            Path(artifact).write_text("", encoding="utf-8")
    seed_everything(args.seed)
    model, tokenizer, model_max_length, eos_token_ids = load_model_and_tokenizer(
        rank,
        args,
        infer_config,
    )
    if rank == 0:
        _record_effective_runtime_config(generate_fn=model, out_root=out_root)
    graph_status_before = _decode_cuda_graph_status(generate_fn=model, rank=rank)
    
    for dataset in datasets:
        data_path = get_longbench_data_path(dataset, args.e)
        if not os.path.isfile(data_path):
            raise FileNotFoundError(
                f"LongBench dataset file not found for dataset '{dataset}': {data_path}"
            )
        
        with open(data_path, "r", encoding="utf-8") as handle:
            data = [json.loads(line) for line in handle if line.strip()]
        for source_idx, row in enumerate(data):
            row.setdefault("_source_idx", source_idx)
        if args.num_samples is not None:
            if len(data) < args.num_samples:
                raise RuntimeError(
                    f"LongBench dataset={dataset} has {len(data)} rows, fewer than "
                    f"--num_samples={args.num_samples}."
                )
            data = data[:args.num_samples]

        if args.min_prompt_tokens is not None:
            from benchmark.sparseengine_regression.longbench_mini import select_longbench_mini_samples

            selected, selection_meta = select_longbench_mini_samples(
                data=data,
                tokenizer=tokenizer,
                dataset=dataset,
                prompt_format=dataset2prompt[dataset],
                min_prompt_tokens=int(args.min_prompt_tokens),
                samples_per_task=int(args.samples_per_task),
                min_required_samples=int(args.min_required_samples),
                no_chat_template=bool(args.no_chat_template),
                thinking_mode=args.thinking_mode,
            )
            if rank == 0:
                _append_jsonl(os.path.join(out_root, "longbench_mini_selection.jsonl"), selection_meta)
            if selection_meta["status"] == "skipped_by_policy":
                if rank == 0:
                    skipped = {
                        "dataset": dataset,
                        "sample_idx": -1,
                        "source_idx": -1,
                        "prompt_tokens": None,
                        "answers": None,
                        "all_classes": None,
                        "length": None,
                        "status": "skipped_by_policy",
                        "pred": "",
                        "raw_pred": "",
                        "error": (
                            f"Only {selection_meta['selected_rows']} samples reached "
                            f"min_prompt_tokens={selection_meta['min_prompt_tokens']}; "
                            f"min_required_samples={selection_meta['min_required_samples']}."
                        ),
                        "selection": selection_meta,
                    }
                    _write_sample_record(
                        out_root=worker_out_root,
                        task_out_path=os.path.join(worker_out_root, f"{dataset}.jsonl"),
                        record=skipped,
                    )
                continue

            selected_data: list[dict[str, Any]] = []
            for selected_idx, item in enumerate(selected):
                row = dict(item.row)
                row["_longbench_source_idx"] = int(item.source_idx)
                row["_longbench_selected_idx"] = int(selected_idx)
                row["_longbench_prompt_tokens"] = int(item.prompt_tokens)
                selected_data.append(row)
            data = selected_data
        
        data_subset = data[rank::world_size]
        if not data_subset: continue
        
        dataset_info = {
            'dataset': dataset,
            'prompt_format': dataset2prompt[dataset],
            'max_gen': dataset2maxlen[dataset],
            'max_length': max_length_limit,
            'out_path': os.path.join(worker_out_root, f"{dataset}.jsonl"),
            'out_root': worker_out_root,
        }
        
        get_pred(
            rank,
            data_subset,
            dataset_info,
            args,
            model,
            tokenizer,
            model_max_length,
            eos_token_ids,
        )
        torch.cuda.empty_cache()

    try:
        from sparseengine.utils.profiler import profiler
        snap = profiler.snapshot()
        if snap:
            with open(os.path.join(out_root, f"profiler_snapshot_rank{rank}.json"), "w", encoding="utf-8") as f:
                json.dump(snap, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"[Warning] Failed to dump profiler snapshot: {exc}")

    _write_decode_cuda_graph_status(
        generate_fn=model,
        out_root=out_root,
        rank=rank,
        before=graph_status_before,
    )
    _write_operator_runtime_stats(
        generate_fn=model,
        out_root=out_root,
        rank=rank,
    )
    _write_worker_load_stats(
        generate_fn=model,
        out_root=out_root,
        rank=rank,
    )


def launch_single_gpu_workers(args, out_root):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        gpu_ids = [gpu.strip() for gpu in visible.split(",") if gpu.strip()]
    else:
        gpu_ids = [str(i) for i in range(torch.cuda.device_count())]

    if len(gpu_ids) < args.ws:
        raise ValueError(
            f"Requested ws={args.ws}, but only {len(gpu_ids)} visible GPUs are available: {gpu_ids}"
        )

    base_master_port = int(os.environ.get("SPARSEENGINE_MASTER_PORT", "2333"))
    if base_master_port <= 0 or base_master_port + args.ws - 1 > 65535:
        raise ValueError(
            "LongBench worker master-port range is invalid: "
            f"base={base_master_port} ws={args.ws}."
        )

    script_path = Path(__file__).resolve()
    child_argv = sys.argv[1:]
    procs = []
    for rank in range(args.ws):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[rank]
        env["SPARSEENGINE_MASTER_PORT"] = str(base_master_port + rank)
        cmd = [
            sys.executable,
            "-u",
            str(script_path),
            *child_argv,
            "--worker_rank",
            str(rank),
            "--worker_world_size",
            str(args.ws),
            "--output_root",
            out_root,
        ]
        print(f"[Parent] launch rank={rank} gpu={gpu_ids[rank]} cmd={' '.join(cmd)}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env, cwd=str(script_path.parent.parent.parent)))

    failed_ranks = []
    for rank, proc in enumerate(procs):
        ret = proc.wait()
        if ret != 0:
            failed_ranks.append((rank, ret))
    if failed_ranks:
        raise RuntimeError(
            "LongBench worker failed; aborting evaluation. "
            + ", ".join(f"rank={rank}, exitcode={ret}" for rank, ret in failed_ranks)
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="my_model")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("--ws", default=1, type=int, help='world size')
    parser.add_argument("--task_start_id", default=0, type=int)
    parser.add_argument("--task", default=None, type=str)

    # DeltaKV related arguments
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--deltakv_checkpoint_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--sparse_method", type=str, default='deltakv')
    parser.add_argument("--num_samples", type=int, default=None, help="Limit the number of samples to process per task")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--no_chat_template", action='store_true', help="Do not use chat template")
    parser.add_argument("--hyper_param", type=str, default=None, help="Path to a JSON file or a JSON string containing hyper-parameters")
    parser.add_argument(
        "--allow_single_omnikv_full_layer",
        action="store_true",
        help="Allow an explicit single-full-layer OmniKV ablation.",
    )
    parser.add_argument(
        "--allow_prefix_caching",
        action="store_true",
        help=(
            "Explicitly opt into prefix caching for a targeted LongBench A/B. "
            "The default quality protocol keeps prefix caching disabled."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinking_mode", type=str, default="off", choices=["off", "on_strip"])
    parser.add_argument("--max_new_tokens_override", type=int, default=None)
    parser.add_argument("--min_prompt_tokens", type=int, default=None)
    parser.add_argument("--samples_per_task", type=int, default=20)
    parser.add_argument("--min_required_samples", type=int, default=5)
    parser.add_argument("--worker_rank", type=int, default=-1)
    parser.add_argument("--worker_world_size", type=int, default=1)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Runtime context limit (default: 121000).",
    )

    return parser.parse_args()


def main() -> None:
    args = None
    out_root = None
    phase = "invalid_input"
    try:
        args = parse_args()

        model_name = args.model
        compressor_name = (
            os.path.basename(args.deltakv_checkpoint_path.rstrip('/'))
            if args.deltakv_checkpoint_path
            else "None"
        )
        if args.output_root:
            out_root = args.output_root
        else:
            time_tag = datetime.now().strftime("%m%d_%H%M")
            out_root = os.path.join(
                BASE_PATH,
                f"benchmark/long_bench/{'pred_e' if args.e else 'pred'}/"
                f"{model_name}/{compressor_name}_{time_tag}",
            )
        os.makedirs(out_root, exist_ok=True)
        if args.worker_rank < 0:
            _write_run_status(out_root, "running")
        print(f"Results will be saved in: {out_root}")

        if args.num_samples is not None and args.num_samples <= 0:
            raise ValueError(
                f"--num_samples must be > 0 when set, got {args.num_samples}."
            )
        mp.set_start_method('spawn', force=True)

        if args.e:
            datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
        else:
            # en + zh
            # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
            # en
            datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", "passage_count",
                        "passage_retrieval_en", "lcc", "repobench-p"]

        datasets = datasets[args.task_start_id:]
        if args.task:
            datasets = args.task.split(',')

        with open("benchmark/long_bench/config/dataset2prompt.json", "r") as f:
            dataset2prompt = json.load(f)
        with open("benchmark/long_bench/config/dataset2maxlen.json", "r") as f:
            dataset2maxlen = json.load(f)
        validate_longbench_data_paths(datasets, args.e)

        if args.worker_rank < 0:
            for dataset in datasets:
                with open(os.path.join(out_root, f"{dataset}.jsonl"), 'w') as f:
                    pass
            for artifact in ("raw_outputs.jsonl", "parsed_outputs.jsonl", "sample_results.jsonl", "longbench_mini_selection.jsonl"):
                with open(os.path.join(out_root, artifact), "w", encoding="utf-8") as f:
                    pass

        max_length_limit = DEFAULT_MAX_MODEL_LEN if args.max_model_len is None else args.max_model_len
        if max_length_limit <= 0:
            raise ValueError(f"--max_model_len must be > 0, got {max_length_limit}.")
        args.max_model_len = max_length_limit
        infer_config = _build_infer_config(args)

        if args.worker_rank < 0:
            resolved_config = {
                "model": args.model,
                "model_path": args.model_path,
                "tokenizer_path": args.tokenizer_path or args.model_path,
                "backend": "sparseengine",
                "sparse_method": args.sparse_method,
                "deltakv_checkpoint_path": args.deltakv_checkpoint_path,
                "datasets": datasets,
                "longbench_data_root": DATA_PREFIX_PATH,
                "max_model_len": args.max_model_len,
                "decoding": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "max_new_tokens_override": args.max_new_tokens_override,
                },
                "selection": {
                    "min_prompt_tokens": args.min_prompt_tokens,
                    "samples_per_task": args.samples_per_task,
                    "min_required_samples": args.min_required_samples,
                },
                "requested_runtime": _requested_runtime_config(args, infer_config),
                "args": vars(args),
            }
            with open(os.path.join(out_root, "resolved_config.json"), "w", encoding="utf-8") as f:
                json.dump(resolved_config, f, ensure_ascii=False, indent=2)
                f.write("\n")

        phase = "model"
        if args.worker_rank >= 0:
            worker(
                args.worker_rank,
                args.worker_world_size,
                datasets,
                dataset2prompt,
                dataset2maxlen,
                args,
                out_root,
                max_length_limit,
                infer_config,
            )
        elif args.ws > 1:
            launch_single_gpu_workers(args, out_root)
            _merge_worker_outputs(
                out_root,
                datasets=datasets,
                world_size=args.ws,
            )
        else:
            worker(
                0,
                1,
                datasets,
                dataset2prompt,
                dataset2maxlen,
                args,
                out_root,
                max_length_limit,
                infer_config,
            )

        if args.worker_rank < 0:
            _merge_operator_runtime_stats(out_root, world_size=args.ws)
            phase = "metric"
            # 记录评测信息到日志文件
            log_path = os.path.join(out_root, "longbench_eval.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Command: python {' '.join(sys.argv)}\n")
                f.write(f"Output Root: {out_root}\n")
                f.write(f"Args: {json.dumps(vars(args), indent=2)}\n")
                f.write("-" * 80 + "\n")

            # 自动运行评测并记录日志
            print(f"正在对 {out_root} 进行自动评测...")
            eval_cmd = [
                sys.executable,
                "benchmark/long_bench/eval.py",
                "--path", out_root
            ]
            if args.e:
                eval_cmd.append("--e")

            subprocess.run(eval_cmd, check=True)

            # 读取评测结果并写入日志
            result_path = os.path.join(out_root, "result.json")
            if not os.path.exists(result_path):
                raise FileNotFoundError(
                    f"LongBench evaluation did not write result.json: {result_path}"
                )
            with open(result_path, "r", encoding="utf-8") as f:
                scores = json.load(f)

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"Evaluation Results ({'LongBench-E' if args.e else 'LongBench'}):\n")
                f.write(json.dumps(scores, indent=4, ensure_ascii=False))
                f.write("\n" + "="*80 + "\n\n")
            print(f"评测结果已成功写入日志: {log_path}")
            _write_run_status(out_root, "success")
    except subprocess.CalledProcessError as error:
        failure_status = "metric_failed" if phase == "metric" else (
            "model_failed" if phase == "model" else "invalid_input"
        )
        if args is None or args.worker_rank < 0:
            _try_write_failed_run_status(
                out_root or (args.output_root if args is not None else None),
                failure_status,
                error,
                traceback.format_exc(),
            )
        raise
    except Exception as error:
        failure_status = "model_failed" if phase == "model" else (
            "metric_failed" if phase == "metric" else "invalid_input"
        )
        if args is None or args.worker_rank < 0:
            _try_write_failed_run_status(
                out_root or (args.output_root if args is not None else None),
                failure_status,
                error,
                traceback.format_exc(),
            )
        raise


if __name__ == '__main__':
    main()
