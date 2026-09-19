"""Run a deterministic LongBench v2 subset on the native Sparse-Engine runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from benchmark.long_bench_v2.preparation import (
    prepare_chat as tokenize_chat,
    prepare_official_samples_parallel,
)
from benchmark.long_bench_v2.contracts import (
    aggregate_results,
    extract_answer,
    file_sha256,
    load_dataset,
    parse_token_buckets,
    prepare_official_samples,
    render_prompt,
    select_samples,
)
from benchmark.model_adapters.sparseengine import get_sparseengine_generate_api
from benchmark.sparseengine_regression.manifest import validate_omnikv_benchmark_config


DEFAULT_DATA_ENV = "SPARSEENGINE_LONGBENCH_V2_DATA"
DEFAULT_PROMPT_PATH = Path(__file__).with_name("upstream") / "prompts" / "0shot.txt"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")
    os.replace(temporary, path)


def _load_json_object(value: str | None) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as exc:
        if value.lstrip().startswith(("{", "[")):
            raise ValueError(
                "Invalid inline JSON configuration."
            ) from exc
        loaded = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("JSON configuration must decode to an object.")
    return loaded


def _build_infer_config(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_json_object(args.hyper_param_json)
    configured_method = config.get("sparse_method")
    if configured_method is not None and configured_method != args.sparse_method:
        raise ValueError(
            f"conflicting sparse_method values: CLI={args.sparse_method!r} "
            f"config={configured_method!r}."
        )
    configured_max_len = config.get("max_model_len")
    if configured_max_len is not None and int(configured_max_len) != args.max_model_len:
        raise ValueError(
            f"conflicting max_model_len values: CLI={args.max_model_len} "
            f"config={configured_max_len}."
        )
    if bool(config.get("enable_prefix_caching", False)) and not args.allow_prefix_caching:
        raise ValueError(
            "LongBench v2 quality regression requires prefix caching disabled unless "
            "--allow-prefix-caching is explicitly passed."
        )
    config["sparse_method"] = args.sparse_method
    config["max_model_len"] = args.max_model_len
    config.setdefault("enable_prefix_caching", False)
    if args.sparse_method == "omnikv":
        validate_omnikv_benchmark_config(config)
    return config


def _submodule_commit(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"LongBench v2 upstream is not an initialized git submodule: {path}. "
            "Run git submodule update --init benchmark/long_bench_v2/upstream."
        ) from exc


def _eos_token_ids(model_path: str, tokenizer: Any) -> list[int]:
    if os.path.isdir(model_path) and not os.path.exists(os.path.join(model_path, "generation_config.json")):
        generation_config = GenerationConfig.from_model_config(
            AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        )
    else:
        generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    configured = generation_config.eos_token_id
    if configured is None:
        values: list[int] = []
    elif isinstance(configured, int):
        values = [configured]
    else:
        values = [int(value) for value in configured]
    for value in (
        tokenizer.eos_token_id,
        getattr(tokenizer, "eot_token_id", None),
    ):
        if value is not None:
            values.append(int(value))
    return list(dict.fromkeys(values))


def _identity(item: dict[str, Any]) -> dict[str, Any]:
    sample = item["sample"]
    return {
        "index": int(item["index"]),
        "source_index": int(item["source_index"]),
        "_id": sample["_id"],
        "domain": sample["domain"],
        "sub_domain": sample["sub_domain"],
        "difficulty": sample["difficulty"],
        "official_length": sample["length"],
        "token_bucket": item["token_bucket"],
        "prompt_tokens": int(item["prompt_tokens"]),
        "original_prompt_tokens": int(item.get("original_prompt_tokens", item["prompt_tokens"])),
        "truncated": bool(item.get("truncated", False)),
        **({"original_prompt_sha256": item["original_prompt_sha256"]}
           if "original_prompt_sha256" in item else {}),
        "prompt_token_ids_sha256": hashlib.sha256(
            json.dumps(item["prompt_token_ids"], separators=(",", ":")).encode()
        ).hexdigest(),
        "prompt_sha256": hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest(),
        "context_sha256": hashlib.sha256(
            sample["context"].encode("utf-8")
        ).hexdigest(),
        "answer": sample["answer"],
    }


def _record_failure(
    item: dict[str, Any], status: str, error: str, *, raw_response: str = ""
) -> dict[str, Any]:
    return {
        **_identity(item),
        "status": status,
        "raw_response": raw_response,
        "predicted_answer": None,
        "correct": False,
        "error": error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LongBench v2 on all samples or a deterministic token-stratified subset."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--deltakv-checkpoint-path", default=None)
    parser.add_argument("--sparse-method", default="vanilla")
    parser.add_argument("--engine", choices=("sparseengine", "vllm", "sglang-http"), default="sparseengine")
    parser.add_argument("--engine-kwargs", default=None)
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--official-length", choices=("short", "medium", "long"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepared-samples", default=None)
    parser.add_argument("--preprocess-workers", type=int, default=8,
                        help="CPU processes for full-dataset preparation (default: 8); ignored for subsets.")
    parser.add_argument("--hyper-param-json", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-path", default=os.getenv(DEFAULT_DATA_ENV))
    parser.add_argument("--prompt-template", default=str(DEFAULT_PROMPT_PATH))
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--allow-prefix-caching", action="store_true")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all-samples", action="store_true",
                           help="Evaluate every source sample without token-bucket sampling.")
    parser.add_argument("--overflow-policy", choices=("error", "official-middle"),
                        default="error", help="Full-dataset truncation policy.")
    parser.add_argument("--truncate-max-tokens", type=int, default=None,
                        help="Pre-chat token limit for official-middle (e.g. upstream's 120000).")
    selection.add_argument(
        "--token-buckets-json",
        help="JSON list of name/min_prompt_tokens/max_prompt_tokens/samples objects.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "run_status.json", {"status": "running"})
    phase = "input"
    try:
        if args.preprocess_workers <= 0:
            raise ValueError("--preprocess-workers must be positive.")
        if not args.all_samples:
            args.preprocess_workers = 1
        if args.all_samples and args.official_length:
            raise ValueError("--all-samples cannot be combined with --official-length.")
        if args.all_samples and args.overflow_policy != "official-middle":
            raise ValueError("--all-samples requires --overflow-policy official-middle.")
        if not args.all_samples and args.overflow_policy != "error":
            raise ValueError("Truncating overflow policies require --all-samples.")
        if args.overflow_policy == "official-middle":
            if args.truncate_max_tokens is None or args.truncate_max_tokens <= 0:
                raise ValueError("official-middle requires a positive --truncate-max-tokens.")
        elif args.truncate_max_tokens is not None:
            raise ValueError("--truncate-max-tokens requires --overflow-policy official-middle.")
        if args.engine == "sparseengine" and (args.engine_kwargs or args.server_url):
            raise ValueError("Native quality does not accept external engine/server options.")
        if args.engine != "sparseengine" and args.hyper_param_json:
            raise ValueError("External quality does not accept native --hyper-param-json.")
        infer_config = _build_infer_config(args)
        engine_kwargs = _load_json_object(args.engine_kwargs)
        if not args.data_path:
            raise FileNotFoundError(
                "LongBench v2 data is not configured. Pass --data-path or set "
                f"{DEFAULT_DATA_ENV} to an official .json/.jsonl export."
            )
        if args.max_model_len <= 0 or args.max_new_tokens <= 0 or args.batch_size <= 0:
            raise ValueError("max-model-len, max-new-tokens, and batch-size must be positive.")
        if args.temperature < 0.0:
            raise ValueError("temperature must be non-negative.")
        max_prompt_tokens = args.max_model_len - args.max_new_tokens
        if max_prompt_tokens <= 0:
            raise ValueError("max-new-tokens leaves no LongBench v2 prompt budget.")
        buckets = () if args.all_samples else parse_token_buckets(args.token_buckets_json)
        oversized = [
            bucket.name
            for bucket in buckets
            if bucket.max_prompt_tokens > max_prompt_tokens
        ]
        if oversized:
            raise ValueError(
                "LongBench v2 token buckets exceed the untruncated prompt budget "
                f"{max_prompt_tokens}: {oversized}."
            )

        data_path = Path(args.data_path).resolve()
        prompt_path = Path(args.prompt_template).resolve()
        if not prompt_path.is_file():
            raise FileNotFoundError(
                f"LongBench v2 official prompt is missing: {prompt_path}. Run "
                "git submodule update --init benchmark/long_bench_v2/upstream."
            )
        upstream_root = Path(__file__).with_name("upstream")
        upstream_commit = _submodule_commit(upstream_root)
        template = prompt_path.read_text(encoding="utf-8")
        source_rows = load_dataset(data_path)
        source_sha256 = file_sha256(data_path)
        if args.official_length:
            source_rows = [row for row in source_rows if row["length"] == args.official_length]
            if not source_rows:
                raise ValueError("The requested official length cohort is empty.")

        phase = "tokenizer"
        tokenizer_path = args.tokenizer_path or args.model_path
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )

        def prepare_chat(prompt: str) -> tuple[str, list[int]]:
            return tokenize_chat(tokenizer, prompt, no_chat_template=args.no_chat_template)

        def prepare_prompt(sample: dict[str, Any]) -> tuple[str, list[int]]:
            return prepare_chat(render_prompt(template, sample))

        phase = "selection"
        if args.prepared_samples:
            prepared = json.loads(Path(args.prepared_samples).read_text())
            expected = {"data_sha256": source_sha256,
                        "prompt_template_sha256": file_sha256(prompt_path),
                        "tokenizer_path": str(Path(tokenizer_path).resolve()),
                        "seed": args.seed, "no_chat_template": args.no_chat_template,
                        "official_length": args.official_length,
                        "token_buckets": [bucket.__dict__ for bucket in buckets]}
            if args.all_samples:
                expected.update(all_samples=True, overflow_policy=args.overflow_policy,
                                max_prompt_tokens=max_prompt_tokens,
                                truncate_max_tokens=args.truncate_max_tokens)
            prepared_identity = dict(prepared["identity"])
            prepared_identity["tokenizer_path"] = str(Path(prepared_identity["tokenizer_path"]).resolve())
            if prepared_identity != expected:
                different = [key for key in expected if prepared_identity.get(key) != expected[key]]
                raise ValueError(f"Prepared sample identity mismatch: {different}.")
            selected = prepared["samples"]
            if any(len(item["prompt_token_ids"]) != item["prompt_tokens"] or
                   not 0 < item["prompt_tokens"] <= max_prompt_tokens for item in selected):
                raise ValueError("Prepared samples have invalid or oversized token sequences.")
        elif args.all_samples and args.preprocess_workers > 1:
            selected = prepare_official_samples_parallel(
                source_rows, tokenizer_path=tokenizer_path, template=template,
                no_chat_template=args.no_chat_template, truncate_max_tokens=args.truncate_max_tokens,
                max_prompt_tokens=max_prompt_tokens, workers=args.preprocess_workers,
            )
        elif args.overflow_policy == "official-middle":
            selected = prepare_official_samples(
                source_rows, template=template, tokenizer=tokenizer, prepare_chat=prepare_chat,
                truncate_max_tokens=args.truncate_max_tokens, max_prompt_tokens=max_prompt_tokens,
            )
        else:
            selected = select_samples(
                source_rows, buckets=buckets, seed=args.seed,
                prepare_prompt=prepare_prompt, max_prompt_tokens=max_prompt_tokens,
            )
        if args.all_samples and [item["sample"] for item in selected] != source_rows:
            raise ValueError("Full-dataset samples must match every source row in source order.")
        identities = [_identity(item) for item in selected]
        _write_jsonl(output_dir / "dataset.jsonl", identities)

        resolved_config = {
            "benchmark": "longbench_v2",
            "protocol": ("official_0shot_direct_all_pre_chat_middle"
                         if args.all_samples else "official_0shot_direct_untruncated_token_stratified"),
            "all_samples": args.all_samples,
            "overflow_policy": args.overflow_policy,
            "max_prompt_tokens": max_prompt_tokens,
            "truncate_max_tokens": args.truncate_max_tokens,
            "truncated_samples": sum(item.get("truncated", False) for item in selected),
            "model_path": str(Path(args.model_path).resolve()),
            "tokenizer_path": str(Path(tokenizer_path).resolve()),
            "sparse_method": args.sparse_method,
            "engine": args.engine,
            "engine_kwargs": engine_kwargs,
            "server_url": args.server_url,
            "official_length": args.official_length,
            "prepared_samples_sha256": file_sha256(args.prepared_samples) if args.prepared_samples else None,
            "deltakv_checkpoint_path": args.deltakv_checkpoint_path,
            "data_path": str(data_path),
            "data_sha256": source_sha256,
            "source_samples": len(source_rows),
            "prompt_template": str(prompt_path),
            "prompt_template_sha256": file_sha256(prompt_path),
            "upstream_commit": upstream_commit,
            "max_model_len": args.max_model_len,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "token_buckets": [bucket.__dict__ for bucket in buckets],
            "selected_samples": len(selected),
            "preprocess_workers": args.preprocess_workers,
            "no_chat_template": args.no_chat_template,
            "requested_runtime": infer_config,
        }
        _write_json(output_dir / "resolved_config.json", resolved_config)
        if args.prepare_only:
            identity_keys = ["data_sha256", "prompt_template_sha256", "tokenizer_path", "seed",
                             "no_chat_template", "official_length", "token_buckets"]
            if args.all_samples:
                identity_keys.extend(["all_samples", "overflow_policy", "max_prompt_tokens",
                                      "truncate_max_tokens"])
            _write_json(output_dir / "prepared_samples.json", {
                "identity": {key: resolved_config[key] for key in identity_keys},
                "samples": selected,
            })
            _write_json(output_dir / "run_status.json", {"status": "prepared", "samples": len(selected)})
            return 0

        phase = "model"
        eos_token_ids = _eos_token_ids(args.model_path, tokenizer)
        llm = None
        if args.engine == "sparseengine":
            generate = get_sparseengine_generate_api(
                model_path=args.model_path, infer_config=infer_config,
                deltakv_checkpoint_path=args.deltakv_checkpoint_path,
                sparse_method=args.sparse_method,
            )
            llm = getattr(generate, "_sparseengine_llm", None)
            if llm is None:
                raise RuntimeError("Sparse-Engine adapter did not expose runtime provenance.")
            resolved_config["effective_runtime"] = llm.worker_info(tags=["longbench-v2-quality"])
        else:
            from benchmark.long_bench_v2.external import get_generate_api
            generate, runtime = get_generate_api(
                engine=args.engine, model_path=args.model_path,
                max_model_len=args.max_model_len, seed=args.seed,
                engine_kwargs=resolved_config["engine_kwargs"], server_url=args.server_url,
                sparse_method=args.sparse_method,
            )
            resolved_config["effective_runtime"] = runtime
        _write_json(output_dir / "resolved_config.json", resolved_config)

        phase = "generation"
        results: list[dict[str, Any]] = []
        aborted_error: str | None = None
        for offset in range(0, len(selected), args.batch_size):
            batch = selected[offset : offset + args.batch_size]
            if aborted_error is not None:
                results.extend(
                    _record_failure(item, "model_failed", aborted_error) for item in batch
                )
                continue
            try:
                responses = generate(
                    [item["prompt_token_ids"] for item in batch],
                    max_new_tokens=args.max_new_tokens,
                    num_beams=1,
                    do_sample=args.temperature > 0.0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    eos_token_id=eos_token_ids,
                )
                if isinstance(responses, str):
                    responses = [responses]
                if len(responses) != len(batch):
                    raise RuntimeError(
                        f"model returned {len(responses)} responses for {len(batch)} prompts."
                    )
            except Exception as exc:
                aborted_error = repr(exc)
                results.extend(
                    _record_failure(item, "model_failed", aborted_error) for item in batch
                )
                continue

            for item, response in zip(batch, responses):
                if not isinstance(response, str):
                    results.append(
                        _record_failure(
                            item,
                            "model_failed",
                            f"model response must be str, got {type(response).__name__}.",
                            raw_response=repr(response),
                        )
                    )
                    continue
                response = response.strip()
                if not response:
                    results.append(
                        _record_failure(
                            item,
                            "model_failed",
                            "model returned an empty response.",
                        )
                    )
                    continue
                predicted = extract_answer(response)
                if predicted is None:
                    results.append(
                        _record_failure(
                            item,
                            "parse_failed",
                            "response did not match the official LongBench v2 answer format.",
                            raw_response=response,
                        )
                    )
                    continue
                results.append(
                    {
                        **_identity(item),
                        "status": "success",
                        "raw_response": response,
                        "predicted_answer": predicted,
                        "correct": predicted == item["sample"]["answer"],
                    }
                )
            _write_jsonl(output_dir / "sample_results.partial.jsonl", results)
            _write_json(output_dir / "progress.json", {
                "finished": len(results), "expected": len(selected),
                "last_sample": batch[-1]["sample"]["_id"],
            })
            print(f"LongBench v2 progress: {len(results)}/{len(selected)}", flush=True)

        if len(results) != len(selected):
            raise RuntimeError(
                f"LongBench v2 result coverage mismatch: expected={len(selected)} "
                f"actual={len(results)}."
            )
        results.sort(key=lambda row: int(row["index"]))
        _write_jsonl(
            output_dir / "raw_outputs.jsonl",
            [
                {
                    **{key: row[key] for key in ("index", "_id", "status", "prompt_tokens")},
                    "raw_response": row.get("raw_response", ""),
                    **({"error": row["error"]} if "error" in row else {}),
                }
                for row in results
            ],
        )
        _write_jsonl(
            output_dir / "parsed_outputs.jsonl",
            [
                {
                    **{key: row[key] for key in ("index", "_id", "status")},
                    "predicted_answer": row.get("predicted_answer"),
                    **({"error": row["error"]} if "error" in row else {}),
                }
                for row in results
            ],
        )
        _write_jsonl(output_dir / "sample_results.jsonl", results)
        aggregate = aggregate_results(results)
        aggregate.update(
            {
                "data_sha256": source_sha256,
                "upstream_commit": upstream_commit,
                "protocol": resolved_config["protocol"],
            }
        )
        _write_json(output_dir / "aggregate_metrics.json", aggregate)
        _write_json(
            output_dir / "operator_runtime_stats.json",
            {"status": "success", "world_ranks": llm.operator_runtime_stats()}
            if llm is not None else {"status": "not_available", "engine": args.engine},
        )
        if aggregate["status"] != "success":
            raise RuntimeError(
                "LongBench v2 contains "
                f"{aggregate['failed_samples']} execution-failed samples."
            )
        _write_json(output_dir / "run_status.json", {"status": "completed"})
        return 0
    except Exception as exc:
        _write_json(
            output_dir / "run_status.json",
            {
                "status": "failed",
                "phase": phase,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
