#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark.sparseengine_regression.grading import (
    GateGrade,
    grade_memory,
    grade_longbench_v2_quality,
    grade_perf,
    grade_quality,
    grade_ruler_quality,
    grade_stress,
    grade_stress_v2,
    worst_required_grade,
)
from benchmark.ruler_vt.tasks import SUPPORTED_TASKS
from benchmark.sparseengine_regression.manifest import (
    deltakv_checkpoint_path_for,
    load_manifest,
    missing_runtime_inputs,
    resolve_method_config,
    resolve_manifest_paths,
    runtime_support_reason,
    select_entries,
)
from sparseengine.method_registry import (
    PREFIX_CACHE_SUPPORTED_METHODS,
    is_decode_cuda_graph_supported,
    is_tp_decode_cuda_graph_supported,
    normalize_sparse_method,
)
DEFAULT_OUTPUT_ROOT = os.getenv("SPARSEENGINE_OUTPUT_DIR", str(REPO_ROOT / "outputs"))


class CommandExecutionError(RuntimeError):
    def __init__(self, message: str, record: dict[str, Any]) -> None:
        super().__init__(message)
        self.record = record


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_int_csv(value: str) -> list[int]:
    items = _parse_csv(value)
    if not items:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    return [int(item) for item in items]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")


def _append_jsonl_file(dst: Path, src: Path, extra: dict[str, Any]) -> None:
    if not src.exists():
        return
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object rows in {src}, got {type(payload).__name__}.")
        _append_jsonl(dst, {**extra, **payload})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _require_synchronized_step_timing(
    rows: list[dict[str, Any]],
    *,
    artifact: Path,
) -> None:
    unsynchronized = [
        row
        for row in rows
        if row.get("status") == "SUCCESS"
        and row.get("synchronize_step_timing") is not True
    ]
    if unsynchronized:
        raise RuntimeError(
            "Regression throughput requires synchronized llm.step timing; "
            f"found {len(unsynchronized)} invalid SUCCESS rows in {artifact}."
        )


def _require_successful_perf_matrix(
    rows: list[dict[str, Any]],
    *,
    methods: tuple[str, ...] | list[str],
    lengths: tuple[int, ...] | list[int],
    batch_sizes: tuple[int, ...] | list[int],
    artifact: Path,
) -> None:
    expected = {
        (str(method), int(length), int(batch_size))
        for method in methods
        for length in lengths
        for batch_size in batch_sizes
    }
    observed: dict[tuple[str, int, int], dict[str, Any]] = {}
    duplicates: list[tuple[str, int, int]] = []
    malformed: list[str] = []
    for index, row in enumerate(rows):
        try:
            key = (
                str(row["method"]),
                int(row["length"]),
                int(row["batch_size"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            malformed.append(f"row={index}: {exc}")
            continue
        if key in observed:
            duplicates.append(key)
            continue
        observed[key] = row

    missing = sorted(expected - set(observed))
    unexpected = sorted(set(observed) - expected)
    non_success = sorted(
        (key, str(observed[key].get("status", "UNKNOWN")))
        for key in expected & set(observed)
        if observed[key].get("status") != "SUCCESS"
    )
    problems: list[str] = []
    if malformed:
        problems.append(f"malformed={malformed}")
    if duplicates:
        problems.append(f"duplicate={sorted(set(duplicates))}")
    if missing:
        problems.append(f"missing={missing}")
    if unexpected:
        problems.append(f"unexpected={unexpected}")
    if non_success:
        problems.append(f"non-success={non_success}")
    if problems:
        raise RuntimeError(
            f"Performance artifact {artifact} does not contain the exact successful "
            f"benchmark matrix: {'; '.join(problems)}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}.")
    return payload


def _ensure_artifacts(output_root: Path, outputs: list[str]) -> None:
    for name in outputs:
        path = output_root / name
        if path.exists():
            continue
        if name.endswith(".jsonl"):
            path.write_text("", encoding="utf-8")
        elif name.endswith(".json"):
            _write_json(path, {})


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def _git_status_short() -> str:
    return subprocess.check_output(["git", "status", "--short"], text=True).strip()


def _terminate_process_group(pid: int, log: Any) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception as exc:  # pragma: no cover - defensive logging for stuck GPU jobs.
        log.write(f"\n[run_suite] failed to terminate process group {pid}: {exc!r}\n")
        log.flush()


def _kill_process_group(pid: int, log: Any) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception as exc:  # pragma: no cover - defensive logging for stuck GPU jobs.
        log.write(f"\n[run_suite] failed to kill process group {pid}: {exc!r}\n")
        log.flush()


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    dry_run: bool,
    log_path: Path,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    timeout_value = float(timeout_s or 0.0)
    record = {
        "cmd": cmd,
        "cwd": str(cwd),
        "log_path": str(log_path),
        "dry_run": dry_run,
        "timeout_s": timeout_value if timeout_value > 0 else None,
    }
    if dry_run:
        return {**record, "status": "skipped_by_policy", "returncode": None}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    pythonpath_parts = [str(cwd), str(cwd / "src")]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=timeout_value if timeout_value > 0 else None)
        except subprocess.TimeoutExpired:
            log.write(f"\n[run_suite] command exceeded timeout_s={timeout_value}; terminating process group.\n")
            log.flush()
            _terminate_process_group(proc.pid, log)
            try:
                returncode = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                log.write("\n[run_suite] process group did not exit after SIGTERM; sending SIGKILL.\n")
                log.flush()
                _kill_process_group(proc.pid, log)
                returncode = proc.wait(timeout=30)
            record["returncode"] = int(returncode)
            record["status"] = "timeout"
            raise CommandExecutionError(f"Command exceeded timeout_s={timeout_value}: {' '.join(cmd)}", record)
    record["returncode"] = int(returncode)
    record["status"] = "success" if returncode == 0 else "model_failed"
    if returncode != 0:
        raise CommandExecutionError(f"Command failed with exit code {returncode}: {' '.join(cmd)}", record)
    return record


def _run_and_record(
    summary: dict[str, Any],
    cmd: list[str],
    *,
    cwd: Path,
    dry_run: bool,
    log_path: Path,
    timeout_s: float | None,
) -> None:
    try:
        record = _run_command(cmd, cwd=cwd, dry_run=dry_run, log_path=log_path, timeout_s=timeout_s)
    except CommandExecutionError as exc:
        summary["commands"].append(exc.record)
        raise
    summary["commands"].append(record)


def _method_config(
    method: dict[str, Any],
    *,
    model: dict[str, Any] | None = None,
    model_id: str | None = None,
    include_method: bool = True,
) -> dict[str, Any]:
    cfg = resolve_method_config(
        method,
        model_id=model_id,
        include_method=include_method,
    )
    checkpoint_path = deltakv_checkpoint_path_for(model or {}, method)
    if method.get("requires_compressor") and checkpoint_path:
        cfg["deltakv_checkpoint_path"] = checkpoint_path
    return cfg


def _tensor_parallel_size_from_config(*configs: dict[str, Any] | None) -> int:
    for cfg in configs:
        if cfg and "tensor_parallel_size" in cfg:
            value = int(cfg["tensor_parallel_size"])
            if value <= 0:
                raise ValueError(f"tensor_parallel_size must be > 0, got {value}.")
            return value
    return 1


def _runtime_tensor_parallel_sizes(
    layer: str,
    resolved: dict[str, Any],
) -> tuple[int, ...]:
    """TP sizes exercised by a regression layer for pair compatibility gates."""
    quality_tp = _tensor_parallel_size_from_config(
        resolved.get("quality"), resolved.get("performance")
    )
    longbench_v2_tp = _tensor_parallel_size_from_config(
        resolved.get("longbench_v2"), resolved.get("performance")
    )
    ruler_tp = _tensor_parallel_size_from_config(
        resolved.get("ruler"), resolved.get("performance")
    )
    perf_tp = _tensor_parallel_size_from_config(resolved.get("performance"))
    stress_tp = _tensor_parallel_size_from_config(
        resolved.get("stress"), resolved.get("performance")
    )
    sizes_by_layer = {
        "validate": (),
        "quality": (quality_tp, longbench_v2_tp, ruler_tp),
        "longbench_v2": (longbench_v2_tp,),
        "ruler": (ruler_tp,),
        "perf": (perf_tp,),
        "stress": (stress_tp,),
        "stress_v2": (_tensor_parallel_size_from_config(resolved.get("stress_v2")),),
        "scbench": (_tensor_parallel_size_from_config(resolved.get("scbench")),),
        "nightly": (quality_tp, longbench_v2_tp, ruler_tp, 1, perf_tp),
        "pre-refactor": (
            quality_tp,
            longbench_v2_tp,
            ruler_tp,
            1,
            perf_tp,
            stress_tp,
        ),
    }
    return tuple(sorted(set(sizes_by_layer[layer])))


def _apply_prefix_cache_config(
    cfg: dict[str, Any],
    method: dict[str, Any],
    *configs: dict[str, Any] | None,
    default_salt: str,
) -> None:
    prefix_cfg: dict[str, Any] = {}
    for source in configs:
        if not source:
            continue
        for key in (
            "enable_prefix_caching",
            "prefix_cache_block_size",
            "prefix_cache_max_blocks",
            "prefix_cache_salt",
        ):
            if key in source:
                prefix_cfg[key] = source[key]

    if not bool(prefix_cfg.get("enable_prefix_caching", False)):
        return

    sparse_method = normalize_sparse_method(method["sparse_method"])
    if sparse_method not in PREFIX_CACHE_SUPPORTED_METHODS:
        supported = ", ".join(repr(name or "vanilla") for name in sorted(PREFIX_CACHE_SUPPORTED_METHODS))
        raise ValueError(
            "enable_prefix_caching in regression runtime config supports these methods only: "
            f"{supported}. got sparse_method={method['sparse_method']!r}."
        )

    cfg["enable_prefix_caching"] = True
    if "prefix_cache_block_size" in prefix_cfg:
        cfg["prefix_cache_block_size"] = int(prefix_cfg["prefix_cache_block_size"])
    if "prefix_cache_max_blocks" in prefix_cfg:
        cfg["prefix_cache_max_blocks"] = int(prefix_cfg["prefix_cache_max_blocks"])
    cfg["prefix_cache_salt"] = str(prefix_cfg.get("prefix_cache_salt") or default_salt)


def _apply_profiler_config(cfg: dict[str, Any], *configs: dict[str, Any] | None) -> None:
    for source in configs:
        if source and bool(source.get("enable_profiler", False)):
            cfg["enable_profiler"] = True
            return


def _decode_cuda_graph_for_method(
    method: dict[str, Any],
    requested: bool,
    *,
    tensor_parallel_size: int = 1,
) -> bool:
    if not requested:
        return False
    if int(tensor_parallel_size) > 1:
        if not is_tp_decode_cuda_graph_supported(method["sparse_method"]):
            raise ValueError(
                "decode_graph with tensor_parallel_size > 1 is a v1 gate for "
                "vanilla, streamingllm, snapkv, pyramidkv, omnikv, quest, rkv, and skipkv only; "
                f"got sparse_method={method['sparse_method']!r}."
            )
        return True
    return is_decode_cuda_graph_supported(method["sparse_method"])


def _quality_command(
    *,
    model_id: str,
    method_id: str,
    model: dict[str, Any],
    method: dict[str, Any],
    quality: dict[str, Any],
    performance: dict[str, Any] | None = None,
    output_root: Path,
) -> list[str]:
    cfg = _method_config(method, model=model, model_id=model_id)
    tensor_parallel_size = _tensor_parallel_size_from_config(quality, performance)
    cfg["tensor_parallel_size"] = int(tensor_parallel_size)
    cfg["decode_graph"] = _decode_cuda_graph_for_method(
        method,
        bool((performance or {}).get("decode_graph", False)),
        tensor_parallel_size=tensor_parallel_size,
    )
    _apply_prefix_cache_config(
        cfg,
        method,
        quality,
        performance,
        default_salt=f"regression-quality:{model_id}:{method_id}",
    )
    _apply_profiler_config(cfg, quality, performance)
    if tensor_parallel_size > 1:
        cfg["decode_graph_capture_sampling"] = False
    if "sparseengine_max_num_seqs_in_batch" in quality:
        cfg["max_num_seqs_in_batch"] = int(quality["sparseengine_max_num_seqs_in_batch"])
    if "sparseengine_max_decoding_seqs" in quality:
        cfg["max_decoding_seqs"] = int(quality["sparseengine_max_decoding_seqs"])
    cmd = [
        sys.executable,
        "benchmark/long_bench/pred.py",
        "--model",
        f"{model_id}-{method_id}",
        "--model_path",
        model["model_path"],
        "--tokenizer_path",
        model["tokenizer_path"],
        "--ws",
        str(int(quality.get("worker_world_size", quality.get("ws", 1)))),
        "--batch_size",
        str(int(quality.get("batch_size", 1))),
        "--sparse_method",
        method["sparse_method"],
        "--task",
        ",".join(quality["tasks"]),
        "--min_prompt_tokens",
        str(int(quality["min_prompt_tokens"])),
        "--samples_per_task",
        str(int(quality["samples_per_task"])),
        "--min_required_samples",
        str(int(quality["min_required_samples"])),
        "--temperature",
        str(float(quality["temperature"])),
        "--top_p",
        str(float(quality["top_p"])),
        "--top_k",
        str(int(quality["top_k"])),
    ]
    if "max_model_len" in quality:
        cmd.extend(["--max_model_len", str(int(quality["max_model_len"]))])
    if bool(cfg.get("enable_prefix_caching", False)):
        cmd.append("--allow_prefix_caching")
    cmd.extend(
        [
            "--hyper_param",
            json.dumps(cfg, sort_keys=True),
            "--output_root",
            str(output_root),
        ]
    )
    return cmd


def _longbench_v2_command(
    *,
    model_id: str,
    method_id: str,
    model: dict[str, Any],
    method: dict[str, Any],
    longbench_v2: dict[str, Any],
    performance: dict[str, Any] | None,
    output_root: Path,
) -> list[str]:
    data_path = longbench_v2.get("data_path")
    if not data_path:
        raise FileNotFoundError(
            "LongBench v2 data is not configured; set "
            f"{longbench_v2['data_path_env']} to an official JSON/JSONL export."
        )
    if not Path(data_path).is_file():
        raise FileNotFoundError(f"LongBench v2 data file does not exist: {data_path}")

    cfg = _method_config(method, model=model, model_id=model_id)
    tensor_parallel_size = _tensor_parallel_size_from_config(
        longbench_v2, performance
    )
    cfg["tensor_parallel_size"] = int(tensor_parallel_size)
    cfg["decode_graph"] = _decode_cuda_graph_for_method(
        method,
        bool((performance or {}).get("decode_graph", False)),
        tensor_parallel_size=tensor_parallel_size,
    )
    _apply_prefix_cache_config(
        cfg,
        method,
        longbench_v2,
        performance,
        default_salt=f"regression-longbench-v2:{model_id}:{method_id}",
    )
    _apply_profiler_config(cfg, longbench_v2, performance)
    if tensor_parallel_size > 1:
        cfg["decode_graph_capture_sampling"] = False
    if "sparseengine_max_num_seqs_in_batch" in longbench_v2:
        cfg["max_num_seqs_in_batch"] = int(
            longbench_v2["sparseengine_max_num_seqs_in_batch"]
        )
    if "sparseengine_max_decoding_seqs" in longbench_v2:
        cfg["max_decoding_seqs"] = int(
            longbench_v2["sparseengine_max_decoding_seqs"]
        )

    cmd = [
        sys.executable,
        "benchmark/long_bench_v2/pred.py",
        "--model-path",
        model["model_path"],
        "--tokenizer-path",
        model["tokenizer_path"],
        "--sparse-method",
        method["sparse_method"],
        "--hyper-param-json",
        json.dumps(cfg, sort_keys=True),
        "--output-dir",
        str(output_root),
        "--data-path",
        str(data_path),
        "--max-model-len",
        str(int(longbench_v2["max_model_len"])),
        "--max-new-tokens",
        str(int(longbench_v2["max_new_tokens"])),
        "--batch-size",
        str(int(longbench_v2["batch_size"])),
        "--temperature",
        str(float(longbench_v2["temperature"])),
        "--top-p",
        str(float(longbench_v2["top_p"])),
        "--top-k",
        str(int(longbench_v2["top_k"])),
        "--seed",
        str(int(longbench_v2["seed"])),
        "--token-buckets-json",
        json.dumps(longbench_v2["token_buckets"], sort_keys=True),
    ]
    checkpoint_path = deltakv_checkpoint_path_for(model, method)
    if checkpoint_path:
        cmd.extend(["--deltakv-checkpoint-path", checkpoint_path])
    if bool(cfg.get("enable_prefix_caching", False)):
        cmd.append("--allow-prefix-caching")
    return cmd


def _ruler_command(
    *,
    task: str,
    task_config: dict[str, Any],
    model_id: str,
    method_id: str,
    model: dict[str, Any],
    method: dict[str, Any],
    ruler: dict[str, Any],
    performance: dict[str, Any] | None,
    output_root: Path,
) -> list[str]:
    cfg = _method_config(method, model=model, model_id=model_id)
    tensor_parallel_size = _tensor_parallel_size_from_config(ruler, performance)
    worker_world_size = int(ruler.get("worker_world_size", ruler.get("ws", 1)))
    if worker_world_size > 1 and tensor_parallel_size > 1:
        raise ValueError(
            "RULER does not support data-parallel worker_world_size > 1 together "
            "with tensor_parallel_size > 1 because each data-parallel worker is "
            "assigned exactly one visible GPU; got "
            f"worker_world_size={worker_world_size}, "
            f"tensor_parallel_size={tensor_parallel_size}."
        )
    cfg["tensor_parallel_size"] = int(tensor_parallel_size)
    cfg["decode_graph"] = _decode_cuda_graph_for_method(
        method,
        bool((performance or {}).get("decode_graph", False)),
        tensor_parallel_size=tensor_parallel_size,
    )
    _apply_prefix_cache_config(
        cfg,
        method,
        ruler,
        performance,
        default_salt=f"regression-ruler:{task}:{model_id}:{method_id}",
    )
    _apply_profiler_config(cfg, ruler, performance)
    if tensor_parallel_size > 1:
        cfg["decode_graph_capture_sampling"] = False
    if "sparseengine_max_num_seqs_in_batch" in ruler:
        cfg["max_num_seqs_in_batch"] = int(
            ruler["sparseengine_max_num_seqs_in_batch"]
        )
    if "sparseengine_max_decoding_seqs" in ruler:
        cfg["max_decoding_seqs"] = int(ruler["sparseengine_max_decoding_seqs"])

    cmd = [
        sys.executable,
        "benchmark/ruler_vt/pred.py",
        "--model-path",
        model["model_path"],
        "--tokenizer-path",
        model["tokenizer_path"],
        "--output-dir",
        str(output_root),
        "--task",
        task,
        "--task-config-json",
        json.dumps(task_config, sort_keys=True),
        "--hyper-param-json",
        json.dumps(cfg, sort_keys=True),
        "--sparse-method",
        method["sparse_method"],
        "--context-lengths",
        ",".join(str(int(length)) for length in ruler["context_lengths"]),
        "--samples-per-length",
        str(int(ruler["samples_per_length"])),
        "--tokens-to-generate",
        str(int(task_config["tokens_to_generate"])),
        "--max-new-tokens",
        str(int(task_config["max_new_tokens"])),
        "--minimum-context-utilization",
        str(float(ruler["minimum_context_utilization"])),
        "--batch-size",
        str(int(ruler["batch_size"])),
        "--temperature",
        str(float(ruler["temperature"])),
        "--seed",
        str(int(ruler.get("seed", 20260608))),
        "--ws",
        str(worker_world_size),
    ]
    if "max_model_len" in ruler:
        cmd.extend(["--max-model-len", str(int(ruler["max_model_len"]))])
    if bool(cfg.get("enable_prefix_caching", False)):
        cmd.extend(
            [
                "--allow-prefix-caching",
                "--prefix-cache-replay",
                "--require-prefix-cache-hit",
            ]
        )
    return cmd


def _perf_command(
    *,
    model_id: str,
    model: dict[str, Any],
    method_id: str,
    method: dict[str, Any],
    performance: dict[str, Any],
    output_jsonl: Path,
) -> list[str]:
    tensor_parallel_size = _tensor_parallel_size_from_config(performance)
    hyper_params = {
        "decode_graph": _decode_cuda_graph_for_method(
            method,
            bool(performance["decode_graph"]),
            tensor_parallel_size=tensor_parallel_size,
        ),
        "tensor_parallel_size": int(tensor_parallel_size),
        "throughput_log_interval_s": 0.0,
    }
    if tensor_parallel_size > 1:
        hyper_params["decode_graph_capture_sampling"] = False
    method_cfg = _method_config(method, model=model, model_id=model_id, include_method=False)
    hyper_params.update(method_cfg)
    _apply_prefix_cache_config(
        hyper_params,
        method,
        performance,
        default_salt=f"regression-perf:{model_id}:{method_id}",
    )
    _apply_profiler_config(hyper_params, performance)
    methods_arg = "vanilla" if method_id == "vanilla" else f"vanilla,{method_id}"
    return [
        sys.executable,
        "scripts/benchmarks/bench_sparse_engine.py",
        "--model_path",
        model["model_path"],
        "--lengths",
        ",".join(str(int(x)) for x in performance["lengths"]),
        "--batch_sizes",
        ",".join(str(int(x)) for x in performance["batch_sizes"]),
        "--methods",
        methods_arg,
        "--output_len",
        str(int(performance["output_len"])),
        "--temperature",
        "0.0",
        "--hyper_params",
        json.dumps(hyper_params, sort_keys=True),
        "--output_jsonl",
        str(output_jsonl),
        "--synchronize_step_timing",
    ]


def _stress_command(
    *,
    model_id: str,
    model: dict[str, Any],
    method_id: str,
    method: dict[str, Any],
    performance: dict[str, Any],
    stress: dict[str, Any],
    output_jsonl: Path,
) -> list[str]:
    request_counts = [int(x) for x in stress["request_counts"]]
    tensor_parallel_size = _tensor_parallel_size_from_config(stress, performance)
    hyper_params = {
        "decode_graph": _decode_cuda_graph_for_method(
            method,
            bool(performance.get("decode_graph", True)),
            tensor_parallel_size=tensor_parallel_size,
        ),
        "tensor_parallel_size": int(tensor_parallel_size),
        "throughput_log_interval_s": 0.0,
        "max_num_seqs_in_batch": int(stress.get("max_num_seqs_in_batch", max(request_counts))),
        "max_decoding_seqs": int(stress.get("max_decoding_seqs", max(request_counts))),
    }
    if tensor_parallel_size > 1:
        hyper_params["decode_graph_capture_sampling"] = False
    method_cfg = _method_config(method, model=model, model_id=model_id, include_method=False)
    hyper_params.update(method_cfg)
    _apply_prefix_cache_config(
        hyper_params,
        method,
        stress,
        performance,
        default_salt=f"regression-stress:{model_id}:{method_id}",
    )
    _apply_profiler_config(hyper_params, stress, performance)
    prefix_cache_stress = bool(hyper_params.get("enable_prefix_caching", False))
    admission_wave_size = int(stress.get("admission_wave_size", 0) or 0)
    if prefix_cache_stress and admission_wave_size <= 0:
        max_request_count = max(request_counts)
        if max_request_count <= 1:
            raise ValueError("Prefix-cache stress requires request_counts greater than 1.")
        admission_wave_size = max(1, max_request_count // 2)
    wave_decode_gap_steps = int(stress.get("wave_decode_gap_steps", 1 if prefix_cache_stress else 0) or 0)
    require_prefix_cache_hit = bool(stress.get("require_prefix_cache_hit", prefix_cache_stress))
    cmd = [
        sys.executable,
        "scripts/benchmarks/bench_sparse_engine.py",
        "--model_path",
        model["model_path"],
        "--lengths",
        str(int(stress["length"])),
        "--batch_sizes",
        ",".join(str(value) for value in request_counts),
        "--methods",
        method_id,
        "--output_len",
        str(int(stress["output_len"])),
        "--temperature",
        "0.0",
        "--hyper_params",
        json.dumps(hyper_params, sort_keys=True),
        "--max_decode_steps_after_full",
        str(int(stress["max_decode_steps_after_full"])),
        "--output_jsonl",
        str(output_jsonl),
        "--synchronize_step_timing",
    ]
    if admission_wave_size > 0:
        cmd.extend(["--admission_wave_size", str(admission_wave_size)])
    if wave_decode_gap_steps > 0:
        cmd.extend(["--wave_decode_gap_steps", str(wave_decode_gap_steps)])
    if require_prefix_cache_hit:
        cmd.append("--require_prefix_cache_hit")
    return cmd


def _stress_v2_cases(method_id: str, method: dict[str, Any]) -> list[str]:
    sparse_method = normalize_sparse_method(method.get("sparse_method", method_id))
    if sparse_method == "":
        sparse_method = "vanilla"
    if sparse_method == "vanilla":
        return ["baseline_full", "prefix_full"]
    if sparse_method == "omnikv":
        return ["prefix_omnikv"]
    if sparse_method == "quest":
        return ["prefix_quest"]
    return []


def _stress_v2_command(
    *,
    model_id: str,
    model: dict[str, Any],
    method_id: str,
    method: dict[str, Any],
    stress_v2: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    cases = _stress_v2_cases(method_id, method)
    if not cases:
        raise ValueError(f"stress_v2 does not support method {method_id!r}.")

    method_cfg = _method_config(method, model=model, model_id=model_id, include_method=False)
    full_attention_layers = method_cfg.get("full_attention_layers", stress_v2.get("full_attention_layers", "0,1,2,4,7,14"))
    bench_hyper_params = dict(method_cfg)
    for managed_key in (
        "gpu_memory_utilization",
        "engine_prefill_chunk_size",
        "sink_keep_tokens",
        "recent_keep_tokens",
        "decode_keep_tokens",
        "full_attention_layers",
        "quest_chunk_size",
        "prefix_cache_block_size",
        "prefix_cache_max_blocks",
        "prefix_cache_salt",
        "enable_prefix_caching",
        "sparse_method",
        "max_num_seqs_in_batch",
        "max_decoding_seqs",
        "max_num_batched_tokens",
        "max_model_len",
        "tensor_parallel_size",
    ):
        bench_hyper_params.pop(managed_key, None)
    cmd = [
        sys.executable,
        "scripts/benchmarks/bench_prefix_cache.py",
        "--model_path",
        model["model_path"],
        "--cases",
        ",".join(cases),
        "--workloads",
        str(stress_v2["workloads"]),
        "--output_dir",
        str(output_dir),
        "--feature",
        "sparseengine_regression_stress_v2",
        "--objective",
        f"run SparseVLLM stress_v2 serving trace for {model_id}/{method_id}",
        "--seed",
        str(int(stress_v2["seed"])),
        "--history_update",
        str(stress_v2["history_update"]),
        "--sessions",
        str(int(stress_v2["sessions"])),
        "--turns",
        str(int(stress_v2["turns"])),
        "--system_prompt_len",
        str(int(stress_v2["system_prompt_len"])),
        "--session_prefix_len",
        str(int(stress_v2["session_prefix_len"])),
        "--user_len",
        str(int(stress_v2["user_len"])),
        "--output_len",
        str(int(stress_v2["output_len"])),
        "--shared_prompts",
        str(int(stress_v2["shared_prompts"])),
        "--shared_prefix_len",
        str(int(stress_v2["shared_prefix_len"])),
        "--shared_suffix_len",
        str(int(stress_v2["shared_suffix_len"])),
        "--gpu_memory_utilization",
        str(float(stress_v2["gpu_memory_utilization"])),
        "--tensor_parallel_size",
        str(_tensor_parallel_size_from_config(stress_v2)),
        "--max_active_requests",
        str(int(stress_v2["max_active_requests"])),
        "--max_num_batched_tokens",
        str(int(stress_v2["max_num_batched_tokens"])),
        "--engine_prefill_chunk_size",
        str(int(method_cfg.get("engine_prefill_chunk_size", stress_v2["engine_prefill_chunk_size"]))),
        "--max_model_len_margin",
        str(int(stress_v2["max_model_len_margin"])),
        "--prefix_cache_block_size",
        str(int(stress_v2["prefix_cache_block_size"])),
        "--prefix_cache_salt",
        str(stress_v2.get("prefix_cache_salt") or f"regression-stress-v2:{model_id}:{method_id}"),
        "--quest_chunk_size",
        str(int(method_cfg.get("quest_chunk_size", stress_v2["quest_chunk_size"]))),
        "--sink_keep_tokens",
        str(int(method_cfg.get("sink_keep_tokens", stress_v2["sink_keep_tokens"]))),
        "--recent_keep_tokens",
        str(int(method_cfg.get("recent_keep_tokens", stress_v2["recent_keep_tokens"]))),
        "--decode_keep_tokens",
        str(int(method_cfg.get("decode_keep_tokens", stress_v2["decode_keep_tokens"]))),
        "--full_attention_layers",
        str(full_attention_layers),
        "--min_performance_prompt_len",
        str(int(stress_v2["min_performance_prompt_len"])),
        "--min_cacheable_prefix_len",
        str(int(stress_v2["min_cacheable_prefix_len"])),
        "--case_timeout_s",
        str(float(stress_v2["case_timeout_s"])),
        "--hyper_params",
        json.dumps(bench_hyper_params, sort_keys=True),
    ]
    for cfg_key, flag in (
        ("session_prefix_min_len", "--session_prefix_min_len"),
        ("user_min_len", "--user_min_len"),
        ("shared_suffix_min_len", "--shared_suffix_min_len"),
        ("prefix_cache_max_blocks", "--prefix_cache_max_blocks"),
    ):
        if cfg_key in stress_v2 and stress_v2[cfg_key] is not None:
            cmd.extend([flag, str(stress_v2[cfg_key])])
    if bool(stress_v2.get("allow_short_trace", False)):
        cmd.append("--allow_short_trace")
    if bool(stress_v2.get("continue_on_failure", False)):
        cmd.append("--continue_on_failure")
    if not bool(stress_v2.get("require_omnikv_prefill_path", True)):
        cmd.append("--no-require_omnikv_prefill_path")
    return cmd


def _scbench_command(
    *,
    manifest_path: Path,
    model_id: str,
    method_ids: list[str],
    scbench: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/benchmarks/run_scbench_sparseengine_methods.py",
        "--manifest",
        str(manifest_path),
        "--model_id",
        model_id,
        "--methods",
        ",".join(method_ids),
        "--tasks",
        ",".join(str(task) for task in scbench["tasks"]),
        "--output_dir",
        str(output_dir),
        "--num_eval_examples",
        str(int(scbench["num_eval_examples"])),
        "--max_turns",
        str(int(scbench["max_turns"])),
        "--max_seq_length",
        str(int(scbench["max_seq_length"])),
        "--batch_size",
        str(int(scbench["batch_size"])),
        "--tensor_parallel_size",
        str(int(scbench.get("tensor_parallel_size", 1))),
        "--prefix_cache_block_size",
        str(int(scbench.get("prefix_cache_block_size", 16))),
    ]
    if scbench.get("trust_remote_code", False):
        cmd.append("--trust_remote_code")
    if scbench.get("use_chat_template", False):
        cmd.append("--use_chat_template")
    if scbench.get("disable_golden_context", False):
        cmd.append("--disable_golden_context")
    if "context_min_tokens" in scbench:
        cmd.extend(["--context_min_tokens", str(int(scbench["context_min_tokens"]))])
    if "context_max_tokens" in scbench:
        cmd.extend(["--context_max_tokens", str(int(scbench["context_max_tokens"]))])
    if "gpu_memory_utilization" in scbench:
        cmd.extend(["--gpu_memory_utilization", str(float(scbench["gpu_memory_utilization"]))])
    if bool(scbench.get("decode_graph", False)):
        cmd.append("--decode_graph")
    return cmd


def _load_result_json(path: Path) -> dict[str, Any] | None:
    result_path = path / "result.json"
    if not result_path.exists():
        return None
    with result_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _overall_score(result: dict[str, Any] | None) -> float | None:
    if not result:
        return None
    value = result.get("overall_category_avg")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _grade_quality_pair(
    vanilla_root: Path,
    sparse_root: Path,
    *,
    minimum_vanilla_score: float,
) -> GateGrade:
    vanilla_score = _overall_score(_load_result_json(vanilla_root))
    sparse_score = _overall_score(_load_result_json(sparse_root))
    if vanilla_score is None or sparse_score is None:
        return GateGrade(
            "quality",
            "D",
            "failed",
            {"vanilla_score": vanilla_score, "sparse_score": sparse_score},
            "Missing LongBench-mini aggregate score.",
        )
    return grade_quality(
        vanilla_score,
        sparse_score,
        minimum_vanilla_score=minimum_vanilla_score,
    )


def _validated_longbench_v2_metrics(
    output_root: Path,
    *,
    token_buckets: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = _read_jsonl(output_root / "sample_results.jsonl")
    expected_count = sum(int(bucket["samples"]) for bucket in token_buckets)
    if len(rows) != expected_count:
        raise RuntimeError(
            f"LongBench v2 artifact {output_root} has incomplete coverage: "
            f"expected={expected_count} actual={len(rows)}."
        )
    if [int(row.get("index", -1)) for row in rows] != list(range(expected_count)):
        raise RuntimeError(
            f"LongBench v2 artifact {output_root} has invalid sample identities."
        )
    failed = [
        row
        for row in rows
        if row.get("status") not in {"success", "parse_failed"}
    ]
    if failed:
        raise RuntimeError(
            f"LongBench v2 artifact {output_root} contains {len(failed)} "
            "execution-failed samples."
        )
    invalid_parse_rows = [
        row
        for row in rows
        if row.get("status") == "parse_failed"
        and (row.get("predicted_answer") is not None or bool(row.get("correct")))
    ]
    if invalid_parse_rows:
        raise RuntimeError(
            f"LongBench v2 artifact {output_root} has invalid parse-failure rows."
        )
    for bucket in token_buckets:
        bucket_rows = [
            row for row in rows if row.get("token_bucket") == bucket["name"]
        ]
        if len(bucket_rows) != int(bucket["samples"]):
            raise RuntimeError(
                f"LongBench v2 artifact {output_root} has incomplete bucket "
                f"{bucket['name']!r}: expected={bucket['samples']} "
                f"actual={len(bucket_rows)}."
            )
        outside = [
            row
            for row in bucket_rows
            if not int(bucket["min_prompt_tokens"])
            <= int(row["prompt_tokens"])
            <= int(bucket["max_prompt_tokens"])
        ]
        if outside:
            raise RuntimeError(
                f"LongBench v2 artifact {output_root} has {len(outside)} rows outside "
                f"bucket {bucket['name']!r}."
            )
    aggregate = _read_json(output_root / "aggregate_metrics.json")
    if aggregate.get("status") != "success":
        raise RuntimeError(
            f"LongBench v2 aggregate is not successful: {output_root}."
        )
    if int(aggregate.get("samples", -1)) != expected_count:
        raise RuntimeError(
            f"LongBench v2 aggregate count mismatch at {output_root}."
        )
    if int(aggregate.get("evaluated_samples", -1)) != expected_count:
        raise RuntimeError(
            f"LongBench v2 evaluated count mismatch at {output_root}."
        )
    if int(aggregate.get("failed_samples", -1)) != 0:
        raise RuntimeError(
            f"LongBench v2 aggregate records execution failures at {output_root}."
        )
    accuracy = aggregate.get("accuracy")
    if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool):
        raise RuntimeError(
            f"LongBench v2 aggregate lacks numeric accuracy: {output_root}."
        )
    return aggregate


def _grade_longbench_v2_pair(
    vanilla_root: Path,
    sparse_root: Path,
    *,
    longbench_v2: dict[str, Any],
) -> GateGrade:
    vanilla_dataset = _read_jsonl(vanilla_root / "dataset.jsonl")
    sparse_dataset = _read_jsonl(sparse_root / "dataset.jsonl")
    if vanilla_dataset != sparse_dataset:
        raise RuntimeError(
            "LongBench v2 vanilla and sparse runs did not use exactly aligned samples: "
            f"vanilla={vanilla_root / 'dataset.jsonl'} "
            f"sparse={sparse_root / 'dataset.jsonl'}."
        )
    token_buckets = list(longbench_v2["token_buckets"])
    vanilla = _validated_longbench_v2_metrics(
        vanilla_root, token_buckets=token_buckets
    )
    sparse = _validated_longbench_v2_metrics(
        sparse_root, token_buckets=token_buckets
    )
    if vanilla.get("data_sha256") != sparse.get("data_sha256"):
        raise RuntimeError(
            "LongBench v2 vanilla and sparse runs used different source dataset hashes."
        )
    grade = grade_longbench_v2_quality(
        float(vanilla["accuracy"]),
        float(sparse["accuracy"]),
        minimum_vanilla_score=float(longbench_v2["minimum_vanilla_score"]),
        maximum_score_loss=float(longbench_v2["maximum_score_loss"]),
    )
    metrics = dict(grade.metrics)
    metrics["vanilla_by_token_bucket"] = vanilla["by_token_bucket"]
    metrics["sparse_by_token_bucket"] = sparse["by_token_bucket"]
    return GateGrade(grade.name, grade.grade, grade.status, metrics, grade.reason)


def _validated_ruler_scores(
    output_root: Path,
    *,
    task: str | None = None,
    context_lengths: list[int],
    samples_per_length: int,
    minimum_context_utilization: float,
) -> dict[int, float]:
    rows = _read_jsonl(output_root / "per_sample_results.jsonl")
    expected_count = len(context_lengths) * int(samples_per_length)
    if len(rows) != expected_count:
        raise RuntimeError(
            f"RULER artifact {output_root} has incomplete coverage: "
            f"expected={expected_count} actual={len(rows)}."
        )
    observed_indices = [int(row["index"]) for row in rows]
    if observed_indices != list(range(expected_count)):
        raise RuntimeError(
            f"RULER artifact {output_root} has invalid sample identities: "
            f"{observed_indices}."
        )
    failed = [row for row in rows if row.get("status") != "success"]
    if failed:
        raise RuntimeError(
            f"RULER artifact {output_root} contains {len(failed)} failed samples."
        )
    if task is not None:
        wrong_task = [row for row in rows if row.get("task", "vt") != task]
        if wrong_task:
            raise RuntimeError(
                f"RULER artifact {output_root} contains {len(wrong_task)} rows "
                f"outside task={task}."
            )
    scores: dict[int, float] = {}
    for context_length in context_lengths:
        length_rows = [
            row for row in rows if int(row["context_length"]) == int(context_length)
        ]
        if len(length_rows) != int(samples_per_length):
            raise RuntimeError(
                f"RULER artifact {output_root} has incomplete context_length="
                f"{context_length}: expected={samples_per_length} actual={len(length_rows)}."
            )
        underfilled = [
            row
            for row in length_rows
            if float(row["length"]) / context_length
            < float(minimum_context_utilization)
        ]
        if underfilled:
            raise RuntimeError(
                f"RULER artifact {output_root} does not exercise context_length="
                f"{context_length} at minimum utilization="
                f"{minimum_context_utilization}."
            )
        scores[int(context_length)] = 100.0 * sum(
            float(row["score"]) for row in length_rows
        ) / len(length_rows)
    return scores


def _require_aligned_ruler_datasets(vanilla_root: Path, sparse_root: Path) -> None:
    vanilla = _read_jsonl(vanilla_root / "dataset.jsonl")
    sparse = _read_jsonl(sparse_root / "dataset.jsonl")
    identity_keys = (
        "index",
        "context_length",
        "input",
        "outputs",
        "length",
        "answer_prefix",
        "others",
    )
    vanilla_identity = [tuple(row.get(key) for key in identity_keys) for row in vanilla]
    sparse_identity = [tuple(row.get(key) for key in identity_keys) for row in sparse]
    if vanilla_identity != sparse_identity:
        raise RuntimeError(
            "RULER vanilla and sparse runs did not use exactly aligned generated samples: "
            f"vanilla={vanilla_root / 'dataset.jsonl'} sparse={sparse_root / 'dataset.jsonl'}."
        )


def _grade_ruler_pair(
    vanilla_root: Path,
    sparse_root: Path,
    *,
    ruler: dict[str, Any],
    task: str | None = None,
) -> list[tuple[int, GateGrade]]:
    _require_aligned_ruler_datasets(vanilla_root, sparse_root)
    context_lengths = [int(value) for value in ruler["context_lengths"]]
    samples_per_length = int(ruler["samples_per_length"])
    vanilla_scores = _validated_ruler_scores(
        vanilla_root,
        task=task,
        context_lengths=context_lengths,
        samples_per_length=samples_per_length,
        minimum_context_utilization=float(ruler["minimum_context_utilization"]),
    )
    sparse_scores = _validated_ruler_scores(
        sparse_root,
        task=task,
        context_lengths=context_lengths,
        samples_per_length=samples_per_length,
        minimum_context_utilization=float(ruler["minimum_context_utilization"]),
    )
    task_config = (ruler.get("task_configs") or {}).get(task or "", {})
    minimum_vanilla_score = float(
        task_config.get("minimum_vanilla_score", ruler["minimum_vanilla_score"])
    )
    maximum_score_loss = float(
        task_config.get("maximum_score_loss", ruler["maximum_score_loss"])
    )
    return [
        (
            context_length,
            grade_ruler_quality(
                vanilla_scores[context_length],
                sparse_scores[context_length],
                minimum_vanilla_score=minimum_vanilla_score,
                maximum_score_loss=maximum_score_loss,
            ),
        )
        for context_length in context_lengths
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed Sparse-VLLM regression gates.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--layer",
        default="validate",
        choices=[
            "validate",
            "quality",
            "longbench_v2",
            "ruler",
            "perf",
            "stress",
            "stress_v2",
            "agent_trace",
            "scbench",
            "nightly",
            "pre-refactor",
        ],
    )
    parser.add_argument("--models", default=None, help="Comma-separated model ids from the manifest.")
    parser.add_argument("--methods", default=None, help="Comma-separated method ids from the manifest.")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--agent_trace", help="Exported MiniSWE agent trace directory")
    parser.add_argument("--agent_api_base", help="Already-running vanilla OpenAI-compatible server /v1 URL")
    parser.add_argument("--agent_server_manifest", help="Actual target server identity/config/hardware JSON")
    parser.add_argument("--agent_concurrency", type=int, default=16)
    parser.add_argument("--agent_think_time_scale", type=float, default=1.0,
                        help="Scale recorded inter-turn waits; 0 measures replay without tool/network waits.")
    parser.add_argument("--agent_synthetic_think_time_max_s", type=float, default=None,
                        help="Replace recorded waits with deterministic uniform waits in [0, max].")
    parser.add_argument("--agent_synthetic_think_time_seed", type=int, default=42)
    parser.add_argument("--agent_force_recorded_responses", action="store_true",
                        help="Force tokenized recorded assistant answers through the model sampler so next-turn caches match.")
    parser.add_argument("--agent_forced_workload",
                        help="Prepared target-model forced workload JSON; requires --agent_force_recorded_responses.")
    parser.add_argument("--agent_prefix_prune_keep_ratio", type=float, default=None,
                        help="Prune only newly added tool bodies after every turn using KVzip.")
    parser.add_argument("--agent_prefix_prune_tokenizer", default=None)
    parser.add_argument("--agent_prefix_prune_trigger_tokens", type=int, default=8192)
    parser.add_argument("--agent_request_timeout", type=float, default=900)
    parser.add_argument("--agent_require_cache_hit", action="store_true",
                        help="Fail the replay if no response reports cached prompt tokens.")
    parser.add_argument("--agent_api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--agent_baseline", help="Previous successful agent_trace.json to compare")
    parser.add_argument("--agent_max_slowdown", type=float, default=1.10)
    parser.add_argument("--agent_allow_estimated_timing", action="store_true",
                        help="Explicitly accept legacy server-log timing estimates, not exact client delays")
    parser.add_argument("--output_root", default=None)
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help=(
            "Override Sparse-VLLM engine tensor_parallel_size for regression commands. "
            "This is separate from LongBench --ws data-worker parallelism."
        ),
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--allow_skipped_policy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--scbench_decode_graph",
        action="store_true",
        help="Run the SCBench regression subset with decode CUDA graph enabled.",
    )
    parser.add_argument(
        "--enable_prefix_caching",
        action="store_true",
        help=(
            "Enable prefix caching for selected regression methods that support it. "
            "Use with methods vanilla, omnikv, and quest for TP prefix-graph validation."
        ),
    )
    parser.add_argument("--prefix_cache_block_size", type=int, default=None)
    parser.add_argument("--prefix_cache_salt", default=None)
    parser.add_argument(
        "--require_prefix_cache_hit",
        action="store_true",
        help="Require stress rows to observe at least one prefix-cache hit.",
    )
    parser.add_argument(
        "--enable_profiler",
        action="store_true",
        help="Enable Sparse-VLLM profiler in quality, performance, and stress child commands.",
    )
    parser.add_argument(
        "--command_timeout_s",
        type=float,
        default=None,
        help="Per child command timeout. Timed-out command process groups are terminated and recorded as failed.",
    )
    parser.add_argument("--quality_tasks", default=None, help="Override LongBench quality tasks with a comma list.")
    parser.add_argument("--quality_batch_size", type=int, default=None)
    parser.add_argument("--quality_samples_per_task", type=int, default=None)
    parser.add_argument("--quality_min_required_samples", type=int, default=None)
    parser.add_argument("--quality_min_prompt_tokens", type=int, default=None)
    parser.add_argument("--quality_sparseengine_max_num_seqs_in_batch", type=int, default=None)
    parser.add_argument("--quality_sparseengine_max_decoding_seqs", type=int, default=None)
    parser.add_argument(
        "--quality_benchmarks",
        default="longbench,longbench_v2,ruler",
        help="Comma-separated quality benchmarks: longbench,longbench_v2,ruler.",
    )
    parser.add_argument("--longbench_v2_data_path", default=None)
    parser.add_argument("--longbench_v2_max_model_len", type=int, default=None)
    parser.add_argument("--longbench_v2_batch_size", type=int, default=None)
    parser.add_argument(
        "--longbench_v2_token_buckets_json",
        default=None,
        help="Override LongBench v2 token buckets with a JSON list.",
    )
    parser.add_argument(
        "--ruler_tasks",
        default=None,
        help="Override RULER core tasks with a comma list.",
    )
    parser.add_argument("--ruler_context_lengths", default=None)
    parser.add_argument("--ruler_samples_per_length", type=int, default=None)
    parser.add_argument("--ruler_batch_size", type=int, default=None)
    parser.add_argument("--ruler_worker_world_size", type=int, default=None)
    parser.add_argument("--ruler_max_model_len", type=int, default=None)
    parser.add_argument("--scbench_tasks", default=None, help="Override SCBench tasks with a comma list.")
    parser.add_argument("--scbench_num_eval_examples", type=int, default=None)
    parser.add_argument("--scbench_max_turns", type=int, default=None)
    parser.add_argument("--scbench_max_seq_length", type=int, default=None)
    parser.add_argument("--scbench_batch_size", type=int, default=None)
    parser.add_argument("--stress_length", type=int, default=None)
    parser.add_argument("--stress_request_counts", default=None)
    parser.add_argument("--stress_output_len", type=int, default=None)
    parser.add_argument("--stress_max_num_seqs_in_batch", type=int, default=None)
    parser.add_argument("--stress_max_decoding_seqs", type=int, default=None)
    parser.add_argument("--stress_max_decode_steps_after_full", type=int, default=None)
    parser.add_argument("--stress_admission_wave_size", type=int, default=None)
    parser.add_argument("--stress_wave_decode_gap_steps", type=int, default=None)
    parser.add_argument("--stress_v2_workloads", default=None)
    parser.add_argument("--stress_v2_seed", type=int, default=None)
    parser.add_argument("--stress_v2_sessions", type=int, default=None)
    parser.add_argument("--stress_v2_turns", type=int, default=None)
    parser.add_argument("--stress_v2_system_prompt_len", type=int, default=None)
    parser.add_argument("--stress_v2_session_prefix_len", type=int, default=None)
    parser.add_argument("--stress_v2_session_prefix_min_len", type=int, default=None)
    parser.add_argument("--stress_v2_user_len", type=int, default=None)
    parser.add_argument("--stress_v2_user_min_len", type=int, default=None)
    parser.add_argument("--stress_v2_output_len", type=int, default=None)
    parser.add_argument("--stress_v2_shared_prompts", type=int, default=None)
    parser.add_argument("--stress_v2_shared_prefix_len", type=int, default=None)
    parser.add_argument("--stress_v2_shared_suffix_len", type=int, default=None)
    parser.add_argument("--stress_v2_shared_suffix_min_len", type=int, default=None)
    parser.add_argument("--stress_v2_max_active_requests", type=int, default=None)
    parser.add_argument("--stress_v2_case_timeout_s", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.layer == "agent_trace":
        from benchmark.sparseengine_regression.agent_trace import run_replay
        return run_replay(args)
    manifest = load_manifest(args.manifest)
    resolved = resolve_manifest_paths(manifest)
    quality_benchmarks = set(_parse_csv(args.quality_benchmarks))
    unknown_quality_benchmarks = sorted(
        quality_benchmarks - {"longbench", "longbench_v2", "ruler"}
    )
    if unknown_quality_benchmarks or not quality_benchmarks:
        raise ValueError(
            "--quality_benchmarks must select longbench, longbench_v2, and/or ruler; "
            f"got {sorted(quality_benchmarks)}."
        )
    quality_overrides: dict[str, Any] = {}
    if args.quality_tasks is not None:
        quality_overrides["tasks"] = _parse_csv(args.quality_tasks)
    for arg_name, cfg_key in (
        ("quality_batch_size", "batch_size"),
        ("quality_samples_per_task", "samples_per_task"),
        ("quality_min_required_samples", "min_required_samples"),
        ("quality_min_prompt_tokens", "min_prompt_tokens"),
        ("quality_sparseengine_max_num_seqs_in_batch", "sparseengine_max_num_seqs_in_batch"),
        ("quality_sparseengine_max_decoding_seqs", "sparseengine_max_decoding_seqs"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            quality_overrides[cfg_key] = int(value)
    if quality_overrides:
        quality_cfg = dict(resolved.get("quality") or {})
        quality_cfg.update(quality_overrides)
        if not quality_cfg.get("tasks"):
            raise ValueError("--quality_tasks must include at least one task.")
        for key in (
            "batch_size",
            "samples_per_task",
            "min_required_samples",
            "sparseengine_max_num_seqs_in_batch",
            "sparseengine_max_decoding_seqs",
        ):
            if key in quality_cfg and int(quality_cfg[key]) <= 0:
                raise ValueError(f"quality {key} must be > 0, got {quality_cfg[key]}.")
        resolved["quality"] = quality_cfg

    longbench_v2_overrides: dict[str, Any] = {}
    if args.longbench_v2_data_path is not None:
        longbench_v2_overrides["data_path"] = args.longbench_v2_data_path
    if args.longbench_v2_max_model_len is not None:
        longbench_v2_overrides["max_model_len"] = args.longbench_v2_max_model_len
    if args.longbench_v2_batch_size is not None:
        longbench_v2_overrides["batch_size"] = args.longbench_v2_batch_size
    if args.longbench_v2_token_buckets_json is not None:
        try:
            token_buckets_override = json.loads(args.longbench_v2_token_buckets_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"--longbench_v2_token_buckets_json is invalid JSON: {exc}"
            ) from exc
        longbench_v2_overrides["token_buckets"] = token_buckets_override
    if longbench_v2_overrides:
        longbench_v2_cfg = dict(resolved.get("longbench_v2") or {})
        longbench_v2_cfg.update(longbench_v2_overrides)
        from benchmark.long_bench_v2.contracts import parse_token_buckets

        buckets = parse_token_buckets(longbench_v2_cfg.get("token_buckets"))
        for key in ("max_model_len", "max_new_tokens", "batch_size"):
            if int(longbench_v2_cfg[key]) <= 0:
                raise ValueError(
                    f"longbench_v2 {key} must be > 0, got {longbench_v2_cfg[key]}."
                )
        prompt_budget = int(longbench_v2_cfg["max_model_len"]) - int(
            longbench_v2_cfg["max_new_tokens"]
        )
        oversized = [
            bucket.name
            for bucket in buckets
            if bucket.max_prompt_tokens > prompt_budget
        ]
        if oversized:
            raise ValueError(
                "LongBench v2 override token buckets exceed the prompt budget: "
                f"{oversized}."
            )
        resolved["longbench_v2"] = longbench_v2_cfg

    ruler_overrides: dict[str, Any] = {}
    if args.ruler_tasks is not None:
        ruler_overrides["tasks"] = _parse_csv(args.ruler_tasks)
    if args.ruler_context_lengths is not None:
        ruler_overrides["context_lengths"] = _parse_int_csv(
            args.ruler_context_lengths
        )
    for arg_name, cfg_key in (
        ("ruler_samples_per_length", "samples_per_length"),
        ("ruler_batch_size", "batch_size"),
        ("ruler_worker_world_size", "worker_world_size"),
        ("ruler_max_model_len", "max_model_len"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            ruler_overrides[cfg_key] = int(value)
    if ruler_overrides:
        ruler_cfg = dict(resolved.get("ruler") or {})
        ruler_cfg.update(ruler_overrides)
        tasks = ruler_cfg.get("tasks") or []
        unsupported_tasks = sorted(set(tasks) - set(SUPPORTED_TASKS))
        if not tasks or unsupported_tasks:
            raise ValueError(
                "--ruler_tasks must select supported tasks; "
                f"supported={SUPPORTED_TASKS} got={tasks}."
            )
        missing_configs = sorted(
            set(tasks) - set(ruler_cfg.get("task_configs") or {})
        )
        if missing_configs:
            raise ValueError(
                f"RULER selected tasks are missing task_configs: {missing_configs}."
            )
        if len(set(ruler_cfg["context_lengths"])) != len(
            ruler_cfg["context_lengths"]
        ):
            raise ValueError(
                f"ruler context_lengths must be unique, got {ruler_cfg['context_lengths']}."
            )
        for key in (
            "samples_per_length",
            "batch_size",
            "worker_world_size",
            "max_model_len",
        ):
            if key in ruler_cfg and int(ruler_cfg[key]) <= 0:
                raise ValueError(f"ruler {key} must be > 0, got {ruler_cfg[key]}.")
        if any(int(value) <= 0 for value in ruler_cfg["context_lengths"]):
            raise ValueError(
                "ruler context_lengths must contain only positive integers."
            )
        resolved["ruler"] = ruler_cfg

    scbench_overrides: dict[str, Any] = {}
    if args.scbench_tasks is not None:
        scbench_overrides["tasks"] = _parse_csv(args.scbench_tasks)
    for arg_name, cfg_key in (
        ("scbench_num_eval_examples", "num_eval_examples"),
        ("scbench_max_turns", "max_turns"),
        ("scbench_max_seq_length", "max_seq_length"),
        ("scbench_batch_size", "batch_size"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            scbench_overrides[cfg_key] = int(value)
    if scbench_overrides:
        scbench_cfg = dict(resolved.get("scbench") or {})
        scbench_cfg.update(scbench_overrides)
        if not scbench_cfg.get("tasks"):
            raise ValueError("--scbench_tasks must include at least one task.")
        for key in ("num_eval_examples", "max_turns", "max_seq_length", "batch_size"):
            if key in scbench_cfg and int(scbench_cfg[key]) <= 0:
                raise ValueError(f"scbench {key} must be > 0, got {scbench_cfg[key]}.")
        resolved["scbench"] = scbench_cfg

    stress_overrides: dict[str, Any] = {}
    if args.stress_request_counts is not None:
        stress_overrides["request_counts"] = _parse_int_csv(args.stress_request_counts)
    for arg_name, cfg_key in (
        ("stress_length", "length"),
        ("stress_output_len", "output_len"),
        ("stress_max_num_seqs_in_batch", "max_num_seqs_in_batch"),
        ("stress_max_decoding_seqs", "max_decoding_seqs"),
        ("stress_max_decode_steps_after_full", "max_decode_steps_after_full"),
        ("stress_admission_wave_size", "admission_wave_size"),
        ("stress_wave_decode_gap_steps", "wave_decode_gap_steps"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            stress_overrides[cfg_key] = int(value)
    if stress_overrides:
        stress_cfg = dict(resolved.get("stress") or {})
        stress_cfg.update(stress_overrides)
        if not stress_cfg.get("request_counts"):
            raise ValueError("stress request_counts must include at least one request count.")
        for key in (
            "length",
            "output_len",
            "max_num_seqs_in_batch",
            "max_decoding_seqs",
            "max_decode_steps_after_full",
        ):
            if key in stress_cfg and int(stress_cfg[key]) <= 0:
                raise ValueError(f"stress {key} must be > 0, got {stress_cfg[key]}.")
        if any(int(value) <= 0 for value in stress_cfg["request_counts"]):
            raise ValueError(f"stress request_counts must be > 0, got {stress_cfg['request_counts']}.")
        resolved["stress"] = stress_cfg

    stress_v2_overrides: dict[str, Any] = {}
    if args.stress_v2_workloads is not None:
        stress_v2_overrides["workloads"] = str(args.stress_v2_workloads)
    for arg_name, cfg_key in (
        ("stress_v2_seed", "seed"),
        ("stress_v2_sessions", "sessions"),
        ("stress_v2_turns", "turns"),
        ("stress_v2_system_prompt_len", "system_prompt_len"),
        ("stress_v2_session_prefix_len", "session_prefix_len"),
        ("stress_v2_session_prefix_min_len", "session_prefix_min_len"),
        ("stress_v2_user_len", "user_len"),
        ("stress_v2_user_min_len", "user_min_len"),
        ("stress_v2_output_len", "output_len"),
        ("stress_v2_shared_prompts", "shared_prompts"),
        ("stress_v2_shared_prefix_len", "shared_prefix_len"),
        ("stress_v2_shared_suffix_len", "shared_suffix_len"),
        ("stress_v2_shared_suffix_min_len", "shared_suffix_min_len"),
        ("stress_v2_max_active_requests", "max_active_requests"),
        ("stress_v2_case_timeout_s", "case_timeout_s"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            stress_v2_overrides[cfg_key] = value
    if stress_v2_overrides:
        stress_v2_cfg = dict(resolved.get("stress_v2") or {})
        stress_v2_cfg.update(stress_v2_overrides)
        for key in (
            "seed",
            "sessions",
            "turns",
            "system_prompt_len",
            "session_prefix_len",
            "user_len",
            "output_len",
            "shared_prompts",
            "shared_prefix_len",
            "shared_suffix_len",
            "max_active_requests",
        ):
            if key in stress_v2_cfg and int(stress_v2_cfg[key]) <= 0:
                raise ValueError(f"stress_v2 {key} must be > 0, got {stress_v2_cfg[key]}.")
        for key in ("session_prefix_min_len", "user_min_len", "shared_suffix_min_len"):
            if key in stress_v2_cfg and stress_v2_cfg[key] is not None and int(stress_v2_cfg[key]) < 0:
                raise ValueError(f"stress_v2 {key} must be >= 0, got {stress_v2_cfg[key]}.")
        resolved["stress_v2"] = stress_v2_cfg

    if args.scbench_decode_graph:
        scbench_cfg = dict(resolved.get("scbench") or {})
        scbench_cfg["decode_graph"] = True
        resolved["scbench"] = scbench_cfg
    if args.enable_prefix_caching:
        for section in ("quality", "longbench_v2", "ruler", "performance", "stress"):
            section_cfg = dict(resolved.get(section) or {})
            section_cfg["enable_prefix_caching"] = True
            if args.prefix_cache_block_size is not None:
                section_cfg["prefix_cache_block_size"] = int(args.prefix_cache_block_size)
            if args.prefix_cache_salt is not None:
                section_cfg["prefix_cache_salt"] = str(args.prefix_cache_salt)
            if section == "stress":
                section_cfg["require_prefix_cache_hit"] = True
            resolved[section] = section_cfg
        scbench_cfg = dict(resolved.get("scbench") or {})
        if args.prefix_cache_block_size is not None:
            scbench_cfg["prefix_cache_block_size"] = int(args.prefix_cache_block_size)
        resolved["scbench"] = scbench_cfg
    elif args.require_prefix_cache_hit:
        stress_cfg = dict(resolved.get("stress") or {})
        stress_cfg["require_prefix_cache_hit"] = True
        resolved["stress"] = stress_cfg
    if args.enable_profiler:
        for section in ("quality", "longbench_v2", "ruler", "performance", "stress"):
            section_cfg = dict(resolved.get(section) or {})
            section_cfg["enable_profiler"] = True
            resolved[section] = section_cfg
    if args.tensor_parallel_size is not None:
        if int(args.tensor_parallel_size) <= 0:
            raise ValueError(f"--tensor_parallel_size must be > 0, got {args.tensor_parallel_size}.")
        resolved.setdefault("performance", {})["tensor_parallel_size"] = int(args.tensor_parallel_size)
        scbench_cfg = dict(resolved.get("scbench") or {})
        scbench_cfg["tensor_parallel_size"] = int(args.tensor_parallel_size)
        resolved["scbench"] = scbench_cfg
    model_ids, method_ids = select_entries(
        resolved,
        [item for item in (args.models or "").split(",") if item] or None,
        [item for item in (args.methods or "").split(",") if item] or None,
    )

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root or DEFAULT_OUTPUT_ROOT) / "sparseengine_regression" / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "resolved_manifest.json", resolved)
    for jsonl_name in ("raw_outputs.jsonl", "parsed_outputs.jsonl", "sample_results.jsonl", "perf.jsonl"):
        (output_root / jsonl_name).write_text("", encoding="utf-8")

    summary: dict[str, Any] = {
        "status": "running",
        "run_id": run_id,
        "layer": args.layer,
        "host": socket.gethostname(),
        "cwd": os.getcwd(),
        "git_commit": _git_commit(),
        "git_status_short": _git_status_short(),
        "models": model_ids,
        "methods": method_ids,
        "tensor_parallel_size": _tensor_parallel_size_from_config(resolved.get("performance")),
        "command_timeout_s": float(args.command_timeout_s) if args.command_timeout_s else None,
        "dry_run": bool(args.dry_run),
        "grades": [],
        "commands": [],
        "skipped": [],
    }
    metrics_records: list[dict[str, Any]] = []
    memory_records: list[dict[str, Any]] = []
    stress_records: list[dict[str, Any]] = []
    stress_v2_records: list[dict[str, Any]] = []
    scbench_records: list[dict[str, Any]] = []
    ruler_records: list[dict[str, Any]] = []
    longbench_v2_records: list[dict[str, Any]] = []

    cwd = Path.cwd()
    try:
        if args.layer == "validate":
            summary["status"] = "completed"
            _write_json(output_root / "metrics.json", {"records": metrics_records})
            _write_json(output_root / "memory.json", {"records": memory_records})
            _write_json(output_root / "stress.json", {"records": stress_records})
            _write_json(output_root / "stress_v2.json", {"records": stress_v2_records})
            _write_json(output_root / "scbench.json", {"records": scbench_records})
            _write_json(output_root / "ruler.json", {"records": []})
            _write_json(output_root / "longbench_v2.json", {"records": []})
            _write_json(output_root / "grade_summary.json", summary)
            _ensure_artifacts(output_root, list(resolved["outputs"]))
            print(f"[validate] manifest ok: {output_root}")
            return 0

        selected_pairs: list[tuple[str, str]] = []
        runtime_tp_sizes = _runtime_tensor_parallel_sizes(args.layer, resolved)
        for model_id in model_ids:
            for method_id in method_ids:
                unsupported_reason = runtime_support_reason(
                    resolved,
                    model_id,
                    method_id,
                    tensor_parallel_sizes=runtime_tp_sizes,
                )
                if unsupported_reason is not None:
                    record = {
                        "model": model_id,
                        "method": method_id,
                        "status": "skipped_by_policy",
                        "reason": unsupported_reason,
                    }
                    summary["skipped"].append(record)
                    if not args.allow_skipped_policy:
                        raise RuntimeError(
                            f"Unsupported runtime pair {model_id}/{method_id}: {unsupported_reason}"
                        )
                    continue
                missing = missing_runtime_inputs(resolved, model_id, method_id)
                if missing:
                    record = {
                        "model": model_id,
                        "method": method_id,
                        "status": "skipped_by_policy",
                        "missing": missing,
                    }
                    summary["skipped"].append(record)
                    if not args.allow_skipped_policy:
                        raise FileNotFoundError(f"Missing runtime inputs for {model_id}/{method_id}: {missing}")
                    continue
                selected_pairs.append((model_id, method_id))

        run_quality_layer = args.layer in {"quality", "nightly", "pre-refactor"}
        run_longbench_quality = run_quality_layer and "longbench" in quality_benchmarks
        run_longbench_v2_quality = (
            args.layer == "longbench_v2"
            or (run_quality_layer and "longbench_v2" in quality_benchmarks)
        )
        run_ruler_quality = (
            args.layer == "ruler"
            or (run_quality_layer and "ruler" in quality_benchmarks)
        )
        run_perf = args.layer in {"perf", "nightly", "pre-refactor"}
        run_stress = args.layer in {"stress", "pre-refactor"}
        run_stress_v2 = args.layer == "stress_v2"
        run_scbench = args.layer == "scbench"

        quality_roots: dict[tuple[str, str], Path] = {}
        if run_longbench_quality:
            for model_id, method_id in selected_pairs:
                model = resolved["models"][model_id]
                method = resolved["methods"][method_id]
                out_dir = output_root / "quality" / model_id / method_id
                cmd = _quality_command(
                    model_id=model_id,
                    method_id=method_id,
                    model=model,
                    method=method,
                    quality=resolved["quality"],
                    performance=resolved["performance"],
                    output_root=out_dir,
                )
                _run_and_record(
                    summary,
                    cmd,
                    cwd=cwd,
                    dry_run=args.dry_run,
                    log_path=out_dir / "run.log",
                    timeout_s=args.command_timeout_s,
                )
                if args.dry_run:
                    continue
                quality_roots[(model_id, method_id)] = out_dir
                _append_jsonl_file(
                    output_root / "raw_outputs.jsonl",
                    out_dir / "raw_outputs.jsonl",
                    {"model": model_id, "method": method_id},
                )
                _append_jsonl_file(
                    output_root / "parsed_outputs.jsonl",
                    out_dir / "parsed_outputs.jsonl",
                    {"model": model_id, "method": method_id},
                )
                _append_jsonl_file(
                    output_root / "sample_results.jsonl",
                    out_dir / "sample_results.jsonl",
                    {"model": model_id, "method": method_id},
                )
                result = _load_result_json(out_dir)
                if result is not None:
                    metrics_records.append(
                        {
                            "benchmark": "longbench",
                            "model": model_id,
                            "method": method_id,
                            "result": result,
                        }
                    )

            for model_id in model_ids:
                vanilla_root = quality_roots.get((model_id, "vanilla"))
                if vanilla_root is None:
                    continue
                for method_id in method_ids:
                    if method_id == "vanilla" or (model_id, method_id) not in quality_roots:
                        continue
                    grade = _grade_quality_pair(
                        vanilla_root,
                        quality_roots[(model_id, method_id)],
                        minimum_vanilla_score=float(resolved["quality"]["minimum_vanilla_score"]),
                    )
                    summary["grades"].append({**grade.to_dict(), "model": model_id, "method": method_id})

        longbench_v2_roots: dict[tuple[str, str], Path] = {}
        if run_longbench_v2_quality:
            selected_pair_set = set(selected_pairs)
            for model_id in model_ids:
                sparse_methods = [
                    method_id
                    for pair_model_id, method_id in selected_pairs
                    if pair_model_id == model_id and method_id != "vanilla"
                ]
                if sparse_methods and (model_id, "vanilla") not in selected_pair_set:
                    raise ValueError(
                        "LongBench v2 sparse quality requires a vanilla baseline in the "
                        f"same run: model={model_id} sparse_methods={sparse_methods}."
                    )
            for model_id, method_id in selected_pairs:
                model = resolved["models"][model_id]
                method = resolved["methods"][method_id]
                out_dir = output_root / "longbench_v2" / model_id / method_id
                cmd = _longbench_v2_command(
                    model_id=model_id,
                    method_id=method_id,
                    model=model,
                    method=method,
                    longbench_v2=resolved["longbench_v2"],
                    performance=resolved["performance"],
                    output_root=out_dir,
                )
                _run_and_record(
                    summary,
                    cmd,
                    cwd=cwd,
                    dry_run=args.dry_run,
                    log_path=out_dir / "run.log",
                    timeout_s=args.command_timeout_s,
                )
                if args.dry_run:
                    continue
                longbench_v2_roots[(model_id, method_id)] = out_dir
                artifact_metadata = {
                    "benchmark": "longbench_v2",
                    "model": model_id,
                    "method": method_id,
                }
                _append_jsonl_file(
                    output_root / "raw_outputs.jsonl",
                    out_dir / "raw_outputs.jsonl",
                    artifact_metadata,
                )
                _append_jsonl_file(
                    output_root / "parsed_outputs.jsonl",
                    out_dir / "parsed_outputs.jsonl",
                    artifact_metadata,
                )
                _append_jsonl_file(
                    output_root / "sample_results.jsonl",
                    out_dir / "sample_results.jsonl",
                    artifact_metadata,
                )
                aggregate = _read_json(out_dir / "aggregate_metrics.json")
                record = {
                    "model": model_id,
                    "method": method_id,
                    "result": aggregate,
                }
                longbench_v2_records.append(record)
                metrics_records.append({"benchmark": "longbench_v2", **record})

            for model_id in model_ids:
                vanilla_root = longbench_v2_roots.get((model_id, "vanilla"))
                if vanilla_root is None:
                    continue
                for method_id in method_ids:
                    sparse_root = longbench_v2_roots.get((model_id, method_id))
                    if method_id == "vanilla" or sparse_root is None:
                        continue
                    grade = _grade_longbench_v2_pair(
                        vanilla_root,
                        sparse_root,
                        longbench_v2=resolved["longbench_v2"],
                    )
                    summary["grades"].append(
                        {
                            **grade.to_dict(),
                            "model": model_id,
                            "method": method_id,
                        }
                    )

        ruler_roots: dict[tuple[str, str, str], Path] = {}
        if run_ruler_quality:
            selected_pair_set = set(selected_pairs)
            for model_id in model_ids:
                sparse_methods = [
                    method_id
                    for pair_model_id, method_id in selected_pairs
                    if pair_model_id == model_id and method_id != "vanilla"
                ]
                if sparse_methods and (model_id, "vanilla") not in selected_pair_set:
                    raise ValueError(
                        "RULER sparse quality requires a vanilla baseline in the same run: "
                        f"model={model_id} sparse_methods={sparse_methods}."
                    )
            for task in resolved["ruler"]["tasks"]:
                task_config = resolved["ruler"]["task_configs"][task]
                for model_id, method_id in selected_pairs:
                    model = resolved["models"][model_id]
                    method = resolved["methods"][method_id]
                    out_dir = output_root / "ruler" / task / model_id / method_id
                    cmd = _ruler_command(
                        task=task,
                        task_config=task_config,
                        model_id=model_id,
                        method_id=method_id,
                        model=model,
                        method=method,
                        ruler=resolved["ruler"],
                        performance=resolved["performance"],
                        output_root=out_dir,
                    )
                    _run_and_record(
                        summary,
                        cmd,
                        cwd=cwd,
                        dry_run=args.dry_run,
                        log_path=out_dir / "run.log",
                        timeout_s=args.command_timeout_s,
                    )
                    if args.dry_run:
                        continue
                    ruler_roots[(task, model_id, method_id)] = out_dir
                    artifact_metadata = {
                        "benchmark": "ruler",
                        "task": task,
                        "model": model_id,
                        "method": method_id,
                        "evaluation_pass": "primary",
                    }
                    _append_jsonl_file(
                        output_root / "raw_outputs.jsonl",
                        out_dir / "raw_outputs.jsonl",
                        artifact_metadata,
                    )
                    _append_jsonl_file(
                        output_root / "parsed_outputs.jsonl",
                        out_dir / "parsed_outputs.jsonl",
                        artifact_metadata,
                    )
                    _append_jsonl_file(
                        output_root / "sample_results.jsonl",
                        out_dir / "per_sample_results.jsonl",
                        artifact_metadata,
                    )
                    for artifact_name, destination in (
                        ("raw_outputs_prefix_cache_replay.jsonl", "raw_outputs.jsonl"),
                        ("parsed_outputs_prefix_cache_replay.jsonl", "parsed_outputs.jsonl"),
                        ("per_sample_results_prefix_cache_replay.jsonl", "sample_results.jsonl"),
                    ):
                        _append_jsonl_file(
                            output_root / destination,
                            out_dir / artifact_name,
                            {
                                **artifact_metadata,
                                "evaluation_pass": "prefix_cache_replay",
                            },
                        )
                    aggregate = _read_json(out_dir / "aggregate_metrics.json")
                    record = {
                        "task": task,
                        "model": model_id,
                        "method": method_id,
                        "result": aggregate,
                    }
                    ruler_records.append(record)
                    metrics_records.append({"benchmark": "ruler", **record})

            for task in resolved["ruler"]["tasks"]:
                for model_id in model_ids:
                    vanilla_root = ruler_roots.get((task, model_id, "vanilla"))
                    if vanilla_root is None:
                        continue
                    for method_id in method_ids:
                        sparse_root = ruler_roots.get((task, model_id, method_id))
                        if method_id == "vanilla" or sparse_root is None:
                            continue
                        for context_length, grade in _grade_ruler_pair(
                            vanilla_root,
                            sparse_root,
                            ruler=resolved["ruler"],
                            task=task,
                        ):
                            summary["grades"].append(
                                {
                                    **grade.to_dict(),
                                    "task": task,
                                    "model": model_id,
                                    "method": method_id,
                                    "context_length": context_length,
                                }
                            )

        if run_perf:
            for model_id in model_ids:
                method_ids_for_model = [
                    method_id
                    for pair_model_id, method_id in selected_pairs
                    if pair_model_id == model_id
                ]
                if not method_ids_for_model:
                    continue
                for method_id in method_ids_for_model:
                    out_path = output_root / "perf" / model_id / f"{method_id}.jsonl"
                    cmd = _perf_command(
                        model_id=model_id,
                        model=resolved["models"][model_id],
                        method_id=method_id,
                        method=resolved["methods"][method_id],
                        performance=resolved["performance"],
                        output_jsonl=out_path,
                    )
                    _run_and_record(
                        summary,
                        cmd,
                        cwd=cwd,
                        dry_run=args.dry_run,
                        log_path=output_root / "perf" / model_id / f"{method_id}.log",
                        timeout_s=args.command_timeout_s,
                    )
                    if args.dry_run:
                        grade = GateGrade("performance", "N/A", "skipped_by_policy", {}, "dry run")
                        summary["grades"].append(
                            {
                                **grade.to_dict(),
                                "model": model_id,
                                "method": method_id,
                            }
                        )
                        continue
                    rows = _read_jsonl(out_path)
                    for row in rows:
                        _append_jsonl(output_root / "perf.jsonl", {"model": model_id, **row})
                    expected_methods = (
                        ("vanilla",)
                        if method_id == "vanilla"
                        else ("vanilla", method_id)
                    )
                    _require_successful_perf_matrix(
                        rows,
                        methods=expected_methods,
                        lengths=resolved["performance"]["lengths"],
                        batch_sizes=resolved["performance"]["batch_sizes"],
                        artifact=out_path,
                    )
                    _require_synchronized_step_timing(rows, artifact=out_path)
                    vanilla_by_shape = {
                        (row["length"], row["batch_size"]): row
                        for row in rows
                        if row.get("method") == "vanilla" and row.get("status") == "SUCCESS"
                    }
                    for row in rows:
                        if row.get("method") == "vanilla" or row.get("status") != "SUCCESS":
                            continue
                        vanilla = vanilla_by_shape.get((row["length"], row["batch_size"]))
                        if not vanilla:
                            continue
                        vanilla_decode_tp = float(vanilla["decode_tp"])
                        vanilla_prefill_tp = float(vanilla["prefill_tp"])
                        if vanilla_decode_tp <= 0.0 or vanilla_prefill_tp <= 0.0:
                            raise RuntimeError(
                                "Vanilla performance baseline must report positive throughput "
                                f"for length={row['length']} batch_size={row['batch_size']}; "
                                f"decode_tp={vanilla_decode_tp} prefill_tp={vanilla_prefill_tp}."
                            )
                        decode_speedup = float(row["decode_tp"]) / vanilla_decode_tp
                        prefill_speedup = float(row["prefill_tp"]) / vanilla_prefill_tp
                        tensor_parallel_size = _tensor_parallel_size_from_config(resolved.get("performance"))
                        method_performance = (
                            resolved["methods"].get(row["method"], {}).get("performance") or {}
                        )
                        grade = grade_perf(
                            decode_speedup,
                            graph_expected=bool(row.get("decode_graph_expected")),
                            graph_active=bool(row.get("decode_graph_active")),
                            require_speedup=tensor_parallel_size <= 1,
                            prefill_speedup=prefill_speedup,
                            minimum_prefill_speedup=method_performance.get(
                                "minimum_prefill_speedup"
                            ),
                        )
                        summary["grades"].append(
                            {
                                **grade.to_dict(),
                                "model": model_id,
                                "method": row["method"],
                                "length": row["length"],
                                "batch_size": row["batch_size"],
                            }
                        )
                        accounting = row.get("memory_accounting") or {}
                        expected = resolved["methods"].get(row["method"], {}).get("memory", {}).get("expected_savings")
                        observed = accounting.get("observed_savings")
                        mem_grade = grade_memory(expected_savings=expected, observed_savings=observed)
                        memory_record = {
                            "model": model_id,
                            "method": row["method"],
                            "length": row["length"],
                            "batch_size": row["batch_size"],
                            "memory_accounting": accounting,
                            "grade": mem_grade.to_dict(),
                        }
                        memory_records.append(memory_record)
                        summary["grades"].append(
                            {
                                **mem_grade.to_dict(),
                                "model": model_id,
                                "method": row["method"],
                                "length": row["length"],
                                "batch_size": row["batch_size"],
                            }
                        )

        if run_stress:
            for model_id, method_id in selected_pairs:
                out_path = output_root / "stress" / model_id / f"{method_id}.jsonl"
                cmd = _stress_command(
                    model_id=model_id,
                    model=resolved["models"][model_id],
                    method_id=method_id,
                    method=resolved["methods"][method_id],
                    performance=resolved["performance"],
                    stress=resolved["stress"],
                    output_jsonl=out_path,
                )
                _run_and_record(
                    summary,
                    cmd,
                    cwd=cwd,
                    dry_run=args.dry_run,
                    log_path=output_root / "stress" / model_id / f"{method_id}.log",
                    timeout_s=args.command_timeout_s,
                )
                rows = _read_jsonl(out_path)
                _require_synchronized_step_timing(rows, artifact=out_path)
                if args.dry_run:
                    grade = GateGrade("stress", "N/A", "skipped_by_policy", {}, "dry run")
                    summary["grades"].append({**grade.to_dict(), "model": model_id, "method": method_id})
                    continue
                if not rows:
                    grade = grade_stress(
                        completed=False,
                        crashed=True,
                        preemptions=0,
                        full_admission_window=False,
                        utilization_ok=False,
                    )
                    stress_records.append({"model": model_id, "method": method_id, "rows": [], "grade": grade.to_dict()})
                    summary["grades"].append({**grade.to_dict(), "model": model_id, "method": method_id})
                    continue
                for row in rows:
                    if row.get("status") == "SKIPPED_BY_POLICY":
                        grade = GateGrade(
                            "stress",
                            "N/A",
                            "skipped_by_policy",
                            row,
                            str(row.get("reason") or "stress case skipped by policy"),
                        )
                    else:
                        grade = grade_stress(
                            completed=row.get("status") == "SUCCESS",
                            crashed=row.get("status") != "SUCCESS",
                            preemptions=int(row.get("scheduler_preemptions", 0) or 0),
                            full_admission_window=bool(row.get("full_admission_reached")),
                            utilization_ok=bool(row.get("utilization_ok", False)),
                        )
                    stress_record = {
                        "model": model_id,
                        "method": method_id,
                        "length": row.get("length"),
                        "batch_size": row.get("batch_size"),
                        "row": row,
                        "grade": grade.to_dict(),
                    }
                    stress_records.append(stress_record)
                    summary["grades"].append(
                        {
                            **grade.to_dict(),
                            "model": model_id,
	                            "method": method_id,
	                            "length": row.get("length"),
	                            "batch_size": row.get("batch_size"),
	                        }
	                    )

        if run_stress_v2:
            for model_id, method_id in selected_pairs:
                method = resolved["methods"][method_id]
                cases = _stress_v2_cases(method_id, method)
                if not cases:
                    grade = GateGrade(
                        "stress_v2",
                        "N/A",
                        "skipped_by_policy",
                        {"method": method_id},
                        "stress_v2 only supports prefix-cache serving traces for vanilla, omnikv, and quest.",
                    )
                    stress_v2_records.append(
                        {"model": model_id, "method": method_id, "status": "skipped_by_policy", "grade": grade.to_dict()}
                    )
                    summary["grades"].append({**grade.to_dict(), "model": model_id, "method": method_id})
                    continue
                out_dir = output_root / "stress_v2" / model_id / method_id
                cmd = _stress_v2_command(
                    model_id=model_id,
                    model=resolved["models"][model_id],
                    method_id=method_id,
                    method=method,
                    stress_v2=resolved["stress_v2"],
                    output_dir=out_dir,
                )
                _run_and_record(
                    summary,
                    cmd,
                    cwd=cwd,
                    dry_run=args.dry_run,
                    log_path=out_dir / "run.log",
                    timeout_s=args.command_timeout_s,
                )
                if args.dry_run:
                    grade = GateGrade("stress_v2", "N/A", "skipped_by_policy", {}, "dry run")
                    summary["grades"].append({**grade.to_dict(), "model": model_id, "method": method_id})
                    continue
                aggregate = _read_json(out_dir / "aggregate_metrics.json")
                grade = grade_stress_v2(aggregate)
                for row in _read_jsonl(out_dir / "performance.jsonl"):
                    _append_jsonl(output_root / "perf.jsonl", {"model": model_id, "method": method_id, "stress_v2": True, **row})
                for case_dir in sorted(path for path in out_dir.iterdir() if path.is_dir()):
                    _append_jsonl_file(
                        output_root / "raw_outputs.jsonl",
                        case_dir / "raw_outputs.jsonl",
                        {"model": model_id, "method": method_id, "stress_v2_case": case_dir.name},
                    )
                    _append_jsonl_file(
                        output_root / "sample_results.jsonl",
                        case_dir / "per_turn_results.jsonl",
                        {"model": model_id, "method": method_id, "stress_v2_case": case_dir.name},
                    )
                stress_v2_records.append(
                    {"model": model_id, "method": method_id, "cases": cases, "summary": aggregate, "grade": grade.to_dict()}
                )
                summary["grades"].append({**grade.to_dict(), "model": model_id, "method": method_id})

        if run_scbench:
            scbench = resolved["scbench"]
            scbench_model_id = str(scbench["model"])
            selected_pair_set = set(selected_pairs)
            method_ids_for_scbench = [
                method_id
                for method_id in scbench["methods"]
                if method_id in method_ids and (scbench_model_id, method_id) in selected_pair_set
            ]
            if scbench_model_id not in model_ids or not method_ids_for_scbench:
                summary["skipped"].append(
                    {
                        "model": scbench_model_id,
                        "methods": method_ids_for_scbench,
                        "status": "skipped_by_policy",
                        "reason": "SCBench configured model/methods are not selected or lack runtime inputs.",
                    }
                )
            else:
                out_dir = output_root / "scbench" / scbench_model_id
                manifest_path = Path(args.manifest) if args.manifest else Path(__file__).with_name("manifest.json")
                cmd = _scbench_command(
                    manifest_path=manifest_path,
                    model_id=scbench_model_id,
                    method_ids=method_ids_for_scbench,
                    scbench=scbench,
                    output_dir=out_dir,
                )
                _run_and_record(
                    summary,
                    cmd,
                    cwd=cwd,
                    dry_run=args.dry_run,
                    log_path=out_dir / "run.log",
                    timeout_s=args.command_timeout_s,
                )
                summary_path = out_dir / "scbench_methods_summary.json"
                if summary_path.exists():
                    with summary_path.open("r", encoding="utf-8") as handle:
                        scbench_summary = json.load(handle)
                    scbench_records.append(scbench_summary)
                    for sample_path in sorted(out_dir.glob("*/*/sample_results_*_multi_turn.jsonl")):
                        _append_jsonl_file(
                            output_root / "sample_results.jsonl",
                            sample_path,
                            {"model": scbench_model_id},
                        )

        grade_objs = [
            GateGrade(item["name"], item["grade"], item["status"], item["metrics"], item.get("reason", ""))
            for item in summary["grades"]
        ]
        summary["worst_required_grade"] = worst_required_grade(grade_objs)
        if summary["worst_required_grade"] == "D":
            failed_gates = [
                "/".join(
                    str(item[key])
                    for key in (
                        "name",
                        "task",
                        "model",
                        "method",
                        "context_length",
                        "length",
                        "batch_size",
                    )
                    if item.get(key) is not None
                )
                for item in summary["grades"]
                if item.get("grade") == "D"
            ]
            raise RuntimeError(
                "Required regression gates failed: "
                + ", ".join(failed_gates)
            )
        summary["status"] = "completed"
        _write_json(output_root / "metrics.json", {"records": metrics_records})
        _write_json(output_root / "memory.json", {"records": memory_records})
        _write_json(output_root / "stress.json", {"records": stress_records})
        _write_json(output_root / "stress_v2.json", {"records": stress_v2_records})
        _write_json(output_root / "scbench.json", {"records": scbench_records})
        _write_json(output_root / "ruler.json", {"records": ruler_records})
        _write_json(
            output_root / "longbench_v2.json",
            {"records": longbench_v2_records},
        )
        _write_json(output_root / "grade_summary.json", summary)
        _ensure_artifacts(output_root, list(resolved["outputs"]))
        print(f"[done] wrote {output_root}")
        return 0
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = repr(exc)
        _write_json(output_root / "metrics.json", {"records": metrics_records})
        _write_json(output_root / "memory.json", {"records": memory_records})
        _write_json(output_root / "stress.json", {"records": stress_records})
        _write_json(output_root / "stress_v2.json", {"records": stress_v2_records})
        _write_json(output_root / "scbench.json", {"records": scbench_records})
        _write_json(output_root / "ruler.json", {"records": ruler_records})
        _write_json(
            output_root / "longbench_v2.json",
            {"records": longbench_v2_records},
        )
        _write_json(output_root / "grade_summary.json", summary)
        _ensure_artifacts(output_root, list(resolved["outputs"]))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
