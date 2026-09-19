#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import queue
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_MODEL_PATH = os.getenv("SPARSEENGINE_MODEL_PATH")
DEFAULT_DELTAKV_CHECKPOINT_PATH = os.getenv("SPARSEENGINE_DELTAKV_CHECKPOINT_PATH")
DEFAULT_OUTPUT_ROOT = os.getenv(
    "SPARSEENGINE_OUTPUT_DIR",
    str(REPO_ROOT / "outputs" / "sparseengine_max_batch_throughput"),
)
DEFAULT_LENGTHS = "64000,128000,256000,512000,900000"
DEFAULT_METHODS = "deltakv-less-memory-cudagraph,omnikv,snapkv,vanilla"
DEFAULT_GPUS = "4,5,6,7"


@dataclass(frozen=True)
class Probe:
    method: str
    length: int
    batch_size: int


def _parse_csv_ints(value: str) -> list[int]:
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("Expected at least one integer.")
    return parsed


def _parse_csv_strings(value: str) -> list[str]:
    parsed = [part.strip() for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("Expected at least one value.")
    return parsed


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _git_output(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(cwd), text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc!r}"


def _base_hparams(args: argparse.Namespace, method: str, batch_size: int) -> dict[str, Any]:
    hparams: dict[str, Any] = {
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "engine_prefill_chunk_size": args.engine_prefill_chunk_size,
        "mlp_chunk_size": args.mlp_chunk_size,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs_in_batch": batch_size,
        "max_decoding_seqs": batch_size,
        "decode_graph": not args.disable_decode_graph,
        "throughput_log_interval_s": 0.0,
    }

    if method == "vanilla":
        return hparams
    if method == "snapkv":
        hparams.update(
            {
                "sink_keep_tokens": 0,
                "recent_keep_tokens": args.recent_keep_tokens,
                "decode_keep_tokens": args.snapkv_decode_keep_tokens,
                "snapkv_window_size": args.snapkv_window_size,
                "pool_kernel_size": args.pool_kernel_size,
            }
        )
        return hparams
    if method == "omnikv":
        hparams.update(
            {
                "full_attention_layers": args.full_attention_layers,
                "sink_keep_tokens": 0,
                "recent_keep_tokens": args.recent_keep_tokens,
                "decode_keep_tokens": args.omnikv_decode_keep_tokens,
                "pool_kernel_size": args.pool_kernel_size,
            }
        )
        return hparams
    if method in {"deltakv", "deltakv-less-memory", "deltakv-less-memory-cudagraph"}:
        hparams.update(
            {
                "full_attention_layers": args.full_attention_layers,
                "sink_keep_tokens": args.deltakv_sink_keep_tokens,
                "recent_keep_tokens": args.deltakv_recent_keep_tokens,
                "decode_keep_tokens": args.deltakv_decode_keep_tokens,
                "deltakv_checkpoint_path": args.deltakv_checkpoint_path,
                "deltakv_latent_dim": args.deltakv_latent_dim,
                "deltakv_center_ratio": args.deltakv_center_ratio,
                "deltakv_neighbor_count": args.deltakv_neighbor_count,
                "deltakv_latent_quant_bits": args.deltakv_latent_quant_bits,
                "deltakv_latent_quant_group_size": args.deltakv_latent_quant_group_size,
                "full_layer_kv_quant_bits": args.full_layer_kv_quant_bits,
                "full_layer_kivi_group_size": args.full_layer_kivi_group_size,
                "full_layer_kivi_residual_length": args.full_layer_kivi_residual_length,
                "enable_full_layer_kivi_quant": not args.disable_full_layer_kivi_quant,
                "enable_sparse_ref_fp8": False,
                "cluster_metric": args.cluster_metric,
                "use_compression": True,
                "pool_kernel_size": args.pool_kernel_size,
                "deltakv_full_pool_reserve_ratio": args.deltakv_full_pool_reserve_ratio,
                "deltakv_cluster_gather_chunk_size": args.deltakv_cluster_gather_chunk_size,
            }
        )
        return hparams
    raise ValueError(f"Unsupported method for max-batch sweep: {method!r}")


def _is_usable(row: dict[str, Any]) -> bool:
    return (
        row.get("status") == "SUCCESS"
        and bool(row.get("full_admission_reached", True))
        and int(row.get("scheduler_preemptions", 0) or 0) == 0
        and (not bool(row.get("decode_graph_expected")) or bool(row.get("decode_graph_active")))
    )


def _run_probe(args: argparse.Namespace, run_root: Path, probe: Probe, gpu: int) -> dict[str, Any]:
    probe_root = run_root / "probes" / probe.method / f"len{probe.length}" / f"bs{probe.batch_size}"
    hparams_path = probe_root / "hparams.json"
    result_path = probe_root / "result.jsonl"
    log_path = probe_root / "run.log"
    cmd_path = probe_root / "cmd.json"

    hparams = _base_hparams(args, probe.method, probe.batch_size)
    _write_json(hparams_path, hparams)

    cmd = [
        sys.executable,
        "-u",
        "benchmark/microbench.py",
        "--model_path",
        args.model_path,
        "--lengths",
        str(probe.length),
        "--batch_sizes",
        str(probe.batch_size),
        "--methods",
        probe.method,
        "--output_len",
        str(args.output_len),
        "--temperature",
        "0.0",
        "--top_p",
        "1.0",
        "--max_decode_steps_after_full",
        str(args.max_decode_steps_after_full),
        "--hyper_params",
        f"@{hparams_path}",
        "--output_jsonl",
        str(result_path),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(args.repo_root), str(args.repo_root / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["SPARSEENGINE_MASTER_PORT"] = str(args.master_port_base + gpu)

    cmd_record = {
        "cmd": cmd,
        "cwd": str(args.repo_root),
        "gpu": gpu,
        "env": {
            "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
            "PYTHONPATH": env["PYTHONPATH"],
            "TOKENIZERS_PARALLELISM": env["TOKENIZERS_PARALLELISM"],
            "PYTORCH_CUDA_ALLOC_CONF": env["PYTORCH_CUDA_ALLOC_CONF"],
            "SPARSEENGINE_MASTER_PORT": env["SPARSEENGINE_MASTER_PORT"],
        },
    }
    _write_json(cmd_path, cmd_record)

    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(args.repo_root),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.probe_timeout_s,
        )
    elapsed = time.time() - start

    rows = _read_jsonl(result_path)
    row = rows[0] if rows else {}
    status = row.get("status") or ("FAILED" if proc.returncode else "NO_RESULT")
    record = {
        "method": probe.method,
        "length": probe.length,
        "batch_size": probe.batch_size,
        "gpu": gpu,
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "status": status,
        "usable": bool(proc.returncode == 0 and rows and _is_usable(row)),
        "probe_root": str(probe_root),
        "hparams": str(hparams_path),
        "result_jsonl": str(result_path),
        "log": str(log_path),
        "cmd_json": str(cmd_path),
        "row": row,
    }
    return record


def _candidate_sequence(max_batch_size: int) -> list[int]:
    values: list[int] = []
    bs = 1
    while bs <= max_batch_size:
        values.append(bs)
        bs *= 2
    if values[-1] != max_batch_size:
        values.append(max_batch_size)
    return sorted(set(values))


def _search_case(args: argparse.Namespace, run_root: Path, method: str, length: int, gpu: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    last_success: dict[str, Any] | None = None
    first_failure: dict[str, Any] | None = None

    for bs in _candidate_sequence(args.max_batch_size):
        record = _run_probe(args, run_root, Probe(method, length, bs), gpu)
        records.append(record)
        if record["usable"]:
            last_success = record
            continue
        first_failure = record
        break

    if last_success is not None and first_failure is not None:
        lo = int(last_success["batch_size"])
        hi = int(first_failure["batch_size"])
        while hi - lo > 1:
            mid = (lo + hi) // 2
            record = _run_probe(args, run_root, Probe(method, length, mid), gpu)
            records.append(record)
            if record["usable"]:
                lo = mid
                last_success = record
            else:
                hi = mid
                first_failure = record

    usable_records = [record for record in records if record["usable"]]
    best_decode = None
    if usable_records:
        best_decode = max(
            usable_records,
            key=lambda record: float(record["row"].get("decode_tp", 0.0) or 0.0),
        )

    if not usable_records:
        case_status = "failed"
    elif first_failure is None and int(last_success["batch_size"]) == args.max_batch_size:
        case_status = "lower_bound"
    else:
        case_status = "success"

    return {
        "method": method,
        "length": length,
        "gpu": gpu,
        "status": case_status,
        "max_success": last_success,
        "first_failure": first_failure,
        "best_decode": best_decode,
        "records": records,
    }


def _summary_row(case: dict[str, Any]) -> dict[str, Any]:
    max_success = case.get("max_success")
    best = case.get("best_decode")
    first_failure = case.get("first_failure")
    row = max_success.get("row") if max_success else {}
    best_row = best.get("row") if best else {}
    return {
        "method": case["method"],
        "length": case["length"],
        "status": case["status"],
        "max_success_bs": max_success.get("batch_size") if max_success else None,
        "decode_tp_at_max_bs": row.get("decode_tp"),
        "prefill_tp_at_max_bs": row.get("prefill_tp"),
        "ttft_at_max_bs": row.get("ttft"),
        "mem_gb_at_max_bs": row.get("mem"),
        "best_decode_bs": best.get("batch_size") if best else None,
        "best_decode_tp": best_row.get("decode_tp"),
        "mem_gb_at_best_decode": best_row.get("mem"),
        "first_failed_bs": first_failure.get("batch_size") if first_failure else None,
        "first_failure_status": first_failure.get("status") if first_failure else None,
        "artifact": max_success.get("probe_root") if max_success else (first_failure or {}).get("probe_root"),
    }


def _format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _write_summary_markdown(run_root: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Sparse-VLLM Max-Batch Throughput",
        "",
        f"Run root: `{run_root}`",
        "",
        "Selection rule: usable rows require `status=SUCCESS`, full admission, zero scheduler preemptions, and active decode CUDA graph when graph mode was requested.",
        "",
        "| Method | Length | Status | Max success BS | Decode tok/s @ max BS | Prefill tok/s @ max BS | Mem GB @ max BS | Best decode BS | Best decode tok/s | First failed BS | Artifact |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["summary_rows"]:
        artifact = row.get("artifact")
        artifact_text = f"`{artifact}`" if artifact else "-"
        lines.append(
            "| {method} | {length} | {status} | {max_bs} | {decode} | {prefill} | {mem} | {best_bs} | {best_decode} | {failed_bs} | {artifact} |".format(
                method=row["method"],
                length=row["length"],
                status=row["status"],
                max_bs=row["max_success_bs"] if row["max_success_bs"] is not None else "-",
                decode=_format_float(row["decode_tp_at_max_bs"], 3),
                prefill=_format_float(row["prefill_tp_at_max_bs"], 1),
                mem=_format_float(row["mem_gb_at_max_bs"], 3),
                best_bs=row["best_decode_bs"] if row["best_decode_bs"] is not None else "-",
                best_decode=_format_float(row["best_decode_tp"], 3),
                failed_bs=row["first_failed_bs"] if row["first_failed_bs"] is not None else "-",
                artifact=artifact_text,
            )
        )
    lines.append("")
    (run_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find max usable Sparse-VLLM batch size per method and context length.")
    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL_PATH,
        required=DEFAULT_MODEL_PATH is None,
    )
    parser.add_argument(
        "--deltakv_checkpoint_path",
        default=DEFAULT_DELTAKV_CHECKPOINT_PATH,
        required=DEFAULT_DELTAKV_CHECKPOINT_PATH is None,
    )
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--lengths", default=DEFAULT_LENGTHS)
    parser.add_argument("--gpus", default=DEFAULT_GPUS)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--max_batch_size", type=int, default=32)
    parser.add_argument("--output_len", type=int, default=128)
    parser.add_argument("--max_decode_steps_after_full", type=int, default=64)
    parser.add_argument("--probe_timeout_s", type=int, default=3600)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--engine_prefill_chunk_size", type=int, default=8192)
    parser.add_argument("--mlp_chunk_size", type=int, default=16384)
    parser.add_argument("--max_num_batched_tokens", type=int, default=65536)
    parser.add_argument("--disable_decode_graph", action="store_true")
    parser.add_argument("--master_port_base", type=int, default=29800)
    parser.add_argument("--full_attention_layers", default="0,2,4,11,16,22")
    parser.add_argument("--recent_keep_tokens", type=int, default=32)
    parser.add_argument("--snapkv_decode_keep_tokens", type=int, default=3072)
    parser.add_argument("--snapkv_window_size", type=int, default=32)
    parser.add_argument("--omnikv_decode_keep_tokens", type=int, default=4096)
    parser.add_argument("--pool_kernel_size", type=int, default=1)
    parser.add_argument("--deltakv_sink_keep_tokens", type=int, default=8)
    parser.add_argument("--deltakv_recent_keep_tokens", type=int, default=128)
    parser.add_argument("--deltakv_decode_keep_tokens", type=int, default=2048)
    parser.add_argument("--deltakv_latent_dim", type=int, default=256)
    parser.add_argument("--deltakv_center_ratio", type=float, default=0.1)
    parser.add_argument("--deltakv_neighbor_count", type=int, default=4)
    parser.add_argument("--deltakv_latent_quant_bits", type=int, default=4)
    parser.add_argument("--deltakv_latent_quant_group_size", type=int, default=32)
    parser.add_argument("--full_layer_kv_quant_bits", type=int, default=4)
    parser.add_argument("--full_layer_kivi_group_size", type=int, default=32)
    parser.add_argument("--full_layer_kivi_residual_length", type=int, default=32)
    parser.add_argument("--disable_full_layer_kivi_quant", action="store_true")
    parser.add_argument("--deltakv_full_pool_reserve_ratio", type=float, default=0.2)
    parser.add_argument("--deltakv_cluster_gather_chunk_size", type=int, default=16384)
    parser.add_argument("--cluster_metric", default="l2")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    args.repo_root = Path.cwd().resolve()
    return args


def main() -> int:
    args = parse_args()
    methods = _parse_csv_strings(args.methods)
    lengths = _parse_csv_ints(args.lengths)
    gpus = _parse_csv_ints(args.gpus)
    if args.max_batch_size < 1:
        raise ValueError("--max_batch_size must be >= 1")
    for required_path in (args.model_path, args.deltakv_checkpoint_path):
        if required_path and not Path(required_path).exists():
            raise FileNotFoundError(required_path)

    run_id = args.run_id or time.strftime("qwen25_7b_maxbs_%Y%m%d_%H%M%S")
    run_root = Path(args.output_root) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    if (run_root / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_root}")

    manifest = {
        "run_id": run_id,
        "host": socket.gethostname(),
        "cwd": str(args.repo_root),
        "git_commit": _git_output(["rev-parse", "HEAD"], args.repo_root),
        "git_status_short": _git_output(["status", "--short"], args.repo_root),
        "python": sys.executable,
        "model_path": args.model_path,
        "deltakv_checkpoint_path": args.deltakv_checkpoint_path,
        "methods": methods,
        "lengths": lengths,
        "gpus": gpus,
        "max_batch_size": args.max_batch_size,
        "output_len": args.output_len,
        "max_decode_steps_after_full": args.max_decode_steps_after_full,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "repo_root"},
    }
    _write_json(run_root / "manifest.json", manifest)

    if args.dry_run:
        print(f"[dry-run] wrote manifest: {run_root / 'manifest.json'}")
        return 0

    cases = [(method, length) for method in methods for length in lengths]
    status_path = run_root / "status.tsv"
    gpu_queue: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    def run_with_gpu(method: str, length: int) -> dict[str, Any]:
        gpu = gpu_queue.get()
        try:
            return _search_case(args, run_root, method, length, gpu)
        finally:
            gpu_queue.put(gpu)

    with status_path.open("w", encoding="utf-8", newline="") as status_handle:
        writer = csv.DictWriter(status_handle, fieldnames=["method", "length", "gpu", "status", "time", "artifact"])
        writer.writeheader()
        status_handle.flush()

        results: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = {}
            for method, length in cases:
                writer.writerow(
                    {
                        "method": method,
                        "length": length,
                        "gpu": "pending",
                        "status": "queued",
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "artifact": str(run_root / "probes" / method / f"len{length}"),
                    }
                )
                status_handle.flush()
                futures[executor.submit(run_with_gpu, method, length)] = (method, length)

            for future in concurrent.futures.as_completed(futures):
                method, length = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "method": method,
                        "length": length,
                        "gpu": None,
                        "status": "runner_failed",
                        "error": repr(exc),
                    }
                results.append(result)
                writer.writerow(
                    {
                        "method": method,
                        "length": length,
                        "gpu": result.get("gpu"),
                        "status": result["status"],
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "artifact": str(run_root / "probes" / method / f"len{length}"),
                    }
                )
                status_handle.flush()
                _write_json(run_root / "partial_results.json", {"cases": results})

    summary_rows = [_summary_row(case) for case in sorted(results, key=lambda row: (row["method"], row["length"]))]
    payload = {"manifest": manifest, "summary_rows": summary_rows, "cases": results}
    _write_json(run_root / "summary.json", payload)
    _write_summary_markdown(run_root, payload)
    print(f"[done] summary: {run_root / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
