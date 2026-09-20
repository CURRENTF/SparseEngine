#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen


PREFIX_METHODS = {"vanilla", "omnikv", "quest", ""}


def method_kwargs(method: str, *, max_model_len: int, prefix_cache: bool) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "sparse_method": method,
        "max_model_len": int(max_model_len),
        "max_num_seqs_in_batch": 16,
        "max_decoding_seqs": 16,
        "gpu_memory_utilization": 0.45,
        "engine_prefill_chunk_size": 1024,
        "throughput_log_interval_s": 0.0,
    }
    if method == "omnikv":
        cfg.update(
            {
                "full_attention_layers": "0,1,3,9,13,16,21,28",
                "sink_keep_tokens": 0,
                "recent_keep_tokens": 32,
                "decode_keep_tokens": 1342,
                "pool_kernel_size": 1,
            }
        )
    elif method == "quest":
        cfg.update(
            {
                "quest_chunk_size": 16,
                "quest_skip_layers": 0,
                "decode_keep_tokens": 2048,
                "sink_keep_tokens": 0,
                "recent_keep_tokens": 32,
            }
        )
    elif method == "snapkv":
        cfg.update(
            {
                "sink_keep_tokens": 0,
                "recent_keep_tokens": 32,
                "decode_keep_tokens": 2048,
                "snapkv_window_size": 32,
                "pool_kernel_size": 7,
            }
        )
    if prefix_cache and method in PREFIX_METHODS:
        cfg.update(
            {
                "enable_prefix_caching": True,
                "prefix_cache_block_size": 16,
                "prefix_cache_salt": f"router-smoke:{method}",
            }
        )
    return cfg


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def request_json(url: str, payload: dict[str, Any] | None = None, *, timeout_s: float = 120.0) -> tuple[int, dict[str, str], Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, method="GET" if payload is None else "POST", headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout_s) as response:
            body = response.read()
            decoded = json.loads(body.decode("utf-8")) if body else {}
            return int(response.status), dict(response.headers.items()), decoded
    except HTTPError as exc:
        body = exc.read()
        try:
            decoded = json.loads(body.decode("utf-8")) if body else {}
        except json.JSONDecodeError:
            decoded = {"raw": body.decode("utf-8", errors="replace")}
        return int(exc.code), dict(exc.headers.items()), decoded


def wait_json(url: str, *, timeout_s: float = 300.0):
    deadline = time.time() + timeout_s
    last_error: str | None = None
    while time.time() < deadline:
        try:
            status, _headers, payload = request_json(url, timeout_s=5.0)
            if status < 500:
                return payload
            last_error = f"status={status} payload={payload}"
        except (RuntimeError, URLError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for {url}; last_error={last_error}")


def start_process(cmd: list[str], *, env: dict[str, str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        start_new_session=True,
    )
    proc._sparseengine_log_handle = log  # type: ignore[attr-defined]
    return proc


def stop_processes(processes: list[subprocess.Popen]):
    for proc in processes:
        if proc.poll() is not None:
            continue
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for proc in processes:
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)
        log = getattr(proc, "_sparseengine_log_handle", None)
        if log is not None:
            log.close()


def make_prompt(prefix: str, length_words: int) -> str:
    words = [prefix]
    for idx in range(max(1, int(length_words))):
        words.append(f"word{idx % 97}")
    return " ".join(words)


def call_completion(router_url: str, payload: dict[str, Any], *, timeout_s: float = 240.0) -> dict[str, Any]:
    status, headers, body = request_json(f"{router_url}/v1/completions", payload, timeout_s=timeout_s)
    if status >= 400:
        raise RuntimeError(f"completion failed status={status} body={body}")
    lower_headers = {key.lower(): value for key, value in headers.items()}
    return {
        "headers": {
            "worker": lower_headers.get("x-sparseengine-worker"),
            "reason": lower_headers.get("x-sparseengine-route-reason"),
            "method": lower_headers.get("x-sparseengine-sparse-method"),
        },
        "usage": body.get("usage"),
        "choice_count": len(body.get("choices", [])),
    }


def run_requests(args: argparse.Namespace, router_url: str, worker_urls: list[str]) -> dict[str, Any]:
    rng = random.Random(args.seed)
    results: dict[str, Any] = {"worker_urls": worker_urls, "requests": []}

    warm_prompt = make_prompt("shared-prefix-router-smoke", args.prefix_words)
    warm = call_completion(
        router_url,
        {
            "model": args.served_model_name,
            "prompt": warm_prompt,
            "max_tokens": 2,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "sengine_target_worker": "0",
        },
    )
    results["warmup"] = warm

    status, _headers, match_after_warm = request_json(
        f"{worker_urls[0]}/v1/prefix_cache/match",
        {"text": warm_prompt},
        timeout_s=60.0,
    )
    results["prefix_match_after_warmup"] = {"status": status, "payload": match_after_warm}

    balanced = call_completion(
        router_url,
        {
            "model": args.served_model_name,
            "prompt": warm_prompt,
            "max_tokens": 2,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
        },
    )
    results["balanced_prefix_route"] = balanced

    def _blocking_call(_idx: int):
        return call_completion(
            router_url,
            {
                "model": args.served_model_name,
                "prompt": make_prompt("busy-prefix", args.prefix_words),
                "max_tokens": args.busy_max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "sengine_target_worker": "0",
            },
            timeout_s=360.0,
        )

    busy_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.busy_requests) as pool:
        futures = [pool.submit(_blocking_call, idx) for idx in range(args.busy_requests)]
        time.sleep(args.overload_probe_delay_s)
        overload_probe = call_completion(
            router_url,
            {
                "model": args.served_model_name,
                "prompt": warm_prompt,
                "max_tokens": 2,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
            },
        )
        results["overload_probe_route"] = overload_probe
        results["busy_results"] = [future.result(timeout=420.0) for future in futures]
    results["busy_target_worker_elapsed_s"] = time.perf_counter() - busy_started

    def _balance_call(idx: int):
        return call_completion(
            router_url,
            {
                "model": args.served_model_name,
                "prompt": make_prompt(f"balance-prefix-{idx}", args.prefix_words),
                "max_tokens": args.balance_max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
            },
            timeout_s=360.0,
        )

    balance_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.balance_requests) as pool:
        futures = [pool.submit(_balance_call, idx) for idx in range(args.balance_requests)]
        results["balance_burst"] = [future.result(timeout=420.0) for future in futures]
    results["balance_burst_elapsed_s"] = time.perf_counter() - balance_started

    for idx in range(args.random_requests):
        batch_size = rng.randint(1, args.max_random_batch_size)
        profile = "conversation" if idx % 2 == 0 else "bulk"
        prompts = [
            make_prompt(f"{profile}-random-{idx}-{batch_idx}", rng.randint(args.min_random_words, args.max_random_words))
            for batch_idx in range(batch_size)
        ]
        payload = {
            "model": args.served_model_name,
            "prompt": prompts,
            "max_tokens": rng.randint(1, args.max_random_output_tokens),
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "sengine_route_profile": profile,
        }
        record = call_completion(router_url, payload, timeout_s=360.0)
        record["profile"] = profile
        record["batch_size"] = batch_size
        results["requests"].append(record)
        if record["choice_count"] != batch_size:
            raise RuntimeError(f"expected {batch_size} choices for profile={profile}, got {record['choice_count']}")

    status, _headers, delete_result = request_json(
        f"{router_url}/v1/prefix_cache/delete_subtree",
        {"text": warm_prompt},
        timeout_s=120.0,
    )
    results["delete_subtree"] = {"status": status, "payload": delete_result}
    if status >= 400:
        raise RuntimeError(f"delete_subtree failed: {delete_result}")
    return results


def _route_reason(record: dict[str, Any]) -> str | None:
    headers = record.get("headers")
    if not isinstance(headers, dict):
        return None
    reason = headers.get("reason")
    return str(reason) if reason is not None else None


def _route_worker(record: dict[str, Any]) -> str | None:
    headers = record.get("headers")
    if not isinstance(headers, dict):
        return None
    worker = headers.get("worker")
    return str(worker) if worker is not None else None


def validate_summary(summary: dict[str, Any], args: argparse.Namespace):
    required_records = ["warmup", "balanced_prefix_route", "overload_probe_route"]
    for key in required_records:
        record = summary.get(key)
        if not isinstance(record, dict):
            raise RuntimeError(f"router smoke missing {key}")
        if int(record.get("choice_count", 0) or 0) <= 0:
            raise RuntimeError(f"router smoke {key} returned no choices: {record}")
        if not _route_worker(record):
            raise RuntimeError(f"router smoke {key} missing X-SparseVLLM-Worker header: {record}")
        if not _route_reason(record):
            raise RuntimeError(f"router smoke {key} missing X-SparseVLLM-Route-Reason header: {record}")

    warm_reason = _route_reason(summary["warmup"])
    if warm_reason != "target_worker":
        raise RuntimeError(f"warmup should target worker 0, got reason={warm_reason!r}")

    match_payload = (summary.get("prefix_match_after_warmup") or {}).get("payload") or {}
    if (summary.get("prefix_match_after_warmup") or {}).get("status") != 200:
        raise RuntimeError(f"prefix match after warmup failed: {summary.get('prefix_match_after_warmup')}")
    if args.require_prefix_route:
        if not bool(match_payload.get("supported")) or not bool(match_payload.get("enabled")):
            raise RuntimeError(f"prefix cache match is not supported/enabled: {match_payload}")
        if int(match_payload.get("matched_tokens", 0) or 0) <= 0:
            raise RuntimeError(f"warmup did not create a positive prefix-cache match: {match_payload}")
        balanced_reason = _route_reason(summary["balanced_prefix_route"])
        if balanced_reason != "best_prefix_match":
            raise RuntimeError(f"balanced request should choose prefix match, got reason={balanced_reason!r}")

    if args.require_overload_reroute:
        overload_reason = _route_reason(summary["overload_probe_route"])
        if overload_reason != "prefix_match_overloaded_lowest_load":
            raise RuntimeError(f"overload probe should reroute to lowest load, got reason={overload_reason!r}")
        if _route_worker(summary["overload_probe_route"]) == _route_worker(summary["warmup"]):
            raise RuntimeError(
                "overload probe stayed on the warmup worker despite --require-overload-reroute: "
                f"warmup={summary['warmup']} overload={summary['overload_probe_route']}"
            )

    busy_results = summary.get("busy_results") or []
    if len(busy_results) != int(args.busy_requests):
        raise RuntimeError(f"expected {args.busy_requests} busy results, got {len(busy_results)}")
    balance_burst = summary.get("balance_burst") or []
    if len(balance_burst) != int(args.balance_requests):
        raise RuntimeError(f"expected {args.balance_requests} balance results, got {len(balance_burst)}")
    random_requests = summary.get("requests") or []
    if len(random_requests) != int(args.random_requests):
        raise RuntimeError(f"expected {args.random_requests} random requests, got {len(random_requests)}")

    delete_subtree = summary.get("delete_subtree") or {}
    if delete_subtree.get("status") != 200:
        raise RuntimeError(f"delete_subtree failed validation: {delete_subtree}")


def build_failure_summary(
    partial_summary: dict[str, Any],
    args: argparse.Namespace,
    methods: list[str],
    gpus: list[str],
    ports: list[int],
    exc: BaseException,
) -> dict[str, Any]:
    summary = dict(partial_summary)
    summary.update(
        {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "methods": methods,
            "gpus": gpus,
            "worker_ports": ports,
            "router_port": args.router_port,
            "model": args.model,
            "served_model_name": args.served_model_name,
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real two-worker SparseEngine OpenAI router smoke test.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="router-smoke-model")
    parser.add_argument("--methods", default="omnikv,snapkv")
    parser.add_argument("--gpus", default="6,7")
    parser.add_argument("--worker-ports", default="18181,18182")
    parser.add_argument("--router-port", type=int, default=18180)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--prefix-words", type=int, default=512)
    parser.add_argument("--busy-requests", type=int, default=4)
    parser.add_argument("--busy-max-tokens", type=int, default=64)
    parser.add_argument("--overload-probe-delay-s", type=float, default=0.5)
    parser.add_argument("--balance-requests", type=int, default=6)
    parser.add_argument("--balance-max-tokens", type=int, default=16)
    parser.add_argument("--random-requests", type=int, default=8)
    parser.add_argument("--min-random-words", type=int, default=16)
    parser.add_argument("--max-random-words", type=int, default=384)
    parser.add_argument("--max-random-batch-size", type=int, default=4)
    parser.add_argument("--max-random-output-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--startup-timeout-s", type=float, default=420.0)
    parser.add_argument("--require-prefix-route", action="store_true")
    parser.add_argument("--require-overload-reroute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    ports = [int(item) for item in args.worker_ports.split(",") if item.strip()]
    if len(methods) != 2 or len(gpus) != 2 or len(ports) != 2:
        raise ValueError("--methods, --gpus, and --worker-ports must each contain exactly two entries.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_urls = [f"http://127.0.0.1:{port}" for port in ports]
    router_url = f"http://127.0.0.1:{args.router_port}"
    processes: list[subprocess.Popen] = []
    summary: dict[str, Any] = {}
    env_base = os.environ.copy()
    pythonpath = os.pathsep.join([str(Path.cwd()), str(Path.cwd() / "src"), env_base.get("PYTHONPATH", "")])
    env_base["PYTHONPATH"] = pythonpath

    try:
        for idx, (method, gpu, port) in enumerate(zip(methods, gpus, ports)):
            kwargs = method_kwargs(method, max_model_len=args.max_model_len, prefix_cache=True)
            kwargs_path = output_dir / f"worker_{idx}_{method}_engine_kwargs.json"
            write_json(kwargs_path, kwargs)
            env = dict(env_base)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["MASTER_ADDR"] = "127.0.0.1"
            env["MASTER_PORT"] = str(24000 + int(port) % 1000)
            env["SPARSEENGINE_MASTER_PORT"] = str(24000 + int(port) % 1000)
            env["SPARSEENGINE_WORKER_TAGS"] = f"{method},{'dialog' if method in PREFIX_METHODS else 'bulk'}"
            cmd = [
                sys.executable,
                "-m",
                "sparseengine.entrypoints.openai.api_server",
                "--model",
                args.model,
                "--served-model-name",
                args.served_model_name,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--engine-kwargs",
                str(kwargs_path),
                "--request-log-dir",
                str(output_dir / f"worker_{idx}_{method}_requests"),
            ]
            processes.append(start_process(cmd, env=env, log_path=output_dir / f"worker_{idx}_{method}.log"))

        for url in worker_urls:
            wait_json(f"{url}/health", timeout_s=args.startup_timeout_s)
            wait_json(f"{url}/v1/worker/info", timeout_s=60.0)

        profiles = {
            "conversation": {"methods": [method for method in methods if method in PREFIX_METHODS], "preferred_tags": ["dialog"]},
            "bulk": {"methods": [method for method in methods if method == "snapkv"] or methods, "preferred_tags": ["bulk"]},
        }
        profiles_path = output_dir / "router_profiles.json"
        write_json(profiles_path, profiles)
        router_cmd = [
            sys.executable,
            "-m",
            "sparseengine.entrypoints.openai.smart_router",
            "--worker-url",
            worker_urls[0],
            "--worker-url",
            worker_urls[1],
            "--host",
            "127.0.0.1",
            "--port",
            str(args.router_port),
            "--request-timeout-s",
            "240",
            "--overload-load-factor",
            "1.0",
            "--load-abs-threshold",
            "0",
            "--profiles-json",
            str(profiles_path),
            "--route-log-dir",
            str(output_dir / "router_routes"),
        ]
        processes.append(start_process(router_cmd, env=env_base, log_path=output_dir / "router.log"))
        wait_json(f"{router_url}/health", timeout_s=120.0)

        summary = run_requests(args, router_url, worker_urls)
        summary.update(
            {
                "status": "success",
                "methods": methods,
                "gpus": gpus,
                "worker_ports": ports,
                "router_port": args.router_port,
                "model": args.model,
                "served_model_name": args.served_model_name,
            }
        )
        validate_summary(summary, args)
        write_json(output_dir / "router_smoke_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        write_json(
            output_dir / "router_smoke_summary.json",
            build_failure_summary(summary, args, methods, gpus, ports, exc),
        )
        raise
    finally:
        stop_processes(processes)


if __name__ == "__main__":
    raise SystemExit(main())
