#!/usr/bin/env python3
"""Freeze and execute MiniSWE campaigns through the canonical adapter.

No engine imports in planning/collection; server and Docker driver may run on
different hosts. All subprocesses use argv arrays and explicit environment bins.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shlex
import socket
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def capture(argv, cwd=None):
    return subprocess.check_output(argv, cwd=cwd, text=True, timeout=30).strip()


def get(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=15) as response:
        return json.load(response)


def prepare(args):
    setting = read(args.setting)
    setting["backend"] = getattr(args, "backend", "sparseengine")
    # Each concurrency has an immutable root; methods are never silently retuned.
    concurrency = args.concurrency
    if not 1 <= concurrency <= 256:
        raise ValueError("Concurrency must be between 1 and 256")
    engine_concurrency = (
        concurrency
        if args.engine_concurrency is None
        else int(args.engine_concurrency)
    )
    if not 1 <= engine_concurrency <= 256:
        raise ValueError("Engine concurrency must be between 1 and 256")
    setting["agent"]["mini_workers"] = concurrency
    for key in ("max_num_seqs_in_batch", "max_decoding_seqs"):
        setting["engine"][key] = engine_concurrency
    resident_concurrency = (
        engine_concurrency
        if args.engine_resident_concurrency is None
        else int(args.engine_resident_concurrency)
    )
    if not engine_concurrency <= resident_concurrency <= 256:
        raise ValueError(
            "Engine resident concurrency must be between engine concurrency and 256"
        )
    setting["engine"]["max_num_seqs_in_gpu"] = resident_concurrency
    buckets = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    setting["engine"]["decode_graph_capture_sizes"] = sorted(
        {b for b in buckets if b <= engine_concurrency} | {engine_concurrency}
    )
    setting["slow_baseline_policy"]["pilot_instances"] = max(64, concurrency)
    root = args.root.resolve()
    if root == HERE or HERE in root.parents:
        raise ValueError("Place artifacts outside the recipe source directory")
    root.mkdir(parents=True, exist_ok=False)
    write(root / "setting.json", setting)
    write(root / "source.json", {
        "commit": capture(["git", "rev-parse", "HEAD"], REPO),
        "dirty": capture(["git", "status", "--porcelain=v1"], REPO),
        "protocol": setting["protocol"], "prepared_at": time.time(),
    })
    for method, overrides in setting["methods"].items():
        config = {**setting["engine"], **overrides}
        if setting["backend"] == "vllm" and method == "vanilla-prefix":
            config = {
                "backend": "vllm", "sparse_method": "vanilla",
                **{key: config[key] for key in (
                    "tensor_parallel_size", "data_parallel_size", "expert_parallel_size",
                    "gpu_memory_utilization", "max_model_len", "max_num_batched_tokens",
                    "enable_prefix_caching",
                )},
                "max_num_seqs": engine_concurrency,
                "enable_chunked_prefill": True, "enforce_eager": not config["decode_graph"],
                "all2all_backend": "allgather_reducescatter",
                "reasoning_parser": "glm45", "tool_call_parser": "glm47",
            }
        write(root / method / "engine.json", config)
    print(f"Prepared {root}; server and benchmark commands are in README.md")


def environment(python):
    python = str(Path(python).absolute())
    env = os.environ.copy()
    env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = os.pathsep.join((str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")))
    env["OPENAI_API_KEY"] = "local-sparseengine"
    return python, env


def stop(process):
    # The process was started in a new session by this script; never pkill by name.
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def process_start_ticks(pid):
    raw = Path(f"/proc/{pid}/stat").read_text()
    return int(raw[raw.rfind(")") + 2:].split()[19])


def run_logged(argv, env, directory, name, timeout, process_record=None):
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / f"{name}.invocation.json", {"argv": argv, "started": time.time()})
    if process_record is not None and process_record.exists():
        raise FileExistsError(f"Refusing to replace process identity record: {process_record}")
    started = time.monotonic()
    result = {"status": "running"}
    with (directory / f"{name}.log").open("x") as log:
        process = subprocess.Popen(argv, env=env, cwd=REPO, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if process_record is not None:
                write(process_record, {
                    "pid": process.pid,
                    "process_group": os.getpgid(process.pid),
                    "start_ticks": process_start_ticks(process.pid),
                    "hostname": socket.gethostname(),
                    "boot_id": boot_id(),
                    "command": shlex.join(argv),
                })
        except BaseException:
            stop(process)
            raise
        try:
            code = process.wait(timeout=timeout)
            stop_record = directory / f"{name}_stop.json"
            requested = stop_record.exists() and read(stop_record).get("status") == "stop_requested"
            result = {
                "status": "success" if code == 0 or requested else "failed",
                "exit_code": code,
            }
            if requested:
                result["stop_reason"] = read(stop_record).get("reason", "requested_by_operator")
        except subprocess.TimeoutExpired:
            result = {"status": "timeout", "timeout_seconds": timeout}
        except BaseException:
            result = {"status": "aborted"}
            raise
        finally:
            stop(process)
            result["elapsed_seconds"] = time.monotonic() - started
            write(directory / f"{name}.result.json", result)
    if result["status"] != "success":
        raise RuntimeError(f"{name}: {result}; inspect {directory}")


def idle_pair(gpus):
    ids = gpus.split(",")
    if len(ids) != 2 or len(set(ids)) != 2 or not all(x.isdigit() for x in ids):
        raise ValueError("--gpus must name exactly two distinct numeric GPU indices")
    raw = capture(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
    rows = {x[0]: x for line in raw.splitlines() if (x := [v.strip() for v in line.split(",")])}
    apps = capture(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"])
    for gpu in ids:
        row = rows[gpu]
        if row[1] in apps or int(row[2]) > 128 or int(row[3]) != 0:
            raise RuntimeError(f"GPU {gpu} is occupied; wait, do not share or kill its processes")
    return {"gpus": raw, "processes": apps}


def serve(args):
    directory = args.root / args.method / args.phase
    directory.mkdir(parents=True, exist_ok=True)
    python, env = environment(args.python)
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["LOG_LEVEL"] = "INFO"
    # Keep compiler scratch and caches off a network-mounted results directory
    # when an explicit node-local root is supplied. Never silently pick a disk.
    if getattr(args, "compile_cache_root", None) is not None:
        cache_root = args.compile_cache_root.resolve()
        for key, subdir in {
            "TRITON_CACHE_DIR": "triton", "CUDA_CACHE_PATH": "cuda",
            "FLASHINFER_WORKSPACE_BASE": "flashinfer", "TMPDIR": "tmp",
        }.items():
            path = cache_root / subdir
            path.mkdir(parents=True, exist_ok=True)
            env[key] = str(path)
    model = args.model.resolve(strict=True)
    config = read(args.root / args.method / "engine.json")
    advertised = f"glm47-{args.method}"
    # Bind test avoids inadvertently attaching the driver to an older service.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    hardware = idle_pair(args.gpus)
    backend = read(args.root / "setting.json").get("backend", "sparseengine")
    if backend == "vllm":
        if args.method != "vanilla-prefix":
            raise ValueError("The upstream vLLM comparison only supports vanilla-prefix")
        world = config["tensor_parallel_size"] * config["data_parallel_size"]
        if config["expert_parallel_size"] not in (1, world):
            raise ValueError("vLLM expert parallel size must be 1 or TP * DP")
        command = [python, "-m", "vllm.entrypoints.openai.api_server",
                   "--model", str(model), "--served-model-name", advertised,
                   "--host", "127.0.0.1", "--port", str(args.port),
                   "--dtype", "bfloat16", "--generation-config", "vllm",
                   "--enable-auto-tool-choice", "--enable-prompt-tokens-details",
                   "--middleware", "scripts.official_experiments.chain_cache_miniswe.request_logging.RequestLoggingMiddleware"]
        for key in ("tensor_parallel_size", "data_parallel_size", "gpu_memory_utilization",
                    "max_model_len", "max_num_batched_tokens", "max_num_seqs",
                    "all2all_backend", "reasoning_parser", "tool_call_parser"):
            command.extend(["--" + key.replace("_", "-"), str(config[key])])
        for key in ("enable_prefix_caching", "enable_chunked_prefill", "enforce_eager"):
            command.append(("--" if config[key] else "--no-") + key.replace("_", "-"))
        if config["expert_parallel_size"] > 1:
            command.append("--enable-expert-parallel")
        env["MINISWE_REQUEST_LOG_DIR"] = str((directory / "server_requests").resolve())
    else:
        command = [python, "-m", "sparseengine.entrypoints.openai.api_server", "--model", str(model),
               "--served-model-name", advertised, "--host", "127.0.0.1", "--port", str(args.port),
               "--engine-kwargs", str((args.root / args.method / "engine.json").resolve()),
               "--request-log-dir", str((directory / "server_requests").resolve())]
    write(directory / "server_manifest.json", {
        "backend": backend,
        "backend_version": capture([python, "-c", "import vllm; print(vllm.__version__)"]) if backend == "vllm" else None,
        "command": shlex.join(command), "model_path": str(model), "served_model_name": advertised,
        "cuda_visible_devices": args.gpus, "server_port": args.port, "engine_kwargs": config,
        "compiler_environment": {key: env.get(key) for key in (
            "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "FLASHINFER_WORKSPACE_BASE", "TMPDIR",
        )},
        "git_commit": capture(["git", "rev-parse", "HEAD"], REPO),
        "git_dirty": capture(["git", "status", "--porcelain=v1"], REPO),
        "hardware": hardware, "model_config": read(model / "config.json"),
    })
    # Server lifetime is explicit, one fresh process per method AND phase.
    run_logged(
        command,
        env,
        directory,
        "server",
        args.timeout,
        process_record=directory / "server_process.json",
    )


def stop_server(args):
    directory = args.root / args.method / args.phase
    result_path = directory / "server_stop.json"
    if result_path.exists():
        print(json.dumps(read(result_path), indent=2))
        return
    record = read(directory / "server_process.json")
    if record["hostname"] != socket.gethostname() or record["boot_id"] != boot_id():
        raise RuntimeError("Server process record belongs to another host or boot")
    pid = int(record["pid"])
    try:
        current_start = process_start_ticks(pid)
    except FileNotFoundError:
        write(result_path, {"status": "already_stopped", "pid": pid, "stopped_at": time.time()})
        print(json.dumps(read(result_path), indent=2))
        return
    if current_start != int(record["start_ticks"]) or os.getpgid(pid) != int(record["process_group"]):
        raise RuntimeError("Server PID identity changed; refusing to signal it")
    if int(record["process_group"]) != pid:
        raise RuntimeError("Recorded server is not the leader of its private process group")
    write(result_path, {"status": "stop_requested", "pid": pid, "stopped_at": time.time(),
                        "reason": getattr(args, "reason", "requested_by_operator")})
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            process_start_ticks(pid)
        except FileNotFoundError:
            break
        time.sleep(.1)
    else:
        os.killpg(pid, signal.SIGKILL)
    print(json.dumps(read(result_path), indent=2))


def gate(root):
    policy = read(root / "setting.json")["slow_baseline_policy"]
    baseline = read(root / "snapkv-no-chain/pilot/generate.result.json")
    reference = read(root / "snapkv-chain/pilot/generate.result.json")
    if reference["status"] != "success":
        raise ValueError("The matched SnapKV Chain Cache pilot must complete first")
    # This is a cost gate, never an estimate of task quality or a matched-trace speedup.
    estimate = baseline["elapsed_seconds"] * 300 / policy["pilot_instances"]
    ratio = baseline["elapsed_seconds"] / reference["elapsed_seconds"]
    if baseline["status"] not in {"success", "timeout"}:
        raise ValueError("A failed pilot is an implementation failure, not a slow baseline")
    skip = baseline["status"] == "timeout" or estimate > policy["full_generation_limit_seconds"] or ratio > policy["slowdown_ratio"]
    return {"status": "skipped_by_policy" if skip else "eligible", "pilot_status": baseline["status"],
            "projected_full_generation_seconds": estimate, "pilot_wall_time_ratio": ratio,
            "policy": policy, "scope": "closed_loop_cost_gate_only"}


def benchmark(args):
    setting = read(args.root / "setting.json")
    directory = args.root / args.method / args.phase
    if args.method == "snapkv-no-chain" and args.phase == "full" and args.stage in {"all", "generate"}:
        decision = gate(args.root)
        write(directory / "slow_baseline_decision.json", decision)
        if decision["status"] == "skipped_by_policy":
            print(json.dumps(decision, indent=2))
            return decision
    python, env = environment(args.python)
    server_manifest = args.server_manifest or directory / "server_manifest.json"
    frozen = read(args.root / args.method / "engine.json")
    if read(server_manifest)["engine_kwargs"] != frozen:
        raise ValueError("Server manifest and frozen method settings differ")
    if args.stage in {"prepare", "generate", "all"}:
        # /models is JSON on both servers and checks the actual advertised model.
        models = get(args.api_base.rstrip("/") + "/models")
        if f"glm47-{args.method}" not in {item["id"] for item in models["data"]}:
            raise ValueError("Server does not advertise the prepared model")
    agent = setting["agent"].copy()
    if args.phase == "smoke":
        agent.update(mini_workers=1, eval_workers=1, batch_size=1)
    command = [python, "-m", "benchmark.swe_bench_lite.run", "--stage", args.stage,
               "--mini-command", shlex.join([python, str(HERE / "timed_mini.py")]),
               "--swe-bench-dir", str(args.swe_bench_dir.resolve(strict=True)),
               "--run-dir", str((directory / "benchmark").resolve()),
               "--model", f"openai/glm47-{args.method}", "--served-model-name", f"glm47-{args.method}",
               "--api-base", args.api_base, "--server-manifest", str(server_manifest.resolve()),
               "--chain-cache" if frozen.get("prefix_cache_mode") == "chain" else "--no-chain-cache"]
    for key, value in agent.items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            command.append(option if value else "--no-" + key.replace("_", "-"))
        else:
            command.extend([option, str(value)])
    if args.phase != "full":
        pilot_count = setting["slow_baseline_policy"]["pilot_instances"]
        command.extend(["--slice", "0:1" if args.phase == "smoke" else f"0:{pilot_count}"])
    # No hidden downloading, concurrency reduction, cache fallback, or retries.
    run_logged(command, env, directory, args.stage, args.timeout)


def detached_benchmark(args):
    """Start once, then query a worker-owned stage independently of SSH."""
    directory = args.root.resolve() / args.method / args.phase
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / f"{args.stage}.detached.result.json"
    marker = directory / f"{args.stage}.detached.json"
    session = "miniswe-" + hashlib.sha256(str(marker).encode()).hexdigest()[:20]
    with (directory / f"{args.stage}.detached.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if result_path.exists():
            return read(result_path)
        if marker.exists():
            running = subprocess.run(
                ["tmux", "has-session", "-t", "=" + session],
                capture_output=True,
            ).returncode == 0
            if not running:
                # Recheck after has-session: completion and tmux exit can race.
                if result_path.exists():
                    return read(result_path)
                raise RuntimeError(f"Detached stage vanished without a result: {marker}")
            # A live worker owns its stage until it publishes a terminal result.
            # A separate HTTP probe can fail while inference keeps progressing.
            return {"status": "running", "session": session}
        if not marker.exists():
            if args.stage in {"prepare", "generate"}:
                try:
                    models = get(args.api_base.rstrip("/") + "/models")
                    if f"glm47-{args.method}" not in {item["id"] for item in models["data"]}:
                        raise ValueError("Server does not advertise the prepared model")
                except (OSError, ValueError) as exc:
                    return {"status": "waiting_for_server", "api_ready": False,
                            "error": f"{type(exc).__name__}: {exc}"}
            command = [args.python, str(HERE / "run.py"), "bench",
                       "--root", str(args.root.resolve()), "--method", args.method,
                       "--phase", args.phase, "--stage", args.stage,
                       "--python", args.python, "--timeout", str(args.timeout),
                       "--swe-bench-dir", str(args.swe_bench_dir.resolve()),
                       "--api-base", args.api_base, "--detached-child"]
            if args.server_manifest:
                command += ["--server-manifest", str(args.server_manifest.resolve())]
            # Record before launching: a lost SSH reply must not start a duplicate.
            write(marker, {"session": session, "command": command, "started": time.time()})
            tmux = ["tmux", "new-session", "-d", "-s", session, "-c", str(REPO)]
            for key in ("PATH", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE",
                        "LITELLM_LOCAL_MODEL_COST_MAP"):
                if key in os.environ:
                    tmux += ["-e", key + "=" + os.environ[key]]
            log = directory / f"{args.stage}.detached.log"
            tmux += [shlex.join(command) + " > " + shlex.quote(str(log)) + " 2>&1"]
            subprocess.run(tmux, check=True, capture_output=True)
        return {"status": "running", "session": session}


def detached_child(args):
    directory = args.root / args.method / args.phase
    result = {"status": "failed", "exit_code": 1}
    try:
        benchmark_result = benchmark(args)
        if benchmark_result is None:
            result = {"status": "success", "exit_code": 0}
        elif benchmark_result["status"] == "skipped_by_policy":
            result = {**benchmark_result, "exit_code": 0}
        else:
            raise ValueError(f"Unknown benchmark terminal state: {benchmark_result}")
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Publish atomically: status polling must never read a partial JSON file.
        path = directory / f"{args.stage}.detached.result.json"
        temporary = path.with_suffix(".tmp")
        write(temporary, {**result, "finished": time.time()})
        temporary.replace(path)


def wait_remote(args):
    """Retry transport/status checks, never the benchmark itself."""
    directory = args.root / args.method / args.phase
    directory.mkdir(parents=True, exist_ok=True)
    ssh = shlex.split(args.ssh_command)
    worker = shlex.split(args.worker_command)
    if not ssh or not worker or "--detach" not in worker:
        raise ValueError("Supply SSH argv and a worker command invoking bench --detach")
    command = ssh + [shlex.join(worker)]
    started = time.monotonic()
    unhealthy_since = None
    failed_probes = 0
    outcome = {"status": "failed"}
    write(directory / f"{args.stage}.remote.invocation.json", {
        "command": command, "recovery_timeout": args.recovery_timeout,
        "reconnect_attempts": args.reconnect_attempts,
        "timeout": args.timeout, "started": time.time(),
    })
    try:
        with (directory / f"{args.stage}.remote.events.jsonl").open("x", buffering=1) as log:
            while time.monotonic() - started < args.timeout:
                try:
                    response = subprocess.run(command, capture_output=True, text=True, timeout=30)
                    if response.returncode not in (0, 255):
                        raise RuntimeError(f"Remote stage control failed: {response.stderr}")
                    state = (json.loads(response.stdout) if response.returncode == 0 else
                             {"status": "disconnected", "error": response.stderr})
                except subprocess.TimeoutExpired:
                    state = {"status": "disconnected", "error": "SSH status probe timed out"}
                log.write(json.dumps({"time": time.time(), **state}) + "\n")
                status = state["status"]
                if status in {"success", "skipped_by_policy"}:
                    outcome = state
                    return
                if status == "failed":
                    outcome = state
                    raise RuntimeError(f"Remote benchmark failed: {state}")
                if status not in {"running", "waiting_for_server", "disconnected"}:
                    raise ValueError(f"Unknown remote stage state: {state}")
                # Accept older workers' api_ready field as diagnostic only.
                # Running confirms control recovery, independently of HTTP health.
                if status == "running":
                    unhealthy_since = None
                    failed_probes = 0
                else:
                    failed_probes += 1
                    if unhealthy_since is None:
                        unhealthy_since = time.monotonic()
                    # The first failed probe detects an outage; allow N further
                    # connection attempts, resetting after confirmed running state.
                    if failed_probes > args.reconnect_attempts:
                        outcome = {"status": "reconnect_exhausted", "last_state": state,
                                   "reconnect_attempts": args.reconnect_attempts}
                        raise ConnectionError("Remote reconnect attempts exhausted; task state may be unknown")
                    if time.monotonic() - unhealthy_since >= args.recovery_timeout:
                        outcome = {"status": "recovery_timeout", "last_state": state}
                        raise TimeoutError("Remote connectivity recovery deadline expired; task state may be unknown")
                time.sleep(args.poll_interval)
            outcome = {"status": "timeout"}
            raise TimeoutError("Remote stage wall-time deadline expired")
    finally:
        write(directory / f"{args.stage}.remote.result.json", {
            **outcome, "elapsed_seconds": time.monotonic() - started,
        })


def collect(args):
    directory = args.root / args.method / args.phase
    summary = read(directory / "benchmark/final_summary.json")
    setting = read(args.root / "setting.json")
    expected = {"smoke": 1, "pilot": setting["slow_baseline_policy"]["pilot_instances"], "full": 300}[args.phase]
    if summary["total_instances"] != expected:
        raise ValueError("Official summary has the wrong instance count")
    selected = set((directory / "benchmark/instances.txt").read_text().split())
    for filename in ("generation_results.jsonl", "per_sample_results.jsonl"):
        rows = [json.loads(line) for line in (directory / "benchmark" / filename).read_text().splitlines()]
        ids = [row["instance_id"] for row in rows]
        if len(ids) != expected or len(set(ids)) != expected or set(ids) != selected:
            raise ValueError(f"Missing, duplicate, or unexpected samples in {filename}")
    per_sample = {row["instance_id"]: row for row in rows}
    timings = sorted((directory / "benchmark/batches").glob("batch_*/*/*.wall_time.json"))
    task_rows = []
    for path in timings:
        row = read(path)
        if not math.isfinite(row["elapsed_s"]) or row["elapsed_s"] < 0:
            raise ValueError(f"Invalid task duration in {path}")
        outcome = per_sample[row["instance_id"]]
        task_rows.append({**row, "status": outcome["status"], "resolved": outcome["resolved"],
                          "generation_exit_status": outcome.get("generation_exit_status"),
                          "model_stats": outcome.get("model_stats")})
    if len(task_rows) != expected or {r["instance_id"] for r in task_rows} != selected:
        raise ValueError("Missing or duplicate task timings; instrumentation was not active or run was interrupted")
    with (directory / "task_samples.jsonl").open("x") as stream:
        for row in task_rows:
            stream.write(json.dumps(row) + "\n")
    # Shared percentile implementation; do not create another quantile formula.
    import sys
    sys.path.insert(0, str(REPO))
    from benchmark.efficiency.metrics import percentile
    task_summary = {}
    for label, subset in (("all_attempts", task_rows), ("resolved", [r for r in task_rows if r["resolved"]])):
        durations = [r["elapsed_s"] for r in subset]
        task_summary[label] = {"count": len(subset),
                               "p50_s": percentile(durations, .5) if durations else None,
                               "p95_s": percentile(durations, .95) if durations else None}
    generation = read(directory / "generate.result.json")
    evaluation = read(directory / "evaluate.result.json")
    logs = sorted((directory / "server_requests").glob("*.json"))
    if not logs:
        raise ValueError("No server request logs; copy them from the GPU host before collecting")
    totals = dict(requests=0, prompt_tokens=0, completion_tokens=0, reused_tokens=0,
                  uncached_prompt_tokens=0, summed_request_seconds=0.0)
    with (directory / "request_samples.jsonl").open("x") as stream:
        for path in logs:
            record = read(path)
            if record["status"] != "success":
                raise ValueError(f"Non-success request log: {path}; inspect the failure before aggregation")
            usage = record["response"]["usage"]
            prompt, output = usage["prompt_tokens"], usage["completion_tokens"]
            reused = usage["prompt_tokens_details"]["cached_tokens"]
            if not 0 <= reused <= prompt or output < 0 or record["elapsed_s"] < 0:
                raise ValueError(f"Invalid usage or timing: {path}")
            row = {"status": "success", "request_id": record["request_id"],
                   "elapsed_s": record["elapsed_s"], "prompt_tokens": prompt,
                   "completion_tokens": output, "reused_tokens": reused,
                   "uncached_prompt_tokens": prompt - reused, "source": str(path),
                   "ttft_ms": None, "tpot_ms": None}
            stream.write(json.dumps(row) + "\n")
            totals["requests"] += 1
            for key in ("prompt_tokens", "completion_tokens", "reused_tokens", "uncached_prompt_tokens"):
                totals[key] += row[key]
            totals["summed_request_seconds"] += row["elapsed_s"]
    write(directory / "report.json", {
        "official": summary, "requests": totals,
        "task_completion": task_summary,
        "generation_stage": generation, "official_evaluation_stage": evaluation,
        "instance_ids_sha256": hashlib.sha256("\n".join(sorted(selected)).encode()).hexdigest(),
        "ttft_status": "not_measured_nonstreaming_api", "tpot_status": "not_measured",
        "prefill_scope": "logical_uncached_prompt_tokens_excludes_unobserved_recompute",
        "time_scope": "sum_of_overlapping_server_request_durations_not_wall_time_or_gpu_time",
        "comparison_scope": "closed_loop_same_tasks_different_generated_trajectories",
    })
    print(json.dumps({"official": summary, "requests": totals}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--setting", type=Path, default=HERE / "setting.json")
    p.add_argument("--backend", choices=("sparseengine", "vllm"), default="sparseengine")
    p.add_argument("--concurrency", type=int, default=64,
                   help="Target concurrent MiniSWE agents; use a fresh root for each value")
    p.add_argument("--engine-concurrency", type=int,
                   help="Server prefill/decode sequence limit; defaults to --concurrency")
    p.add_argument("--engine-resident-concurrency", type=int,
                   help="Resident sequence rows; defaults to --engine-concurrency")
    for action in ("serve", "bench", "collect", "stop", "wait-remote"):
        p = sub.add_parser(action)
        p.add_argument("--root", type=Path, required=True)
        p.add_argument("--method", choices=list(read(HERE / "setting.json")["methods"]), required=True)
        p.add_argument("--phase", choices=("smoke", "pilot", "full"), required=True)
        if action in {"serve", "bench"}:
            p.add_argument("--python", required=True, help="Environment Python; its bin is added to PATH")
            p.add_argument("--timeout", type=int, default=86400)
        if action == "serve":
            p.add_argument("--model", type=Path, required=True)
            p.add_argument("--gpus", required=True)
            p.add_argument("--port", type=int, default=18147)
            p.add_argument("--compile-cache-root", type=Path,
                           help="Explicit node-local compiler cache/scratch directory; check mount and capacity first")
        elif action == "bench":
            p.add_argument("--swe-bench-dir", type=Path, required=True)
            p.add_argument("--stage", choices=("prepare", "generate", "evaluate", "summarize"), required=True)
            p.add_argument("--api-base", default="http://127.0.0.1:18147/v1")
            p.add_argument("--server-manifest", type=Path)
            mode = p.add_mutually_exclusive_group()
            mode.add_argument("--detach", action="store_true", help="Start/query a persistent worker tmux stage")
            mode.add_argument("--detached-child", action="store_true", help=argparse.SUPPRESS)
        elif action == "wait-remote":
            p.add_argument("--stage", choices=("prepare", "generate", "evaluate", "summarize"), required=True)
            p.add_argument("--ssh-command", required=True, help="Shell-quoted SSH argv ending in the worker host")
            p.add_argument("--worker-command", required=True, help="Activated worker launcher command with --detach")
            p.add_argument("--timeout", type=int, default=43200)
            p.add_argument("--recovery-timeout", type=int, default=1800)
            p.add_argument("--reconnect-attempts", type=int, default=30)
            p.add_argument("--poll-interval", type=float, default=10)
        elif action == "stop":
            p.add_argument("--reason", choices=("requested_by_operator", "generation_completed",
                                               "coordinator_failure", "recovery_timeout"),
                           default="requested_by_operator")
    args = parser.parse_args()
    if hasattr(args, "timeout") and args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.action == "wait-remote" and (
        args.recovery_timeout <= 0 or args.poll_interval <= 0 or args.reconnect_attempts <= 0
    ):
        parser.error("recovery timeout, reconnect attempts and poll interval must be positive")
    if args.action == "bench" and args.detach:
        print(json.dumps(detached_benchmark(args)))
        return
    if args.action == "bench" and args.detached_child:
        detached_child(args)
        return
    {"prepare": prepare, "serve": serve, "bench": benchmark,
     "collect": collect, "stop": stop_server, "wait-remote": wait_remote}[args.action](args)


if __name__ == "__main__":
    main()
