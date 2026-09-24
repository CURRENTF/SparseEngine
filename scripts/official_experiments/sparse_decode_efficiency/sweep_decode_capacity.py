"""Sweep the canonical microbench to a verified integer concurrency boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parent


class LaneFailure(RuntimeError):
    """A finished/terminated benchmark or invalid measurement, not a resource failure."""


def run_lane_stages(lanes, smoke, sweep, record_failure, ensure_resources):
    """A failed smoke/sweep skips only that lane; infrastructure errors propagate."""
    failures = {}
    for stage, operation in (("smoke", smoke), ("sweep", sweep)):
        for lane in lanes:
            if lane in failures:
                continue
            ensure_resources()
            try:
                operation(lane)
            except LaneFailure as error:
                ensure_resources()
                failures[lane] = {"phase": stage, "error": str(error)}
                record_failure(lane, failures[lane])
    return failures


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_campaign_config(path):
    config_text = os.path.expandvars(path.read_text())
    unresolved = re.findall(r"\$\{[^}]+\}", config_text)
    if unresolved:
        raise ValueError(f"Set these environment variables first: {sorted(set(unresolved))}")
    return json.loads(config_text)


def resolve_lane(config, lane):
    external = config.get("external_lanes", {}).get(lane)
    if external:
        return external["engine"], external["method"], external
    engine, method = (("vllm", "vanilla") if lane == "vllm-vanilla"
                      else ("sparseengine", lane.removeprefix("sengine-")))
    return engine, method, None


def probe_capacity_boundary(attempts):
    """Require observed powers, the integer maximum, and explicit max+1 failure."""
    successes = {a["concurrency"] for a in attempts if a["status"] == "success"}
    failures = {a["concurrency"] for a in attempts if a["status"] == "capacity_exceeded"}
    if len(successes | failures) != len(attempts) or not successes or not failures:
        raise LaneFailure("Boundary assembly has duplicate, missing or unclassified points")
    maximum, upper = max(successes), min(failures)
    required, power = {maximum}, 1
    while power <= maximum:
        required.add(power)
        power *= 2
    if upper != maximum + 1 or not required <= successes:
        raise LaneFailure("Probes do not establish an integer maximum and max+1")
    return maximum, upper


def minimum_kv_slots(case):
    """Return the minimum logged per-rank KV pool across engine repetitions."""
    slots = []
    for path in case.rglob("*.log"):
        text = path.read_text(errors="replace")
        slots.extend(int(x.replace(",", "")) for x in re.findall(
            r"(?:kv_slots=|GPU KV cache size: |Vortex KV slots: )([\d,]+)", text))
    return min(slots) if slots else None


def full_kv_capacity_hint(case, span):
    """A hint, not a measured boundary: minimum observed per-rank full KV pool."""
    slots = minimum_kv_slots(case)
    return slots // span if slots else None


def capacity_failure(error, log_path):
    if any(text in error.lower() for text in (
            "request finished before all admission waves completed",
            "requests finished before full decode admission")):
        return False
    if any(text in error.lower() for text in ("out of memory", "full decode batch capacity exceeded", "no runnable sequences", "cannot fit", "cannot admit")):
        return True
    if "Full decode batch capacity exceeded: scheduler preemption" in log_path.read_text():
        # SGLang terminates its parent when a scheduler worker fails; retain the
        # worker's explicit capacity diagnosis instead of inferring from SIGKILL.
        return True
    # Require a capacity skip for the exact missing graph, not another family.
    missing = re.search(r"no startup-captured graph for batch_size=(\d+), path=['\"]([^'\"]+)['\"]", error)
    if missing is None:
        return False
    batch, path = missing.groups()
    skipped = re.findall(
        r"Startup CUDA Graph family exceeds KV capacity during prefill or decode preparation: batch=(\d+) path=['\"]([^'\"]+)['\"]\.",
        log_path.read_text(),
    )
    if (batch, path) in skipped:
        return True
    # The runner can also target a checkout from before unified decode.
    if path not in {"long", "short"}:
        return False
    skipped = re.findall(
        r"Startup CUDA Graph family exceeds KV capacity during prefill or decode preparation: batch=(\d+) long=(True|False)\.",
        log_path.read_text(),
    )
    return (batch, "True" if path == "long" else "False") in skipped


def validate_boundary_reuse(previous, config, hp, command, gpus, *, allow_equivalent_gpus=False):
    """Validate workload and device compatibility, not source-file equality."""
    if json.loads((previous.parent.parent / "campaign.json").read_text()) != config:
        raise ValueError(f"Refusing changed campaign for boundary reuse: {previous}")
    identity = json.loads((previous / "identity.json").read_text())
    if identity["env"]["CUDA_VISIBLE_DEVICES"] != gpus:
        if not allow_equivalent_gpus:
            raise ValueError("Boundary reuse requires the same physical GPU assignment or explicit equivalent-GPU validation")
        fields = "name,memory.total,compute_cap,driver_version,mig.mode.current,power.limit,clocks.max.sm,clocks.max.memory"
        records = [subprocess.check_output(["nvidia-smi", "-i", devices,
                   "--query-gpu=" + fields, "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
                   for devices in (identity["env"]["CUDA_VISIBLE_DEVICES"], gpus)]
        if not records[0] or records[0] != records[1] or "[N/A]" in records[0]:
            raise ValueError("Boundary reuse GPU hardware/configuration mismatch")
        identity["reuse_gpu_comparison"] = dict(source_gpus=identity["env"]["CUDA_VISIBLE_DEVICES"],
            target_gpus=gpus, fields=fields, source=records[0], target=records[1],
            scope="live hardware/config comparison; not a cross-device timing equivalence claim")
    if json.loads((previous / "hyper_params.json").read_text()) != hp:
        raise ValueError("Refusing changed hyperparameters for boundary reuse")

    def normalized(argv):
        argv = list(argv)
        for flag in ("--hyper-params", "--output-dir"):
            argv[argv.index(flag) + 1] = "<attempt-path>"
        return argv

    if normalized(identity["command"]) != normalized(command):
        raise ValueError("Refusing changed command/protocol for boundary reuse")
    model_path = Path(command[command.index("--model-path") + 1])
    if hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest() != identity["model_config_sha256"]:
        raise ValueError("Refusing changed model config for boundary reuse")
    return identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True,
                        help="Compatible benchmark checkout (see README); not inferred from this package")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate paths and required adapter options without touching GPUs")
    parser.add_argument("--smoke-only", action="store_true",
                        help="Only guarded 4K/64, B2 adapter smoke; never a paper result or capacity boundary")
    parser.add_argument("--export-measurements-dir", type=Path,
                        help="Export non-smoke measurements, including partial campaigns; smoke stays in its run directory")
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--lanes", default="sengine-vanilla,sengine-snapkv,sengine-quest,sengine-omnikv,vllm-vanilla")
    parser.add_argument("--wait-for-release", action="store_true")
    parser.add_argument("--attempt", default="initial")
    parser.add_argument("--hold-reservation-seconds", type=int, default=0,
                        help="Keep the guarded GPU reservation after this sweep (bounded, opt-in)")
    parser.add_argument("--continue-after-contention", action="store_true",
                        help="Finish a single-lane run after foreign GPU activity, then mark it potentially invalid")
    parser.add_argument("--isolated-cache-root", action="store_true",
                        help="Use this sweep attempt's directory for compiler caches")
    parser.add_argument("--no-binary-search", action="store_true",
                        help="Use powers and the configured/slot-derived hint only; retain a lower bound if not adjacent")
    parser.add_argument("--handoff-reservation-pid", type=int,
                        help="Existing same-user reservation guard; release it within 60s of new guard readiness")
    parser.add_argument("--reuse-equivalent-gpus", action="store_true",
                        help="Allow explicit cross-device reuse after identical GPU hardware/config checks")
    parser.add_argument("--reuse-smoke-from", type=Path)
    parser.add_argument("--reuse-cases-from", type=Path)
    parser.add_argument("--reuse-additional-cases-from", type=Path, action="append", default=[],
                        help="Additional raw-validated probe roots, in deterministic priority order")
    parser.add_argument("--probe-concurrency", type=int)
    parser.add_argument("--additional-probe-concurrency", type=int, action="append", default=[],
                        help="Further missing boundary probes in the same guarded probe-only lane")
    parser.add_argument("--complete-from-capacity", type=Path,
                        help="Revalidate a previous partial curve and complete it using only missing probes")
    parser.add_argument("--completion-note",
                        help="Required provenance explanation when assembling previous measurements and new probes")
    parser.add_argument("--probe-only", action="store_true",
                        help="Run one guarded probe; do not claim a complete capacity boundary")
    args = parser.parse_args()
    if not 0 <= args.hold_reservation_seconds <= 86400:
        raise ValueError("Reservation hold must be between 0 and 86400 seconds")
    if args.probe_only and (not args.probe_concurrency or args.probe_concurrency < 1 or "," in args.lanes):
        raise ValueError("probe-only requires one lane and a positive probe concurrency")
    if args.additional_probe_concurrency and (not args.probe_only or min(args.additional_probe_concurrency) < 1):
        raise ValueError("Additional probes require probe-only and positive concurrency")
    if args.complete_from_capacity and (not args.probe_only or not args.completion_note):
        raise ValueError("Completing a partial curve requires probe-only and an explicit provenance note")
    if args.continue_after_contention and "," in args.lanes:
        raise ValueError("Continue-after-contention requires one lane so only the affected result is invalidated")
    REPO = args.repo.resolve()
    # The orchestration script may live outside the selected benchmark checkout.
    # Validate the same statistics module used by raw-artifact checks before GPUs.
    sys.path.insert(0, str(REPO))
    from benchmark.efficiency import metrics
    if Path(metrics.__file__).resolve() != REPO / "benchmark/efficiency/metrics.py":
        raise RuntimeError("Measurement statistics were imported from a different benchmark checkout")
    config = load_campaign_config(args.config)
    idle_poll_interval_s = config.get("idle_poll_interval_s", 15)
    if not isinstance(idle_poll_interval_s, (int, float)) or isinstance(idle_poll_interval_s, bool) or not 0 < idle_poll_interval_s <= 60:
        raise ValueError("idle_poll_interval_s must be in (0, 60] seconds")
    protocol = config.get("measurement_protocol", "step_sync_v1")
    if protocol not in ("step_sync_v1", "boundary_sync_v2"):
        raise ValueError(f"Unknown measurement protocol: {protocol}")
    window_mode = protocol == "boundary_sync_v2"
    if window_mode and args.reuse_smoke_from:
        raise ValueError("Boundary-sync resumes require a fresh smoke")
    if window_mode and (args.reuse_cases_from or args.reuse_additional_cases_from) and args.lanes != "sengine-snapkv":
        raise ValueError("Boundary-sync reuse currently validates native SnapKV only")
    for key in ("output_root", "scratch_root", "conda", "native_env", "vllm_env"):
        if not Path(config[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute path")
    for key in ("conda", "native_env", "vllm_env"):
        if not Path(config[key]).exists():
            raise FileNotFoundError(config[key])
    model_config = Path(config["models"][args.model]["path"]) / "config.json"
    if not model_config.is_absolute() or not model_config.is_file():
        raise FileNotFoundError(f"Model config must exist at an absolute path: {model_config}")
    from plot_decode_capacity import configured_lanes
    for lane in args.lanes.split(","):
        if lane not in configured_lanes(config) or lane in config.get("unsupported", {}).get(args.model, {}):
            raise ValueError(f"Unknown or unsupported combination: {args.model}/{lane}")
        external = config.get("external_lanes", {}).get(lane)
        if external and not Path(external["env"]).is_dir():
            raise FileNotFoundError(external["env"])
    adapter = REPO / "benchmark" / "microbench.py"
    required = ("--engine", "--require_full_decode_batch", "--decode_window_steps" if window_mode else "--synchronize_step_timing",
                "--decode_warmup_steps_after_full", "--admission_wave_size")
    source = adapter.read_text()
    missing = [option for option in required if option not in source]
    if missing:
        raise RuntimeError(f"{adapter} lacks measured stage adapters: {missing}. "
                           "Use the compatible experiment checkout described in README.")
    if args.check_only:
        print("Paths and adapter options validated; no GPU execution or numerical validation performed.")
        return
    model = config["models"][args.model]
    root = Path(config["output_root"]) / args.model / (args.lanes.replace(",", "_") + "-" + args.attempt)
    root.mkdir(parents=True, exist_ok=False)
    write(root / "campaign.json", config)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus,
           "PYTHONPATH": str(REPO) + ":" + str(REPO / "src"),
           "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_NO_USAGE_STATS": "1",
           "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    for key, name in (("VLLM_CACHE_ROOT", "vllm"), ("TRITON_CACHE_DIR", "triton"),
                      ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("CUDA_CACHE_PATH", "cuda"), ("TMPDIR", "scratch")):
        cache_root = root if args.isolated_cache_root else Path(config["output_root"])
        if key == "TMPDIR":
            suffix = hashlib.sha256(str(root).encode()).hexdigest()[:12]
            directory = Path(config["scratch_root"]) / suffix if args.isolated_cache_root else Path(config["scratch_root"])
        else:
            directory = cache_root / "cache" / name
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    native_env = config["native_env"]
    active = None
    guard = None
    case_results = {}

    def stop(signum, frame):
        raise InterruptedError(f"Queue interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def status(stage, state, **extra):
        value = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stage": stage,
                 "status": state, "gpus": args.gpus, **extra}
        print(json.dumps(value), flush=True)
        with (root / "status.tsv").open("a") as out:
            out.write("\t".join(str(value.get(key, "")) for key in ("time", "stage", "status", "gpus")) + "\t" + json.dumps(extra) + "\n")

    def ensure_guard():
        if guard.poll() is not None:
            raise RuntimeError("GPU reservation exited; inspect guard.log")
        if (root / "guard.contention.json").exists() and not args.continue_after_contention:
            raise RuntimeError("External GPU contention invalidates the run")

    def run_case(lane, batch, smoke=False):
        nonlocal active
        ensure_guard()
        if (lane, batch, smoke) in case_results:
            return case_results[(lane, batch, smoke)]
        case = root / lane / ("smoke" if smoke else f"bs{batch}")
        if case.exists():
            raise FileExistsError(f"Refusing to overwrite existing attempt: {case}")
        engine, method, external = resolve_lane(config, lane)
        if external:
            for key, value in external.get("environment", {}).items():
                if key in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH"):
                    raise ValueError(f"External environment cannot override {key}")
                env[key] = value
            env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(REPO / "src"),
                *external.get("pythonpath", [])])
        hp = dict(tensor_parallel_size=model["tp"], expert_parallel_size=model["ep"],
                  data_parallel_size=1, decode_graph=True, gpu_memory_utilization=config["gpu_memory_utilization"],
                  max_num_batched_tokens=config.get("max_num_batched_tokens", 8192),
                  engine_prefill_chunk_size=config.get("engine_prefill_chunk_size", 8192),
                  enable_prefix_caching=False)
        if external:
            chunk = external.get("max_num_batched_tokens", 8192)
            hp.update(max_num_batched_tokens=chunk, engine_prefill_chunk_size=chunk)
        if engine == "sparseengine":
            hp["decode_graph_capture_sizes"] = [batch]
            if method in config["methods"]:
                hp.update(config["methods"][method])
        length, output = (4096, 64) if smoke else (config["input_len"], config["output_len"])
        override = config.get("curve_protocols", {}).get(args.model, {}).get(lane)
        if override and not smoke:
            if (not override.get("reason") or not override.get("label_suffix")
                    or override["input_len"] + override["output_len"] != length + output):
                raise ValueError("Length override requires an explicit reason/label and unchanged total span")
            length, output = override["input_len"], override["output_len"]
        reuse_roots = ([args.reuse_cases_from] if args.reuse_cases_from else []) + args.reuse_additional_cases_from
        available = [root / lane / f"bs{batch}" for root in reuse_roots
                     if (root / lane / f"bs{batch}" / "performance.jsonl").is_file()]
        if available and not smoke and not window_mode:
            previous = available[0]
            artifact = previous / "performance.jsonl"
            if artifact.exists():
                old_hp = json.loads((previous / "hyper_params.json").read_text())
                identity = json.loads((previous / "identity.json").read_text())
                old_command = identity["command"]
                expected_args = {"--model_path": model["path"], "--lengths": str(length),
                                 "--output_len": str(output), "--batch_sizes": str(batch),
                                 "--decode_warmup_steps_after_full": "32", "--engine": engine, "--methods": method}
                if external:
                    expected_args["--backend_label"] = external["backend_label"]
                    if json.loads((previous / "engine_kwargs.json").read_text()) != external["engine_kwargs"]:
                        raise RuntimeError(f"Refusing changed external algorithm parameters: {previous}")
                if (old_hp != hp
                        or any(old_command[old_command.index(key) + 1] != value for key, value in expected_args.items())
                        or "--require_full_decode_batch" not in old_command
                        or "--synchronize_step_timing" not in old_command):
                    raise RuntimeError(f"Refusing incompatible formal reuse: {previous}")
                rows = [json.loads(line) for line in artifact.read_text().splitlines()]
                if len(rows) != 1:
                    raise RuntimeError(f"Invalid reusable artifact: {artifact}")
                row = rows[0]
                if row.get("status") == "success":
                    from plot_decode_capacity import validate_measurement
                    validate_measurement(artifact, batch, config)
                    result = {"concurrency": batch, "artifact": str(artifact), "status": "success"}
                elif capacity_failure(str(row.get("error", "")), previous / "run.log"):
                    result = {"concurrency": batch, "artifact": str(artifact), "status": "capacity_exceeded", "error": row["error"]}
                else:
                    raise RuntimeError(f"Unclassified previous failure: {artifact}")
                status(lane + f"/bs{batch}", "reused_validated", artifact=str(artifact), result=result["status"])
                case_results[(lane, batch, smoke)] = result
                return result
        case.mkdir(parents=True)
        write(case / "hyper_params.json", hp)
        command = [config["conda"], "run", "--no-capture-output", "-p",
                   external["env"] if external else native_env if engine == "sparseengine" else config["vllm_env"],
                   "python", "-u", "benchmark/microbench.py", "--engine", engine,
                   "--model_path", model["path"], "--lengths", str(length),
                   "--output_len", str(output), "--batch_sizes", str(batch),
                   "--methods", method, "--hyper_params", "@" + str(case / "hyper_params.json"),
                   "--synchronize_step_timing", "--require_full_decode_batch",
                   "--decode_warmup_steps_after_full", "8" if smoke else "32",
                   "--output_dir", str(case)]
        if window_mode:
            command = command[:7] + ["benchmark/efficiency/bench_probe.py",
                "--engine", engine, "--model-path", model["path"], "--scenario", "fixed",
                "--prompt-lens", str(length), "--output-lens", str(output),
                "--batch-sizes", str(batch), "--sparse-method", method,
                "--hyper-params", "@" + str(case / "hyper_params.json"),
                "--tensor-parallel-size", str(model["tp"]), "--expert-parallel-size", str(model["ep"]),
                "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
                "--max-num-batched-tokens", str(hp["max_num_batched_tokens"]),
                "--prompt-length-jitter", "0", "--output-length-jitter", "0", "--seed", "42",
                "--decode-only-steps", str(16 if smoke else config["decode_window_steps"]),
                "--decode-only-warmup-steps", str(4 if smoke else config["decode_warmup_steps"]),
                "--num-warmups", str(0 if smoke else config["num_warmups"]),
                "--num-iters", str(1 if smoke else config["num_iters"]), "--output-dir", str(case)]
        if external:
            write(case / "engine_kwargs.json", external["engine_kwargs"])
            if external.get("environment_kind") == "venv":
                prefix = Path(external["env"])
                if not (prefix / "pyvenv.cfg").is_file():
                    raise ValueError(f"Configured venv lacks pyvenv.cfg: {prefix}")
                command = [str(prefix / "bin/python")] + command[6:]
                env["PATH"] = str(prefix / "bin") + os.pathsep + os.environ["PATH"]
                env["VIRTUAL_ENV"] = str(prefix)
            command += ["--engine-kwargs" if window_mode else "--engine_kwargs", "@" + str(case / "engine_kwargs.json"),
                        "--backend-label" if window_mode else "--backend_label", external["backend_label"]]
        admission = config.get("native_admission", {}).get(method,
            {"wave_size": 1, "decode_gap_steps": 1} if method == "snapkv" else {})
        if engine == "sparseengine" and admission.get("wave_size", 0):
            command += ["--prefill-wave-size" if window_mode else "--admission_wave_size", str(admission["wave_size"]),
                        "--wave-decode-gap-steps" if window_mode else "--wave_decode_gap_steps", str(admission.get("decode_gap_steps", 0))]
        if available and not smoke and window_mode:
            previous = available[0]
            reuse_identity = validate_boundary_reuse(previous, config, hp, command, args.gpus,
                allow_equivalent_gpus=args.reuse_equivalent_gpus)
            artifact = previous / "performance.jsonl"
            rows = [json.loads(line) for line in artifact.read_text().splitlines()]
            if len(rows) != 1:
                raise ValueError(f"Invalid reusable artifact: {artifact}")
            row = rows[0]
            if row.get("status") == "success":
                from plot_decode_capacity import validate_measurement
                validate_measurement(artifact, batch, {**config, "input_len": length, "output_len": output})
                result = dict(concurrency=batch, artifact=str(artifact), status="success")
            elif any(capacity_failure(str(row.get("error", "")), log)
                     for log in (previous / "run.log", previous / "failure.log") if log.is_file()):
                result = dict(concurrency=batch, artifact=str(artifact), status="capacity_exceeded", error=row["error"])
            else:
                raise ValueError(f"Unclassified previous failure: {artifact}")
            write(case / "reuse.json", dict(result=result,
                gpu_comparison=reuse_identity.get("reuse_gpu_comparison"),
                identity_sha256=hashlib.sha256((previous / "identity.json").read_bytes()).hexdigest(),
                artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()))
            status(lane + f"/bs{batch}", "reused_validated", artifact=str(artifact), result=result["status"])
            case_results[(lane, batch, smoke)] = result
            return result
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            env["SPARSEENGINE_MASTER_PORT"] = str(listener.getsockname()[1])
        from benchmark.efficiency.paper import source_version, without_source_fingerprints
        version = source_version(REPO)
        write(case / "identity.json", {"command": command, "env": {key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "VLLM_ENABLE_V1_MULTIPROCESSING", "SPARSEENGINE_MASTER_PORT")},
              "git_commit": version["git_head"], "git_dirty": version["git_dirty"],
              "model_config_sha256": hashlib.sha256((Path(model["path"]) / "config.json").read_bytes()).hexdigest()})
        status(str(case.relative_to(root)), "running")
        with (case / "run.log").open("w") as log:
            active = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = active.wait(timeout=config["case_timeout_s"])
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGTERM)
                active.wait(timeout=30)
                raise LaneFailure(f"Case timeout: {case}")
            finally:
                if active.poll() is not None:
                    active = None
        ensure_guard()
        artifact = case / "performance.jsonl"
        try:
            rows = [json.loads(line) for line in artifact.read_text().splitlines()] if artifact.exists() else []
        except json.JSONDecodeError as error:
            raise LaneFailure(f"Malformed measurement: {artifact}: {error}") from error
        if rows and (len(rows) != 1 or not isinstance(rows[0], dict)):
            raise LaneFailure(f"Expected one measurement object: {artifact}")
        row = rows[0] if len(rows) == 1 else {}
        if code == 0 and row.get("status") == "success":
            from plot_decode_capacity import validate_measurement
            validation_config = {**config, "input_len": length, "output_len": output}
            if smoke and window_mode:
                validation_config.update(decode_window_steps=16, decode_warmup_steps=4, num_iters=1)
            try:
                point = validate_measurement(artifact, batch, validation_config)
            except (ValueError, KeyError, TypeError, FileNotFoundError, AssertionError) as error:
                raise LaneFailure(f"Invalid measurement: {artifact}: {error}") from error
            if row.get("actual_decode_peak") != batch or row.get("completed_requests") != batch:
                raise LaneFailure(f"Missing actual concurrency/completion evidence: {case}")
            if row.get("measurement_scope") != ("full_batch_decode_window" if window_mode else "full_batch_pure_decode_steps") or not row.get("decode_stage_throughput_tps", 0) > 0:
                raise LaneFailure(f"Invalid stage metric: {case}")
            status(str(case.relative_to(root)), "success", throughput=row["decode_stage_throughput_tps"])
            if smoke or args.export_measurements_dir:
                destination = (case / "smoke.json" if smoke else
                               args.export_measurements_dir / args.model / args.attempt / lane / f"bs{batch}.json")
                if destination.exists():
                    raise FileExistsError(f"Refusing to overwrite exported measurement: {destination}")
                write(destination, dict(measurement_protocol=protocol, smoke_only=smoke,
                    model=model, lane=lane, config=validation_config, point=point,
                    run_manifest=without_source_fingerprints(json.loads((case / "run_manifest.json").read_text())) if window_mode else None,
                    source_identity_sha256=hashlib.sha256((case / "identity.json").read_bytes()).hexdigest()))
            result = {"concurrency": batch, "artifact": str(artifact), "status": "success"}
            case_results[(lane, batch, smoke)] = result
            return result
        error = str(row.get("error", ""))
        capacity = capacity_failure(error, case / "run.log")
        if window_mode and (case / "failure.log").is_file():
            capacity = capacity or capacity_failure(error, case / "failure.log")
        status(str(case.relative_to(root)), "capacity_exceeded" if capacity else "failed", error=error, exit_code=code)
        if smoke or args.export_measurements_dir:
            destination = (case / "smoke.json" if smoke else
                           args.export_measurements_dir / args.model / args.attempt / lane / f"bs{batch}.json")
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite exported failure: {destination}")
            write(destination, dict(measurement_protocol=protocol, smoke_only=smoke,
                model=model, lane=lane, concurrency=batch, status="capacity_exceeded" if capacity else "failed",
                error=error, exit_code=code, artifact=str(artifact),
                run_log_sha256=hashlib.sha256((case / "run.log").read_bytes()).hexdigest()))
        if smoke or not capacity:
            raise LaneFailure(f"Benchmark failed, inspect {case / 'run.log'}")
        result = {"concurrency": batch, "artifact": str(artifact), "status": "capacity_exceeded", "error": error}
        case_results[(lane, batch, smoke)] = result
        return result

    try:
        status("resource", "waiting_for_idle")
        deadline = time.monotonic() + config["idle_timeout_s"]
        while time.monotonic() < deadline:
            if args.gpus.startswith("auto:"):
                count = int(args.gpus.split(":")[1])
                devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=15)
                occupied = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"], text=True, timeout=15).splitlines()
                idle = []
                for device in devices.splitlines():
                    index, uuid, memory, utilization = [part.strip() for part in device.split(",")]
                    if uuid not in occupied and int(memory) < 100 and int(utilization) == 0:
                        idle.append(index)
                if len(idle) >= count:
                    args.gpus = ",".join(idle[:count])
                    env["CUDA_VISIBLE_DEVICES"] = args.gpus
                    break
                time.sleep(idle_poll_interval_s)
                continue
            pids = subprocess.check_output(["nvidia-smi", "-i", args.gpus, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=15)
            if not [pid for pid in pids.split() if int(pid) != args.handoff_reservation_pid]:
                break
            time.sleep(idle_poll_interval_s)
        else:
            raise RuntimeError("No idle GPU before resource deadline")
        with (root / "guard.log").open("w") as log:
            guard = subprocess.Popen([config["conda"], "run", "--no-capture-output", "-p", native_env,
                "python", "-u", str(PACKAGE / "decode_capacity_guard.py"), "--parent", str(os.getpid()),
                "--ready", str(root / "guard.json")]
                + (["--contention-policy", "mark"] if args.continue_after_contention else [])
                + (["--handoff-reservation-pid", str(args.handoff_reservation_pid)] if args.handoff_reservation_pid else []),
                cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(120):
            ensure_guard()
            if (root / "guard.json").exists():
                break
            time.sleep(1)
        else:
            raise RuntimeError("GPU reservation readiness timeout")
        status("resource", "reserved")
        if args.handoff_reservation_pid:
            deadline = time.monotonic() + 60
            while Path(f"/proc/{args.handoff_reservation_pid}").exists():
                ensure_guard()
                if time.monotonic() >= deadline:
                    raise RuntimeError("Previous reservation guard was not released")
                time.sleep(1)
        lanes = args.lanes.split(",")
        def smoke_lane(lane):
            candidate = config.get("conservative_candidates", {}).get(lane)
            if candidate:
                from plot_decode_capacity import validate_measurement
                artifact = Path(candidate["reference_artifact"])
                validate_measurement(artifact, 1, config)
                status(lane + "/smoke", "reference_validated", artifact=str(artifact),
                       note="Candidate run validates the new concurrency; no extra smoke probe")
                return
            if args.reuse_smoke_from:
                previous = args.reuse_smoke_from / lane / "smoke" / "performance.jsonl"
                if previous.exists() and json.loads(previous.read_text().splitlines()[0]).get("status") == "success":
                    from plot_decode_capacity import validate_measurement
                    validate_measurement(previous, 2, {"input_len": 4096, "output_len": 64})
                    status(lane + "/smoke", "reused_validated", artifact=str(previous))
                    return
            run_case(lane, 2, smoke=True)

        released = False

        def sweep_lane(lane):
            nonlocal released
            if args.smoke_only:
                return
            if args.wait_for_release and not released:
                status("full_sweep", "waiting_for_validated_smoke_release")
                deadline = time.monotonic() + 7200
                while not (Path(config["output_root"]) / "full.ready").exists():
                    ensure_guard()
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Full sweep release deadline exceeded")
                    time.sleep(5)
                released = True
            attempts = []
            candidate = config.get("conservative_candidates", {}).get(lane)
            if candidate:
                batch = int(candidate["concurrency"])
                if not 1 <= batch <= config["safety_concurrency_limit"]:
                    raise LaneFailure("Conservative candidate exceeds configured limits")
                result = run_case(lane, batch)
                evidence = dict(status="completed" if result["status"] == "success" else "failed",
                    model=args.model, lane=lane, max_concurrency=None,
                    verified_concurrency=batch if result["status"] == "success" else None,
                    first_failed_concurrency=None, maximum_verified=False,
                    selection="conservative_kv_slots", estimate=candidate, attempts=[result])
                write(root / lane / "capacity.json", evidence)
                if result["status"] != "success":
                    raise LaneFailure("Conservative candidate failed; no automatic search or retry")
                status(lane, "completed", verified_concurrency=batch, maximum_verified=False)
                return
            if args.probe_concurrency and lane == lanes[0]:
                probes = [run_case(lane, args.probe_concurrency)]
                if args.probe_only:
                    for batch in args.additional_probe_concurrency:
                        probes.append(run_case(lane, batch))
                    if args.complete_from_capacity:
                        previous = args.complete_from_capacity
                        if json.loads((previous.parent.parent / "campaign.json").read_text()) != config:
                            raise RuntimeError("Cannot assemble probes from a different campaign")
                        if previous.parent.name != lane:
                            raise RuntimeError("Cannot assemble probes from a different method")
                        from plot_decode_capacity import validate_measurement
                        attempts = json.loads(previous.read_text())["attempts"]
                        for attempt in attempts:
                            artifact = Path(attempt["artifact"])
                            if attempt["status"] == "success":
                                validate_measurement(artifact, attempt["concurrency"], config)
                            elif attempt["status"] == "capacity_exceeded":
                                row = json.loads(artifact.read_text())
                                if not capacity_failure(str(row.get("error", "")), artifact.parent / "run.log"):
                                    raise RuntimeError("Previous capacity failure lacks explicit raw evidence")
                            else:
                                raise RuntimeError("Unclassified previous failure cannot establish capacity")
                        attempts = attempts + probes
                        maximum, upper = probe_capacity_boundary(attempts)
                        write(root / lane / "capacity.json", dict(status="completed", model=args.model,
                            lane=lane, max_concurrency=maximum, first_failed_concurrency=upper,
                            attempts=attempts, assembled_from=str(previous),
                            previous_capacity_sha256=hashlib.sha256(previous.read_bytes()).hexdigest(),
                            completion_note=args.completion_note))
                        status(lane, "completed", max_concurrency=maximum, assembled_from=str(previous))
                    return
            lower, upper, batch = 0, None, 1
            hint = None
            while batch <= config["safety_concurrency_limit"]:
                result = run_case(lane, batch)
                attempts.append(result)
                write(root / lane / "capacity.json", {"status": "running", "attempts": attempts})
                if result["status"] != "success":
                    upper = batch
                    break
                lower = batch
                if args.no_binary_search and hint and batch == hint + 1:
                    write(root / lane / "capacity.json", {"status": "partial", "model": args.model,
                          "lane": lane, "max_concurrency": None, "verified_concurrency": lower,
                          "first_failed_concurrency": None, "maximum_verified": False,
                          "selection": "slot_hint_no_binary", "attempts": attempts,
                          "reason": "The direct hint and hint+1 both passed; binary search disabled."})
                    status(lane, "completed_lower_bound", verified_concurrency=lower,
                           first_failed_concurrency=None, maximum_verified=False)
                    return
                if batch == 1:
                    configured_hint = config.get("capacity_hints", {}).get(args.model, {}).get(lane)
                    if configured_hint:
                        hint = int(configured_hint["concurrency"])
                        tokens_per_sequence = int(configured_hint.get("tokens_per_sequence", 0))
                        slots = minimum_kv_slots(root / lane / "bs1")
                        slot_upper = slots // tokens_per_sequence if slots and tokens_per_sequence > 0 else None
                        if (hint < 1 or hint > config["safety_concurrency_limit"]
                                or not configured_hint.get("basis") or slot_upper is None or hint > slot_upper):
                            raise LaneFailure("Configured capacity hint lacks compatible current BS1 slot evidence")
                        status(lane, "capacity_hint", concurrency=hint,
                               basis=configured_hint["basis"], source="configured_conservative_hint",
                               current_kv_slots=slots, slot_quotient_upper=slot_upper,
                               tokens_per_sequence=tokens_per_sequence)
                    elif lane in ("sengine-vanilla", "sengine-quest", "sengine-omnikv", "vllm-vanilla", "vortex-quest"):
                        hint = full_kv_capacity_hint(root / lane / "bs1", config["input_len"] + config["output_len"])
                        if hint and hint >= lower:
                            status(lane, "capacity_hint", concurrency=hint,
                                   basis="minimum logged full KV slots / request total span; not a verified limit",
                                   source="current_bs1_slots")
                if hint and lower < hint < batch * 2:
                    batch = hint
                elif hint and lower == hint:
                    batch = hint + 1
                else:
                    batch = 1 << lower.bit_length()
            if upper is None or lower == 0:
                raise LaneFailure(f"No validated nonzero capacity boundary for {lane}")
            if args.no_binary_search and upper - lower > 1:
                write(root / lane / "capacity.json", {"status": "partial", "model": args.model,
                      "lane": lane, "max_concurrency": None, "verified_concurrency": lower,
                      "first_failed_concurrency": upper, "maximum_verified": False,
                      "selection": "slot_hint_no_binary", "attempts": attempts,
                      "reason": "Direct slot/hint probes did not establish adjacent max/max+1; binary search disabled."})
                status(lane, "completed_lower_bound", verified_concurrency=lower,
                       first_failed_concurrency=upper, maximum_verified=False)
                return
            while upper - lower > 1:
                batch = (upper + lower) // 2
                result = run_case(lane, batch)
                attempts.append(result)
                if result["status"] == "success":
                    lower = batch
                else:
                    upper = batch
                write(root / lane / "capacity.json", {"status": "running", "attempts": attempts})
            write(root / lane / "capacity.json", {"status": "completed", "model": args.model,
                  "lane": lane, "max_concurrency": lower, "first_failed_concurrency": upper, "attempts": attempts})
            status(lane, "completed", max_concurrency=lower)

        def record_failure(lane, failure):
            path = root / lane / "capacity.json"
            evidence = json.loads(path.read_text()) if path.exists() else {"attempts": []}
            evidence.update(status="failed", failure=failure)
            write(path, evidence)
            write(root / lane / "lane_failure.json", failure)
            if args.export_measurements_dir:
                write(args.export_measurements_dir / args.model / args.attempt / lane / "lane_failure.json", failure)
            status(lane, "failed", **failure)

        failures = run_lane_stages(lanes, smoke_lane, sweep_lane, record_failure, ensure_guard)
        contention_path = root / "guard.contention.json"
        contention = json.loads(contention_path.read_text()) if contention_path.exists() else None
        if contention:
            lane = lanes[0]
            failure = {"phase": "contention",
                       "error": "External GPU activity was observed; measurements completed but are potentially invalid.",
                       "artifact": str(contention_path)}
            if lane not in failures:
                failures[lane] = failure
                record_failure(lane, failure)
            capacity_path = root / lane / "capacity.json"
            if capacity_path.exists():
                capacity = json.loads(capacity_path.read_text())
                capacity.update(validity_status="potentially_invalid_external_contention",
                                contention_artifact=str(contention_path))
                write(capacity_path, capacity)
            validity = {"status": "potentially_invalid_external_contention",
                        "contention_artifact": str(contention_path)}
            write(root / lane / "validity.json", validity)
            if args.export_measurements_dir:
                export_lane = args.export_measurements_dir / args.model / args.attempt / lane
                write(export_lane / "validity.json", validity)
                for measurement_path in export_lane.glob("bs*.json"):
                    measurement = json.loads(measurement_path.read_text())
                    measurement.update(validity_status=validity["status"],
                                       contention_artifact=str(contention_path))
                    write(measurement_path, measurement)
        write(root / "queue_summary.json", {"status": "failed" if failures else "completed",
              "lanes": lanes, "failures": failures, "external_contention": contention})
        if failures:
            raise RuntimeError(f"Lane failures retained after continuing other methods: {list(failures)}")
        status("queue", "smoke_completed" if args.smoke_only else "probe_completed" if args.probe_only else "completed")
    except BaseException as error:
        status("queue", "failed", error=repr(error))
        raise
    finally:
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            active.wait(timeout=30)
        try:
            if args.hold_reservation_seconds and guard is not None and guard.poll() is None:
                ensure_guard()
                status("resource", "holding_after_run", seconds=args.hold_reservation_seconds)
                deadline = time.monotonic() + args.hold_reservation_seconds
                while time.monotonic() < deadline:
                    ensure_guard()
                    time.sleep(min(5, max(0, deadline - time.monotonic())))
        finally:
            if guard is not None and guard.poll() is None:
                os.killpg(guard.pid, signal.SIGTERM)
                guard.wait(timeout=30)


if __name__ == "__main__":
    main()
