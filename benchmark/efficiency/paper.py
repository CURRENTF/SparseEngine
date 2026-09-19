"""Paper-window orchestration for bench_probe; reuse canonical engine adapters."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmark.efficiency.metrics import aggregate_decode_windows


def source_version(repo):
    """Record Git identity without hashing files or enforcing source equality."""
    if (repo / ".git").exists():
        return dict(git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            git_dirty=bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo, text=True).strip()))
    snapshot = repo.parent / "manifest.json"
    if snapshot.is_file():
        manifest = json.loads(snapshot.read_text())
        return dict(git_head=manifest["git_head"], git_dirty=manifest.get("git_dirty"))
    return dict(git_head=None, git_dirty=None)


def without_source_fingerprints(value):
    """Export legacy run metadata without recursive source-hash inventories.

    Repetition source_sha256 maps identify raw measurement files, not code.
    Preserve them, along with model/config/trace and result checksums.
    """
    if isinstance(value, list):
        return [without_source_fingerprints(item) for item in value]
    if not isinstance(value, dict):
        return value
    removed = {"runtime_source_sha256", "orchestrator_sha256", "dirty_patch_sha256",
               "package_file_sha256", "plot_script_sha256", "classifier_sha256",
               "baseline_source_sha256", "changed_source_sha256", "source_archive_sha256",
               "export_recipe_sha256", "measured_runtime_sha256", "restored_runtime_sha256",
               "measured_recipe_archive_sha256"}
    if not ("repetition" in value and "window" in value):
        removed.add("source_sha256")
    if "repo" in value and "git_head" in value:
        removed.add("sha256")
    if "source" in str(value.get("archive", "")):
        removed.add("archive_sha256")
    return {key: without_source_fingerprints(item) for key, item in value.items()
            if key not in removed}


def record_package_source(module, case):
    """Record the installed package location and its Git version when available."""
    case.mkdir(parents=True, exist_ok=True)
    source = Path(module.__file__).resolve()
    identity = dict(package_file=str(source))
    checkout = next((p for p in source.parents if (p / ".git").exists()), None)
    if checkout is not None:
        identity.update(source_version(checkout))
        identity["checkout"] = str(checkout)
    (case / "external_source.json").write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def run_paper_decode(args):
    from benchmark.efficiency.bench_probe import _parse_json_arg

    if args.scenario != "fixed" or args.prompt_length_jitter or args.output_length_jitter:
        raise ValueError("Paper windows currently require fixed batch and zero length jitter")
    if (args.decode_only_steps < 1 or args.decode_only_warmup_steps < 1
            or args.num_iters < 1 or args.num_warmups < 0):
        raise ValueError("Invalid paper window/repetition count")
    if args.seed != 42:
        raise ValueError("Current shared stage adapters use seed=42; extend them before changing the seed")
    if any(n <= 0 for n in args.prompt_lens + args.output_lens + args.batch_sizes):
        raise ValueError("Lengths and batches must be positive")
    if (args.tensor_parallel_size < 1 or args.expert_parallel_size < 1
            or not 0 < args.gpu_memory_utilization <= 1 or args.max_num_batched_tokens < 1
            or args.prefill_wave_size < 0 or args.wave_decode_gap_steps < 0):
        raise ValueError("Invalid topology, memory or admission parameters")
    if args.wave_decode_gap_steps and not args.prefill_wave_size:
        raise ValueError("Wave gap requires wave admission")
    if args.engine != "sparseengine" and args.prefill_wave_size:
        raise ValueError("External adapters do not implement wave admission")
    hp = _parse_json_arg(args.hyper_params)
    extra = _parse_json_arg(args.engine_kwargs)
    if args.engine == "sparseengine" and extra:
        raise ValueError("Native engine does not consume external engine kwargs")
    explicit = dict(tensor_parallel_size=args.tensor_parallel_size,
        expert_parallel_size=args.expert_parallel_size, data_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization, decode_graph=True,
        enable_prefix_caching=False, max_num_batched_tokens=args.max_num_batched_tokens)
    hp.setdefault("engine_prefill_chunk_size",
                  8192 if args.engine == "sparseengine" else args.max_num_batched_tokens)
    for key, value in explicit.items():
        if key in hp and hp[key] != value:
            raise ValueError(f"Conflicting paper parameter: {key}")
        hp[key] = value
    if args.sparse_prefill_score_mode:
        hp["sparse_prefill_score_mode"] = args.sparse_prefill_score_mode
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "run_manifest.json").exists() or (root / "performance.jsonl").exists():
        raise FileExistsError(f"Refusing to overwrite paper results: {root}")
    repo = Path(__file__).resolve().parents[2]
    commands = []
    manifest = dict(protocol="boundary_sync_v2", workload="repeated_token_100",
        seed=args.seed, args=vars(args), commands=commands,
        repetition_lifecycle="fresh_engine_per_workload; discarded warmups prime persistent compile caches",
        **source_version(repo))
    manifest["prompt_token_ids_sha256"] = {
        str(n): hashlib.sha256((100).to_bytes(4, "little") * n).hexdigest() for n in args.prompt_lens}
    manifest["trace_encoding"] = "uint32 little endian; identical prompt for each request in batch"
    manifest["versions"] = {}
    for package in ("torch", "triton", "transformers", "vllm", "sglang"):
        try:
            manifest["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            manifest["versions"][package] = "not installed in this interpreter"
    manifest["gpu_inventory"] = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
        "--format=csv,noheader"], text=True, timeout=15)
    def write(name, value):
        (root / name).write_text(json.dumps(value, indent=2) + "\n")
    write("run_manifest.json", manifest)
    all_rows = []
    try:
        for length in args.prompt_lens:
            for output in args.output_lens:
                for batch in args.batch_sizes:
                    case_hp = dict(hp)
                    if args.engine == "sparseengine":
                        case_hp["decode_graph_capture_sizes"] = [batch]
                    rows = []
                    for index in range(-args.num_warmups, args.num_iters):
                        case = root / f"p{length}-o{output}-b{batch}" / (
                            f"warmup{-index}" if index < 0 else f"repeat{index}")
                        command = [sys.executable, "-u", "benchmark/microbench.py",
                            "--engine", args.engine, "--model_path", args.model_path,
                            "--lengths", str(length), "--output_len", str(output),
                            "--batch_sizes", str(batch), "--methods", args.sparse_method,
                            "--hyper_params", json.dumps(case_hp), "--require_full_decode_batch",
                            "--decode_window_steps", str(args.decode_only_steps),
                            "--decode_warmup_steps_after_full", str(args.decode_only_warmup_steps),
                            "--output_dir", str(case)]
                        if args.prefill_wave_size:
                            command += ["--admission_wave_size", str(args.prefill_wave_size),
                                        "--wave_decode_gap_steps", str(args.wave_decode_gap_steps)]
                        if args.engine != "sparseengine":
                            command += ["--engine_kwargs", json.dumps(extra)]
                            if args.backend_label:
                                command += ["--backend_label", args.backend_label]
                        commands.append(command)
                        write("run_manifest.json", manifest)
                        case.parent.mkdir(parents=True, exist_ok=True)
                        with (case.parent / f"{case.name}.log").open("x") as log:
                            process = subprocess.run(command, cwd=repo, env=os.environ.copy(),
                                                     stdout=log, stderr=subprocess.STDOUT)
                        artifact = case / "performance.jsonl"
                        records = [json.loads(line) for line in artifact.read_text().splitlines()] if artifact.is_file() else []
                        if process.returncode or len(records) != 1 or records[0]["status"] != "success":
                            error = records[0].get("error", "") if len(records) == 1 else "missing repetition results"
                            # Preserve worker errors as well: SGLang can terminate its
                            # parent before the parent writes a structured result.
                            with (root / "failure.log").open("w") as failure:
                                failure.write((case.parent / f"{case.name}.log").read_text())
                            failed = dict(status="model_failed", error=error,
                                          child_exitcode=process.returncode, artifact=str(artifact))
                            (root / "performance.jsonl").write_text(json.dumps(failed) + "\n")
                            raise RuntimeError(f"Paper repetition failed: {case}: {error}")
                        if index >= 0:
                            rows.append({**records[0], "artifact": str(case / "performance.jsonl"),
                                         "repetition": index})
                    aggregate = aggregate_decode_windows(rows)
                    all_rows.append(aggregate)
        # Preserve JSONL format, including when the CLI requests several cases.
        (root / "performance.jsonl").write_text("".join(json.dumps(r) + "\n" for r in all_rows))
        write("run_status.json", {"status": "completed"})
    except BaseException as error:
        write("run_status.json", {"status": "failed", "error": repr(error)})
        raise
