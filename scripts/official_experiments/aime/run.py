"""Run the AIME recipe through the canonical MathBench entrypoint."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def validate_data(path, expected):
    text = path.read_text()
    rows = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    if not isinstance(rows, list) or len(rows) != expected:
        raise ValueError(f"Expected exactly {expected} AIME problems")
    normalized = []
    problems = set()
    for index, row in enumerate(rows):
        problem = row.get("Problem", row.get("problem", row.get("question")))
        answer = str(row.get("Answer", row.get("answer", ""))).strip()
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"Invalid problem at row {index}")
        if not answer.isascii() or not answer.isdigit() or not 0 <= int(answer) <= 999:
            raise ValueError(f"Invalid integer answer at row {index}")
        if problem.strip() in problems:
            raise ValueError(f"Duplicate problem at row {index}")
        problems.add(problem.strip())
        normalized.append({"id": str(index), "Problem": problem.strip(), "Answer": answer})
    return normalized


def validate_results(folder, expected_ids):
    for name in ("aime2024.jsonl", "aime2024_parsed_outputs.jsonl", "aime2024_per_sample_results.jsonl"):
        rows = [json.loads(line) for line in (folder / name).read_text().splitlines() if line.strip()]
        ids = [row["id"] for row in rows]
        if len(ids) != len(expected_ids) or set(ids) != set(expected_ids):
            raise ValueError(f"Incomplete or duplicate sample coverage: {name}")
        if any(row["status"] not in {"success", "parse_failed"} for row in rows):
            raise ValueError(f"Invalid input, generation or metric failure: {name}")
    result = read(folder / "result.json")["aime2024"]
    samples = [json.loads(line) for line in (folder / "aime2024_per_sample_results.jsonl").read_text().splitlines() if line.strip()]
    correct = sum(row["correct"] for row in samples)
    if result["total"] != len(expected_ids) or result["correct"] != correct:
        raise ValueError("Aggregate counts disagree with per-sample results")
    if result["pass@1"] != round(100 * correct / len(expected_ids), 2):
        raise ValueError("Aggregate pass@1 disagrees with per-sample results")
    return result


def idle_gpus(gpus, expected):
    ids = gpus.split(",")
    if len(ids) != expected or len(set(ids)) != expected or not all(x.isdigit() for x in ids):
        raise ValueError(f"Supply {expected} distinct numeric GPU indices")
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    rows = {parts[0]: parts for line in raw.splitlines() if (parts := [x.strip() for x in line.split(",")])}
    for gpu in ids:
        row = rows[gpu]
        if row[1] in apps or int(row[2]) > 128 or int(row[3]) != 0:
            raise RuntimeError(f"GPU {gpu} is busy; wait and rerun with idle GPUs")
    return {"devices": raw, "compute_processes": apps}


def run_command(command, env, log, timeout):
    process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        returncode = process.wait(timeout=timeout)
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
    finally:
        # TP workers belong to this new process group, including on timeout.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 10
        while True:
            process.poll()  # Reap the leader without mistaking its exit for group exit.
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            time.sleep(min(0.1, remaining))
        process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", type=Path, default=HERE / "setting.json")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Local AIME 2024 JSON/JSONL export")
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--methods", nargs="+", help="Default: all methods in setting.json")
    parser.add_argument("--timeout", type=int, default=43200, help="Maximum seconds per method")
    parser.add_argument("--execute", action="store_true", help="Without this flag, print commands only")
    args = parser.parse_args()
    setting = read(args.setting)
    evaluation = setting["evaluation"]
    methods = args.methods or list(setting["methods"])
    if len(set(methods)) != len(methods) or any(m not in setting["methods"] for m in methods):
        raise ValueError("Methods must be distinct setting.json labels")
    if args.timeout <= 0:
        raise ValueError("Timeout must be positive")
    model = args.model.resolve(strict=True)
    data = validate_data(args.data, evaluation["expected_samples"])
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {root}")
    commands = {}
    for method in methods:
        command = [sys.executable, str(REPO / "benchmark/math_bench/pred.py"),
                   "--model", "aime", "--model_path", str(model), "--task", evaluation["task"],
                   "--ws", "1", "--data_path_aime2024", str(root / "dataset.json"),
                   "--sparse_method", setting["methods"][method]["sparse_method"],
                   "--hyper_param", str(root / method / "engine.json"),
                   "--max_model_len", str(setting["engine"]["max_model_len"]),
                   "--no_force_think_prefix", "--no_prompt_think_instruction"]
        for key in ("batch_size", "max_new_tokens", "temperature", "top_p", "top_k", "prompt_style"):
            command.extend([f"--{key}", str(evaluation[key])])
        commands[method] = command
        print(method + ": " + shlex.join(command), flush=True)
    if not args.execute:
        return
    if evaluation["seed"] != 42:
        raise ValueError("Canonical MathBench currently fixes the seed to 42")
    if importlib.metadata.version("math-verify") != evaluation["math_verify_version"]:
        raise ValueError("Activate an environment with the configured math-verify version")
    hardware = idle_gpus(args.gpus, setting["engine"]["tensor_parallel_size"])
    root.mkdir(parents=True)
    write(root / "setting.json", setting)
    write(root / "dataset.json", data)
    manifest = {"status": "running", "commands": commands, "model": str(model),
                "python": sys.executable, "gpus": args.gpus, "hardware": hardware,
                "timeout_seconds": args.timeout,
                "dataset_sha256": hashlib.sha256((root / "dataset.json").read_bytes()).hexdigest(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True),
                "methods": {}}
    write(root / "manifest.json", manifest)
    try:
        for method, command in commands.items():
            directory = root / method
            directory.mkdir()
            write(directory / "engine.json", setting["engine"] | setting["methods"][method])
            manifest["methods"][method] = {"status": "running", "hardware": idle_gpus(args.gpus, setting["engine"]["tensor_parallel_size"])}
            write(root / "manifest.json", manifest)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus,
                       SPARSEENGINE_OUTPUT_DIR=str(directory), ENABLE_THINKING="1" if evaluation["enable_thinking"] else "0")
            with (directory / "run.log").open("w") as log:
                run_command(command, env, log, args.timeout)
            folders = list(directory.glob("benchmark/math_bench/pred/aime/*"))
            if len(folders) != 1:
                raise ValueError(f"Expected one prediction directory, found {folders}")
            result = validate_results(folders[0], [row["id"] for row in data])
            manifest["methods"][method].update(status="success", result=result, artifacts=str(folders[0]))
            write(root / "manifest.json", manifest)
        manifest["status"] = "success"
        write(root / "final_summary.json", {m: manifest["methods"][m]["result"] for m in methods})
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        for value in manifest["methods"].values():
            if value["status"] == "running":
                value["status"] = "failed"
        raise
    finally:
        write(root / "manifest.json", manifest)


if __name__ == "__main__":
    main()
