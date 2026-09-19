"""Guarded ABBA comparison using the canonical full-batch decode microbench."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from plot_decode_capacity import validate_measurement

PACKAGE = Path(__file__).resolve().parent
FLAG = "SPARSEENGINE_DISABLE_TILE_MLA_SPLIT_PROFILE"


def require_experiment_patch(repo):
    source = repo / "src/sparseengine/kernels/tilelang/mla/runtime.py"
    if FLAG not in source.read_text():
        raise ValueError("This historical ablation requires the archived profile-bypass.patch "
                         "on its measured runtime source; the temporary engine switch was removed.")


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(os.path.expandvars(args.config.read_text()))
    repo, root = Path(config["repo"]), Path(config["output_root"])
    require_experiment_patch(repo)
    if any(case["variant"] not in ("profile", "fixed32") for case in config["cases"]):
        raise ValueError("Unknown ablation variant")
    root.mkdir(parents=True, exist_ok=False)
    save(root / "config.json", config)
    env = dict(os.environ, PYTHONPATH=f"{repo}:{repo / 'src'}", HF_HUB_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false")
    env.pop("SPARSEENGINE_PLATFORM", None)
    for key in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH", "TILELANG_CACHE_DIR"):
        path = root / "cache" / key.lower()
        path.mkdir(parents=True)
        env[key] = str(path)
    Path(config["scratch_root"]).mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = config["scratch_root"]
    prefix = [config["conda"], "run", "--no-capture-output", "-p", config["native_env"], "python"]
    active = guard = None
    rows = []

    def status(stage, state, **extra):
        value = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stage": stage, "status": state, **extra}
        print(json.dumps(value), flush=True)
        with (root / "status.tsv").open("a") as handle:
            handle.write(f"{value['time']}\t{stage}\t{state}\t{json.dumps(extra)}\n")

    def stop(signum, frame):
        raise InterruptedError(f"Ablation interrupted by signal {signum}")

    def execute(command, log, case_env):
        nonlocal active
        if guard.poll() is not None or (root / "guard.contention.json").exists():
            raise RuntimeError("GPU reservation lost; inspect guard log")
        with log.open("w") as handle:
            active = subprocess.Popen(command, cwd=repo, env=case_env, stdout=handle,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            code = active.wait(timeout=1800)
        active = None
        if code or guard.poll() is not None:
            raise RuntimeError(f"Command failed ({code}): {log}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        status("resource", "waiting_for_idle")
        for _ in range(240):
            occupied = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"], text=True).splitlines()
            devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
            idle = []
            for row in devices.splitlines():
                index, uuid, memory, utilization = [x.strip() for x in row.split(",")]
                if uuid not in occupied and int(memory) < 100 and int(utilization) == 0:
                    idle.append(index)
            if len(idle) >= 2:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(idle[:2])
                break
            time.sleep(15)
        else:
            raise RuntimeError("No idle GPU pair within one hour")
        with (root / "guard.log").open("w") as handle:
            guard = subprocess.Popen(prefix + [str(PACKAGE / "decode_capacity_guard.py"),
                "--parent", str(os.getpid()), "--ready", str(root / "guard.json")],
                cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(120):
            if guard.poll() is not None:
                raise RuntimeError("GPU guard failed")
            if (root / "guard.json").exists():
                break
            time.sleep(1)
        else:
            raise RuntimeError("GPU guard readiness timeout")
        save(root / "identity.json", {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()),
            "gpus": env["CUDA_VISIBLE_DEVICES"], "flag": FLAG})
        status("resource", "reserved", gpus=env["CUDA_VISIBLE_DEVICES"])
        for variant in ("profile", "fixed32"):
            case_env = dict(env, **{FLAG: "0" if variant == "profile" else "1"})
            command = prefix + [str(PACKAGE / "validate_mla_profile_ablation.py"), str(root / f"oracle-{variant}.json")]
            status(f"oracle-{variant}", "running", command=command)
            execute(command, root / f"oracle-{variant}.log", case_env)
            oracle = json.loads((root / f"oracle-{variant}.json").read_text())
            expected = 8 if variant == "profile" else 32
            if oracle["status"] != "success" or oracle["plan"]["batch_configs"][-1]["num_split"] != expected:
                raise ValueError(f"Numerical oracle did not exercise the requested {variant} launch plan")
            status(f"oracle-{variant}", "success")
        for case in config["cases"]:
            name, batch, variant = case["name"], case["batch"], case["variant"]
            path = root / name
            path.mkdir()
            hp = dict(config["hyper_params"], decode_graph_capture_sizes=[batch])
            save(path / "hyper_params.json", hp)
            length, output = (4096, 64) if case.get("smoke") else (131072, 2048)
            case_env = dict(env, **{FLAG: "0" if variant == "profile" else "1"})
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                case_env["SPARSEENGINE_MASTER_PORT"] = str(listener.getsockname()[1])
            command = prefix + ["-u", "benchmark/microbench.py", "--engine", "sparseengine",
                "--model_path", config["model"], "--lengths", str(length), "--output_len", str(output),
                "--batch_sizes", str(batch), "--methods", "omnikv", "--hyper_params", "@" + str(path / "hyper_params.json"),
                "--synchronize_step_timing", "--require_full_decode_batch", "--decode_warmup_steps_after_full", "8" if case.get("smoke") else "32",
                "--output_dir", str(path)]
            save(path / "command.json", {"command": command, "env": {key: case_env[key] for key in
                (FLAG, "CUDA_VISIBLE_DEVICES", "SPARSEENGINE_MASTER_PORT", "PYTHONPATH")}})
            status(name, "running", variant=variant, batch=batch)
            execute(command, path / "run.log", case_env)
            measured = validate_measurement(path / "performance.jsonl", batch, {"input_len": length, "output_len": output})
            if not case.get("smoke"):
                rows.append(dict(name=name, variant=variant, **measured))
                save(root / "measurements.json", rows)
            status(name, "success", throughput=measured["decode_throughput_tps"])
        save(root / "summary.json", {"status": "completed", "rows": rows})
        status("queue", "completed")
    except BaseException as error:
        status("queue", "failed", error=repr(error))
        raise
    finally:
        for process in (active, guard):
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)


if __name__ == "__main__":
    main()
