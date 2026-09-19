"""Preparing jobs must use the chosen checkout without depending on source copies."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def run(script, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / script), *map(str, args)],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=30,
    )


def load(script):
    spec = importlib.util.spec_from_file_location("recipe", ROOT / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_triton_jobs_execute_from_checkout(tmp_path):
    root = tmp_path / "run"
    run("triton_mla_sm_schedule/prepare.py", "--repo", REPO, "--root", root,
        "--conda", sys.executable, "--env", tmp_path)
    jobs = [read(p) for p in (root / "queue/jobs").glob("*.json")]
    assert jobs
    for job in jobs:
        assert Path(job["cwd"]) == REPO
        assert str(REPO / "src") in job["env"]["PYTHONPATH"].split(os.pathsep)
    assert not (root / "source").exists()
    assert not list(root.glob("*.patch"))
    assert not list(root.glob("*.tar.gz"))


def test_related_jobs_keep_parameters_and_use_selected_checkout(tmp_path):
    baseline, root = tmp_path / "baseline", tmp_path / "run"
    hp = tmp_path / "parameters.json"
    write(hp, {"budget": 128})
    for name in ("04_glm_common8", "06_glm_wave64"):
        write(baseline / "queue" / name / "command.json", {
            "command": ["python", "benchmark/microbench.py", "--output-dir", "old",
                        "--monitor-gpus", "old", "--hyper-params", "@" + str(hp)],
            "cwd": "old/source", "env": {"PYTHONPATH": "old/source"},
        })
    run("triton_mla_sm_schedule/prepare_related.py", "--baseline-root", baseline,
        "--updated-source", REPO, "--root", root, "--gpu", "7")
    for path in (root / "queue/jobs").glob("*.json"):
        job = read(path)
        assert Path(job["cwd"]) == REPO
        command = job["command"]
        assert command[command.index("--monitor-gpus") + 1] == "7"
        assert read(Path(command[command.index("--hyper-params") + 1][1:])) == read(hp)
    assert not (root / "source").exists()


def test_vortex_jobs_keep_checkout_import_paths(tmp_path):
    config = tmp_path / "base.json"
    write(config, {"measurement_protocol": "boundary_sync_v2", "output_len": 2048,
                   "models": {"qwen3-30b-fp8": {"tp": 1}, "glm4.7-flash": {"tp": 2}}})
    root = tmp_path / "run"
    run("sparse_decode_efficiency/prepare_vortex.py", "--base-config", config,
        "--vortex-repo", REPO, "--vortex-env", tmp_path, "--overlay", tmp_path,
        "--output-root", root, "--export-data-dir", tmp_path / "export")
    for job in read(root / "jobs.json"):
        command = job["command"]
        assert Path(command[command.index("--repo") + 1]) == REPO
        config = read(Path(command[command.index("--config") + 1]))
        assert str(REPO) in config["external_lanes"]["vortex-quest"]["pythonpath"]
    assert not (root / "source").exists()
    assert not (root / "vortex").exists()


def test_boundary_preflight_uses_checkout_without_copying_it(tmp_path, monkeypatch):
    module = load("sparse_decode_efficiency/run_boundary_sync.py")
    root, config = tmp_path / "run", tmp_path / "config.json"
    write(config, {"measurement_protocol": "boundary_sync_v2", "output_root": str(root),
                   "models": {"example": {"tp": 1}}})
    calls = []
    monkeypatch.setattr(module, "subprocess", SimpleNamespace(
        check_output=subprocess.check_output,
        run=lambda command, **kw: calls.append(command),
    ))
    module.prepare(SimpleNamespace(config=config, run_root=root, data_root=tmp_path / "data"))
    command, = calls
    assert "--check-only" in command
    assert Path(command[1]).is_file()
    assert Path(command[command.index("--repo") + 1]) == REPO
    assert Path(read(root / "manifest.json")["repo"]) == REPO
    assert not (root / "source").exists()


def test_session_control_scripts_resolve_from_selected_checkout(tmp_path):
    config = read(ROOT / "sparseengine_vs_vortex/session/campaign.json")
    config["quality"] = config["quality"][:1]
    config["efficiency32"] = []
    cfg = tmp_path / "campaign.json"
    write(cfg, config)
    paths = {key: str(tmp_path) for key in (
        "model_root", "dataset", "conda", "native_env", "vllm_env", "vortex_env",
        "tangram_env", "hisparse_env", "vortex_repo", "vortex_overlay", "cuda_home", "scratch_root",
    )}
    prepared = tmp_path / "prepared"
    for model in config["models"].values():
        (tmp_path / model["name"]).mkdir()
        write(prepared / model["name"] / "prepared_samples.json", {})
    paths_file = tmp_path / "paths.json"
    write(paths_file, paths)
    root = tmp_path / "run"
    run("sparseengine_vs_vortex/session/build_campaign.py", "--config", cfg,
        "--paths", paths_file, "--root", root, "--prepared", prepared,
        "--source", REPO, "--gpus", "0")
    jobs = read(root / "queued_commands.json")
    assert jobs
    for job in jobs:
        assert Path(job["cwd"]) == REPO
        assert Path(job["command"][1]).is_file()
        assert Path(job["command"][1]).is_relative_to(REPO)
    assert not (root / "control").exists()
    assert not (root / "source_v2").exists()
