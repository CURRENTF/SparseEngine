import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_log_module(log_level: str):
    env = dict(os.environ)
    env["LOG_LEVEL"] = log_level
    python_path = env.get("PYTHONPATH")
    paths = [str(REPO_ROOT / "src")]
    if python_path:
        paths.append(python_path)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from sparseengine.utils.log import log_level; "
                "print(log_level)"
            ),
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )


def test_lowercase_log_level_is_normalized():
    completed = _import_log_module("info")

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "INFO"


def test_invalid_log_level_still_fails_explicitly():
    completed = _import_log_module("not-a-level")

    assert completed.returncode != 0
    assert "Level 'NOT-A-LEVEL' does not exist" in completed.stderr
