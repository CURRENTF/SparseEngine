import subprocess
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pathlib import Path


def _git_command(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _git_value(repo_root: Path, *args: str) -> str | None:
    result = _git_command(repo_root, *args)
    if result is None:
        return None
    value = result.stdout.strip()
    return value or None


def _verified_source_repo_root(module_path: Path) -> Path | None:
    module_path = module_path.resolve()
    try:
        candidate = module_path.parents[3]
    except IndexError:
        return None
    expected_module = (
        candidate
        / "src"
        / "sparseengine"
        / "utils"
        / "code_revision.py"
    )
    if expected_module.resolve() != module_path:
        return None
    top_level = _git_value(
        candidate,
        "rev-parse",
        "--show-toplevel",
    )
    if top_level is None or Path(top_level).resolve() != candidate:
        return None
    return candidate


@lru_cache(maxsize=1)
def code_revision_info() -> dict[str, str | bool | None]:
    repo_root = _verified_source_repo_root(Path(__file__))
    dirty_result = (
        None
        if repo_root is None
        else _git_command(repo_root, "status", "--porcelain")
    )
    try:
        package_version = version("sparseengine")
    except PackageNotFoundError:
        package_version = None
    return {
        "git_commit": (
            None
            if repo_root is None
            else _git_value(repo_root, "rev-parse", "HEAD")
        ),
        "git_branch": (
            None
            if repo_root is None
            else _git_value(repo_root, "branch", "--show-current")
        ),
        "git_dirty": (
            bool(dirty_result.stdout.strip())
            if dirty_result is not None
            else None
        ),
        "package_version": package_version,
    }
