"""Regressions for shared cold-cache writes and process-static cache ownership."""

import json
import os
import select
from pathlib import Path
import subprocess
import sys

import pytest

from sparseengine.kernels.external.flashinfer import jit_cache


@pytest.fixture(autouse=True)
def isolated_cache_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(jit_cache, "_configured_cache", None)
    monkeypatch.setattr(jit_cache, "_cache_lease", None)
    monkeypatch.setattr(jit_cache, "_cache_namespace", lambda: "test-toolchain")
    for variable in ("SPARSEENGINE_TRTLLM_DG_CACHE_ROOT", "TRTLLM_DG_CACHE_DIR"):
        monkeypatch.setenv(variable, "")
        monkeypatch.delenv(variable)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    yield
    if jit_cache._cache_lease is not None:
        jit_cache._cache_lease.close()


@pytest.mark.parametrize("explicit_root", [False, True])
def test_worker_uses_selected_root_without_overwriting_shared_artifacts(
    monkeypatch, tmp_path, explicit_root,
):
    shared = tmp_path / "upstream"
    shared.mkdir()
    artifact = shared / "kernel.cubin"
    artifact.write_bytes(b"existing shared artifact")
    monkeypatch.setenv("TRTLLM_DG_CACHE_DIR", str(shared))
    selected = shared
    if explicit_root:
        selected = tmp_path / "override"
        monkeypatch.setenv("SPARSEENGINE_TRTLLM_DG_CACHE_ROOT", str(selected))

    path = jit_cache.configure_trtllm_cache(rank=0)
    assert path.parent == selected
    assert path != selected
    assert path.is_dir()
    assert Path(os.environ["TRTLLM_DG_CACHE_DIR"]) == path
    assert artifact.read_bytes() == b"existing shared artifact"


def test_default_cache_stays_on_configured_data_filesystem(tmp_path):
    root = jit_cache.resolve_trtllm_cache_root()
    path = jit_cache.configure_trtllm_cache(rank=0, root=root)
    assert path.is_relative_to(tmp_path)
    assert path.parent == root


def test_sequential_engines_reuse_process_static_cache_without_nesting():
    root = jit_cache.resolve_trtllm_cache_root()
    first = jit_cache.configure_trtllm_cache(rank=0, root=root)
    (first / "kernel.cubin").write_bytes(b"compiled")
    assert jit_cache.resolve_trtllm_cache_root() == root
    assert jit_cache.configure_trtllm_cache(rank=0) == first
    assert (first / "kernel.cubin").read_bytes() == b"compiled"


@pytest.mark.parametrize("variable", ["TRTLLM_DG_CACHE_DIR", "SPARSEENGINE_TRTLLM_DG_CACHE_ROOT"])
def test_root_change_fails_before_new_workers_can_be_launched(monkeypatch, tmp_path, variable):
    first = jit_cache.configure_trtllm_cache(rank=0)
    changed = tmp_path / "changed"
    monkeypatch.setenv(variable, str(changed))
    with pytest.raises(RuntimeError, match="Use a fresh process"):
        jit_cache.resolve_trtllm_cache_root()
    with pytest.raises(RuntimeError, match="Use a fresh process"):
        jit_cache.configure_trtllm_cache(rank=0, root=changed)
    assert first.is_dir()
    assert not changed.exists()


def test_bad_root_does_not_publish_partial_configuration(tmp_path):
    root = tmp_path / "file"
    root.write_text("not a directory")
    with pytest.raises(RuntimeError, match="Cannot prepare") as failure:
        jit_cache.configure_trtllm_cache(rank=0, root=root)
    assert isinstance(failure.value.__cause__, OSError)
    assert "TRTLLM_DG_CACHE_DIR" not in os.environ
    assert jit_cache._configured_cache is None


@pytest.mark.parametrize("variable", ["TRTLLM_DG_CACHE_DIR", "SPARSEENGINE_TRTLLM_DG_CACHE_ROOT"])
def test_empty_explicit_root_is_not_silently_ignored(monkeypatch, variable):
    monkeypatch.setenv(variable, "")
    with pytest.raises(ValueError, match="non-empty"):
        jit_cache.resolve_trtllm_cache_root()


def test_spawned_instances_and_ranks_write_same_cache_key_independently():
    # A later engine's children inherit rank zero's modified environment. Pass
    # the resolved root just as LLMEngine does, rather than nesting beneath it.
    root = jit_cache.resolve_trtllm_cache_root()
    parent_cache = jit_cache.configure_trtllm_cache(rank=0)
    script = """
import json, os, sys
from sparseengine.kernels.external.flashinfer.jit_cache import configure_trtllm_cache
path = configure_trtllm_cache(rank=int(sys.argv[2]), root=sys.argv[1])
with (path / 'kernel.cubin').open('x') as artifact:
    artifact.write(str(os.getpid()))
assert configure_trtllm_cache(rank=int(sys.argv[2])) == path
print(json.dumps({'path': str(path), 'pid': os.getpid()}), flush=True)
input()
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(root), str(rank)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for rank in (0, 1, 0, 1)
    ]
    records = []
    try:
        for process in processes:
            assert select.select([process.stdout], [], [], 30)[0], "worker startup timed out"
            records.append(json.loads(process.stdout.readline()))
        for process in processes:
            _, stderr = process.communicate(input="done\n", timeout=30)
            assert process.returncode == 0, stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
    paths = [Path(record["path"]) for record in records]
    assert len(set(paths + [parent_cache])) == len(processes) + 1
    for path, record in zip(paths, records):
        assert path.parent == root
        assert (path / "kernel.cubin").read_text() == str(record["pid"])


def test_restart_reuses_exclusive_cache_artifact(tmp_path):
    script = """
import sys
from sparseengine.kernels.external.flashinfer.jit_cache import configure_trtllm_cache
path = configure_trtllm_cache(0, sys.argv[1])
artifact = path / 'test-artifact'
if sys.argv[2] == 'write':
    artifact.write_text('compiled')
else:
    assert artifact.read_text() == 'compiled'
print(path)
"""
    outputs = [subprocess.check_output(
        [sys.executable, "-c", script, str(tmp_path), mode], text=True, timeout=30,
    ).strip() for mode in ("write", "read")]
    assert outputs[0] == outputs[1]


def test_namespace_change_does_not_reuse_old_artifacts(monkeypatch, tmp_path):
    first, lease = jit_cache._lease_cache(tmp_path, 0)
    lease.close()
    monkeypatch.setattr(jit_cache, "_cache_namespace", lambda: "changed-toolchain")
    second, lease = jit_cache._lease_cache(tmp_path, 0)
    try:
        assert second != first
    finally:
        lease.close()
