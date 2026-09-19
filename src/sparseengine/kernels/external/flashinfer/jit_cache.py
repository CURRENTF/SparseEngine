"""Exclusive, reusable cache leases for FlashInfer's TensorRT-LLM DeepGEMM JIT."""

from __future__ import annotations

import os
import fcntl
import hashlib
from importlib import metadata
from pathlib import Path


# The upstream C++ compiler remembers its first cache directory for the entire
# process. Keep that directory across sequential engines, including rank zero.
_configured_cache: tuple[int, Path, Path] | None = None
# Hold the lease until process exit: upstream keeps a process-static compiler.
_cache_lease = None


def _cache_namespace() -> str:
    """Separate installed compiler/header/toolchain revisions without CUDA init."""
    digest = hashlib.sha256(b"sparseengine-deepgemm-cache-v1-sm90a")
    for name in ("flashinfer-python", "torch", "nvidia-cuda-nvrtc-cu12", "nvidia-cuda-nvrtc-cu13"):
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            digest.update(f"{name}:absent".encode())
            continue
        digest.update(f"{name}:{dist.version}:{dist.locate_file('')}".encode())
        digest.update((dist.read_text("RECORD") or "").encode())
        if name == "flashinfer-python":
            source = Path(dist.locate_file("flashinfer/data/csrc/nv_internal/tensorrt_llm/deep_gemm"))
            for header in sorted(source.rglob("*.cuh")):
                digest.update(str(header.relative_to(source)).encode())
                digest.update(header.read_bytes())
    for name in ("CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH"):
        digest.update(f"{name}={os.environ.get(name, '')}".encode())
    return digest.hexdigest()[:24]


def _lease_cache(root: Path, rank: int):
    namespace = _cache_namespace()
    # Bound contention; a process owns one directory and never shares its writes.
    for slot in range(4096):
        path = root / f"worker-{namespace}-rank-{rank}-slot-{slot}"
        lease = (root / f"{path.name}.lock").open("a+b")
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lease.close()
            continue
        except BaseException:
            lease.close()
            raise
        try:
            path.mkdir(parents=True, exist_ok=True)
        except BaseException:
            lease.close()
            raise
        return path, lease
    raise RuntimeError(f"No free TensorRT-LLM DeepGEMM cache lease under {root}")


def _check_cache_root(root: Path) -> Path:
    configured = _configured_cache
    if configured is not None and configured[0] == os.getpid() and root != configured[1]:
        raise RuntimeError(
            "TensorRT-LLM DeepGEMM cache root cannot change in an initialized "
            f"worker process: {configured[1]} -> {root}. Use a fresh process."
        )
    return root


def resolve_trtllm_cache_root() -> Path:
    """Resolve the user root before spawning workers or rewriting their env."""
    for name in ("SPARSEENGINE_TRTLLM_DG_CACHE_ROOT", "TRTLLM_DG_CACHE_DIR"):
        value = os.environ.get(name)
        if value is None:
            continue
        if not value.strip():
            raise ValueError(f"{name} must name a non-empty cache directory.")
        root = Path(value).expanduser().resolve()
        configured = _configured_cache
        if (
            name == "TRTLLM_DG_CACHE_DIR"
            and configured is not None
            and configured[0] == os.getpid()
            and root == configured[2]
        ):
            return configured[1]
        return _check_cache_root(root)
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return _check_cache_root((base / "sparseengine" / "trtllm-deepgemm").resolve())


def configure_trtllm_cache(rank: int, root: str | Path | None = None) -> Path:
    """Isolate cold JIT writes before any external CUDA kernel initialization."""
    global _configured_cache, _cache_lease
    root = resolve_trtllm_cache_root() if root is None else Path(root).expanduser().resolve()
    _check_cache_root(root)
    configured = _configured_cache
    try:
        if configured is not None and configured[0] == os.getpid():
            path = configured[2]
        else:
            root.mkdir(parents=True, exist_ok=True)
            path, lease = _lease_cache(root, rank)
            # Close an inherited descriptor without unlocking the parent's lease.
            if _cache_lease is not None:
                _cache_lease.close()
            _cache_lease = lease
    except OSError as error:
        raise RuntimeError(
            f"Cannot prepare TensorRT-LLM DeepGEMM cache directory under {root}: {error}"
        ) from error
    os.environ["TRTLLM_DG_CACHE_DIR"] = str(path)
    _configured_cache = (os.getpid(), root, path)
    return path
