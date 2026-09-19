"""Opt-in cgroup-v2 OOM detection for independently failed MiniSWE tasks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


class DockerMemoryLimitExceeded(RuntimeError):
    pass


def memory_settings() -> tuple[int, str]:
    limit = int(os.environ.get("SPARSEENGINE_DOCKER_MEMORY_LIMIT_BYTES", "0"))
    parent = os.environ.get("SPARSEENGINE_DOCKER_MEMORY_PARENT", "")
    if limit <= 0 or not parent:
        raise ValueError("Docker memory guard requires a positive limit and cgroup parent")
    return limit, parent


def constrain_sdk_kwargs(kwargs: dict) -> dict:
    """Apply the same limits to official scorer containers created via Docker SDK."""
    limit, parent = memory_settings()
    kwargs = dict(kwargs)
    for key, value in {"mem_limit": limit, "memswap_limit": limit,
                       "cgroup_parent": parent}.items():
        if kwargs.get(key) not in (None, value):
            raise ValueError(f"Conflicting Docker memory setting: {key}={kwargs[key]!r}")
        kwargs[key] = value
    return kwargs


def install_sdk_limits() -> None:
    from docker.models.containers import ContainerCollection
    original = ContainerCollection.create
    if getattr(original, "_sparseengine_memory_guard", False):
        return

    def create(self, *args, **kwargs):
        return original(self, *args, **constrain_sdk_kwargs(kwargs))

    create._sparseengine_memory_guard = True
    ContainerCollection.create = create


class DockerMemoryGuard:
    def __init__(self, executable: str, container_id: str, image: str):
        self.executable, self.container_id, self.image = executable, container_id, image
        self.failed = False
        limit, parent = memory_settings()
        record = self.inspect()
        config = record["HostConfig"]
        if (config["Memory"] != limit or config["MemorySwap"] != limit
                or config["CgroupParent"] != parent or config["AutoRemove"]):
            raise ValueError(f"Container memory isolation does not match requested limits: {config}")
        pid = int(record["State"]["Pid"])
        entries = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
        paths = [line[3:] for line in entries if line.startswith("0::")]
        if len(paths) != 1:
            raise RuntimeError("Docker memory guard requires a unified cgroup-v2 hierarchy")
        cgroup = Path(os.path.normpath(paths[0]))
        parent_path = Path(os.path.normpath(parent))
        self.events = Path("/sys/fs/cgroup") / cgroup.relative_to("/") / "memory.events"
        if parent_path.is_absolute():
            contained = parent_path in cgroup.parents
        else:
            # Relative paths and systemd slice names are relative to a delegated root.
            contained = any(
                ancestor.parts[-len(parent_path.parts):] == parent_path.parts
                for ancestor in cgroup.parents
            )
        if not contained:
            raise RuntimeError(f"Container is outside the bounded memory parent: {self.events}")
        event_path = os.environ.get("SPARSEENGINE_DOCKER_MEMORY_EVENTS", "")
        if not event_path:
            raise ValueError("SPARSEENGINE_DOCKER_MEMORY_EVENTS is required")
        self.log_path = Path(event_path)
        self.check()

    def inspect(self) -> dict:
        result = subprocess.run([self.executable, "inspect", self.container_id],
                                capture_output=True, text=True, timeout=10, check=True)
        records = json.loads(result.stdout)
        if len(records) != 1 or records[0]["Id"] != self.container_id:
            raise RuntimeError("Unexpected Docker inspect identity")
        return records[0]

    def check(self) -> None:
        if self.failed:
            raise DockerMemoryLimitExceeded(f"Container OOM: {self.container_id}")
        try:
            counters = dict(line.split() for line in self.events.read_text().splitlines())
            # oom_kill includes victims selected by an ancestor's aggregate limit.
            killed = int(counters["oom_kill"]) > 0
        except FileNotFoundError:
            record = self.inspect()
            killed = bool(record["State"]["OOMKilled"])
            if not killed:
                raise RuntimeError(f"Container memory accounting disappeared: {self.container_id}")
        if killed:
            self.failed = True
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            row = {"time": time.time(), "container_id": self.container_id,
                   "image": self.image, "event": "container_oom",
                   "exit_status": "DockerMemoryLimitExceeded"}
            with self.log_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            raise DockerMemoryLimitExceeded(f"Container OOM: {self.container_id}")
