from __future__ import annotations

from dataclasses import dataclass, field

import torch

from sparseengine.platforms import device_runtime


_DEFAULT_ALIGNMENT = 256


def _round_up(value: int, alignment: int) -> int:
    return (int(value) + int(alignment) - 1) // int(alignment) * int(alignment)


class WorkspaceLease:
    """Stable view of reusable scratch storage owned by a workspace manager."""

    def __init__(
        self,
        manager: ReusableWorkspaceManager,
        lane: str,
        required_bytes: int,
        label: str,
    ) -> None:
        self._manager = manager
        self.lane = str(lane)
        self.required_bytes = int(required_bytes)
        self.label = str(label)
        self._buffer: torch.Tensor | None = None

    @property
    def buffer(self) -> torch.Tensor:
        if self._buffer is None:
            raise RuntimeError(
                f"Workspace lease {self.label!r} is no longer backed by storage."
            )
        return self._buffer

    def _bind(self, storage: torch.Tensor | None) -> None:
        self._buffer = (
            None if storage is None else storage[: self.required_bytes]
        )

    def ensure_bytes(self, required_bytes: int) -> None:
        self._manager._resize_lease(self, required_bytes)


@dataclass
class _WorkspaceLane:
    storage: torch.Tensor | None = None
    leases: list[WorkspaceLease] = field(default_factory=list)


class ReusableWorkspaceManager:
    """Own reusable byte buffers for stream-ordered operator scratch space.

    Leases in one lane alias the same prefix allocation and therefore must be
    consumed sequentially on the execution stream. Operators that may overlap,
    or whose scratch contents must outlive a call, use distinct lanes.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        alignment: int = _DEFAULT_ALIGNMENT,
    ) -> None:
        self.device = torch.device(device)
        self.alignment = int(alignment)
        if self.alignment <= 0:
            raise ValueError(
                f"Workspace alignment must be positive, got {self.alignment}."
            )
        self._lanes: dict[str, _WorkspaceLane] = {}
        self._locked = False
        self._closed = False

    @staticmethod
    def _storage_bytes(storage: torch.Tensor | None) -> int:
        return 0 if storage is None else storage.numel() * storage.element_size()

    @property
    def locked(self) -> bool:
        return self._locked

    def reserve_bytes(
        self,
        required_bytes: int,
        *,
        label: str,
        lane: str = "default",
    ) -> WorkspaceLease:
        if self._closed:
            raise RuntimeError("Workspace manager is closed.")
        required_bytes = int(required_bytes)
        if required_bytes <= 0:
            raise ValueError(
                f"Workspace size must be positive, got {required_bytes}."
            )
        lane = str(lane)
        state = self._lanes.setdefault(lane, _WorkspaceLane())
        self._ensure_capacity(lane, required_bytes)
        lease = WorkspaceLease(self, lane, required_bytes, label)
        state.leases.append(lease)
        lease._bind(state.storage)
        return lease

    def _ensure_capacity(self, lane: str, required_bytes: int) -> None:
        state = self._lanes[lane]
        current_bytes = self._storage_bytes(state.storage)
        if current_bytes >= required_bytes:
            return
        if self._locked:
            raise RuntimeError(
                f"Workspace lane {lane!r} is locked at {current_bytes} bytes, "
                f"but {required_bytes} bytes are required."
            )
        if self.device.type == "cuda" and device_runtime.is_stream_capturing():
            raise RuntimeError(
                f"Workspace lane {lane!r} cannot grow during CUDA Graph capture."
            )
        capacity = _round_up(required_bytes, self.alignment)
        storage = torch.empty(capacity, dtype=torch.uint8, device=self.device)
        state.storage = storage
        for lease in state.leases:
            lease._bind(storage)

    def _resize_lease(
        self,
        lease: WorkspaceLease,
        required_bytes: int,
    ) -> None:
        if self._closed:
            raise RuntimeError("Workspace manager is closed.")
        required_bytes = int(required_bytes)
        if required_bytes <= 0:
            raise ValueError(
                f"Workspace size must be positive, got {required_bytes}."
            )
        state = self._lanes.get(lease.lane)
        if state is None or lease not in state.leases:
            raise RuntimeError(
                f"Workspace lease {lease.label!r} is not owned by this manager."
            )
        if required_bytes <= lease.required_bytes:
            return
        self._ensure_capacity(lease.lane, required_bytes)
        lease.required_bytes = required_bytes
        lease._bind(state.storage)

    def lock(self) -> None:
        if self._closed:
            raise RuntimeError("Workspace manager is closed.")
        self._locked = True

    def close(self) -> None:
        if self._closed:
            return
        for state in self._lanes.values():
            for lease in state.leases:
                lease._bind(None)
            state.leases.clear()
            state.storage = None
        self._lanes.clear()
        self._closed = True

    def stats(self) -> dict[str, object]:
        return {
            "device": str(self.device),
            "locked": self._locked,
            "lanes": {
                name: {
                    "capacity_bytes": self._storage_bytes(state.storage),
                    "leases": len(state.leases),
                }
                for name, state in sorted(self._lanes.items())
            },
        }


_workspace_manager: ReusableWorkspaceManager | None = None


def init_workspace_manager(device: torch.device) -> ReusableWorkspaceManager:
    global _workspace_manager
    if _workspace_manager is not None:
        raise RuntimeError(
            "Workspace manager is already initialized on "
            f"{_workspace_manager.device}."
        )
    _workspace_manager = ReusableWorkspaceManager(device)
    return _workspace_manager


def get_workspace_manager(
    device: torch.device,
    *,
    create: bool = False,
) -> ReusableWorkspaceManager:
    global _workspace_manager
    device = torch.device(device)
    if _workspace_manager is None:
        if not create:
            raise RuntimeError("Workspace manager is not initialized.")
        _workspace_manager = ReusableWorkspaceManager(device)
    if _workspace_manager.device != device:
        raise RuntimeError(
            "Workspace manager device mismatch: "
            f"manager={_workspace_manager.device}, requested={device}."
        )
    return _workspace_manager


def lock_workspace_manager() -> None:
    if _workspace_manager is not None:
        _workspace_manager.lock()


def close_workspace_manager() -> None:
    global _workspace_manager
    manager = _workspace_manager
    _workspace_manager = None
    if manager is not None:
        manager.close()


def bind_module_workspace_lane(module: torch.nn.Module, lane: str) -> None:
    """Bind scratch ownership once, before warmup and graph capture.

    Only modules with reusable scratch implement this hook. Layers executed
    sequentially may use the same lane; concurrent branches must not.
    """
    for child in module.modules():
        bind = getattr(child, "bind_workspace_lane", None)
        if callable(bind):
            bind(lane)


__all__ = [
    "bind_module_workspace_lane",
    "ReusableWorkspaceManager",
    "WorkspaceLease",
    "close_workspace_manager",
    "get_workspace_manager",
    "init_workspace_manager",
    "lock_workspace_manager",
]
