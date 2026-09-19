from __future__ import annotations

import pytest
import torch

from sparseengine.operators.workspace import (
    ReusableWorkspaceManager,
    close_workspace_manager,
    get_workspace_manager,
    init_workspace_manager,
    lock_workspace_manager,
)


@pytest.fixture(autouse=True)
def _reset_global_workspace_manager():
    close_workspace_manager()
    yield
    close_workspace_manager()


def test_workspace_lane_reuses_largest_prefix_and_refreshes_leases() -> None:
    manager = ReusableWorkspaceManager(torch.device("cpu"), alignment=256)
    first = manager.reserve_bytes(17, label="first", lane="operators")
    first_pointer = first.buffer.data_ptr()

    second = manager.reserve_bytes(600, label="second", lane="operators")

    assert first.buffer.numel() == 17
    assert second.buffer.numel() == 600
    assert first.buffer.data_ptr() == second.buffer.data_ptr()
    assert first.buffer.data_ptr() != first_pointer
    assert manager.stats()["lanes"]["operators"] == {
        "capacity_bytes": 768,
        "leases": 2,
    }


def test_workspace_lock_rejects_growth_but_allows_existing_capacity() -> None:
    manager = ReusableWorkspaceManager(torch.device("cpu"))
    lease = manager.reserve_bytes(512, label="moe", lane="operators")
    manager.lock()

    manager.reserve_bytes(128, label="linear", lane="operators")
    with pytest.raises(RuntimeError, match="locked"):
        lease.ensure_bytes(1024)
    assert lease.required_bytes == 512
    assert lease.buffer.numel() == 512
    assert manager.stats()["lanes"]["operators"]["leases"] == 2

    with pytest.raises(RuntimeError, match="locked"):
        manager.reserve_bytes(1024, label="late", lane="operators")
    assert manager.stats()["lanes"]["operators"]["leases"] == 2


def test_workspace_lanes_are_independent_and_close_invalidates_leases() -> None:
    manager = ReusableWorkspaceManager(torch.device("cpu"))
    first = manager.reserve_bytes(64, label="first", lane="one")
    second = manager.reserve_bytes(64, label="second", lane="two")

    assert first.buffer.data_ptr() != second.buffer.data_ptr()
    manager.close()

    with pytest.raises(RuntimeError, match="no longer backed"):
        _ = first.buffer
    with pytest.raises(RuntimeError, match="closed"):
        manager.reserve_bytes(64, label="late")


def test_global_workspace_manager_has_explicit_lifecycle() -> None:
    manager = init_workspace_manager(torch.device("cpu"))
    assert get_workspace_manager(torch.device("cpu")) is manager

    lock_workspace_manager()
    assert manager.locked
    with pytest.raises(RuntimeError, match="already initialized"):
        init_workspace_manager(torch.device("cpu"))

    close_workspace_manager()
    with pytest.raises(RuntimeError, match="not initialized"):
        get_workspace_manager(torch.device("cpu"))
