"""Shared allocation must preserve tensor ownership and indexed-copy eligibility."""

import gc
import weakref

import pytest
import torch

from sparseengine.engine.cache_manager.offload.allocation import plan_host_allocation
from sparseengine.engine.cache_manager.offload.host_pool import HostTensorPool
from sparseengine.operators.indexed_host_copy import make_pointer_table


def test_invalid_shapes_fail_before_allocating(monkeypatch):
    # Unlike method capacity tests, this protects arbitrary callers of the pool.
    calls = []
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: calls.append((a, kw)))
    with pytest.raises(ValueError, match="non-negative"):
        HostTensorPool([(2, 3), (4, -1)], dtype=torch.float32, pin_memory=False)
    assert calls == []


def test_separate_views_can_be_released_independently():
    # RawKV chunks must not inherit a global owner that retains released data.
    pool = HostTensorPool([(5, 2), (3, 4)], dtype=torch.float32, pin_memory=False)
    first, second = pool.tensors
    first_ref = weakref.ref(first)
    second.fill_(7)
    del first, pool
    gc.collect()
    assert first_ref() is None
    torch.testing.assert_close(second, torch.full((3, 4), 7.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pinned allocator required")
@pytest.mark.parametrize("allow_packing", [False, True])
def test_pool_layout_lifetime_and_pointer_binding(allow_packing):
    # Existing method tests do not exercise a pool without a KV manager owner.
    shapes = [(5, 1, 64), (9, 1, 512)]
    with torch.device("cuda"):
        pool = HostTensorPool(shapes, dtype=torch.bfloat16, allow_packing=allow_packing)
    tensors = pool.tensors
    assert all(t.device.type == "cpu" and t.is_pinned() and t.is_contiguous() for t in tensors)
    assert [tuple(t.shape) for t in tensors] == shapes
    separate = plan_host_allocation([t.numel() * t.element_size() for t in tensors])
    assert pool.plan.estimated_bytes <= separate.estimated_bytes
    if pool.plan.packed:
        assert tensors[1].data_ptr() - tensors[0].data_ptr() == tensors[0].numel() * tensors[0].element_size()
    pointers = make_pointer_table(tensors, device="cuda")
    assert pointers.cpu().tolist() == [t.data_ptr() for t in tensors]
    tensors[0].fill_(2)
    tensors[1].fill_(3)
    del pool
    gc.collect()
    # Views must retain the backing after the temporary pool object is gone.
    for tensor, value in zip(tensors, (2, 3)):
        torch.testing.assert_close(tensor.cuda(), torch.full_like(tensor, value, device="cuda"))


def test_indexed_transfer_rejects_unpinned_or_unsupported_components():
    # The generic binding no longer relies on OmniKV config to reject bad inputs.
    with pytest.raises(ValueError, match="FP16/BF16"):
        make_pointer_table([torch.empty(4, dtype=torch.float32)], device="cuda")
    with pytest.raises(ValueError, match="pinned"):
        make_pointer_table([torch.empty(4, dtype=torch.float16)], device="cuda")
    with pytest.raises(ValueError, match="CUDA device"):
        make_pointer_table([torch.empty(4, dtype=torch.float16)], device="cpu")
    with pytest.raises(ValueError, match="contiguous"):
        make_pointer_table([torch.empty(2, 4, dtype=torch.float16).t()], device="cuda")
