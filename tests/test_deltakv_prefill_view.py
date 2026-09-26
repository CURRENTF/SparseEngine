import pytest
import torch

from sparseengine.engine.cache_manager.base import SparseSelection
from sparseengine.engine.cache_manager.methods.deltakv_less_memory import DeltaKVLessMemoryCacheManager


@pytest.mark.parametrize("width", [7, 45451])
def test_prefill_preserves_reconstructed_view_larger_than_decode_workspace(width):
    """Prefill must not repack post-RoPE history into the decode scratch pool."""
    manager = object.__new__(DeltaKVLessMemoryCacheManager)
    manager.deltakv_layer_to_idx = {1: 0}
    manager.has_prefill_staging_view = lambda _: False
    # The real failure requested 2 * 45451 slots from a 65536-slot workspace.
    manager.deltakv_materialized_compute_num_slots = min(width, 65536)
    capacity = width + 5
    manager.deltakv_full_kv_cache = torch.arange(
        2 * capacity * 4, dtype=torch.float32,
    ).reshape(2, 1, capacity, 1, 4)
    active = torch.full((2, width), -1, dtype=torch.int32)
    active[0] = torch.arange(width - 1, -1, -1, dtype=torch.int32)
    active[1, :3] = torch.tensor([width + 1, width, width + 2])
    lengths = torch.tensor([width, 3], dtype=torch.int32)
    local_rows = torch.tensor([0, 1], dtype=torch.int32)
    temp = torch.tensor([width, width + 1, width + 2], dtype=torch.int32)
    # The reconstruction contract supplies indexed post-RoPE KV plus the
    # temporary slots the attention layer must release after consumption.
    manager.deltakv_reconstruct = lambda **_: (active, local_rows, lengths, temp)
    selection = SparseSelection(
        kind="deltakv", req_indices=torch.tensor([5, 2], dtype=torch.int32),
        context_lens=lengths, max_context_len=width, release_temp_slots=True,
    )
    current = torch.empty((2, 1, 4))
    view = manager.build_prefill_compute_view(1, current, current, selection)
    assert view.payload.k_cache.data_ptr() == manager.deltakv_full_kv_cache[0, 0].data_ptr()
    assert view.payload.v_cache.data_ptr() == manager.deltakv_full_kv_cache[1, 0].data_ptr()
    assert view.meta.active_slots is active
    assert view.meta.req_indices is local_rows
    assert view.meta.context_lens is lengths
    assert view.meta.temp_slots is temp
    # Both requests retain their valid history, including non-contiguous slots;
    # padded -1 entries must not become part of the visible attention history.
    torch.testing.assert_close(
        view.payload.k_cache[active[0].long()],
        manager.deltakv_full_kv_cache[0, 0, :width].flip(0),
    )
    torch.testing.assert_close(
        view.payload.v_cache[active[1, :3].long()],
        manager.deltakv_full_kv_cache[1, 0, [width + 1, width, width + 2]],
    )


def test_prefill_keeps_offloaded_history_staging_view():
    manager = object.__new__(DeltaKVLessMemoryCacheManager)
    manager.deltakv_layer_to_idx = {1: 0}
    manager.has_prefill_staging_view = lambda _: True
    manager._deltakv_long_prefill_offload_step_active = True
    manager.deltakv_prefill_staging_kv_cache = torch.randn(2, 8, 1, 4)
    active = torch.arange(8, dtype=torch.int32).unsqueeze(0)
    rows = torch.tensor([0], dtype=torch.int32)
    lengths = torch.tensor([8], dtype=torch.int32)
    selection = SparseSelection(kind="deltakv", req_indices=rows, context_lens=lengths)
    current = torch.empty(2, 1, 4)
    k, v, slots, req, lens = manager.get_prefill_compute_view(
        1, current, current, selection, active, rows, lengths,
    )
    assert k.data_ptr() == manager.deltakv_prefill_staging_kv_cache[0].data_ptr()
    assert v.data_ptr() == manager.deltakv_prefill_staging_kv_cache[1].data_ptr()
    assert slots is active and req is rows and lens is lengths


def test_prefill_raw_keys_are_rotated_once_and_reconstructed_keys_are_preserved():
    from sparseengine.utils.context import reset_context, set_context

    manager = object.__new__(DeltaKVLessMemoryCacheManager)
    manager.deltakv_layer_to_idx = {1: 0}
    manager.has_prefill_staging_view = lambda _: False
    manager.deltakv_full_kv_cache = torch.zeros(2, 1, 6, 1, 4)
    manager.deltakv_full_kv_cache[0, 0, 0, 0] = torch.tensor([1., 2., 3., 4.])
    manager.deltakv_full_kv_cache[0, 0, 1, 0] = torch.tensor([5., 6., 7., 8.])
    manager.deltakv_full_kv_cache[1, 0, :2] = 9
    manager.deltakv_slot_to_pos = torch.tensor([0, 1, -1, -1, -1, -1], dtype=torch.int32)
    # A 90-degree rotation has the independent oracle [-x2,-x3,x0,x1].
    manager.cos_sin_cache = torch.tensor([[[0., 0., 1., 1.]], [[0., 0., 1., 1.]]])
    manager._deltakv_postrope_slot_mask = torch.tensor([[False, True, False, False, False, False]])
    manager._allocate_temp_deltakv_full = lambda count: torch.arange(2, 2 + count, dtype=torch.int32)
    active = torch.tensor([[0, 1]], dtype=torch.int32)
    lengths = torch.tensor([2], dtype=torch.int32)
    rows = torch.tensor([0], dtype=torch.int32)
    selection = SparseSelection(kind="deltakv", req_indices=rows, context_lens=lengths)
    set_context(is_prefill=True)
    try:
        slots, temp = manager._materialize_deltakv_active_postrope_view(
            1, active, lengths, torch.tensor([1], dtype=torch.int32),
        )
        current = torch.empty(0, 1, 4)
        k, v, slots, _, _ = manager.get_prefill_compute_view(
            1, current, current, selection, slots, rows, lengths,
        )
        torch.testing.assert_close(k[slots[0].long(), 0], torch.tensor([[-3., -4., 1., 2.], [5., 6., 7., 8.]]))
        torch.testing.assert_close(v[slots[0].long(), 0], torch.full((2, 4), 9.))
        assert temp.tolist() == [2]
        # Persistent raw K is untouched; only the returned temporary view rotates.
        torch.testing.assert_close(manager.deltakv_full_kv_cache[0, 0, 0, 0], torch.tensor([1., 2., 3., 4.]))
    finally:
        reset_context()
