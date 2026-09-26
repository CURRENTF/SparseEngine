import pytest
import torch

from sparseengine.engine.cache_manager.storage.shared_kv_state import SharedKVStateRows


def test_bounded_prefix_window_and_carry_copy_matches_accounting():
    pool = SharedKVStateRows(num_rows=3, reserved_rows=1, compress_ratios=(0, 4, 128),
                             device=torch.device("cpu"))
    assert sum(t.numel()*t.element_size() for t in pool.accounting_tensors()) == 3*pool.row_bytes
    live = pool.allocate()
    for i, tensor in enumerate(pool.accounting_tensors()):
        tensor[live].fill_(i+1)
    prefix = pool.copy(live)
    for tensor in pool.accounting_tensors():
        torch.testing.assert_close(tensor[prefix], tensor[live])
        tensor[live].zero_()
        assert bool((tensor[prefix] > 0).all())
    assert pool.num_free_rows == 0
    with pytest.raises(MemoryError, match="snapshot"):
        pool.copy(prefix)
    pool.release(live)
    reused = pool.allocate()
    assert reused == live
    assert all(not bool(t[reused].any()) for t in pool.accounting_tensors())
    pool.release(reused)
    pool.release(prefix)
    with pytest.raises(ValueError, match="not owned"):
        pool.release(prefix)
    assert pool.num_free_rows == 2



def test_mutable_rows_bind_reserved_pages_without_duplicating_windows():
    # Two page-64 blocks have the same byte size as one page-128 window row.
    page_bytes = ((584*64+575)//576)*576
    arena = torch.ones(8, page_bytes, dtype=torch.uint8)
    windows = arena[:6].view(3, 2*page_bytes)
    pool = SharedKVStateRows(num_rows=3, reserved_rows=1, compress_ratios=(0,),
                            device=torch.device("cpu"), window_storage={0: windows})
    row = pool.allocate()
    assert pool.windows[0].data_ptr() == arena.data_ptr()
    assert not bool(arena[row*2:row*2+2].any())
    assert bool(arena[6:].all())
