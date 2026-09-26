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
