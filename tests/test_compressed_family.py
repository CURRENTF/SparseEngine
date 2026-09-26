import pytest
import torch

from sparseengine.engine.cache_manager.storage.compressed_family import (
    CompressedKVFamily,
)


def family(num_pages=6):
    return CompressedKVFamily(layer_ids=(2, 4), with_index=True, num_pages=num_pages,
                              reserved_pages=1, page_size=64, device=torch.device("cpu"))


def test_partial_prefix_tail_copies_every_payload_and_retains_full_pages():
    pool = family()
    live = pool.new_lease()
    pool.reserve(live, 70)
    pool.mark_materialized(live, 70)
    for i, tensor in enumerate(pool.accounting_tensors()):
        tensor[live.pages[1]].fill_(i + 3)
    prefix = pool.snapshot(live)
    original = list(prefix.pages)
    assert pool.reservation_pages(live, 71) == 1
    pool.reserve(live, 71)
    assert live.pages[0] == prefix.pages[0]
    assert live.pages[1] != prefix.pages[1]
    for tensor in pool.accounting_tensors():
        torch.testing.assert_close(tensor[live.pages[1]], tensor[prefix.pages[1]])
        tensor[live.pages[1]].zero_()
        assert bool((tensor[prefix.pages[1]] > 0).all())
    assert prefix.pages == original
    assert pool.reservation_pages(live, 72) == 0
    pool.release(live)
    assert pool.allocator.num_free_pages == 3
    pool.release(prefix)
    assert pool.allocator.num_free_pages == 5


def test_snapshot_excludes_unmaterialized_pages_and_failed_reservation_is_atomic():
    pool = family(num_pages=4)
    live = pool.new_lease()
    pool.reserve(live, 128)
    pool.mark_materialized(live, 33)
    prefix = pool.snapshot(live)
    assert len(prefix.pages) == 1
    before = list(live.pages), pool.allocator.num_free_pages
    # Needs one COW tail and one extension; there is only one free page.
    with pytest.raises(MemoryError):
        pool.reserve(live, 129)
    assert (live.pages, pool.allocator.num_free_pages) == before
    pool.release(live)
    pool.release(prefix)
    assert pool.allocator.num_free_pages == 3
    with pytest.raises(ValueError, match="released"):
        pool.release(prefix)


def test_full_prefix_page_shares_without_copy_on_next_page_append():
    pool = family()
    live = pool.new_lease()
    pool.reserve(live, 64)
    pool.mark_materialized(live, 64)
    prefix = pool.snapshot(live)
    assert pool.reservation_pages(live, 65) == 1
    pool.reserve(live, 65)
    assert live.pages[0] == prefix.pages[0]
    assert pool.physical_slots(live, 63, 65) == [live.pages[0]*64+63, live.pages[1]*64]
    pool.release(prefix)
    pool.release(live)
    assert pool.allocator.num_free_pages == 5
