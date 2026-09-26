import pytest
import torch

from sparseengine.engine.cache_manager.storage.packed_shared_kv import PackedSharedKVPool


def make_pool():
    return PackedSharedKVPool(num_pages=7, page_size=64, reserved_pages=2,
                              device=torch.device("cpu"))


def test_prefix_and_fork_references_release_storage_only_after_last_owner():
    # An active request, radix entry and fork independently retain immutable KV.
    pool = make_pool()
    capacity = pool.num_free_slots
    pages = pool.allocate_pages(3)
    assert not set(pages) & {0, 1}
    pool.retain_pages(pages)
    pool.retain_pages(pages)
    pool.release_pages(pages)
    pool.release_pages(pages)
    assert pool.num_free_slots == capacity - 3 * pool.page_size
    pool.release_pages(pages)
    assert pool.num_free_slots == capacity
    assert len(set(pool.allocate_pages(5))) == 5


def test_failed_page_transactions_leave_allocation_unchanged():
    pool = make_pool()
    pages = pool.allocate_pages(2)
    before = pool.num_free_pages
    with pytest.raises(MemoryError):
        pool.allocate_pages(before + 1)
    with pytest.raises(ValueError):
        pool.release_pages((pages[0], pages[0], pages[1]))
    with pytest.raises(ValueError):
        pool.retain_pages((pages[0], 0))
    assert pool.num_free_pages == before
    pool.release_pages(pages)
    assert pool.num_free_pages == 5
    with pytest.raises(ValueError):
        pool.release_pages(pages)


def test_page_accounting_includes_scale_tail_and_alignment_without_double_counting():
    pool = make_pool()
    payload = pool.layer_payload()
    assert payload.pages.data_ptr() == pool.byte_storage.data_ptr()
    assert payload.pages.stride(0) % 576 == 0
    assert payload.pages.stride(0) >= pool.page_size * (576 + 8)
    assert sum(t.numel() * t.element_size() for t in pool.accounting_tensors()) == (
        7 * payload.pages.stride(0)
    )


def test_attention_and_index_storage_share_family_ownership_not_tensor_bytes():
    from sparseengine.engine.cache_manager.storage.packed_index_kv import PackedIndexKVPool

    first = make_pool()
    second = PackedSharedKVPool(num_pages=7, page_size=64, reserved_pages=2,
                                device=torch.device("cpu"), allocator=first.allocator)
    index = PackedIndexKVPool(allocator=first.allocator, page_size=64, device="cpu")
    pages = first.allocator.allocate_pages(3)
    first.allocator.retain_pages(pages)  # Prefix owns all layer payloads.
    first.allocator.release_pages(pages)  # Active request ends.
    assert first.num_free_pages == second.num_free_pages == index.allocator.num_free_pages == 2
    storages = [first.byte_storage, second.byte_storage, index.byte_storage]
    assert len({tensor.data_ptr() for tensor in storages}) == 3
    first.allocator.release_pages(pages)
    assert second.num_free_pages == 5
    with pytest.raises(ValueError):
        PackedSharedKVPool(num_pages=8, page_size=64, reserved_pages=2,
                           device=torch.device("cpu"), allocator=first.allocator)
