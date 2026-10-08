from __future__ import annotations

from dataclasses import dataclass, field

from .packed_index_kv import PackedIndexKVPool
from .packed_shared_kv import PackedSharedKVPool
from .page_allocator import PhysicalPageAllocator


@dataclass(eq=False)
class CompressedFamilyLease:
    """A request or prefix owns one reference to each listed family page."""
    owner: object = field(repr=False)
    pages: list[int] = field(default_factory=list)
    materialized_tokens: int = 0
    released: bool = False


class CompressedKVFamily:
    """Joint storage and immutable-prefix ownership for one compression ratio.

    Logical compression and selection remain outside this physical storage
    class. One lease covers every attention/index layer in the family. Partial
    prefix pages are shared until an append requires a private tail copy.
    """

    def __init__(self, *, layer_ids, with_index, num_pages, reserved_pages,
                 page_size, device):
        self.page_size = page_size
        self.allocator = PhysicalPageAllocator(num_pages=num_pages, reserved_pages=reserved_pages)
        self.attention = {
            layer: PackedSharedKVPool(num_pages=num_pages, reserved_pages=reserved_pages,
                                      page_size=page_size, device=device, allocator=self.allocator)
            for layer in layer_ids
        }
        self.index = {
            layer: PackedIndexKVPool(allocator=self.allocator, page_size=page_size, device=device)
            for layer in self.attention
        } if with_index else {}
        if not self.attention:
            raise ValueError("A compressed family must contain attention layers")

    def new_lease(self):
        return CompressedFamilyLease(self)

    def _validate(self, lease):
        if lease.owner is not self:
            raise ValueError("Compressed lease belongs to another physical family")
        if lease.released:
            raise ValueError("Compressed family lease has already been released")

    def reservation_pages(self, lease, token_capacity):
        self._validate(lease)
        if token_capacity < lease.materialized_tokens:
            raise ValueError("Cannot reserve below materialized compressed length")
        required = (token_capacity + self.page_size - 1) // self.page_size
        additional = max(0, required - len(lease.pages))
        tail = lease.materialized_tokens % self.page_size
        copy_tail = (tail != 0 and token_capacity > lease.materialized_tokens
                     and self.allocator.reference_count(
                         lease.pages[lease.materialized_tokens // self.page_size]) > 1)
        return additional + int(copy_tail)

    def reserve(self, lease, token_capacity):
        count = self.reservation_pages(lease, token_capacity)
        allocated = self.allocator.allocate_pages(count)
        required = (token_capacity + self.page_size - 1) // self.page_size
        additional = max(0, required - len(lease.pages))
        copy_tail = count > additional
        if copy_tail:
            index = lease.materialized_tokens // self.page_size
            old, new = lease.pages[index], allocated[0]
            try:
                for pool in (*self.attention.values(), *self.index.values()):
                    pool.byte_storage[new].copy_(pool.byte_storage[old])
            except BaseException:
                self.allocator.release_pages(allocated)
                raise
            lease.pages[index] = new
            self.allocator.release_pages((old,))
        lease.pages.extend(allocated[int(copy_tail):])

    def mark_materialized(self, lease, token_count):
        self._validate(lease)
        if not lease.materialized_tokens <= token_count <= len(lease.pages)*self.page_size:
            raise ValueError("Materialized compressed length exceeds its reservation")
        # Called by CacheManager after all layer writes in the step retire.
        lease.materialized_tokens = token_count

    def snapshot(self, lease):
        self._validate(lease)
        count = (lease.materialized_tokens + self.page_size - 1) // self.page_size
        pages = lease.pages[:count]
        self.allocator.retain_pages(pages)
        return CompressedFamilyLease(self, list(pages), lease.materialized_tokens)

    def release(self, lease):
        self._validate(lease)
        self.allocator.release_pages(lease.pages)
        lease.pages.clear()
        lease.released = True

    def physical_slots(self, lease, start, stop):
        self._validate(lease)
        if not 0 <= start <= stop <= len(lease.pages)*self.page_size:
            raise ValueError("Compressed slot range exceeds physical reservation")
        return [lease.pages[i // self.page_size]*self.page_size + i % self.page_size
                for i in range(start, stop)]

    def accounting_tensors(self):
        return tuple(pool.byte_storage for pool in (*self.attention.values(), *self.index.values()))
