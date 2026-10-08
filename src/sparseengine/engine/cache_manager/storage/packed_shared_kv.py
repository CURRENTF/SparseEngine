from __future__ import annotations

import torch

from ..base import PackedSharedKVPayload
from .page_allocator import PhysicalPageAllocator


class PackedSharedKVPool:
    """Physical pages and references shared by active requests and prefixes.

    Mutable window/carry state must be copied on fork by the cache manager.
    This pool only shares immutable compressed pages. Reserved pages provide
    distinct scratch destinations for graph padding and are never admitted.
    """

    def __init__(self, *, num_pages: int, page_size: int, reserved_pages: int,
                 device: torch.device, allocator: PhysicalPageAllocator | None = None):
        if page_size <= 0 or page_size & (page_size - 1):
            raise ValueError("Packed shared KV page size must be a power of two")
        if num_pages <= 0 or not 0 <= reserved_pages < num_pages:
            raise ValueError("Packed shared KV requires positive allocatable capacity")
        self.page_size = int(page_size)
        self.reserved_pages = int(reserved_pages)
        self.allocator = allocator or PhysicalPageAllocator(
            num_pages=num_pages, reserved_pages=reserved_pages,
        )
        if (self.allocator.num_pages, self.allocator.reserved_pages) != (num_pages, reserved_pages):
            raise ValueError("Shared KV storage differs from its family allocator capacity")
        self.page_bytes = ((584 * page_size + 575) // 576) * 576
        self.byte_storage = torch.empty(num_pages, self.page_bytes, dtype=torch.uint8, device=device)
        # The external reader requires logically contiguous tokens. Physical
        # scale rows follow all 576-byte value rows inside each page.
        self._payload = PackedSharedKVPayload(
            self.byte_storage.as_strided(
                (num_pages, page_size, 1, 584), (self.page_bytes, 584, 584, 1),
            ), page_size,
        )

    @property
    def num_free_pages(self):
        return self.allocator.num_free_pages

    @property
    def num_free_slots(self):
        return self.num_free_pages * self.page_size

    def layer_payload(self):
        return self._payload

    def accounting_tensors(self):
        return (self.byte_storage,)

    def allocate_pages(self, count: int) -> tuple[int, ...]:
        return self.allocator.allocate_pages(count)

    def retain_pages(self, pages):
        self.allocator.retain_pages(pages)

    def release_pages(self, pages):
        self.allocator.release_pages(pages)
