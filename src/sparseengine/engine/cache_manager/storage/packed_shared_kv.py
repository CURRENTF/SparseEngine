from __future__ import annotations

from collections import Counter

import torch

from ..base import PackedSharedKVPayload


class PackedSharedKVPool:
    """Physical pages and references shared by active requests and prefixes.

    Mutable window/carry state must be copied on fork by the cache manager.
    This pool only shares immutable compressed pages. Reserved pages provide
    distinct scratch destinations for graph padding and are never admitted.
    """

    def __init__(self, *, num_pages: int, page_size: int, reserved_pages: int,
                 device: torch.device):
        if page_size <= 0 or page_size & (page_size - 1):
            raise ValueError("Packed shared KV page size must be a power of two")
        if num_pages <= 0 or not 0 <= reserved_pages < num_pages:
            raise ValueError("Packed shared KV requires positive allocatable capacity")
        self.page_size = int(page_size)
        self.reserved_pages = int(reserved_pages)
        self.page_bytes = ((584 * page_size + 575) // 576) * 576
        self.byte_storage = torch.empty(num_pages, self.page_bytes, dtype=torch.uint8, device=device)
        # The external reader requires logically contiguous tokens. Physical
        # scale rows follow all 576-byte value rows inside each page.
        self._payload = PackedSharedKVPayload(
            self.byte_storage.as_strided(
                (num_pages, page_size, 1, 584), (self.page_bytes, 584, 584, 1),
            ), page_size,
        )
        self._references = [0] * num_pages
        self._free = list(range(num_pages - 1, reserved_pages - 1, -1))

    @property
    def num_free_pages(self):
        return len(self._free)

    @property
    def num_free_slots(self):
        return self.num_free_pages * self.page_size

    def layer_payload(self):
        return self._payload

    def accounting_tensors(self):
        return (self.byte_storage,)

    def allocate_pages(self, count: int) -> tuple[int, ...]:
        if count < 0:
            raise ValueError("Cannot allocate a negative page count")
        if count > self.num_free_pages:
            raise MemoryError(f"Packed shared KV needs {count} pages, has {self.num_free_pages}")
        pages = tuple(self._free.pop() for _ in range(count))
        for page in pages:
            self._references[page] = 1
        return pages

    def _counts(self, pages):
        counts = Counter(pages)
        for page, count in counts.items():
            if not isinstance(page, int) or not self.reserved_pages <= page < len(self._references):
                raise ValueError(f"Invalid or reserved packed KV page {page}")
            if self._references[page] <= 0:
                raise ValueError(f"Packed KV page {page} is not allocated")
        return counts

    def retain_pages(self, pages):
        counts = self._counts(pages)
        for page, count in counts.items():
            self._references[page] += count

    def release_pages(self, pages):
        counts = self._counts(pages)
        # Validate the complete transaction before changing any reference.
        if any(count > self._references[page] for page, count in counts.items()):
            raise ValueError("Packed KV release exceeds owned references")
        for page, count in counts.items():
            self._references[page] -= count
            if self._references[page] == 0:
                self._free.append(page)
