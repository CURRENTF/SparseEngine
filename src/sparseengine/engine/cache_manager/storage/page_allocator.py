from __future__ import annotations

from collections import Counter


class PhysicalPageAllocator:
    """One ownership domain for matching storage across layers and payloads.

    A page reference retains all tensors in its cache family. CacheManager
    changes ownership once; individual layer stores only consume addresses.
    Reserved pages belong to graph scratch and never enter admission budgets.
    """

    def __init__(self, *, num_pages: int, reserved_pages: int):
        if num_pages <= 0 or not 0 <= reserved_pages < num_pages:
            raise ValueError("Physical pages require positive allocatable capacity")
        self.num_pages = num_pages
        self.reserved_pages = reserved_pages
        self._references = [0] * num_pages
        self._free = list(range(num_pages - 1, reserved_pages - 1, -1))

    @property
    def num_free_pages(self):
        return len(self._free)

    def allocate_pages(self, count: int) -> tuple[int, ...]:
        if count < 0:
            raise ValueError("Cannot allocate a negative page count")
        if count > self.num_free_pages:
            raise MemoryError(f"Physical cache needs {count} pages, has {self.num_free_pages}")
        pages = tuple(self._free.pop() for _ in range(count))
        for page in pages:
            self._references[page] = 1
        return pages

    def _counts(self, pages):
        counts = Counter(pages)
        for page in counts:
            if not isinstance(page, int) or not self.reserved_pages <= page < self.num_pages:
                raise ValueError(f"Invalid or reserved physical page {page}")
            if self._references[page] <= 0:
                raise ValueError(f"Physical page {page} is not allocated")
        return counts

    def retain_pages(self, pages):
        for page, count in self._counts(pages).items():
            self._references[page] += count

    def release_pages(self, pages):
        counts = self._counts(pages)
        if any(count > self._references[page] for page, count in counts.items()):
            raise ValueError("Physical page release exceeds owned references")
        for page, count in counts.items():
            self._references[page] -= count
            if self._references[page] == 0:
                self._free.append(page)
