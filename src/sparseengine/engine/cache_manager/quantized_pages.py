"""CPU ownership of whole pages; allocation failure leaves all state intact."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PageAppend:
    start: int
    end: int
    pages: tuple[int, ...]


class QuantizedPagePool:
    def __init__(self, num_pages: int, page_size: int, max_length: int):
        if min(num_pages, page_size, max_length) <= 0:
            raise ValueError("Page capacity, size, and maximum length must be positive.")
        self.page_size = page_size
        self.max_length = max_length
        self.free = list(range(num_pages))
        self.pages: dict[int, list[int]] = {}
        self.lengths: dict[int, int] = {}

    def append_cost(self, seq_id: int, count: int) -> int:
        if count < 0:
            raise ValueError("Append token count cannot be negative.")
        start = self.lengths.get(seq_id, 0)
        if start + count > self.max_length:
            raise ValueError("Append exceeds max_model_len.")
        return ((start + count + self.page_size - 1) // self.page_size
                - (start + self.page_size - 1) // self.page_size)

    def append(self, seq_id: int, count: int) -> PageAppend:
        needed = self.append_cost(seq_id, count)
        if needed > len(self.free):
            raise RuntimeError(f"Out of quantized KV pages: need {needed}, free {len(self.free)}.")
        start = self.lengths.get(seq_id, 0)
        added = self.free[len(self.free) - needed:] if needed else []
        if needed:
            del self.free[-needed:]
        pages = self.pages.setdefault(seq_id, [])
        pages.extend(added)
        self.lengths[seq_id] = start + count
        return PageAppend(start, start + count, tuple(pages))

    def release(self, seq_id: int) -> None:
        if seq_id not in self.pages:
            raise ValueError(f"Unknown quantized cache sequence {seq_id}.")
        self.free.extend(self.pages.pop(seq_id))
        del self.lengths[seq_id]
