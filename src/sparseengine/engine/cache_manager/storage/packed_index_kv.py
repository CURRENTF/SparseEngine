from __future__ import annotations

import torch

from .page_allocator import PhysicalPageAllocator


class PackedIndexKVPool:
    """Cache-owned per-layer E2M1 index keys with page-tail UE8M0 scales.

    The ratio-4 family can share its allocator with compressed attention KV
    and other layers. Tensor bytes are independent; references cover the
    entire family, never one key tensor alone.
    """

    def __init__(self, *, allocator: PhysicalPageAllocator, page_size: int, device):
        if page_size <= 0 or page_size & (page_size - 1):
            raise ValueError("Index KV page size must be a positive power of two")
        self.allocator = allocator
        self.page_size = page_size
        self.byte_storage = torch.empty(allocator.num_pages, page_size * 68,
                                        dtype=torch.uint8, device=device)

    def accounting_tensors(self):
        return (self.byte_storage,)
