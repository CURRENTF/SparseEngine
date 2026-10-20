"""Pinned output leases, released only after DMA and host consumption finish."""
from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class HostOutputBuffer:
    storage: torch.Tensor
    view: torch.Tensor
    key: tuple[int, torch.dtype]
    nbytes: int


class HostOutputPool:
    def __init__(self):
        self.free = OrderedDict()
        self.live_bytes = self.peak_live_bytes = self.cached_bytes = 0
        self.allocations = self.reuses = 0

    def acquire(self, tensor):
        capacity = 1 << max(0, (tensor.numel() - 1).bit_length())
        key = (capacity, tensor.dtype)
        bucket = self.free.get(key)
        if bucket:
            buffer = bucket.pop()
            self.cached_bytes -= buffer.nbytes
            self.reuses += 1
            if not bucket:
                del self.free[key]
        else:
            storage = torch.empty(capacity, dtype=tensor.dtype, device="cpu", pin_memory=True)
            buffer = HostOutputBuffer(storage, storage[:tensor.numel()].view(tensor.shape),
                                      key, capacity * tensor.element_size())
            self.allocations += 1
        if buffer.view.shape != tensor.shape:
            buffer.view = buffer.storage[:tensor.numel()].view(tensor.shape)
        self.live_bytes += buffer.nbytes
        self.peak_live_bytes = max(self.peak_live_bytes, self.live_bytes)
        return buffer.view, buffer

    def release(self, buffers):
        for buffer in buffers:
            self.live_bytes -= buffer.nbytes
            self.free.setdefault(buffer.key, []).append(buffer)
            self.free.move_to_end(buffer.key)
            self.cached_bytes += buffer.nbytes
        # Changing batch/logprob shapes must not accumulate an unbounded cache.
        while self.cached_bytes > self.peak_live_bytes:
            _, bucket = self.free.popitem(last=False)
            self.cached_bytes -= sum(b.nbytes for b in bucket)
