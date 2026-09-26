from __future__ import annotations

import torch

from .compression_state import CompressionStatePool


class SharedKVStateRows:
    """Bounded cache-owned window and carry rows for requests and prefixes.

    Every row has the same physical shape. Prefix snapshots consume a row in
    this pool, so their mutable-state bytes remain within the admission budget.
    Reserved rows serve graph padding and never enter the free list.
    """

    @staticmethod
    def bytes_per_row(compress_ratios, *, window_size=128, with_index=True):
        if window_size <= 0 or window_size & (window_size-1):
            raise ValueError("Shared KV window size must be a power of two")
        ratios = tuple(compress_ratios)
        if any(ratio not in (0, 4, 128) for ratio in ratios):
            raise ValueError("Unsupported shared KV compression ratio")
        window_bytes = ((584*window_size+575)//576)*576
        carry_bytes = sum((8*512*4 if r == 4 else 128*512*2)*4 for r in ratios if r)
        index_bytes = ratios.count(4)*8*128*4*4 if with_index else 0
        return len(ratios)*window_bytes + carry_bytes + index_bytes

    def __init__(self, *, num_rows, reserved_rows, compress_ratios, device,
                 window_size=128, with_index=True, window_storage=None):
        self.row_bytes = self.bytes_per_row(compress_ratios, window_size=window_size,
                                          with_index=with_index)
        if not 0 <= reserved_rows < num_rows:
            raise ValueError("Shared KV state pool requires allocatable rows")
        self.num_rows = num_rows
        self.reserved_rows = reserved_rows
        self.window_size = window_size
        window_bytes = ((584*window_size+575)//576)*576
        if window_storage is None:
            self.windows = {layer: torch.zeros(num_rows, window_bytes, dtype=torch.uint8, device=device)
                            for layer in range(len(compress_ratios))}
        else:
            self.windows = dict(window_storage)
            if set(self.windows) != set(range(len(compress_ratios))):
                raise ValueError("Window storage must cover every shared KV layer")
            for tensor in self.windows.values():
                if (tensor.shape != (num_rows, window_bytes) or tensor.dtype != torch.uint8
                        or tensor.device != torch.device(device) or not tensor.is_contiguous()):
                    raise ValueError("Bound window storage differs from the mutable-row contract")
        self.carry = {layer: CompressionStatePool(num_rows=num_rows, ratio=ratio, head_dim=512,
                                                 device=device)
                      for layer, ratio in enumerate(compress_ratios) if ratio}
        self.index_carry = {layer: CompressionStatePool(num_rows=num_rows, ratio=4, head_dim=128,
                                                       device=device)
                            for layer, ratio in enumerate(compress_ratios) if ratio == 4 and with_index}
        self._free = list(range(num_rows-1, reserved_rows-1, -1))
        self._owned = set()

    @property
    def num_free_rows(self):
        return len(self._free)

    def accounting_tensors(self):
        return (*self.windows.values(), *(p.state for p in self.carry.values()),
                *(p.state for p in self.index_carry.values()))

    def _validate_owned(self, row):
        if row not in self._owned:
            raise ValueError("Shared KV mutable-state row is not owned")

    def allocate(self):
        if not self._free:
            raise MemoryError("Shared KV mutable-state row budget exhausted")
        row = self._free.pop()
        for tensor in self.accounting_tensors():
            tensor[row].zero_()
        self._owned.add(row)
        return row

    def copy(self, source_row):
        self._validate_owned(source_row)
        if not self._free:
            raise MemoryError("Shared KV prefix snapshot row budget exhausted")
        row = self._free.pop()
        try:
            for tensor in self.accounting_tensors():
                tensor[row].copy_(tensor[source_row])
        except BaseException:
            self._free.append(row)
            raise
        self._owned.add(row)
        return row

    def release(self, row):
        self._validate_owned(row)
        self._owned.remove(row)
        self._free.append(row)
