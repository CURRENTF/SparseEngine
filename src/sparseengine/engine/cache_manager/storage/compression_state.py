from __future__ import annotations

import torch


class CompressionStatePool:
    """Cache-coupled mutable carry; row ownership belongs to CacheManager."""

    def __init__(self, *, num_rows: int, ratio: int, head_dim: int, device):
        if num_rows <= 0 or ratio not in (4, 128) or head_dim not in (128, 512):
            raise ValueError("Invalid compression state dimensions")
        self.ratio = ratio
        self.head_dim = head_dim
        self.state = torch.zeros(num_rows, 8 if ratio == 4 else 128,
                                 head_dim * (4 if ratio == 4 else 2),
                                 device=device, dtype=torch.float32)

    @property
    def bytes_per_row(self):
        return self.state[0].numel() * self.state.element_size()

    def accounting_tensors(self):
        return (self.state,)

    def _validate_row(self, row):
        if not isinstance(row, int) or not 0 <= row < self.state.shape[0]:
            raise IndexError(f"Compression row {row} is outside the state pool")

    def reset_row(self, row):
        self._validate_row(row)
        self.state[row].zero_()

    def snapshot(self, row):
        self._validate_row(row)
        return self.state[row].clone()

    def restore(self, row, snapshot):
        self._validate_row(row)
        if snapshot.shape != self.state.shape[1:] or snapshot.dtype != self.state.dtype:
            raise ValueError("Compression prefix snapshot differs from the carry contract")
        if snapshot.device != self.state.device:
            raise ValueError("Compression prefix snapshot must be on the carry device")
        self.state[row].copy_(snapshot)

    def fork(self, source_row, destination_row):
        self._validate_row(source_row)
        self._validate_row(destination_row)
        self.state[destination_row].copy_(self.state[source_row])
