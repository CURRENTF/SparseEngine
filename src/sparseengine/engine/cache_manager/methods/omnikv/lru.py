"""Request-private GPU KV pools with observation-group LRU metadata."""

import torch

from sparseengine.kernels.triton.omnikv_lru import plan_lru


class OmniKVLRU:
    @staticmethod
    def metadata_bytes(rows, slots, capacity, selected, groups):
        return groups * rows * (slots * 4 + capacity * 12 + selected * 8 + 12)

    def __init__(self, storage, layer_groups, rows, capacity, selected, device, view):
        self.view = view
        self.layer_groups = layer_groups
        self.planned = set()
        self.parts = {
            layer: tuple(
                torch.empty(
                    rows * capacity + 1,
                    *shape,
                    dtype=storage.dtype,
                    device=device,
                )
                for shape in storage.shapes
            )
            for layer in layer_groups
        }
        for parts in self.parts.values():
            for part in parts:
                part[-1].zero_()
        self.metadata = {}
        for group in set(layer_groups.values()):
            self.metadata[group] = (
                torch.full(
                    (rows, storage.num_slots), -1, dtype=torch.int32, device=device
                ),
                torch.full((rows, capacity), -1, dtype=torch.int32, device=device),
                torch.zeros((rows, capacity), dtype=torch.int64, device=device),
                torch.zeros(rows, dtype=torch.int64, device=device),
                torch.empty((rows, selected), dtype=torch.int32, device=device),
                torch.empty((rows, selected), dtype=torch.int32, device=device),
                torch.zeros(rows, dtype=torch.int32, device=device),
            )

    def prepare(self, layer, table, rows, owners, lengths, writes):
        group = self.layer_groups[layer]
        data = self.metadata[group]
        if group not in self.planned:
            plan_lru(*data, table, rows, owners, lengths, writes, self.view)
            self.planned.add(group)
        return data[4]

    def plan(self, layer):
        return self.metadata[self.layer_groups[layer]][4]

    def misses(self, layer):
        data = self.metadata[self.layer_groups[layer]]
        return data[5], data[6]

    def invalidate(self, row):
        for directory, keys, ages, clock, *_ in self.metadata.values():
            directory[row].fill_(-1)
            keys[row].fill_(-1)
            ages[row].zero_()
            clock[row].zero_()

    def tensors(self):
        return [x for parts in self.parts.values() for x in parts] + [
            x for data in self.metadata.values() for x in data
        ]
