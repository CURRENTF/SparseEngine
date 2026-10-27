"""CPU ownership ledger for independent head slots; no private capacity slack."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeadAppendPlan:
    generation: int
    seq_ids: tuple[int, ...]
    rows: tuple[int, ...]
    positions: tuple[tuple[int, ...], ...]


class HeadSlotAllocator:
    def __init__(self, capacities: tuple[int, ...], heads: int, width: int, rows: int):
        if not capacities or min(*capacities, heads, width, rows) <= 0:
            raise ValueError("Head-slot capacities and dimensions must be positive.")
        self.capacities = tuple(capacities)
        self.heads, self.width, self.rows = heads, width, rows
        shape = (len(capacities), rows, heads, width)
        self.slots = np.full(shape, -1, dtype=np.int32)
        self.positions = np.full(shape, -1, dtype=np.int32)
        self.lengths = np.zeros(shape[:-1], dtype=np.int32)
        self.free = [np.arange(n, dtype=np.int32) for n in capacities]
        self.free_counts = np.asarray(capacities, dtype=np.int64)
        self.seq_rows: dict[int, int] = {}
        self.free_rows = list(reversed(range(rows)))
        self.generation = 0

    def append_capacity(self, seq_id: int, reserved: tuple[int, ...] = ()) -> int:
        available = min((int(f) - (reserved[i] if reserved else 0)) // self.heads
                        for i, f in enumerate(self.free_counts))
        row = self.seq_rows.get(seq_id)
        if row is None:
            return max(0, min(available, self.width)) if self.free_rows else 0
        return max(0, min(available, int((self.width - self.lengths[:, row]).min())))

    def plan_append(self, requests: list[tuple[int, list[int]]]) -> HeadAppendPlan:
        ids = tuple(int(sid) for sid, _ in requests)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("A batch must contain distinct sequence owners.")
        # Width bounds retained entries, not the original logical coordinates.
        # Remapped chains may outlive that width while retaining fewer KV pairs.
        position_limit = np.iinfo(self.positions.dtype).max
        if any(not pos or any(p < 0 or p > position_limit for p in pos)
               or any(b != a + 1 for a, b in zip(pos, pos[1:]))
               for _, pos in requests):
            raise ValueError("Append positions must be nonempty consecutive logical positions.")
        new = sum(sid not in self.seq_rows for sid in ids)
        need = self.heads * sum(len(pos) for _, pos in requests)
        if new > len(self.free_rows) or np.any(self.free_counts < need):
            raise RuntimeError("Insufficient head slots or resident rows for the whole batch.")
        next_rows = iter(reversed(self.free_rows))
        rows = tuple(self.seq_rows[sid] if sid in self.seq_rows else next(next_rows) for sid in ids)
        for (_, pos), row in zip(requests, rows):
            lens = self.lengths[:, row]
            if np.any(lens + len(pos) > self.width):
                raise RuntimeError("Head-slot index capacity exhausted before batch append.")
            previous = self.positions[np.arange(len(self.capacities))[:, None], row,
                                      np.arange(self.heads)[None, :], np.maximum(lens - 1, 0)]
            if np.any((lens > 0) & (pos[0] <= previous)):
                raise ValueError("Append must follow retained original logical positions.")
        return HeadAppendPlan(self.generation, ids, rows,
                              tuple(tuple(pos) for _, pos in requests))

    def apply_append(self, plan: HeadAppendPlan) -> tuple[np.ndarray, np.ndarray]:
        if plan.generation != self.generation:
            raise RuntimeError("Stale head-slot append plan; capacity changed.")
        total = sum(map(len, plan.positions))
        writes = np.empty((len(self.capacities), total, self.heads), dtype=np.int32)
        columns = np.empty_like(writes)
        layers = np.arange(len(self.capacities))[:, None, None]
        heads = np.arange(self.heads)[None, None, :]
        offset = 0
        for sid, row, pos in zip(plan.seq_ids, plan.rows, plan.positions):
            if sid not in self.seq_rows:
                assert self.free_rows.pop() == row
                self.seq_rows[sid] = row
            count = len(pos)
            batch_writes = writes[:, offset:offset + count]
            for l in range(len(self.capacities)):
                end = int(self.free_counts[l])
                selected = self.free[l][end - count * self.heads:end].reshape(count, self.heads)
                batch_writes[l] = selected
            cols = self.lengths[:, row, None, :] + np.arange(count, dtype=np.int32)[None, :, None]
            # Indices broadcast to [KV layer, appended token, KV head].
            self.slots[layers, row, heads, cols] = batch_writes
            self.positions[layers, row, heads, cols] = np.asarray(pos, dtype=np.int32)[None, :, None]
            columns[:, offset:offset + count] = cols
            self.free_counts -= count * self.heads
            self.lengths[:, row] += count
            offset += count
        self.generation += 1
        return writes, columns

    def validate_selection(self, seq_id: int, selections: list[list[np.ndarray]]) -> None:
        row = self.seq_rows[seq_id]
        if len(selections) != len(self.capacities) or any(len(s) != self.heads for s in selections):
            raise ValueError("Selection must contain every layer and head.")
        for l, heads in enumerate(selections):
            for h, indices in enumerate(heads):
                if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
                        or indices[0] < 0 or indices[-1] >= self.lengths[l, row, h]
                        or np.any(indices[1:] <= indices[:-1])):
                    raise ValueError("Head selections must be nonempty, ordered unique candidate indices.")

    def compact(self, seq_id: int, selections: list[list[np.ndarray]]) -> tuple[int, ...]:
        self.validate_selection(seq_id, selections)
        row = self.seq_rows[seq_id]
        released = []
        for l, heads in enumerate(selections):
            count = 0
            for h, indices in enumerate(heads):
                n = int(self.lengths[l, row, h])
                old = self.slots[l, row, h, :n].copy()
                drop = np.ones(n, dtype=bool)
                drop[indices] = False
                discarded = old[drop]
                free_start = int(self.free_counts[l])
                self.free[l][free_start:free_start + len(discarded)] = discarded
                self.free_counts[l] += len(discarded)
                self.slots[l, row, h, :len(indices)] = old[indices]
                self.positions[l, row, h, :len(indices)] = self.positions[l, row, h, indices].copy()
                self.slots[l, row, h, len(indices):n] = -1
                self.positions[l, row, h, len(indices):n] = -1
                self.lengths[l, row, h] = len(indices)
                count += len(discarded)
            released.append(count)
        self.generation += 1
        return tuple(released)

    def free_seq(self, seq_id: int) -> None:
        row = self.seq_rows.pop(seq_id, None)
        if row is None:
            return
        for l, h in np.ndindex(self.lengths[:, row].shape):
            n = int(self.lengths[l, row, h])
            start = int(self.free_counts[l])
            self.free[l][start:start + n] = self.slots[l, row, h, :n]
            self.free_counts[l] += n
        self.slots[:, row] = -1
        self.positions[:, row] = -1
        self.lengths[:, row] = 0
        self.free_rows.append(row)
        self.generation += 1

    def assert_invariants(self, reserved: tuple[int, ...] = ()) -> None:
        for l, capacity in enumerate(self.capacities):
            allocated = self.slots[l][self.slots[l] >= 0]
            free = self.free[l][:self.free_counts[l]]
            assert len(allocated) == int(self.lengths[l].sum())
            assert len(allocated) + len(free) == capacity
            all_ids = np.concatenate((allocated, free))
            assert np.array_equal(np.sort(all_ids), np.arange(capacity))
            if reserved:
                assert 0 <= reserved[l] <= len(free)
                assert capacity == len(allocated) + (len(free) - reserved[l]) + reserved[l]
