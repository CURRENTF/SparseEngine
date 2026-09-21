"""Independent physical-ownership oracle. Never call production capacity hooks.

The universe comes from the allocated pool, not its advertised free count.
A repeated ID, an unowned ID and a free/live alias are three different errors.
Observers are intentionally O(pool size): they run only in tests, after fences.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping


class ContractViolation(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractViolation(message)


def ids(values) -> tuple[int, ...]:
    """Read physical IDs, not a manager's summary/capacity calculation."""
    if hasattr(values, 'detach'):
        values = values.detach().cpu()
    if hasattr(values, 'tolist'):
        values = values.tolist()
    return tuple(int(value) for value in values)


@dataclass(frozen=True)
class Pool:
    name: str
    capacity: int
    free: tuple[int, ...]
    owners: tuple[tuple[Hashable, tuple[int, ...]], ...]

    def check(self) -> None:
        require(self.capacity >= 0, f'{self.name}: negative physical capacity')
        free = Counter(self.free)
        require(all(n == 1 for n in free.values()), f'{self.name}: duplicate free ID')
        owned: dict[int, Hashable] = {}
        for owner, values in self.owners:
            for value in values:
                require(value not in owned, f'{self.name}: {value} owned twice ({owner})')
                require(value not in free, f'{self.name}: free/live alias at {value}')
                owned[value] = owner
        found = set(free) | set(owned)
        expected = set(range(self.capacity))
        require(found == expected,
                f'{self.name}: missing={sorted(expected-found)[:8]}, '
                f'out_of_range={sorted(found-expected)[:8]}')


def check_rows(name: str, table, free_rows, by_seq: Mapping[int, int], lengths) -> None:
    capacity = int(table.shape[0])
    Pool(name, capacity, ids(free_rows),
         tuple((int(seq), (int(row),)) for seq, row in by_seq.items())).check()
    for seq, row in by_seq.items():
        require(0 <= int(lengths[row]) <= int(table.shape[1]),
                f'{name}: invalid row length for {seq}')
    for row in free_rows:
        require(int(lengths[row]) == 0, f'{name}: free row {row} has live length')


class ChainHarness:
    """Exclusive physical rows, plus (optionally) the real chain coordinator."""
    def __init__(self, manager, *, shared_slots: bool = False):
        self.manager = manager
        self.shared_slots = shared_slots

    def observe(self) -> tuple[Pool, ...]:
        m = self.manager
        if self.shared_slots:
            layouts = [('shared', m.free_slots_stack, m._num_free_slots,
                        m.buffer_req_to_token_slots, m.free_rows,
                        m.seq_id_to_row, m.row_seq_lens)]
        else:
            layouts = [(f'layer_{layer}', m.free_slots_stack[layer],
                        m._num_free_slots[layer], m.buffer_req_to_token_slots[layer],
                        m.free_rows[layer], m.seq_id_to_row[layer], m.row_seq_lens[layer])
                       for layer in m.kv_transformer_layer_indices()]
        pools = []
        resident_sets = []
        for name, stack, count, table, free_rows, rows, lengths in layouts:
            require(0 <= count <= len(stack), f'{name}: invalid free-stack pointer')
            check_rows(f'{name}/rows', table, free_rows, rows, lengths)
            resident_sets.append(set(rows))
            owners = tuple((int(seq), ids(table[row, :int(lengths[row])]))
                           for seq, row in sorted(rows.items()))
            pool = Pool(name, len(stack), ids(stack[:count]), owners)
            pool.check()
            pools.append(pool)
        require(all(s == resident_sets[0] for s in resident_sets),
                'chain: partially resident request at a stable boundary')
        return tuple(pools)

    def check(self, coordinator=None) -> tuple[Pool, ...]:
        pools = self.observe()
        if coordinator is None:
            return pools
        resident = {seq for seq, _ in pools[0].owners}
        records = list(coordinator.index.records.values())
        require(len({r.seq_id for r in records}) == len(records), 'chain: duplicate owner')
        # ACTIVE records can reserve a future row without having allocated it.
        for record in records:
            state = str(getattr(record.state, 'value', record.state)).lower()
            require(state in {'active', 'idle'}, 'chain: invalid logical state')
            sid = int(record.seq_id)
            if state == 'idle':
                require(not any(record.reserved_slots_by_layer) and record.reserved_rows == 0,
                        'chain: IDLE record retained an admission promise')
                if record.resident_rows:
                    require(sid in resident, 'chain: IDLE record has no resident row')
                    actual = dict(pools[0].owners)[sid] if self.shared_slots else None
                    lengths = ((len(actual),) * len(record.physical_slots_by_layer)
                               if self.shared_slots else
                               tuple(len(dict(pool.owners)[sid]) for pool in pools))
                    require(tuple(record.physical_slots_by_layer) == lengths,
                            'chain: finished residency differs from physical rows')
                else:
                    require(sid not in resident, 'chain: CPU-only record has a device row')
                    offload = coordinator.offload
                    require(offload is not None and sid in offload.snapshots,
                            'chain: CPU-only record has no snapshot')
                    snapshot = offload.snapshots[sid]
                    require(snapshot.valid and snapshot.completion is None,
                            'chain: CPU-only record has no completed snapshot')
        require(resident <= {int(r.seq_id) for r in records},
                'chain: resident row has no logical owner')
        self.check_host(coordinator.offload)
        return pools

    @staticmethod
    def check_host(controller) -> None:
        if controller is None:
            return
        actual = 0
        global_storages = set()
        for sid, snapshot in controller.snapshots.items():
            tensors = [tensor for pair in snapshot.kv.values() for tensor in pair]
            tensors.extend(snapshot.method.tensors.values())
            nbytes = 0
            local = set()
            for tensor in tensors:
                require(tensor.device.type == 'cpu', 'chain: host payload is not on CPU')
                storage = tensor.untyped_storage()
                key = (int(storage.data_ptr()), int(storage.nbytes()))
                if storage.nbytes() == 0:
                    continue
                require(key not in global_storages,
                        f'chain: snapshot {sid} aliases another snapshot')
                if key not in local:
                    nbytes += int(storage.nbytes())
                    local.add(key)
            global_storages.update(local)
            require(nbytes == snapshot.nbytes, 'chain: snapshot byte charge differs from storage')
            actual += nbytes
            if snapshot.completion is not None:
                require(bool(snapshot.keepalive), 'chain: DMA lost its source keepalive')
                require(not snapshot.valid, 'chain: unfinished DMA published a valid snapshot')
        require(actual == controller.used_bytes, 'chain: host byte accounting mismatch')
        require(actual <= controller.capacity_bytes, 'chain: host budget exceeded')

    def assert_released(self, seq_id: int, runtime=None) -> None:
        require(all(seq_id not in dict(pool.owners) for pool in self.observe()),
                f'chain: {seq_id} still owns physical storage')
        if runtime is not None:
            require(seq_id not in runtime.decode_reservations.requests,
                    'chain: terminal request retained a decode window')
            require(seq_id not in runtime._resident_seq_ids,
                    'chain: terminal request retained runtime residency')


class RadixHarness:
    """Shared blocks own storage; request rows borrow it with counted refs."""
    def __init__(self, manager, *, paged: bool = False):
        self.manager = manager
        self.paged = paged

    def _block_ids(self, block) -> tuple[int, ...]:
        if not block.residency.device_present:
            return ()
        if self.paged:
            require(block.payload.block_slot is not None, 'radix: device block has no page')
            return (int(block.payload.block_slot),)
        require(block.payload.token_slots is not None, 'radix: device block has no slots')
        return ids(block.payload.token_slots)

    def observe(self) -> Pool:
        m = self.manager
        blocks = m.prefix_cache.blocks
        owners = []
        index_owner = {}
        for block_id, block in blocks.items():
            residency = block.residency
            transfer = getattr(residency.transfer, 'value', residency.transfer)
            require(residency.device_present or residency.host_present,
                    'radix: block has neither host nor device storage')
            require(transfer in (None, 'h2d', 'd2h'), 'radix: invalid transfer state')
            if transfer == 'h2d':
                require(residency.host_present and residency.device_present,
                        'radix: H2D has no source or destination')
            if transfer == 'd2h':
                require(residency.device_present, 'radix: D2H source was freed early')
            values = self._block_ids(block)
            owners.append((block_id, values))
            for value in values:
                require(value not in index_owner, 'radix: distinct blocks alias storage')
                index_owner[value] = block_id
        expected_refs: Counter = Counter()
        held_by_seq: dict[int, set] = {}
        for mapping, keyed in ((m.seq_id_to_prefix_blocks, False),
                               (m.seq_id_to_materialized_blocks, True)):
            for seq_id, held in mapping.items():
                for block in held.values() if keyed else held:
                    bid = block.stable_block_id
                    require(blocks.get(bid) is block, 'radix: reference points outside the index')
                    expected_refs[bid] += 1
                    held_by_seq.setdefault(seq_id, set()).add(bid)
        for block_id, block in blocks.items():
            require(block.ref_count == expected_refs[block_id],
                    f'radix: reference count mismatch for {block_id.hex()[:12]}')
        require(set(held_by_seq) <= set(m.seq_id_to_row),
                'radix: request references survived its row')
        check_rows('radix/rows', m.buffer_req_to_token_slots, m.free_rows,
                   m.seq_id_to_row, m.row_seq_lens)
        for seq_id, row in m.seq_id_to_row.items():
            length = int(m.row_seq_lens[row])
            if self.paged:
                page_count = (length + m.page_size - 1) // m.page_size
                values = ids(m.buffer_req_to_page_slots[row, :page_count])
                # Token addresses independently agree with page-table geometry.
                token_ids = ids(m.buffer_req_to_token_slots[row, :length])
                require(all(token // m.page_size == values[i // m.page_size]
                            and token % m.page_size == i % m.page_size
                            for i, token in enumerate(token_ids)),
                        'radix: token/page table disagreement')
            else:
                values = ids(m.buffer_req_to_token_slots[row, :length])
            require(len(values) == len(set(values)), 'radix: repeated storage within one row')
            private = []
            for value in values:
                bid = index_owner.get(value)
                if bid is None:
                    private.append(value)
                else:
                    require(bid in held_by_seq.get(seq_id, set()),
                            'radix: row borrows a block without holding its reference')
            owners.append((int(seq_id), tuple(private)))
        if self.paged:
            stack, count = m.free_pages_stack, m._num_free_pages
        else:
            stack, count = m.free_slots_stack, m._num_free_slots
        require(0 <= count <= len(stack), 'radix: invalid free-stack pointer')
        pool = Pool('pages' if self.paged else 'slots', len(stack), ids(stack[:count]), tuple(owners))
        pool.check()
        return pool

    def assert_released(self, seq_id: int) -> None:
        self.observe()
        m = self.manager
        for mapping in (m.seq_id_to_row, m.seq_id_to_prefix_blocks,
                        m.seq_id_to_materialized_blocks, m.prefix_runtime_states,
                        m.pending_prefix_blocks):
            require(seq_id not in mapping, f'radix: released {seq_id} retained request state')


def check_cost_contract(manager, seq, horizons: Iterable[int]) -> None:
    """Check shape, sign and monotonicity, not an implementation's own formula.

    Physical sufficiency must additionally be tested by executing allocations and
    transitions. This helper alone is explicitly NOT a conformance certificate.
    """
    budgets = manager.decode_window_budgets()
    require(bool(budgets), 'decode: no physical pools')
    require(all(type(n) is int and n >= 0 for n in budgets.values()), 'decode: invalid budget')
    previous = {name: 0 for name in budgets}
    for horizon in sorted(set(horizons)):
        costs = manager.decode_window_costs(seq, horizon)
        require(not (costs.keys() - budgets.keys()), 'decode: unknown pool')
        require(all(type(n) is int and n >= 0 for n in costs.values()), 'decode: invalid cost')
        require(all(costs.get(name, 0) >= old for name, old in previous.items()),
                'decode: non-monotone future physical peak')
        previous = {name: costs.get(name, 0) for name in budgets}
