"""Exact per-request LRU planning shared by an OmniKV observation group."""

import triton
import triton.language as tl


@triton.jit
def _lookup(
    DIRECTORY,
    AGES,
    CLOCK,
    PLAN,
    MISS_COUNTS,
    TABLE,
    ROWS,
    OWNERS,
    LENGTHS,
    WRITES,
    SLOTS: tl.constexpr,
    CACHE: tl.constexpr,
    CAP: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    current = tl.load(WRITES + batch)
    if current >= 0:
        owner = tl.load(OWNERS + batch)
        row = tl.load(ROWS + batch)
        length = tl.load(LENGTHS + batch)
        token = tl.arange(0, BLOCK)
        valid = (token < length) & (token < CAP)
        slot = tl.load(TABLE + row * STRIDE + token, valid, 0)
        position = tl.load(DIRECTORY + owner * SLOTS + slot, valid, -1)
        # The current projection has not been produced at prefetch time.
        hit = valid & (position >= 0)
        clock = tl.load(CLOCK + owner) + 1
        tl.store(CLOCK + owner, clock)
        tl.store(AGES + owner * CACHE + position, clock, hit)
        tl.store(PLAN + batch * CAP + token, tl.where(hit, position, -1), token < CAP)
        tl.store(MISS_COUNTS + batch, tl.sum((valid & ~hit).to(tl.int32), 0))
    else:
        tl.store(MISS_COUNTS + batch, 0)


@triton.jit
def _choose_victims(
    AGES,
    CLOCK,
    PLAN,
    VICTIMS,
    OWNERS,
    LENGTHS,
    WRITES,
    CACHE: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    if tl.load(WRITES + batch) >= 0:
        owner = tl.load(OWNERS + batch)
        i = tl.arange(0, BLOCK)
        length = tl.load(LENGTHS + batch)
        entry = tl.load(PLAN + batch * CAP + i, i < CAP, 0)
        count = tl.sum(((i < CAP) & (i < length) & (entry < 0)).to(tl.int32), 0)
        if count > 0:
            clock = tl.load(CLOCK + owner)
            age = tl.load(AGES + owner * CACHE + i, i < CACHE, clock + 1)
            # Find the smallest timestamp containing enough victims. This is
            # exact step-granularity LRU without sorting the whole cache.
            low = tl.full((), 0, tl.int64)
            high = clock
            while low < high:
                middle = (low + high) // 2
                enough = (
                    tl.sum(((i < CACHE) & (age <= middle)).to(tl.int32), 0) >= count
                )
                high = tl.where(enough, middle, high)
                low = tl.where(enough, low, middle + 1)
            older = (i < CACHE) & (age < low)
            tied = (i < CACHE) & (age == low)
            remaining = count - tl.sum(older.to(tl.int32), 0)
            tied_rank = tl.cumsum(tied.to(tl.int32), 0)
            chosen = older | (tied & (tied_rank <= remaining))
            rank = tl.cumsum(chosen.to(tl.int32), 0) - 1
            tl.store(VICTIMS + batch * CAP + rank, i, chosen)


@triton.jit
def _admit(
    DIRECTORY,
    KEYS,
    AGES,
    CLOCK,
    PLAN,
    VICTIMS,
    TABLE,
    ROWS,
    OWNERS,
    LENGTHS,
    WRITES,
    VIEW,
    VIEW_STRIDE: tl.constexpr,
    PADDING_SLOT: tl.constexpr,
    SLOTS: tl.constexpr,
    CACHE: tl.constexpr,
    CAP: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    if tl.load(WRITES + batch) >= 0:
        owner = tl.load(OWNERS + batch)
        row = tl.load(ROWS + batch)
        clock = tl.load(CLOCK + owner)
        i = tl.arange(0, BLOCK)
        length = tl.load(LENGTHS + batch)
        valid = (i < length) & (i < CAP)
        old_plan = tl.load(PLAN + batch * CAP + i, i < CAP, -1)
        missing = valid & (old_plan < 0)
        rank = tl.cumsum(missing.to(tl.int32), 0) - 1
        victim = tl.load(VICTIMS + batch * CAP + rank, missing, 0)
        previous = tl.load(KEYS + owner * CACHE + victim, missing, -1)
        tl.store(DIRECTORY + owner * SLOTS + previous, -1, missing & (previous >= 0))
        slot = tl.load(TABLE + row * STRIDE + i, valid, 0)
        tl.store(KEYS + owner * CACHE + victim, slot, missing)
        tl.store(AGES + owner * CACHE + victim, clock, missing)
        tl.store(DIRECTORY + owner * SLOTS + slot, victim, missing)
        # Victim positions are no longer needed after admission. Reuse this
        # workspace as the compact selected-token list consumed by host copies.
        tl.store(VICTIMS + batch * CAP + rank, i, missing)
        position = owner * CACHE + tl.where(missing, victim, old_plan)
        # Negative entries encode a host miss; positive entries are GPU hits.
        tl.store(
            PLAN + batch * CAP + i, tl.where(missing, -position - 1, position), valid
        )
        tl.store(
            VIEW + batch * VIEW_STRIDE + i,
            tl.where(valid, position, PADDING_SLOT),
            i < CAP,
        )
    else:
        i = tl.arange(0, BLOCK)
        tl.store(VIEW + batch * VIEW_STRIDE + i, PADDING_SLOT, i < CAP)


def plan_lru(
    directory,
    keys,
    ages,
    clock,
    plan,
    victims,
    miss_counts,
    table,
    rows,
    owners,
    lengths,
    writes,
    view,
):
    capacity = plan.shape[1]
    cache = keys.shape[1]
    args = (
        directory,
        ages,
        clock,
        plan,
        miss_counts,
        table,
        rows,
        owners,
        lengths,
        writes,
        directory.shape[1],
        cache,
        capacity,
        table.stride(0),
    )
    _lookup[(rows.numel(),)](*args, triton.next_power_of_2(capacity))
    _choose_victims[(rows.numel(),)](
        ages,
        clock,
        plan,
        victims,
        owners,
        lengths,
        writes,
        cache,
        capacity,
        triton.next_power_of_2(cache),
        num_warps=8,
    )
    _admit[(rows.numel(),)](
        directory,
        keys,
        ages,
        clock,
        plan,
        victims,
        table,
        rows,
        owners,
        lengths,
        writes,
        view,
        view.stride(0),
        keys.numel(),
        directory.shape[1],
        cache,
        capacity,
        table.stride(0),
        triton.next_power_of_2(capacity),
        num_warps=8,
    )
