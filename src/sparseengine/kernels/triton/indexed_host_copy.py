"""GPU-indexed copies between CUDA and UVA-mapped pinned host pools."""

import triton
import triton.language as tl


@triton.jit
def _copy_rows(
    SRC,
    DST_PTR,
    SLOTS,
    SLOT_MAP,
    WIDTH: tl.constexpr,
    SRC_STRIDE: tl.constexpr,
    COMPONENT: tl.constexpr,
    BLOCK: tl.constexpr,
    DIRECT: tl.constexpr = False,
):
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(SLOTS + row)
    if SLOT_MAP is not None and COMPONENT == 0:  # noqa: SIM102 (constexpr guard)
        if tl.program_id(1) == 0:
            tl.store(SLOT_MAP + slot, slot, slot >= 0)
    if DIRECT:
        dst = DST_PTR
    else:
        dst = tl.load(DST_PTR + COMPONENT).to(tl.pointer_type(SRC.dtype.element_ty))
    x = tl.load(SRC + row * SRC_STRIDE + d, (slot >= 0) & (d < WIDTH), 0)
    tl.store(dst + slot.to(tl.int64) * WIDTH + d, x, (slot >= 0) & (d < WIDTH))


@triton.jit
def _gather(
    SRC_PTR,
    DST,
    TABLE,
    ROWS,
    LENGTHS,
    SLOT_MAP,
    EXCLUDE_SLOTS,
    TABLE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    CAPACITY: tl.constexpr,
    COMPONENT: tl.constexpr,
    SKIP_LAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    row = tl.load(ROWS + batch)
    length = tl.load(LENGTHS + batch)
    src = tl.load(SRC_PTR + COMPONENT).to(tl.pointer_type(DST.dtype.element_ty))
    for tile in range(
        tl.program_id(1), tl.cdiv(CAPACITY * WIDTH, BLOCK), tl.num_programs(1)
    ):
        offset = tile * BLOCK + tl.arange(0, BLOCK)
        token = offset // WIDTH
        d = offset % WIDTH
        valid = (token < length - SKIP_LAST) & (token < CAPACITY)
        active = True
        if EXCLUDE_SLOTS is not None:
            current_slot = tl.load(EXCLUDE_SLOTS + batch)
            active = current_slot >= 0
            valid = valid & active
        slot = tl.load(TABLE + row * TABLE_STRIDE + token, valid, 0)
        if EXCLUDE_SLOTS is not None:
            valid = valid & (slot != current_slot)
        source_slot = slot
        if SLOT_MAP is not None:
            source_slot = tl.load(SLOT_MAP + slot, valid, 0)
        x = tl.load(src + source_slot.to(tl.int64) * WIDTH + d, valid, 0)
        target = batch * CAPACITY + token
        # Padded rows have replicated lengths but no write slot. Do not read
        # their host KV, and leave finite zeros for their private attention view.
        store_mask = valid | ((not active) & (token < CAPACITY))
        tl.store(DST + target.to(tl.int64) * WIDTH + d, x, store_mask)


@triton.jit
def _gather_cached(
    SRC_PTR,
    CACHE,
    TABLE,
    ROWS,
    LENGTHS,
    SLOT_MAP,
    EXCLUDE_SLOTS,
    PLAN,
    MISS_TOKENS,
    MISS_COUNTS,
    TABLE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    CAPACITY: tl.constexpr,
    COMPONENT: tl.constexpr,
    SKIP_LAST: tl.constexpr,
    BATCH: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    requests = tl.arange(0, BATCH_BLOCK)
    counts = tl.load(MISS_COUNTS + requests, requests < BATCH, 0)
    tiles = tl.cdiv(counts * WIDTH, BLOCK)
    ends = tl.cumsum(tiles, 0)
    total = tl.sum(tiles, 0)
    source = tl.load(SRC_PTR + COMPONENT).to(tl.pointer_type(CACHE.dtype.element_ty))
    # Share the fixed CTA budget across misses: per-request quotas leave the
    # transfer waiting for whichever request has the most newly selected KV.
    for tile in range(tl.program_id(0), total, tl.num_programs(0)):
        batch = tl.sum((tile >= ends).to(tl.int32), 0)
        begin = tl.sum(tl.where(requests < batch, tiles, 0), 0)
        offsets = (tile - begin) * BLOCK + tl.arange(0, BLOCK)
        item = offsets // WIDTH
        count = tl.load(MISS_COUNTS + batch)
        token = tl.load(MISS_TOKENS + batch * CAPACITY + item, item < count, 0)
        length = tl.load(LENGTHS + batch)
        row = tl.load(ROWS + batch)
        valid = (item < count) & (token < length - SKIP_LAST) & (token < CAPACITY)
        if EXCLUDE_SLOTS is not None:
            current = tl.load(EXCLUDE_SLOTS + batch)
            valid = valid & (current >= 0)
        slot = tl.load(TABLE + row * TABLE_STRIDE + token, valid, 0)
        if EXCLUDE_SLOTS is not None:
            valid = valid & (slot != current)
        source_slot = slot
        if SLOT_MAP is not None:
            source_slot = tl.load(SLOT_MAP + slot, valid, 0)
        entry = tl.load(PLAN + batch * CAPACITY + token, valid, 0)
        valid = valid & (entry < 0)
        position = (-entry - 1).to(tl.int64)
        value = tl.load(
            source + source_slot.to(tl.int64) * WIDTH + offsets % WIDTH, valid, 0
        )
        tl.store(CACHE + position * WIDTH + offsets % WIDTH, value, valid)


@triton.jit
def _gather_prefill(
    SRC_PTR,
    CURRENT,
    DST,
    TABLE,
    ROWS,
    LENGTHS,
    CU_QUERY,
    SLOT_MAP,
    TABLE_STRIDE: tl.constexpr,
    CURRENT_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    CAPACITY: tl.constexpr,
    COMPONENT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    row = tl.load(ROWS + batch)
    length = tl.load(LENGTHS + batch)
    query_begin = tl.load(CU_QUERY + batch)
    query_end = tl.load(CU_QUERY + batch + 1)
    history = length - (query_end - query_begin)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    token = offsets // WIDTH
    feature = offsets % WIDTH
    valid = (token < length) & (token < CAPACITY)
    slot = tl.load(TABLE + row * TABLE_STRIDE + token, valid, 0)
    # CPU write-through has already preserved the chunk. Avoid reading it back
    # over the host link; only earlier chunks and shared prefix KV need that.
    from_host = valid & (token < history)
    host_slot = tl.load(SLOT_MAP + slot, from_host, 0)
    source = tl.load(SRC_PTR + COMPONENT).to(tl.pointer_type(DST.dtype.element_ty))
    old = tl.load(source + host_slot.to(tl.int64) * WIDTH + feature, from_host, 0)
    current = tl.load(
        CURRENT + (query_begin + token - history).to(tl.int64) * CURRENT_STRIDE + feature,
        valid & (token >= history), 0,
    )
    tl.store(
        DST + slot.to(tl.int64) * WIDTH + feature,
        tl.where(token < history, old, current), valid,
    )


@triton.jit
def _append(
    SRC,
    DST,
    LENGTHS,
    WRITE_SLOTS,
    TABLE,
    ROWS,
    PLAN,
    TABLE_STRIDE: tl.constexpr,
    SEEK_BLOCK: tl.constexpr,
    WIDTH: tl.constexpr,
    SRC_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(LENGTHS + row)
    slot = tl.load(WRITE_SLOTS + row)
    if TABLE is not None:
        token = tl.arange(0, SEEK_BLOCK)
        table_row = tl.load(ROWS + row)
        selected = tl.load(
            TABLE + table_row * TABLE_STRIDE + token,
            (token < length) & (token < CAPACITY),
            -1,
        )
        position = tl.max(tl.where((selected == slot) & (token < length), token, -1), 0)
        length = position + 1
    valid = (slot >= 0) & (length > 0) & (length <= CAPACITY) & (d < WIDTH)
    x = tl.load(SRC + row * SRC_STRIDE + d, valid, 0)
    if PLAN is None:
        tl.store(DST + (row * CAPACITY + length - 1).to(tl.int64) * WIDTH + d, x, valid)
    if PLAN is not None:
        entry = tl.load(
            PLAN + row * CAPACITY + length - 1,
            (slot >= 0) & (length > 0) & (length <= CAPACITY),
            0,
        )
        cache_slot = tl.where(entry < 0, -entry - 1, entry).to(tl.int64)
        tl.store(DST + cache_slot * WIDTH + d, x, valid)


@triton.jit
def _transfer(
    SRC_PTR,
    DST_PTR,
    SOURCE,
    TARGET,
    SLOT_MAP,
    WIDTH: tl.constexpr,
    COMPONENT: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source = tl.load(SOURCE + row)
    target = tl.load(TARGET + row)
    if SLOT_MAP is not None:
        source = tl.load(SLOT_MAP + source)
    component = tl.program_id(2) if COMPONENT is None else COMPONENT
    src = tl.load(SRC_PTR + component).to(tl.pointer_type(DTYPE))
    dst = tl.load(DST_PTR + component).to(tl.pointer_type(DTYPE))
    x = tl.load(src + source.to(tl.int64) * WIDTH + d, d < WIDTH, 0)
    tl.store(dst + target.to(tl.int64) * WIDTH + d, x, d < WIDTH)


@triton.jit
def _gather_prefill_history(
    SOURCE_PTRS,
    DEST,
    TABLE,
    ROWS,
    LENGTHS,
    CU_QUERY,
    SLOT_MAP,
    TABLE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    COMPONENT: tl.constexpr,
    BATCH: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    requests = tl.arange(0, BATCH_BLOCK)
    lengths = tl.load(LENGTHS + requests, requests < BATCH, 0)
    begins = tl.load(CU_QUERY + requests, requests < BATCH, 0)
    ends_q = tl.load(CU_QUERY + requests + 1, requests < BATCH, 0)
    history = lengths - (ends_q - begins)
    tiles = tl.cdiv(history * WIDTH, BLOCK)
    ends = tl.cumsum(tiles, 0)
    total = tl.sum(tiles, 0)
    source = tl.load(SOURCE_PTRS + COMPONENT).to(tl.pointer_type(DEST.dtype.element_ty))
    for tile in range(tl.program_id(0), total, tl.num_programs(0)):
        batch = tl.sum((tile >= ends).to(tl.int32), 0)
        begin = tl.sum(tl.where(requests < batch, tiles, 0), 0)
        offsets = (tile - begin) * BLOCK + tl.arange(0, BLOCK)
        token, feature = offsets // WIDTH, offsets % WIDTH
        count = tl.sum(tl.where(requests == batch, history, 0), 0)
        row = tl.load(ROWS + batch)
        slot = tl.load(TABLE + row * TABLE_STRIDE + token, token < count, 0)
        host_slot = tl.load(SLOT_MAP + slot, token < count, 0)
        value = tl.load(
            source + host_slot.to(tl.int64) * WIDTH + feature, token < count, 0
        )
        tl.store(DEST + slot.to(tl.int64) * WIDTH + feature, value, token < count)
