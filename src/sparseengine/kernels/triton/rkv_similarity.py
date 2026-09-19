"""R-KV representative removal and deterministic FP32 column sums."""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["ROWS", "LENGTH", "ROW_START"])
def _representatives(Sim, Representatives, ROWS, LENGTH, ROW_START,
                     BLOCK: tl.constexpr):
    row = tl.program_id(0)
    unit = tl.program_id(1).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    last = tl.full((), 0, tl.int32)
    for start in range(0, LENGTH, BLOCK):
        col = start + offsets
        value = tl.load(Sim + (unit * ROWS + row) * LENGTH + col,
                        col < LENGTH, other=0).to(tl.float32)
        # The threshold chooses a representative, NOT which values are summed.
        match = (col < LENGTH) & (col != ROW_START + row) & (value > 0.5)
        last = tl.maximum(last, tl.max(tl.where(match, col, 0), axis=0))
    tl.store(Representatives + unit * ROWS + row, last)


@triton.jit(do_not_specialize=["ROWS", "LENGTH", "ROW_START", "PARTS"])
def _column_partials(Sim, Representatives, Partial, ROWS, LENGTH, ROW_START,
                     PARTS, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    part = tl.program_id(0)
    col = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    unit = tl.program_id(2).to(tl.int64)
    row = part * BLOCK_R + tl.arange(0, BLOCK_R)
    rep = tl.load(Representatives + unit * ROWS + row, row < ROWS, other=0)
    value = tl.load(Sim + (unit * ROWS + row[:, None]) * LENGTH + col[None, :],
                    (row[:, None] < ROWS) & (col[None, :] < LENGTH), other=0).to(tl.float32)
    removed = (col[None, :] == ROW_START + row[:, None]) | (col[None, :] == rep[:, None])
    value = tl.where(removed, 0.0, value)
    total = tl.sum(value, axis=0)
    tl.store(Partial + (unit * PARTS + part) * LENGTH + col, total, col < LENGTH)


@triton.jit(do_not_specialize=["LENGTH", "PARTS"])
def _merge_columns(Partial, Output, LENGTH, PARTS,
                   BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    col = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    unit = tl.program_id(1).to(tl.int64)
    offsets = tl.arange(0, BLOCK_R)
    total = tl.full((BLOCK_C,), 0, tl.float32)
    for start in range(0, PARTS, BLOCK_R):
        part = start + offsets
        values = tl.load(Partial + (unit * PARTS + part[:, None]) * LENGTH + col[None, :],
                         (part[:, None] < PARTS) & (col[None, :] < LENGTH), other=0)
        total += tl.sum(values, axis=0)
    tl.store(Output + unit * LENGTH + col, total, col < LENGTH)


def similarity_column_sums(sim, row_start, representatives, partial, output, *, row_block):
    units, rows, length = sim.shape
    parts = partial.shape[1]
    _representatives[(rows, units)](sim, representatives, rows, length, row_start,
                                    BLOCK=4096, num_warps=4)
    _column_partials[(parts, triton.cdiv(length, 128), units)](
        sim, representatives, partial, rows, length, row_start, parts,
        BLOCK_R=row_block, BLOCK_C=128, num_warps=8)
    _merge_columns[(triton.cdiv(length, 128), units)](
        partial, output, length, parts, BLOCK_R=32, BLOCK_C=128, num_warps=4)
    return output
