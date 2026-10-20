"""Request-indexed device feedback and sampling-state mutation."""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N", "CAPACITY"])
def gather_feedback(tokens, metadata, inputs, selected, N,
                    CAPACITY, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    source = tl.where(row < N, row, 0)
    slot = tl.load(metadata + source)
    seed = tl.load(metadata + N + source)
    ready = tl.load(metadata + 2 * N + source)
    previous = tl.load(tokens + slot, ready != 0, 0)
    value = tl.where(ready != 0, previous, seed)
    tl.store(inputs + row, value, row < CAPACITY)
    tl.store(selected + row, value, row < N)


@triton.jit(do_not_specialize=["N", "VOCAB"])
def commit_feedback(sampled, metadata, feedback, flags, N, VOCAB, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = (row < N) & (tl.load(metadata + 4 * N + row, row < N, 0) != 0)
    slot = tl.load(metadata + row, active, 0)
    token = tl.load(sampled + row, active, 0)
    tl.store(feedback + slot, token, active)
    penalty = tl.load(metadata + 3 * N + row, active, -1)
    tl.store(flags + penalty * VOCAB + token, 3, active & (penalty >= 0))


@triton.jit(do_not_specialize=["N", "STATE_CAPACITY"])
def apply_penalties(logits, flags, metadata, parameters, output,
                    STRIDE_ROW: tl.constexpr, STRIDE_COL: tl.constexpr,
                    N, VOCAB: tl.constexpr, STATE_CAPACITY, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(metadata + 3 * N + row)
    state = tl.load(flags + slot * VOCAB + column,
                    (slot >= 0) & (column < VOCAB), 0)
    repetition = tl.load(parameters + slot, slot >= 0, 1.)
    presence = tl.load(parameters + STATE_CAPACITY + slot, slot >= 0, 0.)
    value = tl.load(logits + row * STRIDE_ROW + column * STRIDE_COL,
                    column < VOCAB, 0).to(tl.float32)
    # Match Torch's separately rounded division/multiplication then subtraction.
    repeated = tl.where(value > 0, tl.div_rn(value, repetition), value * repetition)
    value = tl.where((state & 1) != 0, repeated, value)
    value = value - tl.where((state & 2) != 0, presence, 0.)
    tl.store(output + row * VOCAB + column, value, column < VOCAB)
