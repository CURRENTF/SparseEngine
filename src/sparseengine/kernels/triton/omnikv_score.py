"""Candidate-domain per-head softmax followed by head-max, without a BHL probability tensor."""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["SB", "SH", "SL", "LS", "CAPACITY", "SPLITS"])
def _partial_stats(
    Scores, Lengths, Partial, SB, SH, SL,
    LS, HEADS: tl.constexpr, CAPACITY,
    SINK: tl.constexpr, RECENT: tl.constexpr, SCALE: tl.constexpr,
    SPLITS, BLOCK: tl.constexpr, KEEP: tl.constexpr,
):
    split = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    length = tl.minimum(tl.maximum(tl.load(Lengths + batch * LS) - RECENT - SINK, 0), CAPACITY - SINK)
    base = ((batch * HEADS + head) * SPLITS + split) * 2
    if split * BLOCK < length and length > KEEP:
        token = split * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(Scores + batch * SB + head * SH + (token + SINK) * SL,
                        mask=token < length, other=-float('inf')).to(tl.float32) * SCALE
        maximum = tl.max(value, 0)
        total = tl.sum(tl.exp(value - maximum), 0)
        tl.store(Partial + base, maximum)
        tl.store(Partial + base + 1, total)
    elif KEEP < 0:
        tl.store(Partial + base, -float('inf'))
        tl.store(Partial + base + 1, 0.)


@triton.jit(do_not_specialize=["SPLITS"])
def _merge_stats(Partial, Stats, Lengths, LS, HEADS: tl.constexpr,
                 SINK: tl.constexpr, RECENT: tl.constexpr, KEEP: tl.constexpr,
                 SCORE_BLOCK: tl.constexpr, SPLITS, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    length = tl.maximum(tl.load(Lengths + (row // HEADS) * LS) - SINK - RECENT, 0)
    if length > KEEP:
        part = tl.arange(0, BLOCK)
        valid = (part < SPLITS) & (part * SCORE_BLOCK < length)
        maximum = tl.load(Partial + (row * SPLITS + part) * 2, mask=valid, other=-float('inf'))
        total = tl.load(Partial + (row * SPLITS + part) * 2 + 1, mask=valid, other=0.)
        overall_max = tl.max(maximum, 0)
        overall_max = tl.where(overall_max == -float('inf'), 0., overall_max)
        overall_sum = tl.sum(total * tl.exp(maximum - overall_max), 0)
        tl.store(Stats + row * 2, overall_max)
        tl.store(Stats + row * 2 + 1, tl.where(overall_sum > 0, 1. / overall_sum, 0.))


@triton.jit(do_not_specialize=["SB", "SH", "SL", "LS", "CAPACITY"])
def _normalize_head_max(
    Scores, Lengths, Stats, Output,
    SB, SH, SL, LS,
    HEADS: tl.constexpr, CAPACITY, SINK: tl.constexpr, RECENT: tl.constexpr,
    SCALE: tl.constexpr, MIN_SCORE: tl.constexpr, BLOCK: tl.constexpr, HEAD_BLOCK: tl.constexpr,
    KEEP: tl.constexpr,
):
    tile = tl.program_id(0)
    batch = tl.program_id(1)
    token = tile * BLOCK + tl.arange(0, BLOCK)
    end = tl.minimum(tl.maximum(tl.load(Lengths + batch * LS) - RECENT, SINK), CAPACITY)
    valid = (token >= SINK) & (token < end)
    score = tl.full((BLOCK,), MIN_SCORE, tl.float32)
    if (end - SINK > KEEP) & (tile * BLOCK < end) & ((tile + 1) * BLOCK > SINK):
        head = tl.arange(0, HEAD_BLOCK)
        maximum = tl.load(Stats + (batch * HEADS + head) * 2, mask=head < HEADS, other=0.)
        reciprocal = tl.load(Stats + (batch * HEADS + head) * 2 + 1, mask=head < HEADS, other=0.)
        value = tl.load(Scores + batch * SB + head[:, None] * SH + token[None, :] * SL,
                        mask=(head[:, None] < HEADS) & valid[None, :], other=-float('inf')).to(tl.float32)
        probability = tl.exp(value * SCALE - maximum[:, None]) * reciprocal[:, None]
        score = tl.max(probability, 0)
        score = tl.where(valid, score, MIN_SCORE)
        if KEEP >= 0:
            tl.store(Output + batch * CAPACITY + token, score, mask=valid)
    if KEEP < 0:
        tl.store(Output + batch * CAPACITY + token, score, mask=token < CAPACITY)


def launch_omnikv_decode_scores(scores, context_lens, partial, stats, output, *,
                                sink, recent, scale, min_score, block=1024, output_block=128, selection_keep=-1):
    # With selection_keep >= 0 only long-row candidate scores are defined.
    # The length-aware selector never reads skipped rows or invalid tail storage.
    batch, heads, capacity = scores.shape
    splits = triton.cdiv(capacity - sink, block)
    _partial_stats[(splits, heads, batch)](
        scores, context_lens, partial, *scores.stride(), context_lens.stride(0),
        heads, capacity, sink, recent, scale, splits, block, selection_keep,
    )
    _merge_stats[(batch * heads,)](
        partial, stats, context_lens, context_lens.stride(0), heads,
        sink, recent, selection_keep, block, splits, triton.next_power_of_2(splits),
    )
    _normalize_head_max[(triton.cdiv(capacity, output_block), batch)](
        scores, context_lens, stats, output, *scores.stride(), context_lens.stride(0),
        heads, capacity, sink, recent, scale, min_score, output_block, triton.next_power_of_2(heads),
        KEEP=selection_keep, enable_fp_fusion=False,
    )


@triton.jit
def _ordered_score(value):
    value = value.to(tl.float32)
    value = tl.where(value == 0., 0., value)
    bits = value.to(tl.uint32, bitcast=True)
    return tl.where((bits & 0x80000000) != 0, bits ^ 0xFFFFFFFF, bits | 0x80000000)


@triton.jit
def _select_history(Scores, Lengths, Output, SB, SL, SINK: tl.constexpr,
                    K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    length = tl.load(Lengths + row)
    lanes = tl.arange(0, BLOCK)
    if length <= K:
        for start in range(0, K, BLOCK):
            out = start + lanes
            tl.store(Output + row * K + out, tl.where(out < length, out + SINK, -1), mask=out < K)
    else:
        # Four radix passes locate the exact kth FP32 key without sorting a
        # capacity-sized vector. Each pass reads only this row's valid history.
        bins = tl.arange(0, 256)
        prefix = tl.full((), 0, tl.uint32)
        rank = tl.full((), K, tl.int32)
        for shift in tl.static_range(24, -1, -8):
            hist = tl.full((256,), 0, tl.int32)
            for start in range(0, length, BLOCK):
                index = start + lanes
                value = tl.load(Scores + row * SB + (index + SINK) * SL,
                                mask=index < length, other=0.)
                key = _ordered_score(value)
                matches = index < length
                if shift < 24:
                    matches = matches & ((key >> (shift + 8)) == (prefix >> (shift + 8)))
                hist += tl.histogram(((key >> shift) & 255).to(tl.int32), 256, mask=matches)
            greater = tl.sum(hist, 0) - tl.cumsum(hist, 0)
            chosen = (greater < rank) & (greater + hist >= rank)
            byte = tl.max(tl.where(chosen, bins, 0), 0)
            rank -= tl.sum(tl.where(chosen, greater, 0), 0)
            prefix = prefix | (byte.to(tl.uint32) << shift)
        strict_count = tl.full((), 0, tl.int32)
        tie_count = tl.full((), 0, tl.int32)
        for start in range(0, length, BLOCK):
            index = start + lanes
            value = tl.load(Scores + row * SB + (index + SINK) * SL,
                            mask=index < length, other=0.)
            key = _ordered_score(value)
            strict = (index < length) & (key > prefix)
            equal = (index < length) & (key == prefix)
            strict_position = strict_count + tl.cumsum(strict.to(tl.int32), 0) - 1
            tie_position = tie_count + tl.cumsum(equal.to(tl.int32), 0) - 1
            # Boundary ties retain the smallest logical indices deterministically.
            tl.store(Output + row * K + strict_position, index + SINK, mask=strict)
            tl.store(Output + row * K + K - rank + tie_position,
                     index + SINK, mask=equal & (tie_position < rank))
            strict_count += tl.sum(strict.to(tl.int32), 0)
            tie_count += tl.sum(equal.to(tl.int32), 0)


def select_omnikv_history(scores, lengths, k, *, sink):
    import torch
    output = torch.empty((scores.shape[0], k), device=scores.device, dtype=torch.int32)
    _select_history[(scores.shape[0],)](
        scores, lengths, output, *scores.stride(), sink, k, 1024,
    )
    return output
