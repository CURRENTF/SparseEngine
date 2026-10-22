"""Per-token FP8 writes to arbitrary slots, with an activation-precision tail."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["ROW", "START", "END"])
def _store(K, V, KD, VD, KS, VS, RK, RV, Slots, Rows, Lengths,
           K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
           V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
           H: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
           ROW, START, END):
    token, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(Slots + token)
    if slot < 0:
        return
    if Rows is None:
        row, position, end = ROW, START + token, END
    else:
        row = tl.load(Rows + token)
        end = tl.load(Lengths + token)
        position = end - 1
    d = tl.arange(0, D)
    key = tl.load(K + token * K0 + head * K1 + d * K2).to(tl.float32)
    value = tl.load(V + token * V0 + head * V1 + d * V2).to(tl.float32)
    ks = tl.maximum(tl.max(tl.abs(key), 0) / 448.0, 1.0e-30)
    vs = tl.maximum(tl.max(tl.abs(value), 0) / 448.0, 1.0e-30)
    target = (slot.to(tl.int64) * H + head) * D + d
    tl.store(KD + target, tl.minimum(tl.maximum(key / ks, -448.0), 448.0))
    tl.store(VD + target, tl.minimum(tl.maximum(value / vs, -448.0), 448.0))
    tl.store(KS + slot.to(tl.int64) * H + head, ks)
    tl.store(VS + slot.to(tl.int64) * H + head, vs)
    # Only the final partial interval writes the raw buffer. Prefill can span
    # many intervals, so writing every token would race on cyclic raw indices.
    if position >= (end // G) * G:
        raw = ((row * G + position % G) * H + head) * D + d
        tl.store(RK + raw, key)
        tl.store(RV + raw, value)


def fp8_store_tokens(key, value, payload, slots, *, rows=None, lengths=None,
                     row=0, start=0, end=0):
    if key.shape != value.shape or key.shape[0] != slots.numel():
        raise ValueError("FP8 writes require matching K/V and one destination slot per token.")
    if rows is not None and (rows.numel() != slots.numel() or lengths.shape != rows.shape):
        raise ValueError("FP8 decode writes require one row and length per token.")
    if not slots.numel():
        return
    h, d = payload.raw_key.shape[-2:]
    _store[(slots.numel(), h)](
        key, value, payload.k_cache, payload.v_cache, payload.key_scale, payload.value_scale,
        payload.raw_key, payload.raw_value, slots, rows, lengths,
        *key.stride(), *value.stride(), H=h, D=d, G=payload.page_size,
        ROW=row, START=start, END=end, num_warps=4, enable_fp_fusion=False,
    )
