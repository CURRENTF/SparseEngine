"""Page codecs and decode over packed pages plus an unquantized tail."""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_append(K, V, KD, VD, KS, KM, VS, VM, RK, RV, C, Slots, Rows, Lengths, Writes,
                       K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
                       V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
                       SLOT_STRIDE: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                       G: tl.constexpr, T: tl.constexpr, W: tl.constexpr,
                       BITS: tl.constexpr, MODE: tl.constexpr, BW: tl.constexpr):
    batch, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # Padded reads may alias a live row; they must never update its tail/page.
    if tl.load(Writes + batch) < 0:
        return
    row, end = tl.load(Rows + batch), tl.load(Lengths + batch)
    d = tl.arange(0, D)
    k = tl.load(K + batch * K0 + head * K1 + d * K2).to(tl.float32)
    v = tl.load(V + batch * V0 + head * V1 + d * V2).to(tl.float32)
    if end % G == 0:
        # Complete the page from the old raw prefix and this token. Raw data
        # stays untouched, so independent token tiles cannot race each other.
        t = tile * T + tl.arange(0, T)
        raw = ((row * G + t[:, None]) * H + head) * D + d[None, :]
        old_k = tl.load(RK + raw, (t < G - 1)[:, None], 0).to(tl.float32)
        old_v = tl.load(RV + raw, (t < G - 1)[:, None], 0).to(tl.float32)
        pk = tl.where((t == G - 1)[:, None], k[None, :], old_k)
        pv = tl.where((t == G - 1)[:, None], v[None, :], old_v)
        page = tl.load(Slots + row * SLOT_STRIDE + end - G) // G
        _encode_vectors(pk, pv, page, head, t, KD, VD, KS, KM, VS, VM, C,
                        H, D, G, W, BITS, MODE, BW, T)
    elif tile == 0:
        dest = ((row * G + (end - 1) % G) * H + head) * D + d
        tl.store(RK + dest, k)
        tl.store(RV + dest, v)


def quantized_decode_append(key, value, payload, slots, rows, lengths, write_slots):
    """Shared eager/replay writer; negative write slots mask padded requests."""
    g, h, d = payload.page_size, payload.raw_key.shape[-2], payload.raw_key.shape[-1]
    if key.shape != value.shape or tuple(key.shape) != (rows.numel(), h, d) or lengths.numel() != rows.numel():
        raise ValueError("Quantized decode append requires one token and length per cache row.")
    if write_slots.shape != rows.shape:
        raise ValueError("Quantized decode write slots must match request rows.")
    if not rows.numel():
        return
    tile = g if payload.format == "kivi" else min(g, 32)
    _decode_append[(rows.numel(), h, g // tile)](
        key, value, *_args(payload), slots, rows, lengths, write_slots,
        *key.stride(), *value.stride(), slots.stride(0),
        **_constants(payload), T=tile, BW=triton.next_power_of_2(payload.k_cache.shape[-1]),
        num_warps=4, enable_fp_fusion=False,
    )


@triton.jit
def _encode(K, V, Pages, KD, VD, KS, KM, VS, VM, C,
            H: tl.constexpr, D: tl.constexpr, G: tl.constexpr, W: tl.constexpr,
            BITS: tl.constexpr, MODE: tl.constexpr, BW: tl.constexpr, T: tl.constexpr):
    source_page, head = tl.program_id(0) // (G // T), tl.program_id(1)
    page = tl.load(Pages + source_page)
    t = (tl.program_id(0) % (G // T)) * T + tl.arange(0, T)
    d = tl.arange(0, D)
    source = ((source_page * G + t[:, None]) * H + head) * D + d[None, :]
    k, v = tl.load(K + source).to(tl.float32), tl.load(V + source).to(tl.float32)
    _encode_vectors(k, v, page, head, t, KD, VD, KS, KM, VS, VM, C,
                    H, D, G, W, BITS, MODE, BW, T)


@triton.jit
def _encode_vectors(k, v, page, head, t, KD, VD, KS, KM, VS, VM, C,
                    H: tl.constexpr, D: tl.constexpr, G: tl.constexpr, W: tl.constexpr,
                    BITS: tl.constexpr, MODE: tl.constexpr, BW: tl.constexpr, T: tl.constexpr):
    d = tl.arange(0, D)
    if MODE == 0:
        km = tl.min(k, 0)
        ks = tl.maximum(tl.div_rn(tl.max(k, 0) - km, (1 << BITS) - 1), 1.0e-30)
        vg = tl.reshape(v, (G, D // G, G))
        vm = tl.min(vg, 2)
        vs = tl.maximum(tl.div_rn(tl.max(vg, 2) - vm, (1 << BITS) - 1), 1.0e-30)
        tl.store(KS + (page * H + head) * D + d, ks)
        tl.store(KM + (page * H + head) * D + d, km)
        group = tl.arange(0, D // G)
        vm_off = ((page * G + t[:, None]) * H + head) * (D // G) + group[None, :]
        tl.store(VS + vm_off, vs)
        tl.store(VM + vm_off, vm)
        kq = tl.minimum(tl.maximum(tl.floor(tl.div_rn(k - km[None, :], ks[None, :]) + 0.5), 0), (1 << BITS) - 1).to(tl.int32)
        vq = tl.reshape(tl.minimum(tl.maximum(tl.floor(tl.div_rn(vg - vm[:, :, None], vs[:, :, None]) + 0.5), 0), (1 << BITS) - 1), (G, D)).to(tl.int32)
    else:
        if MODE == 1:
            ks = tl.maximum(tl.sqrt(tl.sum(k * k, 1) / D), 1.0e-30)
            vs = tl.maximum(tl.sqrt(tl.sum(v * v, 1) / D), 1.0e-30)
            kn, vn = k / ks[:, None], v / vs[:, None]
            kq, vq = tl.full((T, D), 0, tl.int32), tl.full((T, D), 0, tl.int32)
            for i in tl.static_range((1 << BITS) - 1):
                threshold = (tl.load(C + i) + tl.load(C + i + 1)) * 0.5
                kq += (kn > threshold).to(tl.int32)
                vq += (vn > threshold).to(tl.int32)
        else:
            ks = tl.maximum(tl.max(tl.abs(k), 1) / 448.0, 1.0e-30)
            vs = tl.maximum(tl.max(tl.abs(v), 1) / 448.0, 1.0e-30)
        scale_off = (page * G + t) * H + head
        tl.store(KS + scale_off, ks)
        tl.store(VS + scale_off, vs)
    if MODE == 2:
        dest = ((page * G + t[:, None]) * H + head) * D + d[None, :]
        tl.store(KD + dest, tl.minimum(tl.maximum(k / ks[:, None], -448.0), 448.0))
        tl.store(VD + dest, tl.minimum(tl.maximum(v / vs[:, None], -448.0), 448.0))
    else:
        words = tl.arange(0, BW)
        kp, vp = tl.full((T, BW), 0, tl.int32), tl.full((T, BW), 0, tl.int32)
        for lane in tl.static_range(32 // BITS):
            index = words * (32 // BITS) + lane
            indices = tl.broadcast_to(tl.minimum(index, D - 1)[None, :], (T, BW))
            kc = tl.gather(kq, indices, 1)
            vc = tl.gather(vq, indices, 1)
            kp |= tl.where(index[None, :] < D, kc, 0) << (lane * BITS)
            vp |= tl.where(index[None, :] < D, vc, 0) << (lane * BITS)
        dest = ((page * G + t[:, None]) * H + head) * W + words[None, :]
        tl.store(KD + dest, kp, words[None, :] < W)
        tl.store(VD + dest, vp, words[None, :] < W)


@triton.jit
def _load_vectors(KD, VD, KS, KM, VS, VM, RK, RV, C, Slots,
                  row, positions, length, slot_stride: tl.constexpr,
                  head, H: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
                  W: tl.constexpr, BITS: tl.constexpr, MODE: tl.constexpr):
    d = tl.arange(0, D)
    valid = positions < length
    packed = valid & (positions < (length // G) * G)
    slot = tl.load(Slots + row * slot_stride + positions, valid, 0)
    if MODE == 2:
        off = (slot[:, None] * H + head) * D + d[None, :]
        k = tl.load(KD + off, packed[:, None], 0.0).to(tl.float32)
        v = tl.load(VD + off, packed[:, None], 0.0).to(tl.float32)
    else:
        off = (slot[:, None] * H + head) * W + (d[None, :] // (32 // BITS))
        shift = (d % (32 // BITS)) * BITS
        k = (tl.load(KD + off, packed[:, None], 0) >> shift[None, :]) & ((1 << BITS) - 1)
        v = (tl.load(VD + off, packed[:, None], 0) >> shift[None, :]) & ((1 << BITS) - 1)
        if MODE == 1:
            k, v = tl.load(C + k), tl.load(C + v)
    if MODE == 0:
        ko = ((slot // G)[:, None] * H + head) * D + d[None, :]
        vo = (slot[:, None] * H + head) * (D // G) + d[None, :] // G
        k = k.to(tl.float32) * tl.load(KS + ko, packed[:, None], 0) + tl.load(KM + ko, packed[:, None], 0)
        v = v.to(tl.float32) * tl.load(VS + vo, packed[:, None], 0) + tl.load(VM + vo, packed[:, None], 0)
    else:
        so = slot * H + head
        k = k.to(tl.float32) * tl.load(KS + so, packed, 0)[:, None]
        v = v.to(tl.float32) * tl.load(VS + so, packed, 0)[:, None]
    raw_off = ((row * G + positions[:, None] % G) * H + head) * D + d[None, :]
    tail = valid & ~packed
    raw_k = tl.load(RK + raw_off, tail[:, None], 0).to(tl.float32)
    raw_v = tl.load(RV + raw_off, tail[:, None], 0).to(tl.float32)
    return tl.where(packed[:, None], k, raw_k), tl.where(packed[:, None], v, raw_v)


@triton.jit
def _decode(Q, KD, VD, KS, KM, VS, VM, RK, RV, C, Slots, Rows, Lengths, MO, ML,
            SLOT_STRIDE: tl.constexpr, Q0: tl.constexpr, Q1: tl.constexpr,
            H: tl.constexpr, QH: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
            W: tl.constexpr, BITS: tl.constexpr, MODE: tl.constexpr,
            SPLITS: tl.constexpr, SCALE: tl.constexpr, N: tl.constexpr):
    batch, qhead, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row, length = tl.load(Rows + batch), tl.load(Lengths + batch)
    d = tl.arange(0, D)
    base = (batch * QH + qhead) * SPLITS + split
    # Graphs reserve the full context grid. Inactive splits must not execute
    # page dequantization, and must overwrite stale scratch on length shrink.
    if split * N >= length:
        tl.store(MO + base * D + d, 0.0)
        tl.store(ML + base, -float("inf"))
        return
    q = tl.load(Q + batch * Q0 + qhead * Q1 + d).to(tl.float32)
    # KIVI's per-channel affine metadata makes a full N x D dequantization
    # spill heavily. Bound live vectors without changing the 128-token split
    # workspace or graph grid; online softmax combines the smaller tiles.
    T: tl.constexpr = min(32, 4096 // D) if MODE == 0 else N
    maximum = tl.full((), -float("inf"), tl.float32)
    total = tl.full((), 0.0, tl.float32)
    numerator = tl.full((D,), 0.0, tl.float32)
    for offset in range(0, N, T):
        positions = split * N + offset + tl.arange(0, T)
        k, v = _load_vectors(KD, VD, KS, KM, VS, VM, RK, RV, C, Slots,
                             row, positions, length, SLOT_STRIDE, qhead // (QH // H),
                             H, D, G, W, BITS, MODE)
        score = tl.sum(k * q[None, :], 1) * SCALE
        score = tl.where(positions < length, score, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(score, 0))
        next_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        correction = tl.exp(maximum - next_maximum)
        prob = tl.exp(score - next_maximum)
        numerator = numerator * correction + tl.sum(prob[:, None] * v, 0)
        total = total * correction + tl.sum(prob, 0)
        maximum = next_maximum
    output = numerator / tl.maximum(total, 1.0e-30)
    tl.store(MO + base * D + d, output)
    tl.store(ML + base, tl.where(total > 0, tl.log(total) + maximum, -float("inf")))


@triton.jit
def _materialize(KD, VD, KS, KM, VS, VM, RK, RV, C, Slots, KO, VO,
                 ROW, LENGTH, SLOT_STRIDE: tl.constexpr,
                 H: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
                 W: tl.constexpr, BITS: tl.constexpr, MODE: tl.constexpr,
                 N: tl.constexpr):
    positions = tl.program_id(0) * N + tl.arange(0, N)
    head = tl.program_id(1)
    k, v = _load_vectors(KD, VD, KS, KM, VS, VM, RK, RV, C, Slots,
                         ROW, positions, LENGTH, SLOT_STRIDE, head, H, D, G, W, BITS, MODE)
    off = (positions[:, None] * H + head) * D + tl.arange(0, D)[None, :]
    tl.store(KO + off, k, positions[:, None] < LENGTH)
    tl.store(VO + off, v, positions[:, None] < LENGTH)


def _args(payload):
    return (payload.k_cache, payload.v_cache, payload.key_scale, payload.key_min,
            payload.value_scale, payload.value_min, payload.raw_key, payload.raw_value, payload.codebook)


def _constants(payload):
    return dict(H=payload.raw_key.shape[-2], D=payload.raw_key.shape[-1], G=payload.page_size,
                W=payload.k_cache.shape[-1], BITS=payload.bits,
                MODE={"kivi": 0, "turboquant": 1, "fp8_kv": 2}[payload.format])


def encode_pages(key, value, pages, payload):
    g, h, d = payload.page_size, payload.raw_key.shape[-2], payload.raw_key.shape[-1]
    if key.shape != value.shape or tuple(key.shape) != (pages.numel() * g, h, d):
        raise ValueError("Page codec input must contain exactly the requested complete pages.")
    if pages.numel() == 0:
        return
    if pages.dtype != torch.int32 or key.device != payload.k_cache.device or pages.device != key.device:
        raise ValueError("Page codec input and int32 page IDs must share the cache device.")
    # Only KIVI K statistics couple all tokens in a page. Other codecs tile
    # independent tokens to bound register pressure and compiler work.
    tile = g if payload.format == "kivi" else min(g, 32)
    _encode[(pages.numel() * (g // tile), h)](
        key.contiguous(), value.contiguous(), pages,
        payload.k_cache, payload.v_cache, payload.key_scale, payload.key_min,
        payload.value_scale, payload.value_min, payload.codebook,
        **_constants(payload), BW=triton.next_power_of_2(payload.k_cache.shape[-1]), T=tile, num_warps=4,
        enable_fp_fusion=False,
    )


def materialize_sequence(payload, slots, row, length, key_out, value_out):
    if not 0 <= length <= slots.shape[1] or not 0 <= row < slots.shape[0]:
        raise ValueError("Materialization row/length exceeds the cache map.")
    if length == 0:
        return
    _materialize[(triton.cdiv(length, 32), payload.raw_key.shape[-2])](
        *_args(payload), slots, key_out, value_out,
        ROW=row, LENGTH=length, SLOT_STRIDE=slots.stride(0), **_constants(payload), N=32,
    )


def quantized_decode(q, payload, slots, rows, lengths, mid_o, mid_lse, *, softmax_scale, output, output_lse):
    from sparseengine.kernels.triton.paged_flash_decoding import fixed_grid_flash_decode_stage2

    if q.shape[0] != rows.numel() or rows.shape != lengths.shape:
        raise ValueError("Decode rows and lengths must match the query batch.")
    if q.shape[-1] != payload.raw_key.shape[-1] or q.shape[1] % payload.raw_key.shape[-2]:
        raise ValueError("Decode query head shape is incompatible with quantized KV.")
    if q.stride(-1) != 1 or slots.stride(-1) != 1:
        raise ValueError("Query head dimensions and slot-map token dimensions must be contiguous.")
    batch, heads, dim = q.shape
    splits = mid_o.shape[2]
    _decode[(batch, heads, splits)](
        q, *_args(payload), slots, rows, lengths, mid_o, mid_lse,
        SLOT_STRIDE=slots.stride(0), Q0=q.stride(0), Q1=q.stride(1), QH=heads,
        **_constants(payload), SPLITS=splits, SCALE=softmax_scale, N=128, num_warps=4,
    )
    fixed_grid_flash_decode_stage2(mid_o, mid_lse, lengths, output, output_lse,
                                  target_tokens_per_split=128)
    return output
