# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang; see NOTICE for pinned source and semantic changes.
import triton
import triton.language as tl

INDEX_K_PAYLOAD_BYTES = tl.constexpr(64)
INDEX_K_SCALE_BYTES = tl.constexpr(4)

@triton.jit
def _ceil_ue8m0_exp(x):
    bits = x.to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    exp += mantissa != 0
    return tl.minimum(tl.maximum(exp, 1), 254)


@triton.jit
def _fp4_e2m1_code(x):
    ax = tl.minimum(tl.abs(x), 6.0)
    idx = (ax > 0.25).to(tl.uint8)
    idx += (ax > 0.75).to(tl.uint8)
    idx += (ax > 1.25).to(tl.uint8)
    idx += (ax > 1.75).to(tl.uint8)
    idx += (ax > 2.5).to(tl.uint8)
    idx += (ax > 3.5).to(tl.uint8)
    idx += (ax > 5.0).to(tl.uint8)
    sign = ((x < 0) & (idx != 0)).to(tl.uint8)
    return idx | (sign << 3)


@triton.jit
def _fp4_e2m1_code_rne(x):
    """Round-to-nearest-even e2m1 code, matching the reference rounding."""
    ax = tl.minimum(tl.abs(x), 6.0)
    idx = (ax >= 0.25).to(tl.uint8)
    idx += (ax >= 0.75).to(tl.uint8)
    idx += (ax >= 1.25).to(tl.uint8)
    idx += (ax >= 1.75).to(tl.uint8)
    idx += (ax >= 2.5).to(tl.uint8)
    idx += (ax >= 3.5).to(tl.uint8)
    idx += (ax >= 5.0).to(tl.uint8)
    # Round-half-to-even: an odd index at an exact boundary drops to the even one.
    is_boundary = (
        (ax == 0.25)
        | (ax == 0.75)
        | (ax == 1.25)
        | (ax == 1.75)
        | (ax == 2.5)
        | (ax == 3.5)
        | (ax == 5.0)
    )
    idx = tl.where(is_boundary & ((idx & 1) == 1), idx - 1, idx)
    sign = ((x < 0) & (idx != 0)).to(tl.uint8)
    return idx | (sign << 3)


@triton.jit
def _quantize_fp4_indexer_rows(
    x,
    x_fp4,
    x_sf,
    M,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_N: tl.constexpr,
    RNE: tl.constexpr,
):
    tl.static_assert(BLOCK_N == 128 and GROUP_N == 32)
    # Each reduction covers one scale group. Keep its values for packing,
    # avoiding four masked full-row reductions and a second input load.
    group = tl.program_id(0) * BLOCK_M * 4 + tl.arange(0, BLOCK_M * 4)
    offs = tl.arange(0, GROUP_N)
    values = tl.load(
        x + group[:, None].to(tl.int64) * GROUP_N + offs[None, :],
        group[:, None] < M * 4,
        0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(values), axis=1)
    exp = _ceil_ue8m0_exp(tl.maximum(amax, 6 * 2.0**-126) * (1.0 / 6.0))
    scale = (exp << 23).to(tl.float32, bitcast=True)
    v0, v1 = tl.split(
        tl.reshape(values / scale[:, None], (BLOCK_M * 4, GROUP_N // 2, 2))
    )
    if RNE:
        code0 = _fp4_e2m1_code_rne(v0)
        code1 = _fp4_e2m1_code_rne(v1)
    else:
        code0 = _fp4_e2m1_code(v0)
        code1 = _fp4_e2m1_code(v1)
    packed = (code0 & 0x0F) | ((code1 & 0x0F) << 4)
    tl.store(
        x_fp4
        + group[:, None].to(tl.int64) * (GROUP_N // 2)
        + tl.arange(0, GROUP_N // 2)[None, :],
        packed,
        group[:, None] < M * 4,
    )
    # The four exponents occupy disjoint bytes, so integer sum packs them.
    shifts = tl.arange(0, 4) * 8
    packed_sf = tl.sum(
        tl.reshape(exp.to(tl.uint32), (BLOCK_M, 4)) << shifts[None, :], axis=1
    )
    token_id = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(x_sf + token_id, packed_sf.to(tl.int32), token_id < M)


@triton.jit
def _store_fp4_index_k_cache_kernel(
    k_fp4,
    k_sf,
    cache,
    loc,
    page_size: tl.constexpr,
    cache_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    cache_loc = tl.load(loc + token_id)
    page = cache_loc // page_size
    page_offset = cache_loc - page * page_size

    k = tl.load(k_fp4 + token_id * BLOCK + offsets)
    tl.store(cache + page * cache_stride + page_offset * BLOCK + offsets, k)

    sf = tl.load(k_sf + token_id)
    sf_offsets = tl.arange(0, 4)
    sf_bytes = (sf >> (sf_offsets * 8)) & 0xFF
    tl.store(
        cache + page * cache_stride + page_size * BLOCK + page_offset * 4 + sf_offsets,
        sf_bytes,
    )


@triton.jit
def _e2m1_decode(code):
    # code: uint 0..15 -> e2m1 value. exp = bits 2..1, mantissa = bit 0, sign = bit 3.
    e = (code >> 1) & 3
    m = (code & 1).to(tl.float32)
    sub = m * 0.5
    nor = (1.0 + m * 0.5) * tl.exp2((e - 1).to(tl.float32))
    v = tl.where(e == 0, sub, nor)
    return tl.where((code >> 3) == 1, -v, v)


@triton.jit
def _fp4_index_logits_kernel(
    q_ptr,  # [B, H, D] bf16, fq4 queries (already rope'd)
    w_ptr,  # [B, H] bf16 head weights (softmax scale folded in)
    slots_ptr,  # [B, L] int64 pool slots per (request, compressed position)
    lens_ptr,  # [B] int64 visible compressed positions per request
    table_ptr,  # [num_pages, page_size * 64 + page_size * 4] uint8
    out_ptr,  # [B, L] fp32 logits, -inf beyond lens
    L,
    page_size,
    row_stride,
    stride_qb,
    stride_qh,
    stride_wb,
    H: tl.constexpr,
    HALF_D: tl.constexpr,  # D // 2 == 64 nibble-pairs per row
    BLOCK_L: tl.constexpr,
):
    # The caller returns for L == 0 and makes q contiguous. Exclude singleton
    # heads, whose stride is not constrained by PyTorch contiguity.
    tl.assume(L > 0)
    if H > 1:
        tl.assume(stride_qh == HALF_D * 2)
    b = tl.program_id(0)
    lb = tl.program_id(1)
    offs_l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_h = tl.arange(0, H)
    offs_i = tl.arange(
        0, HALF_D
    )  # byte index i holds elements 2i (low nibble), 2i+1 (high nibble)

    n_vis = tl.load(lens_ptr + b)
    # Graph replay keeps the capacity-sized grid even for short live contexts.
    # Skip whole invisible tiles using the current device-side length; masking
    # only the K loads would still run dequantization, dot products and reduction.
    if lb * BLOCK_L >= n_vis:
        tl.store(out_ptr + b * L + offs_l, float("-inf"), mask=offs_l < L)
    else:
        valid = offs_l < tl.minimum(n_vis, L)
        slot = tl.load(slots_ptr + b * L + offs_l, mask=offs_l < L, other=0).to(
            tl.int64
        )
        page = slot // page_size
        off = slot % page_size
        row_base = page * row_stride

        # K payload: [BLOCK_L, HALF_D] uint8
        pay = tl.load(
            table_ptr
            + row_base[:, None]
            + off[:, None] * INDEX_K_PAYLOAD_BYTES
            + offs_i[None, :],
            mask=valid[:, None],
            other=0,
        )
        low = _e2m1_decode(pay & 0x0F)
        high = _e2m1_decode((pay >> 4) & 0x0F)
        # e8m0 block scales: element j uses block j // 32 -> byte i uses block i // 16.
        sc_idx = offs_i // 16
        exps = tl.load(
            table_ptr
            + row_base[:, None]
            + page_size * INDEX_K_PAYLOAD_BYTES
            + off[:, None] * INDEX_K_SCALE_BYTES
            + sc_idx[None, :],
            mask=valid[:, None],
            other=127,
        )
        scale = tl.exp2(exps.to(tl.float32) - 127.0)
        k_low = (low * scale).to(tl.bfloat16)  # [BLOCK_L, HALF_D] elements 2i
        k_high = (high * scale).to(tl.bfloat16)  # elements 2i+1

        # queries: even / odd elements, [H, HALF_D] bf16
        q_even = tl.load(
            q_ptr + b * stride_qb + offs_h[:, None] * stride_qh + 2 * offs_i[None, :]
        )
        q_odd = tl.load(
            q_ptr
            + b * stride_qb
            + offs_h[:, None] * stride_qh
            + 2 * offs_i[None, :]
            + 1
        )

        acc = tl.dot(q_even, tl.trans(k_low))  # [H, BLOCK_L] fp32
        acc += tl.dot(q_odd, tl.trans(k_high))
        # reference rounding points: bf16 dot -> relu -> * bf16 weight -> bf16 -> sum -> bf16
        s = acc.to(tl.bfloat16).to(tl.float32)
        s = tl.maximum(s, 0.0)
        w = tl.load(w_ptr + b * stride_wb + offs_h).to(tl.float32)
        s = (s * w[:, None]).to(tl.bfloat16).to(tl.float32)
        logit = tl.sum(s, axis=0).to(tl.bfloat16).to(tl.float32)
        logit = tl.where(valid, logit, float("-inf"))
        tl.store(out_ptr + b * L + offs_l, logit, mask=offs_l < L)
