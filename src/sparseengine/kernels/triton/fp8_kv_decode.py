# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Grouped FP8-page decode with per-token scales and a high-precision tail.

Adapted from SGLang's _fwd_grouped_kernel_stage1 at 7e5d3780ef06bcd56756e31df27a3b4bb125568b,
python/sglang/kernels/ops/attention/decode_attention.py (Apache-2.0).
Local changes: 2D slot maps, dynamic per-token/head K/V scales, raw incomplete
pages, activation-precision queries, and neutral writes for inactive splits.

Upstream also credits LightLLM's grouped decode implementation:
https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
"""

import triton
import triton.language as tl


@triton.jit
def _fp8_grouped_stage1(
    Q, KD, VD, KS, VS, RK, RV, EncodedPrefix, Slots, Rows, Lengths, MO, ML,
    Q0: tl.constexpr, Q1: tl.constexpr, SLOT_STRIDE: tl.constexpr,
    H: tl.constexpr, QH: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
    SPLITS: tl.constexpr, SCALE: tl.constexpr,
    BH: tl.constexpr, N: tl.constexpr,
):
    batch, group, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch = batch.to(tl.int64)
    ratio: tl.constexpr = QH // H
    tiles: tl.constexpr = triton.cdiv(ratio, BH)
    head = group // tiles
    qheads = head * ratio + (group % tiles) * BH + tl.arange(0, BH)
    mask_h = qheads < (head + 1) * ratio
    d = tl.arange(0, D)
    row, length = tl.load(Rows + batch), tl.load(Lengths + batch)
    packed_end = (length // G) * G
    if EncodedPrefix is not None:
        packed_end = tl.maximum(packed_end, tl.load(EncodedPrefix + row))
    # Bound split workspace by the prepared grid; long contexts loop over
    # token tiles rather than allocating one partial output per 128 tokens.
    splits = tl.maximum(1, tl.minimum(tl.cdiv(length, 128), SPLITS))
    span = tl.cdiv(tl.cdiv(length, splits), N) * N
    start, end = split * span, tl.minimum((split + 1) * span, length)
    base = (batch * QH + qheads) * SPLITS + split
    if start >= end:
        tl.store(MO + base[:, None] * D + d[None, :], 0.0, mask_h[:, None])
        tl.store(ML + base, -float("inf"), mask_h)
        return

    query = tl.load(Q + batch * Q0 + qheads[:, None] * Q1 + d[None, :],
                    mask_h[:, None], 0.0)
    maximum = tl.full((BH,), -float("inf"), tl.float32)
    total = tl.zeros((BH,), tl.float32)
    numerator = tl.zeros((BH, D), tl.float32)
    for offset in tl.range(start, end, N):
        positions = offset + tl.arange(0, N)
        valid = positions < end
        packed = valid & (positions < packed_end)
        tail = valid & ~packed
        slot = tl.load(Slots + row * SLOT_STRIDE + positions, valid, 0)
        packed_k = tl.load(KD + (slot[None, :] * H + head) * D + d[:, None],
                           packed[None, :], 0.0).to(query.dtype)
        raw_k = tl.load(RK + ((row * G + positions[None, :] % G) * H + head) * D + d[:, None],
                        tail[None, :], 0.0)
        key = tl.where(packed[None, :], packed_k, raw_k)
        kscale = tl.load(KS + slot * H + head, packed, 1.0)
        # E4M3 values are exact in the activation dtype. Apply K scales to
        # FP32 dot outputs instead of rounding dequantized K or quantizing Q.
        logits = tl.dot(query, key) * kscale[None, :] * SCALE
        logits = tl.where(valid[None, :], logits, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(logits, 1))
        correction = tl.exp(maximum - next_maximum)
        probability = tl.exp(logits - next_maximum[:, None])

        packed_v = tl.load(VD + (slot[:, None] * H + head) * D + d[None, :],
                           packed[:, None], 0.0).to(tl.float32)
        vscale = tl.load(VS + slot * H + head, packed, 1.0)
        raw_v = tl.load(RV + ((row * G + positions[:, None] % G) * H + head) * D + d[None, :],
                        tail[:, None], 0.0).to(tl.float32)
        value = tl.where(packed[:, None], packed_v * vscale[:, None], raw_v).to(query.dtype)
        numerator = numerator * correction[:, None] + tl.dot(probability.to(query.dtype), value)
        total = total * correction + tl.sum(probability, 1)
        maximum = next_maximum

    tl.store(MO + base[:, None] * D + d[None, :], numerator / total[:, None], mask_h[:, None])
    tl.store(ML + base, maximum + tl.log(total), mask_h)


def fp8_grouped_decode(q, payload, slots, rows, lengths, mid_o, mid_lse, *,
                       softmax_scale, output, output_lse):
    from sparseengine.kernels.triton.paged_flash_decoding import fixed_grid_flash_decode_stage2

    if payload.format != "fp8_kv":
        raise ValueError("Grouped FP8 decode requires FP8 pages.")
    if q.shape[0] != rows.numel() or rows.shape != lengths.shape:
        raise ValueError("Decode rows and lengths must match the query batch.")
    h, d = payload.raw_key.shape[-2:]
    if q.shape[-1] != d or q.shape[1] % h:
        raise ValueError("Decode query head shape is incompatible with FP8 KV.")
    if q.stride(-1) != 1 or slots.stride(-1) != 1:
        raise ValueError("Query head dimensions and slot-map token dimensions must be contiguous.")
    batch, heads, _ = q.shape
    splits = mid_o.shape[2]
    # One pipeline stage keeps D256 within SM120's shared-memory limit.
    _fp8_grouped_stage1[(batch, h * triton.cdiv(heads // h, 16), splits)](
        q, payload.k_cache, payload.v_cache, payload.key_scale, payload.value_scale,
        payload.raw_key, payload.raw_value, payload.encoded_prefix_lens, slots, rows, lengths, mid_o, mid_lse,
        Q0=q.stride(0), Q1=q.stride(1), SLOT_STRIDE=slots.stride(0),
        H=h, QH=heads, D=d, G=payload.page_size, SPLITS=splits, SCALE=softmax_scale,
        BH=16, N=64, num_warps=4, num_stages=1,
    )
    fixed_grid_flash_decode_stage2(mid_o, mid_lse, lengths, output, output_lse,
                                  target_tokens_per_split=128)
    return output
