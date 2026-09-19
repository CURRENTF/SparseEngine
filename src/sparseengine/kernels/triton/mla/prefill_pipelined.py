"""Sliced-query MLA partials with explicit double-buffered K/V copies.

Uses Triton Gluon to retain the query in MMA registers, overlap the next K/V
loads with current QK/softmax/PV, and avoid cross-warp softmax reductions.
The prepared operator owns compatibility selection and workspace accounting.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout,
    SliceLayout,
    DotOperandLayout,
    NVMMADistributedLayout,
    SwizzledSharedLayout,
)
from triton.experimental.gluon.language.nvidia.ampere import mma_v2, async_copy


from .prefill_plan import select_prefill_splits


# Lengths are launch data, not kernel variants. In particular, each new chain
# suffix changes TQ and CHUNK; specializing them compiles during serving.
@g.jit(do_not_specialize=["TQ", "BLOCKS", "CHUNK"])
def _attention(
    Q,
    K,
    V,
    O,
    L,
    CQ,
    CK,
    q0,
    q1,
    k0,
    k1,
    v0,
    v1,
    TQ,
    H: gl.constexpr,
    D: gl.constexpr,
    SCALE: gl.constexpr,
    CAUSAL: gl.constexpr,
    M: gl.constexpr,
    N: gl.constexpr,
    BLOCKS,
    SPLITS: gl.constexpr,
    CHUNK,
):
    (block, head, batch) = (gl.program_id(0), gl.program_id(1), gl.program_id(2))
    split = batch % SPLITS
    batch = batch // SPLITS
    begin = split * CHUNK
    # Longest causal tiles first; all heads share each query-length wave.
    if CAUSAL:
        flat = gl.program_id(0) + gl.program_id(1) * BLOCKS
        block = BLOCKS - 1 - flat // H
        head = flat % H
    (qs, qe) = (gl.load(CQ + batch), gl.load(CQ + batch + 1))
    (ks, ke) = (gl.load(CK + batch), gl.load(CK + batch + 1))
    mma: gl.constexpr = NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[8, 1], instr_shape=[16, 8]
    )
    mem: gl.constexpr = BlockedLayout([1, 8], [4, 8], [8, 1], [1, 0])
    shared: gl.constexpr = SwizzledSharedLayout(8, 1, 8, [1, 0])
    qm = block * M + gl.arange(0, M, layout=SliceLayout(1, mem))
    dd = gl.arange(0, D, layout=SliceLayout(0, mem))
    # Packed ragged tensors can exceed 2**31 elements even with int32 cu_seqlens.
    q = gl.load(
        Q + (qs.to(gl.int64) + qm[:, None]) * q0 + head * q1 + dd[None, :], qm[:, None] < qe - qs, 0
    )
    q = gl.convert_layout(q, DotOperandLayout(0, mma, 2))
    nn = gl.arange(0, N, layout=SliceLayout(1, mem))
    ptrk = K + (ks.to(gl.int64) + nn[:, None]) * k0 + head * k1 + dd[None, :]
    ptrv = V + (ks.to(gl.int64) + nn[:, None]) * v0 + head * v1 + dd[None, :]
    sk = gl.allocate_shared_memory(K.dtype.element_ty, [2, N, D], shared)
    sv = gl.allocate_shared_memory(V.dtype.element_ty, [2, N, D], shared)
    end = ke - ks
    if SPLITS > 1:
        end = gl.minimum(end, begin + CHUNK)
    if CAUSAL:
        end = gl.minimum(end, (block + 1) * M + ke - ks - (qe - qs))
    end = gl.where(block * M < qe - qs, gl.maximum(end, 0), 0)
    # Two groups keep the next buffer loading while the current one computes.
    for stage in gl.static_range(2):
        async_copy.async_copy_global_to_shared(
            sk.index(stage), ptrk + (begin.to(gl.int64) + stage * N) * k0, begin + stage * N + nn[:, None] < end
        )
        async_copy.async_copy_global_to_shared(
            sv.index(stage), ptrv + (begin.to(gl.int64) + stage * N) * v0, begin + stage * N + nn[:, None] < end
        )
        async_copy.commit_group()
    maximum = gl.full([M], -float("inf"), gl.float32, SliceLayout(1, mma))
    denominator = gl.zeros([M], gl.float32, SliceLayout(1, mma))
    acc = gl.zeros([M, D], gl.float32, mma)
    qi = block * M + gl.arange(0, M, layout=SliceLayout(1, mma))
    nk = gl.arange(0, N, layout=SliceLayout(0, mma))
    for start in range(begin, end, N):
        async_copy.wait_group(1)
        ki = sk.index((start - begin) // N % 2)
        vi = sv.index((start - begin) // N % 2)
        k = ki.permute((1, 0)).load(DotOperandLayout(1, mma, 2))
        # Reuse shared storage only after its operands have reached registers.
        async_copy.async_copy_global_to_shared(
            ki, ptrk + (start.to(gl.int64) + 2 * N) * k0, start + 2 * N + nn[:, None] < end
        )
        z = mma_v2(q, k, gl.zeros([M, N], gl.float32, mma))
        valid = start + nk[None, :] < ke - ks
        if CAUSAL:
            valid = valid & (start + nk[None, :] <= qi[:, None] + ke - ks - (qe - qs))
        z = gl.where(valid, z, -float("inf"))
        updated = gl.maximum(maximum, gl.max(z, 1) * (SCALE * 1.4426950408889634))
        safe = gl.where(updated == -float("inf"), 0.0, updated)
        p = gl.exp2(z * (SCALE * 1.4426950408889634) - safe[:, None])
        alpha = gl.exp2(maximum - safe)
        v = vi.load(DotOperandLayout(1, mma, 2))
        async_copy.async_copy_global_to_shared(
            vi, ptrv + (start.to(gl.int64) + 2 * N) * v0, start + 2 * N + nn[:, None] < end
        )
        async_copy.commit_group()
        acc = mma_v2(
            gl.convert_layout(p.to(q.dtype), DotOperandLayout(0, mma, 2)),
            v,
            acc * alpha[:, None],
        )
        denominator = denominator * alpha + gl.sum(p, 1)
        maximum = updated
    # Drain masked prefetches before releasing the buffers.
    async_copy.wait_group(0)
    out = acc / gl.where(denominator > 0, denominator, 1.0)[:, None]
    do = gl.arange(0, D, layout=SliceLayout(0, mma))
    if SPLITS == 1:
        output_offset = ((qs.to(gl.int64) + qi[:, None]) * H + head) * D + do[None, :]
        lse_offset = head * TQ + qs + qi
    else:
        output_offset = ((split.to(gl.int64) * H + head) * TQ + qs + qi[:, None]) * D + do[None, :]
        lse_offset = (split.to(gl.int64) * H + head) * TQ + qs + qi
    gl.store(O + output_offset, out, qi[:, None] < qe - qs)
    gl.store(
        L + lse_offset,
        (maximum + gl.log2(denominator)) * 0.6931471805599453,
        qi < qe - qs,
    )


@triton.jit(do_not_specialize=["T"])
def _merge_splits(P, PL, O, L, T, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr):
    token, head = tl.program_id(0), tl.program_id(1)
    ss = tl.arange(0, S)
    dd = tl.arange(0, D)
    lse_offsets = (ss.to(tl.int64) * H + head) * T + token
    lse = tl.load(PL + lse_offsets)
    maximum = tl.max(lse, 0)
    safe = tl.where(maximum == -float("inf"), 0.0, maximum)
    weight = tl.exp(lse - safe)
    denominator = tl.sum(weight, 0)
    offsets = lse_offsets[:, None] * D
    vals = tl.load(P + offsets + dd[None, :])
    out = tl.sum(vals * weight[:, None], 0) / tl.where(denominator > 0, denominator, 1.0)
    tl.store(O + (token.to(tl.int64) * H + head) * D + dd, out)
    tl.store(L + head * T + token, maximum + tl.log(denominator))


@lru_cache(None)
def _device_sms(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def _num_splits(q, batch, max_q, max_k, sm_count=None):
    return select_prefill_splits(
        heads=q.shape[1], batch=batch, max_q=max_q, max_k=max_k,
        sm_count=_device_sms(q.device.index) if sm_count is None else sm_count,
    )


def _validate_inputs(q, k, v):
    if (
        any(
            (
                t.ndim != 3
                or t.shape[1:] != q.shape[1:]
                or t.dtype != torch.bfloat16
                or (t.stride(-1) != 1)
                for t in (q, k, v)
            )
        )
        or q.shape[-1] != 256
        or k.shape[0] != v.shape[0]
        or k.device != q.device
        or v.device != q.device
    ):
        raise ValueError(
            "Pipelined MLA requires BF16 [tokens, heads, 256] with contiguous head elements."
        )


def attention_partial(q, k, v, cq, ck, max_q, max_k, *, scale, causal, split_kv=False, sm_count=None):
    """Return BF16 partial output and natural-log LSE.

    The prepared provider enables split-KV and accounts for its FP32 scratch.
    Direct callers may disable splitting for isolated kernel comparisons.
    """
    _validate_inputs(q, k, v)
    o = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    l = torch.empty((q.shape[1], q.shape[0]), device=q.device, dtype=torch.float32)
    if q.shape[0] == 0 or k.shape[0] == 0:
        return o.zero_(), l.fill_(-float("inf"))
    blocks = triton.cdiv(max_q, 128)
    splits = _num_splits(q, cq.numel() - 1, max_q, max_k, sm_count) if split_kv else 1
    chunk = triton.cdiv(max_k, splits * 32) * 32
    if splits > 1:
        partial = torch.empty(
            (splits, q.shape[1], q.shape[0], 256), device=q.device, dtype=torch.float32
        )
        partial_lse = torch.empty(
            (splits, q.shape[1], q.shape[0]), device=q.device, dtype=torch.float32
        )
    else:
        partial, partial_lse = o, l
    _attention[blocks, q.shape[1], (cq.numel() - 1) * splits](
        q,
        k,
        v,
        partial,
        partial_lse,
        cq,
        ck,
        *q.stride()[:2],
        *k.stride()[:2],
        *v.stride()[:2],
        q.shape[0],
        q.shape[1],
        q.shape[2],
        scale,
        causal,
        128,
        32,
        blocks,
        splits,
        chunk,
        num_warps=8,
        num_stages=1,
    )
    if splits > 1:
        _merge_splits[(q.shape[0], q.shape[1])](
            partial, partial_lse, o, l, q.shape[0], q.shape[1], splits, 256,
            num_warps=4,
        )
    return (o, l)
