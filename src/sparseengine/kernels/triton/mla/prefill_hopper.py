"""Hopper MLA partial attention with TMA copies and WGMMA shared operands."""

import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout,
    SliceLayout,
    DotOperandLayout,
    NVMMADistributedLayout,
)
from triton.experimental.gluon.language.nvidia.hopper import warpgroup_mma, tma, mbarrier
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from sparseengine.kernels.triton.mla.prefill_pipelined import (
    _merge_splits,
    _num_splits,
    _validate_inputs,
)


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
        version=[3, 0], warps_per_cta=[8, 1], instr_shape=[16, N, 16]
    )
    mem: gl.constexpr = BlockedLayout([1, 8], [4, 8], [8, 1], [1, 0])
    shared: gl.constexpr = gl.NVMMASharedLayout(128, 16)
    qm = block * M + gl.arange(0, M, layout=SliceLayout(1, mem))
    dd = gl.arange(0, D, layout=SliceLayout(0, mem))
    # Packed ragged tensors can exceed 2**31 elements even with int32 cu_seqlens.
    q = gl.load(
        Q + (qs.to(gl.int64) + qm[:, None]) * q0 + head * q1 + dd[None, :],
        qm[:, None] < qe - qs,
        0,
    )
    q = gl.convert_layout(q, DotOperandLayout(0, mma, 2))
    sk = gl.allocate_shared_memory(K.dtype, [2, N, D], shared)
    sv = gl.allocate_shared_memory(V.dtype, [2, N, D], shared)
    barriers = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for stage in gl.static_range(2):
        mbarrier.init(barriers.index(stage), count=1)
    end = gl.minimum(ke - ks, begin + CHUNK)
    if CAUSAL:
        end = gl.minimum(end, (block + 1) * M + ke - ks - (qe - qs))
    end = gl.where(block * M < qe - qs, gl.maximum(end, 0), 0)
    for stage in gl.static_range(2):
        bar = barriers.index(stage)
        mbarrier.expect(bar, 2 * N * D * 2)
        tma.async_copy_global_to_shared(
            K, [ks + begin + stage * N, head * k1], bar, sk.index(stage)
        )
        tma.async_copy_global_to_shared(
            V, [ks + begin + stage * N, head * v1], bar, sv.index(stage)
        )
    maximum = gl.full([M], -float("inf"), gl.float32, SliceLayout(1, mma))
    denominator = gl.zeros([M], gl.float32, SliceLayout(1, mma))
    acc = gl.zeros([M, D], gl.float32, mma)
    qi = block * M + gl.arange(0, M, layout=SliceLayout(1, mma))
    nk = gl.arange(0, N, layout=SliceLayout(0, mma))
    for start in range(begin, end, N):
        bar = barriers.index((start - begin) // N % 2)
        mbarrier.wait(bar, ((start - begin) // (2 * N)) % 2)
        ki = sk.index((start - begin) // N % 2)
        vi = sv.index((start - begin) // N % 2)
        z = warpgroup_mma(q, ki.permute((1, 0)), gl.zeros([M, N], gl.float32, mma))
        # WGMMA has finished reading K; V remains live until the PV below.
        mbarrier.expect(bar, 2 * N * D * 2)
        tma.async_copy_global_to_shared(K, [ks + start + 2 * N, head * k1], bar, ki)
        valid = start + nk[None, :] < ke - ks
        if CAUSAL:
            valid = valid & (start + nk[None, :] <= qi[:, None] + ke - ks - (qe - qs))
        z = gl.where(valid, z, -float("inf"))
        updated = gl.maximum(maximum, gl.max(z, 1) * (SCALE * 1.4426950408889634))
        safe = gl.where(updated == -float("inf"), 0.0, updated)
        p = gl.exp2(z * (SCALE * 1.4426950408889634) - safe[:, None])
        alpha = gl.exp2(maximum - safe)
        pv_layout: gl.constexpr = gl.NVMMADistributedLayout(
            version=[3, 0], warps_per_cta=[8, 1], instr_shape=[16, 256, 16]
        )
        acc_pv = gl.convert_layout(acc * alpha[:, None], pv_layout)
        acc_pv = warpgroup_mma(
            gl.convert_layout(p.to(q.dtype), DotOperandLayout(0, pv_layout, 2)),
            vi, acc_pv,
        )
        acc = gl.convert_layout(acc_pv, mma)
        tma.async_copy_global_to_shared(V, [ks + start + 2 * N, head * v1], bar, vi)
        denominator = denominator * alpha + gl.sum(p, 1)
        maximum = updated
    # Drain masked prefetches before releasing the buffers.
    iterations = gl.cdiv(gl.maximum(0, end - begin), N)
    mbarrier.wait(barriers.index(0), ((iterations + 1) // 2) % 2)
    mbarrier.wait(barriers.index(1), (iterations // 2) % 2)
    mbarrier.invalidate(barriers.index(0))
    mbarrier.invalidate(barriers.index(1))
    out = acc / gl.where(denominator > 0, denominator, 1.0)[:, None]
    do = gl.arange(0, D, layout=SliceLayout(0, mma))
    if SPLITS == 1:
        output_offset = ((qs.to(gl.int64) + qi[:, None]) * H + head) * D + do[None, :]
        lse_offset = head * TQ + qs + qi
    else:
        output_offset = ((split.to(gl.int64) * H + head) * TQ + qs + qi[:, None]) * D + do[None, :]
        lse_offset = (split.to(gl.int64) * H + head) * TQ + qs + qi
    gl.store(
        O + output_offset,
        out,
        qi[:, None] < qe - qs,
    )
    gl.store(
        L + lse_offset,
        (maximum + gl.log2(denominator)) * 0.6931471805599453,
        qi < qe - qs,
    )


def attention_partial(q, k, v, cq, ck, max_q, max_k, *, scale, causal, split_kv=False, sm_count=None):
    """Hopper partial attention with optional stable split-KV reduction."""
    _validate_inputs(q, k, v)
    if torch.cuda.get_device_capability(q.device) != (9, 0):
        raise ValueError("TMA/WGMMA MLA requires a Hopper GPU.")
    if any(t.data_ptr() % 16 or t.stride(0) % 8 or t.stride(1) % 8 for t in (k, v)):
        raise ValueError("TMA MLA requires 16-byte aligned K/V pointers and row/head strides.")
    if q.shape[0] == 0 or k.shape[0] == 0:
        return torch.zeros_like(q), torch.full(
            (q.shape[1], q.shape[0]), -float("inf"), device=q.device, dtype=torch.float32
        )
    blocks = triton.cdiv(max_q, 128)
    splits = _num_splits(q, cq.numel() - 1, max_q, max_k, sm_count) if split_kv else 1
    chunk = triton.cdiv(max_k, splits * 64) * 64
    o = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    l = torch.empty((q.shape[1], q.shape[0]), device=q.device, dtype=torch.float32)
    if splits > 1:
        p = torch.empty(
            (splits, q.shape[1], q.shape[0], 256), device=q.device, dtype=torch.float32
        )
        pl = torch.empty(
            (splits, q.shape[1], q.shape[0]), device=q.device, dtype=torch.float32
        )
    else:
        p, pl = o, l
    layout = gl.NVMMASharedLayout(128, 16)
    kd = TensorDescriptor(
        k, [k.shape[0], (k.shape[1] - 1) * k.stride(1) + 256],
        [k.stride(0), 1], [64, 256], layout,
    )
    vd = TensorDescriptor(
        v, [v.shape[0], (v.shape[1] - 1) * v.stride(1) + 256],
        [v.stride(0), 1], [64, 256], layout,
    )
    _attention[(blocks, q.shape[1], (cq.numel() - 1) * splits)](
        q, kd, vd, p, pl, cq, ck, *q.stride()[:2], *k.stride()[:2], *v.stride()[:2],
        q.shape[0], q.shape[1], 256, scale, causal, 128, 64, blocks, splits, chunk,
        num_warps=8, num_stages=1,
    )
    if splits > 1:
        _merge_splits[(q.shape[0], q.shape[1])](
            p, pl, o, l, q.shape[0], q.shape[1], splits, 256, num_warps=4
        )
    return o, l
