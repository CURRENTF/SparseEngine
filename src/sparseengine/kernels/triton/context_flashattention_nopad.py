from functools import lru_cache
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sparseengine.platforms import device_runtime

TESLA = "Tesla" in device_runtime.optional_device_name(0)
_HD256_BLOCK128_REQUIRED_SHARED_MEMORY = 163_840


def select_context_attention_launch_config(
    head_dim: int,
    *,
    max_shared_memory: int,
    is_tesla: bool = False,
) -> tuple[int, int, int, int]:
    """Return a resource-safe static launch specialization.

    The head-dim-256 BLOCK_M/BLOCK_N=128 specialization uses 163,840 bytes of
    shared memory with Triton 3.6. SM120 exposes only 101,376 bytes per block,
    while H100 can retain the larger tile.
    """

    if head_dim not in {16, 32, 64, 128, 256}:
        raise ValueError(f"Unsupported context-attention head_dim={head_dim}.")
    if max_shared_memory <= 0:
        raise ValueError(
            "Context-attention max shared memory must be positive, got "
            f"{max_shared_memory}."
        )
    resource_limited_hd256 = (
        head_dim == 256
        and max_shared_memory < _HD256_BLOCK128_REQUIRED_SHARED_MEMORY
    )
    block_m = 64 if is_tesla or resource_limited_hd256 else 128
    block_n = block_m
    num_warps = 4 if head_dim <= 64 else 8
    return block_m, block_n, num_warps, 1


@lru_cache(maxsize=None)
def _device_max_shared_memory(device_index: int) -> int:
    properties = triton.runtime.driver.active.utils.get_device_properties(
        int(device_index)
    )
    try:
        max_shared_memory = int(properties["max_shared_mem"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "Triton device properties do not expose a valid max_shared_mem: "
            f"{properties!r}."
        ) from error
    return max_shared_memory


@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, Out, B_Start_Loc, B_Seqlen, Req_to_tokens, B_req_idx,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_req_to_tokens_b, stride_req_to_tokens_s,
    kv_group_num, b_prompt_cache_len,
    H: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H
    cur_kv_head = cur_head // kv_group_num

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M + prompt_cache_len, cur_batch_seq_len + prompt_cache_len)

    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * (start_n + offs_n),
            mask=(start_n + offs_n) < block_end_loc, other=0,
        )
        off_k = kv_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=(start_n + offs_n[None, :]) < block_end_loc, other=0.0)
        qk = tl.dot(q, k)
        mask = (offs_m[:, None] + prompt_cache_len) >= (start_n + offs_n[None, :])
        qk = tl.where(mask, qk * sm_scale, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        off_v = kv_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=(start_n + offs_n[:, None]) < block_end_loc, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh + offs_d[None, :] * stride_od
    )
    tl.store(Out + off_o, acc, mask=offs_m[:, None] < cur_batch_seq_len)


@triton.jit
def _fwd_kernel_with_score(
    Q, K, V, sm_scale, Out, B_Start_Loc, B_Seqlen, Req_to_tokens, B_req_idx, Attn_Score,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_req_to_tokens_b, stride_req_to_tokens_s,
    stride_asb, stride_ash, stride_asl,
    kv_group_num, b_prompt_cache_len,
    H: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H
    cur_kv_head = cur_head // kv_group_num

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M + prompt_cache_len, cur_batch_seq_len + prompt_cache_len)

    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * (start_n + offs_n),
            mask=(start_n + offs_n) < block_end_loc, other=0,
        )
        off_k = kv_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=(start_n + offs_n[None, :]) < block_end_loc, other=0.0)
        qk = tl.dot(q, k)
        
        # 收集评分：使用原始点积 (Raw Logits)，且掩码位置设为 0 以便后续计算 Mean
        mask = (offs_m[:, None] + prompt_cache_len) >= (start_n + offs_n[None, :])
        score_to_collect = tl.where(mask, qk, 0.0)
        block_sum = tl.sum(score_to_collect, 0)
        tl.atomic_add(Attn_Score + cur_batch * stride_asb + cur_head * stride_ash + (start_n + offs_n) * stride_asl, 
                      block_sum, mask=(start_n + offs_n) < block_end_loc)

        qk = tl.where(mask, qk * sm_scale, -1.0e8)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        off_v = kv_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=(start_n + offs_n[:, None]) < block_end_loc, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh + offs_d[None, :] * stride_od
    )
    tl.store(Out + off_o, acc, mask=offs_m[:, None] < cur_batch_seq_len)


@triton.jit
def _fwd_kernel_with_score_2d(
    Q, K, V, sm_scale, Out, B_Start_Loc, B_Seqlen, Req_to_tokens, B_req_idx, Attn_Score,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_req_to_tokens_b, stride_req_to_tokens_s,
    stride_asb, stride_ash, stride_asl,
    kv_group_num, b_prompt_cache_len,
    H: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H
    cur_kv_head = cur_head // kv_group_num

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M + prompt_cache_len, cur_batch_seq_len + prompt_cache_len)

    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * (start_n + offs_n),
            mask=(start_n + offs_n) < block_end_loc, other=0,
        )
        off_k = kv_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=(start_n + offs_n[None, :]) < block_end_loc, other=0.0)
        qk = tl.dot(q, k)
        mask = (offs_m[:, None] + prompt_cache_len) >= (start_n + offs_n[None, :])
        
        # Max across the complete query chunk and all heads. Query tiles and
        # heads race through atomic_max into one [batch, context] raw-QK row.
        block_max = tl.max(tl.where(mask, qk, -float("inf")), axis=0)
        tl.atomic_max(Attn_Score + cur_batch * stride_asb + (start_n + offs_n) * stride_asl, 
                      block_max, mask=(start_n + offs_n) < block_end_loc)

        qk = tl.where(mask, qk * sm_scale, -1.0e8)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        off_v = kv_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=(start_n + offs_n[:, None]) < block_end_loc, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh + offs_d[None, :] * stride_od
    )
    tl.store(Out + off_o, acc, mask=offs_m[:, None] < cur_batch_seq_len)


@torch.no_grad()
def context_attention_fwd(
    q, k, v, o, b_req_idx, b_start_loc, b_seq_len, b_prompt_cache_len, max_input_len, req_to_token_indexs,
    attn_score=None
):
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    
    # 补齐断言：安全防护
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128, 256}
    assert q.dtype == k.dtype and k.dtype == v.dtype
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1 and o.stride(-1) == 1

    sm_scale = 1.0 / (Lq ** 0.5) * 1.4426950408889634
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]
    device_index = q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    BLOCK_M, BLOCK_N, num_warps, num_stages = (
        select_context_attention_launch_config(
            Lk,
            max_shared_memory=_device_max_shared_memory(device_index),
            is_tesla=TESLA,
        )
    )
    grid = lambda meta: (triton.cdiv(max_input_len, meta["BLOCK_M"]), batch * head, 1)

    if attn_score is None:
        _fwd_kernel[grid](
            q, k, v, sm_scale, o, b_start_loc, b_seq_len, req_to_token_indexs, b_req_idx,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            req_to_token_indexs.stride(0), req_to_token_indexs.stride(1),
            kv_group_num=kv_group_num, b_prompt_cache_len=b_prompt_cache_len,
            H=head, BLOCK_DMODEL=Lk, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=num_warps, num_stages=num_stages,
        )
    elif attn_score.dim() == 3:
        _fwd_kernel_with_score[grid](
            q, k, v, sm_scale, o, b_start_loc, b_seq_len, req_to_token_indexs, b_req_idx,
            attn_score,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            req_to_token_indexs.stride(0), req_to_token_indexs.stride(1),
            attn_score.stride(0), attn_score.stride(1), attn_score.stride(2),
            kv_group_num=kv_group_num, b_prompt_cache_len=b_prompt_cache_len,
            H=head, BLOCK_DMODEL=Lk, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=num_warps, num_stages=num_stages,
        )
    else: # 2D version
        _fwd_kernel_with_score_2d[grid](
            q, k, v, sm_scale, o, b_start_loc, b_seq_len, req_to_token_indexs, b_req_idx,
            attn_score,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            req_to_token_indexs.stride(0), req_to_token_indexs.stride(1),
            attn_score.stride(0), 0, attn_score.stride(1), # ash=0
            kv_group_num=kv_group_num, b_prompt_cache_len=b_prompt_cache_len,
            H=head, BLOCK_DMODEL=Lk, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=num_warps, num_stages=num_stages,
        )
