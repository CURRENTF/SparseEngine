"""Triton kernels for DeltaKV and full-layer KIVI."""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _full_layer_kivi_flash_decode_stage1_kernel(
    Q,
    Raw_K,
    Raw_V,
    Raw_Slots_Map,
    Kivi_Block_Slots_Map,
    Kivi_Block_Start_Pos,
    Key_Packed,
    Key_Scales,
    Key_Mins,
    Value_Packed,
    Value_Scales,
    Value_Mins,
    Req_Indices,
    B_Seqlen,
    Mid_O,
    Mid_O_LogExpSum,
    Attn_Score,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_raw_ks,
    stride_raw_kh,
    stride_raw_kd,
    stride_raw_vs,
    stride_raw_vh,
    stride_raw_vd,
    stride_slots_r,
    stride_slots_p,
    stride_kivi_map_r,
    stride_kivi_map_p,
    stride_kpb,
    stride_kph,
    stride_kpd,
    stride_kpp,
    stride_ksb,
    stride_ksh,
    stride_ksd,
    stride_vpb,
    stride_vph,
    stride_vpt,
    stride_vpp,
    stride_vsb,
    stride_vsh,
    stride_vst,
    stride_vsg,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    stride_asb,
    stride_ash,
    stride_asl,
    sm_scale: tl.constexpr,
    gqa_group_size: tl.constexpr,
    Q_HEAD_NUM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    FEAT_PER_INT: tl.constexpr,
    QUANT_MASK: tl.constexpr,
    STORE_SCORE: tl.constexpr,
    FIXED_GRID: tl.constexpr,
    MAX_EFFECTIVE_SPLITS: tl.constexpr,
    TARGET_TOKENS_PER_SPLIT: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)

    cur_q_head_offs = tl.arange(0, Q_HEAD_NUM)
    cur_q_head_range = cur_kv_head * gqa_group_size + cur_q_head_offs
    offs_d = tl.arange(0, BLOCK_DMODEL)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_row = tl.load(Req_Indices + cur_batch).to(tl.int32)
    if FIXED_GRID:
        requested_splits = tl.cdiv(cur_batch_seq_len, TARGET_TOKENS_PER_SPLIT)
        num_splits = tl.maximum(
            1,
            tl.minimum(requested_splits, MAX_EFFECTIVE_SPLITS),
        )
        split_tokens = tl.where(
            requested_splits <= MAX_EFFECTIVE_SPLITS,
            TARGET_TOKENS_PER_SPLIT,
            tl.cdiv(cur_batch_seq_len, num_splits),
        )
        cur_batch_start_index = seq_start_block * split_tokens
        cur_batch_end_index = tl.minimum(
            cur_batch_seq_len,
            cur_batch_start_index + split_tokens,
        )
    else:
        cur_batch_start_index = seq_start_block * BLOCK_SEQ
        cur_batch_end_index = tl.minimum(
            cur_batch_seq_len,
            cur_batch_start_index + BLOCK_SEQ,
        )

    off_q = cur_batch * stride_qbs + cur_q_head_range[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(
        Q + off_q,
        mask=cur_q_head_range[:, None] < (cur_kv_head + 1) * gqa_group_size,
        other=0.0,
    )

    block_n_size = (
        tl.where(
            cur_batch_end_index - cur_batch_start_index <= 0,
            0,
            cur_batch_end_index - cur_batch_start_index + BLOCK_N - 1,
        )
        // BLOCK_N
    )
    offs_n = cur_batch_start_index + tl.arange(0, BLOCK_N)

    sum_exp = tl.zeros([Q_HEAD_NUM], dtype=tl.float32)
    max_logic = tl.zeros([Q_HEAD_NUM], dtype=tl.float32) - float("inf")
    acc = tl.zeros([Q_HEAD_NUM, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, block_n_size, 1):
        offs_n_new = start_n * BLOCK_N + offs_n
        valid_n = offs_n_new < cur_batch_end_index
        raw_slots = tl.load(
            Raw_Slots_Map + cur_row * stride_slots_r + offs_n_new * stride_slots_p,
            mask=valid_n,
            other=-1,
        ).to(tl.int32)
        raw_mask = raw_slots >= 0
        safe_raw_slots = tl.maximum(raw_slots, 0)

        raw_k = tl.load(
            Raw_K
            + safe_raw_slots[None, :] * stride_raw_ks
            + cur_kv_head * stride_raw_kh
            + offs_d[:, None] * stride_raw_kd,
            mask=valid_n[None, :] & raw_mask[None, :],
            other=0.0,
        )
        raw_v = tl.load(
            Raw_V
            + safe_raw_slots[:, None] * stride_raw_vs
            + cur_kv_head * stride_raw_vh
            + offs_d[None, :] * stride_raw_vd,
            mask=valid_n[:, None] & raw_mask[:, None],
            other=0.0,
        )

        block_slots = tl.load(
            Kivi_Block_Slots_Map + cur_row * stride_kivi_map_r + offs_n_new * stride_kivi_map_p,
            mask=valid_n & (~raw_mask),
            other=-1,
        ).to(tl.int32)
        kivi_mask = valid_n & (~raw_mask) & (block_slots >= 0)
        safe_block_slots = tl.maximum(block_slots, 0)
        block_starts = tl.load(Kivi_Block_Start_Pos + safe_block_slots, mask=kivi_mask, other=0).to(tl.int32)
        local_t = offs_n_new.to(tl.int32) - block_starts
        kivi_mask = kivi_mask & (local_t >= 0) & (local_t < GROUP_SIZE)
        safe_local_t = tl.maximum(local_t, 0)
        slot_valid = valid_n & (raw_mask | kivi_mask)

        key_pack_idx = safe_local_t // FEAT_PER_INT
        key_shift = (safe_local_t % FEAT_PER_INT) * 4
        key_code = tl.load(
            Key_Packed
            + safe_block_slots[None, :] * stride_kpb
                + cur_kv_head * stride_kph
                + offs_d[:, None] * stride_kpd
                + key_pack_idx[None, :] * stride_kpp,
            mask=kivi_mask[None, :],
            other=0,
        )
        key_q = ((key_code >> key_shift[None, :]) & QUANT_MASK).to(tl.float32)
        key_scale = tl.load(
            Key_Scales
            + safe_block_slots[None, :] * stride_ksb
            + cur_kv_head * stride_ksh
            + offs_d[:, None] * stride_ksd,
            mask=kivi_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        key_min = tl.load(
            Key_Mins
            + safe_block_slots[None, :] * stride_ksb
            + cur_kv_head * stride_ksh
            + offs_d[:, None] * stride_ksd,
            mask=kivi_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        kivi_k = key_q * key_scale + key_min
        k = tl.where(raw_mask[None, :], raw_k, kivi_k).to(q.dtype)

        value_pack_idx = offs_d // FEAT_PER_INT
        value_shift = (offs_d % FEAT_PER_INT) * 4
        value_group = offs_d // GROUP_SIZE
        value_code = tl.load(
            Value_Packed
            + safe_block_slots[:, None] * stride_vpb
                + cur_kv_head * stride_vph
                + safe_local_t[:, None] * stride_vpt
                + value_pack_idx[None, :] * stride_vpp,
            mask=kivi_mask[:, None],
            other=0,
        )
        value_q = ((value_code >> value_shift[None, :]) & QUANT_MASK).to(tl.float32)
        value_scale = tl.load(
            Value_Scales
            + safe_block_slots[:, None] * stride_vsb
                + cur_kv_head * stride_vsh
                + safe_local_t[:, None] * stride_vst
                + value_group[None, :] * stride_vsg,
            mask=kivi_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        value_min = tl.load(
            Value_Mins
            + safe_block_slots[:, None] * stride_vsb
                + cur_kv_head * stride_vsh
                + safe_local_t[:, None] * stride_vst
                + value_group[None, :] * stride_vsg,
            mask=kivi_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        kivi_v = value_q * value_scale + value_min
        v = tl.where(raw_mask[:, None], raw_v, kivi_v).to(q.dtype)

        att_value = tl.dot(q, k)
        att_value = tl.where(slot_valid[None, :], att_value, float("-inf"))
        if STORE_SCORE:
            off_as = cur_batch * stride_asb + cur_q_head_range[:, None] * stride_ash + offs_n_new[None, :] * stride_asl
            tl.store(
                Attn_Score + off_as,
                att_value,
                mask=(
                    (cur_q_head_range[:, None] < (cur_kv_head + 1) * gqa_group_size)
                    & slot_valid[None, :]
                ),
            )

        att_value *= sm_scale

        block_has_valid = tl.max(slot_valid.to(tl.int32), axis=0) > 0
        cur_max_logic = tl.max(att_value, axis=1)
        candidate_max_logic = tl.maximum(cur_max_logic, max_logic)
        new_max_logic = tl.where(block_has_valid, candidate_max_logic, max_logic)
        exp_logic = tl.where(slot_valid[None, :], tl.exp(att_value - new_max_logic[:, None]), 0.0)
        logic_scale = tl.where(block_has_valid, tl.exp(max_logic - new_max_logic), 1.0)
        acc *= logic_scale[:, None]
        acc += tl.dot(exp_logic.to(v.dtype), v)

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=1)
        max_logic = new_max_logic

    off_mid_o = (
        cur_batch * stride_mid_ob
        + cur_q_head_range[:, None] * stride_mid_oh
        + seq_start_block * stride_mid_os
        + offs_d[None, :] * stride_mid_od
    )
    off_mid_o_logexpsum = cur_batch * stride_mid_o_eb + cur_q_head_range * stride_mid_o_eh + seq_start_block * stride_mid_o_es
    safe_sum_exp = tl.where(block_n_size == 0, 1.0, sum_exp)
    neutral_o = tl.zeros([Q_HEAD_NUM, BLOCK_DMODEL], dtype=tl.float32)
    tl.store(
        Mid_O + off_mid_o,
        tl.where(block_n_size == 0, neutral_o, acc / safe_sum_exp[:, None]),
        mask=cur_q_head_range[:, None] < (cur_kv_head + 1) * gqa_group_size,
    )
    tl.store(
        Mid_O_LogExpSum + off_mid_o_logexpsum,
        tl.where(block_n_size == 0, -float("inf"), max_logic + tl.log(safe_sum_exp)),
        mask=cur_q_head_range < (cur_kv_head + 1) * gqa_group_size,
    )


@torch.no_grad()
def _validate_full_layer_kivi_decode_maps(
    *,
    raw_slots_map: torch.Tensor,
    kivi_block_slots_map: torch.Tensor,
    req_indices: torch.Tensor,
    context_lens: torch.Tensor,
    max_len_in_batch: int,
) -> None:
    if int(req_indices.numel()) == 0 or int(max_len_in_batch) <= 0:
        return
    if tuple(raw_slots_map.shape) != tuple(kivi_block_slots_map.shape):
        raise ValueError(
            "Full-layer KIVI raw and block slot maps must have identical shapes, "
            f"got raw={tuple(raw_slots_map.shape)} block={tuple(kivi_block_slots_map.shape)}."
        )
    if int(max_len_in_batch) > int(raw_slots_map.shape[1]):
        raise ValueError(
            "Full-layer KIVI max_len_in_batch exceeds map width: "
            f"max_len={int(max_len_in_batch)} width={int(raw_slots_map.shape[1])}."
        )
    rows = req_indices.to(device=raw_slots_map.device, dtype=torch.long)
    if bool(((rows < 0) | (rows >= int(raw_slots_map.shape[0]))).any().item()):
        bad = rows[((rows < 0) | (rows >= int(raw_slots_map.shape[0])))][:8].detach().cpu().tolist()
        raise RuntimeError(
            "Full-layer KIVI decode has out-of-range req_indices: "
            f"rows={bad} num_rows={int(raw_slots_map.shape[0])}."
        )
    positions = torch.arange(int(max_len_in_batch), device=raw_slots_map.device, dtype=torch.long)
    context_lens_dev = context_lens.to(device=raw_slots_map.device, dtype=torch.long)
    valid = positions.unsqueeze(0) < context_lens_dev.unsqueeze(1)
    raw_slots = raw_slots_map[rows.unsqueeze(1), positions.unsqueeze(0)]
    block_slots = kivi_block_slots_map[rows.unsqueeze(1), positions.unsqueeze(0)]
    missing = valid & (raw_slots < 0) & (block_slots < 0)
    if bool(missing.any().item()):
        bad = missing.nonzero(as_tuple=False)[:8].detach().cpu().tolist()
        raise RuntimeError(
            "Full-layer KIVI decode map has valid tokens with neither raw nor packed block slots: "
            f"batch_pos={bad}."
        )


def full_layer_kivi_flash_decode_stage1(
    *,
    q: torch.Tensor,
    raw_k: torch.Tensor,
    raw_v: torch.Tensor,
    raw_slots_map: torch.Tensor,
    kivi_block_slots_map: torch.Tensor,
    kivi_block_start_pos: torch.Tensor,
    key_packed: torch.Tensor,
    key_scales: torch.Tensor,
    key_mins: torch.Tensor,
    value_packed: torch.Tensor,
    value_scales: torch.Tensor,
    value_mins: torch.Tensor,
    req_indices: torch.Tensor,
    context_lens: torch.Tensor,
    max_len_in_batch: int,
    mid_out: torch.Tensor,
    mid_out_logsumexp: torch.Tensor,
    group_size: int,
    block_seq: int,
    block_n: int = 16,
    num_warps: int = 2,
    num_stages: int = 3,
    attn_score: torch.Tensor | None = None,
    max_kv_splits: int | None = None,
    target_tokens_per_split: int | None = None,
):
    assert q.is_cuda and raw_k.is_cuda and raw_v.is_cuda
    assert raw_slots_map.is_cuda and kivi_block_slots_map.is_cuda and kivi_block_start_pos.is_cuda
    assert key_packed.is_cuda and key_scales.is_cuda and key_mins.is_cuda
    assert value_packed.is_cuda and value_scales.is_cuda and value_mins.is_cuda
    assert req_indices.is_cuda and context_lens.is_cuda
    assert mid_out.is_cuda and mid_out_logsumexp.is_cuda
    if attn_score is not None:
        assert attn_score.is_cuda
        if attn_score.dim() != 3:
            raise ValueError("Full-layer KIVI fused decode currently supports rank-3 attention scores only.")
    if q.dim() != 3 or raw_k.dim() != 3 or raw_v.dim() != 3:
        raise ValueError(f"Expected q/raw_k/raw_v rank-3 tensors, got {q.dim()}, {raw_k.dim()}, {raw_v.dim()}.")
    if raw_slots_map.dim() != 2 or kivi_block_slots_map.dim() != 2:
        raise ValueError("Full-layer KIVI decode maps must be rank-2 tensors.")
    if tuple(raw_slots_map.shape) != tuple(kivi_block_slots_map.shape):
        raise ValueError(
            "Full-layer KIVI raw and block slot maps must have identical shapes, "
            f"got raw={tuple(raw_slots_map.shape)} block={tuple(kivi_block_slots_map.shape)}."
        )
    batch = int(q.shape[0])
    if int(req_indices.numel()) != batch or int(context_lens.numel()) != batch:
        raise ValueError("Full-layer KIVI decode expects one req index/context length per batch item.")
    head_dim = int(q.shape[-1])
    if head_dim != int(raw_k.shape[-1]) or head_dim != int(raw_v.shape[-1]):
        raise ValueError("Full-layer KIVI decode head_dim mismatch.")
    if head_dim not in {16, 32, 64, 128}:
        raise ValueError(f"Unsupported decode head_dim={head_dim}.")
    group_size = int(group_size)
    if group_size <= 0 or head_dim % group_size != 0:
        raise ValueError(f"Invalid KIVI group_size={group_size} for head_dim={head_dim}.")
    if group_size % 8 != 0 or head_dim % 8 != 0:
        raise ValueError(f"int4 KIVI requires group_size/head_dim divisible by 8, got {group_size}/{head_dim}.")
    block_seq = int(block_seq)
    if block_seq <= 0 or block_seq % 16 != 0:
        raise ValueError(f"block_seq must be a positive multiple of 16, got {block_seq}.")
    block_n = int(block_n)
    if block_n <= 0 or block_n % 16 != 0 or block_seq % block_n != 0:
        raise ValueError(
            "block_n must be a positive multiple of 16 and divide block_seq, "
            f"got block_n={block_n}, block_seq={block_seq}."
        )
    num_warps = int(num_warps)
    num_stages = int(num_stages)
    max_len_in_batch = int(max_len_in_batch)
    if max_len_in_batch <= 0:
        return
    if max_len_in_batch > int(raw_slots_map.shape[1]):
        raise ValueError(
            "Full-layer KIVI max_len_in_batch exceeds map width: "
            f"max_len={max_len_in_batch} width={int(raw_slots_map.shape[1])}."
        )
    if os.getenv("SPARSEENGINE_VALIDATE_KIVI_DECODE_MAP", "0") == "1":
        _validate_full_layer_kivi_decode_maps(
            raw_slots_map=raw_slots_map,
            kivi_block_slots_map=kivi_block_slots_map,
            req_indices=req_indices,
            context_lens=context_lens,
            max_len_in_batch=max_len_in_batch,
        )

    num_kv_heads = int(raw_k.shape[1])
    if int(raw_v.shape[1]) != num_kv_heads:
        raise ValueError("Full-layer KIVI decode raw K/V head count mismatch.")
    if int(key_packed.shape[1]) != num_kv_heads or int(value_packed.shape[1]) != num_kv_heads:
        raise ValueError("Full-layer KIVI packed K/V head count mismatch.")
    if int(q.shape[1]) % num_kv_heads != 0:
        raise ValueError(f"Q heads must be divisible by KV heads, got {q.shape[1]}/{num_kv_heads}.")

    fixed_grid = max_kv_splits is not None
    if fixed_grid:
        max_kv_splits = int(max_kv_splits)
        target_tokens_per_split = int(target_tokens_per_split or 0)
        if max_kv_splits <= 0 or target_tokens_per_split <= 0:
            raise ValueError(
                "Full-layer KIVI fixed-grid decode requires positive split capacity "
                "and target tokens per split."
            )
        if int(mid_out.shape[2]) != max_kv_splits or int(mid_out_logsumexp.shape[2]) != max_kv_splits:
            raise ValueError(
                "Full-layer KIVI fixed-grid workspace does not match the split envelope: "
                f"mid_out={tuple(mid_out.shape)} mid_lse={tuple(mid_out_logsumexp.shape)} "
                f"max_kv_splits={max_kv_splits}."
            )
        if req_indices.dtype != torch.int32 or req_indices.stride(0) != 1:
            raise TypeError("Full-layer KIVI fixed-grid req_indices must be contiguous int32.")
        if context_lens.dtype != torch.int32 or context_lens.stride(0) != 1:
            raise TypeError("Full-layer KIVI fixed-grid context_lens must be contiguous int32.")
        req_indices_i32 = req_indices
        context_lens_i32 = context_lens
        grid_splits = max_kv_splits
    else:
        max_kv_splits = 1
        target_tokens_per_split = 1
        req_indices_i32 = req_indices.to(torch.int32).contiguous()
        context_lens_i32 = context_lens.to(torch.int32).contiguous()
        grid_splits = triton.cdiv(max_len_in_batch, block_seq)

    grid = (batch, num_kv_heads, grid_splits)
    gqa_group_size = int(q.shape[1]) // num_kv_heads
    score_arg = attn_score if attn_score is not None else mid_out_logsumexp
    score_stride_b = score_arg.stride(0) if attn_score is not None else 0
    score_stride_h = score_arg.stride(1) if attn_score is not None else 0
    score_stride_l = score_arg.stride(2) if attn_score is not None else 0
    _full_layer_kivi_flash_decode_stage1_kernel[grid](
        q,
        raw_k,
        raw_v,
        raw_slots_map,
        kivi_block_slots_map,
        kivi_block_start_pos,
        key_packed,
        key_scales,
        key_mins,
        value_packed,
        value_scales,
        value_mins,
        req_indices_i32,
        context_lens_i32,
        mid_out,
        mid_out_logsumexp,
        score_arg,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        raw_k.stride(0),
        raw_k.stride(1),
        raw_k.stride(2),
        raw_v.stride(0),
        raw_v.stride(1),
        raw_v.stride(2),
        raw_slots_map.stride(0),
        raw_slots_map.stride(1),
        kivi_block_slots_map.stride(0),
        kivi_block_slots_map.stride(1),
        key_packed.stride(0),
        key_packed.stride(1),
        key_packed.stride(2),
        key_packed.stride(3),
        key_scales.stride(0),
        key_scales.stride(1),
        key_scales.stride(2),
        value_packed.stride(0),
        value_packed.stride(1),
        value_packed.stride(2),
        value_packed.stride(3),
        value_scales.stride(0),
        value_scales.stride(1),
        value_scales.stride(2),
        value_scales.stride(3),
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_out.stride(3),
        mid_out_logsumexp.stride(0),
        mid_out_logsumexp.stride(1),
        mid_out_logsumexp.stride(2),
        score_stride_b,
        score_stride_h,
        score_stride_l,
        sm_scale=1.0 / (head_dim ** 0.5),
        gqa_group_size=gqa_group_size,
        Q_HEAD_NUM=max(16, triton.next_power_of_2(gqa_group_size)),
        BLOCK_SEQ=block_seq,
        BLOCK_DMODEL=head_dim,
        BLOCK_N=block_n,
        GROUP_SIZE=group_size,
        FEAT_PER_INT=8,
        QUANT_MASK=15,
        STORE_SCORE=attn_score is not None,
        FIXED_GRID=fixed_grid,
        MAX_EFFECTIVE_SPLITS=max_kv_splits,
        TARGET_TOKENS_PER_SPLIT=target_tokens_per_split,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@triton.jit(do_not_specialize=["stride_ib", "stride_ob"])
def _batch_gather_mean_kernel(
    Src,  # (num_centers, D) - 源数据
    Indices,  # (B, N, K) - 索引
    Out,  # (B, N, D) - 输出
    K: tl.constexpr,
    D: tl.constexpr,
    stride_sb, stride_sd,
    stride_ib, stride_in, stride_ik,
    stride_ob, stride_on, stride_od,
    BLOCK_D: tl.constexpr,
):
    """
    批量 gather + mean: Out[b, n] = mean(Src[Indices[b, n, k]] for k in range(K))
    """
    batch_id = tl.program_id(0)
    n_id = tl.program_id(1)
    block_d = tl.program_id(2)

    offs_d = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # 累加 K 个 neighbors 的值
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for k in range(K):
        idx = tl.load(Indices + batch_id * stride_ib + n_id * stride_in + k * stride_ik)
        src_ptrs = Src + idx * stride_sb + offs_d * stride_sd
        val = tl.load(src_ptrs, mask=mask_d, other=0.0)
        acc += val

    # 计算均值
    mean_val = acc / K

    # 存储结果
    out_ptrs = Out + batch_id * stride_ob + n_id * stride_on + offs_d * stride_od
    tl.store(out_ptrs, mean_val, mask=mask_d)


@torch.no_grad()
def batch_gather_mean(
    src: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    批量 gather + mean 操作

    Args:
        src: (num_centers, D) 源数据
        indices: (B, N, K) 索引

    Returns:
        out: (B, N, D) 输出
    """
    B, N, K = indices.shape
    D = src.shape[1]

    out = torch.empty((B, N, D), dtype=src.dtype, device=src.device)

    BLOCK_D = min(128, triton.next_power_of_2(D))

    grid = (B, N, triton.cdiv(D, BLOCK_D))

    _batch_gather_mean_kernel[grid](
        src, indices, out,
        K, D,
        src.stride(0), src.stride(1),
        indices.stride(0), indices.stride(1), indices.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_D=BLOCK_D,
    )

    return out


@triton.jit
def _deltakv_gather_raw_kv_grouped_heads_kernel(
    slots_ptr,  # (N,) int32
    pos_ptr,  # (N,) int32
    cos_sin_ptr,  # (max_pos, head_dim) where [0:HD2]=cos, [HD2:]=sin
    k_cache_ptr,  # (num_slots, num_kv_heads, head_dim)
    v_cache_ptr,  # (num_slots, num_kv_heads, head_dim)
    out_ptr,  # (N, 2*D) where D=num_kv_heads*head_dim; [0:D]=K_raw_flat, [D:]=V_flat
    stride_cos_p,
    stride_cos_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_out_n,
    stride_out_d,
    D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HD2: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEADS_PER_PROG: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_hb = tl.program_id(1)

    slot = tl.load(slots_ptr + pid_n).to(tl.int32)
    pos = tl.load(pos_ptr + pid_n).to(tl.int32)

    offs = tl.arange(0, HD2)
    cos = tl.load(cos_sin_ptr + pos * stride_cos_p + offs * stride_cos_d).to(tl.float32)
    sin = tl.load(cos_sin_ptr + pos * stride_cos_p + (offs + HD2) * stride_cos_d).to(tl.float32)

    head_ids = pid_hb * HEADS_PER_PROG + tl.arange(0, HEADS_PER_PROG)
    head_mask = head_ids < NUM_KV_HEADS

    # Load K (roped) and de-RoPE it.
    k_base = slot * stride_k_s + head_ids[:, None] * stride_k_h + offs[None, :] * stride_k_d
    y1 = tl.load(k_cache_ptr + k_base, mask=head_mask[:, None], other=0.0).to(tl.float32)
    y2 = tl.load(k_cache_ptr + k_base + (HD2 * stride_k_d), mask=head_mask[:, None], other=0.0).to(tl.float32)
    x1 = y1 * cos[None, :] + y2 * sin[None, :]
    x2 = y2 * cos[None, :] - y1 * sin[None, :]

    # Load V.
    v_base = slot * stride_v_s + head_ids[:, None] * stride_v_h + offs[None, :] * stride_v_d
    v1 = tl.load(v_cache_ptr + v_base, mask=head_mask[:, None], other=0.0).to(tl.float32)
    v2 = tl.load(v_cache_ptr + v_base + (HD2 * stride_v_d), mask=head_mask[:, None], other=0.0).to(tl.float32)

    # Store flatten [K_raw (D)] + [V (D)].
    out_row = pid_n * stride_out_n
    out_k_base = out_row + (head_ids[:, None] * HEAD_DIM + offs[None, :]) * stride_out_d
    tl.store(out_ptr + out_k_base, x1, mask=head_mask[:, None])
    tl.store(out_ptr + out_k_base + (HD2 * stride_out_d), x2, mask=head_mask[:, None])

    out_v_base = out_row + (D + head_ids[:, None] * HEAD_DIM + offs[None, :]) * stride_out_d
    tl.store(out_ptr + out_v_base, v1, mask=head_mask[:, None])
    tl.store(out_ptr + out_v_base + (HD2 * stride_out_d), v2, mask=head_mask[:, None])


@torch.no_grad()
def deltakv_gather_raw_kv_grouped_heads(
    *,
    slots: torch.Tensor,  # (N,) int32
    pos: torch.Tensor,  # (N,) int32
    cos_sin: torch.Tensor,  # (max_pos, head_dim) float/bf16
    k_cache: torch.Tensor,  # (num_slots, num_kv_heads, head_dim)
    v_cache: torch.Tensor,  # (num_slots, num_kv_heads, head_dim)
    heads_per_program: int = 4,
) -> torch.Tensor:
    assert slots.is_cuda and pos.is_cuda and cos_sin.is_cuda
    assert k_cache.is_cuda and v_cache.is_cuda
    assert slots.dim() == 1 and pos.dim() == 1
    assert int(slots.numel()) == int(pos.numel())

    num_slots = int(slots.numel())
    num_kv_heads = int(k_cache.shape[1])
    head_dim = int(k_cache.shape[2])
    assert head_dim % 2 == 0
    d = num_kv_heads * head_dim

    out = torch.empty((num_slots, 2 * d), device=k_cache.device, dtype=k_cache.dtype)
    if num_slots == 0:
        return out

    heads_per_program = int(heads_per_program)
    if heads_per_program <= 0:
        raise ValueError("heads_per_program must be a positive integer.")
    grid = (num_slots, triton.cdiv(num_kv_heads, heads_per_program))
    _deltakv_gather_raw_kv_grouped_heads_kernel[grid](
        slots,
        pos,
        cos_sin,
        k_cache,
        v_cache,
        out,
        cos_sin.stride(0),
        cos_sin.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        out.stride(0),
        out.stride(1),
        D=d,
        HEAD_DIM=head_dim,
        HD2=head_dim // 2,
        NUM_KV_HEADS=num_kv_heads,
        HEADS_PER_PROG=heads_per_program,
        num_warps=4,
    )
    return out


@triton.jit
def _deltakv_reconstruct_writeback_grouped_heads_kernel(
    kv_delta_ptr,  # (N, 2*D) where D = num_kv_heads*head_dim, in de-RoPE space for K
    father_slots_ptr,  # (N, K)
    out_slots_ptr,  # (N,)
    out_pos_ptr,  # (N,)
    cos_sin_ptr,  # (max_pos, head_dim)
    k_cache_ptr,  # (num_slots, num_kv_heads, head_dim)
    v_cache_ptr,  # (num_slots, num_kv_heads, head_dim)
    pre_rope_k_cache_ptr,  # optional (num_slots, num_kv_heads, head_dim)
    ref_v_cache_ptr,  # optional (num_slots, num_kv_heads, head_dim)
    k_norm_weight_ptr,  # optional (head_dim,)
    stride_delta_n, stride_delta_d,
    stride_father_n, stride_father_k,
    stride_cos_p, stride_cos_d,
    stride_k_s, stride_k_h, stride_k_d,
    stride_v_s, stride_v_h, stride_v_d,
    stride_pk_s, stride_pk_h, stride_pk_d,
    stride_rv_s, stride_rv_h, stride_rv_d,
    stride_norm_d,
    D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HD2: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    K: tl.constexpr,
    HEADS_PER_PROG: tl.constexpr,
    USE_PRE_ROPE_K: tl.constexpr,
    USE_REF_V: tl.constexpr,
    APPLY_K_NORM: tl.constexpr,
    K_NORM_EPS: tl.constexpr,
    STORE_RAW_K: tl.constexpr,
):
    pid_n = tl.program_id(0)  # token id
    pid_hb = tl.program_id(1)  # head block id

    out_slot = tl.load(out_slots_ptr + pid_n).to(tl.int32)
    out_pos = tl.load(out_pos_ptr + pid_n).to(tl.int32)
    valid_out = (out_slot >= 0) & (out_pos >= 0)
    safe_out_slot = tl.maximum(out_slot, 0)
    safe_out_pos = tl.maximum(out_pos, 0)

    head_ids = pid_hb * HEADS_PER_PROG + tl.arange(0, HEADS_PER_PROG)
    head_mask = head_ids < NUM_KV_HEADS
    valid_head_mask = head_mask & valid_out

    offs = tl.arange(0, HD2)

    # cos/sin for output position
    cos_out = tl.load(cos_sin_ptr + safe_out_pos * stride_cos_p + offs * stride_cos_d).to(tl.float32)
    sin_out = tl.load(cos_sin_ptr + safe_out_pos * stride_cos_p + (offs + HD2) * stride_cos_d).to(tl.float32)

    # Accumulate mean of fathers in de-RoPE space.
    acc_k1 = tl.zeros([HEADS_PER_PROG, HD2], dtype=tl.float32)
    acc_k2 = tl.zeros([HEADS_PER_PROG, HD2], dtype=tl.float32)
    acc_v1 = tl.zeros([HEADS_PER_PROG, HD2], dtype=tl.float32)
    acc_v2 = tl.zeros([HEADS_PER_PROG, HD2], dtype=tl.float32)

    for kk in tl.static_range(K):
        father_slot = tl.load(
            father_slots_ptr + pid_n * stride_father_n + kk * stride_father_k,
            mask=valid_out,
            other=0,
        ).to(tl.int32)
        father_slot = tl.maximum(father_slot, 0)

        if USE_PRE_ROPE_K:
            pk_base = father_slot * stride_pk_s + head_ids[:, None] * stride_pk_h + offs[None, :] * stride_pk_d
            x1 = tl.load(pre_rope_k_cache_ptr + pk_base, mask=valid_head_mask[:, None], other=0.0).to(tl.float32)
            x2 = tl.load(
                pre_rope_k_cache_ptr + pk_base + (HD2 * stride_pk_d),
                mask=valid_head_mask[:, None],
                other=0.0,
            ).to(tl.float32)
        else:
            k_base = father_slot * stride_k_s + head_ids[:, None] * stride_k_h + offs[None, :] * stride_k_d
            x1 = tl.load(k_cache_ptr + k_base, mask=valid_head_mask[:, None], other=0.0).to(tl.float32)
            x2 = tl.load(
                k_cache_ptr + k_base + (HD2 * stride_k_d),
                mask=valid_head_mask[:, None],
                other=0.0,
            ).to(tl.float32)
        acc_k1 += x1
        acc_k2 += x2

        # Load father V.
        if USE_REF_V:
            v_base = father_slot * stride_rv_s + head_ids[:, None] * stride_rv_h + offs[None, :] * stride_rv_d
            fv1 = tl.load(ref_v_cache_ptr + v_base, mask=valid_head_mask[:, None], other=0.0).to(tl.float32)
            fv2 = tl.load(
                ref_v_cache_ptr + v_base + (HD2 * stride_rv_d),
                mask=valid_head_mask[:, None],
                other=0.0,
            ).to(tl.float32)
        else:
            v_base = father_slot * stride_v_s + head_ids[:, None] * stride_v_h + offs[None, :] * stride_v_d
            fv1 = tl.load(v_cache_ptr + v_base, mask=valid_head_mask[:, None], other=0.0).to(tl.float32)
            fv2 = tl.load(
                v_cache_ptr + v_base + (HD2 * stride_v_d),
                mask=valid_head_mask[:, None],
                other=0.0,
            ).to(tl.float32)
        acc_v1 += fv1
        acc_v2 += fv2

    inv_k = 1.0 / K
    mean_k1 = acc_k1 * inv_k
    mean_k2 = acc_k2 * inv_k
    mean_v1 = acc_v1 * inv_k
    mean_v2 = acc_v2 * inv_k

    # Load delta (de-RoPE space) for this head block.
    delta_k_base = pid_n * stride_delta_n + (head_ids[:, None] * HEAD_DIM + offs[None, :]) * stride_delta_d
    delta_k1 = tl.load(kv_delta_ptr + delta_k_base, mask=head_mask[:, None], other=0.0).to(tl.float32)
    delta_k2 = tl.load(kv_delta_ptr + delta_k_base + (HD2 * stride_delta_d), mask=head_mask[:, None], other=0.0).to(tl.float32)

    delta_v_base = pid_n * stride_delta_n + (D + head_ids[:, None] * HEAD_DIM + offs[None, :]) * stride_delta_d
    delta_v1 = tl.load(kv_delta_ptr + delta_v_base, mask=head_mask[:, None], other=0.0).to(tl.float32)
    delta_v2 = tl.load(kv_delta_ptr + delta_v_base + (HD2 * stride_delta_d), mask=head_mask[:, None], other=0.0).to(tl.float32)

    k1 = delta_k1 + mean_k1
    k2 = delta_k2 + mean_k2
    v1 = delta_v1 + mean_v1
    v2 = delta_v2 + mean_v2

    if APPLY_K_NORM and not STORE_RAW_K:
        norm_w1 = tl.load(k_norm_weight_ptr + offs * stride_norm_d).to(tl.float32)
        norm_w2 = tl.load(k_norm_weight_ptr + (offs + HD2) * stride_norm_d).to(tl.float32)
        var = tl.sum(k1 * k1 + k2 * k2, axis=1) / HEAD_DIM
        rstd = tl.rsqrt(var + K_NORM_EPS)
        k1 = k1 * rstd[:, None] * norm_w1[None, :]
        k2 = k2 * rstd[:, None] * norm_w2[None, :]

    if STORE_RAW_K:
        out_y1 = k1
        out_y2 = k2
    else:
        # Re-RoPE K to its position.
        out_y1 = k1 * cos_out[None, :] - k2 * sin_out[None, :]
        out_y2 = k2 * cos_out[None, :] + k1 * sin_out[None, :]

    # Write back into cache at out_slot.
    out_k_base = safe_out_slot * stride_k_s + head_ids[:, None] * stride_k_h + offs[None, :] * stride_k_d
    write_mask = head_mask[:, None] & valid_out
    tl.store(k_cache_ptr + out_k_base, out_y1, mask=write_mask)
    tl.store(k_cache_ptr + out_k_base + (HD2 * stride_k_d), out_y2, mask=write_mask)

    out_v_base = safe_out_slot * stride_v_s + head_ids[:, None] * stride_v_h + offs[None, :] * stride_v_d
    tl.store(v_cache_ptr + out_v_base, v1, mask=write_mask)
    tl.store(v_cache_ptr + out_v_base + (HD2 * stride_v_d), v2, mask=write_mask)


@torch.no_grad()
def deltakv_reconstruct_writeback_grouped_heads(
    kv_delta: torch.Tensor,  # (N, 2*D) in de-RoPE space for K
    father_slots: torch.Tensor,  # (N, K) int32
    out_slots: torch.Tensor,  # (N,) int32
    out_pos: torch.Tensor,  # (N,) int32
    cos_sin: torch.Tensor,  # (max_pos, head_dim) float/bf16
    k_cache: torch.Tensor,  # (num_slots, num_kv_heads, head_dim)
    v_cache: torch.Tensor,  # (num_slots, num_kv_heads, head_dim)
    *,
    heads_per_program: int = 4,
    pre_rope_k_cache: torch.Tensor | None = None,
    ref_v_cache: torch.Tensor | None = None,
    k_norm_weight: torch.Tensor | None = None,
    k_norm_eps: float = 1e-6,
    store_raw_k: bool = False,
):
    assert kv_delta.is_cuda and father_slots.is_cuda and out_slots.is_cuda and out_pos.is_cuda
    assert k_cache.is_cuda and v_cache.is_cuda and cos_sin.is_cuda
    assert father_slots.dim() == 2
    assert kv_delta.dim() == 2

    N = kv_delta.shape[0]
    K = father_slots.shape[1]
    num_kv_heads = k_cache.shape[1]
    head_dim = k_cache.shape[2]
    assert head_dim % 2 == 0
    D = num_kv_heads * head_dim
    assert kv_delta.shape[1] == 2 * D

    heads_per_program = int(heads_per_program)
    if heads_per_program <= 0:
        raise ValueError("heads_per_program must be a positive integer.")
    use_pre_rope_k = pre_rope_k_cache is not None
    use_ref_v = ref_v_cache is not None
    apply_k_norm = k_norm_weight is not None
    if use_pre_rope_k:
        assert pre_rope_k_cache.is_cuda
        assert pre_rope_k_cache.shape == k_cache.shape
    else:
        pre_rope_k_cache = k_cache
    if use_ref_v:
        assert ref_v_cache.is_cuda
        assert ref_v_cache.shape == v_cache.shape
    else:
        ref_v_cache = v_cache
    if apply_k_norm:
        assert k_norm_weight.is_cuda
        assert k_norm_weight.dim() == 1 and k_norm_weight.shape[0] == head_dim
    else:
        k_norm_weight = cos_sin

    store_raw_k = bool(store_raw_k)
    grid = (N, triton.cdiv(num_kv_heads, heads_per_program))
    _deltakv_reconstruct_writeback_grouped_heads_kernel[grid](
        kv_delta,
        father_slots,
        out_slots,
        out_pos,
        cos_sin,
        k_cache,
        v_cache,
        pre_rope_k_cache,
        ref_v_cache,
        k_norm_weight,
        kv_delta.stride(0), kv_delta.stride(1),
        father_slots.stride(0), father_slots.stride(1),
        cos_sin.stride(0), cos_sin.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        pre_rope_k_cache.stride(0), pre_rope_k_cache.stride(1), pre_rope_k_cache.stride(2),
        ref_v_cache.stride(0), ref_v_cache.stride(1), ref_v_cache.stride(2),
        k_norm_weight.stride(0),
        D=D,
        HEAD_DIM=head_dim,
        HD2=head_dim // 2,
        NUM_KV_HEADS=num_kv_heads,
        K=K,
        HEADS_PER_PROG=heads_per_program,
        USE_PRE_ROPE_K=use_pre_rope_k,
        USE_REF_V=use_ref_v,
        APPLY_K_NORM=apply_k_norm,
        K_NORM_EPS=float(k_norm_eps),
        STORE_RAW_K=store_raw_k,
        num_warps=4,
    )


@torch.no_grad()
def deltakv_materialize_sparse_view(
    active_slots: torch.Tensor,
    context_lens: torch.Tensor,
    slot_to_pos: torch.Tensor,
    postrope_mask: torch.Tensor | None,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out_k: torch.Tensor,
    out_v: torch.Tensor,
    cos_sin: torch.Tensor,
    *,
    k_norm_weight: torch.Tensor | None = None,
    k_norm_eps: float = 1e-6,
    block_tokens: int = 16,
):
    assert active_slots.is_cuda and context_lens.is_cuda and slot_to_pos.is_cuda
    assert k_cache.is_cuda and v_cache.is_cuda and out_k.is_cuda and out_v.is_cuda and cos_sin.is_cuda
    assert active_slots.dim() == 2
    assert context_lens.dim() == 1 and context_lens.shape[0] == active_slots.shape[0]
    assert k_cache.dim() == 3 and v_cache.shape == k_cache.shape
    assert out_k.dim() == 3 and out_v.shape == out_k.shape

    batch, width = active_slots.shape
    total = int(batch) * int(width)
    if total == 0:
        return
    if out_k.shape[0] < total or out_v.shape[0] < total:
        raise RuntimeError(
            "DeltaKV materialize sparse view output is too small: "
            f"out={tuple(out_k.shape)}/{tuple(out_v.shape)} need={total}."
        )
    num_kv_heads = int(k_cache.shape[1])
    head_dim = int(k_cache.shape[2])
    if head_dim % 2 != 0:
        raise RuntimeError(f"DeltaKV materialize sparse view requires an even head_dim, got {head_dim}.")

    if cos_sin.dim() == 3:
        cos_sin = cos_sin[:, 0, :]
    assert cos_sin.dim() == 2 and cos_sin.shape[1] == head_dim

    apply_k_norm = k_norm_weight is not None
    if apply_k_norm:
        assert k_norm_weight.is_cuda
        assert k_norm_weight.dim() == 1 and k_norm_weight.shape[0] == head_dim
    else:
        k_norm_weight = cos_sin

    has_postrope_mask = postrope_mask is not None
    if has_postrope_mask:
        assert postrope_mask.is_cuda
        assert postrope_mask.dim() == 1 and postrope_mask.shape[0] >= k_cache.shape[0]
    else:
        postrope_mask = slot_to_pos

    block_tokens = max(1, int(block_tokens))
    grid = (triton.cdiv(width, block_tokens), num_kv_heads, batch)
    _deltakv_materialize_sparse_view_block_kernel[grid](
        active_slots,
        context_lens,
        slot_to_pos,
        postrope_mask,
        k_cache,
        v_cache,
        out_k,
        out_v,
        cos_sin,
        k_norm_weight,
        active_slots.stride(0),
        active_slots.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        out_k.stride(0),
        out_k.stride(1),
        out_k.stride(2),
        out_v.stride(0),
        out_v.stride(1),
        out_v.stride(2),
        cos_sin.stride(0),
        cos_sin.stride(1),
        k_norm_weight.stride(0),
        WIDTH=int(width),
        BLOCK_N=block_tokens,
        NUM_SLOTS=int(k_cache.shape[0]),
        HEAD_DIM=head_dim,
        HD2=head_dim // 2,
        NUM_KV_HEADS=num_kv_heads,
        APPLY_K_NORM=apply_k_norm,
        HAS_POSTROPE_MASK=has_postrope_mask,
        K_NORM_EPS=float(k_norm_eps),
        num_warps=4,
    )


@triton.jit(do_not_specialize=["WIDTH", "stride_active_b", "stride_active_w"])
def _deltakv_materialize_sparse_view_block_kernel(
    active_slots_ptr,
    context_lens_ptr,
    slot_to_pos_ptr,
    postrope_mask_ptr,
    k_cache_ptr,
    v_cache_ptr,
    out_k_ptr,
    out_v_ptr,
    cos_sin_ptr,
    k_norm_weight_ptr,
    stride_active_b,
    stride_active_w,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_ok_n,
    stride_ok_h,
    stride_ok_d,
    stride_ov_n,
    stride_ov_h,
    stride_ov_d,
    stride_cos_p,
    stride_cos_d,
    stride_norm_d,
    WIDTH,
    BLOCK_N: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HD2: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    APPLY_K_NORM: tl.constexpr,
    HAS_POSTROPE_MASK: tl.constexpr,
    K_NORM_EPS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    head_id = tl.program_id(1)
    # Separate request rows in the grid to avoid runtime integer division.
    batch_ids = tl.program_id(2)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = batch_ids * WIDTH + cols
    token_mask = cols < WIDTH

    slots = tl.load(
        active_slots_ptr + batch_ids * stride_active_b + cols * stride_active_w,
        mask=token_mask,
        other=0,
    ).to(tl.int32)
    valid_slot = token_mask & (slots >= 0) & (slots < NUM_SLOTS)
    safe_slots = tl.minimum(tl.maximum(slots, 0), NUM_SLOTS - 1)
    pos = tl.load(slot_to_pos_ptr + safe_slots, mask=token_mask, other=0).to(tl.int32)
    safe_pos = tl.maximum(pos, 0)

    already_postrope = tl.full((BLOCK_N,), False, tl.int1)
    if HAS_POSTROPE_MASK:
        already_postrope = tl.load(postrope_mask_ptr + safe_slots, mask=valid_slot, other=0).to(tl.int1)

    offs_d = tl.arange(0, HD2)
    k_base = safe_slots[:, None] * stride_k_s + head_id * stride_k_h + offs_d[None, :] * stride_k_d
    k1 = tl.load(k_cache_ptr + k_base, mask=token_mask[:, None], other=0.0).to(tl.float32)
    k2 = tl.load(
        k_cache_ptr + k_base + (HD2 * stride_k_d),
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    if APPLY_K_NORM:
        norm_w1 = tl.load(k_norm_weight_ptr + offs_d * stride_norm_d).to(tl.float32)
        norm_w2 = tl.load(k_norm_weight_ptr + (offs_d + HD2) * stride_norm_d).to(tl.float32)
        var = tl.sum(k1 * k1 + k2 * k2, axis=1) / HEAD_DIM
        rstd = tl.rsqrt(var + K_NORM_EPS)
        norm_k1 = k1 * rstd[:, None] * norm_w1[None, :]
        norm_k2 = k2 * rstd[:, None] * norm_w2[None, :]
    else:
        norm_k1 = k1
        norm_k2 = k2

    cos = tl.load(
        cos_sin_ptr + safe_pos[:, None] * stride_cos_p + offs_d[None, :] * stride_cos_d,
        mask=token_mask[:, None],
        other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_ptr + safe_pos[:, None] * stride_cos_p + (offs_d[None, :] + HD2) * stride_cos_d,
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    rope_k1 = norm_k1 * cos - norm_k2 * sin
    rope_k2 = norm_k2 * cos + norm_k1 * sin
    out_k1 = tl.where(already_postrope[:, None], k1, rope_k1)
    out_k2 = tl.where(already_postrope[:, None], k2, rope_k2)

    out_k_base = offs_n[:, None] * stride_ok_n + head_id * stride_ok_h + offs_d[None, :] * stride_ok_d
    tl.store(out_k_ptr + out_k_base, out_k1, mask=token_mask[:, None])
    tl.store(out_k_ptr + out_k_base + (HD2 * stride_ok_d), out_k2, mask=token_mask[:, None])

    v_base = safe_slots[:, None] * stride_v_s + head_id * stride_v_h + offs_d[None, :] * stride_v_d
    v1 = tl.load(v_cache_ptr + v_base, mask=token_mask[:, None], other=0.0)
    v2 = tl.load(v_cache_ptr + v_base + (HD2 * stride_v_d), mask=token_mask[:, None], other=0.0)
    out_v_base = offs_n[:, None] * stride_ov_n + head_id * stride_ov_h + offs_d[None, :] * stride_ov_d
    tl.store(out_v_ptr + out_v_base, v1, mask=token_mask[:, None])
    tl.store(out_v_ptr + out_v_base + (HD2 * stride_ov_d), v2, mask=token_mask[:, None])


@triton.jit
def _deltakv_static_decode_plan_kernel(
    raw_slots_map_ptr,  # (num_rows, max_positions), int32
    latent_slots_map_ptr,  # (num_rows, max_positions), int32
    active_compressed_ptr,  # (B, K), int32
    req_indices_ptr,  # (B,), int32
    context_lens_ptr,  # (B,), int32
    compressed_lens_ptr,  # (B,), int32
    temp_slots_ptr,  # (B, K), int32
    active_slots_out_ptr,  # (B, MAX_S), int32
    active_pos_out_ptr,  # (B, MAX_S), int32
    new_context_lens_out_ptr,  # (B,), int32
    recon_pos_out_ptr,  # (B*K,), int32
    recon_latent_out_ptr,  # (B*K,), int32
    recon_out_slot_out_ptr,  # (B*K,), int32
    stride_raw_r,
    stride_raw_p,
    stride_latent_r,
    stride_latent_p,
    stride_active_b,
    stride_active_k,
    stride_temp_b,
    stride_temp_k,
    stride_out_b,
    stride_out_s,
    stride_pos_b,
    stride_pos_s,
    SINK: tl.constexpr,
    K_MAX: tl.constexpr,
    MAX_BUFFER: tl.constexpr,
    MAX_S: tl.constexpr,
    MAX_POS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    b = tl.program_id(0)
    cols = tl.arange(0, BLOCK_M)
    mask = cols < MAX_S

    row = tl.load(req_indices_ptr + b).to(tl.int32)
    context_len = tl.load(context_lens_ptr + b).to(tl.int32)
    compressed_len = tl.load(compressed_lens_ptr + b).to(tl.int32)
    context_len = tl.maximum(context_len, 0)
    sink_len = tl.minimum(SINK, context_len)
    compressed_len = tl.minimum(tl.maximum(compressed_len, 0), context_len - sink_len)
    top_len = tl.minimum(compressed_len, K_MAX)

    safe_slot = tl.full([BLOCK_M], 0, dtype=tl.int32)
    if SINK > 0:
        first_sink = tl.load(
            raw_slots_map_ptr + row * stride_raw_r,
            mask=context_len > 0,
            other=0,
        ).to(tl.int32)
        safe_slot = tl.zeros([BLOCK_M], dtype=tl.int32) + tl.maximum(first_sink, 0)

    out_slot = safe_slot
    out_pos = tl.zeros([BLOCK_M], dtype=tl.int32)

    if SINK > 0:
        sink_mask = cols < sink_len
        sink_pos = tl.minimum(cols, MAX_POS)
        sink_slots = tl.load(
            raw_slots_map_ptr + row * stride_raw_r + sink_pos * stride_raw_p,
            mask=sink_mask & mask,
            other=0,
        ).to(tl.int32)
        out_slot = tl.where(sink_mask, sink_slots, out_slot)
        out_pos = tl.where(sink_mask, sink_pos, out_pos)

    top_start = sink_len
    top_capacity_end = sink_len + K_MAX
    top_capacity_mask = (cols >= top_start) & (cols < top_capacity_end)
    top_j = cols - top_start
    top_rel = tl.load(
        active_compressed_ptr + b * stride_active_b + top_j * stride_active_k,
        mask=top_capacity_mask & mask,
        other=-1,
    ).to(tl.int32)
    top_pos = top_rel + SINK
    # Static CUDA graph keeps K_MAX columns for shape stability, but only
    # top_len are logically visible. The buffer must be compacted after those
    # visible top slots; otherwise duplicate safe slots enter attention.
    top_mask = top_capacity_mask & (top_j < top_len)
    valid_top = top_mask & (top_rel >= 0) & (top_rel < compressed_len) & (top_pos < context_len)
    safe_top_pos = tl.minimum(tl.maximum(top_pos, 0), MAX_POS)
    top_raw_slots = tl.load(
        raw_slots_map_ptr + row * stride_raw_r + safe_top_pos * stride_raw_p,
        mask=top_mask & mask,
        other=0,
    ).to(tl.int32)
    top_latent_slots = tl.load(
        latent_slots_map_ptr + row * stride_latent_r + safe_top_pos * stride_latent_p,
        mask=top_mask & mask,
        other=-1,
    ).to(tl.int32)
    temp_slots = tl.load(
        temp_slots_ptr + b * stride_temp_b + top_j * stride_temp_k,
        mask=top_mask & mask,
        other=0,
    ).to(tl.int32)
    # Full-prefill DeltaKV stores latent payloads for all finalized compressed
    # positions, including center tokens that also retain raw slots as refs.
    # Selected compressed positions reconstruct through the latent path, so
    # static decode must prefer latent when present instead of short-circuiting
    # on an existing raw center slot.
    need_reconstruct = valid_top & (top_latent_slots >= 0)
    top_out = tl.where(
        need_reconstruct,
        temp_slots,
        tl.where(valid_top, tl.maximum(top_raw_slots, 0), safe_slot),
    )
    out_slot = tl.where(top_mask, top_out, out_slot)
    out_pos = tl.where(top_mask, tl.where(valid_top, top_pos, 0), out_pos)

    recon_index = b * K_MAX + top_j
    tl.store(
        recon_pos_out_ptr + recon_index,
        tl.where(need_reconstruct, top_pos, -1),
        mask=top_capacity_mask & mask,
    )
    tl.store(
        recon_latent_out_ptr + recon_index,
        tl.where(need_reconstruct, top_latent_slots, -1),
        mask=top_capacity_mask & mask,
    )
    tl.store(
        recon_out_slot_out_ptr + recon_index,
        tl.where(need_reconstruct, temp_slots, -1),
        mask=top_capacity_mask & mask,
    )

    buffer_start = sink_len + compressed_len
    raw_buffer_len = context_len - buffer_start
    buffer_len = tl.minimum(tl.maximum(raw_buffer_len, 0), MAX_BUFFER)
    buffer_start_out = sink_len + top_len
    buffer_j = cols - buffer_start_out
    buffer_mask = cols >= buffer_start_out
    buffer_pos = tl.minimum(tl.maximum(buffer_start + buffer_j, 0), MAX_POS)
    buffer_valid = buffer_mask & (buffer_j < buffer_len)
    buffer_slots = tl.load(
        raw_slots_map_ptr + row * stride_raw_r + buffer_pos * stride_raw_p,
        mask=buffer_valid & mask,
        other=0,
    ).to(tl.int32)
    buffer_out = tl.where(buffer_valid, tl.maximum(buffer_slots, 0), safe_slot)
    out_slot = tl.where(buffer_mask, buffer_out, out_slot)
    out_pos = tl.where(buffer_mask, tl.where(buffer_valid, buffer_pos, 0), out_pos)

    tl.store(
        active_slots_out_ptr + b * stride_out_b + cols * stride_out_s,
        out_slot,
        mask=mask,
    )
    tl.store(
        active_pos_out_ptr + b * stride_pos_b + cols * stride_pos_s,
        out_pos,
        mask=mask,
    )

    tl.store(new_context_lens_out_ptr + b, sink_len + top_len + buffer_len)


@torch.no_grad()
def deltakv_static_decode_plan(
    *,
    raw_slots_map: torch.Tensor,
    latent_slots_map: torch.Tensor,
    active_compressed_indices: torch.Tensor,
    req_indices: torch.Tensor,
    context_lens: torch.Tensor,
    compressed_lens: torch.Tensor,
    temp_slots: torch.Tensor,
    active_slots_out: torch.Tensor,
    active_pos_out: torch.Tensor,
    new_context_lens_out: torch.Tensor,
    recon_pos_out: torch.Tensor,
    recon_latent_out: torch.Tensor,
    recon_out_slot_out: torch.Tensor,
    sink: int,
    max_buffer: int,
):
    assert raw_slots_map.is_cuda and latent_slots_map.is_cuda
    assert active_compressed_indices.is_cuda and req_indices.is_cuda
    assert context_lens.is_cuda and compressed_lens.is_cuda and temp_slots.is_cuda
    assert active_slots_out.is_cuda and active_pos_out.is_cuda and new_context_lens_out.is_cuda
    assert recon_pos_out.is_cuda and recon_latent_out.is_cuda and recon_out_slot_out.is_cuda
    assert raw_slots_map.dim() == 2 and latent_slots_map.dim() == 2
    assert active_compressed_indices.dim() == 2 and temp_slots.dim() == 2
    assert req_indices.dim() == 1 and context_lens.dim() == 1 and compressed_lens.dim() == 1
    assert active_slots_out.dim() == 2 and active_pos_out.dim() == 2

    batch_size = int(req_indices.shape[0])
    k_max = int(active_compressed_indices.shape[1])
    sink = int(sink)
    max_buffer = int(max_buffer)
    max_s = sink + k_max + max_buffer
    if active_slots_out.shape != (batch_size, max_s):
        raise ValueError(
            "DeltaKV static decode plan active_slots_out shape mismatch: "
            f"got={tuple(active_slots_out.shape)}, expected=({batch_size}, {max_s})."
        )
    if active_pos_out.shape != (batch_size, max_s):
        raise ValueError(
            "DeltaKV static decode plan active_pos_out shape mismatch: "
            f"got={tuple(active_pos_out.shape)}, expected=({batch_size}, {max_s})."
        )
    if temp_slots.shape != (batch_size, k_max):
        raise ValueError(
            "DeltaKV static decode plan temp_slots shape mismatch: "
            f"got={tuple(temp_slots.shape)}, expected=({batch_size}, {k_max})."
        )
    if recon_pos_out.numel() != batch_size * k_max:
        raise ValueError("DeltaKV static decode plan recon outputs must have B*K elements.")
    if batch_size == 0:
        return

    block_m = 1 << max(0, (max_s - 1).bit_length())
    _deltakv_static_decode_plan_kernel[(batch_size,)](
        raw_slots_map,
        latent_slots_map,
        active_compressed_indices,
        req_indices,
        context_lens,
        compressed_lens,
        temp_slots,
        active_slots_out,
        active_pos_out,
        new_context_lens_out,
        recon_pos_out,
        recon_latent_out,
        recon_out_slot_out,
        raw_slots_map.stride(0),
        raw_slots_map.stride(1),
        latent_slots_map.stride(0),
        latent_slots_map.stride(1),
        active_compressed_indices.stride(0),
        active_compressed_indices.stride(1),
        temp_slots.stride(0),
        temp_slots.stride(1),
        active_slots_out.stride(0),
        active_slots_out.stride(1),
        active_pos_out.stride(0),
        active_pos_out.stride(1),
        SINK=sink,
        K_MAX=k_max,
        MAX_BUFFER=max_buffer,
        MAX_S=max_s,
        MAX_POS=int(raw_slots_map.shape[1]) - 1,
        BLOCK_M=block_m,
        num_warps=8 if block_m >= 2048 else 4,
    )


@triton.jit
def _deltakv_l2_topk_block_kernel(
    A_ptr,  # (N, D) tokens
    B_ptr,  # (M, D) centers
    Center_Pos_ptr,  # (M - m0,) int64/int32 center positions, optional when HAS_CENTER_POS
    out_scores_ptr,  # (NB, MB, BN, K) low-precision scores
    out_indices_ptr,  # (NB, MB, BN, K) int32 global center indices
    N,
    M,
    m0,  # number of existing centers
    cluster_step,  # int32
    stride_an,
    stride_ad,
    stride_bm,
    stride_bd,
    stride_s_nb,
    stride_s_mb,
    stride_s_bn,
    stride_s_k,
    stride_i_nb,
    stride_i_mb,
    stride_i_bn,
    stride_i_k,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_CENTER_POS: tl.constexpr,
):
    pid_n = tl.program_id(0)  # token block
    pid_m = tl.program_id(1)  # center block

    n_local = tl.arange(0, BLOCK_N)
    n_offs = pid_n * BLOCK_N + n_local
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    n_mask = n_offs < N
    m_mask = m_offs < M

    # GEMM tile: (BLOCK_N, BLOCK_M) = (BLOCK_N, D) x (BLOCK_M, D)^T
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    b_norm = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for d_start in tl.static_range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        a_ptrs = A_ptr + n_offs[:, None] * stride_an + d_offs[None, :] * stride_ad
        b_ptrs = B_ptr + m_offs[:, None] * stride_bm + d_offs[None, :] * stride_bd

        a = tl.load(a_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

        acc += tl.dot(a, tl.trans(b))
        bf = b.to(tl.float32)
        b_norm += tl.sum(bf * bf, axis=1)

    scores = acc * 2.0 - b_norm[None, :]

    # Causal mask for new centers. Existing centers (m < m0) are always visible.
    # Regular centers use (m - m0) * cluster_step. Dynamic-stride centers pass
    # their explicit relative positions via Center_Pos_ptr.
    # Existing centers (m < m0) are always visible.
    g = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
    # Keep scalars in int32 for indexing/masking math.
    m0_i32 = m0.to(tl.int32)
    cs_i32 = cluster_step.to(tl.int32)
    if HAS_CENTER_POS:
        new_id = g - m0_i32
        loaded_pos = tl.load(Center_Pos_ptr + new_id, mask=(new_id >= 0) & (g < M), other=0).to(tl.int32)
        regular_pos = (g - m0_i32) * cs_i32
        new_pos = tl.where(g >= m0_i32, loaded_pos, regular_pos)
    else:
        new_pos = (g - m0_i32) * cs_i32  # negative for existing centers
    allow = new_pos[None, :] <= n_offs[:, None].to(tl.int32)

    valid = n_mask[:, None] & m_mask[None, :] & allow
    scores = tl.where(valid, scores, -float("inf"))

    # Top-k within this center block (over BLOCK_M), store directly to output.
    idxs = tl.arange(0, BLOCK_M)[None, :].to(tl.int32)
    s_base = (
        out_scores_ptr
        + pid_n * stride_s_nb
        + pid_m * stride_s_mb
        + n_local * stride_s_bn
    )
    i_base = (
        out_indices_ptr
        + pid_n * stride_i_nb
        + pid_m * stride_i_mb
        + n_local * stride_i_bn
    )
    for kk in tl.static_range(K):
        maxv = tl.max(scores, axis=1)
        is_max = scores == maxv[:, None]
        arg = tl.min(tl.where(is_max, idxs, BLOCK_M), axis=1).to(tl.int32)
        tl.store(s_base + kk * stride_s_k, maxv, mask=n_mask)
        tl.store(i_base + kk * stride_i_k, pid_m * BLOCK_M + arg, mask=n_mask)
        scores = tl.where(idxs == arg[:, None], -float("inf"), scores)


@torch.no_grad()
def deltakv_l2_topk_blockwise(
    *,
    tokens: torch.Tensor,  # (N, D) bf16/fp16
    centers: torch.Tensor,  # (M, D) bf16/fp16
    m0: int,
    cluster_step: int,
    k: int,
    new_center_rel: torch.Tensor | None = None,
    block_n: int = 16,
    block_m: int = 64,
    block_d: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute top-k L2-equivalent scores in a blockwise fused way.

    Returns:
      partial_scores: (NB, MB, BN, K) low-precision
      partial_indices: (NB, MB, BN, K) int32 global center indices
    """
    assert tokens.is_cuda and centers.is_cuda
    assert tokens.dim() == 2 and centers.dim() == 2
    N, D = tokens.shape
    M, D2 = centers.shape
    assert D == D2
    assert k >= 1
    assert block_m >= k
    if new_center_rel is not None:
        if not new_center_rel.is_cuda:
            raise ValueError("new_center_rel must be a CUDA tensor when provided.")
        if new_center_rel.dim() != 1:
            raise ValueError(f"new_center_rel must be rank-1, got shape={tuple(new_center_rel.shape)}.")
        if int(new_center_rel.numel()) != max(0, M - int(m0)):
            raise ValueError(
                "new_center_rel length must match the number of new centers: "
                f"len={new_center_rel.numel()}, M={M}, m0={m0}."
            )
        new_center_rel = new_center_rel.to(device=tokens.device, dtype=torch.int32).contiguous()
    else:
        new_center_rel = torch.empty((0,), dtype=torch.int32, device=tokens.device)

    block_n = int(os.getenv("SPARSEENGINE_DELTAKV_L2_BLOCK_N", str(block_n)))
    block_m = int(os.getenv("SPARSEENGINE_DELTAKV_L2_BLOCK_M", str(block_m)))
    block_d = int(os.getenv("SPARSEENGINE_DELTAKV_L2_BLOCK_D", str(block_d)))
    num_warps = int(os.getenv("SPARSEENGINE_DELTAKV_L2_NUM_WARPS", "4"))
    if block_n <= 0 or block_m <= 0 or block_d <= 0:
        raise ValueError("DeltaKV L2 block sizes must be positive.")
    if block_m < k:
        block_m = max(block_m, k)

    NB = triton.cdiv(N, block_n)
    MB = triton.cdiv(M, block_m)

    partial_scores = torch.empty((NB, MB, block_n, k), device=tokens.device, dtype=tokens.dtype)
    partial_indices = torch.empty((NB, MB, block_n, k), device=tokens.device, dtype=torch.int32)

    grid = (NB, MB)
    _deltakv_l2_topk_block_kernel[grid](
        tokens,
        centers,
        new_center_rel,
        partial_scores,
        partial_indices,
        N,
        M,
        m0,
        cluster_step,
        tokens.stride(0),
        tokens.stride(1),
        centers.stride(0),
        centers.stride(1),
        partial_scores.stride(0),
        partial_scores.stride(1),
        partial_scores.stride(2),
        partial_scores.stride(3),
        partial_indices.stride(0),
        partial_indices.stride(1),
        partial_indices.stride(2),
        partial_indices.stride(3),
        D=D,
        K=k,
        BLOCK_N=block_n,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        HAS_CENTER_POS=bool(new_center_rel.numel() > 0),
        num_warps=num_warps,
    )
    return partial_scores, partial_indices
