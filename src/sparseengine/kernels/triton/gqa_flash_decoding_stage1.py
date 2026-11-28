import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_flash_decode_stage1(
    Q,
    K,
    V,
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    Attn_Score,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    stride_asbs,
    stride_ash,
    stride_asl,
    gqa_group_size,
    Q_HEAD_NUM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCORE_MODE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)

    cur_q_head_offs = tl.arange(0, Q_HEAD_NUM)
    cur_q_head_range = cur_kv_head * gqa_group_size + cur_q_head_offs

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_index = seq_start_block * BLOCK_SEQ
    cur_batch_end_index = tl.minimum(cur_batch_seq_len, cur_batch_start_index + BLOCK_SEQ)

    off_q = cur_batch * stride_qbs + cur_q_head_range[:, None] * stride_qh + offs_d[None, :]

    block_n_size = (
        tl.where(
            cur_batch_end_index - cur_batch_start_index <= 0,
            0,
            cur_batch_end_index - cur_batch_start_index + BLOCK_N - 1,
        )
        // BLOCK_N
    )

    offs_n = cur_batch_start_index + tl.arange(0, BLOCK_N)

    q = tl.load(Q + off_q, mask=cur_q_head_range[:, None] < (cur_kv_head + 1) * gqa_group_size, other=0.0)

    sum_exp = tl.zeros([Q_HEAD_NUM], dtype=tl.float32)
    max_logic = tl.zeros([Q_HEAD_NUM], dtype=tl.float32) - float("inf")
    acc = tl.zeros([Q_HEAD_NUM, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, block_n_size, 1):
        offs_n_new = start_n * BLOCK_N + offs_n
        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        )
        off_k = k_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
        k = tl.load(K + off_k, mask=offs_n_new[None, :] < cur_batch_end_index, other=0.0)
        att_value = tl.dot(q, k)
        key_mask = offs_n_new < cur_batch_end_index
        if SCORE_MODE != 0:
            att_value = tl.where(key_mask[None, :], att_value, float("-inf"))
            head_mask = cur_q_head_range < (cur_kv_head + 1) * gqa_group_size
            score_offset = cur_batch * stride_asbs + offs_n_new * stride_asl
            if SCORE_MODE == 3:
                tl.store(
                    Attn_Score + score_offset[None, :] + cur_q_head_range[:, None] * stride_ash,
                    att_value, mask=head_mask[:, None] & key_mask[None, :],
                )
            else:
                max_score = tl.max(tl.where(head_mask[:, None], att_value, float("-inf")), axis=0)
                tl.atomic_max(Attn_Score + score_offset, max_score, mask=key_mask)
        att_value *= sm_scale
        if SCORE_MODE == 0:
            att_value = tl.where(key_mask[None, :], att_value, float("-inf"))
        v = tl.load(
            V + k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :],
            mask=offs_n_new[:, None] < cur_batch_end_index,
            other=0.0,
        )

        cur_max_logic = tl.max(att_value, axis=1)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)

        exp_logic = tl.exp(att_value - new_max_logic[:, None])
        logic_scale = tl.exp(max_logic - new_max_logic)
        acc *= logic_scale[:, None]
        acc += tl.dot(exp_logic.to(v.dtype), v)

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=1)
        max_logic = new_max_logic

    off_mid_o = (
        cur_batch * stride_mid_ob
        + cur_q_head_range[:, None] * stride_mid_oh
        + seq_start_block * stride_mid_os
        + offs_d[None, :]
    )
    off_mid_o_logexpsum = cur_batch * stride_mid_o_eb + cur_q_head_range * stride_mid_o_eh + seq_start_block
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
    return


def _assert_supported_layout(
    q,
    k,
    v,
    req_to_tokens,
    b_req_idx,
    b_seqlen,
    mid_out,
    mid_out_logsumexp,
):
    assert q.stride(-1) == 1, f"q head_dim must be contiguous, got stride={q.stride()}."
    assert k.stride(-1) == 1, f"k head_dim must be contiguous, got stride={k.stride()}."
    assert v.stride(-1) == 1, f"v head_dim must be contiguous, got stride={v.stride()}."
    assert k.stride() == v.stride(), (
        "k and v must have identical layouts because the GQA decode kernel shares their strides, "
        f"got k_stride={k.stride()} v_stride={v.stride()}."
    )
    assert req_to_tokens.stride(-1) == 1, (
        f"req_to_tokens sequence dimension must be contiguous, got stride={req_to_tokens.stride()}."
    )
    assert b_req_idx.stride(0) == 1, f"b_req_idx must be contiguous, got stride={b_req_idx.stride()}."
    assert b_seqlen.stride(0) == 1, f"b_seqlen must be contiguous, got stride={b_seqlen.stride()}."
    assert mid_out.stride(-1) == 1, (
        f"mid_out head_dim must be contiguous, got stride={mid_out.stride()}."
    )
    assert mid_out_logsumexp.stride(-1) == 1, (
        "mid_out_logsumexp block dimension must be contiguous, "
        f"got stride={mid_out_logsumexp.stride()}."
    )


def _launch_stage1(
    q, k, v, Req_to_tokens, B_req_idx, B_Seqlen, max_len_in_batch,
    mid_out, mid_out_logsumexp, block_seq, block_n, num_warps, num_stages,
    *, with_score=False, attn_score=None,
):
    BLOCK_SEQ = block_seq
    BLOCK_N = block_n
    assert BLOCK_SEQ % BLOCK_N == 0
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128, 256}
    if with_score:
        assert q.dtype == k.dtype and k.dtype == v.dtype
    _assert_supported_layout(q, k, v, Req_to_tokens, B_req_idx, B_Seqlen, mid_out, mid_out_logsumexp)
    sm_scale = 1.0 / (Lk ** 0.5)
    batch, kv_head_num = B_req_idx.shape[0], k.shape[1]
    grid = (batch, kv_head_num, triton.cdiv(max_len_in_batch, BLOCK_SEQ))
    gqa_group_size = q.shape[1] // k.shape[1]

    score_mode = 0
    stride_asbs = stride_ash = stride_asl = 0
    if with_score:
        score_mode = 3 if attn_score.dim() == 3 else 2
        stride_asbs = attn_score.stride(0)
        stride_ash = attn_score.stride(1) if score_mode == 3 else 0
        stride_asl = attn_score.stride(2 if score_mode == 3 else 1)
    _fwd_kernel_flash_decode_stage1[grid](
        q, k, v, sm_scale, Req_to_tokens, B_req_idx, B_Seqlen,
        mid_out, mid_out_logsumexp, attn_score,
        Req_to_tokens.stride(0), Req_to_tokens.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
        mid_out_logsumexp.stride(0), mid_out_logsumexp.stride(1), mid_out_logsumexp.stride(2),
        stride_asbs, stride_ash, stride_asl, gqa_group_size,
        Q_HEAD_NUM=max(16, triton.next_power_of_2(gqa_group_size)),
        BLOCK_SEQ=BLOCK_SEQ, BLOCK_DMODEL=Lk, BLOCK_N=BLOCK_N, SCORE_MODE=score_mode,
        num_warps=num_warps, num_stages=num_stages,
    )


@torch.no_grad()
def flash_decode_stage1(
    q,
    k,
    v,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    max_len_in_batch,
    mid_out,
    mid_out_logsumexp,
    block_seq,
    block_n=16,
    num_warps=2,
    num_stages=2,
):
    _launch_stage1(
        q, k, v, Req_to_tokens, B_req_idx, B_Seqlen, max_len_in_batch,
        mid_out, mid_out_logsumexp, block_seq, block_n, num_warps, num_stages,
    )


@torch.no_grad()
def flash_decode_stage1_with_score(
    q, k, v, Req_to_tokens, B_req_idx, B_Seqlen, max_len_in_batch,
    mid_out, mid_out_logsumexp, attn_score, block_seq,
):
    _launch_stage1(
        q, k, v, Req_to_tokens, B_req_idx, B_Seqlen, max_len_in_batch,
        mid_out, mid_out_logsumexp, block_seq, 16, 2, 2,
        with_score=True, attn_score=attn_score,
    )
