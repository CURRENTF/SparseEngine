import torch
import triton
import triton.language as tl


class PrefillScoreWorkspace:
    """Reusable probability-score workspace owned by one runtime worker."""

    def __init__(self) -> None:
        self._partial_lse: torch.Tensor | None = None
        self._global_lse: torch.Tensor | None = None
        self._head_score: torch.Tensor | None = None

    @staticmethod
    def _reserve(
        tensor: torch.Tensor | None,
        elements: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if (
            tensor is None
            or tensor.device != device
            or int(tensor.numel()) < int(elements)
        ):
            tensor = torch.empty((int(elements),), device=device, dtype=torch.float32)
        return tensor

    def probability_lse_buffers(
        self,
        *,
        group_count: int,
        candidate_blocks: int,
        block_rows: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        partial_elements = int(group_count) * int(candidate_blocks) * int(block_rows)
        global_elements = int(group_count) * int(block_rows)
        self._partial_lse = self._reserve(
            self._partial_lse,
            partial_elements,
            device=device,
        )
        self._global_lse = self._reserve(
            self._global_lse,
            global_elements,
            device=device,
        )
        partial_shape = (int(group_count), int(candidate_blocks), int(block_rows))
        global_shape = (int(group_count), int(block_rows))
        return (
            self._partial_lse[:partial_elements].view(partial_shape),
            self._global_lse[:global_elements].view(global_shape),
        )

    def probability_head_score_buffer(
        self,
        *,
        batch_size: int,
        query_heads: int,
        score_width: int,
        device: torch.device,
    ) -> torch.Tensor:
        elements = int(batch_size) * int(query_heads) * int(score_width)
        self._head_score = self._reserve(
            self._head_score,
            elements,
            device=device,
        )
        return self._head_score[:elements].view(
            int(batch_size), int(query_heads), int(score_width)
        )

# Exact request lengths stay runtime values; tile shapes remain specialized.
@triton.jit(
    do_not_specialize=["NUM_BLOCKS", "QUERY_BLOCKS", "candidate_start", "recent_keep_tokens"]
)
def _prefill_score_partial_lse_kernel(
    Q,
    K,
    Partial_LSE,
    B_Seqlen,
    Req_to_tokens,
    B_req_idx,
    Batch_Indices,
    Score_Q_Start,
    Score_Q_End,
    B_Start_Loc,
    B_Prompt_Cache_Len,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    H_PER_KV: tl.constexpr,
    H_KV: tl.constexpr,
    HEAD_BLOCKS: tl.constexpr,
    QUERY_BLOCKS,
    USE_BATCH_INDICES: tl.constexpr,
    candidate_start,
    recent_keep_tokens,
    sm_scale: tl.constexpr,
    NUM_BLOCKS,
    BLOCK_H: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    storage_group = tl.program_id(0)
    cur_query_block = storage_group % QUERY_BLOCKS
    cur_group = storage_group // QUERY_BLOCKS
    cur_n_block = tl.program_id(1)
    cur_head_block = cur_group % HEAD_BLOCKS
    cur_bkv = cur_group // HEAD_BLOCKS
    cur_batch = cur_bkv // H_KV
    cur_kv_head = cur_bkv % H_KV
    source_batch = (
        tl.load(Batch_Indices + cur_batch) if USE_BATCH_INDICES else cur_batch
    )

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + source_batch)
    prompt_cache_len = tl.load(B_Prompt_Cache_Len + source_batch)
    context_len = tl.load(B_Seqlen + source_batch)
    cur_batch_seq_len = context_len - prompt_cache_len
    cur_batch_req_idx = tl.load(B_req_idx + source_batch)
    score_q_start = tl.load(Score_Q_Start + cur_batch)
    score_q_end = tl.load(Score_Q_End + cur_batch)

    offs_rows = tl.arange(0, BLOCK_ROWS)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)
    local_head = cur_head_block * BLOCK_H + offs_rows // BLOCK_M
    q_head = cur_kv_head * H_PER_KV + local_head
    q_abs_pos = (
        score_q_start
        + cur_query_block * BLOCK_M
        + (offs_rows % BLOCK_M)
    )
    q_rel_pos = q_abs_pos - prompt_cache_len
    q_row_valid = (
        (local_head < H_PER_KV)
        & (q_abs_pos < score_q_end)
        & (q_rel_pos >= 0)
        & (q_rel_pos < cur_batch_seq_len)
    )

    off_q = (
        (cur_batch_in_all_start_index + q_rel_pos[:, None]) * stride_qt
        + q_head[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + off_q, mask=q_row_valid[:, None], other=0.0)

    start_n = cur_n_block * BLOCK_N
    kv_pos = start_n + offs_n
    candidate_end = tl.maximum(candidate_start, context_len - recent_keep_tokens)
    kv_in_candidate = (kv_pos >= candidate_start) & (kv_pos < candidate_end)
    kv_loc = tl.load(
        Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * kv_pos,
        mask=kv_in_candidate,
        other=0,
    )
    off_k = kv_loc[None, :] * stride_ks + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
    k = tl.load(K + off_k, mask=kv_in_candidate[None, :], other=0.0)

    qk = tl.dot(q, k) * sm_scale
    causal_mask = q_abs_pos[:, None] >= kv_pos[None, :]
    valid = q_row_valid[:, None] & kv_in_candidate[None, :] & causal_mask
    qk = tl.where(valid, qk, -1.0e20)
    m_i = tl.max(qk, axis=1)
    p = tl.exp(qk - m_i[:, None])
    p = tl.where(valid, p, 0.0)
    l_i = tl.sum(p, axis=1)

    stats_offs = (
        (storage_group * NUM_BLOCKS + cur_n_block) * BLOCK_ROWS + offs_rows
    )
    # Empty rows keep a finite sentinel; the final valid mask zeros their scores.
    tl.store(Partial_LSE + stats_offs, tl.where(l_i > 0.0, m_i + tl.log(l_i), -1.0e20))


@triton.jit(do_not_specialize=["NUM_BLOCKS"])
def _prefill_score_reduce_lse_kernel(
    Partial_LSE,
    Global_LSE,
    NUM_BLOCKS,
    BLOCK_ROWS: tl.constexpr,
    REDUCE_BLOCKS: tl.constexpr,
    REDUCE_ROWS: tl.constexpr,
):
    cur_group = tl.program_id(0)
    cur_row_block = tl.program_id(1)
    offs_rows = cur_row_block * REDUCE_ROWS + tl.arange(0, REDUCE_ROWS)
    offs_blocks = tl.arange(0, REDUCE_BLOCKS)

    stats_offs = (
        cur_group * NUM_BLOCKS * BLOCK_ROWS
        + offs_blocks[None, :] * BLOCK_ROWS
        + offs_rows[:, None]
    )
    mask = (offs_rows[:, None] < BLOCK_ROWS) & (offs_blocks[None, :] < NUM_BLOCKS)
    partial_lse = tl.load(Partial_LSE + stats_offs, mask=mask, other=-1.0e20)
    m_i = tl.max(partial_lse, axis=1)
    l_i = tl.sum(tl.where(mask, tl.exp(partial_lse - m_i[:, None]), 0.0), axis=1)

    out_offs = cur_group * BLOCK_ROWS + offs_rows
    tl.store(Global_LSE + out_offs, m_i + tl.log(l_i), mask=offs_rows < BLOCK_ROWS)


@triton.jit(
    do_not_specialize=["QUERY_BLOCKS", "candidate_start", "recent_keep_tokens"]
)
def _prefill_score_final_kernel(
    Q,
    K,
    Attn_Score,
    Head_Score,
    Global_LSE,
    B_Seqlen,
    Req_to_tokens,
    B_req_idx,
    Batch_Indices,
    Score_Q_Start,
    Score_Q_End,
    B_Start_Loc,
    B_Prompt_Cache_Len,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_asb,
    stride_asl,
    stride_hsb,
    stride_hsh,
    stride_hsl,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    H_PER_KV: tl.constexpr,
    H_KV: tl.constexpr,
    HEAD_BLOCKS: tl.constexpr,
    QUERY_BLOCKS,
    WRITE_PER_HEAD: tl.constexpr,
    USE_BATCH_INDICES: tl.constexpr,
    candidate_start,
    recent_keep_tokens,
    sm_scale: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    storage_group = tl.program_id(0)
    cur_query_block = storage_group % QUERY_BLOCKS
    cur_group = storage_group // QUERY_BLOCKS
    cur_n_block = tl.program_id(1)
    cur_head_block = cur_group % HEAD_BLOCKS
    cur_bkv = cur_group // HEAD_BLOCKS
    cur_batch = cur_bkv // H_KV
    cur_kv_head = cur_bkv % H_KV
    source_batch = (
        tl.load(Batch_Indices + cur_batch) if USE_BATCH_INDICES else cur_batch
    )

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + source_batch)
    prompt_cache_len = tl.load(B_Prompt_Cache_Len + source_batch)
    context_len = tl.load(B_Seqlen + source_batch)
    cur_batch_seq_len = context_len - prompt_cache_len
    cur_batch_req_idx = tl.load(B_req_idx + source_batch)
    score_q_start = tl.load(Score_Q_Start + cur_batch)
    score_q_end = tl.load(Score_Q_End + cur_batch)
    score_q_len = tl.maximum(score_q_end - score_q_start, 1)

    offs_rows = tl.arange(0, BLOCK_ROWS)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)
    local_head = cur_head_block * BLOCK_H + offs_rows // BLOCK_M
    q_head = cur_kv_head * H_PER_KV + local_head
    q_abs_pos = (
        score_q_start
        + cur_query_block * BLOCK_M
        + (offs_rows % BLOCK_M)
    )
    q_rel_pos = q_abs_pos - prompt_cache_len
    q_row_valid = (
        (local_head < H_PER_KV)
        & (q_abs_pos < score_q_end)
        & (q_rel_pos >= 0)
        & (q_rel_pos < cur_batch_seq_len)
    )

    off_q = (
        (cur_batch_in_all_start_index + q_rel_pos[:, None]) * stride_qt
        + q_head[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + off_q, mask=q_row_valid[:, None], other=0.0)

    start_n = cur_n_block * BLOCK_N
    kv_pos = start_n + offs_n
    candidate_end = tl.maximum(candidate_start, context_len - recent_keep_tokens)
    kv_in_candidate = (kv_pos >= candidate_start) & (kv_pos < candidate_end)
    kv_loc = tl.load(
        Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * kv_pos,
        mask=kv_in_candidate,
        other=0,
    )
    off_k = kv_loc[None, :] * stride_ks + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
    k = tl.load(K + off_k, mask=kv_in_candidate[None, :], other=0.0)

    qk = tl.dot(q, k) * sm_scale
    valid = (
        q_row_valid[:, None]
        & kv_in_candidate[None, :]
        & (q_abs_pos[:, None] >= kv_pos[None, :])
    )
    qk = tl.where(valid, qk, -1.0e20)

    stats_offs = storage_group * BLOCK_ROWS + offs_rows
    lse_i = tl.load(Global_LSE + stats_offs)
    probs = tl.exp(qk - lse_i[:, None])
    probs = tl.where(valid, probs, 0.0)

    # Reduce observation queries within each head before combining heads.
    head_scores = tl.sum(
        tl.reshape(probs, (BLOCK_H, BLOCK_M, BLOCK_N)), axis=1
    ) / (score_q_len * 1.0)
    if WRITE_PER_HEAD:
        output_local_head = cur_head_block * BLOCK_H + tl.arange(0, BLOCK_H)
        output_head = cur_kv_head * H_PER_KV + output_local_head
        tl.atomic_add(
            Head_Score
            + cur_batch * stride_hsb
            + output_head[:, None] * stride_hsh
            + kv_pos[None, :] * stride_hsl,
            head_scores,
            mask=kv_in_candidate[None, :] & (output_local_head[:, None] < H_PER_KV),
        )
    else:
        tl.atomic_max(
            Attn_Score + cur_batch * stride_asb + kv_pos * stride_asl,
            tl.max(head_scores, axis=0),
            mask=kv_in_candidate,
        )


@triton.jit(do_not_specialize=["SCORE_WIDTH", "QUERY_BLOCKS"])
def _prefill_probability_from_lse_kernel(
    Q,
    K,
    Attention_LSE,
    Attn_Score,
    Head_Score,
    B_Seqlen,
    Req_to_tokens,
    B_req_idx,
    Score_Q_Start,
    Score_Q_End,
    B_Start_Loc,
    B_Prompt_Cache_Len,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_lseh,
    stride_lseq,
    stride_asb,
    stride_asl,
    stride_hsb,
    stride_hsh,
    stride_hsl,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    H_PER_KV: tl.constexpr,
    H_KV: tl.constexpr,
    HEAD_BLOCKS: tl.constexpr,
    QUERY_BLOCKS,
    WRITE_PER_HEAD: tl.constexpr,
    SCORE_WIDTH,
    FULL_K_TILES: tl.constexpr,
    sm_scale: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    storage_group = tl.program_id(0)
    cur_n_block = tl.program_id(1)
    cur_query_block = storage_group % QUERY_BLOCKS
    cur_group = storage_group // QUERY_BLOCKS
    cur_head_block = cur_group % HEAD_BLOCKS
    cur_bkv = cur_group // HEAD_BLOCKS
    cur_batch = cur_bkv // H_KV
    cur_kv_head = cur_bkv % H_KV
    q_start = tl.load(B_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(B_Prompt_Cache_Len + cur_batch)
    context_len = tl.load(B_Seqlen + cur_batch)
    req_idx = tl.load(B_req_idx + cur_batch)
    score_q_start = tl.load(Score_Q_Start + cur_batch)
    score_q_end = tl.load(Score_Q_End + cur_batch)
    offs_rows = tl.arange(0, BLOCK_ROWS)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)
    local_head = cur_head_block * BLOCK_H + offs_rows // BLOCK_M
    q_head = cur_kv_head * H_PER_KV + local_head
    kv_pos = cur_n_block * BLOCK_N + offs_n
    kv_in_output = (
        tl.full((BLOCK_N,), True, tl.int1) if FULL_K_TILES else kv_pos < SCORE_WIDTH
    )
    kv_valid = kv_pos < context_len
    kv_slot = tl.load(
        Req_to_tokens
        + req_idx * stride_req_to_tokens_b
        + kv_pos * stride_req_to_tokens_s,
        mask=kv_valid,
        other=0,
    )
    k_values = tl.load(
        K
        + kv_slot[None, :] * stride_ks
        + cur_kv_head * stride_kh
        + offs_d[:, None] * stride_kd,
        mask=kv_valid[None, :],
        other=0.0,
    )
    score_q_len = tl.maximum(score_q_end - score_q_start, 1)
    q_abs_pos = (
        score_q_start
        + cur_query_block * BLOCK_M
        + (offs_rows % BLOCK_M)
    )
    q_rel_pos = q_abs_pos - prompt_cache_len
    q_valid = (
        (local_head < H_PER_KV)
        & (q_abs_pos < score_q_end)
        & (q_rel_pos >= 0)
    )
    q_token = q_start + q_rel_pos
    q_values = tl.load(
        Q
        + q_token[:, None] * stride_qt
        + q_head[:, None] * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=q_valid[:, None],
        other=0.0,
    )
    row_lse = tl.load(
        Attention_LSE + q_head * stride_lseh + q_token * stride_lseq,
        mask=q_valid,
        other=0.0,
    )
    logits = tl.dot(q_values, k_values) * sm_scale
    valid = q_valid[:, None] & kv_valid[None, :] & (
        q_abs_pos[:, None] >= kv_pos[None, :]
    )
    probabilities = tl.where(
        valid,
        tl.exp(logits - row_lse[:, None]),
        0.0,
    )

    # Reduce observation queries within each head before combining heads.
    head_scores = tl.sum(
        tl.reshape(probabilities, (BLOCK_H, BLOCK_M, BLOCK_N)), axis=1
    ) / (score_q_len * 1.0)
    if WRITE_PER_HEAD:
        output_local_head = cur_head_block * BLOCK_H + tl.arange(0, BLOCK_H)
        output_head = cur_kv_head * H_PER_KV + output_local_head
        tl.atomic_add(
            Head_Score
            + cur_batch * stride_hsb
            + output_head[:, None] * stride_hsh
            + kv_pos[None, :] * stride_hsl,
            head_scores,
            mask=kv_in_output[None, :] & (output_local_head[:, None] < H_PER_KV),
        )
    else:
        tl.atomic_max(
            Attn_Score + cur_batch * stride_asb + kv_pos * stride_asl,
            tl.max(head_scores, axis=0),
            mask=kv_in_output,
        )


@triton.jit(do_not_specialize=["SCORE_WIDTH"])
def _prefill_probability_head_reduce_kernel(
    Head_Score,
    Attn_Score,
    stride_hsb,
    stride_hsh,
    stride_hsl,
    stride_asb,
    stride_asl,
    QUERY_HEADS: tl.constexpr,
    REDUCE_HEADS: tl.constexpr,
    SCORE_WIDTH,
    FULL_K_TILES: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    kv_pos = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    heads = tl.arange(0, REDUCE_HEADS)
    scores = tl.load(
        Head_Score
        + cur_batch * stride_hsb
        + heads[:, None] * stride_hsh
        + kv_pos[None, :] * stride_hsl,
        mask=(heads[:, None] < QUERY_HEADS)
        & (FULL_K_TILES | (kv_pos[None, :] < SCORE_WIDTH)),
        other=0.0,
    )
    tl.store(
        Attn_Score + cur_batch * stride_asb + kv_pos * stride_asl,
        tl.max(scores, axis=0),
        mask=FULL_K_TILES | (kv_pos < SCORE_WIDTH),
    )


# Logits callers use configured sink/recent budgets; variable prefix-prune
# intervals use the probability kernels above. Keep these fixed bounds specialized.
@triton.jit(do_not_specialize=["QUERY_BLOCKS"])
def _prefill_logit_score_kernel(
    Q,
    K,
    Attn_Score,
    B_Seqlen,
    Req_to_tokens,
    B_req_idx,
    Batch_Indices,
    Score_Q_Start,
    Score_Q_End,
    B_Start_Loc,
    B_Prompt_Cache_Len,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_asb,
    stride_asl,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    H_PER_KV: tl.constexpr,
    H_KV: tl.constexpr,
    HEAD_BLOCKS: tl.constexpr,
    QUERY_BLOCKS,
    USE_BATCH_INDICES: tl.constexpr,
    candidate_start: tl.constexpr,
    recent_keep_tokens: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_group = tl.program_id(0)
    cur_n_block = tl.program_id(1)
    cur_query_block = cur_group % QUERY_BLOCKS
    cur_group = cur_group // QUERY_BLOCKS
    cur_head_block = cur_group % HEAD_BLOCKS
    cur_bkv = cur_group // HEAD_BLOCKS
    cur_batch = cur_bkv // H_KV
    cur_kv_head = cur_bkv % H_KV
    source_batch = (
        tl.load(Batch_Indices + cur_batch) if USE_BATCH_INDICES else cur_batch
    )

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + source_batch)
    prompt_cache_len = tl.load(B_Prompt_Cache_Len + source_batch)
    context_len = tl.load(B_Seqlen + source_batch)
    cur_batch_seq_len = context_len - prompt_cache_len
    cur_batch_req_idx = tl.load(B_req_idx + source_batch)
    score_q_start = tl.load(Score_Q_Start + cur_batch)
    score_q_end = tl.load(Score_Q_End + cur_batch)

    offs_rows = tl.arange(0, BLOCK_ROWS)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)
    local_head = cur_head_block * BLOCK_H + offs_rows // BLOCK_M
    q_head = cur_kv_head * H_PER_KV + local_head
    q_abs_pos = (
        score_q_start
        + cur_query_block * BLOCK_M
        + (offs_rows % BLOCK_M)
    )
    q_rel_pos = q_abs_pos - prompt_cache_len
    q_row_valid = (
        (local_head < H_PER_KV)
        & (q_abs_pos < score_q_end)
        & (q_rel_pos >= 0)
        & (q_rel_pos < cur_batch_seq_len)
    )

    off_q = (
        (cur_batch_in_all_start_index + q_rel_pos[:, None]) * stride_qt
        + q_head[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + off_q, mask=q_row_valid[:, None], other=0.0)

    kv_pos = cur_n_block * BLOCK_N + offs_n
    candidate_end = tl.maximum(candidate_start, context_len - recent_keep_tokens)
    kv_in_candidate = (kv_pos >= candidate_start) & (kv_pos < candidate_end)
    kv_loc = tl.load(
        Req_to_tokens
        + stride_req_to_tokens_b * cur_batch_req_idx
        + stride_req_to_tokens_s * kv_pos,
        mask=kv_in_candidate,
        other=0,
    )
    off_k = (
        kv_loc[None, :] * stride_ks
        + cur_kv_head * stride_kh
        + offs_d[:, None] * stride_kd
    )
    k = tl.load(K + off_k, mask=kv_in_candidate[None, :], other=0.0)

    qk = tl.dot(q, k)
    valid = (
        q_row_valid[:, None]
        & kv_in_candidate[None, :]
        & (q_abs_pos[:, None] >= kv_pos[None, :])
    )
    token_score = tl.max(tl.where(valid, qk, -1.0e20), axis=0)
    tl.atomic_max(
        Attn_Score + cur_batch * stride_asb + kv_pos * stride_asl,
        token_score,
        mask=kv_in_candidate,
    )


@torch.no_grad()
def prefill_score_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    attn_score: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_prompt_cache_len: torch.Tensor,
    max_query_len: int,
    req_to_token_indexs: torch.Tensor,
    score_q_start: torch.Tensor,
    score_q_end: torch.Tensor,
    *,
    candidate_start: int = 0,
    recent_keep_tokens: int = 0,
    score_mode: str = "probability",
    workspace: PrefillScoreWorkspace | None = None,
    batch_indices: torch.Tensor | None = None,
):
    head_dim = q.shape[-1]
    assert k.shape[-1] == head_dim
    assert q.dtype == k.dtype
    assert q.stride(-1) == 1 and k.stride(-1) == 1
    assert attn_score.dim() == 2
    assert head_dim in {16, 32, 64, 128, 256}
    batch, head = score_q_start.shape[0], q.shape[1]
    if score_q_end.shape != score_q_start.shape:
        raise ValueError(
            "score_q_start and score_q_end must have the same shape, got "
            f"{tuple(score_q_start.shape)} and {tuple(score_q_end.shape)}."
        )
    if int(attn_score.shape[0]) != int(batch):
        raise ValueError(
            "attn_score must have one row per score range, got "
            f"score_batch={batch} output_shape={tuple(attn_score.shape)}."
        )
    if batch_indices is not None:
        if batch_indices.shape != score_q_start.shape:
            raise ValueError(
                "batch_indices must have one entry per score range, got "
                f"{tuple(batch_indices.shape)} and {tuple(score_q_start.shape)}."
            )
        if batch_indices.dtype != torch.int32 or batch_indices.device != q.device:
            raise TypeError(
                "batch_indices must be int32 on the query device, got "
                f"{batch_indices.dtype} on {batch_indices.device}."
            )
    launch_batch_indices = b_req_idx if batch_indices is None else batch_indices
    kv_head = k.shape[1]
    kv_group_num = head // kv_head
    if kv_group_num <= 0 or head % kv_head != 0:
        raise ValueError(f"num query heads must be divisible by num kv heads: q={head} k={kv_head}")
    score_mode = str(score_mode).strip().lower()
    if score_mode not in {"probability", "logits"}:
        raise ValueError(
            "prefill score_mode must be 'probability' or 'logits', "
            f"got {score_mode!r}."
        )
    max_score_len = int(max_query_len)
    if max_score_len <= 0:
        return
    if score_mode == "logits":
        block_m = min(32, max(16, triton.next_power_of_2(max_score_len)))
        query_blocks = triton.cdiv(max_score_len, block_m)
    elif max_score_len <= 128:
        block_m = max(16, triton.next_power_of_2(max_score_len))
        query_blocks = 1
    else:
        block_m = min(32, max(16, triton.next_power_of_2(max_score_len)))
        query_blocks = triton.cdiv(max_score_len, block_m)

    max_candidate_end = int(attn_score.shape[1])
    if max_candidate_end <= 0:
        return

    block_n = 64 if head_dim >= 128 else 128
    candidate_blocks = triton.cdiv(max_candidate_end, block_n)
    if candidate_blocks <= 0:
        return

    # Keep the dot tile bounded. Common GQA (7 heads per KV, W=32) fits in one
    # head block; larger query windows or MQA split heads across multiple blocks.
    max_rows = 256
    block_h_limit = max(1, min(8, max_rows // block_m))
    block_h = min(triton.next_power_of_2(kv_group_num), block_h_limit)
    head_blocks = triton.cdiv(kv_group_num, block_h)
    block_rows = block_h * block_m
    base_group_count = batch * kv_head * head_blocks
    group_count = base_group_count * query_blocks
    if group_count <= 0:
        return

    attn_score.fill_(-torch.inf if score_mode == "logits" else 0.0)
    dot_warps = 8 if block_rows >= 128 or block_n >= 128 else 4
    if score_mode == "logits":
        _prefill_logit_score_kernel[(group_count, candidate_blocks)](
            q,
            k,
            attn_score,
            b_seq_len,
            req_to_token_indexs,
            b_req_idx,
            launch_batch_indices,
            score_q_start,
            score_q_end,
            b_start_loc,
            b_prompt_cache_len,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            attn_score.stride(0),
            attn_score.stride(1),
            req_to_token_indexs.stride(0),
            req_to_token_indexs.stride(1),
            H_PER_KV=kv_group_num,
            H_KV=kv_head,
            HEAD_BLOCKS=head_blocks,
            QUERY_BLOCKS=query_blocks,
            USE_BATCH_INDICES=batch_indices is not None,
            candidate_start=int(candidate_start),
            recent_keep_tokens=int(recent_keep_tokens),
            BLOCK_H=block_h,
            BLOCK_ROWS=block_rows,
            BLOCK_DMODEL=head_dim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=dot_warps,
            num_stages=3,
        )
        return

    reduce_blocks = triton.next_power_of_2(candidate_blocks)
    reduce_rows = 16
    while reduce_rows > 1 and reduce_rows * reduce_blocks > 32768:
        reduce_rows //= 2

    workspace = PrefillScoreWorkspace() if workspace is None else workspace
    write_per_head = query_blocks > 1
    if write_per_head:
        head_score = workspace.probability_head_score_buffer(
            batch_size=batch,
            query_heads=head,
            score_width=max_candidate_end,
            device=q.device,
        )
        head_score.zero_()
        head_score_strides = head_score.stride()
    else:
        head_score = attn_score
        head_score_strides = (
            attn_score.stride(0),
            0,
            attn_score.stride(1),
        )
    partial_lse, global_lse = workspace.probability_lse_buffers(
        group_count=group_count,
        candidate_blocks=candidate_blocks,
        block_rows=block_rows,
        device=q.device,
    )
    _prefill_score_partial_lse_kernel[(group_count, candidate_blocks)](
        q,
        k,
        partial_lse,
        b_seq_len,
        req_to_token_indexs,
        b_req_idx,
        launch_batch_indices,
        score_q_start,
        score_q_end,
        b_start_loc,
        b_prompt_cache_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        H_PER_KV=kv_group_num,
        H_KV=kv_head,
        HEAD_BLOCKS=head_blocks,
        QUERY_BLOCKS=query_blocks,
        USE_BATCH_INDICES=batch_indices is not None,
        candidate_start=int(candidate_start),
        recent_keep_tokens=int(recent_keep_tokens),
        sm_scale=float(head_dim) ** -0.5,
        NUM_BLOCKS=candidate_blocks,
        BLOCK_H=block_h,
        BLOCK_ROWS=block_rows,
        BLOCK_DMODEL=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=dot_warps,
        num_stages=3,
    )
    reduce_grid = (group_count, triton.cdiv(block_rows, reduce_rows))
    _prefill_score_reduce_lse_kernel[reduce_grid](
        partial_lse,
        global_lse,
        NUM_BLOCKS=candidate_blocks,
        BLOCK_ROWS=block_rows,
        REDUCE_BLOCKS=reduce_blocks,
        REDUCE_ROWS=reduce_rows,
        num_warps=8 if reduce_blocks >= 1024 else 4,
        num_stages=4,
    )
    _prefill_score_final_kernel[(group_count, candidate_blocks)](
        q,
        k,
        attn_score,
        head_score,
        global_lse,
        b_seq_len,
        req_to_token_indexs,
        b_req_idx,
        launch_batch_indices,
        score_q_start,
        score_q_end,
        b_start_loc,
        b_prompt_cache_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        attn_score.stride(0),
        attn_score.stride(1),
        *head_score_strides,
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        H_PER_KV=kv_group_num,
        H_KV=kv_head,
        HEAD_BLOCKS=head_blocks,
        QUERY_BLOCKS=query_blocks,
        WRITE_PER_HEAD=write_per_head,
        USE_BATCH_INDICES=batch_indices is not None,
        candidate_start=int(candidate_start),
        recent_keep_tokens=int(recent_keep_tokens),
        sm_scale=float(head_dim) ** -0.5,
        BLOCK_H=block_h,
        BLOCK_ROWS=block_rows,
        BLOCK_DMODEL=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=dot_warps,
        num_stages=3,
    )
    if write_per_head:
        reduce_heads = triton.next_power_of_2(head)
        _prefill_probability_head_reduce_kernel[(batch, candidate_blocks)](
            head_score,
            attn_score,
            *head_score.stride(),
            *attn_score.stride(),
            QUERY_HEADS=head,
            REDUCE_HEADS=reduce_heads,
            SCORE_WIDTH=max_candidate_end,
            FULL_K_TILES=max_candidate_end % block_n == 0,
            BLOCK_N=block_n,
            num_warps=8 if reduce_heads >= 64 else 4,
            num_stages=3,
        )


@torch.no_grad()
def prefill_score_from_lse_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    attention_lse: torch.Tensor,
    attn_score: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_prompt_cache_len: torch.Tensor,
    max_query_len: int,
    req_to_token_indexs: torch.Tensor,
    score_q_start: torch.Tensor,
    score_q_end: torch.Tensor,
    *,
    workspace: PrefillScoreWorkspace | None = None,
) -> None:
    """Reduce exact FA3 probabilities into one token vector per layer row."""

    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(
            "FA3 probability scoring requires Q/K with shapes [T,H,D]/[S,Hkv,D], "
            f"got {tuple(q.shape)} and {tuple(k.shape)}."
        )
    if q.dtype != k.dtype or q.stride(-1) != 1 or k.stride(-1) != 1:
        raise TypeError("FA3 probability scoring requires matching contiguous Q/K dtypes.")
    if attention_lse.ndim != 2 or tuple(attention_lse.shape) != (
        int(q.shape[1]),
        int(q.shape[0]),
    ):
        raise ValueError(
            "FA3 LSE must have shape [query_heads, total_query_tokens], got "
            f"{tuple(attention_lse.shape)} for Q={tuple(q.shape)}."
        )
    if attention_lse.dtype != torch.float32 or attention_lse.device != q.device:
        raise TypeError(
            f"FA3 LSE must be FP32 on {q.device}, got {attention_lse.dtype} "
            f"on {attention_lse.device}."
        )
    batch = int(score_q_start.numel())
    if (
        score_q_end.shape != score_q_start.shape
        or b_req_idx.numel() != batch
        or b_start_loc.numel() != batch
        or b_seq_len.numel() != batch
        or b_prompt_cache_len.numel() != batch
    ):
        raise ValueError("FA3 probability score metadata must have one entry per batch row.")
    if attn_score.ndim != 2 or int(attn_score.shape[0]) != batch:
        raise ValueError(
            "FA3 probability score output must be [batch, context], got "
            f"{tuple(attn_score.shape)}."
        )
    max_score_len = int(max_query_len)
    if max_score_len <= 0 or batch == 0:
        return

    query_heads = int(q.shape[1])
    kv_heads = int(k.shape[1])
    head_dim = int(q.shape[2])
    if query_heads % kv_heads:
        raise ValueError(
            f"Query heads must be divisible by KV heads: {query_heads}/{kv_heads}."
        )
    heads_per_kv = query_heads // kv_heads
    if max_score_len <= 128:
        block_m = max(16, triton.next_power_of_2(max_score_len))
        query_blocks = 1
    else:
        block_m = min(32, max(16, triton.next_power_of_2(max_score_len)))
        query_blocks = triton.cdiv(max_score_len, block_m)
    block_n = 64 if head_dim >= 128 else 128
    candidate_blocks = triton.cdiv(int(attn_score.shape[1]), block_n)
    max_rows = 256
    block_h = min(
        triton.next_power_of_2(heads_per_kv),
        max(1, min(8, max_rows // block_m)),
    )
    head_blocks = triton.cdiv(heads_per_kv, block_h)
    block_rows = block_h * block_m
    group_count = batch * kv_heads * head_blocks * query_blocks
    workspace = PrefillScoreWorkspace() if workspace is None else workspace
    write_per_head = query_blocks > 1
    if write_per_head:
        head_score = workspace.probability_head_score_buffer(
            batch_size=batch,
            query_heads=query_heads,
            score_width=int(attn_score.shape[1]),
            device=q.device,
        )
        head_score.zero_()
        head_score_strides = head_score.stride()
    else:
        head_score = attn_score
        head_score_strides = (
            attn_score.stride(0),
            0,
            attn_score.stride(1),
        )
    attn_score.zero_()
    _prefill_probability_from_lse_kernel[(group_count, candidate_blocks)](
        q,
        k,
        attention_lse,
        attn_score,
        head_score,
        b_seq_len,
        req_to_token_indexs,
        b_req_idx,
        score_q_start,
        score_q_end,
        b_start_loc,
        b_prompt_cache_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        attention_lse.stride(0),
        attention_lse.stride(1),
        attn_score.stride(0),
        attn_score.stride(1),
        *head_score_strides,
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        H_PER_KV=heads_per_kv,
        H_KV=kv_heads,
        HEAD_BLOCKS=head_blocks,
        QUERY_BLOCKS=query_blocks,
        WRITE_PER_HEAD=write_per_head,
        SCORE_WIDTH=int(attn_score.shape[1]),
        FULL_K_TILES=int(attn_score.shape[1]) % block_n == 0,
        sm_scale=float(head_dim) ** -0.5,
        BLOCK_H=block_h,
        BLOCK_ROWS=block_rows,
        BLOCK_DMODEL=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8 if block_rows >= 128 else 4,
        num_stages=3,
    )
    if write_per_head:
        reduce_heads = triton.next_power_of_2(query_heads)
        _prefill_probability_head_reduce_kernel[(batch, candidate_blocks)](
            head_score,
            attn_score,
            *head_score.stride(),
            *attn_score.stride(),
            QUERY_HEADS=query_heads,
            REDUCE_HEADS=reduce_heads,
            SCORE_WIDTH=int(attn_score.shape[1]),
            FULL_K_TILES=int(attn_score.shape[1]) % block_n == 0,
            BLOCK_N=block_n,
            num_warps=8 if reduce_heads >= 64 else 4,
            num_stages=3,
        )
