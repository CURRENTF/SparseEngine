# SPDX-License-Identifier: Apache-2.0
"""Portable TileLang GQA prefill and attention-score extraction kernels."""

from typing import Any

import torch

_KERNEL_CACHE: dict[tuple, Any] = {}


def _get_heads_per_group(gqa_ratio: int) -> int:
    """Find the largest divisor of gqa_ratio <= 8."""
    for h in (8, 7, 6, 5, 4, 3, 2, 1):
        if gqa_ratio % h == 0:
            return h
    return 1


import tilelang
import tilelang.language as T


@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def _build_fused_gqa_kernel(
    h_q: int,
    h_kv: int,
    head_dim: int,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 2,
    threads: int = 128,
    softmax_scale: float | None = None,
    need_score: bool = True,
):
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5
    scale = float(softmax_scale * 1.44269504)  # log2(e)
    dtype = T.bfloat16
    accum_dtype = T.float32
    gqa_ratio = h_q // h_kv
    rows_per_group = gqa_ratio * block_M

    total_q = T.symbolic("total_q")
    cache_slots = T.symbolic("cache_slots")
    slot_rows = T.symbolic("slot_rows")
    slot_cols = T.symbolic("slot_cols")
    B = T.symbolic("B")
    B_plus_1 = T.symbolic("B_plus_1")
    score_len = T.symbolic("score_len")

    @T.prim_func
    def main_gqa(
        Q: T.Tensor([total_q, h_q, head_dim], dtype),
        K: T.Tensor([cache_slots, h_kv, head_dim], dtype),
        V: T.Tensor([cache_slots, h_kv, head_dim], dtype),
        active_slots: T.Tensor([slot_rows, slot_cols], T.int32),
        request_indices: T.Tensor([B], T.int32),
        context_lens: T.Tensor([B], T.int32),
        prompt_cache_lens: T.Tensor([B], T.int32),
        cu_seqlens_q: T.Tensor([B_plus_1], T.int32),
        Output: T.Tensor([total_q, h_q, head_dim], dtype),
        AttnScore: T.Tensor([B, score_len], accum_dtype),
        batch_size: T.int32,
        max_query_len: T.int32,
    ):
        with T.Kernel(batch_size, h_kv, T.ceildiv(max_query_len, block_M), threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([rows_per_group, head_dim], dtype)
            K_shared = T.alloc_shared([block_N, head_dim], dtype)
            V_shared = T.alloc_shared([block_N, head_dim], dtype)
            acc_s = T.alloc_fragment([rows_per_group, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([rows_per_group, block_N], dtype)
            acc_o = T.alloc_fragment([rows_per_group, head_dim], accum_dtype)
            scores_max = T.alloc_fragment([rows_per_group], accum_dtype)
            scores_max_prev = T.alloc_fragment([rows_per_group], accum_dtype)
            scores_scale = T.alloc_fragment([rows_per_group], accum_dtype)
            scores_sum = T.alloc_fragment([rows_per_group], accum_dtype)
            token_scores = T.alloc_fragment([block_N], accum_dtype)
            logsum = T.alloc_fragment([rows_per_group], accum_dtype)

            request_row = T.max(request_indices[bx], 0)
            q_start = cu_seqlens_q[bx]
            q_end = cu_seqlens_q[bx + 1]
            q_len = q_end - q_start
            ctx_len = context_lens[bx]
            p_len = prompt_cache_lens[bx]

            q_tile_start = bz * block_M
            if q_tile_start < q_len:
                # Load Q tile
                for row, d in T.Parallel(rows_per_group, head_dim):
                    q_local = row % block_M
                    q_head = by * gqa_ratio + row // block_M
                    q_idx = q_start + q_tile_start + q_local
                    Q_shared[row, d] = T.if_then_else(
                        q_tile_start + q_local < q_len,
                        Q[q_idx, q_head, d],
                        0.0,
                    )

                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                # Causal tiles never need keys beyond their last query position.
                # Bounding the loop also avoids atomic score updates from tiles
                # whose entire QK region is masked.
                visible_kv_len = T.min(
                    ctx_len,
                    p_len + q_tile_start + block_M,
                )
                num_kv_blocks = T.ceildiv(visible_kv_len, block_N)
                for k in T.Pipelined(num_kv_blocks, num_stages=num_stages):
                    for j, d in T.Parallel(block_N, head_dim):
                        kv_idx = k * block_N + j
                        slot = T.if_then_else(
                            kv_idx < ctx_len,
                            active_slots[request_row, kv_idx],
                            0,
                        )
                        K_shared[j, d] = K[slot, by, d]
                        V_shared[j, d] = V[slot, by, d]

                    T.clear(acc_s)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))

                    # Apply causal mask
                    for row, j in T.Parallel(rows_per_group, block_N):
                        q_local = row % block_M
                        q_abs = p_len + q_tile_start + q_local
                        kv_abs = k * block_N + j
                        is_valid = (q_tile_start + q_local < q_len) and (kv_abs < ctx_len) and (q_abs >= kv_abs)
                        acc_s[row, j] = T.if_then_else(is_valid, acc_s[row, j], -T.infinity(accum_dtype))

                    # Extract max-reduced raw QK scores for the experimental
                    # logits mode.
                    if need_score:
                        T.reduce_max(acc_s, token_scores, dim=0)
                        for j in T.Parallel(block_N):
                            kv_abs = k * block_N + j
                            if kv_abs < ctx_len:
                                T.atomic_max(
                                    AttnScore[bx, kv_abs],
                                    token_scores[j],
                                )

                    # Online softmax
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for row in T.Parallel(rows_per_group):
                        scores_max[row] = T.max(scores_max[row], scores_max_prev[row])
                    for row in T.Parallel(rows_per_group):
                        scores_scale[row] = T.exp2(scores_max_prev[row] * scale - scores_max[row] * scale)
                    for row, j in T.Parallel(rows_per_group, block_N):
                        acc_s[row, j] = T.exp2(acc_s[row, j] * scale - scores_max[row] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    T.copy(acc_s, acc_s_cast)
                    for row in T.Parallel(rows_per_group):
                        logsum[row] = logsum[row] * scores_scale[row] + scores_sum[row]
                    for row, d in T.Parallel(rows_per_group, head_dim):
                        acc_o[row, d] *= scores_scale[row]
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

                # Normalize output and write back
                for row, d in T.Parallel(rows_per_group, head_dim):
                    q_local = row % block_M
                    q_head = by * gqa_ratio + row // block_M
                    q_idx = q_start + q_tile_start + q_local
                    if q_tile_start + q_local < q_len:
                        Output[q_idx, q_head, d] = T.if_then_else(
                            logsum[row] > 0,
                            acc_o[row, d] / logsum[row],
                            0.0,
                        )

    return main_gqa


def _get_fused_prefill_kernel(
    h_q: int,
    h_kv: int,
    head_dim: int,
    block_M: int = 16,
    block_N: int = 64,
    num_stages: int = 2,
    threads: int = 128,
    softmax_scale: float | None = None,
    need_score: bool = True,
):
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5
    scale = float(softmax_scale * 1.44269504)  # log2(e)

    key = (
        h_q,
        h_kv,
        head_dim,
        block_M,
        block_N,
        num_stages,
        threads,
        round(scale, 6),
        need_score,
    )
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    compiled = _build_fused_gqa_kernel(
        h_q=h_q,
        h_kv=h_kv,
        head_dim=head_dim,
        block_M=block_M,
        block_N=block_N,
        num_stages=num_stages,
        threads=threads,
        softmax_scale=softmax_scale,
        need_score=need_score,
    )
    _KERNEL_CACHE[key] = compiled
    return compiled


@torch.no_grad()
def gqa_paged_prefill_attention_tilelang(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    active_slots: torch.Tensor,
    req_indices: torch.Tensor,
    context_lens: torch.Tensor,
    prompt_cache_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    output: torch.Tensor,
    attn_score: torch.Tensor | None = None,
    sm_scale: float | None = None,
    max_query_len: int | None = None,
) -> torch.Tensor:
    """Execute paged GQA prefill forward pass via TileLang."""
    total_q_tokens, h_q, head_dim = q.shape
    cache_slots, h_kv, _ = k_cache.shape
    batch_size = int(context_lens.numel())
    slot_rows, slot_table_width = int(active_slots.shape[0]), int(active_slots.shape[1])
    if cu_seqlens_q.numel() != batch_size + 1:
        raise ValueError(
            "TileLang GQA prefill requires cu_seqlens_q with batch_size + 1 entries, "
            f"got {cu_seqlens_q.numel()} for batch_size={batch_size}."
        )
    if max_query_len is None:
        query_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        max_query_len = int(query_lens.max().item()) if batch_size else 0
    else:
        max_query_len = int(max_query_len)
    if max_query_len <= 0:
        raise ValueError(f"TileLang GQA prefill requires a positive query length, got {max_query_len}.")

    need_score = attn_score is not None
    if attn_score is None:
        target_score = torch.empty((batch_size, 1), dtype=torch.float32, device=q.device)
    else:
        if attn_score.ndim != 2 or int(attn_score.shape[0]) != batch_size:
            raise ValueError(
                "TileLang GQA prefill only supports reduced 2D scores with shape "
                f"[batch, context], got {tuple(attn_score.shape)}."
            )
        if int(attn_score.shape[1]) > slot_table_width:
            raise ValueError(
                "TileLang GQA prefill score width exceeds the slot table: "
                f"score_width={int(attn_score.shape[1])} slot_width={slot_table_width}."
            )
        if attn_score.dtype != torch.float32:
            raise TypeError(
                f"TileLang GQA prefill requires FP32 score output, got {attn_score.dtype}."
            )
        if attn_score.device != q.device:
            raise ValueError(
                "TileLang GQA prefill requires score output on the Q device, got "
                f"{attn_score.device} and {q.device}."
            )
        target_score = attn_score
        target_score.fill_(-torch.inf)

    block_m = 16 if need_score else 8

    kernel = _get_fused_prefill_kernel(
        h_q=h_q,
        h_kv=h_kv,
        head_dim=head_dim,
        block_M=block_m,
        block_N=64,
        num_stages=1,
        softmax_scale=sm_scale,
        need_score=need_score,
    )
    kernel(
        q,
        k_cache,
        v_cache,
        active_slots,
        req_indices,
        context_lens,
        prompt_cache_lens,
        cu_seqlens_q,
        output,
        target_score,
        batch_size,
        max_query_len,
    )
    return output
