"""TileLang GLM MLA decode kernel.

Adapted from Tile-AI/TileLang at commit
``c7fabc4cc65e480b88b7606eb1bc9c340dbd8c8c``. The local adaptation uses
BF16, page-size-one indirect cache slots, direct strided GLM queries with
internal zero padding to complete MMA tiles, caller-owned outputs/workspaces,
and CUDA Graph capture.
See ``LICENSE.tilelang`` and ``README.md`` in this package.
"""

import tilelang
import tilelang.language as T


@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def build_glm_mla_decode_kernel(
    batch,
    h_q,
    h_kv,
    valid_output_heads,
    dv,
    dpe,
    block_N,
    block_H,
    num_split,
    block_size,
    q_latent_strides,
    q_rope_strides,
    softmax_scale=None,
    need_score=False,
    score_mode="direct",
):
    if softmax_scale is None:
        softmax_scale = (dv + dpe) ** -0.5
    scale = float(softmax_scale * 1.44269504)  # log2(e)
    dtype = T.bfloat16
    accum_dtype = T.float32
    kv_group_num = h_q // h_kv
    VALID_BLOCK_H = min(block_H, kv_group_num)
    VALID_OUTPUT_HEADS = valid_output_heads
    HEAD_TILE_COUNT = h_q // VALID_BLOCK_H
    SCORE_TILE_COUNT = (
        VALID_OUTPUT_HEADS
        if score_mode == "per_head"
        else HEAD_TILE_COUNT
        if score_mode == "partial"
        else 1
    )
    assert h_kv == 1, "h_kv must be 1"
    assert h_q % VALID_BLOCK_H == 0, "h_q must use complete head tiles"
    assert 0 < VALID_OUTPUT_HEADS <= h_q, "valid output heads must fit h_q"
    assert score_mode in ("direct", "atomic", "partial", "per_head")
    assert not need_score or score_mode != "direct" or HEAD_TILE_COUNT == 1
    assert block_size >= block_N and block_size % block_N == 0, (
        "block_size must be at least block_N and a multiple of block_N"
    )

    cache_slots = T.dynamic("cache_slots")
    slot_rows = T.dynamic("slot_rows")
    active_slot_width = T.dynamic("active_slot_width")
    max_seqlen_pad = T.dynamic("score_capacity")
    score_strides = [T.dynamic("score_stride_b"), T.dynamic("score_stride_h"), T.dynamic("score_stride_t")]

    @T.prim_func
    def main_split(
        Q: T.StridedTensor(
            [batch, VALID_OUTPUT_HEADS, dv], q_latent_strides, dtype
        ),
        Q_pe: T.StridedTensor(
            [batch, VALID_OUTPUT_HEADS, dpe], q_rope_strides, dtype
        ),
        KV: T.Tensor([cache_slots, h_kv, dv], dtype),
        K_pe: T.Tensor([cache_slots, h_kv, dpe], dtype),
        active_slots: T.Tensor([slot_rows, active_slot_width], T.int32),
        request_indices: T.Tensor([batch], T.int32),
        cache_seqlens: T.Tensor([batch], T.int32),
        glse: T.Tensor([batch, h_q, num_split], dtype),
        Output_partial: T.Tensor([batch, h_q, num_split, dv], dtype),
        Output: T.Tensor([batch, VALID_OUTPUT_HEADS, dv], dtype),
        AttnScore: T.StridedTensor(
            [batch, SCORE_TILE_COUNT, max_seqlen_pad], score_strides, accum_dtype
        ),
    ):
        # split kv
        with T.Kernel(batch, h_q // min(block_H, kv_group_num), num_split, threads=256) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_H, dv], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, dpe], dtype)
            KV_shared = T.alloc_shared([block_N, dv], dtype)
            K_pe_shared = T.alloc_shared([block_N, dpe], dtype)
            O_shared = T.alloc_shared([block_H, dv], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
            acc_o = T.alloc_fragment([block_H, dv], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            token_scores = T.alloc_fragment([block_N], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            cur_kv_head = 0
            request_row = T.max(request_indices[bx], 0)
            if HEAD_TILE_COUNT == 1:
                T.use_swizzle(10)

            for i, j in T.Parallel(block_H, dv):
                global_head = by * VALID_BLOCK_H + i
                Q_shared[i, j] = T.if_then_else(
                    global_head < VALID_OUTPUT_HEADS,
                    Q[bx, global_head, j],
                    0,
                )
            for i, j in T.Parallel(block_H, dpe):
                global_head = by * VALID_BLOCK_H + i
                Q_pe_shared[i, j] = T.if_then_else(
                    global_head < VALID_OUTPUT_HEADS,
                    Q_pe[bx, global_head, j],
                    0,
                )
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            total_blocks = T.ceildiv(cache_seqlens[bx], block_N)
            blocks_per_split = T.floordiv(total_blocks, num_split)
            remaining_blocks = T.floormod(total_blocks, num_split)
            loop_range = blocks_per_split + T.if_then_else(bz < remaining_blocks, 1, 0)
            start = (blocks_per_split * bz + T.min(bz, remaining_blocks)) * block_N

            for k in T.Pipelined(loop_range, num_stages=2):
                for i, j in T.Parallel(block_N, dv):
                    token_index = start + k * block_N + i
                    slot = T.if_then_else(
                        token_index < cache_seqlens[bx],
                        active_slots[request_row, token_index],
                        0,
                    )
                    KV_shared[i, j] = KV[slot, cur_kv_head, j]
                for i, j in T.Parallel(block_N, dpe):
                    token_index = start + k * block_N + i
                    slot = T.if_then_else(
                        token_index < cache_seqlens[bx],
                        active_slots[request_row, token_index],
                        0,
                    )
                    K_pe_shared[i, j] = K_pe[slot, cur_kv_head, j]
                T.clear(acc_s)
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(start + k * block_N + j >= cache_seqlens[bx], -T.infinity(accum_dtype), acc_s[i, j])
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(by * VALID_BLOCK_H + i >= VALID_OUTPUT_HEADS, -T.infinity(accum_dtype), acc_s[i, j])
                if need_score:
                    if score_mode == "per_head":
                        for i, j in T.Parallel(block_H, block_N):
                            score_index = start + k * block_N + j
                            global_head = by * VALID_BLOCK_H + i
                            if score_index < cache_seqlens[bx]:
                                if global_head < VALID_OUTPUT_HEADS:
                                    AttnScore[bx, global_head, score_index] = acc_s[i, j]
                    else:
                        T.reduce_max(acc_s, token_scores, dim=0)
                    if score_mode == "direct":
                        for j in T.Parallel(block_N):
                            score_index = start + k * block_N + j
                            if score_index < cache_seqlens[bx]:
                                AttnScore[bx, 0, score_index] = token_scores[j]
                    elif score_mode == "atomic":
                        for j in T.Parallel(block_N):
                            score_index = start + k * block_N + j
                            if score_index < cache_seqlens[bx]:
                                T.atomic_max(
                                    AttnScore[bx, 0, score_index],
                                    token_scores[j],
                                )
                    elif score_mode == "partial":
                        for j in T.Parallel(block_N):
                            score_index = start + k * block_N + j
                            if score_index < cache_seqlens[bx]:
                                AttnScore[bx, by, score_index] = token_scores[j]
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_shared)
                T.copy(S_shared, acc_s_cast)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_H, dv):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(acc_s_cast, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
            for i, j in T.Parallel(block_H, dv):
                acc_o[i, j] = T.if_then_else(
                    loop_range > 0,
                    acc_o[i, j] / logsum[i],
                    0.0,
                )
            for i in T.Parallel(block_H):
                logsum[i] = T.if_then_else(
                    loop_range > 0,
                    T.log2(logsum[i]) + scores_max[i] * scale,
                    -T.infinity(accum_dtype),
                )
            T.copy(logsum, glse[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, bz])
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output_partial[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, bz, :])

        # combine
        with T.Kernel(VALID_OUTPUT_HEADS, batch, threads=128) as (by, bz):
            po_local = T.alloc_fragment([dv], dtype)
            o_accum_local = T.alloc_fragment([dv], accum_dtype)
            lse_local_split = T.alloc_var(accum_dtype)
            lse_logsum_local = T.alloc_var(accum_dtype)
            lse_max_local = T.alloc_var(accum_dtype)
            scale_local = T.alloc_var(accum_dtype)

            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            lse_max_local = -T.infinity(accum_dtype)
            for k in T.serial(num_split):
                lse_max_local = T.max(lse_max_local, glse[bz, by, k])
            for k in T.Pipelined(num_split, num_stages=1):
                lse_local_split = glse[bz, by, k]
                lse_logsum_local += T.exp2(lse_local_split - lse_max_local)
            lse_logsum_local = T.log2(lse_logsum_local) + lse_max_local
            for k in T.serial(num_split):
                for i in T.Parallel(dv):
                    po_local[i] = Output_partial[bz, by, k, i]
                lse_local_split = glse[bz, by, k]
                scale_local = T.exp2(lse_local_split - lse_logsum_local)
                for i in T.Parallel(dv):
                    o_accum_local[i] += po_local[i] * scale_local
            for i in T.Parallel(dv):
                Output[bz, by, i] = T.if_then_else(
                    cache_seqlens[bz] > 0,
                    o_accum_local[i],
                    0.0,
                )

    @T.prim_func
    def main_no_split(
        Q: T.StridedTensor(
            [batch, VALID_OUTPUT_HEADS, dv], q_latent_strides, dtype
        ),
        Q_pe: T.StridedTensor(
            [batch, VALID_OUTPUT_HEADS, dpe], q_rope_strides, dtype
        ),
        KV: T.Tensor([cache_slots, h_kv, dv], dtype),
        K_pe: T.Tensor([cache_slots, h_kv, dpe], dtype),
        active_slots: T.Tensor([slot_rows, active_slot_width], T.int32),
        request_indices: T.Tensor([batch], T.int32),
        cache_seqlens: T.Tensor([batch], T.int32),
        glse: T.Tensor([batch, h_q, num_split], dtype),
        Output_partial: T.Tensor([batch, h_q, num_split, dv], dtype),
        Output: T.Tensor([batch, VALID_OUTPUT_HEADS, dv], dtype),
        AttnScore: T.StridedTensor(
            [batch, SCORE_TILE_COUNT, max_seqlen_pad], score_strides, accum_dtype
        ),
    ):
        with T.Kernel(batch, h_q // min(block_H, kv_group_num), threads=256) as (bx, by):
            Q_shared = T.alloc_shared([block_H, dv], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, dpe], dtype)
            KV_shared = T.alloc_shared([block_N, dv], dtype)
            K_pe_shared = T.alloc_shared([block_N, dpe], dtype)
            O_shared = T.alloc_shared([block_H, dv], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_o = T.alloc_fragment([block_H, dv], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            token_scores = T.alloc_fragment([block_N], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            cur_kv_head = 0
            request_row = T.max(request_indices[bx], 0)
            if HEAD_TILE_COUNT == 1:
                T.use_swizzle(10)

            for i, j in T.Parallel(block_H, dv):
                global_head = by * VALID_BLOCK_H + i
                Q_shared[i, j] = T.if_then_else(
                    global_head < VALID_OUTPUT_HEADS,
                    Q[bx, global_head, j],
                    0,
                )
            for i, j in T.Parallel(block_H, dpe):
                global_head = by * VALID_BLOCK_H + i
                Q_pe_shared[i, j] = T.if_then_else(
                    global_head < VALID_OUTPUT_HEADS,
                    Q_pe[bx, global_head, j],
                    0,
                )
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(cache_seqlens[bx], block_N)
            for kr in T.Pipelined(loop_range, num_stages=2):
                k = loop_range - 1 - kr
                for i, j in T.Parallel(block_N, dv):
                    token_index = k * block_N + i
                    slot = T.if_then_else(
                        token_index < cache_seqlens[bx],
                        active_slots[request_row, token_index],
                        0,
                    )
                    KV_shared[i, j] = KV[slot, cur_kv_head, j]
                for i, j in T.Parallel(block_N, dpe):
                    token_index = k * block_N + i
                    slot = T.if_then_else(
                        token_index < cache_seqlens[bx],
                        active_slots[request_row, token_index],
                        0,
                    )
                    K_pe_shared[i, j] = K_pe[slot, cur_kv_head, j]
                T.clear(acc_s)
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                if kr == 0:
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else(k * block_N + j >= cache_seqlens[bx], -T.infinity(accum_dtype), acc_s[i, j])
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(by * VALID_BLOCK_H + i >= VALID_OUTPUT_HEADS, -T.infinity(accum_dtype), acc_s[i, j])
                if need_score:
                    if score_mode == "per_head":
                        for i, j in T.Parallel(block_H, block_N):
                            score_index = k * block_N + j
                            global_head = by * VALID_BLOCK_H + i
                            if score_index < cache_seqlens[bx]:
                                if global_head < VALID_OUTPUT_HEADS:
                                    AttnScore[bx, global_head, score_index] = acc_s[i, j]
                    else:
                        T.reduce_max(acc_s, token_scores, dim=0)
                    if score_mode == "direct":
                        for j in T.Parallel(block_N):
                            score_index = k * block_N + j
                            if score_index < cache_seqlens[bx]:
                                AttnScore[bx, 0, score_index] = token_scores[j]
                    elif score_mode == "atomic":
                        for j in T.Parallel(block_N):
                            score_index = k * block_N + j
                            if score_index < cache_seqlens[bx]:
                                T.atomic_max(
                                    AttnScore[bx, 0, score_index],
                                    token_scores[j],
                                )
                    elif score_mode == "partial":
                        for j in T.Parallel(block_N):
                            score_index = k * block_N + j
                            if score_index < cache_seqlens[bx]:
                                AttnScore[bx, by, score_index] = token_scores[j]
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_shared)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_H, dv):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
            for i, j in T.Parallel(block_H, dv):
                acc_o[i, j] = T.if_then_else(
                    cache_seqlens[bx] > 0,
                    acc_o[i, j] / logsum[i],
                    0.0,
                )
            T.copy(acc_o, O_shared)
            for i, j in T.Parallel(block_H, dv):
                global_head = by * VALID_BLOCK_H + i
                if global_head < VALID_OUTPUT_HEADS:
                    Output[bx, global_head, j] = T.if_then_else(
                        cache_seqlens[bx] > 0,
                        O_shared[i, j],
                        0.0,
                    )

    if num_split > 1:
        return main_split
    else:
        return main_no_split
