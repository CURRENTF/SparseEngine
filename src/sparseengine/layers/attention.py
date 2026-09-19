import os

import torch
from torch import nn

from sparseengine.engine.cache_manager import ExplicitKVPayload
from sparseengine.layers.attention_backend import TritonAttentionBackend
from sparseengine.operators.decode_attention import (
    PreparedDecodeAttentionLaunchOp,
    PreparedDecodeAttentionOp,
    get_decode_workspace,
)
from sparseengine.operators.prefill_attention import (
    PrefillAttentionRunResult,
    PreparedPrefillAttentionOp,
)
from sparseengine.utils.context import get_context

from sparseengine.engine.sparse_controller import SparseController

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        *,
        prefill_op: PreparedPrefillAttentionOp | None = None,
        decode_op: PreparedDecodeAttentionOp | None = None,
        decode_launch_op: PreparedDecodeAttentionLaunchOp | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.attention_backend = TritonAttentionBackend()
        self.full_attention_provider = None
        self.prefill_op = prefill_op
        self.decode_op = decode_op
        self.decode_launch_op = decode_launch_op

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        context = get_context()
        cache_manager = context.cache_manager
        sparse_controller: SparseController = context.sparse_controller
        layer_idx = context.now_layer_idx

        temp_slots = None
        try:
            if context.is_prefill:
                selection = sparse_controller.get_prefill_selection(layer_idx)
                cache_manager.before_prefill_layer_attention(layer_idx, selection)
                prefill_view = cache_manager.build_prefill_compute_view(
                    layer_idx,
                    k,
                    v,
                    selection,
                )
                if not isinstance(prefill_view.payload, ExplicitKVPayload):
                    raise TypeError(
                        "Attention prefill requires ExplicitKVPayload, got "
                        f"{type(prefill_view.payload).__name__}."
                    )
                prefill_meta = prefill_view.meta
                temp_slots = prefill_meta.temp_slots

                if context.cu_seqlens_q is None or context.cu_seqlens_q.numel() <= 1:
                    return torch.empty_like(q)

                b_start_loc = context.cu_seqlens_q[:-1]
                chunk_lens = context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]
                max_context_len = prefill_meta.max_context_len
                if max_context_len is not None:
                    max_input_len = int(max_context_len)
                elif torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                    max_input_len = int(prefill_meta.active_slots.shape[1])
                else:
                    max_input_len = prefill_meta.context_lens.max().item()

                fake_output = self.attention_backend.maybe_run_fake_prefill(
                    q,
                    prefill_view,
                    chunk_lens=chunk_lens,
                    max_input_len=max_input_len,
                )
                attention_lse = None
                if fake_output is not None:
                    o = fake_output
                elif self.prefill_op is None:
                    o = self.attention_backend.run_prefill(
                        q,
                        prefill_view,
                        b_start_loc=b_start_loc,
                        chunk_lens=chunk_lens,
                        max_input_len=max_input_len,
                    )
                else:
                    self.attention_backend.debug_check_prefill_bounds(
                        q,
                        prefill_view,
                        chunk_lens=chunk_lens,
                    )
                    prefill_result = self.prefill_op.run(
                        q,
                        prefill_view,
                        qo_indptr=context.cu_seqlens_q,
                        chunk_lens=chunk_lens,
                        max_context_len=max_input_len,
                        layer_idx=int(layer_idx),
                    )
                    if isinstance(prefill_result, PrefillAttentionRunResult):
                        o = prefill_result.output
                        attention_lse = prefill_result.softmax_lse
                    else:
                        o = prefill_result
                cache_manager.collect_prefill_attention_score(
                    layer_idx,
                    q,
                    prefill_view,
                    b_start_loc=b_start_loc,
                    chunk_lens=chunk_lens,
                    attention_lse=attention_lse,
                )
                cache_manager.record_prefill_query(
                    layer_idx,
                    q,
                    prefill_view,
                    b_start_loc=b_start_loc,
                    chunk_lens=chunk_lens,
                )
            else:    # decode
                batch_size = q.shape[0]
                selection = sparse_controller.get_decode_selection(
                    layer_idx,
                    q,
                )
                decode_view = cache_manager.build_decode_compute_view(
                    layer_idx,
                    q,
                    selection,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                )
                if not isinstance(decode_view.payload, ExplicitKVPayload):
                    raise TypeError(
                        "Attention decode requires ExplicitKVPayload, got "
                        f"{type(decode_view.payload).__name__}."
                    )
                decode_meta = decode_view.meta
                temp_slots = decode_meta.temp_slots

                if (
                    decode_meta.active_slots.dim() == 2
                    and os.environ.get("SENGINE_DEBUG_DECODE_BOUNDS", "0") == "1"
                    and not (
                        torch.cuda.is_available()
                        and torch.cuda.is_current_stream_capturing()
                    )
                ):
                    slot_table_len = decode_meta.slot_table_token_capacity
                    actual_max_len = (
                        int(decode_meta.context_lens.max().item())
                        if decode_meta.context_lens.numel() > 0
                        else 0
                    )
                    if actual_max_len > slot_table_len:
                        raise RuntimeError(
                            "decode context length exceeds active slot table "
                            f"width: layer={layer_idx} "
                            f"context_lens_max={actual_max_len} "
                            f"slot_table_len={slot_table_len}"
                        )

                if self.decode_op is not None:
                    o = self.decode_op.run(
                        q,
                        decode_view,
                        decode_launch_op=self.decode_launch_op,
                    )
                else:
                    max_context_len = decode_meta.max_context_len
                    static_cap = getattr(
                        cache_manager,
                        "_decode_static_max_context_len",
                        None,
                    )
                    if static_cap is not None:
                        max_context_len = max(
                            int(max_context_len)
                            if max_context_len is not None
                            else 0,
                            int(static_cap),
                        )
                    if max_context_len is None:
                        raise RuntimeError(
                            "static decode requires max_context_len, got None "
                            f"at layer={layer_idx}"
                        )
                    max_len_in_batch = int(max_context_len)
                    if decode_meta.active_slots.dim() == 2:
                        slot_table_len = decode_meta.slot_table_token_capacity
                        if max_len_in_batch > slot_table_len:
                            max_len_in_batch = slot_table_len
                        if max_len_in_batch <= 0:
                            raise RuntimeError(
                                "decode requires a positive context length, got "
                                f"{max_len_in_batch} at layer={layer_idx}"
                            )
                    block_seq = cache_manager.get_decode_block_seq(layer_idx, 256)
                    if self.decode_launch_op is None:
                        gqa_block_n, gqa_num_warps = 16, 2
                    else:
                        block_seq, gqa_block_n, gqa_num_warps = (
                            self.decode_launch_op.launch_config(
                                block_seq=block_seq,
                                max_context_len=max_len_in_batch,
                                requires_attention_scores=(
                                    decode_meta.attn_score is not None
                                ),
                            )
                        )
                    num_seq_blocks = (
                        max_len_in_batch + block_seq - 1
                    ) // block_seq
                    workspace_provider = getattr(
                        self.attention_backend,
                        "get_decode_workspace",
                        None,
                    )
                    if callable(workspace_provider):
                        mid_o, mid_o_logexpsum = workspace_provider(
                            batch_size=batch_size,
                            num_heads=self.num_heads,
                            head_dim=self.head_dim,
                            device=q.device,
                        )
                    else:
                        mid_o, mid_o_logexpsum = get_decode_workspace(
                            context,
                            batch_size,
                            self.num_heads,
                            num_seq_blocks,
                            self.head_dim,
                            q.device,
                        )
                    o = self.attention_backend.run_decode(
                        q,
                        decode_view,
                        mid_o=mid_o,
                        mid_o_logexpsum=mid_o_logexpsum,
                        max_len_in_batch=max_len_in_batch,
                        block_seq=block_seq,
                        num_heads=self.num_heads,
                        num_kv_heads=self.num_kv_heads,
                        gqa_block_n=gqa_block_n,
                        gqa_num_warps=gqa_num_warps,
                    )
                cache_manager.record_decode_query(layer_idx, q)

            sparse_controller.on_layer_attention_end(layer_idx)
            cache_manager.on_layer_attention_end(layer_idx)
            return o
        finally:
            if temp_slots is not None and temp_slots.numel() > 0:
                cache_manager.release_layer_temp_slots(layer_idx, temp_slots)
