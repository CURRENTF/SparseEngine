import torch
import triton
import triton.language as tl
import os


@triton.jit
def _omnikv_build_keep_and_slots_kernel(
    topk_indices_ptr,
    topk_lens_ptr,
    hist_lens_ptr,
    recent_lens_ptr,
    context_lens_ptr,
    buffer_req_to_token_slots_ptr,
    req_indices_ptr,
    new_context_lens_ptr,
    keep_indices_ptr,
    active_slots_ptr,
    stride_topk_b,
    stride_topk_k,
    stride_buf_b,
    stride_buf_s,
    stride_keep_b,
    stride_keep_s,
    stride_active_b,
    stride_active_s,
    max_s,
    buf_max_len,
    num_sink,
    BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_blk = tl.program_id(1)
    offs = pid_blk * BLOCK + tl.arange(0, BLOCK)
    mask_out = offs < max_s

    hist_len = tl.load(hist_lens_ptr + pid_b)
    recent_len = tl.load(recent_lens_ptr + pid_b)
    k_b = tl.load(topk_lens_ptr + pid_b).to(tl.int32)
    req_row = tl.load(req_indices_ptr + pid_b)
    context_len = tl.maximum(tl.load(context_lens_ptr + pid_b), 0)
    sink_len = tl.minimum(num_sink, context_len)
    hist_len = tl.minimum(tl.maximum(hist_len, sink_len), context_len).to(tl.int32)
    recent_len = tl.minimum(tl.maximum(recent_len, 0), context_len - hist_len).to(tl.int32)
    k_b = tl.minimum(k_b, hist_len - sink_len)
    new_context_len = sink_len + k_b + recent_len
    tl.store(new_context_lens_ptr + pid_b, new_context_len, mask=pid_blk == 0)

    mask_sink = offs < sink_len
    mask_topk = (offs >= sink_len) & (offs < sink_len + k_b)
    mask_recent = (offs >= sink_len + k_b) & (offs < new_context_len)

    topk_offs = tl.where(mask_topk, offs - sink_len, 0)
    topk_idx = tl.load(
        topk_indices_ptr + pid_b * stride_topk_b + topk_offs * stride_topk_k,
        mask=mask_topk,
        other=0,
    )
    recent_offs = tl.where(mask_recent, offs - sink_len - k_b, 0)
    recent_idx = hist_len + recent_offs

    keep_idx = tl.where(mask_sink, offs, tl.where(mask_topk, topk_idx, tl.where(mask_recent, recent_idx, 0)))
    keep_idx = keep_idx.to(tl.int32)
    tl.store(
        keep_indices_ptr + pid_b * stride_keep_b + offs * stride_keep_s,
        keep_idx,
        mask=mask_out,
    )

    keep_idx_i64 = keep_idx.to(tl.int64)
    req_row_i64 = req_row.to(tl.int64)
    mask_slot = mask_out & (offs < new_context_len) & (keep_idx >= 0) & (keep_idx < buf_max_len)
    slot_val = tl.load(
        buffer_req_to_token_slots_ptr + req_row_i64 * stride_buf_b + keep_idx_i64 * stride_buf_s,
        mask=mask_slot,
        other=0,
    )
    tl.store(
        active_slots_ptr + pid_b * stride_active_b + offs * stride_active_s,
        slot_val,
        mask=mask_out,
    )


def build_omnikv_keep_and_slots(
    topk_indices: torch.Tensor,
    topk_lens: torch.Tensor,
    hist_lens: torch.Tensor,
    recent_chunk_lens: torch.Tensor,
    buffer_req_to_token_slots: torch.Tensor,
    req_indices: torch.Tensor,
    num_sink: int,
    max_s: int | None = None,
    *,
    context_lens: torch.Tensor,
    keep_indices_out: torch.Tensor | None = None,
    active_slots_out: torch.Tensor | None = None,
    new_context_lens_out: torch.Tensor | None = None,
):
    if topk_indices.dtype != torch.int32:
        topk_indices = topk_indices.to(torch.int32)
    if hist_lens.dtype != torch.int32:
        hist_lens = hist_lens.to(torch.int32)
    if recent_chunk_lens.dtype != torch.int32:
        recent_chunk_lens = recent_chunk_lens.to(torch.int32)
    if topk_lens.dtype != torch.int32:
        topk_lens = topk_lens.to(torch.int32)
    if req_indices.dtype != torch.int32:
        req_indices = req_indices.to(torch.int32)

    batch_size, k_max = topk_indices.shape
    if os.environ.get("OMNIKV_ASSERT", "0") == "1":
        assert topk_lens.shape == (batch_size,)
        assert int(topk_lens.min().item()) >= 0
        assert int(topk_lens.max().item()) <= k_max
    if max_s is None:
        computed_context_lens = torch.minimum(
            num_sink + topk_lens + recent_chunk_lens.clamp_min(0), context_lens
        )
        max_s = int(computed_context_lens.max().item())
    else:
        max_s = int(max_s)
        if max_s < 0:
            raise ValueError(f"max_s must be >= 0, got {max_s}.")

    def _resolve_output(
        name: str,
        output: torch.Tensor | None,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        if output is None:
            return torch.empty(
                shape,
                dtype=torch.int32,
                device=topk_indices.device,
            )
        if output.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(output.shape)}."
            )
        if output.dtype != torch.int32 or output.device != topk_indices.device:
            raise TypeError(
                f"{name} must be int32 on {topk_indices.device}, got "
                f"{output.dtype}/{output.device}."
            )
        return output

    keep_indices = _resolve_output(
        "keep_indices_out",
        keep_indices_out,
        (batch_size, max_s),
    )
    active_slots = _resolve_output(
        "active_slots_out",
        active_slots_out,
        (batch_size, max_s),
    )
    new_context_lens = _resolve_output(
        "new_context_lens_out",
        new_context_lens_out,
        (batch_size,),
    )

    if max_s == 0:
        new_context_lens.zero_()
        return keep_indices, active_slots, new_context_lens

    block = 256
    grid = (batch_size, triton.cdiv(max_s, block))
    _omnikv_build_keep_and_slots_kernel[grid](
        topk_indices,
        topk_lens,
        hist_lens,
        recent_chunk_lens,
        context_lens,
        buffer_req_to_token_slots,
        req_indices,
        new_context_lens,
        keep_indices,
        active_slots,
        topk_indices.stride(0),
        topk_indices.stride(1),
        buffer_req_to_token_slots.stride(0),
        buffer_req_to_token_slots.stride(1),
        keep_indices.stride(0),
        keep_indices.stride(1),
        active_slots.stride(0),
        active_slots.stride(1),
        max_s,
        buffer_req_to_token_slots.shape[1],
        num_sink,
        BLOCK=block,
    )
    return keep_indices, active_slots, new_context_lens
