from __future__ import annotations

import torch
import triton
import triton.language as tl

from sparseengine.kernels.triton.moe import (
    _prepare_expert_assignment,
    _quantize_fp8_group128,
    _routed_fp8_gemm,
    moe_sum,
)


@triton.jit(do_not_specialize=["EM", "num_assignments"])
def _fused_gate_up_swiglu_fp8_kernel(
    a_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_assignments,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_bse,
    stride_bsn,
    stride_bsk,
    stride_cm,
    stride_cn,
    INPUT_TOP_K: tl.constexpr,
    NAIVE_ASSIGNMENT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_offsets = tl.arange(0, BLOCK_SIZE_M)
    if NAIVE_ASSIGNMENT:
        assignment_ids = tl.where(
            row_offsets == 0,
            pid_m,
            num_assignments,
        )
    else:
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return
        assignment_ids = tl.load(
            sorted_token_ids_ptr + pid_m * BLOCK_SIZE_M + row_offsets
        )
    assignment_ids = assignment_ids.to(tl.int64)
    assignment_mask = assignment_ids < num_assignments
    expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if expert_id < 0:
        return

    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    input_rows = assignment_ids // INPUT_TOP_K
    gate_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    up_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_block in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        remaining_k = K - k_block * BLOCK_SIZE_K
        a_raw = tl.load(
            a_ptr
            + input_rows[:, None] * stride_am
            + (k_block * BLOCK_SIZE_K + offsets_k[None, :]) * stride_ak,
            mask=assignment_mask[:, None]
            & (offsets_k[None, :] < remaining_k),
            other=0.0,
        ).to(tl.float32)
        a_scale = tl.max(tl.abs(a_raw), axis=1) / 448.0
        a_quant = (a_raw / tl.maximum(a_scale[:, None], 1.0e-12)).to(
            tl.float8e4nv
        )
        weight_offsets = (
            expert_id * stride_be
            + offsets_n[None, :] * stride_bn
            + (k_block * BLOCK_SIZE_K + offsets_k[:, None]) * stride_bk
        )
        weight_mask = (offsets_n[None, :] < N) & (
            offsets_k[:, None] < remaining_k
        )
        gate = tl.load(b_ptr + weight_offsets, mask=weight_mask, other=0.0)
        up = tl.load(
            b_ptr + weight_offsets + N * stride_bn,
            mask=weight_mask,
            other=0.0,
        )
        gate_scale = tl.load(
            b_scale_ptr
            + expert_id * stride_bse
            + pid_n * stride_bsn
            + k_block * stride_bsk
        ).to(tl.float32)
        up_scale = tl.load(
            b_scale_ptr
            + expert_id * stride_bse
            + (pid_n + tl.cdiv(N, BLOCK_SIZE_N)) * stride_bsn
            + k_block * stride_bsk
        ).to(tl.float32)
        gate_accumulator += tl.dot(a_quant, gate) * a_scale[:, None] * gate_scale
        up_accumulator += tl.dot(a_quant, up) * a_scale[:, None] * up_scale

    element_dtype = c_ptr.dtype.element_ty
    gate = gate_accumulator.to(element_dtype).to(tl.float32)
    up = up_accumulator.to(element_dtype)
    gate = (gate / (1.0 + tl.exp(-gate))).to(element_dtype)
    output = gate * up
    tl.store(
        c_ptr
        + assignment_ids[:, None] * stride_cm
        + offsets_n[None, :] * stride_cn,
        output,
        mask=assignment_mask[:, None] & (offsets_n[None, :] < N),
    )


def _fused_gate_up_swiglu_fp8(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale_inv: torch.Tensor,
    activated: torch.Tensor,
    alignment,
    *,
    top_k: int,
) -> None:
    num_assignments = int(activated.shape[0])
    if alignment.naive:
        em = num_assignments * alignment.block_size
        sorted_token_ids = activated
    else:
        if alignment.sorted_token_ids is None:
            raise RuntimeError("Aligned MiniMax MoE is missing sorted_token_ids.")
        em = int(alignment.sorted_token_ids.numel())
        sorted_token_ids = alignment.sorted_token_ids
    intermediate_size = int(activated.shape[1])
    grid = (
        triton.cdiv(em, alignment.block_size),
        triton.cdiv(intermediate_size, 128),
    )
    _fused_gate_up_swiglu_fp8_kernel[grid](
        hidden_states,
        w13_weight,
        w13_scale_inv,
        activated,
        sorted_token_ids,
        alignment.expert_ids,
        alignment.num_tokens_post_padded,
        N=intermediate_size,
        K=int(hidden_states.shape[1]),
        EM=em,
        num_assignments=num_assignments,
        stride_am=hidden_states.stride(0),
        stride_ak=hidden_states.stride(1),
        stride_be=w13_weight.stride(0),
        stride_bn=w13_weight.stride(1),
        stride_bk=w13_weight.stride(2),
        stride_bse=w13_scale_inv.stride(0),
        stride_bsn=w13_scale_inv.stride(1),
        stride_bsk=w13_scale_inv.stride(2),
        stride_cm=activated.stride(0),
        stride_cn=activated.stride(1),
        INPUT_TOP_K=int(top_k),
        NAIVE_ASSIGNMENT=alignment.naive,
        BLOCK_SIZE_M=alignment.block_size,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=128,
        num_warps=4,
        num_stages=3,
    )


def fused_minimax_m2_moe_fp8(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_scale_inv: torch.Tensor,
    w2_scale_inv: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    local_expert_start: int,
) -> torch.Tensor:
    """Run MiniMax M2.7 FP8 experts with fused gate/up GEMM and SwiGLU."""

    tensors = {
        "hidden_states": hidden_states,
        "w13_weight": w13_weight,
        "w2_weight": w2_weight,
        "w13_scale_inv": w13_scale_inv,
        "w2_scale_inv": w2_scale_inv,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
    }
    for name, tensor in tensors.items():
        if not tensor.is_cuda or tensor.device != hidden_states.device:
            raise ValueError(
                f"MiniMax M2.7 fused MoE requires CUDA {name} on one device."
            )
        if not tensor.is_contiguous():
            raise ValueError(f"MiniMax M2.7 fused MoE requires contiguous {name}.")
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError("MiniMax M2.7 fused MoE requires BF16 activations.")
    if (
        w13_weight.dtype != torch.float8_e4m3fn
        or w2_weight.dtype != torch.float8_e4m3fn
    ):
        raise TypeError("MiniMax M2.7 fused MoE requires FP8 E4M3 weights.")
    if w13_scale_inv.dtype != torch.float32 or w2_scale_inv.dtype != torch.float32:
        raise TypeError("MiniMax M2.7 fused MoE requires FP32 expert scales.")
    if (
        hidden_states.ndim != 2
        or topk_ids.ndim != 2
        or topk_weights.shape != topk_ids.shape
    ):
        raise ValueError(
            "MiniMax M2.7 fused MoE expects [tokens, hidden] and [tokens, top_k]."
        )
    if int(topk_ids.shape[0]) != int(hidden_states.shape[0]):
        raise ValueError("MiniMax M2.7 router token count does not match hidden states.")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("MiniMax M2.7 fused MoE requires INT32 or INT64 topk_ids.")
    if topk_weights.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("MiniMax M2.7 fused MoE requires floating-point topk_weights.")
    if w13_weight.ndim != 3 or w2_weight.ndim != 3:
        raise ValueError("MiniMax M2.7 fused MoE requires rank-3 expert weights.")

    num_local_experts = int(w13_weight.shape[0])
    hidden_size = int(hidden_states.shape[1])
    intermediate_size = int(w13_weight.shape[1]) // 2
    if hidden_size % 128 or intermediate_size % 128:
        raise ValueError("MiniMax M2.7 fused MoE dimensions must be 128-aligned.")
    if tuple(w13_weight.shape) != (
        num_local_experts,
        2 * intermediate_size,
        hidden_size,
    ):
        raise ValueError("MiniMax M2.7 packed gate/up weight shape is inconsistent.")
    if tuple(w2_weight.shape) != (num_local_experts, hidden_size, intermediate_size):
        raise ValueError("MiniMax M2.7 down weight shape is inconsistent.")
    expected_w13_scale = (
        num_local_experts,
        2 * intermediate_size // 128,
        hidden_size // 128,
    )
    expected_w2_scale = (
        num_local_experts,
        hidden_size // 128,
        intermediate_size // 128,
    )
    if tuple(w13_scale_inv.shape) != expected_w13_scale:
        raise ValueError(f"MiniMax M2.7 w13 scale shape must be {expected_w13_scale}.")
    if tuple(w2_scale_inv.shape) != expected_w2_scale:
        raise ValueError(f"MiniMax M2.7 w2 scale shape must be {expected_w2_scale}.")

    num_experts = int(num_experts)
    local_expert_start = int(local_expert_start)
    local_expert_end = local_expert_start + num_local_experts
    if not 0 <= local_expert_start < local_expert_end <= num_experts:
        raise ValueError(
            "MiniMax M2.7 local expert range is inconsistent with num_experts."
        )
    num_tokens = int(hidden_states.shape[0])
    top_k = int(topk_ids.shape[1])
    num_assignments = num_tokens * top_k
    alignment = _prepare_expert_assignment(
        topk_ids,
        block_size=16,
        num_experts=num_experts,
        local_expert_start=local_expert_start,
        local_expert_end=local_expert_end,
    )
    activated = torch.empty(
        (num_assignments, intermediate_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    _fused_gate_up_swiglu_fp8(
        hidden_states,
        w13_weight,
        w13_scale_inv,
        activated,
        alignment,
        top_k=top_k,
    )
    w2_output = torch.empty(
        (num_assignments, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    activated_q, activated_scale = _quantize_fp8_group128(activated)
    _routed_fp8_gemm(
        activated_q,
        activated_scale,
        w2_weight,
        w2_scale_inv,
        w2_output,
        topk_weights,
        alignment,
        input_top_k=1,
        multiply_routing_weight=True,
    )
    return moe_sum(
        w2_output.view(num_tokens, top_k, hidden_size),
        topk_ids,
        num_experts=num_experts,
        local_expert_start=local_expert_start,
        local_expert_end=local_expert_end,
    )
