from __future__ import annotations

import torch
import triton
import triton.language as tl


_H20_DECODE_CONFIGS = {
    (2048, 256): dict(block_m=16, block_n=32, block_k=64, warps=4, stages=4),
    (2048, 512): dict(block_m=16, block_n=32, block_k=64, warps=4, stages=4),
}


@triton.jit
def _gate_up_swiglu_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m_offsets = tl.arange(0, BLOCK_M)
    n_offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    input_ptrs = input_ptr + m_offsets[:, None] * K + k_offsets[None, :]
    gate_ptrs = weight_ptr + n_offsets[None, :] * K + k_offsets[:, None]
    up_ptrs = gate_ptrs + N * K
    gate_accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        remaining_k = K - k_start * BLOCK_K
        input_values = tl.load(
            input_ptrs,
            mask=(m_offsets[:, None] == 0)
            & (k_offsets[None, :] < remaining_k),
            other=0.0,
        )
        weight_mask = (k_offsets[:, None] < remaining_k) & (
            n_offsets[None, :] < N
        )
        gate_accumulator += tl.dot(
            input_values,
            tl.load(gate_ptrs, mask=weight_mask, other=0.0),
        )
        up_accumulator += tl.dot(
            input_values,
            tl.load(up_ptrs, mask=weight_mask, other=0.0),
        )
        input_ptrs += BLOCK_K
        gate_ptrs += BLOCK_K
        up_ptrs += BLOCK_K

    element_dtype = weight_ptr.dtype.element_ty
    gate = gate_accumulator.to(element_dtype).to(tl.float32)
    up = up_accumulator.to(element_dtype)
    gate = (gate / (1.0 + tl.exp(-gate))).to(element_dtype)
    tl.store(
        output_ptr + m_offsets[:, None] * N + n_offsets[None, :],
        gate * up,
        mask=(m_offsets[:, None] == 0) & (n_offsets[None, :] < N),
    )


def h20_gate_up_swiglu(inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if inputs.ndim != 2 or weight.ndim != 2 or weight.shape[0] % 2:
        raise ValueError(
            "h20_gate_up_swiglu expects input [1, K] and weight [2N, K]."
        )
    shape = (int(inputs.shape[1]), int(weight.shape[0]) // 2)
    config = _H20_DECODE_CONFIGS.get(shape)
    if inputs.shape != (1, shape[0]) or weight.shape != (2 * shape[1], shape[0]) or config is None:
        raise ValueError(f"No H20 gate/up SwiGLU config for shape {shape}.")
    if inputs.dtype != torch.bfloat16 or weight.dtype != inputs.dtype:
        raise TypeError("h20_gate_up_swiglu requires matching BF16 tensors.")
    if not inputs.is_cuda or weight.device != inputs.device:
        raise ValueError("h20_gate_up_swiglu requires CUDA tensors on one device.")
    if not inputs.is_contiguous() or not weight.is_contiguous():
        raise ValueError("h20_gate_up_swiglu requires contiguous tensors.")

    output = torch.empty((1, shape[1]), dtype=inputs.dtype, device=inputs.device)
    _gate_up_swiglu_kernel[(triton.cdiv(shape[1], config["block_n"]),)](
        inputs,
        weight,
        output,
        N=shape[1],
        K=shape[0],
        BLOCK_M=config["block_m"],
        BLOCK_N=config["block_n"],
        BLOCK_K=config["block_k"],
        num_warps=config["warps"],
        num_stages=config["stages"],
    )
    return output
