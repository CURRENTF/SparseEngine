from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _glm_mla_decode_q_projection_rope_kernel(
    q_lora,
    q_b_weight,
    k_rope,
    positions,
    cos_sin_cache,
    output_q,
    output_k_rope,
    q_lora_stride_row,
    q_lora_stride_dim,
    weight_stride_out,
    weight_stride_in,
    k_rope_stride_row,
    positions_stride_row,
    cache_stride_position,
    output_q_stride_row,
    output_q_stride_dim,
    output_k_stride_row,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_block = tl.program_id(0)
    column_block = tl.program_id(1)
    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = column_block * BLOCK_N + tl.arange(0, BLOCK_N)
    input_offsets = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for input_block in range(0, tl.cdiv(K, BLOCK_K)):
        dimensions = input_block * BLOCK_K + input_offsets
        activations = tl.load(
            q_lora
            + rows[:, None] * q_lora_stride_row
            + dimensions[None, :] * q_lora_stride_dim,
            mask=(rows[:, None] < M) & (dimensions[None, :] < K),
            other=0.0,
        )
        weights = tl.load(
            q_b_weight
            + columns[None, :] * weight_stride_out
            + dimensions[:, None] * weight_stride_in,
            mask=(columns[None, :] < N) & (dimensions[:, None] < K),
            other=0.0,
        )
        accumulator += tl.dot(activations, weights)

    projected = accumulator.to(tl.bfloat16)
    local_dimensions = columns % HEAD_DIM
    tl.store(
        output_q
        + rows[:, None] * output_q_stride_row
        + columns[None, :] * output_q_stride_dim,
        projected,
        mask=(rows[:, None] < M)
        & (columns[None, :] < N)
        & (local_dimensions[None, :] < NOPE_DIM),
    )

    blocks_per_head: tl.constexpr = HEAD_DIM // BLOCK_N
    rope_block: tl.constexpr = NOPE_DIM // BLOCK_N
    is_rope_block = column_block % blocks_per_head == rope_block
    projected_pairs = projected.reshape((BLOCK_M, BLOCK_N // 2, 2))
    projected_even, projected_odd = tl.split(projected_pairs)
    pair_offsets = tl.arange(0, ROPE_DIM // 2)
    positions_row = tl.load(
        positions + rows * positions_stride_row,
        mask=rows < M,
        other=0,
    ).to(tl.int64)
    cache_base = positions_row[:, None] * cache_stride_position
    cos = tl.load(
        cos_sin_cache + cache_base + pair_offsets[None, :],
        mask=(rows[:, None] < M) & is_rope_block,
        other=0.0,
    )
    sin = tl.load(
        cos_sin_cache
        + cache_base
        + ROPE_DIM // 2
        + pair_offsets[None, :],
        mask=(rows[:, None] < M) & is_rope_block,
        other=0.0,
    )
    rotated_first = projected_even * cos - projected_odd * sin
    rotated_second = projected_odd * cos + projected_even * sin
    rope_output_base = column_block * BLOCK_N
    tl.store(
        output_q
        + rows[:, None] * output_q_stride_row
        + (rope_output_base + pair_offsets)[None, :] * output_q_stride_dim,
        rotated_first,
        mask=(rows[:, None] < M) & is_rope_block,
    )
    tl.store(
        output_q
        + rows[:, None] * output_q_stride_row
        + (
            rope_output_base + ROPE_DIM // 2 + pair_offsets
        )[None, :]
        * output_q_stride_dim,
        rotated_second,
        mask=(rows[:, None] < M) & is_rope_block,
    )

    if column_block == blocks_per_head - 1:
        rope_offsets = tl.arange(0, BLOCK_N)
        key_pairs = rope_offsets % (ROPE_DIM // 2)
        key_even = tl.load(
            k_rope
            + rows[:, None] * k_rope_stride_row
            + 2 * key_pairs[None, :],
            mask=(rows[:, None] < M) & (rope_offsets[None, :] < ROPE_DIM),
            other=0.0,
        ).to(tl.float32)
        key_odd = tl.load(
            k_rope
            + rows[:, None] * k_rope_stride_row
            + 2 * key_pairs[None, :]
            + 1,
            mask=(rows[:, None] < M) & (rope_offsets[None, :] < ROPE_DIM),
            other=0.0,
        ).to(tl.float32)
        key_cos = tl.load(
            cos_sin_cache + cache_base + key_pairs[None, :],
            mask=(rows[:, None] < M) & (rope_offsets[None, :] < ROPE_DIM),
            other=0.0,
        )
        key_sin = tl.load(
            cos_sin_cache
            + cache_base
            + ROPE_DIM // 2
            + key_pairs[None, :],
            mask=(rows[:, None] < M) & (rope_offsets[None, :] < ROPE_DIM),
            other=0.0,
        )
        rotated_key = tl.where(
            rope_offsets[None, :] < ROPE_DIM // 2,
            key_even * key_cos - key_odd * key_sin,
            key_odd * key_cos + key_even * key_sin,
        )
        tl.store(
            output_k_rope
            + rows[:, None] * output_k_stride_row
            + rope_offsets[None, :],
            rotated_key,
            mask=(rows[:, None] < M) & (rope_offsets[None, :] < ROPE_DIM),
        )


@triton.jit
def _glm_mla_decode_rope_kernel(
    q_nope,
    q_rope,
    k_rope,
    positions,
    cos_sin_cache,
    output_q,
    output_k_rope,
    q_nope_stride_row,
    q_nope_stride_head,
    q_rope_stride_row,
    q_rope_stride_head,
    k_rope_stride_row,
    cache_stride_position,
    output_q_stride_row,
    output_q_stride_head,
    output_k_stride_row,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_D)
    nope_mask = offsets < NOPE_DIM
    rope_output_offsets = offsets - NOPE_DIM
    rope_mask = (offsets >= NOPE_DIM) & (rope_output_offsets < ROPE_DIM)
    half_rope: tl.constexpr = ROPE_DIM // 2
    pair_offsets = rope_output_offsets % half_rope
    even_offsets = 2 * pair_offsets
    odd_offsets = even_offsets + 1

    position = tl.load(positions + row).to(tl.int64)
    cache_base = position * cache_stride_position
    cos = tl.load(
        cos_sin_cache + cache_base + pair_offsets,
        mask=rope_mask,
        other=0.0,
    )
    sin = tl.load(
        cos_sin_cache + cache_base + half_rope + pair_offsets,
        mask=rope_mask,
        other=0.0,
    )
    q_rope_base = row * q_rope_stride_row + head * q_rope_stride_head
    q_even = tl.load(
        q_rope + q_rope_base + even_offsets,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    q_odd = tl.load(
        q_rope + q_rope_base + odd_offsets,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    q_rotated = tl.where(
        rope_output_offsets < half_rope,
        q_even * cos - q_odd * sin,
        q_odd * cos + q_even * sin,
    )
    q_nope_values = tl.load(
        q_nope
        + row * q_nope_stride_row
        + head * q_nope_stride_head
        + offsets,
        mask=nope_mask,
        other=0.0,
    )
    tl.store(
        output_q
        + row * output_q_stride_row
        + head * output_q_stride_head
        + offsets,
        tl.where(nope_mask, q_nope_values, q_rotated),
        mask=nope_mask | rope_mask,
    )

    if head == 0:
        k_base = row * k_rope_stride_row
        k_even = tl.load(
            k_rope + k_base + even_offsets,
            mask=rope_mask,
            other=0.0,
        ).to(tl.float32)
        k_odd = tl.load(
            k_rope + k_base + odd_offsets,
            mask=rope_mask,
            other=0.0,
        ).to(tl.float32)
        k_rotated = tl.where(
            rope_output_offsets < half_rope,
            k_even * cos - k_odd * sin,
            k_odd * cos + k_even * sin,
        )
        tl.store(
            output_k_rope + row * output_k_stride_row + rope_output_offsets,
            k_rotated,
            mask=rope_mask,
        )


def fuse_glm_mla_decode_rope(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate GLM MLA Q/K RoPE and compose the contiguous decode query."""

    if q_nope.ndim != 3 or q_rope.ndim != 3 or k_rope.ndim != 2:
        raise ValueError("GLM MLA decode RoPE expects rank-3 Q and rank-2 K tensors.")
    batch_size, num_heads, nope_dim = map(int, q_nope.shape)
    rope_dim = int(q_rope.shape[-1])
    if tuple(q_rope.shape[:2]) != (batch_size, num_heads):
        raise ValueError("GLM MLA decode Q tensors must share row/head axes.")
    if tuple(k_rope.shape) != (batch_size, rope_dim):
        raise ValueError("GLM MLA decode K RoPE must be batch aligned with Q.")
    if positions.shape != (batch_size,):
        raise ValueError("GLM MLA decode positions must contain one value per row.")
    if rope_dim <= 0 or rope_dim % 2:
        raise ValueError(
            "GLM MLA decode RoPE dimension must be positive and even, "
            f"got {rope_dim}."
        )
    if cos_sin_cache.ndim != 3 or tuple(cos_sin_cache.shape[1:]) != (1, rope_dim):
        raise ValueError("GLM MLA cos/sin cache must have shape [positions, 1, rope_dim].")
    tensors = (q_rope, k_rope, positions, cos_sin_cache)
    if any(tensor.device != q_nope.device for tensor in tensors):
        raise ValueError("GLM MLA decode RoPE inputs must share one device.")
    if (
        q_nope.dtype != torch.bfloat16
        or q_rope.dtype != q_nope.dtype
        or k_rope.dtype != q_nope.dtype
    ):
        raise TypeError("GLM MLA decode RoPE requires BF16 Q/K tensors.")
    if cos_sin_cache.dtype != torch.float32:
        raise TypeError("GLM MLA cos/sin cache must use FP32.")
    if positions.dtype not in {torch.int32, torch.int64}:
        raise TypeError("GLM MLA decode positions must use int32 or int64.")
    if not q_nope.is_cuda:
        raise ValueError("GLM MLA decode RoPE requires CUDA tensors.")
    if any(
        tensor.stride(-1) != 1
        for tensor in (q_nope, q_rope, k_rope, cos_sin_cache)
    ):
        raise ValueError(
            "GLM MLA decode RoPE inputs must be contiguous in the last dimension."
        )

    output_q = torch.empty(
        (batch_size, num_heads, nope_dim + rope_dim),
        dtype=q_nope.dtype,
        device=q_nope.device,
    )
    output_k_rope = torch.empty_like(k_rope)
    _glm_mla_decode_rope_kernel[(batch_size, num_heads)](
        q_nope,
        q_rope,
        k_rope,
        positions,
        cos_sin_cache,
        output_q,
        output_k_rope,
        q_nope.stride(0),
        q_nope.stride(1),
        q_rope.stride(0),
        q_rope.stride(1),
        k_rope.stride(0),
        cos_sin_cache.stride(0),
        output_q.stride(0),
        output_q.stride(1),
        output_k_rope.stride(0),
        NOPE_DIM=nope_dim,
        ROPE_DIM=rope_dim,
        BLOCK_D=triton.next_power_of_2(nope_dim + rope_dim),
        num_warps=4,
        num_stages=1,
    )
    return output_q, output_k_rope


def project_and_fuse_glm_mla_decode_rope(
    q_lora: torch.Tensor,
    q_b_weight: torch.Tensor,
    k_rope: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    num_heads: int,
    nope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project GLM MLA decode queries and compose interleaved Q/K RoPE."""

    if q_lora.ndim != 2 or q_b_weight.ndim != 2 or k_rope.ndim != 2:
        raise ValueError(
            "GLM MLA fused decode projection expects rank-2 activations, "
            "weights, and K RoPE."
        )
    batch_size, input_dim = map(int, q_lora.shape)
    output_dim, weight_input_dim = map(int, q_b_weight.shape)
    num_heads = int(num_heads)
    nope_dim = int(nope_dim)
    rope_dim = int(k_rope.shape[1])
    head_dim = nope_dim + rope_dim
    if batch_size <= 0 or input_dim <= 0:
        raise ValueError("GLM MLA fused decode projection requires non-empty inputs.")
    if weight_input_dim != input_dim or output_dim != num_heads * head_dim:
        raise ValueError(
            "GLM MLA fused decode projection shape mismatch: "
            f"q={tuple(q_lora.shape)} weight={tuple(q_b_weight.shape)} "
            f"heads={num_heads} head_dim={head_dim}."
        )
    if positions.shape != (batch_size,) or k_rope.shape[0] != batch_size:
        raise ValueError("GLM MLA fused decode positions and K must be batch aligned.")
    if (
        rope_dim != 64
        or nope_dim <= 0
        or nope_dim % 64
        or head_dim % 64
    ):
        raise ValueError(
            "GLM MLA fused decode projection requires RoPE dim 64 and "
            f"64-aligned no-RoPE/head dims, got {rope_dim}/{nope_dim}/{head_dim}."
        )
    if cos_sin_cache.ndim != 3 or tuple(cos_sin_cache.shape[1:]) != (1, rope_dim):
        raise ValueError("GLM MLA cos/sin cache must have shape [positions, 1, 64].")
    tensors = (q_b_weight, k_rope, positions, cos_sin_cache)
    if any(tensor.device != q_lora.device for tensor in tensors):
        raise ValueError("GLM MLA fused decode projection inputs must share a device.")
    if (
        q_lora.dtype != torch.bfloat16
        or q_b_weight.dtype != q_lora.dtype
        or k_rope.dtype != q_lora.dtype
    ):
        raise TypeError("GLM MLA fused decode projection requires BF16 Q/K/weights.")
    if cos_sin_cache.dtype != torch.float32:
        raise TypeError("GLM MLA cos/sin cache must use FP32.")
    if positions.dtype not in {torch.int32, torch.int64}:
        raise TypeError("GLM MLA fused decode positions must use int32 or int64.")
    if not q_lora.is_cuda:
        raise ValueError("GLM MLA fused decode projection requires CUDA tensors.")
    if any(
        tensor.stride(-1) != 1
        for tensor in (q_lora, q_b_weight, k_rope, cos_sin_cache)
    ):
        raise ValueError(
            "GLM MLA fused decode projection inputs must be contiguous in the "
            "last dimension."
        )

    output_q = torch.empty(
        (batch_size, output_dim),
        dtype=q_lora.dtype,
        device=q_lora.device,
    )
    output_k_rope = torch.empty_like(k_rope)
    block_m = 16
    block_n = 64
    _glm_mla_decode_q_projection_rope_kernel[
        (triton.cdiv(batch_size, block_m), triton.cdiv(output_dim, block_n))
    ](
        q_lora,
        q_b_weight,
        k_rope,
        positions,
        cos_sin_cache,
        output_q,
        output_k_rope,
        q_lora.stride(0),
        q_lora.stride(1),
        q_b_weight.stride(0),
        q_b_weight.stride(1),
        k_rope.stride(0),
        positions.stride(0),
        cos_sin_cache.stride(0),
        output_q.stride(0),
        output_q.stride(1),
        output_k_rope.stride(0),
        M=batch_size,
        N=output_dim,
        K=input_dim,
        HEAD_DIM=head_dim,
        NOPE_DIM=nope_dim,
        ROPE_DIM=rope_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=128,
        num_warps=4,
        num_stages=3,
    )
    return output_q.view(batch_size, num_heads, head_dim), output_k_rope


__all__ = [
    "fuse_glm_mla_decode_rope",
    "project_and_fuse_glm_mla_decode_rope",
]
