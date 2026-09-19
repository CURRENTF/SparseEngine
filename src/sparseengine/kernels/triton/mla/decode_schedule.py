# SPDX-License-Identifier: Apache-2.0
# Scheduling math derived from ModelTC/lightllm at commit
# 65c174ee95ac6a6fd36b18b63d0b33d97e76b770:
# lightllm/common/basemodel/triton_kernel/mla_att/decode_att/
# gqa_flash_decoding.py
# Local rewrite: caller-owned workspace, immutable launch configuration, no
# infer_state/global config/device probing, and no allocation in the run path.

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from sparseengine.platforms import device_runtime

from .decode_stage1 import MLA_LATENT_DIM, decode_stage1
from .decode_stage2 import decode_stage2


GLM_MLA_SOFTMAX_SCALE = 256**-0.5


@dataclass(frozen=True, slots=True)
class MlaDecodeLaunchConfig:
    """Static launch configuration for the two-stage decode kernel."""

    program_count: int = 128
    blocks_per_program: int = 4
    block_n: int = 16
    block_q_heads: int = 16
    stage1_num_warps: int = 4
    stage1_pipeline_stages: int = 2
    stage2_num_warps: int = 4
    stage2_pipeline_stages: int = 2
    target_splits_per_request: int | None = None

    def __post_init__(self) -> None:
        if self.target_splits_per_request is not None and self.target_splits_per_request <= 0:
            raise ValueError("target_splits_per_request must be positive")
        positive_fields = (
            "program_count",
            "blocks_per_program",
            "block_n",
            "block_q_heads",
            "stage1_pipeline_stages",
            "stage2_pipeline_stages",
        )
        for field_name in positive_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        for field_name in ("block_n", "block_q_heads"):
            value = getattr(self, field_name)
            if value & (value - 1):
                raise ValueError(f"{field_name} must be a power of two")
        for field_name in ("stage1_num_warps", "stage2_num_warps"):
            if getattr(self, field_name) not in {1, 2, 4, 8}:
                raise ValueError(f"{field_name} must be one of 1, 2, 4, or 8")


DEFAULT_GLM_MLA_DECODE_CONFIG = MlaDecodeLaunchConfig()

def select_glm_mla_decode_config(
    *,
    batch_size: int,
    local_q_heads: int,
    sm_count: int,
) -> MlaDecodeLaunchConfig:
    """Prepare an SM-scaled persistent-CTA schedule, independent of lengths.

    Stage1 loops over requests inside each CTA: batch is not a grid axis.
    Target one SM wave of split/head work per request, not per batch. Four
    program waves leave room for ragged requests with more than average work.
    This is a tuning heuristic, not a cross-device performance guarantee.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if local_q_heads <= 0:
        raise ValueError("local_q_heads must be positive")
    if sm_count <= 0:
        raise ValueError("sm_count must be positive")
    block_q_heads = min(16, 1 << (local_q_heads - 1).bit_length())
    head_tiles = (local_q_heads + block_q_heads - 1) // block_q_heads
    return MlaDecodeLaunchConfig(
        program_count=4 * sm_count,
        block_n=32,
        block_q_heads=block_q_heads,
        stage1_num_warps=8,
        stage1_pipeline_stages=4,
        stage2_pipeline_stages=1,
        target_splits_per_request=(sm_count + head_tiles - 1) // head_tiles,
    )


@dataclass(frozen=True, slots=True)
class MlaDecodeWorkspace:
    """Caller-owned tensors required by MLA decode."""

    block_size: torch.Tensor
    batch_start_indices: torch.Tensor
    mid_output: torch.Tensor
    mid_logsumexp: torch.Tensor


def required_workspace_blocks(
    batch_size: int,
    config: MlaDecodeLaunchConfig,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    target = (batch_size * config.target_splits_per_request
              if config.target_splits_per_request is not None
              else config.program_count * config.blocks_per_program)
    return target + batch_size


def allocate_mla_decode_workspace(
    *,
    batch_size: int,
    head_count: int,
    device: torch.device | str,
    config: MlaDecodeLaunchConfig = DEFAULT_GLM_MLA_DECODE_CONFIG,
) -> MlaDecodeWorkspace:
    """Allocate decode workspace outside the attention execution path."""

    if head_count <= 0:
        raise ValueError("head_count must be positive")
    block_capacity = required_workspace_blocks(batch_size, config)
    return MlaDecodeWorkspace(
        block_size=torch.empty((1,), dtype=torch.int32, device=device),
        batch_start_indices=torch.empty(
            (batch_size,),
            dtype=torch.int32,
            device=device,
        ),
        mid_output=torch.empty(
            (head_count, block_capacity, MLA_LATENT_DIM),
            dtype=torch.float32,
            device=device,
        ),
        mid_logsumexp=torch.empty(
            (head_count, block_capacity),
            dtype=torch.float32,
            device=device,
        ),
    )


@triton.jit
def _build_decode_schedule_kernel(
    context_lens,
    block_size_ptr,
    batch_start_indices,
    program_count,
    blocks_per_program,
    batch_size,
    BLOCK_N: tl.constexpr,
    PADDED_BATCH_SIZE: tl.constexpr,
    TARGET_SPLITS_PER_REQUEST: tl.constexpr,
):
    offsets = tl.arange(0, PADDED_BATCH_SIZE)
    context_mask = offsets < batch_size
    lengths = tl.load(
        context_lens + offsets,
        mask=context_mask,
        other=0,
    )
    total_tokens = tl.sum(lengths, axis=0)
    target_blocks = program_count * blocks_per_program
    if TARGET_SPLITS_PER_REQUEST > 0:
        # Padding must not inflate the partition target during graph replay.
        active_count = tl.sum((lengths > 0).to(tl.int32), axis=0)
        target_blocks = tl.maximum(1, active_count * TARGET_SPLITS_PER_REQUEST)
    unaligned_block_size = tl.maximum(1, tl.cdiv(total_tokens, target_blocks))
    block_size = tl.cdiv(unaligned_block_size, BLOCK_N) * BLOCK_N

    block_counts = tl.cdiv(lengths, block_size)
    cumulative_blocks = tl.cumsum(block_counts, axis=0)
    starts = cumulative_blocks - block_counts
    tl.store(
        batch_start_indices + offsets,
        starts,
        mask=context_mask,
    )
    tl.store(block_size_ptr, block_size)


def _validate_schedule_workspace(
    context_lens: torch.Tensor,
    workspace: MlaDecodeWorkspace,
    config: MlaDecodeLaunchConfig,
) -> None:
    if context_lens.device.type != "cuda":
        raise ValueError(
            f"context_lens must be a CUDA tensor, got {context_lens.device}"
        )
    if context_lens.dtype != torch.int32:
        raise TypeError(
            f"context_lens must use {torch.int32}, got {context_lens.dtype}"
        )
    if context_lens.ndim != 1 or context_lens.numel() == 0:
        raise ValueError("context_lens must be a non-empty one-dimensional tensor")

    batch_size = context_lens.numel()
    workspace_tensors = {
        "block_size": workspace.block_size,
        "batch_start_indices": workspace.batch_start_indices,
        "mid_output": workspace.mid_output,
        "mid_logsumexp": workspace.mid_logsumexp,
    }
    for name, tensor in workspace_tensors.items():
        if tensor.device != context_lens.device:
            raise ValueError(
                f"workspace {name} is on {tensor.device}, expected "
                f"{context_lens.device}"
            )
    if workspace.block_size.dtype != torch.int32 or workspace.block_size.shape != (
        1,
    ):
        raise ValueError("workspace.block_size must be int32 with shape [1]")
    if (
        workspace.batch_start_indices.dtype != torch.int32
        or workspace.batch_start_indices.ndim != 1
        or workspace.batch_start_indices.numel() < batch_size
    ):
        raise ValueError(
            "workspace.batch_start_indices must be int32 and have capacity "
            f"for {batch_size} rows"
        )
    required_blocks = required_workspace_blocks(batch_size, config)
    if (
        workspace.mid_output.dtype != torch.float32
        or workspace.mid_output.ndim != 3
        or workspace.mid_output.shape[-1] != MLA_LATENT_DIM
    ):
        raise ValueError(
            "workspace.mid_output must be float32 with shape "
            "[heads, blocks, 512]"
        )
    if workspace.mid_output.shape[1] < required_blocks:
        raise ValueError(
            "MLA decode workspace is too small: "
            f"blocks={workspace.mid_output.shape[1]}, "
            f"required={required_blocks}"
        )
    if (
        workspace.mid_logsumexp.dtype != torch.float32
        or workspace.mid_logsumexp.shape != workspace.mid_output.shape[:2]
    ):
        raise ValueError(
            "workspace.mid_logsumexp must be float32 and match the first "
            "two mid_output dimensions"
        )


@torch.no_grad()
def prepare_mla_decode_schedule(
    context_lens: torch.Tensor,
    workspace: MlaDecodeWorkspace,
    *,
    config: MlaDecodeLaunchConfig = DEFAULT_GLM_MLA_DECODE_CONFIG,
) -> None:
    """Fill device-side block size and batch offsets without a CPU sync."""

    _validate_schedule_workspace(context_lens, workspace, config)
    batch_size = context_lens.numel()
    _build_decode_schedule_kernel[(1,)](
        context_lens,
        workspace.block_size,
        workspace.batch_start_indices,
        program_count=config.program_count,
        blocks_per_program=config.blocks_per_program,
        batch_size=batch_size,
        BLOCK_N=config.block_n,
        PADDED_BATCH_SIZE=triton.next_power_of_2(batch_size),
        TARGET_SPLITS_PER_REQUEST=config.target_splits_per_request or 0,
        num_warps=4,
        num_stages=1,
    )


def validate_mla_decode_metadata(
    active_slots: torch.Tensor,
    request_indices: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    cache_slot_count: int,
    max_context_len: int | None = None,
    valid_batch_size: int | None = None,
) -> None:
    """Synchronously validate one decode view before per-layer reuse."""

    metadata = {
        "active_slots": active_slots,
        "request_indices": request_indices,
        "context_lens": context_lens,
    }
    for name, tensor in metadata.items():
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA tensor, got {tensor.device}")
        if tensor.device != context_lens.device:
            raise ValueError(
                f"{name} is on {tensor.device}, expected {context_lens.device}"
            )
        if tensor.dtype != torch.int32:
            raise TypeError(
                f"{name} must use {torch.int32}, got {tensor.dtype}"
            )
    if active_slots.ndim != 2:
        raise ValueError("active_slots must have shape [rows, max_context_len]")
    if context_lens.ndim != 1 or context_lens.numel() == 0:
        raise ValueError("context_lens must be a non-empty one-dimensional tensor")
    batch_size = context_lens.numel()
    if valid_batch_size is None:
        valid_batch_size = batch_size
    valid_batch_size = int(valid_batch_size)
    if not 0 < valid_batch_size <= batch_size:
        raise ValueError(
            "valid_batch_size must be within the metadata batch: "
            f"valid={valid_batch_size} batch={batch_size}"
        )
    if request_indices.shape != (batch_size,):
        raise ValueError(
            f"request_indices must have shape ({batch_size},), got "
            f"{tuple(request_indices.shape)}"
        )
    if cache_slot_count <= 0:
        raise ValueError("cache_slot_count must be positive")
    context_capacity = (
        int(active_slots.shape[1])
        if max_context_len is None
        else int(max_context_len)
    )
    if not 0 < context_capacity <= int(active_slots.shape[1]):
        raise ValueError(
            "MLA decode max_context_len must be within the active-slot width: "
            f"max_context_len={context_capacity} "
            f"active_slot_width={int(active_slots.shape[1])}."
        )

    # Shape, dtype, device, and caller-owned capacity are graph-static and safe
    # to validate during capture. Per-row value checks below require GPU-to-host
    # reads and therefore remain eager-only.
    if device_runtime.is_stream_capturing():
        return

    request_rows = request_indices.tolist()
    lengths = context_lens.tolist()
    real_request_rows: list[int] = []
    for batch_index, (request_row, length) in enumerate(
        zip(request_rows, lengths)
    ):
        if length < 0 or length > context_capacity:
            raise ValueError(
                f"context_lens[{batch_index}]={length} is outside "
                f"[0, {context_capacity}]"
            )
        if request_row < 0:
            if length != 0:
                raise ValueError(
                    "padded request rows must have zero context length"
                )
            continue
        if request_row >= active_slots.shape[0]:
            raise ValueError(
                f"request_indices[{batch_index}]={request_row} is outside "
                f"[0, {active_slots.shape[0]})"
            )
        if batch_index < valid_batch_size:
            real_request_rows.append(request_row)
        if length == 0:
            continue
        slots = active_slots[request_row, :length]
        if bool(torch.any(slots < 0).item()) or bool(
            torch.any(slots >= cache_slot_count).item()
        ):
            raise ValueError(
                f"active_slots row {request_row} contains an invalid slot"
            )
        if slots.unique().numel() != slots.numel():
            raise ValueError(
                f"active_slots row {request_row} contains duplicate slots"
            )
    if len(set(real_request_rows)) != len(real_request_rows):
        raise ValueError("request_indices contains duplicate non-padding rows")


@torch.no_grad()
def run_mla_decode(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    active_slots: torch.Tensor,
    request_indices: torch.Tensor,
    context_lens: torch.Tensor,
    output: torch.Tensor,
    workspace: MlaDecodeWorkspace,
    *,
    softmax_scale: float,
    attn_score: torch.Tensor | None = None,
    max_context_len: int | None = None,
    config: MlaDecodeLaunchConfig = DEFAULT_GLM_MLA_DECODE_CONFIG,
    validate_metadata: bool = True,
) -> torch.Tensor:
    """Run GLM MLA decode using only explicit tensors and static config."""

    _validate_schedule_workspace(context_lens, workspace, config)
    if q_latent.ndim != 3:
        raise ValueError("q_latent must have shape [batch, heads, 512]")
    batch_size, head_count = q_latent.shape[:2]
    if context_lens.numel() != batch_size:
        raise ValueError(
            "q_latent batch size and context_lens length must match: "
            f"{batch_size} != {context_lens.numel()}"
        )
    if output.shape != q_latent.shape:
        raise ValueError(
            f"output must have shape {tuple(q_latent.shape)}, got "
            f"{tuple(output.shape)}"
        )
    if output.device != q_latent.device:
        raise ValueError(
            f"output is on {output.device}, expected {q_latent.device}"
        )
    if output.dtype != torch.bfloat16:
        raise TypeError(
            f"output must use {torch.bfloat16}, got {output.dtype}"
        )
    if workspace.mid_output.shape[0] < head_count:
        raise ValueError(
            "MLA decode workspace is too small: "
            f"heads={workspace.mid_output.shape[0]}, required={head_count}"
        )
    softmax_scale = float(softmax_scale)
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError(
            f"softmax_scale must be finite and positive, got {softmax_scale}."
        )
    if validate_metadata:
        validate_mla_decode_metadata(
            active_slots,
            request_indices,
            context_lens,
            cache_slot_count=latent_cache.shape[0],
            max_context_len=max_context_len,
        )

    # The 2D score path uses atomic_max across query-head tiles. Reset inside
    # the captured execution so replay can never inherit a previous step's max.
    # The same reset also keeps padded/tail positions explicit for 3D scores.
    if attn_score is not None:
        attn_score.fill_(-1.0e20)

    prepare_mla_decode_schedule(context_lens, workspace, config=config)
    decode_stage1(
        q_latent,
        q_rope,
        latent_cache,
        rope_cache,
        active_slots,
        request_indices,
        context_lens,
        workspace.block_size,
        workspace.mid_output,
        workspace.mid_logsumexp,
        attn_score=attn_score,
        max_context_len=max_context_len,
        softmax_scale=softmax_scale,
        program_count=config.program_count,
        block_q_heads=config.block_q_heads,
        block_n=config.block_n,
        pipeline_stages=config.stage1_pipeline_stages,
        num_warps=config.stage1_num_warps,
    )
    decode_stage2(
        workspace.block_size,
        workspace.batch_start_indices,
        context_lens,
        workspace.mid_output,
        workspace.mid_logsumexp,
        output,
        pipeline_stages=config.stage2_pipeline_stages,
        num_warps=config.stage2_num_warps,
    )
    return output
