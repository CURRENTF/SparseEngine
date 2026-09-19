# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2023-2026 SGLang Team
"""Unquantized fused MoE kernels adapted from SGLang.

Source: sgl-project/sglang@24d625698d44c78f6e8ab8b7c19f96f45bbaa90a
``python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py`` and
``python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py``.

This local port keeps the BF16/FP16 routed-GEMM path and integrates it with
Sparse-Engine's provider-owned alignment and reduction contracts. Quantized,
LoRA, TMA, and fused-collective branches remain owned by their existing
providers.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import lru_cache
from importlib.resources import files

import torch
import triton
import triton.language as tl

from sparseengine.kernels.moe import MoeAlignment
from sparseengine.kernels.triton.moe import (
    _validate_fused_moe_inputs,
    moe_sum,
)
from sparseengine.kernels.triton.silu_and_mul import silu_and_mul_fwd
from sparseengine.utils.device_name import device_name_contains


@triton.jit(do_not_specialize=["EM", "num_valid_tokens"])
def _sgl_fused_moe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    TOP_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    token_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    assignment_ids = tl.load(sorted_token_ids_ptr + token_offsets).to(tl.int64)
    assignment_mask = assignment_ids < num_valid_tokens
    expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if expert_id == -1:
        return

    n_offsets = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = (
        a_ptr
        + (assignment_ids // TOP_K)[:, None] * stride_am
        + k_offsets[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + expert_id * stride_be
        + k_offsets[:, None] * stride_bk
        + n_offsets[None, :] * stride_bn
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=assignment_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            remaining = K - k_start
            a = tl.load(
                a_ptrs,
                mask=assignment_mask[:, None]
                & (k_offsets[None, :] < remaining),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=k_offsets[:, None] < remaining,
                other=0.0,
            )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        routed_weight = tl.load(
            topk_weights_ptr + assignment_ids,
            mask=assignment_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator *= routed_weight[:, None]

    output_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = (
        c_ptr
        + assignment_ids[:, None] * stride_cm
        + output_offsets[None, :] * stride_cn
    )
    tl.store(
        output_ptrs,
        accumulator.to(b_ptr.dtype.element_ty),
        mask=assignment_mask[:, None] & (output_offsets[None, :] < N),
    )


def _config(
    block_m: int,
    block_n: int,
    block_k: int,
    group_m: int,
    num_warps: int,
    num_stages: int,
) -> dict[str, int]:
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


_QWEN3_PROFILE_RESOURCE = "profiles/sgl_h100_qwen3_bf16.json"
_GLM47_PROFILE_RESOURCE = "profiles/sgl_h100_glm47_bf16.json"
_PROFILE_CONFIG_KEYS = (
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
)


def _load_sgl_profile_resource(
    resource_name: str,
) -> tuple[
    dict[str, object],
    dict[int, dict[int, dict[str, int]]],
]:
    resource = files("sparseengine.kernels.triton").joinpath(resource_name)
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Failed to load SGL MoE profile {resource_name}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError("SGL MoE profile root must be a mapping.")
    if payload.get("schema_version") != 1:
        raise RuntimeError(
            "Unsupported SGL MoE profile schema: "
            f"{payload.get('schema_version')!r}."
        )
    contract = payload.get("contract")
    provenance = payload.get("provenance")
    raw_tables = payload.get("tables")
    if not isinstance(contract, dict) or not isinstance(provenance, dict):
        raise RuntimeError("SGL MoE profile is missing contract or provenance.")
    if not isinstance(payload.get("profile_id"), str) or not isinstance(
        payload.get("kernel"), str
    ):
        raise RuntimeError("SGL MoE profile must identify its profile and kernel.")
    if payload.get("toolchain") not in (
        {"triton": ">=3.5,<4"},
        {"triton": "3.6.0"},
    ):
        raise RuntimeError(
            f"Unsupported SGL MoE profile toolchain: {payload.get('toolchain')!r}."
        )
    if not isinstance(raw_tables, dict):
        raise RuntimeError("SGL MoE profile tables must be a mapping.")
    tables: dict[int, dict[int, dict[str, int]]] = {}
    for intermediate_size, rows in raw_tables.items():
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(
                f"SGL MoE profile table {intermediate_size!r} is empty."
            )
        table: dict[int, dict[str, int]] = {}
        for row in rows:
            if not isinstance(row, list) or len(row) != 7:
                raise RuntimeError(
                    f"Invalid SGL MoE profile row for {intermediate_size}: {row!r}."
                )
            tokens, *values = (int(value) for value in row)
            if tokens <= 0 or tokens in table or any(value <= 0 for value in values):
                raise RuntimeError(
                    f"Invalid SGL MoE profile values for {intermediate_size}: {row!r}."
                )
            table[tokens] = dict(zip(_PROFILE_CONFIG_KEYS, values))
        parsed_intermediate_size = int(intermediate_size)
        if parsed_intermediate_size in tables:
            raise RuntimeError(
                f"Duplicate SGL MoE intermediate size {parsed_intermediate_size}."
            )
        tables[parsed_intermediate_size] = table
    return payload, tables


def _validate_profile_contract(
    payload: dict[str, object],
    expected_contract: dict[str, object],
) -> None:
    contract = payload["contract"]
    assert isinstance(contract, dict)
    mismatches = {
        key: (contract.get(key), expected)
        for key, expected in expected_contract.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"SGL MoE profile contract mismatch: {mismatches}.")


@lru_cache(maxsize=1)
def _load_sgl_h100_qwen3_profile() -> tuple[
    dict[str, object],
    dict[int, dict[int, dict[str, int]]],
]:
    payload, tables = _load_sgl_profile_resource(_QWEN3_PROFILE_RESOURCE)
    _validate_profile_contract(
        payload,
        {
            "device_name_keyword": "H100",
            "compute_capability": [9, 0],
            "activation_dtype": "bfloat16",
            "weight_dtype": "bfloat16",
            "num_local_experts": 128,
            "hidden_size": 2048,
            "top_k": 8,
            "ep_size": 1,
            "weight_layout_id": "packed_gate_up_v1",
            "stages": ["w13", "w2"],
            "tp_size_by_intermediate_size": {"768": 1, "384": 2, "192": 4},
        },
    )
    return payload, tables


@lru_cache(maxsize=1)
def _load_sgl_h100_glm47_profile() -> tuple[
    dict[str, object],
    dict[int, dict[int, dict[str, int]]],
]:
    payload, tables = _load_sgl_profile_resource(_GLM47_PROFILE_RESOURCE)
    _validate_profile_contract(
        payload,
        {
            "device_name_keyword": "H100",
            "compute_capability": [9, 0],
            "activation_dtype": "bfloat16",
            "weight_dtype": "bfloat16",
            "num_local_experts": 65,
            "hidden_size": 2048,
            "intermediate_size": 1536,
            "top_k": 5,
            "tp_size": 1,
            "ep_size": 1,
            "weight_layout_id": "packed_gate_up_v1",
            "stages": ["w13", "w2"],
        },
    )
    return payload, tables


def sgl_moe_profile_support() -> tuple[bool, str]:
    payload, _ = _load_sgl_h100_qwen3_profile()
    raw_version = str(triton.__version__).split("+", 1)[0]
    try:
        major, minor = (int(part) for part in raw_version.split(".")[:2])
    except (TypeError, ValueError):
        return False, f"cannot parse Triton version {triton.__version__!r}"
    if major != 3 or minor < 5:
        return (
            False,
            f"profile {payload['profile_id']} requires Triton >=3.5,<4, "
            f"got {triton.__version__}",
        )
    return True, f"profile {payload['profile_id']} matches Triton {triton.__version__}"


def sgl_glm47_moe_profile_support() -> tuple[bool, str]:
    payload, _ = _load_sgl_h100_glm47_profile()
    raw_version = str(triton.__version__).split("+", 1)[0]
    try:
        major, minor = (int(part) for part in raw_version.split(".")[:2])
    except (TypeError, ValueError):
        return False, f"cannot parse Triton version {triton.__version__!r}"
    if major != 3 or minor < 5:
        return (
            False,
            f"profile {payload['profile_id']} requires Triton >=3.5,<4, "
            f"got {triton.__version__}",
        )
    return True, f"profile {payload['profile_id']} matches Triton {triton.__version__}"


def sgl_moe_profile_metadata() -> dict[str, object]:
    payload, _ = _load_sgl_h100_qwen3_profile()
    provenance = payload["provenance"]
    return {
        "profile_id": payload["profile_id"],
        "profile_status": "tuned",
        "profile_source": {
            "kind": provenance["kind"],
            "source_repository": provenance["source_repository"],
            "source_revision": provenance["source_revision"],
            "source_paths": list(provenance["source_paths"]),
        },
        "kernel": payload["kernel"],
    }


def sgl_glm47_moe_profile_metadata() -> dict[str, object]:
    payload, _ = _load_sgl_h100_glm47_profile()
    provenance = payload["provenance"]
    return {
        "profile_id": payload["profile_id"],
        "profile_status": payload["profile_status"],
        "profile_source": {
            "kind": provenance["kind"],
            "source_repository": provenance["source_repository"],
            "source_revision": provenance["source_revision"],
            "source_paths": list(provenance["source_paths"]),
        },
        "kernel": payload["kernel"],
    }


def resolve_sgl_moe_config(
    *,
    num_tokens: int,
    top_k: int,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    ep_size: int,
    device_name: str,
    device_capability: tuple[int, int],
) -> dict[str, int]:
    """Resolve an offline SGL profile or its generic unquantized heuristic."""

    num_tokens = int(num_tokens)
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}.")
    glm_payload, glm_tables = _load_sgl_h100_glm47_profile()
    glm_contract = glm_payload["contract"]
    glm_profile_supported, _ = sgl_glm47_moe_profile_support()
    if (
        glm_profile_supported
        and device_name_contains(device_name, glm_contract["device_name_keyword"])
        and tuple(device_capability) == tuple(glm_contract["compute_capability"])
        and activation_dtype == torch.bfloat16
        and weight_dtype == torch.bfloat16
        and int(ep_size) == int(glm_contract["ep_size"])
        and int(top_k) == int(glm_contract["top_k"])
        and int(num_local_experts) == int(glm_contract["num_local_experts"])
        and int(hidden_size) == int(glm_contract["hidden_size"])
        and int(intermediate_size) == int(glm_contract["intermediate_size"])
    ):
        table = glm_tables[int(intermediate_size)]
        bucket = min(table, key=lambda value: abs(value - num_tokens))
        return dict(table[bucket])

    payload, tables = _load_sgl_h100_qwen3_profile()
    contract = payload["contract"]
    profile_supported, _ = sgl_moe_profile_support()
    table = None
    if (
        profile_supported
        and device_name_contains(device_name, contract["device_name_keyword"])
        and tuple(device_capability) == tuple(contract["compute_capability"])
        and activation_dtype == torch.bfloat16
        and weight_dtype == torch.bfloat16
        and int(ep_size) == int(contract["ep_size"])
        and int(top_k) == int(contract["top_k"])
        and int(num_local_experts) == int(contract["num_local_experts"])
        and int(hidden_size) == int(contract["hidden_size"])
    ):
        table = tables.get(int(intermediate_size))
    if table:
        bucket = min(table, key=lambda value: abs(value - num_tokens))
        return dict(table[bucket])
    if num_tokens * int(top_k) <= 32:
        return _config(16, 128, 32, 8, 4, 4)
    return _config(16, 64, 64, 8, 4, 3)


def _run_sgl_routed_gemm(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    alignment: MoeAlignment,
    *,
    input_top_k: int,
    multiply_routing_weight: bool,
    config: dict[str, int],
) -> None:
    if alignment.naive or alignment.sorted_token_ids is None:
        raise ValueError("The SGL fused MoE kernel requires grouped alignment metadata.")
    block_m = int(config["BLOCK_SIZE_M"])
    block_n = int(config["BLOCK_SIZE_N"])
    em = int(alignment.sorted_token_ids.numel())
    grid = (
        triton.cdiv(em, block_m) * triton.cdiv(int(weights.shape[1]), block_n),
    )
    _sgl_fused_moe_kernel[grid](
        inputs,
        weights,
        output,
        topk_weights,
        alignment.sorted_token_ids,
        alignment.expert_ids,
        alignment.num_tokens_post_padded,
        N=int(weights.shape[1]),
        K=int(weights.shape[2]),
        EM=em,
        num_valid_tokens=int(topk_weights.numel()),
        stride_am=inputs.stride(0),
        stride_ak=inputs.stride(1),
        stride_be=weights.stride(0),
        stride_bk=weights.stride(2),
        stride_bn=weights.stride(1),
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        TOP_K=int(input_top_k),
        MUL_ROUTED_WEIGHT=bool(multiply_routing_weight),
        EVEN_K=int(weights.shape[2]) % int(config["BLOCK_SIZE_K"]) == 0,
        **config,
    )


def sgl_fused_moe(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    local_expert_start: int,
    alignment_impl: Callable[..., MoeAlignment],
) -> torch.Tensor:
    """Run SGLang's unquantized routed-GEMM pipeline."""

    num_experts = int(num_experts)
    local_expert_start = int(local_expert_start)
    _validate_fused_moe_inputs(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts,
        local_expert_start,
    )
    num_tokens = int(hidden_states.shape[0])
    top_k = int(topk_ids.shape[1])
    num_assignments = int(topk_ids.numel())
    num_local_experts = int(w13_weight.shape[0])
    intermediate_size = int(w13_weight.shape[1]) // 2
    hidden_size = int(hidden_states.shape[1])
    config = resolve_sgl_moe_config(
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation_dtype=hidden_states.dtype,
        weight_dtype=w13_weight.dtype,
        ep_size=num_experts // num_local_experts,
        device_name=torch.cuda.get_device_name(hidden_states.device),
        device_capability=torch.cuda.get_device_capability(hidden_states.device),
    )
    alignment = alignment_impl(
        topk_ids,
        block_size=config["BLOCK_SIZE_M"],
        num_experts=num_experts,
        local_expert_start=local_expert_start,
        local_expert_end=local_expert_start + num_local_experts,
    )

    gate_up = torch.empty(
        (num_assignments, 2 * intermediate_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    _run_sgl_routed_gemm(
        hidden_states,
        w13_weight,
        gate_up,
        topk_weights,
        alignment,
        input_top_k=top_k,
        multiply_routing_weight=False,
        config=config,
    )
    activated = silu_and_mul_fwd(gate_up)
    routed_output = torch.empty(
        (num_assignments, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    _run_sgl_routed_gemm(
        activated,
        w2_weight,
        routed_output,
        topk_weights,
        alignment,
        input_top_k=1,
        multiply_routing_weight=True,
        config=config,
    )
    return moe_sum(
        routed_output.view(num_tokens, top_k, hidden_size),
        topk_ids,
        num_experts=num_experts,
        local_expert_start=local_expert_start,
        local_expert_end=local_expert_start + num_local_experts,
    )


__all__ = [
    "resolve_sgl_moe_config",
    "sgl_fused_moe",
    "sgl_glm47_moe_profile_metadata",
    "sgl_glm47_moe_profile_support",
    "sgl_moe_profile_metadata",
    "sgl_moe_profile_support",
]
