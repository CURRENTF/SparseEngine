from __future__ import annotations

from collections.abc import Mapping

from sparseengine.distributed.topology import ParallelTopology


def _validate_divisible(
    model_name: str,
    fields: Mapping[str, int],
    divisor: int,
    parallelism: str,
) -> None:
    invalid = {
        name: int(value)
        for name, value in fields.items()
        if int(value) <= 0 or int(value) % divisor
    }
    if invalid:
        raise ValueError(
            f"{model_name} dimensions must be positive and divisible by "
            f"{parallelism}={divisor}, invalid={invalid}."
        )


def validate_model_sharding(
    topology: ParallelTopology,
    *,
    model_name: str,
    attention_fields: Mapping[str, int],
    num_experts: int | None = None,
    moe_fields: Mapping[str, int] | None = None,
) -> None:
    _validate_divisible(
        model_name,
        attention_fields,
        topology.attn_tp_size,
        "attention TP",
    )
    if num_experts is None:
        return
    num_experts = int(num_experts)
    if num_experts <= 0 or num_experts % topology.moe_ep_size:
        raise ValueError(
            f"{model_name} num_experts must be positive and divisible by "
            f"EP={topology.moe_ep_size}, got {num_experts}."
        )
    if moe_fields:
        _validate_divisible(
            model_name,
            moe_fields,
            topology.moe_tp_size,
            "MoE TP",
        )


def validate_top_k(model_name: str, top_k: int, num_experts: int) -> None:
    if not 1 <= int(top_k) <= int(num_experts):
        raise ValueError(
            f"{model_name} top_k must be in [1, num_experts], "
            f"got top_k={top_k}, num_experts={num_experts}."
        )
