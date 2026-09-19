from __future__ import annotations

import torch
from torch import nn

from sparseengine.operators.gated_delta_rule import (
    GatedDeltaRuleOpSpec,
    PreparedGatedDeltaRuleOp,
    prepare_gated_delta_rule_op,
)
from sparseengine.operators.moe import model_activation_dtype


def build_gated_delta_rule_op(
    config,
    *,
    attention_tp_size: int,
    device: torch.device,
    cuda_graph: bool,
) -> PreparedGatedDeltaRuleOp:
    tp_size = int(attention_tp_size)
    total_key_heads = int(config.linear_num_key_heads)
    total_value_heads = int(config.linear_num_value_heads)
    if total_key_heads % tp_size or total_value_heads % tp_size:
        raise ValueError(
            "GDN key/value heads must be divisible by attention TP size: "
            f"key_heads={total_key_heads} value_heads={total_value_heads} "
            f"tp_size={tp_size}."
        )
    recurrent_state_dtype = getattr(
        config,
        "runtime_recurrent_state_dtype",
        config.dtype,
    )
    if not isinstance(recurrent_state_dtype, torch.dtype):
        raise TypeError(
            "GDN runtime recurrent-state dtype must be a torch.dtype, got "
            f"{recurrent_state_dtype!r}."
        )
    return prepare_gated_delta_rule_op(
        GatedDeltaRuleOpSpec(
            num_key_heads=total_key_heads // tp_size,
            num_value_heads=total_value_heads // tp_size,
            key_head_dim=int(config.linear_key_head_dim),
            value_head_dim=int(config.linear_value_head_dim),
            activation_dtype=model_activation_dtype(config),
            recurrent_state_dtype=recurrent_state_dtype,
            cuda_graph_decode=bool(cuda_graph),
        ),
        device_index=int(device.index or 0),
    )


def bind_gated_delta_rule_op(
    model: nn.Module,
    gated_delta_rule_op: PreparedGatedDeltaRuleOp,
) -> int:
    # Avoid importing the model class here; the structural marker also keeps
    # this binder usable by future Qwen GDN variants.
    linear_attention_layers = [
        module
        for module in model.modules()
        if getattr(module, "is_gated_delta_rule_layer", False)
    ]
    if not linear_attention_layers:
        raise ValueError("Cannot bind GDN operator: model has no GDN layers.")
    for layer in linear_attention_layers:
        bind = getattr(layer, "bind_gated_delta_rule_op", None)
        if callable(bind):
            bind(gated_delta_rule_op)
        else:
            layer.gated_delta_rule_op = gated_delta_rule_op
    return len(linear_attention_layers)


__all__ = [
    "bind_gated_delta_rule_op",
    "build_gated_delta_rule_op",
]
