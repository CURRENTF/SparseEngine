from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sparseengine.distributed.sharding import validate_model_sharding, validate_top_k
from sparseengine.distributed.topology import ParallelTopology
from sparseengine.utils.config import config_get


@dataclass(frozen=True)
class ModelSpec:
    name: str
    requires_fp8: bool = False
    mixed_attention: bool = False
    allow_raw_config: bool = False
    supports_tiny_random: bool = True
    supports_quantized_tiny_random: bool = False
    tiny_random_requires_standard_head_shape: bool = True
    supports_expert_parallel: bool = False
    supports_data_parallel: bool = False
    prefix_cache_block_size_multiple: int | None = None
    deltakv_checkpoint_model_types: frozenset[str] = frozenset()
    runtime_class_name: str = ""
    attention_cache_layout: str = "explicit_kv"
    attention_tp_fields: tuple[str, ...] = ()
    num_experts_field: str | None = None
    moe_tp_fields: tuple[str, ...] = ()
    top_k_field: str | None = None

    def validate_parallel_execution(self, topology: ParallelTopology) -> None:
        """Reject unimplemented execution paths without changing topology semantics."""
        if topology.moe_ep_size > 1 and not self.supports_expert_parallel:
            raise ValueError(
                f"{self.name} does not support expert parallelism, "
                f"got EP={topology.moe_ep_size}."
            )
        if topology.attn_dp_size > 1 and not self.supports_data_parallel:
            raise ValueError(
                f"{self.name} does not support data parallelism, "
                f"got DP={topology.attn_dp_size}."
            )
        if topology.attn_dp_size > 1 and topology.moe_tp_size != 1:
            raise ValueError(
                "The engine currently supports DP attention only with "
                "MoE TP=1 (EP=world size); "
                f"got attention DP={topology.attn_dp_size}, TP={topology.attn_tp_size}, "
                f"MoE EP={topology.moe_ep_size}, TP={topology.moe_tp_size}."
            )

    def validate_sharding(self, hf_config: Any, topology: ParallelTopology) -> None:
        raw_num_experts = (
            config_get(hf_config, self.num_experts_field, None)
            if self.num_experts_field
            else None
        )
        num_experts = int(raw_num_experts) if raw_num_experts is not None else None
        validate_model_sharding(
            topology,
            model_name=self.name,
            attention_fields={
                field: int(value)
                for field in self.attention_tp_fields
                if (value := config_get(hf_config, field, None)) is not None
            },
            num_experts=num_experts,
            moe_fields={
                field: int(value)
                for field in self.moe_tp_fields
                if (value := config_get(hf_config, field, None)) is not None
            },
        )
        top_k = config_get(hf_config, self.top_k_field, None) if self.top_k_field else None
        if top_k is not None and num_experts is not None:
            validate_top_k(
                self.name,
                int(top_k),
                num_experts,
            )


_DENSE_TP_FIELDS = (
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "intermediate_size",
)
_MOE_TP_FIELDS = ("num_attention_heads", "num_key_value_heads", "vocab_size")
_QWEN35_TP_FIELDS = (
    "num_attention_heads",
    "num_key_value_heads",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "vocab_size",
)


MODEL_SPECS = {
    model_type: ModelSpec(
        name,
        runtime_class_name=runtime_class_name,
        attention_tp_fields=_DENSE_TP_FIELDS,
    )
    for model_type, (name, runtime_class_name) in {
        "qwen2": ("Qwen2", "Qwen2ForCausalLM"),
        "qwen3": ("Qwen3", "Qwen3ForCausalLM"),
        "llama": ("Llama", "LlamaForCausalLM"),
    }.items()
}
MODEL_SPECS.update(
    {
        "qwen3_5": ModelSpec(
            "Qwen3.5",
            mixed_attention=True,
            allow_raw_config=True,
            supports_tiny_random=False,
            prefix_cache_block_size_multiple=4096,
            deltakv_checkpoint_model_types=frozenset({"qwen3_5"}),
            runtime_class_name="Qwen35ForCausalLM",
            attention_tp_fields=_QWEN35_TP_FIELDS,
        ),
        "qwen3_moe": ModelSpec(
            "Qwen3MoE",
            supports_expert_parallel=True,
            supports_data_parallel=True,
            runtime_class_name="Qwen3MoeForCausalLM",
            attention_tp_fields=_MOE_TP_FIELDS,
            num_experts_field="num_experts",
            moe_tp_fields=("moe_intermediate_size",),
            top_k_field="num_experts_per_tok",
        ),
        "qwen3_5_moe": ModelSpec(
            "Qwen3.6 MoE",
            mixed_attention=True,
            allow_raw_config=True,
            supports_tiny_random=False,
            supports_expert_parallel=True,
            prefix_cache_block_size_multiple=4096,
            deltakv_checkpoint_model_types=frozenset({"qwen3_5"}),
            runtime_class_name="Qwen35MoeForCausalLM",
            attention_tp_fields=(
                *_QWEN35_TP_FIELDS,
                "shared_expert_intermediate_size",
            ),
            num_experts_field="num_experts",
            moe_tp_fields=("moe_intermediate_size",),
            top_k_field="num_experts_per_tok",
        ),
        "minimax_m2": ModelSpec(
            "MiniMax M2.7",
            supports_data_parallel=True,
            requires_fp8=True,
            supports_quantized_tiny_random=True,
            tiny_random_requires_standard_head_shape=False,
            supports_expert_parallel=True,
            runtime_class_name="MiniMaxM2ForCausalLM",
            attention_tp_fields=_MOE_TP_FIELDS,
            num_experts_field="num_local_experts",
            moe_tp_fields=("intermediate_size",),
            top_k_field="num_experts_per_tok",
        ),
        "glm4_moe_lite": ModelSpec(
            "GLM-4.7-Flash",
            tiny_random_requires_standard_head_shape=False,
            supports_expert_parallel=True,
            supports_data_parallel=True,
            runtime_class_name="Glm4MoeLiteForCausalLM",
            attention_cache_layout="mla_latent",
            attention_tp_fields=("num_attention_heads", "vocab_size", "intermediate_size"),
            num_experts_field="n_routed_experts",
            moe_tp_fields=("moe_intermediate_size",),
            top_k_field="num_experts_per_tok",
        ),
        "gemma4": ModelSpec(
            "Gemma 4",
            allow_raw_config=True,
            supports_tiny_random=False,
            supports_expert_parallel=True,
            runtime_class_name="Gemma4ForCausalLM",
            attention_tp_fields=(
                "num_attention_heads",
                "vocab_size",
                "intermediate_size",
            ),
            num_experts_field="num_experts",
            moe_tp_fields=("moe_intermediate_size",),
            top_k_field="top_k_experts",
        ),
    }
)


def resolve_model_spec(model_type: str) -> ModelSpec:
    if model_type not in MODEL_SPECS:
        supported = ", ".join(sorted(MODEL_SPECS))
        raise NotImplementedError(
            f"Unsupported Sparse-Engine model_type={model_type!r}; supported: {supported}."
        )
    return MODEL_SPECS[model_type]
