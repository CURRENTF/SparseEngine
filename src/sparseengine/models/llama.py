from typing import Any

from sparseengine.models.standard_mha import (
    StandardAttention,
    StandardDecoderLayer,
    StandardForCausalLM,
    StandardMLP,
    StandardModel,
)


class LlamaAttention(StandardAttention):
    infer_head_dim = False
    output_bias_from_qkv = True

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        head_dim: int,
        qkv_bias: bool,
        rope_theta: float,
        rope_scaling: tuple[tuple[str, object], ...] | None,
        proj_chunk_size: int = 16384,
        quantization=None,
    ) -> None:
        super().__init__(
            hidden_size, num_heads, num_kv_heads, max_position, head_dim,
            qkv_bias=qkv_bias, rope_theta=rope_theta, rope_scaling=rope_scaling,
            proj_chunk_size=proj_chunk_size, quantization=quantization,
        )


class LlamaMLP(StandardMLP):
    def __init__(
        self, hidden_size: int, intermediate_size: int, hidden_act: str,
        mlp_bias: bool, mlp_chunk_size: int = 16384, quantization=None,
    ) -> None:
        super().__init__(
            hidden_size, intermediate_size, hidden_act, mlp_chunk_size, quantization,
            mlp_bias=mlp_bias,
        )

    @staticmethod
    def _check_hidden_act(hidden_act: str) -> None:
        if hidden_act != "silu":
            raise NotImplementedError(f"Unsupported Llama hidden_act={hidden_act!r}.")


class LlamaDecoderLayer(StandardDecoderLayer):
    model_name = "Llama"
    mlp_before_norm = True
    mlp_cls = LlamaMLP
    attention_cls = LlamaAttention

    def __init__(self, config: Any) -> None:
        super().__init__(
            config,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        )

    def _build_mlp(self, config: Any) -> StandardMLP:
        return super()._build_mlp(config, mlp_bias=getattr(config, "mlp_bias", False))


class LlamaModel(StandardModel):
    def __init__(self, config: Any) -> None:
        super().__init__(config, LlamaDecoderLayer)


class LlamaForCausalLM(StandardForCausalLM):
    model_name = "Llama"
    model_cls = LlamaModel
    packed_modules_mapping = StandardForCausalLM.packed_modules_mapping.copy()

    prepare_loaded_weights = StandardForCausalLM._prepare_loaded_weights
