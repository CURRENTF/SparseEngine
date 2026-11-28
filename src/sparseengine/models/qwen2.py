from transformers import Qwen2Config

from sparseengine.models.standard_mha import (
    StandardAttention,
    StandardDecoderLayer,
    StandardForCausalLM,
    StandardMLP,
    StandardModel,
)


class Qwen2Attention(StandardAttention):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = True,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
        proj_chunk_size: int = 16384,
        quantization=None,
    ) -> None:
        super().__init__(
            hidden_size, num_heads, num_kv_heads, max_position, head_dim,
            rms_norm_eps, qkv_bias, rope_theta, rope_scaling, proj_chunk_size, quantization,
        )


class Qwen2MLP(StandardMLP):
    def __init__(
        self, hidden_size: int, intermediate_size: int, hidden_act: str,
        mlp_chunk_size: int = 16384, quantization=None,
    ) -> None:
        super().__init__(hidden_size, intermediate_size, hidden_act, mlp_chunk_size, quantization)


class Qwen2DecoderLayer(StandardDecoderLayer):
    model_name = "Qwen2"
    mlp_before_norm = True
    mlp_cls = Qwen2MLP
    attention_cls = Qwen2Attention

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__(
            config,
            qkv_bias=getattr(config, "attention_bias", True),
            head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            rms_norm_eps=config.rms_norm_eps,
        )


class Qwen2Model(StandardModel):
    debug_add_residual = True

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__(config, Qwen2DecoderLayer)


class Qwen2ForCausalLM(StandardForCausalLM):
    model_name = "Qwen2"
    model_cls = Qwen2Model
    packed_modules_mapping = StandardForCausalLM.packed_modules_mapping.copy()
