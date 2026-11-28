from transformers import Qwen3Config

from sparseengine.models.standard_mha import (
    StandardAttention,
    StandardDecoderLayer,
    StandardForCausalLM,
    StandardMLP,
    StandardModel,
)


class Qwen3Attention(StandardAttention):
    normalize_qk = True


class Qwen3MLP(StandardMLP):
    pass


class Qwen3DecoderLayerBase(StandardDecoderLayer):
    model_name = "Qwen3"
    mlp_before_norm = False
    attention_cls = Qwen3Attention

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(
            config, qkv_bias=config.attention_bias, head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
        )


class Qwen3DecoderLayer(Qwen3DecoderLayerBase):
    mlp_cls = Qwen3MLP

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config)
        self.mlp = self._build_mlp(config)


class Qwen3ModelBase(StandardModel):
    debug_add_residual = True


class Qwen3Model(Qwen3ModelBase):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config, Qwen3DecoderLayer)


class Qwen3ForCausalLM(StandardForCausalLM):
    model_name = "Qwen3"
    model_cls = Qwen3Model
    packed_modules_mapping = StandardForCausalLM.packed_modules_mapping.copy()

    prepare_loaded_weights = StandardForCausalLM._prepare_loaded_weights
