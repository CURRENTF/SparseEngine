from __future__ import annotations

import re
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Gemma4TextConfig

from sparseengine.distributed import get_parallel_context
from sparseengine.distributed.moe_communication import AllReduceMoeCommunication
from sparseengine.layers.attention import Attention
from sparseengine.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from sparseengine.layers.gemma4_rmsnorm import Gemma4RMSNorm
from sparseengine.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedKVQKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sparseengine.layers.rotary_embedding import apply_rotary_emb
from sparseengine.operators.gemma4 import (
    Gemma4OperatorProvider,
    Gemma4OpSpec,
    resolve_gemma4_provider,
)
from sparseengine.operators.gemma4_router import (
    Gemma4RouterOpSpec,
    Gemma4RouterProvider,
    resolve_gemma4_router_provider,
)
from sparseengine.operators.gemma4_moe import Gemma4PackedExperts
from sparseengine.operators.moe import model_activation_dtype
from sparseengine.operators.moe_execution import MoeExecutionPlan
from sparseengine.platforms import device_runtime
from sparseengine.utils.context import get_context
from sparseengine.utils.config import config_get, config_layer_get
from sparseengine.utils.weight_target import WeightTarget

_EXPERT_SOURCE_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.experts\.(gate_up_proj|down_proj)$"
)
_EXPERT_TARGET_RE = re.compile(
    r"^model\.layers\.(\d+)\.experts\.(gate_up_proj|down_proj)\.expert_weight$"
)


class Gemma4RotaryEmbedding(nn.Module):
    def __init__(
        self,
        config: Gemma4TextConfig,
        layer_type: str,
        head_dim: int,
        parameters: dict | None = None,
    ) -> None:
        super().__init__()
        if parameters is None:
            parameters = config_layer_get(config, 0, "rope_parameters")
        parameters = dict(parameters.get(layer_type, parameters))
        rope_type = str(parameters.get("rope_type", "default"))
        if rope_type not in {"default", "proportional"}:
            raise NotImplementedError(f"Unsupported Gemma 4 RoPE type {rope_type!r}.")
        head_dim = int(head_dim)
        proportion = float(parameters.get("partial_rotary_factor", 1.0))
        rotated_pairs = int(proportion * head_dim // 2)
        inv_freq = 1.0 / (
            float(parameters["rope_theta"])
            ** (torch.arange(0, 2 * rotated_pairs, 2, dtype=torch.float32) / head_dim)
        )
        if rotated_pairs < head_dim // 2:
            inv_freq = F.pad(inv_freq, (0, head_dim // 2 - rotated_pairs))
        inv_freq.div_(float(parameters.get("factor", 1.0)))
        positions = torch.arange(
            int(config.max_position_embeddings), dtype=torch.float32
        )
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer(
            "cos_sin_cache",
            torch.cat((freqs.cos(), freqs.sin()), -1).unsqueeze(1),
            persistent=False,
        )

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.cos_sin_cache[positions].chunk(2, -1)
        return apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)

    def forward_query(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        cos, sin = self.cos_sin_cache[positions].chunk(2, -1)
        return apply_rotary_emb(query, cos, sin)


class _Gemma4QKVMixin:
    use_k_eq_v: bool

    def _copy_k_to_v(self, param: nn.Parameter) -> None:
        k_start = self.num_heads * self.head_size
        k = param.data.narrow(0, k_start, self.num_kv_heads * self.head_size)
        v = param.data.narrow(
            0, k_start + self.num_kv_heads * self.head_size, k.shape[0]
        )
        v.copy_(k)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):
        super().weight_loader(param, loaded_weight, loaded_shard_id)
        if self.use_k_eq_v and loaded_shard_id == "k":
            self._copy_k_to_v(param)


class Gemma4QKVParallelLinear(_Gemma4QKVMixin, QKVParallelLinear):
    def __init__(self, *args, use_k_eq_v: bool = False, **kwargs) -> None:
        self.use_k_eq_v = bool(use_k_eq_v)
        super().__init__(*args, **kwargs)


class Gemma4ReplicatedKVQKVParallelLinear(
    _Gemma4QKVMixin, ReplicatedKVQKVParallelLinear
):
    def __init__(self, *args, use_k_eq_v: bool = False, **kwargs) -> None:
        self.use_k_eq_v = bool(use_k_eq_v)
        super().__init__(*args, **kwargs)


class Gemma4QueryParallelLinear(ColumnParallelLinear):
    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        if loaded_shard_id not in {None, "q"}:
            raise ValueError(
                f"Gemma 4 shared-KV query received shard {loaded_shard_id!r}."
            )
        super().weight_loader(param, loaded_weight)


class Gemma4Attention(nn.Module):
    def __init__(
        self,
        config: Gemma4TextConfig,
        layer_idx: int,
        rotary_emb: Gemma4RotaryEmbedding,
        operator_provider: Gemma4OperatorProvider,
    ) -> None:
        super().__init__()
        parallel_context = get_parallel_context()
        tp_size = parallel_context.attn_tp_size
        self.layer_type = str(config.layer_types[layer_idx])
        self.is_sliding = self.layer_type == "sliding_attention"
        shared_start = int(config.num_hidden_layers) - int(config.num_kv_shared_layers)
        self.is_kv_shared_layer = layer_idx >= shared_start > 0
        self.sliding_window = int(config.sliding_window) if self.is_sliding else None
        self.head_dim = int(
            config_layer_get(
                config,
                layer_idx,
                "head_dim",
                "head_dim" if self.is_sliding else "global_head_dim",
            )
        )
        self.total_num_heads = int(config.num_attention_heads)
        self.num_heads = self.total_num_heads // tp_size
        self.use_k_eq_v = bool(config.attention_k_eq_v and not self.is_sliding)
        self.total_num_kv_heads = int(
            config_layer_get(
                config,
                layer_idx,
                "num_key_value_heads",
                "num_global_key_value_heads"
                if self.use_k_eq_v
                else "num_key_value_heads",
            )
        )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        if self.is_kv_shared_layer:
            self.qkv_proj = Gemma4QueryParallelLinear(
                config.hidden_size,
                self.total_num_heads * self.head_dim,
                bias=config.attention_bias,
                quantization=getattr(config, "quantization_config", None),
            )
        else:
            linear_cls = (
                Gemma4ReplicatedKVQKVParallelLinear
                if self.total_num_kv_heads < tp_size
                else Gemma4QKVParallelLinear
            )
            self.qkv_proj = linear_cls(
                config.hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=config.attention_bias,
                quantization=getattr(config, "quantization_config", None),
                use_k_eq_v=self.use_k_eq_v,
            )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            quantization=getattr(config, "quantization_config", None),
        )
        self.q_norm = Gemma4RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, provider=operator_provider
        )
        if not self.is_kv_shared_layer:
            self.k_norm = Gemma4RMSNorm(
                self.head_dim, eps=config.rms_norm_eps, provider=operator_provider
            )
            self.v_norm = Gemma4RMSNorm(
                self.head_dim,
                eps=config.rms_norm_eps,
                with_scale=False,
                provider=operator_provider,
            )
        self.rotary_emb = rotary_emb
        self._ops = operator_provider
        self.attn = Attention(self.num_heads, self.head_dim, 1.0, self.num_kv_heads)
        self.attn.attention_backend = operator_provider.attention_backend(
            sliding_window=self.sliding_window
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self.is_kv_shared_layer:
            q = self.qkv_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
            q, _, _ = self._ops.qkv_norm_rope(
                q,
                None,
                None,
                self.q_norm.weight,
                None,
                self.rotary_emb.cos_sin_cache,
                positions,
                self.q_norm.eps,
            )
            empty = q.new_empty((0, self.num_kv_heads, self.head_dim))
            return self.o_proj(self.attn(q, empty, empty).flatten(1))
        q, k, v = self.qkv_proj(hidden_states).split(
            (self.q_size, self.kv_size, self.kv_size), -1
        )
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k, v = self._ops.qkv_norm_rope(
            q,
            k,
            v,
            self.q_norm.weight,
            self.k_norm.weight,
            self.rotary_emb.cos_sin_cache,
            positions,
            self.q_norm.eps,
        )
        context = get_context()
        context.cache_manager.save_rope_kv_if_needed(context.now_layer_idx, k, v)
        output = self.attn(q, k, v)
        return self.o_proj(output.flatten(1))


class Gemma4MLP(nn.Module):
    def __init__(
        self,
        config: Gemma4TextConfig,
        layer_idx: int,
        operator_provider: Gemma4OperatorProvider,
        *,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        shared_start = int(config.num_hidden_layers) - int(config.num_kv_shared_layers)
        width = int(config.intermediate_size) * (
            2
            if bool(config.use_double_wide_mlp) and layer_idx >= shared_start > 0
            else 1
        )
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [width, width],
            quantization=getattr(config, "quantization_config", None),
        )
        self.down_proj = RowParallelLinear(
            width,
            config.hidden_size,
            reduce_results=reduce_results,
            quantization=getattr(config, "quantization_config", None),
        )
        self._ops = operator_provider

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self._ops.gelu_tanh_and_mul(self.gate_up_proj(hidden_states))
        )


class Gemma4Router(nn.Module):
    def __init__(
        self,
        config: Gemma4TextConfig,
        operator_provider: Gemma4OperatorProvider,
        router_provider: Gemma4RouterProvider,
    ) -> None:
        super().__init__()
        self.top_k = int(config.top_k_experts)
        self.root_size = float(config.hidden_size) ** -0.5
        self.norm = Gemma4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            with_scale=False,
            provider=operator_provider,
        )
        self._ops = operator_provider
        self._router_ops = router_provider
        self.scale = nn.Parameter(torch.ones(config.hidden_size))
        self.proj = ReplicatedLinear(config.hidden_size, config.num_experts)
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_input = self._ops.router_input(
            hidden_states, self.scale, self.root_size, self.norm.eps
        )
        return self._router_ops.topk(
            self.proj(router_input), self.per_expert_scale, self.top_k
        )


class Gemma4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Gemma4TextConfig,
        layer_idx: int,
        operator_provider: Gemma4OperatorProvider,
        router_provider: Gemma4RouterProvider | None,
        rotary_emb: Gemma4RotaryEmbedding,
    ) -> None:
        super().__init__()
        self.self_attn = Gemma4Attention(
            config,
            layer_idx,
            rotary_emb,
            operator_provider,
        )
        self._ops = operator_provider
        self.mlp = Gemma4MLP(
            config, layer_idx, operator_provider,
            reduce_results=not bool(config.enable_moe_block),
        )
        self.input_layernorm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, provider=operator_provider
        )
        self.post_attention_layernorm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, provider=operator_provider
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, provider=operator_provider
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, provider=operator_provider
        )
        self.hidden_size_per_layer_input = int(config.hidden_size_per_layer_input)
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = ReplicatedLinear(
                config.hidden_size,
                self.hidden_size_per_layer_input,
            )
            self.per_layer_projection = ReplicatedLinear(
                self.hidden_size_per_layer_input,
                config.hidden_size,
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                provider=operator_provider,
            )
        self.enable_moe_block = bool(config.enable_moe_block)
        if self.enable_moe_block:
            self.parallel_context = get_parallel_context()
            if router_provider is None:
                raise RuntimeError("Gemma 4 MoE requires a router provider.")
            self.router = Gemma4Router(config, operator_provider, router_provider)
            self.experts = Gemma4PackedExperts(config)
            self.post_feedforward_layernorm_1 = Gemma4RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                provider=operator_provider,
            )
            self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                provider=operator_provider,
            )
            self.post_feedforward_layernorm_2 = Gemma4RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                provider=operator_provider,
            )
            # The common executor owns stream dependencies; collectives stay
            # on the caller stream, before each branch's normalization.
            self.moe_communication = AllReduceMoeCommunication(
                self.parallel_context.world.all_reduce
            )
            self.moe_execution = self._create_moe_execution()
        self.layer_scalar = nn.Parameter(torch.ones(1), requires_grad=False)

    def _create_moe_execution(self):
        return MoeExecutionPlan(
            routed=self._routed_moe_branch, shared=self._dense_moe_branch,
            communication=self.moe_communication,
            chunk_size=None,
            finish=self._finish_moe_branches,
            shared_modules=(self.mlp,),
        )

    def _dense_moe_branch(self, hidden_states):
        return self.mlp(self.pre_feedforward_layernorm(hidden_states))

    def _routed_moe_branch(self, hidden_states):
        weights, ids = self.router(hidden_states)
        return self.experts(self.pre_feedforward_layernorm_2(hidden_states), ids, weights)

    def _finish_moe_branches(self, routed, shared):
        # Preserve two independent reductions and nonlinear post-normalizations.
        shared = self.parallel_context.attn_tp.all_reduce(shared)
        routed = self.moe_communication.combine(routed)
        return (self.post_feedforward_layernorm_1(shared)
                + self.post_feedforward_layernorm_2(routed))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(positions, self.input_layernorm(hidden_states))
        hidden_states = self._ops.rmsnorm_residual(
            hidden_states,
            self.post_attention_layernorm.weight,
            residual,
            self.post_attention_layernorm.eps,
        )
        residual = hidden_states
        if self.enable_moe_block:
            hidden_states = self.moe_execution(residual, is_prefill=get_context().is_prefill)
        else:
            hidden_states = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        hidden_states = self._ops.rmsnorm_residual(
            hidden_states,
            self.post_feedforward_layernorm.weight,
            residual,
            self.post_feedforward_layernorm.eps,
            None if self.hidden_size_per_layer_input else self.layer_scalar,
        )
        if self.hidden_size_per_layer_input:
            if per_layer_input is None:
                raise RuntimeError("Gemma 4 PLE layer requires per_layer_input.")
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self._ops.gelu_mul(hidden_states, per_layer_input)
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self._ops.rmsnorm_residual(
                hidden_states,
                self.post_per_layer_input_norm.weight,
                residual,
                self.post_per_layer_input_norm.eps,
                self.layer_scalar,
            )
        return hidden_states


class Gemma4Model(nn.Module):
    def __init__(
        self,
        config: Gemma4TextConfig,
        operator_provider: Gemma4OperatorProvider,
        router_provider: Gemma4RouterProvider | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.hidden_size_per_layer_input = int(config.hidden_size_per_layer_input)
        if self.hidden_size_per_layer_input:
            packed_size = (
                int(config.num_hidden_layers) * self.hidden_size_per_layer_input
            )
            self.embed_tokens_per_layer = VocabParallelEmbedding(
                config.vocab_size_per_layer_input,
                packed_size,
            )
            self.per_layer_model_projection = ReplicatedLinear(
                config.hidden_size,
                packed_size,
            )
            self.per_layer_projection_norm = Gemma4RMSNorm(
                self.hidden_size_per_layer_input,
                eps=config.rms_norm_eps,
                provider=operator_provider,
            )
            self.per_layer_model_projection_scale = float(config.hidden_size) ** -0.5
            self.per_layer_input_scale = 2.0**-0.5
        rotary_keys = []
        rotary_embeddings = {}
        rotary_signatures = {}
        for layer_idx, layer_type in enumerate(config.layer_types):
            head_dim = int(
                config_layer_get(
                    config,
                    layer_idx,
                    "head_dim",
                    "head_dim"
                    if layer_type == "sliding_attention"
                    else "global_head_dim",
                )
            )
            rope_parameters = config_layer_get(config, layer_idx, "rope_parameters")
            parameters = dict(rope_parameters.get(layer_type, rope_parameters))
            signature = (
                str(layer_type),
                head_dim,
                tuple(
                    sorted((key, repr(value)) for key, value in parameters.items())
                ),
            )
            key = rotary_signatures.setdefault(
                signature, f"rope_{len(rotary_signatures)}"
            )
            rotary_keys.append(key)
            if key not in rotary_embeddings:
                rotary_embeddings[key] = Gemma4RotaryEmbedding(
                    config, str(layer_type), head_dim, parameters
                )
        self.rotary_embeddings = nn.ModuleDict(rotary_embeddings)
        self.layers = nn.ModuleList(
            Gemma4DecoderLayer(
                config,
                layer_idx,
                operator_provider,
                router_provider,
                self.rotary_embeddings[rotary_keys[layer_idx]],
            )
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, provider=operator_provider
        )
        self.embedding_scale = float(config.hidden_size) ** 0.5
        self.sparse_controller = None

    def get_per_layer_inputs(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.hidden_size_per_layer_input:
            return None
        token_inputs = (
            self.embed_tokens_per_layer(input_ids).view(
                -1,
                self.config.num_hidden_layers,
                self.hidden_size_per_layer_input,
            )
            * float(self.hidden_size_per_layer_input) ** 0.5
        )
        model_inputs = self.per_layer_projection_norm(
            (
                self.per_layer_model_projection(hidden_states)
                * self.per_layer_model_projection_scale
            ).view_as(token_inputs)
        )
        return (model_inputs + token_inputs) * self.per_layer_input_scale

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        multimodal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_tokens(input_ids) * self.embedding_scale
            if inputs_embeds is None
            else inputs_embeds
        )
        per_layer_ids = input_ids
        if self.hidden_size_per_layer_input and multimodal_mask is not None:
            per_layer_ids = input_ids.masked_fill(multimodal_mask, int(self.config.pad_token_id))
        per_layer_inputs = self.get_per_layer_inputs(per_layer_ids, hidden_states)
        context = get_context()
        for layer_idx, layer in enumerate(self.layers):
            context.now_layer_idx = layer_idx
            hidden_states = layer(
                positions,
                hidden_states,
                None if per_layer_inputs is None else per_layer_inputs[:, layer_idx],
            )
            if self.sparse_controller is not None:
                hidden_states, _ = self.sparse_controller.apply_activation_hook(
                    layer_idx, hidden_states, None, context
                )
                self.sparse_controller.on_layer_end(layer_idx, context)
        return self.norm(hidden_states)


class Gemma4ForCausalLM(nn.Module):
    special_weight_loaders = (".expert_weight",)
    packed_modules_excluded_prefixes = ("multimodal_encoder.",)
    packed_modules_mapping: ClassVar = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Gemma4TextConfig,
        operator_provider: Gemma4OperatorProvider,
        router_provider: Gemma4RouterProvider | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.operator_provider = operator_provider
        self.router_provider = router_provider
        self.model = Gemma4Model(config, operator_provider, router_provider)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        self.logit_softcap = float(config.final_logit_softcapping or 0.0)
        self.multimodal_encoder = None

    def configure_multimodal(self, outer_config) -> None:
        from sparseengine.models.gemma4_multimodal import Gemma4MultimodalEncoder

        self.multimodal_encoder = Gemma4MultimodalEncoder(outer_config)
        self.multimodal_bidirectional = (
            getattr(outer_config.text_config, "use_bidirectional_attention", None)
            == "vision"
        )

    def encode_multimodal(self, input_ids, tensors):
        if self.multimodal_encoder is None:
            raise RuntimeError("Gemma 4 multimodal encoder is disabled.")
        return self.multimodal_encoder.encode(input_ids, tensors)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids) * self.model.embedding_scale

    def forward_multimodal(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        multimodal_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds, multimodal_mask)

    @staticmethod
    def build_runtime_kwargs(
        config,
        *,
        engine_config,
        parallel_context,
        device,
        **_,
    ):
        tp_size = int(parallel_context.attn_tp_size)
        contracts: set[tuple[int, int, int, int]] = set()
        for layer_idx in range(int(config.num_hidden_layers)):
            is_sliding = str(config.layer_types[layer_idx]) == "sliding_attention"
            head_dim = int(
                config_layer_get(
                    config,
                    layer_idx,
                    "head_dim",
                    "head_dim" if is_sliding else "global_head_dim",
                )
            )
            use_k_eq_v = bool(
                getattr(config, "attention_k_eq_v", False) and not is_sliding
            )
            total_kv_heads = int(
                config_layer_get(
                    config,
                    layer_idx,
                    "num_key_value_heads",
                    (
                        "num_global_key_value_heads"
                        if use_k_eq_v
                        else "num_key_value_heads"
                    ),
                )
            )
            window_left = int(config.sliding_window) - 1 if is_sliding else -1
            contracts.add(
                (
                    int(config.num_attention_heads) // tp_size,
                    max(1, total_kv_heads // tp_size),
                    head_dim,
                    window_left,
                )
            )
        attention_contracts = tuple(sorted(contracts))
        head_dims = tuple(sorted({contract[2] for contract in attention_contracts}))
        operator_provider = resolve_gemma4_provider(
            Gemma4OpSpec(
                activation_dtype=model_activation_dtype(config),
                head_dims=head_dims,
                cuda_graph=bool(engine_config.decode_graph),
                attention_contracts=attention_contracts,
                max_batch_size=int(getattr(engine_config, "max_decoding_seqs", 1)),
                context_capacity=int(getattr(engine_config, "max_model_len", 0) or 0)
                or None,
            ),
            device_index=device.index,
        )
        runtime_kwargs = {"operator_provider": operator_provider}
        if bool(config.enable_moe_block):
            try:
                runtime_kwargs["router_provider"] = resolve_gemma4_router_provider(
                    Gemma4RouterOpSpec(
                        activation_dtype=model_activation_dtype(config),
                        num_experts=int(config.num_experts),
                        top_k=int(config.top_k_experts),
                        cuda_graph=bool(engine_config.decode_graph),
                    ),
                    device_index=device.index,
                )
            except BaseException:
                operator_provider.close()
                raise
        return runtime_kwargs

    def close_runtime_operators(self) -> None:
        close = getattr(self.operator_provider, "close", None)
        if callable(close):
            close()
        router_close = getattr(self.router_provider, "close", None)
        if callable(router_close):
            router_close()

    def map_weight_name(self, source_weight_name: str) -> str | None:
        match = _EXPERT_SOURCE_RE.match(source_weight_name)
        if match is not None:
            layer_idx, projection = match.groups()
            return f"model.layers.{layer_idx}.experts.{projection}.expert_weight"
        prefix = "model.language_model."
        if source_weight_name.startswith(prefix + "layers."):
            parts = source_weight_name.split(".")
            layer_idx = int(parts[3])
            shared_start = int(self.config.num_hidden_layers) - int(
                self.config.num_kv_shared_layers
            )
            if layer_idx >= shared_start > 0 and parts[-2] in {
                "k_proj",
                "v_proj",
                "k_norm",
                "v_norm",
            }:
                return None
        multimodal_prefixes = (
            ("model.vision_tower.", "multimodal_encoder.vision_tower."),
            ("model.audio_tower.", "multimodal_encoder.audio_tower."),
            ("model.embed_vision.", "multimodal_encoder.embed_vision."),
            ("model.embed_audio.", "multimodal_encoder.embed_audio."),
        )
        if self.multimodal_encoder is None and source_weight_name.startswith(
            tuple(prefix for prefix, _ in multimodal_prefixes)
        ):
            return None
        for source_prefix, target_prefix in multimodal_prefixes:
            if source_weight_name.startswith(source_prefix):
                return target_prefix + source_weight_name[len(source_prefix) :]
        return (
            "model." + source_weight_name[len(prefix) :]
            if source_weight_name.startswith(prefix)
            else None
        )

    def resolve_special_weight(self, target_weight_name: str) -> WeightTarget | None:
        match = _EXPERT_TARGET_RE.match(target_weight_name)
        if match is None:
            return None
        layer_idx, projection = match.groups()
        return WeightTarget(self.model.layers[int(layer_idx)].experts, projection)

    def load_special_weight(
        self,
        target_weight_name: str,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None,
    ) -> int:
        if loaded_scale is not None:
            raise ValueError("Gemma 4 BF16 packed experts do not accept scales.")
        target = self.resolve_special_weight(target_weight_name)
        if target is None:
            return 0
        target.module.load_packed_weight(str(target.shard_id), loaded_weight)
        return 1

    def validate_loaded_weights(self, loaded_parameter_names: set[str]) -> None:
        packed_experts = {
            name
            for name, _ in self.named_parameters()
            if name.endswith((".experts.w13_weight", ".experts.w2_weight"))
        }
        optional_tied_head = (
            {"lm_head.weight"} if self.config.tie_word_embeddings else set()
        )
        missing = sorted(
            {name for name, _ in self.named_parameters()}
            - packed_experts
            - optional_tied_head
            - loaded_parameter_names
        )
        if missing:
            raise ValueError(f"Missing Gemma 4 weights: {missing[:8]}.")
        if self.config.enable_moe_block:
            for layer in self.model.layers:
                layer.experts.validate_loaded_weights()

    @torch.inference_mode()
    def warmup_moe(self, num_tokens: int = 1) -> None:
        if not self.config.enable_moe_block:
            return
        experts = self.model.layers[0].experts
        hidden = torch.zeros(
            (int(num_tokens), experts.hidden_size),
            dtype=model_activation_dtype(self.config),
            device=experts.w13_weight.device,
        )
        ids = (
            torch.arange(
                int(num_tokens) * int(self.config.top_k_experts),
                device=hidden.device,
            )
            .remainder(experts.num_experts)
            .view(int(num_tokens), -1)
        )
        weights = torch.full_like(
            ids, 1.0 / int(self.config.top_k_experts), dtype=hidden.dtype
        )
        experts(hidden, ids, weights)
        device_runtime.synchronize()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        multimodal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds, multimodal_mask)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.lm_head(hidden_states)
        if logits is not None and self.logit_softcap:
            logits = torch.tanh(logits / self.logit_softcap) * self.logit_softcap
        return logits
