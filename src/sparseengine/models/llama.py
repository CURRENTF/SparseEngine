import os
from typing import Any

import torch
from torch import nn

from sparseengine.distributed import get_parallel_context
from sparseengine.layers.activation import SiluAndMul
from sparseengine.layers.attention import Attention
from sparseengine.layers.layernorm import RMSNorm
from sparseengine.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from sparseengine.layers.rotary_embedding import get_rope
from sparseengine.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from sparseengine.models.attention_runtime import (
    bind_mha_full_attention_provider,
    build_mha_full_attention_provider,
)
from sparseengine.models.rope import (
    resolve_rope_max_position,
    resolve_rope_scaling,
    resolve_rope_theta,
)
from sparseengine.operators.full_attention import FullAttentionProvider
from sparseengine.utils.context import get_context
from sparseengine.utils.log import logger


def _get_rope_theta(config: Any) -> float:
    return resolve_rope_theta(config)


def _get_rope_scaling(config: Any) -> tuple[tuple[str, object], ...] | None:
    return resolve_rope_scaling(config, model_name="Llama")


class LlamaAttention(nn.Module):
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
        super().__init__()
        tp_size = get_parallel_context().attn_tp_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.proj_chunk_size = int(proj_chunk_size)
        if self.proj_chunk_size <= 0:
            raise ValueError(f"proj_chunk_size must be > 0, got {proj_chunk_size}.")

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quantization=quantization,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=qkv_bias,
            quantization=quantization,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def _o_proj_chunked(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        chunk_size = int(self.proj_chunk_size)
        if int(x.shape[0]) <= chunk_size:
            out.copy_(self.o_proj(x))
            return out
        for start in range(0, int(x.shape[0]), chunk_size):
            end = min(start + chunk_size, int(x.shape[0]))
            out[start:end].copy_(self.o_proj(x[start:end]))
        return out

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        cache_manager = context.cache_manager
        layer_idx = context.now_layer_idx
        cache_manager.save_raw_kv_if_needed(layer_idx, k, v)
        debug_layers = os.getenv("SPARSEENGINE_DEBUG_CAPTURE_PRE_ROPE_LAYERS")
        if debug_layers:
            wanted = {int(part) for part in debug_layers.split(",") if part.strip()}
            if int(layer_idx) in wanted:
                self.debug_last_pre_rope_positions = positions.detach().clone()
                self.debug_last_pre_rope_k = k.detach().clone()
                self.debug_last_pre_rope_v = v.detach().clone()
        q, k = self.rotary_emb(positions, q, k)
        cache_manager.save_rope_kv_if_needed(layer_idx, k, v)
        o = self.attn(q, k, v)
        return self._o_proj_chunked(o.flatten(1, -1), hidden_states)


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        mlp_bias: bool,
        mlp_chunk_size: int = 16384,
        quantization=None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=mlp_bias,
            quantization=quantization,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=mlp_bias,
            quantization=quantization,
        )
        if hidden_act != "silu":
            raise NotImplementedError(f"Unsupported Llama hidden_act={hidden_act!r}.")
        self.act_fn = SiluAndMul()
        self.mlp_chunk_size = int(mlp_chunk_size)
        if self.mlp_chunk_size <= 0:
            raise ValueError(f"mlp_chunk_size must be > 0, got {mlp_chunk_size}.")

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        return self.down_proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunk_size = int(self.mlp_chunk_size)
        if int(x.shape[0]) <= chunk_size:
            return self._forward_chunk(x)

        out = torch.empty_like(x)
        for start in range(0, int(x.shape[0]), chunk_size):
            end = min(start + chunk_size, int(x.shape[0]))
            out[start:end].copy_(self._forward_chunk(x[start:end]))
        return out


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        quantization = getattr(config, "quantization_config", None)
        self.self_attn = LlamaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=resolve_rope_max_position(config, model_name="Llama"),
            head_dim=head_dim,
            qkv_bias=getattr(config, "attention_bias", False),
            rope_theta=_get_rope_theta(config),
            rope_scaling=_get_rope_scaling(config),
            proj_chunk_size=getattr(config, "mlp_chunk_size", 16384),
            quantization=quantization,
        )
        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            mlp_bias=getattr(config, "mlp_bias", False),
            mlp_chunk_size=getattr(config, "mlp_chunk_size", 16384),
            quantization=quantization,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.sparse_controller = None

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        context = get_context()
        debug_layers_env = os.getenv("SPARSEENGINE_DEBUG_HIDDEN_LAYERS")
        debug_layers = None
        if debug_layers_env:
            debug_layers = {int(part) for part in debug_layers_env.split(",") if part.strip()}
            self.debug_last_hidden_states = {
                -1: hidden_states[-1:].detach().clone(),
            }

        for i, layer in enumerate(self.layers):
            context.now_layer_idx = i
            hidden_states, residual = layer(positions, hidden_states, residual)
            if self.sparse_controller is not None:
                hidden_states, residual = self.sparse_controller.apply_activation_hook(
                    i,
                    hidden_states,
                    residual,
                    context,
                )
            if debug_layers is not None and i in debug_layers:
                self.debug_last_hidden_states[int(i)] = hidden_states[-1:].detach().clone()
            if self.sparse_controller is not None:
                self.sparse_controller.on_layer_end(i, context)

        hidden_states, _ = self.norm(hidden_states, residual)
        if debug_layers is not None:
            self.debug_last_hidden_states[self.config.num_hidden_layers] = hidden_states[-1:].detach().clone()
        return hidden_states


class LlamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    @staticmethod
    def build_runtime_kwargs(
        config: Any,
        *,
        engine_config,
        parallel_context,
        device: torch.device,
        **_,
    ) -> dict:
        return {
            "full_attention_provider": build_mha_full_attention_provider(
                config,
                sparse_method=engine_config.sparse_method,
                attention_tp_size=parallel_context.attn_tp_size,
                device=device,
                max_batch_size=engine_config.max_decoding_seqs,
                cuda_graph=engine_config.decode_graph,
                runtime_config=engine_config,
            ),
        }

    def __init__(
        self,
        config: Any,
        full_attention_provider: FullAttentionProvider | None = None,
    ) -> None:
        super().__init__()
        self.full_attention_provider = full_attention_provider
        self.model = LlamaModel(config)
        if full_attention_provider is not None:
            bind_mha_full_attention_provider(self.model, full_attention_provider)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        logger.info(
            "Loaded Llama prefill_provider={} decode_provider={}",
            (
                "legacy_triton"
                if full_attention_provider is None
                else full_attention_provider.prefill_name
            ),
            (
                "legacy_triton"
                if full_attention_provider is None
                else full_attention_provider.decode_name
            ),
        )

    def close_runtime_operators(self) -> None:
        if self.full_attention_provider is not None:
            self.full_attention_provider.close()

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
