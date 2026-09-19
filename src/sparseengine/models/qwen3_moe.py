from __future__ import annotations

import os
import re
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3MoeConfig

from sparseengine.distributed import DecodeParallelCollectives, get_parallel_context
from sparseengine.distributed.moe_communication import prepare_moe_communication
from sparseengine.layers.embed_head import ParallelLMHead
from sparseengine.layers.packed_moe import PackedMoeExperts
from sparseengine.models.qwen3 import (
    Qwen3DecoderLayerBase,
    Qwen3ModelBase,
)
from sparseengine.models.attention_runtime import (
    bind_mha_full_attention_provider,
    build_mha_full_attention_provider,
)
from sparseengine.operators.moe import (
    model_activation_dtype,
    resolve_moe_provider,
)
from sparseengine.operators.moe_router import (
    MoeRouterOpSpec,
    resolve_moe_router_provider,
)
from sparseengine.operators.full_attention import FullAttentionProvider
from sparseengine.platforms import device_runtime
from sparseengine.utils.log import logger
from sparseengine.utils.weight_target import WeightTarget


_EXPERT_SOURCE_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
_EXPERT_TARGET_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.expert_weight$"
)


class Qwen3MoeRouter(nn.Module):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_experts = int(config.num_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.norm_topk_prob = bool(config.norm_topk_prob)
        self.op_spec = MoeRouterOpSpec(
            num_experts=self.num_experts,
            top_k=self.top_k,
            activation_dtype=model_activation_dtype(config),
            norm_topk_prob=self.norm_topk_prob,
            cuda_graph=bool(getattr(config, "decode_graph", False)),
        )
        self.provider = resolve_moe_router_provider(self.op_spec)
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = F.linear(hidden_states, self.weight)
        topk_weights, topk_ids = self.provider.run(self.op_spec, router_logits)
        return router_logits, topk_weights, topk_ids


class Qwen3MoePackedExperts(PackedMoeExperts):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__(
            num_experts=int(config.num_experts),
            hidden_size=int(config.hidden_size),
            intermediate_size=int(config.moe_intermediate_size),
            top_k=int(config.num_experts_per_tok),
            activation_dtype=model_activation_dtype(config),
            fp8_enabled=bool(
                getattr(
                    getattr(config, "quantization_config", None),
                    "enabled",
                    False,
                )
            ),
            cuda_graph=bool(getattr(config, "decode_graph", False)),
            max_num_tokens=int(getattr(config, "moe_max_num_tokens", 1)),
            model_label="Qwen3MoE",
            provider_resolver=resolve_moe_provider,
            parallel_context=get_parallel_context(),
        )


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(
        self, config: Qwen3MoeConfig,
        parallel_collectives: DecodeParallelCollectives | None = None,
    ) -> None:
        super().__init__()
        self.parallel_context = get_parallel_context()
        self.mlp_chunk_size = int(getattr(config, "mlp_chunk_size", 16384))
        if self.mlp_chunk_size <= 0:
            raise ValueError(
                f"mlp_chunk_size must be > 0, got {self.mlp_chunk_size}."
            )
        self.moe_communication = prepare_moe_communication(
            self.parallel_context, parallel_collectives
        )
        self.gate = Qwen3MoeRouter(config)
        self.experts = Qwen3MoePackedExperts(config)

    def _route(self, hidden_states):
        _, weights, ids = self.gate(hidden_states)
        return ids, weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() != 2:
            raise ValueError(
                f"Qwen3MoeSparseMoeBlock expects [tokens, hidden], got {tuple(hidden_states.shape)}."
            )
        if self.parallel_context.attn_dp_size > 1:
            return self.moe_communication.run(
                hidden_states, route=self._route, experts=self.experts,
                chunk_size=self.mlp_chunk_size,
            )
        debug_enabled = os.getenv("SPARSEENGINE_DEBUG_MOE", "0") == "1"
        if debug_enabled:
            self.debug_last_input = hidden_states.detach().clone()

        if int(hidden_states.shape[0]) <= self.mlp_chunk_size:
            router_logits, topk_weights, topk_ids = self.gate(hidden_states)
            local_output = self.experts(
                hidden_states,
                topk_ids,
                topk_weights,
            )
        else:
            router_logits_chunks = [] if debug_enabled else None
            topk_weights_chunks = [] if debug_enabled else None
            topk_ids_chunks = [] if debug_enabled else None
            local_output_chunks = []
            for chunk in hidden_states.split(self.mlp_chunk_size, dim=0):
                router_logits, topk_weights, topk_ids = self.gate(chunk)
                local_output = self.experts(
                    chunk,
                    topk_ids,
                    topk_weights,
                )
                if debug_enabled:
                    router_logits_chunks.append(router_logits)
                    topk_weights_chunks.append(topk_weights)
                    topk_ids_chunks.append(topk_ids)
                local_output_chunks.append(local_output)

            local_output = torch.cat(local_output_chunks, dim=0)
            if debug_enabled:
                router_logits = torch.cat(router_logits_chunks, dim=0)
                topk_weights = torch.cat(topk_weights_chunks, dim=0)
                topk_ids = torch.cat(topk_ids_chunks, dim=0)

        if debug_enabled:
            self.debug_last_router_logits = router_logits.detach().clone()
            self.debug_last_topk_ids = topk_ids.detach().clone()
            self.debug_last_topk_weights = topk_weights.detach().clone()
            self.debug_last_local_output = local_output.detach().clone()
            local_mask = (topk_ids >= self.experts.local_expert_start) & (
                topk_ids < self.experts.local_expert_end
            )
            local_hit_count = local_mask.sum()
            self.debug_last_local_hit_count = (
                local_hit_count
                if torch.cuda.is_available()
                and torch.cuda.is_current_stream_capturing()
                else int(local_hit_count.item())
            )

        output = self.moe_communication.combine(local_output)
        if debug_enabled:
            self.debug_last_output = output.detach().clone()
        return output


class Qwen3MoeDecoderLayer(Qwen3DecoderLayerBase):
    def __init__(
        self, config: Qwen3MoeConfig,
        parallel_collectives: DecodeParallelCollectives | None = None,
    ) -> None:
        super().__init__(config)
        self.parallel_context = get_parallel_context()
        self.mlp = Qwen3MoeSparseMoeBlock(config, parallel_collectives)

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


class Qwen3MoeModel(Qwen3ModelBase):
    def __init__(
        self, config: Qwen3MoeConfig,
        parallel_collectives: DecodeParallelCollectives | None = None,
    ) -> None:
        super().__init__(
            config, partial(Qwen3MoeDecoderLayer, parallel_collectives=parallel_collectives)
        )


class Qwen3MoeForCausalLM(nn.Module):
    special_weight_loaders = (".expert_weight",)
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    @staticmethod
    def build_runtime_kwargs(
        config: Qwen3MoeConfig,
        *,
        engine_config,
        parallel_context,
        device: torch.device,
        collective_runtime,
        max_decode_tokens,
        **_,
    ) -> dict:
        return {
            "parallel_collectives": collective_runtime.request_moe_collectives(
                attention_max_rows=max_decode_tokens, moe_max_rows=max_decode_tokens,
                max_local_tokens=engine_config.max_num_batched_tokens,
                hidden_size=int(config.hidden_size), dtype=model_activation_dtype(config),
                backend=engine_config.moe_backend, num_experts=int(config.num_experts),
                top_k=int(config.num_experts_per_tok),
            ),
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
        config: Qwen3MoeConfig,
        full_attention_provider: FullAttentionProvider | None = None,
        parallel_collectives: DecodeParallelCollectives | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.parallel_context = get_parallel_context()
        self.full_attention_provider = full_attention_provider
        self.model = Qwen3MoeModel(config, parallel_collectives)
        if full_attention_provider is not None:
            bind_mha_full_attention_provider(self.model, full_attention_provider)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        self._intentionally_skipped_expert_weights: set[str] = set()
        self._intentionally_skipped_expert_scales: set[str] = set()
        logger.info(
            "Loaded Qwen3MoE prefill_provider={} decode_provider={}",
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

    @torch.inference_mode()
    def warmup_moe(self, num_tokens: int = 1) -> None:
        num_tokens = int(num_tokens)
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be > 0, got {num_tokens}.")
        block = self.model.layers[0].mlp
        experts = block.experts
        top_k = int(self.config.num_experts_per_tok)
        device = experts.w13_weight.device
        dtype = block.gate.weight.dtype
        hidden_states = torch.zeros(
            (num_tokens, experts.hidden_size),
            dtype=dtype,
            device=device,
        )
        if experts.fp8_enabled:
            layer = self.model.layers[0]
            layer.self_attn.qkv_proj(hidden_states)
            layer.self_attn.o_proj(
                torch.zeros(
                    (num_tokens, layer.self_attn.q_size),
                    dtype=dtype,
                    device=device,
                )
            )
            block.gate(hidden_states)
        topk_ids = (
            torch.arange(
                num_tokens * top_k,
                dtype=torch.int64,
                device=device,
            )
            .remainder(experts.num_local_experts)
            .add(experts.local_expert_start)
            .view(num_tokens, top_k)
        )
        topk_weights = torch.full(
            (num_tokens, top_k),
            1.0 / top_k,
            dtype=dtype,
            device=device,
        )
        experts(hidden_states, topk_ids, topk_weights)
        device_runtime.synchronize()

    def map_weight_name(self, source_weight_name: str) -> str | None:
        match = _EXPERT_SOURCE_RE.match(source_weight_name)
        if match is None:
            return source_weight_name
        layer_idx, global_expert_id, projection = match.groups()
        global_expert_id = int(global_expert_id)
        experts = self.model.layers[int(layer_idx)].mlp.experts
        if not experts.is_local_expert(global_expert_id):
            self._intentionally_skipped_expert_weights.add(source_weight_name)
            return None
        return (
            f"model.layers.{layer_idx}.mlp.experts.{global_expert_id}."
            f"{projection}.expert_weight"
        )

    def resolve_special_weight(
        self,
        target_weight_name: str,
    ) -> WeightTarget | None:
        match = _EXPERT_TARGET_RE.match(target_weight_name)
        if match is None:
            return None
        layer_idx, global_expert_id, projection = match.groups()
        experts = self.model.layers[int(layer_idx)].mlp.experts
        return WeightTarget(experts, (int(global_expert_id), projection))

    def record_skipped_weight(
        self,
        source_weight_name: str,
        loaded_weight_shape: tuple[int, ...] | None,
        loaded_weight_dtype: str | None,
        loaded_scale_shape: tuple[int, ...] | None,
        loaded_scale_dtype: str | None,
    ) -> None:
        match = _EXPERT_SOURCE_RE.match(source_weight_name)
        if match is None:
            raise ValueError(
                f"Qwen3MoE loader unexpectedly skipped {source_weight_name!r}."
            )
        layer_idx, global_expert_id, projection = match.groups()
        experts = self.model.layers[int(layer_idx)].mlp.experts
        if experts.is_local_expert(int(global_expert_id)):
            raise ValueError(
                f"Qwen3MoE loader skipped local expert weight {source_weight_name!r}."
            )
        expected_weight_shape = (
            (experts.hidden_size, experts.global_intermediate_size)
            if projection == "down_proj"
            else (experts.global_intermediate_size, experts.hidden_size)
        )
        if loaded_weight_shape != expected_weight_shape:
            raise ValueError(
                "Remote Qwen3MoE expert weight shape mismatch for "
                f"{source_weight_name!r}: expected={expected_weight_shape}, "
                f"got={loaded_weight_shape}."
            )
        if experts.fp8_enabled:
            if loaded_weight_dtype != "F8_E4M3":
                raise TypeError(
                    "Remote Qwen3MoE expert weight must be FP8 E4M3, got "
                    f"safetensors dtype {loaded_weight_dtype}."
                )
            expected_scale_shape = (
                (experts.hidden_size // 128, experts.global_intermediate_size // 128)
                if projection == "down_proj"
                else (
                    experts.global_intermediate_size // 128,
                    experts.hidden_size // 128,
                )
            )
            if loaded_scale_shape != expected_scale_shape:
                raise ValueError(
                    "Remote Qwen3MoE expert scale shape mismatch for "
                    f"{source_weight_name!r}: expected={expected_scale_shape}, "
                    f"got={loaded_scale_shape}."
                )
            if loaded_scale_dtype != "BF16":
                raise TypeError(
                    "Remote Qwen3MoE expert scale must be BF16, got "
                    f"safetensors dtype {loaded_scale_dtype}."
                )
            self._intentionally_skipped_expert_scales.add(
                source_weight_name[: -len(".weight")] + ".weight_scale_inv"
            )
        elif loaded_scale_shape is not None or loaded_scale_dtype is not None:
            raise ValueError(
                "Unquantized remote Qwen3MoE expert unexpectedly has "
                f"weight_scale_inv: {source_weight_name!r}."
            )
        elif loaded_weight_dtype not in {"BF16", "F16", "F32"}:
            raise TypeError(
                "Remote unquantized Qwen3MoE expert has unsupported safetensors "
                f"dtype {loaded_weight_dtype}."
            )
        self._intentionally_skipped_expert_weights.add(source_weight_name)

    def load_special_weight(
        self,
        target_weight_name: str,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None,
    ) -> int:
        target = self.resolve_special_weight(target_weight_name)
        if target is None:
            return 0
        global_expert_id, projection = target.shard_id
        target.module.load_expert_weight(
            global_expert_id,
            projection,
            loaded_weight,
            loaded_scale,
        )
        return 1

    def validate_loaded_weights(self, loaded_parameter_names: set[str]) -> None:
        packed_expert_parameters = {
            name
            for name, _ in self.named_parameters()
            if name.endswith(".mlp.experts.w13_weight")
            or name.endswith(".mlp.experts.w2_weight")
        }
        expected_dense_parameters = {
            name for name, _ in self.named_parameters()
        } - packed_expert_parameters
        missing_dense = sorted(expected_dense_parameters - loaded_parameter_names)
        if missing_dense:
            raise ValueError(
                f"Missing replicated Qwen3MoE weights: {missing_dense[:8]}."
            )

        for layer in self.model.layers:
            layer.mlp.experts.validate_loaded_weights()

        expected_skipped = {
            f"model.layers.{layer_idx}.mlp.experts.{expert_id}.{projection}.weight"
            for layer_idx in range(int(self.config.num_hidden_layers))
            for expert_id in range(int(self.config.num_experts))
            if not self.model.layers[layer_idx].mlp.experts.is_local_expert(expert_id)
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        expected_skipped_scales = (
            {name[: -len(".weight")] + ".weight_scale_inv" for name in expected_skipped}
            if self.model.layers[0].mlp.experts.fp8_enabled
            else set()
        )
        missing_skips = sorted(
            expected_skipped - self._intentionally_skipped_expert_weights
        )
        if missing_skips:
            raise ValueError(
                "Checkpoint is missing expected remote expert entries: "
                f"{missing_skips[:8]}."
            )
        unexpected_skips = sorted(
            self._intentionally_skipped_expert_weights - expected_skipped
        )
        missing_scale_skips = sorted(
            expected_skipped_scales - self._intentionally_skipped_expert_scales
        )
        unexpected_scale_skips = sorted(
            self._intentionally_skipped_expert_scales - expected_skipped_scales
        )
        if missing_scale_skips:
            raise ValueError(
                "Checkpoint is missing expected remote expert scales: "
                f"{missing_scale_skips[:8]}."
            )
        if unexpected_skips or unexpected_scale_skips:
            raise ValueError(
                "Unexpectedly skipped Qwen3MoE expert entries: "
                f"weights={unexpected_skips[:4]}, "
                f"scales={unexpected_scale_skips[:4]}."
            )
        logger.info(
            "Loaded Qwen3MoE rank {} provider={} attention TP {}/{} MoE TP {}/{} "
            "and local experts "
            "[{}, {}) across {} layers; intentionally skipped {} remote expert tensors.",
            self.parallel_context.world_rank,
            self.model.layers[0].mlp.experts.provider.name,
            self.parallel_context.attn_tp_rank,
            self.parallel_context.attn_tp_size,
            self.parallel_context.moe_tp_rank,
            self.parallel_context.moe_tp_size,
            self.model.layers[0].mlp.experts.local_expert_start,
            self.model.layers[0].mlp.experts.local_expert_end,
            len(self.model.layers),
            len(self._intentionally_skipped_expert_weights),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def forward_idle_experts(self, hidden_states):
        for layer in self.model.layers:
            layer.mlp(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
