from __future__ import annotations

import os
import re

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Glm4MoeLiteConfig

from sparseengine.distributed import (
    DecodeParallelCollectives,
    ParallelCollectiveRuntime,
    ParallelContext,
    get_parallel_context,
)
from sparseengine.distributed.moe_communication import prepare_moe_communication
from sparseengine.kernels.triton.glm_mla_decode import (
    project_and_fuse_glm_mla_decode_rope,
)
from sparseengine.layers.activation import SiluAndMul
from sparseengine.layers.embed_head import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sparseengine.layers.layernorm import RMSNorm
from sparseengine.layers.linear import (
    AbsorbedColumnParallelLinear,
    ColumnParallelLinear,
    MergedReplicatedLinear,
    RowParallelLinear,
)
from sparseengine.layers.mla_attention import MLAAttention
from sparseengine.layers.packed_moe import PackedMoeExperts
from sparseengine.layers.rotary_embedding import RotaryEmbedding, get_rope
from sparseengine.method_registry import sparse_decode_attention_score_kind
from sparseengine.models.layout import resolve_attention_qk_head_dim
from sparseengine.models.qwen3 import Qwen3MLP
from sparseengine.operators.activation import resolve_silu_and_mul_provider
from sparseengine.operators.attention_capabilities import AttentionScoreKind
from sparseengine.operators.mla_attention import MlaAttentionOpSpec
from sparseengine.operators.moe import (
    MoeOpSpec,
    append_shared_expert_route,
    model_activation_dtype,
    packed_shared_expert_token_limit,
    resolve_moe_provider,
    use_packed_shared_experts,
    use_packed_shared_experts_in_prefill,
)
from sparseengine.operators.moe_execution import MoeExecutionPlan
from sparseengine.operators.moe_router import (
    MoeRouterOpSpec,
    resolve_moe_router_provider,
)
from sparseengine.platforms import device_runtime
from sparseengine.utils.profiler import profiler
from sparseengine.utils.context import get_context
from sparseengine.utils.weight_target import WeightTarget

_EXPERT_SOURCE_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
_EXPERT_TARGET_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.expert_weight$"
)
_SHARED_EXPERT_SOURCE_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.shared_experts\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)


def build_glm4_moe_lite_mla_attention(
    config: Glm4MoeLiteConfig,
    *,
    device: torch.device | str,
    max_batch_size: int,
    prefill_workspace_bytes: int,
    decode_graph: bool,
    context_capacity: int,
    projection_chunk_size: int,
    score_output: AttentionScoreKind = AttentionScoreKind.NONE,
    history_chunk_size: int = 16384,
) -> MLAAttention:
    """Bind the one process-local MLA operator from explicit runtime inputs."""

    parallel_context = get_parallel_context()
    activation_dtype = model_activation_dtype(config)
    spec = MlaAttentionOpSpec(
        num_q_heads=int(config.num_attention_heads),
        kv_lora_rank=int(config.kv_lora_rank),
        rope_dim=int(config.qk_rope_head_dim),
        qk_head_dim=resolve_attention_qk_head_dim(config),
        value_head_dim=int(config.v_head_dim),
        activation_dtype=activation_dtype,
        cache_dtype=activation_dtype,
        tp_size=int(parallel_context.attn_tp_size),
        cuda_graph=bool(decode_graph),
        score_output=score_output,
        context_capacity=int(context_capacity),
        batch_capacity=int(max_batch_size),
    )
    return MLAAttention.bind(
        spec=spec,
        device=device,
        max_batch_size=max_batch_size,
        prefill_workspace_bytes=prefill_workspace_bytes,
        hidden_size=int(config.hidden_size),
        projection_chunk_size=projection_chunk_size,
        history_chunk_size=history_chunk_size,
    )


class Glm4MoeLiteAttention(nn.Module):
    """GLM projections around a shared latent-MLA execution object."""

    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        mla_attention: MLAAttention,
        *,
        projection_chunk_size: int,
        parallel_collectives: DecodeParallelCollectives | None = None,
    ) -> None:
        super().__init__()
        self.mla_attention = mla_attention
        self.parallel_context = get_parallel_context()
        self.parallel_collectives = parallel_collectives
        self.num_heads = int(config.num_attention_heads)
        self.local_heads = int(mla_attention.spec.local_q_heads)
        self.q_lora_rank = int(config.q_lora_rank)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.qk_head_dim = resolve_attention_qk_head_dim(config)
        self.v_head_dim = int(config.v_head_dim)
        self.proj_chunk_size = int(projection_chunk_size)
        if self.proj_chunk_size <= 0:
            raise ValueError(
                f"mlp_chunk_size must be positive, got {self.proj_chunk_size}."
            )
        if self.proj_chunk_size != int(mla_attention.projection_chunk_size):
            raise ValueError(
                "GLM projection chunk size must match the MLA workspace bound: "
                f"model={self.proj_chunk_size} "
                f"mla={mla_attention.projection_chunk_size}."
            )
        if int(config.hidden_size) != int(mla_attention.hidden_size):
            raise ValueError(
                "GLM hidden size must match the MLA workspace bound: "
                f"model={config.hidden_size} mla={mla_attention.hidden_size}."
            )
        quantization = getattr(config, "quantization_config", None)

        self.fused_qkv_a_proj = MergedReplicatedLinear(
            int(config.hidden_size),
            [
                self.q_lora_rank,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ],
            bias=bool(config.attention_bias),
            quantization=quantization,
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quantization=quantization,
        )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            eps=config.rms_norm_eps,
        )
        self.kv_b_proj = AbsorbedColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quantization=quantization,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            int(config.hidden_size),
            bias=bool(config.attention_bias),
            reduce_results=parallel_collectives is None,
            quantization=quantization,
        )

    def _project_kv_history(self, latent: torch.Tensor) -> torch.Tensor:
        if int(latent.shape[0]) <= self.proj_chunk_size:
            return self.kv_b_proj(latent)
        output = torch.empty(
            latent.shape[0],
            self.local_heads * (self.qk_nope_head_dim + self.v_head_dim),
            dtype=latent.dtype,
            device=latent.device,
        )
        for start in range(0, int(latent.shape[0]), self.proj_chunk_size):
            end = min(start + self.proj_chunk_size, int(latent.shape[0]))
            with profiler.trace("mla.prefill.kv_projection.chunk_into"):
                self.kv_b_proj(latent[start:end], out=output[start:end])
        return output

    def _project_output(
        self, value_output: torch.Tensor, chunk_buffer: torch.Tensor
    ) -> torch.Tensor:
        flattened = value_output.flatten(1, -1)
        if int(flattened.shape[0]) <= self.proj_chunk_size:
            return self.o_proj(flattened)
        # Only chunked projection needs the old hidden-state storage to assemble
        # a complete result. Callers consume the returned tensor in either case.
        for start in range(0, int(flattened.shape[0]), self.proj_chunk_size):
            end = min(start + self.proj_chunk_size, int(flattened.shape[0]))
            chunk_buffer[start:end].copy_(self.o_proj(flattened[start:end]))
        return chunk_buffer

    def _decode_absorbed_query(self, q_nope: torch.Tensor) -> torch.Tensor:
        kv_b_weight = self.kv_b_proj.absorbed_weight.view(
            self.local_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        k_weight = kv_b_weight[:, : self.qk_nope_head_dim]
        return torch.bmm(
            q_nope.transpose(0, 1),
            k_weight,
        ).transpose(0, 1)

    def _reconstruct_decode_values(self, latent_output: torch.Tensor) -> torch.Tensor:
        kv_b_weight = self.kv_b_proj.absorbed_weight.view(
            self.local_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        v_weight = kv_b_weight[:, self.qk_nope_head_dim :]
        return torch.bmm(
            latent_output.transpose(0, 1),
            v_weight.transpose(1, 2),
        ).transpose(0, 1)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        rotary_emb: RotaryEmbedding,
    ) -> torch.Tensor:
        compressed_qkv = self.fused_qkv_a_proj(hidden_states)
        compressed_q, compressed_kv = compressed_qkv.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        normalized_q = self.q_a_layernorm(compressed_q)

        latent, k_rope = compressed_kv.split(
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        latent = self.kv_a_layernorm(latent)
        if get_context().is_prefill or self.q_b_proj.quantized:
            q = self.q_b_proj(normalized_q).view(
                -1,
                self.local_heads,
                self.qk_head_dim,
            )
            q_nope, q_rope = q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim],
                dim=-1,
            )
            with profiler.trace("mla.qk_rope"):
                q_rope, k_rope = rotary_emb(
                    positions,
                    q_rope,
                    k_rope.unsqueeze(1),
                )
            k_rope = k_rope.squeeze(1)
            with profiler.trace("mla.q_assembly"):
                # The projection owns Q; only its rotated tail changed. Reuse
                # it instead of copying the much wider non-RoPE portion.
                q[..., self.qk_nope_head_dim:].copy_(q_rope)
        else:
            q, k_rope = project_and_fuse_glm_mla_decode_rope(
                normalized_q,
                self.q_b_proj.weight,
                k_rope,
                positions,
                rotary_emb.cos_sin_cache,
                num_heads=self.local_heads,
                nope_dim=self.qk_nope_head_dim,
            )
            q_nope, q_rope = q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim],
                dim=-1,
            )
        value_output = self.mla_attention.run_cached_attention(
            q,
            q_nope,
            q_rope,
            latent,
            k_rope,
            project_latent=self._project_kv_history,
            absorb_query=self._decode_absorbed_query,
            reconstruct_values=self._reconstruct_decode_values,
        )
        output = self._project_output(value_output, hidden_states)
        if self.parallel_collectives is None:
            return output
        return self.parallel_collectives.attention.run(output)


class Glm4MoeLiteRouter(nn.Module):
    def __init__(self, config: Glm4MoeLiteConfig) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_experts = int(config.n_routed_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.routed_scaling_factor = float(config.routed_scaling_factor)
        self.weight = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.hidden_size,
                dtype=torch.float32,
            )
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts, dtype=torch.float32)
        )
        self.op_spec = MoeRouterOpSpec(
            num_experts=self.num_experts,
            top_k=self.top_k,
            activation_dtype=torch.float32,
            norm_topk_prob=True,
            cuda_graph=bool(getattr(config, "decode_graph", False)),
            routing_method="biased_sigmoid",
        )
        self.provider = resolve_moe_router_provider(self.op_spec)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = F.linear(hidden_states.float(), self.weight)
        topk_weights, topk_ids = self.provider.run(
            self.op_spec,
            router_logits,
            self.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        return router_logits, topk_weights, topk_ids


class Glm4MoeLitePackedExperts(PackedMoeExperts):
    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        *,
        decode_graph: bool,
    ) -> None:
        parallel_context = get_parallel_context()
        self.routed_num_experts = int(config.n_routed_experts)
        self.routed_top_k = int(config.num_experts_per_tok)
        fp8_enabled = bool(getattr(getattr(config, "quantization_config", None), "enabled", False))
        fp8_tensor_scales = fp8_enabled and config.quantization_config.checkpoint_scale_layout == "per_tensor"
        self.shared_fusion_token_limit = packed_shared_expert_token_limit(
            torch.float8_e4m3fn if fp8_enabled else model_activation_dtype(config))
        self.fuses_shared_decode = use_packed_shared_experts(
            num_routed_experts=self.routed_num_experts,
            num_shared_experts=int(config.n_shared_experts),
            top_k=self.routed_top_k,
            hidden_size=int(config.hidden_size),
            intermediate_size=int(config.moe_intermediate_size),
            tp_size=int(parallel_context.moe_tp_size),
            ep_size=int(parallel_context.moe_ep_size),
            cuda_graph=decode_graph,
            weight_dtype=torch.float8_e4m3fn if fp8_enabled else model_activation_dtype(config),
            fp8_tensor_scales=fp8_tensor_scales,
        )
        self.fuses_shared_prefill = use_packed_shared_experts_in_prefill(
            num_routed_experts=self.routed_num_experts,
            num_shared_experts=int(config.n_shared_experts),
            top_k=self.routed_top_k,
            hidden_size=int(config.hidden_size),
            intermediate_size=int(config.moe_intermediate_size),
            tp_size=int(parallel_context.moe_tp_size),
            ep_size=int(parallel_context.moe_ep_size),
            cuda_graph=decode_graph,
            weight_dtype=torch.float8_e4m3fn if fp8_enabled else model_activation_dtype(config),
        )
        if self.fuses_shared_prefill and not self.fuses_shared_decode:
            raise RuntimeError("Prefill shared-expert fusion requires packed experts.")
        packed_num_experts = self.routed_num_experts + int(
            self.fuses_shared_decode
        )
        packed_top_k = self.routed_top_k + int(self.fuses_shared_decode)
        super().__init__(
            num_experts=packed_num_experts,
            hidden_size=int(config.hidden_size),
            intermediate_size=int(config.moe_intermediate_size),
            top_k=packed_top_k,
            activation_dtype=model_activation_dtype(config),
            fp8_enabled=fp8_enabled,
            fp8_tensor_scales=fp8_tensor_scales,
            cuda_graph=bool(decode_graph),
            routing_method="biased_sigmoid",
            max_num_tokens=int(getattr(config, "moe_max_num_tokens", 1)),
            model_label="GLM-4.7-Flash",
            provider_resolver=resolve_moe_provider,
            parallel_context=parallel_context,
        )
        self.shared_expert_id = (
            self.routed_num_experts if self.fuses_shared_decode else None
        )
        self.shared_act = SiluAndMul(
            provider=resolve_silu_and_mul_provider(
                activation_dtype=model_activation_dtype(config),
            )
        )
        if self.fuses_shared_decode:
            self.routed_op_spec = MoeOpSpec(
                num_experts=self.routed_num_experts,
                num_local_experts=self.routed_num_experts,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                top_k=self.routed_top_k,
                activation_dtype=self.op_spec.activation_dtype,
                weight_dtype=self.op_spec.weight_dtype,
                block_shape=self.op_spec.block_shape,
                ep_size=1,
                cuda_graph=self.op_spec.cuda_graph,
                tp_size=self.op_spec.tp_size,
                routing_method=self.op_spec.routing_method,
                scale_dtype=self.op_spec.scale_dtype,
                max_num_tokens=self.op_spec.max_num_tokens,
            )
            self.routed_provider = resolve_moe_provider(self.routed_op_spec)
            self.routed_provider.prepare(
                self.routed_op_spec,
                device=self.w13_weight.device,
                tp_rank=int(self.tp_rank),
                ep_rank=int(self.ep_rank),
            )
        else:
            self.routed_op_spec = self.op_spec
            self.routed_provider = self.provider

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if not self.fuses_shared_decode:
            return super().forward(hidden_states, topk_ids, topk_weights)
        return self.routed_provider.run(
            self.routed_op_spec,
            hidden_states,
            topk_ids,
            topk_weights,
            self.w13_weight[: self.routed_num_experts],
            self.w2_weight[: self.routed_num_experts],
            None if self.w13_scale_inv is None else self.w13_scale_inv[:self.routed_num_experts],
            None if self.w2_scale_inv is None else self.w2_scale_inv[:self.routed_num_experts],
            local_expert_start=0,
            tp_rank=int(self.tp_rank),
            ep_rank=int(self.ep_rank),
        )

    def forward_shared(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.shared_expert_id is None:
            raise RuntimeError("Packed shared expert is not enabled.")
        if self.fp8_enabled:
            return self.provider.run_shared_expert(hidden_states)
        gate_up = F.linear(
            hidden_states,
            self.w13_weight[self.shared_expert_id],
        )
        return F.linear(
            self.shared_act(gate_up),
            self.w2_weight[self.shared_expert_id],
        )

    def validate_loaded_weights(self):
        super().validate_loaded_weights()
        if self.shared_expert_id is not None and self.fp8_enabled:
            idx = self.shared_expert_id
            self.provider.prepare_shared_expert(
                self.op_spec, self.w13_weight[idx], self.w2_weight[idx],
                self.w13_scale_inv[idx], self.w2_scale_inv[idx])

    def forward_routed_and_shared(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.shared_expert_id is None:
            raise RuntimeError("Packed shared expert is not enabled.")
        fused_ids, fused_weights = append_shared_expert_route(
            topk_ids,
            topk_weights,
            shared_expert_id=self.shared_expert_id,
        )
        return super().forward(hidden_states, fused_ids, fused_weights)


class Glm4MoeLiteSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        *,
        mlp_chunk_size: int,
        decode_graph: bool,
        parallel_collectives: DecodeParallelCollectives | None = None,
    ) -> None:
        super().__init__()
        self.parallel_context = get_parallel_context()
        self.parallel_collectives = parallel_collectives
        self.mlp_chunk_size = int(mlp_chunk_size)
        self.moe_communication = prepare_moe_communication(self.parallel_context, parallel_collectives)
        if self.mlp_chunk_size <= 0:
            raise ValueError(
                f"mlp_chunk_size must be positive, got {self.mlp_chunk_size}."
            )
        self.gate = Glm4MoeLiteRouter(config)
        self.experts = Glm4MoeLitePackedExperts(
            config,
            decode_graph=decode_graph,
        )
        self.shared_experts = (
            None
            if self.experts.fuses_shared_decode
            else Qwen3MLP(
                hidden_size=int(config.hidden_size),
                intermediate_size=(
                    int(config.moe_intermediate_size)
                    * int(config.n_shared_experts)
                ),
                hidden_act=str(config.hidden_act),
                mlp_chunk_size=self.mlp_chunk_size,
                quantization=getattr(config, "quantization_config", None),
                reduce_results=False,
                activation_provider=resolve_silu_and_mul_provider(
                    activation_dtype=model_activation_dtype(config),
                ),
            )
        )

        self.moe_execution = (
            self._create_moe_execution()
            if self.parallel_context.attn_dp_size == 1 else None
        )

    def _create_moe_execution(self):
        experts = getattr(self, "experts", None)
        return MoeExecutionPlan(
            routed=self._routed_chunk, shared=self._shared_chunk,
            fused=self._routed_and_shared_chunk,
            communication=self.moe_communication, chunk_size=self.mlp_chunk_size,
            fuse_prefill=getattr(experts, "fuses_shared_prefill", False),
            fuse_decode=getattr(experts, "fuses_shared_decode", False),
            fusion_token_limit=getattr(experts, "shared_fusion_token_limit", None),
            reduce_decode_branches_separately=self.parallel_context.moe_ep_size == 1,
            shared_modules=(
                self.shared_experts if self.shared_experts is not None else experts,
            ),
        )

    def _route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, topk_weights, topk_ids = self.gate(hidden_states)
        return topk_ids, topk_weights

    def _routed_chunk(self, hidden_states: torch.Tensor) -> torch.Tensor:
        topk_ids, topk_weights = self._route(hidden_states)
        return self.experts(hidden_states, topk_ids, topk_weights)

    def _shared_chunk(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.shared_experts is not None:
            return self.shared_experts(hidden_states)
        return self.experts.forward_shared(hidden_states)

    def _routed_and_shared_chunk(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        _, topk_weights, topk_ids = self.gate(hidden_states)
        return self.experts.forward_routed_and_shared(
            hidden_states,
            topk_ids,
            topk_weights,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError(
                "Glm4MoeLiteSparseMoeBlock expects [tokens, hidden], got "
                f"{tuple(hidden_states.shape)}."
            )
        if self.parallel_context.attn_dp_size > 1:
            return self._forward_distributed_tokens(hidden_states)
        debug_enabled = os.getenv("SPARSEENGINE_DEBUG_MOE", "0") == "1"
        if debug_enabled:
            self.debug_last_input = hidden_states.detach().clone()
        if not debug_enabled:
            return self.moe_execution(hidden_states, is_prefill=get_context().is_prefill)
        if int(hidden_states.shape[0]) <= self.mlp_chunk_size:
            router_logits, topk_weights, topk_ids = self.gate(hidden_states)
            routed = self.experts(hidden_states, topk_ids, topk_weights)
        else:
            router_logits_chunks = []
            topk_weights_chunks = []
            topk_ids_chunks = []
            routed_chunks = []
            for chunk in hidden_states.split(self.mlp_chunk_size, dim=0):
                router_logits, topk_weights, topk_ids = self.gate(chunk)
                routed_chunks.append(
                    self.experts(chunk, topk_ids, topk_weights)
                )
                router_logits_chunks.append(router_logits)
                topk_weights_chunks.append(topk_weights)
                topk_ids_chunks.append(topk_ids)
            routed = torch.cat(routed_chunks, dim=0)
            router_logits = torch.cat(router_logits_chunks, dim=0)
            topk_weights = torch.cat(topk_weights_chunks, dim=0)
            topk_ids = torch.cat(topk_ids_chunks, dim=0)
        if debug_enabled:
            self.debug_last_router_logits = router_logits.detach().clone()
            self.debug_last_topk_ids = topk_ids.detach().clone()
            self.debug_last_topk_weights = topk_weights.detach().clone()
            self.debug_last_local_output = routed.detach().clone()
            local_mask = (topk_ids >= self.experts.local_expert_start) & (
                topk_ids < self.experts.local_expert_end
            )
            local_hit_count = local_mask.sum()
            self.debug_last_local_hit_count = (
                local_hit_count
                if device_runtime.is_stream_capturing()
                else int(local_hit_count.item())
            )
        # Debug evidence intentionally keeps routed/shared reductions explicit.
        routed = self.moe_communication.combine(routed)
        self.debug_last_routed_output = routed.detach().clone()
        shared = self._shared_chunk(hidden_states)
        if self.parallel_context.attn_tp_size > 1:
            shared = self.parallel_context.attn_tp.all_reduce(shared)
        output = routed + shared
        if debug_enabled:
            # ModelRunner's cross-rank evidence contract consumes the final
            # MoE block output, including both the synced routed experts and
            # the synchronized shared-expert branch.
            self.debug_last_output = output.detach().clone()
        return output

    def _forward_distributed_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.moe_communication.run_with_shared_experts(
            hidden_states, route=self._route, experts=self.experts,
            chunk_size=self.mlp_chunk_size,
            shared_experts=self._shared_chunk,
        )


class Glm4MoeLiteDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        layer_idx: int,
        mla_attention: MLAAttention,
        *,
        mlp_chunk_size: int,
        decode_graph: bool,
        parallel_collectives: DecodeParallelCollectives | None = None,
    ) -> None:
        super().__init__()
        self.parallel_context = get_parallel_context()
        self.parallel_collectives = parallel_collectives
        self.self_attn = Glm4MoeLiteAttention(
            config,
            mla_attention,
            projection_chunk_size=mlp_chunk_size,
            parallel_collectives=parallel_collectives,
        )
        layer_types = list(config.mlp_layer_types)
        if len(layer_types) != int(config.num_hidden_layers):
            raise ValueError(
                "GLM mlp_layer_types length must match num_hidden_layers, got "
                f"{len(layer_types)} and {config.num_hidden_layers}."
            )
        layer_type = str(layer_types[int(layer_idx)])
        if layer_type == "dense":
            self.mlp = Qwen3MLP(
                hidden_size=int(config.hidden_size),
                intermediate_size=int(config.intermediate_size),
                hidden_act=str(config.hidden_act),
                mlp_chunk_size=int(mlp_chunk_size),
                quantization=getattr(config, "quantization_config", None),
                reduce_results=parallel_collectives is None,
                activation_provider=resolve_silu_and_mul_provider(
                    activation_dtype=model_activation_dtype(config),
                ),
            )
        elif layer_type == "sparse":
            self.mlp = Glm4MoeLiteSparseMoeBlock(
                config,
                mlp_chunk_size=mlp_chunk_size,
                decode_graph=decode_graph,
                parallel_collectives=parallel_collectives,
            )
        else:
            raise ValueError(
                f"Unsupported GLM MLP layer type at layer {layer_idx}: {layer_type!r}."
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        rotary_emb: RotaryEmbedding,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, rotary_emb)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        hidden_states = self.mlp(hidden_states)
        if self.parallel_collectives is not None and isinstance(self.mlp, Qwen3MLP):
            hidden_states = self.parallel_collectives.attention.run(hidden_states)
        return hidden_states, residual


class Glm4MoeLiteModel(nn.Module):
    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        mla_attention: MLAAttention,
        *,
        mlp_chunk_size: int,
        decode_graph: bool,
        parallel_collectives: DecodeParallelCollectives | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mla_attention = mla_attention
        self.parallel_context = get_parallel_context()
        self.parallel_collectives = parallel_collectives
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            reduce_results=parallel_collectives is None,
        )
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        self.rotary_emb = get_rope(
            int(config.qk_rope_head_dim),
            rotary_dim=int(config.qk_rope_head_dim),
            max_position=int(config.max_position_embeddings),
            base=float(rope_parameters.get("rope_theta", 1_000_000.0)),
            rope_scaling=None,
            backend="torch",
            interleaved=True,
        )
        self.layers = nn.ModuleList(
            [
                Glm4MoeLiteDecoderLayer(
                    config,
                    layer_idx,
                    mla_attention,
                    mlp_chunk_size=mlp_chunk_size,
                    decode_graph=decode_graph,
                    parallel_collectives=parallel_collectives,
                )
                for layer_idx in range(int(config.num_hidden_layers))
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.sparse_controller = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        context = get_context()
        hidden_states = self.embed_tokens(input_ids)
        if self.parallel_collectives is not None:
            hidden_states = self.parallel_collectives.attention.run(hidden_states)
        residual = None
        debug_layers_env = os.getenv("SPARSEENGINE_DEBUG_HIDDEN_LAYERS")
        debug_layers = None
        if debug_layers_env:
            debug_layers = {
                int(part) for part in debug_layers_env.split(",") if part.strip()
            }
            self.debug_last_hidden_states = {
                -1: hidden_states[-1:].detach().clone(),
            }

        for layer_idx, layer in enumerate(self.layers):
            context.now_layer_idx = layer_idx
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                self.rotary_emb,
            )
            if self.sparse_controller is not None:
                hidden_states, residual = self.sparse_controller.apply_activation_hook(
                    layer_idx,
                    hidden_states,
                    residual,
                    context,
                )
            if debug_layers is not None and layer_idx in debug_layers:
                layer_output = (
                    hidden_states if residual is None else hidden_states + residual
                )
                self.debug_last_hidden_states[layer_idx] = (
                    layer_output[-1:].detach().clone()
                )
            if self.sparse_controller is not None:
                self.sparse_controller.on_layer_end(layer_idx, context)

        hidden_states, _ = self.norm(hidden_states, residual)
        if debug_layers is not None:
            self.debug_last_hidden_states[int(self.config.num_hidden_layers)] = (
                hidden_states[-1:].detach().clone()
            )
        return hidden_states


class Glm4MoeLiteForCausalLM(nn.Module):
    special_weight_loaders = (".expert_weight",)
    packed_modules_mapping = {
        "self_attn.q_a_proj": ("self_attn.fused_qkv_a_proj", 0),
        "self_attn.kv_a_proj_with_mqa": ("self_attn.fused_qkv_a_proj", 1),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    @staticmethod
    def build_runtime_kwargs(
        config: Glm4MoeLiteConfig,
        *,
        engine_config,
        parallel_context: ParallelContext,
        collective_runtime: ParallelCollectiveRuntime,
        device: torch.device,
        max_decode_tokens: int,
    ) -> dict:
        decode_graph = bool(engine_config.decode_graph)
        decode_context_capacity = int(engine_config.max_model_len)
        if engine_config.sparse_method == "quest":
            page_size = int(engine_config.quest_chunk_size)
            page_budget = max(
                3,
                int(engine_config.quest_token_budget) // page_size,
            )
            decode_context_capacity = min(
                decode_context_capacity,
                max(
                    int(engine_config.quest_token_budget),
                    page_budget * page_size,
                    page_size,
                ),
            )
        kwargs = {
            "mla_attention": build_glm4_moe_lite_mla_attention(
                config,
                device=device,
                max_batch_size=max(
                    engine_config.max_num_seqs_in_batch,
                    engine_config.max_decoding_seqs,
                ),
                prefill_workspace_bytes=engine_config.mla_prefill_workspace_bytes,
                history_chunk_size=engine_config.mla_prefill_history_chunk_size,
                decode_graph=decode_graph,
                context_capacity=decode_context_capacity,
                projection_chunk_size=engine_config.mlp_chunk_size,
                score_output=sparse_decode_attention_score_kind(
                    engine_config.sparse_method,
                    h2o_decode_eviction=getattr(engine_config, "h2o_decode_eviction", False),
                    attention_cache_layout=engine_config.attention_cache_layout,
                ),
            ),
            "mlp_chunk_size": engine_config.mlp_chunk_size,
            "decode_graph": decode_graph,
        }
        kwargs["parallel_collectives"] = collective_runtime.request_moe_collectives(
            attention_max_rows=int(max_decode_tokens),
            moe_max_rows=2 * int(max_decode_tokens),
            max_local_tokens=engine_config.max_num_batched_tokens,
            hidden_size=int(config.hidden_size),
            dtype=model_activation_dtype(config),
            backend=engine_config.moe_backend,
            num_experts=int(config.n_routed_experts),
            top_k=int(config.num_experts_per_tok),
        )
        return kwargs

    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        *,
        mla_attention: MLAAttention,
        mlp_chunk_size: int,
        decode_graph: bool,
        parallel_collectives: DecodeParallelCollectives | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.parallel_context = get_parallel_context()
        self.parallel_collectives = parallel_collectives
        self.model = Glm4MoeLiteModel(
            config,
            mla_attention,
            mlp_chunk_size=mlp_chunk_size,
            decode_graph=decode_graph,
            parallel_collectives=parallel_collectives,
        )
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        self.ignored_weight_prefixes = tuple(
            f"model.layers.{layer_idx}."
            for layer_idx in range(
                int(config.num_hidden_layers),
                int(config.num_hidden_layers)
                + int(getattr(config, "num_nextn_predict_layers", 0)),
            )
        )

    def release_cache_runtime_bindings(self, cache_manager: object) -> None:
        self.model.mla_attention.release_cache_runtime_bindings(cache_manager)

    def _sparse_block(self, layer_idx: int) -> Glm4MoeLiteSparseMoeBlock:
        if not 0 <= int(layer_idx) < len(self.model.layers):
            raise ValueError(
                f"GLM expert checkpoint layer {layer_idx} is outside the base model."
            )
        block = self.model.layers[int(layer_idx)].mlp
        if not isinstance(block, Glm4MoeLiteSparseMoeBlock):
            raise ValueError(
                f"GLM checkpoint contains experts for dense layer {layer_idx}."
            )
        return block

    def iter_tiny_reference_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ):
        """Expand Transformers' packed tiny experts into checkpoint-style names."""

        gate_up_suffix = ".mlp.experts.gate_up_proj"
        down_suffix = ".mlp.experts.down_proj"
        for source_name, weight in state_dict.items():
            if source_name.endswith(gate_up_suffix):
                prefix = source_name[: -len("gate_up_proj")]
                intermediate_size = int(weight.shape[1]) // 2
                for expert_id in range(int(weight.shape[0])):
                    yield (
                        f"{prefix}{expert_id}.gate_proj.weight",
                        weight[expert_id, :intermediate_size],
                    )
                    yield (
                        f"{prefix}{expert_id}.up_proj.weight",
                        weight[expert_id, intermediate_size:],
                    )
                continue
            if source_name.endswith(down_suffix):
                prefix = source_name[: -len("down_proj")]
                for expert_id in range(int(weight.shape[0])):
                    yield f"{prefix}{expert_id}.down_proj.weight", weight[expert_id]
                continue
            yield source_name, weight

    def map_weight_name(self, source_weight_name: str) -> str | None:
        shared_match = _SHARED_EXPERT_SOURCE_RE.match(source_weight_name)
        if shared_match is not None:
            layer_idx, projection = shared_match.groups()
            experts = self._sparse_block(int(layer_idx)).experts
            if experts.shared_expert_id is not None:
                return (
                    f"model.layers.{layer_idx}.mlp.experts."
                    f"{experts.shared_expert_id}.{projection}.expert_weight"
                )
        match = _EXPERT_SOURCE_RE.match(source_weight_name)
        if match is None:
            return source_weight_name
        layer_idx, global_expert_id, projection = match.groups()
        experts = self._sparse_block(int(layer_idx)).experts
        if not experts.is_local_expert(int(global_expert_id)):
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
        return WeightTarget(
            self._sparse_block(int(layer_idx)).experts,
            (int(global_expert_id), projection),
        )

    def load_special_weight(
        self,
        target_weight_name: str,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None,
    ) -> int:
        target = self.resolve_special_weight(target_weight_name)
        if target is None:
            return 0
        expert_id, projection = target.shard_id
        target.module.load_expert_weight(
            expert_id,
            projection,
            loaded_weight,
            loaded_scale,
        )
        return 1

    def validate_loaded_weights(self, loaded_parameter_names: set[str]) -> None:
        packed_experts = {
            name
            for name, _ in self.named_parameters()
            if name.endswith((".mlp.experts.w13_weight", ".mlp.experts.w2_weight"))
        }
        missing = sorted(
            {name for name, _ in self.named_parameters()}
            - packed_experts
            - loaded_parameter_names
        )
        if missing:
            raise ValueError(f"Missing replicated GLM base weights: {missing[:8]}.")
        for layer_idx, layer_type in enumerate(self.config.mlp_layer_types):
            if layer_type == "sparse":
                self._sparse_block(layer_idx).experts.validate_loaded_weights()

    @torch.inference_mode()
    def warmup_moe(self, num_tokens: int = 1) -> None:
        num_tokens = int(num_tokens)
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}.")
        block = next(
            (
                layer.mlp
                for layer in self.model.layers
                if isinstance(layer.mlp, Glm4MoeLiteSparseMoeBlock)
            ),
            None,
        )
        if block is None:
            raise RuntimeError("GLM model has no sparse MoE layer to warm up.")
        experts = block.experts
        hidden_states = torch.zeros(
            (num_tokens, experts.hidden_size),
            dtype=model_activation_dtype(self.config),
            device=experts.w13_weight.device,
        )
        block.gate(hidden_states)
        top_k = int(self.config.num_experts_per_tok)
        topk_ids = (
            torch.arange(
                num_tokens * top_k,
                dtype=torch.int64,
                device=hidden_states.device,
            )
            .remainder(experts.routed_num_experts)
            .add(experts.local_expert_start)
            .view(num_tokens, top_k)
        )
        topk_weights = torch.full(
            (num_tokens, top_k),
            1.0 / top_k,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        experts(hidden_states, topk_ids, topk_weights)
        device_runtime.synchronize()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def forward_idle_experts(self, hidden_states: torch.Tensor) -> None:
        for layer in self.model.layers:
            if isinstance(layer.mlp, Glm4MoeLiteSparseMoeBlock):
                layer.mlp(hidden_states)


__all__ = [
    "Glm4MoeLiteAttention",
    "Glm4MoeLiteForCausalLM",
    "Glm4MoeLiteModel",
    "Glm4MoeLitePackedExperts",
    "Glm4MoeLiteRouter",
    "Glm4MoeLiteSparseMoeBlock",
    "build_glm4_moe_lite_mla_attention",
]
