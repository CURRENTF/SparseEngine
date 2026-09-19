from __future__ import annotations

import re
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from sparseengine.distributed import (
    DecodeParallelCollectives,
    ParallelContext,
    ParallelCollectiveRuntime,
    get_parallel_context,
)
from sparseengine.distributed.moe_communication import prepare_moe_communication
from sparseengine.layers.attention import Attention
from sparseengine.layers.embed_head import ParallelLMHead
from sparseengine.layers.layernorm import ColumnParallelRMSNorm, RMSNorm
from sparseengine.layers.linear import QKVParallelLinear, RowParallelLinear
from sparseengine.layers.packed_moe import PackedMoeExperts
from sparseengine.layers.rotary_embedding import (
    apply_partial_rotary_emb,
    get_rope,
)
from sparseengine.models.qwen3 import Qwen3ModelBase
from sparseengine.models.rope import (
    resolve_rope_max_position,
    resolve_rope_scaling,
    resolve_rope_theta,
)
from sparseengine.models.attention_runtime import (
    bind_mha_full_attention_provider,
    build_mha_full_attention_provider,
)
from sparseengine.operators.moe import model_activation_dtype, resolve_moe_provider
from sparseengine.operators.moe_router import (
    MoeRouterOpSpec,
    resolve_moe_router_provider,
)
from sparseengine.operators.decode_attention import (
    DecodeAttentionLaunchSpec,
    PreparedDecodeAttentionLaunchOp,
    prepare_decode_attention_launch_op,
)
from sparseengine.operators.full_attention import FullAttentionProvider
from sparseengine.platforms import device_runtime
from sparseengine.utils.context import get_context
from sparseengine.utils.log import logger
from sparseengine.utils.weight_target import WeightTarget


_EXPERT_SOURCE_RE = re.compile(
    r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\."
    r"(w1|w2|w3)\.weight$"
)
_EXPERT_TARGET_RE = re.compile(
    r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\."
    r"(w1|w2|w3)\.expert_weight$"
)


@dataclass
class MiniMaxM2RuntimeConfig:
    full_attention_provider: FullAttentionProvider
    decode_launch_op: PreparedDecodeAttentionLaunchOp
    parallel_collectives: DecodeParallelCollectives
    cuda_graph: bool
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self.full_attention_provider.close()
        self._closed = True


def build_minimax_m2_runtime_config(
    config,
    parallel_context: ParallelContext,
    collective_runtime: ParallelCollectiveRuntime,
    *,
    sparse_method: str | None,
    max_decode_tokens: int,
    cuda_graph: bool,
    device: torch.device,
    engine_config=None,
) -> MiniMaxM2RuntimeConfig:
    tp_size = int(parallel_context.attn_tp_size)
    if (
        int(config.num_attention_heads) % tp_size
        or int(config.num_key_value_heads) % tp_size
    ):
        raise ValueError(
            "MiniMax attention heads must be divisible by attention TP size."
        )
    num_query_heads = int(config.num_attention_heads) // tp_size
    num_kv_heads = int(config.num_key_value_heads) // tp_size
    head_dim = int(config.head_dim)
    activation_dtype = model_activation_dtype(config)
    full_attention_provider = build_mha_full_attention_provider(
        config,
        sparse_method=sparse_method,
        attention_tp_size=parallel_context.attn_tp_size,
        device=device,
        max_batch_size=max_decode_tokens,
        cuda_graph=cuda_graph,
        runtime_config=engine_config,
    )
    decode_launch_op = prepare_decode_attention_launch_op(
        DecodeAttentionLaunchSpec(
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            activation_dtype=activation_dtype,
            page_size=1,
        ),
        device_index=int(device.index or 0),
    )
    parallel_collectives = collective_runtime.request_moe_collectives(
        attention_max_rows=int(max_decode_tokens),
        moe_max_rows=int(max_decode_tokens),
        max_local_tokens=(
            max_decode_tokens if engine_config is None else engine_config.max_num_batched_tokens
        ),
        backend=("all-reduce" if engine_config is None else engine_config.moe_backend),
        num_experts=int(config.num_local_experts), top_k=int(config.num_experts_per_tok),
        hidden_size=int(config.hidden_size),
        dtype=activation_dtype,
    )
    return MiniMaxM2RuntimeConfig(
        full_attention_provider=full_attention_provider,
        decode_launch_op=decode_launch_op,
        parallel_collectives=parallel_collectives,
        cuda_graph=bool(cuda_graph),
    )


class MiniMaxM2Router(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_experts = int(config.num_local_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, dtype=torch.float32)
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
        )
        return router_logits, topk_weights, topk_ids


class MiniMaxM2PackedExperts(PackedMoeExperts):
    checkpoint_projection_map = {"w1": "gate", "w2": "down", "w3": "up"}
    checkpoint_scale_dtype = torch.float32

    def __init__(
        self,
        config,
        *,
        cuda_graph: bool | None = None,
    ) -> None:
        block_shape = tuple(config.quantization_config.weight_block_size)
        if block_shape != (128, 128):
            raise ValueError(
                "MiniMax packed FP8 experts require weight_block_size=(128, 128), "
                f"got {block_shape}."
            )
        super().__init__(
            num_experts=int(config.num_local_experts),
            hidden_size=int(config.hidden_size),
            intermediate_size=int(config.intermediate_size),
            top_k=int(config.num_experts_per_tok),
            activation_dtype=model_activation_dtype(config),
            fp8_enabled=True,
            cuda_graph=(
                bool(getattr(config, "decode_graph", False))
                if cuda_graph is None
                else bool(cuda_graph)
            ),
            routing_method="biased_sigmoid",
            scale_dtype=torch.float32,
            max_num_tokens=int(getattr(config, "moe_max_num_tokens", 1)),
            model_label="MiniMax",
            provider_resolver=resolve_moe_provider,
            parallel_context=get_parallel_context(),
        )


class MiniMaxM2SparseMoeBlock(nn.Module):
    def __init__(
        self,
        config,
        runtime_config: MiniMaxM2RuntimeConfig | None = None,
    ) -> None:
        super().__init__()
        self.parallel_context = get_parallel_context()
        self.runtime_config = runtime_config
        self.mlp_chunk_size = int(getattr(config, "mlp_chunk_size", 16384))
        if self.mlp_chunk_size <= 0:
            raise ValueError(
                f"mlp_chunk_size must be > 0, got {self.mlp_chunk_size}."
            )
        self.moe_communication = prepare_moe_communication(
            self.parallel_context,
            None if runtime_config is None else runtime_config.parallel_collectives,
        )
        self.gate = MiniMaxM2Router(config)
        self.experts = MiniMaxM2PackedExperts(
            config,
            cuda_graph=(None if runtime_config is None else runtime_config.cuda_graph),
        )

    @property
    def e_score_correction_bias(self) -> nn.Parameter:
        return self.gate.e_score_correction_bias

    def _route(self, hidden_states):
        _, weights, ids = self.gate(hidden_states)
        return ids, weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError(
                f"MiniMax M2 MoE expects [tokens, hidden], got {tuple(hidden_states.shape)}."
            )
        if self.parallel_context.attn_dp_size > 1:
            return self.moe_communication.run(
                hidden_states, route=self._route, experts=self.experts,
                chunk_size=self.mlp_chunk_size,
            )
        if int(hidden_states.shape[0]) <= self.mlp_chunk_size:
            _, topk_weights, topk_ids = self.gate(hidden_states)
            local_output = self.experts(
                hidden_states,
                topk_ids,
                topk_weights,
            )
        else:
            local_output_chunks = []
            for chunk in hidden_states.split(self.mlp_chunk_size, dim=0):
                _, topk_weights, topk_ids = self.gate(chunk)
                local_output_chunks.append(
                    self.experts(chunk, topk_ids, topk_weights)
                )
            local_output = torch.cat(local_output_chunks, dim=0)
        return self.moe_communication.combine(local_output)


class MiniMaxM2Attention(nn.Module):
    def __init__(
        self,
        config,
        runtime_config: MiniMaxM2RuntimeConfig | None = None,
    ) -> None:
        super().__init__()
        self.parallel_context = get_parallel_context()
        self.runtime_config = runtime_config
        tp_size = int(self.parallel_context.attn_tp_size)
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_heads % tp_size or self.total_num_kv_heads % tp_size:
            raise ValueError("MiniMax attention heads must be divisible by TP size.")
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = int(config.head_dim)
        self.rotary_dim = int(config.rotary_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            int(config.hidden_size),
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quantization=config.quantization_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            int(config.hidden_size),
            bias=False,
            quantization=config.quantization_config,
            reduce_results=runtime_config is None,
        )
        self.q_norm = ColumnParallelRMSNorm(
            self.total_num_heads * self.head_dim,
            eps=float(config.rms_norm_eps),
            parallel_context=self.parallel_context,
        )
        self.k_norm = ColumnParallelRMSNorm(
            self.total_num_kv_heads * self.head_dim,
            eps=float(config.rms_norm_eps),
            parallel_context=self.parallel_context,
        )
        self.rotary_emb = get_rope(
            self.rotary_dim,
            rotary_dim=self.rotary_dim,
            max_position=resolve_rope_max_position(config, model_name="MiniMax M2"),
            base=resolve_rope_theta(config),
            rope_scaling=resolve_rope_scaling(config, model_name="MiniMax M2"),
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            self.num_kv_heads,
            decode_launch_op=(
                None if runtime_config is None else runtime_config.decode_launch_op
            ),
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        context = get_context()
        layer_idx = context.now_layer_idx
        raw_k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context.cache_manager.save_raw_kv_if_needed(layer_idx, raw_k, v)
        q, k = self.q_norm.forward_pair(q, k, self.k_norm)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        q, k = apply_partial_rotary_emb(
            self.rotary_emb,
            positions,
            q,
            k,
            self.rotary_dim,
        )
        context.cache_manager.save_rope_kv_if_needed(layer_idx, k, v)
        output = self.attn(q, k, v).flatten(1, -1)
        output = self.o_proj(output)
        if self.runtime_config is not None:
            return self.runtime_config.parallel_collectives.attention.run(output)
        return output


class MiniMaxM2DecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        runtime_config: MiniMaxM2RuntimeConfig | None = None,
    ) -> None:
        super().__init__()
        self.parallel_context = get_parallel_context()
        self.self_attn = MiniMaxM2Attention(config, runtime_config)
        self.block_sparse_moe = MiniMaxM2SparseMoeBlock(config, runtime_config)
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

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
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual


class MiniMaxM2Model(Qwen3ModelBase):
    def __init__(
        self,
        config,
        runtime_config: MiniMaxM2RuntimeConfig | None = None,
    ) -> None:
        layer_factory = partial(
            MiniMaxM2DecoderLayer,
            runtime_config=runtime_config,
        )
        super().__init__(config, layer_factory)
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )


class MiniMaxM2ForCausalLM(nn.Module):
    special_weight_loaders = (".expert_weight",)
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    @staticmethod
    def build_runtime_kwargs(
        config,
        *,
        engine_config,
        parallel_context: ParallelContext,
        collective_runtime: ParallelCollectiveRuntime,
        device: torch.device,
        max_decode_tokens: int,
    ) -> dict:
        return {
            "runtime_config": build_minimax_m2_runtime_config(
                config,
                parallel_context,
                collective_runtime,
                sparse_method=engine_config.sparse_method,
                max_decode_tokens=max_decode_tokens,
                cuda_graph=engine_config.decode_graph,
                device=device,
                engine_config=engine_config,
            )
        }

    def __init__(
        self,
        config,
        runtime_config: MiniMaxM2RuntimeConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.parallel_context = get_parallel_context()
        self.runtime_config = runtime_config
        self.model = MiniMaxM2Model(config, runtime_config)
        if runtime_config is not None:
            bind_mha_full_attention_provider(
                self.model,
                runtime_config.full_attention_provider,
            )
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self._intentionally_skipped_expert_weights: set[str] = set()
        self._intentionally_skipped_expert_scales: set[str] = set()

    def close_runtime_operators(self) -> None:
        if self.runtime_config is not None:
            self.runtime_config.close()

    @torch.inference_mode()
    def warmup_moe(self, num_tokens: int = 1) -> None:
        num_tokens = int(num_tokens)
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be > 0, got {num_tokens}.")
        layer = self.model.layers[0]
        experts = layer.block_sparse_moe.experts
        device = experts.w13_weight.device
        hidden_states = torch.zeros(
            (num_tokens, experts.hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
        layer.self_attn.qkv_proj(hidden_states)
        layer.self_attn.o_proj(
            torch.zeros(
                (num_tokens, layer.self_attn.q_size),
                dtype=hidden_states.dtype,
                device=device,
            )
        )
        layer.block_sparse_moe.gate(hidden_states)
        top_k = int(self.config.num_experts_per_tok)
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
            dtype=torch.float32,
            device=device,
        )
        experts(hidden_states, topk_ids, topk_weights)
        device_runtime.synchronize()

    def map_weight_name(self, source_weight_name: str) -> str | None:
        if source_weight_name.endswith(".block_sparse_moe.e_score_correction_bias"):
            return source_weight_name.replace(
                ".block_sparse_moe.e_score_correction_bias",
                ".block_sparse_moe.gate.e_score_correction_bias",
            )
        match = _EXPERT_SOURCE_RE.match(source_weight_name)
        if match is None:
            return source_weight_name
        layer_idx, global_expert_id, projection = match.groups()
        global_expert_id = int(global_expert_id)
        experts = self.model.layers[int(layer_idx)].block_sparse_moe.experts
        if not experts.is_local_expert(global_expert_id):
            return None
        return (
            f"model.layers.{layer_idx}.block_sparse_moe.experts."
            f"{global_expert_id}.{projection}.expert_weight"
        )

    def resolve_special_weight(
        self,
        target_weight_name: str,
    ) -> WeightTarget | None:
        match = _EXPERT_TARGET_RE.match(target_weight_name)
        if match is None:
            return None
        layer_idx, global_expert_id, projection = match.groups()
        experts = self.model.layers[int(layer_idx)].block_sparse_moe.experts
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
                f"MiniMax loader unexpectedly skipped {source_weight_name!r}."
            )
        layer_idx, global_expert_id, projection = match.groups()
        experts = self.model.layers[int(layer_idx)].block_sparse_moe.experts
        if experts.is_local_expert(int(global_expert_id)):
            raise ValueError(
                f"MiniMax loader skipped local expert weight {source_weight_name!r}."
            )
        if loaded_weight_shape is None or loaded_weight_dtype is None:
            raise ValueError(
                f"Skipped remote MiniMax expert is missing weight metadata: "
                f"{source_weight_name!r}."
            )
        if loaded_weight_dtype != "F8_E4M3":
            raise TypeError(
                "Remote MiniMax expert weight must be FP8 E4M3, got "
                f"safetensors dtype {loaded_weight_dtype}."
            )
        if loaded_scale_shape is None or loaded_scale_dtype is None:
            raise ValueError(
                f"Skipped remote MiniMax expert is missing weight_scale_inv: "
                f"{source_weight_name!r}."
            )
        if loaded_scale_dtype != "F32":
            raise TypeError(
                "Remote MiniMax expert scale must be FP32, got safetensors dtype "
                f"{loaded_scale_dtype}."
            )
        expected_shape = (
            (experts.hidden_size // 128, experts.global_intermediate_size // 128)
            if projection == "w2"
            else (experts.global_intermediate_size // 128, experts.hidden_size // 128)
        )
        expected_weight_shape = (
            (experts.hidden_size, experts.global_intermediate_size)
            if projection == "w2"
            else (experts.global_intermediate_size, experts.hidden_size)
        )
        if loaded_weight_shape != expected_weight_shape:
            raise ValueError(
                "Remote MiniMax expert weight shape mismatch for "
                f"{source_weight_name!r}: expected={expected_weight_shape}, "
                f"got={loaded_weight_shape}."
            )
        if loaded_scale_shape != expected_shape:
            raise ValueError(
                "Remote MiniMax expert scale shape mismatch for "
                f"{source_weight_name!r}: expected={expected_shape}, "
                f"got={loaded_scale_shape}."
            )
        self._intentionally_skipped_expert_weights.add(source_weight_name)
        self._intentionally_skipped_expert_scales.add(
            source_weight_name[: -len(".weight")] + ".weight_scale_inv"
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
            if name.endswith(".block_sparse_moe.experts.w13_weight")
            or name.endswith(".block_sparse_moe.experts.w2_weight")
        }
        expected_dense = {
            name for name, _ in self.named_parameters()
        } - packed_expert_parameters
        missing_dense = sorted(expected_dense - loaded_parameter_names)
        if missing_dense:
            raise ValueError(
                f"Missing replicated MiniMax M2 weights: {missing_dense[:8]}."
            )
        for layer in self.model.layers:
            layer.block_sparse_moe.experts.validate_loaded_weights()

        expected_skipped_weights = {
            f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}."
            f"{projection}.weight"
            for layer_idx in range(int(self.config.num_hidden_layers))
            for expert_id in range(int(self.config.num_local_experts))
            if not self.model.layers[
                layer_idx
            ].block_sparse_moe.experts.is_local_expert(expert_id)
            for projection in ("w1", "w2", "w3")
        }
        expected_skipped_scales = {
            name[: -len(".weight")] + ".weight_scale_inv"
            for name in expected_skipped_weights
        }
        missing_skips = sorted(
            expected_skipped_weights - self._intentionally_skipped_expert_weights
        )
        unexpected_skips = sorted(
            self._intentionally_skipped_expert_weights - expected_skipped_weights
        )
        missing_scale_skips = sorted(
            expected_skipped_scales - self._intentionally_skipped_expert_scales
        )
        unexpected_scale_skips = sorted(
            self._intentionally_skipped_expert_scales - expected_skipped_scales
        )
        if missing_skips or missing_scale_skips:
            raise ValueError(
                "Checkpoint is missing expected remote MiniMax expert entries: "
                f"weights={missing_skips[:4]}, scales={missing_scale_skips[:4]}."
            )
        if unexpected_skips or unexpected_scale_skips:
            raise ValueError(
                "Unexpectedly skipped MiniMax expert entries: "
                f"weights={unexpected_skips[:4]}, scales={unexpected_scale_skips[:4]}."
            )
        prefill_provider = (
            self.runtime_config.full_attention_provider.prefill_name
            if self.runtime_config is not None
            else "legacy_triton"
        )
        if self.runtime_config is None:
            all_reduce_providers = "legacy_torch_distributed"
        else:
            all_reduce_providers = (
                "attention="
                f"{self.runtime_config.parallel_collectives.attention.name},"
                "moe="
                f"{self.runtime_config.parallel_collectives.moe.name}"
            )
        logger.info(
            "Loaded MiniMax M2 rank {} provider={} prefill_provider={} "
            "all_reduce_providers={} "
            "attention TP {}/{} MoE TP "
            "{}/{} local experts [{}, {}) across {} layers; intentionally skipped "
            "{} remote expert weight/scale pairs.",
            self.parallel_context.world_rank,
            self.model.layers[0].block_sparse_moe.experts.provider.name,
            prefill_provider,
            all_reduce_providers,
            self.parallel_context.attn_tp_rank,
            self.parallel_context.attn_tp_size,
            self.parallel_context.moe_tp_rank,
            self.parallel_context.moe_tp_size,
            self.model.layers[0].block_sparse_moe.experts.local_expert_start,
            self.model.layers[0].block_sparse_moe.experts.local_expert_end,
            len(self.model.layers),
            len(self._intentionally_skipped_expert_weights),
        )

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def forward_idle_experts(self, hidden_states):
        for layer in self.model.layers:
            layer.block_sparse_moe(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
