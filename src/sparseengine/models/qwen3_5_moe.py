from __future__ import annotations

import re

import torch
import torch.nn.functional as F
from torch import nn

from sparseengine.distributed import get_parallel_context
from sparseengine.distributed.moe_communication import AllReduceMoeCommunication
from sparseengine.engine.recurrent_state_manager import (
    RecurrentStateSpec,
    RecurrentTensorSpec,
)
from sparseengine.layers.embed_head import ParallelLMHead
from sparseengine.models.qwen3_5 import (
    Qwen35DecoderLayer,
    Qwen35ForCausalLM,
    Qwen35MLP,
    Qwen35Model,
)
from sparseengine.models.attention_runtime import (
    bind_mha_full_attention_provider,
)
from sparseengine.models.gdn_runtime import bind_gated_delta_rule_op
from sparseengine.operators.full_attention import FullAttentionProvider
from sparseengine.operators.gated_delta_rule import PreparedGatedDeltaRuleOp
from sparseengine.models.qwen3_moe import Qwen3MoePackedExperts
from sparseengine.operators.gated_shared_add import gated_shared_add
from sparseengine.operators.moe import model_activation_dtype
from sparseengine.operators.moe_execution import MoeExecutionPlan
from sparseengine.operators.moe_router import (
    MoeRouterOpSpec,
    resolve_moe_router_provider,
)
from sparseengine.platforms import device_runtime
from sparseengine.utils.context import get_context
from sparseengine.utils.log import logger
from sparseengine.utils.weight_target import WeightTarget


_PACKED_EXPERT_SOURCE_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\."
    r"(gate_up_proj|down_proj)$"
)
_PACKED_EXPERT_TARGET_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\."
    r"(gate_up_proj|down_proj)\.packed_expert_weight$"
)
_FP8_EXPERT_SOURCE_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
_FP8_EXPERT_TARGET_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.expert_weight$"
)


class Qwen35MoeRouter(nn.Module):
    """Replicated router with the checkpoint's FP32 softmax semantics."""

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_experts = int(config.num_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.op_spec = MoeRouterOpSpec(
            num_experts=self.num_experts,
            top_k=self.top_k,
            activation_dtype=model_activation_dtype(config),
            norm_topk_prob=True,
            cuda_graph=bool(getattr(config, "decode_graph", False)),
        )
        self.provider = resolve_moe_router_provider(self.op_spec)
        self.weight = nn.Parameter(
            torch.empty(self.num_experts + 1, self.hidden_size)
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        target = param.data[-1:] if loaded_shard_id == "shared" else param.data[:-1]
        if loaded_shard_id not in {None, "shared"} or target.shape != loaded_weight.shape:
            raise ValueError(
                "Qwen3.6 fused router/shared gate weight mismatch: "
                f"shard={loaded_shard_id!r} expected={tuple(target.shape)} "
                f"got={tuple(loaded_weight.shape)}."
            )
        target.copy_(loaded_weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits, shared_gate_logits = F.linear(
            hidden_states,
            self.weight,
        ).split((self.num_experts, 1), dim=-1)
        topk_weights, topk_ids = self.provider.run(
            self.op_spec, router_logits
        )
        return topk_weights, topk_ids, shared_gate_logits


class Qwen35MoePackedExperts(Qwen3MoePackedExperts):
    """Adapt Qwen3.6 BF16/FP8 checkpoints to the shared MoE provider."""

    checkpoint_projection_map = {
        "gate_up_proj": "gate_up",
        "down_proj": "down",
        "gate_proj": "gate",
        "up_proj": "up",
    }

    def __init__(self, config) -> None:
        super().__init__(config)
        self._loaded_packed_projections: set[str] = set()

    def rank_local_weight_slice(
        self,
        source_shape: tuple[int, ...],
        *,
        loaded_shard_id: str | tuple[int, str],
        is_scale: bool = False,
    ) -> tuple[slice, ...] | None:
        if isinstance(loaded_shard_id, tuple):
            if not self.fp8_enabled:
                raise ValueError(
                    "Qwen3.6 BF16 checkpoints must use packed 3D expert weights."
                )
            return super().rank_local_weight_slice(
                source_shape,
                loaded_shard_id=loaded_shard_id,
                is_scale=is_scale,
            )
        if self.fp8_enabled:
            raise ValueError(
                "Qwen3.6 FP8 checkpoints must use per-expert projections and scales."
            )
        if is_scale:
            raise ValueError("Qwen3.6 MoE BF16 experts do not have weight scales.")
        expected = {
            "gate_up_proj": (
                self.num_experts,
                2 * self.global_intermediate_size,
                self.hidden_size,
            ),
            "down_proj": (
                self.num_experts,
                self.hidden_size,
                self.global_intermediate_size,
            ),
        }.get(str(loaded_shard_id))
        if expected is None:
            raise ValueError(
                f"Unsupported Qwen3.6 packed expert projection {loaded_shard_id!r}."
            )
        if tuple(source_shape) != expected:
            raise ValueError(
                "Qwen3.6 packed expert checkpoint shape mismatch: "
                f"projection={loaded_shard_id} expected={expected} "
                f"got={tuple(source_shape)}."
            )
        return (
            slice(self.local_expert_start, self.local_expert_end),
            slice(None),
            slice(None),
        )

    def load_packed_expert_weight(
        self,
        projection: str,
        loaded_weight: torch.Tensor,
    ) -> None:
        if self.fp8_enabled:
            raise ValueError(
                "Qwen3.6 FP8 checkpoints must use per-expert projections and scales."
            )
        projection = str(projection)
        if projection in self._loaded_packed_projections:
            raise ValueError(
                f"Duplicate Qwen3.6 packed expert projection {projection!r}."
            )
        if loaded_weight.dtype != torch.bfloat16:
            raise TypeError(
                "Qwen3.6 packed expert weights must be BF16, "
                f"got {loaded_weight.dtype}."
            )
        expected_local = {
            "gate_up_proj": (
                self.num_local_experts,
                2 * self.global_intermediate_size,
                self.hidden_size,
            ),
            "down_proj": (
                self.num_local_experts,
                self.hidden_size,
                self.global_intermediate_size,
            ),
        }.get(projection)
        if expected_local is None:
            raise ValueError(
                f"Unsupported Qwen3.6 packed expert projection {projection!r}."
            )
        if tuple(loaded_weight.shape) != expected_local:
            raise ValueError(
                "Qwen3.6 rank-local packed expert shape mismatch: "
                f"projection={projection} expected={expected_local} "
                f"got={tuple(loaded_weight.shape)}."
            )

        for local_expert_id in range(self.num_local_experts):
            global_expert_id = self.local_expert_start + local_expert_id
            if projection == "gate_up_proj":
                gate, up = loaded_weight[local_expert_id].split(
                    self.global_intermediate_size,
                    dim=0,
                )
                self.load_expert_weight(
                    global_expert_id, "gate_proj", gate, None
                )
                self.load_expert_weight(
                    global_expert_id, "up_proj", up, None
                )
            else:
                self.load_expert_weight(
                    global_expert_id,
                    "down_proj",
                    loaded_weight[local_expert_id],
                    None,
                )
        self._loaded_packed_projections.add(projection)

    def validate_loaded_weights(self) -> None:
        if not self.fp8_enabled:
            missing = {"gate_up_proj", "down_proj"} - self._loaded_packed_projections
            if missing:
                raise ValueError(
                    f"Missing Qwen3.6 packed expert projections: {sorted(missing)}."
                )
        expected = {
            (expert_id, projection)
            for expert_id in range(self.local_expert_start, self.local_expert_end)
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        missing = sorted(expected - self._loaded_expert_shards)
        if missing:
            raise ValueError(
                "Missing local Qwen3.6 expert weights: "
                f"local_range=[{self.local_expert_start}, "
                f"{self.local_expert_end}), missing={missing[:8]}."
            )


class Qwen35MoeSparseMoeBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.parallel_context = get_parallel_context()
        self.mlp_chunk_size = int(getattr(config, "mlp_chunk_size", 16384))
        if self.mlp_chunk_size <= 0:
            raise ValueError(
                f"mlp_chunk_size must be > 0, got {self.mlp_chunk_size}."
            )
        self.gate = Qwen35MoeRouter(config)
        self.experts = Qwen35MoePackedExperts(config)
        self.shared_expert = Qwen35MLP(
            config,
            intermediate_size=int(config.shared_expert_intermediate_size),
            reduce_results=False,
        )

        self.moe_communication = AllReduceMoeCommunication(
            self.parallel_context.world.all_reduce
        )
        self.moe_execution = self._create_moe_execution()

    def _create_moe_execution(self):
        return MoeExecutionPlan(
            routed=self._routed_chunk, shared=self.shared_expert,
            communication=self.moe_communication, chunk_size=self.mlp_chunk_size,
            finish=self._finish_branches,
            shared_modules=(self.shared_expert,),
        )

    def _routed_chunk(self, hidden_states):
        weights, ids, shared_gate_logits = self.gate(hidden_states)
        return self.experts(hidden_states, ids, weights), shared_gate_logits

    def _finish_branches(self, routed, shared_output):
        local_output, shared_gate_logits = routed
        if self.parallel_context.world.size > 1:
            # Keep the gate after reduction, including the original BF16 rounding.
            local_output, shared_output = self.moe_communication.combine(
                torch.stack((local_output, shared_output))
            )
        return gated_shared_add(local_output, shared_output, shared_gate_logits)

    def _forward_chunk(self, hidden_states):
        return self.moe_execution(hidden_states, is_prefill=get_context().is_prefill)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() != 2:
            raise ValueError(
                "Qwen35MoeSparseMoeBlock expects [tokens, hidden], "
                f"got {tuple(hidden_states.shape)}."
            )
        chunks = hidden_states.split(self.mlp_chunk_size, dim=0)
        outputs = []
        for chunk in chunks:
            outputs.append(self._forward_chunk(chunk))
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


class Qwen35MoeDecoderLayer(Qwen35DecoderLayer):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__(config, layer_idx, mlp_cls=Qwen35MoeSparseMoeBlock)


class Qwen35MoeModel(Qwen35Model):
    def __init__(self, config) -> None:
        setattr(config, "runtime_recurrent_state_dtype", torch.float32)
        super().__init__(config, Qwen35MoeDecoderLayer)


class Qwen35MoeForCausalLM(Qwen35ForCausalLM):
    ignored_weight_prefixes = ("model.visual.", "visual.", "mtp.")
    special_weight_loaders = {
        **Qwen35ForCausalLM.special_weight_loaders,
        ".packed_expert_weight": "load_packed_expert_weight",
        ".expert_weight": "load_expert_weight",
    }

    @staticmethod
    def build_runtime_kwargs(config, **runtime_kwargs) -> dict:
        # Qwen3.5/3.6 MoE checkpoints use FP32 live recurrent state. Resolve
        # the GDN provider against that contract before model construction.
        setattr(config, "runtime_recurrent_state_dtype", torch.float32)
        return Qwen35ForCausalLM.build_runtime_kwargs(config, **runtime_kwargs)

    def __init__(
        self,
        config,
        full_attention_provider: FullAttentionProvider | None = None,
        gated_delta_rule_op: PreparedGatedDeltaRuleOp | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.parallel_context = get_parallel_context()
        self.full_attention_provider = full_attention_provider
        self.gated_delta_rule_op = gated_delta_rule_op
        self.model = Qwen35MoeModel(config)
        if full_attention_provider is not None:
            bind_mha_full_attention_provider(self.model, full_attention_provider)
        if gated_delta_rule_op is not None:
            bind_gated_delta_rule_op(self.model, gated_delta_rule_op)
        self.lm_head = ParallelLMHead(
            int(config.vocab_size), int(config.hidden_size)
        )
        if bool(getattr(config, "tie_word_embeddings", False)):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        self.multimodal_encoder = None
        self._loaded_linear_special_weights: set[str] = set()
        self._intentionally_skipped_weights: set[str] = set()
        self._intentionally_skipped_expert_weights: set[str] = set()
        self._intentionally_skipped_expert_scales: set[str] = set()
        logger.info(
            "Loaded Qwen3.5 MoE prefill_provider={} decode_provider={} gdn_provider={}",
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
            "unbound" if gated_delta_rule_op is None else gated_delta_rule_op.name,
        )

    @staticmethod
    def recurrent_state_spec(config, attention_tp_size: int) -> RecurrentStateSpec:
        attention_tp_size = int(attention_tp_size)
        num_k_heads = int(config.linear_num_key_heads) // attention_tp_size
        num_v_heads = int(config.linear_num_value_heads) // attention_tp_size
        key_head_dim = int(config.linear_key_head_dim)
        value_head_dim = int(config.linear_value_head_dim)
        conv_dim = 2 * num_k_heads * key_head_dim + num_v_heads * value_head_dim
        return RecurrentStateSpec(
            name="qwen3_5_moe gated delta net",
            tensor_specs=(
                RecurrentTensorSpec(
                    "conv_state",
                    (conv_dim, int(config.linear_conv_kernel_dim) - 1),
                    config.dtype,
                ),
                RecurrentTensorSpec(
                    "recurrent_state",
                    (num_v_heads, key_head_dim, value_head_dim),
                    torch.float32,
                ),
            ),
        )

    @torch.inference_mode()
    def warmup_moe(self, num_tokens: int = 1) -> None:
        num_tokens = int(num_tokens)
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be > 0, got {num_tokens}.")
        block = self.model.layers[0].mlp
        experts = block.experts
        device = experts.w13_weight.device
        dtype = block.gate.weight.dtype
        hidden_states = torch.zeros(
            (num_tokens, experts.hidden_size), dtype=dtype, device=device
        )
        top_k = int(self.config.num_experts_per_tok)
        topk_ids = (
            torch.arange(num_tokens * top_k, dtype=torch.int64, device=device)
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
        block(hidden_states)
        device_runtime.synchronize()

    def map_weight_name(self, source_weight_name: str) -> str | None:
        match = _FP8_EXPERT_SOURCE_RE.match(source_weight_name)
        if match is not None:
            layer_idx, global_expert_id, projection = match.groups()
            global_expert_id = int(global_expert_id)
            experts = self.model.layers[int(layer_idx)].mlp.experts
            if not experts.fp8_enabled:
                raise ValueError(
                    "Qwen3.6 BF16 checkpoints must use packed 3D expert weights, "
                    f"got {source_weight_name!r}."
                )
            if not experts.is_local_expert(global_expert_id):
                return None
            return (
                f"model.layers.{layer_idx}.mlp.experts.{global_expert_id}."
                f"{projection}.expert_weight"
            )
        match = _PACKED_EXPERT_SOURCE_RE.match(source_weight_name)
        if match is not None:
            layer_idx, projection = match.groups()
            experts = self.model.layers[int(layer_idx)].mlp.experts
            if experts.fp8_enabled:
                raise ValueError(
                    "Qwen3.6 FP8 checkpoints must use per-expert projections and scales, "
                    f"got packed tensor {source_weight_name!r}."
                )
            return (
                f"model.layers.{layer_idx}.mlp.experts.{projection}."
                "packed_expert_weight"
            )
        return super().map_weight_name(source_weight_name)

    def resolve_special_weight(
        self,
        target_weight_name: str,
    ) -> WeightTarget | None:
        match = _FP8_EXPERT_TARGET_RE.match(target_weight_name)
        if match is not None:
            layer_idx, global_expert_id, projection = match.groups()
            return WeightTarget(
                self.model.layers[int(layer_idx)].mlp.experts,
                (int(global_expert_id), projection),
            )
        match = _PACKED_EXPERT_TARGET_RE.match(target_weight_name)
        if match is not None:
            layer_idx, projection = match.groups()
            return WeightTarget(
                self.model.layers[int(layer_idx)].mlp.experts,
                projection,
            )
        return super().resolve_special_weight(target_weight_name)

    def load_special_weight(
        self,
        target_weight_name: str,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None,
    ) -> int:
        match = _FP8_EXPERT_TARGET_RE.match(target_weight_name)
        if match is not None:
            layer_idx, global_expert_id, projection = match.groups()
            self.model.layers[int(layer_idx)].mlp.experts.load_expert_weight(
                int(global_expert_id),
                projection,
                loaded_weight,
                loaded_scale,
            )
            return 1
        match = _PACKED_EXPERT_TARGET_RE.match(target_weight_name)
        if match is not None:
            if loaded_scale is not None:
                raise ValueError(
                    "Qwen3.6 BF16 packed experts unexpectedly have weight scales."
                )
            layer_idx, projection = match.groups()
            self.model.layers[int(layer_idx)].mlp.experts.load_packed_expert_weight(
                projection,
                loaded_weight,
            )
            return 1
        loaded = super().load_special_weight(
            target_weight_name,
            loaded_weight,
            loaded_scale,
        )
        if loaded:
            self._loaded_linear_special_weights.add(target_weight_name)
        return loaded

    def record_skipped_weight(
        self,
        source_weight_name: str,
        loaded_weight_shape: tuple[int, ...] | None,
        loaded_weight_dtype: str | None,
        loaded_scale_shape: tuple[int, ...] | None,
        loaded_scale_dtype: str | None,
    ) -> None:
        match = _FP8_EXPERT_SOURCE_RE.match(source_weight_name)
        if match is not None:
            layer_idx, global_expert_id, projection = match.groups()
            experts = self.model.layers[int(layer_idx)].mlp.experts
            if not experts.fp8_enabled:
                raise ValueError(
                    f"Qwen3.6 BF16 loader unexpectedly skipped {source_weight_name!r}."
                )
            if experts.is_local_expert(int(global_expert_id)):
                raise ValueError(
                    f"Qwen3.6 FP8 loader skipped local expert {source_weight_name!r}."
                )
            expected_weight_shape = (
                (experts.hidden_size, experts.global_intermediate_size)
                if projection == "down_proj"
                else (experts.global_intermediate_size, experts.hidden_size)
            )
            expected_scale_shape = (
                (experts.hidden_size // 128, experts.global_intermediate_size // 128)
                if projection == "down_proj"
                else (
                    experts.global_intermediate_size // 128,
                    experts.hidden_size // 128,
                )
            )
            if loaded_weight_shape != expected_weight_shape:
                raise ValueError(
                    "Remote Qwen3.6 FP8 expert weight shape mismatch for "
                    f"{source_weight_name!r}: expected={expected_weight_shape}, "
                    f"got={loaded_weight_shape}."
                )
            if loaded_weight_dtype != "F8_E4M3":
                raise TypeError(
                    "Remote Qwen3.6 expert weight must be FP8 E4M3, got "
                    f"safetensors dtype {loaded_weight_dtype}."
                )
            if loaded_scale_shape != expected_scale_shape:
                raise ValueError(
                    "Remote Qwen3.6 FP8 expert scale shape mismatch for "
                    f"{source_weight_name!r}: expected={expected_scale_shape}, "
                    f"got={loaded_scale_shape}."
                )
            if loaded_scale_dtype != "BF16":
                raise TypeError(
                    "Remote Qwen3.6 expert scale must be BF16, got "
                    f"safetensors dtype {loaded_scale_dtype}."
                )
            self._intentionally_skipped_expert_weights.add(source_weight_name)
            self._intentionally_skipped_expert_scales.add(
                source_weight_name[: -len(".weight")] + ".weight_scale_inv"
            )
            return
        if not source_weight_name.startswith(self.ignored_weight_prefixes):
            raise ValueError(
                f"Qwen3.6 MoE loader unexpectedly skipped {source_weight_name!r}."
            )
        is_mtp = source_weight_name.startswith("mtp.")
        if not is_mtp and (
            loaded_scale_shape is not None or loaded_scale_dtype is not None
        ):
            raise ValueError(
                "Qwen3.6 MoE visual intentional skips must not consume "
                f"quantization scales: {source_weight_name!r}."
            )
        if is_mtp and loaded_scale_shape is not None:
            if loaded_weight_dtype != "F8_E4M3" or loaded_scale_dtype != "BF16":
                raise TypeError(
                    "Qwen3.6 skipped MTP quantized weights require FP8 E4M3 weights "
                    f"and BF16 scales: {source_weight_name!r}."
                )
            if loaded_weight_shape is None or len(loaded_weight_shape) != 2:
                raise ValueError(
                    "Qwen3.6 skipped MTP FP8 weights must be rank-2, "
                    f"got {source_weight_name!r} shape={loaded_weight_shape}."
                )
            expected_scale_shape = tuple(
                (int(dimension) + 127) // 128
                for dimension in loaded_weight_shape
            )
            if loaded_scale_shape != expected_scale_shape:
                raise ValueError(
                    "Qwen3.6 skipped MTP FP8 scale shape mismatch for "
                    f"{source_weight_name!r}: expected={expected_scale_shape}, "
                    f"got={loaded_scale_shape}."
                )
        elif is_mtp and loaded_weight_dtype == "F8_E4M3":
            raise ValueError(
                f"Qwen3.6 skipped MTP FP8 weight is missing its scale: {source_weight_name!r}."
            )
        self._intentionally_skipped_weights.add(source_weight_name)

    def validate_loaded_weights(self, loaded_parameter_names: set[str]) -> None:
        packed_parameters = {
            name
            for name, _ in self.named_parameters()
            if name.endswith(".mlp.experts.w13_weight")
            or name.endswith(".mlp.experts.w2_weight")
        }
        linear_special_parameters = {
            name
            for name, _ in self.named_parameters()
            if ".linear_attn.in_proj_" in name
            and name.endswith(".weight")
            and name.rsplit(".", 2)[-2] in {"in_proj_q", "in_proj_k", "in_proj_v"}
        }
        expected_dense = {
            name for name, _ in self.named_parameters()
        } - packed_parameters - linear_special_parameters
        missing_dense = sorted(expected_dense - loaded_parameter_names)
        if missing_dense:
            raise ValueError(
                f"Missing Qwen3.6 MoE replicated/sharded weights: {missing_dense[:8]}."
            )

        expected_linear_special = {
            f"model.layers.{layer_idx}.linear_attn.in_proj_qkv.weight"
            for layer_idx in self.config.runtime_layout.linear_attention_layer_indices
        }
        missing_linear = sorted(
            expected_linear_special - self._loaded_linear_special_weights
        )
        if missing_linear:
            raise ValueError(
                f"Missing Qwen3.6 packed GDN weights: {missing_linear[:8]}."
            )
        for layer in self.model.layers:
            layer.mlp.experts.validate_loaded_weights()

        first_experts = self.model.layers[0].mlp.experts
        if first_experts.fp8_enabled:
            expected_skipped_experts = {
                "model.language_model.layers."
                f"{layer_idx}.mlp.experts.{expert_id}.{projection}.weight"
                for layer_idx in range(int(self.config.num_hidden_layers))
                for expert_id in range(int(self.config.num_experts))
                if not self.model.layers[layer_idx].mlp.experts.is_local_expert(
                    expert_id
                )
                for projection in ("gate_proj", "up_proj", "down_proj")
            }
            expected_skipped_scales = {
                name[: -len(".weight")] + ".weight_scale_inv"
                for name in expected_skipped_experts
            }
            if self._intentionally_skipped_expert_weights != expected_skipped_experts:
                missing = sorted(
                    expected_skipped_experts
                    - self._intentionally_skipped_expert_weights
                )
                unexpected = sorted(
                    self._intentionally_skipped_expert_weights
                    - expected_skipped_experts
                )
                raise ValueError(
                    "Qwen3.6 FP8 remote expert skip audit failed: "
                    f"missing={missing[:4]}, unexpected={unexpected[:4]}."
                )
            if self._intentionally_skipped_expert_scales != expected_skipped_scales:
                missing = sorted(
                    expected_skipped_scales
                    - self._intentionally_skipped_expert_scales
                )
                unexpected = sorted(
                    self._intentionally_skipped_expert_scales
                    - expected_skipped_scales
                )
                raise ValueError(
                    "Qwen3.6 FP8 remote expert scale skip audit failed: "
                    f"missing={missing[:4]}, unexpected={unexpected[:4]}."
                )

        skip_groups = {
            "visual": any(
                name.startswith(("model.visual.", "visual."))
                for name in self._intentionally_skipped_weights
            ),
            "mtp": any(
                name.startswith("mtp.")
                for name in self._intentionally_skipped_weights
            ),
        }
        required_skip_groups = (
            {"mtp"} if self.multimodal_encoder is not None else {"visual", "mtp"}
        )
        missing_skip_groups = [
            name for name in required_skip_groups if not skip_groups[name]
        ]
        if missing_skip_groups:
            raise ValueError(
                "Qwen3.6 MoE checkpoint is missing expected intentional-skip "
                f"groups: {missing_skip_groups}."
            )
        logger.info(
            "Loaded Qwen3.6 MoE rank {} quantization={} expert_provider={} "
            "router_provider={} "
            "attention TP {}/{} "
            "MoE TP {}/{} EP {}/{} local experts [{}, {}) across {} layers; "
            "intentionally skipped {} visual/MTP tensors.",
            self.parallel_context.world_rank,
            "fp8" if first_experts.fp8_enabled else "bf16",
            first_experts.provider.name,
            self.model.layers[0].mlp.gate.provider.name,
            self.parallel_context.attn_tp_rank,
            self.parallel_context.attn_tp_size,
            self.parallel_context.moe_tp_rank,
            self.parallel_context.moe_tp_size,
            self.parallel_context.moe_ep_rank,
            self.parallel_context.moe_ep_size,
            first_experts.local_expert_start,
            first_experts.local_expert_end,
            len(self.model.layers),
            len(self._intentionally_skipped_weights),
        )
