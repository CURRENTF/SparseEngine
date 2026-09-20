from __future__ import annotations

import os

import torch
from torch import nn
import torch.nn.functional as F

from sparseengine.distributed import get_parallel_context
from sparseengine.layers.attention import Attention
from sparseengine.layers.layernorm import GemmaRMSNorm
from sparseengine.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    divide,
)
from sparseengine.models.attention_runtime import (
    bind_mha_full_attention_provider,
    build_mha_full_attention_provider,
)
from sparseengine.models.gdn_runtime import (
    bind_gated_delta_rule_op,
    build_gated_delta_rule_op,
)
from sparseengine.operators.full_attention import FullAttentionProvider
from sparseengine.operators.gated_delta_rule import PreparedGatedDeltaRuleOp
from sparseengine.operators.gate_up_swiglu import (
    GateUpSwiGLUOpSpec,
    resolve_gate_up_swiglu_provider,
)
from sparseengine.layers.rotary_embedding import apply_partial_rotary_emb, get_rope
from sparseengine.operators.qwen35_mrope import Qwen35MRotaryEmbedding
from sparseengine.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from sparseengine.utils.context import get_context
from sparseengine.utils.log import logger
from sparseengine.utils.weight_target import WeightTarget
from sparseengine.engine.recurrent_state_manager import RecurrentStateSpec, RecurrentTensorSpec


def _get_rope_theta(config) -> float:
    if hasattr(config, "rope_theta"):
        return config.rope_theta
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
        return rope_parameters["rope_theta"]
    return 10000


def _get_rope_scaling(config):
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        return None
    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        is_default_rope = rope_type in (None, "default")
        allowed_default_keys = {
            "rope_type",
            "type",
            "rope_theta",
            "mrope_interleaved",
            "mrope_section",
            "partial_rotary_factor",
        }
        if is_default_rope and set(rope_scaling).issubset(allowed_default_keys):
            return None
    raise NotImplementedError(f"Unsupported qwen3_5 rope_scaling={rope_scaling!r}.")


def _get_rotary_dim(config, head_dim: int) -> int:
    explicit_dim = getattr(config, "qk_rope_head_dim", None)
    if explicit_dim is not None:
        rotary_dim = int(explicit_dim)
    else:
        rope_parameters = getattr(config, "rope_parameters", None)
        partial_factor = getattr(config, "partial_rotary_factor", None)
        if partial_factor is None and isinstance(rope_parameters, dict):
            partial_factor = rope_parameters.get("partial_rotary_factor")
        if partial_factor is None:
            partial_factor = 1.0
        rotary_dim = int(int(head_dim) * float(partial_factor))
    if rotary_dim <= 0 or rotary_dim > int(head_dim) or rotary_dim % 2 != 0:
        raise ValueError(
            f"Invalid qwen3_5 rotary_dim={rotary_dim} for head_dim={head_dim}."
        )
    return rotary_dim


class Qwen35QKVGatedParallelLinear(ColumnParallelLinear):
    rank_local_weight_slice = None
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        bias: bool = False,
        quantization=None,
    ) -> None:
        tp_size = get_parallel_context().attn_tp_size
        self.head_size = int(head_size)
        self.total_num_heads = int(total_num_heads)
        self.total_num_kv_heads = int(total_num_kv_heads)
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        output_size = (2 * self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias, quantization=quantization)

    @property
    def _local_q_size(self) -> int:
        return self.num_heads * self.head_size

    @property
    def _local_kv_size(self) -> int:
        return self.num_kv_heads * self.head_size

    def _offset_and_size(self, loaded_shard_id: str) -> tuple[int, int]:
        q_size = self._local_q_size
        kv_size = self._local_kv_size
        if loaded_shard_id == "q":
            return 0, q_size
        if loaded_shard_id == "k":
            return q_size, kv_size
        if loaded_shard_id == "v":
            return q_size + kv_size, kv_size
        if loaded_shard_id == "gate":
            return q_size + 2 * kv_size, q_size
        raise ValueError(f"Unsupported qwen3_5 qkv gate shard id {loaded_shard_id!r}.")

    def _split_q_gate_weight(
        self,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        q_rows = self.total_num_heads * self.head_size
        if int(loaded_weight.shape[0]) == q_rows:
            return loaded_weight, loaded_scale, None, None
        if int(loaded_weight.shape[0]) != 2 * q_rows:
            raise ValueError(
                f"qwen3_5 q_proj rows mismatch: expected {q_rows} or {2 * q_rows}, got {loaded_weight.shape[0]}."
            )
        hidden = int(loaded_weight.shape[-1])
        weight = loaded_weight.view(self.total_num_heads * 2, self.head_size, hidden)
        q = weight[0::2].reshape(q_rows, hidden)
        gate = weight[1::2].reshape(q_rows, hidden)
        if loaded_scale is None:
            return q, None, gate, None
        if self.head_size % 128 != 0:
            raise ValueError(
                "qwen3_5 q_proj FP8 q/gate split requires head_size to be 128-aligned, "
                f"got {self.head_size}."
            )
        scale_cols = int(loaded_scale.shape[-1])
        blocks_per_head = self.head_size // 128
        expected_scale_rows = self.total_num_heads * 2 * blocks_per_head
        if int(loaded_scale.shape[0]) != expected_scale_rows:
            raise ValueError(
                f"qwen3_5 q_proj scale rows mismatch: expected={expected_scale_rows}, got={loaded_scale.shape[0]}."
            )
        scale = loaded_scale.view(self.total_num_heads * 2, blocks_per_head, scale_cols)
        q_scale = scale[0::2].reshape(-1, scale_cols)
        gate_scale = scale[1::2].reshape(-1, scale_cols)
        return q, q_scale, gate, gate_scale

    def _load_one_shard(
        self,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None,
        loaded_shard_id: str,
    ) -> None:
        shard_offset, shard_size = self._offset_and_size(loaded_shard_id)
        weight_target = self.weight.data.narrow(self.tp_dim, shard_offset, shard_size)
        weight_shard = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        if loaded_scale is None:
            weight_target.copy_(weight_shard)
            return
        if not self.quantized:
            raise ValueError("qwen3_5 QKVGated received FP8 scale but module is not quantized.")
        if shard_offset % 128 != 0 or shard_size % 128 != 0:
            raise ValueError(
                "qwen3_5 QKVGated FP8 scale sharding requires each local shard to be 128-aligned, "
                f"got offset={shard_offset}, size={shard_size}."
            )
        scale_shard_size = self._require_scale_shardable(
            "Qwen35QKVGatedParallelLinear",
            loaded_scale.size(0),
            self.tp_size,
        )
        scale_shard = loaded_scale.narrow(0, self.tp_rank * scale_shard_size, scale_shard_size)
        scale_target = self.weight_scale_inv.narrow(0, shard_offset // 128, shard_size // 128)
        self._copy_quantized_weight_and_scale(
            weight_shard,
            scale_shard,
            weight_target=weight_target,
            scale_target=scale_target,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        if loaded_shard_id == "q":
            q, _, gate, _ = self._split_q_gate_weight(loaded_weight, None)
            self._load_one_shard(q, None, "q")
            if gate is not None:
                self._load_one_shard(gate, None, "gate")
            return
        self._load_one_shard(loaded_weight, None, loaded_shard_id)

    def load_quantized_weight(
        self,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor,
        loaded_shard_id: str,
    ) -> None:
        self._ensure_quantized_loader()
        if loaded_shard_id == "q":
            q, q_scale, gate, gate_scale = self._split_q_gate_weight(loaded_weight, loaded_scale)
            self._load_one_shard(q, q_scale, "q")
            if gate is not None:
                self._load_one_shard(gate, gate_scale, "gate")
            return
        self._load_one_shard(loaded_weight, loaded_scale, loaded_shard_id)


class Qwen35FullAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        tp_size = get_parallel_context().attn_tp_size
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_heads % tp_size != 0 or self.total_num_kv_heads % tp_size != 0:
            raise ValueError("qwen3_5 attention heads must be divisible by tensor_parallel_size.")
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        self.rotary_dim = _get_rotary_dim(config, self.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.proj_chunk_size = int(getattr(config, "mlp_chunk_size", 16384))
        quantization = getattr(config, "quantization_config", None)

        self.qkv_gate_proj = Qwen35QKVGatedParallelLinear(
            int(config.hidden_size),
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bool(getattr(config, "attention_bias", False)),
            quantization=quantization,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            int(config.hidden_size),
            bias=False,
            quantization=quantization,
        )
        rope_parameters = getattr(config, "rope_parameters", None)
        mrope_sections = (
            rope_parameters.get("mrope_section")
            if isinstance(rope_parameters, dict)
            else None
        )
        self.rotary_emb = (
            Qwen35MRotaryEmbedding(
                self.head_dim,
                self.rotary_dim,
                int(config.max_position_embeddings),
                _get_rope_theta(config),
                mrope_sections,
            )
            if mrope_sections
            else get_rope(
                self.rotary_dim,
                rotary_dim=self.rotary_dim,
                max_position=int(config.max_position_embeddings),
                base=_get_rope_theta(config),
                rope_scaling=_get_rope_scaling(config),
            )
        )
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)
        self.q_norm = Qwen35RMSNorm(self.head_dim, eps=float(getattr(config, "rms_norm_eps", 1.0e-6)))
        self.k_norm = Qwen35RMSNorm(self.head_dim, eps=float(getattr(config, "rms_norm_eps", 1.0e-6)))

    def _o_proj_chunked(
        self, x: torch.Tensor, chunk_buffer: torch.Tensor
    ) -> torch.Tensor:
        chunk_size = int(self.proj_chunk_size)
        if int(x.shape[0]) <= chunk_size:
            return self.o_proj(x)
        for start in range(0, int(x.shape[0]), chunk_size):
            end = min(start + chunk_size, int(x.shape[0]))
            chunk_buffer[start:end].copy_(self.o_proj(x[start:end]))
        return chunk_buffer

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv_gate = self.qkv_gate_proj(hidden_states)
        qkv, gate = qkv_gate.split([self.q_size + self.kv_size + self.kv_size, self.q_size], dim=-1)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        layer_idx = context.now_layer_idx
        cache_manager = context.cache_manager
        pre_rope_k = k
        cache_manager.save_raw_kv_if_needed(layer_idx, pre_rope_k, v)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = (
            self.rotary_emb(positions, q, k)
            if isinstance(self.rotary_emb, Qwen35MRotaryEmbedding)
            else apply_partial_rotary_emb(
                self.rotary_emb, positions, q, k, self.rotary_dim
            )
        )
        cache_manager.save_rope_kv_if_needed(layer_idx, k, v)
        o = self.attn(q, k, v)
        o = o.flatten(1, -1) * torch.sigmoid(gate)
        return self._o_proj_chunked(o, hidden_states)


class Qwen35GatedRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.gated_delta_rule_op: PreparedGatedDeltaRuleOp | None = None

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if x.shape != gate.shape:
            raise RuntimeError(f"qwen3_5 gated RMSNorm shape mismatch: x={tuple(x.shape)} gate={tuple(gate.shape)}.")
        if not x.is_cuda:
            raise RuntimeError("qwen3_5 linear attention requires CUDA LightLLM kernels; CPU fallback is not supported.")
        if self.gated_delta_rule_op is None:
            raise RuntimeError(
                "qwen3_5 gated RMSNorm requires a prepared GDN provider."
            )
        return self.gated_delta_rule_op.run_gated_rmsnorm(
            x=x,
            weight=self.weight,
            eps=self.eps,
            gate=gate,
        )


class Qwen35RMSNorm(GemmaRMSNorm):
    """Qwen3.5 RMSNorm using its Hugging Face offset-weight semantics."""


class Qwen35LinearConv1D(nn.Module):
    def __init__(self, conv_dim: int, kernel_size: int, qk_dim: int, v_dim: int) -> None:
        super().__init__()
        self.conv_dim = int(conv_dim)
        self.kernel_size = int(kernel_size)
        self.qk_dim = int(qk_dim)
        self.v_dim = int(v_dim)
        parallel_context = get_parallel_context()
        self.tp_rank = parallel_context.attn_tp_rank
        self.tp_size = parallel_context.attn_tp_size
        self.weight = nn.Parameter(torch.empty(self.conv_dim, self.kernel_size), requires_grad=False)
        self.register_parameter("bias", None)
        self.weight.weight_loader = self.weight_loader

    def _shard_qkv_conv_rows(self, loaded: torch.Tensor) -> torch.Tensor:
        if loaded.dim() == 3:
            if loaded.shape[1] != 1:
                raise ValueError(f"qwen3_5 conv1d weight expects middle dim 1, got {tuple(loaded.shape)}.")
            loaded = loaded.squeeze(1)
        expected_rows = self.qk_dim * 2 + self.v_dim
        if loaded.shape[0] != expected_rows:
            raise ValueError(
                f"qwen3_5 conv1d rows mismatch: expected={expected_rows}, got={loaded.shape[0]}."
            )
        q, k, v = torch.split(loaded, [self.qk_dim, self.qk_dim, self.v_dim], dim=0)
        q = q.chunk(self.tp_size, dim=0)[self.tp_rank]
        k = k.chunk(self.tp_size, dim=0)[self.tp_rank]
        v = v.chunk(self.tp_size, dim=0)[self.tp_rank]
        sharded = torch.cat([q, k, v], dim=0)
        if tuple(sharded.shape) != tuple(self.weight.shape if sharded.dim() == 2 else self.bias.shape):
            target_shape = self.weight.shape if sharded.dim() == 2 else self.bias.shape
            raise ValueError(f"qwen3_5 conv1d shard shape mismatch: expected={tuple(target_shape)}, got={tuple(sharded.shape)}.")
        return sharded

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(self._shard_qkv_conv_rows(loaded_weight))

class Qwen35LinearAttention(nn.Module):
    is_gated_delta_rule_layer = True

    def __init__(self, config) -> None:
        super().__init__()
        tp_size = get_parallel_context().attn_tp_size
        self.total_num_k_heads = int(getattr(config, "linear_num_key_heads"))
        self.total_num_v_heads = int(getattr(config, "linear_num_value_heads"))
        self.head_k_dim = int(getattr(config, "linear_key_head_dim"))
        self.head_v_dim = int(getattr(config, "linear_value_head_dim"))
        self.conv_kernel_dim = int(getattr(config, "linear_conv_kernel_dim", 4))
        if self.total_num_k_heads % tp_size != 0 or self.total_num_v_heads % tp_size != 0:
            raise ValueError("qwen3_5 linear attention heads must be divisible by tensor_parallel_size.")
        if self.total_num_v_heads % self.total_num_k_heads != 0:
            raise ValueError("qwen3_5 linear attention requires linear_num_value_heads % linear_num_key_heads == 0.")
        self.num_k_heads = self.total_num_k_heads // tp_size
        self.num_v_heads = self.total_num_v_heads // tp_size
        self.key_dim = self.total_num_k_heads * self.head_k_dim
        self.value_dim = self.total_num_v_heads * self.head_v_dim
        self.tp_key_dim = self.num_k_heads * self.head_k_dim
        self.tp_value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.tp_key_dim * 2 + self.tp_value_dim
        self.activation = getattr(config, "hidden_act", "silu")
        if self.activation not in ("silu", "swish"):
            raise NotImplementedError(f"qwen3_5 linear attention supports silu/swish activation, got {self.activation!r}.")
        self.proj_chunk_size = int(getattr(config, "mlp_chunk_size", 16384))
        self.recurrent_state_dtype = getattr(
            config,
            "runtime_recurrent_state_dtype",
            config.dtype,
        )
        if not isinstance(self.recurrent_state_dtype, torch.dtype):
            raise TypeError(
                "qwen3_5 runtime_recurrent_state_dtype must be a torch.dtype, "
                f"got {self.recurrent_state_dtype!r}."
            )
        quantization = getattr(config, "quantization_config", None)

        hidden_size = int(config.hidden_size)
        self.in_proj_qkvz = MergedColumnParallelLinear(
            hidden_size,
            [self.key_dim, self.key_dim, self.value_dim, self.value_dim],
            bias=False,
            quantization=quantization,
        )
        self.in_proj_ba = MergedColumnParallelLinear(
            hidden_size,
            [self.total_num_v_heads, self.total_num_v_heads],
            bias=False,
            quantization=None,
        )
        self.conv1d = Qwen35LinearConv1D(
            self.conv_dim,
            self.conv_kernel_dim,
            self.key_dim,
            self.value_dim,
        )
        self.out_proj = RowParallelLinear(
            self.value_dim,
            int(config.hidden_size),
            bias=False,
            quantization=quantization,
        )
        self.norm = Qwen35GatedRMSNorm(self.head_v_dim, eps=float(getattr(config, "rms_norm_eps", 1.0e-6)))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32), requires_grad=False)
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32), requires_grad=False)
        self.A_log.weight_loader = self._tp_vector_weight_loader
        self.dt_bias.weight_loader = self._tp_vector_weight_loader
        self.gated_delta_rule_op: PreparedGatedDeltaRuleOp | None = None

    def bind_gated_delta_rule_op(
        self,
        gated_delta_rule_op: PreparedGatedDeltaRuleOp,
    ) -> None:
        self.gated_delta_rule_op = gated_delta_rule_op
        self.norm.gated_delta_rule_op = gated_delta_rule_op

    @staticmethod
    def _split_row_block_scale(loaded_scale: torch.Tensor | None, row_sizes: list[int]) -> list[torch.Tensor | None]:
        if loaded_scale is None:
            return [None for _ in row_sizes]
        scale_parts = []
        offset = 0
        for size in row_sizes:
            size = int(size)
            source_rows = []
            for target_block_start in range(0, size, 128):
                source_start = offset + target_block_start
                source_end = offset + min(target_block_start + 128, size) - 1
                source_scale_row = source_start // 128
                if source_scale_row != source_end // 128:
                    raise ValueError(
                        "qwen3_5 packed FP8 scale split cannot represent a target 128-row block "
                        f"that crosses source scale blocks: offset={offset}, size={size}."
                    )
                source_rows.append(source_scale_row)
            scale_parts.append(loaded_scale[source_rows, :].contiguous())
            offset += int(size)
        expected_rows = (offset + 127) // 128
        if expected_rows != int(loaded_scale.shape[0]):
            raise ValueError(
                f"qwen3_5 packed FP8 scale rows mismatch: expected={expected_rows}, got={loaded_scale.shape[0]}."
            )
        return scale_parts

    def _load_projection_weight(
        self,
        module: MergedColumnParallelLinear,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None,
        shard_id: int,
    ) -> None:
        if loaded_scale is not None:
            module.load_quantized_weight(
                loaded_weight,
                loaded_scale,
                loaded_shard_id=shard_id,
            )
            return
        if bool(getattr(module, "quantized", False)):
            raise ValueError(f"Missing FP8 weight_scale_inv for quantized qwen3_5 projection {module}.")
        module.weight_loader(module.weight, loaded_weight, shard_id)

    def load_packed_in_proj_qkv(
        self,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None = None,
    ) -> int:
        row_sizes = [self.key_dim, self.key_dim, self.value_dim]
        if int(loaded_weight.shape[0]) != sum(row_sizes):
            raise ValueError(
                f"qwen3_5 in_proj_qkv rows mismatch: expected={sum(row_sizes)}, got={loaded_weight.shape[0]}."
            )
        q, k, v = torch.split(loaded_weight, row_sizes, dim=0)
        q_scale, k_scale, v_scale = self._split_row_block_scale(loaded_scale, row_sizes)
        for shard_id, (weight, scale) in enumerate(
            ((q, q_scale), (k, k_scale), (v, v_scale))
        ):
            self._load_projection_weight(
                self.in_proj_qkvz,
                weight,
                scale,
                shard_id,
            )
        return 3

    def _split_interleaved_qkvz(
        self,
        loaded_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_v_per_k = self.total_num_v_heads // self.total_num_k_heads
        v_block = num_v_per_k * self.head_v_dim
        group_size = self.head_k_dim + self.head_k_dim + v_block + v_block
        expected_rows = self.total_num_k_heads * group_size
        if int(loaded_weight.shape[0]) != expected_rows:
            raise ValueError(
                f"qwen3_5 in_proj_qkvz rows mismatch: expected={expected_rows}, got={loaded_weight.shape[0]}."
            )
        hidden = int(loaded_weight.shape[-1])
        weight = loaded_weight.view(self.total_num_k_heads, group_size, hidden)
        q = weight[:, : self.head_k_dim, :].reshape(-1, hidden)
        k = weight[:, self.head_k_dim : 2 * self.head_k_dim, :].reshape(-1, hidden)
        v = weight[:, 2 * self.head_k_dim : 2 * self.head_k_dim + v_block, :].reshape(-1, hidden)
        z = weight[:, 2 * self.head_k_dim + v_block :, :].reshape(-1, hidden)
        return q, k, v, z

    def _split_interleaved_qkvz_scale(
        self,
        loaded_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if loaded_scale is None:
            return None, None, None, None
        if self.head_k_dim % 128 != 0 or self.head_v_dim % 128 != 0:
            raise ValueError(
                "qwen3_5 in_proj_qkvz FP8 scale split requires linear head dims to be 128-aligned, "
                f"got key={self.head_k_dim}, value={self.head_v_dim}."
            )
        num_v_per_k = self.total_num_v_heads // self.total_num_k_heads
        q_blocks = self.head_k_dim // 128
        k_blocks = self.head_k_dim // 128
        v_blocks = (num_v_per_k * self.head_v_dim) // 128
        group_blocks = q_blocks + k_blocks + v_blocks + v_blocks
        expected_rows = self.total_num_k_heads * group_blocks
        if int(loaded_scale.shape[0]) != expected_rows:
            raise ValueError(
                f"qwen3_5 in_proj_qkvz scale rows mismatch: expected={expected_rows}, got={loaded_scale.shape[0]}."
            )
        scale_cols = int(loaded_scale.shape[-1])
        scale = loaded_scale.view(self.total_num_k_heads, group_blocks, scale_cols)
        q = scale[:, :q_blocks, :].reshape(-1, scale_cols)
        k = scale[:, q_blocks : q_blocks + k_blocks, :].reshape(-1, scale_cols)
        v = scale[:, q_blocks + k_blocks : q_blocks + k_blocks + v_blocks, :].reshape(-1, scale_cols)
        z = scale[:, q_blocks + k_blocks + v_blocks :, :].reshape(-1, scale_cols)
        return q, k, v, z

    def load_packed_in_proj_qkvz(
        self,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None = None,
    ) -> int:
        q, k, v, z = self._split_interleaved_qkvz(loaded_weight)
        q_scale, k_scale, v_scale, z_scale = self._split_interleaved_qkvz_scale(loaded_scale)
        for shard_id, (weight, scale) in enumerate(
            ((q, q_scale), (k, k_scale), (v, v_scale), (z, z_scale))
        ):
            self._load_projection_weight(
                self.in_proj_qkvz,
                weight,
                scale,
                shard_id,
            )
        return 4

    def load_packed_in_proj_ba(
        self,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None = None,
    ) -> int:
        num_v_per_k = self.total_num_v_heads // self.total_num_k_heads
        group_size = 2 * num_v_per_k
        expected_rows = self.total_num_k_heads * group_size
        if int(loaded_weight.shape[0]) != expected_rows:
            raise ValueError(
                f"qwen3_5 in_proj_ba rows mismatch: expected={expected_rows}, got={loaded_weight.shape[0]}."
            )
        hidden = int(loaded_weight.shape[-1])
        weight = loaded_weight.view(self.total_num_k_heads, group_size, hidden)
        b = weight[:, :num_v_per_k, :].reshape(-1, hidden)
        a = weight[:, num_v_per_k:, :].reshape(-1, hidden)
        b_scale, a_scale = self._split_row_block_scale(loaded_scale, [self.total_num_v_heads, self.total_num_v_heads])
        self._load_projection_weight(self.in_proj_ba, b, b_scale, 0)
        self._load_projection_weight(self.in_proj_ba, a, a_scale, 1)
        return 2

    def _tp_vector_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        expected = self.total_num_v_heads
        if loaded_weight.numel() != expected:
            raise ValueError(
                f"qwen3_5 linear vector parameter size mismatch: expected={expected}, got={loaded_weight.numel()}."
            )
        parallel_context = get_parallel_context()
        shard_size = expected // parallel_context.attn_tp_size
        start = parallel_context.attn_tp_rank * shard_size
        param.data.copy_(loaded_weight.reshape(-1).narrow(0, start, shard_size).to(dtype=param.dtype))

    def _project_qkvzba(self, hidden_states: torch.Tensor):
        mixed_qkv, z = self.in_proj_qkvz(hidden_states).split(
            [2 * self.tp_key_dim + self.tp_value_dim, self.tp_value_dim],
            dim=-1,
        )
        b, a = self.in_proj_ba(hidden_states).chunk(2, dim=-1)
        z = z.view(-1, self.num_v_heads, self.head_v_dim)
        return mixed_qkv, z, b, a

    def _empty_conv_state(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(self.conv_dim, self.conv_kernel_dim - 1, dtype=dtype, device=device)

    def _empty_recurrent_state(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(self.num_v_heads, self.head_k_dim, self.head_v_dim, dtype=dtype, device=device)

    def _load_batch_states(
        self,
        context,
        recurrent_state_manager,
        *,
        require_existing: bool,
        activation_dtype: torch.dtype,
        device: torch.device,
    ):
        seqs = context.seqs
        if seqs is None:
            raise RuntimeError("qwen3_5 linear attention requires context.seqs.")
        conv_dtype = activation_dtype
        recurrent_dtype = self.recurrent_state_dtype
        conv_states = []
        recurrent_states = []
        has_initial = []
        missing = []
        layer_idx = int(context.now_layer_idx)
        for row, seq in enumerate(seqs):
            state = recurrent_state_manager.get_layer_state(int(seq.seq_id), layer_idx)
            has_state = state is not None and "conv_state" in state and "recurrent_state" in state
            if require_existing and not has_state:
                missing.append((row, int(seq.seq_id)))
            if has_state:
                conv_states.append(state["conv_state"].to(device=device, dtype=conv_dtype))
                recurrent_states.append(state["recurrent_state"].to(device=device, dtype=recurrent_dtype))
                has_initial.append(True)
            else:
                conv_states.append(self._empty_conv_state(device, conv_dtype))
                recurrent_states.append(self._empty_recurrent_state(device, recurrent_dtype))
                has_initial.append(False)
        if missing:
            raise RuntimeError(
                "qwen3_5 decode linear attention requires recurrent state for every sequence; "
                f"missing={missing[:5]}."
            )
        return (
            torch.stack(conv_states, dim=0).contiguous(),
            torch.stack(recurrent_states, dim=0).contiguous(),
            torch.tensor(has_initial, dtype=torch.bool, device=device),
        )

    def _store_batch_states(self, context, recurrent_state_manager, conv_states: torch.Tensor, recurrent_states: torch.Tensor):
        layer_idx = int(context.now_layer_idx)
        for row, seq in enumerate(context.seqs):
            recurrent_state_manager.set_layer_state(
                int(seq.seq_id),
                layer_idx,
                {
                    "conv_state": conv_states[row],
                    "recurrent_state": recurrent_states[row].to(
                        dtype=self.recurrent_state_dtype
                    ),
                },
            )

    @staticmethod
    def _pad_decode_states_for_static_batch(
        conv_states: torch.Tensor,
        recurrent_states: torch.Tensor,
        *,
        token_batch: int,
        real_batch: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_batch < real_batch:
            raise RuntimeError(
                "qwen3_5 decode linear attention received fewer tokens than real sequences: "
                f"tokens={token_batch} seqs={real_batch}."
            )
        if token_batch == real_batch:
            return conv_states, recurrent_states
        if real_batch <= 0:
            raise RuntimeError("qwen3_5 decode linear attention static batch requires a real sequence.")
        pad_rows = int(token_batch) - int(real_batch)
        conv_pad = conv_states[:1].repeat((pad_rows,) + (1,) * (conv_states.dim() - 1))
        recurrent_pad = recurrent_states[:1].repeat((pad_rows,) + (1,) * (recurrent_states.dim() - 1))
        return (
            torch.cat([conv_states, conv_pad], dim=0).contiguous(),
            torch.cat([recurrent_states, recurrent_pad], dim=0).contiguous(),
        )

    def _prefill_gdn(self, mixed_qkv: torch.Tensor, z: torch.Tensor, b: torch.Tensor, a: torch.Tensor, context, recurrent_state_manager):
        if self.gated_delta_rule_op is None:
            raise RuntimeError(
                "qwen3_5 linear attention requires a prepared GDN provider."
            )
        if context.cu_seqlens_q is None:
            raise RuntimeError("qwen3_5 prefill linear attention requires cu_seqlens_q.")
        conv_states, recurrent_states, has_initial = self._load_batch_states(
            context,
            recurrent_state_manager,
            require_existing=False,
            activation_dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
        )
        g, beta = self.gated_delta_rule_op.run_gating(
            A_log=self.A_log,
            a=a,
            b=b,
            dt_bias=self.dt_bias,
        )
        mixed_qkv = self.gated_delta_rule_op.run_prefill_conv(
            mixed_qkv=mixed_qkv,
            weight=self.conv1d.weight,
            bias=self.conv1d.bias,
            query_start_loc=context.cu_seqlens_q,
            cache_indices=torch.arange(len(context.seqs), dtype=torch.int32, device=mixed_qkv.device),
            has_initial_state=has_initial,
            conv_states=conv_states,
            activation=self.activation,
        )
        q, k, v = torch.split(mixed_qkv, [self.tp_key_dim, self.tp_key_dim, self.tp_value_dim], dim=-1)
        q = q.view(1, -1, self.num_k_heads, self.head_k_dim)
        k = k.view(1, -1, self.num_k_heads, self.head_k_dim)
        v = v.view(1, -1, self.num_v_heads, self.head_v_dim)
        core_attn_out, last_recurrent_state = self.gated_delta_rule_op.run_prefill(
            q=q,
            k=k,
            v=v,
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=recurrent_states,
            cu_seqlens=context.cu_seqlens_q,
        )
        self._store_batch_states(context, recurrent_state_manager, conv_states, last_recurrent_state)
        return core_attn_out, z

    def _decode_gdn(self, mixed_qkv: torch.Tensor, z: torch.Tensor, b: torch.Tensor, a: torch.Tensor, context, recurrent_state_manager):
        if self.gated_delta_rule_op is None:
            raise RuntimeError(
                "qwen3_5 linear attention requires a prepared GDN provider."
            )
        token_batch = int(mixed_qkv.shape[0])
        state_buffers, state_indices = recurrent_state_manager.get_decode_layer_state(
            context.seqs,
            layer_idx=int(context.now_layer_idx),
            token_batch=token_batch,
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
        )
        conv_states = state_buffers["conv_state"]
        recurrent_states = state_buffers["recurrent_state"]
        q, k, v, z, a, b = self.gated_delta_rule_op.prepare_decode_inputs(
            mixed_qkv=mixed_qkv,
            z_raw=z,
            a_raw=a,
            b_raw=b,
            conv_state=conv_states,
            conv_weight=self.conv1d.weight,
            conv_bias=self.conv1d.bias,
            conv_state_indices=state_indices,
            activation=self.activation,
            conv_size=self.conv_kernel_dim,
            num_k_heads=self.num_k_heads,
            head_k_dim=self.head_k_dim,
            num_v_heads=self.num_v_heads,
            head_v_dim=self.head_v_dim,
        )
        core_attn_out = self.gated_delta_rule_op.run_decode(
            q=q,
            k=k,
            v=v,
            initial_state=recurrent_states,
            state_indices=state_indices,
            A_log=self.A_log,
            a=a,
            dt_bias=self.dt_bias,
            b=b,
        )
        return core_attn_out, z

    def _out_proj_chunked(
        self, x: torch.Tensor, chunk_buffer: torch.Tensor
    ) -> torch.Tensor:
        chunk_size = int(self.proj_chunk_size)
        if int(x.shape[0]) <= chunk_size:
            return self.out_proj(x)
        for start in range(0, int(x.shape[0]), chunk_size):
            end = min(start + chunk_size, int(x.shape[0]))
            chunk_buffer[start:end].copy_(self.out_proj(x[start:end]))
        return chunk_buffer

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        del positions
        context = get_context()
        recurrent_state_manager = context.recurrent_state_manager
        if recurrent_state_manager is None:
            raise RuntimeError("qwen3_5 linear attention requires RecurrentStateManager in runtime context.")

        mixed_qkv, z, b, a = self._project_qkvzba(hidden_states)
        if context.is_prefill:
            core_attn_out, z = self._prefill_gdn(mixed_qkv, z, b, a, context, recurrent_state_manager)
        else:
            if mixed_qkv.shape[0] < len(context.seqs):
                raise RuntimeError(
                    "qwen3_5 decode linear attention batch mismatch: "
                    f"tokens={mixed_qkv.shape[0]} seqs={len(context.seqs)}."
                )
            core_attn_out, z = self._decode_gdn(mixed_qkv, z, b, a, context, recurrent_state_manager)

        num_tokens = int(z.shape[0])
        core_attn_out = core_attn_out.view(-1, self.head_v_dim)
        z = z.contiguous().view(-1, self.head_v_dim)
        norm_out = self.norm(core_attn_out, z).view(num_tokens, self.tp_value_dim)
        return self._out_proj_chunked(norm_out, hidden_states)


class Qwen35MLP(nn.Module):
    def __init__(
        self,
        config,
        *,
        intermediate_size: int | None = None,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        intermediate_size = int(
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        quantization = getattr(config, "quantization_config", None)
        self.gate_up_proj = MergedColumnParallelLinear(
            int(config.hidden_size),
            [intermediate_size] * 2,
            bias=False,
            quantization=quantization,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            int(config.hidden_size),
            bias=False,
            quantization=quantization,
            reduce_results=reduce_results,
        )
        if getattr(config, "hidden_act", "silu") != "silu":
            raise NotImplementedError(f"qwen3_5 supports hidden_act='silu', got {config.hidden_act!r}.")
        activation_dtype = config.dtype
        self.gate_up_swiglu_spec = GateUpSwiGLUOpSpec(
            hidden_size=int(config.hidden_size),
            intermediate_size=intermediate_size,
            tp_size=self.gate_up_proj.tp_size,
            activation_dtype=activation_dtype,
            weight_dtype=(
                torch.float8_e4m3fn
                if self.gate_up_proj.quantized
                else activation_dtype
            ),
            cuda_graph=bool(getattr(config, "decode_graph", False)),
        )
        self.gate_up_swiglu_provider = resolve_gate_up_swiglu_provider(
            self.gate_up_swiglu_spec
        )
        self.mlp_chunk_size = int(getattr(config, "mlp_chunk_size", 16384))

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.gate_up_swiglu_provider.run(
                self.gate_up_swiglu_spec, x, self.gate_up_proj
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if int(x.shape[0]) <= self.mlp_chunk_size:
            return self._forward_chunk(x)
        out = torch.empty_like(x)
        for start in range(0, int(x.shape[0]), self.mlp_chunk_size):
            end = min(start + self.mlp_chunk_size, int(x.shape[0]))
            out[start:end].copy_(self._forward_chunk(x[start:end]))
        return out


class Qwen35DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, mlp_cls=Qwen35MLP) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        runtime_layout = getattr(config, "runtime_layout", None)
        if runtime_layout is None:
            raise ValueError("qwen3_5 config requires runtime_layout.")
        if runtime_layout.is_full_attention(layer_idx):
            self.self_attn = Qwen35FullAttention(config)
            self.attention_type = "full_attention"
        else:
            self.linear_attn = Qwen35LinearAttention(config)
            self.attention_type = "linear_attention"
        self.mlp = mlp_cls(config)
        self.input_layernorm = Qwen35RMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps))
        self.post_attention_layernorm = Qwen35RMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps))

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
        if self.attention_type == "linear_attention":
            hidden_states = self.linear_attn(positions, hidden_states)
        else:
            hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen35Model(nn.Module):
    def __init__(self, config, layer_cls=Qwen35DecoderLayer) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(int(config.vocab_size), int(config.hidden_size))
        self.layers = nn.ModuleList(
            [layer_cls(config, layer_idx) for layer_idx in range(int(config.num_hidden_layers))]
        )
        self.norm = Qwen35RMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps))
        self.sparse_controller = None
        self.recurrent_state_manager = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        residual = None
        context = get_context()
        debug_layers_env = os.getenv("SPARSEENGINE_DEBUG_HIDDEN_LAYERS")
        debug_layers = None
        if debug_layers_env:
            debug_layers = {int(part) for part in debug_layers_env.split(",") if part.strip()}
            self.debug_last_hidden_states = {-1: hidden_states[-1:].detach().clone()}

        for layer_idx, layer in enumerate(self.layers):
            context.now_layer_idx = layer_idx
            hidden_states, residual = layer(positions, hidden_states, residual)
            if self.sparse_controller is not None:
                hidden_states, residual = self.sparse_controller.apply_activation_hook(
                    layer_idx,
                    hidden_states,
                    residual,
                    context,
                )
            if debug_layers is not None and layer_idx in debug_layers:
                layer_output = hidden_states if residual is None else hidden_states + residual
                self.debug_last_hidden_states[int(layer_idx)] = layer_output[-1:].detach().clone()
            if self.sparse_controller is not None:
                self.sparse_controller.on_layer_end(layer_idx, context)

        hidden_states, _ = self.norm(hidden_states, residual)
        if debug_layers is not None:
            self.debug_last_hidden_states[int(self.config.num_hidden_layers)] = hidden_states[-1:].detach().clone()
        return hidden_states


class Qwen35ForCausalLM(nn.Module):
    ignored_weight_prefixes = ("model.visual.", "visual.", "mtp.")
    packed_modules_excluded_prefixes = ("multimodal_encoder.",)
    special_weight_loaders = {
        ".linear_attn.in_proj_qkv.weight": "load_packed_in_proj_qkv",
        ".linear_attn.in_proj_qkvz.weight": "load_packed_in_proj_qkvz",
        ".linear_attn.in_proj_ba.weight": "load_packed_in_proj_ba",
    }
    packed_modules_mapping = {
        "o_gate_proj": ("qkv_gate_proj", "gate"),
        "q_proj": ("qkv_gate_proj", "q"),
        "k_proj": ("qkv_gate_proj", "k"),
        "v_proj": ("qkv_gate_proj", "v"),
        "in_proj_z": ("in_proj_qkvz", 3),
        "in_proj_b": ("in_proj_ba", 0),
        "in_proj_a": ("in_proj_ba", 1),
        "shared_expert_gate": ("gate", "shared"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    @staticmethod
    def build_runtime_kwargs(
        config,
        *,
        engine_config,
        parallel_context,
        device: torch.device,
        **_,
    ) -> dict:
        full_attention_provider = build_mha_full_attention_provider(
            config,
            sparse_method=engine_config.sparse_method,
            attention_tp_size=parallel_context.attn_tp_size,
            device=device,
            max_batch_size=engine_config.max_decoding_seqs,
            cuda_graph=engine_config.decode_graph,
            runtime_config=engine_config,
        )
        try:
            gated_delta_rule_op = build_gated_delta_rule_op(
                config,
                attention_tp_size=parallel_context.attn_tp_size,
                device=device,
                cuda_graph=engine_config.decode_graph,
            )
        except BaseException:
            full_attention_provider.close()
            raise
        return {
            "full_attention_provider": full_attention_provider,
            "gated_delta_rule_op": gated_delta_rule_op,
        }

    def __init__(
        self,
        config,
        full_attention_provider: FullAttentionProvider | None = None,
        gated_delta_rule_op: PreparedGatedDeltaRuleOp | None = None,
    ) -> None:
        super().__init__()
        self.full_attention_provider = full_attention_provider
        self.gated_delta_rule_op = gated_delta_rule_op
        self.model = Qwen35Model(config)
        if full_attention_provider is not None:
            bind_mha_full_attention_provider(self.model, full_attention_provider)
        if gated_delta_rule_op is not None:
            bind_gated_delta_rule_op(self.model, gated_delta_rule_op)
        self.lm_head = ParallelLMHead(int(config.vocab_size), int(config.hidden_size))
        if bool(getattr(config, "tie_word_embeddings", False)):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        self.multimodal_encoder = None
        logger.info(
            "Loaded Qwen3.5 prefill_provider={} decode_provider={} gdn_provider={}",
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

    def close_runtime_operators(self) -> None:
        if self.full_attention_provider is not None:
            self.full_attention_provider.close()
        if self.gated_delta_rule_op is not None:
            self.gated_delta_rule_op.close()

    def configure_multimodal(self, outer_config) -> None:
        from sparseengine.models.qwen3_5_multimodal import Qwen35MultimodalEncoder

        self.multimodal_encoder = Qwen35MultimodalEncoder(outer_config)
        self.ignored_weight_prefixes = ("mtp.",)

    def encode_multimodal(self, input_ids, tensors):
        if self.multimodal_encoder is None:
            raise RuntimeError("Qwen3.5 multimodal encoder is disabled.")
        return self.multimodal_encoder.encode(input_ids, tensors)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward_multimodal(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        multimodal_mask: torch.Tensor,
    ) -> torch.Tensor:
        del multimodal_mask
        return self.model(input_ids, positions, inputs_embeds)

    @staticmethod
    def recurrent_state_spec(config, world_size: int) -> RecurrentStateSpec:
        world_size = int(world_size)
        num_k_heads = int(config.linear_num_key_heads) // world_size
        num_v_heads = int(config.linear_num_value_heads) // world_size
        key_head_dim = int(config.linear_key_head_dim)
        value_head_dim = int(config.linear_value_head_dim)
        conv_dim = 2 * num_k_heads * key_head_dim + num_v_heads * value_head_dim
        return RecurrentStateSpec(
            name="qwen3_5 gated delta net",
            tensor_specs=(
                RecurrentTensorSpec(
                    "conv_state",
                    (conv_dim, int(config.linear_conv_kernel_dim) - 1),
                    config.dtype,
                ),
                RecurrentTensorSpec(
                    "recurrent_state",
                    (num_v_heads, key_head_dim, value_head_dim),
                    config.dtype,
                ),
            ),
        )

    def map_weight_name(self, source_weight_name: str) -> str:
        prefix = "model.language_model."
        if source_weight_name.startswith(prefix):
            return "model." + source_weight_name[len(prefix) :]
        if source_weight_name.startswith("model.visual."):
            return "multimodal_encoder.visual." + source_weight_name[len("model.visual.") :]
        if source_weight_name.startswith("visual."):
            return "multimodal_encoder.visual." + source_weight_name[len("visual.") :]
        return source_weight_name

    def resolve_special_weight(
        self,
        target_weight_name: str,
    ) -> WeightTarget | None:
        for suffix, loader_name in self.special_weight_loaders.items():
            if not target_weight_name.endswith(suffix):
                continue
            module_name = target_weight_name[: -len(".weight")].rpartition(".")[0]
            return WeightTarget(self.get_submodule(module_name), loader_name)
        return None

    def load_special_weight(
        self,
        target_weight_name: str,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None,
    ) -> int:
        target = self.resolve_special_weight(target_weight_name)
        if target is None:
            return 0
        loader = getattr(target.module, target.shard_id, None)
        if loader is None:
            raise ValueError(
                f"Found qwen3_5 packed linear attention weight {target_weight_name!r}, "
                f"but target module {type(target.module).__name__!r} has no "
                f"{target.shard_id}()."
            )
        return int(loader(loaded_weight, loaded_scale))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
