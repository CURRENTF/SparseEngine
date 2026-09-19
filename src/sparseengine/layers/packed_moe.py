from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from sparseengine.distributed import get_parallel_context
from sparseengine.layers.expert_weights import (
    PackedExpertWeightLoader,
    UnquantizedExpertTpShard,
)
from sparseengine.operators.moe import (
    MoeOpSpec,
    MoeProvider,
    resolve_moe_provider,
)
from sparseengine.quantization.fp8_tp import Fp8ExpertTpShard


class PackedMoeExperts(PackedExpertWeightLoader, nn.Module):
    """Shared packed expert storage, execution, and checkpoint loading."""

    def bind_workspace_lane(self, lane: str) -> None:
        # Standalone shared projections can live inside a packed expert owner.
        self.provider.bind_workspace_lane(lane)

    checkpoint_projection_map = {
        "gate_proj": "gate",
        "up_proj": "up",
        "down_proj": "down",
    }
    checkpoint_scale_dtype = torch.bfloat16

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        activation_dtype: torch.dtype,
        fp8_enabled: bool,
        cuda_graph: bool,
        fp8_tensor_scales: bool = False,
        routing_method: str = "softmax",
        scale_dtype: torch.dtype | None = None,
        activation: str = "silu",
        max_num_tokens: int = 1,
        model_label: str = "PackedMoE",
        provider_resolver: Callable[[MoeOpSpec], MoeProvider] = resolve_moe_provider,
        parallel_context=None,
    ) -> None:
        super().__init__()
        self.model_label = str(model_label)
        if parallel_context is None:
            parallel_context = get_parallel_context()
        self.tp_rank = parallel_context.moe_tp_rank
        self.tp_size = parallel_context.moe_tp_size
        self.ep_rank = parallel_context.moe_ep_rank
        self.ep_size = parallel_context.moe_ep_size
        self.num_experts = int(num_experts)
        self.hidden_size = int(hidden_size)
        self.global_intermediate_size = int(intermediate_size)
        if self.global_intermediate_size % self.tp_size:
            raise ValueError(
                f"{self.model_label} intermediate_size must be divisible by "
                f"MoE tensor parallel size, got "
                f"{self.global_intermediate_size} and {self.tp_size}."
            )
        self.logical_intermediate_size = (
            self.global_intermediate_size // self.tp_size
        )
        self.fp8_enabled = bool(fp8_enabled)
        self.fp8_tensor_scales = bool(fp8_tensor_scales)
        if self.fp8_tensor_scales and not self.fp8_enabled:
            raise ValueError("Tensor FP8 scales require FP8 weights")
        scale_dtype = (scale_dtype or torch.float32) if self.fp8_enabled else None
        if self.fp8_enabled and (
            self.hidden_size % 128 or self.global_intermediate_size % 128
        ):
            raise ValueError(
                f"{self.model_label} FP8 requires hidden/intermediate sizes "
                "aligned to 128, got "
                f"{self.hidden_size}/{self.global_intermediate_size}."
            )
        self.fp8_tp_shard = (
            Fp8ExpertTpShard(
                self.global_intermediate_size,
                self.tp_rank,
                self.tp_size,
            )
            if self.fp8_enabled
            else None
        )
        self.checkpoint_tp_shard = (
            self.fp8_tp_shard
            if self.fp8_tp_shard is not None
            else UnquantizedExpertTpShard(
                self.global_intermediate_size,
                self.tp_rank,
                self.tp_size,
            )
        )
        self.intermediate_size = (
            self.fp8_tp_shard.physical_size
            if self.fp8_tp_shard is not None
            else self.logical_intermediate_size
        )
        if self.num_experts <= 0 or self.num_experts % self.ep_size:
            raise ValueError(
                f"{self.model_label} num_experts must be positive and divisible "
                f"by EP size, got {self.num_experts} and {self.ep_size}."
            )
        self.num_local_experts = self.num_experts // self.ep_size
        self.local_expert_start = self.ep_rank * self.num_local_experts
        self.local_expert_end = self.local_expert_start + self.num_local_experts
        self.op_spec = MoeOpSpec(
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            top_k=int(top_k),
            activation_dtype=activation_dtype,
            weight_dtype=(
                torch.float8_e4m3fn if self.fp8_enabled else activation_dtype
            ),
            block_shape=(128, 128) if self.fp8_enabled and not self.fp8_tensor_scales else None,
            ep_size=int(self.ep_size),
            cuda_graph=bool(cuda_graph),
            tp_size=int(self.tp_size),
            routing_method=str(routing_method),
            scale_dtype=scale_dtype,
            activation=str(activation),
            max_num_tokens=int(max_num_tokens),
        )
        self.provider = provider_resolver(self.op_spec)
        self.w13_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                2 * self.intermediate_size,
                self.hidden_size,
                dtype=torch.float8_e4m3fn if self.fp8_enabled else None,
            ),
            requires_grad=not self.fp8_enabled,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_size,
                self.intermediate_size,
                dtype=torch.float8_e4m3fn if self.fp8_enabled else None,
            ),
            requires_grad=not self.fp8_enabled,
        )
        if self.fp8_enabled:
            self.register_buffer(
                "w13_scale_inv",
                torch.empty(
                    self.num_local_experts,
                    2 * self.intermediate_size // 128,
                    self.hidden_size // 128,
                    dtype=scale_dtype,
                ),
            )
            self.register_buffer(
                "w2_scale_inv",
                torch.empty(
                    self.num_local_experts,
                    self.hidden_size // 128,
                    self.intermediate_size // 128,
                    dtype=scale_dtype,
                ),
            )
        else:
            self.register_buffer("w13_scale_inv", None)
            self.register_buffer("w2_scale_inv", None)
        self._loaded_expert_shards: set[tuple[int, str]] = set()
        self.provider.prepare(
            self.op_spec,
            device=self.w13_weight.device,
            tp_rank=int(self.tp_rank),
            ep_rank=int(self.ep_rank),
        )

    def is_local_expert(self, global_expert_id: int) -> bool:
        return self.local_expert_start <= int(global_expert_id) < self.local_expert_end

    def load_expert_weight(
        self,
        global_expert_id: int,
        projection: str,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None = None,
    ) -> None:
        global_expert_id = int(global_expert_id)
        if not self.is_local_expert(global_expert_id):
            raise ValueError(
                f"Expert {global_expert_id} is outside local range "
                f"[{self.local_expert_start}, {self.local_expert_end})."
            )
        logical_projection = self.checkpoint_projection_map.get(projection)
        if logical_projection is None:
            raise ValueError(f"Unsupported expert projection {projection!r}.")
        load_key = (global_expert_id, projection)
        if load_key in self._loaded_expert_shards:
            raise ValueError(
                f"Duplicate {self.model_label} expert weight for "
                f"expert={global_expert_id}, projection={projection}."
            )

        if self.fp8_enabled:
            if loaded_scale is None:
                raise ValueError(
                    f"Missing FP8 weight_scale_inv for {self.model_label} "
                    f"expert={global_expert_id}, projection={projection}."
                )
            if self.fp8_tensor_scales and not torch.equal(loaded_scale, loaded_scale.flatten()[0].expand_as(loaded_scale)):
                raise ValueError("Per-tensor FP8 expert projection requires one constant scale")
            if loaded_weight.dtype != torch.float8_e4m3fn:
                raise TypeError(
                    f"{self.model_label} expert weight must be FP8 E4M3, "
                    f"got {loaded_weight.dtype}."
                )
            if loaded_scale.dtype != self.checkpoint_scale_dtype:
                raise TypeError(
                    f"{self.model_label} expert weight_scale_inv must use "
                    f"{self.checkpoint_scale_dtype}, "
                    f"got {loaded_scale.dtype}."
                )
        elif loaded_scale is not None:
            raise ValueError(
                f"Unexpected weight_scale_inv for unquantized {self.model_label} "
                f"expert={global_expert_id}, projection={projection}."
            )

        if self.fp8_tp_shard is not None:
            loaded_weight, loaded_scale = self.fp8_tp_shard.prepare_projection(
                loaded_weight,
                loaded_scale,
                hidden_size=self.hidden_size,
                down_projection=logical_projection == "down",
            )
        else:
            loaded_weight = self._local_projection_shard(
                logical_projection,
                loaded_weight,
            )
        local_expert_id = global_expert_id - self.local_expert_start
        self.provider.load_expert_projection(
            self.op_spec,
            local_expert_id=local_expert_id,
            projection=logical_projection,
            loaded_weight=loaded_weight,
            loaded_scale=loaded_scale,
            w13_weight=self.w13_weight.data,
            w2_weight=self.w2_weight.data,
            w13_scale_inv=self.w13_scale_inv,
            w2_scale_inv=self.w2_scale_inv,
        )
        self._loaded_expert_shards.add(load_key)

    def _local_projection_shard(
        self,
        logical_projection: str,
        loaded_weight: torch.Tensor,
    ) -> torch.Tensor:
        down_projection = logical_projection == "down"
        local_shape = (
            (self.hidden_size, self.intermediate_size)
            if down_projection
            else (self.intermediate_size, self.hidden_size)
        )
        if tuple(loaded_weight.shape) == local_shape:
            return loaded_weight

        global_shape = (
            (self.hidden_size, self.global_intermediate_size)
            if down_projection
            else (self.global_intermediate_size, self.hidden_size)
        )
        if tuple(loaded_weight.shape) != global_shape:
            raise ValueError(
                f"{self.model_label} expert projection shape mismatch: "
                f"projection={logical_projection}, expected local={local_shape} "
                f"or global={global_shape}, got={tuple(loaded_weight.shape)}."
            )
        shard_dim = 1 if down_projection else 0
        return loaded_weight.chunk(self.tp_size, dim=shard_dim)[self.tp_rank]

    def validate_loaded_weights(self) -> None:
        expected = {
            (global_expert_id, projection)
            for global_expert_id in range(
                self.local_expert_start,
                self.local_expert_end,
            )
            for projection in self.checkpoint_projection_map
        }
        missing = sorted(expected - self._loaded_expert_shards)
        if missing:
            raise ValueError(
                f"Missing local {self.model_label} expert weights: "
                f"local_range=[{self.local_expert_start}, "
                f"{self.local_expert_end}), missing={missing[:8]}."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.provider.run(
            self.op_spec,
            hidden_states,
            topk_ids,
            topk_weights,
            self.w13_weight,
            self.w2_weight,
            self.w13_scale_inv,
            self.w2_scale_inv,
            local_expert_start=self.local_expert_start,
            tp_rank=int(self.tp_rank),
            ep_rank=int(self.ep_rank),
        )


__all__ = ["PackedMoeExperts"]
