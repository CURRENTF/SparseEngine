from __future__ import annotations

import torch
from torch import nn

from sparseengine.distributed import get_parallel_context
from sparseengine.layers.expert_weights import UnquantizedExpertTpShard
from sparseengine.layers.packed_moe import PackedMoeExperts
from sparseengine.operators.moe import MoeOpSpec, resolve_moe_provider


class PackedMxfp4Experts(PackedMoeExperts):
    """Logical packed E2M1 checkpoint storage with provider-owned preparation.

    Reuses packed experts' ownership, validation, forward and workspace contracts.
    MXFP4 supports expert sharding with TP=1; each checkpoint K byte stores two
    values and each scale covers 32 values.
    """
    checkpoint_projection_map = {"w1": "gate", "w3": "up", "w2": "down"}

    def __init__(self, *, num_experts, hidden_size, intermediate_size, top_k,
                 activation_limit, cuda_graph, max_num_tokens, routing_method="sqrtsoftplus",
                 parallel_context=None, provider_resolver=resolve_moe_provider):
        nn.Module.__init__(self)
        parallel = parallel_context or get_parallel_context()
        self.tp_rank, self.tp_size = parallel.moe_tp_rank, parallel.moe_tp_size
        self.ep_rank, self.ep_size = parallel.moe_ep_rank, parallel.moe_ep_size
        if self.tp_size != 1:
            raise ValueError("MXFP4 packed experts require MoE TP=1")
        self.model_label = "MXFP4"
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = self.global_intermediate_size = intermediate_size
        if num_experts <= 0 or num_experts % self.ep_size:
            raise ValueError("MXFP4 expert count must be positive and divisible by EP size")
        self.num_local_experts = num_experts // self.ep_size
        self.local_expert_start = self.ep_rank*self.num_local_experts
        self.local_expert_end = self.local_expert_start+self.num_local_experts
        self.checkpoint_tp_shard = UnquantizedExpertTpShard(intermediate_size, 0, 1)
        self.op_spec = MoeOpSpec(
            num_experts=num_experts, num_local_experts=self.num_local_experts,
            hidden_size=hidden_size, intermediate_size=intermediate_size, top_k=top_k,
            activation_dtype=torch.bfloat16, weight_dtype=torch.uint8, block_shape=(1, 32),
            ep_size=self.ep_size, tp_size=1, cuda_graph=cuda_graph, routing_method=routing_method,
            scale_dtype=torch.float8_e8m0fnu, activation="clipped_silu",
            activation_limit=activation_limit, max_num_tokens=max_num_tokens,
        )
        self.provider = provider_resolver(self.op_spec)
        self.w13_weight = nn.Parameter(torch.empty(self.num_local_experts, 2*intermediate_size,
                                                   hidden_size//2, dtype=torch.uint8), requires_grad=False)
        self.w2_weight = nn.Parameter(torch.empty(self.num_local_experts, hidden_size,
                                                  intermediate_size//2, dtype=torch.uint8), requires_grad=False)
        self.register_buffer("w13_scale_inv", torch.empty(self.num_local_experts, 2*intermediate_size,
                                                         hidden_size//32, dtype=torch.float8_e8m0fnu))
        self.register_buffer("w2_scale_inv", torch.empty(self.num_local_experts, hidden_size,
                                                        intermediate_size//32, dtype=torch.float8_e8m0fnu))
        self._loaded_expert_shards = set()
        self._weights_prepared = False
        self.provider.prepare(self.op_spec, device=self.w13_weight.device, tp_rank=0,
                              ep_rank=self.ep_rank)

    def load_expert_weight(self, global_expert_id, projection, loaded_weight, loaded_scale=None):
        if self._weights_prepared:
            raise RuntimeError("Cannot change prepared MXFP4 expert weights")
        if not self.is_local_expert(global_expert_id):
            raise ValueError("MXFP4 checkpoint expert is outside the local EP shard")
        logical = self.checkpoint_projection_map.get(projection)
        if logical is None:
            raise ValueError(f"Unsupported MXFP4 projection {projection!r}")
        key = (global_expert_id, projection)
        if key in self._loaded_expert_shards:
            raise ValueError(f"Duplicate MXFP4 checkpoint projection {key}")
        # Original V4 safetensors declares packed E2M1 bytes as I8. Preserve
        # their bit pattern; interpreting signed bytes numerically corrupts FP4.
        if loaded_weight.dtype == torch.int8:
            loaded_weight = loaded_weight.view(torch.uint8)
        if loaded_weight.dtype != torch.uint8 or loaded_scale is None:
            raise ValueError("MXFP4 checkpoint requires packed E2M1 weights and UE8M0 scales")
        if loaded_scale.dtype != torch.float8_e8m0fnu:
            raise TypeError("MXFP4 checkpoint scales must be UE8M0")
        self.provider.load_expert_projection(
            self.op_spec, local_expert_id=global_expert_id-self.local_expert_start,
            projection=logical, loaded_weight=loaded_weight, loaded_scale=loaded_scale,
            w13_weight=self.w13_weight.data, w2_weight=self.w2_weight.data,
            w13_scale_inv=self.w13_scale_inv, w2_scale_inv=self.w2_scale_inv,
        )
        self._loaded_expert_shards.add(key)

    def prepare_weights(self):
        self.validate_loaded_weights()
        if self._weights_prepared:
            raise RuntimeError("MXFP4 expert weights can only be prepared once")
        self.provider.prepare_weights(self.w13_weight.data, self.w2_weight.data,
                                      self.w13_scale_inv, self.w2_scale_inv)
        self._weights_prepared = True
