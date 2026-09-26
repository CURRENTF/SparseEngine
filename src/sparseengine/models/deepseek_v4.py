from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, log, pi
import re

import torch
from torch import nn

from sparseengine.distributed import get_parallel_context
from sparseengine.distributed.moe_communication import prepare_moe_communication
from sparseengine.engine.sparse_methods.deepseek_v4 import SharedKVSelectionQuery
from sparseengine.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from sparseengine.layers.layernorm import RMSNorm
from sparseengine.layers.linear import LinearBase
from sparseengine.layers.mxfp4_experts import PackedMxfp4Experts
from sparseengine.operators.activation import clipped_swiglu
from sparseengine.operators.compressed_index import CompressedIndexOpSpec, resolve_compressed_index_provider
from sparseengine.operators.hyper_connection import HyperConnectionOpSpec, resolve_hyper_connection_provider
from sparseengine.operators.kv_compression import KVCompressionOpSpec, resolve_kv_compression_provider
from sparseengine.operators.moe_router import MoeRouterOpSpec, resolve_moe_router_provider
from sparseengine.operators.shared_kv_attention import SharedKVAttentionOpSpec, resolve_shared_kv_attention_provider
from sparseengine.operators.shared_kv_transform import (
    SharedKVTransformOpSpec, resolve_shared_kv_transform_provider,
    GroupedSharedKVProjection, inverse_shared_kv_rope,
)
from sparseengine.utils.context import get_context
from sparseengine.utils.weight_target import WeightTarget


_EXPERT = re.compile(r"^model\.layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.(weight|expert_weight)$")


def _linear(config, input_size, output_size, *, quantized=True):
    return LinearBase(input_size, output_size, quantization=config.quantization_config if quantized else None)


def _freqs(config, max_length, *, compressed, device):
    # Original V4 uses complex/interleaved YaRN with no attention rescaling.
    dim = int(config.qk_rope_head_dim)
    theta = float(config.compress_rope_theta if compressed else config.rope_theta)
    inverse = theta**(-torch.arange(0, dim, 2, device=device, dtype=torch.float32)/dim)
    if compressed:
        yarn = config.rope_scaling
        if not isinstance(yarn, dict):
            yarn = vars(yarn)
        original = int(yarn["original_max_position_embeddings"])
        low = max(0, floor(dim*log(original/(float(yarn["beta_fast"])*2*pi))/(2*log(theta))))
        high = min(dim-1, ceil(dim*log(original/(float(yarn["beta_slow"])*2*pi))/(2*log(theta))))
        ramp = ((torch.arange(dim//2, device=device, dtype=torch.float32)-low)
                /max(float(high-low), .001)).clamp(0, 1)
        inverse = inverse*(1-ramp)+inverse/float(yarn["factor"])*ramp
    angles = torch.arange(max_length, device=device, dtype=torch.float32)[:, None]*inverse
    return torch.stack((angles.cos(), angles.sin()), -1).flatten(1).contiguous()


class V4Compressor(nn.Module):
    def __init__(self, config, ratio, head_dim, device_index):
        super().__init__()
        width = head_dim*(2 if ratio == 4 else 1)
        self.wkv = _linear(config, config.hidden_size, width, quantized=False)
        self.wgate = _linear(config, config.hidden_size, width, quantized=False)
        self.ape = nn.Parameter(torch.empty(ratio, width, dtype=torch.float32), requires_grad=False)
        self.norm = RMSNorm(head_dim, config.rms_norm_eps)
        self.provider = resolve_kv_compression_provider(
            KVCompressionOpSpec(ratio, head_dim, config.hidden_size, torch.bfloat16,
                                config.rms_norm_eps, bool(config.decode_graph)), device_index=device_index)

    def prepare_weights(self):
        self.provider.prepare_weights(self.wkv.weight, self.wgate.weight, self.ape, self.norm.weight)
        del self.wkv, self.wgate, self.ape, self.norm

    def forward(self, hidden_states, view, freqs):
        return self.provider.compute(self.provider.project(hidden_states), view, freqs)


class V4GroupedOutput(nn.Module):
    quantized = True
    _quantized_weight_loaded = False

    def __init__(self, config):
        super().__init__()
        self.provider = GroupedSharedKVProjection(num_groups=config.o_groups,
                                                  input_size_per_group=config.num_attention_heads*config.head_dim//config.o_groups,
                                                  output_size_per_group=config.o_lora_rank)
        self.weight = nn.Parameter(torch.empty(0, dtype=torch.bfloat16), requires_grad=False)

    def load_quantized_weight(self, weight, scales, shard_id=None):
        if shard_id is not None or self._quantized_weight_loaded:
            raise ValueError("Grouped output checkpoint weights must load once without sharding")
        self.provider.prepare_weights(weight.to(self.weight.device), scales.to(self.weight.device))
        self.weight.data = self.provider.weight.flatten(0, 1)
        self._quantized_weight_loaded = True

    def forward(self, values):
        return self.provider.forward(values)


class V4Indexer(nn.Module):
    def __init__(self, config, device_index):
        super().__init__()
        self.wq_b = _linear(config, config.q_lora_rank, config.index_n_heads*config.index_head_dim)
        self.weights_proj = _linear(config, config.hidden_size, config.index_n_heads, quantized=False)
        self.compressor = V4Compressor(config, 4, config.index_head_dim, device_index)
        self.provider = resolve_compressed_index_provider(
            CompressedIndexOpSpec(64, 128, 64, config.index_topk, 64, bool(config.decode_graph)),
            device_index=device_index)

    def forward(self, hidden_states, query_latent, positions, freqs, manager, layer_idx):
        keys = self.compressor(hidden_states, manager.compression_compute_view(layer_idx, index=True), freqs)
        self.provider.store_keys(keys.bfloat16(), manager.index_cache(layer_idx), manager.compressed_write_slots(layer_idx))
        query = self.provider.prepare_query(self.wq_b(query_latent).view(-1, 64, 128), freqs, positions.to(torch.int32))
        weights = self.weights_proj(hidden_states)*(128**-.5*64**-.5)
        return query, weights


@dataclass
class _SharedKVGraphState:
    planner: object
    captured: bool = False


class PreparedSharedKVDecode:
    supports_decode_graph = True
    decode_graph_lifecycle = True

    def __init__(self, provider):
        self.provider = provider
        self.active = None
        self.eager = {}

    def init_decode_graph_state(self, contract, inputs):
        return _SharedKVGraphState(self.provider.create_decode_state(contract.batch_capacity))

    def prepare_decode_graph_out(self, state):
        self.active = state
        if not state.captured:
            self.provider.prepare_decode_state(state.planner)

    def prepare_decode_graph_in(self, state):
        self.active = state
        if torch.cuda.is_current_stream_capturing():
            state.captured = True

    def decode_graph_keepalive_tensors(self, state):
        return state.planner.keepalive_tensors()

    def close_decode_graph_state(self, state):
        if self.active is state:
            self.active = None

    def run(self, query, payload, indices, lengths, sink):
        if torch.cuda.is_current_stream_capturing() and self.active is not None:
            state = self.active.planner
        else:
            capacity = len(query)
            if capacity not in self.eager:
                self.eager[capacity] = self.provider.create_decode_state(capacity)
            state = self.eager[capacity]
            self.provider.prepare_decode_state(state)
        return self.provider.decode(query, payload, indices, lengths, sink, state)


class V4Attention(nn.Module):
    is_attention_layer = True

    def __init__(self, config, layer_idx, freqs, device_index, max_length):
        super().__init__()
        self.layer_idx = layer_idx
        self.ratio = int(config.compress_ratios[layer_idx])
        self.wq_a = _linear(config, config.hidden_size, config.q_lora_rank)
        self.q_norm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.wq_b = _linear(config, config.q_lora_rank, config.num_attention_heads*config.head_dim)
        self.wkv = _linear(config, config.hidden_size, config.head_dim)
        self.kv_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.wo_a = V4GroupedOutput(config)
        self.wo_b = _linear(config, config.o_groups*config.o_lora_rank, config.hidden_size)
        self.attn_sink = nn.Parameter(torch.empty(config.num_attention_heads, dtype=torch.float32), requires_grad=False)
        self.register_buffer("freqs", freqs, persistent=False)
        self.groups = config.o_groups
        self.heads = config.num_attention_heads
        self.transforms = resolve_shared_kv_transform_provider(
            SharedKVTransformOpSpec(64, torch.bfloat16, config.rms_norm_eps, bool(config.decode_graph)),
            device_index=device_index)
        compressed_capacity = (config.index_topk if self.ratio == 4 else ceil(max_length/self.ratio) if self.ratio else 0)
        capacity = ((config.sliding_window+compressed_capacity+63)//64)*64
        self.attention_provider = resolve_shared_kv_attention_provider(
            SharedKVAttentionOpSpec(config.num_attention_heads, config.head_dim, torch.bfloat16,
                                    64, capacity, config.head_dim**-.5, bool(config.decode_graph)),
            device_index=device_index)
        self.decode_op = PreparedSharedKVDecode(self.attention_provider)
        self.compressor = V4Compressor(config, self.ratio, config.head_dim, device_index) if self.ratio else None
        self.indexer = V4Indexer(config, device_index) if self.ratio == 4 else None

    def forward(self, positions, hidden_states):
        context = get_context()
        manager, controller = context.cache_manager, context.sparse_controller
        layer = self.layer_idx
        query_latent = self.q_norm(self.wq_a(hidden_states))
        query = self.transforms.normalize_rotate_query(self.wq_b(query_latent).view(-1, self.heads, 512),
                                                       positions, self.freqs)
        if self.compressor is not None:
            compressed = self.compressor(hidden_states, manager.compression_compute_view(layer), self.freqs)
            self.transforms.store(compressed, manager.pools[layer].byte_storage, manager.compressed_write_slots(layer))
        index_query, head_weights = (self.indexer(hidden_states, query_latent, positions, self.freqs, manager, layer)
                                     if self.indexer is not None else (None, None))
        manager.update_window(layer, self.wkv(hidden_states), self.kv_norm.weight, self.freqs, self.transforms)
        candidates = manager.candidate_view(layer, self.transforms)
        selection_query = SharedKVSelectionQuery(candidates, index_query, head_weights)
        if context.is_prefill:
            selection = controller.get_prefill_selection(layer, selection_query=selection_query)
            output = self.attention_provider.prefill(query, candidates.payload, selection.active_slots,
                                                     selection.context_lens, self.attn_sink)
        else:
            selection = controller.get_decode_selection(layer, query, selection_query=selection_query)
            output = self.decode_op.run(query, candidates.payload, selection.active_slots,
                                         selection.context_lens, self.attn_sink)
        output = inverse_shared_kv_rope(output, positions, self.freqs)
        return self.wo_b(self.wo_a(output.reshape(-1, self.groups, self.heads*512//self.groups)))


class V4Router(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hash = layer_idx < config.num_hash_layers
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size), requires_grad=False)
        self.spec = MoeRouterOpSpec(config.n_routed_experts, config.num_experts_per_tok, torch.float32,
                                    True, bool(config.decode_graph),
                                    "hash_sqrtsoftplus" if self.hash else "sqrtsoftplus")
        self.provider = resolve_moe_router_provider(self.spec)
        self.scale = float(config.routed_scaling_factor)
        if self.hash:
            # Checkpoint INT64 is losslessly narrowed after loading; provider consumes INT32.
            self.register_buffer("tid2eid", torch.empty(config.vocab_size, config.num_experts_per_tok, dtype=torch.int32))
            self.bias = None
        else:
            self.register_buffer("tid2eid", None)
            self.bias = nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.float32), requires_grad=False)

    def forward(self, hidden_states, input_ids):
        logits = self.provider.project(hidden_states, self.weight)
        kwargs = {"input_ids": input_ids, "tid2eid": self.tid2eid} if self.hash else {}
        weights, ids = self.provider.run(self.spec, logits, self.bias, routed_scaling_factor=self.scale, **kwargs)
        return ids, weights


class V4SharedExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.n_shared_experts*config.moe_intermediate_size
        self.w1 = _linear(config, config.hidden_size, width)
        self.w3 = _linear(config, config.hidden_size, width)
        self.w2 = _linear(config, width, config.hidden_size)
        self.limit = float(config.swiglu_limit)

    def forward(self, hidden_states):
        return self.w2(clipped_swiglu(self.w1(hidden_states), self.w3(hidden_states), self.limit))


class V4Moe(nn.Module):
    def __init__(self, config, layer_idx, communication):
        super().__init__()
        self.communication = communication
        self.gate = V4Router(config, layer_idx)
        self.experts = PackedMxfp4Experts(num_experts=config.n_routed_experts, hidden_size=config.hidden_size,
                                         intermediate_size=config.moe_intermediate_size,
                                         top_k=config.num_experts_per_tok, activation_limit=config.swiglu_limit,
                                         cuda_graph=bool(config.decode_graph), max_num_tokens=config.moe_max_num_tokens,
                                         routing_method="hash_sqrtsoftplus" if self.gate.hash else "sqrtsoftplus")
        self.shared_experts = V4SharedExpert(config)
        self.chunk_size = config.mlp_chunk_size

    def forward(self, hidden_states, token_metadata):
        offset = 0
        def route(chunk):
            nonlocal offset
            ids = token_metadata[offset:offset+len(chunk)]
            offset += len(chunk)
            return self.gate(chunk, ids)
        return self.communication.run_with_shared_experts(hidden_states, route=route, experts=self.experts,
                                                          chunk_size=self.chunk_size, shared_experts=self.shared_experts)


class V4Block(nn.Module):
    def __init__(self, config, layer_idx, freqs, mhc, communication, device_index, max_length):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = V4Attention(config, layer_idx, freqs, device_index, max_length)
        self.ffn = V4Moe(config, layer_idx, communication)
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mhc = mhc
        for branch in ("attn", "ffn"):
            for suffix, shape in (("fn", (24, 4*config.hidden_size)), ("base", (24,)), ("scale", (3,))):
                self.register_parameter(f"hc_{branch}_{suffix}", nn.Parameter(torch.empty(*shape, dtype=torch.float32), requires_grad=False))

    def forward(self, positions, residual, token_metadata):
        get_context().now_layer_idx = self.layer_idx
        post, mix, hidden = self.mhc.pre(residual, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        residual = self.mhc.post(self.attn(positions, self.attn_norm(hidden)), residual, post, mix)
        post, mix, hidden = self.mhc.pre(residual, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        return self.mhc.post(self.ffn(self.ffn_norm(hidden), token_metadata), residual, post, mix)


class DeepseekV4ForCausalLM(nn.Module):
    checkpoint_block_scale_suffixes = (".scale",)
    special_weight_loaders = (".expert_weight",)

    @staticmethod
    def build_runtime_kwargs(config, *, engine_config, parallel_context, collective_runtime,
                             device, max_decode_tokens, **_):
        return {"max_length": engine_config.max_model_len,
                "parallel_collectives": collective_runtime.request_moe_collectives(
                    attention_max_rows=max_decode_tokens, moe_max_rows=max_decode_tokens,
                    max_local_tokens=engine_config.max_num_batched_tokens,
                    hidden_size=config.hidden_size, dtype=torch.bfloat16, backend=engine_config.moe_backend,
                    num_experts=config.n_routed_experts, top_k=config.num_experts_per_tok)}

    def __init__(self, config, *, max_length, parallel_collectives=None):
        super().__init__()
        self.config = config
        parallel = get_parallel_context()
        if parallel.attn_tp_size != 1 or parallel.moe_tp_size != 1:
            raise ValueError("DeepSeek V4 native inference requires attention and MoE TP=1")
        self.communication = prepare_moe_communication(parallel, parallel_collectives)
        device_index = torch.cuda.current_device()
        device = torch.device("cuda", device_index)
        self.mhc = resolve_hyper_connection_provider(
            HyperConnectionOpSpec(config.hidden_size, 4, torch.bfloat16, config.rms_norm_eps,
                                   config.hc_eps, config.hc_sinkhorn_iters), device_index=device_index)
        self.model = nn.Module()
        self.model.embed = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        plain = _freqs(config, max_length, compressed=False, device=device)
        compressed = _freqs(config, max_length, compressed=True, device=device)
        self.model.layers = nn.ModuleList([V4Block(config, i, compressed if config.compress_ratios[i] else plain,
                                                  self.mhc, self.communication, device_index, max_length)
                                          for i in range(config.num_hidden_layers)])
        self.model.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.hc_head_fn = nn.Parameter(torch.empty(4, 4*config.hidden_size, dtype=torch.float32), requires_grad=False)
        self.hc_head_base = nn.Parameter(torch.empty(4, dtype=torch.float32), requires_grad=False)
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)

    def map_weight_name(self, source):
        if source.startswith("layers."):
            index = int(source.split(".")[1])
            if index >= self.config.num_hidden_layers:
                return None
            source = "model."+source
        elif source == "embed.weight":
            return "model.embed.weight"
        elif source == "head.weight":
            return "lm_head.weight"
        elif source == "norm.weight":
            return "model.norm.weight"
        elif source.startswith(("mtp.", "dspark.")):
            return None
        match = _EXPERT.match(source)
        if match:
            layer, expert, projection, _ = match.groups()
            if not self.model.layers[int(layer)].ffn.experts.is_local_expert(int(expert)):
                return None
            return source.rsplit(".", 1)[0]+".expert_weight"
        return source

    def resolve_special_weight(self, target):
        match = _EXPERT.match(target)
        if not match:
            return None
        layer, expert, projection, _ = match.groups()
        return WeightTarget(self.model.layers[int(layer)].ffn.experts, (int(expert), projection))

    def load_special_weight(self, target, weight, scale):
        destination = self.resolve_special_weight(target)
        if destination is None:
            return 0
        expert, projection = destination.shard_id
        destination.module.load_expert_weight(expert, projection, weight, scale)
        return 1

    def validate_loaded_weights(self, loaded):
        expected = {name for name, _ in self.named_parameters()
                    if not name.endswith((".experts.w13_weight", ".experts.w2_weight"))}
        expected.update(name for name, _ in self.named_buffers() if name.endswith(".tid2eid"))
        missing = expected-loaded
        if missing:
            raise ValueError(f"Missing native V4 checkpoint weights: {sorted(missing)[:8]}")
        for layer in self.model.layers:
            layer.ffn.experts.prepare_weights()
            if layer.attn.compressor is not None:
                layer.attn.compressor.prepare_weights()
            if layer.attn.indexer is not None:
                layer.attn.indexer.compressor.prepare_weights()

    def forward(self, input_ids, positions):
        metadata = self.communication.routing_token_metadata(input_ids)
        hidden = self.model.embed(input_ids)[:, None].expand(-1, 4, -1).contiguous()
        for layer in self.model.layers:
            hidden = layer(positions, hidden, metadata)
        hidden = self.mhc.head(hidden, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
        return self.model.norm(hidden)

    def forward_idle_experts(self, hidden_states):
        metadata = self.communication.routing_token_metadata(torch.empty(0, dtype=torch.int64, device=hidden_states.device))
        for layer in self.model.layers:
            layer.ffn(hidden_states, metadata)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
