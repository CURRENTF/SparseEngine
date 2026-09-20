"""Prepared phase providers for grouped low-rank attention."""
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import torch
import sparseengine.platforms as platforms
from sparseengine.operators.registry import OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult
from sparseengine.platforms.interface import PlatformEnum


@dataclass(frozen=True)
class PaluSpec:
    query_heads: int
    kv_heads: int
    head_dim: int
    group_size: int
    key_rank: int
    value_rank: int
    dtype: torch.dtype
    max_batch_size: int
    capacity: int

    def __post_init__(self):
        if (self.kv_heads <= 0 or self.query_heads % self.kv_heads
                or self.group_size <= 0 or self.kv_heads % self.group_size
                or self.head_dim not in (64, 128, 256)
                or min(self.max_batch_size, self.capacity) <= 0
                or any(r < 16 or r % 16 or r > min(256, self.group_size * self.head_dim)
                       for r in (self.key_rank, self.value_rank))):
            raise ValueError('Invalid grouped low-rank attention shape/capacity.')


def _supports(spec, caps):
    if caps.platform != PlatformEnum.CUDA or not caps.supports_triton:
        return SupportResult.unsupported('Palu requires CUDA and Triton')
    if spec.dtype not in (torch.float16, torch.bfloat16):
        return SupportResult.unsupported('Palu requires FP16/BF16')
    if spec.dtype == torch.bfloat16 and not caps.supports_bfloat16:
        return SupportResult.unsupported('Device does not support BF16')
    return SupportResult.yes()


PREFILL = OpRegistry('palu_prefill', portfolio=PortfolioPolicy(upstream_standard=('flashinfer',)))
DECODE = OpRegistry('palu_decode', portfolio=PortfolioPolicy(repo_nonstandard=('triton_fused',)))


@PREFILL.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class FlashInferPaluPrefill:
    name = 'flashinfer'

    @classmethod
    def supports(cls, spec, caps):
        base = _supports(spec, caps)
        if not base.supported:
            return base
        if find_spec('flashinfer') is None:
            return SupportResult.unsupported('Palu prefill requires FlashInfer')
        return SupportResult.yes()

    def __init__(self, *, op_spec, device):
        from flashinfer import single_prefill_with_kv_cache
        from sparseengine.kernels.triton.palu import materialize_prefill
        self.spec, self.materialize, self.attention = op_spec, materialize_prefill, single_prefill_with_kv_cache

    def run(self, q, view, weights, plan):
        bk, norm, rope, eps = weights
        spec = self.spec
        output = torch.empty(q.shape[0], spec.query_heads, spec.value_rank, dtype=q.dtype, device=q.device)
        for start, size, row, length in plan:
            k, v = self.materialize(view.payload, bk, norm, rope, view.meta.active_slots,
                                   row, length, spec.group_size, eps)
            result = self.attention(q[start:start + size], k, v, causal=True,
                                   sm_scale=spec.head_dim ** -.5)
            output[start:start + size].copy_(result[..., :spec.value_rank])
        return output


@DECODE.register_atomic(ProviderRole.REPO_NONSTANDARD)
class TritonPaluDecode:
    name = 'triton_fused'
    supports_decode_graph = True
    decode_graph_lifecycle = False

    @classmethod
    def supports(cls, spec, caps):
        return _supports(spec, caps)

    def __init__(self, *, op_spec, device):
        from sparseengine.kernels.triton.palu import decode
        self.spec, self.kernel = op_spec, decode
        s = op_spec
        # Expose enough CTAs to fill the device, without redundant splits at
        # larger batches. Only batch size selects a prepared plan; lengths stay
        # on-device, so growing contexts reuse the same graph and JIT kernel.
        caps = platforms.current_platform.get_device_caps(int(device.index or 0))
        sms = caps.multi_processor_count
        if sms is None or sms <= 0:
            raise RuntimeError('Palu requires the CUDA multiprocessor count at preparation.')
        self.split_counts = tuple(
            min(32, (s.capacity + 31) // 32,
                max(1, (2 * sms) // (batch * s.kv_heads)))
            for batch in range(1, s.max_batch_size + 1)
        )
        splits = max(self.split_counts)
        self.mid_o = torch.empty(s.max_batch_size, s.query_heads, splits, s.value_rank, device=device, dtype=torch.float32)
        self.mid_lse = torch.empty(s.max_batch_size, s.query_heads, splits, device=device, dtype=torch.float32)
        self.output = torch.empty(s.max_batch_size, s.query_heads, s.value_rank, device=device, dtype=s.dtype)

    def run(self, q, view, weights):
        if q.shape[0] > self.spec.max_batch_size:
            raise ValueError('Palu decode batch exceeds prepared workspace.')
        bk, norm, rope, eps = weights
        batch = q.shape[0]
        splits = self.split_counts[batch - 1]
        count = batch * self.spec.query_heads * splits
        mid_o = self.mid_o.view(-1)[:count * self.spec.value_rank].view(
            batch, self.spec.query_heads, splits, self.spec.value_rank)
        mid_lse = self.mid_lse.view(-1)[:count].view(batch, self.spec.query_heads, splits)
        return self.kernel(q, view, bk, norm, rope, self.spec.group_size, eps,
                           mid_o, mid_lse, self.output, self.spec.head_dim ** -.5)


class PaluFullAttentionProvider:
    prefill_name = 'flashinfer'
    decode_name = 'triton_fused'

    def __init__(self, hf_config, config, *, device):
        from sparseengine.kernels.triton.palu import rotate_query
        self.rotate_query = rotate_query
        self.hf_config, self.config, self.device = hf_config, config, device
        self.group_size = config.palu_manifest['group_size']
        self._ops = {}
        self._closed = False
        for rk, rv in config.palu_manifest['ranks']:
            if (rk, rv) in self._ops:
                continue
            spec = PaluSpec(hf_config.num_attention_heads, hf_config.num_key_value_heads,
                            config.palu_manifest['model']['head_dim'], self.group_size,
                            rk, rv, hf_config.dtype, config.max_decoding_seqs, config.max_model_len)
            caps = platforms.current_platform.get_device_caps(int(device.index or 0))
            self._ops[rk, rv] = tuple(OpResolver(reg).resolve(spec, caps, op_spec=spec, device=device).provider
                                     for reg in (PREFILL, DECODE))

    def phase_ops(self, rk, rv):
        return self._ops[rk, rv]

    def bind(self, model):
        from sparseengine.models.palu import PaluSelfAttention
        for layer, (rk, rv) in zip(model.layers, self.config.palu_manifest['ranks'], strict=True):
            layer.self_attn = PaluSelfAttention(layer.self_attn, self, rk, rv)
        return len(model.layers)

    def prepare_weights(self, model):
        from safetensors.torch import load_file
        factors = load_file(str(Path(self.config.palu_checkpoint_path) / 'palu.safetensors'))
        for i, layer in enumerate(model.layers):
            layer.self_attn.prepare_weights(factors, i, self.config.palu_manifest['source_weight_sha256'])

    def binding_metadata(self):
        return {'implementation_kind': 'composite_provider', 'semantic_operator': 'palu_attention',
                'prefill_provider': self.prefill_name, 'decode_provider': self.decode_name}

    def close(self):
        self._ops.clear()
        self._closed = True
