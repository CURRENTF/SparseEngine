from __future__ import annotations

from dataclasses import dataclass
import importlib.util

import torch

import sparseengine.platforms as platforms
from sparseengine.engine.cache_manager.base import PackedSharedKVPayload, SharedKVPayload
from sparseengine.operators.registry import (
    OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult,
)
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum


@dataclass(frozen=True)
class SharedKVAttentionOpSpec:
    num_heads: int
    head_dim: int
    activation_dtype: torch.dtype
    page_size: int
    selection_capacity: int
    softmax_scale: float
    cuda_graph: bool

    def __post_init__(self):
        if self.num_heads <= 0 or self.head_dim <= 0 or self.selection_capacity <= 0:
            raise ValueError("Shared KV attention dimensions must be positive")
        if self.page_size <= 0 or self.page_size & (self.page_size - 1):
            raise ValueError("Shared KV page size must be a power of two")


@dataclass
class SharedKVDecodeState:
    """One prepared planner per batch graph; never shared across graph identities."""

    batch_capacity: int
    scheduler: object

    def keepalive_tensors(self):
        return [t for t in (self.scheduler.tile_scheduler_metadata,
                           self.scheduler.num_splits) if isinstance(t, torch.Tensor)]


class SharedKVAttentionProvider:
    name = ""


SHARED_KV_ATTENTION_REGISTRY = OpRegistry(
    "shared KV sparse attention",
    portfolio=PortfolioPolicy(upstream_standard=("sgl_flashmla",)),
)


@SHARED_KV_ATTENTION_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class SGLSharedKVAttentionProvider(SharedKVAttentionProvider):
    name = "sgl_flashmla"

    @classmethod
    def supports(cls, spec: SharedKVAttentionOpSpec, caps: DeviceCaps):
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability not in {(9, 0), (10, 0)}:
            return SupportResult.unsupported("SGL sparse FlashMLA requires SM90 or SM100")
        if spec.activation_dtype != torch.bfloat16 or spec.head_dim != 512:
            return SupportResult.unsupported("SGL sparse FlashMLA requires BF16 queries with D512")
        if spec.num_heads not in {64, 128}:
            return SupportResult.unsupported("SGL sparse FlashMLA requires 64 or 128 query heads")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph")
        if importlib.util.find_spec("sgl_kernel") is None:
            return SupportResult.dependency_absent("install sglang-kernel 0.4.5")
        from sgl_kernel import flash_mla
        if flash_mla._flashmla_import_error is not None:
            return SupportResult.dependency_broken(str(flash_mla._flashmla_import_error))
        return SupportResult.yes()

    def __init__(self, *, op_spec: SharedKVAttentionOpSpec):
        from sgl_kernel.flash_mla import (
            flash_mla_sparse_fwd, flash_mla_with_kvcache, get_mla_metadata,
        )
        self.spec = op_spec
        self._prefill = flash_mla_sparse_fwd
        self._decode = flash_mla_with_kvcache
        self._metadata = get_mla_metadata

    def create_decode_state(self, batch_capacity: int):
        if batch_capacity <= 0:
            raise ValueError("Decode batch capacity must be positive")
        scheduler, splits = self._metadata()
        if splits is not None:
            raise RuntimeError("SGL FlashMLA API requires a shape-bound scheduler")
        return SharedKVDecodeState(batch_capacity, scheduler)

    def prepare_decode_state(self, state):
        # The planner depends on dynamic topk lengths. Eager steps rebuild it;
        # capture must include it so replay updates its stable device buffers.
        # Call only outside capture, before an eager step or initial capture.
        state.scheduler.have_initialized = False
        state.scheduler.config = None
        state.scheduler.tile_scheduler_metadata = None
        state.scheduler.num_splits = None

    def prefill(self, query, payload: SharedKVPayload, indices, lengths, sink):
        shared_kv = payload.values
        if query.ndim != 3 or query.shape[1:] != (self.spec.num_heads, self.spec.head_dim):
            raise ValueError("Shared KV prefill query differs from its bound shape")
        if shared_kv.ndim != 3 or shared_kv.shape[1:] != (1, self.spec.head_dim):
            raise ValueError("Shared KV prefill payload must be [slots, 1, head_dim]")
        if shared_kv.dtype != self.spec.activation_dtype or shared_kv.device != query.device:
            raise ValueError("Shared KV prefill payload dtype/device differs from query")
        self._validate_selection(query, indices, lengths, sink, bound_capacity=False)
        # SM90 sparse prefill uses two 64-key tiles per iteration. Decode
        # accepts 64-key alignment; adapt the provider's prefill layout here.
        padding = (-indices.shape[-1])%128
        if padding:
            indices = torch.nn.functional.pad(indices, (0, padding), value=-1)
        return self._prefill(
            query, shared_kv, indices, self.spec.softmax_scale,
            d_v=self.spec.head_dim, attn_sink=sink, topk_length=lengths,
        )[0]

    def decode(self, query, payload: PackedSharedKVPayload, indices, lengths, sink, state):
        packed_pages = payload.pages
        if payload.page_size != self.spec.page_size:
            raise ValueError("Packed KV page size differs from the bound contract")
        if query.shape != (state.batch_capacity, self.spec.num_heads, self.spec.head_dim):
            raise ValueError("Shared KV decode query differs from its prepared batch shape")
        if indices.shape != (state.batch_capacity, 1, self.spec.selection_capacity):
            raise ValueError("Shared KV decode indices differ from their bound capacity")
        if packed_pages.ndim != 4 or packed_pages.shape[1:] != (self.spec.page_size, 1, 584):
            raise ValueError("Packed KV must have shape [pages, page_size, 1, 584]")
        if packed_pages.dtype != torch.uint8 or packed_pages.device != query.device:
            raise ValueError("Packed KV must be UINT8 on the query device")
        page_bytes = ((584 * self.spec.page_size + 575) // 576) * 576
        if packed_pages.stride() != (page_bytes, 584, 584, 1):
            raise ValueError("Packed KV requires contiguous logical pages with aligned page tails")
        self._validate_selection(query, indices, lengths, sink)
        return self._decode(
            query.unsqueeze(1), packed_pages, None, None,
            head_dim_v=self.spec.head_dim, tile_scheduler_metadata=state.scheduler,
            softmax_scale=self.spec.softmax_scale, is_fp8_kvcache=True,
            indices=indices, attn_sink=sink, topk_length=lengths,
        )[0].squeeze(1)

    def _validate_selection(self, query, indices, lengths, sink, *, bound_capacity=True):
        rows = query.shape[0]
        if not query.is_cuda or query.dtype != self.spec.activation_dtype:
            raise ValueError("Shared KV attention requires bound CUDA query dtype")
        if (indices.ndim != 3 or indices.shape[:2] != (rows, 1)
                or indices.dtype != torch.int32 or indices.shape[-1] <= 0
                or indices.shape[-1]%64
                or (bound_capacity and indices.shape[-1] != self.spec.selection_capacity)):
            raise ValueError("Selected indices must be INT32 [rows, 1, selection_capacity]")
        if lengths.shape != (rows,) or lengths.dtype != torch.int32:
            raise ValueError("Selected lengths must be INT32 [rows]")
        if sink.shape != (self.spec.num_heads,) or sink.dtype != torch.float32:
            raise ValueError("Attention sink must be FP32 [heads]")
        if any(t.device != query.device for t in (indices, lengths, sink)):
            raise ValueError("Attention selection and sink must share the query device")
        if not indices.is_contiguous() or not lengths.is_contiguous() or not sink.is_contiguous():
            raise ValueError("Attention selection and sink must be contiguous")


def resolve_shared_kv_attention_provider(spec: SharedKVAttentionOpSpec, *, device_index: int):
    caps = platforms.current_platform.get_device_caps(device_index)
    return OpResolver(SHARED_KV_ATTENTION_REGISTRY).resolve(
        spec, caps, op_spec=spec,
    ).provider
