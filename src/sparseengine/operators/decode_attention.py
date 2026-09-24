from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import torch

import sparseengine.platforms as platforms
from sparseengine.kernels.external.flashinfer.decode import (
    flashinfer_paged_decode_support,
)
from sparseengine.kernels.external.sgl.fa3 import (
    SglFa3DecodeKernel,
    sgl_fa3_device_support,
)
from sparseengine.operators.attention_capabilities import (
    AttentionKernelCapabilities,
    AttentionKernelRequest,
    AttentionScoreKind,
    match_attention_capabilities,
)
from sparseengine.operators.flashinfer_decode_state import (
    FlashInferPagedDecodeGraphState as _FlashInferPagedDecodeGraphState,
    FlashInferPagedDecodeState as _FlashInferPagedDecodeState,
)
from sparseengine.operators.registry import (
    OpRegistry,
    OpResolver,
    PortfolioPolicy,
    ProfileMatch,
    ProviderRole,
    SupportResult,
)
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum
from sparseengine.utils.context import get_context
from sparseengine.utils.device_name import device_name_contains
from sparseengine.utils.log import logger


def get_decode_workspace(
    context,
    batch_size: int,
    num_heads: int,
    num_blocks: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape_o = (batch_size, num_heads, num_blocks, head_dim)
    shape_lse = (batch_size, num_heads, num_blocks)
    mid_o = context.decode_mid_o
    if (
        mid_o is None
        or mid_o.device != device
        or mid_o.shape[0] < batch_size
        or mid_o.shape[1] < num_heads
        or mid_o.shape[2] < num_blocks
        or mid_o.shape[3] < head_dim
    ):
        mid_o = torch.empty(shape_o, dtype=torch.float32, device=device)
        context.decode_mid_o = mid_o

    mid_lse = context.decode_mid_o_logexpsum
    if (
        mid_lse is None
        or mid_lse.device != device
        or mid_lse.shape[0] < batch_size
        or mid_lse.shape[1] < num_heads
        or mid_lse.shape[2] < num_blocks
    ):
        mid_lse = torch.empty(shape_lse, dtype=torch.float32, device=device)
        context.decode_mid_o_logexpsum = mid_lse

    return (
        mid_o[:batch_size, :num_heads, :num_blocks, :head_dim],
        mid_lse[:batch_size, :num_heads, :num_blocks],
    )


@dataclass(frozen=True)
class DecodeAttentionOpSpec:
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    activation_dtype: torch.dtype
    softmax_scale: float
    max_batch_size: int
    causal: bool = True
    page_size: int = 1
    may_require_attention_scores: bool = False
    layer_varying_page_table: bool = False
    cuda_graph: bool = True
    h2o_layerwise_probability_scores: bool = False
    h2o_headwise_logits: bool = False
    context_capacity: int | None = None
    sparse_context_budget: int | None = None
    may_use_full_layer_kivi_int4: bool = False
    full_layer_kivi_decode_block_seq: int = 256
    full_layer_kivi_decode_block_n: int = 16
    full_layer_kivi_decode_num_warps: int = 2
    full_layer_kivi_decode_num_stages: int = 3
    kv_storage_format: str = "dense"

    def __post_init__(self) -> None:
        if self.kv_storage_format not in {"dense", "kivi", "turboquant", "fp8_kv"}:
            raise ValueError(f"Unknown decode KV storage format {self.kv_storage_format!r}.")
        if self.num_query_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("Decode attention head counts must be positive.")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("Decode query heads must be divisible by KV heads.")
        if self.head_dim <= 0 or self.page_size <= 0 or self.max_batch_size <= 0:
            raise ValueError("Decode attention dimensions and capacity must be positive.")
        if self.softmax_scale <= 0:
            raise ValueError("Decode attention softmax_scale must be positive.")
        if (
            (self.h2o_layerwise_probability_scores or self.h2o_headwise_logits)
            and not self.may_require_attention_scores
        ):
            raise ValueError(
                "H2O layer-wise probability scoring requires decode score output."
            )
        if self.h2o_layerwise_probability_scores and self.h2o_headwise_logits:
            raise ValueError("Decode cannot request both probabilities and raw H2O logits.")
        if self.context_capacity is not None and self.context_capacity <= 0:
            raise ValueError("Decode attention context_capacity must be positive.")
        if (
            self.sparse_context_budget is not None
            and self.sparse_context_budget <= 0
        ):
            raise ValueError("Decode sparse_context_budget must be positive.")
        if self.may_use_full_layer_kivi_int4 and not self.layer_varying_page_table:
            raise ValueError(
                "Full-layer KIVI decode requires a layer-varying KV view contract."
            )
        if (
            self.may_use_full_layer_kivi_int4
            and (
                self.full_layer_kivi_decode_block_seq <= 0
                or self.full_layer_kivi_decode_block_seq % 16
            )
        ):
            raise ValueError(
                "Full-layer KIVI decode block_seq must be a positive multiple "
                f"of 16, got {self.full_layer_kivi_decode_block_seq}."
            )
        if self.may_use_full_layer_kivi_int4 and (
            self.full_layer_kivi_decode_block_n <= 0
            or self.full_layer_kivi_decode_block_n % 16
            or self.full_layer_kivi_decode_block_seq
            % self.full_layer_kivi_decode_block_n
        ):
            raise ValueError(
                "Full-layer KIVI decode block_n must be a positive multiple of "
                "16 and divide block_seq, got "
                f"block_n={self.full_layer_kivi_decode_block_n}, "
                f"block_seq={self.full_layer_kivi_decode_block_seq}."
            )
        if (
            self.may_use_full_layer_kivi_int4
            and self.full_layer_kivi_decode_num_warps not in {1, 2, 4, 8}
        ):
            raise ValueError(
                "Full-layer KIVI decode num_warps must be one of 1, 2, 4, "
                f"or 8, got {self.full_layer_kivi_decode_num_warps}."
            )
        if (
            self.may_use_full_layer_kivi_int4
            and self.full_layer_kivi_decode_num_stages <= 0
        ):
            raise ValueError(
                "Full-layer KIVI decode num_stages must be positive, got "
                f"{self.full_layer_kivi_decode_num_stages}."
            )

    @property
    def kernel_request(self) -> AttentionKernelRequest:
        return AttentionKernelRequest(
            activation_dtype=self.activation_dtype,
            head_dim=self.head_dim,
            page_size=self.page_size,
            score_output=(
                AttentionScoreKind.RAW_QK_PER_HEAD
                if self.may_require_attention_scores
                and not self.h2o_layerwise_probability_scores
                else AttentionScoreKind.NONE
            ),
            requires_softmax_lse=(self.h2o_layerwise_probability_scores or self.h2o_headwise_logits),
            layer_varying_page_table=self.layer_varying_page_table,
            varlen=True,
            cuda_graph=self.cuda_graph,
        )


class DecodeAttentionProvider:
    name = ""
    capabilities: AttentionKernelCapabilities
    decode_graph_lifecycle = False

    def prepare(
        self,
        spec: DecodeAttentionOpSpec,
        *,
        device_index: int | None = None,
    ) -> None:
        del spec, device_index

    def close(self) -> None:
        pass

    def run(
        self,
        spec: DecodeAttentionOpSpec,
        q: torch.Tensor,
        view: Any,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError


@dataclass(frozen=True)
class DecodeAttentionRunResult:
    output: torch.Tensor
    softmax_lse: torch.Tensor


@dataclass(frozen=True)
class GraphStableDecodeLaunchPlan:
    """Capture-time launch envelope for context-stable MHA/GQA decode."""

    plan_id: str
    context_capacity: int
    max_kv_splits: int
    target_tokens_per_split: int
    block_n: int
    stage1_num_warps: int
    stage1_num_stages: int
    stage2_num_warps: int
    stage2_num_stages: int

    def __post_init__(self) -> None:
        positive = (
            self.context_capacity,
            self.max_kv_splits,
            self.target_tokens_per_split,
            self.block_n,
            self.stage1_num_warps,
            self.stage1_num_stages,
            self.stage2_num_warps,
            self.stage2_num_stages,
        )
        if any(value <= 0 for value in positive):
            raise ValueError(f"Decode launch plan values must be positive: {self}.")

    def as_dict(self) -> dict[str, int | str]:
        return {
            "plan_id": self.plan_id,
            "context_capacity": self.context_capacity,
            "max_kv_splits": self.max_kv_splits,
            "target_tokens_per_split": self.target_tokens_per_split,
            "block_n": self.block_n,
            "stage1_num_warps": self.stage1_num_warps,
            "stage1_num_stages": self.stage1_num_stages,
            "stage2_num_warps": self.stage2_num_warps,
            "stage2_num_stages": self.stage2_num_stages,
        }


def build_graph_stable_decode_launch_plan(
    spec: DecodeAttentionOpSpec,
    caps: DeviceCaps,
) -> GraphStableDecodeLaunchPlan:
    """Resolve one context-invariant portable plan before provider preparation."""
    del caps
    if spec.context_capacity is None:
        raise ValueError(
            "context-stable decode requires a static context_capacity."
        )
    if spec.head_dim == 256:
        block_n, stage1_warps, stage2_warps = 128, 4, 8
    elif spec.head_dim in {64, 128}:
        block_n, stage1_warps, stage2_warps = 64, 2, 4
    else:
        raise ValueError(
            f"No context-stable decode launch plan for head_dim={spec.head_dim}."
        )

    # The grid is derived from the configured capacity, never the current
    # request length. Capping the envelope bounds workspace and empty programs;
    # each replay derives its effective split count from device context_lens.
    max_kv_splits = min(
        64,
        max(16, math.ceil(int(spec.context_capacity) / 4096)),
    )
    return GraphStableDecodeLaunchPlan(
        plan_id="portable_fixed_grid_v1",
        context_capacity=int(spec.context_capacity),
        max_kv_splits=max_kv_splits,
        target_tokens_per_split=256,
        block_n=block_n,
        stage1_num_warps=stage1_warps,
        stage1_num_stages=2,
        stage2_num_warps=stage2_warps,
        stage2_num_stages=2,
    )


def build_deltakv_kivi_decode_launch_plan(
    spec: DecodeAttentionOpSpec,
    caps: DeviceCaps,
) -> GraphStableDecodeLaunchPlan:
    """Resolve the fixed split envelope for packed full-layer KIVI decode."""
    base = build_graph_stable_decode_launch_plan(spec, caps)
    target_tokens_per_split = int(spec.full_layer_kivi_decode_block_seq)
    max_kv_splits = min(
        64,
        max(4, math.ceil(base.context_capacity / target_tokens_per_split)),
    )
    return replace(
        base,
        plan_id="deltakv_kivi_fixed_grid_v1",
        max_kv_splits=max_kv_splits,
        target_tokens_per_split=target_tokens_per_split,
        block_n=int(spec.full_layer_kivi_decode_block_n),
        stage1_num_warps=int(spec.full_layer_kivi_decode_num_warps),
        stage1_num_stages=int(spec.full_layer_kivi_decode_num_stages),
    )


DECODE_ATTENTION_REGISTRY: OpRegistry[
    DecodeAttentionOpSpec, DecodeAttentionProvider
] = OpRegistry(
    "paged decode attention",
    portfolio=PortfolioPolicy(
        upstream_standard=(
            "sgl_fa3_paged_decode_sm90",
            "flashinfer_paged_decode",
        ),
        repo_portable=(
            "triton_paged_decode",
            "triton_fixed_grid_paged_decode",
        ),
        repo_nonstandard=("triton_deltakv_fixed_grid_decode", "triton_quantized_pages_decode"),
    ),
)


@DECODE_ATTENTION_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class SglFa3PagedDecodeAttentionProvider(DecodeAttentionProvider):
    name = "sgl_fa3_paged_decode_sm90"
    supports_decode_graph = True
    capabilities = AttentionKernelCapabilities(
        platforms=frozenset({PlatformEnum.CUDA}),
        compute_capabilities=frozenset({(9, 0)}),
        activation_dtypes=frozenset({torch.bfloat16}),
        head_dims=frozenset({128, 256}),
        page_sizes=None,
        score_outputs=frozenset({AttentionScoreKind.NONE}),
        returns_softmax_lse=True,
        layer_varying_page_table=True,
        varlen=True,
        cuda_graph=True,
        minimum_runtime_version=(12, 3),
    )

    def __init__(self) -> None:
        self._kernel: SglFa3DecodeKernel | None = None

    @classmethod
    def supports(
        cls,
        spec: DecodeAttentionOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        if spec.kv_storage_format != "dense":
            return SupportResult.unsupported("requires dense KV storage")
        if spec.may_use_full_layer_kivi_int4:
            return SupportResult.unsupported(
                "does not support mixed dense and full-layer KIVI int4 storage"
            )
        common = match_attention_capabilities(
            spec.kernel_request,
            caps,
            cls.capabilities,
        )
        if not common.supported:
            return common
        if not spec.causal:
            return SupportResult.unsupported("requires causal attention")
        supported, reason = sgl_fa3_device_support(caps.device_index)
        return SupportResult.yes(reason) if supported else SupportResult.unsupported(reason)

    def prepare(
        self,
        spec: DecodeAttentionOpSpec,
        *,
        device_index: int | None = None,
    ) -> None:
        current_device = torch.cuda.current_device()
        if device_index is None:
            device_index = current_device
        if int(device_index) != current_device:
            raise RuntimeError(
                "SGL FA3 decode must be prepared on the selected CUDA device: "
                f"selected={device_index} current={current_device}."
            )
        self._kernel = SglFa3DecodeKernel(
            device=torch.device("cuda", int(device_index)),
            max_batch_size=spec.max_batch_size,
            softmax_scale=spec.softmax_scale,
        )

    def close(self) -> None:
        self._kernel = None

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "implementation_source": "sglang-kernel",
            "kernel_path": "sgl_kernel.fa3.fwd",
            "page_sizes": "any positive native page size",
        }

    def run(
        self,
        spec: DecodeAttentionOpSpec,
        q: torch.Tensor,
        view: Any,
        **kwargs,
    ) -> torch.Tensor:
        return_softmax_lse = spec.kernel_request.requires_softmax_lse
        if view.meta.attn_score is not None and not return_softmax_lse:
            raise RuntimeError("SGL FA3 decode does not produce attention scores.")
        result = self._run_sgl(
            spec,
            q,
            view,
            return_softmax_lse=return_softmax_lse,
            **kwargs,
        )
        if not return_softmax_lse:
            return result
        if not isinstance(result, tuple):
            raise RuntimeError("SGL FA3 decode did not return the requested softmax LSE.")
        output, softmax_lse = result
        return DecodeAttentionRunResult(output=output, softmax_lse=softmax_lse)

    def _run_sgl(
        self,
        spec: DecodeAttentionOpSpec,
        q: torch.Tensor,
        view: Any,
        return_softmax_lse: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        kwargs.pop("decode_launch_op", None)
        if kwargs:
            raise TypeError(
                "SGL FA3 decode received unsupported runtime arguments: "
                f"{sorted(kwargs)}."
            )
        if self._kernel is None:
            raise RuntimeError("SGL FA3 decode provider was not prepared.")
        payload = view.payload
        meta = view.meta
        if q.dtype != spec.activation_dtype:
            raise TypeError(
                f"SGL FA3 decode expected {spec.activation_dtype} Q, got {q.dtype}."
            )
        if payload.k_cache.dtype != q.dtype or payload.v_cache.dtype != q.dtype:
            raise TypeError(
                "SGL FA3 decode requires Q/K/V with the same dtype, got "
                f"{q.dtype}/{payload.k_cache.dtype}/{payload.v_cache.dtype}."
            )
        if meta.active_slots.dtype != torch.int32 or meta.active_slots.ndim != 2:
            raise TypeError(
                "SGL FA3 decode requires a rank-2 int32 physical-slot page table."
            )
        if meta.req_indices.dtype != torch.int32:
            raise TypeError("SGL FA3 decode requires int32 request indices.")
        if meta.context_lens.dtype != torch.int32:
            raise TypeError("SGL FA3 decode requires int32 context lengths.")
        page_size = int(spec.page_size)
        if int(payload.k_cache.shape[0]) % page_size:
            raise ValueError(
                "SGL FA3 KV slot capacity must be divisible by page_size: "
                f"slots={int(payload.k_cache.shape[0])} page_size={page_size}."
            )
        k_cache = payload.k_cache.view(
            -1,
            page_size,
            int(payload.k_cache.shape[1]),
            int(payload.k_cache.shape[2]),
        )
        v_cache = payload.v_cache.view_as(k_cache)
        output = torch.empty_like(q)
        return self._kernel.run_explicit(
            q,
            k_cache,
            v_cache,
            meta.active_slots,
            meta.req_indices,
            meta.context_lens,
            output,
            validation_scope=get_context().attention_validation_scope,
            return_softmax_lse=return_softmax_lse,
        )


@DECODE_ATTENTION_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class FlashInferPagedDecodeAttentionProvider(DecodeAttentionProvider):
    name = "flashinfer_paged_decode"
    decode_graph_lifecycle = True
    supports_decode_graph = True
    capabilities = AttentionKernelCapabilities(
        platforms=frozenset({PlatformEnum.CUDA}),
        activation_dtypes=frozenset({torch.bfloat16, torch.float16}),
        page_sizes=None,
        score_outputs=frozenset({AttentionScoreKind.NONE}),
        returns_softmax_lse=True,
        layer_varying_page_table=True,
        varlen=True,
        cuda_graph=True,
    )

    def __init__(self) -> None:
        self._state: _FlashInferPagedDecodeState | None = None

    @classmethod
    def supports(
        cls,
        spec: DecodeAttentionOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        if spec.kv_storage_format != "dense":
            return SupportResult.unsupported("requires dense KV storage")
        if spec.may_use_full_layer_kivi_int4:
            return SupportResult.unsupported(
                "does not support mixed dense and full-layer KIVI int4 storage"
            )
        common = match_attention_capabilities(
            spec.kernel_request,
            caps,
            cls.capabilities,
        )
        if not common.supported:
            return common
        if not spec.causal:
            return SupportResult.unsupported("requires causal attention")
        supported, reason = flashinfer_paged_decode_support()
        return SupportResult.yes(reason) if supported else SupportResult.unsupported(reason)

    def prepare(
        self,
        spec: DecodeAttentionOpSpec,
        *,
        device_index: int | None = None,
    ) -> None:
        if self._state is not None:
            return
        current_device = torch.cuda.current_device()
        if device_index is None:
            device_index = current_device
        if int(device_index) != current_device:
            raise RuntimeError(
                "FlashInfer decode must be prepared on the selected CUDA device: "
                f"selected={device_index} current={current_device}."
            )
        if not spec.cuda_graph:
            self._state = _FlashInferPagedDecodeState(
                torch.device("cuda", int(device_index))
            )

    def close(self) -> None:
        self._state = None
        self._active_graph_state = None

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "implementation_source": "flashinfer-python",
            "kernel_path": "flashinfer.BatchDecodeWithPagedKVCacheWrapper",
            "kv_layout": "NHD",
            "page_sizes": "any positive native page size",
            "cuda_graph": True,
            "graph_metadata": "fixed buffers + graph-out plan + graph-in page packing",
        }

    def init_decode_graph_state(
        self,
        spec: DecodeAttentionOpSpec,
        contract,
        inputs,
    ) -> _FlashInferPagedDecodeGraphState:
        if not spec.cuda_graph:
            raise RuntimeError("FlashInfer graph state requires a CUDA Graph spec.")
        if contract.batch_capacity > spec.max_batch_size:
            raise ValueError(
                "FlashInfer graph batch exceeds the prepared operator capacity: "
                f"graph={contract.batch_capacity} operator={spec.max_batch_size}."
            )
        return _FlashInferPagedDecodeGraphState(
            spec,
            contract=contract,
            inputs=inputs,
        )

    def prepare_decode_graph_out(
        self,
        state: _FlashInferPagedDecodeGraphState,
    ) -> None:
        self._active_graph_state = state
        state.prepare_out_graph()

    def prepare_decode_graph_in(
        self,
        state: _FlashInferPagedDecodeGraphState,
    ) -> None:
        self._active_graph_state = state
        state.begin_graph_in()

    def decode_graph_keepalive_tensors(
        self,
        state: _FlashInferPagedDecodeGraphState,
    ) -> list[torch.Tensor]:
        return state.keepalive_tensors()

    def close_decode_graph_state(
        self,
        state: _FlashInferPagedDecodeGraphState,
    ) -> None:
        if getattr(self, "_active_graph_state", None) is state:
            self._active_graph_state = None

    def run(
        self,
        spec: DecodeAttentionOpSpec,
        q: torch.Tensor,
        view: Any,
        **kwargs,
    ) -> torch.Tensor | DecodeAttentionRunResult:
        kwargs.pop("decode_launch_op", None)
        if kwargs:
            raise TypeError(
                "FlashInfer decode received unsupported runtime arguments: "
                f"{sorted(kwargs)}."
            )
        graph_state = getattr(self, "_active_graph_state", None)
        if spec.cuda_graph:
            if not isinstance(graph_state, _FlashInferPagedDecodeGraphState):
                raise RuntimeError("FlashInfer graph decode state is not active.")
            state = graph_state
        else:
            if self._state is None:
                raise RuntimeError("FlashInfer decode provider was not prepared.")
            state = self._state
        payload = view.payload
        meta = view.meta
        if q.dtype != spec.activation_dtype:
            raise TypeError(
                f"FlashInfer decode expected {spec.activation_dtype} Q, got {q.dtype}."
            )
        if payload.k_cache.dtype != q.dtype or payload.v_cache.dtype != q.dtype:
            raise TypeError(
                "FlashInfer decode requires Q/K/V with the same dtype, got "
                f"{q.dtype}/{payload.k_cache.dtype}/{payload.v_cache.dtype}."
            )
        wrapper = state.wrapper
        if spec.cuda_graph:
            is_sparse = bool(getattr(meta, "is_sparse", False))
            state.pack_page_indices_once(
                active_slots=meta.active_slots,
                req_indices=meta.req_indices,
                context_lens=meta.context_lens,
                is_sparse=is_sparse,
                force=bool(spec.layer_varying_page_table),
            )
            wrapper = state.wrapper_for(is_sparse)
        else:
            max_context_len = getattr(meta, "max_context_len", None)
            if max_context_len is None:
                raise RuntimeError(
                    "FlashInfer decode requires host-side max_context_len metadata."
                )
            context = get_context()
            plan_key = (
                context.attention_validation_scope,
                meta.active_slots.data_ptr(),
                meta.req_indices.data_ptr(),
                meta.context_lens.data_ptr(),
                int(max_context_len),
            )
            if (
                spec.layer_varying_page_table
                or getattr(state, "plan_key", None) != plan_key
            ):
                state.plan(
                    spec,
                    active_slots=meta.active_slots,
                    req_indices=meta.req_indices,
                    context_lens=meta.context_lens,
                    max_context_len=int(max_context_len),
                )
                state.plan_key = plan_key
        output = torch.empty_like(q)
        return_softmax_lse = spec.kernel_request.requires_softmax_lse
        page_size = int(spec.page_size)
        if int(payload.k_cache.shape[0]) % page_size:
            raise ValueError(
                "FlashInfer KV slot capacity must be divisible by page_size: "
                f"slots={int(payload.k_cache.shape[0])} page_size={page_size}."
            )
        paged_k_cache = payload.k_cache.view(
            -1,
            page_size,
            int(payload.k_cache.shape[1]),
            int(payload.k_cache.shape[2]),
        )
        paged_v_cache = payload.v_cache.view_as(paged_k_cache)
        result = wrapper.run(
            q,
            (
                paged_k_cache,
                paged_v_cache,
            ),
            out=output,
            return_lse=return_softmax_lse,
        )
        if not return_softmax_lse:
            if not isinstance(result, torch.Tensor) or (
                result.data_ptr() != output.data_ptr()
            ):
                raise RuntimeError(
                    "FlashInfer decode did not write to the supplied output."
                )
            return output
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError(
                "FlashInfer decode did not return the requested softmax LSE."
            )
        returned_output, softmax_lse_log2 = result
        if returned_output.data_ptr() != output.data_ptr():
            raise RuntimeError("FlashInfer decode did not write to the supplied output.")
        expected_shape = (int(q.shape[0]), spec.num_query_heads)
        if (
            softmax_lse_log2.dtype != torch.float32
            or tuple(softmax_lse_log2.shape) != expected_shape
        ):
            raise RuntimeError(
                "FlashInfer decode returned an unexpected softmax LSE: "
                f"shape={tuple(softmax_lse_log2.shape)} "
                f"dtype={softmax_lse_log2.dtype} "
                f"expected={expected_shape}/torch.float32."
            )
        softmax_lse = softmax_lse_log2.mul(math.log(2.0)).transpose(0, 1)
        return DecodeAttentionRunResult(output=output, softmax_lse=softmax_lse)


@DECODE_ATTENTION_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class TritonPagedDecodeAttentionProvider(DecodeAttentionProvider):
    name = "triton_paged_decode"
    capabilities = AttentionKernelCapabilities(
        platforms=frozenset({PlatformEnum.CUDA}),
        activation_dtypes=frozenset(
            {torch.bfloat16, torch.float16, torch.float32}
        ),
        page_sizes=frozenset({1}),
        score_outputs=frozenset(
            {
                AttentionScoreKind.NONE,
                AttentionScoreKind.RAW_QK_PER_HEAD,
                AttentionScoreKind.RAW_QK_REDUCED,
            }
        ),
        layer_varying_page_table=True,
        varlen=True,
        cuda_graph=True,
        requires_triton=True,
    )

    def __init__(self) -> None:
        self._backend = None

    @classmethod
    def supports(
        cls,
        spec: DecodeAttentionOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        if spec.cuda_graph:
            return SupportResult.unsupported("split count depends on context length")
        if spec.h2o_headwise_logits:
            return SupportResult.unsupported("H2O logits use the shared fixed-grid eager/graph path")
        if spec.kv_storage_format != "dense":
            return SupportResult.unsupported("requires dense KV storage")
        return match_attention_capabilities(
            spec.kernel_request,
            caps,
            cls.capabilities,
        )

    def prepare(
        self,
        spec: DecodeAttentionOpSpec,
        *,
        device_index: int | None = None,
    ) -> None:
        del spec, device_index
        from sparseengine.layers.attention_backend import TritonAttentionBackend

        self._backend = TritonAttentionBackend()

    def close(self) -> None:
        self._backend = None

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "implementation_source": "repo_triton",
            "kernel_path": "triton_flash_decode_stage1_stage2",
        }

    def run(
        self,
        spec: DecodeAttentionOpSpec,
        q: torch.Tensor,
        view: Any,
        **kwargs,
    ) -> torch.Tensor:
        if self._backend is None:
            raise RuntimeError("Triton decode provider was not prepared.")
        decode_launch_op = kwargs.pop("decode_launch_op", None)
        if kwargs:
            raise TypeError(
                "Triton decode received unsupported runtime arguments: "
                f"{sorted(kwargs)}."
            )

        context = get_context()
        cache_manager = context.cache_manager
        layer_idx = int(context.now_layer_idx)
        meta = view.meta
        max_context_len = meta.max_context_len
        static_cap = getattr(cache_manager, "_decode_static_max_context_len", None)
        if static_cap is not None and spec.cuda_graph:
            max_context_len = max(
                int(max_context_len) if max_context_len is not None else 0,
                int(static_cap),
            )
        if max_context_len is None:
            raise RuntimeError(
                "static decode requires max_context_len, got None at "
                f"layer={layer_idx}"
            )
        max_len_in_batch = int(max_context_len)
        if meta.active_slots.dim() == 2:
            slot_table_len = int(meta.active_slots.shape[1])
            if max_len_in_batch > slot_table_len:
                max_len_in_batch = slot_table_len
            if max_len_in_batch <= 0:
                raise RuntimeError(
                    "decode requires a positive context length, got "
                    f"{max_len_in_batch} at layer={layer_idx}"
                )

        block_seq = cache_manager.get_decode_block_seq(layer_idx, 256)
        if decode_launch_op is None:
            gqa_block_n, gqa_num_warps = 16, 2
        else:
            block_seq, gqa_block_n, gqa_num_warps = (
                decode_launch_op.launch_config(
                    block_seq=block_seq,
                    max_context_len=max_len_in_batch,
                    requires_attention_scores=meta.attn_score is not None,
                )
            )
        num_seq_blocks = (max_len_in_batch + block_seq - 1) // block_seq
        mid_o, mid_o_logexpsum = get_decode_workspace(
            context,
            int(q.shape[0]),
            spec.num_query_heads,
            num_seq_blocks,
            spec.head_dim,
            q.device,
        )
        return self._backend.run_decode(
            q,
            view,
            mid_o=mid_o,
            mid_o_logexpsum=mid_o_logexpsum,
            max_len_in_batch=max_len_in_batch,
            block_seq=block_seq,
            num_heads=spec.num_query_heads,
            num_kv_heads=spec.num_kv_heads,
            gqa_block_n=gqa_block_n,
            gqa_num_warps=gqa_num_warps,
        )


@DECODE_ATTENTION_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class FixedGridTritonPagedDecodeAttentionProvider(DecodeAttentionProvider):
    """Fixed-grid Triton MHA/GQA decode provider for CUDA Graph."""

    name = "triton_fixed_grid_paged_decode"
    supports_decode_graph = True
    capabilities = replace(
        TritonPagedDecodeAttentionProvider.capabilities,
        activation_dtypes=frozenset({torch.bfloat16, torch.float16}),
        head_dims=frozenset({64, 128, 256}),
        returns_softmax_lse=True,
    )

    def __init__(
        self,
        *,
        launch_plan: GraphStableDecodeLaunchPlan,
    ) -> None:
        self.launch_plan = launch_plan
        self._mid_o: torch.Tensor | None = None
        self._mid_lse: torch.Tensor | None = None
        self._softmax_lse: torch.Tensor | None = None
        self._headwise_logits: torch.Tensor | None = None

    @classmethod
    def bind(
        cls,
        spec: DecodeAttentionOpSpec,
        caps: DeviceCaps,
        **provider_kwargs,
    ) -> FixedGridTritonPagedDecodeAttentionProvider:
        if provider_kwargs:
            raise TypeError(
                "Fixed-grid Triton decode does not accept provider "
                f"arguments: {sorted(provider_kwargs)}."
            )
        return cls(launch_plan=build_graph_stable_decode_launch_plan(spec, caps))

    @classmethod
    def supports(
        cls, spec: DecodeAttentionOpSpec, caps: DeviceCaps
    ) -> SupportResult:
        if spec.kv_storage_format != "dense":
            return SupportResult.unsupported("requires dense KV storage")
        if not spec.cuda_graph and not spec.h2o_headwise_logits:
            return SupportResult.unsupported("reserved for CUDA Graph or H2O headwise logits")
        if spec.may_use_full_layer_kivi_int4:
            return SupportResult.unsupported(
                "full-layer KIVI int4 requires the DeltaKV fixed-grid provider"
            )
        if spec.context_capacity is None:
            return SupportResult.unsupported("requires a static context capacity")
        return match_attention_capabilities(
            spec.kernel_request,
            caps,
            cls.capabilities,
        )

    def prepare(
        self,
        spec: DecodeAttentionOpSpec,
        *,
        device_index: int | None = None,
    ) -> None:
        if self.launch_plan.context_capacity != spec.context_capacity:
            raise RuntimeError(
                "Fixed-grid decode launch plan does not match the operator "
                f"capacity: plan={self.launch_plan.context_capacity} "
                f"spec={spec.context_capacity}."
            )
        if device_index is None:
            device_index = torch.cuda.current_device()
        device = torch.device("cuda", int(device_index))
        self._mid_o = torch.empty(
            (
                spec.max_batch_size,
                spec.num_query_heads,
                self.launch_plan.max_kv_splits,
                spec.head_dim,
            ),
            dtype=torch.float32,
            device=device,
        )
        self._mid_lse = torch.empty(
            (
                spec.max_batch_size,
                spec.num_query_heads,
                self.launch_plan.max_kv_splits,
            ),
            dtype=torch.float32,
            device=device,
        )
        self._softmax_lse = torch.empty(
            (spec.num_query_heads, spec.max_batch_size),
            dtype=torch.float32,
            device=device,
        )
        if spec.h2o_headwise_logits:
            # One prepared provider serves sequential layers and graph replays.
            # Consume each layer's raw logits before this shared storage is reused.
            self._headwise_logits = torch.empty(
                (spec.max_batch_size, spec.num_query_heads, spec.context_capacity),
                dtype=torch.float32, device=device,
            )

    def close(self) -> None:
        self._mid_o = None
        self._mid_lse = None
        self._softmax_lse = None
        self._headwise_logits = None

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "implementation_source": "repo_triton",
            "kernel_path": "paged_flash_decode",
            "cuda_graph_mode": "batch_indexed",
            "launch_plan": self.launch_plan.as_dict(),
            "workspace_owner": "provider",
            "headwise_logits_workspace_shape": (
                list(self._headwise_logits.shape) if self._headwise_logits is not None else None
            ),
            "headwise_logits_workspace_bytes": (
                self._headwise_logits.numel() * self._headwise_logits.element_size()
                if self._headwise_logits is not None else 0
            ),
        }

    def run(
        self,
        spec: DecodeAttentionOpSpec,
        q: torch.Tensor,
        view: Any,
        **kwargs,
    ) -> torch.Tensor:
        kwargs.pop("decode_launch_op", None)
        if kwargs:
            raise TypeError(
                "Fixed-grid decode received unsupported arguments: "
                f"{sorted(kwargs)}."
            )
        if (
            self._mid_o is None
            or self._mid_lse is None
            or self._softmax_lse is None
        ):
            raise RuntimeError("Fixed-grid decode provider was not prepared.")
        payload = view.payload
        if getattr(payload, "backend", None) != "dense":
            raise RuntimeError(
                "Fixed-grid decode requires dense explicit KV storage."
            )
        batch_size = int(q.shape[0])
        score = view.meta.attn_score
        raw_logits = None
        if spec.h2o_headwise_logits:
            if (
                self._headwise_logits is None or score is None or score.ndim != 2
                or score.shape[0] != batch_size
                or batch_size > self._headwise_logits.shape[0]
                or not 0 < score.shape[1] <= self._headwise_logits.shape[2]
            ):
                raise ValueError("H2O reduced score exceeds the prepared shared logits workspace.")
            raw_logits = self._headwise_logits[:batch_size, :, :score.shape[1]]
        from sparseengine.kernels.triton.paged_flash_decoding import (
            paged_flash_decode,
        )

        result = paged_flash_decode(
            q,
            payload.k_cache,
            payload.v_cache,
            view.meta.active_slots,
            view.meta.req_indices,
            view.meta.context_lens,
            self._mid_o[:batch_size],
            self._mid_lse[:batch_size],
            attn_score=(raw_logits if spec.h2o_headwise_logits else
                        (None if spec.h2o_layerwise_probability_scores else score)),
            softmax_scale=spec.softmax_scale,
            target_tokens_per_split=self.launch_plan.target_tokens_per_split,
            block_n=self.launch_plan.block_n,
            num_warps=self.launch_plan.stage1_num_warps,
            num_stages=self.launch_plan.stage1_num_stages,
            stage2_num_warps=self.launch_plan.stage2_num_warps,
            stage2_num_stages=self.launch_plan.stage2_num_stages,
            return_softmax_lse=(spec.h2o_layerwise_probability_scores or spec.h2o_headwise_logits),
            output_lse=self._softmax_lse[:, :batch_size],
        )
        if spec.h2o_headwise_logits:
            from sparseengine.kernels.triton.h2o_decode_score import h2o_headwise_probability_from_lse

            if not isinstance(result, tuple):
                raise RuntimeError("Shared-logits H2O decode did not return softmax LSE.")
            h2o_headwise_probability_from_lse(
                raw_logits, result[1], view.meta.context_lens, score,
                softmax_scale=spec.softmax_scale,
            )
            return result[0]
        if not spec.h2o_layerwise_probability_scores:
            return result
        if not isinstance(result, tuple):
            raise RuntimeError("Fixed-grid decode did not return softmax LSE.")
        return DecodeAttentionRunResult(output=result[0], softmax_lse=result[1])


@dataclass
class _DeltaKVFixedGridDecodeState:
    batch_capacity: int
    launch_plan: GraphStableDecodeLaunchPlan
    kivi_launch_plan: GraphStableDecodeLaunchPlan
    mid_o: torch.Tensor
    mid_lse: torch.Tensor
    kivi_mid_o: torch.Tensor
    kivi_mid_lse: torch.Tensor
    output: torch.Tensor
    output_lse: torch.Tensor

    @classmethod
    def allocate(
        cls,
        spec: DecodeAttentionOpSpec,
        caps: DeviceCaps,
        *,
        batch_capacity: int,
        context_capacity: int,
        device: torch.device,
    ) -> _DeltaKVFixedGridDecodeState:
        graph_spec = replace(
            spec,
            max_batch_size=int(batch_capacity),
            context_capacity=int(context_capacity),
        )
        launch_plan = build_graph_stable_decode_launch_plan(graph_spec, caps)
        kivi_launch_plan = build_deltakv_kivi_decode_launch_plan(graph_spec, caps)
        return cls(
            batch_capacity=int(batch_capacity),
            launch_plan=launch_plan,
            kivi_launch_plan=kivi_launch_plan,
            mid_o=torch.empty(
                (
                    batch_capacity,
                    spec.num_query_heads,
                    launch_plan.max_kv_splits,
                    spec.head_dim,
                ),
                dtype=torch.float32,
                device=device,
            ),
            mid_lse=torch.empty(
                (
                    batch_capacity,
                    spec.num_query_heads,
                    launch_plan.max_kv_splits,
                ),
                dtype=torch.float32,
                device=device,
            ),
            kivi_mid_o=torch.empty(
                (
                    batch_capacity,
                    spec.num_query_heads,
                    kivi_launch_plan.max_kv_splits,
                    spec.head_dim,
                ),
                dtype=torch.float32,
                device=device,
            ),
            kivi_mid_lse=torch.empty(
                (
                    batch_capacity,
                    spec.num_query_heads,
                    kivi_launch_plan.max_kv_splits,
                ),
                dtype=torch.float32,
                device=device,
            ),
            output=torch.empty(
                (batch_capacity, spec.num_query_heads, spec.head_dim),
                dtype=spec.activation_dtype,
                device=device,
            ),
            output_lse=torch.empty(
                (spec.num_query_heads, batch_capacity),
                dtype=torch.float32,
                device=device,
            ),
        )

    def keepalive_tensors(self) -> list[torch.Tensor]:
        return [
            self.mid_o,
            self.mid_lse,
            self.kivi_mid_o,
            self.kivi_mid_lse,
            self.output,
            self.output_lse,
        ]


@DECODE_ATTENTION_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class DeltaKVFixedGridDecodeAttentionProvider(DecodeAttentionProvider):
    """Graph-stable provider for DeltaKV dense and full-layer KIVI views."""

    name = "triton_deltakv_fixed_grid_decode"
    supports_decode_graph = True
    decode_graph_lifecycle = True
    capabilities = replace(
        FixedGridTritonPagedDecodeAttentionProvider.capabilities,
        head_dims=frozenset({64, 128}),
    )

    def __init__(
        self,
        *,
        caps: DeviceCaps,
        launch_plan: GraphStableDecodeLaunchPlan,
        kivi_launch_plan: GraphStableDecodeLaunchPlan,
    ) -> None:
        self._caps = caps
        self.launch_plan = launch_plan
        self.kivi_launch_plan = kivi_launch_plan
        self._active_graph_state: _DeltaKVFixedGridDecodeState | None = None

    @classmethod
    def bind(
        cls,
        spec: DecodeAttentionOpSpec,
        caps: DeviceCaps,
        **provider_kwargs,
    ) -> DeltaKVFixedGridDecodeAttentionProvider:
        if provider_kwargs:
            raise TypeError(
                "DeltaKV fixed-grid decode does not accept provider arguments: "
                f"{sorted(provider_kwargs)}."
            )
        return cls(
            caps=caps,
            launch_plan=build_graph_stable_decode_launch_plan(spec, caps),
            kivi_launch_plan=build_deltakv_kivi_decode_launch_plan(spec, caps),
        )

    @classmethod
    def supports(
        cls,
        spec: DecodeAttentionOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        if not spec.cuda_graph:
            return SupportResult.unsupported("reserved for CUDA Graph")
        if spec.kv_storage_format != "dense":
            return SupportResult.unsupported("does not support standalone quantized pages")
        if not spec.may_use_full_layer_kivi_int4:
            return SupportResult.unsupported(
                "reserved for mixed dense and full-layer KIVI int4 storage"
            )
        if spec.context_capacity is None:
            return SupportResult.unsupported("requires a static context capacity")
        return match_attention_capabilities(
            spec.kernel_request,
            caps,
            cls.capabilities,
        )

    def prepare(
        self,
        spec: DecodeAttentionOpSpec,
        *,
        device_index: int | None = None,
    ) -> None:
        del device_index
        plans = (self.launch_plan, self.kivi_launch_plan)
        if any(plan.context_capacity != spec.context_capacity for plan in plans):
            raise RuntimeError(
                "DeltaKV fixed-grid launch plans do not match the operator "
                f"capacity: plans={[plan.context_capacity for plan in plans]} "
                f"spec={spec.context_capacity}."
            )

    def close(self) -> None:
        self._active_graph_state = None

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "implementation_source": "repo_triton",
            "kernel_path": "paged_flash_decode + full_layer_kivi_flash_decode",
            "cuda_graph_mode": "batch_indexed",
            "launch_plan": self.launch_plan.as_dict(),
            "kivi_launch_plan": self.kivi_launch_plan.as_dict(),
            "workspace_owner": "per_graph_provider_state",
            "payload_routes": ["dense", "full_layer_kivi"],
        }

    def init_decode_graph_state(
        self,
        spec: DecodeAttentionOpSpec,
        contract,
        inputs,
    ) -> _DeltaKVFixedGridDecodeState:
        if int(contract.batch_capacity) > int(spec.max_batch_size):
            raise ValueError(
                "DeltaKV graph batch exceeds the prepared operator capacity: "
                f"graph={contract.batch_capacity} operator={spec.max_batch_size}."
            )
        if spec.context_capacity is None or int(contract.context_capacity) > int(
            spec.context_capacity
        ):
            raise ValueError(
                "DeltaKV graph context exceeds the prepared operator capacity: "
                f"graph={contract.context_capacity} operator={spec.context_capacity}."
            )
        return _DeltaKVFixedGridDecodeState.allocate(
            spec,
            self._caps,
            batch_capacity=int(contract.batch_capacity),
            context_capacity=int(contract.context_capacity),
            device=inputs.context_lens.device,
        )

    def prepare_decode_graph_out(
        self,
        state: _DeltaKVFixedGridDecodeState,
    ) -> None:
        self._active_graph_state = state

    def prepare_decode_graph_in(
        self,
        state: _DeltaKVFixedGridDecodeState,
    ) -> None:
        self._active_graph_state = state

    def decode_graph_keepalive_tensors(
        self,
        state: _DeltaKVFixedGridDecodeState,
    ) -> list[torch.Tensor]:
        return state.keepalive_tensors()

    def close_decode_graph_state(
        self,
        state: _DeltaKVFixedGridDecodeState,
    ) -> None:
        if self._active_graph_state is state:
            self._active_graph_state = None

    def run(
        self,
        spec: DecodeAttentionOpSpec,
        q: torch.Tensor,
        view: Any,
        **kwargs,
    ) -> torch.Tensor:
        kwargs.pop("decode_launch_op", None)
        if kwargs:
            raise TypeError(
                "DeltaKV fixed-grid decode received unsupported arguments: "
                f"{sorted(kwargs)}."
            )
        state = self._active_graph_state
        if state is None:
            raise RuntimeError(
                "DeltaKV fixed-grid decode has no active graph participant state."
            )
        batch_size = int(q.shape[0])
        if batch_size > state.batch_capacity:
            raise RuntimeError(
                "DeltaKV fixed-grid decode batch exceeds active state capacity: "
                f"batch={batch_size} capacity={state.batch_capacity}."
            )
        payload = view.payload
        backend = getattr(payload, "backend", None)
        if backend == "dense":
            return self._run_dense(spec, q, view, state, batch_size)
        if backend == "full_layer_kivi":
            return self._run_full_layer_kivi(q, view, state, batch_size)
        raise RuntimeError(
            "DeltaKV fixed-grid decode requires dense or full-layer KIVI storage, "
            f"got {backend!r}."
        )

    @staticmethod
    def _run_dense(
        spec: DecodeAttentionOpSpec,
        q: torch.Tensor,
        view: Any,
        state: _DeltaKVFixedGridDecodeState,
        batch_size: int,
    ) -> torch.Tensor:
        from sparseengine.kernels.triton.paged_flash_decoding import (
            paged_flash_decode,
        )

        return paged_flash_decode(
            q,
            view.payload.k_cache,
            view.payload.v_cache,
            view.meta.active_slots,
            view.meta.req_indices,
            view.meta.context_lens,
            state.mid_o[:batch_size],
            state.mid_lse[:batch_size],
            attn_score=view.meta.attn_score,
            softmax_scale=spec.softmax_scale,
            target_tokens_per_split=state.launch_plan.target_tokens_per_split,
            block_n=state.launch_plan.block_n,
            num_warps=state.launch_plan.stage1_num_warps,
            num_stages=state.launch_plan.stage1_num_stages,
            stage2_num_warps=state.launch_plan.stage2_num_warps,
            stage2_num_stages=state.launch_plan.stage2_num_stages,
            output_lse=state.output_lse[:, :batch_size],
            output=state.output[:batch_size],
        )

    @staticmethod
    def _run_full_layer_kivi(
        q: torch.Tensor,
        view: Any,
        state: _DeltaKVFixedGridDecodeState,
        batch_size: int,
    ) -> torch.Tensor:
        metadata = getattr(view.payload, "metadata", None)
        if metadata is None:
            raise RuntimeError("Full-layer KIVI decode view is missing metadata.")
        required = (
            "kivi_block_slots_map",
            "kivi_block_start_pos",
            "key_packed",
            "key_scales",
            "key_mins",
            "value_packed",
            "value_scales",
            "value_mins",
            "group_size",
        )
        missing = [name for name in required if name not in metadata]
        if missing:
            raise RuntimeError(
                f"Full-layer KIVI decode view is missing metadata: {missing}."
            )

        from sparseengine.kernels.triton.deltakv_kernels import (
            full_layer_kivi_flash_decode_stage1,
        )
        from sparseengine.kernels.triton.paged_flash_decoding import (
            fixed_grid_flash_decode_stage2,
        )

        plan = state.kivi_launch_plan
        mid_o = state.kivi_mid_o[:batch_size]
        mid_lse = state.kivi_mid_lse[:batch_size]
        full_layer_kivi_flash_decode_stage1(
            q=q,
            raw_k=view.payload.k_cache,
            raw_v=view.payload.v_cache,
            raw_slots_map=view.meta.active_slots,
            kivi_block_slots_map=metadata["kivi_block_slots_map"],
            kivi_block_start_pos=metadata["kivi_block_start_pos"],
            key_packed=metadata["key_packed"],
            key_scales=metadata["key_scales"],
            key_mins=metadata["key_mins"],
            value_packed=metadata["value_packed"],
            value_scales=metadata["value_scales"],
            value_mins=metadata["value_mins"],
            req_indices=view.meta.req_indices,
            context_lens=view.meta.context_lens,
            max_len_in_batch=plan.context_capacity,
            mid_out=mid_o,
            mid_out_logsumexp=mid_lse,
            group_size=int(metadata["group_size"]),
            block_seq=plan.target_tokens_per_split,
            block_n=plan.block_n,
            num_warps=plan.stage1_num_warps,
            num_stages=plan.stage1_num_stages,
            attn_score=view.meta.attn_score,
            max_kv_splits=plan.max_kv_splits,
            target_tokens_per_split=plan.target_tokens_per_split,
        )
        output = state.output[:batch_size]
        fixed_grid_flash_decode_stage2(
            mid_o,
            mid_lse,
            view.meta.context_lens,
            output,
            state.output_lse[:, :batch_size],
            target_tokens_per_split=plan.target_tokens_per_split,
            num_warps=plan.stage2_num_warps,
            num_stages=plan.stage2_num_stages,
        )
        return output


@DECODE_ATTENTION_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class QuantizedPagesDecodeAttentionProvider(DecodeAttentionProvider):
    name = "triton_quantized_pages_decode"
    supports_decode_graph = True

    @classmethod
    def supports(cls, spec, caps):
        if spec.kv_storage_format == "dense":
            return SupportResult.unsupported("requires quantized page storage")
        if caps.platform is not PlatformEnum.CUDA or not caps.supports_triton:
            return SupportResult.unsupported("requires CUDA and Triton")
        if spec.kv_storage_format == "fp8_kv" and not caps.supports_native_fp8:
            return SupportResult.unsupported("E4M3 KV storage requires native FP8 conversion support")
        if spec.may_require_attention_scores or spec.may_use_full_layer_kivi_int4:
            return SupportResult.unsupported("requires score-free quantized page decode")
        if spec.head_dim not in {64, 128, 256} or spec.activation_dtype not in {torch.float16, torch.bfloat16}:
            return SupportResult.unsupported("requires head_dim 64/128/256 and FP16/BF16 queries")
        if spec.page_size != 1 or spec.context_capacity is None:
            return SupportResult.unsupported("requires token slot maps and a bounded context capacity")
        return SupportResult.yes()

    def prepare(self, spec, *, device_index=None):
        device = torch.device("cuda", int(device_index or 0))
        splits = math.ceil(spec.context_capacity / 128)
        self.mid_o = torch.empty(spec.max_batch_size * spec.num_query_heads * splits * spec.head_dim,
                                 dtype=torch.float32, device=device)
        self.mid_lse = torch.empty(spec.max_batch_size * spec.num_query_heads * splits,
                                   dtype=torch.float32, device=device)
        self.output = torch.empty(spec.max_batch_size, spec.num_query_heads, spec.head_dim,
                                   dtype=spec.activation_dtype, device=device)
        self.output_lse = torch.empty(spec.num_query_heads * spec.max_batch_size,
                                       dtype=torch.float32, device=device)

    def run(self, spec, q, view, **kwargs):
        from sparseengine.engine.cache_manager.storage.quantized_kv import QuantizedKVPayload
        from sparseengine.kernels.triton.quantized_kv import quantized_decode

        kwargs.pop("decode_launch_op", None)
        if kwargs:
            raise TypeError(f"Unexpected quantized decode arguments: {sorted(kwargs)}.")
        payload = view.payload
        if not isinstance(payload, QuantizedKVPayload) or payload.format != spec.kv_storage_format:
            raise TypeError("Quantized decode payload does not match the prepared storage contract.")
        if view.meta.attn_score is not None:
            raise ValueError("Quantized decode does not produce attention scores.")
        batch, heads, dim = q.shape
        if (heads, dim) != (spec.num_query_heads, spec.head_dim) or q.dtype != spec.activation_dtype:
            raise ValueError("Quantized decode query differs from its prepared shape/dtype.")
        length = view.meta.max_context_len
        if length is None or not 0 < length <= spec.context_capacity or batch > spec.max_batch_size:
            raise ValueError("Quantized decode exceeds its prepared batch/context capacity.")
        # Graph topology depends only on the prepared capacity, never live length.
        splits = math.ceil((spec.context_capacity if spec.cuda_graph else length) / 128)
        mid_o = self.mid_o[:batch * heads * splits * dim].view(batch, heads, splits, dim)
        mid_lse = self.mid_lse[:batch * heads * splits].view(batch, heads, splits)
        query = q if payload.rotation is None else (q.float() @ payload.rotation).to(q.dtype)
        output = quantized_decode(query, payload, view.meta.active_slots, view.meta.req_indices,
                                  view.meta.context_lens, mid_o, mid_lse, softmax_scale=spec.softmax_scale,
                                  output=self.output[:batch], output_lse=self.output_lse[:heads * batch].view(heads, batch))
        return output if payload.rotation is None else (output.float() @ payload.rotation.T).to(q.dtype)

    def close(self):
        self.mid_o = self.mid_lse = self.output = self.output_lse = None

    def binding_metadata(self):
        return {"implementation_kind": "atomic_provider", "implementation_source": "repo_triton",
                "kernel_path": "quantized_kv", "workspace_owner": "provider"}


class PreparedDecodeAttentionOp:
    """One prepared decode provider shared by all compatible MHA layers."""

    def __init__(
        self,
        spec: DecodeAttentionOpSpec,
        provider: DecodeAttentionProvider,
    ) -> None:
        self.spec = spec
        self.provider = provider
        self._closed = False

    @property
    def name(self) -> str:
        return self.provider.name

    @property
    def supports_decode_graph(self) -> bool:
        return bool(
            getattr(self.provider, "supports_decode_graph", False)
        )

    def run(self, q: torch.Tensor, view: Any, **kwargs) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("Decode attention operator is closed.")
        if view.meta.attn_score is not None and not self.spec.may_require_attention_scores:
            raise RuntimeError(
                "Decode attention view requested scores after a score-free provider "
                "was bound during model preparation."
            )
        result = self.provider.run(self.spec, q, view, **kwargs)
        if not self.spec.h2o_layerwise_probability_scores:
            if isinstance(result, DecodeAttentionRunResult):
                raise RuntimeError(
                    "Decode provider returned an unrequested softmax LSE."
                )
            return result
        if not isinstance(result, DecodeAttentionRunResult):
            raise RuntimeError(
                "H2O decode requested softmax LSE but the provider returned none."
            )
        score = view.meta.attn_score
        if score is None or score.ndim != 2:
            raise RuntimeError(
                "Layer-wise H2O decode requires a reduced [batch, width] score."
            )
        from sparseengine.kernels.triton.h2o_decode_score import (
            h2o_probability_from_lse,
        )

        h2o_probability_from_lse(
            q,
            view.payload.k_cache,
            result.softmax_lse,
            view.meta.active_slots,
            view.meta.req_indices,
            view.meta.context_lens,
            score,
            softmax_scale=self.spec.softmax_scale,
        )
        return result.output

    @property
    def decode_graph_lifecycle(self) -> bool:
        return self.spec.cuda_graph and bool(
            getattr(self.provider, "decode_graph_lifecycle", False)
        )

    def init_decode_graph_state(self, contract, inputs):
        initializer = getattr(self.provider, "init_decode_graph_state", None)
        if not callable(initializer):
            raise TypeError(
                f"Decode provider {self.provider.name!r} has no graph-state initializer."
            )
        return initializer(self.spec, contract, inputs)

    def prepare_decode_graph_out(self, state) -> None:
        self.provider.prepare_decode_graph_out(state)

    def prepare_decode_graph_in(self, state) -> None:
        self.provider.prepare_decode_graph_in(state)

    def decode_graph_keepalive_tensors(self, state) -> list[torch.Tensor]:
        return list(self.provider.decode_graph_keepalive_tensors(state))

    def close_decode_graph_state(self, state) -> None:
        self.provider.close_decode_graph_state(state)

    def close(self) -> None:
        if self._closed:
            return
        self.provider.close()
        self._closed = True


def prepare_decode_attention_op(
    spec: DecodeAttentionOpSpec,
    *,
    device_index: int | None = None,
) -> PreparedDecodeAttentionOp:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    resolved = OpResolver(DECODE_ATTENTION_REGISTRY).resolve(
        spec,
        caps,
    )
    logger.info(
        "Resolved MHA decode provider={} rejected={}",
        resolved.provider.name,
        dict(resolved.rejected),
    )
    resolved.provider.prepare(spec, device_index=device_index)
    return PreparedDecodeAttentionOp(spec, resolved.provider)


def collect_decode_graph_participants(model: torch.nn.Module) -> tuple[object, ...]:
    """Collect unique prepared decode operators with graph-out lifecycle state."""

    from sparseengine.layers.attention import Attention

    participants: list[object] = []
    seen: set[int] = set()
    for module in model.modules():
        if not isinstance(module, Attention):
            continue
        participant = getattr(module, "decode_op", None)
        if participant is None or not bool(
            getattr(participant, "decode_graph_lifecycle", False)
        ):
            continue
        identity = id(participant)
        if identity not in seen:
            seen.add(identity)
            participants.append(participant)
    return tuple(participants)


def validate_decode_graph_model(model: torch.nn.Module) -> int:
    """Audit every semantic decode path after construction-time binding."""
    from sparseengine.layers.attention import Attention

    validated = 0
    for module in model.modules():
        if isinstance(module, Attention):
            decode_op = getattr(module, "decode_op", None)
            implementation = (
                decode_op
                if decode_op is not None
                else getattr(module, "attention_backend", None)
            )
            if not bool(
                getattr(implementation, "supports_decode_graph", False)
            ):
                raise RuntimeError(
                    "decode CUDA Graph requires a graph-stable "
                    f"attention provider, got {type(implementation).__name__}."
                )
            validated += 1
        if getattr(module, "is_gated_delta_rule_layer", False):
            op = getattr(module, "gated_delta_rule_op", None)
            if not bool(getattr(op, "supports_decode_graph", False)):
                raise RuntimeError(
                    "decode CUDA Graph requires a graph-stable "
                    "GDN provider."
                )
            validated += 1

    model_body = getattr(model, "model", None)
    mla_attention = getattr(model_body, "mla_attention", None)
    if mla_attention is not None:
        provider = getattr(mla_attention, "provider", None)
        if not bool(
            getattr(provider, "supports_decode_graph", False)
        ):
            raise RuntimeError(
                "decode CUDA Graph requires a graph-stable "
                "MLA provider."
            )
        validated += 1
    if validated == 0:
        raise RuntimeError(
            "decode CUDA Graph found no validated decode operator."
        )
    return validated


@dataclass(frozen=True)
class DecodeAttentionLaunchSpec:
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    activation_dtype: torch.dtype
    page_size: int = 1

    def __post_init__(self) -> None:
        if self.num_query_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("Decode attention head counts must be positive.")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("Decode query heads must be divisible by KV heads.")
        if self.head_dim <= 0 or self.page_size <= 0:
            raise ValueError("Decode attention dimensions must be positive.")


class DecodeAttentionLaunchProvider:
    name = ""

    def launch_config(
        self,
        *,
        block_seq: int,
        max_context_len: int,
        requires_attention_scores: bool,
    ) -> tuple[int, int, int]:
        raise NotImplementedError


DECODE_ATTENTION_LAUNCH_REGISTRY: OpRegistry[
    DecodeAttentionLaunchSpec, DecodeAttentionLaunchProvider
] = OpRegistry(
    "decode attention launch",
    portfolio=PortfolioPolicy(repo_portable=("default_gqa",)),
    profile_order=("h100_gqa_12q_2kv_hd128_profile",),
)


@DECODE_ATTENTION_LAUNCH_REGISTRY.register_atomic(
    ProviderRole.REPO_PORTABLE,
    profile_only=True,
)
class H100GqaDecodeLaunchProvider(DecodeAttentionLaunchProvider):
    name = "h100_gqa_12q_2kv_hd128"

    @classmethod
    def supports(
        cls,
        spec: DecodeAttentionLaunchSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.unsupported(
                f"requires CUDA SM90, got {caps.platform.name} {caps.compute_capability}"
            )
        if not caps.supports_triton:
            return SupportResult.unsupported("platform does not support Triton")
        if spec.activation_dtype != torch.bfloat16:
            return SupportResult.unsupported(
                f"requires BF16 query/KV tensors, got {spec.activation_dtype}"
            )
        if spec.page_size != 1:
            return SupportResult.unsupported(
                f"requires token-page KV storage (page_size=1), got {spec.page_size}"
            )
        return SupportResult.yes()

    def launch_config(
        self,
        *,
        block_seq: int,
        max_context_len: int,
        requires_attention_scores: bool,
    ) -> tuple[int, int, int]:
        del max_context_len
        if not requires_attention_scores:
            return 1024, 128, 4
        return int(block_seq), 16, 2


@DECODE_ATTENTION_LAUNCH_REGISTRY.register_profile
class H100GqaDecodeLaunchProfile:
    name = "h100_gqa_12q_2kv_hd128_profile"

    @classmethod
    def atomic_provider_names(
        cls,
        spec: DecodeAttentionLaunchSpec,
    ) -> tuple[str, ...]:
        del spec
        return ("h100_gqa_12q_2kv_hd128",)

    @classmethod
    def matches(
        cls,
        spec: DecodeAttentionLaunchSpec,
        caps: DeviceCaps,
    ) -> ProfileMatch:
        expected_shape = (12, 2, 128)
        actual_shape = (
            spec.num_query_heads,
            spec.num_kv_heads,
            spec.head_dim,
        )
        if not device_name_contains(caps.device_name, "H100"):
            return ProfileMatch.no(
                f"requires profiled H100 hardware, got {caps.device_name}"
            )
        if actual_shape != expected_shape:
            return ProfileMatch.no(
                f"requires profiled local Q/KV/head shape {expected_shape}, "
                f"got {actual_shape}"
            )
        return ProfileMatch.yes("matched H100 GQA launch profile")

    @classmethod
    def bind(cls, spec: DecodeAttentionLaunchSpec, caps: DeviceCaps, **kwargs):
        del spec, caps
        if kwargs:
            raise TypeError(
                f"{cls.name} does not accept provider arguments: {sorted(kwargs)}"
            )
        return H100GqaDecodeLaunchProvider()


@DECODE_ATTENTION_LAUNCH_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class DefaultGqaDecodeLaunchProvider(DecodeAttentionLaunchProvider):
    name = "default_gqa"

    @classmethod
    def supports(
        cls,
        spec: DecodeAttentionLaunchSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        del spec, caps
        return SupportResult.yes()

    def launch_config(
        self,
        *,
        block_seq: int,
        max_context_len: int,
        requires_attention_scores: bool,
    ) -> tuple[int, int, int]:
        del max_context_len, requires_attention_scores
        return int(block_seq), 16, 2


class PreparedDecodeAttentionLaunchOp:
    def __init__(
        self,
        spec: DecodeAttentionLaunchSpec,
        provider: DecodeAttentionLaunchProvider,
    ) -> None:
        self.spec = spec
        self.provider = provider

    @property
    def name(self) -> str:
        return self.provider.name

    def launch_config(self, **kwargs) -> tuple[int, int, int]:
        return self.provider.launch_config(**kwargs)


def prepare_decode_attention_launch_op(
    spec: DecodeAttentionLaunchSpec,
    *,
    device_index: int | None = None,
) -> PreparedDecodeAttentionLaunchOp:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    provider = OpResolver(DECODE_ATTENTION_LAUNCH_REGISTRY).resolve(spec, caps).provider
    return PreparedDecodeAttentionLaunchOp(spec, provider)
