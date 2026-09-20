from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import torch

import sparseengine.platforms as platforms
from sparseengine.kernels.external.sgl.fa3 import (
    SglFa3DecodeKernel,
    sgl_fa3_device_support,
)
from sparseengine.kernels.external.flashinfer.prefill import (
    flashinfer_paged_prefill_support,
    make_flashinfer_paged_prefill_wrapper,
)
from sparseengine.kernels.external.flashprefill_v2.prefill import (
    build_flashprefill_v2_page_table,
    make_flashprefill_v2,
)
from sparseengine.kernels.external.flashprefill_v2.support import (
    flashprefill_v2_support,
)
from sparseengine.operators.attention_capabilities import (
    AttentionKernelCapabilities,
    AttentionKernelRequest,
    AttentionScoreKind,
    match_attention_capabilities,
)
from sparseengine.operators.registry import (
    OpRegistry,
    OpResolver,
    NoProviderError,
    PortfolioPolicy,
    ProviderRole,
    SupportResult,
)
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum
from sparseengine.utils.log import logger


@dataclass(frozen=True, kw_only=True)
class FlashPrefillV2Semantics:
    abs_threshold: float
    k_block_m: int = 128
    k_block_n: int = 128
    attention_sink_blocks: int = 2
    window_blocks: int = 4
    last_query_blocks: int = 8
    min_sparse_q_len: int = 4096
    use_mean_correction: bool = True

    def __post_init__(self) -> None:
        if self.k_block_m <= 0 or self.k_block_m % 16:
            raise ValueError("FlashPrefill V2 k_block_m must be a positive multiple of 16.")
        if (
            self.k_block_n <= 0
            or self.k_block_n % 64
            or self.k_block_n & (self.k_block_n - 1)
        ):
            raise ValueError(
                "FlashPrefill V2 k_block_n must be a power-of-two multiple of 64."
            )
        if not 0.0 <= self.abs_threshold <= 1.0:
            raise ValueError("FlashPrefill V2 abs_threshold must be in [0, 1].")
        if min(
            self.attention_sink_blocks,
            self.window_blocks,
            self.last_query_blocks,
            self.min_sparse_q_len,
        ) < 0:
            raise ValueError("FlashPrefill V2 block-retention counts must be non-negative.")


@dataclass(frozen=True)
class PrefillAttentionOpSpec:
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    activation_dtype: torch.dtype
    softmax_scale: float
    causal: bool = True
    page_size: int = 1
    score_output: AttentionScoreKind = AttentionScoreKind.NONE
    optional_score_output: bool = False
    layer_varying_page_table: bool = False
    varlen: bool = True
    cuda_graph: bool = False
    return_softmax_lse: bool = False
    allow_softmax_lse_fallback: bool = False
    prefill_sparse_method: str = ""
    flashprefill_v2: FlashPrefillV2Semantics | None = None

    def __post_init__(self) -> None:
        if self.num_query_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("Prefill attention head counts must be positive.")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("Query heads must be divisible by KV heads.")
        if self.head_dim <= 0 or self.page_size <= 0:
            raise ValueError("Prefill attention dimensions must be positive.")
        if self.softmax_scale <= 0:
            raise ValueError("Prefill attention softmax_scale must be positive.")
        if self.optional_score_output and self.score_output == AttentionScoreKind.NONE:
            raise ValueError("Optional score output requires a non-NONE score kind.")
        if self.prefill_sparse_method not in {
            "", "h2o_prefill", "flashprefill_v2", "omnikv_prefill"
        }:
            raise ValueError(
                "Unknown prefill sparse method in operator spec: "
                f"{self.prefill_sparse_method!r}."
            )
        if (self.prefill_sparse_method == "flashprefill_v2") != (
            self.flashprefill_v2 is not None
        ):
            raise ValueError(
                "FlashPrefill V2 operator semantics must be present exactly when "
                "prefill_sparse_method='flashprefill_v2'."
            )

    @property
    def kernel_request(self) -> AttentionKernelRequest:
        return AttentionKernelRequest(
            activation_dtype=self.activation_dtype,
            head_dim=self.head_dim,
            page_size=self.page_size,
            score_output=self.score_output,
            requires_softmax_lse=self.return_softmax_lse,
            layer_varying_page_table=self.layer_varying_page_table,
            varlen=self.varlen,
            cuda_graph=self.cuda_graph,
        )


class PrefillAttentionProvider:
    name = ""
    capabilities: AttentionKernelCapabilities

    def prepare(
        self,
        spec: PrefillAttentionOpSpec,
        *,
        device_index: int | None = None,
    ) -> None:
        del spec, device_index

    def close(self) -> None:
        pass

    def run(
        self,
        spec: PrefillAttentionOpSpec,
        q: torch.Tensor,
        view: Any,
        *,
        qo_indptr: torch.Tensor,
        chunk_lens: torch.Tensor,
        max_context_len: int,
        layer_idx: int,
    ) -> torch.Tensor:
        raise NotImplementedError


@dataclass(frozen=True)
class PrefillAttentionRunResult:
    output: torch.Tensor
    softmax_lse: torch.Tensor


def _validate_token_page_table(view: Any) -> None:
    if view.active_slots.dtype != torch.int32:
        raise TypeError(
            "Paged prefill requires an int32 physical-slot page table, got "
            f"{view.active_slots.dtype}."
        )
    if view.active_slots.ndim != 2:
        raise ValueError(
            "Paged prefill expects a 2D physical-slot page table, got "
            f"shape={tuple(view.active_slots.shape)}."
        )


def _require_dense_prefill_semantics(
    spec: PrefillAttentionOpSpec,
) -> SupportResult:
    if spec.prefill_sparse_method == "flashprefill_v2":
        return SupportResult.unsupported(
            "provider implements dense prefill semantics, got "
            f"prefill_sparse_method={spec.prefill_sparse_method!r}"
        )
    return SupportResult.yes()


PREFILL_ATTENTION_REGISTRY: OpRegistry[
    PrefillAttentionOpSpec, PrefillAttentionProvider
] = OpRegistry(
    "paged prefill attention",
    portfolio=PortfolioPolicy(
        upstream_standard=(
            "sgl_fa3_paged_prefill_sm90",
            "flashinfer_paged_prefill_fa3_sm90",
            "flashinfer_paged_prefill_fa2_sm120",
        ),
        repo_portable=("triton_paged_prefill",),
        repo_nonstandard=(
            "flashprefill_v2",
            "tilelang_gqa_paged_prefill",
        ),
    ),
)


def _view_parts(view: Any) -> tuple[Any, Any]:
    """Return the physical payload and logical metadata for a prefill view."""

    return getattr(view, "payload", view), getattr(view, "meta", view)


def _view_score_kind(view: Any) -> AttentionScoreKind:
    _, meta = _view_parts(view)
    score = meta.attn_score
    if score is None:
        return AttentionScoreKind.NONE
    if score.ndim == 3:
        return AttentionScoreKind.RAW_QK_PER_HEAD
    if score.ndim == 2:
        return AttentionScoreKind.RAW_QK_REDUCED
    raise ValueError(
        "Prefill attention score must have shape [batch, heads, tokens] or "
        f"[batch, tokens], got {tuple(score.shape)}."
    )


@PREFILL_ATTENTION_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class FlashInferPagedPrefillAttentionProvider(PrefillAttentionProvider):
    name = "flashinfer_paged_prefill_fa3_sm90"
    backend = "fa3"
    capabilities = AttentionKernelCapabilities(
        platforms=frozenset({PlatformEnum.CUDA}),
        compute_capabilities=frozenset({(9, 0)}),
        activation_dtypes=frozenset({torch.bfloat16}),
        head_dims=frozenset({128}),
        page_sizes=frozenset({1}),
        score_outputs=frozenset({AttentionScoreKind.NONE}),
        layer_varying_page_table=False,
        varlen=True,
        minimum_runtime_version=(12, 8),
    )

    @classmethod
    def supports(
        cls, spec: PrefillAttentionOpSpec, caps: DeviceCaps
    ) -> SupportResult:
        semantics = _require_dense_prefill_semantics(spec)
        if not semantics.supported:
            return semantics
        common = match_attention_capabilities(
            spec.kernel_request, caps, cls.capabilities
        )
        if not common.supported:
            return common
        if not spec.causal:
            return SupportResult.unsupported("requires causal attention")
        supported, reason = flashinfer_paged_prefill_support(cls.backend)
        return SupportResult.yes(reason) if supported else SupportResult.unsupported(reason)

    def __init__(self) -> None:
        self._state: _FlashInferPagedPrefillState | None = None

    def prepare(
        self,
        spec: PrefillAttentionOpSpec,
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
                "FlashInfer prefill must be prepared on the selected CUDA device: "
                f"selected={device_index} current={current_device}."
            )
        device = torch.device("cuda", int(device_index))
        self._state = _FlashInferPagedPrefillState(device, backend=self.backend)

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "source": "flashinfer-python",
            "backend": self.backend,
            "kv_layout": "NHD",
            "page_size": 1,
        }

    def close(self) -> None:
        self._state = None

    def run(
        self,
        spec,
        q,
        view,
        *,
        qo_indptr,
        chunk_lens,
        max_context_len,
        layer_idx,
    ):
        del chunk_lens, layer_idx
        payload, meta = _view_parts(view)
        _validate_token_page_table(meta)
        if self._state is None:
            raise RuntimeError("FlashInfer prefill provider was not prepared.")
        state = self._state
        assert state is not None
        if q.dtype != spec.activation_dtype:
            raise TypeError(
                f"FlashInfer paged prefill expected {spec.activation_dtype} Q, got {q.dtype}."
            )
        if payload.k_cache.dtype != q.dtype or payload.v_cache.dtype != q.dtype:
            raise TypeError(
                "FlashInfer paged prefill requires Q/K/V with the same dtype, got "
                f"{q.dtype}/{payload.k_cache.dtype}/{payload.v_cache.dtype}."
            )
        from sparseengine.utils.context import get_context

        plan_scope = get_context().attention_validation_scope
        if state.plan_scope is not plan_scope:
            state.plan(
                spec,
                qo_indptr=qo_indptr,
                active_slots=meta.active_slots,
                req_indices=meta.req_indices,
                context_lens=meta.context_lens,
                max_context_len=max_context_len,
                plan_scope=plan_scope,
            )
        output = torch.empty_like(q)
        state.wrapper.run(
            q,
            (
                payload.k_cache.unsqueeze(1),
                payload.v_cache.unsqueeze(1),
            ),
            out=output,
        )
        return output


class _FlashInferPagedPrefillState:
    def __init__(self, device: torch.device, *, backend: str) -> None:
        self.backend = backend
        self.workspace = torch.empty(
            128 * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )
        self.wrapper = make_flashinfer_paged_prefill_wrapper(
            self.workspace,
            backend=backend,
        )
        self.int_workspace: torch.Tensor | None = None
        if backend == "fa2":
            self.int_workspace = torch.empty(
                8 * 1024 * 1024,
                dtype=torch.uint8,
                device=device,
            )
            self.wrapper.reset_workspace_buffer(
                self.workspace,
                self.int_workspace,
            )
        self.plan_scope: object | None = None

    def _ensure_workspace(self, *args, **kwargs) -> None:
        if self.backend != "fa2":
            return
        int_workspace = self.int_workspace
        if int_workspace is None:
            raise RuntimeError(
                "FlashInfer FA2 prefill requires a caller-owned integer workspace."
            )
        required_float, required_int = self.wrapper.workspace_size(*args, **kwargs)
        try:
            required_float = int(required_float)
            required_int = int(required_int)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "FlashInfer paged prefill returned invalid workspace sizes: "
                f"float={required_float!r} int={required_int!r}."
            ) from error
        if required_float < 0 or required_int < 0:
            raise RuntimeError(
                "FlashInfer paged prefill returned negative workspace sizes: "
                f"float={required_float} int={required_int}."
            )

        current_float = self.workspace.numel() * self.workspace.element_size()
        current_int = int_workspace.numel() * int_workspace.element_size()
        if required_float <= current_float and required_int <= current_int:
            return

        next_float = max(current_float, required_float)
        next_int = max(current_int, required_int)
        workspace = torch.empty(
            next_float,
            dtype=torch.uint8,
            device=self.workspace.device,
        )
        int_workspace = torch.empty(
            next_int,
            dtype=torch.uint8,
            device=int_workspace.device,
        )
        self.wrapper.reset_workspace_buffer(workspace, int_workspace)
        self.workspace = workspace
        self.int_workspace = int_workspace
        logger.info(
            "Grew FlashInfer paged prefill workspace: float_bytes={}->{} "
            "int_bytes={}->{}",
            current_float,
            next_float,
            current_int,
            next_int,
        )

    def plan(
        self,
        spec: PrefillAttentionOpSpec,
        *,
        qo_indptr: torch.Tensor,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
        plan_scope: object,
    ) -> None:
        if active_slots.dim() != 2:
            raise ValueError(
                "FlashInfer paged prefill expects a 2D active slot table, got "
                f"{tuple(active_slots.shape)}."
            )
        batch_size = int(context_lens.numel())
        if batch_size <= 0 or int(req_indices.numel()) != batch_size:
            raise ValueError("FlashInfer paged prefill requires matched non-empty metadata.")
        max_context_len = int(max_context_len)
        if max_context_len <= 0 or max_context_len > int(active_slots.shape[1]):
            raise ValueError(
                "FlashInfer paged prefill max context is outside the active slot table: "
                f"max_context_len={max_context_len} width={int(active_slots.shape[1])}."
            )
        rows = active_slots.index_select(0, req_indices.to(torch.long))[
            :, :max_context_len
        ]
        positions = torch.arange(
            max_context_len,
            device=context_lens.device,
            dtype=context_lens.dtype,
        )
        valid = positions.unsqueeze(0) < context_lens.unsqueeze(1)
        paged_kv_indices = rows.masked_select(valid).to(torch.int32).contiguous()
        zero = torch.zeros(1, device=context_lens.device, dtype=torch.int32)
        paged_kv_indptr = torch.cat(
            (
                zero,
                context_lens.to(torch.int32).cumsum(0, dtype=torch.int32),
            )
        )
        last_page_len = torch.ones(
            batch_size,
            device=context_lens.device,
            dtype=torch.int32,
        )
        plan_args = (
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            last_page_len,
        )
        plan_kwargs = dict(
            num_qo_heads=spec.num_query_heads,
            num_kv_heads=spec.num_kv_heads,
            head_dim_qk=spec.head_dim,
            page_size=spec.page_size,
            causal=spec.causal,
            sm_scale=spec.softmax_scale,
            q_data_type=spec.activation_dtype,
            kv_data_type=spec.activation_dtype,
        )
        self._ensure_workspace(*plan_args, **plan_kwargs)
        self.wrapper.plan(
            *plan_args,
            **plan_kwargs,
            non_blocking=True,
        )
        self.plan_scope = plan_scope


@PREFILL_ATTENTION_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class FlashInferFa2Sm120PagedPrefillAttentionProvider(
    FlashInferPagedPrefillAttentionProvider
):
    """FlashInfer FA2 over its declared SM120 paged-prefill contract."""

    name = "flashinfer_paged_prefill_fa2_sm120"
    backend = "fa2"
    capabilities = AttentionKernelCapabilities(
        platforms=frozenset({PlatformEnum.CUDA}),
        compute_capabilities=frozenset({(12, 0)}),
        activation_dtypes=frozenset({torch.bfloat16}),
        head_dims=frozenset({64, 128, 256}),
        page_sizes=frozenset({1}),
        score_outputs=frozenset({AttentionScoreKind.NONE}),
        layer_varying_page_table=False,
        varlen=True,
        minimum_runtime_version=(12, 8),
    )

    @classmethod
    def supports(
        cls, spec: PrefillAttentionOpSpec, caps: DeviceCaps
    ) -> SupportResult:
        semantics = _require_dense_prefill_semantics(spec)
        if not semantics.supported:
            return semantics
        common = match_attention_capabilities(
            spec.kernel_request, caps, cls.capabilities
        )
        if not common.supported:
            return common
        if not spec.causal:
            return SupportResult.unsupported("requires causal attention")
        supported, reason = flashinfer_paged_prefill_support(cls.backend)
        return SupportResult.yes(reason) if supported else SupportResult.unsupported(reason)


@PREFILL_ATTENTION_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class SglFa3PagedPrefillAttentionProvider(PrefillAttentionProvider):
    """SGL FA3 over SparseEngine's page-size-one physical KV table."""

    name = "sgl_fa3_paged_prefill_sm90"
    capabilities = AttentionKernelCapabilities(
        platforms=frozenset({PlatformEnum.CUDA}),
        compute_capabilities=frozenset({(9, 0)}),
        activation_dtypes=frozenset({torch.bfloat16}),
        head_dims=frozenset({128, 256}),
        page_sizes=frozenset({1}),
        score_outputs=frozenset({AttentionScoreKind.NONE}),
        returns_softmax_lse=True,
        layer_varying_page_table=True,
        varlen=True,
        minimum_runtime_version=(12, 3),
    )

    @classmethod
    def supports(
        cls, spec: PrefillAttentionOpSpec, caps: DeviceCaps
    ) -> SupportResult:
        semantics = _require_dense_prefill_semantics(spec)
        if not semantics.supported:
            return semantics
        common = match_attention_capabilities(
            spec.kernel_request, caps, cls.capabilities
        )
        if not common.supported:
            return common
        if not spec.causal:
            return SupportResult.unsupported("requires causal attention")
        supported, reason = sgl_fa3_device_support(caps.device_index)
        return SupportResult.yes(reason) if supported else SupportResult.unsupported(reason)

    def __init__(self) -> None:
        self._kernel: SglFa3DecodeKernel | None = None
        self._query_plan_scope: object | None = None
        self._max_query_len = 0

    def prepare(
        self,
        spec: PrefillAttentionOpSpec,
        *,
        device_index: int | None = None,
    ) -> None:
        if self._kernel is not None:
            return
        current_device = torch.cuda.current_device()
        if device_index is None:
            device_index = current_device
        if int(device_index) != current_device:
            raise RuntimeError(
                "SGL FA3 prefill must be prepared on the selected CUDA device: "
                f"selected={device_index} current={current_device}."
            )
        self._kernel = SglFa3DecodeKernel(
            device=torch.device("cuda", int(device_index)),
            # Explicit prefill supplies its own cu_seqlens_q. The decode-only
            # buffer capacity is therefore intentionally unused by this provider.
            max_batch_size=1,
            softmax_scale=spec.softmax_scale,
        )

    def close(self) -> None:
        self._kernel = None
        self._query_plan_scope = None
        self._max_query_len = 0

    def run(
        self,
        spec,
        q,
        view,
        *,
        qo_indptr,
        chunk_lens,
        max_context_len,
        layer_idx,
    ):
        del max_context_len, layer_idx
        payload, meta = _view_parts(view)
        _validate_token_page_table(meta)
        if self._kernel is None:
            raise RuntimeError("SGL FA3 prefill provider was not prepared.")
        kernel = self._kernel
        assert kernel is not None
        if q.dtype != spec.activation_dtype:
            raise TypeError(
                f"SGL FA3 paged prefill expected {spec.activation_dtype} Q, got {q.dtype}."
            )
        if payload.k_cache.dtype != q.dtype or payload.v_cache.dtype != q.dtype:
            raise TypeError(
                "SGL FA3 paged prefill requires Q/K/V with the same dtype, got "
                f"{q.dtype}/{payload.k_cache.dtype}/{payload.v_cache.dtype}."
            )
        if qo_indptr.dtype != torch.int32 or chunk_lens.dtype != torch.int32:
            raise TypeError("SGL FA3 paged prefill requires int32 sequence metadata.")

        from sparseengine.utils.context import get_context

        validation_scope = get_context().attention_validation_scope
        if self._query_plan_scope is not validation_scope:
            self._max_query_len = int(chunk_lens.max().item())
            self._query_plan_scope = validation_scope
        output = torch.empty_like(q)
        result = kernel.run_explicit_varlen(
            q,
            payload.k_cache,
            payload.v_cache,
            meta.active_slots,
            meta.req_indices,
            meta.context_lens,
            output,
            cu_seqlens_q=qo_indptr,
            max_seqlen_q=self._max_query_len,
            validation_scope=validation_scope,
            return_softmax_lse=spec.return_softmax_lse,
        )
        if not spec.return_softmax_lse:
            return result
        if not isinstance(result, tuple):
            raise RuntimeError("SGL FA3 prefill did not return the requested softmax LSE.")
        output, softmax_lse = result
        return PrefillAttentionRunResult(output=output, softmax_lse=softmax_lse)


@PREFILL_ATTENTION_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class FlashPrefillV2Provider(PrefillAttentionProvider):
    """FlashPrefill V2 sparse-prefill semantics over explicit paged KV."""

    name = "flashprefill_v2"
    capabilities = AttentionKernelCapabilities(
        platforms=frozenset({PlatformEnum.CUDA}),
        compute_capabilities=frozenset({(9, 0)}),
        activation_dtypes=frozenset({torch.bfloat16}),
        head_dims=frozenset({128}),
        page_sizes=frozenset({1}),
        score_outputs=frozenset({AttentionScoreKind.NONE}),
        layer_varying_page_table=True,
        varlen=True,
        minimum_runtime_version=(12, 3),
    )

    def __init__(self) -> None:
        self._pipeline = None
        self._query_plan_scope: object | None = None
        self._host_query_lens: tuple[int, ...] = ()

    @classmethod
    def supports(
        cls, spec: PrefillAttentionOpSpec, caps: DeviceCaps
    ) -> SupportResult:
        if spec.prefill_sparse_method != "flashprefill_v2":
            return SupportResult.unsupported(
                "provider requires prefill_sparse_method='flashprefill_v2'"
            )
        common = match_attention_capabilities(
            spec.kernel_request, caps, cls.capabilities
        )
        if not common.supported:
            return common
        if not spec.causal:
            return SupportResult.unsupported("FlashPrefill V2 requires causal attention")
        if spec.flashprefill_v2 is None:
            return SupportResult.unsupported("FlashPrefill V2 semantics are missing")
        supported, reason = flashprefill_v2_support()
        return SupportResult.yes(reason) if supported else SupportResult.unsupported(reason)

    def prepare(
        self,
        spec: PrefillAttentionOpSpec,
        *,
        device_index: int | None = None,
    ) -> None:
        if self._pipeline is not None:
            return
        current_device = torch.cuda.current_device()
        if device_index is None:
            device_index = current_device
        if int(device_index) != current_device:
            raise RuntimeError(
                "FlashPrefill V2 must be prepared on the selected CUDA device: "
                f"selected={device_index} current={current_device}."
            )
        if spec.flashprefill_v2 is None:
            raise RuntimeError("FlashPrefill V2 semantics are missing during prepare.")
        self._pipeline = make_flashprefill_v2(
            semantics=spec.flashprefill_v2,
            softmax_scale=spec.softmax_scale,
        )

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "source": "FlashPrefillv2",
            "validated_upstream_revision": "75b58f2ecdba1c269a87dd34d8f1ae57bef50c57",
            "kv_layout": "NHD",
            "page_size": 1,
            "prefill_sparse_method": "flashprefill_v2",
        }

    def close(self) -> None:
        pipeline = self._pipeline
        if pipeline is not None:
            clear = getattr(pipeline, "clear_workspaces", None)
            if callable(clear):
                clear()
        self._pipeline = None
        self._query_plan_scope = None
        self._host_query_lens = ()

    def run(
        self,
        spec,
        q,
        view,
        *,
        qo_indptr,
        chunk_lens,
        max_context_len,
        layer_idx,
    ):
        del layer_idx
        payload, meta = _view_parts(view)
        _validate_token_page_table(meta)
        if self._pipeline is None:
            raise RuntimeError("FlashPrefill V2 provider was not prepared.")
        if q.dtype != spec.activation_dtype:
            raise TypeError(
                f"FlashPrefill V2 expected {spec.activation_dtype} Q, got {q.dtype}."
            )
        if payload.k_cache.dtype != q.dtype or payload.v_cache.dtype != q.dtype:
            raise TypeError(
                "FlashPrefill V2 requires Q/K/V with the same dtype, got "
                f"{q.dtype}/{payload.k_cache.dtype}/{payload.v_cache.dtype}."
            )
        if qo_indptr.dtype != torch.int32 or chunk_lens.dtype != torch.int32:
            raise TypeError("FlashPrefill V2 requires int32 query sequence metadata.")
        if meta.context_lens.dtype != torch.int32:
            raise TypeError("FlashPrefill V2 requires int32 cache sequence lengths.")

        from sparseengine.utils.context import get_context

        validation_scope = get_context().attention_validation_scope
        if self._query_plan_scope is not validation_scope:
            self._host_query_lens = tuple(int(value) for value in chunk_lens.tolist())
            if sum(self._host_query_lens) != int(q.shape[0]):
                raise ValueError(
                    "FlashPrefill V2 query lengths do not sum to the flattened Q rows: "
                    f"q_lens={self._host_query_lens} q_rows={int(q.shape[0])}."
                )
            self._query_plan_scope = validation_scope
        page_table = build_flashprefill_v2_page_table(
            meta.active_slots,
            meta.req_indices,
            meta.context_lens,
            max_context_len=int(max_context_len),
        )
        return self._pipeline(
            q,
            payload.k_cache.unsqueeze(1),
            payload.v_cache.unsqueeze(1),
            page_table,
            meta.context_lens,
            qo_indptr,
            q_lens=self._host_query_lens,
            max_cache_seqlen=int(max_context_len),
            softmax_scale=spec.softmax_scale,
        )


@PREFILL_ATTENTION_REGISTRY.register_atomic(
    ProviderRole.REPO_NONSTANDARD,
)
class TilelangGqaPagedPrefillAttentionProvider(PrefillAttentionProvider):
    """Portable TileLang GQA paged prefill with fused score extraction."""

    name = "tilelang_gqa_paged_prefill"
    capabilities = AttentionKernelCapabilities(
        platforms=frozenset({PlatformEnum.CUDA}),
        activation_dtypes=frozenset({torch.bfloat16}),
        head_dims=frozenset({128}),
        page_sizes=frozenset({1}),
        score_outputs=frozenset({
            AttentionScoreKind.NONE,
            AttentionScoreKind.RAW_QK_REDUCED,
        }),
        layer_varying_page_table=True,
        varlen=True,
        minimum_runtime_version=(12, 3),
    )

    def __init__(self) -> None:
        self._query_plan_scope: object | None = None
        self._max_query_len = 0

    def close(self) -> None:
        self._query_plan_scope = None
        self._max_query_len = 0

    @classmethod
    def supports(
        cls, spec: PrefillAttentionOpSpec, caps: DeviceCaps
    ) -> SupportResult:
        semantics = _require_dense_prefill_semantics(spec)
        if not semantics.supported:
            return semantics
        common = match_attention_capabilities(
            spec.kernel_request, caps, cls.capabilities
        )
        if not common.supported:
            return common
        if not spec.causal:
            return SupportResult.unsupported("requires causal attention")
        from sparseengine.kernels.tilelang.gqa.runtime import (
            tilelang_gqa_device_support,
        )

        supported, reason = tilelang_gqa_device_support(caps.device_index)
        return SupportResult.yes(reason) if supported else SupportResult.unsupported(reason)

    def run(
        self,
        spec,
        q,
        view,
        *,
        qo_indptr,
        chunk_lens,
        max_context_len,
        layer_idx,
    ):
        del layer_idx
        payload, meta = _view_parts(view)
        _validate_token_page_table(meta)
        if q.dtype != spec.activation_dtype:
            raise TypeError(
                f"TileLang GQA paged prefill expected {spec.activation_dtype} Q, got {q.dtype}."
            )
        if payload.k_cache.dtype != q.dtype or payload.v_cache.dtype != q.dtype:
            raise TypeError(
                "TileLang GQA paged prefill requires Q/K/V with the same dtype, got "
                f"{q.dtype}/{payload.k_cache.dtype}/{payload.v_cache.dtype}."
            )
        if qo_indptr.dtype != torch.int32 or chunk_lens.dtype != torch.int32:
            raise TypeError("TileLang GQA paged prefill requires int32 sequence metadata.")
        if meta.attn_score is not None and tuple(meta.attn_score.shape) != (
            int(chunk_lens.numel()),
            int(max_context_len),
        ):
            raise ValueError(
                "TileLang scored prefill requires [batch, max_context_len] output, "
                f"got {tuple(meta.attn_score.shape)} for batch={int(chunk_lens.numel())} "
                f"max_context_len={int(max_context_len)}."
            )

        from sparseengine.kernels.tilelang.gqa.prefill import (
            gqa_paged_prefill_attention_tilelang,
        )
        from sparseengine.utils.context import get_context

        prompt_cache_lens = meta.context_lens - chunk_lens
        validation_scope = get_context().attention_validation_scope
        if self._query_plan_scope is not validation_scope:
            self._max_query_len = int(chunk_lens.max().item())
            self._query_plan_scope = validation_scope
        output = torch.empty_like(q)
        return gqa_paged_prefill_attention_tilelang(
            q,
            payload.k_cache,
            payload.v_cache,
            meta.active_slots,
            meta.req_indices,
            meta.context_lens,
            prompt_cache_lens,
            qo_indptr,
            output,
            attn_score=meta.attn_score,
            sm_scale=spec.softmax_scale,
            max_query_len=self._max_query_len,
        )


@PREFILL_ATTENTION_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class TritonPagedPrefillAttentionProvider(PrefillAttentionProvider):
    name = "triton_paged_prefill"
    capabilities = AttentionKernelCapabilities(
        platforms=frozenset({PlatformEnum.CUDA, PlatformEnum.ROCM}),
        activation_dtypes=frozenset({torch.bfloat16, torch.float16}),
        head_dims=frozenset({16, 32, 64, 128, 256}),
        page_sizes=frozenset({1}),
        score_outputs=frozenset(AttentionScoreKind),
        layer_varying_page_table=True,
        varlen=True,
        requires_triton=True,
    )

    def __init__(self) -> None:
        self._query_plan_scope: object | None = None
        self._max_query_len = 0

    def close(self) -> None:
        self._query_plan_scope = None
        self._max_query_len = 0

    @classmethod
    def supports(
        cls, spec: PrefillAttentionOpSpec, caps: DeviceCaps
    ) -> SupportResult:
        semantics = _require_dense_prefill_semantics(spec)
        if not semantics.supported:
            return semantics
        common = match_attention_capabilities(
            spec.kernel_request, caps, cls.capabilities
        )
        if not common.supported:
            return common
        if not spec.causal:
            return SupportResult.unsupported("Triton prefill requires causal attention")
        expected_scale = spec.head_dim**-0.5
        if not math.isclose(
            spec.softmax_scale,
            expected_scale,
            rel_tol=1.0e-6,
            abs_tol=0.0,
        ):
            return SupportResult.unsupported(
                "Triton prefill requires the default head-dimension scale "
                f"{expected_scale}, got {spec.softmax_scale}"
            )
        return SupportResult.yes()

    def run(
        self,
        spec,
        q,
        view,
        *,
        qo_indptr,
        chunk_lens,
        max_context_len,
        layer_idx,
    ):
        del spec, max_context_len, layer_idx
        payload, meta = _view_parts(view)
        _validate_token_page_table(meta)
        from sparseengine.kernels.triton.context_flashattention_nopad import (
            context_attention_fwd,
        )

        from sparseengine.utils.context import get_context

        validation_scope = get_context().attention_validation_scope
        if self._query_plan_scope is not validation_scope:
            self._max_query_len = int(chunk_lens.max().item())
            self._query_plan_scope = validation_scope
        output = torch.empty_like(q)
        context_attention_fwd(
            q,
            payload.k_cache,
            payload.v_cache,
            output,
            meta.req_indices,
            qo_indptr[:-1],
            meta.context_lens,
            meta.context_lens - chunk_lens,
            self._max_query_len,
            meta.active_slots,
            attn_score=meta.attn_score,
        )
        return output


def _resolve_prefill_attention_provider(
    spec: PrefillAttentionOpSpec,
    *,
    device_index: int | None = None,
) -> tuple[PrefillAttentionProvider, PrefillAttentionOpSpec]:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    execution_spec = spec
    try:
        resolved = OpResolver(PREFILL_ATTENTION_REGISTRY).resolve(
            execution_spec, caps
        )
    except NoProviderError:
        if not (spec.return_softmax_lse and spec.allow_softmax_lse_fallback):
            raise
        execution_spec = replace(spec, return_softmax_lse=False)
        resolved = OpResolver(PREFILL_ATTENTION_REGISTRY).resolve(
            execution_spec, caps
        )
        logger.info(
            "No provider can return optional prefill softmax LSE; "
            "resolved provider={} with method-owned posthoc scoring",
            resolved.provider.name,
        )
    logger.info(
        "Resolved MHA prefill provider={} rejected={}",
        resolved.provider.name,
        dict(resolved.rejected),
    )
    return resolved.provider, execution_spec


def resolve_prefill_attention_provider(
    spec: PrefillAttentionOpSpec,
    *,
    device_index: int | None = None,
) -> PrefillAttentionProvider:
    provider, _ = _resolve_prefill_attention_provider(
        spec, device_index=device_index
    )
    return provider


class PreparedPrefillAttentionOp:
    """Prepared prefill dispatch shared by all layers in one model runtime."""

    def __init__(
        self,
        spec: PrefillAttentionOpSpec,
        provider: PrefillAttentionProvider,
        *,
        execution_spec: PrefillAttentionOpSpec | None = None,
        unscored_op: PreparedPrefillAttentionOp | None = None,
    ) -> None:
        self.spec = spec
        self.execution_spec = execution_spec or spec
        self.provider = provider
        self.unscored_op = unscored_op
        self._closed = False

    @property
    def name(self) -> str:
        if self.unscored_op is not None:
            return f"scored={self.provider.name},unscored={self.unscored_op.name}"
        return self.provider.name

    def run(self, q, view, **kwargs):
        if self._closed:
            raise RuntimeError("Prefill attention operator is closed.")
        actual_score = _view_score_kind(view)
        if actual_score == AttentionScoreKind.NONE and self.unscored_op is not None:
            return self.unscored_op.run(q, view, **kwargs)
        if actual_score is not self.spec.score_output:
            raise RuntimeError(
                "Prefill attention view violates the resolved score contract: "
                f"resolved={self.spec.score_output.name} actual={actual_score.name}."
            )
        return self.provider.run(self.execution_spec, q, view, **kwargs)

    def close(self) -> None:
        if self._closed:
            return
        self.provider.close()
        if self.unscored_op is not None:
            self.unscored_op.close()
        self._closed = True


def prepare_prefill_attention_op(
    spec: PrefillAttentionOpSpec,
    *,
    device_index: int | None = None,
) -> PreparedPrefillAttentionOp:
    provider, execution_spec = _resolve_prefill_attention_provider(
        spec, device_index=device_index
    )
    provider.prepare(execution_spec, device_index=device_index)
    unscored_op = None
    if spec.optional_score_output:
        try:
            unscored_op = prepare_prefill_attention_op(
                replace(
                    spec,
                    score_output=AttentionScoreKind.NONE,
                    optional_score_output=False,
                ),
                device_index=device_index,
            )
        except Exception:
            provider.close()
            raise
    return PreparedPrefillAttentionOp(
        spec,
        provider,
        execution_spec=execution_spec,
        unscored_op=unscored_op,
    )
