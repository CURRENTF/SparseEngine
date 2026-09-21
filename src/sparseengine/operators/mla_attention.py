from __future__ import annotations

from dataclasses import dataclass

import torch

import sparseengine.platforms as platforms
from sparseengine.platforms import device_runtime
from sparseengine.engine.cache_manager.base import (
    DecodeComputeView,
    MlaLatentPayload,
    PrefillComputeView,
)
from sparseengine.kernels.external.sgl.fa3 import (
    SglFa3DecodeKernel,
    sgl_fa3_device_support,
)
from sparseengine.kernels.tilelang.mla.runtime import (
    TileMlaDecodeKernel,
    TileMlaLaunchPlan,
    tilelang_mla_support,
)
from sparseengine.kernels.triton.mla import (
    DEFAULT_GLM_MLA_DECODE_CONFIG,
    MlaDecodeLaunchConfig,
    allocate_mla_decode_workspace,
    run_mla_decode,
    select_glm_mla_decode_config,
    validate_mla_decode_metadata,
)
from sparseengine.operators.registry import (
    OpRegistry,
    OpResolver,
    PortfolioPolicy,
    ProfileMatch,
    ProviderRole,
    SupportResult,
    operator_binding_report,
)
from sparseengine.operators.attention_capabilities import (
    AttentionKernelCapabilities,
    AttentionKernelRequest,
    AttentionScoreKind,
    match_attention_capabilities,
)
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum
from sparseengine.utils.device_name import device_name_contains

_GLM_MLA_NUM_Q_HEADS = 20
_GLM_MLA_KV_LORA_RANK = 512
_GLM_MLA_ROPE_DIM = 64
_GLM_MLA_QK_HEAD_DIM = 256
_GLM_MLA_VALUE_HEAD_DIM = 256
_PROFILED_H100_NAME = "H100"


@dataclass(frozen=True, slots=True)
class MlaAttentionOpSpec:
    """Construction-time contract for one MLA attention implementation."""

    num_q_heads: int
    kv_lora_rank: int
    rope_dim: int
    qk_head_dim: int
    value_head_dim: int
    activation_dtype: torch.dtype
    cache_dtype: torch.dtype
    tp_size: int
    cuda_graph: bool
    score_output: AttentionScoreKind = AttentionScoreKind.NONE
    context_capacity: int | None = None
    batch_capacity: int | None = None

    def __post_init__(self) -> None:
        dimensions = {
            "num_q_heads": self.num_q_heads,
            "kv_lora_rank": self.kv_lora_rank,
            "rope_dim": self.rope_dim,
            "qk_head_dim": self.qk_head_dim,
            "value_head_dim": self.value_head_dim,
            "tp_size": self.tp_size,
        }
        for name, value in dimensions.items():
            if int(value) <= 0:
                raise ValueError(f"MLA {name} must be positive, got {value}.")
        if self.num_q_heads % self.tp_size:
            raise ValueError(
                "MLA query heads must be divisible by tensor parallel size: "
                f"heads={self.num_q_heads} tp_size={self.tp_size}."
            )
        if self.context_capacity is not None and self.context_capacity <= 0:
            raise ValueError("MLA context_capacity must be positive.")
        if self.batch_capacity is not None and self.batch_capacity <= 0:
            raise ValueError("MLA batch_capacity must be positive.")
        if self.score_output not in {
            AttentionScoreKind.NONE,
            AttentionScoreKind.RAW_QK_PER_HEAD,
            AttentionScoreKind.RAW_QK_REDUCED,
        }:
            raise ValueError(
                "MLA decode currently supports NONE, RAW_QK_PER_HEAD, or "
                "RAW_QK_REDUCED score "
                f"contracts, got {self.score_output.name}."
            )

    @property
    def local_q_heads(self) -> int:
        return int(self.num_q_heads // self.tp_size)

    @property
    def softmax_scale(self) -> float:
        return float(self.qk_head_dim**-0.5)

    @property
    def kernel_request(self) -> AttentionKernelRequest:
        return AttentionKernelRequest(
            activation_dtype=self.activation_dtype,
            head_dim=self.qk_head_dim,
            score_output=self.score_output,
            layer_varying_page_table=True,
            varlen=True,
            cuda_graph=self.cuda_graph,
        )


class MlaAttentionProvider:
    name = ""
    capabilities: AttentionKernelCapabilities

    def run(
        self,
        q_nope_absorbed: torch.Tensor,
        q_rope: torch.Tensor,
        view: DecodeComputeView,
        output: torch.Tensor,
        *,
        validation_scope: object | None = None,
        valid_batch_size: int | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


MLA_ATTENTION_REGISTRY: OpRegistry[
    MlaAttentionOpSpec,
    MlaAttentionProvider,
] = OpRegistry(
    "MLA attention",
    portfolio=PortfolioPolicy(
        upstream_standard=("sgl_fa3_sm90",),
        repo_nonstandard=("triton_mla",),
    ),
    profile_order=(
        "tilelang_score_h100_profile",
    ),
)


@MLA_ATTENTION_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class MlaTritonProvider(MlaAttentionProvider):
    """Portable Triton provider with caller-independent decode workspace."""

    name = "triton_mla"
    supports_decode_graph = True
    capabilities = AttentionKernelCapabilities(
        platforms=frozenset({PlatformEnum.CUDA}),
        activation_dtypes=frozenset({torch.bfloat16}),
        head_dims=frozenset({_GLM_MLA_QK_HEAD_DIM}),
        score_outputs=frozenset(AttentionScoreKind),
        layer_varying_page_table=True,
        varlen=True,
        cuda_graph=True,
        requires_triton=True,
    )

    def __init__(
        self,
        *,
        op_spec: MlaAttentionOpSpec,
        device: torch.device | str,
        max_batch_size: int,
        launch_config: MlaDecodeLaunchConfig | None = None,
        sm_count: int | None = None,
    ) -> None:
        self.spec = op_spec
        requested_device = torch.device(device)
        self.max_batch_size = int(max_batch_size)
        if self.max_batch_size <= 0:
            raise ValueError(
                "MLA max_batch_size must be positive, got "
                f"{self.max_batch_size}."
            )
        self._fixed_launch_config = launch_config
        self._sm_count = sm_count
        self.launch_config = launch_config or (
            select_glm_mla_decode_config(
                batch_size=self.max_batch_size,
                local_q_heads=self.spec.local_q_heads,
                sm_count=sm_count,
            ) if sm_count is not None else DEFAULT_GLM_MLA_DECODE_CONFIG
        )
        self.workspace = allocate_mla_decode_workspace(
            batch_size=self.max_batch_size,
            head_count=self.spec.local_q_heads,
            device=requested_device,
            config=self.launch_config,
        )
        self.device = self.workspace.block_size.device
        self._validated_decode_metadata: tuple[
            object,
            list[
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    int,
                    int | None,
                ]
            ],
        ] | None = None
        self._runtime_kernel_path_counts: dict[str, dict[str, int]] = {}
        self._runtime_fallback_reasons: dict[str, int] = {}
        self._prefill = None
        if (
            self.device.type == "cuda"
            and type(self).run_prefill_chunk is MlaTritonProvider.run_prefill_chunk
        ):
            from sparseengine.operators.mla_prefill_attention import resolve_mla_prefill

            caps = platforms.current_platform.get_device_caps(self.device.index)
            self._prefill = resolve_mla_prefill(self.spec, caps)

    def run_prefill_chunk(self, q, k, v, cu_q, cu_k, max_q, max_k, *, causal):
        if self._prefill is None:
            raise RuntimeError("MLA prefill was not prepared on a CUDA device")
        self._record_runtime_kernel_path(self._prefill.kernel_path)
        return self._prefill(
            q, k, v, cu_q, cu_k, max_q, max_k,
            scale=self.spec.softmax_scale, causal=causal,
        )

    def prefill_workspace_bytes(self, **shape):
        # Upstream FA3 owns its opaque workspace; startup profiling includes it.
        return 0 if self._prefill is None else self._prefill.workspace_bytes(**shape)

    @classmethod
    def bind(
        cls,
        spec: MlaAttentionOpSpec,
        caps: DeviceCaps,
        **kwargs,
    ) -> "MlaTritonProvider":
        if cls is not MlaTritonProvider:
            return cls(**kwargs)
        if caps.multi_processor_count is None or caps.multi_processor_count <= 0:
            raise ValueError("Triton MLA requires a positive SM count in DeviceCaps.")
        return cls(
            sm_count=caps.multi_processor_count,
            **kwargs,
        )

    def binding_metadata(self) -> dict[str, object]:
        metadata = {
            "implementation_kind": "atomic_provider",
            "implementation_source": "repo_triton",
            "decode_kernel_path": "triton_mla_stage1_stage2",
            "prefill": (
                operator_binding_report(self._prefill).as_dict()
                if self._prefill is not None else None
            ),
            "launch_config_source": (
                "explicit_config" if self._fixed_launch_config is not None else
                "sm_per_request_v1" if self._sm_count is not None else "portable_default"
            ),
            "sm_count": self._sm_count,
            "program_count": self.launch_config.program_count,
            "target_splits_per_request": self.launch_config.target_splits_per_request,
            "block_q_heads": self.launch_config.block_q_heads,
        }
        if not self.spec.cuda_graph:
            return metadata
        return {
            **metadata,
            "cuda_graph_mode": "batch_indexed",
            "context_capacity": self.spec.context_capacity,
            "launch_plan_source": "device_sm_local_heads" if self._sm_count is not None else "explicit_static_config",
        }

    def _record_runtime_kernel_path(self, path: str) -> None:
        counts = getattr(self, "_runtime_kernel_path_counts", None)
        if counts is None:
            counts = {}
            self._runtime_kernel_path_counts = counts
        path_counts = counts.setdefault(
            str(path),
            {"eager_dispatches": 0, "cuda_graph_capture_dispatches": 0},
        )
        key = (
            "cuda_graph_capture_dispatches"
            if device_runtime.is_stream_capturing()
            else "eager_dispatches"
        )
        path_counts[key] += 1

    def _record_runtime_fallback(self, reason: str) -> None:
        reasons = getattr(self, "_runtime_fallback_reasons", None)
        if reasons is None:
            reasons = {}
            self._runtime_fallback_reasons = reasons
        reasons[str(reason)] = int(reasons.get(str(reason), 0)) + 1

    def runtime_kernel_stats(self) -> dict[str, object]:
        paths = getattr(self, "_runtime_kernel_path_counts", {})
        reasons = getattr(self, "_runtime_fallback_reasons", {})
        return {
            "kernel_paths": {
                path: {key: int(value) for key, value in sorted(counts.items())}
                for path, counts in sorted(paths.items())
            },
            "fallback_reasons": {
                reason: int(count) for reason, count in sorted(reasons.items())
            },
        }

    @classmethod
    def _common_contract_support(
        cls,
        spec: MlaAttentionOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        common = match_attention_capabilities(
            spec.kernel_request, caps, cls.capabilities
        )
        if not common.supported:
            return common
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported(
                "decode CUDA Graph requires platform graph capture support"
            )
        if spec.cache_dtype != torch.bfloat16:
            return SupportResult.unsupported(
                f"requires BF16 cache storage, got {spec.cache_dtype}"
            )
        expected_shape = (
            _GLM_MLA_NUM_Q_HEADS,
            _GLM_MLA_KV_LORA_RANK,
            _GLM_MLA_ROPE_DIM,
            _GLM_MLA_QK_HEAD_DIM,
            _GLM_MLA_VALUE_HEAD_DIM,
        )
        actual_shape = (
            spec.num_q_heads,
            spec.kv_lora_rank,
            spec.rope_dim,
            spec.qk_head_dim,
            spec.value_head_dim,
        )
        if actual_shape != expected_shape:
            return SupportResult.unsupported(
                f"requires GLM MLA shape {expected_shape}, got {actual_shape}"
            )
        if spec.tp_size not in {1, 2, 4}:
            return SupportResult.unsupported(
                f"requires tensor parallel size 1, 2, or 4, got {spec.tp_size}"
            )
        return SupportResult.yes()

    @classmethod
    def supports(
        cls,
        spec: MlaAttentionOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        if spec.cuda_graph and spec.context_capacity is None:
            return SupportResult.unsupported("requires a static context capacity")
        return cls._common_contract_support(spec, caps)

    def _validate_run_inputs(
        self,
        q_nope_absorbed: torch.Tensor,
        q_rope: torch.Tensor,
        view: DecodeComputeView,
        output: torch.Tensor,
    ) -> MlaLatentPayload:
        if not isinstance(view, DecodeComputeView):
            raise TypeError(
                "MlaTritonProvider.run requires DecodeComputeView, got "
                f"{type(view).__name__}."
            )
        if not isinstance(view.payload, MlaLatentPayload):
            raise TypeError(
                "MLA decode requires MlaLatentPayload, got "
                f"{type(view.payload).__name__}."
            )
        if q_nope_absorbed.ndim != 3:
            raise ValueError(
                "q_nope_absorbed must have shape [batch, local_heads, 512], "
                f"got {tuple(q_nope_absorbed.shape)}."
            )
        expected_query_shape = (
            int(q_nope_absorbed.shape[0]),
            self.spec.local_q_heads,
            self.spec.kv_lora_rank,
        )
        if tuple(q_nope_absorbed.shape) != expected_query_shape:
            raise ValueError(
                "q_nope_absorbed must have shape "
                f"{expected_query_shape}, got {tuple(q_nope_absorbed.shape)}."
            )
        expected_rope_shape = (
            expected_query_shape[0],
            expected_query_shape[1],
            self.spec.rope_dim,
        )
        if tuple(q_rope.shape) != expected_rope_shape:
            raise ValueError(
                f"q_rope must have shape {expected_rope_shape}, got "
                f"{tuple(q_rope.shape)}."
            )
        if output.shape != q_nope_absorbed.shape:
            raise ValueError(
                f"output must have shape {tuple(q_nope_absorbed.shape)}, got "
                f"{tuple(output.shape)}."
            )
        if expected_query_shape[0] > self.max_batch_size:
            raise ValueError(
                "MLA decode batch exceeds the bound workspace: "
                f"batch={expected_query_shape[0]} max_batch_size="
                f"{self.max_batch_size}."
            )
        tensors = {
            "q_nope_absorbed": q_nope_absorbed,
            "q_rope": q_rope,
            "output": output,
            "latent_cache": view.payload.latent_cache,
            "rope_cache": view.payload.rope_cache,
        }
        for name, tensor in tensors.items():
            if tensor.device != self.device:
                raise ValueError(
                    f"{name} is on {tensor.device}, expected {self.device}."
                )
            expected_dtype = (
                self.spec.cache_dtype
                if name in {"latent_cache", "rope_cache"}
                else self.spec.activation_dtype
            )
            if tensor.dtype != expected_dtype:
                raise TypeError(
                    f"{name} must use {expected_dtype}, got {tensor.dtype}."
                )
        return view.payload

    def _validate_metadata(
        self,
        view: DecodeComputeView | PrefillComputeView,
        payload: MlaLatentPayload,
        *,
        validation_scope: object | None,
        valid_batch_size: int | None,
    ) -> None:
        cache_slot_count = int(payload.latent_cache.shape[0])
        metadata_key = (
            view.meta.active_slots,
            view.meta.req_indices,
            view.meta.context_lens,
            cache_slot_count,
            view.meta.max_context_len,
            valid_batch_size,
        )
        cached = self._validated_decode_metadata
        cached_entries = (
            cached[1]
            if validation_scope is not None
            and cached is not None
            and cached[0] is validation_scope
            else []
        )
        metadata_is_validated = any(
            entry[0] is metadata_key[0]
            and entry[1] is metadata_key[1]
            and entry[2] is metadata_key[2]
            and entry[3] == metadata_key[3]
            and entry[4] == metadata_key[4]
            and entry[5] == metadata_key[5]
            for entry in cached_entries
        )
        if metadata_is_validated:
            return
        validate_mla_decode_metadata(
            view.meta.active_slots,
            view.meta.req_indices,
            view.meta.context_lens,
            cache_slot_count=cache_slot_count,
            max_context_len=view.meta.max_context_len,
            valid_batch_size=valid_batch_size,
        )
        if validation_scope is None:
            self._validated_decode_metadata = None
        else:
            cached_entries.append(metadata_key)
            self._validated_decode_metadata = (validation_scope, cached_entries)

    def _launch_config_for(
        self,
        *,
        batch_size: int,
        max_context_len: int | None,
        active_slot_width: int,
    ) -> MlaDecodeLaunchConfig:
        del batch_size, max_context_len, active_slot_width
        return self.launch_config

    @torch.no_grad()
    def run(
        self,
        q_nope_absorbed: torch.Tensor,
        q_rope: torch.Tensor,
        view: DecodeComputeView,
        output: torch.Tensor,
        *,
        validation_scope: object | None = None,
        valid_batch_size: int | None = None,
    ) -> torch.Tensor:
        payload = self._validate_run_inputs(
            q_nope_absorbed,
            q_rope,
            view,
            output,
        )
        self._validate_metadata(
            view,
            payload,
            validation_scope=validation_scope,
            valid_batch_size=valid_batch_size,
        )
        launch_config = self._launch_config_for(
            batch_size=int(q_nope_absorbed.shape[0]),
            max_context_len=view.meta.max_context_len,
            active_slot_width=int(view.meta.active_slots.shape[1]),
        )
        self._record_runtime_kernel_path(
            "triton_score" if view.meta.attn_score is not None else "triton"
        )
        return run_mla_decode(
            q_nope_absorbed,
            q_rope,
            payload.latent_cache,
            payload.rope_cache,
            view.meta.active_slots,
            view.meta.req_indices,
            view.meta.context_lens,
            output,
            self.workspace,
            softmax_scale=self.spec.softmax_scale,
            attn_score=view.meta.attn_score,
            max_context_len=view.meta.max_context_len,
            config=launch_config,
            validate_metadata=False,
        )


@MLA_ATTENTION_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class MlaSglFa3Provider(MlaTritonProvider):
    """SGL FA3 decode with the score-producing Triton path kept explicit."""

    name = "sgl_fa3_sm90"
    supports_decode_graph = True

    def __init__(
        self,
        *,
        op_spec: MlaAttentionOpSpec,
        device: torch.device | str,
        max_batch_size: int,
        launch_config: MlaDecodeLaunchConfig | None = None,
    ) -> None:
        super().__init__(
            op_spec=op_spec,
            device=device,
            max_batch_size=max_batch_size,
            launch_config=launch_config,
        )
        self.fa3 = SglFa3DecodeKernel(
            device=self.device,
            max_batch_size=self.max_batch_size,
            softmax_scale=self.spec.softmax_scale,
        )

    @classmethod
    def supports(
        cls,
        spec: MlaAttentionOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        base = cls._common_contract_support(spec, caps)
        if not base.supported:
            return base
        if spec.score_output is not AttentionScoreKind.NONE:
            return SupportResult.unsupported(
                "does not satisfy the prepared score-output contract"
            )
        supported, reason = sgl_fa3_device_support(caps.device_index)
        return SupportResult.yes(reason) if supported else SupportResult.unsupported(reason)

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "implementation_source": "sglang-kernel",
            "prefill_kernel_path": "sgl_kernel.fa3.fwd",
            "decode_kernel_path": "sgl_kernel.fa3.fwd",
        }

    @torch.no_grad()
    def run(
        self,
        q_nope_absorbed: torch.Tensor,
        q_rope: torch.Tensor,
        view: DecodeComputeView,
        output: torch.Tensor,
        *,
        validation_scope: object | None = None,
        valid_batch_size: int | None = None,
    ) -> torch.Tensor:
        if view.meta.attn_score is not None:
            raise RuntimeError(
                "SGL FA3 MLA was bound for a score-free operation, but the "
                "runtime view requested attention scores."
            )
        payload = self._validate_run_inputs(
            q_nope_absorbed,
            q_rope,
            view,
            output,
        )
        self._validate_metadata(
            view,
            payload,
            validation_scope=validation_scope,
            valid_batch_size=valid_batch_size,
        )
        self._record_runtime_kernel_path("sgl_fa3")
        return self.fa3(
            q_rope,
            q_nope_absorbed,
            payload.rope_cache,
            payload.latent_cache,
            view.meta.active_slots,
            view.meta.req_indices,
            view.meta.context_lens,
            output,
            # Zero enables FA3's measured context-aware split heuristic.
            num_splits=0,
            validation_scope=validation_scope,
        )

    def use_compressed_prefill(self, plan, score_request) -> bool:
        # A runtime route between two prepared algorithms, not a decode step.
        # The expanded route retains sparse scoring and long-query efficiency.
        # Keep the conservative query crossover separate from FA3 eligibility.
        return (
            score_request is None
            and not plan.meta.is_sparse
            and max(b - a for a, b in zip(plan.query_starts, plan.query_starts[1:])) <= 384
        )

    @torch.no_grad()
    def run_compressed_prefill(self, q_latent, q_rope, view, plan):
        payload = view.payload
        output = torch.empty_like(q_latent, memory_format=torch.contiguous_format)
        self._record_runtime_kernel_path("sgl_fa3_prefill_latent")
        return self.fa3.run_varlen(
            q_rope, q_latent, payload.rope_cache, payload.latent_cache,
            view.meta.active_slots, view.meta.req_indices, view.meta.context_lens,
            output, cu_seqlens_q=plan.cu_q,
            max_seqlen_q=max(b - a for a, b in zip(plan.query_starts, plan.query_starts[1:])),
            validation_scope=plan.scope, return_softmax_lse=True,
        )

    @torch.no_grad()
    def run_prefill_chunk(self, q, k, v, cu_q, cu_k, max_q, max_k, *, causal):
        output = torch.empty((*q.shape[:2], v.shape[-1]), dtype=q.dtype, device=q.device)
        self._record_runtime_kernel_path("sgl_fa3_prefill_contiguous")
        return self.fa3.run_contiguous_explicit_varlen(
            q,
            k,
            v,
            output,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            causal=causal,
            return_softmax_lse=True,
        )


@MLA_ATTENTION_REGISTRY.register_atomic(
    ProviderRole.REPO_NONSTANDARD,
    profile_only=True,
)
class MlaTileLangScoreProvider(MlaSglFa3Provider):
    """Score-aware Composite over FA3 and statically planned TileLang."""

    name = "tilelang_score"
    supports_decode_graph = True

    def __init__(
        self,
        *,
        op_spec: MlaAttentionOpSpec,
        device: torch.device | str,
        max_batch_size: int,
        sm_count: int,
        launch_config: MlaDecodeLaunchConfig | None = None,
    ) -> None:
        super().__init__(
            op_spec=op_spec,
            device=device,
            max_batch_size=max_batch_size,
            launch_config=launch_config,
        )
        if self.spec.context_capacity is None:
            raise ValueError(
                "TileLang MLA requires a capture-time context capacity."
            )
        self.tilelang_launch_plan = TileMlaLaunchPlan.build(
            context_capacity=self.spec.context_capacity,
            local_q_heads=self.spec.local_q_heads,
            max_batch_size=self.max_batch_size,
            need_score=True,
            score_mode="per_head",
            sm_count=sm_count,
        )
        self.tilelang_score = TileMlaDecodeKernel(
            device=self.device,
            softmax_scale=self.spec.softmax_scale,
            valid_heads=self.spec.local_q_heads,
            launch_plan=self.tilelang_launch_plan,
        )

    @classmethod
    def bind(cls, spec: MlaAttentionOpSpec, caps: DeviceCaps, **kwargs):
        del spec
        if caps.multi_processor_count is None or caps.multi_processor_count <= 0:
            raise ValueError("TileLang MLA requires a positive SM count in DeviceCaps.")
        return cls(sm_count=caps.multi_processor_count, **kwargs)

    @classmethod
    def supports(
        cls,
        spec: MlaAttentionOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        base = cls._common_contract_support(spec, caps)
        if not base.supported:
            return base
        if spec.score_output is not AttentionScoreKind.RAW_QK_PER_HEAD:
            return SupportResult.unsupported(
                "requires the RAW_QK_PER_HEAD decode score contract"
            )
        if spec.context_capacity is None:
            return SupportResult.unsupported(
                "requires a capture-time context capacity"
            )
        supported, reason = sgl_fa3_device_support(caps.device_index)
        if not supported:
            return SupportResult.unsupported(reason)
        supported, reason = tilelang_mla_support()
        return SupportResult.yes(reason) if supported else SupportResult.unsupported(reason)

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "composite_provider",
            "implementation_source": "sglang-kernel+tilelang",
            "routes": {
                "score_free": "sgl_kernel.fa3.fwd",
                "raw_qk_per_head": "tilelang_mla_decode",
            },
            "tilelang_launch_plan": self.tilelang_launch_plan.metadata(),
        }

    def runtime_kernel_stats(self) -> dict[str, object]:
        return {
            **super().runtime_kernel_stats(),
            "tilelang": self.tilelang_score.runtime_metadata(),
        }

    def _validate_tilelang_score_contract(
        self,
        attn_score: torch.Tensor,
        *,
        max_context_len: int | None,
    ) -> None:
        if attn_score.ndim != 3:
            raise ValueError(
                "TileLang MLA RAW_QK_PER_HEAD score must have shape "
                f"[batch, heads, capacity], got {tuple(attn_score.shape)}."
            )
        if int(attn_score.shape[1]) != self.spec.local_q_heads:
            raise ValueError(
                "TileLang MLA score head count does not match the bound TP "
                f"shape: expected={self.spec.local_q_heads} "
                f"got={attn_score.shape[1]}."
            )
        if attn_score.dtype != torch.float32:
            raise TypeError(
                "TileLang MLA RAW_QK_PER_HEAD score must use FP32, got "
                f"{attn_score.dtype}."
            )
        if max_context_len is None or not 0 < int(max_context_len) <= int(
            attn_score.shape[2]
        ):
            raise ValueError(
                "TileLang MLA score capacity must cover max_context_len: "
                f"max={max_context_len} capacity={attn_score.shape[2]}."
            )

    @staticmethod
    def _tilelang_layout_rejection_reason(
        view: DecodeComputeView,
        output: torch.Tensor,
    ) -> str | None:
        if not isinstance(view.payload, MlaLatentPayload):
            return "payload_type"
        attn_score = view.meta.attn_score
        tensors = {
            "latent_cache": view.payload.latent_cache,
            "rope_cache": view.payload.rope_cache,
            "active_slots": view.meta.active_slots,
            "request_indices": view.meta.req_indices,
            "context_lens": view.meta.context_lens,
            "output": output,
        }
        rejected = [
            name
            for name, tensor in tensors.items()
            if not isinstance(tensor, torch.Tensor) or not tensor.is_contiguous()
        ]
        return None if not rejected else "noncontiguous:" + ",".join(rejected)

    @torch.no_grad()
    def run(
        self,
        q_nope_absorbed: torch.Tensor,
        q_rope: torch.Tensor,
        view: DecodeComputeView,
        output: torch.Tensor,
        *,
        validation_scope: object | None = None,
        valid_batch_size: int | None = None,
    ) -> torch.Tensor:
        attn_score = view.meta.attn_score
        if attn_score is None:
            return super().run(
                q_nope_absorbed,
                q_rope,
                view,
                output,
                validation_scope=validation_scope,
                valid_batch_size=valid_batch_size,
            )
        self._validate_tilelang_score_contract(
            attn_score,
            max_context_len=view.meta.max_context_len,
        )
        layout_rejection = self._tilelang_layout_rejection_reason(view, output)
        if layout_rejection is not None:
            raise ValueError(
                "TileLang MLA runtime view violates the bound layout contract: "
                f"{layout_rejection}."
            )
        payload = self._validate_run_inputs(
            q_nope_absorbed,
            q_rope,
            view,
            output,
        )
        self._validate_metadata(
            view,
            payload,
            validation_scope=validation_scope,
            valid_batch_size=valid_batch_size,
        )
        self._record_runtime_kernel_path("tilelang_score")
        return self.tilelang_score(
            q_nope_absorbed,
            q_rope,
            payload.latent_cache,
            payload.rope_cache,
            view.meta.active_slots,
            view.meta.req_indices,
            view.meta.context_lens,
            output,
            attn_score=attn_score,
            max_context_len=int(view.meta.max_context_len),
        )


@MLA_ATTENTION_REGISTRY.register_profile
class MlaTileLangScoreProfile:
    name = "tilelang_score_h100_profile"

    @classmethod
    def atomic_provider_names(cls, spec: MlaAttentionOpSpec) -> tuple[str, ...]:
        del spec
        return ("tilelang_score",)

    @classmethod
    def matches(
        cls,
        spec: MlaAttentionOpSpec,
        caps: DeviceCaps,
    ) -> ProfileMatch:
        del spec
        if not device_name_contains(caps.device_name, _PROFILED_H100_NAME):
            return ProfileMatch.no(
                f"requires profiled H100 hardware, got {caps.device_name}"
            )
        return ProfileMatch.yes("matched H100 TileLang MLA score profile")

    @classmethod
    def bind(cls, spec: MlaAttentionOpSpec, caps: DeviceCaps, **kwargs):
        return MlaTileLangScoreProvider.bind(spec, caps, **kwargs)


def resolve_mla_attention_provider(
    spec: MlaAttentionOpSpec,
    *,
    device: torch.device | str,
    max_batch_size: int,
    launch_config: MlaDecodeLaunchConfig | None = None,
) -> MlaAttentionProvider:
    """Resolve and bind an MLA provider during model construction."""

    device = torch.device(device)
    device_index = 0 if device.index is None else int(device.index)
    caps = platforms.current_platform.get_device_caps(device_index)
    return OpResolver(MLA_ATTENTION_REGISTRY).resolve(
        spec,
        caps,
        op_spec=spec,
        device=device,
        max_batch_size=max_batch_size,
        launch_config=launch_config,
    ).provider


__all__ = [
    "MLA_ATTENTION_REGISTRY",
    "MlaAttentionOpSpec",
    "MlaAttentionProvider",
    "MlaSglFa3Provider",
    "MlaTileLangScoreProvider",
    "MlaTritonProvider",
    "resolve_mla_attention_provider",
]
