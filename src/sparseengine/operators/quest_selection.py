from __future__ import annotations

from dataclasses import dataclass

import torch

import sparseengine.platforms as platforms
from sparseengine.kernels.external.flashinfer.topk import (
    flashinfer_top_k_page_table_transform,
    flashinfer_top_k_page_table_transform_support,
)
from sparseengine.operators.registry import (
    OpRegistry,
    OpResolver,
    PortfolioPolicy,
    ProfileMatch,
    ProviderRole,
    SupportResult,
)
from sparseengine.platforms import device_runtime
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum
from sparseengine.utils.device_name import device_name_contains


@dataclass(frozen=True)
class QuestPageSelectionOpSpec:
    score_dtype: torch.dtype
    cuda_graph: bool

    def __post_init__(self) -> None:
        if self.score_dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise TypeError(
                "QuEST page selection requires FP32, FP16, or BF16 scores, got "
                f"{self.score_dtype}."
            )


def _validate_inputs(
    spec: QuestPageSelectionOpSpec,
    scores: torch.Tensor,
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
) -> int:
    if scores.ndim != 2 or not scores.is_contiguous():
        raise ValueError("QuEST page scores must be contiguous and rank 2.")
    if scores.dtype != spec.score_dtype:
        raise TypeError(
            f"QuEST page selector expected {spec.score_dtype}, got {scores.dtype}."
        )
    if page_table.shape != scores.shape or page_table.dtype != torch.int32:
        raise TypeError(
            "QuEST page table must be int32 and match page scores, got "
            f"scores={tuple(scores.shape)} pages={tuple(page_table.shape)}/"
            f"{page_table.dtype}."
        )
    if not page_table.is_contiguous():
        raise ValueError("QuEST page table must be contiguous.")
    if lengths.shape != (int(scores.shape[0]),) or lengths.dtype != torch.int32:
        raise TypeError("QuEST page selection requires one int32 length per row.")
    if not lengths.is_contiguous():
        raise ValueError("QuEST page-selection lengths must be contiguous.")
    if page_table.device != scores.device or lengths.device != scores.device:
        raise ValueError("QuEST page-selection inputs must share a device.")
    k = int(k)
    if not 0 < k <= int(scores.shape[1]):
        raise ValueError(
            f"QuEST page selection requires 0 < k <= {scores.shape[1]}, got {k}."
        )
    return k


class QuestPageSelectionProvider:
    name = ""

    def __init__(self, *, op_spec: QuestPageSelectionOpSpec) -> None:
        self.spec = op_spec

    def select(
        self,
        scores: torch.Tensor,
        page_table: torch.Tensor,
        lengths: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def select_and_finalize_paged_view(
        self,
        scores: torch.Tensor,
        page_table: torch.Tensor,
        previous_page_counts: torch.Tensor,
        num_pages: torch.Tensor,
        context_lens: torch.Tensor,
        *,
        k: int,
        page_size: int,
        token_budget: int,
        outputs: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        use_dense_fallback: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from sparseengine.kernels.triton.quest_decode_view import (
            finalize_quest_paged_decode_view,
        )

        selected = self.select(scores, page_table, previous_page_counts, k)
        return finalize_quest_paged_decode_view(
            selected,
            page_table,
            num_pages,
            context_lens,
            page_size=page_size,
            token_budget=token_budget,
            output_page_table=outputs[0],
            output_req_indices=outputs[1],
            output_context_lens=outputs[2],
            output_page_counts=outputs[3],
            output_last_page_lens=outputs[4],
            use_dense_fallback=use_dense_fallback,
        )


QUEST_PAGE_SELECTION_REGISTRY: OpRegistry[
    QuestPageSelectionOpSpec,
    QuestPageSelectionProvider,
] = OpRegistry(
    "QuEST page selection",
    portfolio=PortfolioPolicy(
        upstream_standard=("flashinfer",),
        repo_portable=("torch",),
    ),
    profile_order=("h100_exact_paged_view_dispatch",),
)


@QUEST_PAGE_SELECTION_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class FlashInferQuestPageSelectionProvider(QuestPageSelectionProvider):
    name = "flashinfer"

    @classmethod
    def supports(
        cls,
        spec: QuestPageSelectionOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        if caps.platform != PlatformEnum.CUDA:
            return SupportResult.unsupported(f"requires CUDA, got {caps.platform.name}")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        supported, reason = flashinfer_top_k_page_table_transform_support(caps.device_index)
        return SupportResult.yes(reason) if supported else SupportResult.unsupported(reason)

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "implementation_source": "flashinfer-python",
            "kernel_path": "flashinfer.top_k_page_table_transform",
            "deterministic": True,
            "tie_break": "small",
            "cuda_graph": self.spec.cuda_graph,
        }

    def select(self, scores, page_table, lengths, k):
        k = _validate_inputs(self.spec, scores, page_table, lengths, k)
        if not scores.is_cuda:
            raise ValueError("FlashInfer QuEST page selection requires CUDA scores.")
        return flashinfer_top_k_page_table_transform(
            scores,
            page_table,
            lengths,
            k,
            cuda_graph=self.spec.cuda_graph,
        )


@QUEST_PAGE_SELECTION_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class TorchQuestPageSelectionProvider(QuestPageSelectionProvider):
    name = "torch"

    @classmethod
    def supports(
        cls,
        spec: QuestPageSelectionOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        del spec, caps
        return SupportResult.yes("Torch stable sort/gather baseline")

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "implementation_source": "torch",
            "kernel_path": "torch.argsort(stable=True)+torch.gather",
            "deterministic": True,
            "tie_break": "small",
        }

    def select(self, scores, page_table, lengths, k):
        k = _validate_inputs(self.spec, scores, page_table, lengths, k)
        columns = torch.arange(
            int(scores.shape[1]),
            dtype=torch.int32,
            device=scores.device,
        )
        valid = columns[None, :] < lengths[:, None]
        # Preserve the upstream contract on devices without enough shared
        # memory: ties prefer smaller columns, then emit in column order.
        top_indices = scores.masked_fill(~valid, -float("inf")).argsort(
            dim=-1, descending=True, stable=True,
        )[:, :k].sort(dim=-1).values
        selected = page_table.gather(1, top_indices)
        return selected.masked_fill(
            top_indices >= lengths[:, None].to(torch.long),
            -1,
        )


@QUEST_PAGE_SELECTION_REGISTRY.register_atomic(
    ProviderRole.REPO_NONSTANDARD,
    profile_only=True,
)
class TritonExactQuestPageSelectionProvider(QuestPageSelectionProvider):
    name = "triton_exact"

    @classmethod
    def supports(
        cls,
        spec: QuestPageSelectionOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        if caps.platform != PlatformEnum.CUDA:
            return SupportResult.unsupported(f"requires CUDA, got {caps.platform.name}")
        if not caps.supports_triton:
            return SupportResult.unsupported("platform does not support Triton")
        if spec.score_dtype != torch.bfloat16:
            return SupportResult.unsupported(
                f"requires BF16 scores, got {spec.score_dtype}"
            )
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        return SupportResult.yes("exact BF16 bitonic selection route")

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "implementation_source": "repo_triton",
            "kernel_path": "triton.quest_fused_selection",
            "deterministic": True,
            "tie_break": "small",
            "max_profiled_width": 512,
        }

    def select(self, scores, page_table, lengths, k):
        k = _validate_inputs(self.spec, scores, page_table, lengths, k)
        from sparseengine.kernels.triton.quest_fused_selection import (
            exact_select_quest_pages,
        )

        return exact_select_quest_pages(scores, page_table, lengths, k)

    def select_and_finalize_paged_view(
        self,
        scores,
        page_table,
        previous_page_counts,
        num_pages,
        context_lens,
        *,
        k,
        page_size,
        token_budget,
        outputs,
        use_dense_fallback,
    ):
        del num_pages
        _validate_inputs(self.spec, scores, page_table, previous_page_counts, k)
        from sparseengine.kernels.triton.quest_fused_selection import (
            fused_exact_select_quest_paged_view,
        )

        return fused_exact_select_quest_paged_view(
            scores,
            page_table,
            previous_page_counts,
            context_lens,
            k=k,
            page_size=page_size,
            token_budget=token_budget,
            use_dense_fallback=use_dense_fallback,
            output_page_table=outputs[0],
            output_req_indices=outputs[1],
            output_context_lens=outputs[2],
            output_page_counts=outputs[3],
            output_last_page_lens=outputs[4],
        )


@QUEST_PAGE_SELECTION_REGISTRY.register_profile
class H100ExactQuestPagedViewDispatch(QuestPageSelectionProvider):
    name = "h100_exact_paged_view_dispatch"

    @classmethod
    def atomic_provider_names(
        cls,
        spec: QuestPageSelectionOpSpec,
    ) -> tuple[str, ...]:
        del spec
        return ("triton_exact", "flashinfer")

    @classmethod
    def matches(
        cls,
        spec: QuestPageSelectionOpSpec,
        caps: DeviceCaps,
    ) -> ProfileMatch:
        if not device_name_contains(caps.device_name, "H100"):
            return ProfileMatch.no("requires profiled H100 hardware")
        if spec.score_dtype != torch.bfloat16:
            return ProfileMatch.no(f"requires BF16 scores, got {spec.score_dtype}")
        return ProfileMatch.yes("matched H100 exact QuEST paged-view profile")

    @classmethod
    def bind(
        cls,
        spec: QuestPageSelectionOpSpec,
        caps: DeviceCaps,
        **kwargs,
    ) -> H100ExactQuestPagedViewDispatch:
        del caps
        op_spec = kwargs.pop("op_spec", spec)
        if kwargs:
            raise TypeError(
                f"{cls.name} does not accept provider arguments: {sorted(kwargs)}"
            )
        if op_spec != spec:
            raise ValueError(f"{cls.name} received inconsistent operator specs.")
        return cls(op_spec=spec)

    def __init__(self, *, op_spec: QuestPageSelectionOpSpec) -> None:
        super().__init__(op_spec=op_spec)
        self.fused = TritonExactQuestPageSelectionProvider(op_spec=op_spec)
        self.fallback = FlashInferQuestPageSelectionProvider(op_spec=op_spec)
        self._route_counts: dict[str, dict[str, int]] = {}

    def _record_route(self, route: str) -> None:
        counts = self._route_counts.setdefault(
            route,
            {"eager_dispatches": 0, "cuda_graph_capture_dispatches": 0},
        )
        key = (
            "cuda_graph_capture_dispatches"
            if device_runtime.is_stream_capturing()
            else "eager_dispatches"
        )
        counts[key] += 1

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "dispatch_plan",
            "routes": [
                {
                    "condition": "sparse paged view, 0<k<width<=512",
                    "provider": self.fused.name,
                    "provider_metadata": self.fused.binding_metadata(),
                },
                {
                    "condition": "otherwise",
                    "provider": self.fallback.name,
                    "provider_metadata": self.fallback.binding_metadata(),
                },
            ],
        }

    def runtime_kernel_stats(self) -> dict[str, object]:
        return {
            "kernel_paths": {
                route: dict(counts)
                for route, counts in sorted(self._route_counts.items())
            },
            "fallback_reasons": {},
        }

    def select(self, scores, page_table, lengths, k):
        self._record_route("flashinfer")
        return self.fallback.select(scores, page_table, lengths, k)

    def select_and_finalize_paged_view(
        self,
        scores,
        page_table,
        previous_page_counts,
        num_pages,
        context_lens,
        *,
        k,
        page_size,
        token_budget,
        outputs,
        use_dense_fallback,
    ):
        use_fused = (
            not use_dense_fallback
            and 0 < int(k) < int(scores.shape[1])
            and int(scores.shape[1]) <= 512
        )
        provider = self.fused if use_fused else self.fallback
        self._record_route("fused" if use_fused else "flashinfer")
        return provider.select_and_finalize_paged_view(
            scores,
            page_table,
            previous_page_counts,
            num_pages,
            context_lens,
            k=k,
            page_size=page_size,
            token_budget=token_budget,
            outputs=outputs,
            use_dense_fallback=use_dense_fallback,
        )


def resolve_quest_page_selection_provider(
    spec: QuestPageSelectionOpSpec,
    *,
    device_index: int | None = None,
) -> QuestPageSelectionProvider:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    return OpResolver(QUEST_PAGE_SELECTION_REGISTRY).resolve(
        spec,
        caps,
        op_spec=spec,
    ).provider


__all__ = [
    "FlashInferQuestPageSelectionProvider",
    "H100ExactQuestPagedViewDispatch",
    "QUEST_PAGE_SELECTION_REGISTRY",
    "QuestPageSelectionOpSpec",
    "QuestPageSelectionProvider",
    "TorchQuestPageSelectionProvider",
    "TritonExactQuestPageSelectionProvider",
    "resolve_quest_page_selection_provider",
]
