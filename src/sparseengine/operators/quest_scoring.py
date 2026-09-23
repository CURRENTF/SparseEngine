from __future__ import annotations

from dataclasses import dataclass

import torch

import sparseengine.platforms as platforms
from sparseengine.operators.registry import (
    OpRegistry,
    OpResolver,
    PortfolioPolicy,
    ProviderRole,
    SupportResult,
)
from sparseengine.platforms import device_runtime
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum


@dataclass(frozen=True)
class QuestPageScoreSpec:
    dtype: torch.dtype
    query_heads: int
    kv_heads: int
    head_dim: int
    cuda_graph: bool

    def __post_init__(self):
        if self.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError("QuEST page scores require BF16, FP16, or FP32.")
        if min(self.query_heads, self.kv_heads, self.head_dim) <= 0 or self.query_heads % self.kv_heads:
            raise ValueError("QuEST scoring requires positive dimensions and query heads divisible by KV heads.")
        if self.head_dim > 65536:
            raise ValueError("QuEST scoring head dimension exceeds the Triton reduction limit.")


class QuestPageScoreProvider:
    name = ""

    def score(self, query, page_max, page_min, page_table):
        raise NotImplementedError


QUEST_PAGE_SCORE_REGISTRY = OpRegistry(
    "QuEST page scoring",
    portfolio=PortfolioPolicy(repo_nonstandard=("tiled", "triton")),
)


@QUEST_PAGE_SCORE_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class TritonQuestPageScoreProvider(QuestPageScoreProvider):
    name = "triton"

    @classmethod
    def supports(cls, spec, caps):
        if caps.platform not in (PlatformEnum.CUDA, PlatformEnum.ROCM) or not caps.supports_triton:
            return SupportResult.unsupported("requires CUDA or ROCm with Triton")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("requires graph capture support")
        return SupportResult.yes()

    def score(self, query, page_max, page_min, page_table):
        from sparseengine.kernels.triton.quest_decode_view import score_quest_pages
        return score_quest_pages(query, page_max, page_min, page_table)


@QUEST_PAGE_SCORE_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD, profile_only=True)
class TensorCoreQuestPageScoreProvider(TritonQuestPageScoreProvider):
    name = "triton_tensorcore"

    @classmethod
    def supports(cls, spec, caps):
        base = super().supports(spec, caps)
        if not base.supported:
            return base
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability is None or caps.compute_capability < (8, 0):
            return SupportResult.unsupported("requires CUDA SM80+ matrix-product lowering")
        if spec.dtype not in (torch.bfloat16, torch.float16):
            return SupportResult.unsupported("requires BF16 or FP16 inputs")
        if spec.dtype == torch.bfloat16 and not caps.supports_bfloat16:
            return SupportResult.unsupported("requires native BF16 support")
        return SupportResult.yes()

    def __init__(self, *, block_dim=128):
        self.block_dim = block_dim

    def score(self, query, page_max, page_min, page_table):
        from sparseengine.kernels.triton.quest_page_score import score_quest_pages_tensorcore
        return score_quest_pages_tensorcore(
            query, page_max, page_min, page_table, block_dim=self.block_dim,
        )


def _score_kernel_kind(spec, *, batch, pages, multiprocessors, tensorcore_supported):
    """Estimate reuse and work per SM; constants are calibrated, not support gates.

    Scalar scoring avoids an intermediate merge for short inputs. GQA reuses
    metadata across heads in MMA. Single-query bounds lack head reuse and keep the one-page reduction.
    Matrix products include a sparse repair pass for rounding-sensitive heads.
    """
    group = spec.query_heads // spec.kv_heads
    total_pages = batch * pages
    # DeviceCaps can omit this optional performance fact in capability-only
    # environments. Use a reference parallelism estimate, not a support failure.
    sms = multiprocessors or 128
    padded_dim = 1 << (spec.head_dim - 1).bit_length()
    if tensorcore_supported:
        if group > 1:
            parts = spec.kv_heads * ((group + 15) // 16)
            launch_work = 32768 if parts == 1 else 98304
            reuse = max(1, min(group / 8, 4))
            scalar_work = total_pages * spec.query_heads * padded_dim
            if scalar_work * reuse >= sms * launch_work:
                return "triton_tensorcore"
    # Without shared metadata, the tested page-tiled vector path trades fewer
    # CTAs for more register pressure and a head merge. Keep the one-page
    # reduction as the default; the tiled vector kernel remains a benchmark
    # candidate, with correctness eligibility independent of this cost policy.
    return "triton"


@QUEST_PAGE_SCORE_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class QuestPageScoreDispatch(TritonQuestPageScoreProvider):
    """Default nonstandard operator with deterministic, shape-based kernel choice."""
    name = "tiled"

    @classmethod
    def bind(cls, spec, caps):
        return cls(spec=spec, caps=caps)

    def __init__(self, *, spec: QuestPageScoreSpec, caps: DeviceCaps):
        self.spec = spec
        self.multiprocessors = caps.multi_processor_count
        support = TensorCoreQuestPageScoreProvider.supports(spec, caps)
        self.tensorcore_supported = support.supported
        self.tensorcore_rejection = None if support.supported else support.reason
        self.scalar = TritonQuestPageScoreProvider()
        self.tensorcore = TensorCoreQuestPageScoreProvider(
            block_dim=64 if spec.query_heads == spec.kv_heads and spec.head_dim >= 256 else 128,
        )
        self.counts = {}

    def binding_metadata(self):
        return {
            "implementation_kind": "shape_dispatch",
            "selection_policy": "reuse_and_work_per_sm",
            "multiprocessors": self.multiprocessors,
            "tensorcore_supported": self.tensorcore_supported,
            "tensorcore_rejection": self.tensorcore_rejection,
            "tensorcore_page_tile": 64,
            "tensorcore_dim_tile": self.tensorcore.block_dim,
            "evidence": "benchmark/kernel_profiles/quest_qwen3_h100/README.md",
        }

    def score(self, query, page_max, page_min, page_table):
        if query.ndim != 3 or tuple(query.shape[1:]) != (self.spec.query_heads, self.spec.head_dim):
            raise ValueError("QuEST score query does not match its bound head contract.")
        if query.dtype != self.spec.dtype or tuple(page_max.shape[1:]) != (self.spec.kv_heads, self.spec.head_dim):
            raise ValueError("QuEST score metadata/dtype does not match its bound contract.")
        if page_table.ndim != 2:
            raise ValueError("QuEST page table must have rank 2.")
        kind = _score_kernel_kind(
            self.spec, batch=int(query.shape[0]), pages=int(page_table.shape[1]),
            multiprocessors=self.multiprocessors, tensorcore_supported=self.tensorcore_supported,
        )
        provider = {"triton": self.scalar, "triton_tensorcore": self.tensorcore}[kind]
        counts = self.counts.setdefault(kind, {"eager_dispatches": 0, "cuda_graph_capture_dispatches": 0})
        key = "cuda_graph_capture_dispatches" if device_runtime.is_stream_capturing() else "eager_dispatches"
        counts[key] += 1
        return provider.score(query, page_max, page_min, page_table)

    def runtime_kernel_stats(self):
        return {"kernel_paths": self.counts, "fallback_reasons": {}}


def resolve_quest_page_score_provider(spec, *, device_index):
    caps = platforms.current_platform.get_device_caps(device_index)
    return OpResolver(QUEST_PAGE_SCORE_REGISTRY).resolve(spec, caps).provider


def score_quest_pages_batched(
    q_heads: torch.Tensor,
    page_max: torch.Tensor,
    page_min: torch.Tensor,
    num_metadata_heads: int,
) -> torch.Tensor:
    batch_size, num_heads, head_dim = q_heads.shape
    q_dtype = page_max.dtype
    if num_heads == num_metadata_heads:
        num_pages = page_max.shape[2]
        q_heads = q_heads.to(q_dtype)
        q_pos = q_heads.clamp_min(0).reshape(batch_size * num_heads, 1, head_dim)
        q_neg = q_heads.clamp_max(0).reshape(batch_size * num_heads, 1, head_dim)
        page_max_t = page_max.reshape(batch_size * num_heads, num_pages, head_dim).transpose(1, 2)
        page_min_t = page_min.reshape(batch_size * num_heads, num_pages, head_dim).transpose(1, 2)
        page_scores = torch.bmm(q_pos, page_max_t).squeeze(1)
        page_scores += torch.bmm(q_neg, page_min_t).squeeze(1)
        return page_scores.view(batch_size, num_heads, num_pages).amax(dim=1)

    if num_heads % num_metadata_heads:
        raise ValueError(
            "QuEST selection-query heads must be divisible by metadata heads: "
            f"query_heads={num_heads} metadata_heads={num_metadata_heads}."
        )
    group_size = num_heads // num_metadata_heads
    num_pages = page_max.shape[2]
    q_grouped = q_heads.view(
        batch_size,
        num_metadata_heads,
        group_size,
        head_dim,
    ).to(q_dtype)
    q_pos = q_grouped.clamp_min(0).reshape(
        batch_size * num_metadata_heads,
        group_size,
        head_dim,
    )
    q_neg = q_grouped.clamp_max(0).reshape(
        batch_size * num_metadata_heads,
        group_size,
        head_dim,
    )
    page_max_t = page_max.reshape(
        batch_size * num_metadata_heads,
        num_pages,
        head_dim,
    ).transpose(1, 2)
    page_min_t = page_min.reshape(
        batch_size * num_metadata_heads,
        num_pages,
        head_dim,
    ).transpose(1, 2)
    page_scores = torch.bmm(q_pos, page_max_t)
    page_scores += torch.bmm(q_neg, page_min_t)
    return page_scores.view(
        batch_size,
        num_metadata_heads,
        group_size,
        num_pages,
    ).amax(dim=2).amax(dim=1)
