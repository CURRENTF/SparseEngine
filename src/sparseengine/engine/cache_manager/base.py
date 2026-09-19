from __future__ import annotations

import os
import hashlib
from collections import deque
from dataclasses import dataclass, fields, is_dataclass
from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
import torch.distributed as dist

from sparseengine.config import Config
from sparseengine.distributed import ParallelContext
from sparseengine.engine.decode_graph_contract import (
    CacheDecodeGraphState,
    DecodeGraphContract,
    DecodeGraphInputs,
)
from sparseengine.engine.prefill import (
    PREFILL_EXECUTION_CHUNKED,
    PREFILL_EXECUTION_RAW_OFFLOAD,
)
from sparseengine.engine.sequence import Sequence
from sparseengine.method_registry import (
    SUPPORTED_SPARSE_METHODS,
    decode_graph_path_id,
    normalize_sparse_method,
    resolve_cache_sparse_method,
)
from sparseengine.kernels.triton.store_kvcache import store_kvcache
import sparseengine.platforms as platforms
from sparseengine.models.layout import resolve_attention_qk_head_dim
from sparseengine.utils.log import logger, log_level
from sparseengine.utils.profiler import profiler


def _debug_tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach().contiguous().cpu()
    raw = detached.reshape(-1).view(torch.uint8).numpy().tobytes()
    summary = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "numel": int(detached.numel()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if detached.numel() > 0 and not detached.is_complex():
        numeric = detached.to(torch.float64)
        summary.update(
            {
                "min": float(numeric.min().item()),
                "max": float(numeric.max().item()),
                "sum": float(numeric.sum().item()),
            }
        )
    return summary


def _debug_value_summary(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _debug_tensor_summary(value)
    if is_dataclass(value):
        return {
            field.name: _debug_value_summary(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): _debug_value_summary(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_debug_value_summary(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return _debug_value_summary(value.tolist())
    return repr(value)


@dataclass
class LayerBatchStates:
    """存储当前 Batch 在特定层的前向计算状态。

    仅包含与物理存储和基本前向元数据相关的字段。
    """

    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    max_context_len: int | None = None
    req_indices: torch.Tensor | None = None


@dataclass(frozen=True)
class JointPrefixCapacity:
    block_capacity: int
    kv_allocatable_bytes: int
    recurrent_capacity_bytes: int
    unallocated_bytes: int


def resolve_joint_prefix_capacity(
    *,
    available_bytes: int,
    kv_bytes_per_block: int,
    recurrent_bytes_per_block: int,
    requested_max_blocks: int | None,
) -> JointPrefixCapacity:
    available_bytes = int(available_bytes)
    kv_bytes_per_block = int(kv_bytes_per_block)
    recurrent_bytes_per_block = int(recurrent_bytes_per_block)
    if available_bytes < 0:
        raise ValueError(f"available_bytes must be >= 0, got {available_bytes}.")
    if kv_bytes_per_block <= 0 or recurrent_bytes_per_block <= 0:
        raise ValueError(
            "Joint prefix block bytes must be positive: "
            f"kv={kv_bytes_per_block} recurrent={recurrent_bytes_per_block}."
        )
    block_capacity = available_bytes // (kv_bytes_per_block + recurrent_bytes_per_block)
    if requested_max_blocks is not None:
        requested_max_blocks = int(requested_max_blocks)
        if requested_max_blocks <= 0:
            raise ValueError(
                f"requested_max_blocks must be positive when set, got {requested_max_blocks}."
            )
        block_capacity = min(block_capacity, requested_max_blocks)
    kv_allocatable_bytes = block_capacity * kv_bytes_per_block
    recurrent_capacity_bytes = block_capacity * recurrent_bytes_per_block
    return JointPrefixCapacity(
        block_capacity=int(block_capacity),
        kv_allocatable_bytes=int(kv_allocatable_bytes),
        recurrent_capacity_bytes=int(recurrent_capacity_bytes),
        unallocated_bytes=int(
            available_bytes - kv_allocatable_bytes - recurrent_capacity_bytes
        ),
    )


@dataclass
class SparseSelection:
    """Logical token selection produced by SparseController for one layer."""

    kind: str
    req_indices: torch.Tensor
    context_lens: torch.Tensor
    max_context_len: int | None = None
    attn_score: torch.Tensor | None = None
    active_indices: torch.Tensor | None = None
    active_slots: torch.Tensor | None = None
    active_compressed_indices: torch.Tensor | None = None
    global_req_indices: torch.Tensor | None = None
    chunk_lens: torch.Tensor | None = None
    release_temp_slots: bool = False


@dataclass(frozen=True)
class AttentionViewMeta:
    """Logical request/slot coordinates shared by attention payloads."""

    active_slots: torch.Tensor
    req_indices: torch.Tensor
    context_lens: torch.Tensor
    max_context_len: int | None = None
    attn_score: torch.Tensor | None = None
    temp_slots: torch.Tensor | None = None
    is_sparse: bool = False

    @property
    def slot_table_token_capacity(self) -> int:
        if self.active_slots.ndim != 2:
            raise ValueError("Attention slot-table capacity requires a rank-2 view.")
        return int(self.active_slots.shape[1])


@dataclass(frozen=True)
class PagedDecodeViewMeta:
    """Canonical physical-page coordinates for a paged decode provider."""

    page_table: torch.Tensor
    req_indices: torch.Tensor
    context_lens: torch.Tensor
    page_counts: torch.Tensor
    last_page_lens: torch.Tensor
    page_size: int
    is_sparse: bool = False
    max_context_len: int | None = None
    attn_score: torch.Tensor | None = None
    temp_slots: torch.Tensor | None = None

    @property
    def active_slots(self) -> torch.Tensor:
        """Compatibility name used by method-agnostic attention execution."""

        return self.page_table

    @property
    def slot_table_token_capacity(self) -> int:
        return int(self.page_table.shape[1]) * int(self.page_size)


@dataclass(frozen=True)
class ExplicitKVPayload:
    """Materialized key/value tensors consumed by ordinary attention."""

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    backend: str = "dense"
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class MlaLatentPayload:
    """Latent and RoPE caches consumed by MLA attention providers."""

    latent_cache: torch.Tensor
    rope_cache: torch.Tensor


AttentionPayload = ExplicitKVPayload | MlaLatentPayload


@dataclass(frozen=True)
class MlaLatentSelectionQuery:
    """Decode query expressed in the same coordinates as MLA latent storage."""

    latent: torch.Tensor
    rope: torch.Tensor

    def fused(self) -> torch.Tensor:
        if self.latent.ndim != 3 or self.rope.ndim != 3:
            raise ValueError(
                "MLA latent selection queries must have shape [batch, heads, dim], "
                f"got latent={tuple(self.latent.shape)} rope={tuple(self.rope.shape)}."
            )
        if self.latent.shape[:2] != self.rope.shape[:2]:
            raise ValueError(
                "MLA latent and RoPE selection queries must share batch/head axes, "
                f"got latent={tuple(self.latent.shape)} rope={tuple(self.rope.shape)}."
            )
        if self.latent.device != self.rope.device:
            raise ValueError(
                "MLA latent and RoPE selection queries must share a device, got "
                f"latent={self.latent.device} rope={self.rope.device}."
            )
        if self.latent.dtype != self.rope.dtype:
            raise TypeError(
                "MLA latent and RoPE selection queries must share a dtype, got "
                f"latent={self.latent.dtype} rope={self.rope.dtype}."
            )
        return torch.cat((self.latent, self.rope), dim=-1)


AttentionSelectionQuery = torch.Tensor | MlaLatentSelectionQuery


@dataclass(frozen=True)
class ExplicitKVWrite:
    """Current-token key/value tensors to persist."""

    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class MlaLatentWrite:
    """Current-token latent and RoPE tensors to persist."""

    latent: torch.Tensor
    rope: torch.Tensor


AttentionCacheWrite = ExplicitKVWrite | MlaLatentWrite


@dataclass(frozen=True)
class DecodeComputeView:
    """Decode metadata paired with exactly one physical payload layout."""

    meta: AttentionViewMeta | PagedDecodeViewMeta
    payload: AttentionPayload


@dataclass(frozen=True)
class PrefillScoreRequest:
    """Method-owned score semantics in each request's physical coordinates.

    Half-open query ranges select current queries; (0, 0) skips a request.
    Probability denominators use eligible causal keys after sink/recent masks.
    """

    query_ranges: tuple[tuple[int, int], ...]
    mode: str
    candidate_start: int = 0
    recent_keep_tokens: int = 0


@dataclass(frozen=True)
class PrefillComputeView:
    """Prefill metadata paired with exactly one physical payload layout."""

    meta: AttentionViewMeta
    payload: AttentionPayload
    token_scores: torch.Tensor | None = None  # Temporary [batch, max_context] scores.


@dataclass(frozen=True)
class AttentionKeyComputeView:
    """Physical payload plus the slots whose actual attention keys are needed."""

    active_slots: torch.Tensor
    payload: AttentionPayload


AttentionKeyMaterializer = Callable[[AttentionKeyComputeView], torch.Tensor]


class CacheManager(ABC):
    """每个 Rank 只有一个 CacheManager，内部管理所有层的物理槽位和 KV Cache。"""

    validate_runtime_invariants = False

    def __init__(
        self,
        config: Config,
        parallel_context: ParallelContext,
        *,
        allocation_budget_bytes: int | None = None,
    ):
        self.config = config
        self.validate_runtime_invariants = bool(
            getattr(config, "validate_runtime_invariants", False)
        )
        self.parallel_context = parallel_context
        self.rank = parallel_context.world_rank
        self.world_size = parallel_context.world_size
        self.tp_rank = parallel_context.attn_tp_rank
        self.tp_size = parallel_context.attn_tp_size
        self.ep_rank = parallel_context.moe_ep_rank
        self.ep_size = parallel_context.moe_ep_size
        self.dp_rank = parallel_context.attn_dp_rank
        self.dp_size = parallel_context.attn_dp_size
        self.platform = platforms.current_platform
        self.device = self.platform.get_device(self.rank)
        self.hf_config = config.hf_config
        self.num_layers = self.hf_config.num_hidden_layers
        self.runtime_layout = getattr(config, "runtime_layout", None)
        if self.runtime_layout is None:
            raise ValueError("CacheManager requires config.runtime_layout.")
        self.num_kv_layers = int(self.runtime_layout.num_kv_layers)
        self.allocation_budget_bytes = (
            None
            if allocation_budget_bytes is None
            else int(allocation_budget_bytes)
        )
        if (
            self.allocation_budget_bytes is not None
            and self.allocation_budget_bytes <= 0
        ):
            raise ValueError(
                "Cache allocation budget must be positive, got "
                f"{self.allocation_budget_bytes}."
            )

        layout_heads = tuple(getattr(self.runtime_layout, "kv_num_heads", ()))
        layout_dims = tuple(getattr(self.runtime_layout, "kv_head_dims", ()))
        self.num_kv_heads = (
            int(layout_heads[0]) // self.tp_size
            if layout_heads
            else int(self.hf_config.num_key_value_heads) // self.tp_size
        )
        self.head_dim = (
            int(layout_dims[0])
            if layout_dims
            else resolve_attention_qk_head_dim(self.hf_config)
        )

        self.max_model_len = config.max_model_len
        resident_buffer_rows = int(config.max_num_seqs_in_gpu)
        recurrent_row_capacity = getattr(
            config,
            "recurrent_state_row_capacity",
            None,
        )
        has_recurrent_layers = bool(
            getattr(self.runtime_layout, "linear_attention_layer_indices", ())
        )
        if has_recurrent_layers and recurrent_row_capacity is not None:
            recurrent_row_capacity = int(recurrent_row_capacity)
            if recurrent_row_capacity != resident_buffer_rows:
                raise RuntimeError(
                    "Cache-manager and recurrent live-row capacities disagree: "
                    f"cache_rows={resident_buffer_rows} "
                    f"recurrent_rows={recurrent_row_capacity}."
                )
            self.max_buffer_rows = recurrent_row_capacity
        else:
            self.max_buffer_rows = resident_buffer_rows

        self.kv_cache = None
        self._decode_static_max_context_len: int | None = None
        self._raw_offload_prefill_phases: dict[int, bool] = {}
        self._attention_key_materializers: dict[int, AttentionKeyMaterializer] = {}

    def synchronize_prefix_cache_delete_plan(
        self,
        local_plan: dict[str, object],
    ) -> None:
        if self.parallel_context.attn_tp_size <= 1:
            return
        plans: list[dict[str, object] | None] = [None] * self.parallel_context.attn_tp_size
        dist.all_gather_object(
            plans,
            local_plan,
            group=self.parallel_context.attn_tp.process_group,
        )
        if any(plan != plans[0] for plan in plans[1:]):
            raise RuntimeError(
                "Prefix-cache subtree deletion plan diverged across attention TP ranks: "
                f"plans={plans!r}."
            )

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        return bool(self.runtime_layout.is_full_attention(int(layer_idx)))

    def kv_layer_index(self, layer_idx: int) -> int:
        return int(self.runtime_layout.kv_layer_index(int(layer_idx)))

    def kv_transformer_layer_indices(self) -> tuple[int, ...]:
        return tuple(int(layer_idx) for layer_idx in self.runtime_layout.kv_idx_to_layer_idx)

    def _is_stream_capturing(self) -> bool:
        platform = getattr(self, "platform", None)
        if platform is not None:
            return platform.is_stream_capturing()
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())

    @staticmethod
    def create(
        config: Config,
        parallel_context: ParallelContext,
        *,
        allocation_budget_bytes: int | None = None,
    ) -> "CacheManager":
        def create_manager(manager_cls):
            if allocation_budget_bytes is None:
                return manager_cls(config, parallel_context)
            return manager_cls(
                config,
                parallel_context,
                allocation_budget_bytes=allocation_budget_bytes,
            )

        sparse_method = resolve_cache_sparse_method(
            config.sparse_method,
            prefill_sparse_method=getattr(config, "prefill_sparse_method", None),
        )
        if sparse_method not in SUPPORTED_SPARSE_METHODS:
            raise ValueError(f"Unsupported sparse_method={sparse_method!r}.")
        from sparseengine.method_registry import QUANTIZED_KV_METHODS

        if sparse_method in QUANTIZED_KV_METHODS:
            from .quantized import QuantizedCacheManager

            return create_manager(QuantizedCacheManager)
        if sparse_method == "deltakv":
            from .methods.deltakv_runtime import DeltaKVCacheManager

            return create_manager(DeltaKVCacheManager)
        if sparse_method in ("streamingllm", "attention-sink", "attention_sink"):
            from .methods.streamingllm import StreamingLLMCacheManager

            return create_manager(StreamingLLMCacheManager)
        if sparse_method in ("snapkv", "pyramidkv"):
            from .methods.snapkv import SnapKVCacheManager

            return create_manager(SnapKVCacheManager)
        if sparse_method == "kvzip":
            from .methods.kvzip import KVzipCacheManager

            return create_manager(KVzipCacheManager)
        if sparse_method == "h2o":
            from .methods.h2o import H2OCacheManager

            return create_manager(H2OCacheManager)
        if sparse_method == "rkv":
            from .methods.rkv import RKVCacheManager

            return create_manager(RKVCacheManager)
        if sparse_method == "skipkv":
            from .methods.skipkv import SkipKVCacheManager

            return create_manager(SkipKVCacheManager)
        if sparse_method == "quest":
            from .methods.quest import QuestCacheManager

            return create_manager(QuestCacheManager)
        if sparse_method == "omnikv":
            from .methods.omnikv.manager import OmniKVCacheManager

            return create_manager(OmniKVCacheManager)

        from .standard import StandardCacheManager

        return create_manager(StandardCacheManager)

    def _get_available_slots_info(self) -> tuple[int, int]:
        """返回 (可用显存字节数, 每层每 token 的字节数)"""
        config = self.config
        hf_config = config.hf_config
        slot_bytes_per_layer = self.attention_cache_bytes_per_slot_per_layer()
        allocation_budget_bytes = getattr(self, "allocation_budget_bytes", None)
        if allocation_budget_bytes is not None:
            available_memory, _, _ = self._resolve_joint_prefix_budget(
                int(allocation_budget_bytes),
                slot_bytes_per_layer,
            )
            return available_memory, slot_bytes_per_layer

        free, total = self.platform.get_available_memory(self.device.index or 0)

        # 动态估计 max_num_batched_tokens
        reserved_mem = total * (1 - config.gpu_memory_utilization)
        intermediate_size = getattr(hf_config, "intermediate_size", hf_config.hidden_size * 4)
        # Dense MLP activations are sharded only by tensor parallelism.
        intermediate_size_per_rank = intermediate_size // self.tp_size
        dtype_size = torch.tensor([], dtype=hf_config.dtype).element_size()

        # Keep this heuristic conservative: large prefill batches can still peak on
        # MLP/linear-attention projections and allocator fragmentation after KV
        # cache allocation.
        estimated_max_tokens = int(reserved_mem / (intermediate_size_per_rank * dtype_size * 16))
        allow_large_prefill_chunk = os.getenv("SPARSEENGINE_ALLOW_LARGE_PREFILL_CHUNK", "0") == "1"
        prefill_policy = getattr(config, "prefill_schedule_policy", None)
        if estimated_max_tokens <= 0:
            raise RuntimeError(
                "Estimated prefill token capacity must be positive: "
                f"estimated_max_tokens={estimated_max_tokens}."
            )
        engine_prefill_chunk_size = int(config.engine_prefill_chunk_size)
        long_prefill_offload_threshold = int(
            getattr(config, "long_prefill_offload_threshold", engine_prefill_chunk_size)
        )
        if (
            prefill_policy == "long_bs1full_short_batch"
            and not 0 < engine_prefill_chunk_size <= long_prefill_offload_threshold
        ):
            raise ValueError(
                "long_bs1full_short_batch requires 0 < engine_prefill_chunk_size <= "
                "long_prefill_offload_threshold after normalization: "
                f"engine_prefill_chunk_size={engine_prefill_chunk_size}, "
                f"long_prefill_offload_threshold={long_prefill_offload_threshold}."
            )
        if (
            prefill_policy == "long_bs1full_short_batch"
            and long_prefill_offload_threshold > estimated_max_tokens
        ):
            msg = (
                "long_prefill_offload_threshold="
                f"{long_prefill_offload_threshold} > "
                f"estimated_max_tokens={estimated_max_tokens} "
                f"(prefill_schedule_policy={prefill_policy!r})"
            )
            if allow_large_prefill_chunk:
                logger.warning(
                    "{}; continuing because SPARSEENGINE_ALLOW_LARGE_PREFILL_CHUNK=1. "
                    "This is an explicit experiment override and may OOM.",
                    msg,
                )
            else:
                logger.warning(
                    "{}; capping long_prefill_offload_threshold to {} and "
                    "engine_prefill_chunk_size to at most that value to avoid OOM.",
                    msg,
                    estimated_max_tokens,
                )
                config.long_prefill_offload_threshold = estimated_max_tokens
                config.engine_prefill_chunk_size = min(
                    int(config.engine_prefill_chunk_size),
                    estimated_max_tokens,
                )

        if estimated_max_tokens < config.max_num_batched_tokens and not allow_large_prefill_chunk:
            logger.warning(
                f"Estimated max_num_batched_tokens ({estimated_max_tokens}) is smaller than config "
                f"({config.max_num_batched_tokens}). Updating to avoid OOM."
            )
            config.max_num_batched_tokens = estimated_max_tokens
        elif estimated_max_tokens < config.max_num_batched_tokens:
            logger.warning(
                "Keeping max_num_batched_tokens={} above estimated {} because "
                "SPARSEENGINE_ALLOW_LARGE_PREFILL_CHUNK=1.",
                config.max_num_batched_tokens,
                estimated_max_tokens,
            )

        logger.info(f"Set dynamically max_num_batched_tokens = {config.max_num_batched_tokens}")

        used = total - free
        allocator_stats = self.platform.get_allocator_stats(self.device)
        peak = allocator_stats.peak_allocated_bytes
        current = allocator_stats.current_allocated_bytes

        target_persistent_bytes = int(total * config.gpu_memory_utilization)
        activation_reserve_bytes = int(total - target_persistent_bytes)
        recurrent_pool_bytes = int(getattr(config, "recurrent_state_pool_bytes", 0) or 0)
        recurrent_peak_before = int(
            getattr(config, "recurrent_state_allocator_peak_before_bytes", peak) or 0
        )
        recurrent_peak_after = int(
            getattr(config, "recurrent_state_allocator_peak_after_bytes", peak) or 0
        )
        recurrent_peak_growth = min(
            recurrent_pool_bytes,
            max(0, recurrent_peak_after - recurrent_peak_before),
        )
        recurrent_explicit_deduction = max(
            0,
            recurrent_pool_bytes - recurrent_peak_growth,
        )
        available_memory = int(
            target_persistent_bytes
            - used
            - peak
            + current
            - recurrent_explicit_deduction
        )
        available_memory, joint_capacity, kv_bytes_per_block = (
            self._resolve_joint_prefix_budget(
                available_memory,
                slot_bytes_per_layer,
            )
        )
        recurrent_bytes_per_block = int(
            getattr(config, "prefix_recurrent_bytes_per_block", 0) or 0
        )
        prefix_block_capacity = (
            0 if joint_capacity is None else int(joint_capacity.block_capacity)
        )
        prefix_recurrent_capacity_bytes = (
            0
            if joint_capacity is None
            else int(joint_capacity.recurrent_capacity_bytes)
        )

        model_current_bytes = max(0, int(current) - recurrent_pool_bytes)
        ratio = (
            0.0
            if kv_bytes_per_block <= 0
            else float(recurrent_bytes_per_block) / float(kv_bytes_per_block)
        )
        num_kv_layers = int(
            getattr(self, "num_kv_layers", getattr(self, "num_layers", 1))
        )
        kv_slots = (
            prefix_block_capacity * int(config.prefix_cache_block_size)
            if prefix_block_capacity > 0
            else available_memory // (num_kv_layers * slot_bytes_per_layer)
        )
        logger.info(
            "Persistent GPU budget: target={} model_current={} allocator_current={} "
            "device_used={} historical_peak={} recurrent_pool={} "
            "recurrent_peak_growth={} recurrent_explicit_deduction={} "
            "prefix_recurrent_capacity={} "
            "prefix_recurrent_to_kv_ratio={:.6f} activation_reserve={} "
            "kv_allocatable={} kv_slots={} prefix_blocks={}.",
            target_persistent_bytes,
            model_current_bytes,
            current,
            used,
            peak,
            recurrent_pool_bytes,
            recurrent_peak_growth,
            recurrent_explicit_deduction,
            prefix_recurrent_capacity_bytes,
            ratio,
            activation_reserve_bytes,
            available_memory,
            kv_slots,
            prefix_block_capacity,
        )

        if log_level == "DEBUG":
            logger.debug(
                f"[DEBUG] Available Memory: {available_memory / 1024**3:.2f} GB, "
                f"Slot Bytes Per Layer: {slot_bytes_per_layer / 1024**2:.4f} MB"
            )

        return available_memory, slot_bytes_per_layer

    def _resolve_joint_prefix_budget(
        self,
        available_memory: int,
        slot_bytes_per_layer: int,
    ) -> tuple[int, JointPrefixCapacity | None, int]:
        config = self.config
        recurrent_bytes_per_block = int(
            getattr(config, "prefix_recurrent_bytes_per_block", 0) or 0
        )
        if (
            str(getattr(config, "resolved_prefix_cache_mode", "disabled"))
            != "radix"
            or recurrent_bytes_per_block <= 0
        ):
            return int(available_memory), None, 0

        kv_bytes_per_block = self._kv_allocation_bytes_per_prefix_block(
            int(slot_bytes_per_layer)
        )
        requested_max_blocks = getattr(
            config,
            "prefix_cache_requested_max_blocks",
            getattr(config, "prefix_cache_max_blocks", None),
        )
        config.prefix_cache_requested_max_blocks = requested_max_blocks
        joint_capacity = resolve_joint_prefix_capacity(
            available_bytes=int(available_memory),
            kv_bytes_per_block=kv_bytes_per_block,
            recurrent_bytes_per_block=recurrent_bytes_per_block,
            requested_max_blocks=requested_max_blocks,
        )
        if joint_capacity.block_capacity <= 0:
            raise RuntimeError(
                "Insufficient GPU memory for one mixed prefix block: "
                f"available_bytes={available_memory} "
                f"kv_bytes_per_block={kv_bytes_per_block} "
                f"recurrent_bytes_per_block={recurrent_bytes_per_block}."
            )
        config.prefix_cache_max_blocks = int(joint_capacity.block_capacity)
        config.prefix_recurrent_capacity_bytes = int(
            joint_capacity.recurrent_capacity_bytes
        )
        config.prefix_kv_bytes_per_block = int(kv_bytes_per_block)
        config.prefix_kv_block_capacity = int(joint_capacity.block_capacity)
        config.kv_allocatable_bytes = int(joint_capacity.kv_allocatable_bytes)
        return (
            int(joint_capacity.kv_allocatable_bytes),
            joint_capacity,
            int(kv_bytes_per_block),
        )

    def attention_cache_bytes_per_slot_per_layer(self) -> int:
        """Persistent attention-cache bytes for one token in one KV layer."""
        dtype_size = self._cache_slot_dtype_size()
        return int(2 * self.num_kv_heads * self.head_dim * dtype_size)

    def _kv_allocation_bytes_per_prefix_block(
        self,
        slot_bytes_per_layer: int,
    ) -> int:
        block_size = int(self.config.prefix_cache_block_size or 0)
        if block_size <= 0:
            raise RuntimeError(
                "Mixed prefix capacity requires a positive prefix_cache_block_size."
            )
        return int(block_size * self.num_kv_layers * int(slot_bytes_per_layer))

    def prepare_step(self, seqs: list[Sequence], is_prefill: bool):
        if is_prefill:
            return self._prepare_prefill(seqs)
        return self._prepare_decode(seqs)

    def init_decode_graph_state(
        self,
        contract: DecodeGraphContract,
        inputs: DecodeGraphInputs,
    ) -> CacheDecodeGraphState:
        """Bind cache-owned metadata to one graph's stable public inputs."""
        inputs.validate(contract)
        return CacheDecodeGraphState(contract=contract, inputs=inputs)

    def prepare_decode_graph_step(
        self,
        seqs: list[Sequence],
        state: CacheDecodeGraphState,
    ):
        """Compatibility adapter while method-specific managers migrate."""
        inputs = state.inputs
        result = self.prepare_decode_static(
            seqs,
            inputs.input_ids,
            inputs.positions,
            inputs.write_slot_mapping,
            inputs.context_lens,
            inputs.request_indices,
        )
        real_batch_size = len(seqs)
        inputs.active_mask[:real_batch_size].fill_(True)
        inputs.active_mask[real_batch_size:].fill_(state.contract.padding.active)
        return result

    def prepare_decode_graph_in(self, state: CacheDecodeGraphState) -> None:
        """Run fixed device-side cache metadata preparation during capture/replay."""
        del state

    def decode_graph_state_keepalive_tensors(
        self,
        state: CacheDecodeGraphState,
    ) -> list[torch.Tensor]:
        del state
        return self.decode_graph_keepalive_tensors()

    @abstractmethod
    def allocate_kv_cache(self):
        """自动计算并物理分配 KV Cache 张量"""
        raise NotImplementedError

    @abstractmethod
    def get_layer_batch_states(self, layer_idx: int) -> LayerBatchStates:
        raise NotImplementedError

    @abstractmethod
    def get_layer_kv_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def get_layer_store_view(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def get_layer_compute_tensors(self, layer_idx: int, selection: SparseSelection | None = None):
        raise NotImplementedError

    def get_layer_store_tensors(
        self,
        layer_idx: int,
        *,
        k_post_rope: torch.Tensor,
        v: torch.Tensor,
        pre_rope_k: torch.Tensor | None = None,
        pre_rope_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the tensors that should be written to the layer's physical KV cache."""
        del layer_idx, pre_rope_k, pre_rope_v
        return k_post_rope, v

    def _store_layer_kv(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        k_cache, v_cache, slot_mapping = self.get_layer_store_view(layer_idx)
        if slot_mapping is None:
            raise RuntimeError(f"KV store requires slot_mapping at layer={layer_idx}.")
        if int(slot_mapping.numel()) != int(k.shape[0]):
            raise RuntimeError(
                "KV store shape mismatch: "
                f"layer={layer_idx} k={tuple(k.shape)} v={tuple(v.shape)} "
                f"slot_mapping={tuple(slot_mapping.shape)}."
            )
        store_kvcache(k, v, k_cache, v_cache, slot_mapping)
        return slot_mapping

    def store_attention_payload(
        self,
        layer_idx: int,
        payload: AttentionCacheWrite,
    ) -> torch.Tensor:
        """Store one layer's current-token payload using the configured layout."""
        if not isinstance(payload, ExplicitKVWrite):
            raise TypeError(
                f"{type(self).__name__} supports only ExplicitKVWrite stores, got "
                f"{type(payload).__name__}."
            )
        return self._store_layer_kv(
            layer_idx,
            payload.key,
            payload.value,
        )

    def save_raw_kv_if_needed(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """Optional pre-norm/pre-RoPE KV storage point."""
        del layer_idx, k, v
        return None

    def save_rope_kv_if_needed(
        self,
        layer_idx: int,
        k_post_rope: torch.Tensor,
        v: torch.Tensor,
    ):
        """Store the post-RoPE KV representation used by ordinary cache layouts."""
        store_k, store_v = self.get_layer_store_tensors(
            layer_idx,
            k_post_rope=k_post_rope,
            v=v,
        )
        slot_mapping = self.store_attention_payload(
            layer_idx,
            ExplicitKVWrite(key=store_k, value=store_v),
        )
        self.on_kv_stored(
            layer_idx,
            store_k,
            slot_mapping,
        )

    def get_layer_compute_view(
        self,
        layer_idx: int,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        selection: SparseSelection | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return KV tensors and logical view used by attention kernels."""
        try:
            k_cache, v_cache = self.get_layer_compute_tensors(layer_idx, selection)
        except NotImplementedError:
            k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        return k_cache, v_cache, active_slots, req_indices, context_lens

    def get_layer_compute_payload(
        self,
        layer_idx: int,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        selection: SparseSelection | None = None,
    ) -> tuple[AttentionPayload, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the physical payload and logical coordinates for decode."""
        k_cache, v_cache, active_slots, req_indices, context_lens = (
            self.get_layer_compute_view(
                layer_idx,
                active_slots,
                req_indices,
                context_lens,
                selection,
            )
        )
        return (
            ExplicitKVPayload(k_cache=k_cache, v_cache=v_cache),
            active_slots,
            req_indices,
            context_lens,
        )

    def register_attention_key_materializer(
        self,
        layer_idx: int,
        materializer: AttentionKeyMaterializer,
    ) -> None:
        """Bind a model/operator hook that reconstructs actual keys from a layout."""

        layer_idx = int(layer_idx)
        self.kv_layer_index(layer_idx)
        if not callable(materializer):
            raise TypeError(
                "Attention key materializer must be callable, got "
                f"{type(materializer).__name__}."
            )
        registry = getattr(self, "_attention_key_materializers", None)
        if registry is None:
            registry = {}
            self._attention_key_materializers = registry
        existing = registry.get(layer_idx)
        if existing is not None and existing != materializer:
            raise RuntimeError(
                "Attention key materializer is already bound for "
                f"layer={layer_idx}."
            )
        registry[layer_idx] = materializer

    def has_attention_key_materializer(self, layer_idx: int) -> bool:
        registry = getattr(self, "_attention_key_materializers", {})
        return int(layer_idx) in registry

    def build_attention_key_compute_view(
        self,
        layer_idx: int,
        active_slots: torch.Tensor,
    ) -> AttentionKeyComputeView:
        """Build a tagged view without assuming an explicit or latent layout."""

        layer_idx = int(layer_idx)
        kv_idx = self.kv_layer_index(layer_idx)
        storage = getattr(self, "attention_cache_storage", None)
        if storage is None:
            k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
            payload: AttentionPayload = ExplicitKVPayload(
                k_cache=k_cache,
                v_cache=v_cache,
            )
        else:
            payload = storage.layer_payload(kv_idx)
        return AttentionKeyComputeView(
            active_slots=active_slots,
            payload=payload,
        )

    @torch.no_grad()
    def materialize_attention_keys(
        self,
        layer_idx: int,
        active_slots: torch.Tensor,
    ) -> torch.Tensor:
        """Return the actual post-RoPE per-head keys for arbitrary cache slots."""

        view = self.build_attention_key_compute_view(layer_idx, active_slots)
        slots = view.active_slots
        if slots.ndim == 0:
            raise ValueError("Attention key slots must have at least one dimension.")
        if slots.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "Attention key slots must use int32 or int64, got "
                f"{slots.dtype}."
            )

        if isinstance(view.payload, ExplicitKVPayload):
            k_cache = view.payload.k_cache
            flat_slots = slots.to(device=k_cache.device, dtype=torch.long).reshape(-1)
            keys = k_cache.index_select(0, flat_slots).view(
                *slots.shape,
                *k_cache.shape[1:],
            )
        else:
            registry = getattr(self, "_attention_key_materializers", {})
            materializer = registry.get(int(layer_idx))
            if materializer is None:
                raise RuntimeError(
                    "The attention cache layout requires an actual-key "
                    f"materializer at layer={int(layer_idx)}."
                )
            keys = materializer(view)

        expected_ndim = int(slots.ndim) + 2
        if keys.ndim != expected_ndim or tuple(keys.shape[: slots.ndim]) != tuple(slots.shape):
            raise RuntimeError(
                "Attention key materializer returned an invalid shape: "
                f"slots={tuple(slots.shape)} keys={tuple(keys.shape)}; expected "
                "the slot shape followed by [heads, head_dim]."
            )
        if keys.device != slots.device:
            raise RuntimeError(
                "Materialized attention keys must share the slot device: "
                f"slots={slots.device} keys={keys.device}."
            )
        return keys

    def get_prefill_compute_view(
        self,
        layer_idx: int,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        selection: SparseSelection,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return KV tensors and logical view used by prefill attention kernels."""
        del k_current, v_current
        return self.get_layer_compute_view(
            layer_idx,
            active_slots,
            req_indices,
            context_lens,
            selection,
        )

    def get_prefill_compute_payload(
        self,
        layer_idx: int,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        selection: SparseSelection,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> tuple[AttentionPayload, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the physical payload and logical coordinates for prefill."""
        k_cache, v_cache, active_slots, req_indices, context_lens = (
            self.get_prefill_compute_view(
                layer_idx,
                k_current,
                v_current,
                selection,
                active_slots,
                req_indices,
                context_lens,
            )
        )
        return (
            ExplicitKVPayload(k_cache=k_cache, v_cache=v_cache),
            active_slots,
            req_indices,
            context_lens,
        )

    def _default_active_slots_for_selection(self, layer_idx: int, selection: SparseSelection) -> torch.Tensor:
        if selection.active_slots is not None:
            return selection.active_slots
        return self.get_layer_buffer_req_to_token_slots(layer_idx)

    def build_prefill_compute_view(
        self,
        layer_idx: int,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        selection: SparseSelection,
    ) -> PrefillComputeView:
        temp_slots = None
        if self.has_prefill_staging_view(layer_idx):
            active_slots, req_indices, context_lens, temp_slots = self.get_prefill_staging_view(layer_idx)
        elif self.has_full_layer_quantized_view(layer_idx):
            active_slots, req_indices, context_lens = self.build_full_layer_quantized_view(
                layer_idx,
                selection.req_indices,
                selection.context_lens,
            )
        else:
            active_slots = self._default_active_slots_for_selection(layer_idx, selection)
            req_indices = selection.req_indices
            context_lens = selection.context_lens
        payload, active_slots, req_indices, context_lens = self.get_prefill_compute_payload(
            layer_idx,
            k_current,
            v_current,
            selection,
            active_slots,
            req_indices,
            context_lens,
        )
        return PrefillComputeView(
            meta=AttentionViewMeta(
                active_slots=active_slots,
                req_indices=req_indices,
                context_lens=context_lens,
                attn_score=selection.attn_score,
                max_context_len=selection.max_context_len,
                temp_slots=temp_slots,
            ),
            payload=payload,
        )

    def prefill_score_request(self, layer_idx: int, seqs) -> PrefillScoreRequest | None:
        """Describe optional scores before attention without changing score state."""
        return None

    def collect_prefill_attention_score(
        self,
        layer_idx: int,
        q: torch.Tensor,
        view: PrefillComputeView,
        *,
        b_start_loc: torch.Tensor,
        chunk_lens: torch.Tensor,
        attention_lse: torch.Tensor | None = None,
    ):
        """Optional method-owned prefill score collection after attention output is computed."""
        del layer_idx, q, view, b_start_loc, chunk_lens, attention_lse
        return None

    def record_prefill_query(
        self,
        layer_idx: int,
        q: torch.Tensor,
        view: PrefillComputeView,
        *,
        b_start_loc: torch.Tensor,
        chunk_lens: torch.Tensor,
    ):
        """Optional method-owned prefill query cache update after attention output is computed."""
        del layer_idx, q, view, b_start_loc, chunk_lens
        return None

    def before_prefill_layer_attention(self, layer_idx: int, selection: SparseSelection):
        """Optional hook immediately before building a prefill layer compute view."""
        del selection
        coordinator = getattr(self, "prefix_cache_coordinator", None)
        if coordinator is not None:
            coordinator.before_prefill_layer_attention(int(layer_idx))
        return None

    def defer_prefill_eviction(self) -> bool:
        """Whether the current method should skip chunk-end sparse eviction."""
        return False

    def record_decode_query(self, layer_idx: int, q: torch.Tensor):
        """Optional method-owned decode query cache update after attention output is computed."""
        del layer_idx, q
        return None

    def pop_prefill_attention_score(self, layer_idx: int, seq: Sequence) -> torch.Tensor | None:
        """Return and clear a method-owned prefill score for one completed sequence."""
        del layer_idx, seq
        return None

    @abstractmethod
    def get_layer_buffer_req_to_token_slots(self, layer_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def on_kv_stored(
        self,
        layer_idx: int,
        k: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """Optional method-specific hook after KV has been written into cache."""
        return None

    def on_pre_rope_kv_stored(
        self,
        layer_idx: int,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        slot_mapping: torch.Tensor,
    ):
        """Optional hook for methods that need RoPE-independent KV metadata."""
        return None

    def on_layer_attention_end(self, layer_idx: int):
        """Optional method-specific hook after a layer's attention has consumed KV."""
        return None

    def release_layer_temp_slots(self, layer_idx: int, temp_slots: torch.Tensor | None):
        """Release temporary physical slots returned with a layer read/compute view."""
        del layer_idx, temp_slots
        return None

    def decode_graph_keepalive_tensors(self) -> list[torch.Tensor]:
        """Cache-manager-owned tensors captured by decode CUDA graphs."""
        return []

    def validate_decode_cuda_graph_slot_mappings(self) -> None:
        """Optionally validate every layer's static decode mapping together.

        Runtime validation is a debug invariant, not part of cache allocation.
        When enabled, graph capture calls it again after eager warmup because a
        storage backend may consume the one-forward prevalidation during that
        warmup. Sparse methods may bind a different stable mapping per layer,
        so storage receives the exact layer-ordered set.
        """
        if not self.validate_runtime_invariants:
            return
        storage = getattr(self, "attention_cache_storage", None)
        if storage is None:
            return
        slot_mappings: list[torch.Tensor] = []
        for layer_idx in self.kv_transformer_layer_indices():
            state = self.get_layer_batch_states(layer_idx)
            slot_mapping = state.slot_mapping
            if slot_mapping is None:
                raise RuntimeError(
                    "Decode CUDA graph capture requires a slot mapping for "
                    f"layer={layer_idx}."
                )
            slot_mappings.append(slot_mapping)
        with profiler.record("cache_validate_decode_slot_mappings"):
            storage.validate_slot_mappings(tuple(slot_mappings))

    def select_decode_cuda_graph_batch_size(
        self,
        real_batch_size: int,
        capture_sizes: list[int],
    ) -> int | None:
        """Optional method-specific graph batch-size selection.

        Return None to use the runner's standard capture-size buckets.
        """
        del real_batch_size, capture_sizes
        return None

    def decode_graph_path_id(self) -> str:
        return decode_graph_path_id(str(getattr(self.config, "sparse_method", "") or ""))

    def decode_graph_path_capacity(self) -> int:
        return int(self.config.max_model_len)

    def validate_decode_graph_path_capacity(
        self,
        seqs: list[Sequence],
        *,
        capacity: int,
    ) -> None:
        actual = max(int(seq.num_tokens) for seq in seqs)
        if int(capacity) < actual:
            raise RuntimeError(
                "decode CUDA Graph path capacity does not cover the "
                f"request: capacity={capacity}, actual={actual}."
            )

    def decode_graph_force_eager(self) -> bool:
        """Whether this method should bypass graph replay for diagnostics."""
        return False

    requires_committed_token_history = False

    def acquire_step_host_buffer(self, template: torch.Tensor) -> torch.Tensor:
        """Keep CPU upload storage exclusive until the submission is retired."""
        allocator = getattr(self, "_step_host_allocator", None)
        return template if allocator is None else allocator.acquire_host_tensor(template)

    def on_forward_end(self, seqs: list[Sequence], is_prefill: bool):
        """Optional hook after all layers have stored KV for a forward step."""
        if is_prefill:
            for seq in seqs:
                if seq.is_last_chunk_prefill:
                    self.complete_prefill_execution(seq)
        return None

    def prefix_cache_inspect(
        self,
        token_ids: list[int],
        *,
        include_subtree: bool = False,
    ) -> dict[str, object]:
        del token_ids, include_subtree
        raise RuntimeError("prefix cache is not enabled or not supported by this cache manager.")

    def prefix_cache_match(self, token_ids: list[int]) -> dict[str, object]:
        del token_ids
        return {
            "supported": False,
            "enabled": False,
            "matched_tokens": 0,
            "matched_blocks": 0,
            "match_ratio": 0.0,
            "reason": "prefix cache is not supported by this cache manager.",
        }

    def prefix_cache_delete_subtree(self, token_ids: list[int]) -> dict[str, object]:
        del token_ids
        raise RuntimeError("prefix cache is not enabled or not supported by this cache manager.")

    def prefix_cache_set_eviction_priority(
        self,
        token_ids: list[int],
        *,
        priority: int,
    ) -> dict[str, object]:
        del token_ids, priority
        raise RuntimeError("prefix cache is not enabled or not supported by this cache manager.")

    def prefix_cache_prune(
        self,
        token_ids: list[int],
        *,
        range_start: int,
        range_end: int,
        keep_indices: torch.Tensor,
        policy: str,
        prune_id: str,
        allow_recompress: bool = False,
    ) -> dict[str, object]:
        del (
            token_ids,
            range_start,
            range_end,
            keep_indices,
            policy,
            prune_id,
            allow_recompress,
        )
        raise RuntimeError(
            "physical prefix-cache pruning is unsupported by this cache manager; "
            "QuEST prefix cache remains available without pruning."
        )

    def validate_prefix_cache_prune_target(
        self,
        token_ids: list[int],
        *,
        range_start: int,
        range_end: int,
        allow_recompress: bool = False,
    ) -> list[object]:
        del token_ids, range_start, range_end, allow_recompress
        raise RuntimeError(
            "physical prefix-cache pruning is unsupported by this cache manager; "
            "QuEST prefix cache remains available without pruning."
        )

    def has_prefill_staging_view(self, layer_idx: int) -> bool:
        """Whether the current prefill layer should read from a temporary staging KV view."""
        return False

    def get_prefill_staging_view(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return (active_slots, req_indices, context_lens, temp_slots) for prefill staging."""
        raise NotImplementedError

    def has_full_layer_quantized_view(self, layer_idx: int) -> bool:
        """Whether a full-attention layer should read a reconstructed quantized KV view."""
        return False

    def build_full_layer_quantized_view(
        self,
        layer_idx: int,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (active_slots, local_req_indices, context_lens) for quantized full layers."""
        raise NotImplementedError

    def build_decode_view(
        self,
        layer_idx: int,
        q: AttentionSelectionQuery,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        *,
        num_heads: int,
        num_kv_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optional method-specific decode-time logical view builder."""
        return active_slots, req_indices, context_lens

    def build_decode_selection_query(
        self,
        q: torch.Tensor,
        *,
        mla_latent: torch.Tensor | None = None,
        mla_rope: torch.Tensor | None = None,
    ) -> AttentionSelectionQuery:
        """Translate an attention query into this cache manager's score space."""
        del mla_latent, mla_rope
        return q

    def build_decode_compute_view(
        self,
        layer_idx: int,
        q: AttentionSelectionQuery,
        selection: SparseSelection,
        *,
        num_heads: int,
        num_kv_heads: int,
    ) -> DecodeComputeView:
        if self.has_full_layer_quantized_view(layer_idx):
            active_slots, req_indices, context_lens = self.build_full_layer_quantized_view(
                layer_idx,
                selection.req_indices,
                selection.context_lens,
            )
        else:
            active_slots = self._default_active_slots_for_selection(layer_idx, selection)
            active_slots, req_indices, context_lens = self.build_decode_view(
                layer_idx,
                q,
                active_slots,
                selection.req_indices,
                selection.context_lens,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )
        payload, active_slots, req_indices, context_lens = (
            self.get_layer_compute_payload(
                layer_idx,
                active_slots,
                req_indices,
                context_lens,
                selection,
            )
        )
        max_context_len = selection.max_context_len
        if max_context_len is not None and active_slots.ndim >= 2:
            max_context_len = min(
                int(max_context_len),
                int(active_slots.shape[1]),
            )
        return DecodeComputeView(
            meta=AttentionViewMeta(
                active_slots=active_slots,
                req_indices=req_indices,
                context_lens=context_lens,
                attn_score=selection.attn_score,
                max_context_len=max_context_len,
            ),
            payload=payload,
        )

    def get_decode_block_seq(self, layer_idx: int, default: int) -> int:
        """Optional per-layer decode stage block size override."""
        return int(default)

    def set_decode_static_max_context_len(self, max_context_len: int):
        """Pin graph-captured decode kernels to a fixed max context length."""
        max_context_len = int(max_context_len)
        self._decode_static_max_context_len = max_context_len
        layer_batch_state = getattr(self, "layer_batch_state", None)
        if layer_batch_state is not None:
            layer_batch_state.max_context_len = max_context_len
        layer_batch_states = getattr(self, "layer_batch_states", None)
        if layer_batch_states is not None:
            for state in layer_batch_states:
                state.max_context_len = max_context_len
        for attr_name in ("full_layer_batch_states", "deltakv_layer_batch_states"):
            state = getattr(self, attr_name, None)
            if state is not None:
                state.max_context_len = max_context_len

    @property
    @abstractmethod
    def num_free_slots(self) -> int:
        raise NotImplementedError

    def num_free_slots_full_layers(self) -> int:
        """Free slots in the KV pool that is not subject to sparse eviction.

        Default behavior: treat `num_free_slots` as the only pool.
        DeltaKV overrides this to expose the full-attention pool capacity, which
        bounds how many long prompts can be admitted without thrashing.
        """
        return self.num_free_slots

    # ---- Scheduler hooks (default implementations) ----
    def prefill_batched_tokens_margin(self) -> int:
        """Extra headroom the scheduler should leave in `max_num_batched_tokens` for this cache manager."""
        return 0

    def remaining_prefill_tokens(self, seq: Sequence) -> int:
        """Effective remaining prefill tokens for scheduling decisions."""
        virtual_prefilled = max(
            int(seq.num_prefilled_tokens),
            int(getattr(seq, "prefix_cache_hit_len", 0) or 0),
        )
        return int(seq.num_prompt_tokens - virtual_prefilled)

    def prefill_execution_mode(self, seq: Sequence) -> str:
        """Return the method-owned execution contract for the remaining prompt."""
        del seq
        return PREFILL_EXECUTION_CHUNKED

    def _apply_sticky_raw_offload_mode(self, seq: Sequence, mode: str) -> str:
        """Keep RawKV staging active for one logical prefill lifecycle."""
        phases = getattr(self, "_raw_offload_prefill_phases", None)
        if phases is None:
            phases = {}
            self._raw_offload_prefill_phases = phases
        seq_id = int(seq.seq_id)
        replay_phase = bool(getattr(seq, "is_recompute_replay", False))
        previous_phase = phases.get(seq_id)
        if previous_phase is not None and previous_phase != replay_phase:
            phases.pop(seq_id, None)
        if seq_id in phases:
            return PREFILL_EXECUTION_RAW_OFFLOAD
        if mode == PREFILL_EXECUTION_RAW_OFFLOAD:
            phases[seq_id] = replay_phase
        return mode

    def reset_prefill_execution_state(self, seq_id: int) -> None:
        phases = getattr(self, "_raw_offload_prefill_phases", None)
        if phases is not None:
            phases.pop(int(seq_id), None)

    def complete_prefill_execution(self, seq: Sequence) -> None:
        self.reset_prefill_execution_state(int(seq.seq_id))

    def prefill_batch_compatibility_key(self, seq: Sequence) -> object:
        """Return a method-owned key for prefill requests that may share a batch."""
        del seq
        return None

    def reserved_prefill_slots(self, waiting_seqs: deque[Sequence], engine_prefill_chunk_size: int) -> int:
        """Persistent slots reserved by waiting/running prefills.

        This must not include temporary staging KV or decode reconstruction scratch.
        """
        reserved = 0
        for seq in waiting_seqs:
            if 0 < seq.num_prefilled_tokens < seq.num_prompt_tokens:
                reserved += int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
        return reserved

    def prefill_step_free_slots(self) -> int:
        """Writable KV capacity for the current prefill step.

        Temporary pools with a different lifetime should expose their own accounting
        instead of being mixed into this persistent step capacity.
        """
        return int(self.num_free_slots)

    def should_schedule_full_prefill(self, seq: Sequence) -> bool:
        """Whether scheduler should route this first prefill as a full bs1 step."""
        return False

    def requires_full_prefill_step(self, seq: Sequence) -> bool:
        """Whether this prefill candidate must run its remaining tokens in one step."""
        return False

    def prefill_step_free_slots_for(self, seq: Sequence) -> int:
        """Writable KV capacity for a specific prefill candidate."""
        return int(self.prefill_step_free_slots())

    def prefill_private_slots_for(self, seq: Sequence) -> int:
        """Writable tokens already owned by this request, outside the shared budget."""
        return 0

    def min_final_prefill_chunk_size(self, seq: Sequence) -> int:
        """Minimum final chunk size required by method-specific prefill logic."""
        del seq
        return 0

    def prefill_staging_context_lens_cpu(
        self,
        layer_idx: int,
    ) -> tuple[int, ...] | None:
        """CPU mirror of staging lengths for synchronization-free finalization."""
        del layer_idx
        return None

    def requires_long_prefill_offload(self, seq: Sequence) -> bool:
        """Whether this long prefill should be internally chunked through offload staging."""
        del seq
        return False

    def prefill_step_reservation_cost(self, seq: Sequence, scheduled_tokens: int) -> int:
        """Scheduler-side capacity consumed by scheduling a prefill chunk."""
        return int(scheduled_tokens)

    def decode_window_budgets(self) -> dict[str, int]:
        return {"slots": int(self.prompt_admission_free_slots())}

    def decode_window_costs(self, seq: Sequence, tokens: int) -> dict[str, int]:
        return {"slots": int(tokens)}

    def prefill_capacity_after_decode_reservations(
        self, free_slots: int, reserved: dict[str, int], *, admission: bool,
    ) -> int:
        """Deduct token-slot headroom from a scalar prefill capacity.

        The default covers one slot pool or homogeneous per-layer slot pools.
        Mixed raw/compressed pools must project their own capacity. Return the
        unclamped balance so RuntimeState can add reclaimable prefix slots.
        """
        return int(free_slots) - max(reserved.values(), default=0)

    def decode_step_free_slots(self) -> int:
        """Writable KV capacity for one decode step."""
        return int(self.num_free_slots)

    def decode_step_free_slots_for(self, seq: Sequence) -> int:
        """Writable KV capacity for a specific decode candidate."""
        return int(self.decode_step_free_slots())

    def decode_step_reservation_cost(self, seq: Sequence) -> int:
        """Scheduler-side capacity consumed by scheduling one decode token."""
        return 1

    def prompt_admission_free_slots(self) -> int:
        """Slots pool used to decide whether a new prompt can be admitted."""
        return int(self.num_free_slots)

    def prompt_admission_cost(self, seq: Sequence) -> int:
        """Persistent slots needed to admit a complete prompt to its final representation."""
        hit_len = int(getattr(seq, "prefix_cache_hit_len", 0) or 0)
        return int(seq.num_prompt_tokens - hit_len)

    def prompt_logical_reservation_cost(self, seq: Sequence) -> int:
        """Logical slots reserved when a new prompt is admitted (scheduler-side accounting)."""
        return int(self.prompt_admission_cost(seq))

    def prompt_admission_failure_action(self) -> str:
        """Action when a prompt cannot be admitted: 'raise' or 'defer'."""
        return "defer"

    def prompt_admission_budgets(
        self,
        waiting_seqs: deque[Sequence],
        engine_prefill_chunk_size: int,
    ) -> dict[str, int]:
        """Return admission budgets used by Scheduler for new prompts.

        Default behavior merges the reserved-prefill headroom into the same
        budget that gates new-prompt admission. This keeps the first budget
        check aligned with the later logical reservation accounting.
        """
        reserved = int(self.reserved_prefill_slots(waiting_seqs, engine_prefill_chunk_size))
        free_slots = int(self.prompt_admission_free_slots())
        return {"slots": max(0, free_slots - reserved)}

    def prompt_admission_costs(self, seq: Sequence) -> dict[str, int]:
        """Return persistent final-representation admission costs per budget."""
        return {"slots": int(self.prompt_admission_cost(seq))}

    def on_prompt_admitted(self, seq: Sequence, costs: dict[str, int]):
        """Hook called when Scheduler admits a new prompt."""
        return

    def refresh_prefix_cache_hit(self, seq: Sequence) -> None:
        """Populate scheduler-visible prefix hit metadata for a fresh prompt."""
        self.clear_prefix_cache_hit(seq)

    def clear_prefix_cache_hit(self, seq: Sequence) -> None:
        """Clear scheduler-visible prefix hit metadata."""
        seq.clear_prefix_cache_hit()

    def build_prefix_kv_payload(self, seq: Sequence, block_start: int, block_end: int) -> object:
        del seq, block_start, block_end
        raise RuntimeError("This cache manager does not support mixed prefix KV payloads.")

    def attach_prefix_kv_payload(self, seq: Sequence, payload: object) -> None:
        del seq, payload
        raise RuntimeError("This cache manager does not support mixed prefix KV payload attach.")

    def validate_prefix_kv_attach(self, seq: Sequence) -> bool:
        del seq
        raise RuntimeError("This cache manager does not support mixed prefix KV payload attach.")

    def rollback_prefix_kv_attach(
        self,
        seq: Sequence,
        payloads: list[object],
        *,
        row_preexisted: bool,
    ) -> None:
        del seq, payloads, row_preexisted
        raise RuntimeError("This cache manager does not support mixed prefix KV attach rollback.")

    def free_prefix_kv_payload(self, payload: object) -> None:
        del payload
        raise RuntimeError("This cache manager does not support mixed prefix KV payload free.")

    def allocate_prefix_kv_payload_device(self, payload: object) -> None:
        del payload
        raise RuntimeError("This cache manager does not support mixed prefix KV promotion.")

    def allocate_prefix_kv_payloads_device(self, payloads: list[object]) -> None:
        for payload in payloads:
            self.allocate_prefix_kv_payload_device(payload)

    def free_prefix_kv_payload_device(self, payload: object) -> None:
        del payload
        raise RuntimeError("This cache manager does not support mixed prefix KV demotion.")

    def prefix_kv_payload_nbytes(self, payload: object) -> int:
        del payload
        raise RuntimeError("This cache manager does not support mixed prefix KV payload accounting.")

    def mark_materialized_prefix_kv_payload(self, seq: Sequence, payload: object) -> None:
        del seq, payload
        return None

    def free_slot_stats(self) -> dict[str, int]:
        """Return a small set of free-slot stats for logging/debugging."""
        return {"free_slots": int(self.num_free_slots)}

    def _debug_token_slots_for_mapping(
        self,
        layer_idx: int | None,
    ) -> torch.Tensor:
        token_slots = getattr(self, "buffer_req_to_token_slots")
        return token_slots if layer_idx is None else token_slots[layer_idx]

    def debug_state_summary(self) -> dict[str, Any]:
        """Return a synchronized-test snapshot without touching the inference hot path."""
        live_rows = {}
        seq_id_to_row = getattr(self, "seq_id_to_row", {})
        mappings = (
            [(layer_idx, mapping) for layer_idx, mapping in enumerate(seq_id_to_row)]
            if isinstance(seq_id_to_row, list)
            else [(None, seq_id_to_row)]
        )
        for layer_idx, mapping in mappings:
            if not isinstance(mapping, dict) or not mapping:
                continue
            row_seq_lens = getattr(self, "row_seq_lens")
            token_slots = self._debug_token_slots_for_mapping(layer_idx)
            if layer_idx is not None:
                row_seq_lens = row_seq_lens[layer_idx]
            records = []
            for seq_id, row_idx in sorted(mapping.items()):
                row_len = int(row_seq_lens[row_idx])
                record = {
                    "seq_id": int(seq_id),
                    "row_idx": int(row_idx),
                    "row_len": row_len,
                    "token_slots": _debug_tensor_summary(
                        token_slots[row_idx, :row_len]
                    ),
                }
                page_slots = getattr(self, "buffer_req_to_page_slots", None)
                if layer_idx is None and page_slots is not None:
                    page_size = int(getattr(self, "page_size"))
                    num_pages = (row_len + page_size - 1) // page_size
                    record["page_slots"] = _debug_tensor_summary(
                        page_slots[row_idx, :num_pages]
                    )
                records.append(record)
            live_rows["shared" if layer_idx is None else str(layer_idx)] = records

        prefix_state: dict[str, Any] | None = None
        prefix_cache = getattr(self, "prefix_cache", None)
        if prefix_cache is not None:
            blocks = []
            for block_id, block in sorted(prefix_cache.blocks.items()):
                blocks.append(
                    {
                        "stable_block_id": block_id.hex(),
                        "parent_block_id": (
                            None
                            if block.parent_block_id is None
                            else block.parent_block_id.hex()
                        ),
                        "logical_block_idx": int(block.logical_block_idx),
                        "token_ids": [int(token_id) for token_id in block.token_ids],
                        "ref_count": int(block.ref_count),
                        "last_access": int(block.last_access),
                        "eviction_priority": int(block.eviction_priority),
                        "device_present": bool(block.residency.device_present),
                        "host_present": bool(block.residency.host_present),
                        "transfer": (
                            None
                            if block.residency.transfer is None
                            else block.residency.transfer.value
                        ),
                        "payload": _debug_value_summary(block.payload),
                    }
                )
            prefix_state = {
                "fingerprint": prefix_cache.fingerprint.hex(),
                "stats": {
                    str(key): int(value)
                    for key, value in prefix_cache.stats().items()
                },
                "blocks": blocks,
            }

        return {
            "cache_manager_class": type(self).__name__,
            "free_slot_stats": {
                str(key): int(value) for key, value in self.free_slot_stats().items()
            },
            "live_rows": live_rows,
            "prefix_cache": prefix_state,
        }

    def _cache_slot_dtype_size(self) -> int:
        hf_config = getattr(self, "hf_config", getattr(self.config, "hf_config", None))
        dtype = hf_config.dtype
        if not isinstance(dtype, torch.dtype):
            dtype = torch.float16
        return int(torch.tensor([], dtype=dtype).element_size())

    def _dense_baseline_slots(self) -> int:
        slot_candidates = [
            getattr(self.config, "num_kvcache_slots", None),
            getattr(self, "num_slots", None),
            getattr(self, "full_num_slots", None),
            getattr(self, "deltakv_latent_num_slots", None),
            getattr(self, "deltakv_full_num_slots", None),
        ]
        slots = [int(value) for value in slot_candidates if isinstance(value, (int, float)) and int(value) > 0]
        if slots:
            return max(slots)
        return int(getattr(self.config, "max_num_seqs_in_gpu", 1)) * int(self.max_model_len)

    def _dense_baseline_bytes(self) -> int:
        dtype_size = self._cache_slot_dtype_size()
        slots = self._dense_baseline_slots()
        storage = getattr(self, "attention_cache_storage", None)
        layout = getattr(storage, "layout", None)
        layout_name = str(getattr(layout, "value", layout) or "")
        if layout_name == "mla_latent":
            hf_config = self.hf_config
            global_heads = int(hf_config.num_attention_heads)
            attention_tp_size = int(
                getattr(
                    getattr(self, "parallel_context", None),
                    "attn_tp_size",
                    getattr(self, "tp_size", 1),
                )
            )
            if global_heads % attention_tp_size != 0:
                raise ValueError(
                    "MLA dense baseline requires attention heads divisible by TP: "
                    f"heads={global_heads} tp={attention_tp_size}."
                )
            local_heads = global_heads // attention_tp_size
            qk_head_dim = resolve_attention_qk_head_dim(hf_config)
            value_head_dim = int(hf_config.v_head_dim)
            values_per_token = local_heads * (qk_head_dim + value_head_dim)
        else:
            values_per_token = 2 * self.num_kv_heads * self.head_dim
        return int(
            slots
            * self.num_kv_layers
            * values_per_token
            * dtype_size
        )

    @staticmethod
    def _tensor_storage_key(tensor: torch.Tensor) -> tuple[Any, ...]:
        storage = tensor.untyped_storage()
        return (
            str(tensor.device),
            int(storage.data_ptr()),
            int(storage.nbytes()),
        )

    @staticmethod
    def _tensor_storage_nbytes(tensor: torch.Tensor) -> int:
        return int(tensor.untyped_storage().nbytes())

    def _iter_accounting_tensors(self):
        seen_containers: set[int] = set()

        def visit(path: str, value):
            if torch.is_tensor(value):
                yield path, value
                return
            if value is None or isinstance(value, (str, bytes, int, float, bool)):
                return
            obj_id = id(value)
            if obj_id in seen_containers:
                return
            seen_containers.add(obj_id)
            if isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(key, (str, int)):
                        child_path = f"{path}.{key}"
                    else:
                        child_path = f"{path}.{type(key).__name__}"
                    yield from visit(child_path, item)
                return
            if isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    yield from visit(f"{path}.{idx}", item)
                return
            if isinstance(value, LayerBatchStates):
                for field_name, item in value.__dict__.items():
                    yield from visit(f"{path}.{field_name}", item)

        storage = getattr(self, "attention_cache_storage", None)
        if storage is not None:
            layout = getattr(getattr(storage, "layout", None), "value", "unknown")
            for index, tensor in enumerate(storage.accounting_tensors()):
                yield f"attention_cache_storage.{layout}.{index}_cache", tensor

        for name, value in self.__dict__.items():
            if name in {"config", "hf_config", "attention_cache_storage"}:
                continue
            yield from visit(name, value)

    @staticmethod
    def _memory_accounting_category(path: str) -> str:
        lower = path.lower()
        if any(token in lower for token in ("slot", "mapping", "req_to_token", "_map", "map_")):
            return "slot_map"
        if any(token in lower for token in ("scale", "scales", "min", "mins", "zero")):
            return "scale_min_metadata"
        if any(token in lower for token in ("pos", "lens", "length", "score", "indices", "idx")):
            return "metadata"
        if "cache" in lower:
            return "kv_or_latent"
        return "other"

    def _logical_live_kv_bytes(self) -> int:
        row_seq_lens = getattr(self, "row_seq_lens", None)
        if row_seq_lens is None:
            return 0
        try:
            live_tokens = int(row_seq_lens.sum())
        except Exception:
            return 0
        return int(
            live_tokens
            * self.num_kv_layers
            * self.attention_cache_bytes_per_slot_per_layer()
        )

    def memory_accounting(self) -> dict[str, Any]:
        """Return read-only tensor memory accounting for regression gates.

        The accounting is intentionally generic: cache-manager-specific tensors are
        grouped by stable attribute-name patterns, and unique tensor storages are
        counted once so views do not inflate the result.
        """
        seen_storages: set[tuple[Any, ...]] = set()
        categories = {
            "kv_or_latent": 0,
            "slot_map": 0,
            "scale_min_metadata": 0,
            "metadata": 0,
            "other": 0,
        }
        tensors: list[dict[str, Any]] = []
        for path, tensor in self._iter_accounting_tensors():
            key = self._tensor_storage_key(tensor)
            if key in seen_storages:
                continue
            seen_storages.add(key)
            nbytes = self._tensor_storage_nbytes(tensor)
            category = self._memory_accounting_category(path)
            categories[category] += nbytes
            tensors.append(
                {
                    "path": path,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "nbytes": nbytes,
                    "category": category,
                }
            )

        dense_baseline_bytes = self._dense_baseline_bytes()
        allocated_tensor_bytes = int(sum(categories.values()))
        metadata_bytes = int(
            categories["slot_map"] + categories["scale_min_metadata"] + categories["metadata"]
        )
        observed_savings = None
        if dense_baseline_bytes > 0:
            observed_savings = 1.0 - (allocated_tensor_bytes / dense_baseline_bytes)

        theoretical = getattr(self.config, "memory_expected_savings", None)
        if theoretical is not None:
            theoretical = float(theoretical)

        return {
            "status": "success",
            "cache_manager_class": type(self).__name__,
            "dense_baseline_bytes": int(dense_baseline_bytes),
            "allocated_tensor_bytes": allocated_tensor_bytes,
            "logical_live_kv_bytes": int(self._logical_live_kv_bytes()),
            "slot_map_bytes": int(categories["slot_map"]),
            "scale_min_metadata_bytes": int(categories["scale_min_metadata"]),
            "metadata_bytes": metadata_bytes,
            "kv_or_latent_tensor_bytes": int(categories["kv_or_latent"]),
            "other_tensor_bytes": int(categories["other"]),
            "theoretical_savings": theoretical,
            "observed_savings": observed_savings,
            "tensor_count": len(tensors),
            "unique_storage_count": len(seen_storages),
            "dense_baseline": {
                "slots": int(self._dense_baseline_slots()),
                "layers": int(self.num_layers),
                "kv_layers": int(self.num_kv_layers),
                "num_kv_heads": int(self.num_kv_heads),
                "head_dim": int(self.head_dim),
                "dtype_size": int(self._cache_slot_dtype_size()),
            },
            "by_category": categories,
            "tensors": tensors,
        }

    def debug_live_seq_slots(self) -> dict[int, int]:
        """Return live seq_id -> occupied slot count for debugging."""
        return {}

    def chain_capacity_deficits(
        self,
        *,
        suffix_tokens: int,
        generation_tokens: int = 0,
        existing_slots_by_layer: tuple[int, ...] = (),
        outstanding_reserved_slots_by_layer: tuple[int, ...] = (),
        outstanding_reserved_rows: int = 0,
        needs_resident_row: bool,
    ) -> tuple[tuple[int, ...], int, tuple[int, ...], int]:
        """Return requirements and deficits for chain admission.

        Chain-capable cache managers override this with their physical layout.
        """
        del (
            suffix_tokens,
            generation_tokens,
            existing_slots_by_layer,
            outstanding_reserved_slots_by_layer,
            outstanding_reserved_rows,
            needs_resident_row,
        )
        raise RuntimeError(
            f"{type(self).__name__} does not implement chain cache capacity accounting."
        )

    def chain_physical_residency(self, seq_id: int) -> tuple[int, ...]:
        del seq_id
        raise RuntimeError(
            f"{type(self).__name__} does not implement chain cache residency accounting."
        )

    def chain_has_residency(self, seq_id: int) -> bool:
        del seq_id
        return False

    def chain_physical_kv_len(self, layer_idx: int, seq_id: int) -> int:
        del layer_idx, seq_id
        raise RuntimeError(
            f"{type(self).__name__} does not expose chain physical KV length."
        )

    def on_chain_turn_finished(
        self,
        seq_id: int,
        processed_token_count: int,
    ) -> None:
        del seq_id, processed_token_count

    @abstractmethod
    def free_seq(self, seq_id: int):
        raise NotImplementedError

    @abstractmethod
    def free_part_slots(self, layer_idx: int, seq: Sequence, keep_indices: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def _prepare_prefill(self, seqs: list[Sequence]):
        raise NotImplementedError

    @abstractmethod
    def _prepare_decode(self, seqs: list[Sequence]):
        raise NotImplementedError

    def get_compressed_lens(self, req_indices):
        raise NotImplementedError
