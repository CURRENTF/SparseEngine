from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from sparseengine.config import Config
from sparseengine.distributed import ParallelContext
from sparseengine.engine.decode_graph_contract import (
    CacheDecodeGraphState,
    DecodeGraphContract,
    DecodeGraphInputs,
)
from sparseengine.engine.prefix_cache import PrefixCacheBlock
from sparseengine.engine.prefix_prune import (
    select_global_keep_block_indices,
)
from sparseengine.engine.sequence import Sequence
from sparseengine.kernels.triton.quest_decode_view import (
    finalize_quest_decode_view,
    prepare_quest_decode_geometry,
    prepare_quest_decode_graph_metadata,
)
from sparseengine.operators.quest_selection import (
    QuestPageSelectionOpSpec,
    resolve_quest_page_selection_provider,
)
from sparseengine.utils.context import get_context
from sparseengine.utils.profiler import profiler

from ..base import (
    AttentionCacheWrite,
    AttentionViewMeta,
    CacheManager,
    DecodeComputeView,
    LayerBatchStates,
    MlaLatentSelectionQuery,
    PagedDecodeViewMeta,
    SparseSelection,
)
from ..prefix_cache_mixin import PrefixCacheMixin
from ..prefix_prune_scoring import PrefixPruneScoringMixin
from ..storage import (
    ExplicitKVStorage,
    MlaLatentStorage,
    create_attention_cache_storage,
)
from ..storage.components import CacheComponentSpec


@dataclass
class QuestPrefixBlockPayload:
    block_slot: int | None
    token_slots: torch.Tensor | None
    block_start: int = 0
    block_end: int = 0
    block_slots: torch.Tensor | None = None
    host_block_index: int | None = None
    retained_offsets: tuple[int, ...] | None = None

    def resident_tokens(self, block_size: int) -> int:
        if self.retained_offsets is None:
            return int(block_size)
        if self.retained_offsets == tuple(range(block_size)):
            return int(block_size)
        if self.retained_offsets:
            raise RuntimeError("QuEST prefix blocks cannot retain a partial page.")
        return 0


@dataclass(frozen=True)
class QuestPrefillPagePlan:
    """Packed-token runs, each owned by exactly one physical QuEST page."""

    segments: torch.Tensor


@dataclass
class QuestDecodeGraphState(CacheDecodeGraphState):
    """Stable QuEST decode metadata for the no-prefix graph path."""

    row_page_slots: torch.Tensor
    num_pages: torch.Tensor
    previous_page_counts: torch.Tensor
    host_write_slots: torch.Tensor


class QuestCacheManager(PrefixPruneScoringMixin, PrefixCacheMixin, CacheManager):
    """Paged KV cache + page metadata cache for QuEST."""

    _prefix_payload_type = QuestPrefixBlockPayload
    _prefix_label = "QuEST prefix"
    _prefix_wait_label = "QuEST prefix"
    _prefix_profile = "quest_prefix_cache"
    _prefix_unit_name = "page"
    _prefix_hit_capacity_device_resident_only = True

    @property
    def _prefix_device_unit_tokens(self) -> int:
        return self.page_size

    @property
    def _radix_block_size(self) -> int:
        return self.page_size

    def _prefix_hit_method(self) -> str:
        return "quest"

    def _prefix_block_resident_tokens(self, block: PrefixCacheBlock) -> int:
        return self._quest_payload(block).resident_tokens(self.page_size)

    def _prefix_resident_units_for_ids(
        self, block_ids: frozenset[bytes], *, view: str
    ) -> int:
        return self._prefix_resident_pages_for_ids(block_ids, view=view)

    def __init__(
        self,
        config: Config,
        parallel_context: ParallelContext,
        *,
        allocation_budget_bytes: int | None = None,
    ):
        super().__init__(
            config,
            parallel_context,
            allocation_budget_bytes=allocation_budget_bytes,
        )
        self.page_size = int(config.quest_chunk_size)
        self.max_pages_per_row = (self.max_model_len + self.page_size - 1) // self.page_size
        self.page_offsets_i32 = torch.arange(self.page_size, dtype=torch.int32, device=self.device)
        self.page_offsets_i64 = self.page_offsets_i32.to(torch.int64)

        self.attention_cache_storage = create_attention_cache_storage(
            config,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )
        if isinstance(self.attention_cache_storage, MlaLatentStorage):
            self.metadata_num_heads = 1
            self.metadata_head_dim = (
                int(self.attention_cache_storage.kv_lora_rank)
                + int(self.attention_cache_storage.rope_dim)
            )
        elif isinstance(self.attention_cache_storage, ExplicitKVStorage):
            self.metadata_num_heads = self.num_kv_heads
            self.metadata_head_dim = self.head_dim
        else:
            raise NotImplementedError(
                "QuEST requires homogeneous explicit KV or MLA latent storage, got "
                f"{type(self.attention_cache_storage).__name__}."
            )

        self.quest_page_selector = resolve_quest_page_selection_provider(
            QuestPageSelectionOpSpec(
                score_dtype=self.hf_config.dtype,
                cuda_graph=bool(config.decode_graph),
            ),
            device_index=self.device.index or 0,
        )
        self.allocate_kv_cache()

        self.free_pages_stack = torch.arange(self.num_pages, dtype=torch.int32, device=self.device)
        self.free_pages_cpu_stack = np.arange(self.num_pages, dtype=np.int32)
        self._num_free_pages = self.num_pages

        self.buffer_req_to_token_slots = torch.zeros(
            (self.max_buffer_rows, self.max_model_len), dtype=torch.int32, device=self.device
        )
        self.buffer_req_to_page_slots = torch.full(
            (self.max_buffer_rows, self.max_pages_per_row), -1, dtype=torch.int32, device=self.device
        )
        self.buffer_req_to_page_slots_cpu = np.full(
            (self.max_buffer_rows, self.max_pages_per_row), -1, dtype=np.int32
        )

        self.seq_id_to_row: dict[int, int] = {}
        self.free_rows = deque(range(self.max_buffer_rows))
        self.row_seq_lens = np.zeros((self.max_buffer_rows,), dtype=np.int32)
        self.row_logical_lens = np.zeros((self.max_buffer_rows,), dtype=np.int32)
        self._prefix_prune_scoring = None
        self._prefill_slot_reservations = None
        self.layer_batch_state = LayerBatchStates()
        self._decode_static_index_buffers: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._decode_paged_view_buffers: dict[
            tuple[int, int],
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ] = {}
        self._decode_row_page_slots_buffers: dict[tuple[int, int], torch.Tensor] = {}
        self._decode_row_page_slots: torch.Tensor | None = None
        self._decode_row_page_slots_req_indices: torch.Tensor | None = None
        self._decode_page_geometry_buffers: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._decode_num_pages: torch.Tensor | None = None
        self._decode_previous_page_counts: torch.Tensor | None = None
        self._decode_page_geometry_context_lens: torch.Tensor | None = None
        self._decode_page_geometry_ready = False
        self._prefill_page_plan: QuestPrefillPagePlan | None = None
        self.enable_prefix_caching = bool(
            config.enable_prefix_caching
            and config.sparse_method == "quest"
            and not getattr(getattr(config, "runtime_layout", None), "linear_attention_layer_indices", ())
        )
        self.prefix_cache_block_size = int(config.prefix_cache_block_size)
        if self.enable_prefix_caching and self.prefix_cache_block_size != self.page_size:
            raise ValueError(
                "Quest prefix cache requires prefix_cache_block_size == quest_chunk_size: "
                f"prefix_cache_block_size={self.prefix_cache_block_size}, quest_chunk_size={self.page_size}."
            )
        self.seq_id_to_cached_pages: dict[int, set[int]] = {}
        self._prefill_metadata_full_pages = False

        # [2, L, P, H_meta, D_meta] -> 0:max, 1:min. Explicit KV uses
        # per-KV-head keys; MLA uses the fused [latent, RoPE] key coordinates.
        self.metadata_cache = torch.empty(
            2,
            self.num_kv_layers,
            self.num_pages,
            self.metadata_num_heads,
            self.metadata_head_dim,
            dtype=self.hf_config.dtype,
            device=self.device,
        )
        self._init_radix_prefix_cache()

    def _prefix_offload_device_blocks(self) -> int:
        return int(self.num_pages)

    def _prefix_storage_components(self):
        summary_specs = tuple(
            CacheComponentSpec(
                name, (self.metadata_num_heads, self.metadata_head_dim),
                self.metadata_cache.dtype, "page",
            )
            for name in ("page_max", "page_min")
        )
        return tuple(
            layer + tuple(
                (spec, self.metadata_cache[component, index])
                for component, spec in enumerate(summary_specs)
            )
            for index, layer in enumerate(
                super()._prefix_storage_components()
            )
        )

    def _prefix_transfer_metadata(self) -> torch.Tensor:
        return self.metadata_cache

    def allocate_kv_cache(self):
        available_memory, slot_bytes_per_layer = self._get_available_slots_info()

        int32_bytes = torch.empty((), dtype=torch.int32).element_size()
        max_buffer_rows = int(getattr(self, "max_buffer_rows", 0))
        max_model_len = int(getattr(self, "max_model_len", 0))
        max_pages_per_row = int(getattr(self, "max_pages_per_row", 0))
        page_offset_bytes = (
            self.page_size
            * (
                int32_bytes
                + torch.empty((), dtype=torch.int64).element_size()
            )
            if hasattr(self, "page_offsets_i32")
            else 0
        )
        fixed_metadata_bytes = (
            max_buffer_rows * max_model_len * int32_bytes
            + max_buffer_rows * max_pages_per_row * int32_bytes
            + page_offset_bytes
        )
        available_memory -= fixed_metadata_bytes
        if available_memory <= 0:
            raise RuntimeError(
                "Available memory is insufficient for QuEST row and page metadata: "
                f"budget={available_memory + fixed_metadata_bytes} "
                f"metadata={fixed_metadata_bytes}."
            )
        page_bytes_per_layer = (
            self.page_size * int(slot_bytes_per_layer)
            + self._metadata_bytes_per_page_per_layer()
        )
        total_pages = available_memory // (
            self.num_kv_layers * page_bytes_per_layer + int32_bytes
        )
        total_token_slots = int(total_pages * self.page_size)
        assert total_token_slots > 0, "Available memory is insufficient for QuEST paged KV cache"

        self.config.num_kvcache_slots = total_token_slots
        self.num_pages = total_token_slots // self.page_size

        self.attention_cache_storage.allocate(
            num_layers=self.num_kv_layers,
            num_slots=total_token_slots,
            device=self.device,
        )
        self.kv_cache = getattr(self.attention_cache_storage, "cache", None)

    def attention_cache_bytes_per_slot_per_layer(self) -> int:
        storage = getattr(self, "attention_cache_storage", None)
        if storage is None:
            return super().attention_cache_bytes_per_slot_per_layer()
        return int(storage.bytes_per_slot_per_layer())

    def _metadata_bytes_per_page_per_layer(self) -> int:
        dtype_size = torch.empty((), dtype=self.hf_config.dtype).element_size()
        metadata_num_heads = getattr(self, "metadata_num_heads", None)
        metadata_head_dim = getattr(self, "metadata_head_dim", None)
        if metadata_num_heads is None:
            metadata_num_heads = self.num_kv_heads
        if metadata_head_dim is None:
            metadata_head_dim = self.head_dim
        return int(
            2
            * int(metadata_num_heads)
            * int(metadata_head_dim)
            * dtype_size
        )

    def _kv_allocation_bytes_per_prefix_block(
        self,
        slot_bytes_per_layer: int,
    ) -> int:
        block_size = int(self.config.prefix_cache_block_size or 0)
        if block_size <= 0 or block_size % self.page_size != 0:
            raise RuntimeError(
                "Quest mixed prefix blocks must contain whole pages for memory accounting: "
                f"block_size={block_size} page_size={self.page_size}."
            )
        return self._prefix_kv_allocation_nbytes(block_size, slot_bytes_per_layer)

    def _prefix_kv_allocation_nbytes(
        self,
        token_count: int,
        slot_bytes_per_layer: int,
    ) -> int:
        page_count = int(token_count) // self.page_size
        token_kv_bytes = int(token_count) * self.num_kv_layers * int(slot_bytes_per_layer)
        page_metadata_bytes = (
            page_count
            * self.num_kv_layers
            * self._metadata_bytes_per_page_per_layer()
        )
        return int(token_kv_bytes + page_metadata_bytes)

    def get_layer_kv_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        kv_idx = self.kv_layer_index(layer_idx)
        payload = self.attention_cache_storage.layer_payload(kv_idx)
        if not isinstance(self.attention_cache_storage, ExplicitKVStorage):
            raise TypeError(
                "get_layer_kv_cache requires explicit KV storage, got "
                f"{type(self.attention_cache_storage).__name__}."
            )
        return payload.k_cache, payload.v_cache

    def store_attention_payload(
        self,
        layer_idx: int,
        payload: AttentionCacheWrite,
    ) -> torch.Tensor:
        slot_mapping = self.layer_batch_state.slot_mapping
        if slot_mapping is None:
            raise RuntimeError(
                f"Attention cache store requires slot_mapping at layer={layer_idx}."
            )
        kv_idx = self.kv_layer_index(layer_idx)
        if not get_context().is_prefill:
            storage = self.attention_cache_storage
            if not isinstance(storage, (ExplicitKVStorage, MlaLatentStorage)):
                raise AssertionError(
                    f"Unhandled QuEST storage {type(storage).__name__}."
                )
            storage.store_with_quest_metadata(
                kv_idx,
                slot_mapping,
                payload,
                self.metadata_cache[0, kv_idx],
                self.metadata_cache[1, kv_idx],
                page_size=self.page_size,
            )
        elif isinstance(self.attention_cache_storage, ExplicitKVStorage):
            plan = self._prefill_page_plan
            if plan is None:
                raise RuntimeError("QuEST explicit-KV prefill requires a page-segment plan.")
            self.attention_cache_storage.store_prefill_with_quest_metadata(
                kv_idx,
                slot_mapping,
                payload,
                plan.segments,
                self.metadata_cache[0, kv_idx],
                self.metadata_cache[1, kv_idx],
                page_size=self.page_size,
            )
        else:
            self.attention_cache_storage.store(
                kv_idx,
                slot_mapping,
                payload,
            )
        return slot_mapping

    @property
    def num_free_slots(self) -> int:
        return int(self._num_free_pages * self.page_size)

    def _prefix_resident_pages_for_ids(
        self, block_ids: frozenset[bytes], *, view: str = "explicit"
    ) -> int:
        return self._prefix_resident_weight_for_ids(block_ids, view=view)

    def _prefix_step_reclaimable_pages(self) -> int:
        return self._prefix_step_reclaimable_units()

    def _partial_page_free_slots(self) -> int:
        total = 0
        for row_idx in self.seq_id_to_row.values():
            row_len = int(self.row_seq_lens[row_idx])
            page_offset = row_len % self.page_size
            if page_offset:
                total += self.page_size - page_offset
        return total

    def prefill_step_free_slots(self) -> int:
        return int(
            self.num_free_slots
            + self._prefix_step_reclaimable_slots()
            + self._partial_page_free_slots()
        )

    def prefill_step_free_slots_for(self, seq: Sequence) -> int:
        row_idx = self.seq_id_to_row.get(seq.seq_id)
        partial = 0
        if row_idx is not None:
            page_offset = int(self.row_seq_lens[row_idx]) % self.page_size
            if page_offset:
                partial = self.page_size - page_offset
        return int(self.num_free_slots + self._prefix_step_reclaimable_slots() + partial)

    def prefill_step_reservation_cost(self, seq: Sequence, scheduled_tokens: int) -> int:
        row_idx = self.seq_id_to_row.get(seq.seq_id)
        cur_len = 0 if row_idx is None else int(self.row_seq_lens[row_idx])
        remaining = int(scheduled_tokens)
        cost = 0
        page_offset = cur_len % self.page_size
        if page_offset:
            take = min(remaining, self.page_size - page_offset)
            cost += take
            remaining -= take
        if remaining > 0:
            cost += self._ceil_to_page_slots(remaining)
        return int(cost)

    def decode_step_free_slots(self) -> int:
        partial_decode_slots = 0
        for row_idx in self.seq_id_to_row.values():
            if int(self.row_seq_lens[row_idx]) % self.page_size:
                partial_decode_slots += 1
        page_slots = (
            self._num_free_pages + self._prefix_step_reclaimable_pages()
        ) * self.page_size
        return int(page_slots + partial_decode_slots)

    def decode_step_free_slots_for(self, seq: Sequence) -> int:
        if self._required_new_pages(seq.seq_id, 1) == 0:
            return 1
        return (
            self.page_size
            if (self._num_free_pages + self._prefix_step_reclaimable_pages()) > 0
            else 0
        )

    def decode_window_budgets(self) -> dict[str, int]:
        return {"slots": int(self.prompt_admission_free_slots())}

    def decode_window_costs(self, seq: Sequence, tokens: int) -> dict[str, int]:
        return {"slots": int(self._required_new_pages(seq.seq_id, tokens)) * self.page_size}

    def decode_step_reservation_cost(self, seq: Sequence) -> int:
        if self._required_new_pages(seq.seq_id, 1) == 0:
            return 1
        return self.page_size

    def _ceil_to_page_slots(self, n_tokens: int) -> int:
        n_tokens = int(n_tokens)
        if n_tokens <= 0:
            return 0
        return ((n_tokens + self.page_size - 1) // self.page_size) * self.page_size

    def prompt_admission_cost(self, seq: Sequence) -> int:
        hit_len = int(getattr(seq, "prefix_cache_hit_len", 0) or 0)
        suffix_len = int(seq.num_prompt_tokens - hit_len)
        if hit_len <= 0:
            return self._ceil_to_page_slots(suffix_len)
        reclaimable_blocks, promotion_blocks = self._prefix_hit_capacity_pages(seq)
        return (
            self._ceil_to_page_slots(suffix_len)
            + (promotion_blocks + reclaimable_blocks) * self.page_size
        )

    def prompt_admission_shared_costs(
        self, seq: Sequence
    ) -> dict[str, dict[object, int]]:
        if self.prefix_cache is None or int(getattr(seq, "prefix_cache_hit_len", 0) or 0) <= 0:
            return {}
        self._prefix_hit_capacity_counts(seq)
        entry = self.prefix_hit_capacity_cache.get(seq)
        chain = tuple(self._prefix_hit_chain(seq)) if entry is None else entry.chain
        offload_enabled = self._prefix_offload_enabled()
        reclaimable_ids = (
            self.prefix_cache.device_reclaimable_block_ids()
            if offload_enabled
            else self.prefix_cache.freeable_block_ids()
        )
        costs: dict[object, int] = {}
        for block in chain:
            reclaimable = (
                block.stable_block_id in reclaimable_ids
                and block.residency.device_present
            )
            promotion = offload_enabled and not block.residency.device_present
            if not reclaimable and not promotion:
                continue
            resident_tokens = self._quest_payload(block).resident_tokens(self.page_size)
            if resident_tokens > 0:
                costs[block.stable_block_id] = int(resident_tokens)
        return {"slots": costs} if costs else {}

    def _prefix_hit_capacity_pages(self, seq: Sequence) -> tuple[int, int]:
        return self._prefix_hit_capacity_units(seq)

    def reserved_prefill_slots(self, waiting_seqs: deque[Sequence], engine_prefill_chunk_size: int) -> int:
        reserved = 0
        for seq in waiting_seqs:
            if 0 < seq.num_prefilled_tokens < seq.num_prompt_tokens:
                remaining = int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
                reserved += self._ceil_to_page_slots(remaining)
        return reserved

    def _prefix_free_allocator_stats(self) -> dict[str, int]:
        return {"quest_free_pages": int(self._num_free_pages)}

    def _quest_payload(self, block: PrefixCacheBlock) -> QuestPrefixBlockPayload:
        payload = block.payload
        if not isinstance(payload, QuestPrefixBlockPayload):
            raise RuntimeError("QuEST prefix cache block has an invalid payload.")
        payload.resident_tokens(self.page_size)
        return payload

    _prefix_block_payload = _quest_payload

    def validate_prefix_prune_keep_tokens(self, keep_tokens: int) -> None:
        if int(keep_tokens) % self.page_size:
            raise ValueError(
                "QuEST prefix pruning requires keep_tokens to be page aligned: "
                f"keep_tokens={keep_tokens} page_size={self.page_size}."
            )

    def _validate_prefix_prune_selection(self, selected: list[int]) -> None:
        selected_page_counts: dict[int, int] = {}
        for index in selected:
            page_index = index // self.page_size
            selected_page_counts[page_index] = selected_page_counts.get(page_index, 0) + 1
        partial_pages = [
            (page_index, count)
            for page_index, count in selected_page_counts.items()
            if count != self.page_size
        ]
        if partial_pages:
            page_index, count = partial_pages[0]
            raise ValueError(
                "QuEST prefix pruning cannot retain a partial page: "
                f"block_index={page_index} retained={count} "
                f"page_size={self.page_size}."
            )

    def _compact_prefix_prune_payloads(
        self, plans: list[tuple[PrefixCacheBlock, tuple[int, ...]]]
    ) -> dict[str, object]:
        drop_blocks = [block for block, offsets in plans if not offsets]
        dropped_pages: list[int] = []
        for block in drop_blocks:
            payload = self._quest_payload(block)
            if block.residency.device_present:
                if payload.block_slot is None or not isinstance(payload.token_slots, torch.Tensor):
                    raise RuntimeError("QuEST prune found an inconsistent device page payload.")
                self._validate_page_slots(payload.token_slots, payload.block_slot)
                dropped_pages.append(int(payload.block_slot))
        if len(set(dropped_pages)) != len(dropped_pages):
            raise RuntimeError("QuEST prune found duplicate device page ownership.")
        if self._num_free_pages + len(dropped_pages) > int(self.free_pages_stack.numel()):
            raise RuntimeError("QuEST prefix page free stack would overflow during pruning.")

        empty_slots = torch.empty(0, dtype=torch.int32, device=self.device)
        for block, offsets in plans:
            self._quest_payload(block).retained_offsets = offsets
        for block in drop_blocks:
            payload = self._quest_payload(block)
            if block.residency.device_present:
                payload.block_slot = None
                payload.block_slots = None
                payload.token_slots = empty_slots
        if dropped_pages:
            self._return_prefix_device_pages(torch.tensor(
                dropped_pages, dtype=torch.int32, device=self.device,
            ))
        return {
            "retained_pages": len(plans) - len(drop_blocks),
            "freed_device_slots": len(dropped_pages) * self.page_size,
            "selection_granularity": "quest_page",
        }

    def select_prefix_prune_keep_indices(
        self,
        scores: torch.Tensor,
        *,
        keep_tokens: int,
        protected_suffix_tokens: int = 0,
    ) -> torch.Tensor:
        return select_global_keep_block_indices(
            scores,
            keep_tokens=keep_tokens,
            block_size=self.page_size,
            protected_suffix_tokens=protected_suffix_tokens,
        )

    def _free_device_prefix_blocks(self, blocks: list[PrefixCacheBlock]) -> None:
        if not blocks:
            return
        payloads = []
        for block in blocks:
            payload = block.payload
            if not isinstance(payload, QuestPrefixBlockPayload):
                raise RuntimeError("Quest prefix cache block is missing its device payload.")
            if payload.block_slots is not None:
                raise RuntimeError("QuEST radix offload does not support mixed multi-page payloads.")
            if payload.retained_offsets == ():
                if payload.block_slot is not None:
                    raise RuntimeError("Pruned QuEST prefix block still owns a device page.")
                payload.token_slots = None
                continue
            if payload.block_slot is None or not isinstance(payload.token_slots, torch.Tensor):
                raise RuntimeError(
                    "QuEST prefix cache block is missing its device page payload: "
                    f"block={block.stable_block_id.hex()[:16]}."
                )
            if payload.token_slots.numel() != self.page_size:
                raise RuntimeError("Quest prefix block must contain exactly one full page.")
            payloads.append(payload)
        if not payloads:
            return
        ptr = self._num_free_pages
        if ptr + len(payloads) > int(self.free_pages_stack.numel()):
            raise RuntimeError("QuEST prefix page free stack overflow.")
        page_slots = torch.tensor([payload.block_slot for payload in payloads],
                                  dtype=torch.int32, device=self.device)
        slots = torch.cat([payload.token_slots for payload in payloads])
        self._validate_page_slot_matrix(slots, page_slots)
        self.free_pages_stack[ptr:ptr + len(payloads)].copy_(page_slots)
        self._num_free_pages += len(payloads)
        for payload in payloads:
            payload.block_slot = None
            payload.token_slots = None

    def _prefix_cache_materialization_subject(self) -> str:
        return "Quest prefix materialization"

    def _prefix_cache_negative_refcount_message(self) -> str:
        return "Quest prefix cache block ref_count became negative."

    def _prefix_cache_materialize_profile_name(self) -> str:
        return "quest_prefix_cache_materialize"

    def _validate_page_slots(self, slots: torch.Tensor, page_slot: int | None = None) -> int:
        if int(slots.numel()) != self.page_size:
            raise RuntimeError(
                f"Quest prefix block must contain exactly one full page: "
                f"num_slots={int(slots.numel())} page_size={self.page_size}."
            )
        slots_i32 = slots.to(dtype=torch.int32)
        page_slots = torch.div(slots_i32, self.page_size, rounding_mode="floor")
        page_offsets = torch.remainder(slots_i32, self.page_size)
        first_page_slot = int(page_slots[0].item())
        if page_slot is not None and int(page_slot) != first_page_slot:
            raise RuntimeError(
                f"Quest prefix block page_slot does not match token slots: "
                f"page_slot={page_slot} slots_page={first_page_slot}."
            )
        if hasattr(self, "num_pages") and not (0 <= first_page_slot < int(self.num_pages)):
            raise RuntimeError(
                f"Quest prefix block page_slot out of range: page_slot={first_page_slot} "
                f"num_pages={int(self.num_pages)}."
            )
        if not torch.all(page_slots == first_page_slot).item():
            raise RuntimeError("Quest prefix block token slots span multiple pages.")
        expected_offsets = self.page_offsets_i32.to(device=slots_i32.device)
        if not torch.equal(page_offsets, expected_offsets):
            raise RuntimeError("Quest prefix block token slots are not a contiguous full page.")
        return first_page_slot

    def _validate_page_slot_matrix(
        self,
        slots: torch.Tensor,
        expected_page_slots: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_slots = int(slots.numel())
        if num_slots <= 0 or num_slots % self.page_size != 0:
            raise RuntimeError(
                "Quest prefix payload must contain contiguous full pages: "
                f"num_slots={num_slots} page_size={self.page_size}."
            )
        slots_i32 = slots.to(device=self.device, dtype=torch.int32).reshape(-1, self.page_size)
        page_slots = torch.div(slots_i32[:, 0], self.page_size, rounding_mode="floor")
        expected_slots = (
            page_slots[:, None] * self.page_size
            + self.page_offsets_i32.to(device=slots_i32.device)[None, :]
        )
        valid = torch.all(slots_i32 == expected_slots)
        if expected_page_slots is not None:
            expected_page_slots = expected_page_slots.to(
                device=slots_i32.device,
                dtype=torch.int32,
            ).reshape(-1)
            if int(expected_page_slots.numel()) != int(page_slots.numel()):
                raise RuntimeError(
                    "Quest prefix payload page count does not match page metadata: "
                    f"payload_pages={int(page_slots.numel())} "
                    f"metadata_pages={int(expected_page_slots.numel())}."
                )
            valid = valid & torch.all(page_slots == expected_page_slots)
        if hasattr(self, "num_pages"):
            valid = valid & torch.all((page_slots >= 0) & (page_slots < int(self.num_pages)))
        if not bool(valid.item()):
            raise RuntimeError(
                "Quest prefix payload contains non-contiguous, mismatched, or out-of-range page slots."
            )
        return page_slots.contiguous()

    def _make_prefix_block_payload(self, slots: torch.Tensor) -> QuestPrefixBlockPayload:
        return QuestPrefixBlockPayload(
            block_slot=self._validate_page_slots(slots),
            token_slots=slots,
        )

    def _make_prefix_block_payloads(self, slots: list[torch.Tensor]) -> list[QuestPrefixBlockPayload]:
        # Keep CPU page IDs for radix ownership, but validate/copy the completed
        # pages together instead of synchronizing separately for each page.
        packed = slots[0] if len(slots) == 1 else torch.cat(slots)
        page_ids = self._validate_page_slot_matrix(packed).cpu().tolist()
        return [QuestPrefixBlockPayload(block_slot=int(page_id), token_slots=block_slots)
                for page_id, block_slots in zip(page_ids, slots, strict=True)]

    def _payload_page_slots(self, payload: QuestPrefixBlockPayload) -> torch.Tensor:
        if not isinstance(payload.token_slots, torch.Tensor):
            raise RuntimeError("Quest prefix payload has no device token slots.")
        if payload.retained_offsets == ():
            if payload.token_slots.numel() != 0 or payload.block_slot is not None:
                raise RuntimeError("Pruned QuEST prefix payload has inconsistent device state.")
            return torch.empty(0, dtype=torch.int32, device=self.device)
        if payload.block_slots is None:
            if payload.block_slot is None:
                raise RuntimeError("Quest single-page prefix payload is missing block_slot.")
            page_slots = torch.tensor([int(payload.block_slot)], dtype=torch.int32, device=self.device)
        else:
            page_slots = payload.block_slots.to(device=self.device, dtype=torch.int32)
            if page_slots.ndim != 1:
                page_slots = page_slots.reshape(-1)
        expected_pages = int(payload.token_slots.numel()) // self.page_size
        if int(payload.token_slots.numel()) % self.page_size != 0 or int(page_slots.numel()) != expected_pages:
            raise RuntimeError(
                "Quest prefix payload page metadata does not match token slots: "
                f"token_slots={int(payload.token_slots.numel())} page_size={self.page_size} "
                f"page_slots={int(page_slots.numel())}."
            )
        return page_slots

    def _release_prefix_payload_pages(self, payload: QuestPrefixBlockPayload) -> None:
        if payload.retained_offsets == ():
            if payload.block_slot is not None or payload.block_slots is not None:
                raise RuntimeError("Pruned QuEST prefix payload still owns device pages.")
            return
        if payload.block_slots is None:
            if payload.block_slot is None:
                raise RuntimeError("Quest single-page prefix payload is missing block_slot.")
            self.free_pages_stack[self._num_free_pages] = int(payload.block_slot)
            self._num_free_pages += 1
            return
        page_slots = self._payload_page_slots(payload)
        start = int(self._num_free_pages)
        end = start + int(page_slots.numel())
        if end > int(self.free_pages_stack.numel()):
            raise RuntimeError(
                "Quest prefix page free stack overflow: "
                f"start={start} pages={int(page_slots.numel())} "
                f"capacity={int(self.free_pages_stack.numel())}."
            )
        self.free_pages_stack[start:end].copy_(page_slots)
        self._num_free_pages = end

    def _mark_materialized_prefix_block(self, seq: Sequence, block: PrefixCacheBlock) -> None:
        cached_pages = self.seq_id_to_cached_pages.setdefault(seq.seq_id, set())
        chain = self.seq_id_to_prefix_blocks.get(seq.seq_id, ())
        resident_prefix_pages = sum(
            self._quest_payload(item).resident_tokens(self.page_size) // self.page_size
            for item in chain
        )
        private_page_offset = int(block.logical_block_idx) - len(chain)
        cached_pages.add(resident_prefix_pages + private_page_offset)

    def _validate_mixed_prefix_range(self, start: int, end: int) -> None:
        if start % self.page_size != 0 or end % self.page_size != 0:
            raise RuntimeError(
                "Quest mixed prefix payload must be page aligned: "
                f"range={start}:{end} page_size={self.page_size}."
            )

    def _build_mixed_prefix_payload(
        self, slots: torch.Tensor, block_start: int, block_end: int
    ) -> QuestPrefixBlockPayload:
        page_slots = self._validate_page_slot_matrix(slots)
        return QuestPrefixBlockPayload(
            block_slot=None,
            token_slots=slots,
            block_start=block_start,
            block_end=block_end,
            block_slots=page_slots,
        )

    def _mixed_prefix_attach_units(
        self, seq_id: int, start: int, payload: QuestPrefixBlockPayload, slots: torch.Tensor
    ) -> torch.Tensor:
        expected_tokens = int(payload.block_end) - int(payload.block_start)
        if expected_tokens <= 0 or int(slots.numel()) != expected_tokens:
            raise RuntimeError(
                "Quest mixed prefix KV payload token count does not match its range: "
                f"range={int(payload.block_start)}:{int(payload.block_end)} "
                f"slots={int(slots.numel())}."
            )
        if int(payload.block_start) != start or start % self.page_size != 0:
            raise RuntimeError(
                "Quest mixed prefix KV payload attach must be contiguous and page aligned: "
                f"seq_id={seq_id} block_start={int(payload.block_start)} row_len={start}."
            )
        return self._payload_page_slots(payload)

    def _pack_mixed_prefix_units(
        self, start: int, slots: torch.Tensor, unit_parts: list[torch.Tensor]
    ) -> torch.Tensor:
        pages = unit_parts[0] if len(unit_parts) == 1 else torch.cat(unit_parts)
        pages = self._validate_page_slot_matrix(slots, pages)
        if start // self.page_size + pages.numel() > self.buffer_req_to_page_slots.shape[1]:
            raise RuntimeError("Attaching mixed Quest prefix KV payload exceeds page row capacity.")
        return pages

    def free_prefix_kv_payload(self, payload: object) -> None:
        if not isinstance(payload, QuestPrefixBlockPayload):
            raise RuntimeError("Quest mixed prefix KV payload is missing page payload.")
        self._release_prefix_payload_pages(payload)

    def _mixed_prefix_allocation_count(self, tokens: int) -> int:
        if tokens <= 0 or tokens % self.page_size != 0:
            raise RuntimeError("Quest mixed prefix KV promotion requires whole pages.")
        return tokens // self.page_size

    def _stage_mixed_prefix_payloads(
        self, payloads: list[QuestPrefixBlockPayload], page_counts: list[int], pages: torch.Tensor
    ) -> None:
        token_slots = (
            pages[:, None] * self.page_size + self.page_offsets_i32[None, :]
        ).reshape(-1)
        page_offset = 0
        token_offset = 0
        for payload, page_count in zip(payloads, page_counts):
            token_count = page_count * self.page_size
            payload.block_slot = None
            payload.block_slots = pages[page_offset : page_offset + page_count]
            payload.token_slots = token_slots[token_offset : token_offset + token_count]
            page_offset += page_count
            token_offset += token_count

    def _release_mixed_prefix_payload_device(self, payload: QuestPrefixBlockPayload) -> None:
        pages = self._payload_page_slots(payload)
        self._return_prefix_device_pages(pages)
        payload.block_slot = None
        payload.block_slots = None
        payload.token_slots = None

    def _mixed_prefix_kv_nbytes(self, token_count: int) -> int:
        if token_count <= 0 or token_count % self.page_size != 0:
            raise RuntimeError(
                "Quest mixed prefix KV payload must contain whole pages for memory accounting: "
                f"token_count={token_count} page_size={self.page_size}."
            )
        slot_bytes_per_layer = (
            2 * self.num_kv_heads * self.head_dim * self._cache_slot_dtype_size()
        )
        return self._prefix_kv_allocation_nbytes(token_count, slot_bytes_per_layer)

    def mark_materialized_prefix_kv_payload(self, seq: Sequence, payload: object) -> None:
        if not isinstance(payload, QuestPrefixBlockPayload):
            raise RuntimeError("Quest mixed prefix KV payload is missing page payload.")
        if not isinstance(payload.token_slots, torch.Tensor):
            raise RuntimeError("Quest mixed prefix KV payload has no device token slots.")
        start_page = int(payload.block_start) // int(self.page_size)
        page_count = int(payload.token_slots.numel()) // int(self.page_size)
        self.seq_id_to_cached_pages.setdefault(int(seq.seq_id), set()).update(
            range(start_page, start_page + page_count)
        )

    def _mixed_prefix_payload_range(self, payload: QuestPrefixBlockPayload) -> tuple[int, int]:
        if not isinstance(payload.token_slots, torch.Tensor):
            raise RuntimeError("Quest mixed prefix KV payload has no device token slots.")
        start = (int(payload.block_start) // self.page_size) * self.page_size
        return start, start + (int(payload.token_slots.numel()) // self.page_size) * self.page_size

    def _reset_prefix_cache_allocator_after_clear(self) -> None:
        if self.seq_id_to_row:
            raise RuntimeError("Cannot reset prefix cache while QuEST sequences are active.")
        controller = getattr(self, "prefix_offload_controller", None)
        if controller is not None:
            controller.reset()
        self.free_pages_stack[: self.num_pages] = torch.arange(self.num_pages, dtype=torch.int32, device=self.device)
        if not hasattr(self, "free_pages_cpu_stack"):
            self.free_pages_cpu_stack = np.arange(self.num_pages, dtype=np.int32)
        self.free_pages_cpu_stack[: self.num_pages] = np.arange(self.num_pages, dtype=np.int32)
        self._num_free_pages = int(self.num_pages)
        self.seq_id_to_cached_pages.clear()
        getattr(self, "_prefix_write_through_candidates", {}).clear()

    def reset_after_warmup(self) -> None:
        if self.enable_prefix_caching and self.prefix_cache is not None:
            self.reset_prefix_cache()
            return
        self._reset_prefix_cache_allocator_after_clear()

    @property
    def _prefix_cached_rows(self) -> dict[int, set[int]]:
        return self.seq_id_to_cached_pages

    def _prefix_cached_entries(self, ranges: list[tuple[int, int]]) -> set[int]:
        return {
            page for start, end in ranges
            for page in range(start // self.page_size, end // self.page_size)
        }

    def _mark_mixed_prefix_ranges(self, seq_id: int, ranges: list[tuple[int, int]]) -> None:
        self.seq_id_to_cached_pages.setdefault(seq_id, set()).update(
            self._prefix_cached_entries(ranges)
        )

    def _take_prefix_device_units(self, count: int) -> torch.Tensor:
        return self._take_prefix_device_pages(count)

    def _return_prefix_device_units(self, units: torch.Tensor) -> None:
        self._return_prefix_device_pages(units)

    def _validate_prefix_row_capacity(self, seq_id: int, end: int, *, mixed: bool) -> None:
        if mixed:
            if end > self.buffer_req_to_token_slots.shape[1]:
                raise RuntimeError("Attaching mixed Quest prefix KV payload exceeds row capacity.")
        elif (end > self.buffer_req_to_token_slots.shape[1]
                or end // self.page_size > self.buffer_req_to_page_slots.shape[1]):
            raise ValueError("QuEST prefix attachment exceeds the physical row capacity.")

    def _validate_radix_block_index(self, block: PrefixCacheBlock, block_index: int) -> None:
        if int(block.logical_block_idx) != block_index:
            raise ValueError("QuEST prefix attachment is not root-contiguous.")

    def _prepare_empty_radix_payload(self, block: PrefixCacheBlock, expected_tokens: int) -> bool:
        if expected_tokens:
            return False
        payload = block.payload
        if block.residency.device_present:
            if payload.block_slot is not None:
                raise RuntimeError("Pruned QuEST block still owns a device page.")
            if not isinstance(payload.token_slots, torch.Tensor) or payload.token_slots.numel():
                raise RuntimeError("Pruned QuEST block has inconsistent empty slots.")
        else:
            if not self._prefix_offload_enabled() or not block.residency.host_present:
                raise RuntimeError("Pruned QuEST block has no usable cache residency.")
            payload.token_slots = torch.empty(0, dtype=torch.int32, device=self.device)
        # Empty logical blocks do not affect the contiguity of device pages.
        return True

    def _prefix_attach_rollback_logical_len(self, previous_row: int | None) -> int:
        return 0

    def _validate_resident_radix_payload(
        self, seq_id: int, block: PrefixCacheBlock, expected_tokens: int
    ) -> None:
        payload = block.payload
        if payload.block_slot is None or not isinstance(payload.token_slots, torch.Tensor):
            raise RuntimeError("Device-resident QuEST prefix block has no device page.")
        if payload.token_slots.ndim != 1 or payload.token_slots.numel() != self.page_size:
            raise RuntimeError("QuEST prefix block must contain exactly one full page.")
        if not 0 <= int(payload.block_slot) < self.num_pages:
            raise RuntimeError("QuEST prefix block page_slot is out of range.")

    def _pack_radix_prefix_blocks(
        self, blocks: list[PrefixCacheBlock],
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        if not blocks:
            return [], None
        page_ids = [int(block.payload.block_slot) for block in blocks]
        pages = torch.tensor(page_ids, dtype=torch.int32, device=self.device)
        slot_parts, _ = super()._pack_radix_prefix_blocks(blocks)
        slots = slot_parts[0] if len(slot_parts) == 1 else torch.cat(slot_parts)
        expected = (pages[:, None] * self.page_size + self.page_offsets_i32[None, :]).reshape(-1)
        if not torch.equal(slots.to(dtype=torch.int32), expected):
            raise RuntimeError("QuEST prefix token slots do not match contiguous page geometry.")
        return [slots], pages

    def _stage_radix_prefix_blocks(
        self, blocks: list[PrefixCacheBlock], units: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not units.numel():
            return None, None
        token_slots = units[:, None] * self.page_size + self.page_offsets_i32[None, :]
        with_pages = (block.payload for block in blocks if block.payload.retained_offsets != ())
        for index, (payload, page) in enumerate(zip(with_pages, units.cpu().tolist(), strict=True)):
            payload.block_slot = int(page)
            payload.token_slots = token_slots[index]
        return token_slots.reshape(-1), units

    def _clear_radix_promotion_payloads(self, blocks: list[PrefixCacheBlock]) -> None:
        for block in blocks:
            if block.payload.retained_offsets != ():
                block.payload.block_slot = None
                block.payload.token_slots = None

    def _write_prefix_row_slots(
        self, row_idx: int, start: int, end: int,
        slots: torch.Tensor, device_units: torch.Tensor | None,
    ) -> None:
        assert device_units is not None
        start_page = start // self.page_size
        end_page = start_page + int(device_units.numel())
        self.buffer_req_to_page_slots[row_idx, start_page:end_page].copy_(device_units)
        super()._write_prefix_row_slots(row_idx, start, end, slots, device_units)

    def _clear_prefix_row_slots(
        self, row_idx: int, start: int, end: int | None = None
    ) -> None:
        super()._clear_prefix_row_slots(row_idx, start, end)
        end_page = None if end is None else end // self.page_size
        self.buffer_req_to_page_slots[row_idx, start // self.page_size:end_page].fill_(-1)
        if end is None and hasattr(self, "buffer_req_to_page_slots_cpu"):
            self.buffer_req_to_page_slots_cpu[row_idx, :] = -1

    @contextmanager
    def reserve_prefill_slots(
        self,
        requests: list[tuple[int, int]],
        *,
        prefix_blocks: dict[int, list[PrefixCacheBlock]] | None = None,
    ):
        if self._prefill_slot_reservations is not None:
            raise RuntimeError("A QuEST prefill page reservation is already active.")
        if not requests or len({sid for sid, _ in requests}) != len(requests):
            raise ValueError("Prefill reservation requires distinct sequence IDs.")
        if any(size <= 0 for _, size in requests):
            raise ValueError("Prefill reservation sizes must be positive.")
        candidates: list[tuple[int, int, int]] = []
        required_pages: list[int] = []
        promotion_ids: set[bytes] = set()
        query_pages = promotion_pages = 0
        available_rows = len(self.free_rows)
        for sid, size in requests:
            if sid not in self.seq_id_to_row:
                if not available_rows:
                    break
                available_rows -= 1
            pages = self._required_new_pages(sid, size)
            candidates.append((sid, size, pages))
            query_pages += pages
            for block in (() if prefix_blocks is None else prefix_blocks[sid]):
                if (
                    not block.residency.device_present
                    and block.stable_block_id not in promotion_ids
                ):
                    promotion_ids.add(block.stable_block_id)
                    promotion_pages += (
                        self._quest_payload(block).resident_tokens(self.page_size)
                        // self.page_size
                    )
            required_pages.append(query_pages + promotion_pages)
        if not candidates:
            raise RuntimeError("Prefill reservation requires an idle cache row.")
        self._evict_prefix_cache_until_free(required_pages[-1] * self.page_size)
        admitted: list[tuple[int, int, int]] = []
        total_query_pages = 0
        for candidate, required in zip(candidates, required_pages, strict=True):
            if required > self._num_free_pages:
                break
            admitted.append(candidate)
            total_query_pages += candidate[2]
        if not admitted:
            raise RuntimeError(
                "Out of QuEST pages after prefix eviction for prefill reservation: "
                f"need={required_pages[0]} free={self._num_free_pages}."
            )
        pages = self._take_prefix_device_pages(total_query_pages)
        reserved: dict[int, torch.Tensor] = {}
        offset = 0
        for sid, _, page_count in admitted:
            reserved[sid] = pages[offset : offset + page_count]
            offset += page_count
        new_rows = [sid for sid, _, _ in admitted if sid not in self.seq_id_to_row]
        self._prefill_slot_reservations = reserved
        try:
            for sid in new_rows:
                self._get_free_row(sid)
            yield len(admitted)
        finally:
            self._prefill_slot_reservations = None
            if reserved:
                unused = [item for item in reserved.values() if item.numel()]
                if unused:
                    self._return_prefix_device_pages(torch.cat(unused))
            for sid in new_rows:
                if sid in self.seq_id_to_row:
                    self.free_seq(sid)

    def _take_prefix_device_pages(self, count: int) -> torch.Tensor:
        count = int(count)
        self._evict_prefix_cache_until_free(count * self.page_size)
        if self._num_free_pages < count:
            raise RuntimeError(
                "Out of QuEST pages while promoting a CPU prefix: "
                f"need={count} free={self._num_free_pages}."
            )
        ptr = self._num_free_pages
        pages = self.free_pages_stack[ptr - count:ptr].clone()
        self._num_free_pages -= count
        return pages

    def _return_prefix_device_pages(self, pages: torch.Tensor) -> None:
        pages = pages.to(device=self.device, dtype=torch.int32).reshape(-1)
        count = int(pages.numel())
        ptr = self._num_free_pages
        if ptr + count > int(self.free_pages_stack.numel()):
            raise RuntimeError("QuEST prefix page free stack overflow.")
        self.free_pages_stack[ptr:ptr + count] = pages
        self._num_free_pages += count

    def _prefill_positions(self, seq: Sequence, start: int, end: int) -> np.ndarray:
        return np.arange(start, end)

    def _required_new_pages(self, seq_id: int, size: int) -> int:
        row_idx = self.seq_id_to_row.get(seq_id)
        cur_len = 0 if row_idx is None else int(self.row_seq_lens[row_idx])
        before_pages = (cur_len + self.page_size - 1) // self.page_size
        after_len = cur_len + int(size)
        after_pages = (after_len + self.page_size - 1) // self.page_size
        return max(0, after_pages - before_pages)

    @torch.no_grad()
    def _allocate(self, seq_id: int, size: int) -> torch.Tensor:
        with profiler.record("cache_allocate"):
            if not hasattr(self, "row_logical_lens"):
                self.row_logical_lens = np.asarray(self.row_seq_lens, dtype=np.int32).copy()
            if type(size) is not int or size < 0:
                raise ValueError("QuEST allocation size must be a non-negative integer.")
            existing_row = self.seq_id_to_row.get(seq_id)
            cur_len = 0 if existing_row is None else int(self.row_seq_lens[existing_row])
            old_logical_len = (
                cur_len if existing_row is None else int(self.row_logical_lens[existing_row])
            )
            max_model_len = min(
                int(getattr(self, "max_model_len", self.buffer_req_to_token_slots.shape[1])),
                int(self.buffer_req_to_token_slots.shape[1]),
                int(self.buffer_req_to_page_slots.shape[1]) * int(self.page_size),
            )
            if old_logical_len + size > max_model_len:
                raise RuntimeError("QuEST allocation exceeds max_model_len.")
            if cur_len + size > int(self.buffer_req_to_token_slots.shape[1]):
                raise RuntimeError("QuEST allocation exceeds the cache row capacity.")
            if existing_row is None and not self.free_rows:
                raise RuntimeError("No free rows in cache manager buffer!")
            needed_pages = self._required_new_pages(seq_id, size)
            reservations = getattr(self, "_prefill_slot_reservations", None)
            reserved_pages = None if reservations is None else reservations.get(seq_id)
            if reserved_pages is not None and int(reserved_pages.numel()) != needed_pages:
                raise RuntimeError("QuEST prefill allocation differs from its page reservation.")
            if needed_pages > 0 and reserved_pages is None:
                self._evict_prefix_cache_until_free(needed_pages * self.page_size)
            ptr = self._num_free_pages
            if reserved_pages is None and ptr < needed_pages:
                raise RuntimeError(
                    f"Out of QuEST KV pages: need_pages={needed_pages}, free_pages={ptr}."
                )
            first_new_page = (cur_len + self.page_size - 1) // self.page_size
            try:
                row_idx = self._get_free_row(seq_id)
                if needed_pages > 0:
                    if reserved_pages is not None:
                        new_page_slots = reserved_pages
                    elif self.enable_prefix_caching:
                        new_page_slots = self.free_pages_stack[ptr - needed_pages:ptr].flip(0)
                    else:
                        new_page_slots_cpu = self.free_pages_cpu_stack[ptr - needed_pages:ptr][::-1].copy()
                        new_page_slots = torch.from_numpy(new_page_slots_cpu).to(
                            device=self.device, dtype=torch.int32,
                        )
                        self.buffer_req_to_page_slots_cpu[
                            row_idx, first_new_page:first_new_page + needed_pages,
                        ] = new_page_slots_cpu
                    self.buffer_req_to_page_slots[
                        row_idx, first_new_page:first_new_page + needed_pages,
                    ] = new_page_slots
                positions = torch.arange(cur_len, cur_len + size, dtype=torch.int64, device=self.device)
                page_indices = torch.div(positions, int(self.page_size), rounding_mode="floor")
                page_offsets = torch.remainder(positions, int(self.page_size)).to(torch.int32)
                page_slots = self.buffer_req_to_page_slots[row_idx, page_indices]
                allocated_slots = page_slots * int(self.page_size) + page_offsets
                self.buffer_req_to_token_slots[row_idx, cur_len:cur_len + size] = allocated_slots
                # Publish ownership once every metadata write has succeeded.
                self.row_seq_lens[row_idx] = cur_len + size
                self.row_logical_lens[row_idx] = old_logical_len + size
                if reserved_pages is None:
                    self._num_free_pages = ptr - needed_pages
            except BaseException:
                self._num_free_pages = ptr
                row_idx = self.seq_id_to_row.get(seq_id)
                if row_idx is not None:
                    self.row_seq_lens[row_idx] = cur_len
                    self.row_logical_lens[row_idx] = old_logical_len
                    self.buffer_req_to_token_slots[row_idx, cur_len:cur_len + size] = 0
                    self.buffer_req_to_page_slots[
                        row_idx, first_new_page:first_new_page + needed_pages,
                    ] = -1
                    if hasattr(self, "buffer_req_to_page_slots_cpu"):
                        self.buffer_req_to_page_slots_cpu[
                            row_idx, first_new_page:first_new_page + needed_pages,
                        ] = -1
                    if existing_row is None:
                        self.seq_id_to_row.pop(seq_id, None)
                        self.free_rows.appendleft(row_idx)
                raise
            if reserved_pages is not None:
                del reservations[seq_id]
            return allocated_slots

    @torch.no_grad()
    def _allocate_batch(
        self,
        seq_ids: list[int],
        size: int,
        *,
        graph_batch_size: int | None = None,
    ) -> torch.Tensor:
        assert size == 1, "Batch allocation currently only supports size=1 (Decode)"
        with profiler.record("cache_allocate"):
            if not hasattr(self, "row_logical_lens"):
                self.row_logical_lens = np.asarray(self.row_seq_lens, dtype=np.int32).copy()
            batch_size = len(seq_ids)
            row_indices, pending_rows = self._plan_decode_rows(np.asarray(seq_ids, dtype=np.int64))
            cur_lens = self.row_seq_lens[row_indices]
            logical_lens = self.row_logical_lens[row_indices]
            max_model_len = int(getattr(self, "max_model_len", self.buffer_req_to_token_slots.shape[1]))
            if len(logical_lens) > 0 and int(max(logical_lens)) + 1 > max_model_len:
                raise RuntimeError(
                    "KV row length exceeds max_model_len in QuEST _allocate_batch: "
                    f"max_logical_len={int(max(logical_lens))} max_model_len={max_model_len}"
                )

            page_indices = cur_lens // self.page_size
            page_offsets = cur_lens % self.page_size
            new_page_positions = np.nonzero(page_offsets == 0)[0]
            needed_pages = int(new_page_positions.size)
            if needed_pages > 0:
                self._evict_prefix_cache_until_free(needed_pages * self.page_size)
            assert self._num_free_pages >= needed_pages, (
                f"Out of QuEST KV pages: need_pages={needed_pages}, free_pages={self._num_free_pages}, "
                f"size={batch_size}, free_slots={self.num_free_slots}"
            )

            self._commit_decode_rows(pending_rows)

            if graph_batch_size is None:
                rows_gpu = torch.tensor(row_indices, dtype=torch.long, device=self.device)
                page_indices_gpu = torch.tensor(page_indices, dtype=torch.long, device=self.device)
                cur_lens_gpu = torch.tensor(cur_lens, dtype=torch.long, device=self.device)
            else:
                rows_gpu, page_indices_gpu, cur_lens_gpu = self._get_decode_static_index_buffers(
                    int(graph_batch_size)
                )
                rows_gpu[:batch_size].copy_(torch.from_numpy(row_indices))
                page_indices_gpu[:batch_size].copy_(torch.from_numpy(page_indices.astype(np.int64, copy=False)))
                cur_lens_gpu[:batch_size].copy_(torch.from_numpy(cur_lens.astype(np.int64, copy=False)))
                rows_gpu = rows_gpu[:batch_size]
                page_indices_gpu = page_indices_gpu[:batch_size]
                cur_lens_gpu = cur_lens_gpu[:batch_size]
            if needed_pages > 0:
                ptr = self._num_free_pages
                if self.enable_prefix_caching:
                    new_page_slots = self.free_pages_stack[ptr - needed_pages : ptr].flip(0)
                else:
                    new_page_slots_cpu = self.free_pages_cpu_stack[ptr - needed_pages : ptr][::-1].copy()
                    new_page_slots = torch.from_numpy(new_page_slots_cpu).to(
                        device=self.device,
                        dtype=torch.int32,
                    )
                    self.buffer_req_to_page_slots_cpu[
                        row_indices[new_page_positions],
                        page_indices[new_page_positions],
                    ] = new_page_slots_cpu
                self._num_free_pages -= needed_pages
                new_pos_gpu = torch.tensor(new_page_positions, dtype=torch.long, device=self.device)
                self.buffer_req_to_page_slots[
                    rows_gpu.index_select(0, new_pos_gpu),
                    page_indices_gpu.index_select(0, new_pos_gpu),
                ] = new_page_slots

            if needed_pages == 0:
                allocated_slots = self.buffer_req_to_token_slots[rows_gpu, cur_lens_gpu - 1] + 1
            elif self.enable_prefix_caching:
                page_offsets_gpu = torch.tensor(page_offsets, dtype=torch.int32, device=self.device)
                page_slots = self.buffer_req_to_page_slots[rows_gpu, page_indices_gpu]
                allocated_slots = page_slots * int(self.page_size) + page_offsets_gpu
            else:
                page_slots_cpu = self.buffer_req_to_page_slots_cpu[row_indices, page_indices]
                allocated_slots_cpu = page_slots_cpu * int(self.page_size) + page_offsets
                allocated_slots = torch.from_numpy(allocated_slots_cpu.astype(np.int32, copy=False)).to(
                    device=self.device,
                    dtype=torch.int32,
                )
            self.buffer_req_to_token_slots[rows_gpu, cur_lens_gpu] = allocated_slots
            self.row_seq_lens[row_indices] += 1
            self.row_logical_lens[row_indices] += 1
            return allocated_slots.to(torch.int32)

    def _get_decode_static_index_buffers(
        self,
        graph_batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        graph_batch_size = int(graph_batch_size)
        if not hasattr(self, "_decode_static_index_buffers"):
            self._decode_static_index_buffers = {}
        buffers = self._decode_static_index_buffers.get(graph_batch_size)
        if buffers is None:
            buffers = (
                torch.empty((graph_batch_size,), dtype=torch.long, device=self.device),
                torch.empty((graph_batch_size,), dtype=torch.long, device=self.device),
                torch.empty((graph_batch_size,), dtype=torch.long, device=self.device),
            )
            self._decode_static_index_buffers[graph_batch_size] = buffers
        return buffers

    def free_seq(self, seq_id: int):
        with profiler.record("cache_free_seq"):
            self._poll_prefix_offload()
            row_idx = self.seq_id_to_row.pop(seq_id, None)
            if row_idx is None:
                self.seq_id_to_cached_pages.pop(seq_id, None)
                released = self._release_prefix_request(seq_id)
                if released:
                    self._schedule_write_through_prefix_blocks(released)
                return

            cur_len = int(self.row_seq_lens[row_idx])
            num_pages = (cur_len + self.page_size - 1) // self.page_size
            cached_pages = self.seq_id_to_cached_pages.pop(seq_id, set())
            if num_pages > 0:
                # Maintenance snapshots always cover a contiguous page prefix.
                borrowed = seq_id in getattr(self, "_prefix_prune_borrowed_seqs", ())
                private_pages = [
                    page_idx
                    for page_idx in range(len(cached_pages) if borrowed else 0, num_pages)
                    if page_idx not in cached_pages
                ]
                if private_pages:
                    count = len(private_pages)
                    if not self.enable_prefix_caching:
                        free_page_slots = self.buffer_req_to_page_slots_cpu[row_idx, private_pages]
                        self.free_pages_cpu_stack[
                            self._num_free_pages : self._num_free_pages + count
                        ] = free_page_slots
                        page_slots = torch.from_numpy(free_page_slots).to(self.free_pages_stack.device)
                    elif count == num_pages:
                        page_slots = self.buffer_req_to_page_slots[row_idx, :num_pages]
                    else:
                        indices = torch.tensor(private_pages, dtype=torch.long, device=self.device)
                        page_slots = self.buffer_req_to_page_slots[row_idx].index_select(0, indices)
                    ptr = self._num_free_pages
                    self.free_pages_stack[ptr: ptr + count] = page_slots
                    self._num_free_pages += count
            released_prefix_blocks = self._release_prefix_request(seq_id)

            self.buffer_req_to_token_slots[row_idx, :] = 0
            self.buffer_req_to_page_slots[row_idx, :] = -1
            if hasattr(self, "buffer_req_to_page_slots_cpu"):
                self.buffer_req_to_page_slots_cpu[row_idx, :] = -1
            self.row_seq_lens[row_idx] = 0
            self.row_logical_lens[row_idx] = 0
            self.free_rows.append(row_idx)
            self._schedule_write_through_prefix_blocks(released_prefix_blocks)

    def free_part_slots(self, layer_idx: int, seq: Sequence, keep_indices: torch.Tensor):
        raise ValueError("QuEST does not physically evict token slots")

    @torch.no_grad()
    def _prepare_prefill(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_prefill"):
            (
                input_ids_np, positions_np, cu_seqlens_q, slot_mapping,
                context_lens_list, req_indices,
            ) = self._prepare_prefill_rows(seqs)
            metadata_full_pages = True
            page_segments: list[tuple[int, int]] = []

            token_offset = 0
            for seq in seqs:
                chunk_size = seq.current_chunk_size
                start_idx = seq.num_prefilled_tokens
                metadata_full_pages = metadata_full_pages and (
                    int(start_idx) % int(self.page_size) == 0
                    and int(chunk_size) > 0
                    and int(chunk_size) % int(self.page_size) == 0
                )
                segment_token_offset = 0
                while segment_token_offset < chunk_size:
                    page_offset = (int(start_idx) + segment_token_offset) % self.page_size
                    segment_size = min(
                        chunk_size - segment_token_offset,
                        self.page_size - page_offset,
                    )
                    page_segments.append(
                        (token_offset + segment_token_offset, segment_size)
                    )
                    segment_token_offset += segment_size
                token_offset += chunk_size

            context_lens = torch.tensor(context_lens_list, dtype=torch.int32, device=self.device)
            req_indices_tensor = torch.tensor(req_indices, dtype=torch.int32, device=self.device)

            self.layer_batch_state.slot_mapping = slot_mapping
            self.layer_batch_state.context_lens = context_lens
            self.layer_batch_state.max_context_len = max(context_lens_list, default=0)
            self.layer_batch_state.req_indices = req_indices_tensor
            self._prefill_metadata_full_pages = bool(metadata_full_pages)
            self._prefill_page_plan = QuestPrefillPagePlan(
                segments=torch.tensor(
                    page_segments,
                    dtype=torch.int32,
                    device=self.device,
                ).reshape(-1, 2)
            )

            input_ids = torch.from_numpy(input_ids_np).to(self.device)
            positions = torch.from_numpy(positions_np).to(self.device)
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=self.device)
            return input_ids, positions, cu_seqlens_q

    @torch.no_grad()
    def _prepare_decode(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_decode"):
            self._poll_prefix_offload()
            self._prefix_offload_step_h2d_operations = {}
            batch_size = len(seqs)
            input_ids_list = [seq.decode_input_token for seq in seqs]
            positions_list = [seq.decode_input_position for seq in seqs]
            seq_ids = [seq.seq_id for seq in seqs]

            new_slots_batch = self._allocate_batch(seq_ids, 1)
            row_indices = [self.seq_id_to_row[sid] for sid in seq_ids]
            for seq, slot in zip(seqs, new_slots_batch):
                self._record_prefix_materialization(seq, [seq.decode_input_token], slot.reshape(1))
            context_lens = torch.tensor(
                self.row_seq_lens[row_indices],
                dtype=torch.int32,
                device=self.device,
            )
            req_indices = torch.tensor(row_indices, dtype=torch.int32, device=self.device)

            slot_mapping = torch.empty((batch_size,), dtype=torch.int32, device=self.device)
            slot_mapping[:] = new_slots_batch

            self.layer_batch_state.slot_mapping = slot_mapping
            self.layer_batch_state.context_lens = context_lens
            self.layer_batch_state.req_indices = req_indices
            self._prepare_decode_row_page_slots(
                req_indices,
                max_context_len=int(max(self.row_seq_lens[row_indices])),
            )
            self._prepare_decode_page_geometry(context_lens)

            input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device=self.device)
            positions = torch.tensor(positions_list, dtype=torch.int64, device=self.device)
            return input_ids, positions, None

    def init_decode_graph_state(
        self,
        contract: DecodeGraphContract,
        inputs: DecodeGraphInputs,
    ) -> CacheDecodeGraphState:
        inputs.validate(contract)
        if self.enable_prefix_caching:
            return CacheDecodeGraphState(contract=contract, inputs=inputs)

        page_width = min(
            int(self.max_pages_per_row),
            max(
                1,
                (int(contract.context_capacity) + int(self.page_size) - 1)
                // int(self.page_size),
            ),
        )
        batch_capacity = int(contract.batch_capacity)
        return QuestDecodeGraphState(
            contract=contract,
            inputs=inputs,
            row_page_slots=torch.empty(
                (batch_capacity, page_width),
                dtype=torch.int32,
                device=self.device,
            ),
            num_pages=torch.empty(
                batch_capacity,
                dtype=torch.int32,
                device=self.device,
            ),
            previous_page_counts=torch.empty(
                batch_capacity,
                dtype=torch.int32,
                device=self.device,
            ),
            host_write_slots=torch.empty(
                batch_capacity,
                dtype=torch.int32,
                device="cpu",
                pin_memory=bool(inputs.host.input_ids.is_pinned()),
            ),
        )

    def _plan_decode_rows(
        self,
        seq_ids: np.ndarray,
    ) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
        rows = np.empty(int(seq_ids.size), dtype=np.int64)
        pending: list[tuple[int, int]] = []
        free_rows = iter(self.free_rows)
        planned_by_seq: dict[int, int] = {}
        for index, raw_seq_id in enumerate(seq_ids):
            seq_id = int(raw_seq_id)
            row = self.seq_id_to_row.get(seq_id)
            if row is None:
                row = planned_by_seq.get(seq_id)
            if row is None:
                try:
                    row = int(next(free_rows))
                except StopIteration as error:
                    raise RuntimeError(
                        "No free rows for QuEST graph decode: "
                        f"need={len(pending) + 1} free={len(self.free_rows)}."
                    ) from error
                planned_by_seq[seq_id] = row
                pending.append((seq_id, row))
            rows[index] = row
        return rows, tuple(pending)

    def _commit_decode_rows(
        self,
        pending: tuple[tuple[int, int], ...],
    ) -> None:
        for seq_id, expected_row in pending:
            if not self.free_rows or int(self.free_rows[0]) != int(expected_row):
                raise RuntimeError(
                    "QuEST graph decode row plan changed before commit: "
                    f"expected={expected_row} "
                    f"actual={self.free_rows[0] if self.free_rows else None}."
                )
            row = int(self.free_rows.popleft())
            self.seq_id_to_row[int(seq_id)] = row

    def _prepare_decode_graph_step_no_prefix(
        self,
        seqs: list[Sequence],
        state: QuestDecodeGraphState,
    ):
        inputs = state.inputs
        host = inputs.host
        host_write_slots = self.acquire_step_host_buffer(state.host_write_slots)
        real_batch_size = len(seqs)
        graph_batch_size = int(inputs.batch_capacity)
        if real_batch_size <= 0 or real_batch_size > graph_batch_size:
            raise ValueError(
                "QuEST decode graph requires a non-empty active batch within capacity: "
                f"real={real_batch_size} capacity={graph_batch_size}."
            )

        seq_ids = host.pack_requests(seqs)
        row_indices, pending_rows = self._plan_decode_rows(seq_ids)
        current_lens = self.row_seq_lens[row_indices]
        next_lens = current_lens + 1
        max_context_len = int(next_lens.max())
        if max_context_len > int(state.contract.context_capacity):
            raise ValueError(
                "QuEST decode request exceeded the captured graph context capacity: "
                f"requested={max_context_len} "
                f"captured={state.contract.context_capacity}."
            )
        if max_context_len > int(self.max_model_len):
            raise RuntimeError(
                "QuEST decode request exceeded max_model_len: "
                f"requested={max_context_len} max_model_len={self.max_model_len}."
            )

        page_indices = current_lens // int(self.page_size)
        page_offsets = current_lens % int(self.page_size)
        new_page_positions = np.flatnonzero(page_offsets == 0)
        needed_pages = int(new_page_positions.size)
        if needed_pages > 0:
            self._evict_prefix_cache_until_free(needed_pages * int(self.page_size))
        if int(self._num_free_pages) < needed_pages:
            raise RuntimeError(
                "Out of QuEST KV pages during graph reservation: "
                f"need={needed_pages} free={self._num_free_pages}."
            )

        self._commit_decode_rows(pending_rows)
        if needed_pages > 0:
            ptr = int(self._num_free_pages)
            new_page_slots = self.free_pages_cpu_stack[
                ptr - needed_pages : ptr
            ][::-1].copy()
            self.buffer_req_to_page_slots_cpu[
                row_indices[new_page_positions],
                page_indices[new_page_positions],
            ] = new_page_slots
            self._num_free_pages -= needed_pages

        page_slots = self.buffer_req_to_page_slots_cpu[row_indices, page_indices]
        if np.any(page_slots < 0):
            raise RuntimeError(
                "QuEST graph reservation resolved an unallocated physical page."
            )
        allocated_slots = (
            page_slots * int(self.page_size) + page_offsets
        ).astype(np.int32, copy=False)
        self.row_seq_lens[row_indices] = next_lens
        self.row_logical_lens[row_indices] += 1

        host.pack_cache_facts(
            context_lens=next_lens.astype(np.int32, copy=False),
            request_indices=row_indices.astype(np.int32, copy=False),
            real_batch_size=real_batch_size,
            padding_active=bool(state.contract.padding.active),
        )
        host_write_slots[:real_batch_size].copy_(
            torch.from_numpy(allocated_slots)
        )
        if graph_batch_size > real_batch_size:
            host_write_slots[real_batch_size:].fill_(
                int(state.contract.padding.write_slot)
            )

        non_blocking = bool(host.input_ids.is_pinned())
        inputs.input_ids.copy_(host.input_ids, non_blocking=non_blocking)
        inputs.positions.copy_(host.positions, non_blocking=non_blocking)
        inputs.context_lens.copy_(host.context_lens, non_blocking=non_blocking)
        inputs.request_indices.copy_(host.request_indices, non_blocking=non_blocking)
        inputs.active_mask.copy_(host.active_mask, non_blocking=non_blocking)
        inputs.write_slot_mapping.copy_(
            host_write_slots,
            non_blocking=bool(host_write_slots.is_pinned()),
        )

        self.layer_batch_state.slot_mapping = inputs.write_slot_mapping
        self.layer_batch_state.context_lens = inputs.context_lens
        self.layer_batch_state.max_context_len = max_context_len
        self.layer_batch_state.req_indices = inputs.request_indices
        self._decode_row_page_slots = state.row_page_slots
        self._decode_row_page_slots_req_indices = inputs.request_indices
        self._decode_num_pages = state.num_pages
        self._decode_previous_page_counts = state.previous_page_counts
        self._decode_page_geometry_context_lens = inputs.context_lens
        self._decode_page_geometry_ready = False
        return inputs.input_ids, inputs.positions, None

    @torch.no_grad()
    def prepare_decode_graph_step(
        self,
        seqs: list[Sequence],
        state: CacheDecodeGraphState,
    ):
        if not isinstance(state, QuestDecodeGraphState):
            return super().prepare_decode_graph_step(seqs, state)
        with profiler.record("cache_prepare_decode"):
            self._poll_prefix_offload()
            self._prefix_offload_step_h2d_operations = {}
            return self._prepare_decode_graph_step_no_prefix(seqs, state)

    @torch.no_grad()
    def prepare_decode_static(
        self,
        seqs: list[Sequence],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        req_indices: torch.Tensor,
    ):
        """Prepare decode metadata into caller-owned static CUDA buffers."""
        with profiler.record("cache_prepare_decode"):
            self._poll_prefix_offload()
            self._prefix_offload_step_h2d_operations = {}
            real_batch_size = len(seqs)
            graph_batch_size = int(input_ids.numel())
            if real_batch_size <= 0:
                raise ValueError("Static decode requires a non-empty real decode batch.")
            if positions.numel() != graph_batch_size:
                raise ValueError("Static decode input buffers must have the same graph batch size.")
            if (
                slot_mapping.numel() != graph_batch_size
                or context_lens.numel() != graph_batch_size
                or req_indices.numel() != graph_batch_size
            ):
                raise ValueError("Static decode metadata buffers must have the same graph batch size.")
            if real_batch_size > graph_batch_size:
                raise ValueError(
                    "Static decode graph batch is smaller than the real decode batch: "
                    f"graph={graph_batch_size}, real={real_batch_size}."
                )

            input_ids_list = [seq.decode_input_token for seq in seqs]
            positions_list = [seq.decode_input_position for seq in seqs]
            seq_ids = [seq.seq_id for seq in seqs]

            new_slots_batch = self._allocate_batch(seq_ids, 1, graph_batch_size=graph_batch_size)
            row_indices = [self.seq_id_to_row[sid] for sid in seq_ids]
            for seq, slot in zip(seqs, new_slots_batch):
                self._record_prefix_materialization(seq, [seq.decode_input_token], slot.reshape(1))
            real_context_lens = self.row_seq_lens[row_indices]
            input_ids[:real_batch_size].copy_(torch.tensor(input_ids_list, dtype=torch.int64))
            positions[:real_batch_size].copy_(torch.tensor(positions_list, dtype=torch.int64))
            slot_mapping[:real_batch_size].copy_(new_slots_batch)
            context_lens[:real_batch_size].copy_(
                torch.from_numpy(real_context_lens.astype(np.int32, copy=False))
            )
            req_indices[:real_batch_size].copy_(
                torch.tensor(row_indices, dtype=torch.int32)
            )

            if graph_batch_size > real_batch_size:
                input_ids[real_batch_size:].fill_(int(input_ids_list[0]))
                positions[real_batch_size:].fill_(int(positions_list[0]))
                slot_mapping[real_batch_size:].fill_(-1)
                context_lens[real_batch_size:].fill_(int(real_context_lens[0]))
                req_indices[real_batch_size:].fill_(int(row_indices[0]))

            self.layer_batch_state.slot_mapping = slot_mapping
            self.layer_batch_state.context_lens = context_lens
            self.layer_batch_state.max_context_len = int(max(real_context_lens)) if row_indices else 0
            self.layer_batch_state.req_indices = req_indices
            static_max_context_len = getattr(
                self,
                "_decode_static_max_context_len",
                None,
            )
            self._prepare_decode_row_page_slots(
                req_indices,
                max_context_len=(
                    int(static_max_context_len)
                    if static_max_context_len is not None
                    else int(max(real_context_lens))
                ),
            )

            return input_ids, positions, None

    def _prepare_decode_row_page_slots(
        self,
        req_indices: torch.Tensor,
        *,
        max_context_len: int,
    ) -> None:
        """Pack the step's physical page rows once for all decode layers."""

        max_pages_per_row = int(
            getattr(
                self,
                "max_pages_per_row",
                self.buffer_req_to_page_slots.shape[1],
            )
        )
        width = min(
            max_pages_per_row,
            max(1, (int(max_context_len) + self.page_size - 1) // self.page_size),
        )
        key = (int(req_indices.numel()), width)
        if not hasattr(self, "_decode_row_page_slots_buffers"):
            self._decode_row_page_slots_buffers = {}
        packed = self._decode_row_page_slots_buffers.get(key)
        if packed is None:
            packed = torch.empty(key, dtype=torch.int32, device=self.device)
            self._decode_row_page_slots_buffers[key] = packed
        torch.index_select(
            self.buffer_req_to_page_slots[:, :width],
            0,
            req_indices,
            out=packed,
        )
        self._decode_row_page_slots = packed
        self._decode_row_page_slots_req_indices = req_indices
        if not hasattr(self, "_decode_page_geometry_buffers"):
            self._decode_page_geometry_buffers = {}
        geometry = self._decode_page_geometry_buffers.get(int(req_indices.numel()))
        if geometry is None:
            row_buffer = torch.empty_like(req_indices)
            geometry = (row_buffer, torch.empty_like(row_buffer))
            self._decode_page_geometry_buffers[int(req_indices.numel())] = geometry
        self._decode_num_pages, self._decode_previous_page_counts = geometry
        self._decode_page_geometry_context_lens = self.layer_batch_state.context_lens
        self._decode_page_geometry_ready = False

    def _prepare_decode_page_geometry(self, context_lens: torch.Tensor) -> None:
        num_pages = self._decode_num_pages
        previous_page_counts = self._decode_previous_page_counts
        if num_pages is None or previous_page_counts is None:
            raise RuntimeError("QuEST decode page geometry buffers are not initialized.")
        if context_lens.is_cuda:
            prepare_quest_decode_geometry(
                context_lens,
                page_size=self.page_size,
                num_pages=num_pages,
                previous_page_counts=previous_page_counts,
            )
        else:
            torch.div(
                context_lens + self.page_size - 1,
                self.page_size,
                rounding_mode="floor",
                out=num_pages,
            )
            torch.sub(num_pages, 1, out=previous_page_counts)
            previous_page_counts.clamp_min_(0)
        self._decode_page_geometry_context_lens = context_lens
        self._decode_page_geometry_ready = True

    def prepare_decode_graph_in(self, state: CacheDecodeGraphState) -> None:
        if not isinstance(state, QuestDecodeGraphState):
            super().prepare_decode_graph_in(state)
            self._prepare_decode_page_geometry(state.inputs.context_lens)
            return
        prepare_quest_decode_graph_metadata(
            self.buffer_req_to_token_slots,
            self.buffer_req_to_page_slots,
            state.inputs.request_indices,
            state.inputs.context_lens,
            state.inputs.write_slot_mapping,
            state.inputs.active_mask,
            page_size=int(self.page_size),
            row_page_slots=state.row_page_slots,
            num_pages=state.num_pages,
            previous_page_counts=state.previous_page_counts,
        )
        self._decode_page_geometry_ready = True

    def decode_graph_state_keepalive_tensors(
        self,
        state: CacheDecodeGraphState,
    ) -> list[torch.Tensor]:
        tensors = super().decode_graph_state_keepalive_tensors(state)
        if isinstance(state, QuestDecodeGraphState):
            tensors.extend(
                (
                    state.row_page_slots,
                    state.num_pages,
                    state.previous_page_counts,
                    state.host_write_slots,
                )
            )
        return tensors

    @torch.no_grad()
    def _attention_metadata_keys(
        self,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        current_keys: torch.Tensor | None = None,
    ) -> torch.Tensor:
        storage = self.attention_cache_storage
        if isinstance(storage, ExplicitKVStorage):
            if current_keys is not None:
                keys = current_keys
            else:
                payload = storage.layer_payload(self.kv_layer_index(layer_idx))
                keys = payload.k_cache.index_select(
                    0,
                    slot_mapping.to(device=payload.k_cache.device, dtype=torch.long),
                )
        elif isinstance(storage, MlaLatentStorage):
            payload = storage.layer_payload(self.kv_layer_index(layer_idx))
            slots = slot_mapping.to(
                device=payload.latent_cache.device,
                dtype=torch.long,
            )
            keys = torch.cat(
                (
                    payload.latent_cache.index_select(0, slots),
                    payload.rope_cache.index_select(0, slots),
                ),
                dim=-1,
            )
        else:
            raise AssertionError(
                f"Unhandled QuEST storage {type(storage).__name__}."
            )
        expected_shape = (
            int(slot_mapping.numel()),
            self.metadata_num_heads,
            self.metadata_head_dim,
        )
        if tuple(keys.shape) != expected_shape:
            raise RuntimeError(
                "QuEST metadata keys do not match the registered score space: "
                f"layer={layer_idx} expected={expected_shape} got={tuple(keys.shape)}."
            )
        return keys

    @torch.no_grad()
    def on_kv_stored(
        self,
        layer_idx: int,
        k: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        kv_idx = self.kv_layer_index(layer_idx)
        if slot_mapping is None or slot_mapping.numel() == 0:
            return
        if not get_context().is_prefill:
            # Decode stores update QuEST bounds in the same KV/MLA kernel.
            return
        if isinstance(self.attention_cache_storage, ExplicitKVStorage):
            # Explicit-KV prefill updates bounds in the fused physical store.
            return
        metadata_keys = self._attention_metadata_keys(
            layer_idx,
            slot_mapping,
            current_keys=k,
        )
        if self._is_stream_capturing():
            self._on_kv_stored_prefill_capture(
                layer_idx,
                metadata_keys,
                slot_mapping,
            )
            return

        with profiler.record("quest_update_metadata"):
            if self._prefill_metadata_full_pages:
                page_max_cache = self.metadata_cache[0, kv_idx]
                page_min_cache = self.metadata_cache[1, kv_idx]
                full_page_slots = torch.div(
                    slot_mapping[:: self.page_size],
                    self.page_size,
                    rounding_mode="floor",
                ).to(torch.long)
                full_page_k = metadata_keys.view(
                    -1,
                    self.page_size,
                    self.metadata_num_heads,
                    self.metadata_head_dim,
                )
                page_min, page_max = torch.aminmax(full_page_k, dim=1)
                page_max_cache.index_copy_(0, full_page_slots, page_max)
                page_min_cache.index_copy_(0, full_page_slots, page_min)
                return

            page_slots = torch.div(
                slot_mapping,
                self.page_size,
                rounding_mode="floor",
            )
            page_offsets = torch.remainder(slot_mapping, self.page_size)
            unique_pages, counts = torch.unique_consecutive(page_slots, return_counts=True)
            page_max_cache = self.metadata_cache[0, kv_idx]
            page_min_cache = self.metadata_cache[1, kv_idx]
            run_starts = counts.cumsum(0) - counts
            start_offsets = page_offsets.index_select(0, run_starts)
            end_offsets = start_offsets + counts
            page_slots_i64 = unique_pages.to(torch.int64)
            page_token_indices = (
                page_slots_i64[:, None] * self.page_size
                + self.page_offsets_i64[None, :]
            )
            page_keys = self._attention_metadata_keys(
                layer_idx,
                page_token_indices.reshape(-1),
            ).view(
                -1,
                self.page_size,
                self.metadata_num_heads,
                self.metadata_head_dim,
            )
            valid_offsets = (
                self.page_offsets_i64[None, :]
                < end_offsets.to(torch.int64)[:, None]
            )[:, :, None, None]
            page_max = page_keys.masked_fill(
                ~valid_offsets,
                -float("inf"),
            ).amax(dim=1)
            page_min = page_keys.masked_fill(
                ~valid_offsets,
                float("inf"),
            ).amin(dim=1)
            page_max_cache.index_copy_(0, page_slots_i64, page_max)
            page_min_cache.index_copy_(0, page_slots_i64, page_min)

    @torch.no_grad()
    def _on_kv_stored_prefill_capture(
        self,
        layer_idx: int,
        metadata_keys: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """Update QuEST full-page metadata without dynamic shape ops during capture.

        Prefill graph capture is currently exercised for first-prefill chunks, so
        touched pages begin at page offset 0.
        """
        with profiler.record("quest_update_metadata_capture"):
            kv_idx = self.kv_layer_index(layer_idx)
            full_token_count = (int(slot_mapping.numel()) // self.page_size) * self.page_size
            page_max_cache = self.metadata_cache[0, kv_idx]
            page_min_cache = self.metadata_cache[1, kv_idx]
            if full_token_count > 0:
                full_page_slots = torch.div(
                    slot_mapping[:full_token_count:self.page_size],
                    self.page_size,
                    rounding_mode="floor",
                ).to(torch.long)
                full_page_k = metadata_keys[:full_token_count].view(
                    -1,
                    self.page_size,
                    self.metadata_num_heads,
                    self.metadata_head_dim,
                )
                page_min, page_max = torch.aminmax(full_page_k, dim=1)
                page_max_cache.index_copy_(0, full_page_slots, page_max)
                page_min_cache.index_copy_(0, full_page_slots, page_min)

            if full_token_count < int(slot_mapping.numel()):
                partial_page_slot = torch.div(
                    slot_mapping[full_token_count],
                    self.page_size,
                    rounding_mode="floor",
                ).to(torch.long)
                partial_page_k = metadata_keys[full_token_count:]
                page_min, page_max = torch.aminmax(partial_page_k, dim=0)
                page_max_cache.index_copy_(
                    0,
                    partial_page_slot.reshape(1),
                    page_max.unsqueeze(0),
                )
                page_min_cache.index_copy_(
                    0,
                    partial_page_slot.reshape(1),
                    page_min.unsqueeze(0),
                )

    @torch.no_grad()
    def on_forward_end(self, seqs: list[Sequence], is_prefill: bool):
        self._poll_prefix_offload()
        super().on_forward_end(seqs, is_prefill)

    @torch.no_grad()
    def build_decode_view(
        self,
        layer_idx: int,
        q: torch.Tensor | MlaLatentSelectionQuery,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        *,
        num_heads: int,
        num_kv_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del layer_idx, q, num_heads, num_kv_heads
        return active_slots, req_indices, context_lens

    @torch.no_grad()
    def build_decode_compute_view(
        self,
        layer_idx: int,
        q: torch.Tensor | MlaLatentSelectionQuery,
        selection: SparseSelection,
        *,
        num_heads: int,
        num_kv_heads: int,
    ) -> DecodeComputeView:
        if int(num_kv_heads) != self.metadata_num_heads:
            raise ValueError(
                "QuEST attention/metadata head contract is inconsistent: "
                f"attention_kv_heads={num_kv_heads} "
                f"metadata_heads={self.metadata_num_heads}."
            )
        plan = selection.page_selection
        if isinstance(self.attention_cache_storage, MlaLatentStorage) and plan is None:
            return super().build_decode_compute_view(
                layer_idx,
                q,
                selection,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )
        if isinstance(self.attention_cache_storage, MlaLatentStorage):
            selected_pages = self.quest_page_selector.select(
                plan.scores,
                plan.row_page_slots,
                plan.previous_page_counts,
                plan.previous_page_budget,
            )
            dense_slots = self.buffer_req_to_token_slots[:, :plan.max_keep_tokens].index_select(
                0, selection.req_indices.to(torch.long)
            )
            active_slots, req_indices, context_lens = finalize_quest_decode_view(
                selected_pages,
                plan.row_page_slots,
                plan.num_pages.to(torch.int32).contiguous(),
                selection.context_lens,
                dense_slots,
                page_size=self.page_size,
                token_budget=plan.token_budget,
                output_width=plan.max_keep_tokens,
            )
            return DecodeComputeView(
                meta=AttentionViewMeta(
                    active_slots=active_slots,
                    req_indices=req_indices,
                    context_lens=context_lens,
                    max_context_len=min(int(selection.max_context_len), int(active_slots.shape[1])),
                    attn_score=selection.attn_score,
                ),
                payload=self.attention_cache_storage.layer_payload(
                    self.kv_layer_index(layer_idx)
                ),
            )
        del num_heads
        if plan is None:
            dense_width = min(
                self.max_pages_per_row,
                max(1, (int(selection.max_context_len) + self.page_size - 1) // self.page_size),
            )
            page_table, req_indices, context_lens, page_counts, last_page_lens, is_sparse = (
                self._dense_paged_decode_view(
                    selection.req_indices, selection.context_lens, width=dense_width,
                )
            )
        else:
            output_width = max(
                plan.previous_page_budget + 1,
                (plan.token_budget + self.page_size - 1) // self.page_size,
            )
            outputs = self._get_decode_paged_view_buffers(
                int(plan.scores.shape[0]), int(output_width),
            )
            page_table, req_indices, context_lens, page_counts, last_page_lens = (
                self.quest_page_selector.select_and_finalize_paged_view(
                    plan.scores,
                    plan.row_page_slots,
                    plan.previous_page_counts,
                    plan.num_pages.to(torch.int32).contiguous(),
                    selection.context_lens,
                    k=plan.previous_page_budget,
                    page_size=self.page_size,
                    token_budget=plan.token_budget,
                    outputs=outputs,
                    use_dense_fallback=True,
                )
            )
            is_sparse = True
        max_context_len = selection.max_context_len
        if max_context_len is not None:
            max_context_len = min(
                int(max_context_len),
                int(page_table.shape[1]) * self.page_size,
            )
        return DecodeComputeView(
            meta=PagedDecodeViewMeta(
                page_table=page_table,
                req_indices=req_indices,
                context_lens=context_lens,
                page_counts=page_counts,
                last_page_lens=last_page_lens,
                page_size=self.page_size,
                is_sparse=is_sparse,
                max_context_len=max_context_len,
                attn_score=selection.attn_score,
            ),
            payload=self.attention_cache_storage.layer_payload(
                self.kv_layer_index(layer_idx)
            ),
        )

    def _get_decode_paged_view_buffers(
        self,
        batch_size: int,
        width: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        key = (int(batch_size), int(width))
        if not hasattr(self, "_decode_paged_view_buffers"):
            self._decode_paged_view_buffers = {}
        buffers = self._decode_paged_view_buffers.get(key)
        if buffers is None:
            page_table = torch.empty(
                key, dtype=torch.int32, device=self.device
            )
            row_buffer = torch.empty(
                (key[0],), dtype=torch.int32, device=self.device
            )
            buffers = (
                page_table,
                row_buffer,
                torch.empty_like(row_buffer),
                torch.empty_like(row_buffer),
                torch.empty_like(row_buffer),
            )
            self._decode_paged_view_buffers[key] = buffers
        return buffers

    def _dense_paged_decode_view(
        self,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        *,
        width: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        page_counts = torch.div(
            context_lens + self.page_size - 1,
            self.page_size,
            rounding_mode="floor",
        ).to(torch.int32)
        last_page_lens = context_lens - (page_counts - 1) * self.page_size
        return (
            self.buffer_req_to_page_slots[:, :width],
            req_indices,
            context_lens,
            page_counts,
            last_page_lens.to(torch.int32),
            False,
        )

    def quest_decode_page_inputs(
        self,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        *,
        max_pages: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Borrow current physical page rows and replay-safe geometry buffers."""

        batch_size = int(req_indices.numel())
        num_pages = getattr(self, "_decode_num_pages", None)
        previous_page_counts = getattr(
            self, "_decode_previous_page_counts", None
        )
        if (
            num_pages is None
            or previous_page_counts is None
            or getattr(self, "_decode_page_geometry_context_lens", None)
            is not context_lens
            or int(num_pages.numel()) != batch_size
        ):
            num_pages = torch.empty_like(context_lens)
            previous_page_counts = torch.empty_like(context_lens)
            self._decode_num_pages = num_pages
            self._decode_previous_page_counts = previous_page_counts
            self._prepare_decode_page_geometry(context_lens)
        elif not getattr(self, "_decode_page_geometry_ready", False):
            self._prepare_decode_page_geometry(context_lens)
        row_page_slots = getattr(self, "_decode_row_page_slots", None)
        if (
            row_page_slots is None
            or getattr(self, "_decode_row_page_slots_req_indices", None)
            is not req_indices
            or int(row_page_slots.shape[0]) != batch_size
            or int(row_page_slots.shape[1]) < max_pages
        ):
            row_page_slots = self.buffer_req_to_page_slots.index_select(
                0, req_indices
            )[:, :max_pages]
        else:
            row_page_slots = row_page_slots[:, :max_pages]
        return row_page_slots.contiguous(), num_pages, previous_page_counts
