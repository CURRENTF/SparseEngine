from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from sparseengine.engine.decode_graph_contract import (
    DecodeGraphContract,
    DecodeGraphInputs,
)
from sparseengine.kernels.external.flashinfer.decode import (
    make_flashinfer_paged_decode_wrapper,
)
from sparseengine.kernels.triton.flashinfer_decode_metadata import (
    pack_flashinfer_page_indices,
)

if TYPE_CHECKING:
    from sparseengine.operators.decode_attention import DecodeAttentionOpSpec


class FlashInferPagedDecodeState:
    """Provider-owned eager plan state for FlashInfer paged decode."""

    def __init__(self, device: torch.device) -> None:
        self.workspace = torch.empty(
            128 * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )
        self.wrapper = make_flashinfer_paged_decode_wrapper(self.workspace)
        self.plan_key: tuple[object, ...] | None = None
        self.indices = torch.empty(0, dtype=torch.int32, device=device)

    def plan(
        self,
        spec: DecodeAttentionOpSpec,
        *,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
    ) -> None:
        if active_slots.dtype != torch.int32 or active_slots.ndim != 2:
            raise TypeError(
                "FlashInfer decode requires a rank-2 int32 physical-slot page table."
            )
        if req_indices.dtype != torch.int32 or context_lens.dtype != torch.int32:
            raise TypeError("FlashInfer decode requires int32 request metadata.")
        batch_size = int(context_lens.numel())
        if batch_size <= 0 or int(req_indices.numel()) != batch_size:
            raise ValueError("FlashInfer decode requires matched non-empty metadata.")
        max_context_len = int(max_context_len)
        page_size = int(spec.page_size)
        max_page_count = (max_context_len + page_size - 1) // page_size
        if max_context_len <= 0 or max_page_count > int(active_slots.shape[1]):
            raise ValueError(
                "FlashInfer decode context is outside the active slot table: "
                f"max_context_len={max_context_len} "
                f"max_page_count={max_page_count} "
                f"width={int(active_slots.shape[1])}."
            )
        capacity = batch_size * max_page_count
        if self.indices.numel() < capacity:
            self.indices = torch.empty(
                capacity, dtype=torch.int32, device=context_lens.device
            )
        indices = self.indices[:capacity]
        pack_flashinfer_page_indices(
            active_slots,
            req_indices.contiguous(),
            context_lens.contiguous(),
            indices,
            context_capacity=max_context_len,
            page_size=page_size,
        )
        page_counts = torch.div(
            context_lens + page_size - 1,
            page_size,
            rounding_mode="floor",
        )
        indptr = torch.cat(
            (
                torch.zeros(1, device=context_lens.device, dtype=torch.int32),
                page_counts.cumsum(0, dtype=torch.int32),
            )
        )
        last_page_len = context_lens - (page_counts - 1) * page_size
        self.wrapper.plan(
            indptr,
            indices,
            last_page_len,
            num_qo_heads=spec.num_query_heads,
            num_kv_heads=spec.num_kv_heads,
            head_dim=spec.head_dim,
            page_size=spec.page_size,
            sm_scale=spec.softmax_scale,
            q_data_type=spec.activation_dtype,
            kv_data_type=spec.activation_dtype,
            non_blocking=True,
        )


class FlashInferPagedDecodeGraphState:
    """Graph-stable FlashInfer plans and layer-varying page-index buffers."""

    def __init__(
        self,
        spec: DecodeAttentionOpSpec,
        *,
        contract: DecodeGraphContract,
        inputs: DecodeGraphInputs,
    ) -> None:
        self.spec = spec
        self.contract = contract
        self.inputs = inputs
        device = inputs.context_lens.device
        batch_size = int(contract.batch_capacity)
        context_capacity = int(contract.context_capacity)
        self.workspace = torch.empty(
            128 * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )
        self.indptr = torch.empty(
            batch_size + 1,
            dtype=torch.int32,
            device=device,
        )
        self.indices = torch.empty(
            batch_size * context_capacity,
            dtype=torch.int32,
            device=device,
        )
        self.last_page_len = torch.ones(
            batch_size,
            dtype=torch.int32,
            device=device,
        )
        pin_memory = bool(inputs.host.context_lens.is_pinned())
        self.host_indptr = torch.empty(
            batch_size + 1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.host_last_page_len = torch.ones(
            batch_size,
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.wrapper = make_flashinfer_paged_decode_wrapper(
            self.workspace,
            use_cuda_graph=True,
            paged_kv_indptr_buffer=self.indptr,
            paged_kv_indices_buffer=self.indices,
            paged_kv_last_page_len_buffer=self.last_page_len,
        )
        self.sparse_indptr: torch.Tensor | None = None
        self.sparse_indices: torch.Tensor | None = None
        self.sparse_last_page_len: torch.Tensor | None = None
        self.host_sparse_indptr: torch.Tensor | None = None
        self.host_sparse_last_page_len: torch.Tensor | None = None
        self.sparse_wrapper = None
        if spec.sparse_context_budget is not None:
            self.sparse_indptr = torch.empty_like(self.indptr)
            self.sparse_indices = torch.empty_like(self.indices)
            self.sparse_last_page_len = torch.ones_like(self.last_page_len)
            self.host_sparse_indptr = torch.empty_like(self.host_indptr)
            self.host_sparse_last_page_len = torch.ones_like(
                self.host_last_page_len
            )
            self.sparse_wrapper = make_flashinfer_paged_decode_wrapper(
                self.workspace,
                use_cuda_graph=True,
                paged_kv_indptr_buffer=self.sparse_indptr,
                paged_kv_indices_buffer=self.sparse_indices,
                paged_kv_last_page_len_buffer=self.sparse_last_page_len,
            )
        self.planned = False
        self._eager_page_pack_keys: dict[bool, tuple[object, ...] | None] = {
            False: None,
            True: None,
        }
        self._captured_page_pack_keys: dict[bool, tuple[object, ...] | None] = {
            False: None,
            True: None,
        }
        self._capture_pack_started = False

    def _plan(
        self,
        wrapper: Any,
        host_indptr: torch.Tensor,
        indices: torch.Tensor,
        host_last_page_len: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> None:
        page_size = int(self.spec.page_size)
        page_counts = torch.div(
            context_lens + page_size - 1,
            page_size,
            rounding_mode="floor",
        )
        host_indptr[0] = 0
        torch.cumsum(
            page_counts, dim=0, dtype=torch.int32, out=host_indptr[1:]
        )
        host_last_page_len.copy_(
            context_lens - (page_counts - 1) * page_size
        )
        total_pages = int(host_indptr[-1])
        wrapper.plan(
            host_indptr,
            indices[:total_pages],
            host_last_page_len,
            num_qo_heads=self.spec.num_query_heads,
            num_kv_heads=self.spec.num_kv_heads,
            head_dim=self.spec.head_dim,
            page_size=self.spec.page_size,
            sm_scale=self.spec.softmax_scale,
            q_data_type=self.spec.activation_dtype,
            kv_data_type=self.spec.activation_dtype,
            non_blocking=True,
        )

    def prepare_out_graph(self) -> None:
        dense_context_lens = self.inputs.host.context_lens
        if torch.any(dense_context_lens <= 0):
            raise ValueError(
                "FlashInfer graph decode requires positive context lengths."
            )
        if torch.any(dense_context_lens > self.contract.context_capacity):
            raise ValueError(
                "FlashInfer graph decode context exceeds its captured capacity."
            )
        self._plan(
            self.wrapper,
            self.host_indptr,
            self.indices,
            self.host_last_page_len,
            dense_context_lens,
        )
        if self.sparse_wrapper is not None:
            self._plan_sparse(dense_context_lens)
        self.planned = True

    def _plan_sparse(self, dense_context_lens: torch.Tensor) -> None:
        assert self.spec.sparse_context_budget is not None
        assert self.sparse_wrapper is not None
        assert self.host_sparse_indptr is not None
        assert self.sparse_indices is not None
        assert self.host_sparse_last_page_len is not None
        page_size = int(self.spec.page_size)
        page_budget = max(
            3,
            int(self.spec.sparse_context_budget) // page_size,
        )
        prev_budget = min(
            page_budget - 1,
            (int(self.contract.context_capacity) + page_size - 1)
            // page_size
            - 1,
        )
        last_page_lens = torch.remainder(
            dense_context_lens - 1,
            page_size,
        ) + 1
        num_pages = torch.div(dense_context_lens + page_size - 1, page_size, rounding_mode="floor")
        use_dense = (dense_context_lens <= int(self.spec.sparse_context_budget)) | (num_pages <= prev_budget + 1)
        sparse_context_lens = torch.where(
            use_dense, dense_context_lens, prev_budget * page_size + last_page_lens,
        )
        self._plan(
            self.sparse_wrapper,
            self.host_sparse_indptr,
            self.sparse_indices,
            self.host_sparse_last_page_len,
            sparse_context_lens,
        )

    def begin_graph_in(self) -> None:
        """Reset page-pack deduplication at one warmup/capture boundary."""

        if torch.cuda.is_current_stream_capturing():
            self._captured_page_pack_keys = {False: None, True: None}
            self._capture_pack_started = True
        else:
            self._eager_page_pack_keys = {False: None, True: None}
            self._capture_pack_started = False

    def wrapper_for(self, is_sparse: bool) -> Any:
        if not is_sparse:
            return self.wrapper
        if self.sparse_wrapper is None:
            raise RuntimeError(
                "FlashInfer graph received a sparse paged view without a "
                "prepared sparse plan."
            )
        return self.sparse_wrapper

    @staticmethod
    def _page_pack_key(
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> tuple[object, ...]:
        return (
            active_slots.device,
            active_slots.dtype,
            active_slots.data_ptr(),
            tuple(active_slots.shape),
            tuple(active_slots.stride()),
            req_indices.device,
            req_indices.dtype,
            req_indices.data_ptr(),
            tuple(req_indices.shape),
            tuple(req_indices.stride()),
            context_lens.device,
            context_lens.dtype,
            context_lens.data_ptr(),
            tuple(context_lens.shape),
            tuple(context_lens.stride()),
        )

    def pack_page_indices_once(
        self,
        *,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        is_sparse: bool = False,
        force: bool = False,
    ) -> bool:
        """Pack unless the destination already holds this exact paged view."""

        if not self.planned:
            raise RuntimeError(
                "FlashInfer graph decode was not planned before forward."
            )
        is_capturing = torch.cuda.is_current_stream_capturing()
        if is_capturing:
            if not self._capture_pack_started:
                self._captured_page_pack_keys = {False: None, True: None}
                self._capture_pack_started = True
            keys = self._captured_page_pack_keys
        else:
            keys = self._eager_page_pack_keys
        key = self._page_pack_key(active_slots, req_indices, context_lens)
        if not force and key == keys[bool(is_sparse)]:
            return False

        packed_indices = self.indices
        if is_sparse:
            if self.sparse_indices is None:
                raise RuntimeError(
                    "FlashInfer sparse page-index storage is unavailable."
                )
            packed_indices = self.sparse_indices
        pack_flashinfer_page_indices(
            active_slots,
            req_indices,
            context_lens,
            packed_indices,
            context_capacity=min(
                int(self.contract.context_capacity),
                int(active_slots.shape[1]) * int(self.spec.page_size),
            ),
            page_size=int(self.spec.page_size),
        )
        keys[bool(is_sparse)] = key
        return True

    def keepalive_tensors(self) -> list[torch.Tensor]:
        tensors = [
            self.workspace,
            self.indptr,
            self.indices,
            self.last_page_len,
            self.host_indptr,
            self.host_last_page_len,
        ]
        for tensor in (
            self.sparse_indptr,
            self.sparse_indices,
            self.sparse_last_page_len,
            self.host_sparse_indptr,
            self.host_sparse_last_page_len,
        ):
            if tensor is not None:
                tensors.append(tensor)
        return tensors


__all__ = [
    "FlashInferPagedDecodeGraphState",
    "FlashInferPagedDecodeState",
]
