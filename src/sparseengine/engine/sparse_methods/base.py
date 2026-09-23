from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Any, Callable

import torch

from sparseengine.config import Config
from sparseengine.engine.cache_manager import CacheManager, SparseSelection
from sparseengine.engine.cache_manager.base import (
    _debug_tensor_summary,
    _debug_value_summary,
)
from sparseengine.engine.sequence import Sequence
from sparseengine.method_registry import normalize_sparse_method
from sparseengine.models.layout import resolve_attention_qk_head_dim
from sparseengine.utils.profiler import profiler


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off; got {value!r}."
    )


@dataclass
class LayerBatchSparseState:
    """Logical sparse state for one layer in the current batch."""

    attn_score: torch.Tensor | None = None
    active_indices: torch.Tensor | None = None
    active_slots: torch.Tensor | None = None
    req_indices: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    max_context_len: int | None = None
    active_compressed_indices: torch.Tensor | None = None
    global_req_indices: torch.Tensor | None = None
    deltakv_free_temp_slots: bool = False


@dataclass(frozen=True)
class AuxiliaryPrefillRequest:
    """Unsampled teacher-forced tokens appended to an existing cache row."""

    seq: Sequence
    token_ids: tuple[int, ...]
    prefix_len: int


@dataclass(frozen=True)
class SparseStepContext:
    seqs: list[Sequence]
    is_prefill: bool
    forward_context: Any


@dataclass(frozen=True)
class PrefillSelectionRequest:
    layer_idx: int
    forward_context: Any


@dataclass(frozen=True)
class DecodeSelectionRequest:
    layer_idx: int
    query: torch.Tensor
    forward_context: Any
    selection_query: Any = None


@dataclass(frozen=True)
class AttentionEndEvent:
    layer_idx: int
    forward_context: Any


@dataclass(frozen=True)
class LayerEndEvent:
    layer_idx: int
    layer_context: Any
    forward_context: Any


class SparseMethodRuntime(ABC):
    """Method-owned logical sparse runtime behind the controller facade."""

    def __init__(self, config: Config, cache_manager: CacheManager):
        self.config = config
        self.cache_manager = cache_manager
        self.sparse_method = normalize_sparse_method(config.sparse_method)
        self.validate_runtime_invariants = bool(
            getattr(config, "validate_runtime_invariants", False)
        )
        self.device = getattr(
            cache_manager,
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.obs_layer_ids = self.config.obs_layer_ids
        self.full_attention_layers = self.config.full_attention_layers
        self.num_layers = self.config.hf_config.num_hidden_layers
        self.num_sink = self.config.sink_keep_tokens
        self.num_recent = self.config.recent_keep_tokens
        self.decode_keep_tokens = self.config.decode_keep_tokens

        layout_dims = tuple(
            getattr(getattr(self.config, "runtime_layout", None), "kv_head_dims", ())
        )
        head_dim = (
            int(layout_dims[0])
            if layout_dims
            else resolve_attention_qk_head_dim(self.config.hf_config)
        )
        self.attn_softmax_scale = float(head_dim) ** -0.5
        score_dtype_name = str(
            getattr(self.config, "sparse_attn_score_dtype", "float32") or "float32"
        ).lower()
        self.attn_score_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[score_dtype_name]
        self.snapkv_decode_score_dtype = torch.float32

        self.layer_batch_sparse_states = {
            layer_idx: LayerBatchSparseState()
            for layer_idx in range(self.num_layers)
        }
        self._decode_attn_score_buffers: dict[int, torch.Tensor] = {}
        self.debug_dynamic_selection: dict[str, object] = {}
        self.debug_dynamic_selection_detail = os.environ.get(
            "SPARSEENGINE_DEBUG_DYNAMIC_SELECTION_DETAIL", ""
        ).lower() in ("1", "true", "yes", "on")
        self.dynamic_deltakv_topk_tiebreak = _env_bool(
            "SPARSEENGINE_DELTAKV_DETERMINISTIC_TOPK_TIEBREAK",
            False,
        )
        self.sparse_config = {
            "sparse_method": self.sparse_method,
            "sink_keep_tokens": self.config.sink_keep_tokens,
            "recent_keep_tokens": self.config.recent_keep_tokens,
            "decode_keep_tokens": self.config.decode_keep_tokens,
            "obs_layer_ids": self.config.obs_layer_ids,
            "full_attention_layers": self.config.full_attention_layers,
            "dynamic_deltakv_topk_tiebreak": self.dynamic_deltakv_topk_tiebreak,
        }

    def bind_auxiliary_prefill(
        self, executor: Callable[[AuxiliaryPrefillRequest], None],
    ) -> None:
        self.auxiliary_prefill = executor

    def set_auxiliary_prefill_prompt(self, token_ids: list[int]) -> None:
        if token_ids:
            raise ValueError("This sparse runtime does not accept an auxiliary prefill prompt.")

    def _is_kv_layer(self, layer_idx: int) -> bool:
        runtime_layout = getattr(self.config, "runtime_layout", None)
        if runtime_layout is None:
            return True
        return bool(runtime_layout.is_full_attention(int(layer_idx)))

    def _kv_layer_index(self, layer_idx: int) -> int:
        runtime_layout = getattr(self.config, "runtime_layout", None)
        if runtime_layout is None:
            return int(layer_idx)
        return int(runtime_layout.kv_layer_index(int(layer_idx)))

    def _state_max_context_len(self, state: LayerBatchSparseState) -> int:
        if state.max_context_len is not None:
            return int(state.max_context_len)
        return int(state.context_lens.max().item())

    def get_layer_max_context_len(self, layer_idx: int) -> int | None:
        return self.layer_batch_sparse_states[layer_idx].max_context_len

    def clear_decode_attn_score_buffers(self) -> None:
        self._decode_attn_score_buffers.clear()

    def decode_graph_keepalive_tensors(self) -> list[torch.Tensor]:
        return []

    def reset_decode_attn_scores_for_graph(
        self,
        refs: dict[int, dict[str, object]],
    ) -> bool:
        del refs
        return False

    def _debug_record_dynamic_selection(
        self,
        bucket: str,
        layer_idx: int,
        **fields,
    ) -> None:
        entry = self.debug_dynamic_selection.setdefault(bucket, {}).setdefault(
            str(int(layer_idx)),
            {"calls": 0},
        )
        entry["calls"] += 1
        entry.update(fields)

    @staticmethod
    def _debug_tensor_preview(tensor: torch.Tensor, limit: int = 16):
        preview = tensor.detach().flatten()[:limit].cpu()
        if preview.dtype in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.long,
            torch.bool,
        ):
            return [int(value) for value in preview.tolist()]
        return [float(value) for value in preview.tolist()]

    def debug_state_summary(self) -> dict[str, object]:
        layers = {}
        for layer_idx, state in sorted(self.layer_batch_sparse_states.items()):
            tensors = {}
            for name in (
                "active_indices",
                "active_slots",
                "active_compressed_indices",
                "req_indices",
                "global_req_indices",
                "context_lens",
            ):
                tensor = getattr(state, name)
                if tensor is not None:
                    tensors[name] = _debug_tensor_summary(tensor)
            if tensors or state.max_context_len is not None:
                layers[str(layer_idx)] = {
                    "max_context_len": state.max_context_len,
                    "tensors": tensors,
                }
        return {
            "sparse_method": str(self.sparse_method or ""),
            "layers": layers,
            "dynamic_selection": _debug_value_summary(
                self.debug_dynamic_selection
            ),
            "cache": self.cache_manager.debug_state_summary(),
        }

    def _decode_softmax_token_scores(
        self,
        scores: torch.Tensor,
        *,
        candidate_start: int,
        candidate_lens: torch.Tensor,
    ) -> torch.Tensor:
        if scores.dim() != 3:
            raise ValueError(
                "Expected decode scores with shape [B, H, L], got "
                f"{tuple(scores.shape)}."
            )
        candidate_start = int(candidate_start)
        if candidate_start < 0 or candidate_start > scores.shape[-1]:
            raise ValueError(
                "candidate_start must be within score length; got "
                f"{candidate_start} for L={scores.shape[-1]}."
            )
        candidate_scores = scores[:, :, candidate_start:]
        candidate_lens = (
            candidate_lens.to(device=scores.device, dtype=torch.long)
            .clamp_min(0)
            .clamp_max(candidate_scores.shape[-1])
        )
        candidate_pos = torch.arange(
            candidate_scores.shape[-1],
            device=scores.device,
        )
        candidate_mask = candidate_pos.unsqueeze(0) < candidate_lens.unsqueeze(1)

        logits = candidate_scores.float() * float(self.attn_softmax_scale)
        logits = logits.masked_fill(
            ~candidate_mask[:, None, :],
            torch.finfo(logits.dtype).min,
        )
        candidate_token_scores = torch.softmax(logits, dim=-1).max(dim=1).values

        model_dtype = self.config.hf_config.dtype
        if isinstance(model_dtype, str):
            model_dtype = {
                "float16": torch.float16,
                "torch.float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "torch.bfloat16": torch.bfloat16,
            }.get(model_dtype.lower())
        if model_dtype in (torch.float16, torch.bfloat16):
            candidate_token_scores = candidate_token_scores.to(model_dtype)
        min_score = torch.finfo(candidate_token_scores.dtype).min
        candidate_token_scores = candidate_token_scores.masked_fill(
            ~candidate_mask,
            min_score,
        )
        token_scores = torch.full(
            (scores.shape[0], scores.shape[-1]),
            min_score,
            dtype=candidate_token_scores.dtype,
            device=candidate_token_scores.device,
        )
        token_scores[:, candidate_start:] = candidate_token_scores
        return token_scores

    requires_committed_token_history = False

    def prepare_step(self, step: SparseStepContext) -> None:
        self._begin_prepare_step(step)
        for layer_idx in range(self.num_layers):
            state = self.layer_batch_sparse_states[layer_idx]
            if not self._is_kv_layer(layer_idx):
                state.context_lens = None
                state.max_context_len = None
                state.req_indices = None
                state.global_req_indices = None
                state.attn_score = None
                state.active_indices = None
                state.active_slots = None
                state.active_compressed_indices = None
                state.deltakv_free_temp_slots = False
                continue

            batch_state = self.cache_manager.get_layer_batch_states(layer_idx)
            state.context_lens = batch_state.context_lens
            state.max_context_len = batch_state.max_context_len
            state.req_indices = batch_state.req_indices
            state.global_req_indices = batch_state.req_indices
            state.attn_score = None
            state.active_indices = None
            state.active_slots = None
            state.active_compressed_indices = None
            state.deltakv_free_temp_slots = False

            if not self.needs_attention_score(layer_idx, step):
                continue
            batch_size = (
                int(state.context_lens.numel())
                if not step.is_prefill and state.context_lens is not None
                else len(step.seqs)
            )
            num_heads = (
                self.config.hf_config.num_attention_heads
                // self.config.tensor_parallel_size
            )
            max_len = self._state_max_context_len(state)
            with profiler.record("sparse_prepare_attn_score"):
                if step.is_prefill:
                    self._prepare_prefill_attention_score(
                        state, batch_size, num_heads, max_len
                    )
                else:
                    self._prepare_decode_attention_score(
                        layer_idx,
                        state,
                        batch_size,
                        num_heads,
                        max_len,
                    )
        self._end_prepare_step(step)

    def _begin_prepare_step(self, step: SparseStepContext) -> None:
        del step

    def _end_prepare_step(self, step: SparseStepContext) -> None:
        del step

    def prefill_score_shape(
        self,
        batch_size: int,
        num_heads: int,
        max_len: int,
    ) -> tuple[int, ...]:
        return batch_size, num_heads, max_len

    def prefill_score_fill_value(self) -> float:
        return 0.0

    def _prepare_prefill_attention_score(
        self, state, batch_size: int, num_heads: int, max_len: int
    ) -> None:
        state.attn_score = torch.full(
            self.prefill_score_shape(batch_size, num_heads, max_len),
            self.prefill_score_fill_value(),
            dtype=self.attn_score_dtype,
            device=self.device,
        )

    def _prepare_decode_attention_score(
        self,
        layer_idx: int,
        state: LayerBatchSparseState,
        batch_size: int,
        num_heads: int,
        max_len: int,
    ) -> None:
        state.attn_score = self._get_decode_attn_score_buffer(
            layer_idx,
            batch_size,
            num_heads,
            max_len,
            fill_value=-1e20,
        )

    def _get_decode_attn_score_buffer(
        self,
        layer_idx: int,
        batch_size: int,
        num_heads: int,
        max_len: int,
        *,
        fill_value: float,
    ) -> torch.Tensor:
        if batch_size <= 0 or num_heads <= 0 or max_len <= 0:
            raise RuntimeError(
                "Decode attention score buffer requires positive shape: "
                f"layer={layer_idx} batch={batch_size} heads={num_heads} "
                f"max_len={max_len}."
            )
        buffer = self._decode_attn_score_buffers.get(int(layer_idx))
        needs_alloc = (
            buffer is None
            or buffer.dtype != self.attn_score_dtype
            or buffer.device != self.device
            or int(buffer.shape[0]) < int(batch_size)
            or int(buffer.shape[1]) < int(num_heads)
            or int(buffer.shape[2]) < int(max_len)
        )
        if needs_alloc:
            buffer = torch.empty(
                (int(batch_size), int(num_heads), int(max_len)),
                dtype=self.attn_score_dtype,
                device=self.device,
            )
            self._decode_attn_score_buffers[int(layer_idx)] = buffer
        view = buffer[:batch_size, :num_heads, :max_len]
        view.fill_(fill_value)
        return view

    def _full_selection(self, layer_idx: int) -> SparseSelection:
        if not self._is_kv_layer(layer_idx):
            raise RuntimeError(
                f"layer_idx={layer_idx} is linear_attention and has no KV sparse selection"
            )
        state = self.layer_batch_sparse_states[layer_idx]
        return SparseSelection(
            kind="full",
            req_indices=state.req_indices,
            context_lens=state.context_lens,
            max_context_len=state.max_context_len,
            attn_score=state.attn_score,
            global_req_indices=state.global_req_indices,
        )

    @abstractmethod
    def needs_attention_score(
        self,
        layer_idx: int,
        step: SparseStepContext,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def build_prefill_selection(
        self,
        request: PrefillSelectionRequest,
    ) -> SparseSelection:
        raise NotImplementedError

    @abstractmethod
    def build_decode_selection(
        self,
        request: DecodeSelectionRequest,
    ) -> SparseSelection:
        raise NotImplementedError

    def on_attention_end(self, event: AttentionEndEvent) -> None:
        del event

    def on_layer_end(self, event: LayerEndEvent) -> None:
        del event

    def finish_step(self, step: SparseStepContext) -> None:
        del step
