from __future__ import annotations

import torch

from sparseengine.config import Config
from sparseengine.engine.activation_controller import ActivationController
from sparseengine.engine.cache_manager import CacheManager, SparseSelection
from sparseengine.engine.sequence import Sequence
from sparseengine.engine.sparse_methods import (
    AttentionEndEvent,
    DecodeSelectionRequest,
    LayerBatchSparseState,
    LayerEndEvent,
    PrefillSelectionRequest,
    SparseStepContext,
    create_sparse_method_runtime,
)
from sparseengine.utils.context import get_context


class SparseController:
    """Method-agnostic sparse lifecycle facade used by the inference engine."""

    def __init__(self, config: Config, cache_manager: CacheManager):
        self.config = config
        self.cache_manager = cache_manager
        self.activation_controller = ActivationController.create(
            config,
            cache_manager,
        )
        self.runtime = create_sparse_method_runtime(config, cache_manager)
        self.sparse_method = self.runtime.sparse_method
        self.layers = None

    @property
    def layer_batch_sparse_states(self) -> dict[int, LayerBatchSparseState]:
        return self.runtime.layer_batch_sparse_states

    @property
    def sparse_config(self) -> dict[str, object]:
        return self.runtime.sparse_config

    def _is_kv_layer(self, layer_idx: int) -> bool:
        return self.runtime._is_kv_layer(layer_idx)

    def _kv_layer_index(self, layer_idx: int) -> int:
        return self.runtime._kv_layer_index(layer_idx)

    def set_tokenizer_metadata(
        self,
        *,
        delimiter_token_ids: list[int] | set[int] | tuple[int, ...] | None = None,
        auxiliary_prefill_token_ids: list[int] | None = None,
        non_execution_token_ids: (
            list[int] | set[int] | tuple[int, ...] | None
        ) = None,
    ) -> None:
        self.runtime.set_auxiliary_prefill_prompt(auxiliary_prefill_token_ids or [])
        self.activation_controller.set_tokenizer_metadata(
            delimiter_token_ids=delimiter_token_ids,
            non_execution_token_ids=non_execution_token_ids,
        )

    def bind_auxiliary_prefill(self, executor) -> None:
        self.runtime.bind_auxiliary_prefill(executor)

    @property
    def requires_committed_token_history(self) -> bool:
        return bool(self.runtime.requires_committed_token_history
                    or self.activation_controller.requires_committed_token_history
                    or self.cache_manager.requires_committed_token_history)

    def prepare_decode_replay(self, seqs: list[Sequence]) -> None:
        # Python activation policy is not replayed by a CUDA Graph. Refresh its
        # real-row metadata and stable device inputs for every submission.
        context = get_context()
        context.sparse_controller = self
        context.sparse_config = self.sparse_config if self.sparse_method else None
        self.activation_controller.prepare_forward(seqs, False)

    def clear_decode_attn_score_buffers(self) -> None:
        self.runtime.clear_decode_attn_score_buffers()

    def decode_graph_keepalive_tensors(self) -> list[torch.Tensor]:
        return [
            *self.activation_controller.decode_graph_keepalive_tensors(),
            *self.runtime.decode_graph_keepalive_tensors(),
        ]

    def reset_decode_attn_scores_for_graph(
        self,
        refs: dict[int, dict[str, object]],
    ) -> bool:
        return self.runtime.reset_decode_attn_scores_for_graph(refs)

    def debug_state_summary(self) -> dict[str, object]:
        return self.runtime.debug_state_summary()

    def set_modules(self, modules) -> None:
        self.layers = modules

    def apply_activation_hook(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        context,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.activation_controller.apply_layer_hook(
            layer_idx,
            hidden_states,
            residual,
            context,
        )

    @torch.no_grad()
    def prepare_forward(self, seqs: list[Sequence], is_prefill: bool) -> None:
        context = get_context()
        context.sparse_config = self.sparse_config if self.sparse_method else None
        self.activation_controller.prepare_forward(seqs, is_prefill)
        self.runtime.prepare_step(
            SparseStepContext(
                seqs=seqs,
                is_prefill=is_prefill,
                forward_context=context,
            )
        )

    def get_layer_max_context_len(self, layer_idx: int) -> int | None:
        return self.runtime.get_layer_max_context_len(layer_idx)

    def get_prefill_selection(self, layer_idx: int) -> SparseSelection:
        return self.runtime.build_prefill_selection(
            PrefillSelectionRequest(
                layer_idx=layer_idx,
                forward_context=get_context(),
            )
        )

    def get_decode_selection(
        self,
        layer_idx: int,
        q: torch.Tensor,
        active_slots: torch.Tensor | None = None,
        req_indices: torch.Tensor | None = None,
        context_lens: torch.Tensor | None = None,
        *,
        selection_query=None,
    ) -> SparseSelection:
        del active_slots, req_indices, context_lens
        return self.runtime.build_decode_selection(
            DecodeSelectionRequest(
                layer_idx=layer_idx,
                query=q,
                forward_context=get_context(),
                selection_query=selection_query,
            )
        )

    @torch.no_grad()
    def on_layer_attention_end(self, layer_idx: int) -> None:
        self.runtime.on_attention_end(
            AttentionEndEvent(
                layer_idx=layer_idx,
                forward_context=get_context(),
            )
        )

    def on_layer_end(self, layer_idx: int, context) -> None:
        self.runtime.on_layer_end(
            LayerEndEvent(
                layer_idx=layer_idx,
                layer_context=context,
                forward_context=get_context(),
            )
        )

    @torch.no_grad()
    def post_forward(self, seqs: list[Sequence], is_prefill: bool) -> None:
        self.activation_controller.post_forward(seqs, is_prefill)
        self.runtime.finish_step(
            SparseStepContext(
                seqs=seqs,
                is_prefill=is_prefill,
                forward_context=get_context(),
            )
        )


__all__ = ["LayerBatchSparseState", "SparseController"]
