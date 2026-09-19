from sparseengine.engine.cache_manager import SparseSelection

from .base import (
    DecodeSelectionRequest,
    PrefillSelectionRequest,
    SparseMethodRuntime,
    SparseStepContext,
)


class PassThroughRuntime(SparseMethodRuntime):
    """Full logical selection for methods whose sparse view is cache-owned."""

    def needs_attention_score(
        self,
        layer_idx: int,
        step: SparseStepContext,
    ) -> bool:
        del layer_idx, step
        return False

    def build_prefill_selection(
        self,
        request: PrefillSelectionRequest,
    ) -> SparseSelection:
        return self._full_selection(request.layer_idx)

    def build_decode_selection(
        self,
        request: DecodeSelectionRequest,
    ) -> SparseSelection:
        return self._full_selection(request.layer_idx)
