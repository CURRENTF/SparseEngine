from __future__ import annotations

from .deltakv_chain import DeltaKVChainMixin
from .deltakv_less_memory_cuda_graph import DeltaKVLessMemoryCudaGraphCacheManager


class DeltaKVCacheManager(DeltaKVChainMixin, DeltaKVLessMemoryCudaGraphCacheManager):
    """Public DeltaKV runtime, also available as residualkv.

    Sparse residuals may use trained compressors or retain the full KV dimension.
    Raw full layers accept floating-point or int4 residuals; KIVI int4 full layers
    require int4 residuals.

    CUDA Graph support is inherited from the same static decode path.  The graph
    runner owns batch/context bucketing, so this manager does not need separate
    graph-specific public methods.
    """
