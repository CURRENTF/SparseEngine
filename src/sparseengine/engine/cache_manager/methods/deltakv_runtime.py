from __future__ import annotations

from .deltakv_less_memory_cuda_graph import DeltaKVLessMemoryCudaGraphCacheManager


class DeltaKVCacheManager(DeltaKVLessMemoryCudaGraphCacheManager):
    """Slim public DeltaKV runtime.

    Supported storage paths:
    1. raw BF16/FP16 full layers + BF16/FP16 compressor latent residual sparse layers;
    2. KIVI int4 full layers + int4 compressor latent residual sparse layers.

    CUDA Graph support is inherited from the same static decode path.  The graph
    runner owns batch/context bucketing, so this manager does not need separate
    graph-specific public methods.
    """
