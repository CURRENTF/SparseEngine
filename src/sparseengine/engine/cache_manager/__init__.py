from __future__ import annotations

from .base import (
    AttentionCacheWrite,
    AttentionKeyComputeView,
    AttentionPayload,
    AttentionViewMeta,
    CacheManager,
    DecodeComputeView,
    ExplicitKVPayload,
    ExplicitKVWrite,
    LayerBatchStates,
    MlaLatentPayload,
    MlaLatentSelectionQuery,
    MlaLatentWrite,
    PagedDecodeViewMeta,
    PrefillComputeView,
    SparseSelection,
)

__all__ = [
    "AttentionCacheWrite",
    "AttentionKeyComputeView",
    "AttentionPayload",
    "AttentionViewMeta",
    "CacheManager",
    "DecodeComputeView",
    "ExplicitKVPayload",
    "ExplicitKVWrite",
    "LayerBatchStates",
    "MlaLatentPayload",
    "MlaLatentSelectionQuery",
    "MlaLatentWrite",
    "PagedDecodeViewMeta",
    "PrefillComputeView",
    "SparseSelection",
    "StandardCacheManager",
    "QuantizedCacheManager",
    "StreamingLLMCacheManager",
    "SnapKVCacheManager",
    "H2OCacheManager",
    "RKVCacheManager",
    "SkipKVCacheManager",
    "QuestCacheManager",
    "OmniKVCacheManager",
    "DeltaKVCacheManager",
    "DeltaKVCacheTritonManagerV4",
    "DeltaKVLessMemoryCacheManager",
    "DeltaKVLessMemoryCudaGraphCacheManager",
]


def __getattr__(name: str):
    if name == "QuantizedCacheManager":
        from .quantized import QuantizedCacheManager

        return QuantizedCacheManager
    if name == "StandardCacheManager":
        from .standard import StandardCacheManager

        return StandardCacheManager
    if name == "StreamingLLMCacheManager":
        from .methods.streamingllm import StreamingLLMCacheManager

        return StreamingLLMCacheManager
    if name == "SnapKVCacheManager":
        from .methods.snapkv import SnapKVCacheManager

        return SnapKVCacheManager
    if name == "H2OCacheManager":
        from .methods.h2o import H2OCacheManager

        return H2OCacheManager
    if name == "RKVCacheManager":
        from .methods.rkv import RKVCacheManager

        return RKVCacheManager
    if name == "SkipKVCacheManager":
        from .methods.skipkv import SkipKVCacheManager

        return SkipKVCacheManager
    if name == "QuestCacheManager":
        from .methods.quest import QuestCacheManager

        return QuestCacheManager
    if name == "OmniKVCacheManager":
        from .methods.omnikv.manager import OmniKVCacheManager

        return OmniKVCacheManager
    if name == "DeltaKVCacheManager":
        from .methods.deltakv_runtime import DeltaKVCacheManager

        return DeltaKVCacheManager
    if name == "DeltaKVCacheTritonManagerV4":
        from .methods.deltakv_base import DeltaKVCacheTritonManagerV4

        return DeltaKVCacheTritonManagerV4
    if name == "DeltaKVLessMemoryCacheManager":
        from .methods.deltakv_less_memory import DeltaKVLessMemoryCacheManager

        return DeltaKVLessMemoryCacheManager
    if name == "DeltaKVLessMemoryCudaGraphCacheManager":
        from .methods.deltakv_less_memory_cuda_graph import DeltaKVLessMemoryCudaGraphCacheManager

        return DeltaKVLessMemoryCudaGraphCacheManager

    raise AttributeError(name)
