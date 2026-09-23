from __future__ import annotations

from sparseengine.config import Config
from sparseengine.engine.cache_manager import CacheManager
from sparseengine.method_registry import (
    resolve_cache_sparse_method,
    resolve_prefill_sparse_method,
)

from .base import SparseMethodRuntime
from .dynamic import DeltaKVRuntime, OmniKVRuntime
from .h2o import H2ORuntime
from .kvzip import KVzipRuntime
from .joint import SkipKVRuntime
from .rkv import RKVRuntime
from .passthrough import PassThroughRuntime
from .quest import QuestRuntime
from .snapkv import PyramidKVRuntime, SnapKVRuntime
from .streamingllm import StreamingLLMRuntime


RUNTIME_BINDINGS: dict[str, type[SparseMethodRuntime]] = {
    "palu": PassThroughRuntime,
    "kivi": PassThroughRuntime,
    "turboquant": PassThroughRuntime,
    "fp8_kv": PassThroughRuntime,
    "": PassThroughRuntime,
    "streamingllm": StreamingLLMRuntime,
    "snapkv": SnapKVRuntime,
    "kvzip": KVzipRuntime,
    "h2o": H2ORuntime,
    "pyramidkv": PyramidKVRuntime,
    "omnikv": OmniKVRuntime,
    "quest": QuestRuntime,
    "rkv": RKVRuntime,
    "skipkv": SkipKVRuntime,
    "deltakv": DeltaKVRuntime,
}


def create_sparse_method_runtime(
    config: Config,
    cache_manager: CacheManager,
) -> SparseMethodRuntime:
    method = resolve_cache_sparse_method(
        config.sparse_method,
        prefill_sparse_method=getattr(config, "prefill_sparse_method", None),
    )
    runtime_cls = RUNTIME_BINDINGS.get(method)
    if runtime_cls is None:
        raise ValueError(f"Unsupported sparse_method={method!r}.")
    runtime = runtime_cls(config, cache_manager)
    if resolve_prefill_sparse_method(
        getattr(config, "prefill_sparse_method", None),
        sparse_method=config.sparse_method,
    ) == "omnikv_prefill":
        if method not in {"", "omnikv"}:
            raise ValueError("OmniKV prefill requires vanilla or OmniKV decode lifecycle.")
        from .omnikv_prefill import OmniKVPrefillRuntime
        from .phases import PrefillOverrideRuntime

        return PrefillOverrideRuntime(OmniKVPrefillRuntime(config, cache_manager), runtime)
    return runtime
