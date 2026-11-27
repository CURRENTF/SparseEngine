from __future__ import annotations

# Public DeltaKV entrypoint: the large historical implementation moved to
# deltakv_base.py and is reused only as a base utility layer by the slim runtime.
from .deltakv_runtime import DeltaKVCacheManager

__all__ = ["DeltaKVCacheManager"]
