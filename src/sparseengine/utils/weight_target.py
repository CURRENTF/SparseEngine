from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class WeightTarget:
    """Resolved module and logical shard for one checkpoint tensor."""

    module: nn.Module
    shard_id: object = None
