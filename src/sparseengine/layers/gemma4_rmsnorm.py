from __future__ import annotations

import torch
from torch import nn

from sparseengine.operators.gemma4 import Gemma4OperatorProvider


class Gemma4RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        with_scale: bool = True,
        provider: Gemma4OperatorProvider,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self._ops = provider
        if with_scale:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._ops.rmsnorm(x, self.weight, self.eps)


__all__ = ["Gemma4RMSNorm"]
