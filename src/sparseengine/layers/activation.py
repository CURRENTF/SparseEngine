import torch
from torch import nn

from sparseengine.operators.activation import (
    SiluAndMulProvider,
    TorchSiluAndMulProvider,
)


class SiluAndMul(nn.Module):

    def __init__(self, provider: SiluAndMulProvider | None = None):
        super().__init__()
        self.provider = provider if provider is not None else TorchSiluAndMulProvider()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.provider(x)
