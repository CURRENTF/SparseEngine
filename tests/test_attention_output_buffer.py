from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sparseengine.layers.linear import RowParallelLinear
from sparseengine.models.llama import LlamaAttention
from sparseengine.models.qwen2 import Qwen2Attention
from sparseengine.models.qwen3 import Qwen3Attention
from sparseengine.models.qwen3_5 import Qwen35FullAttention


@pytest.mark.parametrize("attention", [LlamaAttention, Qwen2Attention, Qwen3Attention, Qwen35FullAttention])
@pytest.mark.parametrize("tokens", [3, 19])
def test_attention_projection_preserves_partial_chunk_and_buffer(attention, tokens):
    # Exercise each real model call site against an independent full projection.
    projection = object.__new__(RowParallelLinear)
    nn.Module.__init__(projection)
    projection.weight = nn.Parameter(torch.randn(13, 8), requires_grad=False)
    projection.bias = nn.Parameter(torch.randn(13), requires_grad=False)
    projection.tp_rank = 0
    projection.quantized = False
    projection.reduce_results = False
    owner = SimpleNamespace(o_proj=projection, proj_chunk_size=7)
    x = torch.randn(tokens, 8)
    buffer = torch.full((tokens, 13), float('nan'))
    with torch.inference_mode():
        result = attention._o_proj_chunked(owner, x, buffer)
        expected = F.linear(x, projection.weight, projection.bias)
    torch.testing.assert_close(result, expected)
    if tokens > owner.proj_chunk_size:
        assert result.data_ptr() == buffer.data_ptr()
