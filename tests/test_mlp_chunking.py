import unittest
from unittest.mock import patch

import pytest
import torch

from sparseengine.models.llama import LlamaMLP
from sparseengine.models.qwen2 import Qwen2MLP
from sparseengine.models.qwen3 import Qwen3MLP
from sparseengine.distributed import ParallelContext, ParallelGroup


def _single_process_parallel_context() -> ParallelContext:
    group = ParallelGroup(process_group=None, ranks=(0,), rank=0, size=1)
    return ParallelContext(world=group, moe_tp=group, attn_tp=group, moe_ep=group, attn_dp=group)


class MLPChunkingTest(unittest.TestCase):
    def _assert_chunked_matches_full(self, cls):
        with patch(
            "sparseengine.layers.linear.get_parallel_context",
            return_value=_single_process_parallel_context(),
        ):
            torch.manual_seed(0)
            full = cls(8, 16, "silu", mlp_chunk_size=1024)
            chunked = cls(8, 16, "silu", mlp_chunk_size=5)
            for param in full.parameters():
                param.data.normal_(mean=0.0, std=0.02)
            chunked.load_state_dict(full.state_dict())

            x = torch.randn(17, 8)
            with torch.inference_mode():
                expected = full(x)
                actual = chunked(x)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_qwen2_mlp_chunking_matches_full_forward(self):
        self._assert_chunked_matches_full(Qwen2MLP)

    def test_qwen3_mlp_chunking_matches_full_forward(self):
        self._assert_chunked_matches_full(Qwen3MLP)



@pytest.mark.parametrize("cls", [Qwen2MLP, Qwen3MLP, LlamaMLP])
@pytest.mark.parametrize("tokens", [1, 32, 33, 97])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_chunked_mlp_matches_independent_reference(cls, tokens, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    extra = {"mlp_bias": True} if cls is LlamaMLP else {}
    with patch("sparseengine.layers.linear.get_parallel_context",
               return_value=_single_process_parallel_context()):
        layer = cls(128, 256, "silu", mlp_chunk_size=32, **extra).to(device, dtype)
    with torch.no_grad():
        for param in layer.parameters():
            param.normal_(std=0.02)
    x = torch.randn(tokens, 128, device=device, dtype=dtype)
    with torch.inference_mode():
        gate, up = torch.nn.functional.linear(
            x.float(), layer.gate_up_proj.weight.float(),
            None if layer.gate_up_proj.bias is None else layer.gate_up_proj.bias.float(),
        ).chunk(2, dim=-1)
        expected = torch.nn.functional.linear(
            torch.nn.functional.silu(gate) * up, layer.down_proj.weight.float(),
            None if layer.down_proj.bias is None else layer.down_proj.bias.float(),
        )
        actual = layer(x)
    torch.testing.assert_close(actual.float(), expected, atol=.001 if device == "cuda" else 1e-6,
                               rtol=.03 if device == "cuda" else 1e-5)


if __name__ == "__main__":
    unittest.main()
