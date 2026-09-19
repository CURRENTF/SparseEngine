import unittest
from unittest.mock import patch

import torch

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


if __name__ == "__main__":
    unittest.main()
