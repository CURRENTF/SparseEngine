import unittest
from unittest.mock import patch

import torch

from sparseengine.engine.cache_manager import (
    AttentionViewMeta,
    DecodeComputeView,
    ExplicitKVPayload,
)
from sparseengine.layers.attention_backend import TritonAttentionBackend
from sparseengine.kernels.triton.flash_decoding_stage2 import flash_decode_stage2
from sparseengine.kernels.triton.gqa_flash_decoding_stage1 import (
    flash_decode_stage1,
    flash_decode_stage1_with_score,
)


class Qwen35Hd256DecodeRoutingTest(unittest.TestCase):
    def _make_view(self, *, head_dim: int, attn_score=None):
        active_slots = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        req_indices = torch.tensor([0], dtype=torch.int32)
        context_lens = torch.tensor([3], dtype=torch.int32)
        k_cache = torch.zeros(4, 4, head_dim, dtype=torch.float32)
        v_cache = torch.zeros_like(k_cache)
        return DecodeComputeView(
            meta=AttentionViewMeta(
                active_slots=active_slots,
                req_indices=req_indices,
                context_lens=context_lens,
                attn_score=attn_score,
                max_context_len=3,
            ),
            payload=ExplicitKVPayload(k_cache=k_cache, v_cache=v_cache),
        )

    def test_head_dim_256_uses_grouped_gqa_stage1_and_unified_stage2(self):
        q = torch.zeros(1, 16, 256)
        view = self._make_view(head_dim=256)
        mid_o = torch.empty(1, 16, 1, 256)
        mid_lse = torch.empty(1, 16, 1)
        calls = {"stage1": 0, "stage2": 0}

        def stage1_grouped(*args, **kwargs):
            calls["stage1"] += 1

        def stage2(mid_o, mid_o_logsumexp, context_lens, o, block_seq):
            calls["stage2"] += 1
            o.fill_(7.0)

        with (
            patch("sparseengine.layers.attention_backend.gqa_flash_decode_stage1", side_effect=stage1_grouped),
            patch("sparseengine.layers.attention_backend.flash_decode_stage2", side_effect=stage2),
        ):
            out = TritonAttentionBackend().run_decode(
                q,
                view,
                mid_o=mid_o,
                mid_o_logexpsum=mid_lse,
                max_len_in_batch=3,
                block_seq=256,
                num_heads=16,
                num_kv_heads=4,
            )

        self.assertEqual(calls, {"stage1": 1, "stage2": 1})
        self.assertTrue(torch.equal(out, torch.full_like(out, 7.0)))

    def test_head_dim_256_with_score_uses_grouped_score_stage1(self):
        attn_score = torch.zeros(1, 16, 3)
        q = torch.zeros(1, 16, 256)
        view = self._make_view(head_dim=256, attn_score=attn_score)
        mid_o = torch.empty(1, 16, 1, 256)
        mid_lse = torch.empty(1, 16, 1)
        calls = {"stage1_score": 0, "stage2": 0}

        def stage1_grouped_with_score(*args, **kwargs):
            calls["stage1_score"] += 1

        def stage2(mid_o, mid_o_logsumexp, context_lens, o, block_seq):
            calls["stage2"] += 1
            o.fill_(11.0)

        with (
            patch(
                "sparseengine.layers.attention_backend.gqa_flash_decode_stage1_with_score",
                side_effect=stage1_grouped_with_score,
            ),
            patch("sparseengine.layers.attention_backend.flash_decode_stage2", side_effect=stage2),
        ):
            out = TritonAttentionBackend().run_decode(
                q,
                view,
                mid_o=mid_o,
                mid_o_logexpsum=mid_lse,
                max_len_in_batch=3,
                block_seq=256,
                num_heads=16,
                num_kv_heads=4,
            )

        self.assertEqual(calls, {"stage1_score": 1, "stage2": 1})
        self.assertTrue(torch.equal(out, torch.full_like(out, 11.0)))

    def test_head_dim_128_keeps_existing_gqa_decode_kernels(self):
        q = torch.zeros(1, 16, 128)
        view = self._make_view(head_dim=128)
        mid_o = torch.empty(1, 16, 1, 128)
        mid_lse = torch.empty(1, 16, 1)
        calls = {"stage1": 0, "stage2": 0}

        def stage1(*args, **kwargs):
            calls["stage1"] += 1

        def stage2(mid_o, mid_o_logsumexp, context_lens, o, block_seq):
            calls["stage2"] += 1
            o.fill_(3.0)

        with (
            patch("sparseengine.layers.attention_backend.gqa_flash_decode_stage1", side_effect=stage1),
            patch("sparseengine.layers.attention_backend.flash_decode_stage2", side_effect=stage2),
        ):
            out = TritonAttentionBackend().run_decode(
                q,
                view,
                mid_o=mid_o,
                mid_o_logexpsum=mid_lse,
                max_len_in_batch=3,
                block_seq=256,
                num_heads=16,
                num_kv_heads=4,
            )

        self.assertEqual(calls, {"stage1": 1, "stage2": 1})
        self.assertTrue(torch.equal(out, torch.full_like(out, 3.0)))

    def test_head_dim_128_with_score_keeps_existing_gqa_decode_kernels(self):
        attn_score = torch.zeros(1, 16, 3)
        q = torch.zeros(1, 16, 128)
        view = self._make_view(head_dim=128, attn_score=attn_score)
        mid_o = torch.empty(1, 16, 1, 128)
        mid_lse = torch.empty(1, 16, 1)
        calls = {"stage1_score": 0, "stage2": 0}

        def stage1_with_score(*args, **kwargs):
            calls["stage1_score"] += 1

        def stage2(mid_o, mid_o_logsumexp, context_lens, o, block_seq):
            calls["stage2"] += 1
            o.fill_(5.0)

        with (
            patch("sparseengine.layers.attention_backend.gqa_flash_decode_stage1_with_score", side_effect=stage1_with_score),
            patch("sparseengine.layers.attention_backend.flash_decode_stage2", side_effect=stage2),
        ):
            out = TritonAttentionBackend().run_decode(
                q,
                view,
                mid_o=mid_o,
                mid_o_logexpsum=mid_lse,
                max_len_in_batch=3,
                block_seq=256,
                num_heads=16,
                num_kv_heads=4,
            )

        self.assertEqual(calls, {"stage1_score": 1, "stage2": 1})
        self.assertTrue(torch.equal(out, torch.full_like(out, 5.0)))

    def test_grouped_gqa_wrappers_accept_head_dim_256(self):
        q = torch.zeros(1, 24, 256)
        k = torch.zeros(3, 4, 256)
        v = torch.zeros_like(k)
        req_to_tokens = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        req_indices = torch.tensor([0], dtype=torch.int32)
        context_lens = torch.tensor([3], dtype=torch.int32)
        mid_o = torch.empty(1, 24, 1, 256)
        mid_lse = torch.empty(1, 24, 1)
        attn_score = torch.empty(1, 24, 3)

        with patch(
            "sparseengine.kernels.triton.gqa_flash_decoding_stage1._fwd_kernel_flash_decode_stage1"
        ) as kernel:
            flash_decode_stage1(
                q,
                k,
                v,
                req_to_tokens,
                req_indices,
                context_lens,
                3,
                mid_o,
                mid_lse,
                256,
            )
            kernel.__getitem__.return_value.assert_called_once()

        with patch(
            "sparseengine.kernels.triton.gqa_flash_decoding_stage1._fwd_kernel_flash_decode_stage1_with_score"
        ) as kernel:
            flash_decode_stage1_with_score(
                q,
                k,
                v,
                req_to_tokens,
                req_indices,
                context_lens,
                3,
                mid_o,
                mid_lse,
                attn_score,
                256,
            )
            kernel.__getitem__.return_value.assert_called_once()

    def test_grouped_gqa_wrapper_rejects_unsupported_strides(self):
        valid = [
            torch.zeros(1, 24, 256),
            torch.zeros(3, 4, 256),
            torch.zeros(3, 4, 256),
            torch.tensor([[0, 1, 2]], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([3], dtype=torch.int32),
            3,
            torch.empty(1, 24, 1, 256),
            torch.empty(1, 24, 1),
            256,
        ]
        cases = (
            ("q", 0, torch.empty(1, 24, 512)[..., ::2], "q head_dim must be contiguous"),
            ("v", 2, torch.empty(3, 8, 256)[:, ::2, :], "k and v must have identical layouts"),
            (
                "req_to_tokens",
                3,
                torch.arange(6, dtype=torch.int32).view(1, 6)[:, ::2],
                "req_to_tokens sequence dimension must be contiguous",
            ),
            ("b_req_idx", 4, torch.tensor([0, 1], dtype=torch.int32)[::2], "b_req_idx must be contiguous"),
            ("b_seqlen", 5, torch.tensor([3, 3], dtype=torch.int32)[::2], "b_seqlen must be contiguous"),
            ("mid_out", 7, torch.empty(1, 24, 1, 512)[..., ::2], "mid_out head_dim must be contiguous"),
            (
                "mid_lse",
                8,
                torch.empty(1, 24, 2)[..., ::2],
                "mid_out_logsumexp block dimension must be contiguous",
            ),
        )

        for name, index, tensor, message in cases:
            with self.subTest(name=name):
                args = list(valid)
                args[index] = tensor
                with self.assertRaisesRegex(AssertionError, message):
                    flash_decode_stage1(*args)

    def test_unified_stage2_forwards_noncontiguous_strides_and_selects_warps(self):
        for head_dim, expected_warps in ((128, 4), (256, 8)):
            with self.subTest(head_dim=head_dim):
                mid_out = torch.empty(1, 24, 2, head_dim * 2)[..., ::2]
                mid_lse = torch.empty(1, 24, 4)[..., ::2]
                b_seqlen = torch.tensor([257], dtype=torch.int32)
                output = torch.empty(1, 24, head_dim * 2)[..., ::2]
                with patch(
                    "sparseengine.kernels.triton.flash_decoding_stage2._fwd_kernel_flash_decode_stage2"
                ) as kernel:
                    flash_decode_stage2(mid_out, mid_lse, b_seqlen, output, 256)

                launch = kernel.__getitem__.return_value
                launch.assert_called_once()
                args = launch.call_args.args
                self.assertEqual(args[4:8], mid_out.stride())
                self.assertEqual(args[8:11], mid_lse.stride())
                self.assertEqual(args[11:14], output.stride())
                self.assertEqual(launch.call_args.kwargs["BLOCK_DMODEL"], head_dim)
                self.assertEqual(launch.call_args.kwargs["num_warps"], expected_warps)

    def test_unified_stage2_rejects_noncontiguous_sequence_lengths(self):
        with self.assertRaisesRegex(AssertionError, "B_Seqlen"):
            flash_decode_stage2(
                torch.empty(1, 24, 1, 256),
                torch.empty(1, 24, 1),
                torch.tensor([1, 1], dtype=torch.int32)[::2],
                torch.empty(1, 24, 256),
                256,
            )


if __name__ == "__main__":
    unittest.main()
