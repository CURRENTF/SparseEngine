import os
import unittest
from unittest.mock import patch

import torch

from sparseengine.engine.cache_manager import (
    AttentionViewMeta,
    DecodeComputeView,
    ExplicitKVPayload,
    MlaLatentPayload,
    PrefillComputeView,
)
from sparseengine.layers.attention_backend import TritonAttentionBackend


class FakeAttentionBackendTest(unittest.TestCase):
    _ENV_KEYS = (
        "SPARSEENGINE_FAKE_ATTENTION",
        "SPARSEENGINE_FAKE_PREFILL_ATTENTION",
        "SPARSEENGINE_FAKE_DECODE_ATTENTION",
        "SPARSEENGINE_FAKE_ATTENTION_MODE",
        "SPARSEENGINE_ALLOW_FAKE_ATTENTION",
        "SENGINE_DEBUG_DECODE_BOUNDS",
    )

    def setUp(self):
        self._old_env = {key: os.environ.get(key) for key in self._ENV_KEYS}

    def tearDown(self):
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _make_prefill_view(self, *, attn_score=None):
        return PrefillComputeView(
            meta=AttentionViewMeta(
                active_slots=torch.tensor([[0, 1, 2]], dtype=torch.int32),
                req_indices=torch.tensor([0], dtype=torch.int32),
                context_lens=torch.tensor([3], dtype=torch.int32),
                attn_score=attn_score,
                max_context_len=3,
            ),
            payload=ExplicitKVPayload(
                k_cache=torch.ones(8, 2, 4),
                v_cache=torch.ones(8, 2, 4),
            ),
        )

    def _make_decode_view(self, *, attn_score=None):
        return DecodeComputeView(
            meta=AttentionViewMeta(
                active_slots=torch.tensor([[0, 1, 2]], dtype=torch.int32),
                req_indices=torch.tensor([0], dtype=torch.int32),
                context_lens=torch.tensor([3], dtype=torch.int32),
                attn_score=attn_score,
                max_context_len=3,
            ),
            payload=ExplicitKVPayload(
                k_cache=torch.ones(8, 2, 4),
                v_cache=torch.ones(8, 2, 4),
            ),
        )

    def test_fake_prefill_returns_zeros_and_skips_kernel(self):
        os.environ["SPARSEENGINE_FAKE_ATTENTION"] = "1"
        os.environ["SPARSEENGINE_ALLOW_FAKE_ATTENTION"] = "1"
        q = torch.ones(3, 2, 4)
        attn_score = torch.full((1, 2, 3), 9.0)
        view = self._make_prefill_view(attn_score=attn_score)

        with patch(
            "sparseengine.layers.attention_backend.context_attention_fwd",
            side_effect=AssertionError("real prefill kernel called"),
        ):
            out = TritonAttentionBackend().run_prefill(
                q,
                view,
                b_start_loc=torch.tensor([0], dtype=torch.int32),
                chunk_lens=torch.tensor([3], dtype=torch.int32),
                max_input_len=3,
            )

        self.assertTrue(torch.equal(out, torch.zeros_like(q)))
        self.assertTrue(torch.equal(attn_score, torch.zeros_like(attn_score)))

    def test_fake_decode_copy_mode_skips_kernels(self):
        os.environ["SPARSEENGINE_FAKE_ATTENTION"] = "1"
        os.environ["SPARSEENGINE_ALLOW_FAKE_ATTENTION"] = "1"
        os.environ["SPARSEENGINE_FAKE_ATTENTION_MODE"] = "copy"
        q = torch.arange(8, dtype=torch.float32).view(1, 2, 4)
        attn_score = torch.full((1, 2, 3), -1e20)
        view = self._make_decode_view(attn_score=attn_score)

        with (
            patch(
                "sparseengine.layers.attention_backend.gqa_flash_decode_stage1",
                side_effect=AssertionError("real decode stage1 called"),
            ),
            patch(
                "sparseengine.layers.attention_backend.flash_decode_stage2",
                side_effect=AssertionError("real decode stage2 called"),
            ),
        ):
            out = TritonAttentionBackend().run_decode(
                q,
                view,
                mid_o=torch.empty(1, 2, 1, 4),
                mid_o_logexpsum=torch.empty(1, 2, 1),
                max_len_in_batch=3,
                block_seq=256,
                num_heads=2,
                num_kv_heads=1,
            )

        self.assertIsNot(out, q)
        self.assertTrue(torch.equal(out, q))
        self.assertTrue(torch.equal(attn_score, torch.zeros_like(attn_score)))

    def test_debug_decode_bounds_checks_flash_attn_contiguous(self):
        os.environ["SENGINE_DEBUG_DECODE_BOUNDS"] = "1"
        q = torch.zeros(1, 2, 4)
        view = DecodeComputeView(
            meta=AttentionViewMeta(
                active_slots=torch.tensor([[0, 1, 99]], dtype=torch.int32),
                req_indices=torch.tensor([0], dtype=torch.int32),
                context_lens=torch.tensor([3], dtype=torch.int32),
                max_context_len=3,
            ),
            payload=ExplicitKVPayload(
                k_cache=torch.ones(3, 2, 4),
                v_cache=torch.ones(3, 2, 4),
                backend="flash_attn_contiguous",
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "decode physical slot out of bounds"):
            TritonAttentionBackend().run_decode(
                q,
                view,
                mid_o=torch.empty(1, 2, 1, 4),
                mid_o_logexpsum=torch.empty(1, 2, 1),
                max_len_in_batch=3,
                block_seq=256,
                num_heads=2,
                num_kv_heads=1,
            )

    def test_fake_attention_requires_explicit_allow(self):
        os.environ["SPARSEENGINE_FAKE_ATTENTION"] = "1"
        os.environ.pop("SPARSEENGINE_ALLOW_FAKE_ATTENTION", None)
        q = torch.ones(3, 2, 4)
        view = self._make_prefill_view()

        with self.assertRaisesRegex(RuntimeError, "SPARSEENGINE_ALLOW_FAKE_ATTENTION"):
            TritonAttentionBackend().run_prefill(
                q,
                view,
                b_start_loc=torch.tensor([0], dtype=torch.int32),
                chunk_lens=torch.tensor([3], dtype=torch.int32),
                max_input_len=3,
            )

    def test_fake_prefill_only_keeps_decode_kernels(self):
        os.environ["SPARSEENGINE_FAKE_PREFILL_ATTENTION"] = "1"
        os.environ.pop("SPARSEENGINE_FAKE_ATTENTION", None)
        os.environ.pop("SPARSEENGINE_FAKE_DECODE_ATTENTION", None)
        q = torch.arange(8, dtype=torch.float32).view(1, 2, 4)
        attn_score = torch.full((1, 2, 3), -1e20)
        view = self._make_decode_view(attn_score=attn_score)
        calls = []

        def stage1(*args, **kwargs):
            calls.append("stage1")

        def stage2(mid_o, mid_o_logexpsum, context_lens, out, block_seq):
            calls.append("stage2")
            out.zero_()

        with (
            patch("sparseengine.layers.attention_backend.gqa_flash_decode_stage1_with_score", side_effect=stage1),
            patch("sparseengine.layers.attention_backend.flash_decode_stage2", side_effect=stage2),
        ):
            out = TritonAttentionBackend().run_decode(
                q,
                view,
                mid_o=torch.empty(1, 2, 1, 4),
                mid_o_logexpsum=torch.empty(1, 2, 1),
                max_len_in_batch=3,
                block_seq=256,
                num_heads=2,
                num_kv_heads=1,
            )

        self.assertEqual(calls, ["stage1", "stage2"])
        self.assertTrue(torch.equal(out, torch.zeros_like(q)))
        self.assertTrue(torch.equal(attn_score, torch.full_like(attn_score, -1e20)))

    def test_prefill_rejects_mla_payload_before_kernel(self):
        explicit_view = self._make_prefill_view()
        view = PrefillComputeView(
            meta=explicit_view.meta,
            payload=MlaLatentPayload(
                latent_cache=torch.empty(8, 1, 512),
                rope_cache=torch.empty(8, 1, 64),
            ),
        )

        with self.assertRaisesRegex(TypeError, "requires ExplicitKVPayload"):
            TritonAttentionBackend().run_prefill(
                torch.empty(3, 2, 4),
                view,
                b_start_loc=torch.tensor([0], dtype=torch.int32),
                chunk_lens=torch.tensor([3], dtype=torch.int32),
                max_input_len=3,
            )

    def test_decode_rejects_mla_payload_before_kernel(self):
        explicit_view = self._make_decode_view()
        view = DecodeComputeView(
            meta=explicit_view.meta,
            payload=MlaLatentPayload(
                latent_cache=torch.empty(8, 1, 512),
                rope_cache=torch.empty(8, 1, 64),
            ),
        )

        with self.assertRaisesRegex(TypeError, "requires ExplicitKVPayload"):
            TritonAttentionBackend().run_decode(
                torch.empty(1, 2, 4),
                view,
                mid_o=torch.empty(1, 2, 1, 4),
                mid_o_logexpsum=torch.empty(1, 2, 1),
                max_len_in_batch=3,
                block_seq=256,
                num_heads=2,
                num_kv_heads=1,
            )


if __name__ == "__main__":
    unittest.main()
