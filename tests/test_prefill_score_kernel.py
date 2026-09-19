import unittest
from unittest.mock import patch

import torch



def _prefill_score_baseline(
    q,
    k_cache,
    req_to_tokens,
    b_req_idx,
    b_start_loc,
    context_lens,
    prompt_cache_lens,
    score_starts,
    score_ends,
    candidate_start,
    recent_keep_tokens,
):
    batch = int(context_lens.numel())
    num_heads = int(q.shape[1])
    num_kv_heads = int(k_cache.shape[1])
    kv_group = num_heads // num_kv_heads
    max_len = int(req_to_tokens.shape[1])
    out = torch.zeros((batch, max_len), dtype=torch.float32, device=q.device)
    for b in range(batch):
        q_start = int(b_start_loc[b].item())
        prompt_cache_len = int(prompt_cache_lens[b].item())
        context_len = int(context_lens[b].item())
        score_start = int(score_starts[b].item())
        score_end = int(score_ends[b].item())
        req_row = int(b_req_idx[b].item())
        cand_start = int(candidate_start)
        cand_end = max(cand_start, context_len - int(recent_keep_tokens))
        q_scores = []
        for h in range(num_heads):
            kv_h = h // kv_group
            head_scores = torch.zeros((max_len,), dtype=torch.float32, device=q.device)
            for q_pos in range(score_start, score_end):
                if q_pos < prompt_cache_len or q_pos >= context_len:
                    continue
                q_vec = q[q_start + q_pos - prompt_cache_len, h].float()
                logits = []
                positions = []
                for k_pos in range(cand_start, cand_end):
                    if q_pos < k_pos:
                        continue
                    slot = int(req_to_tokens[req_row, k_pos].item())
                    logits.append(torch.dot(q_vec, k_cache[slot, kv_h].float()) * (q.shape[-1] ** -0.5))
                    positions.append(k_pos)
                if not logits:
                    continue
                probs = torch.softmax(torch.stack(logits), dim=-1)
                for pos, prob in zip(positions, probs):
                    head_scores[pos] += prob / max(1, score_end - score_start)
            q_scores.append(head_scores)
        if q_scores:
            out[b] = torch.stack(q_scores).max(dim=0).values
    return out


def _raw_qk_score_baseline(
    q,
    k_cache,
    req_to_tokens,
    b_req_idx,
    b_start_loc,
    context_lens,
    prompt_cache_lens,
    score_starts,
    score_ends,
    candidate_start,
    recent_keep_tokens,
):
    batch = int(context_lens.numel())
    num_heads = int(q.shape[1])
    num_kv_heads = int(k_cache.shape[1])
    kv_group = num_heads // num_kv_heads
    score_len = int(req_to_tokens.shape[1])
    out = torch.full(
        (batch, score_len),
        -torch.inf,
        dtype=torch.float32,
        device=q.device,
    )
    for b in range(batch):
        q_start = int(b_start_loc[b].item())
        prompt_cache_len = int(prompt_cache_lens[b].item())
        context_len = int(context_lens[b].item())
        req_row = int(b_req_idx[b].item())
        candidate_end = context_len - int(recent_keep_tokens)
        for q_pos in range(int(score_starts[b]), int(score_ends[b])):
            q_rel = q_pos - prompt_cache_len
            if q_rel < 0 or q_rel >= context_len - prompt_cache_len:
                continue
            for q_head in range(num_heads):
                kv_head = q_head // kv_group
                q_vec = q[q_start + q_rel, q_head].float()
                for kv_pos in range(int(candidate_start), candidate_end):
                    if kv_pos > q_pos:
                        continue
                    slot = int(req_to_tokens[req_row, kv_pos].item())
                    score = torch.dot(q_vec, k_cache[slot, kv_head].float())
                    out[b, kv_pos] = torch.maximum(out[b, kv_pos], score)
    return out


    def test_tilelang_prefill_wrapper_rejects_per_head_score_tensor(self):
        try:
            from sparseengine.kernels.tilelang.gqa.prefill import (
                gqa_paged_prefill_attention_tilelang,
            )
        except ImportError as exc:
            self.skipTest(str(exc))

        with self.assertRaisesRegex(ValueError, "only supports reduced 2D scores"):
            gqa_paged_prefill_attention_tilelang(
                torch.empty((1, 16, 128), dtype=torch.bfloat16),
                torch.empty((1, 2, 128), dtype=torch.bfloat16),
                torch.empty((1, 2, 128), dtype=torch.bfloat16),
                torch.tensor([[0]], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0, 1], dtype=torch.int32),
                torch.empty((1, 16, 128), dtype=torch.bfloat16),
                attn_score=torch.empty((1, 16, 1), dtype=torch.float32),
            )

    def test_tilelang_prefill_launch_uses_query_length_not_page_table_capacity(self):
        try:
            from sparseengine.kernels.tilelang.gqa import prefill as tilelang_prefill
        except ImportError as exc:
            self.skipTest(str(exc))

        captured = {}

        def fake_kernel(*args):
            captured["max_query_len"] = args[-1]

        with patch.object(tilelang_prefill, "_get_fused_prefill_kernel", return_value=fake_kernel):
            tilelang_prefill.gqa_paged_prefill_attention_tilelang(
                torch.empty((4, 16, 128), dtype=torch.bfloat16),
                torch.empty((8, 2, 128), dtype=torch.bfloat16),
                torch.empty((8, 2, 128), dtype=torch.bfloat16),
                torch.zeros((1, 65536), dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([4], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0, 4], dtype=torch.int32),
                torch.empty((4, 16, 128), dtype=torch.bfloat16),
            )

        self.assertEqual(captured["max_query_len"], 4)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for prefill score Triton tests.")
class PrefillScoreKernelTest(unittest.TestCase):
    def test_shared_scorer_matches_torch_for_mha_reconstruction(self):
        """MLA prefill reconstructs one explicit K head per query head."""
        from sparseengine.kernels.triton.prefill_score import prefill_score_fwd

        torch.manual_seed(19)
        device = "cuda"
        prompt_len = 8
        prompt_cache_len = 5
        score_start = 5
        score_end = 8
        num_heads = 4
        head_dim = 16
        q = torch.randn(
            (score_end - prompt_cache_len, num_heads, head_dim),
            dtype=torch.float32,
            device=device,
        )
        k_cache = torch.randn(
            (prompt_len, num_heads, head_dim),
            dtype=torch.float32,
            device=device,
        )
        req_to_tokens = torch.arange(
            prompt_len, dtype=torch.int32, device=device
        ).unsqueeze(0)
        req_indices = torch.tensor([0], dtype=torch.int32, device=device)
        b_start_loc = torch.tensor([0], dtype=torch.int32, device=device)
        context_lens = torch.tensor(
            [prompt_len], dtype=torch.int32, device=device
        )
        prompt_cache_lens = torch.tensor(
            [prompt_cache_len], dtype=torch.int32, device=device
        )
        score_starts = torch.tensor(
            [score_start], dtype=torch.int32, device=device
        )
        score_ends = torch.tensor([score_end], dtype=torch.int32, device=device)

        for mode, baseline in (
            ("probability", _prefill_score_baseline),
            ("logits", _raw_qk_score_baseline),
        ):
            with self.subTest(mode=mode):
                actual = torch.empty(
                    (1, prompt_len), dtype=torch.float32, device=device
                )
                prefill_score_fwd(
                    q,
                    k_cache,
                    actual,
                    req_indices,
                    b_start_loc,
                    context_lens,
                    prompt_cache_lens,
                    score_end - score_start,
                    req_to_tokens,
                    score_starts,
                    score_ends,
                    candidate_start=1,
                    recent_keep_tokens=1,
                    score_mode=mode,
                )
                expected = baseline(
                    q,
                    k_cache,
                    req_to_tokens,
                    req_indices,
                    b_start_loc,
                    context_lens,
                    prompt_cache_lens,
                    score_starts,
                    score_ends,
                    candidate_start=1,
                    recent_keep_tokens=1,
                )
                torch.testing.assert_close(
                    actual, expected, rtol=2e-2, atol=2e-2
                )

    def test_logits_score_matches_torch(self):
        from sparseengine.kernels.triton.prefill_score import prefill_score_fwd

        torch.manual_seed(23)
        device = "cuda"
        context_lens = torch.tensor([9, 7], dtype=torch.int32, device=device)
        prompt_cache_lens = torch.tensor([5, 4], dtype=torch.int32, device=device)
        b_start_loc = torch.tensor([0, 4], dtype=torch.int32, device=device)
        active_slots = torch.full((2, 9), -1, dtype=torch.int32, device=device)
        active_slots[0, :7] = torch.tensor(
            [13, 4, 15, 6, 17, 8, 19], dtype=torch.int32, device=device
        )
        active_slots[1, :9] = torch.tensor(
            [0, 11, 2, 9, 5, 14, 3, 16, 7], dtype=torch.int32, device=device
        )
        req_indices = torch.tensor([1, 0], dtype=torch.int32, device=device)
        score_starts = torch.tensor([5, 5], dtype=torch.int32, device=device)
        score_ends = torch.tensor([9, 7], dtype=torch.int32, device=device)
        q = (
            torch.randn((7, 16, 128), dtype=torch.bfloat16, device=device)
            * 0.1
        )
        k_cache = (
            torch.randn((20, 2, 128), dtype=torch.bfloat16, device=device)
            * 0.1
        )
        actual = torch.empty((2, 9), dtype=torch.float32, device=device)

        prefill_score_fwd(
            q,
            k_cache,
            actual,
            req_indices,
            b_start_loc,
            context_lens,
            prompt_cache_lens,
            int((score_ends - score_starts).max().item()),
            active_slots,
            score_starts,
            score_ends,
            candidate_start=1,
            recent_keep_tokens=1,
            score_mode="logits",
        )
        torch.cuda.synchronize()

        expected = _raw_qk_score_baseline(
            q,
            k_cache,
            active_slots,
            req_indices,
            b_start_loc,
            context_lens,
            prompt_cache_lens,
            score_starts,
            score_ends,
            candidate_start=1,
            recent_keep_tokens=1,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_logit_score_supports_query_windows_larger_than_128(self):
        from sparseengine.kernels.triton.prefill_score import prefill_score_fwd

        torch.manual_seed(29)
        device = "cuda"
        prompt_len = 192
        window = 160
        num_heads = 4
        num_kv_heads = 2
        head_dim = 16
        q = torch.randn(
            (window, num_heads, head_dim), dtype=torch.float32, device=device
        )
        k_cache = torch.randn(
            (prompt_len, num_kv_heads, head_dim),
            dtype=torch.float32,
            device=device,
        )
        req_to_tokens = torch.arange(
            prompt_len, dtype=torch.int32, device=device
        ).unsqueeze(0)
        req_indices = torch.tensor([0], dtype=torch.int32, device=device)
        b_start_loc = torch.tensor([0], dtype=torch.int32, device=device)
        context_lens = torch.tensor([prompt_len], dtype=torch.int32, device=device)
        prompt_cache_lens = torch.tensor(
            [prompt_len - window], dtype=torch.int32, device=device
        )
        score_starts = prompt_cache_lens.clone()
        score_ends = context_lens.clone()
        actual = torch.empty((1, prompt_len), dtype=torch.float32, device=device)

        prefill_score_fwd(
            q,
            k_cache,
            actual,
            req_indices,
            b_start_loc,
            context_lens,
            prompt_cache_lens,
            window,
            req_to_tokens,
            score_starts,
            score_ends,
            candidate_start=1,
            recent_keep_tokens=1,
            score_mode="logits",
        )
        expected = _raw_qk_score_baseline(
            q,
            k_cache,
            req_to_tokens,
            req_indices,
            b_start_loc,
            context_lens,
            prompt_cache_lens,
            score_starts,
            score_ends,
            candidate_start=1,
            recent_keep_tokens=1,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_probability_score_supports_full_query_larger_than_128(self):
        from sparseengine.kernels.triton.prefill_score import prefill_score_fwd

        torch.manual_seed(37)
        device = "cuda"
        prompt_len = 192
        query_len = 160
        num_heads = 3
        num_kv_heads = 1
        head_dim = 16
        prompt_cache_len = prompt_len - query_len
        q = torch.randn(
            (query_len, num_heads, head_dim), dtype=torch.float32, device=device
        )
        k_cache = torch.randn(
            (prompt_len, num_kv_heads, head_dim),
            dtype=torch.float32,
            device=device,
        )
        req_to_tokens = torch.arange(
            prompt_len, dtype=torch.int32, device=device
        ).unsqueeze(0)
        req_indices = torch.tensor([0], dtype=torch.int32, device=device)
        b_start_loc = torch.tensor([0], dtype=torch.int32, device=device)
        context_lens = torch.tensor([prompt_len], dtype=torch.int32, device=device)
        prompt_cache_lens = torch.tensor(
            [prompt_cache_len], dtype=torch.int32, device=device
        )
        score_starts = prompt_cache_lens.clone()
        score_ends = context_lens.clone()
        actual = torch.empty((1, prompt_len), dtype=torch.float32, device=device)

        prefill_score_fwd(
            q,
            k_cache,
            actual,
            req_indices,
            b_start_loc,
            context_lens,
            prompt_cache_lens,
            query_len,
            req_to_tokens,
            score_starts,
            score_ends,
            score_mode="probability",
        )
        expected = _prefill_score_baseline(
            q,
            k_cache,
            req_to_tokens,
            req_indices,
            b_start_loc,
            context_lens,
            prompt_cache_lens,
            score_starts,
            score_ends,
            0,
            0,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_probability_score_reuses_fa3_lse(self):
        from sparseengine.kernels.triton.prefill_score import (
            prefill_score_from_lse_fwd,
        )

        torch.manual_seed(41)
        device = "cuda"
        prompt_len = 73
        query_len = 41
        prompt_cache_len = prompt_len - query_len
        num_heads = 4
        num_kv_heads = 2
        head_dim = 16
        q = torch.randn(
            (query_len, num_heads, head_dim), dtype=torch.float32, device=device
        )
        k_cache = torch.randn(
            (prompt_len, num_kv_heads, head_dim),
            dtype=torch.float32,
            device=device,
        )
        req_to_tokens = torch.arange(
            prompt_len, dtype=torch.int32, device=device
        ).unsqueeze(0)
        req_indices = torch.tensor([0], dtype=torch.int32, device=device)
        b_start_loc = torch.tensor([0], dtype=torch.int32, device=device)
        context_lens = torch.tensor([prompt_len], dtype=torch.int32, device=device)
        prompt_cache_lens = torch.tensor(
            [prompt_cache_len], dtype=torch.int32, device=device
        )
        score_starts = prompt_cache_lens.clone()
        score_ends = context_lens.clone()
        lse = torch.empty(
            (num_heads, query_len), dtype=torch.float32, device=device
        )
        heads_per_kv = num_heads // num_kv_heads
        scale = head_dim**-0.5
        for q_rel in range(query_len):
            q_pos = prompt_cache_len + q_rel
            for head in range(num_heads):
                kv_head = head // heads_per_kv
                logits = torch.matmul(
                    k_cache[: q_pos + 1, kv_head].float(),
                    q[q_rel, head].float(),
                ) * scale
                lse[head, q_rel] = torch.logsumexp(logits, dim=0)
        actual = torch.empty((1, prompt_len), dtype=torch.float32, device=device)

        prefill_score_from_lse_fwd(
            q,
            k_cache,
            lse,
            actual,
            req_indices,
            b_start_loc,
            context_lens,
            prompt_cache_lens,
            query_len,
            req_to_tokens,
            score_starts,
            score_ends,
        )
        expected = _prefill_score_baseline(
            q,
            k_cache,
            req_to_tokens,
            req_indices,
            b_start_loc,
            context_lens,
            prompt_cache_lens,
            score_starts,
            score_ends,
            0,
            0,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_score_batch_indices_pack_active_rows(self):
        from sparseengine.kernels.triton.prefill_score import prefill_score_fwd

        torch.manual_seed(31)
        device = "cuda"
        context_lens = torch.tensor([12, 9], dtype=torch.int32, device=device)
        prompt_cache_lens = torch.tensor([8, 5], dtype=torch.int32, device=device)
        b_start_loc = torch.tensor([0, 4], dtype=torch.int32, device=device)
        req_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
        req_to_tokens = torch.full((2, 12), -1, dtype=torch.int32, device=device)
        req_to_tokens[0, :12] = torch.arange(12, dtype=torch.int32, device=device)
        req_to_tokens[1, :9] = torch.arange(12, 21, dtype=torch.int32, device=device)
        q = torch.randn((8, 4, 16), dtype=torch.float32, device=device)
        k_cache = torch.randn((21, 2, 16), dtype=torch.float32, device=device)
        full_starts = torch.tensor([8, 5], dtype=torch.int32, device=device)
        full_ends = torch.tensor([12, 9], dtype=torch.int32, device=device)
        batch_indices = torch.tensor([1], dtype=torch.int32, device=device)

        for mode in ("probability", "logits"):
            with self.subTest(mode=mode):
                full = torch.empty((2, 12), dtype=torch.float32, device=device)
                packed = torch.empty((1, 12), dtype=torch.float32, device=device)
                prefill_score_fwd(
                    q,
                    k_cache,
                    full,
                    req_indices,
                    b_start_loc,
                    context_lens,
                    prompt_cache_lens,
                    4,
                    req_to_tokens,
                    full_starts,
                    full_ends,
                    candidate_start=1,
                    recent_keep_tokens=1,
                    score_mode=mode,
                )
                prefill_score_fwd(
                    q,
                    k_cache,
                    packed,
                    req_indices,
                    b_start_loc,
                    context_lens,
                    prompt_cache_lens,
                    4,
                    req_to_tokens,
                    full_starts[1:],
                    full_ends[1:],
                    candidate_start=1,
                    recent_keep_tokens=1,
                    score_mode=mode,
                    batch_indices=batch_indices,
                )
                torch.testing.assert_close(packed[0], full[1], rtol=2e-2, atol=2e-2)

    def test_prefill_score_matches_torch_for_query_range(self):
        from sparseengine.kernels.triton.prefill_score import prefill_score_fwd

        torch.manual_seed(7)
        device = "cuda"
        dtype = torch.float32
        num_heads = 4
        num_kv_heads = 2
        head_dim = 16
        context_lens = torch.tensor([12, 8], dtype=torch.int32, device=device)
        prompt_cache_lens = torch.tensor([7, 5], dtype=torch.int32, device=device)
        chunk_lens = context_lens - prompt_cache_lens
        b_start_loc = torch.tensor([0, int(chunk_lens[0].item())], dtype=torch.int32, device=device)
        max_len = int(context_lens.max().item())
        req_to_tokens = torch.full((3, max_len), -1, dtype=torch.int32, device=device)
        req_to_tokens[0, :8] = torch.arange(20, 28, dtype=torch.int32, device=device)
        req_to_tokens[1, :12] = torch.arange(0, 12, dtype=torch.int32, device=device)
        b_req_idx = torch.tensor([1, 0], dtype=torch.int32, device=device)
        score_starts = torch.tensor([8, 6], dtype=torch.int32, device=device)
        score_ends = torch.tensor([11, 8], dtype=torch.int32, device=device)
        candidate_start = 1
        recent_keep_tokens = 2

        total_q = int(chunk_lens.sum().item())
        q = torch.randn((total_q, num_heads, head_dim), dtype=dtype, device=device)
        k_cache = torch.randn((32, num_kv_heads, head_dim), dtype=dtype, device=device)
        attn_score = torch.zeros((2, max_len), dtype=torch.float32, device=device)

        prefill_score_fwd(
            q,
            k_cache,
            attn_score,
            b_req_idx,
            b_start_loc,
            context_lens,
            prompt_cache_lens,
            int(chunk_lens.max().item()),
            req_to_tokens,
            score_starts,
            score_ends,
            candidate_start=candidate_start,
            recent_keep_tokens=recent_keep_tokens,
        )
        torch.cuda.synchronize()

        expected = _prefill_score_baseline(
            q,
            k_cache,
            req_to_tokens,
            b_req_idx,
            b_start_loc,
            context_lens,
            prompt_cache_lens,
            score_starts,
            score_ends,
            candidate_start,
            recent_keep_tokens,
        )
        torch.testing.assert_close(attn_score, expected, rtol=2e-2, atol=2e-2)

    def test_prefill_score_handles_offset_query_window(self):
        from sparseengine.kernels.triton.prefill_score import prefill_score_fwd

        torch.manual_seed(11)
        device = "cuda"
        dtype = torch.float32
        num_heads = 3
        num_kv_heads = 1
        head_dim = 16
        prompt_len = 14
        prompt_cache_len = 9
        chunk_len = 5
        score_start = 10
        score_end = prompt_len
        req_to_tokens = torch.arange(0, prompt_len, dtype=torch.int32, device=device).unsqueeze(0)
        b_req_idx = torch.tensor([0], dtype=torch.int32, device=device)
        q = torch.randn((chunk_len, num_heads, head_dim), dtype=dtype, device=device)
        k_cache = torch.randn((prompt_len, num_kv_heads, head_dim), dtype=dtype, device=device)
        acc = torch.zeros((1, prompt_len), dtype=torch.float32, device=device)
        context_lens = torch.tensor([prompt_len], dtype=torch.int32, device=device)
        prompt_cache_lens = torch.tensor([prompt_cache_len], dtype=torch.int32, device=device)
        b_start_loc = torch.tensor([0], dtype=torch.int32, device=device)
        score_starts = torch.tensor([score_start], dtype=torch.int32, device=device)
        score_ends = torch.tensor([score_end], dtype=torch.int32, device=device)
        candidate_start = 2
        recent_keep_tokens = 4
        prefill_score_fwd(
            q,
            k_cache,
            acc,
            b_req_idx,
            b_start_loc,
            context_lens,
            prompt_cache_lens,
            chunk_len,
            req_to_tokens,
            score_starts,
            score_ends,
            candidate_start=candidate_start,
            recent_keep_tokens=recent_keep_tokens,
        )
        torch.cuda.synchronize()

        expected = _prefill_score_baseline(
            q,
            k_cache,
            req_to_tokens,
            b_req_idx,
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.tensor([prompt_len], dtype=torch.int32, device=device),
            torch.tensor([prompt_cache_len], dtype=torch.int32, device=device),
            score_starts,
            score_ends,
            candidate_start,
            recent_keep_tokens,
        )
        torch.testing.assert_close(acc, expected, rtol=2e-2, atol=2e-2)

    def test_prefill_score_matches_torch_for_gqa_seven_heads(self):
        from sparseengine.kernels.triton.prefill_score import prefill_score_fwd

        torch.manual_seed(17)
        device = "cuda"
        dtype = torch.bfloat16
        num_heads = 28
        num_kv_heads = 4
        head_dim = 32
        prompt_len = 96
        window = 32
        prompt_cache_len = prompt_len - window
        req_to_tokens = torch.arange(0, prompt_len, dtype=torch.int32, device=device).unsqueeze(0)
        b_req_idx = torch.tensor([0], dtype=torch.int32, device=device)
        q = torch.randn((window, num_heads, head_dim), dtype=dtype, device=device)
        k_cache = torch.randn((prompt_len, num_kv_heads, head_dim), dtype=dtype, device=device)
        acc = torch.zeros((1, prompt_len), dtype=torch.float32, device=device)
        context_lens = torch.tensor([prompt_len], dtype=torch.int32, device=device)
        prompt_cache_lens = torch.tensor([prompt_cache_len], dtype=torch.int32, device=device)
        b_start_loc = torch.tensor([0], dtype=torch.int32, device=device)
        score_starts = torch.tensor([prompt_cache_len], dtype=torch.int32, device=device)
        score_ends = torch.tensor([prompt_len], dtype=torch.int32, device=device)
        candidate_start = 3
        recent_keep_tokens = 9

        prefill_score_fwd(
            q,
            k_cache,
            acc,
            b_req_idx,
            b_start_loc,
            context_lens,
            prompt_cache_lens,
            window,
            req_to_tokens,
            score_starts,
            score_ends,
            candidate_start=candidate_start,
            recent_keep_tokens=recent_keep_tokens,
        )
        torch.cuda.synchronize()

        expected = _prefill_score_baseline(
            q,
            k_cache,
            req_to_tokens,
            b_req_idx,
            b_start_loc,
            context_lens,
            prompt_cache_lens,
            score_starts,
            score_ends,
            candidate_start,
            recent_keep_tokens,
        )
        torch.testing.assert_close(acc, expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
