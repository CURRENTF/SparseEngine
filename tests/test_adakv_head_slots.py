"""Protect the allocation objective, ragged capacity and nonempty-head contract."""

import itertools
import math
import os

import pytest
import torch

from sparseengine.engine.sparse_methods.adakv import select_adakv_indices


def exhaustive_oracle(rows, budget, alpha, sink=0, recent=0):
    """Enumerate every feasible subset, independently of the production sorts."""
    candidates = [(h, t) for h, row in enumerate(rows) for t in range(len(row))]
    required = {(h, t) for h, row in enumerate(rows) for t in range(len(row))
                if t < sink or t >= len(row) - recent}
    minima = [min(len(row), max(1, math.floor(budget * alpha))) for row in rows]
    target = min(len(candidates), len(rows) * budget)
    best = None
    for selected in itertools.combinations(candidates, target):
        if not required.issubset(selected):
            continue
        if any(sum(h == head for h, _ in selected) < n for head, n in enumerate(minima)):
            continue
        score = sum(rows[h][t] for h, t in selected)
        if best is None or score > best[0]:
            best = (score, selected)
    assert best is not None
    return tuple(tuple(t for h, t in best[1] if h == head) for head in range(len(rows)))


CASES = [
    ([[9., 8., 7., 6.], [3., 2., 1.], [.3, .2]], 2, .5, 0, 0),
    ([[.1, 6., 3., .2], [.4, 2., 5., .3]], 3, .5, 1, 1),
    ([[.01], [9., 8., 7., 6.], [5., 4., 3., 2.]], 2, 1., 0, 0),
    ([[1., 1., 1.], [1., 1., 1.]], 2, .5, 0, 0),
]


@pytest.mark.parametrize("rows,budget,alpha,sink,recent", CASES)
def test_allocation_maximizes_score_subject_to_independent_constraints(rows, budget, alpha, sink, recent):
    width = max(map(len, rows))
    scores = torch.full((len(rows), width), float("nan"))
    for h, row in enumerate(rows):
        scores[h, :len(row)] = torch.tensor(row)
    actual = select_adakv_indices(scores, list(map(len, rows)), budget=budget, alpha=alpha,
                                  sink_keep=sink, recent_keep=recent)
    assert tuple(tuple(x.tolist()) for x in actual) == exhaustive_oracle(rows, budget, alpha, sink, recent)


def test_pooling_cannot_leak_padded_scores_and_budget_growth_cannot_resurrect():
    rows = [[.1, .7], [.3, .2, .8, .4, .6]]
    scores = torch.full((2, 5), 1e20)
    for h, row in enumerate(rows):
        scores[h, :len(row)] = torch.tensor(row)
    original = scores.clone()
    pooled = [[max(row[max(0, i - 1):i + 2]) for i in range(len(row))] for row in rows]
    actual = select_adakv_indices(scores, [2, 5], budget=2, alpha=.5, pool_kernel_size=3)
    assert tuple(tuple(x.tolist()) for x in actual) == exhaustive_oracle(pooled, 2, .5)
    torch.testing.assert_close(scores, original, atol=0, rtol=0)
    # Even a much larger target only returns the supplied candidate domain.
    grown = select_adakv_indices(scores, [2, 5], budget=8, alpha=.5)
    assert [x.tolist() for x in grown] == [list(range(2)), list(range(5))]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_candidate_scores_fail_before_a_selection_is_published(bad):
    with pytest.raises(ValueError, match="NaN or infinity"):
        select_adakv_indices(torch.tensor([[1., bad], [2., 3.]]), [2, 2], budget=1, alpha=.5)


@pytest.mark.skipif(os.environ.get("SPARSEENGINE_TEST_CUDA") != "1",
                    reason="Requires an explicitly selected idle CUDA device")
def test_cuda_allocation_matches_exhaustive_oracle():
    for rows, budget, alpha, sink, recent in CASES:
        scores = torch.full((len(rows), max(map(len, rows))), float("nan"), device="cuda")
        for h, row in enumerate(rows):
            scores[h, :len(row)] = torch.tensor(row, device="cuda")
        actual = select_adakv_indices(scores, list(map(len, rows)), budget=budget, alpha=alpha,
                                      sink_keep=sink, recent_keep=recent)
        assert tuple(tuple(x.tolist()) for x in actual) == exhaustive_oracle(rows, budget, alpha, sink, recent)


def test_prefill_compaction_precedes_decode_and_respects_mixed_chunk_boundaries():
    """Catch turn-end-only compression, which makes single-turn quality dense."""
    from types import SimpleNamespace
    from sparseengine.engine.sparse_methods.adakv import AdaKVRuntime
    from sparseengine.engine.sparse_methods.base import SparseStepContext

    runtime = object.__new__(AdaKVRuntime)
    events = []
    runtime.cache_manager = SimpleNamespace(
        compact_head_slots=lambda sid, **kw: events.append((sid, kw["reason"])))
    seqs = [SimpleNamespace(seq_id=7, is_last_chunk_prefill=False),
            SimpleNamespace(seq_id=9, is_last_chunk_prefill=True)]
    runtime.finish_step(SparseStepContext(seqs, True, None))
    assert events == [(9, "prefill")]
    runtime.finish_step(SparseStepContext(seqs, False, None))
    assert events == [(9, "prefill")]
    seqs[0].is_last_chunk_prefill = True
    runtime.finish_step(SparseStepContext(seqs[:1], True, None))
    assert events == [(9, "prefill"), (7, "prefill")]


def test_adakv_chain_fingerprint_rejects_changed_selection_semantics():
    """Prevent reuse of cached KV under a different budget/window/selection rule."""
    from types import SimpleNamespace
    from sparseengine.engine.chain_cache import build_chain_cache_fingerprint

    config = SimpleNamespace(sparse_method="adakv", model="model", kv_cache_layout="head_slots",
                             hf_config=SimpleNamespace(model_type="qwen3", dtype=torch.bfloat16),
                             sink_keep_tokens=0, recent_keep_tokens=2, decode_keep_tokens=6,
                             observation_window_size=4, pool_kernel_size=3, adakv_alpha=.2,
                             sparse_prefill_score_mode="probability")
    original = build_chain_cache_fingerprint(config)
    for field, value in (("decode_keep_tokens", 7), ("observation_window_size", 8),
                         ("pool_kernel_size", 1), ("adakv_alpha", .5)):
        changed = SimpleNamespace(**vars(config))
        setattr(changed, field, value)
        assert build_chain_cache_fingerprint(changed) != original


def test_head_slot_config_rejects_incompatible_storage_and_lifecycle_before_allocation():
    """Avoid late kernel errors or accidental radix/offload under a ragged layout."""
    from types import SimpleNamespace
    from sparseengine.configs.head_slots import normalize_head_slot_layout, validate_head_slots

    config = SimpleNamespace(
        sparse_method="adakv", kv_cache_layout="auto", head_slot_budgets=None,
        prefill_sparse_method="", world_size=1, enable_prefix_caching=False,
        enable_prefix_cache_offload=False, resolved_prefix_cache_mode="disabled",
        attention_cache_layout="explicit_kv", quantization_config=SimpleNamespace(enabled=False),
        hf_config=SimpleNamespace(dtype=torch.bfloat16, model_type="qwen3",
                                  num_attention_heads=4, num_key_value_heads=2),
        runtime_layout=SimpleNamespace(heterogeneous_kv=False, linear_attention_layer_indices=(),
                                       num_kv_layers=2, num_layers=2),
        observation_window_size=4, pool_kernel_size=1, snapkv_decode_eviction=False,
        snapkv_num_full_layers=0, full_attention_layers=[], pyramid_layer_ratios=None,
        adakv_alpha=.5, sink_keep_tokens=1, recent_keep_tokens=1, decode_keep_tokens=2,
        max_model_len=16, sparse_prefill_score_mode="probability")
    normalize_head_slot_layout(config)
    validate_head_slots(config)
    for changes, error in [
        ({"kv_cache_layout": "token"}, "requires independent"),
        ({"enable_prefix_caching": True, "resolved_prefix_cache_mode": "radix"}, "exclusive chain"),
        ({"enable_prefix_cache_offload": True}, "exclusive chain"),
        ({"head_slot_budgets": [[2, 2], [2, 2]]}, "do not set"),
        ({"sparse_prefill_score_mode": "logits"}, "probability"),
        ({"world_size": 2}, "TP=EP=DP=1"),
        ({"snapkv_decode_eviction": True}, "decode eviction"),
    ]:
        changed = SimpleNamespace(**(vars(config) | changes))
        with pytest.raises(ValueError, match=error):
            normalize_head_slot_layout(changed)
            validate_head_slots(changed)
