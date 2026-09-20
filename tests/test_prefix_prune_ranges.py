"""Multi-range pruning contracts: union budgets, failure isolation and score packing."""
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from sparseengine.engine.llm_engine import LLMEngine
from sparseengine.engine.model_runner import ModelRunner
from sparseengine.engine.prefix_prune import normalize_prefix_prune_ranges, validate_prefix_prune_request
from sparseengine.entrypoints.openai.protocol.prefix_cache import PrefixCachePruneRequest


@pytest.mark.parametrize("kwargs", [
    {"ranges": []},
    {"ranges": [(0, 4), (2, 6)]},
    {"ranges": [(0, 2), (0, 2)]},
    {"ranges": [(0, 3)]},
    {"ranges": [(4, 2)]},
    {"ranges": [(0, 10)]},
    {"ranges": [(0, 2)], "range_start": 0, "range_end": 2},
    {"ranges": [(0.0, 2)]},
    {"ranges": [(False, 2)]},
])
def test_invalid_ranges_fail_before_enqueue(kwargs):
    engine = SimpleNamespace(
        config=SimpleNamespace(sparse_method="omnikv", prefix_cache_block_size=2, enable_prefix_caching=True),
        _prefix_prune_jobs={}, _pending_prefix_prune_ids=deque(),
    )
    with pytest.raises(ValueError):
        LLMEngine.prefix_cache_prune_start(engine, list(range(8)), keep_tokens=1, policy="kvzip_global", **kwargs)
    assert not engine._prefix_prune_jobs and not engine._pending_prefix_prune_ids


def test_union_budget_excludes_gaps_and_normalizes_arbitrary_length():
    spans = [(i, i + 1) for i in reversed(range(0, 100, 2))]
    normalized = normalize_prefix_prune_ranges(token_count=100, block_size=1, ranges=spans)
    assert normalized == sorted(spans)
    with pytest.raises(ValueError, match="total range width"):
        validate_prefix_prune_request(
            token_count=100, block_size=1, ranges=spans, keep_tokens=50, policy="kvzip_global",
        )
    assert normalize_prefix_prune_ranges(
        token_count=8, block_size=2, ranges=[(4, 8), (0, 2), (2, 4)],
    ) == [(0, 8)]


@pytest.mark.parametrize("kwargs", [
    {}, {"ranges": []}, {"range_start": 0},
    {"ranges": [[0, 2]], "range_start": 0, "range_end": 2},
    {"ranges": [[0.5, 2]]}, {"ranges": [[0, 2, 3]]},
])
def test_http_selector_rejects_ambiguous_or_lossy_input(kwargs):
    with pytest.raises(ValueError):
        PrefixCachePruneRequest(token_ids=[0, 1], keep_tokens=1, policy="kvzip_global", **kwargs)


def _runner(fail_at=None):
    events = []
    def validate(*args, **kwargs):
        events.append(("validate", kwargs))
    def forward(**kwargs):
        events.append(("score", kwargs))
        if fail_at is not None and sum(kind == "score" for kind, _ in events) == fail_at:
            raise RuntimeError("scoring failed")
        # Gap tokens have the highest scores and must never consume the budget.
        scores = torch.tensor([1., 8., 100., 100., 7., 6., 100., 100., 8., 2.])
        return scores
    def commit(*args, **kwargs):
        events.append(("commit", kwargs))
        return kwargs
    def reduce(scores, **kwargs):
        events.append(("reduce", scores.clone()))
    runner = SimpleNamespace(
        config=SimpleNamespace(prefix_cache_block_size=1), device=torch.device("cpu"),
        cache_manager=SimpleNamespace(validate_prefix_cache_prune_target=validate, prefix_cache_prune=commit),
        parallel_context=SimpleNamespace(world=SimpleNamespace(all_reduce=reduce)),
        _prefix_prune_score_forward=forward,
    )
    return runner, events


def test_kvzip_scores_all_ranges_before_one_shared_selection_and_commit():
    runner, events = _runner()
    result = ModelRunner.prefix_cache_prune(
        runner, list(range(10)), ranges=[(8, 10), (0, 2), (4, 6)], keep_tokens=3,
        policy="kvzip_global", prune_id="multi", score_chunk_size=1,
        prev_postfix_size=1, kvzip_replay_prefix_ids=[99],
    )
    # Packed union [0,1,4,5,8,9], global oracle selects logical [1,4,8].
    assert result["keep_indices"].tolist() == [1, 2, 4]
    forwards = [kwargs for kind, kwargs in events if kind == "score"]
    assert [kwargs["token_ids"][11:] for kwargs in forwards] == [[0], [0, 1], [4], [4, 5], [8], [8, 9]]
    assert all(kwargs["token_ids"][:11] == list(range(10)) + [99] for kwargs in forwards)
    assert all(kwargs["prefix_hit_len"] == 10 for kwargs in forwards)
    assert [kind for kind, _ in events].count("reduce") == 1
    assert events[-1][0] == "commit"
    assert events[-2][1].tolist() == [1., 8., 7., 6., 8., 2.]


def test_later_scoring_failure_never_commits_earlier_intervals():
    runner, events = _runner(fail_at=2)
    with pytest.raises(RuntimeError, match="scoring failed"):
        ModelRunner.prefix_cache_prune(
            runner, list(range(10)), ranges=[(0, 2), (4, 6), (8, 10)], keep_tokens=1,
            kvzip_replay_prefix_ids=[99], policy="kvzip_global",
        )
    assert not any(kind == "commit" for kind, _ in events)


def test_zero_budget_skips_all_reconstruction_forwards():
    runner, events = _runner()
    result = ModelRunner.prefix_cache_prune(
        runner, list(range(10)), ranges=[(0, 2), (4, 6), (8, 10)], keep_tokens=0, policy="kvzip_global",
    )
    assert result["keep_indices"].numel() == 0
    assert [kind for kind, _ in events] == ["validate", "commit"]


def test_snapkv_shared_budget_preserves_observation_tokens_only_in_union():
    runner, events = _runner()
    result = ModelRunner.prefix_cache_prune(
        runner, list(range(10)), ranges=[(0, 2), (4, 6), (8, 10)], keep_tokens=3,
        policy="snapkv_global", observation_tokens=2,
    )
    assert result["keep_indices"].tolist() == [1, 4, 5]
    assert sum(kind == "score" for kind, _ in events) == 1


def test_engine_job_round_trip_preserves_ranges_and_budget():
    runner, events = _runner()
    def call(method, *args):
        assert method == "prefix_cache_prune_batch"
        jobs, replay = args
        return [{"result": ModelRunner.prefix_cache_prune(
            runner, **job, policy="kvzip_global", kvzip_replay_prefix_ids=replay,
        )} for job in jobs]
    engine = SimpleNamespace(
        config=SimpleNamespace(sparse_method="omnikv", prefix_cache_block_size=1, enable_prefix_caching=True),
        _prefix_prune_jobs={}, _pending_prefix_prune_ids=deque(),
        tokenizer=SimpleNamespace(encode=lambda *args, **kwargs: [99]),
        model_runner=SimpleNamespace(call=call),
    )
    queued = LLMEngine.prefix_cache_prune_start(
        engine, list(range(10)), ranges=[(8, 10), (0, 2), (4, 6)], keep_tokens=3, policy="kvzip_global",
    )
    assert queued["ranges"] == [[0, 2], [4, 6], [8, 10]]
    assert LLMEngine.run_pending_prefix_prune(engine)
    job = engine._prefix_prune_jobs[queued["prune_id"]]
    assert job.status == "completed", job.error
    assert job.result["keep_indices"].tolist() == [1, 2, 4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_device_global_selection_matches_independent_tie_order():
    from sparseengine.engine.prefix_prune import select_global_keep_indices
    # Repeated values and a non-contiguous layout expose unstable GPU tie selection.
    values = [float((i * 17) % 31) for i in range(4097)]
    values[0], values[-1] = float("-inf"), float("inf")
    scores = torch.tensor([[v, -v] for v in values], device="cuda")[:, 0]
    expected_order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    for count in (0, 137, len(values)):
        actual = select_global_keep_indices(scores, keep_tokens=count)
        assert actual.device == scores.device
        assert actual.tolist() == sorted(expected_order[:count])


def test_missing_budget_never_defaults_to_deleting_everything():
    runner, events = _runner()
    with pytest.raises(ValueError, match="keep_tokens must be an integer"):
        ModelRunner.prefix_cache_prune(
            runner, list(range(10)), ranges=[(0, 2)], policy="kvzip_global",
        )
    assert not events


@pytest.mark.parametrize("rows,slots,expected_batches", [
    (3, 100, [[0,20,0], [0]]),
    (1, 100, [[0], [20], [0], [0]]),
    (3, 4, [[0], [20], [0], [0]]),
])
def test_batched_reconstruction_round_robin_preserves_independent_budgets(rows, slots, expected_batches):
    # Two jobs with different ranges must interleave chunks, then fill spare
    # rows with the remaining task; no high-scoring gap may consume a budget.
    class Cache:
        def __init__(self):
            self.blocks = [SimpleNamespace(ref_count=0), SimpleNamespace(ref_count=0)]
        def block_ids_for_tokens(self, tokens, **kwargs):
            return tokens
        def match_longest_block_ids(self, tokens):
            return len(tokens), tokens[0], len(tokens)
        def get_chain(self, last, count):
            return [self.blocks[last // 20]]
        def acquire_block_ref(self, block):
            block.ref_count += 1
        def release_block_ref(self, block):
            block.ref_count -= 1
    cache = Cache()
    batches, commits = [], []
    def validate(tokens, **kwargs):
        return cache.get_chain(tokens[0], len(tokens))
    def forward(requests):
        batches.append([r['token_ids'][0] for r in requests])
        assert all(b.ref_count == 1 for b in cache.blocks)
        return [torch.tensor([1., 8., 100., 100., 7., 6., 100., 100., 8., 2.]) for _ in requests]
    def commit(tokens, **kwargs):
        assert all(b.ref_count == 0 for b in cache.blocks)
        commits.append(kwargs['keep_indices'].tolist())
        return {}
    from contextlib import contextmanager
    @contextmanager
    def reserve(requests, *, prefix_blocks):
        assert set(prefix_blocks) == {sid for sid, _ in requests}
        assert all(blocks for blocks in prefix_blocks.values())
        total = 0
        count = 0
        for _, size in requests[:rows]:
            if total + size > slots:
                break
            total += size
            count += 1
        assert count > 0
        yield count
    runner = SimpleNamespace(
        config=SimpleNamespace(prefix_cache_block_size=1, max_model_len=100,
                               max_num_batched_tokens=100, engine_prefill_chunk_size=100,
                               max_num_seqs_in_batch=3), device=torch.device('cpu'),
        cache_manager=SimpleNamespace(prefix_cache=cache, reserve_prefill_slots=reserve,
            validate_prefix_cache_prune_target=validate, prefix_cache_prune=commit),
        parallel_context=SimpleNamespace(world=SimpleNamespace(all_reduce=lambda *a, **k: None)),
        _prefix_prune_score_forward_batch=forward,
    )
    jobs = [dict(token_ids=list(range(base, base+10)), ranges=ranges, keep_tokens=keep,
                 allow_recompress=False, score_chunk_size=2, prev_postfix_size=1, prune_id=str(base))
            for base, ranges, keep in [(0, [(0,2),(4,6),(8,10)], 3), (20, [(0,2)], 1)]]
    results = ModelRunner.prefix_cache_prune_batch(runner, jobs, [99])
    assert batches == expected_batches
    assert commits == [[1,2,4], [1]]
    assert all('result' in result for result in results)
    batches.clear()
    commits.clear()
    def fail(requests):
        raise RuntimeError('scoring failed')
    runner._prefix_prune_score_forward_batch = fail
    with pytest.raises(RuntimeError, match='scoring failed'):
        ModelRunner.prefix_cache_prune_batch(runner, jobs, [99])
    assert not commits
    assert all(b.ref_count == 0 for b in cache.blocks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_explicit_kv_batched_scores_keep_row_normalizers_and_layer_maxima():
    from sparseengine.engine.cache_manager.standard import StandardCacheManager
    from sparseengine.engine.cache_manager.base import AttentionViewMeta, ExplicitKVPayload, PrefillComputeView
    torch.manual_seed(27)
    q = torch.randn(5, 4, 64, device='cuda', dtype=torch.float16)
    keys = torch.randn(20, 2, 64, device='cuda', dtype=torch.float16)
    slots = torch.randperm(20, device='cuda', dtype=torch.int32).reshape(2,10)
    rows = torch.tensor([1,0], device='cuda', dtype=torch.int32)
    lengths = torch.tensor([7,9], device='cuda', dtype=torch.int32)
    manager = object.__new__(StandardCacheManager)
    states = [dict(score=None, physical_window=(4,7,1)), dict(score=None, physical_window=(7,9,3))]
    manager._prefix_prune_scoring = dict(batch=states)
    view = PrefillComputeView(
        meta=AttentionViewMeta(active_slots=slots, req_indices=rows, context_lens=lengths),
        payload=ExplicitKVPayload(k_cache=keys, v_cache=keys),
    )
    expected = [torch.zeros(7,device='cuda'), torch.zeros(9,device='cuda')]
    for query in (q, -q):
        manager.collect_prefill_attention_score(0, query, view,
            b_start_loc=torch.tensor([0,3], device='cuda', dtype=torch.int32),
            chunk_lens=torch.tensor([3,2], device='cuda', dtype=torch.int32))
        for i, (a,b,lo,hi) in enumerate([(0,3,1,4),(3,5,3,7)]):
            k = keys[slots[1-i,lo:hi].long()].float().repeat_interleave(2,dim=1)
            logits = torch.einsum('qhd,khd->hqk',query[a:b].float(),k) / 8
            score = logits.softmax(-1).mean(1).amax(0)
            expected[i][lo:hi] = torch.maximum(expected[i][lo:hi],score)
            torch.testing.assert_close(states[i]['score'],expected[i],atol=3e-4,rtol=.003)
