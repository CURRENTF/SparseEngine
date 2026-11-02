"""Multi-range pruning contracts: union budgets, failure isolation and score packing."""
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from sparseengine.engine.llm_engine import LLMEngine
from sparseengine.engine.model_runner import ModelRunner
from sparseengine.engine.prefix_prune import (
    normalize_prefix_prune_ranges,
    select_global_keep_block_indices,
    select_global_keep_indices,
    validate_prefix_prune_request,
)
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


def test_quest_job_rejects_non_page_aligned_keep_budget_before_enqueue():
    engine = SimpleNamespace(
        config=SimpleNamespace(
            sparse_method="quest",
            prefix_cache_block_size=2,
            enable_prefix_caching=True,
        ),
        _prefix_prune_jobs={},
        _pending_prefix_prune_ids=deque(),
    )
    with pytest.raises(ValueError, match="page aligned"):
        LLMEngine.prefix_cache_prune_start(
            engine,
            list(range(8)),
            ranges=[(0, 4)],
            keep_tokens=1,
            policy="kvzip_global",
        )
    assert not engine._prefix_prune_jobs and not engine._pending_prefix_prune_ids


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
        return [block]
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
    def select(scores, *, keep_tokens, protected_suffix_tokens=0):
        candidate_count = scores.numel() - protected_suffix_tokens
        selected = select_global_keep_indices(
            scores[:candidate_count],
            keep_tokens=keep_tokens - protected_suffix_tokens,
        )
        if protected_suffix_tokens:
            selected = torch.cat((selected, torch.arange(candidate_count, scores.numel())))
        return selected
    from contextlib import contextmanager
    @contextmanager
    def reserve(requests, *, prefix_blocks):
        yield len(requests)
    block = SimpleNamespace(ref_count=0)
    cache = SimpleNamespace(
        block_ids_for_tokens=lambda ids, **kwargs: ids,
        match_longest_block_ids=lambda ids: (len(ids), 1, len(ids)),
        get_chain=lambda last, count: [block] * count,
        acquire_block_ref=lambda item: setattr(item, "ref_count", item.ref_count + 1),
        release_block_ref=lambda item: setattr(item, "ref_count", item.ref_count - 1),
    )
    runner = SimpleNamespace(
        config=SimpleNamespace(prefix_cache_block_size=1, max_model_len=100,
                               max_num_batched_tokens=100, engine_prefill_chunk_size=100,
                               max_num_seqs_in_batch=1), device=torch.device("cpu"),
        cache_manager=SimpleNamespace(
            prefix_cache=cache, reserve_prefill_slots=reserve,
            validate_prefix_cache_prune_target=validate,
            validate_prefix_prune_keep_tokens=lambda _: None,
            select_prefix_prune_keep_indices=select,
            prefix_cache_prune=commit,
        ),
        parallel_context=SimpleNamespace(world=SimpleNamespace(all_reduce=reduce)),
        _prefix_prune_score_forward=forward,
        _prefix_prune_score_forward_batch=lambda requests: [forward(**r) for r in requests],
    )
    return runner, events


def test_block_selection_sums_token_scores_and_expands_whole_blocks():
    scores = torch.tensor([5.0, 0.0, 3.0, 3.0, 1.0, 1.0])
    selected = select_global_keep_block_indices(
        scores, keep_tokens=2, block_size=2,
    )
    assert selected.tolist() == [2, 3]


def test_block_selection_preserves_aligned_protected_suffix():
    scores = torch.tensor([9.0, 9.0, 1.0, 1.0, 0.0, 0.0])
    selected = select_global_keep_block_indices(
        scores,
        keep_tokens=4,
        block_size=2,
        protected_suffix_tokens=2,
    )
    assert selected.tolist() == [0, 1, 4, 5]
    with pytest.raises(ValueError, match="multiple"):
        select_global_keep_block_indices(scores, keep_tokens=3, block_size=2)


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
    assert [kwargs["token_ids"][1:] for kwargs in forwards] == [[0], [0, 1], [4], [4, 5], [8], [8, 9]]
    assert all(kwargs["token_ids"][0] == 99 and kwargs["prefix"].logical_length == 10 for kwargs in forwards)
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
        if method == "prefix_cache_prune_batch_start":
            jobs, replay = args
            call.outcomes = [{"result": ModelRunner.prefix_cache_prune(
                runner, **job, policy="kvzip_global", kvzip_replay_prefix_ids=replay,
            )} for job in jobs]
            return
        assert method == "prefix_cache_prune_batch_step"
        return call.outcomes
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
        batches.append([0 if r['prefix'].blocks[0] is cache.blocks[0] else 20 for r in requests])
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
            validate_prefix_cache_prune_target=validate,
            validate_prefix_prune_keep_tokens=lambda _: None,
            select_prefix_prune_keep_indices=lambda scores, *, keep_tokens, **kwargs:
                select_global_keep_indices(scores, keep_tokens=keep_tokens),
            prefix_cache_prune=commit),
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
    # P1: a serving turn must not drain the whole reconstruction or leave
    # pinned blocks behind when shutdown interrupts a suspended task.
    batches.clear()
    commits.clear()
    runner._prefix_cache_prune_batch_iter = lambda *args: ModelRunner._prefix_cache_prune_batch_iter(runner, *args)
    runner.prefix_cache_prune_batch_abort = lambda: ModelRunner.prefix_cache_prune_batch_abort(runner)
    ModelRunner.prefix_cache_prune_batch_start(runner, jobs, [99])
    assert ModelRunner.prefix_cache_prune_batch_step(runner) is None
    assert len(batches) == 1 and not commits
    assert all(b.ref_count == 1 for b in cache.blocks)
    ModelRunner.prefix_cache_prune_batch_abort(runner)
    assert all(b.ref_count == 0 for b in cache.blocks)
    assert not commits
    batches.clear()
    commits.clear()
    def fail(requests):
        raise RuntimeError('scoring failed')
    runner._prefix_prune_score_forward_batch = fail
    with pytest.raises(RuntimeError, match='scoring failed'):
        ModelRunner.prefix_cache_prune_batch(runner, jobs, [99])
    assert not commits
    assert all(b.ref_count == 0 for b in cache.blocks)


@pytest.mark.parametrize("outcome", [{"result": {}}, {"error": "scoring failed"}])
def test_terminal_jobs_release_inputs_and_bound_history(outcome, monkeypatch):
    # P1: repeated maintenance must not retain every historical prompt.
    import sparseengine.engine.llm_engine as engine_module
    monkeypatch.setattr(engine_module, "PREFIX_PRUNE_HISTORY_LIMIT", 2)
    engine = SimpleNamespace(
        config=SimpleNamespace(sparse_method="", prefix_cache_block_size=1,
                               enable_prefix_caching=True),
        _prefix_prune_jobs={}, _pending_prefix_prune_ids=deque(),
        tokenizer=SimpleNamespace(encode=lambda *args, **kwargs: [99]),
    )
    def call(method, *args):
        if method == "prefix_cache_prune_batch_start":
            call.steps = 0
            return
        assert method == "prefix_cache_prune_batch_step"
        call.steps += 1
        return None if call.steps == 1 else [outcome]
    engine.model_runner = SimpleNamespace(call=call)
    ids = []
    for _ in range(4):
        queued = LLMEngine.prefix_cache_prune_start(
            engine, list(range(100)), ranges=[(0, 100)],
            keep_tokens=1, policy="kvzip_global",
        )
        prune_id = queued["prune_id"]
        ids.append(prune_id)
        assert LLMEngine.run_pending_prefix_prune(engine)
        assert LLMEngine.prefix_cache_prune_status(engine, prune_id)["status"] == "running"
        assert LLMEngine.run_pending_prefix_prune(engine)
        job = engine._prefix_prune_jobs[prune_id]
        assert job.status == ("failed" if "error" in outcome else "completed")
        assert not job.token_ids
        assert not LLMEngine.run_pending_prefix_prune(engine)
    assert list(engine._prefix_prune_jobs) == ids[-2:]


def test_partitioned_query_scores_preserve_mean_before_head_maximum(monkeypatch):
    # P0: reducing heads before accumulating query partitions changes the mask.
    from sparseengine.engine.cache_manager.standard import StandardCacheManager
    from sparseengine.engine.cache_manager.base import AttentionViewMeta, ExplicitKVPayload, PrefillComputeView
    import sparseengine.engine.cache_manager.prefix_prune_scoring as scoring
    torch.manual_seed(19)
    q = torch.randn(259, 4, 16)
    keys = torch.randn(20, 2, 16)
    slots = torch.randperm(20, dtype=torch.int32).reshape(2, 10)
    manager = object.__new__(StandardCacheManager)
    state = dict(score=None, physical_window=(6, 265, 1))
    manager._prefix_prune_scoring = dict(batch=[state])
    calls = []
    def oracle(queries, k, output, rows, b_start, lengths, cached, max_query,
               table, q_start, q_end, **kwargs):
        assert int(q_end[0] - q_start[0]) == len(queries)
        assert int(cached[0]) == int(q_start[0])
        lo, hi = kwargs["candidate_start"], int(lengths[0]) - kwargs["recent_keep_tokens"]
        selected = k[table[int(rows[0]), lo:hi].long()].repeat_interleave(2, dim=1)
        probabilities = torch.einsum("qhd,khd->hqk", queries, selected).div(4).softmax(-1)
        output[0, :, lo:hi] = probabilities.mean(1)
        calls.append(len(queries))
    monkeypatch.setattr(scoring, "prefill_score_fwd", oracle)
    view = PrefillComputeView(
        meta=AttentionViewMeta(active_slots=slots, req_indices=torch.tensor([1]),
                               context_lens=torch.tensor([265])),
        payload=ExplicitKVPayload(k_cache=keys, v_cache=keys),
    )
    manager.collect_prefill_attention_score(
        0, q, view, b_start_loc=torch.tensor([0]), chunk_lens=torch.tensor([259]),
    )
    selected = keys[slots[1, 1:6].long()].repeat_interleave(2, dim=1)
    expected = torch.einsum("qhd,khd->hqk", q, selected).div(4).softmax(-1).mean(1).amax(0)
    torch.testing.assert_close(state["score"][1:6], expected)
    assert sum(calls) == len(q) and len(calls) > 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("query_length", [3, 259])
def test_explicit_kv_batched_scores_keep_row_normalizers_and_layer_maxima(query_length):
    from sparseengine.engine.cache_manager.standard import StandardCacheManager
    from sparseengine.engine.cache_manager.base import AttentionViewMeta, ExplicitKVPayload, PrefillComputeView
    torch.manual_seed(27)
    q = torch.randn(query_length + 2, 4, 64, device='cuda', dtype=torch.float16)
    keys = torch.randn(540, 2, 64, device='cuda', dtype=torch.float16)
    slots = torch.randperm(540, device='cuda', dtype=torch.int32).reshape(2,270)
    rows = torch.tensor([1,0], device='cuda', dtype=torch.int32)
    lengths = torch.tensor([4 + query_length,9], device='cuda', dtype=torch.int32)
    manager = object.__new__(StandardCacheManager)
    states = [dict(score=None, physical_window=(4,4 + query_length,1)), dict(score=None, physical_window=(7,9,3))]
    manager._prefix_prune_scoring = dict(batch=states)
    view = PrefillComputeView(
        meta=AttentionViewMeta(active_slots=slots, req_indices=rows, context_lens=lengths),
        payload=ExplicitKVPayload(k_cache=keys, v_cache=keys),
    )
    expected = [torch.zeros(4 + query_length,device='cuda'), torch.zeros(9,device='cuda')]
    for query in (q, -q):
        manager.collect_prefill_attention_score(0, query, view,
            b_start_loc=torch.tensor([0,query_length], device='cuda', dtype=torch.int32),
            chunk_lens=torch.tensor([query_length,2], device='cuda', dtype=torch.int32))
        for i, (a,b,lo,hi) in enumerate([(0,query_length,1,4),(query_length,query_length+2,3,7)]):
            k = keys[slots[1-i,lo:hi].long()].float().repeat_interleave(2,dim=1)
            logits = torch.einsum('qhd,khd->hqk',query[a:b].float(),k) / 8
            score = logits.softmax(-1).mean(1).amax(0)
            expected[i][lo:hi] = torch.maximum(expected[i][lo:hi],score)
            torch.testing.assert_close(states[i]['score'],expected[i],atol=3e-4,rtol=.003)
