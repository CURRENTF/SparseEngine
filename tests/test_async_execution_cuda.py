"""Real graph replay and DMA ownership against independent integer arithmetic."""
from types import SimpleNamespace

import pytest
import torch

from sparseengine.engine.async_scheduling.execution import AsyncDrainRequired, AsyncExecution, DeviceLogprobs
from sparseengine.engine.sequence import Sequence
from sparseengine.sampling_params import SamplingParams


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_device_feedback_survives_reorder_graph_output_reuse_and_async_copy():
    device = torch.device('cuda:0')
    source = torch.zeros(2, dtype=torch.long, device=device)
    capture = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            target = (source * 17 + 3) % 101
    stream.synchronize()
    with torch.cuda.graph(capture):
        target = (source * 17 + 3) % 101
    runner = SimpleNamespace(
        device=device, cache_manager=SimpleNamespace(),
        decode_graph_runner=SimpleNamespace(),
        parallel_context=SimpleNamespace(attn_tp_rank=0, attn_tp_size=1),
        _post_sparse_forward=lambda *_: None,
    )
    execution = AsyncExecution(runner)
    def run(seqs, is_prefill):
        assert not is_prefill
        execution.prepare_inputs(source, seqs)
        capture.replay()
        return target, DeviceLogprobs(target.float())
    runner.run = run
    seqs = [Sequence([7], SamplingParams()), Sequence([11], SamplingParams())]
    expected = {s.seq_id: s.last_token for s in seqs}
    answers = []
    for step in range(12):
        order = seqs if step % 2 else list(reversed(seqs))
        execution.submit(step, order, False)
        expected = {sid: (token*17+3) % 101 for sid, token in expected.items()}
        answers.append([expected[s.seq_id] for s in order])
        if step:
            tokens, (logs, _) = execution.collect(step-1)
            assert tokens == answers[step-1]
            assert logs == [float(x) for x in tokens]
    assert execution.collect(11)[0] == answers[-1]
    assert execution.submitted == execution.completed == 12
    assert not execution.results


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_private_host_metadata_is_not_overwritten_by_queued_steps():
    device = torch.device('cuda:0')
    runner = SimpleNamespace(device=device, cache_manager=SimpleNamespace(),
                             decode_graph_runner=SimpleNamespace(),
                             parallel_context=SimpleNamespace(attn_tp_rank=0, attn_tp_size=1))
    execution = AsyncExecution(runner)
    template = torch.empty(1, dtype=torch.long, pin_memory=True)
    uploaded = torch.empty(1, dtype=torch.long, device=device)
    seq = Sequence([1], SamplingParams())
    def run(seqs, prefill):
        host = execution.acquire_host_tensor(template)
        host.fill_(seq.num_tokens)
        uploaded.copy_(host, non_blocking=True)
        return uploaded * 3, None
    runner.run = run
    for ticket in range(8):
        seq.num_tokens = ticket + 1
        execution.submit(ticket, [seq], False)
    for ticket in range(8):
        assert execution.collect(ticket)[0] == [3 * (ticket + 1)]
    assert not execution.results
    assert execution.output_pool.live_bytes == 0


def _execution():
    runner = SimpleNamespace(
        device=torch.device('cuda:0'), cache_manager=SimpleNamespace(),
        decode_graph_runner=SimpleNamespace(),
        parallel_context=SimpleNamespace(attn_tp_rank=0, attn_tp_size=1),
    )
    return runner, AsyncExecution(runner)


def _penalty_reference(logits, seqs, generated):
    expected = logits.float().cpu().clone()
    for row, seq in enumerate(seqs):
        repeated = set(seq.prompt_token_ids) | set(generated[seq.seq_id])
        for token in repeated:
            value = expected[row, token]
            expected[row, token] = value / seq.repetition_penalty if value > 0 else value * seq.repetition_penalty
        for token in set(generated[seq.seq_id]):
            expected[row, token] -= seq.presence_penalty
    return expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_batched_penalties_follow_unretired_tokens_and_request_identity(dtype):
    """A CPU token-set oracle catches stale state, mixed penalties and row reuse."""
    runner, execution = _execution()
    seqs = [Sequence([1, 3, 3], SamplingParams(repetition_penalty=1.3, presence_penalty=.7)),
            Sequence([2, 4], SamplingParams()),
            Sequence([3, 5], SamplingParams(repetition_penalty=.8, presence_penalty=-.4))]
    seqs[0].append_token(2)
    generated = {s.seq_id: list(s.completion_token_ids) for s in seqs}
    source = torch.arange(3 * 19 * 2, device=runner.device).view(3, 38).to(dtype) / 10 - 4
    logits = source[:, ::2]  # Real strides, odd vocabulary, and both signs.
    checked = []
    def run(order, prefill):
        inputs = torch.empty(5, dtype=torch.long, device=runner.device)
        execution.prepare_inputs(inputs, order)
        result = execution.apply_penalties(logits, order)
        checked.append(result)
        return result.argmax(-1), DeviceLogprobs(result[:, 0], result[:, :3], result[:, :3].long())
    runner.run = run
    for step in range(5):
        order = seqs if step % 2 == 0 else list(reversed(seqs))
        reference = _penalty_reference(logits, order, generated)
        execution.submit(step, order, False)
        torch.testing.assert_close(checked[-1].cpu(), reference, rtol=1e-6, atol=1e-6)
        for seq, token in zip(order, reference.argmax(-1).tolist()):
            generated[seq.seq_id].append(token)
    with pytest.raises(AsyncDrainRequired):
        execution.forget([seqs[0].seq_id])
    for step in range(5):
        execution.collect(step)
    old_slot = execution._penalty_rows[seqs[0].seq_id]
    execution.forget([seqs[0].seq_id])
    fresh = Sequence([7], SamplingParams(presence_penalty=.2, repetition_penalty=1.7))
    generated[fresh.seq_id] = []
    seqs[0] = fresh
    execution.submit(5, seqs, False)
    torch.testing.assert_close(checked[-1].cpu(), _penalty_reference(logits, seqs, generated), rtol=1e-6, atol=1e-6)
    assert execution._penalty_rows[fresh.seq_id] == old_slot
    execution.collect(5)
    assert execution.output_pool.live_bytes == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_output_leases_survive_graph_reuse_discard_and_shape_churn():
    """All output fields stay private until collect; cached bytes remain bounded."""
    runner, execution = _execution()
    seqs = [Sequence([i + 1], SamplingParams(logprobs=i % 3 + 1)) for i in range(9)]
    counter = 0
    def run(order, prefill):
        nonlocal counter
        counter += 1
        tokens = torch.full((len(order),), counter, dtype=torch.long, device=runner.device)
        values = torch.full((len(order), 3), float(counter), device=runner.device)
        return tokens, DeviceLogprobs(values[:, 0], values, values.long())
    runner.run = run
    expected = []
    leases = []
    for step in range(18):
        order = seqs[:step % len(seqs) + 1]
        execution.submit(step, order, False)
        leases.extend(execution.results[step].output_buffers)
        expected.append((len(order), step + 1))
        if step >= 3:
            count, value = expected[step - 3]
            tokens, (logs, tops) = execution.collect(step - 3, discarded=[seqs[0].seq_id])
            assert tokens == [value] * count
            assert logs == [float(value)] * count
            assert all(set(top.values()) == {float(value)} for top in tops)
    for step in range(15, 18):
        count, value = expected[step]
        assert execution.collect(step)[0] == [value] * count
    assert len({id(buffer) for buffer in leases}) < len(leases)
    assert execution.output_pool.live_bytes == 0
    assert execution.output_pool.cached_bytes <= execution.output_pool.peak_live_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_partial_prefill_does_not_publish_feedback_and_recovery_reseeds_history():
    runner, execution = _execution()
    seq = Sequence([7, 11, 13], SamplingParams(repetition_penalty=1.2))
    seq.current_chunk_size = 1
    seen = []
    def run(order, prefill):
        source = torch.empty(4, dtype=torch.long, device=runner.device)
        execution.prepare_inputs(source, order)
        seen.append(source.clone())
        return source[:1] + 1, None
    runner.run = run
    execution.submit(0, [seq], True)
    execution.collect(0)
    assert seq.seq_id not in execution._feedback_ready
    execution.submit(1, [seq], False)
    assert seen[-1].tolist() == [seq.decode_input_token] * 4
    token = execution.collect(1)[0][0]
    seq.append_token(token)
    execution.prepare_synchronous_execution()
    execution.submit(2, [seq], False)
    assert seen[-1].tolist() == [token] * 4
    execution.collect(2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_state_tables_grow_with_prior_tickets_in_flight():
    runner, execution = _execution()
    seqs = [Sequence([i + 1, 13], SamplingParams(repetition_penalty=1.3, presence_penalty=.7))
            for i in range(9)]
    generated = {s.seq_id: [] for s in seqs}
    expected = []
    torch.manual_seed(31)
    logits = torch.randn(9, 257, device=runner.device)
    def run(order, prefill):
        source = torch.empty(max(16, len(order)), dtype=torch.long, device=runner.device)
        execution.prepare_inputs(source, order)
        return execution.apply_penalties(logits[:len(order)], order).argmax(-1), None
    runner.run = run
    for ticket, count in enumerate([1, 2, 4, 9, 3, 7, 2, 9]):
        order = list(reversed(seqs[:count]))
        reference = _penalty_reference(logits[:count], order, generated)
        tokens = reference.argmax(-1).tolist()
        expected.append(tokens)
        execution.submit(ticket, order, False)
        for seq, token in zip(order, tokens):
            generated[seq.seq_id].append(token)
        if ticket >= 2:
            assert execution.collect(ticket - 2)[0] == expected[ticket - 2]
    assert execution.collect(6)[0] == expected[6]
    assert execution.collect(7)[0] == expected[7]
    # Reuse holes, then grow: live request count must not alias occupied slots.
    retired = seqs[1::3]
    execution.forget([s.seq_id for s in retired])
    fresh = [Sequence([i + 20], SamplingParams(repetition_penalty=1.3, presence_penalty=.7))
             for i in range(12)]
    order = [s for s in seqs if s not in retired] + fresh
    generated.update((s.seq_id, []) for s in fresh)
    logits = torch.randn(len(order), 257, device=runner.device)
    reference = _penalty_reference(logits, order, generated)
    execution.submit(8, order, False)
    assert execution.collect(8)[0] == reference.argmax(-1).tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_deferred_prefix_records_keep_each_steps_input_tokens():
    runner, execution = _execution()
    seq = Sequence([7], SamplingParams())
    recorded = []
    runner.cache_manager._record_frozen_prefix_materialization = lambda seq, tokens, slots: recorded.append(tokens)
    runner.cache_manager.publish_pending_prefix_blocks = lambda seqs: None
    def run(order, prefill):
        source = torch.empty(1, dtype=torch.long, device=runner.device)
        execution.prepare_inputs(source, order)
        runner.cache_manager._async_prefix_records.append((seq, [], None))
        return source + 1, None
    runner.run = run
    for ticket in range(5):
        execution.submit(ticket, [seq], False)
    for ticket in range(5):
        assert execution.collect(ticket)[0] == [8 + ticket]
    assert recorded == [[7 + ticket] for ticket in range(5)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_flashinfer_plan_host_storage_waits_only_for_its_upload():
    """A delayed real DMA exposes rewriting persistent pinned planning inputs."""
    from sparseengine.operators.flashinfer_decode_state import FlashInferPagedDecodeGraphState
    state = object.__new__(FlashInferPagedDecodeGraphState)
    host_lengths = torch.empty(2, dtype=torch.int32, pin_memory=True)
    state.inputs = SimpleNamespace(host=SimpleNamespace(context_lens=host_lengths),
                                   context_lens=torch.empty(2, dtype=torch.int32, device='cuda'))
    state.contract = SimpleNamespace(context_capacity=128)
    state.spec = SimpleNamespace(page_size=4, num_query_heads=4, num_kv_heads=1,
                                 head_dim=128, softmax_scale=.1, activation_dtype=torch.bfloat16)
    state.host_indptr = torch.empty(3, dtype=torch.int32, pin_memory=True)
    state.host_last_page_len = torch.empty(2, dtype=torch.int32, pin_memory=True)
    state.indices = torch.empty(64, dtype=torch.int32, device='cuda')
    state.sparse_wrapper = None
    state._plan_upload_done = None
    uploads = []
    def plan(indptr, indices, last, **kwargs):
        torch.cuda._sleep(2_000_000)
        uploads.append((indptr.to('cuda', non_blocking=True), last.to('cuda', non_blocking=True)))
    state.wrapper = SimpleNamespace(plan=plan)
    lengths = [[3, 5], [7, 8], [13, 1]]
    for values in lengths:
        host_lengths.copy_(torch.tensor(values, dtype=torch.int32))
        state.prepare_out_graph()
    for values, (indptr, last) in zip(lengths, uploads):
        pages = [(n + 3) // 4 for n in values]
        assert indptr.cpu().tolist() == [0, pages[0], sum(pages)]
        assert last.cpu().tolist() == [n - (p - 1) * 4 for n, p in zip(values, pages)]
