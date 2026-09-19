"""Preserve ordering, error reporting and sampling outputs while batching work."""
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sparseengine.engine.scheduler import Scheduler
from sparseengine.engine.sequence import Sequence
from sparseengine.engine.model_runner import ModelRunner
from sparseengine.engine.cache_manager.methods.deltakv_less_memory import DeltaKVLessMemoryCacheManager
from sparseengine.layers.sampler import Sampler
from sparseengine.sampling_params import SamplingParams
from test_prefill_schedule_policy import make_scheduler


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    return request.param


def request(n=1):
    return Sequence([1] * n, SamplingParams(temperature=0, ignore_eos=True))


def test_prefill_replay_checks_scale_with_queue_not_batch(monkeypatch):
    scheduler = make_scheduler('all_chunked', max_tokens=128)
    scheduler.max_num_seqs_in_batch = 64
    seqs = [request() for _ in range(1000)]
    scheduler.waiting.extend(seqs)
    reads = 0
    original = Sequence.is_recompute_replay.fget
    def counted(seq):
        nonlocal reads
        reads += 1
        return original(seq)
    monkeypatch.setattr(Sequence, 'is_recompute_replay', property(counted))
    batch, prefill, _ = scheduler.schedule()
    assert prefill and batch == seqs[:64]
    assert list(scheduler.waiting) == seqs[64:]
    assert reads < 10 * len(seqs)


def test_decode_mixed_lengths_preserve_queue_order():
    scheduler = make_scheduler('all_chunked')
    seqs = [request(n) for n in (20, 1, 6, 1000)]
    scheduler.max_decoding_seqs = 3
    scheduler.decoding = deque(seqs)
    batch, prefill, _ = scheduler.schedule()
    assert not prefill and batch == seqs[:3]
    assert list(scheduler.decoding) == seqs


def test_prefill_skipped_requests_survive_admission_exception(monkeypatch):
    scheduler = make_scheduler('all_chunked')
    skipped, selected = request(), request()
    scheduler.waiting = deque([skipped, selected])
    monkeypatch.setattr(scheduler, '_prefill_mode_order', lambda: [('chunked', None)])
    monkeypatch.setattr(scheduler, '_prefill_batch_key', lambda seq: ('other' if seq is skipped else 'chunked', None))
    def fail(*args):
        raise RuntimeError('admission failed')
    monkeypatch.setattr(scheduler.memory_oracle, 'prompt_admission_costs', fail)
    with pytest.raises(RuntimeError, match='admission failed'):
        scheduler.schedule()
    assert skipped in scheduler.waiting


def test_device_sampling_and_batched_logprobs_match_reference(device):
    runner = object.__new__(ModelRunner)
    runner.sampler = Sampler()
    seqs = [SimpleNamespace(temperature=0., logprobs=n, should_publish_sample=live)
            for n, live in [(None, False), (0, True), (1, True), (7, True)]]
    logits = torch.tensor([[2., 1., -1.], [1., 2., 3.], [-1., 4., 0.], [2., 4., 3.]], device=device)
    tokens = runner._sample_model_outputs(logits, seqs, return_device_tokens=True)
    assert tokens.tolist() == [0, 2, 1, 1]
    transfers = []
    original = torch.Tensor.cpu
    def counted(tensor, *args, **kwargs):
        transfers.append(tuple(tensor.shape))
        return original(tensor, *args, **kwargs)
    with patch.object(torch.Tensor, 'cpu', counted):
        sampled, top = runner._collect_logprobs(logits, tokens, seqs)
    assert len(transfers) == 3
    reference = logits.double().log_softmax(-1)
    for row, token in enumerate(tokens.tolist()):
        assert sampled[row] == pytest.approx(reference[row, token].item(), abs=1e-6)
        if top[row] is not None:
            expected = reference[row].topk(min(seqs[row].logprobs, 3))
            assert top[row] == pytest.approx(dict(zip(expected.indices.tolist(), expected.values.tolist())), abs=1e-6)
    assert runner._mask_recompute_logprobs(seqs, (sampled, top))[0][0] is None


def test_sampling_buffers_reuse_storage_and_refresh_changed_rows(device):
    runner = object.__new__(ModelRunner)
    runner.device = torch.device(device)
    runner.platform = SimpleNamespace(supports_pin_memory=lambda: True)
    first = [SimpleNamespace(temperature=.7, top_p=.9, top_k=3)] * 3
    tensors = runner.prepare_sample(first)
    ptrs = [t.data_ptr() for t in tensors]
    second = [SimpleNamespace(temperature=.2, top_p=1., top_k=0)]
    actual = runner.prepare_sample(second)
    assert [t.data_ptr() for t in actual] == ptrs
    for value, expected in zip(actual, [.2, 1., 0]):
        assert value.tolist() == pytest.approx([expected])


def test_top_k_only_distribution_and_mixed_greedy_without_full_sort(device):
    torch.manual_seed(71)
    runner = object.__new__(ModelRunner)
    runner.device = torch.device(device)
    runner.platform = SimpleNamespace(supports_pin_memory=lambda: True)
    runner.sampler = Sampler()
    seqs = [SimpleNamespace(temperature=.7, top_p=1., top_k=2)] * 10000
    seqs += [SimpleNamespace(temperature=0., top_p=.1, top_k=0)]
    logits = torch.tensor([0., 3., 2., -1.], device=device).repeat(len(seqs), 1)
    with patch('torch.sort', side_effect=AssertionError('full sort')):
        tokens = runner._sample_model_outputs(logits, seqs, return_device_tokens=True)
    assert tokens[-1] == 1
    assert set(tokens[:-1].tolist()) == {1, 2}
    frequency = (tokens[:-1] == 1).float().mean().item()
    assert frequency == pytest.approx(torch.tensor([3., 2.]).div(.7).softmax(0)[0].item(), abs=.02)


@pytest.mark.parametrize('slots,error', [([0, 2], None), ([-1], 'out of range'), ([3], 'out of range'), ([1], 'without live positions')])
def test_center_validation_preserves_address_and_liveness_errors(slots, error, device):
    manager = object.__new__(DeltaKVLessMemoryCacheManager)
    manager.deltakv_slot_to_pos = torch.tensor([0, -1, 8], device=device)
    manager._describe_deltakv_full_slots_for_debug = lambda *args, **kwargs: 'debug'
    def validate():
        manager._validate_live_deltakv_center_slots(torch.tensor(slots, device=device), row_idx=0,
            total_len=9, compressed_len=0, evict_start=1, evict_end=8, label='test')
    if error:
        with pytest.raises(RuntimeError, match=error):
            validate()
    else:
        validate()


def test_prefill_incompatible_prefix_is_scanned_once_and_stays_ordered(monkeypatch):
    scheduler = make_scheduler('all_chunked', max_tokens=128)
    scheduler.max_num_seqs_in_batch = 64
    incompatible = [request(2) for _ in range(1000)]
    eligible = [request() for _ in range(65)]
    scheduler.waiting = deque(incompatible + eligible)
    reads = 0
    def key(seq):
        nonlocal reads
        reads += 1
        return ('chunked', seq.num_prompt_tokens)
    # Exercise a later compatibility bucket after an earlier bucket was deferred.
    monkeypatch.setattr(scheduler, '_prefill_mode_order', lambda: [('chunked', 1)])
    monkeypatch.setattr(scheduler, '_prefill_batch_key', key)
    batch, prefill, _ = scheduler.schedule()
    assert prefill and batch == eligible[:64]
    assert list(scheduler.waiting) == incompatible + eligible[64:]
    assert reads <= len(incompatible) + len(eligible)


def test_decode_preemption_preserves_other_requests(monkeypatch):
    scheduler = make_scheduler('all_chunked')
    long, victim = request(20), request()
    scheduler.decoding = deque([victim, long])
    monkeypatch.setattr(scheduler.memory_oracle, 'decode_step_free_slots', lambda: 0)
    def preempt(seq, scheduled, preempted, **kwargs):
        assert seq is victim
        assert list(scheduler.decoding) == [long]
        return [], False, [seq]
    monkeypatch.setattr(scheduler, '_preempt_decode_victim', preempt)
    assert scheduler.schedule() == ([], False, [victim])
    assert list(scheduler.decoding) == [long]
