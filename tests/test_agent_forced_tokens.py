"""Forced replay tokens stay on the sampling rank and out of TP transport."""
import asyncio
import pickle
from types import SimpleNamespace

import pytest
import torch

from sparseengine.engine.async_scheduling.scheduler import execution_snapshot
from sparseengine.engine.model_runner import ModelRunner
from sparseengine.engine.sequence import Sequence
from sparseengine.sampling_params import SamplingParams


def test_forced_replay_cursor_stays_on_rank_zero_snapshot():
    seq = Sequence([7, 8], SamplingParams(temperature=0, max_tokens=3, ignore_eos=True,
                                           benchmark_forced_token_ids=(11, 12, 13)))
    seq.num_pending_outputs = 1
    snapshot = execution_snapshot(seq)
    transmitted = pickle.loads(pickle.dumps(snapshot))
    assert snapshot.benchmark_forced_token_ids == (11, 12, 13)
    assert transmitted.benchmark_forced_token_ids is None
    assert transmitted.num_completion_tokens == 1

    runner = SimpleNamespace(sampler=lambda *_args, **_kwargs: torch.tensor([99]))
    sampled = ModelRunner._sample_model_outputs(
        runner, torch.zeros((1, 128)), [snapshot], return_device_tokens=True
    )
    assert sampled.tolist() == [12]


def test_forced_replay_tokens_do_not_grow_tp_payload():
    seqs = [Sequence([7, 8], SamplingParams(
        max_tokens=4500, ignore_eos=True,
        benchmark_forced_token_ids=tuple(range(4500)),
    )) for _ in range(80)]
    payload = pickle.dumps(["run", [execution_snapshot(seq) for seq in seqs], False])
    assert len(payload) < 1 << 20


def test_forced_replay_rejects_wrong_length_and_out_of_vocab():
    with pytest.raises(ValueError, match="exactly max_tokens"):
        SamplingParams(max_tokens=2, ignore_eos=True, benchmark_forced_token_ids=(1,))
    with pytest.raises(ValueError, match="requires ignore_eos"):
        SamplingParams(max_tokens=1, benchmark_forced_token_ids=(1,))
    seq = Sequence([7], SamplingParams(temperature=0, max_tokens=1, ignore_eos=True,
                                       benchmark_forced_token_ids=(129,)))
    runner = SimpleNamespace(sampler=lambda *_args, **_kwargs: torch.tensor([1]))
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        ModelRunner._sample_model_outputs(runner, torch.zeros((1, 128)), [seq])


@pytest.mark.parametrize(
    ("forced", "max_tokens"),
    [([11], 2), ([128], 1)],
)
def test_openai_rejects_bad_forced_tokens_before_submission(forced, max_tokens):
    from fastapi import HTTPException
    from sparseengine.entrypoints.openai.protocol.chat import ChatCompletionRequest
    from sparseengine.entrypoints.openai.serving.chat import serve_chat_completion

    class Dispatcher:
        engine = SimpleNamespace(config=SimpleNamespace(hf_config=SimpleNamespace(vocab_size=128)))

        async def submit(self, *_args, **_kwargs):
            raise AssertionError("invalid request reached engine admission")

    request = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "prompt"}],
        max_tokens=max_tokens, ignore_eos=True,
        benchmark_forced_token_ids=forced,
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(serve_chat_completion(request, Dispatcher(), None, "m", None))
    assert error.value.status_code == 400


def test_engine_admission_rejects_out_of_vocab_forced_token():
    from sparseengine.engine.llm_engine import LLMEngine

    engine = object.__new__(LLMEngine)
    engine.config = SimpleNamespace(hf_config=SimpleNamespace(vocab_size=128))
    params = SamplingParams(max_tokens=1, ignore_eos=True,
                            benchmark_forced_token_ids=(128,))
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        engine.admit_request([7], params)
