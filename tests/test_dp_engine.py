"""CPU-owned request routing contracts; no claims about CUDA numerics."""

from types import SimpleNamespace
from unittest.mock import Mock

from sparseengine.engine.chain_cache import RequestAdmission
from sparseengine.engine.dp_engine import DPAttentionEngine, DPAttentionPrefixSnapshot
from sparseengine.sampling_params import SamplingParams


def _frontend():
    engine = object.__new__(DPAttentionEngine)
    engine.config = SimpleNamespace(resolved_prefix_cache_mode="disabled")
    engine.ps = [None, None]
    engine._owners = {}
    engine._global_ids = {}
    engine._chain_owners = {}
    engine._chain_seq_ids = set()
    engine._chain_sequences = {}
    engine._next_id = 0
    engine._active = [set(), set()]
    engine._closed = True  # No actual processes to clean up in this test.
    return engine


def test_rank_local_sequence_ids_do_not_collide_and_finished_ids_are_released():
    engine = _frontend()
    engine._call = lambda ranks, method, *args: {
        rank: RequestAdmission(7, None, "disabled", 0) for rank in ranks
    }
    first = engine.add_request([1, 2], SamplingParams())
    second = engine.add_request([3, 4], SamplingParams())
    assert first != second
    assert engine._owners[first] == (0, 7)
    assert engine._owners[second] == (1, 7)
    engine._call = Mock(
        return_value={
            rank: (([(7, [rank + 10], [], [])], -1), [(7, [rank + 10])], [], [], True)
            for rank in range(2)
        }
    )
    finished, _ = engine.step()
    assert {row[0] for row in finished} == {first, second}
    assert engine.is_finished()
    assert not engine._owners and not engine._global_ids
    assert tuple(engine._call.call_args.args[0]) == (0, 1)


def test_chain_resume_keeps_original_owner_despite_changed_load():
    engine = _frontend()
    calls = []

    def admit(ranks, method, *args):
        calls.append(tuple(ranks))
        return {rank: RequestAdmission(9, "chain-a", "idle", 12) for rank in ranks}

    engine._call = admit
    initial = engine.admit_request([1, 2], SamplingParams())
    engine._active[0].clear()
    engine._release_request_identity(initial.seq_id)
    engine._active[0].update({100, 101})
    resumed = engine.admit_request(
        [3], SamplingParams(), chain_id="chain-a", chain_append_only=True
    )
    assert resumed.seq_id == initial.seq_id
    assert calls == [(0,), (0,)]


def test_prefix_snapshot_reports_a_single_replica_hit_not_sum_of_partial_hits():
    replicas = []
    for count in (16, 32):
        replicas.append(
            SimpleNamespace(match=Mock(return_value={"matched_tokens": count}))
        )
    snapshot = DPAttentionPrefixSnapshot(replicas)
    assert snapshot.match([1, 2])["matched_tokens"] == 32


def test_control_failure_drains_other_rank_responses_before_next_rpc():
    from multiprocessing import Pipe

    import pytest

    engine = _frontend()
    pairs = [Pipe(), Pipe()]
    engine._connections = [pair[0] for pair in pairs]
    engine.ps = [SimpleNamespace(is_alive=lambda: True)] * 2
    engine._terminate = Mock()
    try:
        pairs[0][1].send(
            (False, (ValueError("invalid cache prefix"), "test failure"), None)
        )
        pairs[1][1].send((True, {"ok": True}, None))
        with pytest.raises(ValueError, match="invalid cache prefix"):
            engine._receive((0, 1))
        assert not engine._connections[1].poll()
        engine._terminate.assert_not_called()
    finally:
        for pair in pairs:
            for connection in pair:
                connection.close()


def test_discard_is_idempotent_but_rejects_a_different_resident_sequence():
    import pytest

    from sparseengine.engine.chain_cache import ChainOwnerMismatchError

    engine = _frontend()
    assert engine.discard_chain("gone", expected_seq_id=10) is False
    engine._chain_sequences["resident"] = 11
    with pytest.raises(ChainOwnerMismatchError):
        engine.discard_chain("resident", expected_seq_id=10)


def test_dp_entrypoint_rejects_unknown_configuration_before_loading_model():
    import pytest

    from sparseengine import LLM

    with pytest.raises(ValueError, match="Unknown SparseEngine config keys"):
        LLM(
            "unused-model",
            data_parallel_size=2,
            expert_parallel_size=2,
            unknown_option=True,
        )


def test_dp_rejects_uncaptured_sampling_plan_at_configuration_boundary():
    import pytest

    from sparseengine.configs.cuda_graph import normalize_decode_cuda_graph

    config = SimpleNamespace(
        data_parallel_size=2,
        decode_graph=True,
        decode_graph_startup_capture=None,
        decode_graph_startup_capture_limit=None,
        decode_graph_capture_sampling=True,
        sparse_method="",
    )
    with pytest.raises(ValueError, match="DP attention does not support"):
        normalize_decode_cuda_graph(config)


def test_generate_rejects_sampling_parameter_length_mismatch_before_submission():
    # Scalar broadcast coverage does not catch zip() silently dropping prompts
    # when an explicit SamplingParams list is short.
    import pytest

    engine = _frontend()
    engine.add_request = Mock()

    with pytest.raises(ValueError, match="must have the same length"):
        engine.generate(
            [[1, 2], [3, 4]],
            [SamplingParams(max_tokens=1)],
            use_tqdm=False,
        )

    engine.add_request.assert_not_called()
