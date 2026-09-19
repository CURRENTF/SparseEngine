import hashlib
from types import SimpleNamespace

import pytest
import torch

from sparseengine.engine.decode_cuda_graph import DecodeCudaGraphKey, DecodeCudaGraphState

from scripts.debug.compare_decode_graph_eager_logits import (
    _build_method_trigger_evidence,
    _compare_logits,
    _canonicalize_decode_logits,
    _generated_token_ids,
    _has_physical_compaction,
    _graph_runtime_summary,
    _save_full_logits_artifact,
    _start_graph_measurement,
    _validate_eager_runtime,
    _validate_graph_runtime,
    _validate_method_trigger,
)


def test_mixed_length_trace_compares_physical_rows_to_their_own_request():
    # A short dense row must not count as eviction merely because another
    # request in the same decode batch is longer.
    trace = {
        "logical_context_len": 129,
        "logical_context_lens": {"7": 3, "9": 129},
        "cache": {"live_rows": {"0": [
            {"seq_id": 7, "row_len": 3}, {"seq_id": 9, "row_len": 129},
        ]}},
    }
    assert not _has_physical_compaction([trace])
    trace["cache"]["live_rows"]["0"][1]["row_len"] = 44
    assert _has_physical_compaction([trace])


def test_logit_comparison_aligns_request_identity_across_batch_schedules():
    def event(requests):
        return {
            "round": 0, "stage": "decode",
            "token_outputs": [
                {"request_idx": request, "token_ids": [token]}
                for request, token in requests
            ],
        }

    eager = {"generated_token_outputs": [
        event([(0, 10)]), event([(0, 11)]),
        event([(1, 20)]), event([(1, 21)]),
    ]}
    graph = {"generated_token_outputs": [
        event([(1, 20), (0, 10)]), event([(1, 21), (0, 11)]),
    ]}
    expected = torch.tensor([[10.0], [11.0], [20.0], [21.0]])
    actual = torch.tensor([[20.0], [10.0], [21.0], [11.0]])
    torch.testing.assert_close(
        _canonicalize_decode_logits(expected, eager),
        _canonicalize_decode_logits(actual, graph),
    )
    assert _generated_token_ids(eager) == _generated_token_ids(graph)
    with pytest.raises(RuntimeError, match="rows do not match"):
        _canonicalize_decode_logits(actual[:2], graph)


def _trace(*, row_len=4, logical_context_len=8, h2o=None, omni=False, rkv=False):
    layer = {}
    if omni:
        layer = {
            "active_slots": {"numel": 4},
            "context_lens": {"max": 4},
        }
    cache = {
        "live_rows": {
            "0": [{"row_len": row_len}],
        }
    }
    if h2o is not None:
        cache["h2o"] = h2o
    return {
        "logical_context_len": logical_context_len,
        "layers": {"1": layer} if layer else {},
        "cache": cache,
        "rkv_materializer_layers": [0, 1] if rkv else [],
    }


def test_graph_measurement_preserves_warmup_graph_pool_ownership():
    warmup_graph = object()

    class Runner:
        def __init__(self):
            self._graphs = {"warmup": SimpleNamespace(graph=warmup_graph)}
            self.capture_count = 1
            self.replay_count = 1
            self.eager_static_count = 0
            self.force_eager_count = 0
            self.clear_calls = 0

        def clear_captured_graphs(self):
            self.clear_calls += 1
            self._graphs.clear()

    runner = Runner()
    llm = SimpleNamespace(
        model_runner=SimpleNamespace(decode_graph_runner=runner)
    )

    baseline = _start_graph_measurement(llm)

    assert runner.clear_calls == 0
    assert runner._graphs["warmup"].graph is warmup_graph
    assert baseline == {
        "capture_count": 1,
        "replay_count": 1,
        "eager_static_count": 0,
        "force_eager_count": 0,
        "graph_count": 1,
    }


def test_graph_summary_reads_capacity_from_capture_state_not_batch_only_key():
    state = DecodeCudaGraphState(
        key=DecodeCudaGraphKey("h2o", 2, False, "unified"),
        capture_context_capacity=129,
        graph=object(),
    )
    runner = SimpleNamespace(
        _graphs={state.key: state}, method="h2o", capture_count=1,
        replay_count=3, eager_static_count=0, force_eager_count=0,
    )
    llm = SimpleNamespace(
        config=SimpleNamespace(decode_graph=True, model="test", sparse_method="h2o", hf_config=SimpleNamespace()),
        model_runner=SimpleNamespace(decode_graph_runner=runner),
    )
    summary = _graph_runtime_summary(
        llm, use_graph=True,
        counters_before={"capture_count": 1, "replay_count": 0, "eager_static_count": 0, "force_eager_count": 0},
    )
    assert summary["graph_keys"][0]["context_capacity"] == state.capture_context_capacity
    assert summary["graph_keys"][0]["graph_path_id"] == state.key.graph_path_id
    _validate_graph_runtime(summary)


@pytest.mark.parametrize(
    ("method", "trace", "calls"),
    [
        ("vanilla", _trace(row_len=8), {}),
        (
            "streamingllm",
            _trace(),
            {"cache.free_prefix_recent_slots_batch_layers": 1},
        ),
        ("snapkv", _trace(), {"cache.free_part_slots_batch_layers": 1}),
        (
            "h2o",
            _trace(
                h2o={
                    "counters": {
                        "intermediate_prefill_evictions": 0,
                        "final_prefill_evictions": 1,
                        "decode_evictions": 2,
                        "dropped_tokens": 3,
                    },
                    "ring_counters": {"fast_rows": 2, "fallback_rows": 0},
                }
            ),
            {"cache.evict_after_decode": 2},
        ),
        (
            "omnikv",
            _trace(row_len=8, omni=True),
            {"controller._update_dynamic_omnikv_indices": 2},
        ),
        (
            "rkv",
            _trace(rkv=True),
            {
                "cache.rkv_joint_scores": 1,
                "cache.materialize_attention_keys": 2,
                "cache.free_part_slots_batch_layers": 1,
            },
        ),
    ],
)
def test_glm_graph_method_trigger_evidence_is_machine_checkable(
    method,
    trace,
    calls,
):
    evidence = _build_method_trigger_evidence(method, [trace], calls)

    assert evidence["triggered"] is True
    _validate_method_trigger(evidence)


def test_sparse_method_trigger_gate_rejects_unexercised_path():
    evidence = _build_method_trigger_evidence("snapkv", [_trace(row_len=8)], {})

    with pytest.raises(RuntimeError, match="trigger gate failed"):
        _validate_method_trigger(evidence)


def test_rkv_explicit_keys_do_not_require_an_mla_materializer_registration():
    trace = {**_trace(), "attention_cache_layout": "explicit_kv"}
    calls = {"cache.rkv_joint_scores": 1, "cache.materialize_attention_keys": 1}
    _validate_method_trigger(_build_method_trigger_evidence("rkv", [trace], calls))
    trace["attention_cache_layout"] = "mla_latent"
    with pytest.raises(RuntimeError, match="trigger gate failed"):
        _validate_method_trigger(_build_method_trigger_evidence("rkv", [trace], calls))


def test_graph_runtime_gate_requires_capture_replay_and_zero_fallback():
    valid = {
        "config_enabled": True,
        "graph_active": True,
        "graph_count": 1,
        "capture_count": 1,
        "replay_count": 3,
        "eager_static_count": 0,
        "force_eager_count": 0,
        "counter_delta": {
            "capture_count": 0,
            "replay_count": 2,
            "eager_static_count": 0,
            "force_eager_count": 0,
        },
        "fallback": False,
    }
    _validate_graph_runtime(valid)

    with pytest.raises(RuntimeError, match="forced eager"):
        _validate_graph_runtime(
            {
                **valid,
                "counter_delta": {
                    **valid["counter_delta"],
                    "force_eager_count": 1,
                },
            }
        )


def test_eager_runtime_gate_rejects_a_captured_graph():
    valid = {
        "config_enabled": False,
        "graph_active": False,
        "graph_count": 0,
        "capture_count": 0,
        "replay_count": 0,
        "eager_static_count": 3,
        "force_eager_count": 0,
        "counter_delta": {
            "capture_count": 0,
            "replay_count": 0,
            "eager_static_count": 2,
            "force_eager_count": 0,
        },
    }
    _validate_eager_runtime(valid)

    with pytest.raises(RuntimeError, match="retained a captured"):
        _validate_eager_runtime(
            {**valid, "graph_active": True, "graph_count": 1}
        )


def test_omnikv_graph_replay_uses_tensor_selection_evidence():
    trace = _trace(row_len=8, omni=True)
    trace["use_graph"] = True

    evidence = _build_method_trigger_evidence("omnikv", [trace], {})

    assert evidence["triggered"] is True
    assert evidence["execution_mode"] == "captured_replay"


def test_full_logits_artifact_contains_both_complete_tensors_and_hashes(tmp_path):
    eager = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    graph = eager.clone()
    path = tmp_path / "comparison.full_logits.pt"

    metadata = _save_full_logits_artifact(path, eager=eager, graph=graph)
    artifact = torch.load(path, weights_only=True)

    torch.testing.assert_close(artifact["eager"], eager)
    torch.testing.assert_close(artifact["graph"], graph)
    assert artifact["scope"] == "all_decode_rows_and_full_vocabulary"
    assert metadata["eager"]["shape"] == [3, 8]
    assert metadata["eager"]["sha256"] == metadata["graph"]["sha256"]
    assert metadata["artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_logits_comparison_reports_tolerance_and_all_rows():
    eager = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    graph = eager + 0.01

    result = _compare_logits(eager, graph, atol=0.02, rtol=0.0)

    assert result["within_tolerance"] is True
    assert result["shape"] == [2, 2]
    assert len(result["rows"]) == 2
