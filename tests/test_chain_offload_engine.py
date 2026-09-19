"""Opt-in engine oracle: CPU-restored continuation equals GPU-resident continuation."""

import os
import socket
import json

import pytest
import torch

from sparseengine import LLM, SamplingParams


@pytest.mark.parametrize("method", ["snapkv", "h2o"])
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_chain_restore_matches_resident_engine(method, graph, monkeypatch):
    model = os.getenv("SPARSEENGINE_CHAIN_OFFLOAD_MODEL")
    if not model or not torch.cuda.is_available():
        pytest.skip("set SPARSEENGINE_CHAIN_OFFLOAD_MODEL and expose idle CUDA devices")
    tp = int(os.getenv("SPARSEENGINE_CHAIN_OFFLOAD_TP", "1"))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    monkeypatch.setenv("SPARSEENGINE_MASTER_PORT", str(port))
    llm = LLM(
        model, sparse_method=method, tensor_parallel_size=tp,
        enable_prefix_caching=True, enable_prefix_cache_offload=True,
        prefix_cache_host_size_gb=0.25, max_model_len=256,
        max_num_batched_tokens=256, engine_prefill_chunk_size=64,
        max_num_seqs_in_batch=1, max_num_seqs_in_gpu=2,
        gpu_memory_utilization=float(os.getenv("SPARSEENGINE_CHAIN_OFFLOAD_GPU_FRACTION", "0.04")),
        sink_keep_tokens=4, recent_keep_tokens=8, decode_keep_tokens=16,
        h2o_decode_budget=32, h2o_prefill_budget=32,
        h2o_decode_eviction_interval=1,
        decode_graph=graph, decode_graph_capture_sizes=[1],
    )
    try:
        params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
        coordinator = llm.model_runner.runtime_state.chain_cache_coordinator
        manager = llm.model_runner.cache_manager

        def turn(prompt, chain_id=None):
            admission = llm.admit_request(prompt, params, chain_id=chain_id)
            for _ in range(512):
                outputs, _ = llm.step()
                if outputs:
                    assert len(outputs) == 1 and outputs[0][0] == admission.seq_id
                    assert llm.is_finished()
                    return admission, outputs[0][1]
            pytest.fail("chain turn exceeded its bounded step budget")

        prompt = list(range(100, 180))
        a, first_a = turn(prompt)
        b, first_b = turn(prompt)
        assert first_a == first_b
        turn(list(range(200, 280)))  # row pressure demotes only the oldest chain
        assert coordinator.index.lookup(a.chain_id).resident_rows == 0
        assert coordinator.index.lookup(b.chain_id).resident_rows == 1
        demoted_ranks = llm.debug_sparse_state_summaries(synchronize=True)
        assert len(demoted_ranks) == tp
        for rank in demoted_ranks:
            rows = rank["state"]["cache"]["live_rows"]
            for layer_rows in rows.values():
                owners = {row["seq_id"] for row in layer_rows}
                assert a.seq_id not in owners and b.seq_id in owners
        graph_runner = llm.model_runner.decode_graph_runner
        replay_before = graph_runner.replay_count if graph else 0

        suffix = prompt + first_b + [300, 301, 302]
        _, resident_output = turn(suffix, b.chain_id)
        expected = []
        for layer, length in zip(manager.kv_transformer_layer_indices(), manager.chain_physical_residency(b.seq_id)):
            row = manager.seq_id_to_row[layer][b.seq_id]
            slots = manager.buffer_req_to_token_slots[layer][row, :length].long()
            expected.append(tuple(t[slots].clone() for t in manager.chain_storage_tensors(layer)))
        expected_method = {k: v.clone() for k, v in manager.snapshot_chain_method_state(b.seq_id).tensors.items()}
        llm.model_runner.call("chain_invalidate", b.chain_id, b.seq_id)
        _, restored_output = turn(suffix, a.chain_id)
        assert restored_output == resident_output
        for layer, payload in zip(manager.kv_transformer_layer_indices(), expected):
            row = manager.seq_id_to_row[layer][a.seq_id]
            slots = manager.buffer_req_to_token_slots[layer][row, :len(payload[0])].long()
            for actual, reference in zip(manager.chain_storage_tensors(layer), payload):
                torch.testing.assert_close(actual[slots], reference, rtol=0, atol=0)
        for name, reference in expected_method.items():
            torch.testing.assert_close(manager.snapshot_chain_method_state(a.seq_id).tensors[name], reference, rtol=0, atol=0)
        assert coordinator.offload.h2d_bytes > 0
        if graph:
            assert graph_runner.capture_count > 0
            assert graph_runner.replay_count > replay_before
        restored_ranks = llm.debug_sparse_state_summaries(synchronize=True)
        for rank in restored_ranks:
            for layer_rows in rank["state"]["cache"]["live_rows"].values():
                assert a.seq_id in {row["seq_id"] for row in layer_rows}
            if graph:
                assert rank["decode_graph"]["capture_count"] > 0
                assert rank["decode_graph"]["replay_count"] > 0
        print("chain_validation=" + json.dumps({
            "method": method, "tp": tp, "graph": graph,
            "resident_output": resident_output, "restored_output": restored_output,
            "d2h_bytes_rank0": coordinator.offload.d2h_bytes,
            "h2d_bytes_rank0": coordinator.offload.h2d_bytes,
            "rank_replays": [rank["decode_graph"]["replay_count"] for rank in restored_ranks],
        }))
    finally:
        llm.exit()
