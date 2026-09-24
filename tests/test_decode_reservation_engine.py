"""Opt-in CUDA oracle: reservation windows do not alter deterministic output."""
import json
import os
import socket

import pytest
import torch

from sparseengine import LLM, SamplingParams


@pytest.mark.parametrize('method', ['', 'quest', 'snapkv', 'h2o'])
@pytest.mark.parametrize('graph', [False, True])
def test_window_output_invariance(method, graph, tmp_path, monkeypatch):
    model = os.getenv('SPARSEENGINE_DECODE_WINDOW_MODEL')
    if not model or not torch.cuda.is_available():
        pytest.skip('set SPARSEENGINE_DECODE_WINDOW_MODEL and expose idle CUDA devices')
    tiny = tmp_path / 'tiny.json'
    tiny.write_text(json.dumps(dict(num_hidden_layers=2, hidden_size=256,
                                   intermediate_size=512, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=64)))
    results = []
    for window in (1, 7):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        monkeypatch.setenv('SPARSEENGINE_MASTER_PORT', str(port))
        llm = LLM(model, tiny_random=True, tiny_random_config=str(tiny),
                  tensor_parallel_size=int(os.getenv('SPARSEENGINE_DECODE_WINDOW_TP', '1')),
                  sparse_method=method, enable_prefix_caching=True,
                  max_model_len=128, max_num_batched_tokens=128,
                  engine_prefill_chunk_size=64, max_num_seqs_in_batch=2,
                  max_num_seqs_in_gpu=4, gpu_memory_utilization=0.04,
                  decode_reservation_tokens=window, decode_graph=graph,
                  decode_graph_capture_sizes=[1, 2], sink_keep_tokens=2,
                  recent_keep_tokens=4, decode_keep_tokens=8,
                  observation_window_size=4, h2o_prefill_budget=16,
                  h2o_decode_budget=16, h2o_decode_eviction=(method == "h2o"),
                  h2o_decode_eviction_interval=4)
        try:
            params = SamplingParams(max_tokens=17, ignore_eos=True, temperature=0)
            admissions = [llm.admit_request(list(range(10+i, 42+i)), params) for i in range(2)]
            outputs = {}
            for _ in range(128):
                done, _ = llm.step()
                outputs.update((row[0], row[1]) for row in done)
                if llm.is_finished():
                    break
            assert llm.is_finished()
            results.append([outputs[a.seq_id] for a in admissions])
            assert all(len(tokens) == 17 for tokens in results[-1])
            assert not llm.model_runner.runtime_state.decode_reservations.requests
            assert llm.scheduler.total_preemptions == 0
            if graph:
                assert llm.model_runner.decode_graph_runner.replay_count > 0
        finally:
            llm.exit()
    assert results[0] == results[1]
