"""Opt-in GPU/TP oracle for the scheduler's IDLE-reclaim control RPC."""
import json
import os
import socket

import pytest
import torch

from sparseengine import LLM, SamplingParams
from sparseengine.engine.chain_cache import ChainGoneError


@pytest.mark.parametrize("offload", [False, True])
def test_idle_reclaim_rpc_preserves_other_chain_and_graph_execution(offload, tmp_path, monkeypatch):
    model = os.getenv("SPARSEENGINE_DECODE_WINDOW_MODEL")
    if not model or not torch.cuda.is_available():
        pytest.skip("set SPARSEENGINE_DECODE_WINDOW_MODEL and expose idle CUDA devices")
    tp = int(os.getenv("SPARSEENGINE_DECODE_WINDOW_TP", "1"))
    tiny = tmp_path / "tiny.json"
    tiny.write_text(json.dumps(dict(num_hidden_layers=2, hidden_size=256,
        intermediate_size=512, num_attention_heads=4, num_key_value_heads=2, head_dim=64)))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    monkeypatch.setenv("SPARSEENGINE_MASTER_PORT", str(port))
    llm = LLM(model, tiny_random=True, tiny_random_config=str(tiny),
        tensor_parallel_size=tp, sparse_method="snapkv", enable_prefix_caching=True,
        enable_prefix_cache_offload=offload, prefix_cache_host_size_gb=0.25,
        max_model_len=256, max_num_batched_tokens=256, engine_prefill_chunk_size=64,
        max_num_seqs_in_batch=1, max_num_seqs_in_gpu=3, gpu_memory_utilization=0.04,
        decode_reservation_tokens=4, decode_graph=True, decode_graph_capture_sizes=[1],
        sink_keep_tokens=4, recent_keep_tokens=8, decode_keep_tokens=16, observation_window_size=4)
    try:
        params = SamplingParams(max_tokens=8, ignore_eos=True, temperature=0)
        def turn(prompt, chain_id=None):
            admission = llm.admit_request(prompt, params, chain_id=chain_id)
            for _ in range(128):
                done, _ = llm.step()
                if done:
                    assert len(done) == 1 and done[0][0] == admission.seq_id
                    assert llm.is_finished()
                    return admission, done[0][1]
            pytest.fail("turn exceeded bounded step budget")

        prompt = list(range(100, 180))
        a, first_a = turn(prompt)
        b, first_b = turn(prompt)
        assert first_a == first_b
        before = llm.debug_sparse_state_summaries(synchronize=True)
        llm.model_runner.call("chain_reclaim_idle", a.chain_id, a.seq_id, offload)
        after = llm.debug_sparse_state_summaries(synchronize=True)
        assert len(before) == len(after) == tp
        for old, new in zip(before, after):
            old_cache, new_cache = old["state"]["cache"], new["state"]["cache"]
            for layer, old_rows in old_cache["live_rows"].items():
                old_by_owner = {row["seq_id"]: row for row in old_rows}
                new_by_owner = {row["seq_id"]: row for row in new_cache["live_rows"][layer]}
                assert a.seq_id not in new_by_owner
                assert new_by_owner[b.seq_id] == old_by_owner[b.seq_id]
                released = old_by_owner[a.seq_id]["row_len"]
                assert new_cache["free_slot_stats"]["free_slots"] == old_cache["free_slot_stats"]["free_slots"] + released
        coordinator = llm.model_runner.runtime_state.chain_cache_coordinator
        if offload:
            assert coordinator.index.lookup(a.chain_id).resident_rows == 0
        else:
            with pytest.raises(ChainGoneError):
                coordinator.index.lookup(a.chain_id)
        replay_before = llm.model_runner.decode_graph_runner.replay_count
        suffix = prompt + first_b + [300, 301, 302]
        _, resident_output = turn(suffix, b.chain_id)
        assert len(resident_output) == 8
        if offload:
            _, restored_output = turn(suffix, a.chain_id)
            assert restored_output == resident_output
        assert llm.model_runner.decode_graph_runner.replay_count > replay_before
        final = llm.debug_sparse_state_summaries(synchronize=True)
        for old, new in zip(after, final):
            assert new["decode_graph"]["replay_count"] > old["decode_graph"]["replay_count"]
        (tmp_path / "rank_states.json").write_text(json.dumps({"before": before, "after_reclaim": after, "after_decode": final}))
        assert llm.scheduler.total_preemptions == 0
    finally:
        llm.exit()
