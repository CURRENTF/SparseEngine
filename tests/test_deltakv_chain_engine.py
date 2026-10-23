"""P1: real-model chain append must preserve history and conserve every pool.

Opt in with SPARSEENGINE_TEST_MODEL and SPARSEENGINE_TEST_OUTPUT_DIR. This catches
multi-pool leaks/corruption and recurrent-state loss that CPU chain mocks cannot.
"""
import json
import os
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available() or not os.getenv("SPARSEENGINE_TEST_MODEL"),
                    reason="requires CUDA and an explicit local model")
def test_deltakv_chain_pressure():
    from sparseengine import LLM, SamplingParams
    from sparseengine.engine.chain_cache import ChainBusyError, ChainCapacityError, ChainGoneError, ChainPrefixMismatchError

    result = {"status": "model_failed"}
    out = Path(os.environ["SPARSEENGINE_TEST_OUTPUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    llm = None
    try:
        kivi = os.environ.get("DELTAKV_TEST_KIVI", "0") == "1"
        config = dict(sparse_method="residualkv", use_compression=False,
            deltakv_checkpoint_path=None, deltakv_latent_quant_bits=4, deltakv_latent_quant_group_size=32,
            full_layer_kv_quant_bits=4 if kivi else 0, enable_full_layer_kivi_quant=kivi,
            full_attention_layers=[int(x) for x in os.environ.get("DELTAKV_TEST_FULL_LAYERS", "3,7").split(",")],
            sink_keep_tokens=4, recent_keep_tokens=32, decode_keep_tokens=128,
            deltakv_neighbor_count=4, deltakv_center_ratio=0.1,
            deltakv_sparse_decode_backend="custom", enable_prefix_caching=True, prefix_cache_mode="chain",
            max_model_len=4096, max_num_batched_tokens=2048, engine_prefill_chunk_size=128,
            max_num_seqs_in_batch=2, max_decoding_seqs=2, max_num_seqs_in_gpu=3,
            gpu_memory_utilization=0.8, decode_graph=True, decode_graph_capture_sizes=[1, 2],
            decode_graph_startup_capture=True)
        result["config"] = config
        llm = LLM(os.environ["SPARSEENGINE_TEST_MODEL"], **config)
        cm = llm.model_runner.cache_manager
        coordinator = llm.model_runner.runtime_state.chain_cache_coordinator
        graph = llm.model_runner.decode_graph_runner
        force = os.getenv("DELTAKV_TEST_FORCE_EAGER", "0") == "1"
        if force:
            os.environ["SPARSEENGINE_DELTAKV_CUDAGRAPH_FORCE_EAGER"] = "1"
        result["force_eager"] = force
        def counters():
            return {k: int(getattr(graph, k)) for k in ("capture_count", "replay_count", "eager_static_count", "force_eager_count")}
        result["graph_before"] = counters()
        # Compare the new staging path with the existing GPU reconstruction
        # kernel, independently of duplicate-chain output equivalence.
        original_view = cm.build_prefill_compute_view
        reconstruction_checks = []
        prefill_batches = set()
        max_visible = [0]
        def checked_view(layer_idx, k_current, v_current, selection):
            view = original_view(layer_idx, k_current, v_current, selection)
            from sparseengine.utils.context import get_context
            ctx = get_context()
            prefill_batches.add(len(ctx.seqs))
            max_visible[0] = max(max_visible[0], sum(int(cm.row_seq_lens[cm.seq_id_to_row[s.seq_id]]) for s in ctx.seqs))
            if (not reconstruction_checks and layer_idx in cm.deltakv_layer_to_idx
                    and any(cm._has_attached_prefill_prefix(seq) for seq in ctx.seqs)):
                rows = [cm.seq_id_to_row[seq.seq_id] for seq in ctx.seqs]
                count = min(16, min(int(cm.row_deltakv_compressed_lens[r]) for r in rows))
                indices = torch.arange(count, dtype=torch.int32, device=cm.device).expand(len(rows), -1).contiguous()
                slots, _, lengths, temporary = cm.deltakv_reconstruct(
                    layer_idx, indices, selection.context_lens, selection.req_indices, selection.chunk_lens)
                try:
                    l_idx = cm.deltakv_layer_to_idx[layer_idx]
                    for b in range(len(rows)):
                        selected = slots[b, :int(lengths[b])].long()
                        # The reconstruction plan orders sink, selected compressed
                        # positions, then the uncompressed tail. Temporary slots
                        # need not publish a persistent slot-to-position mapping.
                        sink = int(cm.config.sink_keep_tokens)
                        tail = sink + int(cm.row_deltakv_compressed_lens[rows[b]])
                        pos = torch.cat((torch.arange(sink, device=cm.device),
                            torch.arange(sink, sink + count, device=cm.device),
                            torch.arange(tail, int(cm.row_seq_lens[rows[b]]), device=cm.device)))
                        assert pos.numel() == selected.numel()
                        staged = view.meta.active_slots[b, pos].long()
                        torch.testing.assert_close(view.payload.k_cache[staged],
                            cm.deltakv_full_kv_cache[0, l_idx, selected], rtol=0.02, atol=0.03)
                        torch.testing.assert_close(view.payload.v_cache[staged],
                            cm.deltakv_full_kv_cache[1, l_idx, selected], rtol=0.02, atol=0.03)
                    reconstruction_checks.append({"layer": layer_idx, "selected_compressed_per_row": count})
                finally:
                    cm.release_layer_temp_slots(layer_idx, temporary)
            return view
        cm.build_prefill_compute_view = checked_view
        def pools():
            return {**cm.decode_window_budgets(), "rows": len(cm.free_rows)}
        baseline = pools()
        def conserve():
            totals = [0] * len(cm.chain_capacity_keys())
            for sid in cm.seq_id_to_row:
                for i, used in enumerate(cm.chain_capacity_residency(sid)):
                    totals[i] += used
            now = pools()
            for key, used in zip(cm.chain_capacity_keys(), totals):
                assert now.get(key, 0) + used == baseline.get(key, 0), (key, now, totals, baseline)
            assert len(cm.seq_id_to_row) + len(cm.free_rows) == baseline["rows"]
        params = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
        def drain(admissions):
            outputs = {}
            for _ in range(512):
                done, _ = llm.step()
                for sid, tokens, *_ in done:
                    assert sid not in outputs
                    assert len(tokens) == 32
                    outputs[sid] = list(tokens)
                if llm.is_finished():
                    break
            else:
                raise AssertionError("chain generation exceeded bounded steps")
            assert set(outputs) == {a.seq_id for a in admissions}
            conserve()
            return [outputs[a.seq_id] for a in admissions]
        messages = [{"role": "user", "content": "Remember: the archive code is blue. " * 80 + "Reply with the archive code."}]
        prompt = llm.tokenizer.encode(llm.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)
        suffix = llm.tokenizer.encode("\nNew records: keep the same archive code. " * 14, add_special_tokens=False)
        rounds = int(os.environ.get("DELTAKV_TEST_ROUNDS", "8"))
        turns = int(os.environ.get("DELTAKV_TEST_TURNS", "5"))
        waves = []
        reused = 0
        for wave in range(rounds):
            a = llm.admit_request(prompt, params)
            b = llm.admit_request(prompt, params)
            # A second writer must be rejected without changing ownership.
            with pytest.raises(ChainBusyError):
                coordinator.plan_admission(chain_id=a.chain_id, seq_id=a.seq_id, token_ids=prompt)
            outputs = drain([a, b])
            assert outputs[0] == outputs[1], "Duplicate independent chains diverged"
            histories = [prompt + row for row in outputs]
            wave_outputs = [outputs]
            for turn in range(1, turns):
                with pytest.raises(ChainPrefixMismatchError):
                    coordinator.plan_admission(chain_id=a.chain_id, seq_id=a.seq_id,
                        token_ids=[999] + histories[0][1:] + suffix)
                a = llm.admit_request(histories[0] + suffix, params, chain_id=a.chain_id)
                b = llm.admit_request(suffix, params, chain_id=b.chain_id, chain_append_only=True)
                assert a.chain_status == b.chain_status == "resumed"
                assert a.reused_tokens == len(histories[0]) - 1
                assert b.reused_tokens == len(histories[1]) - 1
                reused += a.reused_tokens + b.reused_tokens
                outputs = drain([a, b])
                assert outputs[0] == outputs[1], "Full-history and append-only resume diverged"
                histories = [hist + suffix + output for hist, output in zip(histories, outputs)]
                wave_outputs.append(outputs)
            # IDLE eviction at the resident-row limit, followed by slot reuse.
            c = llm.admit_request(prompt[:257], params)
            drain([c])
            victim_id = next(iter(coordinator.index.idle_resident_lru())).chain_id
            d = llm.admit_request(prompt[:193], params)
            drain([d])
            with pytest.raises(ChainGoneError):
                coordinator.index.lookup(victim_id)
            # Cancellation before preparation and after actual model execution.
            for stepped in (False, True):
                pending = llm.admit_request(prompt[:321], params)
                if stepped:
                    llm.step()
                llm.abort_request(pending.seq_id)
                cm.free_seq(pending.seq_id)  # Repeated payload release must be safe.
                conserve()
            for record in list(coordinator.index.records.values()):
                llm.model_runner.call("chain_invalidate", record.chain_id, record.seq_id)
            assert pools() == baseline
            assert not cm.seq_id_to_row
            waves.append(wave_outputs)
            if wave:
                assert waves[wave] == waves[0], "Allocator reuse changed deterministic outputs"
            print(json.dumps({"wave": wave, "status": "success", "reused_tokens": reused, "pools": pools()}), flush=True)
        # Exhaust resident promises with ACTIVE chains. Rejection must not evict them.
        active = [llm.admit_request(prompt[:129], params) for _ in range(3)]
        owners = set(coordinator.index.records)
        with pytest.raises(ChainCapacityError):
            llm.admit_request(prompt[:129], params)
        assert set(coordinator.index.records) == owners
        for a in active:
            llm.abort_request(a.seq_id)
        assert pools() == baseline
        result.update(rounds=rounds, turns_per_chain=turns,
            successful_generations=rounds * (2 * turns + 2),
            cancellations=rounds * 2 + 3, reused_tokens=reused, outputs=waves,
            chain_stats=coordinator.stats(), graph_after=counters(), final_pools=pools(), initial_pools=baseline,
            reconstruction_checks=reconstruction_checks, prefill_batch_sizes=sorted(prefill_batches),
            max_batch_visible_tokens=max_visible[0])
        assert result["graph_after"]["capture_count"] == result["graph_before"]["capture_count"]
        if not force:
            assert result["graph_after"]["replay_count"] > result["graph_before"]["replay_count"]
            assert result["graph_after"]["force_eager_count"] == result["graph_before"]["force_eager_count"]
            assert result["graph_after"]["eager_static_count"] == result["graph_before"]["eager_static_count"]
        assert reconstruction_checks
        result["status"] = "success"
    finally:
        (out / "result.json").write_text(json.dumps(result, indent=2))
        if llm is not None:
            llm.exit()
