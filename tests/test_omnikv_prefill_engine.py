"""Opt-in real-model phase handoff and repeated decode graph regression.

Set SPARSEENGINE_TEST_MODEL and SPARSEENGINE_TEST_OUTPUT_DIR; select an idle GPU
with CUDA_VISIBLE_DEVICES. The model must have at least four KV layers. Set
SPARSEENGINE_TEST_PREFILL_LAYERS to comma-separated global observers for models
with sliding layers. Heterogeneous Gemma4 storage uses the vanilla cases only.
This is a correctness smoke, not quality scoring.
"""

import json
import os
from pathlib import Path

import pytest
import torch


def _prefill_layers():
    return [int(layer) for layer in os.environ.get("SPARSEENGINE_TEST_PREFILL_LAYERS", "0,2").split(",")]


@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("SPARSEENGINE_TEST_MODEL"),
    reason="requires CUDA and an explicit local test model",
)
@pytest.mark.parametrize("method,offload", [("", False), ("omnikv", False), ("omnikv", True)])
def test_prefill_then_repeated_graph_decode_matches_eager(method, offload):
    from sparseengine import LLM, SamplingParams

    output_dir = Path(os.environ["SPARSEENGINE_TEST_OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for graph in (False, True):
        kwargs = dict(
            model=os.environ["SPARSEENGINE_TEST_MODEL"], sparse_method=method,
            prefill_sparse_method="omnikv_prefill",
            omnikv_prefill_full_attention_layers=_prefill_layers(),
            omnikv_prefill_keep_tokens=32, omnikv_prefill_sink_keep_tokens=2,
            omnikv_prefill_recent_keep_tokens=8,
            full_attention_layers=[0, 2], sink_keep_tokens=2,
            recent_keep_tokens=8, decode_keep_tokens=32,
            max_model_len=1024, max_num_batched_tokens=256,
            max_num_seqs_in_batch=3, engine_prefill_chunk_size=128,
            gpu_memory_utilization=0.3, decode_graph=graph,
            decode_graph_capture_sizes=[1, 2, 3], enable_prefix_caching=False,
            enable_omnikv_offload=offload,
            validate_runtime_invariants=True,
        )
        llm = LLM(**kwargs)
        try:
            prompts = [[42] * 513, [43] * 257, [44]]
            waves = []
            stats = []
            for _ in range(2):
                result = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=8, ignore_eos=True), use_tqdm=False)
                assert [len(row["token_ids"]) for row in result] == [8, 8, 8]
                waves.append(result)
                runner = llm.model_runner.decode_graph_runner
                stats.append(dict(replay=runner.replay_count, capture=runner.capture_count))
                del runner
            (output_dir / f"{method or 'vanilla'}-offload{int(offload)}-graph{int(graph)}.json").write_text(
                json.dumps(dict(config=kwargs, prompts=prompts, waves=waves, graph_stats=stats,
                                operators=llm.operator_runtime_stats()), indent=2, default=str)
            )
            assert waves[0] == waves[1]
            if graph:
                assert stats[1]["replay"] > stats[0]["replay"] > 0
                assert stats[1]["capture"] == stats[0]["capture"]
            outputs.append(waves)
        finally:
            llm.exit()
    assert outputs[0] == outputs[1]


@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("SPARSEENGINE_TEST_MODEL"),
    reason="requires CUDA and an explicit local test model",
)
@pytest.mark.parametrize("method", ["", "omnikv"])
def test_chain_resume_and_eviction_match_omnikv_offload(method):
    from sparseengine import LLM, SamplingParams
    from sparseengine.engine.chain_cache import ChainGoneError, ChainPrefixMismatchError

    output_dir = Path(os.environ["SPARSEENGINE_TEST_OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = {}
    for offload_cache in ([None] if not method else [None, 0, 64]):
        for graph in (False, True):
            kwargs = dict(
                model=os.environ["SPARSEENGINE_TEST_MODEL"], sparse_method=method,
                prefill_sparse_method="omnikv_prefill",
                omnikv_prefill_full_attention_layers=_prefill_layers(),
                omnikv_prefill_keep_tokens=32, omnikv_prefill_sink_keep_tokens=2,
                omnikv_prefill_recent_keep_tokens=8,
                # Layer 2 scores full prefill from host KV; layer 1 consumes a
                # sparse prefill selection from its decode-resident GPU pool.
                full_attention_layers=[0, 1], sink_keep_tokens=2,
                recent_keep_tokens=8, decode_keep_tokens=32,
                max_model_len=1024, max_num_batched_tokens=128,
                max_num_seqs_in_batch=1, max_num_seqs_in_gpu=2,
                engine_prefill_chunk_size=128,
                gpu_memory_utilization=0.3, decode_graph=graph,
                decode_graph_capture_sizes=[1], enable_prefix_caching=True,
                enable_omnikv_offload=offload_cache is not None,
                omnikv_offload_cache_tokens=offload_cache,
                validate_runtime_invariants=True,
            )
            llm = LLM(**kwargs)
            try:
                params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)

                def turn(prompt, chain_id=None, *, append_only=False):
                    admission = llm.admit_request(prompt, params, chain_id=chain_id,
                                                 chain_append_only=append_only)
                    for _ in range(64):
                        result, _ = llm.step()
                        if result:
                            assert llm.is_finished()
                            assert len(result) == 1 and result[0][0] == admission.seq_id
                            assert len(result[0][1]) == 8
                            return admission, result[0][1]
                    pytest.fail("chain turn exceeded its bounded step budget")

                prompt = list(range(100, 357))
                a, first = turn(prompt)
                b, duplicate = turn(prompt)
                assert first == duplicate
                manager = llm.model_runner.cache_manager
                assert manager.prefix_cache is None
                coordinator = llm.model_runner.runtime_state.chain_cache_coordinator
                assert manager.chain_physical_residency(a.seq_id) == (len(prompt) + 7,) * manager.num_kv_layers
                # Strict prefix planning must retain the old chain and row on
                # failure; the public full-context API may deliberately recreate.
                with pytest.raises(ChainPrefixMismatchError):
                    coordinator.plan_admission(
                        chain_id=a.chain_id, seq_id=a.seq_id,
                        token_ids=[999] + prompt[1:] + first + [45],
                    )
                assert manager.chain_has_residency(a.seq_id)
                suffix = prompt + first + list(range(400, 533))
                resumed, second = turn(suffix, b.chain_id)
                assert resumed.seq_id == b.seq_id
                assert resumed.reused_tokens == len(prompt) + len(first) - 1
                assert resumed.prefilled_tokens == len(suffix) - resumed.reused_tokens
                replay_before = llm.model_runner.decode_graph_runner.replay_count
                captures_before = llm.model_runner.decode_graph_runner.capture_count
                # A third ID evicts only the oldest idle chain. Recycled slots
                # must invalidate decode LRU entries before their next use.
                c, third = turn([55] * 129)
                with pytest.raises(ChainGoneError):
                    coordinator.index.lookup(a.chain_id)
                assert manager.chain_has_residency(b.seq_id)
                _, last = turn([61, 62, 63], b.chain_id, append_only=True)
                if graph:
                    assert llm.model_runner.decode_graph_runner.replay_count > replay_before > 0
                    assert llm.model_runner.decode_graph_runner.capture_count == captures_before
                outputs = [first, second, third, last]
                (output_dir / f"chain-{method or 'vanilla'}-offload{offload_cache}-graph{int(graph)}.json").write_text(
                    json.dumps(dict(config=kwargs, outputs=outputs,
                                    reused_tokens=resumed.reused_tokens,
                                    prefilled_tokens=resumed.prefilled_tokens,
                                    chain_stats=coordinator.stats(),
                                    graph_replays=llm.model_runner.decode_graph_runner.replay_count,
                                    graph_captures=llm.model_runner.decode_graph_runner.capture_count), indent=2)
                )
                # Compare storage paths with the same decode implementation.
                # Eager and Graph use different BF16 kernels and may select
                # different greedy tokens when the top logits are nearly tied.
                if graph not in reference:
                    reference[graph] = outputs
                else:
                    assert outputs == reference[graph]
                llm.model_runner.call("chain_invalidate", b.chain_id, b.seq_id)
                llm.model_runner.call("chain_invalidate", c.chain_id, c.seq_id)
                assert manager.debug_live_seq_slots() == {}
                del manager, coordinator
            finally:
                llm.exit()
