from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip(
    "sgl_kernel.kvcacheio",
    reason="prefix offload engine tests require the prefix-offload extra",
)

from sparseengine import LLM, SamplingParams


MODEL_ENV = "SPARSEENGINE_TP2_PREFIX_OFFLOAD_MODEL"


def _logical_prefix_signature(summary: dict[str, object]) -> list[tuple[object, ...]]:
    prefix_cache = summary["state"]["cache"]["prefix_cache"]
    return [
        (
            block["stable_block_id"],
            block["parent_block_id"],
            block["logical_block_idx"],
            block["token_ids"],
            block["ref_count"],
            block["eviction_priority"],
            block["device_present"],
            block["host_present"],
            block["transfer"],
        )
        for block in prefix_cache["blocks"]
    ]


def test_tp2_prefix_offload_engine_demotes_and_promotes_real_model():
    model = os.getenv(MODEL_ENV)
    if not model:
        pytest.skip(f"set {MODEL_ENV} to run the TP2 prefix-offload engine test")
    if torch.cuda.device_count() < 2:
        pytest.skip("TP2 prefix-offload engine test requires two visible CUDA devices")

    llm = None
    try:
        llm = LLM(
            model,
            tensor_parallel_size=2,
            enable_prefix_caching=True,
            enable_prefix_cache_offload=True,
            prefix_cache_block_size=16,
            prefix_cache_max_blocks=1024,
            prefix_cache_host_size_gb=1.5,
            max_model_len=10240,
            max_num_batched_tokens=16384,
            engine_prefill_chunk_size=16384,
            max_num_seqs_in_batch=1,
            max_decoding_seqs=1,
            max_num_seqs_in_gpu=1,
            # On a 96 GiB H20 this leaves about 12K KV slots per rank: each
            # request fits alone, while the two completed prefixes do not.
            gpu_memory_utilization=0.0618,
        )
        sampling = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        short_prefix = list(range(1_000, 7_144))
        pressure_prompt = list(range(20_000, 30_224))
        local_kv_slots = int(llm.config.num_kvcache_slots)
        assert len(pressure_prompt) + 1 <= local_kv_slots < (
            len(short_prefix) + len(pressure_prompt)
        ), (
            "TP2 prefix-offload test did not create its pressure window: "
            f"pressure={len(pressure_prompt) + 1} slots={local_kv_slots} "
            f"combined={len(short_prefix) + len(pressure_prompt)}"
        )

        first = llm.generate([short_prefix], sampling, use_tqdm=False)
        pressure = llm.generate([pressure_prompt], sampling, use_tqdm=False)
        reused = llm.generate([short_prefix], sampling, use_tqdm=False)

        assert len(first[0]["token_ids"]) == 1
        assert len(pressure[0]["token_ids"]) == 1
        assert len(reused[0]["token_ids"]) == 1
        assert first[0]["token_ids"] == reused[0]["token_ids"]

        summaries = llm.debug_sparse_state_summaries()
        assert len(summaries) == 2
        assert _logical_prefix_signature(summaries[0]) == _logical_prefix_signature(
            summaries[1]
        )
        logical_stat_keys = (
            "prefix_cache_live_blocks",
            "prefix_cache_device_blocks",
            "prefix_cache_host_blocks",
            "prefix_cache_device_demoted_blocks",
            "prefix_cache_inflight_transfers",
        )
        rank_stats = [
            summary["state"]["cache"]["free_slot_stats"] for summary in summaries
        ]
        assert {key: rank_stats[0][key] for key in logical_stat_keys} == {
            key: rank_stats[1][key] for key in logical_stat_keys
        }
        for summary in summaries:
            stats = summary["state"]["cache"]["free_slot_stats"]
            assert stats["prefix_cache_d2h_completed_operations"] > 0, stats
            assert stats["prefix_cache_h2d_completed_operations"] > 0, stats
            assert stats["prefix_cache_h2d_layer_waits"] > 0, stats
    finally:
        if llm is not None:
            llm.exit()
