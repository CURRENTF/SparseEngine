from types import SimpleNamespace

from sparseengine.engine.llm_engine import LLMEngine
from sparseengine.engine.model_runner import ModelRunner
from sparseengine.engine.startup import (
    CacheRuntimeBuildMeasurement,
    DeviceMemorySnapshot,
    MemoryProfileMeasurement,
)


def test_engine_rebuilds_production_runtime_before_final_graph_warmup():
    engine = object.__new__(LLMEngine)
    engine.config = SimpleNamespace(
        engine_prefill_chunk_size=8,
        max_model_len=32,
        max_num_batched_tokens=16,
        max_num_seqs_in_batch=4,
        max_decoding_seqs=3,
        gpu_memory_utilization=0.9,
        model_spec=SimpleNamespace(num_experts_field=None),
        hf_config=SimpleNamespace(vocab_size=128),
    )
    engine.scheduler = object()
    calls = []
    batches = []
    snapshot = DeviceMemorySnapshot(700, 1000, 300, 300)

    def profile_record(phase):
        measurement = {
            "cuda_graph": MemoryProfileMeasurement(50, 20),
            "decode": MemoryProfileMeasurement(0, 120),
        }[phase]
        record = {"world_rank": 0, "measurement": measurement}
        if phase == "cuda_graph":
            record["after"] = DeviceMemorySnapshot(700, 1000, 300, 300)
        return [record]

    def runner_call(method, *args):
        calls.append((method, *args))
        if method == "finish_startup_memory_profile":
            return profile_record(args[0])
        if method == "profile_startup_prefill":
            return [{"world_rank": 0, "measurement": MemoryProfileMeasurement(0, 180)}]
        if method == "release_profiling_cache_runtime":
            assert engine.scheduler is None
            return [
                {
                    "world_rank": 0,
                    "snapshot": snapshot,
                    "pre_graph_release_snapshot": DeviceMemorySnapshot(
                        700, 1000, 300, 300
                    ),
                    "post_graph_release_snapshot": DeviceMemorySnapshot(
                        750, 1000, 250, 250
                    ),
                    "profiling_kv_budget_bytes": 0,
                    "runtime_build": CacheRuntimeBuildMeasurement(0, 0, 0),
                }
            ]
        if method == "build_production_cache_runtime":
            assert args == (370,)
            return [{"world_rank": 0, "num_kvcache_slots": 64}]
        if method == "capture_startup_memory_snapshot":
            return [{"world_rank": 0, "snapshot": snapshot}]
        return None

    engine.model_runner = SimpleNamespace(call=runner_call)
    engine._create_scheduler = lambda: "production-scheduler"

    def run_batch(prompt_lengths, sampling_params, prompt_offset):
        batches.append(
            (
                tuple(prompt_lengths),
                int(sampling_params.max_tokens),
                bool(sampling_params.ignore_eos),
            )
        )
        return prompt_offset + len(prompt_lengths)

    engine._run_startup_batch = run_batch
    engine._capture_startup_decode_graphs = lambda prompt_offset, **kwargs: (
        calls.append(("capture_graphs", prompt_offset, kwargs)) or prompt_offset
    )

    engine._warmup()

    assert engine.scheduler == "production-scheduler"
    assert [batch[1:] for batch in batches] == [
        (1, False),
        (2, True),
        (2, True),
    ]
    assert [call[0] for call in calls].count("capture_graphs") == 2
    # Temporary layer-skewed KV pools can reject graph batches too, before
    # production capacity is known (PyramidKV's final layers are smallest).
    assert all(
        call[2].get("respect_runtime_capacity") is True
        for call in calls if call[0] == "capture_graphs"
    )
    assert calls.count(("profile_startup_prefill",)) == 1
    assert ("begin_startup_memory_profile", "prefill") not in calls
    assert calls.index(("profile_startup_prefill",)) < calls.index(
        ("begin_startup_memory_profile", "cuda_graph")
    )
    assert calls.index(("build_production_cache_runtime", 370)) < calls.index(
        ("capture_graphs", 0, {"respect_runtime_capacity": True})
    )


def test_model_runner_restores_tokenizer_metadata_after_controller_rebuild():
    class Controller:
        def __init__(self):
            self.calls = []

        def set_tokenizer_metadata(self, **kwargs):
            self.calls.append(kwargs)

    runner = object.__new__(ModelRunner)
    runner._tokenizer_metadata = None
    profiling_controller = Controller()
    runner.sparse_controller = profiling_controller
    runner.set_tokenizer_metadata([3, 7], [11], [13, 17])

    production_controller = Controller()
    runner.sparse_controller = production_controller
    runner._restore_tokenizer_metadata()

    expected = {
        "auxiliary_prefill_token_ids": [13, 17],
        "delimiter_token_ids": [3, 7],
        "non_execution_token_ids": [11],
    }
    assert profiling_controller.calls == [expected]
    assert production_controller.calls == [expected]
