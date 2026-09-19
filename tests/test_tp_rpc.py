import os
import threading
import time
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
import torch
import torch.distributed as dist

from sparseengine.engine.model_runner import (
    ModelRunner,
    TP_RUN_STATUS_FAILED,
    TP_RUN_STATUS_SUCCESS,
    TP_SHM_NAME_PREFIX,
    _create_model,
    _init_process_group,
    make_tp_shm_name,
)
from sparseengine.engine.startup import (
    CacheRuntimeBuildMeasurement,
    DeviceMemorySnapshot,
)
from sparseengine.models.spec import ModelSpec
from sparseengine.operators import registry as operator_registry


def _runner():
    runner = object.__new__(ModelRunner)
    runner.world_size = 1
    runner.parallel_context = SimpleNamespace(attn_tp_size=1, attn_tp_rank=0, attn_dp_size=1)
    return runner


def test_init_process_group_binds_the_current_device():
    device = torch.device("cuda", 2)

    with patch("sparseengine.engine.model_runner.dist.init_process_group") as init:
        _init_process_group(
            backend="nccl",
            init_method="tcp://localhost:1234",
            world_size=4,
            rank=2,
            device=device,
        )

    init.assert_called_once_with(
        backend="nccl",
        init_method="tcp://localhost:1234",
        world_size=4,
        rank=2,
        device_id=device,
    )


def test_init_process_group_omits_device_id_for_legacy_torch():
    with (
        patch(
            "sparseengine.engine.model_runner._INIT_PROCESS_GROUP_SUPPORTS_DEVICE_ID",
            False,
        ),
        patch("sparseengine.engine.model_runner.dist.init_process_group") as init,
    ):
        _init_process_group(
            backend="nccl",
            init_method="tcp://localhost:1234",
            world_size=2,
            rank=0,
            device=torch.device("cuda", 0),
        )

    init.assert_called_once_with(
        backend="nccl",
        init_method="tcp://localhost:1234",
        world_size=2,
        rank=0,
    )


def test_create_model_delegates_optional_runtime_binding():
    config = object()
    context = object()

    class Model:
        @classmethod
        def build_runtime_kwargs(cls, hf_config, **runtime_kwargs):
            assert hf_config is config
            assert runtime_kwargs == {"parallel_context": context}
            return {"runtime": "bound"}

        def __init__(self, hf_config, *, runtime):
            self.args = hf_config, runtime

    with patch.dict(_create_model.__globals__, {"Model": Model}):
        model = _create_model(
            config,
            ModelSpec("test", runtime_class_name="Model"),
            parallel_context=context,
        )

    assert model.args == (config, "bound")


def test_create_model_closes_runtime_bindings_when_construction_fails():
    binding = SimpleNamespace(close=lambda: calls.append("close"))
    calls = []

    class Model:
        @classmethod
        def build_runtime_kwargs(cls, hf_config, **runtime_kwargs):
            del cls, hf_config, runtime_kwargs
            return {"runtime": binding, "shared_runtime": binding}

        def __init__(self, hf_config, *, runtime, shared_runtime):
            del self, hf_config, runtime, shared_runtime
            raise RuntimeError("model construction failed")

    with (
        patch.dict(_create_model.__globals__, {"Model": Model}),
        pytest.raises(RuntimeError, match="model construction failed"),
    ):
        _create_model(object(), ModelSpec("test", runtime_class_name="Model"))

    assert calls == ["close"]


def test_write_shm_waits_until_worker_reads_command():
    ctx = get_context("spawn")
    command_event = ctx.Event()
    completion_event = ctx.Event()
    event = (command_event, completion_event)
    shm = SharedMemory(
        name=f"sparseengine_test_rpc_{os.getpid()}_{uuid4().hex}",
        create=True,
        size=2**20,
    )
    rank0 = SimpleNamespace(
        parallel_context=SimpleNamespace(attn_tp_size=2, attn_tp_rank=0),
        event=[event],
        shm=shm,
        _run_status_offset=lambda rank: len(shm.buf) - 2 + rank,
    )
    worker = SimpleNamespace(parallel_context=SimpleNamespace(attn_tp_size=2, attn_tp_rank=1), event=event, shm=shm)
    errors: list[BaseException] = []

    def write_command():
        try:
            ModelRunner.write_shm(rank0, "free_slots", 123)
        except BaseException as exc:  # pragma: no cover - surfaced below.
            errors.append(exc)

    writer = threading.Thread(target=write_command)
    writer.start()

    try:
        assert command_event.wait(timeout=1.0)
        time.sleep(0.02)
        assert writer.is_alive()

        method_name, args = ModelRunner.read_shm(worker)
        writer.join(timeout=1.0)

        assert not writer.is_alive()
        assert errors == []
        assert method_name == "free_slots"
        assert args == [123]
    finally:
        if command_event.is_set():
            command_event.clear()
        writer.join(timeout=1.0)
        shm.close()
        shm.unlink()


def test_write_shm_can_defer_read_ack_to_status_sync():
    ctx = get_context("spawn")
    command_event = ctx.Event()
    completion_event = ctx.Event()
    shm = SharedMemory(
        name=f"sparseengine_test_rpc_{os.getpid()}_{uuid4().hex}",
        create=True,
        size=2**20,
    )
    rank0 = SimpleNamespace(
        parallel_context=SimpleNamespace(attn_tp_size=2, attn_tp_rank=0),
        event=[(command_event, completion_event)],
        shm=shm,
        _run_status_offset=lambda rank: len(shm.buf) - 2 + rank,
    )
    worker = SimpleNamespace(
        parallel_context=SimpleNamespace(attn_tp_size=2, attn_tp_rank=1),
        event=(command_event, completion_event),
        shm=shm,
    )

    try:
        ModelRunner.write_shm(
            rank0,
            "run",
            [1],
            False,
            wait_for_read=False,
        )

        assert command_event.is_set()
        method_name, args = ModelRunner.read_shm(worker)
        assert method_name == "run"
        assert args == [[1], False]
    finally:
        if command_event.is_set():
            command_event.clear()
        shm.close()
        shm.unlink()


def test_call_defers_read_ack_only_for_status_synchronized_rpc():
    runner = _runner()
    runner.parallel_context.attn_tp_size = 2
    runner.rank = 0
    runner.parallel_context.attn_tp_rank = 0
    runner.config = SimpleNamespace(decode_graph=False)
    calls = []
    runner.write_shm = lambda method, *args, **kwargs: calls.append(
        ("write", method, args, kwargs)
    )
    runner._sync_tp_rpc_status = lambda method, error: calls.append(
        ("sync", method, error)
    )
    runner.free_slots = lambda seq_id: calls.append(("free", seq_id))
    runner.unsynchronized = lambda value: value + 1

    ModelRunner.call(runner, "free_slots", 7)
    result = ModelRunner.call(runner, "unsynchronized", 8)

    assert result == 9
    assert calls == [
        ("write", "free_slots", (7,), {"wait_for_read": False}),
        ("free", 7),
        ("sync", "free_slots", None),
        ("write", "unsynchronized", (8,), {"wait_for_read": True}),
    ]


def test_tp_shm_name_is_unique_per_engine_instance():
    names = {make_tp_shm_name() for _ in range(3)}

    assert len(names) == 3
    assert all(name.startswith(TP_SHM_NAME_PREFIX) for name in names)
    assert "sparseengine" not in names


def test_free_slots_batch_releases_each_seq_id():
    freed: list[int] = []

    class FakeRuntimeState:
        def free_seq(self, seq_id: int):
            freed.append(int(seq_id))

    runner = _runner()
    runner.runtime_state = FakeRuntimeState()

    ModelRunner.free_slots_batch(runner, [3, 5, 8])

    assert freed == [3, 5, 8]


def test_operator_implementation_log_runs_only_on_rank_zero():
    rank_zero = _runner()
    rank_zero.parallel_context = SimpleNamespace(world_rank=0)
    rank_one = _runner()
    rank_one.parallel_context = SimpleNamespace(world_rank=1)

    with patch.object(operator_registry, "log_operator_implementations") as log_implementations:
        ModelRunner.log_operator_implementations(rank_zero)
        ModelRunner.log_operator_implementations(rank_one)

    log_implementations.assert_called_once_with()


def test_operator_runtime_stats_gather_one_record_per_world_rank():
    runner = _runner()
    runner.collective_runtime = SimpleNamespace(moe_transport_stats=lambda: {"provider": "test"})
    runner.parallel_context.attn_tp_size = 2
    runner.rank = 0
    runner.parallel_context.attn_tp_rank = 0
    runner.parallel_context = SimpleNamespace(
        world_rank=0,
        attn_tp_size=2, attn_tp_rank=0,
        attn_tp=SimpleNamespace(process_group="world"),
    )
    sync_calls = []
    runner._sync_tp_rpc_status = (
        lambda method_name, error: sync_calls.append((method_name, error))
    )

    def gather(output, local, *, group):
        assert group == "world"
        output[:] = [local, {"world_rank": 1, "bindings": [], "operators": {}}]

    with (
        patch.object(operator_registry, "operator_binding_reports", return_value=[]),
        patch.object(operator_registry, "operator_runtime_stats", return_value={"MLA": []}),
        patch.object(dist, "all_gather_object", side_effect=gather),
    ):
        stats = ModelRunner.operator_runtime_stats(runner)

    assert sync_calls == [("operator_runtime_stats", None)]
    assert stats == [
        {"world_rank": 0, "bindings": [], "operators": {"MLA": []},
         "moe_communication": {"provider": "test"}, "async_execution": None},
        {"world_rank": 1, "bindings": [], "operators": {}},
    ]


def test_prefix_offload_release_rpc_surfaces_local_failure_after_status_sync():
    for method_name, args in (("free_slots", (7,)), ("free_slots_batch", ([7, 9],))):
        runner = _runner()
        runner.parallel_context.attn_tp_size = 1
        runner.rank = 0
        runner.parallel_context.attn_tp_rank = 0
        expected = RuntimeError(f"{method_name} failed")
        setattr(runner, method_name, lambda *unused, error=expected: (_ for _ in ()).throw(error))
        calls = []
        runner._sync_tp_rpc_status = lambda method, error: calls.append((method, error))

        try:
            ModelRunner.call(runner, method_name, *args)
        except RuntimeError as exc:
            assert exc is expected
        else:
            raise AssertionError(f"expected {method_name} failure")
        assert calls == [(method_name, expected)]


def test_prefix_cache_control_rpc_reports_any_tp_worker_failure():
    runner = _runner()
    runner.parallel_context.attn_tp_size = 2
    runner.device = torch.device("cpu")
    runner.parallel_context = SimpleNamespace(
        attn_tp_size=2, attn_tp_rank=0,
        attn_tp=SimpleNamespace(all_reduce=lambda tensor, op: dist.all_reduce(tensor, op=op))
    )

    def mark_failed(tensor, op=None):
        assert op == dist.ReduceOp.MAX
        tensor.fill_(1)

    with patch.object(dist, "is_initialized", return_value=True), patch.object(dist, "all_reduce", side_effect=mark_failed):
        try:
            ModelRunner._sync_prefix_cache_control_rpc_status(runner, "prefix_cache_delete_subtree", None)
        except RuntimeError as exc:
            assert "At least one attention TP worker failed" in str(exc)
        else:
            raise AssertionError("expected worker failure to be surfaced on rank 0")


def test_run_rpc_reports_any_tp_worker_failure():
    ctx = get_context("spawn")
    events = (ctx.Event(), ctx.Event())
    shm = SharedMemory(
        name=f"sparseengine_test_status_{os.getpid()}_{uuid4().hex}",
        create=True,
        size=2**20,
    )
    rank0 = _runner()
    rank0.parallel_context.attn_tp_size = 2
    rank0.rank = 0
    rank0.parallel_context.attn_tp_rank = 0
    rank0.event = [events]
    rank0.shm = shm
    rank0.device = torch.device("cpu")
    rank0.platform = SimpleNamespace(synchronize=lambda: None)
    worker = _runner()
    worker.parallel_context.attn_tp_size = 2
    worker.rank = 1
    worker.parallel_context.attn_tp_rank = 1
    worker.event = events
    worker.shm = shm
    worker.device = torch.device("cpu")
    worker.platform = SimpleNamespace(synchronize=lambda: None)

    try:
        ModelRunner._sync_tp_run_status(worker, RuntimeError("worker failed"))
        assert shm.buf[ModelRunner._run_status_offset(worker, 1)] == TP_RUN_STATUS_FAILED
        try:
            ModelRunner._sync_tp_run_status(rank0, None)
        except RuntimeError as exc:
            assert "TP worker rank(s) 1 failed during run" in str(exc)
        else:
            raise AssertionError("expected worker failure to be surfaced on rank 0")
    finally:
        shm.close()
        shm.unlink()


def test_run_rpc_uses_host_completion_without_collective():
    ctx = get_context("spawn")
    events = (ctx.Event(), ctx.Event())
    shm = SharedMemory(
        name=f"sparseengine_test_status_{os.getpid()}_{uuid4().hex}",
        create=True,
        size=2**20,
    )
    sync_calls: list[int] = []
    rank0 = _runner()
    rank0.parallel_context.attn_tp_size = 2
    rank0.rank = 0
    rank0.parallel_context.attn_tp_rank = 0
    rank0.event = [events]
    rank0.shm = shm
    rank0.device = torch.device("cpu")
    rank0.platform = SimpleNamespace(synchronize=lambda: sync_calls.append(0))
    worker = _runner()
    worker.parallel_context.attn_tp_size = 2
    worker.rank = 1
    worker.parallel_context.attn_tp_rank = 1
    worker.event = events
    worker.shm = shm
    worker.device = torch.device("cpu")
    worker.platform = SimpleNamespace(synchronize=lambda: sync_calls.append(1))

    try:
        ModelRunner._sync_tp_run_status(worker, None)
        assert shm.buf[ModelRunner._run_status_offset(worker, 1)] == TP_RUN_STATUS_SUCCESS
        ModelRunner._sync_tp_run_status(rank0, None)
        assert sync_calls == [1, 0]
    finally:
        shm.close()
        shm.unlink()


def test_run_rpc_uses_host_status_with_decode_graph():
    runner = _runner()
    runner.parallel_context.attn_tp_size = 1
    runner.rank = 0
    runner.parallel_context.attn_tp_rank = 0
    runner.config = SimpleNamespace(decode_graph=True)
    runner.run = lambda seqs, is_prefill: (seqs, is_prefill)
    calls = []
    runner._sync_tp_run_status = lambda error: calls.append(("host", error))
    runner._sync_tp_rpc_status = lambda method, error: calls.append(
        ("collective", method, error)
    )

    result = ModelRunner.call(runner, "run", [1], False)

    assert result == ([1], False)
    assert calls == [("host", None)]


def test_run_rpc_keeps_collective_status_without_decode_graph():
    runner = _runner()
    runner.parallel_context.attn_tp_size = 1
    runner.rank = 0
    runner.parallel_context.attn_tp_rank = 0
    runner.config = SimpleNamespace(decode_graph=False)
    runner.run = lambda seqs, is_prefill: (seqs, is_prefill)
    calls = []
    runner._sync_tp_run_status = lambda error: calls.append(("host", error))
    runner._sync_tp_rpc_status = lambda method, error: calls.append(
        ("collective", method, error)
    )

    result = ModelRunner.call(runner, "run", [1], False)

    assert result == ([1], False)
    assert calls == [("collective", "run", None)]


def test_decode_graph_lifecycle_rpc_uses_host_status():
    runner = _runner()
    runner.parallel_context.attn_tp_size = 1
    runner.rank = 0
    runner.parallel_context.attn_tp_rank = 0
    runner.config = SimpleNamespace(decode_graph=True)
    runner.collect_decode_cuda_graph_metadata = lambda: None
    calls = []
    runner._sync_tp_host_status = lambda method, error: calls.append(
        ("host", method, error)
    )
    runner._sync_tp_rpc_status = lambda method, error: calls.append(
        ("collective", method, error)
    )

    ModelRunner.call(runner, "collect_decode_cuda_graph_metadata")

    assert calls == [("host", "collect_decode_cuda_graph_metadata", None)]


def test_model_runner_moe_workspace_warmup_delegates_token_count():
    calls = []
    runner = _runner()
    runner.model = SimpleNamespace(
        warmup_moe=lambda **kwargs: calls.append(kwargs)
    )

    ModelRunner.warmup_moe_workspace(runner, 16_384)

    assert calls == [{"num_tokens": 16_384}]


def test_prefix_cache_lookup_rpc_checks_rank_results():
    runner = _runner()
    runner.parallel_context.attn_tp_size = 1
    runner.rank = 0
    runner.parallel_context.attn_tp_rank = 0
    calls = []
    result = {"enabled": False, "hit_len": 0}
    runner.refresh_prefix_cache_hit = lambda seq: calls.append(("lookup", seq)) or result
    runner._sync_tp_rpc_status = lambda method, error: calls.append(("status", method, error))
    runner._sync_prefix_cache_lookup_result = lambda value: calls.append(("result", value))

    seq = object()
    actual = ModelRunner.call(runner, "refresh_prefix_cache_hit", seq)

    assert actual is result
    assert calls == [
        ("lookup", seq),
        ("status", "refresh_prefix_cache_hit", None),
        ("result", result),
    ]


def test_prefix_cache_lookup_rejects_rank_divergence():
    runner = _runner()
    runner.parallel_context.attn_tp_size = 2
    runner.parallel_context = SimpleNamespace(
        attn_tp_size=2, attn_tp_rank=0,
        attn_tp=SimpleNamespace(process_group=object())
    )

    def gather(results, local_result, group=None):
        assert group is runner.parallel_context.attn_tp.process_group
        results[:] = [local_result, {**local_result, "hit_len": 0}]

    with patch.object(dist, "all_gather_object", side_effect=gather):
        try:
            ModelRunner._sync_prefix_cache_lookup_result(
                runner,
                {"enabled": True, "hit_len": 8},
            )
        except RuntimeError as exc:
            assert "lookup diverged across attention TP ranks" in str(exc)
        else:
            raise AssertionError("expected divergent prefix-cache lookup to fail")


def test_model_runner_prefix_cache_lookup_returns_sequence_metadata():
    runner = _runner()

    def refresh(seq):
        seq.prefix_cache_enabled = True
        seq.prefix_cache_hit_len = 8
        seq.prefix_cache_hit_block_count = 2
        seq.prefix_cache_hit_last_block_id = b"block"
        seq.prefix_cache_block_size = 4
        seq.prefix_cache_method = "quest"

    runner.runtime_state = SimpleNamespace(refresh_prefix_cache_hit=refresh)
    seq = SimpleNamespace(
        prefix_cache_enabled=False,
        prefix_cache_hit_len=0,
        prefix_cache_hit_block_count=0,
        prefix_cache_hit_last_block_id=None,
        prefix_cache_block_size=0,
        prefix_cache_method="",
    )

    result = ModelRunner.refresh_prefix_cache_hit(runner, seq)

    assert result == {
        "enabled": True,
        "hit_len": 8,
        "hit_block_count": 2,
        "hit_last_block_id": b"block",
        "block_size": 4,
        "method": "quest",
    }


def test_tp_worker_continues_after_multimodal_registration_failure():
    runner = _runner()
    commands = iter(
        [
            ("register_multimodal_shared", []),
            ("free_multimodal", []),
            ("exit", []),
        ]
    )
    calls = []
    runner.read_shm = lambda: next(commands)

    def call(method_name, *_args):
        calls.append(method_name)
        if method_name == "register_multimodal_shared":
            raise ValueError("rank-local encoder failure")

    runner.call = call
    ModelRunner.loop(runner)

    assert calls == ["register_multimodal_shared", "free_multimodal", "exit"]


def test_tp_worker_continues_after_chain_admission_validation_failure():
    runner = _runner()
    commands = iter(
        [
            ("chain_validate_admission_plan", []),
            ("free_slots", []),
            ("exit", []),
        ]
    )
    calls = []
    runner.read_shm = lambda: next(commands)

    def call(method_name, *_args):
        calls.append(method_name)
        if method_name == "chain_validate_admission_plan":
            raise ValueError("rank-local chain capacity rejected")

    runner.call = call
    ModelRunner.loop(runner)

    assert calls == ["chain_validate_admission_plan", "free_slots", "exit"]


def test_tp_worker_can_exit_cleanly_after_graph_registration_failure():
    runner = _runner()
    commands = iter(
        [
            ("register_decode_cuda_graph_buffers", []),
            ("exit", []),
        ]
    )
    calls = []
    runner.read_shm = lambda: next(commands)

    def call(method_name, *_args):
        calls.append(method_name)
        if method_name == "register_decode_cuda_graph_buffers":
            raise RuntimeError("rank-local registration failure")

    runner.call = call
    ModelRunner.loop(runner)

    assert calls == ["register_decode_cuda_graph_buffers", "exit"]


def test_model_runner_reset_after_warmup_resets_local_runtime_state():
    calls = []
    runner = _runner()
    runner.runtime_state = SimpleNamespace(
        reset_after_warmup=lambda: calls.append("runtime")
    )
    runner.decode_graph_runner = SimpleNamespace(
        clear_captured_graphs=lambda: calls.append("graphs")
    )
    runner.sparse_controller = SimpleNamespace(
        clear_decode_attn_score_buffers=lambda: calls.append("scores")
    )

    with patch.dict(
        os.environ,
        {
            "SPARSEENGINE_DELTAKV_CLEAR_GRAPHS_AFTER_WARMUP": "0",
            "SPARSEENGINE_DELTAKV_CLEAR_ATTN_SCORE_BUFFERS_AFTER_WARMUP": "0",
        },
    ):
        ModelRunner.reset_after_warmup(runner)

    assert calls == ["runtime"]


def test_model_runner_releases_the_complete_profiling_cache_runtime():
    calls = []
    controller = object()
    cache_manager = object()
    runner = _runner()
    runner.dp_idle_graphs = {1: object()}
    runner.cache_runtime_phase = "profiling"
    runner.platform = SimpleNamespace(synchronize=lambda: calls.append("sync"))
    runner.reset_after_warmup = lambda: calls.append("reset")
    runner.decode_graph_runner = SimpleNamespace(
        clear_captured_graphs=lambda: calls.append("clear_graphs")
    )
    runner.config = SimpleNamespace(decode_graph=True)
    runner.collective_runtime = SimpleNamespace(
        reset_for_cuda_graph_recapture=lambda: calls.append("reset_collectives")
    )
    runner.model = SimpleNamespace(
        model=SimpleNamespace(sparse_controller=controller),
        release_cache_runtime_bindings=lambda manager: calls.append(
            ("release_bindings", manager)
        ),
    )
    runner.sparse_controller = controller
    runner.runtime_state = object()
    runner.prefix_cache_coordinator = object()
    runner.chain_cache_coordinator = object()
    runner.cache_manager = cache_manager
    runner.parallel_context.attn_tp_size = 1
    runner.rank = 0
    runner.parallel_context.attn_tp_rank = 0
    runner.device = torch.device("cpu")
    runner.profiling_kv_budget_bytes = 123
    runner.cache_runtime_build_measurement = CacheRuntimeBuildMeasurement(120, 0, 3)
    snapshot = DeviceMemorySnapshot(700, 1000, 300, 300)

    with (
        patch(
            "sparseengine.engine.model_runner.release_unused_device_memory",
            side_effect=lambda _platform: calls.append("release_memory"),
        ),
        patch.object(DeviceMemorySnapshot, "capture", return_value=snapshot),
    ):
        records = ModelRunner.release_profiling_cache_runtime(runner)

    assert calls == [
        "sync",
        "reset",
        "release_memory",
        "clear_graphs",
        "reset_collectives",
        "release_memory",
        ("release_bindings", cache_manager),
        "release_memory",
    ]
    assert runner.cache_runtime_phase == "released"
    assert runner.model.model.sparse_controller is None
    assert runner.decode_graph_runner is None
    assert runner.runtime_state is None
    assert runner.cache_manager is None
    assert not runner.dp_idle_graphs
    assert records == [
        {
            "world_rank": 0,
            "snapshot": snapshot,
            "pre_graph_release_snapshot": snapshot,
            "post_graph_release_snapshot": snapshot,
            "profiling_kv_budget_bytes": 123,
            "runtime_build": CacheRuntimeBuildMeasurement(120, 0, 3),
        }
    ]


def test_model_runner_runtime_rebuild_resolves_graph_shapes_locally():
    runner = _runner()
    runner.parallel_context = object()
    runner.platform = object()
    runner.device = torch.device("cpu")
    runner.recurrent_state_manager = None
    runner.model = SimpleNamespace()
    runner.load_deltakv_compressors = lambda: None
    runner.run_model = object()

    runner.collective_runtime = object()
    config = SimpleNamespace(
        resolved_prefix_cache_mode="disabled",
        max_num_seqs_in_batch=8,
        max_decoding_seqs=16,
        decode_graph_capture_sizes="auto",
        max_model_len=1024,
        sparse_method="vanilla",
    )
    graph_kwargs = {}
    auxiliary_executors = []

    with (
        patch(
            "sparseengine.engine.model_runner.CacheManager.create",
            return_value=object(),
        ),
        patch("sparseengine.engine.model_runner.RuntimeState", return_value=object()),
        patch(
            "sparseengine.engine.model_runner.SparseController",
            return_value=SimpleNamespace(bind_auxiliary_prefill=auxiliary_executors.append),
        ),
        patch(
            "sparseengine.engine.model_runner.collect_decode_graph_participants",
            return_value=(),
        ),
        patch(
            "sparseengine.engine.model_runner._decode_cuda_graph_max_real_batch_size",
            return_value=8,
        ),
        patch(
            "sparseengine.engine.model_runner._resolve_decode_cuda_graph_capture_sizes",
            return_value=(1, 8),
        ),
        patch(
            "sparseengine.engine.model_runner.DecodeCudaGraphRunner",
            side_effect=lambda **kwargs: graph_kwargs.update(kwargs) or object(),
        ),
        patch(
            "sparseengine.engine.model_runner.release_unused_device_memory",
        ),
        patch.object(
            DeviceMemorySnapshot,
            "capture",
            side_effect=(
                DeviceMemorySnapshot(900, 1000, 100, 100),
                DeviceMemorySnapshot(700, 1000, 300, 300),
                DeviceMemorySnapshot(680, 1000, 320, 320),
            ),
        ),
    ):
        ModelRunner._build_cache_runtime(
            runner,
            config,
            allocation_budget_bytes=1234,
        )

    assert graph_kwargs["capture_sizes"] == (1, 8)
    assert graph_kwargs["method"] == "vanilla"
    assert auxiliary_executors == [runner._run_auxiliary_prefill]
    assert runner.cache_runtime_build_measurement == CacheRuntimeBuildMeasurement(
        manager_consumed_bytes=200,
        manager_budget_overflow_bytes=0,
        auxiliary_consumed_bytes=20,
    )


def test_model_runner_decode_graph_startup_controls_use_live_runner():
    calls = []
    runner = _runner()
    runner.decode_graph_runner = SimpleNamespace(
        seal_startup_plan=lambda: calls.append(("seal",)),
        run=lambda seqs, capture_sampling, replay_after_capture: calls.append(
            ("capture", seqs, capture_sampling, replay_after_capture)
        ),
    )
    runner.collective_runtime = SimpleNamespace(
        begin_cuda_graph_capture=lambda: calls.append(("begin",)),
        collect_local_cuda_graph_metadata=lambda: calls.append(("collect",)),
        exchange_cuda_graph_metadata=lambda: calls.append(("exchange",)),
        register_cuda_graph_buffers=lambda: calls.append(("register",)),
        mark_cuda_graph_replayable=lambda: calls.append(("replayable",)),
    )
    seqs = [object()]

    with patch(
        "sparseengine.engine.model_runner.reset_context",
        side_effect=lambda: calls.append(("reset",)),
    ):
        ModelRunner.begin_decode_cuda_graph_capture(runner)
        ModelRunner.collect_decode_cuda_graph_metadata(runner)
        ModelRunner.exchange_decode_cuda_graph_metadata(runner)
        ModelRunner.register_decode_cuda_graph_buffers(runner)
        ModelRunner.seal_decode_cuda_graph_startup_plan(runner)
        ModelRunner.capture_decode_cuda_graph_warmup(runner, seqs)

    assert calls == [
        ("begin",),
        ("collect",),
        ("exchange",),
        ("register",),
        ("replayable",),
        ("seal",),
        ("capture", seqs, False, False),
        ("reset",),
    ]


def test_model_runner_exit_drains_graphs_before_barrier():
    calls = []
    runner = _runner()
    runner.device = torch.device("cuda:1")
    runner.dp_idle_graphs = {1: object()}
    runner.platform = SimpleNamespace(
        set_device=lambda device: calls.append(("device", device)),
        synchronize=lambda: calls.append("sync"),
        barrier_device_ids=lambda rank: [rank],
    )
    runner.config = SimpleNamespace(decode_graph=True)
    runner.decode_graph_runner = SimpleNamespace(
        clear_captured_graphs=lambda: calls.append("clear_graphs")
    )
    runner.model = SimpleNamespace(
        close_runtime_operators=lambda: calls.append("close_ops")
    )
    runner.collective_runtime = SimpleNamespace(
        close=lambda: calls.append("close_collectives")
    )
    runner.parallel_context.attn_tp_size = 2
    runner.rank = 0
    runner.parallel_context.attn_tp_rank = 0
    runner.shm = SimpleNamespace(
        close=lambda: calls.append("close_shm"),
        unlink=lambda: calls.append("unlink_shm"),
    )
    runner.parallel_context = SimpleNamespace(
        attn_tp_size=2, attn_tp_rank=0,
        attn_tp=SimpleNamespace(barrier=lambda **_: calls.append("barrier")),
    )

    with (
        patch(
            "sparseengine.engine.model_runner.reset_parallel_context",
            side_effect=lambda: calls.append("reset"),
        ),
        patch(
            "sparseengine.engine.model_runner.dist.destroy_process_group",
            side_effect=lambda: calls.append("destroy"),
        ),
    ):
        ModelRunner.exit(runner)

    assert calls == [
        ("device", runner.device),
        "sync",
        "clear_graphs",
        "sync",
        "close_ops",
        "sync",
        "close_collectives",
        "sync",
        "close_shm",
        "barrier",
        "unlink_shm",
        "reset",
        "destroy",
    ]
    assert not runner.dp_idle_graphs


def test_tp_worker_decode_skips_rank0_sampling_path():
    calls: list[str] = []

    runner = _runner()
    runner.rank = 1
    runner.parallel_context.attn_tp_rank = 1
    runner.config = SimpleNamespace(decode_graph=False)
    runner.decode_graph_runner = SimpleNamespace(
        run_eager_static=lambda seqs: calls.append("decode") or None
    )
    runner.sparse_controller = SimpleNamespace(
        post_forward=lambda seqs, is_prefill: calls.append(f"sparse_post:{is_prefill}")
    )
    runner.runtime_state = SimpleNamespace(
        on_forward_end=lambda seqs, is_prefill: calls.append(f"cache_post:{is_prefill}")
    )
    runner.sampler = lambda *args, **kwargs: calls.append("sample")
    runner._collect_logprobs = lambda *args, **kwargs: calls.append("logprobs")

    token_ids, logprobs = ModelRunner.run(runner, [SimpleNamespace()], is_prefill=False)

    assert token_ids is None
    assert logprobs is None
    assert calls == ["decode", "sparse_post:False", "cache_post:False"]


def test_prefix_batch_uses_one_combined_result_boundary():
    runner = _runner()
    calls = []
    runner.refresh_prefix_cache_hit = lambda seq: calls.append(seq) or {"hit_len": seq}
    runner._sync_prefix_cache_batch_result = lambda result, error: calls.append((result, error))
    runner._sync_tp_rpc_status = lambda *_args: pytest.fail("redundant status collective")
    assert runner.call("refresh_prefix_cache_hits", [3, 7]) == [{"hit_len": 3}, {"hit_len": 7}]
    assert calls == [3, 7, ([{"hit_len": 3}, {"hit_len": 7}], None)]


def test_prefix_batch_partial_failure_still_enters_result_boundary():
    runner = _runner()
    error = ValueError("lookup failed")
    calls = []

    def lookup(seq):
        if seq == 2:
            raise error
        return {"hit_len": seq}

    def check(result, local_error):
        calls.append((result, local_error))
        raise local_error

    runner.refresh_prefix_cache_hit = lookup
    runner._sync_prefix_cache_batch_result = check
    with pytest.raises(ValueError, match="lookup failed"):
        runner.call("refresh_prefix_cache_hits", [1, 2, 3])
    assert calls == [(None, error)]


def test_prefix_batch_splits_before_publishing_oversized_payload():
    import pickle

    runner = _runner()
    runner.parallel_context = SimpleNamespace(attn_tp_size=2, attn_tp_rank=0)
    runner.shm = SimpleNamespace(buf=bytearray(300))
    signalled = []
    command = SimpleNamespace(set=lambda: signalled.append(bytes(runner.shm.buf)))
    completion = SimpleNamespace(clear=lambda: None)
    runner.event = [(command, completion)]
    runner.refresh_prefix_cache_hit = lambda seq: {"hit_len": seq[0]}
    runner._sync_prefix_cache_batch_result = lambda *_args: None
    seqs = [(i, bytes([i]) * 100) for i in range(6)]
    assert runner.call("refresh_prefix_cache_hits", seqs) == [{"hit_len": i} for i in range(6)]
    received = []
    for buf in signalled:
        n = int.from_bytes(buf[:4], "little")
        name, batch = pickle.loads(buf[4:n + 4])
        assert name == "refresh_prefix_cache_hits"
        received.extend(batch)
    assert received == seqs
    assert 1 < len(signalled) <= len(seqs)


def _prefix_batch_gloo_worker(rank, rendezvous, output):
    from datetime import timedelta

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=20))
    try:
        runner = _runner()
        runner.parallel_context = SimpleNamespace(
            attn_tp_size=2, attn_tp_rank=rank,
            attn_tp=SimpleNamespace(process_group=dist.group.WORLD),
        )
        runner.tp_control_group = dist.group.WORLD
        outcomes = []
        for case in ("success", "divergence", "failure"):
            result = [{"hit_len": 8}, {"hit_len": 16}]
            error = None
            if case == "divergence" and rank == 1:
                result[1]["hit_len"] = 0
            if case == "failure" and rank == 1:
                error = ValueError("worker lookup failed")
                result = None
            try:
                runner._sync_prefix_cache_batch_result(result, error)
                outcomes.append((case, "success"))
            except (ValueError, RuntimeError) as exc:
                outcomes.append((case, str(exc)))
        output.put((rank, outcomes))
    finally:
        dist.destroy_process_group()


def test_prefix_batch_real_two_rank_result_and_failure_agreement(tmp_path):
    ctx = get_context("spawn")
    output = ctx.Queue()
    rendezvous = (tmp_path / "gloo_init").as_uri()
    workers = [ctx.Process(target=_prefix_batch_gloo_worker, args=(rank, rendezvous, output))
               for rank in range(2)]
    try:
        for worker in workers:
            worker.start()
        outcomes = dict(output.get(timeout=40) for _ in workers)
        for worker in workers:
            worker.join(timeout=10)
            assert worker.exitcode == 0
        for rank in range(2):
            results = dict(outcomes[rank])
            assert results["success"] == "success"
            assert "diverged" in results["divergence"]
            assert "worker lookup failed" in results["failure"]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        output.close()
