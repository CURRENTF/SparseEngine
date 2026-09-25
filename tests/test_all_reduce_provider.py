import os
import sys
from dataclasses import replace
from io import StringIO
from types import ModuleType, SimpleNamespace
from unittest.mock import ANY, Mock, patch

import pytest
import torch

from sparseengine.kernels.external.support import (
    KernelFamilyHealth,
    KernelFamilyState,
    RequiredExternalKernelFamilyError,
)
from sparseengine.operators.all_reduce import (
    ALL_REDUCE_REGISTRY,
    AllReduceGraphBufferMetadata,
    AllReduceOpSpec,
    FlashInferTrtllmAllReduceProvider,
    FlashInferVllmAllReduceProvider,
    PreparedAllReduceOp,
    TorchDistributedAllReduceProvider,
    _expandable_segments_enabled,
    _flashinfer_dependency_support,
    _loaded_cuda_runtime_path,
)
from sparseengine.operators.registry import OpResolver, SupportResult
from sparseengine.platforms import DeviceCaps, PlatformEnum


@pytest.fixture(autouse=True)
def mock_peer_access(monkeypatch):
    # These tests exercise selection/adapter contracts, not physical GPU topology.
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda left, right: True)


def _spec(
    device_ordinals: tuple[int, ...],
    *,
    cuda_graph: bool = False,
) -> AllReduceOpSpec:
    return AllReduceOpSpec(
        world_size=2,
        ranks=(2, 3),
        max_rows=16,
        hidden_size=2048,
        dtype=torch.bfloat16,
        cuda_graph=cuda_graph,
        backend="nccl",
        device_ordinals=device_ordinals,
    )


def _caps() -> DeviceCaps:
    return DeviceCaps(
        platform=PlatformEnum.CUDA,
        device_type="cuda",
        device_index=0,
        device_name="NVIDIA H100 80GB HBM3",
        compute_capability=(9, 0),
        runtime_version="13.0",
        supports_graph_capture=True,
        supports_triton=True,
        supports_bfloat16=True,
    )


def _fake_flashinfer_modules() -> dict[str, ModuleType]:
    comm = ModuleType("flashinfer.comm")
    for name in (
        "CudaRTLibrary",
        "create_shared_buffer",
        "vllm_all_reduce",
        "vllm_dispose",
        "vllm_get_graph_buffer_ipc_meta",
        "vllm_init_custom_ar",
        "vllm_meta_size",
        "vllm_register_buffer",
        "vllm_register_graph_buffers",
    ):
        setattr(comm, name, object())
    flashinfer = ModuleType("flashinfer")
    flashinfer.comm = comm
    cuda_ipc = ModuleType("flashinfer.comm.cuda_ipc")
    cuda_ipc.cudart = SimpleNamespace(_library=None)
    return {
        "flashinfer": flashinfer,
        "flashinfer.comm": comm,
        "flashinfer.comm.cuda_ipc": cuda_ipc,
    }


def test_all_reduce_rejects_missing_required_flashinfer() -> None:
    error = RequiredExternalKernelFamilyError(
        KernelFamilyHealth(
            family="flashinfer-python",
            state=KernelFamilyState.ABSENT,
            version=None,
            reason="flashinfer-python is not installed",
        ),
        feature="communication",
    )
    with (
        patch(
            "sparseengine.operators.all_reduce.flashinfer_kernel_support",
            side_effect=error,
        ),
        pytest.raises(
            RequiredExternalKernelFamilyError,
            match=r'pip install -e "\.\[cu130\]"',
        ),
    ):
        _flashinfer_dependency_support()


def test_vllm_all_reduce_uses_explicit_cuda_ordinals_not_global_ranks():
    with (
        patch.dict(sys.modules, _fake_flashinfer_modules()),
        patch(
            "sparseengine.operators.all_reduce._flashinfer_dependency_support",
            return_value=SupportResult.yes("test dependency"),
        ),
    ):
        resolved = OpResolver(ALL_REDUCE_REGISTRY).resolve(_spec((0, 1)), _caps())

    assert isinstance(resolved.provider, FlashInferVllmAllReduceProvider)


def test_vllm_all_reduce_rejects_duplicate_ordinals_from_multiple_hosts():
    with (
        patch.dict(sys.modules, _fake_flashinfer_modules()),
        patch(
            "sparseengine.operators.all_reduce._flashinfer_dependency_support",
            return_value=SupportResult.yes("test dependency"),
        ),
    ):
        resolved = OpResolver(ALL_REDUCE_REGISTRY).resolve(_spec((0, 0)), _caps())

    assert isinstance(resolved.provider, TorchDistributedAllReduceProvider)


def test_vllm_all_reduce_is_selected_for_cuda_graph_replay():
    with (
        patch.dict(sys.modules, _fake_flashinfer_modules()),
        patch(
            "sparseengine.operators.all_reduce._flashinfer_dependency_support",
            return_value=SupportResult.yes("test dependency"),
        ),
    ):
        resolved = OpResolver(ALL_REDUCE_REGISTRY).resolve(
            _spec((0, 1), cuda_graph=True),
            _caps(),
        )

    assert isinstance(resolved.provider, FlashInferVllmAllReduceProvider)


def test_vllm_all_reduce_registers_captured_graph_buffers_across_ranks():
    calls = []
    comm = _fake_flashinfer_modules()["flashinfer.comm"]
    comm.vllm_all_reduce = lambda *args: calls.append(("run", args))
    comm.vllm_get_graph_buffer_ipc_meta = lambda handle: ([10, 11], [100, 200])
    comm.vllm_register_graph_buffers = (
        lambda handle, handles, offsets: calls.append(
            ("register", handle, handles, offsets)
        )
    )
    flashinfer = ModuleType("flashinfer")
    flashinfer.comm = comm
    provider = FlashInferVllmAllReduceProvider()
    provider._handle = 7
    provider._group = "tp"
    provider._rank = 0
    provider._buffer_ptrs = [1000, 2000]
    provider._max_size_bytes = 4096

    with (
        patch.dict(
            sys.modules,
            {"flashinfer": flashinfer, "flashinfer.comm": comm},
        ),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
    ):
        output = provider.run(
            _spec((0, 1), cuda_graph=True),
            torch.ones((2, 2048), dtype=torch.bfloat16),
            group="tp",
        )
        local_metadata = provider.collect_local_cuda_graph_metadata(
            _spec((0, 1), cuda_graph=True),
            group="tp",
        )
        provider.register_cuda_graph_buffers(
            _spec((0, 1), cuda_graph=True),
            [
                local_metadata,
                AllReduceGraphBufferMetadata(
                    handles=(20, 21),
                    offsets=(100, 200),
                ),
            ],
            group="tp",
        )

    assert output.shape == (2, 2048)
    assert calls == [
        ("run", (7, ANY, ANY, 0, 0, 32)),
        (
            "register",
            7,
            [[10, 11], [20, 21]],
            [[100, 200], [100, 200]],
        ),
    ]


def test_vllm_all_reduce_rejects_expandable_segments_for_cuda_graph():
    provider = FlashInferVllmAllReduceProvider()
    with patch.dict(
        os.environ,
        {"PYTORCH_ALLOC_CONF": "max_split_size_mb:64,expandable_segments:True"},
        clear=True,
    ):
        assert _expandable_segments_enabled()
        with pytest.raises(RuntimeError, match="expandable segments"):
            provider.prepare(
                _spec((0, 1), cuda_graph=True),
                group="tp",
                rank=0,
                device_index=0,
            )


def test_vllm_all_reduce_ignores_loaded_cuda_runtime_stub():
    mappings = """\
7f000000-7f001000 r-xp 0 00:00 0 /env/tilelang/lib/libcudart_stub.so
7f002000-7f003000 r-xp 0 00:00 0 /env/nvidia/cu13/lib/libcudart.so.13
"""
    with patch("builtins.open", return_value=StringIO(mappings)):
        assert _loaded_cuda_runtime_path() == "/env/nvidia/cu13/lib/libcudart.so.13"


def test_vllm_all_reduce_rejects_cuda_runtime_stub_only():
    mappings = """\
7f000000-7f001000 r-xp 0 00:00 0 /env/tilelang/lib/libcudart_stub.so
"""
    with (
        patch("builtins.open", return_value=StringIO(mappings)),
        pytest.raises(RuntimeError, match="only compiler stubs"),
    ):
        _loaded_cuda_runtime_path()


@pytest.mark.parametrize("rank", [0, 1])
def test_vllm_all_reduce_uses_selected_runtime_for_shared_buffer_lifecycle(rank):
    modules = _fake_flashinfer_modules()
    comm = modules["flashinfer.comm"]
    proxy = modules["flashinfer.comm.cuda_ipc"].cudart
    runtime = Mock()
    runtime.lib.cudaIpcCloseMemHandle.return_value = 0
    comm.CudaRTLibrary = Mock(return_value=runtime)
    allocations = iter(((1000, 3000), (2000, 4000)))

    def create_shared_buffer(size, group):
        assert proxy._library is runtime
        assert group == "tp"
        pointers = list(next(allocations))
        return pointers if rank == 0 else pointers[::-1]

    comm.create_shared_buffer = Mock(side_effect=create_shared_buffer)
    comm.vllm_meta_size = Mock(return_value=64)
    comm.vllm_init_custom_ar = Mock(return_value=7)
    comm.vllm_register_buffer = Mock()
    comm.vllm_dispose = Mock()
    mappings = """\
7f000000-7f001000 r-xp 0 00:00 0 /env/tilelang/lib/libcudart_stub.so
7f002000-7f003000 r-xp 0 00:00 0 /env/nvidia/cu13/lib/libcudart.so.13
"""

    provider = FlashInferVllmAllReduceProvider()
    rank_data = torch.empty(0, dtype=torch.uint8)
    with (
        patch.dict(sys.modules, modules),
        patch("builtins.open", side_effect=lambda *a, **kw: StringIO(mappings)),
        patch.object(torch.cuda, "current_device", return_value=rank),
        patch.object(torch.cuda, "can_device_access_peer", return_value=True),
        patch.object(torch, "empty", return_value=rank_data),
        patch.object(torch.distributed, "barrier"),
    ):
        spec = _spec((0, 1))
        provider.prepare(spec, group="tp", rank=rank, device_index=rank)
        comm.CudaRTLibrary.assert_called_once_with("/env/nvidia/cu13/lib/libcudart.so.13")
        assert modules["flashinfer.comm.cuda_ipc"].cudart is proxy
        assert comm.create_shared_buffer.call_count == 2
        row_bytes = spec.hidden_size * spec.dtype.itemsize
        staging_bytes = comm.create_shared_buffer.call_args_list[1].args[0]
        assert staging_bytes == provider.prefill_staging_max_bytes
        assert provider.prefill_reuse_capacity(spec) * row_bytes <= staging_bytes
        assert provider._meta_ptrs[rank] == 1000
        assert provider._meta_ptrs[1 - rank] == 3000
        assert provider._buffer_ptrs[rank] == 2000
        assert provider._buffer_ptrs[1 - rank] == 4000

        provider.close()
        provider.close()

    comm.vllm_dispose.assert_called_once_with(7)
    cleanup = [
        (name, args[0].value)
        for name, args, _ in runtime.mock_calls
        if name in {"lib.cudaIpcCloseMemHandle", "cudaFree"}
    ]
    assert cleanup == [
        ("lib.cudaIpcCloseMemHandle", 4000),
        ("cudaFree", 2000),
        ("lib.cudaIpcCloseMemHandle", 3000),
        ("cudaFree", 1000),
    ]


def test_vllm_all_reduce_rejects_missing_runtime_before_allocating():
    modules = _fake_flashinfer_modules()
    comm = modules["flashinfer.comm"]
    comm.CudaRTLibrary = Mock()
    comm.create_shared_buffer = Mock()
    provider = FlashInferVllmAllReduceProvider()
    with (
        patch.dict(sys.modules, modules),
        patch("builtins.open", return_value=StringIO("")),
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(torch.cuda, "can_device_access_peer", return_value=True),
        patch.object(torch, "empty") as allocate,
        pytest.raises(RuntimeError, match="no libcudart mapping"),
    ):
        provider.prepare(_spec((0, 1)), group="tp", rank=0, device_index=0)

    comm.CudaRTLibrary.assert_not_called()
    comm.create_shared_buffer.assert_not_called()
    allocate.assert_not_called()


def test_trtllm_all_reduce_binds_runtime_before_workspace_allocation():
    modules = _fake_flashinfer_modules()
    comm = modules["flashinfer.comm"]
    proxy = modules["flashinfer.comm.cuda_ipc"].cudart
    runtime = Mock()
    comm.CudaRTLibrary = Mock(return_value=runtime)
    workspace = Mock()

    def allocate_workspace(**kwargs):
        assert proxy._library is runtime
        return workspace

    comm.create_allreduce_fusion_workspace = Mock(side_effect=allocate_workspace)
    provider = FlashInferTrtllmAllReduceProvider()
    with (
        patch.dict(sys.modules, modules),
        patch(
            "sparseengine.operators.all_reduce._loaded_cuda_runtime_path",
            return_value="/env/nvidia/cu13/lib/libcudart.so.13",
        ),
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(torch.distributed, "get_backend", return_value="nccl"),
        patch("sparseengine.operators.all_reduce.platforms.current_platform") as platform,
        patch(
            "sparseengine.operators.all_reduce._flashinfer_trtllm_profile",
            return_value=SimpleNamespace(max_rows=16, provider_output_buffer=False),
        ),
    ):
        platform.get_device_caps.return_value = _caps()
        provider.prepare(_spec((0, 1), cuda_graph=True), group="tp", rank=0)
        provider.close()

    comm.CudaRTLibrary.assert_called_once_with("/env/nvidia/cu13/lib/libcudart.so.13")
    comm.create_allreduce_fusion_workspace.assert_called_once()
    workspace.destroy.assert_called_once()


def test_vllm_all_reduce_stages_only_eager_input():
    calls = []
    comm = _fake_flashinfer_modules()["flashinfer.comm"]
    comm.vllm_all_reduce = lambda *args: calls.append(args)
    flashinfer = ModuleType("flashinfer")
    flashinfer.comm = comm
    provider = FlashInferVllmAllReduceProvider()
    provider._handle = 7
    provider._rank = 0
    provider._buffer_ptrs = [1000, 2000]
    provider._max_size_bytes = 4096
    with patch.dict(sys.modules, {"flashinfer": flashinfer, "flashinfer.comm": comm}):
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            graph_output = provider.run(
                _spec((0, 1), cuda_graph=True),
                torch.ones((2, 2048), dtype=torch.bfloat16),
                group="tp",
            )
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
            eager_output = provider.run(
                _spec((0, 1), cuda_graph=True),
                torch.ones((2, 2048), dtype=torch.bfloat16),
                group="tp",
            )

    assert graph_output.shape == eager_output.shape == (2, 2048)
    assert calls == [
        (7, ANY, ANY, 0, 0, 32),
        (7, ANY, ANY, 1000, 4096, 32),
    ]


@pytest.mark.parametrize("world_size,full_nvlink", [(2, False), (4, True), (4, False)])
def test_vllm_all_reduce_initializes_the_upstream_nvlink_mode(world_size, full_nvlink):
    # For >2 ranks the upstream kernel launches only with full_nvlink=True.
    modules = _fake_flashinfer_modules()
    comm = modules["flashinfer.comm"]
    comm.create_shared_buffer = Mock(return_value=list(range(world_size)))
    comm.vllm_meta_size = Mock(return_value=64)
    comm.vllm_init_custom_ar = Mock(return_value=7)
    comm.vllm_register_buffer = Mock()
    topology = Mock(return_value=full_nvlink)
    spec = replace(
        _spec((0, 1)),
        world_size=world_size,
        ranks=tuple(range(world_size)),
        device_ordinals=tuple(range(world_size)),
    )
    rank_data = torch.empty(0, dtype=torch.uint8)
    with (
        patch.dict(sys.modules, modules),
        patch(
            "sparseengine.operators.all_reduce.platforms",
            SimpleNamespace(
                current_platform=SimpleNamespace(supports_nvlink_group=topology)
            ),
        ),
        patch("sparseengine.operators.all_reduce._prepare_flashinfer_cuda_runtime"),
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(torch.cuda, "can_device_access_peer", return_value=True),
        patch.object(torch, "empty", return_value=rank_data),
    ):
        provider = FlashInferVllmAllReduceProvider()
        if world_size > 2 and not full_nvlink:
            with pytest.raises(RuntimeError, match="fully connected NVLink"):
                provider.prepare(spec, group="tp", rank=0)
            comm.create_shared_buffer.assert_not_called()
            comm.vllm_init_custom_ar.assert_not_called()
        else:
            provider.prepare(spec, group="tp", rank=0)
            comm.vllm_init_custom_ar.assert_called_once_with(
                list(range(world_size)), rank_data, 0, full_nvlink
            )
    if world_size == 2:
        topology.assert_not_called()
    else:
        topology.assert_called_once_with(spec.device_ordinals)


def test_vllm_all_reduce_rejects_odd_rank_count_before_loading_extension():
    spec = replace(
        _spec((0, 1)), world_size=3, ranks=(0, 1, 2), device_ordinals=(0, 1, 2)
    )
    with patch(
        "sparseengine.operators.all_reduce._flashinfer_dependency_support"
    ) as dependency:
        assert not FlashInferVllmAllReduceProvider.supports(spec, _caps()).supported
    dependency.assert_not_called()


@pytest.mark.parametrize("cuda_graph", [False, True])
def test_vllm_prefill_reuse_is_bounded_by_prepared_staging(cuda_graph):
    spec = _spec((0, 1), cuda_graph=cuda_graph)
    provider = FlashInferVllmAllReduceProvider()
    for shape in (spec, replace(spec, hidden_size=3072)):
        rows = provider.prefill_reuse_capacity(shape)
        row_bytes = shape.hidden_size * shape.dtype.itemsize
        assert rows > shape.max_rows
        assert rows * row_bytes <= provider.prefill_staging_max_bytes
        assert (rows + 1) * row_bytes > provider.prefill_staging_max_bytes
    assert TorchDistributedAllReduceProvider().prefill_reuse_capacity(spec) == 0


def test_extended_prefill_contract_preserves_decode_shape_limit():
    spec = replace(_spec((0, 1)), max_rows=16)
    provider = FlashInferVllmAllReduceProvider()
    provider.run = Mock(side_effect=lambda _spec, tensor, *, group: tensor.clone())
    op = PreparedAllReduceOp(spec, provider, group=None)
    larger_prefill = torch.ones((256, spec.hidden_size), dtype=spec.dtype)
    assert torch.equal(op.run_prefill(larger_prefill), larger_prefill)
    with pytest.raises(ValueError, match="outside the prepared range"):
        op.run(larger_prefill)
    with pytest.raises(ValueError, match="outside the prepared range"):
        op.run_prefill(
            torch.ones((op.prefill_max_rows + 1, spec.hidden_size), dtype=spec.dtype)
        )


@pytest.mark.parametrize("world_size", [2, 4, 6, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_vllm_all_reduce_is_upstream_default_beyond_local_profiles(
    world_size, dtype, cuda_graph
):
    spec = replace(
        _spec((0, 1), cuda_graph=cuda_graph),
        world_size=world_size, ranks=tuple(range(world_size)),
        device_ordinals=tuple(range(world_size)),
        dtype=dtype, hidden_size=4096, max_rows=2048,
    )
    caps = replace(_caps(), device_name="NVIDIA A100", compute_capability=(8, 0))
    with (
        patch.dict(sys.modules, _fake_flashinfer_modules()),
        patch("sparseengine.operators.all_reduce._flashinfer_dependency_support",
              return_value=SupportResult.yes("test dependency")),
        patch("sparseengine.operators.all_reduce.importlib.util.find_spec",
              return_value=object()),
        patch("sparseengine.operators.all_reduce.platforms.current_platform") as platform,
    ):
        platform.supports_nvlink_group.return_value = True
        resolved = OpResolver(ALL_REDUCE_REGISTRY).resolve(spec, caps)
    assert isinstance(resolved.provider, FlashInferVllmAllReduceProvider)
    assert resolved.report.selection_basis == "upstream_default"


@pytest.mark.parametrize("dtype,hidden", [(torch.float16, 7), (torch.float32, 5),
                                          (torch.int32, 2048)])
def test_vllm_all_reduce_rejects_invalid_packing_or_dtype(dtype, hidden):
    # max_rows=16 makes the entire maximum tensor aligned even when a row is not.
    spec = replace(_spec((0, 1)), dtype=dtype, hidden_size=hidden)
    with patch("sparseengine.operators.all_reduce._flashinfer_dependency_support") as dep:
        result = FlashInferVllmAllReduceProvider.supports(spec, _caps())
    assert not result.supported
    dep.assert_not_called()


@pytest.mark.parametrize("peer_access,nvlink", [(False, True), (True, False)])
def test_vllm_all_reduce_falls_back_before_preparation_for_unsupported_fabric(
    peer_access, nvlink
):
    spec = replace(_spec((0, 1)), world_size=4, ranks=(0, 1, 2, 3),
                   device_ordinals=(0, 1, 2, 3))
    caps = replace(_caps(), device_name="NVIDIA A100", compute_capability=(8, 0))
    with (
        patch.object(torch.cuda, "can_device_access_peer", return_value=peer_access),
        patch("sparseengine.operators.all_reduce.importlib.util.find_spec",
              return_value=object()),
        patch("sparseengine.operators.all_reduce.platforms.current_platform") as platform,
    ):
        platform.supports_nvlink_group.return_value = nvlink
        resolved = OpResolver(ALL_REDUCE_REGISTRY).resolve(spec, caps)
    assert isinstance(resolved.provider, TorchDistributedAllReduceProvider)


def test_vllm_all_reduce_accepts_large_two_rank_pcie_capacity():
    spec = replace(_spec((0, 1)), max_rows=2048, hidden_size=8192)
    with (
        patch.dict(sys.modules, _fake_flashinfer_modules()),
        patch("sparseengine.operators.all_reduce._flashinfer_dependency_support",
              return_value=SupportResult.yes("test dependency")),
        patch("sparseengine.operators.all_reduce.platforms.current_platform") as platform,
    ):
        assert FlashInferVllmAllReduceProvider.supports(spec, _caps()).supported
    platform.supports_nvlink_group.assert_not_called()


@pytest.mark.parametrize("change", ["non_cuda", "no_bf16", "element_overflow"])
def test_vllm_all_reduce_rejects_unsupported_device_or_size_before_dependencies(change):
    spec, caps = _spec((0, 1)), _caps()
    if change == "non_cuda":
        caps = replace(caps, platform=PlatformEnum.CPU)
    elif change == "no_bf16":
        caps = replace(caps, supports_bfloat16=False)
    else:
        spec = replace(spec, max_rows=2**31)
    with patch("sparseengine.operators.all_reduce._flashinfer_dependency_support") as dep:
        assert not FlashInferVllmAllReduceProvider.supports(spec, caps).supported
    dep.assert_not_called()
