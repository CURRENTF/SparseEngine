from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sparseengine.kernels.external.support import (
    ExternalKernelContractError,
    ExternalKernelFamilyError,
    KernelFamilyState,
    RequiredExternalKernelFamilyError,
)
from sparseengine.kernels.external.sgl.fa3 import (
    _FWD_ARGUMENTS,
    SglFa3DecodeKernel,
    sgl_fa3_device_support,
    sgl_fa3_support,
)
from sparseengine.kernels.external.sgl.support import sgl_kernel_metadata_health


def test_sgl_fa3_support_rejects_missing_package() -> None:
    with patch("importlib.util.find_spec", return_value=None):
        with pytest.raises(ExternalKernelFamilyError) as exc_info:
            sgl_fa3_support()

    assert exc_info.value.health.state is KernelFamilyState.ABSENT
    assert isinstance(exc_info.value, RequiredExternalKernelFamilyError)
    assert "sglang-kernel is not installed" in str(exc_info.value)
    assert 'pip install -e ".[cu129]"' in str(exc_info.value)


def test_sgl_metadata_health_does_not_import_device_bound_ops() -> None:
    with (
        patch("importlib.util.find_spec", return_value=object()),
        patch("importlib.metadata.version", return_value="0.4.5"),
        patch("importlib.import_module", side_effect=AssertionError("unexpected import")),
    ):
        health = sgl_kernel_metadata_health()

    assert health.state is KernelFamilyState.READY


@pytest.mark.parametrize("version", ["0.4.4", "0.4.5.post1", "0.4.6.post1"])
def test_sgl_fa3_support_rejects_unpinned_version(version: str) -> None:
    with (
        patch("importlib.util.find_spec", return_value=object()),
        patch("importlib.metadata.version", return_value=version),
    ):
        with pytest.raises(ExternalKernelFamilyError) as exc_info:
            sgl_fa3_support()

    assert exc_info.value.health.state is KernelFamilyState.BROKEN
    assert "sglang-kernel==0.4.5" in str(exc_info.value)


def test_sgl_fa3_support_accepts_pinned_version() -> None:
    version = "0.4.5"
    op = SimpleNamespace(
        _schema=SimpleNamespace(
            arguments=[
                SimpleNamespace(name=name)
                for name in _FWD_ARGUMENTS
            ]
        )
    )
    with (
        patch(
            "sparseengine.kernels.external.sgl.fa3._sgl_fa3_op",
            return_value=op,
        ),
        patch("importlib.util.find_spec", return_value=object()),
        patch("importlib.metadata.version", return_value=version),
        patch("importlib.import_module", return_value=object()),
    ):
        supported, reason = sgl_fa3_support()

    assert supported
    assert version in reason


def test_sgl_fa3_support_rejects_binary_load_failure() -> None:
    with (
        patch("importlib.util.find_spec", return_value=object()),
        patch("importlib.metadata.version", return_value="0.4.5"),
        patch(
            "importlib.import_module",
            side_effect=ImportError("undefined symbol: c10_cuda_check"),
        ),
    ):
        with pytest.raises(ExternalKernelFamilyError) as exc_info:
            sgl_fa3_support()

    assert exc_info.value.health.state is KernelFamilyState.BROKEN
    assert "undefined symbol" in str(exc_info.value)


def test_sgl_fa3_support_rejects_missing_op_schema() -> None:
    with (
        patch(
            "sparseengine.kernels.external.sgl.fa3.sgl_kernel_support",
            return_value=(True, "available"),
        ),
        patch(
            "sparseengine.kernels.external.sgl.fa3._sgl_fa3_op",
            return_value=object(),
        ),
    ):
        with pytest.raises(ExternalKernelContractError) as exc_info:
            sgl_fa3_support()

    assert "failed to load" in str(exc_info.value)


def test_sgl_fa3_device_support_keeps_package_probe_and_device_probe_separate() -> None:
    with patch(
        "sparseengine.kernels.external.sgl.fa3.sgl_fa3_support",
        side_effect=RuntimeError("ABI mismatch"),
    ):
        with pytest.raises(RuntimeError, match="ABI mismatch"):
            sgl_fa3_device_support(3)


def _mock_fa3_kernel(*, workspace: torch.Tensor | None) -> tuple[
    SglFa3DecodeKernel,
    Mock,
    Mock,
    torch.Tensor,
]:
    raw_op = Mock()
    raw_op.side_effect = lambda *args: (
        args[_FWD_ARGUMENTS.index("out")],
        torch.empty(12, 155, dtype=torch.float32),
        workspace,
        torch.empty(0),
    )
    metadata = torch.empty(17, dtype=torch.int32)
    scheduler_op = Mock(return_value=metadata)
    kernel = object.__new__(SglFa3DecodeKernel)
    kernel._op = raw_op
    kernel._scheduler_op = scheduler_op
    kernel._scheduler_plan = None
    kernel._captured_scheduler_plans = []
    kernel.softmax_scale = 128**-0.5
    kernel.num_splits = 0
    return kernel, raw_op, scheduler_op, metadata


def _call_arguments(call) -> dict[str, object]:
    return dict(zip(_FWD_ARGUMENTS, call.args, strict=True))


def test_sgl_fa3_ragged_prefill_binds_metadata_to_forward_split() -> None:
    # MiniMax-M2.7 TP4 production contract: the real query total is 155, not
    # batch * max_seqlen_q (278), and the 8192-wide table crosses the upstream
    # automatic split heuristic boundary on a 132-SM H100.
    kernel, raw_op, scheduler_op, metadata = _mock_fa3_kernel(
        workspace=torch.empty(8, 1),
    )
    q = torch.empty(155, 12, 128, dtype=torch.bfloat16)
    k_cache = torch.empty(1, 2, 128, dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    page_table = torch.empty(2, 8192, dtype=torch.int32)
    request_indices = torch.tensor([0, 1], dtype=torch.int32)
    context_lens = torch.tensor([256, 2048], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 139, 155], dtype=torch.int32)
    output = torch.empty_like(q)
    scope = object()
    assert q.shape[0] != context_lens.numel() * 139

    with patch("torch.cuda.is_current_stream_capturing", return_value=False):
        kernel.run_explicit_varlen(
            q,
            k_cache,
            v_cache,
            page_table,
            request_indices,
            context_lens,
            output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=139,
            validation_scope=scope,
        )
        kernel.run_explicit_varlen(
            q,
            k_cache,
            v_cache,
            page_table,
            request_indices,
            context_lens,
            output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=139,
            validation_scope=scope,
        )

    first_fwd = _call_arguments(raw_op.call_args_list[0])
    second_fwd = _call_arguments(raw_op.call_args_list[1])
    assert first_fwd["scheduler_metadata"] is None
    assert first_fwd["num_splits"] == 0
    assert scheduler_op.call_count == 1
    assert scheduler_op.call_args.args[21] == 8
    assert second_fwd["scheduler_metadata"] is metadata
    assert second_fwd["num_splits"] == 8

    # total_q participates in the cached plan contract even when every other
    # metadata pointer and static maximum remains unchanged.
    q_with_different_total = torch.empty(156, 12, 128, dtype=torch.bfloat16)
    with patch("torch.cuda.is_current_stream_capturing", return_value=False):
        kernel.run_explicit_varlen(
            q_with_different_total,
            k_cache,
            v_cache,
            page_table,
            request_indices,
            context_lens,
            torch.empty_like(q_with_different_total),
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=139,
            validation_scope=scope,
        )
    third_fwd = _call_arguments(raw_op.call_args_list[2])
    assert third_fwd["scheduler_metadata"] is None
    assert third_fwd["num_splits"] == 0
    assert scheduler_op.call_count == 2


def test_sgl_fa3_latent_varlen_resolves_missing_workspace_to_one_split() -> None:
    kernel, raw_op, scheduler_op, metadata = _mock_fa3_kernel(workspace=None)
    q_rope = torch.empty(155, 12, 64, dtype=torch.bfloat16)
    q_latent = torch.empty(155, 12, 512, dtype=torch.bfloat16)
    rope_cache = torch.empty(1, 1, 64, dtype=torch.bfloat16)
    latent_cache = torch.empty(1, 1, 512, dtype=torch.bfloat16)
    page_table = torch.empty(2, 8192, dtype=torch.int32)
    request_indices = torch.tensor([0, 1], dtype=torch.int32)
    context_lens = torch.tensor([256, 2048], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 139, 155], dtype=torch.int32)
    output = torch.empty_like(q_latent)
    scope = object()

    with patch("torch.cuda.is_current_stream_capturing", return_value=False):
        for _ in range(2):
            kernel.run_varlen(
                q_rope,
                q_latent,
                rope_cache,
                latent_cache,
                page_table,
                request_indices,
                context_lens,
                output,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=139,
                validation_scope=scope,
            )

    assert scheduler_op.call_count == 1
    assert scheduler_op.call_args.args[21] == 1
    second_fwd = _call_arguments(raw_op.call_args_list[1])
    assert second_fwd["scheduler_metadata"] is metadata
    assert second_fwd["num_splits"] == 1


def test_sgl_fa3_capture_keeps_plans_separate_from_eager_plan() -> None:
    kernel, _, scheduler_op, _ = _mock_fa3_kernel(workspace=None)
    scheduler_op.side_effect = [
        torch.empty(1, dtype=torch.int32),
        torch.empty(2, dtype=torch.int32),
        torch.empty(3, dtype=torch.int32),
    ]
    q = torch.empty(5, 12, 128, dtype=torch.bfloat16)
    k_cache = torch.empty(1, 2, 128, dtype=torch.bfloat16)
    page_table = torch.empty(2, 8, dtype=torch.int32)
    context_lens = torch.tensor([5, 4], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 3, 5], dtype=torch.int32)
    scope = object()
    kwargs = dict(
        headdim_v=128,
        max_seqlen_q=3,
        validation_scope=scope,
    )

    with patch(
        "torch.cuda.is_current_stream_capturing",
        side_effect=[False, True, False, True],
    ):
        eager_metadata, _ = kernel._scheduler_metadata(
            q, k_cache, page_table, context_lens, cu_seqlens_q, num_splits=2, **kwargs
        )
        captured_metadata, _ = kernel._scheduler_metadata(
            q, k_cache, page_table, context_lens, cu_seqlens_q, num_splits=3, **kwargs
        )
        newer_eager_metadata, _ = kernel._scheduler_metadata(
            q, k_cache, page_table, context_lens, cu_seqlens_q, num_splits=4, **kwargs
        )
        reused_capture, reused_split = kernel._scheduler_metadata(
            q, k_cache, page_table, context_lens, cu_seqlens_q, num_splits=3, **kwargs
        )

    assert eager_metadata is not newer_eager_metadata
    assert len(kernel._captured_scheduler_plans) == 1
    assert captured_metadata is reused_capture
    assert reused_split == 3


@pytest.mark.skipif(
    not torch.cuda.is_available() or not sgl_fa3_support()[0],
    reason="CUDA and a validated sglang-kernel are required",
)
def test_sgl_fa3_decode_matches_torch_and_replays_cuda_graph() -> None:
    torch.manual_seed(20260807)
    device = torch.device("cuda")
    batch_size, heads, width = 3, 10, 8
    slots = 4 * width
    q_rope = torch.randn(
        batch_size, heads, 64, device=device, dtype=torch.bfloat16
    )
    q_latent = torch.randn(
        batch_size, heads, 512, device=device, dtype=torch.bfloat16
    )
    rope_cache = torch.randn(
        slots, 1, 64, device=device, dtype=torch.bfloat16
    )
    latent_cache = torch.randn(
        slots, 1, 512, device=device, dtype=torch.bfloat16
    )
    page_table = torch.arange(
        slots, device=device, dtype=torch.int32
    ).view(4, width)
    request_indices = torch.tensor(
        [2, 0, -1], device=device, dtype=torch.int32
    )
    context_lens = torch.tensor(
        [7, 5, 0], device=device, dtype=torch.int32
    )
    output = torch.empty_like(q_latent)
    kernel = SglFa3DecodeKernel(
        device=device,
        max_batch_size=batch_size,
        softmax_scale=256**-0.5,
    )

    validation_scope = object()
    scheduler_op = kernel._scheduler_op
    scheduler_call_count = 0
    if scheduler_op is not None:

        def counted_scheduler_op(*args, **kwargs):
            nonlocal scheduler_call_count
            scheduler_call_count += 1
            return scheduler_op(*args, **kwargs)

        kernel._scheduler_op = counted_scheduler_op
    actual = kernel(
        q_rope,
        q_latent,
        rope_cache,
        latent_cache,
        page_table,
        request_indices,
        context_lens,
        output,
        validation_scope=validation_scope,
    )
    kernel(
        q_rope,
        q_latent,
        rope_cache,
        latent_cache,
        page_table,
        request_indices,
        context_lens,
        output,
        validation_scope=validation_scope,
    )
    assert scheduler_call_count == int(scheduler_op is not None)
    kernel(
        q_rope,
        q_latent,
        rope_cache,
        latent_cache,
        page_table,
        request_indices,
        context_lens,
        output,
        validation_scope=object(),
    )
    assert scheduler_call_count == 2 * int(scheduler_op is not None)
    expected_rows = []
    for batch_index in range(batch_size):
        length = int(context_lens[batch_index].item())
        if length == 0:
            expected_rows.append(torch.zeros_like(q_latent[batch_index]))
            continue
        row = int(request_indices[batch_index].item())
        active = page_table[row, :length].long()
        logits = q_rope[batch_index].float() @ rope_cache[active, 0].float().T
        logits += q_latent[batch_index].float() @ latent_cache[active, 0].float().T
        probs = torch.softmax(logits * (256**-0.5), dim=-1)
        expected_rows.append(
            (probs @ latent_cache[active, 0].float()).to(torch.bfloat16)
        )
    expected = torch.stack(expected_rows)

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    graph = torch.cuda.CUDAGraph()
    graph_output = torch.empty_like(output)
    with torch.cuda.graph(graph):
        kernel(
            q_rope,
            q_latent,
            rope_cache,
            latent_cache,
            page_table,
            request_indices,
            context_lens,
            graph_output,
            validation_scope=object(),
        )
    second_graph = torch.cuda.CUDAGraph()
    second_graph_output = torch.empty_like(output)
    with torch.cuda.graph(second_graph):
        kernel(
            q_rope,
            q_latent,
            rope_cache,
            latent_cache,
            page_table,
            request_indices,
            context_lens,
            second_graph_output,
            validation_scope=object(),
        )
    graph.replay()
    second_graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, expected, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(
        second_graph_output,
        expected,
        rtol=3e-2,
        atol=3e-2,
    )

    request_indices.copy_(
        torch.tensor([1, 3, 0], device=device, dtype=torch.int32)
    )
    context_lens.copy_(
        torch.tensor([3, 8, 6], device=device, dtype=torch.int32)
    )
    replay_rows = []
    for batch_index in range(batch_size):
        length = int(context_lens[batch_index].item())
        row = int(request_indices[batch_index].item())
        active = page_table[row, :length].long()
        logits = q_rope[batch_index].float() @ rope_cache[active, 0].float().T
        logits += q_latent[batch_index].float() @ latent_cache[active, 0].float().T
        probs = torch.softmax(logits * (256**-0.5), dim=-1)
        replay_rows.append(
            (probs @ latent_cache[active, 0].float()).to(torch.bfloat16)
        )
    replay_expected = torch.stack(replay_rows)
    graph.replay()
    second_graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        graph_output,
        replay_expected,
        rtol=3e-2,
        atol=3e-2,
    )
    torch.testing.assert_close(
        second_graph_output,
        replay_expected,
        rtol=3e-2,
        atol=3e-2,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not sgl_fa3_support()[0],
    reason="CUDA and a validated sglang-kernel are required",
)
def test_sgl_fa3_varlen_latent_prefill_matches_causal_torch() -> None:
    torch.manual_seed(20260807)
    device = torch.device("cuda")
    heads, width = 10, 8
    chunk_lens = (3, 2)
    context_lens = torch.tensor([5, 4], device=device, dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
    query_tokens = int(cu_seqlens_q[-1].item())
    q_rope = torch.randn(
        query_tokens, heads, 64, device=device, dtype=torch.bfloat16
    )
    q_latent = torch.randn(
        query_tokens, heads, 512, device=device, dtype=torch.bfloat16
    )
    rope_cache = torch.randn(
        2 * width, 1, 64, device=device, dtype=torch.bfloat16
    )
    latent_cache = torch.randn(
        2 * width, 1, 512, device=device, dtype=torch.bfloat16
    )
    page_table = torch.arange(
        2 * width, device=device, dtype=torch.int32
    ).view(2, width)
    request_indices = torch.tensor([1, 0], device=device, dtype=torch.int32)
    output = torch.empty_like(q_latent)
    kernel = SglFa3DecodeKernel(
        device=device,
        max_batch_size=2,
        softmax_scale=256**-0.5,
    )

    kernel.run_varlen(
        q_rope,
        q_latent,
        rope_cache,
        latent_cache,
        page_table,
        request_indices,
        context_lens,
        output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max(chunk_lens),
    )
    expected_rows = []
    query_start = 0
    for batch_index, chunk_len in enumerate(chunk_lens):
        context_len = int(context_lens[batch_index].item())
        row = int(request_indices[batch_index].item())
        for query_offset in range(chunk_len):
            visible_len = context_len - chunk_len + query_offset + 1
            active = page_table[row, :visible_len].long()
            query_index = query_start + query_offset
            logits = q_rope[query_index].float() @ rope_cache[active, 0].float().T
            logits += q_latent[query_index].float() @ latent_cache[active, 0].float().T
            probs = torch.softmax(logits * (256**-0.5), dim=-1)
            expected_rows.append(
                (probs @ latent_cache[active, 0].float()).to(torch.bfloat16)
            )
        query_start += chunk_len

    torch.testing.assert_close(
        output,
        torch.stack(expected_rows),
        rtol=3e-2,
        atol=3e-2,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not sgl_fa3_support()[0],
    reason="CUDA and a validated sglang-kernel are required",
)
def test_sgl_fa3_varlen_explicit_prefill_matches_causal_torch() -> None:
    torch.manual_seed(20260807)
    device = torch.device("cuda")
    heads, width, head_dim = 10, 8, 256
    chunk_lens = (3, 2)
    context_lens = torch.tensor([5, 4], device=device, dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
    query_tokens = int(cu_seqlens_q[-1].item())
    q = torch.randn(
        query_tokens, heads, head_dim, device=device, dtype=torch.bfloat16
    )
    k_cache = torch.randn(
        2 * width, heads, head_dim, device=device, dtype=torch.bfloat16
    )
    v_backing = torch.randn(
        2 * width, heads, 448, device=device, dtype=torch.bfloat16
    )
    v_cache = v_backing[..., 192:]
    assert not v_cache.is_contiguous()
    page_table = torch.arange(
        2 * width, device=device, dtype=torch.int32
    ).view(2, width)
    request_indices = torch.tensor([1, 0], device=device, dtype=torch.int32)
    output = torch.empty_like(q)
    kernel = SglFa3DecodeKernel(
        device=device,
        max_batch_size=2,
        softmax_scale=head_dim**-0.5,
    )
    validation_scope = object()

    kernel.run_explicit_varlen(
        q,
        k_cache,
        v_cache,
        page_table,
        request_indices,
        context_lens,
        output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max(chunk_lens),
        validation_scope=validation_scope,
    )
    packed_indices = torch.cat(
        (
            page_table[1, : int(context_lens[0].item())],
            page_table[0, : int(context_lens[1].item())],
        )
    ).long()
    packed_output = torch.empty_like(q)
    kernel.run_contiguous_explicit_varlen(
        q,
        k_cache[packed_indices],
        v_cache[packed_indices],
        packed_output,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=torch.tensor(
            [0, 5, 9], device=device, dtype=torch.int32
        ),
        max_seqlen_q=max(chunk_lens),
        max_seqlen_k=int(context_lens.max().item()),
    )
    expected_rows = []
    query_start = 0
    for batch_index, chunk_len in enumerate(chunk_lens):
        context_len = int(context_lens[batch_index].item())
        row = int(request_indices[batch_index].item())
        for query_offset in range(chunk_len):
            visible_len = context_len - chunk_len + query_offset + 1
            active = page_table[row, :visible_len].long()
            query_index = query_start + query_offset
            logits = torch.einsum(
                "hd,lhd->hl",
                q[query_index].float(),
                k_cache[active].float(),
            )
            probs = torch.softmax(logits * (head_dim**-0.5), dim=-1)
            expected_rows.append(
                torch.einsum("hl,lhd->hd", probs, v_cache[active].float()).to(
                    torch.bfloat16
                )
            )
        query_start += chunk_len

    torch.testing.assert_close(
        output,
        torch.stack(expected_rows),
        rtol=3e-2,
        atol=3e-2,
    )

    # Scheduler metadata is intentionally reusable across layers, while the
    # raw op must consume each layer's own logical-to-physical page table.
    second_page_table = page_table.flip((0, 1))
    second_output = torch.empty_like(q)
    kernel.run_explicit_varlen(
        q,
        k_cache,
        v_cache,
        second_page_table,
        request_indices,
        context_lens,
        second_output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max(chunk_lens),
        validation_scope=validation_scope,
    )
    expected_rows = []
    query_start = 0
    for batch_index, chunk_len in enumerate(chunk_lens):
        context_len = int(context_lens[batch_index].item())
        row = int(request_indices[batch_index].item())
        for query_offset in range(chunk_len):
            visible_len = context_len - chunk_len + query_offset + 1
            active = second_page_table[row, :visible_len].long()
            query_index = query_start + query_offset
            keys = k_cache[active]
            values = v_cache[active]
            logits = torch.einsum("hd,lhd->hl", q[query_index].float(), keys.float())
            probabilities = torch.softmax(logits * (head_dim**-0.5), dim=-1)
            expected_rows.append(
                torch.einsum("hl,lhd->hd", probabilities, values.float()).to(
                    torch.bfloat16
                )
            )
        query_start += chunk_len
    torch.testing.assert_close(
        second_output,
        torch.stack(expected_rows),
        rtol=3e-2,
        atol=3e-2,
    )
    torch.testing.assert_close(packed_output, output)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not sgl_fa3_support()[0],
    reason="CUDA and a validated sglang-kernel are required",
)
def test_sgl_fa3_minimax_ragged_split_boundary_matches_torch() -> None:
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    if properties.multi_processor_count != 132:
        pytest.skip("the regression requires a 132-SM Hopper GPU")

    torch.manual_seed(20260825)
    q_heads, kv_heads, head_dim = 12, 2, 128
    chunk_lens = (139, 16)
    context_values = (256, 2048)
    total_q = sum(chunk_lens)
    total_k = sum(context_values)
    page_table_width = 8192
    q = torch.randn(
        total_q, q_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    k_cache = torch.randn(
        total_k, kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    v_cache = torch.randn_like(k_cache)
    page_table = torch.zeros(
        2, page_table_width, device=device, dtype=torch.int32
    )
    page_table[0, : context_values[0]] = torch.arange(
        context_values[0], device=device, dtype=torch.int32
    )
    page_table[1, : context_values[1]] = torch.arange(
        context_values[0], total_k, device=device, dtype=torch.int32
    )
    request_indices = torch.tensor([0, 1], device=device, dtype=torch.int32)
    context_lens = torch.tensor(context_values, device=device, dtype=torch.int32)
    cu_seqlens_q = torch.tensor(
        [0, chunk_lens[0], total_q], device=device, dtype=torch.int32
    )
    assert total_q != len(chunk_lens) * max(chunk_lens)

    kernel = SglFa3DecodeKernel(
        device=device,
        max_batch_size=2,
        softmax_scale=head_dim**-0.5,
    )
    raw_op = kernel._op
    scheduler_op = kernel._scheduler_op
    assert scheduler_op is not None
    forward_splits = []
    metadata_splits = []

    def recorded_raw_op(*args):
        forward_splits.append(int(args[_FWD_ARGUMENTS.index("num_splits")]))
        return raw_op(*args)

    def recorded_scheduler_op(*args):
        metadata_splits.append(int(args[21]))
        return scheduler_op(*args)

    kernel._op = recorded_raw_op
    kernel._scheduler_op = recorded_scheduler_op
    validation_scope = object()
    outputs = []
    for _ in range(2):
        output = torch.empty_like(q)
        returned_output = kernel.run_explicit_varlen(
            q,
            k_cache,
            v_cache,
            page_table,
            request_indices,
            context_lens,
            output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max(chunk_lens),
            validation_scope=validation_scope,
        )
        outputs.append(returned_output)

    assert forward_splits[0] == 0
    assert len(metadata_splits) == 1
    assert metadata_splits[0] > 1
    assert forward_splits[1] == metadata_splits[0]

    expected_rows = []
    query_start = 0
    group_size = q_heads // kv_heads
    for batch_index, chunk_len in enumerate(chunk_lens):
        context_len = context_values[batch_index]
        for query_offset in range(chunk_len):
            visible_len = context_len - chunk_len + query_offset + 1
            active = page_table[batch_index, :visible_len].long()
            query_index = query_start + query_offset
            keys = k_cache[active].repeat_interleave(group_size, dim=1)
            values = v_cache[active].repeat_interleave(group_size, dim=1)
            logits = torch.einsum("hd,lhd->hl", q[query_index].float(), keys.float())
            probabilities = torch.softmax(logits * (head_dim**-0.5), dim=-1)
            expected_rows.append(
                torch.einsum("hl,lhd->hd", probabilities, values.float()).to(
                    torch.bfloat16
                )
            )
        query_start += chunk_len
    expected = torch.stack(expected_rows)

    for output in outputs:
        torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("q_heads,kv_heads", [(32, 4), (16, 2), (12, 2)])
@pytest.mark.skipif(
    not torch.cuda.is_available() or not sgl_fa3_support()[0],
    reason="CUDA and a validated sglang-kernel are required",
)
def test_sgl_fa3_qwen3_moe_gqa_prefill_matches_torch(
    q_heads: int,
    kv_heads: int,
) -> None:
    torch.manual_seed(20260817 + q_heads)
    device = torch.device("cuda")
    head_dim = 128
    chunk_lens = (3, 2)
    context_lens = torch.tensor([7, 5], device=device, dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
    q = torch.randn(5, q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(29, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    physical_slots = torch.randperm(29, device=device)[:12]
    page_table = torch.zeros(2, 7, device=device, dtype=torch.int32)
    page_table[0, :7] = physical_slots[:7].to(torch.int32)
    page_table[1, :5] = physical_slots[7:].to(torch.int32)
    request_indices = torch.tensor([0, 1], device=device, dtype=torch.int32)
    output = torch.empty_like(q)
    kernel = SglFa3DecodeKernel(
        device=device,
        max_batch_size=2,
        softmax_scale=head_dim**-0.5,
    )

    kernel.run_explicit_varlen(
        q,
        k_cache,
        v_cache,
        page_table,
        request_indices,
        context_lens,
        output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max(chunk_lens),
    )

    group_size = q_heads // kv_heads
    expected_rows = []
    query_start = 0
    for batch_index, chunk_len in enumerate(chunk_lens):
        context_len = int(context_lens[batch_index].item())
        row = int(request_indices[batch_index].item())
        for query_offset in range(chunk_len):
            visible_len = context_len - chunk_len + query_offset + 1
            active = page_table[row, :visible_len].long()
            query_index = query_start + query_offset
            keys = k_cache[active].repeat_interleave(group_size, dim=1)
            values = v_cache[active].repeat_interleave(group_size, dim=1)
            logits = torch.einsum("hd,lhd->hl", q[query_index].float(), keys.float())
            probabilities = torch.softmax(logits * (head_dim**-0.5), dim=-1)
            expected_rows.append(
                torch.einsum("hl,lhd->hd", probabilities, values.float()).to(
                    torch.bfloat16
                )
            )
        query_start += chunk_len

    torch.testing.assert_close(
        output,
        torch.stack(expected_rows),
        rtol=3e-2,
        atol=3e-2,
    )
