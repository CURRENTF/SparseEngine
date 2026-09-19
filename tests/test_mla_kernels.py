from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from sparseengine.kernels.triton.mla import (
    DEFAULT_GLM_MLA_DECODE_CONFIG,
    GLM_MLA_SOFTMAX_SCALE,
    MlaDecodeWorkspace,
    allocate_mla_decode_workspace,
    copy_latent_to_cache,
    decode_stage1,
    decode_stage2,
    gather_latent_history,
    prepare_mla_decode_schedule,
    run_mla_decode,
    validate_copy_slot_mappings,
    validate_mla_decode_metadata,
)


CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for MLA Triton tests",
)
DECODE_CONTEXTS = (1, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1024, 4096)


class _CaptureMetadataTensor:
    """CUDA-shaped metadata double that rejects graph-unsafe host reads."""

    device = torch.device("cuda")
    dtype = torch.int32

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.ndim = len(shape)

    def numel(self) -> int:
        result = 1
        for size in self.shape:
            result *= size
        return result

    def tolist(self):
        raise AssertionError("capture validation must not read CUDA values")


def test_mla_decode_metadata_capture_validation_is_host_read_free(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sparseengine.kernels.triton.mla.decode_schedule.device_runtime.is_stream_capturing",
        lambda: True,
    )
    validate_mla_decode_metadata(
        _CaptureMetadataTensor((2, 32)),
        _CaptureMetadataTensor((2,)),
        _CaptureMetadataTensor((2,)),
        cache_slot_count=64,
        max_context_len=32,
    )


def _torch_mla_decode(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    active_slots: torch.Tensor,
    request_indices: torch.Tensor,
    context_lens: torch.Tensor,
) -> torch.Tensor:
    rows = []
    for batch_index in range(q_latent.shape[0]):
        context_len = int(context_lens[batch_index].item())
        if context_len == 0:
            rows.append(torch.zeros_like(q_latent[batch_index]))
            continue
        request_row = int(request_indices[batch_index].item())
        slots = active_slots[request_row, :context_len].long()
        keys_latent = latent_cache[slots, 0].float()
        keys_rope = rope_cache[slots, 0].float()
        logits = torch.matmul(q_latent[batch_index].float(), keys_latent.T)
        logits += torch.matmul(q_rope[batch_index].float(), keys_rope.T)
        probabilities = torch.softmax(
            logits * GLM_MLA_SOFTMAX_SCALE,
            dim=-1,
        )
        rows.append(torch.matmul(probabilities, keys_latent).to(torch.bfloat16))
    return torch.stack(rows)


def _make_decode_case(
    batch_size: int,
    head_count: int,
    max_context_len: int,
) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    lengths = [
        max(1, max_context_len - ((batch_index * 17) % max(1, max_context_len // 3 + 1)))
        for batch_index in range(batch_size)
    ]
    request_rows = list(reversed(range(batch_size)))
    slot_count = sum(lengths) + 31
    physical_slots = torch.randperm(slot_count, dtype=torch.int32)
    active_slots_cpu = torch.full(
        (batch_size, max_context_len),
        -1,
        dtype=torch.int32,
    )
    cursor = 0
    for batch_index, (request_row, length) in enumerate(
        zip(request_rows, lengths)
    ):
        del batch_index
        active_slots_cpu[request_row, :length] = physical_slots[
            cursor : cursor + length
        ]
        cursor += length

    q_latent = torch.randn(
        (batch_size, head_count, 512),
        dtype=torch.bfloat16,
        device=device,
    )
    q_rope = torch.randn(
        (batch_size, head_count, 64),
        dtype=torch.bfloat16,
        device=device,
    )
    latent_cache = torch.randn(
        (slot_count, 1, 512),
        dtype=torch.bfloat16,
        device=device,
    )
    rope_cache = torch.randn(
        (slot_count, 1, 64),
        dtype=torch.bfloat16,
        device=device,
    )
    active_slots = active_slots_cpu.to(device)
    request_indices = torch.tensor(
        request_rows,
        dtype=torch.int32,
        device=device,
    )
    context_lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    return (
        q_latent,
        q_rope,
        latent_cache,
        rope_cache,
        active_slots,
        request_indices,
        context_lens,
    )


@CUDA_REQUIRED
def test_copy_latent_skips_padding_and_supports_strides() -> None:
    torch.manual_seed(3)
    latent_source = torch.randn(
        (5, 1, 1024),
        dtype=torch.bfloat16,
        device="cuda",
    )
    rope_source = torch.randn(
        (5, 1, 128),
        dtype=torch.bfloat16,
        device="cuda",
    )
    latent = latent_source[..., ::2]
    rope = rope_source[..., ::2]
    assert not latent.is_contiguous()
    assert not rope.is_contiguous()
    slots = torch.tensor([2, -1, 5, 0, -1], dtype=torch.int32, device="cuda")
    latent_cache = torch.full(
        (8, 1, 512),
        7.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    rope_cache = torch.full(
        (8, 1, 64),
        9.0,
        dtype=torch.bfloat16,
        device="cuda",
    )

    copy_latent_to_cache(latent, rope, slots, latent_cache, rope_cache)
    torch.cuda.synchronize()

    torch.testing.assert_close(latent_cache[2], latent[0])
    torch.testing.assert_close(latent_cache[5], latent[2])
    torch.testing.assert_close(latent_cache[0], latent[3])
    torch.testing.assert_close(
        latent_cache[1],
        torch.full_like(latent_cache[1], 7.0),
    )
    torch.testing.assert_close(rope_cache[2], rope[0])
    torch.testing.assert_close(rope_cache[5], rope[2])
    torch.testing.assert_close(rope_cache[0], rope[3])
    torch.testing.assert_close(
        rope_cache[1],
        torch.full_like(rope_cache[1], 9.0),
    )


@CUDA_REQUIRED
def test_copy_latent_rejects_duplicate_and_out_of_range_slots() -> None:
    latent = torch.zeros((2, 1, 512), dtype=torch.bfloat16, device="cuda")
    rope = torch.zeros((2, 1, 64), dtype=torch.bfloat16, device="cuda")
    latent_cache = torch.zeros((4, 1, 512), dtype=torch.bfloat16, device="cuda")
    rope_cache = torch.zeros((4, 1, 64), dtype=torch.bfloat16, device="cuda")

    with pytest.raises(ValueError, match="duplicate"):
        copy_latent_to_cache(
            latent,
            rope,
            torch.tensor([1, 1], dtype=torch.int32, device="cuda"),
            latent_cache,
            rope_cache,
        )
    with pytest.raises(ValueError, match="outside"):
        copy_latent_to_cache(
            latent,
            rope,
            torch.tensor([0, 4], dtype=torch.int32, device="cuda"),
            latent_cache,
            rope_cache,
        )


def test_batched_copy_slot_validation_is_row_local() -> None:
    validate_copy_slot_mappings(
        torch.tensor([[0, 1, -1], [0, 1, -1]], dtype=torch.int32),
        cache_slot_count=4,
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_copy_slot_mappings(
            torch.tensor([[0, 0], [1, 2]], dtype=torch.int32),
            cache_slot_count=4,
        )
    with pytest.raises(ValueError, match="outside"):
        validate_copy_slot_mappings(
            torch.tensor([[0, 4], [1, 2]], dtype=torch.int32),
            cache_slot_count=4,
        )


@CUDA_REQUIRED
def test_gather_latent_full_ragged_history_with_padded_row() -> None:
    torch.manual_seed(5)
    latent_cache = torch.randn(
        (24, 1, 512),
        dtype=torch.bfloat16,
        device="cuda",
    )
    rope_cache = torch.randn(
        (24, 1, 64),
        dtype=torch.bfloat16,
        device="cuda",
    )
    active_storage = torch.full((3, 14), -1, dtype=torch.int32, device="cuda")
    active_slots = active_storage[:, ::2]
    assert not active_slots.is_contiguous()
    row_two = torch.tensor([7, 1, 13, 4, 18], dtype=torch.int32, device="cuda")
    row_zero = torch.tensor([6, 21, 2], dtype=torch.int32, device="cuda")
    active_slots[2, :5] = row_two
    active_slots[0, :3] = row_zero
    request_indices = torch.tensor([2, 0, -1], dtype=torch.int32, device="cuda")
    context_lens = torch.tensor([5, 3, 0], dtype=torch.int32, device="cuda")
    packed_starts = torch.tensor([0, 5, 8], dtype=torch.int32, device="cuda")
    gathered_latent = torch.full(
        (8, 512),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    gathered_rope = torch.full(
        (8, 64),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )

    gather_latent_history(
        latent_cache,
        rope_cache,
        active_slots,
        request_indices,
        context_lens,
        packed_starts,
        gathered_latent,
        gathered_rope,
        max_context_len=7,
    )
    torch.cuda.synchronize()

    expected_slots = torch.cat((row_two, row_zero)).long()
    torch.testing.assert_close(gathered_latent, latent_cache[expected_slots, 0])
    torch.testing.assert_close(gathered_rope, rope_cache[expected_slots, 0])


@CUDA_REQUIRED
def test_gather_latent_rejects_duplicate_source_positions() -> None:
    latent_cache = torch.zeros((4, 1, 512), dtype=torch.bfloat16, device="cuda")
    rope_cache = torch.zeros((4, 1, 64), dtype=torch.bfloat16, device="cuda")
    active_slots = torch.tensor([[1, 1]], dtype=torch.int32, device="cuda")
    request_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    context_lens = torch.tensor([2], dtype=torch.int32, device="cuda")
    packed_starts = torch.tensor([0], dtype=torch.int32, device="cuda")
    gathered_latent = torch.empty((2, 512), dtype=torch.bfloat16, device="cuda")
    gathered_rope = torch.empty((2, 64), dtype=torch.bfloat16, device="cuda")

    with pytest.raises(ValueError, match="duplicate"):
        gather_latent_history(
            latent_cache,
            rope_cache,
            active_slots,
            request_indices,
            context_lens,
            packed_starts,
            gathered_latent,
            gathered_rope,
            max_context_len=2,
        )


@pytest.mark.parametrize("batch_size", (1, 2, 8))
@pytest.mark.parametrize("head_count", (5, 10, 20))
@CUDA_REQUIRED
def test_mla_decode_matches_torch_matrix(
    batch_size: int,
    head_count: int,
) -> None:
    torch.manual_seed(17 + batch_size * 10 + head_count)
    for max_context_len in DECODE_CONTEXTS:
        case = _make_decode_case(batch_size, head_count, max_context_len)
        q_latent, q_rope, latent_cache, rope_cache = case[:4]
        active_slots, request_indices, context_lens = case[4:]
        output = torch.empty_like(q_latent)
        workspace = allocate_mla_decode_workspace(
            batch_size=batch_size,
            head_count=head_count,
            device=q_latent.device,
        )

        run_mla_decode(
            q_latent,
            q_rope,
            latent_cache,
            rope_cache,
            active_slots,
            request_indices,
            context_lens,
            output,
            workspace,
            softmax_scale=GLM_MLA_SOFTMAX_SCALE,
        )
        torch.cuda.synchronize()
        expected = _torch_mla_decode(*case)
        torch.testing.assert_close(
            output.float(),
            expected.float(),
            rtol=3e-2,
            atol=3e-2,
            msg=lambda message: (
                f"batch={batch_size}, heads={head_count}, "
                f"context={max_context_len}: {message}"
            ),
        )


@pytest.mark.parametrize("reduce_heads", [False, True])
@CUDA_REQUIRED
def test_mla_decode_writes_raw_attention_scores(reduce_heads: bool) -> None:
    torch.manual_seed(211)
    case = _make_decode_case(batch_size=2, head_count=20, max_context_len=33)
    q_latent, q_rope, latent_cache, rope_cache = case[:4]
    active_slots, request_indices, context_lens = case[4:]
    output = torch.empty_like(q_latent)
    score_shape = (
        (2, active_slots.shape[1])
        if reduce_heads
        else (2, 20, active_slots.shape[1])
    )
    scores = torch.full(
        score_shape,
        -1.0e20,
        dtype=torch.float32,
        device="cuda",
    )
    workspace = allocate_mla_decode_workspace(
        batch_size=2,
        head_count=20,
        device="cuda",
    )

    run_mla_decode(
        q_latent,
        q_rope,
        latent_cache,
        rope_cache,
        active_slots,
        request_indices,
        context_lens,
        output,
        workspace,
        softmax_scale=GLM_MLA_SOFTMAX_SCALE,
        attn_score=scores,
    )
    torch.cuda.synchronize()

    for batch_idx in range(2):
        length = int(context_lens[batch_idx].item())
        request_row = int(request_indices[batch_idx].item())
        slots = active_slots[request_row, :length].long()
        expected = torch.matmul(
            q_latent[batch_idx].float(), latent_cache[slots, 0].float().T
        ) + torch.matmul(
            q_rope[batch_idx].float(), rope_cache[slots, 0].float().T
        )
        actual = scores[batch_idx, :length] if reduce_heads else scores[batch_idx, :, :length]
        if reduce_heads:
            expected = expected.max(dim=0).values
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
        assert torch.all(scores[batch_idx, ..., length:] == -1.0e20)


@CUDA_REQUIRED
@pytest.mark.parametrize("graph_mode", [False, True])
def test_mla_h2o_reduced_score_softmax_matches_approximation(graph_mode):
    from sparseengine.kernels.triton.h2o_score import h2o_softmax_accumulate

    torch.manual_seed(218)
    case = _make_decode_case(batch_size=1, head_count=20, max_context_len=33)
    q, qr, latent, rope, slots, rows, lengths = case
    length = int(lengths[0].item())
    output = torch.empty_like(q)
    scores = torch.empty((1, slots.shape[1]), device="cuda", dtype=torch.float32)
    history = torch.zeros((1, 1, slots.shape[1]), device="cuda", dtype=torch.float32)
    workspace = allocate_mla_decode_workspace(batch_size=1, head_count=20, device="cuda")

    def attention():
        run_mla_decode(
            q, qr, latent, rope, slots, rows, lengths, output, workspace,
            softmax_scale=GLM_MLA_SOFTMAX_SCALE, attn_score=scores,
        )

    attention()
    graph = None
    if graph_mode:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            attention()
    expected = torch.zeros(length, device="cuda")
    active = slots[int(rows[0].item()), :length].long()
    for _ in range(3):
        q.mul_(0.9)
        graph.replay() if graph is not None else attention()
        # H2O accumulation is intentionally graph-out, following model replay.
        h2o_softmax_accumulate(
            scores.unsqueeze(0), history, width=length, previous_width=length,
            softmax_scale=GLM_MLA_SOFTMAX_SCALE,
        )
        raw = q[0].float() @ latent[active, 0].float().T
        raw += qr[0].float() @ rope[active, 0].float().T
        expected += (raw.amax(0) * GLM_MLA_SOFTMAX_SCALE).softmax(-1)
        torch.testing.assert_close(history[0, 0, :length], expected, rtol=3e-2, atol=2e-3)
    assert torch.all(history[..., length:] == 0)


@CUDA_REQUIRED
def test_mla_decode_score_capacity_can_be_smaller_than_slot_table() -> None:
    torch.manual_seed(219)
    case = _make_decode_case(batch_size=1, head_count=20, max_context_len=33)
    q_latent, q_rope, latent_cache, rope_cache = case[:4]
    active_slots, request_indices, _context_lens = case[4:]
    context_lens = torch.tensor([17], dtype=torch.int32, device="cuda")
    output = torch.empty_like(q_latent)
    scores = torch.empty((1, 17), dtype=torch.float32, device="cuda")
    workspace = allocate_mla_decode_workspace(
        batch_size=1,
        head_count=20,
        device="cuda",
    )

    run_mla_decode(
        q_latent,
        q_rope,
        latent_cache,
        rope_cache,
        active_slots,
        request_indices,
        context_lens,
        output,
        workspace,
        softmax_scale=GLM_MLA_SOFTMAX_SCALE,
        attn_score=scores,
        max_context_len=17,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(output).all()
    assert torch.isfinite(scores).all()


@CUDA_REQUIRED
def test_mla_reduced_scores_reset_before_each_decode_step() -> None:
    torch.manual_seed(223)
    case = _make_decode_case(batch_size=1, head_count=20, max_context_len=33)
    q_latent, q_rope, latent_cache, rope_cache = case[:4]
    active_slots, request_indices, context_lens = case[4:]
    output = torch.empty_like(q_latent)
    scores = torch.empty(
        (1, active_slots.shape[1]),
        dtype=torch.float32,
        device="cuda",
    )
    workspace = allocate_mla_decode_workspace(
        batch_size=1,
        head_count=20,
        device="cuda",
    )

    run_mla_decode(
        q_latent,
        q_rope,
        latent_cache,
        rope_cache,
        active_slots,
        request_indices,
        context_lens,
        output,
        workspace,
        softmax_scale=GLM_MLA_SOFTMAX_SCALE,
        attn_score=scores,
    )
    first_scores = scores.clone()
    run_mla_decode(
        torch.zeros_like(q_latent),
        torch.zeros_like(q_rope),
        latent_cache,
        rope_cache,
        active_slots,
        request_indices,
        context_lens,
        output,
        workspace,
        softmax_scale=GLM_MLA_SOFTMAX_SCALE,
        attn_score=scores,
    )
    torch.cuda.synchronize()

    length = int(context_lens[0].item())
    assert torch.any(first_scores[0, :length] != 0)
    torch.testing.assert_close(
        scores[0, :length],
        torch.zeros_like(scores[0, :length]),
    )
    assert torch.all(scores[0, length:] == -1.0e20)


@CUDA_REQUIRED
def test_mla_decode_cuda_graph_replay_resets_reduced_scores() -> None:
    torch.manual_seed(227)
    case = _make_decode_case(batch_size=2, head_count=20, max_context_len=33)
    q_latent, q_rope, latent_cache, rope_cache = case[:4]
    active_slots, request_indices, context_lens = case[4:]
    output = torch.empty_like(q_latent)
    scores = torch.empty(
        (2, active_slots.shape[1]),
        dtype=torch.float32,
        device="cuda",
    )
    workspace = allocate_mla_decode_workspace(
        batch_size=2,
        head_count=20,
        device="cuda",
    )

    def run_decode() -> None:
        run_mla_decode(
            q_latent,
            q_rope,
            latent_cache,
            rope_cache,
            active_slots,
            request_indices,
            context_lens,
            output,
            workspace,
            softmax_scale=GLM_MLA_SOFTMAX_SCALE,
            attn_score=scores,
            validate_metadata=False,
        )

    run_decode()
    run_decode()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_decode()

    q_latent.zero_()
    q_rope.zero_()
    scores.fill_(12345.0)
    graph.replay()
    graph_output = output.clone()
    graph_scores = scores.clone()
    run_decode()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_output, output, rtol=0, atol=0)
    torch.testing.assert_close(graph_scores, scores, rtol=0, atol=0)
    for batch_idx, length in enumerate(context_lens.tolist()):
        torch.testing.assert_close(
            graph_scores[batch_idx, :length],
            torch.zeros_like(graph_scores[batch_idx, :length]),
        )
        assert torch.all(graph_scores[batch_idx, length:] == -1.0e20)

@CUDA_REQUIRED
def test_mla_decode_zeroes_padded_rows() -> None:
    torch.manual_seed(19)
    batch_size = 3
    head_count = 5
    q_latent = torch.randn(
        (batch_size, head_count, 512),
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_rope = torch.randn(
        (batch_size, head_count, 64),
        dtype=torch.bfloat16,
        device="cuda",
    )
    latent_cache = torch.randn((12, 1, 512), dtype=torch.bfloat16, device="cuda")
    rope_cache = torch.randn((12, 1, 64), dtype=torch.bfloat16, device="cuda")
    active_slots = torch.full((2, 5), -1, dtype=torch.int32, device="cuda")
    active_slots[0, :5] = torch.tensor([1, 3, 5, 7, 9], device="cuda")
    active_slots[1, :3] = torch.tensor([2, 4, 6], device="cuda")
    request_indices = torch.tensor([0, -1, 1], dtype=torch.int32, device="cuda")
    context_lens = torch.tensor([5, 0, 3], dtype=torch.int32, device="cuda")
    output = torch.empty_like(q_latent)
    workspace = allocate_mla_decode_workspace(
        batch_size=batch_size,
        head_count=head_count,
        device="cuda",
    )

    run_mla_decode(
        q_latent,
        q_rope,
        latent_cache,
        rope_cache,
        active_slots,
        request_indices,
        context_lens,
        output,
        workspace,
        softmax_scale=GLM_MLA_SOFTMAX_SCALE,
    )
    torch.cuda.synchronize()

    expected = _torch_mla_decode(
        q_latent,
        q_rope,
        latent_cache,
        rope_cache,
        active_slots,
        request_indices,
        context_lens,
    )
    torch.testing.assert_close(output.float(), expected.float(), rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(output[1], torch.zeros_like(output[1]))


@CUDA_REQUIRED
def test_mla_decode_rejects_duplicate_active_slots() -> None:
    q_latent = torch.zeros((1, 5, 512), dtype=torch.bfloat16, device="cuda")
    q_rope = torch.zeros((1, 5, 64), dtype=torch.bfloat16, device="cuda")
    latent_cache = torch.zeros((4, 1, 512), dtype=torch.bfloat16, device="cuda")
    rope_cache = torch.zeros((4, 1, 64), dtype=torch.bfloat16, device="cuda")
    active_slots = torch.tensor([[1, 1]], dtype=torch.int32, device="cuda")
    request_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    context_lens = torch.tensor([2], dtype=torch.int32, device="cuda")
    output = torch.empty_like(q_latent)
    workspace = allocate_mla_decode_workspace(
        batch_size=1,
        head_count=5,
        device="cuda",
    )

    with pytest.raises(ValueError, match="duplicate"):
        run_mla_decode(
            q_latent,
            q_rope,
            latent_cache,
            rope_cache,
            active_slots,
            request_indices,
            context_lens,
            output,
            workspace,
            softmax_scale=GLM_MLA_SOFTMAX_SCALE,
        )


@CUDA_REQUIRED
def test_mla_decode_metadata_ignores_duplicate_graph_padding_rows() -> None:
    active_slots = torch.tensor(
        [[0, 1], [2, 3]], dtype=torch.int32, device="cuda"
    )
    context_lens = torch.tensor([2, 2, 2, 2], dtype=torch.int32, device="cuda")
    padded_request_indices = torch.tensor(
        [0, 1, 0, 0], dtype=torch.int32, device="cuda"
    )

    validate_mla_decode_metadata(
        active_slots,
        padded_request_indices,
        context_lens,
        cache_slot_count=4,
        valid_batch_size=2,
    )

    duplicate_real_request_indices = torch.tensor(
        [0, 0, 0, 0], dtype=torch.int32, device="cuda"
    )
    with pytest.raises(ValueError, match="duplicate non-padding rows"):
        validate_mla_decode_metadata(
            active_slots,
            duplicate_real_request_indices,
            context_lens,
            cache_slot_count=4,
            valid_batch_size=2,
        )


@CUDA_REQUIRED
def test_decode_stage1_matches_per_block_oracle() -> None:
    torch.manual_seed(23)
    case = _make_decode_case(batch_size=2, head_count=5, max_context_len=129)
    q_latent, q_rope, latent_cache, rope_cache = case[:4]
    active_slots, request_indices, context_lens = case[4:]
    config = replace(
        DEFAULT_GLM_MLA_DECODE_CONFIG,
        program_count=8,
        blocks_per_program=2,
    )
    workspace = allocate_mla_decode_workspace(
        batch_size=2,
        head_count=5,
        device="cuda",
        config=config,
    )
    prepare_mla_decode_schedule(context_lens, workspace, config=config)
    decode_stage1(
        q_latent,
        q_rope,
        latent_cache,
        rope_cache,
        active_slots,
        request_indices,
        context_lens,
        workspace.block_size,
        workspace.mid_output,
        workspace.mid_logsumexp,
        softmax_scale=GLM_MLA_SOFTMAX_SCALE,
        program_count=config.program_count,
        block_q_heads=config.block_q_heads,
        block_n=config.block_n,
        pipeline_stages=config.stage1_pipeline_stages,
        num_warps=config.stage1_num_warps,
    )
    torch.cuda.synchronize()

    block_size = int(workspace.block_size.item())
    starts = workspace.batch_start_indices.cpu().tolist()
    lengths = context_lens.cpu().tolist()
    for batch_index, (start, length) in enumerate(zip(starts, lengths)):
        request_row = int(request_indices[batch_index].item())
        all_slots = active_slots[request_row, :length].long()
        for block_index, token_start in enumerate(range(0, length, block_size)):
            slots = all_slots[token_start : token_start + block_size]
            keys_latent = latent_cache[slots, 0].float()
            keys_rope = rope_cache[slots, 0].float()
            logits = torch.matmul(q_latent[batch_index].float(), keys_latent.T)
            logits += torch.matmul(q_rope[batch_index].float(), keys_rope.T)
            logits *= GLM_MLA_SOFTMAX_SCALE
            probabilities = torch.softmax(logits, dim=-1)
            expected_output = torch.matmul(
                probabilities.to(torch.bfloat16).float(),
                keys_latent,
            )
            expected_lse = torch.logsumexp(logits, dim=-1)
            output_index = start + block_index
            torch.testing.assert_close(
                workspace.mid_output[:5, output_index],
                expected_output,
                rtol=3e-2,
                atol=3e-2,
            )
            torch.testing.assert_close(
                workspace.mid_logsumexp[:5, output_index],
                expected_lse,
                rtol=2e-2,
                atol=2e-2,
            )


@CUDA_REQUIRED
def test_decode_stage2_matches_weighted_block_oracle() -> None:
    torch.manual_seed(29)
    head_count = 5
    context_lens = torch.tensor([33, 65], dtype=torch.int32, device="cuda")
    block_size = torch.tensor([32], dtype=torch.int32, device="cuda")
    batch_starts = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    mid_output = torch.randn((head_count, 5, 512), dtype=torch.float32, device="cuda")
    mid_lse = torch.randn((head_count, 5), dtype=torch.float32, device="cuda")
    output = torch.empty((2, head_count, 512), dtype=torch.bfloat16, device="cuda")

    decode_stage2(
        block_size,
        batch_starts,
        context_lens,
        mid_output,
        mid_lse,
        output,
        pipeline_stages=2,
        num_warps=4,
    )
    torch.cuda.synchronize()

    expected_rows = []
    for batch_index, (start, block_count) in enumerate(((0, 2), (2, 3))):
        del batch_index
        weights = torch.softmax(mid_lse[:, start : start + block_count], dim=-1)
        expected_rows.append(
            torch.einsum(
                "hb,hbd->hd",
                weights,
                mid_output[:, start : start + block_count],
            ).to(torch.bfloat16)
        )
    expected = torch.stack(expected_rows)
    torch.testing.assert_close(output.float(), expected.float(), rtol=1e-2, atol=1e-2)


@CUDA_REQUIRED
def test_decode_rejects_small_workspace_and_non_bf16_input() -> None:
    case = _make_decode_case(batch_size=1, head_count=5, max_context_len=33)
    q_latent, q_rope, latent_cache, rope_cache = case[:4]
    active_slots, request_indices, context_lens = case[4:]
    output = torch.empty_like(q_latent)
    workspace = allocate_mla_decode_workspace(
        batch_size=1,
        head_count=5,
        device="cuda",
    )
    small_workspace = MlaDecodeWorkspace(
        block_size=workspace.block_size,
        batch_start_indices=workspace.batch_start_indices,
        mid_output=workspace.mid_output[:, :-1],
        mid_logsumexp=workspace.mid_logsumexp[:, :-1],
    )

    with pytest.raises(ValueError, match="workspace is too small"):
        run_mla_decode(
            q_latent,
            q_rope,
            latent_cache,
            rope_cache,
            active_slots,
            request_indices,
            context_lens,
            output,
            small_workspace,
            softmax_scale=GLM_MLA_SOFTMAX_SCALE,
        )
    with pytest.raises(TypeError, match="q_latent"):
        run_mla_decode(
            q_latent.float(),
            q_rope,
            latent_cache,
            rope_cache,
            active_slots,
            request_indices,
            context_lens,
            output,
            workspace,
            softmax_scale=GLM_MLA_SOFTMAX_SCALE,
        )
