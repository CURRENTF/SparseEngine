"""Contracts that scalar-to-batched metadata rewrites can accidentally break."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sparseengine.engine.cache_manager.methods.deltakv_base import DeltaKVCacheManager
from sparseengine.engine.prefix_prune import select_global_keep_indices
from sparseengine.multimodal.runtime import MultiModalRuntime, MultiModalState
from sparseengine.utils.context import get_context, set_context

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
@pytest.mark.parametrize("keep", [0, 3, 7])
def test_prefix_prune_cuda_ties_and_noncontiguous_scores(keep):
    # The existing selector test only covers contiguous CPU inputs.
    values = [float("inf"), 1.0, 1.0, -float("inf"), 0.0, 1.0, -0.0]
    scores = torch.tensor([[value, value] for value in values], device="cuda")[:, 0]
    protected = [] if keep == 0 else [3]
    result = select_global_keep_indices(
        scores,
        keep_tokens=keep,
        protected_indices=torch.tensor(protected, device="cuda", dtype=torch.long),
    )
    candidates = [i for i in range(len(values)) if i not in protected]
    expected = sorted(
        protected
        + sorted(candidates, key=lambda i: (-values[i], i))[: keep - len(protected)]
    )
    assert result.tolist() == expected


def _delta_manager(total_lengths, compressed_lengths, sink):
    manager = object.__new__(DeltaKVCacheManager)
    manager.device = torch.device("cuda")
    manager.config = SimpleNamespace(sink_keep_tokens=sink)
    manager.full_layer_to_idx = {}
    manager.row_seq_lens = np.asarray(total_lengths, dtype=np.int32)
    manager.row_deltakv_compressed_lens = np.asarray(compressed_lengths, dtype=np.int32)
    batch, width = len(total_lengths), max(max(total_lengths), 1)
    raw = torch.arange(batch * width, dtype=torch.int32).reshape(batch, width)
    latent = torch.full_like(raw, -1)
    for row, (length, comp) in enumerate(zip(total_lengths, compressed_lengths)):
        for position in range(sink, min(length, sink + comp)):
            if position % 2:
                raw[row, position] = -1
                latent[row, position] = row * width + position
    # Interleaved storage catches forgotten map strides in the fused kernels.
    manager.sparse_layer_raw_slots_map = torch.stack((raw, raw), -1).cuda()[..., 0]
    manager.sparse_layer_latent_slots_map = torch.stack((latent, latent), -1).cuda()[
        ..., 0
    ]
    capacity = max(256, batch * width)
    manager.free_slots_stack_deltakv_full = torch.arange(
        batch * width, batch * width + capacity, device="cuda", dtype=torch.int32
    )
    manager._num_free_slots_deltakv_full = capacity
    return manager, raw, latent


@cuda
@pytest.mark.parametrize(
    "total,compressed,sink,candidates",
    [
        ([0, 3, 13], [0, 0, 7], 4, [[-1, 0, 0, 1, 6, 7]] * 3),
        ([0, 1, 1031], [0, 0, 0], 0, [[], [], []]),
        ([15, 23], [10, 16], 2, [[3, 3, -1, 8, 100], [14, 1, 0, 5, -2]]),
        ([7, 2053, 4100], [0, 1536, 3072], 8, [list(range(1471, -1, -1))] * 3),
    ],
)
def test_delta_eager_plan_preserves_order_counts_and_exact_allocation(
    total, compressed, sink, candidates
):
    # Existing static-plan tests do not cover exact eager allocation, duplicate
    # selections, or zero selections with a long raw tail.
    m, raw, latent = _delta_manager(total, compressed, sink)
    batch = len(total)
    rows_cpu = list(reversed(range(batch)))
    rows = torch.tensor(
        [[row, row] for row in rows_cpu], device="cuda", dtype=torch.int32
    )[:, 0]
    source = torch.tensor(candidates, dtype=torch.int32, device="cuda")
    selected = torch.stack((source, source), -1)[..., 0]
    expected_rows, positions, latents, allocated = [], [], [], []
    initial_free = free = m._num_free_slots_deltakv_full
    base = raw.numel()
    for row, choice in zip(rows_cpu, candidates):
        length = total[row]
        sink_len = min(sink, length)
        comp = min(compressed[row], max(0, length - sink))
        picked = [p + sink for p in choice if 0 <= p < comp]
        need = [p for p in picked if latent[row, p] >= 0]
        slots = list(range(base + free - len(need), base + free))
        free -= len(need)
        temp = iter(slots)
        expected_rows.append(
            raw[row, :sink_len].tolist()
            + [next(temp) if latent[row, p] >= 0 else int(raw[row, p]) for p in picked]
            + raw[row, sink_len + comp : length].tolist()
        )
        positions.extend(need)
        latents.extend(int(latent[row, p]) for p in need)
        allocated.extend(slots)
    for _ in range(2):
        m._num_free_slots_deltakv_full = initial_free
        output, local, lengths, temp, pos, lat, dst = (
            m._deltakv_build_view_and_plan_reconstruct_impl(1, selected, rows)
        )
        assert lengths.tolist() == [len(row) for row in expected_rows]
        for got, expected in zip(output.cpu(), expected_rows):
            assert got.tolist() == expected + [0] * (output.shape[1] - len(expected))
        assert local.tolist() == list(range(batch))
        assert pos.tolist() == positions and lat.tolist() == latents
        assert temp.tolist() == dst.tolist() == allocated
        assert m._num_free_slots_deltakv_full == free


@cuda
@pytest.mark.parametrize("failure", ["sink", "tail", "capacity"])
def test_delta_eager_validation_does_not_allocate_on_failure(failure):
    m, _, _ = _delta_manager([16], [8], 2)
    if failure == "sink":
        m.sparse_layer_raw_slots_map[0, 0] = -1
    elif failure == "tail":
        m.sparse_layer_raw_slots_map[0, 15] = -1
    else:
        m._num_free_slots_deltakv_full = 0
    free = m._num_free_slots_deltakv_full
    with pytest.raises(
        RuntimeError,
        match={
            "sink": "sink window",
            "tail": "buffer contains",
            "capacity": "Out of DeltaKV",
        }[failure],
    ):
        m._deltakv_build_view_and_plan_reconstruct_impl(
            1,
            torch.tensor([[1, 3]], dtype=torch.int32, device="cuda"),
            torch.tensor([0], dtype=torch.int32, device="cuda"),
        )
    assert m._num_free_slots_deltakv_full == free


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda)])
def test_multimodal_ranges_cross_chunks_and_mixed_requests(device):
    # Adjacent image/video types must remain one group, but groups from distinct
    # requests must not merge. The existing test covers one whole prompt only.
    types = [0, 1, 1, 2, 2, 0, 3, 1, 0]
    features = {
        i: torch.arange(types.count(i) * 3, device=device).reshape(-1, 3).float()
        + i * 100
        for i in (1, 2, 3)
    }

    class Model:
        multimodal_bidirectional = True

        def encode_multimodal(self, ids, tensors):
            return MultiModalState(
                torch.tensor(types, device=device),
                features,
                torch.arange(27, device=device).reshape(3, 9),
                2,
            )

        def embed_input_ids(self, ids):
            return ids[:, None].expand(-1, 3).float().clone()

    runtime = MultiModalRuntime(Model(), torch.device(device))
    runtime.register(1, list(range(9)), {})
    runtime.register(2, list(range(9)), {})
    seqs = [
        SimpleNamespace(seq_id=1, num_prefilled_tokens=2, current_chunk_size=3),
        SimpleNamespace(seq_id=9, num_prefilled_tokens=0, current_chunk_size=2),
        SimpleNamespace(seq_id=2, num_prefilled_tokens=3, current_chunk_size=5),
    ]
    ids = torch.arange(10, device=device)
    set_context(True)
    for _ in range(2):
        embeds, positions, mask = runtime.prepare(seqs, ids, ids, True)
        expected = torch.stack(
            (
                features[1][1],
                features[2][0],
                features[2][1],
                torch.full((3,), 3.0, device=device),
                torch.full((3,), 4.0, device=device),
                features[2][0],
                features[2][1],
                torch.full((3,), 7.0, device=device),
                features[3][0],
                features[1][2],
            )
        )
        torch.testing.assert_close(embeds, expected, atol=0, rtol=0)
        assert mask.tolist() == [
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            False,
            True,
            True,
        ]
        assert get_context().multimodal_image_groups.tolist() == [
            1,
            1,
            1,
            0,
            0,
            2,
            2,
            0,
            0,
            3,
        ]
        expected_positions = torch.cat(
            (
                runtime.states[1].position_ids[:, 2:5],
                ids[3:5].expand(3, -1),
                runtime.states[2].position_ids[:, 3:8],
            ),
            dim=1,
        )
        assert torch.equal(positions, expected_positions)
    runtime.free_batch([1, 2])
    assert not runtime.states


@cuda
def test_flashinfer_eager_ragged_plan_reuses_storage_without_stale_indices():
    from sparseengine.operators.flashinfer_decode_state import FlashInferPagedDecodeState

    # Real public wrapper execution catches stale packed indices after shortening
    # a request; mocks of plan() cannot establish this numerical contract.
    torch.manual_seed(81)
    state = FlashInferPagedDecodeState(torch.device("cuda"))
    spec = SimpleNamespace(
        page_size=16,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=64,
        softmax_scale=64**-0.5,
        activation_dtype=torch.bfloat16,
    )
    kv = torch.randn(16, 2, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(2, 4, 64, device="cuda", dtype=torch.bfloat16)
    table = torch.arange(16, dtype=torch.int32, device="cuda").reshape(2, 8)
    for rows_host, lengths_host in [
        ([1, 0], [117, 35]),
        ([0, 1], [1, 65]),
        ([1, 0], [80, 128]),
    ]:
        rows = torch.tensor(
            [[n, n] for n in rows_host], device="cuda", dtype=torch.int32
        )[:, 0]
        lengths = torch.tensor(
            [[n, n] for n in lengths_host], device="cuda", dtype=torch.int32
        )[:, 0]
        state.plan(
            spec,
            active_slots=table,
            req_indices=rows,
            context_lens=lengths,
            max_context_len=max(lengths_host),
        )
        out = state.wrapper.run(q, kv)
        expected = []
        for b, (row, length) in enumerate(zip(rows_host, lengths_host)):
            selected = kv[row * 8 : (row + 1) * 8]
            k = (
                selected[:, 0]
                .reshape(-1, 2, 64)[:length]
                .repeat_interleave(2, 1)
                .float()
            )
            v = (
                selected[:, 1]
                .reshape(-1, 2, 64)[:length]
                .repeat_interleave(2, 1)
                .float()
            )
            prob = (
                torch.einsum("hd,thd->ht", q[b].float(), k) * spec.softmax_scale
            ).softmax(-1)
            expected.append(torch.einsum("ht,thd->hd", prob, v).bfloat16())
        torch.testing.assert_close(out, torch.stack(expected), atol=0.016, rtol=0.016)
