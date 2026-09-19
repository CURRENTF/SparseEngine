"""Short rows, changing lengths, and padding must retain only real KV entries."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sparseengine.engine.sparse_methods.base import LayerEndEvent, SparseStepContext
from sparseengine.engine.sparse_methods.dynamic import OmniKVRuntime, DeltaKVRuntime
from sparseengine.kernels.triton.deltakv_kernels import deltakv_static_decode_plan
from sparseengine.kernels.triton.quest_decode_view import finalize_quest_paged_decode_view
from sparseengine.kernels.triton.quest_fused_selection import fused_exact_select_quest_paged_view
from sparseengine.operators.flashinfer_decode_state import FlashInferPagedDecodeGraphState

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _dynamic_runtime(method, lengths, *, device="cpu", capacity=None, graph=False):
    lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
    batch = len(lengths)
    width = capacity or max(1, int(lengths.max()))
    rows = torch.arange(batch, dtype=torch.int32, device=device).flip(0)
    table = torch.arange(batch * width, dtype=torch.int32, device=device).reshape(batch, width)
    state = SimpleNamespace(context_lens=lengths, max_context_len=width, req_indices=rows)
    manager = SimpleNamespace(
        device=torch.device(device), get_layer_batch_states=lambda _: state,
        get_layer_buffer_req_to_token_slots=lambda _: table,
        get_compressed_lens=lambda _: torch.zeros_like(lengths),
    )
    config = SimpleNamespace(
        sparse_method=method, obs_layer_ids=[0], full_attention_layers=[0],
        runtime_layout=None, tensor_parallel_size=1,
        hf_config=SimpleNamespace(num_hidden_layers=2, num_attention_heads=2,
                                 head_dim=4, dtype=torch.bfloat16),
        sink_keep_tokens=4, recent_keep_tokens=3, decode_keep_tokens=4,
        sparse_attn_score_dtype="float32", decode_graph=graph,
    )
    runtime = (OmniKVRuntime if method == "omnikv" else DeltaKVRuntime)(config, manager)
    context = SimpleNamespace(is_prefill=False)
    runtime.prepare_step(SparseStepContext([None] * batch, False, context))
    return runtime, context, lengths, rows, table


@pytest.mark.parametrize("method", ["omnikv", "deltakv"])
def test_dynamic_all_short_score_workspace_has_empty_candidate_domain(method):
    # An eager batch shorter than sink used to fail score preparation/normalization.
    runtime, _, lengths, _, _ = _dynamic_runtime(method, [1, 2])
    state = runtime.layer_batch_sparse_states[0]
    assert state.attn_score.shape == (2, 2, 4)
    scores = runtime._normalize_decode_scores(state)
    assert scores.shape == (2, 4)
    assert torch.isfinite(scores).all()
    assert torch.all(scores == torch.finfo(scores.dtype).min)
    assert lengths.tolist() == [1, 2]


@CUDA
def test_omnikv_mixed_selection_replays_with_new_lengths_and_rows():
    # Covers fewer history candidates than common k, sink overlap, an empty
    # padded row, and reuse of the same graph across the former long boundary.
    torch.manual_seed(19)
    runtime, context, lengths, rows, table = _dynamic_runtime(
        "omnikv", [1] * 8, device="cuda", capacity=32, graph=True,
    )
    state = runtime.layer_batch_sparse_states[0]
    raw = state.attn_score
    logits = torch.randn_like(raw)
    raw.copy_(logits)

    def run():
        state.attn_score = raw
        runtime.on_layer_end(LayerEndEvent(0, context, context))

    run()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    target = runtime.layer_batch_sparse_states[1]
    ptrs = (target.active_indices.data_ptr(), target.active_slots.data_ptr())
    for values in ([0, 1, 3, 4, 5, 8, 11, 29], [32, 12, 9, 7, 4, 2, 1, 0]):
        lengths.copy_(torch.tensor(values, dtype=torch.int32, device="cuda"))
        rows.copy_(rows.flip(0))
        raw.copy_(logits)
        graph.replay()
        assert ptrs == (target.active_indices.data_ptr(), target.active_slots.data_ptr())
        for b, n in enumerate(values):
            prefix = list(range(min(n, 4)))
            tail_start = max(4, n - 3)
            history = list(range(4, tail_start))
            if history:
                score = (logits[b, :, history].float() * 0.5).softmax(-1).amax(0).bfloat16()
                chosen = [history[j] for j in score.topk(min(4, len(history))).indices.tolist()]
            else:
                chosen = []
            expected = prefix + chosen + list(range(tail_start, n))
            count = int(target.context_lens[b])
            actual = target.active_indices[b, :count].tolist()
            assert count == len(expected) and set(actual) == set(expected)
            assert len(set(actual)) == len(actual)
            # Providers may emit the same selected set in score or logical order.
            # Physical slots must follow the emitted logical indices exactly.
            torch.testing.assert_close(
                target.active_slots[b, :count], table[rows[b].long(), actual],
            )


@CUDA
def test_deltakv_mixed_static_plan_replays_sink_and_reconstruction_boundaries():
    batch, capacity, sink, k, tail = 8, 32, 4, 4, 6
    shape = (batch, sink + k + tail)
    raw = torch.arange(batch * capacity, dtype=torch.int32, device="cuda").reshape(batch, capacity)
    latent = raw + 1000
    rows = torch.arange(batch, dtype=torch.int32, device="cuda").flip(0)
    lengths = torch.ones(batch, dtype=torch.int32, device="cuda")
    compressed = torch.zeros_like(lengths)
    selected = torch.arange(k, dtype=torch.int32, device="cuda").expand(batch, k).contiguous()
    temp = (2000 + torch.arange(batch * k, dtype=torch.int32, device="cuda")).reshape(batch, k)
    slots, positions = (torch.empty(shape, dtype=torch.int32, device="cuda") for _ in range(2))
    outlens = torch.empty_like(lengths)
    recon = [torch.empty(batch * k, dtype=torch.int32, device="cuda") for _ in range(3)]

    def run():
        deltakv_static_decode_plan(
            raw_slots_map=raw, latent_slots_map=latent,
            active_compressed_indices=selected, req_indices=rows,
            context_lens=lengths, compressed_lens=compressed, temp_slots=temp,
            active_slots_out=slots, active_pos_out=positions, new_context_lens_out=outlens,
            recon_pos_out=recon[0], recon_latent_out=recon[1], recon_out_slot_out=recon[2],
            sink=sink, max_buffer=tail,
        )

    run()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    for values in ([0, 1, 3, 4, 5, 8, 11, 29], [29, 12, 9, 7, 4, 2, 1, 0]):
        lengths.copy_(torch.tensor(values, dtype=torch.int32, device="cuda"))
        compressed.copy_((lengths - sink - tail).clamp_min(0))
        graph.replay()
        for b, n in enumerate(values):
            c = max(0, n - sink - tail)
            chosen = list(range(sink, sink + min(c, k)))
            expected = list(range(min(sink, n))) + chosen + list(range(min(sink, n) + c, n))
            count = int(outlens[b])
            assert count == len(expected)
            assert positions[b, :count].tolist() == expected
            expected_slots = [int(temp[b, p - sink]) if p in chosen else int(raw[rows[b], p]) for p in expected]
            assert slots[b, :count].tolist() == expected_slots
            assert recon[0].view(batch, k)[b].tolist() == chosen + [-1] * (k - len(chosen))
            assert recon[1].view(batch, k)[b].tolist() == [int(latent[rows[b], p]) for p in chosen] + [-1] * (k - len(chosen))
            assert recon[2].view(batch, k)[b].tolist() == temp[b, :len(chosen)].tolist() + [-1] * (k - len(chosen))


def _quest_inputs(device):
    # A non-page-aligned budget needs four dense pages although sparse k is two.
    page_size, budget, width, batch = 16, 50, 8, 8
    scores = torch.arange(width, dtype=torch.bfloat16, device=device).repeat(batch, 1)
    pages = torch.arange(batch * width, dtype=torch.int32, device=device).reshape(batch, width).flip(1).contiguous()
    lengths = torch.tensor([0, 1, 15, 16, 32, 48, 50, 100], dtype=torch.int32, device=device)
    outputs = (torch.empty((batch, 4), dtype=torch.int32, device=device),
               *(torch.empty(batch, dtype=torch.int32, device=device) for _ in range(4)))
    return page_size, budget, scores, pages, lengths, outputs


def _check_quest(pages, lengths, outputs):
    for b, n in enumerate(lengths.tolist()):
        count = (n + 15) // 16
        selected = list(range(count)) if n <= 50 or count <= 3 else [count - 3, count - 2, count - 1]
        assert int(outputs[3][b]) == len(selected) <= outputs[0].shape[1]
        assert outputs[0][b, :len(selected)].tolist() == pages[b, selected].tolist()
        assert torch.count_nonzero(outputs[0][b, len(selected):]) == 0
        assert int(outputs[2][b]) == (n if len(selected) == count else 32 + (n - 1) % 16 + 1)
        assert int(outputs[4][b]) == ((n - 1) % 16 + 1 if n else 0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_quest_paged_finalizer_preserves_partial_budget_and_empty_rows(device):
    page_size, budget, _, pages, lengths, outputs = _quest_inputs(device)
    counts = (lengths + page_size - 1) // page_size
    selected = torch.stack([pages[b, max(0, int(n) - 3):max(0, int(n) - 3) + 2] for b, n in enumerate(counts)])
    finalize_quest_paged_decode_view(
        selected.contiguous(), pages, counts, lengths, page_size=page_size,
        token_budget=budget, output_page_table=outputs[0], output_req_indices=outputs[1],
        output_context_lens=outputs[2], output_page_counts=outputs[3],
        output_last_page_lens=outputs[4], use_dense_fallback=True,
    )
    _check_quest(pages, lengths, outputs)


@CUDA
def test_quest_fused_mixed_selection_graph_replays_new_lengths():
    page_size, budget, scores, pages, lengths, outputs = _quest_inputs("cuda")
    previous = ((lengths + page_size - 1) // page_size - 1).clamp_min(0)

    def run():
        fused_exact_select_quest_paged_view(
            scores, pages, previous, lengths, k=2, page_size=page_size,
            token_budget=budget, use_dense_fallback=True,
            output_page_table=outputs[0], output_req_indices=outputs[1],
            output_context_lens=outputs[2], output_page_counts=outputs[3], output_last_page_lens=outputs[4],
        )

    run()
    _check_quest(pages, lengths, outputs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    lengths.copy_(lengths.flip(0))
    previous.copy_(((lengths + page_size - 1) // page_size - 1).clamp_min(0))
    graph.replay()
    _check_quest(pages, lengths, outputs)


def test_flashinfer_sparse_host_plan_matches_mixed_selected_lengths():
    # CPU ownership: the prepared host indptr must describe the exact device
    # view. A fixed sparse length silently made short rows read padding pages.
    state = object.__new__(FlashInferPagedDecodeGraphState)
    state.spec = SimpleNamespace(sparse_context_budget=50, page_size=16,
                                 num_query_heads=2, num_kv_heads=1, head_dim=16,
                                 softmax_scale=0.25, activation_dtype=torch.bfloat16)
    state.contract = SimpleNamespace(context_capacity=128)
    state.sparse_wrapper = Mock()
    state.host_sparse_indptr = torch.empty(9, dtype=torch.int32)
    state.sparse_indices = torch.empty(64, dtype=torch.int32)
    state.host_sparse_last_page_len = torch.empty(8, dtype=torch.int32)
    lengths = torch.tensor([1, 2, 16, 32, 48, 50, 51, 100], dtype=torch.int32)
    state._plan_sparse(lengths)
    selected_lengths = [1, 2, 16, 32, 48, 50, 35, 36]
    pages = [(n + 15) // 16 for n in selected_lengths]
    assert state.host_sparse_indptr.tolist() == [0] + [sum(pages[:i]) for i in range(1, 9)]
    assert state.host_sparse_last_page_len.tolist() == [(n - 1) % 16 + 1 for n in selected_lengths]
