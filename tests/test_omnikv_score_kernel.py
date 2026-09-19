"""Independent probability oracle and replay lifecycle for fused OmniKV scores."""
import pytest
import torch
import triton

from sparseengine.kernels.triton.omnikv_score import launch_omnikv_decode_scores

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def oracle(raw, lengths, sink, recent, scale, dtype):
    # Deliberately normalize row-by-row in float64, independently of tiling.
    result = torch.full((raw.shape[0], raw.shape[2]), torch.finfo(dtype).min, device=raw.device, dtype=dtype)
    for row, length in enumerate(lengths.cpu().tolist()):
        end = min(max(length - recent, sink), raw.shape[2])
        if end > sink:
            values = raw[row, :, sink:end].double() * scale
            result[row, sink:end] = values.softmax(-1).amax(0).to(dtype)
    return result


def make_call(raw, lengths, sink, recent, dtype):
    b, h, c = raw.shape
    partial = torch.empty((b, h, triton.cdiv(c-sink, 1024), 2), device=raw.device)
    stats = torch.empty((b, h, 2), device=raw.device)
    output = torch.empty((b, c), dtype=dtype, device=raw.device)
    def run():
        launch_omnikv_decode_scores(raw, lengths, partial, stats, output,
                                    sink=sink, recent=recent, scale=.0625,
                                    min_score=torch.finfo(dtype).min)
        return output
    return run, (partial, stats, output)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize('heads,capacity,strided', [(1, 67, False), (3, 4099, True), (10, 202752, False)])
def test_ragged_scores_exclude_sink_recent_and_poisoned_padding(dtype, heads, capacity, strided):
    # Catch reading stale score tails, normalizing over sink/recent, or reducing
    # heads before softmax. Include empty candidates and noncontiguous strides.
    torch.manual_seed(17)
    sink, recent = 5, 13
    raw = torch.randn(4, heads, capacity * (2 if strided else 1), device='cuda') * 16
    raw = raw[..., ::2] if strided else raw
    lengths = torch.tensor([0, sink+recent, capacity//2, capacity], device='cuda', dtype=torch.int32)
    for row, length in enumerate(lengths.cpu().tolist()):
        raw[row, :, :sink] = float('nan')
        raw[row, :, max(sink, length-recent):] = float('nan')
    run, keepalive = make_call(raw, lengths, sink, recent, dtype)
    expected = oracle(raw, lengths, sink, recent, .0625, dtype)
    torch.testing.assert_close(run(), expected, rtol=2e-5 if dtype == torch.float32 else .008, atol=1e-8)
    assert keepalive[-1].data_ptr() == run().data_ptr()


def test_graph_replay_updates_lengths_and_scores_without_stale_output():
    # A single captured graph must survive empty/full/ragged/padded transitions
    # without stale reductions, new addresses, or host-length specialization.
    raw = torch.randn(4, 10, 65539, device='cuda') * 16
    lengths = torch.full((4,), 65539, device='cuda', dtype=torch.int32)
    run, buffers = make_call(raw, lengths, 64, 512, torch.bfloat16)
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    pointers = [x.data_ptr() for x in buffers]
    for values in ([0, 1, 575, 576], [577, 2047, 32768, 65539], [65539]*4, [8192, 0, 32769, 1025]):
        lengths.copy_(torch.tensor(values, device='cuda', dtype=torch.int32))
        raw.normal_(0, 16)
        graph.replay()
        expected = oracle(raw, lengths, 64, 512, .0625, torch.bfloat16)
        torch.testing.assert_close(buffers[-1], expected, rtol=.008, atol=1e-8)
        assert pointers == [x.data_ptr() for x in buffers]


def test_head_normalization_cannot_be_replaced_by_raw_head_max():
    raw = torch.tensor([[[10., 9.], [0., 2.]]], device='cuda') * 16
    lengths = torch.tensor([2], device='cuda', dtype=torch.int32)
    run, _ = make_call(raw, lengths, 0, 0, torch.float32)
    assert run().argmax(-1).item() == 1
    assert raw.amax(1).argmax(-1).item() == 0


def test_prepared_provider_keeps_observers_and_old_graph_storage_independent():
    # Later observers share raw QK but must not overwrite earlier reduced scores.
    # Growing/clearing the prepared provider must not invalidate captured graphs.
    from sparseengine.operators.omnikv_score import OmniKVScoreSpec, prepare_omnikv_score_provider

    raw = torch.randn(2, 3, 4099, device='cuda') * 16
    lengths = torch.tensor([4099, 1234], device='cuda', dtype=torch.int32)
    provider = prepare_omnikv_score_provider(OmniKVScoreSpec(5, 13, .0625, torch.bfloat16), device=raw.device)
    for slot in (0, 2):
        provider.prepare(raw, slot=slot)
        provider.run(raw, lengths, slot=slot)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        first = provider.run(raw, lengths, slot=0)
        second = provider.run(raw + 1, lengths, slot=2)
    retained = provider.keepalive_tensors()
    assert first.data_ptr() != second.data_ptr()
    old = first.clone()
    provider.run(raw * 2, lengths, slot=2)
    torch.testing.assert_close(first, old, rtol=0, atol=0)
    provider.prepare(torch.empty(4, 3, 8192, device='cuda'), slot=0)
    assert provider.workspaces[0].output.data_ptr() != first.data_ptr()
    provider.clear()
    raw.normal_(0, 16)
    lengths.copy_(torch.tensor([0, 3000], device='cuda', dtype=torch.int32))
    graph.replay()
    torch.testing.assert_close(first, oracle(raw, lengths, 5, 13, .0625, torch.bfloat16), rtol=.008, atol=1e-8)
    torch.testing.assert_close(second, oracle(raw + 1, lengths, 5, 13, .0625, torch.bfloat16), rtol=.008, atol=1e-8)
    assert retained


@pytest.mark.parametrize('raw_dtype', [torch.float16, torch.bfloat16])
def test_prepared_provider_handles_extreme_logits_and_noncontiguous_lengths(raw_dtype):
    from sparseengine.operators.omnikv_score import OmniKVScoreSpec, prepare_omnikv_score_provider

    raw = torch.tensor([[[10000., -10000., 9992., 0.], [-10000., 10000., 0., 9992.]]], device='cuda', dtype=raw_dtype).expand(2, -1, -1)
    lengths = torch.tensor([4, 99, 0, 99], device='cuda', dtype=torch.int64)[::2]
    provider = prepare_omnikv_score_provider(OmniKVScoreSpec(0, 0, .0625, torch.float32), device=raw.device)
    provider.prepare(raw, slot=0)
    torch.testing.assert_close(provider.run(raw, lengths, slot=0), oracle(raw, lengths, 0, 0, .0625, torch.float32), rtol=2e-5, atol=1e-8)


def test_prepared_provider_empty_candidate_capacity():
    from sparseengine.operators.omnikv_score import OmniKVScoreSpec, prepare_omnikv_score_provider

    raw = torch.full((1, 2, 4), float('nan'), device='cuda')
    lengths = torch.tensor([4], device='cuda', dtype=torch.int32)
    provider = prepare_omnikv_score_provider(OmniKVScoreSpec(4, 0, 1., torch.bfloat16), device=raw.device)
    provider.prepare(raw, slot=0)
    torch.testing.assert_close(provider.run(raw, lengths, slot=0), oracle(raw, lengths, 4, 0, 1., torch.bfloat16), rtol=0, atol=0)
