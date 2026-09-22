"""Independent numerical and lifecycle regressions for compressed KV pages."""

import copy
import math
import os
from types import SimpleNamespace

import pytest
import torch

from sparseengine.configs.kv_quant import validate_quantized_kv
from sparseengine.engine.cache_manager.quantized_pages import QuantizedPagePool
from sparseengine.engine.cache_manager.storage.quantized_kv import (
    QuantizedKVStorage, gaussian_codebook, orthogonal_rotation,
)
from sparseengine.kernels.triton.quantized_kv import encode_pages, quantized_decode_append, materialize_sequence, quantized_decode
from sparseengine.operators.decode_attention import DecodeAttentionOpSpec, DECODE_ATTENTION_REGISTRY
from sparseengine.operators.registry import OpResolver
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum


def test_failed_append_is_atomic_and_release_reuses_pages():
    pool = QuantizedPagePool(3, 32, 128)
    first = pool.append(8, 31)
    assert pool.append_cost(8, 1) == 0
    before = copy.deepcopy(pool.__dict__)
    with pytest.raises(RuntimeError, match="Out of"):
        pool.append(9, 97)
    assert pool.__dict__ == before
    with pytest.raises(ValueError, match="max_model_len"):
        pool.append(8, 128)
    assert pool.__dict__ == before
    pool.append(8, 2)
    second = pool.append(9, 32)
    assert not set(pool.pages[8]) & set(second.pages)
    assert first.pages[0] == pool.pages[8][0]
    pool.release(8)
    pool.release(9)
    assert len(pool.free) == len(set(pool.free)) == 3
    pool.append(10, 96)
    assert not pool.free


def test_single_token_append_returns_only_touched_history_page():
    pool = QuantizedPagePool(5000, 16, 100000)
    pool.append(8, 65535)
    assert pool.append(8, 0).pages == ()
    plan = pool.append(8, 1)
    assert plan.start == 65535 and plan.end == 65536
    assert len(pool.pages[8]) == 4096
    assert plan.pages == (pool.pages[8][-1],)

    other = QuantizedPagePool(5000, 16, 100000)
    other.append(9, 65535)
    crossing = other.append(9, 2)
    assert crossing.pages == tuple(other.pages[9][-2:])


def test_rotation_is_orthogonal_and_does_not_change_global_rng():
    state = torch.random.get_rng_state().clone()
    rotation = orthogonal_rotation(64, 19)
    torch.testing.assert_close(rotation.T @ rotation, torch.eye(64), atol=1e-6, rtol=1e-6)
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(rotation, orthogonal_rotation(64, 19))
    for bits in (2, 3, 4):
        centers = torch.tensor(gaussian_codebook(bits))
        assert torch.all(centers[1:] > centers[:-1])
        torch.testing.assert_close(centers, -centers.flip(0))


def test_rotation_construction_is_independent_of_default_device():
    # Model startup sets a CUDA default device; metadata generation stays on CPU.
    with torch.device("meta"):
        rotation = orthogonal_rotation(64, 0)
    assert rotation.device.type == "cpu"
    torch.testing.assert_close(rotation.T @ rotation, torch.eye(64), atol=1e-6, rtol=1e-6)


def _config(**changes):
    config = dict(sparse_method="kivi", kivi_bits=4, turboquant_bits=3, turboquant_seed=0,
                  kv_quant_page_size=32, hf_config=SimpleNamespace(model_type="llama", head_dim=64,
                                                                 dtype=torch.bfloat16),
                  attention_cache_layout="explicit_kv", tensor_parallel_size=1,
                  expert_parallel_size=1, data_parallel_size=1, decode_graph=False,
                  enable_prefix_caching=False, enable_prefix_cache_offload=False, prefill_sparse_method="")
    config.update(changes)
    return SimpleNamespace(**config)


@pytest.mark.parametrize("changes,match", [
    ({"kivi_bits": True}, "kivi_bits"), ({"turboquant_bits": 5}, "turboquant_bits"),
    ({"kv_quant_page_size": 17}, "page_size"),
    ({"enable_prefix_caching": True}, "prefix"),
    ({"prefill_sparse_method": "h2o_prefill"}, "dense prefill"),
])
def test_unsupported_storage_contract_fails_before_allocation(changes, match):
    with pytest.raises(ValueError, match=match):
        validate_quantized_kv(_config(**changes))


@pytest.mark.parametrize("method", ["kivi", "turboquant", "fp8_kv"])
@pytest.mark.parametrize("graph", [False, True])
def test_parallel_budget_matches_local_attention_heads(tmp_path, method, graph):
    """TP halves head-owned storage; changing EP must not change attention KV."""
    from transformers import Qwen3MoeConfig
    from sparseengine.config import Config
    from sparseengine.engine.startup.capacity import profiling_kv_budget_bytes

    hf = Qwen3MoeConfig(hidden_size=256, intermediate_size=512,
                        moe_intermediate_size=256, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                        num_experts=4, num_experts_per_tok=2,
                        vocab_size=128, max_position_embeddings=160, dtype="bfloat16")
    hf.save_pretrained(tmp_path)
    options = dict(model=str(tmp_path), sparse_method=method, max_model_len=160,
                   max_num_batched_tokens=128, max_num_seqs_in_batch=2,
                   max_decoding_seqs=2, max_num_seqs_in_gpu=2, decode_graph=graph)
    single = Config(**options)
    tp = Config(**options, tensor_parallel_size=2)
    hybrid = Config(**options, tensor_parallel_size=2, expert_parallel_size=2)
    budget = lambda config: profiling_kv_budget_bytes(config, 96)
    assert budget(hybrid) == budget(tp) < budget(single)
    # Independent shape oracle: a TP=1 model with the same local KV head count.
    hf.num_key_value_heads = 1
    hf.save_pretrained(tmp_path)
    assert budget(Config(**options)) == budget(tp)


def test_quantization_preserves_model_parallel_validation(tmp_path):
    """Opening quantization must not bypass illegal head/expert shards or model DP."""
    from transformers import Qwen3MoeConfig
    from sparseengine.config import Config

    Qwen3MoeConfig(hidden_size=256, intermediate_size=512, moe_intermediate_size=256,
                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                   head_dim=64, num_experts=4, num_experts_per_tok=2,
                   vocab_size=128, max_position_embeddings=160,
                   dtype="bfloat16").save_pretrained(tmp_path)
    options = dict(model=str(tmp_path), sparse_method="kivi", max_model_len=160)
    with pytest.raises(ValueError, match="num_key_value_heads"):
        Config(**options, tensor_parallel_size=4)
    with pytest.raises(ValueError, match="must be divisible by MoE EP"):
        Config(**options, expert_parallel_size=3)
    with pytest.raises(ValueError, match="[Dd]ata|DP"):
        Config(**options, data_parallel_size=2)
    # Independent replicas are not a quantization restriction; internal topology
    # eligibility remains owned by ModelSpec, not this storage validator.
    validate_quantized_kv(_config(data_parallel_size=2))


def test_decode_resolution_rejects_dense_provider_for_compressed_payload():
    caps = DeviceCaps(platform=PlatformEnum.CUDA, device_type="cuda", device_index=0,
                      device_name="contract-test", compute_capability=(9, 0),
                      supports_triton=True, supports_native_fp8=True)
    spec = DecodeAttentionOpSpec(num_query_heads=4, num_kv_heads=2, head_dim=64,
                                activation_dtype=torch.bfloat16, softmax_scale=0.125,
                                max_batch_size=2, cuda_graph=False, context_capacity=256,
                                kv_storage_format="fp8_kv")
    selected = OpResolver(DECODE_ATTENTION_REGISTRY).resolve(spec, caps)
    assert selected.provider.supports(spec, caps).supported
    # Failure to preserve the new format would bind a dense attention kernel.
    assert selected.provider.name == "triton_quantized_pages_decode"
    unavailable = DeviceCaps(platform=PlatformEnum.CUDA, device_type="cuda", device_index=0,
                             device_name="no-fp8", supports_triton=True, supports_native_fp8=False)
    with pytest.raises(RuntimeError, match="FP8"):
        OpResolver(DECODE_ATTENTION_REGISTRY).resolve(spec, unavailable)


@pytest.fixture
def kernel_device():
    if os.environ.get("TRITON_INTERPRET") == "1":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        pytest.skip("requires an idle CUDA device or explicit TRITON_INTERPRET=1")
    return torch.device("cuda")


def _reference_page(x, format, bits, *, key, codebook):
    g, heads, dim = x.shape
    x = x.float()
    if format == "kivi":
        grouped = x if key else x.reshape(g, heads, dim // g, g)
        axis = 0 if key else -1
        minimum = grouped.amin(axis, keepdim=True)
        scale = ((grouped.amax(axis, keepdim=True) - minimum) / ((1 << bits) - 1)).clamp_min(1e-30)
        codes = torch.floor((grouped - minimum) / scale + 0.5).clamp(0, (1 << bits) - 1)
        return (codes * scale + minimum).reshape_as(x)
    if format == "fp8_kv":
        scale = (x.abs().amax(-1, keepdim=True) / 448).clamp_min(1e-30)
        return (x / scale).to(torch.float8_e4m3fn).float() * scale
    scale = x.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-30)
    distances = ((x / scale)[..., None] - codebook).abs()
    codes = distances.argmin(-1)
    return codebook[codes] * scale


@pytest.mark.parametrize("format,bits,g,d,dtype", [
    ("fp8_kv", 8, 16, 64, torch.float16), ("fp8_kv", 8, 32, 128, torch.bfloat16),
    ("fp8_kv", 8, 128, 256, torch.bfloat16),
    ("kivi", 2, 32, 128, torch.bfloat16), ("kivi", 4, 128, 256, torch.bfloat16),
    ("turboquant", 2, 32, 128, torch.bfloat16), ("turboquant", 3, 32, 128, torch.bfloat16),
    ("turboquant", 4, 128, 256, torch.bfloat16),
])
@pytest.mark.parametrize("graph", [False, True])
def test_batched_decode_append_preserves_page_bytes_and_raw_tails(kernel_device, format, bits, g, d, dtype, graph):
    """Catch mixed page crossings, row reuse, strided QKV and old-tail races.

    The oracle is the original concatenate/page-codec/writeback sequence, not
    a duplicate of the batched writer's indexing or conditional stores.
    """
    if kernel_device.type != "cuda":
        pytest.skip("Capture and codec byte equivalence require CUDA")
    torch.manual_seed(619)
    batch, h = 5, 2
    stores = [QuantizedKVStorage(format=format, bits=bits, page_size=g, num_kv_heads=h,
                                head_dim=d, dtype=dtype, seed=0) for _ in range(2)]
    for storage in stores:
        storage.allocate(num_layers=1, num_slots=batch * 4 * g, num_rows=batch, device=kernel_device)
        storage.data.zero_()
    fast, reference = [storage.layer_payload(0) for storage in stores]
    for name in ("raw_key", "raw_value", "key_scale", "value_scale", "key_min", "value_min"):
        tensor = getattr(fast, name)
        tensor.copy_(torch.randn_like(tensor))
        getattr(reference, name).copy_(tensor)
    rows = torch.tensor([4, 0, 3, 1, 2], dtype=torch.int32, device=kernel_device)
    starts = [0, g - 2, g - 1, g, 2 * g - 1]
    pages = torch.randperm(batch * 4, device=kernel_device).reshape(batch, 4).to(torch.int32)
    slots = torch.empty(batch, 4 * g + 7, device=kernel_device, dtype=torch.int32)[:, :4 * g]
    slots.copy_((pages[:, :, None] * g + torch.arange(g, device=kernel_device)).reshape(batch, -1))
    # Stable strided inputs; the padded last row aliases the first live row.
    key = torch.empty(batch + 1, h, 2 * d, device=kernel_device, dtype=dtype)[..., ::2]
    value = torch.empty(h, batch + 1, d, device=kernel_device, dtype=dtype).transpose(0, 1)
    live_rows = rows.tolist()
    rows = torch.cat((rows, rows[:1]))
    ends = torch.ones(batch + 1, dtype=torch.int32, device=kernel_device)
    writes = torch.full_like(ends, -1)
    def append():
        quantized_decode_append(key, value, fast, slots, rows, ends, writes)
    append()  # Compile with writes disabled, without mutating reference state.
    if graph:
        torch.cuda.synchronize()
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            append()
    for step in range(g + 2):
        # Distinct strides for K and V, including non-unit channel stride.
        key.copy_(torch.randn_like(key))
        value.copy_(torch.randn_like(value))
        key[0].zero_()
        value[1].mul_(1e-7)
        ends.copy_(torch.tensor([start + step + 1 for start in starts] + [step + 1],
                               dtype=torch.int32, device=kernel_device))
        writes[:batch] = slots[rows[:batch].long(), (ends[:batch] - 1).long()]
        if graph:
            captured.replay()
        else:
            append()
        for b, row in enumerate(live_rows):
            end = starts[b] + step + 1
            previous = (end - 1) % g
            k = torch.cat((reference.raw_key[row, :previous], key[b:b + 1]))
            v = torch.cat((reference.raw_value[row, :previous], value[b:b + 1]))
            if k.shape[0] == g:
                encode_pages(k, v, pages[row, end // g - 1:end // g], reference)
            else:
                reference.raw_key[row, :end % g].copy_(k)
                reference.raw_value[row, :end % g].copy_(v)
    for name in ("k_cache", "v_cache"):
        assert torch.equal(getattr(fast, name).view(torch.uint8), getattr(reference, name).view(torch.uint8))
    for name in ("raw_key", "raw_value", "key_scale", "value_scale", "key_min", "value_min"):
        torch.testing.assert_close(getattr(fast, name), getattr(reference, name), atol=0, rtol=0)


def _unpack_page(payload, page, *, key):
    data = payload.k_cache[page] if key else payload.v_cache[page]
    scale = payload.key_scale[page] if key else payload.value_scale[page]
    dim = payload.raw_key.shape[-1]
    if payload.format == "fp8_kv":
        return data.float() * scale[..., None]
    feature = torch.arange(dim, device=data.device)
    codes = (data[..., feature // (32 // payload.bits)] >>
             ((feature % (32 // payload.bits)) * payload.bits)) & ((1 << payload.bits) - 1)
    if payload.format == "turboquant":
        return payload.codebook[codes.long()] * scale[..., None]
    minimum = payload.key_min[page] if key else payload.value_min[page]
    if not key:
        scale = scale.repeat_interleave(payload.page_size, -1)
        minimum = minimum.repeat_interleave(payload.page_size, -1)
    return codes.float() * scale + minimum


@pytest.mark.parametrize("method", ["kivi", "turboquant", "fp8_kv"])
def test_manager_chunk_append_and_free_preserve_history(tmp_path, kernel_device, method):
    """Exercise real config/factory and raw-tail transitions across chunk boundaries."""
    from transformers import LlamaConfig
    from sparseengine.config import Config
    from sparseengine.distributed.parallel_context import ParallelContext, ParallelGroup
    from sparseengine.engine.cache_manager import CacheManager, ExplicitKVWrite, SparseSelection
    from sparseengine.engine.sequence import Sequence

    if kernel_device.type == "cpu" and method == "fp8_kv":
        pytest.skip("FP8 conversion requires CUDA; Triton interpreter has a known rounding bug")
    LlamaConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                vocab_size=128, max_position_embeddings=160, dtype="float16").save_pretrained(tmp_path)
    config = Config(model=str(tmp_path), sparse_method=method, max_model_len=160,
                    max_num_batched_tokens=256, engine_prefill_chunk_size=32,
                    max_num_seqs_in_batch=2, max_decoding_seqs=2, max_num_seqs_in_gpu=2,
                    decode_graph=False)
    group = ParallelGroup(None, (0,), 0, 1)
    from sparseengine.engine.startup.capacity import profiling_kv_budget_bytes, profiling_kv_slots
    config.startup_cache_phase = "profiling"
    manager = CacheManager.create(config, ParallelContext(world=group, attn_tp=group, moe_ep=group, attn_dp=group, moe_tp=group),
                                  allocation_budget_bytes=profiling_kv_budget_bytes(config, profiling_kv_slots(config)))
    seq = Sequence(list(range(67)))
    torch.manual_seed(15)
    key = torch.randn(67, 2, 64, device=kernel_device, dtype=torch.float16)
    value = torch.randn_like(key)
    initial_free = manager.num_free_slots
    for start, end in ((0, 17), (17, 33), (33, 67)):
        previous = None
        if start:
            previous = (torch.empty(start, 2, 64, device=kernel_device, dtype=torch.float32),
                        torch.empty(start, 2, 64, device=kernel_device, dtype=torch.float32))
            payload = manager.attention_cache_storage.layer_payload(0)
            materialize_sequence(payload, manager.buffer_req_to_token_slots,
                                 manager.seq_id_to_row[seq.seq_id], start, *previous)
            if payload.rotation is not None:
                previous = tuple(t @ payload.rotation.T for t in previous)
        seq.current_chunk_size = end - start
        manager._prepare_prefill([seq])
        manager.store_attention_payload(0, ExplicitKVWrite(key[start:end], value[start:end]))
        state = manager.get_layer_batch_states(0)
        selection = SparseSelection(kind="full", req_indices=state.req_indices,
                                    context_lens=state.context_lens, max_context_len=end)
        view = manager.build_prefill_compute_view(0, key[start:end], value[start:end], selection)
        torch.testing.assert_close(view.payload.k_cache[start:end], key[start:end], atol=0, rtol=0)
        torch.testing.assert_close(view.payload.v_cache[start:end], value[start:end], atol=0, rtol=0)
        if previous:
            torch.testing.assert_close(view.payload.k_cache[:start], previous[0].half(), atol=0, rtol=0)
            torch.testing.assert_close(view.payload.v_cache[:start], previous[1].half(), atol=0, rtol=0)
        seq.num_prefilled_tokens = end
    for step in range(31):
        seq.append_token(2)
        # Eager engine decode uses this stable-buffer path even without graphs.
        batch = 1 + step % 2  # Alternate exact and padded stable batches.
        manager._prepare_decode_graph_buffers(
            [seq], input_ids=torch.empty(batch, dtype=torch.int64, device=kernel_device),
            positions=torch.empty(batch, dtype=torch.int64, device=kernel_device),
            slot_mapping=torch.empty(batch, dtype=torch.int32, device=kernel_device),
            context_lens=torch.empty(batch, dtype=torch.int32, device=kernel_device),
            req_indices=torch.empty(batch, dtype=torch.int32, device=kernel_device),
        )
        new = torch.randn(batch, 2, 64, dtype=torch.float16, device=kernel_device)
        manager.store_attention_payload(0, ExplicitKVWrite(new, new))
    expected_pages = math.ceil((67 + 31) / 32)
    assert initial_free - manager.num_free_slots == expected_pages * 32
    manager.free_seq(seq.seq_id)
    assert manager.num_free_slots == initial_free
    assert not manager.seq_id_to_row
    manager.reset_after_warmup()


@pytest.mark.parametrize("format,bits", [("kivi", 2), ("kivi", 4), ("turboquant", 2),
                                         ("turboquant", 3), ("turboquant", 4), ("fp8_kv", 8)])
@pytest.mark.parametrize("g,d,dtype", [(32, 64, torch.float16), (16, 128, torch.bfloat16),
                                      (128, 256, torch.bfloat16)])
def test_packed_page_attention_matches_independent_reference(kernel_device, format, bits, g, d, dtype):
    """Catch packing, scale-axis, raw-tail, GQA and noncontiguous row-map errors."""
    device = kernel_device
    torch.manual_seed(42)
    h, qh = 2, 4
    storage = QuantizedKVStorage(format=format, bits=bits, page_size=g, num_kv_heads=h,
                                 head_dim=d, dtype=dtype, seed=7)
    storage.allocate(num_layers=1, num_slots=8 * g, num_rows=3, device=device)
    payload = storage.layer_payload(0)
    # The map is deliberately not contiguous and active batch rows are reordered.
    capacity = 5 * g
    slots = torch.zeros(3, capacity + 32, dtype=torch.int32, device=device)[:, :capacity]
    rows = torch.tensor([2, 0], dtype=torch.int32, device=device)
    lengths = torch.tensor([4 * g + 5, g - 1], dtype=torch.int32, device=device)
    references = []
    for row, length, page_ids in [(2, 4 * g + 5, [6, 1, 4, 0, 3]), (0, g - 1, [7])]:
        key = torch.randn(length, h, d, dtype=dtype, device=device)
        value = torch.randn_like(key)
        key[:, 0, 0] = 0  # constant channel; tests zero-range scale handling.
        slot_list = [page_ids[t // g] * g + t % g for t in range(length)]
        slots[row, :length] = torch.tensor(slot_list, dtype=torch.int32, device=device)
        full = length // g * g
        if full:
            encode_pages(key[:full], value[:full], torch.tensor(page_ids[:length // g],
                         dtype=torch.int32, device=device), payload)
            if format == "fp8_kv" and device.type == "cpu":
                pytest.xfail("Triton 3.6 interpreter FP32-to-E4M3 rounding loses exponent carry; requires CUDA validation")
        payload.raw_key[row].fill_(float("nan"))
        payload.raw_value[row].fill_(float("nan"))
        payload.raw_key[row, :length - full] = key[full:]
        payload.raw_value[row, :length - full] = value[full:]
        kr, vr = key.float().clone(), value.float().clone()
        for start in range(0, full, g):
            for source, restored, is_key in ((key, kr, True), (value, vr, False)):
                block = source[start:start + g].float()
                ideal = _reference_page(block, format, bits, key=is_key, codebook=payload.codebook)
                unpacked = _unpack_page(payload, page_ids[start // g], key=is_key)
                # At an exact bin midpoint, FP32 divide rounding may choose either
                # neighbor. Both must attain the independently computed nearest error.
                torch.testing.assert_close((unpacked - block).abs(), (ideal - block).abs(), atol=3e-5, rtol=3e-5)
                restored[start:start + g] = unpacked
        ko, vo = torch.empty_like(kr), torch.empty_like(vr)
        materialize_sequence(payload, slots, row, length, ko, vo)
        torch.testing.assert_close(ko, kr, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(vo, vr, atol=2e-5, rtol=2e-5)
        references.append((kr, vr))
    q = torch.randn(2, qh, d, dtype=dtype, device=device)
    splits = math.ceil((4 * g + 5) / 128)
    mid_o = torch.empty(2, qh, splits, d, device=device, dtype=torch.float32)
    mid_lse = torch.empty(2, qh, splits, device=device, dtype=torch.float32)
    output = torch.empty_like(q)
    output_lse = torch.empty(qh, 2, device=device, dtype=torch.float32)
    actual = quantized_decode(q, payload, slots, rows, lengths, mid_o, mid_lse,
                             softmax_scale=d ** -0.5, output=output, output_lse=output_lse)
    for b, (key, value) in enumerate(references):
        k = key.repeat_interleave(qh // h, 1).transpose(0, 1)
        v = value.repeat_interleave(qh // h, 1).transpose(0, 1)
        scores = torch.einsum("hd,hnd->hn", q[b].float(), k) / math.sqrt(d)
        expected = torch.einsum("hn,hnd->hd", scores.softmax(-1), v)
        tolerance = 2e-3 if dtype == torch.float16 else 1e-2
        torch.testing.assert_close(actual[b].float(), expected, atol=tolerance, rtol=tolerance)
    # Count physical words and scales, including int3 padding, against allocation.
    page_tensors = (storage.data, storage.key_scale, storage.value_scale, storage.key_min, storage.value_min)
    assert sum(t.numel() * t.element_size() for t in page_tensors) == 8 * storage.bytes_per_page_per_layer()


@pytest.mark.parametrize("method,bits", [("kivi", 4), ("turboquant", 3), ("fp8_kv", 8)])
def test_gqa_head_shards_match_unsharded_quantization(kernel_device, method, bits):
    """Independent head shards must preserve encoding, rotation and GQA decoding."""
    device = kernel_device
    if device.type == "cpu" and method == "fp8_kv":
        pytest.skip("FP8 rounding requires CUDA")
    torch.manual_seed(37)
    g, d, length = 32, 64, 69
    key = torch.randn(length, 4, d, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    query = torch.randn(1, 8, d, device=device, dtype=torch.bfloat16)

    def run(k, v, q):
        storage = QuantizedKVStorage(format=method, bits=bits, page_size=g,
                                     num_kv_heads=k.shape[1], head_dim=d,
                                     dtype=k.dtype, seed=19)
        storage.allocate(num_layers=1, num_slots=3 * g, num_rows=1, device=device)
        payload = storage.layer_payload(0)
        if payload.rotation is not None:
            k = (k.float() @ payload.rotation).to(k.dtype)
            v = (v.float() @ payload.rotation).to(v.dtype)
            q = (q.float() @ payload.rotation).to(q.dtype)
        encode_pages(k[:64], v[:64], torch.tensor([0, 1], dtype=torch.int32, device=device), payload)
        payload.raw_key[0, :5] = k[64:]
        payload.raw_value[0, :5] = v[64:]
        slots = torch.arange(3 * g, dtype=torch.int32, device=device)[None]
        rows = torch.zeros(1, dtype=torch.int32, device=device)
        lengths = torch.tensor([length], dtype=torch.int32, device=device)
        mid_o = torch.empty(1, q.shape[1], 1, d, dtype=torch.float32, device=device)
        mid_lse = torch.empty(1, q.shape[1], 1, dtype=torch.float32, device=device)
        output = torch.empty_like(q)
        quantized_decode(q, payload, slots, rows, lengths, mid_o, mid_lse,
                         softmax_scale=d ** -0.5, output=output,
                         output_lse=torch.empty(q.shape[1], 1, device=device))
        if payload.rotation is not None:
            output = (output.float() @ payload.rotation.T).to(output.dtype)
        return output, _unpack_page(payload, 0, key=True), _unpack_page(payload, 0, key=False)

    whole = run(key, value, query)
    shards = [run(key[:, i:i + 2].contiguous(), value[:, i:i + 2].contiguous(),
                  query[:, 2 * i:2 * i + 4].contiguous()) for i in (0, 2)]
    for index in range(3):
        torch.testing.assert_close(torch.cat([shard[index] for shard in shards], dim=1),
                                   whole[index], atol=0, rtol=0)


@pytest.mark.parametrize("method,bits,d", [("kivi", 4, 64), ("kivi", 4, 128),
                                         ("kivi", 2, 256), ("turboquant", 3, 64), ("fp8_kv", 8, 64)])
def test_provider_graph_replay_reads_live_lengths_and_rows(kernel_device, method, bits, d):
    """Catch stale inactive splits when a large captured grid changes live lengths."""
    from sparseengine.operators.decode_attention import QuantizedPagesDecodeAttentionProvider

    if kernel_device.type != "cuda":
        pytest.skip("requires CUDA Graph replay")
    torch.manual_seed(31)
    g, h, qh, capacity = 32, 2, 4, 160
    storage = QuantizedKVStorage(format=method, bits=bits, page_size=g,
                                 num_kv_heads=h, head_dim=d, dtype=torch.bfloat16, seed=7)
    storage.allocate(num_layers=1, num_slots=2 * capacity, num_rows=2, device=kernel_device)
    payload = storage.layer_payload(0)
    key = torch.randn(2 * capacity, h, d, device=kernel_device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    encode_pages(key, value, torch.arange(10, device=kernel_device, dtype=torch.int32), payload)
    payload.raw_key.normal_()
    payload.raw_value.normal_()
    slots = torch.arange(2 * capacity, device=kernel_device, dtype=torch.int32).view(2, capacity)
    rows = torch.tensor([0, 1, 0], device=kernel_device, dtype=torch.int32)
    lengths = torch.ones(3, device=kernel_device, dtype=torch.int32)
    q = torch.randn(3, qh, d, device=kernel_device, dtype=torch.bfloat16)
    spec = DecodeAttentionOpSpec(num_query_heads=qh, num_kv_heads=h, head_dim=d,
                                activation_dtype=q.dtype, softmax_scale=d ** -0.5,
                                max_batch_size=3, context_capacity=40960,
                                cuda_graph=True, kv_storage_format=method)
    provider = QuantizedPagesDecodeAttentionProvider()
    provider.prepare(spec, device_index=kernel_device.index)
    meta = SimpleNamespace(attn_score=None, max_context_len=1, active_slots=slots,
                           req_indices=rows, context_lens=lengths)
    view = SimpleNamespace(payload=payload, meta=meta)
    provider.run(spec, q, view)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = provider.run(spec, q, view)
    # Empty subtiles/splits must not introduce NaNs during online softmax;
    # D=128/256 also exercises the long-head decode used by real GQA models.
    for step, live in enumerate(([15, 17, 15], [31, 32, 31], [127, 129, 127],
                                 [160, 159, 160], [1, 33, 1], [0, 33, 1])):
        row_ids = [step % 2, 1 - step % 2, step % 2]
        rows.copy_(torch.tensor(row_ids, device=kernel_device, dtype=torch.int32))
        lengths.copy_(torch.tensor(live, device=kernel_device, dtype=torch.int32))
        q.normal_()
        provider.mid_o.fill_(float("nan"))
        provider.mid_lse.fill_(float("nan"))
        graph.replay()
        for b, (row, length) in enumerate(zip(row_ids, live)):
            if length == 0:
                torch.testing.assert_close(actual[b], torch.zeros_like(actual[b]), atol=0, rtol=0)
                continue
            full = length // g
            history = []
            for is_key in (True, False):
                raw = payload.raw_key if is_key else payload.raw_value
                pages = [_unpack_page(payload, row * 5 + page, key=is_key) for page in range(full)]
                history.append(torch.cat([*pages, raw[row, :length % g].float()]))
            kr, vr = [x.repeat_interleave(qh // h, dim=1).transpose(0, 1) for x in history]
            query = q[b] if payload.rotation is None else (q[b].float() @ payload.rotation).to(q.dtype)
            probability = (torch.einsum("hd,hnd->hn", query.float(), kr) * spec.softmax_scale).softmax(-1)
            expected = torch.einsum("hn,hnd->hd", probability, vr).to(q.dtype)
            if payload.rotation is not None:
                expected = (expected.float() @ payload.rotation.T).to(q.dtype)
            torch.testing.assert_close(actual[b], expected, atol=1e-2, rtol=1e-2)
    provider.close()
