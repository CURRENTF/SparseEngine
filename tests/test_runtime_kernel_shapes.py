"""Length changes must not create JIT variants or corrupt dynamic indexing.

Existing fixed-shape numerical tests do not catch exact-length specialization.
These CUDA tests pair a Torch oracle with real Triton compile notifications.
"""
import importlib

import pytest
import torch
import triton

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

MODULES = {
    "omnikv": "omnikv_score",
    "quest": "quest_page_score",
    "gather": "deltakv_kernels",
    "minmax": "quant",
    "materialize": "deltakv_kernels",
}
TARGETS = {
    "omnikv": {"_partial_stats", "_merge_stats", "_normalize_head_max"},
    "quest": {"_gqa_page_bounds", "_repair_and_merge_page_bounds", "_vector_page_bounds", "_max_head_bounds"},
    "gather": {"_batch_gather_mean_kernel"},
    "minmax": {"_minmax_along_last_dim"},
    "materialize": {"_deltakv_materialize_sparse_view_block_kernel"},
}


def make_case(operation, size, module=None, *, strided=False, quest_backend=None):
    """Return the serving wrapper (minmax: internal stage) and its oracle check."""
    module = module or importlib.import_module("sparseengine.kernels.triton." + MODULES[operation])
    device = "cuda"
    torch.manual_seed(123 + size)
    if operation == "omnikv":
        heads, sink, recent = 3, 5, 13
        raw = torch.randn(3, heads, size * (2 if strided else 1), device=device)
        if strided:
            raw = raw[..., ::2]
        lengths = torch.tensor([0, size // 2, size], device=device, dtype=torch.int32)
        partial = torch.empty((3, heads, triton.cdiv(size-sink, 1024), 2), device=device)
        stats = torch.empty((3, heads, 2), device=device)
        output = torch.empty((3, size), device=device)
        expected = torch.full_like(output, -1e20)
        for row, length in enumerate([0, size // 2, size]):
            end = max(sink, length-recent)
            if end > sink:
                expected[row, sink:end] = (raw[row, :, sink:end].double() / 16).softmax(-1).amax(0).float()
            raw[row, :, end:] = float("nan")
        def run():
            module.launch_omnikv_decode_scores(raw, lengths, partial, stats, output,
                sink=sink, recent=recent, scale=1/16, min_score=-1e20)
            return output
        def check(actual):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-8)
    elif operation == "quest":
        query = torch.randn(2, 6, 64, device=device, dtype=torch.bfloat16)
        high = torch.randn(1024, 2, 64, device=device, dtype=query.dtype).abs()
        low = -torch.randn_like(high).abs()
        slots = torch.randint(0, 1024, (2, size), device=device, dtype=torch.int32)
        slots[:, -1] = -1
        rows = []
        for row in range(2):
            bounds = []
            for head in range(6):
                q = query[row, head].float()
                ids = slots[row].long().clamp_min(0)
                pos = (q.clamp_min(0)[None, :] * high[ids, head//3].float()).sum(-1).to(query.dtype)
                neg = (q.clamp_max(0)[None, :] * low[ids, head//3].float()).sum(-1).to(query.dtype)
                bounds.append(pos + neg)
            rows.append(torch.stack(bounds).amax(0))
        expected = torch.stack(rows)
        def run():
            if quest_backend == "tensorcore":
                return (module.score_quest_pages_tensorcore(query, high, low, slots),)
            return (module.score_quest_pages_tensorcore(query, high, low, slots),
                    module.score_quest_pages_vector(query, high, low, slots))
        def check(actual):
            for score in actual:
                torch.testing.assert_close(score, expected, rtol=0, atol=0)
    elif operation == "gather":
        src = torch.randn(113, 80, device=device, dtype=torch.float16)
        indices = torch.randint(0, 113, (2, size, 3), device=device, dtype=torch.int32)
        expected = src[indices.long()].float().mean(2).to(src.dtype)
        def run():
            return module.batch_gather_mean(src, indices)
        def check(actual):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif operation == "minmax":
        # Vary both token/batch count and group count without changing group size.
        groups = 2 + size % 3
        raw = torch.randn(size, groups, 32, device=device, dtype=torch.float16)
        lo = torch.empty((size, groups), device=device, dtype=raw.dtype)
        hi = torch.empty_like(lo)
        expected = raw.amin(-1), raw.amax(-1)
        def run():
            module._minmax_along_last_dim[(triton.cdiv(size*groups, 128),)](
                raw, lo, hi, raw.numel(), size, groups, 32, 128, num_warps=8)
            return lo, hi
        def check(actual):
            for value, ref in zip(actual, expected):
                torch.testing.assert_close(value, ref, rtol=0, atol=0)
    else:
        assert operation == "materialize"
        batch, heads, dim, slots_count = 3, 2, 64, 257
        key = torch.randn(slots_count, heads, dim, device=device, dtype=torch.float16)
        value = torch.randn_like(key)
        backing = torch.randint(0, slots_count, (batch, size * (2 if strided else 1)), device=device, dtype=torch.int32)
        slots = backing[:, ::2] if strided else backing
        lengths = torch.tensor([size, size//2, 0], device=device, dtype=torch.int32)
        positions = torch.randperm(slots_count, device=device).int()
        mask = (torch.arange(slots_count, device=device) % 3 == 0)
        angle = torch.randn(slots_count, dim//2, device=device)
        rope = torch.cat([angle.cos(), angle.sin()], -1).half()
        norm = torch.randn(dim, device=device, dtype=torch.float16)
        out_k = torch.empty((batch*size, heads, dim), device=device, dtype=key.dtype)
        out_v = torch.empty_like(out_k)
        ids = slots.flatten().long()
        raw = key[ids].float()
        normalized = raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + 1e-6) * norm.float()
        a, b = normalized.chunk(2, -1)
        cos, sin = rope[positions[ids].long()].float().chunk(2, -1)
        rotated = torch.cat([a*cos[:, None] - b*sin[:, None], b*cos[:, None] + a*sin[:, None]], -1)
        expected_k = torch.where(mask[ids, None, None], raw, rotated).half()
        def run():
            module.deltakv_materialize_sparse_view(slots, lengths, positions, mask,
                key, value, out_k, out_v, rope, k_norm_weight=norm, block_tokens=16)
            return out_k, out_v
        def check(actual):
            torch.testing.assert_close(actual[0], expected_k, rtol=.002, atol=.002)
            torch.testing.assert_close(actual[1], value[ids], rtol=0, atol=0)
    return run, check


@pytest.mark.parametrize("operation", list(MODULES))
def test_length_changes_reuse_compiled_kernels_and_match_oracle(operation, monkeypatch):
    # Cross scalar alignment classes, odd tails, runtime strides and split counts.
    # OmniKV keeps the genuine power-of-two merge tile fixed for this sweep.
    sizes = [4112, 4113, 4128, 4161, 5137, 6143] if operation == "omnikv" else [32, 1, 17, 31, 33, 65]
    run, check = make_case(operation, sizes[0])
    check(run())
    events = []
    previous = triton.knobs.compilation.listener
    def listener(**kwargs):
        if kwargs["src"].name in TARGETS[operation]:
            events.append(kwargs["src"].name)
        if previous is not None:
            previous(**kwargs)
    monkeypatch.setattr(triton.knobs.compilation, "listener", listener)
    for index, size in enumerate(sizes[1:]):
        run, check = make_case(operation, size, strided=bool(index % 2))
        check(run())
    assert events == [], f"New length/stride caused compilation or cache lookup: {events}"
    # Allocating wrappers must also capture and return usable graph-owned outputs.
    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    graph.replay()
    check(actual)
