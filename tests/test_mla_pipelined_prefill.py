"""Protect buffer reuse, ragged masking, and mutable metadata on the GPU."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("backend", ["pipelined", "hopper"])
@pytest.mark.parametrize("split_kv", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_new_lengths_reuse_compiled_partials(monkeypatch, record_property, backend, split_kv, causal):
    # Regression: MiniSWE suffix lengths compiled thousands of MLA variants.
    # Existing fixed-shape graph tests cannot detect a new binary per length.
    from importlib import import_module

    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("pipelined MLA requires Ampere or newer")
    if backend == "hopper" and torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("TMA/WGMMA MLA requires Hopper")
    module = import_module(f"sparseengine.kernels.triton.mla.prefill_{backend}")
    binaries = {"attention": set(), "merge": set()}

    def track(kernel, name):
        run = kernel.run

        def launch(*args, **kwargs):
            compiled = run(*args, **kwargs)
            binaries[name].add(compiled.hash)
            return compiled

        monkeypatch.setattr(kernel, "run", launch)

    track(module._attention, "attention")
    track(module._merge_splits, "merge")
    torch.manual_seed(719)
    # Hold the structural split family fixed with the real split selector;
    # vary exact lengths, alignment classes, grid size and transposed-Q stride.
    for qn, kn in ((1, 4101), (16, 4223), (127, 4288), (129, 4377), (257, 4489), (385, 4601)):
        q = (torch.randn(2, qn, 256, device="cuda", dtype=torch.bfloat16) * 0.2).transpose(0, 1)
        k = torch.randn(kn, 2, 256, device="cuda", dtype=torch.bfloat16) * 0.2
        v = torch.randn(kn, 2, 448, device="cuda", dtype=torch.bfloat16)[..., 192:]
        cq = torch.tensor([0, qn], device="cuda", dtype=torch.int32)
        ck = torch.tensor([0, kn], device="cuda", dtype=torch.int32)
        out, lse = module.attention_partial(
            q, k, v, cq, ck, qn, kn, scale=0.0625, causal=causal,
            split_kv=split_kv, sm_count=1024,
        )
        z = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * 0.0625
        if causal:
            valid = torch.arange(kn, device="cuda")[None] <= torch.arange(qn, device="cuda")[:, None] + kn - qn
            z.masked_fill_(~valid[None], -torch.inf)
        expected = torch.einsum("hqk,khd->qhd", z.softmax(-1), v.float())
        torch.testing.assert_close(out.float(), expected, atol=0.015, rtol=0.03)
        torch.testing.assert_close(lse, z.logsumexp(-1), atol=0.003, rtol=0.001)
        assert len(binaries["attention"]) == 1, "new request length compiled another attention binary"
        assert len(binaries["merge"]) == int(split_kv), "new request length compiled another merge binary"
    record_property("compiled_binaries", {name: sorted(values) for name, values in binaries.items()})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("backend", ["pipelined", "hopper"])
@pytest.mark.parametrize("split_kv", [False, True])
@pytest.mark.parametrize(
    "queries,keys,causal,magnitude",
    [
        ((1025, 7), (1025, 193), True, 0.2),
        ((257, 17), (63, 0), True, 2.0),
        ((35, 5), (193, 0), False, 2.0),
        ((127, 129), (65, 257), False, 0.2),
        ((35, 5), (0, 0), False, 2.0),
    ],
)
def test_pipelined_partial_ragged_graph_and_empty_rows(
    queries, keys, causal, magnitude, split_kv, backend
):
    # Catch double-buffer wraparound, sharp logits, and fully masked rows.
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("pipelined MLA requires Ampere or newer")
    if backend == "hopper":
        if torch.cuda.get_device_capability() != (9, 0):
            pytest.skip("TMA/WGMMA MLA requires Hopper")
        from sparseengine.kernels.triton.mla.prefill_hopper import attention_partial
    else:
        from sparseengine.kernels.triton.mla.prefill_pipelined import attention_partial

    torch.manual_seed(918)
    q = (
        torch.randn(5, sum(queries), 256, device="cuda", dtype=torch.bfloat16)
        * magnitude
    ).transpose(0, 1)
    k = torch.randn(sum(keys), 5, 256, device="cuda", dtype=torch.bfloat16) * magnitude
    v = torch.randn(sum(keys), 5, 448, device="cuda", dtype=torch.bfloat16)[..., 192:]
    cq = torch.tensor([0, queries[0], sum(queries)], device="cuda", dtype=torch.int32)
    ck = torch.tensor([0, keys[0], sum(keys)], device="cuda", dtype=torch.int32)

    def run():
        return attention_partial(
            q, k, v, cq, ck, max(queries), max(keys),
            scale=0.0625, causal=causal, split_kv=split_kv,
        )

    def check(output, lse):
        qa = ka = 0
        for qn, kn in zip(queries, keys):
            z = (
                torch.einsum(
                    "qhd,khd->hqk", q[qa : qa + qn].float(), k[ka : ka + kn].float()
                )
                * 0.0625
            )
            if causal:
                valid = (
                    torch.arange(kn, device="cuda")[None]
                    <= torch.arange(qn, device="cuda")[:, None] + kn - qn
                )
                z.masked_fill_(~valid[None], -torch.inf)
            expected = torch.einsum(
                "hqk,khd->qhd", z.softmax(-1).nan_to_num(0), v[ka : ka + kn].float()
            )
            torch.testing.assert_close(
                output[qa : qa + qn].float(), expected, atol=0.015, rtol=0.03
            )
            torch.testing.assert_close(
                lse[:, qa : qa + qn], z.logsumexp(-1), atol=0.003, rtol=0.001
            )
            qa += qn
            ka += kn

    check(*run())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        (output, lse) = run()
    for factor in (0.5, 1.5):
        q.mul_(factor)
        graph.replay()
        check(output, lse)
    # Same allocation and graph, different ragged boundaries and causal offsets.
    queries = tuple(reversed(queries))
    cq[1] = queries[0]
    graph.replay()
    check(output, lse)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tp_size", [1, 4])
def test_bound_provider_executes_partial_and_preserves_failure(monkeypatch, tp_size):
    # A previously unprofiled short ragged batch must use the prepared provider,
    # expose its binding report, and propagate launch failures without rerouting.
    from sparseengine.operators.mla_attention import MlaAttentionOpSpec, MlaTritonProvider

    spec = MlaAttentionOpSpec(
        num_q_heads=20, kv_lora_rank=512, rope_dim=64,
        qk_head_dim=256, value_head_dim=256,
        activation_dtype=torch.bfloat16, cache_dtype=torch.bfloat16,
        tp_size=tp_size, cuda_graph=False,
    )
    provider = MlaTritonProvider(op_spec=spec, device="cuda:0", max_batch_size=2)
    q = torch.zeros(26, spec.local_q_heads, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(514, spec.local_q_heads, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.full_like(k, 0.25)
    cq = torch.tensor([0, 7, 26], device="cuda", dtype=torch.int32)
    ck = torch.tensor([0, 257, 514], device="cuda", dtype=torch.int32)
    allocated = []
    empty = torch.empty

    def track_empty(*args, **kwargs):
        tensor = empty(*args, **kwargs)
        if tensor.dtype == torch.float32 and tensor.ndim > 2:
            allocated.append(tensor.numel() * tensor.element_size())
        return tensor

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", track_empty)
        output, lse = provider.run_prefill_chunk(q, k, v, cq, ck, 19, 257, causal=False)
    torch.testing.assert_close(output, torch.full_like(q, 0.25))
    torch.testing.assert_close(lse, torch.full_like(lse, 257).log())
    assert provider.prefill_workspace_bytes(tokens=26, batch=2, max_q=19, max_k=257) == sum(allocated)
    assert provider.binding_metadata()["prefill"]["selected_provider"] == provider._prefill.name
    assert provider.runtime_kernel_stats()["kernel_paths"][provider._prefill.kernel_path]["eager_dispatches"] == 1

    def failed(*args, **kwargs):
        raise RuntimeError("regression: kernel launch failed")

    monkeypatch.setattr(provider._prefill, "kernel", failed)
    with pytest.raises(RuntimeError, match="kernel launch failed"):
        provider.run_prefill_chunk(q, k, v, cq, ck, 19, 257, causal=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("backend", ["pipelined", "hopper", "old"])
def test_partial_large_storage_offsets(backend):
    # Batch-packed strided V exceeds signed 32-bit element offsets at B=4,
    # K=64K, H=20. Three widely separated rows reproduce it with tiny math.
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("pipelined MLA requires Ampere or newer")
    if backend == "pipelined":
        from sparseengine.kernels.triton.mla.prefill_pipelined import attention_partial
    elif backend == "hopper":
        if torch.cuda.get_device_capability() != (9, 0):
            pytest.skip("TMA/WGMMA MLA requires Hopper")
        from sparseengine.kernels.triton.mla.prefill_hopper import attention_partial
    else:
        from sparseengine.kernels.triton.mla.prefill import attention_partial
    stride = 2**30 + 256
    if torch.cuda.mem_get_info()[0] < 5 * 1024**3:
        pytest.skip("large-address regression requires 5 GiB free")
    kv = torch.empty_strided((3, 1, 256), (stride, 256, 1), device="cuda", dtype=torch.bfloat16)
    kv[0].fill_(0.25)
    kv[1].fill_(0.5)
    kv[2].fill_(1.0)
    q = torch.zeros((2, 1, 256), device="cuda", dtype=torch.bfloat16)
    cq = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    ck = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32)
    out, lse = attention_partial(q, kv, kv, cq, ck, 1, 2, scale=0.0625, causal=False)
    torch.cuda.synchronize()
    expected = torch.tensor([0.25, 0.75], device="cuda")[:, None, None].expand_as(q)
    torch.testing.assert_close(out.float(), expected, atol=0, rtol=0)
    torch.testing.assert_close(lse, torch.tensor([[1.0, 2.0]], device="cuda").log(), atol=1e-6, rtol=0)
