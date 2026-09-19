"""Runtime lengths must reuse the MLA executable and preserve captured storage."""
import pytest
import torch

from sparseengine.kernels.tilelang.mla.runtime import TileMlaDecodeKernel, TileMlaLaunchConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("mode,heads,split", [("per_head", 10, 16), ("per_head", 5, 1), ("partial", 20, 4), ("direct", 10, 4), ("none", 10, 4)])
def test_eager_shape_changes_reuse_kernel_and_old_graph(mode, heads, split, monkeypatch, request):
    # Existing graph tests change tensor contents but keep every tensor shape fixed.
    # This reproduces eager score slicing and changing physical slot-table shapes.
    from tilelang.jit.kernel import JITKernel
    runner = TileMlaDecodeKernel(device="cuda:0", softmax_scale=1/16, valid_heads=heads,
        fixed_config=TileMlaLaunchConfig(split, block_h=16, score_mode="direct" if mode == "none" else mode))
    torch.manual_seed(31)
    q = torch.randn(4, heads, 576, device="cuda", dtype=torch.bfloat16)
    graphs = []
    original = JITKernel._compile_and_create_adapter
    compiles = []
    def compile_adapter(self, *args, **kwargs):
        compiles.append(1)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(JITKernel, "_compile_and_create_adapter", compile_adapter)
    from sparseengine.utils.compilation_guard import RuntimeCompilationGuard
    guard = RuntimeCompilationGuard(limit=0, rank=0)
    request.addfinalizer(guard.close)
    initial_compiles = None
    for index, capacity in enumerate([65, 129, 257, 513, 33]):
        cache = torch.randn(capacity+17, 1, 512, device="cuda", dtype=torch.bfloat16)
        rope = torch.randn(capacity+17, 1, 64, device="cuda", dtype=torch.bfloat16)
        slots = torch.randint(0, capacity+17, (3+index, capacity), device="cuda", dtype=torch.int32)
        req = torch.tensor([index+1, 0, index+2, -1], device="cuda", dtype=torch.int32)
        lens = torch.tensor([capacity-3, capacity//3, 1, 0], device="cuda", dtype=torch.int32)
        output = torch.empty(4, heads, 512, device="cuda", dtype=torch.bfloat16)
        shape = (4, heads, capacity*2+7) if mode == "per_head" else (4, capacity)
        backing = torch.full(shape, 123., device="cuda")
        score = backing[..., :2*capacity:2] if mode == "per_head" else backing
        references = []
        for row, (request_id, length) in enumerate(zip(req.tolist(), lens.tolist())):
            if length == 0:
                references.append(None)
                continue
            ids = slots[request_id, :length].long()
            raw = q[row,:,:512].float() @ cache[ids,0].float().T + q[row,:,512:].float() @ rope[ids,0].float().T
            expected = (raw / 16).softmax(-1) @ cache[ids,0].float()
            references.append((length, raw, expected))
        def run(cache=cache, rope=rope, slots=slots, req=req, lens=lens, output=output, score=score, capacity=capacity):
            score.fill_(-1e20)
            runner(q[:,:,:512],q[:,:,512:],cache,rope,slots,req,lens,output,attn_score=None if mode == "none" else score,max_context_len=capacity)
        def check(output=output, score=score, backing=backing, references=references, capacity=capacity):
            for row, reference in enumerate(references):
                if reference is None:
                    torch.testing.assert_close(output[row], torch.zeros_like(output[row]), rtol=0, atol=0)
                    if mode != "none":
                        assert (score[row] == -1e20).all()
                    continue
                length, raw, expected = reference
                torch.testing.assert_close(output[row], expected.bfloat16(), rtol=.02, atol=.02)
                if mode != "none":
                    ref = raw if mode == "per_head" else raw.amax(0)
                    torch.testing.assert_close(score[row,...,:length], ref, rtol=.02, atol=.02)
                    assert (score[row,...,length:] == -1e20).all()
            if mode == "per_head":
                assert (backing[...,1:2*capacity:2] == 123.).all()
        run();check()
        if initial_compiles is None:
            initial_compiles=len(compiles)
            guard.arm()
        else:
            assert len(compiles)==initial_compiles, "new length triggered TileLang compilation"
        assert runner.runtime_metadata()["compiled_variant_count"]==1
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):run()
        graphs.append((graph,run,check))
    # Grow and shrink again after capture: old graph pointers must still be valid.
    for graph,run,check in graphs:
        graph.replay()
        check()

    assert guard.count == 0
