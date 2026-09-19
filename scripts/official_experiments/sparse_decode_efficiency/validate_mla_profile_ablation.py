"""GPU oracle for the split-profile ablation at the observed BS5 boundary."""
import json
from pathlib import Path
import sys

import torch

from sparseengine.kernels.tilelang.mla.runtime import TileMlaDecodeKernel, TileMlaLaunchPlan


def main():
    output_path = Path(sys.argv[1])
    torch.manual_seed(42)
    batch, heads, capacity = 5, 10, 133120
    device = "cuda:0"
    q = torch.randn(batch, heads, 512, dtype=torch.bfloat16, device=device)
    qr = torch.randn(batch, heads, 64, dtype=torch.bfloat16, device=device)
    kv = torch.randn(capacity, 1, 512, dtype=torch.bfloat16, device=device)
    kr = torch.randn(capacity, 1, 64, dtype=torch.bfloat16, device=device)
    slots = torch.stack([torch.randperm(capacity, device=device).to(torch.int32)
                         for _ in range(batch)])
    req = torch.arange(batch, dtype=torch.int32, device=device)
    lengths = torch.tensor([capacity, capacity - 1, 8192, 17, 0], dtype=torch.int32, device=device)
    output = torch.empty_like(q)
    score = torch.empty(batch, heads, capacity, dtype=torch.float32, device=device)
    plan = TileMlaLaunchPlan.build(context_capacity=capacity, local_q_heads=heads,
                                  sm_count=torch.cuda.get_device_properties(device).multi_processor_count,
                                  max_batch_size=batch, need_score=True, score_mode="per_head")
    runner = TileMlaDecodeKernel(device=device, softmax_scale=256**-0.5,
                                valid_heads=heads, launch_plan=plan)

    def run():
        score.fill_(-1e20)
        runner(q, qr, kv, kr, slots, req, lengths, output,
               attn_score=score, max_context_len=capacity)

    def check():
        for row, length in enumerate(lengths.tolist()):
            if length:
                ids = slots[row, :length].long()
                keys, rope = kv[ids, 0].float(), kr[ids, 0].float()
                raw = q[row].float() @ keys.T + qr[row].float() @ rope.T
                expected = (raw * (256**-0.5)).softmax(-1) @ keys
                torch.testing.assert_close(output[row], expected.to(torch.bfloat16), rtol=3e-2, atol=3e-2)
                torch.testing.assert_close(score[row, :, :length], raw, rtol=3e-2, atol=3e-2)
            else:
                torch.testing.assert_close(output[row], torch.zeros_like(output[row]))
            assert torch.all(score[row, :, length:] == -1e20)

    run()
    torch.cuda.synchronize()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    check()
    lengths.copy_(torch.tensor([31, 2048, 65536, capacity, 0], dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()
    check()
    output_path.write_text(json.dumps({"status": "success", "oracle": "float32 torch attention and per-head raw scores",
        "plan": plan.metadata(), "graph_replays": 2, "dynamic_lengths": True,
        "rtol": 0.03, "atol": 0.03}, indent=2) + "\n")
    print("PASS: BS5 long/ragged/empty contexts, output and raw scores, graph replay", flush=True)


if __name__ == "__main__":
    main()
