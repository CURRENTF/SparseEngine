"""Reproduce H100 Qwen3 FP8 MoE and QuEST scoring callable timings.

Expose one idle device and activate the environment before invoking this file.
"""
import argparse
import functools
import importlib.metadata
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import triton
from sparseengine.kernels.triton.quest_decode_view import _score_quest_pages_kernel

def current_score_with_warps(
    query: torch.Tensor,
    page_max: torch.Tensor,
    page_min: torch.Tensor,
    row_page_slots: torch.Tensor,
    *,
    num_warps: int,
) -> torch.Tensor:
    batch_size, num_query_heads, head_dim = map(int, query.shape)
    _, num_metadata_heads, _ = map(int, page_max.shape)
    num_pages = int(row_page_slots.shape[1])
    output = torch.empty(
        (batch_size, num_pages),
        dtype=query.dtype,
        device=query.device,
    )
    _score_quest_pages_kernel[(batch_size, num_pages)](
        query,
        page_max,
        page_min,
        row_page_slots,
        output,
        query.stride(0),
        query.stride(1),
        page_max.stride(0),
        page_max.stride(1),
        row_page_slots.stride(0),
        output.stride(0),
        NUM_QUERY_HEADS=num_query_heads,
        NUM_METADATA_HEADS=num_metadata_heads,
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=num_warps,
        num_stages=1,
    )
    return output

def torch_oracle(
    query: torch.Tensor,
    page_max: torch.Tensor,
    page_min: torch.Tensor,
    row_page_slots: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_query_heads, head_dim = map(int, query.shape)
    _, num_metadata_heads, _ = map(int, page_max.shape)
    num_pages = int(row_page_slots.shape[1])
    selected_max = page_max.index_select(
        0,
        row_page_slots.to(torch.long).reshape(-1),
    ).view(batch_size, num_pages, num_metadata_heads, head_dim)
    selected_min = page_min.index_select(
        0,
        row_page_slots.to(torch.long).reshape(-1),
    ).view(batch_size, num_pages, num_metadata_heads, head_dim)
    group_size = num_query_heads // num_metadata_heads
    grouped_query = query.view(
        batch_size,
        num_metadata_heads,
        group_size,
        head_dim,
    )
    query_positive = grouped_query.clamp_min(0).reshape(
        batch_size * num_metadata_heads,
        group_size,
        head_dim,
    )
    query_negative = grouped_query.clamp_max(0).reshape(
        batch_size * num_metadata_heads,
        group_size,
        head_dim,
    )
    max_transposed = selected_max.permute(0, 2, 3, 1).reshape(
        batch_size * num_metadata_heads,
        head_dim,
        num_pages,
    )
    min_transposed = selected_min.permute(0, 2, 3, 1).reshape(
        batch_size * num_metadata_heads,
        head_dim,
        num_pages,
    )
    expected = torch.bmm(query_positive, max_transposed)
    expected += torch.bmm(query_negative, min_transposed)
    return expected.view(
        batch_size,
        num_metadata_heads,
        group_size,
        num_pages,
    ).amax(dim=2).amax(dim=1)

def measure(variants, root, tag):
    graphs = {}
    for name, fn in variants.items():
        for _ in range(5): fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = [fn() for _ in range(32)]
        graphs[name] = (graph, outputs)
    samples = {name: [] for name in variants}
    with (root / 'raw_samples.jsonl').open('a') as f:
        for rep in range(20):
            names = list(variants)
            if rep % 2: names.reverse()
            for name in names:
                g,_ = graphs[name]
                g.replay()
                start,end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(8): g.replay()
                end.record(); end.synchronize()
                us = start.elapsed_time(end)*1000/256
                samples[name].append(us)
                f.write(json.dumps(dict(tag=tag,variant=name,rep=rep,us=us))+'\n')
    result = {name: statistics.median(v) for name,v in samples.items()}
    print(tag, result, flush=True)
    return result

def moe(root):
    from sparseengine.operators.moe import FlashInferCutlassFp8MoeProvider, TritonUpGateFp8MoeProvider, MoeOpSpec
    from sparseengine.quantization.fp8 import fp8_blockwise_linear_reference
    e,h,i,k=128,2048,768,8
    w13=torch.randn(e,2*i,h,device='cuda').to(torch.float8_e4m3fn)
    w2=torch.randn(e,h,i,device='cuda').to(torch.float8_e4m3fn)
    s13=torch.rand(e,2*i//128,h//128,device='cuda')*.02+.01
    s2=torch.rand(e,h//128,i//128,device='cuda')*.02+.01
    results={}
    for tokens in (1,2,4,8,16):
        x=torch.randn(tokens,h,device='cuda',dtype=torch.bfloat16)
        ids=torch.stack([torch.randperm(e,device='cuda')[:k] for _ in range(tokens)]).int()
        weights=torch.softmax(torch.randn(tokens,k,device='cuda'),dim=-1)
        spec=MoeOpSpec(e,e,h,i,k,torch.bfloat16,torch.float8_e4m3fn,(128,128),1,True,max_num_tokens=tokens,scale_dtype=torch.float32)
        variants={}
        for provider in (FlashInferCutlassFp8MoeProvider(), TritonUpGateFp8MoeProvider()):
            provider.prepare(spec,device=x.device,tp_rank=0,ep_rank=0,max_num_tokens=tokens)
            variants[provider.name]=functools.partial(provider.run,spec,x,ids,weights,w13,w2,s13,s2,local_expert_start=0,tp_rank=0,ep_rank=0)
        oracle=torch.zeros_like(x[:1],dtype=torch.float32)
        for route in range(k):
            eid=int(ids[0,route])
            up,gate=fp8_blockwise_linear_reference(x[:1],w13[eid],s13[eid]).chunk(2,dim=-1)
            act=(torch.nn.functional.silu(gate.float())*up.float()).bfloat16()
            y=fp8_blockwise_linear_reference(act,w2[eid],s2[eid])
            oracle.add_(y.float()*weights[0,route])
        correctness={}
        for name,fn in variants.items():
            y=fn()
            err=(y[:1].float()-oracle).abs().mean()/oracle.abs().mean()
            cosine=torch.nn.functional.cosine_similarity(y[:1].float(),oracle).min()
            correctness[name]=dict(mean_relative=float(err),cosine=float(cosine))
            assert torch.isfinite(y).all() and err<.05 and cosine>.995, correctness
        results[str(tokens)]=dict(correctness=correctness,median_us=measure(variants,root,f'moe{tokens}'))
    return results

def score(root):
    from sparseengine.operators.quest_scoring import QuestPageScoreSpec, resolve_quest_page_score_provider
    provider = resolve_quest_page_score_provider(
        QuestPageScoreSpec(torch.bfloat16, 32, 4, 128, True), device_index=0,
    )
    results = {}
    for n in (2060, 8320):
        q = torch.randn(1, 32, 128, device="cuda", dtype=torch.bfloat16)
        high = torch.randn(n, 4, 128, device="cuda", dtype=torch.bfloat16).abs()
        low = -torch.randn_like(high).abs()
        pages = torch.randperm(n, device="cuda", dtype=torch.int32)[None]
        baseline = functools.partial(current_score_with_warps, q, high, low, pages, num_warps=1)
        candidate = functools.partial(provider.score, q, high, low, pages)
        expected = baseline()
        torch.testing.assert_close(expected, torch_oracle(q, high, low, pages), rtol=0.008, atol=0)
        torch.testing.assert_close(candidate(), torch_oracle(q, high, low, pages), rtol=0.008, atol=0)
        results[str(n)] = measure({"baseline": baseline, "candidate": candidate}, root, str(n))
    return results


def score_general(root, shapes, dtype, working_set_copies=1, metadata_sharing="disjoint"):
    """Compare complete score callables for caller-specified tensor contracts."""
    from sparseengine.kernels.triton.quest_decode_view import score_quest_pages
    from sparseengine.kernels.triton.quest_page_score import (
        score_quest_pages_tensorcore,
        score_quest_pages_vector,
    )
    from sparseengine.operators.quest_scoring import (
        QuestPageScoreSpec,
        TensorCoreQuestPageScoreProvider,
        resolve_quest_page_score_provider,
    )

    from sparseengine import platforms

    caps = platforms.current_platform.get_device_caps(0)
    results = {}
    for batch, pages, heads, kv_heads, dim in shapes:
        tag = f"b{batch}_n{pages}_h{heads}_kv{kv_heads}_d{dim}"
        q = torch.randn(batch, heads, dim, device="cuda", dtype=dtype)
        physical_pages = pages if metadata_sharing == "shared" else batch * pages
        high = torch.randn(physical_pages, kv_heads, dim, device="cuda", dtype=dtype).abs()
        low = -torch.randn_like(high).abs()
        slots = torch.stack([
            torch.randperm(pages, device="cuda", dtype=torch.int32)
            + (0 if metadata_sharing == "shared" else row * pages)
            for row in range(batch)
        ])
        spec = QuestPageScoreSpec(dtype, heads, kv_heads, dim, True)
        provider = resolve_quest_page_score_provider(spec, device_index=0)
        functions = {
            "scalar": score_quest_pages,
            "vector": functools.partial(score_quest_pages_vector, block_pages=2, num_warps=1),
            "selected": provider.score,
        }
        tensorcore_support = TensorCoreQuestPageScoreProvider.supports(spec, caps)
        if tensorcore_support.supported:
            functions["tensorcore"] = functools.partial(
                score_quest_pages_tensorcore,
                block_dim=64 if heads == kv_heads and dim >= 256 else 128,
            )
        # Different addresses with identical values isolate metadata working-set
        # effects without changing the numerical oracle. A captured32-call graph
        # visits every copy; rotation adds no GPU eviction kernel to the timing.
        high_copies = high.unsqueeze(0).expand(working_set_copies, -1, -1, -1).contiguous()
        low_copies = low.unsqueeze(0).expand(working_set_copies, -1, -1, -1).contiguous()
        metadata = list(zip(high_copies.unbind(0), low_copies.unbind(0)))

        def rotating_call(fn):
            index = 0
            def call():
                nonlocal index
                hi, lo = metadata[index % working_set_copies]
                index += 1
                return fn(q, hi, lo, slots)
            return call

        variants = {name: rotating_call(fn) for name, fn in functions.items()}
        # Accumulate in FP64, but preserve the two separately rounded bounds.
        group = heads // kv_heads
        expected_rows = []
        for row in range(batch):
            bounds = []
            for head in range(kv_heads):
                query = q[row, head * group:(head + 1) * group].double()
                index = slots[row].long()
                pos = (query.clamp_min(0) @ high[index, head].double().T).to(dtype)
                neg = (query.clamp_max(0) @ low[index, head].double().T).to(dtype)
                bounds.append((pos + neg).amax(0))
            expected_rows.append(torch.stack(bounds).amax(0))
        expected = torch.stack(expected_rows)
        correctness = {}
        for name, fn in variants.items():
            actual = fn()
            rtol, atol = (2e-5, 1e-4) if dtype == torch.float32 else (0.008, 0.02)
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            k = min(127, pages)
            selected = actual.argsort(dim=1, descending=True, stable=True)[:, :k]
            reference = expected.argsort(dim=1, descending=True, stable=True)[:, :k]
            overlap = (selected[:, :, None] == reference[:, None, :]).any(-1)
            correctness[name] = {
                "max_abs_error": float((actual.float() - expected.float()).abs().max()),
                "topk_overlap": float(overlap.float().mean()),
            }
        results[tag] = {
            "shape": [batch, pages, heads, kv_heads, dim],
            "dtype": str(dtype),
            "selected_provider": provider.name,
            "working_set_copies": working_set_copies,
            "metadata_sharing": metadata_sharing,
            "metadata_working_set_bytes": 2 * high.numel() * high.element_size() * working_set_copies,
            "candidate_rejections": {} if tensorcore_support.supported else {
                "tensorcore": tensorcore_support.reason,
            },
            "correctness": correctness,
            "median_us": measure(variants, root, tag),
        }
        stats = getattr(provider, "runtime_kernel_stats", None)
        if stats is not None:
            results[tag]["runtime_kernel_stats"] = stats()
        (root / "summary.json").write_text(json.dumps(results, indent=2))
    return results


def score_shape(value):
    shape = tuple(int(part) for part in value.split(","))
    if len(shape) != 5 or min(shape) <= 0 or shape[2] % shape[3]:
        raise argparse.ArgumentTypeError("Expected positive B,N,H,KV,D with H divisible by KV")
    return shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("score", "score-general", "moe"), required=True)
    parser.add_argument("--shape", type=score_shape, action="append", help="B,N,H,KV,D; repeat for multiple cases")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--working-set-copies", type=int, choices=(1, 4, 16, 32), default=1,
                        help="Rotate identical metadata at distinct addresses within each32-call graph")
    parser.add_argument("--metadata-sharing", choices=("disjoint", "shared"), default="disjoint",
                        help="Use separate physical pages per request, or intentionally share all pages")
    args = parser.parse_args()
    if args.mode != "score-general" and args.working_set_copies != 1:
        parser.error("--working-set-copies applies only to score-general")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    repo = Path(__file__).resolve().parents[3]
    manifest = dict(
        command=sys.argv, interpreter=sys.executable, seed=42,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True),
        torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                  if any(x in d.metadata["Name"].lower() for x in ("flashinfer", "sgl"))},
        timing="32 calls per CUDA Graph; 8 replays/sample; 20 interleaved samples",
        working_set_copies=args.working_set_copies,
        scope="synthetic callable microbenchmark; serving comparison uses the canonical efficiency probe",
        source_tree=str(repo),
        compute_capability=torch.cuda.get_device_capability(),
        multiprocessors=torch.cuda.get_device_properties(0).multi_processor_count,
    )
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    try:
        if args.mode == "score-general":
            shapes = args.shape or [(1, 8320, 32, 4, 128), (1, 8320, 32, 8, 128), (1, 8320, 1, 1, 576)]
            result = score_general(args.output, shapes, getattr(torch, args.dtype), args.working_set_copies, args.metadata_sharing)
        else:
            result = {"score": score, "moe": moe}[args.mode](args.output)
    except Exception as error:
        (args.output / "run_status.json").write_text(json.dumps({"status": "failed", "error": repr(error)}))
        raise
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    (args.output / "run_status.json").write_text(json.dumps({"status": "success"}))


if __name__ == "__main__":
    main()
