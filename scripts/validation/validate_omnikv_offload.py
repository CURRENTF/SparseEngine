"""Reproducible model parity and shared-prefix lifecycle validation.

Pass the same model/topology/layers to baseline and offload, then --reference
with the baseline JSON. Performance belongs in efficiency/bench_probe.py.
"""

import argparse
import json
import subprocess
from pathlib import Path

from sparseengine import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--full-layers", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--cache-tokens", type=int)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--pressure", action="store_true")
    parser.add_argument("--no-prefix", action="store_true")
    parser.add_argument("--cancel", action="store_true")
    parser.add_argument("--prefix-offload", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.no_prefix and (args.prefix_offload or args.pressure):
        parser.error("--no-prefix cannot be combined with prefix offload or pressure")
    if args.output.exists():
        raise FileExistsError(args.output)
    config = {
        "sparse_method": "omnikv",
        "enable_omnikv_offload": args.offload,
        "omnikv_offload_cache_tokens": args.cache_tokens,
        "full_attention_layers": args.full_layers,
        "tensor_parallel_size": args.tp,
        "expert_parallel_size": args.ep,
        "decode_graph": not args.eager,
        "max_model_len": 4096,
        "max_num_batched_tokens": 2048,
        "engine_prefill_chunk_size": 2048,
        "max_num_seqs_in_batch": 2,
        "max_decoding_seqs": 2,
        "max_num_seqs_in_gpu": 2,
        "sink_keep_tokens": 0,
        "recent_keep_tokens": 32,
        "decode_keep_tokens": 512,
        "enable_prefix_caching": not args.no_prefix,
        "prefix_cache_block_size": 16,
        "prefix_cache_max_blocks": None if args.pressure else 512,
        "enable_prefix_cache_offload": args.prefix_offload,
        "prefix_cache_host_size_gb": (4 if args.pressure else 2)
        if args.prefix_offload
        else None,
    }
    artifact = {
        "status": "running",
        "model": args.model,
        "config": config,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(["git", "diff", "--name-only"], text=True).strip()
        ),
        "extended": args.extended,
        "pressure": args.pressure,
        "cancel": args.cancel,
        "samples": [],
    }
    llm = None
    try:
        llm = LLM(args.model, **config)
        params = SamplingParams(temperature=0, max_tokens=24, ignore_eos=True)
        prefix = list(range(1000, 3560))
        cases = [
            [prefix + [21, 22]],
            [prefix + [31, 32], prefix + [41, 42]],
            [prefix + [21, 22]],
            [[5, 6, 7]],
            [prefix + [51, 52]],
        ]
        if args.extended:
            cases.extend(
                [
                    [prefix[:540]],
                    [prefix[:2000] + [61, 62], prefix + [71, 72]],
                ]
            )
        if args.pressure:
            cases.extend(
                [
                    [list(range(start, start + 3500))]
                    for start in (10000, 20000, 30000, 40000)
                ]
            )
            cases.append([prefix + [51, 52]])
        for index, prompts in enumerate(cases):
            sampling = [params] * len(prompts)
            if args.extended and len(prompts) == 2:
                sampling[0] = SamplingParams(
                    temperature=0, max_tokens=12, ignore_eos=True
                )
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            artifact["samples"].append(
                {
                    "case": index,
                    "status": "success",
                    "outputs": outputs,
                    "input_token_ids": prompts,
                    "max_tokens": [p.max_tokens for p in sampling],
                }
            )
        if args.cancel:
            prompts = [prefix + [81, 82], prefix + [91, 92]]
            request_ids = [llm.add_request(prompt, params) for prompt in prompts]
            for _ in range(64):
                llm.step()
                running = list(llm.scheduler.decoding)
                if len(running) == 2 and all(
                    seq.num_completion_tokens >= 3 for seq in running
                ):
                    break
            else:
                raise AssertionError(
                    "shared-prefix requests never reached concurrent decode"
                )
            cancelled = next(seq for seq in running if seq.seq_id == request_ids[0])
            cancelled_tokens = list(cancelled.completion_token_ids)
            llm.abort_request(request_ids[0])
            completed = {}
            for _ in range(64):
                if llm.is_finished():
                    break
                finished, _ = llm.step()
                completed.update(
                    {seq_id: token_ids for seq_id, token_ids, _, _ in finished}
                )
            assert llm.is_finished()
            assert request_ids[1] in completed and request_ids[0] not in completed
            artifact["samples"].append(
                {
                    "case": "cancel_shared_prefix",
                    "status": "success",
                    "input_token_ids": prompts,
                    "cancelled_partial_token_ids": cancelled_tokens,
                    "outputs": [{"token_ids": completed[request_ids[1]]}],
                }
            )
        artifact["cold_hit_token_match"] = (
            artifact["samples"][0]["outputs"][0]["token_ids"]
            == artifact["samples"][2]["outputs"][0]["token_ids"]
        )
        artifact["state"] = llm.debug_sparse_state_summaries()
        for rank in artifact["state"]:
            if args.pressure and args.offload and args.prefix_offload:
                stats = rank["state"]["cache"]["free_slot_stats"]
                assert stats["prefix_cache_h2d_completed_operations"] > 0, stats
            if not args.eager:
                assert rank["decode_graph"]["recapture_count"] == 0
                assert rank["decode_graph"]["replay_count"] > 0
        if args.reference:
            baseline = json.loads(args.reference.read_text())
            assert baseline["status"] == "success"
            for case, reference in zip(
                artifact["samples"], baseline["samples"], strict=True
            ):
                assert [x["token_ids"] for x in case["outputs"]] == [
                    x["token_ids"] for x in reference["outputs"]
                ], case["case"]
            for actual_rank, reference_rank in zip(
                artifact["state"], baseline["state"], strict=True
            ):
                for layer, state in reference_rank["state"]["layers"].items():
                    indices = state["tensors"].get("active_indices")
                    if indices is not None:
                        assert indices == actual_rank["state"]["layers"][layer][
                            "tensors"
                        ].get("active_indices"), (actual_rank["world_rank"], layer)
        artifact["status"] = "success"
    except Exception as exc:
        artifact["status"] = (
            "metric_failed" if isinstance(exc, AssertionError) else "model_failed"
        )
        artifact["error"] = repr(exc)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2, default=str) + "\n")
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
