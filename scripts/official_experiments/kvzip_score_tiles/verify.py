"""Run the canonical probe with selection/output recording; not a timing run."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--verification-output", type=Path, required=True)
    parser.add_argument("--score-output-dir", type=Path)
    args, remaining = parser.parse_known_args()
    if args.verification_output.exists():
        raise FileExistsError(args.verification_output)
    if args.baseline_source:
        from bench import load_baseline
        from sparseengine.kernels import triton as kernels
        baseline = load_baseline(args.baseline_source)
        sys.modules["sparseengine.kernels.triton.prefill_score"] = baseline
        kernels.prefill_score = baseline

    from benchmark.efficiency import bench_probe
    from sparseengine import LLM
    from sparseengine.engine.cache_manager.methods.kvzip import KVzipCacheManager
    if args.baseline_source:
        assert KVzipCacheManager.collect_prefill_attention_score.__wrapped__.__globals__["prefill_score_fwd"] is baseline.prefill_score_fwd

    selections, outputs = [], []
    original_compact = KVzipCacheManager.compact_reconstruction
    original_step = LLM.step
    if args.score_output_dir:
        import torch
        from sparseengine.engine.sparse_methods import kvzip
        args.score_output_dir.mkdir(parents=True, exist_ok=False)
        original_select = kvzip.select_kvzip_layers
        selection_index = 0

        def select(scores, budget, prefix_length, **kwargs):
            nonlocal selection_index
            keep = original_select(scores, budget, prefix_length, **kwargs)
            torch.save({"scores": scores.cpu(), "keep": [x.cpu() for x in keep],
                        "budget": budget, "prefix_length": prefix_length},
                       args.score_output_dir / f"{selection_index:04d}.pt")
            selection_index += 1
            return keep

        kvzip.select_kvzip_layers = select

    def compact(self, seq, keep):
        selections.append({"seq_id": seq.seq_id, "context_length": seq.context_length,
                           "layer_tokens": [x.numel() for x in keep],
                           "layer_indices_sha256": [hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
                                                    for x in keep]})
        return original_compact(self, seq, keep)

    def step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        for seq_id, tokens, _logprobs, _top_logprobs in result[0]:
            outputs.append({"seq_id": int(seq_id), "token_ids": list(tokens)})
        return result

    KVzipCacheManager.compact_reconstruction = compact
    LLM.step = step
    sys.argv = [sys.argv[0], *remaining]
    bench_probe.main()
    args.verification_output.write_text(json.dumps({"selections": selections, "outputs": outputs}, indent=2) + "\n")


if __name__ == "__main__":
    main()
