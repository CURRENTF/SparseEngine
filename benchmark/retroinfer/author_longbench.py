"""Evaluate the pinned RetroInfer author GPU-only API on frozen LongBench tokens.

The author repository's LongBench entrypoint runs one prompt at a time and has
no v2 protocol. This adapter only supplies shared token IDs and batch grouping;
model execution, cache management, and attention remain in the author repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_samples(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        samples = [json.loads(line) for line in handle if line.strip()]
    ids = [sample["sample_id"] for sample in samples]
    if not samples or len(ids) != len(set(ids)):
        raise ValueError("Input must contain nonempty, unique sample IDs")
    for sample in samples:
        tokens = sample["prompt_token_ids"]
        if not tokens or not all(isinstance(token, int) for token in tokens):
            raise ValueError(f"Invalid token IDs for {sample['sample_id']}")
        if int(sample["max_new_tokens"]) <= 0:
            raise ValueError(f"Invalid generation budget for {sample['sample_id']}")
    return samples


def batch_limit(length: int, requested: int) -> int:
    # GPU-only author cache allocates a full KV array for the padded batch.
    if length <= 8192:
        capacity = 16
    elif length <= 32768:
        capacity = 8
    elif length <= 65536:
        capacity = 4
    elif length <= 90000:
        capacity = 2
    else:
        capacity = 1
    return min(capacity, requested)


def make_batches(samples: list[dict], max_batch: int) -> list[list[int]]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        length = len(sample["prompt_token_ids"])
        # The pinned author's GPU-only prefill asserts that every request in
        # a batch has exactly the same length; left padding is unsupported.
        groups[(int(sample["max_new_tokens"]), length)].append(index)
    batches = []
    for key in sorted(groups):
        indices = groups[key]
        offset = 0
        while offset < len(indices):
            limit = batch_limit(len(samples[indices[offset]]["prompt_token_ids"]), max_batch)
            batch = indices[offset:offset + limit]
            batches.append(batch)
            offset += len(batch)
    return batches


def read_progress(path: Path, samples: list[dict]) -> dict[int, dict]:
    if not path.exists():
        return {}
    results = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            index = int(row["index"])
            if index in results or not 0 <= index < len(samples):
                raise ValueError("Progress contains duplicate or invalid index")
            if row["sample_id"] != samples[index]["sample_id"] or row["status"] != "success":
                raise ValueError("Progress identity or status mismatch")
            results[index] = row
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-repo", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("longbench_v1", "longbench_v2"), required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--retrieval-ratio", type=float, default=0.018)
    parser.add_argument("--estimation-ratio", type=float, default=0.232)
    parser.add_argument("--limit", type=int, default=0, help="Smoke subset only; 0 means all samples")
    args = parser.parse_args()
    if args.max_model_len <= 0 or args.max_batch <= 0 or args.limit < 0:
        raise ValueError("Length, batch size and limit must be valid")

    samples = load_samples(args.input)
    if args.limit:
        samples = samples[:args.limit]
    for sample in samples:
        if len(sample["prompt_token_ids"]) + sample["max_new_tokens"] > args.max_model_len:
            raise ValueError(f"Prompt exceeds model length: {sample['sample_id']}")
    batches = make_batches(samples, args.max_batch)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "run_status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("status") == "completed":
        raise FileExistsError("Completed author run already exists")
    author_commit = subprocess.check_output(
        ["git", "-C", str(args.author_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    metadata = {
        "benchmark": args.benchmark,
        "implementation": "author_retroinfer_gpu_only",
        "author_repo": str(args.author_repo.resolve()),
        "author_commit": author_commit,
        "adapter_sha256": file_sha256(Path(__file__)),
        "model_path": str(args.model_path.resolve()),
        "input": str(args.input.resolve()),
        "input_sha256": file_sha256(args.input),
        "samples": len(samples),
        "max_model_len": args.max_model_len,
        "max_batch": args.max_batch,
        "retrieval_ratio": args.retrieval_ratio,
        "estimation_ratio": args.estimation_ratio,
        "greedy": True,
        "gpu_only": True,
        "ignore_eos_in_model": False,
        "batch_sizes": [len(batch) for batch in batches],
    }
    config_path = args.output_dir / "resolved_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != metadata:
        raise ValueError("Existing run metadata differs from requested configuration")
    config_path.write_text(json.dumps(metadata, indent=2) + "\n")
    status_path.write_text(json.dumps({"status": "running"}) + "\n")

    sys.path.insert(0, str(args.author_repo.resolve()))
    import torch
    from transformers import AutoTokenizer
    from config import generate_config
    from model_hub.llama import LlamaModel

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    eos_ids = json.loads((args.model_path / "generation_config.json").read_text())["eos_token_id"]
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    eos_ids = set(eos_ids)
    torch.manual_seed(20260901)
    model = LlamaModel(str(args.model_path), args.max_model_len, torch.bfloat16, "cuda:0", tokenizer)
    progress_path = args.output_dir / "raw_outputs.partial.jsonl"
    completed = read_progress(progress_path, samples)
    started = time.monotonic()
    with progress_path.open("a", encoding="utf-8") as progress:
        for batch_number, batch in enumerate(batches):
            if all(index in completed for index in batch):
                continue
            if any(index in completed for index in batch):
                raise ValueError("Partial batch progress is not resumable")
            selected = [samples[index] for index in batch]
            padded_length = max(len(sample["prompt_token_ids"]) for sample in selected)
            budget = int(selected[0]["max_new_tokens"])
            input_ids = torch.full(
                (len(batch), padded_length), tokenizer.pad_token_id,
                dtype=torch.long, device="cuda:0",
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, sample in enumerate(selected):
                tokens = sample["prompt_token_ids"]
                input_ids[row, -len(tokens):] = torch.tensor(tokens, device="cuda:0")
                attention_mask[row, -len(tokens):] = 1
            config = generate_config(
                str(args.model_path), padded_length, "RetroInfer",
                args.retrieval_ratio, args.estimation_ratio, cache_ratio=0,
                use_cuda_graph=False, gpu_only=True,
            )
            output_ids = model.generate(
                attention_type="RetroInfer", inputs_ids=input_ids,
                attention_masks=attention_mask, max_new_length=budget,
                attn_config=config, do_sample=False, ignore_eos=False,
                prefill_bsz=1, prefill_method="full",
            )
            if len(output_ids) != len(batch):
                raise RuntimeError("Author model returned the wrong batch size")
            for index, generated in zip(batch, output_ids, strict=True):
                if not 0 < len(generated) <= budget:
                    raise RuntimeError(f"Author model returned an invalid token count for {samples[index]['sample_id']}")
                first_eos = next((i for i, token in enumerate(generated) if token in eos_ids), len(generated))
                raw_pred = tokenizer.decode(
                    generated[:first_eos], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                row = {"index": index, "sample_id": samples[index]["sample_id"],
                       "status": "success", "raw_pred": raw_pred,
                       "prompt_tokens": len(samples[index]["prompt_token_ids"]),
                       "generated_tokens_before_eos": first_eos,
                       "author_batch_size": len(batch)}
                progress.write(json.dumps(row, ensure_ascii=False) + "\n")
                completed[index] = row
            progress.flush()
            print(f"Author progress {len(completed)}/{len(samples)} batch={batch_number + 1}/{len(batches)} "
                  f"size={len(batch)} padded_len={padded_length}", flush=True)
    if len(completed) != len(samples):
        raise RuntimeError("Author result coverage mismatch")
    with (args.output_dir / "raw_outputs.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(len(samples)):
            handle.write(json.dumps(completed[index], ensure_ascii=False) + "\n")
    status_path.write_text(json.dumps({"status": "completed", "samples": len(samples),
                                       "elapsed_seconds": time.monotonic() - started}, indent=2) + "\n")


if __name__ == "__main__":
    main()
