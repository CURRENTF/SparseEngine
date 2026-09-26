"""Run the aligned AIME batch with the pinned upstream R-KV vLLM port."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

# The official R-KV driver uses a local callable RPC to count real compactions.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def git_commit(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def read_compactions(llm) -> int:
    def get_count(worker):
        runner = getattr(worker, "model_runner", None)
        compactor = getattr(runner, "rkv_compactor", None)
        if compactor is None:
            raise RuntimeError("R-KV compactor missing from a vLLM worker")
        return int(compactor._n_compactions)

    counts = llm.collective_rpc(get_count)
    if not counts or min(counts) <= 0:
        raise RuntimeError(f"No R-KV compactions were observed: {counts}")
    return max(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--rkv-source", type=Path, required=True)
    parser.add_argument("--rkv-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.87)
    parser.add_argument("--concurrency", type=int, default=60)
    parser.add_argument("--max-model-len", type=int, default=41984)
    parser.add_argument("--max-new-tokens", type=int, default=40960)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inputs = args.inputs.resolve(strict=True)
    model = args.model.resolve(strict=True)
    vllm_source = args.vllm_source.resolve(strict=True)
    rkv_source = args.rkv_source.resolve(strict=True)
    if git_commit(vllm_source) != "752a3a504485790a2e8491cacbb35c137339ad34":
        raise ValueError("vLLM source is not the pinned v0.25.1 revision")
    if git_commit(rkv_source) != args.rkv_commit:
        raise ValueError("R-KV source revision differs from the requested commit")
    if not (vllm_source / "vllm" / "rkv" / "integration.py").is_file():
        raise FileNotFoundError("The upstream R-KV vLLM patch is missing")
    if not (model / "config.json").is_file():
        raise FileNotFoundError(model / "config.json")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.concurrency <= 60 or not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("Concurrency must be 1..60 and GPU memory fraction must be in (0, 1]")
    if (os.environ.get("VLLM_V1_R_KV_BUDGET") != "4176"
            or os.environ.get("VLLM_V1_R_KV_BUFFER") != "1024"):
        raise ValueError("R-KV budget 4176 and buffer 1024 are required")
    rows = json.loads(inputs.read_text(encoding="utf-8"))
    if len(rows) != 60 or [str(row["id"]) for row in rows] != [str(i) for i in range(60)]:
        raise ValueError("Expected the aligned 60 AIME requests with IDs 0..59")
    if any(
        not isinstance(row.get("input_ids"), list)
        or not row["input_ids"]
        or not isinstance(row.get("gold"), dict)
        or len(row["input_ids"]) + args.max_new_tokens > args.max_model_len
        for row in rows
    ):
        raise ValueError("Invalid prompt IDs, gold row, or context length")

    import vllm
    from vllm import LLM, SamplingParams

    if not Path(vllm.__file__).resolve().is_relative_to(vllm_source):
        raise RuntimeError(f"Unexpected vLLM import: {vllm.__file__}")

    sampling = {
        "max_tokens": args.max_new_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
    }
    engine_args = {
        "model": str(model),
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.concurrency,
        "max_num_batched_tokens": 65536,
        "block_size": 16,
        "enable_prefix_caching": False,
        "enforce_eager": False,
        "disable_log_stats": False,
        "seed": args.seed,
    }
    rkv_env = {
        key: os.environ[key] for key in sorted(os.environ)
        if key.startswith("VLLM_V1_R_KV_")
    }
    manifest = {
        "status": "running",
        "engine": "upstream_rkv_vllm",
        "model": str(model),
        "input_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
        "requests": len(rows),
        "vllm_source": str(vllm_source),
        "vllm_version": vllm.__version__,
        "vllm_commit": git_commit(vllm_source),
        "rkv_source": str(rkv_source),
        "rkv_commit": args.rkv_commit,
        "sampling": sampling,
        "engine_args": engine_args,
        "rkv_env": rkv_env,
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "manifest.json", manifest)

    try:
        llm = LLM(**engine_args)
        params = SamplingParams(**sampling)
        prompts = [{"prompt_token_ids": row["input_ids"]} for row in rows]
        start = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        if len(outputs) != len(rows):
            raise RuntimeError(f"Expected 60 vLLM outputs, got {len(outputs)}")
        compactions = read_compactions(llm)
        completion_tokens = 0
        with (args.output / "raw_outputs.jsonl").open("w", encoding="utf-8") as raw_out, (
            args.output / "aime2024.jsonl"
        ).open("w", encoding="utf-8") as pred_out:
            for row, result in zip(rows, outputs):
                if list(result.prompt_token_ids) != row["input_ids"]:
                    raise RuntimeError(f"Request order mismatch for {row['id']}")
                if len(result.outputs) != 1:
                    raise RuntimeError(f"Expected one completion for {row['id']}")
                completion = result.outputs[0]
                if completion.finish_reason not in ("stop", "length"):
                    raise RuntimeError(f"Request {row['id']} ended as {completion.finish_reason}")
                completion_tokens += len(completion.token_ids)
                raw_out.write(json.dumps({
                    "id": row["id"], "prompt_tokens": len(result.prompt_token_ids),
                    "completion_tokens": len(completion.token_ids),
                    "completion_ids": list(completion.token_ids),
                    "text": completion.text, "finish_reason": completion.finish_reason,
                    "status": "success",
                }, ensure_ascii=False) + "\n")
                pred_out.write(json.dumps({
                    "id": row["id"], "status": "success", "pred": completion.text,
                    "gold": row["gold"],
                }, ensure_ascii=False) + "\n")
        manifest.update(
            status="generated", compactions=compactions,
            generation_elapsed_s=elapsed, completion_tokens=completion_tokens,
            completion_tokens_per_s=completion_tokens / elapsed,
        )
        write_json(args.output / "manifest.json", manifest)
        print(json.dumps({"status": "generated", "generation_elapsed_s": elapsed,
                          "completion_tokens": completion_tokens,
                          "compactions": compactions}), flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(args.output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
