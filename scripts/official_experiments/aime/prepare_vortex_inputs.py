"""Export the exact MathBench AIME prompts as token IDs for Vortex."""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from benchmark.math_bench.pred import _build_prompt, _get_problem_text, build_chat
from sparseengine.engine.input_processor import tokenize_text_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    args = parser.parse_args()

    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    if len(data) != 60 or [str(row["id"]) for row in data] != [str(i) for i in range(60)]:
        raise ValueError("Expected the aligned 60 AIME requests with IDs 0..59")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    max_prompt_len = max(1, args.max_model_len - args.max_new_tokens - 32)
    rows = []
    for row in data:
        problem = _get_problem_text(row, "aime2024")
        prompt = _build_prompt("aime2024", problem, "", "deepseek")
        raw_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(raw_ids) > max_prompt_len:
            half = max_prompt_len // 2
            prompt = (
                tokenizer.decode(raw_ids[:half], skip_special_tokens=True)
                + tokenizer.decode(raw_ids[-half:], skip_special_tokens=True)
            )
        prompt = build_chat(tokenizer, prompt, False)
        token_ids = tokenize_text_prompt(tokenizer, prompt)
        if len(token_ids) + args.max_new_tokens > args.max_model_len:
            raise ValueError(f"Request {row['id']} exceeds context length")
        rows.append({"id": str(row["id"]), "prompt": prompt, "input_ids": token_ids, "gold": row})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "requests": len(rows),
        "prompt_tokens_min": min(len(row["input_ids"]) for row in rows),
        "prompt_tokens_max": max(len(row["input_ids"]) for row in rows),
        "output": str(args.output),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }))


if __name__ == "__main__":
    main()
