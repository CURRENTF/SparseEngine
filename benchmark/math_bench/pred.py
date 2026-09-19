import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer

from benchmark.model_adapters.sparseengine import get_sparseengine_generate_api

# Keep defaults consistent with benchmark/long_bench/pred.py, but allow env overrides.
BASE_PATH = os.getenv("SPARSEENGINE_OUTPUT_DIR", str(REPO_ROOT / "outputs"))
DATA_PREFIX_PATH = os.getenv("SPARSEENGINE_DATA_DIR")
DEFAULT_GSM8K_DATASET = ("openai/gsm8k", "main", "test")
DEFAULT_AIME2024_DATASET = ("Maxwell-Jia/AIME_2024", None, "train")
DEFAULT_MATH500_DATASET = ("HuggingFaceH4/MATH-500", None, "test")
DEFAULT_HMMT_NOV_DATASET = ("MathArena/hmmt_nov_2025", None, "train")
OPENR1_MATH_QUERY_TEMPLATE = (
    "Solve the following math problem efficiently and clearly.\n"
    "The last line of your response should be of the following format: "
    "'Therefore, the final answer is: $\\\\boxed{{ANSWER}}$. I hope it is correct' "
    "(without quotes) where ANSWER is just the final number or expression that solves the problem. "
    "Think step by step before answering.\n"
    "{problem}"
)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def build_chat(
    tokenizer,
    prompt: str,
    no_chat_template: bool,
    *,
    prefill_think_prefix: bool = False,
    think_prefix: str = "<think>\n",
) -> str:
    if not no_chat_template and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        msgs = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=os.getenv("ENABLE_THINKING", "1") not in ("0", "false", "False"),
        )
    if prefill_think_prefix and not prompt.endswith(think_prefix):
        prompt = prompt + think_prefix
    if os.getenv("DEBUG"):
        print("input prompt:", prompt)
    return prompt


def _read_json_or_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON list in {path}")
            return data
        return [json.loads(line) for line in f if line.strip()]


def _load_hf_dataset(dataset_name: str, config_name: str, split: str):
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("datasets is required for HF dataset loading. Install `datasets`.") from e

    if config_name:
        ds = load_dataset(dataset_name, config_name, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    return [ds[i] for i in range(len(ds))]


def _resolve_default_data_path(data_dir: str, dataset: str, split: str) -> str:
    candidates = []
    if dataset == "gsm8k":
        candidates = [
            f"gsm8k/{split}.jsonl",
            f"gsm8k/{split}.json",
            f"GSM8K/{split}.jsonl",
            f"GSM8K/{split}.json",
            f"gsm8k_{split}.jsonl",
            f"gsm8k_{split}.json",
        ]
    elif dataset == "aime2024":
        candidates = [
            f"aime2024/{split}.jsonl",
            f"aime2024/{split}.json",
            f"aime_2024/{split}.jsonl",
            f"aime_2024/{split}.json",
            f"AIME2024/{split}.jsonl",
            f"AIME2024/{split}.json",
            f"aime2024_{split}.jsonl",
            f"aime2024_{split}.json",
        ]
    elif dataset == "math500":
        candidates = [
            f"math500/{split}.jsonl",
            f"math500/{split}.json",
            f"math_500/{split}.jsonl",
            f"math_500/{split}.json",
            f"MATH-500/{split}.jsonl",
            f"MATH-500/{split}.json",
            f"MATH500/{split}.jsonl",
            f"MATH500/{split}.json",
            f"math500_{split}.jsonl",
            f"math500_{split}.json",
        ]
    elif dataset == "hmmt_nov":
        candidates = [
            f"hmmt_nov/{split}.jsonl",
            f"hmmt_nov/{split}.json",
            f"hmmt_nov_2025/{split}.jsonl",
            f"hmmt_nov_2025/{split}.json",
            f"hmmt_nov_{split}.jsonl",
            f"hmmt_nov_{split}.json",
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    for rel in candidates:
        path = os.path.join(data_dir, rel)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Cannot find dataset file for {dataset} (split={split}) under {data_dir}. "
        f"Tried: {', '.join(candidates)}. Use --data_path_{dataset} to override."
    )


def _get_problem_text(example: dict, dataset: str) -> str:
    # HF AIME_2024 uses "Problem"; GSM8K uses "question".
    keys = ["question", "Question", "problem", "Problem", "prompt", "input"]
    for k in keys:
        if k in example and isinstance(example[k], str) and example[k].strip():
            return example[k].strip()
    raise KeyError(f"Cannot find problem text keys {keys} for dataset={dataset}. Keys: {list(example.keys())}")


def _get_example_id(example: dict, idx: int) -> str:
    for k in ("id", "idx", "index", "qid", "uid"):
        if k in example:
            return str(example[k])
    return str(idx)


def _build_prompt(dataset: str, problem: str, think_instruction: str, prompt_style: str) -> str:
    if prompt_style == "openr1":
        if dataset != "math500":
            raise ValueError("prompt_style='openr1' is only defined for math500.")
        return OPENR1_MATH_QUERY_TEMPLATE.format(problem=problem)

    prompt_format = {
        "gsm8k": (
            "Please reason step by step, and put your final answer within \\\\boxed{{}}.\n"
            "{think_instruction}\n"
            "Problem:\n{problem}\n"
        ),
        "aime2024": (
            "Please reason step by step, and put your final answer within \\\\boxed{{}}.\n"
            "The final answer is an integer.\n"
            "{think_instruction}\n"
            "Problem:\n{problem}\n"
        ),
        "math500": (
            "Please reason step by step, and put your final answer within \\\\boxed{{}}.\n"
            "{think_instruction}\n"
            "Problem:\n{problem}\n"
        ),
        "hmmt_nov": (
            "Please reason step by step, and put your final answer within \\\\boxed{{}}.\n"
            "{think_instruction}\n"
            "Problem:\n{problem}\n"
        ),
    }[dataset]
    return prompt_format.format(problem=problem, think_instruction=think_instruction)


def load_model_and_tokenizer(rank: int, args):
    infer_config = {
        "max_model_len": args.max_model_len,
    }

    if args.hyper_param:
        if os.path.exists(args.hyper_param):
            with open(args.hyper_param, "r") as f:
                extra_config = json.load(f)
            infer_config.update(extra_config)
            print(f"Loaded hyper-parameters from {args.hyper_param}: {extra_config}")
        else:
            try:
                extra_config = json.loads(args.hyper_param)
                infer_config.update(extra_config)
                print(f"Parsed hyper-parameters from string: \n{extra_config}")
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse --hyper_param '{args.hyper_param}'. "
                    f"It is neither a valid file path nor a valid JSON string. Error: {e}"
                )

    generate_fn = get_sparseengine_generate_api(
        model_path=args.model_path,
        infer_config=infer_config,
        deltakv_checkpoint_path=args.deltakv_checkpoint_path,
        sparse_method=args.sparse_method,
    )

    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return generate_fn, tokenizer


def _count_generated_text_tokens(tokenizer, texts: list[str]) -> int:
    if not texts:
        return 0
    tokenized = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)
    return int(sum(len(ids) for ids in tokenized["input_ids"]))


def _decode_cuda_graph_status(generate_fn) -> dict:
    llm = getattr(generate_fn, "_sparseengine_llm", None)
    runner = getattr(getattr(llm, "model_runner", None), "decode_graph_runner", None)
    states = getattr(runner, "_graphs", {}) if runner is not None else {}
    graph_count = sum(
        1
        for state in states.values()
        if getattr(state, "graph", None) is not None
    )
    configured = bool(getattr(getattr(llm, "config", None), "decode_graph", False))
    return {
        "decode_graph_configured": configured,
        "decode_graph_runner_initialized": runner is not None,
        "decode_graph_state_count": int(len(states)),
        "decode_graph_graph_count": int(graph_count),
        "decode_graph_last_state_key": str(getattr(runner, "last_state_key", None)) if runner is not None else None,
        "decode_graph_active": bool(configured and graph_count > 0),
    }


def get_pred(rank: int, data, dataset: str, args, model, tokenizer, out_path: str) -> dict:
    think_instruction = ""
    if not args.no_prompt_think_instruction:
        think_instruction = 'Begin your response with "<think>\\n" and do not output an empty think block.\n'

    max_gen = args.max_new_tokens
    batch_size = args.batch_size
    max_prompt_len = max(1, int(args.max_model_len) - int(max_gen) - 32)
    perf = {
        "rank": rank,
        "dataset": dataset,
        "samples": 0,
        "batches": [],
        "generated_text_tokens": 0,
        "generation_elapsed_s": 0.0,
        "generated_text_tokens_per_s": 0.0,
    }

    for i in tqdm(range(0, len(data), batch_size), desc=f"[Rank {rank}] {dataset}"):
        batch_data = data[i : i + batch_size]
        prompts = []
        meta = []
        for j, example in enumerate(batch_data):
            problem = _get_problem_text(example, dataset)
            prompt = _build_prompt(dataset, problem, think_instruction, args.prompt_style)

            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokenized_prompt) > max_prompt_len:
                half = int(max_prompt_len / 2)
                prompt = (
                    tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
                    + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
                )
            prompts.append(
                build_chat(
                    tokenizer,
                    prompt,
                    args.no_chat_template,
                    prefill_think_prefix=args.prefill_think_prefix,
                    think_prefix=args.think_prefix,
                )
            )
            meta.append({"id": _get_example_id(example, i + j)})

        eos_token_id = [tokenizer.eos_token_id]
        if hasattr(tokenizer, "eot_token_id"):
            eos_token_id.append(tokenizer.eot_token_id)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_start = time.perf_counter()
        preds = model(
            prompts,
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            eos_token_id=eos_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_elapsed = time.perf_counter() - batch_start
        if isinstance(preds, str):
            preds = [preds]
        batch_tokens = _count_generated_text_tokens(tokenizer, preds)
        perf["samples"] += len(batch_data)
        perf["generated_text_tokens"] += batch_tokens
        perf["generation_elapsed_s"] += batch_elapsed
        perf["batches"].append(
            {
                "start_index": i,
                "batch_size": len(batch_data),
                "generated_text_tokens": batch_tokens,
                "generation_elapsed_s": batch_elapsed,
                "generated_text_tokens_per_s": batch_tokens / batch_elapsed if batch_elapsed > 0 else 0.0,
            }
        )

        for example, pred, info in zip(batch_data, preds, meta):
            if args.force_think_prefix:
                if pred.startswith("<think>") and not pred.startswith("<think>\n"):
                    pred = "<think>\n" + pred[len("<think>") :].lstrip("\n")
                elif not pred.startswith("<think>\n"):
                    pred = args.think_prefix + pred
            record = {
                "id": info["id"],
                "status": "success",
                "pred": pred,
                "gold": example,
            }
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
    perf["generated_text_tokens_per_s"] = (
        perf["generated_text_tokens"] / perf["generation_elapsed_s"]
        if perf["generation_elapsed_s"] > 0
        else 0.0
    )
    return perf


def worker(rank: int, world_size: int, datasets: List[str], args, out_root: str) -> None:
    seed_everything(42)
    model, tokenizer = load_model_and_tokenizer(rank, args)
    perf_records = []

    for dataset in datasets:
        if dataset == "gsm8k":
            if args.data_path_gsm8k:
                data = _read_json_or_jsonl(args.data_path_gsm8k)
            else:
                try:
                    data = _load_hf_dataset(args.hf_dataset_gsm8k, args.hf_config_gsm8k, args.hf_split_gsm8k)
                except Exception as e:
                    if os.getenv("DEBUG"):
                        print(f"[gsm8k] HF load failed ({e}); falling back to local files under {args.data_dir}")
                    data_path = _resolve_default_data_path(args.data_dir, "gsm8k", args.split)
                    data = _read_json_or_jsonl(data_path)
        elif dataset == "aime2024":
            if args.data_path_aime2024:
                data = _read_json_or_jsonl(args.data_path_aime2024)
            else:
                try:
                    data = _load_hf_dataset(args.hf_dataset_aime2024, args.hf_config_aime2024, args.hf_split_aime2024)
                except Exception as e:
                    if os.getenv("DEBUG"):
                        print(f"[aime2024] HF load failed ({e}); falling back to local files under {args.data_dir}")
                    data_path = _resolve_default_data_path(args.data_dir, "aime2024", args.split)
                    data = _read_json_or_jsonl(data_path)
        elif dataset == "math500":
            if args.data_path_math500:
                data = _read_json_or_jsonl(args.data_path_math500)
            else:
                try:
                    data = _load_hf_dataset(args.hf_dataset_math500, args.hf_config_math500, args.hf_split_math500)
                except Exception as e:
                    if os.getenv("DEBUG"):
                        print(f"[math500] HF load failed ({e}); falling back to local files under {args.data_dir}")
                    data_path = _resolve_default_data_path(args.data_dir, "math500", args.split)
                    data = _read_json_or_jsonl(data_path)
        elif dataset == "hmmt_nov":
            if args.data_path_hmmt_nov:
                data = _read_json_or_jsonl(args.data_path_hmmt_nov)
            else:
                try:
                    data = _load_hf_dataset(args.hf_dataset_hmmt_nov, args.hf_config_hmmt_nov, args.hf_split_hmmt_nov)
                except Exception as e:
                    if os.getenv("DEBUG"):
                        print(f"[hmmt_nov] HF load failed ({e}); falling back to local files under {args.data_dir}")
                    data_path = _resolve_default_data_path(args.data_dir, "hmmt_nov", args.split)
                    data = _read_json_or_jsonl(data_path)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        if args.num_samples:
            data = data[: args.num_samples]

        data_subset = data[rank::world_size]
        if not data_subset:
            continue

        out_path = os.path.join(out_root, f"{dataset}.jsonl")
        perf_records.append(get_pred(rank, data_subset, dataset, args, model, tokenizer, out_path))
        torch.cuda.empty_cache()

    total_tokens = sum(int(record["generated_text_tokens"]) for record in perf_records)
    total_elapsed = sum(float(record["generation_elapsed_s"]) for record in perf_records)
    graph_status = _decode_cuda_graph_status(model)
    perf_summary = {
        "rank": rank,
        "world_size": world_size,
        "backend": "sparseengine",
        "sparse_method": args.sparse_method,
        "model": args.model,
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path or args.model_path,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "prompt_style": args.prompt_style,
        "prefill_think_prefix": args.prefill_think_prefix,
        "force_think_prefix": args.force_think_prefix,
        "prompt_think_instruction": not args.no_prompt_think_instruction,
        "think_prefix": args.think_prefix,
        "hyper_param": getattr(model, "_sparseengine_infer_config", args.hyper_param),
        "datasets": perf_records,
        "generated_text_tokens": total_tokens,
        "generation_elapsed_s": total_elapsed,
        "generated_text_tokens_per_s": total_tokens / total_elapsed if total_elapsed > 0 else 0.0,
        **graph_status,
    }
    perf_path = os.path.join(out_root, f"perf_rank{rank}.json")
    with open(perf_path, "w", encoding="utf-8") as f:
        json.dump(perf_summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote performance summary to: {perf_path}")
    if (
        bool(perf_summary.get("decode_graph_configured"))
        and not bool(perf_summary.get("decode_graph_active"))
    ):
        raise RuntimeError(
            "decode_graph=True was configured, but no active decode CUDA graph "
            f"was observed. See {perf_path}."
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="my_model")
    parser.add_argument("--ws", default=1, type=int, help="world size")
    parser.add_argument("--task", default="gsm8k,aime2024", type=str, help="Comma-separated: gsm8k,aime2024,math500,hmmt_nov")
    parser.add_argument("--split", default="test", type=str, help="Dataset split name (used for default path resolution)")
    parser.add_argument("--data_dir", default=DATA_PREFIX_PATH, type=str, help="Root folder for datasets")
    parser.add_argument("--data_path_gsm8k", default=None, type=str)
    parser.add_argument("--data_path_aime2024", default=None, type=str)
    parser.add_argument("--data_path_math500", default=None, type=str)
    parser.add_argument("--data_path_hmmt_nov", default=None, type=str)
    parser.add_argument("--hf_dataset_gsm8k", default=DEFAULT_GSM8K_DATASET[0], type=str)
    parser.add_argument("--hf_config_gsm8k", default=DEFAULT_GSM8K_DATASET[1], type=str)
    parser.add_argument("--hf_split_gsm8k", default=DEFAULT_GSM8K_DATASET[2], type=str)
    parser.add_argument("--hf_dataset_aime2024", default=DEFAULT_AIME2024_DATASET[0], type=str)
    parser.add_argument("--hf_config_aime2024", default=DEFAULT_AIME2024_DATASET[1], type=str)
    parser.add_argument("--hf_split_aime2024", default=DEFAULT_AIME2024_DATASET[2], type=str)
    parser.add_argument("--hf_dataset_math500", default=DEFAULT_MATH500_DATASET[0], type=str)
    parser.add_argument("--hf_config_math500", default=DEFAULT_MATH500_DATASET[1], type=str)
    parser.add_argument("--hf_split_math500", default=DEFAULT_MATH500_DATASET[2], type=str)
    parser.add_argument("--hf_dataset_hmmt_nov", default=DEFAULT_HMMT_NOV_DATASET[0], type=str)
    parser.add_argument("--hf_config_hmmt_nov", default=DEFAULT_HMMT_NOV_DATASET[1], type=str)
    parser.add_argument("--hf_split_hmmt_nov", default=DEFAULT_HMMT_NOV_DATASET[2], type=str)

    # DeltaKV related arguments (aligned with benchmark/long_bench/pred.py)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--deltakv_checkpoint_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--sparse_method", type=str, default="deltakv")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples per task")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--no_chat_template", action="store_true", help="Do not use chat template")
    parser.add_argument("--prompt_style", choices=["deepseek", "openr1"], default="deepseek")
    parser.add_argument("--hyper_param", type=str, default=None, help="Path to JSON file or inline JSON string")
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--max_model_len", type=int, default=131000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--think_prefix", type=str, default="<think>\n")
    parser.add_argument(
        "--prefill_think_prefix",
        action="store_true",
        help="Append --think_prefix to the actual generation prompt so the model continues after it.",
    )
    parser.add_argument(
        "--no_prompt_think_instruction",
        action="store_true",
        help="Remove the extra user-prompt sentence that asks the model to begin with <think>.",
    )
    parser.add_argument("--no_force_think_prefix", action="store_false", dest="force_think_prefix")
    parser.set_defaults(force_think_prefix=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mp.set_start_method("spawn", force=True)
    if not (0.5 <= float(args.temperature) <= 0.7):
        raise ValueError(f"--temperature must be within [0.5, 0.7], got {args.temperature}")

    model_name = args.model
    compressor_name = os.path.basename(args.deltakv_checkpoint_path.rstrip("/")) if args.deltakv_checkpoint_path else "None"

    datasets = [d.strip() for d in args.task.split(",") if d.strip()]
    time_tag = datetime.now().strftime("%m%d_%H%M")
    out_root = os.path.join(BASE_PATH, f"benchmark/math_bench/pred/{model_name}/{compressor_name}_{time_tag}")
    os.makedirs(out_root, exist_ok=True)
    print(f"Results will be saved in: {out_root}")

    for dataset in datasets:
        with open(os.path.join(out_root, f"{dataset}.jsonl"), "w", encoding="utf-8") as f:
            f.write("")

    if args.ws > 1:
        processes = []
        for rank in range(args.ws):
            p = mp.Process(target=worker, args=(rank, args.ws, datasets, args, out_root))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        worker(0, 1, datasets, args, out_root)

    log_path = os.path.join(BASE_PATH, "mathbench_eval.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command: python {' '.join(sys.argv)}\n")
        f.write(f"Output Root: {out_root}\n")
        f.write(f"Args: {json.dumps(vars(args), indent=2)}\n")
        f.write("-" * 80 + "\n")

    print(f"Evaluating {out_root} ...")
    eval_cmd = [sys.executable, "benchmark/math_bench/eval.py", "--path", out_root]
    try:
        subprocess.run(eval_cmd, check=True)
        result_path = os.path.join(out_root, "result.json")
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                scores = json.load(f)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("Evaluation Results (pass@1):\n")
                f.write(json.dumps(scores, indent=4, ensure_ascii=False))
                f.write("\n" + "=" * 80 + "\n\n")
            print(f"Wrote eval results to: {log_path}")
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed: {e}")
