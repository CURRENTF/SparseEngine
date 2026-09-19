import os
import torch
import argparse
from pathlib import Path
from sparseengine import LLM, SamplingParams
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = Path(os.getenv("DELTAKV_MODEL_ROOT", PROJECT_ROOT / "models"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_ROOT / "Qwen3-0.6B"))
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--sparse_method", type=str, default="vanilla")
    parser.add_argument("--no_tqdm", action="store_true")
    args = parser.parse_args()
    
    path = args.model_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model path {path} not found.")
    
    print(f"Loading model from: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, tensor_parallel_size=args.tp, sparse_method=args.sparse_method)
    
    # 接近贪婪搜索，保证结果可复现，方便对比
    sampling_params = SamplingParams(temperature=1e-5, max_tokens=128)

    print("\n" + "="*20 + " Test Case 1: Heterogeneous Batching " + "="*20)
    # 不同长度、不同内容的 Prompt 同时生成，检查 Batch 管理是否隔离
    prompts = [
        "How to make a cake?",
        "Translate to Chinese: 'I love programming.'",
        "1 + 1 =",
        "Explain the theory of relativity in one sentence."
    ]
    
    outputs = llm.generate(prompts, sampling_params, use_tqdm=not args.no_tqdm)
    for i, out in enumerate(outputs):
        print(f"\n[Prompt {i}]: {prompts[i]}")
        print(f"[Output {i}]: {out['text'][:200]}...")

    print("\n" + "="*20 + " Test Case 2: Long Generation (Cache Stability) " + "="*20)
    # 检查生成大量 Token 后，KV Cache 索引是否偏移或损坏
    long_gen_params = SamplingParams(temperature=0.7, max_tokens=512)
    long_prompt = "Write a long fairy tale about a brave small model in a big GPU forest."
    
    out = llm.generate([long_prompt], long_gen_params, use_tqdm=not args.no_tqdm)
    text = out[0]['text']
    print(f"\n[Long Output (Length {len(text)})]:\n{text[:500]}...")
    
    # 简单启发式检查：如果出现大量重复词或乱码，说明 KV Cache 逻辑有 bug
    if "the the the" in text or "  " in text[:50] and text[5:10] == text[10:15]:
        print("\n[WARNING]: Potential KV Cache corruption detected (repetitive patterns)!")
    else:
        print("\n[INFO]: Long generation seems coherent.")

    print("\n" + "="*20 + " Test Case 3: Batch Consistency Check " + "="*20)
    # 检查单条生成的答案，是否和放在 Batch 里生成的一模一样
    # 这是验证 b_start_loc 和 cu_seqlens 逻辑最硬核的方式
    test_p = "The capital of France is"
    
    # 1. 单独生成
    out_single = llm.generate([test_p], sampling_params, use_tqdm=False)
    # 2. 放在 Batch 中间生成
    batch_p = ["Something else", test_p, "Another random prompt"]
    out_batch = llm.generate(batch_p, sampling_params, use_tqdm=False)
    
    if out_single[0]['text'] == out_batch[1]['text']:
        print("\n[SUCCESS]: Batch consistency check passed!")
    else:
        print("\n[FAIL]: Batch consistency check failed!")
        print(f"Single mode: {out_single[0]['text']!r}")
        print(f"Batch mode : {out_batch[1]['text']!r}")

if __name__ == "__main__":
    main()
