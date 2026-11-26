"""PACE quickstart: plan-aware KV retention on one long prompt.

A short fact sits among thousands of filler lines, followed by a question that needs it.
The KV budget is far below the prompt length; the PACEPlan names the fact's token span, so
PACE keeps it. `--sparse_method h2o` runs the same prompt without a plan for comparison.
On this single prompt H2O usually keeps the fact too, because it stands out from uniform
filler; the script shows how to pass a plan and checks that PACE works, not that it beats
H2O. Plans matter when a span gets little attention at the time it is evicted but is needed
later, as in multi-step agent sessions.

    python scripts/validation/pace_quickstart.py --model_path /path/to/Qwen3-8B
    python scripts/validation/pace_quickstart.py --model_path /path/to/Qwen3-8B --sparse_method h2o
    # past the model's RoPE range (bounded positions):
    python scripts/validation/pace_quickstart.py --model_path /path/to/Qwen3-8B --bounded --filler_lines 7000
"""
import argparse
import os
import random
import sys

from transformers import AutoTokenizer

from sparseengine import LLM, PACEPlan, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="local model directory (Qwen2, Qwen3 or Llama family)")
    parser.add_argument("--sparse_method", choices=["pace", "h2o"], default="pace")
    parser.add_argument("--budget", type=int, default=2048, help="KV budget in tokens")
    parser.add_argument("--filler_lines", type=int, default=2500,
                        help="filler lines in the prompt (~14 tokens each)")
    parser.add_argument("--fact_position", type=float, default=0.5,
                        help="where the fact sits among the filler lines, 0 (start) to 1 (end)")
    parser.add_argument("--bounded", action="store_true",
                        help="PACE with bounded positions (pace_position_limit=8192)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path {args.model_path} not found.")
    if args.bounded and args.sparse_method != "pace":
        parser.error("--bounded requires --sparse_method pace")

    rng = random.Random(args.seed)
    code = f"{rng.randint(1000, 9999)}-{rng.choice('ABCDEFGHJKMNPQRSTUVWXYZ')}{rng.choice('ABCDEFGHJKMNPQRSTUVWXYZ')}"
    fact = f"The release code for project Kestrel is {code}."
    lines = [f"Note {i}: routine status update, nothing to report." for i in range(args.filler_lines)]
    lines.insert(round(min(max(args.fact_position, 0.0), 1.0) * len(lines)), fact)
    question = "What is the release code for project Kestrel? Reply with the code only."

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "\n".join(lines) + "\n" + question}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    start_char = text.index(fact)
    end_char = start_char + len(fact)
    fact_tokens = [i for i, (a, b) in enumerate(enc["offset_mapping"])
                   if a < end_char and b > start_char]

    kwargs = dict(sparse_method=args.sparse_method, h2o_decode_eviction=True,
                  h2o_decode_budget=args.budget, h2o_prefill_budget=2 * args.budget,
                  decode_graph=False)
    if args.bounded:
        # the resident row (prefill budget + chunk) plus renumber headroom must stay below
        # the position limit
        kwargs.update(pace_bounded_positions=True, pace_position_limit=8192,
                      engine_prefill_chunk_size=1024)
    llm = LLM(args.model_path, **kwargs)
    sampling = SamplingParams(temperature=1e-5, max_tokens=32)
    if args.sparse_method == "pace":
        # the fact's tokens [start, end) will be read at step 1, the current step
        plan = PACEPlan(current_step=1, spans=((fact_tokens[0], fact_tokens[-1] + 1, (1,)),))
        out = llm.generate([enc["input_ids"]], sampling, pace_plans=[plan])
    else:
        out = llm.generate([enc["input_ids"]], sampling)
    answer = out[0]["text"].strip()
    correct = code in answer
    print(f"{args.sparse_method}{' (bounded positions)' if args.bounded else ''}: "
          f"prompt {len(enc['input_ids'])} tokens, fact at tokens {fact_tokens[0]}-{fact_tokens[-1]}, "
          f"KV budget {args.budget}, "
          f"expected {code!r}, answer {answer!r} -> {'correct' if correct else 'wrong'}")
    if args.sparse_method == "pace" and not correct:
        sys.exit(1)


if __name__ == "__main__":
    main()
