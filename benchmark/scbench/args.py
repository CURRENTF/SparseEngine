# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import json
from argparse import ArgumentParser, Namespace

from eval_utils import DATA_NAME_TO_MAX_NEW_TOKENS

try:
    from minference import MInferenceConfig
    ATTN_TYPES = MInferenceConfig.get_available_attn_types()
    KV_TYPES = MInferenceConfig.get_available_kv_types()
except ImportError:
    ATTN_TYPES = ["hf", "vllm", "vllm_blend", "vllm_kv", "inf_llm"]
    KV_TYPES = ["dense", "retr_attn", "kivi"]

for attn_type in ("sparseengine",):
    if attn_type not in ATTN_TYPES:
        ATTN_TYPES.append(attn_type)


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument(
        "--task",
        type=str,
        required=True,
        help='Which task to use. Note that "all" can only be used in `compute_scores.py`.',  # noqa
    )
    p.add_argument(
        "--data_dir", type=str, default="../data", help="The directory of data."
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="benchmark/scbench/results",
        help="Where to dump the prediction results.",
    )  # noqa
    p.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-350m",
        help="For `compute_scores.py` only, specify which model you want to compute the score for.",  # noqa
    )
    p.add_argument(
        "--num_eval_examples",
        type=int,
        default=-1,
        help="The number of test examples to use, use all examples in default.",
    )  # noqa
    p.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="The index of the first example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data.",
    )  # noqa
    p.add_argument(
        "--stop_idx",
        type=int,
        help="The index of the last example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data. Defaults to the length of dataset.",
    )  # noqa
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--use_sparq", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_seq_length", type=int, default=131_072)
    p.add_argument(
        "--ws",
        type=int,
        default=1,
        help="World size for data parallel evaluation (one process per GPU).",
    )
    p.add_argument("--rewrite", action="store_true")
    p.add_argument("--topk", type=int, default=-1)
    p.add_argument("--starting_layer", type=int, default=-1)
    p.add_argument("--start_example_id", type=int, default=0)
    p.add_argument("--topk_dims_file_path", type=str, default=None)
    p.add_argument("--kv_cache_cpu", action="store_true")
    p.add_argument("--kv_cache_cpu_device", type=str, default="cpu")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument(
        "--model_torch_dtype",
        type=str,
        default=None,
        help="Optional model dtype override for non-quantized and skipped modules, e.g. bf16 or fp16.",
    )
    p.add_argument("--use_chat_template", action="store_true")
    p.add_argument("--same_context_different_query", action="store_true")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--max_turns", type=int, default=5)
    p.add_argument("--use_llmlingua", action="store_true")
    p.add_argument("--disable_golden_context", action="store_true")
    p.add_argument("--use_v2_data", action="store_true")
    p.add_argument("--copy_on_gpu", action="store_true", help="If true, backup KV cache on GPU instead of CPU.")
    p.add_argument(
        "--context_min_tokens",
        type=int,
        default=-1,
        help="Optional lower bound (inclusive) on the tokenized context length used to filter examples.",
    )
    p.add_argument(
        "--context_max_tokens",
        type=int,
        default=-1,
        help="Optional upper bound (exclusive) on the tokenized context length used to filter examples.",
    )
    p.add_argument(
        "--subset_indices_file",
        type=str,
        default=None,
        help="Optional path to a txt/json file containing example indices to evaluate.",
    )
    p.add_argument(
        "--num_data_shards",
        type=int,
        default=1,
        help="Split dataset indices into N modulo shards and evaluate only one shard.",
    )
    p.add_argument(
        "--data_shard_id",
        type=int,
        default=0,
        help="Modulo shard id in [0, num_data_shards).",
    )
    p.add_argument(
        "--attn_type",
        type=str,
        choices=ATTN_TYPES,
        default="hf",
    )
    p.add_argument(
        "--kv_type",
        type=str,
        default="dense",
        choices=KV_TYPES,
    )
    p.add_argument("--is_search", action="store_true")
    p.add_argument("--hyper_param", type=json.loads, default={})
    return p.parse_args()
