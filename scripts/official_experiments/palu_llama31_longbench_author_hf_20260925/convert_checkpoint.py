#!/usr/bin/env python3
"""Map SparseEngine's frozen grouped Palu factors into the authors' HF modules."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

from palu.model.svd_llama import PaluLlamaConfig, PaluLlamaForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    metadata = json.loads((args.factors / "palu.json").read_text())
    if metadata["group_size"] != 1 or any(pair != [96, 96] for pair in metadata["ranks"]):
        raise ValueError("Expected frozen group=1, K/V rank=96 factors")
    layers = metadata["model"]["num_hidden_layers"]
    heads = metadata["model"]["num_key_value_heads"]
    ranks = {
        f"model.layers.{layer}.self_attn.{projection}_proj": [96] * heads
        for layer in range(layers) for projection in ("k", "v")
    }
    base_config = AutoConfig.from_pretrained(args.base_model)
    config = PaluLlamaConfig(**{**base_config.to_dict(), "head_wise_ranks": ranks})
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    base_state = base.state_dict()
    for name, expected in metadata["source_weight_sha256"].items():
        actual = hashlib.sha256(base_state[name].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Source weight mismatch: {name}")
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = PaluLlamaForCausalLM(config)
    finally:
        torch.set_default_dtype(torch.float32)
    result = model.load_state_dict(base_state, strict=False)
    expected_missing = {
        f"model.layers.{layer}.self_attn.{projection}_proj.{suffix}"
        for layer in range(layers) for projection in ("k", "v")
        for suffix in (["VT.weight"] + [f"U.{head}.weight" for head in range(heads)])
    }
    expected_unexpected = {
        f"model.layers.{layer}.self_attn.{projection}_proj.weight"
        for layer in range(layers) for projection in ("k", "v")
    }
    if set(result.missing_keys) != expected_missing or set(result.unexpected_keys) != expected_unexpected:
        raise RuntimeError(f"Unexpected state mismatch: {result}")
    del base_state, base

    factor_path = args.factors / "palu.safetensors"
    with safe_open(factor_path, framework="pt", device="cpu") as factors:
        for layer in range(layers):
            attention = model.model.layers[layer].self_attn
            for projection, label in (("k", "key"), ("v", "value")):
                module = getattr(attention, f"{projection}_proj")
                down = factors.get_tensor(f"layers.{layer}.{label}_down")
                up = factors.get_tensor(f"layers.{layer}.{label}_up")
                if down.shape != (heads, 96, config.hidden_size) or up.shape != (heads, 96, config.hidden_size // config.num_attention_heads):
                    raise ValueError(f"Invalid factors for layer {layer} {label}")
                module.VT.weight.data.copy_(down.flatten(0, 1))
                for head in range(heads):
                    module.U[head].weight.data.copy_(up[head].T.contiguous())
    args.output.mkdir(parents=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="4GB")
    report = {
        "base_model": str(args.base_model), "factor_dir": str(args.factors),
        "factorization": metadata["factorization"], "group_size": 1,
        "key_rank": 96, "value_rank": 96,
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "dtype": str(next(model.parameters()).dtype),
    }
    (args.output / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
