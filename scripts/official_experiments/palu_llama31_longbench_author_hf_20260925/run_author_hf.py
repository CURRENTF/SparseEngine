#!/usr/bin/env python3
"""Run the Palu authors' HF model with the frozen LongBench baseline protocol."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = Path(__file__).resolve().parent
    sys.path.insert(0, str(Path(os.environ["PALU_BASELINE_RUNNER"]).resolve().parent))
    import run_hf

    run_hf.SOURCE_NAME["palu"] = "palu"

    def checked_source(method):
        if method != "palu":
            raise ValueError(method)
        upstream = Path(os.environ["PALU_SOURCE"]).resolve()
        commit = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True,
        ).strip()
        if commit != "bb22666e2ef96707e8dd21d93fc00146c2e0d615":
            raise RuntimeError(f"Unexpected Palu source commit: {commit}")
        patch = subprocess.check_output(
            ["git", "-C", str(upstream), "diff", "--binary"],
        )
        if hashlib.sha256(patch).hexdigest() != digest(source / "palu_compat.patch"):
            raise RuntimeError("Palu source patch differs from recorded patch")
        return upstream, commit, hashlib.sha256(patch).hexdigest()

    def load_model(args, upstream, settings, torch):
        sys.path.insert(0, str(upstream))
        from palu.model.svd_llama import PaluLlamaForCausalLM

        checkpoint = Path(settings["checkpoint"])
        model = PaluLlamaForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", low_cpu_mem_usage=True,
        ).to("cuda")
        expected_layers = model.config.num_hidden_layers
        from palu.model.modules.svd_linear import HeadwiseLowRankModule

        actual = sum(
            isinstance(layer.self_attn.k_proj, HeadwiseLowRankModule)
            and isinstance(layer.self_attn.v_proj, HeadwiseLowRankModule)
            for layer in model.model.layers
        )
        if actual != expected_layers:
            raise RuntimeError(f"Palu modules present in {actual}/{expected_layers} layers")
        run_hf.keep_only_last_prefill_logit(model)
        return model

    run_hf.checked_source = checked_source
    run_hf.load_model = load_model
    try:
        result = run_hf.main()
    finally:
        if "--output-dir" in sys.argv:
            output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
            manifest_path = output / "run_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                manifest["prefill_logits_last_token_only"] = True
                manifest["author_model_class"] = "palu.model.svd_llama.PaluLlamaForCausalLM"
                manifest["author_checkpoint"] = json.loads(
                    Path(sys.argv[sys.argv.index("--settings-json") + 1]).read_text()
                )["checkpoint"]
                manifest["kv_cache_semantics"] = "full reconstructed K/V from author projection modules"
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
