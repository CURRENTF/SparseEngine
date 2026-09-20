from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from benchmark.ruler_vt.tasks import SUPPORTED_TASKS, resolve_task_config
from sparseengine.method_registry import (
    PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    get_default_prefill_schedule_policy,
)


REQUIRED_METHODS = {
    "vanilla",
    "streamingllm",
    "snapkv",
    "h2o",
    "pyramidkv",
    "omnikv",
    "quest",
    "rkv",
    "skipkv",
    "deltakv",
    "deltakv-less-memory",
    "deltakv-less-memory-cudagraph",
}

REQUIRED_MODELS = {
    "qwen25_7b",
    "qwen25_32b",
    "qwen3_4b",
    "llama31_8b",
}

REQUIRED_ARTIFACTS = [
    "resolved_manifest.json",
    "raw_outputs.jsonl",
    "parsed_outputs.jsonl",
    "sample_results.jsonl",
    "metrics.json",
    "perf.jsonl",
    "memory.json",
    "stress.json",
    "stress_v2.json",
    "scbench.json",
    "ruler.json",
    "longbench_v2.json",
    "grade_summary.json",
]
OMNIKV_REQUIRED_BENCHMARK_PARAMS = {
    "decode_keep_tokens",
    "engine_prefill_chunk_size",
    "full_attention_layers",
    "pool_kernel_size",
    "recent_keep_tokens",
    "sink_keep_tokens",
}


class ManifestError(ValueError):
    pass


def _validate_prefill_controls(
    *,
    method_id: str,
    sparse_method: str,
    config: dict[str, Any],
    config_label: str,
) -> None:
    try:
        policy = get_default_prefill_schedule_policy(sparse_method)
    except ValueError as exc:
        raise ManifestError(
            f"method {method_id!r} has unsupported sparse_method={sparse_method!r}."
        ) from exc

    if policy == PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH:
        boundary = config.get("long_prefill_offload_threshold")
        if (
            not isinstance(boundary, int)
            or isinstance(boundary, bool)
            or boundary <= 0
        ):
            raise ManifestError(
                f"{config_label} long_prefill_offload_threshold must be a positive integer."
            )
        chunk_size = config.get("engine_prefill_chunk_size", boundary)
        if (
            not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or chunk_size <= 0
        ):
            raise ManifestError(
                f"{config_label} engine_prefill_chunk_size must be a positive integer."
            )
        if chunk_size > boundary:
            raise ManifestError(
                f"{config_label} engine_prefill_chunk_size must be <= "
                "long_prefill_offload_threshold."
            )
        return
    else:
        if "long_prefill_offload_threshold" in config:
            raise ManifestError(
                f"{config_label} uses {policy}; declare engine_prefill_chunk_size "
                "and remove long_prefill_offload_threshold."
            )
    value = config.get("engine_prefill_chunk_size")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ManifestError(
            f"{config_label} engine_prefill_chunk_size must be a positive integer."
        )


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(path) if path else Path(__file__).with_name("manifest.json")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise ManifestError("manifest must be a JSON object.")
    for key in (
        "models",
        "methods",
        "quality",
        "longbench_v2",
        "ruler",
        "performance",
        "stress",
        "stress_v2",
        "scbench",
        "outputs",
    ):
        if key not in manifest:
            raise ManifestError(f"manifest is missing required key: {key}")

    models = manifest["models"]
    methods = manifest["methods"]
    if not isinstance(models, dict) or not isinstance(methods, dict):
        raise ManifestError("manifest models and methods must be JSON objects.")

    missing_models = sorted(REQUIRED_MODELS - set(models))
    missing_methods = sorted(REQUIRED_METHODS - set(methods))
    if missing_models:
        raise ManifestError(f"manifest is missing required models: {missing_models}")
    if missing_methods:
        raise ManifestError(f"manifest is missing required methods: {missing_methods}")

    quality = manifest["quality"]
    if not isinstance(quality, dict):
        raise ManifestError("manifest quality must be a JSON object.")
    minimum_vanilla_score = quality.get("minimum_vanilla_score")
    if not isinstance(minimum_vanilla_score, (int, float)) or minimum_vanilla_score <= 0:
        raise ManifestError("quality minimum_vanilla_score must be a positive number.")

    longbench_v2 = manifest["longbench_v2"]
    if not isinstance(longbench_v2, dict):
        raise ManifestError("manifest longbench_v2 must be a JSON object.")
    data_path_env = longbench_v2.get("data_path_env")
    if not isinstance(data_path_env, str) or not data_path_env:
        raise ManifestError("longbench_v2 data_path_env must be a non-empty string.")
    for score_key in ("minimum_vanilla_score", "maximum_score_loss"):
        value = longbench_v2.get(score_key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 < float(value) <= 100.0
        ):
            raise ManifestError(f"longbench_v2 {score_key} must be in (0, 100].")
    for int_key in ("max_model_len", "max_new_tokens", "batch_size", "seed"):
        value = longbench_v2.get(int_key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ManifestError(f"longbench_v2 {int_key} must be a positive integer.")
    for float_key in ("temperature", "top_p"):
        value = longbench_v2.get(float_key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ManifestError(
                f"longbench_v2 {float_key} must be a finite non-negative number."
            )
    top_k = longbench_v2.get("top_k")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
        raise ManifestError("longbench_v2 top_k must be a non-negative integer.")
    token_buckets = longbench_v2.get("token_buckets")
    try:
        from benchmark.long_bench_v2.contracts import parse_token_buckets

        resolved_buckets = parse_token_buckets(token_buckets)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"invalid longbench_v2 token_buckets: {exc}") from exc
    prompt_budget = int(longbench_v2["max_model_len"]) - int(
        longbench_v2["max_new_tokens"]
    )
    if prompt_budget <= 0:
        raise ManifestError("longbench_v2 max_new_tokens leaves no prompt budget.")
    oversized_buckets = [
        bucket.name
        for bucket in resolved_buckets
        if bucket.max_prompt_tokens > prompt_budget
    ]
    if oversized_buckets:
        raise ManifestError(
            "longbench_v2 token buckets exceed the untruncated prompt budget: "
            f"{oversized_buckets}."
        )

    ruler = manifest["ruler"]
    if not isinstance(ruler, dict):
        raise ManifestError("manifest ruler must be a JSON object.")
    ruler_tasks = ruler.get("tasks")
    if (
        not isinstance(ruler_tasks, list)
        or not ruler_tasks
        or any(not isinstance(task, str) for task in ruler_tasks)
        or len(set(ruler_tasks)) != len(ruler_tasks)
    ):
        raise ManifestError("ruler tasks must be a non-empty list of unique strings.")
    unsupported_tasks = sorted(set(ruler_tasks) - set(SUPPORTED_TASKS))
    if unsupported_tasks:
        raise ManifestError(f"ruler tasks are unsupported: {unsupported_tasks}.")
    task_configs = ruler.get("task_configs")
    if not isinstance(task_configs, dict):
        raise ManifestError("ruler task_configs must be a JSON object.")
    missing_task_configs = sorted(set(ruler_tasks) - set(task_configs))
    if missing_task_configs:
        raise ManifestError(
            f"ruler task_configs are missing selected tasks: {missing_task_configs}."
        )
    for task in ruler_tasks:
        task_config = task_configs[task]
        if not isinstance(task_config, dict):
            raise ManifestError(f"ruler task_configs.{task} must be a JSON object.")
        for key in ("tokens_to_generate", "max_new_tokens"):
            value = task_config.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ManifestError(
                    f"ruler task_configs.{task}.{key} must be a positive integer."
                )
        try:
            resolve_task_config(task, task_config)
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"invalid ruler task_configs.{task}: {exc}") from exc
    context_lengths = ruler.get("context_lengths")
    if (
        not isinstance(context_lengths, list)
        or not context_lengths
        or any(
            not isinstance(length, int)
            or isinstance(length, bool)
            or length <= 0
            for length in context_lengths
        )
        or len(set(context_lengths)) != len(context_lengths)
    ):
        raise ManifestError(
            "ruler context_lengths must be a non-empty list of unique positive integers."
        )
    for int_key in (
        "samples_per_length",
        "batch_size",
        "worker_world_size",
    ):
        value = ruler.get(int_key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ManifestError(f"ruler {int_key} must be a positive integer.")
    for score_key in ("minimum_vanilla_score", "maximum_score_loss"):
        value = ruler.get(score_key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 < float(value) <= 100.0
        ):
            raise ManifestError(f"ruler {score_key} must be in (0, 100].")
    minimum_context_utilization = ruler.get("minimum_context_utilization")
    if (
        not isinstance(minimum_context_utilization, (int, float))
        or isinstance(minimum_context_utilization, bool)
        or not math.isfinite(float(minimum_context_utilization))
        or not 0.0 < float(minimum_context_utilization) <= 1.0
    ):
        raise ManifestError(
            "ruler minimum_context_utilization must be a finite number in (0, 1]."
        )
    temperature = ruler.get("temperature")
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(float(temperature))
        or float(temperature) < 0.0
    ):
        raise ManifestError("ruler temperature must be a finite non-negative number.")
    max_model_len = ruler.get("max_model_len")
    if max_model_len is not None and (
        not isinstance(max_model_len, int)
        or isinstance(max_model_len, bool)
        or max_model_len < max(context_lengths)
    ):
        raise ManifestError(
            "ruler max_model_len must be an integer covering the largest context length."
        )

    for model_id, model in models.items():
        if "model_path_env" not in model:
            raise ManifestError(f"model {model_id!r} is missing model_path_env.")
        if "tokenizer_path_env" not in model:
            raise ManifestError(f"model {model_id!r} is missing tokenizer_path_env.")
        compressor_env = model.get("deltakv_checkpoint_path_env")
        if compressor_env is not None and not isinstance(compressor_env, str):
            raise ManifestError(
                f"model {model_id!r} deltakv_checkpoint_path_env must be a string."
            )
        mixed_attention = model.get("mixed_attention", False)
        if not isinstance(mixed_attention, bool):
            raise ManifestError(f"model {model_id!r} mixed_attention must be a boolean.")

    for method_id, method in methods.items():
        if "sparse_method" not in method:
            raise ManifestError(f"method {method_id!r} is missing sparse_method.")
        if "config" not in method or not isinstance(method["config"], dict):
            raise ManifestError(f"method {method_id!r} must define config object.")
        _validate_prefill_controls(
            method_id=method_id,
            sparse_method=method["sparse_method"],
            config=method["config"],
            config_label=f"method {method_id!r} config",
        )
        if method["sparse_method"] == "omnikv":
            validate_omnikv_benchmark_config(method["config"])
        model_configs = method.get("model_configs")
        if model_configs is not None:
            if not isinstance(model_configs, dict):
                raise ManifestError(f"method {method_id!r} model_configs must be a JSON object.")
            unknown_model_configs = sorted(set(model_configs) - set(models))
            if unknown_model_configs:
                raise ManifestError(
                    f"method {method_id!r} model_configs references unknown models: {unknown_model_configs}"
                )
            for model_id, override in model_configs.items():
                if not isinstance(override, dict):
                    raise ManifestError(
                        f"method {method_id!r} model_configs[{model_id!r}] must be a JSON object."
                    )
                merged_config = {**method["config"], **override}
                _validate_prefill_controls(
                    method_id=method_id,
                    sparse_method=method["sparse_method"],
                    config=merged_config,
                    config_label=f"method {method_id!r} model_configs[{model_id!r}]",
                )
                if method["sparse_method"] == "omnikv":
                    validate_omnikv_benchmark_config(merged_config)
        supported_families = method.get("supported_model_families")
        if supported_families is not None:
            if (
                not isinstance(supported_families, list)
                or not supported_families
                or any(not isinstance(family, str) or not family for family in supported_families)
            ):
                raise ManifestError(
                    f"method {method_id!r} supported_model_families must be a non-empty string list."
                )
        supported_tp_sizes = method.get("supported_tensor_parallel_sizes")
        if supported_tp_sizes is not None:
            if (
                not isinstance(supported_tp_sizes, list)
                or not supported_tp_sizes
                or any(not isinstance(size, int) or size <= 0 for size in supported_tp_sizes)
            ):
                raise ManifestError(
                    f"method {method_id!r} supported_tensor_parallel_sizes must be a non-empty positive integer list."
                )
        for bool_key in ("requires_compressor",):
            if bool_key not in method or not isinstance(method[bool_key], bool):
                raise ManifestError(f"method {method_id!r} must define boolean {bool_key}.")
        compressor_env = method.get("deltakv_checkpoint_path_env")
        if compressor_env is not None and not isinstance(compressor_env, str):
            raise ManifestError(
                f"method {method_id!r} deltakv_checkpoint_path_env must be a string."
            )
        performance_policy = method.get("performance")
        if performance_policy is not None:
            if not isinstance(performance_policy, dict):
                raise ManifestError(f"method {method_id!r} performance must be a JSON object.")
            unknown_performance_keys = sorted(
                set(performance_policy) - {"minimum_prefill_speedup"}
            )
            if unknown_performance_keys:
                raise ManifestError(
                    f"method {method_id!r} performance has unknown keys: "
                    f"{unknown_performance_keys}"
                )
            minimum_prefill_speedup = performance_policy.get("minimum_prefill_speedup")
            if (
                minimum_prefill_speedup is not None
                and (
                    not isinstance(minimum_prefill_speedup, (int, float))
                    or isinstance(minimum_prefill_speedup, bool)
                    or not math.isfinite(float(minimum_prefill_speedup))
                    or minimum_prefill_speedup <= 0
                )
            ):
                raise ManifestError(
                    f"method {method_id!r} performance minimum_prefill_speedup "
                    "must be a positive number."
                )
        if method["requires_compressor"] and "deltakv_checkpoint_path_env" not in method:
            model_specific = [
                model_id
                for model_id, model in models.items()
                if model.get("deltakv_checkpoint_path_env")
            ]
            if not model_specific:
                raise ManifestError(
                    f"method {method_id!r} requires compressor but no model or method "
                    "defines deltakv_checkpoint_path_env."
                )

    outputs = manifest["outputs"]
    missing_artifacts = sorted(set(REQUIRED_ARTIFACTS) - set(outputs))
    if missing_artifacts:
        raise ManifestError(f"manifest outputs missing required artifacts: {missing_artifacts}")

    scbench = manifest["scbench"]
    if not isinstance(scbench, dict):
        raise ManifestError("manifest scbench must be a JSON object.")
    if scbench.get("model") not in models:
        raise ManifestError(f"scbench model must reference a known model, got {scbench.get('model')!r}.")
    scbench_methods = scbench.get("methods")
    if not isinstance(scbench_methods, list) or not scbench_methods:
        raise ManifestError("scbench methods must be a non-empty list.")
    unknown_scbench_methods = sorted(set(scbench_methods) - set(methods))
    if unknown_scbench_methods:
        raise ManifestError(f"scbench methods reference unknown methods: {unknown_scbench_methods}")
    scbench_tasks = scbench.get("tasks")
    if not isinstance(scbench_tasks, list) or not scbench_tasks:
        raise ManifestError("scbench tasks must be a non-empty list.")
    for int_key in ("num_eval_examples", "max_turns", "max_seq_length", "batch_size"):
        value = scbench.get(int_key)
        if not isinstance(value, int) or value <= 0:
            raise ManifestError(f"scbench {int_key} must be a positive integer.")


def select_entries(manifest: dict[str, Any], models: list[str] | None, methods: list[str] | None):
    model_ids = models or list(manifest["models"])
    method_ids = methods or list(manifest["methods"])
    unknown_models = sorted(set(model_ids) - set(manifest["models"]))
    unknown_methods = sorted(set(method_ids) - set(manifest["methods"]))
    if unknown_models:
        raise ManifestError(f"Unknown model ids: {unknown_models}")
    if unknown_methods:
        raise ManifestError(f"Unknown method ids: {unknown_methods}")
    return model_ids, method_ids


def resolve_method_config(
    method: dict[str, Any],
    *,
    model_id: str | None = None,
    require_model_config: bool = False,
    include_method: bool = True,
) -> dict[str, Any]:
    """Merge one method's canonical config with its model-specific override."""
    sparse_method = method.get("sparse_method")
    if not isinstance(sparse_method, str):
        raise ManifestError("method is missing string sparse_method.")
    base = method.get("config")
    if not isinstance(base, dict):
        raise ManifestError(f"method {sparse_method!r} is missing config object.")

    model_configs = method.get("model_configs") or {}
    if not isinstance(model_configs, dict):
        raise ManifestError(
            f"method {sparse_method!r} model_configs must be a JSON object."
        )
    if require_model_config and (not model_id or model_id not in model_configs):
        available = sorted(model_configs)
        raise ManifestError(
            f"method {sparse_method!r} requires a calibrated model-specific config; "
            f"got model_id={model_id!r}, available={available}."
        )

    config = dict(base)
    if model_id and model_id in model_configs:
        override = model_configs[model_id]
        if not isinstance(override, dict):
            raise ManifestError(
                f"method {sparse_method!r} model_configs[{model_id!r}] must be a JSON object."
            )
        config.update(override)
    if include_method:
        config["sparse_method"] = sparse_method
    return config


def validate_omnikv_benchmark_config(
    config: dict[str, Any],
    *,
    allow_single_full_layer: bool = False,
) -> list[int]:
    """Validate the explicit, accuracy-affecting OmniKV benchmark contract."""
    missing = sorted(OMNIKV_REQUIRED_BENCHMARK_PARAMS - set(config))
    if missing:
        raise ManifestError(
            "OmniKV benchmarks require an explicit calibrated method config; "
            f"missing parameters={missing}. Resolve the model-specific config from "
            "benchmark/sparseengine_regression/manifest.json."
        )

    value = config["full_attention_layers"]
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        layers = [int(part) for part in parts]
    elif isinstance(value, (list, tuple)):
        if any(isinstance(layer, bool) for layer in value):
            raise ManifestError("full_attention_layers cannot contain booleans.")
        layers = [int(layer) for layer in value]
    else:
        raise ManifestError(
            "full_attention_layers must be a comma-separated string or integer list."
        )
    if not layers:
        raise ManifestError("full_attention_layers must contain at least one layer.")
    if any(layer < 0 for layer in layers):
        raise ManifestError("full_attention_layers cannot contain negative layers.")
    if layers != sorted(set(layers)):
        raise ManifestError("full_attention_layers must be unique and sorted.")
    if len(layers) == 1 and not allow_single_full_layer:
        raise ManifestError(
            "OmniKV benchmarks refuse a single full-attention layer by default because "
            "one observation layer would drive every later sparse layer. Pass a calibrated "
            "multi-layer config, or explicitly enable the single-layer ablation."
        )
    return layers


def runtime_support_reason(
    manifest: dict[str, Any],
    model_id: str,
    method_id: str,
    *,
    tensor_parallel_sizes: list[int] | tuple[int, ...],
) -> str | None:
    """Return why a model/method pair is outside the declared runtime matrix."""
    model = manifest["models"][model_id]
    method = manifest["methods"][method_id]
    supported_families = method.get("supported_model_families")
    model_family = str(model.get("family") or "")
    if supported_families is not None and model_family not in supported_families:
        return (
            f"method supports model families {supported_families}, "
            f"got model={model_id!r} family={model_family!r}"
        )
    supported_tp_sizes = method.get("supported_tensor_parallel_sizes")
    unsupported_tp_sizes = sorted(
        {
            int(size)
            for size in tensor_parallel_sizes
            if supported_tp_sizes is not None and int(size) not in supported_tp_sizes
        }
    )
    if unsupported_tp_sizes:
        return (
            f"method supports tensor_parallel_size values {supported_tp_sizes}, "
            f"got {unsupported_tp_sizes}"
        )
    return None


def resolve_manifest_paths(manifest: dict[str, Any]) -> dict[str, Any]:
    resolved = json.loads(json.dumps(manifest))
    for model in resolved["models"].values():
        model["model_path"] = os.getenv(model["model_path_env"])
        tokenizer_env = model["tokenizer_path_env"]
        model["tokenizer_path"] = os.getenv(tokenizer_env) or model["model_path"]
        checkpoint_env = model.get("deltakv_checkpoint_path_env")
        model["deltakv_checkpoint_path"] = (
            os.getenv(checkpoint_env) if checkpoint_env else None
        )
    for method in resolved["methods"].values():
        env_key = method.get("deltakv_checkpoint_path_env")
        method["deltakv_checkpoint_path"] = os.getenv(env_key) if env_key else None
    data_env = resolved["longbench_v2"]["data_path_env"]
    resolved["longbench_v2"]["data_path"] = os.getenv(data_env)
    return resolved


def deltakv_checkpoint_path_for(
    model: dict[str, Any], method: dict[str, Any]
) -> str | None:
    if not method.get("requires_compressor"):
        return None
    if model.get("deltakv_checkpoint_path_env"):
        return model.get("deltakv_checkpoint_path")
    return model.get("deltakv_checkpoint_path") or method.get(
        "deltakv_checkpoint_path"
    )


def deltakv_checkpoint_env_for(model: dict[str, Any], method: dict[str, Any]) -> str:
    return (
        model.get("deltakv_checkpoint_path_env")
        or method.get("deltakv_checkpoint_path_env")
        or "deltakv_checkpoint_path_env"
    )


def missing_runtime_inputs(resolved: dict[str, Any], model_id: str, method_id: str) -> list[str]:
    missing: list[str] = []
    model = resolved["models"][model_id]
    method = resolved["methods"][method_id]
    if not model.get("model_path"):
        missing.append(model["model_path_env"])
    elif not Path(model["model_path"]).exists():
        missing.append(f"{model['model_path_env']}={model['model_path']}")
    tokenizer_path = model.get("tokenizer_path")
    if tokenizer_path and not Path(tokenizer_path).exists():
        missing.append(f"{model['tokenizer_path_env']}={tokenizer_path}")
    if method.get("requires_compressor"):
        checkpoint_path = deltakv_checkpoint_path_for(model, method)
        checkpoint_env = deltakv_checkpoint_env_for(model, method)
        if not checkpoint_path:
            missing.append(checkpoint_env)
        elif not Path(checkpoint_path).exists():
            missing.append(f"{checkpoint_env}={checkpoint_path}")
    return missing


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve one canonical SparseEngine benchmark method config."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--method", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--require-model-config", action="store_true")
    parser.add_argument("--overrides-json", default="{}")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    manifest = load_manifest(args.manifest)
    if args.method not in manifest["methods"]:
        raise ManifestError(
            f"Unknown method id {args.method!r}; available={sorted(manifest['methods'])}."
        )
    try:
        overrides = json.loads(args.overrides_json)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"--overrides-json is invalid JSON: {exc}") from exc
    if not isinstance(overrides, dict):
        raise ManifestError("--overrides-json must decode to a JSON object.")

    config = resolve_method_config(
        manifest["methods"][args.method],
        model_id=args.model_id,
        require_model_config=args.require_model_config,
    )
    config.update(overrides)
    if manifest["methods"][args.method]["sparse_method"] == "omnikv":
        validate_omnikv_benchmark_config(config)
    print(json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
