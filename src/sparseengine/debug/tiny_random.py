import copy
import json
import os
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn


TINY_RANDOM_ENV = "SPARSEENGINE_TINY_RANDOM"
TINY_RANDOM_CONFIG_ENV = "SPARSEENGINE_TINY_RANDOM_CONFIG"
TINY_RANDOM_SEED_ENV = "SPARSEENGINE_TINY_RANDOM_SEED"

SUPPORTED_OVERRIDES = frozenset(
    {
        "head_dim",
        "hidden_size",
        "intermediate_size",
        "max_position_embeddings",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "vocab_size",
    }
)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be an explicit boolean, got {raw!r}.")


def resolve_tiny_random_settings(
    *,
    enabled: bool,
    config_path: str | None,
    seed: int,
) -> tuple[bool, str | None, int]:
    resolved_enabled = env_flag(TINY_RANDOM_ENV, enabled)
    resolved_path = os.getenv(TINY_RANDOM_CONFIG_ENV, config_path)
    resolved_seed_raw = os.getenv(TINY_RANDOM_SEED_ENV)
    resolved_seed = int(resolved_seed_raw) if resolved_seed_raw is not None else int(seed)
    if resolved_seed < 0:
        raise ValueError(f"tiny_random_seed must be >= 0, got {resolved_seed}.")
    if resolved_enabled:
        if not resolved_path:
            raise ValueError(
                f"tiny random mode requires a JSON override file via "
                f"{TINY_RANDOM_CONFIG_ENV} or tiny_random_config."
            )
        resolved_path = os.path.abspath(os.path.expanduser(resolved_path))
        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(f"Tiny random config does not exist: {resolved_path}")
    return resolved_enabled, resolved_path, resolved_seed


def load_tiny_random_overrides(path: str) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"Tiny random config must contain one JSON object, got {type(raw).__name__}.")
    unknown = sorted(set(raw) - SUPPORTED_OVERRIDES)
    if unknown:
        raise ValueError(
            f"Unsupported tiny random config keys: {unknown}. "
            f"Allowed keys: {sorted(SUPPORTED_OVERRIDES)}."
        )
    required = {"num_hidden_layers", "hidden_size", "intermediate_size"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"Tiny random config is missing required shrink overrides: {missing}.")
    overrides: dict[str, int] = {}
    for name, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Tiny random override {name} must be a positive integer, got {value!r}.")
        overrides[name] = int(value)
    return overrides


def apply_tiny_random_overrides(
    hf_config: Any,
    path: str,
    *,
    validate_standard_head_shape: bool = True,
) -> dict[str, int]:
    overrides = load_tiny_random_overrides(path)
    original_values = {
        name: int(getattr(hf_config, name))
        for name in overrides
        if hasattr(hf_config, name)
    }
    for name, value in overrides.items():
        if not hasattr(hf_config, name):
            raise ValueError(
                f"Tiny random override {name!r} is not present on "
                f"{type(hf_config).__name__}."
            )
        setattr(hf_config, name, value)
    for name in ("num_hidden_layers", "hidden_size", "intermediate_size"):
        if overrides[name] > original_values[name]:
            raise ValueError(
                f"Tiny random override {name} must not enlarge the source model: "
                f"{overrides[name]} > {original_values[name]}."
            )

    layer_types = getattr(hf_config, "layer_types", None)
    if layer_types is not None:
        num_layers = int(hf_config.num_hidden_layers)
        if len(layer_types) < num_layers:
            raise ValueError(
                "Tiny random config cannot expand layer_types: "
                f"requested={num_layers}, available={len(layer_types)}."
            )
        hf_config.layer_types = list(layer_types[:num_layers])

    mlp_layer_types = getattr(hf_config, "mlp_layer_types", None)
    if mlp_layer_types is not None:
        num_layers = int(hf_config.num_hidden_layers)
        if len(mlp_layer_types) < num_layers:
            raise ValueError(
                "Tiny random config cannot expand mlp_layer_types: "
                f"requested={num_layers}, available={len(mlp_layer_types)}."
            )
        hf_config.mlp_layer_types = list(mlp_layer_types[:num_layers])

    hidden_size = int(hf_config.hidden_size)
    num_heads = int(hf_config.num_attention_heads)
    num_kv_heads = int(hf_config.num_key_value_heads)
    head_dim = int(getattr(hf_config, "head_dim", hidden_size // num_heads))
    if validate_standard_head_shape and hidden_size != num_heads * head_dim:
        raise ValueError(
            "Tiny random config requires hidden_size == num_attention_heads * head_dim, "
            f"got {hidden_size} != {num_heads} * {head_dim}."
        )
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            "Tiny random config requires num_attention_heads divisible by "
            f"num_key_value_heads, got {num_heads} % {num_kv_heads}."
        )
    return overrides


@contextmanager
def _cpu_default_dtype(dtype: torch.dtype) -> Iterator[None]:
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device("cpu"):
            yield
    finally:
        torch.set_default_dtype(previous_dtype)


def build_tiny_random_hf_model(
    hf_config: Any,
    *,
    seed: int,
) -> nn.Module:
    from transformers import AutoModelForCausalLM

    config = copy.deepcopy(hf_config)
    dtype = config.dtype
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError(f"Tiny random mode requires a floating dtype, got {dtype!r}.")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        with _cpu_default_dtype(dtype):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.eval()
    return model


def initialize_sparse_model(
    model: nn.Module,
    hf_config: Any,
    *,
    seed: int,
    quantized: bool = False,
) -> None:
    from sparseengine.utils.loader import (
        _target_weight_name_for_model,
        default_weight_loader,
    )

    if quantized:
        _initialize_quantized_sparse_model(model, seed=seed)
        print(
            "Initialized quantized model weights from deterministic tiny random "
            f"seed={int(seed)} without reading checkpoint tensors"
        )
        return

    reference = build_tiny_random_hf_model(hf_config, seed=seed)
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    loaded_count = 0
    loaded_parameter_names: set[str] = set()
    try:
        reference_state = reference.state_dict()
        reference_items = reference_state.items()
        reference_adapter = getattr(model, "iter_tiny_reference_weights", None)
        if callable(reference_adapter):
            reference_items = reference_adapter(reference_state)
        for source_weight_name, loaded_weight in reference_items:
            param_name = _target_weight_name_for_model(model, source_weight_name)
            if param_name is None:
                continue
            special_loader = getattr(model, "load_special_weight", None)
            special_suffixes = tuple(getattr(model, "special_weight_loaders", ()))
            if callable(special_loader) and param_name.endswith(special_suffixes):
                special_count = int(special_loader(param_name, loaded_weight, None))
                if special_count < 0:
                    raise ValueError(
                        f"load_special_weight() returned a negative count for {param_name!r}."
                    )
                if special_count:
                    loaded_count += special_count
                    continue
            for source_name, (packed_name, shard_id) in packed_modules_mapping.items():
                if source_name in param_name:
                    packed_param_name = param_name.replace(source_name, packed_name)
                    param = model.get_parameter(packed_param_name)
                    weight_loader = getattr(param, "weight_loader")
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_parameter_names.add(packed_param_name)
                    loaded_count += 1
                    break
            else:
                param = model.get_parameter(param_name)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_parameter_names.add(param_name)
                loaded_count += 1
    finally:
        del reference

    if loaded_count <= 0:
        raise RuntimeError("Tiny random initialization did not load any parameters.")
    strict_validator = getattr(model, "validate_loaded_weights", None)
    if callable(strict_validator):
        strict_validator(loaded_parameter_names)
    print(
        f"Initialized {loaded_count} model weights from deterministic tiny random "
        f"seed={int(seed)} without reading checkpoint tensors"
    )


@torch.inference_mode()
def _initialize_quantized_sparse_model(model: nn.Module, *, seed: int) -> None:
    generators: dict[torch.device, torch.Generator] = {}

    def generator_for(device: torch.device) -> torch.Generator:
        generator = generators.get(device)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            generators[device] = generator
        return generator

    initialized = 0
    for parameter in model.parameters():
        if not parameter.dtype.is_floating_point:
            parameter.zero_()
            initialized += parameter.numel()
            continue
        values = torch.empty(
            parameter.shape,
            dtype=torch.float32,
            device=parameter.device,
        )
        values.normal_(mean=0.0, std=0.02, generator=generator_for(parameter.device))
        parameter.copy_(values.to(parameter.dtype))
        initialized += parameter.numel()

    for name, buffer in model.named_buffers():
        if name.endswith("weight_scale_inv"):
            buffer.fill_(1.0)

    for module in model.modules():
        if hasattr(module, "_quantized_weight_loaded"):
            module._quantized_weight_loaded = True
            module._quantized_loaded_ranges = [(0, int(module.weight.shape[0]))]

    if initialized <= 0:
        raise RuntimeError("Quantized tiny random initialization found no parameters.")
