import json
import os
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from glob import glob

import torch
from safetensors import safe_open
from torch import nn
from tqdm.auto import tqdm

from sparseengine.utils.log import logger
from sparseengine.utils.weight_target import WeightTarget
from sparseengine.quantization.fp8 import expand_fp8_tensor_scale


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _packed_mapping_for(model: nn.Module, target_parameter_name: str) -> dict:
    excluded = tuple(getattr(model, "packed_modules_excluded_prefixes", ()))
    return (
        {}
        if target_parameter_name.startswith(excluded)
        else getattr(model, "packed_modules_mapping", {})
    )


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple[int, ...]
    dtype: str


@dataclass
class SafetensorsShard:
    metadata: dict[str, TensorMetadata]
    tensors: dict[str, torch.Tensor]


def _canonical_block_scale_name(model: nn.Module | None, name: str) -> str:
    """Normalize declared checkpoint block-scale aliases at the I/O boundary."""
    for suffix in tuple(getattr(model, "checkpoint_block_scale_suffixes", ())):
        if not suffix.startswith(".") or suffix in {".weight", ".weight_scale"}:
            raise ValueError(f"Invalid checkpoint block-scale suffix {suffix!r}.")
        if name.endswith(suffix):
            return name[:-len(suffix)] + ".weight_scale_inv"
    return name


def _normalize_block_scale_names(shard: SafetensorsShard, model: nn.Module) -> None:
    # Rank-local checkpoints are read without slicing/model filtering.
    for name in list(shard.metadata):
        canonical = _canonical_block_scale_name(model, name)
        if canonical == name:
            continue
        if canonical in shard.metadata:
            raise ValueError(f"Checkpoint contains duplicate block scales for {canonical!r}.")
        shard.metadata[canonical] = shard.metadata.pop(name)
        if name in shard.tensors:
            shard.tensors[canonical] = shard.tensors.pop(name)


def _resolve_weight_target(
    model: nn.Module,
    target_parameter_name: str,
) -> WeightTarget | None:
    special_loader = getattr(model, "load_special_weight", None)
    special_suffixes = tuple(getattr(model, "special_weight_loaders", ()))
    if callable(special_loader) and target_parameter_name.endswith(special_suffixes):
        resolver = getattr(model, "resolve_special_weight", None)
        if not callable(resolver):
            return None
        target = resolver(target_parameter_name)
        if target is None:
            return None
        if not isinstance(target, WeightTarget):
            raise TypeError(
                "resolve_special_weight() must return WeightTarget or None, "
                f"got {type(target).__name__}."
            )
        return target

    loaded_shard_id = None
    packed_modules_mapping = _packed_mapping_for(model, target_parameter_name)
    for source_fragment, (target_fragment, shard_id) in packed_modules_mapping.items():
        if source_fragment in target_parameter_name:
            target_parameter_name = target_parameter_name.replace(
                source_fragment, target_fragment
            )
            loaded_shard_id = shard_id
            break

    module = _module_for_parameter(model, target_parameter_name)
    return WeightTarget(module, loaded_shard_id)


def _rank_local_slice_for_tensor(
    model: nn.Module,
    source_weight_name: str,
    source_shape: tuple[int, ...],
) -> tuple[slice, ...] | None:
    is_tensor_scale = source_weight_name.endswith(".weight_scale")
    if is_tensor_scale:
        # A checkpoint scalar is replicated even when its weight is TP-sharded.
        return None
    is_scale = source_weight_name.endswith(".weight_scale_inv")
    source_parameter_name = (
        source_weight_name[: -len(".weight_scale_inv")] + ".weight"
        if is_scale
        else source_weight_name
    )
    target_parameter_name = _target_weight_name_for_model(
        model, source_parameter_name
    )
    if target_parameter_name is None:
        return None
    target = _resolve_weight_target(model, target_parameter_name)
    if target is None:
        return None
    slicer = getattr(target.module, "rank_local_weight_slice", None)
    if not callable(slicer):
        return None
    return slicer(
        source_shape,
        loaded_shard_id=target.shard_id,
        is_scale=is_scale,
    )


def _read_safetensors_shard(
    path: str,
    model: nn.Module | None = None,
) -> SafetensorsShard:
    with safe_open(path, "pt", "cpu") as handle:
        metadata: dict[str, TensorMetadata] = {}
        tensors: dict[str, torch.Tensor] = {}
        for source_key in handle.keys():
            key = _canonical_block_scale_name(model, source_key)
            if key in metadata:
                raise ValueError(f"Checkpoint contains duplicate tensor or block scale {key!r}.")
            tensor_slice = handle.get_slice(source_key)
            source_shape = tuple(int(dim) for dim in tensor_slice.get_shape())
            metadata[key] = TensorMetadata(
                shape=source_shape,
                dtype=str(tensor_slice.get_dtype()),
            )
            if model is None:
                tensors[key] = handle.get_tensor(source_key)
                continue

            scale_suffix = (
                ".weight_scale" if key.endswith(".weight_scale") else ".weight_scale_inv"
            )
            is_scale = key.endswith(scale_suffix)
            source_parameter_name = (
                key[: -len(scale_suffix)] + ".weight"
                if is_scale
                else key
            )
            if _target_weight_name_for_model(model, source_parameter_name) is None:
                continue
            rank_slice = _rank_local_slice_for_tensor(
                model,
                key,
                source_shape,
            )
            tensors[key] = (
                tensor_slice[rank_slice]
                if rank_slice is not None
                else handle.get_tensor(source_key)
            )
        for key in list(metadata):
            if not key.endswith(".weight_scale"):
                continue
            weight_key = key[: -len(".weight_scale")] + ".weight"
            block_key = _scale_key_for_weight_key(weight_key)
            if block_key in metadata:
                raise ValueError(f"Checkpoint contains both tensor and block scales for {weight_key}.")
            metadata[block_key] = metadata.pop(key)
            if key not in tensors:
                continue
            if weight_key not in tensors:
                raise ValueError(f"FP8 tensor scale {key} has no weight in its shard.")
            # Broadcast the scalar after rank-local slicing. This preserves every
            # checkpoint weight value; no weight requantization is performed.
            tensors[block_key] = expand_fp8_tensor_scale(tensors[weight_key], tensors.pop(key))
        return SafetensorsShard(metadata=metadata, tensors=tensors)


def _iter_safetensors_shards(
    files: list[str],
    *,
    num_threads: int,
    model: nn.Module | None = None,
):
    """Prefetch a bounded number of shards while preserving checkpoint order."""
    if num_threads == 1 or len(files) == 1:
        for file in files:
            yield file, _read_safetensors_shard(file, model)
        return

    file_iter = iter(files)
    pending: deque[tuple[str, Future[SafetensorsShard]]] = deque()
    with ThreadPoolExecutor(max_workers=min(num_threads, len(files))) as executor:
        for _ in range(min(num_threads, len(files))):
            file = next(file_iter)
            pending.append(
                (file, executor.submit(_read_safetensors_shard, file, model))
            )

        while pending:
            file, future = pending.popleft()
            tensors = future.result()
            next_file = next(file_iter, None)
            if next_file is not None:
                pending.append(
                    (
                        next_file,
                        executor.submit(_read_safetensors_shard, next_file, model),
                    )
                )
            yield file, tensors


def _module_for_parameter(model: nn.Module, param_name: str) -> nn.Module:
    module_name, sep, _ = param_name.rpartition(".")
    if not sep:
        return model
    return model.get_submodule(module_name)


def _scale_key_for_weight_key(weight_key: str) -> str:
    if not weight_key.endswith(".weight"):
        raise ValueError(f"Expected a weight key ending in '.weight', got {weight_key!r}.")
    return weight_key[: -len(".weight")] + ".weight_scale_inv"


def _target_weight_name_for_model(model: nn.Module, source_weight_name: str) -> str | None:
    ignored_prefixes = tuple(getattr(model, "ignored_weight_prefixes", ()))
    if source_weight_name.startswith(ignored_prefixes):
        return None
    mapper = getattr(model, "map_weight_name", None)
    target_weight_name = mapper(source_weight_name) if callable(mapper) else source_weight_name
    if target_weight_name is not None and not isinstance(target_weight_name, str):
        raise TypeError(
            f"map_weight_name() must return str or None, got {type(target_weight_name).__name__}."
        )
    return target_weight_name


def _load_grouped_quantized_weight(
    *,
    model: nn.Module,
    module: nn.Module,
    source_weight_name: str,
    target_weight_name: str,
    loaded_weight: torch.Tensor,
    loaded_scale: torch.Tensor | None,
    loaded_shard_id=None,
) -> bool:
    del model
    if not source_weight_name.endswith(".weight"):
        return False
    has_quantized_loader = hasattr(module, "load_quantized_weight")
    if loaded_scale is None:
        if bool(getattr(module, "quantized", False)):
            raise ValueError(
                f"Missing FP8 weight_scale_inv for quantized weight {source_weight_name!r} "
                f"(target {target_weight_name!r})."
            )
        return False
    if not has_quantized_loader:
        raise ValueError(
            f"Found {source_weight_name!r} with weight_scale_inv, but target module "
            f"{type(module).__name__} does not support grouped quantized loading."
        )
    if not bool(getattr(module, "quantized", False)):
        raise ValueError(
            f"Found FP8 weight_scale_inv for {source_weight_name!r}, but target module "
            f"{type(module).__name__} was not constructed with quantization enabled."
        )
    module.load_quantized_weight(loaded_weight, loaded_scale, loaded_shard_id)
    return True


def _validate_all_quantized_weights_loaded(model: nn.Module) -> None:
    missing = []
    for name, module in model.named_modules():
        if bool(getattr(module, "quantized", False)) and not bool(
            getattr(module, "_quantized_weight_loaded", False)
        ):
            loaded_ranges = getattr(module, "_quantized_loaded_ranges", [])
            missing.append(
                f"{name or '<root>'} ({type(module).__name__}, loaded_ranges={loaded_ranges})"
            )
    if missing:
        raise ValueError(
            "Missing FP8 weight loads for quantized Linear modules: "
            f"{missing[:8]}."
        )


def _iter_deltakv_compressor_items(state_dict: dict[str, torch.Tensor]):
    for key, weight in state_dict.items():
        parts = key.split(".")
        # Supported formats:
        # 1) HF-style per-layer: "...layers.{i}....compress_down...."
        # 2) Shared compressors: "compress_down...." (broadcast to all layers)
        if "compress_down" in parts:
            yield key, "compress_down", parts, weight
        elif "compress_up" in parts:
            yield key, "compress_up", parts, weight


def _infer_single_compressor_spec(state_dict: dict[str, torch.Tensor], comp_name: str):
    bias = False
    inferred_kind = None
    inferred_intermediate = None
    inferred_dtype = None

    for key, name, parts, weight in _iter_deltakv_compressor_items(state_dict):
        if name != comp_name:
            continue
        comp_token_idx = parts.index(comp_name)
        sub_parts = parts[comp_token_idx + 1:]
        if not sub_parts:
            continue

        inferred_dtype = inferred_dtype or weight.dtype
        if sub_parts[-1] == "bias":
            bias = True

        head = sub_parts[0]
        # nn.Linear: "...compress_down.weight" / "...compress_down.bias"
        if head in ("weight", "bias"):
            inferred_kind = inferred_kind or "linear"
            continue
        # nn.Sequential: "...compress_down.0.weight" / "...compress_down.2.weight"
        if head.isdigit():
            inferred_kind = inferred_kind or "mlp_gelu"
            if head == "0" and sub_parts[-1] == "weight":
                inferred_intermediate = inferred_intermediate or int(weight.shape[0])
            continue
        # SwiGLU: "...compress_down.w12.weight" / "...compress_down.w3.weight"
        if head == "w12":
            inferred_kind = inferred_kind or "mlp_swiglu"
            if sub_parts[-1] == "weight":
                inferred_intermediate = inferred_intermediate or int(weight.shape[0] // 2)
            continue

    return inferred_kind, inferred_intermediate, bias, inferred_dtype


def _infer_kv_compressed_size(state_dict: dict[str, torch.Tensor]) -> int | None:
    for _, name, parts, weight in _iter_deltakv_compressor_items(state_dict):
        comp_token_idx = parts.index(name)
        sub_parts = parts[comp_token_idx + 1:]
        if not sub_parts or sub_parts[-1] != "weight":
            continue

        head = sub_parts[0]
        if name == "compress_down":
            if head in ("weight", "2", "w3"):
                return int(weight.shape[0])
        else:
            if head in ("weight", "0", "w12"):
                return int(weight.shape[1])
    return None


def _resolve_deltakv_checkpoint_files(path: str) -> tuple[str, list[str], bool]:
    if os.path.isdir(path):
        files = sorted(glob(os.path.join(path, "*.safetensors")))
        is_safetensors = len(files) > 0
        if not is_safetensors:
            files = sorted(glob(os.path.join(path, "*.bin")) + glob(os.path.join(path, "*.pt")))
        ckpt_dir = path
    elif os.path.isfile(path):
        files = [path]
        is_safetensors = path.endswith(".safetensors")
        ckpt_dir = os.path.dirname(path)
    else:
        raise FileNotFoundError(f"No compressor weights found in {path}")

    if not files:
        raise FileNotFoundError(f"No compressor weights found in {path}")
    return ckpt_dir, files, is_safetensors


def _load_deltakv_state_dict(file: str, *, is_safetensors: bool) -> dict[str, torch.Tensor]:
    if is_safetensors:
        with safe_open(file, "pt", "cpu") as f:
            return {k: f.get_tensor(k) for k in f.keys()}
    return torch.load(file, map_location="cpu")


def sync_deltakv_config_from_checkpoint(config) -> bool:
    path = getattr(config, "deltakv_checkpoint_path", None)
    method = str(getattr(config, "sparse_method", "") or "")
    if not path or not method.startswith("deltakv"):
        return False

    ckpt_dir, files, is_safetensors = _resolve_deltakv_checkpoint_files(path)
    updates: dict[str, object] = {}
    config_json = os.path.join(ckpt_dir, "config.json")
    if os.path.isfile(config_json):
        with open(config_json, "r", encoding="utf-8") as f:
            ckpt_cfg = json.load(f)

        for key in (
            "deltakv_latent_dim",
            "use_nonlinear_compressor",
            "compressor_intermediate_size",
            "compressor_linear_bias",
            "compressor_down_type",
            "compressor_up_type",
            "compressor_down_intermediate_size",
            "compressor_up_intermediate_size",
        ):
            if key in ckpt_cfg:
                updates[key] = ckpt_cfg[key]

        if ckpt_cfg.get("split_kv", False):
            raise NotImplementedError(
                "Detected split_kv DeltaKV checkpoint from config.json. "
                "sparseengine DeltaKVCacheManager currently expects unified compress_down/compress_up."
            )

    missing_shape_keys = {
        "deltakv_latent_dim",
        "compressor_down_type",
        "compressor_up_type",
        "compressor_down_intermediate_size",
        "compressor_up_intermediate_size",
        "compressor_linear_bias",
        "use_nonlinear_compressor",
    }
    need_weight_inference = bool(missing_shape_keys.difference(updates.keys()))
    state_dict = None
    if need_weight_inference:
        for file in files:
            state_dict = _load_deltakv_state_dict(file, is_safetensors=is_safetensors)
            if any("compress_down" in key or "compress_up" in key for key in state_dict):
                break
            state_dict = None

        if state_dict is not None:
            deltakv_latent_dim = _infer_kv_compressed_size(state_dict)
            down_kind, down_inter, down_bias, _ = _infer_single_compressor_spec(state_dict, "compress_down")
            up_kind, up_inter, up_bias, _ = _infer_single_compressor_spec(state_dict, "compress_up")
            if deltakv_latent_dim is not None and "deltakv_latent_dim" not in updates:
                updates["deltakv_latent_dim"] = deltakv_latent_dim
            if down_kind is not None and "compressor_down_type" not in updates:
                updates["compressor_down_type"] = down_kind
            if up_kind is not None and "compressor_up_type" not in updates:
                updates["compressor_up_type"] = up_kind
            if down_kind is not None and "compressor_down_intermediate_size" not in updates:
                updates["compressor_down_intermediate_size"] = -1 if down_kind == "linear" else int(down_inter or -1)
            if up_kind is not None and "compressor_up_intermediate_size" not in updates:
                updates["compressor_up_intermediate_size"] = -1 if up_kind == "linear" else int(up_inter or -1)
            if down_kind is not None and up_kind is not None and "use_nonlinear_compressor" not in updates:
                updates["use_nonlinear_compressor"] = (down_kind != "linear" or up_kind != "linear")
            if down_bias == up_bias and "compressor_linear_bias" not in updates:
                updates["compressor_linear_bias"] = bool(down_bias)

            for key in state_dict.keys():
                if ".k_compress_down." in key or ".v_compress_down." in key or ".k_compress_up." in key or ".v_compress_up." in key:
                    raise NotImplementedError(
                        "Detected split_kv compressor checkpoint (k_compress_*/v_compress_*). "
                        "sparseengine DeltaKVCacheManager currently expects unified compress_down/compress_up."
                    )

    changed: dict[str, tuple[object, object]] = {}
    for key, value in updates.items():
        if value is None or not hasattr(config, key):
            continue
        current = getattr(config, key)
        if isinstance(value, str):
            value = value.strip().lower() or "auto"
        if current != value:
            setattr(config, key, value)
            changed[key] = (current, value)

    if changed:
        changes_str = ", ".join(f"{key}: {old} -> {new}" for key, (old, new) in changed.items())
        logger.info(f"Synced DeltaKV config from checkpoint {ckpt_dir}: {changes_str}")
    return bool(changed)


def _compressor_signature(mod: nn.Module):
    if isinstance(mod, nn.Linear):
        return "linear", None, (mod.bias is not None)
    if hasattr(mod, "w12") and hasattr(mod, "w3"):
        w3 = getattr(mod, "w3")
        inter = int(w3.in_features) if isinstance(w3, nn.Linear) else None
        has_bias = bool(getattr(getattr(mod, "w12", None), "bias", None) is not None) and bool(
            getattr(getattr(mod, "w3", None), "bias", None) is not None
        )
        return "mlp_swiglu", inter, has_bias
    if isinstance(mod, nn.Sequential) and len(mod) >= 3 and isinstance(mod[0], nn.Linear) and isinstance(mod[1], nn.GELU) and isinstance(mod[2], nn.Linear):
        return "mlp_gelu", int(mod[0].out_features), (mod[0].bias is not None and mod[2].bias is not None)
    return mod.__class__.__name__, None, None


def _maybe_rebuild_cache_manager_compressors(cache_manager, state_dict: dict[str, torch.Tensor]):
    config = getattr(cache_manager, "config", None)
    if config is None:
        return

    down_kind, down_inter, down_bias, down_dtype = _infer_single_compressor_spec(state_dict, "compress_down")
    up_kind, up_inter, up_bias, up_dtype = _infer_single_compressor_spec(state_dict, "compress_up")
    if down_kind is None or up_kind is None:
        return

    current_down = cache_manager.compress_down[0] if getattr(cache_manager, "compress_down", None) else None
    current_up = cache_manager.compress_up[0] if getattr(cache_manager, "compress_up", None) else None
    cur_down_kind, cur_down_inter, cur_down_bias = _compressor_signature(current_down) if current_down is not None else (None, None, None)
    cur_up_kind, cur_up_inter, cur_up_bias = _compressor_signature(current_up) if current_up is not None else (None, None, None)

    need_rebuild = (
        cur_down_kind != down_kind
        or cur_up_kind != up_kind
        or (down_kind != "linear" and cur_down_inter != down_inter)
        or (up_kind != "linear" and cur_up_inter != up_inter)
        or cur_down_bias != down_bias
        or cur_up_bias != up_bias
    )
    target_dtype = down_dtype or up_dtype
    if not need_rebuild:
        # Even if the structure matches, ensure dtype matches checkpoint weights (avoid bf16->fp32 silent casts).
        if target_dtype is not None and current_down is not None:
            try:
                cur_dtype = next(current_down.parameters()).dtype
            except StopIteration:
                cur_dtype = None
            if cur_dtype is not None and cur_dtype != target_dtype:
                logger.info(f"Casting sparseengine compressors to {target_dtype} to match checkpoint weights.")
                cache_manager.compress_down = [m.to(dtype=target_dtype) for m in cache_manager.compress_down]
                cache_manager.compress_up = [m.to(dtype=target_dtype) for m in cache_manager.compress_up]
        return

    logger.info(
        f"Rebuilding sparseengine compressors to match checkpoint: "
        f"down={down_kind}(inter={down_inter},bias={down_bias}) "
        f"up={up_kind}(inter={up_inter},bias={up_bias})"
    )

    # Update config so downstream logs/debug reflect the effective compressor architecture.
    if hasattr(config, "compressor_linear_bias"):
        if down_bias == up_bias:
            config.compressor_linear_bias = bool(down_bias)
        else:
            logger.warning(
                "Checkpoint has asymmetric bias setting between compress_down and compress_up; "
                "sparseengine config only has a single `compressor_linear_bias`, keeping current value."
            )
    if hasattr(config, "use_nonlinear_compressor"):
        config.use_nonlinear_compressor = (down_kind != "linear" or up_kind != "linear")
    if hasattr(config, "compressor_down_type"):
        config.compressor_down_type = down_kind
    if hasattr(config, "compressor_up_type"):
        config.compressor_up_type = up_kind
    if hasattr(config, "compressor_down_intermediate_size"):
        config.compressor_down_intermediate_size = -1 if down_kind == "linear" else int(down_inter or -1)
    if hasattr(config, "compressor_up_intermediate_size"):
        config.compressor_up_intermediate_size = -1 if up_kind == "linear" else int(up_inter or -1)

    from sparseengine.utils.compressor import create_compressor

    num_layers = len(cache_manager.compress_down)
    device = cache_manager.device
    cache_manager.compress_down = [
        create_compressor(is_down=True, config=config, bias_override=down_bias).to(device=device, dtype=target_dtype)
        for _ in range(num_layers)
    ]
    cache_manager.compress_up = [
        create_compressor(is_down=False, config=config, bias_override=up_bias).to(device=device, dtype=target_dtype)
        for _ in range(num_layers)
    ]


def load_deltakv_compressors_to_cache_manager(cache_manager, path: str):
    """
    Load DeltaKV compressor weights into cache manager compressor modules.
    """
    _, files, is_safetensors = _resolve_deltakv_checkpoint_files(path)

    loaded_count = 0
    for file in files:
        state_dict = _load_deltakv_state_dict(file, is_safetensors=is_safetensors)

        # Detect unsupported split_kv checkpoints early (k_compress_down/v_compress_down).
        for key in state_dict.keys():
            if ".k_compress_down." in key or ".v_compress_down." in key or ".k_compress_up." in key or ".v_compress_up." in key:
                raise NotImplementedError(
                    "Detected split_kv compressor checkpoint (k_compress_*/v_compress_*). "
                    "sparseengine DeltaKVCacheManager currently expects unified compress_down/compress_up."
                )

        if loaded_count == 0:
            _maybe_rebuild_cache_manager_compressors(cache_manager, state_dict)

        for key, weight in state_dict.items():
            parts = key.split('.')
            if "compress_down" not in parts and "compress_up" not in parts:
                continue

            # Two formats:
            # 1) HF-style per-layer: "...layers.{i}....compress_down...."
            # 2) Shared compressors: "compress_down...." (broadcast to all DeltaKV layers)
            is_shared = ("layers" not in parts)
            layer_idx = None
            if not is_shared:
                try:
                    layer_token_idx = parts.index("layers")
                    layer_idx = int(parts[layer_token_idx + 1])
                except (ValueError, IndexError):
                    raise ValueError(f"无法从权重键名中解析层索引: {key}")

            if "compress_down" in parts:
                comp_name = "compress_down"
            else:
                comp_name = "compress_up"

            comp_token_idx = parts.index(comp_name)
            sub_key = ".".join(parts[comp_token_idx + 1:])

            if is_shared:
                target_layer_indices = list(range(len(cache_manager.compress_down)))
            else:
                if layer_idx not in cache_manager.deltakv_layer_to_idx:
                    logger.debug(f"权重 {key} 对应的层索引 {layer_idx} 不在 deltakv_layer_to_idx 中，跳过")
                    continue
                target_layer_indices = [cache_manager.deltakv_layer_to_idx[layer_idx]]

            for l_idx in target_layer_indices:
                compressor = cache_manager.compress_down[l_idx] if comp_name == "compress_down" else cache_manager.compress_up[l_idx]
                try:
                    # 尝试获取对应的参数
                    if '.' in sub_key:
                        prefix, name = sub_key.rsplit('.', 1)
                        param = compressor.get_submodule(prefix).get_parameter(name)
                    else:
                        param = compressor.get_parameter(sub_key)

                    if param.shape != weight.shape:
                        raise ValueError(f"权重 {key} 形状不匹配: 预期 {param.shape}, 实际 {weight.shape}")

                    param.data.copy_(weight)
                    loaded_count += 1
                except Exception as e:
                    # 尝试直接访问属性作为备选方案
                    try:
                        target = getattr(compressor, sub_key)
                        if isinstance(target, nn.Parameter):
                            if target.shape != weight.shape:
                                raise ValueError(f"权重 {key} 形状不匹配: 预期 {target.shape}, 实际 {weight.shape}")
                            target.data.copy_(weight)
                            loaded_count += 1
                            continue
                    except Exception:
                        pass

                    raise RuntimeError(f"未能将权重 {key} 加载到压缩器模块 (layer {layer_idx}): {e}")

    assert loaded_count > 0, f"No DeltaKV compressor weights were loaded into cache manager from {path}"
    print(f"Successfully loaded {loaded_count} DeltaKV compressor weights into cache manager from {path}")


def load_model(
    model: nn.Module,
    path: str,
    *,
    tp_rank: int | None = None,
    tp_size: int | None = None,
    num_threads: int = 8,
    show_progress: bool | None = None,
    progress_rank: int | None = None,
):
    num_threads = int(num_threads)
    if num_threads <= 0:
        raise ValueError(f"num_threads must be positive, got {num_threads}.")
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    assert len(files) > 0, f"No safetensors found in {path}"
    checkpoint_is_rank_local = False

    # Some tensor-parallel converters emit one file per rank:
    #   model{tp_rank}-mp{tp_size}.safetensors
    # They may also split each rank across multiple files with the same prefix.
    # In either case load only the local rank files and do not slice them again.
    if tp_rank is not None and tp_size is not None:
        rank_prefix = os.path.join(path, f"model{tp_rank}-mp{tp_size}")
        rank_files = sorted(glob(f"{rank_prefix}*.safetensors"))
        if rank_files:
            files = rank_files
            checkpoint_is_rank_local = True
        else:
            mp_files = sorted(
                glob(os.path.join(path, f"model*-mp{tp_size}*.safetensors"))
            )
            if mp_files:
                raise FileNotFoundError(
                    "Detected per-rank weight shards but missing expected shard for this rank. "
                    f"expected_prefix={rank_prefix} available={mp_files}"
                )

    load_start = time.perf_counter()
    rank_local_tensor_slicing = (
        not checkpoint_is_rank_local
        and tp_rank is not None
        and tp_size is not None
        and tp_size > 1
    )
    loaded_count = 0
    loaded_tensor_bytes = 0
    loaded_parameter_names: set[str] = set()
    seen_source_keys: set[str] = set()
    if show_progress is None:
        show_progress = tp_rank is None or tp_rank == 0
    description = (
        "Multi-thread loading shards"
        if num_threads > 1 and len(files) > 1
        else "Loading weight shards"
    )
    if progress_rank is not None:
        description = f"{description} (rank {int(progress_rank)})"
    with tqdm(
        total=len(files),
        desc=description,
        unit="shard",
        dynamic_ncols=True,
        disable=not show_progress,
    ) as progress:
        for _, shard in _iter_safetensors_shards(
            files,
            num_threads=num_threads,
            model=None if checkpoint_is_rank_local else model,
        ):
            _normalize_block_scale_names(shard, model)
            tensors = shard.tensors
            loaded_tensor_bytes += sum(
                tensor.numel() * tensor.element_size()
                for tensor in tensors.values()
            )
            keys = list(shard.metadata)
            duplicate_source_keys = sorted(seen_source_keys.intersection(keys))
            if duplicate_source_keys:
                raise ValueError(
                    "Checkpoint contains duplicate tensor keys across safetensor shards: "
                    f"{duplicate_source_keys[:5]}."
                )
            seen_source_keys.update(keys)
            scale_keys = {key for key in keys if key.endswith(".weight_scale_inv")}
            consumed_scale_keys: set[str] = set()
            for source_weight_name in keys:
                if source_weight_name.endswith(".weight_scale_inv"):
                    continue
                scale_key = None
                if source_weight_name.endswith(".weight"):
                    scale_key = _scale_key_for_weight_key(source_weight_name)
                param_name = _target_weight_name_for_model(model, source_weight_name)
                if param_name is None:
                    skipped_weight_hook = getattr(model, "record_skipped_weight", None)
                    if callable(skipped_weight_hook):
                        weight_metadata = shard.metadata[source_weight_name]
                        scale_metadata = shard.metadata.get(scale_key)
                        skipped_weight_hook(
                            source_weight_name,
                            weight_metadata.shape,
                            weight_metadata.dtype,
                            (
                                scale_metadata.shape
                                if scale_metadata is not None
                                else None
                            ),
                            (
                                scale_metadata.dtype
                                if scale_metadata is not None
                                else None
                            ),
                        )
                    if scale_key is not None and scale_key in scale_keys:
                        consumed_scale_keys.add(scale_key)
                    continue
                loaded_scale = (
                    tensors.get(scale_key)
                    if scale_key is not None and scale_key in scale_keys
                    else None
                )
                special_loader = getattr(model, "load_special_weight", None)
                special_suffixes = tuple(getattr(model, "special_weight_loaders", ()))
                if callable(special_loader) and param_name.endswith(special_suffixes):
                    special_count = int(
                        special_loader(
                            param_name,
                            tensors[source_weight_name],
                            loaded_scale,
                        )
                    )
                    if special_count < 0:
                        raise ValueError(
                            f"load_special_weight() returned a negative count for {param_name!r}."
                        )
                    if special_count:
                        if scale_key is not None and loaded_scale is not None:
                            consumed_scale_keys.add(scale_key)
                        loaded_count += special_count
                        continue
                packed_modules_mapping = _packed_mapping_for(model, param_name)
                for k, (v, shard_id) in packed_modules_mapping.items():
                    if k in param_name:
                        packed_param_name = param_name.replace(k, v)
                        module = _module_for_parameter(model, packed_param_name)
                        if _load_grouped_quantized_weight(
                            model=model,
                            module=module,
                            source_weight_name=source_weight_name,
                            target_weight_name=packed_param_name,
                            loaded_weight=tensors[source_weight_name],
                            loaded_scale=loaded_scale,
                            loaded_shard_id=shard_id,
                        ):
                            consumed_scale_keys.add(scale_key)
                        else:
                            param = model.get_parameter(packed_param_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, tensors[source_weight_name], shard_id)
                        loaded_parameter_names.add(packed_param_name)
                        loaded_count += 1
                        break
                else:
                    try:
                        buffer = model.get_buffer(param_name)
                    except AttributeError:
                        buffer = None
                    if buffer is not None:
                        loaded_weight = tensors[source_weight_name]
                        if loaded_scale is not None:
                            raise ValueError(
                                f"Buffer {param_name!r} cannot load a quantization scale."
                            )
                        if buffer.shape != loaded_weight.shape:
                            raise ValueError(
                                f"Buffer {param_name!r} shape mismatch: "
                                f"expected {tuple(buffer.shape)}, got {tuple(loaded_weight.shape)}."
                            )
                        buffer.copy_(loaded_weight)
                        loaded_parameter_names.add(param_name)
                        loaded_count += 1
                        continue
                    module = _module_for_parameter(model, param_name)
                    loaded_scale = None
                    if source_weight_name.endswith(".weight"):
                        scale_key = _scale_key_for_weight_key(source_weight_name)
                        loaded_scale = tensors.get(scale_key)
                    if loaded_scale is not None:
                        consumed_scale_keys.add(scale_key)
                    if _load_grouped_quantized_weight(
                        model=model,
                        module=module,
                        source_weight_name=source_weight_name,
                        target_weight_name=param_name,
                        loaded_weight=tensors[source_weight_name],
                        loaded_scale=loaded_scale,
                    ):
                        loaded_parameter_names.add(param_name)
                        loaded_count += 1
                        continue
                    param = model.get_parameter(param_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, tensors[source_weight_name])
                    loaded_parameter_names.add(param_name)
                    loaded_count += 1
            unused_scale_keys = sorted(scale_keys - consumed_scale_keys)
            if unused_scale_keys:
                raise ValueError(
                    "Found weight_scale_inv tensors without a grouped FP8 weight load: "
                    f"{unused_scale_keys[:5]}."
                )
            progress.update(1)
            progress.set_postfix(
                loaded=f"{loaded_tensor_bytes / 2**30:.1f} GiB"
            )

    assert loaded_count > 0, f"No weights were loaded from {path}"
    _validate_all_quantized_weights_loaded(model)
    strict_validator = getattr(model, "validate_loaded_weights", None)
    if callable(strict_validator):
        strict_validator(loaded_parameter_names)
    prepare_loaded_weights = getattr(model, "prepare_loaded_weights", None)
    if callable(prepare_loaded_weights):
        prepare_loaded_weights()
    logger.info(
        f"Load weight end. elapsed={time.perf_counter() - load_start:.2f} s, "
        f"type={type(model).__name__}, shards={len(files)}, "
        f"workers={min(num_threads, len(files))}, weights={loaded_count}, "
        f"tensor_bytes={loaded_tensor_bytes / 2**30:.2f} GiB, "
        f"rank_local_checkpoint={checkpoint_is_rank_local}, "
        f"rank_local_tensor_slicing={rank_local_tensor_slicing}."
    )
