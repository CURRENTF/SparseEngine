"""Model-specific full-attention layer profile resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Iterable

from sparseengine.utils.config import config_get
from sparseengine.utils.log import logger


_PROFILE_RESOURCE = "profiles/full_attention_layers.json"
_PROFILE_METHODS = frozenset({"omnikv", "deltakv", "omnikv_prefill"})


@dataclass(frozen=True)
class FullAttentionLayerProfile:
    profile_id: str
    model_names: tuple[str, ...]
    sparse_methods: tuple[str, ...]
    full_attention_layers: tuple[int, ...]


def _model_name_suffix(model_name: str) -> str:
    value = str(model_name).strip().replace("\\", "/").rstrip("/")
    if not value:
        return ""
    parts = [part for part in value.split("/") if part]
    for part in parts:
        if part.startswith("models--"):
            cache_parts = [item for item in part.split("--") if item]
            if len(cache_parts) >= 3:
                return cache_parts[-1].casefold()
    return parts[-1].casefold() if parts else value.casefold()


def _parse_profile_catalog(payload: Any) -> tuple[FullAttentionLayerProfile, ...]:
    schema_version = payload.get("schema_version") if isinstance(payload, dict) else None
    if isinstance(schema_version, bool) or schema_version != 1:
        raise ValueError("Full-attention profile catalog must use schema_version=1.")
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, list):
        raise ValueError("Full-attention profile catalog profiles must be a list.")

    profiles: list[FullAttentionLayerProfile] = []
    profile_ids: set[str] = set()
    aliases_by_method: dict[tuple[str, str], str] = {}
    for raw in raw_profiles:
        if not isinstance(raw, dict):
            raise ValueError("Each full-attention profile must be a JSON object.")
        profile_id = str(raw.get("id", "")).strip()
        if not profile_id or profile_id in profile_ids:
            raise ValueError(
                "Full-attention profile id must be unique and non-empty: "
                f"{profile_id!r}."
            )
        profile_ids.add(profile_id)

        raw_names = raw.get("model_names")
        if not isinstance(raw_names, list) or not raw_names:
            raise ValueError(f"Profile {profile_id!r} requires non-empty model_names.")
        model_names = tuple(str(name).strip() for name in raw_names)
        if any(not name or _model_name_suffix(name) != name.casefold() for name in model_names):
            raise ValueError(
                f"Profile {profile_id!r} model_names must be bare model-name suffixes."
            )

        raw_methods = raw.get("sparse_methods")
        if not isinstance(raw_methods, list) or not raw_methods:
            raise ValueError(f"Profile {profile_id!r} requires non-empty sparse_methods.")
        sparse_methods = tuple(str(method).strip().lower() for method in raw_methods)
        unknown_methods = sorted(set(sparse_methods) - _PROFILE_METHODS)
        if unknown_methods:
            raise ValueError(
                f"Profile {profile_id!r} has unsupported sparse_methods={unknown_methods}."
            )

        raw_layers = raw.get("full_attention_layers")
        if (
            not isinstance(raw_layers, list)
            or not raw_layers
            or any(isinstance(layer, bool) or not isinstance(layer, int) for layer in raw_layers)
            or any(layer < 0 for layer in raw_layers)
            or raw_layers != sorted(set(raw_layers))
        ):
            raise ValueError(
                f"Profile {profile_id!r} full_attention_layers must be a non-empty, "
                "sorted list of unique non-negative integers."
            )

        for method in sparse_methods:
            for model_name in model_names:
                key = (method, model_name.casefold())
                previous = aliases_by_method.get(key)
                if previous is not None:
                    raise ValueError(
                        f"Model alias {model_name!r} for method {method!r} is ambiguous "
                        f"between profiles {previous!r} and {profile_id!r}."
                    )
                aliases_by_method[key] = profile_id

        profiles.append(
            FullAttentionLayerProfile(
                profile_id=profile_id,
                model_names=model_names,
                sparse_methods=sparse_methods,
                full_attention_layers=tuple(raw_layers),
            )
        )
    return tuple(profiles)


def load_full_attention_layer_profiles() -> tuple[FullAttentionLayerProfile, ...]:
    resource = files("sparseengine.configs").joinpath(_PROFILE_RESOURCE)
    with resource.open("r", encoding="utf-8") as handle:
        return _parse_profile_catalog(json.load(handle))


def resolve_full_attention_layer_profile(
    model_names: str | Iterable[str],
    sparse_method: str,
    *,
    profiles: tuple[FullAttentionLayerProfile, ...] | None = None,
) -> FullAttentionLayerProfile:
    """Resolve a profile by an exact model basename or Hugging Face cache name."""
    candidates = (model_names,) if isinstance(model_names, str) else tuple(model_names)
    normalized = tuple(
        dict.fromkeys(
            suffix
            for name in candidates
            if (suffix := _model_name_suffix(name))
        )
    )
    method = str(sparse_method).strip().lower()
    available = profiles if profiles is not None else load_full_attention_layer_profiles()
    matches = [
        profile
        for profile in available
        if method in profile.sparse_methods
        and any(
            candidate == alias.casefold()
            for candidate in normalized
            for alias in profile.model_names
        )
    ]
    # A prefill-specific calibration can override the shared decode profile;
    # existing explicit-KV/MLA profiles remain reusable without duplication.
    if method == "omnikv_prefill" and not matches:
        return resolve_full_attention_layer_profile(candidates, "omnikv", profiles=available)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            "Automatic full_attention_layers resolution is ambiguous for "
            f"model names={list(candidates)!r}, sparse_method={method!r}; "
            f"matched profiles={[profile.profile_id for profile in matches]}."
        )
    known = sorted(
        profile.profile_id for profile in available if method in profile.sparse_methods
    )
    raise ValueError(
        "No automatic full_attention_layers profile matches "
        f"model names={list(candidates)!r}, sparse_method={method!r}. "
        f"Known profiles={known}. Pass explicit full_attention_layers or calibrate "
        "the model with python -m sparseengine.utils.select_omnikv_full_layers."
    )


def resolve_auto_full_attention_layers(config: Any) -> None:
    resolve_auto_prefill_full_attention_layers(config)
    value = config.full_attention_layers
    if not isinstance(value, str) or value.strip().lower() != "auto":
        return
    if config.sparse_method not in _PROFILE_METHODS:
        config.full_attention_layers = []
        return

    model_names = [config.model]
    for hf_config in (config.outer_hf_config, config.hf_config):
        for key in ("_name_or_path", "name_or_path"):
            candidate = config_get(hf_config, key, None)
            if candidate:
                model_names.append(str(candidate))
    profile = resolve_full_attention_layer_profile(model_names, config.sparse_method)
    config.full_attention_layers = list(profile.full_attention_layers)
    config.resolved_full_attention_profile = profile.profile_id
    logger.info(
        "Resolved full_attention_layers='auto' with profile {} for model {}: {}.",
        profile.profile_id,
        config.model,
        config.full_attention_layers,
    )


def resolve_auto_prefill_full_attention_layers(config: Any) -> None:
    """Use the decode catalog while keeping the prefill layer axis independent."""
    if getattr(config, "prefill_sparse_method", "") != "omnikv_prefill":
        return
    value = config.omnikv_prefill_full_attention_layers
    if not isinstance(value, str) or value.strip().lower() != "auto":
        return
    from sparseengine.method_registry import omnikv_prefill_layer_indices

    model_names = [config.model]
    for hf_config in (config.outer_hf_config, config.hf_config):
        for key in ("_name_or_path", "name_or_path"):
            candidate = config_get(hf_config, key, None)
            if candidate:
                model_names.append(str(candidate))
    profile = resolve_full_attention_layer_profile(model_names, "omnikv_prefill")
    invalid_layers = set(profile.full_attention_layers) - set(range(config.hf_config.num_hidden_layers))
    if invalid_layers:
        raise ValueError(f"Prefill profile {profile.profile_id!r} contains unknown layer indices: {sorted(invalid_layers)}.")
    eligible = set(omnikv_prefill_layer_indices(config))
    config.omnikv_prefill_full_attention_layers = [
        layer for layer in profile.full_attention_layers if layer in eligible
    ]
    config.resolved_prefill_full_attention_profile = profile.profile_id
    logger.info(
        "Resolved omnikv_prefill_full_attention_layers='auto' with profile {} "
        "for model {} (global layers only): {}.",
        profile.profile_id, config.model, config.omnikv_prefill_full_attention_layers,
    )
