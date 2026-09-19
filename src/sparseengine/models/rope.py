from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from sparseengine.utils.config import config_get


STATIC_ROPE_TYPES = frozenset({"linear", "yarn", "llama3"})
SEQUENCE_DEPENDENT_ROPE_TYPES = frozenset({"dynamic", "longrope"})


def _rope_parameters(config: Any) -> dict[str, Any]:
    parameters = config_get(config, "rope_parameters", None)
    if not parameters:
        parameters = config_get(config, "rope_scaling", None)
    if parameters is None:
        return {}
    if not isinstance(parameters, Mapping):
        raise TypeError(
            "rope_parameters/rope_scaling must be a mapping, got "
            f"{type(parameters).__name__}."
        )
    return dict(parameters)


def resolve_rope_parameters(config: Any) -> dict[str, Any]:
    parameters = _rope_parameters(config)
    rope_type_value = parameters.get("rope_type", parameters.get("type"))
    rope_type = "default" if rope_type_value is None else str(rope_type_value).lower()
    parameters.pop("type", None)
    parameters["rope_type"] = rope_type
    return parameters


def resolve_rope_theta(config: Any, *, default: float = 10_000.0) -> float:
    parameters = _rope_parameters(config)
    value = parameters.get("rope_theta", config_get(config, "rope_theta", default))
    theta = float(value)
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError(f"rope_theta must be finite and positive, got {value!r}.")
    return theta


def _require_positive_factor(parameters: Mapping[str, Any], rope_type: str) -> float:
    if "factor" not in parameters:
        raise ValueError(f"{rope_type} rope scaling requires factor.")
    factor = float(parameters["factor"])
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError(
            f"{rope_type} rope scaling factor must be finite and >= 1, got {factor}."
        )
    return factor


def _require_original_max_position(parameters: Mapping[str, Any], rope_type: str) -> int:
    if "original_max_position_embeddings" not in parameters:
        raise ValueError(
            f"{rope_type} rope scaling requires original_max_position_embeddings."
        )
    original = int(parameters["original_max_position_embeddings"])
    if original <= 0:
        raise ValueError(
            f"{rope_type} original_max_position_embeddings must be positive, got {original}."
        )
    return original


def resolve_rope_scaling(
    config: Any,
    *,
    model_name: str,
) -> tuple[tuple[str, object], ...] | None:
    parameters = resolve_rope_parameters(config)
    rope_type = str(parameters["rope_type"])
    if rope_type == "default":
        return None
    if rope_type in SEQUENCE_DEPENDENT_ROPE_TYPES:
        raise NotImplementedError(
            f"{model_name} rope_type={rope_type!r} is sequence-length dependent and "
            "is not supported by the shared static RoPE cache."
        )
    if rope_type not in STATIC_ROPE_TYPES:
        raise NotImplementedError(
            f"{model_name} rope_type={rope_type!r} is unsupported; supported scaled "
            f"types are {sorted(STATIC_ROPE_TYPES)}."
        )

    _require_positive_factor(parameters, rope_type)
    if rope_type == "yarn":
        _require_original_max_position(parameters, rope_type)
        beta_fast = float(parameters.get("beta_fast", 32.0))
        beta_slow = float(parameters.get("beta_slow", 1.0))
        if not math.isfinite(beta_fast) or not math.isfinite(beta_slow):
            raise ValueError("yarn beta_fast and beta_slow must be finite.")
        if beta_fast <= beta_slow or beta_slow <= 0:
            raise ValueError(
                "yarn requires beta_fast > beta_slow > 0, got "
                f"beta_fast={beta_fast} beta_slow={beta_slow}."
            )
        for name in ("attention_factor", "mscale", "mscale_all_dim"):
            value = parameters.get(name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"yarn {name} must be finite, got {value!r}.")
        attention_factor = parameters.get("attention_factor")
        if attention_factor is not None and float(attention_factor) <= 0:
            raise ValueError(
                f"yarn attention_factor must be positive, got {attention_factor!r}."
            )
    elif rope_type == "llama3":
        _require_original_max_position(parameters, rope_type)
        required = {"low_freq_factor", "high_freq_factor"}
        missing = sorted(required.difference(parameters))
        if missing:
            raise ValueError(f"llama3 rope scaling missing required keys: {missing}.")
        low = float(parameters["low_freq_factor"])
        high = float(parameters["high_freq_factor"])
        if not 0 < low < high:
            raise ValueError(
                "llama3 requires 0 < low_freq_factor < high_freq_factor, got "
                f"{low} and {high}."
            )

    frozen = {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in parameters.items()
    }
    return tuple(sorted(frozen.items()))


def resolve_rope_max_position(config: Any, *, model_name: str) -> int:
    declared = int(config_get(config, "max_position_embeddings", 0) or 0)
    if declared <= 0:
        raise ValueError(
            f"{model_name} max_position_embeddings must be positive, got {declared}."
        )
    scaling = resolve_rope_scaling(config, model_name=model_name)
    if scaling is None:
        return declared
    parameters = dict(scaling)
    rope_type = str(parameters["rope_type"])
    if rope_type == "linear":
        if "original_max_position_embeddings" in parameters:
            return declared
        return int(declared * float(parameters["factor"]))
    if rope_type == "yarn":
        original = int(parameters["original_max_position_embeddings"])
        factor = float(parameters["factor"])
        scaled_value = original * factor
        if not math.isfinite(scaled_value):
            raise ValueError(
                f"{model_name} yarn context length must be finite, got "
                f"original={original} factor={factor}."
            )
        scaled = int(round(scaled_value))
        if not math.isclose(scaled_value, scaled, rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError(
                f"{model_name} yarn context length must be an integer, got "
                f"original={original} factor={factor} scaled={scaled_value}."
            )
        if declared not in {original, scaled}:
            raise ValueError(
                f"{model_name} yarn context lengths are inconsistent: "
                f"max_position_embeddings={declared}, "
                f"original_max_position_embeddings={original}, factor={factor}, "
                f"scaled_max_position_embeddings={scaled}. Expected the declared "
                "length to equal either the original length (legacy config) or the "
                "scaled length."
            )
        return scaled
    return declared
