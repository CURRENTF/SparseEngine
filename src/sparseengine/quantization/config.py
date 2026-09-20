from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sparseengine.utils.config import config_get


@dataclass(frozen=True)
class QuantizationConfig:
    enabled: bool = False
    quant_method: str = ""
    weight_dtype: str = ""
    activation_scheme: str = ""
    weight_block_size: tuple[int, int] | None = None
    model_name: str = "qwen3_5"
    activation_dtype: str = "bfloat16"
    checkpoint_scale_layout: str = "block_128x128"

    @classmethod
    def disabled(
        cls,
        *,
        model_name: str = "qwen3_5",
        activation_dtype: Any = "bfloat16",
    ) -> "QuantizationConfig":
        return cls(
            model_name=model_name,
            activation_dtype=cls._normalize_activation_dtype(activation_dtype),
        )

    @staticmethod
    def _normalize_activation_dtype(value: Any) -> str:
        normalized = str(value or "bfloat16").strip().lower().removeprefix("torch.")
        aliases = {
            "bf16": "bfloat16",
            "bfloat16": "bfloat16",
            "fp16": "float16",
            "half": "float16",
            "float16": "float16",
            "fp32": "float32",
            "float": "float32",
            "float32": "float32",
        }
        try:
            return aliases[normalized]
        except KeyError as error:
            raise ValueError(
                "Unsupported model activation dtype for quantization: "
                f"{value!r}."
            ) from error

    def to_dict(self) -> dict[str, Any]:
        if not self.enabled:
            return {}
        payload: dict[str, Any] = {
            "quant_method": self.quant_method,
            "fmt": self.weight_dtype,
            "activation_scheme": self.activation_scheme,
        }
        if self.checkpoint_scale_layout == "per_tensor":
            payload["is_checkpoint_fp8_serialized"] = True
        elif self.weight_block_size is not None:
            payload["weight_block_size"] = list(self.weight_block_size)
        return payload

    @classmethod
    def from_hf_config(
        cls,
        value: Any,
        *,
        required_fp8: bool = False,
        model_name: str = "qwen3_5",
        activation_dtype: Any = "bfloat16",
    ) -> "QuantizationConfig":
        normalized_activation_dtype = cls._normalize_activation_dtype(
            activation_dtype
        )
        if value is None:
            if required_fp8:
                raise ValueError(
                    f"{model_name} requires FP8 quantization_config; "
                    "BF16/FP16 fallback is not supported."
                )
            return cls.disabled(
                model_name=model_name,
                activation_dtype=normalized_activation_dtype,
            )

        quant_method = str(
            config_get(value, "quant_method", config_get(value, "method", ""))
            or ""
        ).strip().lower()
        if quant_method not in {"fp8", "fbgemm_fp8"}:
            if required_fp8:
                raise ValueError(
                    f"{model_name} requires quantization_config.quant_method='fp8', "
                    f"got {quant_method!r}."
                )
            if quant_method:
                raise NotImplementedError(
                    f"SparseEngine does not support quant_method={quant_method!r} "
                    f"for {model_name}."
                )
            return cls.disabled(
                model_name=model_name,
                activation_dtype=normalized_activation_dtype,
            )

        weight_dtype = str(
            config_get(
                value,
                "weight_dtype",
                config_get(value, "fmt", config_get(value, "format", "e4m3")),
            )
            or ""
        ).strip().lower()
        if "e4m3" not in weight_dtype:
            raise ValueError(
                f"SparseEngine {model_name} FP8 supports e4m3 weights only, "
                f"got weight_dtype={weight_dtype!r}."
            )

        activation_scheme = str(
            config_get(
                value,
                "activation_scheme",
                config_get(value, "activation", "dynamic"),
            )
            or ""
        ).strip().lower()
        if activation_scheme != "dynamic":
            raise ValueError(
                f"SparseEngine {model_name} FP8 supports dynamic activation only, "
                f"got activation_scheme={activation_scheme!r}."
            )

        block_size = config_get(
            value,
            "weight_block_size",
            config_get(
                value,
                "weight_block_shape",
                config_get(value, "block_size", (128, 128)),
            ),
        )
        if isinstance(block_size, int):
            block_tuple = (int(block_size), int(block_size))
        elif isinstance(block_size, (list, tuple)) and len(block_size) == 2:
            block_tuple = (int(block_size[0]), int(block_size[1]))
        else:
            raise ValueError(f"weight_block_size must be a pair, got {block_size!r}.")
        if block_tuple != (128, 128):
            raise ValueError(
                f"SparseEngine {model_name} FP8 supports "
                "weight_block_size=(128, 128) only, "
                f"got {block_tuple}."
            )

        per_tensor = bool(config_get(value, "is_checkpoint_fp8_serialized", False)) and all(
            config_get(value, key, None) is None
            for key in ("weight_block_size", "weight_block_shape", "block_size")
        )

        return cls(
            enabled=True,
            quant_method="fp8",
            weight_dtype="e4m3",
            activation_scheme="dynamic",
            weight_block_size=block_tuple,
            model_name=model_name,
            activation_dtype=normalized_activation_dtype,
            checkpoint_scale_layout="per_tensor" if per_tensor else "block_128x128",
        )
