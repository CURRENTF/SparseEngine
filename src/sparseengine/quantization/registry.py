from __future__ import annotations

from typing import Any

import torch

from sparseengine.operators.fp8_linear import resolve_fp8_linear_provider


class QuantizationRegistry:
    """Resolve quantized Linear providers from validated config objects."""

    @staticmethod
    def resolve_linear_provider(
        quantization: Any,
        *,
        input_features: int,
        output_features: int,
    ):
        if not bool(getattr(quantization, "enabled", False)):
            return None
        quant_method = str(getattr(quantization, "quant_method", "") or "").strip().lower()
        if quant_method != "fp8":
            raise ValueError(f"Unsupported quantized Linear method={quant_method!r}.")
        activation_dtype_name = str(
            getattr(quantization, "activation_dtype", "bfloat16") or ""
        ).strip().lower()
        activation_dtypes = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        try:
            activation_dtype = activation_dtypes[activation_dtype_name]
        except KeyError as error:
            raise ValueError(
                "Unsupported quantized Linear activation dtype="
                f"{activation_dtype_name!r}."
            ) from error
        return resolve_fp8_linear_provider(
            tuple(quantization.weight_block_size),
            input_features=input_features,
            output_features=output_features,
            activation_dtype=activation_dtype,
            weight_layout_id=("tensor_scales_in_block_grid"
                              if quantization.checkpoint_scale_layout == "per_tensor"
                              else "block_128x128_nt_k_major"),
        )
