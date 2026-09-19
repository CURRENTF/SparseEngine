from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from threading import Lock

import torch

import sparseengine.platforms as platforms
from sparseengine.platforms import device_runtime
from sparseengine.operators.registry import (
    OpRegistry,
    OpResolver,
    PortfolioPolicy,
    ProfileMatch,
    ProviderRole,
    SupportResult,
    runtime_version_at_least,
)
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum
from sparseengine.utils.device_name import device_name_contains


@dataclass(frozen=True)
class Fp8LinearSpec:
    block_shape: tuple[int, int]
    input_features: int
    output_features: int
    activation_dtype: torch.dtype = torch.bfloat16
    weight_dtype: torch.dtype = torch.float8_e4m3fn
    scale_dtype: torch.dtype = torch.float32
    weight_layout_id: str = "block_128x128_nt_k_major"
    cuda_graph: bool = True

    def __post_init__(self) -> None:
        if self.input_features <= 0 or self.output_features <= 0:
            raise ValueError("FP8 Linear feature sizes must be positive.")
        if any(int(value) <= 0 for value in self.block_shape):
            raise ValueError(
                f"FP8 Linear block shape must be positive: {self.block_shape}."
            )


class Fp8LinearProvider:
    name = ""

    def __init__(
        self,
        spec: Fp8LinearSpec | None = None,
        caps: DeviceCaps | None = None,
    ) -> None:
        self.spec = spec
        self.caps = caps
        self.workspace_lane = "default"
        self._workspace_used = False

    @classmethod
    def bind(
        cls,
        spec: Fp8LinearSpec,
        caps: DeviceCaps,
    ) -> "Fp8LinearProvider":
        return cls(spec, caps)

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "weight_layout_id": (
                self.spec.weight_layout_id if self.spec is not None else "unknown"
            ),
        }

    def bind_workspace_lane(self, lane: str) -> None:
        if self._workspace_used:
            raise RuntimeError("FP8 workspace lane must be bound before warmup/capture")
        self.workspace_lane = lane

    def prepare_weights(self, weight: torch.Tensor, scales: torch.Tensor) -> None:
        """Prepare provider-owned derived layouts after all logical shards load."""

    def _validate_call(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale_inv: torch.Tensor,
    ) -> None:
        if x.ndim < 2 or weight.ndim != 2 or weight_scale_inv.ndim != 2:
            raise ValueError(
                "FP8 Linear requires rank-2 weights/scales and rank-2-or-higher "
                f"activations, got {tuple(x.shape)}, {tuple(weight.shape)}, "
                f"{tuple(weight_scale_inv.shape)}."
            )
        if x.device != weight.device or weight.device != weight_scale_inv.device:
            raise ValueError("FP8 Linear inputs, weights, and scales must share a device.")
        if weight.dtype != torch.float8_e4m3fn:
            raise TypeError(f"FP8 Linear requires E4M3 weights, got {weight.dtype}.")
        if weight_scale_inv.dtype != torch.float32:
            raise TypeError(
                f"FP8 Linear requires FP32 weight scales, got {weight_scale_inv.dtype}."
            )
        if int(x.shape[-1]) != int(weight.shape[1]):
            raise ValueError(
                f"FP8 Linear K mismatch: x={tuple(x.shape)}, weight={tuple(weight.shape)}."
            )
        expected_scale_shape = (
            (int(weight.shape[0]) + 127) // 128,
            (int(weight.shape[1]) + 127) // 128,
        )
        if tuple(weight_scale_inv.shape) != expected_scale_shape:
            raise ValueError(
                "FP8 Linear weight scale shape mismatch: "
                f"expected={expected_scale_shape}, got={tuple(weight_scale_inv.shape)}."
            )
        if self.spec is not None:
            if x.dtype != self.spec.activation_dtype:
                raise TypeError(
                    "Bound FP8 Linear activation dtype changed after preparation: "
                    f"expected={self.spec.activation_dtype}, got={x.dtype}."
                )
            actual_features = (int(weight.shape[1]), int(weight.shape[0]))
            expected_features = (
                self.spec.input_features,
                self.spec.output_features,
            )
            if actual_features != expected_features:
                raise ValueError(
                    "Bound FP8 Linear weight shape changed after preparation: "
                    f"expected={expected_features}, got={actual_features}."
                )

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale_inv: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


FP8_LINEAR_REGISTRY: OpRegistry[Fp8LinearSpec, Fp8LinearProvider] = OpRegistry(
    "block-scaled FP8 Linear",
    portfolio=PortfolioPolicy(
        upstream_standard=(
            "sgl_tensor_fp8",
            "flashinfer_sm90",
            "flashinfer_groupwise_sm120",
        ),
        repo_portable=("triton",),
    ),
    profile_order=("sm120_fp8_linear_dispatch_plan",),
)


@FP8_LINEAR_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class SglTensorFp8LinearProvider(Fp8LinearProvider):
    """Per-logical-tensor weights, per-token activations, channel-scale epilogue."""

    name = "sgl_tensor_fp8"

    @classmethod
    def supports(cls, spec, caps):
        if spec.weight_layout_id != "tensor_scales_in_block_grid":
            return SupportResult.unsupported("requires tensor-scaled checkpoint weights")
        if caps.platform != PlatformEnum.CUDA or not caps.supports_native_fp8:
            return SupportResult.unsupported("requires CUDA native FP8")
        capability = caps.compute_capability
        if capability is None or capability < (8, 9):
            return SupportResult.unsupported("requires SM89 or newer")
        minimum = (12, 8) if capability >= (10, 0) else (12, 4) if capability == (8, 9) else (12, 0)
        if not runtime_version_at_least(caps.runtime_version, minimum):
            return SupportResult.unsupported(f"requires CUDA runtime >= {minimum}")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("requires graph capture support")
        if spec.activation_dtype not in (torch.bfloat16, torch.float16):
            return SupportResult.unsupported("requires BF16 or FP16 activations")
        if spec.weight_dtype != torch.float8_e4m3fn or spec.scale_dtype != torch.float32:
            return SupportResult.unsupported("requires E4M3 weights and FP32 scales")
        if spec.block_shape != (128, 128) or spec.input_features % 16 or spec.output_features % 8:
            return SupportResult.unsupported("requires scale grid 128, K aligned 16, N aligned 8")
        from sparseengine.kernels.external.sgl.fp8_linear import tensor_fp8_ops
        tensor_fp8_ops()
        return SupportResult.yes()

    def prepare_weights(self, weight, scales):
        if not torch.equal(scales, scales[:, :1].expand_as(scales)):
            raise ValueError("Tensor-scaled Linear cannot consume scales varying along K")
        # Preserve distinct fused projection scales without requantizing weights.
        self.channel_scales = scales[:, 0].repeat_interleave(128)[:weight.shape[0]].contiguous()

    def binding_metadata(self):
        return {**super().binding_metadata(), "activation_quantization": "dynamic_per_token",
                "weight_quantization": "per_logical_tensor", "gemm": "sgl_kernel.fp8_scaled_mm"}

    def __call__(self, x, weight, weight_scale_inv, bias=None):
        self._validate_call(x, weight, weight_scale_inv)
        if not hasattr(self, "channel_scales"):
            raise RuntimeError("Tensor-scaled FP8 Linear weights were not prepared")
        from sparseengine.kernels.external.sgl.fp8_linear import tensor_fp8_ops
        quantize, gemm = tensor_fp8_ops()
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        if not flat.shape[0]:
            return x.new_empty((*x.shape[:-1], weight.shape[0]))
        quantized = torch.empty_like(flat, dtype=torch.float8_e4m3fn)
        scales = torch.empty((flat.shape[0], 1), device=x.device, dtype=torch.float32)
        quantize(flat, quantized, scales)
        output = gemm(quantized, weight.t(), scales, self.channel_scales, x.dtype, bias)
        return output.reshape(*x.shape[:-1], weight.shape[0])


@dataclass(frozen=True, slots=True)
class _Sm120ActivationWorkspace:
    padded_inputs: torch.Tensor
    quantized: torch.Tensor
    scales: torch.Tensor


# Keep shape buckets stable across graph captures. Sequential layers share a
# lane; the auxiliary MoE branch is bound to a separate lane before warmup.
_SM120_ACTIVATION_WORKSPACES: dict[
    tuple[str, str, int | None, int, int], _Sm120ActivationWorkspace
] = {}
_SM120_ACTIVATION_WORKSPACE_LOCK = Lock()


def _sm120_activation_buffers(
    inputs: torch.Tensor,
    *,
    rows: int | None = None,
    lane: str = "default",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_rows, features = map(int, inputs.shape)
    rows = input_rows if rows is None else int(rows)
    if rows <= 0:
        raise ValueError("SM120 FP8 Linear requires at least one activation row.")
    capacity = 1 << max(rows - 1, 0).bit_length()
    key = (lane, inputs.device.type, inputs.device.index, features, capacity)
    workspace = _SM120_ACTIVATION_WORKSPACES.get(key)
    if workspace is None:
        if device_runtime.is_stream_capturing():
            raise RuntimeError(
                "SM120 FP8 Linear activation workspace was not created during "
                f"warmup for shape bucket M<={capacity}, K={features}."
            )
        with _SM120_ACTIVATION_WORKSPACE_LOCK:
            workspace = _SM120_ACTIVATION_WORKSPACES.get(key)
            if workspace is None:
                workspace = _Sm120ActivationWorkspace(
                    padded_inputs=torch.empty(
                        (capacity, features),
                        dtype=inputs.dtype,
                        device=inputs.device,
                    ),
                    quantized=torch.empty(
                        (capacity, features),
                        dtype=torch.float8_e4m3fn,
                        device=inputs.device,
                    ),
                    scales=torch.empty(
                        (capacity, features // 128),
                        dtype=torch.float32,
                        device=inputs.device,
                    ),
                )
                _SM120_ACTIVATION_WORKSPACES[key] = workspace
    if workspace.padded_inputs.dtype != inputs.dtype:
        raise TypeError(
            "SM120 FP8 Linear activation workspace dtype changed after warmup: "
            f"workspace={workspace.padded_inputs.dtype} inputs={inputs.dtype}."
        )
    return (
        workspace.padded_inputs[:rows],
        workspace.quantized[:rows],
        workspace.scales[:rows],
    )


def _sm120_activation_workspace(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, quantized, scales = _sm120_activation_buffers(inputs)
    return quantized, scales

_SM120_FP8_LINEAR_PROFILE_RESOURCE = "profiles/sm120_fp8_linear.json"


@lru_cache(maxsize=1)
def _load_sm120_fp8_linear_profile() -> tuple[
    dict[str, object],
    dict[tuple[int, int], int],
]:
    resource = files("sparseengine.operators").joinpath(
        _SM120_FP8_LINEAR_PROFILE_RESOURCE
    )
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Failed to load SM120 FP8 Linear profile "
            f"{_SM120_FP8_LINEAR_PROFILE_RESOURCE}: {error}"
        ) from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("Unsupported SM120 FP8 Linear profile schema.")
    expected_device = {
        "device_name_keyword": "RTX PRO 6000",
        "compute_capability": [12, 0],
    }
    expected_contract = {
        "activation_dtype": "bfloat16",
        "weight_dtype": "float8_e4m3fn",
        "scale_dtype": "float32",
        "block_shape": [128, 128],
        "weight_layout_id": "block_128x128_nt_k_major",
        "cuda_graph": True,
    }
    for field, expected in (
        ("device", expected_device),
        ("contract", expected_contract),
    ):
        if payload.get(field) != expected:
            raise RuntimeError(
                f"SM120 FP8 Linear profile {field} mismatch: "
                f"{payload.get(field)!r}."
            )
    toolchain = payload.get("toolchain")
    required_toolchain_fields = {
        "torch",
        "cuda_runtime",
        "triton",
        "flashinfer_python",
        "sglang_kernel",
    }
    if not isinstance(toolchain, dict) or set(toolchain) != required_toolchain_fields:
        raise RuntimeError(
            "SM120 FP8 Linear profile must record the complete profiled "
            f"toolchain, got {toolchain!r}."
        )
    if not isinstance(payload.get("profile_id"), str) or not isinstance(
        payload.get("provenance"), dict
    ):
        raise RuntimeError(
            "SM120 FP8 Linear profile must identify its provenance."
        )
    if payload.get("profile_status") not in {"tuned", "provisional"}:
        raise RuntimeError(
            "SM120 FP8 Linear profile_status must be 'tuned' or 'provisional'."
        )
    raw_routes = payload.get("routes")
    if not isinstance(raw_routes, list) or not raw_routes:
        raise RuntimeError("SM120 FP8 Linear profile routes are empty.")
    routes: dict[tuple[int, int], int] = {}
    for row in raw_routes:
        if not isinstance(row, dict):
            raise RuntimeError(f"Invalid SM120 FP8 Linear route: {row!r}.")
        key = (int(row["input_features"]), int(row["output_features"]))
        threshold = int(row["flashinfer_min_tokens"])
        if key in routes or any(value <= 0 for value in key) or threshold <= 0:
            raise RuntimeError(f"Invalid SM120 FP8 Linear route: {row!r}.")
        routes[key] = threshold
    return payload, routes


def _sm120_fp8_linear_profile_support(
    spec: Fp8LinearSpec,
    caps: DeviceCaps,
) -> ProfileMatch:
    payload, routes = _load_sm120_fp8_linear_profile()
    device = payload["device"]
    if (
        not device_name_contains(caps.device_name, device["device_name_keyword"])
        or list(caps.compute_capability or ()) != device["compute_capability"]
    ):
        return ProfileMatch.no(
            "requires profiled NVIDIA RTX PRO 6000 SM120 hardware, got "
            f"{caps.device_name} {caps.compute_capability}"
        )
    contract = payload["contract"]
    actual_contract = {
        "activation_dtype": str(spec.activation_dtype).removeprefix("torch."),
        "weight_dtype": str(spec.weight_dtype).removeprefix("torch."),
        "scale_dtype": str(spec.scale_dtype).removeprefix("torch."),
        "block_shape": list(spec.block_shape),
        "weight_layout_id": spec.weight_layout_id,
        "cuda_graph": spec.cuda_graph,
    }
    if actual_contract != contract:
        return ProfileMatch.no(
            f"requires profiled FP8 Linear contract {contract}, got {actual_contract}"
        )
    shape = (spec.input_features, spec.output_features)
    if shape not in routes:
        return ProfileMatch.no(
            f"requires a profiled SM120 FP8 Linear shape, got {shape}"
        )
    runtime_toolchain = _sm120_fp8_linear_runtime_toolchain(caps)
    if runtime_toolchain != payload["toolchain"]:
        return ProfileMatch.no(
            "requires exact profiled toolchain "
            f"{payload['toolchain']}, got {runtime_toolchain}"
        )
    return ProfileMatch.yes(
        f"matched offline profile {payload['profile_id']}; "
        "atomic route contracts are independently eligible"
    )


def _sm120_fp8_linear_runtime_toolchain(caps: DeviceCaps) -> dict[str, str | None]:
    import triton

    from sparseengine.kernels.external.flashinfer.support import (
        flashinfer_kernel_health,
    )
    from sparseengine.kernels.external.sgl.support import sgl_kernel_health

    flashinfer_health = flashinfer_kernel_health()
    sgl_health = sgl_kernel_health()
    return {
        "torch": str(torch.__version__).split("+", 1)[0].rsplit(".", 1)[0],
        "cuda_runtime": str(caps.runtime_version),
        "triton": str(triton.__version__).rsplit(".", 1)[0],
        "flashinfer_python": flashinfer_health.version,
        "sglang_kernel": sgl_health.version,
    }


@FP8_LINEAR_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class FlashInferSm90Fp8LinearProvider(Fp8LinearProvider):
    name = "flashinfer_sm90"

    @classmethod
    def supports(cls, spec: Fp8LinearSpec, caps: DeviceCaps) -> SupportResult:
        if spec.block_shape != (128, 128):
            return SupportResult.unsupported(
                f"requires block_shape=(128, 128), got {spec.block_shape}"
            )
        if spec.activation_dtype != torch.bfloat16:
            return SupportResult.unsupported(
                f"requires BF16 activations, got {spec.activation_dtype}"
            )
        if (
            spec.weight_dtype != torch.float8_e4m3fn
            or spec.scale_dtype != torch.float32
        ):
            return SupportResult.unsupported(
                "requires E4M3 weights with FP32 block scales, got "
                f"{spec.weight_dtype} and {spec.scale_dtype}"
            )
        if spec.weight_layout_id != "block_128x128_nt_k_major":
            return SupportResult.unsupported(
                f"requires block_128x128_nt_k_major, got {spec.weight_layout_id}"
            )
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.unsupported(
                f"requires CUDA SM90, got {caps.platform.name} {caps.compute_capability}"
            )
        if not runtime_version_at_least(caps.runtime_version, (12, 8)):
            return SupportResult.unsupported(
                "requires CUDA runtime >= 12.8, "
                f"got {caps.runtime_version or 'unknown'}"
            )
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        if not caps.supports_native_fp8:
            return SupportResult.unsupported("device does not provide native FP8 tensor cores")
        if spec.input_features % 128:
            return SupportResult.unsupported(
                f"requires input_features divisible by 128, got {spec.input_features}"
            )
        if spec.output_features % 64:
            return SupportResult.unsupported(
                f"requires output_features divisible by 64, got {spec.output_features}"
            )
        from sparseengine.kernels.external.flashinfer.fp8_linear import (
            flashinfer_sm90_fp8_linear_support,
        )

        supported, reason = flashinfer_sm90_fp8_linear_support()
        return SupportResult.yes(reason) if supported else SupportResult.unsupported(reason)

    def __call__(self, x, weight, weight_scale_inv, bias=None):
        self._validate_call(x, weight, weight_scale_inv)
        if x.dtype != torch.bfloat16:
            raise TypeError(
                f"FlashInfer SM90 FP8 Linear requires BF16 activations, got {x.dtype}."
            )
        from sparseengine.kernels.external.flashinfer.fp8_linear import (
            flashinfer_fp8_blockscale_gemm_sm90,
        )

        original_shape = x.shape[:-1]
        output = flashinfer_fp8_blockscale_gemm_sm90(
            x.reshape(-1, x.shape[-1]).contiguous(),
            weight,
            weight_scale_inv,
            out_dtype=torch.bfloat16,
        )
        if bias is not None:
            output.add_(bias)
        return output.reshape(*original_shape, weight.shape[0])


@FP8_LINEAR_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class FlashInferGroupwiseSm120Fp8LinearProvider(Fp8LinearProvider):
    """SGL activation quantization plus FlashInfer SM120 CUTLASS GEMM."""

    name = "flashinfer_groupwise_sm120"

    @classmethod
    def supports(cls, spec: Fp8LinearSpec, caps: DeviceCaps) -> SupportResult:
        if spec.block_shape != (128, 128):
            return SupportResult.unsupported(
                f"requires block_shape=(128, 128), got {spec.block_shape}"
            )
        if spec.activation_dtype != torch.bfloat16:
            return SupportResult.unsupported(
                f"requires BF16 activations, got {spec.activation_dtype}"
            )
        if (
            spec.weight_dtype != torch.float8_e4m3fn
            or spec.scale_dtype != torch.float32
        ):
            return SupportResult.unsupported(
                "requires E4M3 weights with FP32 block scales, got "
                f"{spec.weight_dtype} and {spec.scale_dtype}"
            )
        if spec.weight_layout_id != "block_128x128_nt_k_major":
            return SupportResult.unsupported(
                f"requires block_128x128_nt_k_major, got {spec.weight_layout_id}"
            )
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (12, 0):
            return SupportResult.unsupported(
                f"requires CUDA SM120, got {caps.platform.name} {caps.compute_capability}"
            )
        if not runtime_version_at_least(caps.runtime_version, (12, 8)):
            return SupportResult.unsupported(
                "requires CUDA runtime >= 12.8, "
                f"got {caps.runtime_version or 'unknown'}"
            )
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        if not caps.supports_native_fp8:
            return SupportResult.unsupported("device does not provide native FP8 tensor cores")
        if spec.input_features % 128:
            return SupportResult.unsupported(
                f"requires input_features divisible by 128, got {spec.input_features}"
            )
        if spec.output_features % 128:
            return SupportResult.unsupported(
                f"requires output_features divisible by 128, got {spec.output_features}"
            )
        from sparseengine.kernels.external.flashinfer.fp8_linear import (
            flashinfer_sm120_groupwise_fp8_linear_support,
        )
        from sparseengine.kernels.external.sgl.moe import (
            sgl_fp8_group_quantization_support,
        )

        gemm_supported, gemm_reason = (
            flashinfer_sm120_groupwise_fp8_linear_support()
        )
        if not gemm_supported:
            return SupportResult.unsupported(gemm_reason)
        quant_supported, quant_reason = sgl_fp8_group_quantization_support()
        if not quant_supported:
            return SupportResult.unsupported(quant_reason)
        return SupportResult.yes(f"{quant_reason}; {gemm_reason}")

    def binding_metadata(self) -> dict[str, object]:
        return {
            **super().binding_metadata(),
            "activation_quantizer": "sglang-kernel:sgl_per_token_group_quant_8bit",
            "gemm": "flashinfer:gemm_fp8_nt_groupwise",
            "activation_scale_layout": "K-major_per_token_group128",
            "weight_scale_layout": "K-major_block128x128",
            "activation_workspace": "lane_geometric_warmup_cache",
            "workspace_lane": self.workspace_lane,
        }

    def __call__(self, x, weight, weight_scale_inv, bias=None):
        self._validate_call(x, weight, weight_scale_inv)
        if x.dtype != torch.bfloat16:
            raise TypeError(
                "FlashInfer SM120 groupwise FP8 Linear requires BF16 activations, "
                f"got {x.dtype}."
            )
        from sparseengine.kernels.external.flashinfer.fp8_linear import (
            flashinfer_fp8_nt_groupwise_sm120,
        )
        from sparseengine.kernels.external.sgl.moe import (
            sgl_per_token_group_quant_8bit,
        )

        original_shape = x.shape[:-1]
        inputs = x.reshape(-1, x.shape[-1]).contiguous()
        rows = int(inputs.shape[0])
        kernel_rows = (rows + 3) & ~3
        self._workspace_used = True
        padded_inputs, quantized, scales = _sm120_activation_buffers(
            inputs,
            rows=kernel_rows,
            lane=self.workspace_lane,
        )
        if kernel_rows != rows:
            padded_inputs.zero_()
            padded_inputs[:rows].copy_(inputs)
            kernel_inputs = padded_inputs
        else:
            kernel_inputs = inputs
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        sgl_per_token_group_quant_8bit(
            kernel_inputs,
            quantized,
            scales,
            128,
            1.0e-10,
            fp8_info.min,
            fp8_info.max,
            enable_v2=True,
        )
        output = flashinfer_fp8_nt_groupwise_sm120(
            quantized,
            weight,
            scales,
            weight_scale_inv,
            out_dtype=torch.bfloat16,
        )[:rows]
        if bias is not None:
            output.add_(bias)
        return output.reshape(*original_shape, weight.shape[0])


@FP8_LINEAR_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class TritonFp8LinearProvider(Fp8LinearProvider):
    name = "triton"

    @classmethod
    def supports(cls, spec: Fp8LinearSpec, caps: DeviceCaps) -> SupportResult:
        if spec.block_shape != (128, 128):
            return SupportResult.unsupported(
                f"requires block_shape=(128, 128), got {spec.block_shape}"
            )
        if caps.platform != PlatformEnum.CUDA:
            return SupportResult.unsupported(f"requires CUDA, got {caps.platform.name}")
        if not caps.supports_triton:
            return SupportResult.unsupported("platform does not support Triton")
        if not caps.supports_native_fp8:
            return SupportResult.unsupported("device does not provide native FP8 tensor cores")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        if spec.activation_dtype not in (torch.bfloat16, torch.float16):
            return SupportResult.unsupported(
                f"requires BF16 or FP16 activations, got {spec.activation_dtype}"
            )
        if (
            spec.weight_dtype != torch.float8_e4m3fn
            or spec.scale_dtype != torch.float32
        ):
            return SupportResult.unsupported(
                "requires E4M3 weights with FP32 block scales, got "
                f"{spec.weight_dtype} and {spec.scale_dtype}"
            )
        if spec.weight_layout_id != "block_128x128_nt_k_major":
            return SupportResult.unsupported(
                f"requires block_128x128_nt_k_major, got {spec.weight_layout_id}"
            )
        return SupportResult.yes()

    def __call__(self, x, weight, weight_scale_inv, bias=None):
        self._validate_call(x, weight, weight_scale_inv)
        from sparseengine.kernels.triton.fp8_blockwise import fp8_blockwise_matmul

        original_shape = x.shape[:-1]
        output = fp8_blockwise_matmul(
            x.reshape(-1, x.shape[-1]).contiguous(),
            weight,
            weight_scale_inv,
            output_dtype=x.dtype,
        )
        if bias is not None:
            output.add_(bias)
        return output.reshape(*original_shape, weight.shape[0])


@dataclass(frozen=True, slots=True)
class Fp8LinearDispatchRoute:
    min_tokens: int
    max_tokens: int | None
    provider: Fp8LinearProvider

    def matches(self, num_tokens: int) -> bool:
        return num_tokens >= self.min_tokens and (
            self.max_tokens is None or num_tokens <= self.max_tokens
        )


@FP8_LINEAR_REGISTRY.register_profile
class Sm120Fp8LinearDispatchPlan(Fp8LinearProvider):
    """Prepared model-independent token routes over atomic SM120 providers."""

    name = "sm120_fp8_linear_dispatch_plan"

    def __init__(self, spec: Fp8LinearSpec, caps: DeviceCaps) -> None:
        super().__init__(spec, caps)
        payload, profiles = _load_sm120_fp8_linear_profile()
        threshold = profiles[(spec.input_features, spec.output_features)]
        self.profile = payload
        self.runtime_toolchain = _sm120_fp8_linear_runtime_toolchain(caps)
        self.routes = (
            Fp8LinearDispatchRoute(
                0,
                threshold - 1,
                TritonFp8LinearProvider.bind(spec, caps),
            ),
            Fp8LinearDispatchRoute(
                threshold,
                None,
                FlashInferGroupwiseSm120Fp8LinearProvider.bind(spec, caps),
            ),
        )
        self._runtime_kernel_path_counts: dict[str, dict[str, int]] = {}

    def bind_workspace_lane(self, lane: str) -> None:
        super().bind_workspace_lane(lane)
        for route in self.routes:
            route.provider.bind_workspace_lane(lane)

    @classmethod
    def atomic_provider_names(cls, spec: Fp8LinearSpec) -> tuple[str, ...]:
        del spec
        return ("triton", "flashinfer_groupwise_sm120")

    @classmethod
    def matches(cls, spec: Fp8LinearSpec, caps: DeviceCaps) -> ProfileMatch:
        return _sm120_fp8_linear_profile_support(spec, caps)

    def _route(self, num_tokens: int) -> Fp8LinearDispatchRoute:
        for route in self.routes:
            if route.matches(num_tokens):
                return route
        raise RuntimeError(
            f"{self.name} has no prepared route for M={num_tokens}."
        )

    def _record_runtime_kernel_path(self, path: str) -> None:
        counts = self._runtime_kernel_path_counts.setdefault(
            path,
            {"eager_dispatches": 0, "cuda_graph_capture_dispatches": 0},
        )
        key = (
            "cuda_graph_capture_dispatches"
            if device_runtime.is_stream_capturing()
            else "eager_dispatches"
        )
        counts[key] += 1

    def runtime_kernel_stats(self) -> dict[str, object]:
        return {
            "kernel_paths": {
                path: dict(sorted(counts.items()))
                for path, counts in sorted(
                    self._runtime_kernel_path_counts.items()
                )
            },
            "fallback_reasons": {},
        }

    def binding_metadata(self) -> dict[str, object]:
        provenance = self.profile["provenance"]
        return {
            "implementation_kind": "dispatch_plan",
            "weight_layout_id": self.spec.weight_layout_id,
            "profile_id": self.profile["profile_id"],
            "profile_status": self.profile["profile_status"],
            "profile_source": dict(provenance),
            "profiled_toolchain": dict(self.profile["toolchain"]),
            "runtime_toolchain": dict(self.runtime_toolchain),
            "profile_toolchain_match": (
                self.runtime_toolchain == self.profile["toolchain"]
            ),
            "routes": [
                {
                    "min_tokens": route.min_tokens,
                    "max_tokens": route.max_tokens,
                    "provider": route.provider.name,
                    "provider_metadata": route.provider.binding_metadata(),
                }
                for route in self.routes
            ],
        }

    def __call__(self, x, weight, weight_scale_inv, bias=None):
        self._validate_call(x, weight, weight_scale_inv)
        num_tokens = int(x.numel()) // int(x.shape[-1])
        route = self._route(num_tokens)
        self._record_runtime_kernel_path(route.provider.name)
        return route.provider(x, weight, weight_scale_inv, bias)


def resolve_fp8_linear_provider(
    block_shape: tuple[int, int],
    *,
    input_features: int,
    output_features: int,
    activation_dtype: torch.dtype = torch.bfloat16,
    weight_dtype: torch.dtype = torch.float8_e4m3fn,
    scale_dtype: torch.dtype = torch.float32,
    weight_layout_id: str = "block_128x128_nt_k_major",
    cuda_graph: bool = True,
    device_index: int | None = None,
) -> Fp8LinearProvider:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    return OpResolver(FP8_LINEAR_REGISTRY).resolve(
        Fp8LinearSpec(
            tuple(int(value) for value in block_shape),
            input_features=int(input_features),
            output_features=int(output_features),
            activation_dtype=activation_dtype,
            weight_dtype=weight_dtype,
            scale_dtype=scale_dtype,
            weight_layout_id=str(weight_layout_id),
            cuda_graph=bool(cuda_graph),
        ),
        caps,
    ).provider
