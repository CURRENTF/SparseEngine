import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


class _SwiGLUCompressor(nn.Module):
    def __init__(self, input_size: int, intermediate_size: int, output_size: int, bias: bool = True):
        super().__init__()
        self.w12 = nn.Linear(input_size, intermediate_size * 2, bias=bias)
        self.w3 = nn.Linear(intermediate_size, output_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


def _normalize_compressor_type(kind: str) -> str:
    if kind is None:
        return "auto"
    kind = str(kind).lower().strip()
    aliases = {
        "": "auto",
        "auto": "auto",
        "linear": "linear",
        "mlp": "mlp_gelu",
        "gelu": "mlp_gelu",
        "mlp_gelu": "mlp_gelu",
        "swiglu": "mlp_swiglu",
        "mlp_swiglu": "mlp_swiglu",
    }
    if kind in aliases:
        return aliases[kind]
    raise ValueError(f"Unknown compressor type: {kind}. Use auto|linear|mlp_gelu|mlp_swiglu.")

def create_compressor(is_down: bool, config, bias_override: Optional[bool] = None):
    """
    为每个压缩器组创建独立的序列压缩和解压层。
    假设 split_kv=False；支持新/旧版本配置：
    - 旧版（对称）：use_nonlinear_compressor + compressor_intermediate_size + compressor_linear_bias
    - 新版（可非对称）：compressor_down/up_type + compressor_down/up_intermediate_size
    """
    hf_config = config.hf_config
    head_dim = getattr(hf_config, "head_dim", None) or (hf_config.hidden_size // hf_config.num_attention_heads)
    kv_factor = 2 # 假设 split_kv 恒为 False
    
    # 计算输入输出维度
    kv_dim = head_dim * hf_config.num_key_value_heads * kv_factor
    input_size = kv_dim if is_down else config.deltakv_latent_dim
    output_size = config.deltakv_latent_dim if is_down else kv_dim

    # 支持 compressor_linear_bias 参数
    bias = getattr(config, 'compressor_linear_bias', True) if bias_override is None else bool(bias_override)

    kind_attr = "compressor_down_type" if is_down else "compressor_up_type"
    kind = _normalize_compressor_type(getattr(config, kind_attr, "auto"))
    if kind == "auto":
        kind = "mlp_gelu" if getattr(config, "use_nonlinear_compressor", True) else "linear"

    if kind == "linear":
        return nn.Linear(input_size, output_size, bias=bias)

    inter_attr = "compressor_down_intermediate_size" if is_down else "compressor_up_intermediate_size"
    intermediate_size = getattr(config, inter_attr, -1)
    if intermediate_size <= 0:
        intermediate_size = getattr(config, 'compressor_intermediate_size', 2048)
    if intermediate_size <= 0:
        intermediate_size = (input_size + output_size) // 2

    if kind == "mlp_gelu":
        return nn.Sequential(
            nn.Linear(input_size, intermediate_size, bias=bias),
            nn.GELU(),
            nn.Linear(intermediate_size, output_size, bias=bias),
        )

    if kind == "mlp_swiglu":
        return _SwiGLUCompressor(input_size, intermediate_size, output_size, bias=bias)

    raise ValueError(f"Unhandled compressor type after normalization: {kind}")
