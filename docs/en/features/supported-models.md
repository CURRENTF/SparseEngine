# Supported Models

This page summarizes the model, precision, parallelism, and sparse-method
combinations supported by Sparse-Engine.

`Precision` describes the checkpoint weight format. `TP`, `DP`, and `EP`
stand for tensor, data, and expert parallelism. A check mark means that the
mode is supported; a numeric restriction means that the corresponding
parallel size must use that value.

## Model and Parallelism Support

| Model | `model_type` | Precision | TP | DP | EP |
| --- | --- | --- | :---: | :---: | :---: |
| Qwen2.5 | `qwen2` | BF16 / FP16 / block FP8 | ✅ | 1 only | 1 only |
| Qwen3 Dense | `qwen3` | BF16 / FP16 / block FP8 | ✅ (FP8: 1/2/4/8) | 1 only | 1 only |
| Qwen3MoE | `qwen3_moe` | BF16 / FP16 / block FP8 | ✅ | ✅ | ✅ |
| Qwen3.5 / 3.6 / 3.8 | `qwen3_5` | BF16 / block FP8 | ✅ | 1 only | 1 only |
| Qwen3.6 MoE | `qwen3_5_moe` | BF16 / block FP8 | ✅ | 1 only | ✅ |
| GLM-4.7-Flash | `glm4_moe_lite` | BF16 / experimental per-tensor FP8 | ✅ | ✅⁵ | ✅ |
| Gemma 4 Dense / MoE | `gemma4` | BF16 / FP16 | ✅ | 1 only | ✅ (MoE only) |
| Llama 3 / 3.1 | `llama` | BF16 / FP16 / block FP8 | ✅ | 1 only | 1 only |
| MiniMax M2.7 | `minimax_m2` | block FP8 with BF16 non-quantized weights | ✅ | ✅ | ✅ |

TP is limited to sizes 1 through 8 and requires the checkpoint dimensions,
including the attention heads and vocabulary size, to be divisible by the
selected TP size.

`tensor_parallel_size` and `data_parallel_size` describe attention: the total
process count is `world_size = DP * TP`. `expert_parallel_size` describes routed
expert placement within that same world; expert TP is `world_size / EP`, so EP
must divide the world size. EP does not add processes. For example, `TP=4,
DP=1, EP=2` uses four processes with attention TP=4 and expert TP=2.

GLM, Qwen3MoE, and MiniMax support attention TP×DP with `moe_backend="agrs"`
and `EP=world_size` (expert TP=1). For example, `TP=2, DP=2, EP=4` uses four
GPUs; valid larger topologies are accepted subject to model and hardware limits.
Dense MLPs and shared experts use attention TP within each DP replica.
Each replica owns its scheduler and prefix cache; decode CUDA Graphs are supported.
Existing sparse-method and prefix-cache restrictions still apply.

AG/RS is the default DP transport. `moe_backend="deepepv1"` still requires
attention TP=1 and supported NVLink hardware; unsupported combinations fail at
startup. With DP=1, the default remains `all-reduce`.

Block FP8 support requires E4M3 weights, dynamic activation quantization, and
a `128 x 128` weight block size. Llama, Qwen2, and Qwen3 dense FP8 checkpoints
also require every TP-local dense projection dimension to be 128-aligned and
keep non-quantized parameters in BF16.

GLM per-tensor FP8 loading accepts E4M3 weights with a BF16 scalar
`weight_scale` for each quantized projection, as used by
`marksverdhei/GLM-4.7-Flash-FP8`. It requires dynamic activation quantization
and a device with a compatible native FP8 provider. Weight precision does not
change the MLA KV-cache dtype. This experimental path does not imply a
performance improvement over BF16; compare matched workloads before deployment.

## Sparse Method Support

| Model | Vanilla | StreamingLLM | SnapKV | H2O | PyramidKV | OmniKV | QuEST | R-KV | SkipKV | DeltaKV |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5 | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | Selected checkpoints¹ | Compressor required² |
| Qwen3 | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | — | Compressor required² |
| Qwen3MoE | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | — | — |
| Qwen3.5 / 3.6 / 3.8 | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | — | Matched checkpoint³ |
| Qwen3.6 MoE | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | — | — |
| GLM-4.7-Flash | ✅ | ✅ | ✅ | Experimental⁴ | — | ✅ | ✅⁵ | ✅ | — | — |
| Gemma 4 Dense / MoE | ✅ | ✅⁶ | — | — | — | ✅ | — | — | — | — |
| Llama 3 / 3.1 | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | Selected checkpoint¹ | Compressor required² |
| MiniMax M2.7 | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | — | — |

¹ SkipKV is limited to the released steering-vector models:
`DeepSeek-R1-Distill-Qwen-7B`, `DeepSeek-R1-Distill-Qwen-14B`, and
`DeepSeek-R1-Distill-Llama-8B`.

² DeltaKV requires a compressor checkpoint trained for the base model.

³ Qwen3.5, Qwen3.6, and Qwen3.8 require a matching DeltaKV checkpoint accepted
by the mixed-attention runtime.

⁴ H2O tensor-parallel execution may produce different sparse selections from
TP=1. Model-specific TP, EP, and DP restrictions still apply.

⁵ GLM QuEST supports radix prefix caching (including CPU offload) and decode
CUDA Graph. Selection uses TP-local heads; equivalence with TP=1 is not
guaranteed. CPU offload supports attention TP=1 or TP=2.

⁶ Gemma 4 checkpoints with shared KV layers reject per-layer StreamingLLM
eviction. Vanilla and OmniKV remain supported.

## Native Multimodal Support

Set `enable_multimodal=True` to accept supported image and video inputs through
the OpenAI-compatible Chat and Responses APIs. Unsupported media return an error.

| Model family | Image | Video | Audio |
| --- | :---: | :---: | :---: |
| Qwen3.5 / 3.6 / 3.8 Dense and Qwen3.5 / 3.6 MoE | ✅ | ✅ | — |
| Gemma 4 Dense and MoE | ✅ | ✅ | — |

`—` means that the combination is not currently supported.
