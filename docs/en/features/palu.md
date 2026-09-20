# Palu low-rank KV cache

Use `sparse_method="palu"` with an offline factor directory. Palu keeps all
tokens and compresses K and V into low-rank coordinates. Decode reconstructs
K inside the attention kernel and absorbs V reconstruction into the output
projection; it does not materialize a dense history cache.

## Prepare factors

Prepare a representative calibration corpus as JSONL, with a nonempty `text`
field in each row. Use held-out evaluation data separately. Run from the repo
root in the inference environment:

```bash
PYTHONPATH=src:. python scripts/compression/prepare_palu.py \
  --model "<MODEL_PATH>" --output "<PALU_FACTOR_DIR>" \
  --group-size 1 --rank-ratio 0.75 \
  --calibration-jsonl "<CALIBRATION_JSONL>" \
  --calibration-samples 32 --calibration-seqlen 1024
```

The tool performs activation-whitened grouped SVD and writes `palu.json` and
`palu.safetensors`. It records source projection hashes, calibration hashes,
actual token counts, and ranks. The original checkpoint is still required at
runtime. A factor directory belongs to that checkpoint and weight dtype;
architecture or source-weight mismatches fail at loading.

`--group-size` counts KV heads and must divide their number. The retained rank
is `floor(group_size * head_dim * rank_ratio / 16) * 16`, with a minimum of 16.
Supported ranks are multiples of 16 up to `min(group_size * head_dim, 256)`.
`--ranks-json` accepts one `[K_rank, V_rank]` pair per transformer layer;
groups within a layer use the same ranks. It overrides the ratio.

Calibration is required by default. `--uncalibrated` explicitly selects a
plain-SVD ablation that can severely degrade quality, even at modest compression.
The tool does not perform Fisher-based automatic rank allocation or latent
quantization. Rank ratio and calibration size are experiment parameters, not
quality guarantees.

## Run

```python
from sparseengine import LLM, SamplingParams

llm = LLM(
    "<MODEL_PATH>",
    sparse_method="palu",
    palu_checkpoint_path="<PALU_FACTOR_DIR>",
    tensor_parallel_size=1,
    max_model_len=8192,
    decode_graph=True,
)
outputs = llm.generate(["Explain why the sky is blue."],
                       SamplingParams(temperature=0, max_tokens=64))
llm.exit()
```

Supported: unquantized Llama and Qwen3, FP16/BF16, head dimension 64/128/256,
TP=EP=DP=1, chunked/batched prefill, eager and CUDA Graph decode. FlashInfer and
Triton are required. Prefix caching/offload, sparse-prefill combinations,
quantized model weights, and `tiny_random` are rejected.

Smaller caches do not guarantee faster inference. K reconstruction adds work,
and a V rank larger than the head dimension increases the fused output
projection. Use the [efficiency probe](../benchmarking/efficiency.md), passing
`palu_checkpoint_path` through `--hyper-params`, and compare quality and both
request and decode metrics against vanilla with matched settings.
