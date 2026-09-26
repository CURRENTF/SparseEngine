# DeltaKV author HF LongBench on Llama 3.1 8B

Quality evaluation of the author DeltaKV Hugging Face implementation on
Llama-3.1-8B-Instruct (BF16), LongBench v1 and v2. Author source:
[`CURRENTF/DeltaKV`](https://github.com/CURRENTF/DeltaKV), commit
`11a837b21acfa475957210c5b2d339b89c1b7e3b`.

The run uses six independent TP1 processes on six RTX 4090 48 GiB GPUs. Each
process loads the BF16 base model and the same DeltaKV compressor checkpoint as
the existing SparseEngine v2 run. The checkpoint SHA256 is
`7e3ac589241556b1661b3e6200f61aae121461628db1a4bccd6a9a7cd861fe07`.
The method parameters are in [config.json](config.json). The author HF method
`delta_compressed_quant_kivi_full_fp8_ref` uses int4 latent and full-layer KV
compression, with a 2048-token decode keep budget.
The runtime used PyTorch `2.11.0+cu130` and Transformers `5.12.1`.

Frozen prompt token IDs and task IDs come from the matched LongBench baseline
preparation. v1 contains 3750 samples from 16 tasks; v2 contains 503 samples.
The frozen input shards are at
`/root/autodl-fs/datasets/SparseEngine-Baselines/prepared/turboquant-upstream-vllm-dp6-20260926`.
The six shards assign original evaluation index modulo six. The wrapper
[run_hf.py](run_hf.py) decodes each frozen prompt and verifies that tokenizing it
again yields exactly the input token IDs. Both benchmarks use greedy decoding,
`temperature=0`, `top_p=1`, `top_k=1`; v2 uses a 128-token output cap and seed
`20260901`. Maximum model length is 131072. The runner forces PyTorch cuDNN
SDPA for long-context memory use and uses 8192-token HF prefill chunks.

Two small changes to the author checkout were necessary for the installed
Transformers/PyTorch versions: the current `create_causal_mask` argument names,
and lower-right causal alignment for non-square chunked-prefill SDPA. The exact
patch and failed smoke logs remain with the remote raw artifacts. A 31K prompt
was checked against full prefill, and a 120K v2 prompt completed generation and
scoring before the full run.

The remote campaign launcher runs six shards per benchmark, verifies complete
sample coverage and successful per-sample status with [merge.py](merge.py), then
uses the frozen LongBench scorer. Raw outputs, logs, launcher, source patch and
per-shard manifests are on persistent storage at
`/root/autodl-fs/outputs/Sparse-vLLM/deltakv-hf-longbench-20260926`.

For each benchmark (`v1`, then `v2`) and GPU index `0` through `5`, the launch
arguments are:

```bash
source "$ENV_DIR/bin/activate"
export PYTHONPATH="$TRANSFORMERS_OVERLAY:$AUTHOR_REPO/src"
export DELTAKV_HF_ATTN_IMPLEMENTATION=sdpa
export PYTORCH_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES="$GPU" python -u run_hf.py \
  --model "$MODEL_DIR" --checkpoint "$COMPRESSOR_DIR" \
  --input "$FROZEN_INPUT_DIR/$BENCHMARK-shard-$GPU.jsonl" \
  --output-dir "$RUN_DIR/$BENCHMARK/shard-$GPU" \
  --config config.json --benchmark "$BENCHMARK" \
  --source-commit 11a837b21acfa475957210c5b2d339b89c1b7e3b \
  --sdpa-kernel cudnn
```

Compared with the prior SparseEngine v2 condition, this author HF run enables
sparse-reference FP8 and uses the author's cache implementation. The earlier
SparseEngine runtime disabled sparse-reference FP8. See
[RESULTS.md](RESULTS.md) for scores and the other protocol differences.
