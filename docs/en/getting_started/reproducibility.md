# Reproducibility

Use this page as the stable checklist for reproducing Sparse-Engine experiments.
Keep local run ledgers out of this repo; when a repo-facing result needs
evidence, cite the original run artifact path.

## Environment

The README contains the current install command. The expected baseline is:

- Python 3.10 or newer.
- Install one CUDA extra from `pyproject.toml`: `cu129` or `cu130`.
- Treat `pyproject.toml` as the current dependency contract; record the resolved
  package versions with each experiment because the repository does not ship a
  frozen environment lock.
- Optional matching `flashinfer-jit-cache` installed from the CUDA-specific FlashInfer
  wheel index when precompiled JIT modules are desired.
- `flash-attn` installed with `MAX_JOBS=8 pip install flash-attn --no-build-isolation`.
- Editable install from the repository root with `pip install -e ".[cu129]"`
  or `pip install -e ".[cu130]"`. Training, benchmark, and test dependencies
  are included in the main installation.
- Record the operator binding report, including `selected_provider`,
  `selected_profile`, `selection_basis`, provider metadata, and CUDA compute
  capability. Cite separate validation artifacts for numerical correctness and
  performance claims; an upstream default is not evidence of a local performance
  benchmark. FP8 providers do not download Hub kernels during warmup.
- RMSNorm defaults to `SPARSEENGINE_RMSNORM_PROVIDER=auto`, which prefers
  FlashInfer when installed. Set the variable to `triton` to force the local
  Triton kernel, or to `flashinfer` to require FlashInfer explicitly.

Record CUDA version, GPU type/count, visible GPU ids, branch, commit, and any
relevant uncommitted changes with every reported benchmark.

## Models And Checkpoints

Base models and DeltaKV compressor checkpoints must match. Public compressor
checkpoints are listed in the README section
[Download DeltaKV compressor checkpoints](README.md#deltakv-checkpoints).

Pass the downloaded local directory as `deltakv_checkpoint_path`. Current
loaders read local `model.safetensors` files; do not assume a Hugging Face repo
id can be passed directly everywhere.

## Data Paths

LongBench and MathBench read data roots from environment variables:

- `SPARSEENGINE_OUTPUT_DIR`: output root for benchmark predictions and logs.
- `SPARSEENGINE_DATA_DIR`: general benchmark dataset root.
- `SPARSEENGINE_LONGBENCH_DATA_DIR`: LongBench root containing `data/*.jsonl`.
- `SCBENCH_LOCAL_DATA_DIR`: optional local root for standard SCBench files.
- `SCBENCH_PREPROCESSED_ROOT`: root containing SCBench preprocessed `<task>.parquet` files.

Benchmark entrypoints do not assume host-specific dataset paths. If required
data roots or files are missing, the command should fail fast and print the
environment variable or CLI flag that must be set.

If a command uses local placeholders such as `<DATA_ROOT>`, `<MODEL_ROOT>`, or
`<OUTPUT_ROOT>`, rewrite them for the target machine and record the final paths
in the run record.

## Parameter Rules

Use canonical public parameter names:

- `sparse_method`
- `deltakv_checkpoint_path`
- `decode_keep_tokens`
- `sink_keep_tokens`
- `recent_keep_tokens`
- `full_attention_layers`
- `engine_prefill_chunk_size`

Use these names unchanged in commands, manifests, `LLM(...)`, and internal
configuration. Do not use legacy keys such as `engine_prefill_chunk_size`,
`sparse_method`, `model_cls`, `compressor_path`, `deltakv_checkpoint_path`,
`num_top_tokens`, or `seq_chunk_size`. See
[runtime-parameter-semantics.md](../configuration/runtime-parameter-semantics.md)
for the canonical contract.

Sparse-Engine requires explicit integer keep budgets. Do not pass ratios such as
`decode_keep_tokens=0.17`; convert each policy to a token count first.

## Smoke Checks

Start with small commands before long benchmarks:

```bash
PYTHONPATH=$PWD/src python scripts/benchmarks/bench_sparse_engine.py \
  --model_path <LOCAL_BASE_MODEL> \
  --lengths 1024 \
  --batch_sizes 1 \
  --methods vanilla \
  --output_len 4 \
  --hyper_params '{"gpu_memory_utilization":0.8,"engine_prefill_chunk_size":512}'
```

For a compressor-backed DeltaKV Sparse-Engine smoke:

```bash
PYTHONPATH=$PWD/src python scripts/benchmarks/bench_sparse_engine.py \
  --model_path <LOCAL_BASE_MODEL> \
  --lengths 1024 \
  --batch_sizes 2 \
  --methods deltakv \
  --output_len 4 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":512,"max_num_seqs_in_batch":2,"max_decoding_seqs":2,"max_num_batched_tokens":2048,"full_attention_layers":"0,1","sink_keep_tokens":4,"recent_keep_tokens":32,"decode_keep_tokens":64,"deltakv_checkpoint_path":"<LOCAL_COMPRESSOR_CHECKPOINT>","deltakv_center_ratio":0.1,"deltakv_neighbor_count":1,"deltakv_latent_dim":256,"deltakv_latent_quant_bits":4,"full_layer_kv_quant_bits":4,"enable_full_layer_kivi_quant":true,"deltakv_full_pool_reserve_ratio":0.2}'
```

Verify the loader logs that compressor weights were loaded. Omit the checkpoint
only in construction-only tests that explicitly set `allow_missing_deltakv_path`.

## Artifact Expectations

For reported results, save or record:

- Exact command and working directory.
- Runtime config and canonical sparse parameters.
- Model, tokenizer, checkpoint, precision, backend, and quantization settings.
- Dataset path, split, sample count, filtering/truncation, and seed.
- Raw outputs, parsed outputs, per-sample records, aggregate metrics, and
  run info when the benchmark supports them.
- Log paths and result file paths.
- Failure status and key error lines for failed or inconclusive runs.

Do not report a metric without a source log or result artifact.
