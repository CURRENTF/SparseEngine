# DeltaKV Llama 3.1 8B LongBench v2

This run applies the spreadsheet's `LongBench v1` `deltakv_2048` setting to
Llama-3.1-8B-Instruct on LongBench v2. The spreadsheet has no DeltaKV v2
result; this is a new condition. Source workbook SHA256:
`5b512b091754367faf0aafd2bf8cb1f3b04c14abab962c461ee2d7fbcfab6f71`.

SparseEngine commit: `7b32be0fa67fc06d0df8bfabab5a2700f8825591` in an isolated
remote worktree. Hardware: 4 × RTX 4090, one independent TP1 process per GPU.
The base model is BF16 Llama-3.1-8B-Instruct. The DeltaKV compressor checkpoint
SHA256 (`model.safetensors`) is
`7e3ac589241556b1661b3e6200f61aae121461628db1a4bccd6a9a7cd861fe07`.
The requested runtime settings are in `runtime.json`. The checkpoint loader
also synchronizes its compressor architecture from the checkpoint config.

Protocol: official LongBench v2 zero-shot direct prompt, all 503 questions,
official pre-chat middle truncation at 120,000 tokens, max model length 131,072,
128 generated tokens, greedy decoding (`temperature=0`, `top_p=1`, `top_k=1`),
seed 20260901. The four data shards assign original index modulo 4. Each shard
uses the repository's native `benchmark/long_bench_v2/pred.py`; after all shards
finish, their per-sample rows are checked against the original IDs and
rescored together with `aggregate_results`. The source dataset SHA256 is
`15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2`.

For each shard, set `CUDA_VISIBLE_DEVICES` to its GPU index and run:

```bash
python -u benchmark/long_bench_v2/pred.py \
  --model-path "$MODEL_PATH" \
  --deltakv-checkpoint-path "$DELTAKV_CHECKPOINT_PATH" \
  --sparse-method deltakv --data-path "$SHARD_DATA" \
  --hyper-param-json runtime.json \
  --all-samples --overflow-policy official-middle \
  --truncate-max-tokens 120000 --preprocess-workers 4 \
  --max-model-len 131072 --max-new-tokens 128 --batch-size 1 \
  --temperature 0 --top-p 1 --top-k 1 --seed 20260901 \
  --output-dir "$SHARD_OUTPUT"
```

The validated aggregate and category scores are in [RESULTS.md](RESULTS.md)
and [results.json](results.json).

Remote raw outputs and logs:
`/root/autodl-fs/outputs/Sparse-vLLM/longbench_v2_deltakv_2048_20260925`.
The isolated remote code checkout is
`/root/autodl-tmp/Sparse-vLLM-deltakv-20260925`.
