# Qwen3-4B-Thinking-2507 AIME 2024: upstream HF R-KV control

Status: stopped before completion. R-KV ran on one H100 80GB (GPU2), and the
Hugging Face FullKV control ran on another H100 80GB (GPU4). Each completed
7 of 60 sequential requests. FullKV was stopped at the user's request; R-KV
was stopped after the user requested a batched run with the official vLLM port.
Neither partial run has a reportable AIME score.
The upstream R-KV checkout is `Zefan-Cai/R-KV` at
`6715468b9872442be72e5c97322e4d9c9a2abf55`; this adapter runs from
SparseEngine commit `7b32be0fa67fc06d0df8bfabab5a2700f8825591`.
The environment has Transformers 4.51.3, PyTorch 2.8.0, FlashAttention 2.8.3,
and `math-verify==0.9.0`.

Both HF runs use the exact 60 prompt token sequences and gold rows from the
[two-try AIME comparison](../2026-09-25-qwen3-4b-thinking-official-sampling-2try-c32c64/RESULTS.md):
30 AIME 2024 train problems, two samples each. The samples run sequentially
with per-request seeds `42 + request_id`, temperature 0.6, top-p 0.95,
top-k 20, no min-p filter, at most 40,960 new tokens, and a 41,984-token
context limit. The prompt and dataset are shared with SparseEngine; sampling
RNG and execution order differ, so score differences are not paired-token
comparisons. MathBench scores both HF outputs with the same parser and metric.

R-KV uses the upstream Qwen3 monkeypatch with a total per-head retained budget
of 4,176 tokens, an 8-query observation window, kernel size 7, mixing weight
0.1, and compression at each 1,024-token total-length boundary. This matches
the prior serving run's total budget of 16 + 64 + 4,096 = 4,176 tokens, but the
upstream HF implementation selects independently per KV head and does not
reserve the serving run's sink/recent partitions. Before each request, the
adapter resets the upstream model's stored sequence length and compression
flag so previous requests cannot shift the next request's compression schedule.
It checks the returned per-layer KV lengths and fails if a sufficiently long
R-KV generation never evicts.

Launch: `scripts/tmp/start_aime4b_hf_rkv_20260925.sh`, which starts one
`scripts/tmp/run_aime4b_hf_rkv_method_20260925.sh` tmux session per method.
The sessions are `aime4b-hf-rkv-20260925` and
`aime4b-hf-fullkv-20260925`; each method has its own `*.status.tsv` and
`*.run.log` in the raw run root.
Generation entrypoint: `scripts/official_experiments/aime/run_rkv_hf.py`.
Raw manifests, partial per-request outputs, logs, and status files are under
`/data2/haojitai/outputs/Sparse-vLLM/aime_qwen3_4b_hf_rkv_vs_fullkv_20260925`.

The interrupted HF runs are retained for diagnosis only. A complete 60-request
run is required before reporting quality. Their sequential generation time is
not a serving end-to-end speed comparison.
