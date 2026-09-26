# Llama-3.1-8B Palu versus Hugging Face FullKV

Device: 4 × NVIDIA GeForce RTX 4090 48 GB. Both use the same BF16 model,
benchmark data, prompt templates, and tokenizer. Hugging Face uses FlashAttention 2,
single-request generation,
and an 8192-token MLP prefill chunk to bound memory. This is a quality comparison;
generation time is not a matched serving-throughput comparison.

| Benchmark | Samples | HF FullKV | SparseEngine Palu | Palu − HF |
|---|---:|---:|---:|---:|
| LongBench v1, six-category average | 3750 | 50.47 | 41.77 | -8.70 |
| LongBench v2, accuracy (%) | 503 | 29.03 | 28.03 | -0.99 |

Palu uses grouped rank-96 factors (group size 1), activation whitening, TP1,
batch size 1, prefill chunk 4096, and CUDA Graph decode. The factors were prepared
from 32 calibration sequences capped at 1024 tokens each. FullKV retains every token.

LongBench v1: max model length 121000; all 16 English tasks and 3750 samples;
official per-task output limits; temperature 0, top-p 1, top-k 1, seed 42.
LongBench v2: max model length 131072; all 503 samples; official middle truncation
to 120000 pre-chat tokens; output limit 128; temperature 0, top-p 1, top-k 1,
seed 20260901. The v2 prompt-token identity file matched byte for byte across
the two runs. The v1 prompt token lengths matched for all 3750 examples.

HF FullKV ran on GPUs 0–2 in two v1 and three v2 shards. Palu v1 ran on GPUs
0–2 and Palu v2 on GPU 3. The raw outputs and manifests are under
`/root/autodl-fs/outputs/Sparse-vLLM/hf-fullkv-palu-compare-20260925` and
`/root/autodl-fs/outputs/Sparse-vLLM/palu-llama31-longbench-20260925`.

SparseEngine commit: `a07c3e30be63646137cc6aeeff131adc987afe5d`. KVCache-Factory upstream commit:
`68cd9551a63fedf362ddb0edb008badd04c24996`. The exact launch arguments,
dependency versions, and runner identity are in the run manifests at the raw
artifact root.
