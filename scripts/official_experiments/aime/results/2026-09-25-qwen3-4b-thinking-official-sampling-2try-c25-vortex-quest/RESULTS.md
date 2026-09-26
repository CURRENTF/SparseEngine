# Qwen3-4B-Thinking-2507 AIME 2024: C25 and Vortex QuEST

Device: one NVIDIA H100 80GB HBM3 per run. Vanilla and SparseEngine QuEST ran
on GPU3; OmniKV ran on GPU4; Vortex QuEST ran on GPU3. All used TP1 and one
60-request generation call: 30 AIME 2024 problems, two samples per problem.
The normalized dataset SHA-256 is
`bfe60715bc6fd32f1f6438c9a479cd5b6f4daaaa6aa6255b375b81a7ea84c136`.
The Vortex input token IDs were built with the same MathBench prompt functions
and SparseEngine tokenizer; all 60 prompt lengths matched Vortex's reported
prompt lengths. Sampling used temperature 0.6, `top_p=0.95`, `top_k=20`,
`min_p=0`, seed 42, at most 40960 new tokens and context length 41984.
Prefix caching was disabled, and CUDA Graph was active. The concurrency cap
was 25 for every method.

SparseEngine ran at commit `7b32be0fa67fc06d0df8bfabab5a2700f8825591`;
Vortex v0.6 ran at commit `ab9ac68c7b9b81f6ba17752741f9e9d92444bf56`.
The [SparseEngine setting](setting.requested.json) and
[Vortex setting](vortex_quest.setting.json) record the exact parameters.

| Method | GPU | Correct / 60 | Trial 0 / 30 | Trial 1 / 30 | At least one correct / 30 | Generation time (s) | vs Vanilla | vs SparseEngine QuEST |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 3 | 47 | 22 | 25 | 27 | 1186.8 | 1.00× | — |
| OmniKV | 4 | 49 | 25 | 24 | 26 | 997.7 | 1.19× | — |
| SparseEngine QuEST | 3 | 39 | 20 | 19 | 21 | 1274.8 | 0.93× | 1.00× |
| Vortex QuEST | 3 | 45 | 22 | 23 | 23 | 1199.6 | 0.99× | 1.06× |

Each ratio is reference time divided by method time, with no output-token
normalization. The generation time is
the wall time of the complete 60-request call, excluding startup, prompt
preparation and answer scoring. SparseEngine synchronizes the GPU at call
boundaries; Vortex times the blocking `Engine.generate()` call through its
final response. All methods have 60 raw and scored records;
all 60 Vortex scores have status `success`. Vortex generated 1,126,092 actual
completion tokens; four requests reached the 40960-token cap. Its per-request
metadata reports zero retractions. SparseEngine logged one recompute
preemption for QuEST, one for OmniKV, and none for Vanilla.

Vortex kept 128 selected pages, one initial page and four recent pages at
16 tokens per page: 2048 + 16 + 64 = 2128 attended tokens. It skipped sparse
selection in layers 0 and 1. SparseEngine QuEST used the same 2128-token
total budget and skipped the same layers, but its current selection path does
not force the initial and recent pages to remain. This is a matched workload
and capacity comparison, not identical page-selection semantics. Vortex used
`mem_fraction_static=0.9`; SparseEngine used
`gpu_memory_utilization=0.95`, which are different engine allocation controls.

Vortex's GQA decode used FlashInfer paged attention and a Triton-compiled
QuEST indexer. Because `vortex_max_topk_val` was unset, the indexer's
`topk_output` used the `flashinfer/default` CUB-sort CUDA implementation rather
than the `k_128` specialization. The performance effect of this dispatch choice
was not isolated.

## Launch and artifacts

The final native runs used
`scripts/tmp/run_aime4b_official_sampling_2try_c25_gpu34_20260925.sh`
with GPU3 (`vanilla quest`) and GPU4 (`omnikv`). The launcher called
`scripts/official_experiments/aime/run.py` through
`conda run --no-capture-output -p /data2/haojitai/conda_envs/sparse-vllm-cu130-py312`
with `--model /data2/pretrain_models/Qwen3-4B-Thinking-2507`,
`--setting setting.requested.json`, `--data aime2024.json`,
`--gpus 3` or `--gpus 4`, `--gpu-wait-timeout 21600`, and `--execute`.
Its exact per-method commands are in the raw manifests.

The Vortex run used
`scripts/tmp/run_aime_vortex_quest_c25_20260925_retry2.sh` in a separate
`systemd-run --user` unit. It invoked
`scripts/official_experiments/aime/run_vortex_quest.py` in the `vortex_glm`
environment with `--concurrency 25 --max-model-len 41984`
`--max-new-tokens 40960 --mem-fraction-static 0.9 --seed 42`. The
`prepare_vortex_inputs.py` step exported the same 60 prompt token sequences.
`benchmark/math_bench/eval.py` scored all four methods with
`math-verify==0.9.0`. See [compact results](results.json) for the validated
artifact and manifest locations. The raw runs remain under
`/data1/haojitai/outputs/Sparse-vLLM/`.
