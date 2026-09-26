# AIME 2024: methods without auxiliary checkpoints

This recipe calls `benchmark/math_bench/pred.py` and its canonical `eval.py`
scorer. The default `setting.json` covers Vanilla, StreamingLLM, SnapKV, H2O,
PyramidKV, OmniKV, QuEST, RKV, KIVI, TurboQuant and FP8 KV. DeltaKV and SkipKV
are excluded because that campaign does not provision method-specific assets.
Aliases and independent `prefill_sparse_method` combinations are not separate
methods in this campaign.

## Settings

`setting.json` is the experiment configuration. Use the same
Qwen3-30B-A3B-Thinking-2507-FP8 checkpoint for every method (FP8 weights,
BF16 activations; KV representation depends on the method). GLM-4.7-Flash from the MiniSWE
recipe cannot cover the full matrix: quantized KV requires explicit KV and
does not support GLM MLA. Model compatibility is based on current source.
SnapKV and PyramidKV passed a short TP1 Qwen3 MoE GPU smoke with decode
eviction; the full TP2 recipe has not been GPU validated.

Inherited from `../chain_cache_miniswe/setting.json`: TP2/EP2/DP1, memory
fraction .95, engine concurrency limits 64, prefill chunk 4096, batch token
budget 65536, decode reservation 1024, CUDA Graph enabled, FP32 attention
scores, temperature .7, top_p 1 and maximum output 16384 tokens. H2O,
OmniKV and QuEST budgets are retained. SnapKV uses a new 4K retention budget
so its decode eviction activates within this output limit.

Explicit differences: context capped at 40960 for this campaign;
30 independent single-turn problems in one batch; prefix caching disabled
for every method (quantized KV does not support it). There are no tools,
agent retries, chain continuations or preserved previous thinking turns.
Thinking uses the tokenizer chat template with `enable_thinking=true`.
The MathBench `deepseek` problem prompt is retained, while synthetic output
think-prefix insertion and the extra user instruction to emit that prefix
are disabled. Seed is 42, top_k is 0. These are borrowed MiniSWE sampling
settings, not a claim to reproduce a published AIME score.

| Method | Budget / representation |
|---|---|
| Vanilla | Dense KV |
| StreamingLLM | 64 sink + 1984 recent |
| SnapKV | 64 sink + 512 recent + 3520 selected = 4096; window 32; decode eviction enabled, interval 1024 |
| H2O | Prefill 16384, decode 8192; eviction interval 128 |
| PyramidKV | Base budget 16384; layer ratios .6 to .01; decode eviction always enabled, interval 1024 |
| OmniKV | Selection 2048; model-profile full layers |
| QuEST | Selection 2048; chunks 16; first 2 layers full |
| RKV | Budget 16384; compression interval 128; observation 8 |
| KIVI | 4-bit, page size 32 |
| TurboQuant | 4-bit, page size 32, rotation seed 0 |
| FP8 KV | FP8, page size 32 |

These settings are not equal physical-memory budgets. Short AIME prompts
may not trigger prefill compression; retain this limitation when interpreting
SnapKV/PyramidKV scores. Decode compression differs by method. This v2 recipe
changes SnapKV's retention budget and PyramidKV's decode eviction timing; do
not aggregate its measurements with runs from the earlier v1 recipe.

## Qwen3-4B-Thinking-2507 campaign

`setting.qwen3-4b-thinking-2507.json` is an independent eight-method AIME
pass@1 protocol on one H100 (TP1/EP1/DP1). It submits all 30 problems together,
with batch/decode/resident sequence limits of 32, output cap 40960,
`max_model_len=41984`, Graph on, prefix caching off, and the same seed, prompt
and sampling settings above. The context limit leaves room for all 30 complete
prompts (at most 390 tokens with this tokenizer and template) plus the output.
Actual simultaneously resident decode requests may be lower than 30 if KV
capacity is exhausted; the runner records each method's execution separately.

| Method | Parameters |
|---|---|
| Vanilla | Dense KV |
| StreamingLLM | Sink 64, recent 2048, selected 0 |
| SnapKV | Sink 16, recent 64, selected 4096; window 16; no full layers; probability score; decode eviction interval 1024 |
| H2O | Prefill/decode budget 4096; decode eviction interval 1024; recent ratio .2; prefill score window 128; probability score |
| PyramidKV | Sink 16, recent 64, selected 7987 / 205 / 4096 at first / last / mean layer; ratios .6 to .0154712 across 36 layers; observation window 32; decode eviction interval 1024 |
| OmniKV | Sink 16, recent 64, selected 2048; Qwen3-4B-Thinking model profile full layers 0,3,9,13,16,21,28 |
| QuEST | Sink 16, recent 64, selected 2048; page size 16; first 2 layers use full attention |
| RKV | Sink 16, recent 64, selected 4096; compression interval 1024; observation 8; alpha .1; kernel size 7; score chunk 512 MB |

Existing result artifacts retain their recorded code/configuration. Rerun
SnapKV and PyramidKV before attributing those results to the decode query
observation-window implementation.

The PyramidKV selected-token mean is exactly 4096 after integer truncation;
sink and recent add 80 physical KV slots per layer. The
[QuEST paper](https://arxiv.org/html/2406.10774) states that
its first two layers use the full KV cache; `quest_skip_layers=2` matches that
choice. The selected-token fields above are separate from sink and recent
fields; their sums are the method's base retention or attention budget, where
applicable. Only these eight methods are included in this 4B config.

For the local GPU3 run, `scripts/tmp/run_aime4b_40k_gpu3_20260924.sh` starts
all eight full evaluations without a smoke run. The launcher stores status,
manifests and raw outputs under its `RUN_ROOT` on the persistent output disk.
The query-observation-window rerun uses
`scripts/tmp/run_aime4b_querywindow_aligned_gpu3_20260925.sh` for only SnapKV
and PyramidKV, with a fresh output root and the layer-budget check above.
Its validated result is in
[`results/2026-09-25-qwen3-4b-thinking-querywindow-aligned/RESULTS.md`](results/2026-09-25-qwen3-4b-thinking-querywindow-aligned/RESULTS.md).

`setting.qwen3-4b-thinking-2507-official-sampling.json` keeps the same eight
methods, budgets, 30-question batch, and 40960-token output cap, but follows
the [Qwen3-4B-Thinking-2507 model card](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)
for sampling: temperature .6, top_p .95, top_k 20, and min_p 0. The last value
does not filter tokens and needs no separate engine parameter. The model card
suggests a longer output cap for math competitions; this campaign retains the
previously requested 40960-token cap.

`setting.qwen3-4b-thinking-2507-official-sampling-c32c64-4try.json` repeats
each of the 30 questions four times with distinct request IDs. MathBench
submits all 120 requests in one `generate()` call per method; SparseEngine
queues them and enforces the configured batch, decode and GPU residency
limits. Vanilla, OmniKV and QuEST use 32; StreamingLLM, SnapKV, H2O,
PyramidKV and RKV use 64. The GPU3 launcher is
`scripts/tmp/run_aime4b_official_sampling_4try_c32c64_gpu3_20260925.sh`.
It supersedes the earlier three-seed launcher, which submitted each
30-question trial separately. Sampling uses seed 42 for the combined
120-request run; repeated questions receive different RNG draws as generation
advances. Compare each method's complete 120-request generation time for
end-to-end performance, and report quality by trial and by question.

`setting.qwen3-4b-thinking-2507-official-sampling-c32c64-2try.json` is the
follow-up GPU3 protocol. It keeps the same model, sampling, output cap,
sparse-method budgets and C32/C64 engine limits, but repeats each problem
twice. MathBench submits all 60 requests in one `generate()` call per method.
The launcher is `scripts/tmp/run_aime4b_official_sampling_2try_c32c64_gpu3_20260925.sh`.
The prior 120-request queue was superseded after Vanilla and StreamingLLM
completed; its partial artifacts remain separate. All eight methods,
including Vanilla, start fresh in the 60-request run.
The validated GPU3/4/5 result package is in
[`results/2026-09-25-qwen3-4b-thinking-official-sampling-2try-c32c64/RESULTS.md`](results/2026-09-25-qwen3-4b-thinking-official-sampling-2try-c32c64/RESULTS.md).

`setting.qwen3-4b-thinking-2507-official-sampling-c25-2try.json` reruns
Vanilla, OmniKV and QuEST with the same 60-request protocol and method
parameters, setting all three sequence limits to 25 and capturing the 25-way
decode Graph bucket. The GPU3 launcher runs Vanilla then QuEST; GPU4 runs
OmniKV. Each method still receives all 60 requests in one `model()` call.
The matched Vortex QuEST run and C25 results are in
[`results/2026-09-25-qwen3-4b-thinking-official-sampling-2try-c25-vortex-quest/RESULTS.md`](results/2026-09-25-qwen3-4b-thinking-official-sampling-2try-c25-vortex-quest/RESULTS.md).

The matched GPU2 C20 and C24 runs use the same 60-request protocol. Their
results are in
[`C20/RESULTS.md`](results/2026-09-25-qwen3-4b-thinking-official-sampling-2try-c20-vortex-quest/RESULTS.md)
and
[`C24/RESULTS.md`](results/2026-09-25-qwen3-4b-thinking-official-sampling-2try-c24-vortex-quest/RESULTS.md).
The additional C20 SparseEngine QuEST run raises its total sparse-layer budget
from 2128 to 3072 tokens while keeping all other C20 parameters; see
[`QuEST 3072/RESULTS.md`](results/2026-09-25-qwen3-4b-thinking-official-sampling-2try-c20-quest3072/RESULTS.md).

The pinned official R-KV vLLM port was run at C60 on the same 60 AIME requests;
see [`official vLLM R-KV C60/RESULTS.md`](results/2026-09-25-qwen3-4b-thinking-official-vllm-rkv-c60/RESULTS.md).
The local HiSparse QuEST C20 run completed with a quality anomaly; its result
and diagnostic boundary are in
[`HiSparse QuEST C20/RESULTS.md`](results/2026-09-25-qwen3-4b-thinking-hisparse-quest-c20/RESULTS.md).

## Run

Activate the inference environment first (including its compiler executables).
It needs the usual MathBench dependencies and `math-verify==0.9.0`.
Export `Maxwell-Jia/AIME_2024`, split `train`, to local JSON/JSONL with
`Problem` and `Answer` fields. Supply exactly 30 distinct problems; the driver
saves the normalized dataset and its checksum. Verify the export provenance;
row count alone cannot authenticate the dataset.

From the repository root, set `MODEL_PATH`, `AIME_DATA`, `RUN_ROOT` and
`GPU_PAIR` for your machine, then preview:

```bash
python scripts/official_experiments/aime/run.py \
  --model "$MODEL_PATH" --data "$AIME_DATA" \
  --output "$RUN_ROOT" --gpus "$GPU_PAIR"
```

Add `--execute` to run all 11 methods sequentially. `--methods vanilla quest`
selects a subset; use a new output directory for each invocation. Run long
jobs in tmux. The driver checks the selected GPUs for compute processes,
memory use and activity before each method, waiting up to 120 seconds for an
idle GPU. Method failures are recorded and the remaining methods continue;
the overall run then fails. It does not retry or overwrite existing runs. `--timeout` bounds each
method (default 43200 seconds). Perform GPU smoke validation before treating
the full matrix as runnable on a new device/model/toolchain.

## Artifacts and acceptance

The run root contains settings, normalized inputs, a Git commit/dirty-state
manifest, exact commands and per-method logs/engine configs. MathBench saves
raw predictions, parsed outputs, per-sample results and `result.json` under
each method's `benchmark/math_bench/pred/aime/` directory.

`final_summary.json` is written only after every selected method has all
expected unique request IDs (30 by default, 60 for two tries, 120 for four tries) and
consistent aggregate/per-sample results. Parse failures
count as incorrect in the denominator; invalid inputs or metric failures
reject the run. A failed run retains its manifest, logs and partial artifacts.
The wrapper checks artifacts even if the underlying entrypoint exits zero
after a scorer failure. This is one sampled answer per problem (pass@1),
not pass@k or a multi-seed estimate.

MathBench performance fields count re-tokenized output text over generation
call time. They are diagnostic and are not HTTP throughput, exact generated
token counts or isolated decode-stage throughput. Before making an acceleration
claim, record physical KV lengths, eviction counts, and the active decode batch
distribution; use the standardized efficiency probe for matched decode-window
and end-to-end measurements.

## Decode and end-to-end efficiency

Use the existing `benchmark/efficiency/bench_probe.py` for a separate synthetic
length-controlled comparison of Vanilla, SnapKV and PyramidKV. The 30 AIME
prompts in the recorded dataset tokenize to roughly 120 tokens at the median
and at most 390 tokens with this model and prompt template. The probe uses a
fixed 128-token input and 16384-token output, greedy decoding and ignored EOS;
it measures long-decode efficiency, not the stochastic AIME pass@1 workload.
Run each method on the same idle TP2/EP2 GPU pair with a fresh `RUN_ROOT`:

```bash
METHOD=snapkv  # repeat with vanilla and pyramidkv
METHOD_DIR="$RUN_ROOT/$METHOD"
mkdir "$METHOD_DIR"
python3 - "$METHOD" "$METHOD_DIR/engine.json" <<'PY'
import json
import sys
from pathlib import Path

setting = json.loads(Path("scripts/official_experiments/aime/setting.json").read_text())
method, output = sys.argv[1], Path(sys.argv[2])
output.write_text(json.dumps(setting["engine"] | setting["methods"][method], indent=2) + "\n")
PY

export CUDA_VISIBLE_DEVICES="$GPU_PAIR"
export PYTHONPATH="$PWD:$PWD/src"
python3 benchmark/efficiency/bench_probe.py \
  --engine sparseengine --sparse-method "$METHOD" --model-path "$MODEL_PATH" \
  --tensor-parallel-size 2 --expert-parallel-size 2 --monitor-gpus "$GPU_PAIR" \
  --gpu-memory-utilization 0.95 --max-num-batched-tokens 65536 \
  --scenario fixed --prompt-lens 128 --output-lens 16384 \
  --prompt-length-jitter 0 --output-length-jitter 0 --batch-sizes 1,4 \
  --decode-only-warmup-steps 4096 --decode-only-steps 2048 \
  --num-warmups 1 --num-iters 3 --seed 42 \
  --hyper-params "@$METHOD_DIR/engine.json" --output-dir "$METHOD_DIR/decode"
python3 benchmark/efficiency/bench_probe.py \
  --engine sparseengine --sparse-method "$METHOD" --model-path "$MODEL_PATH" \
  --tensor-parallel-size 2 --expert-parallel-size 2 --monitor-gpus "$GPU_PAIR" \
  --gpu-memory-utilization 0.95 --max-num-batched-tokens 65536 \
  --scenario fixed --prompt-lens 128 --output-lens 16384 \
  --prompt-length-jitter 0 --output-length-jitter 0 --batch-sizes 30 \
  --num-warmups 1 --num-iters 3 --seed 42 \
  --hyper-params "@$METHOD_DIR/engine.json" --output-dir "$METHOD_DIR/e2e"
```

The 4096 warmup plus 2048 measured decode steps include SnapKV's first
1024-step eviction intervals from the short prompt. The decode window requires
full residency at both batch sizes; a failed batch is not a throughput point.
The 30-request E2E case includes admission, prefill, decode, waiting and any
waves caused by capacity limits. Compare `output_token_throughput_tps` for E2E
and the validated continuous decode-window throughput for decode; inspect
`run_status.json`, `performance.jsonl` or `request_samples.jsonl`, and Graph
counter deltas before publishing a result. Save the compact final table with
device, commands and Git commit in this directory; keep raw probe artifacts on
the persistent data disk.
