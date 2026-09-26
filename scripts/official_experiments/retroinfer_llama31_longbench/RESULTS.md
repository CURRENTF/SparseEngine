# RetroInfer GPU-only on Llama 3.1 LongBench

Device: NVIDIA H100 80GB HBM3, one GPU per run (GPU 4 or 5). Model: Llama-3.1-8B-Instruct, BF16, greedy decoding. Both implementations used the same frozen token inputs and external scorers. Retrieval ratio was 0.018 and estimation ratio was 0.232.

Source revisions: SparseEngine base `7b32be0fa67fc06d0df8bfabab5a2700f8825591`; RetroInfer author implementation `03f912c6e917c380d9d90c5ec85bb0f161ba53ef`; LongBench scorer `2e00731f8d0bff23dc4325161044d0ed8af94c1e`. The SparseEngine base revision precedes the native RetroInfer integration and alone cannot reproduce it.

## LongBench v1

Full 3,750 samples, 16 tasks. Metric: `overall_category_avg`, points; larger is better. All samples completed without execution failures.

| Variant | Overall | SDQA | MDQA | SUM | FewShot | Syn | Code |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native GPU-only | 50.78 | 43.61 | 46.54 | 28.85 | 69.35 | 55.88 | 60.45 |
| Author GPU-only | 50.60 | 43.38 | 46.43 | 28.95 | 69.37 | 55.55 | 59.92 |
| Native minus author | +0.18 | +0.23 | +0.11 | -0.10 | -0.02 | +0.33 | +0.53 |

## LongBench v2

Full 503 samples. Metric: accuracy, percent; larger is better. All samples completed without execution failures. Parse failures count as incorrect: 24 native, 23 author.

| Variant | Overall | Short | Medium | Long | Correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native GPU-only | 29.22 | 33.33 | 26.51 | 27.78 | 147/503 |
| Author GPU-only | 29.62 | 33.33 | 26.98 | 28.70 | 149/503 |
| Native minus author | -0.40 | 0.00 | -0.47 | -0.93 | -2 |

## Launch arguments

Set `MODEL`, `LONGBENCH_V1_DATA`, `LONGBENCH_V2_DATA`, `PREPARED_V1`, `PREPARED_V2`, `AUTHOR_REPO`, and `OUT` for the target machine. Run inside the matching conda environment with `PYTHONPATH=src:.`. The native configuration files in this directory contain the full engine settings. `PREPARED_V1` and `PREPARED_V2` are the frozen paired inputs used by the author adapter; the native results were normalized to the same inputs before external scoring.

| Run | Entrypoint and material arguments |
| --- | --- |
| Native v1 | `benchmark/long_bench/pred.py --model llama31_retroinfer --model_path "$MODEL" --sparse_method retroinfer --batch_size 500 --max_model_len 121000 --hyper_param native_v1_hparams.json --output_root "$OUT"` with `SPARSEENGINE_LONGBENCH_DATA_DIR="$LONGBENCH_V1_DATA"` |
| Native v2 | `benchmark/long_bench_v2/pred.py --model-path "$MODEL" --data-path "$LONGBENCH_V2_DATA" --sparse-method retroinfer --all-samples --overflow-policy official-middle --truncate-max-tokens 120000 --max-model-len 131072 --max-new-tokens 128 --temperature 0 --top-p 1 --top-k 1 --seed 20260901 --batch-size 16 --prepared-samples "$PREPARED_V2" --hyper-param-json native_v2_hparams.json --output-dir "$OUT"` |
| Author v1 | `benchmark/retroinfer/author_longbench.py --benchmark longbench_v1 --author-repo "$AUTHOR_REPO" --model-path "$MODEL" --input "$PREPARED_V1" --output-dir "$OUT" --max-model-len 121000 --max-batch 16` |
| Author v2 | `benchmark/retroinfer/author_longbench.py --benchmark longbench_v2 --author-repo "$AUTHOR_REPO" --model-path "$MODEL" --input "$PREPARED_V2" --output-dir "$OUT" --max-model-len 131072 --max-batch 16` |

The author v1 run used a copy of the original author revision with a guard for division by zero in a latency print when generation has zero decode steps. Its model computation was unchanged. The original author revision ran v2. The author's equal-length batch requirement resulted in mostly single-sample batches, so these results compare quality and do not establish relative throughput.

## Native v1 concurrency diagnostic

The same 64 `multifieldqa_en` samples were run on GPU 5 with only the native row limits changed. `--batch_size 64`, `--num_samples 64`, `--max_model_len 121000`, and the v1 configuration above were held fixed. All 64 samples succeeded in each run.

| Row limit | Whole-process elapsed, s | Mean sampled GPU Util, % |
| ---: | ---: | ---: |
| 16 | 88.48 | 47.6 |
| 24 | 75.27 | 55.3 |
| 32 | 76.54 | 53.6 |

Elapsed time includes model loading and scoring. GPU Util is the sampled `nvidia-smi` activity percentage, not SM occupancy or theoretical utilization. This probe establishes that 24 and 32 rows completed on these inputs; it is not a serving-throughput benchmark.

Compact table data: [results.json](results.json). Raw per-sample outputs, resolved configurations, logs, and paired scoring artifacts remain under the experiment output root on persistent storage.
