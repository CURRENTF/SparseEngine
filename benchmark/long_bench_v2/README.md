# LongBench v2

This directory keeps LongBench v2 separate from the existing LongBench v1
runner under `benchmark/long_bench/`.

The official THUDM repository is pinned as the `upstream/` Git submodule. The
native runner uses its official zero-shot prompt and records the submodule
commit in every result. Initialize it after cloning:

```bash
git submodule update --init benchmark/long_bench_v2/upstream
```

The official dataset is distributed separately through Hugging Face and is not
duplicated in the source submodule. Export the `zai-org/LongBench-v2` train
split (the former `THUDM/LongBench-v2` alias redirects there) to one local JSON
or JSONL file, then set:

```bash
export SPARSEENGINE_LONGBENCH_V2_DATA=<LONGBENCH_V2_JSON_OR_JSONL>
```

`pred.py` runs the native Sparse-Engine engine. With `--token-buckets-json`, it
selects a deterministic subset in configured post-chat-template token buckets
without truncating source prompts. A bucket with insufficient samples that fit
the requested model budget fails explicitly. It saves the selected identities and hashes, raw responses,
parsed answers, per-sample statuses, aggregate metrics, runtime configuration,
and source/submodule provenance.

Use `--all-samples` instead of `--token-buckets-json` to evaluate every row in
the input dataset (503 rows for the complete official export). This mode cannot
be combined with `--official-length` and requires `--overflow-policy official-middle`
with an explicit `--truncate-max-tokens` budget.

For the official Hugging Face tokenizer truncation procedure, use:

```bash
python benchmark/long_bench_v2/pred.py \
  --model-path "$MODEL_PATH" \
  --data-path "$SPARSEENGINE_LONGBENCH_V2_DATA" \
  --sparse-method vanilla \
  --all-samples \
  --overflow-policy official-middle \
  --truncate-max-tokens 120000 \
  --max-model-len 131072 \
  --max-new-tokens 128 \
  --temperature 0.1 --top-p 1 --top-k 0 \
  --output-dir "$OUTPUT_DIR"
```

This follows pinned `upstream/pred.py::query_llm`: encode the complete plain-text
prompt with the tokenizer's default special-token behavior, retain the beginning
and end if it exceeds the explicit truncation limit, decode with
`skip_special_tokens=True`, then apply the chat template and tokenize for inference.
Prompts within the limit pass through without a decode round trip. The upstream
`config/model2maxlen.json` uses 120000 for its Qwen2.5/Llama 128K-class models;
the CLI requires an explicit limit rather than inferring it from a model path.
The truncation limit is independent of the runtime context window and output
budget. If the resulting chat input plus reserved generation exceeds the runtime
limit, the run fails; it does not silently truncate again or exclude the sample.

The example uses the paper's direct-answer temperature and output length;
`--top-k 0` disables the runner's default top-k=1 restriction. Chat templates and
other sampling defaults still need to match the compared deployment. For OmniKV,
change `--sparse-method` and supply its model-specific `--hyper-param-json`.

Full-mode artifacts record original/effective prompt lengths, truncation flags,
effective token-ID hashes, and the number of truncated samples. In official-middle
mode, `original_prompt_tokens` and `original_prompt_sha256` describe the original
plain-text prompt before chat templating; `prompt` and `prompt_sha256` describe
the effective chat prompt. Inference always uses the effective `prompt_token_ids`.
Keep the same model, input budget, overflow policy,
and generation settings across vanilla and sparse runs. `--prepare-only` and
`--prepared-samples` also support full mode and bind reuse to its prompt budget
and overflow policy, including the pre-chat limit in official-middle mode.

`--preprocess-workers N` parallelizes full-dataset tokenization, middle truncation,
and chat preparation across CPU processes. Full-dataset preparation defaults to
8 workers; pass 1 for serial preparation. Token-bucket subsets remain serial.
Workers load separate tokenizers, so increasing the count also increases RAM usage.
The queue is bounded, source order and token IDs are preserved, and any worker
failure fails the run. Progress is printed as samples finish. This setting does
not change GPU parallelism or generation batch size and is recorded in the run
configuration without changing prepared-sample identity. When `--prepared-samples`
is supplied, preparation is skipped regardless of the worker count. Preparing
once and reusing the export across methods avoids repeating this CPU work.

For paired external-runtime evaluation, `--engine vllm` accepts constructor
options through `--engine-kwargs`, and `--engine sglang-http --server-url ...`
uses an already-running server with matching model identity and disabled radix
cache. Both receive the exact prepared token IDs and use the same answer parser
and aggregate metrics as the native runner. Preserve the external server launch
configuration separately; a matching method label alone does not establish
algorithm or budget parity.

`--official-length medium` filters the dataset's official length label before
token-bucket selection. `--prepare-only` exports prepared samples for reuse with
`--prepared-samples`; model/tokenizer, dataset, prompt, seed and bucket identities
must match. Preparation is not a model evaluation. If a common context-limited
subset is used, retain excluded sample IDs and reasons and label its accuracy as
a subset score, not the complete official length-cohort score. Full runs retain
incremental per-sample output in `sample_results.partial.jsonl`.

As in the official evaluator, a non-empty model response that does not contain
the required answer pattern is retained with `status="parse_failed"` and scored
as incorrect. Model/runtime failures remain fatal and invalidate the run.

The canonical 120-sample, greedy-decoding profile is a repository regression
gate, not a reproduction of the full 503-sample LongBench v2 leaderboard.

Use `benchmark/sparseengine_regression/run_suite.py --layer longbench_v2` for the
canonical gate. The default quality layer runs LongBench v1, LongBench v2, and
RULER; use `--quality_benchmarks` only for an intentional focused run.
