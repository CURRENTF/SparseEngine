# SWE-bench Lite

## Purpose

`benchmark/swe_bench_lite/run.py` is a thin adapter around two external,
upstream-owned components:

- mini-SWE-agent generates a patch for each issue through an OpenAI-compatible
  model API.
- `swebench.harness.run_evaluation` applies each patch and computes the official
  SWE-bench result in Docker.

The adapter does not reimplement SWE-bench datasets, environments, tests, or
metrics. It supports `SWE-bench/SWE-bench_Lite`, split `test`, including the
full 300-instance benchmark and smaller smoke selections.

Upstream sources:

- [SWE-bench](https://github.com/SWE-bench/SWE-bench)
- [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent)
- [SWE-bench Lite dataset](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite)

## Prerequisites

Use a separate Python environment that contains both `mini-swe-agent` with its
SWE-bench extra and the official SWE-bench harness. Keep the upstream checkout
outside this repository, for example at `../SWE-bench`.

Set `SWE_BENCH_PYTHON` to that environment's Python. The shell entrypoint also
puts the same environment's `bin/` directory first on `PATH`, so `mini-extra`
and the imported SWE-bench harness cannot accidentally come from different
environments.

The adapter defaults to cached/offline Hugging Face access and requires every
selected Docker image to exist locally. It does not silently download the
dataset or hundreds of gigabytes of images. Use `--allow-dataset-download` or
`--allow-image-pulls` only when that network and storage use is intentional.

The selected instance and image names are written to `instances.txt` and
`images.txt` in the run directory, including when the local-image check fails.

## Container memory isolation

The writable-layer guard limits disk usage, not RAM. To keep a sample OOM
from exhausting the worker, configure both per-container memory limits and a
bounded parent cgroup for the run; do not put the coordinator in that cgroup.
Concurrency does not need to change.

The opt-in memory guard requires Linux cgroup v2, a local Docker daemon, and
the writable-layer guard enabled. Export
`SPARSEENGINE_DOCKER_MEMORY_LIMIT_BYTES`, `SPARSEENGINE_DOCKER_MEMORY_PARENT`, and
`SPARSEENGINE_DOCKER_MEMORY_EVENTS` (the OOM JSONL path). In the MiniSWE extra
config, set `environment.run_args` to matching `--memory`, `--memory-swap`,
and `--cgroup-parent` values. Set memory and memory-swap to the same size to
disable container swap. Omit `--rm`: the guard needs Docker's OOM metadata
until failure detection and explicitly removes its container afterward.
Create and bound the parent cgroup before launching.

A detected generation-container OOM raises `DockerMemoryLimitExceeded` and
fails that sample without retrying it; the normal sample denominator is kept.
The same memory limits are applied to official scorer containers, whose
failures remain reported by the upstream scorer. Record these limits with the
run, since resource failures affect the score.

## SparseEngine Server

Start `sparseengine.entrypoints.openai.api_server` as a separate long-running
process. A typical command is:

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=$PWD/src \
python -m sparseengine.entrypoints.openai.api_server \
  --model <MODEL_PATH> \
  --served-model-name sparseengine-swe \
  --host 127.0.0.1 \
  --port 18000 \
  --engine-kwargs /path/to/engine_kwargs.json \
  --request-log-dir /path/to/server_requests
```

For example, `engine_kwargs.json` can contain:

```json
{
  "tensor_parallel_size": 1,
  "gpu_memory_utilization": 0.88,
  "max_model_len": 131072,
  "engine_prefill_chunk_size": 4096,
  "sparse_method": "vanilla"
}
```

Create a JSON manifest next to the server logs. Do not put API keys or other
secrets in it.

```json
{
  "command": "python -m sparseengine.entrypoints.openai.api_server --model <MODEL_PATH> --served-model-name sparseengine-swe --host 127.0.0.1 --port 18000 --engine-kwargs /path/to/engine_kwargs.json",
  "model_path": "<MODEL_PATH>",
  "served_model_name": "sparseengine-swe",
  "cuda_visible_devices": "2",
  "server_port": 18000,
  "engine_kwargs": {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.88,
    "max_model_len": 131072,
    "engine_prefill_chunk_size": 4096,
    "sparse_method": "vanilla"
  }
}
```

For a local API, the adapter requires this manifest and snapshots it into the
benchmark run. It also checks `/v1/models` and verifies that the advertised
model matches `--served-model-name`. It does not start, restart, or stop the
server.

## One-Instance Smoke

Run one instance before committing to all 300. `openai/` is the LiteLLM
provider prefix; `sparseengine-swe` is the exact model name advertised by the
server. SparseEngine does not require authentication, but LiteLLM expects a
non-empty OpenAI key, so use a dummy local value.

```bash
export OPENAI_API_KEY=local-sparseengine

SWE_BENCH_PYTHON=/path/to/swebench-env/bin/python \
bash scripts/benchmarks/run_swe_bench_lite.sh \
  --stage all \
  --swe-bench-dir ../SWE-bench \
  --run-dir /path/to/outputs/swe-bench-lite/sparseengine-smoke \
  --model openai/sparseengine-swe \
  --api-base http://127.0.0.1:18000/v1 \
  --served-model-name sparseengine-swe \
  --server-manifest /path/to/server_manifest.json \
  --slice 0:1 \
  --batch-size 1 \
  --mini-workers 1 \
  --eval-workers 1 \
  --step-limit 80 \
  --cost-tracking ignore_errors \
  --cost-limit 0
```

Local models have no provider billing metadata. With
`--cost-tracking ignore_errors`, cost is recorded as zero and `--cost-limit`
cannot enforce a budget; the bounded controls are `--step-limit` and
`--wall-time-limit-seconds`.

## Full Lite Run

Use a new run directory and remove `--slice 0:1`. Keep model, API, prompt,
decoding, dataset, and server settings identical to the smoke run. Start with
conservative generation concurrency and raise `--mini-workers` only after the
server is stable under simultaneous tool-calling requests.

```bash
export OPENAI_API_KEY=local-sparseengine

SWE_BENCH_PYTHON=/path/to/swebench-env/bin/python \
bash scripts/benchmarks/run_swe_bench_lite.sh \
  --stage all \
  --swe-bench-dir ../SWE-bench \
  --run-dir /path/to/outputs/swe-bench-lite/sparseengine-lite300 \
  --model openai/sparseengine-swe \
  --api-base http://127.0.0.1:18000/v1 \
  --served-model-name sparseengine-swe \
  --server-manifest /path/to/server_manifest.json \
  --batch-size 50 \
  --mini-workers 1 \
  --eval-workers 6 \
  --step-limit 80 \
  --cost-tracking ignore_errors \
  --cost-limit 0
```

Completed batches are skipped only after both their predictions and
`batch_done.json` prediction hash are validated. Partial mini-SWE-agent batch
directories are passed back to mini-SWE-agent, whose default behavior is to
skip completed trajectories. The adapter only merges declared numeric batch
directories, so backup directories cannot introduce duplicate predictions.

## Separate Generation And Evaluation

The LLM server is needed only for `prepare` and `generate`. Official evaluation
does not call the model API. To run the stages separately, repeat the same
semantic arguments and change only `--stage` and operational worker counts:

```bash
COMMON_ARGS=(
  --swe-bench-dir ../SWE-bench
  --run-dir /path/to/outputs/swe-bench-lite/sparseengine-lite300
  --model openai/sparseengine-swe
  --api-base http://127.0.0.1:18000/v1
  --served-model-name sparseengine-swe
  --server-manifest /path/to/server_manifest.json
  --batch-size 50
  --step-limit 80
  --cost-tracking ignore_errors
  --cost-limit 0
)

# Requires the model API and API key.
bash scripts/benchmarks/run_swe_bench_lite.sh \
  --stage generate --mini-workers 1 "${COMMON_ARGS[@]}"

# Requires Docker images, but not the model API or API key.
bash scripts/benchmarks/run_swe_bench_lite.sh \
  --stage evaluate --eval-workers 6 "${COMMON_ARGS[@]}"

# Requires only completed artifacts in the run directory.
bash scripts/benchmarks/run_swe_bench_lite.sh \
  --stage summarize \
  --run-dir /path/to/outputs/swe-bench-lite/sparseengine-lite300
```

The adapter rejects semantic configuration changes in an existing run
directory. Use a new run directory when changing the model, dataset selection,
step limit, decoding settings, API endpoint, or server configuration.
For all non-`summarize` stages it also compares the current adapter source,
SWE-bench source, Python executable, and package versions with
`run_manifest.json`; source or toolchain drift requires a new run directory.

Official evaluation uses a prediction-bound run id such as
`<RUN_ID>-pred-<HASH>`. Its report and per-instance cache live under
`<RUN_DIR>/official/`. The adapter writes an identity marker before allowing
cache reuse and rejects an existing cache directory that it cannot associate
with the current prediction and runtime provenance.

The shared adapter fixes `temperature=0`, `top_p=1`, and `max_tokens=4096` by
default. It records `seed=null` because this OpenAI-compatible route does not
currently expose a shared seed control.

## External API Providers

For DeepSeek or another LiteLLM provider, omit `--api-base`,
`--served-model-name`, and `--server-manifest`, then select the provider key
environment variable explicitly. Model calls are direct by default: proxy
environment variables are removed from the mini-SWE-agent process. Use
`--api-proxy-from-environment` only when the provider requires that proxy.

```bash
export DEEPSEEK_API_KEY=<KEY>

SWE_BENCH_PYTHON=/path/to/swebench-env/bin/python \
bash scripts/benchmarks/run_swe_bench_lite.sh \
  --stage all \
  --swe-bench-dir ../SWE-bench \
  --run-dir /path/to/outputs/swe-bench-lite/deepseek-lite300 \
  --model deepseek/deepseek-v4-flash \
  --api-key-env DEEPSEEK_API_KEY \
  --mini-extra-config /path/to/deepseek_nonthinking.yaml \
  --cost-tracking default \
  --cost-limit 0.05 \
  --step-limit 80
```

The optional provider config used above contains only the DeepSeek-specific
request field:

```yaml
model:
  model_kwargs:
    extra_body:
      thinking:
        type: disabled
```

Provider-specific request fields, such as DeepSeek thinking controls, are not
part of the shared adapter configuration. Pass them with a provider-specific
`--mini-extra-config`; the adapter hashes and snapshots that file. Do not reuse
such a config for SparseEngine, and never store credentials in it. Config and
server-manifest validation rejects sensitive field names, common provider token
formats, authorization headers, and URL credentials before snapshotting.

## Outputs

Each run directory contains:

| Artifact | Meaning |
| --- | --- |
| `run_config.json` | Immutable semantic experiment configuration and selected instance ids. |
| `run_manifest.json` | Code revisions, package versions, Python, credential variable name, and runtime policy. |
| `server_manifest.json` | Snapshot of the local SparseEngine server configuration, when applicable. |
| `evaluation_identity.json` | Prediction hash, official run id, and runtime-provenance hash used for cache ownership. |
| `invocations.jsonl` | Stage invocations and operational concurrency settings. |
| `status.jsonl` | Append-only stage and batch status events. |
| `batches/*/*traj.json` | Raw mini-SWE-agent trajectories and model responses. |
| `preds_all.json` | Strictly validated patches in official SWE-bench prediction format. |
| `generation_results.jsonl` | Parsed generation status and model statistics per instance. |
| `official/*.json` | Unmodified official SWE-bench aggregate report. |
| `official/logs/run_evaluation/` | Prediction-bound official per-instance evaluation cache and logs. |
| `per_sample_results.jsonl` | Normalized per-instance status and official outcome. |
| `final_summary.json` | Aggregate score, status counts, API calls, cost, and artifact paths. |

The normalized `status` values are `success`, `invalid_input`, `model_failed`,
`parse_failed`, `metric_failed`, and `skipped_by_policy`. `success` means the
official harness completed the instance; the separate `resolved` field records
whether its tests passed. The benchmark score is always
`resolved_instances / total_instances`, including empty patches and failures in
the denominator.

## Prune only tool-result KV

Add these arguments to the smoke command above:

```bash
--no-chain-cache \
--prefix-prune-policy kvzip_global \
--prefix-prune-target tool_results \
--prefix-prune-tokenizer /path/to/server-tokenizer \
--prefix-prune-keep-ratio 0.5
```

The server must enable Vanilla or OmniKV radix prefix caching and multi-range
pruning. Supply a local fast tokenizer matching the server, including its chat
template; the harness does not download it. Its environment needs Transformers
and the shared chat renderer's dependencies.

The harness locates only `role=tool` bodies in the complete rendered prompt,
protecting user input, assistant reasoning, call arguments and template markers.
It retains tokens crossing body boundaries, aligns ranges inward to the server's
block size, and verifies the token path ID through `/prefix_cache/match` before
pruning. A mismatch fails explicitly. The engine receives only token IDs, ranges
and a total keep budget, with no tool-specific metadata.

The ratio retains `floor(total_eligible_tool_tokens * ratio)` tokens across all
ranges together; it is not a per-result quota. Use `0 <= ratio < 1`; omit the
prune policy for a no-pruning baseline. Static range and keep-token options are
ignored in this mode. After each inference turn with new tool results, only
previously unprocessed tool bodies are pruned, sharing one keep budget across
that turn's new ranges. Earlier ranges are never recompressed. The trigger-token
threshold applies only to static-range mode; tool mode starts on the first
eligible turn. Empty or unaligned bodies are logged as `skipped_by_policy` and
advance the message cursor. Failed jobs do not advance it. Rewritten history
fails explicitly.

Reuse verification is merged into the next completed turn's prefix match.
Configuration is saved in `run_config.json`; message cursors, ranges, budgets,
freed slots and reuse checks go to `prefix_prune_events.jsonl`. Original messages
remain in the agent trace. The last prune without a later turn has no verified
reuse. Span selection still renders the complete template and tokenizes it once
to preserve BPE boundaries and template semantics; it does not concatenate
independently tokenized message suffixes.
