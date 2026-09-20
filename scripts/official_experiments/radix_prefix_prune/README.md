# Radix prefix pruning by conversation region

Status: experiment design with a SWE-lite tool-result harness; no measured results yet.

This package studies physical radix KV pruning with `sparse_method="omnikv"`
and `policy="kvzip_global"`. OmniKV controls decode selection; KVzip scores
cached tokens for physical removal. This does not select `sparse_method="kvzip"`
and does not claim original per-head KVzip parity.

## Region contract

Annotate spans in the exact rendered token sequence used to populate the cache:

| Region | Content |
| --- | --- |
| `user_input` | User message bodies |
| `tool_result` | Tool response bodies |
| `model_thinking` | Explicit assistant reasoning spans |
| `model_output` | Assistant content outside reasoning, including tool calls |

Keep system instructions, template delimiters and unclassified tokens intact.
Do not infer reasoning boundaries when the trace lacks explicit annotations.
One category can contain multiple disjoint spans. Do not tokenize each message
independently and concatenate lengths: chat rendering and tokenization can change
boundaries. Verify the final token IDs against the populated radix path, including
whether historical reasoning survives chat-template rendering.

For semantic span `[s,e)` and runtime block size `B`, prune only the interior
`[ceil(s/B)*B, floor(e/B)*B)`. Preserve mixed boundary blocks. Record empty
interiors as `skipped_by_policy`. Record both requested and actual removal rates
relative to the complete semantic span.

## Initial comparison

Suggested removal rates: 0%, 25%, 50%, 75%. These are candidate experimental
settings, not validated quality thresholds. For aligned width `N`, use
`keep_tokens = N - floor(N * removal_rate)`. Skip the API call when removal
rounds to zero: the current API requires `keep_tokens < N`.

Start with no pruning and single-category ablations on fixed conversation traces.
Rebuild an unpruned cache for every independent comparison. Keep model, trace,
OmniKV decode budget, template and generation parameters fixed. After pruning,
continue the conversation to measure effects on subsequent answers; an answer
generated before pruning cannot measure pruning quality.

## Current implementation constraints

- The [prune API](../../../docs/en/features/prefix-cache-pruning.md) accepts an
  arbitrary-length `ranges: [[L1, R1], ...]` list and one shared `keep_tokens`
  budget. Gaps are preserved. It has no category labels or per-region budgets.
- Targets must exist in the radix tree, with no references or transfers on the
  affected blocks. Use radix mode; `omnikv_prefill` chain mode is outside this
  experiment's contract.
- `kvzip_global` scores all intervals against the resident prefix, then selects
  one token mask shared by layers, heads and TP ranks before committing changes.
  Reconstruction scoring needs additional context capacity beyond the largest
  endpoint. With a zero budget, reconstruction is unnecessary and skipped.
- Collect a category's disjoint spans into one request. Do not submit successive
  requests overlapping already-pruned blocks: `allow_recompress` remains unsupported.
  New disjoint intervals after a compacted prefix are supported.
- For the initial shared-budget study, compute `N` as the sum of aligned interval
  widths. Independent keep quotas for the four categories remain future work.

## Required runner evidence

Save the exact token trace and labeled spans, aligned intervals and keep budgets,
model/config/command/seed/code identity, every prune request and terminal job
status, retained tokens and freed slots, subsequent cache-hit observations, raw
continuations, parsed outputs and per-sample quality results separately. Missing
spans, failed jobs and scoring errors must remain explicit. Bound job polling.

Measure pruning maintenance cost separately from subsequent request latency and
complete workload time. Logical prefix hits do not prove physical KV savings;
freed slots do not prove a throughput gain. Use the canonical efficiency runner
for performance comparisons after validating the functional path on an idle GPU.


## SWE-lite tool-result smoke

Use the canonical [SWE-lite runner](../../../docs/en/benchmarking/swe-bench-lite.md#prune-only-tool-result-kv)
with `--no-chain-cache --prefix-prune-policy kvzip_global
--prefix-prune-target tool_results --prefix-prune-tokenizer <TOKENIZER_DIR>
--prefix-prune-keep-ratio 0.5`. Start with one instance and a fresh unpruned
server cache. After each inference turn containing new tool results, the harness
submits that turn's eligible tool-body ranges in one job with a shared keep
budget. Previously processed tool bodies are not recompressed. Empty or unaligned
bodies are recorded as `skipped_by_policy`; failed jobs do not advance the message
cursor. The trigger-token threshold applies only to static-range mode.
The engine remains unaware of message roles. User input, reasoning and assistant
output are outside this experiment's pruning scope.
