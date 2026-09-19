# GLM MLA TileLang kernel

The decode kernel is adapted from
`examples/deepseek_mla/example_mla_decode_paged.py` in Tile-AI/TileLang commit
`c7fabc4cc65e480b88b7606eb1bc9c340dbd8c8c` under the MIT license.

Local changes implement the GLM-4.7-Flash TP1/TP2/TP4 decode contracts used by
Sparse-Engine:

- BF16 query and cache tensors;
- direct strided 20/10/5 TP-local queries, zero-padded inside complete MMA tiles;
- page-size-one indirect cache slots;
- explicit caller-owned output and split-KV workspaces;
- optional fused FP32 raw-QK score reduced by max over the real local heads
  before applying the attention softmax scale;
- indirect request rows and safe `-1` padding outside each context;
- CUDA Graph-compatible execution.

The module only defines kernels. Provider selection, workspace ownership,
dependency checks, launch-config selection, and fallback policy belong under
`sparseengine.operators`.

The production score adapter uses `sm_parallel_nearest_v1` to generate its static
split plan from the device SM count, batch and padded-head tile parallelism:

```
ideal_splits = SM_count / (batch * head_tiles)  # target one CTA per SM
splits = nearest_absolute(ideal_splits, candidates=[4, 8, 16, 32])
# Exact ties choose the smaller candidate.
```

The head tile remains 32 for 20 local heads with batch > 1, otherwise 16.
Neither actual context length nor context capacity enters split selection.
The conservative candidate range is profile policy, not a kernel correctness
limit. The formula scales with hardware but is not a claim of optimal tuning on
every GPU. Historical context-aware experiments remain frozen in
`scripts/official_experiments/tilelang_mla_split_profiles`; they do not measure
this context-independent rule. The capacity rerun recipe is in
`scripts/official_experiments/sparse_decode_efficiency`.

The provider passes SM count from `DeviceCaps` at binding. Context capacity
remains a storage/validation bound only. Plans are indexed only by batch;
replay changes device-side masks but does not reselect splits, resize workspaces
or add context graph buckets. Binding metadata records the rule and its inputs.
Direct adapter callers must provide a prepared plan, a fixed experimental
configuration, or an explicit positive SM count.

The production score contract is FP32 raw QK per local head. Strided queries and
score staging are supported; slot tables must be contiguous. Invalid runtime
contracts fail explicitly rather than silently switching providers. Score-free
attention continues through SGL FA3; its selection and tuning are unchanged.
