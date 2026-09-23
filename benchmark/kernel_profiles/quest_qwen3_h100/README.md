# QuEST scoring and small-batch FP8 MoE benchmarks

This directory contains the original Qwen/H100 calibration and a reusable
shape-driven QuEST benchmark. The directory name identifies the initial
calibration; the scoring implementation is model-independent.

## Current scoring policy

`QuestPageScoreDispatch` is the default repository-owned QuEST scoring
provider. It chooses scalar or repaired matrix-product kernels from
query/KV head counts, dimension, batch, page-table capacity, and device
capabilities. There are no model-name, exact-page-count, or package-version
whitelists in the default scoring policy.

The cost estimate uses padded reduction work per SM, GQA metadata reuse, and
the head-partial score and rounding-repair work. Short inputs retain the
scalar reduction. Multiple query heads sharing metadata can amortize matrix
multiplication. Without shared query-head metadata, including single-query MLA,
the default retains the original one-page reduction. A page-tiled vector kernel
remains an explicit benchmark candidate: fewer CTAs alone do not guarantee a
win once register pressure, partial merging and metadata working sets are included.
The constants are calibrated estimates, not correctness conditions or promises
of optimal performance. An unavailable SM count uses a documented reference
estimate of 128 SMs. Only measured hardware and workloads establish validation.

The dispatcher binds capability eligibility before execution. Shape decisions
use captured tensor dimensions, not GPU context values or changing logical
lengths. There is no online tuning and no retry with another kernel after a
JIT or execution failure. Runtime reports identify the selected kernel paths.

## Kernel contracts

- All paths accept contiguous `[B, query_heads, D]` queries, contiguous
  `[physical_pages, KV_heads, D]` maximum/minimum metadata, and contiguous
  int32 `[B, logical_pages]` page tables. Query heads must be divisible by KV
  heads. Inputs share dtype and device; invalid page sentinels clamp to page0
  as in the original scorer. Downstream selection still owns valid lengths.
- Vector reduction handles BF16, FP16 and FP32. The matrix-product path handles
  BF16/FP16 on CUDA SM80+; BF16 requires native device support. Unsupported
  matrix-product contracts retain the eligible scalar default. Other hardware
  has not been locally validated by this calibration.
- Dimension tails are masked. Matrix products split arbitrary dimensions into
  K tiles and arbitrary GQA groups into 16-head tiles. Rounding repair and
  head-partial reduction share one pass that writes final scores. Single-part
  contracts reuse the output buffer for partial scores.
- Both implementations preserve separately accumulated positive/negative
  bounds, separately rounded to input dtype before adding, then take the
  maximum across query heads. The matrix path bounds FP32 accumulation error
  using two additional absolute-product sums. Heads near a BF16/FP16 rounding
  midpoint are recomputed using a vector reduction inside the merged repair pass.
Repair work stays on the GPU, including under Graph replay. This addresses
observed selection changes from tiny score differences; it is not a proof of
bitwise identity on every architecture or input. The QuEST sparse runtime owns
MLA head-mean query formation; the scoring operator does not change MLA versus
GQA selection semantics.

## Reproduce

Activate an appropriate environment and expose one idle permitted GPU. From the
repository root, use a new output directory for every invocation:

```bash
PYTHONPATH=src python benchmark/kernel_profiles/quest_qwen3_h100/bench.py \
  --mode score-general --output <OUTPUT_ROOT>/score \
  --shape 1,8320,32,4,128 --shape 1,8320,32,8,128 \
  --shape 1,8320,1,1,576 --shape 3,3001,24,3,160
PYTHONPATH=src python benchmark/kernel_profiles/quest_qwen3_h100/bench.py \
  --mode moe --output <OUTPUT_ROOT>/moe
PYTHONPATH=src python -m pytest -q tests/test_quest_kernels.py \
  tests/test_triton_fp8_operators.py
```

`--shape` means `B,pages,query_heads,KV_heads,D`; repeat it for multiple cases.
`--dtype` accepts bfloat16, float16, or float32. Requests use disjoint physical
metadata by default; `--metadata-sharing shared` explicitly tests page reuse
across requests. Scoring compares the scalar,
vector, eligible matrix-product, and actual resolved provider callables.
The FP64 oracle independently computes the two reductions with the required
output rounding. Reports include score error and selected-page overlap.
CUDA tests additionally exercise changed Graph inputs, signed bounds, zero
queries, incomplete dimensions, and query heads spanning multiple tiles, and rounding-sensitive Graph replay
against the original scalar path.

Microbenchmarks use seed42, complete callable boundaries, 32 calls per captured
graph, 8 replays per sample, and 20 interleaved samples. By default, metadata is
reused. `--working-set-copies 32` rotates identical metadata at32 distinct
addresses within each captured graph. The summary reports the metadata footprint,
so a larger working set can expose cache effects without timing an artificial
eviction kernel. This is a controlled memory-reuse experiment, not a complete
model-cache simulation. Compare both working-set modes when calibrating a new
architecture; warm-cache operator wins need separate model validation. Numerical tolerances are BF16/FP16
rtol0.008/atol0.02 and FP32 rtol2e-5/atol1e-4 in the general benchmark. Selected
page overlap is measured separately; score tolerance does not imply identical
selection near ties. Quality validation must also compare actual model outputs.

Serving measurements reuse
`scripts/official_experiments/sparse_decode_efficiency/sweep_decode_capacity.py`
with `--probe-only --probe-concurrency`, matching model, workload and GPU between
variants. Decode throughput uses 32 warmup +256 contiguous steps with two
boundary synchronizations, one discarded full workload and three measured full
workloads; complete output is validated. Kernel latency is a separate metric.

Stable FlashInfer page selection requires 128 KiB shared memory per SM for
its FilteredTopK buffers. Devices below that resource requirement select the
existing Torch provider during resolution. Its stable ordering preserves
small-column tie breaking and invalid-row padding, including CUDA Graph replay.
This eligibility check is independent of the scoring policy and H100 profiles.

The H100 BF16 paged-view profile uses the exact fused selection and view kernel
for widths up to 512 pages with CUDA Graph, and up to 2048 pages in eager mode.
The kernel preserves the dense short-row fallback. Wider views and graph views
above 512 pages use FlashInfer selection followed by view finalization.

## Original Qwen/H100 evidence

The MoE override remains a separate exact performance
profile: H100, Qwen-compatible FP8 E4M3 block weights, BF16 activations, FP32
scales, 128 experts, H2048/I768/top-k8, TP1/EP1, CUDA Graph, Torch2.11.0+cu130,
Triton3.6.0, FlashInfer0.6.17. Both providers consume `[up, gate]` packed weights.
This standard-operation override is not generalized by the scoring policy.
Within this existing profile domain, single- and multi-request work now use
one layout-compatible Triton route. The prior dtype, graph and toolchain gates
remain in place; this does not promote the local provider on other GPUs.
The native block-scaled FP8 pipeline fuses SiLU, multiplication and group
quantization, retaining both intermediate low-precision rounding operations.
Tensor-scaled experts retain their separate quantization path. Independent
BF16/FP16 oracles cover both packing orders; the Graph regression also covers
changing per-request expert IDs and weights in multi-request batches.
GLM's tensor-scaled FP8 expert contract must not be substituted with a
block-scaled provider merely because both checkpoints are labeled FP8.

| Original complete Graph callable | Scalar/FlashInfer (µs) | Selected (µs) |
|---|---:|---:|
| FP8 MoE, one token | 50.536 | 18.529 |
| Page scoring, capacity2060 | 6.635 | 3.898 |
| Page scoring, capacity8320 | 20.694 | 7.460 |

Original evidence: [MoE samples](data/moe-raw_samples.jsonl),
[MoE summary](data/moe-summary.json), [scoring samples](data/score-raw_samples.jsonl),
[scoring summary](data/score-summary.json), [decode comparison](data/decode-comparison.json),
and [historical measured source](data/measured-source.json). These artifacts
predate the general dispatcher and do not establish its wider performance.

## Vortex reference boundary

The inspected Vortex revision is
`c016fdfc871130481c76e3beead83e8c2b54cb99` (Apache-2.0). Its generated
`gqaquestsparseattention_4a629439_subgraph_0_kernel` tiles64 pages and8 grouped
query heads, reusing queries across pages and metadata across grouped heads.
Its BF16 product/max/single-sum arithmetic and per-KV-head Top-K differ from
SparseEngine's separately rounded positive/negative sums and shared-head pages.
The scheduling idea is reusable; these methods do not imply equal selection.
