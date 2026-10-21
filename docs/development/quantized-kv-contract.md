# Quantized KV cache contract

The `kivi`, `turboquant`, and `fp8_kv` methods retain every token. They share
page allocation and expose compressed storage to a prepared decode provider.
They do not use sparse attention scores, token selection, or a compressor
checkpoint. Logical orchestration uses `PassThroughRuntime`.

| Axis | Initial contract |
| --- | --- |
| Identity | `kivi`, `turboquant`, `fp8_kv`; no external assets |
| Representation | KIVI asymmetric min/max int2/int4; TurboQuant MSE random rotation and Gaussian Lloyd-Max codebook, int2/int3/int4; dynamically scaled E4M3 FP8 |
| Persistent state | Cache-manager-owned pages, per-page or per-token scales, and one high-precision incomplete page per live row and layer |
| Selection and scores | All logical tokens; no score output |
| Model/layout | Homogeneous explicit KV, Llama/Qwen2/Qwen3, FP16/BF16 activations, head dimension 64/128/256 |
| Prefill | Batched/chunked standard attention; materialize historical compressed KV into a bounded layer workspace and use original current-chunk KV |
| Decode | Read packed full pages and the raw incomplete page directly; no full-history dense materialization |
| Admission | Page-rounded costs, raw-tail pool, row metadata, and prefill workspaces included in allocation budget |
| Lifecycle | Append and free; no prefix attach/fork, cache offload, or rollback API advertised |
| Graph/topology | Shared eager/decode-CUDA-Graph updates; model-validated TP and Qwen3-MoE EP/outer-TP; no quantization collectives; internal DP remains model-rejected |
| Validation | Independent numerical codec/attention oracles, allocation failure atomicity and reclamation, configuration/provider rejection, actual GPU generation when an idle device is available |

KIVI quantizes K over tokens within a page and V over channel groups within a
token. Both full pages are quantized together; the incomplete page stays in
the activation dtype. This intentionally differs from the official residual
window lifecycle and does not retain an additional sink window.

TurboQuant implements the MSE variant with a seeded orthogonal rotation and
a Gaussian approximation to the rotated-coordinate distribution. It does not
implement QJL residual correction or mixed outlier bit budgets. Integer codes
are packed into 32-bit words without straddling word boundaries; padding is
included in byte accounting, particularly for int3.

FP8 uses separate per-token, per-KV-head dynamic scales for K and V. This
recipe is calibration-free and is not a claim of parity with a particular
vLLM/SGLang backend or of FP8 tensor-core attention acceleration.

FP8 GQA decode groups query heads sharing a KV head and uses FP16/BF16 matrix
products with FP32 accumulation. E4M3 keys are exactly representable in the
query dtype; their per-token scales multiply the FP32 QK result. Values are
dequantized to the activation dtype before the probability/value product.
Queries remain in the activation dtype. The raw incomplete page participates
in the same softmax with unit scales. Groups larger than 16 use multiple head
tiles, without crossing KV-head boundaries. MHA retains the scalar provider.
The grouped provider bounds its split workspace at 32 partial outputs per
query head and loops over token tiles within each split. Device lengths
determine the split spans; no host length read or full-history copy is needed.
The baseline launch uses one pipeline stage: with this tile layout and
Triton 3.6, two stages exceed SM120's shared-memory limit at D256. This is a
resource-compatible baseline, not a claim of optimal tuning across devices.

This adapts SGLang's
[`_fwd_grouped_kernel_stage1`](https://github.com/sgl-project/sglang/blob/7e5d3780ef06bcd56756e31df27a3b4bb125568b/python/sglang/kernels/ops/attention/decode_attention.py)
(Apache-2.0). The local adapter adds dynamic per-token scales, the raw tail,
2D slot maps, and neutral writes for inactive splits. SGLang's standard FA3
FP8 path uses a different scale/cache contract and is not interchangeable
with this payload. Grouped arithmetic changes reduction order and rounds V
and softmax weights to the activation dtype; bitwise equality is not claimed.
GPU numerical, graph-replay and matched serving performance validation are
required before treating the grouped path as performance-ready.

The custom mixed packed-page/raw-tail payload requires a repository provider.
Standard prefill remains upstream-first. There is no runtime provider
reselection or silent dense fallback.

## Efficiency status

Standalone TurboQuant also has a known unresolved decode-efficiency issue.
Numerical and CUDA Graph lifecycle validation do not establish performance
readiness; enabling CUDA Graph alone does not resolve the observed latency
problem. Treat its current decode implementation as experimental, not as an
end-to-end acceleration claim or evidence about official TurboQuant kernels.

Standalone KIVI has a known unresolved decode-efficiency issue. Bounding live
dequantization tiles removes severe register spilling in validated shapes,
but does not establish acceptable serving latency. Keep functional/graph
support separate from performance readiness. Before further KIVI tuning,
assess adapting DeltaKV's existing grouped-Q-head matrix-multiply decode;
the two paths have different packed layouts and raw-tail metadata, so they
are not interchangeable launchers. This issue is not a support predicate and
must not silently route compressed requests to dense attention.

## Decode graph contract

CPU admission reserves pages and publishes row/length/write-slot metadata
outside capture. The batched GPU writer uses that metadata to append a raw
token or encode a newly completed page. Negative write slots suppress padded
writes even when padded read rows alias a live request. Prefill and decode
share the page codec, including rounding and scale axes; TurboQuant rotation
is unchanged. Prefill still executes eagerly.

Decode attention prepares a capacity-bounded workspace once. Graph launches
use a fixed split grid for the prepared context capacity; device lengths mask
unused tokens. Context length does not add graph keys, recapture triggers,
provider choices, or new workspace shapes. Eager mode can use fewer splits
with the same kernel. Cache storage and provider workspaces remain alive with
their owners; request release reclaims pages without replacing captured tensors.

The append work is O(batch * KV_heads * head_dim) for raw tails and
O(completed_pages * page_size * KV_heads * head_dim) for page encoding, with
no full-history copy. Decode remains O(batch * context * query_heads * head_dim);
its FP32 split workspace is bounded by configured capacity (and capped at 32
splits for grouped FP8). TurboQuant retains
its per-head rotation cost. No new inter-rank communication is introduced.

Reuse review: vLLM `reshape_and_cache_kernel` (v0.26.0,
`568afb3a13806beb53bb2e6bd518269357b237c0`,
`csrc/libtorch_stable/cache_kernels.cu`) provides the negative-slot write-mask
pattern, but not this mixed compressed-page/raw-tail lifecycle. SGL kernel 0.4.5's
`sgl_per_token_quant_fp8` and `sgl_per_token_group_quant_fp8` interfaces consume
dense token tensors, not page completion state or KIVI/TurboQuant metadata.
Neither directly substitutes for the state transition. Reuse the repository's
codec and fixed-grid attention reducer; only the grouped FP8 compute kernel
adapts upstream code, as described above.

Validation must distinguish kernel replay/codec equivalence, real-model
generation and resource reclamation, and matched eager/graph throughput.
Single-GPU validation does not establish multi-rank replay correctness.
