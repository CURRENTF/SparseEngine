# RetroInfer GPU-only integration

Paper: [RetroInfer, arXiv:2505.02922](https://arxiv.org/abs/2505.02922).
Author implementation: `microsoft/RetrievalAttention` at
`03f912c6e917c380d9d90c5ec85bb0f161ba53ef` (MIT).

## Method contract

RetroInfer keeps the complete, post-RoPE historical KV on GPU. It clusters
older keys separately for each KV head. Each cluster stores the mean key,
sum of values, token count, and a reverse index into the full KV cache.
At decode, the current query scores the centroids. The highest ranked clusters
are read exactly; the next ranked clusters contribute an estimated softmax
numerator and denominator. The sink and unindexed recent tokens are read
exactly. All three regions share one normalization. The remaining clusters
are omitted. The retrieval and estimation budgets count clusters, not tokens.

The initial index is built after a completed prefill when at least 16,384
tokens remain between the 4-token sink and 64-token recent region. It uses
mean-centered segmented spherical k-means training and a final global
assignment. Each later 1,024-token segment receives its own index; the most
recent 64 tokens remain outside the index. Shorter contexts use the same
exact-read kernel over their complete KV until an index is built.

The first serving scope is eager, uniform explicit FP16/BF16 KV, attention
TP=1, and standard full causal MHA/GQA attention in Llama, Qwen2, Qwen3,
Qwen3-MoE, or MiniMax-M2. MoE routing is separate from the attention cache.
CUDA Graph, async scheduling, prefix caching, MLA, mixed or shared per-layer
KV layouts, sliding attention, and KV quantization are rejected by configuration
or provider resolution. The GPU-only method has no
host KV tier, CPU wave buffer, or KV transfer policy.

## Ownership and cost

`RetroInferGPUCacheManager` retains Standard's physical GPU KV cache and slot
allocator. Its request-owned index is keyed by sequence ID, and each layer
has its own centroids, value sums, packed reverse index, and indexed end.
Completed prefill and decode-step hooks build and extend the index. Freeing a
request drops all index tensors before the Standard row can be reused.
The typed decode view gives the provider the request row and layer index.
The provider scores and ranks clusters, then its Triton kernel reads the
physical KV slots and merges exact and estimated contributions.

For `Hk` KV heads, head dimension `D`, `L` indexed tokens, `C` clusters,
`G` query heads per KV head, `S` exact sink/recent tokens, `R` retrieved
tokens, and `E` estimated clusters:

| Phase | Approximate work |
| --- | --- |
| Index build | 9 segmented assignment/reduction rounds plus one global assignment, each `O(Hk L C_segment D)` within a segment; final packed reverse sort `O(Hk L log L)`. Training uses bounded 256-token score chunks. |
| Decode | Centroid scoring `O(Hk G C D)`, top-k ranking, then `O(Hk G (S + R + E) D)` attention. The exact number of retrieved tokens varies with cluster size. |
| Persistent storage | Full GPU KV `2 Hk L D b` per layer, plus centroids, FP32 value sums, sizes, offsets, and 4 bytes per indexed token in the packed reverse index. |
| Peak workspace | One layer's gathered KV and centered keys, clustering sums/counts/assignments, and the old and new index tensors during an update. KV-slot capacity reserves index and one-context build workspace before allocation. |

The implementation uses chunked PyTorch GPU operations for index construction
and a Triton direct-slot attention kernel. It does not depend on the author's
CPU wave-buffer extension. The source paper and pinned author implementation
are the algorithmic reference; the packed reverse index and full GPU slot
storage are SparseEngine adaptations. Their quality and speed need matched
validation before a parity or performance claim.

## Validation boundary

The independent oracle in `benchmark/retroinfer/reference.py` checks
assignment, centroid/value statistics, cluster ranking, and three-zone
attention on small inputs. CUDA tests must compare the serving kernel with
that oracle, including empty clusters, GQA, uneven cluster sizes, and the
1,024-token update boundary. A real-model gate must cover interleaved
requests, free/reuse, quality, and matched decode throughput on an idle GPU.
CPU-only tests establish the math and static contract, not CUDA correctness
or any quality/efficiency result.
