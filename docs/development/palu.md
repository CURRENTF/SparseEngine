# Palu integration contract

Palu preserves tokens and caches grouped low-rank projections of pre-RoPE K
and V. It changes the model projection/storage contract, not token selection.
Reference: https://arxiv.org/abs/2407.21118; official implementation inspected
at `bb22666e2ef96707e8dd21d93fc00146c2e0d615` (MIT).

## Supported vertical slice

- Identity: `sparse_method="palu"`, mandatory `palu_checkpoint_path` to an
  offline factor artifact. Original model weights remain the model input.
- Models: unquantized Llama and Qwen3, FP16/BF16, TP/EP/DP 1 initially.
  Qwen3 K normalization occurs after low-rank reconstruction, before RoPE.
- Factors: activation-whitened group-head SVD, fixed K/V rank per layer (uniform across groups
  within a layer), optional explicit layer rank schedule. No online SVD.
  Calibration accumulates uncentered input covariance for each layer, adds
  an explicit diagonal ridge, and applies Cholesky whitening before SVD.
  Plain SVD requires an explicit ablation flag. Automatic Fisher rank search
  and latent quantization are not claimed.
- Storage: separate low-rank K/V slot tensors and original token positions;
  layer ranks may differ. CacheManager owns allocation, append and release.
- Runtime: pass-through selection; no scoring, pruning or token eviction.
- Prefill: chunked, batched requests, reconstruct K once per request/layer;
  upstream attention aggregates latent V. Reconstruction scratch is temporary
  and included in startup peak profiling, never retained across layers.
- Decode: fused tiled K reconstruction, normalization, RoPE, QK and latent-V
  aggregation. V reconstruction is absorbed into each query head's output
  projection offline at model initialization. No full-history dense KV buffer.
- Graph: shared eager/replay kernel, fixed split envelope and provider-owned
  buffers; device lengths and negative write slots mask padding. No length
  specialization, graph keys or provider changes during replay.
- Prefix/offload: initially rejected explicitly. Ordinary sequence free and
  row/slot reuse use the StandardCacheManager allocator.
- Assets: shape/config and source projection hashes validated before replacing
  loaded projections. Missing, mismatched or malformed assets fail explicitly.

## Cost and reuse

Let T be context length, B batch, H KV heads, Q query heads, D head dimension,
G heads per compression group, and Rk/Rv ranks. Persistent payload per layer
and token is `element_bytes * (H/G) * (Rk+Rv) + 4` position bytes. Slot-stack
and request-table metadata are counted separately. Decode reconstruction costs
`O(B T H Rk D)`, attention `O(B T Q (D+Rv))`; it avoids writing reconstructed
K or V to HBM. Fixed split partials cost `4 B Q S (Rv+1)` bytes per distinct
rank plan, shared by compatible layers, plus output. Fused output projection
has width `Q*Rv`; grouped compression can increase this width, so cache savings
are not automatically compute savings. Short and long contexts must be timed.

Reuse StandardCacheManager allocation/graph metadata, typed compute views,
Torch GEMM and the public FlashInfer prefill API. The latter supports different
QK and V dimensions. Inspected SGL attention/MLA and vLLM attention interfaces
consume explicit KV or native shared MLA latent with decoupled RoPE; neither
expresses Palu's grouped pre-RoPE reconstruction. A new Triton decode kernel
therefore owns that semantic gap; no upstream kernel code is copied.

Validation requires independent reconstructed-attention numerical tests,
chunked versus full prefill, ragged/padded decode, actual repeated graph replay,
source-asset mismatch failures, real model generation, quality and matched
request/decode benchmarks. Measurements belong in external run artifacts.
