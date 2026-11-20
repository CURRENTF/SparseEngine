# PACE runtime contract

Native `sparse_method="pace"` (Planner-Aware Context Engine, arXiv:2611.10240)
is H2O eviction over foresight-adjusted scores, with optional bounded RoPE
positions for sessions longer than the model's position range. The reference
implementation is the PACE overlay at tag `paper-v1` (SparseEngine
`b9f08d57fb23c3b053f40fc209d1b72dff4f00ab`). This port must reproduce it under a
pre-registered parity check before any capability below is advertised.

Out of scope for this method: host/NVMe/Valkey rehydration tiers, the
`noattn`/`block16` retention experiments, and any planner. The engine consumes a
plan; it never produces one.

## Algorithm and request boundary

A request may carry a `PACEPlan`: `current_step: int >= 0` and a tuple of spans
`(start, end, steps)`. Spans are half-open token ranges in logical session
coordinates (chain turns included); `steps` are the future agent steps that will
read the span. Empty or inverted spans are dropped, `start` is clamped to 0 and
steps are sorted and deduplicated. A plan on a non-PACE request is rejected; a
PACE request without a plan behaves exactly like H2O.

For logical position `p` the foresight bonus is

```
next_ref(span) = min{s in steps : s >= current_step}     (none -> span ignored)
w(span)        = 0.5 + 0.5 * exp(-lam * (next_ref - current_step))
bonus(p)       = scale * max{w(span) : start <= p < end}  (0 outside every span)
```

with `scale = pace_bonus_scale` (1e5) and `lam = pace_bonus_decay` (0.35), in
float32. The range is `[scale/2, scale]`, so every token with a future reference
outranks every unreferenced token while `scale/2` exceeds the largest reachable
cumulative H2O score; among referenced tokens the soonest-needed rank first, and
attention breaks ties inside a span. The engine does not enforce the
`scale/2` floor assumption.

The bonus is added once, when a prefill chunk's scores are appended to the
per-(layer, sequence) H2O score vector, at logical positions
`num_prefilled_tokens + [0, chunk)`. Decode tokens get no bonus at creation. On
the first prefill chunk of a resumed chain turn, every resident token is
re-scored: `score += bonus_new(logical) - bonus_folded_in`. Spans whose last
reference has passed drop to 0 and become ordinary H2O candidates. Selection,
the recent window, budgets, trigger intervals and compaction are H2O's,
unchanged.

With `pace_bounded_positions=True`, each sequence keeps a shift with
`physical = logical - shift`. Positions written for prefill and decode are
physical. When `next_physical + headroom >= limit`, every KV layer's resident
row is renumbered to `0..R-1` in logical order, stored keys are re-rotated from
their old to their new physical position, and `shift += next_physical - R`.
Gaps between survivors collapse; this is the intended semantics. `limit` is
`min(rope_max_position, pace_position_limit)`; `headroom` is
`engine_prefill_chunk_size + PACE_POSITION_HEADROOM_MARGIN` (64). Renumber checks run
after every prefill chunk and decode step, so at most one chunk is appended between two
checks. The margin is fixed at the reference implementation's value because changing it
moves the renumber points. Configuration fails unless
`max(decode_budget + interval, prefill_budget + chunk) + headroom < limit` and
`h2o_decode_eviction=True`.

Re-rotation does not compound rounding. A key's first renumber copies it and its
position into a per-layer reference store of `pace_reference_tokens` entries
(default 4096); later renumbers rotate from the reference in float32. Keys that
do not fit fall back to in-place rotation of the stored key; that compounds rounding,
so a store smaller than the worst resident row (or `pace_reference_tokens=0`) logs a
warning at configuration and on the first overflow. One store serves every sequence on a
layer: concurrent bounded sessions compete for entries, so size it as concurrent
sessions x resident tokens (validated configurations are single-session). When entries
run short, tokens carrying a plan bonus capture first, then the oldest; held entries
are not preempted. Entries are owned
by row columns and released when a column is evicted or its sequence freed, so
the overlay's stale-reference guard is unnecessary. The delta rotation
is derived from a cos/sin table with the model's RoPE parameters, built on the KV
device: `cos(n-o) = (cn*co + sn*so)/(co^2+so^2)`
and `sin(n-o) = (sn*co - cn*so)/(co^2+so^2)`, which also cancels YaRN magnitude
scaling.

## Capability contract

| Axis | Contract |
| --- | --- |
| Identity | Public and canonical name `pace`; no aliases. Default prefill method `h2o_prefill`. No assets. Knobs: `pace_bonus_scale=1e5`, `pace_bonus_decay=0.35`, `pace_bounded_positions=False`, `pace_position_limit=None` (RoPE maximum), `pace_reference_tokens=4096`. All `h2o_*` knobs apply unchanged. No environment variables. |
| Semantic delta | Versus H2O: prefill score vectors receive the plan's bonus; resident scores are re-based at each chain turn; with bounded positions, positions are physical and keys are re-rotated at renumber. What is kept, evicted and compacted is otherwise H2O's. |
| Persistent state | H2O's `_h2o_scores`. Per (layer, sequence), row-aligned like the scores: `logical int64 [R]`, `bonus float32 [R]`, `physical int64 [R]`, `ref int64 [R]`, plus the host logical end of recorded columns. Per sequence: `shift`. Per KV layer: reference store `K [cap, Hkv, D]` in cache dtype, `pos int64 [cap]`, `in_use bool [cap]`. Per request: the plan, compiled once per turn into sorted breakpoints and weights. |
| Score contract | H2O's: one float32 vector per (layer, sequence), shared by KV heads, aligned to the physical row. Prefill scoring is required; decode scoring is required with `h2o_decode_eviction`. The bonus is a deterministic function of logical position and plan and is identical across layers and ranks. |
| Selection contract | H2O posthoc cumulative selection, unchanged: `select_h2o_indices[_batch]` over `score + bonus`. Physically mutating via H2O compaction and decode eviction. Row-aligned PACE state is gathered by the same keep indices at every site where H2O gathers kept scores. |
| Storage | Explicit, homogeneous, unquantized BF16/FP16 KV (keys are edited in place when bounded). Actual post-RoPE keys required. MLA latent, shared-KV, head-slot layouts and quantized KV are rejected. |
| Attention view | H2O's logical views; no custom payload. Only position values differ. |
| Prefill execution | `h2o_prefill`, all chunked, batchable like H2O. Decode-only (`prefill_sparse_method=""`) and `flashprefill_v2` are rejected until validated. Score mode follows H2O (probability when `h2o_decode_eviction=True`). |
| Admission | Physical capacity as H2O. With bounded positions, logical length is unbounded: `request_logical_length_limit` reports none, `max_model_len` becomes the physical row capacity (validated at config time against the worst resident row plus headroom), and decode-path capacity checks the physical row. The reference store is deducted from the KV budget at startup and allocated with the pool. |
| Lifecycle | Allocate: empty state, `shift=0`. Append: prefill writes logical/physical immediately; decode tokens are materialized from `next_physical` before any compaction or renumber. Evict/compact: gather by keep indices; references of evicted tokens are released. Chain turn finished: materialize decode tokens, assert `next_physical + shift == processed_token_count`. Chain snapshot/restore: row-aligned state and per-sequence shift travel with the H2O scores; reference entries are dropped (at most one extra rounding per offload). Radix prefix attach and fork are rejected. Free and warmup reset clear all state. Rollback: not supported, rejected. |
| CUDA Graph | Foresight only: same as H2O (bonus and re-basing are host-scheduled prefill work). Bounded positions: eager only in this PR, rejected with `decode_graph=True`. Positions are written into the static buffer before replay and renumber runs in `finish_step` outside capture, so no algorithmic blocker is known; the gap is validation of every position producer under replay. |
| Topology | TP=EP=DP=1 only. TP is plausible (the bonus and shift are deterministic per rank) but unvalidated; DP lacks per-request field forwarding. Rejected otherwise. |
| Models | `qwen2`, `qwen3`, `llama` with full-width NeoX (split-half) RoPE. Interleaved or partial rotary, hybrid, MoE and `gemma4` are rejected. |
| Prefix cache | `chain` only. `radix` and `auto` resolving to radix are rejected. |
| Time cost | See below. |
| Space cost | See below. |
| Kernel reuse | See below. |
| Validation | See below. |

## Cost and reuse review

Dimensions: `A` KV layers, `Hkv` KV heads, `D` head dimension, `R` resident row
length (`<= decode_budget + interval`), `C` prefill chunk, `S` plan spans,
`N` resident slots in the pool, `cap = pace_reference_tokens`, `L = limit`.

| Event | Cost |
| --- | --- |
| Plan compile (once per turn) | `O(S log S)` host: merge spans into piecewise-constant max weights. |
| Prefill chunk | `O(C log S)` once per sequence (searchsorted), then `A` adds of `C` floats. No host syncs. |
| Turn re-basing | `O(A R log S)` once per resumed turn. |
| Decode step | Unchanged from H2O, plus one integer subtraction per sequence. |
| Eviction | H2O's, plus gathers of four `[R]` vectors per layer. |
| Renumber | `O(A R Hkv D)` rotation plus an `O(R log R)` sort per layer. One event per `L - headroom - R` new tokens per sequence. |

The overlay's per-span Python loop with a host sync per span and layer is
removed. The overlay's slot-indexed tables (`O(A N)`) become row-aligned state
(`O(A R)`).

Bytes per sequence: `A * R * 28` for row-aligned state. Per KV layer:
`cap * (Hkv * D * esize + 9)` for the reference store. For Qwen3-8B
(`A=36`, `Hkv=8`, `D=128`, BF16) at `cap=4096` that is about 303 MB, shared by
all sequences, plus about 2.2 MB per sequence at `R=2176`. The peak temporary
at a renumber is about `4 * R * Hkv * D * 4` bytes for one layer (fp32
reference, rotated key and cos/sin), about 36 MB here. There is no graph
workspace.

Reuse decisions:

- **Selection and compaction.** Reuse `select_h2o_indices[_batch]` and H2O
  compaction unmodified. A narrow H2O hook is added where kept scores are
  gathered, so subclass row state follows the same keep indices.
- **Rotation.** Reuse `apply_rotary_emb` (`layers/rotary_embedding.py`); its
  float32 arithmetic matches the overlay bit for bit.
- **Cos/sin table.** `build_rope_cos_sin_table` evaluates `RotaryEmbedding`'s
  formula on the KV device. The model's cache is built on the CPU and differs by
  one float32 rounding step in some entries. The reference implementation
  rebuilt the table on the GPU, and parity tracing showed that these rounding
  differences alone change answers in the arm without a plan. The device-built
  table reproduces the reference bit for bit. `reverse_rotary_emb` is not
  needed: the delta form avoids a de-rotation round trip.
- **Positions.** `Sequence.position_offset` generalizes the existing
  `multimodal_position_delta`: `decode_input_position` subtracts it, and the
  Standard, SnapKV and H2O prefill producers subtract it from their logical
  ranges. Every decode producer (eager, static, uniform graph staging) reads
  `decode_input_position`, so there is one source. The PACE manager publishes
  each sequence's shift before `prepare_step` and `prepare_decode_graph_step`
  in the process that prepares the step; it is 0 for every other method.
- **Rejected alternatives.** `normalize_rotate_store` (DeepSeek-V4 packed
  latent storage) has an incompatible layout. vLLM and SGLang have no delta
  re-rotation primitive for paged GQA keys; their RoPE kernels rotate fresh
  Q/K by absolute position only.

No new CUDA kernels. Renumber uses torch gathers and scatters because it is
rare. A fused kernel would need a measured need.

Expected benefit and how to falsify it. PACE adds no attention work. Its claims
are retention quality under a plan at a fixed budget, and sessions beyond the
RoPE range. Overhead to measure: prefill time against H2O at the same budget
with and without a plan, and renumber latency per event. Short sessions with no
plan must match H2O exactly.

## Validation boundaries

- **CPU unit tests.** An independent oracle for the bonus (per-span reference
  against the compiled piecewise form, including overlaps, passed steps and an
  empty plan).
  - **Ledger.** Shift and renumber invariants.
  - **Gather.** Row-state gathers against H2O keep indices.
  - **Chain bookkeeping.** Snapshot and restore.
  - **Rejection.** Each unsupported combination.
- **CUDA tests.** Re-rotation compared with a fresh rotation at the new
  position. Non-compounding across many renumbers, against a float64 oracle.
  Decode positions actually consumed by the forward on eager and static paths.
  An end-to-end H2O equivalence check for PACE without a plan.
- **Parity.** `pace reproduce headline` on the native backend against the
  `paper-v1` overlay under the pre-registered rule held in the PACE repository.
  Quality and performance results stay out of this repository's docs.
