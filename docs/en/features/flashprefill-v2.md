# FlashPrefill V2

FlashPrefill V2 is an optional sparse-prefill attention provider selected with
`prefill_sparse_method="flashprefill_v2"`. This selector is orthogonal to
`sparse_method`: vanilla, OmniKV, QuEST, SnapKV, or H2O continues to own cache
allocation, prefill page tables, and decode behavior, while FlashPrefill V2
only replaces the prefill attention computation.

## Install the validated upstream revision

The adapter requires `flashprefill==3.0.0` built from upstream revision
`75b58f2ecdba1c269a87dd34d8f1ae57bef50c57`. The locally validated binary
contract is CUDA SM90, BF16, head dimension 128, causal varlen paged prefill,
and page size 1.

```bash
git clone https://github.com/qhfan/FlashPrefillv2.git
cd FlashPrefillv2
git checkout 75b58f2ecdba1c269a87dd34d8f1ae57bef50c57

export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE
export FLASH_ATTENTION_DISABLE_SPLIT=TRUE
export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
export FLASH_ATTENTION_DISABLE_LOCAL=TRUE
export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
export FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_SM80=TRUE
export FLASH_ATTENTION_DISABLE_HDIM64=TRUE
export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
export FLASH_ATTENTION_DISABLE_HDIM256=TRUE
export FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE
export FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE
MAX_JOBS=8 python -m pip install ./flashprefill_ops --no-build-isolation
```

Package absence, an incompatible version, or binary load failure is reported
at provider resolution. Sparse-Engine does not silently replace the requested
sparse semantics with a dense provider.

## Configure

`flashprefill_v2_abs_threshold` has no universal default and must be passed
explicitly. Larger values select fewer blocks. Calibrate it for each model and
report a matched quality regression together with any performance result.

```python
llm = LLM(
    "/path/to/model",
    sparse_method="omnikv",
    prefill_sparse_method="flashprefill_v2",
    flashprefill_v2_abs_threshold=0.002,
    flashprefill_v2_k_block_m=128,
    flashprefill_v2_k_block_n=128,
    flashprefill_v2_attention_sink_blocks=2,
    flashprefill_v2_window_blocks=4,
    flashprefill_v2_last_query_blocks=8,
    flashprefill_v2_min_sparse_q_len=4096,
    flashprefill_v2_use_mean_correction=True,
)
```

The supported cache/decode combinations are `vanilla`, `omnikv`, `quest`,
`snapkv`, and `h2o` on explicit-KV MHA models. MLA latent models such as
GLM-4.7-Flash reject this configuration during initialization. H2O normally
resolves an omitted `prefill_sparse_method` to `h2o_prefill`. Selecting
`flashprefill_v2` instead replaces only the prefill attention computation.
H2O still computes its method-owned posthoc importance scores and runs the same
final decode-cache preparation. It does not also run `h2o_prefill` intermediate
compaction because both are alternatives on the same prefill axis. SnapKV likewise
preserves its existing
posthoc score-and-compact lifecycle. These extra scorers remain part of the
cache method's cost and must be included in matched performance measurements.
H2O decode scoring and periodic eviction default to disabled; the independent
`h2o_decode_eviction=True` switch enables them and forces probability scoring.

The threshold in this example is only a Qwen3-4B calibration starting point;
it is not a model-independent recommendation. Prefix-cache hits are supported:
the cache manager supplies the full physical page table and cache length, while
the upstream kernel derives the attached prefix length as cache length minus
query length. The current adapter does not request attention-score or
softmax-LSE outputs from the upstream kernel.
