# Getting Started

This page covers environment setup, checkpoint download, and a minimal
SparseEngine usage example.


## Install with Conda

```bash
conda create -n sparseengine-cu130-py312 python=3.12 -y
conda activate sparseengine-cu130-py312

python -m pip config --site set global.extra-index-url \
  "https://download.pytorch.org/whl/cu130 https://flashinfer.ai/whl"
python -m pip install "transformers==5.13.1" -e ".[cu130]"
python -m pip check

# Optional
MAX_JOBS=8 pip install flash-attn --no-build-isolation
```

## Install with uv

```bash
uv venv --python 3.12
source .venv/bin/activate

uv pip install -e ".[cu130]"

# Optional
MAX_JOBS=8 uv pip install flash-attn --no-build-isolation
```


Use `cu129` instead of `cu130` for CUDA 12.9.

`einops`, `sglang-kernel==0.4.5`, and the training, benchmark, and test packages
are runtime dependencies, so workflow-specific extras are not required. The SGL
kernel package is pinned to the validated PyTorch/CUDA ABI; other versions fail
provider setup instead of falling back silently.

The CUDA extras also require `flashinfer-python>=0.6.15,<0.7`. GPU engine
startup fails if FlashInfer or SGL kernel is absent, has incompatible package
metadata, or cannot load. Reinstall the matching dependency set with
`pip install -e ".[cu129]"` or `pip install -e ".[cu130]"`.

SparseEngine supports Qwen3.5/Qwen3.6/Qwen3.8 checkpoints in unquantized BF16
and block-scaled FP8 formats. All three share the `qwen3_5` runtime architecture
and support matrix.

Its causal Conv1D and decode packing paths remain repo-owned Triton kernels.
The GDN core is resolved once during model preparation. With the supported
`flashinfer-python>=0.6.15,<0.7` family, eligible SM90, SM100/SM103, and SM120/SM121
contracts bind the public FlashInfer prefill dispatcher plus the repo fused
Triton decode kernel; other supported contracts bind the repo Triton
implementation. SM100/SM103 requires CUDA 13, and all Blackwell paths require
head dimension 128. Value heads must equal or be an integer multiple of key
heads. The adapter presents FP32 initial/final state at the FlashInfer boundary
while preserving the configured BF16 or FP32 repo runtime state. Provider
resolution validates the public dispatcher signature and the architecture-specific
kernel symbol before execution.

`flashinfer-jit-cache` is an optional acceleration package:

```bash
pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu130
```

Block-scaled FP8 Linear selects an implementation from the local operator
registry using the active CUDA device capabilities. An SM90 operation matching
the BF16 block-scale contract uses the optimized FlashInfer implementation;
other supported SM90 contracts use generic Triton. Any model whose FP8 Linear
operation matches a profiled shape and contract on the RTX PRO 6000 binds the
same model-independent dispatch plan: `M < 512` uses Triton, while `M >= 512`
uses an atomic provider composed of SGL per-token-group activation quantization
and FlashInfer's public `gemm_fp8_nt_groupwise` CUTLASS kernel. On an unprofiled
SM120 shape inside that upstream atomic contract, the default portfolio uses the
upstream groupwise provider; generic Triton remains the portable baseline when
the upstream contract is ineligible or its optional packages are absent. The
binding report records the selection basis, profile decision, and routes. No Hub
kernel is downloaded during warmup. The current profile was measured with
Qwen3-30B TP1 shapes, but the model name is provenance rather than a selection
key.

## DeltaKV Checkpoints

Compressor-backed DeltaKV runs require a local checkpoint directory matching
the base model. Pass that directory as `deltakv_checkpoint_path`. Do not reuse a
compressor checkpoint with a different base model unless it was trained for that
model and its layer/head dimensions match.

## Minimal Usage

```python
from sparseengine import LLM, SamplingParams

llm = LLM(
    "/path/to/Qwen2.5-7B-Instruct-1M",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    engine_prefill_chunk_size=4096,
    sparse_method="omnikv",
    full_attention_layers="0,1,2,4,7,14",
    decode_keep_tokens=2096,
)

outputs = llm.generate(
    prompts=["Write a short story about sparse attention."],
    sampling_params=SamplingParams(temperature=0.7, max_tokens=128),
)
print(outputs[0]["text"])
llm.exit()
```

## Key Parameters

SparseEngine runtime knobs are defined in `src/sparseengine/configs/groups.py` and
`runtime.py` and can be passed unchanged as keyword args to `LLM(...)`.
`sparse_method` and `engine_prefill_chunk_size` are canonical names. Legacy
names such as `sparse_method`, `engine_prefill_chunk_size`, `num_top_tokens`,
`model_cls`, and `compressor_path` are rejected at the runtime boundary.

Common knobs:

- `tensor_parallel_size`: number of GPU ranks to spawn.
- `gpu_memory_utilization`: fraction of total GPU memory to allocate for the KV cache.
- `max_model_len`: max prompt plus generated tokens allowed.
- `engine_prefill_chunk_size`: SparseEngine prefill scheduling and memory-admission chunk size.
- `max_num_batched_tokens`: per-step token budget.
- `max_num_seqs_in_batch`: prefill batch limit and the default decode batch limit.
- `max_decoding_seqs`: optional decode batch override; when set, its exact value is included in the decode CUDA Graph capture buckets.

Sparse knobs:

- `sparse_method`: method selector.
- `deltakv_checkpoint_path`: local DeltaKV compressor checkpoint directory or file.
- `sink_keep_tokens`: always-kept prefix/sink tokens.
- `recent_keep_tokens`: always-kept recent tail tokens.
- `decode_keep_tokens`: shared sparse top/important token budget.
- `full_attention_layers`: comma-separated layer indices or list of layers that run full attention.

## Documentation Map

- [Core sparse methods](../features/sparse-methods.md)
- [Benchmarks](../benchmarking/README.md)
- [DeltaKV](../features/deltakv.md)
- [Troubleshooting](troubleshooting.md)

### Scheduler phase affinity

Phase affinity is enabled by default with `favor_min_decoding_seqs` set to
`ceil(max_decoding_seqs * 0.75)`. Set an explicit integer from zero to
`max_decoding_seqs` to override it; `0` restores strict prefill priority. Pass
it as an LLM Python argument, as
`--favor-min-decoding-seqs 4` to the OpenAI server, or via
`--hyper-params '{"favor_min_decoding_seqs":4}'` to the efficiency probe.

During decode, retain the phase while the next executable decode batch meets the
threshold; otherwise try prefill. Counting includes mixed-length requests and respects
the decode batch cap and writable KV capacity, including recompute decode work.
Resident KV rows are not the count. Without waiting prefill, decode continues
without waiting to fill a batch. After 60 seconds of prefill waiting, override the
decode preference and try the oldest admissible prefill first. Capacity and replay
recovery constraints still apply: this grants an execution opportunity, not a
completion deadline.

Once prefill starts, continue chunks and admit eligible arrivals without fixing
the request set or switching back merely because decode reaches the threshold.
Return to decode when no prefill can execute. New-request admission limits do not
block admitted partial prefills. There is no prefill phase duration cap; continuously
arriving short requests that finish immediately can still delay decode.

Fresh waiting starts at scheduler admission, partial waiting after the previous
chunk completes, and recompute waiting at preemption. Queue scans and unsuccessful
rescheduling preserve timestamps; prefill execution ends that wait and cancellation
clears it. This policy is local to one scheduler, without DP phase coordination or
mixed prefill/decode execution.
