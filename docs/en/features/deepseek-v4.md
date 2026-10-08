# DeepSeek V4 Flash native inference

[简体中文](../../zh/features/deepseek-v4.md)

Experimental support for the original DeepSeek-V4-Flash-0731 checkpoint on
CUDA SM90. Use `sparse_method="deepseek_v4"`; this is the model's native
attention and cache format, rather than a generic KV eviction policy.
The checkpoint's block FP8 projections, MXFP4 routed experts, and mixed
BF16/FP32 parameters are loaded in their original formats. MTP weights are
skipped; speculative decoding is unavailable.

## Environment

Install the base CUDA environment described in the repository README first.
Add the pinned Hadamard dependency in that activated environment:

```bash
pip install --no-build-isolation -e ".[deepseek-v4]"
```

The native path requires FlashInfer 0.6.18.post1 or newer in the 0.6 series,
SGL Kernel 0.4.5, and the CUDA compiler for upstream JIT operators. AGRS does
not require the optional DeepEP package or a vLLM runtime installation.

## Run

Attention TP and expert TP must both be 1, so `expert_parallel_size` equals
`data_parallel_size`. The example uses two attention replicas with AGRS.
Sequence and token budgets apply to each replica.

Use the checkpoint's `encoding/encoding_dsv4.py` to encode chat messages.
Pass the resulting token IDs without adding tokenizer special tokens again.

```python
from pathlib import Path
import sys

from sparseengine import LLM, SamplingParams

model_dir = Path("<MODEL_PATH>")
sys.path.insert(0, str(model_dir / "encoding"))
from encoding_dsv4 import encode_messages

llm = LLM(
    str(model_dir),
    sparse_method="deepseek_v4",
    tensor_parallel_size=1,
    data_parallel_size=2,
    expert_parallel_size=2,
    moe_backend="agrs",
    max_model_len=4352,
    max_num_batched_tokens=512,
    engine_prefill_chunk_size=512,
    max_num_seqs_in_batch=2,
    max_decoding_seqs=2,
    decode_graph=True,
    enable_prefix_caching=True,
    prefix_cache_block_size=128,
)
try:
    prompt = encode_messages(
        [{"role": "user", "content": "What is the capital of France?"}],
        thinking_mode="chat",
    )
    tokens = llm.tokenizer.encode(prompt, add_special_tokens=False)
    outputs = llm.generate([tokens], SamplingParams(temperature=0, max_tokens=64))
    print(outputs[0]["text"])
finally:
    llm.exit()
```

## Cache and execution

Chunked prefill, eager decode, decode CUDA Graphs, and radix prefix reuse
share the native sliding-window and compressed-history cache. The checkpoint
controls the compression ratios and sparse selection budget; generic
`sink_keep_tokens`, `recent_keep_tokens`, and `decode_keep_tokens` do not tune
this method. Other sparse methods and sparse prefill overrides cannot be
combined with it.

Prefix reuse preserves the sliding window, compression carry, and compressed
pages. Reusable blocks remain on GPU and count toward cache capacity.
`prefix_cache_max_blocks` bounds retained snapshots; reuse is opportunistic
when snapshots or pages cannot be retained. Chain prefix caching and prefix
offload are unavailable. Set `decode_graph=False` to run eager decode.
