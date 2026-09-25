# KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction [NeurIPS'25 Oral]

> SparseEngine vendors the runtime subset only. The upstream benchmark code is
> intentionally omitted; use `../../benchmark/` for repository-owned runs.

[Blog](https://janghyun1230.github.io/kvzip/)

<img src="./images/method.png" width="800">


## News
- **01/2026**: We've released [Fast KVzip](https://github.com/Janghyun1230/FastKVzip), which eliminates compression overhead and enhances both prefill and decoding efficiency!
- **09/2025**: 🎉 KVzip has been accepted at NeurIPS 2025 as an **Oral Presentation**! 
- **07/2025**: [NVIDIA KVpress](https://github.com/NVIDIA/kvpress) adds support for KVzip (see also [Leaderboard](https://huggingface.co/spaces/nvidia/kvpress-leaderboard)).
- **07/2025**: KVzip is presented at the [ES-FoMo III ICML Workshop](https://es-fomo.com).
- **05/2025**: arXiv preprint is released.


## Highlights
- KVzip compresses the KV cache to support **diverse future queries**.
- [Context-dependent] Achieve a **3–4× reduction in KV cache size** and a **2× decrease in decoding latency**, with minimal performance degradation.
- [Context-independent] Enhance [DuoAttention](https://github.com/mit-han-lab/duo-attention)-style head-level KV compression, using only **a few forward passes within one minute** for head-level importance-score optimization (100x faster).
### Benchmarking on a query-agnostic setting
- Tasks: [SQuAD](https://huggingface.co/datasets/rajpurkar/squad), [NIAH](https://github.com/gkamradt/LLMTest_NeedleInAHaystack), [SCBench](https://github.com/microsoft/MInference/tree/main/scbench), [GSM8K](https://huggingface.co/datasets/openai/gsm8k/viewer/main/train?row=7294). 
- Model: [Qwen2.5-7B-Instruct-1M](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

<img src="./images/benchmark.png" width="800">


## Installation
We used CUDA 12.1 and Python 3.10
```bash
cd KVzip
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
make i
```
- To use [QServe](https://github.com/mit-han-lab/omniserve) quantization, please follow [`./model/quant_model`](https://github.com/snu-mllab/KVzip/tree/main/model/quant_model).


## Quick Start
```python
from model import ModelKVzip

model = ModelKVzip("Qwen/Qwen2.5-7B-Instruct-1M")
context = "This is my basic profile. My name is Kim living in Seoul. My major is computer science."
queries = ["What is my name?", "Do I live in Seoul?"]

kv = model.prefill(context, load_score=False)  # prefill KV cache + importance scoring
kv.prune(ratio=0.3)  # compression ratio, evict 70% KV

for q in queries:
    query_ids = model.apply_template(q)
    output = model.generate(query_ids, kv=kv, update_cache=False)  # efficient inference
    print(q, output)
```
- Supported models are listed in [`model/load.py`](https://github.com/snu-mllab/KVzip/blob/main/model/load.py), including **LLaMA3, Qwen2.5/3, Gemma3**.
- Set `load_score=True` to eliminate compression overhead. This enables context-independent KV eviction, with a trade-off in compression ratio of `ratio=0.6`.
- After generation, KV pairs corresponding to the queries and generated tokens are selectively evicted from the cache for further processing. Set `update_cache=True` to enable multi-turn inference, retaining full interaction histories throughout the inference. 

## Applying to New Models
To integrate KVzip for a new model, you will need to update the following files:
- `attention/attn.py`  
  Modify the attention forward pass logic as needed. In certain cases, updates to kvcache.py and score.py may also be required.
- `model/monkeypatch.py`  
  Implement model-specific monkey patching for integration.
- `model/template.py`   
  Define the model's system prompt and chat formatting templates.


## Citation
```bibtex
@article{kim2025kvzip,
        title={KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction},
        author={Kim, Jang-Hyun and Kim, Jinuk and Kwon, Sangwoo and Lee, Jae W and Yun, Sangdoo and Song, Hyun Oh},
        journal={Advances in Neural Information Processing Systems},
        year={2025}
}
```

## License
MIT License
