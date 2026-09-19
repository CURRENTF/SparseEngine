# FlashPrefill V2

FlashPrefill V2 是通过 `prefill_sparse_method="flashprefill_v2"` 选择的可选
sparse-prefill attention Provider。该选择器与 `sparse_method` 正交：vanilla、
OmniKV、QuEST、SnapKV 或 H2O 继续拥有 cache 分配、prefill page table 和
decode 行为，FlashPrefill V2 只替换 prefill attention 计算。

## 安装已验证的上游版本

Adapter 要求从上游 revision
`75b58f2ecdba1c269a87dd34d8f1ae57bef50c57` 构建
`flashprefill==3.0.0`。本地已验证的 binary contract 是 CUDA SM90、BF16、
head dimension 128、causal varlen paged prefill 和 page size 1。

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

Package 缺失、版本不兼容或 binary 加载失败会在 Provider 解析阶段明确报错；
Sparse-Engine 不会把请求的稀疏语义静默替换为 dense Provider。

## 配置

`flashprefill_v2_abs_threshold` 没有通用默认值，必须显式传入；数值越大，选择的
block 越少。每个模型都需要独立校准，并在报告性能结果时同时给出匹配的质量回归。

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

在 explicit-KV MHA 模型上，支持的 cache/decode 组合是 `vanilla`、`omnikv`、
`quest`、`snapkv` 和 `h2o`。GLM-4.7-Flash 等 MLA latent 模型会在初始化阶段拒绝
该配置。H2O 未显式设置 `prefill_sparse_method` 时会默认解析为 `h2o_prefill`；改为
`flashprefill_v2` 只替换 prefill attention 计算。H2O 仍通过 method-owned
posthoc scorer 计算重要性分数，并执行最终 decode cache 准备；它不会同时执行
`h2o_prefill` 的中间 chunk 压缩，因为两者是同一 prefill 轴上的备选方法。SnapKV
同样保留已有的 posthoc 评分和压缩生命周期。这些额外 scorer 仍属于 cache method
的开销，匹配性能测量时必须计入。H2O decode 评分和周期淘汰默认关闭，可通过
独立的 `h2o_decode_eviction=True` 开启；开启时强制使用概率评分。

示例中的 threshold 只适合作为 Qwen3-4B 的校准起点，不是模型无关的推荐值。
Prefix-cache hit 已支持：CacheManager 提供包含完整 prefix 的物理 page table 和
cache length，上游 kernel 通过 cache length 减 query length 得到 attached-prefix
长度。当前 adapter 不会向上游 kernel 请求 attention score 或 softmax LSE 输出。
