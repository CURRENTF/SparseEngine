# 架构

Sparse-Engine 只有一个位于 `src/sparseengine/` 下的原生 runtime：sparse-first
inference engine，包含自己的 scheduler、model runner、cache manager、sparse
controller、模型定义和 Triton kernel。

## Sparse-Engine 流程

Sparse-Engine inference path：

1. `sparseengine.LLM(model, **kwargs)`
2. `LLMEngine` 按规范 `Config` 字段校验 kwargs。
3. `src/sparseengine/config.py` 验证 engine config 和方法兼容性。
4. `engine/cache_manager/base.py` 中的 `CacheManager.create()` 选择 cache
   manager，`engine/sparse_methods/factory.py` 选择逻辑方法 runtime。
5. `Scheduler`、`ModelRunner`、通用 `SparseController`、选中的
   `SparseMethodRuntime`、kernel 和 cache manager 执行 prefill 与 decode。

`src/sparseengine/layers/attention.py` 有意保持通用。它写入 K/V，向 sparse controller 请求 logical read view，并允许 cache manager 通过 `build_decode_view(...)` 等 hook 自定义 decode-time view。

## 方法所有权

新增 Sparse-Engine 稀疏方法应遵循 cache-manager-first 设计和
[稀疏方法运行时架构](sparse-method-runtime.md)：

- 持久物理缓存和跟随 Prefix Cache 的元数据属于
  `src/sparseengine/engine/cache_manager/`。
- 当前推理步骤中的打分、选择和跨层协调属于
  `src/sparseengine/engine/sparse_methods/` 下的 runtime。
- `src/sparseengine/engine/sparse_controller.py` 保持稳定的统一入口，不得增加
  方法名热路径分支。
- `src/sparseengine/layers/attention.py` 只应调用 generic hook 或 shared kernel。
- Public runtime 参数应使用[运行时参数语义](../configuration/runtime-parameter-semantics.md)中记录的规范名称。

新增一等方法时，请使用仓库内的 `$add-sparse-method` skill。该 skill 编码了预期的文件位置、cache-manager hook 和验证流程。

## Scheduling 所有权

Prefill scheduling 是方法 contract 的一部分。默认 policy 位于 `src/sparseengine/method_registry.py`；`Config` 负责解析和验证 policy；`src/sparseengine/engine/scheduler.py` 实现 scheduling 行为。

engine 当前支持：

- `all_chunked`：所有 prefill request 都通过常规 scheduler limit 进行 chunking 和 batching，每个 sequence 每步最多调度 `engine_prefill_chunk_size` 个 token。
- `long_bs1full_short_batch`：在附加受支持的 prefix 后，residual token 数不超过 `long_prefill_offload_threshold` 的 request 使用 atomic full prefill，并且可以互相 batch；更大的 residual 单独运行 chunked RawKV offload prefill，每个 chunk 不超过 `engine_prefill_chunk_size`。

DeltaKV 和 PyramidKV 通过 `prefill_execution_mode()` 实现 long branch。threshold 默认是 `65536` token（64K）。未设置 `engine_prefill_chunk_size` 时，`Config` 默认令其等于 threshold；显式设置时必须满足 `0 < engine_prefill_chunk_size <= long_prefill_offload_threshold`。必要时，`Config` 会把 `max_num_batched_tokens` 提高到 threshold，使边界大小的 full prefill 保持 atomic。PyramidKV 根据 chain prefix attach 后的 residual 应用该边界。DeltaKV 不提供 prefix caching；如果 attached-prefix prefill 到达其 cache manager，会快速失败，因为其 compressed row state 没有 prefix residency contract。

不要在 benchmark script 或一次性 config default 中编码某个方法的 prefill policy。应把 method-to-policy mapping 添加到 registry，并更新 `tests/test_prefill_schedule_policy.py`。

## 重要文件

- `src/sparseengine/configs/groups.py` 与 `runtime.py`：定义规范 runtime 字段、default 和 validation；public 与 internal 名称完全一致。
- `src/sparseengine/config.py`：`Config` 的兼容导入入口。
- `src/sparseengine/method_registry.py`：支持的方法名和 prefill policy default。
- `src/sparseengine/engine/cache_manager/base.py`：公共 CacheManager 接口、构造入口
  和通用 hook。
- `src/sparseengine/engine/sparse_controller.py`：与方法无关的引擎统一入口。
- `src/sparseengine/engine/sparse_methods/`：方法 runtime interface、factory、
  当前步骤的打分/选择协调和各类方法 runtime。
- `src/sparseengine/engine/scheduler.py`：prefill/decode scheduling 和 admission。
- `src/sparseengine/layers/attention.py`：通用 K/V storage 和 attention compute path。
- `benchmark/`：LongBench、MathBench、SCBench、NIAH 和多模态 benchmark 入口。
- `benchmark/model_adapters/sparseengine.py`：文本 benchmark 共用的原生 generation adapter。
