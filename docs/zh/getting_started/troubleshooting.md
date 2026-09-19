# 故障排查

## `SamplingParams` 不允许 greedy decode

`SamplingParams.temperature` 必须大于 `1e-10`。如需近似 greedy decode，请使用 `1e-5` 之类的极小 temperature。

## `Insufficient KV cache slots to admit prompt`

engine 无法为 prompt 或 prompt chunk 分配足够的 KV slot。请提高 `gpu_memory_utilization`，减小 `max_model_len` 或 batch size，或者降低 keep-token budget。

## TensorRT-LLM DeepGEMM 编译缓存

CUDA worker 使用独立的 TensorRT-LLM DeepGEMM 缓存目录，避免冷启动时并发编译写入。
缓存根目录依次取 `SPARSEENGINE_TRTLLM_DG_CACHE_ROOT`、`TRTLLM_DG_CACHE_DIR`、
`${XDG_CACHE_HOME:-~/.cache}/sparseengine/trtllm-deepgemm`。两个显式变量都表示根目录；
每个 worker 独占一个带文件锁的子目录，并在启动时打印实际路径。主目录空间有限时，应将根目录设在可写的数据盘。

并发 worker（包括 rank 相同的不同服务实例）在整个进程生命周期持有各自的文件锁。
进程退出后，后续 worker 可以复用已释放目录。已安装包与工具链元数据、DeepGEMM 头文件用于区分缓存目录。
自定义工具链或未纳入包元数据的外部头文件发生修改时，应为新构建选择新的缓存根目录。
文件系统必须支持参与主机之间的建议式文件锁。
同一进程连续创建引擎时复用原目录，因为上游编译器会记住首次读取的路径；修改根目录需要新进程。
请在启动引擎或直接使用此外部 kernel 前设置缓存变量。如果同一进程还直接调用该 kernel，
应确保其编译器没有在引擎配置缓存前初始化。

进程退出后保留目录供复用；只有使用同一根目录的所有进程停止后，才能清理目录和锁文件。
不要删除仍在使用的锁文件。已有 UUID 目录不会自动迁移。
其他 FlashInfer 和 Triton 缓存不受影响。此措施仅隔离缓存写入，编译和 kernel 错误仍会明确抛出。
