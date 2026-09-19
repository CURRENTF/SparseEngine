# TP radix prefix cache 与 decode CUDA Graph 使用 rank 本地镜像 cache

本决策适用于 radix prefix cache。Chain 生命周期和准入见
[ADR 0004](0004-linear-chain-prefix-cache.md)。

Sparse-Engine 支持同时启用 `vanilla`、`omnikv` 或 `quest`、TP、prefix cache 和 decode CUDA Graph。其实现方式是让 prefix cache 保持 rank 本地存储但在逻辑上互为镜像：每个 rank 存储自己的 KV payload，同时共享由相同 token path 和 fingerprint 派生的稳定 block ID。decode CUDA Graph key 仍然只由 shape 和 execution family 构成，不包含 prefix hit length 或 block ID；prefix hit 通过现有 static decode preparation buffer 更新 row、slot、page 和 context metadata。prefix cache 控制 API 在所有 rank 上执行，并在同步 worker 故障后返回 rank 0 的逻辑视图；这一组合会拒绝 `decode_graph_capture_sampling=True`。当前模型、方法及拓扑的兼容性由 `configs/prefix_cache.py`、`configs/cuda_graph.py` 和方法注册表检查，不代表已经完成对应验证。
