# 块级基数树 prefix cache

Sparse-Engine 的 radix prefix cache 使用块级基数树索引，而不是 token 级基数树匹配。`RadixPrefixIndex` 负责块标识、匹配、驻留元数据、淘汰优先级和可序列化的控制面状态；cache manager 则负责方法特定的载荷，例如 token slot 或 QuEST chunk alias。prefix cache 控制 API 可以检查、删除子树和调整子树优先级，但不会暴露树节点、tensor 或载荷；这些 API 操作通过 engine dispatcher 执行，而不是由 HTTP handler 直接修改 cache 状态。负数淘汰优先级表示硬保护，首个版本有意不提供全局 reset。

本决策适用于 radix 模式。保存压缩后可变 chain 状态的方法使用独立的
[线性 chain cache](0004-linear-chain-prefix-cache.md)。Radix 复用管理已缓存
的前缀块，不按每个请求的完整未来输出长度预留容量。
