# SparseEngine 文档

[English](../en/README.md) | 简体中文

本目录包含面向用户的稳定文档：安装指南、功能说明、架构说明、配置参考和基准测试运行手册。

`docs/` 应专注于稳定的项目指南、约定和运行手册。不要在这里添加本地实验台账；当面向仓库的结果声明需要证据时，应直接引用仓库中的具体产物。

## 稳定文档

- [快速开始](getting_started/README.md)：安装、checkpoint 下载和最小 SparseEngine 使用示例。
- [功能](features/README.md)：稀疏方法分类、DeltaKV 说明和 Qwen3MoE 专家并行。
- [设计](design/README.md)：仓库布局、运行时流程和方法所有权边界。
- [配置](configuration/README.md)：规范的运行时参数和原生运行时语义。
- [基准测试](benchmarking/README.md)：吞吐量、LongBench、MathBench / AIME /
  MATH-500、SCBench、Claw-Eval、多模态、RULER core、NIAH 和回归基准入口。
- [治理](governance/README.md)：研究代码可靠性规则。

## 参考文档

- [支持的模型](features/supported-models.md)
- [研究代码指南](governance/research-code-guidelines.md)
- [运行时参数语义](configuration/runtime-parameter-semantics.md)
- [SparseEngine 控制图](design/control-map.md)

## 基准测试运行手册

- [基准测试目录](benchmarking/README.md)
- [效率与吞吐性能套件](benchmarking/efficiency.md)
- [SparseEngine 回归测试](benchmarking/sparseengine-regression-tests.md)
- [多模态基准测试](benchmarking/multimodal/README.md)
