# Operator Provider 选择架构

SparseEngine 在模型构造阶段只解析一次 operator：

```text
OpSpec
  -> atomic capability filter
  -> exact local profile overlay
     -> 命中：绑定 profiled dispatch plan
     -> 未命中：执行默认 portfolio policy
  -> prepare 已选择的实现
  -> 运行期禁止重新进入 resolver 或静默 fallback
```

## Atomic Capability

`supports()` 只回答 atomic 实现能否在当前平台上正确满足 operation contract，
并返回一种有类型的状态：

- `SUPPORTED`：实现正确满足契约。
- `UNSUPPORTED_CONTRACT`：正常的语义或平台不匹配。
- `DEPENDENCY_ABSENT`：可选上游依赖未安装。
- `DEPENDENCY_BROKEN`：已安装依赖的版本、ABI 或 callable contract 损坏。

本地 benchmark 覆盖不是 atomic support 状态。尤其不能因为 SparseEngine 没有
在本地测过某个 shape，就拒绝上游实现声明支持的 shape。

## Repo-Owned DSL Kernel 的可移植性

`repo_nonstandard` 描述的是 operation contract 的语义归属，不表示实现必须是
硬件特化的。对于使用 Triton 或 TileLang portable primitives 实现的 repo-owned
kernel，atomic support 应尽量宽，并由语义、tensor contract、DSL/toolchain、实际
使用的硬件特性和已知编译器限制决定。缺少某个本地 GPU 型号或 shape 的验证，不能
单独成为 device-name whitelist、compute-capability whitelist 或 atomic rejection
的理由。

SparseEngine 是研究导向项目，无法为所有硬件组合提供预先验证。对于没有已知不兼容
的设备，resolver 可以乐观绑定 portable DSL provider，并在 prepare、JIT、warmup
或首次执行时尝试编译。无法编译或运行时必须保留清晰的原始错误；不得静默重选
provider、伪造默认输出或掩盖失败。确认某项不兼容后，应优先用所需硬件特性、DSL
能力或已知 toolchain 问题描述 exclusion，而不是长期维护具体设备型号白名单。

Atomic eligibility 与验证、性能证据必须分开。实际验证过的设备、shape、dtype、
graph mode 和结果应写入可复现的 benchmark 或 validation artifact，不能根据 provider
role 自动推导。缺少记录不自动缩小 portable support。调参 profile 和性能结论仍须
限制在可复现的实测范围内；计算、访存或并行方式的算法改进，在说明泛化依据并通过
代表性正确性和性能验证后，可以按实际兼容范围进入默认 portfolio。已测 shape、
batch、TP 和 GPU 型号是覆盖记录，不是启用白名单。一个 kernel 可以在较宽
设备范围内允许尝试，但只能在已有证据的范围内声称正确性已验证或性能占优。

Repo-owned nonstandard kernel 应遵循**乐观可移植、保守声明证据**的原则。当一种
非标准契约使用 portable Triton 或 TileLang primitives 实现，且没有已知不兼容时，
它的通用 atomic provider 应允许在本地少量 GPU 之外的设备上尝试。如果该 provider
是这项契约的常规实现，就应进入对应的默认 portfolio。本地硬件覆盖有限只应体现在
验证 artifact 中，不能仅凭这一点把整个 provider 变成 exact-device profile。

`profile_only=True` 应保留给真正的特化替代方案，例如 exact-device launch schedule、
经过实测的 token-range dispatcher，或只应在已记录性能范围内覆盖通用 provider 的
混合 dispatch plan。优先采用以下结构：

```text
非标准 operation contract
  -> 默认 portfolio 中的通用 portable Triton/TileLang atomic provider
  -> 可选 exact profile，选择调优 provider、schedule 或 dispatch plan
```

不能仅因为仓库无法测试很多 GPU 型号，就用 exact profile 代替 portable default。
只要已知存在可移植实现，profile 未命中时仍应保留满足该非标准契约的有效实现。

## 默认 Portfolio

每个 operator registry 都显式持有 `PortfolioPolicy`。标准上游 provider 排在
repo portable baseline 之前；只有 attention score、特殊 cache layout、state
mutation 等上游标准算子无法表达的契约，才使用 repo-owned nonstandard
provider。Provider class 不再声明整数 priority。

不进入默认 portfolio 的 atomic provider 必须显式注册为 `profile_only=True`。
该入口只供 exact profile 引用 specialized implementation；未声明的隐藏
provider 会在 registry 校验时失败。该标记控制的是默认选择资格，而不是 provider
的正确性支持域，也不是本地验证证据的覆盖范围。

## Profile Overlay

Profile 使用独立 registry。每个 profile 声明它引用的 atomic provider，严格
匹配 device、shape、operation contract 和必要 toolchain，并构建 prepared
dispatch plan。Resolver 会先验证所有 atomic 路径的正确性资格，再调用 profile
matcher。Profile 未命中不会改变 atomic eligibility，也不会改变默认 portfolio。

Profile 覆盖顺序由 operator registry 显式声明。Profile 可以覆盖默认性能选择，
但不能定义标准上游算子的支持范围，也不能因为缺少本地性能数据而关闭一个
portable repo-owned nonstandard 路径。

## 依赖与证据

FlashInfer 与 SGL kernel 是标准 CUDA 安装的必需依赖。GPU engine 在启动 worker
前只检查 package discovery 和版本 metadata，不导入 device-bound binary；每个 rank
完成 CUDA device 绑定后，才 import 并验证这些 binary。Provider resolution 遇到任一
必需 family 缺失或损坏也会直接失败，并提示按 CUDA 版本执行
`pip install -e ".[cu129]"` 或 `pip install -e ".[cu130]"`。只有真正可选的上游依赖
缺失时，resolver 才可以绑定 repo baseline 并记录
`selection_basis=dependency_degraded`。运行期异常不会触发 provider 重选。

每份 binding report 都记录最终 provider、可选 profile、`selection_basis`、全部
atomic/profile 判断和 provider metadata。它只解释实现为何被选中，不证明数值
正确性、上游支持域或本地性能。Adapter equivalence、kernel correctness 和性能证据
必须记录在相应的可复现 validation artifact 中，声明范围不得超过实测契约。

## Ownership Rule

标准算子优先复用上游 atomic provider，repo 只维护 adapter 和 portable
baseline。新增稀疏语义、上游无法表达的 runtime contract，或具有明确泛化依据的
算法改进，都可以成为新增 repo-owned production kernel 的理由。本地 profile 只能覆盖默认选择，不能缩小上游
支持域。Repo-owned nonstandard kernel 应优先写成可移植的 Triton/TileLang
实现；它的语义可以非标准，但硬件支持范围不应被有限的本地机器资源人为缩窄。

## Phase 组合

一个语义 operator 可以组合独立选择的执行阶段。Full attention 由一个 prepared
`FullAttentionProvider` 统一拥有；prefill 和 decode 仍保留独立 atomic registry，
因为它们的 kernel、workspace、CUDA Graph contract 和支持域可能不同。FlexPrefill
这类只支持 prefill 的实现只进入 prefill portfolio，不需要伪造 decode 实现。

Full-attention provider 会在准备任一阶段前，统一检查 head、dtype、scale、causal、
page layout 和 page-table contract。之后两个 prepared phase operator 作为同一个
生命周期一次性绑定到模型，并统一关闭。两个阶段仍独立选择，因此只要共享 cache
contract 兼容，就允许组合不同的上游 prefill/decode provider。
