# 未完成事项

这里保留仍需处理或取得当前证据的问题，不作为稳定架构说明，也不表示 GitHub 状态已更新。
完成记录与旧实现补丁由 Git/Issue 历史保存。原报告没有当前复现的条目继续明确标为待核实。

## 本轮处理清单

按以下顺序逐项讨论、实施；完成后勾选。

- [x] **1. QSA 成员模板调用语法** — [#428](https://github.com/Neroued/ninfer/issues/428)

  已补齐 `indexer/select.cuh` 的依赖成员模板调用消歧义关键字 `template`；其余三处
  `jobs.build` 的对象类型不依赖模板参数，无需修改。RTX 5090 dev 完整构建通过（CUDA 13.4）；
  当前环境未安装 CUDA 13.2，未在该版本复测。

- [x] **2. 流式请求准入超时的 HTTP 状态** — [#420](https://github.com/Neroued/ninfer/issues/420)、[#137](https://github.com/Neroued/ninfer/issues/137)

  Engine 提供不消费输出的准入等待，三个 HTTP 入口在准入成功后才提交 SSE。
  真实服务验证：Chat、Responses 排队超时返回 HTTP 503；Anthropic 沿用 HTTP 504 / `timeout_error`。
  正常流式与非流式请求、排队取消和等待后准入均通过；Engine 检查覆盖事件顺序、错误保留、立即完成及 handle 清理。

- [x] **3. 请求终结与统计发布顺序** — [#422](https://github.com/Neroued/ninfer/issues/422)

  Engine 统一先解除逻辑成员、发布统计，再公开终态；整体失败统一清理后分发原始错误。
  完整构建及 RTX 5090 真实模型检查通过，覆盖完成、取消、排队/原生错误、执行失败和暂停/Replay 取消。

- [x] **4. Thinking 统计与 budget 解耦** — [#373](https://github.com/Neroued/ninfer/issues/373)

  语义计数独立于 budget 和输出呈现；模型生成与控制注入分别计数。
  Qwen、GLM 的自然关闭、raw、隐式工具边界、preview 丢弃及预算控制测试通过。

- [x] **5. HTTP listener 端口复用** — [#341](https://github.com/Neroued/ninfer/issues/341)、[PR #340](https://github.com/Neroued/ninfer/pull/340)

  Linux listener 使用 `SO_REUSEADDR`，不启用 `SO_REUSEPORT`。实际连接测试覆盖双向切换、
  NInfer 自身重绑和活动端口独占，原有连接存活检测测试通过。

- [x] **6. 上下文超限错误信息** — [#401](https://github.com/Neroued/ninfer/issues/401)

  Qwen、GLM 与 Engine 共用错误格式，区分完整 token 计数与提前拒绝时的下界，保留分词早停。
  文本、原始 token、媒体及协议测试通过；HTTP 400 与 `context_length_exceeded` 保持一致。

- [x] **7. Workspace 布局分配异常安全** — [PR #297](https://github.com/Neroued/ninfer/pull/297)

  对齐与大小检查成功后才提交 cursor/peak；现有内存测试覆盖字节、Tensor 分配溢出后的
  继续分配及 scope 回退，检查通过。

## 其余正确性与资源行为

| 来源 | 剩余工作 |
|---|---|
| [#399](https://github.com/Neroued/ninfer/issues/399) grammar violation | 取得可回放请求及失败 token/matcher 状态，区分模型输出、强制控制和内部状态错误，再决定正确的错误隔离边界。原触发原因未确认。 |
| [#391](https://github.com/Neroued/ninfer/issues/391) NVFP4/VFIO 崩溃 | 需要报告环境的首个失败 kernel 与具体负载；现有异步报错位置不能证明根因。 |
| [#394](https://github.com/Neroued/ninfer/issues/394) 深前缀保留 | 在当前统一池下固定多轮负载和保留前提，确认是否存在有效历史无法成为后续来源；不沿用旧池诊断。 |
| [#371](https://github.com/Neroued/ninfer/issues/371) 取消后的续接 | 检查接管旧 checkpoint 后取消、编辑和重试时上一完整恢复点的保留语义。 |
| [#349](https://github.com/Neroued/ninfer/issues/349) proposal CDF | Candidate path 的 CDF 未命中仍回退最后项，核对累计舍入与零概率尾项边界；处理实际 draw 与 q 的一致性。 |
| [#381](https://github.com/Neroued/ninfer/issues/381) Direct I/O | 按 read_direct 与调用者的真实短读合同检查，不把一次 pread 自身视为完整读取保证；buffered 存储支持另行决定。 |

## 性能工作

- [#356](https://github.com/Neroued/ninfer/issues/356)：评估 NVFP4 长上下文 tiled attention 的
  KV 分区与合并，沿当前 QK 数值合同优化。
- [#364](https://github.com/Neroued/ninfer/issues/364)：RTX 5090 NVFP4 down projection 小 T 的
  实际利用率与候选分派；重新以当前生产入口测量，不沿用旧 tile 结论。
- [#370](https://github.com/Neroued/ninfer/issues/370)：权重 job 并行的收益、临时张量峰值和 CPU
  线程竞争；已有 I/O 预取不等于多个完整量化 job 并行。

### MLA 性能

BF16/FP8 Kpool MLA 已接入模型，完整目标区间的 roofline 验收仍未完成：

- 长上下文 Prefill 的分散 Kpool4 读取、跨 query 实际复用与因子映射成本。
- 长上下文多 batch Indexer 的 key 重扫、候选访问与 Top512 成本。
- Fresh Prefill 与已有长前缀的小 chunk 分开分析；展开 K/V 的方案包含额外生成、存储与消费成本。
- 保留不同 latent/index 格式、ragged batch 和 selection reuse 的真实曲线；只按明确相同的
  selection 复用，不把存储压缩比或单阶段事件时间解释成完整分支加速。

测量入口为 [Kpool benches](../bench/ops/softmax_attention/kpool_mla/)，
方法见 [Op 开发](maintainer/op-development.md#调优与测量)。已有性能不足不因删除交付清单而视为完成。

## 尚需决策

- [Qwen4-27B 架构与量化](designs/qwen4-27b-rtx5090.md)、
  [Op 需求](designs/qwen4-27b-op-checklist.md)：独立的新适配任务。
- [#395](https://github.com/Neroued/ninfer/issues/395)：是否扩展长工具名及其约束、历史和协议边界。
- [#348](https://github.com/Neroued/ninfer/issues/348)：Responses 新字段与外部工具执行责任分别评估。
