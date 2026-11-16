# 未完成事项

这里保留仍需处理或取得当前证据的问题，不作为稳定架构说明，也不表示 GitHub 状态已更新。
完成记录与旧实现补丁由 Git/Issue 历史保存。原报告没有当前复现的条目继续明确标为待核实。

## 待核实的正确性与资源行为

| 来源 | 剩余工作 |
|---|---|
| [#399](https://github.com/Neroued/ninfer/issues/399) grammar violation | 取得可回放请求及失败 token/matcher 状态，区分模型输出、强制控制和内部状态错误，再决定正确的错误隔离边界。原触发原因未确认。 |
| [#391](https://github.com/Neroued/ninfer/issues/391) NVFP4/VFIO 崩溃 | 需要报告环境的首个失败 kernel 与具体负载；现有异步报错位置不能证明根因。 |
| [#394](https://github.com/Neroued/ninfer/issues/394) 深前缀保留 | 固定多轮负载和保留前提，确认深前缀在当前输入身份、保留与回收策略下是否仍有遗漏。 |
| [#371](https://github.com/Neroued/ninfer/issues/371) 取消后的续接 | 检查接管旧 checkpoint 后取消、编辑和重试时上一完整恢复点的保留语义。 |
| [#349](https://github.com/Neroued/ninfer/issues/349) proposal CDF | 核对累计舍入下实际 draw 与输出 q 的一致性。当前无命中时已回退最后一个正概率项，旧的固定尾项描述不再适用。 |
| [#381](https://github.com/Neroued/ninfer/issues/381) Direct I/O | 按 read_direct 与调用者的真实短读合同检查，不把一次 pread 自身视为完整读取保证；buffered 存储支持另行决定。 |

## 性能工作

- [#356](https://github.com/Neroued/ninfer/issues/356)：评估 NVFP4 长上下文 tiled attention 的
  KV 分区与合并，沿当前 QK 数值合同优化。
- [#364](https://github.com/Neroued/ninfer/issues/364)：RTX 5090 NVFP4 down projection 小 T 的
  剩余 roofline 差距；按当前生产分派与完整入口成本定位。
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
方法见 [Op 开发](maintainer/op-development.md#调优与测量)。

## 尚需决策

- [Qwen4-27B 架构与量化](designs/qwen4-27b-rtx5090.md)：候选几何、正式源权重与最终 recipe 的确认。
- [#395](https://github.com/Neroued/ninfer/issues/395)：是否扩展长工具名及其约束、历史和协议边界。
- [#348](https://github.com/Neroued/ninfer/issues/348)：Responses 新字段与外部工具执行责任分别评估。
