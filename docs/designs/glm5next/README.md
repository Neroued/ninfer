# GLM5Next 设计与实施索引

本目录集中保存需要由具体实现接管的设计与执行清单。长期参考独立维护：

- [模型语义](../../maintainer/glm5next-model.md)：完整数学、逻辑权重、因果关系、Main/MTP/Vision 和历史依赖。
- [量化与数值表示](../../maintainer/glm5next-quantization.md)：源表示、选定量化、精确编码、计算精度、显式舍入及权重容量。

阅读实现设计前先读相应长期参考。这里的接口名称和代码片段是待落地合同，不能
据此宣称已有运行能力；已确认的模型／数值决策不因实现进度或文档迁移而重开。

## 1. 文档与状态

以下为本地已记录状态。基础执行任务的完成证据只在对应 checklist 条目维护，
未勾选不等于现有实现不支持，也不从本地未更新记录推断远端执行进度。

| 内容 | 决策状态 | 实施状态／入口 |
|---|---|---|
| 架构、源表示、量化与数值政策 | 已确定 | 长期参考；新量化产物尚未生成或做模型精度评测 |
| [双 Spark 分工与资源](parallel.md) | 模块分工、owner、mHC 复制已确定 | 待通信与模型执行接入；容量为推导，暂无 GLM 双机端到端测量 |
| [基础 Op 清单](op-checklist.md) | 已有范围固定 | 沿用原 20 个条目的状态、验证和性能证据 |
| [KDA ReplaySSM](replayssm-kda.md) | 内部 gate、recurrent decode/verify/fold、每轮即时 fold 已确定 | 设计待实现／验收，另行下发 |
| [Kpool Sparse MLA](kpool-sparse-mla.md) | 接口、page、codec、selection/tail 已确定 | 设计待实现／验收，另行下发 |
| [mHC](mhc.md) | 五个入口、布局、精度、融合与复制放置已确定 | 设计待实现／验收，另行下发 |
| Converter 与模型接入 | 数值规则已定，接入未完成 | block-FP8 编码入口、GLM converter/binding 与运行路径待实现 |

量化格式存在不等于转换方法存在：当前 block-FP8 converter 只有 `import_encoded`，
已定 `fp8_block_maxabs` 需要实现。KDA 通用入口用于算法／性能验证，不替代生产
ReplaySSM 的调用约束。各设计的“待实现”范围与现有通用算子能力分开解释。

## 2. 当前实施范围与待设计事项

现有基础清单仅覆盖 TP2/EP2 的纯投影、无状态融合、dense/shared MLP 和 Main/MTP
本地 MoE。KDA 状态链、MLA/Indexer、mHC 不自动追加；实现与调优需明确下发任务。
本次文档整理不修改条目 ID、状态、profile 或验收范围，也不触发 `.50` 同步。

仍需设计或落实的工作：

- 跨模块 collective、pack/unpack、通算融合、模型 Graph 和跨 owner 前缀恢复。
- MTP stem、embedding、Vision 的本地放置与接口；分片 logits、采样、acceptance 和 scoring 交接。
- 完整 KDA 输入／输出融合及长 prefill 的执行选择；已定 decode/verify/fold 的 recurrent 约束保持。
- Converter、artifact binding、模型加载／Frontend／Engine 接入和完整状态组合。

设计决定、实现完成和性能验证是三种不同状态。算子 kernel 完成不自动表示模型
接入完成，本地算子性能也不自动表示双机模型吞吐收益。

## 3. 文档职责

| 内容 | 当前唯一归属 |
|---|---|
| 模型数学、逻辑角色与历史依赖 | 长期模型参考 |
| 表示、codec、activation permission、可观察舍入 | 长期数值参考；已有通用格式直接引用 |
| 分片、owner、通信交接与部署预算 | parallel.md |
| 具体 Op/views、physical layout、写域、alias、融合与验收 | 对应 family 设计 |
| 执行条目与完成证据 | op-checklist.md |
| 跨主题进度与未决范围 | 本索引 |

接口表为自洽所列的 dtype/shape 应引用并满足长期语义；不在不同文档维护两份
encoder、完整模型数学或不同精度政策。实现设计的数学组合可引用长期公式，再明确
自身的表示边界和状态效果。来源差异保留必要说明，旧讨论过程由 Git 保存。

## 4. 接管与退出

| 设计内容 | 落地后的权威位置 |
|---|---|
| 公共输入输出、alias、数学组合与状态合同 | 实际 `include/ninfer/ops/` 声明 |
| 编码和物理布局 | 通用格式／布局文档及对应实现 |
| 数学、精确提交和边界验收 | 对应资格测试 |
| 当前层序、资源和 Graph 交接 | 模型执行代码与届时建立的 `glm5next-execution.md` |
| 性能结果 | 对应 benchmark／性能证据 |

一个部分实际落地并完成验收后，更新正式归属和链接，再移除已被接管的设计段落。
跨 Op 组合尚未接入时保留剩余设计；全部必要内容被接管后删除该文件。清单全项完成
且证据已归位后删除清单。长期模型和数值选择继续维护，不随设计退役而丢失。

此处不提前创建声称已支持的 execution 文档，不保留旧路径别名或并存的旧版规格。
历史由 Git 保存；[通用分布式设计](../../distributed-execution-design.md)、Qwen 文档和
现有引擎合同继续在原位置维护。
