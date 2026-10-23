# GLM5Next 设计与实施索引

本目录集中保存需要由具体实现接管的设计与执行清单。长期参考独立维护：

- [模型语义](../../maintainer/glm5next-model.md)：完整数学、逻辑权重、因果关系、Main/MTP/Vision 和历史依赖。
- [量化与数值表示](../../maintainer/glm5next-quantization.md)：源表示、选定量化、精确编码、计算精度、显式舍入及权重容量。

完整集成从[双 Spark 支持与优化路线](roadmap.md)开始：包含当前代码起点、G0–G7
阶段、并行 Op 任务包和交接条件。[Linear/MoE 清单](op-checklist.md)按当前主线区分
已有能力与待补任务，列出 `.50` 执行所需的几何、代码入口和顺序。
[Frontend 清单](frontend-checklist.md)列出共享基础提取后，GLM 输入输出需要逐项
实现的能力、方法与完成条件。

阅读实现设计前先读相应长期参考。这里的接口名称和代码片段是待落地合同，不能
据此宣称已有运行能力；已确认的模型／数值决策不因实现进度或文档迁移而重开。

Context 存储采用现有[统一容量、记录视图与执行借用合同](../../maintainer/paged-kv-cache.md)。
GLM 的 State variants 和 owner placement 由[并行设计](parallel.md#12-state-variants-与字段布局)
维护，KDA／Kpool 定义自身字段访问；Main/MTP history 与 State 共享各 rank 的物理容量。

## 1. 文档与状态

以下区分设计与实现。既有基础 Op 成果按代码和原验证证据确认；新格式需求以
当前清单为准，不能用已通过的 BF16 shared 结果代替 Q8 路线的验证。

| 内容 | 决策状态 | 实施状态／入口 |
|---|---|---|
| 架构、源表示、量化与数值政策 | 已确定 | 长期参考；新量化产物尚未生成或做模型精度评测 |
| [双 Spark 分工与资源](parallel.md) | 模块分工、State variants、owner histories 与统一容量需求已确定 | 通用目录已有对应能力，待 GLM 绑定、通信与执行接入；容量为推导 |
| [Linear/MoE 清单](op-checklist.md) | Q8 shared 完整链、MLA query/O、Q4 draft head、stem/Vision 纯矩阵与 MoE 优化迁入 | `.50` 工作区按清单逐项实施 |
| [KDA 计算与 ReplaySSM](replayssm-kda.md) | 字段／I64 offsets、source/destination、五个 frame record planes、Host/Device fold 与搬迁后 Graph 使用已明确 | 原位 batch update 已使用字段视图；新增输入链、独立目的及 record/fold 待实现／验收 |
| [Kpool Sparse MLA](kpool-sparse-mla.md) | 读写 views、多层页组、冻结 selection、Context/frame tail 和五个计算入口已明确 | 设计待实现／验收 |
| [mHC](mhc.md) | 五个 C++ 入口、M 列布局、容量查询、两阶段融合与资源归属已明确 | 设计待实现／验收 |
| [MTP](mtp.md) | 输入移位、stem/residual、共享 Main hidden、provisional 页与 frame、首步采用及双机交接已明确 | 模型组合与新增 norm Ops 待实现／验收 |
| [Vision](vision.md) | GLM processor、patch 顺序、bias/2D norm-RoPE/D64 attention、segment 分配与媒体生命周期已明确 | 输入基础、Op 扩展与模型组合待实现／验收 |
| [Frontend 清单](frontend-checklist.md) | 资源、tokenizer、模板、缓存边界、thinking/tools/constraints 与媒体准备 | 分阶段实施，进度见清单 |
| Converter 与模型接入 | 数值规则已定，接入未完成 | block-FP8 编码入口、GLM converter/binding 与运行路径待实现 |

量化格式存在不等于转换方法存在：当前 block-FP8 converter 只有 `import_encoded`，
已定 `fp8_block_maxabs` 需要实现。KDA 通用入口用于算法／性能验证，不替代生产
ReplaySSM 的调用约束。各设计的“待实现”范围与现有通用算子能力分开解释。

## 2. 当前实施范围与待设计事项

当前清单覆盖 TP2/EP2 的纯投影、无状态融合、dense/shared MLP 和 Main/MTP
本地 MoE，并补充 MLA query/O、optimized draft head、stem/Vision 的纯矩阵需求。
KDA 状态链、MLA/Indexer、mHC 按路线中的独立任务包实施与调优。

仍需设计或落实的工作：

- 跨模块 collective、pack/unpack、通算融合、模型 Graph 和跨 owner 前缀恢复。
- 按 MTP/Vision 设计落实执行、媒体输入与资源交接；连接分片 logits、采样、acceptance 和 scoring。
- KDA 输入准备、record/fold 的实现与 Spark 调优；输出复用 sigmoid gated RMSNorm，长 prefill 由完整窗口入口内部选择算法。
- Converter、artifact binding、模型加载／Frontend／Engine 接入和完整状态组合。

设计决定、实现完成和性能验证是三种不同状态。算子 kernel 完成不自动表示模型
接入完成，本地算子性能也不自动表示双机模型吞吐收益。

## 3. 文档职责

| 内容 | 当前唯一归属 |
|---|---|
| 模型数学、逻辑角色与历史依赖 | 长期模型参考 |
| 表示、codec、activation permission、可观察舍入 | 长期数值参考；已有通用格式直接引用 |
| 通用 Context 容量、记录寻址、借用与搬迁 | [Context 存储合同](../../maintainer/paged-kv-cache.md) |
| 分片、owner、State variants、通信交接与部署预算 | parallel.md |
| 具体 Op/views、physical layout、写域、alias、融合与验收 | 对应 family 设计 |
| 执行条目与完成证据 | op-checklist.md |
| Frontend 接入条目与完成条件 | [Frontend 清单](frontend-checklist.md) |
| 集成步骤、并行任务与阶段进度 | [实施路线](roadmap.md) |
| 文档导航与既有设计范围 | 本索引 |

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
