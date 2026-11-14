# 并发、物理批次与执行存储

本文记录 Engine、模型与 Op 之间的并发合同。具体字段、几何支持范围和 kernel 分派以所属代码为准。

## 容量与所有权

| 符号 | 含义 | 所属模块 |
|---|---|---|
| C | Engine 允许驻留的真实请求容量 | 启动配置与资源规划 |
| B | 本轮真实计算成员数 | Engine 选批，Session 冻结成员 |
| P | 本轮全局物理行数 | 模型的 BatchExecutionPlan |
| L | 各 ADP rank 共同使用的局部 Attention 行数 | 同一模型计划 |
| W | 当前数学阶段的 token 列数 | 模型/backend |

并发容量由配置给出；资源规划验证实际工作集；Op 负责自己的数学和表示范围。
调用方不复制这些限制。C 是驻留容量，不是每轮计算量，也不承诺所有请求同时达到最大上下文。

Runtime 处理逻辑成员、票据与生命周期，不决定 Graph 桶或 ADP 行布局。Session 的模型 Work 持有
选定的物理形状与映射；RankProgram 消费这些值，不能在某个 rank 上重新选形状。
ContextWork 只描述存储动作，不携带计算批次。其绑定合同为一个请求，可同时释放多个请求。

## Graph 几何

共享 [BatchExecutionPlan](../../src/models/common/batch.h)负责桶、取整和可行几何，
各模型保留自己的数学组合与 Graph 组织。小批次采用精确桶，大容量使用稀疏桶并覆盖最终 C；
具体分桶策略由[实现](../../src/models/common/batch.cpp)维护。
关闭 Graph 时 Decode 使用精确几何；Prefill、Replay、Score 和 Control 保持各自的精确几何。

Qwen Decode 使用 P；GLM 的 ADP Decode 使用 `(P,L)`。P 对 B 向上取整，L 对实际最大 owner
成员数向上取整。Graph catalog 只保留可由合法真实分布选中的 pair，不枚举 owner 向量。
捕获与预热使用能选中该桶的真实代表分布，再补 padding，不能把 P 个物理位置都当作真实请求。

Graph 变体表达物理几何与必要计算主体。采样参数、请求身份、draft extent 和接受长度作为数据输入，
不增加变体轴。上下文变化沿用 Attention Op 的 profile/child-node 更新，外层通信次序保持不变。
需要 Host grammar、CPU expert 或接受结果的边界保留显式交接。

## 逻辑成员与物理行

Work 在 submit 前拥有并冻结真实成员、generation、形状和行映射，持续到最后消费者退役。
Engine 的 Flight 与 Session 的事务不能借用下一轮会覆盖的 Scheduler scratch。

Qwen 逻辑成员顺序对应物理 `[0,B)`。GLM 先按 owner 稳定分组，仍将全局真实行压紧在 `[0,B)`，
每个 owner 的真实局部行压紧在 `[0,owner_count)`。全局尾部 `[B,P)` 和局部尾部
`[owner_count,L)` 是 padding。全局矩阵保持 P 行，不扩成 rank 数乘 L。

GLM 的 local-to-global 和 gather 映射来自同一个冻结分组。空 owner rank 的局部 Attention 无有效
请求，但仍按本轮 L 参与相同大小、相同顺序的 collective；全局 TP KDA/FFN 仍处理真实全局成员。
映射构建按成员计数、前缀和及稳定填充完成，不逐成员扫描所有其他成员。

Decision、Result、Commit 和对外统计使用逻辑 `[0,B)`。Program 先还原逻辑顺序，再交给 Executor。
部分取消、零接受或某行提前结束不改变当前 Work 的成员与形状，只影响采用决定；下一轮才重新选批。

## 有效域与 padding

live、阶段有效列数、draft extent、提交长度表达不同事实。已有元数据足以定义无访问行时复用它；
零接受仍需复制状态的算子、sampling/acceptance 等入口单独接收 live 标志。
Host 行标志使用连续字节，Device 行标志使用 I32；普通单值 bool 不受此约定影响。

| 情况 | 行为 |
|---|---|
| padding | 不访问该行 State/KV/records，不产生采样、expert assignment 或状态提交 |
| 真实请求、阶段有效长度为零 | 按所属 Op 的数学合同处理；需要保留状态时 selectors 必须有效 |
| 真实请求、接受长度为零 | 保持原有零接受语义，包括所需 tail/history 复制 |
| 真实请求、有采用前缀 | 模型将逻辑提交换算为各类状态的数学前沿 |

所有真实请求的 draft extent 都为零时使用 ordinary Work；仅部分为零时仍是同一 speculative Work。
DFlash proposal 窗口与 target verify 的有效长度各有语义，不能互相替代。

每次提交初始化所选物理范围的 live、长度、位置、映射、selectors、采样参数与提交计数。
padding selectors 为 -1，长度和计数为零；有效域判断先于选中数据的访问。
无状态 dense contraction 可以读取已初始化的有限 padding 输入。无效 records 不必整体清零，
消费者不得读取其未定义区域。MoE 沿 fused、TP/EP 与 offload 路径跳过无效 token 的 assignment，
不能因 padding 多加载 expert 或增加 CPU expert 工作。

## Frame 与生命周期

每个模型拥有帧布局及对应视图。启动时分配实际峰值 backing，各 Graph 桶共享它；
桶只持有轻量布局和视图，不复制 logits、records
或 workspace。资源的索引域保持明确：

- 驻留 token counts、conditioning 等按 C，使用真实 residency 身份访问。
- 本轮 logits、hidden、proposal 与控制数据按 P/L/W 形成视图。
- ReplaySSM records 的层 stride 来自分配容量；当前 P/L 只截取该层，不能改写层间 stride。
- 持久 State/page selectors 属于 Context owner，不把物理行号当成持久 slot。

控制区采用当前 P/L/W 的紧凑布局，Host/Device 视图和复制范围由 frame owner 同时生成。
小 B 只上传本轮控制数据。Context 绑定 staging 按绑定事务分配，Device 驻留计数表覆盖 C。
不同 Graph 桶可以使用同一 backing 的不同控制偏移，遵守 frame 复用边界。

PLE producer window 拥有容量化 metadata；真实 identity 只有 B 项，embedding 视图覆盖 P/W。
producer 不为 padding 读取 SSD 或生成 identity，pipeline 保持窗口到最后消费者结束。

ResultReady、CommitReceipt、DeviceRetired 各自推进不同事实。取消不提前释放 DMA、Graph、
Host producer 或来源借用，lane 复用不能改变旧 ticket 的统计归属。失败清理停止提交，结束消费者，
再释放 owning backing；不通过逐 step 全设备同步替代生命周期管理。

## 规划、调度与观测

模型规划组合 roots、records、workspace、通信与 Context 的真实重叠峰值；Op 容量接口覆盖自身
区间内的非单调分派。GLM 将局部 Attention 容量与全局 roots 分开求高水位，避免逐层重复枚举
所有 global/local 组合。Vision 沿用共享有界窗口，不随 C 为每个请求复制 encoder workspace。

Session 持有成员身份与在途状态，Engine 持有请求和 placement 负载。准入扫描的单次工作量
与 C 解耦，保持公平推进；批量结束先结算成员与快照，再通知各请求，各自保留独立计时截止点。
缓存来源与回收沿用 [Context 策略](context-management.md)，复杂度按实际候选与动作评估。

模型声明完整 Work/Decision/Commit payload 上界，Executor 根据 frame 数与 rank 数准备 Transport
的独立发送、continuation 和接收额度。额度是有界传输合同，不等于立即分配其上限大小的 buffer；
bootstrap 默认额度不构成模型并发上限。

对外 decode-by-batch、接受率和请求统计使用真实 B。Executor 按 C 准备计数域，每请求只记录实际
参与过的 B，发布时恢复公共索引形式。P/L/W 用于解释计算量与 padding，不改变产品吞吐和 step
口径。测量以完整提交和退役为边界，不能在结果返回后立即复用仍在使用的存储。
