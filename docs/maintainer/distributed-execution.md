# 分布式执行

一个执行实例集中产生调度计划，由一组 rank 执行。单 rank 在进程内直接调用，多 rank 使用
独立 worker；模型数学、Context 所有权和提交语义遵循[Engine 架构](engine-architecture.md)。

## 进程与计划

协调进程持有 Frontend、Scheduler、ResourceManager、模型 Session 和 ContextDirectory。
每个 worker 持有本地 CUDA 环境、权重、RankProgram 与通信资源。Worker 按计划执行，
不重新选择请求、缓存来源或逻辑页。

本地进程通过 posix_spawn、远端通过非交互 SSH 启动。Bootstrap 在 CUDA 初始化前交付 rank、
节点、设备和环境。SSH 承担启动与进程日志；持久控制连接承担命令和结果；GPU 数据通过 NCCL。

模型 ParallelPlan 从全局 config 与绑定派生权重切片、局部几何、状态 placement、通信角色和
拓扑。全局 config 不改成局部 head 数。参数切片、输出组合和 State/history 必须来自同一计划。

TP、EP、DP、ADP 描述不同分工，角色可以共享物理 rank/group：

- TP 分割模型计算；EP 分割完整 experts，expert TP 则分割每个 expert 的 intermediate。
- DP 表示各自独立推进的完整模型副本。
- ADP 表示同一实例内按请求分配 attention owner，其他层仍可使用 TP/EP。

DP 与 ADP 不共用配置含义。当前集中执行的模型计划使用一个完整副本；配置能表达某个并行度，
不代表具体模型已支持它。支持组合由模型准备入口决定。ADP 的准入和迁移见[Context 管理](context-management.md#attention-dp)。

## 启动与失败

准备过程依次建立执行组、模型计划、权重和必要通信资源，协调 Graph 校准、容量选择和正式准备。
每个 rank 先报告本地准备结果，协调侧收齐后才允许进入涉及 collective 的阶段。
不能让一侧在准备失败后退出，另一侧仍自行进入下一次通信。

资源按 rank 报告；共享物理内存还计入协调节点自身的控制与 Frontend 开销。
Context 容量可不对称，模型声明每个 placement 所需的完整工作集，公共容量求解使用这些事实。
预算和 Graph 重建计费见[启动资源](engine-architecture.md#启动与资源准备)。

启动事件带 rank/node 与阶段身份，进度转发有界且允许合并。终态保留最后进度，展示失败不影响
引擎启动控制。Worker 报告的原始加载原因优先于随后观察到的连接关闭或退出码；协调侧补充来源，
不以通用进程错误覆盖具体原因。

正常结束等待既有工作和消费者完成，再销毁图与存储。通信或执行组失败时停止新提交、终止通信，
清理其余进程；不等待一个已无法成立的正常 collective 完成条件。

实现入口：[ExecutionGroup](../../src/distributed/bootstrap.h)、
[准备编排](../../src/runtime/executor/preparation.h)、
[rank 准备](../../src/runtime/executor/rank_preparation.h)。

## 命令与完成

控制消息按字段编码拥有型数据，不传进程指针或 C++ 对象内存。执行组使用同一套内部合同，
没有跨版本兼容层。消息顺序、ticket 和 frame generation 防止迟到结果作用于新请求。

Executor 在接收工作前保障后续 supply、commit、discard 与完成事件的额度。普通工作占满队列，
不能阻止已有 ticket 收尾。大媒体输入通过独立有界输入传输；Context 动作按批传递，
整个事务持续持有源与目的。控制描述不随整个物理池容量展开。

Compute 的结果由指定结果方交付，退役覆盖全部必要参与者；Context 发布同样等待完整参与集合。
各 rank 可以有不同的本地动作与地址，但最终采用同一逻辑前缀。Host decision 输入属于原 ticket，
其消费和传输寿命独立于 Host 函数返回。

[Executor 合同](../../src/runtime/executor/contract.h)维护事件含义；
[模型 Context 动作](../../src/models/common/context/contract.h)维护局部与 peer 范围的交接。

## 通信与图

模型定义哪里需要完整结果、局部贡献或按行汇集；collective Op 和通信后端实现相应数学与 stream
效果。Op 不持有请求调度，通信后端不解释层序、缓存价值或 speculative 前沿。

同一执行要求各参与方匹配通信次序、count、dtype 与角色。某侧没有本地 expert 或 attention 行，
仍履行已安排的通信。多个 communicator 的相对提交顺序也属于模型执行合同。
NCCL 的入队进度与设备完成分别处理，后续设备工作按实际入队和 stream 依赖推进。

计算和通信可以一起捕获。上下文长度只改变局部 Attention kernel 时，在固定外层图内更新 child，
保持通信节点和缓冲稳定。只有实际需要不同计算几何或通信形状时才创建图变体，
不按 owner 向量、采样方式和每行接受长度枚举图。

权重分片后的普通 Linear/MoE 贡献遵循既有输出数值合同，分布式层不统一强制 FP32。
Router、selection 和 token 决策需要共同离散结果时，由模型保证生产方式与消费一致。

模型放置见 [Qwen4Exp](qwen4_exp-model.md#并行与放置)和
[GLM5Next](glm5next-model.md#并行与放置)；通信代码见
[distributed](../../src/distributed/)和[collective Op](../../include/ninfer/ops/collective.h)。
