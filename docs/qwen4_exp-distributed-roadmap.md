# Qwen4Exp 双 Spark 加载与推理实施路线

目标：在两个 DGX Spark 上，通过公共 Engine、CLI 与 serving 加载现有
Qwen3.8-Flash-Next 产物，完成 TP2/EP2 推理，接通 Text、MTP、Vision、约束、
上下文复用和 CUDA Graph，同时保留现有单机执行与 Op 分派。

本文维护实施顺序、依赖、共享能力和阶段产出。执行与生命周期合同由
[分布式执行设计](distributed-execution-design.md)维护；具体分片、数学边界和
Graph 组合由 [Qwen4Exp 并行策略](qwen4_exp-parallel-design.md)维护。已实现的
单 rank 所有权见[引擎架构](maintainer/engine-architecture.md)。实施后将稳定合同
更新到相应权威文档，本文件记录阶段完成状态。

## 1. 起点与总路径

2026-10-18 的起点：D0 首版策略已确定，D1 单 rank 执行所有权重构已完成，
提交为 `71cdd615`。Engine 集中调度，Session 管逻辑状态，RankProgram 管物理
资源和设备执行，LocalExecutor 在 Engine worker 线程直接驱动本地 rank。

本地 Op 成果分布在已有工作分支，迁入是后续集成的前置任务：

| 来源 | 已有成果 | 后续动作 |
|---|---|---|
| `.50` 的 `feat/spark-tp2-ep2-ops` | TP2 主要 Linear、split、GDN、Sparse GQA 和 EP 本地 expert 计算；对应清单已标记完成 | 按 Qwen 相关提交范围核对主线，迁入尚未合入的成果 |
| `feat/spark-parallel-ops` | 本地 QSA 投影与 prepare、shared=320、GPU 分片 PLE lookup、分片词表决策、scoring 统计 | 迁入当前重构后的主线，对齐调用接口 |
| `dev` | Definition/Shard/Parameters、Session/RankProgram、LocalExecutor、逻辑目录与物理存储 | 在现有所有权上增加多 rank 实现 |

已有局部数学与性能证据随成果迁入复用；集成验证覆盖新增调用、布局和通信组合。

| 阶段 | 核心工作 | 完成后的实际能力 |
|---|---|---|
| D2：分布式执行基础 | worker 启动、控制协议、DistributedExecutor、通信组、NCCL、失败与退役 | 两机接受共同计划，完成真实 GPU 通信和有序执行 |
| D3：并行物化与资源 | ParallelPlan、编码分片加载、本地参数、State/KV、frame、共同容量 | 两机按最终布局加载模型，完成资源准备与销毁 |
| D4：完整 eager text | TP/EP 层组合、PLE、分片输出、提交、缓存与恢复 | 公共 Engine/serve 支持双机普通生成、并发和上下文复用 |
| D5：完整模型功能 | MTP、Vision、约束、scoring 与状态交接 | 双机覆盖目标模型的完整使用流程 |
| D6：Graph 与性能 | 协调捕获、完整设备段、动态子图、通信重叠与融合 | 完成可长期使用、继续调优的双机实现 |

执行顺序为 D2 → D3 → D4 → D5 → D6。Op 迁入可与 D2 并行；共同计划和
owner/载荷合同确定后，分片物化可与通信执行并行开发。Graph 的地址寿命、通信
次序和失败规则在 D2/D3 确定，D6 完成整模型捕获与优化。

## 2. 从开始就建立的共享能力

| 能力 | 共同机制 | 模型提供的内容 |
|---|---|---|
| Deployment / Group | node、rank、device、有序参与者、通信组 | 哪个计算角色使用哪个组 |
| ParallelPlan | 复制、分片、owner 集合、参数选择、布局身份与执行边界 | heads、expert、PLE、HC 等具体分工 |
| Executor | 执行额度、命令顺序、结果交接、提交、退役、失败 | 原生 Work 和计算阶段 |
| 编码权重物化 | 源区间读取、编码 planes 搬运、局部 backing、去重 | 每个逻辑参数如何切片和拼接 |
| ContextDirectory | 身份、引用、coverage、owner 完成条件、容量预留 | 哪些状态构成一个可恢复前沿 |
| 通信与执行边界 | 通信组、缓冲寿命、stream 依赖、完整结果或局部和 | 归约位置、residual 消费和适用融合 |

### 2.1 全局定义与本地几何

Config 保留全局模型含义。无地址、不可变的 ParallelPlan 同时驱动参数分片、
本地几何、状态 schema、通信边界和容量规划；RankExecutionPlan 再结合实际
表示与设备资源，生成本地执行资源计划。

权重、workspace、StateImage、KV 和原生执行消费同一份派生几何。具体格式和
shape 能力由 Parameters 与 Op 准备入口判断，错误带 rank 和准备阶段返回。

### 2.2 本地 Tensor 与分布式边界

Tensor 继续表达本地地址、位置和布局。执行边界描述 token/feature 的复制或
分片、结果是完整值还是某个 group 的局部和，以及 residual/norm 的消费次序。
准备期绑定 collective、布局转换或完整融合 Op；局部算子保留自己的 shape policy。

例如 `[2560,M]` mixer output 是局部贡献，归约后才进入 HC combine/norm。
以后将归约与 HC 融合时，沿用同一个数学边界和寿命合同。

### 2.3 owner、状态与未来模型

Group 和 owner 集合使用显式有序成员，分开 node ID、global rank、group 内 rank
与本地 CUDA ordinal。模型角色与物理 communicator 的关系由计划解析。

状态可以分片、复制，或仅存在于部分 owners。目录保存逻辑 image/pool 和共同
前沿，各 rank 描述本地 schema；分字段复制在下发前解析为本地字节范围。
可恢复来源的完成条件覆盖全部必需 owners。

四机时两个 QSA KV heads 会涉及复制；其他模型也可能让 attention 状态只归
某个 owner。这些情况复用相同目录和执行合同，具体布局由新的模型计划提供。
Checkpoint 的布局身份包含固定 ParallelPlan；本轮复用发生在同一计划内。

单 rank 使用对应的退化计划、size-one 边界和 LocalExecutor。Spark 采用 GPU
模型计算与 Device State/KV；RTX 5090 的既有 CPU expert、Host cache 路径继续
由本地执行实现提供。

## 3. D2：分布式执行基础

### 3.1 启动、控制传输与构建

协调节点持有 Engine、Frontend 和服务入口，每个 GPU rank 使用独立 worker
进程；协调节点上的 GPU 也由本地 worker 持有。部署文件描述节点地址、artifact
本地路径、rank/device 映射和模型并行配置。

启动按共同阶段推进：

```text
Join → Resolve → Connect → Materialize → Prepare → Ready
```

本机控制连接使用 Unix socket，远端使用持久 TCP。GPU 张量走通信层；控制消息
传递执行描述、资源增量和必要的 Host 输入/结果。网络收发与 CUDA 执行分离，
worker 执行线程应用命令并观察设备进度。

新增 `NINFER_ENABLE_DISTRIBUTED` 构建选项，启用时编译通信 backend、collective
实现和 worker 应用，并发现 NCCL。`NINFER_TARGET` 继续选择硬件目标，单机与
多机是部署方式；现有单机及 Ops-only 构建保持独立。

### 3.2 类型化协议与 DistributedExecutor

沿用已有 `submit/supply/commit/discard/poll` 合同，补齐跨进程实现：

- 消息携带 epoch、命令序号、ticket、阶段、计划/variant 身份和参与者。
- 原生 Work 逐字段编码。Vision 等进程内 owner 转为明确数据或已绑定资源身份。
- 首次绑定发送所需输入与资源，后续发送执行窗口、页表变化、有效长度和提交增量。
- output rank 返回候选输出；Context 物化与资源退役按必需 owners 汇总。
- 执行预留同时保留后续 supply、commit/discard、结果和完成事件的推进容量。
- CommitReceipt 表示执行组已接受有序提交；正式逻辑采用与设备完成分开。
- DeviceRetired 覆盖对应 rank 的 compute/copy/communication 及其他消费者。

首版每个 sequence 一轮生成事务在途。后续跨 ticket 提前执行使用无地址的
DeviceResultRef 表达 producer、结果字段、版本和寿命；当前同 ticket 的设备
阶段交接继续由 RankProgram 持有。

### 3.3 通信与失败

CommunicationContext 拥有 communicator、内部通信资源和注册 lease；调用者
持有输入/输出缓冲。实现 AllReduce、Broadcast、AllGather，以及设计要求的
ReduceScatter/P2P 基础。GPU 通信缓冲使用与正式模型相同的存储和完成合同。

区分 Host 调用入队完成与 GPU 完成。NCCL nonblocking 的进度、异步错误和终止
由通信 owner 处理，入队完成后才能提交依赖操作或记录完成凭证。参考
[NCCL communicator 合同](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)。

每个 variant 明确组内及跨 communicator 次序，各 rank 采用共同的 count、dtype、
root/peer 和 capture 选择。D2 完成真实 GPU buffer 通信与最小 capture/replay，
确认实际传输路径和代表性消息延迟。整模型 Graph 在 D6 接通；其参与者规则遵循
[NCCL Graph 合同](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html)。

固定成员执行组遇到通信断开或设备错误后停止新提交，终结未决请求并统一收尾。
普通请求取消沿既定通信序列完成必要动作，再丢弃结果和退役资源。准备阶段先
确认所有参与者具备本地资源，再共同进入含通信的操作。

**阶段产出：两个 worker 能运行共同命令序列，完成设备通信、结果汇总、取消和失败收尾。**

## 4. D3：并行物化与资源

### 4.1 计划、本地参数与权重读取

先将已确定的 D0 角色表转为 ParallelPlan，再生成各 rank 的参数和状态几何：

| 角色 | 每 rank 的基础放置 |
|---|---|
| GDN | Q/K 各 8 heads、V 24 heads，以及对应 controls、卷积和 recurrent state |
| QSA | Q 12 heads、KV 1 head |
| Indexer | 4 个 query heads、index cache、raw tail 复制 |
| Routed experts | 每层 256 个完整 experts；global ID 与物理 bank slot 分开 |
| Shared | intermediate=320，形成局部贡献 |
| HC / residual / embedding | 完整复制 |
| LM head | 词表的一半，Main/MTP 共享本地物理 head |
| PLE | 64 个完整 NF4 shards，后续计算复制 |
| MTP | QSA TP2、experts EP2，两个 stem FC 复制 |

编码物化需要支持：输出行选择、按量化分组切 K、多个逻辑段组成本地 parent、
按 expert 集合收集完整 planes，以及完整 PLE shard 读取。

GDN QKVZ、AB 和 QSA joint parent 按角色分别切片；卷积保持 tap 顺序并重建
本地 Q/K/V 通道排列。Q8 K 切片保持 G32 编码分组；expert 的 codes/scales/divisors
一起搬运。加载后直接使用本地编码和稳定视图。

各节点读取自己的 artifact，只物化本地需要的内容。源编码保持不变，Main/MTP
等共享关系继续去重；加载使用有界 staging，统一内存中不保留额外完整可写权重副本。
通用 materializer 负责编码区间与 backing，模型 parallel 模块负责参数语义和切片配方。

### 4.2 State/KV、frame 与共同容量

ContextDirectory 管理共同逻辑页组，各 rank 根据自己的 schema 建立物理 backing。
一个页组代表同一 token 区间的全部必需分片，相同 ID 对应的本地地址和字节跨度可以不同。

资源计划覆盖：

- GDN source/destination images、checkpoint、verify records 和 fold scratch；
- Main/MTP 主 KV 分片及复制的 Indexer cache/tail；
- PLE history、sample hidden、conditioning 和设备 raw-ID tail 镜像；
- 执行 frame、logits、候选、决策输入、mapping staging 和通信缓冲；
- 主计算与 PLE 旁路同时存活的独立区域、Graph 和通信内部保留量。

逻辑 KV/State 容量由必需 owners 中的最紧约束决定。每个 rank 报告本地容量与
资源事实，协调侧选择共同容量；显式用户容量保持原要求。

先建立通信环境，再获取规划快照。Spark 按节点共享物理内存核算，协调节点额外
承担的 Frontend/media 资源计入本节点预算。准备重试使用共同 attempt/phase ID，
保证资源销毁、通信进入和正式重建的顺序一致。

**阶段产出：两机按最终分片加载模型，建立一致参数和状态布局，报告真实资源需求，完成启动和销毁。**

## 5. D4：完整 eager text

### 5.1 原生层组合与 EP

从单请求完整链开始，在同一实现内扩展到长 prefill、多请求和缓存恢复。Main 层
边界如下；eager 与后续 Graph 复用同一 enqueue 流程：

```text
复制 residual → HC → 本地 GDN/QSA → O projection 局部贡献 → AllReduce
  → HC combine/norm/mix → 本地 routed + shared TP 局部贡献 → AllReduce
  → HC combine
```

BF16 局部贡献沿既定合同参与归约，GDN 的 FP32 controls/state 保持原表示。
原生执行调用派生的本地几何；Op 根据实际格式、shape 和 token 数分派。

EP 保留全局 top-k 和归一化系数，在读取权重和建立 grouped jobs 前筛选本地
assignments。没有本地 routed 工作的 rank 仍输出正确的 shared/零贡献并参与归约。
Shared 与 routed 的本地组合保持既定舍入和加入次序。

复制的 Router/Indexer 绑定共同离散决策路径、输入布局和 global-ID tie-breaking。
不同上下文、短上下文全选及有效列数继续由对应 Op 处理，通信形状和阶段顺序共同选择。

### 5.2 GPU PLE producer

```text
设备 IDs + 已提交 raw-ID tail
  → 本地 NF4 shard lookup → 精确合并 lookup words
  → 稳定 E[2560,T,B] → PLE projection/gate/convolution
```

已有 lookup 保留 BF16 words，包括 signed zero。合并采用独占 owner 的整数 word
语义；Main 消费稳定 E、就绪依赖和消费寿命。

IDs 确定后可提前 lookup，与 embedding、第 0 层重叠；projection 也可在依赖
允许时提前。PLE 使用独立存活区域，通信序列与后续 block 归约保持 D0 约定。
GPU producer 与 Main 的计算接口对齐，存储位置不决定 Main 的切图粒度。

### 5.3 分片输出与正式采用

Greedy 合并有全序 tie-breaking 的候选 key；随机采样汇集局部候选，由 output
rank 完成全局过滤和采样后分发。排除物理词表 padding，允许局部分片为空，
dead-end 按全局候选判断。持久 counts 只更新 Frontend 最终采用的输出。

当前分片候选 Op 的 counts/mask 按 global token ID 编址。首版保留复制 counts，
约每请求 1 MiB；mask 的传输与本地落位遵循同一合同。词表切片不自动改变这些
输入的索引方式。

### 5.4 缓存、调度与请求生命周期

普通生成贯通后，完成多请求与不同上下文、精确/部分前缀命中、尾页 COW、暂停、
Device checkpoint 恢复、Replay、取消、输出截断、forced control 和资源回收。

Prefix 匹配集中执行一次。选中的 manifest 关联同一逻辑状态的必需 owner 分片；
目录、Session 和 Executor 分别负责引用/容量、模型前沿、物理完成汇总。副本
发布等待必要分片完成，槽位复用继承最后消费者依赖。

策略消费实际 T/B、有效长度与上下文，保持现有资源调度算法；多 rank 完成语义
由目录和 Executor 承接。checkpoint 的复制字段与分片字段共同保存、恢复和采用。

**阶段产出：公共 Engine/serve 支持双机普通生成，并具备完整并发、缓存和恢复生命周期。**

## 6. D5：完整模型功能

### 6.1 MTP 与约束交接

```text
共同 conditioning → 分布式 draft chain → Main verify
  → 分片候选与 acceptance → Frontend 确定采用前缀 → 各 rank Finish
```

每步 draft 的 ID 和 conditioning 在设备侧衔接，整批实际深度共同选择，各行
有效长度使用设备 metadata。Main verify 的整窗口 IDs 确定后启动一次 GPU PLE lookup。

Finish 消费相同的最终接受长度，执行本地 GDN fold、QSA/PLE tail 提取、末端
hidden/conditioning、token counts 和 MTP history 对齐。Session 管理 Main `F`
与 MTP `max(F-1,0)` 的正式前沿，临时 draft tail 保持在候选状态中。

约束 matcher 位于协调侧。drafts 就绪后回传必要事实并生成 masks，Main verify
与本地 head 继续推进，Decide 消费其结果和 masks。RNG、全局 token ID、逻辑
位置、终态截断和采用语义保持现有合同。

### 6.2 Vision、协议与评分

Vision 按完整 item 分工：patches 发给 encoder owner，完整编码后的 merger
features 分发给 Text ranks。保留 item 顺序、MRoPE、scatter 列序和 Main/MTP
最后消费者寿命；workspace 计入实际并行编码峰值。

工具调用、thinking、OpenAI/Anthropic 协议输出继续由协调侧 Frontend 解释，
输入资源和最终采用的 token 流接入既有路径。

CausalScoring 使用已有局部统计 Op：每 rank 计算 `(max, exp-sum, target-logit)`，
汇集后合成全词表 logprob。评分经公共 Engine，沿用独立 purpose 与 fresh-root
状态隔离，通信只传必要统计量。

**阶段产出：双机接通 MTP、Vision、约束、评分及其取消、缓存和提交语义。**

## 7. D6：Graph 与性能

### 7.1 完整设备段

| 路径 | 目标图组织 |
|---|---|
| ordinary，决策输入已齐 | Main、PLE、head、分布式决策组成完整图 |
| masks 尚未齐 | Compute 与 Decide 两段，保留 logits 和输入寿命 |
| MTP，无 draft-dependent Host 输入 | draft chain、verify、head、acceptance 合图 |
| MTP，需要 Host masks | Compute 先推进，Decide 消费随后到达的 masks |
| 最终采用 | 接受长度作为设备输入，执行 Finish 设备段 |
| prefill / Replay | 按真实 T 执行；完整固定 chunk 可增加 capture profile |

保留 QSA 动态 envelope，通过完整外层图中的纯本地 child 更新实现。通信节点、
共同次序和缓冲绑定稳定；改变通信合同的情况选择另一条已准备 variant。

各 rank 先完成本地资源准备，再共同进入 warmup/capture/replay。Graph、注册
lease、backing 按消费者寿命销毁。旁路 scratch 和跨阶段结果使用已规划区域，
完整 Graph 的内存占用纳入共同容量协商。

### 7.2 从完整链条判断瓶颈

当前 48 层基础计划中，Main 有 96 次 block 归约；加上 PLE 和 greedy head，
ordinary 为 98 次设备通信，K-step greedy MTP 为 `98+3K` 次。每次 block 的
BF16 逻辑 payload 为 `5120*M` 字节：

| M | 单次 payload | 96 次合计 |
|---:|---:|---:|
| 1 | 5 KiB | 480 KiB |
| 8 | 40 KiB | 3.75 MiB |
| 4096 | 20 MiB | 1.875 GiB |

这些是模型逻辑载荷，实际 wire 流量与耗时由通信实现和测量确定。小 M 观察逐次
通信延迟、rank 偏斜和 Host 提交间隙；大 M 同时观察链路吞吐、局部计算和重叠。

优化顺序：消除多余复制/同步/Host 往返，改善 PLE 重叠和 EP 负载分配，再推进
归约边界融合。HC token 分区、Indexer query 分工等后续候选根据完整阶段成本选择。
单 rank 的局部调优仍由对应 Op 维护。

trace 使用 rank、ticket、命令和阶段关联；组延迟按实际完成边界度量，各 rank
时长分别报告。最终通过公共 serve 测 ordinary/MTP、并发和长 prefill 的实际
输出速度，区分接受长度、每步计算时间、暴露通信和资源成本。

**阶段产出：完成全模型分布式 Graph，明确主要性能开销，并收敛关键路径上的实现。**

## 8. 文件组织与推进清单

```text
apps/worker/                          worker 启动和收尾
src/distributed/                      bootstrap、控制传输、groups、NCCL
src/runtime/executor/                 DistributedExecutor、额度、事件和退役
src/models/common/parallel/           共享分片与执行边界描述
src/models/common/context/            owner-aware 目录、引用和完成条件
src/models/common/moe/                本地 expert 与通信边界的组合
src/artifact/                         编码区间物化
src/models/qwen4_exp/parallel/         模型分片配方、本地几何和执行计划
src/models/qwen4_exp/work.* / wire.*   原生 Work 值合同与逐字段编码
src/models/qwen4_exp/session/          逻辑事务与缓存语义
src/models/qwen4_exp/program/          本地资源、设备阶段与 Graph
src/models/qwen4_exp/execution/        原生层计算和归约位置
src/ops/                             局部数学、collective 与完整融合实现
```

共同结构按实际职责集中组织；每个模型提供自己的角色映射、schema 和边界。
其他模型复用部署、物化、通信、目录和执行生命周期，保留各自的数学与调优。

- [x] D0：收敛首版 TP2/EP2 模型策略。
- [x] D1：完成单 rank 原生执行所有权重构。
- [ ] 前置：核对并迁入两处已完成的 Qwen Op 成果。
- [x] D2 执行基础：执行组、控制协议、通信、事件汇总与失败收尾通过双 Spark 验证。
- [ ] D2/D3 接线：产品 worker、原生 codec adapter、Engine 启动与逐 rank 资源呈现；与实际 rank 物化一同完成。
- [ ] D3：最终权重分片、本地参数、State/KV/frame 与共同容量可加载。
- [ ] D4：完整 eager text、并发、前缀缓存、暂停恢复和 Replay 可运行。
- [ ] D5：MTP、Vision、约束、评分与完整提交语义接通。
- [ ] D6：分布式 Graph、动态 profile 和关键性能路径收敛。

每阶段直接检查其新增数学、状态或执行路径，形成对应可运行产出。D2 当前验证入口为
`ninfer_distributed_execution_test` 和 `ninfer_collective_bench`，两者使用正式的进程、
传输、通信和 Executor 实现。尚未提供脱离实际模型准备的产品 worker；产品入口随 D3
绑定 ModelShard、RankProgram 和共同容量，D4 再接通完整请求链。
