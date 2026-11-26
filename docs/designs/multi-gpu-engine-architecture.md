# NInfer 多卡引擎架构设计

> **文档定位**：NInfer 多卡引擎重构的唯一权威架构与实施蓝图（已融合源码级对抗复审与第一性原理收敛）。
> **演进前提**：本次属于引擎骨架的大版本演进，**重构期间不要求中间状态随时可用，内部合同不保留历史兼容分支**，一步到位立清所有权边界。
> **平台无关原则**：计算、图捕获、性能剖析与集合通信的硬件平台差异严格收敛在既有的 [`src/core/`](../../src/core/)、[`src/ops/`](../../src/ops/) 与 [`distributed::detail::Backend`](../../src/distributed/communication/implementation.h#L9) 内部；**引擎、控制面、前缀路由、多副本调度与模型并行骨架完全平台无关**，后续无论接入国产 GPU 集群还是其他多卡服务器，均不触动引擎核心架构。
>
> **两大目标负载战场（最终面向 8 卡 / 2×8 16 卡集群）**：
> 1. **Decode 低延迟 / 高并发战场（`PP = 1`）**：`tp8+ep8`、`dp8+ep8`、`tp2+dp4+ep8`、旗舰 **`tp2+dp4+全局adp8+ep8`**，以及解耦纯 DP（如 `Qwen3.5 dp8`、`tp2+dp4+ep2`）。
> 2. **Prefill 多请求吞吐 / 未来 PD 分离 P 端战场（`PP > 1, W = 1`）**：`pp2+tp4`、`pp4+tp2`、`pp2+tp8`（2×8 跨机），以及双机 Spark 实机 `pp2+tp1`。

---

## 目录

- [一、设计边界、本轮取舍与第一性原理准则](#一设计边界本轮取舍与第一性原理准则)
  - [1.1 本轮显式边界与 9 项取舍](#11-本轮显式边界与-9-项取舍)
  - [1.2 第一性原理：明确拒绝的 5 类伪复杂度](#12-第一性原理明确拒绝的-5-类伪复杂度)
- [二、总体架构：四层拓扑、多副本骨架与模块职责](#二总体架构四层拓扑多副本骨架与模块职责)
  - [2.1 四层嵌套并行模型](#21-四层嵌套并行模型)
  - [2.2 单主控多副本进程与线程骨架](#22-单主控多副本进程与线程骨架)
  - [2.3 平台无关边界与三步准备流水](#23-平台无关边界与三步准备流水)
- [三、核心运行机制与源码级落地合同](#三核心运行机制与源码级落地合同)
  - [3.1 连接直挂、单一监督者、两阶段零重启启动与资源核算](#31-连接直挂单一监督者两阶段零重启启动与资源核算)
  - [3.2 端到端信用闭环、命令序解耦与安全取消](#32-端到端信用闭环命令序解耦与安全取消)
  - [3.3 双层准入许可、无状态预路由与四套缓存管理](#33-双层准入许可无状态预路由与四套缓存管理)
  - [3.4 耦合 EP 往返合同、可行集握手协调与 ZeroSource](#34-耦合-ep-往返合同可行集握手协调与-zerosource)
  - [3.5 Graph 捕获与回放（跨 Rank 异构与通信门控约束）](#35-graph-捕获与回放跨-rank-异构与通信门控约束)
  - [3.6 PP 流水线设计（`dp = 1, pp > 1, W = 1`）](#36-pp-流水线设计dp--1-pp--1-w--1)
- [四、公共 Engine 多域语义与外围子系统适配](#四公共-engine-多域语义与外围子系统适配)
  - [4.1 公共 `Engine` 多域语义合同](#41-公共-engine-多域语义合同)
  - [4.2 配置、可观测性与外围模块](#42-配置可观测性与外围模块)
- [五、分层验证体系与大演进实施路线](#五分层验证体系与大演进实施路线)
  - [5.1 依托现网原生边界的分层验证](#51-依托现网原生边界的分层验证)
  - [5.2 必过核心不变量测试矩阵](#52-必过核心不变量测试矩阵)
  - [5.3 实施阶段与源码推进路线（Stage 1 → Stage 3 → 平台接入）](#53-实施阶段与源码推进路线stage-1--stage-3--平台接入)

---

## 一、设计边界、本轮取舍与第一性原理准则

### 1.1 本轮显式边界与 9 项取舍

本轮重构遵循 **「先立清所有权与关键合同，高阶分支按实测证据演进」**：

| 本轮刻意不做 / 显式限定 | 理由与代价 | 架构预留的演进边界 |
| :--- | :--- | :--- |
| **1. `PP = 1` 按单计算槽落地** | 提前在 `PP=1` 开 2 槽需双份推测记录/采样缓冲并牵动复杂重叠计时，且拆半批损失带宽效率 | Stage 1 先打通 `DistributedExecutor` 与 `WorkerLoop` 多帧独立推进；Host `ReplicaEngine + Session + RankProgram` 多槽统一作为 Stage 3 首批次落地 |
| **2. `PP > 1` 显式限定 `dp = 1, W = 1`** | `W > 1` 跨阶段需复制多份线性状态记录窗并增设反向草稿链路；Stage 3 主攻多请求 Prefill / P 端 | 启动校验限定 `pp > 1` 仅运行 `dp = 1, W = 1`；未来若需扩展再按每槽独立计费 |
| **3. 步内双微批重叠（DBO / TBO）** | 复杂度高且对混合线性注意力收益不稳；但不预设“互联快就无重叠价值”（尤其跨机 EP16 风险显著高于 `pp2+tp8`） | 不在协议塞占位字段，未来按目标机通信占比实测评估后版本化扩展 |
| **4. 复杂黑盒式延迟预测器** | 状态分支多难验证；改为透明可测的双向公平约束（限制 Prefill 占用与间隔，同时确保混合负载下 Decode 不被长续块饿死） | 策略收敛在 `EpStepCoordinator::decide()`，参数通过目标负载实测标定 |
| **5. 自研共享内存（SHM）控制环** | Grammar 稀疏分片后载荷小，补齐端到端信用与 RX 唤醒的 Unix Socket 足够支撑本轮 | 复用 [`Executor<Types>`](../../src/runtime/executor/executor.h#L14) 边界且每副本独占 [`Transport`](../../src/distributed/transport.h#L86) |
| **6. 多卡关闭 Host KV Swap（`host_capacity_bytes = 0`）** | 暂不处理同机多副本 Host KV 瓜分与复杂回收；**不假设 8 卡显存必然充裕**，超出物理显存的配置在启动期直接拒绝 | 仅关闭 Host KV Swap，Host PLE、媒体、Staging、Pinned 与主控目录仍独立计费 |
| **7. 跨 DP 副本活体检查点迁移与 PD 网关** | 热点前缀靠有成本的溢流路由到冷副本重算扩散，不建全局 `dp×tp` 拷贝域（注：耦合 EP 下这次重算会产生全组 Prefill 成本） | `PeerExchange` 仅限副本内；状态清单按逻辑层与逻辑头定义，供未来 PD 分离复用 |
| **8. 同时叠加 `PP > 1` 与 `DP > 1 / EP > TP`** | 避免同时耦合流水线与跨副本协调；且本轮限定 EP 与 TP 为**整除嵌套关系** | 拓扑校验允许「`pp=1` 的合法 `dp/tp/adp/ep`」或「`dp=1` 的合法 `pp/tp/ep`（`tp % ep == 0`）」 |
| **9. 异步 HTTP Server 重写 / 多卡 Expert Offload** | HTTP 协议层不阻塞核心执行验证；`ExpertCache` Offload **仅单卡 5090 使用，多卡固定禁用** | 保留单一共享 `Frontend` 与公共 `Engine` 接口 |

### 1.2 第一性原理：明确拒绝的 5 类伪复杂度

对照现网代码真实结构，实施期间严格执行以下剃刀准则，**绝不引入叠床架屋的中间层**：

1. **不搞假想平台降级探测矩阵（如多维 `GraphCapabilities` 或哑页兜底）**：目标 GPU 平台（含国产多卡平台）完整支持所需图捕获、局部节点开关与通信语义。硬件/驱动/剖析器适配下沉在 `core / ops / Backend` 内部；通信图节点可否动态门控仅保留现网唯一的 [`Group::supports_graph_gating()`](../../src/distributed/communication.h#L117)。
2. **不在 `Executor<Types>` 底下再造 `IRankEndpoint / DirectEndpoint`**：现网 [`Executor<Types>`](../../src/runtime/executor/executor.h#L14) **本身就是单卡与分布式的统一多态边界**。单卡 `dp=1, pp=1, tp=1` 直接保留零序列化、同线程直调的 [`LocalExecutor`](../../src/runtime/executor/local_executor.h)；多卡直接升级 [`DistributedExecutor`](../../src/runtime/executor/distributed_executor.h)。
3. **不在生产代码中塞 `IEpExchange / ICollectivePort / INodeInventory` 等 OOP 虚接口群**：集合通信与跨副本 EP 交换直接作为 [`distributed::Group`](../../src/distributed/communication.h#L107) 与 `ops::*` 原语（底层由 [`Backend`](../../src/distributed/communication/implementation.h#L9) 承接）；[`WorkerLoop`](../../src/runtime/executor/worker_loop.h#L15) 将连接层参数化即可在 CPU 单测直跑真实分发状态机与真实 Socket。
4. **不改造 `Frontend` 线程模型，也不再外包一层重复信号量**：现网 [`MediaCache`](../../src/models/common/frontend/media_cache.cpp#L197) 内部已自带 `MediaPreparationGate`、有界 `TaskPool` 与同键 single-flight，内置 XGrammar 编译器（[`grammar_compiler.cc`](../../third_party/xgrammar/cpp/grammar_compiler.cc#L1234)）也已具备 2-slot 冷编译信号量与 single-flight。多副本直接零改动共享同一个 `Frontend` 实例即可。
5. **不推翻解耦 DP 模型的 `Session` 接口**：纯 DP 与副本内 TP/EP（`ep <= tp`）模型的 `Session` 保持原有单步启动闭环（如 `Qwen3.5` `Session` 零改动）；仅在耦合 EP（`pp == 1 && ep > tp`）路径提供轻量副本侧可行集握手 `prepare_step_offer() → decide() → submit_step()`。

---

## 二、总体架构：四层拓扑、多副本骨架与模块职责

### 2.1 四层嵌套并行模型

任意部署形态统一满足 **`world_size = dp × pp × tp`**，按四层严格嵌套（**铁律：EP 绝不跨 PP 阶段；本轮 EP 与 TP 满足整除嵌套关系**）：

```text
Cluster World（总卡数 = dp × pp × tp）
  └── 第 1 层：DP 副本（Replica，共 dp 个）
        │  独立接请求、独立 ReplicaEngine 线程、独立 Scheduler/Session/ContextDirectory、独立 Executor
        │  权威维护全局 RankId ↔ 副本内局部下标 [0, pp*tp) 的唯一双射
        │
        ├── 第 2 层：PP 流水线阶段（Stage，p = 0 .. pp-1，每阶段占 tp 张卡）
        │     │  持有连续层区间 [layer_begin(p), layer_end(p)) 的权重、KV 与线性状态
        │     │  · 只有 Stage 0 持有 Embedding 与 Vision
        │     │  · 只有 Stage pp-1 持有 LM Head、采样、Grammar、草稿头与 result_owner
        │     │  · 相邻阶段通过点对点 PpStageLink 传递完整层边界载体
        │     │
        │     ├── 第 3 层：阶段内 TP 组（tp 张卡）—— 切分稠密矩阵、线性注意力、词表与共享专家
        │     │     └── 第 4 层：Attention 属主（ADP Owner，每副本 A 个，每属主占 tp / A 张卡）
        │     │           独占该属主请求的 MLA 潜变量 KV，消除 TP 复制浪费
        │     │
        │     └── 阶段内 EP 专家组（ep 张卡，绝不跨 PP 阶段）
        │           · 若 ep <= tp：EP 闭环在副本内部（解耦 DP，独立失败域）；当 ep < tp 时，计划显式声明专家内 TP 分组
        │           · 若 ep > tp：EP 横跨 ep / tp 个完整 DP 副本，每组共享一个 EpStepCoordinator 与同一失败域
```

> **坐标与参数口径铁律**：
> 1. 沿用 [`include/ninfer/deployment.h`](../../include/ninfer/deployment.h#L15-L18)，`adp` 表示**单副本内的属主数 `A`**（如旗舰 `tp2 + dp4 + 全局 adp8 + ep8` 配置为 `{tp=2, dp=4, pp=1, adp=2, ep=8}`，日志同时输出 `owners_per_replica=2` 与 `total_attention_owners=8`）。
> 2. **严格区分全局 `RankId`、副本局部下标与并行坐标**：`Transport`、进程监督、日志与公开 `RankSnapshot` 统一使用全局 `RankId`；副本内 `DistributedExecutor` 成员数组、`ContextDirectory` 分片与模型 `Codec` 统一使用 `ReplicaTopology` 提供的副本局部下标 `[0, pp*tp)`，在边界处单点转换，严禁把全局 `RankId` 直接当作局部 vector 下标。

**目标覆盖拓扑速查表**：

| 场景 | 典型组合 | `dp` | `pp` | `tp` | `adp(A)` | `ep` | 跨副本同步？ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 单卡基线 | 单卡本地执行 | 1 | 1 | 1 | 1 | 1 | 否（`LocalExecutor` 同线程直调） |
| 解耦纯 DP | `Qwen3.5 dp8` / `tp2+dp4+ep2` | 8 / 4 | 1 | 1 / 2 | 1 | 1 / 2 | 否（独立失败域） |
| Decode 主力 8 卡 | `tp8+ep8` / `dp8+ep8` / `tp2+dp4+ep8` | 1 / 8 / 4 | 1 | 8 / 1 / 2 | 1 | 8 | 仅 `dp > 1` 时 |
| **Decode 旗舰 8 卡** | **`tp2 + dp4 + 全局 adp8 + ep8`** | 4 | 1 | 2 | 2 | 8 | **是**（共享 `EpStepCoordinator`） |
| **Prefill / P 端 8 卡** | **`pp2+tp4+ep4`** / **`pp4+tp2+ep2`**（`W=1`） | 1 | 2 / 4 | 4 / 2 | 1 | 4 / 2 | 否（多请求 PP 流水） |
| **2×8 跨机 16 卡** | `tp2+dp8+全局adp16+ep16` / `pp2+tp8+ep8` | 8 / 1 | 1 / 2 | 2 / 8 | 2 / 1 | 16 / 8 | 是（跨机 EP 延迟敏感）/ 否 |
| **双机 Spark 实测** | `dp2+ep1` / `dp2+ep2` / `tp2+adp2+ep2` / **`pp2+tp1`** | 2 / 2 / 1 / 1 | 1 / 1 / 1 / 2 | 1 / 1 / 2 / 1 | 1 / 1 / 2 / 1 | 1 / 2 / 2 / 1 | 按需（不覆盖 `tp>1 ∧ ep>tp`） |

---

### 2.2 单主控多副本进程与线程骨架

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          主控进程（Coordinator Process）                          │
│                                                                                  │
│  [HTTP / CLI / Bench 调用线程]                                                    │
│        │  零改动共享单一 Frontend（内置 Media 任务池/限流门 + XGrammar 编译信号量）  │
│        ▼                                                                         │
│  [EngineCoordinator + DpRouter]                                                  │
│  （双层许可 C+pending / C + 惰性 PrefixKey 预路由 + 三分离容量选路 + 提交切口测量栅栏） │
│        │                 │                 │                 │                   │
│        ▼                 ▼                 ▼                 ▼                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ReplicaEngine0│  │ReplicaEngine1│  │ReplicaEngine2│  │ReplicaEngine3│          │
│  │ 独占Scheduler │  │ 独占Scheduler │  │ 独占Scheduler │  │ 独占Scheduler │          │
│  │ 独占Session   │  │ 独占Session   │  │ 独占Session   │  │ 独占Session   │          │
│  │ 独占KV目录    │  │ 独占KV目录    │  │ 独占KV目录    │  │ 独占KV目录    │          │
│  │ Executor<T>  │  │ Executor<T>  │  │ Executor<T>  │  │ Executor<T>  │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                 │                  │
│         └────────────┬────┴─────────────────┴────┬────────────┘                  │
│                      ▼                           ▼                               │
│           [EpStepCoordinator]       [LocalExecutor（1卡直调） /                   │
│        （仅 pp=1 && ep>tp 按域建立）  DistributedExecutor（每副本独占 Transport）] │
│                                                                                  │
│  [NodeProcessSupervisor（唯一子进程拉起/日志排空/按域信号回收属主线程）]             │
└──────────────────────────────────────────────────┼───────────────────────────────┘
                                                   │ （每副本仅连自身 pp×tp 个 Worker）
                                                   ▼
┌───────────────────────────┐            ┌───────────────────────────┐
│     Worker 0 (Rank 0)     │ ... N 张卡  │   Worker N-1 (Rank N-1)   │
│ 有序 Submit + 票据续行循环  │            │ 有序 Submit + 票据续行循环  │
└───────────────────────────┘            └───────────────────────────┘
```

---

### 2.3 平台无关边界与三步准备流水

- **平台事实全部沉入底层基础设施**：设备上下文、物理/共享内存事实、流/事件/图原语及 GPU Profiler 启停统一归属 [`src/core/`](../../src/core/device.h)（消除 `runtime/` 直接调特定平台 profiler 或编译期宏判断共享内存）；算子归属 [`src/ops/`](../../include/ninfer/ops/)；通信组归属 [`distributed::Group`](../../src/distributed/communication.h) 与 [`Backend`](../../src/distributed/communication/implementation.h)。
- **三步并行准备流水**：
  1. **第 1 步（框架层）**：验证并规范化部署配置，生成四层 `ClusterTopology`（物理 rank 坐标、每副本局部映射、`failure_domain_id`）；
  2. **第 2 步（模型架构层）**：按**副本局部槽位数 `ceil(C/dp)` 与局部属主数 `adp`** 构建具体 `ParallelPlan`（显式目标绑定三态权重计划、State/KV 布局、有序物理通信组角色、EP/PP 边界合同与动态执行事实），最终生成覆盖模型切分与通信角色的 **`TopologyPlanDigest`**；
  3. **第 3 步（通信与资源模块）**：完成常驻通信组连接预热与一次性权重物化，在统一物理采样时刻求解全集群 GPU / `NodeId` 容量预算。

---

## 三、核心运行机制与源码级落地合同

### 3.1 连接直挂、单一监督者、两阶段零重启启动与资源核算

#### 1. 单一属主进程监督与完整协议态移交
1. **`NodeProcessSupervisor` 唯一进程属主**：在主控内建立与集群同寿命的专职监督线程，**独占所有本地/SSH 子进程的 `posix_spawn`、日志管道持续排空、状态记录与 `terminate/waitpid` 回收**（彻底杜绝主线程/副本线程并发 reap 或日志竞争，同时解决 Linux `PR_SET_PDEATHSIG` 绑定调用线程的寿命问题；某失败域慢速销毁期间健康域管道照常排空）。
2. **连接直挂与单消费者协议态移交**：Worker `Join` 时直接按所属副本挂入 `replica_transports_[r]`。启动期由主启动编排作为每套 `Transport` 的唯一消费者驱动 Phase 1/2；全员 `Ready` 后，将 `Transport` **连同已出队缓存消息（`inbox`）、`epoch`、全局↔局部 rank 映射表与首错优先级状态**完整移交对应 `ReplicaEngine` 的 `DistributedExecutor` 独占，运行期监督线程绝不触碰 `Transport` 收发队列。
3. **两阶段零重启启动**：
   - **Phase 1（常驻底座 `LoadedShard`，仅执行 1 次）**：Worker 核验真实 `.ninfer` 产物与本地分片计划摘要匹配 → 建立并预热常驻通信组 → 按节点预估字节门控并发 Staging 峰值并物化权重 **1 次**。
   - **Phase 2（可恢复执行层 `PreparedProgram`，原地重试）**：在全体权重与常驻通信落盘后的**同一物理时刻**采样，求解各卡预算并执行探针/校准/分配（覆盖 `entered` 前后所有可恢复分配失败边界）；窄图回退或自动容量纠偏仅在集合安全边界向参与方发 `Release` 释放 `RankProgram`/Graph/KV 与主控目录后原地重试（保留 `attempt` 独立身份），**Worker 进程、通信组与已加载权重不动**。

#### 2. 目标绑定级三态权重与精确物理资源核算
1. **目标 `WeightId` 三态分类与绑定器联动**：
   - 按模型实际定义的参数集合校验每个**目标绑定 `WeightId`** 有且仅有一次归入 `Whole`、`Sliced`（允许单次拼装读取同一源卷积参数多段，或别名绑定同一切片）、`Expert` 或 `Omitted`，未分类直接抛错；
   - **同步改造选择器与 [`bind_selected_weights()`](../../src/models/common/parallel_weights.cpp#L149)**：`Omitted` 参数必须显式清空本地 `weight.view` 并跳过其激活校准 I/O 请求（保留独立专家库与 `Qwen4Exp` 独立 PLE 请求机制）；
   - **词表 128 对齐一致性**：当词表足够切分时按 128-token 对齐划分（小词表夹具定义显式退化规则），且同一分片区间严格贯穿 Target / MTP / Draft 输出头切片、logits、采样偏移、评分归约与稀疏 Grammar 掩码。
2. **不重不漏的物理 GPU 与 `NodeId` 内存求解**：
   - **单一采样基准防重复扣减**：全员权重与通信预热完成后统一重采样空闲内存——该快照已自然扣除同卡多 rank 权重与常驻通信显存，求解器直接在此净余量上分配各 rank 运行时工作集与 KV（同卡测量型探针串行或合并计峰，防止互相污染增量）；
   - **主控目录计入 `coordinator_node`**：全部 `dp` 个副本的主控侧 `ContextDirectory` 元数据（含 Worker 全在远端节点如 Spark `.51` 的副本！）统一扣减 `coordinator_node` 的 Host / UMA 预算；共享 `Frontend` 仅计 `media_live_bytes + grammar_cache_bytes`（[`media_cache.cpp`](../../src/models/common/frontend/media_cache.cpp#L225) 已将缓存驻留媒体计入 `media_live_bytes`，绝不重复叠加 `media_cache_bytes`）；
   - **四类设备资源独立计费**：请求驻留态/`token_counts`（按局部槽数）、Graph I/O 与采样区（Stages `0..pp-2` 零采样缓冲）、EP 接收工作区（按 $N \times \min(k, e_r)$）、PP 链路与票据保留态各自独立核算。

---

### 3.2 端到端信用闭环、命令序解耦与安全取消

#### 1. 端到端无阻塞信用不变量（覆盖流式媒体与异步遥测）
单条有序字节流上一旦前序帧因接收缓冲耗尽阻塞读头，后续续行帧在内核 Socket 缓冲中也无法越过。因此信用设计必须端到端闭合：
- **发送端按槽位严格限流 + 独立大输入滑动窗口**：在途计算/上下文票据数严格受限于 `(compute_frames, context_frames)`；流式大媒体输入（`InputTransfers` 多块 `256 KiB` chunk）采用独立的 per-peer 字节与帧数窗口，随 Worker 将 chunk 拷入固定 Pinned 接收槽（`InputRelease(old)` 严格先于 `InputBegin(new)` 复用该槽，且注册异常可原子回滚）渐进归还额度；
- **双向票据控制预留 + 非请求遥测独立限流/合并**：双向预留覆盖所有合法在飞票据的 `Submit / Supply / Commit / Discard`、`InputReady / InputRelease`、退役、栅栏应答与错误帧；Worker 周期性 `Observation` 与 `Progress` 独立限额并支持覆盖合并，绝不挤占控制额度；普通 `Submit` 遇瞬时背压返回重试而非直接判定执行器崩溃；
- **有界持续排空与精确休眠**：即使某个 `Submit` 因执行槽忙暂缓启动，`WorkerLoop` 每轮仍以有界步数排空 Socket 入站帧（不因局部命令队列长度停收包）；空闲等待统一监听入站可读、发送额度恢复、错误、停止与最近截止时间，且用**真实在飞设备事件**（而非包含 Host 等待 `Decision / Result` 的宽泛 `active` 状态）决定是否轮询 GPU。

#### 2. 命令序解耦、按事件定义应答集与 `Supply/Discard` 集合安全规则
1. **接收序、单票据序与通信组序分离**：`Submit` 按副本/域单调 `step_seq` 保序启动（且计算步不得越过安装其驻留态的 `Bind` 或旧代数释放）；续行命令按完整 `(epoch, ticket_id, frame_kind, index, generation)` 挂载至目标票据并在到达合法本地阶段时推进，保证共享同一 communicator 的所有集合通信在各参与 rank 上顺序 100% 一致。
2. **按事件类型定义应答集合**：Compute 票据仅 `result_owner` 回报 `DecisionNeeded` 与 `ResultReady`，仅 `decision_owners`（`decision_owners ⊆ participants`，即末阶段 TP 卡）接收 `Supply` 并回报 `DecisionConsumed`；**Context 票据保持全体 `participants` 回报本地 `ResultReady` 以归并传输统计**；全体 `participants` 回报最终 `DeviceRetired`。
3. **`Discard` 集合通信安全边界（以主控原子 `Supply` 扇出为界）**：
   - **`BeforeSupply`（主控未发出 `Supply`）**：对完整票据（含仍在 `submit_queue_` 排队的票据）锁存取消标志，**照常按 `step_seq` 跑完含集合通信的前向公共前缀**，在进入等待外部 `Decision`（或无外部决策工作的 `Result`）安全点时直接转入退役，不无限干等已抑制的 Grammar 回复；
   - **`AfterSupply`（主控已完成 `Supply` 原子扇出）**：即便某 Worker 先收到 `Discard` 再到达本地 `Decision`，也**必须应用该 `Supply` 并跑完采样阶段的集合通信直至本地 `Result`**，随后不执行 `Commit` 直接退役并以该 `Discard` 命令号回报 `DeviceRetired`；
   - **`AfterCommit`**：保持原 `Commit` 执行与命令身份退役不变；混合批次单行取消继续走普通行级 `CommitDecision`（`accepted_tokens = 0`）。

---

### 3.3 双层准入许可、无状态预路由与四套缓存管理

#### 1. 双层全局许可与抢占恢复防饿死协议
严格区分两个不同生命周期的全局许可证，各归单一权威管理：
- **全局在途许可 `GlobalOutstandingGate`（上限 `C + max_pending_requests`）**：在调用线程进入 `make_output_session()` 前原子获取并分配全局递增 `(request_id, publication_order)`；请求被抢占暂停（`paused_`）期间**继续持有**该许可；仅当终态响应已发布**且**调用方句柄释放/放弃后精确归还一次。
- **全局驻留许可 `GlobalResidencyGate`（上限 `C`）**：从主控将物理槽绑定权授予某副本起计入（覆盖在途派发与 `materializing_`），在暂停真正腾出物理槽、绑定取消/失败或终态释放物理槽时归还。
- **抢占与恢复优先权铁律**：当某副本为恢复其挂起的 `progress_owner` 而暂停受害者腾出槽位时，**在槽位腾空瞬间直接将该驻留许可原地移交给可运行的本地恢复请求**（若暂停仅用于给已在槽内的 `progress_owner` 腾显存则归还全局池）；所有可运行的 `paused_` 恢复请求优先于中心队列新请求获取驻留许可，即使全局驻留打满 `C` 也绝不被新请求流饿死。当部分副本故障时，有效并发上限自然收敛为 $\min(C, \text{健康副本数} \times \text{局部槽数})$ 并对外如实暴露。

#### 2. 无状态惰性 `PrefixKey` 预路由与目录级容量口径
1. **可移植请求所有权与惰性 `key_at(frontier)`**：
   - 现网 `Session::plan_request()` 会消费 `PreparedPrompt` 并绑定特定副本的 `PrefixStore`。重构后让 `PreparedPrompt` 持有不可变可移植输入，并提供模型权威的**惰性 `key_at(frontier)`**（复用 [`PrefixFingerprints`](../../src/models/common/context/prefix.cpp#L131) 从最近已算前沿增量递推，包含真实 token、位置、媒体身份与 domain）；
   - 副本侧 `PrefixIndex` 在插入检查点句柄时一并保存 `(portable_key, recoverable_frontier, placement)`，在插入、替换、消费与删除时按序向 `DpRouter` 发布引用计数可见性增量（支持句柄已失效后的精确退订及故障副本整 `epoch` 作废）；`DpRouter` 仅在请求开启前缀复用时查询已通告的前沿长度。
2. **复用目录权威统计与未绑定无损退回（Bounce）**：
   - 各副本直接从所属 `ContextDirectory` 与 `ResourceManager` 导出一致快照（去重共享页，区分活跃借用/预留与真正具备驱逐权限的缓存，并同时反映 State、目录元数据与最紧 rank 约束）供 `DpRouter` 排序候选副本；
   - 最终可行性由目标副本 `assess_binding()` 权威裁决：若某新请求在穷尽目标副本本地来源与 placement 后仍因瞬时容量不足无法绑定（且尚未建立绑定所有权），释放本地来源临时引用与驻留许可，**携带原始可移植 `PreparedPrompt`、`request_id` 与 `pending_deadline` 退回 `DpRouter`** 在资源版本变化后重选健康副本。
   - 可选 `max_cached_checkpoints_per_replica` 通过既有回收权限直接作用于唯一、当前无租约的可驱逐句柄（租约结束时重访，不在每次发布检查点时构造全量排序受害者游标）。

---

### 3.4 耦合 EP 往返合同、可行集握手协调与 ZeroSource

> **生效条件**：仅当 `pp == 1 && ep > tp` 时按耦合 EP 域建立 `EpStepCoordinator`；解耦形态（`ep <= tp`）完全不经过该协调路径。

#### 1. 跨副本 EP 数据往返合同与最坏缓冲上限
- **往返流程与三条不变式**：唯一指定 TP 来源卡产生派发 `(replica, ticket, row, col)` → 全局路由与权重 → 目标专家 rank 计算本地专家贡献 → 按身份归并回原副本（与到达序无关） → 恢复本副本 TP/ADP 布局；`ep < tp` 专家内 TP 归约不误计重复路由；共享专家在本副本内独立计算绝不重复累加。
- **最坏显存上限**：单步全 EP 组唯一来源 token 上限为 $N$，每 token 选 $k$ 个不同专家，rank $r$ 持 $e_r$ 个专家，接收上限严格按 $M_{\max}(r) = N \times \min(k, e_r)$ 预留；装不下时在启动期收窄单步上限，运行期绝不丢 token 或临时扩容。

#### 2. 可行集预留握手（`prepare_step_offer → decide → submit_step`）与双向公平调度
针对耦合 EP 下单凭候选成员数尚不足以承诺逐行草稿窗口或 Prefill 边界的事实（如逐行 KV 增长许可 `[4, 0]` 不等价于 `[2, 2]`，收窄 Prefill token 数也会改变该行是否采样、是否触达检查点或是否包含完整视觉单元）：
- **三步确定性握手（仅耦合 EP 路径启用）**：
  1. **`Session::prepare_step_offer() -> StepOffer`**：副本线程持有该 move-only 提议（内含已预留的执行帧、冻结成员、逐行 KV 增长授予上限、捕获/视觉预留以及本地图路由存在性），仅将几何与合法选择集合上报给 `EpStepCoordinator`（空闲或正在 `context_stream` 分段整理的副本上报 `ZeroSource` 能力）；
  2. **`EpStepCoordinator::decide()`**：在全部参与者宣告的合法交集内选定本步 `(WorkContent, ExecMode, 统一桶或 Eager, 合法验证宽度与 agreed_mtp_depth, 各副本 Prefill 选集)` 并分配全域单调 **`domain_step_seq`**；
  3. **`Session::submit_step(offer, decision)`**（名称与模型输出采用的 `commit` 严格区分）：**凡落在各参与者宣告合法集内的决定，必能零容量背压直接提交**；未选中提议立即释放预留；达成一致后若某请求取消，仅抑制其输出采用，仍照常跑完已约定的全域集合通信。
- **双向公平与流序安全铁律**：
  - 守住全组单步 Prefill token 上限、耗时预算与最长服务间隔；**只要全组仍有可执行 Decode，每跑完 1 轮含 Prefill 的 `Mixed` 步后必跟至少 1 轮纯 Decode**；严重倾斜纯 Decode 切 `Decode + Eager` 时**保持 MTP 开启**；
  - **单槽流序安全**：任何上一轮仍处于未决 `Decision` 或尚未将 `Commit / AfterSupply-Discard` 入队并确认流序隔离的副本，严禁进入下一轮协调（连作为 `ZeroSource` 也不允许）。
- **严格的 `ZeroSource` 与分段碎片整理**：`ZeroSource` 绕过 `Session/ContextDirectory` 直提 `Executor`（零请求存储句柄，测试期以 `PROT_NONE` 内存保护直接验证零读零写）；单副本池整理拆为有界小段在独立 `context_stream` 推进，计算流持续服务远端专家请求。

---

### 3.5 Graph 捕获与回放（跨 Rank 异构与通信门控约束）

| 场景 | Capture 规则 | Replay 规则 |
| :--- | :--- | :--- |
| **1. 副本内 ADP 异构** | 同副本 TP 组按 2D 桶 `(rows, attention_rows)` 同步 Capture | 空余行标空跳过；各卡按本地 `max_visible_tokens` 独立热补丁 [`ContextBucketSite`](../../src/models/common/graph/patch_site.h#L11-L38) |
| **2. 解耦 DP** | 各副本独立捕获 | 各副本完全独立跑 Graph、Eager、整理或休眠 |
| **3. 耦合 DP + EP** | 同 EP 组全员按一致桶清单、通信次序与 communicator 顺序同步 Capture | 均衡纯 Decode 跑统一桶 Graph（各卡本地补丁 `ContextBucket`，空闲副本跑 `ZeroSource`）；严重倾斜纯 Decode 跑 `Eager + MTP`；`Mixed / Prefill` 跑变长 Eager |
| **4. PP 流水线** | **`PpStageLink` 收发置于图外**；各 Stage 仅拉本阶段 TP 组独立 Capture | 不同 Stage 并发执行不同微批；同 Stage 复用前向工作区严格服从 §3.6 寿命表 |

> **通信门控硬边界（[`Group::supports_graph_gating()`](../../src/distributed/communication.h#L117)）**：当后端不支持跳过已捕获图中的集合通信节点时，① `ZeroSource` 图模式仅对**纯本地读写请求 Context 的算子区段**调 `set_enabled(..., false)`，副本内 TP 与跨副本 EP 通信节点全部保持启用；② 动态 `agreed_mtp_depth` 拆分为「1 张 Target Verify 图 + 独立可串联的 MTP 单步图」按深度顺序 launch，不在单张大图内关通信节点。

---

### 3.6 PP 流水线设计（`dp = 1, pp > 1, W = 1`）

#### 1. Host 多槽执行合同、`W = 1` 预置 Grammar 与 5 类缓冲寿命
在 Stage 3 首批次（接入 `PpStageLink` 之前）一次性打通 Host `ReplicaEngine` + `GLM5Next::Session` 多槽重叠合同，再开启 `RankProgram` 多物理帧：
- **Host 多槽四项完备性改造**：① 调度循环由全局 `compute_pending()` 门控改为“存在空闲计算帧 + 每序列至多 1 张在飞票据的不相交选集”；② 按完整票据独立追踪**语义进度完成**与**物理 `DeviceRetired`**（妥善处理未采样 Prefill/Replay/Control 先回报物理完成再触发延迟语义事件的时序），两项义务均结清后才回收槽位；③ `Session` 按 `reservation.frame()` 选择事务槽并在入口校验序列不相交；④ 副本线程以单一时钟记录 Host 耗时，票据专属调用归属冻结成员，共享等待归属活跃 Decode 并集，绝不按在飞票据数倍乘引擎总耗时。
- **`W = 1` 预置稀疏 Grammar 与私有票据保留态**：随 `Submit` 提前下发稀疏 Grammar Mask 至 Stage `pp-1` 一步完成前向+掩码+采样；采样 ID、行映射与处理计数存入 `flights_[slot]` 私有槽位，彻底解除 [`commit.cpp`](../../src/models/glm5next/program/commit.cpp#L22) 对共享 `frame` 的依赖。

| 资源类别 | 属主 | 允许复用 / 释放的精确条件 |
| :--- | :--- | :--- |
| **阶段前向工作区（每 Stage 1 份）** | 当前 Stage `RankProgram` | 本阶段前向计算 **+ 边界输出考入 `pp_send_buf`** 在设备流上全部完成 |
| **双发送缓冲 `pp_send_buf[2]`** | `PpStageLink` | 对应跨阶段 P2P 发送事件完成；双满时仅背压新 `Submit`，非阻塞继续处理续行与取消 |
| **接收缓冲** | `PpStageLink` | 下游完成读取或已考入独立输入区 |
| **票据保留态（采样 ID、行映射、计数）** | `flights_[slot]` | 该票据 `Commit / Discard` 最终设备消费者回报 **`DeviceRetired`** |
| **State / KV 租约** | `Session` 与票据事务 | 所有参与阶段与传输消费者全部 **`DeviceRetired`** |

#### 2. 层边界拆融合、`StageBoundaryPayload` 与动态微批
- 仅在 `layer + 1 == layer_end(p)` 的真实跨 Stage 边界拆开 [`finish_layer()`](../../src/models/glm5next/execution/text.cpp#L222) 与下一层预处理的融合；
- `StageBoundaryPayload` 明确声明多张量（如 `GLM5Next` mHC 多流残差 + 跨层特征）及 `TpLayout { Sharded, Replicated }`（复制态支持按 `tp_rank` 各传 $1/\text{tp}$ 切片至下游再 `AllGather`）；
- 自动切层兼顾显存可行与瓶颈阶段耗时最小化；`compute_frames` 以 `pp + 1` 为起点结合往返延迟与独立并发序列数规划，动态拼装不相交微批。

---

## 四、公共 Engine 多域语义与外围子系统适配

### 4.1 公共 `Engine` 多域语义合同

| 公共接口 / 合同 | 多副本 & 耦合 EP 下的精确语义 |
| :--- | :--- |
| **幂等失败域隔离** | 拆分“标记失联不可路由（立即失效报价与 Hint、唤醒等待者）”与“后台有序销毁（先停接收与设备读者、Worker 主线程 `comm.abort()`、监督者回收该域子进程，再清理 Session 租约）”；在途请求精确结清一次并保留结果供句柄读取；若最后一个健康域失败，未分配等待请求以 `Unavailable` 结束 |
| **`is_available()`** | `Engine` 未停止，且至少存在 1 个完整、健康、符合当前 `purpose` 的服务域 |
| **`request_stop()`** | 立即停止接纳新请求；已接受的 `score_tokens()` 执行完返回；耦合 EP 域保持 `ZeroSource` 专家服务直至在途步全部退役 |
| **`begin / end_measurement()` 与 `reset_measurement_peaks()`** | **采用提交切口（Submission Cut）避免死锁**：在请求 ID 锁下划定切口，切口后新提交按自身超时等待，**切口前已接受请求（含编译语法、中心排队、暂停待恢复、分块续算）照常派发执行直至终态发布**，并待各健康参与副本确认完全静默（含延迟 Session 释放与 `InputRelease`；已完成但未消费句柄不阻塞）后执行同步栅栏；若栅栏期间某参与副本崩溃，取消幸存副本半开窗口并明确报错 |
| **`score_tokens()`** | 保持全引擎级单调用串行语义直至结果返回；耦合 EP 下同域其余副本由协调器同步驱动发射 `ZeroSource` 步 |
| **RNG 确定性合同** | 仅由 `解析后 seed + 逻辑输出位置 + 采样用途 + 抽样序号` 决定；padding、物理行号、所落副本与执行槽位绝不参与 |

### 4.2 配置、可观测性与外围模块

| 子系统 | 源码入口 | 适配要点 |
| :--- | :--- | :--- |
| **配置与启动报告** | [`options.h`](../../src/product/config/options.h)、[`file.cpp`](../../src/product/config/file.cpp#L81) | 沿用现有 `[distributed.parallel]` 五维拼写，补充 `allow_shared_device`、确定性本地纯 DP 展开、`max_cached_checkpoints_per_replica` 与可选 `stage_layers`；启动报告固定不可变，双口径输出属主数与阶段明细 |
| **不可变描述 vs 实时快照** | [`engine.h`](../../include/ninfer/engine.h#L117)、[`snapshot.h`](../../include/ninfer/observability/snapshot.h#L225) | `Engine::description()` 保持构造期不可变 `const&`（汇总启动总容量）；实时健康可用容量、去重后的物理 GPU/节点内存及 **`EngineSnapshot::replicas[]`** 放入 `EngineSnapshot`（保留故障域累计计数，仅剔除其可用容量；局部 `placement_id / ticket / progress_owner` 均带 `replica_id` 限定）；`replica_id` 写入终态 `GenerationResult` 与 `request_jsonl`（未分配请求为空） |
| **Serving / Eval / Bench** | `src/serve/`, `eval/`, `bench/` | OpenAI / Anthropic 协议零改动；`bench/` 覆盖固定负载下的 Decode 尾延迟、多请求 Prefill 吞吐/TTFT 与混合干扰 |

---

## 五、分层验证体系与大演进实施路线

### 5.1 依托现网原生边界的分层验证

直接利用现网四道边界（`Executor<Types>`、参数化连接层的 `WorkerLoop`、本地 Unix `Transport`、`distributed::detail::Backend`），零生产 `if (test)`：
- **Tier 0（本机 CPU 显式注册 CTest）**：运行真实 8/16 卡拓扑、三态权重与绑定校验、wire `Codec`、真实 Socket + CPU 测试 `Program` 下的 `WorkerLoop` 多帧乱序续行/`Discard`/流式媒体窗口、`DpRouter` 惰性 `PrefixKey` 命中/退回/抢占许可移交、提交切口测量栅栏、耦合 EP 可行集握手与 `ZeroSource` `PROT_NONE` 零访问、PP 双缓冲背压与 Host 多槽重叠。
- **Tier 1（单卡 RTX 5090 实机，`.10`）**：带声明容差与重复度标准对比重构前单卡 TTFT/TPOT 基线、独立数学参考核验分片/掩码采样/因果对数概率、同卡多进程共享显存求解、无后续提交下的空闲断连检测、空闲零自旋与杀主防孤儿。
- **Tier 2（双机 DGX Spark 实机，`.50 + .51`）**：跨机验证 `dp2+ep1`（含纯远端副本目录计费）、`dp2+ep2`、`tp2+adp2+ep2`、`pp2+tp1`，**显式分别运行 `Native` 与 `Nccl` 两条通信路线**，实测主控启动期/空闲期强杀后的远端有界自毁租约。
- **Tier 3（目标 8 卡 / 16 卡国产 GPU 集群终验）**：完成底层 `core / ops / Backend` 平台适配后，实跑 `tp2+dp4+全局adp8+ep8`、`tp8+ep8`、`dp8+ep8`、`pp2+tp4`、`pp4+tp2`、`pp2+tp8`，分别完成独立数学参考精度验收与实机吞吐/尾延迟基准报告（不以模拟运行充当 8 卡性能结论）。

### 5.2 必过核心不变量测试矩阵

| 要证明的核心不变量 | 必须提供的硬证据 |
| :--- | :--- |
| **1. 跨副本坐标映射与三态权重精确** | 非零起始全局 rank 副本（如 `dp2+tp2` 副本 1）端到端寻址正确；别名/多切片拼装、`Omitted` 视图清空与词表边界头切片逐行核验 |
| **2. 续行、大媒体流与安全取消绝不死锁** | 慢速非主卡、排队态/运行态/`BeforeSupply`/`AfterSupply` 各时机 `Discard`、流式媒体窗口打满与空闲对端暴毙均在有界时间内安全收尾 |
| **3. 双许可抢占、退回与测量切口不死锁** | 全局满载 `C` 下抢占恢复不被新请求抢走许可、容量不足无损退回重选、带中心积压/暂停请求的 `begin_measurement` 正常排空完成 |
| **4. 跨副本 EP 守恒与 `ZeroSource` 零访问** | 含复制 TP 来源、零源、热点专家、异构长度对照独立数学参考精确匹配；`PROT_NONE` 证明 `ZeroSource` 对请求 Context 零读写 |
| **5. 混合调度双向不饿死且流序安全** | 持续 Decode 叠加长续块与慢速 Grammar `Supply`，验证 `Mixed` 后必跟纯 Decode、倾斜 `Eager` 保住 MTP、未决副本不跨步插队 |
| **6. PP 多槽重叠与缓冲寿命正确** | 慢发送打满双缓冲背压、`ExecutionComplete` 早于延迟语义进度、Decode 退役与 Prefill 重叠下的精确计时归属及槽复用 |

### 5.3 实施阶段与源码推进路线（Stage 1 → Stage 3 → 平台接入）

> **跨模型构建纪律**：[`Qwen4Exp`](../../src/models/CMakeLists.txt#L102) 与 `GLM5Next`、`Qwen3.5` 共处于公共引擎变体与分布式准备测试夹具（[`test_model_preparation.cpp`](../../tests/distributed/test_model_preparation.cpp#L230)）中。因此**不搞临时屏蔽编译开关**：基础接口变更批次同步迁移 `Qwen4Exp` 既有调用点（保持其既有 `1-rank / TP2` 能力与现有测试常绿），而将 `Qwen4Exp` 的新增多卡拓扑扩展放到 `GLM5Next` 跑通 Stage 1~3 后一次性平移。同时全程验证 `NINFER_ENABLE_DISTRIBUTED=ON` 与 `OFF` 两种构建配置。

| 阶段 | 核心源码改造交付 | 阶段硬验收标准 |
| :--- | :--- | :--- |
| **Stage 1：分布式底座重构 + 多副本引擎骨架 + 解耦 DP**<br>（详见 [Stage 1 实施方案](multi-gpu-stage1-plan.md)） | ① 四层 `ClusterTopology`（全局↔局部映射、`failure_domain_id`、模型后 `TopologyPlanDigest`）+ 目标 `WeightId` 三态选择与绑定器清理 + 词表 128 对齐；<br>② 单一属主 `NodeProcessSupervisor`（本地进程父死信号 + 远端控制租约）+ `ClusterGroup` 按副本直挂 `Transport` 及完整协议态移交 + 端到端信用与流式媒体窗口 + `WorkerLoop` 命令序解耦、`BeforeSupply/AfterSupply` 安全 `Discard`、空闲休眠 + `hwloc` 全机物理核切分；<br>③ 两阶段零重启启动（全可恢复阶段 `Release` 重试）+ 单一采样时刻物理 GPU/`NodeId`（含远端副本主控目录）内存求解；<br>④ 零改动共享 `Frontend` + 单槽 `ReplicaEngine` + `DpRouter`（双层许可、抢占许可移交、可移植惰性 `key_at` 预路由与无损退回）+ `EngineCoordinator`（幂等失败域隔离、提交切口测量栅栏、不可变 `description` + 实时 `EngineSnapshot::replicas[]`）；<br>⑤ `Qwen3.5` 完整 Worker 契约（`can_discard`、无地址描述、全变体 `wire.cpp`）+ `GLM5Next` 局部几何解耦 DP（`tp <= 2, dp >= 1`）。 | ① 5090 单卡回归全绿且对比固定负载基线无退化；<br>② Tier 0 显式 CTest 跑通 `WorkerLoop` 双帧乱序续行/取消、非零基址副本、`C < dp`、满载抢占恢复、带积压测量切口与单副本故障隔离；<br>③ 5090 同卡多进程与双 Spark 实跑 `Qwen3.5 dp2`（含 MoE/媒体/语法/推测/暂停/评分）及 `GLM5Next` 解耦多副本，共享显存/远端目录核算准确、空闲 CPU `< 1%`、启动期与空闲期强杀主控无孤儿。 |
| **Stage 2：模型通用 $N$ 卡化 + 耦合 EP 往返与协调** | ① 清除 `GLM5Next` 双卡 `size == 2` 与 `1 - rank / 1 - owner` 特例，打通通用 $N$ 卡 `TP + ADP + EP`；<br>② 落地跨副本 EP 唯一来源派发、本地专家计算、按身份归并及 $N \times \min(k, e_r)$ 最坏缓冲预算；<br>③ 落地耦合 EP `prepare_step_offer → EpStepCoordinator::decide → submit_step` 可行集握手、双向公平状态机、单槽流序安全边界、严格 `ZeroSource`（遵从 `supports_graph_gating()` 局部算子门控 + 分段碎片整理）与耦合域 `EpochAbort`。 | ① 独立数学参考验证跨副本 EP 精度，`PROT_NONE` 证明 `ZeroSource` 零访问；<br>② 8/16 卡仿真验证 `tp2+dp4+全局adp8+ep8` 可行集握手、多通信组图回放与双向不饿死；<br>③ 双机 Spark 分别在 `Native` 与 `Nccl` 下实跑通过 `dp2+ep2` 与 `tp2+adp2+ep2`。 |
| **Stage 3：Host/设备多槽 + PP 流水线（`dp = 1, pp > 1, W = 1`）** | ① **首批次先落地 Host 多槽**：`ReplicaEngine` 与 `GLM5Next::Session` 空闲帧不相交调度、语义完成与物理退役解耦、单一时钟重叠计时归属（双帧测试 `Program` 锁死）；<br>② 模型跨 Stage 边界拆融合、`StageBoundaryPayload`（多张量 + `Sharded/Replicated`）与耗时均衡 `stage_layers`；<br>③ 图外 `PpStageLink` 双缓冲流同步、`W = 1` 随 `Submit` 预发稀疏 Grammar、末阶段单次采样与 `RankProgram` 票据私有保留态。 | ① Tier 0 跑通双帧重叠退役计时、`pp4+tp2` 双缓冲背压、延迟 Commit 与取消槽复用；<br>② 双机 Spark 在 `Native` 与 `Nccl` 下实跑通过 `pp2+tp1`，数值对齐独立参考且零租约泄漏。 |
| **目标集群接入与终验** | 在 `src/core/`、`src/ops/` 与 `distributed::detail::Backend` 内接入目标 8/16 卡国产 GPU 平台构建、通信库与设备侧 EP 派发/归并算子，并将 `Qwen4Exp` 平移扩展至通用多卡拓扑。 | 实机跑通 8 卡与 2×8 16 卡全部主力拓扑，通过独立数学参考精度验收并产出正式延迟/吞吐基准报告。 |
