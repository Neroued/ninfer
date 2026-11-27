# NInfer 多卡重构 Stage 1 详细设计与实施方案

> **所属总纲**：[NInfer 多卡引擎架构设计](multi-gpu-engine-architecture.md)（已融合与 `astra` 的两轮源码级对抗审查与收敛裁定）
> **阶段目标**：一次性立清 **四层拓扑与全局↔局部坐标双射、单一属主进程监督与完整协议态移交、端到端信用闭环与安全 `Discard`、两阶段零重启启动与统一采样时刻内存求解、零改动共享 `Frontend` + `EngineCoordinator + DpRouter + ReplicaEngine` 多副本骨架**，跑通 **单卡零退化基线 + `Qwen3.5`（`dp1..dp8`）与 `GLM5Next`（`dp >= 1, tp <= 2`）解耦多副本全链路**。
> **实施纪律**：内部合同不留历史兼容分支；`Qwen4Exp` 在接口变更批次同步更新既有调用点以保持现有 `1-rank / TP2` 与准备测试常绿（不搞半残屏蔽宏），新增多卡拓扑留待 `GLM5Next` 定型后平移；全程覆盖 `NINFER_ENABLE_DISTRIBUTED=ON` 与 `OFF` 构建。

---

## 目录

- [一、Stage 1 精确范围与关键收敛决策](#一stage-1-精确范围与关键收敛决策)
- [二、分模块源码级详细设计（W1 → W6）](#二分模块源码级详细设计w1--w6)
  - [2.1 W1：四层拓扑、全局↔局部坐标双射与计划摘要](#21-w1四层拓扑全局局部坐标双射与计划摘要)
  - [2.2 W2：单一属主进程监督、端到端信用与安全取消](#22-w2单一属主进程监督端到端信用与安全取消)
  - [2.3 W3：目标绑定三态权重、全边界原地重试与集群内存求解](#23-w3目标绑定三态权重全边界原地重试与集群内存求解)
  - [2.4 W4：双层许可、`DpRouter` 惰性预路由与 `EngineCoordinator`](#24-w4双层许可dprouter-惰性预路由与-enginecoordinator)
  - [2.5 W5：`Qwen3.5` 完整 Worker 契约与 `GLM5Next` 局部几何解耦 DP](#25-w5qwen35-完整-worker-契约与-glm5next-局部几何解耦-dp)
  - [2.6 W6：配置沿用、不可变描述与实时多副本快照](#26-w6配置沿用不可变描述与实时多副本快照)
- [三、编译常绿的分批次实施计划（S1.1 → S1.6）](#三编译常绿的分批次实施计划s11--s16)
- [四、Stage 1 验证矩阵与出阶段硬标准](#四stage-1-验证矩阵与出阶段硬标准)

---

## 一、Stage 1 精确范围与关键收敛决策

| 决策项 | Stage 1 最终裁定 | 依据与源码事实 |
| :--- | :--- | :--- |
| **1. 几何表示 vs 模型可运行范围分离** | `W1` 拓扑层完整表达并校验 1~16 卡四层几何（禁止 `dp>1 && pp>1` 及 `pp>1 && ep>tp`）；Stage 1 模型层实际运行 `pp = 1` 下的 `Qwen3.5`（`tp = 1, dp >= 1`）、`GLM5Next`（`tp in {1,2}, adp = tp, ep in {1,tp}, dp >= 1`）及 `Qwen4Exp` 既有 `1-rank / TP2` | 区分集群拓扑表达能力与分阶段模型接入范围，验收清晰无歧义 |
| **2. 多槽执行切分边界（选定 A2）** | **Stage 1 在 `DistributedExecutor` + `WorkerLoop` 层证明双帧（`compute_frames = 2`）乱序续行独立推进**；生产 `ReplicaEngine` 与模型 `Session` 在 `PP = 1` 下保持单计算槽（`compute_frames = 1`），Host 多槽重叠（调度解耦、语义/物理双完成、重叠计时归属）统一作为 **Stage 3 第一批次**（PP 之前）落地 | `PP = 1` 无需双算批重叠；避免在 Stage 1 无收益地改动 [`engine_metrics.inl`](../../src/runtime/engine/engine_metrics.inl#L27) 计时归属与 [`qwen3_5/session/commit.cpp`](../../src/models/qwen3_5/session/commit.cpp#L476) 的 `normalize → pause` 链 |
| **3. 共享 `Frontend` 零新增信号量** | 直接在主控层共享唯一 `Frontend`（保留 `frontend.identity()` 校验），**不额外包外层信号量** | [`media_cache.cpp`](../../src/models/common/frontend/media_cache.cpp#L197) 已有 `MediaPreparationGate` + 有界 `TaskPool`，内置 XGrammar（[`grammar_compiler.cc`](../../third_party/xgrammar/cpp/grammar_compiler.cc#L1234)）已有 2-permit 编译信号量 |
| **4. 平台事实下沉至 `src/core/`** | 将 [`measurement.cpp`](../../src/runtime/executor/measurement.cpp#L32) 直接调用的 CUDA Profiler API 与 [`glm5next/startup.cpp`](../../src/models/glm5next/startup.cpp#L127) 的编译期共享内存分支收口至 `core/` 函数；`WorkerLoop` 解耦具体 `WorkerConnection` 以支持纯 CPU + 真实 Socket 单元测试 | 消除控制层平台泄漏，让 Tier 0 CTest 直接驱动真实生产分发状态机 |

---

## 二、分模块源码级详细设计（W1 → W6）

### 2.1 W1：四层拓扑、全局↔局部坐标双射与计划摘要

**涉及文件**：[`include/ninfer/deployment.h`](../../include/ninfer/deployment.h)、[`src/distributed/deployment.cpp`](../../src/distributed/deployment.cpp)、[`src/distributed/topology.h`](../../src/distributed/topology.h)、[`src/distributed/topology.cpp`](../../src/distributed/topology.cpp)、[`src/distributed/wire.cpp`](../../src/distributed/wire.cpp)

1. **部署校验与规范化（`validate_deployment`）**
   - `DeploymentOptions` 新增 `bool allow_shared_device = false;`：放宽同 `(node, device)` 唯一性检查时**仍保留 `rank.device >= 0` 非负校验**；
   - 算术乘积 `dp * pp * tp` 先做防溢出宽整型校验再与 `ranks.size()` 比对；对输入的 `DeploymentRank` 列表按全局 `rank` 规范化排序，不依赖 TOML 书写顺序；
   - 合法组合校验：`tp % adp == 0`，且满足「`pp == 1` 下 `tp % ep == 0` 或 `(ep % tp == 0 && (dp * tp) % ep == 0)`」或「`dp == 1 && pp > 1 && tp % ep == 0`」。

2. **`ReplicaTopology` 权威全局↔局部双射（根治非零基址副本越界）**
   - 在 `dp2 + tp2` 中，副本 1 的 wire peer 为全局 rank `{2, 3}`，但副本内 `DistributedExecutor` 观测/进度数组、[`glm5next/parallel/preparation.cpp`](../../src/models/glm5next/parallel/preparation.cpp#L17) 目录分片与 [`glm5next/wire.cpp`](../../src/models/glm5next/wire.cpp#L213) 动作筛选均需要局部连续下标 `{0, 1}`。
   - 在 `ExecutionTopology` 中废除单一 `output_rank`，引入 `std::vector<ReplicaTopology> replicas`：
     ```cpp
     struct ReplicaTopology {
         std::uint32_t replica_id        = 0;
         std::uint32_t failure_domain_id = 0; // ep <= tp 时为 replica_id；ep > tp 时为 replica_id / (ep / tp)
         RankId result_owner             = 0; // 全局 RankId（stage_ranks[pp - 1][0]）
         std::vector<RankId> ranks;           // 局部下标 [0, pp*tp) -> 全局 RankId
         std::vector<std::vector<RankId>> stage_ranks; // [pp][tp] 全局 RankId

         [[nodiscard]] std::uint32_t local_index(RankId global_rank) const;
         [[nodiscard]] RankId global_rank(std::uint32_t local_index) const;
     };
     ```
   - **边界转换铁律**：`Transport`、进程日志、`StartupEvent::origin` 与公开 `RankSnapshot::rank` 始终使用**全局 `RankId`**；`DistributedExecutor` 内部成员状态数组（`observations_`、`progress_`、`retired`、`consumed`、`context_results`）、`Executor::observe_retirement()` 与模型副本内数组统一按 **`local_index`（`0 .. ranks.size()-1`）** 寻址，在收发边界经 `ReplicaTopology` 单点转换。

3. **模型计划后计算 `TopologyPlanDigest` 与 Worker 真实核验**
   - `topology.plan` 在模型产出全部分片计划与有序通信角色后统一计算（纳入 `artifact_id`、五维并行度、全体 rank 物理位置与坐标、通信组及角色、各副本 `result_owner` 与本地权重切片配方）；可变窄图宽度与容量重试使用独立的 `attempt` 序号，不混入静态 `plan`；
   - Worker 在收到 `Prepare` 后不再只比对两个主控传来的整数（[`worker.cpp`](../../src/runtime/engine/worker.cpp#L71)），而是打开本地 `.ninfer` 产物核验真实 `artifact_id` 并复算本 rank 分片摘要与已安装的 `topology.plan` 一致。

---

### 2.2 W2：单一属主进程监督、端到端信用与安全取消

**涉及文件**：[`src/distributed/launch.h`](../../src/distributed/launch.h)、[`src/distributed/launch.cpp`](../../src/distributed/launch.cpp)、[`src/distributed/transport.h`](../../src/distributed/transport.h)、[`src/distributed/transport.cpp`](../../src/distributed/transport.cpp)、[`src/distributed/bootstrap.h`](../../src/distributed/bootstrap.h)、[`src/distributed/bootstrap.cpp`](../../src/distributed/bootstrap.cpp)、[`src/core/cpu/system.h`](../../src/core/cpu/system.h)、[`src/core/cpu/system.cpp`](../../src/core/cpu/system.cpp)、[`src/runtime/executor/protocol.h`](../../src/runtime/executor/protocol.h)、[`src/runtime/executor/protocol.cpp`](../../src/runtime/executor/protocol.cpp)、[`src/runtime/executor/input.h`](../../src/runtime/executor/input.h)、[`src/runtime/executor/input.cpp`](../../src/runtime/executor/input.cpp)、[`src/runtime/executor/distributed_executor.h`](../../src/runtime/executor/distributed_executor.h)、[`src/runtime/executor/worker_loop.h`](../../src/runtime/executor/worker_loop.h)、[`src/runtime/engine/worker.cpp`](../../src/runtime/engine/worker.cpp)

1. **`NodeProcessSupervisor` 唯一进程属主与远端控制租约**
   - 由常驻 `NodeProcessSupervisor` 线程**唯一拥有**全部 `WorkerProcess` 实例：串行处理拉起请求、后台轮询排空所有健康与销毁中域的 stdout/stderr 管道（维护有界日志尾缓冲）、按值上报 `(RankId, exit_status, log_tail)`，并独占执行按失败域 `terminate_domain()` 与 `waitpid()`（彻底消除 [`WorkerProcess`](../../src/distributed/launch.cpp#L142) 无锁状态竞争）；
   - 本地 Worker 经该常驻线程 `posix_spawn` 拉起，在初始化设备与任何辅助线程前设置 `prctl(PR_SET_PDEATHSIG, SIGTERM)` 并核验 `getppid() == expected_launcher_pid`；
   - **远端防孤儿双向保活**：弥补单靠客户端 `ServerAliveInterval` 无法覆盖本地主控暴毙后孤立 `ssh` 会话或静默网络分区的盲区——主控 `Transport` 周期性发送轻量心跳（覆盖长启动物化阶段与运行空闲期），Worker 侧监控控制连接最后活跃时间：一旦超时未收包或遇非 `Shutdown` 的连接关闭，主线程立即 `comm.abort()` 并退出进程。

2. **CPU 亲和性在辅助线程创建前绑定**
   - 修复 [`system.cpp`](../../src/core/cpu/system.cpp#L65)：开启 `HWLOC_TYPE_FILTER_KEEP_IMPORTANT` 发现 PCIe/NUMA 拓扑，且不因调用线程临时掩码丢失整机可用核信息；为同节点全部竞争 Worker 统一规划基于 GPU NUMA 局部性的不相交物理核集；
   - 调整 [`WorkerConnection`](../../src/distributed/bootstrap.cpp#L423) 启动顺序：Worker 在 `read_worker_bootstrap()` 拿到分配核集后**先完成线程亲和性绑定，再启动 `Transport` I/O 线程与初始化 `DeviceContext`**，避免辅助线程继承未绑核的默认掩码。

3. **`ClusterGroup` 连接直挂与完整协议态移交**
   - Worker `Join` 握手时按 `r = topology.rank(rank).dp.rank` 直接挂入 `replica_transports_[r]`；
   - Phases 1/2 启动期间，由主启动编排跨全部相关 `replica_transports_` 同步广播每一个涉及集合通信的准备阶段再等待全员回复；
   - 全员 `Ready` 后，以单消费者原子移交结构 `ReplicaTransportHandoff { unique_ptr<Transport>, deque<IncomingMessage> buffered_inbox, uint64_t epoch, ReplicaTopology mapping, optional<string> pending_error }` 移交对应 `ReplicaEngine` 的 `DistributedExecutor`（保留完整 `Failed` 帧优先于裸 EOF 的诊断优先级）；
   - `StartupObserver` 由主控在整个启动与推测校准期间集中持有并在锁外串行回调（携带 `rank / replica_id / attempt`），校准全部结束开放准入前清空启动回调，消除 [`engine_core.h`](../../src/runtime/engine/engine_core.h#L110) 异步线程悬垂引用风险。

4. **端到端信用闭环与流式媒体窗口（[`transport.cpp`](../../src/distributed/transport.cpp) + [`input.cpp`](../../src/runtime/executor/input.cpp)）**
   - 新增消费者侧 `poll_fd()` / `wait_readable()`（覆盖入站可读、充足发送额度恢复、断连/错误）；`read_peer()` 有界批量读空（如单轮至多 16 帧）兼顾吞吐与公平；修复 `peer.blocked` 下超总容量单帧死等与 `POLLHUP / POLLERR` 空转，在有界时间内收完尾帧或报错关闭；
   - **流式视觉输入窗口控制**：修复单张视觉票据连续发大量 `256 KiB` chunk 打爆慢速 Worker 普通接收额度的问题——为 `InputTransfers` 建立显式 per-peer 字节与帧窗口，Worker 每将 chunk 拷入固定 Pinned 接收槽即归还额度；`InputTransfers::begin()` 改为异常安全原子注册，严格保证 `InputRelease(old)` 先于复用同一接收槽的 `InputBegin(new)` 发出，并按冻结票据序发射 `Submit`（禁止后提交但无媒体的票据越过先提交待媒体的同流票据）；
   - **双向控制预留与遥测解耦**：按有界票据数精确预留双向 `Submit / Supply / Commit / Discard / InputReady / InputRelease / DeviceRetired / Measurement / Failed` 额度；非请求触发的 `Observation` 与 `Progress` 独立限额并就地覆盖合并；普通 `Submit` 发送遇满返回背压重试，不再在 `progress_inputs()` 直接炸毁执行器。

5. **参数化 `WorkerLoop`、按事件应答集与 `BeforeSupply / AfterSupply` 安全取消**
   - 将 [`WorkerLoop`](../../src/runtime/executor/worker_loop.h#L15) 的连接类型模板化（不强绑构造 `DeviceContext` 的具体 `WorkerConnection`），使 `S1.2` 单测用真实 `Transport` + CPU 测试 `Program` 直接验证生产 `WorkerLoop`；
   - `protocol::Scope` 拆为 `{ participants, decision_owners, result_owner }`（`decision_owners ⊆ participants`）：Compute 仅 `result_owner` 发 `DecisionNeeded / ResultReady`、仅 `decision_owners` 收 `Supply` 并回 `DecisionConsumed`；**Context 保持全体 `participants` 回 `ResultReady` 归并传输观测**；未参与某局部票据的 rank 不收空包，也不计入序号等待；
   - `WorkerLoop` 持续有界排空入站包：新 `Submit` 按序入 `submit_queue_`（计算步不越过前置 `Bind`）；续行挂载至匹配完整 `(epoch, ticket_id, frame, generation)` 的票据并在到达合法本地阶段时推进；
   - **`Discard` 精确定义（以主控 `Supply` 原子扇出成功为分界）**：
     - `DistributedExecutor::supply()` 改为**在 `send()` 原子扇出成功后再置 `flight.supplied = true`**；
     - `BeforeSupply`（`!flight.supplied`）：携带 `BeforeSupply` 发出 `Discard`；Worker 对完整票据（含仍在 `submit_queue_` 中的票据）锁存取消，按序跑完含集合通信的前向公共前缀，在等待外部 `Decision`（或无决策工作的 `Result`）处直接退役；
     - `AfterSupply`（`flight.supplied`）：携带 `AfterSupply` 发出 `Discard`；Worker 即便先收到 `Discard` 也必须应用该 `Supply` 并跑完采样集合通信至本地 `Result`，随后跳过 `Commit` 直接以该 `Discard` 命令号回报 `DeviceRetired`；
   - 执行器级测量边界保留“执行器先排空在途票据与 `InputRelease`、再发有序 per-peer `Measurement` 栅栏收集 `query_id` 应答”的简洁不变量。

---

### 2.3 W3：目标绑定三态权重、全边界原地重试与集群内存求解

**涉及文件**：[`src/models/common/parallel_weights.h`](../../src/models/common/parallel_weights.h)、[`src/models/common/parallel_weights.cpp`](../../src/models/common/parallel_weights.cpp)、[`src/runtime/executor/preparation.h`](../../src/runtime/executor/preparation.h)、[`src/runtime/executor/preparation.cpp`](../../src/runtime/executor/preparation.cpp)、[`src/runtime/executor/rank_preparation.h`](../../src/runtime/executor/rank_preparation.h)、[`src/runtime/engine/context_capacity.h`](../../src/runtime/engine/context_capacity.h)、[`src/runtime/engine/context_capacity.cpp`](../../src/runtime/engine/context_capacity.cpp)、[`src/runtime/engine/distributed_startup.inl`](../../src/runtime/engine/distributed_startup.inl)

1. **目标绑定级三态权重与绑定器清理**
   - 移除 [`parallel_weights.cpp`](../../src/models/common/parallel_weights.cpp#L113-L121) 隐式 `whole_tensor` 兜底；按模型 `ModelDefinition` 实际注册的参数集合，强校验每个目标 `WeightId`（来自 `whole`、`assemblies[].bindings[].parameter`、`experts[].parameters`、`omitted`）被唯一分类；
   - 同步修改 `select_weights()` 与 [`bind_selected_weights()`](../../src/models/common/parallel_weights.cpp#L149)：对 `Omitted` 参数显式清空 `weight.view = {}` 并跳过其激活校准切片请求；保留 `Qwen4Exp` 独立 PLE 请求路径；
   - 128 对齐词表切分工具同时定义小词表（`vocabulary < 128 * tp`）回退紧凑均分规则，并统一作用于所有输出头（Target / MTP / Draft）、logits 视图、采样词表偏移与 `score_tokens` 归约偏移。

2. **覆盖全部可恢复失败边界的原地零重启启动**
   - Phase 1 先完成 `PrepareCommunication` 连接预热，再按节点预估峰值门控 `MaterializeWeights` 一次性加载权重；
   - Phase 2 修复 [`rank_preparation.h`](../../src/runtime/executor/rank_preparation.h#L208-L255) 中 `entered` 置位前探针/分配抛 `DeviceAllocationError` 直接炸掉整组的缺陷：所有可恢复分配阶段（`PrepareProbe`、`PrepareRebuildProbe`、`Allocate`、`PrepareExecution`、`CheckMemory`）均在集合安全边界捕获 `DeviceAllocationError` 转为结构化失败应答返回主控；主控对已成功分配的对端下发 `Release` 并调用主控目录 `release` 回调后，在同一常驻进程与权重上发起下一次 `attempt`（含窄图与全宽基线实测容量对比及显式容量超限直接拒绝）。

3. **单一物理采样时刻的集群内存求解器**
   - 在全部 Worker 完成通信预热与权重物化后**统一采样一次**物理内存快照（此时同 GPU 多进程权重与通信显存已真实落账，绝不再二次扣减 `weight_bytes`）；同物理 GPU 上的 Graph 重建探针串行执行，防止并发探针互相放大保留增量；
   - 节点级 Host / UMA 求解：将**全部 `dp` 个副本的主控 `ContextDirectory` 元数据开销显式记在 `coordinator_node` 账上**（解决跨双机 Spark `dp2` 时远端副本无本地 Worker 导致主控目录漏计费的漏洞），共享 `Frontend` 仅计 `media_live_bytes + grammar_cache_bytes`；`world_size > 1` 时固定 `host_capacity_bytes = 0` 且禁用 `ExpertCache` Offload。

---

### 2.4 W4：双层许可、`DpRouter` 惰性预路由与 `EngineCoordinator`

**涉及文件**：`src/runtime/engine/replica_engine.h`（由 [`engine_core.h`](../../src/runtime/engine/engine_core.h) 重构）、`src/runtime/engine/dp_router.h/.cpp`、`src/runtime/engine/engine_coordinator.h`、[`src/runtime/engine/engine.cpp`](../../src/runtime/engine/engine.cpp)、[`src/runtime/engine/context_cache/prefix_index.h`](../../src/runtime/engine/context_cache/prefix_index.h)、[`src/runtime/engine/context_cache/resource_manager.h`](../../src/runtime/engine/context_cache/resource_manager.h)

1. **双层许可（`Outstanding` vs `Residency`）与抢占优先移交**
   - `GlobalOutstandingGate`（上限 `C + max_pending_requests`）：调用线程 `make_output_session()` 前获取并分配全局 `(request_id, publication_order)`，跨 `paused_` 持有，终态快照发布且句柄释放后归还；
   - `GlobalResidencyGate`（上限 `C`）：副本提供可绑定空槽、主控授予许可时即计入（覆盖派发在途与 `materializing_`）；暂停腾空物理槽时，若存在本地等待恢复的目标请求则**直接将许可原地移交给该恢复请求**（若仅给已在槽内的 `progress_owner` 腾空间则归还全局池）；所有可运行 `paused_` 恢复请求在门控处严格优先于中心队列新请求；副本线程绝不在门控上阻塞等待自身需要退役才能腾出的许可。

2. **可移植 `PreparedPrompt`、惰性 `key_at(frontier)` 预路由与无损退回**
   - 改造 `PreparedPrompt` 以 `shared_ptr` 持有不可变可移植输入并暴露模型权威的惰性 `key_at(frontier)`（内部复用稀疏 `PrefixFingerprints` 增量计算），使 `Session::plan_request()` 不破坏原始可移植输入；
   - `PrefixIndex` 插入时随句柄保存 `(PrefixKey, recoverable_frontier, placement)`，在增删清改时向 `DpRouter` 顺序发布带 `epoch` 的引用计数增量；
   - `DpRouter` 维护中心等待队列（沿用现有 FIFO + 有界 bypass 计数策略）：按所属目录导出的去重活跃/预留/可驱逐统计与命中前沿为有空槽健康副本打分；若新请求在目标副本穷尽本地来源与 placement 后 `assess_binding()` 仍不可行，释放本地临时引用与驻留许可，**保留原可移植 `PreparedPrompt`、`request_id` 与 `pending_deadline` 退回 `DpRouter`** 在有效资源版本更新后重派。

3. **`EngineCoordinator` 幂等失败域隔离与提交切口测量栅栏**
   - **两步幂等失败隔离**：副本异常时先原子标记该失败域不可路由（作废其 `epoch` Hint 与槽位报价，唤醒所有等待者；若为最后一个健康域则令中心排队请求以 `Unavailable` 结束）；再由该副本线程停收/中止通信、监督线程回收该域进程、清理本副本在途与暂停请求（保留已完成/已报错句柄状态供调用方 `wait()` 读取）；
   - **提交切口（Submission Cut）测量栅栏**：在 ID 分配锁下记录 `cut_request_id`，阻塞后续新提交（遵守其 `pending_deadline`），**继续正常派发、恢复、执行并结清全部 `id <= cut_request_id` 的在途请求**；待主控确认切口前请求全部终态发布、且冻结的健康参与副本全员确认完全静默（含延迟 Session 释放与 `InputRelease`）后，并发执行 `executor->measure()` 并汇总快照（中途若有参与副本崩溃，立即取消幸存者半开测量窗并报错唤醒等待提交者）；
   - **`score_tokens()`**：保持全引擎单一互斥串行边界直至评分结果交付，收到 `request_stop()` 时允许已接纳的评分跑完。

---

### 2.5 W5：`Qwen3.5` 完整 Worker 契约与 `GLM5Next` 局部几何解耦 DP

**涉及文件**：[`src/models/qwen3_5/program/program.h`](../../src/models/qwen3_5/program/program.h)、`src/models/qwen3_5/program/lifecycle.cpp`、`src/models/qwen3_5/wire.h/.cpp`、`src/models/qwen3_5/parallel/preparation.h/.cpp`、[`src/models/glm5next/parallel/plan.cpp`](../../src/models/glm5next/parallel/plan.cpp)、[`src/models/glm5next/parallel/rank_preparation.cpp`](../../src/models/glm5next/parallel/rank_preparation.cpp)、[`src/models/glm5next/parallel/startup.cpp`](../../src/models/glm5next/parallel/startup.cpp)、[`src/runtime/engine/worker.cpp`](../../src/runtime/engine/worker.cpp)、[`src/runtime/engine/startup.cpp`](../../src/runtime/engine/startup.cpp)

1. **补齐 `Qwen3.5` 真实分布式 Worker 契约（不漏任何工作变体）**
   - 在 [`qwen3_5::RankProgram`](../../src/models/qwen3_5/program/program.h#L21) 上补齐 `can_discard(ExecutionTicket)` 接口与视觉输入固定接收槽访问器；
   - 实现无进程指针的 `Qwen3.5` 准备描述编码（供主控重建 `SessionPlan`）与完整有界 `wire.h/.cpp`：覆盖全部工作与结果类型——普通/自适应推测 `DecodeWork`（含 `RoundPlan`、几何与窄化输入）、打包 `PrefillWork`（含 ragged 结果）、`NormalizeWork`、`ScoreWork`、`ContextWork`（含绑定与驻留释放）及视觉输入流；
   - 纯 DP（`tp = 1, ep = 1, pp = 1`）复用本地常驻权重加载路径（禁用多卡 `ExpertCache` Offload），将实际定义的参数集显式归入三态分类，注册进 `worker.cpp` 与 `startup.cpp`。

2. **`GLM5Next` 按副本局部几何构建解耦多副本（`dp >= 1, tp <= 2`）**
   - 彻底改写 [`glm5next/parallel/plan.cpp`](../../src/models/glm5next/parallel/plan.cpp#L128)、[`rank_preparation.cpp`](../../src/models/glm5next/parallel/rank_preparation.cpp#L18) 与 [`parallel/startup.cpp`](../../src/models/glm5next/parallel/startup.cpp#L18)：
     - 每个分片按自身 **TP 坐标（`tp.rank, tp.size`）** 而非全局 `ranks.size()` 切分；
     - 每个 `tp = 2` 副本生成独立的 `"tensor"` 与 `"context"`（`PeerExchange`）通信组，Worker 按**包含自身全局 `RankId` 的组成员关系**查找对应角色组；`tp = 1` 副本直接走无通信组本地执行路径，不强求 `"tensor"` 角色；
     - 批次规划、CUDA Graph 桶目录、采样缓冲、Context 布局、wire 预算与解码批次直方图统一使用**副本局部槽位数 `local_lanes = ceil(C / dp)` 与局部属主数 `adp`**；成本模型签名直接采用该副本真实局部并行几何与选定通信后端。

---

### 2.6 W6：配置沿用、不可变描述与实时多副本快照

**涉及文件**：[`src/product/config/file.cpp`](../../src/product/config/file.cpp#L81)、[`src/product/config/configuration.cpp`](../../src/product/config/configuration.cpp#L332)、[`include/ninfer/observability/snapshot.h`](../../include/ninfer/observability/snapshot.h)、[`src/serve/request_log.cpp`](../../src/serve/request_log.cpp#L669)、[`src/serve/observations.cpp`](../../src/serve/observations.cpp)

1. **保持 `[distributed.parallel]` 既有 TOML 拼写**：该表已支持解析和渲染 `tp/ep/dp/pp/adp` 五维参数，仅增量补齐 `allow_shared_device`、单机纯 DP 确定性本地展开、以及 `context_cache.max_cached_checkpoints_per_replica`（含配置溯源与打印）。
2. **`EngineDescription` 不可变 vs `EngineSnapshot` 实时可用容量**：
   - `Engine::description()` 保持构造期一次性生成的不可变对象（汇总启动期各副本受限容量估计之和，绝不在运行期修改以免数据竞争）；
   - `EngineSnapshot` 新增实时健康可用容量字段、按物理 `(node, device)` / `NodeId` 去重的硬件资源视图、以及 `replicas[]` 切片（故障副本保留累计历史计数，仅将其可用容量置零；所有副本局部 `placement_id`、在飞 `ticket` 与 `progress_owner` 均附带 `replica_id` 消歧）；
   - `replica_id`（未分配时为 `std::nullopt`）写入 `RequestSnapshot`、终态 `GenerationResult` 与 JSONL 请求日志；严格保持“先移除调度成员并发布对应快照，再唤醒调用方 `wait()`”的公开可见序。

---

## 三、编译常绿的分批次实施计划（S1.1 → S1.6）

每个批次同时迁移受该批次接口变更影响的现有调用点（含 `Qwen4Exp` 既有 `1-rank / TP2` 调用与 `bench/distributed/collective.cpp`），保证**每批结束时工作树编译全绿、测试全过**：

| 批次 | 核心改造与同步迁移边界 | 验证证据（CPU CTest + 独立分布式夹具） |
| :--- | :--- | :--- |
| **S1.1**<br>拓扑双射与三态权重绑定 | ① 四层 `ClusterTopology`、`ReplicaTopology` 全局↔局部双射、乱序 `ranks` 规范化、`allow_shared_device` 与模型后 `TopologyPlanDigest`（同步更新 `distributed/wire.cpp` 与 `bench/distributed/collective.cpp`）；<br>② 目标 `WeightId` 三态 `select_weights` + `bind_selected_weights` 清空 `Omitted` 视图与校准请求 + 128 对齐词表切分（同步迁移 `GLM5Next` 与 `Qwen4Exp` 现有权重计划及 `parallel/wire.cpp`，保留 `Qwen4Exp` PLE 请求） | 新增并注册 CPU CTest：`test_cluster_topology`（1~16 卡合法/非法组合、乱序全局 ID、非零起始副本双射）、`test_parallel_weights_tristate`（漏分类报错、多切片/别名、`Omitted` 视图与校准清空、非整除词表边界真实头矩阵切片重建核验） |
| **S1.2**<br>单一监督者、信用闭环与无死锁 `WorkerLoop` | ① `NodeProcessSupervisor` 单一属主拉起/管道排空/按域回收 + 远端控制心跳租约 + `hwloc` 全机核切分与连接前绑核；<br>② `Transport` 四项修复 + `poll_fd()` + 流式视觉 `InputTransfers` 字节/帧窗口与有序释放 + 遥测覆盖合并；<br>③ `protocol::Scope` 按事件应答集 + 参数化 `WorkerLoop` 命令序解耦 + `BeforeSupply / AfterSupply` 集合安全 `Discard` + `ClusterGroup` 直挂 `Transport` 与完整协议态移交（同步更新 `DistributedExecutor` 局部下标转换、`Qwen4Exp` 子集 `Scope` 与测试 codec） | 升级 `test_protocol`（CPU CTest）并用真实 Unix Socket + CPU 测试 `Program` 驱动生产 `WorkerLoop`：验证双帧（`compute_frames=2`）逆序退役、慢速非主卡、排队态/`BeforeSupply`/`AfterSupply` 取消、保留 RX 载荷、视觉槽释放复用、无后续提交的空闲断连及杀主零孤儿 |
| **S1.3**<br>全边界原地重试与统一采样时刻内存求解 | ① 先通信预热后按节点门控权重物化、单一物理时刻重采样、同卡探针串行；<br>② 覆盖 `entered` 前后全可恢复阶段的集合安全 `Release` 重试；<br>③ 物理 GPU 共享分摊 + 主控节点汇总全部 `dp` 份 `ContextDirectory`（含纯远端副本）+ 不重复计费 `media_live_bytes`（同步迁移 `GLM5Next` / `Qwen4Exp` 准备回调与 `test_model_preparation`） | 升级 `test_context_capacity`（CPU CTest）：验证纯远端副本主控目录计费、同 GPU 双 rank 分摊、UMA 防双计、单侧探针失败协同 `Release` 重试且权重零重读 |
| **S1.4**<br>多副本协调器、双层许可与预路由 | ① 共享单一 `Frontend` + 可移植 `PreparedPrompt` 惰性 `key_at(frontier)`；<br>② 单计算槽 `ReplicaEngine` + `PrefixIndex` 增量发布 + 目录级容量导出与可选检查点上限；<br>③ `DpRouter`（双层许可、抢占许可原地移交、无损退回重选）+ `EngineCoordinator`（两步幂等失败隔离、提交切口静默测量栅栏、不可变 `description` + 实时 `EngineSnapshot::replicas[]`、全引擎串行 `score_tokens`） | 新增 CPU CTest `test_dp_router` 与 `test_multi_replica_engine`：覆盖 `C < dp`、非整除 `C`、全局满载 `C` 下抢占恢复零饥饿、慢消费/丢弃句柄、过期 Hint、容量不足退回、带切口前积压/暂停请求的测量栅栏、单副本崩溃隔离 |
| **S1.5**<br>`Qwen3.5` Worker 契约、`GLM5Next` 解耦 DP 与双构建收口 | ① 补齐 `Qwen3.5` `can_discard`、视觉接收槽、无地址描述与全变体 `wire.cpp` 并注册 Worker/Startup；<br>② `GLM5Next` 局部几何解耦多副本（`tp <= 2, dp >= 1`）；<br>③ `[distributed.parallel]` 增量配置、监控与终态 JSONL `replica_id` 落地；显式区分注册 CPU CTest 与 GPU 测试 | 分别验证 `NINFER_ENABLE_DISTRIBUTED=ON`（含 `ninfer`、`ninfer-serve`、`ninfer-worker` 与分布式测试目标）与 `OFF` 完整编译链接通过，全部 CPU CTest 绿灯 |
| **S1.6**<br>5090 与双机 Spark 实机验收 | 先在 `.10`（5090）采集并封存固定负载单卡基准数据，再按第四节矩阵经 `gpu-run` 跑完 Tier 1 与 Tier 2 全项实测 | 达成第四节全部硬验收标准，清理所有临时运行目录 |

---

## 四、Stage 1 验证矩阵与出阶段硬标准

所有实机 GPU 操作严格通过 `gpu-run` 独占锁执行，且将仿真证据与实机证据严格区分：

1. **单卡数值验证与固定负载零退化（Tier 1：`.10` RTX 5090）**
   - 精确核对分片边界、稀疏掩码的允许集合和评分目标位置。浮点结果按对应 Op 和表示的数值合同验收。
   - 在声明的固定模型、输入、并发与重复轮数标准下，对比重构前实测基线，单卡 `LocalExecutor` 的 TTFT 与 Decode TPOT 无退化。

2. **多副本全变体 Worker 功能与共享显存核算（Tier 1：`.10` 5090 多进程）**
   - 在 `allow_shared_device = true, dp = 2` 下通过真实 `ninfer-worker` 进程跑通 `Qwen3.5-MoE` 全路径：**文本生成、XGrammar 约束、多模态视觉输入、推测解码、`Normalize → Pause → Restore` 抢占恢复与 `score_tokens()`**；
   - 核验同卡双进程分配的总显存严格受限于单张 5090 物理净余量，长前缀请求稳定命中缓存副本。

3. **跨节点解耦 DP/TP、故障隔离与防孤儿卫生（Tier 2：`.50 + .51` 双机 Spark）**
   - 实机跑通跨节点 `Qwen3.5 dp2`（覆盖纯远端副本目录计入 `.50` UMA 预算）与 `GLM5Next` 解耦并行（含非零基址副本 `dp2` 或 `tp2+adp2`），验证带在途请求的 `begin_measurement / end_measurement` 正常排空返回且 `EngineSnapshot::replicas[]` 与终态 `request_jsonl` 的 `replica_id` 准确；
   - **故障隔离实测**：并发压测中途强杀副本 1 的 Worker，验证副本 1 在途请求精确报错一次、副本 0 零中断持续吐字、中心等待请求自动改道副本 0；
   - **空闲与防孤儿实测**：引擎空闲静置无后续提交时主控与 Worker CPU 均 `< 1%`；测试空闲期杀 Worker 能立即被主控捕获报错，测试**启动物化期**与**空闲服务期**分别对主控发 `SIGKILL`，本地与远端 Spark Worker 均在租约上限内 100% 自动退出，零残留进程。
