# Attention DP 放置与上下文管理

本文是待实施设计，确定请求放置、前缀恢复、资源预留和跨 owner 转换的合同。
首个消费者是 GLM5Next 双 Spark：一个执行实例，DP1、MLA/Indexer ADP2、KDA TP2、
routed experts EP2。模块分工、State 字段和页布局由 [GLM 并行设计](glm5next/parallel.md)维护。

共享机制沿用 [Engine 所有权](../maintainer/engine-architecture.md)、
[资源与缓存策略](../maintainer/resource-scheduling-and-context-cache.md)及
[统一 Context 存储](../maintainer/paged-kv-cache.md)。本文只定义这些合同需要增加的
能力；不表示当前已支持 attention DP 或跨 rank Context 恢复。

## 1. 并行语义与放置策略

### 1.1 DP 与 ADP

**DP 表示完整模型执行副本，ADP 表示一个执行实例内部的 attention 请求分工。**
可以把 ADP 直观理解为将 DP 的按请求分工用于 attention，移除这部分跨 owner 的 TP：
一个 owner 计算该请求的完整 attention，其他模块仍采用模型指定的 TP/EP。
`ADP = DP − TP` 描述这种执行关系；配置中的并行度分别声明，不做数值减法。
ADP 作用于哪些模块由模型定义：GLM 是 MLA/Indexer，KDA 仍按 heads TP。

| 合同 | DP | ADP |
|---|---|---|
| 请求分配 | 选择一个完整执行副本 | 在已选实例内选择 attention owner |
| 推进与组批 | 各副本独立推进 prefill、decode、MTP | 同一实例统一组批，生成 owner-local 行映射 |
| 状态与资源 | 每个副本拥有独立 Session、ContextDirectory 和执行资源 | 一个目录管理实例内各 rank 的 State/history |
| 计算通信 | 独立副本之间没有模型层 collective | TP/EP 及 attention 与其他模块的交接仍可通信 |
| 前缀亲和 | 根据副本公开摘要选择接收方 | 根据实例目录中的完整恢复资格选择 placement |

完整副本可以跨多台设备。部署计划列出每个副本的 rank 成员，模型计划决定副本内部
的 TP/EP/ADP 分工；这些角色可以复用同一组 ranks，设备数由实际拓扑确定。

| 部署意图 | DP | 每副本 TP / EP / ADP | 总设备数 |
|---|---:|---|---:|
| GLM 双 Spark，本设计首个目标 | 1 | 2 / 2 / 2 | 2 |
| GLM 四 Spark，两个独立双机副本 | 2 | 2 / 2 / 2 | 4 |
| Flash Next 双 Spark，各自完整运行 | 2 | 1 / 1 / 1 | 2 |

后两项表达后续组合方式，不表示当前已实现。Flash Next 的完整 DP 可以消除副本之间
的模型通信，代价是每侧保存完整权重与执行资源；适合按总并发比较聚合吞吐。其 QSA
也可独立设计 ADP 路线，继续保留其他模块的 TP/EP。当前 QSA 的 KV 已按 heads TP
切分，因此 ADP 的容量收益需按实际布局计算，不能照搬 MLA 去除重复 latent 的收益。

配置合同保留 `ParallelOptions::dp` 表示完整副本数，新增独立的 `adp`，默认 1；
rank 身份保留 `dp` 坐标并增加 `adp` 坐标。配置层只解析，拓扑层校验成员和坐标，
模型计划声明支持的内部组合。本文不复用现有 `dp` 字段承载 ADP。

### 1.2 两级路由与实例边界

```text
公共 Engine / 服务
  └─ DP 副本选择
       ├─ 执行实例 0：Scheduler → Session / ContextDirectory → Executor
       │                └─ 实例内 ADP placement 选择
       └─ 执行实例 1：Scheduler → Session / ContextDirectory → Executor
                        └─ 实例内 ADP placement 选择
```

副本路由消费负载、缓存亲和和可接收状态摘要，实例内部负责准确准入、预留和执行。
两层可以共享纯排序方法，各自持有计数与生命周期；副本路由不取得内部页或 State
的操作权。缓存摘要用于路由提示，真正恢复仍由接收实例检查。

`ReplicaId` 定位完整副本，`PlacementId` 在该副本的 Session 内有效；跨实例观测使用
二者组合。Checkpoint、State、History 和 reservation 仍归各自目录，不共享物理
身份。本文的 owner 转换、原子预留、Context 通信和退役都限定在一个执行实例内，
不跨 DP 副本联合提交。跨完整副本迁移另有传输与请求交接合同，不由 owner 转换代替。

本设计的实施目标为 DP1 / ADP2，完整 DP 的副本路由后续接入；
以下负载公式、来源选择和容量压力策略均作用于一个执行实例。

### 1.3 选定的放置策略

放置使用三个事实：**完整资源可行性、本地可恢复前缀、已分配的 token 负载**。
容量判断准确执行，负载使用简单计数，复制与重算只比较当前恢复工作。

- 先检查资源，再在负载接近的 placements 中优先本地前缀。
- 普通执行保持 owner；新绑定、暂停恢复和容量压力可以重新选择。
- State 与 history 共用各 rank 的 Context 容量，不给 owner、缓存副本或请求另设池配额。
- 同一语义前缀可以保留不同 placement 的完整 checkpoint。
- 副本由实际请求和正常 checkpoint 保存产生；首版不做后台热点预测与主动复制。
- 不预测最终输出长度、未来若干轮收益、MTP 接受率或逐 kernel 延迟。

参考的是成熟实现中的简单规则：[vLLM 内部 DP 路由](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)
使用当前请求负载；[SGLang DP controller](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/data_parallel_controller.py)
提供 total-requests/total-tokens，分配后立即记账；
[SGLang cache-aware 路由](https://docs.sglang.io/docs/advanced_features/sgl_model_gateway#load-balancing-policies)
在缓存亲和与负载均衡间使用阈值。NInfer 使用目录中真实的恢复资格，采用集中请求调度。

## 2. 必须成立的场景

| 场景 | 行为 |
|---|---|
| 冷启动、两侧空闲 | 依据加入请求后的 token 负载选择，相同条件轮转 |
| 连续到达长 prompt | 绑定成功即计入完整已知 prompt 长度，后续分配能看到该负载 |
| 仅一侧有前缀 | 负载在允许范围内时优先该侧；明显失衡时为另一侧选择恢复方式 |
| 前缀所在侧空间不足 | 检查回收与其他 placement，不在第一个暂不可执行候选上一直等待 |
| 两侧均有相同前缀 | 保留各自恢复实例，按本地命中和负载选择 |
| 多请求共享前缀 | 同侧共享不可变 history；写尾页按需 COW；State 按完整恢复语义管理 |
| 活动请求持续增长 | 逐执行单元预留；短缺时回收、压力转换或暂停/Replay |
| 压力来自两侧都有的 KDA State | 使用真实释放量；换 attention owner 不被视为释放共同 State |
| 部分页、不同 pool 相位 | 分别维护 Main/MTP coverage 和 tail，不把未完成 pool 当成完整历史 |
| MTP 中途暂停或转换 | 在合法提交边界完成 canonical history 后操作；临时 selection/records 不发布 |
| 等待来源被回收或取消 | 逻辑等待引用按既有权限撤销；实际在途读者保持到退役 |
| 转换目的最终能容纳、峰值不能容纳 | 不提交转换；尝试其他来源、回收或 Replay |
| 一个 rank 没有有效 attention 行 | 不读取该侧无效状态，仍执行模型安排的通信序列 |
| 池整理、compact batch、换 owner | 下一执行重新解析 offsets/行映射，旧地址借用保持到最后消费者退役 |
| 两侧都满、请求互相需要对方空间 | 不各持一半预留等待；按现有年龄规则释放工作集保证进展 |
| 单请求超过任一合法 owner 的可容纳范围 | 模型报告无法履行容量合同，不把两侧空闲字节相加当作可用历史 |

## 3. 身份与所有权

### 3.1 身份

| 身份 | 含义 |
|---|---|
| Request/SequenceId | 请求与逻辑序列；暂停、Replay 和转换保持身份 |
| ReplicaId | 完整 DP 执行副本；副本选择层的身份 |
| ResidencyId | 一次执行绑定；重新绑定时换代 |
| ContinuationOwnerToken | 私有续接记录的归属，保留现有含义 |
| PlacementId | 模型 Session 内的上下文放置；不等同于 lane、row 或物理 offset |
| CheckpointHandle | 完整、不可变恢复点，具有确定 placement |
| ExecutionTicket | 一次计算或 Context 操作的完成与退役身份 |

共享 `PlacementId` 是不透明的整数标识。GLM 的两个 placement 分别对应 attention
owner 0/1；模型计划解释它们使用哪些 ranks、State variants 和 history parts。
Runtime 不按 PlacementId 的数值推导 rank、字段或并行拓扑。

一个 checkpoint 在既有 metadata 中增加 placement，并共同持有输入身份、StateId、
Main/MTP HistoryPrefix 及模型前沿。输入语义摘要不包含 placement；同一摘要的
不同实例分别验证完整恢复资格。

GLM Main/MTP 共用 owner。完整恢复点的前沿、conditioning 与 tail 关系沿用
[MTP 设计](glm5next/mtp.md#5-数据归属与-canonical-frontier)。两个独立计算出的相同
文本前缀可以各自恢复，但不能拆取其中一个的 State 与另一个的 history 拼装。

### 3.2 模块分工

| 所有者 | 职责 |
|---|---|
| 模型 ParallelPlan/Session | 声明合法 placement、负载计数输入、恢复资格、State 转换、history coverage、执行及恢复需求 |
| ResourceManager | 前缀候选、缓存亲和、来源选择、等待保护、缓存保留与回收权限 |
| 共享 placement policy | 按第 4 节对公开候选摘要做负载与缓存亲和排序；不访问私有模型状态 |
| EngineCore/Scheduler | 请求生命周期、负载记账时机、年龄/恢复优先、执行成员和进展门控 |
| ContextDirectory | 本执行实例各 rank 的真实分配、metadata、引用、完整预留、借用与发布 |
| Executor/RankProgram | frame、通信及计算提交、固定 backing、地址解析后的执行、退役与失败清理 |
| Ops | 给定 Tensor/views 上的数学与写域；不接收 PlacementId 或 CheckpointHandle |

现有单 placement 模型也通过同一合同提供唯一合法 placement。其 State/history
布局与数学不变；共享模块不按模型名称分叉放置策略。

## 4. 负载计数与 owner 选择

### 4.1 计数

GLM 对一条请求给出的计数为：

```text
w(request) = max(已知完整 prompt token 数, 已提交 Main frontier)
L[p]       = Σ 分配到 placement p 且计入负载的请求 w
N[p]       = 同一组请求的数量
```

计数由模型的公开请求摘要取得。它是 attention 工作量的代理，不是内存字节或预测
耗时。同一前缀被多个活动请求使用时，各请求分别计入负载；真实物理容量仍按共享
引用计算。Main/MTP 不重复加两次 w，可选缓存不计入计算负载。

| 请求状态 | 负载处理 |
|---|---|
| Pending、尚未取得绑定许可 | 不计入任何 placement |
| Binding/Restoring 已取得完整许可 | 立即在目标 placement 计入一次 w |
| Prefill/Replay/Decode/Control | 在绑定 placement 计入；提交前沿超过 prompt 后增量更新 |
| 压力转换已确认且停止旧执行 | 计数从旧 placement 移到待发布的目标；实际资源仍由目录同时计费 |
| Paused，当前在途处理已完成 | 移除计数；保留的设备记录仍占真实容量 |
| Terminal/Cancelled | 停止后续执行并结清当前绑定时移除；在途字节继续等退役回收 |
| 提交前撤销目标许可 | 撤销目标计数；若原请求继续运行则恢复其原计数 |

Engine 每个请求最多拥有一条负载记录，记录 request、目标 placement 和 w。
正在转换的目标计数不提前改变 checkpoint/sequence 的实际 placement；后者在事务
发布时改变。失败或重复完成通知不能再次加减。队列中尚未绑定的请求不会占据虚拟
资源或长期影响某个 rank 的计数。

### 4.2 均衡范围

对请求 w，先从合法且当前可以推进恢复的 placements 构造候选集合 A。可立即预留，
或经过有权限的有限回收／维护即可预留，才进入本次选择；需要等待外部完成事件的
方案另存为等待候选。某个来源不可行，不排除同 placement 的其他来源。
如请求已经计入负载，评估时先从 L 中扣除它自己。对每个候选 p 计算：

```text
S[p] = max_q(L[q] + (q == p ? w : 0))
Smin = min_{p∈A} S[p]
E    = {p∈A | 2*S[p] <= 3*Smin}
```

max 的 q 覆盖全部已声明 placement，包括暂不能接新请求、但仍有活动请求的一侧。
使用整数比较，乘法采用足够宽的中间类型。**1.5 倍**是首版固定的负载容忍度，
只影响选择，不限制任何 rank 的物理容量或请求额度；不增加 CLI/TOML 开关。

在 E 中按以下顺序选择：

1. 目标侧已有完整且可用于本次绑定的 Device checkpoint，复用前缀最长者优先。
2. S[p] 更小者优先。
3. L[p] 更小、随后 N[p] 更小者优先。
4. 其余相同条件使用循环游标；游标只在绑定取得许可后推进。

远端前缀不算本地亲和；仅有 token 摘要而缺失 State/history 的记录也不算命中。
同一 placement 内的来源选择见第 6 节。

例如 L=[8K,0]、新请求 w=8K，两种放置的 S 分别为 16K/8K，选择第二侧；
L=[24K,16K]、w=8K 时 S 为 32K/24K，两侧均在范围内，可优先前缀命中。
无命中时依次按 S、L、N 选择。不同容量仍由 A 中的资源判断处理，不把 token
计数除以容量伪装成计算吞吐。

## 5. 容量筛选与准入

### 5.1 真实许可

Session 为每个来源/目标组合给出完整 `ContextDemand`，包括：

- 新 State、保留来源时的写目的、目标 variant 的所有 rank parts；
- 必要 history 副本、增长页、COW、metadata 和执行 table rows；
- 采用源 checkpoint、释放旧入口等已经具备权限的动作；
- 恢复至原提交前沿并执行第一个真实新单元所需的保护额度。

必要物理需求由 ContextDirectory 评估并联合预留；Executor 同时提供操作所需额度。
固定 frame/records/workspace/staging 已在启动规划中计费，不重复放进 Context 池。
借用中的源和未退役页不属于当前可释放量。

已知完整 prompt 只进入负载计数，不为其所有未来页预留容量；更不预留整个最大输出
预算。后续按执行单元增长。启动仍按各合法 owner 验证单请求上限、最小并发工作集
和模型必要恢复峰值；多请求同时达到各自最大长度不属于启动容量保证。

### 5.2 对评估结果的处理

| 目录结果 | 准入动作 |
|---|---|
| Ready | 可立即尝试联合预留 |
| NeedsMaintenance | 经维护可用；旧地址借用尚未退役时先检查其他候选 |
| NeedsReclaim | 保留为候选，选中后使用现有回收权限进行有限回收尝试 |
| Waiting | 不忙等；本次扫描先尝试其他当前可推进的来源/placement |
| Impossible | 排除此具体需求；其他来源、消费方式或 placement 仍可评估 |

可选缓存占满一侧不代表该侧不能接请求。选中方案可以同步回收有贡献的可选对象；
但 fresh admission 不为了兑现缓存偏好暂停现有活动请求。需要等待传输、租约释放
或活跃请求结束时，先检查其他候选，再保留一个等待来源。

一次扫描在有限的来源/placement 集合中推进；同步回收改变资源时重新评估受影响
候选。不能在首个暂不可用来源上直接终止，也不能因失败后的反复重排回到同一状态。
新的容量、来源或执行完成事件开启下一次评估，普通 decode 不重复扫描未变化的队列。

公平性继续由现有恢复优先、请求年龄和有限越过规则保证。长请求可以在回收与暂停后
获得进展；不存在任何合法单请求执行方案时，由无法履行合同的模型/资源模块报错。

### 5.3 候选生命周期

语义索引只查询一次，Session 将每个来源展开为合法 placement 方案，加上冷启动方案。
候选至少区分 source handle、目标 placement、恢复方式及源消费/保留权限。引用的
handle generation/content epoch 必须在正式预留前重新检查。

评估只读，不 pin 所有候选。等待时仅保留选中的逻辑来源，遵循已有可撤销等待权限；
提交前取得实际 source leases。成功取得完整许可后，记入目标负载并启动绑定。
下一条请求立即看到更新。暂停请求的自有 snapshot 也走这一过程，不再走绕过
placement 选择的特殊绑定分支。

## 6. 目标侧的恢复方式

owner 选定后，GLM 双 Spark 比较三种合法方式：目标本地 checkpoint、远端 checkpoint
转换、冷启动。共享合同继续接收已有部署提供的 Host 恢复来源；支持的存储域与恢复
方式由模型／存储合同声明，不因加入 attention DP 删除原有路径。
Session 提供各方案实际需要重算的 tokens/chunks 和复制范围，Frontend/Runtime
不推导 GLM 的 Main/MTP 位置关系。

```text
T(plan) = 必要本地复制 + 必要跨机复制 + 从该恢复点到准备完成的 prefill/Replay
复制    ≈ 有效 payload bytes / 对应传输路径的有效带宽
重算    ≈ 需要重算的 tokens / 对应模型的粗粒度 prefill 吞吐
```

复用现有 context-cost preset 和成本查询基础，补充带源/目标 rank 的传输路径。
已有 batch/operation 固定成本可继续使用。GLM 的 preset 使用实际整体重算数据，
不能继承 Qwen dense attention 的 pair 系数来代表 sparse MLA/Indexer。

本地候选按既有恢复与重算成本排序。只有远端方案估计总耗时更低时才主动采用；
估计相同时选本地。这里不增加未来 decode 收益项，也不引入另一组收益阈值。

模型集成时提供与部署匹配的粗粒度 prefill 和跨 rank 传输数据。若缺少对应路径数据，
首版采用本地来源/冷启动，不凭网卡标称速度选择主动远端复制；容量压力下的保值
转换仍按可行性执行，不依赖这项收益判断。已有单 placement 模型的成本规则不变。

恢复准备的目标是 fresh 请求的输入前沿，或 paused 请求的原已提交前沿加第一个
新执行单元。已经发布的生成 token、RNG、OutputSession/grammar 状态留在原请求，
Replay 重建数学上下文，不重复对外生成或发布旧输出。

## 7. 按需求身份联合预留

现有 `ContextDemand` 只有一项 State variant，`take_history(pool)` 会合并同 pool
的预留。混合 placement 需要一份 bundle 同时描述不同 variant 和不同 history owner。

目标接口采用 bundle 内的需求标识：

```cpp
struct ContextDemandId { std::uint32_t index; };

// 每次添加返回本 bundle 内的稳定 ID。
ContextDemandId add_state(StateDemand);
ContextDemandId add_history(HistoryDemand);

// reserve(bundle) 全部成功或全部不取得；按 ID 取回对应的一份许可。
RecordReservation ContextReservation::take(ContextDemandId);
```

这是目标接口形态，实际声明归 ContextDirectory 所在模块。StateDemand 保留 variant、
rank parts、memory domain 和逻辑/物理计数；HistoryDemand 保留 pool、placement、
该 history 的增长/恢复/COW 及 metadata 需求。一个语义需求可以展开为多个底层
RecordDemand，但返回时必须保持来源映射。

例如 A 的 Main(rank0) 与 B 的 Main(rank1) 使用相同 pool，仍是两项独立需求。
目录可在内部合并容量评估，不能把许可先合成 pool 总额再按对称 rank 数量拆回。
State variants 同样不能在分配或取回时混用。需求 ID 在各自 bundle 内有效，不能
跨 reservation 使用；一项只允许消费一次，未消费部分由 owner 正常释放。

沿用 `RecordDomains::reserve()` 的原子物理预留和 metadata 生命周期。迁移相关
调用方后删除旧的 pool-only/单 State 取回合同；不改变 Op 的 StateField/PagedPlane views。

## 8. 跨 owner Context 转换

### 8.1 完整来源和变换范围

转换输入是完整 checkpoint，或已经停止写入、到达合法 canonical 边界的活动请求。
目标是新的 placement、相应 State variant 和 history placement。

GLM 模型产生字段变换表：KDA conv/recurrent 和公共 hidden 在各自 rank 本地复制；
Main/MTP raw tail 从旧 owner 传到新 owner；Main/MTP history 在新 owner 建立相同
有效 coverage。StateImage 保持整条记录的既定布局，不新增逐字段分配器。

复制只使用该来源的 State 和 history。其他私有锚点保留原 placement，后续仍可作为
候选。首次实现不依据相同 token 摘要把另一份独立计算的目标 history 拼入本次恢复。

目标必须同时容纳新记录和必要执行峰值。源和目的并存期间不提前扣除源占用；
分批传输只限制 staging，不消除源与完整目的记录的并存要求。峰值不足时，转换
计划不提交，回到其他恢复/回收方案。

### 8.2 事务阶段

```text
评估
  → 取得 source leases、目的联合预留和 Context frame
  → 准备局部复制、跨 rank 复制、目标页表
  → 提交全部参与 rank
  → 所有参与方 DeviceRetired
  → 发布目标 coverage/State/placement
  → 交给请求执行，结算源引用与未用许可
```

| 阶段 | 合同 |
|---|---|
| 评估 | 不改源、不分配可见目标；保留实际需要的复制工作和需求 |
| 已预留 | 目标仍不可见；取得来源内容、coverage 和地址借用 |
| 已提交 | 同一 ticket/command 关联全部参与 rank；保持源和目的至最后消费者退出 |
| 已退役 | 仅在完整完成事实成立后发布；ResultReady 本身不足以释放源 |
| 已发布 | 原子更新请求绑定；释放源只会释放真正没有其他引用的对象 |

保值转换保留数学内容代际及 canonical metadata，使用新的绑定/物理身份；下一次
数学写入按已有合同推进内容代际。新的 residency 使旧 Work 无法误用新的行或地址。

默认只满足必要请求绑定。若正常缓存策略要求保留目标 checkpoint，其额外写目的
属于可选保存需求，必须明确计费；申请不到时不阻止必要执行。转换结果不会自动
复制整组私有历史锚点或永久成为共享缓存。

同一来源/目标的后续请求在当前转换完成后重新查询目录。若已有完整目标 checkpoint，
直接复用；若未保留可共享恢复点，则正常选择来源。不建立第二套跨请求复制缓存。

### 8.3 字节动作、coverage 和通信

共享传输合同增加源/目标 domain（rank、memory）、记录身份、记录内范围及字节数。
目录验证记录、coverage 与借用；取得地址借用后生成 rank-local offsets，wire 中不
传进程指针。模型字段映射由 Session/State layout 提供，传输模块不解释 KDA/MLA。

完整不可变页可按连续区间合并。部分尾页按对应 history 的有效 plane 行复制，不能
把整个物理页中尚未有效、或正被另一个 writer 追加的后缀一起读取。Kpool 的 latent
与完整 pooled-key coverage 各自按其布局解释；未完成 pool 由 canonical State tail
恢复。目的只发布来源允许的 coverage，物理字节写入不自动扩张逻辑历史。

跨 rank 数据走既有 NCCL P2P，控制通道只发送有界描述和完成事件。建立独立的
Context 通信 group/stream，按协调端确定的事务与批次顺序提交；其资源在启动计费。
它与模型计算共享实际链路，不假定通信可以免费重叠。

首版一个实例同一时间推进一个 Context 转换，复用已有 context 操作门控。离散范围
使用预先规划的有界 Device pack/transfer/unpack 缓冲，连续范围可直接传输。批次缓冲只有
在本批最后消费者完成后复用；全部目的容量在首批前取得。使用现有 byte-copy、事件
和通信能力，必要的 gather/scatter helper 只处理范围，不管理 placement。

Context group 与模型 collective 分离仍需统一提交顺序和 stream 依赖；双方不得因
本地取消各自跳过已经约定的 send/recv。其他请求只在资源借用不冲突、既有执行
额度允许时继续计算。池整理等待相关地址借用退出。

### 8.4 取消与失败

- 提交前取消：释放目的预留、metadata 和源租约，撤销目标负载记录。
- 提交后取消：停止该请求的新计算；完成已约定传输或在实例失败路径中 drain，
  退役后丢弃未发布目的，最后结算来源。
- 转换成功但请求已取消：不发布新的活动绑定；缓存保留仅按正常策略发生。
- 源失效或资源变化发生在预留前：丢弃旧评估，重新选择。
- CUDA/NCCL 执行失败：沿用 Executor 实例失败与全体 drain，不采用部分目标。

跨 owner 恢复的目的记录与现有活动写入隔离。正常取消不会破坏原来源；消费者尚未
退出时不承诺已经释放物理资源。

## 9. 容量压力与暂停恢复

普通 unit 申请不足时，先处理已可回收对象、必要维护和在途退役。Fresh admission
保持现有权限；现有活动请求的进展按请求年龄与恢复优先规则处理。

对因 history 增长而受阻的请求，按以下顺序推进：

1. 回收对短缺 domain 有贡献的可选缓存。
2. 检查该请求在其他合法 placement 的完整转换与继续执行是否可行。目标选择使用
   最小 S、随后 L/N；源是该请求自己的完整提交状态，不重新做缓存亲和选择。
3. 若仍不足，按既有年龄规则选择对短缺有贡献的较年轻请求。这个受害者若能完整
   转移到另一侧且能释放所需资源，可用转换保留上下文；否则暂停并使用原恢复流程。
4. 所有转换峰值都不可满足时，通过释放较年轻请求的可释放上下文、保留逻辑恢复
   信息并 Replay 取得进展，不实现两个请求互持半份预留的交换协议。

迁移受阻请求本身，目的是让它在目标继续增长；即使旧共享前缀仍被别人引用，也
可能有效。迁移受害者以帮助另一个请求时，则必须有目录证明的实际释放贡献。
不能用请求长度或总 KV payload 代替这两个判断。

### 9.1 暂停合同

将目前 `save_snapshot` 布尔选择扩展为模型提供的暂停/恢复动作：

| 动作 | 保留及效果 |
|---|---|
| 保留 Device 上下文、解除执行绑定 | 可释放 lane/row；保留字节继续占 Context，不能当作容量释放 |
| 转换 placement | 保值恢复到目标；实际释放量取决于旧引用和转换后持有内容 |
| Host snapshot | 仅由具备对应 Host 存储的部署提供，延续现有语义 |
| 释放并 Replay | 保留请求输入、已提交 token 和逻辑恢复目标，释放可释放数学上下文 |

Runtime 选择合法动作，模型负责 canonical 边界与实际需求。Spark 不为此建立 Host
Context cache。暂停及转换不能发布尚未采用的 MTP proposal/selection；正常提交
先完成 KDA fold、Main/MTP coverage、tail 和 conditioning 的一致性。

恢复保护覆盖旧 frontier 和首个真实新单元，不能只恢复到原位置就解除保护。压力
转换只在资源不足且能推进时触发，不在每轮 decode 按负载再平衡。其他旧锚点留在
原 placement，不要求随活动请求整体搬迁。

## 10. 前缀保留与回收

### 10.1 共享节点

语义索引节点可包含多个 placement 的 checkpoint handles。共享发布去重和
`retain_shared_checkpoint` 查询增加目标 placement；另一侧已有相同前缀不代表
当前侧已经有可保留或直接采用的恢复点。

同 placement 的相同完整前缀，按已有合法替换条件保留代表；正在使用的实例继续
由引用/租约管理。不同 placement 不因摘要相等而互相替换。等待来源、私有 E/R/
长锚点仍沿用原所有权，不为每个 rank 复制一组私有记录。

### 10.2 简单保留规则

使用现有恢复价值、近期使用和保存优先级。使用证据关联到具体 placement，另一个
owner 上的命中不自动把本副本标为已使用。实际跨 owner 采用会记录来源使用和目标
执行，后续正常 checkpoint 保存可在目标留下副本。

回收首先要求对短缺 domain 有贡献；同优先级按既有最近使用顺序处理。保留另侧
等价恢复点时可以回收本侧副本，首版不计算未来复制次数或长期副本收益。每次释放
后更新引用和可恢复来源，不能因两侧互为备份而在同一评估中把两份都算作零损失。
必要执行许可优先于可选副本，池容量持续供请求及缓存共同使用。

## 11. 执行行映射、MTP 与 Graph

本节“全局”指一个执行实例内的完整 batch。一次模型 Work 冻结全局成员、residency
和 placement，生成 global-row 到
owner-local-row 的映射。KDA、TP/EP 和 head 消费全局成员；MLA/Indexer 使用该侧
压紧的有效行，完整输出恢复为全局列序后进入 mHC/residual。

Main/MTP 共用 placement。不同请求的长度、有效 draft 数和接受长度保存在当前
frame metadata，canonical 状态按 [MTP 合同](glm5next/mtp.md)提交。Context 操作不
保存或迁移可在下一轮重新生成的 selection、临时 records 和递归 hidden。

Graph 使用稳定的 frame/backing 与 metadata 地址。单次执行前更新 lengths、
positions、offsets、table rows 和行映射；对某侧零有效行使用合法的 padding/valid
metadata，或已规划的对应执行容量。通信节点顺序由模型整体保证，不能由本地
attention 是否有有效行独自改变。不按请求 ID、前缀身份或 owner 排列复制 Graph。

容量与映射应在模型 Graph 规划时选定；本设计不要求把所有 owner-local 计算都填充
到最大 B。可用固定容量区间覆盖有效行变化，调用现有 Op 的容量与更新合同。
Prefill 继续沿用集中分块调度；attention DP 本身不额外引入一套 rank 独立调度器。

## 12. 目标接口与文件归属

下列名称描述目标职责；实现时在已有模块内收敛，不建立第二套目录、缓存或调度器。

| 位置 | 需要落地的内容 |
|---|---|
| `include/ninfer/deployment.h`、`distributed/topology.*` | 独立的 adp 配置与 adp 坐标，保留 dp 的完整副本语义 |
| `runtime/contract/placement.h` | PlacementId、负载摘要、选择原因等跨模块纯值合同 |
| `runtime/engine/context_cache/placement_policy.{h,cpp}` | S/E 计算和固定排序；输入候选摘要，输出选择，不持有模型/目录 |
| `runtime/engine/context_cache/resource_manager.h` | 来源×目标候选、placement-aware 共享发布/查询、等待引用和回收 |
| `runtime/engine/engine_schedule.inl`、请求记录 | 负载加减、完整扫描与重试、暂停动作、恢复/转换生命周期 |
| `models/common/context/directory*` | 多项 State/history demand、按需求 ID 取得许可、转换目的与引用发布 |
| `models/common/context/contract.h`、`transfer.*` | 有端点的记录范围、局部/跨 rank 传输阶段、借用与退役 |
| `runtime/contract/resources.h`、`context_cost.*` | Peer 端点及粗粒度路径成本；沿用现有 preset 管理 |
| `distributed/`、Executor 既有 codec/worker 合同 | Context group、范围传输、参与 rank 完成与错误传播 |
| `models/glm5next/session/`、`state/` | placement 解释、canonical source、typed demand/复制范围、压力恢复 |
| `models/glm5next/parallel/`、`program/` | 行映射、固定 frame、模型通信与 Graph 绑定 |

模型侧需要提供的接口能力：

```text
列出本请求合法 placements
取得公开的 token 负载计数
检查 source + target + 恢复意图，返回恢复摘要与完整需求
按需求创建绑定/转换事务
提供合法暂停动作及真实释放集合
报告完成后的 placement、residency、前沿及退役事实
```

恢复候选属于模型类型，Runtime 通过公开摘要排序。候选身份、实际资源方案与
准备阶段必须使用同一权威事实，不能在 Engine 重新解释 State variant 或页布局。
GLM 专有参数、合法恢复边界和不支持情形由模型报告；Op 支持仍由对应 Op 负责。

## 13. 实施顺序与检查

| 步骤 | 产出 | 关键检查 |
|---|---|---|
| A：共享选择与预留合同 | 独立 ADP 配置/坐标、placement/负载、来源候选、按需求 ID 联合预留；已有模型提供唯一 placement | DP/ADP 身份分开；混合 variant/owner bundle 归属正确；失败不留下半份许可；单 placement 行为有效 |
| B：GLM 放置与同 owner 生命周期 | 绑定 State variants/Main/MTP histories，接通亲和、capture、COW、释放 | 真实 source 资格；同侧共享；不同 placement checkpoint 共存；负载事件恰好一次 |
| C：跨 rank 转换与压力恢复 | 带端点的 Context 事务、暂停动作、合法边界转换和继续执行 | 有效 coverage 保值；转换峰值；部分页；取消/失败/退役；等待来源变化 |
| D：模型执行交接 | global/local 行映射、MTP canonical 状态与 Graph 更新 | owner 分布不均、零有效行、不同接受长度、移动后 replay 与真实请求 |

A 可以在算子工作并行期间推进；B–D 与 GLM Definition/Session/RankProgram 接入
衔接。`.50` 的 Linear/MoE、`.51` 的 Kpool/mHC/Vision Ops 保持现有数学接口，消费
当前 StateField/PagedPlane 和设备 metadata。算子工作区不承担请求放置或资源策略。

验证直接覆盖第 2 节的场景及以下边界，不增加重复产品入口：

- 固定负载输入可精确验证 1.5 边界、命中优先、轮转和突发分配的即时记账。
- 同 pool 不同 owner、不同 State variant 的需求必须取得各自实际预留，失败完整撤销。
- 引用共享时以实际释放集合检查回收；释放暂停绑定不能虚报仍被保留的 history 字节。
- 跨 rank 使用独立构造的 State/page words，覆盖部分页与正在追加的共享尾页；只
  比较来源合同覆盖的数据，并确认目标 coverage、字段与前沿一致。
- MTP 恢复保持完整 Main/MTP 前沿和实际采用列的关系。零个 draft 被接受仍可能
  采用 anchor；只有实际采用零列时才保持数学前沿不变，不能把两者混为一谈。
- 正常取消、执行失败、延迟退役、来源撤销和池搬移不能留下可见的半完成绑定。

运行观测复用现有机制，在执行实例内记录所选 placement、来源 placement/frontier、选择原因、
实际复制字节/耗时及各 placement token 负载。用实际数据判断规则是否需要调整，
不要求启动时重新跑一套模型或 kernel 性能标定。策略只在准入、恢复和资源事件上
评估；先用摘要筛选，选中后展开复制范围，工作量不随整个 Context 池容量扫描增长。
