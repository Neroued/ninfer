# NInfer 多卡引擎重构 · Stage 2 实施方案

本轮把公共 Engine 推进到这样一个状态。在 `pp=1` 的部署中，请求可以属于不同副本，各自持有 KV、草稿和采样状态，同时把命中的专家工作交给其他副本所在的设备。一个没有本地请求的副本也能参与计算。每份结果回到产生它的请求，调度器可以继续处理取消、评分和停机。增加卡数时，模型用同一套合同描述新的权重切片、状态属主和通信参与者。

**本轮有三个独立的交付结论。** 引擎用生产调度、协议和故障路径证明上述机制。Qwen4Exp 在双 Spark 上用真实前向验收跨副本 EP，并验证 PLE 放置可以独立选择。GLM5Next 在代码中接通通用 TP、ADP、EP 的规划、调度、状态和执行，在已有相应算子的目标八芯上验收整模。双 Spark 上的 GLM5Next 继续覆盖已能运行的 TP2/ADP2 路线。验收报告分别记载每一层实际通过的范围。

源码起点为 `refactor/multi-gpu-engine` 的 `bbaeffdfe`。本文替换此前的 Stage 2 方案，规定本轮的接入顺序、合同和验收口径。Stage 1 的独立 DP 是回归基线。PP、Host 多计算槽和跨流水级缓冲交接按总设计归入 Stage 3。下文标为新增的接口、配置和用例都是实施内容。[多卡总设计](multi-gpu-engine-architecture.md) · [Stage 1 方案](multi-gpu-stage1-plan.md) · [引擎所有权](../maintainer/engine-architecture.md)

[MoE 接口](#3-moe-从全局选择到紧凑专家计算) · [PLE 放置](#4-qwen4exp-的-ple-由资源计划决定) · [模型改动](#5-两个模型分别要接通什么) · [副本调度](#6-怎样让不同副本一起工作) · [实施批次](#8-实施顺序与每批出口) · [验收矩阵](#9-验收矩阵与判定方法)

## 1. 怎样判断哪一层完成了工作

| 层次 | 必须兑现的合同 | 足以支持该层结论的证据 |
|---|---|---|
| 引擎与分布式运行时 | 部署描述 rank 和成员。运行时依照已选执行路径的通信要求组织参与、提交、退役和故障隔离。每个副本独占修改本地请求状态 | CPU 多 rank 夹具驱动生产拓扑、Executor/Worker 协议和协调器，跑出单源、双源、迟到结果、部分提交失败及健康域继续服务 |
| 模型与权重实现 | 从实际 artifact 和部署生成 rank 计划，明确权重、行、KV、专家和输出属主。Session、Program、wire 和准备期消费同一份计划。Main、草稿、评分各自声明所需阶段 | 实际模型规划器和序列化覆盖目标部署，逐项核对切片、状态和阶段。在可运行的几何上，从公共 Engine 验证采用前沿、评分、迁移与回放 |
| 算子与设备通信 | Op 接受或拒绝本地形状、表示、工作区和 Graph 能力。通信实现接受或拒绝具体参与者、传输和录图要求。成功准备的路线能兑现其数值及执行合同 | 生产 Op 对独立数学参考，真实后端验证配对与回传，真实模型端到端验证该设备组合。缺失实现保留所属 Op 或通信后端给出的原因 |

还有一种失败需要单独记录。合法且有实现的计划，可能在给定并发、缓存、内存余量下放不下。容量求解器报告受限的物理池、所需和可用字节。模型缺少某个状态分片或行映射时，由模型报告该能力。部署的成员数量或嵌套本身有错时，由部署层报告。上层补充模型、rank、阶段和输入形状，负责清理已经占用的资源。这使目标机安装了新算子后，可以直接沿用已有的模型调度路径。[部署校验](../../src/distributed/deployment.cpp) · [容量求解](../../src/runtime/engine/context_capacity.cpp) · [Op 开发合同](../maintainer/op-development.md)

### 1.1 源码起点与要拆开的耦合

| 现状 | 本轮在何处改正 |
|---|---|
| `DpRouter` 和每个副本的 Session、Executor 已就位。生成、逐段评分和启动推测校准都有机会真正调用模型 | 保留请求属主，把三种入口都接入可预留、可提交的通信阶段。当前逐个构造并等待副本的顺序也要变更。[副本线程](../../src/runtime/engine/replica_engine.h) · [评分主控](../../src/runtime/engine/causal_score_core.h) |
| GLM 规划只放行 TP1/2、ADP=TP 等组合。执行中可见固定两个属主、`1-root` 对端及将属主号当成本地 rank 的假设 | 用成员表、属主组、切片和逐段行映射替换这些假设。DFlash2 现有 TP2 限制单独沿该组件的实际布局和 Op 检查。[GLM 规划](../../src/models/glm5next/parallel/plan.cpp) · [Packed 执行](../../src/models/glm5next/execution/text.cpp) |
| Qwen 规划只放行单 rank 或 DP1/ADP1 的 TP2。TP 大小还决定 PLE 放在哪里，Program 见到并行对象便合并 Device PLE | 分开查询 tensor、expert、PLE 各自的角色。单机和分布式准备统一消费已选 PLE 计划。[Qwen 规划](../../src/models/qwen4_exp/parallel/plan.cpp) · [Device PLE](../../src/models/qwen4_exp/program/device_ple.cpp) |
| 当前 MoE 有源端稠密 TopK、按全局 ID 筛选的专家子集、融合计算及 Host/GPU offload。局部执行可以在内部重新按专家分组 | 保留这些有价值的生产路径，为已按目的地派发的输入增加可直接消费的专家计算入口。本地融合与跨设备计算共享模型数学合同。[MoE Op](../../include/ninfer/ops/moe.h) · [现有公共执行](../../src/models/common/moe/execution.cpp) |

## 2. 并行计划给出哪些事实

部署层继续拥有全局坐标和有序成员。对本轮 `PP=1`，记 `T=tp`、`A=adp`、`E=ep`，世界大小为 `dp×T`。沿用当前连续嵌套布局，`A` 整除 `T`。副本内的专家属主数整除 `T`，跨副本的 EP 组由整数个完整副本组成且能分满整域。计划显式给出下面的映射，执行点只消费结果。[部署几何](../../src/distributed/deployment.cpp) · [rank 和有序组](../../src/distributed/topology.cpp)

| 几何 | 本方案采用的布局 |
|---|---|
| attention 属主 | GLM 的每个属主由 `G=T/A` 张卡组成。副本内第 `t` 张卡的属主为 `floor(t/G)`，头分片序号为 `t mod G`。计划列出组内代表、持页的物理成员、各自头区间及结果归并顺序 |
| `E≤T` | 每个副本内有 `E` 个专家属主。每个属主的 `T/E` 张卡合作完成同一组专家的 intermediate 分片。每个切片恰有一次进入最终专家和的途径 |
| `E>T` | 当前布局每 `E/T` 个副本共用一组路由专家，每个专家由一张卡持完整 intermediate。每个副本仍拥有其独立的请求、attention、shared expert 和词表输出 |

在现有 `RankPlan` 上记录全局 rank、副本内下标、各通信角色的成员与组内序号、逐层全局专家 ID 到本地槽及 intermediate 区间的映射、词表和状态放置。Worker 用全局身份找计划，用组内序号指定 peer，用副本局部下标找 Session/Executor 参与者。相同成员在 context 迁移与前向交换中若有不同的流序，分别绑定。单成员角色直接走本地实现。模型准备、Work 编解码、Program、图与权重绑定共享这些事实。[角色绑定](../../src/models/common/parallel_execution.h) · [Worker 启动](../../src/distributed/bootstrap.h)

**第一个可独立复核的例子**是 `dp4/tp2/adp2/ep8` 的全局 rank 7。它属于副本 3，本地 TP 序号和 attention 属主均为 1，EP 序号为 7。它能为副本 0 的列执行第 7 分区的专家。把同一部署改为 `ep2`，它只和 rank 6 在副本 3 内分担专家。必须用 GLM5Next 的真实模型规划器生成并经 wire 解码得到这两个结果。另用 `T=4,A=2,E=2` 验证一组 attention 有两张卡、一个专家属主也有两片 intermediate。后者正好能检出现有权重选择器对部分专家 ID 加部分 intermediate 的缺口。[公共分片与物化](../../src/models/common/parallel_weights.cpp)

**三类成员关系分别有自己的来源。** 哪些副本必须就某次交换达成一致，取决于所选执行路径的通信阶段。丢失某个 rank 后哪些副本失去必要资源，取决于计划中的必需依赖。哪些 rank 可以合并一份 PLE 查表结果，取决于表的放置和消费者的输入布局。拓扑目前按 EP/TP 比例给出失败域，实施时用必需依赖生成或校验执行期的失败域。当前同步 EP 实现可以恰好使用相同一组成员。Engine 仍须分别取得故障范围与本步的参与范围，以便本地阶段、其他域和未来采用不同通信方式的实现独立推进。[拓扑与进程监督](../../src/distributed/bootstrap.cpp) · [通信准备](../../src/distributed/communication.h)

## 3. MoE 从全局选择到紧凑专家计算

### 3.1 接口的分工

| 拥有者 | 输入、输出与寿命 |
|---|---|
| 模型与路由 Op | 模型规定路由公式、TopK 顺序、shared 的系数和输出舍入点。源端可继续生成 `ExpertRoutes`，即每个源 token 的全局 ID 和系数。融合本地路线可以直接消费它或在融合 Op 内计算 |
| 放置与加载 | 生成逐层 `global expert → 有序属主/切片 → local slot`，同时物化 gate/up、down、编码、weight scale 与每种 use 需要的 activation calibration。权重和只读映射在准备期持有，计算阶段借用 |
| 模型公共 MoE 执行与通信后端 | 根据路由选择目的地，产生紧凑输入和派发句柄。句柄保存来源列、TopK 槽、回传位置及事件的对应关系，归本轮本层的在途交换持有，直到最后一份贡献归并。后端决定打包、传输、分块和可重叠的流序 |
| 本地专家 Op | 新增直接消费按本地专家分组的 assignment 入口。取得唯一 hidden 行、每个专家的 offsets/counts、每项对应的行及系数、本地 bank/切片和输出视图。通过 prepare 查询允许的表示、上限、对齐、工作区和 Graph 能力。传入准备期的容量上限和设备上的实际计数，零项也是合法输入 |
| 模型收尾 | 按派发句柄把局部和归回源列，合入一次 shared，恢复下一层需要的复制或序列分片布局。Main 的 HC/mHC、MTP 的 residual 以及评分/采样均由相应模型阶段接续 |

这里的 assignment 是“一列 token 命中一个专家”的一次计算。源端继续使用现有 `topk_softmax_route`、`topk_sigmoid_route`、本机融合 Op 和既有 offload 实现。现有公共 `models::moe::execute` 绑定的是 softmax 模型，GLM 的 sigmoid 路径也要显式接入。新接口允许把多个步骤融合成一次设备执行，不要求每个实现都物化中间张量。双 Spark 的跨副本生产路径必须真实使用紧凑入口，包括 Eager 和选作验收的 Graph 回放。接收端若把数据铺回全域 `[K,T]`、再做一次全量 mask/分组，尚未完成这个交付。[路由与融合合同](../../include/ninfer/ops/moe.h) · [专家权重视图](../../src/models/common/moe/weights.h) · [既有 offload](../../src/models/common/moe/offload/execution.cpp)

双 Spark 的紧凑计算入口按以下视图实现。令 `S` 为本地专家槽数，`U_cap` 和 `M_cap` 是这个接收块的准备上限。来自不同来源的同号列占不同的 `u`。生命周期由该块的派发句柄持有。

```text
hidden[H, U_cap]                   去重后的输入，每个有效 u 对应一个来源列
offsets[S + 1]                     第 e 个本地专家消费区间 [offsets[e], offsets[e+1])
assign_row[M_cap], weight[M_cap]   每项引用一个有效 u，并携带全局路由给出的原系数
U, M                               本块设备端有效计数，offsets[0] = 0，offsets[S] = M
partial[H, U_cap]                  每个有效 u 得到本地所有命中项的加权和，供派发层归回
```

只读本地 bank 与其校准参数来自放置计划。执行只读有效区间，空专家的两个相邻 offset 相等。`U=M=0` 的块也能合法完成。参考路线在此边界交付 `partial`。能够融合归并的目标实现可以直接写约定的回传区并记录完成事件。两种方式都由句柄保持源列的对应关系。[当前局部计算的工作区](../../src/ops/moe/kernels/cuda/block_scaled/workspace.h)

### 3.2 一层的执行顺序和精度

1. **指定唯一来源。** 每个有效列用副本、执行票据及其逻辑行列标识。TP 复制的列指定一个派发者，序列分片沿用列属主，Packed 通过段表取得原位置。不同副本都有“第 0 列”时，各有自己的回传槽。实际 wire 可用步号和紧凑索引表达身份。
2. **在来源做全局选择。** Qwen4Exp 在全局 routed logits 中选 TopK，使用选中项的 softmax 系数，另取 shared 的 sigmoid gate。GLM5Next 按 `sigmoid(logit)+bias` 选择，系数按所选 sigmoid 值、归一化 epsilon 与 scaling factor 计算，shared 系数为一。同分按全局 ID，padding 贡献为零。拆分后保留原 TopK 槽和系数，目的地不再归一化。
3. **按目的地打包并计算。** 同一来源列命中同一目的地的多个专家，仅传一份 hidden，加上该目的地实际需要的 assignments。接收端按 local slot 建一次专家分组，允许传输实现直接产出分组结果。Op 计算每项专家函数，按原系数加权一次。本地命中可借用源缓冲。无有效 assignment 的专家跳过权重和输入访问。
4. **归回并完成模型输出。** 参考路线按“来源列、目的地”回传一份加权局部和。专家 intermediate 切片的每个部分只进入总和一次。`E>T` 时 shared 的 TP 切片仍需各自取得正确的源输入，合并切片后按模型乘一次 shared 系数，再与 routed 合并。恢复源行序，按所选模型输出合同写入或加 residual。句柄、借用的源行与工作区在消费事件后才能复用。

采用独立的高精度数学参考。双 Spark 的保守路线在已定义的算子边界之间使用 FP32 加权与局部和回传，并在模型约定的位置转换。参考须包括现有激活、clamp、量化 scale、原有中间舍入和 epilogue。将共享计算分拆时，必要的部分和输出由对应 Op 提供。线上的 hidden、系数、返回值表示是选定执行路线的属性。允许目标平台选择满足该模型数值合同的融合或传输表示，任何新增的近似或舍入都要在数值约定中显式给出并独立验证。Engine 不规定线上 dtype。[专家 Op 与 epilogue](../../include/ninfer/ops/moe.h) · [数值验收标准](../maintainer/op-development.md)

### 3.3 如何在双 Spark 落地，也给扩规模保留空间

按来源列分块。对一个源 `s` 和目的地 `d`，设本块有 `B_s` 个来源列槽，目的地持有 `e_d` 个不同专家。唯一 hidden 行数 `U_sd ≤ B_s`，assignment 数 `M_sd ≤ B_s × min(K,e_d)`。按各来源界限相加，求出每个目的地的接收和回传上限。中间切片协作时，每个参与切片的目的地独立计费。容量计算包含头部、对齐、分组索引、逆映射、返回行、并发在途块和每个通信方向的缓冲。prepare 拒绝超出其资源合同的计划。

现有 `Group::exchange` 接受 Host 已知的字节区间。双 Spark 在这条接口上使用有界缓冲。GPU 将每个目的地的有效行和 assignments 写入紧凑前缀，同时写入设备计数。对端按双方约定的物理长度配对收发，只消费有效前缀。Eager 按本次已提交的物理列上限准备交换，Graph 使用录图桶的固定地址和上限。实际计数为零依然可以匹配协议。协议会读取的填充预先初始化。接收 Op 的计算量按有效 assignment 组织，稳态热路径无 Host 计数回读、临时设备分配或权重重排。允许其他后端使用设备端变长收发、目的地归约或分块重叠，条件是保留同一来源和退役合同。[交换与 Graph 接口](../../src/distributed/communication.h) · [后端选择](../../src/distributed/communication/selection.cpp)

**必须跑通的反例。** A 的某一列命中 B 的三个专家。B 当轮没有本地请求。在只有这列发往 B 的用例中，B 的本地来源列数是 0，接收的唯一 hidden 行数是 1，本地 assignment 数是 3。B 计算三项并把一份局部和还给 A。再让 B 也有自己的第 0 列，检查两份结果分别回到各自的请求。`ZeroSource` 描述本地请求来源，`ExpertSubsetKind::None` 描述一次局部计算的专家集合，两者有不同的含义。性能记录分别报告这三种计数、有效载荷和含填充的实际收发字节。直接消费分组结果省去了一次全域 TopK 重建。后端是否节省链路带宽取决于实际收发的物理长度。

### 3.4 扩展到更多设备时的接口边界

放置表分别表达“一个完整专家需要哪些 intermediate 切片”和“这些切片可以由哪些物理成员提供”。本轮按第 2 节给每个切片放置一份。以后若复制热点专家，一次逻辑命中仍只选一份完整计算所需的切片集合。选择物理副本不改变全局专家 ID、路由系数或回传位置。这样可以在物化与派发层改变负载分布，无须改模型路由公式或请求调度。

同一模型允许为 Prefill 和 Decode 准备不同的派发实现或块大小。实现向准备期给出容量、所需通信阶段和完成事件。逐块生产、边传边算以及按目标融合归并，都在这些资源寿命内完成。已承诺的有效 assignment 必须全部计算，热点超过单块容量时按准备的上限分块，不能截断命中或重新路由。双 Spark 验收有界收发与紧凑计算。更大平台可以沿同一入口以实际传输量、专家负载和端到端延迟比较其他实现。

## 4. Qwen4Exp 的 PLE 由资源计划决定

PLE 每次要为本副本的 token 和因果历史生成逐位确定的 BF16 embedding。同一份模型可以用磁盘读取、整表设备查找或分片设备查找完成这件事。三者改变的是常驻字节、带宽、准备时机和可录图的部分。把它们变为显式计划后，用户可以在同一 TP 部署上选择保留更多 KV，或选择更多设备常驻数据以减少 Host 交接。[PLE 查表数学](../../include/ninfer/ops/ple.h) · [现有两条装载入口](../../src/models/qwen4_exp/startup.cpp) · [分布式装载](../../src/models/qwen4_exp/parallel/weights.cpp)

### 4.1 配置、候选和确定的选择规则

新增 TOML 表 `[model.ple]`，通过产品配置和 EngineOptions 传到模型准备。三个字段为 `storage = "auto" | "disk" | "device"`、`device_layout = "auto" | "replicated" | "sharded"`、`preference = "capacity" | "latency"`。默认分别是 `auto`、`auto`、`capacity`。显式 Device 布局将候选限定到 Device，显式 Disk 与显式 Device 布局同时设置由 PLE 配置校验报错。单消费者的两种 Device 布局规范化为同一整表计划。显式选择无法满足时，报告该选择及缺少的资源或能力。公共配置层解析字段，PLE 实现解释其组合。[配置定义](../../src/product/config/schema.cpp) · [Engine 选项](../../include/ninfer/types.h)

| 候选 | 消费者与实现 | 必须计入的成本 |
|---|---|---|
| Disk | 本轮在每个需要输入的 rank 对本副本的窗口查本机 artifact，沿用异步直接 I/O、行缓存、pinned 缓冲与设备上传。TP2 两个消费者分别查同一窗口，直接取得各自的完整 embedding | 分节点的行缓存、pinned 与 Device 上传槽、草稿交接、I/O 并发与读者寿命。实际打开的文件、进程内缓存和共享物理池按其所有权记账 |
| Device 整表 | 每个消费者持完整的编码表和查表参数，本地查找就得到完整结果。TP1 和有足够容量的 TP2 都可用 | 每个实际副本的编码页、LUT、查找输出和必要的图资源。使用整表的 rank 跳过分片合并 |
| Device 按行分片 | 在**读取同一逻辑窗口**的副本内有序组中，按 artifact 的不重叠全局行区间划分既有表 shard。每个参与者用相同 token、tail、有效列和段布局查本地页，缺失行产生精确的零字。按独占行所有权把有效的 BF16 位模式合并 | 每个 rank 的真实本地 shard、LUT、部分输出、归并缓冲和通信。分片角色绑定该查表组及序号，不随 expert 组扩张 |

选择按以下固定顺序执行，日志给出每个被淘汰候选的原因，以及最终的逐物理池开销和逻辑 Context 容量。

1. PLE 实现从产物、行覆盖、消费者输入布局、I/O 和查表实现给出可供准备的候选。同一布局在本地和多 rank 路径下有相同含义。每个候选一并声明查表需要的通信、图段和工作区。
2. 启动规划把候选的真实开销交给既有 Context 容量求解。在同一设备或 UMA/Host 池上的多 rank 合并计费。先满足配置的最大上下文和并发所对应的既有最低工作集、显式 Context 容量、headroom、Graph 与其他必要资源。显式容量沿用原有语义。候选不得靠缩小用户指定值入围。
3. `capacity` 优先选择能为最紧的逻辑属主留下最多可用 Context 块的可行候选。用相同页几何和物理池约束比较，包含继续增加完整逻辑块需要的所有成员。显式容量场景仅用可额外承载的块数排序，发布的容量仍为指定值。`latency` 对可行候选的固定偏好为整表 Device、分片 Device、Disk，体现尽量减少 Host 交接和跨 rank 合并的策略。容量排序同分时沿用该次序。这是可复现的选择规则，最快路线须由同负载端到端测量确认。

准备期为一个引擎实例冻结选择。两副本处理不同请求时，只在各自副本内生产和归并 embedding。未来若要让它们共用一套物理表分片，PLE 实现需要按“来源、列、head、全局行键”分发查找，并按来源返回。单凭同号列做字级归并会把无关请求的位模式合在一起。本轮交付副本内整表、分片及 Disk，跨 DP 的表服务单列为后续能力。Disk PLE 的 Host 开销和分布式 Context 是否具备 Host 换出能力分别按各自的资源合同处理。

### 4.2 输入、异步寿命与采用前沿

每个活跃窗口都绑定 sequence、epoch、已提交前沿、前两个原始 token、有效列和 Packed 段边界。查表按真实 IDs、EOS 及各段各自的历史计算，padding 输出精确正零。多个 TP 消费者若要查同一窗口，模型先把该窗口的草稿 ID、有效长度和段顺序交给每个消费者。序列分片的本地列序不满足这个条件时，先恢复共同输入布局或选择独立查表。设备查找可以从设备读草稿，Disk 路线按已准备的 Host 交接取得草稿。Disk 的 Host 准备落在图外，ready 之后可以回放下游计算图。Device 整表或分片能否把查找录图，依据查表 Op、实际通信组和地址稳定性判断。模型执行等待生产者的 ready 事件，生产者在最后一个消费者完成后才复用物理槽。`ple::Ticket` 的 slot/generation 用来拒绝迟到的上传或取消。Session 只按真正接受的前缀采用 PLE 及其他状态。零源专家服务不会预读表、等本地 PLE 事件或改动 raw tail。[输入窗口与票据](../../src/models/qwen4_exp/ple/input.h) · [Disk/草稿交接](../../src/models/qwen4_exp/program/ple_pipeline.cpp) · [设备查找与归并](../../src/models/qwen4_exp/program/device_ple.cpp)

PLE 的交付要能复现四条实机路径。单消费者显式 Disk 和整表 Device，双消费者显式 Disk 和 Device 分片。另以双消费者显式整表验证独立查表和两份表的内存记账。对同一批原始输入逐 BF16 字比较，其中包含零符号、分片边界、不同 raw tail 的同号列、Packed、部分 MTP 接受及恢复后的下一步。分别检查表的装载位置、图内查找或图外生产的实际路径、最终容量和临时槽释放。[查表参考](../../tests/ops/ple/test_lookup.cpp) · [Packed 模型用例](../../tests/models/qwen4_exp/test_packed_prefill.cpp)

## 5. 两个模型分别要接通什么

### 5.1 GLM5Next

**权重与张量。** 在同一架构入口按真实绑定生成 KDA 整头切片、dense/shared FFN intermediate、mHC、embedding 和词表范围。Main、MTP、DFlash2 的所选完整或优化 head 使用各自绑定及同一逻辑词表分区，沿用已有的 128 对齐分区合同，检查局部 logits、候选归并、Grammar 与逐 token 评分。专家 bank 支持同时选择部分全局 ID 和部分 intermediate。gate/up 行与 down 列取同一区间，同时校验 block scales、FP8 refine、FP4 Local CSF、别名和实际字节。GLM 的连续专家分布和 Qwen 的逐层优先序分别由各自模型提供。[权重规划](../../src/models/glm5next/parallel/plan.cpp) · [物化与编码](../../src/models/common/parallel_weights.cpp) · [词表分片](../../src/models/common/parallel.cpp)

**MLA/ADP 与状态。** 一个 `PlacementId` 指向一个 attention 属主组。`G=T/A` 大于一时，Q 的头、`U_K/U_V` 和 O 的输入列按组内头区间切片。按现有 fused prepare 的读写合同，为每个需要的组员绑定本属主的 latent/index 历史和 raw tail，每份物理页分别计费。索引选择由代表产生，组内共享同一选择，Main/MTP 按模型已有的复用规则处理。组内合并 O 的局部贡献后，该属主输出一份结果。逻辑页的可用量受所有必要物理副本中的最紧成员约束。KDA 的 recurrent state 保持全副本按 TP 切分。请求更换属主时按字段和页搬移实际变化的数据，保留位置相同的 KDA 切片。旧页、新页和尾页持有至设备完成，随后才发布新放置。验证 COW、暂停恢复、前缀复用与半页迁移。[MLA 执行](../../src/models/glm5next/execution/mla.cpp) · [状态放置](../../src/models/glm5next/state/context.cpp) · [Session 绑定](../../src/models/glm5next/session/binding.cpp)

**批次、草稿与媒体。** Decode 提供副本逻辑行序、属主物理行序及逆映射。Packed Prefill 由序列分片生产者的列区间与 attention 属主段的交集导出每笔传输，覆盖空属主、奇数列和跨界长段。Work、codec、图输入、页表及 workspace 一起替换 `owner_tokens[2]`、`1-root` 等写法。MTP 的 routed 草稿层逐步进入专家交换，按本行实际接受数采用历史。DFlash2 按它自己的块 proposal、GQA 查询头到 KV 头的映射、滑窗 KV 和 target verify 执行。多个查询头分片需要同一 KV 头时，该头在相应成员复制并逐份计费。将现有 TP2 守卫替换为该组件的实际分片能力及 Op 的本地几何检查。Vision 保留完整媒体段的分工和源请求的列位置。生成、评分、前缀复用和图回放最终使用同一套行映射。[批次表示](../../src/models/glm5next/execution/batch.h) · [图输入](../../src/models/glm5next/program/graph_inputs.cpp) · [DFlash2 执行](../../src/models/glm5next/program/dflash.cpp)

### 5.2 Qwen4Exp

Qwen 的具体验收几何为 `dp2/tp1/adp1/ep2`，并保留已有的单 rank 和 `dp1/tp2/adp1` 下 `ep1`、`ep2` 两条路线。GDN、QSA、shared、HC 和词表取本副本的 TP 几何。Main 与 MTP routed bank 各自取实际 EP 属主和表示。即使每个副本只有一张卡，它也能持半个专家库、加入跨副本专家组，同时本地独立处理 attention 和 PLE。Qwen 的多 attention 属主需要该模型另外实现对应状态与行映射，本轮由 GLM 验证 ADP 的多种几何。[Qwen 权重规划](../../src/models/qwen4_exp/parallel/plan.cpp) · [Main](../../src/models/qwen4_exp/execution/text.cpp) · [MTP](../../src/models/qwen4_exp/execution/mtp.cpp)

准备请求、Worker 校验和 Program 各取副本身份、实际 result owner 与本地槽数。沿用 Stage 1 的总准入上限和副本槽准备口径 `ceil(max_concurrency/dp)`，单独核对 `max_concurrency<dp` 时无请求的副本仍有专家服务资源。替换把“存在 parallel 对象”同时解释成 tensor 求和、词表分片和 PLE 合并的调用点。Qwen 的 TP1 专家 profile 绑定实际物化的局部 bank，避免以 TP1 推导持有全局专家库。模型启动报告包含每个 rank 的模型常驻量、表布局、Host 资源和它服务哪个副本。[本地与分布式入口](../../src/models/qwen4_exp/parallel/startup.cpp) · [rank 准备](../../src/models/qwen4_exp/parallel/preparation.cpp) · [公共启动](../../src/runtime/engine/distributed_startup.inl)

## 6. 怎样让不同副本一起工作

### 6.1 同步的单位是一次实际通信阶段

模型 Program 与选定的 MoE 执行路线为每种可提交的工作给出有序通信阶段。模型和后端共同生成可匹配的阶段描述，其中含有序组、操作和方向、dtype、两端须匹配的物理长度/桶，以及可提供零源服务的成员。用阶段标识定位模型层或块。协调器只比较已经准备好的描述并选择可兑现的匹配，不自行推导模型层数或后端规则。与通信无关的本地 attention、PLE、草稿处理保有各自的形状和执行时机。对本轮双 Spark 的同步收发，只在涉及同一组的阶段约定步号和顺序。无需交换的 Stage 1 副本直接提交自己的工作。不同域可独立推进。

沿用每副本拥有线程、Session、Executor 和单个计算槽。给共享通信阶段增加以下三步，每一步都能在生产协议上观察。

1. **提出可兑现的工作。** 副本拥有线程持有可撤销的预留，向协调器给出值描述，包括合法 Prefill 截点、每行写入许可、草稿深度、可用图、frame 及兼容的通信阶段。每行许可分别校验，例如两行的 `[4,0]` 不可替换成 `[2,2]`。空闲副本可预留专家服务 frame 和通信信用。协调器不借用请求、页或 Session 指针。
2. **决定本轮的匹配。** 对本次通信所需成员，在各自已预留的选择中匹配阶段，发布有单调步号的共同决定。参与者可有不同的请求数、请求类别和有效列。用 offer 的代数原子裁决撤销与发布，迟到通知不能复活旧预留。全员只有零源时不派空步，由新工作、退役或失败唤醒。
3. **各自提交、独立收尾。** 每个副本由原线程将真实 Work 或服务 Work 交给自己的 Executor。同步组步号与各副本的 `ExecutionTicket`、frame generation、peer 命令序号分别维护。每个 Executor 的 `Scope` 只列本副本的物理参与者。已发布的决定须由全体完成该阶段，或共同进入故障处理。一个副本已发 Submit 后另一个发送失败也遵循此规则。

协调器处理 offer、退役和失败是事件驱动的。等待同组参与者时，各副本线程仍能推进 Worker 事件、Grammar supply、取消与停止。把对端的本地请求线程同步阻塞在一次模型调用内，会妨碍这些进展。[调度提交](../../src/runtime/engine/engine_execution.inl) · [Executor 合同](../../src/runtime/executor/contract.h) · [分布式 Executor](../../src/runtime/executor/distributed_executor.h)

### 6.2 请求类型、推测与公平

| 实际场景 | 本轮规则 |
|---|---|
| 两个副本的普通 Decode | 选兼容的交换桶与阶段。图均已准备时可共同回放，本地行数和 ContextBucket 可不同。形状无法在图上匹配时用已准备的 Eager 路线 |
| 一个长 Prefill，另一个单列 Decode | 两者能在同一序列的 Main 专家层会合时，采用 Eager，各自做本地前后处理。若阶段冲突，选一方的有效步，另一方按需服务。Prefill 在 Session 已许可的段内切分，保留媒体、checkpoint 及采样边界 |
| 持续同时有 Prefill 和 Decode | 当两类都有可兑现的工作，每轮含 Prefill 后给一次纯 Decode 机会。纯 Decode 后优先给等待最久的 Prefill 一个合法分块，能配对则同轮做单列 Decode。逐轮检验两类均获进展，记录最长等待与实际 token 配额 |
| MTP、DFlash2 和评分 | MTP 预留每行合法深度和 target 宽度，决定真实需要的共同 routed 草稿阶段及 verify 阶段。固定图按 `K+1` 验证宽度准备，每行的实际有效宽度和接受数独立，空行履行通信。DFlash2 的本地 dense proposal 单独调度，只把与跨副本专家有关的阶段加入约定。`score_tokens()` 的每个真实分块同样取得相应服务，结果按源 token 顺序发布 |

不同草稿深度的副本可以在共同阶段中屏蔽失效行，或选择一方做相应阶段、由对端服务。决定必须落在每行预留和已准备图/Eager 路线内。评分专用 Engine 沿用 `CausalScoreCore` 对整次 `score_tokens()` 的串行接纳。选中副本持有私有评分上下文，每一段单独协商，直到所有目标 token 按序结清或失败。生成侧沿用模型的有效长度、回放和采用语义，测试偏斜负载下可执行的 MTP 能取得进展。[推测执行合同](../maintainer/speculative-decoding.md) · [实际评分分块](../../src/models/glm5next/session/scoring.cpp) · [Qwen 评分分块](../../src/models/qwen4_exp/session/scoring.cpp)

### 6.3 零源、取消、停止与故障

零源是一种独立的服务 Work。它只携带组步号、模型阶段、本次交换上限和本地服务资源。它有物理 frame、只读权重和协议缓冲，可以收到非零专家工作。若服务段还须匹配副本内的 TP 通信，也使用已准备的协议缓冲。其 payload 不含请求的 KV、State、PLE token、采样或 Grammar 地址。沿用现有结果协议，由本副本的 `result_owner` 返回服务完成，决策属主集合为空，拥有线程提交不修改请求的收尾 Commit，等待所有参与者的 `DeviceRetired`。该流程不能增加生成 token、采样或评分计数。[Worker 协议](../../src/runtime/executor/protocol.h) · [WorkerLoop](../../src/runtime/executor/worker_loop.h)

取消在发布决定前可撤销预留。决定发布后先履行已承诺的通信，再在 BeforeSupply/AfterSupply 对应边界丢弃该行的结果和未采用写入。单槽在参与者退役后重新开放，包含等待 Grammar 的情形。Context 整理在自己的流上分段，对仍有读者的 backing 等到安全事件再迁移。正常 stop 停止新准入，保留所需专家服务至在途生成和评分排空。测量按原提交切口包含其依赖的服务票据。

运行期 Worker 丢失或通信中止时，依据冻结计划的必需依赖立即关闭故障域的准入，撤销未发布预留，唤醒等待者并幂等 abort 该 epoch。各通信拥有线程完成中止和进程回收，确认设备读者结束后释放租约。部分提交、迟到结果和重复通知只能结清同一逻辑请求一次。未派请求仍可去健康域，测量窗口若只有部分参与者建立则回滚并返回原始错误。[进程监督](../../src/distributed/bootstrap.cpp) · [副本路由与生命周期](../../src/runtime/engine/engine_coordinator.h)

## 7. 启动、Graph 与资源如何收尾

启动依次完成初始拓扑、候选收集、共同准备、校准和开放准入。当前分布式 bootstrap 已经需要拓扑来启动和监督 Worker，而 PLE 的最终选择需要 Worker 的实际资源数据。实施时将“足以建立控制连接的几何”和“可以执行的最终计划”分两次确认。单 rank 采用相同的候选评估和最终计划，省去跨进程交换。[bootstrap](../../src/distributed/bootstrap.cpp) · [准备编排](../../src/runtime/executor/preparation.cpp) · [本地启动](../../src/runtime/engine/startup.inl)

1. **先取得每个 rank 的事实。** 按部署身份建立控制连接和启动期进程监督。各 Worker 核对 artifact、模型几何、Op profile、可用通信能力、物理池与当前资源，返回候选及原始错误。模型给出可能的角色和组需求。此时可以准备候选共用的只读权重，PLE 表、所选专家放置特有的权重和依赖其布局的资源分开归属。共用权重的加载仍遵守节点 staging 限额。
2. **选计划，再准备依赖。** 把 PLE 候选与模型已准许的专家权重放置组合起来，就共同权重、预期通信常驻量和各组合的峰值计算可行解。两项选择各自遵守本模块的显式配置。按选定候选绑定 PLE、专家 bank 和通信角色，每个有序组的成员共同确认 peer、类型、上限和顺序，随后连接、预热。物化和通信预热后按实际快照复核容量，避免把快照中已扣除的常驻量再扣一遍。在本次准备尝试内冻结放置、后端与图上限，将标识传给 Worker 和图，防止重建时误用上一次的缓冲或轨迹。容量复核、图和校准通过后才发布最终计划与启动报告，执行期失败域随该计划生效。启动期监督在整个交接期间始终持有已启动进程。
3. **集体阶段按参与成员准备。** 权重可按物理池分波物化。图探测、warmup、重建和 capture 如果含通信，分波单位须包含该阶段所有相互等待的组成员。现有 `build_replica_waves` 只覆盖一个副本的进度要求。先收齐本地准备成功，再让各成员依相同组序进入。预留共享设备的总量，纯本地操作可分段串行。这样首个副本不会在 capture 中等待尚未被安排执行的对端。
4. **处理可恢复失败。** 在尚可共同退出的准备边界，如果资源实测使候选失效，释放本次尝试的表、Program、图、Context 及候选专用通信资源并收齐确认。共享资源只在所有权和布局匹配时复用。自动选择按已记录的顺序尝试尚未试过的可行候选，显式固定的选择保留原配置。原有自动容量重算也须保持同一用户约束。所有候选耗尽时列出各自原因。某成员已进入无法安全撤回的通信，则中止参与域并清理进程。执行期已经发布给请求的计划保持冻结。
5. **全体就绪后再跑模型校准。** 现有副本构造会等待自身的推测启动完成。改为先建齐需要相互服务的副本线程和 Executor，收齐 ready，再让真实校准与服务 Work 从同一调度入口进入，最后开放准入。校准用临时 COW 上下文。失败则唤醒并回收本次启动的所有未开放成员。正常关闭依次排空设备读者和通信、销毁图、释放组和 backing。[校准入口](../../src/runtime/engine/speculative_calibration.h) · [副本构造](../../src/runtime/engine/replica_engine.h)

**图的兼容条件由执行路线声明。** 对同一通信阶段，捕获和回放匹配组、顺序、物理桶及双方收发长度。rank 本地的算子形状、ContextBucket patch 和有效行由本地 Program 决定。为活跃来源和纯服务角色准备能履行相同通信阶段的图段。服务图只绑定协议缓冲，跳过所有读取请求、页表、状态、PLE、Grammar 或采样的节点。MTP 的本地事件要等到实际生产者完成。若 Group 不支持捕获后门控通信，按已有分段方式在图外决定调用哪些草稿单步图和 target 图。一旦选定某次交换，全部必要成员都执行其通信节点。每个已捕获 Group 记录到对应图，核对实际 replay 计数。[通信图能力](../../src/distributed/communication.h) · [Graph suite](../../src/models/common/graph/suite.cpp) · [Qwen 图执行](../../src/models/qwen4_exp/program/graphs.cpp)

**后端选择贯穿产品入口。** 将已有 `CommunicationPolicy` 通过新增 TOML 字段 `distributed.communication_policy`、部署参数、wire 和 Worker prepare 传递，取值为 `auto`、`native`、`nccl`，默认 `auto`。显式值须对所需通信实际生效。报告每个目标组/操作的后端、捕获与回放情况。无法满足显式要求时，返回后端给出的原因。Native 在双端的具体支持范围由它自身处理。该配置让两次真实 Engine 运行分别检验两套通信实现。[部署读取](../../src/product/config/file.cpp) · [现有通信策略](../../src/distributed/communication.h) · [通信测试入口](../../tests/distributed/test_execution.cpp)

下列片段给出双 Spark 跨副本验收所需的新增选择和并行几何，合入现有的节点、rank 与 artifact 配置。第二次保持同一产物、相同几何与 Disk，单独把 `communication_policy` 改为 `"nccl"`。做 PLE 的同拓扑对照时则只改 `[model.ple]`，例如 TP2 先选 Disk，再选 `storage="device"` 和 `device_layout="sharded"`。

```toml
[distributed]
communication_policy = "native"

[distributed.parallel]
dp = 2
tp = 1
adp = 1
ep = 2
pp = 1

[model.ple]
storage = "disk"
```

## 8. 实施顺序与每批出口

第一个整模里程碑定为**双 Spark 上的 Qwen4Exp 能通过公共 Engine 完成一次跨副本生成，并在评分专用 Engine 上完成多段评分**。使用第 7 节的 `dp2/tp1/adp1/ep2` 配置，显式选 Disk PLE，关闭 CUDA Graph 和推测解码。这是一条最终要支持的产品路线。它的前置条件是本地权重正确、专家能真实往返、各副本能共同提交并独立退役，以及启动能为这些参与者准备好资源。

### 8.1 从模型计划到 Worker 实际使用的权重

开工时固定受影响旧路线的产物、部署、输入和已有测量，缺少可比基线时在改动前补测。现有 CPU 拓扑测试已覆盖八卡、十六卡和多个专家域。第一项改动落在两模型的生产规划器、加载请求编解码、Worker 身份校验和公共权重选择器。复用拓扑提供的坐标，分别绑定 tensor、expert、attention 等角色。在加载合同中区分启动时即确定的权重分片、需要评估的资源候选，以及准备成功后发布的选择。Worker 能识别这次准备所用的布局，候选改变后须重新确认。[已有拓扑用例](../../tests/distributed/test_cluster_topology.cpp) · [Qwen Worker 校验](../../src/models/qwen4_exp/parallel/preparation.cpp) · [GLM 规划](../../src/models/glm5next/parallel/plan.cpp)

用三个例子约束同一套合同。Qwen 的两个 Worker 都是 TP 本地序号 0，却必须绑定不同的 Main/MTP 专家。GLM 的 rank 7 在 `dp4/tp2/adp2` 下随 `ep8 → ep2` 改变专家分区和通信成员。再用 `dp2/tp4/adp2/ep2` 验证同一专家的两段 intermediate 与部分专家 ID 可以共存。真实模型元数据和生产 codec 验证 rank 计划，合法小产物通过加载器重建 gate/up 行、down 列与量化参数。各分支拿到它实际使用的映射就可以开工。完整 GLM 状态迁移、所有权重表示和 PLE 自动选址有各自的后续验收点。[公共权重选择](../../src/models/common/parallel_weights.cpp) · [切片重建用例](../../tests/models/common/test_parallel_weights.cpp)

### 8.2 专家计算和副本调度分别验证，在 Qwen Eager 汇合

可分别开发设备上的专家计算和 CPU 上的副本调度。两者共用由模型与后端准备的阶段描述，包括有序成员、操作、两端可匹配的物理上限和零源服务能力。路由后的有效行数和命中专家数留在设备端。调度器据本轮预留匹配已准备的阶段。

| 工作 | 开始条件与应交付的生产路径 | 如何验证、何时交给整模 |
|---|---|---|
| 一层 MoE 的计算与往返 | 有了逐层专家映射即可做紧凑 Op 和独立参考。接入两端的通信角色、实际 workspace 预算、组准备与显式后端选择后，在双 Spark 执行派发和归回。GLM 的 sigmoid 与 Qwen 的 softmax 分别验证 | 用可控路由驱动生产 Op，B 本地 0 个来源、收到 1 行、执行 3 项。核对加权结果、shared 贡献和源列身份，再让两端同时发送同号列。Qwen 在一个后端通过即可开始该后端的整模联调，另一后端及 GLM 路由分别继续验证。单层结项时 Native 和 NCCL 均有真实往返记录，同时给出有效行、assignment 与物理收发字节。[专家参考](../../tests/ops/moe/expert_oracle.h) · [通信测试](../../tests/distributed/test_execution.cpp) |
| 一轮工作怎样提交和结束 | 有了通信阶段描述即可用 CPU 夹具驱动生产 Executor/WorkerLoop，实现副本预留、共同决定、真实或零源 Work、本地 Commit 和设备退役。生成及逐段评分都接上这一协商边界 | 单源、源端互换、双源均可推进。零源只有物理票据，真实请求只发布一次。至少在一次撤销、一次部分 Submit 失败和正常 stop 中验证预留及票据能结清、等待者被唤醒。公平、Grammar 与多域故障在同一协议上扩展。[多副本用例](../../tests/runtime/test_multi_replica_engine.cpp) · [Worker 协议](../../tests/distributed/test_protocol.cpp) |

**第一条 Qwen 单层路线与生成调度都就绪时，就接公共 Engine。** Qwen 按副本准备本地槽、专家 bank 和显式选择的 Disk 窗口。启动先落实当前路线需要的组准备、缓冲计费和失败清理。以一边有请求、另一边只做专家服务开始，交换两边角色，再跑双源。生成至少跨过一轮退役并继续下一步，用独立的多段 `score_tokens()` 检查每一段都能得到对端服务。检查逐 token 分数、已采用前沿及 stop 排空。同一产物分别显式选择 Native、NCCL。固定 Disk 是控制这个用例的变量，TP1 仍通过第 4 节的 PLE 计划取得它。[通信准备入口](../../src/runtime/executor/preparation.cpp) · [Qwen 评分提交](../../src/models/qwen4_exp/session/scoring.cpp)

### 8.3 按实际依赖推进 PLE、GLM 和图执行

| 工作 | 什么时候做、为什么 | 独立验收点 |
|---|---|---|
| PLE 放置与启动重试 | 角色和资源候选明确后即可接显式 Disk、Device 整表和分片。启动侧同时拆出候选专有表、通信资源及重试所需的产物读取寿命。当前 Device 表被放入权重请求，物化后 Worker 释放 reader，Qwen 的重试主要释放 Program 与 probe。自动择优和已分配候选的撤回要随这段所有权改造一起交付 | 在 TP1 对照 Disk 与 Device 整表，在 TP2 对照 Disk、Device 分片和整表。验证 embedding 位模式和采用后历史。在可安全撤回的边界注入容量失败，核对全组释放确认、下一候选、显式选择的原始错误和物理池账目。第一轮 Qwen Eager 只依赖它所用的显式 Disk 路线。[PLE 表的物化](../../src/models/qwen4_exp/parallel/weights.cpp) · [Worker 物化与释放](../../src/runtime/executor/rank_preparation.h) · [Qwen 资源释放](../../src/models/qwen4_exp/parallel/preparation.cpp) |
| GLM 的调度与实机 | rank 计划可用后即开始真实 Session、Work、codec、属主页和逐行采用验证。已有双 Spark TP2/ADP2 路线随相关改动回归。目标八芯上，公共 EP 路径和 GLM Main Eager 接通即可先跑 `dp4/tp2/adp2/ep2 → dp4/tp2/adp2/ep8`，两次配置只改 EP，比较专家放置和跨副本归回。组内多卡 attention 接通后再跑 `dp4/tp2/adp1/ep8`，这次只改 ADP，比较头分片和状态放置 | 第一组对照验证 TP 复制列只派发一次、shared 的各 TP 片只计一次。这是 Qwen TP1 覆盖不到的组合。第二组对照验证两卡同属主的头切片、KV 和结果归并。其余 TP/ADP/EP、Packed、迁移及所选草稿组件按第 9 节逐项完成。Spark 的 GLM 局部 Op 缺口保留原始诊断。[GLM Session](../../tests/models/glm5next/test_session.cpp) · [MLA 参考](../../tests/ops/softmax_attention/kpool_mla/reference.h) |
| Graph、推测与多请求生命周期 | 单层通信就绪时，用同一个桶改变设备计数，检查后端捕获、回放和零源缓冲。这项检查可与 Eager 联调并行。整模普通 Decode Graph 以同配置的 Eager 结果作对照。MTP 先在 Eager 验证每步路由、部分接受和回滚。启用自适应启动校准前，先建齐同域参与线程。含跨副本通信的图探测与重建按实际通信成员共同进入。MTP Graph 结合这两条已验证的路径，DFlash2 按自身 proposal/target 阶段接通 | 同一请求的 Eager 与 Graph 都通过相应数值参考。零源无请求存储访问，取消、Grammar 等待、持续 Prefill/Decode 混合、测量切口、故障域隔离及关闭均能结清。已有独立 DP、本地融合和 offload 路径做相应回归。[图准备分波](../../src/runtime/executor/preparation.cpp) · [副本启动校准](../../src/runtime/engine/replica_engine.h) · [推测执行合同](../maintainer/speculative-decoding.md) |

每个阶段保留对应配置和结果。最后按第 9 节补齐尚未覆盖的组合、失败路径及固定负载性能对照。已有证据在相关代码或条件改变时重跑。平台不可用时，分别记载已验证的引擎协议、模型调度和待测的整模能力。

现有六项 CPU 基线可用下面的命令定位和运行。各阶段按受影响合同补入用例。双 Spark 实机使用明确的产物与部署文件，通过已有的 `gpu-run -r spark-dual -- ...` 入口执行。

```bash
ctest --test-dir build/rtx5090-dev --output-on-failure \
  -R '^ninfer_(cluster_topology|context_capacity|dp_router|multi_replica_engine|parallel_weights|distributed_protocol)_test$'
```

## 9. 验收矩阵与判定方法

下表中的配置均为 `pp=1`。每行说明计划取得的证据。比较既有实现时，实施前固定可运行的 artifact、配置和负载。新增能力单列其绝对指标。

| 环境与证明对象 | 必跑场景 | 通过依据 |
|---|---|---|
| CPU · 引擎和真实模型规划 | 八 rank 的 `dp4/tp2/adp2/ep8`、`dp2/tp4/adp2/ep2`，十六 rank 的 `dp4/tp4/adp2/ep8`。后者有两个八 rank 的专家域，用来验证一域停滞或失败时另一域继续前进。另测并发小于副本数。生产模型计划与 wire 单独检查，运行时夹具驱动生产 Executor/WorkerLoop | 精确核对组、切片、来源、归回、每行许可、请求发布次数及每张物理票据的终态。注入准备失败、部分 Submit、等 Supply、Commit 后断开和迟到通知。健康域仍完成请求 |
| 既有产品路线 · 回归 | 单卡本地融合与 offload、单卡生成和评分，Qwen3.5 已接通的独立 DP，现有容量/图行为。构建分布式开关两侧，对受影响入口运行原合同测试 | 数值、状态和协议按原标准通过。既有性能在同配置实测。单卡与多副本分别在能运行该部署的环境验收。[已有测试入口](../../tests/README.md) |
| 双 Spark · Qwen4Exp 跨副本 EP | `dp2/tp1/adp1/ep2`，固定 Disk PLE 以复现这条负载。**显式 Native 和 NCCL 各跑**单源、交换源端、双源不同历史、热点对端、长 Prefill 对短 Decode、MTP3 接受/拒绝、Grammar、取消和多段评分。分别走 Eager 和适用的 Graph 生成路径 | 有真实的 compact 接收、服务票据和回传。核对层级数学、逐 token 分数、下一步状态、实际后端和图回放，零源无请求访问。验证 `0/1/3` 时用可控路由的生产算子夹具，真实 checkpoint 另记录其实际命中，不依赖它碰巧产生指定路由 |
| Spark · Qwen4Exp PLE | 单 rank 的 Disk、Device 整表，`dp1/tp2/adp1` 的 Disk、Device 分片和 Device 整表。选可容纳该候选的上下文与并发，并在同一 TP 的介质对照中保持一致。另用约束资源用例验证显式失败与 auto 选择 | 表覆盖、窗口、embedding 位模式、接受后历史精确一致。核对 Host、设备/UMA 账、读取/上传和归并次数。两副本不同 raw tail 的同号列始终得到各自的结果。资源不足用例报告实际缺口 |
| 双 Spark · GLM5Next 回归 | `dp1/tp2/adp2/ep2` 和副本内专家 TP 的 `ep1` 对照。Native 与 NCCL 跑已有适用的 Packed、属主迁移、MTP、DFlash2、评分和图路径 | 基线按各自 artifact 和配置复现。缺失的 Spark 局部形状核查原始 Op 错误。该设备的 GLM TP1 不作为跨副本 EP 实机验收门槛 |
| 目标八芯 · GLM5Next 整模 | `dp1/tp8/adp1/ep8`、`dp8/tp1/adp1/ep8`、`dp4/tp2/adp1/ep8`、`dp4/tp2/adp2/ep8`。再以 `dp4/tp2/adp2/ep2` 检查独立副本，`dp2/tp4/adp2/ep2` 检查部分 ID 与部分 intermediate 共存。用目标已有算子、匹配的实际产物和后端 | 从公共 Engine 观察真实 Main/MTP、相应草稿组件、Packed、评分、属主迁移与状态采用。核对 TP 复制列唯一派发、shared 只贡献一次、多卡属主页容量和故障处理。输出此平台的数值、峰值资源和性能结果 |

**没法在开发机跑完整前向时，仍要验证模型调度。** 使用实际 GLM 定义、rank 计划、Session 和 Work 构造器，在目标几何上冻结本轮成员，经过生产 codec 记录 Worker 将收到的行、属主页、专家阶段及目标宽度。用受控的完成、逐行接受与丢弃事件驱动真实 Session，核对下一步前沿和资源释放。只替代设备数值结果，不另写一套调度模型。这项证据支持该几何的模型调度结论，整模数值与实机运行另由目标设备给出。

**数值与生命周期分开判定。** 专家用独立参考从实际权重及其量化参数计算全局路由、逐专家输出、shared 和模型收尾。离散 ID、同分排序、有效列、无效槽、逆映射和写入域精确比较。浮点按相应 Op/表示预先规定的参考和容差，包括热点、零命中、多专家同目的地、远端全命中、本地切片和非有限值检查。端到端固定输入检查 logits、逐 token 分数、MTP 部分接受后的 KV/State/PLE 及下一次前向。CPU 服务用例对请求页设为不可访问。Spark Eager 与 Graph 为服务 Work 只绑定协议缓冲，用不可访问请求地址或设备访问检查，另验活跃行的边界。对无读写权限的验证不能只靠观察输出为零。[专家参考](../../tests/ops/moe/expert_oracle.h) · [MLA 参考](../../tests/ops/softmax_attention/kpool_mla/reference.h) · [数值规范](../maintainer/op-development.md)

**性能与资源按负载解释。** 比较前固定 artifact、权重表示、草稿头和深度、上下文、并发、输入、后端、图模式、预热和重复次数，用既有路线重复运行的分散程度约定可分辨的差异。对单源、均衡双源、路由倾斜和 Prefill/Decode 混合分别报告 TTFT、逐 token 延迟及 p95、Prefill/Decode 吞吐、MTP 接受率和每轮有效输出数。各 rank 同时给出来源列、接收唯一行、各专家 assignments、逻辑/物理收发字节、整理/通信/计算的关键路径时间、Graph/Eager 次数、工作区峰值、表和可用 KV 容量。发生差异可以据此区分传输填充、计算不均、PLE 交接与调度等待。可复现的回退写明原因或待定位项及处置，不以单个 kernel 的提速代替 Engine 结果。[基准入口](../../bench/README.md)

**收口方式。** 报告为每个配置分别列出“引擎协议通过”“模型计划与调度通过”“该平台整模通过”及其实际证据。出现拒绝时注明是部署几何、模型布局、具体 Op/后端或资源容量。八芯使用已有算子并在目标机完成最终组合验证，十六 rank 的 CPU 用例只覆盖计划与协议。Spark 上的开发工作集中在两模型共用的专家机制、Qwen 的真实前向和已有 GLM 路径的回归。
