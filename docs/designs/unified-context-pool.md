# 统一 Context 存储池

本文定义选定的重构目标：KV history 与请求 state 共用底层容量，保留各自的对象语义，
通过执行视图隔离物理分配。启动验证完整工作集，运行按执行链成组预留，缓存按真实释放结果让出容量，
调度负责在压力下保障请求推进。现有固定 Device KV pools、State slots 及相关容量合同将整体替换，
不保留旧接口或兼容执行分支。本文是设计与实施依据，不表示新池已经实现或通过性能验证。

所有权遵循 [Engine 架构](../maintainer/engine-architecture.md)。现有行为参考
[Context 缓存](../maintainer/resource-scheduling-and-context-cache.md)与
[Paged KV](../maintainer/paged-kv-cache.md)；其中固定分区、物理 slot 与固定池 Tensor 的约定，
在本次重构范围内由本文替换。模型数学、量化编码及 checkpoint 的数学条件继续由各自合同维护。

## 1. 目标、对象与边界

每个 rank 的 Device Context 有一份固定 backing 和总容量。KV 与 state 按实际对象需求占用，
不在启动时分别预扣 KV 页数和 state 数量。RTX 5090 的 Host Context 是另一个容量域；
Spark 不建立重复的 Host Context cache。不同 rank、不同存储域的空闲量不能相加冒充本地可用量。

| 对象 | 语义 | 分配与修改 |
|---|---|---|
| KV history | 按原 token ordinal 增长；前缀共享、分叉、截断；不同 backend 可以有不同 frontier | 按逻辑页增量申请；引用保护共享内容；写入冲突时做必要的尾页 COW |
| State | 表示某个 token 位置的完整续算状态 | 单份大小由布局确定；当前版本可以更新；checkpoint 冻结；后续独立写入需要独立对象 |

State 可以包含 GDN/KDA recurrent state、convolution history、Indexer tail、PLE convolution history、
续算 hidden、DFlash cyclic KV 等。字段不是内存池单独调度的对象。

不进入本池的内容包括权重与 expert cache、PLE embedding 表及其 I/O cache、Vision 临时 K/V、
activation/workspace、ReplaySSM transition records、draft 临时 tail/hidden/selection、采样缓冲。
它们有独立的数学或执行生命周期，仍由原 owner 显式规划。PLE 的跨 token convolution history
属于 State；PLE 的大表属于权重放置。

本设计保持三种独立粒度：逻辑 KV 页、逐 token 的有效 frontier、模型允许恢复的 checkpoint。
池内大块不改变其中任何一种。当前逻辑 KV 页仍为 64 token；每个模型实例的尺寸来自 config、
selected backend 与 rank placement，不成为引擎中的架构常数。

## 2. 所有权

```mermaid
flowchart TD
    M["模型：字段、历史、frontier 与 rank placement"] --> S["ContextDirectory：对象、引用、事务"]
    Q["Scheduler：准入、执行成员、抢占与恢复顺序"] --> M
    C["Context cache：来源选择、保留与淘汰"] --> S
    S --> P["ContextPool：记录分配、预留、物理绑定与整理计划"]
    S --> X["Executor：有序提交与消费者退休"]
    X --> R["RankProgram / ContextStorage：backing、搬运、执行表"]
    R --> O["Ops：计算视图与数学"]
```

| 所属模块 | 唯一负责的事实 |
|---|---|
| 模型 Definition / Session | 字段与 history 几何、放置、Main/backend 前沿关系、完整恢复条件、启动工作集与执行链的峰值需求 |
| ContextDirectory / KV、State 管理 | 对象身份、引用、写权限、版本、replica、历史覆盖、成组预留、执行绑定和事务 |
| ContextPool | 某个容量域内的记录位置、空位、大块归属、预留、可行性与物理整理计划 |
| ResourceManager | 根据实际恢复价值和可回收结果选择 checkpoint/continuation；不解释池内大块 |
| Scheduler / EngineCore | 活动集合、请求优先顺序、准入、抢占与恢复机会；消费 Session 的需求评估，不推导模型前沿或记录大小 |
| Executor | frame 与后续提交额度、ticket、依赖、维护窗口、各 rank 完成与退休；不选择缓存受害者 |
| RankProgram / ContextStorage | 拥有 backing，兑现字节操作与视图更新；不再选择物理位置或淘汰对象 |
| Op | 计算语义、视图和 alias 合同、支持范围、实现分派；不拥有对象或分配器 |

集中调度下，ContextDirectory 持有各 rank 的分配账本；rank 仅执行确定的 offset/length 动作。
本地执行采用相同关系。池中的记录不保存 token、模型层号、checkpoint 价值或并行策略。
模型向存储声明布局，存储向池提供字节数和对齐，是公开合同；池的块编号、增长方向与空闲链表
不进入模型、缓存策略或 kernel。

## 3. 身份、存储布局与借用

### 3.1 身份与物理位置分开

- `StateId`、`PageId`、`HistoryId` 是逻辑身份，带 generation；分叉与物理移动不混为一件事。
- 每份物理 replica 持有一个 `AllocationId`。它属于确定的 rank/存储域，移动时 ID 不变。
- 分配器维护 `AllocationId → {class, byte_offset, binding_epoch}`。数学写入更新 content epoch；
  物理移动只更新 binding epoch。
- 调度、缓存和请求长期持有对象身份；物理地址仅在执行准备或传输准备时解析。

引用保证对象存在并维持读写语义；执行借用保证消费者期间地址稳定。普通 checkpoint 引用
不永久锁住物理位置。GPU 计算、Host DMA 和物理复制的借用延续到相应 ticket 完成。
最后一个逻辑引用结束时，仍被执行借用的记录进入待退休集合；Directory 在借用结束后才调用池的
`release`。释放对象身份、归还物理位置和内容提交是三个不同动作。

### 3.2 自然对象布局

State 采用一份连续的 rank-local image。字段及层内矩阵保留固定相对偏移与自然连续轴；
允许复制指定字段组。是否需要复制、哪些字段由下一次计算覆盖，由模型执行语义决定。

KV 管理将同一逻辑页组内、共享 frontier 与生命周期的全部 planes 打包为一个连续记录。
记录内按 layer/plane 保存 codes、scales、K/V 或 latent、pooled index keys；每个 plane 的布局
来自对应 codec。此选择便于页复制和搬移，属于 KV 存储实现，不是分配器要求。
Main、MTP、Draft Full 保持独立 history 与记录布局，不为凑大块而合并生命周期。

保留每个 plane 内现有的 feature/token/head 顺序；仅改变页组之间的定位。对于固定层，跨页访问
将跨过其他层的 payload，区别于原先按 plane 分开分配的数组。页内合并访问可以保留，但跨页地址
局部性与 TLB 行为需要在长上下文 Attention/Indexer 中验证，不能只凭逻辑表相同就认定性能相同。

首版字段/plane 对齐为 256 B，沿用现有布局生成与 packed Host page 的规则。
记录布局必须计入 plane 间 padding，不能只累加每 token 的数值载荷。
Device/Host 使用同一记录字节表示，迁移不重新量化。

GPU 执行表与 valid frontier 不放进 KV payload；请求/版本元数据也不放进 State payload。
State 中只有数学消费者需要持久化的字段。已有 CPU token ledger 能恢复的 PLE token IDs
继续作为 frame 输入准备，不额外复制进每份 StateImage。

### 3.3 控制侧接口轮廓

以下为目标职责与类型关系；完整声明由对应模块实现，不引入公共 SDK：

```cpp
struct RecordSpec {
    std::size_t bytes;
    std::size_t alignment;
};

struct RecordRequest {
    RecordClassId record_class;
    std::uint32_t count;
};

struct RecordMove {
    AllocationId allocation;
    std::uint64_t source_offset;
    std::uint64_t destination_offset;
    std::uint64_t bytes;
};

class ContextPool {
public:
    PoolAssessment assess(std::span<const RecordRequest> demand,
                          std::span<const AllocationId> releases) const;
    PoolReserveResult reserve(std::span<const RecordRequest> demand);
    void release(AllocationId allocation);
    RecordBinding resolve(AllocationId allocation) const;
};
```

`assess` 只预览可行性与搬移需求，不删除对象、不提交 GPU 工作。`reserve` 对整个 demand
全成或全败；成功返回拥有预留的事务，可能需要先执行整理，才允许解析新记录的可用视图。
`resolve` 不隐式分配、搬移、同步或改变权限。尚未就绪的预留不能得到可使用的绑定。
缓存策略通过 Session 查询这些结果，不直接持有 `RecordClassId` 或 `AllocationId`。
逻辑需求可以列出 State/history，但这些用途不再对应独立的物理容量或 shortage 计数。

## 4. 统一池的分配算法

### 4.1 固定记录尺寸、动态大块归属

模型实例准备时注册有限的 `{bytes, alignment}`，相同规格复用尺寸类别。运行时增加对象数量，
不临时增加格式类别。类别 ID 是分配器内部身份，不代表 KV/State 的语义类型。
记录 bytes 为正，alignment 为二的幂；backing 基址与块起点满足所有已注册对齐。

令类别 i 的对齐后记录跨度为 `s_i`。首版采用：

```text
A = max(record alignments)
B = align_up(max(s_i), A)
N = floor(context_capacity / B)
q_i = floor(B / s_i)
r_i = B - q_i * s_i
```

每个大块是以下两种状态之一：公共空块，或属于某个尺寸类别、容纳 `q_i` 个记录的块。
不要求同类大块相邻，不存在 Main/MTP/State 区域顺序。`B`、`q_i`、大块编号仅由池解释。
最大记录一块一份，因而现有模型的完整 State 通常无需为整理而移动。

该 B 是确定的首版算法，后续可以仅在分配器内改进选块规格；不能把 B 传给算子，或要求模型
为提高装载率调整矩阵、KV 页大小、checkpoint 位置。首版不增加可变大块、跨块记录或多套后备分配器。

### 4.2 普通分配与释放

1. 在同类别已有空位中分配，优先填充占用较高的块；批量申请在块内尽量形成连续 run。
2. 不足时，从公共空块取得任意块并赋予该类别。
3. 普通释放只归还记录；块完全空闲时立即回到公共集合，不做 payload 复制。
4. 同一类别可保留多个有空位的块，但不能为了未来可能的需求私扣空块。
5. 未完成消费者或未完成迁移占据的记录仍计为占用，不作为现在可用的容量。

索引按类别维护非满块、空位及占用计数。正常分配不扫描全部对象；物理整理的候选搜索只在压力时进行。
元数据按记录管理，不建立逐层、逐 head 的物理分配记录。

### 4.3 整理与容量交换

当公共空块不足，且某类别的总空位允许归还完整块时，选择该类别中存活记录较少的块作为 donor。
先确定 donor 集，再将其记录安排到保留块的空位；目标不能又成为本批 donor。
所有源区间与目标区间互不重叠，复制可批量并行，没有重叠 `memmove` 或搬移循环。

对一个类别，若总空位至少为 q，选定 donor 中有 h 个空位，则外部至少有 `q-h` 个空位，
足够承接 donor 的 `q-h` 个记录。回收一块的 payload 搬移量 `< B`。
回收 k 块时搬移量 `< kB`；这是可移动记录满足条件时的上界，不包含等待、表更新和提交成本。

整理只为满足当前成组需求执行足够数量的块，不在每次释放后全池压实。正在使用的对象不能边写边搬。
如果空间只有在消费者退休后才能使用，返回等待条件；如果缺少足够的可转换空位，返回回收压力。

### 4.4 空间损失与可行性

类别 i 有 n_i 个记录，充分整理后需要：

\[
m_i=\lceil n_i/q_i\rceil,
\qquad C_{used}=B\sum_i m_i.
\]

损失包含各块 `r_i`、各类别最后一个未满块，以及 backing 的不足一块尾部。
类别尾部损失在可充分整理时每类小于 B；每块 r_i 则随容量累积。借用未结束时，暂不可转换的空位可能更多。

本方案不保证任意空闲字节立即互换。例如 Qwen4Exp TP2/BF16 下，Main 空出 40×816 KiB，
MTP 空出 500×68 KiB，合计 65.078 MiB，仍不足以分别清空一个 69-record 或 832-record 的块。
没有公共空块时，不能据此承诺一份 55.268 MiB State 的申请成功。

因此准入和回收使用池的成组可行性，不使用 `free_bytes >= requested_bytes` 代替。

## 5. 执行需求、事务与异步管理

### 5.1 完整执行需求

一次需求对应可独立提交并完成的工作，而非某次底层 `allocate`。Session 根据原生工作与当前来源
给出 State 的读写关系、各 history 的实际读写前沿、恢复与迁移目标及生命周期；Directory 据此解析
现有对象复用、实际 COW、新增记录和需要保护的来源。两种模型共享这个管理过程，各自保留数学需求生成。

| 需求内容 | 约定 |
|---|---|
| 已有对象 | 借用已有 allocation，不重复计费；同一物理对象的多个引用去重 |
| State 写入 | 区分原位 writer、接管、读源写目的、Host 恢复；不统一假设每次都新增一份 image |
| History 写入 | 按最大实际写入 frontier 计算增长和受保护范围冲突产生的 COW；不只计算提交后的长度 |
| 后续阶段 | 包含本轮 draft、target、决定、fold、canonical 补齐和 commit 所需的 Context 资源 |
| 物理生命周期 | 源和目标同时存活的期间都计费；仅在真实消费者退休后才允许复用其空间 |
| 执行资源 | frame、后续 supply/commit/discard、复制描述符和 staging 由 Executor/RankProgram 保障 |
| 可选保存 | 单独作为可撤销附加需求，不混入必要执行的最低需求 |

Session 保留模型相关的需求描述；Directory 将其转换为各容量域的记录请求。执行许可持有对应的 owning
reservation，Work 只携带已确定的值描述。完整执行开始后，commit 或 discard 不再申请未被许可覆盖的
Context 空间。允许按明确的退休顺序复用资源，但不能把尚未完成的释放提前抵扣。

缓存来源、引用保护、执行窗口或成员改变后，重新评估受影响的需求。不能先按未共享尾页取得许可，
再添加 checkpoint 引用，使已承诺执行突然需要额外 COW。没有这些变化的普通 step 只计算实际增量，
不扫描全部缓存或重建整个 history。

### 5.2 分配与整理的提交顺序

```text
语义需求 → 成组预留 → 必要的存储整理 → 绑定/初始化就绪 → 数学执行 → 内容提交 → 释放
```

成组预留先取得所有目标记录、搬移目的空位和待回收块的独占权。未提交前取消，完整撤销元数据预留。
整理提交后，即使请求取消，也要完成已经提交的字节移动和位置发布，再释放不再需要的预留。
不能通过回滚 CPU 地址记录，假装 GPU 没有执行过移动。

Executor 的执行额度与 Context 预留在一次不等待的准备尝试中协调取得。必要资源未取得时，撤销该次
未提交的临时占位，交回明确的压力或等待原因。需要整理/交换时，先保障该存储操作的执行资源，再提交；
不能占着计算 frame 等待一个必须借用同一额度才能完成的回收操作。受保护恢复的长期预留见 §6.6。

预留对象明确处于 `Reserved → Submitted → Ready → Consumed/Released` 生命周期；未提交时允许
直接取消。Directory 持有预留，Executor 持有提交 ticket，完成事件交回 Directory 后才发布或释放。
被选作 donor 的块及其目标空位期间不接受普通分配，物理复制不与后续数学写入并行。

物理整理完成顺序固定为：复制 payload → 更新受影响执行绑定 → 确认完成 → 发布新位置、归还旧块。
期间旧位置不能交给其他 writer，新位置也不能提前给计算使用。content epoch 保持不变。
物理执行错误沿现有实例失败/排空路径传播，不在地址状态未知时继续调度。

### 5.3 借用与整理窗口

首版在该 rank、该存储域的既有计算/传输消费者退休后执行整理。Executor 用已有 ticket/event
建立顺序，整理期间取得该池的独占维护权；不逐页创建事件，不在普通 decode 中增加全设备同步。
没有整理需求时不获取独占维护权，原来的计算与传输调度继续适用。

取得执行借用先于解析地址和填充 frame。进入维护窗口时，先关闭该域的新借用，再等待既有借用退休；
已经准备但尚未提交的物理 frame 同样持有借用，取消时显式释放。期间可以继续准备不含物理地址的
逻辑计划，不能提前解析后续执行的 State offset。整理发布后重新开放借用。
既有已准备 frame 必须按原依赖提交或取消，不能等待维护结束才处理，从而阻止自身借用退休。

这项选择避免为每次 decode 遍历整个历史来 pin/unpin 所有物理页。执行借用以 frame/history binding
为单位登记，State 的源/目的按当前 batch 登记。逻辑引用仍按对象维护，不与维护窗口混用。
维护需要等待尚未结束的传输，是明确的成本；首版不增加边计算边重定位的复杂路径。

### 5.4 执行表与元数据

只有 active execution row 持有 GPU KV 页表，inactive cached history 只保存逻辑 PageId 序列与覆盖信息。
新绑定/恢复建立一次执行表；增长、COW、截断和搬移只更新受影响位置。

存储绑定维护 `PageId → active table locations` 的反向关系。它只覆盖当前绑定，不为缓存中的每个
共享前缀复制一份 GPU 表。搬移修改该页的已登记位置；State 只需在后续 frame 准备中解析新地址。
不在每个 step 重新翻译全部 cached pages，也不让 kernel 逐层追踪 AllocationId。

目录使用稳定 ID 与可增长的分段元数据。元数据预算由总池容量、最小记录规格和请求/目录合同估算并计费，
统一分配记录描述符，不为每个尺寸类别分别预分配“独占整个池”时的最大数组。
原先固定 state slots、physical pages 派生的逻辑目录上限同时退出。
共享引用可以产生多于物理记录数的逻辑 histories/checkpoints，其目录预算由缓存目录单独限制；
不以“未新增 payload”为由无限增长 Host 元数据。Spark 上这部分预算在创建 Device backing 前扣除。
逻辑目录耗尽与 payload 空间不足分别报告。回收无用 checkpoint 元数据也不等于释放了其引用的共享页。

## 6. 请求准入、缓存与持续推进

### 6.1 State 与 KV 的申请、归还

StateDirectory 管理逻辑版本与 writer；HistoryDirectory 管理前缀覆盖、分叉与写入范围；两者的
物理记录都来自 ContextDirectory 持有的统一池。模型提交完整需求后才物化这些记录，不再由两个
子目录各自消耗一份独立配额。

| 操作 | State | KV | 结束时的处理 |
|---|---|---|---|
| Root 绑定 | 申请 writer 并建立模型初值 | 申请首个合法执行窗口所需页 | 预留移交实际执行 |
| 接管已有 continuation | 权限允许时接管原 writer；不增加 image | 接管 history，补齐增长/COW | 来源 owner 的移交与绑定一起提交 |
| 共享前缀分叉 | 保护不可变来源，取得独立写目标 | 共享前缀，实际写入冲突才 COW | 原来源继续由剩余引用持有 |
| 普通 prefill/decode | 独占 writer 可原位；已有独立目标则使用目标 | 预留本次写入峰值的增量 | 提交真实前沿，退休未使用预留 |
| 中途 capture | 先取得后续 writer 的资源，再冻结当前版本 | 保留精确历史前缀 | checkpoint 与下一 writer 同时取得正确所有权 |
| Speculative round | 保护 committed source 和必需写目标；逐 token records 留在 frame | 为候选执行、补齐与 COW 预留峰值 | 采用前缀后裁剪；最后消费者结束后释放其余记录 |
| 请求结束并保留 | 当前 writer 可直接冻结并移交 | 移交 prefix 引用 | 不为结束事件强制复制整份数据 |
| 取消、放弃保留 | 释放 writer/引用 | 释放 owner、增长预留及不再需要的覆盖 | 在途借用进入待退休集合 |
| Host 迁移/恢复 | 源、目标 replica 在传输期间同时计费 | 同样按实际覆盖与副本计费 | 完成后按事务保留或移除源副本 |

State 最后一个 owner 结束且没有 reader/writer/transfer 借用时，归还其记录。KV 只有在所有相关
history/prefix 都不再需要相应覆盖、且物理消费者退休后才能归还；请求结束不代表共享页可以释放。
截断部分尾页只改变可见覆盖，仍有需要的内容时不释放整页。

模型拥有 Main/MTP/DFlash 的前沿关系；GLM 的 `MTP frontier=max(F-1,0)` 不进入池。现有读源、
写目的的 GDN/KDA 路径保留，不能因新存储形式无条件增加一次完整 State 复制。QSA/MLA 本轮未选中
的活跃历史页仍可能被后续 query 选中，不属于可以按本轮 selection 随意丢弃的缓存。

### 6.2 必需资源与可撤销缓存

管理区分活跃请求必需对象、在途借用、已承诺预留、可撤销缓存与空闲空间。这些是生命周期和权限，
不建立物理分区。同一对象既被缓存引用又被实际执行依赖时，执行依赖优先；不能按两个用途重复计费，
也不能仅删除缓存引用就声称得到该对象的全部容量。

当前执行和受保护恢复已经取得的 reservation 不被新请求或可选 capture 消耗。尚未提交的可选
保存可以取消；已经提交的动作必须先完成其退休流程。缓存保留优先级只影响受害者选择，不形成永不可撤销
的额外容量承诺。

空闲容量允许用于 checkpoint、共享前缀及 continuation，不长期空置一个固定 State/KV 安全区。
启动工作集保证见 §10.1；运行时依靠真实预留、可撤销缓存和必要抢占实现该保证。

### 6.3 来源选择与准入

空 lane 只是请求绑定的必要条件。ResourceManager 提供有恢复价值的候选，Session 对每个候选生成
完整绑定需求：需保留的 source、State 接管/分叉/恢复、各 history 覆盖及首个实际执行单元。
被抢占请求从旧 checkpoint 重建时，同时遵循 §6.6 的恢复范围。初次请求从命中前缀继续 prefill，
仍按下一合法单元增量预留，不把尚未计算的整个 prompt 作为受保护恢复范围。

来源比较继续使用已有复用证据和恢复成本，并计入实际传输、COW 和新增记录。较长命中不自动代表
该候选可行；暂时被其他活跃请求阻挡，也不直接把有价值来源丢弃并改为从头 prefill。Session 区分
等待容量变化与该来源方案本身超出容量，ResourceManager 再选择等待、回收或其他合法来源。

绑定的 Context 预留可以移交后续执行；frame 额度按实际提交取得，绑定期间不长期占用未来计算 frame。
等待请求持有的来源选择是可撤销保留，不等同于已提交消费者的地址借用。正式绑定/迁移开始时才取得
相应读 lease，防止等待队列把整个缓存永久钉住。

每轮先保障已选择的必要执行，再评估新请求与可选保存。Scheduler 决定成员，Session 汇总其实际需求；
目标 batch 放不下时，可以先执行已能取得完整许可的成员。失败成员不留下零散 State/KV 占位，也不
取消其他成员已有的许可。没有为每个请求提前占满 `max_context` 的要求。

### 6.4 统一压力评估与回收

Session 对外提供以下结果和涉及的容量域；Pool 的类别与块位置留在存储内部：

| 结果 | 后续动作 |
|---|---|
| 可立即预留 | 原子取得完整需求 |
| 等待消费者/操作完成 | 返回实际 ticket 或释放依赖；完成后重新评估 |
| 需要物理整理 | 取得维护执行资源，完成足够数量的局部搬移 |
| 需要回收或减少活跃工作集 | ResourceManager 选择可撤销对象；必要时 Scheduler 抢占 |
| 独立工作集超出总容量 | Session/资源实现报告该方案不可行；选择其他合法来源或向调用方传播原因 |

另行区分执行 frame 忙、目录元数据不足和 payload 不足，不能全部报告为“缺 KV”。只有存在具体
完成条件时才等待。某个请求被其他活跃请求阻挡，不等于它独立运行也超出总容量。

回收流程如下：

1. 撤销妨碍必要执行的未提交可选预留；退休已完成的消费者，更新真实可用量。
2. ResourceManager 按保留价值、近期复用和恢复损失排列可撤销 checkpoint/continuation 或副本动作。
3. Directory 为候选组合建立释放 inventory，纳入全部 State 和 history；按所有实际持有者去重，
   排除执行来源、在途消费者及其他仍存活 owner 的内容。
4. 累积评估候选集合，调用统一池判断完整需求是否可满足，同时取得必要整理/迁移的工作描述。
5. 一旦可满足就停止扩展受害者集合，完成释放、预留和存储动作。状态在评估后改变或异步返回后，
   重新取得有效评估；准备提交在协调侧串行完成，其他准入不能抢走刚为本次需求回收的容量。

保留现有 `ReleaseInventory` 对共同持有者的证明能力，去掉按 `shortage.state_slots` 或某个
`kv_page_groups` 过滤对象的做法。单个候选暂时不能归还公共块，不代表其释放无用；几个候选的合计
可能刚好让一块可归还。不能只按独立候选的即时通用空闲增量过滤，也不能累加共享对象的名义大小。

停止条件是原完整需求已经可满足，不是释放了指定数量的 State 或某类页。用于排序的容量收益采用
相关容量域内的真实释放结果，并结合完整组合的可行性；某 rank 的释放不能抵扣另一 rank 的不足。
不用固定“先 State、后 KV”的顺序，也不在每删除一个 checkpoint 后立即整理全池。

### 6.5 checkpoint 与剩余容量

语义 checkpoint、稀疏中间 checkpoint、共享 system prefix 和最终 continuation 使用同一容量。
`checkpoint_interval` 仅产生保存机会，不再导出启动 State 数量。保留决策使用恢复价值与新增的真实
物理占用；多个 checkpoint 共享的 KV 只计一次，同时考虑它们延长覆盖和引入未来 COW 的影响。

中途 capture 按以下顺序准备：

1. 确定合法 frontier、需要保留的完整 State/history 及其来源身份。
2. 预览冻结来源后，当前请求的后续 writer、实际写入范围和 COW 需求。
3. 在已有必要许可之外取得附加资源；若改变已有许可的前提，先完成许可的原子调整。
4. 资源和执行条件都具备后才冻结/发布；失败时保留原先可执行的 writer，跳过保存或按保留价值替换旧点。

Device capture 可以冻结当前 image，并让后续数学直接读源写目的；Host capture 的接管方式见 §9.1。
都不要求无条件预复制整份 State。请求最终结束时不再需要下一 writer，可以直接移交当前版本。
保存价值不足或无法取得附加资源时，跳过本次机会，不要求抢占活跃请求来完成可选保存。

原有按各独立池占用比例保护缓存的规则退出。近期被复用的内容仍可优先保留，但执行压力下保留等级
不能成为不可撤销引用。大 State 较多的短请求负载可以占用更多统一容量；长历史负载可以通过撤销
低价值 checkpoint 将容量交回 KV。比例来自实际请求和引用关系，不再预先固定。

一个直接恢复命中仍需完整 State、各 history 的必要覆盖以及正确的模型/backend/media 身份和 placement。
撤销 checkpoint 时同步撤销相应恢复资格；共享 KV 若还被其他合法来源持有则继续保留。物理搬移只改变
位置，不改变 hash、frontier、命中长度或数学版本。

### 6.6 压力抢占与恢复进展

若可撤销缓存仍不足，Scheduler 减少活动集合，优先保障较早请求的必要执行。暂停只发生在已退休的
提交边界；Directory 先撤销其未提交许可和可选预留，再按模型支持的恢复方式处理对象。

- Host 可用且保存方案可行时，预留完整 Host 目标，完成快照迁移后释放 Device 副本。
- Spark 或无法完成快照保存时，释放相应计算状态，保留逻辑请求与重放所需信息，后续从合法来源 replay。
- 保存快照若需要 backend normalization 或新增页，也必须先取得完整许可；无法取得时采用模型支持的
  重算恢复，不以“为了回收”作为越过预留的理由。

恢复的保护范围为：从选定来源恢复到被抢占时的进度，再完成至少一个新的必要执行单元。Session 计算
这一范围内各 history、State 与转换阶段的峰值，Directory 持有相应预留；不能每次只保留一个 replay
chunk。可以逐步物化已经预留的记录，但保护不能在刚恢复旧进度、尚未完成新工作时消失。
已经终止或达到输出/上下文限额的请求走结束路径，不为它预留超出语义上限的新 token。

受保护恢复期间跳过会增加 State/COW 需求的可选 capture。保留现有较早请求优先、恢复优先于较年轻
请求抢占的关系；资源条件没有变化的失败不忙重试。被抢占请求重新准入时再次检查完整恢复范围，
避免恢复/抢占同一请求却没有新增进度。

持续推进的条件是：已提交消费者能够退休，可撤销 owner 按权限释放，必要时其他活跃请求可以暂停，
而至少一个剩余合法请求的独立执行路径满足启动工作集保证。Source 方案若因必须并存的副本过大而
无法成立，允许选择其他模型支持的来源，包括重算；不能把某个缓存分叉方案不可行当成所有请求都不可运行。

### 6.7 其他执行与结束路径

| 路径 | 必须保持的行为 |
|---|---|
| MTP / DFlash / DFlash2 | 提交采用的真实前沿；canonical 补齐的容量已取得；未采用页和临时 writer 在消费者退休后归还 |
| Vision 请求 | 按模型合法窗口和 checkpoint 边界规划；释放 Context 后仍保留重放所需媒体/位置来源，其内存由原 owner 计费 |
| Causal scoring | 通过相同 State/KV 成组预留完成每个工作单元；不能继续在独立 score 路径直接向各页池申请 |
| EOS / 正常完成 | 按策略移交最终可保留来源，释放其他 owner 与未消费预留；完成通知不代替设备退休 |
| 取消 | 未提交预留可撤销；已提交事务完成其复制/整理与退休，再决定发布或丢弃 |
| 执行错误 / 退出 | 沿既有失败与排空流程终止提交；地址或内容状态不确定时不继续复用记录 |

Token ledger、位置和媒体身份、已输出内容、采样及协议状态由各自 owner 保留。它们不随 Context
payload 释放而清除，replay 不重新发布已经输出的 tokens。PLE token IDs 仍可从 ledger 准备；
GDN/KDA、Indexer tail、PLE convolution history 等数学状态随完整恢复来源或重算建立。

## 7. Op 访问合同

### 7.1 通用视图

Tensor 继续表示普通仿射张量。混合存储池不能伪装成一个覆盖所有 allocation 的固定 slot Tensor。
新增的非拥有视图表达记录寻址；其内容不包括分配器 ID、slab 大小或引用计数。

```cpp
struct RecordBatchView {
    MemorySpan storage;    // Stable rank-local arena.
    Tensor byte_offsets;   // Device I64 [B], one resolved record offset per row.
};

struct StateFieldView {
    RecordBatchView records;
    StateFieldLayout field; // Dtype, inner shape/strides, layer stride and record-relative offset.
};

struct PagedRecordView {
    MemorySpan storage;
    Tensor page_offsets;   // Device I64 [logical_page_capacity, table_rows].
    std::uint32_t page_tokens;
};

struct PagedPlaneView {
    PagedRecordView records;
    PagePlaneLayout plane; // Dtype, record-relative offset, rows/heads and byte strides.
};
```

State 批量输入直接提供本轮 B 行的地址，不再附加“对象 ID → 物理 slot”的 kernel 查表。
KV 表直接存整页组的 arena byte offset，K/V/scale/index planes 共用该表，通过固定 plane 偏移解释。
记录内部的 head/feature/row 连续性仍由数学布局约束，池内大块大小完全不可见。
kernel 在已有行/页定位处读取一次 offset，再按原 tile 访问；不能在逐元素内层重复查表。
以 B=8、262144-token 上限计，一个完整 history 的 I64 页表为 256 KiB；1M 上限为 1 MiB。
这是活跃执行表的大小，与池中保留多少 inactive histories 无关。

```text
state(row, layer, element) = arena + row_offset[row] + field_offset(layer, element)
kv(row, page, plane, item) = arena + page_offsets[page,row] + plane_offset(plane,item)
```

只读接口使用对应的 const 访问权限。源、目的状态提供独立 offset 表；合法原位更新时两者可以相同。
未启用行使用无效 offset，消费者必须先按有效行/列屏蔽，不得提前解引用。有效行的 offset 必须落在已借用记录内。
Op 校验 Host 可见的 dtype、shape、stride、对齐与布局；生成设备索引的调用方保证值域、权限和写入分离，
不通过 D2H 检查每轮设备索引。池与目录不复制 Op 的数学支持限制。

共享 arena 的两个视图可以合法共存。Alias 判断针对实际访问的记录和字段，而非整个 arena 的包围区间。
独立 writer 的目标记录必须互不重叠；共享只读来源允许复用；源/目的能否原位由具体 Op 合同决定。

### 7.2 必须迁移的消费者

| 消费者 | 改动 |
|---|---|
| Dense GQA append、causal/context attention、Draft Full attention | 页号乘固定 page stride 改为页记录 offset；所有 KV 类型一致迁移 |
| QSA prepare、Indexer、sparse GQA、tail extract | 共享页记录表；tail 改为 StateFieldView；selection 仍是原逻辑位置 |
| Kpool prepare、Indexer、sparse MLA、tail extract | 采用同一资源视图；latent 与 pooled key 共享逻辑页组；owner-only 执行不变 |
| GDN input-proj-conv、causal conv、recurrent/chunked、record/fold | 初始/最终 state 解析为字段视图；内部连续矩阵与数值语义保留 |
| KDA 前置、recurrent/chunked、ReplaySSM record/fold | 使用同一 State 字段/批量地址合同，不复制 GDN 的模型流程 |
| DFlash/DFlash2 cyclic KV | 整份循环状态属于 State，按 State 行地址取得本层 K/V；窗口和 ordinal 算法不变 |
| PLE history 更新与 tail 提取 | 跨 token history 使用字段视图；来自 token ledger 的短 token tail 仍是 frame 输入 |
| continuation hidden、MTP conditioning gather/scatter | 使用实际字段地址，不再把 slot 当作物理位置 |
| copy、clear、Host transfer、Graph 初始化 | 消费 allocation 绑定或字节区间，不遍历旧固定 typed pool |

Replay fold 的长期 Plan 绑定记录缓冲、字段几何及固定地址的设备 offset 表，不缓存某份 State 的永久 payload 地址。
Host 便利调用可接收解析后的地址；Graph 生产路径消费设备 offset 表的当前内容。
Verify/fold 的有限精度等价、zero-commit no-op、source/destination alias 合同必须保持。

迁移覆盖 RTX 5090、Spark 的全部已选 route，以及相应 Op tests/bench 的输入构造。
纯 linear、HC/mHC、MoE、Vision 数学内核不因 Context 池重写而增加池依赖。
需要改变的是 I/O 访问器与资源视图，不预先要求改变 MMA tile、数值算法或既有 shape 分派。

### 7.3 初始化与字节搬运

池不默认清空整个 backing。Root State 的数学初值由 State 存储动作建立；新 KV 页按消费者对
未写入区域的要求初始化；完整 COW 复制覆盖目标时不重复清空。搬移复制原 bytes，包括布局 padding，
不触发 codec、norm 或数学版本变化。

共享 `core/transfer` 提供有界描述符的批量独立字节复制，扩展现有 batch-copy 基础即可。
连续 run 合并；离散记录用批量提交，不逐记录创建 event。D2D 整理的源、目的全集必须 disjoint。
CUDA 端可采用批量 memcpy 或一个共享字节复制 kernel；表示、调用合同与统计相同，具体 dispatch 属于复制实现。
描述符、staging 和 event 由执行资源预先准备，Op 内不隐藏 Device allocation。

## 8. CUDA Graph 与实际执行

Graph 固定的是 arena 基址、执行表地址、frame/workspace 地址和计算几何；实际记录 offset 是输入数据。
decode 的设备计算不查询分配器；Host 准备仅处理必要的增量预留，不遍历全部缓存或重新捕获 Graph。
页表增长/修订使用已有准备阶段提交。
QSA 等节点参数更新机制继续由 Op 自己实现，与池整理分开。

Eager 与 Graph 消费相同的执行需求和 owning reservation；Graph 不形成单独的 State/KV 配额。
进入 replay 前，各 rank 的资源、地址表与依赖必须就绪。后续决定与提交继续使用本轮许可，不能在
Graph 完成数学计算后再尝试申请尚未保障的 canonical 目标。

整理发生在 graph replay 之外的显式 Context 操作中。旧消费者退休后，复制、patch、完成发布依次进行；
后续 replay 使用更新后的执行表。Graph 对象存在本身不持有历史 payload 的永久地址借用。

Graph 校准/准备只取得该次 capture 真正需要的 dummy records 与执行缓冲，保持合法 alias 和有效行约定。
不得为了 warmup `clear_all` 整个 Context arena，也不得把校准时记录保留成一套额外长期 KV/state 池。
退出仍先停止提交、drain tickets，再销毁 Graph、执行表、池 backing 和 DeviceContext。

## 9. Host 与分布式

### 9.1 RTX 5090 Host Context

Host KV 与 State 使用同一个 Host 容量域和同一记录分配合同，backing 为显式拥有的 pinned memory。
Device 与 Host 的 replica 可以共存，按各自真实 allocation 计费。传输期间两端均有借用，完成后
再按事务决定是否移除源 replica。Host 整理用字节复制，不重新生成模型状态。

迁移方案同时评估源端保护、完整目标和必要执行增量。不能先复制 State，再发现其配套 histories
无法在 Host 落下。目标域不足时，可以回收该域的可撤销缓存；不能建立“为了腾出目标，又必须迁回
等待被本次操作释放的源域”的循环依赖。保存方案不可行时，由缓存/调度选择放弃保留或模型支持的重算。

Host checkpoint 允许保留旧逻辑版本的 Host 副本，并在权限允许时让新 writer 接管原 Device
allocation。接管必须等待复制完成，且没有其他不允许写入的消费者；此路径不需要额外 Device image。
否则仍需独立写目标。Source 的引用、Host 目标以及后续 KV/COW 一起计入需求，不因接管省去生命周期检查。

传输按记录/run 计划，更新原先依赖 plane-separated Device layout 的 transfer profile。
日志分别统计实际 payload、padding、copy segments 与等待；不能沿用旧的“每页固定 N 个 plane copy”估算。
Expert Host 来源和 Context Host cache 仍是不同资源所有者，共同受启动时的实际 RAM 预算约束。

### 9.2 单 Spark 与双 Spark

Spark 的 Device pool、权重、pinned/ordinary 控制数据都消耗同一物理 RAM，启动预算只计算一次。
不再添加一个“CPU cache”扩大统计容量。PLE Disk page cache 可被 OS 回收，按现有 MemAvailable 预算语义处理。

双机的 backing 各自固定，大小可不同。协调侧进行跨 rank 的原子逻辑预留；全部满足后才提交各 rank
的存储动作。移动只发生在本 rank 内，不改变请求 owner 或 TP shard 放置，也不发送 CUDA 指针。
各 rank 接收自己的 offset/length 描述；涉及的物理事务全部完成后，才发布共同的恢复/执行结果。

某一 rank 不满足时，不保留其他 rank 的零散申请，也不先让它们进入数学 collective。需要先完成的
回收/整理是独立存储事务；在其完成前计算保持未就绪。所有 rank 的本轮计算、提交和丢弃资源均已取得
后才发布共同执行许可，提交后的失败遵循整个执行实例的失败/排空流程。

| 模型放置 | 申请规则 |
|---|---|
| 单 rank | 一份本地 KV 与 State 需求 |
| Qwen4Exp TP2/EP2 | 每条请求在两 rank 都有 head-sharded KV 与 GDN state；Indexer 历史按现有计划复制；两侧都必须满足 |
| GLM TP2/EP2 + attention DP2 | KDA state 分片在两 rank；Main/MTP MLA、Indexer KV 与 tail 仅在请求 owner；conditioning 按既定计划复制 |

GLM 建立 owner 前，用完整 checkpoint 所在 owner 与各 rank 的可行性选择恢复/放置；执行中 owner 保持稳定。
池不自行把 KV 搬到另一 rank。一个 owner 的长请求不能使用另一 rank 的空闲 KV 字节。
全局 snapshot 继续关联全部所需 State shards 和 owner histories，不把单 rank 命中当成完整命中。

ResourceManager 比较候选的命中、恢复成本和容量评估；Session 负责产生该候选在各 rank 的需求。
另一 owner 上从合法输入重算是不同来源方案，不视为现有 checkpoint 的原地恢复。准入、回收、暂停
和恢复都按相同容量域集合评估，不用两 rank 的字节和或一个固定共同页数代替。

## 10. 启动、配置与观测

### 10.1 启动工作集保证

启动分别规划固定执行资源、Context 总容量和必须可执行的工作集。模型提供无地址的对象布局、
每域需求及各阶段同时存活的关系，存储使用与运行时一致的记录规格和装载规则检查；不能只检查原始
payload 字节和。需要保证的是以下工作集各自可行，而非把所有场景的峰值相加。

| 必须可执行的工作集 | 保证范围 |
|---|---|
| 单个合法最大上下文请求 | 在 `max_context` 内，从合法初始来源推进到结束；包含所选 backend 的实际写入与提交峰值 |
| 基本并发工作集 | `max_concurrency` 条独立短请求具备合法输入后，能执行所选 backend 的基本批次；包含同时存活的 State 与写入峰值 |
| 必需转换和恢复 | 必需的读源/写目标、补齐、恢复至旧进度并执行新单元的可行路径；可使用模型支持的重算方案 |
| 实例 purpose | CausalScoring 等 purpose 按自身执行链声明需求，不借用 Generation 中不会执行的回收流程 |

`max_context` 是单请求语义上限，`max_concurrency` 是并发执行上限；不承诺所有并发同时达到最大长度。
可选 checkpoint、某一个必须保留大 source 的分叉方案、Host 保存和额外缓存不构成启动最低保证。
这些方案在运行时按自身完整需求准入；不可行时可以跳过可选保存或选择其他合法恢复来源。

基本并发与单请求长上下文的保证可在不同池占用状态下实现，不保留固定的 B 份 State 加 L 页 KV
区域。启动工作集保证也不永久锁住空位：缓存可以使用这些字节，执行需要时按 §6 回收。
其他活跃请求占据容量时由 Scheduler 减少活动集合，才能兑现单请求进展保证。

模型用解析上界及会达到峰值的边界位置描述工作集，包括跨页、verify 窗口和 source/destination
重叠存活；不要求启动逐 token 执行模拟。通用启动层消费声明，不重新推导模型前沿或默认“两份 State”。
双机对每个声明可承载请求的 owner 和所有相关 rank 检查完整需求向量，不以总 RAM 通过检查。
无法满足时由资源实现报告失败工作集、容量域和所需容量，启动层保留该原因。

### 10.2 准备顺序与预算

1. 从无地址模型 profile 取得布局、类别和启动工作集，规划权重、固定 workspace/frame、通信、
   PLE/Vision/expert 资源以及目录/执行表预算。
2. 按现有机制加载权重并完成 Graph 校准。校准使用临时小型 Context backing；销毁校准资源后
   重取可用内存，计入正式 Graph 重建预算。
3. 选择每个容量域的 Context 字节预算。存储返回该容量的有效几何及控制元数据预算，统一检查
   Context backing、元数据和固定资源能否共同放下，以及 §10.1 的工作集是否可执行。
4. 创建正式 backing、Directory 和执行表，用最终地址准备 Graph；实际准备后核对预算，按已有
   所有权顺序销毁/重建需要纠偏的资源，不同时保留两套完整 Context。

每 rank 的预算关系为：可用总预算减去权重、固定执行资源、Graph、控制元数据及系统 headroom，
剩余部分才可交给 Context。使用同一预算快照，已反映在可用量中的 allocation 不重复扣除。
MoE cache/window、Vision handoff、PLE 缓冲保留原 owner，不参与本池的大块交换。

控制元数据随总容量、最小记录规格及目录合同估算，与 Context 容量共同求解；不能先把 Spark
MemAvailable 全部分给 arena，再在同一 RAM 中增配目录。协调侧跨 rank 目录只计入其实际所在
主机；远端只计入自己拥有的控制/执行资源。正式 Graph 不引用校准 backing。

### 10.3 配置

- `engine.context_capacity = "auto"`，或显式字节容量；CLI 对应 `--context-capacity`。
  表示每 rank 的 Device Context 容量，auto 可因 rank 的剩余资源不同而不同。
- 删除 `engine.kv_capacity` / `--kv-capacity` 与 `context_cache.extra_state_slots` / `--extra-state-slots`，
  不保留换算别名。`max_context`、`max_concurrency` 仍是各自语义上限。
- `context_cache.host_capacity` 保留字节语义；`prefix_reuse`、`checkpoint_interval` 保留缓存策略语义。
- `NINFER_VRAM_BUDGET_MIB` 仍约束总 Device 预算，不等于 Context pool 大小。

实施时一起迁移 example.toml、CLI/serve/bench、相关工具和配置 fixtures；本设计不提前修改当前程序行为。

### 10.4 观测口径

启动报告显示每域总字节、块大小、各记录规格与装载率，以及通过检查的工作集和最低需求；运行报告显示：

| 指标 | 含义 |
|---|---|
| allocated / reserved / pending-retirement bytes | 已持有、已预留、等待消费者完成的真实占用，口径互斥 |
| free shared blocks / bytes | 可以立即交给任意类别的容量 |
| class free records / padding | 类内可用记录、块尾及未满块空间，不能混称立即通用容量 |
| live State / KV records | 用途统计，不是固定配额 |
| relocation bytes / batches / wait / execution time | 区分容量整理、数学 COW、Host 交换 |
| allocation pressure | 完整需求的可满足性、涉及 rank/存储域、等待原因；区分 payload、目录与执行额度 |
| cache retention / reclaim | 保存与跳过原因、实际唯一对象释放量、Host 迁移或来源撤销；不以 checkpoint 名义大小当释放量 |
| preemption / recovery | 暂停方式、恢复来源、受保护恢复范围及新工作是否已完成 |

不再报告一个脱离 State 用量的“固定最大 KV token 容量”。如需要估算可增长 token 数，必须说明当前
状态数量、表示、rank placement 与假设，不能作为可保证的准入结果。
物理分配状态按互斥口径统计；活跃/缓存用途和共享引用作为属性统计，不与物理占用相加。若最后引用
已释放但消费者未结束，仍计为待退休占用，不提前记入可用容量。

## 11. 当前模型的对象几何

以下核算于 2026-10-22。Qwen 数据来自当前本地 artifact config、StateImage builder 和 KV codec；
GLM 来自已经选定的量化/并行/Op 设计。MiB、GiB 分别为 2^20、2^30 B。
“27B”以当前 Qwen3.8-27B 为主；Qwen3.6-27B 的 Main GDN/KV 几何相同，draft 配置仍须读各自 artifact。

### 11.1 State

| 模型 / 放置 | 无额外 cyclic draft 的完整 State | 启用相关 backend 后 |
|---|---:|---:|
| Qwen27B，单 rank | 146.822266 MiB | MTP 相同；DFlash2 为 186.822266 MiB |
| Qwen35B，单 rank | 61.410156 MiB | MTP 相同；DFlash 为 141.410156 MiB |
| Flash Next，单 rank | 110.301758 MiB | MTP 为 110.322266 MiB |
| Flash Next，TP2，每 rank | 55.247070 MiB | MTP 为 55.267578 MiB |
| GLM5Next，TP2 + attention DP2，MTP | owner 70.416016 MiB | non-owner 70.398438 MiB |

Qwen27B 为 48 层×48 heads 的 FP32 recurrent（144 MiB），加 BF16 conv history 和 continuation hidden。
35B 为 30 层×32 heads（60 MiB）加对应字段。DFlash2 的五层 2048-position cyclic K/V 增加 40 MiB；
35B DFlash 的五层 4096-position cyclic K/V 增加 80 MiB，其另一个 full-attention 层属于 growing KV。

Flash Next 的 State 计入 GDN、QSA raw tail/coordinates、PLE 九列 history、sample hidden 与选定 MTP 字段。
TP2 只切分声明分片的字段，不把整个 StateImage 简单除二。

GLM 目标布局：每 rank KDA recurrent 68 MiB、conv history 2.390625 MiB，保存一份 BF16 `[4096]`
实际 Main final hidden（8 KiB），供续算与 MTP conditioning 共用；owner 另有 12 层 raw K/gate tail 共 18 KiB。
共同字段在两类 image 中采用相同偏移，owner-only tails 位于末端。两种大小可以注册两个记录类别；
都一块一份，non-owner 仅产生 18 KiB/份的块内余量。数学 frontier/phase 保存在模型 metadata，
不为无 RoPE 的 GLM 增加 QSA 三轴坐标字段。此处是待实现布局的字节推算，不是 GLM 启动测量。

### 11.2 64-token 页组记录

表中均为记录跨度，含 256 B plane 对齐。Qwen BF16 profile 保留既有 BF16 K / FP16 V 语义；
GLM BF16 保存 BF16 latent，不沿用 GQA 的 V 表示。

| 模型 / rank 放置 | Main BF16 | MTP BF16 | Main FP8 | MTP FP8 |
|---|---:|---:|---:|---:|
| Qwen27B | 4096 KiB | 256 KiB | 2064 KiB | 129 KiB |
| Qwen35B | 1280 KiB | 128 KiB | 645 KiB | 64.5 KiB |
| Flash Next，单 rank | 1584 KiB | 132 KiB | 822 KiB | 68.5 KiB |
| Flash Next，TP2，每 rank | 816 KiB | 68 KiB | 438 KiB | 36.5 KiB |
| GLM5Next，attention owner | 748 KiB | 68 KiB | 407 KiB | 37 KiB |

Flash Next 含每层 BF16 pooled index keys；TP2 时它们仍复制。FP8/TP2 的单-head scale plane
只有 128 B，256 B 对齐使每层记录另有 padding，不能直接用纯 payload/token 替代表中跨度。
GLM FP8 是 512 codes＋四个 FP32 scales，Indexer 仍为 BF16。

Qwen35B DFlash 的 Draft Full 页组为 256 KiB，采用自身 D128/BF16 表示；不能套用 Main D256 FP8 codec。
Qwen27B DFlash2 当前全为 cyclic layers，没有额外 growing Draft Full 页组。
其他现有 KV 表示的记录跨度如下，单元格为 `Main / MTP`，单位 KiB：

| 模型 / rank 放置 | INT8 | NVFP4 | K8V4 |
|---|---:|---:|---:|
| Qwen27B | 2112 / 132 | 1152 / 72 | 1608 / 100.5 |
| Qwen35B | 660 / 66 | 360 / 36 | 502.5 / 50.25 |
| Flash Next，单 rank | 840 / 70 | 480 / 40 | 651 / 54.25 |
| Flash Next，TP2，每 rank | 444 / 37 | 264 / 22 | 351 / 29.25 |

这些格式使用同一布局生成路径，按当前 code/scale 规格注册记录，不另设池实现。
GLM 本阶段只定义 BF16/FP8 latent，不把现有 GQA codec 的支持范围扩展到 MLA。

### 11.3 大块装载实例

| 模型，MTP，BF16 | B | 每块 Main 记录 | Main 块尾 | 每块 MTP 记录 |
|---|---:|---:|---:|---:|
| Qwen27B | 146.822266 MiB | 36 | 2890 KiB | 587 |
| Qwen35B | 61.410156 MiB | 49 | 164 KiB | 491 |
| Flash Next，单 rank | 110.322266 MiB | 71 | 506 KiB | 855 |
| Flash Next，TP2 | 55.267578 MiB | 69 | 290 KiB | 832 |
| GLM5Next，DP owner | 70.416016 MiB | 96 | 298 KiB | 1060 |

27B 的大 State 使整理一块的搬运上界也更大，且每块重复 padding 比 Flash Next 明显。
这是真实代价，应在该模型的容量压力场景中验收，不能只用 Flash Next 的数字代表全部模型。

## 12. 部署可行性

### 12.1 硬件、权重与已有证据

5090 的现有启动记录报告 34,162,016,256 B CUDA total，约 31.816 GiB；单 Spark 的物理 RAM 为
130,661,203,968 B，约 121.688 GiB。实际可用预算须扣除系统/其他进程、权重、固定执行资源、Graph 和 headroom。

以下权重数值按当前绑定引用的物理对象去重，选用 Vision、MTP 和 optimized proposal head；
不包含 Context、workspace、Graph 或 Host 控制数据：

| 模型 / 放置 | 常驻权重依据 |
|---|---:|
| Qwen3.8-27B groupwise，单 rank | 16.948 GiB，artifact payload 核算 |
| Qwen3.8-27B NVFP4，单 rank | 20.004 GiB，artifact payload 核算 |
| Qwen3.6-35B-A3B，单 rank 全驻留 | 20.825 GiB，artifact payload 核算 |
| Flash Next，单 Spark，PLE Disk | 73.854 GiB，artifact 去除 27.127 GiB PLE 表后的核算 |
| Flash Next，双 Spark，每 rank | 52.244 GiB，已有启动实际 allocation，包含本地 PLE 分片 |
| GLM5Next，双 Spark，每 rank | 约 91.462 GiB，选定量化/placement 的估算 |

完整 `.ninfer` 文件大小不等于选定 backend 的常驻权重。Flash Next 当前
[`rank_plan`](../../src/models/qwen4_exp/parallel/plan.cpp)明确单 rank 使用 PLE Disk，双 rank 使用 Device 分片。
5090 的 Flash Next 另使用现有 expert offload，不能把 73.854 GiB 全部列为 Device 权重。

已有运行证据：27B 的 5090 FP8 KV、ordinary/MTP/DFlash2 见
[服务测量](../performance/qwen3.8-27b.md)；35B 5090 的一份 MTP3/Vision/BF16 启动记录中，
Device 权重为 20.691 GiB（full proposal head），workspace 为 0.404 GiB。
Flash Next 的单/双 Spark 启动记录证明现有执行路径可运行；本设计尚未运行新池。

GLM 全量目标权重约 178.313 GiB；它在当前方案下不能全驻留单 Spark，更不能全驻留 5090。
当前 5090 主机约 93.854 GiB 系统 RAM，也不能据现有 Host offload 就宣称可容纳该模型全部权重。
此设计不会新增 GLM 单机 offload 或改变已选定双 Spark 目标。

### 12.2 相同工作集的池容量

下面按八条互不共享 KV 的请求、每条 8192 个 canonical tokens、16 份 State 计算。
16 是用于比较的实际对象数量假设，包含 working/retained/reserved images，不是启动配额。
GLM 按 owner 各四条请求分配，但两 rank 都保存全部请求所需的 KDA shards。
数字已按整大块向上取整，未计新一轮追加/COW 的额外峰值、控制元数据和 backing 最末尾余量。
这些表验证既定对象组合的装载量；启动与准入仍按 §5、§10.1 的完整执行需求检查。表中的 State 数
不是每请求固定两份的实现要求，也不能用 canonical 容量表替代 speculative/恢复峰值规划。

| 模型 / 放置，每 rank | BF16 池需求 | FP8 池需求 |
|---|---:|---:|
| Qwen27B + MTP | 6.739 GiB | 4.588 GiB |
| Qwen35B + MTP | 2.399 GiB | 1.739 GiB |
| Flash Next + MTP，单 rank | 3.555 GiB | 2.693 GiB |
| Flash Next + MTP，TP2 | 1.781 GiB | 1.349 GiB |
| GLM5Next + MTP，TP2/attention DP2 | 1.582 GiB | 1.375 GiB |
| Qwen27B + DFlash2 | 7.115 GiB | 5.108 GiB |
| Qwen35B + DFlash | 3.867 GiB | 3.176 GiB |

真实执行的增长页数由前沿决定：`ceil(write_end/64)-ceil(old_end/64)`，另加实际尾页 COW。
例如短 spec 窗口也可能跨页，同时复制共享尾页；必须预留二者，不能只按接受后的 token 数分配。
prefill chunk、verify width 不直接扩大 StateImage；对应临时记录已计入执行资源。

### 12.3 5090

- **27B**：FP8/MTP 的上述工作集与 NVFP4/Vision 权重合计约 24.593 GiB，仍需为全部固定资源、
  Graph、系统及 headroom 留空间；groupwise 权重进一步减少约 3.057 GiB。已有服务测量验证过
  ordinary 262k 与 MTP/DFlash2 的相应配置，不能把它扩展成所有 Vision/batch/backend 组合的保证。
- **35B**：BF16/MTP 示例的权重加池约 23.223 GiB；Host Context 可继续服务缓存交换。
  降低显存预算时仍由现有 MoE 放置策略选择 offload，Context 内部不干预 expert 集合。
- **Flash Next**：小工作集 Context 本身为约 3.56/2.69 GiB。全模型可行性依赖已有 GPU/Host
  expert 分工、PLE Disk 和 CPU RAM 预算。新池只在最终分给 Context 的字节内工作，不挤占 streaming window。
- **GLM5Next**：当前量化、硬件和已选执行计划不支持该单机部署，不能归因于或通过此池解决。

单条 262144-token、两份 State、MTP 的 canonical 池需求为：27B BF16 **17.636 GiB**、FP8 **9.033 GiB**；
35B BF16 **5.697 GiB**、FP8 **2.999 GiB**；Flash Next BF16 **7.003 GiB**、FP8 **3.771 GiB**。
这说明 27B 的 BF16 长上下文在 5090 上即使统一池也放不下；FP8/MTP 是否留足固定资源须由实际启动规划确定。
表中 L 是已持有长度；到达模型上限后不能再承诺额外输出空间。

### 12.4 单 Spark

27B、35B 沿用单 rank 模型路径。其工作集远小于物理容量，统一池主要增加长期前缀与 state 的
弹性保留机会。八条各 262k、16 份 State、MTP 时，27B FP8 池约 71.834 GiB，35B BF16 池约
45.098 GiB；加上述权重后分别约 91.838 / 65.923 GiB，尚有固定运行资源的预算空间。
27B BF16 的相同工作集仅池就要 140.944 GiB，超过单机物理 RAM。

Flash Next 当前单机 PLE 在 Disk，权重按约 73.854 GiB 核算。八条各 262k 的 BF16 池约
55.700 GiB，加权重已超过单 Spark RAM；FP8 池约 29.735 GiB，加权重约 103.589 GiB，
还剩约 18.10 GiB 承担固定执行资源、系统及 headroom，属于有明确预算余量的容量候选。
这不是新池的整模型运行结果；必须按实际准备后的剩余预算决定并发准入。

GLM 的 178.313 GiB 权重不适合本阶段单 Spark 全驻留目标。Spark 统一内存不能被当成
“GPU 128 GB 再加 CPU 128 GB”重复计算。

### 12.5 双 Spark

**Qwen27B/35B**：当前 Qwen3.5 架构没有同一模型实例的 TP2 执行实现。可以在两台各运行一个
独立单 rank 实例，各用本地池；不因此获得一个请求的双倍容量。若以后加入模型 TP，本池只需消费
新 placement 和记录布局，相关模型/通信支持仍需单独实现。

**Flash Next**：现有 TP2/EP2 路径适用。八条各 262k、16 份 State、MTP/BF16 时每 rank 的池约
28.659 GiB，加权重约 80.903 GiB；FP8 对应池约 15.814 GiB。两 rank 的 head shards 表示同一批
逻辑 tokens，不能将两侧 token-equivalent 容量直接相加。

**GLM5Next**：模型尚未完成集成。按每 rank 91.462 GiB 权重，若最终能给 Context 分到 16 GiB，
单机剩余约 14.226 GiB 承担全部其他资源。按 16 份 State：

| 每个 owner 持有的长请求 | 每 rank 池需求 | 对 16 GiB 预算的判断 |
|---|---:|---|
| BF16，一条 1048576-token 请求 | 13.959 GiB | canonical 工作集可容纳，池内剩余约 2.041 GiB |
| FP8，两条各 1048576-token 请求 | 15.060 GiB | canonical 工作集可容纳，池内剩余约 0.940 GiB |
| BF16，四条各 1048576-token 请求 | 52.331 GiB | 不成立 |
| FP8，四条各 1048576-token 请求 | 28.950 GiB | 不成立 |

因此双机容量上可分别容纳两条 BF16 1M、或四条 FP8 1M 的候选工作集；前提是 owner 分布对应表中
情形，且实际固定资源开销满足上述预算。GLM 的 workspace/Graph 尚未通过完整启动测量，不能把
16 GiB 当成已确认的自动池容量。KDA State shards 在两台都占空间，只有 owner KV 可在总容量中相加。

### 12.6 用现有双 Spark 启动记录校验收益边界

读取 `.51:/home/neroued/ninfer/.agent-tmp/qwen3_8fn-eval-20261021/requests.jsonl` 中的一份
实际启动记录：BF16、MTP3、optimized head、Vision、Graph、B=8、chunk=4096；Main KV 为
2,503,488 token-equivalents、State 322 份。每 rank KV payload **32.978 GiB**，State **17.379 GiB**，
合计 **50.357 GiB**。这是已运行实现的 allocation，既不是新池结果，也不是当前占用快照。

将相同 50.357 GiB 作为候选池 backing，保留 MTP 比 Main 多八页的余量，按当前整块规则计算：

| 实际 State 份数 | 可容纳 Main token-equivalents |
|---:|---:|
| 16 | 3,735,936 |
| 64 | 3,541,632 |
| 128 | 3,281,088 |
| 322 | 2,490,624 |

最后一行比旧布局略少，反映大块与 padding 的真实成本；不能隐藏。前面几行说明在 State 需求较低时，
原本锁在 State 配额里的容量能够转给 KV。反向负载则允许更多 State 占用。这里未计新旧元数据的差异，
也未假定缓存策略一定保留这些数量。大于单请求或活跃 batch 上限的部分用于多个 inactive histories。

## 13. 对比与性能目标

[vLLM 的统一固定块](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)通常不需为分配碎片
搬移活对象，分配确定性优于本方案；其混合布局、分组与 padding 在形状匹配良好时也很高效。
[SGLang unified pool](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/unified_memory_pool.py)
的字节区域组织可以有更细的容量利用，代价是边界和重定位协调。本方案用可归还的任意位置大块，
避免区域顺序依赖，并保留细粒度 KV 与完整 State。

本方案的收益目标是：取消固定配额导致的闲置；普通执行只消费已准备的地址；压力下整理量局部有界。
并不预先声称优于其他引擎的 decode 延迟。按读写合计带宽 W，搬移 D 字节的下限为 `2D/W`，
实际需另计提交、映射 patch 和等待；不能用未测量的峰值带宽代替数据。

验收同时看稳定 prefill/decode 与负载切换时的整理量/停顿。没有压力时不应出现整理；普通 decode
不应新增全历史表重建、逐页 Host readback、逐页 event 或隐式 Device allocation。
控制侧的无压力路径只做批次需求的增量计算与成组预留；释放 inventory 和缓存候选搜索在需要回收时建立。
压力路径同时记录有用执行与 replay、暂停、搬移的开销，避免只看单步 kernel 时间而漏掉恢复振荡。
性能差异按原本的负载与误差尺度解释，不把少量波动当成回退反复重测。

## 14. 实施组织与完成条件

目标文件组织保持现有语义边界：

```text
src/core/
    context_pool.h/.cpp        # 无 CUDA 地址的记录/大块分配与整理计划
    record_view.h              # 非拥有记录视图与字段/plane 几何
    state_image.*              # 单 image 的字段布局，不再拥有 slots 布局
    paged_kv_*                 # KV 记录几何与消费视图，不再定义固定容量分区
    transfer.*                 # 通用独立字节 batch copy
src/models/common/context/
    directory/*               # 对象、引用、replica、pool reservations、执行绑定
    storage.* / transfer.*     # rank backing、映射 patch、初始化、迁移与完成发布
    physical_reclaim.*         # 真实释放 inventory 与成组需求适配
    release_inventory.h       # 共同持有者与候选组合释放证明
src/models/<architecture>/
    state/ / session/ / program/ # schema、placement、工作集、数学前沿与原生执行
src/runtime/
    contract/resources.h      # 容量域、完整需求评估与预留结果；用途统计独立
    engine/context_capacity.* # 固定资源、元数据与统一 Context 的启动预算
    engine/engine_schedule.inl # 准入、执行成员、抢占与受保护恢复
    engine/context_cache/     # 来源、保留、释放集合选择与缓存价值
src/ops/<family>/              # 视图消费、数值实现与设备分派
```

不为每个模型写一套分配器，也不把模型 Session 重写成统一 forward 模板。
核心分配元数据属于 Directory 持有的 owner，Core backing/执行对象不再维护另一份 free list。

### 14.1 必须替换的现有假设

| 现有位置 | 替换内容 |
|---|---|
| `ContextCapacityCurve` / `ContextCapacityResolution` | 从固定 pages/slots 曲线改为容量域预算、控制开销和模型声明的工作集可行性 |
| `ContextResources::useful()` / `first_kv_deficit()` 及相关分支 | 不再按首先短缺的物理类型衡量释放收益；消费完整需求与相关域评估，数量仅保留逻辑需求/用途统计 |
| `StateDirectorySpec`、各 KV `PageAllocator` 与目录规划 | 统一物理记录所有权；逻辑目录上限不再从固定 State slots / typed pages 派生 |
| `start_binding` / `reserve_units` / `release_units` | State、全部 history、source 保护与后续提交需求共同取得和释放 |
| `prepare_capture` / `reclaim_capture_reservation` | 预览冻结后的写入需求；未提交保存可为同域任何必要需求让出容量 |
| `PhysicalReclaim` / `ReleaseInventory` | 保留共同持有者证明；去除按 shortage 类型过滤，返回可供池评估的真实释放集合 |
| `ResourceManager::reclaim()` / `balance_reused()` | 回收停止条件改为完整需求可满足；独立池容量占比与按页/slot 的收益比较退出 |
| pause / replay / scoring / 分布式准备 | 消费同一预留基础，保留恢复至旧进度加新单元的保障及跨 rank 提交关系 |
| 配置、观测、测试及工具 | 迁移容量和压力口径，删除被替代配置、别名、固定 slot 输入及其专用测试 |

### 14.2 实施步骤

按四个有完整产出的部分推进：

1. **记录布局、池与需求合同**：落实物理记录、成组预留、装载/整理算法、借用与退休、字节复制和视图；
   明确 Session 需求到 Directory/Pool 的转换、工作集评估和释放集合接口。先验证控制与存储合同。
2. **全部存储消费者与 Op 迁移**：两种现有架构的 KV/State、全部 KV 类型、GDN/PLE/DFlash、
   ReplaySSM 与 Graph 使用新视图；同步 GLM 待实现接口与测试/bench 构造，删除受影响旧 slot 合同。
3. **引擎与多 rank 完整切换**：启动保证和元数据预算、绑定/执行/capture/score 的完整预留、统一
   回收、Host、抢占/恢复、跨 rank 事务及配置/观测一起接通；不保留固定两池的选择分支。
4. **集成收尾**：对 §14.3 的合同选择代表性组合，运行现有 5090、单/双 Spark 模型，核对数学、
   缓存与容量压力下的进展；审查旧接口/资源字段/配置残留，再集中更新当前实现文档。

### 14.3 完成条件

| 合同 | 必须成立的结果 |
|---|---|
| 启动保证 | 固定资源、目录和池共同满足预算；声明工作集按真实装载可行，失败原因归属正确 |
| State/KV 容量交换 | 必要 State 可以利用回收 KV 让出的容量，反向同样成立；覆盖不足一整块的中间状态 |
| 成组执行 | 需新增 State、多个 history 和 COW 的申请全成或全败；开始计算后 commit/discard 不因缺额外记录失败 |
| 缓存保留 | 可选 capture 不破坏已承诺执行；无空间时仍能继续计算；共享引用只按实际最后释放计费 |
| 持续推进 | 无 Host、活跃工作集过大时可抢占与重算；受保护恢复能越过旧进度，未因新增 capture 再次失去保障 |
| 异步结束 | source/target、副本和地址在消费者期间有效；取消、EOS、迁移及整理不会提前释放或留下失去 owner 的预留 |
| 多 rank | 任一域不足时不出现部分数学提交；TP 完整来源与 DP owner 放置按全体所需 shards 判断 |
| 入口完整性 | 普通/spec、Vision、scoring、eager/Graph、Host 恢复与 CLI/serve/bench 使用同一资源合同 |

数学消费者按 Op 合同使用独立参考验证，不能仅靠新旧输出相同证明数学正确。调度与缓存验证关注
同一来源仍可恢复、共享页不被覆盖、State/各 history frontier 一致、释放后没有残留 writer/reader。
物理搬移要求 bytes 不变、逻辑 identity/content epoch 不变；源/目的与 padding 边界单独检查。

实际运行选择 §12 中已经存在的部署，不把尚未实现的 Qwen3.5 TP2 或 GLM 集成计为已验收。
GLM 在本阶段完成布局/容量向量/Op 视图及纯控制流程验证，整模型验证随其正式接入完成。
对正常 decode 的访问成本、压力整理延迟和 prefix-cache 行为都保留证据，文档推算不替代这些结果。

## 15. 依据

- 现有 [StateImage](../../src/core/state_image.h)、[KV 几何](../../src/core/paged_kv_geometry.h)、
  [KV 表示](../../src/core/paged_kv_storage.h)、[统一目录](../../src/models/common/context/directory.h)、
  [资源向量](../../src/runtime/contract/resources.h)与 [GDN fold](../../include/ninfer/ops/gdn_replay.h)。
- 现有 [启动容量规划](../../src/runtime/engine/context_capacity.cpp)、
  [请求调度](../../src/runtime/engine/engine_schedule.inl)、
  [来源与回收策略](../../src/runtime/engine/context_cache/resource_manager.h)、
  [共同持有者释放证明](../../src/models/common/context/release_inventory.h)、
  [Qwen4Exp 绑定与执行预留](../../src/models/qwen4_exp/session/context.cpp)及
  [Qwen3.5 checkpoint](../../src/models/qwen3_5/session/capture.cpp)。这些实现用于确认迁移范围，目标合同见正文。
- [GLM 并行与预算](glm5next/parallel.md)、[KDA](glm5next/replayssm-kda.md)、
  [Kpool Sparse MLA](glm5next/kpool-sparse-mla.md)、[MTP](glm5next/mtp.md)。
- 当前 artifact：`out/qwen3_8_27b.ninfer`、`out/qwen3_8_27b_nvfp4.ninfer`、
  `out/qwen3_6_35b_a3b.ninfer`、`out/qwen3_8_flash_next.ninfer` 及其目录声明的 continuation files。
- 现有启动观测：本机 `.agent-tmp/observability/rtx-mtp/requests.jsonl`、
  `.agent-tmp/observability/spark-updated-single/requests.jsonl`，以及 §12.6 的 `.51` 记录。
  关键数值已写入本文，不依赖临时目录永久保留；路径用于核对其测量范围。
- [vLLM Hybrid KV Cache](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)、
  [vLLM 分组实现](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py)、
  [SGLang unified pool](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/unified_memory_pool.py)、
  [SGLang 共享容量下的回收](https://github.com/sgl-project/sglang/pull/33091)。比较以实现为准，不以旧设计文档的限制代表全部新代码。
