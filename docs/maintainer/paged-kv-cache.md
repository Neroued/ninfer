# NInfer Context 存储与 Paged KV

本文定义 Context 物理存储、State/KV 身份、执行视图与迁移合同。
请求调度、缓存选择和抢占见[资源调度与上下文缓存](resource-scheduling-and-context-cache.md)；
执行与提交见[Engine 架构](engine-architecture.md)。

## 1. 容量与所有权

每个 rank 的 Device Context 使用一份固定 backing。State images、Main KV、MTP KV 和
Draft Full KV 按实际需要竞争这份容量，不在启动时分别划出固定配额。
RTX 5090 的 Host Context 是独立的 pinned backing，使用同一记录表示和分配机制；
Spark 不建立重复的 Host Context cache。不同 rank、不同存储域的余量分别评估。

| 模块 | 职责 |
|---|---|
| 模型 Session | State 字段、history 几何与 placement、模型前沿、完整工作集和恢复条件 |
| ContextDirectory | 逻辑身份、引用、版本、覆盖、写权限、成组预留和迁移事务 |
| ContextPool | 字节记录的放置、实际预留、AllocationId、物理绑定、回收和整理计划 |
| RankProgram / ContextStorage | 拥有 / 借用 backing，执行复制与映射更新 |
| Executor | 执行额度、提交顺序、依赖、结果与消费者退休 |
| Op | 数学和视图合同，不分配记录、不选择受害者 |

`engine.context_capacity` / `--context-capacity` 接受字节容量或 `auto`。启动按模型声明的
工作集验证最低容量，自动模式从权重、workspace、Graph、Host 固定资源和 headroom 之外
的预算求解。`max_context` 限制单请求长度，`max_concurrency` 限制执行 lanes；两者不限制
缓存总量，也不把容量平均切给请求。公开容量是每 rank、每域的字节数。

模型声明有限种 `{bytes, alignment}` 记录。ContextPool 将同类记录放入大块，块大小取可容纳
最大记录的对齐尺寸；普通申请优先使用已有同类块。空块可以改用于另一类别。总空闲字节
不是可分配性的充分条件，整组需求必须由同一放置评估确认。池不解释 token、模型层、
checkpoint 价值或并行策略。

权重、expert cache、PLE 表及其 I/O cache、activation/workspace、ReplaySSM transition
records、Vision 临时 K/V、采样与 frame 临时数据不属于 Context 池。

## 2. 逻辑身份与物理记录

```text
StateId / PageId                    带 generation 的逻辑身份
  └─ schema 指定的 rank-local 部分
       ├─ Device record             AllocationId、binding epoch、offset
       └─ Host record               AllocationId、binding epoch、offset
```

`StateId` 表示一个位置的完整续算状态。每个 rank-local image 连续，字段及层内矩阵保持
固定相对偏移与自然连续轴。GDN/KDA recurrent state、conv history、Indexer raw tail、
PLE conv history、continuation hidden、DFlash cyclic KV 可以作为字段。State variants
允许不同 owner placement 使用不同 image 布局，分配和复制必须使用相应 variant。

`PageId` 还包含 history pool ID，表示一组共享 coverage 与生命周期的 KV planes。
Main、MTP、Draft Full 保持独立 history，pool ID 标识数学用途和记录布局，不标识独占容量。
`HistoryId` 表示带 generation 的逻辑 PageId 序列和前沿。

逻辑引用保护内容，执行借用保护本次消费者期间的地址。长期 checkpoint 不永久固定物理位置。
最后一个内容 owner 离开后，记录仍被执行读取时进入 pending retirement；消费者结束后才能
归还位置。读取租约在有效期间可以交接给新的 checkpoint owner。

移动记录保持 AllocationId 和数学内容不变，只增加 binding epoch。State 数学提交产生新的
内容版本；KV 合法追加扩展 coverage，不使已有较短的 Host 前缀失效。绑定、复制、冻结和
内容提交具有不同的生命周期，不能用某个 token frontier 代替执行完成依赖。

## 3. KV 记录布局

逻辑页跨度固定为 `P=64` 个原 token。只有具有相同 ordinal、frontier、生命周期与页跨度的
planes 才组成一个记录。记录内按模型声明的 layer/plane 顺序打包，plane 起点和记录末端
按 256 B 对齐；Device 与 Host 使用完全相同的字节表示。

每个 plane 的几何包含 dtype、leading extent `X`、heads `H` 和 `tokens_per_row=s`。
`s` 为 1 或 4；页内物理行数为 `R=P/s`。plane 页内形状为 `[X,R,H]`，feature 连续，
token 次之，head 再外；层与 plane 的相对偏移由 schema 给出。不存在跨物理页容量的
head stride，Draft Full 也使用相同记录寻址。

| Profile | D=256 时 K code + scale | V code + scale | 每 token/head 合计 |
|---|---:|---:|---:|
| BF16 | 512 B | 512 B | 1024 B |
| INT8-G64 | 256 B + 8 B | 256 B + 8 B | 528 B |
| FP8-E4M3FN-G128-UE8M0 | 256 B + 2 B | 256 B + 2 B | 516 B |
| NVFP4-G16 | 128 B + 16 B | 128 B + 16 B | 288 B |
| K8V4 | 256 B + 2 B | 128 B + 16 B | 402 B |

BF16 profile 的 K 为 BF16、V 为 FP16；写入端完成转换。K8V4 是固定的非对称 profile，
K 使用 FP8、V 使用 NVFP4，不能按两倍单侧字节数计费。Draft Full 使用自身 BF16 profile。
上述 K/V planes 的 `s=1`；Qwen4Exp 每个 QSA 层的 BF16 index plane 使用 `s=4`。
未完成 pooled 行的 raw tail 属于 State。

单 plane payload 为 `sizeof(dtype) * X * H * (P/s)`。记录大小还包含 plane 间与末端 padding，
容量、迁移和视图使用同一布局生成结果，不另写一套字节公式。
Cache 编码的共同定义见 [kv_cache_format.h](../../include/ninfer/kv_cache_format.h)；
各 Op 的基底与计算约定见 [KV append](../../include/ninfer/ops/kv_cache_append.h)、
[Softmax Attention](../../include/ninfer/ops/softmax_attention.h) 和
[QSA](../../include/ninfer/ops/qsa.h)。

## 4. 执行视图与直接寻址

活动 history 租用固定执行表的一行。表的形状为 `[Nlogical,C]`，每项为 **I64 byte offset**，
相对于本 rank 的 Context backing。`Nlogical=ceil(max_context/64)`；batch 的 `table_rows`
仍为 I32 行索引，与 byte offset、token position、临时 frame slot 分属不同语义。

```text
logical_page = position / 64
record_offset = block_table[logical_page]
plane_row = (position % 64) / tokens_per_row
address = arena_base + record_offset + plane_offset
        + head * head_stride + plane_row * row_stride + feature * element_stride
```

`PagedRecordView` / `ConstPagedRecordView` 绑定 backing 与 I64 映射；
`PagedKVLayerView` / `PagedKVBatchLayerView` 绑定所需 codec planes、heads 与表。
读接口使用 Const views；append、prepare 等写接口接收可写 view。消费者不持有 allocator、
HistoryId、请求身份或容量配额，也不依赖相邻逻辑页的物理连续性。

State 使用 `StateFieldView` / `ConstStateFieldView`，字段 schema 与 `RecordBatchView`
提供的 I64 image offsets 组合解析地址。单记录可取对应 Tensor；批量计算直接使用字段和
offsets。StateImageLayout 不包含槽数或 backing 容量。DFlash 临时 frame/lane selectors
仍按各自 Op 的 I32 合同工作，不因持久状态改用 byte offset 而改变。

调用方保证所访问逻辑页已物化、读域不超过有效前沿、code/scale 写入完成后再发布 coverage，
并保护映射及记录直到消费者退休。Op 检查自身 dtype、geometry、stride、alias 和执行范围。
页定位在 page/tile 粒度复用，不增加与 context 长度成比例的 gather-to-contiguous copy。

## 5. 前沿、共享与 COW

页 `b` 在前沿 `F` 的有效原 token 数为 `c=clamp(F-b*64,0,64)`。
行跨度 `s` 的 plane 只有 `[0,floor(c/s))` 完整有效；consumer 还需遵守每个 query 的因果读域。
RoPE/MRoPE 与位置偏移是计算输入，不改变 KV ordinal。

例如 Device 覆盖 48 列、Host 覆盖 40 列，Host 仍可恢复不超过 40 列的前缀；还需 48 列时
不能丢弃 Device。合法追加不更换既有前缀的 content epoch，后缀重写按实际冲突缩小副本覆盖。
未采用 speculative 字节可以残留，但不成为可恢复的内容。

`HistoryOwner` 是唯一可写 owner，`HistoryPrefix` 保留同一 history 的 `[0,F)`。
保存非对齐前缀不复制尾页。各 history 按可见内容贡献 required coverage，按最深保留前缀
贡献 protection；独立 writer 或回退写入与保护冲突时，先分支目录并建立必要的尾页 COW。
目录树节点共享不替代内容引用。

Main 与 MTP 各自提交前沿。稳定 MTP 边界是 Main `F`、MTP `max(F-1,0)`；本轮峰值由
模型按 draft/verify 宽度给出。DFlash Full 保留自己的 context frontier；local cyclic KV
属于 State，Vision/query 临时 KV 属于 workspace。

## 6. 成组预留与元数据

Session 先计算完整执行链的 State、缺失副本、尾页 COW、增长页及恢复峰值。
ContextDirectory 的 `assess` 是只读报价，`reserve` 返回拥有实际位置和描述符的 reservation。
任一 rank 或域不足时不得留下部分物理预留。未来通过维护取得的位置也属于原 reservation，
只有维护完成后才能物化。

History 描述符、分支与树节点的写入峰值纳入同一需求。控制元数据按段申请并共用预算，
空段可回收；复用段后的对象 generation 不回退。Checkpoint catalog 在准备时保留条目，
可选保存无配额时跳过，必要暂停在接受前准备 snapshot 的条目和身份。

释放报价分别携带物理副本与逻辑 State/Page/History/catalog 的释放证据。
Device→Host 搬运不等于释放逻辑描述符；只删除别名可能仅释放元数据。共享持有者集合按并集
结算，回收策略重新评估整个需求，不能将各个单独可行的申请合取成成组可行。

普通执行预留本轮峰值；恢复预留跨 chunk 保留“重建到旧 frontier + 首个真实新单元”的空间。
写入准备使用已保留的记录和树节点，不推进已提交前沿。Commit 被接受后采用实际前沿；
消费者退休后释放未采用后缀和余额。需要再次使用的尾部记录直接回到执行前准备的 reservation，
不先释放到全局池再竞争，也不为旧页和下一轮余额暂占两份容量。

## 7. 迁移、维护与取消

普通副本迁移遵循：目的预留 → 来源借用 → 复制 → 校验版本/coverage → 发布 → 释放旧副本。
复制完整连续记录，包括 padding，不重新量化。相邻范围合并，非连续范围分批执行；实际完成
观测记录 rank、方向、字节数、State 数、页组数与 copy operations。

当总容量足够而类别块分布妨碍申请时，ContextPool 返回维护计划。Directory 停止新地址借用，
等待已有消费者退休，保留逻辑内容并复制受影响记录；更新所有活动执行表后才发布新的绑定。
维护不移动 schema、改变内容版本或隐式扩容。

独立 Context work 分为有界批次，整个事务持续持有预留、来源和维护门。
普通取消仍完成已接受的全部搬移动作，随后丢弃请求自己的结果；执行失败则排空各 rank，
放弃未完整执行的计划并使目录进入失败状态，不能把已排空误当作所有搬移已经完成。

Device/Host 的 backing、执行表和 frame 输入地址在 Graph 生命周期内稳定。每次 replay 更新
I64 offsets、表内容、行选择、lengths 与 positions。记录移动不改变 graph key；Graph 不捕获
某个请求或物理记录的固定地址。维护后通过原有输入路径读取新映射。

## 8. 观测

每 rank、每域报告 capacity、live、reserved、pending retirement、padding、同类块空位、
未分配块、尾部字节和峰值占用。`relocation_bytes` 是占用中的搬移部分，不能再次相加；
`future_reserved_bytes` 是未来位置的承诺，不代表另一份已占物理 backing。
State/History/各逻辑 KV pool 的对象数是内容统计，不是独立容量。
控制元数据报告 budget、allocated、reserved。

端到端评估分别检查正常执行开销、同一 backing 容量下的有效缓存量，以及受压情况下的回收、
恢复与前缀命中。仅有正确页表或单个 Op 的结果，不能证明整个调度与迁移链路已满足这些要求。
