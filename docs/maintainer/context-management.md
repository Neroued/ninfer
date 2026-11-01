# Context 管理

Context 同时管理随历史增长的 KV 与某个执行位置的定长 State。两者共享底层容量，
保留不同的内容、共享与写入语义。请求调度和事务发布属于[Engine](engine-architecture.md)。

## 分层与容量

| 模块 | 负责的事实 |
|---|---|
| 模型 Session | State 字段、各 history 的数学前沿、placement、恢复资格和完整工作需求 |
| ResourceManager | 来源选择、语义前缀索引、保留价值、回收权限 |
| ContextDirectory | 对象身份、内容版本、覆盖、引用、预留、迁移与发布 |
| ContextPool | 记录放置、真实可分配性、物理绑定、回收和整理 |
| RankProgram | 固定 backing、执行表、复制和映射更新 |

每个 rank 的 Device backing 由 State、Main/MTP/Draft history 按需使用，没有各自固定配额。
RTX 5090 的 Host backing 是另一个容量域；Spark 不建立重复的 Host Context cache。
跨 rank 或 Host/Device 的空闲字节不能直接互相抵消。

池只接收有限种记录大小与对齐。同类记录占用大块，空块可改用于其他类别，必要时整理部分占用块。
所以空闲总字节充足不等于某个完整需求可分配。准入、回收和启动容量都使用所属分配器的放置计算。
模型不为适配大块几何修改数学状态或 KV 页跨度，Op 也不感知分配类别。

权重、expert cache、PLE 表、workspace、transition records、Vision 窗口和采样 frame 不属于此池。
启动保证必要工作集可执行，剩余容量由执行与可选缓存共同使用；详见[启动预算](engine-architecture.md#启动与资源准备)。

## 身份、内容与地址

StateId 表示某个位置的完整续算状态，可以包含多个 rank-local image。每个 image 连续，
模型决定 recurrent state、conv history、raw index tail、conditioning 等字段及 owner variant。
State 是位置相关的完整内容，不能只把 KV 截短就恢复到较早 State。

History 是逻辑页序列；PageId 表示同 frontier、同生命周期的一组 KV planes。
Main、MTP 和 Draft 保持独立 history，pool ID 表达数学用途，不划分独占容量。
共享前缀引用保护历史的精确范围；只有写冲突才分支并复制必要尾页。

逻辑引用保证内容存在，执行借用保证消费者期间地址稳定。长期 checkpoint 不永久固定物理地址。
记录搬移保留逻辑内容和 allocation 身份，更新绑定；最后引用消失时仍有消费者的记录，
须等待退役再归还空间。

State 的数学写入产生新内容版本，复制和冻结保留版本。KV 合法追加扩大有效 coverage，
不使既有较短 Host 前缀失效；后缀重写按冲突范围撤销覆盖。未采用 speculative 后缀的字节
不成为合法历史。QSA/MLA 的完整 pool 与 raw tail 按各自前沿共同构成恢复点。

独立重算的 State 和 KV 不能仅因 token 相同而拼成一份 checkpoint。输入索引的 digest 只缩小
候选范围，恢复还要验证真实 token、位置、媒体和模型执行身份。索引身份查询的有效边界与
checkpoint 可捕获边界分别解释。

## 执行寻址

KV 页组内打包相关层与 planes；Device/Host 使用相同记录表示。State 字段保持固定相对布局。
活动 history 租用执行表行，表项是相对 backing 的 I64 byte offset；State 也通过 image offsets
访问。Batch 行、lane、table row、token ordinal 和物理 offset 各有独立含义。

图固定 backing 和执行表地址，当前记录位置通过设备数据进入。池整理后更新绑定与映射，
后续执行读取新 offsets；不按记录位置重新捕获图。Op 消费字段或 paged views，
不查询目录、不管理引用、不选择受害者。

布局和读写合同由 [StateImage](../../src/core/state_image.h)、
[Paged KV](../../src/core/paged_kv_storage.h)及具体 Op 维护。

## 预留与持续推进

Session 给出一次可完成工作的需求，包括读源/写目的、增长、共享尾页 COW、speculative 峰值和
提交所需对象。Directory 联合评估所有 rank/domain，并取得物理位置与必要元数据；
Executor 另保留 frame 和收尾额度。失败不留下半份许可。

预留可以包含已存在记录的复用、尚未物化的增长，以及明确消费者退役后的接管。
同一位置在这些状态之间转移，不先释放到公共池再竞争，也不重复计费。
需求或共享关系改变后重新评估；稳定的正常执行只处理实际增量。

普通执行逐单元预留。暂停恢复则保护“重建至旧 frontier，再完成首个真实新单元”的完整需求，
跨 Replay chunk 保留，期间跳过可选捕获。只重建到旧位置不算取得进展。

必要执行优先于可选缓存。空间不足时先回收对短缺域有贡献的可选内容，再按年龄规则暂停或
恢复请求。Snapshot 保留完整上下文，Replay 保留已提交账本和恢复目标；两者沿用同一个请求的
输出、grammar 和 RNG 语义。没有真实释放者时不能无限等待，无法满足合法单请求工作集须报告原因。

## 保存、来源与回收

缓存服务生成终点续接、输入改写恢复、显式锚点和公共共享前缀。Frontend 给出实际序列化边界；
模型验证该位置是否可恢复。稀疏中间 checkpoint 在合适的执行边界尝试，不为精确间隔打断正常工作。
同位置的私有与公共用途可以共享物理内容。

捕获是可选写入，空间不足可以跳过，不抢占必要执行。来源引用、等待保护与实际读者 lease
分别管理。来源查询不分配一份新 State/KV；真正绑定时才取得完整许可和消费保护。

来源排序比较恢复传输与剩余重算成本。成本模型只决定偏好，精确匹配、coverage 与真实放置
决定可用性。已经成功绑定后不再重新选择另一来源。公开 session key 是提示，较早请求晚完成
不能覆盖更新的续接关系。

保留价值来自真实跨请求采用和已提交的重复计算需求；查询、保存、搬运和自身 Replay 不增加热度。
回收按实际持有者集合计算释放量，删除一个别名可能释放零字节。损失由动作后真正幸存的恢复点
决定，不能让同批被删的别名互相充当 fallback。

先比较普通内容与复用内容，再在当前短缺域选择足以满足完整需求的动作。Device→Host 降级也要
先检查 Host 目的和它牺牲的内容，不能只看到 Device 释放。可选捕获只有自己的替换权限，
不能借必要执行的权限扩大缓存。

具体排序、恢复损失与事件驱动准入由
[context_cache](../../src/runtime/engine/context_cache/)维护；物理引用和释放证明由
[ContextDirectory](../../src/models/common/context/directory.h)维护。

## Attention DP

ADP 在同一执行实例内选择 attention owner；完整 DP 则有独立的调度、目录和执行资源。
PlacementId 由模型解释，Runtime 不用其整数值推导 rank 或 State 字段。

选择先看完整资源可行性，再在 token 负载接近的候选中优先本地可恢复前缀。绑定成功即计入
已知 prompt/执行前沿负载，结束或暂停结清；每轮普通 decode 不重新平衡 owner。
[放置策略](../../src/runtime/engine/context_cache/placement_policy.cpp)维护具体比较与平局规则。

同一前缀可以有多个 placement 的完整 checkpoint。远端恢复需要比较转换与重算，
引用同一输入摘要不等于目标侧已经有可用副本。GLM Main/MTP 共用 attention owner，
KDA 分片则由两侧共同持有；改变 owner 不能被误计为释放所有 State。

转换由模型声明字段与 history 关系，Directory 一次取得完整目的与来源保护；数据通道执行有界
批次，全部必要参与者完成后再发布 placement。源与目的并存的峰值必须可行。
压力下可以转换、回收或暂停/Replay，不让两个请求各占半份预留互相等待。

## 迁移与结束

```text
取得完整目的与来源借用 → 提交复制和映射 → 消费者退役 → 发布覆盖/绑定 → 结算来源与余额
```

池整理期间停止新地址借用，等待旧消费者，再复制记录、更新执行表和发布新位置。
普通取消不能提前释放已经被提交的源、目的或 staging；执行错误也不能把排空解释为复制成功。
Context 动作分批仅限制执行资源，整个事务的内容保护持续到最后完成。

统计按容量域区分 backing、实际记录、预留和待退役占用，逻辑 State/history 数量与物理容量
分别报告，详见[观测](observability.md)。
