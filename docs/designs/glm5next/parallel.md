# GLM5Next 双 Spark 并行与资源设计

性质：待落地的并行与资源设计。本文维护模块分工、权重/状态归属和部署预算；
决策及实现状态统一见[设计索引](README.md)。[模型语义](../../maintainer/glm5next-model.md)
与[数值政策](../../maintainer/glm5next-quantization.md)独立于本文的放置方式。
执行所有权遵循[分布式执行设计](../../distributed-execution-design.md)和
[Engine 架构](../../maintainer/engine-architecture.md)。

## 1. 首版并行分工

一个完整执行实例使用两个 Spark ranks，即 DP1；实例内部 MLA/Indexer ADP2、KDA TP2、
dense/shared MLP TP2、LM head TP2、routed experts EP2，mHC 权重与四流 residual 复制。
总 GPU 数为 2，各模块并行度不相乘。DP 与 ADP 的命名、两级路由及实例边界统一见
[Attention DP 设计](../attention-dp-placement.md#11-dp-与-adp)。

| 部分 | 固定的分工 |
|---|---|
| Token embedding | 完整 BF16 表在两台复制；每 rank 内 Main/MTP 共享同一对象 |
| Main 11 层及 MTP 1 层 MLA | 按请求 ADP2；Q_a/Q_b/KV_a/KV_b/O 的完整逻辑权重在两台复制；owner 计算该请求的完整 64 heads |
| Main/MTP Indexer | 与对应 MLA 共用请求 owner；完整投影、32 个 query heads、pooling、评分与 selection 由 owner 计算 |
| MLA/Indexer 历史 | 一个请求的完整 latent、pooled index cache 和未完成 pool tail 只在 owner 保存；Main 与 MTP 同 owner、不同 pool/frontier |
| KDA | 按 heads TP2，每台 32 heads；Q/K/V/O、beta、f_b/g_b、卷积及大状态按相应 heads 切分；f_a/g_a 保持复制 |
| Main 前三层 dense MLP | intermediate 12288 分为 6144＋6144，down 产生局部贡献 |
| Main/MTP shared expert | intermediate 2048 分为 1024＋1024；本地 shared 与 routed 输出合并后统一归约 |
| Main/MTP routed experts | EP2，每台 144 个完整 experts，expert 内部不再 TP；global Top8 与 route 系数保持完整语义 |
| Router | 完整权重及控制计算复制，消费一致的 block input |
| Main mHC | 两台复制全部权重与完整四流 residual，处理同一组物理列；完整 block 输出求和／汇集后，各自执行 post 与下一 pre |
| Main LM head / optimized draft head | vocabulary TP2，每台各 `[77440,4096]`；Main 为 BF16，额外 draft 为 Q4 A16；未启用 optimized head 时 MTP 使用 Main head |
| MTP stem／普通 residual／norm | 完整 BF16 stem 和 norms 复制，两台处理同一组物理列；MLA 输出汇集、MoE 局部贡献求和后进入 residual |
| Vision | 完整 BF16 tower 复制，按独立 temporal-grid attention segment 分配任务；完整 features 在进入 Main 前按原顺序交换 |

ADP 的分配单位是请求。普通执行保持请求的 MLA/Indexer owner 稳定，该请求的全部
历史由此 owner 消费；不按序列位置拆开一条历史。KDA 仍由两台共同持有分片，并
处理同一组请求。模型使用 ContextDirectory 的 State variants 与 history placement
表达两种 owner 集合；checkpoint 与恢复覆盖全部必需分片，只在 attention owner
预留 MLA history。动态 owner 选择、来源恢复与容量压力处理由
[Attention DP 设计](../attention-dp-placement.md)维护；下文定义转换所需的模型布局。

MLA 向后续 TP/EP 计算交付完整的 4096 维 block 输出；KDA、dense/shared 和 routed
的局部贡献在 Main mHC 或 MTP residual 消费前完成求和。Main MLA owner 的输出
按冻结的请求/token 映射汇集为两台一致的完整 BF16 y，再进入各自 mHC post。
mHC 不按 stream、hidden 或 owner 切分，不承担通信；其复制不增加 MLA/Indexer
cache 副本。两台采用一致的输入 words、物理列映射和精度策略。

具体 collective、pack/unpack、模型 Graph 与通算融合仍由后续执行接入设计；输入
交换、输出归约与布局转换统一安排，避免为同一交接重复通信。mHC 接收完整 y 的
接口与 residual 放置不再作为未定项。已有基础清单的 KDA/FFN/head 局部几何继续适用。

本地 vLLM 的 `TP=1, DP=2, enable_expert_parallel=true` 会将 KDA、dense/shared、
head 等非 routed-expert 权重一并复制。它的 MLA 与 KDA 使用同一个 TP group；本节
采用的是按模块选择的混合分工，不能直接等同于该 vLLM 参数组合。

### 1.1 Context 容量与记录归属

每个 rank 使用一份 Device Context backing，State、Main history 和 MTP history
共同使用其容量。Spark 不建立重复的 Host Context cache。分配、成组预留、借用、
搬移和退役遵循[Context 存储合同](../../maintainer/paged-kv-cache.md)。

模型准备时声明 State variants、Main/MTP 页记录与各 rank 的容量；数学层的 history
pool ID 只区分用途和布局。已有目录支持不等容量、State 分片和 owner-only histories。
GLM Session 将本轮 State 目的、缺失副本、尾页 COW、增长页及提交峰值组成完整需求，
通过目录统一评估和预留；不能分别以“KV 够用”和“State 够用”替代成组可行性。

两 rank 分别使用自身容量。准入时结合可恢复来源与完整需求选择 owner，普通执行
期间保持 owner；另一 rank 的空闲字节不直接成为当前 owner 的可寻址历史。
启动按每个可选 owner 检查完整工作集，固定 workspace/frame/执行表/Graph 在 Context
之外计费，算子容量查询不包含持久记录或池管理开销。

### 1.2 State variants 与字段布局

一份逻辑 StateId 包含两个 rank-local images。模型按 attention owner 声明两个
variants：owner=0 时 rank0 使用 owner image、rank1 使用 non-owner image；owner=1
时相反。每个 image 连续，字段使用 `StateImageBuilder`，两种布局的公共前缀相同：

| 字段顺序 | 单层形状／层数 | 放置 |
|---|---|---|
| KDA conv | BF16 `[12288,3]` × 34 | 每 rank 的本地 head 分片 |
| KDA recurrent | FP32 `[128,128,32]` × 34 | 每 rank 的本地 head 分片 |
| Main final hidden | BF16 `[4096]` × 1 | 两 rank 复制；续算与 MTP conditioning 共用 |
| Main raw K、raw gate | 各 BF16 `[128,3]` × 11 | owner image 末端 |
| MTP raw K、raw gate | 各 BF16 `[128,3]` × 1 | 启用 MTP 时追加于 owner image 末端 |

每个字段及 layer stride 按 256 B 对齐。Non-owner image 为 **70.398438 MiB**；
owner image 在无 MTP／有 MTP 时分别为 **70.414551／70.416016 MiB**。
这些是当前几何的记录大小，最终注册使用布局计算结果。Frontier、pool 相位和 owner
属于模型 metadata；无 RoPE 的 GLM 不增加 QSA 坐标字段。

KDA views 在两种布局中使用相同公共字段偏移；MLA 仅为 owner 行绑定 tail，临时
draft tail 则使用 frame 字段。Ops 不读取 variant 或 owner，只接收实际布局与 offsets。
完整 checkpoint 同时持有两 rank State 分片、owner 上的 Main/MTP history 和输入身份。

同 owner 恢复复用现有目录合同。跨 owner 恢复需要模型安排新的 variant、history
placement、公共字段复制和 owner-only tail／history 传输，并在全部完成后发布。
当前普通 `begin_write` 和 history binding 会拒绝不同 variant／placement，不能用整份
image 的字节复制替代这种转换；该事务随 G3 集成落实。模型声明字段转换，共享
Context 事务负责预留、复制和发布，Op 与底层分配器不解释 owner 转换的模型语义。

## 2. 参数合并与本地几何

纯投影及本地 FFN/MoE 的 TP2/EP2 几何、P/F/M/R/E 条目由
[基础 Op 清单](op-checklist.md)维护。KDA Q/K/V 分别按 heads 切片后按 Q/K/V 行序
合并；BF16 beta、f_a、g_a 不混入 Q8 parent。KDA O 按对应 heads 的 K 轴切片。
Convolution 的 channel/tap 映射与投影一致，转换时保留源 FP32 words 和时间顺序，
按 [KDA 布局](replayssm-kda.md#4-输入-projectionconv)转为每个 tap 的 channels 连续。

Dense/shared 的 GU 沿 intermediate 切分，down 沿对应 K 轴切分；shared 的 GU/down
均使用 Q8 A16，K 切点保持 G32 对齐。Routed bank 每台
持有 144 个完整 experts，expert 内部不再 TP。Global ID 到本地 bank 的映射显式提供。
量化域和 GU 合并的数值条件见数值文档，不随这些分片变化。

MLA/Indexer 的 W_A/W_B 合并、U_K/U_V 因子布局与独立 O 由
[Kpool Sparse MLA](kpool-sparse-mla.md#2-权重投影与执行链)维护；mHC 的 F 行序与
四流布局由 [mHC 设计](mhc.md#1-权重与-tensor-布局)维护。每份物理布局只保留一个定义。

### 2.1 Vision 权重的保值布局

Vision 权重保留 BF16，转换期确定与输入一致的 feature 顺序：

- Patch projection 展平为 `[1024,1176]`，输入 K 轴按源 `(channel,temporal,y,x)`
  排列；processor 直接产生这一顺序的 patch，不在执行时转置权重。
- QKV 保持 `[Q;K;V]`，GU 合并为 `[gate;up]`；对应 bias 使用相同行顺序。
- Downsample 输入按四个 spatial patches、每个 1024 channels 连续排列，
  `p=2*dy+dx`。将源 `[4096,1024,2,2]` 保值重排为 `[4096,4096]`：
  `W_flat[out,p*1024+channel]=W_source[out,channel,dy,dx]`。
  Encoder 输出按此顺序组织后直接消费，不再为 Conv2d 做完整 NCHW 转置。
- Attention O、down、merger projection 保持独立矩阵；merger GU 同样合并 gate/up。

Vision 按第 1 节复制权重、分配完整 segments；processor 直接产生上述列顺序。
具体 Op、零拷贝 downsample view 和媒体交接见 [Vision 设计](vision.md)。

### 2.2 Embedding 与 MTP stem

每 rank 保存完整 BF16 embedding 1210 MiB，Main/MTP 共享；Main head 则继续按词表
TP2 切分，两者不是同一对象。MTP stem 的 `[4096,8192]` 权重每 rank 为 64 MiB，
输入按 `[embedding_norm;conditioning_norm]` 排列。Stem、普通 residual 与 norms
复制计算；MLA/Indexer 仍仅在请求 owner 计算和保存历史。

这使 stem 不需要独立的输入交换，并维持后续 TP/EP 的完整 hidden 输入；代价是
相比 embedding vocabulary TP2，每 rank 增加 605 MiB 的表载荷及对应本地计算。
MTP 的配对、历史补齐与首步采用见 [MTP 设计](mtp.md)。

## 3. 1M 上下文的持久状态容量

令 `L=1048576`、B=1，按已定 [family layout](kpool-sparse-mla.md)计算。
Main 有 11 层 MLA，每 token、每层保存一个 512 维 latent：

Latent 原始 payload 为 BF16 的 1024 B/token，或 FP8 的 512 个 codes＋4 个 UE8M0
scales，即 516 B/token。Indexer 每四 tokens 一个 pooled key，BF16 为 256 B，FP8
为 129 B；每个 64-token 页组的 FP8 index scale plane 另按 256 B 对齐。

下面容量已含完整 planes 的对齐，未计其他资源；列名为 latent/index 格式：

| 范围 | bf16/bf16 | bf16/fp8 | fp8/bf16 | fp8/fp8 |
|---|---:|---:|---:|---:|
| Main 11 层 | 11.6875 GiB | 11.38671875 GiB | 6.23046875 GiB | 5.9296875 GiB |
| MTP 额外一层 | 1.0625 GiB | 1.03515625 GiB | 0.56640625 GiB | 0.5390625 GiB |
| Main＋MTP | 12.75 GiB | 12.421875 GiB | 6.796875 GiB | 6.46875 GiB |

Indexer 默认为 FP8。格式选择见[数值政策](../../maintainer/glm5next-quantization.md#3-mlaindexer-cache-与计算政策)。
MLA/Indexer 按请求 ADP2，一份请求的完整历史只驻留在其 owner rank，单请求的这份
容量不除以二。

三列 K/gate BF16 canonical tail 每层 1536 B，Main 为 16.5 KiB、Main＋MTP 为 18 KiB。
KDA 的完整 34 层 FP32 state 为 136 MiB，三列 BF16 convolution history 为
4.78125 MiB；按 KDA heads TP2 后每 rank 分别为 68 MiB、2.390625 MiB。

采用已选定的「MLA/Indexer ADP2、KDA TP2」，计入第 1.2 节的一份本地 StateImage：

| 单请求在指定 rank 的占用 | Main | Main＋原生 MTP |
|---|---:|---:|
| Owner：FP8 MLA＋默认 FP8 Indexer＋StateImage | 约 6.00 GiB | 约 6.54 GiB |
| Owner：BF16 MLA＋默认 FP8 Indexer＋StateImage | 约 11.46 GiB | 约 12.49 GiB |
| 非 owner：本地 StateImage | 约 0.069 GiB | 约 0.069 GiB |

表中按各层均满 1M 计费，真实 Main/MTP 各自 frontier 的有效行数按 family 合同计算。
不计执行 records/workspace、Graph、额外 checkpoint images、执行表与池装载余量。
Main final hidden 已计入 StateImage，MTP 不再为 conditioning 增加第二份字段。
多个请求按各自 owner、长度和实际共享关系累加 history；两台都持有各请求的本地
State 分片，其容量按状态实例数增加，不随 L 增长。恢复来源须覆盖完整 checkpoint。

### 3.1 Page、Indexer 与 raw records 的补充计费

每个完整 64-token 页组记录如下；格式列为 latent/index：

| 完整页组记录 | bf16/bf16 | bf16/fp8 | fp8/bf16 | fp8/fp8 |
|---|---:|---:|---:|---:|
| Main 11 层 | 748 KiB | 728.75 KiB | 398.75 KiB | 379.5 KiB |
| MTP 1 层 | 68 KiB | 66.25 KiB | 36.25 KiB | 34.5 KiB |

表中已含当前 planes 的 256 B 对齐；精确大小由 `KVPageLayout` 给出。Context 实际
需求还包含记录类别的装载余量，使用 `describe_context_pool()` 的几何和成组需求
计算，不能把上面的 payload 相加当作池容量。执行表按活动 rows 与最大上下文规划，
与池里保留的 inactive histories 数量分开计费。

每层每有效 raw record column 为 `128*2*2=512 B`，跨层 frame 两个 planes 共
`512*W*L*R` B，不含 slab 对齐。W、L、R 分别是 raw record 窗口容量、层数和行容量，
布局按 Kpool 设计；update 出口不要求保留整段 raw records。

1M 时单层 Indexer 有 262144 个 pooled keys，BF16 原始 payload 为 64 MiB，FP8 为
32.25 MiB（包括 scale，不含 plane padding）。MLA 选中 2048 个 tokens 的 latent payload
为 BF16 2 MiB 或 FP8 1.0078125 MiB。这是唯一数据量，
不代表实测 HBM 流量或耗时。

## 4. mHC 执行资源

以下为单 rank 的 payload，不含 arena 对齐、私有 workspace 或其他模型模块：

| 对象 | 容量 |
|---|---:|
| 单 site 的 BF16 F | 786432 B = 0.75 MiB |
| 90 个 site 的 BF16 F | 67.5 MiB |
| 90 组 BF16 base＋scale | 4860 B |
| 90 个 branch gamma | 720 KiB |
| Main final gamma | 8 KiB |
| 每物理 token 的四流 residual | 32768 B = 32 KiB |
| 每物理 token 的 block input 或 output | 各 8192 B = 8 KiB |
| 每物理 token 的一组 FP32 post＋C | 80 B |

每组 FP32 系数只有 4 个 post 与 16 个 mix，pre 不跨分支保存。融合调用的新旧
系数不 alias，可交替使用两组共 160 B/token；独立 residual 双缓冲为 64 KiB/token，
合法 in-place 路线可减少相应公共缓冲。实际 workspace 必须另按容量查询计费。

这些激活随当前物理窗口 `M=T*B` 增长，按执行层序复用，不按 45 层全部保留。
例如一份 `M=4096` 的四流 residual 为 128 MiB。这里的 M 不是请求累计上下文 L，
不能用 1M 历史长度直接为每层常驻分配 mHC 激活。

mHC 的非历史属性与 MTP conditioning 关系见[逻辑状态](../../maintainer/glm5next-model.md#7-逻辑状态与历史依赖)。
这些缓冲属于 RankProgram frame，不增加 growing KV pool 或请求 StateImage 字段。

## 5. KDA records 容量与成本

按 [KDA record 布局](replayssm-kda.md#3-几何行身份与-record-布局)，L 为本地层数，
T 为窗口宽度，R 为 record 行容量，C 为 Q/K/V conv channels。以 BF16 K/V/conv、
FP32 log_decay/beta 计算：

```text
StateBytes/instance = L * (4*Hv*Dv*Dk + 2*C*3)
RecordBytes/column/row = L * (2*C + 2*Hq*Dk + 2*Hv*Dv + 4*Hv*Dk + 4*Hv)
RecordAllocationPayload = T * R * RecordBytes/column/row
```

GLM TP2 单 rank 的载荷为：

| 内容 | MiB |
|---|---:|
| 一份 recurrent state | 68.000000 |
| 一份 conv history | 2.390625 |
| 一份 recurrent＋conv | 70.390625 |
| 每列 conv records | 0.796875 |
| 每列 key records | 0.265625 |
| 每列 value records | 0.265625 |
| 每列 log_decay records | 0.531250 |
| 每列 beta records | 0.004150 |
| 每列每行全部 records | 1.863525 |
| T=6、R=1 的 records | 11.181152 |
| T=16、R=1 的 records | 29.816406 |
| T=6、R=8 的 records | 89.449219 |
| T=16、R=8 的 records | 238.531250 |

以上不含 alignment padding、输入/输出、其他模型 records、Graph 或 source/destination
image 的数量。每列完整 state snapshot 与 raw records 的载荷比约 37.77，不能作为
加速比。成本变化是减少逐位置大 state 写入，增加紧凑 records 写入及接受后的一次
recurrent fold；fold 不重跑权重投影或 readout。

## 6. 并行方案的容量与性能判断

以 MLA heads TP2、latent/Indexer 历史两台复制，且 prepare/Indexer 权重保持复制
为对照，改为本节 ADP2 会去掉一份重复历史，同时增加 Q_b/KV_b/O 权重复制。
BF16 Q_b 为 48 MiB、BF16 KV_b 因子为 32 MiB、Q8 O 为 68 MiB，每层合计
148 MiB；TP2 时每 rank 为 74 MiB，Main＋MTP 共增加 **0.8671875 GiB/rank**。

对于单个 1M 请求，Main＋MTP 的一份 growing cache 为 6.46875 GiB（FP8 MLA）或
12.421875 GiB（BF16 MLA），均包含默认 FP8 Indexer、UE8M0 scales 和 plane 对齐。
去掉副本并扣除两台合计 1.734375 GiB 新增权重后，双机 payload 分别净减少
4.734375 / 10.6875 GiB。Owner 仍需
保存完整历史，ADP 不把单请求在 owner 上的 cache 减半；最终启动占用还取决于各 rank
统一 Context 容量、临时数据与通信资源。

相对非 routed-expert 全部按请求复制计算的方案，保留 KDA/shared/head TP2 是低并发性能的
主要预期收益来源。KDA Q/K/V/O 每层 Q8 权重为 136 MiB，TP2 后每台 68 MiB；
4096 维 BF16 block 输出为 8 KiB/token。这个数据量关系支持小 batch 下分摊权重读取
与计算，但实际收益取决于 kernel 效率、通信延迟和完整层序。MLA ADP 的直接目标是
减少 cache 重复，不承诺其单请求延迟优于 MLA TP2。尚无 GLM 双机端到端性能结果。

mHC 复制方案保持既有 TP/EP 输出交接。若把 residual 仅放在请求 owner，每个
KDA/FFN 分支需安排输入分发和输出回收；是否有收益必须包含通信阶段共同测量。
当前选择复制，没有实测对照证明其性能优势。

### 6.1 当前量化方案的权重预算

全模型编码载荷见[数值文档](../../maintainer/glm5next-quantization.md#6-权重容量)。
按 Main/MTP/Vision 与 optimized draft head 全部启用，embedding、MTP stem 和
Vision 权重复制，head 词表 TP2，其余使用第 1 节分工，每 rank 的权重载荷约
**91.462 GiB**。这是已定布局的 payload 估算，不是启动实测值；其中 Vision 的
完整载荷为每 rank 1.049837 GiB，不能再按任务分配比例削减权重预算。

Main/MTP 的 12 个 O 从 BF16 改 Q8，每 rank 少 720 MiB；43 个 shared FFN 的
TP2 分片从 BF16 改 Q8，每 rank 少 483.75 MiB。合计每 rank 少 **1.175537 GiB**。
此预算包含 codes/scales 和额外 Q4 head，不包含 backing 对齐、KV、StateImage、
workspace、Graph 或其他运行资源。

## 7. 接管与退出条件

后续执行接入明确 collective、pack/unpack、Graph、前缀恢复与跨 rank 提交，并对
完整路径验证。具体未决工作集中在设计索引，不在各模块文档维护重复任务列表。
实现落地后，由实际 placement/binding、模型执行和测试接管；必要的当前执行说明
归入届时建立的 `glm5next-execution.md`。预算公式随实际几何和布局维护，移入相应
资源说明后删除本设计，不保留第二套已过期执行规格。
