# GLM5Next 双 Spark 并行与资源设计

性质：待落地的并行与资源设计。本文维护模块分工、权重/状态归属和部署预算；
决策及实现状态统一见[设计索引](README.md)。[模型语义](../../maintainer/glm5next-model.md)
与[数值政策](../../maintainer/glm5next-quantization.md)独立于本文的放置方式。
执行所有权遵循[分布式执行设计](../../distributed-execution-design.md)和
[Engine 架构](../../maintainer/engine-architecture.md)。

## 1. 首版并行分工

同一组两个 Spark ranks 按模块选择分工：MLA/Indexer DP2、KDA TP2、dense/shared
MLP TP2、LM head TP2、routed experts EP2，mHC 权重与四流 residual 复制。
总 GPU 数为 2，各模块并行度不相乘。

| 部分 | 固定的分工 |
|---|---|
| Token embedding | 完整 BF16 表在两台复制；每 rank 内 Main/MTP 共享同一对象 |
| Main 11 层及 MTP 1 层 MLA | 按请求 DP2；Q_a/Q_b/KV_a/KV_b/O 的完整逻辑权重在两台复制；owner 计算该请求的完整 64 heads |
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

DP 的分配单位是请求。普通执行保持请求的 MLA/Indexer owner 稳定，该请求的全部
历史由此 owner 消费；不按序列位置拆开一条历史。KDA 仍由两台共同持有分片，并
处理同一组请求。ModelSession/ContextDirectory 必须同时表达这两种 owner 集合；
checkpoint 与恢复覆盖各自必需的分片，不能按两台都必须有 MLA cache 的规则预留。
跨 owner 的 prefix 恢复或迁移策略在后续上下文接口设计中单独收敛。

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

```text
BF16 latent = layers * L * 1024 B
FP8 latent  = layers * L * (512 + 4*4) B
Indexer    = layers * (L/4) * 128 * 2 B
```

| 范围 | BF16 latent | FP8 codes | FP32 scales | BF16 Indexer |
|---|---:|---:|---:|---:|
| Main 11 层 | 11 GiB | 5.5 GiB | 176 MiB | 704 MiB |
| MTP 额外一层 | 1 GiB | 0.5 GiB | 16 MiB | 64 MiB |
| Main＋MTP | 12 GiB | 6 GiB | 192 MiB | 768 MiB |

这是每份完整 latent cache。第 1 节选定 MLA/Indexer 按请求 DP2，一份请求的完整
latent 只驻留在其 owner rank；单请求的这份容量不除以二。按 heads 做 MLA TP2 的
对照方案则会在两台各保存一份完整 512 维 latent。

Growing cache 合计如下，FP8 列已经包含全部 MLA scales；Indexer 均为 BF16：

| 范围 | BF16 MLA＋BF16 Indexer | FP8 MLA＋BF16 Indexer |
|---|---:|---:|
| Main 11 层 | 11.6875 GiB | 6.359375 GiB |
| MTP 额外一层 | 1.0625 GiB | 0.578125 GiB |
| Main＋MTP | 12.75 GiB | 6.9375 GiB |

三列 K/gate BF16 canonical tail 每层 1536 B，Main 为 16.5 KiB、Main＋MTP 为 18 KiB。
KDA 的完整 34 层 FP32 state 为 136 MiB，三列 BF16 convolution history 为
4.78125 MiB；按 KDA heads TP2 后每 rank 分别为 68 MiB、2.390625 MiB。

采用已选定的「MLA/Indexer DP2、KDA TP2」，计入一份本地 KDA state/history 与 tail：

| 单请求在指定 rank 的占用 | Main | Main＋原生 MTP |
|---|---:|---:|
| Owner：FP8 MLA＋BF16 Indexer＋本地 KDA state/history＋tail | 约 6.43 GiB | 约 7.01 GiB |
| Owner：BF16 MLA＋BF16 Indexer＋本地 KDA state/history＋tail | 约 11.76 GiB | 约 12.82 GiB |
| 非 owner：该请求的另一份 KDA state/history 分片 | 约 0.069 GiB | 约 0.069 GiB |

表中按各层均满 1M 计费，真实 Main/MTP 各自 frontier 的有效行数按 family 合同计算。
不计执行 records/workspace、Graph、额外 StateImage/checkpoint 副本、page tables、
分配对齐及其他小型状态字段；MTP conditioning 等仍需单独计费。多个无共享前缀的请求
按各自 owner 和长度累加 growing cache；两台都持有各请求的本地 KDA 分片，其容量
按状态实例数增加，不随 L 增长。Prefix restore 除 token/latent 前沿外，还要恢复两台
的 KDA state/convolution history，以及 owner 上的 Indexer tail 和 MTP 状态。

### 3.1 Page、Indexer 与 raw records 的补充计费

每层每原 token 的 growing payload 为 BF16 MLA＋BF16 Indexer 的 1088 B，
或 FP8 MLA＋BF16 Indexer 的 592 B；对应每个 64-token page 为 68 KiB 或 37 KiB。
每层每有效 raw record column 为 `128*2*2=512 B`，跨层 frame 两个 planes 共
`512*W*L*R` B，不含 slab 对齐。W、L、R 分别是 raw record 窗口容量、层数和行容量，
布局按 Kpool 设计；update 出口不要求保留整段 raw records。

1M 时单层 Indexer 有 262144 个 BF16 pooled keys，共 64 MiB。MLA 选中 2048 个
tokens 的 latent payload 为 BF16 2 MiB 或 FP8 1.03125 MiB。这是唯一数据量，
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
为对照，改为本节 DP2 会去掉一份重复历史，同时增加 Q_b/KV_b/O 权重复制。
BF16 Q_b 为 48 MiB、BF16 KV_b 因子为 32 MiB、Q8 O 为 68 MiB，每层合计
148 MiB；TP2 时每 rank 为 74 MiB，Main＋MTP 共增加 **0.8671875 GiB/rank**。

对于单个 1M 请求，Main＋MTP 的一份 growing cache 为 6.9375 GiB（FP8 MLA）或
12.75 GiB（BF16 MLA），均包含固定的 BF16 Indexer，FP8 列包含全部 FP32 scales。
去掉副本并扣除两台合计 1.734375 GiB 新增权重后，双机 payload 分别净减少
5.203125 / 11.015625 GiB。Owner 仍需
保存完整历史，DP 不把单请求在 owner 上的 cache 减半；最终启动占用还取决于固定
pool 预算、临时数据与通信资源。

相对非 routed-expert 全部 DP 的方案，保留 KDA/shared/head TP2 是低并发性能的
主要预期收益来源。KDA Q/K/V/O 每层 Q8 权重为 136 MiB，TP2 后每台 68 MiB；
4096 维 BF16 block 输出为 8 KiB/token。这个数据量关系支持小 batch 下分摊权重读取
与计算，但实际收益取决于 kernel 效率、通信延迟和完整层序。MLA DP 的直接目标是
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
