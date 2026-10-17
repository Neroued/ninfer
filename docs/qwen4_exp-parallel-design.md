# Qwen4Exp 双 Spark 并行策略（D0）

状态：首版 TP2/EP2 并行策略已收敛，尚未实施。本文维护首个 Qwen4Exp TP2/EP2
模型计划的放置、计算与状态交接；单列的后续优化不属于首版实现要求。执行框架和生命周期由
[分布式执行设计](distributed-execution-design.md)定义。
模型数学与存储表示以[模型与权重](maintainer/qwen4_exp-model.md)、
[执行与状态](maintainer/qwen4_exp-execution.md)及 Op 合同为准。

目标为两个同构 DGX Spark，每机一个 GPU rank。使用现有 Flash Next 量化产物，
Main routed experts 为 NVFP4 / AllowA4，MTP routed experts 为 block-FP8 / AllowA8；
其他投影保持各自 A16 合同，KV 表示独立。Spark 使用 GPU 模型计算和 Device State/KV。

## 1. 确定的首版策略

| 部分 | 首版选择 | 实施依赖 |
|---|---|---|
| GDN / sparse GQA | 按 head TP2，O projection 产生局部贡献 | 合并参数切片、卷积重排、本地状态与 Op 几何 |
| Residual / HC | 所有 M 使用复制路径 | 完整四-stream residual，block 归约后再 combine/norm |
| Indexer | 四个 query heads、计算、cache 和 raw tail 均复制 | 共同 decision profile、完整 append 与一致 selection |
| Router / shared | Router 复制，shared intermediate TP2 | 全局 route 与本地 bank 解耦、shared 独立几何 |
| Routed experts | EP2，每层各 256 个完整专家，启动时确定静态映射 | Main 按已有逐层优先级交错分配；MTP 按 global ID 奇偶分配 |
| Embedding / head | Embedding 复制，head 按词表 TP2 | 候选交换、约束、penalty、MTP 验收和 scoring |
| PLE | 按表行分片、提前 lookup；后续投影和卷积复制 | 每机约 13.56 GiB 表，合并 lookup 输出，与 Main 前序计算重叠 |
| MTP | QSA TP2 / experts EP2，共用分片 head；stem 两个 FC 复制 | 设备侧 draft 交接、Main verify 与 history 对齐 |
| Vision | encoder 权重复制，独立编码单元分配给不同 rank | 编码边界、特征顺序、缓存和多媒体输入分发 |
| 状态 | 大状态随 head 分片，小型公共字段复制 | ordinary、verify、checkpoint/restore 的共同前沿 |

`T` 为每行物理输入列数，`B` 为紧凑 batch 行数，`M=T*B`。布局与执行策略消费
实际 M、每行有效长度和上下文，不按 prefill/decode/verify 名称决定分派。
本轮依据源码与已有单机测量审查数学、接口、加载、通信、状态和容量；双机延迟尚未测量。
首版的 HC token/lowrank TP、Indexer query/history 分工、MTP FC TP 与单 owner MTP
均不启用；后文的候选分析保留为后续优化依据，不阻塞当前接口和加载计划。
PLE 表分片见第 7 节，完整计算与 Graph 组织见第 10 节。

## 2. 局部几何与加载

矩阵 shape 采用 `[N,K]`。下表使用完整复制 Indexer 的方案：

| 角色 | 每 rank 的几何 |
|---|---|
| GDN heads | Q/K 各 8，V 24，dimension=128；每个 Q/K 对应三个连续 V heads |
| GDN QKVZ | `[8192,2560]`，Q 1024、K 1024、V 3072、Z 3072 |
| GDN AB / output | `[48,2560]` / `[2560,3072]` |
| GDN convolution | 四个 taps，各含本地 Q/K/V 的 5120 通道 |
| QSA heads | Q 12、KV 1，dimension=256 |
| Main QSA Q/gate | `[6144,2560]`，Q/gate 各 3072 |
| Main QSA KV/index | `[1152,2560]`，K/V 各 256、index-Q 512、index-K 128 |
| Main/MTP QSA output | `[2560,3072]`，保留各自原有表示 |
| MTP QSA joint input | `[7296,2560]`，保留完整 index-Q/index-K |
| Router / shared score | `[512,2560]` / `[1,2560]`，复制 |
| Shared GU / down | `[640,2560]` / `[2560,320]` |
| 合并 router/score/shared GU | 若保留一次投影，为 `[1153,2560]` |
| Routed experts | 256 个完整 experts；各自 GU `[1280,2560]`、down `[2560,640]` |
| LM head | `[124160,2560]`；global vocabulary ID 范围互不重叠 |

Config 保留全局几何，ParallelPlan 产生本地参数、状态 schema 和通信描述。合并矩阵
按逻辑角色分别切片后建立本地 parent；卷积的 tap/channel 行序随本地 Q/K/V 重建。
Q8 output 的 K 切分保持 G32 编码分组；expert EP 保留完整 expert 的 codes/scales/divisor。
加载直接物化本地编码与稳定视图，不先加载完整 GPU 模型再切片，不在执行期重新量化。

这部分的数学闭合性可以确定：GDN 的归一化、卷积、递推均在完整 head 内完成，
8 个 Q/K heads 与对应 24 个 V heads 一起分配；A_log、dt_bias、AB 和 state 使用同一
V-head 范围。QSA 的每个 KV head 原本服务 12 个 Q heads，TP2 恰好完整切开这两组。
Head 内 RMSNorm、RoPE、KV Hadamard 与存储量化保持原 dimension，不引入归约通信。

需要修改的是真实的本地参数与执行几何。当前 GDN/QSA 调用使用全局 Config，QSA Op
合同仍写明 24/2 heads；仅对权重切片不能完成接入。LocalParameters、workspace 查询、
prepare、attention、snapshot/record/fold 及 State/KV schema 都要消费派生后的几何。
GDN QKVZ、AB 与 QSA joint parent 不能按总 N 直接对半截取；应分别截取每个角色，
再按原来的角色次序拼接。本轮沿用纯 Linear 几何先支持、再接 split/融合出口的工作顺序。

vLLM 的 QSA 同样采用 Q/KV head TP 和 RowParallelLinear output，并在 TP 大于 KV
heads 时复制 KV。这个做法支持本方案的 head 分工；其输入编码和私有融合不能替代
NInfer 对 Q8 分组、BF16 出口及本地状态的处理。

## 3. Residual 与 HC

首版每个 rank 持有完整 `[2560,4,M]` residual，HC norm、down/injection、up 和
mix 重复执行。GDN/QSA 和 MoE 各产生 `[2560,M]` 局部贡献，完成归约后再 combine/norm。
正常 Main 层对应两个主要归约边界，shared 的贡献并入 MoE 边界。

vLLM 的 HC down/up/injection 也复制；其 Qwen4Exp 明确拒绝
`use_sequence_parallel_moe`。因此大 M 的 HC token 切分需要建立完整的新执行路径，
不能把其他模型的 sequence-parallel 开关直接套入本模型。

该候选在数学上可行：block 局部贡献 → ReduceScatter token 行 → 完整单-token
residual combine/norm/HC mix → AllGather `[2560,M]` 输入 → 下一段 TP/EP 计算。
每个 token 的四条 streams 留在同一个 owner，residual 和 injection 在连续层间维持
相同的 token 分区。GDN/QSA/experts 仍消费所有 tokens，本地 HC 才只处理自己的部分。
这不把 GDN 的时间序列切开，也不改变 expert 分工。

TP2 下 RS/AG 的主体字节量与同尺寸 AR 相当，但调用数由一次变为两次。小 M 的 HC
主要读完整权重，分 token 不减少每机读到的权重，收益很小；大 M 才有分摊计算的价值。
完整 variant 还要解决三个交接：

- PLE 使用跨 token 的九列 history。可在 PLE 之前维持复制 residual，PLE 之后再进入
  token 分区，避免只为这一个层增加时序 halo 或一次完整 residual 汇集。
- Final HC 可按 token 继续计算，但 Main 的 sample/conditioning 必须按实际消费者
  汇集。普通生成只取末列；MTP history 需要其配对的 Main conditioning 列，不能只取末列。
- `[T,B]` 的有效列、padding、token owner 和输出行序由同一 variant 固定；不能把
  一段扁平 token 切片重新解释为一条序列。必要的 pack/unpack 和通信缓冲计入成本。

另一候选是按 lowrank=320 切成 160+160：本地 down/SiLU → up 的 10240 维局部和
→ 归约 → sigmoid/mix；四行 injection 复制。单 rank 少读约 6.25 MiB 权重，但增加
`10240*M` 的通信与新的 HC 分段接口。此项保留为待测候选，不作为首版默认布局。
`g` 原本是 HC 的内部量，其通信表示要满足完整 HC 数值合同，不能直接套用普通
Linear 出口的 BF16 舍入边界。

低秩切分不能先在两机分别做 sigmoid 再相加。按 residual feature/stream 切分虽然也
成立，却会在 norm、down 或最终 stream mix 之间增加归约；其完整通信链比只看某个
Linear 的减半更重要。目前没有证据支持它优于复制路径或大 M 的 token 分区。

已有迁入阶段的 Spark Graph/cold 微基准，三轮 GPU 中位数范围：HC mix 的
M=1 约 56–58 μs，M=64 约 58–59 μs，
M=4096 约 1.13–1.14 ms。按仓库持续读参考 257.42 GB/s，6.25 MiB 的读取量对应
约 25 μs；这是判断分片潜力的量级，不是分布式收益测量。

## 4. QSA Indexer

每个 rank 复制四个 index query heads、一个 raw index key、compressed cache 和 raw tail。
主 attention 的 Q/K/V 按 head 切分。最大上下文下，Main 12 层 BF16 index cache
约 192 MiB/rank/请求。

基础路径重复计算完整 Indexer，直接供本地 sparse GQA 使用。vLLM 同样复制
index-Q/K 投影和 index cache。这里需要一致的是 selection 集合与 cache 内容：
公共 Op 的精度容限本身并不保证两个不同实现选择相同的临界候选。
同构双 Spark 的复制区域使用相同输入、T/B、envelope、计算路径和 global-ID
tie-breaking；当前 score kernel 独立生成每个分数，TopK 按 score/ID 全序选择，
不依赖浮点 atomic 累加完成顺序。其他分片 Op 仍保留自己的私有分派。
将来采用不同设备或不同数值路径时，需改为一次生产、显式分发共同 selection。

大 M 候选按 query 分工，每机仍读取完整 index cache，随后交换最终 block IDs/counts。
每 query 最多 512 个 I32 IDs；M=4096 的完整 IDs 为 8 MiB。切换还取决于上下文长度，
短上下文的全选路径不承担重复评分成本。

现有 `qsa_prepare` 同时追加主 KV、压缩 index key、记录 raw tail，并生成全部 Q。
它仍必须处理完整输入窗口；分 query 应发生在准备之后。当前 Indexer 要求连续
`[128,4,T,B]` 输入及 `[512,T,B]` 输出，不能直接把跨 batch 的半个 T 视为连续 Tensor。
需增加明确的 query 工作域/输出视图，或显式打包本地查询与 metadata；成本一并计算。
每个 query 只有一个 selection 生产者，汇集后两边 GQA 才消费完整结果。

Prefill 的后半段 query 可见历史更多，按 query 个数各半不保证负载均衡。分配至少要
依据可见块数和全选区间；多请求时还需考虑各行上下文差异。是否分工应比较
`完整 Indexer 耗时` 与 `较慢分片耗时 + 暴露的交换/布局成本`。
减少 query 也会降低 K tile 的复用，不能默认计算量减半就会耗时减半。

按 query heads 切分需要在 TopK 前合并完整 score：必须在每个 head 的 ReLU 后求和，
不能先选各 head 的局部 TopK。该方案还不减少 key cache 扫描量。
按历史 blocks 切分可减少扫描：各分片计算完整四-head score，交换局部 Top512 并归并；
按相同 score/ID 顺序，局部 Top512 的并集包含全局 Top512，但须传 score 和 ID，
而当前接口只返回 ID/count。它还需要历史分片的页表、追加与 checkpoint 设计，暂不纳入首版。
vLLM 返回展开后的 token IDs；NInfer 继续交换压缩 block IDs，不引入这项输出膨胀。

## 5. MoE

两机复制 Router，产生全局 top-10 和归一化系数。本地只计算拥有的 global expert IDs，
不按本地子集重新归一化。Shared FFN 的 intermediate 切成 320+320，独立 scalar gate
复制，本地 shared contribution 与 routed contribution 汇合后统一归约。

每层本地 bank 包含 256 个完整 experts，global-ID 到 local bank 的映射独立于存储顺序。
首版 Main 使用已有逐层优先级，排序位置 j 的专家归属 `j%2`；MTP 使用 global ID
奇偶分配。两者均保持相同数量，加载时生成固定映射。优先级只有排序，不能替代精确频次
或共同激活数据；优化目标是较慢 rank 的实际完成时间。小 M 看触及权重与负载偏斜，
大 M 同时看 per-expert token 数和 grouped GEMM 效率。

首版 token 输入复制，每机计算本地 expert 子集后合并输出，不需要 token AllToAll。
vLLM 的 `FusedMoEParallelConfig` 也区分这两件事：TP2+EP 将 routed TP 设为 1、EP 设为 2；
DP/PCP 或 sequence-parallel tokens 才进一步触发其 AllToAll 路径。Shared MLP 的 TP
仍独立于 routed expert 的 EP。

NInfer 已有的能力与缺口如下：

| 层级 | 已有基础 | 本次并行接入需要的工作 |
|---|---|---|
| Route | 全局 512、top-10、独立 FP32 coefficients | 两机复制相同 control 计算，不按本地专家重新归一化 |
| Device bank | indexed offsets、独立 divisor、subset mask | 512 个逻辑地址映射到 256 个本地物理 slots；远端项不被读取 |
| 本地计算 | `routed_swiglu` 支持 subset，也支持叠加 shared | 保留全局 route profile；本地 active assignments 在读权重前筛选 |
| Shared | 公共 profile 已独立 | validation、kernel、workspace 仍固定 640，需完整支持 320 |
| 加载 | 原产物的 affine bank 描述和整 parent 绑定 | 按专家搬运编码 planes，直接生成本地 slots 和 global-ID offsets |
| 完整融合入口 | 假定全部 routed experts 可读 | 单机入口保留；EP 的融合路径必须显式包含 subset 与独立 shared 几何 |

因此首版可以保留 `bank.count=512` 的逻辑域，物理容量只有 256 个专家。
沿用现有 expert slot 的 codes/scales 组织即可让 indexed offsets 指向本地存储；
divisor 表体积很小，可继续按 global ID 编址。任意映射不要求重写 expert GEMM，
但不能继续通过 `bind_banks` 绑定完整 GPU parent 来实现“分片”。

完整融合路径中的 route/group counts 也要按本地 subset 构建。只替换 bank 指针或在
最终输出处清零，会让中间 grouped jobs 仍访问远端权重。Shared=320 的 gate/up、down、
activation scratch 与末端累加必须一起改，不能只放宽入口检查。
本地 shared 与 routed 尽量在现有累加内部合并，再输出一次 BF16 局部贡献并参与 AR；
不能先各自额外舍入为 BF16 后，把它当作已有融合语义。

全局 Router 是需要一致的离散决策区域，与 Indexer 采用相同原则。合并投影若保留为
`[1153,2560]`，两机的 router/score 段必须采用共同计算路径，shared 段才是各自的权重。

EP 不保证每步各算五个专家。单 query 的十个 experts 可能偏向一侧；同数量、按全局
优先级交错放置只是合理初值，不能保证每请求负载均衡。大 M 的每个 expert 仍接收
原本路由到它的所有 tokens，所以单 expert GEMM 的 token 数不会因 EP2 自动减半。
这有利于保留 grouped GEMM 效率，但实际加速由较忙 rank 决定。

## 6. Embedding、head 与输出决策

Embedding 完整复制，每机约 1.184 GiB；lookup 仅读取所需行。LM head 沿词表行切分，
每机约 0.592 GiB，Main/MTP 继续共享同一物理 head。

复制 embedding 的代价明确：每机比 TP2 多放约 0.592 GiB，换取逐 token lookup 不增加
通信。vLLM 的 vocab-parallel embedding 使用零填充局部结果并归约；我们的容量条件下，
embedding 复制仍是合理选择，head 则通过词表切分减少每步的大矩阵读取。

各 rank 先对本地 logits 应用 mask 和 penalty，再求本地 Top-K。当前采样最多保留
全局 Top-20，按相同分数/ID 顺序，各分片的 Top-20 必然覆盖全局 Top-20。
合并候选后才执行 min-p/top-p、归一化和 RNG。这精确保留当前采样算法；不能先在
本地算 top-p 或 softmax 再合并。候选使用 FP32 adjusted score 和 I32 global ID，
每 rank、每列最多 160 B。

Greedy 每机只需一个候选。可将 FP32 score 与逆序 global ID 编成唯一全序 U64 key，
做一次 MAX AllReduce，使两机直接取得相同 argmax；mask 无候选使用独立的无效 key，
signed-zero/tie 规则沿用现有 selector。这样 MTP 每步不需要 gather 后再 broadcast ID。
Stochastic 路径汇集候选，由 output rank 在 GPU 上决策并分发小型结果。
允许单个词表分片无合法候选，dead-end 按全局词表判定。

当前 `sample`/acceptance 接口消费完整 logits，并含计数效果，不能在两机各调用一次
来实现上述协议。需要拆出本地候选提取与全局决策的数学接口，保留单机融合路径；
现有 kernel 内已有 partial TopK 与 merge，可作为计算基础。权重物理词表是 248320 行，
实际 `token_domain` 来自模型资源，分片末尾的非公开 IDs 必须排除。

MTP 验证列 j 的 penalty 叠加此前 j 个 draft IDs，因为只有接受该前缀后才消费该列。
各列候选可并行产生，后续 acceptance 顺序执行。Global ID、tie-breaking、seed、
逻辑位置和 SamplePurpose 保持现有语义。Canonical token counts 可随词表分片；
候选构造只读它及 round-local overlay，最终采用命令才增加本分片命中的输出 IDs。
许可输出与 Frontend 实际采用前缀仍分开，不能把未发布的 correction/bonus 计入持久状态。

CausalScoring 先归约全局 max，再归约移位 exp sum，并取得 target owner 的 logit，
得到完整词表 log-probability；不使用 Top-20 近似。它仍只需标量统计而非完整 logits。
vLLM 已有分布式 argmax/TopK 方法，但常规 logits 路径仍可 gather 全词表，其快捷方法
不自动提供 NInfer 的 penalty、约束和 speculative acceptance 合同。

已有 Spark Graph/cold 迁入测量中，完整 BF16 head 在 M=1/8 的 GPU 中位数约
4.7–4.8 ms。词表切分缩小每机
head 计算与读取范围；候选通信后的真实收益需要双机执行测量。

## 7. PLE

确定按现有表 shard 分配所有权：两个 rank 各保存 64 个完整 NF4 shards，约
13.56 GiB/rank；rank 0 保存 shards 0–63，rank 1 保存 64–127，每个逻辑行只有一个
owner。原有 128 个 shard 的 metadata/哈希参数
很小，可复制。表的存储所有权在加载期确定，本双机计划不保留完整复制表的执行 variant。

vLLM 的 `Qwen4ExpPLEEmbedding` 继承 vocab-parallel embedding，通过 ETP group
分片表行并合并 lookup 输出。NInfer 可以采用同样的放置原则，同时保留自己的 NF4
编码、4 KiB row-page、per-shard divisor 和精确 BF16 LUT，不转换为其 BF16/FP8 表。
完整原始 shard 的分配还避免了拆页、修改 divisor 和额外重排。

每轮的 GPU 流程为：

```text
已知 token IDs + 已执行的 raw-ID tail
  → 按现有 INT64 wrap/XOR/有符号模语义生成 16 个全局行号
  → 本地 owner 查询并精确解码，远端行写零
  → 合并成完整 [2560,M] BF16 lookup
  → 复制的 PLE projection / gate / convolution
```

Lookup 出口应保留 BF16 words，包括 signed zero。不能用普通浮点加零来声称逐 word
不变；每行 160 维为偶数，可以将 BF16 两两打包为 32-bit word，每个 word 仅一个
owner 非零，用整数 SUM 合并后按 BF16 解释，或使用等价的按 owner 收集。
该合并的逻辑语义是物化独占行，不是对多个数值贡献做近似求和。

通信载荷为 `5120*M` B：M=1 为 5 KiB，M=4096 为 20 MiB，每轮 Main 只做一次。
Lookup 不依赖 residual，可在 token IDs 确定后开始，利用 embedding 和整个第 0 层
直到第 1 层 PLE 消费之间的时间。它可以排在同一通信序列的 layer-0 归约之前，
与 GPU 计算重叠；若尚未完成，仍可能延迟后续归约，不能预先把通信耗时算成零。
MTP 本体无 PLE；Main verify 的 IDs 到齐后即可启动整窗口 lookup，不再为 PLE
把 draft IDs 搬到 Host。普通输出、约束与 Frontend 的 Host 交接仍按原有职责执行。

GPU hash/lookup 是新增能力，现有代码只有 CPU hash、SSD 读取和 Host LUT 解码。
现有 `make_nf4_bf16_lut` 可在加载期生成每个 shard 的小型 LUT 并上传；kernel
消费原始 row 编码，不在启动时展开整张 BF16 表。PLE projection/gate/convolution
与九列 history 复制，保持既有时序、归一化和 BF16 边界。

将 Main/MTP experts 和 PLE 表分别分半，其余权重暂按复制计算，约 54.29 GiB/rank；
其他 TP 切片还会减少容量。
这是由现有产物各组件字节数推导的权重上界，不含 State/KV、workspace、Graph、
staging、Frontend 与系统余量。GPU 与 CPU 分配共用机器物理内存，加载期也必须
沿用有界 staging，不能再保留一份完整可写 Host 权重。

## 8. MTP 与 Vision

MTP 主体采用 QSA TP2 / experts EP2，Indexer 和 HC 复制，共用 Main 的分片 head。
每步 draft 的主体有 attention 与 MoE 两个主要归约边界。单 owner 主体加分布式 head
是对照候选，需比较节省的通信和增加的单机计算。该候选需要 owner 持有完整 MTP
experts、完整两-head MTP KV，再向 head ranks 分发 sample hidden；不是只关闭两次 AR。
基础路径继续与 Main 的本地 QSA/EP 接口共用。

Stem 的两个 `[2560,2560]` FC 初始复制；fc_embedding 处理 M 列，fc_hidden 处理 4M 列。
候选切法为两者均按输出 feature 切分，本地组合 residual 后只 AllGather 一次，
再做完整 norm/HC。它与复制路径的比较应计入原有 BF16 出口和 residual 舍入边界。

vLLM 两个 FC 都使用 ColumnParallelLinear 且 `gather_output=True`，各自汇集输出。
NInfer 若切分，应让两支采用相同的输出 feature 范围，保留各自 BF16 FC 输出，
本地 broadcast-add 并舍入成 residual 后，只汇集一次 `[2560,4,M]`，再做完整 norm。
这需要支持局部 feature 的相加出口，不能直接把现有固定 H 的 HC norm 用于半个 H。
两份 FC 共约 25 MiB，小 M 下权重读量减半按持续读参考带宽折算约 51 μs；这是和
一次汇集比较的量级依据，不是实测收益或上限。实际应同时测 FC、相加、pack、通信与 norm。

Draft steps 串行依赖前一步的 ID 和 pre-final residual。全局 argmax 后在设备侧
直接进入下一步，不增加每步的协调侧往返。vLLM 的后续 draft 可复用 step-0 Indexer
selection 属于另一个计算选择，本方案保持逐步重新选择的现有模型语义。
MTP history 建立只执行 stem、attention HC、QSA prepare/tail；不能误接成完整 draft forward。

Vision 权重复制，约 0.836 GiB/rank。首版以现有 `VisionItemView/Control` 为分工单位，
各机执行完整 encoder，最后将 merger 特征分发给 Text ranks。vLLM 的 Qwen4Exp
也可通过 `mm_encoder_tp_mode=data` 使用数据分工的 Qwen Vision tower。
NInfer 需要把现有逐 item 编码的调度扩展到 owner 映射，而不是修改 encoder 数学。

输入 patches 只发送给编码 owner，最终特征保留原有 item、MRoPE 与 scatter 列映射。
Main 和 MTP history 消费结束前保留特征引用。视频或大图不能按任意 patches 切开；
进一步拆分需以实际 attention segment 为依据。单个不可分 item 仍只有一机编码，
这个首版选择不会改善它的编码耗时。两机并行编码的 workspace 峰值也要单独规划。

## 9. 状态与完整执行流程

| 状态 | 放置 |
|---|---|
| GDN recurrent / convolution | 随本地 heads 分片，recurrent 约 54 MiB/rank/Main image |
| Main/MTP 主 KV | 每 rank 一个 KV head，共同逻辑页组 |
| Index cache / raw tail / 块坐标 | 复制 |
| PLE history | 复制，180 KiB/image |
| Sample hidden / conditioning | 保存续算所需的完整行 |
| 前沿、输入身份、接受长度 | 协调侧决定，各 rank 共同采用 |

Head 分片只改变物理 schema，不改变 prefix 匹配算法及逻辑 token 覆盖。一个逻辑页组
在两机分别持有主 KV head shard 和完整 index cache；相同 group ID 不要求相同地址。
容量按各 owner 的可用资源和每组字节数共同规划，不能只用 rank 0 的页数代表全组。

BF16 主 KV 在 262144 context 下，Main 每 rank 约 3 GiB/请求，另加约 192 MiB index
cache；MTP 主 KV 再加约 256 MiB 和 16 MiB index。实际容量使用所选 codec 的 schema。
GDN 的 54 MiB 是一个 image 的本地 recurrent 字节数，source/destination、checkpoint、
verify records 不能只计一次。Indexer tail 和 PLE history 即使复制，也要随同一
source/destination 与接受长度一起保存、恢复和提交。

三个流程需在 D0 闭合：

1. Ordinary：共同 token/position → 本地 lookup/HC → TP mixer/EP MoE 与归约 →
   分片 head/候选汇合 → 输出决定 → 按共同前沿采用本地状态。
2. MTP：共享 conditioning → 分布式 draft steps → Main verify records → 按列候选与
   acceptance → Frontend 采用前缀 → 本地 GDN fold、QSA/PLE tail 与 MTP 历史对齐。
3. Restore：一次 prefix 匹配取得同一 checkpoint manifest → 各 owner 恢复自身 State/KV
   → 必要 Replay → 续算；复制字段与分片字段必须属于同一逻辑执行状态。

Main canonical frontier 为 F，MTP 为 `max(F-1,0)`；PLE raw-ID tail 对应已执行输入，
不是已经预测但尚未采用的输出。采用 a 个许可输出后，各 rank 的 GDN fold、QSA/PLE
extract、sample/conditioning 选择和 MTP history 对齐共同推进这些前沿。GPU lookup
只借用 tail 的设备输入/镜像，不能提前把整个 draft 窗口变成 canonical tail。

前缀可用性由一个全局 manifest 关联全部必要 owner 的有效分片。各机不独立命中、
独立驱逐某个仍被全局引用的 shard；恢复沿用已有引用、COW 和 Replay 语义。
模型数学决定 schema 与前沿，ContextDirectory 管逻辑引用，RankProgram 管本地 backing。
新分片方式需要自己的布局身份；本轮 checkpoint 在同一固定 ParallelPlan 内复用，
不引入把已保存 state 在线转换到另一套 TP/EP 放置的能力。

四机时 QSA 的两个 KV heads 需要在对应 ranks 中复制，Q heads 可继续切成每机六个。
ParallelPlan 显式描述这种复制，不能假定所有 State/KV 字段都按 rank 数等分。

## 10. 计算组织与 Graph 组装

### 10.1 原生计算边界

模型计算使用一套原生 enqueue 流程，eager 直接执行，Graph 捕获同一流程。
基本设备段为 Main、draft chain、输出决策、状态采用；段内继续调用现有数学 Ops。
层级组织和 Graph launch 粒度分开：模型保留清楚的 layer/Op 边界，Spark replay
不由 Host 逐层 launch。

Main 统一消费准备好的 PLE lookup Tensor `E[2560,T,B]`。表驻留磁盘、Host 内存或
Device 是 producer 的选择，不进入 Main 的数学接口，也不决定 Main 是否完整抓图。
Producer 提供稳定 E 视图、本轮就绪依赖和消费寿命；本双机计划的 producer 是分片 GPU
lookup/合并，可与 Main 一起捕获。现有 Host producer 也可向同一完整 Main 图供数，
无须先实现 GPU lookup。包含 producer 的整轮组合与 Main 自身的抓图范围分别处理。

vLLM 将 `CudagraphDispatcher` 的模式/shape 选择与 `CUDAGraphWrapper` 的捕获重放
分开。这里沿用这一职责划分：Session/原生计划选择执行组合，RankProgram 绑定本地
views 与 Graph；Op 继续负责数学与私有 shape policy，不决定模型级切图。

每个 Main 层的主链为：

```text
完整 residual / normalized
  → HC mix
  → 本地 GDN 或 QSA（含本地 State/KV 写入）
  → output projection 的 BF16 局部贡献
  → AllReduce
  → HC combine / norm / mix
  → 全局 router + 本地 routed subset + shared TP 局部贡献
  → AllReduce
  → HC combine / 下一入口 norm
```

第 0 层末端先物化 BF16 residual，再在第 1 层入口执行 PLE；PLE 自身产生下一组
normalized。最后一层输出 pre-final residual 供 MTP conditioning 使用，final HC
产生 sample hidden，随后执行本地词表 head。上述归约都位于实际数值依赖处。

### 10.2 PLE 旁路与 stream 组织

RankProgram 持有有序主计算流、PLE 准备流、通信流，以及输入/结果传输所需的 copy
流。调用者传递稳定的 execution views，各 Op 不创建 stream。
本计划的 TP、EP、PLE 和 head 通信参与者相同，可以共用一个物理 communicator；
模型角色仍独立表达。通信流承载固定的共同顺序，不额外为 PLE 创建 communicator。

```text
                         ┌─ hash / 本地 NF4 lookup ─ PLE words 归约 ─ projection ─┐
输入 IDs / raw-ID tail ──┤                                                       ├─ PLE gate/conv/norm
                         └─ embedding / HC ─ 第 0 层（两次归约）─ BF16 residual ───┘
                                                                               ↓
                                                                       第 1 层至末层
```

PLE projection 只依赖 lookup 输出，也可以提前；gate 才开始依赖 Main residual。
Storage producer 的出口仍是 E。Main 图内可以提前执行 projection，形成
`PleProjectionView` 所需的稳定 key/value parent，到插入层再执行既有
`ple_gate → ple_conv_add_norm` 和对应 record/state 动作；这是模型内部排程，
不改变 E 接口或中间 BF16 边界。

通信次序为 PLE words 归约、第 0 层 mixer 归约、第 0 层 MoE 归约、后续各层归约。
PLE 的结果合并一结束，主链就能继续使用通信流，不等待旁路 projection。
主链到 PLE 消费点才等待准备结果。跨流使用 event 表达数据依赖，捕获时所有旁路
最终回接主流；这是 CUDA 支持的同图跨流捕获方式。
见 [CUDA 跨流捕获](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cross-stream-dependencies-and-events)。

提前 projection 提供重叠机会，但两条计算分支共用 SM 和内存带宽。实际执行计划可以
令 projection 并行或在 lookup 合并后等主链到达再执行，依据完整阶段耗时选择；两者
都是图内依赖，不增加 Host 交接。PLE 合并位于第一层归约之前，未完成时仍会阻塞该
归约，不能把旁路的全部耗时视为已隐藏。

旁路存储与主计算 scratch 分开规划。Lookup 为 `5120*M` B，projection 为
`25600*M` B；M=4096 分别为 20 MiB、100 MiB，还需计入各自 scratch。
Projection 从提前计算开始存活到 PLE 消费结束，不能沿用当前单流 MemoryArena 的
作用域复用规则，与正在执行的第 0 层互相覆盖。其余不重叠阶段继续共享主 scratch。

### 10.3 Graph 的外部边界

将设备计算、输出许可和最终采用分别组织。GPU 结果仍属于当前 frame 的候选状态，
Frontend 确定最终输出前缀后，Session 才发出 commit/discard。

| 路径 | 计算与决策 Graph | 后续采用 |
|---|---|---|
| Ordinary，决策输入已齐 | Main（含 PLE 旁路）+ 本地 head + 分布式决策，合为一图 | 单独的采用动作 |
| Ordinary，mask 尚未齐 | Compute 图执行 Main + 本地 head；mask 到达后执行 Decide 图 | 单独的采用动作 |
| MTP，无依赖 draft 的 Host 决策输入 | draft chain + Main verify + 本地 head + acceptance，合为一图 | Finish 图/设备段 |
| MTP，约束 mask 依赖 drafts | Compute 图执行 draft chain + verify + 本地 head；Decide 图等待 mask | Finish 图/设备段 |
| Prefill / Replay | 按实际 T 执行相同原生流程；先保持 eager，固定完整 chunk 可作为独立 capture profile | 包含所需 MTP history 建立 |
| Vision | 每个 owner 执行完整独立 item，特征分发完成后供 Text 使用 | 特征保留到 Main/MTP 的最后消费者 |

决策输入已齐时不必强行沿用分段 launch。需要等待外部输入时，Compute 的结束点放在
本地 head 之后，让大矩阵计算先完成；Decide 只消费保留的本地 logits、mask、counts
与配置。两个组合共享原生 enqueue 函数，Graph 数量与 frame 绑定计入准备预算。

Prefill 不能只为复用最大 T 图而随意补列：现有 sequence GDN 入口会推进其真实 T
列，尾部 padding 也会影响递推状态。尾块使用实际几何的 eager 或已准备 profile。

### 10.4 MTP 整轮与约束重叠

启动固定 draft 上限 K；每轮按共同预算选择实际深度 S（1..K），使用对应的有限
chain variant，在图内展开 S 次原生 draft step。每步包含 MTP stem、QSA TP、EP MoE、
本地 head 与全局 argmax，下一步直接消费设备 ID 和 pre-final conditioning。
各行有效 draft 数通过设备 extents 控制，保留第一步 bridge 的语义；没有实际工作的
后续行不写有效状态，各 rank 履行所选 variant 的共同通信合同。Main verify 仍使用
既定 K+1 物理列和逐行有效长度，不因某行剩余预算较小而擅自更改通信形状。
Graph 不在每步 draft 后返回协调侧取得下一枚 ID，也不为复用最大深度图额外计算
整批都不需要的后续 drafts。

Draft chain 完成后，一次性组装 `[anchor,drafts...]`，Main verify 的 PLE 对整窗口做
一次 lookup 合并，不沿用单机 Host producer 的增量 D2H/准备流程。

约束路径可以在一个 Compute 图内保留以下分叉：

```text
draft chain ─┬─ prepare verify IDs ─ Main verify + local head ─ 保留 local logits
             └─ output rank 将 drafts 异步 D2H ─ DraftsReady
                                                   ↓
                                      协调侧 matcher 生成分片 masks
                                                   ↓
                              Compute 结果依赖 + masks 已就绪 → Decide 图
```

Output rank 的 copy 分支使用 frame 自有 pinned staging 和可供 Host 观察的 external
event；执行线程观察本轮事件后发布 `DecisionInputNeeded`，target 计算继续。
外部 event 保留到其 Graph 和传输消费者结束，capture 期间不查询事件。
该机制可用 [CUDA external event nodes](https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/constraints.html#external-events-and-streams)
实现，不需要在图中调用 Host matcher 或等待控制消息的 callback。
Mask 输入和共同命令顺序确定后即可入队 Decide，GPU 用 stream/event 继承 Compute
完成依赖；Host 不必先等待整个 Compute 完成。

Mask 在各词表 shard 上应用。普通 sampling 使用候选汇集、output rank 决策和结果
分发；greedy 的全局 argmax 已在所有 ranks 可见，整数前缀比较可在各机得到相同
acceptance，由 output rank 对外发布。RNG 和公开输出决策的所有权保持不变。

Finish 消费 Frontend 最终接受的 a，执行本地 GDN fold、QSA/PLE extract、末端 hidden
选择、token counts 采用及 MTP history 对齐。其物理窗口固定，a 是设备输入。
该阶段需要将当前 Host 循环中的逐行复制/metadata 准备转为可捕获的设备准备动作；
MTP history 继续只执行 stem、attention HC、QSA prepare/tail，不追加一次完整 draft。
Canonical state 的逻辑采用由 Session 管理，Finish 的设备完成依赖交给下一轮。

### 10.5 完整 replay 与局部 Graph 更新

单机 `TextGraph` 的 resident Main 已使用完整外层 replay，QSA envelope 与 PLE 上传槽
通过 child 更新。分布式实现沿用该组装方式，使上表每个设备段只有一次外层 replay，
capture 直接包含计算和 NCCL。
PLE 的模型计算位于 Main 图内部；上游准备位置不是逐层切图的理由。两个 E 槽的地址
切换通过输入绑定或局部 consumer 更新处理，不为每个槽复制完整模型 executable。

保留现有 QSA 随上下文更新 launch 的能力，采用“完整外层图 + 可更新的本地计算
子图”：

1. 启动时按实际绑定的 frame 地址，捕获 QSA 本地计算区域的各 envelope definitions。
   基础复制 Indexer 的路径中，该区域没有通信。
2. 捕获外层原生设备段时，将相应 definition 插入为 child graph，保留节点句柄；
   NCCL 在外层流程中按共同顺序直接捕获。
3. 跨上下文档位时，只更新这些 child 的参数/计算 definition，保留通信节点和 backing。
   不按内核名称查找节点，也不把完整层和通信重新捕获一次。
4. Child 的拓扑、节点加入顺序和可更新参数满足 CUDA 更新要求；若计算/通信拓扑
   改变，使用另一条已准备的共同 variant。大 M 的 query 分工会把本地计算区域放在
   selection 通信两侧，不能把通信藏进可任意替换的 child。

Core 的 Graph 封装已提供捕获中插入 child、保留绑定句柄和更新 child 的能力。
实现结合 `cudaStreamGetCaptureInfo` / `cudaStreamUpdateCaptureDependencies` 与显式
Graph API 组装，使用 `cudaGraphExecChildGraphNodeSetParams` 更新。
相关能力及限制见 [CUDA 捕获与显式节点组合](https://developer.nvidia.com/blog/constructing-cuda-graphs-with-dynamic-parameters/)
和 [CUDA child graph 更新](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__GRAPH.html)。
这是对现有 Graph 基础的扩展，不要求 Op 暴露 kernel 参数，也不建立通用模型 DAG。

外层 executable 按 ParallelPlan、设备段/通信协议、T/B、状态更新模式和 frame 绑定
建立。上下文档位是已安装的本地计算 profile，不把整套 Main 图按所有档位复制。
Workspace 按所有允许 profile 的峰值预留；child definitions、外层 executable、NCCL
保留量均计入实际 Graph 准备开销。多 frame 不自动共享可写 logits/records；有序计算
可以共享 scratch，跨阶段等待的结果保留到实际消费者结束。
Frame 视图与并行 scratch 分区先于 capture 确定；普通 enqueue 与 child 插入消费
同一份绑定和容量，不能因为插入 child 跳过了 Host 计算调用，就漏掉其存储预留。

各 rank 的本地 kernel、地址和输出侧 copy 分支可以不同，通信协议必须共同选择。
NCCL 要求 capture/replay 的参与者一致，各 rank 使用对应共同 capture 所派生的图；
它的 group 调用若混合多个 stream，会增加这些 stream 的依赖，因此此处通信保持在
同一有序流中。见 [NCCL Graph 合同](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html)
及 [NCCL stream 语义](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html)。

### 10.6 成本与可扩展边界

基础 Main 的 48 层各有 mixer output 和 MoE output 两个归约边界，共 96 次。
PLE 表分片另有一次合并；每个 draft step 有两次 block 归约和一次全局 argmax。
按本文基础布局，greedy ordinary 的设备通信为 98 次，K-step greedy MTP 为
`98+3K` 次；一般采样的候选汇集和结果分发比 greedy 输出多一次。这里不含 Host
控制传输、Vision 特征分发或后续新增的布局优化。完整图减少 Host launch 和交接，
这些数学依赖中的通信仍然存在。
这 96 次 block 归约各自的 BF16 逻辑 payload 为 `2*2560*M` B：

| M | 单次 H 归约 payload | 96 次合计 |
|---:|---:|---:|
| 1 | 5 KiB | 480 KiB |
| 8 | 40 KiB | 3.75 MiB |
| 48 | 240 KiB | 22.5 MiB |
| 4096 | 20 MiB | 1.875 GiB |

以上不含 PLE、head、MTP，也不是链路实际 wire bytes。各次归约之间存在 HC、gate、
norm 等非线性依赖，不能把这 96 次按总字节量合成一次传输。小 M 需控制逐次启动和
同步成本；大 M 要同时看有效链路带宽、计算重叠及较慢 rank 的完成时间。
BF16 局部贡献和通信是既定合同，不为并行额外强制 FP32；新的分片组合仍需按其
数学与数值合同检查，不要求和单卡另一种归约顺序逐位一致。

层间耗时的判断单位是：本地前置计算 → 较慢 rank 的局部结果 → 暴露的通信 →
后续 HC/投影，而非把每个 Linear 的读量减半后直接相加。共享 Router/Indexer/HC
仍占时间，EP 的每步负载可能不均，head 的减半也只是其中一项。

没有本地 active expert 的 rank 仍输出正确的零/共享贡献并参与约定通信。Indexer
全选与混合上下文继续在 kernel 内处理。可选择不同的已准备 variant，但不能在某一
rank 依据本地观察单独增减 collective 或改变消息尺寸。图的分派和更新不引入逐层
控制消息或每轮全组 Host barrier。

Eager 与 Graph 共用这些边界。真正通算融合前，先保留清晰的本地 Op 输出和 collective
合同；之后可以将归约与 HC combine/norm 接在一起，仍保留 BF16 residual 的原数值顺序。
执行框架无需知道这些 Op 的私有 tile、调优表或专家 GEMM 算法。

权重放置与执行时分工还需要分开。Experts/PLE 所有权、head 范围、MTP 单 owner 或
TP/EP 主体在启动时选定；HC token 分区与 Indexer query 分工可以复用相同权重和
canonical state，只改变 frame 的临时布局。若同时保留 HC lowrank TP、MTP FC 的
复制/TP 两条路径，需要在加载期明确准备各自可消费的矩阵视图/编码并计入容量，
不能把切换 variant 变成执行期搬运、重排权重。

## 11. 实施依赖与后续优化

| 项目 | 可行性结论 | 下一步依赖 |
|---|---|---|
| GDN/QSA head TP、expert EP、vocab head TP | 数学和主要分片方式确定 | 本地 Op shape、分片加载和完整 state schema |
| EP 复制输入、统一结果归约 | 现有 subset/indexed bank 可承接，不需要先引入 AllToAll | 直接加载本地 slots，完成 shared=320 与融合筛选 |
| Embedding、HC、Indexer 基础复制 | 可形成完整基础路径；复制的离散决策有共同执行要求 | 绑定共同 decision profile 和本地 cache |
| PLE 表行分片、计算复制 | 已确定；编码保持，lookup 合并提前 | GPU hash/NF4 lookup、精确 word 交接和旁路存储 |
| 词表候选与采样 | 当前 Top-20 合同下可精确分布式执行 | 候选/决策接口、mask/count 分片及采用协议 |
| MTP / Vision | 主体与独立 item 分工可行，history 与 Frontend 语义保留 | 分片 head 的设备决策、history 路径、特征分发 |
| 大 M HC token 分区 | 后续优化，首版关闭 | PLE 前后布局、末端特征、RS/AG 的完整耗时 |
| 大 M Indexer query 分工 | 后续优化，首版关闭 | query 工作域、负载分配、selection 交换耗时 |
| HC lowrank TP、MTP FC TP、单 owner MTP | 后续优化，首版关闭 | 完整候选路径对照，而非只测局部 GEMM |
| 完整设备段 Graph、本地计算 child 更新 | 单机 resident Main、draft chain 和 Core child 更新已实现；分布式尚未实现 | 共同 capture、通信旁路容量与设备 Finish |

D0 的基础角色所有权、编码切片、本地几何、状态和输出决策方式按本文首版路径执行。
后续候选不进入首版分派，也不预设未经测量的 M 阈值。Shared=320、分片采样、GPU
lookup 与 Graph 组合基础是明确的新增能力；它们不能靠现有 TP shape 清单自动补齐。
单机 resident Main 已直接复用现有 PLE producer 完整抓图，与双机实现解耦。

## 12. 源码依据

本轮对照本地 vLLM 工作副本的具体实现，而非将其全部默认策略视为 NInfer 的选择：

单机耗时复用 `.agent-tmp/spark-op-migration/units/` 下 `hc-mix` 和
`bf16-n248320-k2560` 的既有迁入记录；读带宽参考来自
[roofline](../bench/common/roofline.h)。这些历史数据仅用于判断候选成本量级。

| vLLM 路径，相对 `~/vllm/vllm/` | 对应事实 |
|---|---|
| `models/qwen4_exp/nvidia/model.py`、`hyperconnection.py` | HC 复制、sequence-parallel MoE 限制、PLE 位置与 Vision 模式 |
| `models/qwen4_exp/nvidia/qsa.py`、`indexer_qsa.py` | 主 heads TP、Indexer 复制、展开 token selection 与 MTP 复用选项 |
| `model_executor/models/qwen3_next.py`、`model_executor/layers/fused_moe/config.py` | Routed EP、shared TP 与 AllToAll 的适用条件 |
| `models/qwen4_exp/nvidia/ngram_embedding.py`、`models/qwen4_exp/common/ple.py` | ETP 表行分片与 checkpoint 行区间 |
| `models/qwen4_exp/nvidia/mtp.py` | 双 FC 的 column TP/gather、conditioning 和 draft 层 |
| `model_executor/layers/logits_processor.py`、`model_executor/layers/vocab_parallel_embedding.py` | 分布式 argmax/TopK、普通 logits gather、词表 embedding |
| `v1/cudagraph_dispatcher.py`、`compilation/cuda_graph.py` | 执行模式/profile 分派与捕获重放的职责划分 |

NInfer 的落点包括：

- [GDN/QSA/MTP 执行](../src/models/qwen4_exp/execution/)、
  [StateImage](../src/models/qwen4_exp/state/state_image.cpp)、
  [状态采用](../src/models/qwen4_exp/program/commit.cpp)。
- [MoE 公共合同](../include/ninfer/ops/moe.h)、
  [bank 视图](../src/core/weight_bank.h)、[bank 加载](../src/models/common/moe/weights.cpp)、
  [MoE 参数支持](../src/ops/moe/operands.cpp)、
  [block-scaled shared 计算](../src/ops/moe/kernels/cuda/block_scaled/shared.cu)。
- [HC 合同](../include/ninfer/ops/gated_residual.h)、[QSA 合同](../include/ninfer/ops/qsa.h)、
  [Indexer](../src/ops/softmax_attention/qsa/indexer/)。
- [Sampling](../include/ninfer/ops/sampling.h)、
  [Speculative acceptance](../include/ninfer/ops/speculative_accept.h)、
  [现有候选/归并实现](../src/ops/sampling/kernels/distribution.cuh)。
- [PLE 参数](../src/models/qwen4_exp/ple/parameters.cpp)、
  [哈希](../src/models/qwen4_exp/ple/hash.cpp)、[精确 NF4 LUT](../src/core/nf4.h)、
  [权重物化](../src/artifact/materializer.cpp)、
  [Vision item](../src/models/common/qwen/vision/control.h)。
