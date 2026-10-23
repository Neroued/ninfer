# GLM5Next MTP 执行与接口设计

本文确定原生 MTP 的输入配对、计算组合、历史提交、并行交接和新增 Op 合同。
它是待实现设计；模型数学与权重表示分别由[模型参考](../../maintainer/glm5next-model.md#61-原生-mtp)、
[数值参考](../../maintainer/glm5next-quantization.md)维护。MLA 的 prepare、selection、
cache、tail 与容量接口沿用 [Kpool Sparse MLA](kpool-sparse-mla.md)，不在这里重新定义。

## 1. 范围与模块边界

当前 checkpoint 有一个 MTP block，位于源 layer 45，hidden=4096，词表=154880。
该 block 使用普通 residual、MLA/Indexer 和 MoE，没有 Main 的四流 mHC 或 KDA。

MTP 是模型执行组合。新增的成对 norm/拼接、residual/norm 是 Normalization Ops；
stem 和 head 是 Linear；历史准备和 attention 属于 Kpool family；MoE 使用已有接口。
模型负责调用顺序、token 配对和状态采用，不建立一个包含整层的 MTP Op。

| 所有者 | 本部分的职责 |
|---|---|
| Converter／Definition／Shard | 源参数映射、可选 draft head、加载使用关系、权重 backing |
| 模型 Parameters | 实际几何、epsilon、Op 权重 views 与数值政策 |
| 模型 Session | Main/MTP ordinal、canonical frontier、owner、采用列与 checkpoint 关系 |
| ContextDirectory | canonical/provisional history、State 身份、成组预留、COW、借用及回收 |
| RankProgram | Context backing、frame、临时 tail/hidden、执行和通信顺序、Graph |
| Ops | 显式输入上的数学、写域、数值表示、支持检查与私有分派 |

维度从模型配置和绑定取得。表中的具体数字用于当前 checkpoint 的实现与调优，
不变成 Engine 或 CLI 的算子支持检查。

## 2. 输入配对与完整计算

### 2.1 Main 和 MTP 的位置

用 `x_j` 表示 Main token，`h_j` 表示 Main 处理该 token 后的最终 BF16 hidden。
MTP 的逻辑 ordinal 为 p：

```text
MTP(p).embedding    = embedding(x[p+1])
MTP(p).conditioning = h[p]
MTP(p).proposal     = x[p+2]
```

递归 draft 的后续步骤以先前 MTP 输出替代真实 h，其他配对偏移不变。
MTP ordinal 0 的 embedding 分量置零，conditioning 仍正常归一化并参与计算。
Ordinal 0 本身仍产生 latent、pool raw data 和输出，不跳过这一历史列。

Main 第一个 token 没有可供配对的前一个 Main hidden，因此仅形成 h[0]；
得到 x[1] 后才具备 MTP ordinal 0 的输入。这与“ordinal 0 的 embedding 置零”
是两件事。Position、pool 相位和因果可见性均使用 MTP ordinal。

本版选择词表 embedding 作为 MTP 的 e，包括多模态占位 token；Main 的 h 已包含
视觉上下文。Main 仍正常用 Vision features 替换媒体 embedding。此选择遵循本地
vLLM GLM MTP 的输入约定；SGLang 的 NextN 可以使用 target 提供的媒体 embedding，
两者不作为同一输入合同处理。媒体场景的 proposal 接受率留待组合评测。

### 2.2 Stem、block 与 head

定义 direct-gamma RMSNorm：

```text
RMS(x, gamma, eps) = x * gamma / sqrt(mean(x*x) + eps)
```

本实例各 MTP RMSNorm 的 epsilon 为 1e-5。计算顺序为：

```text
e_hat = RMS(e, gamma_e, eps)              # ordinal 0 输出零分量
h_hat = RMS(conditioning, gamma_h, eps)
eh    = concat(e_hat, h_hat)              # feature 顺序 [e;h]
x     = W_eh * eh                        # [4096,8192]，无 bias

n0    = RMS(x, gamma_input, eps)
a     = MLA(n0)                          # 含 prepare/indexer/O，完整输出
r1    = BF16(x + a)
n1    = RMS(r1, gamma_post_attention, eps)
f     = MoE(n1)                          # routed + shared，完整输出
r2    = BF16(r1 + f)
hnext = RMS(r2, gamma_final, eps)

draft_logits = draft_head(hnext)
next_conditioning = hnext
```

公共 activation 均为 BF16；控制量、cache codec 和内部中间值遵循数值参考。
残差与 norm 的融合保持第 4 节定义的公共 residual 边界。hnext 同时作为
head 输入和下一步 conditioning，head 前不再执行一次 final norm。

MLA 使用完整的 64 heads，Q_b/U_K/U_V 保持 BF16，O 为 Q8 A16。
MTP routed experts 为 block128×128 FP8/BF16-scale，允许 A8；shared 为 Q8 A16。
Router、Top8、clamp、route 系数和 shared contribution 与模型 MoE 语义相同。

### 2.3 权重与复用

| 对象 | 物理含义与使用 |
|---|---|
| Token embedding | BF16 [4096,154880] 的查表视图；每 rank 一份，Main/MTP 共同借用 |
| Stem W_eh | BF16 [4096,8192]，K 轴先 e 后 h，无 bias |
| Norms | 五个独立 gamma：e、h、input、post-attention、final，均 BF16 [4096] |
| MLA／Indexer | 独立 MTP 参数；W_A 的 History 前缀和 W_B 的 query 前缀沿用 Kpool 设计 |
| MoE | 独立 MTP router、routed bank 和 shared；不是 Main 最后一层的别名 |
| Proposal head | 启用 optimized head 时使用额外 Q4-G64/FP16、A16、完整词表权重 |
| Main head | 始终 BF16；未启用 optimized head 时，MTP 借用同一份本地 Main head |

Artifact 用逻辑角色和 Use 表达共享与可选加载，不复制 embedding 或未优化 head。
启用 MTP 才加载其独有权重；启用 optimized draft head 才加载额外 Q4 对象。
纯 Linear 的矩阵清单沿用 [Linear/MoE 任务](op-checklist.md)。

## 3. 成对 RMSNorm 与拼接

新增入口放在 Normalization：

```cpp
void paired_rmsnorm_concat(
    const Tensor& embedding, const Tensor& hidden,
    const Tensor& embedding_weight, const Tensor& hidden_weight,
    const Tensor& embedding_enabled, float epsilon,
    Tensor& out, CudaExecutionView execution);
```

| 参数 | 合同 |
|---|---|
| embedding、hidden | 连续 BF16 [D,M]，相同形状；D>0，M≥0 |
| 两个 weight | 连续 BF16 [D]，直接 gamma |
| embedding_enabled | Device I32 [M]，每列取 0 或 1 |
| out | 连续 BF16 [2D,M]，每列前 D 项为 e_norm，后 D 项为 h_norm |
| epsilon | 正且有限的 FP32 值 |

两个归约独立进行。计算使用 FP32，输出各自转 BF16；enabled=0 的前半部分精确
写正零，后半部分照常计算。该列无需读取 embedding 数据，但 hidden 必须有效。
全部输入保持不变，out 与任何输入 disjoint；只读输入可共享 backing。

本入口支持正的 feature 维度，当前调优重点 D=4096；M=0 不提交设备工作。
Tensor 基址自然对齐、元素连续；所有 operands 与 execution 位于同一设备。
无 transient workspace、持久状态或隐藏分配，执行完成后全部 out 已定义。

模型依据 ordinal 生成 enabled，而不是让 Op 理解“首 token”或请求状态。
Prefill 将其加入既有输入 metadata 的上传；递归 draft 由已有 Device 输入准备更新。
不为生成这一个 mask 额外安排专用 launch。无效物理列使用安全输入，其历史写入仍由
Kpool 的 valid_columns 控制。

这是两个 norm 与拼接的一次数学操作，不先生成两份独立 norm Tensor 再 concat。
私有实现可以按 D/M 选择 warp、CTA 或其他归约组织；调优不改变输出顺序和 mask 语义。

## 4. Residual add 与 RMSNorm

MTP 和 Vision 共用以下 Normalization 入口：

```cpp
void residual_add_rmsnorm(
    const Tensor& residual, const Tensor& branch,
    const Tensor& weight, float epsilon,
    Tensor& residual_out, Tensor& normalized_out,
    CudaExecutionView execution);
```

对每个 feature 向量，先产生实际的 BF16 residual，再对它归一化：

```text
r[d] = RN_BF16(FP32(residual[d]) + FP32(branch[d]))
n[d] = RN_BF16(RMS(FP32(r), FP32(weight), epsilon)[d])
```

数值边界由[数值参考 §4.4](../../maintainer/glm5next-quantization.md#44-mtp-与-vision-的融合边界)
维护；norm 消费的 r 与 residual_out 完全相同。融合可以在寄存器中保持 r，
不要求额外读回，但不能对未舍入的 sum 求 norm。

输入和两份输出均连续 BF16 [D,M]，D>0、M≥0，weight 为 BF16 [D]，
epsilon 正且有限。输入保持不变，唯一允许的写入 alias 是
`residual_out` 与 `residual` 完全相同；其余写入与输入/输出 disjoint，
禁止部分重叠。只读 residual/branch 可以共享；采用原位形式时，branch 必须与
residual_out disjoint，避免把另一个只读输入隐式变成写入目标。

无 workspace 或持久状态。M=0 不提交设备工作，非空时完整写出两个结果；
同设备、自然对齐。初始调优覆盖 D=4096 和 Vision D=1024，其他合法 D 由通用路线处理。
MTP 的 final norm 同样使用该入口，即使 r2 在此后不再被模型消费。

## 5. 数据归属与 canonical frontier

| 数据 | 内容 | 存储与生命周期 |
|---|---|---|
| Canonical／provisional history | 已采用历史、draft 与补齐期间的 latent／pooled keys | Context KV records；目录管理共享、COW、预留、采用与退役 |
| Canonical tail | 已采用前沿的 raw K/gate | owner StateImage；请求与 checkpoint |
| Main conditioning | 当前最后一个实际 Main final hidden | 两 rank 的 StateImage；与 Main 续算字段共用 |
| 本轮临时结果 | 首步／递归 tail、递归 hidden、selection、raw records、Main verify hidden | 执行 frame；最后消费者退役后复用 |

Main 已处理的输入前缀为 [0,F) 时，完整可恢复 checkpoint 应满足：

```text
Main frontier      = F
MTP frontier       = max(F-1, 0)
saved conditioning = h[F-1]，仅在 F>0 时有效
```

MTP 的每一列都只依赖 checkpoint 覆盖的实际 token/hidden。Pending anchor、
尚未验证的 proposal 和冻结 selection 不作为 canonical checkpoint 的组成部分。
F=0 的空来源不读取 conditioning；首次 prefill 建立它。

MTP tail 是 Kpool 已定义的三列 raw K/gate，phase 由 MTP frontier 决定。
StateImage 使用[公共 hidden 与 owner tail 布局](parallel.md#12-state-variants-与字段布局)，
完整上下文还包括 Main 的 KDA 分片及 MLA/Indexer tails。ContextDirectory 保存声明的
记录与历史；`F→max(F-1,0)` 的解释只属于模型 Session。

Stem 的 norm 接口继续接收连续 hidden。准备阶段在有效执行借用内解析保存的 hidden
字段，使用既有 `copy_bytes_async` 填入连续 frame staging，在同一 stream 中先复制、后计算；
跨 chunk 的首列也从该字段取得。F=0 不读该字段，按首段配对规则准备输入。递归 draft
直接消费 frame 中的 hnext。提交最后采用的 Main hidden 使用已有
`scatter_batch_columns(..., StateFieldView, ...)`；字段选择由 Device I64 offsets 提供。
这些访问不改变 norm 接口，也不让 normalization 接收 StateId 或 placement。

Provisional history 共享可读 canonical 前缀，并使用已保留的可写后缀；共享尾页需要
COW 时由目录安排。其页与 canonical 页共用 Context 容量，临时 tail 则通过相同
StateFieldView 合同绑定 frame，不为每个 draft step 申请完整 StateImage。
后缀重写等待旧读者退役；逻辑 frontier 回退不会取消在途 Graph／selection 消费。

## 6. 各种执行情况

### 6.1 Prefill 与跨 chunk 配对

Main prefill 产生当前 chunk 的真实 final hidden。对 chunk [A,B)：

- A=0 时，形成配对 (h[0],x[1]) 到 (h[B-2],x[B-1])。
- A>0 时，先使用保存的 h[A-1] 配对 x[A]，随后处理 chunk 内其余相邻配对。
- 完成后保存 h[B-1]；MTP canonical frontier 更新到 B-1。

每个配对只执行：

```text
embedding → paired_rmsnorm_concat → W_eh
          → input RMSNorm → Kpool History prepare
```

History prepare 一次消费 W_A 的历史前缀 [768,4096]，产生 latent/index raw data、
新完成 pools 和 tail；不执行 query、Indexer、attention、MoE 或 head。
Chunk 边界、四 token pool 边界和 MTP offset 分别处理，不能把它们视为同一边界。

模型可以复用 Main 的批量 final hidden 缓冲；仅在最后消费者完成后复用其 backing。
Checkpoint 发布等待 MTP history 与 conditioning 完成。Prefix restore 后从恢复的
frontier 继续同一配对规则，不重算已经完整保存的 MTP 前缀。

### 6.2 Draft

设 Main frontier=N>0，anchor 为已确定但尚未输入 Main 的 x[N]：

1. 首步使用 (h[N-1],x[N])，MTP ordinal=N-1，运行 Full prepare、完整 W_B、
   Indexer、Sparse MLA、O、MoE、final norm 和 proposal head。
2. 保存首步 selection 的 IDs、count 和冻结 E，以及首步之后的 raw tail。
3. 后续每步使用递归 hnext 和上一步 proposal，运行 QueryAndHistory prepare、
   W_B 的 MLA-Q 前缀、Sparse MLA、O、MoE、final norm 和 head。
4. 后续步骤沿显式映射复用首步的 IDs/count/E；不再评分，也不扩张首步 tail 的可见边界。

每步仍写自己的 provisional latent/pools/tail。首步 tail 使用独立 frame slot 保留，
后续步骤在其他临时 tail slots 中推进；它同时是 canonical 可采用候选，不能被覆盖。
Frozen E 与 selection 内容一同保留，不能只保留 IDs 再使用当前上下文长度解释它。

各请求可拥有不同长度和有效 draft 数。固定的物理行映射、有效列和 owner-local
映射进入 Device metadata；空 owner 也参与模型已安排的后续 collective 顺序。

### 6.3 Verify 与 canonical history 补齐

Target 使用 Main head 验证 proposal。提交依据 Session 采用的真实 Main 输入列，
不能用递归 MTP hidden 直接替代对应真实 Main hidden。

首个 draft 输入已经使用真实 h 和实际 anchor。若 Session 确认其输入配对、前缀、
owner 与本次采用结果一致，且生产完成/租约仍有效，可以直接采用该列的
latent、pooled key 增量和所保留的 tail；不重新执行它的 stem/History prepare。

其后的必要列用 Main verify 实际产生的 h 和被采用 token，批量执行 History 链。
后续递归 draft 的 token 即使被接受，其 history 也不自动成为 canonical。

以非终止的普通一轮为例，接受 a 个 draft tokens：

```text
轮前 Main frontier  = N
轮前 MTP frontier   = N-1
Main 本轮采用输入   = anchor x[N] 加 a 个 draft tokens
轮后 Main frontier  = N+a+1
轮后 MTP frontier   = N+a

采用首步 MTP 列     = N-1
重建其余 a 列       = N .. N+a-1
最终 conditioning   = Main h[N+a]
```

因此 a=0 可以只采用首步历史；全部接受时，最后一个 canonical 配对可能尚未在
draft 中作为输入出现，仍由真实 Main hidden 补齐。EOS、输出限额、强制控制或取消
按实际采用的 Main 输入列计算，不能直接套用 `accepted_drafts+1`。

没有可采用首步候选时，按实际 canonical 缺口执行完整 History 链。这个入口也服务于
恢复后的补齐和不经过 draft 的控制路径。Main KDA 的 replay/fold 与各 Indexer 的
tail 提取仍按各自合同执行；MTP 补齐不能代替 Main 的状态提交。

### 6.4 提交、失败与释放

Session 先确定采用范围；RankProgram 在所有旧读者完成后写入目标后缀和 state，
再报告完成；目录和 Session 才发布新 frontier/checkpoint。未采用的尾部不对外可见，
backing 在最后消费者完成后回收或复用。

整轮许可覆盖 Main verify、MTP provisional/COW、canonical 补齐及目标 State 的峰值。
采用结果确定后，剩余目的位置从原许可取得，不在提交中重新竞争池容量；可复用的
未采用页按目录合同退回预留，其他记录在消费者退役后释放。

取消或失败不发布半套 Main/MTP 上下文。已提交的设备工作和存储动作按已有执行事务
退役，不通过提前释放 frame、selection 或 page lease 来取消 CUDA 工作。
这些依赖使用既有 ticket/stream/event 机制，不在 MTP 新建线程池或提交系统。

## 7. 双机放置与通信边界

放置权威在[并行设计](parallel.md#1-首版并行分工)：

| 计算 | 输入与交付 |
|---|---|
| Embedding、stem、norm、residual | 两 rank 复制计算，消费相同物理 token 行；每 rank 的 Main/MTP 共享 embedding |
| MLA／Indexer | 请求 owner 执行完整 heads、持有该请求的 MTP 历史；Main/MTP owner 相同 |
| MLA O 之后 | 按冻结行映射汇集完整 [4096,M]，两 rank 再执行 residual/norm |
| Routed MoE | EP2，本地 144 experts；global route 与系数一致 |
| Shared | intermediate TP2，本地 1024；与 routed 局部结果合并后统一求和 |
| Final norm | 消费完整 MoE 结果，两 rank 产生一致的 hnext |
| Proposal head | vocabulary TP2，每 rank 77440 rows，沿用统一采样/约束/接受接口 |

复制 stem 使小 T 不必为这份投影增加单独的输入交换；其输出在 owner 侧通过行 view
或已有 pack 进入 MLA。History-only 链同样在两侧形成 stem/input norm，只有 owner
写对应 cache。两侧不为 MLA 创建重复请求历史。

Head 的 global token ID 使用本地词表偏移。Greedy draft 可以只归约局部候选，
其他采样/约束消费其所需的 logits，不默认汇集整份 [154880,M]。
模型和公共 speculative decision 合同负责交接；Linear 不知道 rank 或 token ID。

## 8. 资源与 Graph

以下为当前几何的载荷，精确启动占用另计对齐、workspace 和实际保留窗口：

| 对象 | 单 rank 容量或计费方式 |
|---|---:|
| 复制的共享 embedding | 1210 MiB；Main/MTP 共用一份 |
| 复制的 W_eh | 64 MiB |
| Q4 proposal head 本地分片 | 160.703125 MiB，包含 FP16 scales |
| Main final hidden／conditioning 字段 | 8192 B，已计入 StateImage；另按实际 batch 计 frame staging |
| 成对 norm/concat 出口 | 16384 B/物理列 |
| 一个普通 hidden／residual 缓冲 | 8192 B/物理列 |
| MTP canonical／provisional cache | 按 Kpool 页布局与峰值需求占用 Context |
| MTP canonical tail | 已计入 owner StateImage |
| 临时 tail、selection、raw records | 按实际本地 owner 行容量占用 frame |

Recursive hidden、两组 residual/normalized buffers 和 first-step tail 按同时存活范围
规划。Verify final hidden 至少保留 canonical 补齐所需的真实列及最后 conditioning；
不能提前被 Main head scratch 覆盖。Projection/MLA/MoE scratch 使用各 Op 的无地址
容量查询，公共 selection/records 不算进临时 arena。

Model Graph 固定调用顺序和物理形状，Device metadata 控制行有效性、ordinal、
enabled、State offsets、页映射和选择内容。Saved hidden 的准备复制在 replay 前提交，
Graph 读取固定 frame staging；字段写回读取当前 Device offsets。各次执行借用覆盖
实际消费者，记录搬移后刷新 offsets／映射，不捕获永久 payload 地址。
Norm Ops 不读回 Host；首步 Full、后续 QueryAndHistory、canonical History 分别进入
相应固定执行拓扑。Kpool 不增加单独的模型子图要求，prefill 仍走 eager。

## 9. 文件组织与实施顺序

```text
include/ninfer/ops/
    paired_rmsnorm_concat.h
    residual_add_rmsnorm.h
src/ops/normalization/
    paired_rmsnorm_concat.cu
    residual_add_rmsnorm.cu
src/models/glm5_next/
    execution/mtp.*       stem、block、history-only 的模型组合
    state/                frontier、conditioning、tail 的模型绑定
    program/              frame、通信、Graph 和提交执行
```

两个简单 norm Ops 初始各用一个实现文件；需要多种复杂路线时再扩展目录。
数学公共基础放在现有 ops/common，模型目录不出现 CUDA 数学 kernel。

1. **算子与权重准备**：完成两个 norm Ops，接入既有 stem/head 纯 Linear 成果，
   完成可选 head 的 converter/binding。与 Kpool/MoE 实现可并行。
2. **Canonical 基础**：实现 prefill 配对、跨 chunk conditioning、History prepare、
   prefix capture/restore 与 owner 放置。
3. **完整 MTP**：连接递归 draft、冻结 selection、Main verify、首步采用和批量历史补齐，
   接入既有 decision/constraint/取消流程。
4. **Graph 与调优**：复用统一 program 结构捕获 decode，并按实际 draft/target/commit
   时间定位；与 Vision 组合只增加其输入语义检查，不另建 MTP 路径。

## 10. 验证与性能口径

算子验证使用独立数学参考：分别覆盖两份 gamma、mask、concat 顺序、真实 BF16
residual 边界及合法原位。精度要求由接口决定，不依据某条私有路线的内部实现提高门槛。

状态验证聚焦实际边界：首列、跨 chunk、非四 token 对齐、prefix restore、零/部分/全部
采用、EOS/控制后的实际采用列、不同 owner 和混合长度。检查完整 latent/index/tail/
conditioning，而不只比较最后 logits；首步采用与重建应满足相同的规范化历史语义。

性能分开记录 stem/norm、MLA/Indexer、MoE/head，以及完整 draft、target、commit 和
decode step。小 M=1..8 是重点，同时覆盖并发物理列与 History 的 2048/4096；
比较时保留上下文、有效行数、draft 数和存储 profile。报告实测耗时与流量/FLOPs
roofline，未测之前不推导端到端加速比或接受率变化。

## 11. 参考入口

- [模型语义](../../maintainer/glm5next-model.md#61-原生-mtp)、
  [量化与数值](../../maintainer/glm5next-quantization.md)、
  [Kpool Sparse MLA](kpool-sparse-mla.md)、[并行与资源](parallel.md)。
- 本地 vLLM：`vllm/models/glm5next/common/mtp.py` 的 norm/stem、final hidden 和
  head；`vllm/v1/spec_decode/llm_base_proposer.py` 的输入移位及媒体 embedding 选择。
- 本地 SGLang：`python/sglang/srt/models/deepseek_nextn.py` 的多模态输入分支。
  这些实现用于确认组合和差异，数学资格仍使用独立参考。
