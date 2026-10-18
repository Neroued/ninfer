# GLM5Next：Spark TP2 / EP2 基础 Op 清单

目标：为两台 DGX Spark 上的 GLM-5.3-Flash 准备本地 GPU 算子，完成本清单已确定的
Linear、无状态投影融合、dense/shared MLP 与 Main/MTP MoE 支持和调优。
架构、数学和量化决策见[模型与权重](glm5next-model.md)。部署只考虑 TP2/EP2；
不为 GLM 另做单 rank 完整几何或 RTX 5090 的新 profile。

本清单沿用 `.50` 的 `docs/qwen4-exp-spark-tp2-ep2-op-checklist.md` 形式。
执行基线是 `.50` `/home/neroued/ninfer` 已完成的 Qwen TP2/EP2 本地算子成果，
尤其复用其 indexed expert bank 和本地子集计算。未勾选表示本轮任务未完成，
不等于现有实现不支持；已有适用实现先确认、测量，再决定是否需要改动。

**KDA 计算链暂缓，等待 ReplaySSM 定案。** 本轮只准备其权重布局已确定的纯投影和
无状态 split。MLA、Indexer、mHC 等尚未收敛接口的计算不进入本轮实施。
不开展 converter、权重全量转换、模型接入、通信或引擎重构。

## 1. 局部分工、合并与实施顺序

### 1.1 本轮采用的局部几何

| 部分 | TP2/EP2 分工 | 每 rank 计算 |
|---|---|---|
| Block hidden | 复制 4096 维输入 | 同一组 tokens，T/B 不减半 |
| KDA 投影 | 按完整 heads 切为 32＋32 | Q/K/V 各 4096 维；beta 32；f_a/g_a 各完整 128 维；f_b/g_b 各 4096 维 |
| Main 前三层 dense MLP | intermediate 12288→6144＋6144 | 本地 clamped SwiGLU，down 输出 4096 维局部贡献 |
| Main/MTP shared expert | intermediate 2048→1024＋1024 | 本地 clamped SwiGLU，输出直接加入本 rank 的 MoE 贡献 |
| Main/MTP routed experts | EP2，expert 内部不做 TP | 全局 288 个 experts，每 rank 实存 144 个完整 experts，global top-8 |
| Router | 复制 | 完整 288 维 logits、correction bias 和同一 global Top8 |
| Main/MTP LM head | vocabulary/N 轴 TP2 | 每 rank `[77440,4096]`，Main 与 MTP 共享权重 |

Dense/shared 的 gate/up 分别沿 intermediate 行切片，down 沿相同 intermediate 的 K
范围切片。它们逐通道的 clamp/SiLU/product 不跨分片，因此两个局部输出之和对应完整
FFN。Shared 是一个 FFN 的两份 intermediate 分片，不在两侧各计算一份完整 shared。

Routed experts 的归属通过显式 global ID→本地 payload 映射给出，本地集合大小为 144。
本地 Op 不固定连续区间、奇偶分配或负载均衡策略；这些由后续并行计划提供。

### 1.2 公共约定

- 权重写作 `W[N,K]`，输入 `[K,M]`，`M=T*B`、`column=t+T*b`；native Tensor 第 0
  维连续。下面的无状态算子可按 M 展平，KDA 序列与状态语义不由 M 代替。
- 普通 Linear、split、MLP 和本地 MoE 的公开输入/输出均为 BF16。沿 K 切片的 down/O
  以及 EP 局部和也输出 BF16。Router 的内部 logits 与公开 coefficients 使用 FP32。
- 本地 FFN/MoE 输出使用 Write 语义。跨 rank 求和之后才进行 Main mHC 或 MTP residual
  消费；不在两侧各加入一份完整 residual。
- **先完成对应 shape 的纯 `linear()`，再扩展 split、clamp/SiLU/product 及完整 FFN。**
  每条依赖链独立推进。融合 Op 按完整数学验证，纯 Linear 的 BF16 输出不强制成为其
  内部舍入边界，融合也不要求先调用公共 Linear 再串接后处理。
- 新增 clamp、路由系数等语义必须在公共 profile/参数和合同中显式表达，不根据模型名
  或 shape 偷换原算子的数学。沿用相应 family，Spark 拥有自己的分派；现有无 clamp
  路径及其他模型已支持的行为继续成立。
- 权重、输出与 caller-owned workspace 的 view/对齐/不重叠要求沿用所在 family；
  容量查询覆盖完整调用与请求的 M 区间。无隐藏 device allocation、传输或 Host readback，
  支持 CUDA Graph capture/replay。本地 Op 不接收 rank、world size 或 communicator。

依赖关系：

```text
P1 → F1：QKV split                    P3 → F2：beta / f_a / g_a split
                                      └→ R1：FP32 logits / sigmoid / global Top8
P5 → F3：NVFP4 clamped GU ┐
P6 ─────────────────────┴→ M1：完整 dense MLP 局部贡献
P7 → F4：BF16 clamped GU ┐
P8 ─────────────────────┴→ M2：完整 shared MLP 局部贡献
R1 + E1 + M2 → E3：Main 完整本地 MoE
R1 + E2 + M2 → E4：MTP 完整本地 MoE
P2 / P4 / P9 独立完成；E1 / E2 在 MoE family 内完成 contraction 与融合
```

## 2. 纯 Linear：九个 profile

纯 Linear 阶段只写单个连续 BF16 输出，不包含 split、norm、卷积、clamp 或 routing。

| 编号 | 权重表示 / permission | 本地 W[N,K] | 用途 |
|---|---|---|---|
| P1 | Q8-G32/FP16，A16Only | `[12288,4096]` | KDA 合并 QKV：`[Q4096;K4096;V4096]` |
| P2 | Q8-G32/FP16，A16Only | `[4096,4096]` | KDA O，沿输入 heads/K 切片的局部和 |
| P3 | BF16，A16Only | `[288,4096]` | KDA 合并 `beta32;f_a128;g_a128` |
| P4 | BF16，A16Only | `[4096,128]` | KDA f_b 与 g_b，两次独立投影，共用同一 profile |
| P5 | NVIDIA NVFP4，AllowA4 | `[12288,4096]` | dense MLP 本地 GU：`[gate6144;up6144]` |
| P6 | NVIDIA NVFP4，AllowA4 | `[4096,6144]` | dense MLP 本地 down |
| P7 | BF16，A16Only | `[2048,4096]` | Main/MTP shared 本地 GU：`[gate1024;up1024]` |
| P8 | BF16，A16Only | `[4096,1024]` | Main/MTP shared 本地 down |
| P9 | BF16，A16Only | `[77440,4096]` | Main/MTP LM head 词表分片 |

- [ ] **P1**：Q8 `[12288,4096]`。
- [ ] **P2**：Q8 `[4096,4096]`。
- [ ] **P3**：BF16 `[288,4096]`。
- [ ] **P4**：BF16 `[4096,128]`；用不同 f_b/g_b 权重和各自输入确认角色复用。
- [ ] **P5**：NVFP4 `[12288,4096]`、AllowA4。
- [ ] **P6**：NVFP4 `[4096,6144]`、AllowA4。
- [ ] **P7**：BF16 `[2048,4096]`。
- [ ] **P8**：BF16 `[4096,1024]`。
- [ ] **P9**：BF16 `[77440,4096]`。

切片与合并约定：

| 对象 | 本地行/列范围 |
|---|---|
| KDA heads | 两侧分别取原 heads `[0,32)`、`[32,64)`，每个 head 完整保留 128 个 channels |
| P1 | 从源 Q、K、V 各自取对应 4096 行，再按 Q/K/V 排列；不直接平分全量 QKV parent |
| P2 | N=4096 保留，K 取与本地 KDA output heads 相同的 4096 列 |
| P3 | beta 取本地 32 行；f_a/g_a 各复制完整 128 行，顺序为 `[beta;f_a;g_a]` |
| P4 | f_b/g_b 各取本地 4096 输出行，K=128 保留；f_a 与 g_a 是不同输入，不能将 f_b/g_b 当作一次同输入投影简单拼接 |
| P5/P6 | 每侧各取 gate/up 的 6144 行，以及 down 对应的 6144 列 |
| P7/P8 | 每侧各取 gate/up 的 1024 行，以及 down 对应的 1024 列 |
| P9 | 两侧分别是物理词表 `[0,77440)`、`[77440,154880)`；词表有效性与 token ID 决策由后续输出层处理 |

Q8 的 K 切点保持 G32 对齐，NVFP4 的 K 切点保持 G16 对齐。保留 source codes/scales
及 weight/input divisors 的含义，按本地矩阵重排存储；不因分片重新量化或重新校准。
P5/P6 的 GU/down 分别使用源 dense MLP 对应的 activation divisor。

P3 的数值几何恰好与 router 的 `[288,4096]` 相同，但其 BF16 输出不能作为 R1 的
logits 输入。R1 要求完整 FP32 logits；可以共享 contraction 基础，不能共用错误的出口。

## 3. 无状态投影融合

### 3.1 LinearSplit

- [ ] **F1 — KDA QKV split，依赖 P1。** Q8 `[12288,4096]`，直接写出三个独立连续
  BF16 输出 Q/K/V，各 `[4096,M]`。这里输出的是**卷积前投影值**，不做 SiLU、
  normalization、卷积或 history 更新。
- [ ] **F2 — KDA control split，依赖 P3。** BF16 `[288,4096]`，直接写出
  `beta_raw[32,M]`、`f_a[128,M]`、`g_a[128,M]`。Beta 尚未 sigmoid，f_a/g_a
  后仍各自有第二次投影；本项不融合任何 KDA gate/recurrence 语义。

沿用 [LinearSplit](../include/ninfer/ops/linear_split.h) 的分段数学与完整 capacity/launch
约定。调优目标是 contraction 直接写连续分段，最终实现不额外物化完整输出后再复制。
无状态 split 的确定不承诺未来 KDA 状态融合必须消费这组中间 buffers。

### 3.2 合并 GU＋clamped SwiGLU

本轮明确新增的非线性为：

```text
gate = sum_k(decode(W_gate)[i,k] * x[k,m])
up   = sum_k(decode(W_up)[i,k]   * x[k,m])
product[i,m] = SiLU(min(gate, limit)) * clamp(up, -limit, limit)
limit = 10
```

`limit` 是显式正有限语义参数，本模型取 10；gate 只截上界，up 截上下界。
Clamp 在 SiLU/product 之前。此参数同步进入相应 SwiGLU/MLP/MoE profile；已有无 clamp
计算保留原义。按 family 一致的接口表达，不引入 checkpoint 专用执行入口。

- [ ] **F3 — NVFP4 GU＋clamped SwiGLU，依赖 P5。** 输入 BF16 `[4096,M]`、GU
  NVFP4 `[12288,4096]`、AllowA4，输出连续 BF16 `[6144,M]`。使用 GU 的原
  activation divisor；gate/up 同一个对应位置产生一个 product。
- [ ] **F4 — BF16 GU＋clamped SwiGLU，依赖 P7。** 输入 BF16 `[4096,M]`、GU
  BF16 `[2048,4096]`、A16Only，输出连续 BF16 `[1024,M]`。

现有 [Linear SwiGLU](../include/ninfer/ops/linear_swiglu.h) 只有无 clamp 公式，且其
当前格式入口未包含 BF16，F3/F4 要同步补齐公共合同、格式支持、capacity、分派与测量。
融合计算不要求先把 gate/up 舍入为 BF16，再执行非线性。

## 4. 完整 dense/shared MLP 的本地贡献

这里的完整 MLP 数学为 `y=W_down*product`，product 使用第 3.2 节定义。
公开输入是 BF16 `[4096,M]`、GU/down 权重及 activation permission/metadata，
输出是连续 BF16 `[4096,M]`，Write 语义，无 routes、gate coefficient 或 residual 输入。

- [ ] **M1 — Main dense MLP，依赖 P5/P6/F3。** 本地 I=6144，GU/down 均为 NVIDIA
  NVFP4、AllowA4。完整调用覆盖输入量化、GU、clamp/SiLU/product、down 输入量化、
  down 与输出写入。保留源 GU/down 各自的 activation divisor。
- [ ] **M2 — Main/MTP shared MLP，依赖 P7/P8/F4。** 本地 I=1024，GU/down 均为
  BF16、A16Only，输出为本地 shared contribution，系数为数学上的 1。入口直接表达
  无缩放 shared FFN，不要求调用方伪造 router 行或全 1 coefficient tensor。

F3/F4 为独立 GU-product 算子，其公开输出是 BF16。M1/M2 是完整 FFN 语义，内部
product 不增加强制 BF16 舍入边界；特别是 M1 的 A4 down 可以从内部 product 直接
量化。独立阶段有助于实现与定位，但最终以完整 MLP 的数学与时间验收。

这些计算进入现有 SwiGLU/MoE 计算基础，复用合适的 contraction。是否使用持久
kernel、分阶段 kernels 或共享 scratch 属于实现选择，不要求一个 CUDA kernel 包办。

## 5. GLM sigmoid＋correction-bias 路由

- [ ] **R1 — 完整 global Top8 路由，先完成 P3 的纯投影准备。** 每 rank 使用相同 BF16 input `[4096,M]`、
  BF16 router weight `[288,4096]`、FP32 correction bias `[288]`，输出连续 I32 IDs
  `[8,M]` 和 FP32 coefficients `[8,M]`。

本项的完整数学固定为：

```text
logit[e,m] = FP32_Dot(router_weight[e,:], x[:,m])
s[e,m] = sigmoid(logit[e,m])
choice[e,m] = s[e,m] + correction_bias[e]
ids[:,m] = Top8(choice[:,m])
coefficient[j,m] = 2.5 * s[ids[j,m],m]
                   / (sum_{i=0..7}(s[ids[i,m],m]) + 1e-20)
```

IDs 按 choice 降序，同值按较小 global expert ID 排在前。Correction bias 只用于
选择，不进入最终系数。Top-k=8、scaling factor=2.5、normalization epsilon=1e-20
作为语义配置明确传入/记录；本实例只有一个 expert group，本轮不扩展 grouped routing。

Router weight 保留原 BF16 words，logits 直接以 FP32 计算和消费，不经过普通 Linear
的 BF16 输出。每 rank 复制同一 global 路由，采用相同数值路径和 tie-break；不先将
experts 分成 144 个再各选 Top8，也不输出 shared sigmoid coefficient。

现有 `topk_softmax_route` 的 softmax＋额外 shared logit 公式不适用。本项应提供
明确的 sigmoid/correction 路由语义，可以复用公共 TopK 和投影基础。整个入口包含
projection、sigmoid、TopK、系数归一化与写出，计时覆盖完整调用。

## 6. Main/MTP EP2 与完整本地 MoE

### 6.1 已知 routes 的局部 expert 贡献

| 对象 | 每 rank 要求 |
|---|---|
| Global E / top-k / local E | 288 / 8 / 144；每 token 本地命中数可以是 0..8 |
| Bank | 仅有本地 144 个 experts 的 payload，保留显式 global-ID 映射与本地 subset |
| 单 expert GU / down | `[4096,4096]` / `[4096,2048]`；GU 的 gate/up 各 2048 行 |
| Input | 连续 BF16 `[4096,M]` |
| Routes | Global I32 IDs `[8,M]`、FP32 coefficients `[8,M]`，直接消费、不重新归一化 |
| Output | 连续 BF16 `[4096,M]` 本地贡献，Write；无本地命中时写零 |
| 激活 | 第 3.2 节 clamped SwiGLU，limit=10 |

```text
y_routed_local[:,m] = sum_{j: ids[j,m] belongs to local bank}
                       coefficient[j,m] * expert_ids[j,m](x[:,m])
```

- [ ] **E1 — Main NVFP4 EP 贡献。** AllowA4；GU/down 的 G16 E4M3 scales、每 expert
  weight divisors、每层 GU/down activation divisors 保持原含义。扩展本地 H=4096、
  I=2048、global E=288、top-k=8 几何及 clamp 数学，完成小 M 与分组大 M 的调优。
- [ ] **E2 — MTP block-FP8 EP 贡献。** AllowA8；128×128 BF16 block scales，
  activation 按 token 的 K128 动态量化，无静态 activation divisor。其余几何、
  local-ID 映射和 clamp 数学与 E1 相同。

沿用 `.50` 已完成的 indexed bank、`ExpertSubset` 和本地贡献机制，不为方便测试
分配完整 288 份 payload。不读取未归属 expert 的地址或权重，不将 top-k 改成 4。
本地 subset 过滤不改变源系数，系数 2.5 的含义保留在 R1 的输出中。

Expert banks 不列入普通 Linear 的九个 profile。MoE family 内先落实本地矩阵
contraction 的正确性，再加入 clamp、route weighting 与 reduction；不为专家银行
增加一套普通 `linear()` 接口。测试可构造合法编码权重，不依赖全量 GLM 转换器。

### 6.2 Router＋EP＋shared 的完整局部融合

完整局部 MoE 消费以下显式数据：BF16 block input、R1 router 权重和 bias、global
路由配置、本地 routed bank/subset、TP2 shared GU/down、clamp 参数及对应的
activation permissions/divisors。对外只写 BF16 `[4096,M]`：

```text
y_local = routed_contribution(R1(x), local_bank, x)
          + shared_intermediate_shard(x)
```

- [ ] **E3 — Main 完整本地 MoE，依赖 R1/E1/M2。** NVFP4 routed＋BF16 shared，
  global top-8，local E=144、shared I=1024，输出完整本 rank 的贡献。
- [ ] **E4 — MTP 完整本地 MoE，依赖 R1/E2/M2。** Block-FP8 routed＋BF16 shared，
  与 E3 共用语义入口和参数组织，格式与 permission 决定内部实现。

Routes 和暂存 contribution 可以作为 caller workspace 中的私有中间量，不强制对外
发布；融合输出按完整局部公式限定一次 BF16 出口。零 routed 命中仍计算本地 shared。
Shared 只计一次，既不乘 routed scaling factor，也不乘额外 sigmoid。

两 rank 的 `y_local` 后续相加才是完整 MoE。Main mHC 与 MTP residual 更新均在
这个通信边界之后，E3/E4 不承担 residual、通信或状态提交。融合目标是减少重复
输入读取、分配与中间贡献写出，并允许 route/分组准备与 shared 计算合理组织；
不要求强制将全流程塞进同一个 kernel。

## 7. 调优范围与完成方式

沿用[Op 开发规范](maintainer/op-development.md)、[Linear 调优方法](maintainer/linear-tuning.md)
及现有 [tests](../tests/README.md) / [bench](../bench/README.md)。先检查本项的
fixture、独立数学参考、precision policy 与流量模型，再实现和调优。验收以公共 Op
完整调用为单位，容量查询、分派、测试和 bench 同步覆盖新增几何。

### 7.1 投影与 MLP

| 范围 | 工作负载与目标 |
|---|---|
| 小 M | 每个整数 `M=1..128`；重点 1、4、8，另保留 6；降低 M=1 延迟，随 M 增加复用权重与有效并行度 |
| 并发/spec | 普通生成/单步 proposal 为 `M=B`；verify 的代表宽度为 `M=4B、6B、8B、16B`，B=1..8，不固定实际 draft 策略 |
| 中间区间 | M=256 与实际 route 切换附近补点，不逐点扫描到 4096 |
| 大 M | **512、1024、2048、4096** 均为独立锚点；报告总延迟、每列成本、带宽与实际算术吞吐参考 |

P3 的窄输出与 P4 的窄 K 单独观察启动/归约成本，P9 计入完整 logits 写出。
P1/P2 的 Q8 是 A16 权重解码路径，不能套用 INT8/NVFP4 Tensor Core 峰值。
NVFP4 的 `AllowA4` 不等于每个 M 都采用 A4；计算参考由实际生产分派的算术决定。

纯 Linear 的 useful FLOPs 为 `2*N*K*M`，weight bytes 按实际 codes/scales；
BF16 输入/输出分别为 `2*K*M`、`2*N*M`。F1/F2 计入全部 split 输出；F3/F4
计入 clamp/product 的完整出口；M1/M2 计入两次 contraction、必要量化与 workspace
读写，不能用 GU 内核耗时代替完整 MLP 结果。

### 7.2 路由与 EP/MoE

R1 使用与投影相同的 M 区间，单独观察 FP32 logits、TopK 与 coefficients 写出。
E1–E4 覆盖 `M=1..128` 和上述大 M 锚点，使用代表性的 0 命中、全部 8 个命中、
两侧混合、跨 token 重复热点与较分散分布；不在每个 M 做所有分布的笛卡尔积。

本地实际 token-expert 数 `P=sum_m(local_hits[m])` 决定 routed 的 useful
contraction FLOPs：`6*4096*2048*P`。按实际访问的 expert 集合计算去重权重载荷，
同时保留 token-expert 工作量；不能用固定 global top-k/2 推定载荷。
单 expert GU＋down 的编码载荷约为：NVFP4 **13.5 MiB**、block-FP8
**24.00293 MiB**，另计 NVFP4 scalar metadata。E3/E4 另计完整 router 和 shared TP2
分片，shared 不因 local routed hits=0 而消失。

小 M 重点是权重读取、启动与零/少命中成本；大 M 重点是分组复用、量化准备、
reduction 与 workspace 流量。Graph 的冷/热缓存结果分列，保留代表点 Eager 的
提交开销；以 Spark 的带宽及实际执行精度引用 roofline。记录中位数、P95、重要切换点、
最差变化和仍存在的限制，不将逻辑 GB/s 直接解释为 DRAM counters。

### 7.3 数学与回归检查

- Pure Linear、split、GU product、完整 MLP、routes、EP contribution 和完整局部
  MoE 分别对其独立数学参考验证。融合参考不由已舍入的阶段输出拼成。
- Clamp 覆盖 gate 上界与 up 双边界，route 覆盖 correction bias 改变选择但不改变
  系数来源、tie-break 和 normalization。精度标准按公共语义/permission，
  不按内部实际走 A16/A8/A4 路径切换不同验收口径。
- EP 覆盖非连续 global IDs、实际 144 份 bank、0/8 本地命中、共享项只加一次和
  caller workspace 范围；两份局部数学贡献可在单机测试中组成完整参考，无需通信实现。
- 检查被修改共享基础实际影响到的既有无 clamp/softmax profile；不扩展成与本任务
  无关的模型回归或整机 benchmark。已有能力通过本轮检查即可保留，不要求重写。

完成一项后更新本清单并关联简短的验证/性能结果。新增 capability 与已有优化路径的
改进分别说明；本地 Op 结果不声明为双机模型吞吐收益。

## 8. 本轮暂缓

以下内容不交给本轮执行 agent 实现或调优：

- **KDA 全部计算链与状态接口**：卷积、Q/K normalization、bounded gate、递推、
  batch update、output gated norm，以及 projection＋conv/gate 等 KDA 融合；
  snapshot、record、fold、ReplaySSM 与 accepted-prefix 恢复须先整体定案。
  现有 `kimi_delta_attention` 可参考，但不因此提前开展 KDA 优化。
- **MLA 与 Indexer**：latent attention、准备/cache/codec、Kpool4、评分/TopK/tail；
  其外围投影的最终合并、吸收与本地布局也先不固定。MLA FP8 cache 暂不讨论。
- **mHC**：pre/post、Sinkhorn、norm/residual 组合、final stream collapse 及与相邻
  block 的融合；这些接口需单独收敛。
- **MTP stem、embedding、Vision**：本地放置和算子组合不在本轮固定，暂不增加其
  Linear/fusion profiles。MTP MoE 与共享 LM head 的准备已包含在前文。
- 分片 logits 的候选合并、sampling、MTP acceptance、scoring；本轮 P9 只产生本地
  Linear logits。
- 全量量化/converter、模型绑定与加载、通信/通算融合、Engine/Program 接入、跨 rank
  状态与资源规划，以及完整模型性能测量。

这些部分的方案确定后再追加需求。执行过程中若某项依赖上述未定接口，应停止该依赖
上的扩展并反馈具体边界，继续其余独立条目。
