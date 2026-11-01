# Qwen4-27B：RTX 5090 架构与量化方案

日期：2026-10-27。

本文确定 NInfer 在单张 RTX 5090 上适配 Qwen4-27B 的架构参数、权重分配、
矩阵计算精度和资源预算。目标配置为 262144-token 上下文、FP8 KV、Vision，
并在同一权重预算内配置 MTP。

RTX 5090 的 Op 准备范围、当前支持缺口和验收条件见
[Op 需求清单](qwen4-27b-op-checklist.md)。本文维护配置与精度选择，清单维护实施需求。

## 1. 架构参数

| 项目 | 配置 |
|---|---|
| 架构 | Dense，GDN/QSA 混合，四路 Gated Residual / Hyper-Connection |
| Decoder 层数 | 64 |
| Hidden size | 5120 |
| FFN intermediate size | 14336 |
| FFN 激活 | SwiGLU |
| 层分布 | 每四层为 GDN、GDN、GDN、QSA；共 48 GDN、16 QSA |
| Vocabulary size | 248320 |
| Embedding / LM head | 不绑定，分别持有权重 |
| HC streams | 4 |
| HC residual width | 20480 |
| HC lowrank | 640 |
| HC 数量 | 每层两组，共 128 组；另有一个 final mixer |
| 上下文长度 | 262144 |
| RoPE theta | 10000000 |
| Rotary dimension | 64，主 attention head 的前 25% |
| MRoPE | interleaved，sections `[11,11,10]` |
| RMSNorm epsilon | `1e-6` |
| PLE / N-gram | 不配置 |
| MTP | 一个独立 HC/QSA/Dense FFN 层 |
| 语言主干参数量 | 27,322,657,280 |

主干执行顺序：

```text
embedding / Vision features
  → 扩展为四路 residual streams
  → 每层：attention HC → GDN 或 QSA → HC combine
          FFN HC       → Dense SwiGLU → HC combine
  → final HC norm/mixer
  → LM head
```

HC 将四路 residual 混合为一个 5120 维 block 输入。GDN、QSA 和 FFN 均消费该输入。
Final mixer 输出 5120 维 hidden；MTP conditioning 使用 mixer 前的 20480 维 residual。

### 1.1 GDN

| 项目 | 配置 |
|---|---|
| Q/K heads | 16 |
| V heads | 48 |
| Key / value head dimension | 128 / 128 |
| Q、K width | 各 2048 |
| V、Z width | 各 6144 |
| A、B width | 各 48 |
| Convolution | 10240 channels，4 taps |
| Recurrent state | 每层 `[48,128,128]`，FP32 |
| 输出门控 | head-wise RMSNorm 后乘 `sigmoid(Z)` |

权重形状以 `[输出维度, 输入维度]` 表示：

| 参数 | Shape |
|---|---|
| QKVZ 合并投影 | `[16384,5120]` |
| A/B 合并投影 | `[96,5120]` |
| Output projection | `[5120,6144]` |
| Depthwise Conv1D 源权重 | `[10240,1,4]` |
| Gated RMSNorm weight | `[128]` |
| `A_log`、`dt_bias` | 各 `[48]` |

TP2 的 `[8192,5120]` 与 `[48,5120]` 对应 QKVZ 和 A/B 的局部分片。
单卡使用上表完整形状。

### 1.2 QSA

| 项目 | 配置 |
|---|---|
| Main Q heads / KV heads | 24 / 4 |
| Main head dimension | 256 |
| GQA ratio | 6 |
| Output gate | sigmoid，逐 query feature |
| Indexer Q heads / KV heads | 4 / 1 |
| Indexer head dimension | 128 |
| Indexer compression | 4 tokens / block |
| Indexer budget | 2048 tokens，即 512 个完整块 |

| 参数 | Shape |
|---|---|
| Q + output gate projection | `[12288,5120]` |
| K projection | `[1024,5120]` |
| V projection | `[1024,5120]` |
| O projection | `[5120,6144]` |
| Indexer Q/K projection | `[640,5120]` |
| Main Q/K norm | 各 `[256]` |
| Indexer Q/K norm | 各 `[128]` |

QSA 保留完整主 K/V 历史。每四列 raw index-K 平均、舍入后执行 norm/RoPE，
形成一个压缩 index key。Indexer 选择完整块，Sparse GQA 读取对应主 K/V，
同时处理当前未完成块的可见尾部。压缩比例只作用于 index key。

#### 1.2.1 合并权重与行视图

上表按逻辑参数列出 shape。每个 QSA 层将 Q、output gate、K、V、index-Q、index-K
物化为一个 FP8 row-scale parent `[14976,5120]`，六段统一使用 `AllowA8`。
Main 的 16 层与 MTP 的独立层使用相同布局，各层仍持有自己的权重。

| 逻辑参数 | Parent 行区间，半开 | 行数 |
|---|---|---:|
| K | `[0,1024)` | 1024 |
| V | `[1024,2048)` | 1024 |
| index-K | `[2048,2176)` | 128 |
| Q | `[2176,8320)` | 6144 |
| output gate | `[8320,14464)` | 6144 |
| index-Q | `[14464,14976)` | 512 |

完整执行通过一次六段 `linear_split` 消费全部 parent；MTP 历史建立和接受后的
历史重建通过一次三段 `linear_split` 消费前缀 view `[2176,5120]`，仅产生 K/V/index-K。
完整 view 与前缀 view 共享同一个不可变 backing，不保存第二份历史权重。
具体阶段、激活 view 与生命周期见 [Op 清单第 4 节](qwen4-27b-op-checklist.md#4-qsa-与-mtp-历史补齐)。

转换时将源 Q/gate 的按 head 交错行解交错，并将源 Indexer 的 Q/K 行移到上表位置。
若导入已编码的 row-scaled FP8，codes 与对应 BF16 row scales 一起重排；从高精度源
编码时按同一逻辑行映射生成。各角色的 Binding 保留自身行范围，实际投影按完整或
三段前缀 view 合并调用。合并不改变各逻辑权重的值、参数量或第 7 节 payload 预算。

前缀 view 的 codes 指向 parent 的前 2176 行，scales 指向 parent scale plane 的
前 2176 项。Scale plane 的物理位置由完整 parent 布局确定，不能按前缀 shape
重新推算为紧接前缀 codes 的位置。视图的 shape、可读范围、backing 生命周期和
零拷贝绑定由权重接口表达，执行阶段不重打包。

### 1.3 HC 与 FFN

| 参数 | Shape | 数量 |
|---|---|---:|
| HC norm | `[20480]` | 129 |
| HC down | `[640,20480]` | 129 |
| HC up | `[20480,640]` | 129 |
| HC injection | `[4,20480]` | 128 |
| FFN gate | `[14336,5120]` | 64 |
| FFN up | `[14336,5120]` | 64 |
| FFN down | `[5120,14336]` | 64 |

Final HC mixer 只有 norm、down、up。HC 数学为：

```text
z = grouped_offset_rmsnorm(residual)
u = SiLU(W_down × flatten(z) / 4)
g = sigmoid(W_up × u)
block_input = mean_streams(g × z)
injection = 2 × sigmoid(W_inject × flatten(z) / 4)
residual_out = residual + injection × block_output
```

Norm 按每条 stream 独立归约，gain 使用 `1 + weight`。
Residual 的公开表示为 BF16，下一次 norm 消费实际写入的 BF16 residual。

## 2. 权重格式与计算精度

格式定义引用[数值格式](../maintainer/weights-and-loading.md#数值表示与物理布局)，物理编码引用
[存储布局](../maintainer/weights-and-loading.md#数值表示与物理布局)。

| 简称 | 持久格式 | 权重体积 |
|---|---|---|
| NVFP4 | `nvfp4` | E2M1 codes + G16 E4M3 scale，约 4.5 bit/参数 |
| FP8 | `fp8_e4m3fn_row_bf16` | 1 byte/参数 + 每行 2-byte scale |
| Q4 | `q4_g64_fp16` | 4.25 bit/参数 |
| Q5 | `q5_g64_fp16` | 5.25 bit/参数 |
| Q6 | `q6_g64_fp16` | 6.25 bit/参数 |
| Q8 | `q8_g32_fp16` | 8.5 bit/参数 |

矩阵激活策略采用 [LinearPolicy](../../include/ninfer/ops/linear.h) 的计算许可：
`AllowA8` 允许 A16 或 A8，`AllowA4` 允许 A16、A8 或 A4；普通 Linear 的公开输入/输出
仍为 BF16，融合 Op 的输出按各自合同规定。
Op 根据几何与实际列数选择路径，小 M 可以保留 A16。许可不强制低精度执行，
也不替代实际路径的数值与性能验证。Main/MTP 的同构模块使用相同许可，
不按模型角色另设精度分支。

## 3. 语言主干量化清单

| 模块 | 参数 | 权重格式 | 矩阵激活精度 |
|---|---|---|---|
| Embedding | token embedding | FP8 | 查表 |
| LM head | output head | FP8 | A16Only |
| 全部 64 层 FFN | gate/up/down | NVFP4 | AllowA4 |
| 48 层 GDN | Q/K/V/Z 投影 | NVFP4 | AllowA4 |
| 48 层 GDN | output projection | NVFP4 | AllowA4 |
| 48 层 GDN | A/B 控制投影 | BF16 | A16Only |
| 48 层 GDN | convolution、gated RMSNorm | BF16 | 对应算子合同 |
| 48 层 GDN | `A_log`、`dt_bias` | FP32 | 对应算子合同 |
| 16 层 QSA | Q + output gate、K、V 投影 | FP8 | AllowA8 |
| 16 层 QSA | O projection | NVFP4 | AllowA4 |
| 16 层 QSA | Q/K norm | BF16 | 对应算子合同 |
| Indexer | Q/K projection | FP8 | AllowA8 |
| Indexer | Q/K norm | BF16 | 对应算子合同 |
| 每层 attention/FFN HC | down/up | FP8 | AllowA8 |
| 每层 attention/FFN HC | injection | BF16 | A16Only |
| 每层 attention/FFN HC | norm | BF16 | 对应算子合同 |
| Final HC mixer | down/up | FP8 | AllowA8 |
| Final HC mixer | norm | BF16 | 对应算子合同 |

全部 FFN 使用相同分配。HC injection 独立于 down/up 保存格式；injection 的 FP32
系数广播到对应 stream。主模型 embedding 和 LM head 分别持有 FP8 权重。

NVFP4 权重使用 NVIDIA NVFP4 checkpoint 或用户的 ModelOpt 量化结果，转换时保留
codes、block scales、weight scale 和对应 activation scale 的数值语义。
源中的融合投影和 NInfer packing 按逻辑参数及 scale 合同组织。

### 3.1 精度选择与风险

| 部分 | 保留当前精度的依据 | 验证重点 |
|---|---|---|
| HC injection | 每个系数广播到一条 5120 维 stream，每层两次回注；Main BF16 权重共 20 MiB，改 FP8 仅省约 10 MiB | 多层 residual 与最终输出，不能只看局部投影误差 |
| GDN A/B | 决定 state 衰减与写入；Main BF16 权重共 45 MiB，改 FP8 仅省约 22.5 MiB，且仍与 NVFP4 QKVZ 分属不同计算表示 | 长序列递推与实际消费的 FP32 controls |
| Main LM head | 直接决定最终 logits，激活误差可能改变相近 token 的排序与采样概率 | 保持 FP8/A16Only；输出分布与评分 |
| HC down/up | 采用 FP8 存储并允许 A8 计算，进一步改 NVFP4 会影响各层 stream 混合 | 多层误差传播；不为统一格式直接降为 NVFP4 |
| Indexer projection | 投影误差可能改变 TopK 选块，造成历史信息遗漏 | 超过 512 个完整块的长上下文选择与检索质量 |

HC injection 权重继续独立使用 BF16/A16，系数保留 FP32；其格式差异由 HC Op 表达。
Norm、bias、卷积参数继续使用 BF16；GDN recurrent state、`A_log`、`dt_bias`
保持 FP32。小型控制参数和递推状态按实际用途保留精度，不以存储格式数量作为降精度依据。

## 4. Vision 量化清单

Vision 使用 27 层、hidden 1152、16 heads、head dimension 72、MLP intermediate 4304。
Patch 输入为 `[3,2,16,16]`，空间合并为 2×2，merger 为 `4608 → 4608 → 5120`。

| 参数 | 权重格式 | 矩阵激活精度 |
|---|---|---|
| Patch embedding projection | Q6 | A16Only |
| Attention Q/K/V | Q4 | A16Only |
| Attention output | Q5 | A16Only |
| MLP fc1 | Q4 | A16Only |
| MLP fc2 | Q5 | A16Only |
| Merger fc1/fc2 | Q8 | A16Only |
| Position embedding | BF16 | 查表 |
| Norm weight/bias | BF16 | 对应算子合同 |
| Projection bias | BF16 | 对应算子合同 |

直接复用现有公共 Qwen Vision 及其完整量化实现。上述 Q4/Q5/Q6/Q8 分配不增加
本次新格式开发，不为统一 Text 的 NVFP4/FP8 格式而重写 Vision。未来若更换 Vision
实现，再结合图像质量、编码延迟与资源预算重新设计其表示。

## 5. MTP 量化清单

MTP 使用一个独立的 HC/QSA/Dense FFN 层，主几何与语言主干一致。
Token embedding 与 Main 共享物理权重，草拟使用独立的 Q4 optimized draft head。
完整 output head 与 Main 共享物理权重，供 full-head 模式使用。MTP 输入包含当前 token embedding
和 Main 的 20480 维 HC residual；`fc_hidden` 以共享的 `[5120,5120]` 矩阵处理各 stream。

| 参数 | 权重格式 | 矩阵激活精度 |
|---|---|---|
| Token embedding | 共享 Main FP8 | 查表 |
| Optimized draft head | Q4，独立物理权重 | A16Only |
| Full output head | 共享 Main FP8 | A16Only |
| `fc_embedding`、`fc_hidden` | FP8 | AllowA8 |
| QSA Q + output gate、K、V | FP8 | AllowA8 |
| QSA O | NVFP4 | AllowA4 |
| Indexer Q/K projection | FP8 | AllowA8 |
| FFN gate/up/down | NVFP4 | AllowA4 |
| Attention HC、FFN HC down/up | FP8 | AllowA8 |
| Attention HC、FFN HC injection | BF16 | A16Only |
| Final HC mixer down/up | FP8 | AllowA8 |
| Pre-FC、HC、QSA、Indexer norm | BF16 | 对应算子合同 |

Indexer 和 final HC mixer 与 Main 统一使用 `AllowA8`，覆盖草拟及历史补齐中实际
执行的投影；小 M 仍可选择 A16。这两处调整不改变持久权重格式和第 7 节权重预算，
实际 workspace 由选择的 Op 路径单独核算。

MTP Indexer 误差可能改变选块，final mixer 误差可能改变候选 token。在正确的 target
验证与接受机制下，这些草稿变化主要影响接受率与加速收益。验收同时看平均接受长度、
每轮耗时和最终生成速度；Indexer 必须覆盖超过 512 个完整块的历史，不能只用全选的
短上下文判断精度。Main 的 Indexer 另按主模型检索质量验收。

### 5.1 Q4 optimized draft head

| 项目 | 配置 |
|---|---|
| 逻辑参数 | `proposal/head` |
| Shape | `[248320,5120]` |
| 词表域 | 完整词表，行号直接对应 token ID |
| 权重格式 | `q4_g64_fp16` |
| 存储布局 | `row_split_k128_v1` |
| 激活策略 | `A16Only` |
| 转换来源 | 原始高精度 LM head |
| 转换方法 | `grouped_absmax` |
| 转换选项 | `--proposal` |
| MTP 运行选项 | `--spec mtp --proposal-head optimized` |

Q4 head 用于 MTP 草拟，Main 验证和最终生成继续使用完整 FP8 LM head。
选择 `--proposal-head full` 时，MTP 使用共享 FP8 head，Q4 head 不加载到设备。
Draft token 数属于运行配置，与 MTP decoder 层数分别管理。

Q4 head 的 payload 包含 635,699,200 bytes codes 和 39,731,200 bytes FP16 scales，
合计 **675,430,400 bytes / 0.629044 GiB**，独立计入权重预算。

继续复用 Q4 实现。Vision 已使用 Q4，将 draft head 改为 NVFP4 不会消除这一格式，
还会使该 head 的 payload 增加约 37.9 MiB。两种编码的接受率和执行效率需要分别测量，
本方案不做该替换。

### 5.2 接受率不足时的候选调整

若正式权重上的测量表明 MTP 草拟质量限制加速，可评估仅将 MTP FFN 和 QSA O
从 NVFP4/AllowA4 升为 FP8/AllowA8。按本文 shape 估算，权重 payload 增加约
105 MiB，另计布局对齐和执行 workspace；是否采用取决于接受长度、每轮耗时和
最终生成速度的共同结果。

这是有条件的后续选择，不属于当前 recipe 或首批 Op 需求。它需要额外的 FP8 FFN
shape 支持，不能预先认定为实现简化或性能提升；采用时再更新权重预算。

## 6. 运行时表示

| 对象 | 表示 |
|---|---|
| Main K/V cache | FP8 E4M3 row-256，FP16 row scale |
| MTP K/V cache | FP8 E4M3 row-256，独立计费 |
| QSA compressed index keys | BF16 |
| QSA raw index tail | BF16 |
| GDN recurrent state | FP32 |
| GDN convolution history | BF16 |
| HC residual、公开 block 输入/输出 | BF16 |
| HC injection 系数 | FP32 |

输入矩阵行数 M 由执行批次、prefill chunk 或 verification 窗口决定。
上下文长度决定历史容量；它与投影权重存储位宽、矩阵激活精度分别管理。

## 7. 参数量与权重预算

### 7.1 主干参数量

| 模块 | 参数量 |
|---|---:|
| Embedding + LM head | 2,542,796,800 |
| FFN ×64 | 14,092,861,440 |
| GDN ×48 | 5,562,051,072 |
| Main attention ×16 | 1,677,729,792 |
| HC，含 final mixer | 3,394,785,280 |
| QSA Indexer ×16 | 52,432,896 |
| **合计** | **27,322,657,280** |

### 7.2 主干权重 payload

核算包含 codes、scales、NVFP4 weight divisors 及小型直接参数。
GiB 使用 `2^30 bytes`；文件目录、对象间对齐和 frontend 资源单独计费。

| 模块 | Bytes | GiB |
|---|---:|---:|
| FFN | 7,927,235,072 | 7.382813 |
| GDN 大投影 | 3,114,271,104 | 2.900391 |
| QSA 输入 FP8、输出 NVFP4 | 1,457,979,456 | 1.357849 |
| HC，含 injection 与 norm | 3,413,361,920 | 3.178941 |
| Embedding + LM head | 2,543,790,080 | 2.369089 |
| Indexer projection | 52,449,280 | 0.048847 |
| 其他小参数 | 51,173,376 | 0.047659 |
| **合计** | **18,560,260,288** | **17.285589** |

主干权重为 **17.286 GiB / 18.560 GB**。

### 7.3 全部启用权重预算

| 项目 | GiB |
|---|---:|
| 全部启用权重上限 | 18.976244 |
| Main 权重 | 17.285589 |
| Q4 optimized draft head | 0.629044 |
| Main + Q4 optimized draft head | 17.914633 |
| Vision、MTP 其余独有权重及其他权重增量的共同预算 | 1.061611 |

共享参数只计一次。权重预算包含 Main、启用的 Vision、MTP 及独立 proposal head。
KV、State、workspace、Graph 和媒体执行资源分别计费。

### 7.4 262144-token 上下文预算

Main 每个 token 的 16 层 FP8 K/V 存储量：

```text
16 × 4 × 2 × (256 + 2) = 33024 bytes
```

Indexer 平均每 token 的存储量：

```text
16 × 128 × 2 / 4 = 1024 bytes
```

| 项目 | GiB |
|---|---:|
| Main FP8 K/V，含 scale | 8.062500 |
| Main compressed index keys | 0.250000 |
| Main KV + index keys | 8.312500 |
| Main 权重 + 上述缓存 | 25.598089 |

每份完整 GDN recurrent state 为 144 MiB，卷积历史为 2.8125 MiB。
单个 MTP QSA 层的同长度 FP8 KV 与 index keys 为 0.519531 GiB。

与[现有 Qwen3.8 配置](../performance/qwen3.8-27b.md)按主干口径比较：

| 配置 | 主干权重 GiB | 主干权重 + 262K FP8 KV/index GiB |
|---|---:|---:|
| Qwen3.8-27B NVFP4 | 18.976244 | 27.038744 |
| 本方案 | 17.285589 | 25.598089 |
| Qwen3.8-27B groupwise-int | 15.920 | 23.983 |

## 8. 参考

- [架构与参数量分析来源：Intel XPU PR #22](https://github.com/analytics-zoo/vllm-xpu-kernels/pull/22)
- [Intel kernel shapes 与 HC/GDN/QSA 几何](https://github.com/analytics-zoo/vllm-xpu-kernels/blob/downstream/v0.31.0/custom-kernels-inventory.md)
- [Transformers GR/QSA 数学定义](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen4_exp/modular_qwen4_exp.py)
- [NInfer Qwen4Exp 数学与权重](../maintainer/qwen4_exp-model.md)
- [NInfer 权重转换](../weight-conversion.md)
- [现有 Qwen3.8 recipe](../../tools/convert/official_recipes.py)
