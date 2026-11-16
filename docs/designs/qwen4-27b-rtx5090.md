# Qwen4-27B：RTX 5090 预研假设与量化方案

本文保留基于候选几何制定的架构假设、基准权重方案与预算依据，目标为 RTX 5090 上的
262144-token 上下文、FP8 KV、Vision 和 MTP。它是独立模型适配的设计输入；
发布模型的配置与表示须由实际源权重确认，主线产品支持见 [README](../../README.md)。

共享的 HC/GDN/QSA 数学见 [Qwen4Exp](../maintainer/qwen4_exp-model.md)。同数学沿用同一架构
入口，Dense FFN 与可选 PLE 由模型定义、绑定和执行表达。Op 的支持范围与调优由代码维护。

## 架构假设

| 项目 | 配置 |
|---|---|
| Decoder | 64 层，hidden 5120，SwiGLU intermediate 14336 |
| 层分布 | 每四层 GDN/GDN/GDN/QSA，共 48 GDN、16 QSA |
| Vocabulary | 248320；embedding 与 Main head 不绑定 |
| HC | 4 streams，residual width 20480，lowrank 640 |
| HC 数量 | 每层 attention/FFN 各一组，另一个 final mixer |
| PLE | 不配置 |
| MTP | 一个独立 HC/QSA/Dense FFN 层 |
| RoPE | theta 10000000，rotary dimension 64 |
| MRoPE | interleaved，sections `[11,11,10]` |
| RMSNorm epsilon | `1e-6` |
| 主干参数量 | 27,322,657,280 |

```text
embedding / Vision → 四流 residual
  → attention HC → GDN 或 QSA → HC combine
  → FFN HC → Dense SwiGLU → HC combine
  → final HC mixer → Main head
```

MTP conditioning 使用 final mixer 前的四流 residual。HC streams 是特征轴；
MTP `fc_hidden` 对每条 stream 使用同一个 `[5120,5120]` 权重，因此它的计算列数为 4M。

| 子层 | 几何 |
|---|---|
| GDN | Q/K heads 16，V heads 48，head dimension 128 |
| GDN convolution | 10240 channels，4 taps；权重为 tap 外、channel 连续 |
| GDN recurrent State | 每层 `[48,128,128]` FP32 |
| QSA | 24 query heads、4 KV heads、head dimension 256，GQA ratio 6 |
| Indexer | 4 query heads、1 KV head、dimension 128 |
| Indexer selection | 每 4 token 一个块，最多选 512 个完整块，另读可见 tail |
| Vision | 27 层，hidden 1152，16 heads，head dimension 72，intermediate 4304 |
| Vision 输入与 merger | patch `[3,2,16,16]`，2×2 merge，`4608 → 4608 → 5120` |

## 基准权重方案

Shape 按 `[输出行, 输入特征]` 表示。FP8 指 `fp8_e4m3fn_row_bf16`；NVFP4 指 `nvfp4`。
格式、activation permission 与物理布局的分工见[权重与加载](../maintainer/weights-and-loading.md)。

| Main 模块 | 权重 shape / 参数 | 表示与计算许可 |
|---|---|---|
| Embedding | `[248320,5120]` | FP8 查表 |
| Main head | `[248320,5120]` | FP8 / A16Only |
| FFN gate/up | 合并 `[28672,5120]` | NVFP4 / AllowA4 |
| FFN down | `[5120,14336]` | NVFP4 / AllowA4 |
| GDN QKVZ | 合并 `[16384,5120]`，输出 QKV/Z 两段 | NVFP4 / AllowA4 |
| GDN output | `[5120,6144]` | NVFP4 / AllowA4 |
| GDN A/B | 合并 `[96,5120]` | BF16 / A16Only，输出 FP32 controls |
| GDN conv / gated norm | `[10240,4]` / `[128]` | BF16 |
| GDN A_log / dt_bias | 各 `[48]` | FP32 |
| QSA 完整输入 | 合并 `[14976,5120]` | FP8 / AllowA8 |
| QSA output | `[5120,6144]` | NVFP4 / AllowA4 |
| QSA / Indexer norm | `[256]` / `[128]` | BF16 |
| HC down / up | `[640,20480]` / `[20480,640]` | FP8 / AllowA8 |
| HC injection | `[4,20480]` | BF16 / A16Only，输出 FP32 系数 |
| HC norm | `[20480]` | BF16 offset words |

Final HC mixer 使用相同 down/up/norm 表示，不包含 injection。小型控制参数保留较高精度：
Main HC injection 共 20 MiB，GDN A/B 共 45 MiB，降为 FP8 的容量收益分别只有约 10 MiB、
22.5 MiB，而它们控制整条 residual 或递推 State。Main head 保留 A16 activation 许可，
HC 与 Indexer 的 FP8 投影允许 A8；实际执行路径归 Op。

NVFP4 使用具有对应 activation 校准数据的源编码。编码、合并与加载保留源 scale 语义，
weight scale 与每个 Use 的 activation scale 分别绑定。

## 合并与视图

QSA 的完整 FP8 parent 按下列行序保存，Main 与 MTP 使用相同布局：

| 角色 | 行区间，半开 |
|---|---|
| K | `[0,1024)` |
| V | `[1024,2048)` |
| index-K | `[2048,2176)` |
| Q | `[2176,8320)` |
| output gate | `[8320,14464)` |
| index-Q | `[14464,14976)` |

完整计算通过一次六段 LinearSplit 产生连续输出；MTP history-only 使用前缀 `[2176,5120]`
一次产生 K/V/index-K。两种 view 共享不可变 backing，scale plane 位置由完整 parent 决定。
转换时解交错 Q/gate 并重排 Indexer 行，codes 与各自 row scales 一起移动。

FFN GU 行序为 `[gate;up]`；GDN QKVZ 行序与卷积 channel 顺序一致，Z 不经过卷积。
MTP stem 的 `fc_embedding` 与 `fc_hidden` 是两次不同输入的 `[5120,5120]` 投影。

## MTP 与 Vision

MTP 共享 Main embedding 与 full head；QSA、FFN、HC 使用与 Main 同构部分相同的表示和许可。
Stem 的 `fc_embedding`、`fc_hidden` 使用 FP8/AllowA8，norm 使用 BF16。

额外的 optimized draft head 为完整词表 `[248320,5120]`、`q4_g64_fp16`、A16Only，
由高精度源 head 通过 `grouped_absmax` 生成。其 payload 为 675,430,400 bytes，即 0.629044 GiB。
`--proposal-head full` 共享 Main FP8 head；`optimized` 使用该 Q4 对象。

若调整 MTP FFN 与 QSA O 为 FP8/AllowA8，按此几何增加约 105 MiB payload；
这一候选的取舍由实际接受长度与 step 耗时共同决定。

Vision 复用公共 Qwen 实现，权重分配为：

| 参数 | 表示 |
|---|---|
| Patch embedding | Q6 |
| Attention Q/K/V / output | Q4 / Q5 |
| MLP fc1 / fc2 | Q4 / Q5 |
| Merger fc1/fc2 | Q8 |
| Position embedding、norm、bias | BF16 |

Vision 的量化投影均为 A16Only。矩阵和卷积布局在转换与加载时准备，执行期直接消费。

## 容量依据

以下为候选几何和基准配方的 payload 计算。GiB 使用 `2^30 bytes`；
文件元数据、对象对齐、执行资源与 Context 池布局另行计费。

| Main 权重模块 | Bytes | GiB |
|---|---:|---:|
| FFN | 7,927,235,072 | 7.382813 |
| GDN 大投影 | 3,114,271,104 | 2.900391 |
| QSA 主投影，不含 Indexer | 1,457,979,456 | 1.357849 |
| HC，含 injection 与 norm | 3,413,361,920 | 3.178941 |
| Embedding + Main head | 2,543,790,080 | 2.369089 |
| Indexer projection | 52,449,280 | 0.048847 |
| 其余小参数 | 51,173,376 | 0.047659 |
| **Main 合计** | **18,560,260,288** | **17.285589** |

Main 加 Q4 head 为 17.914633 GiB。原定全部启用权重目标为 18.976244 GiB，
为 Vision、MTP 其余独有权重及其他增量留下 1.061611 GiB；共享参数只计一次。

Main 与 MTP 的 FP8 KV 使用当前 E4M3FN/G128/UE8M0 表示；compressed index keys 与 raw tail
使用 BF16。完整缓存格式由 [KV codec](../../include/ninfer/kv_cache_format.h)维护。
每个 256 维 FP8 向量有 256 bytes codes 与 2 bytes scales，因此 Main 的每 token 历史为：

```text
K/V:   16 layers × 4 heads × 2 × (256 + 256/128) = 33024 bytes
Index: 16 layers × 128 × 2 / 4                    =  1024 bytes
```

| 262144-token 历史 | GiB |
|---|---:|
| Main K/V | 8.062500 |
| Main compressed index keys | 0.250000 |
| Main 历史合计 | 8.312500 |
| Main 权重 + Main 历史 | 25.598089 |
| 单个 MTP QSA 层的 K/V + index keys | 0.519531 |

每份完整 GDN recurrent State 为 144 MiB，卷积历史为 2.8125 MiB，其他 State 字段另计。
KV 与 State 在统一 Context backing 内按需分配；并发、checkpoint、spec records、workspace、
Graph 与 Vision 窗口的峰值由模型规划组合，不从上述单份历史 payload 推导启动容量。

## 分析来源

- [Intel XPU PR #22](https://github.com/analytics-zoo/vllm-xpu-kernels/pull/22)
- [Intel kernel shapes 与 HC/GDN/QSA 几何](https://github.com/analytics-zoo/vllm-xpu-kernels/blob/downstream/v0.31.0/custom-kernels-inventory.md)
- [Transformers GR/QSA 数学](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen4_exp/modular_qwen4_exp.py)
- [权重转换](../weight-conversion.md)与 [Qwen4Exp](../maintainer/qwen4_exp-model.md)
