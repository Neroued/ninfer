# GLM5Next FP4-Local：Op 支持与优化清单

盘点基点：`d49b9af7`。对应已生成的 `glm5_3_flash_fp4_local.ninfer`。
原生格式、转换、压缩 TP/EP 物化与真实 rank 权重加载已完成；本文规定接下来需要交付的计算能力。

目标是在源量化数值、CSF 常驻容量和压缩访存优势完整保留的前提下，完成 Main、MTP、Vision
实际调用的全部 Op，并优化完整调用。实现步骤服务于同一套最终合同，不设展开权重的过渡执行路径。

开发方法沿用 [Op 规范](../maintainer/op-development.md)；本清单只补充本任务的几何、接口、边界和性能要求。

## 实施状态（2026-11-04）

格式基础、27 组纯 linear、融合投影、完整 FFN/MoE、KDA input/conv、mHC、Kpool Prepare/factors
已实现，并完成独立数学、Graph 与逐点性能检查。模型的 typed weight、calibration、workspace
调用接口已同步；新产物的整模型 Main/MTP/Vision、TP/EP、eager/Graph 运行验收仍未关闭。

勾选表示对应功能和检查已完成，不表示全部性能目标已经达到。Main CSF MoE 小 M 的完整 routed
入口仍约 191～195 GB/s，部分小矩阵、融合消费者和大 M 的计算吞吐也低于本清单参照。
完整曲线与被拒绝的候选均已保留；E3 的性能目标继续保持未完成。

## 1. 部署范围与计量轴

- 正式优化目标：DGX Spark / SM121，双机 `TP2 + ADP2`，expert 同时支持 `EP1`（expert TP）与 `EP2`。
- 5090 保留现有能力与构建；新增 Spark shape/tuning 不进入 5090 编译。本任务不要求在单台 Spark 装入完整 GLM。
- 纯 linear 矩阵写为 `[N,K]`，数学为 `Y = W X`；实际 Tensor 为 BF16 `[K,M] → [N,M]`，feature 连续。
- `B` 是物理请求行数，`T` 是每行物理 token 数，`M=T×B`。KDA/FFN 使用全局 batch；MLA 使用 owner-local 物理 batch。
- ADP 切请求，不切 MLA heads：每个 attention owner 仍计算 64 个 MLA heads 和 32 个 index heads。
- 当前 `B=1..8`。调优用的 draft 宽度不是接口上限；合法正 `T` 和 capacity 区间仍按所属 Op 合同处理。
- Vision 用 `P` 表示 patch 列数、`F=P/4` 表示 merge 后 feature 列数；整塔不按 TP 切矩阵。

| 场景 | 主要调用形状 |
|---|---|
| Ordinary decode | 全局 `T=1`；ADP 本地行数及 validity 可不均衡 |
| Main speculative verify | `T=draft+1`，全局 `M=B×T`；每行有效长度可以不同 |
| MTP proposal | 每个 proposal step 通常 `T=1`，支持 full/optimized head |
| MTP history repair | `T` 随修复长度变化，调用 History prepare；不能仅优化 proposal |
| Prefill/replay | 典型 chunk 512、1024、2048、4096，包括短尾块 |
| Vision | 编码器处理 P 列，downsample/merger 处理 F 列；等长和 ragged attention segments |

## 2. 优化目标与测量要求

### 2.1 完整入口的 roofline

使用 [bench/common/roofline.h](../../bench/common/roofline.h) 的统一参照，不在各 bench 另写常数。
当前 Spark 参照为：

| 项目 | 参照 |
|---|---:|
| 标称内存带宽 | 273.0 GB/s |
| 已标定持续读取带宽 | 257.42 GB/s |
| BF16 / FP32 accumulate Tensor Core | 125.89 TFLOP/s |
| MXFP8 / FP32 accumulate Tensor Core | 249.52 TFLOP/s |
| NVFP4 / FP32 accumulate Tensor Core | 499.02 TFLOP/s |

对足以形成流式读取、且权重带宽主导的区间，**235 GB/s 是已有实现水准，优化目标是逼近
250～257.42 GB/s，而不是达到 235 后停止**。CSF 按实际压缩流量计算，包含索引与 exceptions。
小数据、缓存命中、launch 或指令吞吐主导的点，按对应下界分析；不能机械套用 DRAM 带宽阈值。

完整路线至少报告：

```text
t_mem = necessary_bytes / bandwidth_reference
t_compute = useful_FLOPs / matching_compute_reference
t_roof = max(t_mem, t_compute)
```

同时报告实际延迟、必要字节数、有效 GB/s、useful TFLOP/s 和距下界的绝对差距。
launch、格式恢复、同步、padding 和重复读取是需要继续削减的成本；不能靠把这些成本加入
“必要工作量”来抬高利用率。SIMT 路线不填 Tensor Core 利用率，A16 路线不使用 NVFP4 峰值作分母。
阶段测量可以用于归因，验收以完整公共 Op 为准。

用于校核 fixture 流量的几个权重读取参考如下。这里只包含一次权重读取，正式 Op 还要计输入、输出、
校准、State 和必要 scratch；这些数值不是完整调用的固定延迟门槛。

| 几何/格式 | 权重 MiB | 257.42 GB/s 读取时间 | 235 GB/s 读取时间 |
|---|---:|---:|---:|
| MXFP8 KDA QKV `[12288,4096]` | 49.500 | 201.63 μs | 220.87 μs |
| MXFP8 KDA O `[4096,4096]` | 16.500 | 67.21 μs | 73.62 μs |
| MXFP8 local head `[77440,4096]` | 311.953 | 1270.71 μs | 1391.94 μs |
| Main Full W_A 两个 parent 合计 | 10.252 | 41.76 μs | 45.74 μs |
| Main W_B 两个 parent 合计 | 36.006 | 146.67 μs | 160.66 μs |
| MXFP8 MTP W_B `[20480,1536]` | 30.938 | 126.02 μs | 138.04 μs |

### 2.2 调优覆盖

- 纯 linear、split、SwiGLU、完整 FFN/MoE：`M=1..256` 逐点测量与选型；512、1024、2048、4096
  是明确的大 M 调优点。shape policy 可以覆盖连续区间，不能把逐点测量变成逐点模板实例膨胀。
- 小 M 重点同时包含 1、2、4、8、16、24、32、48、64。需要解释耗时增长来自更多有效计算、
  expert 去重集合变化，还是重复读权重、tile padding、量化、调度和 launch。
- KDA input/record：区分相同 M 的不同 `(B,T)`；覆盖 `B=1,2,4,8`、`T=1,2,3,4,6,8,16`，
  Prefill 覆盖上述 chunk 与尾块。不能把所有形状当无序二维 GEMM 测完即结束。
- Kpool：覆盖本地 B、T、混合 validity、pool phase；factor/attention 测量上下文
  2048 附近、8K、64K、1M 的代表点。按问题选择组合，不穷举所有轴的笛卡尔积。
- Vision 的纯 contraction 同样扫描 1..256 与大 M 点；实际融合链补 P/F 关系及运行时预算允许的更大窗口。
- 保留 warm 与流式/cold 条件，说明缓存复用。小矩阵反复驻留 L2 的吞吐不能当作 DRAM 带宽。
- 全部关键点给出真实曲线。性能未达到目标时记录瓶颈与剩余差距，不能只展示 4096 或最快点。
  重复次数用于辨别噪声，不因无显著差异反复开展整模测试。

### 2.3 数学与性能共同要求

- 使用源 represented weights 的独立数学参考；codec 恢复按字节验证。测试精度只由公开语义/许可决定。
- `AllowA8/AllowA4` 不强迫每个 M 使用低精度 MMA；A16/SIMT/MMA 的选择由完整成本决定。
- 禁止持久或逐层全局展开 CSF scales、BF16 expert/factor 副本；只允许消费 tile 内的片上恢复。
- 不能为套现有模板而改变 multiplier、窄化 FP32 scale、重新量化权重，或增加未约定的中间 BF16 舍入。
- 低精度输入量化、路由/分组、CSF 恢复、reduce/join 都计入完整 Op 延迟与 workspace。

## 3. 格式接入和代码组织

### 已确定的表示合同

`n,k` 是矩阵行列，`e` 是 expert；`decode` 按对应浮点编码解释原字。

| 格式 | represented weight | 物理表示 |
|---|---|---|
| `NVFP4_MUL` | `decode(E2M1[n,k]) × decode(E4M3[n,k/16]) × multiplier[e]` | 两个 FP4 code/byte；普通 scale 或 CSF；每矩阵原始 FP32 multiplier |
| `MXFP8` | `decode(E4M3[n,k]) × 2^(UE8M0[n,k/32]−127)` | `RowGroupScale`；code plane 与每行 K32 scale plane |
| `FP8_E4M3FN_BLOCK128X128_FP32` | `decode(E4M3[n,k]) × FP32_scale[n/128,k/128]` | `BlockScaleM128K128`；原始 FP32 block scale |

CSF 是 E4M3 scale **字节的无损压缩**：每行 base 加 packed nibble，局部 exception 覆盖原字；
code plane 不变。`bases/nibbles/offsets/page_bases` 按物理矩阵步进，exceptions 是整个 bank 共享的变长区域。
GPU 直接借用 `Weight.csf` 或 `ExpertMatrixDeviceView.csf`；子矩阵保留原 geometry 和 `row_begin`。
几何、对齐与寻址权威仍在 Core，Op 不按一个固定 bytes/expert 猜测整个压缩布局。

使用 [Weight](../../src/core/weight.h)、[expert views](../../src/core/weight_bank.h) 和
[calibration view](../../include/ninfer/ops/weight_input.h) 已有的类型；独立精确参考在
[CSF codec](../../tools/artifact/codecs/csf32.py)。格式名称不承担 activation policy 的含义。

### F0：公共表示与 device primitive

- [x] `NVFP4_MUL` 按 `E2M1 × E4M3 × FP32 multiplier` 消费；普通 scale 与 `Csf32` 共用数值合同。
  旧 `NVFP4` 的 divisor 算术保持独立，不能把新类型传进原除法 kernel。
- [x] CSF reader 原生消费 `Csf32ScaleView/Csf32BankDeviceView`。按 tile 定位 page/offset，合作恢复 base+nibble，
  再覆盖局部 exceptions；完整支持稀疏与密集异常块。异常处理不扫描整 expert，也不依赖全局解压 scratch。
- [x] 新 A4 quantizer 原生消费 `ActivationScaleOperation::Multiply`，恢复值使用原始 activation multiplier。
  不能直接调用旧 divisor quantizer，也不在加载或执行时把 multiplier 改成持久 reciprocal。
  GU 的 BF16 输入与 down 的 FP32 product 都支持 G16 量化，包含 scale 舍入、饱和和零组的完整语义。
- [x] MXFP8 按 E4M3＋沿 K 的 G32 UE8M0 消费。A8 输入量化也采用对应 G32/UE8M0，正确处理 zero、舍入和饱和。
  复用现有 `fp8_ue8m0_scale` 与 `mma_fp8_e4m3_scaled` 等基础，不能用 row-FP8 或 KV 的 G128 分组代替。
- [x] FP32-scale block FP8 保留 `[N/128,K/128]` scale 原字。A16 可用 E4M3 code 精确提升后的 BF16 MMA
  计算 K128 partial，再乘原 FP32 scale 并跨块累加；不要先把 scale 或完整反量化权重降成 BF16。
- [x] 所有 operand 检查、profile、capacity、fixture 和流量模型识别新 format/layout，检查由对应实现者负责。

实现落点：

```text
src/ops/common/quant/
  fp8.cuh                 共享 E4M3/UE8M0 数学；扩展实际共用的 MXFP8 基础
  nvfp4.{h,cuh}           FP4 解释与明确的 multiplier/divisor 算术
  csf32.cuh               合作式 device scale reader，供 linear/MoE 共用
src/ops/common/cuda/mma.cuh
                            共享 MMA 指令与 fragment 基础
src/ops/linear/kernels/
  fp8/native_*            MXFP8/block-FP8 共用的 A16 contraction、schedule 与输出接口
  mxfp8/                  原生 A8 contraction、输入 pack/quantize 与 MXFP8 operands
  fp8_block/              block scales 的 A16 contraction，scale dtype 明确
  nvfp4/                  普通/CSF loader 与乘法/除法 specialization
src/ops/linear/{projection,split,bias,swiglu}/
  <format-family>/dgx_spark/shapes/<n..._k...>/...
src/ops/moe/
  kernels/cuda/...         共享 expert contraction/压缩读取/分组基础
  swiglu/...               完整单 FFN 与 shared 分支
  routed_swiglu/...        routed 数学入口与独立 shape policy
  topk_sigmoid_swiglu_moe/... 完整路由、routed、shared、join 入口与独立 policy
```

目录按已有 family 语义组织：新增 `mxfp8`、`fp8_block` 分支；NVFP4 两种 scalar 数学和 CSF loader
留在 NVFP4 family。单纯共享的 helper 放现有共享文件，有实质规模的 CSF reader 独立成文件。
不创建 `fp4_local_model` 一类模型专属 Op 目录，不将 launch/shape policy 放进模型代码。

projection、split、bias、SwiGLU 各自拥有调优记录，contraction 模板共享。
KDA input、Kpool prepare/factor、mHC 留在自己的 family，并复用真正共同的 primitive。
`dgx_spark` 私有源码通过现有 `sources.cmake` 注册；不通过运行期 target 字符串分派。

### F1：成对投影保留每个 parent 的许可

- [x] `PairedProjectionWeights` 改为保存两个 `SingleProjectionWeight` 或等价的独立 weight＋policy。
- [x] `describe_projection_weights`、`projection_profile`、split 的 capacity/dispatch 不再把 pair 强制改为 A16。
- [x] Main 的 block-FP8＋BF16 都是 A16；MTP W_A 的 MXFP8 parent 允许 A8，BF16 pool-gate 仍为 A16。
  每个 parent 使用自己的输入表示，不能整体降低前者的许可，也不能扩大后者的许可。
- [x] 更新既有 Q4/Q5 paired 调用和测试，保留其本来 A16 的语义，删除被替代的内部表示。

入口：[projection_weights.h](../../include/ninfer/ops/projection_weights.h)、
[weight_input.cpp](../../src/ops/weight_input.cpp)。

## 4. 先完成纯 linear

以下是**双机实际使用的本地矩阵**，所有条目要求正 M 域及第 2 节的调优点。
源 KDA QKV `[24576,4096]`、head `[154880,4096]` 等已在加载期切片，本清单不将其误作本地形状。
MLA 和 Vision 则保留其完整 head/hidden 几何。Expert 的矩阵在第 8 节处理，不以逐 expert 循环调用 linear 实现。

### L1：MXFP8，AllowA8

| 状态 | `[N,K]` | 用途 |
|---|---|---|
| [x] | `[12288,4096]` | KDA QKV；Main dense GU |
| [x] | `[4096,4096]` | KDA O |
| [x] | `[4096,6144]` | Main dense down |
| [x] | `[77440,4096]` | Main head、full MTP head；包括 PPL 的批量 logits |
| [x] | `[2048,4096]` | MTP shared GU |
| [x] | `[4096,1024]` | MTP shared down |
| [x] | `[2208,4096]` | MTP W_A Full |
| [x] | `[2176,4096]` | MTP W_A QueryAndHistory |
| [x] | `[640,4096]` | MTP W_A History |
| [x] | `[20480,1536]` | MTP W_B：Q_b＋indexQ |
| [x] | `[16384,1536]` | MTP query-only |
| [x] | `[4096,16384]` | MTP MLA O |
| [x] | `[3072,1024]` | Vision QKV |
| [x] | `[1024,1024]` | Vision O |

必须具备完整 A16 与原生 A8 路线的选型空间；A8 的量化成本计入入口。
小 M 要充分读取权重并减少重复加载，大 M 要使用正确的 block-scaled Tensor Core 和足够深的流水线。

### L2：block FP8 128×128 / FP32 scales，A16Only

| 状态 | `[N,K]` | 用途 |
|---|---|---|
| [x] | `[2048,4096]` | Main W_A `[latent,Q_a]`；Main shared GU |
| [x] | `[512,4096]` | Main latent-only prefix |
| [x] | `[16384,1536]` | Main Q_b |
| [x] | `[4096,16384]` | Main MLA O |
| [x] | `[4096,1024]` | Main shared down |

同 shape 的 contraction 可以共享；不同融合消费者不共用私有调优结果。

### L3：普通 scales 的 NVFP4_MUL，A16Only

| 状态 | `[N,K]` | 用途 |
|---|---|---|
| [x] | `[8192,1024]` | Vision FFN GU |
| [x] | `[1024,4096]` | Vision FFN down |
| [x] | `[4096,4096]` | Vision merger projection |
| [x] | `[20480,4096]` | Vision merger GU |
| [x] | `[4096,10240]` | Vision merger down |

**A16 必须覆盖大 M。** 现有某些 NVFP4 A16 路线的 `T<=16` 限制不能带进这些新需求。
大 M 在片上恢复权重后使用适合的 BF16/FP16 MMA；global multiplier 在正确的数值位置应用。

### L4：parent 拆分产生的新 BF16 几何

| 状态 | `[N,K]` | 用途 |
|---|---|---|
| [x] | `[256,4096]` | Main W_A History/QueryAndHistory 的 indexK＋gate |
| [x] | `[128,4096]` | MTP pool gate |
| [x] | `[4096,1536]` | Main indexQ |

当前 BF16 registry 缺少这三项。`[288,4096]` 已有，用于 Main W_A 完整 BF16 parent、KDA control 和 router。

## 5. Linear 融合消费者与完整 FFN

### P1：linear_split

| 状态 | 输入权重 | 连续输出行数 | 附加要求 |
|---|---|---|---|
| [x] | Main W_B：block-FP8 `[16384,1536]`＋BF16 `[4096,1536]` | `[16384,4096]` | 两个 parent，均 A16 |
| [x] | MTP W_B：MXFP8 `[20480,1536]` | `[16384,4096]` | AllowA8 |
| [x] | Vision QKV：MXFP8 `[3072,1024]` | `[1024,1024,1024]` | BF16 bias `[3072]`，AllowA8 |

- [x] bias overload 接入实际 policy 和 caller-owned workspace/capacity；bias 在最终输出舍入之前相加。
- [x] paired contraction 在同一 Op 内调度，消除额外输出 split/copy；不通过多次小公共 linear 串联实现融合入口。
- [x] KDA control 的既有 BF16 `[288,4096] → [32,128,128]` 保持。

### P2：linear_bias、linear_swiglu

| 状态 | 入口 | 表示/shape | 数学 |
|---|---|---|---|
| [x] | `linear_bias` | MXFP8 `[1024,1024]` | Vision O＋BF16 bias |
| [x] | `linear_bias` | NVFP4_MUL `[1024,4096]` | Vision down＋BF16 bias |
| [x] | `linear_swiglu` | MXFP8 `[12288,4096]` | Main dense GU |
| [x] | `linear_swiglu` | block-FP8/FP32 `[2048,4096]` | Main shared GU |
| [x] | `linear_swiglu` | MXFP8 `[2048,4096]` | MTP shared GU |
| [x] | `linear_swiglu`＋bias | NVFP4_MUL `[8192,1024]` | Vision FFN |
| [x] | `linear_swiglu` | NVFP4_MUL `[20480,4096]` | Vision merger |

SwiGLU 都使用实际配置的 clamp，当前 checkpoint 为 10：
`SiLU(min(g,limit))*clamp(u,-limit,limit)`。有 bias 时先形成 g/u；projection、bias 到 product
之间不增加 BF16 舍入。带 bias 的现有入口没有 policy/workspace，需与无 bias 的完整表示能力对齐。

### P3：完整 swiglu

| 状态 | H / local I | GU / down | 许可 |
|---|---|---|---|
| [x] | `4096 / 6144` | MXFP8 / MXFP8 | AllowA8；Main 前三层 dense |
| [x] | `4096 / 1024` | block-FP8 FP32 / 同格式 | A16Only；Main shared |
| [x] | `4096 / 1024` | MXFP8 / MXFP8 | AllowA8；MTP shared |

模型调用的是 `ops::swiglu` 和完整 MoE，单独实现 `linear_swiglu` 不足以完成这些路径。
完整 FFN 的 GU→clamp/product→down 使用私有高精度/量化流水线，保留其公开中间语义，
不能通过两个带额外 BF16 边界的公共 Op 冒充完整融合实现。

## 6. KDA 与 mHC

### K1：KDA input projection＋conv update/record

- [x] 扩展 `kda_input_proj_conv_update` 与 `kda_input_proj_conv_record`：
  MXFP8 `[12288,4096]`、AllowA8、BF16 四 tap conv `[12288,4]`（Tensor 轴）。
- [x] 输出 Q/K/V 各 `[4096,T,B]`，相当于 TP2 的 32 heads×128；history 为 BF16 `[12288,3]`，
  records 为 BF16 `[12288,T,B]`。源/目标 State 的 stride 和 offsets 继续由现有视图表达。
- [x] Projection→conv 不增加 BF16 seam。conv 实际读取源 BF16 权重，累加、SiLU 与最终发布边界按既有合同。
- [x] A8 路线直接把 contraction 的 FP32 结果交给当前 conv；`conv_record` 保存同次投影按合同舍入的 BF16 列，
  用于接受前缀的历史恢复，不把 record 的 BF16 舍入插回当前卷积计算。
- [x] `T=1,2,3`、update 原地/分离、record source只读、每行不同有效长度和跨私有 tile 的行为完整。
- [x] profile、capacity、Arguments 的卷积权重 dtype 与所有 vector/MMA/TMA 路线同步接入。

落点：[input_conv](../../src/ops/linear_attention/kimi_delta_attention/input_conv/)、
[kda_input_proj.h](../../include/ninfer/ops/kda_input_proj.h)。保留现有 Q8＋FP32-conv 路径。

KDA recurrence/chunked、batch update、ReplaySSM record/fold、sigmoid-gated RMSNorm 已满足输入/状态格式。
本轮不重写这些算子；使用新 producer 的 records 验证 accepted `0..W`、混合 acceptance 和同次重放。
KDA O 由 L1 提供，control split 与两路 BF16 `[4096,128]` 的 `linear_pair` 已有。

### H1：mHC 的 FP32 小参数

固定数学几何 `S=4,D=4096`，projection BF16 `[24,16384]`，计算列数 M 不作 TP 切分。

| 参数 | Native dtype |
|---|---|
| 所有 Main base `[24]`、scale `[3]` | FP32 |
| attention-pre gamma `[4096]` | BF16 |
| FFN-pre gamma `[4096]` | FP32 |
| final gamma `[4096]` | FP32 |

- [x] `mhc_pre`、`mhc_expand_pre`、`mhc_post_pre`、`mhc_post_mean_norm` 均支持这些 dtype 组合。
- [x] 覆盖 vector、MMA、partition、cluster、finish 等已存在的分派；直接 typed load，不生成窄化参数副本/cast kernel。
- [x] 保留 coefficient/collapse/Sinkhorn 的 FP32 计算及 residual 的既定 BF16 写回边界。
- [x] 沿用 post/pre、post/mean/norm 融合，优化 M=1..256 和大 M 的实际读写、launch 与并行度。

`mhc_post` 不读取上述权重，保持现有实现。当前 Main 的 FP32 gamma 均由 mHC 消费，
本任务不因此扩大所有 standalone RMSNorm 接口的 dtype 范围。

## 7. Kpool Sparse MLA

### A1：带表示与行域映射的 Prepare

扩展 `KpoolMlaPrepareParameters` 与 `KpoolMlaPrepareProfile`，使其持有 typed `ProjectionWeights/Profile`，
以及 latent/indexK/pool-gate/Q_a/coeff 对应的 `(parent,row_begin)`。角色长度来自现有语义维度，
不重复存一套 shape。映射由模型根据已绑定行域提供，Op 验证覆盖及自己支持的表示。

| 模式 | Main parent 0 / parent 1 | MTP parent 0 / parent 1 |
|---|---|---|
| Full | block-FP8 `[2048,4096]`：latent512,Q_a1536；BF16 `[288,4096]`：indexK128,gate128,coeff32 | MXFP8 `[2208,4096]`：latent512,indexK128,Q_a1536,coeff32；BF16 gate `[128,4096]` |
| QueryAndHistory | parent 0 完整；parent 1 前256行 | parent 0 前2176行；gate完整 |
| History | parent 0 前512行；parent 1 前256行 | parent 0 前640行；gate完整 |

- [x] 六组投影均支持 `kpool_mla_prepare_record/update`，保留每 parent 独立 policy。
- [x] 全部支持 latent BF16/FP8 × index BF16/FP8；权重表示不决定 KV 类型。
- [x] 保留 FP32 projection→norm→latent encode/pooling 的融合边界，不能先调用输出 BF16 的普通 linear 再接 norm。
- [x] raw K/G、tail、records 保持 BF16；FP8 index 的 Hadamard/pooling/encode、FP32 signed coefficients 数学不变。
- [x] 覆盖混合 valid、全 inert ADP 行、pool phase 0..3、跨页、update alias、record suffix 保持。
- [x] capacity 按真实两个 parent、policy 与状态模式计算；联合调度压缩 projection、norm 和写回，避免按五个角色拆成小投影。

落点：[prepare](../../src/ops/softmax_attention/kpool_mla/prepare/)、
[kpool_mla.h](../../include/ninfer/ops/kpool_mla.h)。旧单 BF16 parent 通过相同语义映射保留支持。

### A2：原生 MXFP8 factors 的 sparse MLA

Main factors 仍 BF16；MTP 的 U_K/U_V 都为 MXFP8。逻辑形状分别是 `[64,256,512]`，scale 沿末维每32元素一组。

- [x] `KpoolMlaFactors` 从固定 BF16 Tensor 扩为明确的编码 Weight view；Profile 描述 factor 表示和几何。
- [x] **U_K 是转置收缩：**`q_latent = U_Kᵀ q`，归约轴为256；原 scale 轴是 latent512。
  转置后是 N32×K1 的 scale 域，不能伪装成普通 K32 MXFP8 GEMM。
  专用 reader 按原域片上恢复/缩放，不能重新量化或在显存常驻完整 BF16 factors。
- [x] U_V 为正常的 `U_V u`，归约轴为 latent512，按原 G32 域读取。
- [x] factor 收缩保持现有 BF16/F32 数值合同；输入量化许可不从权重 QType 自动推导。
  BF16/FP8 cache 的 QK/PV、FP32 softmax/merge 和最终 BF16 输出合同保持。
- [x] 覆盖 small/direct、分块/split、merge/value 等完整路线；按 M、选择规模和 cache 类型选择实现。
- [x] 允许同一 CTA 内因子/latent 中间值复用，减少 q_latent/u 的全局读写，但不改变可观察数值边界。
- [x] workspace 与 Graph profile/capture/update 都包含 factor 表示；固定物理形状下各上下文仍满足节点更新合同。

Scope 包含新旧 factor × 两种 latent cache、selection reuse、空选择/全 inert 行、尾部不完整 pool、
不同历史长度及不同 selection row 映射。Kpool indexer 的 BF16/FP8 数学实现已有，保持并验证顺序消费。

## 8. Routed experts 与完整 MoE

几何 `H=4096,E_global=288,topk=8`；sigmoid router＋FP32 correction bias；当前 routed scaling factor=2.5，
clamp=10。输出保持现有 BF16 local contribution，跨 rank collective 由模型组织。

| 状态 | 范围 | 物理 bank | 表示/许可 |
|---|---|---|---|
| [x] | Main expert TP2 | GU `[288,2048,4096]`，down `[288,4096,1024]` | NVFP4_MUL＋CSF，AllowA4 |
| [x] | Main expert EP2 | GU `[144,4096,4096]`，down `[144,4096,2048]` | NVFP4_MUL＋CSF，AllowA4 |
| [x] | MTP expert TP2 | 同 TP2 几何 | 普通 scale 的 NVFP4_MUL，A16Only |
| [x] | MTP expert EP2 | 同 EP2 几何 | 普通 scale 的 NVFP4_MUL，A16Only |

### E1：routed_swiglu

- [x] 四组均完成 `routed_swiglu` 的实际生产路径；保留现有 coefficient、Write/Accumulate 和合法 subset 语义。
- [x] CSF 在消费 tile 内解码；普通/压缩 scale 通过明确 loader specialization 复用 contraction。
- [x] TP 使用所有逻辑 expert；CSF EP 先查 `expert_indices[global_id]` 得到物理 slot。
  权重 multiplier 按物理位置取，激活 calibration 按 **global ID** 取，不能混用。
- [x] 普通 MTP EP bank 保持现有 indexed offset 寻址，不能只实现 compact CSF 的一种布局。
- [x] A4 GU 输入按 `(token,expert)` 使用真实 calibration；同 expert 的 gate/up 共用这份量化。
  不使用层内首值、max 或平均值替代校准。Down 使用自己的 per-expert multiplier。
- [x] 尽可能融合 gather/quantize、GU/clamp/product/quantize；同一 tile 的权重解码结果跨 token 复用。
- [x] Main A4 路线直接使用 NVFP4 block-scaled Tensor Core：片上恢复 E4M3 scales 并形成 MMA scale fragment，
  与压缩 code 载入流水重叠；CSF 不能成为退回全部 A16 计算或增加全局 scale pass 的理由。
  小 M 的 A16/A4 选择仍由完整调用成本决定，两种路线都读取同一压缩 bank。
- [x] MTP A16 大 M 路线完整，不能引入 T<=16 限制或把它派发进旧 block-FP8 expert kernel。
- [x] EP 本地无命中、冷热 expert、尾组不满时均正确；缺席 slot 不解引用，不通过 Host readback 决定调度。

### E2：topk_sigmoid_swiglu_moe 的完整路径

- [x] 注册 Main CSF 与 MTP raw-multiplier，包含 I1024/I2048。将当前 `NVFP4 ? nvfp4 : fp8`
  的二选一改为明确完整的 format/layout/policy 分派。
- [x] Main shared 接 block-FP8/FP32 A16；MTP shared 接 MXFP8 AllowA8；router BF16 `[288,4096]` 保持。
- [x] 扩展 shared/down/join 实现；当前 `launch_a16_ffn_join` 仅支持 BF16/Q8，不能继续默认所有 shared 都属于这两类。
- [x] 保留 routed＋shared 的融合输出，不在 join 前增加新的 BF16 中间写回；shared 的 intermediate 仍为 TP2 的1024。
- [x] 本地 routed 为零时 shared 仍执行；完整调用输出覆盖与累加行为正确。

### E3：分布敏感的深度优化

复用现有 MoE bench 的 `reuse/pool/tail/spread` 分布，并明确实际去重 expert 数 U 与 assignments A。
加入真实 route 分布和 EP 的 0/少量/多数本地命中，以及两 rank 倾斜的代表负载。

- [x] 每种表示在 M=1..256 逐点测量，大 M 测512/1024/2048/4096。
- [x] CSF fixture 使用真实 scale 编码分布，覆盖普通和异常较多的 expert；不能全用无 exception 的理想样例。
- [x] 有效权重流量按实际选中 expert 的 codes、bases、nibbles、offsets、page bases、exceptions、multiplier 计算。
  不用全 bank 平均字节数代替每个 expert 的变长区域，不用展开后的 scale 字节数抬高带宽数字。
- [x] `F_routed≈6×A_local×H×I_local`，另计实际 shared/router 工作。
  按输出 token tile 重复读取的权重、填充 MMA 和空组不是 useful work，另列为开销。
- [x] 完整入口计入 route、分组、激活量化、CSF、GU/down、shared 和 join。分别呈现冷/热权重与完整 latency 曲线。
- [ ] 小 M 争取权重带宽极限；重用增多后优化 tile 利用、分组粒度、persistent scheduling、流水线与计算吞吐。
  不把阶段名作为策略条件，以实际 M、group load、表示和有效调用合同选型。

## 9. Vision 全链路与资源闭合

| 顺序 | 入口 | 实际表示与几何 | 列数 | 本轮 |
|---|---|---|---|---|
| 1 | `linear_bias` patch | BF16 `[1024,1176]`＋bias | P | 保留 |
| 2 | norm / residual norm | BF16 gamma1024 | P | 保留 |
| 3 | `linear_split` QKV＋bias | MXFP8 `[3072,1024]` | P | L1/P1 |
| 4 | `rmsnorm_rope_2d` | 16 heads×64，BF16 gamma | P | 保留 |
| 5 | `packed_softmax_attention` | BF16 Q/K/V，等长/ragged segments | P | 保留 |
| 6 | `linear_bias` O | MXFP8 `[1024,1024]` | P | L1/P2 |
| 7 | `linear_swiglu`＋bias | NVFP4_MUL `[8192,1024]`，clamp10 | P | L3/P2 |
| 8 | `linear_bias` down | NVFP4_MUL `[1024,4096]` | P | L3/P2 |
| 9 | `linear_bias` downsample | BF16 `[4096,4096]`＋bias | F | 保留 |
| 10 | `linear` merger projection | NVFP4_MUL `[4096,4096]` | F | L3 |
| 11 | `layer_norm`、`gelu(Exact)` | BF16 gamma/bias4096 | F | 保留 |
| 12 | `linear_swiglu` merger | NVFP4_MUL `[20480,4096]`，clamp10 | F | L3/P2 |
| 13 | `linear` merger down | NVFP4_MUL `[4096,10240]` | F | L3 |

- [x] `vision::layout/plan_resources/encode` 使用各入口实际 policy 与 workspace 查询。
  覆盖 merger、attention、QKV、O、FFN GU/down 的实际 scratch 高水位。
- [x] 保留现有 patch/merge 布局与有界 encoder window；按 P/F 分别验证容量，不扩成无限显存预算。

## 10. 已有能力与模型接口接通

以下现有数学实现保持，参加最终组合验证：

| 范围 | 现有能力 |
|---|---|
| 输入与布局 | BF16 embedding、Vision feature scatter、gather/scatter batch rows、state fields、causal pairs |
| KDA | BF16 control split、BF16 linear_pair、gated RMSNorm、chunked/recurrent、Replay record/fold |
| Kpool | BF16/FP8 indexer、KV codec、pool selection、tail extraction、State/record 分配 |
| MTP | BF16 stem `[4096,8192]`、paired RMSNorm concat、RMSNorm、residual-add-RMSNorm |
| Head | 已有 Q4 optimized head `[77440,4096]`；Main/full MTP 的 MXFP8 由 L1 接入 |
| 输出 | sampling、distributed vocabulary、logprob/PPL、greedy-draft acceptance |
| 跨 rank | TP sum、ADP gather、Vision segment 放置、既有通信顺序 |

`linear_add`、`linear_topk` 当前不在 GLM 调用链中，不为本任务额外增加新格式路线。

### J1：模型组装与 Graph/workspace

- [x] `execution::Parameters/Profile` 消费实际 typed weights；W_A/W_B 改用对应 single/paired 形式和明确行域，
  不再按旧单 BF16 parent 顺序拼接。
- [x] MTP factors 改成编码引用；`MlaReadout::scratch_bytes/index_profile/attention_profile` 与相应 child capture/update
  使用带表示的 profile。格式选择在准备期固定，上下文长度变化遵守现有节点更新合同。
- [x] 执行参数从已物化 `ModelShard` 获取 GPU calibration，不能借用早期 planning profile 中为空的 device view。
- [x] KDA、Kpool、FFN、Vision 的 capacity 覆盖每条实际 route；区间容量覆盖切换高水位，而不是只查询最大 T。
- [ ] MTP full/optimized head、selection 新建/reuse、History 修复、Main prefill/verify/ordinary decode 都接通同一语义实现。
- [x] 无热路径 allocation、权重重打包、全量解压、CPU 校准回读或为了省事拆碎外层 decode Graph。

## 11. 实施依赖与完成判据

推荐顺序：

1. F0/F1：device codec、正确的表示/许可视图、独立参考与原生 fixture。
2. L1～L4：先逐个完成纯 linear 的数学与全 M 调优，得到可复用 contraction。
3. P1/P2、K1、H1、A1/A2：扩展实际融合消费者，各 family 独立调优；E1 的 MoE contraction 可在 F0 后并行。
4. P3/E2/E3：完成 FFN 与完整 MoE，把 shared/join、路由与输入量化成本纳入优化。
5. Vision 资源、J1 与完整模型组合验证；Op 成果和 decode step 结果分别报告。

### B1：测试与 bench 基础

- [x] 扩展现有 quantized-weight fixture 与独立数学参考，支持四种实际组合：MXFP8、FP32 block FP8、
  raw NVFP4_MUL、CSF NVFP4_MUL；有非平凡 multiplier、逐 expert 校准、FP32 scale 原字及 exceptions。
- [x] 扩展现有 linear/split/bias/SwiGLU、KDA、mHC、Kpool、MoE bench 的格式和 profile，保留公共入口。
- [x] CSF 流量在 fixture 准备阶段按实际压缩布局确定，测量热路径不做 Host readback。
- [x] 永久正确性 cases 覆盖数学和真实边界；性能的密集 shape/M 扫描留给 bench，不复制成庞大测试笛卡尔积。
- [x] 新格式按独立数学参考验收，旧支持对象通过相关回归检查；比较实现之间相同不能替代正确性。
- [x] 每个新增入口提交完整逐点数据、匹配的 roofline 和仍存在的差距；纯 contraction 的成绩不能替代融合入口成绩。
- [ ] 最终实际运行新产物的 Main/MTP/Vision、TP2/EP2、eager/decode Graph；未覆盖的路径明确保留未完成状态。

本清单的完成意味着所有实际消费者都原生使用已生成的表示，压缩与数值优势在执行端保留，
并有完整 Op 的优化数据；能分派、能出结果或仅单个大 M 点很快，都不足以标记整项完成。
