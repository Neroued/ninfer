# GLM5Next Linear / MoE 实施清单

在 `.50` 工作区完成以下算子的补齐与 Spark 调优。按纯 Linear、投影扩展、完整
FFN/MoE 的顺序推进，已有实现直接复用。

量化和数学依据：[量化方案](../../maintainer/glm5next-quantization.md)、
[模型定义](../../maintainer/glm5next-model.md)。本任务包含下列确定的矩阵和 FFN/MoE；
MLA prepare/attention、KDA 状态链和 mHC 按各自 family 设计实施，不属于本清单。

## 1. 输入输出与格式

- 权重记作 `W[N,K]`。普通 Linear 输入为连续 BF16 `[K,M]`，输出为连续 BF16
  `[N,M]`，第零维连续；计算 `Y=W*X`，无 bias 或 activation。
- Q8 使用 `Q8_G32_FP16 / RowSplit / A16Only`；Q4 使用
  `Q4_G64_FP16 / RowSplit / A16Only`；BF16 使用 `Contiguous / A16Only`。
- Main routed/dense 使用原生 NVFP4、`BlockScaleK16M128x4 / AllowA4`；MTP routed
  使用 `FP8_E4M3FN_BLOCK128X128_BF16 / BlockScaleM128K128 / AllowA8`。
- 表中 shared 已按 intermediate TP2 切为 I=1024，head 已按 vocabulary TP2 切分。
  MLA query/O 为完整 heads，M 是本 owner 的列数。Stem/Vision 按表中完整矩阵
  实现纯 Linear。
- 本地 FFN/MoE 输出 BF16 `[4096,M]` 局部贡献。沿用现有 workspace、execution 和
  Graph 合同，分片与通信由调用方负责。

## 2. 补齐纯 Linear

- [ ] 完成下表各 shape 的 Spark 实现、分派、构建注册与调优。

| 编号 | 格式 | W[N,K] | 用途 / 行序 | 当前缺口 |
|---|---|---|---|---|
| P7 | Q8 | `[2048,4096]` | Shared GU：`[gate1024;up1024]` | 已有通用实现，补 Spark 调优 |
| P8 | Q8 | `[4096,1024]` | Shared down | 新增 |
| P10 | Q8 | `[4096,16384]` | Main/MTP MLA O | 新增 |
| P11 | Q4 | `[77440,4096]` | Optimized draft head 词表分片 | 新增 |
| P12 | BF16 | `[20480,1536]` | W_B：`[MLA Q16384;Indexer Q4096]` | 新增 |
| P13 | BF16 | `[16384,1536]` | W_B 的 MLA Q 前缀 | 新增 |
| P14 | BF16 | `[4096,8192]` | MTP stem，输入 `[e4096;h4096]` | 新增 |
| V1 | BF16 | `[1024,1176]` | Vision patch，K=`3*2*14*14` | 新增 |
| V2 | BF16 | `[3072,1024]` | Vision `[Q1024;K1024;V1024]` | 新增 |
| V3 | BF16 | `[1024,1024]` | Vision attention O | 新增 |
| V4 | BF16 | `[8192,1024]` | Vision block `[gate4096;up4096]` | 新增 |
| V5 | BF16 | `[1024,4096]` | Vision block down | 新增 |
| V6 | BF16 | `[4096,4096]` | Vision downsample / merger projection，共用几何 | 新增 |
| V7 | BF16 | `[20480,4096]` | Vision merger `[gate10240;up10240]` | 新增 |
| V8 | BF16 | `[4096,10240]` | Vision merger down | 新增 |

具体要求：

- P13 消费 P12 parent 的前 16384 行只读 view，只计算这个前缀。
- P11 输出完整 `[77440,M]` logits，计入完整输出写入的成本。
- P14 消费已准备好的连续 `[8192,M]` 输入。
- V1 正确处理 K=1176 的归约尾部。Patch feature 顺序和 V6 downsample 权重映射
  见 [Vision 布局](parallel.md#21-vision-权重的保值布局)。
- Vision 本清单只做纯 Linear；bias、norm、activation 和相应融合按
  [Vision 设计](vision.md)另行实施。

实现位置：`src/ops/linear/projection/{q8,q4,bf16}/dgx_spark/shapes/`，同步各格式的
shape 声明、dispatch 和 sources。复用 `src/ops/linear/kernels/`。
P7 当前通用实现位于 `projection/q8/shapes/n2048_k4096.cu`，按现有 target 组织
分出 Spark 调度。各 shape 保持自己的调优策略。

测试使用 `tests/ops/linear/projection/cases_{q8,q4,bf16}.cpp`；测量使用
`ninfer_linear_bench --qtype ... --n N --k K --tokens ...`。

## 3. 扩展 split 与 clamped GU

- [ ] **F5：BF16 W_B split。** 基于 P12，扩展 `linear_split()`，将
  `[20480,1536]` 的投影直接写为两个独立连续输出：
  `MLA query[16384,M]`、`Indexer query[4096,M]`。
  实现放在 `src/ops/linear/split/bf16/dgx_spark/`，同步分段注册和容量查询。
  测试使用 segmented split suite；bench 使用
  `ninfer_linear_split_multi_output_bench --format bf16-segmented
  --n 20480 --k 1536 --rows 16384,4096`。
- [ ] **F4：Q8 clamped GU。** 基于 P7，扩展现有
  `linear_swiglu(..., limit, policy, ...)`。输入 BF16 `[4096,M]`，Q8 GU
  `[2048,4096]`，输出连续 BF16 `[1024,M]`：

  ```text
  gate = W_gate * x
  up = W_up * x
  product = SiLU(min(gate, limit)) * clamp(up, -limit, limit)
  ```

  本模型 `limit=10`，沿用现有显式参数。Gate 只截上界，up 截双边界，在非线性前
  clamp。Contraction 直接产生 product；融合数学不增加 gate/up 的 BF16 舍入边界。

F4 需要修改 `src/ops/linear/swiglu/linear_swiglu.cpp`、`dispatch.*`，增加
`q8/dgx_spark/` 实现，并同步公共格式说明。测试扩展
`tests/ops/linear/swiglu/cases_q8.cpp`；`bench/ops/linear_swiglu_clamped.cu`
增加 Q8 fixture。Split、SwiGLU 各自维护调优策略，计算模板按需共享。

## 4. 完整 Q8 shared FFN 与 MoE

- [ ] **M2：完整 shared `swiglu()`。** H=4096、I=1024，GU/down 均为 Q8 A16，
  clamp=10。计算 `y=W_down*product`，输出 BF16 `[4096,M]`，shared 系数为 1。
  按完整 FFN 数学实现，内部 product 不增加公开的 BF16 舍入边界。
- [ ] **E3：Main 完整本地 MoE。** 扩展 `topk_sigmoid_swiglu_moe()`，支持
  NVFP4 routed＋Q8 shared。
- [ ] **E4：MTP 完整本地 MoE。** 同一入口支持 block-FP8 routed＋Q8 shared。

Routed 几何为 H=4096、I=2048、global E=288、TopK=8、local E=144，使用现有
indexed bank/subset。Router 继续执行现有 sigmoid＋correction-bias 全局 Top8，
系数 normalization/scaling 保持不变；本地筛选后不重新归一化。

完整输出为 `routed_sum + shared`，最后写 BF16。Shared 不乘 routed scaling factor
或额外 gate，零 routed 命中时也计算一次。

| 修改位置 | 必须补齐的内容 |
|---|---|
| `src/ops/moe/swiglu/entry.cpp` | Q8 profile、codes/scales 验证与读域、workspace 容量、执行选择 |
| `src/ops/moe/swiglu/dgx_spark/` | 完整 Q8 GU/down 计算及合并输出 |
| `src/ops/moe/operands.cpp` 的 `unscaled_shared()` | 当前强制 BF16/group=0；改为按 shared profile 验证原生 Q8-G32/FP16 |
| `src/ops/moe/topk_sigmoid_swiglu_moe/entry.cpp` | 当前强制 BF16 shared、直调 BF16 join；增加 Q8 格式的执行与合并 |
| `include/ninfer/ops/moe.h` | 同步支持格式；沿用现有 profile 和入口 |

测试扩展 `tests/ops/moe/test_swiglu*`、`test_sigmoid_moe.cpp`，使用独立的 Q8
权重解码与数学参考。Shared 单独按 A16 精度检查；完整 MoE 覆盖零/部分/全部本地
命中、非连续 expert IDs、None 和 Graph 数据更新。

`bench/ops/swiglu_bench.cu`、`bench/ops/moe/sigmoid_moe.cu` 增加 Q8 shared
fixture 和流量计费，测量完整公共 Op。

## 5. 迁入已有 routed MoE 优化

- [ ] 对照当前实现，迁入 `.50` 上有效的 GLM NVFP4 优化候选。

来源：

```text
/home/neroued/ninfer/.agent-tmp/glm-dev-reorg/worktree
branch: feat/glm5next-dev-reorg
```

相关提交：`629bc16f`（小 M vector）、`e99a73ef`（non-RDC 编译组织）、
`128028b9`（按 expert load 选择调度）、`4ccc3fff`（资格 case 整理）；
公共 schedule/grouping 依赖参考 `46ee5c3b`。

主要位置为 `src/ops/moe/kernels/cuda/block_scaled/`、
`routed_swiglu/cuda/nvfp4/dgx_spark/shapes/h4096_i2048_e288_k8/` 和 sources。
按当前组织接入，检查所需依赖与 workspace 容量；load-aware 分派保留在 device。

使用现有 `ninfer_moe_clamped_ep_bench` 比较迁入前后相同负载，保留有效改动。
覆盖小 M 的不同本地命中数，以及大 M 的分散/热点分布。沿用
`tests/ops/moe/test_clamped_ep.cpp`、`clamped_ep_fixture.*`；公共 kernel 修改
影响到的已有 profile 做相应回归。最后测完整 Q8 shared MoE 的组合成本。

## 6. 已有能力

以下直接复用，不重复实现：

| 内容 | 已有几何 / 能力 |
|---|---|
| KDA QKV / O | Q8 `[12288,4096]`、`[4096,4096]`；QKV split `[4096,4096,4096]` |
| KDA control | BF16 `[288,4096]`、split `[32,128,128]`；f_b/g_b 各 `[4096,128]` |
| Dense FFN | NVFP4 GU `[12288,4096]`、down `[4096,6144]`、clamped GU 和完整 `swiglu()` |
| Main head | BF16 `[77440,4096]` |
| Router | `topk_sigmoid_route()`，H=4096、E=288、TopK=8，FP32 logits/coefficients |
| Routed experts | NVFP4/AllowA4 与 block-FP8/AllowA8 的 clamped EP144 实现 |
| BF16 shared | GU/down、clamped GU、完整 shared FFN 与完整 sigmoid MoE，供 Q8 扩展参考 |

MLA W_A、U_K/U_V 和 mHC F 的收缩随各自完整算子处理；本清单只新增第 2 节的
普通 BF16 输出矩阵。

## 7. 调优与完成要求

沿用[现有调优方法](../../maintainer/linear-tuning.md)和
[Op 规范](../../maintainer/op-development.md)。先确认所用 test/bench 的数学与计费，
逐项实现、验证和调优。

- Linear/FFN 覆盖 `M=1..8`，以及 `12,16,24,32,48,64,96,128,256,512,1024,2048,4096`。
  重点是小 M 带宽和延迟，异常区间与调度切换处补点；Vision 兼顾非整 tile patch 数。
- MoE 在上述区间选择代表点，覆盖本地 0/少量/8 命中、热点复用和分散路由。
  按实际本地 token-expert 数计算 FLOPs，按实际去重 expert 集合计权重流量。
- 报告完整 Op 延迟、有效 GB/s、useful FLOPs 和 roofline；冷/热缓存分开，
  带宽与算力参照使用 `bench/common/roofline.h`。权重流量包括 codes/scales，
  workspace/重排另计；Q8/Q4 A16 使用相应计算精度的算力参照。
- 数值按实际表示的输入和独立数学参考检查。`AllowA4`/`AllowA8` 分别使用其公开
  精度要求；融合参考不由已舍入的独立算子输出拼接。
- 常驻测试覆盖代表尺寸、尾部、关键语义和 Graph，不复制整个性能扫描矩阵。
- 完成生产实现、分派、构建注册及对应 tests/bench；运行 Spark 完整编译和受影响
  检查。共享实现改动同时保持 RTX5090 的构建与已有行为。
