# Inkling-2-Mini RTX 5090 扩展现有 Op 需求与性能规范（E1–E9）

> [!NOTE]
> **范围声明**：本文档明确 `thinkingmachines/Inkling-2-Mini`（Main Decoder + DSpark `mtp.safetensors` + Vision `model.visual`）在 **RTX 5090（SM120）** 上需要**扩展现有 NInfer Op 家族**的 **9 个 Op（`E1`–`E9`）** 的完整需求，包括公开接口、输入/输出与资源合同、精确算法定义、数值验收标准、RTX 5090 理论 Roofline 分析与极致性能指标。
> 已完成的 Phase 1 原生 BF16 `linear` / `linear_*` epilogue（30 个 Op，分支 `feat/inkling2-linear-rtx5090`）与 Inkling-2-Mini 全新专属 Op（Block-AttnRes、MessageGate、SConv、Inkling SWA/Full Attention、Logit Softcap、DFlare/Markov、LayerScale）不在本文档范围内。

---

## 1. 全局硬件参照与工程铁律

### 1.1 RTX 5090 权威 Roofline 参照

所有带宽下界、算力下界、算术强度（AI）和有效利用率统一以 [`bench/common/roofline.h`](../../bench/common/roofline.h) 中维护的 RTX 5090（SM120，170 SMs，32 GiB GDDR7，96 MiB L2）参照值为准：

| 指标 | 数值 | 对应岭点（vs Peak / vs Sustained） | 用途 |
|---|---:|---:|---|
| 理论显存带宽 `memory_gbs` | **1,792.0 GB/s** | — | 官方 `bandwidth_reference_us = bytes / (1792.0 × 1000)` |
| 实测持续读带宽 `sustained_read_gbs` | **1,674.5 GB/s** | — | 流式读物理上界（理论峰值的 **93.4%**） |
| BF16 Tensor Core（FP32 累加）`bf16_f32_tflops` | **209.5 TFLOP/s** | 116.9 / 125.1 FLOPs/B | BF16 GEMM/MMA、A16 路径与 BF16 Attention |
| FP8 Tensor Core（FP32 累加）`fp8_f32_tflops` | **419.0 TFLOP/s** | 233.8 / 250.2 FLOPs/B | 参考 |
| MXFP8 Block-Scaled MMA（FP32 累加）`mxfp8_f32_tflops` | **838.0 TFLOP/s** | 467.6 / 500.4 FLOPs/B | `E2` MXFP8 `AllowA8` W8A8 路径 |
| NVFP4 Block-Scaled MMA（FP32 累加）`nvfp4_f32_tflops` | **1,676.0 TFLOP/s** | 935.3 / 1,000.9 FLOPs/B | `E2` NVFP4_MUL `AllowA4` W4A4 路径 |

### 1.2 核心优化与验证铁律

1. **一切性能以 Roofline 分析为准，一切追求极致**：
   - 每个 Op 必须核算**逻辑最小必要流量（`useful_bytes`）**与**有用浮点运算量（`useful_flops`）**（MoE 权重按实际去重触及的 expert 集合含 codes + scales 计费；Attention 按实际可见历史 KV 与 query 计费；原位 Op 计读+写）。
   - 理想下界 $\text{bound}_{\text{roofline}} = \max(\text{bytes}/\text{BW},\; \text{FLOPs}/\text{Compute})$；小规模工作负载（下界 $< 3\text{ µs}$）额外核算 **CUDA Graph Body 发射/同步地板**，通过融合减少 launch 数、消除中间写回与冗余同步来逼近硬件极限。
   - 所有性能结论必须基于公开 Op 入口在 RTX 5090 上实测的 **CUDA Graph Body（Cold 驱逐 L2 256 MiB + Warm 对照）** 完整曲线，计入输入量化、分组排序、workspace 读写与全链路 kernel，绝不单独报 contraction 子阶段耗时。
2. **`AllowA4` / `AllowA8` 严禁预设精度切换阈值**：
   - `LinearPolicy::A16Only`、`AllowA8`、`AllowA4` 仅表达调用方授予的**公开数值精度许可**，不绑定任何内部算术机制。
   - 在 `AllowA4`（NVFP4_MUL）与 `AllowA8`（MXFP8）许可下，**严禁在设计或编码前预设固定 `T` 或固定 per-expert 负载阈值来决定 A16 vs A4/A8 或 SIMT/Vector vs MMA/TMA 切换**。
   - 实际切换边界（包括按 expert 负载动态分桶的混合策略，如同一批次内低负载 expert 与高负载 expert 分别走不同路线）必须由 RTX 5090 全分布（`uniform / reuse / pool / tail / spread`）× 密集 `T` 曲线实测耗时唯一决定，以各区间**实测延迟最短、Roofline 效率最高**为唯一判据。
3. **公开合同与独立数学验证**（遵循 [`docs/maintainer/op-development.md`](../maintainer/op-development.md)）：
   - 全部 9 个 Op 使用独立 FP64（或精确整数/位级）参考实现按公开表示输入验证；融合路径不得用若干生产 Op 拼接作为参考。
   - 严格覆盖：边界 extent、非对齐/尾块、无效行/零提交（`valid_columns == 0` / `live_rows == 0` / `ExpertSubsetKind::None`）、未触及内存逐位保护、aliasing 拒绝、workspace 非单调高水位验证、CUDA Graph 固定地址 replay 变值正确性、零隐藏设备分配与零 Host 回读。

---

## 2. 九个扩展 Op 总览清单

| 编号 | Op 入口 | 所属子系统 | 公开头文件 | 核心增量（RTX 5090） | 主导瓶颈 |
|---|---|---|---|---|---|
| **E1** | `topk_sigmoid_route` | Main MoE | [`include/ninfer/ops/moe.h`](../../include/ninfer/ops/moe.h) | RTX 5090 启用并注册 `H=2304, E=128, K=8`（`scale=8.0, eps=1e-20`） | 小 T 发射受限；大 T 带宽→BF16 算力 |
| **E2** | `routed_swiglu` | Main MoE | [`include/ninfer/ops/moe.h`](../../include/ninfer/ops/moe.h) | 新增 SoftcappedSwiGLU `(4.0, 25.0)`；RTX 5090 注册 `H=2304, I=1152, E=128, K=8` 的 `NVFP4_MUL`（L0–19）与 `MXFP8`（L20–23） | 小/中 T 带宽受限；高负载/skew 算力受限 |
| **E3** | `kv_cache_append_prefix`（Cyclic） | Main SWA / DSpark | [`include/ninfer/ops/kv_cache_append.h`](../../include/ninfer/ops/kv_cache_append.h) | Cyclic 重载新增 `capacity=1024`（`D=128, Hkv=8`，BF16 K + FP16 V） | 纯带宽 / 小批量发射受限 |
| **E4** | `rmsnorm_rope`（Pair） | DSpark | [`include/ninfer/ops/rmsnorm_rope.h`](../../include/ninfer/ops/rmsnorm_rope.h) | 成对重载新增 `(Hq=24, Hkv=8), D=128`（`W=2..16, B>=1`） | 纯带宽 / 小批量发射受限 |
| **E5** | `rmsnorm_pack_tail` | DSpark | [`include/ninfer/ops/rmsnorm_pack_tail.h`](../../include/ninfer/ops/rmsnorm_pack_tail.h) | 新增 `H=2304`（`W=2..16, 1<=g<W, B>=1`） | 纯带宽 / 小批量发射受限 |
| **E6** | `sliding_window_attention` | DSpark L0/L2 | [`include/ninfer/ops/sliding_window_attention.h`](../../include/ninfer/ops/sliding_window_attention.h) | 新增非因果 `(Hq=24, Hkv=8, G=3), D=128, window=1024` | 纯带宽受限（AI ≤ 48） |
| **E7** | `context_softmax_attention` | DSpark L1 | [`include/ninfer/ops/softmax_attention.h`](../../include/ninfer/ops/softmax_attention.h) | 新增非因果 `(Hq=24, Hkv=8, G=3), D=128`，支持至 `M1`（或至少 `K256`） | 纯带宽受限（AI ≤ 48） |
| **E8** | `speculative_accept_sparse_drafts` | DSpark Verify | [`include/ninfer/ops/speculative_accept.h`](../../include/ninfer/ops/speculative_accept.h) | 解除 `248077/248320` 硬编码，支持 `token_domain=200058, physical_rows=200064`（及通用合法域） | Raw-greedy 发射受限；通用路径纯带宽受限 |
| **E9** | `rope_2d_interleaved` | Vision ViT | [`include/ninfer/ops/rope.h`](../../include/ninfer/ops/rope.h) | 新增 `D=64, H=16` **Pairwise Interleaved** + `[W, H]` 轴序 2D RoPE | 纯带宽受限 |

---

## 3. 逐项详细需求与 Roofline 性能规范

---

### E1 · `topk_sigmoid_route`（Main 48 逻辑层 MoE 路由）

#### 3.1.1 现有缺口
- [`src/ops/moe/topk_sigmoid_route/entry.cpp`](../../src/ops/moe/topk_sigmoid_route/entry.cpp) 当前仅在 `#if NINFER_TARGET_SPARK` 下启用，且硬编码 `H=4096, E=288, K=8`；RTX 5090 编译路径直接抛出 `unsupported profile`。
- 现有 DGX Spark 实现采用「GEMV/MMA 写回 FP32 `scores[E, T]` + 独立 TopK kernel」两段发射；在 Inkling-2-Mini 的 `E=128` 几何下，小/中 `T` 时多一次 kernel launch 与全局 logits 读写开销不可忽略。

#### 3.1.2 公开接口与输入/输出合同
沿用 [`include/ninfer/ops/moe.h`](../../include/ninfer/ops/moe.h#L157-L176) 既有公开签名，**无需修改头文件签名**：
```cpp
std::size_t topk_sigmoid_route_workspace_capacity_bytes(const SigmoidRouteProfile& profile,
                                                        std::int32_t min_tokens,
                                                        std::int32_t max_tokens);
void topk_sigmoid_route(const Tensor& x, const SigmoidRouteProfile& profile,
                        const Weight& control, const Tensor& correction_bias,
                        ExpertRoutes& routes, MoeTokenValidity validity,
                        MemoryArena& workspace, CudaExecutionView execution);
```
- **注册 Profile（RTX 5090）**：`profile.route = {hidden=2304, experts=128, top_k=8}`；`scaling_factor > 0` 有限（Inkling-2-Mini 实参为 `8.0f`）；`normalization_epsilon > 0` 有限（Inkling-2-Mini 实参为 `1e-20f`）。
- **输入**：
  - `x`：连续 16B 对齐 Device `BF16 [2304, T]`，`T >= 1`。
  - `control`：Device `Weight`，`QType::BF16`, `QuantLayout::Contiguous`, `[128, 2304]`，16B 对齐（`589,824 B`）。
  - `correction_bias`：连续 4B 对齐 Device `FP32 [128]`（`512 B`）。
  - `validity`：可选 `MoeTokenValidity`（空则全量有效；否则 `valid_columns` 为 Device `I32 [P]`，`T = P * tokens_per_row`，每行有效前缀 `0 <= valid_columns[p] <= tokens_per_row`）。
- **输出**：
  - `routes.ids`：连续 16B 对齐 Device `I32 [8, T]`。
  - `routes.coefficients`：连续 16B 对齐 Device `FP32 [8, T]`。
- **所有权与执行合同**：输出及 workspace 彼此且与所有输入严格不重叠；输入只读保留；支持 CUDA Graph capture/replay（消费当前 Device `x / bias / valid_columns`）；`workspace` 容量查询覆盖 `[min_tokens, max_tokens]` 全区间真实高水位（若小 `T` 走单阶段融合零 scratch、大 `T` 走两阶段需要 `FP32 [128, T]`，容量查询必须准确反映区间最大值）。

#### 3.1.3 精确算法定义
对每个有效列 $t \in [0, T)$，全部计算在 FP32 中执行，**`control * x` 与 `sigmoid` 之间不存在 BF16 舍入边界**：
1. Logits 与无偏分数：
   $$\ell_{e,t} = \sum_{d=0}^{2303} \text{FP32}(W_{\text{ctrl}}[e, d]) \cdot \text{FP32}(x[d, t]), \qquad s_{e,t} = \frac{1}{1 + e^{-\ell_{e,t}}}, \quad e \in [0, 128)$$
2. 带偏置选拔：计算选择键 $c_{e,t} = s_{e,t} + b_e$，选取最大的 $K=8$ 个专家索引 $(e_0, \dots, e_7)$，按 $c_{e,t}$ **严格降序排列**，相等时**较小全局 ID 优先**。
3. 无偏归一化与缩放（`correction_bias` 仅影响选拔，不进入权重）：
   $$w_{j,t} = \text{scaling\_factor} \cdot \frac{s_{e_j, t}}{\sum_{m=0}^{7} s_{e_m, t} + \varepsilon}, \qquad j \in [0, 8)$$
4. 无效列（由 `validity` 标定）：不保证跳过计算，但必须精确写出 `routes.ids[:, t] = -1`、`routes.coefficients[:, t] = 0.0f`。

#### 3.1.4 数值验收标准
- 独立 FP64 参考逐列计算 `logits → sigmoid → select → renorm`。
- 在足够分离的测试输入上 Top-8 索引集合与顺序完全匹配（且覆盖 ties-by-lower-ID 精确边界测试）；`coefficients` 满足 FP32 normwise/pointwise 精度要求，无 NaN/Inf；`validity` 无效列逐位精确为 `-1 / +0.0f`。

#### 3.1.5 RTX 5090 Roofline 理论分析
逻辑最小字节 $B(T) = 589{,}824 + 512 + 4{,}608 T + 64 T = 590{,}336 + 4{,}672 T$；有用算力 $F(T) = 2 \cdot 2304 \cdot 128 \cdot T = 589{,}824\,T$ FLOPs：

| `T` | 逻辑字节 | 有用 FLOPs | AI (F/B) | 带宽下界 Peak / Sust (µs) | BF16-F32 算力下界 (µs) | 理想 Roofline 下界 (µs) | 主导机制 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.595 MB | 0.001 GF | 0.99 | 0.33 / 0.36 | 0.003 | **0.33** | 发射延迟主导 |
| 4 | 0.609 MB | 0.002 GF | 3.87 | 0.34 / 0.36 | 0.011 | **0.34** | 发射延迟主导 |
| 8 | 0.628 MB | 0.005 GF | 7.52 | 0.35 / 0.37 | 0.023 | **0.35** | 发射延迟主导 |
| 16 | 0.665 MB | 0.009 GF | 14.19 | 0.37 / 0.40 | 0.045 | **0.37** | 发射延迟主导 |
| 64 | 0.889 MB | 0.038 GF | 42.45 | 0.50 / 0.53 | 0.180 | **0.50** | 发射+小矩阵带宽 |
| 256 | 1.786 MB | 0.151 GF | 84.53 | 1.00 / 1.07 | 0.721 | **1.00** | 带宽主导 |
| 512 | 2.982 MB | 0.302 GF | 101.26 | 1.66 / 1.78 | 1.441 | **1.66** | 带宽≈算力交界 |
| **1024** | **5.374 MB** | **0.604 GF** | **112.38** | **3.00 / 3.21** | **2.882** | **3.00** | **岭点附近** |
| 4096 | 19.727 MB | 2.416 GF | 122.47 | 11.01 / 11.78 | 11.533 | **11.53** | BF16-TC 算力主导 |

#### 3.1.6 极致性能指标与调优要求
- **融合优势挖掘**：`E=128` 极适合小/中 `T` **单 kernel 融合**（例如每 token 1 CTA / 4 warps = 128 线程，或 Split-K 规约后片上直接完成 `sigmoid + bias + Top-8 + renorm`，彻底消除第 2 次 kernel launch 与 `scores` 全局往返——每步 48 层可省 ~100–150 µs）。大 `T` 若实测「BF16 Tensor Core MMA + 轻量 warp Top-8」更快，则由实测曲线决定切换点。
- **目标性能（RTX 5090 CUDA Graph Body Cold）**：
  - **关键锚点 `T = 1, 4, 8`**：单次融合 launch，中位耗时逼近 GPU 单 kernel 地板 + 0.6 MB 冷读时间，目标 **`≤ 3.5–4.5 µs`**（Warm **`≤ 2.5–3.2 µs`**）；
  - **`T = 16..128`**：保持平滑无抖动爬坡，避免 2-launch 开销退化；
  - **`T = 1024`**：目标 **`≤ 6.0 µs`**（有效利用率 ≥ 50% Roofline）；
  - **`T = 4096`**：目标 **`≤ 16.0 µs`**（有效算力 ≥ 150 TFLOP/s，≥ 72% BF16-F32 峰值）。
- **Benchmark**：扩展 [`bench/ops/sigmoid_route_bench.cu`](../../bench/ops/sigmoid_route_bench.cu) 支持 `--profile h2304-e128-k8`，扫描 `1:128:1` + `{256, 512, 1024, 2048, 4096}`。

---

### E2 · `routed_swiglu`（Main MoE 44× NVFP4_MUL + 4× MXFP8，核心重头）

#### 3.2.1 现有缺口
1. [`SwiGLUProfile`](../../include/ninfer/ops/moe.h#L19-L27) 目前仅支持 `std::optional<float> clamp_limit`（无截断或 `SiLU(min(g,L))*clamp(u,-L,L)`），不支持 Inkling-2-Mini 的 **SoftcappedSwiGLU**。
2. [`src/ops/moe/routed_swiglu/cuda/dispatch.cpp`](../../src/ops/moe/routed_swiglu/cuda/dispatch.cpp) 在 RTX 5090（`!NINFER_TARGET_SPARK`）上尚未注册 `H=2304, I=1152, E=128, K=8`，也未接入 `QType::NVFP4_MUL`。
3. 全仓库 `routed_swiglu` **完全缺失 `QType::MXFP8`（`QuantLayout::RowGroupScale`）专家路由执行路径**。

#### 3.2.2 公开接口与输入/输出合同
扩展 [`include/ninfer/ops/moe.h`](../../include/ninfer/ops/moe.h) 中 `SwiGLUProfile` 的激活策略字段（保持向后兼容，不引入历史兼容分支，清晰表达三种正交激活语义）：
```cpp
struct SwiGLUSoftcap {
    float gate_cap = 0.0f; // 有限正数，Inkling-2-Mini = 4.0f
    float up_cap   = 0.0f; // 有限正数，Inkling-2-Mini = 25.0f
    friend bool operator==(SwiGLUSoftcap, SwiGLUSoftcap) = default;
};
// SwiGLUProfile 中：clamp_limit 与 softcap 互斥（最多设置其一，同时设置抛出 std::invalid_argument）
std::optional<float> clamp_limit;
std::optional<SwiGLUSoftcap> softcap;
```
公共调用入口沿用 [`routed_swiglu`](../../include/ninfer/ops/moe.h#L223-L233)（Inkling-2-Mini `n_shared_experts = 0`，不调用 shared 重载）：
```cpp
std::size_t routed_swiglu_workspace_capacity_bytes(const RoutedSwiGLUProfile& profile,
                                                   ExpertSubsetKind subset,
                                                   DType output_type, MoeEpilogue epilogue,
                                                   std::int32_t min_tokens,
                                                   std::int32_t max_tokens);
void routed_swiglu(const Tensor& x, const RoutedSwiGLUProfile& profile,
                   const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, MoeEpilogue epilogue, Tensor& destination,
                   MemoryArena& workspace, CudaExecutionView execution);
```
- **注册 Profile（RTX 5090）**：
  - 共同几何：`hidden = 2304`, `intermediate = 1152`, `experts = 128`, `top_k = 8`, `softcap = {4.0f, 25.0f}`, `clamp_limit = nullopt`。
  - **格式 1（L0–19，44 逻辑层）**：`gate_up_type = down_type = QType::NVFP4_MUL`，`layout = QuantLayout::BlockScaleK16M128x4`（如后续扩展 `Csf32` 可同构挂接），`policy ∈ {LinearPolicy::A16Only, LinearPolicy::AllowA4}`。
    - 权重编码：`E2M1` 4-bit 打包（低 nibble 对应低特征维，`[2304, 2304]` `gate_up` 按 `[gate(1152); up(1152)]` 反交错落盘；`[2304, 1152]` `down`），`E4M3` G16 swizzled block scales，per-expert FP32 乘子 `matrix_scales[e] = amax[e] / outer_scale_denominator`。
    - 激活校准（`ActivationCalibrationView`）：`A16Only` 下不需要激活校准（可为空）；`AllowA4` 下提供 `ActivationScaleOperation::Multiply` 的正有限 FP32 标量或按 expert 数组（支持单标量共享与每 expert 独立，零运行时动态全向量 amax 规约）。
  - **格式 2（L20–23，4 逻辑层）**：`gate_up_type = down_type = QType::MXFP8`，`layout = QuantLayout::RowGroupScale`，`policy ∈ {LinearPolicy::A16Only, LinearPolicy::AllowA8}`。
    - 权重编码：`E4M3FN` 8-bit 码字 + `UE8M0`（`2^(scale - 127)`）G32 行组 scales，无需离线 tensor-level 激活校准（MXFP8 激活自身按每 32 元素动态生成 UE8M0 scale）。
- **输入/输出与状态合同**：
  - `x`：连续 16B 对齐 Device `BF16 [2304, T]`，`T >= 1`。
  - `routes`：`ids` Device `I32 [8, T]`（值域 `{-1} ∪ [0, 128)`，同 token 内非负 ID 互异）、`coefficients` Device `FP32 [8, T]`（直接消费不重归一化；`-1` 路由不读权重与 mask）。
  - `subset`：支持全部 4 种 `ExpertSubsetKind`（`All`, `None`, `MaskOnes`, `MaskZeros`）；`None` 写零且不访问 `routes` 和 `experts`。
  - `destination`：连续 16B 对齐 Device `BF16 [2304, T]` 或 `FP32 [2304, T]`；支持 `MoeEpilogue::Write`（BF16/FP32）与 `MoeEpilogue::AddResidual`（BF16，保留 Inkling-2-Mini 主路径所需的 `BF16 Write` 与通用 `FP32 Write`）。
  - 严格支持固定 profile 的 CUDA Graph capture/replay（replay 时允许 `x`、`routes`、`mask`、`expert_indices` 内容变化）。

#### 3.2.3 精确算法定义
对每个有效路由 $(j, t)$，记 $e = \text{ids}[j, t] \in [0, 128)$（仅当 $e$ 属于 `subset`）：
1. 权重解码：
   - **`NVFP4_MUL`**：$W_{r,k} = \text{E2M1}(\text{code}_{r,k}) \cdot \text{E4M3}(\text{scale}_{r, \lfloor k/16 \rfloor}) \cdot m_e$（$m_e$ 为 `matrix_scales[e]`）。
   - **`MXFP8`**：$W_{r,k} = \text{E4M3FN}(\text{code}_{r,k}) \cdot 2^{\text{ue8m0}_{r, \lfloor k/32 \rfloor} - 127}$。
2. Gate/Up 投影与 **SoftcappedSwiGLU**（$g_i, u_i$ 无可观察 BF16 边界）：
   $$g = W_{\text{gate}, e}\, x[:, t] \in \mathbb{R}^{1152}, \qquad u = W_{\text{up}, e}\, x[:, t] \in \mathbb{R}^{1152}$$
   $$a_i = \left(c_g \tanh\frac{g_i}{c_g}\right) \cdot \frac{1}{1 + e^{-g_i}} \cdot \left(c_u \tanh\frac{u_i}{c_u}\right), \qquad c_g = 4.0,\; c_u = 25.0$$
   > **关键语义点**：$\sigma(g_i)$ 必须作用于**未 softcap 的原始 $g_i$**；因第一项 $\in [-0.2785, 4.0]$、第三项 $\in [-25.0, 25.0]$，激活值严格有界于 $a_i \in [-100.0, +100.0]$。
3. Down 投影与路由加权累加：
   $$y[:, t] = \sum_{\substack{0 \le j < 8 \\ e = \text{ids}[j,t] \in \text{subset}}} \text{coeff}[j, t] \cdot \big(W_{\text{down}, e}\, a^{(e,t)}\big)$$
   `Write` 直接落盘（BF16 为单次 RN_BF16 舍入，FP32 直接写出）；`AddResidual` 在 FP32 加上旧 `destination` 后做单次 RN_BF16 舍入。

#### 3.2.4 数值验收标准
- 独立 FP64 参考按解码后的精确权重计算完整公式（不复制内部 A4/A8 激活量化或分块归约）。
- 精度准入严格按**公开 `LinearPolicy`**（与现有 `tests/ops/moe/` 一致，不因某 `T` 走更高精度而漂移标准）：
  - `A16Only`：满足 A16 normwise 相对误差与有限 gross pointwise 上限；
  - `AllowA8`（MXFP8）：满足 A8 normwise / pointwise 容差；
  - `AllowA4`（NVFP4_MUL）：满足 A4 normwise / pointwise 容差（特别验证校准乘子与 `[-100, 100]` 硬上界下的无饱和精度）。
- 额外必测：`-1` 路由零污染、`None / MaskOnes / MaskZeros` 严格子集隔离、热专家集中（`reuse`）与极端长尾（`tail`）负载、非单调 workspace 区间高水位、CUDA Graph 动态路由 replay。

#### 3.2.5 RTX 5090 Roofline 理论分析

单专家编码权重体量：
- **`NVFP4_MUL` (`BlockScaleK16M128x4`)**：codes $3 \times 2304 \times 1152 / 2 = 3{,}981{,}312\text{ B}$ + G16 scales $497{,}664\text{ B}$ + 2×FP32 `matrix_scales` $8\text{ B}$ = **`4,478,984 B`（4.271 MiB/expert）**；全 128 专家 = **546.75 MiB**。
- **`MXFP8` (`RowGroupScale`)**：codes $7{,}962{,}624\text{ B}$ + G32 UE8M0 scales $248{,}832\text{ B}$ = **`8,211,456 B`（7.831 MiB/expert）**；全 128 专家 = **1,002.38 MiB**。
- 每 token-expert assignment 有用 FLOPs：$2 \times 3 \times 2304 \times 1152 = \mathbf{15.925\text{ MFLOPs}}$；每 token（8 assignments）= **127.40 MFLOPs**。

##### (a) `NVFP4_MUL`（L0–19）Roofline 参照表（Uniform 路由 $U = E(1 - (1-K/E)^T)$ vs 极端热点 Hot-8 $U=8$）

| 分布 | `T` | 去重专家 $U$ | 平均负载 $c=8T/U$ | 有效字节 (MB) | 有用 FLOPs (GF) | AI (F/B) | 带宽下界 Peak/Sust (µs) | A16 算力下界 (µs) | A4 算力下界 (µs) | A16 Roofline (µs) | A4 Roofline (µs) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Uniform | 1 | 8 | 1.0 | 35.84 | 0.13 | 3.6 | 20.00 / 21.40 | 0.61 | 0.08 | **20.00** | **20.00** |
| Uniform | 4 | 29 | 1.1 | 129.93 | 0.51 | 3.9 | 72.50 / 77.59 | 2.43 | 0.30 | **72.50** | **72.50** |
| Uniform | 8 | 52 | 1.2 | 232.98 | 1.02 | 4.4 | 130.01 / 139.13 | 4.86 | 0.61 | **130.01** | **130.01** |
| Uniform | 16 | 82 | 1.6 | 367.43 | 2.04 | 5.5 | 205.04 / 219.42 | 9.73 | 1.22 | **205.04** | **205.04** |
| Uniform | 32 | 112 | 2.3 | 501.94 | 4.08 | 8.1 | 280.10 / 299.76 | 19.46 | 2.43 | **280.10** | **280.10** |
| Uniform | 64 | 126 | 4.1 | 564.95 | 8.15 | 14.4 | 315.26 / 337.38 | 38.92 | 4.86 | **315.26** | **315.26** |
| Uniform | 128 | 128 | 8.0 | 574.50 | 16.31 | 28.4 | 320.59 / 343.09 | 77.84 | 9.73 | **320.59** | **320.59** |
| Uniform | 256 | 128 | 16.0 | 575.69 | 32.61 | 56.7 | 321.25 / 343.80 | 155.68 | 19.46 | **321.25** | **321.25** |
| Uniform | 512 | 128 | 32.0 | 578.06 | 65.23 | 112.8 | 322.58 / 345.21 | 311.36 | 38.92 | **322.58** | **322.58** |
| Uniform | **1024** | **128** | **64.0** | **582.81** | **130.46** | **223.8** | **325.23 / 348.05** | **622.72** | **77.84** | **622.72** | **325.23** |
| Uniform | 4096 | 128 | 256.0 | 611.32 | 521.84 | 853.6 | 341.14 / 365.08 | 2490.88 | 311.36 | **2490.88** | **341.14** |
| Hot-8 | 64 | 8 | 64.0 | 36.43 | 8.15 | 223.8 | 20.33 / 21.75 | 38.92 | 4.86 | **38.92** | **20.33** |
| Hot-8 | 512 | 8 | 512.0 | 40.58 | 65.23 | 1607.3 | 22.65 / 24.24 | 311.36 | 38.92 | **311.36** | **38.92** |
| Hot-8 | 1024 | 8 | 1024.0 | 45.33 | 130.46 | 2877.7 | 25.30 / 27.07 | 622.72 | 77.84 | **622.72** | **77.84** |

##### (b) `MXFP8`（L20–23）Roofline 参照表

| 分布 | `T` | 去重专家 $U$ | 平均负载 $c=8T/U$ | 有效字节 (MB) | 有用 FLOPs (GF) | AI (F/B) | 带宽下界 Peak/Sust (µs) | A16 算力下界 (µs) | A8 算力下界 (µs) | A16 Roofline (µs) | A8 Roofline (µs) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Uniform | 1 | 8 | 1.0 | 65.70 | 0.13 | 1.9 | 36.66 / 39.24 | 0.61 | 0.15 | **36.66** | **36.66** |
| Uniform | 4 | 29 | 1.1 | 238.17 | 0.51 | 2.1 | 132.91 / 142.23 | 2.43 | 0.61 | **132.91** | **132.91** |
| Uniform | 8 | 52 | 1.2 | 427.07 | 1.02 | 2.4 | 238.32 / 255.04 | 4.86 | 1.22 | **238.32** | **238.32** |
| Uniform | 16 | 82 | 1.6 | 673.49 | 2.04 | 3.0 | 375.83 / 402.20 | 9.73 | 2.43 | **375.83** | **375.83** |
| Uniform | 64 | 126 | 4.1 | 1035.24 | 8.15 | 7.9 | 577.70 / 618.24 | 38.92 | 9.73 | **577.70** | **577.70** |
| Uniform | 256 | 128 | 16.0 | 1053.44 | 32.61 | 31.0 | 587.86 / 629.11 | 155.68 | 38.92 | **587.86** | **587.86** |
| Uniform | 512 | 128 | 32.0 | 1055.82 | 65.23 | 61.8 | 589.18 / 630.53 | 311.36 | 77.84 | **589.18** | **589.18** |
| Uniform | **1024** | **128** | **64.0** | **1060.57** | **130.46** | **123.0** | **591.84 / 633.36** | **622.72** | **155.68** | **622.72** | **591.84** |
| Uniform | 4096 | 128 | 256.0 | 1089.08 | 521.84 | 479.2 | 607.74 / 650.39 | 2490.88 | 622.72 | **2490.88** | **622.72** |
| Hot-8 | 128 | 8 | 128.0 | 66.88 | 16.31 | 243.8 | 37.32 / 39.94 | 77.84 | 19.46 | **77.84** | **37.32** |
| Hot-8 | 512 | 8 | 512.0 | 70.44 | 65.23 | 926.0 | 39.31 / 42.07 | 311.36 | 77.84 | **311.36** | **77.84** |
| Hot-8 | 1024 | 8 | 1024.0 | 75.19 | 130.46 | 1735.0 | 41.96 / 44.91 | 622.72 | 155.68 | **622.72** | **155.68** |

#### 3.2.6 极致性能指标与无预设阈值实测调优要求
- **多候选路线实测收敛（严禁预设 A16↔A4/A8 阈值）**：
  - **`NVFP4_MUL`（`AllowA4`）**至少构建并同台竞速以下候选家族，覆盖 `T = 1..128` 逐点 + `{192, 256, 384, 512, 1024, 2048, 4096}` × `{uniform, reuse, pool, tail, spread}`：
    1. **W4A16 Vector / Warp-Specialized Kernel**（零输入量化开销，直接解码 E2M1×E4M3×matrix_scale，极小负载下冲击 1674.5 GB/s 持续带宽）；
    2. **W4A16 Sliced-K / BF16-MMA Kernel**（验证小/中负载下硬件解包 vs A4 量化成本的真实分界）；
    3. **W4A4 Grouped MMA / TMA Kernel**（`mxf4nvf4` 硬件块缩放指令 + `gate_up` epilogue 原地融合 `SoftcappedSwiGLU → NVFP4 量化写回` + 按 expert 负载分桶的 `short_expert` / `grouped_tma` / `dense_down` 候选）。
  - **`MXFP8`（`AllowA8`）**同理构建并同台竞速：
    1. **W8A16 Vector / Sliced-K Kernel**（无输入量化开销，直接消费 E4M3FN + UE8M0）；
    2. **W8A8 MXFP8 Grouped MMA / TMA Kernel**（`mxf8f6f4` 硬件指令 + `gate_up` epilogue 原地融合 `SoftcappedSwiGLU → MXFP8 动态 G32 UE8M0 量化写回`）。
  - **`A16Only`**：仅允许从 W4A16 / W8A16 候选中按实测最优选择。
- **极致性能目标（RTX 5090 Cold Graph Body，含完整量化/分组/epilogue 开销）**：
  - **Decode 极小批量（`T = 1, 4, 8`，带宽主导）**：
    - `NVFP4_MUL`：有效带宽 **`≥ 1,250–1,450 GB/s`**（`T=1` 目标 **`≤ 28 µs`**，≥ 72% 理论 / ≥ 77% 持续带宽；`T=4, 8` 目标 ≥ 80–85% 持续带宽，即 `T=8` **`≤ 165 µs`**）；
    - `MXFP8`：有效带宽 **`≥ 1,350–1,500 GB/s`**（`T=1` 目标 **`≤ 48 µs`**；`T=8` 目标 **`≤ 290 µs`**，≥ 82–88% 持续带宽）。
  - **中批量（`T = 16..512`，带宽→混合主导）**：
    - 在全 `T` 曲线与所有路由分布（含 `tail` 长尾）上保持 **≥ 75–85% 持续带宽 Roofline 效率**，无跨路由断崖掉速。
  - **高吞吐锚点（`T = 1024, 4096`，`AllowA4 / AllowA8` 释放低比特 MMA）**：
    - `NVFP4_MUL` `T=1024`（Uniform）：受限于 583 MB 全专家读取，`AllowA4` 需打穿 A16 的 623 µs 算力墙，冲击 **`≤ 430–460 µs`**（≥ 72–76% 理想 Roofline）；`T=4096`（Uniform）冲击 **`≤ 520–580 µs`**（有效算力 ≥ 900–1,000 TFLOP/s）；Hot-8 `T=1024` 冲击 **≥ 55–65% NVFP4-F32 算力峰值**；
    - `MXFP8` `T=1024`（Uniform）：目标 **`≤ 760–820 µs`**（≥ 72–78% 理想 Roofline）；`T=4096`（Uniform）冲击 **`≤ 880–960 µs`**（有效算力 ≥ 540–590 TFLOP/s，≥ 65–70% MXFP8-F32 峰值）。
- **Benchmark**：在 `bench/ops/moe/` 新增或扩展 Inkling-2-Mini 专用 benchmark（同时支持 `NVFP4_MUL` 与 `MXFP8`、`A16Only` 与 `AllowA4/AllowA8`、5 种路由分布与逐点 sweep），完整输出 `unique_operand_bytes`、`routed_memory_bound_us`、`allowed_compute_bound_us`、`roofline_efficiency_pct` 与实测选择路线。

---

### E3 · `kv_cache_append_prefix`（Cyclic `capacity=1024`）

#### 3.3.1 现有缺口
- [`src/ops/kv_cache/append/kv_cache_append.cpp`](../../src/ops/kv_cache/append/kv_cache_append.cpp#L140) 与 [`dispatch.cu`](../../src/ops/kv_cache/append/dispatch.cu#L107-L110) 对 `CyclicKVCacheLayerView` 仅注册 `capacity ∈ {2048, 4096}`；Main 40 个逻辑 SWA 层与 DSpark 2 个 SWA 层均为 `capacity = 1024`（`D=128, Hkv=8`）。

#### 3.3.2 公开接口与输入/输出合同
沿用 [`include/ninfer/ops/kv_cache_append.h`](../../include/ninfer/ops/kv_cache_append.h#L67-L82)，仅在头文件注释中增列 `capacity 1024`：
```cpp
void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& counts, KVCacheAppendPrefixExecutionEnvelope envelope,
                            CyclicKVCacheLayerView cache, CudaExecutionView execution);
```
- **输入**：`k, v` 连续 16B 对齐 Device `BF16 [128, 8, T, B]`（`T >= 1, B >= 1`）；`positions` Device `I32 [T, B]`；`counts` Device `I32 [B]`；`envelope` 满足 `0 <= min_count <= counts[b] <= max_count <= min(T, 1024)`。
- **输出（局部 State 修改）**：`cache` 为 `CyclicKVCacheLayerView`（`head_dim=128, num_kv_heads=8, capacity=1024, padded_capacity>=1024`，单层字段，16B 对齐，`k` 为 `BF16 [128, padded_capacity, 8]`，`v` 为 `FP16 [128, padded_capacity, 8]`，通过 Device `I64 [B]` 字节偏移寻址记录）。
- **零提交与隔离合同**：`counts[b] == 0` 行不读 `positions`、`byte_offsets` 或 `k/v`，不写任何 cache 字节；被拒绝尾部 `[counts[b], T)` 与所有未寻址槽位逐位保持不变；零 workspace，支持 CUDA Graph replay。

#### 3.3.3 精确算法定义
对每个 $b \in [0, B)$ 与 $i \in [0, \text{counts}[b])$，设 $p = \text{positions}[i, b] \ge 0$，环形槽位 $s = p \bmod 1024$（位运算 `p & 1023`）：
- `cache.k[b][d, s, h] = k[d, h, i, b]`（**BF16 逐位精确拷贝**）；
- `cache.v[b][d, s, h] = FP16_RNE(BF16(v[d, h, i, b]))`（经精确 FP32 值的 round-to-nearest-even 转 IEEE FP16）。

#### 3.3.4 数值验收标准
- 独立参考实现检查：K 逐 bit 精确一致，V 与 FP32→FP16 RNE 逐 bit 精确一致；环回跨越 `1023 → 0` 边界、`padded_capacity > 1024`（如 `1088 / 1152`）、`counts[b] == 0`、`max_count == 0`、Graph replay 变 `counts/positions/offsets` 全覆盖，未触及内存逐 bit 校验。

#### 3.3.5 RTX 5090 Roofline 理论分析
逻辑字节 $B(B_{\text{batch}}, C) = 8{,}192 \cdot B_{\text{batch}} \cdot C + 4 B_{\text{batch}} C + 12 B_{\text{batch}}$（读 K BF16 + V BF16，写 K BF16 + V FP16，每 token 8 KiB）：

| `(B, C)` | 逻辑字节 | 带宽下界 Peak / Sust (µs) | 主导机制 |
|---|---:|---:|---|
| `(1, 1)` | 0.008 MB | 0.005 / 0.005 | 纯发射延迟 |
| `(1, 8)` | 0.066 MB | 0.037 / 0.039 | 纯发射延迟 |
| `(4, 8)` | 0.262 MB | 0.146 / 0.157 | 发射延迟主导 |
| `(16, 8)` | 1.049 MB | 0.585 / 0.627 | 发射 + L2/DRAM |
| `(1, 128)` | 1.049 MB | 0.585 / 0.627 | 发射 + 带宽 |
| `(1, 1024)` / `(8, 128)` | 8.393 MB | **4.683 / 5.012** | 带宽主导 |

#### 3.3.6 极致性能指标与调优要求
- 128-bit 向量化搬运（单指令 8×BF16 / 8×FP16），审视 RTX 5090 170 SM 下的 CTA 粒度和 `select_cyclic_threads` 策略（避免 `B=1, C=1..8` 启动过多空转线程，同时保证 `C=1024` 吃满内存通道）。
- **目标性能（RTX 5090 Graph Body）**：
  - `(1, 1)` / `(1, 8)`：Warm/Cold **`≤ 2.5–3.2 µs`**（纯发射地板）；
  - `(1, 1024)` / `(8, 128)`（8.39 MB Cold）：目标 **`≤ 6.5–7.2 µs`**（有效带宽 **≥ 1,160–1,300 GB/s**，大批量如 `B=8, C=1024` 达 **≥ 85% 持续带宽 ≈ 1,420+ GB/s**）。
- **Benchmark**：扩展 [`bench/ops/kv_cache_append_bench.cu`](../../bench/ops/kv_cache_append_bench.cu) 支持 `--cyclic-capacity 1024`。

---

### E4 · `rmsnorm_rope`（DSpark Q+K 成对重载，`Hq=24, Hkv=8, D=128`）

#### 3.4.1 现有缺口
- [`src/ops/rope/rmsnorm/rmsnorm_rope.cu`](../../src/ops/rope/rmsnorm/rmsnorm_rope.cu#L118) 成对重载硬编码 `(Hq, Hkv) ∈ {(32, 8), (16, 4)}`；DSpark 为 `(Hq=24, Hkv=8), D=128, theta=1e6, eps=1e-6`。（单 K 重载已原生支持 `Hkv=8`，直接复用。）

#### 3.4.2 公开接口与输入/输出合同
沿用 [`include/ninfer/ops/rmsnorm_rope.h`](../../include/ninfer/ops/rmsnorm_rope.h#L29-L30)，在合同文档新增支持 `(24, 8)`：
```cpp
void rmsnorm_rope(const Tensor& positions, const Tensor& q_norm_weight,
                  const Tensor& k_norm_weight, float epsilon, float theta,
                  Tensor& q, Tensor& k, CudaExecutionView execution);
```
- **输入/输出**：`q` 原地覆写 Device `BF16 [128, 24, W, B]`、`k` 原地覆写 Device `BF16 [128, 8, W, B]`（`W ∈ [2, 16]`, `B >= 1`）；`q_norm_weight / k_norm_weight` Device `BF16 [128]`（直接乘子，可共享 backing）；`positions` Device `I32 [W, B]`；全部连续且 ≥ 4B 对齐（生产中 16B 对齐）；`q` 与 `k` 互不重叠且不与任何只读输入重叠；零 workspace，支持 CUDA Graph replay。

#### 3.4.3 精确算法定义
对每个物理 token $(w, b)$ 与每个头（Q 24 头、K 8 头），**RMSNorm 与 RoPE 融合于单一语义操作，无中间 BF16 舍入边界**：
$$\text{inv} = \frac{1}{\sqrt{\frac{1}{128}\sum_{d=0}^{127} x_d^2 + \varepsilon}}, \qquad n_d = x_d \cdot \text{inv} \cdot \gamma_d$$
$$\phi_i = \text{positions}[w, b] \cdot \theta^{-2i/128}, \quad i \in [0, 64)$$
$$\text{out}_i = \text{RN\_BF16}(n_i \cos\phi_i - n_{i+64}\sin\phi_i), \qquad \text{out}_{i+64} = \text{RN\_BF16}(n_{i+64}\cos\phi_i + n_i\sin\phi_i)$$

#### 3.4.4 数值验收标准
- 独立 FP64 完整公式参考（覆盖 `theta=1e6` 及大位置如 `1,048,560`、非 16B 仅 4B 对齐、共享 `q/k` norm weight、Graph replay 动态 `positions`）。

#### 3.4.5 RTX 5090 Roofline 理论分析
逻辑字节 $B(B_{\text{batch}}, W) = 4 \cdot 128 \cdot (24 + 8) \cdot B_{\text{batch}} W + 512 + 4 B_{\text{batch}} W = 16{,}388 \cdot B_{\text{batch}} W + 512$：

| `(B, W)` (`T=B×W`) | 逻辑字节 | 带宽下界 Peak / Sust (µs) | 主导机制 |
|---|---:|---:|---|
| `(1, 8)` (`T=8`) | 0.132 MB | 0.073 / 0.079 | 纯发射延迟 |
| `(4, 8)` (`T=32`) | 0.525 MB | 0.293 / 0.313 | 发射延迟主导 |
| `(8, 8)` (`T=64`) | 1.049 MB | 0.586 / 0.627 | 发射 + 带宽 |
| `(16, 8)` (`T=128`) | 2.098 MB | 1.171 / 1.253 | 带宽开始显现 |
| `(32, 8)` (`T=256`) | 4.196 MB | **2.341 / 2.506** | 带宽主导 |
| `(64, 16)` (`T=1024`) | 16.782 MB | **9.365 / 10.022** | 纯带宽主导 |

#### 3.4.6 极致性能指标与调优要求
- **2 的幂几何红利**：总头数 `24 + 8 = 32` 精确整除 `HeadsPerBlock ∈ {4, 8, 16, 32}`（彻底避免现有 `(32, 8)=40` 头的 `HeadsPerBlock=20` 整除限制）。需针对 RTX 5090 170 SM 在 `T=8..1024` 实测选择最优 `(HeadsPerBlock, LanesPerHead ∈ {16, 32})`，并共享同 token 内的 `sin/cos` 角度计算（省去 32× 重复 `sincosf`）。
- **目标性能（RTX 5090 Graph Body）**：
  - `(1, 8)`：Warm/Cold **`≤ 2.5–3.2 µs`**；
  - `(32, 8)`（4.20 MB Cold）：**`≤ 4.2 µs`**（有效带宽 ≥ 1,000 GB/s）；
  - `(64, 16)`（16.78 MB Cold）：**`≤ 12.0 µs`**（有效带宽 **≥ 1,400 GB/s**，≥ 84% 持续带宽）。
- **Benchmark**：扩展 [`bench/ops/rmsnorm_rope_bench.cu`](../../bench/ops/rmsnorm_rope_bench.cu) 支持 `--query-heads 24 --key-heads 8` 与 `theta=1e6`。

---

### E5 · `rmsnorm_pack_tail`（DSpark `H=2304` 尾列打包归一化）

#### 3.5.1 现有缺口
- [`src/ops/normalization/rmsnorm/pack_tail.cu`](../../src/ops/normalization/rmsnorm/pack_tail.cu#L116) 仅支持 `H ∈ {4096, 5120}`，且 `static_assert((Hidden / 2) % kBlock == 0)` 要求 `kBlock` 整除 `Hidden / 2`（对 `H=2304`，`Hidden / 2 = 1152 = 9 × 128`）。

#### 3.5.2 公开接口与输入/输出合同
沿用 [`include/ninfer/ops/rmsnorm_pack_tail.h`](../../include/ninfer/ops/rmsnorm_pack_tail.h)，扩展支持 `D = 2304`：
```cpp
void rmsnorm_pack_tail(const Tensor& input, const Tensor& weight, float epsilon,
                       Tensor& output, CudaExecutionView execution);
```
- **输入**：`input` 连续 16B 对齐 Device `BF16 [2304, W, B]`（`W ∈ [2, 16], B >= 1`）；`weight` 连续 16B 对齐 Device `BF16 [2304]`（直接乘子）；`epsilon > 0` 有限（`1e-6f`）。
- **输出**：`output` 连续 16B 对齐 Device `BF16 [2304, g*B]`（`1 <= g < W`，DSpark 中 `W=8, g=7`）。
- **合同**：绝对不读取 anchor 列 `input[:, 0, b]`；三张量两两不重叠；零 workspace，支持 CUDA Graph replay。

#### 3.5.3 精确算法定义
对每个 $b \in [0, B)$ 与 $i \in [1, g]$：
$$\text{output}[d, g \cdot b + i - 1] = \text{RN\_BF16}\!\left(\text{FP32}(\text{input}[d, i, b]) \cdot \text{FP32}(\text{weight}[d]) \cdot \left(\frac{1}{2304}\sum_{j=0}^{2303} \text{FP32}(\text{input}[j, i, b])^2 + \varepsilon\right)^{-1/2}\right)$$

#### 3.5.4 数值验收标准
- 独立 FP64 参考；用 NaN/Inf 填充 anchor 列 `input[:, 0, b]` 验证绝不读取；覆盖 `g < W - 1` 部分尾列截断、`B = 1..64` 与 Graph replay。

#### 3.5.5 RTX 5090 Roofline 理论分析
逻辑字节 $B(B_{\text{batch}}, g) = 2304 \cdot (4 \cdot g \cdot B_{\text{batch}} + 2)$（每输出列读+写各 4,608 B）：

| `(B, W=8, g=7)` (`cols`) | 逻辑字节 | 带宽下界 Peak / Sust (µs) | 主导机制 |
|---|---:|---:|---|
| `(1, 8)` (`7`) | 0.069 MB | 0.039 / 0.041 | 纯发射延迟 |
| `(4, 8)` (`28`) | 0.263 MB | 0.147 / 0.157 | 纯发射延迟 |
| `(8, 8)` (`56`) | 0.521 MB | 0.291 / 0.311 | 发射延迟主导 |
| `(32, 8)` (`224`) | 2.069 MB | 1.155 / 1.236 | 发射 + 带宽 |
| `(64, 8)` (`448`) | 4.133 MB | **2.307 / 2.468** | 带宽主导 |
| `(256, 8)` (`1792`) | 16.519 MB | **9.218 / 9.865** | 纯带宽主导 |

#### 3.5.6 极致性能指标与调优要求
- **128-bit 向量化与精确整除**：`2304` BF16 恰好是 **`288` 个 128-bit `uint4` 向量**（`288 = 9 × 32 = 3 × 96 = 2 × 144`）或 `1152` 个 `bf162`（`9 × 128`）。实测对比 `uint4` 128-bit 加载（如 `kBlock = 96` 3 warps × 3 `uint4`/thread、`kBlock = 144`、`kBlock = 288` 9 warps × 1 `uint4`/thread）与 `bf162` `kBlock = 128`，选择 RTX 5090 上最小延迟、最高带宽实现。
- **目标性能（RTX 5090 Graph Body）**：
  - `(1, 8)` / `(4, 8)`：Warm/Cold **`≤ 2.2–2.8 µs`**；
  - `(64, 8)`（4.13 MB Cold）：**`≤ 4.0 µs`**；
  - `(256, 8)`（16.52 MB Cold）：**`≤ 11.5 µs`**（有效带宽 **≥ 1,430 GB/s**，≥ 85% 持续带宽）。
- **Benchmark**：扩展 [`bench/ops/rmsnorm_pack_tail_bench.cu`](../../bench/ops/rmsnorm_pack_tail_bench.cu) 支持 `--hidden 2304`。

---

### E6 · `sliding_window_attention`（DSpark L0/L2 非因果滑窗注意力）

#### 3.6.1 现有缺口
- [`src/ops/softmax_attention/sliding_window/sliding_window_attention.cpp`](../../src/ops/softmax_attention/sliding_window/sliding_window_attention.cpp#L22-L30) 仅允许 `(32, 8)/W∈{2048, 4096}` 与 `(16, 4)/W=2048`。
- [`src/ops/softmax_attention/kernels/context_query/kernel.cuh`](../../src/ops/softmax_attention/kernels/context_query/kernel.cuh#L19-L21) 硬编码 `kContextQueryGroup = 4`，而 DSpark 几何为 **`(Hq=24, Hkv=8, G=3), D=128, window=1024`**。

#### 3.6.2 公开接口与输入/输出合同
沿用 [`include/ninfer/ops/sliding_window_attention.h`](../../include/ninfer/ops/sliding_window_attention.h)，在合同中新增 `(Hq=24, Hkv=8), D=128, window=1024`：
```cpp
OpGraphProfile sliding_window_attention_graph_profile(AttentionHeadGeometry geometry,
    std::uint32_t window, ContextBucket bucket, std::int32_t batch_size, std::int32_t tokens);
std::size_t sliding_window_attention_workspace_capacity_bytes(AttentionHeadGeometry geometry,
    std::uint32_t window, ContextBucketRange buckets, std::int32_t batch_size,
    std::int32_t min_tokens, std::int32_t max_tokens);
void sliding_window_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
    const Tensor& positions, const Tensor& valid_columns, AttentionHeadGeometry geometry,
    std::uint32_t window, float scale, const ConstCyclicKVCacheLayerView& context,
    ContextBucket bucket, MemoryArena& workspace, Tensor& out, CudaExecutionView execution);
```
- **输入**：`q` `BF16 [128, 24, T, B]`、`query_k / query_v` `BF16 [128, 8, T, B]`（`T ∈ [1, 16], B >= 1`）；`positions` `I32 [T, B]`；`valid_columns` `I32 [B]`（`0 <= valid_columns[b] <= T`）；`scale = 1/√128`；只读 `ConstCyclicKVCacheLayerView`（`capacity=1024, padded_capacity>=1024`, BF16 K / FP16 V）；`bucket ∈ [K2, M1]`。
- **输出**：`out` `BF16 [128, 24, T, B]`；`valid_columns[b] == 0` 的行不读 `positions / offsets / context` 并写零；`t >= valid_columns[b]` 的尾列写零。
- **Graph Profile**：遵守 `OpGraphProfile` 相等即可在扁平 CUDA Graph 内复用切片的合同（且 `window=1024` 时所有 `bucket >= K2` 可共享同一 profile）。

#### 3.6.3 精确算法定义
对每个有效行 $b$（`V = valid_columns[b] > 0`），设基础位置 $p_0 = \text{positions}[0, b]$，历史可见键集合为 $C_b(q_t) = \{k \in [0, p_0) : k \ge \text{positions}[t, b] - 1023\}$（从环形槽 `k & 1023` 读取 BF16 K 与 FP16 V），加上当前块全部有效 query 键 $Q_b = \{0, \dots, V - 1\}$（**非因果全可见**，不掩码未来块内 token）：
$$\text{out}[:, h, t, b] = \text{RN\_BF16}\!\left(\sum_{j \in C_b(q_t) \cup Q_b} \text{softmax}_j\!\left(\frac{q_{t, h} \cdot k_{j, \lfloor h/3 \rfloor}}{\sqrt{128}}\right) v_{j, \lfloor h/3 \rfloor}\right)$$

#### 3.6.4 数值验收标准
- 独立 FP64 参考（逐个验证对称滑窗截断边界 `p - 1023`、环回跨越 `1024` 边界、空历史 `p0 = 0`、短历史 `< 1024`、长历史 `> 1024`、`valid_columns = 0` / 部分有效 / `T = 1..16`、多 batch 异长 `skewed-1long / staggered / with-inert`、Graph replay 与 workspace 高水位）。

#### 3.6.5 RTX 5090 Roofline 理论分析
逻辑字节（历史可见 $L_{\text{vis}} = \min(L_{\text{ctx}}, 1023)$，每历史 token 读 $2 \times 2 \times 128 \times 8 = 4{,}096\text{ B}$，加 Q/qK/qV/O 读写 $16{,}384 \cdot T B$）；有用算力 $F = 4 \cdot 24 \cdot 128 \cdot T (L_{\text{vis}} + T) B$：

| `(B, T, L_ctx)` | 有效字节 (MB) | 有用 FLOPs (GF) | AI (F/B) | 带宽下界 Peak / Sust (µs) | BF16 算力下界 (µs) | 理想 Roofline 下界 (µs) |
|---|---:|---:|---:|---:|---:|---:|
| `(1, 8, 128)` | 0.66 | 0.013 | 20.4 | 0.37 / 0.39 | 0.06 | **0.37** |
| **`(1, 8, 1024)`** | **4.32** | **0.101** | **23.5** | **2.41 / 2.58** | **0.48** | **2.41** |
| `(4, 8, 1024)` | 17.29 | 0.405 | 23.5 | 9.65 / 10.32 | 1.94 | **9.65** |
| **`(8, 8, 1024)`** | **34.57** | **0.811** | **23.5** | **19.29 / 20.65** | **3.87** | **19.29** |
| `(16, 8, 1024)` | 69.14 | 1.622 | 23.5 | 38.58 / 41.29 | 7.74 | **38.58** |
| `(1, 16, 1024)` | 4.45 | 0.204 | 45.9 | 2.48 / 2.66 | 0.98 | **2.48** |
| `(8, 16, 1024)` | 35.62 | 1.634 | 45.9 | 19.88 / 21.27 | 7.80 | **19.88** |

#### 3.6.6 极致性能指标与调优要求
- **模板泛化 `Group = 3`**：将 `context_query/kernel.cuh` 中的 `kContextQueryGroup` 模板化为 `Group ∈ {3, 4}`，精确打包 $G \cdot T$ 行进 `m16n8k16` MMA（关键点 `T=8`: $3 \times 8 = 24$ 行 → 2 个 `M=16` tile；零无效行写越界）。
- **RTX 5090 170 SM 专属 Split-KV 调优**：`W=1024` 仅有 `32` 个 32-token tile/KV-head（`B=1` 共 `8 × 32 = 256` tiles）。需在 RTX 5090 上实测确定 `splits`、`KeyBlock ∈ {32, 64}`、`reduce_warps ∈ {1, 4}` 与 `Direct` vs `SplitKv` 切换（注意避免在 `B=1` 拆出超过有效 tile 数的空 split，或在 `B>=8` 时产生过多 partial 写回与 reduce 流量）。
- **目标性能（RTX 5090 Cold Graph Body）**：
  - `(1, 8, 1024)`（4.32 MB，DSpark 单请求满窗热路径）：目标 **`≤ 6.5–7.5 µs`**（含 split + reduce 两 kernel 发射）；
  - `(4, 8, 1024)`（17.29 MB）：目标 **`≤ 14.0–15.5 µs`**（≥ 67–73% 持续带宽）；
  - `(8, 8, 1024)`（34.57 MB）：目标 **`≤ 25.0–27.0 µs`**（≥ 76–82% 持续带宽）；
  - `(16, 8, 1024)`（69.14 MB）：目标 **`≤ 48.0–51.0 µs`**（≥ 81–86% 持续带宽）。
- **Benchmark**：扩展 [`bench/ops/sliding_window_attention_bench.cu`](../../bench/ops/sliding_window_attention_bench.cu) 支持 `--query-heads 24 --kv-heads 8 --window 1024`。

---

### E7 · `context_softmax_attention`（DSpark L1 非因果全局 PagedKV 注意力）

#### 3.7.1 现有缺口
- [`src/ops/softmax_attention/dense/context/context_softmax_attention.cpp`](../../src/ops/softmax_attention/dense/context/context_softmax_attention.cpp#L19-L28) 硬编码 `kQHeads = 32, kKVHeads = 8`，且 [`plan.h`](../../src/ops/softmax_attention/dense/context/plan.h#L7) 将最大上下文桶限制为 `kContextAttentionMaxBucket = ContextBucket::K256`。DSpark Layer 1 几何为 **`(Hq=24, Hkv=8, G=3), D=128`**，Inkling-2-Mini 架构最大上下文为 `1M`（RTX 5090 首发 BF16 KV 覆盖至 256K，建议将 bucket 校验同步放开至 `M1` 以消除长上下文人为上限）。

#### 3.7.2 公开接口与输入/输出合同
沿用 [`include/ninfer/ops/softmax_attention.h`](../../include/ninfer/ops/softmax_attention.h#L156-L181)，新增注册几何 `(Hq=24, Hkv=8), D=128`：
```cpp
OpGraphProfile context_softmax_attention_graph_profile(AttentionHeadGeometry geometry,
    ContextBucket bucket, std::int32_t batch_size, std::int32_t tokens);
std::size_t context_softmax_attention_workspace_capacity_bytes(AttentionHeadGeometry geometry,
    ContextBucketRange buckets, std::int32_t batch_size,
    std::int32_t min_tokens, std::int32_t max_tokens);
void context_softmax_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
    const Tensor& context_lengths, const Tensor& valid_columns, const Tensor& table_rows,
    AttentionHeadGeometry geometry, float scale, const ConstPagedKVBatchLayerView& context,
    ContextBucket bucket, MemoryArena& workspace, Tensor& out, CudaExecutionView execution);
```
- **输入**：`q` `BF16 [128, 24, T, B]`、`query_k / query_v` `BF16 [128, 8, T, B]`（`T ∈ [1, 16], B >= 1`）；`context_lengths / valid_columns / table_rows` Device `I32 [B]`；只读 `ConstPagedKVBatchLayerView`（`KvCacheStorage::BFloat16`, `head_dim=128, num_kv_heads=8, page_tokens=64`, BF16 K / FP16 V）；`scale = 1/√128`。
- **输出**：`out` `BF16 [128, 24, T, B]`；无效行/无效尾列写零且不访问页表或历史 KV。

#### 3.7.3 精确算法定义
对每个有效行 $b$（`V = valid_columns[b] > 0`），历史可见键集合为经 `table_rows[b]` 映射的全部 committed 历史 $C_b = [0, L_b)$（$L_b = \text{context\_lengths}[b]$），拼接当前块全部有效 query 键 $Q_b = [0, V)$（**非因果全可见**）：
$$\text{out}[:, h, t, b] = \text{RN\_BF16}\!\left(\sum_{j \in C_b \cup Q_b} \text{softmax}_j\!\left(\frac{q_{t, h} \cdot k_{j, \lfloor h/3 \rfloor}}{\sqrt{128}}\right) v_{j, \lfloor h/3 \rfloor}\right)$$

#### 3.7.4 数值验收标准
- 独立 FP64 参考（覆盖 `L = 0, 1, 63, 64, 65, 2048, 32768, 262144`、非连续乱序页表、`table_rows` 共享前缀与交叉映射、异长混合 batch、`valid_columns = 0`、全 bucket `OpGraphProfile` 等价性与 Graph replay）。

#### 3.7.5 RTX 5090 Roofline 理论分析
逻辑字节 $B = 4{,}096 \cdot L_{\text{ctx}} \cdot B_{\text{batch}} + 16{,}384 \cdot T \cdot B_{\text{batch}}$；有用算力 $F = 4 \cdot 24 \cdot 128 \cdot T (L_{\text{ctx}} + T) B_{\text{batch}}$：

| `(B, T, L_ctx)` | 有效字节 (MB) | 有用 FLOPs (GF) | AI (F/B) | 带宽下界 Peak / Sust (µs) | BF16 算力下界 (µs) | 理想 Roofline 下界 (µs) |
|---|---:|---:|---:|---:|---:|---:|
| `(1, 8, 2048)` | 8.52 | 0.20 | 23.7 | 4.75 / 5.09 | 0.96 | **4.75** |
| `(1, 8, 8192)` | 33.69 | 0.81 | 23.9 | 18.80 / 20.12 | 3.85 | **18.80** |
| **`(1, 8, 32768)`** | **134.35** | **3.22** | **24.0** | **74.97 / 80.23** | **15.38** | **74.97** |
| **`(1, 8, 131072)`** | **537.00** | **12.89** | **24.0** | **299.67 / 320.69** | **61.51** | **299.67** |
| `(1, 8, 262144)` | 1073.87 | 25.77 | 24.0 | 599.26 / 641.31 | 123.01 | **599.26** |
| `(4, 8, 32768)` | 537.40 | 12.89 | 24.0 | 299.89 / 320.93 | 61.52 | **299.89** |
| `(8, 8, 32768)` | 1074.79 | 25.78 | 24.0 | 599.77 / 641.86 | 123.04 | **599.77** |
| `(1, 16, 131072)` | 537.13 | 25.77 | 48.0 | 299.74 / 320.77 | 123.02 | **299.74** |

#### 3.7.6 极致性能指标与调优要求
- 与 `E6` 共享模板化的 `Group = 3` `context_query` MMA 核，并针对 **RTX 5090 170 SM** 重新调优各 `(ContextBucket, B, T)` 下的 `split_capacity` 与 `KeyBlock ∈ {32, 64}`（注意 `B=1, L=32K..262K` 时 `Hkv=8`，需足够 `splits` 如 `32..64` 让 `8 × splits ≥ 256` 吃满 170 SM；而 `B>=4` 时收敛 `splits` 压低 partial 读写与 reduce 开销）。
- **目标性能（RTX 5090 Cold Graph Body）**：
  - `(1, 8, 8192)`（33.69 MB）：目标 **`≤ 25.0 µs`**（≥ 80% 持续带宽）；
  - `(1, 8, 32768)`（134.35 MB）：目标 **`≤ 92.0 µs`**（有效带宽 **≥ 1,460 GB/s**，≥ 87% 持续带宽）；
  - `(1, 8, 131072)` / `(1, 8, 262144)`：有效带宽 **≥ 1,500–1,560 GB/s**（**≥ 90–93% 持续读带宽峰值**，`131072` **`≤ 355 µs`**，`262144` **`≤ 705 µs`**）；
  - `(4, 8, 32768)` / `(8, 8, 32768)`：同达 **≥ 90% 持续带宽**。
- **Benchmark**：扩展 [`bench/ops/context_softmax_attention_bench.cu`](../../bench/ops/context_softmax_attention_bench.cu) 支持 `--query-heads 24 --kv-heads 8`（及多 bucket 至 `M1`）。

---

### E8 · `speculative_accept_sparse_drafts`（DSpark 稀疏随机 Proposal 接收）

#### 3.8.1 现有缺口
- [`src/ops/sampling/acceptance/sparse_drafts/entry.cpp`](../../src/ops/sampling/acceptance/sparse_drafts/entry.cpp#L10-L12) 硬编码 `kSparseTokenDomain = 248077` 与 `kSparsePhysicalRows = 248320`。底层 [`launch.cu`](../../src/ops/sampling/acceptance/sparse_drafts/launch.cu) 与 [`plan.h`](../../src/ops/sampling/acceptance/sparse_drafts/plan.h) 本身已参数化 `token_domain` 与 `logits.ne[0]`，但需放开域限制支持 Inkling-2-Mini 的 **`token_domain = 200058, physical_rows = 200064`**（建议直接泛化为任意合法 `1 <= token_domain <= physical_rows`，与 `speculative_accept_greedy_drafts` 一致）并在 RTX 5090 上针对 `V=200058, W=8` 实测调优 `TopKSchedule` / `partial_tile`。

#### 3.8.2 公开接口与输入/输出合同
沿用 [`include/ninfer/ops/speculative_accept.h`](../../include/ninfer/ops/speculative_accept.h#L61-L102)，更新合同注释解除单一 `248077/248320` 限制：
```cpp
std::size_t speculative_accept_sparse_drafts_workspace_capacity_bytes(
    std::int32_t token_domain, SpeculativeAcceptExecutionEnvelope envelope,
    std::int32_t min_drafts, std::int32_t max_drafts,
    std::int32_t min_batch, std::int32_t max_batch);
void speculative_accept_sparse_drafts(
    const Tensor& target_tokens, const Tensor& logits, const Tensor& drafts,
    const Tensor& candidate_ids, const Tensor& proposal_q,
    const Tensor& current_extents, const Tensor& base_positions,
    Tensor& licensed_tokens, Tensor& licensed_counts, std::int32_t token_domain,
    const SamplingConfig* configs, SpeculativeAcceptExecutionEnvelope envelope,
    const Tensor& live_rows, MemoryArena& workspace, CudaExecutionView execution);
```
- **输入**：`logits` `BF16 [physical_rows, W, B]`（Inkling-2-Mini 为 `[200064, 8, B]`，已完成 `softcap_30`；仅 `[0, 200058)` 参与）；`target_tokens` `I32 [W, B]`；`drafts` strided `I32 [D, B]`（`D ∈ [1, 15], W ∈ [2, 16]`，DSpark 主路径 `D=7, W=8`）；`candidate_ids` `I32 [16, D, B]`、`proposal_q` `FP32 [16, D, B]`；`current_extents / base_positions` `I32 [B]`；`configs` Device `SamplingConfig[B]`；可选 `live_rows` `I32 [B]`。
- **输出**：`licensed_tokens` `I32 [W, B]`（接受前缀 + 1 个纠偏/bonus token + 零填充后缀）、`licensed_counts` `I32 [B]`（`A + 1`，非活跃行写 0）。
- **Envelope 合同**：`all_rows_greedy_without_penalties = true` 走零 workspace warp 快速路由；`false` 支持任意贪心/随机/惩罚/mask 混合批次。

#### 3.8.3 精确算法定义
与 [`speculative_accept.h`](../../include/ninfer/ops/speculative_accept.h#L66-L93) 完全一致：
- 仅消费列 $0 \dots P$（$P = \text{current\_extents}[b] \in [0, \min(D, W-1)]$），列 $i$ 叠加前缀草稿 `drafts[0:i, b]` 的 presence/frequency penalty 与 mask 位置 $i$；
- 贪心行（`temperature <= 0`）：比较调整后 target argmax（同分取小 ID）与 `drafts[i, b]`，在首个不匹配处输出纠偏 token，全接受则输出 bonus；
- 随机行（`temperature > 0`）：对目标分布执行 `Top-K (<=20) → min_p → top_p` 归一化得 $p_i$，以 $\min(1, p_i(d_i)/q_i(d_i))$ 接收，首次拒绝从 $\text{normalize}(\max(p_i - q_i, 0))$ 采样（FP32 舍入剩零质量时回退 $p_i$），全接受从 $p_P$ 采样 bonus；RNG 键为 `(seed, base_positions[b] + i + 1, purpose)`。

#### 3.8.4 数值验收标准
- 独立 FP64 参考（复用 `tests/ops/sampling/accept_sparse.cpp` 框架补充 `200058 / 200064` 完整用例，覆盖 raw-greedy、带惩罚/mask 贪心、随机混合、零草稿 `extent=0`、非活跃行、边界词 ID `200057` 与越界 padding 行 `200058..200063` 填 `+inf/NaN` 绝不参与验证、Graph replay）。

#### 3.8.5 RTX 5090 Roofline 理论分析
- Raw-greedy 快速路径：仅读 `target_tokens / drafts / extents`（< 1 KB），纯发射延迟主导。
- 通用/随机路径：主要读取 `logits`（$2 \cdot 200{,}058 \cdot W \cdot B$）及可选 `token_counts`（$+4 \cdot 200{,}058 \cdot B$）：

| `(B, W)` | 仅 Logits 字节 / 带宽下界 Peak/Sust | 含 `token_counts` 字节 / 带宽下界 Peak | 主导机制 |
|---|---:|---:|---|
| `(1, 8)` | 3.20 MB · **1.79 / 1.91 µs** | 4.00 MB · **2.23 µs** | 发射 + 归约开销 |
| `(4, 8)` | 12.80 MB · **7.14 / 7.65 µs** | 16.00 MB · **8.93 µs** | 带宽主导 |
| **`(8, 8)`** | **25.61 MB · 14.29 / 15.29 µs** | **32.01 MB · 17.86 µs** | **带宽主导** |
| `(16, 8)` | 51.21 MB · **28.58 / 30.59 µs** | 64.02 MB · **35.72 µs** | 纯带宽主导 |
| `(8, 16)` | 51.21 MB · **28.58 / 30.59 µs** | 57.62 MB · **32.15 µs** | 纯带宽主导 |

#### 3.8.6 极致性能指标与调优要求
- 核查 RTX 5090 上 `partial_tile`（如 `2048 / 4096 / 8192`）与 `TopKSchedule` 对 `V = 200058` 的两阶段归约效率（`200058 / 4096 = 49` 个 partial block，在 `B=1, W=8` 下产生 `49 × 8 = 392` CTAs，可饱满覆盖 RTX 5090 170 SM）。
- **目标性能（RTX 5090 Cold Graph Body）**：
  - Raw-greedy 路径（`B=1..16, W=8`）：**`≤ 2.5–3.2 µs`**；
  - 通用/随机路径：`(1, 8)` **`≤ 7.5–9.0 µs`**；`(4, 8)` **`≤ 12.5 µs`**；`(8, 8)` **`≤ 20.0–22.0 µs`**（≥ 72–78% 持续带宽）；`(16, 8)` **`≤ 37.0–40.0 µs`**（≥ 78–83% 持续带宽）。
- **Benchmark**：扩展 [`bench/ops/sampling_select_bench.cu`](../../bench/ops/sampling_select_bench.cu) 支持 `--token-domain 200058 --physical-rows 200064 --drafts 7`。

---

### E9 · `rope_2d_interleaved`（Vision 23 层 ViT，Pairwise Interleaved `[W, H]` 2D RoPE）

#### 3.9.1 现有缺口
- [`include/ninfer/ops/rope.h`](../../include/ninfer/ops/rope.h#L9-L24) 现有的 Vision 2D 模式执行 **NeoX split-half** 旋转且轴序为 **`[H, W]`**（前 32 维用 axis 0 = H、后 32 维用 axis 1 = W）。
- Inkling-2-Mini Vision（`Inkling2VisionRotaryEmbedding` + `apply_rotary_pos_emb_vision`）使用完全不同的旋转定义：**Pairwise Interleaved**（相邻奇偶对 `(x[2p], x[2p+1])`）且轴序相反为 **`[W, H]`**（前 32 维用 W，后 32 维用 H）。为避免静默语义混淆，在 [`rope.h`](../../include/ninfer/ops/rope.h) 新增语义明确的独立入口 `rope_2d_interleaved`。

#### 3.9.2 公开接口与输入/输出合同
在 [`include/ninfer/ops/rope.h`](../../include/ninfer/ops/rope.h) 新增独立公开入口：
```cpp
/**
 * Applies full-head pairwise-interleaved 2-D Vision RoPE in place with [W, H] feature order.
 * positions is contiguous Device I32 [N, 2] with axis 0 = height (h) and axis 1 = width (w).
 * q, k are BF16 [D, H, N] (registered D=64, positive H, optional padded token stride >= D*H*2).
 */
void rope_2d_interleaved(const Tensor& positions, float theta, Tensor& q, Tensor& k,
                         CudaExecutionView execution);
void rope_2d_interleaved(const Tensor& positions, float theta, Tensor& x,
                         CudaExecutionView execution);
```
- **输入/输出**：
  - `positions`：连续 Device `I32 [N, 2]`（`positions[t]` 为行坐标 `h_t`，`positions[N + t]` 为列坐标 `w_t`，与现有 `rope.h` / `rmsnorm_rope.h` 的 `[N, 2]` 两轴连续存储约定 100% 一致；对 CLS token 坐标 `(0, 0)` 自然恒等变换）。
  - `q, k`（或单张量 `x`）：Device `BF16 [64, H, N]`（Inkling-2-Mini `H = 16`），特征与头维连续，允许外层 token stride 填充（BF16 对齐；fast path 需 4B/16B 对齐），原地覆写，未触及 padding 逐位保留；`q`、`k`、`positions` 两两不重叠；`theta > 0` 有限（Inkling-2-Mini 默认 `10000.0f`）；`N == 0` 无设备工作；零 workspace，支持 CUDA Graph replay。

#### 3.9.3 精确算法定义
记 `D = 64`、`spatial_dim = D / 2 = 32`，对每个 token $t \in [0, N)$、每个 head 与每个相邻奇偶对 $p \in [0, 32)$：
- 逆频率与角度（**前半 16 对用宽度 `w_t = positions[N + t]`，后半 16 对用高度 `h_t = positions[t]`**）：
  $$\phi_p(t) = \begin{cases} w_t \cdot \theta^{-2p / 32}, & 0 \le p < 16 \\ h_t \cdot \theta^{-2(p - 16) / 32}, & 16 \le p < 32 \end{cases}$$
- 成对旋转（FP32 计算，单次 RN_BF16 写回）：
  $$y[2p] = \text{RN\_BF16}\!\big(x[2p]\cos\phi_p - x[2p+1]\sin\phi_p\big)$$
  $$y[2p+1] = \text{RN\_BF16}\!\big(x[2p+1]\cos\phi_p + x[2p]\sin\phi_p\big)$$

#### 3.9.4 数值验收标准
- 独立 FP64 精确公式参考，重点验证：
  1. `h ≠ w` 非对称坐标下 `[W, H]` 轴序与 pairwise 相邻旋转方向（显式验证不与 NeoX split-half 混淆）；
  2. CLS token `(0, 0)` 精确恒等；
  3. 带 token stride padding 时 padding 字节逐 bit 保留；
  4. `N = 0, 1, 7, 65, 257, 4096` 与成对/单张量重载、非 16B 对齐回退、Graph replay。

#### 3.9.5 RTX 5090 Roofline 理论分析
成对重载（`Q + K`，`H = 16, D = 64`）逻辑字节 $B(N) = 4 \cdot 2 \cdot 16 \cdot 64 \cdot N + 8 N = 8{,}200\,N$：

| `N`（patches，含 CLS） | 逻辑字节 | 带宽下界 Peak / Sust (µs) | 主导机制 |
|---:|---:|---:|---|
| 64 | 0.52 MB | 0.29 / 0.31 | 发射延迟主导 |
| 256 | 2.10 MB | 1.17 / 1.25 | 发射 + 带宽 |
| 1024 | 8.40 MB | **4.69 / 5.01** | 带宽主导 |
| **4096**（单张 1280×1280 图） | **33.59 MB** | **18.74 / 20.06** | **纯带宽主导** |
| 16384（4 图批处理） | 134.35 MB | **74.97 / 80.23** | 纯带宽主导 |

#### 3.9.6 极致性能指标与调优要求
- **天然硬件亲和优势**：每个旋转对 `(x[2p], x[2p+1])` 恰好位于**同一个 32-bit `__nv_bfloat162`** 中，连续 4 对即 **128-bit `uint4`**！当 `q/k` 满足 16B 对齐与 stride 16B 对齐时，**每 8 线程即可 128-bit 合并加载整个 `D=64` 头**（1 个 warp 处理 4 个头），CTA 内仅算一次 32 组 `sincosf` 放共享内存（256 B）复用于 `Q(16) + K(16) = 32` 头，实现零跨步、极低寄存器压力与满带宽写回。
- **目标性能（RTX 5090 Cold Graph Body）**：
  - `N = 256`（2.10 MB）：**`≤ 3.5 µs`**；
  - `N = 1024`（8.40 MB）：**`≤ 6.5 µs`**（≥ 77% 持续带宽）；
  - `N = 4096`（33.59 MB）：**`≤ 22.0–23.0 µs`**（有效带宽 **≥ 1,460–1,525 GB/s**，**≥ 87–91% 持续带宽**）；
  - `N = 16384`（134.35 MB）：**`≤ 86.0–89.0 µs`**（有效带宽 **≥ 1,500–1,560 GB/s**，**≥ 90–93% 持续带宽**）。
- **Benchmark**：扩展 [`bench/ops/axial_rope_bench.cu`](../../bench/ops/axial_rope_bench.cu) 支持 `--op rope-interleaved`（`N ∈ {64, 256, 1024, 4096, 16384}`）。

---

## 4. 交付准出标准（DoD）与实施约定

1. **逐单元独立交付**：每个 Op（其中 `E2` 建议拆为 `E2a: SoftcappedSwiGLU 合同 + NVFP4_MUL RTX 5090` 与 `E2b: MXFP8 RTX 5090` 两个独立单元，其余各 1 单元或按同家族紧耦合合并）均独立完成「合同 → 独立 FP64 单测 → 候选族全景实测调优 → 生产 Selector 收敛 → 全回归验证 → 独立 Conventional Commit」。
2. **禁止预设阈值审计**：代码审查时显式核验 `E2`（NVFP4_MUL `AllowA4` 与 MXFP8 `AllowA8`）的 selector 分界线必须附带 RTX 5090 Cold/Warm 实测 CSV 曲线证据，严禁出现无数据支撑的魔数阈值。
3. **验证与 Roofline 报告**：每个单元通过 `ctest` 全量相关测试（含既有几何零回归），并在验收记录中如实报告 RTX 5090 Cold/Warm Graph Body 中位延迟、有效 GB/s、有用 TFLOP/s 及相对 [`bench/common/roofline.h`](../../bench/common/roofline.h) 的 Roofline 利用率。
