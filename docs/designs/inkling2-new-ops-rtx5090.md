# Inkling-2-Mini RTX 5090 全新专属 Op 架构、接口合同与 Roofline 性能规范（N1–N8）

> [!NOTE]
> **范围声明**：本文档冻结 `thinkingmachines/Inkling-2-Mini`（Main Decoder + 3-pass Loop Core + DSpark `mtp.safetensors` + Vision `model.visual`）在 **RTX 5090（`SM120a`）** 上需要新增的 **8 组全新 Op 家族（`N1`–`N8`）** 的公开 C++ 声明、精确数学语义、内存/别名/状态/CUDA Graph 合同、独立 FP64 验证参考、RTX 5090 Roofline 极致性能目标及防基准特化铁律。
>
> **KV Cache 精度要求**：Main Attention（`N4a` + `N4b`）的 **Full Attention（`PagedKVBatchLayerView`）与 Sliding Window Attention（`CyclicKVCacheLayerView`）均完整支持 `KvCacheStorage::BFloat16`（BF16 K + FP16 V）与 `KvCacheStorage::Fp8E4m3fnGroup128Ue8m0`（每 128 维 1 字节 UE8M0 scale + E4M3FN codes，无 Hadamard 变换）两种存储格式**。

---

## 1. 全局硬件参照与工程铁律

### 1.1 RTX 5090（`SM120a`）权威 Roofline 参照

统一沿用 [`bench/common/roofline.h`](../../bench/common/roofline.h) 中维护的 RTX 5090（170 SMs，32 GiB GDDR7，96 MiB L2）基准值：

| 指标 | 数值 | 用途 |
|---|---:|---|
| 理论峰值显存带宽 `memory_gbs` | **1,792.0 GB/s** | 理论下界 `bytes / 1792.0e3` (µs) |
| 实测持续读带宽 `sustained_read_gbs` | **1,674.5 GB/s** | 带宽受限算子真实物理天花板（峰值的 93.4%） |
| BF16 Tensor Core（FP32 累加）`bf16_f32_tflops` | **209.5 TFLOP/s** | Prefill Relative Attention、Loop Message、Markov GEMM |
| FP8 Tensor Core（FP32 累加）`fp8_f32_tflops` | **419.0 TFLOP/s** | 参考 |

### 1.2 防基准特化（Anti-Bench-Overfitting）铁律

> [!IMPORTANT]
> 所有内核、调度表、编译期常量与模板参数必须严格遵守以下边界，并在每次跨模型审查（`gpt-6.2-astra xhigh`）和指挥官 Diff 审计中逐行核查：

1. **允许的模型架构固定 Profile 维度**（与 checkpoint 架构绑定、构成公开合同注册 Profile）：
   - Main/DSpark 隐藏维 `D = 2304`，Vision 隐藏维 `D = 1024`（同时通用 kernel 不应拒绝其他合法 `D`）；
   - Attention 几何 `(Hq = 24, Hkv = 8, G = 3, Dh = 128)`、`d_rel = 16`、`rel_extent = 1024`、`window = 1024`；
   - ShortConv 卷积核宽 `W_conv = 4`（状态宽 `3`）；
   - Loop Message 秩 `r = 64`；Markov Head 秩 `r = 256`。
2. **严禁特化的动态/基准维度**（任何针对下列值硬编码分支、常量折叠或私有快速路径均视为 **P0 阻断缺陷**）：
   - Token 数 / 批次大小 / 序列长度（禁止 `T == 1024`、`T == 4096`、`B == 8`、`kv_len == 32768` 等等值分支）；
   - 词表大小与参数值（禁止硬编码 `V == 200058` / `201024`、`softcap == 30.0f`、`N_stack == 3`、`D_steps == 7`）；
   - 特定 ragged 分段拓扑（禁止为 `4x256`、`1x4096` 等基准分段模式写特殊路径）。
3. **平滑调度原则**：
   - 若基于运行时维度存在路线切换（如 Decode Vector vs Prefill TC MMA、单 Split vs 多 Split-K），切换边界必须为平滑区间阈值（由硬件 occupancy / Roofline 交叉实测决定），不得出现单点等值判断。
4. **强制 Off-Grid / 奇异维度验证与基准抽测**：
   - 每个 Op 的 CTest 必须包含非 2 的幂、素数、奇数和边界尺寸：如 `T ∈ {1, 3, 7, 17, 63, 129, 509, 1021, 2053}`、`B ∈ {1, 3, 5, 13}`、`kv_len ∈ {1, 63, 64, 65, 1023, 1024, 1025, 4099, 16387}`、`V ∈ {1024, 32003, 200058, 201024}`、`softcap ∈ {15.0f, 30.0f, 50.0f}`、`N_stack ∈ {0, 1, 2, 3, 5, 7}`。
   - 每个 Op 的基准验证必须对比标准网格点与相邻 off-grid 点（如 `T=1021` vs `T=1024`，`kv_len=16387` vs `16384`），确认性能平滑无断崖。

### 1.3 通用工程与代码质量规范

- **独立数学参考**：每个 Op 测试均使用独立 FP64（或合同规定的精确 BF16 舍入边界 + FP64 累加）参考实现，严禁用生产 Op 拼接验证融合 Op；测试翻译单元遵循 `ninfer_op_oracle_options`（`-fno-fast-math -ffp-contract=off`）。
- **张量布局约定**：严格遵循 NInfer 列优先（最内维 `ne[0]` 连续）约定，16 字节向量化对齐；所有公开入口按 `docs/maintainer/op-development.md` 校验设备归属、dtype、shape、对齐、非重叠别名与标量参数合法性。
- **零隐藏开销**：所有执行路径零隐藏设备分配、零 Host 回读、零局部内存溢出（`cuobjdump --dump-resource-usage` 验证 `STACK: 0`, `LOCAL: 0`）、零新增编译告警、完整支持 CUDA Graph Capture/Replay。

---

## 2. 八组全新 Op 家族总览与文件组织

| 编号 | Op 家族与公开入口 | 公开头文件 | 源码目录 | 并行波次 | 核心瓶颈 |
|---|---|---|---|---|---|
| **N1** | `block_attn_res` | `include/ninfer/ops/block_attn_res.h` | `src/ops/block_attn_res/` | Wave 1 (`W1-A`) | 带宽受限（单遍多查询复用消除 `Q×` 重复读） |
| **N2** | `short_conv_residual` + 融合 `short_conv_block_attn_res` | `include/ninfer/ops/short_conv.h` | `src/ops/short_conv/` | Wave 1 (`W1-A`) | 发射+带宽受限（消除 96 次/token 发射与中间写回） |
| **N3** | `loop_message_blend_rmsnorm` | `include/ninfer/ops/loop_message.h` | `src/ops/loop_message/` | Wave 1 (`W1-B`) | 小 T 带宽+发射；大 T BF16-TC + 融合消写回 |
| **N4a** | `relative_attention_prepare`（Full & SWA，BF16 & FP8 KV） | `include/ninfer/ops/relative_attention.h` | `src/ops/softmax_attention/relative/prepare/` | Wave 2 (`W2-A`) | 带宽+发射受限（1 launch 融合 K/V 动静卷积+写 KV+Q/R+sink） |
| **N4b** | `relative_softmax_attention`（Full & SWA，BF16 & FP8 KV） | `include/ninfer/ops/relative_attention.h` | `src/ops/softmax_attention/relative/attention/` | Wave 2 (`W2-B`) | Decode 带宽受限（FP8 减半）；Prefill SM120a TC + 片上 rel bias |
| **N5** | `logit_softcap`（FP32 & BF16） | `include/ninfer/ops/logit_softcap.h` | `src/ops/elementwise/logit_softcap.cu` | Wave 1 (`W1-C`) | 纯带宽受限 |
| **N6** | `weighted_sum_rmsnorm` | `include/ninfer/ops/weighted_sum_rmsnorm.h` | `src/ops/normalization/weighted_sum_rmsnorm.cu` | Wave 1 (`W1-C`) | 纯带宽受限 |
| **N7** | `markov_rnn_context` / `markov_rnn_step` / `markov_logits_add_softcap` | `include/ninfer/ops/markov_head.h` | `src/ops/sampling/markov_head/` | Wave 1 (`W1-D`) | `step` 发射受限；`logits_add_softcap` 纯带宽（102 MB 权重） |
| **N8** | `scaled_residual_add` / `scaled_residual_add_layer_norm` | `include/ninfer/ops/scaled_residual_add_layer_norm.h` | `src/ops/normalization/scaled_residual_add_layer_norm.cu` | Wave 1 (`W1-C`) | 纯带宽受限 |

---

## 3. 逐项详细合同、数学定义与 RTX 5090 Roofline 规范

---

### N1 · `block_attn_res`（块注意力残差混合 + 可选输出 RMSNorm）

- **公开头文件**：`include/ninfer/ops/block_attn_res.h`
- **源码目录**：`src/ops/block_attn_res/`（`sources.cmake`, `entry.cpp`, `launch.cu`）
- **测试 / 基准**：`tests/ops/block_attn_res/`（`ninfer_block_attn_res_test`），`bench/ops/block_attn_res_bench.cu`（`ninfer_block_attn_res_bench`）
- **长期扩展性**：同时覆盖 Inkling-2-Mini（`query_scale = 1/sqrt(D)`, `normalize_values = true`, `Q ∈ {1, 3, 4}`）及 Kimi K3 / 通用 Block-AttnRes（`query_scale = 1.0f`, `normalize_values = false`, `Q = 1`）。

#### 3.1.1 公开声明
```cpp
namespace ninfer::ops {

struct BlockAttnResParameters {
    float query_scale     = 1.0f;  // Finite positive; Inkling-2: 1/sqrt(D), Kimi K3: 1.0f
    float rms_eps         = 1e-6f; // Finite positive; source RMSNorm epsilon
    float norm_eps        = 1e-6f; // Finite positive; output RMSNorm epsilon (when norm_weight non-empty)
    bool normalize_values = true;  // true: mix normalized sources V_n=rms(x_n); false: raw x_n
};

/**
 * Block Attention Residual over N = N_stack + 1 sources per column m in [0, M):
 *   x_n[:, m]     = FP32(stack[:, m, n])                                    for n in [0, N_stack)
 *   x_{N_stack}   = FP32(partial[:, m]) + (addend ? FP32(addend[:, m]) : 0) (no intermediate BF16 cast)
 *   inv_rms[n]    = rsqrt((1 / D) * sum_d x_n[d, m]^2 + params.rms_eps)
 *   logit[q, n]   = params.query_scale * (sum_d queries[d, q] * x_n[d, m]) * inv_rms[n]
 *   prob[q, :]    = softmax(logit[q, :])
 *   weight[q, n]  = params.normalize_values ? prob[q, n] * inv_rms[n] : prob[q, n]
 *   mix[d, m, q]  = sum_{n=0..N_stack} weight[q, n] * x_n[d, m]             (FP32)
 *
 * Output boundary:
 *   - If norm_weight is empty: out is contiguous FP32 [D, M, Q] storing mix directly.
 *   - If norm_weight is non-empty (contiguous BF16 [D, Q]): out is contiguous BF16 [D, M, Q]:
 *       out[d, m, q] = RN_BF16(mix[d, m, q] * rsqrt((1/D)*sum_d mix[:,m,q]^2 + params.norm_eps)
 *                              * FP32(norm_weight[d, q])).
 *
 * Operand contract:
 *   - partial: contiguous BF16 [D, M], D > 0, M > 0.
 *   - stack: empty Tensor (N_stack = 0) or contiguous BF16 [D, M, N_stack], N_stack >= 1.
 *   - addend: empty Tensor or contiguous BF16 [D, M].
 *   - queries: contiguous FP32 [D, Q], Q >= 1.
 *   - norm_weight: empty Tensor or contiguous BF16 [D, Q].
 *   - out: contiguous (BF16 if norm_weight non-empty, else FP32) [D, M, Q], disjoint from all inputs.
 *   - All non-empty operands reside on execution.device(), 16B-aligned for D%8==0 (naturally aligned
 *     otherwise). Zero workspace, CUDA Graph capture/replay safe, no hidden allocations.
 */
void block_attn_res(const Tensor& stack, const Tensor& partial, const Tensor& addend,
                    const Tensor& queries, const Tensor& norm_weight,
                    const BlockAttnResParameters& params, Tensor& out,
                    CudaExecutionView execution);

} // namespace ninfer::ops
```

#### 3.1.2 数值验收与 RTX 5090 Roofline 目标
- **独立参考**：FP64 逐列累加 `sum_sq`、`dot`、`exp(logit - max)`、`mix` 与输出 `RMSNorm`，对 BF16 输出校验 ULP / `rtol + atol`（≤1–2 BF16 ULP 或 normwise `<= 3e-3`），对 FP32 raw 输出校验 normwise `<= 1e-5`。
- **关键架构优化**：每个 CTA 处理一个或多个 token 列并**联合计算全部 `Q` 个查询**（`Q ∈ {1, 3, 4}`），`inv_rms[n]` 在全部 `Q` 查询间共享，`stack / partial / addend` 仅从 HBM 读取**一次**（与逐 query 扫描的 2–3× 读相比减少 2×–12× HBM 流量）。对于 `D = 2304`，256 线程每线程仅持 9 个 FP32 寄存器，支持单遍寄存器/SMEM 驻留。
- **Roofline 目标**（`D=2304, N_stack=3, has_addend=true`，逻辑字节 = `((N_stack + 2) * 2 + Q * out_elem) * D * M + queries/weights`）：
  - `M=1, Q=1`（~37 KB）：发射延迟主导，Cold **`≤ 4.0 µs`**（Warm `≤ 3.2 µs`）；
  - `M=8, Q=3`（~260 KB）：Cold **`≤ 5.5 µs`**；
  - `M=1024, Q=1`（28.3 MB，持续下界 16.9 µs）：**`≤ 20.0 µs`**（`≥ 85%` sust BW）；
  - `M=4096, Q=3`（151.0 MB，持续下界 90.2 µs）：**`≤ 105.0 µs`**（`≥ 86%` sust BW）；
  - `M=4096, Q=4, FP32 out`（`N_stack=1, no addend`，113.2 MB，下界 67.6 µs）：**`≤ 79.0 µs`**（`≥ 85%` sust BW）。

---

### N2 · `short_conv`（残差缩放短卷积 + 融合 `short_conv_block_attn_res`）

- **公开头文件**：`include/ninfer/ops/short_conv.h`
- **源码目录**：`src/ops/short_conv/`（`sources.cmake`, `entry.cpp`, `launch.cu`, `fused_attn_res.cu`）
- **测试 / 基准**：`tests/ops/short_conv/`（`ninfer_short_conv_test`），`bench/ops/short_conv_bench.cu`（`ninfer_short_conv_bench`）
- **设计目的**：覆盖 Main Decoder 48 逻辑层 × 2 子层（Attn + MLP）共 **96 次/token** 的隐藏流短卷积，同时支持：
  1. 独立 `short_conv_residual`（无融合或尾部更新）；
  2. 深度融合 `short_conv_block_attn_res`（单 kernel 完成短卷积 + 状态更新 + 可选残差更新 + 写出 `conv_out` + 片上直接消费 `conv_out` 执行下一子层的 `block_attn_res`，彻底消除 96 次/token kernel launch 与 `conv_out` 的 HBM 重读）；
  3. 完整支持 **Batched Direct (`StateFieldView`)**（Decode `W=1`）、**Batched Snapshot (`Tensor& snapshots`)**（DSpark Speculative Verify `W=2..16`）、以及 **`RaggedSequenceView` Packed Prefill**（`N >= 1, S >= 1`）。

#### 3.2.1 公开声明
```cpp
namespace ninfer::ops {

/**
 * Residual-scaled causal depthwise width-4 convolution (oldest tap first):
 *   For sequence window u[c, -3..-1] = state_in[c, 0..2], u[c, t] = x[c, t] for t >= 0:
 *     y_conv[c, t] = RN_BF16((sum_{k=0..3} FP32(weight[c, k]) * FP32(u[c, t - 3 + k])
 *                             + FP32(u[c, t])) * (1.0f / sqrtf(5.0f)))
 *   Optional residual update (matching Inkling-2 reference order):
 *     - If addend is empty (residual must also be empty): conv_out = y_conv.
 *     - If addend non-empty and residual empty:           conv_out = RN_BF16(FP32(addend) + FP32(y_conv)).
 *     - If addend and residual both non-empty:
 *         conv_out = RN_BF16((FP32(residual) + FP32(addend)) + FP32(y_conv)).
 *       residual may exactly alias conv_out.
 *
 * Layouts:
 *   - Batched: x/conv_out/residual/addend are contiguous BF16 [C, W, B], weight is BF16 [C, 4],
 *     state_in/state_out select contiguous BF16 [C, 3] fields via Device I64 [B] byte offsets.
 *     valid_columns is empty (all W live) or Device I32 [B] with 0 <= valid_columns[b] <= W.
 *     Inactive columns j >= valid_columns[b] write exact BF16 zero to conv_out; V_b == 0 reads no
 *     state_in/x/residual/addend and writes no state_out/snapshot.
 *     Direct form writes the final width-3 window after V_b columns to state_out (exact per-row
 *     alias state_out == state_in allowed). Snapshot form writes post-column-j window to
 *     snapshots[:, :, snapshot_base_slots[b] + j] for j < V_b (same contract as causal_conv1d_silu_snapshot).
 *   - Ragged: x/conv_out/residual/addend are contiguous BF16 [C, N] for N = ragged.total_tokens(),
 *     state_in/state_out have Device I64 [S] offsets for S = ragged.num_seqs() (exact same-segment
 *     alias allowed). Each segment convolves independently and writes its trailing width-3 window.
 */
void short_conv_residual(const Tensor& x, const Tensor& weight, ConstStateFieldView state_in,
                         StateFieldView state_out, const Tensor& valid_columns,
                         const Tensor& residual, const Tensor& addend, Tensor& conv_out,
                         CudaExecutionView execution);

void short_conv_residual_snapshot(const Tensor& x, const Tensor& weight,
                                  ConstStateFieldView state_in, Tensor& snapshots,
                                  const Tensor& valid_columns, const Tensor& snapshot_base_slots,
                                  const Tensor& residual, const Tensor& addend, Tensor& conv_out,
                                  CudaExecutionView execution);

void short_conv_residual(const Tensor& x, const Tensor& weight, const RaggedSequenceView& ragged,
                         ConstStateFieldView state_in, StateFieldView state_out,
                         const Tensor& residual, const Tensor& addend, Tensor& conv_out,
                         CudaExecutionView execution);

/**
 * Fused short_conv_residual + block_attn_res:
 *   1. Executes short_conv_residual, publishing BF16 conv_out and updating state_out / snapshots.
 *   2. Feeds the exact published BF16 conv_out as `partial` into block_attn_res(stack, conv_out,
 *      attnres_addend, queries, norm_weight, params, mixed_out), without re-reading conv_out from HBM.
 *
 * Shapes:
 *   - Batched: x/conv_out/conv_residual/conv_addend/attnres_addend are BF16 [D, W, B],
 *     stack is empty or BF16 [D, W, B, N_stack] (Wait: Tensor max 4D! Use flattened column view:
 *     stack is empty or BF16 [D, W*B, N_stack] where column m = b*W + j, and mixed_out is
 *     BF16/FP32 [D, W*B, Q] — identical 3D convention to block_attn_res!).
 *   - Ragged: x/conv_out/conv_residual/conv_addend/attnres_addend are BF16 [D, N],
 *     stack is empty or BF16 [D, N, N_stack], mixed_out is BF16/FP32 [D, N, Q].
 *   - Inactive batched columns (j >= valid_columns[b]) write exact zero to both conv_out and
 *     mixed_out, and do not read stack/attnres_addend or mutate conv state.
 *   - conv_residual may exactly alias conv_out; mixed_out is disjoint from all inputs and conv_out.
 *   - Zero caller workspace; CUDA Graph capture/replay safe.
 */
void short_conv_block_attn_res(const Tensor& x, const Tensor& conv_weight,
                               ConstStateFieldView state_in, StateFieldView state_out,
                               const Tensor& valid_columns, const Tensor& conv_residual,
                               const Tensor& conv_addend, const Tensor& stack,
                               const Tensor& attnres_addend, const Tensor& queries,
                               const Tensor& norm_weight, const BlockAttnResParameters& params,
                               Tensor& conv_out, Tensor& mixed_out, CudaExecutionView execution);

void short_conv_block_attn_res_snapshot(const Tensor& x, const Tensor& conv_weight,
                                        ConstStateFieldView state_in, Tensor& snapshots,
                                        const Tensor& valid_columns,
                                        const Tensor& snapshot_base_slots,
                                        const Tensor& conv_residual, const Tensor& conv_addend,
                                        const Tensor& stack, const Tensor& attnres_addend,
                                        const Tensor& queries, const Tensor& norm_weight,
                                        const BlockAttnResParameters& params, Tensor& conv_out,
                                        Tensor& mixed_out, CudaExecutionView execution);

void short_conv_block_attn_res(const Tensor& x, const Tensor& conv_weight,
                               const RaggedSequenceView& ragged, ConstStateFieldView state_in,
                               StateFieldView state_out, const Tensor& conv_residual,
                               const Tensor& conv_addend, const Tensor& stack,
                               const Tensor& attnres_addend, const Tensor& queries,
                               const Tensor& norm_weight, const BlockAttnResParameters& params,
                               Tensor& conv_out, Tensor& mixed_out, CudaExecutionView execution);

} // namespace ninfer::ops
```
> **重要注记**：`Tensor` 为最多 4 维列优先视图，故 Batched `short_conv_block_attn_res` 中 `stack` 固定为 `[D, W*B, N_stack]`、`mixed_out` 为 `[D, W*B, Q]`（其中展平列 `m = b * W + j`），与 `block_attn_res` 的 `[D, M, N_stack]` / `[D, M, Q]` 完全一致，调用方无需 reshape 拷贝！

#### 3.2.2 数值验收与 RTX 5090 Roofline 目标
- **独立参考**：FP64 逐段递推卷积（按精确 `RN_BF16(y_conv)` 边界），再接入独立 FP64 `block_attn_res` 参考；完整覆盖 `W=1`, `W=2..16`, `valid_columns` 含 `0`/部分/满行、in-place `state_out == state_in`、in-place `conv_residual == conv_out`、ragged 多段混长（含 `L_s = 1, 2, 3, 4, 5, 17, 1021`）、CUDA Graph Replay 与状态逐位隔离。
- **Roofline 目标**（`D=2304`）：
  - `short_conv_block_attn_res` Decode `B=1, W=1, N_stack=3, Q=1`（~65 KB）：单 kernel 融合发射，Cold **`≤ 4.5 µs`**（较分体 2-launch 节省 ~3–4 µs × 96 = **~300–380 µs/step**）；
  - Verify `B=8, W=8`（`M=64, N_stack=3, Q=3`）：Cold **`≤ 9.0 µs`**；
  - Ragged Prefill `N=1024`（`N_stack=3, Q=1`）：**`≥ 85%` sustained BW**；`N=4096`（`4x1024` 与 off-grid `313, 977, 1217, 1589`）：**`≥ 85%` sustained BW**，且与独立 2-op 组合相比总流量与延迟严格下降。

---

### N3 · `loop_message_blend_rmsnorm`（3-Pass 循环核心消息门控混合 + RMSNorm）

- **公开头文件**：`include/ninfer/ops/loop_message.h`
- **源码目录**：`src/ops/loop_message/`（`sources.cmake`, `entry.cpp`, `implementation.h`, `rtx5090/{small_m.cu,cluster_m.cu,large_m.cu,kernel.cuh,mma_tile.cuh}`）
- **测试 / 基准**：`tests/ops/loop_message/`（`ninfer_loop_message_test`），`bench/ops/loop_message_bench.cu`（`ninfer_loop_message_bench`）
- **设计目的**：Inkling-2-Mini Loop Core（layers 6..17 在 pass 1 & 2 执行 24 轮 × 2 次调用 = 48 次/step，每次 `S=3`（Attn Q/K/V）或 `S=1`（MLP））。参考实现需 4 次 launch + 写回 `pair [2D, M, S]` 和 `FP32 gain_forget [2D, M, S]`（`M=4096, S=3` 时 `gain_forget` 单独高达 **226 MB** 读写！）。端到端融合可彻底消除 `pair` 和 `gain_forget` 的 HBM 往返！

#### 3.3.1 公开声明
```cpp
namespace ninfer::ops {

struct LoopMessageWeights {
    // Contiguous BF16 [2*D, r, S] (stream s has matrix down[s] of shape [r, 2*D] in row-major,
    // i.e. ne[0]=2*D contiguous input features, ne[1]=r rank rows, ne[2]=S streams).
    Tensor down;
    // Contiguous BF16 [r, 2*D, S] (stream s has matrix up[s] = [gain_w; forget_w] of shape [2*D, r]
    // in row-major, i.e. ne[0]=r contiguous rank inputs, ne[1]=2*D output rows [0..D)=gain,
    // [D..2D)=forget, ne[2]=S streams).
    Tensor up;
};

[[nodiscard]] std::size_t loop_message_blend_rmsnorm_workspace_capacity_bytes(
    std::int32_t hidden, std::int32_t rank, std::int32_t streams,
    std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * End-to-end fused Loop MessageController + sublayer RMSNorm, batched over S >= 1 streams:
 *   For each stream s in [0, S) and token m in [0, M):
 *     s_d = (S_dest == 1) ? 0 : s
 *     s_n = (S_norm == 1) ? 0 : s
 *     d_vec = dest[:, m, s_d]                              (FP32 [D])
 *     m_vec = message[:, m, s]                             (FP32 [D])
 *     pair[0..D-1]  = RN_BF16(d_vec * rsqrt(mean(d_vec^2) + pair_eps))
 *     pair[D..2D-1] = RN_BF16(m_vec * rsqrt(mean(m_vec^2) + pair_eps))
 *     h[0..r-1]     = RN_BF16(down[s] @ pair)              (BF16 [r] boundary!)
 *     gain_raw[0..D-1]   = (up[s][0..D-1, :]   @ h)        (FP32, NO BF16 boundary)
 *     forget_raw[0..D-1] = (up[s][D..2D-1, :]  @ h)        (FP32, NO BF16 boundary)
 *     blended[d] = (1.0f - tanhf(forget_raw[d])) * d_vec[d] + tanhf(gain_raw[d]) * m_vec[d]
 *     out[d, m, s] = RN_BF16(blended[d] * rsqrt(mean(blended^2) + norm_eps)
 *                            * FP32(norm_weight[d, s_n]))
 *
 * Operand contract:
 *   - Registered RTX 5090 primary profile: D = 2304, r = 64, S >= 1 (general D%8==0, r%16==0 supported).
 *   - dest: contiguous FP32 [D, M, S_dest], S_dest in {1, S}, M >= 1.
 *   - message: contiguous FP32 [D, M, S].
 *   - weights.down: contiguous BF16 [2*D, r, S]; weights.up: contiguous BF16 [r, 2*D, S].
 *   - norm_weight: contiguous BF16 [D, S_norm], S_norm in {1, S}.
 *   - pair_eps, norm_eps: finite positive floats (Inkling-2: 1e-6f).
 *   - out: contiguous BF16 [D, M, S], disjoint from all inputs and workspace.
 *   - All buffers 16B-aligned on execution.device(); CUDA Graph capture/replay safe.
 */
void loop_message_blend_rmsnorm(const Tensor& dest, const Tensor& message,
                                const LoopMessageWeights& weights, const Tensor& norm_weight,
                                float pair_eps, float norm_eps, MemoryArena& workspace,
                                Tensor& out, CudaExecutionView execution);

} // namespace ninfer::ops
```

#### 3.3.2 数值验收与 RTX 5090 Roofline 目标
- **独立参考**：严格复现 `RN_BF16(pair)` 与 `RN_BF16(h)` 两个合同舍入边界，其余点积、`tanh`、blend、RMSNorm 在 FP64 中参考计算；覆盖 `S=1`, `S=3`（含 `S_dest=1, S_norm=1` 广播与 `S_dest=3, S_norm=3` 独立）、`S=2, 4` 泛化验证及 off-grid `M ∈ {1, 3, 7, 17, 63, 129, 509, 1021}`。
- **核心融合优化**：
  - **Decode/Verify（小 `M`）**：`down [4608, 64]` + `up [64, 4608]` 每 stream 共 `1.18 MB`；`S=3` 共 `3.54 MB`（Roofline 下界 **2.1 µs**）。实现 2-stage 或全融合 warp-cooperative 内核，仅在 workspace 暂存极小的 `h [64, M, S]`（`M=8, S=3` 仅 **3 KB**！），完全消除 `pair` 与 `gain_forget` HBM 流量！
  - **Prefill（大 `M`）**：Stage 1（`pair_rms_norm` 融合进 `down` SM120a BF16 TC MMA → 写 `h [r, M, S]`，`M=4096, S=3` 仅 **1.5 MB**！）+ Stage 2（`up` SM120a BF16 TC MMA **epilogue 直接片上执行 `tanh + blend + rmsnorm`** 写出 BF16 `out`，彻底消灭 **226 MB** 的 `FP32 gain_forget` HBM 往返！）。
- **Roofline 目标**（`D=2304, r=64`）：
  - `M=1, S=1`（1.21 MB，下界 0.72 µs）：Cold **`≤ 5.5 µs`**；
  - `M=1, S=3`（3.60 MB，下界 2.15 µs）：Cold **`≤ 7.0 µs`**；
  - `M=8, S=3`（3.86 MB，下界 2.31 µs）：Cold **`≤ 8.5 µs`**；
  - `M=1024, S=3`（逻辑最小字节 75.5 MB，BF16 FLOPs 3.62 GF，下界 45.1 µs）：**`≤ 58 µs`**（`≥ 78%` Roofline）；
  - `M=4096, S=3`（逻辑最小字节 291.5 MB，BF16 FLOPs 14.50 GF，下界 174.1 µs）：**`≤ 215 µs`**（`≥ 81%` Roofline，较未融合版本加速 **`> 2×`**）。

---

### N4a & N4b · Main Relative Attention（`relative_attention_prepare` + `relative_softmax_attention`，**Full & SWA 双支持 BF16 + FP8 KV**）

- **公开头文件**：`include/ninfer/ops/relative_attention.h`
- **共享 Core KV 扩展**（Wave 2 启动前提交到主集成仓）：
  1. [`src/core/paged_kv_storage.h`](../../src/core/paged_kv_storage.h)：允许 `storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0 && head_dim == 128`（每 head-token `128 B` E4M3FN codes + `1 B` UE8M0 scale = `129 B`；`PagedKVBatchLayerView` 原生支持 D128/Hkv=8 FP8 KV，零新枚举！）。
  2. [`src/core/cyclic_kv_cache.h`](../../src/core/cyclic_kv_cache.h) & [`src/core/cyclic_kv_cache.cpp`](../../src/core/cyclic_kv_cache.cpp)：为 `CyclicKVCacheLayerView`、`ConstCyclicKVCacheLayerView`、`CyclicKVCacheLayout` 增加默认向后兼容成员 `k_scales`、`v_scales`（`StateFieldView` / `ConstStateFieldView`，`U8 [head_dim/128, padded_capacity, Hkv]`）和 `KvCacheStorage storage = KvCacheStorage::BFloat16`；`plan_cyclic_kv_cache(..., KvCacheStorage storage = KvCacheStorage::BFloat16)` 支持 `BFloat16`（BF16 K + FP16 V）与 `Fp8E4m3fnGroup128Ue8m0`（FP8 K/V + U8 UE8M0 scales）；既有 BF16-only `kv_cache_append` / `sliding_window_attention` / `context_kv_materialize` 显式校验 `storage == KvCacheStorage::BFloat16`。
- **源码目录**：
  - `N4a`：`src/ops/softmax_attention/relative/prepare/`（`W2-A`）
  - `N4b`：`src/ops/softmax_attention/relative/attention/`（`entry.cpp`, `plan.cpp`, `rtx5090/{decode_launch.cu,ragged_launch.cu,tiled.cuh,device.cuh}`，`W2-B`）
- **测试 / 基准**：
  - `tests/ops/softmax_attention/relative/{test_prepare.cpp,test_attention.cpp,test_chain.cpp}`（`ninfer_relative_attention_prepare_test`、`ninfer_relative_softmax_attention_test`、`ninfer_relative_attention_chain_test`，按 `FORMATS bf16 fp8` 注册 CTest！）
  - `bench/ops/softmax_attention/relative/{prepare.cu,attention.cu}`（`ninfer_relative_attention_prepare_bench`、`ninfer_relative_softmax_attention_bench`）

#### 3.4.1 D128 Relative Attention KV 存储编码合同（BF16 与 FP8）
1. **`KvCacheStorage::BFloat16`**（用于 Full `PagedKVBatchLayerView` 与 SWA `CyclicKVCacheLayerView`）：
   - K：`RN_BF16(k_clipped)` 逐位存入 BF16 `k` 平面；
   - V：`FP16_RNE(v_normed)` 存入 FP16 `v` 平面（与 NInfer 全局 BF16 KV 约定一致）；
   - `k_scales` / `v_scales` 为空。
2. **`KvCacheStorage::Fp8E4m3fnGroup128Ue8m0`**（用于 Full `PagedKVBatchLayerView` 与 SWA `CyclicKVCacheLayerView`）：
   - **无 Hadamard 变换**（Inkling-2 的 K 已经过 L2-clip，V 已经过 weightless RMSNorm，能量高度均衡）；
   - 对每个 128 维 FP32 向量 `x[0..127]`（即 `k_clipped` 或 `v_normed`），严格执行 [`include/ninfer/kv_cache_format.h`](../../include/ninfer/kv_cache_format.h#L35-L37) 规范的 FP8 G128/UE8M0 编解码：
     - `a = max_{d=0..127} |x[d]|`；
     - 若 `a == 0`：`scale_byte = 127`（代表 `1.0`），全部 128 个 code 为 `0`；
     - 否则：`e = max(-127, ceil(log2(a / 448.0)))`，`s = 2^e`，`scale_byte = uint8(e + 127)`（`∈ [0, 254]`），`code[d] = E4M3FN_RNE_SATFINITE(x[d] / s)`；
     - 解码：`decoded[d] = float(E4M3FN(code[d])) * 2^(int(scale_byte) - 127)`（GPU 上 `scale = __uint_as_float(uint32_t(scale_byte) << 23)` 单指令零开销！）。
   - 每 head-token 物理字节：`128 + 1 = 129 B`（K+V 合计 `258 B`，相比 BF16 的 `512 B` 减少 **49.6% 显存占用与 KV 读取流量**！）。

#### 3.4.2 N4a 公开声明（`relative_attention_prepare`）
```cpp
namespace ninfer::ops {

struct RelativeAttentionPrepareParams {
    float qk_clip_threshold        = 22.627416997969522f; // 16 * sqrt(2)
    float rel_clip_threshold       = 8.0f;
    float clip_eps                 = 1e-6f;
    float v_norm_eps               = 1e-6f;
    bool use_log_scaling           = false; // true on Full layers, false on SWA
    std::int32_t log_scaling_floor = 8192;
    float log_scaling_alpha        = 0.1f;
};

struct RelativeAttentionPrepareInputs {
    Tensor q;          // BF16 [128, 24, W, B] (batched) or [128, 24, N] (ragged)
    Tensor r;          // BF16 [16,  24, W, B] or [16,  24, N]
    Tensor k;          // BF16 [128,  8, W, B] or [128,  8, N]
    Tensor v;          // BF16 [128,  8, W, B] or [128,  8, N]
    Tensor q_scale;    // BF16 [128, 24]
    Tensor r_scale;    // BF16 [16,  24]
    // Conv weights: BF16 [1024, 4] (D*Hkv=1024 contiguous).
    // Static mode (k_dyn/v_dyn empty): tap k in 0..3 is oldest-to-newest (lag = 3 - k).
    // Dynamic mode (k_dyn/v_dyn non-empty): tap l in 0..3 is newest-to-oldest (lag = l).
    Tensor k_conv_weight;
    Tensor v_conv_weight;
    Tensor k_dyn;      // Optional BF16 [4, 8, W, B] or [4, 8, N] (lag 0..3); empty => static conv
    Tensor v_dyn;      // Optional BF16 [4, 8, W, B] or [4, 8, N] (must match k_dyn presence)
    Tensor positions;  // Device I32 [W, B] or [N], nonnegative sequential per sequence
    // Optional fused sink projection (for Full layers where x_q split omits 24 sink rows):
    Tensor sink_x;      // Optional BF16 [H, W, B] or [H, N] (empty => disabled)
    Tensor sink_weight; // Optional BF16 [H, 24] (ne[0]=H contiguous, ne[1]=24 heads)
};

struct RelativeAttentionPrepareOutputs {
    Tensor q_out;    // BF16 [128, 24, W, B] or [128, 24, N]
    Tensor r_out;    // BF16 [16,  24, W, B] or [16,  24, N]
    Tensor sink_out; // Optional BF16 [24, W, B] or [24, N] (required iff sink_x non-empty)
};

/**
 * Fused Inkling-2 attention preparation:
 *   1. Q/R per-head scaling, L2 clipping, and optional log-scaling tau(pos):
 *        tau(p)    = use_log_scaling ? (1.0f + alpha * logf(max(1.0f, (p + 1.0f) / floor))) : 1.0f
 *        q_scaled  = FP32(q) * FP32(q_scale)
 *        q_clipped = RN_BF16(q_scaled * min(qk_clip / (||q_scaled||_2 + clip_eps), 1.0f))
 *        q_out     = use_log_scaling ? RN_BF16(FP32(q_clipped) * tau(p)) : q_clipped
 *        (Identical formula for r with r_scale and rel_clip_threshold.)
 *   2. Causal width-4 K/V short convolution (using conv_state [1024, 3] BF16 per K and V):
 *        - Static:  y = RN_BF16((sum_{k=0..3} w[c,k] * u[c, t-3+k] + u[c, t]) / sqrt(5))
 *        - Dynamic: y = RN_BF16((sum_{l=0..3} (w[c,l] + dyn[l, head(c), t]) * u[c, t-l] + u[c, t]) / sqrt(9))
 *      Note the exact RN_BF16(y) boundary before K L2-clip / V RMSNorm.
 *   3. Post-conv K/V normalization & direct KV cache write (BOTH BF16 and FP8 G128/UE8M0):
 *        - K_final = y_k * min(qk_clip / (||y_k||_2 + clip_eps), 1.0f)
 *        - V_final = y_v * rsqrt((1/128) * sum_d y_v[d]^2 + v_norm_eps)
 *      Written directly to CyclicKVCacheLayerView (slot = pos % capacity, capacity = 1024) or
 *      PagedKVBatchLayerView (logical page = pos / 64, slot = pos % 64 via kv_table_rows),
 *      encoded per cache.storage (BFloat16 or Fp8E4m3fnGroup128Ue8m0).
 *   4. Optional fused sink_out[h, t] = RN_BF16(dot(sink_weight[:, h], sink_x[:, t])).
 *
 * State handling:
 *   - Direct batched: writes trailing width-3 K/V windows to k_state_out / v_state_out (exact alias
 *     with k_state_in / v_state_in allowed); inactive columns j >= valid_columns[b] write exact BF16
 *     zero to q_out/r_out/sink_out and do NOT mutate conv state or KV cache.
 *   - Snapshot batched (speculative verify): writes post-column-j windows to k_snapshots / v_snapshots
 *     at snapshot_base_slots[b] + j; writes K/V cache only for live columns j < valid_columns[b].
 *   - Ragged packed prefill: independent per segment s in [0, S), writes trailing width-3 windows to
 *     k_state_out / v_state_out and writes every token's K/V to cache.
 * Zero caller workspace; CUDA Graph capture/replay safe.
 */
// Temporary contiguous current-chunk encoded K/V buffer for SWA multi-token prepare -> attention -> commit:
struct RelativeAttentionChunkKV {
    KvCacheStorage storage = KvCacheStorage::BFloat16;
    Tensor k;
    Tensor v;
    Tensor k_scales;
    Tensor v_scales;
};

// 1) Batched SWA — Direct W==1 (CyclicKVCacheLayerView) & Multi-token / Snapshot (RelativeAttentionChunkKV):
void relative_attention_prepare(const RelativeAttentionPrepareInputs& inputs,
                                const Tensor& valid_columns,
                                ConstStateFieldView k_state_in, StateFieldView k_state_out,
                                ConstStateFieldView v_state_in, StateFieldView v_state_out,
                                CyclicKVCacheLayerView kv_cache,
                                const RelativeAttentionPrepareParams& params,
                                RelativeAttentionPrepareOutputs outputs,
                                CudaExecutionView execution);

void relative_attention_prepare(const RelativeAttentionPrepareInputs& inputs,
                                const Tensor& valid_columns,
                                ConstStateFieldView k_state_in, StateFieldView k_state_out,
                                ConstStateFieldView v_state_in, StateFieldView v_state_out,
                                RelativeAttentionChunkKV chunk_kv,
                                const RelativeAttentionPrepareParams& params,
                                RelativeAttentionPrepareOutputs outputs,
                                CudaExecutionView execution);

void relative_attention_prepare_snapshot(const RelativeAttentionPrepareInputs& inputs,
                                         const Tensor& valid_columns,
                                         const Tensor& snapshot_base_slots,
                                         ConstStateFieldView k_state_in, Tensor& k_snapshots,
                                         ConstStateFieldView v_state_in, Tensor& v_snapshots,
                                         RelativeAttentionChunkKV chunk_kv,
                                         const RelativeAttentionPrepareParams& params,
                                         RelativeAttentionPrepareOutputs outputs,
                                         CudaExecutionView execution);

// 2) Batched Full (PagedKVBatchLayerView, BF16 or FP8) — Direct & Snapshot:
void relative_attention_prepare(const RelativeAttentionPrepareInputs& inputs,
                                const Tensor& valid_columns, const Tensor& kv_table_rows,
                                ConstStateFieldView k_state_in, StateFieldView k_state_out,
                                ConstStateFieldView v_state_in, StateFieldView v_state_out,
                                PagedKVBatchLayerView kv_cache,
                                const RelativeAttentionPrepareParams& params,
                                RelativeAttentionPrepareOutputs outputs,
                                CudaExecutionView execution);

void relative_attention_prepare_snapshot(const RelativeAttentionPrepareInputs& inputs,
                                         const Tensor& valid_columns, const Tensor& kv_table_rows,
                                         const Tensor& snapshot_base_slots,
                                         ConstStateFieldView k_state_in, Tensor& k_snapshots,
                                         ConstStateFieldView v_state_in, Tensor& v_snapshots,
                                         PagedKVBatchLayerView kv_cache,
                                         const RelativeAttentionPrepareParams& params,
                                         RelativeAttentionPrepareOutputs outputs,
                                         CudaExecutionView execution);

// 3) Ragged Packed Prefill — SWA (RelativeAttentionChunkKV) & Full (Paged, BF16 or FP8):
void relative_attention_prepare(const RelativeAttentionPrepareInputs& inputs,
                                const RaggedSequenceView& ragged,
                                ConstStateFieldView k_state_in, StateFieldView k_state_out,
                                ConstStateFieldView v_state_in, StateFieldView v_state_out,
                                RelativeAttentionChunkKV chunk_kv,
                                const RelativeAttentionPrepareParams& params,
                                RelativeAttentionPrepareOutputs outputs,
                                CudaExecutionView execution);

void relative_attention_prepare(const RelativeAttentionPrepareInputs& inputs,
                                const RaggedSequenceView& ragged, const Tensor& kv_table_rows,
                                ConstStateFieldView k_state_in, StateFieldView k_state_out,
                                ConstStateFieldView v_state_in, StateFieldView v_state_out,
                                PagedKVBatchLayerView kv_cache,
                                const RelativeAttentionPrepareParams& params,
                                RelativeAttentionPrepareOutputs outputs,
                                CudaExecutionView execution);

// 4) Commit accepted/surviving SWA chunk K/V into CyclicKVCacheLayerView (slot = pos % 1024):
void relative_attention_commit_cyclic_kv(const RelativeAttentionChunkKV& chunk_kv,
                                         const Tensor& positions, const Tensor& counts,
                                         CyclicKVCacheLayerView kv_cache,
                                         CudaExecutionView execution);

void relative_attention_commit_cyclic_kv(const RelativeAttentionChunkKV& chunk_kv,
                                         const RaggedSequenceView& ragged,
                                         const Tensor& positions, CyclicKVCacheLayerView kv_cache,
                                         CudaExecutionView execution);
```

#### 3.4.3 N4b 公开声明（`relative_softmax_attention`）
```cpp
/**
 * Causal Grouped-Query Attention with Relative Logit Bias + Fused Gate/Sink Epilogue.
 *
 * Registered RTX 5090 profile:
 *   geometry = {head_dim = 128, query_heads = 24, kv_heads = 8} (G = 3),
 *   scale = 1 / sqrt(128), d_rel = 16, rel_extent = 1024, SWA window = 1024,
 *   storage in {KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4m3fnGroup128Ue8m0},
 *   buckets K2 through M1.
 *
 * Public API passes compact `r` [16, 24, ...] and `rel_logits_proj` [16, 1024] (32 KB, L1/SMEM
 * resident) — NEVER a pre-materialized sheared bias tensor.
 *
 * Exact math for each live query (h, i) at absolute position p_i (kvh = h / 3):
 *   - Visible key positions J:
 *       SWA (CyclicKVCacheLayerView, cap=1024): J = [max(0, p_i - 1023), p_i], slot = p_j % 1024
 *       Full (PagedKVBatchLayerView):           J = [0, p_i], page = p_j / 64, slot = p_j % 64
 *   - Decoded K[:, kvh, p_j], V[:, kvh, p_j] follow cache.storage (BF16/FP16 or FP8 G128/UE8M0).
 *   - Distance d = p_i - p_j >= 0:
 *       rel_bias(p_j) = (d < 1024) ? sum_{u=0..15} FP32(r[u, h, i]) * FP32(rel_logits_proj[u, d])
 *                                  : 0.0f
 *   - score(p_j)  = scale * dot(q[:, h, i], K[:, kvh, p_j]) + rel_bias(p_j)
 *   - lse         = log(sum_{j in J} exp(score(p_j)))                      (natural logarithm)
 *   - attn_val    = sum_{j in J} softmax(score)[j] * V[:, kvh, p_j]
 *   - out[:, h, i] = RN_BF16(attn_val * sigmoid(FP32(gate[:, h, i]))
 *                                     * sigmoid(lse - FP32(sink[h, i])))
 *   - Inert columns (i >= valid_columns[b]) write exact BF16 zero and read no cache.
 *   - Cache is read-only (already populated by relative_attention_prepare); out is disjoint from
 *     all inputs, cache, and workspace. CUDA Graph capture/replay safe for batched forms.
 */
struct RelativeAttentionQueryInputs {
    Tensor q;               // BF16 [128, 24, W, B] (batched) or [128, 24, N] (ragged)
    Tensor r;               // BF16 [16,  24, W, B] or [16,  24, N]
    Tensor rel_logits_proj; // BF16 [16, 1024] (d_rel=16 fastest, distance d in [0, 1024) outer)
    Tensor gate;            // BF16 [128, 24, W, B] or [128, 24, N]
    Tensor sink;            // BF16 [24,  W, B] or [24, N]
    Tensor positions;       // Device I32 [W, B] or [N]
};

// 1) Batched Decode/Verify — SWA (Cyclic + optional/required chunk_kv) & Full (Paged, BF16 or FP8):
void relative_softmax_attention(const RelativeAttentionQueryInputs& inputs,
                                const Tensor& valid_columns, AttentionHeadGeometry geometry,
                                std::uint32_t window, float scale,
                                const ConstCyclicKVCacheLayerView& cache,
                                const RelativeAttentionChunkKV& chunk_kv, ContextBucket bucket,
                                MemoryArena& workspace, Tensor& out, CudaExecutionView execution);

void relative_softmax_attention(const RelativeAttentionQueryInputs& inputs,
                                const Tensor& valid_columns, const Tensor& kv_table_rows,
                                AttentionHeadGeometry geometry, float scale,
                                const ConstPagedKVBatchLayerView& cache, ContextBucket bucket,
                                MemoryArena& workspace, Tensor& out, CudaExecutionView execution);

// 2) Ragged Packed Prefill — SWA (Cyclic + chunk_kv) & Full (Paged, BF16 or FP8):
void relative_softmax_attention(const RelativeAttentionQueryInputs& inputs,
                                const RaggedSequenceView& ragged, AttentionHeadGeometry geometry,
                                std::uint32_t window, float scale,
                                const ConstCyclicKVCacheLayerView& cache,
                                const RelativeAttentionChunkKV& chunk_kv, ContextBucket bucket,
                                MemoryArena& workspace, Tensor& out, CudaExecutionView execution);

void relative_softmax_attention(const RelativeAttentionQueryInputs& inputs,
                                const RaggedSequenceView& ragged, const Tensor& kv_table_rows,
                                AttentionHeadGeometry geometry, float scale,
                                const ConstPagedKVBatchLayerView& cache, ContextBucket bucket,
                                MemoryArena& workspace, Tensor& out, CudaExecutionView execution);

// Workspace capacity & CUDA Graph profile queries (covering both BFloat16 & Fp8E4m3fnGroup128Ue8m0):
[[nodiscard]] std::size_t relative_softmax_attention_swa_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, KvCacheStorage storage, std::uint32_t window,
    ContextBucketRange buckets, std::int32_t batch_size, std::int32_t min_tokens,
    std::int32_t max_tokens, std::int32_t multiprocessor_count);

[[nodiscard]] std::size_t relative_softmax_attention_full_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, KvCacheStorage storage, ContextBucketRange buckets,
    std::int32_t batch_size, std::int32_t min_tokens, std::int32_t max_tokens,
    std::int32_t multiprocessor_count);

[[nodiscard]] std::size_t relative_softmax_attention_ragged_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, KvCacheStorage storage, bool is_swa, ContextBucketRange buckets,
    std::int32_t min_tokens, std::int32_t max_tokens, std::int32_t min_segments,
    std::int32_t max_segments, std::int32_t multiprocessor_count);

[[nodiscard]] OpGraphProfile relative_softmax_attention_swa_graph_profile(
    AttentionHeadGeometry geometry, KvCacheStorage storage, std::uint32_t window,
    ContextBucket bucket, std::int32_t batch_size, std::int32_t tokens,
    std::int32_t multiprocessor_count, bool masked = true);

[[nodiscard]] OpGraphProfile relative_softmax_attention_full_graph_profile(
    AttentionHeadGeometry geometry, KvCacheStorage storage, ContextBucket bucket,
    std::int32_t batch_size, std::int32_t tokens, std::int32_t multiprocessor_count,
    bool masked = true);

} // namespace ninfer::ops
```

#### 3.4.4 N4a & N4b RTX 5090 Roofline 目标
- **`N4a relative_attention_prepare`**：
  - Decode `B=1, W=1`（SWA 静态 / Full 动态，BF16 & FP8）：单 launch 融合，Cold **`≤ 5.0–6.0 µs`**；
  - Verify `B=8, W=8`（`M=64`）：Cold **`≤ 10.0 µs`**；
  - Ragged Prefill `N=1024..4096`（BF16 & FP8）：**`≥ 82%` sustained BW**（FP8 写入字节更小，延迟优于或持平 BF16）。
- **`N4b relative_softmax_attention`**：
  - **SWA Decode `B=1, W=1, L=1024`**：
    - BF16 KV（`1024 × 8 × 512 B = 4.19 MB`，下界 `2.50 µs`）：Cold **`≤ 7.5 µs`**；
    - FP8 KV（`1024 × 8 × 258 B = 2.11 MB`，下界 `1.26 µs`）：Cold **`≤ 5.5 µs`**；
  - **SWA Verify `B=8, W=8, L=1024`**：
    - BF16 KV（`33.6 MB`，下界 `20.0 µs`）：Cold **`≤ 28 µs`**（`≥ 71%` sust BW）；
    - FP8 KV（`16.9 MB`，下界 `10.1 µs`）：Cold **`≤ 16 µs`**（较 BF16 加速 **`≥ 1.65×`**）；
  - **Full Decode `B=1, W=1, L=32768`**：
    - BF16 KV（`134.2 MB`，下界 `80.1 µs`）：Cold **`≤ 98 µs`**（`≥ 82%` sust BW）；
    - FP8 KV（`67.6 MB`，下界 `40.4 µs`）：Cold **`≤ 52 µs`**（`≥ 78%` sust BW，较 BF16 加速 **`≥ 1.85×`**！）；
  - **Full Verify `B=8, W=8, L=32768`**：
    - BF16 KV（`1,074 MB`，下界 `641 µs`）：Cold **`≤ 765 µs`**（`≥ 84%` sust BW）；
    - FP8 KV（`541 MB`，下界 `323 µs`）：Cold **`≤ 405 µs`**（`≥ 80%` sust BW，较 BF16 加速 **`≥ 1.85×`**！）；
  - **Ragged Prefill `N=4096`（单段与多段混合，按 RTX 5090 BF16 FP32-accum TC 峰值 `209.5 TFLOP/s` 计）**：
    - Full Causal `1x4096`（`~105.9 GFLOPs`，BF16-TC 物理下界 `505.6 µs`）：SM120a TC FlashAttention + 片上 `r @ rel_logits_proj`（对 `d >= 1024` 块零 bias 开销），目标 BF16 KV **`≥ 60–67%` BF16 TC（`≤ 800 µs`）**，FP8 KV 接近 BF16；
    - SWA Causal `1x4096`（`window=1024`，`~47.9 GFLOPs`，物理下界 `228.7 µs`）：目标 **`≤ 435 µs`**。

---

### N5 · `logit_softcap`（通用元素级双曲正切软截断）

- **公开头文件**：`include/ninfer/ops/logit_softcap.h`
- **源码**：`src/ops/elementwise/logit_softcap.cu`
- **测试 / 基准**：`tests/ops/elementwise/`（并入 `ninfer_elementwise_test` 或独立测试），`bench/ops/logit_softcap_bench.cu`
```cpp
namespace ninfer::ops {
// Elementwise softcap:
//   - FP32: out[i] = softcap * tanh(x[i] / softcap)
//   - BF16: out[i] = RN_BF16(softcap * tanh(FP32(x[i]) / softcap))
// Contiguous FP32 or BF16 matching shapes; softcap > 0 finite; out may exactly alias x.
// Empty tensor performs no device work. Zero workspace, CUDA Graph safe.
void logit_softcap(const Tensor& x, float softcap, Tensor& out, CudaExecutionView execution);
}
```
- **Roofline 目标**：`B=1, V=200058`（BF16 800 KB / FP32 1.6 MB）Cold **`≤ 4.0 µs`**；`B=8, V=201024`（BF16 6.4 MB / FP32 12.9 MB）**`≥ 88%` sustained BW**（`≤ 4.5 µs` / `≤ 8.8 µs`）；`M=64, V=201024`（BF16 51.5 MB / FP32 103 MB）**`≥ 90%` sustained BW**。

---

### N6 · `weighted_sum_rmsnorm`（DFlare 多层特征加权求和 + RMSNorm）

- **公开头文件**：`include/ninfer/ops/weighted_sum_rmsnorm.h`
- **源码**：`src/ops/normalization/weighted_sum_rmsnorm.cu`
- **测试 / 基准**：`tests/ops/normalization/`（并入 `ninfer_normalization_test`），`bench/ops/weighted_sum_rmsnorm_bench.cu`
```cpp
namespace ninfer::ops {
// For each column m in [0, M):
//   mixed[d, m] = sum_{k=0..K-1} FP32(alphas[k]) * FP32(inputs[d, k, m])
//   out[d, m]   = RN_BF16(mixed[d, m] * rsqrt((1/D)*sum_d mixed[d, m]^2 + eps) * FP32(weight[d]))
// inputs: contiguous BF16 [D, K, M] (D > 0, K >= 1, M >= 0; M=0 no-op).
// alphas: contiguous FP32 or BF16 [K]; weight: contiguous BF16 [D] (direct gain); out: BF16 [D, M].
// out disjoint from all inputs; eps > 0 finite; zero workspace; CUDA Graph safe.
void weighted_sum_rmsnorm(const Tensor& inputs, const Tensor& alphas, const Tensor& weight,
                          float eps, Tensor& out, CudaExecutionView execution);
}
```
- **Roofline 目标**（`D=2304, K=4`）：`M=1..8` Cold **`≤ 4.0–4.5 µs`**；`M=1024`（23.6 MB，下界 14.1 µs）**`≤ 16.5 µs`**；`M=4096`（94.4 MB，下界 56.4 µs）**`≤ 65 µs`**（`≥ 88%` sustained BW）。

---

### N7 · `sampling/markov_head`（DSpark 循环 Markov Head 三件套）

- **公开头文件**：`include/ninfer/ops/markov_head.h`
- **源码目录**：`src/ops/sampling/markov_head/`（`sources.cmake`, `entry.cpp`, `context.cu`, `step.cu`, `logits_add_softcap.cu`）
- **测试 / 基准**：`tests/ops/sampling/markov_head.cpp`（`ninfer_markov_head_test`），`bench/ops/markov_head_bench.cu`（`ninfer_markov_head_bench`）

#### 3.7.1 公开声明
```cpp
namespace ninfer::ops {

// 1) Precompute step-invariant f_rnn contribution ONCE per batch row + per-step hidden contribution:
//    base[c, b]           = FP32(bias[c]) + dot(w_frnn[c, :], f_rnn[:, b])
//    context[c, b, step]  = base[c, b] + dot(w_hidden[c, :], hidden[:, step, b])   (FP32)
//    hidden: BF16 [H, D_steps, B], f_rnn: BF16 [H, B], w_hidden/w_frnn: Weight BF16 Contiguous
//    [3*r, H], bias: FP32 [3*r], context: FP32 [3*r, B, D_steps] (step-outer so each step is a
//    contiguous [3*r, B] slice). D_steps >= 1, B >= 1. Registered primary: H=2304, r=256 (general H%8==0, r%16==0).
[[nodiscard]] std::size_t markov_rnn_context_workspace_capacity_bytes(
    std::int32_t hidden, std::int32_t rank, std::int32_t max_steps, std::int32_t max_batch);

void markov_rnn_context(const Tensor& hidden, const Tensor& f_rnn,
                        const Weight& w_hidden, const Weight& w_frnn, const Tensor& bias,
                        MemoryArena& workspace, Tensor& context, CudaExecutionView execution);

inline constexpr std::int32_t kMarkovRnnStepMaxInPlaceRank = 65536;

// 2) Single fused recurrent step (embed gather + [3*r, 2*r] addmm + exact BF16-boundary markov_gate):
//    For each live row b in [0, B):
//      prev_emb = w1_embed[:, prev_tokens[b]]                                   (BF16 [r])
//      raw[0..3r-1] = context_step[:, b] + w_recurrent @ [state_in[:, b]; prev_emb]  (FP32)
//      gate = RN_BF16(raw[0..r-1]), cand = RN_BF16(raw[r..2r-1]), out_r = RN_BF16(raw[2r..3r-1])
//      g    = RN_BF16(1.0f / (1.0f + expf(-FP32(gate))))
//      c    = RN_BF16(tanhf(FP32(cand)))
//      state_out[:, b] = RN_BF16(FP32(RN_BF16(FP32(g) * FP32(state_in[:, b])))
//                              + FP32(RN_BF16(FP32(RN_BF16(1.0f - FP32(g))) * FP32(c))))
//      latent[:, b]    = RN_BF16(tanhf(FP32(out_r)))
//    Inactive rows (live_rows non-empty && live_rows[b] == 0): write exact BF16 zero to state_out
//    and latent without reading prev_tokens / w1_embed / state_in / context_step.
//    state_in/state_out/latent: BF16 [r, B] (state_out may exactly alias state_in);
//    prev_tokens: I32 [B]; context_step: FP32 [3*r, B]; w1_embed: BF16 [r, V_pad];
//    w_recurrent: Weight BF16 Contiguous [3*r, 2*r]; live_rows: empty or I32 [B]. Zero workspace.
void markov_rnn_step(const Tensor& state_in, const Tensor& prev_tokens, const Tensor& context_step,
                     const Tensor& w1_embed, const Weight& w_recurrent, const Tensor& live_rows,
                     Tensor& state_out, Tensor& latent, CudaExecutionView execution);

// 3) Fused low-rank folded LM-head delta + base_logits + softcap:
//    For each live row b in [0, B) and v in [0, token_domain):
//      delta = RN_BF16(dot(folded_weight[v, :], latent[:, b]) + FP32(folded_bias[v]))
//      sum   = RN_BF16(FP32(base_logits[v, b]) + FP32(delta))
//      logits_out[v, b] = RN_BF16(softcap * tanhf(FP32(sum) / softcap))
//    For v in [token_domain, V_phys) or inactive rows (live_rows[b] == 0): exact BF16 zero.
//    latent: BF16 [r, B]; folded_weight: Weight BF16 Contiguous [V_phys, r]; folded_bias: BF16 [V_phys];
//    base_logits/logits_out: BF16 [V_phys, B] (logits_out may exactly alias base_logits);
//    1 <= token_domain <= V_phys; softcap > 0 finite; live_rows: empty or I32 [B]. Zero workspace.
void markov_logits_add_softcap(const Tensor& latent, const Weight& folded_weight,
                               const Tensor& folded_bias, const Tensor& base_logits,
                               std::int32_t token_domain, float softcap, const Tensor& live_rows,
                               Tensor& logits_out, CudaExecutionView execution);

} // namespace ninfer::ops
```

#### 3.7.2 N7 RTX 5090 Roofline 目标
- `markov_rnn_context`（`H=2304, r=256, D_steps=7`，权重 `7.08 MB`）：`B=1`（下界 `4.2 µs`）Cold **`≤ 8.5 µs`**；`B=8`（下界 `4.5 µs`）Cold **`≤ 10.5 µs`**；
- `markov_rnn_step`（`r=256`，权重 `786 KB`）：单 kernel 融合发射，`B=1..8` Cold **`≤ 4.5–5.5 µs`**（每步 7 次循环总计 ≤ 35 µs）；
- `markov_logits_add_softcap`（`V_phys=200064, token_domain=200058, r=256`，权重 **`102.43 MB`**）：
  - `B=1`（`103.6 MB`，持续下界 `61.9 µs`）：Cold **`≤ 72.5 µs`**（`≥ 85%` sustained BW）；
  - `B=4`（`106.0 MB`，持续下界 `63.3 µs`）：Cold **`≤ 74.5 µs`**（`≥ 85%` sustained BW）；
  - `B=8`（`109.2 MB`，持续下界 `65.2 µs`）：Cold **`≤ 76.5 µs`**（`≥ 85%` sustained BW）；
  - `B=16`（`115.6 MB`，持续下界 `69.1 µs`）：Cold **`≤ 82.0 µs`**（`≥ 84%` sustained BW）。

---

### N8 · `scaled_residual_add` & `scaled_residual_add_layer_norm`（Vision LayerScale + LayerNorm）

- **公开头文件**：`include/ninfer/ops/scaled_residual_add_layer_norm.h`
- **源码**：`src/ops/normalization/scaled_residual_add_layer_norm.cu`
- **测试 / 基准**：`tests/ops/normalization/`（并入 `ninfer_normalization_test`），`bench/ops/scaled_residual_add_layer_norm_bench.cu`
```cpp
namespace ninfer::ops {
// LayerScale residual update (exact PyTorch eager BF16 boundaries):
//   r[d, m] = RN_BF16(FP32(residual[d, m]) + FP32(RN_BF16(FP32(update[d, m]) * FP32(gamma[d]))))
// residual/update/residual_out: contiguous BF16 [D, M] (D > 0, M >= 0; M=0 no-op);
// gamma: contiguous BF16 or FP32 [D]. residual_out may exactly alias residual.
// Zero workspace, CUDA Graph safe.
void scaled_residual_add(const Tensor& residual, const Tensor& update, const Tensor& gamma,
                         Tensor& residual_out, CudaExecutionView execution);

// Fused LayerScale residual update + LayerNorm consuming exact published BF16 r:
//   mean = sum_d FP32(r[d, m]) / D; var = sum_d (FP32(r[d, m]) - mean)^2 / D
//   normalized_out[d, m] = RN_BF16((FP32(r[d, m]) - mean) * rsqrt(var + eps)
//                                  * FP32(weight[d]) + FP32(bias[d]))
// weight/bias: contiguous BF16 [D]; normalized_out: BF16 [D, M] disjoint from all inputs and
// residual_out; eps > 0 finite. Zero workspace, CUDA Graph safe.
void scaled_residual_add_layer_norm(const Tensor& residual, const Tensor& update,
                                    const Tensor& gamma, const Tensor& weight, const Tensor& bias,
                                    float eps, Tensor& residual_out, Tensor& normalized_out,
                                    CudaExecutionView execution);
}
```
- **Roofline 目标**（`D=1024`，单遍寄存器驻留）：`M=257`（单图，2.1 MB）Cold **`≤ 4.5 µs`**；`M=1024`（8.4 MB，下界 `5.0 µs`）Cold **`≤ 6.8 µs`**；`M=4096`（33.6 MB，下界 `20.0 µs`）**`≤ 23.0 µs`**（`≥ 88%` sustained BW）。

---

## 4. 并行开发与验收门禁计划

1. **Wave 1（4 独立 Worktree 并发，基于本规范提交 HEAD 切出）**：
   - `W1-A`（`N1 + N2`）：`block_attn_res` + `short_conv` + 融合 `short_conv_block_attn_res`
   - `W1-B`（`N3`）：`loop_message_blend_rmsnorm`
   - `W1-C`（`N5 + N6 + N8`）：`logit_softcap` + `weighted_sum_rmsnorm` + `scaled_residual_add[_layer_norm]`
   - `W1-D`（`N7`）：`sampling/markov_head`（3 个入口）
2. **Pre-Wave-2 共享合同提交**（在 Wave 2 切出前提交至主集成仓，零冲突共享）：
   - `src/core/paged_kv_storage.h`（放行 D128 `Fp8E4m3fnGroup128Ue8m0`）
   - `src/core/cyclic_kv_cache.{h,cpp}`（增加 `k_scales`/`v_scales`/`storage` 的 BF16 + FP8 支持）+ 既有 BF16-only cyclic Op 显式守卫 + core 单元测试
   - `include/ninfer/ops/relative_attention.h`（冻结 `N4a` + `N4b` 全部公开声明）
3. **Wave 2（2 独立 Worktree 并发，随 Wave 1 槽位释放滚动启动）**：
   - `W2-A`（`N4a`）：`relative_attention_prepare`（Full & SWA，BF16 & FP8 KV，Batched Direct/Snapshot & Ragged）
   - `W2-B`（`N4b`）：`relative_softmax_attention`（Full & SWA，BF16 & FP8 KV，Decode/Verify Split-K CUDA Graph & Ragged SM120a TC FlashAttention）
4. **每单元强制合入门禁**：
   - 独立 FP64 参考测试全过（含 off-grid、边界、别名、零长、CUDA Graph Replay、`compute-sanitizer --tool memcheck` 0 错）
   - `cuobjdump --dump-resource-usage` 验证 `STACK: 0, LOCAL: 0`；编译零新增告警
   - RTX 5090 CUDA Graph Body（Cold + Warm）实测达 Roofline 目标且 off-grid 平滑
   - 强制 `gpt-6.2-astra xhigh` 跨模型只读复审 + 指挥官防基准特化人工审计全部清零
   - 合入主集成仓后由指挥官独立重新编译并在 RTX 5090 上二次复验（Wave 2 合入后追加端到端 `relative_attention_prepare -> relative_softmax_attention` 串联测试覆盖 SWA/Full × BF16/FP8 × Batched/Ragged）。
