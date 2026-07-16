# Linear Kernel 架构

> 状态：已实现。本文描述当前源码中的 Linear-family 架构，而不是迁移步骤。
>
> 适用目标：当前注册产品 `qwen3_6_27b_rtx5090`。未来 35B shape 只作为扩展压力，
> 不属于本文声明的运行时支持面。

## 1. 结论

当前架构采用两类彼此独立的执行所有权：

1. 纯 `linear()` 只表达单个投影，并在内部按 weight format 分发到
   Q4、Q5、Q6、W8 或预留的 BF16 backend；
2. `linear_add`、`linear_swiglu`、`linear_pair`、`attn_input_proj`、
   `gdn_input_proj`、`gdn_gating_proj` 各自是闭合语义 Op，各自拥有 exact problem、
   route plan、workspace 合同和 fixed execution backend。

调用者只传原有语义输入，不传 target、profile、policy、kernel id 或性能提示。
Linear-family Op 自己负责：

```text
验证语义输入
  -> 构造 exact problem
  -> 检查是否注册
  -> 解析唯一 route plan
  -> 按 plan 申请固定 workspace
  -> 启动 plan 指定的 fixed kernel/composition
```

这条边界解决了旧设计的核心问题：运行 target 不再向 Linear 接口注入调度语义，
也不存在调用者选择 kernel 的第二条控制面。

## 2. 目标与非目标

### 2.1 当前目标

- 保持 Qwen3.6-27B Text、Vision、MTP 的全部已注册 Linear-family 语义；
- 对已注册 exact problem fail closed，不为未注册 shape 提供任意 fallback；
- 让同一 weight format 的 kernel 机制可复用，但让 Q4/Q5/Q6/W8 独立演化；
- 允许融合、成组、成对和显式物化成为一等 route；
- 将 workspace 纳入 route plan，保证 CUDA Graph 地址稳定；
- 让每个固定 route 可以独立做数值、microbenchmark、NSYS、NCU 和 roofline 验收；
- 后续加入相似 shape 时，优先复用已有模板实例和 fixed kernel。

### 2.2 非目标

- 不支持任意 `N/K/T` 的通用 GEMM；
- 不引入运行时 autotune、target profile 或字符串驱动的 policy registry；
- 不承诺未注册 shape 的正确性或性能；
- 不建立通用运行时 epilogue callback/framework；
- 不在本阶段注册 35B runtime target；
- 不把“物理可启动”误写成“已经达到 roofline”。

## 3. 公共语义边界

纯 Linear 的接口保持不变：

```cpp
void linear(const Tensor& x, const Weight& w, Tensor& out,
            WorkspaceArena& ws, cudaStream_t stream);
```

其语义仍然是一个投影：

```text
out[N,T] = Linear(x[K,T], w[N,K])
```

`linear()` 不知道调用者把输出用于 Q、K、V、MLP、MTP 还是 Vision，也不接受外部
调度提示。它只根据 `w.qtype`、weight metadata 和实际 `N/K/padded_K/T` 决定执行。

融合 Op 的公共接口也只包含自身数学合同需要的输入。例如 `linear_add` 接受 residual，
`linear_swiglu` 接受 gate/up 权重，`attn_input_proj` 接受四组权重；它们不暴露内部
route 或 target 信息。

## 4. 统一的内部生命周期

每个 backend 使用相同的控制结构，但不共享一个巨型基类。

### 4.1 Exact problem

Problem 是决定执行所需的最小物理事实，例如：

```cpp
struct Problem {
    int32_t rows;
    int32_t k;
    int32_t padded_k;
    int32_t cols;
};
```

融合 Op 还会加入输出拓扑需要的维度，例如 `gate_up_rows`、`output_rows`、
`query_rows`、`kv_rows` 或 `heads`。Problem 不包含模型层号、调用者角色、target key
或 profile。

### 4.2 Admission

`*_admits(problem)` 只接受显式注册的精确 shape 和 T 域。

- 相似 shape 不自动获得支持；
- 未注册 shape 不落入低速 reference fallback；
- 对齐、payload、plane、stride 和指针要求在 Op 边界检查；
- 无调用者的旧 shape 直接删除。

### 4.3 Route plan

`*_resolve_plan(problem)` 返回唯一闭合计划，通常包含：

- schedule id；
- Full/Predicated 等 fixed variant；
- 必要时的格式子计划；
- exact workspace bytes；
- 有明确标记的 performance-qualified 状态。

Route 表在编译期检查连续性和闭合性。语义 Op 的 `execute_plan()` 会用实际输入重新解析
计划并比较，因此不能执行一个“物理上能启动、但不是该 exact problem 注册路径”的
伪造计划。

格式 backend 还提供更低一层的 fixed candidate executor：它检查 schedule/variant 的
物理合法性，但不会要求该 candidate 是 public pure Linear 的当前赢家。这让
LinearSwiGLU 等语义 Op 可以显式固定一个经过自身测量的格式子计划，同时不扩大
public `linear()` 的 admission。

### 4.4 Fixed execution

执行层只消费已经确定的 plan：

```text
resolve_plan() 负责选择
execute_plan() 负责执行
launch_fixed() 不再二次调度
```

物化组合中的基础投影直接调用格式 backend 的 fixed executor，不会绕回 public
`linear()` 再做一次 dispatch；public pure Linear 仍只能由格式 backend 自己的
exact planner 选择 route。

### 4.5 Workspace

workspace 是 plan 的组成部分，不是 wrapper 的经验公式。

容量查询必须扫描 `[1, max_T]` 内所有 route 端点，因为最佳路线可能不单调。
例如 LinearSwiGLU 在 folded 和 materialized 路线之间多次切换；只查看 `max_T`
对应的 route 会低估容量。

## 5. 源码所有权

当前目录结构为：

```text
src/ops/
  linear/
    linear.cpp
    q4/
    q5/
    q6/
    w8/
    bf16/

  linear_add/q5/
  linear_swiglu/q4/
  linear_pair/w8/
  attn_input_proj/q4_q5/
  gdn_input_proj/q4_q5/
  gdn_gating_proj/bf16/

  common/
    rowsplit_mma.cuh
    rowsplit_grouped_mma.cuh
```

所有权规则：

- `linear/<format>` 拥有纯 Linear 的格式解码、kernel family、候选合法性和 exact route；
- 语义 Op 目录拥有自身的融合/成组 kernel、route 和 workspace；
- `ops/common` 只放稳定的 CUDA 机制，例如 MMA tile 配置、异步搬运和 grouped job mapping；
- common 机制不拥有 shape catalog、route、语义 finalization 或 weight-format dispatch；
- wrapper 只做语义验证并进入对应 backend。

旧的 `src/ops/linear/gemv`、`linear/gemm`、`linear/codec` 和
`linear/common` 兼容组织已经删除。

## 6. 纯 Linear：按 weight format 分发

`linear()` 只做一次 format dispatch：

| Weight format | Backend | 当前状态 |
|---|---|---|
| `Q4G64_F16S` | `linear/q4` | 已注册 exact supports |
| `Q5G64_F16S` | `linear/q5` | 已注册 exact supports |
| `Q6G64_F16S` | `linear/q6` | 已注册 exact supports |
| `W8G32_F16S` | `linear/w8` | 已注册 exact supports |
| `BF16_CTRL` | `linear/bf16` | 格式边界已预留，纯 Linear registry 为空 |

BF16 不能因为存在 `gdn_gating_proj` 的 BF16 kernel 就自动成为通用纯 Linear。
`gdn_gating_proj` 是独立语义 Op；未来只有在真实纯 BF16 Linear problem 被注册并测量后，
`linear/bf16` 才能加入 schedule。

### 6.1 Q4

Q4 使用六个生产 schedule，覆盖 GEMV、SIMT GEMM 和 MMA GEMM 三个模板族：

- `GemvR4W1Direct`
- `GemvR1W8Direct`
- `SimtR8C4`
- `SimtR8C8`
- `MmaR64C64`
- `MmaR64C128`

当前 exact routes：

| `[N,K]` | T 域 | Route |
|---|---:|---|
| `[1024,5120]` | 1 / 2–15 / 16 | R1W8 GEMV / SIMT C4 / SIMT C8 |
| `[4096,5120]` | 1 / 2–4 / 5–16 | R1W8 GEMV / SIMT C4 / SIMT C8 |
| `[6144,5120]` | 1 / 2–7 / 8–16 | R1W8 GEMV / SIMT C4 / SIMT C8 |
| `[34816,5120]` | 2–4 / 5–16 | SIMT C4 / SIMT C8 |
| `[131072,5120]` | 1 | R4W1 direct GEMV |
| `[3456,1152]` | 4–36 / 40–320 / 324–131072，步长 4 | SIMT C4 / MMA C64 / MMA C128 |
| `[4304,1152]` | 4 / 8 / 12 / 16–24 / 28–320 / 324–131072，步长 4 | C4 / C8 / C4 / C8 / MMA C64 / MMA C128 |

`[34816,5120], T=1` 由 `linear_swiglu` 的 paired GEMV 拥有；更大 T 也由该语义 Op
选择 folded 或 materialized 路线，不能重复注册为 public pure Linear 的隐式 fallback。

### 6.2 Q5

Q5 独立拥有：

- `GemvR16S2X`
- `SimtR8C4`
- `SimtR8C8`
- `SimtSplit2Exact`
- `SimtSplit4Exact`
- `MmaR64C64`
- `MmaR64C128`

当前 exact routes：

| `[N,K,Kpad]` | T 域 | Route |
|---|---:|---|
| `[1024,5120,5120]` | 1–4 / 5–16 | SIMT C4 / SIMT C8 |
| `[6144,5120,5120]` | 1 / 2–6 / 7–24 / 25–64 / 65–8388480 | GEMV / Split4 / C8 / MMA C64 / MMA C128 |
| `[5120,6144,6144]` | 2–6 / 7–24 | Split2 / SIMT C8 |
| `[5120,17408,17408]` | 2–6 / 7–24 | Split2 / SIMT C8 |
| `[1152,1152,1152]` | 4 / 8–56 / 60–131072，步长 4 | C4 / C8 / MMA C128 |
| `[1152,4304,4352]` | 4 / 8–84 / 88–131072，步长 4 | C4 / C8 / MMA C128 |

两个 `[5120,*]` residual 投影的 T1 和 T25+ 由 `linear_add` 拥有。

### 6.3 Q6

Q6 保留与 Q4/Q5 不同的独立 schedule 生命周期：

| `[N,K]` | T 域 | Route |
|---|---:|---|
| `[248320,5120]` | 1–6 | SIMT C4 |
| `[1152,1536]` | 4–36 | SIMT C4 |
| `[1152,1536]` | 40–704 | MMA C64 |
| `[1152,1536]` | 708–1088，步长 4 | 按已测 tile-wave 区间在 MMA C128/C64 间切换 |
| `[1152,1536]` | 1092–131072，步长 4 | MMA C128 |

Q6 不与 Q5 共用 route 表或 kernel 文件；二者只复用经过验证的更底层 CUDA 机制。

### 6.4 W8

W8 使用 `SimtR8C4`、`SimtR8C8`、`MmaR32C128` 和 `MmaR64C128`：

| `[N,K]` | T route |
|---|---|
| `[5120,10240]` | 1–4 C4；5–16 C8；17+ MMA R64 |
| `[14336,5120]` | 1–4 C4；5–8 C8；9+ MMA R64 |
| `[1024,5120]` | 1–4 C4；5–16 C8；17+ MMA R32 |
| `[6144,5120]` | 1–4 C4；5–16 C8；17+ MMA R64 |
| `[5120,6144]` | 1–4 C4；5–16 C8；17+ MMA R64 |
| `[34816,5120]` | 1–4 C4；5–8 C8；9+ MMA R64 |
| `[5120,17408]` | 1–4 C4；5–16 C8；17+ MMA R64 |
| `[4608,4608]` | 1–4 C4；5 C8；6–32768 MMA R64 |
| `[5120,4608]` | 1–4 C4；5 C8；6–32768 MMA R64 |

同 shape 的双投影不是两个 pure W8 route 的简单别名，而由 `linear_pair/w8` 独立选择
“两次 SIMT”或“一次 dual MMA”。

## 7. 融合、成组和成对语义 Op

### 7.1 总表

| Op | Exact problem | Route 所有权 | 最大 capacity workspace |
|---|---|---|---:|
| `linear_add` | Q5 `[5120,6144]`、`[5120,17408]` | `linear_add/q5` | 245,760 B |
| `linear_swiglu` | Q4 `[34816,5120] -> [17408,T]` | `linear_swiglu/q4` | 44,564,480 B |
| `linear_pair` | W8 `[1024,5120] x2` | `linear_pair/w8` | 0 |
| `attn_input_proj` | Q4/Q5 四投影 | `attn_input_proj/q4_q5` | 0 |
| `gdn_input_proj` | Q4/Q5 双投影并按行拼接 | `gdn_input_proj/q4_q5` | 327,680 B |
| `gdn_gating_proj` | BF16 `[48,5120] x2` 加 GDN gate 变换 | `gdn_gating_proj/bf16` | 3,145,728 B |

这些 Op 的 plan 都在合法 T 域内闭合；当前 `performance_qualified` 标记到 T=1024。
T1025+ 的终端 route 表示物理可执行和数值合同，不自动表示已经完成 roofline 验收。

### 7.2 LinearAdd / Q5

| T | Schedule | Workspace |
|---:|---|---:|
| 1 | fused residual GEMV | 0 |
| 2–24 | fixed Q5 projection + `residual_add` | `5120*T*2` |
| 25–128 | MMA R64C64 + CTA-collective residual writeback | 0 |
| 129–8388480 | MMA R64C128 + CTA-collective residual writeback | 0 |

这里的 residual 不是任意 callback。它是编译期固定语义：

```text
投影的 BF16 舍入边界
  -> 读取 BF16 residual
  -> FP32 add
  -> BF16 写回
```

### 7.3 LinearSwiGLU / Q4

| T | Schedule | Workspace |
|---:|---|---:|
| 1 | paired-row GEMV | 0 |
| 2–128 | materialized Q4 projection + `silu_mul` | `34816*T*2` |
| 129–256 | folded split-half-pair MMA | 0 |
| 257–384 | materialized | `34816*T*2` |
| 385–512 | folded | 0 |
| 513–640 | materialized | `34816*T*2` |
| 641–8388480 | folded | 0 |

materialized 子计划也固定：

- T2–4：Q4 `SimtR8C4`
- T5–16：Q4 `SimtR8C8`
- T17+：Q4 `MmaR64C128`

非单调 route 是测量结果，不应被压缩成 `SmallT/LargeT` 单阈值。

### 7.4 LinearPair / W8

| T | Schedule |
|---:|---|
| 1–4 | 两次 fixed `SimtR8C4` |
| 5–56 | 两次 fixed `SimtR8C8` |
| 57–8388480 | 一次 `DualMmaR32C128` |

双投影 MMA 的 crossover 是 T56/57，而不是单投影 W8 的 T16/17。
因此 topology 必须进入独立 plan，不能继承 base W8 阈值。

### 7.5 AttnInputProj / Q4+Q5

语义是同一个 `[5120,T]` 输入上的四个独立 BF16 输出：

```text
Q    [6144,T] Q4
gate [6144,T] Q5
K    [1024,T] Q4
V    [1024,T] Q5
```

| T | Schedule |
|---:|---|
| 1–16 | 四个格式 backend 的 fixed subplan |
| 17–8388480 | 一次 Q4 homogeneous pair MMA + 一次 Q5 homogeneous pair MMA |

大 T 仍是两次 launch，因为 Q4 和 Q5 的解码生命周期不同。强行做一个运行时混合格式
大 kernel 会扩大寄存器、分支和实例组合，不是当前测量赢家。

### 7.6 GdnInputProj / Q4+Q5

语义是：

```text
qk  = Q4 Linear [4096,5120] x X
v   = Q5 Linear [6144,5120] x X
qkv = concat_rows(qk, v)  // [10240,T]
```

| T | Schedule | Workspace |
|---:|---|---:|
| 1–16 | 两个 fixed subplan + 两次 D2D 2D copy | `10240*T*2` |
| 17–8388480 | 一次 mixed Q4/Q5 grouped MMA，直接写最终行区间 | 0 |

### 7.7 GdnGatingProj / BF16

这个 Op 不是通用 BF16 Linear。它拥有两个 `[48,5120]` BF16 投影及后续 GDN gate
变换的完整数学合同。

| T | Schedule | Split-K |
|---:|---|---:|
| 1 | paired-row GEMV | 1 |
| 2–8 | Small-T partial + reduce | 10 |
| 9–1024 | cooperative MMA | 8 |
| 1025–2048 | cooperative MMA | 4 |
| 2049–4096 | cooperative MMA | 2 |
| 4097–8388480 | unsplit MMA | 1 |

workspace 是 `split_k * 96 * T * sizeof(float)`；最大值在多个区间都达到
3,145,728 B。

## 8. Epilogue 扩展策略

Linear-family 可以支持 fused epilogue，但边界不是“给 `linear()` 传一个任意 epilogue”。
当前采用三种有限形式：

1. **固定 finalization**：如 LinearAdd，将 residual 语义编译进专用 kernel；
2. **固定 accumulator topology**：如 SwiGLU 的 split-half pair、W8 的 dual projection；
3. **固定 problem map**：如 attention/GDN 的 homogeneous 或 mixed grouped jobs。

显式物化同样是一等 route。当融合 kernel 在某个 T 区间更慢时，plan 可以选择
“fixed projection + 第二个 Op”，而不是为了形式上的融合牺牲性能。

不采用通用运行时 epilogue framework，原因是其代价会落在所有 plain Linear 路径上：

- 额外参数和分支；
- 更大的寄存器/共享内存压力；
- accumulator ownership 被迫统一；
- 模板笛卡尔积和编译产物膨胀；
- 不同语义舍入边界容易被错误合并；
- route 调优矩阵失控。

只有当多个已注册 Op 确实共享同一零成本代码形态时，才把机制下沉到 `ops/common`。

## 9. 新 shape 如何复用 kernel

注册一个相似 shape 时，不根据“看起来相似”直接选 kernel。流程是：

1. 明确 weight format、`N/K/Kpad/T` 域和语义舍入边界；
2. 构造新的 exact problem；
3. 用该 format 的 `candidate_is_legal()` 枚举物理合法的现有 schedule/variant；
4. 运行独立数值 oracle，排除不满足格式和 BF16 边界的候选；
5. 对合法候选做 matched benchmark，覆盖 crossover、tail、wave 和 Full/Predicated；
6. 若已有 kernel 达标，只在 support/route catalog 增加一项；
7. 只有所有现有候选都不能满足性能目标时，才增加新的 compile-time schedule axis；
8. 高影响 shape 再做真实模型端到端和 NSYS/NCU 验收。

因此，未来相似 shape 的底层 kernel 通常来自现有 GEMV/SIMT/MMA 模板实例；新增工作主要是
exact admission 和 route 数据。Q4 的未来 `[131072,2048]` 实验已经证明：
R4 GEMV、C8、C64、C128 现有模板可以覆盖其不同 T 区间，但它仍不会在未注册前被
production 接受。

35B 引入时也遵循同一规则。新 shape 不需要 target 向 Linear 注入 profile；只需要在
对应 format 或新语义 Op 中增加 exact problem、候选证据和闭合 route。

## 10. 复杂度控制

复杂度通过以下边界受控：

- 公共语义接口数量有限且稳定；
- 每个 format/Op 有自己的小型 plan，不建立全局万能 policy 类型；
- route 表只包含真实注册 problem；
- kernel 名称忠实描述执行结构，不使用模型角色命名；
- compile-time 参数只保留能够改变代码生成或性能的轴；
- Q4/Q5/Q6/W8 不因代码相似而强制共用完整 mainloop；
- common 层只共享经过 SASS/resource/timing 验证的机械组件；
- materialized route 不被视为失败或临时兼容；
- 删除无调用者 shape、旧目录和 fallback，而不是继续维护迁移分支。

## 11. Roofline 与性能验收

不存在一个对所有 Linear route 都合理的统一 roofline 百分比：

- T1 常受权重流量和 launch/ownership 限制；
- Small-T 常受重复读权重、occupancy、issue 和 tile 数限制；
- prefill MMA 受 tensor pipe、tile wave、tail 和输出写回限制；
- grouped、paired、folded、residual route 又有不同的数据流。

因此验收单位是固定 `(format, exact problem, schedule, variant, T/wave class)`：

1. 独立数值 oracle；
2. matched microbenchmark 和 crossover；
3. 资源/SASS 检查，确认无意外 spill 或抽象成本；
4. NSYS 证明它是端到端热点；
5. NCU 分析实际 DRAM、SM、tensor pipe、occupancy 和 stall；
6. 高影响切换回到真实 `ninfer_bench` 做 A/B。

当前证据已经支持架构选择，但不声称所有 route 都完成 roofline 收敛：

- Q4 新模板相对旧实现：两个 GEMV 和 C128 持平，C64 约快 1.2%，C4/C8 约快
  12.9%/34.1%，代表性饱和 C128 达到 90.66% SM SOL；
- LinearAdd 的 CTA-collective writeback 相对旧 fused 路径快 1.30%–5.74%，并消除
  大量全局 load/store 指令；
- LinearSwiGLU 的 non-monotonic route 来自完整区间测量，而不是经验阈值；
- W8 Pair 的真实 crossover 为 T56/57，避免了 T17 时约 94.7% 的错误切换回归；
- 同会话端到端消融显示新 Add/SwiGLU route 使 prefill 提升 0.63%–1.38%，TG32
  在 0.16 ms 内不变；
- Release NSYS 显示 Linear/GEMM 占 P128 kernel 时间 93.0%，占 MTP kernel 时间
  77.7%，说明继续逐 route 优化是正确方向；
- GdnGatingProj 所有权迁移前后代表点在测量噪声内：
  T1 9.528→9.393 us，T1024 16.471→16.401 us，T4097 50.068→49.939 us。

完整实验过程和原始命令保存在
[`../archive/optimization-era/plans/2026-07-16-linear-kernel-architecture-experiment-log.md`](../archive/optimization-era/plans/2026-07-16-linear-kernel-architecture-experiment-log.md)。

## 12. 与被否决设计的差异

| 被否决设计 | 当前架构 |
|---|---|
| target 构造 `LinearExecutionProfile` 并传给 Op | Op 从实际输入和 weight format 自主解析 |
| 调用者知道目标 kernel/policy | 调用者只知道语义接口 |
| 一个全局 T1/Small/Large regime | 每个 exact problem 有自己的有限 route 区间 |
| Q4/Q5/Q6/W8 受同一 gemv/gemm 目录牵制 | 每个 format 独立 backend |
| fused op 作为 base Linear 的附属布尔开关 | 每个 fused/grouped/paired Op 自有 plan/backend |
| 只允许 fused 或只允许 materialized | 两者都是可测量、可选择的一等 route |
| workspace 由 wrapper 估算 | workspace 来自同一个 route plan |
| 相似 shape 自动 fallback | 新 shape 先验证、测量，再显式注册 |
| 物理可启动等于支持/达标 | legal、registered、performance-qualified 分开 |
| 保留旧 compatibility 路径 | 原子切换后直接删除旧组织 |

## 13. 如何满足最初需求

- **更合理的 Linear 边界**：接口与 target 无关，Op 全权负责内部最佳调度；
- **保持 27B**：所有当前 Text/Vision/MTP exact problem 均显式注册，并通过真实 artifact
  路径验证；
- **方便以后引入 35B shape**：复用格式模板和候选合法性，只增 exact support/route；
- **控制抽象与实际的边界**：共享机械 CUDA 组件，不共享未经证明的完整生命周期；
- **控制复杂度**：format 与语义 Op 分治，没有全局万能 planner；
- **支持 fused epilogue**：通过固定 finalization/topology/problem-map 扩展，同时保留
  materialized winner；
- **面向 roofline**：每个 fixed route 可独立观测、分析和替换，不再被隐藏二次 dispatch
  干扰；
- **原子切换**：public 语义不变，旧 backend 和兼容目录已删除，没有双轨状态。

## 14. 实现边界提交

本轮迁移按语义边界拆分为：

- `1bcbc24 refactor(linear-add): isolate q5 fused schedules`
- `4c61b06 refactor(linear-swiglu): isolate q4 paired schedules`
- `c1223ed refactor(input-proj): isolate q4-q5 grouped schedules`
- `9fce052 refactor(linear-pair): complete w8 op ownership`
- `0e6ee59 refactor(gdn-gating-proj): isolate bf16 schedules`
- `9b846a0 refactor(linear): neutralize shared mma mechanics`

最终验收要求：

- 默认 Release 全构建和完整 CTest；
- Q4/Q5/Q6/W8 plan、dispatch、candidate 和数值测试；
- 所有 fused/grouped/paired plan 的 route/workspace 边界测试；
- full Linear numerical regression；
- 真实 27B Text/Vision/MTP/prefix 路径；
- 当前端到端 benchmark，并在发生回归时先用 NSYS 定位、再对确定 kernel 使用 NCU；
- `git diff --check`，且 tracked tree 中不存在旧 compatibility 目录。
