# ReplaySSM：GDN speculative state 的 raw-input replay

本文讨论 Gated DeltaNet（GDN）在短窗口 speculative decoding 中的 ReplaySSM：
target verify 保留原有的逐 token recurrence，但不保存每个 verify position 的完整 recurrent
state；它只记录驱动状态转移的 raw inputs。最终接受长度确定后，再从 committed checkpoint
顺序重放 accepted prefix，得到下一轮 state。

这里的关键问题不只是“能否由 record 重建 state”。GDN recurrence 可以写出多种代数等价形式，
但不同形式会改变归一化、reduction、乘加结合、Tensor Core 精度和 cast boundary。State 是跨轮持久
值；很小的重建误差也会进入下一轮并继续传播。因此，本技术的核心要求是：

> Replay fold 必须执行与 verify recurrence 相同的有限精度状态转移，而不仅是在实数域中计算一个
> 等价公式。

本文依次说明状态和 record 的数学定义、accepted-prefix replay、浮点漂移的来源、closed-loop
bitwise clone 的条件、causal-conv history，以及当前 Qwen3.5 模型实例的空间与计算特征。

Record/Fold 的实现与合同见 [`gdn_replay.h`](../../include/ninfer/ops/gdn_replay.h) 和
[`replay.cpp`](../../src/ops/linear_attention/gated_delta_net/replay.cpp)。模型配置决定 layer/head
数量，Program 按启用的 MTP、DFlash 或 DFlash2 窗口预留 record capacity。

---

## 1. 问题：speculative verify 需要可选择的状态前缀

### 1.1 GDN state 的尺寸

对每个 GDN layer 和 value head，recurrent state 是一个 FP32 矩阵

\[
S\in\mathbb{R}^{V\times K}.
\]

当前实例使用 \(K=V=128\)。一个 value head 的 state 包含 16,384 个 FP32 元素，即 64 KiB。
乘上全部 GDN layers 和 value heads，一份完整 recurrent state image 为：

| 模型 | GDN layers | value heads | 一份 recurrent state |
|---|---:|---:|---:|
| Qwen3.6/3.8-27B | 48 | 48 | 144 MiB |
| Qwen3.6-35B-A3B | 30 | 32 | 60 MiB |

### 1.2 Snapshot baseline

设一次 target verify 处理 \(T\) 个 inputs，并从 committed state \(S_0\) 顺序得到

\[
S_1,S_2,\ldots,S_T.
\]

最终只会提交其中一个 prefix \(S_m\)。直接支持 rollback 的方法是把每个 state 都保存下来：

~~~text
S0 ── input 1 ──> S1 ── input 2 ──> ... ── input T ──> ST
                    │                    │                  │
                    └──────── 保存完整 state trajectory ──┘

final commit length = m：选择 Sm
~~~

这会为每个 verify position 写一份 \(V\times K\) FP32 matrix。窗口增加一列，就增加一份完整
state 的容量和写流量。

### 1.3 Raw-input ReplaySSM

Raw-input ReplaySSM 只保留：

\[
\text{committed checkpoint }S_0
\quad+\quad
R_1,R_2,\ldots,R_T,
\]

其中 \(R_t\) 是第 \(t\) 个 state transition 的紧凑 record。Verify 仍然产生全部 outputs，但不把
\(S_1,\ldots,S_T\) 写入持久存储。最终 \(m\) 已知后计算

\[
S_m=F_{\mathrm{fp}}\left(
F_{\mathrm{fp}}\left(
\cdots F_{\mathrm{fp}}(S_0,R_1),R_2
\right)\cdots,R_m
\right),
\]

并只发布这一份 state。

这里 \(F_{\mathrm{fp}}\) 特意表示实际有限精度 transition，而不是抽象实数公式。后文会说明这一区别
为何决定了 reconstructed state 能否与 verify trajectory 对齐。

---

## 2. GDN recurrence 与 raw transition record

### 2.1 有限精度路径所对应的逻辑顺序

以下省略 layer 和 value-head 下标。设

\[
q_t^{raw},k_t^{raw}\in\mathbb{R}^{K},
\qquad
v_t\in\mathbb{R}^{V}.
\]

Query 和 key 先做 L2 normalization：

\[
\bar q_t=
\frac{q_t^{raw}}
{\sqrt{\sum_i(q_{t,i}^{raw})^2+\epsilon}},
\qquad
\bar k_t=
\frac{k_t^{raw}}
{\sqrt{\sum_i(k_{t,i}^{raw})^2+\epsilon}}.
\]

令

\[
\alpha_t=\exp(g_t).
\]

一次 GDN transition 可以按以下顺序写出：

\[
S_t^{decay}=\alpha_tS_{t-1},
\]

\[
r_t=S_t^{decay}\bar k_t,
\]

\[
u_t=\beta_t(v_t-r_t),
\]

\[
S_t=S_t^{decay}+u_t\bar k_t^{\mathsf T},
\]

\[
y_t=c\,S_t\bar q_t,
\qquad
c=\frac{1}{\sqrt{128}}.
\]

合并后得到常见的紧凑写法：

\[
u_t=\beta_t
\left(v_t-\alpha_tS_{t-1}\bar k_t\right),
\]

\[
S_t=\alpha_tS_{t-1}+u_t\bar k_t^{\mathsf T}.
\]

这两组公式在实数域中完全相同；第一组同时展示了实际 transition 中需要保持一致的运算边界：
先 decay state，再从 decayed state 读取 correction，最后做 rank-one update。

### 2.2 Grouped q/k heads

若 \(H_v\) 个 value heads 共享 \(H_{qk}\) 个 q/k heads，令

\[
G=H_v/H_{qk},
\qquad
h_q=\lfloor h/G\rfloor.
\]

Value head \(h\) 使用 q/k head \(h_q\)。每个 value head 有独立的 \(S\)、\(v\)、\(g\) 和
\(\beta\)，同一 group 共享 raw \(q/k\)。

### 2.3 State transition 的充分输入

给定 \(S_{t-1}\)，计算 \(S_t\) 只需要

\[
R_t^{raw}=
\left(k_t^{raw},v_t,g_t,\beta_t\right).
\]

各字段的作用是：

- \(k_t^{raw}\)：重新执行与 verify 相同的 normalization、state read 和 rank-one write；
- \(v_t\)：重新计算 corrected value；
- \(g_t\)：通过相同的 \(\exp\) 路径形成 decay；
- \(\beta_t\)：形成 correction；
- \(q_t\)：只用于本轮 output readout，不改变 state，因此不进入 replay record。

对于常见的 Qwen3.6 数值边界，raw \(k/v\) 是 BF16 represented values，\(g/\beta\) 是 FP32
represented values，checkpoint 是 FP32。所谓 raw record，是对这些实际 transition inputs 的
lossless side copy，而不是重新从 hidden state 或上游 projection 推导一次。

---

## 3. Accepted-prefix replay

### 3.1 Verify input、output 与 commit length

设 speculative drafter 给出 \(D\) 个 draft tokens。Target verify 处理

\[
T=D+1
\]

个 inputs：

~~~text
input 1       = round 开始时尚未处理的 anchor
input i + 1   = draft i, 1 <= i <= D
~~~

若前 \(A\) 个 drafts 与 target 匹配，target 许可输出

\[
p=A+1
\]

个 tokens：\(A\) 个 accepted drafts 加一个 correction/bonus token。

需要推进 state 的 inputs 恰好是前 \(p\) 个 verify inputs，也就是旧 anchor 加前 \(A\) 个 accepted
drafts。最后产生的 correction/bonus 尚未作为 input 执行；它成为下一轮 anchor。

最终输出边界可能只保留 licensed outputs 的前 \(m\) 个，因此 state 的 commit length 是

\[
0\le m\le p.
\]

例如 \(D=5,A=3\) 时，\(p=4\)。四个候选 state transitions 是“旧 anchor + 前三个 accepted
drafts”，四个 outputs 是“三个 accepted drafts + correction”。若最终 \(m=2\)，只重放“旧 anchor +
第一个 accepted draft”；第二个 output 是新的 pending anchor。

### 3.2 Verify：产生 outputs 和 raw records

Verify 从 committed checkpoint 开始，顺序执行原 recurrence：

~~~text
S_verify <- S0

for t = 1 .. T:
    q <- normalize(q_raw[t])
    k <- normalize(k_raw[t])
    alpha <- exp(g[t])
    S_verify <- alpha * S_verify
    u <- beta[t] * (v[t] - S_verify * k)
    S_verify <- S_verify + outer(u, k)
    y[t] <- scale * S_verify * q
    record[t] <- (k_raw[t], v[t], g[t], beta[t])
~~~

这里的 \(S_{verify}\) 是生成当前 window outputs 所需的 transient trajectory。Verify 结束后，它不作为
committed state 发布；持久 checkpoint \(S_0\) 保持不变。

### 3.3 Fold：只重放 accepted prefix

最终 \(m\) 已知后，fold 从同一 \(S_0\) 出发：

~~~text
S_fold <- S0

for t = 1 .. m:
    k <- normalize(record[t].k_raw)
    alpha <- exp(record[t].g)
    S_fold <- alpha * S_fold
    u <- record[t].beta * (record[t].v - S_fold * k)
    S_fold <- S_fold + outer(u, k)

publish S_fold
~~~

\(m=0\) 时 state 严格不变。Rejected suffix \(R_{m+1:T}\) 从未被 fold 读取。

单个 head 内仍有最多 \(m\) 次顺序 transition，但不同 layers、value heads 和 batch rows 相互独立。
Qwen3.6 的并行宽度来自 30/48 个 GDN layers、32/48 个 value heads 和多个 active rows，而单行
token loop 最多只有 6 或 16 次。

### 3.4 Accepted-prefix 正确性

把 verify 使用的有限精度 transition 记为 \(F_{\mathrm{fp}}\)：

\[
S_t^{verify}=
F_{\mathrm{fp}}(S_{t-1}^{verify},R_t),
\qquad
S_0^{verify}=S_0.
\]

若 fold 使用完全相同的 \(F_{\mathrm{fp}}\)：

\[
S_t^{fold}=
F_{\mathrm{fp}}(S_{t-1}^{fold},R_t),
\qquad
S_0^{fold}=S_0,
\]

则可以直接按 \(t\) 归纳：

\[
S_t^{fold}=S_t^{verify},
\qquad 0\le t\le m.
\]

所以

\[
S_m^{fold}=S_m^{verify}
\]

可以是有限精度下的逐 bit 结论，而不只是实数域中的等价。它成立的前提是两条路径调用的是同一个
deterministic floating-point transition，并消费相同的 record bits。

这里比较的是同一个物理 verify block 的 recorded prefix。它不要求另起一次较短 verify、不同
prefill 分块或另一套数值实现产生相同 inputs、logits 或 state。Record 之前的计算已确定本轮输入；
Fold 要忠实提交这些输入形成的轨迹。

---

## 4. 数值核心：为什么“代数等价”仍会产生 state drift

### 4.1 有限精度中的计算路径

对同一组 inputs，一次 GDN transition 在实数域中可以写成

\[
S_t=
\alpha_tS_{t-1}
+\beta_t\left(v_t-\alpha_tS_{t-1}\bar k_t\right)
\bar k_t^{\mathsf T},
\]

也可以写成

\[
S_t=
\alpha_tS_{t-1}
\left(I-\beta_t\bar k_t\bar k_t^{\mathsf T}\right)
+\beta_tv_t\bar k_t^{\mathsf T}.
\]

这两个表达式定义同一个实数映射，但 floating-point state 是具体求值程序的结果。它取决于：

- raw key normalization 的 reduction tree、sqrt 和 division；
- \(g_t\) 到 \(\alpha_t\) 的 \(\exp\) 路径；
- \(S\bar k\) 的累加顺序和 operand precision；
- state decay、correction 和 rank-one update 的先后顺序；
- FMA contraction、tile decomposition 和 state-store boundary。

代数等价不会自动带来 represented state 相等。Replay 一旦改变上述任一环节，就定义了一个新的有限精度
transition；单步差异会进入 committed state，并成为下一轮 recurrence 的输入。

### 4.2 Raw record 与 replay path

Raw record

\[
R_t=(k_t^{raw},v_t,g_t,\beta_t)
\]

固定了 transition 的 represented inputs。完整的 replay state 还由有限精度映射 \(F_{\mathrm{fp}}\) 共同决定：

\[
S_t=F_{\mathrm{fp}}(S_{t-1},R_t).
\]

若 fold 实际执行另一个求值程序 \(\widetilde F_{\mathrm{fp}}\)，即使两者在实数域中对应同一个公式，

\[
\widetilde F_{\mathrm{fp}}(S,R)
\ne
F_{\mathrm{fp}}(S,R)
\]

仍然可能成立。Raw record 固定 inputs identity；verbatim replay 固定 transition identity。

### 4.3 Verbatim closed-loop replay

Closed-loop replay 的 record 和计算职责是：

| 项目 | 要求 |
|---|---|
| raw \(v\) | 保存 verify recurrence 实际消费的 represented bits |
| raw \(k\) | 保存 normalization 之前的 represented bits |
| \(g\) | 保存 verify 已得到的 FP32 log-decay gate，不从上游重新计算 |
| \(\beta\) | 保存 verify 已得到的 FP32 correction gate |
| normalization | 相同的 epsilon、reduction tree、sqrt/division 形式 |
| decay | 相同的 \(\exp\) 与 state multiply 顺序 |
| correction | 从 replay 到当前位置的 state 重新计算 \(u\) |
| rank-one update | 相同的 operand precision、FMA contraction 和 tile decomposition |
| state store | 相同的 FP32 represented boundary |

“Closed loop”表示每个 corrected value 都从当前 replay state、raw \(v/k\) 和 gate bits 当场计算。
这使 fold 的每一步都进入与 verify 相同的 transition，并使上一步得到的 represented state 成为下一步
correction 的直接输入。

SGLang 在 Kimi K3 bring-up 中观察到了这一数值边界：早期 fold 从另一条计算路径重算 gate，虽然当时
outputs 仍看起来正常，recurrent state 已经开始漂移。改为直接记录 verify 产生的 gate values，并在 fold
中复刻 recurrence 后，state 与 recurrent baseline 达到 bit-identical。SGLang 的 GDN exact fold 同样对齐了
verify branch 的 state tile、division-form L2 normalization 和 operation order。

### 4.4 State drift 如何传播

先考虑两条 trajectory 使用相同 represented inputs 和同一个实数 GDN 公式，但起始 state 存在差异

\[
\Delta S_t=\widetilde S_t-S_t.
\]

由 recurrence 可得

\[
\Delta S_{t+1}
=\alpha_{t+1}\Delta S_t
\left(I-\beta_{t+1}\bar k_{t+1}\bar k_{t+1}^{\mathsf T}\right).
\]

若 reconstructed transition 本身还引入局部浮点偏差 \(E_{t+1}\)，则可以写成误差模型

\[
\Delta S_{t+1}
=\alpha_{t+1}\Delta S_t
\left(I-\beta_{t+1}\bar k_{t+1}\bar k_{t+1}^{\mathsf T}\right)
+E_{t+1}.
\]

对应的 state readout 差异是

\[
\Delta y_{t+1}=c\,\Delta S_{t+1}\bar q_{t+1}.
\]

本轮已由 verify trajectory 产生的 outputs 不会被 fold 追溯修改。真正的问题出现在下一轮：
attention/KV、hidden 和已发布 outputs 来自 verify trajectory，而 GDN checkpoint 来自另一个数值
trajectory。即使单轮 output cast 暂时掩盖了差异，persistent state 仍会在后续 token 中反复参与
correction 和 readout。

Decay \(\alpha<1\) 可以衰减部分旧误差，但

\[
I-\beta\bar k\bar k^{\mathsf T}
\]

是方向相关的更新，且每轮还可能注入新的 \(E_t\)。因此不能用“gate 会衰减误差”代替对 committed
state 的直接验证。

### 4.5 正确性的两个层次

ReplaySSM state reconstruction 应区分两个判据：

1. **数学正确性**

   从 represented BF16/FP32 inputs 和 FP32 \(S_0\) 出发，相对于独立 FP32/FP64 GDN oracle，
   output 和 final state 满足规定误差。

2. **有限精度 clone**

   对同一 \(S_0\)、同一 raw record bits 和同一 accepted prefix，fold 后的 FP32 state 与
   同一物理 verify block 的对应 trajectory 在每个 element 上逐 bit 相同。

验证 bitwise clone 时，最终文本或 BF16 output parity 都不够。直接证据应覆盖：

- raw \(k/v\) 与 \(g/\beta\) record 的 exact bit copy；
- verify 期间 committed checkpoint 完全不变；
- \(m=0,1,T\) 和中间 \(m\) 的 final state exact comparison；
- 不同 accepted-length 序列组成的长链 replay；
- 很小 key norm、gate 极值和多个 q/k-to-value-head groups；
- rejected suffix 改写后，committed state 仍保持不变。

若 fold 与 verify 采用不同的算术路径，就应按独立 oracle 声明 numerical tolerance，而不能把结果称为
bitwise clone。

---

## 5. Causal convolution 的有限窗口状态

GDN block 通常在 recurrence 之前带有 causal depthwise convolution。若卷积宽度为 \(W\)，持久
history 只包含最近 \(W-1\) 个 projection columns：

\[
H_0=[p_{-(W-2)},\ldots,p_0].
\]

Verify 产生

\[
p_1,p_2,\ldots,p_T.
\]

最终 commit length 为 \(m\) 时，正确 history 是

\[
H_m=
\operatorname{tail}_{W-1}
\left(H_0\mathbin\Vert[p_1,\ldots,p_m]\right).
\]

因此每个 verify position 只需记录一列 represented projection history，而不是保存整个
\(W-1\)-column window：

- \(m=0\)：history 不变；
- \(0<m<W-1\)：保留部分旧 columns 并追加 accepted columns；
- \(m\ge W-1\)：使用 accepted prefix 的最后 \(W-1\) 列。

只要 record 是 baseline 将写入 history 的同一 BF16 represented column，这个 commit 是 exact gather，
没有 recurrent reduction 或浮点重关联。Qwen3.6 使用 \(W=4\)，所以一列 record 是一份三列 snapshot
的 \(1/3\)。

---

## 6. 空间与计算特征

### 6.1 通用公式

令：

- \(L_g\)：GDN layer 数；
- \(H_q\)：q/k head 数；
- \(H_v\)：value head 数；
- \(K,V\)：key/value dimension；
- \(C_p\)：causal-conv projection channels；
- \(W-1\)：conv history columns；
- \(T\)：verify window。

一份 FP32 recurrent state 的字节数为

\[
R=4L_gH_vVK.
\]

一份 BF16 conv history 的字节数为

\[
Q=2L_gC_p(W-1).
\]

若 raw \(k/v\) 为 BF16、\(g/\beta\) 为 FP32，每 token 的 GDN record 为

\[
P_{gdn}=
2L_g(H_qK+H_vV)+8L_gH_v.
\]

每 token 的 conv column record 为

\[
P_{conv}=2L_gC_p.
\]

长度为 \(T\) 的完整 replay log 为

\[
P_{record}(T)=T(P_{gdn}+P_{conv}).
\]

Snapshot trajectory 的窗口相关容量是

\[
T(R+Q),
\]

而 raw ReplaySSM 是

\[
T(P_{gdn}+P_{conv}).
\]

### 6.2 当前 Qwen 实例尺寸

| 模型 | \(L_g\) | \(H_q\) | \(H_v\) | \(K/V\) | \(C_p\) | \(W\) |
|---|---:|---:|---:|---:|---:|---:|
| 27B | 48 | 16 | 48 | 128/128 | 10,240 | 4 |
| 35B-A3B | 30 | 16 | 32 | 128/128 | 8,192 | 4 |

对应的每 token state/record 尺寸为：

| 模型 | recurrent image | conv history | raw GDN record | conv record | record total |
|---|---:|---:|---:|---:|---:|
| 27B | 144.000 MiB | 2.8125 MiB | 0.767578 MiB | 0.9375 MiB | 1.705078 MiB |
| 35B-A3B | 60.000 MiB | 1.40625 MiB | 0.358887 MiB | 0.46875 MiB | 0.827637 MiB |

单个 raw record 与一份 recurrent+conv snapshot 的比例是：

| 模型 | snapshot/position | raw record/position | 尺寸比 |
|---|---:|---:|---:|
| 27B | 146.8125 MiB | 1.705078 MiB | 约 86.1× |
| 35B-A3B | 61.40625 MiB | 0.827637 MiB | 约 74.2× |

典型 verify windows 的 record 容量为：

| 模型与窗口 | raw GDN records | conv records | 合计 |
|---|---:|---:|---:|
| 27B，\(T=6\) | 4.605469 MiB | 5.625000 MiB | 10.230469 MiB |
| 27B，\(T=16\) | 12.281250 MiB | 15.000000 MiB | 27.281250 MiB |
| 35B-A3B，\(T=6\) | 2.153320 MiB | 2.812500 MiB | 4.965820 MiB |
| 35B-A3B，\(T=16\) | 5.742188 MiB | 7.500000 MiB | 13.242188 MiB |

### 6.3 计算形态

Raw-input replay 保持 verify 的 serial recurrence，并在 commit 增加最多 \(m\) 次 transition：

| 项目 | Snapshot baseline | Raw-input replay |
|---|---|---|
| verify state read | 一份 checkpoint | 一份 checkpoint |
| verify full-state writes | \(T\) 份 | 0 |
| verify record writes | 0 | \(T\) 份小 record |
| commit work | 选择 snapshot | 重放 \(m\) 次 transition，写一份 state |
| rollback | 选择对应 snapshot | 只读取 accepted record prefix |
| persistent numerical path | verify recurrence | verify recurrence 的 closed-loop clone |

本场景的 token 维度很短：MTP 最多 \(T=6\)，DFlash/DFlash2 最多 \(T=16\)。沿单个 layer、value head
和 batch row，fold 有 \(m\) 次顺序 transition；不同 layers、heads 和 batch rows 之间相互独立。
因此整体计算形态是大量彼此独立的短 recurrence。Fold 计算量随 accepted length \(m\) 线性增长，
record traffic 随 verify length \(T\) 线性增长，最后只写一份 committed state。

这项技术首先是 capacity 与 state-traffic 优化。Verify 少写 \(T\) 份大 state，代价是写小 records，并在
接受后多做一次短 prefix fold。端到端 latency 取决于 state traffic、accept length 和可用的
layer/head/batch parallelism，不能从空间压缩比直接推出。

---

## 7. 核心结论

GDN speculative ReplaySSM 的状态表示是

\[
\text{one committed checkpoint}
+
\text{one short raw transition log}.
\]

其正确性依赖以下不变量：

1. Verify 从 committed checkpoint 产生 outputs 和 raw records，但不修改 checkpoint；
2. record 保存 raw pre-normalization \(k\)、raw \(v\) 以及 verify 自己产生的 \(g/\beta\) bits；
3. final commit length 同时定义需要重放的 record prefix；
4. rejected suffix 不被 fold 读取；
5. fold closed-loop 重算每个 corrected value；
6. fold 与 verify 使用相同的 normalization、gate、reduction、operation order 和 state-store boundary；
7. committed state 直接与同一物理 verify block 的对应 state prefix 比较。

Raw inputs 决定“可以重放什么”，verbatim recurrence 决定“重放后是否得到同一个有限精度 state”。
前者解决 snapshot 容量，后者阻止跨轮 state drift；两者共同构成短窗口 GDN ReplaySSM。

---

## 参考资料

- [SGLang and Miles Add Day-0 Support for Kimi K3](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support)：raw-input replay、stored gates 与 bit-identical state fold。
- [SGLang GDN exact fold](https://github.com/sgl-project/sglang/blob/bc285b2064c0373227cfb6ada77a37e7b8c43510/python/sglang/kernels/ops/attention/fla/gdn_replayssm_spec_fold.py)：与 verify recurrence 对齐的 closed-loop fold。
- [ReplaySSM: Cache SSM Inputs, Not State](https://tridao.me/blog/2026/replayssm/)：缓存 SSM transition inputs 并在需要时 fold state 的基本思路。
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464)：GDN 与 gated delta-rule 的数学来源。
