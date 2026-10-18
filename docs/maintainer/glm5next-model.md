# GLM5Next 模型语义

本文长期维护 GLM-5.3-Flash 的架构、完整计算、逻辑权重角色和历史依赖。
[量化与数值表示](glm5next-quantization.md)维护选定表示、精确编码、计算精度和
显式舍入边界；本文的数学公式在这些边界下执行。

模型语义与选定表示不代表当前已实现能力。实施状态和任务入口统一见
[GLM5Next 设计索引](../designs/glm5next/README.md)，双机放置及执行接口不在本文重复定义。

## 1. 架构与实例

GLM-5.3-Flash 使用 `Glm5NextForConditionalGeneration / glm5_next`，内含
`glm5_next_text` 和 `glm5_next_vision`。下表是当前 checkpoint 的实际配置；这些数值
属于模型实例，后续执行应读取 config，而不是将其作为架构常数。

| 参数 | GLM-5.3-Flash |
|---|---|
| Text hidden / vocabulary | 4096 / 154880；embedding 与 LM head 不绑权重 |
| Main 层序 | 45 层；零基层号 `3,7,11,15,19,23,27,31,35,39,43` 为 sparse MLA，其余 34 层为 KDA |
| MLP 层序 | 零基 0–2 为 dense MLP，3–44 为 MoE，共 42 个 MoE 层 |
| mHC | 4 条 residual streams，完整 residual 宽度 16384；每层 attention、FFN 各一组独立参数 |
| mHC 映射 / Sinkhorn | `[24,16384]` 映射、24 个 base、3 个 scale；20 次列归一化，完整数学见第 2 节 |
| KDA | 64 heads × 128；Q/K/V 各 8192 维；四 tap depthwise convolution |
| KDA controls | forget/output gate 各有 128 维低秩瓶颈；beta 每 head 一个标量；forget log-decay 下界 -5 |
| MLA | 64 heads，Q/K 每 head 256 维，V 每 head 256 维 |
| MLA latent | Q bottleneck=1536，KV latent=512；`qk_rope_head_dim=0`、`mla_use_nope=true` |
| Indexer | 32 query heads × 128，单个 128 维 index key；Key LayerNorm |
| Kpool4 / selection | 逐通道加权压缩四个 index keys；最多选 512 个完整 pools，展开为 2048 tokens，另加至多三个 tail tokens |
| Main indexer sharing | `indexer_types` 全为 `full`，每个 MLA 层独立产生 selection |
| Dense MLP | intermediate=12288 |
| Routed experts | 每个 MoE 层 288 个，top-8，intermediate=2048 |
| Shared expert | 每个 MoE 层一个，intermediate=2048，直接加到 routed 输出 |
| Router | FP32 logits，sigmoid scoring，`noaux_tc` correction bias，选中系数归一化后乘 2.5；`n_group=topk_group=1` |
| SwiGLU | gate 上限 10，up 限制在 [-10,10]；clamp 在 SiLU/product 之前 |
| MTP | 一个独立 sparse MLA＋MoE 层，源层号 45；普通 residual，无 mHC；共享 Main embedding/head |
| Vision | 24 层，hidden=1024，16 heads × 64；MLP intermediate=4096 |
| Vision patch / merge | patch `[3,2,14,14]`；2×2 spatial downsample 到 4096 维；merger intermediate=10240 |
| Context | 1048576 tokens |
| Epsilon | Text/Vision RMSNorm `1e-5`；KDA Q/K L2 norm、Indexer LayerNorm、mHC pre/Sinkhorn `1e-6` |
| EOS / padding | EOS `[154820,154827,154829]`，padding=154820 |

文本的 MLA 与 Indexer 在此实例中均没有 RoPE 分量。Vision 仍有自己的二维空间 RoPE。
KDA、MLA、MLP 类型分别由各自层序确定；最后一个 Main 层是 KDA，MTP 则是 MLA。

本文使用逻辑矩阵 `W[N,K]`，N 为输出行、K 为输入特征。公式中的向量按列向量解释，
T 为每请求输入列数、B 为 batch、`M=T*B`。这里的逻辑轴不指定 CUDA kernel 的私有布局。

## 2. Main 与 mHC residual

```text
token embedding，在媒体占位位置替换为 Vision 特征
  → 将每个 4096 维向量复制为四条 residual streams
  → 每层：attention mHC pre → RMSNorm → KDA / sparse MLA → mHC post
          FFN mHC pre       → RMSNorm → dense MLP / MoE → mHC post
  → 对四条 streams 求均值
  → final RMSNorm → 4096 维 hidden → LM head / MTP conditioning
```

每层 attention、FFN 各有独立参数，45 层合计 90 个 mHC site。以下省略 token/batch
轴，计算逐 token 独立：`X[i,d]` 为四流 residual，`i,j=0..3`、`d=0..4095`、
`D=4096`。i 是旧 stream，j 是新 stream。`eps_coeff=eps_branch=eps_final=1e-5`，
`eps_pre=eps_sinkhorn=1e-6`，`sinkhorn_iters=20`，post multiplier 为 2。

### 2.1 全宽 RMS 与 24 行投影

当前 site 持有逻辑矩阵 `F[24,16384]`、`base[24]`、`scale[3]`，以及该分支的
`gamma[4096]`。令：

```text
r = rsqrt(sum_{i,d}(X[i,d]^2) / (4*D) + eps_coeff)
z[a] = r * sum_{i,d}(F[a,i*D+d] * X[i,d])
```

24 行的语义固定为：

| 行范围 | 含义 |
|---|---|
| `[0,4)` | pre logits |
| `[4,8)` | post logits |
| `[8,24)` | mixing logits，行号 `8+4*i+j` |

```text
p[i]   = sigmoid(scale[0]*z[i]       + base[i])       + eps_pre
q[j]   = 2*sigmoid(scale[1]*z[4+j]   + base[4+j])
A[i,j] = scale[2]*z[8+4*i+j]         + base[8+4*i+j]
```

`p` 为读取 gate，`q` 为写入 gate。`p` 不再跨 stream 归一化，也不除以 4。
sigmoid 饱和端点按数值结果保留。该模型没有附加的 temperature 除法。

该全宽 RMS 只用于系数生成；分支汇合仍读取原始 X。

### 2.2 固定 Sinkhorn 序列

```text
C = softmax_row(A) + eps_sinkhorn
C[i,j] /= sum_i(C[i,j]) + eps_sinkhorn

repeat sinkhorn_iters-1 times:
    C[i,j] /= sum_j(C[i,j]) + eps_sinkhorn
    C[i,j] /= sum_i(C[i,j]) + eps_sinkhorn
```

每一步的分母取自该步开始时的完整行或列；不能边修改元素边累计其余分母。
初始 softmax 按行计算，数值实现采用减去该行最大值的稳定形式。

当前 `sinkhorn_iters=20`，所以是初始按行 softmax、一次列归一化，再做 19 轮
「行归一化、列归一化」：总共 20 次列归一化、19 次显式行归一化。每次分母均加
epsilon；不能替换成 20 轮完整行列归一化、一次 softmax、sigmoid 初始化或按容差早停。

该有限迭代产生近似双随机矩阵。`C` 的行列和不保证严格为 1；后续计算不能依赖
严格双随机恒等式来省略混合或最终均值。概念来源见
[mHC 论文 §4](https://arxiv.org/html/2512.24880)，实际算法以本节为准。

### 2.3 四流汇合与分支 norm

```text
v[d] = sum_i(p[i] * X[i,d])
u[d] = gamma[d] * v[d] * rsqrt(sum_d(v[d]^2) / D + eps_branch)
y = Branch(u)
```

`Branch` 由层序选择 KDA、sparse MLA、dense MLP 或 MoE。
4096 维带权分支 norm 与 16384 维无权系数 norm 是两个不同归约域。
`gamma` 是直接 gain，不采用 `1+gamma`。

`u` 已经完成 attention/FFN input norm，后续分支不重复执行这一 norm。

### 2.4 residual 混合与分支注入

```text
X_next[j,d] = sum_i(C[i,j] * X[i,d]) + q[j] * y[d]
```

矩阵方向固定为 `C[old,new]`，即 `C^T X + q*y`。`q/C` 来自当前分支执行前的
同一份 `X`，在分支返回后直接使用，不重新计算 gate 或 Sinkhorn。

`y` 是整个分支的完整输出。X_next 的 BF16 表示边界以及下一 site 消费的实际值，
由[数值政策](glm5next-quantization.md#5-mhc-数值边界)规定。

### 2.5 初始化与 Main 最终出口

初始化将同一 embedding 复制为四条流：

```text
X[i,d] = embedding[d], i=0..3
```

媒体特征替换在此之前完成。四条流初始相同不改变当前 site 的完整系数数学。

最后一个 site 先完成第 2.4 节的 residual 更新，然后：

```text
mean[d] = sum_j(X_next[j,d]) / 4
hidden[d] = gamma_final[d] * mean[d]
            * rsqrt(sum_d(mean[d]^2) / D + eps_final)
```

`hidden` 同时服务 Main head 和 MTP conditioning，不在这两个消费者前重复 final norm。
四流均值、final norm 和 residual 的精度边界统一见数值文档。

### 2.6 与 Qwen4Exp GR 的区别

| 环节 | GLM5Next mHC | Qwen4Exp GR |
|---|---|---|
| residual | 4×4096 | 4×2560 |
| 系数生成前归一化 | 全部 16384 维，无 affine RMSNorm | 每条流独立做带权 RMSNorm |
| 动态映射 | `[24,16384]` 一次投影 | lowrank=320 的 down/up 和 injection 投影 |
| 读取 gate | 每条流一个标量，共 4 个 | 每条流、每个通道一个 |
| 分支输入 | 原始四流加权汇合，再做 4096 维 RMSNorm | 对归一化四流逐通道 gating，再除以 4 |
| norm gain | 直接 gamma | offset gain `1+weight` |
| 分支输出注入 | 每条流一个 `2*sigmoid` gate | 同样是每条流一个 |
| residual 自身转移 | 动态 4×4 矩阵跨流混合 | 各流保留自身 residual |
| Main 最终出口 | 四流均值，再 final RMSNorm | 学习得到的 final GR mix |
| 原生 MTP | 普通 residual | 四流 HC |

[Qwen 技术报告 §2.2](https://arxiv.org/html/2608.30320#S2.SS2) 在其消融规模上报告
GR 与动态 mHC 能力相近，并将去掉跨流混合后的效率收益归因于减少整份 residual
的读取。这不构成任意模型几何下投影 FLOPs 更少的结论，也不改变 GLM 的混合数学。

## 3. KDA

### 3.1 投影、卷积与 gate

每层 KDA 从 4096 维 block input `x_t` 生成：

| 源权重角色 | 逻辑 shape | 作用 |
|---|---|---|
| `q_proj / k_proj / v_proj` | 各 `[8192,4096]` | 64×128 维 Q/K/V |
| `q_conv1d / k_conv1d / v_conv1d` | 各 `[8192,1,4]` | 四 tap 因果 depthwise convolution，FP32 权重，无 bias |
| `b_proj` | `[64,4096]` | 每 head 的 beta logit |
| `f_a_proj / f_b_proj` | `[128,4096]` / `[8192,128]` | 每 key channel 的 forget logit |
| `A_log / dt_bias` | `[64]` / `[8192]` | FP32 decay controls |
| `g_a_proj / g_b_proj` | `[128,4096]` / `[8192,128]` | 每 output channel 的 sigmoid gate logit |
| `o_norm` | `[128]` | 所有 heads 共享的直接 RMSNorm gamma |
| `o_proj` | `[4096,8192]` | 回到 block hidden |

Q/K/V 分别经过 convolution 与 SiLU，gate 分支绕过 convolution。按源 tap 顺序，
四个 taps 对应 `t-3,t-2,t-1,t`；历史不足时补零。需要保留最近三列**卷积前投影值**。

对每个 head，以 FP32 归约得到：

```text
q = conv_silu(Q) / sqrt(sum(conv_silu(Q)^2) + 1e-6)
k = conv_silu(K) / sqrt(sum(conv_silu(K)^2) + 1e-6)
v = conv_silu(V)

f = W_f_b * (W_f_a * x)
g[h,k] = -5 * sigmoid(exp(A_log[h]) * (f[h,k] + dt_bias[h,k]))
beta[h] = sigmoid((W_b*x)[h])
z = W_g_b * (W_g_a * x)
```

两个低秩投影之间没有额外非线性。Forget gate 是每个 key channel 独立的 bounded
log-decay；beta 是每 head 一个标量。`g` 与 output gate `z` 是不同控制量。

### 3.2 递推与输出

以 `S[v,k]` 表示单 head 的 FP32 state，则每个 token 的完整递推为：

```text
S_decay[v,k] = S_prev[v,k] * exp(g[k])
delta[v]    = beta * (v[v] - sum_k(S_decay[v,k] * k[k]))
S_next[v,k] = S_decay[v,k] + delta[v] * k[k]
o[v]        = sum_k(S_next[v,k] * q[k]) / sqrt(128)
```

Decay 作用于 key 轴，delta 用**已衰减**的 state 计算，输出使用更新后的 state。
Prefill、逐 token decode 和 speculative verify 都要保持这一数学转移。

输出按 head 做 128 维 RMSNorm，乘直接 gamma，再乘 `sigmoid(z)`：

```text
r[h,:] = gamma * o[h,:] / sqrt(mean(o[h,:]^2)+1e-5) * sigmoid(z[h,:])
y = W_o * flatten(r)
```

每层逻辑历史由 recurrent state 和最近三列卷积前 Q/K/V 组成。恢复接受前缀时，
必须同时恢复两者；仅截短 token history 或 MLA cache 不能撤销 KDA 递推。
状态组成见第 7 节，record/fold 的实现设计见设计索引。

## 4. Sparse MLA 与 Kpool4 Indexer

本节定义模型数学、分组及因果可见性；cache 表示、codec 和计算边界见
[量化与数值表示](glm5next-quantization.md#3-mlaindexer-cache-与计算政策)。

### 4.1 MLA 数学与 latent cache

Main 的 11 个 MLA 层及 MTP 的一个 MLA 层使用以下几何：

| 权重 | 逻辑 shape | 输出 |
|---|---|---|
| `q_a_proj` | `[1536,4096]` | Q bottleneck；随后 1536 维 RMSNorm |
| `q_b_proj` | `[16384,1536]` | 64 个 256 维 Q heads |
| `kv_a_proj_with_mqa` | `[512,4096]` | 单个 512 维 latent；随后 512 维 RMSNorm |
| `kv_b_proj` | `[32768,512]` | 每 head 依次为 256 维 K、256 维 V |
| `o_proj` | `[4096,16384]` | 64 个 256 维 attention outputs 合回 hidden |

记 `q_r=RMSNorm(W_qa*x)`、`c=RMSNorm(W_kva*x)`，两处 epsilon 都为 `1e-5`。
每 head 的 `W_kvb` 行分别解释为 `U_K[h]`、`U_V[h]`，二者均为 `[256,512]`。
Indexer 给出当前 query 可见的 token 集合 `J_t` 后：

```text
q_h = W_qb[h] * q_r
k_jh = U_K[h] * c_j
v_jh = U_V[h] * c_j
a_thj = softmax_{j in J_t}(dot(q_h,k_jh) / 16)
y_h = sum_{j in J_t}(a_thj * v_jh)
y = W_o * concat_heads(y_h)
```

保存完整 K/V 展开不是数学要求。等价的 latent 域计算为：

```text
q_latent_h = transpose(U_K[h]) * q_h
a_thj = softmax_{j in J_t}(dot(q_latent_h,c_j) / 16)
u_h = sum_{j in J_t}(a_thj * c_j)
y_h = U_V[h] * u_h
```

注意缩放仍为 `1/sqrt(256)=1/16`。512 是 cache latent 的宽度，不改变训练时的
attention scale。只需缓存每 token 的一个 `c[512]`，供所有 heads 的 K/V 计算共用，
也没有额外 RoPE cache。以上是实数等价式，有限精度的 latent 域计算按数值文档的
映射和舍入边界验证，不能由代数等价推断逐位相同。

Sparse selection 减少每次读取的 token 集合，历史 latent 仍逐 token 保存，供后续
query 选择。每层有独立 latent 和 indexer cache。

### 4.2 Indexer 输入与 learned pooling

Indexer 消费同一个 block input `x` 和 MLA 的归一化 Q bottleneck `q_r`：

| 权重 | 逻辑 shape | 作用 |
|---|---|---|
| `wq_b` | `[4096,1536]` | 32 个 128 维 index queries |
| `wk` | `[128,4096]` | 单个 raw index key |
| `k_norm.weight / bias` | 各 `[128]` | 带 affine 的 LayerNorm，epsilon=`1e-6` |
| `weights_proj` | `[32,4096]` | 对 index heads 加权的 32 个系数 |
| `index_kpool_compress_gate` | `[128,4096]` | 每 token、每 channel 的池化 logit |
| `index_kpool_compress_ape` | `[4,128]` | pool 内四个位置的 learned bias |

```text
q_index[h,:] = reshape(W_index_q*q_r)[h,:]
k_raw = LayerNorm(W_index_k*x, gamma, bias, eps=1e-6)
p_raw = W_pool_gate*x
w[h]  = (W_head_weight*x)[h] / sqrt(32)
```

`w` 是带符号的线性投影结果，没有 sigmoid 或 softmax。对四个连续有效 tokens 形成的
完整 pool p，逐 channel d 沿四个 token 位置归一化：

```text
P[r,d] = softmax_{r=0..3}(p_raw[4p+r,d] + APE[r,d])
k_pool[p,d] = sum_{r=0..3}(P[r,d] * k_raw[4p+r,d])
```

每个 channel 有自己的四个权重。压缩对象是 index key，主 MLA latent 不经过这一步。
分组从请求的第一个有效 token 开始，padding 不应改变 pool 相位。

### 4.3 评分、selection 与 tail

```text
score[t,p] = sum_h(w[t,h] * ReLU(dot(q_index[t,h],k_pool[p]) / sqrt(128)))
```

只允许 pool 的最后一个 token 对当前 query 可见时参与 TopK。选至多 512 个完整
pools，每个 pool 代表四个原始 tokens；再加入当前未完成 pool 的 0–3 个可见
tail tokens。因此有效预算最多为 2051 tokens，所有 64 个 MLA heads 共用这个集合。
最新的完整 pool 不额外强制选中；完整 pool 的选择与未完成 tail 的加入是两个步骤。
Indexer 分数只用于选择，不加入主 attention logits。

短上下文全选可以省略评分与 TopK，但仍要维护将来需要的 compressed keys 和 tail。
主模型各 MLA 层均独立评分，当前配置没有跨层 selection 共享。

增量状态包括完整 pools 的 compressed keys，以及未完成 pool 的 raw K/gate。
恢复到非四 token 边界时必须同时恢复 tail。Prefill chunk、decode、verify 和 restore
保持同一分组相位与因果可见性，分组相位只由已处理有效 token 数决定。

## 5. Dense MLP 与 MoE

所有 Text、MTP 和 Vision 的 SwiGLU 都包含 clamp：

```text
a = min(W_gate*x + b_gate, 10)
b = clamp(W_up*x + b_up, -10, 10)
ffn(x) = W_down * (SiLU(a) * b) + b_down
```

Text/MTP 没有这些 projection biases；Vision blocks 有 bias。Gate 只有上界，up 有
上下界；不能将 clamp 移到 SiLU/product 之后。

前三层 dense MLP 的 GU 各为 `[12288,4096]`、down 为 `[4096,12288]`。
后续 42 层每个 routed/shared expert 的 GU 各为 `[2048,4096]`、down 为 `[4096,2048]`。

MoE 的路由定义为：

```text
logit_e = FP32_Dot(W_router[e,:], x)
s_e = sigmoid(logit_e)
choice_e = s_e + correction_bias_e
E_selected = Top8(choice)
p_e = 2.5 * s_e / (sum_{j in E_selected}(s_j) + 1e-20)
moe(x) = sum_{e in E_selected}(p_e * expert_e(x)) + shared_ffn(x)
```

Correction bias 只影响选择，最终系数使用未加 bias 的 sigmoid scores。此实例只有一个
expert group，group 筛选不会再排除候选。Shared expert 直接加入，不参与 Top8，
也没有 Qwen4Exp 的独立 shared sigmoid gate。

Router 源权重是 BF16，correction bias 是 FP32；config 明确要求 FP32 router logits。
权重量化和 activation permission 见数值文档；表示选择不改变 clamp、route
coefficient 或 shared contribution 的数学位置。

## 6. MTP、Vision 与输入输出

### 6.1 原生 MTP

本地 NVIDIA checkpoint 实际包含 `model.language_model.layers.45.*`，其 routed
experts 源权重是 BF16。MTP 复用 Main token embedding 和 LM head，但有自己的
`enorm`、`hnorm`、`eh_proj`、MLA/Indexer、MoE 及 `shared_head.norm`。

```text
e = RMSNorm_enorm(token_embedding)
h = RMSNorm_hnorm(previous_hidden)
x = W_eh * concat(e,h)                         # [4096,8192]
r = x + sparse_MLA(RMSNorm_input(x))
u = r + MoE(RMSNorm_post_attention(r))
next_hidden = RMSNorm_shared_head(u)
draft_logits = shared_Main_LM_head(next_hidden)
```

Main 提供的 conditioning 是最终 RMSNorm 后的 4096 维 hidden；后续 draft step
使用上一步 `next_hidden`。MTP 没有四流 residual，也不需要 Main 的 16384 维 mHC
激活。vLLM 在 position 0 将 MTP embedding 输入置零后再做 `enorm`，其余位置按输入
token 与 previous hidden 的对应关系执行。

配置 `index_share_for_mtp_iteration=true`：一轮 draft 首步计算 selection，后续 steps
复用首步选中的完整 pools 和冻结的可见边界，包括首步当时的 tail；不按后续位置
自动追加新 tail。复用步仍维护自身的 latent、pool cache 和 tail，可以省去 index
query、评分与 TopK。Selection 是本轮临时选择，不进入 checkpoint；Main/MTP 历史独立。

Canonical MTP 历史按真实 Main conditioning 更新；递归 draft hidden 不因 token
被接受就成为真实 Main hidden。需要重建的接受部分用真实 Main conditioning 重新
计算，不能仅截短 draft cache 直接采用。Draft 层经过自身 final norm 后的 hidden 同时用于
下一步 conditioning 和 LM head，head 前不重复执行这一 norm。

### 6.2 Vision

```text
图像/视频预处理、按 grid 切分 patch
  → Conv3d patch projection：1176 → 1024，带 bias
  → 24 层：RMSNorm → QKV → Q/K head RMSNorm → 2D RoPE
                    → 非因果 attention → O projection → residual add
           RMSNorm → clamped SwiGLU → residual add
  → 1024 维 post RMSNorm
  → 每 2×2 patches 做 Conv2d downsample：4×1024 → 4096，带 bias
  → merger projection 4096→4096 → LayerNorm → GELU
  → clamped SwiGLU：4096→10240→4096
  → 替换 Text 输入中的媒体占位 embedding
```

Vision 有 16 个 64 维 heads，QKV projection `[3072,1024]`，O projection
`[1024,1024]`；block FFN 的 gate/up 为 `[4096,1024]`，down 为 `[1024,4096]`。
Q/K head norm 的 gamma 各 `[64]`。Patch 权重源形状 `[1024,3,2,14,14]`，
downsample 为 `[4096,1024,2,2]`。Merger GU 各 `[10240,4096]`、down 为
`[4096,10240]`，这些 merger projections 没有 bias，LayerNorm 有 weight/bias。

Block norm、Q/K norm 和 post norm 都是直接 gamma 的 RMSNorm，epsilon=`1e-5`；
merger 使用普通 LayerNorm，epsilon=`1e-5`，再接非近似 GELU。空间 RoPE theta=10000。
输入没有独立 learned absolute-position embedding。Encoder 序列按媒体 grid 分隔，
媒体 token 数为 `T_grid*H_grid*W_grid/4`。

源 processor 使用 patch=14、temporal patch=2、merge=2；图像 token 范围 16–8000，
视频配置最大 240000、fps=2。RGB 归一化 mean 为 `[0.48145466,0.4578275,0.40821073]`，
std 为 `[0.26862954,0.26130258,0.27577711]`。这些是源预处理配置，本地 vLLM 另有
运行时媒体预算限制；它们不修改 Text 最大上下文，也不自动成为 NInfer 的产品限制。

### 6.3 Tokenizer、thinking 与 tools

源 checkpoint 的 tokenizer、`chat_template.jinja`、generation 和 processor 配置共同
定义输入输出资源。模板以 `[gMASK]<sop>` 开始，使用 GLM 的 system/user/assistant/
observation 角色标记；thinking 使用 `<think>…</think>`，工具调用采用
`<tool_call>`、`<arg_key>`、`<arg_value>` 结构，工具结果用 `<tool_response>` 表示。
Image/video markers、展开后的 token 数与 Vision 特征顺序必须一致。

Frontend 负责模板、token 边界及 thinking/tool 解析之间的对应关系；模型计算消费
准备好的输入，不负责输入协议或媒体获取。

## 7. 逻辑状态与历史依赖

| 分量 | 逻辑内容 | 与累计上下文的关系 |
|---|---|---|
| Main/MTP MLA | 每层、每 token 一个 512 维 latent | 随历史增长；Main 与 MTP 独立 |
| Indexer 完整 pools | 每 4 个有效 tokens 一个 128 维 pooled key | 随完整 pool 数增长，与对应 MLA 历史关联 |
| Indexer 未完成 tail | 当前 pool 的 raw K 与 raw gate，最多各 3 列 | 每请求、每层的定长前沿状态；APE 不预先加入 |
| KDA recurrent | 每层 64 个 `[value=128,key=128]` state | 累积全部已采用前缀，尺寸不随长度增长 |
| KDA convolution | 每层 Q/K/V 各 8192 通道的最近 3 列卷积前值 | 每请求的定长历史，顺序为最旧到最新 |
| MTP conditioning | Main 最终 norm 后的 4096 维 hidden；递归 draft 消费上一步 MTP hidden | 按实际 Main/MTP 前沿区分，不能混用 |
| mHC residual、动态系数 | 当前 token 在网络深度方向的四流激活与系数 | 不累积历史，不是请求 checkpoint 字段 |
| MTP selection、verify records | 本轮执行或接受前缀恢复使用的临时结果 | 不成为长期历史或 checkpoint |

Checkpoint 必须同时描述所保留前缀及其 KDA state/conv、Indexer tail 和必要的 MTP
conditioning。仅截断 growing history 不能恢复这些累计状态。Indexer tail 的有效列数
由已处理有效 token 数模 4 决定；MLA 与 Indexer 必须恢复到一致的可见前沿。

NInfer 将随历史增长的内容与每请求累计状态分别管理；物理 pool、slot、record plane
和提交接口由实现设计定义，不改变上表的逻辑依赖。双机分片与 owner 也不改变模型
状态含义。各请求的私有可写历史独立，前缀共享与恢复由上下文管理明确处理。

## 8. 来源

- 本地 NVIDIA checkpoint：两台 Spark 的
  `/home/neroued/models/GLM-5.3-Flash/nvidia-nvfp4`；已核对 config、quant config、
  processor/tokenizer/template、全部 tensor headers 与 NVFP4 scalar metadata。
  [官方配置](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/blob/main/config.json)与
  [NVIDIA recipe](https://github.com/NVIDIA/Model-Optimizer/blob/4956213d670c382385d9bc43e17379b9fc064e50/modelopt_recipes/models/zai-org/GLM-5.3-Flash/ptq/nvfp4_experts_dense_mlp-kv_fp8_cast.yaml)
  提供源表示说明，实际表示和容量见[数值文档](glm5next-quantization.md)。
- [Transformers 5.16.1 模型定义](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/glm5_next/modeling_glm5_next.py)：
  mHC、KDA、sparse MLA、learned pooling、clamped SwiGLU 和 Vision 的直接数学定义。
- 本地 vLLM revision `00b7847c8036b667742b4efb21aab1de51fd4721` 的
  [模型主体](https://github.com/vllm-project/vllm/blob/00b7847c8036b667742b4efb21aab1de51fd4721/vllm/models/glm5next/common/model.py)、
  [KDA](https://github.com/vllm-project/vllm/blob/00b7847c8036b667742b4efb21aab1de51fd4721/vllm/models/glm5next/common/kda.py)、
  [MLA / Indexer](https://github.com/vllm-project/vllm/blob/00b7847c8036b667742b4efb21aab1de51fd4721/vllm/models/glm5next/common/attention.py)、
  [Kpool codec](https://github.com/vllm-project/vllm/blob/00b7847c8036b667742b4efb21aab1de51fd4721/vllm/models/glm5next/nvidia/ops/kpool_compress.py)、
  [MTP](https://github.com/vllm-project/vllm/blob/00b7847c8036b667742b4efb21aab1de51fd4721/vllm/models/glm5next/common/mtp.py)和
  [Vision](https://github.com/vllm-project/vllm/blob/00b7847c8036b667742b4efb21aab1de51fd4721/vllm/models/glm5next/common/multimodal.py)：
  核对实际权重映射、latent 执行、增量状态、Indexer FP8 编码与原生 MTP 交接。
