# GLM5Next 模型与权重

本文记录 GLM-5.3-Flash 的完整计算流程、源权重表示，以及已选定的 NInfer 量化范围，
作为后续算子准备与模型接入的共同依据。当前尚未实现 GLM5Next converter 或模型执行。

来源以两台 Spark 上的 NVIDIA checkpoint、Transformers 5.16.1 的模型定义和本地
vLLM 实现为依据，具体入口见第 10 节。通用格式和容器分别由
[数值格式](maintainer/tensor-formats.md)、[存储布局](maintainer/storage-layouts.md)及
[V3 容器](maintainer/artifact-container.md)维护。

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
| mHC 映射 / Sinkhorn | `[24,16384]` 映射、24 个 base、3 个 scale；20 次列归一化，细节见第 2 节 |
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
  → final RMSNorm → 4096 维 hidden → LM head
```

设单 token residual 为 `X[s,d]`，`s=0..3`、`d=0..4095`。每个 mHC site 有
`F[24,16384]`、`base[24]`、`scale[3]`。先沿完整 16384 维做无 affine 的 RMSNorm，
再投影为 24 个动态系数：

```text
z = F * flatten(X) / sqrt(mean(flatten(X)^2) + 1e-5)
(z_pre, z_post, z_mix) = split(z, 4, 4, 16)

pre[s]  = sigmoid(scale[0]*z_pre[s]  + base[s])   + 1e-6
post[s] = 2*sigmoid(scale[1]*z_post[s] + base[4+s])
A[i,j]  = scale[2]*z_mix[i,j] + base[8+4*i+j]
```

Residual mixer 从按行 softmax 开始：`C=softmax_row(A)+1e-6`，先做一次列归一化，
再做 19 轮「行归一化、列归一化」。每次分母都加 `1e-6`。这给出固定 20 次迭代的
近似双随机矩阵，不能替换成一次 softmax 或收敛条件不同的循环。

```text
collapsed[d] = sum_s(pre[s] * X[s,d])
u = RMSNorm_gamma(collapsed, eps=1e-5)
y = Attention(u) 或 FFN(u)
X_next[j,d] = post[j]*y[d] + sum_i(C[i,j]*X[i,d])
```

矩阵的第一个轴是旧 stream、第二个轴是新 stream，因此 residual combine 是 `C^T X`。
Block input 的 4096 维带权 RMSNorm 与计算系数时的 16384 维无权 RMSNorm 是两个不同归约。
每次 post 产出下一 site 使用的 residual；最终四流均值后还有 Main final RMSNorm。

源 `hc_*_fn/base/scale` 是 BF16。vLLM 将其精确展开为 FP32 做系数与混合计算，
FP32 计算精度不意味着源 checkpoint 存的是 FP32 权重。mHC 的 streams 是当前 token
的激活，不形成随序列长度增长的历史 cache。

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
Chunked prefill、逐 token decode 和 speculative verify 都要实现这一状态转移。

输出按 head 做 128 维 RMSNorm，乘直接 gamma，再乘 `sigmoid(z)`：

```text
r[h,:] = gamma * o[h,:] / sqrt(mean(o[h,:]^2)+1e-5) * sigmoid(z[h,:])
y = W_o * flatten(r)
```

每层持久状态为 FP32 `[64,128,128]` recurrent state，以及三组
`[8192,3]` BF16 convolution history。Accepted-prefix 恢复需要同时恢复两者；仅截短
token history 或 MLA cache 不能撤销 KDA 的递推。Verify records、fold 和 checkpoint 的
具体物理实现留给后续状态/Op 合同，状态的数学内容由这里确定。

## 4. Sparse MLA 与 Kpool4 Indexer

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
也没有额外 RoPE cache。以上是实数等价式；实际融合、投影吸收与有限精度出口仍需按
Op 合同验证，不能由代数等价直接推断逐位相同。

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
pools，每个 pool 展开为四个原始 token IDs；再加入当前未完成 pool 的 0–3 个可见
tail tokens。因此有效预算最多为 2051 tokens，所有 64 个 MLA heads 共用这组 IDs。
Indexer 分数只用于选择，不加入主 attention logits。

短上下文全选可以省略评分与 TopK，但仍要维护将来需要的 compressed keys 和 tail。
主模型各 MLA 层均独立评分，当前配置没有跨层 selection 共享。

增量状态包括完整 pools 的 compressed keys，以及未完成 pool 的 raw K、raw gate
scores 和相位。最少需要保存三个历史位置；vLLM 使用四个位置的循环区，每位置有
`K[128]` 和 `gate[128]` 两组 BF16 words。Prefill chunk、decode、verify 和 restore
必须保持同一分组与因果可见性；恢复到非四 token 边界时不能丢掉这个 tail。

### 4.4 FP8 Indexer 参考路径

本地 vLLM 对 index query 与 pooled key 应用同一归一化 Hadamard-128 变换，再采用
E4M3FN per-vector 量化；cache 每个 pool 为 128 B codes 加 4 B FP32 scale。
Scale 数值取 2 的幂，但以 FP32 word 保存。Query scale 可乘入 head coefficient，
key scale 在评分时应用。

具体参考编码顺序是：pool 在 FP32 做 weighted sum，物化为 BF16；Hadamard 用 FP32
计算，结果再次物化为 BF16；按 `2^ceil(log2(max(absmax,1e-4)/448))` 选 scale，
将 scaled values 编码为 E4M3FN。Query 同样在 Hadamard 后物化 BF16 再量化。
这是上游执行表示，独立于 NVIDIA 的静态权重量化。

Transformers 的直接参考路径没有这套 FP8 codec，且 pooling 中间舍入位置不同。
文档分别给出数学定义与上游量化路径；后续 Indexer 合同需明确其可观察表示与 selection
精度要求，不能将两条路径当作逐位相同的 oracle。

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
Main experts 的 NVFP4 与 MTP experts 的 block-FP8 采用各自的 activation permission，
不改变 clamp、route coefficient 或 shared contribution 的数学位置。

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

配置 `index_share_for_mtp_iteration=true`：本地 vLLM 在一轮 draft 的首步计算
selection，后续 draft steps 复用该轮结果。这是 MTP proposal 内的约定，不表示与 Main
或其他请求共享 cache。MTP 有自己的 512 维 latent 历史、pool cache 与 tail；提交时需
与 Main 的 accepted prefix 保持一致。Draft 层经过自身 final norm 后的 hidden 同时
用于下一步 conditioning 和 LM head，head 前不重复执行这一 norm。

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

这些语义由未来 GLM Frontend 解释；接入时需保留模板、token 边界及 thinking/tool
解析的对应关系，模型层只消费准备好的输入。

## 7. 选定的量化范围与权重边界

两台 Spark 的来源路径都是：

```text
/home/neroued/models/GLM-5.3-Flash/nvidia-nvfp4
```

以这份 checkpoint 为唯一权重起点，只增加 KDA 大投影与 MTP routed experts 两项量化：

| 部分 | 目标持久表示 | 计算许可或保留内容 |
|---|---|---|
| Main routed experts | 原 NVIDIA NVFP4 | `AllowA4`；保留 codes、G16 E4M3 scales 和原校准参数的含义 |
| Main 前三层 dense MLP | 原 NVIDIA NVFP4 | `AllowA4`，与 routed experts 独立的 input/weight scales |
| 34 层 KDA 的 Q/K/V/O | `q8_g32_fp16`，由源 BF16 转换 | `A16Only`；只量化这四个大投影 |
| KDA beta、两组低秩 gate 投影、output norm | 原 BF16 | 投影 `A16Only`，不额外量化 |
| KDA convolution、A_log、dt_bias | 原 FP32 | 保留源 words 与控制含义 |
| Main/MTP MLA 与 Indexer 权重 | 原 BF16 | 投影保持 A16；router/index score 的 FP32 计算不因存储 BF16 而降级 |
| Main/MTP shared experts | 原 BF16 | `A16Only` |
| Main/MTP router | 原 BF16 weight、FP32 correction bias | FP32 logits 与原 route 数学 |
| mHC、Text/MTP norms | 原 BF16 | 系数、norm 与混合的归约精度独立规定 |
| MTP routed experts | `fp8_e4m3fn_block128x128_bf16`，由源 BF16 转换 | `AllowA8` |
| MTP stem、embedding、LM head、全部 Vision | 原 BF16 | 保留源权重，投影 `A16Only` |

MLA cache 只支持 BF16、FP8 两类表示。Cache codec 与上表的静态权重量化分别定义。
Main MLA Q_b/O、shared experts 不加入本轮额外量化范围。

### 7.1 NVIDIA NVFP4 导入

源 E2M1 codes 沿 K 两个值打包为一个 U8；每 K16 group 一个 E4M3 scale，每矩阵一个
FP32 `weight_scale_2`，每输入一个 FP32 `input_scale`。两种 scalar 都是 multiplier：

```text
w_source = E2M1(code) * E4M3(scale) * weight_scale_2
d_w = RN_FP32(1 / weight_scale_2)
d_a = RN_FP32(1 / input_scale)
w_ninfer = E2M1(code) * E4M3(scale) / d_w
```

沿用 NInfer 的 divisor 格式时，codes/scales 可保值重排；FP32 reciprocal 本身有舍入，
不是逐位恒等变换。`d_a` 属于使用位置的 activation metadata，不能与 weight divisor
合并解释。A4 输入的 K16 量化及 clamped SwiGLU 的激活出口按相应 Op 合同处理。

本轮检查了全部 42×288 个 Main experts：同 expert 的 gate/up `weight_scale_2`
相同；同层所有 experts 的 gate/up `input_scale` 相同，down `input_scale` 也各自
相同。前三层 dense MLP 的 gate/up 两类 scalar 也分别一致。因此 GU 可以在不修改
这些校准值的情况下沿输出行合并。

### 7.2 KDA Q8 与 MTP block-FP8

KDA Q/K/V/O 沿 K 每 32 个值使用一个 FP16 scale，转换采用现有 `grouped_absmax`
数值规则；每个逻辑权重占 1 B code＋`2/32` B scale。输入 activation 保持 A16，
量化后的权重进入既定 KDA 公式，不改变 convolution、gate 或 recurrent state 类型。

MTP 的 GU/down 使用 128×128 块 E4M3FN codes 与 BF16 multiplier：

```text
w_hat[e,n,k] = FP32(E4M3FN(code[e,n,k]) * BF16(scale[e,n/128,k/128]))
```

每个 expert 的 gate/up/down 分别量化，scale 不跨 expert 或矩阵边界。GU 沿 N 合并时，
2048 行的分界与 128 行块对齐。权重没有静态 activation divisor；允许的 A8 路径按
token 的 K128 分组动态量化激活，采用各自 activation scales。

BF16→block-FP8 的转换方法定为 `fp8_block_maxabs`，沿用
[现有行 FP8 编码器](../tools/convert/quantization/fp8_row.py)的数值规则，将 max-abs
归约范围改为每个独立的 128×128 块。源 BF16 words 精确展开为 FP32；全零块写
正零 BF16 scale 和全正零 E4M3FN codes。非零块按以下顺序编码：

```text
amax       = max(abs(block_fp32))
raw_scale  = RN_FP32(amax / 448)
scale_word = RN_BF16(raw_scale)
if scale_word == 0x0000:
    scale_word = 0x0001                    # 最小正 BF16 subnormal
scale32    = exact_bf16_to_fp32(scale_word)
inv_scale  = RN_FP32(1 / scale32)
normalized = RN_FP32(weight_fp32 * inv_scale)
code       = RN_E4M3FN(clamp(normalized, -448, 448))
```

所有 `RN` 均采用 round-to-nearest、ties-to-even。Codes 使用**实际存储的 BF16
scale** 的 FP32 reciprocal 编码；保留先算 reciprocal、再乘 weight 的舍入顺序。
非零块内的零值保留符号。源含 NaN/Inf、非零块 scale 非正或非有限、归一化结果
非有限时，转换报错。流式读取批次和后续 GU 合并均不改变块的归约域及编码结果。

这一方法仅使用源权重，无需激活校准数据；运行时动态 A8 量化保持前述 K128 约定。
Block-FP8 表示、`AllowA8` 许可和编码规则均已确定。现有 converter 的 block-FP8
入口仍只有 `import_encoded`，`fp8_block_maxabs` 编码器与转换入口待实现，尚未生成
MTP 的新量化 words。

### 7.3 逻辑矩阵与合并约束

本阶段按逻辑角色保留权重身份。后续持久 parent、packing 与 Op 边界须符合以下条件：

- KDA Q/K/V 同输入、同为 Q8，可按 `[Q8192,K8192,V8192]` 沿 N 组合；BF16 的
  beta、f_a、g_a 不混入同一个 Q8 parent。O projection 为独立 `[4096,8192]`。
- KDA convolution 的 Q/K/V 通道顺序与投影一致。源是三组 `[8192,1,4]`；若转成
  channel 连续布局，要显式映射 tap/channel，保持原 FP32 words 和时间顺序。
- MLA Q_a 与 KV_a 同输入，归一化则分别在 1536、512 维范围执行。Q_b 和 index-Q
  都消费 Q bottleneck，但各自有独立角色；KV_b 的行序是逐 head 的 `[K256,V256]`。
- Indexer 的 `wk`、head coefficients、pool gate 虽同源，计算精度与出口不同；物理
  合并不能将 FP32 head coefficient 计算隐式改成 BF16 出口。
- Main/MTP routed banks 的逻辑 GU 为 `[288,4096,4096]`，down 为
  `[288,4096,2048]`，GU 每 expert 内依次为 gate 2048、up 2048。Main 为 NVFP4，
  MTP 为 block-FP8；shared 不并入 routed bank。
- Norm 参数都是直接 gamma/affine，不采用 Qwen4Exp 的 offset-gamma 解释。
  mHC、convolution、pool APE 和 route bias 也不作为普通 Linear 权重重新量化。

持久合并与单次 GEMM 的输出组合分别决定，二者均需保留上述逻辑角色和数值边界。

## 8. 权重容量

源容量由 `.50` 的 33 个 safetensors 文件头逐 tensor 统计，共 147661 个 tensors；
与源 index 的 `total_size=204419110596` B 一致。`.51` 的模型/量化配置相同。
下表是全模型的 tensor payload，单位 GiB；不含文件头、对象对齐、runtime、分片复制
或额外模型权重。目标列是按已选格式计算的估算，尚非实际 artifact 大小。

| 部分 | NVIDIA 来源 | 选定表示后的估算 |
|---|---:|---:|
| Main routed experts | 159.469020 | 159.469020 |
| Main dense MLP | 0.237305 | 0.237305 |
| Main KDA | 8.729325 | 4.744950 |
| Main MLA＋Indexer | 2.559384 | 2.559384 |
| Main shared experts | 1.968750 | 1.968750 |
| Main router | 0.092330 | 0.092330 |
| Main mHC＋norm | 0.066617 | 0.066617 |
| Embedding＋LM head | 2.363281 | 2.363281 |
| MTP routed experts | 13.500000 | 6.750824 |
| MTP 其他独有权重 | 0.344283 | 0.344283 |
| Vision | 1.049837 | 1.049837 |
| **合计** | **190.380132** | **179.646581** |

KDA 的 Q/K/V/O 原载荷恰为 8.5 GiB，Q8-G32 后 4.515625 GiB，减少 3.984375 GiB。
MTP experts 共 7247757312 个权重，block-FP8 codes＋BF16 scales 为
7248642048 B，减少 6.749176 GiB。两项共减少约 **10.733551 GiB**。

不启用 MTP 时应扣除其 experts 与其他独有权重；embedding/head 不重复扣除。
不启用 Vision 时扣除独立 Vision 权重。双机每 rank 的权重容量还取决于实际分片与复制
计划，不能直接以总量除二作为最终驻留预算。

NVIDIA 模型卡中量化 shared experts 的措辞与实际配置不符，所链接 recipe 中「MTP
未导出」的注释也不符合这份本地 checkpoint。本文采用实际 tensors：shared 为 BF16，
完整 MTP 权重存在且原 experts 为 BF16。

## 9. 1M 上下文的持久状态容量

令 `L=1048576`、B=1。Main 有 11 层 MLA，每 token、每层保存一个 512 维 latent：

```text
BF16 payload = layers * L * 512 * 2 B
FP8 payload  = layers * L * 512 * 1 B
```

| 范围 | BF16 latent | FP8 codes，不含 scales |
|---|---:|---:|
| Main 11 层 | 11 GiB | 5.5 GiB |
| MTP 额外一层 | 1 GiB | 0.5 GiB |
| Main＋MTP | 12 GiB | 6 GiB |

这是每份完整 latent cache。按 attention heads 做 TP 时，每个 rank 通常仍需完整
512 维 latent；本表不随 head 数除以二。对 cache 做进一步分片需要另一套计算与通信
组织。MLA cache 的正式 FP8 codec 尚未规定 scale 粒度；例如每 256 维一个 FP16 scale，
Main 的 scales 为 44 MiB，含 MTP 为 48 MiB。NVIDIA recipe 的 FP8 cast 设置不能
自动替代 NInfer 的运行时 cache 数值合同。

Kpool4 每层保存 `L/4` 个 128 维 compressed keys：

| 范围 | BF16 pooled keys | vLLM FP8 pooled keys＋FP32 per-vector scales |
|---|---:|---:|
| Main 11 层 | 704 MiB | 363 MiB |
| Main＋MTP 12 层 | 768 MiB | 396 MiB |

Indexer tail 若采用每层四位置的 K/gate BF16 ring，Main＋MTP 共 24 KiB。
KDA 的完整 34 层 FP32 state 为 136 MiB，三列 BF16 convolution history 为
4.78125 MiB；按 KDA heads TP2 后每 rank 分别为 68 MiB、2.390625 MiB。

以「MLA latent 和 FP8 Indexer cache 各 rank 复制，KDA heads TP2」为估算条件，
并仅为容量计算采用上述 MLA FP8 scale 粒度：

| 每 rank、单请求 | Main | Main＋原生 MTP |
|---|---:|---:|
| FP8 MLA＋FP8 Indexer＋KDA state/history＋tail | 约 5.97 GiB | 约 6.50 GiB |
| BF16 MLA＋FP8 Indexer＋KDA state/history＋tail | 约 11.42 GiB | 约 12.46 GiB |

这张表不固定模型并行策略，也不计执行 workspace、Graph、额外 StateImage/checkpoint
副本、page tables 或分配对齐。多个无共享前缀的请求按各自长度累加；KDA state 按状态
实例数增加，不随 L 增长。Prefix restore 除 token/latent 前沿外，还要恢复 KDA state、
convolution history 和 indexer tail；MTP conditioning 与自身状态单独管理。

## 10. 来源与当前边界

- 本地 NVIDIA checkpoint：两台 Spark 的
  `/home/neroued/models/GLM-5.3-Flash/nvidia-nvfp4`；本轮读取 config、quant config、
  processor/tokenizer/template、全部 tensor headers 与 NVFP4 scalar metadata。
  [官方配置](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/blob/main/config.json)与
  [NVIDIA recipe](https://github.com/NVIDIA/Model-Optimizer/blob/4956213d670c382385d9bc43e17379b9fc064e50/modelopt_recipes/models/zai-org/GLM-5.3-Flash/ptq/nvfp4_experts_dense_mlp-kv_fp8_cast.yaml)
  提供源表示说明，实际存在的 tensors 按第 8 节核实。
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

本轮完成架构与源表示核对、量化范围及 MTP 编码规则决策、容量计算。新量化尚未生成或做模型精度评测；
具体 Op 合同、持久物理布局、模型并行与执行接入由后续工作落实。
