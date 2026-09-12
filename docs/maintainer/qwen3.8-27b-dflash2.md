# Qwen3.8-27B DFlash2 算法核心

> 状态：正式 Engine 已接入，支持启动固定 `K=1..15`、`B=1..8`、eager/CUDA Graph、Text/Vision 和 Device/Host 状态复用。
>
> 范围：本文只固定 Qwen3.8-27B DFlash2 的模型计算、proposal、target
> verification 和状态语义。Artifact inventory 与存储格式由
> [`qwen3.8-27b-artifact.md`](qwen3.8-27b-artifact.md) 管理；Op 拆分、shape 和融合边界由
> [`2026-09-05-qwen3.8-27b-dflash2-op-checklist.md`](2026-09-05-qwen3.8-27b-dflash2-op-checklist.md)
> 管理。

## 1. 核对依据与结论

本文以以下本地内容为核对依据：

- DFlash2 checkpoint：`z-lab/Qwen3.8-27B-DFlash2` revision
  `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`；
- vLLM checkout `8f816a3f665489d7f0d222115d4f72ebab01076b`，重点是
  `qwen3_dflash.py`、`qwen3_dflash2.py` 以及 V2 DFlash/DFlash2 speculator；
- SGLang checkout `01e66a62db889b6ed921eaf08360119ea4e2ba75`，重点是
  `srt/models/dflash.py`、`dflash_worker_v2.py` 和 speculative DFlash kernels；
- NInfer 现有 35B-A3B DFlash target-conditioning、target verification 和状态事务语义。

两个上游实现对本 checkpoint 的模型计算一致；NInfer 的 Frontend preview、状态事务和
counter-based RNG 语义由本地 Engine 合同固定。DFlash2 不是多步扩散采样器；
它在一次五层 masked-block forward 中为运行配置指定的 `K` 个位置并行生成候选，再在一个
top-16 候选格上逐位选出一条路径。主模型仍是唯一的输出授权者。

`qwen3_dflash2.py` 用 `1 + num_speculative_tokens` 构造各层卷积的 block width；
SGLang worker 解析运行配置后调用 `set_block_size()`，同步更新 attention/MLP 卷积边界。
两者均允许覆盖 checkpoint 的推荐 block size。卷积用真实 block 内位置判断 previous tap，
非二次幂宽度同样合法。

## 2. 模型几何、运行配置与符号

### 2.1 Checkpoint 几何

| 量 | 符号 | 固定值 |
|---|---:|---:|
| draft layers | `L_d` | `5` |
| hidden width | `H` | `5120` |
| MLP width | `I` | `17408` |
| query heads | `N_q` | `32` |
| KV heads | `N_kv` | `8` |
| head width | `D` | `128` |
| Q/K/V projection widths |  | `4096 / 1024 / 1024` |
| target feature layers |  | `[5,19,33,47,61]` |
| concatenated target feature width | `H_f` | `25600` |
| sliding-window scalar | `S` | `2048` |
| convolution taps | `C_tap` | `2` |
| convolution group width | `G_s` | `16` |
| convolution groups | `G=H/G_s` | `320` |
| selector candidates per position | `M` | `16` |
| selector rank | `R` | `256` |
| physical vocabulary rows | `V_p` | `248320` |
| candidate token domain | `V` | `248077` |
| mask token id |  | `248070` |

Checkpoint 的 `block_size=8` 是包含 anchor 的推荐总 query width，对应七个 draft token。
它不是权重 shape、位置 embedding 容量或算法强制长度。NInfer 的运行范围为：

| 量 | 符号 | 运行范围 |
|---|---|---|
| active requests | `B` | `1..8` |
| startup-selected proposal count | `K` | `1..15`，推荐 `7` |
| proposal/verify block width | `W=K+1` | `2..16` |
| backbone/verify matrix columns | `T=W*B` | 最大 `128` |
| mask-head/selector matrix columns | `U=K*B` | 最大 `120` |

一个 Engine 在启动时选定 K；同一 compact batch 使用相同物理 W。每请求的有效 verify
extent 另由 Section 7.4 定义。本文不引入请求间或轮次间自动调节 K 的调度策略。

```text
draft query width = 1 anchor + K masks = W
target verify width = 1 anchor + K drafts = W
maximum newly licensed tokens per decode round = K accepted drafts + 1 bonus = W
```

SGLang 的 `--speculative-num-draft-tokens` 传递总 block width；vLLM 的
`num_speculative_tokens` 和 NInfer 的 `draft_tokens` 传递 mask/proposal 数。推荐配置分别
写作 `8`、`7`、`7`；NInfer 最大配置 `15` 对应总宽度 `16`。Converter 对注册 source
config 中 `block_size=8` 的检查不限制运行 K；改变 K 不需要改变或重新转换权重。

### 2.2 Round 起点

每次 proposal 开始前，Program 先把上一轮 pending target features 补入 DFlash2 context。
补齐后，target 已经处理位置 `0..F-1` 的 token，Text KV/GDN execution frontier 与
DFlash2 materialized-context frontier 均为 `F`。上一次 target sampling 已生成、但还没有
被 target forward 消费的 token 是 anchor `a=x_F`。本轮 DFlash2 预测：

```text
absolute position     F       F+1       F+2       ...       F+K
draft-model input     a       MASK      MASK                 MASK
role                  anchor  proposal  proposal              proposal
```

anchor 是已经由 target 生成的真实 token。它不产生 unary candidates，也不作为 proposal
输出；其 predecessor-codebook row 条件化第一枚 draft 的 selector edge。K 个 mask 输出
分别预测同一绝对位置 `F+1..F+K` 的 token，不使用 causal LM 的一位右移解释。

## 3. Target hidden conditioning

### 3.1 Feature capture 位置

令 `r_t^l` 是 target 的零起始 Text block `l` 完成 attention/GDN 和 MLP residual
add 之后的完整 residual stream，即下一个 block 的 input norm 之前的值。
DFlash2 对每个已由 target 处理的位置 `t` 捕获：

```text
s_t = concat(r_t^5, r_t^19, r_t^33, r_t^47, r_t^61)   # [25600]
c_t = plain_rmsnorm(W_fc s_t, w_context, eps=1e-6)     # [5120]
```

`W_fc [5120,25600]` 无 bias，输入分段顺序严格为 `[5,19,33,47,61]`。上游把
embedding state 计为 hidden-state boundary 0 时，对应的 boundary ids 是
`[6,20,34,48,62]`；这不改变它们仍然是 target block ids 的事实。

Vision 在 Text embedding scatter 之后进入相同 target decoder，因此 image/video prompt 捕获
同样的五组 Text residual features。DFlash2 不读取 Vision tower 状态，也不存在
Vision 专用数学分支。

### 3.2 Context K/V

同一个 `c_t` 被五个 draft layer 各自的 K/V projection 直接消费：

```text
k_ctx_raw^l(t) = W_k^l c_t
k_ctx^l(t) = rope_1d(
    plain_head_rmsnorm(k_ctx_raw^l(t), w_k_norm^l, eps=1e-6),
    position=t,
    head_dim=128,
    theta=1e7)
v_ctx^l(t) = W_v^l c_t

K_ctx_store^l(t) = BF16(k_ctx^l(t))
V_ctx_store^l(t) = FP16_RNE(BF16(v_ctx^l(t)))
```

Context 不经过 draft layer input norm、dynamic convolution、Q projection、attention output
projection 或 MLP。初始 prefill 只对 target-produced features 执行上述 feature/K/V
materialization，不把 prompt token 逐层跑过 DFlash2 backbone。五层全部使用 2048
local window；若一个 prefill chunk 超过 2048 行，只 materialize 该 chunk 的最后 2048
个绝对位置，ring 保留新的最后 2048 行，而逻辑 context frontier 按整个 chunk 前进。

上述公式没有在 K projection 与 head norm 之间定义可观察的 BF16 cast；融合 materialization
的 raw K 精度属于实现 profile。最终 K 的 BF16 store 和 V 的 `FP16_RNE(BF16)` storage
转换保持显式边界。Context materialization 的合同和独立数学 oracle 直接使用上述完整
公式。可变 context 宽度、count envelope 及实现路线的当前验收状态统一见 Op checklist，
算法文档不维护逐 Op 完成进度。

DFlash2 RoPE 是完整 128 维、split-half/NeoX-style 的一维 RoPE。它使用绝对 Text
cache position 和 `theta=1e7`，不使用 target 的 64 维 partial MRoPE，也不消费 Vision
的三轴 position。

## 4. Dynamic grouped convolution

Attention 和 MLP 子层各有一套独立的 dynamic grouped convolution；每套包含
`prepare/input side` 和 `finish/output side` 两个 side。

对任一子层，输入 `X [B,W,H]` 重排为 `X[b,i,g,j]`，其中
`g in [0,320)`、`j in [0,16)`。一次无 bias projection 生成两个 side 的动态系数：

```text
delta = reshape(W_delta X, [B,W,2 sides,2 taps,320 groups])
```

`W_delta` 的 shape 为 `[1280,5120] = [2*2*320,5120]`。每个 side 还有学习到的
`base [2 taps,5120]`。对 side `s` 定义：

```text
Conv_s(X)[b,i,g,j]
  = (base[s,0,g,j] + delta[b,i,s,0,g]) * X[b,i,g,j]
  + 1{i>=1} * (base[s,1,g,j] + delta[b,i,s,1,g]) * X[b,i-1,g,j]
```

动态增量在同一 group 的 16 个channel上共享，base 则为channel-specific。第一个
block 位置的 tap-1 为零，不读取前一个 request 或前一轮的值。因此该卷积
只有当前 W 个 query row 内的短程依赖，没有跨轮 persistent convolution state。W 改变时，
卷积的 request stride 和位置零边界必须同时改变；taps、groups 和权重布局保持固定。

`finish` 必须复用同一子层 `prepare` 从归一化输入算出的 output-side
`delta`，不得从子层输出重新投影。

## 5. 五层 DFlash2 backbone

设 `x_l [B,W,H]` 是第 `l` 层输入的完整 residual stream。每层严格执行：

```text
n_l = plain_rmsnorm(x_l, input_norm_l, eps=1e-6)

(a_in_l, delta_a_out_l) = attention_conv_l.prepare(n_l)
q_raw, k_raw, v = split(W_qkv_l a_in_l, [4096,1024,1024])
q = rope_1d(plain_head_rmsnorm(q_raw, q_norm_l), positions)
k = rope_1d(plain_head_rmsnorm(k_raw, k_norm_l), positions)

attn_l = noncausal_symmetric_swa(
    q,
    concat_sequence(K_ctx_store_l, k),
    concat_sequence(V_ctx_store_l, v),
    scale=1/sqrt(128),
    window=2048)
a_raw_l = W_o_l attn_l
a_out_l = attention_conv_l.finish(a_raw_l, delta_a_out_l)
y_l = x_l + a_out_l

m_l = plain_rmsnorm(y_l, post_attention_norm_l, eps=1e-6)
(m_in_l, delta_m_out_l) = mlp_conv_l.prepare(m_l)
m_raw_l = W_down_l(SiLU(W_gate_l m_in_l) * W_up_l m_in_l)
m_out_l = mlp_conv_l.finish(m_raw_l, delta_m_out_l)
x_(l+1) = y_l + m_out_l
```

Q 有 32 个 head，K/V 有 8 个 head，每四个 Q head 共享一个 K/V head。全部 norm 都是
multiplicative plain RMSNorm，不使用 target Qwen3.8 的 offset-norm 语义。

五层均为 non-causal sliding attention。Checkpoint 的标量 `sliding_window=2048` 对应
FlashAttention 的 `(window_left,window_right)=(2047,2047)`。对 query 位置 `p_i`和已存在的
context/query key 位置 `p_j`：

```text
allowed(p_i,p_j) <=> abs(p_j-p_i) <= 2047
```

对本轮 `p_i=F+i, i in [0,K]`，W 个 query row 之间全部可见，包括位于当前
query 右侧的 mask row。已填满左边界后，第 `i` 个 query 最多读取 `2047-i` 个
context key。这是对称非因果窗口，不能替换为 causal window、只看左侧的 decode
window 或 full attention。

五层后对完整 residual 应用最终 plain RMSNorm：

```text
h = plain_rmsnorm(x_5, final_norm, eps=1e-6)   # [B,W,5120]
```

anchor output `h[:,0]` 不产生 proposal。Selector 只消费 mask hidden；以下将其重新编号为
`h_i=h[:,i+1], i=0..K-1`，与 candidate、edge 和 proposal q 的下标一致。

## 6. Top-16 candidates 和 coherent path selector

### 6.1 Unary candidates

Checkpoint 没有 token embedding 或 LM head。Query embedding 复用 target
`text/token_embedding`，candidate logits 复用启动时选定的 proposal head。

对每个 mask hidden `h_i`：

```text
z_vocab_i = Head(h_i)
(u_i[c], C_i[c]), c=0..15 = stable_top16(z_vocab_i)
```

- full route 的 `Head` 是 `text/output_head [248320,5120]`，只允许 token ids
  `0..248076` 进入 top-16；
- optimized route 的 `Head` 是 `text/draft_head [131072,5120]`，top-16 shortlist rows
  必须先经 `text/draft_head_token_ids` 映射为 global token ids，再访问 selector codebook；
- `input_embedding_scale=1`、`output_multiplier=1`，且 final-logit softcap 关闭；
- unary candidates 不经 target sampling 的 temperature、top-k/top-p/min-p、penalty 或 grammar
  filter。Proposal `q` 可以与 target `p` 有不同 support，target correction 保证输出语义。

Full 和 optimized route 产生不同 candidate set 是允许的；这会改变 acceptance 和性能，
不改变 target verification 的授权边界。

### 6.2 Edge score

设：

```text
W_pred [248320,256] = predecessor_codebook
W_succ [248320,256] = successor_codebook
W_h    [256,5120]   = hidden_projection
g_i = W_h h_i
```

对 proposal position `i in [0,K-1]`、前驱候选索引 `p`、当前候选索引 `c`：

```text
pred_token(i,p) = a                    if i=0
                  C_(i-1)[p]           if i>0

E_i[p,c] = u_i[c]
           + sum_(r=0..255) W_pred[pred_token(i,p),r] * g_i[r] * W_succ[C_i[c],r]
```

`E` 的数学 shape 是 `[B,K,16,16]`。第一个位置的 16 个 predecessor row 因为都是
anchor 而完全相同；保留该轴便于表达后续 K-1 个位置的同一公式。实现可以只计算实际
前驱对应的 row，也可以预计算部分或完整 lattice；不要求将完整 E 物化为 public tensor。

### 6.3 Path walk 与 proposal distribution

Path 是从左到右的条件 walk，不是独立地对 K 个位置取 unary argmax，也不是
对整条路径求全局 Viterbi/MAP。令已选前驱索引为 `j_(i-1)`，并在 `i=0` 使用
任一相同 anchor row：

```text
NInfer greedy proposal (request temperature <= 0):
    j_i = lowest-rank argmax_c E_i[j_(i-1),c]
    q_i = one_hot(j_i)

NInfer stochastic proposal (request temperature > 0):
    q_i[c | j_(i-1)] = softmax_c(E_i[j_(i-1),c] / temperature)
    j_i ~ q_i

d_(i+1) = C_i[j_i]
```

NInfer 将 target request 的 temperature 模式同时作为 selector 模式：greedy target 使用
one-hot proposal，positive-temperature target 使用上述 stochastic proposal。K 个 draft token
为 `d_1..d_K`。概率路径必须保留每个实际选中 predecessor 对应的
16 项 `q_i` 及其 global candidate ids，供同一轮 target rejection/correction 使用。`q_i`
是 FP32 可观察语义；不得把未选 predecessor row、unary softmax 或 BF16 重算值冒充为
实际 proposal distribution。

Draft draws 与 target accept/correction/bonus draws 是互相独立的随机变量。NInfer 将其绑定到
独立的 DFlash2-proposal RNG purpose；对预测位置 `F+i+1` 的第 `i` 个 draft，counter key
使用 request seed、该 token 之前的逻辑位置 `F+i` 和这一 purpose，不依赖 compact-batch
row 或 kernel 执行顺序。

## 7. Target verification 与无损性

对 request `b`，令 `P_b in [0,K]` 为本轮可验证的 draft 数，令
`Q_b=P_b+1` 为 target verify 的 live input-column 数。Target 以启动选定的物理 `W=K+1` 执行；
只有 column `0..Q_b-1` 属于该 request 的逻辑轨迹。

### 7.1 Causal verify alignment

Target 一次 causal forward 消费：

```text
verify input column     0       1       2       ...       K
token                   a       d_1     d_2               d_K
absolute position       F       F+1     F+2               F+K
target distribution     p_0     p_1     p_2               p_K
predicts position       F+1     F+2     F+3               F+K+1
```

Draft-model mask output 的 same-position 语义和 target causal next-token 语义在此对齐：
`d_i` 由 verify column `i-1` 的 target distribution 验证。对于当前 extent `P_b`，
column `P_b` 只在前 `P_b` 个 draft 全部接受时生成 bonus。

每个 live verify column 的 target sampling 对象都只覆盖 token ids `0..248076`。Penalty
history 包含已经生成到 anchor `a` 为止的 token；处理 column `i` 时再包含本轮此前已接受的
`d_1..d_i`。Presence/frequency penalty、请求约束和 vocabulary mask 先作用于 target
logits。Temperature 为零时，`p_i` 表示在调整后 logits 的最小-id argmax 上的 point mass；
temperature 为正时，`p_i` 是继续经过 temperature、top-k、top-p 和 min-p 后的归一化分布。

### 7.2 Greedy

从左到右找最大 `A in [0,P]`，使：

```text
d_i == argmax(p_(i-1)), for every i=1..A
```

如果 `A<P`，`argmax(p_A)` 是 correction；如果 `A=P`，`argmax(p_P)` 是 bonus。本轮
target 授权的 provisional output 是：

```text
[d_1, ..., d_A, correction_or_bonus]   # licensed count L=A+1
```

这里的“无损”指没有任何未经当前 target verify logits 授权的 token 被提交。

### 7.3 Positive-temperature sampling

对 `i=0..P-1`，Proposal `q_i` 是 Section 6.3 保留的稀疏 16-candidate 条件分布。
`d_(i+1)` 的 standard speculative rejection rule 为：

```text
acceptance probability = min(1, p_i[d_(i+1)] / q_i[d_(i+1)])
```

首次拒绝时，correction 从归一化残差分布采样：

```text
r_i(v) = max(p_i(v) - q_i(v), 0)
correction ~ r_i / sum_v r_i(v)
```

若前 `P` 个 draft 全部接受，bonus 从 `p_P` 采样。因此任意的 selector、
proposal-head 量化和 optimized shortlist 都只改变 acceptance；只要拒绝与 correction 消费的
`q_i` 就是实际 path walk 采样的分布，最终输出保留 target `p_i` 的分布。

Greedy row 的 `q` 是 one-hot；stochastic selector 的 `q` 是实际 predecessor row 上的
16-way 条件分布。

### 7.4 Per-request valid extent

非因果 attention 使改变 W 后的 hidden、candidates 和 q 通常不等于更宽 block 的前缀；
卷积也必须使用匹配的 request 边界。这不禁止较短或较长 block：任意受支持 K 都定义了一次
合法 proposal，target rejection 只需消费实际抽取该 draft 的条件 q。算法合法性不要求不同
K 的 proposal 相同，也不要求新 K 的 acceptance 或吞吐优于推荐 K=7。

本支持计划的调度策略是在启动时选择 K，并按该 K 真正计算 W 列。当剩余输出预算或
capacity tail 只允许 `P<K` 个 draft 时，可以保留这次 K 位 proposal，target verify 的 live
input 取 `[a,d_1..d_P]`，即 Q=P+1 列。Proposal attention 的 valid columns 为 W，target
attention/GDN 的 valid columns 为 Q；两者不能共用一个错误的有效宽度值。上述策略不是
Op 必须计算推荐八列或最大十六列的数学限制，也不授权用另一宽度的 q 验证当前 draft。

Target 最多授权 `L=A+1<=P+1` 个 provisional output；correction/bonus 是 target output。
`P=0` 时只处理 anchor 并采一个 target output，无需定义 K=0 的 selector 或 record Op。

Frontend preview 在该 provisional output 上决定最终提交数 `N in [0,L]`。普通非终止行
取 `N=L`；EOS、stop 或用户 token budget 可以得到 `1<=N<=L`；cancellation 取 `N=0`
并释放 sequence。Target KV/GDN、continuation hidden、token counts、execution frontier 和
DFlash2 pending features 都以 `N` 为提交量。Target 物理 column `Q..W-1` 不改变任何逻辑状态。

## 8. 状态和事务

### 8.1 DFlash2 持久状态

StateImage 中 DFlash2-private payload 是五层已 materialize 的 cyclic context K/V：

```text
five layers x (BF16 K + FP16 V) x 8 heads x 128 x 2048 = 40 MiB
```

该持久 payload 与 K 无关；增大 K 扩展的是 proposal、verify 和 pending 的 transient capacity。

同一 StateImage 还拥有 target GDN state 和 continuation hidden；target full-attention KV
由其既有 paged storage 管理。DFlash2 materialized-context frontier 属于 sequence/checkpoint
metadata，不存放在 cyclic-cache payload 内。Frontier 为 `F_ctx` 时，ring 的 live interval
是 `[max(0,F_ctx-2048),F_ctx)`，绝对位置 `p` 使用物理 slot `p mod 2048`。五层都是
local layer，不存在 DFlash2 full paged-KV pool。

DFlash2 没有以下持久状态：

- 没有跨轮 dynamic-convolution history；
- 没有扩散 timestep 或 iterative-refinement state；
- 没有跨轮 candidate lattice/path state；
- 没有独立 token embedding、LM head 或 Vision state；
- 没有可提交的 draft-query K/V。

Prefix fork、assistant boundary、Host replica 和 restore 的逻辑状态包含 target state、
continuation hidden、五层 cyclic K/V 以及与其 coverage 一致的 frontier metadata。D2D fork
复制完整 StateImage ring，Host demote/restore 搬运同一组 payload；ring 中处于 frontier
之外的旧字节始终不可观察。

### 8.2 Transient 与 pending 状态

下列值只需存活到当前 proposal/verify 和 Frontend preview 完成：

- full top-16 candidate ids/unary logits；
- 实际 predecessor row 的 sparse FP32 `q [B,K,16]`；
- 已选 draft path `[B,K]`；
- target GDN ReplaySSM records、verify logits 和 accepted-count 结果；
- 当前 query block 的五层临时 K/V。

Target verify 捕获的五组 hidden features 使用 lane-owned
`BF16 [25600,W,N_lane]` staging，其中 startup-fixed `N_lane in [1,8]` 且不小于 active
request count。Frontend 提交的前 `N` 行成为 pending features，跨过本轮
commit 存活到下一次 proposal 前的 context catch-up；其余物理行失效。Pending features
不进入 StateImage 或 checkpoint；创建 prefix fork、可恢复 checkpoint、Host replica 或
retained terminal state 前，必须先 materialize pending span，使 DFlash2 context coverage
与 checkpoint frontier 一致。

Decode context materialization 的 W_c 表示 pending storage/gather 的物理宽度，counts[b]=N_b
才是要写 ring 的前缀长度。当前固定 K 的 Engine 可取 W_c=W；Op 应接纳 batched W_c=1..16，
prefill 继续接纳 B_c=1、W_c=1..2048。W_c、W、P_b 和 N_b 各有独立含义，不能用候选数 K
代替上轮已提交输入数 N_b，也不能丢掉 all-accept 后第 K+1 个输入行的 feature。

### 8.3 State transaction

若 target 接受 `A` 个 draft，令 `z` 为 correction/bonus，则 target 先产生：

```text
y = [d_1, ..., d_A, z]   # licensed count L=A+1
```

Target causal verify 只记录候选轨迹，不立即 Fold GDN 最终状态。Frontend preview 从 `y`
选择最终提交前缀 `y_1..y_N`。当 `N>0` 时：

- target Text KV/GDN 和 execution frontier 提交 verify input rows
  `[a,d_1,...,d_(N-1)]`，共 `N` 行；generated-token counts 提交已授权输出
  `[y_1,...,y_N]`，同样增加 `N`；
- continuation hidden 选择 verify column `N-1`，`y_N` 是尚未由 target forward 消费的
  continuation anchor；只有 `N=L` 时它才是原始 `z`，若 `N<L` 则它是 `d_N`；
- verify hidden features 的前 `N` 列写入 pending-feature staging，DFlash2 materialized-context
  frontier 暂时保持 `F`，target execution frontier 前进到 `F+N`；
- 继续生成时，下一次 proposal 前把 pending span `[F,F+N)` materialize 到五层 ring，
  再把 DFlash2 frontier 前进到 `F+N`；保留 terminal/checkpoint state 时也先完成该 catch-up。

`N=0` 的 pending cancellation 不提交任何 target 或 DFlash2 frontier，并释放 sequence；已经
adopt 的更早 commit 不会被后来的 cancellation 回滚。被拒绝或被 Frontend 截断的 target
KV/GDN records、draft-query K/V 和 hidden 尾部都不再逻辑可达。

## 9. 完整 decode round

```text
committed target prefix + unprocessed target anchor
    -> materialize previous pending target features; align DFlash2 context frontier
    -> build [anchor, MASK x K] at absolute positions F..F+K
    -> shared target token embedding
    -> five DFlash2 layers
       [input norm
        -> attention dynamic-conv prepare
        -> non-causal symmetric SWA over context + query
        -> attention dynamic-conv finish + residual
        -> post-attention norm
        -> MLP dynamic-conv prepare
        -> SwiGLU MLP
        -> MLP dynamic-conv finish + residual]
    -> final norm; discard anchor output for proposal
    -> full/optimized proposal head top-16 at K mask positions
    -> rank-256 edge scores, evaluated on demand or precomputed
    -> conditional greedy/stochastic path walk; retain the realized q rows
    -> target causal verify, physical W=K+1 and per-request live width P+1
    -> greedy prefix match or stochastic p/q rejection and correction
    -> produce L=A+1 provisional outputs
    -> Frontend preview selects committed prefix N
    -> commit N target input rows and retain their target features as pending
    -> publish N outputs; retain output N as the next anchor when generation continues
```

## 10. 与 DFlash1 的精确差异

| 边界 | 现有 35B-A3B DFlash | Qwen3.8-27B DFlash2 |
|---|---|---|
| NInfer runtime draft count | `K=1..15` | `K=1..15`，checkpoint 推荐 7 |
| runtime block | `1 anchor + K masks` | `1 anchor + K masks` |
| draft layers | 6 | 5 |
| attention | 5 local + 1 full | 5 local，无 full layer |
| local window | 4096 | 2048 |
| per-sublayer dynamic conv | 无 | attention/MLP 各一组 prepare+finish |
| proposal per position | argmax/one-hot | top-16 + conditional selector path |
| stochastic proposal `q` | one-hot | sparse FP32 16-way conditional distribution |
| DFlash-private persistent state | local K/V + full K/V | 仅五层 local K/V |

共享的是 target-feature conditioning、masked-block 并行 proposal、causal target verify 和提交事务。
DFlash2 不能通过在 DFlash1 schedule 上只换一组权重来实现；dynamic convolution、
candidate edge/conditional walk 和 sparse-`q` accept 都是必须的语义边界。

## 11. Engine 实现与验收

27B package 的 immutable model view 绑定本 checkpoint 的 DFlash2 payload；family runtime
拥有共同的 masked-draft prefill、round、Frontend commit、StateImage 和 Graph 生命周期。
五层动态卷积及 coherent selector 在 family 的 `dflash_impl.h` 中按编译期配置选择，target
projection/post-mixer 继续使用既有三个 execution-leaf families。所有计算 kernel 仍归 `src/ops`。

每个 Engine 仅为所选 K 分配 proposal、verify、top-16 ids/q 和 ReplaySSM records。Graph
按 exact B 和有界 attention frontier 区间构造；普通 target KV 保持所选 codec，DFlash2
local ring 的格式固定。DFlash2 不建立 full backend KV pool，其 paged backend frontier
恒为零；local context coverage 由独立的 `dflash_context_frontier` 记录，保留的完整
StateImage 对齐 main frontier。

Sparse accept 保留实际 q 且只读 token counts。Frontend preview 后，Program 仅为最终提交
的 N 个 token 增加 counts，以 N fold ReplaySSM；partial terminal 的 continuation hidden
选 N−1。terminal、checkpoint 和保留状态的 context materialization 包含全部已提交输入。

真实 Engine 验证入口为 `ninfer_qwen3_8_27b_dflash2_real_test`，命令和所覆盖行为见
[tests README](../../tests/README.md)。RTX 5090、sm_120a、CUDA 13.1 上已验证两种本地
companion artifact、K=1/2/7/15、full/optimized head、BF16/INT8 target KV、eager/Graph、
B=1/2/8、penalty counts、固定 seed 重放、partial terminal、超过 2048 token 的 ring
替换/续接、image/video 及 Host State restore。固定贪心 fixture 与 ordinary decoding
比较用于检测接线/状态回归；它不把不同浮点执行路线的任意输入都要求为 token parity。
各 Op 的数学或 exact-state 判据仍由对应 qualification 定义。

正式吞吐/逐阶段测量使用 [product benchmark](../../bench/README.md) 的
`--spec dflash2 --draft-tokens K`；无需私有推理入口。每个常规 decode round 传输固定
1184 B ingress 和 576 B egress，前者为位置/slot/采样控制，后者为 token ids 和接受数量。
q、hidden、KV、ReplaySSM records 和 local ring 留在 Device；Host StateImage 传输属于
checkpoint/restore 生命周期。Frontend 确定最终 N 后，fold 的 row 描述通过 kernel 参数传入。
