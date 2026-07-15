# Qwen3.6-35B-A3B Operator Inventory

> Status: authoritative operator inventory and current implementation-support matrix for the
> future `qwen3_6_35b_a3b_rtx5090` target.

This document enumerates every device-side logical operator required by the exact
Qwen3.6-35B-A3B Text, MTP, and Vision graphs. It fixes each operator's logical shape and
mathematical semantics, then classifies the current NInfer implementation under the performance
definition in Section 1.

The checkpoint graph and dimensions come from
[`qwen3.6-35b-a3b-architecture.md`](qwen3.6-35b-a3b-architecture.md). Persistent formats and target
storage are defined by
[`qwen3.6-35b-a3b-ninfer-artifact.md`](qwen3.6-35b-a3b-ninfer-artifact.md). This inventory does not
define numeric precision, rounding, tensor layout, fusion policy, memory budgets, implementation
priority, or a test plan.

## 1. Support definition

The status is deliberately stricter than API availability or functional execution:

- **Supported** means that the exact 35B logical shape and required execution route select a
  current high-performance kernel, and retained RTX 5090 evidence establishes that kernel at its
  applicable bandwidth, compute, or fixed-resource roofline. A dimension-generic kernel counts
  only where its exact required shape has that evidence.
- **Adapt existing** means that a matching semantic Op, a close fixed-shape kernel, or reusable
  implementation machinery exists, but the exact 35B shape, one required execution route, or
  roofline qualification is missing. The operator is **not supported** for the 35B target until
  that gap is closed.
- **New implementation** means that no suitable high-performance Op/kernel exists. The operator
  is **not supported** and requires a new implementation rather than composition from generic
  linear or elementwise calls.

A generic/reference fallback, a Python implementation, a correctness test, a benchmark binary
without retained results, or evidence for only a different model shape does not establish
support. Archived profiler reports cited below are descriptive evidence; this active document owns
the current support conclusion. The measured kernels were also checked to remain present in the
current `src/ops` implementation.

### 1.1 Shape notation

NInfer's logical matrix convention is used throughout:

```text
X[K,T]                    T token columns, each with K features
W[N,K]                    linear weight
Linear(W,X) = W X         result [N,T]

H  = 2048                 Text hidden size
Vm = 248320               model vocabulary rows
Vt = 248077               token-policy domain
T  = 1                    ordinary decode
T  <= 6                   MTP target verification and rebuild
T  <= 1024                one Text prefill chunk
L  <= 262144              visible Text context

P                          raw Vision patches for one media item
S                          Vision attention segments
V  = P / 4                 merged Vision tokens
A  = 8T                    routed MoE assignments
M_e                        assignments routed to expert e
eps = 1e-6                 normalization epsilon
```

For compactness, the tables use these definitions:

```text
OffsetRMS(x,w) = (1+w) .* x / sqrt(mean(x^2) + eps)
PlainRMS(x,w)  = w .* x / sqrt(mean(x^2) + eps)
SiLU(x)        = x .* sigmoid(x)
SwiGLU(g,u)    = SiLU(g) .* u
```

The Text graph is fixed:

```text
embedding
  -> [GDN, GDN, GDN, FullAttention] x 10
  -> final OffsetRMS
  -> output head
```

Every one of the 40 layers applies the same sparse MoE after its token mixer:

```text
h = OffsetRMS(x, input_norm)
x = x + Mixer(h)
h = OffsetRMS(x, post_mixer_norm)
x = x + SparseMoe(h)
```

Host construction of multimodal metadata, raw host/device transfers, zero-copy tensor views, arena
allocation, cache cursors, and Program commit/rollback policy are not Ops and do not appear in the
inventory. Stateful attention, convolution, and delta-rule transformations do appear because
their state mutations are part of their mathematical contracts.

## 2. Supported exact operator domains

These are the only current operator domains that meet the support definition. They are inherited
from the identical 35B/27B Vision backbone geometry.

| Operator | Exact required domain | Mathematical semantics | Current support evidence |
|---|---|---|---|
| Vision affine LayerNorm | `x[1152,4096]`, `gamma[1152]`, `beta[1152]` | For patch `p`, `mu_p=mean_d x[d,p]`, `var_p=mean_d (x[d,p]-mu_p)^2`, and `y[:,p]=gamma.*(x[:,p]-mu_p)/sqrt(var_p+eps)+beta`. | **Supported for this exact domain.** The current fixed-`D=1152` warp kernel remains selected and reaches 95.4% of the fixed bandwidth roofline in the [retained report](archive/optimization-era/bench/vision-layer-norm-roofline.md). Other patch counts are unqualified and appear in Section 6. |
| Vision QKV bias add | `x[3456,4096]`, `b[3456]` | `x'[d,p]=x[d,p]+b[d]`. | **Supported for this exact domain.** The current paired kernel remains selected and reaches 86.8% of the fixed bandwidth roofline in the [retained report](archive/optimization-era/bench/vision-add-bias-roofline.md). Other patch counts and bias widths are unsupported rows in Section 6. |
| Vision block tanh GELU | `x[4304,4096]` | `x'=0.5*x.*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`. | **Supported for this exact domain.** The current paired kernel remains selected and reaches about 90% of the available-memory roofline in the [retained report](archive/optimization-era/bench/vision-gelu-roofline.md). Other patch counts and the merger's width-4608 exact GELU are unqualified. |
| Segmented Vision attention | Q/K/V/O `[72,16,4096]` for one segment | For head `h` and `i,j in [0,4096)`, `score[i,j]=dot(q[:,h,i],k[:,h,j])/sqrt(72)` and `o[:,h,i]=sum_j softmax(score[i,:])[j]*v[:,h,j]`. | **Supported for this exact domain.** The current fixed `16 x 72` Flash kernel remains present and the measured long route approaches its fixed resource roofline in the [retained report](archive/optimization-era/bench/vision-attention-roofline.md). The measured 256/1024 routes remain below roofline, so they are included with all other unqualified segment domains in Section 6. |

No Text, GDN, sparse-MoE, generation, or MTP operator currently meets this definition for the
35B target.

## 3. Text root and shared operators: adapt existing

Every row in this section is currently **unsupported** for the 35B target. The project has a
matching semantic contract or a close 27B implementation, but not a fully qualified exact 35B
path.

| Operator | Exact required shape | Mathematical semantics | Current project state |
|---|---|---|---|
| Sequential positions | `positions[T]` | `positions[t]=start+t`. | Exact semantic Op exists; no retained roofline qualification for the required routes. |
| Position offset | `source[T]`, scalar `delta`, `destination[T]` | `destination[t]=source[t]+delta`; used to continue scalar decode after multimodal prefill. | Exact semantic Op exists; no retained roofline qualification. |
| Token embedding gather | table `[248320,2048]`, ids `[T]` -> `x[2048,T]` | `x[:,t]=table[ids[t],:]^T`. | Embedding machinery exists, but its accepted table domain and measured shape are the 27B path; there is no exact 35B high-performance gather path or evidence. |
| Visual embedding scatter | visual `[2048,V]`, indices `[V]`, Text embedding `[2048,T]` | For every media column `j`, `x[:,indices[j]]=visual[:,j]`; all other Text columns are unchanged. | Column-tiled scatter exists, but retained evidence is for width 5120 and does not reach the exact 35B shape. |
| Offset RMSNorm, hidden | `x[2048,T]`, `w[2048]` -> `y[2048,T]`; 81 calls per main Text pass, plus MTP stem/block/final calls | `y[:,t]=OffsetRMS(x[:,t],w)`. | Semantic Op exists. The dedicated measured path is width 5120; width 2048 uses an unqualified route. |
| Offset RMSNorm, attention Q/K | `q[256,16,T]`, `k[256,2,T]`, `wq,wk[256]` | Apply `OffsetRMS` independently to each 256-element head. | Semantic Op exists, but no retained exact 35B performance qualification. |
| Residual add, Text | two `[2048,T]` tensors; 80 calls per main Text pass | `x'=x+y`. | Semantic Op and an optimized 27B path exist; the width-2048 routes have no exact roofline evidence. |
| General linear projection | `W[N,K]`, `x[K,T]` -> `y[N,T]` | `y=W x`. | A generic matrix-projection planner exists. Generic execution is not support; the required fixed shapes are listed separately below. |
| Full output head | weight `[248320,2048]`, hidden `[2048,C]` -> logits `[248320,C]`, `1<=C<=6` | `logits=W_head*hidden`. Only rows `0..248076` participate in token policy. | Vocab-head kernels exist for `K=5120`; no fixed `K=2048` head kernel or exact evidence exists. |
| Draft shortlist head | weight `[131072,2048]`, hidden `[2048,1]` -> logits `[131072,1]` | `draft_logits=W_draft*hidden`; shortlist row `i` names the token in `id_map[i]`. | Draft-head kernel exists for `K=5120`; no fixed `K=2048` specialization or exact evidence exists. |
| Argmax | full logits `[248320,C]` with 248077 valid rows, or shortlist logits `[131072,C]` with all rows valid -> ids `[C]` | `id[c]=min argmax_{0<=r<R} logits[r,c]`, where R is the valid-row count. | Exact-vocabulary implementation and benchmarks exist, but no retained roofline evidence for the current path. |
| Token sampling | logits `[248320,C]`, valid rows `R=248077`, occurrence counts and explicit random keys -> ids `[C]` | Let `a_v=logits_v-presence*1[count_v>0]-frequency*count_v`. Sort by descending `a_v`, breaking ties by lower token id. Keep configured top-k when it is 1..19, otherwise keep `min(20,R)`. Set `w_v=exp(a_v/temperature-max a)`, remove the suffix below `min_p*max(w)` when min-p is enabled, then keep the shortest remaining prefix whose cumulative weight reaches top-p times the sum of all top-k weights before min-p/top-p truncation. Retain at least the best token, normalize, and draw using the explicit `(seed,position,purpose)` key. A stochastic draw increments its occurrence count; nonpositive temperature reduces to argmax and does not update counts. | Exact-vocabulary implementation exists, but the current multi-stage path has no retained roofline evidence. |

The required fixed linear shapes not qualified by the current planner are:

| Role | Projection shape | Mathematical result |
|---|---|---|
| Full-attention input | `[9216,2048] x [2048,T]` | Logical `Q[256,16,T]`, `K[256,2,T]`, `Gate[256,16,T]`, and `V[256,2,T]`. |
| Full-attention output | `[2048,4096] x [4096,T]` | Mixer update `[2048,T]`, followed by the Text residual. |
| GDN Q/K/V/Z input | `[12288,2048] x [2048,T]` | `Q[128,16,T]`, `K[128,16,T]`, `V[128,32,T]`, and `Z[128,32,T]`. |
| GDN A/B controls | `[64,2048] x [2048,T]` | `A[32,T]` and `B[32,T]`. |
| GDN output | `[2048,4096] x [4096,T]` | Mixer update `[2048,T]`, followed by the Text residual. |
| MTP stem | `[2048,4096] x [4096,T]` | MTP block input `[2048,T]`. |
| Full output head | `[248320,2048] x [2048,C]` | Full target logits `[248320,C]`. |
| Draft shortlist head | `[131072,2048] x [2048,1]` | Draft logits `[131072,1]`. |

The current linear shape registry contains only 27B Text/MTP specializations; therefore every row
above remains in **adapt existing**, even where a shape-generic kernel can execute it.

## 4. Full Attention operators: adapt existing

The main Text model calls this subgraph in layers `3,7,...,39`; MTP calls the same geometry once
with an independent KV state. Every row is currently **unsupported** for 35B.

| Operator | Exact required shape and state | Mathematical semantics | Current project state |
|---|---|---|---|
| Q/K/Gate/V input projection | input `[2048,T]`; output `Q,Gate[256,16,T]`, `K,V[256,2,T]` | Let `p=W_qkgv*x` with 9216 rows. Rows `[0,4096)`, `[4096,4608)`, `[4608,8704)`, and `[8704,9216)` are respectively reshaped to Q, K, Gate, and V. | A semantically similar 27B fused projection exists, but it is fixed to hidden 5120 and `24Q/4KV`; no 35B kernel or evidence exists. |
| Partial interleaved MRoPE | Q `[256,16,T]`, K `[256,2,T]`, positions `[T]` or `[T,3]`; first 64 dimensions | Pair `j<32` is dimensions `(j,j+32)`, with `phi=position*10000000^(-2j/64)`, and transforms as `(a,b)'=(a*cos(phi)-b*sin(phi), b*cos(phi)+a*sin(phi))`; dimensions 64..255 are unchanged. With three-axis positions, pair `j` uses `positions[t,j mod 3]`, yielding temporal/height/width counts `[11,11,10]`. | RoPE Op exists, but its Text contract and optimized evidence are fixed to `24Q/4KV`; exact `16Q/2KV` is unqualified. |
| KV append | K/V `[256,2,T]`; per-layer K/V state `[256,C,2]` | At absolute position `p_t`, set `Kcache[:,p_t,h]=K[:,h,t]` and `Vcache[:,p_t,h]=V[:,h,t]` for both KV heads. | KV append machinery exists for `4KV` heads, not the exact 2-head geometry. No exact evidence exists for either registered cache representation. |
| Causal grouped-query attention with append | Q `[256,16,T]`, new K/V `[256,2,T]`, visible cache length `L` -> O `[256,16,T]` | `kvh=floor(h/8)`; for visible causal keys `s`, `score[h,t,s]=dot(Q[:,h,t],Kcache[:,s,kvh])/sqrt(256)` and `O[:,h,t]=sum_s softmax(score)[s]*Vcache[:,s,kvh]`. The call first makes the new K/V columns visible through the append above. | Current GQA is fixed to `24Q/4KV` and ratio 6. Decode, `T<=6`, prefill, and both cache representations all need exact 16/2 qualification. |
| Cached-only GQA | Q `[256,16,T]`, populated K/V `[256,L,2]` -> O `[256,16,T]` | Same grouped causal attention equation without a cache write. | A 27B cached-only path exists; head geometry and evidence do not match 35B. |
| Attention output gate | Gate and O `[256,16,T]` | `O'=sigmoid(Gate).*O`. | Semantic elementwise Op exists. Retained optimized evidence is for width 6144 rather than the required width 4096. |
| Attention output projection plus residual | input `[4096,T]`, weight `[2048,4096]`, residual `[2048,T]` | `x'=x+W_o*flatten(O')`. | Similar 27B projection/residual machinery exists; no exact 35B specialization or evidence exists. |

The standalone KV-append and cached-only forms are required by MTP prefill/alignment in addition to
the ordinary append-and-attend form. Treating only one of the three entry points as qualified does
not support the complete 35B attention graph.

## 5. Gated-DeltaNet operators: adapt existing

Thirty Text layers execute this subgraph. Every row is currently **unsupported** for the 35B
target.

| Operator | Exact required shape and state | Mathematical semantics | Current project state |
|---|---|---|---|
| Q/K/V/Z input projection | input `[2048,T]`; Q/K `[128,16,T]`; V/Z `[128,32,T]` | `p=W_qkvz*x`; reshape rows `[0,2048)`, `[2048,4096)`, `[4096,8192)`, and `[8192,12288)` to Q, K, V, and Z. Z bypasses the convolution. | Similar fused 27B input machinery exists, but its hidden, row, and V-head dimensions differ; no 35B kernel or evidence exists. |
| A/B control projection and transforms | input `[2048,T]`; A/B `[32,T]`; parameters `[32]`; outputs `g,beta[32,T]` | `AB=W_ab*x`, split into A and B, then `g[j,t]=-exp(A_log[j])*softplus(A[j,t]+dt_bias[j])` and `beta[j,t]=sigmoid(B[j,t])`. | Fused 27B control machinery is fixed to 48 heads and hidden 5120. Exact 32-by-2048 support and evidence are absent. |
| Depthwise causal convolution plus SiLU | Q/K/V concatenation `U[8192,T]`, weights `[8192,4]`, history `[8192,3]` | With history supplying `U[c,-3..-1]`, `C[c,t]=SiLU(sum_{r=0}^3 W[c,r]*U[c,t-3+r])`. New history is the final three columns of the concatenated old history and U. | Width-4 stateful kernels exist, with in-place, distinct-state, and snapshot forms. Existing performance work covers the 27B channel count, not the exact 8192-channel domain. |
| Q/K per-head L2 normalization | Q and K independently `[128,16,T]` | For every head/token vector, `y=x/sqrt(sum_d x_d^2+eps)`. | The head geometry matches the 27B path, but the retained performance point uses `T=4096`, outside the target's `T<=1024` prefill route; decode and `T<=6` are also unqualified. |
| Gated delta-rule recurrence | Q/K `[128,16,T]`, V/O `[128,32,T]`, controls `[32,T]`, state `[128,128,32]` | For V head `j`, let `S_j=state[:,:,j]` and use Q/K head `floor(j/2)`. In token order: `Sbar=exp(g[j,t])*S_j`; `delta=beta[j,t]*(v-Sbar*k)`; `S_j'=Sbar+delta*k^T`; `o=(S_j'*q)/sqrt(128)`. | Current recurrent/chunked/snapshot Ops are fixed to 48 V heads with mapping `floor(j/3)`. All 32-head execution forms require adaptation and qualification. |
| Gated plain RMSNorm | O/Z `[128,32,T]`, weight `[128]` | Per V head, `n=PlainRMS(O,w)` and `y=n.*SiLU(Z)`. | Semantic Op exists, but no exact 32-head roofline evidence exists. |
| GDN output projection plus residual | input `[4096,T]`, weight `[2048,4096]`, residual `[2048,T]` | `x'=x+W_o*flatten(y)`. | Similar 27B projection/residual machinery exists; no exact 35B specialization or evidence exists. |

The convolution and recurrence each require their ordinary in-place, distinct input/output state,
and speculative snapshot variants. With maximum draft count five, snapshot storage has seven
logical slots: convolution snapshots are `[8192,3,7]` and recurrent snapshots are
`[128,128,32,7]`. After token `t`, the snapshot variants publish the resulting state in slot `t`
while leaving later slots unchanged.

## 6. Vision operators requiring adaptation or qualification

The four supported Vision domains are in Section 2. Every row below is currently **unsupported**
under the strict definition, even though most of the Vision graph runs through existing 27B Ops.

| Operator | Exact required shape and multiplicity | Mathematical semantics | Current project state |
|---|---|---|---|
| Vision LayerNorm outside the measured point | `x[1152,P]` with `P!=4096` | Same affine LayerNorm equation as Section 2. | The fixed-width kernel exists, but no retained roofline evidence covers these exact patch-count domains. |
| Vision QKV bias outside the measured point | `x[3456,P]` with `P!=4096` | `x'[d,p]=x[d,p]+b[d]`. | The paired kernel exists, but no retained roofline evidence covers these exact patch-count domains. |
| Vision block tanh GELU outside the measured point | `x[4304,P]` with `P!=4096` | Same tanh GELU equation as Section 2. | The paired kernel exists, but no retained roofline evidence covers these exact patch-count domains. |
| Vision attention outside the roofline-qualified segment | Q/K/V/O `[72,16,P]` with segment length other than 4096 or packed `S>1` cases | For each independent segment, use the segmented softmax-attention equation in Section 2 over that segment's bounds. | The fixed Flash kernel exists, but retained roofline evidence covers only the exact single-segment 4096 domain in Section 2. |
| Patch ingress conversion | patch matrix `[1536,P]` -> activation matrix `[1536,P]` | Convert each packed patch scalar into the model activation representation without changing its logical index. | Exact semantic conversion exists, but only correctness evidence is retained. |
| Patch projection | `[1152,1536] x [1536,P]` -> `[1152,P]`; once per media item | `x=W_patch*pixels`. | Generic linear execution exists; no exact-shape roofline evidence exists. |
| Non-QKV bias adds | widths `1152`, `4304`, `4608`, and `2048` | For each required width D, `x'[d,p]=x[d,p]+b[d]`. | Same optimized elementwise machinery exists, but retained roofline evidence covers only QKV width 3456. |
| Learned position interpolation and add | table `[1152,2304]`, four indices/weights per patch, x `[1152,P]` | `pos[:,p]=sum_{c=0}^3 lambda[c,p]*table[:,index[c,p]]`; then `x[:,p]'=x[:,p]+pos[:,p]`. | A fixed-width optimized kernel exists, but its report explicitly does not establish a roofline; it remains unqualified. |
| Vision QKV projection | `[3456,1152] x [1152,P]` -> `[3456,P]`; 27 calls | `qkv=W_qkv*h`, then view as Q/K/V `[72,16,P]`. | Generic linear route exists; no exact-shape roofline evidence exists. |
| Vision 2-D RoPE | Q/K `[72,16,P]`, positions `[P,2]`; 27 calls | Pair `j<36` is dimensions `(j,j+36)`. Its axis is 0 for `j<18` and 1 otherwise; `phi=positions[p,axis]*10000^(-2*(j mod 18)/36)`, followed by the rotation equation from Section 4. | Exact optimized geometry exists, but its retained report explicitly declines to use the measurements as roofline evidence. |
| Vision attention output projection | `[1152,1152] x [1152,P]` -> `[1152,P]`; 27 calls | `y=W_o*flatten(attention)`. | Generic linear route exists; no exact-shape roofline evidence exists. |
| Vision residual add | two `[1152,P]` tensors; 54 calls | `x'=x+y`. | Semantic Op exists; no exact width-1152 roofline evidence exists. |
| Vision block MLP fc1 | `[4304,1152] x [1152,P]` -> `[4304,P]`; 27 calls | `u=W_1*h`; the supported tanh GELU follows after its bias. | Generic linear route exists; no exact-shape roofline evidence exists. |
| Vision block MLP fc2 | `[1152,4304] x [4304,P]` -> `[1152,P]`; 27 calls | `y=W_2*GELU_tanh(u)`. | Generic linear route exists; no exact-shape roofline evidence exists. |
| Merger fc1 | `[4608,4608] x [4608,V]` -> `[4608,V]`; once per media item | `u=W_m1*merged`. | Generic linear route exists; no exact-shape roofline evidence exists. |
| Merger exact GELU | `[4608,V]`; once per media item | `y=0.5*u.*(1+erf(u/sqrt(2)))`. | The current optimized GELU kernel has exact-mode evidence only at width 4304, not the required 4608 domain. |
| Merger fc2 | `[2048,4608] x [4608,V]` -> `[2048,V]`; once per media item | `visual=W_m2*y`. | The 27B merger output has width 5120. Generic execution is possible, but no exact width-2048 specialization or evidence exists. |

Vision height/width position ids and segment bounds are built by the target frontend and uploaded as
already-owned metadata. Their construction is not a device Op. Q/K/V splits and the merge-major
reshape are views when the stated index prerequisites hold.

## 7. MTP-only operators: adapt existing

MTP reuses the 35B token embedding, full/draft heads, one full-attention block, and one sparse-MoE
block. Those reused operators retain the statuses in Sections 3, 4, and 8. The operators below are
MTP-specific and currently **unsupported** under the strict performance definition.

| Operator | Exact required shape | Mathematical semantics | Current project state |
|---|---|---|---|
| MTP stem pack | normalized embedding and target hidden `[2048,T]` -> `[4096,T]` | `packed[0:2048,:]=embedding_norm`; `packed[2048:4096,:]=hidden_norm`. | Exact semantic Op exists only with fixed 27B widths 5120/10240. It requires a 2048/4096 path and qualification. |
| MTP stem projection | `[2048,4096] x [4096,T]` -> `[2048,T]` | `u=W_fc*concat(OffsetRMS(embed(x_{t+1})),OffsetRMS(h_t))`. | Generic linear execution exists; no exact specialization or evidence exists. |
| Prepare verification inputs | token scalar, drafts `[K]`, length scalar -> verify ids/positions `[K+1]`, `K<=5` | `verify_ids[0]=token`; `verify_ids[i+1]=drafts[i]`; `positions[i]=length+i`; `window_base=length`. | Exact K-window semantic Op exists, but no retained roofline qualification exists. |
| Accept verification tokens | drafts `[K]`, target ids/logit columns `[K+1]`, output `[K+1]`, statistics `[>=9]` | Greedy mode accepts the longest prefix whose drafts equal target argmax and then emits the first mismatch or bonus target token. Sampling mode accepts deterministic draft `d_i` with target probability `p_i(d_i)`; on rejection it samples the normalized target distribution with `d_i` removed, and samples a bonus from column K if all drafts are accepted. If A drafts are accepted, output slots `0..A-1` are those drafts, slot A is the correction/bonus, and later slots are zero; `num_sampled=A+1`, `accepted=A`, `token=output[A]`, `length'=length+A+1`, and `ar_pos'=length'`. Statistics apply `stats[0]+=K`, `stats[1]+=A`, `stats[2]+=1`, and `stats[4+i]+=1` for `0<=i<min(A,5)`; stochastic produced tokens increment occurrence counts. | Exact semantic implementation exists, but neither greedy nor sampling route has retained roofline evidence for the current path. |
| Prepare shifted MTP ids | verify ids `[K+1]`, accepted scalar A, correction/bonus token -> shifted ids `[K+1]` | `shifted[i]=verify_ids[i+1]` for `i<K`, then `shifted[A]=token`; slot K changes only when `A=K`. | Exact semantic Op exists; no retained performance qualification. |
| Gather accepted hidden | target hidden `[2048,K+1]`, accepted scalar A -> `[2048,1]` | `out[:,0]=hidden[:,A]`. | Existing Op is dimension-generic; exact width-2048 evidence is absent. |
| Draft token remap | draft scalar and id map `[131072]` | `draft'=id_map[draft]`. | Exact semantic Op exists; no retained performance qualification. |
| Scalar state assignments | integer scalars for token, length, positions, accepted count, and graph control | `dst'=value`, `dst'=src`, or `dst'=dst+1`, according to the call. | Exact semantic Ops exist; no retained performance qualification. |

Target verification itself is not a separate model-math Op: it is one main Text invocation with
`T=K+1<=6`, followed by `Accept verification tokens`. MTP cache selection, target state commit,
rollback, and rebuild ordering are Program policy rather than Ops.

## 8. Sparse MoE: new implementation

Sparse MoE is required after every one of the 40 main Text mixers and once in MTP. NInfer has no
`SparseMoe` Op or expert-grouped kernel implementation today. Generic linear calls, one launch per
selected expert, or an all-expert scan do not constitute support.

### 8.1 Closed SparseMoe semantics

For normalized input `x[2048,T]`, the router and independent shared score are:

```text
r = W_router_shared * x                 # [257,T]
logits = r[0:256,:]
shared_score = r[256,:]

p[e,t] = exp(logits[e,t]) / sum_j exp(logits[j,t])
I_t = indices of the eight largest p[:,t]
alpha[e,t] = p[e,t] / sum_{j in I_t} p[j,t],  e in I_t
```

Each routed expert and the shared expert have intermediate width 512:

```text
Expert_e(x) = W_down[e] * SwiGLU(W_gate[e]*x, W_up[e]*x)

Shared(x) = W_shared_down
            * SwiGLU(W_shared_gate*x, W_shared_up*x)

SparseMoe(x[:,t]) = sum_{e in I_t} alpha[e,t] * Expert_e(x[:,t])
                    + sigmoid(shared_score[t]) * Shared(x[:,t])
```

The enclosing layer then performs `x'=x+SparseMoe(h)`. SparseMoe is one semantically closed Op:
router selection, all selected experts, the always-on shared expert, and the final reduction are
not target-schedule calls.

### 8.2 Current project state

**New implementation.** NInfer has no semantic `SparseMoe` Op, router/top-8 kernel, selected-expert
decode path, assignment grouping, expert-grouped matrix multiplication, inverse scatter, or routed
reduction. For `T>1`, the closed Op requires private grouped work equivalent to
`[1024,2048] x [2048,M_e]` followed by SwiGLU and `[2048,512] x [512,M_e]` for every active expert,
with `sum_e M_e=8T`; these grouped stages are not separately target-callable Ops. Decode (`T=1`),
MTP Small-T (`T<=6`), and prefill (`T<=1024`) are distinct required routes. Generic linear calls,
one host launch per expert, or an all-expert scan support none of them.

## 9. Completeness summary

The complete logical call inventory for one main Text invocation is:

```text
1 embedding gather
40 input OffsetRMS
10 Full Attention mixers
30 GDN mixers
40 mixer residual updates
40 post-mixer OffsetRMS
40 SparseMoe calls = 320T routed assignments
40 MoE residual updates
1 final OffsetRMS
1 full target head when target logits are required
1 token-policy operation for every emitted target decision
```

One MTP invocation adds:

```text
2 stem OffsetRMS calls
1 stem pack and projection
1 block input OffsetRMS
1 Full Attention mixer with independent KV state
1 attention residual update
1 block post-attention OffsetRMS
1 SparseMoe call = 8T routed assignments
1 MoE residual update
1 final OffsetRMS
1 full or shortlist proposal head
the K<=5 verification-control operators in Section 7
```

One media-item Vision invocation adds:

```text
1 patch ingress and projection
1 learned-position interpolation/add
54 Vision LayerNorm calls
27 QKV projections, QKV bias adds, 2-D RoPE calls, and segmented-attention calls
27 attention-output projections and residual updates
27 fc1 projections, tanh GELUs, fc2 projections, and MLP residual updates
1 merger LayerNorm, merge-major view, fc1, exact GELU, and fc2
1 scatter into Text embeddings
```

Under the support definition in Section 1, the current result is:

| Category | Result |
|---|---|
| Supported | Four exact `P=4096` Vision operator domains: width-1152 LayerNorm, width-3456 QKV bias, width-4304 tanh GELU, and one-segment `16 x 72` attention. |
| Adapt existing | Every non-MoE Text, Full Attention, GDN, generation, MTP-control, and remaining Vision operator. These remain unsupported until their exact routes are specialized where necessary and roofline-qualified. |
| New implementation | The closed SparseMoe Op and its decode, Small-T, assignment/grouping, grouped-GEMM, inverse-scatter, and reduction kernels. |
