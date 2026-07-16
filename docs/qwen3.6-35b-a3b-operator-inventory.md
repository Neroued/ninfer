# Qwen3.6-35B-A3B Operator Inventory

> Status: authoritative operation inventory and current implementation-support matrix for the
> future `qwen3_6_35b_a3b_rtx5090` target.

This document is organized by the operation that an Op or kernel performs, not by the model block
that happens to call it. An attention input projection, a GDN input projection, an MTP projection,
and a Vision projection therefore appear under the Linear family. Their call sites remain in the
tables only to establish the exact domains and multiplicities that the implementation must cover.

The checkpoint graph and dimensions come from
[`qwen3.6-35b-a3b-architecture.md`](qwen3.6-35b-a3b-architecture.md). Persistent weight formats and
target storage come from
[`qwen3.6-35b-a3b-ninfer-artifact.md`](qwen3.6-35b-a3b-ninfer-artifact.md). Op boundaries follow
[`op-development.md`](op-development.md): a semantically closed target-callable transformation is
an Op, while a grouped GEMM or partial reduction inside the closed `SparseMoeAdd` Op is an
implementation-private stage. This inventory still lists those private MoE stages because they
are independent, performance-critical kernel work and cannot be replaced by generic Linear calls.

The tables include logical tensor types and registered weight formats because they determine the
implementation domain. They do not define rounding, accumulation precision, tolerance, tensor
layout, fusion policy, memory budgets, implementation order, or a test plan.

## 1. Scope, status, and notation

### 1.1 Support definition

The status is deliberately stricter than API availability or functional execution:

- **Supported** means that the exact 35B logical domain and required execution route select a
  current high-performance kernel, and retained RTX 5090 evidence establishes that kernel at its
  applicable bandwidth, compute, or fixed-resource roofline. A dimension-generic kernel counts
  only for exact domains with that evidence.
- **Adapt existing** means that the same mathematical Op, a close fixed-shape kernel, or reusable
  implementation machinery exists, but the exact 35B domain, one required execution route, or
  roofline qualification is missing. The domain remains **unsupported** until that gap is closed.
- **New implementation** means that no suitable high-performance Op or kernel algorithm exists.
  The domain is **unsupported** and must not be implemented as a sequence of generic fallbacks or
  many model-role-specific small launches.

A generic/reference CUDA path, Python implementation, correctness test, benchmark binary without
retained results, or evidence for another shape does not establish support. Stage-level reuse
inside `SparseMoeAdd` also does not make the closed Op partially supported.

The bounded control/copy rows I4, I5, and I7-I12 are the explicit exception to standalone
roofline qualification. Their complete registered work is at most six MTP columns or 1024 I32
positions, and their performance concern is launch topology and future fusion rather than a
shape-specific kernel algorithm. For these rows, **Supported** means that the exact domain selects
a direct device implementation and is covered by exact correctness checks; no separate timing
report is required.

### 1.2 Logical notation and types

NInfer's matrix convention is used throughout:

```text
X[K,T]                    T token columns, each with K features
W[N,K]                    linear weight
Linear(W,X) = W X         result [N,T]

H  = 2048                 Text hidden size
Vm = 248320               model vocabulary rows
Vt = 248077               valid token-policy rows
T  = 1                    ordinary decode
T  <= 6                   target verification and MTP rebuild Small-T
T  <= 1024                one Text prefill chunk
L  <= 262144              visible Text context

P = g_t*g_h*g_w            raw Vision patches for one media item
S = g_t                    Vision attention segments; each has g_h*g_w patches
V = P / 4                  merged Vision tokens; g_h and g_w are divisible by 2
A  = 8T                    routed expert assignments
M_e                        assignments routed to expert e; sum_e M_e = 8T
eps = 1e-6                 normalization epsilon
```

The accepted frontend budgets bound one image item at `P<=65536` and one video item at
`P<=49152`. All Vision rows cover one item at a time within those bounds.

Unless a row says otherwise, activation inputs and outputs are BF16. Positions, token ids, expert
ids, and index maps are I32. GDN recurrent state and routing probabilities are FP32. Registered
quantized weights use `row-split-k128-v1` with the named `Q4G64_F16S`, `Q5G64_F16S`,
`Q6G64_F16S`, or `W8G32_F16S` format; direct weights use contiguous BF16 or FP32.
Tables abbreviate these four registered formats as Q4, Q5, Q6, and W8 respectively.

Host construction of multimodal metadata, raw transfers, allocation, published cache frontiers, state
lifetime, graph capture, commit/rollback policy, and zero-work tensor views are not Ops. They are
not included as implementation support.

## 2. Linear, GEMM, and projection operations

The base operation is independent of model role:

```text
Linear:       Y[:,t]  = W X[:,t]
LinearAdd:    R[:,t]' = R[:,t] + W X[:,t]
LinearSwiGLU: G,U     = split(W_gu X)
              Y       = SiLU(G) .* U
GroupedLinear job e:  Y_e = W_e X_e
```

A packed parent weight with several logical outputs is still a Linear operation. Its output
topology is part of the exact domain: a high-performance implementation must write each declared
output view, but the model names Q, K, V, gate, or Z do not create new arithmetic families.

### 2.1 Target-callable dense and quantized Linear domains

| ID | Operation and exact typed domain | Call sites | Mathematical result | Current support |
|---|---|---|---|---|
| L1 | Multi-output Linear; W8 weight `[9216,2048]`, `X[2048,T]` | 10 Text full-attention layers and one MTP layer; all three T regimes | `Y=W X`; rows `0:4096`, `4096:4608`, `4608:8704`, and `8704:9216` produce Q `[256,16,T]`, K `[256,2,T]`, gate `[256,16,T]`, and V `[256,2,T]`. | **Adapt existing.** The current attention-input Op and kernels are fixed to the 27B `5120 -> 24Q/4KV` domain and different formats. |
| L2 | Multi-output Linear; W8 weight `[12288,2048]`, `X[2048,T]` | 30 GDN layers; all three T regimes | `Y=W X`; row widths `2048,2048,4096,4096` produce Q `[128,16,T]`, K `[128,16,T]`, V `[128,32,T]`, and Z `[128,32,T]`. | **Adapt existing.** The same operation exists for the 27B widths, but not this W8 parent and output geometry. |
| L3 | Linear with GDN-control epilogue; BF16 weight `[64,2048]`, FP32 `A_log,dt_bias[32]`, `X[2048,T]` -> FP32 `g,beta[32,T]` | 30 GDN layers; all three T regimes | `AB=W X`, split A/B into 32 rows each; `g[j,t]=-exp(A_log[j])*softplus(A[j,t]+dt_bias[j])`, `beta[j,t]=sigmoid(B[j,t])`. | **Adapt existing.** The closed projection/control Op exists for 48 rows over hidden 5120; the exact 32-by-2048 route is absent. |
| L4 | LinearAdd; W8 weight `[2048,4096]`, `X[4096,T]`, residual `[2048,T]` | All 40 Text mixer outputs and the MTP attention output | `residual'=residual+W X`. Attention, GDN, and MTP are caller roles of this one exact operation domain. | **Adapt existing.** `LinearAdd` exists for 27B Q5 output shapes, not this W8 domain. |
| L5 | Linear; W8 weight `[2048,4096]`, packed input `[4096,T]` -> `[2048,T]` | MTP stem | `Y=W X`. The two RMSNorms and the concat that form X are separate Ops. | **Adapt existing.** The base format machinery exists; the exact plain-epilogue shape is unqualified. |
| L6 | Linear; Q6 weight `[248320,2048]`, hidden `[2048,C]` -> logits `[248320,C]`, `1<=C<=6` | Full target head for ordinary decisions and verification; also MTP proposal when the draft head is disabled | `logits=W_head X`; token policy later consumes only rows `0:248077`. | **Adapt existing.** The head kernel is fixed to K=5120; no K=2048 specialization or evidence exists. |
| L7 | Linear; Q4 weight `[131072,2048]`, hidden `[2048,1]` -> shortlist logits `[131072,1]` | Optimized MTP proposal | `logits=W_draft X`; the downstream I32 id map assigns each row its full token id. | **Supported for the complete target domain.** The exact singleton route uses the runtime-K Q4 GEMV winner, sustains 1350.91 GB/s or 89.58% of the same-session cold-copy ceiling, and is profiler-confirmed DRAM-bound without spilling. See the [retained RTX 5090 report](archive/optimization-era/bench/qwen3.6-35b-linear-roofline.md). |
| L8 | Linear; Q6 weight `[1152,1536]`, patches `[1536,P]` -> `[1152,P]` | Vision patch projection, once per media item | `Y=W_patch X`. | **Supported for the complete target domain.** The exact SIMT/MMA64/MMA128 route covers every admitted patch count through maximum video/image `P=49152/65536`. The maxima sustain 197.22/197.98 executed TFLOP/s, respectively 94.08%/94.44% of the same-session BF16 MMA probe; smaller routes select the measured fixed-resource winner. See the [retained RTX 5090 report](archive/optimization-era/bench/qwen3.6-35b-linear-roofline.md). |
| L9 | Linear; Q4 weight `[3456,1152]`, `X[1152,P]` -> `[3456,P]` | Vision QKV, 27 calls per item | `Y=W_qkv X`; the three 1152-row regions are Q/K/V views. | **Supported for the complete target domain.** The exact SIMT/MMA64/MMA128 route covers every admitted patch count through maximum video/image `P=49152/65536`. NCU measures the two maxima at 87.36%/86.68% Compute SOL with the tensor pipeline limiting both, while smaller domains select their matched fixed-resource winner. See the [retained RTX 5090 report](archive/optimization-era/bench/qwen3.6-35b-linear-roofline.md). |
| L10 | Linear; Q5 weight `[1152,1152]`, `X[1152,P]` -> `[1152,P]` | Vision attention output, 27 calls per item | `Y=W_o X`. | **Supported for the complete target domain.** A measured SIMT/MMA64/MMA128 route covers every admitted patch count through maximum video/image `P=49152/65536`. The maxima sustain 194.42/194.80 executed TFLOP/s and reach 86.59%/87.00% Compute SOL with the tensor pipeline limiting both. See the [retained RTX 5090 report](archive/optimization-era/bench/qwen3.6-35b-linear-roofline.md). |
| L11 | Linear; Q4 weight `[4304,1152]`, `X[1152,P]` -> `[4304,P]` | Vision block MLP fc1, 27 calls per item | `Y=W_1 X`. Bias and GELU are separate operations in Section 5. | **Supported for the complete target domain.** The exact SIMT/MMA64/MMA128 route covers every admitted patch count through maximum video/image `P=49152/65536`. The predicated maxima remain within 3.16%/3.62% of a same-executed-work Full-row control; NCU identifies the tensor pipeline as the limiter and reports no spilling. See the [retained RTX 5090 report](archive/optimization-era/bench/qwen3.6-35b-linear-roofline.md). |
| L12 | Linear; Q5 weight `[1152,4304]`, `X[4304,P]` -> `[1152,P]` | Vision block MLP fc2, 27 calls per item | `Y=W_2 X`; logical K is 4304 even though registered storage pads K to 4352. | **Supported for the complete target domain.** The exact SIMT/MMA64/MMA128 route covers every admitted patch count through maximum video/image `P=49152/65536`. The K-tail maxima remain within 6.19%/8.08% of a same-topology Full-K control, are tensor-pipe limited, and have no spilling. See the [retained RTX 5090 report](archive/optimization-era/bench/qwen3.6-35b-linear-roofline.md). |
| L13 | Linear; W8 weight `[4608,4608]`, `X[4608,V]` -> `[4608,V]` | Vision merger fc1, once per item | `Y=W_m1 X`. | **Supported for the complete target domain.** The exact SIMT/MMA32/MMA64 route covers every merger column count through maximum video/image `V=12288/16384`. The maxima sustain 167.75/167.99 executed TFLOP/s, respectively -0.19%/+2.21% versus the same-K, same-format W8 online-decode/MMA ceiling; NCU confirms the same resource envelope and no spilling. See the [retained RTX 5090 report](archive/optimization-era/bench/qwen3.6-35b-linear-roofline.md). |
| L14 | Linear; W8 weight `[2048,4608]`, `X[4608,V]` -> visual `[2048,V]` | Vision merger fc2, once per item | `Y=W_m2 X`. | **Supported for the complete target domain.** The exact finite SIMT/MMA32/MMA64 route covers every merger column count through maximum video/image `V=12288/16384`. The maxima sustain 184.55/187.66 executed TFLOP/s, or 88.31%/91.42% of the same-session BF16 MMA probe; NCU identifies the maximum route as tensor-pipe limited and reports no spilling. See the [retained RTX 5090 report](archive/optimization-era/bench/qwen3.6-35b-linear-roofline.md). |

### 2.2 SparseMoeAdd-private Linear and expert-grouped domains

All rows in this subsection are private stages of the closed sparse Op in Section 9, not
target-callable Linear Ops. They remain classified here because the arithmetic to implement is
Linear, LinearSwiGLU, or expert-grouped Linear rather than a model-role-specific projection.

| ID | Operation and exact typed domain | Execution regimes | Mathematical result | Current support |
|---|---|---|---|---|
| L15 | Linear; BF16 weight `[257,2048]`, `X[2048,T]` -> BF16 scores `[257,T]` | All three T regimes | Rows `0:256` are router logits and row 256 is the independent shared-expert score. | **Adapt existing.** Dense projection machinery is reusable, but there is no qualified exact 257-row path. This stage alone does not support `SparseMoeAdd`. |
| L16 | LinearSwiGLU; W8 weight `[1024,2048]`, `X[2048,T]` -> `[512,T]` | All three T regimes | Split `W X` into gate/up `[512,T]`; output `SiLU(gate).*up` for the always-on shared expert. | **Adapt existing.** The semantic operation and low-bit machinery exist, but not this W8 exact domain. |
| L17 | Linear; W8 weight `[2048,512]`, `X[512,T]` -> `[2048,T]` | All three T regimes | `Y_shared=W_shared_down X`. | **Adapt existing.** W8 Linear machinery exists without an exact qualified K=512 path. |
| L18 | Expert-grouped LinearSwiGLU; per expert `W_gu[e][1024,2048]`, `X_e[2048,M_e]` -> `[512,M_e]` | Decode `M_e in {0,1}`; Small-T `sum M_e<=48`; prefill `sum M_e<=8192` | For every active expert, split `W_gu[e]X_e` into 512-row gate/up halves and return `SiLU(gate_e).*up_e`. | **New implementation.** No selected-expert or device-described grouped path exists. Main Text uses Q4 banks; MTP uses W8 banks. |
| L19 | Expert-grouped Linear; per expert `W_down[e][2048,512]`, `X_e[512,M_e]` -> `[2048,M_e]` | Same three regimes | `Y_e=W_down[e]X_e` for every active expert. | **New implementation.** No K=512 grouped expert kernel exists. Main Text uses Q5 except Q6 in layers 34/38/39; MTP uses W8. |

The physical parents are `[262144,2048]` for 256 gate/up experts and `[524288,512]` for 256
down experts. Expert id selects an equal-stride view directly; gathering or repacking selected
weights is not another supported route.

## 3. Indexing and logical tensor transforms

| ID | Operation and exact typed domain | Mathematical semantics | Call sites | Current support |
|---|---|---|---|---|
| I1 | FP32-to-BF16 cast, `[1536,P]` -> `[1536,P]` | `dst[k,p]=BF16(src[k,p])` without changing logical indices. | Vision patch ingress | **Supported for the complete target domain.** A dimension-generic aligned x4 route covers every valid Vision patch count through maximum video/image `P=49152/65536`; it remains within 1.3% of the same-grid, same-payload control across the retained matrix and reaches about 1.53 TB/s at maximum image size. See the [retained RTX 5090 report](archive/optimization-era/bench/qwen3.6-35b-fp32-bf16-cast-roofline.md). |
| I2 | Embedding row gather; W8 table `[248320,2048]`, ids `[T]` -> BF16 `[2048,T]` | `out[:,t]=table[ids[t],:]^T`. | Text and MTP share the same table | **Supported for the complete target domain.** Exact W8G32 group-parallel Small-T and vector row-prefill routes cover `T=1..6/1024`; production remains within 3.9% of same-grid, same-payload controls. See the [retained RTX 5090 report](archive/optimization-era/bench/qwen3.6-35b-embedding-w8-roofline.md). |
| I3 | Column scatter; BF16 visual `[2048,V]`, I32 indices `[V]`, destination `[2048,T]` | `dst[:,indices[j]]=visual[:,j]`; all other columns remain unchanged. | Multimodal Text/MTP prefill | **Supported for the complete target domain.** The aligned BF16x8 route covers chunk-local scatters and complete maximum video/image item sizes `V=12288/16384`; it remains within 3.6% of the same-topology payload floor over the retained matrix and is DRAM-bound at maximum image size. See the [retained RTX 5090 report](archive/optimization-era/bench/qwen3.6-35b-scatter-roofline.md). |
| I4 | Sequential I32 fill, `[T]` | `positions[t]=start+t`. | Text-only positions and verification windows | **Supported for the complete target domain.** The dimension-driven device Op covers scalar, verification, and `T=1024` position vectors directly. |
| I5 | I32 position offset, source/destination `[T]`, scalar delta | `destination[t]=source[t]+delta`. | Scalar decode after multimodal prefill | **Supported for the complete target domain.** The same direct device route covers distinct and in-place destination storage. |
| I6 | Four-corner gather, weighted interpolation, and add; BF16 table `[1152,2304]`, I32 indices `[4,P]`, FP32 weights `[4,P]`, BF16 x `[1152,P]` | `x[:,p]'=x[:,p]+sum_c weights[c,p]*table[:,indices[c,p]]`. | Vision stem | **Supported for the complete target domain.** A warp-tiled route reaches the fixed-work floor for small media, while the 128-thread patch CTA reaches 87.98% memory SOL at maximum image size; production remains within 8.3% of same-topology payload controls over the complete retained P matrix. See the [retained RTX 5090 report](archive/optimization-era/bench/qwen3.6-35b-vision-pos-embed-roofline.md). |
| I7 | MTP concat pack; two BF16 inputs `[2048,T]` -> `[4096,T]` | Rows `0:2048` copy embedding-norm output and rows `2048:4096` copy hidden-norm output. | MTP stem | **Supported for the complete target domain.** The dimension-driven single device launch packs `[D,T]+[D,T]` into `[2D,T]` and covers the exact `D=2048`, `T<=6` route while preserving `D=5120` for 27B. |
| I8 | Prepare verification ids/positions; I32 token scalar, drafts `[K]`, length scalar -> I32 ids/positions `[K+1]` and `window_base` scalar, `K<=5` | `ids[0]=token`, `ids[i+1]=drafts[i]`, `positions[i]=length+i`, and `window_base=length`. | MTP verification | **Supported for the complete target domain.** One direct device launch writes the scalar and both complete `K+1` vectors for `K<=5`. |
| I9 | Prepare shifted MTP ids; verify ids `[K+1]`, accepted scalar A, output token -> shifted `[K+1]` | `shifted[i]=verify_ids[i+1]` for `i<K`, then `shifted[A]=token`; slot K changes only when `A=K`. | MTP rebuild | **Supported for the complete target domain.** The direct device Op covers partial and full acceptance at exact `K=5`. |
| I10 | Indexed hidden-column gather; hidden `[2048,K+1]`, accepted scalar A -> `[2048,1]` | `out[:,0]=hidden[:,A]`. | MTP continuation | **Supported for the complete target domain.** The dimension-driven device gather exactly copies the selected `D=2048` BF16 column. |
| I11 | Draft-token remap; I32 scalar and map `[131072]` | `draft'=id_map[draft]`. | Optimized MTP proposal | **Supported for the complete target domain.** The direct device lookup covers the exact 131072-entry map. |
| I12 | Typed scalar assignment/increment; I32 or I64 scalar | `dst'=value`, `dst'=src`, or `dst'=dst+1`, as selected by the semantic entry point. | Token, length, position, and MTP control state | **Supported for the complete target domain.** The typed scalar entry points perform the declared device-resident transitions directly and are CUDA Graph compatible. |

Q/K/V row splits, head reshapes, the Vision 2x2 merge-major `[1152,P] -> [4608,V]`
reinterpretation, and shortlist row naming are zero-work views when their documented storage
prerequisites hold. They are not additional device Ops.

## 4. Normalization operations

Define:

```text
OffsetRMS(x,w) = (1+w) .* x / sqrt(mean(x^2) + eps)
PlainRMS(x,w)  = w .* x / sqrt(mean(x^2) + eps)
L2(x)          = x / sqrt(sum(x^2) + eps)
```

| ID | Operation and exact typed domain | Mathematical semantics and call sites | Current support |
|---|---|---|---|
| N1 | Offset RMSNorm; BF16 `x[2048,T]`, weight `[2048]` -> `[2048,T]` | Apply `OffsetRMS` independently to each column. Main Text calls it 81 times; one MTP invocation calls it five times across its stem, block, and final output. | **Supported for exactly this domain.** The aligned wide-row CTA route covers decode, Small-T, and T=1024 while remaining at its same-grid fixed-work control. See the [retained RTX 5090 report](archive/optimization-era/bench/qwen3.6-35b-norm-roofline.md). |
| N2 | Per-head Offset RMSNorm; Q `[256,16,T]`, K `[256,2,T]`, weights `[256]` | Apply `OffsetRMS` independently to every 256-element head. Ten Text attention layers and one MTP layer each call Q and K forms. | **Supported for exactly this domain.** The general warp-row kernel covers both head counts and every T route; Q prefill reaches 95.9% of the traffic roofline and K is fixed-work limited. See the [retained report](archive/optimization-era/bench/qwen3.6-35b-norm-roofline.md). |
| N3 | Per-head L2Norm; Q or K `[128,16,T]` | Apply `L2` independently to each 128-element head. Each of 30 GDN layers calls it for Q and K. | **Supported for exactly this domain.** The BF16x2 warp-row route covers T=1..6 and T=1024 within 5.6% of its same-payload control. See the [retained report](archive/optimization-era/bench/qwen3.6-35b-norm-roofline.md). |
| N4 | Gated plain RMSNorm; BF16 O/Z `[128,32,T]`, weight `[128]` -> `[128,32,T]` | `out[:,j,t]=PlainRMS(O[:,j,t],w).*SiLU(Z[:,j,t])`. | **Supported for exactly this domain.** The compile-time gated epilogue uses the same warp-row reduction geometry and covers every T route within 4.8% of its same-payload control. See the [retained report](archive/optimization-era/bench/qwen3.6-35b-norm-roofline.md). |
| N5 | Affine LayerNorm; BF16 x `[1152,4096]`, gamma/beta `[1152]` | Per patch, `mu=mean_d x`, `var=mean_d(x-mu)^2`, `y=gamma.*(x-mu)/sqrt(var+eps)+beta`. | **Supported for exactly P=4096.** The fixed-D warp kernel reaches 96.9% of the nominal traffic roofline in the current qualification. See the [retained report](archive/optimization-era/bench/vision-layer-norm-roofline.md). |
| N6 | Same affine LayerNorm, `[1152,P]` with `P!=4096` | Same equation. It is called 54 times in Vision blocks and once before merger for every media item. | **Supported for the complete target domain.** The same fixed-D warp route covers every valid patch count through maximum video/image `P=49152/65536`; small cases reach their same-grid launch floor and both maxima reach at least 97.9% of the measured same-payload memory ceiling. See the [retained report](archive/optimization-era/bench/vision-layer-norm-roofline.md). |

## 5. Pointwise, broadcast, and activation operations

| ID | Operation and exact typed domain | Mathematical semantics and call sites | Current support |
|---|---|---|---|
| E1 | Broadcast bias add; BF16 x `[3456,4096]`, bias `[3456]` | `x[d,p]'=x[d,p]+bias[d]`; Vision QKV. | **Supported for exactly P=4096.** The aligned BF16x8 cache-sized route preserves the broadcast dimension and reduces the retained median from 36.5 to 15.6 us. See the [complete Section 5 report](archive/optimization-era/bench/qwen3.6-35b-pointwise-roofline.md). |
| E2 | Same bias add, `[3456,P]` with `P!=4096` | Same equation; Vision QKV. | **Supported for the complete target domain.** Finite dispatch uses the BF16x8 route through the cache-sized regime and a BF16x2 stream for larger media; small cases reach their fixed-work control and maximum video/image sizes reach the same-payload memory ceiling. See the [retained report](archive/optimization-era/bench/qwen3.6-35b-pointwise-roofline.md). |
| E3 | Broadcast bias add for `(D,C)=(1152,P),(4304,P),(4608,V),(2048,V)` | `x[d,c]'=x[d,c]+bias[d]`; used by patch/output/fc2, fc1, merger fc1, and merger fc2 respectively. | **Supported for the complete target domain.** The same D-aware BF16x8/BF16x2 dispatch covers every registered D and complete P/V bounds, including maximum image/video items, with retained exact and payload-control evidence. See the [retained report](archive/optimization-era/bench/qwen3.6-35b-pointwise-roofline.md). |
| E4 | Residual add; two BF16 `[1152,P]` tensors | `x'=x+y`; 27 Vision attention and 27 Vision MLP residuals. Text mixer residuals are L4; sparse-MoE residuals are part of the closed Op in Section 9. | **Supported for the complete target domain.** One aligned BF16x8 stream covers all valid P values; maximum video/image cases reach about 1.54 TB/s and remain within 0.3% of the same-topology control. See the [retained report](archive/optimization-era/bench/qwen3.6-35b-pointwise-roofline.md). |
| E5 | Sigmoid multiply; gate/x `[256,16,T]` | `x'=sigmoid(gate).*x`; full-attention output gate in ten Text layers and MTP. | **Supported for the complete target domain.** The aligned BF16x8 route covers `T=1..6/1024`; Small-T lies at the fixed launch floor and T=1024 remains within 4.4% of its same-payload control. See the [retained report](archive/optimization-era/bench/qwen3.6-35b-pointwise-roofline.md). |
| E6 | Tanh GELU; BF16 x `[4304,4096]` | `x'=0.5*x.*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`; Vision block fc1 activation. | **Supported for exactly P=4096.** The cache-sized BF16x8 route reduces the retained median from 44.0 to 19.1 us without changing FP32 operation order or BF16 rounding. See the [complete Section 5 report](archive/optimization-era/bench/qwen3.6-35b-pointwise-roofline.md). |
| E7 | Same tanh GELU, `[4304,P]` with `P!=4096` | Same equation. | **Supported for the complete target domain.** Cache-sized media use BF16x8 packs and larger media use the higher-throughput BF16x2 stream; maximum video/image cases track the same-payload control and sustain about 1.55 TB/s. See the [retained report](archive/optimization-era/bench/qwen3.6-35b-pointwise-roofline.md). |
| E8 | Exact GELU; BF16 x `[4608,V]` | `x'=0.5*x.*(1+erf(x/sqrt(2)))`; Vision merger. | **Supported for the complete target domain.** The same finite dispatch retains the exact-erf FP32 formula and covers `V=P/4` through maximum video/image `V=12288/16384`, tracking the corresponding payload control. See the [retained report](archive/optimization-era/bench/qwen3.6-35b-pointwise-roofline.md). |

SwiGLU is already part of LinearSwiGLU rows L16 and L18. The convolution's SiLU and the GDN Z
gate are parts of the closed operations in Sections 8 and 4, so they are not duplicate standalone
calls here.

## 6. Positional transforms

| ID | Operation and exact typed domain | Mathematical semantics | Current support |
|---|---|---|---|
| R1 | Partial interleaved Text MRoPE; BF16 Q `[256,16,T]`, K `[256,2,T]`, I32 positions `[T]` or NInfer Tensor `[T,3]`; rotary dim 64, theta `1e7` | For pair `j<32`, rotate dimensions `(j,j+32)` by `phi=position*theta^(-2j/64)`. The three-axis form is indexed mathematically as `position[axis,t]`, with pair j using axis `j mod 3` and temporal/height/width counts `[11,11,10]`; dimensions 64:256 are unchanged. Tensor `[T,3]` has `ne[0]=T` and represents the architecture's axis-major mathematical array `[3,T]`. | **Supported for exactly this domain.** Fixed 16Q/2KV paired and required single-Q/single-K routes cover one-axis and three-axis `T=1..6/1024` at their same-grid fixed-work ceilings. See the [retained RTX 5090 report](archive/optimization-era/bench/qwen3.6-35b-rope-roofline.md). |
| R2 | Vision 2-D RoPE; BF16 Q/K `[72,16,P]`, I32 positions `[P,2]`; theta `10000` | Pair `j<36` rotates `(j,j+36)`. Axis is height for `j<18` and width otherwise; local frequency is `theta^(-2*(j mod 18)/36)`. | **Supported for the complete target domain.** The packed-stride BF16x2 route covers every accepted patch count through maximum video/image `P=49152/65536`; the measured matrix remains within 1.6% of its same-grid control. See the [retained report](archive/optimization-era/bench/qwen3.6-35b-rope-roofline.md). |

For either row, the pair transform is
`(a,b)'=(a*cos(phi)-b*sin(phi), b*cos(phi)+a*sin(phi))`.

## 7. Attention operations

### 7.1 Causal grouped-query attention

For query head h and input column t, the 35B mapping is `kvh=floor(h/8)`. Let
`p=cache_positions[t]` be the absolute KV slot for that column. The softmax and value reduction
cover exactly `0<=s<=p`:

```text
score[h,t,s] = dot(Q[:,h,t], Kcache[:,s,kvh]) / sqrt(256)
O[:,h,t]     = sum_{s=0..p} softmax_{u=0..p}(score[h,t,u])[s] * Vcache[:,s,kvh]
```

The logical cache for one layer is K/V `[256,C,2]`. With `C_pad>=C` denoting the allocated padded
capacity, its registered execution representations are BF16 K and V `[256,C_pad,2]`, or separate
K/V I8 code planes `[256,C_pad,2]` with FP16 scale planes `[4,C_pad,2]`.

| ID | Operation and exact typed domain | State effect | Current support |
|---|---|---|---|
| A1 | Append-and-attend GQA; BF16 Q `[256,16,T]`, new K/V `[256,2,T]`, I32 `cache_positions[T]`, visible `L<=262144`, BF16 output `[256,16,T]`; BF16 and INT8-G64 cache routes | Writes new K/V at the supplied absolute slots, makes each column's own write visible to its causal domain `0:s<=cache_positions[t]`, and returns O. It does not publish a Program cache frontier. | **Supported for exactly this domain.** Compile-time 16Q/2KV group-8 instances cover split-KV `T=1..6` and prompt `T<=1024` for both cache formats. At `L=261120`, T=1/T=6 reach 97.8%/93.7% of the measured BF16 copy roofline and 88.4%/80.8% for INT8; 1024-token prefill sustains 152.1/155.8 useful TFLOP/s. See the [retained RTX 5090 report](archive/optimization-era/bench/qwen3.6-35b-gqa-a1-roofline.md). |
| A2 | KV append only; BF16 K/V `[256,2,T]`, I32 `cache_positions[T]`, `1<=T<=1024`, both cache routes | Overwrites every addressed K/V value and scale without computing attention or publishing a Program cache frontier. | **Supported for exactly this domain.** The BF16 vector-copy and INT8-G64 warp-per-group quantizing kernels cover the complete target chunk range. At T=1024, NCU measures 4.51/4.77 us, respectively 92.2%/89.3% of same-grid, same-payload fixed-resource controls; T=1..6 sits at the same batched launch-interval floor as those controls. See the [retained RTX 5090 report](archive/optimization-era/bench/qwen3.6-35b-gqa-a2-roofline.md). |
| A3 | Cached-only GQA; BF16 Q/O `[256,16,T]`, I32 `cache_positions[T]`, populated BF16 or INT8-G64 cache, `1<=T<=6`, `L<=262144` | Applies the equation above without accepting new K/V or mutating any cache code/scale plane. Device positions define the causal domain; an explicit host envelope only bounds launch selection. | **Supported for exactly this domain.** A compile-time cached-only effect reuses the split-KV Q staging, cache decode, QK/PV, online-softmax, and reducer while removing all new-K/V operands, quantization, and writes. At `L=261120`, T=1/T=6 reaches 98.4%/94.8% of the measured BF16 copy roofline and 88.1%/80.9% for INT8. See the [retained RTX 5090 report](archive/optimization-era/bench/qwen3.6-35b-gqa-a3-roofline.md). |

All three entries are required: ordinary Text uses A1, while MTP alignment and rebuild also require
the standalone A2 and A3 state-effect contracts. `cache_positions` contains sequential absolute
cache slots; it is distinct from R1's scalar or three-axis RoPE coordinates and is never derived
from multimodal coordinate values.

### 7.2 Segmented Vision attention

Within each segment and head:

```text
score[i,j] = dot(Q[:,h,i], K[:,h,j]) / sqrt(72)
O[:,h,i]  = sum_j softmax(score[i,:])[j] * V[:,h,j]
```

No index attends across segment boundaries. On the 35B item route, the boundaries are the `S`
equal consecutive ranges of length `g_h*g_w`; they are derived directly from that length and do
not require a descriptor tensor or setup launch.

| ID | Exact typed domain | Current support |
|---|---|---|
| A4 | BF16 Q/K/V/O `[72,16,P]`; `S=g_t` equal segments of length `g_h*g_w`, with the complete image/video P bounds in Section 1. The required Q/K/V route is a view of packed QKV `[3456,P]`; O is contiguous. | **Supported for the complete domain.** A direct no-descriptor route dispatches measured 16/32/64-row FlashAttention instances. Same-grid payload controls qualify short and tail-heavy segments, while maximum-size image/video cases sustain 176.9/176.7 issued-MMA TFLOP/s and the maximum-video NCU capture reaches 80.66% Compute SOL. See the [retained RTX 5090 report](archive/optimization-era/bench/vision-attention-roofline.md). |

## 8. Stateful convolution and recurrence

| ID | Operation and exact typed domain | Mathematical semantics and state effect | Current support |
|---|---|---|---|
| S1 | Depthwise causal width-4 convolution plus SiLU; BF16 input/output `[8192,T]`, weight `[8192,4]`, history `[8192,3]` | With history supplying negative chunk indices, `out[c,t]=SiLU(sum_{r=0}^3 W[c,r]*x[c,t-3+r])`; new history is the final three columns of old-history concatenated with x. Ordinary in-place and distinct-state forms are required. | **Supported for exactly this domain.** T=1 and `2<=T<=6` use qualified single-launch decode and channel/token kernels for both state forms; T=1024 retains the balanced BF16x2 prefill route. See the [retained RTX 5090 report](archive/optimization-era/bench/qwen3.6-35b-causal-conv-roofline.md). |
| S2 | Snapshot form of S1 for `1<=T<=6`; BF16 state slots `[8192,3,7]`, I32 `initial_slot` in `[0,7)` | Reads the initial history from `initial_slot`; after each input column t, publishes the resulting history in slot t while leaving later slots unchanged. | **Supported for exactly this domain.** T=1 uses a loop-free snapshot decode kernel and `2<=T<=6` publishes all snapshots from one channel/token launch; both meet the exact fixed-work roofline in the [retained report](archive/optimization-era/bench/qwen3.6-35b-causal-conv-roofline.md). |
| S3 | Gated delta-rule recurrence; BF16 Q/K `[128,16,T]`, V/O `[128,32,T]`, FP32 g/beta `[32,T]`, FP32 state `[128_v,128_k,32_heads]` | For V head j, `S_j` has axes `[Vdim,Kdim]` and uses Q/K head `floor(j/2)`: `Sbar=exp(g[j,t])*S_j`; `delta=beta[j,t]*(v-Sbar*k)`; `S_j'=Sbar+delta*k^T`; `o=(S_j'*q)/sqrt(128)`. Tokens update state in order. Ordinary in-place and distinct-state forms are required. | **Supported for exactly this domain.** The native recurrent and 64-token chunked paths implement the 16-to-32 grouped head map and preserve the same state transition. |
| S4 | Snapshot form of S3 for `1<=T<=6`; FP32 state slots `[128_v,128_k,32_heads,7]`, I32 `initial_slot` in `[0,7)` | Reads the initial recurrent state from `initial_slot`; after column t, publishes the resulting state in slot t and leaves later slots unchanged. | **Supported for exactly this domain.** The native recurrent snapshot path implements the same 32-head geometry and grouped head map. |

## 9. Sparse routing, grouped expert work, and reduction

`SparseMoeAdd` is the target-callable, semantically closed residual-updating sparse Op. For
normalized input `X[2048,T]` and decoder residual `R[2048,T]`, it owns the router, top-8
selection, routed experts, always-on shared expert, final merge, and decoder residual update:

```text
scores = W_router_shared X                  # L15, [257,T]
p[e,t] = exp(scores[e,t]) / sum_j exp(scores[j,t])
I_t = indices of the eight largest p[:,t]
alpha[e,t] = p[e,t] / sum_{j in I_t} p[j,t]

Expert_e(x) = W_down[e] * SwiGLU(W_gate_up[e] * x)
Shared(x)   = W_shared_down * SwiGLU(W_shared_gate_up * x)

Y[:,t]  = sum_{e in I_t} alpha[e,t] * Expert_e(X[:,t])
          + sigmoid(scores[256,t]) * Shared(X[:,t])
R[:,t]' = R[:,t] + Y[:,t]
```

At an exact top-8 boundary tie, lower expert id wins. There is no capacity limit and no token drop:
all `8T` assignments enter the grouped contractions and reduction. The operation-class inventory
beneath the closed boundary is:

| ID | Required operation/stage | Exact domain and semantics | Current support |
|---|---|---|---|
| M1 | Router softmax, top-8, selected-weight renormalization, and shared sigmoid | Promote L15's BF16 router scores `[256,T]` and shared score `[T]` to FP32; produce I32 ids `[8,T]`, BF16 route weights `[8,T]`, and FP32 shared scale `[T]`. Select the eight largest softmax probabilities per token using the stated tie rule and renormalize only those eight; shared scale is the sigmoid of the independent score. | **New implementation.** No device router/top-8 path exists. |
| M2 | Assignment grouping and inverse-map construction | Form assignments `(expert,token,route_slot,weight)` for `A=8T`; partition them into expert jobs of size `M_e` while preserving a bijection back to token/route order. | **New implementation.** No device grouping path or job representation exists. |
| M3 | Inverse scatter, route-weighted reduction, shared merge, and residual update | Restore token order, compute `Y_routed[:,t]=sum_r alpha[r,t]*Y_r[:,t]`, then update `R[:,t]'=R[:,t]+Y_routed[:,t]+shared_scale[t]*Y_shared[:,t]`. | **New implementation.** No routed reduction/merge/residual path exists. |
| M4 | Closed SparseMoeAdd Op | Compose private Linear stages L15-L19 and sparse stages M1-M3 for decode T=1, Small-T `T<=6`, and prefill `T<=1024`, with the residual update and all workspace effects owned by one Op call. | **New implementation.** NInfer has no `SparseMoeAdd` contract or high-performance route today. |

L18 and L19 define the two exact grouped-contraction kernel domains and are not repeated as sparse
stage IDs here. Generic Linear calls, eight selected-expert launches, one launch per active prefill
expert, an all-expert scan, or materialized selected weight copies support neither those domains
nor M2-M4.

## 10. Token decision and MTP control operations

| ID | Operation and exact typed domain | Mathematical semantics | Current support |
|---|---|---|---|
| G1 | Argmax; BF16 full logits `[248320,C]` with `1<=C<=6` and 248077 valid rows, or shortlist `[131072,1]`; output I32 ids `[C]` or `[1]` | `id[c]=min argmax_{0<=r<R} logits[r,c]`, where R is the valid-row count. | **Supported for exactly this domain.** The 512-thread tiled route covers every full-vocabulary column count and the shortlist at its same-grid fixed-work control. See the [retained RTX 5090 report](archive/optimization-era/bench/qwen3.6-35b-token-decision-roofline.md). |
| G2 | Token sampling; BF16 logits `[248320,1]`, valid rows `R=248077`, optional I32 occurrence counts `[R]`, and one explicit random key | Let `a_v=logits_v-presence*1[count_v>0]-frequency*count_v`. Sort by descending a, breaking ties by lower token id. Keep configured top-k when it is 1..19, otherwise keep `min(20,R)`. Set `w_v=exp(a_v/temperature-max a)`, remove the suffix below `min_p*max(w)` when min-p is enabled, then keep the shortest remaining prefix whose cumulative weight reaches top-p times the sum of all top-k weights before min-p/top-p truncation. Retain at least the best token, normalize, and draw from `(seed,position,purpose)`. A stochastic draw increments its occurrence count; nonpositive temperature reduces to G1 and does not update counts. | **Supported for exactly this domain.** Greedy, stochastic, and optional-count routes use one graph-stable partial/group implementation with caller-owned workspace and retained exact/probabilistic evidence. See the [retained report](archive/optimization-era/bench/qwen3.6-35b-token-decision-roofline.md). |
| G3 | Accept MTP verification tokens; I32 drafts `[K]`, target ids `[K+1]`, BF16 target logits `[248320,K+1]` with 248077 valid rows, I32 output `[K+1]`, I32 length/token/num-sampled/accepted/ar-position scalars, I64 statistics `[>=9]`, `1<=K<=5` | Greedy mode accepts the longest prefix equal to target argmax and emits the first mismatch or bonus. Sampling distribution i uses G2 with `count_i(v)=committed_count(v)+sum_{j<i}1[draft_j=v]`; accept draft `d_i` with probability `p_i(d_i)`, otherwise sample its normalized distribution with `d_i` removed. Accept and resample RNG keys use logical position `old_length+i+1` with their distinct purposes; the all-accepted bonus uses `old_length+K+1` and the bonus purpose. If A drafts are accepted, output slots `0:A-1` are those drafts, slot A is correction/bonus, and later slots are zero; publish `num_sampled=A+1`, `accepted=A`, `token=output[A]`, `length'=old_length+A+1`, and `ar_pos'=length'`. Apply `stats[0]+=K`, `stats[1]+=A`, `stats[2]+=1`, and `stats[4+i]+=1` for `0<=i<min(A,5)`; stochastic produced tokens increment committed occurrence counts after the decision. | **Supported for exactly this domain.** Exact greedy/stochastic state transitions, round-local penalty overlays, and the complete `K=1..5` matrix use the shared caller-workspace route with retained RTX 5090 evidence. See the [retained report](archive/optimization-era/bench/qwen3.6-35b-token-decision-roofline.md). |

Target verification itself is not another Op: it is a main Text invocation with `T=K+1<=6`
followed by G3. MTP cache selection, state commit/rollback, rebuilding call order, and publication of
generated tokens are Program/runtime policy rather than device Ops.

## 11. Coverage and status summary

This call-site index cross-checks device-Op coverage without using model modules as the
classification:

| Graph region | Operation IDs used |
|---|---|
| Text root and output policy | I2, I4/I5, N1, L6, G1/G2 |
| Full Attention, including MTP's layer | L1, N2, R1, A1-A3, E5, L4 |
| Gated-DeltaNet | L2, L3, S1-S4, N3, N4, L4 |
| Sparse expert layer, including MTP's layer | M4; private implementation domains L15-L19 and M1-M3 |
| Vision stem and 27 blocks | I1, L8, E3, I6, N5/N6, L9, E1/E2, R2, A4, L10-L12, E4, E6/E7 |
| Vision merger and Text insertion | N5/N6, L13/L14, E3, E8, I3 |
| MTP stem/proposal/control | I2, N1, I7, L5, the Full Attention and Sparse expert rows above, L6/L7, I8-I12, G1-G3 |

This is a device-operation cross-check, not a second graph authority. MTP shifted-token/hidden
alignment and accepted-prefix state selection remain target schedule semantics in the architecture
document. Vision position/interpolation metadata and segment bounds remain frontend inputs; their
construction is not an additional device Op.

Under the strict support definition, the complete result is:

| Status | Exact result |
|---|---|
| **Supported** | L7 complete Q4 draft head `[131072,2048]`; L8 complete Q6 Vision patch projection `[1152,1536]`; L9 complete Q4 Vision QKV projection `[3456,1152]`; L10 complete Q5 Vision attention output `[1152,1152]`; L11 complete Q4 Vision MLP fc1 `[4304,1152]`; L12 complete Q5 Vision MLP fc2 `[1152,4304]`; L13 complete W8 Vision merger fc1 `[4608,4608]`; L14 complete W8 Vision merger fc2 `[2048,4608]`; I1-I12 complete indexing/control domains; N1-N4 complete Text/GDN/MTP normalization domains; N5/N6 complete Vision LayerNorm `[1152,P]`; E1-E8 complete pointwise/broadcast/activation domains; R1 Text RoPE/MRoPE `[256,16|2,T]`; R2 complete packed Vision RoPE `[72,16,P]`; A1 append-and-attend GQA `[256,16|2,T]`; A2 standalone KV append `[256,2,T]`; A3 cached-only GQA `[256,16,T]`; A4 complete segmented Vision attention `[72,16,P]`; S1/S2 GDN causal convolution and snapshot state; S3/S4 GDN recurrence and snapshot state transition `[128,16,32]`; and G1-G3 complete token-decision/MTP-accept domains. |
| **Adapt existing** | Target-callable dense/quantized Linear domains L1-L6 plus private L15-L17. All remain unsupported. |
| **New implementation** | L18-L19 expert-grouped contractions and M1-M4 sparse routing/grouping/reduction/closed `SparseMoeAdd` execution. |

No complete Text/GDN block, sparse-MoE execution, generation path, or MTP execution currently meets
the support definition for the 35B target.
