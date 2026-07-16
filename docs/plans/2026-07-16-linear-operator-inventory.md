# Current Linear-family Operator Inventory

> Status: current-source inventory and per-operator work queue. No migration or production
> implementation has started.

This document inventories the Linear-family implementation that exists on the current product
baseline. It is deliberately separate from the proposed architecture in
[`2026-07-16-linear-kernel-architecture-refactor.md`](2026-07-16-linear-kernel-architecture-refactor.md).
The architecture document says what the ownership and dispatch boundary should become; this
document records what exists today and what must be reviewed before deciding whether to retain,
rewrite, broaden, generalize, or delete it.

The inventory is organized at four different levels:

1. a **semantic Op** is a target-callable mathematical contract such as `linear_add`;
2. a **physical problem** is an exact weight format, logical `N/K`, padded `K`, token domain, and
   output topology;
3. a **policy** is one complete current execution choice;
4. a **kernel specialization** is one CUDA implementation or compile-time configuration used by a
   policy.

These levels must not be conflated. A wrapper that calls `linear()` twice is one semantic Op but
not one kernel. Conversely, one current `LinearPolicyId` may still choose several kernel
specializations inside its launcher.

## 1. Inventory summary

The current source contains:

- seven target-callable Linear-family semantic Ops;
- six weight formats accepted by base `linear()`:
  contiguous BF16/FP32 and RowSplit Q4/Q5/Q6/W8;
- 14 base `LinearPolicyId` values;
- 12 `ShapeFamily` values, including `Generic`, and three global `LinearRegime` values;
- 24 product-reachable base physical problems when contractions used inside fused or grouped Ops
  are included;
- two exact `LinearAdd` problems, one `LinearSwiGLU` problem, one reachable `LinearPair` problem,
  two grouped-input problems, and one dense GDN-control problem;
- additional compiled tuned kernels for dead target branches;
- broad arbitrary-shape quantized and dense execution used mainly by tests and measurement.

The registered 27B Text topology contains 16 Full Attention layers, 48 GDN layers, and 64 MLPs.
One complete `run_layers()` performs 352 Linear-family semantic calls representing 496 mathematical
`W @ X` contractions:

| Region | Semantic calls | Mathematical contractions |
|---|---:|---:|
| 16 Full Attention mixers | 32 | 80 |
| 48 GDN mixers | 192 | 288 |
| 64 MLPs | 128 | 128 |
| **Total** | **352** | **496** |

The final target head adds one semantic call and one contraction. Vision contributes 111 ordinary
`linear()` calls per media encode. MTP has a smaller but route-sensitive mixture of base Linear,
LinearPair, and head calls.

For a Text verification window:

- at `T=1`, the current wrappers enter public base `linear()` 209 times;
- at `T=2..6`, materialized `LinearAdd` and `LinearSwiGLU` routes raise that to 401 calls.

This makes T1 and Small-T dispatch/launch cost a first-order product concern, not merely a
microkernel detail.

## 2. Current semantic Ops

| ID | Semantic Op | Exact public contract | Current execution forms | Product status |
|---|---|---|---|---|
| S1 | `linear` | One BF16 `X[K,T]`, one dense or RowSplit `W[N,K]`, BF16 `Y[N,T]` | Dense reference GEMV/GEMM; generic low-bit Small-T; low-bit MMA; W8 MMA; exact T1 GEMVs | Reachable throughout Text, MTP, and Vision |
| S2 | `linear_pair` | Two same-shaped Q5 or W8 projections of one input | Exact Q5 dual GEMV at T1; W8 dual MMA for `T>16`; otherwise two calls to public `linear()` | W8 `[1024,5120] x2` is reachable; the Q5 dual route is not |
| S3 | `attn_input_proj` | Four fixed Q4/Q5 projections from `[5120,T]` to Q/gate/K/V outputs | Four public `linear()` calls for `T<=16`; two grouped MMA launches for `T>16` | Reachable in 16 Full Attention layers |
| S4 | `gdn_input_proj` | Fixed Q4 Q/K plus Q5 V projections into one concatenated output | Two public `linear()` calls plus two D2D copies for `T<=16`; one mixed grouped MMA launch for `T>16` | Reachable in 48 GDN layers |
| S5 | `linear_add` | Fixed Q5 projection followed by in-place BF16 residual add | Exact fused T1; materialized Linear plus residual for `T=2..16`; residual MMA for `T>16` | Two reachable problems, called 128 times per Text layer pass |
| S6 | `linear_swiglu` | Fixed Q4 gate/up projection followed by SwiGLU | Exact fused T1; materialized Linear plus `silu_mul` for `T=2..16`; folded MMA for `T>16` | One reachable problem, called 64 times per Text layer pass |
| S7 | `gdn_gating_proj` | Two dense BF16 `[48,5120]` projections plus FP32 GDN gate preparation | Exact T1; Small-T partial/reduce; cooperative or direct dense MMA for larger T | Reachable in 48 GDN layers |

The source-level ownership is currently split across:

- `src/ops/linear/linear.cpp` for S1 and S2;
- `src/ops/wrapper/input_proj.cpp` for S3 and S4;
- `src/ops/wrapper/linear_add.cpp` for S5;
- `src/ops/wrapper/linear_swiglu.cpp` for S6;
- `src/ops/wrapper/gdn_gating_proj.cpp` for S7.

## 3. Base `linear()` dispatch as implemented

The current planner key is:

```text
(LinearFormat, ShapeFamily, LinearRegime)
```

`LinearRegime` is globally classified as:

```text
T <= 1       -> T1
2 <= T <=16 -> SmallT
T > 16       -> LargeT
```

This coarse plan is not the final execution choice. Several launchers inspect shape or T again.

### 3.1 Dense BF16 and FP32

| T/domain | Current policy | Launcher-side selection |
|---|---|---|
| `T=1`, arbitrary valid `N/K` | `GenericDenseGemv` | One 256-thread reduction block per output row |
| `T>1`, `N<=128 && T<=16` | `GenericDenseGemm` | One GEMV-like block per `(row,column)` |
| Other `T>1` | `GenericDenseGemm` | Scalar 16x16 output tile; each thread loops over all K |

These kernels are correctness/reference implementations. They are not tensor-core production
GEMMs and do not establish optimized arbitrary dense support.

### 3.2 Q4/Q5/Q6 RowSplit

| Domain | Current policy | Important internal selection |
|---|---|---|
| Registered exact T1 shapes | One of nine tuned policy IDs | Dedicated shape kernel or a fixed Q5-core instantiation |
| Other T1 shapes | `RowsplitLowbitGemmSmallt` | Generic TT4 Small-T kernel |
| `T=2..16` | `RowsplitLowbitGemmSmallt` | Q5 exact direct kernels for selected shapes/T2..6; otherwise TT4 or TT8 |
| `T>16`, `K%8==0` | `RowsplitLowbitGemmMma` | Shape-dependent BN64 or default BN128 config, then full/edge specialization |
| `T>16`, `K%8!=0` | `RowsplitLowbitGemmSmallt` | Generic TT8 fallback re-streams weights by token tile |

The current dedicated T1 policy IDs are:

- Q4 `[34816,5120]`;
- Q4 and Q5 parent `[7168,5120]`;
- Q4 `[4096,5120]`;
- Q5 `[5120,17408]`;
- Q6 and Q4 vocab heads with fixed `K=5120` and runtime N;
- Q5 `[6144,5120]`;
- Q5 `[5120,6144]`.

The Q5 Small-T launcher adds another exact dispatch for `T=2..6`:

| Exact shape | Kernel form |
|---|---|
| `[7168,5120]` | direct split-2 |
| `[6144,5120]` | direct split-4 |
| `[5120,6144]` | direct split-2 |
| `[5120,17408]` | direct split-2 |

The low-bit MMA launcher then selects:

- BN64 for Q4 `[4096,5120]` through T128;
- BN64 for Q5 `[5120,17408]` and `[5120,6144]` through T128;
- BN64 for Q5 `[6144,5120]` through T64;
- the default BN128 configuration otherwise;
- full-tile or edge generated code after that selection.

### 3.3 W8G32 RowSplit

| Domain | Current policy | Launcher-side selection |
|---|---|---|
| `T<=16` | `RowsplitLowbitGemmSmallt` | TT4 or TT8 generic W8 Small-T |
| `T>16`, `K%8==0`, `Kpad%256==0` | `RowsplitW8G32GemmMma` | `N<=1024` small-M config or general config, then full/edge |
| Other T | `RowsplitLowbitGemmSmallt` | Generic Small-T fallback |

The plan therefore does not completely determine the kernel. Geometry and edge-state decisions
remain inside launchers.

## 4. Existing kernel families

This section groups implementations by algorithmic family rather than counting every generated
template instantiation.

### 4.1 Generic or dimension-driven kernels

| Family | Source | Actual support surface | Generalization value |
|---|---|---|---|
| Dense reference GEMV/GEMM | `linear_generic_dense.cu/.cuh` | BF16/FP32, arbitrary valid N/K/T | Correctness oracle and fallback candidate only |
| RowSplit Small-T | `linear_rowsplit_gemm_smallt.cu/.cuh` | Q4/Q5/Q6/W8, arbitrary N/K, TT4/TT8, scalar K tail | Main reusable SIMT substrate |
| Q5 direct Small-T | Same files | Four exact shapes, T2..6, split-2/split-4 | Exact policy candidates, not a general regime |
| Low-bit MMA | `linear_rowsplit_gemm_mma.cu/.cuh` | Q4/Q5/Q6, runtime N/K/T with BN64/BN128 and full/edge variants | Main reusable tensor-core substrate |
| W8 MMA | `linear_rowsplit_w8g32_gemm_mma.cu/.cuh` | W8, runtime N/K/T, small-M/general and full/edge variants | Separate W8 lifecycle |
| Runtime-N head GEMV | `linear_rowsplit_gemv_lm_head.cu` | Q4 or Q6, fixed K5120, runtime N | Narrow but already generalized over vocabulary rows |

### 4.2 Exact T1 or topology-specific kernels

| Family | Exact domain | Notes |
|---|---|---|
| Q4 MLP gate/up GEMV | `[34816,5120]` | Separate plain and fused-SwiGLU kernels |
| Q4 attention parent GEMV | `[7168,5120]` | Mixed 6144-row and 1024-row geometry; current product branch is dead |
| Q4 GDN Q/K GEMV | `[4096,5120]` | Eight-way split K per row |
| Parameterized Q5 core | Several fixed N/K configurations | Plain, residual, and dual-output forms share one templated substrate |
| Q5 GDN V/Z dual GEMV | two `[6144,5120]` weights | Current product branch is dead |
| Dense GDN gating T1 | two BF16 `[48,5120]` weights | Fuses both projections and FP32 gate transforms |

### 4.3 Multi-output and fused tensor-core kernels

| Family | Semantic consumer | Topology |
|---|---|---|
| W8 paired K/V MMA | `linear_pair` | One input, two same-shaped W8 weights, two outputs |
| Grouped Q4/Q5 input MMA | `attn_input_proj`, `gdn_input_proj` | Two or four jobs, separate or concatenated outputs |
| Q4 folded gate/up SwiGLU MMA | `linear_swiglu` | Paired gate/up rows and SwiGLU final store |
| Q5 residual low-bit MMA | `linear_add` | Base low-bit MMA mainloop with residual final store |
| Dense GDN cooperative MMA | `gdn_gating_proj` | Two dense projections, split-K workspace when required, FP32 outputs |

### 4.4 Reuse seams already visible in source

The current code already demonstrates useful narrow reuse:

- Q4/Q5/Q6/W8 decode codecs;
- one generic RowSplit Small-T topology;
- one Q4/Q5/Q6 MMA topology;
- a parameterized Q5 T1 core with plain, residual, and dual-output forms;
- grouped job descriptors for two/four projection jobs;
- full-tile versus edge specializations;
- residual and SwiGLU final-store variants.

It also demonstrates where generalization is currently unsafe:

- W8 has a different group size and tensor-core lifecycle;
- grouped input, paired output, residual, and SwiGLU have different launch economics;
- several exact T1 kernels use different warp ownership and K splitting;
- the dense GDN-control Op has FP32 outputs and a semantic BF16 projection rounding seam.

These are review inputs, not a conclusion that one universal kernel template should replace them.

## 5. Product-reachable base physical problems

The 24 rows below are the current base contraction inventory. Problems used only inside a fused or
grouped semantic Op remain listed because a materialized policy may execute the same physical
contraction. The dense GDN-control problem is listed separately in Section 6 because it is not a
base `linear()` production call.

### 5.1 Text and Text fused-inner problems

| ID | Format and exact `W[N,K,Kpad]` | Reachable semantic use | Multiplicity |
|---|---|---|---:|
| B01 | Q4 `[6144,5120,5120]` | Full Attention Q view in `attn_input_proj` | 16 |
| B02 | Q5 `[6144,5120,5120]` | Full gate, GDN V, and plain GDN Z | 112 weight slots |
| B03 | Q4 `[1024,5120,5120]` | Full Attention K view | 16 |
| B04 | Q5 `[1024,5120,5120]` | Full Attention V view | 16 |
| B05 | Q4 `[4096,5120,5120]` | GDN concatenated Q/K projection | 48 |
| B06 | Q5 `[5120,6144,6144]` | Full/GDN output `linear_add` | 64 |
| B07 | Q4 `[34816,5120,5120]` | All Text `linear_swiglu` gate/up projections | 64 |
| B08 | Q5 `[5120,17408,17408]` | All Text MLP-down `linear_add` projections | 64 |
| B09 | Q6 `[248320,5120,5120]` | Full target head | 1 |

Reachable T is phase-dependent:

- ordinary verification is T1;
- MTP target verification reaches `T=2..6`;
- one Text prefill chunk is at most 1024 columns and may have any positive tail length;
- grouped and fused wrappers decide whether the base contraction is materialized.

### 5.2 MTP problems

| ID | Format and exact `W[N,K,Kpad]` | Reachable semantic use |
|---|---|---|
| B10 | W8 `[5120,10240,10240]` | MTP stem input projection |
| B11 | W8 `[14336,5120,5120]` | Combined MTP Q/K/gate/V parent projection |
| B12 | W8 `[1024,5120,5120]` | MTP prompt K/V `linear_pair`, two weights |
| B13 | W8 `[6144,5120,5120]` | Final prompt Q and gate row views |
| B14 | W8 `[5120,6144,6144]` | MTP attention output |
| B15 | W8 `[34816,5120,5120]` | MTP gate/up projection |
| B16 | W8 `[5120,17408,17408]` | MTP MLP down projection |
| B17 | Q4 `[131072,5120,5120]` | Optimized proposal head |

The parent W8 `[14336,5120]` problem and its `[6144]`/`[1024]` row views are all genuinely
reachable in different MTP schedules. A future exact catalog must model both physical parent and
effective view problems; registering only one representation would be incomplete.

### 5.3 Vision problems

Let P be raw patch columns and V be merged Vision columns. The product frontend bounds
`P<=131072`, `V<=32768`, and `P=4V`.

| ID | Format and exact `W[N,K,Kpad]` | Calls per media encode | T |
|---|---|---:|---|
| B18 | Q6 `[1152,1536,1536]` | 1 | P |
| B19 | Q4 `[3456,1152,1152]` | 27 | P |
| B20 | Q5 `[1152,1152,1152]` | 27 | P |
| B21 | Q4 `[4304,1152,1152]` | 27 | P |
| B22 | Q5 `[1152,4304,4352]` | 27 | P |
| B23 | W8 `[4608,4608,4608]` | 1 | V |
| B24 | W8 `[5120,4608,4608]` | 1 | V |

B22 is the only current registered problem whose logical K differs from padded K. Its padding and
edge behavior must be measured explicitly rather than inferred from aligned Text shapes.

## 6. Independent composite physical problems

| ID | Semantic Op | Exact physical domain | Current route surface |
|---|---|---|---|
| C01 | `linear_add` | Q5 B06 | fused T1; materialized T2..16; residual MMA T17+ |
| C02 | `linear_add` | Q5 B08 | fused T1; materialized T2..16; residual MMA T17+ |
| C03 | `linear_swiglu` | Q4 B07 | fused T1; materialized T2..16; folded MMA T17+ |
| C04 | `linear_pair` | two W8 B12 weights | two base calls through T16; paired MMA T17+ |
| C05 | `attn_input_proj` | B01+B02+B03+B04 | four base calls through T16; two grouped MMA launches T17+ |
| C06 | `gdn_input_proj` | B05+B02 | two base calls and copies through T16; one grouped MMA launch T17+ |
| C07 | `gdn_gating_proj` | two dense BF16 `[48,5120]` weights plus FP32 vectors/outputs | exact T1; two-kernel Small-T; split-K dense MMA for larger T |

Current workspace behavior is likewise Op-specific:

- base `linear()`, `linear_pair`, and `attn_input_proj` use no effective scratch today;
- `linear_add` materializes `[5120,T]` for `T=2..16`;
- `linear_swiglu` materializes `[34816,T]` for `T=2..16`;
- `gdn_input_proj` materializes `[4096,T]` and `[6144,T]` for `T<=16`;
- `gdn_gating_proj` uses split-K FP32 workspace for selected T ranges.

## 7. Compiled but product-unreachable execution

`Phase::Decode` has no caller in the current target. `run_layers()` is invoked only with
`Phase::Verify` or `Phase::Prefill`. The two `Phase::Decode` branches therefore do not establish
current product support.

The following tuned execution is compiled but unreachable from the product schedule:

- Q4 parent `[7168,5120]` T1 attention GEMV;
- Q5 parent `[7168,5120]` T1 attention GEMV;
- Q5 dual `[6144,5120]` V/Z `linear_pair` T1 GEMV.

Several views are bound but not passed to a Linear-family Op:

- GDN Q and K `[2048,5120]` child views;
- Text MLP gate and up `[17408,5120]` child views;
- MTP MLP gate and up `[17408,5120]` child views.

The packed parent weights remain real artifact storage. Removing a dead execution problem does not
imply changing the artifact layout or deleting the row views used for non-Linear tensor semantics.

Other source-visible policies are callable through the repository-internal API but not selected by
the current target:

- the plain T1 Q4 `[34816,5120]` kernel, because Text uses fused `linear_swiglu`;
- the plain T1 Q5 `[5120,17408]` and `[5120,6144]` kernels, because Text uses fused
  `linear_add`;
- the single-output W8 MMA small-M configuration: the only reachable W8 `N=1024` problem is B12,
  whose `T>16` route is intercepted by the paired W8 kernel;
- arbitrary dense and quantized shapes admitted through `Generic`.

These should be reviewed as reusable measurement/reference candidates, not silently treated as
registered product requirements.

## 8. Source inconsistencies exposed by the inventory

The current tree contains several stale descriptions that should not guide architecture decisions:

- `linear_rowsplit_gemm_smallt.cu/.cuh` says that no W8 MMA path exists, while the planner and
  dedicated W8 launcher route Large-T W8 to MMA;
- the base planner names one policy, but Small-T, low-bit MMA, W8 MMA, and dense launchers all make
  additional behavioral selections;
- fused wrappers independently repeat the same global `T=1/16` boundaries;
- current benchmark “all target” matrices do not equal the actual reachable physical inventory;
- the GDN gating benchmark's split-K model differs from the implementation.

These are not separate cleanup tasks to perform now. They are evidence that source comments,
planner names, and benchmark labels cannot substitute for the physical inventory above.

## 9. Current verification and measurement coverage

### 9.1 Retained source tests

| Area | Current evidence | Main gap |
|---|---|---|
| Base dense Linear | FP64 CPU oracle over BF16/FP32 weights, varied aligned/unaligned N/K/T | Performance implementation is intentionally reference-grade |
| Base quantized Linear | CPU dequantized oracle across Q4/Q5/Q6/W8, T1/Small-T/Large-T and tail shapes | Exact high-T Vision problems are not independently covered at full scale |
| `linear_pair` | W8 correctness on small generic shapes | Exact product B12 geometry and all route boundaries are not covered |
| Grouped input Ops | Large-T comparison against separate base Linear results | Comparison is not a fully independent mathematical oracle |
| `linear_add` | Large-T comparison against Linear plus residual at T17/128/129 | T1 and materialized Small-T semantic boundaries are not directly retained |
| `linear_swiglu` | Large-T comparison against Linear plus `silu_mul` at T17/128/129 | T1 and materialized Small-T semantic boundaries are not directly retained |
| `gdn_gating_proj` | Independent CPU oracle at T1, 2, 6, 7, 8, 128, 512, 1024, 2048, 4096, 4097; graph capture at T1024 | No retained NCU or same-campaign performance baseline |

### 9.2 Current benchmark binaries

| Binary | Useful coverage | Inventory problem |
|---|---|---|
| `ninfer_linear_bench` | Fast dense and selected decode signals | Shape labels include older child geometries and do not represent the complete registered domain |
| `ninfer_linear_op_bench` | Cold/warm base Linear and W8 pair timing with roofline probes | `--all-targets` contains 17 rows, includes dead parent/draft variants, and omits reachable row views and Vision |
| `ninfer_gdn_gating_proj_bench` | Dedicated dense fused timing | Its split-K model no longer matches the implementation's current thresholds |

The current benchmark source must therefore not be used as an automatic definition of “all
Linear operators.”

### 9.3 Historical experiment evidence

The retained experiment report and raw profiler files contain useful physical evidence for:

- the 24 base problems and prototype 83-route manifest;
- geometry-local Small-T/MMA crossovers;
- Q5 direct kernels;
- W8 single and paired routes;
- non-monotonic `LinearAdd` and `LinearSwiGLU` behavior;
- Vision high-T and partial-wave behavior;
- representative SASS, resource, NSYS, and NCU captures.

That evidence was produced while exploring an architecture whose target/profile ownership was
later rejected. The measurements remain candidate evidence, but the prototype route tables and
implementation must not be treated as current source behavior or copied without requalification.

The retained raw campaign currently contains more than one thousand benchmark files plus dozens of
NCU and NSYS reports. Quantity is not qualification: only matched, release-build, exact-domain
evidence should be carried into an operator decision.

The current tree has no tests for the proposed Op-owned exact catalog, closed route intervals, or
fixed materialized subplans. Historical structural tests exist only on the archived experiment
branch and must be rewritten for the revised ownership and supported token envelope.

## 10. Per-operator review record

Each semantic Op should be reviewed with the same compact record:

1. exact reachable physical problems and T envelope;
2. mathematical and BF16/FP32 rounding seams;
3. existing policies and all launcher-side secondary dispatch;
4. independent numerical evidence;
5. cold/warm microbenchmark and roofline model;
6. end-to-end multiplicity and hotspot share;
7. reusable substrate versus shape-hard-coded logic;
8. decision: retain, tune, rewrite, generalize, split, or delete;
9. resulting exact support and route table;
10. workspace and CUDA Graph consequences.

The experiment log should be extended during every operator review with commands, hardware/tool
context, raw output paths, numerical results, performance results, rejected candidates, and final
decision.

## 11. Recommended review order

This is an audit order, not an implementation or migration order.

1. **Base RowSplit Linear substrate**
   - Q4/Q5/Q6/W8 T1 and Small-T;
   - Q4/Q5/Q6 large-T MMA;
   - W8 large-T MMA;
   - exact head kernels.
2. **LinearAdd**
   - two high-multiplicity Text problems;
   - fused, materialized, and residual-MMA policies.
3. **LinearSwiGLU**
   - one high-multiplicity Text problem;
   - paired-row topology and non-monotonic fusion economics.
4. **AttnInputProj and GdnInputProj**
   - shared-input grouped topology;
   - Small-T launch count and Large-T grouped kernels.
5. **LinearPair**
   - reachable W8 MTP K/V problem;
   - delete or separately justify the dead Q5 route.
6. **Vision base problems**
   - large dynamic P/V envelope, partial waves, and B22 padded-K cost.
7. **GdnGatingProj**
   - independent dense lifecycle and split-K/workspace policy.
8. **Dead and generic surfaces**
   - remove unreachable parent policies;
   - retain only reference/measurement generality that has an explicit purpose.

The likely generalization boundary should be evaluated inside each review rather than decided in
advance. Codecs, mainloop topology, fixed geometry, output topology, and finalizer are separate
axes; sharing one axis does not require sharing the complete semantic policy.
