# Linear Kernel Architecture Experiment Log

> Status: retained experiment evidence. E051 supersedes the target-owned profile/injection
> conclusion; kernel, route, numerical, performance, and profiler evidence remains applicable to
> the revised Op-owned architecture.
>
> Started: 2026-07-16.
>
> Scope: evidence used to review and replace
> [`2026-07-16-linear-kernel-architecture-refactor.md`](2026-07-16-linear-kernel-architecture-refactor.md).
> This file records unsuccessful experiments as well as retained decisions. The companion
> architecture document is the recommendation authority; this report is its evidence trail.

## 1. Agreed objective and boundaries

The work will design a better repository-internal Linear kernel architecture rather than preserve
the draft's proposed class tree. The final architecture must:

- preserve correctness and measured performance for every registered Qwen3.6-27B RTX 5090 Linear
  domain;
- make later admission of the accepted Qwen3.6-35B-A3B shapes substantially cheaper without
  implementing or advertising the 35B runtime target in this work;
- use only finite, compiled production policies and permit an exact custom implementation when a
  shared mechanism cannot express the measured winner without cost;
- treat roofline qualification as the final per-policy outcome, reached in measured stages rather
  than assumed from an abstraction;
- drop correctness or performance obligations for unregistered arbitrary shapes;
- investigate current fused projection Ops as compile-time epilogue or output-topology consumers,
  while preserving each Op's semantic rounding, aliasing, and workspace contract;
- retain enough commands, artifacts, measurements, and reasoning to reproduce every architectural
  conclusion.

No current mainloop is presumed optimal. A source abstraction is accepted only after representative
code generation and execution evidence shows that it preserves the required hot path.

## 2. Evidence and acceptance rules

Every experiment entry records:

1. the question and falsifiable hypothesis;
2. source state and exact affected production domain;
3. build and run commands;
4. correctness oracle and numerical boundary;
5. elapsed-time method and cache condition;
6. generated-code resources when relevant: kernel identity, registers, shared memory, spills, and
   SASS differences;
7. NCU limiting-resource evidence only after the target kernel is identified;
8. end-to-end evidence when a high-impact production family or launch composition changes;
9. the retained or rejected conclusion and next experiment.

Matched before/after timing is the primary refactor gate. A single universal roofline percentage is
not used: T1, Small-T, Large-T, tail-heavy, and partial-wave domains have different attainable
limits. Useful traffic/FLOP accounting is kept separate from physical NCU traffic and hardware SOL.

## 3. Initial hypotheses to test

| ID | Hypothesis | Required falsification evidence |
|---|---|---|
| H1 | Numeric-format semantics, storage addressing, and path-specific decode can be separated without changing hot decode instructions. | Extra address/decode instructions, registers, spills, or a matched regression in a representative format/regime. |
| H2 | Small-T has a reusable mainloop across Q4/Q5/Q6/W8, with exact schedule instances rather than hidden launcher dispatch. | A format needs a materially different loop lifetime or the shared body loses to its current implementation. |
| H3 | T1 needs a small set of ownership-specific mainloops rather than one universal T1 loop. | One statically composed body reproduces warp-per-row, split-K, and mixed ownership winners with equal resources and timing. |
| H4 | Q4/Q5/Q6 Large-T can retain one shared lifecycle; W8 and dense may share narrower MMA/epilogue primitives but not necessarily the complete pipeline. | A broader shared lifecycle reproduces current SASS/resources and matched performance for both split-low4 and W8. |
| H5 | Fused Ops can consume a compile-time epilogue/output policy without forcing their semantics into `linear()` or penalizing `StoreBf16`. | Plain Linear changes codegen/performance, or a fused Op cannot preserve its intermediate rounding/output topology through the seam. |
| H6 | Structural finite dispatch can replace caller-role identities and reject unregistered domains without affecting CUDA Graph replay. | A registered route requires model-role information, runtime allocation/synchronization, or unstable launch addresses. |

## 4. Experiment ledger

| ID | Date | Question / change | Domain | Evidence | Result | Decision / next step |
|---|---|---|---|---|---|---|
| E000 | 2026-07-16 | Establish review scope before code or GPU experiments. | Architecture | User alignment plus repository authority review. | Current plan returned to draft; registered-only support, staged roofline work, future-35B shape pressure, and fused-epilogue study accepted. | Inventory exact domains and capture a fresh baseline before prototyping. |
| E001 | 2026-07-16 | Verify that the local build and profiler environment can produce the planned evidence. | Toolchain / RTX 5090 | NCU skill preflight, `nvidia-smi`, `nvcc --version`, CMake cache and compile-option inspection. | `4 OK / 0 WARN / 0 FAIL`; RTX 5090, driver 591.86, CUDA 13.1.115, NCU 2025.4.1.0, Release `sm_120a`; CUDA Op and bench targets compile with `-lineinfo`. | Environment is usable. Record exact clocks per performance run and begin with matched benchmarks before narrow NCU captures. |
| E002 | 2026-07-16 | Capture the pre-refactor aggregate Linear correctness baseline. | Current pure and fused Linear test inventory, including registered and arbitrary fallback shapes. | Release build; existing host mathematical/dequantization references and backend-specific tolerances; timed invocation below. | `OK linear correctness`; 39.46 s elapsed and 11,157,068 KiB maximum RSS. Large-T relative L2 observations were approximately 0.0021--0.0024 against the current `linear_tc` tolerance 0.004; focused pair/add/SwiGLU comparisons passed. | Retain as a broad before-change regression baseline. It is not the future support contract because the test also exercises arbitrary shapes that the registered-only architecture may delete. |
| E003 | 2026-07-16 | Calibrate the existing cold/warm timing rig and locate obvious policy boundaries before choosing the full matrix. | Q4 row-split, `N=34816`, `K=5120`, `T=1..2048`. | 30 CUDA-event samples after five warmups; 256 MiB L2 flush before every cold sample; measured 512 MiB copy and BF16 MMA probes; CSV below. | Copy ceiling 1510.916 GB/s and MMA probe 197.458 TFLOP/s. Cold medians: T1 71.104 us (88.22% useful-bandwidth ceiling), T2 79.488, T4 87.648, T8 206.016, T16 392.800, T32 266.560, T64 273.152, T128 277.888 (83.17% MMA probe), T512 921.216, T2048 3823.456 us. | The rig resolves T1/Small-T/Large-T behavior, but one useful-byte roofline is misleading once Small-T re-streams weights and the synthetic MMA probe can be slightly below an application kernel. Add physical traffic and selected-policy identity; directly compare competing policies at T8/T16/T32 instead of accepting the global threshold. |
| E004 | 2026-07-16 | Determine the actual registered 27B Linear consumer domain rather than infer it from launcher names or benchmark rows. | Text, MTP, Vision schedules and wrappers; future 35B L1--L19 inventory as a design pressure test. | Static call-path and binding audit summarized in Section 6. | Current decode executes `Phase::Verify`, making the target `Phase::Decode` branches and their Q4/Q5 `[7168,5120]` and Q5 paired `[6144,5120]` paths unreachable. Conversely, input-projection wrappers hide six reachable pure-Linear row-view domains. The existing 17-row benchmark omits all seven Vision shapes and is not the registered support matrix. | Base dispatch, correctness, and benchmarks must be rebuilt from reachable structural domains. Dead caller-role specializations may be removed; wrapper-owned grouped/fused paths remain separate consumers of shared internals. |
| E005 | 2026-07-16 | Test whether the global Small-T/Large-T threshold is already selecting the measured winner. | Q4 row-split `[34816,5120]`, T=4/8/16/32; current Small-T and low-bit MMA candidates. | Added measurement-only `--candidate auto|smallt|lowbit_mma|w8_mma` routing and policy identity to `ninfer_linear_op_bench`; 50 cold samples after eight warmups with the E003 1510.916 GB/s copy ceiling. | Auto equals Small-T: 87.392/204.416/390.784 us at T=4/8/16; T32 auto/MMA is 265.856 us. Forced MMA is 261.728/262.432/263.680/265.856 us. T8 favors Small-T by 22.1%, while T16 favors MMA by 32.5%. | Reject the universal threshold of 16. Measure candidate crossovers per structural policy; for this domain the observed boundary is between T8 and T16, with T9--T15 still to confirm. Keep the forcing hook as experimental infrastructure and verify the newly selected numerical boundary before changing production dispatch. |
| E006 | 2026-07-16 | Determine whether fused projection Ops can be expressed as an epilogue extension and what a zero-cost seam must exclude. | LinearAdd, LinearSwiGLU, `linear_pair`, grouped input, GDN control projection. | Semantic-contract and source-dataflow audit; focused prefill-fusion correctness; `cuobjdump --dump-resource-usage` on current explicit instances. | LinearAdd is a single-accumulator finalizer; plain/residual Q5 instances have identical registers/shared/local resources. SwiGLU requires paired weight-row and accumulator topology; W8 Large-T pair is dual projection, while Q5 T1 pair is only a two-job CTA map. The existing grouped Q5 full-tile implementation uses 12 more registers than matched plain, although this does not isolate descriptor cost from topology. | Replace the draft's single epilogue axis with finite `CtaProblemMap`, `AccumulatorTopology`, and final-reduction-only `Finalizer` seams. Make LinearAdd and folded SwiGLU mandatory vertical slices before general refactoring; do not put unproven auxiliary pointers or job state in plain policies. |
| E007 | 2026-07-16 | Capture a broad current-dispatch timing screen and test whether its reported roofline ratios can be treated as physical limits. | Historical 17-row Text/MTP tuning set at T=1/2/4/8/16/32/128/512. | 30 cold samples after five warmups, 256 MiB L2 flush, fixed per-run copy/MMA probes; CSV below. | T1 useful bandwidth spans 349.6--1541.6 GB/s; Q6 head reports 102.1% of the copy ceiling. T512 useful TC reaches 98.2% on a draft head but only 33.6% on W8 KV. At least 15 points have P95/median >=1.5 while minima remain near medians. | Retain absolute median/minimum screening data, but reject `bytes_streamed`, `DRAM_%`, and `TC_%` as physical roofline proof. The sweep is neither a support contract nor a matched candidate comparison; use NCU physical traffic/pipe evidence for selected policies. |
| E008 | 2026-07-16 | Locate structural candidate crossovers instead of replacing one global threshold with another. | Q4 gate/up, Q5 down, Q5 projection, W8 MTP FC. | Three interleaved processes per candidate; 40 cold samples after eight warmups; fixed 1510.916 GB/s ceiling; candidate CSVs below. | Q4 `[34816,5120]`: Small-T wins T8 204.427 vs MMA 261.749 us, but MMA wins from T9 (263.168 vs 375.136 us). Q5 `[5120,17408]`: Small-T wins T24 364.277 vs 407.104 us; MMA wins T32 410.528 vs 456.960 us. Q5 `[6144,5120]`: Small-T wins T17 122.635 vs 130.677 us, T24 is tied within 0.04 us, MMA wins T32 128.640 vs 165.845 us. W8 `[5120,10240]`: Small-T wins T16 167.477 vs 200.309 us; W8 MMA narrowly wins T17 200.288 vs 204.331 us. | Dispatch must encode measured candidate boundaries by format and geometry. Confirm the Q5 exact transition and transfer to all reachable row views/Vision shapes before changing production routes. Candidate identity belongs in plan evidence; `T1/SmallT/LargeT` alone is too coarse to explain policy selection. |
| E009 | 2026-07-16 | Explain the Q4 T16 crossover with a first narrow profiler capture and verify the intended kernels were measured. | Q4 `[34816,5120]`, forced Small-T TT8 vs forced low-bit MMA. | NCU 2025.4.1 `basic`, application replay, one matching launch, reports below; microbench timings remain the performance authority. | Small-T matched `Q4Smallt,TT=8`, grid `(4352,2)`, 67 regs, 8.70 KiB shared, 48.86% achieved occupancy, 84.70% SM SOL and 14.20% DRAM SOL. MMA matched Q4 default edge, grid 544, 152 regs, 45.57 KiB shared, 14.05% occupancy, 74.09% SM SOL and 16.06% DRAM SOL. NCU durations were 534.53 vs 361.79 us and are not used for the speed claim. | The useful-byte interpretation was wrong: T16 Small-T is compute/issue-limited while launching two token tiles and re-streaming weights; the low-occupancy MMA policy still completes less work per output through one BN tile. Proceed to detailed stall/pipe metrics only for retained or regressing candidates. |
| E010 | 2026-07-16 | Test the smallest typed-finalizer vertical slice and remove the current in-place residual alias violation without changing hot loops. | Q5 T1 and split-low4 Large-T plain/residual paths. | Replaced the residual boolean with data-free `StoreBf16` and `ResidualAddBf16InPlace`; build, focused fusion correctness, BF16-word exact direct/fallback cases, formatting, diff check, and before/after `cuobjdump` resources. | Build passed. `[5120,6144]` and `[5120,17408]` T1 direct and T6 fallback each matched `linear + residual_add` with zero BF16-word mismatches; T=17/128/129 remained `max_abs=0`. Registers/shared/stack/local are identical for every matched plain/residual Q5 instance and for Q4/Q5/Q6 plain Large-T. Removing the duplicate pointer reduced kernel constant-parameter storage by 8 B: 960 to 952 B Large-T and 944 to 936 B T1. | Retain the typed finalizer and single read-write pointer seam. Measure matched fused timing; resource and numerical identity do not alone prove timing identity. |
| E011 | 2026-07-16 | Transfer candidate selection to the seven previously unmeasured registered Vision geometries. | Q4/Q5/Q6 Vision encoder and W8 merger shapes. | Added exact shape parsing; three processes per forced candidate with 40 cold samples and product-relevant T sentinels. | Last observed Small-T winner / first observed MMA winner: Q6 `[1152,1536]` T28 34.400 vs 42.603 us / T32 48.736 vs 42.592; Q4 `[3456,1152]` T24 30.304 vs 32.331 / T28 35.339 vs 32.331; Q4 `[4304,1152]` T12 32.213 vs 32.309 / T16 35.136 vs 32.224; Q5 `[1152,1152]` T56 30.304 vs 32.672 / T64 36.288 vs 32.405; Q5 `[1152,4304]` T80 tied at 112.224 / T96 128.597 vs 112.245; both W8 merger shapes favor Small-T at T5 and MMA at T6. | A format-wide crossover is also invalid. Register finite geometry-specific candidate ranges, restricted to reachable P multiples for quantized Vision and V for merger. Add exact numerical tests before routing an earlier T to MMA. |
| E012 | 2026-07-16 | Freeze a code-size and generated-resource checkpoint after adding measurement hooks and the retained E010 finalizer slice. | Linked Linear benchmark/test CUDA binaries. | `stat`, `readelf`, `cuobjdump --dump-elf-symbols`, and resource-usage inspection. | Benchmark 13,320,320 B; test 13,257,600 B. Benchmark sections: `.text` `0x855a2`, `.nv_fatbin` `0x4a5f30`, `__nv_relfatbin` `0x6f0d90`; 89 matching Linear/rowsplit `STO_ENTRY` records. Representative resource table is recorded below; sampled kernels have zero local memory. | Use this checkpoint to reject uncontrolled template Cartesian growth. Entry count and binary sections are diagnostics, not a goal to minimize when an explicit measured policy is required. |
| E013 | 2026-07-16 | Measure the implementation cost/benefit of the retained LinearAdd finalizer rather than infer it from equal resources. | Q5 `[5120,6144]` and `[5120,17408]`, wrapper vs `linear + residual_add`, T1/Small-T/Large-T. | New matched fusion benchmark; three processes, 40 cold samples after eight warmups; residual restore and 256 MiB L2 flush outside timing. | T1 fused latency is 16.1% lower for `[5120,6144]` (20.043 vs 23.893 us) and 5.8% lower for `[5120,17408]` (60.992 vs 64.736). T6/T16 fallbacks match the explicit two-Op reference within 0.5%. Large-T direct finalization is not uniformly faster: it ranges from a 0.66% win to a 4.97% regression on output projection, and up to a 1.82% regression on down projection; repeated regressions are largest at T64--129. | Retain typed finalizer as an expressibility seam, but do not equate compile-time fusion with a winning policy. T1 is qualified; Large-T needs a collective/vectorized epilogue experiment or a measured materialized composition policy plus workspace/end-to-end analysis. |
| E014 | 2026-07-16 | Audit whether format/layout/decode and T regimes correspond to real reusable mainloops, then challenge the draft's dedicated-T1 premise. | Current Q4/Q5/Q6/W8 codecs, Small-T, tuned T1, and Large-T implementations; forced Small-T T1 screens. | Source/dataflow audit, linked-binary resources, and three-process cold timing for the two vocabulary heads and a Q5 down-projection control. | The existing staged Small-T kernel is already one genuine Q4/Q5/Q6/W8 mainloop. Q5 T2--6 direct split-K, ownership-specific tuned T1, split-low4 Large-T, and W8 Large-T have materially different lifecycles and must remain distinct schedules. Codec currently mixes numeric semantics, physical plane geometry, and path decode; its four `load_group()` APIs are unused. Existing Small-T TT4 beats tuned T1 by 3.10% on Q6 `[248320,5120]` and 5.73% on Q4 `[131072,5120]`, but loses by 3.15% on Q5 down and by 41--60% on representative Q4/Q5 projections. | Treat T1 as a problem regime, not an architecture-mandated mainloop. Compose closed compile-time `NumericFormat`, `PhysicalEncoding`, row-split layout, and path-specific decode atoms at row/tile boundaries only; preserve hand-written hot atoms and finite named schedules. Prototype TT1 before selecting the two heads, and reject any source-only separation that changes resources/hot-loop SASS or repeats a >1% median regression. |
| E015 | 2026-07-16 | Verify that every earlier Vision crossover candidate is numerically admissible before any production dispatch change. | All seven registered Vision Linear geometries, two reachable T values bracketing each E011 crossover, `auto` plus both legal forced backends. | New opt-in focused suite; independent FP64 CPU W@x oracle from BF16-rounded activation and test-side row-split dequantization; frozen `linear_bf16` for Small-T and `linear_tc` for MMA. | All 42 checks passed in 0.68 s with 423,160 KiB maximum RSS. Small-T relative L2 was `1.646e-3..1.665e-3`; forced MMA was `2.077e-3..2.186e-3`, below the `4e-3` tensor-core contract. Auto matched the metric of its currently resolved forced backend. | The observed E011 crossover candidates are correct at both sides, including Q4 `[4304,1152]` T12/16 and W8 `[5120,4608]` T5/6. Performance, not numerical admissibility, now gates finite Vision route changes. Retain this exact registered-domain suite. |
| E016 | 2026-07-16 | Determine whether a TT1 specialization makes row-streaming a generally superior T1 schedule or only another finite candidate. | Q4 draft heads N=65,536/98,304/131,072; Q6 full head; three W8 controls, all K=5,120 except W8 FC K=10,240. | Measurement-only `smallt_tt1`; linked-binary resource/SASS inspection; three interleaved cold processes per auto/TT4/TT1 candidate. | TT1 cuts registers from TT4's Q4/Q5/Q6/W8 56/58/59/60 to 40/44/42/40 with unchanged shared and zero stack/local. Q4 TT1 has 640 vs 1,208 static instructions, 42 vs 168 FFMA, four vs 16 `LDG.E.128`, and one vs four `STG.E.U16`, proving columns 2--4 compile away. Nevertheless TT1 loses to TT4 by 2.70/4.62/5.93% across the three Q4 heads and 2.43% on Q6. It wins W8 `[5120,10240]` by 3.69%, ties the launch floor for `[1024,5120]`, and loses 0.32% on `[34816,5120]`. | Reject TT1 as a regime rule. Keep it as measurement evidence and a possible exact W8 FC plan. TT4 is the measured row-streaming head candidate despite higher resources; add exact-head numerical oracle and end-to-end evidence before replacing tuned head routes. |
| E017 | 2026-07-16 | Test whether LinearAdd's Large-T regression is inherent to fusion or caused by per-lane scalar output ownership. | Q5 `[5120,6144]` and `[5120,17408]`, T32/64/128/129; short/full and default/edge MMA instances. | Measurement-only CTA-collective epilogue reusing `Bs[0]`, forced `LDG/STG.E.128`, BF16-word exact tests, three cold processes, linked resources/SASS, NCU basic plus transaction counters at output T128. | All ten T17/32/64/128/129 exact checks per two shapes have zero BF16-word mismatches. Against the current fused wrapper the collective path is 3.88--5.74% faster for output projection and 1.30--2.07% faster for down projection; it also beats the two-launch reference by 1.08--2.49% on output and 0.04--1.20% on down. It adds no shared/local/stack/register cost. Short-full static instructions fall from 2,424 to 1,968. NCU shows identical tensor instructions, global load/store warp instructions falling 51,200/20,480 to 33,280/2,560, with only +2,560 shared loads and +20,480 shared stores. | Fusion is viable, but finalization scope is an architectural axis: semantic `ResidualAddBf16` must compose with a CTA-collective output map, not only `Finalizer::store(index)`. Retain the prototype for end-to-end qualification; plain and direct policies must not inherit its barrier/shared remap. |
| E018 | 2026-07-16 | Complete the missing registered-geometry crossover screen and test the wrapper-private W8 paired threshold. | Reachable Q4/Q5 row views, Q6 head T2--8, six non-Vision W8 geometries, and dual W8 `[1024,5120]` K/V. | Forced existing candidates; broad 20-sample screen, then three interleaved 40-cold-process confirmations at every W8 boundary; paired benchmark can force two Small-T launches or the dual-projection MMA kernel. | W8 `[14336,5120]` and `[34816,5120]` cross at T8/9; W8 `[5120,17408]`, `[5120,6144]`, `[6144,5120]`, and `[5120,10240]` cross at T16/17. The paired `[1024,5120]x2` path crosses only at T56/57: current auto wrongly switches at T17, where dual MMA is 94.7% slower, and remains slower through T56. Reachable Q4/Q5 row views remain decisively Small-T through T16; Q6 head T2--8 also remains Small-T. | Replace the global 16 threshold and wrapper-local duplicate threshold with finite `(format,N,K,T-range,topology)` routes. Pair/dual-projection needs its own range; it cannot inherit the single-projection W8 rule. Do not switch until E015-style exact candidate correctness and end-to-end MTP checks are complete. |
| E019 | 2026-07-16 | Close the registered-domain numerical gaps needed to qualify E016/E018 candidate routes. | Real W8 parent `[14336,5120]` row views, W8 `[1024,5120]x2` K/V, Q4 draft heads, and the Q6 full head. | Opt-in exact-shape suite; suite-private hashed row/group payloads, BF16-rounded activations, test-side dequantization, and an FP64 W@x oracle; wrapper and forced candidates at the observed policy boundaries. | All 32 checks pass in 3.21 s (1,183,836 KiB peak RSS). Small-T/row-view relative L2 is `1.617e-3..1.671e-3`; paired two-Small-T T56/57 is `1.653e-3..1.657e-3`; dual MMA T17/56/57 is `2.310e-3..2.326e-3`; head auto/TT4 is `1.658e-3..1.669e-3`. | The E016 head TT4 and E018 pair candidates are numerically admissible. Retain the exact registered-domain suite; production routing is still gated by end-to-end Text/MTP evidence and the structural plan implementation. |
| E020 | 2026-07-16 | Establish end-to-end before-change baselines for later head, paired-K/V, and LinearAdd routing experiments. | Text prefill/decode and MTP CUDA Graph decode with full or optimized proposal head. | Real 17.50 GB product artifact; Text one process with three repetitions; MTP three independent processes with three repetitions each after one warmup; raw JSON retained. | Text: pp128/512/1024 = 1638.05/3282.75/3464.10 tok/s and tg32 = 77.584 tok/s. Robust MTP tg32 process median-of-medians: full head 72.984 tok/s (72.844--73.097), optimized proposal head 68.333 (68.119--68.338). One full-head repetition had a 57.20 tok/s system slow tail and is excluded by the declared median aggregation. | Use only matched same-path comparisons: full and optimized heads have different deterministic proposal acceptance traces. Preserve per-repetition JSON and the process-median rule; rerun the affected Text/MTP configuration after a production route change. |
| E021 | 2026-07-16 | Determine whether folded LinearSwiGLU is a zero-cost scalar epilogue and whether the current `T>16` fusion route is performance-valid. | Q4 `[34816,5120]` gate/up plus SwiGLU, T1/2/6/8/9/16/17/32/64/128/129. | New fused-vs-materialized benchmark; explicit compile-time `SplitHalfPair` accumulator topology and `SwiGluBf16` finalizer; BF16-word dumps, raw SASS hashes, linked resources, focused correctness, and three 40-cold-sample processes. | The refactor is codegen-zero-cost: all 11 outputs are word-exact; folded edge/full SASS hashes, 107/105 registers, 46,592 B shared, 944 B constants, and zero local/stack are identical. But fusion policy is not monotonic: T1 fused wins 6.25%; materialized T2--16 ties as expected; folded loses 6.06/5.97/4.61/2.16% at T17/32/64/128, then wins 4.40% at T129. | Accept topology plus semantic-finalizer seams, reject “fused Op = scalar epilogue” and reject current `T>16 => folded` routing as a performance rule. Measure the exact transition near T128/129 and qualify workspace/end-to-end impact before changing production dispatch. |
| E022 | 2026-07-16 | Test whether E021's T128/129 reversal is one crossover or a tile-wave policy effect. | LinearSwiGLU folded versus materialized, every T=120..260 plus confirmed boundary sentinels. | No-concurrency dense screen; three independent 40-cold-sample confirmations; launch geometry and exact materialization bytes. | Materialized wins through T128, folded wins T129--256, and materialized wins again at T257--260; every confirmed boundary has the same sign in all three processes. Both contractions use BN128 and grid x=544, but folded uses 256 threads/CTA while plain uses 128 plus a SiLU launch. Full/edge transitions explain discontinuities but not route choice by themselves. | Reject every monotonic threshold and do not extrapolate tile parity. LinearSwiGLU needs a finite `(ceil(T/128), tail/full)` measured policy table and matching workspace query; measure each larger product-reachable tile interval before changing production routing. |
| E023 | 2026-07-16 | Test whether numeric semantics, physical row-split encoding, and path-specific decode can be separated without changing the hot path. | Shared Small-T Q4/Q5/Q6/W8 TT4/TT8, TT1/direct instances, scalar tails, and other codec consumers. | Closed compile-time `NumericFormat`, `PhysicalEncoding`, `RowSplitCodec`, and specialized `SmalltDecodeAtom`; before/after linked resources, extracted SASS hashes/instruction counts, full and focused correctness, and three interleaved 40-cold-sample processes per side. | All 32 Small-T resources match; 121,596 extracted SASS lines are byte-identical (`4f915f...d5ee`), as are other codec consumers. Full correctness passes in 42.59 s. Eight matched timing points differ by at most 0.228%. Four unused whole-group decode APIs are deleted. | Accept H1 with strict limits: finite legal Numeric×Encoding aliases, path-specialized hand-written atoms, and row/tile-boundary composition only. No runtime format tag/view, generic decode formula, or hot-loop address abstraction is justified. |
| E024 | 2026-07-16 | Derive the smallest planner that can express measured discontinuous policies without caller roles or generic quantized fallback. | Current `ShapeFamily/LinearRegime` planner, all reachable 27B base/wrapper domains, and future 35B non-grouped pressure shapes. | Call-path and binding audit, exact physical-view inventory, measured-boundary mapping, workspace/CUDA Graph ownership analysis, and source audit of post-plan hidden routing. | Current key drops exact T, padded K, view/topology, treats any N>=65,536 as a head, applies one threshold 16, accepts arbitrary quantized shapes, and then silently changes routes in `linear.cpp` and launchers. Exact product draft head is only `[131072,5120]`; tuning N=65,536/98,304 are not support. | Replace `ShapeFamily/Regime` with separate exact `SupportSpec` and finite `RouteSpec`, keyed by format, effective `(N,K,Kpad)` view, token set, and semantic topology. Each semantic wrapper owns a resolver but shares structural vocabulary; workspace sizing and launch consume the same plan. |
| E025 | 2026-07-16 | Close the remaining single-projection route gaps needed by an exact finite table. | Three Q5 Text geometries and the two unresolved Q5 Vision points. | Sequential candidate screen followed by three independent 40-cold-sample confirmations at every boundary/near-tie; no concurrent GPU work. | Q5 `[5120,17408]` and `[5120,6144]` both switch cleanly at T24/25. Q5 `[6144,5120]` favors Small-T through T22, ties at T23, is within 0.03% at T24, and favors MMA from T25. Vision `[1152,1152]` favors MMA at reachable P60; `[1152,4304,Kpad4352]` favors Small-T at P84, ties at P88, and favors MMA at P92. | Encode continuous Text ranges, retaining tie bands as evidence rather than false precision. Vision can route MMA from P60 and P88 respectively because those points match or beat Small-T, but do not infer unmeasured non-multiple/product points. |
| E026 | 2026-07-16 | Determine the exact product T domain and workspace ownership needed to qualify discontinuous LinearSwiGLU routes. | 27B Text prefill, ordinary/MTP verify, prefix tails, and sequence-arena planning. | Source/call-path and layout arithmetic audit. | Default product prefill reaches every integer T=1..1024, not only chunk multiples; ordinary verify uses T1 and MTP verify T2..6. A workspace query at only configured chunk C is invalid for non-monotonic routes: it must maximize the shared resolver over all reachable T<=min(C,M). Materialization costs 69,632*T B, but current 27B global workspace remains unchanged because GDN's 164,440*C-B peak exceeds the materialized MLP's 124,952*C B. | Qualify all eight default BN128 tile-count bands before production selection. Use the same resolver for exact launch and range-max workspace planning; do not rely on GDN dominance as a cross-target contract. |
| E027 | 2026-07-16 | Audit whether E017 is sufficient to promote CTA-collective LinearAdd and define its true competitor/domain. | Both Q5 residual geometries across Text prefill, ordinary/MTP verify, arena planning, and short/default MMA variants. | Source/call-multiplicity/tile/layout audit against E013/E017/E025. | Each Text pass invokes 64 of each geometry (128 total); default prefill reaches every T1..1024. E017 lacks collective performance at T17--31 and all default-full execution; its composed reference follows the currently wrong auto MMA route, while E025 shows forced Small-T is the real winner through T24. Current layout asks workspace only at chunk C and relies on unrelated arena headroom for materialized tails. | Do not promote `T>16 => collective`. Extend the benchmark with forced-Small-T+residual and forced-MMA+residual; independently validate backend numerics, compare collective from T17 through tile boundaries, and make route/workspace share one resolver. |
| E028 | 2026-07-16 | Implement the Phase-1 exact structural planner while holding CUDA policy behavior constant. | Base dense/quantized Linear and dead Q5 pair specialization. | 24 exact quantized `SupportSpec`, 72 T-range `RouteSpec`, new host plan-matrix test, adapted internal arbitrary-shape tests, full/focused correctness, linked entry/resources, and matched end-to-end reports. | Caller-role families/regimes and arbitrary quantized fallback are removed; Q4 65,536/98,304 and Q4/Q5 7,168 are rejected; dense reference remains separate. All tests pass; Linear entries remain 89 and resources unchanged. MTP cases remain within 0.52% of robust baselines. Three Text processes show a near-constant +0.95--1.47 ms prefill offset (-1.30% at P128, -0.61% P512, -0.50% P1024), while graph TG32 is +0.91%; no hot kernel changed. | Retain the structural cutover, but treat the full 72-route scan as a host-overhead hypothesis. Benchmark old/full-scan/indexed lookup and restrict resolution to the matched support's route span before declaring the host seam zero-cost. |
| E029 | 2026-07-16 | Measure and remove the Phase-1 planner's avoidable full-table scan, then test whether it explains E028's eager-prefill offset. | CPU-only base-Linear resolution over the exact 209/401-call Text projection mix. | New `ninfer_linear_plan_bench`; faithful legacy classifier, frozen 24-support/72-route full scan, and production exact resolver; 5,000 passes and nine medians plus a pinned 10,000-pass/11-median confirmation. | Production resolution falls from 47.40--47.57 to 7.91--7.94 ns/query (-83.3%); pinned confirmation is 7.94--8.03 ns. A 209-call pass falls from about 9.9 to 1.7 us and a 401-call pass from about 19.0 to 3.2 us. The old role classifier is about 1.1 ns/query. Even the former full scan accounts for only 1.3--2.0% of E028's +0.95--1.47 ms offset. | Retain exact support lookup plus a compile-time-derived variable contiguous route span; it preserves non-contiguous/multiple future token ranges without scanning unrelated supports. Reject planner cost as the explanation for the E028 millisecond offset and do not manufacture an end-to-end regression claim from unmatched environmental samples. |
| E030 | 2026-07-16 | Select LinearAdd's real Small-T/MMA/finalization route over its complete default product domain. | Q5 residual `[5120,6144]` and `[5120,17408]`, every T17--31 plus every BN64/128 boundary sentinel through T1024. | Five explicit benchmark paths; identical residual restore/L2 flush; 82 BF16-word validations; 5/20 screen followed by three independent 8/40 cold processes and a 274-row exact summary. | All collective outputs are word-exact to forced MMA plus separate residual. Both shapes decisively prefer forced Small-T plus residual through T24 and switch at T25 to CTA-collective MMA. Cross-process medians choose collective at every measured T25--1024. A few single-process high-T down-projection reversals occur inside an approximately 0--1% margin but do not reproduce in the aggregate. | Recommend a finite LinearAdd policy: fused tuned T1; materialized Small-T for T2--24 with `10240*T` B scratch; CTA-collective MMA for T25--1024 with zero scratch. Do not extrapolate beyond the registered max. Exact launch and range-max workspace sizing must share this resolver. |
| E031 | 2026-07-16 | Make every base-Linear schedule selected by the host plan rather than by a second launcher-side classifier. | All 24 exact quantized supports; existing TT4/TT8, Q5 direct, split-low4 tile, and W8 tile schedules. | 17 structural policy IDs, 83 contiguous variable-length routes, fixed production launchers, compile-time closure, expanded plan test, all related builds, and full GPU Linear correctness. | Product dispatch now names and directly launches TT4, TT8, Q5 direct split2/split4, split-low4 BN64/BN128, and W8 small-M/general. Full/edge remains a uniform variant. Measurement-only auto launchers remain outside product resolution. The unsupported Q5 N7168 direct branch is deleted. Plan/build/format/diff checks pass; full correctness reports `OK` in 40.87 s with 11,149,516 KiB peak RSS. | Accept the visible-policy seam. It removes dual route truth without changing a kernel choice, and gives subsequent measured crossover/fused updates one authoritative place to change. Preserve shared policy IDs when two exact shapes truly use the same schedule; the plan's weight signature retains their distinct legality. |
| E032 | 2026-07-16 | Decide whether split-low4 and W8 Large-T should share one complete producer-driven mainloop. | Q4/Q5/Q6 split-low4 MMA, W8G32 MMA, grouped input, folded SwiGLU, dual W8, and collective residual consumers. | Line-by-line lifecycle audit plus linked resource/SASS evidence; no source or GPU change. | The ready-BF16 fragment traversal/MMA math is common, but producer state determines the whole K-loop. Split-low4 uses two raw stages and `wait<1>`; W8 uses a single raw code tile, a scale slab persistent across four K tiles, `wait<0>`, a different prefetch point, eight warps, and forced fragment ping-pong. A superset adds at least 4--5 KiB shared to W8's existing 47,104 B and threatens its measured two-CTA residency. Static SASS schedules/resources are materially different. | Accept H4 in a narrower form: retain complete `SplitLow4MmaMainloop` and `W8G32MmaMainloop`; share only Linear-local swizzle, closed serial/ping-pong fragment consumers, MMA atoms, and finalization helpers. A universal producer-hook K-loop would be a scheduling DSL and is rejected absent a new measured pipeline. |
| E033 | 2026-07-16 | Resolve LinearSwiGLU's non-monotonic route over every registered default token count. | Q4 `[34816,5120]` gate/up plus SwiGLU, every T17--1024; fused T1 and materialized T2--16 retained from E021. | 5/20 cold screen at all 1,008 values, then three independent 8/40 confirmations at 43 band boundaries/representatives/reversal points; identical restore/256 MiB flush and explicit folded versus materialized paths. | Stable bands are materialized T17--128, folded T129--256, materialized T257--384, folded T385--512, materialized T513--640, and folded T641--1024. The first five BN128 tile-count bands alternate; bands six/eight contain sub-1% noisy screen reversals, but confirmation supports retaining the current folded route, including a practical tie at T1024. | Recommend exact policy T1 fused, T2--128 materialized, 129--256 folded, 257--384 materialized, 385--512 folded, 513--640 materialized, and 641--1024 folded. Range-max workspace is the maximum materialized scratch below the requested limit (at C>=640, `69632*640` B), not scratch for route(C). |
| E034 | 2026-07-16 | Promote the measured LinearAdd and LinearSwiGLU policies into independent production plans with route-consistent capacity sizing. | Two Q5 LinearAdd projections and Q4 `[34816,5120]` SwiGLU; performance-qualified T1--1024. | Exact finite route tables; fixed launch switches; compile-time closure and CPU plan/workspace tests; focused GPU comparisons at every measured policy boundary. | LinearAdd resolves T1 tuned fused, T2--24 materialized, T25--128 BN64 collective, and T129+ BN128 collective. SwiGLU resolves the seven E033 ranges, with folded as the conservative terminal route. Capacity queries maximize exact-route scratch over finite route intersections with `[1,C]`. Focused GPU test passed in 1.52 s with 659,124 KiB RSS: every LinearAdd word comparison and every SwiGLU boundary comparison through T1024 was exact. | Retain per-Op planners and range-max workspace. Epilogue/topology reuse is an implementation seam, not a shared route table or a promise that fusion always wins. T>1024 is a correctness fallback, not performance evidence. Full build and end-to-end timing remain follow-up gates. |
| E035 | 2026-07-16 | Promote the measured base-Linear crossovers and give `linear_pair` one exact semantic plan instead of inheriting base routes. | All 24 registered base views; W8 pair `[1024,5120]` x2 with candidate-performance evidence through T1024 and production closure still pending. | Updated finite route tables and fixed launch selection; compile-time closure plus CPU boundary/rejection tests; product-route GPU sentinels added but deliberately not yet executed. | Base routes now encode E008/E011/E018/E025 boundaries. Pair resolves two TT4 launches at T1--4, two TT8 launches at T5--56, and dual W8 MMA at T57+; it rejects non-identical/non-registered views and nonpositive T. Plan test reports `OK linear structural plan`; builds, formatting, and diff checks pass. | Retain independent semantic planners. A fused/multi-output Op may reuse fixed Linear kernels, but must not inherit a base Op's route table when its joint launch economics differ. T>1024 retains the legal terminal schedule without a speed claim. Run focused GPU correctness and MTP end-to-end evidence before qualifying production behavior. |
| E036 | 2026-07-16 | Build a measurement-only 35B pressure harness without accidentally admitting 35B production support. | Ten non-grouped 35B geometries across W8, Q4/Q6 heads, and dense BF16. | Exact benchmark names and format contracts; CUDA-hidden parser tests; benchmark build/help/format/diff checks. | Eight quantized shapes require an explicit forced candidate and reject `auto`; two dense shapes use only the generic dense reference route. None is added to `--all-targets` or a production support table. All parser contracts and the build pass. | Use the harness to test whether current format/mainloop/policy vocabulary transfers to 35B geometry. Performance evidence may nominate policies, but cannot register a target or imply grouped-expert support. |
| E037 | 2026-07-16 | Re-audit finite support against real Engine reachability and fixed-launch physical legality. | Base/fused/pair planners, public prefill chunk, target arena/bindings, stale `Phase::Decode` branches. | Whole call-chain review, CUDA grid arithmetic, alignment audit of every registered view and target allocation/slice, compile-time closure changes, and host validation implementation. | Public prefill chunks can exceed the 1024 performance matrix; terminal BN128 routes are launch-legal through `128*65535=8,388,480`. INT32_MAX was a false closure. All current target operands/planes are at least 16B aligned, while Q5 direct/MMA/vector finalizers require that fact. `Phase::Decode` has no caller; its Q4/Q5 7168 and dead Q5 pair paths were stale. | Set one quantized Linear T ceiling at 8,388,480, cap target prefill chunks there, keep T1025+ as an unqualified terminal fallback, require 16B Linear-family operands/planes, and delete the unreachable Decode branches instead of registering dead shapes. CPU/GPU regression verification follows in E038. |
| E038 | 2026-07-16 | Run the first whole-suite regression after the planner and fused-policy cutover. | Base Linear, LinearAdd, LinearSwiGLU, alignment, tails. | Full GPU correctness suite. | All numerical and exact-word checks passed. A later build-provenance audit found this was a Debug binary, so its 11:09 wall time is invalid performance evidence. | Retain correctness only; repeat under a verified Release build in E043. |
| E039 | 2026-07-16 | Close product-shape `linear_pair` correctness above its T56/57 crossover. | W8 `[1024,5120]` K/V row views, T128/T1024. | Public pair wrapper versus two public single projections, plus existing independent crossover oracle. | K and V had zero BF16-word mismatches. The run was Debug, so its elapsed time is discarded. | Retain topology equivalence; repeat the focused suite in Release and measure pair performance separately. |
| E040 | 2026-07-16 | Distinguish launch legality from the T<=1024 tuning domain. | Registered Add/SwiGLU/pair paths at T1025/T2048. | Focused exact-word comparisons. | All paths passed through T2048. The runs were Debug, so only functionality survives. | Treat T1--2048 as measured functional coverage; keep T2049--8,388,480 as grid-proved terminal fallback and make no blanket speed claim. |
| E041 | 2026-07-16 | Pressure-test current policy vocabulary on future 35B non-grouped shapes. | Eight quantized and two dense measurement-only geometries. | Candidate screen and three-process confirmation. | The experiment demonstrated non-monotonic, shape-local route structure and a missing dense schedule, but the entire timing set was later found to be Debug. | Preserve the hypothesis and harness, invalidate every timing/crossover, and rerun from scratch in verified Release as E044. |
| E042 | 2026-07-16 | Numerically validate the future-shape candidates without admitting 35B. | Eight future quantized geometries. | Independent packed-row FP64 T1 oracle plus fixed-candidate comparisons at lifecycle seams. | All checks passed; cross-lifecycle `rel_l2=2.519e-3..2.543e-3`, and same-lifecycle high-T candidates were word-exact. The Debug elapsed time is discarded. | Retain the numerical method, then repeat in Release to close build provenance. |
| E043 | 2026-07-16 | Audit build provenance and recover a trustworthy final correctness baseline. | Entire Linear change set. | CMake cache/compile-command inspection, explicit Release reconfigure, focused and full reruns. | The cache had silently become Debug after E037. Reconfigured `Release sm_120a`; nvcc again uses `-O3 -DNDEBUG -lineinfo`. Future35, full, fusion, and registered-gap suites all pass in 2.67/44.21/2.54/3.15 s. | Correctness conclusions survive; all E038--E042 elapsed times and all E041 route timings are formally superseded. Require build-type evidence beside every final performance claim. |
| E044 | 2026-07-16 | Repeat future-35B pressure in verified Release and decide whether it requires a generic loop. | Six W8, two head, two dense future geometries; no product admission. | Full candidate screen, three-process boundary confirmation, Release future-shape correctness. | Existing closed codecs and fixed kernels cover all eight quantized shapes, but route ranges remain geometry-local and sometimes non-monotonic. Dense BF16 reaches only 2.55--3.16 useful TFLOP/s at T2048. Release numerical suite passes. | A future target should add exact supports and measured route spans, not a universal mainloop. Dense BF16 needs its own optimized schedule; grouped experts remain separate work. |
| E045 | 2026-07-16 | Replace the stale tuning list with the real 27B support matrix and sample high-T registered workloads. | 24 exact base views, pair through T2048, seven Vision views through T32768. | Three Release processes, 8 warmups/40 cold samples; public/focused correctness. | `--all-targets` now enumerates exactly 24 registered views. Pair auto matches dual MMA at T128--2048. At T32768, Vision kernels reach 187.4--211.8 useful TFLOP/s; all routes run successfully. | Keep the 24-view inventory as the support/performance matrix. T32768 is terminal only for the two W8 merger views; encoder P remains reachable through 131072. |
| E046 | 2026-07-16 | Determine whether final Text drift is caused by the new fused routes. | Real-artifact P128/P512/P1024 and TG32. | Seven Release processes, GPU power/clock observation, matched A/B/C/B/A policy ablation. | Absolute results drifted 2--4% slower as the GPU entered power limiting. Within one matched bracket, current Add and SwiGLU routes improve prefill by 0.63--1.38%; TG32 is unchanged within 0.16 ms. Clock locking was unavailable. | Do not attribute environment drift to the refactor. Retain the new routes because same-session causal ablation shows an end-to-end win. |
| E047 | 2026-07-16 | Preserve real product behavior across MTP and Vision. | Six MTP configurations and public one-image Vision request. | Three Release processes per MTP case; three isolated public Engine Vision runs. | All 54 MTP repetitions preserve exact speculative rounds/drafts/accepts/fallbacks; timing shows the same 2--4% environmental drift as Text. Vision completes through the 17.5 GB artifact with 9.15--10.11 ms media time and about 100.9 ms total. | Product semantics are preserved. Use the isolated Vision branch as a small real integration gate; do not infer 32K-token numerical proof from it. |
| E048 | 2026-07-16 | Re-attribute final full-inference latency under Release. | P128, TG32, and MTP P32+G32 optimized-head routes. | Isolated `cudaProfilerApi` capture around one measured request; NSYS report plus SQLite summary. | Linear/GEMM is 93.0% of P128 kernel time and 77.7% of the MTP kernel time. Decode remains dominated by T1/Small-T Linear families. Release kernel sums are lower than the invalid Debug captures. | Keep Linear as the dominant optimization target. Use NSYS for product attribution and NCU only for selected fixed kernels. |
| E049 | 2026-07-16 | Establish physical roofline/resource evidence for representative final policies. | T1, TT4, Q5 direct, BN64/BN128, W8 small/general, pair, Add, SwiGLU, 32K Vision. | NCU application replay, exact base-name filters, one launch; microbench timing remains authoritative. | The sampled policies occupy distinct resource/traffic regimes: 8.3--89.9% achieved occupancy, 24.2--90.7% SM SOL, 3.8--76.8% DRAM SOL, 37--154 registers, and 4.1--47.6 KiB static shared. No sampled kernel spills. | Reject one global roofline threshold and qualify per fixed policy plus wave/tail class. Retain separate split-low4 and W8 lifecycles and treat profiler duration only as diagnostic. |
| E050 | 2026-07-16 | Close source/binary congruence and verify the exact final worktree. | Single exact support/route-profile authority, support-derived benchmark inventory, fail-closed launch, dead tuned kernels, all Linear tests and default build. | Release build, CPU structural tests, four GPU correctness modes, 24-view/pair smoke, formatting/diff checks, linked entry/SASS/resource audit. | All checks pass after fixing one missing CUDA include path on the host-only planner benchmark. The final benchmark has 85 true Linear entries, zero 7168/dual-Q5 entries, zero stack/local, and is 123,152 B smaller than the pre-cleanup checkpoint. | Accept the implemented foundation. Keep per-policy roofline closure as Stage B rather than claiming all 83 routes are qualified. |
| E051 | 2026-07-16 | Re-evaluate the semantic boundary before implementing the clean cut-over. | Proposed target-owned `LinearExecutionProfile`, all Linear-family APIs and callers. | Interface/ownership review against the original Op contract and the requirement that callers remain unaware of dispatch. The uncommitted clean implementation was discarded and its branch deleted. | Explicit or hidden target-profile injection makes target code part of Linear routing, leaks implementation semantics through every caller, and cannot be repaired by moving the same dependency into Engine, Program, `Weight`, global state, or workspace. The original `linear(x,w,out,ws,stream)` boundary is correct. | Supersede the ownership conclusion of E050. Retain its physical problem inventory, route winners, fixed policies, codegen, correctness, and performance evidence; rebuild exact admission and routing as an immutable hardware-specific catalog wholly owned by each semantic Op. Reject unqualified grid-ceiling tails as “best” production routing. |
| E052 | 2026-07-16 | Define the implementation boundary for Q4 pure Linear before writing kernels. | Q4G64 RowSplit only; current seven physical contractions plus future `[131072,2048]` pressure. | Current-source audit, retained E003/E008/E011/E014/E016/E044/E049 evidence, three independent design/implementation/verification reviews, and the accepted Q4 template design document. | The bounded architecture is three template lifecycles—GEMV, SIMT GEMM, and MMA GEMM—with runtime Rows/K/Cols, a finite `ScheduleId + KernelVariant` set, exact Op-owned admission, and no application-role naming. Split-K partitions aligned scale pairs; SIMT owns vectorized partial K stages; MMA fixes BK64. | Implement dormant fixed candidates first while production dispatch remains unchanged. Rebuild the measurement harness from current source before making any new performance claim, then qualify and cut over Q4 atomically. |
| E053 | 2026-07-16 | Integrate the first dormant Q4 templates and reject any abstraction-induced regression before route work. | Three GEMV schedules, SIMT C4/C8 Full+Predicated, MMA C64/C128 Full+Predicated; matched legacy points. | Fresh Release `sm_120a` build; independent BF16-boundary/Q4-dequant/FP64 oracle; `cuobjdump` resources; one-process 5-warmup/20-cold structural screen with canonical candidate CSV. | All 11 initial candidates passed; no local/stack spill. New versus legacy cold median: warp-row 68.896/70.880 us (-2.8%), split-K 15.616/11.552 us (+35.2%), SIMT C4 27.904/32.032 us (-12.9%), SIMT C8 134.112/204.064 us (-34.3%), MMA C64 105.728/105.760 us (tie), MMA C128 29.952/29.920 us (tie). The split-K candidate used 48 registers versus the old winner's 34 and changed the lane/decode/scale topology. | Keep the three-family architecture and the successful schedules, but reject the first R1/W8 schedule. Make GEMV lane/decode/transfer topology an explicit compile-time schedule axis and regenerate split-K as byte-per-lane, scalar-scale/shuffle, synchronous-vector code staging before any qualification or commit. |
| E054 | 2026-07-16 | Recover the split-K winner without reintroducing an application- or exact-Rows kernel. | `q4_rowsplit_gemv_kernel` R1/W8 at K5120; runtime and static group-ownership variants. | Bounded sequence: runtime PackedWord8 15.616 us; runtime PackedByte2 pair-loop 19.488 us; active-groups-10 unrolled 17.408 us; then the same template with `StaticGroupsPerRow=80`. Each step used the same Release benchmark and independent numerical oracle; final resources came from linked `cuobjdump`. | The final static-group schedule is 11.520 us versus legacy 11.520 us; R4/W1 is 70.880 versus 70.912 us. R1/W8 uses 36 versus 34 registers, identical 5,152 B linked static shared, zero local/stack spill, and a matching hot mix of 20 FFMA, 10 X loads, 10 code shared loads, and 10 shuffles. The complete candidate suite, including K5120 numerical coverage and static-K rejection, passes. Runtime ownership prevented the compiler from recovering the legacy hot-loop ILP even after the lane/decode path matched; specializing group count, not Rows or role, restored it. | Accept `StaticGroupsPerRow` as an evidence-gated codegen axis with 0 meaning runtime. Bind R1/W8 to 80 groups/K5120 and reject other K; any K5120 Rows can reuse it. Do not instantiate further static K values unless a runtime schedule fails the same >1% gate. |
| E055 | 2026-07-16 | Qualify the Q4 production route catalog and its physical limits before atomic cut-over. | Seven current pure-Linear supports, all reachable Text integers/Vision multiples of four, and measurement-only future `[131072,2048]`. | Frozen 1512.483 GB/s copy and 220.246 TFLOP/s BF16-MMA context; 5/20 screen; three independent 8/40 confirmations at matched anchors and every retained seam; NCU detailed, explicit traffic, and stall captures for six final topologies. | New/legacy medians are equal for both GEMVs and C128, C64 is 1.2% faster, C4/C8 are 12.9%/34.1% faster. Final current routes use R1/W8 for three K5120 T1 views, R4/W1 direct for the draft head, C4/C8 for Text and early Vision, C64 for Vision mid bands through C320, and C128 from C324. R4/W1 shared has no winner route. Saturated C128 reaches 90.66% SM SOL with 63.7M tensor instructions; R4 direct reaches 75.18% DRAM SOL with 356.55 MB DRAM reads matching the Q4 payload; no topology spills. Future K2048 needs no new kernel: R4 at C1, C8 at C8, C64 at C9--64, C128 at C65+. | Freeze the exact finite route table in the Q4 Op-owned planner; retain six production schedules and both Full/Predicated variants where measured, delete R4 shared and all legacy measurement selectors during cut-over, and do not admit future K2048 yet. |
| E056 | 2026-07-16 | Atomically cut production Q4 pure Linear over to the qualified three-template architecture without changing caller semantics. | Seven exact supports, six schedules/ten binary entries, Q4 public dispatch, dead parent routes, Text/Vision/MTP real product paths. | Full Release build; fixed oracle; host plan closure; 34 public-auto/fixed BF16 word-exact route points; full Linear regression; real-artifact prefix Text/Vision/MTP test; detached `eb9257e` old/new `ninfer_bench` brackets. | All correctness and real-artifact gates pass. Stable Text bracket: new 1633.09/1635.07 versus old 1633.87 prefill tok/s; new 78.61/78.55 versus old 77.90 decode tok/s. Optimized-head MTP: new 120.77/120.82 versus old 108.33 output tok/s with identical 0.5 acceptance and 24 rounds/12 fallbacks. Initial 1583/1677 Text values were frequency/power drift and disappear in the immediate bracket. | Keep the cutover. Q4 now resolves entirely inside `linear()`, unknown problems fail closed, old Q4 GEMV/Small-T/MMA routes and legacy selectors are deleted, and future shape work adds admission/route data before considering a new schedule. |

## 5. Environment and baseline inventory

The initial source state is commit `5834bec` (`docs(linear): define kernel architecture refactor`)
with a clean worktree before the documentation changes recorded here.

E001 environment:

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB, PCI `00000000:01:00.0` |
| Observed maximum clocks | SM 3090 MHz; memory 14001 MHz (reported by `nvidia-smi`) |
| Driver / host | 591.86; WSL2 6.6.87.1 |
| CUDA compiler | CUDA 13.1, nvcc 13.1.115 |
| Nsight Compute | 2025.4.1.0 |
| Build | Release, `CMAKE_CUDA_ARCHITECTURES=120a`, CUDA `-O3 -DNDEBUG` |
| Source attribution | `src` and benchmark CUDA targets add `-lineinfo` |
| NCU permission preflight | `4 OK / 0 WARN / 0 FAIL`; WSL2 profiling permission is host-controlled |

Commands:

```bash
/home/neroued/.codex/skills/ncu-kernel-profile/scripts/preflight.sh
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.memory,pci.bus_id \
  --format=csv,noheader
/usr/local/cuda/bin/nvcc --version
rg -n 'CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_FLAGS|CMAKE_CXX_FLAGS' \
  build/CMakeCache.txt
rg -n 'lineinfo' src/CMakeLists.txt bench/CMakeLists.txt
```

E002 command:

```bash
cmake --build build -j --target ninfer_linear_test ninfer_linear_op_bench ninfer_bench
/usr/bin/time -f 'elapsed=%e max_rss_kb=%M' ./build/tests/ninfer_linear_test
```

The build was already up to date (`ninja: no work to do`). The existing test chooses its comparison
tolerance from token count, so it is adequate for the frozen baseline but must not become the
selection oracle once policy crossovers are made explicit.

E003 command and retained artifact:

```bash
nvidia-smi --query-gpu=name,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit \
  --format=csv,noheader
./build/bench/ninfer_linear_op_bench \
  --shape MlpGateUp34816x5120 --qtype Q4 \
  --warmup 5 --repeat 30 --copy-repeat 12 \
  --t-sweep 1,2,4,8,16,32,64,128,512,2048 \
  --csv-out profiles/bench/linear-architecture-20260716/baseline/q4_gateup.csv
```

Before the run the idle GPU reported P0, 765 MHz SM, 14001 MHz memory, 37 C, and 104.39 W;
the benchmark warmups were responsible for clock ramp. The cold P95 contained occasional large
host/clock outliers at T8, T16, T512, and T2048, while the medians and minima were close for most
points. Matched comparisons therefore need interleaving or repeated processes in addition to a
single within-process median.

E005 commands and retained artifacts:

```bash
cmake --build build -j --target ninfer_linear_op_bench
./build/bench/ninfer_linear_op_bench \
  --shape MlpGateUp34816x5120 --qtype Q4 --candidate CANDIDATE \
  --warmup 8 --repeat 50 --stream-ceiling-gbs 1510.916 --t-sweep 4,8,16,32 \
  --csv-out profiles/bench/linear-architecture-20260716/candidates/q4_gateup_CANDIDATE.csv
```

`CANDIDATE` was each of `auto`, `smallt`, and `lowbit_mma`. The instrumentation calls the same
internal production launchers; it does not add an alternate kernel. The CSV appends `candidate`
and `resolved_policy`, leaving existing columns in place.

E007 retained artifact:

```bash
./build/bench/ninfer_linear_op_bench \
  --all-targets --candidate auto --warmup 5 --repeat 30 --copy-repeat 12 \
  --t-sweep 1,2,4,8,16,32,128,512 \
  --csv-out profiles/bench/linear-architecture-20260716/baseline/all_targets_auto.csv
```

The measured ceilings for this process were 1509.761 GB/s and 206.560 useful TFLOP/s. Its
historical target list is retained only as a broad screen; Section 6 is the actual domain audit.

E008 used the same command shape as E005 with `--warmup 8 --repeat 40`, a fixed
`--stream-ceiling-gbs 1510.916`, and three alternating `smallt`/MMA process rounds. Artifacts are
under `profiles/bench/linear-architecture-20260716/crossover/` and include:

- `q4_gateup_{smallt,lowbit_mma}_r{1,2,3}.csv`, T=8..17 plus 24/32;
- `q5_down_{smallt,lowbit_mma}_r{1,2,3}.csv`, T=6/8/9/12/16/17/24/32/64/128;
- `q5_proj_{smallt,lowbit_mma}_r{1,2,3}.csv`, adding T65;
- `w8_mtp_fc_{smallt,w8_mma}_r{1,2,3}.csv`, T=4..129 sentinels.

E009 commands and reports:

```bash
ncu --force-overwrite --replay-mode application --set basic \
  --kernel-name-base function --kernel-name regex:linear_rowsplit_gemm_smallt_kernel \
  --launch-skip 0 --launch-count 1 \
  -o profiles/ncu/linear-architecture-20260716/q4_gateup_t16/smallt_basic \
  ./build/bench/ninfer_linear_op_bench \
  --shape MlpGateUp34816x5120 --qtype Q4 --candidate smallt \
  --t-sweep 16 --warmup 0 --repeat 1 --copy-repeat 1 --stream-ceiling-gbs 1510.916

ncu --force-overwrite --replay-mode application --set basic \
  --kernel-name-base function --kernel-name regex:linear_rowsplit_gemm_mma_kernel \
  --launch-skip 0 --launch-count 1 \
  -o profiles/ncu/linear-architecture-20260716/q4_gateup_t16/mma_basic \
  ./build/bench/ninfer_linear_op_bench \
  --shape MlpGateUp34816x5120 --qtype Q4 --candidate lowbit_mma \
  --t-sweep 16 --warmup 0 --repeat 1 --copy-repeat 1 --stream-ceiling-gbs 1510.916
```

Both `.ncu-rep` files were imported with `ncu --import REPORT --page details --csv`; kernel names,
grid, resources, and SOL values in E009 come from those reports. `--set basic` required nine
application-replay passes on this NCU version. The benchmark's event timing while NCU is attached
is intentionally discarded.

E010 verification:

```bash
cmake --build build --target ninfer_linear_test ninfer_linear_op_bench -j 4
/usr/bin/time -v env NINFER_LINEAR_TEST_PREFILL_FUSIONS_ONLY=1 \
  ./build/tests/ninfer_linear_test
/usr/local/cuda/bin/cuobjdump --dump-resource-usage build/tests/ninfer_linear_test 2>/dev/null \
  | c++filt
clang-format --dry-run --Werror \
  src/ops/linear/linear_finalizer.cuh \
  src/ops/linear/gemm/linear_rowsplit_gemm_mma.cu \
  src/ops/linear/gemm/linear_rowsplit_gemm_mma.cuh \
  src/ops/linear/gemv/linear_rowsplit_gemv_q5_core.cuh \
  tests/ops/test_linear.cpp
git diff --check
```

The expanded focused suite took 0.85 s wall time and 377,408 KiB maximum host RSS in this run.
The largest new case uses an approximately 55.78 MiB Q5 payload plus a 64 MiB workspace arena;
the two shapes execute sequentially. The exact checks copy outputs as raw `uint16_t`, report a
first differing word on failure, and observed zero mismatches for all four T1/T6 cases.

E011 first added the seven exact Vision names and `MtpQGate6144x5120` to the benchmark's explicit
`--shape` parser without changing its historical `--all-targets` list. Each Vision comparison used
the E008 protocol and writes three process rounds per candidate under
`profiles/bench/linear-architecture-20260716/crossover/`, with prefixes `vision_` and
`vision_exact_`. Quantized Vision P is a multiple of four, so the retained policy boundaries use
T=12/16, 24/28, 28/32, 56/64, and 80/96 rather than unreachable odd-P points. Merger V is not
subject to that multiple-of-four restriction; its W8 boundary was narrowed to T5/T6.

E012 commands:

```bash
stat -c '%n %s bytes' build/bench/ninfer_linear_op_bench build/tests/ninfer_linear_test
readelf -SW build/bench/ninfer_linear_op_bench | rg '\.text |\.nv_fatbin|__nv_relfatbin'
/usr/local/cuda/bin/cuobjdump --dump-elf-symbols --gpu-architecture sm_120a \
  build/bench/ninfer_linear_op_bench 2>/dev/null \
  | rg 'STO_ENTRY.*(linear_|rowsplit)' | wc -l
/usr/local/cuda/bin/cuobjdump --dump-resource-usage \
  build/bench/ninfer_linear_op_bench 2>/dev/null | c++filt
```

Representative generated resources at this checkpoint:

| Kernel family / variant | Registers per thread | Static shared |
|---|---:|---:|
| Small-T Q4 TT4 / TT8 | 56 / 67 | 9,728 B |
| Small-T Q5 TT4 / TT8 | 58 / 71 | 11,776 B |
| Small-T Q6 TT4 / TT8 | 59 / 72 | 13,824 B |
| Small-T W8 TT4 / TT8 | 60 / 66 | 18,432 B |
| Q5 direct split2 / split4 | approximately 64 / 48 | no local memory |
| split-low4 default Q4 / Q5 | 152 / 154--156 | 46,592 / 47,616 B |
| split-low4 short Q4 / Q5 | 99--101 / 102--103 | 29,952 / 30,976 B |
| W8 general / small-M | 119--120 / 83--84 | 47,104 / 40,448 B |

The typed finalizer changed demangled template identities and removed one pointer parameter but did
not add policy instances. Future code-size comparisons must therefore use entry count and section
sizes together with normalized policy inventory rather than raw symbol-name equality.

E013 added `ninfer_linear_fusion_bench`, which compares the semantic wrapper against an explicit
`linear + residual_add` composition while restoring identical residual input and flushing L2 before
the timed CUDA events:

```bash
cmake --build build -j --target ninfer_linear_fusion_bench
for round in 1 2 3; do
  ./build/bench/ninfer_linear_fusion_bench \
    --shape all --path both \
    --t-sweep 1,6,16,17,32,64,128,129 \
    --warmup 8 --repeat 40 \
    --csv-out \
      profiles/bench/linear-architecture-20260716/fusion/linear_add_r${round}.csv
done
```

Each process reports CUDA-event medians, minima, and P95. The ledger uses the mean of the three
process medians below; positive delta means the fused wrapper is slower than the two-launch
semantic reference:

| Shape | T | Fused wrapper (us) | Two-Op reference (us) | Fused latency delta |
|---|---:|---:|---:|---:|
| Q5 `[5120,6144]` | 1 | 20.043 | 23.893 | -16.12% |
|  | 6 | 33.931 | 34.080 | -0.44% |
|  | 16 | 107.104 | 107.275 | -0.16% |
|  | 17 | 155.264 | 156.288 | -0.66% |
|  | 32 | 157.269 | 154.944 | +1.50% |
|  | 64 | 153.824 | 151.349 | +1.64% |
|  | 128 | 153.152 | 148.843 | +2.90% |
|  | 129 | 175.445 | 167.136 | +4.97% |
| Q5 `[5120,17408]` | 1 | 60.992 | 64.736 | -5.78% |
|  | 6 | 73.099 | 73.088 | +0.01% |
|  | 16 | 277.824 | 277.717 | +0.04% |
|  | 17 | 417.312 | 417.205 | +0.03% |
|  | 32 | 417.312 | 415.019 | +0.55% |
|  | 64 | 407.328 | 401.056 | +1.56% |
|  | 128 | 402.773 | 398.709 | +1.02% |
|  | 129 | 458.091 | 449.899 | +1.82% |

The direct Large-T store is distributed in scalar BF16 fragment order, whereas the standalone
residual Op can use contiguous vector loads/stores. This is a source-supported hypothesis for the
regression, not yet a profiler-backed conclusion. The next experiment must compare a collective
vectorized finalizer and materialized composition without weakening the exact LinearAdd rounding
contract.

E014 source audit and timing protocol:

```bash
for round in 1 2 3; do
  for spec in \
    'LmHead248320x5120 Q6' \
    'LmHeadDraft131072x5120 Q4' \
    'MlpDown5120x17408 Q5'
  do
    set -- $spec
    for candidate in auto smallt; do
      ./build/bench/ninfer_linear_op_bench \
        --shape "$1" --qtype "$2" --candidate "$candidate" \
        --t-sweep 1 --warmup 8 --repeat 40 \
        --stream-ceiling-gbs 1510.916
    done
  done
done
```

The comparison above is only a candidate screen because forced Small-T output was not checked
against an independent oracle at the exact head geometries. It is sufficient to falsify a mandatory
T1/source-mainloop boundary, but not to change production dispatch. The follow-up instantiates a
measurement-only TT1 specialization and validates Q4 heads at N=65,536/98,304/131,072 and the Q6
head at N=248,320 before any route change.

E015 builds a focused correctness matrix from the actual Vision geometry and E011 transition
sentinels rather than the historical benchmark Cartesian set:

```bash
cmake --build build -j --target ninfer_linear_test
/usr/bin/time -f 'elapsed=%e max_rss_kib=%M exit=%x' \
  env NINFER_LINEAR_TEST_VISION_CANDIDATES_ONLY=1 \
  ./build/tests/ninfer_linear_test
clang-format --dry-run --Werror tests/ops/test_linear.cpp
git diff --check
```

The suite invokes the same production launchers used by the forcing benchmark and does not alter
the production plan. It covers Q6 `[1152,1536]` T28/32; Q4 `[3456,1152]` T24/28; Q5
`[1152,1152]` T56/64; Q4 `[4304,1152]` T12/16; padded-K Q5 `[1152,4304]` T80/96; and W8
`[4608,4608]` plus `[5120,4608]` T5/6. For each point it checks `auto`, forced Small-T, and the
legal forced MMA backend. The current plan's `uses_tensor_cores` flag chooses only the auto
tolerance; it is not the oracle for the forced candidates.

E016 keeps TT1 entirely in the benchmark binary by directly instantiating the existing Small-T
kernel template; neither the library nor production dispatch changes. Each shape alternated
candidate order across three processes:

```bash
./build/bench/ninfer_linear_op_bench \
  --shape "$shape" --qtype "$qtype" --candidate "$candidate" \
  --t-sweep 1 --warmup 8 --repeat 40 --flush-mib 256 \
  --stream-ceiling-gbs 1510.916 --csv-out "$csv"
```

Raw evidence is under `profiles/bench/linear-architecture-20260716/tt1_candidate/`: 63 process
CSVs plus stdout, merged `all_runs.csv`, `resource_usage.txt`, full `sass.txt`, and isolated Q4
TT1/TT4 SASS. Means of the three cold medians are:

| Geometry | Auto (us) | TT4 (us) | TT1 (us) | Retained observation |
|---|---:|---:|---:|---|
| Q4 `[65536,5120]` | 130.688 | 128.331 | 131.797 | TT4 wins; TT1 is 2.70% slower than TT4 |
| Q4 `[98304,5120]` | 190.165 | 181.867 | 190.272 | TT4 wins; TT1 is 4.62% slower |
| Q4 `[131072,5120]` | 249.803 | 235.477 | 249.440 | TT4 wins; TT1 is 5.93% slower |
| Q6 `[248320,5120]` | 640.512 | 622.059 | 637.184 | TT4 wins; TT1 is 2.43% slower |
| W8 `[1024,5120]` | 15.968 | 15.968 | 15.968 | indistinguishable launch/timing floor |
| W8 `[5120,10240]` | 54.859 | 54.869 | 52.843 | TT1 wins by 3.69% over TT4 |
| W8 `[34816,5120]` | 136.896 | 137.056 | 137.493 | auto/TT4 retained |

The resource reduction is therefore not a sufficient performance selector. Geometry-specific
parallel work and CTA count dominate the vocabulary-head result; policy selection must use measured
shape behavior rather than occupancy or token-tile size alone.

E017 adds a measurement-only collective path to `ninfer_linear_fusion_bench`. After the complete
K reduction it rounds every projection fragment to BF16 in the reused `Bs[0]` shared allocation,
synchronizes the CTA, then assigns contiguous eight-row packs to threads. An explicit `int4` carrier
is required to preserve one `LDG.E.128` and one `STG.E.128`; the first typed-pack attempt compiled to
four 32-bit global loads/stores and was therefore superseded before the retained measurements.
Neither variant changes the mainloop, tile, grid, or semantic rounding order.

Correctness and timing commands:

```bash
cmake --build build -j 4 --target ninfer_linear_test ninfer_linear_fusion_bench
env NINFER_LINEAR_TEST_PREFILL_FUSIONS_ONLY=1 ./build/tests/ninfer_linear_test
for round in 1 2 3; do
  ./build/bench/ninfer_linear_fusion_bench \
    --shape all --path all --t-sweep 32,64,128,129 \
    --warmup 8 --repeat 40 \
    --csv-out profiles/bench/linear-architecture-20260716/fusion/\
linear_add_collective_vec_r${round}.csv
done
```

Means of three cold-process medians are:

| Shape | T | Collective (us) | Current fused (us) | vs fused | Two-launch ref (us) | vs ref |
|---|---:|---:|---:|---:|---:|---:|
| Q5 `[5120,6144]` | 32 | 151.136 | 157.237 | -3.88% | 154.997 | -2.49% |
|  | 64 | 147.733 | 153.707 | -3.89% | 150.805 | -2.04% |
|  | 128 | 146.891 | 154.987 | -5.22% | 150.613 | -2.47% |
|  | 129 | 165.387 | 175.467 | -5.74% | 167.200 | -1.08% |
| Q5 `[5120,17408]` | 32 | 411.979 | 419.243 | -1.73% | 416.981 | -1.20% |
|  | 64 | 399.328 | 407.755 | -2.07% | 401.440 | -0.53% |
|  | 128 | 399.573 | 404.853 | -1.30% | 401.909 | -0.58% |
|  | 129 | 451.904 | 459.648 | -1.68% | 452.075 | -0.04% |

The final linked resource comparison is exact for short edge/full: 103/102 registers, 30,976 B
shared, zero local/stack for both direct and collective. Default collective edge/full is 155/154
registers versus direct 156/154, with the same 47,616 B shared. Existing Q4/Q5/Q6 `StoreBf16` and
direct residual resources remain at the E012/E010 checkpoint.

NCU commands follow the E009 application-replay rule, with benchmark event timings under NCU
discarded:

```bash
ncu --force-overwrite --replay-mode application --set basic \
  --kernel-name-base function --kernel-name regex:linear_rowsplit_gemm_mma_kernel \
  --launch-skip 0 --launch-count 1 \
  -o profiles/ncu/linear-architecture-20260716/linear_add_out_t128/PATH_basic_vec \
  ./build/bench/ninfer_linear_fusion_bench \
  --shape Out5120x6144 --path PATH --t-sweep 128 --warmup 0 --repeat 1
```

`PATH` is `wrapper` or `collective`; retained reports use prefixes `direct_basic_vec` and
`collective_basic_vec`. Both launch the Q5 short-full kernel with grid `(80,2)`, block 128, 102
registers, 29.95 KiB static shared, and approximately 8.3% achieved occupancy. NCU reports 200.67
vs 193.18 us, 22.88% vs 23.95% SM SOL, and 6.65% vs 6.91% DRAM SOL; only the independent cold
benchmark supplies the speed claim.

Focused reports `direct_transactions_vec.ncu-rep` and `collective_transactions_vec.ncu-rep` add
global L1 sectors, executed global/shared load/store instructions, LSU/tensor instructions, and
duration. Tensor instructions remain 1,966,080. Global load/store warp instructions fall from
51,200/20,480 to 33,280/2,560; L1 global load/store sectors fall from 6,963,200/81,920 to
6,922,240/40,960. Shared load/store instructions rise only from 2,949,120/1,013,760 to
2,951,680/1,034,240. This supports instruction/output-ownership reduction, rather than reduced
weight traffic or a different MMA mainloop, as the cause of the retained gain.

E027 prevents over-promoting this result. Every Text layer calls both residual projections; one
layer pass therefore contains 64 Q5 `[5120,6144]`, 64 Q5 `[5120,17408]`, and 128 total LinearAdd
calls. Default prefill can present every integer T1--1024; ordinary verify is T1 and MTP verify is
T2--6. The two geometries select short BN64 MMA through T128 and default BN128 from T129. Short
edge/full boundaries are T63/64, 127/128; default full first executes at T256.

E017's performance points are T32/64/128/129. T17 has only exactness, T18--31 are absent, and no
default-full instance executes. More importantly, its two-launch reference calls auto `linear()`.
E025 shows that auto's current T17+ MMA choice is not the best single projection at T17--24; the
missing competitor is forced Small-T followed by residual add. A production decision therefore
needs four explicit identities where relevant: collective MMA, direct lane-owned MMA, forced MMA
plus residual, and forced Small-T plus residual. Collective must be BF16-word exact to the same MMA
projection plus residual; Small-T and MMA separately use their FP64 oracle/tolerance because their
accumulation orders need not match word-for-word.

The minimum default qualification includes every T17--31, short boundaries 63/64/65 and
127/128/129, default boundaries 255/256/257, and 1023/1024 for both geometries. Performance first
screens all T17--31 and every `m-1,m,m+1` tile boundary, then confirms reversals/near-ties in three
processes. NCU reuses E017 T128 and adds only retained or diagnostic candidates at Out T25,
Down T129, Out T256, and optionally Down T1024.

LinearAdd materialization needs `10,240*T` B. Current layout queries only configured C (at least
128), so it sees zero under the old threshold and happens to have enough unrelated activation/GDN
headroom for smaller materialized tails. The shared resolver must instead expose the maximum stage
scratch over reachable route endpoints such as 1/2/16/17/24/25 and tile boundaries; incidental
arena slack is not the contract.

E018 extends the forcing hook to paired W8 K/V without adding kernels: `smallt` launches the two
existing row-streaming contractions, `w8_mma` invokes the existing dual-projection kernel, and
`auto` remains the current product wrapper. The broad screen used:

```bash
./build/bench/ninfer_linear_op_bench \
  --shape "$shape" --qtype "$qtype" --candidate "$candidate" \
  --t-sweep 4,5,6,8,9,12,16,17,24,32 \
  --warmup 5 --repeat 20 --stream-ceiling-gbs 1510.916 \
  --csv-out "$csv"

./build/bench/ninfer_linear_op_bench \
  --shape MtpKV1024x5120 --qtype W8G32 --paired-kv \
  --candidate "$candidate" --t-sweep 56,57,58,59,60 \
  --warmup 8 --repeat 40 --stream-ceiling-gbs 1510.916 \
  --csv-out "$csv"
```

Artifacts are under `registered_crossovers/{screen,confirmed}/` and `paired_kv/` in the retained
benchmark profile directory. Confirmed means of three process medians at the two W8 single-policy
sentinels are:

| W8 geometry | Last Small-T point (Small / MMA us) | First MMA point (Small / MMA us) |
|---|---:|---:|
| `[14336,5120]` | T8: 108.160 / 167.200 | T9: 177.856 / 167.349 |
| `[34816,5120]` | T8: 220.800 / 331.317 | T9: 390.475 / 331.360 |
| `[5120,17408]` | T16: 294.539 / 335.467 | T17: 356.608 / 335.456 |
| `[5120,6144]` | T16: 101.920 / 122.464 | T17: 128.523 / 122.443 |
| `[6144,5120]` | T16: 97.920 / 104.032 | T17: 124.512 / 104.043 |

E008 already confirms W8 `[5120,10240]` at T16/17. For paired W8 `[1024,5120]x2`, three-process
means are 112.245 vs 118.325 us at T56 and 120.971 vs 118.368 us at T57; every T57--60 process
selects dual MMA. The one-process wide screen shows the consequence of the current wrapper rule:
at T17 auto/dual takes about 118.24 us while two Small-T launches take 60.80 us.

The preliminary low-bit controls are used only to preserve reachable ranges, not to claim exact
unreachable crossovers: Q4 `[6144,5120]`, `[4096,5120]`, and `[1024,5120]`, plus Q5
`[1024,5120]`, all strongly favor Small-T through T16. Q6 `[248320,5120]` Small-T takes
624.64/670.91/1383.71/1451.62 us at T2/4/6/8 versus approximately 1.80 ms for MMA.
Wrapper-owned grouped paths, not base Linear, own larger T for those row views.

E025 closes the Q5 gaps with a sequential 20-cold screen and three no-concurrency confirmation
processes per candidate, each using eight warmups and 40 cold samples. Mean process medians are:

| Geometry / T | Small-T (us) | MMA (us) | MMA relative to Small-T | Decision |
|---|---:|---:|---:|---|
| Q5 `[5120,17408]` T24 / T25 | 366.325 / 437.867 | 408.981 / 408.256 | +11.64% / -6.76% | switch 24/25 |
| Q5 `[5120,6144]` T24 / T25 | 136.597 / 165.131 | 149.227 / 149.099 | +9.25% / -9.71% | switch 24/25 |
| Q5 `[6144,5120]` T21 | 126.581 | 128.608 | +1.60% | Small-T |
| Q5 `[6144,5120]` T22 | 127.851 | 128.597 | +0.58% | Small-T near-tie |
| Q5 `[6144,5120]` T23 | 128.608 | 128.608 | 0.00% | exact tie |
| Q5 `[6144,5120]` T24 | 128.640 | 128.597 | -0.03% | practical tie |
| Q5 `[6144,5120]` T25 | 161.376 | 128.608 | -20.31% | MMA |
| Vision Q5 `[1152,1152]` P60 | 34.400 | 33.408 | -2.88% | MMA |
| Vision Q5 `[1152,4304,Kpad4352]` P84 / P88 / P92 | 110.144 / 112.224 / 123.808 | 112.245 / 112.224 / 112.235 | +1.91% / 0.00% / -9.35% | Small / tie / MMA |

The Text screens contain no reversal: both Q5 output geometries switch at T24/25, while
`[6144,5120]` has a T23--24 tie band rather than a statistically meaningful single threshold.
Vision conclusions apply only to reachable P multiples tested here. Raw screen, confirmation, and
summaries are under `profiles/bench/linear-architecture-20260716/route_gaps/`.

E019 adds an opt-in, registered-only numerical suite without changing production dispatch:

```bash
NINFER_LINEAR_TEST_REGISTERED_GAPS_ONLY=1 ./build/tests/ninfer_linear_test
```

The fixtures deliberately avoid the old 256-stride patterned-row degeneracy: weights and FP16
group scales use suite-private hashed row/group codes, inputs are rounded to BF16 before both GPU
and CPU evaluation, and the CPU oracle dequantizes the exact row-split storage into FP64 W@x. The
suite exercises the real W8 `[14336,5120]` parent as query/gate `[6144,5120]` row views at offsets
0/7168 and K/V `[1024,5120]` views at offsets 6144/13312. The K/V wrapper is checked at T1--6 and
T17; the observed E018 boundary is checked at T56 and T57 through both two forced Small-T launches
and the forced dual-projection MMA launcher. Q4 draft heads N=65,536/98,304/131,072 and Q6
N=248,320 are checked through current auto and forced TT4 Small-T policies.

All 32 comparisons pass. Small-T and row-view relative L2 spans `1.617e-3..1.671e-3`; paired
two-Small-T T56/57 spans `1.653e-3..1.657e-3`; paired dual MMA at T17/56/57 spans
`2.310e-3..2.326e-3`; and head auto/TT4 spans `1.658e-3..1.669e-3`. The suite completes in
3.21 s with 1,183,836 KiB maximum RSS. Raw output is retained at
`profiles/bench/linear-architecture-20260716/correctness/registered_gaps.txt`.

E020 freezes the end-to-end baseline before any production route is changed. It uses the exact
17,502,555,648-byte `out/qwen3_6_27b_rtx5090.ninfer` artifact, CUDA Graph replay, one warmup, and
three measured repetitions. The Text command is:

```bash
./build/bench/ninfer_bench \
  --weights out/qwen3_6_27b_rtx5090.ninfer \
  -p 128,512,1024 -n 32 --warmup 1 -r 3 --output json \
  --output-file profiles/bench/linear-architecture-20260716/end_to_end/baseline_text.json
```

It reports 1,638.051/3,282.751/3,464.100 prefill tok/s at P128/512/1024 and 77.584 output tok/s
at TG32. MTP is run with `--mtp-draft-tokens 5`, once with the full head and once with
`--lm-head-draft`, in three independent processes per configuration. Each process contributes the
median of its three output-token rates, and the table reports the median and range of those three
process medians:

| MTP TG32 proposal head | Median-of-medians (tok/s) | Process range | Per-repetition proposal trace |
|---|---:|---:|---|
| full | 72.984 | 72.844--73.097 | 13 rounds, 65 drafted, 18 accepted, 1 fallback |
| optimized draft | 68.333 | 68.119--68.338 | 14 rounds, 70 drafted, 15 accepted, 3 fallback |

One full-head repetition fell to 57.199 tok/s while its adjacent repetitions were 73.126 and
72.984; the declared robust aggregation keeps the raw observation but prevents it from defining a
regression threshold. Full and optimized head rates are not compared against each other because
their proposal traces differ. Reports are retained under
`profiles/bench/linear-architecture-20260716/end_to_end/baseline_{text,mtp_*}.json`.

Before changing the E018 paired-K/V route, E020 also adds combined prompt/decode cases whose prompt
lengths sit on different sides of the measured T56/57 boundary. Each cell is again the median of
three process medians, with the process range in parentheses:

| Case | Proposal head | Prefill tok/s | Decode output tok/s | Total seconds |
|---|---|---:|---:|---:|
| P32+G32 | full | 384.682 (384.016--386.278) | 103.209 (103.200--103.446) | 0.393510 |
| P32+G32 | optimized | 393.867 (393.782--394.203) | 109.005 (108.500--109.158) | 0.375103 |
| P128+G32 | full | 1511.228 (1505.785--1511.284) | 60.212 (59.734--60.274) | 0.616493 |
| P128+G32 | optimized | 1543.759 (1539.877--1547.006) | 63.809 (63.512--63.957) | 0.584304 |

P32 executes the currently misrouted dual-projection K/V policy, while P128 is a control where
dual MMA is the measured winner. The per-repetition proposal traces are stable within each case:
P32 uses eight rounds, 40 drafted, 20 accepted, and four fallbacks; P128 uses 16 rounds, 80 drafted,
15 accepted, and one fallback. Reports use prefixes `baseline_mtp_pg32x32_` and
`baseline_mtp_pg128x32_` in the same end-to-end directory.

Required baseline coverage:

- registered 27B Text, MTP, and Vision pure Linear domains;
- `linear_pair`, `LinearAdd`, and `LinearSwiGLU` domains that physically share current kernels;
- T1, every selected Small-T direct/fallback policy, Large-T short/default tiles, W8 small-M/general
  tiles, and full/edge variants where they change generated code;
- independent correctness at each backend rounding boundary;
- matched cold/warm microbenchmarks and current plan identity;
- ptxas/SASS resource snapshots for representative kernels;
- NCU only for representative or regressing policies, followed by an end-to-end check when launch
  composition or a dominant family changes.

## 6. Registered-domain inventory

E004 found an important difference between source-visible candidates and reachable product calls.
Ordinary decode builds a one-token verify request, and `target_verify` invokes the schedule with
`Phase::Verify`. The `Phase::Decode` schedule branches are therefore dead in the current product.
Their Q4/Q5 `[7168,5120]` parent projections and Q5 paired `[6144,5120]` implementation are not
baseline obligations merely because they have tuned kernels.

### 6.1 Reachable Text domains

A complete Text layer pass has 16 full-attention and 48 GDN layers. Direct semantic-Op consumers
are:

| Op | Format and `W[N,K]` | Calls per pass | Reachable T and current topology |
|---|---|---:|---|
| Linear | Q5 `[6144,5120]` | 48 GDN Z | T1/Small-T pure SIMT; Large-T low-bit MMA |
| Linear | Q6 `[248320,5120]` | one head | final prefill column T1; verify `C=1..6` |
| LinearAdd | Q5 `[5120,6144]` | 64 output projections | fused T1 and Large-T; pure Linear plus residual kernel for Small-T |
| LinearSwiGLU | Q4 `[34816,5120]` | 64 MLP gate/up | fused T1 and Large-T; pure Linear plus SiLU/mul for Small-T |
| LinearAdd | Q5 `[5120,17408]` | 64 MLP down | fused T1 and Large-T; pure Linear plus residual kernel for Small-T |

`AttnInputProj` and `GdnInputProj` also call base `linear()` for `T<=16`, exposing row views hidden
behind their outer contracts:

| Outer Op | Reachable pure-Linear views | Calls per pass | Large-T behavior |
|---|---|---:|---|
| AttnInputProj | Q4 `[6144,5120]`, Q5 `[6144,5120]`, Q4 `[1024,5120]`, Q5 `[1024,5120]` | 16 each; 64 total | separate grouped-input MMA |
| GdnInputProj | Q4 `[4096,5120]`, Q5 `[6144,5120]` | 48 each; 96 total | separate grouped-input MMA |

Consequently one T1 target verify enters base `linear()` 209 times. A `T=2..6` verify enters it 401
times because the 128 LinearAdd and 64 LinearSwiGLU Small-T fallbacks add 192 contractions. This
multiplicity makes Small-T dispatch and launch cost product-critical even though it is not a long
prompt regime.

### 6.2 Reachable MTP and Vision domains

MTP uses plain W8 Linear extensively rather than the Text fused semantic Ops:

| Op | Format and `W[N,K]` | Reachable T / note |
|---|---|---|
| Linear | W8 `[5120,10240]` | prompt chunks and rebuild `1..6` |
| Linear | W8 `[14336,5120]` | rebuild/autoregressive `1..6` |
| LinearPair | two W8 `[1024,5120]` views | prompt chunks; two Small-T calls or paired Large-T MMA |
| Linear | two W8 `[6144,5120]` views | final prompt Q and gate at T1 |
| Linear | W8 `[5120,6144]`, `[34816,5120]`, `[5120,17408]` | attention/MLP at `1..6` |
| Linear | Q4 `[131072,5120]` or Q6 `[248320,5120]` | draft or full proposal head at T1 |

Vision contributes 111 pure Linear calls per media encode and was absent from the initial
microbenchmark inventory:

| Format and `W[N,K]` | Calls | Dynamic T |
|---|---:|---|
| Q6 `[1152,1536]` | 1 | `P=4V`, up to 131072 |
| Q4 `[3456,1152]` | 27 | P |
| Q5 `[1152,1152]` | 27 | P |
| Q4 `[4304,1152]` | 27 | P |
| Q5 `[1152,4304]`, storage-padded K | 27 | P |
| W8 `[4608,4608]` | 1 | V, up to 32768 |
| W8 `[5120,4608]` | 1 | V |

### 6.3 Future-35B pressure matrix, not a support commitment

The accepted future inventory adds the following non-grouped contractions:

| Format and `W[N,K]` | Architectural pressure |
|---|---|
| W8 `[9216,2048]`, `[12288,2048]` | multi-output Text projections |
| BF16 `[64,2048]` | control finalizer |
| W8 `[2048,4096]` | both plain Linear and LinearAdd from one structural contraction |
| Q6 `[248320,2048]`, Q4 `[131072,2048]` | full/draft heads with a new K |
| BF16 `[257,2048]` | narrow router/shared-gate output |
| W8 `[1024,2048]`, `[2048,512]` | non-grouped shared-expert SwiGLU/down paths |
| current Vision shapes, with W8 `[2048,4608]` merger output | reuse across target widths |

Future routed-expert Q4/W8 `[1024,2048]` grouped LinearSwiGLU and Q5/Q6/W8 `[2048,512]`
grouped Linear remain outside this implementation. The design must leave an ownership seam for a
different grouped mainloop; it must not force grouped execution into the base-Linear Cartesian
product.

### 6.4 Coverage gaps exposed by the inventory

- E015 plus the complete Release suite cover independent numerical admissibility at all Vision
  route seams; E045 covers all seven views through T32768 for performance, E049 physically profiles
  one representative, and E047 closes one public media request. T32768 still has no independent
  host numerical oracle; the five encoder views' reachable T32769--131072 range is unmeasured.
- E019/E039/E040/E045 cover the exact W8 `[1024,5120]` pair from the T56/57 crossover through
  T2048, including real row offsets, exact topology equivalence, and matched route timing.
- E030/E033/E043 cover both registered LinearAdd views and LinearSwiGLU through every production
  route seam; E046 provides a same-session end-to-end causal ablation.
- E045 replaces the stale 17-row benchmark inventory with an explicit 24-row matrix matching
  `SupportSpec`; E050 later removes that duplicate authority and derives `--all-targets` directly
  from the table. Its shape-by-T sweep remains a stress matrix, not a declaration that every
  Cartesian point is reachable from the target schedule.
- Future W8 LinearAdd/SwiGLU, grouped expert mainloops, dense 35B schedules, and a general GDN
  split-K final-reducer seam remain unimplemented and unmeasured.

## 7. Fused-consumer classification

E006 rejects both extremes: fused consumers should reuse Linear implementation components, but
they are not all scalar epilogues. E017 further shows that semantic finalization and physical
output ownership must be separable. The evidence supports four narrow seams:

```text
CtaProblemMap       selects one or more physical weight/output jobs before the loop
AccumulatorTopology defines accumulator count, weight-row mapping, and co-resident fragments
FinalizationScope   chooses lane-direct or CTA-collective fragment-to-output ownership
Finalizer           runs only after the complete K reduction and owns exact semantic rounding
```

Only named production policies combine them. They are not public free-form template axes.

| Semantic Op/path | Required implementation shape | Why a scalar epilogue is insufficient |
|---|---|---|
| pure Linear | single problem, single projection, `StoreBf16` | baseline with no auxiliary state |
| LinearAdd | single projection, `ResidualAddBf16InPlace` | a true semantic finalizer: projection BF16 round, BF16 residual read, FP32 add, BF16 store; E017 requires a collective scope for the winning Large-T store |
| LinearSwiGLU | `SplitHalfPair` plus `SwiGluBf16` | gate/up accumulators and half-split weight rows must coexist before finalization |
| Q5 T1 `linear_pair` | `StaticCtaJobMap<2>` with one accumulator per CTA | grid/job selection, not a dual-accumulator loop |
| W8 Large-T `linear_pair` | `DualProjection` | two accumulator sets share one activation tile |
| grouped input | typed CTA job map and output route | job/output metadata is selected before the mainloop |
| split-K GDN control | finalizer in the final reducer | partial accumulators may not cross the semantic BF16 seam |

Current resource snapshots make the zero-overhead boundary concrete:

| Matched current instances | Registers, edge/full | Static shared | Observation |
|---|---:|---:|---|
| Q5 default Large-T plain vs residual | 156/154 for both | 47,616 B | current compile-time residual branch adds no resource cost |
| Q5 short Large-T plain vs residual | 103/102 for both | 30,976 B | same result for the short policy |
| Q4 default plain | 152/152 | 46,592 B | WN=32, four-warps policy |
| Q4 folded SwiGLU | 107/105 | 46,592 B | WN=16, eight-warps; topology changes the mainloop tile |
| Q4 grouped full vs matched plain full | 160 vs 152 | 46,592 B | grouped topology is +8 registers; descriptor-only cost is not isolated |
| Q5 grouped full vs matched plain full | 166 vs 154 | 47,616 B | grouped topology is +12 registers; carrying this state into plain is a material risk |
| W8 dual-projection KV pair | 105/104 | 43,008 B | a distinct shared-X, dual-accumulator policy |

All reported instances have zero stack and local memory in this snapshot. The current Q5 residual
kernels declare separate `__restrict__ residual` and `__restrict__ out` pointers but launch them
with the same address. A typed in-place finalizer must carry one read-write pointer, removing this
invalid alias promise rather than preserving the old signature.

The focused command was:

```bash
NINFER_LINEAR_TEST_PREFILL_FUSIONS_ONLY=1 ./build/tests/ninfer_linear_test
/usr/local/cuda/bin/cuobjdump --dump-resource-usage build/tests/ninfer_linear_test 2>/dev/null \
  | c++filt
```

It covered T=17, 128, and 129 grouped attention/GDN, folded SwiGLU, and residual paths; every focused
comparison printed `max_abs=0` and the run ended with `OK linear prefill fusion correctness`. It
does not cover fused T1 or the Small-T wrapper fallbacks.

Three vertical-slice experiments gate the seam:

1. **Implemented in E010.** The current residual boolean in Q5 T1 and Large-T is now typed
   `StoreBf16` / `ResidualAddBf16InPlace`. Loops, schedules, resources, and focused outputs are
   unchanged; the in-place alias now has one read-write pointer. Removing the duplicate kernel
   parameter also reduced constant-parameter storage by 8 B. Both real Q5 shapes now have
   BF16-word exact T1 direct and T6 fallback comparisons. E013 qualifies the fused T1 policies but
   rejects the assumption that the same direct Large-T finalizer is automatically faster than a
   vectorized second kernel.
2. **Implemented in E021.** The folded Large-T kernel now expresses `SplitHalfPair<MT>` plus
   `SwiGluBf16` without changing its mainloop, tiles, pipeline, or dispatch. All 11 tested outputs
   are BF16-word identical; raw edge/full SASS hashes are identical before and after; resources
   remain 107/105 registers, 46,592 B shared, 944 B constants, and zero local/stack. The seam is
   therefore codegen-zero-cost, but the current folded route loses to materialized composition at
   T17--128 and wins at T129, so an independent direct Small-T candidate and an exact Large-T
   crossover remain separate performance questions.
3. **Rejected/deferred after E006/E032.** The grouped-versus-plain comparison does not isolate a
   descriptor-only cost, but its +8/+12-register topology delta is enough to reject injecting a
   universal job map into plain policies. Grouped input retains its typed implementation; extract
   a helper only after a real second consumer proves the same lifecycle. W8 dual projection remains
   a separate pair topology.

Current Small-T materialization remains a valid policy: LinearAdd needs `10240*T` bytes and
LinearSwiGLU `69632*T` bytes before their second kernel. Removing that workspace requires a direct
candidate to win; it is not an architectural cleanliness requirement. FP32 split-K control
workspace belongs to reduction topology and cannot be erased by a finalizer.

E021's matched benchmark records wrapper route, launch count, materialized bytes, and an explicit
two-launch reference. Each value below is the mean of three process medians, with eight warmups and
40 cold samples per process; output restore and a 256 MiB L2 flush occur outside the CUDA events:

| T | Wrapper route | Wrapper / materialized reference (us) | Wrapper delta |
|---:|---|---:|---:|
| 1 | fused T1 warp pair | 71.029 / 75.765 | -6.25% |
| 2 | materialized Small-T | 85.451 / 85.344 | +0.13% |
| 6 | materialized Small-T | 200.704 / 200.843 | -0.07% |
| 8 | materialized Small-T | 210.944 / 210.272 | +0.32% |
| 9 | materialized Small-T | 380.864 / 381.664 | -0.21% |
| 16 | materialized Small-T | 396.523 / 395.925 | +0.15% |
| 17 | folded Large-T | 288.779 / 272.288 | +6.06% |
| 32 | folded Large-T | 290.805 / 274.432 | +5.97% |
| 64 | folded Large-T | 294.219 / 281.248 | +4.61% |
| 128 | folded Large-T | 294.827 / 288.587 | +2.16% |
| 129 | folded Large-T edge | 520.747 / 544.704 | -4.40% |

T2--16 is intentionally the same two-launch materialized implementation on both sides. Folded
SwiGLU changes the Large-T accumulator topology from the plain projection's four-warps/CTA and
152 registers to eight warps/CTA and 105--107 registers while keeping the same 46,592 B shared
allocation; fewer launches and fewer materialized bytes do not imply a faster schedule. Raw data,
word dumps, and before/after SASS are under
`profiles/bench/linear-architecture-20260716/swiglu/{baseline,post_apply,baseline_words,post_words_apply,baseline_sass,post_sass_final}/`.

E022 then screens every T=120--260 with no concurrent GPU work and confirms all observed route
boundaries in three processes (eight warmups and 40 cold samples each). Only final-prefixed data is
retained: `swiglu/crossover/final_dense_t120_260_screen.csv` and
`final_confirm_boundaries_r{1,2,3}.csv`; earlier files are preliminary and not evidence. Confirmed
means are:

| T | Folded (us) | Materialized (us) | Folded delta |
|---:|---:|---:|---:|
| 120 | 301.067 | 298.997 | +0.692% |
| 128 | 294.208 | 288.672 | +1.918% |
| 129 | 520.896 | 546.197 | -4.632% |
| 255 | 536.747 | 577.525 | -7.061% |
| 256 | 526.229 | 548.171 | -4.003% |
| 257 | 720.875 | 705.216 | +2.220% |
| 260 | 718.837 | 704.491 | +2.036% |

The folded launch is grid `(544,ceil(T/128))`, 256 threads, WN16, and one kernel. Materialized
Linear is grid `(544,ceil(T/128))`, 128 threads, WN32, followed by `silu_and_mul_dim0_split` grid
`(17,T)`. Both Linear paths use BN128. If `T%128==0`, the entire launch uses a full-tile instance;
otherwise even complete earlier y tiles run the edge instance. This explains the visible timing
discontinuities but not a monotonic route: one tile favors materialization, two favor folding, and
the measured beginning of the third favors materialization again. Full/edge alone also fails
(full T128 and T256 select different winners). No tile-parity rule may be inferred without further
measurements.

Materialization requires `69,632*T` bytes: 8,912,896 B at T128, 8,982,528 B at T129,
17,825,792 B at T256, and 17,895,424 B at T257. Therefore policy selection and
`linear_swiglu_workspace_bytes()` must consume the same plan; a launcher-only threshold is an
incorrect ownership boundary.

E026 establishes the route domain that this plan must cover. `linear_swiglu()` is called once per
each of 64 Text layers. Prefill chunks at configured C=1024 can be shortened by the prompt tail,
prefix-reuse suffix, or snapshot boundary, so the default product reaches every integer T=1..1024.
Ordinary decoding runs the one-column Verify path, and MTP target verification reaches T2--6;
the MTP proposal model materializes its own gate/up and does not call this wrapper. More generally,
the domain is every integer through `min(prefill_chunk,max_context)`, not a fixed universal 1024.

This exposes a second threshold-duplication bug: target layout currently asks the workspace helper
only at configured C. For a non-monotonic route it must instead evaluate the shared plan's maximum
scratch over all reachable `1..min(C,M)`. Materialized local MLP storage is gate/up
`69,632*T` B; with existing hidden and activation matrices the folded and materialized local peaks
are `45,056*T` and `114,688*T` B. The 64 layers reuse one scope. In the current 27B layout,
root plus GDN reaches `164,440*C` B while root plus a materialized MLP reaches `124,952*C` B, so at
C=1024 the already allocated 168,386,560-B global workspace remains GDN-dominated. This is useful
cost evidence, not permission to leave launch and sizing ownership split.

The minimum default qualification grid covers all eight BN128 bands with first edge, midpoint,
last edge, and full endpoint:

```text
17,64,127,128; 129,192,255,256; 257,320,383,384; 385,448,511,512;
513,576,639,640; 641,704,767,768; 769,832,895,896; 897,960,1023,1024
```

T1/2/8/9/16 remain low-T sentinels. A band with inconsistent winners or margin below 3% requires
an every-T dense sweep; sparse sentinels cannot prove every T optimal.

E017 qualifies one such direct candidate and refines the seam contract. `Finalizer::store(index)`
is sufficient for T1 and for codegen/semantic experiments, but not a complete Large-T interface.
The winning policy first maps distributed MMA fragments to a reused shared BF16 tile, then applies
the same residual finalizer over contiguous packs. This shared remap and barrier belong only to the
named collective policy; `StoreBf16` and lane-direct finalizers must not pay for them.

## 8. Format, mainloop, and T-regime boundary

E014 shows that the source already contains useful reuse, but its naming and ownership obscure the
real schedule boundaries. The desired refactor should expose those boundaries without turning hot
addresses or decode into runtime-generic objects.

| Responsibility | Closed compile-time owner | Must not absorb |
|---|---|---|
| numeric format | bits, group K, signed reconstruction/range, per-group FP16 multiplier semantics | plane byte offsets or lane-specific decode instructions |
| physical encoding | split-low4 or signed-byte plane packing, bytes/group, plane presence and ordering | K-loop lifetime or a runtime format tag |
| row-split layout | K128 padding, groups/row, row/plane/group base addressing | per-element hot-loop view construction |
| decode atom | path-specific lane/chunk ownership and exact FP32-pair, eight-value, or BF16-pair instruction sequence | an obligation to derive every format from one formula |
| schedule/pipeline | tile/slab K, TT, row ownership, stages, split-K, cache/load cadence, full/edge policy | caller/model role |
| mainloop | K-loop lifetime, activation traversal, accumulator reduction, output traversal | semantic auxiliary pointers that plain Linear does not use |

Only a finite set of named aliases may compose these pieces. `Q4/Q5/Q6/W8` decode atoms may remain
hand-specialized, and a row-split view should be resolved to row/tile bases before entering the hot
loop. Q4 wrappers need not grow an unused high-plane argument merely to make signatures uniform.

Observed lifecycle boundaries are:

- `RowStreamingSimt<TT>` is the existing shared staged Small-T body for all four formats;
- Q5 direct T2--6 is an explicit `DirectSplitK<2|4>` schedule, not a Small-T branch hidden inside
  the launcher;
- tuned T1 includes warp-row staged, per-row split-K, mixed row ownership, and paired-accumulator
  schedules; these may reuse atoms but cannot be forced into one lifecycle;
- Q4/Q5/Q6 split-low4 Large-T share an MMA lifecycle;
- W8 Large-T has a different scale-cache and decode cadence and should initially share only
  narrower MMA/finalizer vocabulary;
- folded SwiGLU and dual projection are accumulator topologies, not scalar finalizer choices.

T1 is therefore a problem regime with measured candidates rather than a mandatory implementation
family. Current candidates include `RowStreamingSimt<TT=1|4>`, tuned warp-row, per-row split-K, and
an exact custom topology. The existing TT4 screen is promising for vocabulary heads but loses
badly for several projections, so no production route changes until TT1 resources and an
independent exact-shape oracle are available.

A source-only H1 separation is accepted only if the final linked binary retains zero stack/local
memory, does not increase registers/shared/constant parameters, adds no runtime format branch or
per-group integer address work, and preserves normalized hot-loop `cp.async`, decode, FFMA,
`ldmatrix`, MMA, and barrier structure. Same-backend output must remain BF16-word exact. Three
interleaved cold processes are the timing gate; a repeatable median regression above 1% rejects or
shrinks the abstraction.

E023 satisfies that gate for the shared Small-T path. It adds closed `NumericFormat` and
`PhysicalEncoding` types, leaves the `RowSplitCodec<Numeric,Encoding>` primary template undefined,
and explicitly registers only Q4G64, Q5G64, Q6G64, and W8G32. `SmalltDecodeAtom` is likewise
specialized only for those four combinations. Existing concrete codec names remain stable, and
the four chunk decoders stay hand-written; only plane geometry and scalar-tail pair decode consume
the new compile-time seam. Four unused `load_group()` APIs are deleted. There is no production
dispatch, launcher, kernel-parameter, or runtime-tag change.

The linked result preserves all 32 generic, TT1, and Q5-direct Small-T resource records. TT4/TT8
registers and shared bytes remain Q4 56/67 and 9,728 B, Q5 58/71 and 11,776 B, Q6 59/72 and
13,824 B, and W8 60/66 and 18,432 B; every instance has 956 B constants and zero local/stack.
The 121,596-line extracted Small-T SASS is byte-identical before and after with SHA-256
`4f915f270866106b1087be371686dabd400d4bfa161a4e7b945e43951c59d5ee`; all other codec consumers
are also byte-identical (`c39dadb6c3b770f92f8f359e2f24db8d8e26e4dcaeefa564e6070f392a453f6f`).
The full Linear suite, including scalar-tail `[130,130]` cases, passes in 42.59 s with
11,151,044 KiB maximum RSS; the focused 42-case Vision suite also passes.

Three interleaved processes per before/after executable, 40 cold samples per process, show these
mean process medians:

| Format / geometry | T | Before (us) | After (us) | Delta |
|---|---:|---:|---:|---:|
| Q4 `[34816,5120]` | 4 / 8 | 87.669 / 206.219 | 87.680 / 205.749 | +0.012% / -0.228% |
| Q5 `[1152,4304]` | 4 / 8 | 22.005 / 38.507 | 21.995 / 38.528 | -0.049% / +0.055% |
| Q6 `[248320,5120]` | 4 / 8 | 675.925 / 1454.777 | 676.181 / 1454.753 | +0.038% / -0.002% |
| W8 `[34816,5120]` | 4 / 8 | 139.328 / 220.864 | 139.125 / 221.088 | -0.146% / +0.101% |

With identical SASS and a maximum absolute timing delta of 0.228%, the separation is accepted as
zero-cost. Artifacts are under `profiles/bench/linear-architecture-20260716/format_traits/`.

## 9. Structural support and route-plan boundary

E024 shows that the existing planner's abstraction is coarse in the wrong dimension. Its
`ShapeFamily` values encode caller roles, while the actual kernel legality and measured winner are
determined by a physical weight view, T, and semantic topology. The current implementation also
contains multiple sources of route truth:

| Current behavior | Failure exposed by evidence |
|---|---|
| `ShapeFamily` names MTP/attention/GDN/MLP/head roles | identical effective row views cannot share an identity without caller knowledge |
| `LinearRegime` reduces T to T1/Small/Large | E022 requires non-contiguous SwiGLU intervals and E018 gives a pair-specific T56/57 boundary |
| one `regime_threshold()` returns 16 | contradicted by every measured family except a subset of W8 single problems |
| `N>=65536,K=5120` means vocabulary head | admits arbitrary N and tuning-only 65,536/98,304 rows; the artifact registers only Q4 N=131,072 |
| generic low-bit fallthrough | turns test convenience into an unsupported product contract |
| `linear.cpp` changes regime after resolving the plan | reported plan identity is not necessarily the launched policy |
| Small-T launcher privately chooses Q5 direct T2--6 | generated schedule identity remains invisible to the planner |
| wrappers duplicate thresholds in launch and workspace helpers | a route change can under-allocate the captured arena |

The recommended host-only vocabulary is deliberately small:

```cpp
struct WeightViewSig {
    LinearFormat format;
    int32_t n;
    int32_t k;
    int32_t padded_k;
    PhysicalViewClass view;
};

struct TopologySig {
    ProblemMap problem_map;               // Single, StaticSet2/4
    AccumulatorTopology accumulators;     // Single, DualIndependent, SplitHalfPair
    FinalizerKind finalizer;               // Store, ResidualAdd, SwiGlu, ...
    FinalizationScope finalization_scope; // Lane, CTA collective, materialized Op
};

struct SupportSpec { ProblemSig problem; TokenSet reachable_tokens; };
struct RouteSpec   { ProblemSig problem; TokenSet tokens; PolicyId policy; WorkspaceKind ws; };
```

These are host descriptors, not hot-loop template axes. `SupportSpec` answers whether a product
problem is admitted; `RouteSpec` selects one already compiled policy within that support set. A
single runtime-N/K Small-T binary may therefore serve many exact support rows without generating a
template Cartesian product. Only concrete `PolicyId` instances compile CUDA. Full/edge predication
is a variant of an already selected schedule; it may not silently choose a competing mainloop.

Base `linear()`, `linear_pair()`, `linear_add()`, `linear_swiglu()`, and grouped-input Ops own
separate resolvers because their semantic topologies and workspace differ. They share the
signature vocabulary, not one universal runtime epilogue descriptor. In particular:

- an effective whole weight and row view with the same `(format,N,K,Kpad,alignment)` share a route;
- the Vision Q5 `[1152,4304]` view is distinct through `Kpad=4352`;
- exact Q4 draft product support is `[131072,5120,5120]`; N=65,536/98,304 remain benchmark-only;
- Q5 direct split2/split4, TT4/TT8 row streaming, dual projection, folded SwiGLU, and collective
  residual finalization are visible policies, not hidden branches under `SmallT/LargeT`;
- a route table may contain multiple disjoint `TokenSet` rows for the same problem;
- an unmeasured interval retains its current correct policy but is explicitly not described as
  optimized or roofline-qualified.

Validation first checks existing storage metadata, then constructs the exact physical signature.
Unsupported quantized signatures fail instead of falling back. A supported signature must have
exactly one covering route. T=0 may remain the current no-op after metadata validation. Arbitrary
shape numerical tests call internal forced launchers rather than widening the product Op contract.

The resolver remains CUDA-Graph safe: it is pure host code with no allocation, synchronization, or
autotuning; capture fixes the selected kernel and addresses, and replay does not run the resolver.
Workspace sizing and launch must call the same plan function before capture. Updating a route
therefore changes both the arena layout and captured launch sequence through one source of truth.

Future 35B admission adds exact rows and qualified policy mappings, not family inheritance.
`[131072,2048]` and `[131072,5120]` are naturally distinct keys; W8 `[2048,4096]` may have both
plain-store and residual-add topology rows. Routed-expert grouped contractions remain a different
semantic Op/mainloop. No placeholder 35B runtime or generic graph is needed to prove this fit.

E024 Phase 1 will first replace the host planner while preserving each retained launch choice.
Later changes can then move one evidence-qualified route at a time and compare exact plan identity,
correctness, resources, microbenchmarks, and end-to-end behavior.

E028 implements that cutover. The quantized registry contains 24 exact effective views and 72
routes covering T1, T2--16, and T17--`INT_MAX`. Dense BF16/FP32 keeps a deliberately separate
generic reference contract. Seven T1 structural policies retain existing tuned launchers; other T1
and T2--16 use row streaming; T17+ retains the Phase-1 split-low4 or W8 MMA choice. There is no
post-plan legality rewrite. The Q4/Q5 `[7168,5120]` base policies and dead Q5
`[6144,5120]x2` specialization are removed from dispatch; their storage parents remain because
effective row views/grouped Ops still consume them.

A CPU plan test locks the 24 views, unique route coverage at T1/2/16/17/`INT_MAX`, exact padded-K,
dense reference behavior, and rejection of Q4 65,536/98,304, Q6 65,536, Q4/Q5 7,168, arbitrary
low-bit shapes, and wrong Vision Kpad. Arbitrary-shape numerical tests now call explicit internal
forced launchers and no longer widen product support. The plan test, full Linear test (40.31 s,
11,151,408 KiB RSS), registered-gap suite, and Vision forced-candidate suite all pass. The linked
test still contains 89 matching Linear/row-split CUDA entries, and every representative
register/shared/local/stack resource remains at E012/E023 values.

Matched end-to-end MTP reports remain within 0.52% of the robust E020 median-of-medians across
TG32 and P32/P128+G32 full/optimized configurations. Text results need one more host-side
qualification:

| Text case | E020 rate | E028 three-process median | Rate delta | Mean-time delta |
|---|---:|---:|---:|---:|
| P128 | 1638.051 tok/s | 1616.698 | -1.30% | +1.032 ms |
| P512 | 3282.751 | 3262.888 | -0.61% | +0.950 ms |
| P1024 | 3464.100 | 3446.898 | -0.50% | +1.474 ms |
| TG32 CUDA Graph | 77.584 tok/s | 78.287 | +0.91% | -3.706 ms |

All hot kernel instances/resources are unchanged, and graph replay does not execute the resolver.
The roughly constant eager-prefill offset could include environment drift, but it also makes the
current implementation's 24-support then 72-route linear scan a falsifiable host-overhead
hypothesis. E029 will compare it with the old classifier and a support-indexed route span, then
remove the unnecessary full scan regardless of whether it explains the entire end-to-end offset.
Raw reports use `phase1_planner_*` in the E020 end-to-end directory.

E029 adds a CPU-only `ninfer_linear_plan_bench` whose 401-query row exactly mirrors one registered
Text pass: 16 full-attention layers contribute five base/fused projection resolutions each, 48 GDN
layers contribute four each, 64 MLP tails contribute two each, and the vocabulary head contributes
one. The executable directly compiles the planner and `ldd` confirms that it does not link CUDA or
`cudart`; a checksum prevents the compiler from deleting the resolution loop. It compares three
algorithms over P128/P512/P1024 query values:

1. the pre-E024 caller-role `ShapeFamily`/threshold classifier, reproduced faithfully for cost
   context only;
2. the frozen E028 24-support lookup followed by a scan of all 72 routes;
3. the production exact-support lookup followed by a compile-time-derived contiguous route span
   for only that support.

Across 5,000 passes and nine median samples, production before/after results are 47.568/7.910,
47.514/7.915, and 47.402/7.938 ns/query for P128/P512/P1024 respectively. The faithful old
classifier is 1.087--1.199 ns/query and the separately retained full scan is 45.624--45.862
ns/query. A single-core pinned confirmation with 10,000 passes and eleven medians reports
7.936/8.007/8.027 ns/query. Thus the exact representation has a real but small fixed host cost:
the optimized form estimates 1.65--1.68 us for 209 calls and 3.17--3.22 us for 401, versus about
9.9 and 19.0 us for E028. The entire old scan is still two orders of magnitude smaller than the
unmatched +0.95--1.47 ms observations, so it cannot explain them.

The production table does not hard-code the current three routes per support. A constexpr
`RouteSpan{first,count}` is derived independently for each support, and closure assertions prove
that every span is nonempty/in-bounds, contains all and only that support's routes, and that those
routes are contiguous. This keeps `SupportSpec` and `RouteSpec` semantically separate while making
lookup cost proportional only to one support's actual route count. The planner test, full Linear
test binary build, registered benchmark build, formatting, and diff checks pass. No GPU claim is
attached to this CPU-only change.

E030 replaces E017's incomplete LinearAdd comparison with five explicit paths in
`ninfer_linear_fusion_bench`: the current wrapper, current auto Linear plus residual, forced
Small-T plus residual, forced split-low4 MMA plus residual, and forced split-low4 MMA with the
CTA-collective finalization marker. Every timed sample restores the same residual image and flushes
256 MiB outside its CUDA events. CSV rows identify projection policy/tile/fullness, grid/block,
finalization scope, launch count, materialized bytes, and required candidate workspace rather than
reporting all fused variants as one opaque route.

The validation matrix covers both registered Q5 geometries at 41 T sentinels: every T17--31, then
63/64/65, 127/128/129, and each analogous BN128 boundary through 1023/1024. The collective result
is compared with forced MMA followed by the independent BF16x8 residual kernel because those two
paths must have the same MMA accumulation and semantic rounding order. All 82 full-output checks
have zero BF16-word mismatches. Small-T is not required to be word-identical to MMA; its existing
independent numerical contract and the explicit residual composition are the correctness oracle.

A 5-warmup/20-repeat screen is followed by three independent processes with eight warmups and 40
cold samples. The exact route transition is the same for both shapes:

| Shape / T | Selected path, three process medians (us) | Nearest competing path, three process medians (us) |
|---|---:|---:|
| `[5120,6144]`, T24 | Small-T + residual: 140.576 / 140.608 / 140.544 | collective: 150.976 / 150.848 / 150.848 |
| `[5120,17408]`, T24 | Small-T + residual: 370.144 / 370.752 / 371.872 | collective: 409.024 / 411.008 / 411.008 |
| `[5120,6144]`, T25 | collective: 149.824 / 150.912 / 151.040 | MMA + residual: 152.928 / 154.944 / 154.912 |
| `[5120,17408]`, T25 | collective: 409.216 / 409.184 / 410.944 | MMA + residual: 413.248 / 413.280 / 414.816 |
| `[5120,6144]`, T1024 | collective: 333.472 / 335.200 / 335.232 | MMA + residual: 340.896 / 341.312 / 341.344 |
| `[5120,17408]`, T1024 | collective: 925.504 / 933.248 / 927.072 | MMA + residual: 929.088 / 937.248 / 930.752 |

The cross-process median selects collective at every screened T25--1024. The down projection's
high-T margin is often below 1%: one process each at Out T64, Down T64, and Down T1023 slightly
favors separate MMA plus residual, while the other processes and aggregate retain collective.
That uncertainty is recorded rather than inflated into a universal speedup claim. Collective is
still the recommended registered-domain choice because it is never the aggregate loser, removes
one launch and all materialization, and has the E017 transaction explanation; the recommendation
does not extend past T1024.

Forced Small-T uses grid `(640,ceil(T/8))`, 256 threads, then one residual kernel, and materializes
`10240*T` bytes. Forced MMA/materialized uses BN64 through T128 and BN128 after it, grid
`(80,ceil(T/BN))`, 128 threads, then the same residual kernel and scratch. Collective uses the
identical selected MMA tile/grid but performs CTA-scoped finalization in one launch with no
intermediate. The production policy should therefore be T1's already-qualified tuned fused path,
materialized Small-T for T2--24, and collective MMA for T25--1024. A capacity query for maximum T
must return the maximum exact-route scratch over the reachable range, not merely the scratch of the
route chosen at that maximum. Full process medians/ranges and geometry are retained in
`profiles/bench/linear-architecture-20260716/linear_add_routes/confirm_exact_summary.csv`.

E031 completes the structural half of the base-Linear planner before changing measured
crossovers. The 24 exact physical supports now expand into 83 routes, whose lengths vary by the
number of schedules actually needed by that support. Seventeen policy IDs include seven tuned T1
instances, two dense reference paths, and eight reusable structural schedules:

- warp-per-row TT4 and TT8;
- Q5 direct split2 and split4 for exact T2--6 instances;
- split-low4 BF16 MMA with BN64 and BN128 configurations;
- W8G32 BF16 MMA with small-M and general configurations.

`linear()` switches on the final policy and calls a fixed launcher. It no longer asks a broad
Small-T or MMA launcher to reclassify shape/T and choose a competing schedule. The same policy may
serve multiple exact support signatures when its lifecycle is identical: for example Q5 direct
split2 covers both registered K6144 and K17408 instances, while the support signature and fixed
launcher legality select the appropriate compile-time full-slab specialization. Full versus edge
predication remains inside a fixed policy because it changes only uniform bounds handling, not the
mainloop/tile decision. Broad auto launchers remain explicitly measurement-only.

Compile-time closure now proves that every support owns one contiguous, nonempty route span; its
routes start at T1, end at `INT32_MAX`, and have neither overlap nor gaps. Runtime tests lock the
83-row count and every relevant T1/2/4/5/6/7/16/17/64/65/128/129 boundary. The unsupported Q5
N7168 direct branch is removed. Planner tests pass, all Linear/fusion benchmark and test targets
build, and formatting/diff checks pass. The complete GPU Linear test then reports `OK linear
correctness` in 40.87 s with 11,149,516 KiB maximum RSS, covering fixed production entries as well
as retained internal forcing paths. Because the route table deliberately reproduces the old
launcher choices, this is a code-structure experiment only; measured crossover changes follow as
separate experiments.

E032 tests H4 at the actual lifecycle boundary rather than treating two kernels with the same MMA
instruction as one mainloop. Split-low4 and W8 both convert raw row-split weights into swizzled
BF16 `As`, stage BF16 activations in `Bs`, use BK64, the same XOR swizzle and lane coordinates,
load `m16n8k16` fragments, accumulate with BF16 MMA into FP32, and expose the same direct C-fragment
layout. Those facts justify common tile vocabulary and a common consumer atom, but the raw producer
owns substantially more than a leaf `decode()` function:

| Lifecycle fact | Q4/Q5/Q6 split-low4 | W8G32 | Consequence |
|---|---|---|---|
| Raw storage | two-stage `Cr/Hr/Sr` | single `Cr`; scale slab persists across four K tiles | different shared lifetime |
| Group/scale cadence | one G64 scale per BK64 | two G32 scales per BK64, loaded eight at a time | different producer state |
| Pipeline lead | pre-issue two full raw+x stages; `cp_wait<1>` | pre-issue tile zero; `cp_wait<0>` | different wait contract |
| Next prefetch | after current MMA into recycled stage | after dequant and before current MMA over single raw buffer | different barrier/commit order |
| Dequant ownership | one warp/row with optional high plane | half warp/row plus scale shuffle | different warp mapping |
| Fragment schedule | default serial; short config ping-pong | always ping-pong; outer loop unroll four | different register/ILP policy |
| Default warp tile | WM64 x WN32, four warps | WM64 x WN16, eight warps | different residency target |

The difference is resource-critical. Split-low4 default Q4/Q5/Q6 use 46,592/47,616/48,640 B
static shared and approximately 152--156 registers; W8 general uses 47,104 B and 119--120
registers, while its historical qualification depends on two CTA/SM residency. Giving W8 a second
raw BK64 code stage alone adds 4,096 B, and staging its scale slab similarly adds another 1,024 B.
Conversely, adopting W8's single-stage timing for split-low4 removes its measured load lead. The
full SASS schedules also differ materially: current default split-low4 Store instances contain
roughly 752--800 static instructions, 64 HMMA, 32 LDSM, and three barriers; W8 general contains
1,264, 160 HMMA, 120 LDSM, and ten barriers because of its different compiler unroll and cadence.

Therefore a supposed common lifecycle parameterized by `raw_stage_count`, prefetch placement,
wait distance, scale-cache period, barrier hooks, and fragment schedule would move the complete
schedule into producer callbacks. It would be a small scheduling DSL, not a useful abstraction,
while risking unused pointer state, barriers, shared storage, and template Cartesian growth. The
recommended zero-codegen experiment is intentionally narrower: extract one Linear-local
`rowsplit_mma_swizzle64` and two closed `consume_bk64` variants (`SerialFragments` and
`PingPongFragments`) that begin only after `As` and one `Bs` stage are ready. Before/after gates are
per-instance raw SASS equality, exact resources/constants, entry/section diagnostics, word-exact
outputs, and matched full/edge timings. Grouped input, folded SwiGLU, and dual W8 may consume this
fragment helper while retaining their own problem maps and lifecycles. CTA-collective residual is
a post-reduction finalization helper and likewise does not justify merging weight pipelines.

E033 then completes the E022 SwiGLU domain rather than extrapolating from T260. The first screen
runs both folded and materialized paths at every T17--1024 with five warmups and 20 cold samples.
The first five BN128 tile-count bands have an exact alternating sign. Bands six and eight contain
many sub-1% point reversals in the screen, so 43 boundary, midpoint, reversal, and endpoint values
are repeated in three independent processes with eight warmups and 40 cold samples:

| Reachable T | `ceil(T/128)` | Confirmed choice | Representative folded delta versus materialized |
|---:|---:|---|---:|
| 17--128 | 1 | materialized | T17 +6.03%; T64 +4.30%; T128 +2.78% |
| 129--256 | 2 | folded | T129 -4.46%; T192 -5.48%; T256 -3.99% |
| 257--384 | 3 | materialized | T257 +2.31%; T320 +2.00%; T384 +3.02% |
| 385--512 | 4 | folded | T385 -3.15%; T448 -3.68%; T512 -1.53% |
| 513--640 | 5 | materialized | T513 +1.61%; T576 +1.07%; T640 +2.45% |
| 641--768 | 6 | retain folded | mostly within +/-1%; T704 -0.76%, T767 -1.67%, T768 +0.74% |
| 769--896 | 7 | folded | T769 -3.44%; T832 -3.70%; T896 -2.35% |
| 897--1024 | 8 | retain folded | T897 -0.96% through T1023 -1.85%; T1024 practical tie (+0.10%) |

Positive deltas mean folded is slower. Every representative in bands one through five has the
same sign in all three processes. In band six, early points are noisy below 1%, but T704--767
consistently favors folded; changing individual near-ties would add unstable routes. In band eight,
the 20-sample screen's isolated reversals disappear under confirmation, and only full T1024 remains
a 0.10% aggregate tie with one folded-winning and two materialized-winning processes. Retaining
folded for bands six through eight preserves one launch and zero scratch without claiming a
measurable speedup at the tie.

Together with E021's fused T1 and existing materialized T2--16 route, the recommended registered
policy is therefore T1 fused; T2--128 materialized; T129--256 folded; T257--384 materialized;
T385--512 folded; T513--640 materialized; and T641--1024 folded. Materialization is `69632*T` B.
For a maximum token capacity C, the range-max workspace is the greatest materialized point not
exceeding C; at C>=640 it saturates at `69632*640 = 44,564,480` B even though the exact T1024 route
uses zero. Raw full-domain and three confirmation CSVs are retained in
`profiles/bench/linear-architecture-20260716/swiglu/full_domain/`.

E034 turns E030 and E033 into two closed production policy spaces rather than adding an epilogue
tag to the base Linear plan. `LinearAddPlan` has eight exact rows over its two Q5 geometries:
tuned residual T1, materialized base Linear plus `residual_add` at T2--24, fixed BN64
CTA-collective residual at T25--128, and fixed BN128 collective residual at T129 through the
quantized execution ceiling. `LinearSwiGluPlan` has the seven measured ranges selected in E033 and
retains folded BN128 as its terminal correctness route. Both reject nonpositive T, T above
8,388,480, and every non-registered physical view. Performance is qualified only through T1024.
Their wrappers switch directly on the final semantic policy; there is no downstream choice among
competing schedules.

Workspace is part of those plans. An exact materialized LinearAdd route consumes `10240*T` B and
an exact materialized SwiGLU route consumes `69632*T` B; fused/collective routes consume zero.
Capacity queries first validate exact support, intersect each finite route with `[1,C]`, and
evaluate only the reached route endpoints. Consequently LinearAdd saturates at
`10240*24 = 245,760` B for `C>=24`, while SwiGLU saturates at
`69632*640 = 44,564,480` B for `C>=640`. The target workspace layout calls these O(route-count)
queries inside the same stage scopes that own the operation scratch.

Compile-time closure and CPU tests cover every range boundary, rejection, exact scratch value, and
range maximum. Focused production-wrapper GPU verification used:

```bash
/usr/bin/time -v env NINFER_LINEAR_TEST_PREFILL_FUSIONS_ONLY=1 \
  ./build/tests/ninfer_linear_test
```

It exited zero in 1.52 s with 659,124 KiB maximum RSS. Both Q5 LinearAdd shapes at
T1/6/24/25/128/129/1024 have zero BF16-word mismatches against their materialized references.
SwiGLU at T1/2/128/129/256/257/384/385/512/513/640/641/1024 has `max_abs=max_rel=0`; the older
collective validation sentinels at T17/32/64/128/129 also remain word-exact. This demonstrates the
intended epilogue boundary: the typed finalizer and accumulator topology are reusable, but the
semantic Op owns whether a given token band is fused, collective, or materialized.

E035 applies the already-qualified base-Linear boundaries without inventing new kernel families.
The 24 exact support signatures retain the fixed TT4, TT8, Q5 direct-split, split-low4 BN64/BN128,
and W8 small-M/general policy vocabulary introduced in E031; only their finite token ranges change
to the E008/E011/E018/E025 winners. Because `linear_pair` has a different two-output launch
economy, it now owns a separate three-row plan rather than making two calls through the base plan:

| Pair token range | Fixed production policy | Workspace |
|---:|---|---:|
| 1--4 | two W8G32 TT4 launches | 0 |
| 5--56 | two W8G32 TT8 launches | 0 |
| 57--8,388,480 | one dual-W8 MMA launch | 0 |

Only two identical W8G32 row-split `[1024,5120,Kpad5120]` views are accepted. Non-identical views,
the unreachable historical Q5 pair, nonpositive T, and T>8,388,480 are rejected before launch. T
through 1024 has route-performance evidence; the larger range retains the same legal BN128
schedule as a correctness fallback. This
also removes the former accidental behavior in which the base W8 threshold selected two MMA
launches from T17 even though E018 measured TT8 as the pair winner through T56. The CPU structural
test covers every base crossover endpoint plus pair T1/4/5/56/57/1024/2048/max and rejection
sentinels;
`ninfer_linear_plan_test` reports `OK linear structural plan`, both affected test targets build,
formatting passes, and `git diff --check` passes. Focused GPU correctness and end-to-end MTP timing
remain explicit follow-up gates rather than being inferred from plan closure.

E036 adds a deliberately measurement-only future-35B geometry set to
`ninfer_linear_op_bench`:

| Format | Exact benchmark geometries |
|---|---|
| W8G32 | `[9216,2048]`, `[12288,2048]`, `[2048,4096]`, `[1024,2048]`, `[2048,512]`, `[2048,4608]` |
| Q4G64 | draft head `[131072,2048]` |
| Q6G64 | full head `[248320,2048]` |
| dense BF16 | `[64,2048]`, `[257,2048]` |

The eight quantized shapes reject `auto` and require an explicit measurement candidate
(`smallt`, `smallt_tt1`, and the appropriate W8 or split-low4 MMA path). Dense BF16 accepts only
`auto` and continues through the generic contiguous reference contract. Exact qtype mismatch is
rejected while parsing; `W8` is only a CLI spelling alias for `W8G32`. None of these names joins
`--all-targets`, `SupportSpec`, a target binder, or an Op contract. With CUDA hidden, all ten names,
all candidate restrictions, and qtype mismatch paths pass their parser checks; the benchmark,
help, formatting, and diff checks pass. The largest planned run is the Q6 head at T1024, estimated
at about 1,188 MiB of target data including a 378.906 MiB weight payload, so the complete matrix is
comfortably inside the local GPU capacity. GPU crossover evidence is the next step; this entry
records the admission boundary before those measurements exist.

E037 rechecks the plan against public reachability rather than treating the default
`prefill_chunk=1024` as an Op contract. `EngineOptions::prefill_chunk` is configurable, and the MTP
prompt path passes a complete prefill chunk into W8 pair, LinearAdd, and LinearSwiGLU. Therefore
rejecting T1025 inside those Ops would have regressed an existing Engine configuration. At the
other extreme, declaring every positive int32 T legal was also false: every terminal quantized
route uses BN128 and CUDA `grid.y` is limited to 65,535. The shared exact execution envelope is now
`1 <= T <= 128*65535 = 8,388,480`; target option validation and `TextContext` cap prefill chunks at
the same value. Dense reference Linear remains a separate generic contract. The routes above 1024
are legal conservative terminal schedules and carry no roofline claim.

The same review traces fixed-launch alignment. Generic TT4/TT8 could silently use a scalar tail on
an unaligned x, but Q5 direct T2--6 requires its full uint4 slabs, split-low4/W8 MMA use
`cp.async<16>`, tuned kernels use vector staging, and the CTA-collective residual finalizer uses
BF16x8 residual packs. All current target workspace/persistent/vision tensors are allocated at
256-byte boundaries; every column slice preserves that alignment because registered K is a
multiple of eight BF16 elements. Artifact planes start at 256-byte boundaries, and every exact
row-view offset preserves at least 16-byte plane alignment. Address class therefore does not
belong in the plan. A shared host legality helper now requires every nonempty Linear-family tensor
base and nonnull weight-plane base to be 16-byte aligned in base Linear, pair, LinearAdd,
LinearSwiGLU, attention input projection, and GDN input projection. This is an explicit finite
execution contract, not a hidden fallback or defensive requirement for untrusted artifacts.

Finally, whole-repository reachability proves `Phase::Decode` had no caller: eager and captured
ordinary decode both call `target_verify`, which fixes `Phase::Verify`; MTP uses the same target
verify route; prefill fixes `Phase::Prefill`. The stale branches were the only users of Q4/Q5
`[7168,5120]` parent projections and the historical Q5 `[6144,5120]` pair. The branches, enum value,
and redundant full-attention weight aliases are deleted rather than re-admitting dead shapes.
This closes support-table ownership: live row views remain registered, parent storage remains
bound for those views, and no unreachable caller can contradict the exact planner.

E038 closes the first whole-suite GPU regression after E034--E037 rather than inferring safety
from planner tests and focused fusion checks. A later provenance audit in E043 found that this
binary was built with `CMAKE_BUILD_TYPE=Debug`. Its numerical result remains valid, but every
elapsed-time/resource-performance interpretation in E038--E042 is superseded. The exact command
was:

```bash
/usr/bin/time -v ./build/tests/ninfer_linear_test
```

It exited zero with `OK linear correctness`, 11:09.23 wall time, 1,595.21 s user time, 15.11 s
system time, and 11,154,104 KiB maximum RSS. The suite covers the production base, LinearAdd, and
LinearSwiGLU planners; fixed Small-T and Large-T launchers; registered and generic dense/low-bit
numerics; alignment rejection; fused accumulator paths; and Large-T/tail sentinels through T2048
on small geometries. Every new LinearAdd BF16-word check and every SwiGLU boundary remains exact;
the accepted tensor-core comparisons remain within the predeclared `rel_l2 <= 4e-3` tail contract.
This run is deliberately not counted as product-pair closure: the exact MTP W8 row-view suite is
behind `NINFER_LINEAR_TEST_REGISTERED_GAPS_ONLY`, and its public `linear_pair()` coverage stopped at
T57. Audit of that opt-in suite therefore led to a separate test change: retain the FP64 packed-row
oracle through T57, extend only the BF16 input fixture through T1024, and at T128/T1024 compare the
public paired wrapper word-for-word with two public `linear()` calls on the exact
`[1024,5120]` K/V row views. This avoids a prohibitive O(N*K*T) host oracle while directly testing
the production pair planner, dual-output topology, exact row offsets, and output-tile tails. That
new coverage is pending its rebuilt focused GPU run and is not included in the E038 pass.

E039 runs that rebuilt pair coverage with the exact command below. This was also a Debug build;
only the exact comparison is retained, and its 9.86 s elapsed time is not performance evidence.

```bash
NINFER_LINEAR_TEST_REGISTERED_GAPS_ONLY=1 \
  /usr/bin/time -v ./build/tests/ninfer_linear_test
```

The CPU structural tests first pass with the additional T128 planner sentinel. The GPU suite then
exits zero with `OK linear registered-gap correctness` in 9.86 s and 1,185,248 KiB maximum RSS.
The exact W8 parent `[14336,5120]` row views retain their FP64-oracle checks at T1--6, T17, T56,
and T57, including both forced topologies at the 56/57 crossover. The new public-wrapper controls
report zero BF16-word mismatches for K and V independently at both T128 and T1024:

```text
pair-vs-single T=128  K=0 V=0 mismatches
pair-vs-single T=1024 K=0 V=0 mismatches
```

Thus the T57--1024 production dual-W8 route is no longer justified only by a small synthetic
geometry or planner closure. It has exact product row offsets, public admission, fixed-route
dispatch, large output grids, and dual-versus-single numerical equivalence. The comparison is a
topology-equivalence oracle rather than an independent real-number oracle; independence remains
provided at the route crossover by the packed-row FP64 checks.

E040 separates the quantized CUDA launch envelope from the previously measured T<=1024 domain.
The focused product-geometry suites add T1025 and T2048 without changing any production route.
Both were Debug executions: launch and numerical coverage survives, while the printed elapsed
times are invalid as performance evidence.

```bash
NINFER_LINEAR_TEST_PREFILL_FUSIONS_ONLY=1 \
  /usr/bin/time -v ./build/tests/ninfer_linear_test
NINFER_LINEAR_TEST_REGISTERED_GAPS_ONLY=1 \
  /usr/bin/time -v ./build/tests/ninfer_linear_test
```

The fusion suite exits zero in 26.99 s with 991,768 KiB maximum RSS. Both Q5 LinearAdd geometries
are BF16-word exact against `linear + residual_add` at T1025/T2048, and folded Q4 SwiGLU is exactly
equal to materialized `linear + silu_mul` at both points. The MTP suite exits zero in 10.56 s with
1,188,832 KiB maximum RSS; paired K/V versus two public single projections has zero BF16-word
mismatches at T1025 and T2048. Existing small-geometry FP64 tests already cover generic T2048.
These results promote `[1,2048]` from launch arithmetic to measured functional coverage for the
affected Add/SwiGLU/pair routes and their exact underlying Q5/Q4/W8 views; they do not cover all 24
base supports at T2048. T2049 through 8,388,480 remains a legal fixed-BN128 fallback by
grid proof only; nothing above T1024 is performance-qualified or roofline-qualified.

E041 challenges whether future 35B geometries require a new abstraction or only new measured
route data. **All timing values and crossovers in this entry are invalidated by E043's Debug-build
finding and are retained only as a failed experiment record. E044 is the Release replacement.**
The benchmark gains six measurement-only selectors that call already-linked fixed
launchers directly: TT4, TT8, split-low4 BN64/BN128, and W8 small-M/general. Their legality checks
lock format, K/Kpad, W8 scale-load alignment, and each fixed CUDA `grid.y` envelope. CSV rows name
the actual `LinearPolicyId`; the selectors do not join `SupportSpec`, any target binder, or product
dispatch. This was necessary because the earlier broad candidates hid exactly the schedule choice
the experiment was meant to test.

The screen retains 3,462 cold/warm rows under
`profiles/bench/linear-architecture-20260716/future35/screen/`, plus 26 dense-reference rows. Every
quantized shape compares TT1 at T1, TT4/TT8 over the relevant low-T interval, both MMA schedules at
tile edges through T1024, and Vision merger MMA through T32768. Five warmups and twenty cold
samples use the frozen 1,510.916 GB/s copy ceiling; cold medians choose routes. All reversal,
sub-2%, and tile-boundary cases are then repeated in three independent processes with eight
warmups and forty cold samples, alternating candidate order. The 336 confirmation rows are under
`future35/confirm/run{1,2,3}/`; reported values below are median-of-process-medians.

The low-T W8 envelope is shape-local even though its vocabulary is closed:

| Future geometry | Confirmed low-T envelope | Representative boundary evidence |
|---|---|---|
| `[1024,2048]` shared gate/up | TT1 T1; TT4 T2--90; small-M MMA from T91 in the measured interval | T90 31.968 vs 32.000 us; T91--94 practical ties at 32.000 us |
| `[2048,4096]` output | TT4 through T54; small-M from T55 | T54 62.400 vs 62.720 us; T55 62.752 vs 62.688 us |
| `[9216,2048]` attention input | TT4 retained through T15; small-M from T16 | T13--15 exact practical ties at 46.336 us; T16 48.192 vs 46.336 us |
| `[12288,2048]` GDN input | TT4 through T16; general MMA from T17 | T16 60.672 vs 66.816 us; T17 68.608 vs 66.816 us |
| `[2048,512]` shared down | small-M MMA from T1 | T1 11.520 us versus TT1 13.536 us and general 13.568 us |
| `[2048,4608]` Vision merger | TT1 T1; TT4 T2--14; small-M from T15 | T14 66.784 vs 68.864 us; T15 72.608 vs 68.864 us |

TT8 is never the winner for these six new geometries, but that is measured data, not a reason to
remove the policy from the current registered target. The W8 MMA configuration is also not a
single monotonic `N` or T threshold. `[9216,2048]` small-M/general tie at T127--128, general wins
T129 and T256, then small-M wins T257 because the CTA-wave geometry changes. `[12288,2048]`
general wins T128, small-M wins T129 by 3.8%, and general wins again by T255. The three N2048
geometries favor small-M at T768 (or tie for K512) and general at T896. A future target therefore
needs its own finite contiguous route ranges; a hidden `N<=1024` or one global crossover would be
wrong, while the current support-local route-span representation expresses the result without a
new dispatch mechanism.

Both K2048 vocabulary heads produce the same stable three-regime envelope:

| Geometry | T1--8 | T9--64 | T65--1024 |
|---|---|---|---|
| Q4 `[131072,2048]` | TT4 | split-low4 BN64 | split-low4 BN128 |
| Q6 `[248320,2048]` | TT4 | split-low4 BN64 | split-low4 BN128 |

At T1, TT4 beats TT1 by 0.4% for Q4 and 2.2% for Q6 in confirmation. At T8, Q4/Q6 TT4 are
234.688/543.360 us versus BN64 295.904/563.840 us; at T9 BN64 wins 294.176/564.512 us versus
328.896/783.520 us. BN64 wins T64 at 283.936/595.232 us, while BN128 wins T65 at
378.112/690.880 us versus BN64 523.296/1,102.750 us. Thus the existing closed codec, Small-T
mainloop, and two split-low4 schedules already span the new N/K geometry; admission would add
support and measured ranges, not a generic loop or new format abstraction.

Two limits remain explicit. Dense BF16 `[64,2048]` and `[257,2048]` still use the correctness
reference path: at T2048 they reach only 2.57 and 3.17 useful TFLOP/s, so 35B admission needs a
separate dense performance implementation rather than pretending quantized mainloops cover it.
Vision W8 general at T32768 measures 3,180.512 us and 194.46 useful TFLOP/s (88.07% of that
process's synthetic MMA probe), but those useful counters are not physical roofline evidence;
NCU traffic/SOL is still required. Finally, this timing harness does not prove future-shape
numerics. An opt-in measurement-only correctness suite with independent T1 packed-row FP64
oracles and fixed-candidate crossover comparisons is the next gate before drawing the final 35B
extension-cost conclusion.

E042 closes that numerical gate without admitting 35B. This was a Debug build, so its numerical
checks remain evidence but the 12.15 s timing is discarded; E043 repeats it under Release.
`NINFER_LINEAR_TEST_FUTURE35_ONLY=1` selects an isolated suite over the eight quantized
measurement geometries. Each shape constructs the exact row-split payload directly, computes an
independent packed-row FP64 oracle at T1, and launches representative fixed candidates selected by
the then-current screen. E044, not E041/E042, is the Release winner authority. Higher-T checks
avoid an O(N*K*T) host oracle but compare fixed schedules at representative lifecycle seams: TT4
versus W8 MMA at 15/16, 16/17, 54/55, 90/91, and 14/15; TT4 versus
split-low4 BN64 at head T8/9; W8 small-M versus general at T1024; and BN64 versus BN128 at head
T64/65/1024. The comparison walks outputs in bounded host chunks, so peak memory does not scale
with an additional full dequantized matrix.

```bash
NINFER_LINEAR_TEST_FUTURE35_ONLY=1 \
  /usr/bin/time -v ./build/tests/ninfer_linear_test
```

The command exits zero with `OK linear future-35B measurement-only correctness` in 12.15 s and
635,600 KiB maximum RSS. All eight independent T1 oracles satisfy the declared BF16/tensor-core
contracts. Cross-lifecycle W8 and head comparisons have `rel_l2=2.519e-3` through `2.543e-3`, below
the predeclared `4e-3` tensor-core bound. W8 small-M/general and Q4/Q6 BN64/BN128 produce zero
difference at T1024; BN64/BN128 are also identical at T64/65. The result establishes that current
closed codecs and fixed mainloops are numerically valid at N/K=512/2048/4096/4608 and the two
future vocabulary sizes. It does not register those shapes, qualify dense BF16, or turn candidate
agreement into a roofline claim.

E043 audits the build itself because the eleven-minute E038 run was incompatible with every prior
Release execution. `build/CMakeCache.txt` showed `CMAKE_BUILD_TYPE=Debug`; consequently every
E038--E042 functional result remains usable, but their elapsed times and the whole E041 candidate
ranking are invalid. The build was explicitly restored with:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build -j
ninja -C build -t commands ninfer_linear_test | rg -m2 'nvcc|c\+\+'
```

The cache now reports `Release`, and the inspected CUDA commands contain
`-O3 -DNDEBUG -lineinfo` for `sm_120a`. Four independent Release gates then pass:

| Gate | Result | Wall time | Maximum RSS |
|---|---|---:|---:|
| `NINFER_LINEAR_TEST_FUTURE35_ONLY=1` | future-35B measurement-only correctness OK | 2.67 s | 635,592 KiB |
| complete `ninfer_linear_test` | complete Linear correctness OK | 44.21 s | 11,147,112 KiB |
| `NINFER_LINEAR_TEST_PREFILL_FUSIONS_ONLY=1` | prefill-fusion correctness OK | 2.54 s | 991,836 KiB |
| `NINFER_LINEAR_TEST_REGISTERED_GAPS_ONLY=1` | registered-gap correctness OK | 3.15 s | 1,188,844 KiB |

The future suite again accepts all eight independent packed-row FP64 T1 oracles, every
representative cross-lifecycle sentinel, and the `rel_l2 <= 4e-3` tensor-core contract. The full
suite repeats every base, fused, pair, alignment, tail, and dense-reference check. The focused
suites repeat Add/SwiGLU and pair exact comparisons through T2048. This supersedes only
performance metadata, not E038--E042's mathematical conclusions. A performance artifact is now
admissible only when its CMake cache and one representative compile command are recorded beside it.

E044 reruns the complete future-35B pressure matrix under that verified Release build. The broad
screen uses five warmups and twenty cold samples per candidate; ambiguous, reversing, and tile
boundary points use three independent processes with eight warmups and forty cold samples. Raw
evidence is under:

```text
profiles/bench/linear-architecture-20260716/future35_release/screen/
profiles/bench/linear-architecture-20260716/future35_release/confirm/run{1,2,3}/
```

Cold medians are aggregated as the median of the three process medians. Unlike the invalid E041
table, the Release data does not justify inventing a sharp boundary inside a practical tie band:

| Future W8 geometry | Release result within the measured domain |
|---|---|
| `[1024,2048]` | TT1 at T1; TT4 is strictly best through T91; T92--99 is within 0.1%; small-M is clearly better from T100 (1.80% at T100, 19.20% at T128). |
| `[2048,4096]` | TT4 through T54, a tie at T55, small-M from T56 through T832, near-tie at T896, and general clearly wins by T960/T1024. |
| `[9216,2048]` | TT4 is clear through T12 and practically tied with MMA at T13--16; small-M wins T17--128, general T129/T255/T256, small-M again at T257, and general by T511. |
| `[12288,2048]` | TT4 through T16, general from T17 through T128, small-M wins T129 by 4.03%, and general wins again at T255/T256. |
| `[2048,512]` | small-M wins through T640, T704/T768 are ties within 0.2%, and general wins clearly from T832 through T1024. |
| `[2048,4608]` | TT1 at T1, TT4 through T14, small-M T15--832, general from T896 through the measured T32768 endpoint. |

These reversals are caused by CTA-wave and tail geometry, not by a missing runtime heuristic. For
example `[9216,2048]` changes general -> small-M -> general around T256/257/511, while
`[12288,2048]` changes general -> small-M -> general at T128/129/255. A future registered target
therefore needs finite support-local route ranges and an explicit rule for practical ties; one
global `N`, `T`, or `ceil(T/128)` threshold would be wrong.

The two K2048 vocabulary heads are cleaner and reproduce the same three fixed regimes:

| Geometry | T1--8 | T9--64 | T65--1024 |
|---|---|---|---|
| Q4 `[131072,2048]` | TT4 | split-low4 BN64 | split-low4 BN128 |
| Q6 `[248320,2048]` | TT4 | split-low4 BN64 | split-low4 BN128 |

At T8, TT4 versus BN64 is 234.464 versus 294.176 us for Q4 and 546.112 versus 563.136 us for Q6.
At T9, BN64 wins by 11.8% and 38.8%, respectively. BN64 remains decisive at T64, and BN128 wins
at T65 by 37.8% for Q4 and 59.0% for Q6. This is strong transfer evidence for the closed
split-low4 codec plus existing TT4/BN64/BN128 lifecycles; it is not evidence for one generic
quantized loop.

Dense BF16 is the counterexample. The generic reference implementation reaches only 2.554 useful
TFLOP/s for `[64,2048]` and 3.165 useful TFLOP/s for `[257,2048]` at T2048, with a large schedule
discontinuity at T17. A future 35B admission must implement and qualify a dedicated dense schedule.
The Release future correctness gate from E043 proves the eight quantized candidates but does not
admit any of these supports, optimize dense BF16, or cover grouped experts.

E045 replaces the benchmark's misleading historical tuning set with the product support matrix.
The former `--all-targets` list contained only 13 of the 24 live base-Linear physical views and
four stale or unreachable tuning rows. At this checkpoint, an explicit 24-row list matches
`quantized_support_specs()`; E050 later derives the rows directly from that table so friendly
display labels are the only remaining presentation mapping. A CUDA smoke run executes all 24 at
T1, and the retained Release matrix measures T1/128/512/1024 in three processes with eight warmups
and forty cold samples:

```text
profiles/bench/linear-architecture-20260716/registered24_release/*.csv
```

There are 288 rows: one row per 24 supports x four T values x three processes. This is a complete
absolute measurement inventory, not a claim that every view/T pair is target-reachable. A separate
legacy-list final sweep under
`end_to_end/final_release/all_targets_large_p{1,2,3}.csv` supplies the backward comparison,
including its stale rows. Among its 34 large-T points with a directly matching pre-change
measurement, 27/34 cold medians and 26/34 warm medians remain within +/-1%. The live Q5
`[6144,5120]` T128/T512 points are within 1%; a 2.94% cold change on stale Q4 `[7168,5120]` has a
-0.59% warm change and is not a product route. No matched base kernel explains the later
whole-model drift.

The independent pair benchmark then covers the production policy beyond its original candidate
screen. Three processes compare auto, two Small-T launches, and dual W8 at
T128/512/1024/1025/2048:

| T | Auto (us) | two Small-T (us) | forced dual W8 (us) |
|---:|---:|---:|---:|
| 128 | 113.568 | 219.872 | 113.856 |
| 512 | 113.920 | 779.296 | 113.920 |
| 1024 | 152.864 | 1,532.740 | 154.112 |
| 1025 | 161.024 | 1,542.270 | 161.056 |
| 2048 | 300.032 | 3,036.990 | 300.224 |

Raw files are under `pair_large_release/`. Auto resolves dual W8 and stays within 0.82% of the
forced launch through T2048. Together with E039/E040 exact equivalence, pair is functionally and
performance qualified through T2048; T2049+ remains only the legal terminal schedule.

All seven registered Vision views are also measured from T128 through the sampled high-T point
T32768. This is the reachable terminal only for the two W8 merger views; the five encoder views use
P=4V and can reach T131072. Median-of-process cold results at T32768 are:

| Registered Vision view | Time (us) | Useful TFLOP/s | Percent of local synthetic MMA probe |
|---|---:|---:|---:|
| W8 `[4608,4608]` | 7,418.14 | 187.59 | 85.39% |
| W8 `[5120,4608]` | 8,250.27 | 187.41 | 85.40% |
| Q5 `[1152,4304]` | 1,731.84 | 187.63 | 85.52% |
| Q4 `[4304,1152]` | 1,649.89 | 196.95 | 89.54% |
| Q5 `[1152,1152]` | 447.744 | 194.25 | 88.43% |
| Q6 `[1152,1536]` | 582.912 | 198.94 | 90.58% |
| Q4 `[3456,1152]` | 1,232.13 | 211.76 | 96.38% |

Raw files are under `vision_registered_high_release/`. These useful-operation rates demonstrate
that the sampled high-T points execute efficiently, but do not substitute for physical traffic/SOL
evidence or an independent T32768 numerical oracle. Numerical confidence transfers from the same
fixed BN128 full/edge kernel covered at smaller registered T, plus E047's public Vision
integration. No performance, physical, or independent numerical result is claimed for encoder
T32769--131072.

E046 investigates the apparent final Text regression rather than accepting an unmatched before /
after percentage. Three initial final Release processes report median prefill times of
79.013/161.571/304.224 ms for P128/P512/P1024 and 421.670 ms for TG32. Later processes drift to
80.689/162.998/307.876/427.052 ms, and a monitored seventh process reaches
81.259/163.009/309.034/426.467 ms. Historical baseline medians were
78.138/156.131/295.414/412.451 ms; the earlier phase-1 state was
79.248/157.077/297.255/408.867 ms.

During the monitored request the GPU drew about 596--604 W, reported a 21--42% power-violation
fraction, and ran at 2,895--2,925 MHz rather than the observed 2,955 MHz ceiling, at 55--57 C.
Attempting `nvidia-smi -lgc 2805,2805` was rejected for insufficient permission, so an externally
locked matched run was unavailable. This is an environmental limitation, not a larger noise band.

The causal test instead alternates policy binaries in one session: current A, old direct Add B,
old direct Add plus the old folded SwiGLU at P128 C, then B and A again. Each report uses five
measured repetitions after one warmup:

| State | P128 (ms) | P512 (ms) | P1024 (ms) | TG32 (ms) |
|---|---:|---:|---:|---:|
| A current 1 | 78.975 | 159.173 | 302.455 | 419.327 |
| B old Add 1 | 80.374 | 160.797 | 305.308 | 418.697 |
| C old Add + old P128 SwiGLU | 81.050 | 160.641 | 305.121 | 418.964 |
| B old Add 2 | 79.772 | 160.955 | 305.289 | 419.281 |
| A current 2 | 78.989 | 160.555 | 303.028 | 418.966 |

Against bracketed B, current A is faster by 1.091 ms (1.38%) at P128, 1.012 ms (0.63%) at P512,
and 2.557 ms (0.84%) at P1024; TG32 changes by only 0.158 ms. C is another 0.976 ms slower than B
at P128. Thus CTA-collective Add and materialized P128 SwiGLU improve whole prefill. The absolute
2--4% historical gap was not reproducible under the same power/clock state and follows unchanged
paths. All temporary ablations were reverted and the current Release policy rebuilt.

E047 closes remaining real-product routes. Six MTP configurations use three independent Release
processes and three repetitions per process. Every one of the 54 repetitions preserves exact
speculative rounds, drafted tokens, accepted tokens, fallback steps, and
`accepted_per_position` from its matching baseline. Robust process-median timings are:

| Workload | Baseline (ms) | Final Release (ms) | Change |
|---|---:|---:|---:|
| TG32 full-head decode | 438.453 | 447.427 | +2.05% |
| TG32 optimized-head decode | 468.295 | 485.553 | +3.69% |
| P32+G32 full prefill / decode | 83.186 / 310.050 | 84.182 / 321.387 | +1.20% / +3.66% |
| P32+G32 optimized prefill / decode | 81.246 / 293.566 | 80.916 / 300.709 | -0.41% / +2.43% |
| P128+G32 full prefill / decode | 84.699 / 531.455 | 85.333 / 548.914 | +0.75% / +3.29% |
| P128+G32 optimized prefill / decode | 82.914 / 501.496 | 83.636 / 518.000 | +0.87% / +3.29% |

The decode shift is the same environmental drift seen in Text TG32; route behavior is exact. Raw
reports are the 18 `mtp_*` JSON files under `end_to_end/final_release/`.

The first attempt to reach Vision through the full real-artifact test spent 39 minutes on unrelated
prefix/MTP cases and was terminated; this was test selection, not a GPU hang. The test now has an
opt-in public-Engine-only branch:

```bash
NINFER_QWEN3_6_27B_WEIGHTS=out/qwen3_6_27b_rtx5090.ninfer \
NINFER_QWEN3_6_27B_VISION_ONLY=1 \
  /usr/bin/time -v ./build/tests/ninfer_qwen3_6_27b_prefix_real_test
```

Three Release processes load the exact 17,502,555,648-byte artifact and pass. Reported
`vision_seconds` are 9.483/9.148/10.113 ms, prefill is 86.706/87.334/86.282 ms, and total time is
100.679/100.918/100.990 ms. The default integration route is unchanged except that it now prints
public `GenerationTimings` for its existing Vision case. This is an integration gate, not a 32K
token numerical oracle.

E048 performs final whole-inference attribution with NSYS. Benchmark-only `--profile-measured`
places `cudaProfilerStart/Stop` around exactly one measured repetition after warmup, synchronizing
at both boundaries. Validation requires one benchmark test and `-r 1`, so load, upload, graph
construction, and warmup cannot enter the capture. The three Release reports use
`--capture-range=cudaProfilerApi`; graph workloads additionally trace CUDA Graph nodes. Reports,
SQLite, official `nsys stats` CSVs, JSON, and summaries are under
`profiles/nsys/linear-architecture-20260716/release/`.

| Isolated workload | GPU kernel sum | Linear/GEMM attribution | Dominant evidence |
|---|---:|---:|---|
| Text P128 | 75.007 ms | 69.777 ms, 93.0% | Q5 MMA 36.269 ms; Q4 MMA 17.285 ms |
| Text TG32 | 417.527 ms | name heuristic undercounts it | six Linear families total 355.581 ms; fused gate/up 134.709 ms and Q5 down 84.844 ms lead |
| MTP P32+G32 optimized head | 367.211 ms | 285.427 ms, 77.7% | Q4 TT8 123.165 ms; Q5 MMA 36.662 ms; Q5 direct 29.533 + 19.279 ms |

The decode summary's broad class heuristic labels tuned symbol names as `other` or
`mlp / activation`; top-kernel rows are therefore the authority for its Linear total. CUDA stream
synchronization accounts for 413.946 ms in TG32, while graph-launch API time is 25.029 ms under
profiling. NSYS kernel sums are attribution, not end-to-end latency. Release sums are lower than
the invalid Debug captures (75.007 vs 76.225, 417.527 vs 423.927, and 367.211 vs 373.637 ms).

E049 applies the NCU workflow only after E048 identifies Linear and the microbench identifies fixed
policies. Preflight passes on RTX 5090 / WSL2 / NCU 2025.4.1. Each retained report uses application
replay, an exact demangled base-name filter, one matched launch, SpeedOfLight, Occupancy, and
MemoryWorkloadAnalysis plus explicit physical-byte/resource/tensor-instruction metrics. Two initial
template-heavy regex attempts (one Q5, one W8) matched no launch and are retained only as failed
selection attempts; they contribute no data. Final reports are under:

```text
profiles/ncu/linear-architecture-20260716/release/*_app.{ncu-rep,csv,txt}
```

The performance clock is always the three-process cold microbenchmark median, never NCU replay
duration. Q5 BN128 required one application replay for standard sections and another for custom
metrics; those two reports are not combined into a bandwidth calculation. Exported `Mbyte/Gbyte`
use SI units and Gbyte rows have only about three significant digits.

For contraction accounting, useful FLOP is `2*N*K*T`. MMA executed FLOP is
`2*ceil(N/BM)*BM*ceil(T/BN)*BN*Kpad`; pair uses effective N=2048 and folded SwiGLU counts all
34,816 gate/up contraction rows. Residual addition and SiLU/multiply are excluded from FLOP but
remain in measured physical traffic. Every MMA report independently satisfies
`tensor instructions * 4096 == executed BF16 FLOP`.

| Representative fixed policy | Bench us | Useful / executed TFLOP/s | DRAM MB; useful / executed AI (F/B) | SM / DRAM SOL | Waves |
|---|---:|---:|---:|---:|---:|
| Q5 BN64 `[5120,6144]`, T128 | 144.672 | 55.664 / 55.664 | 22.23; 362 / 362 | 24.19% / 6.63% | 0.31 |
| Q5 BN128 `[6144,5120]`, T512 | 261.376 | 123.241 / 123.241 | 25.91; 1,243 / 1,243 | 52.55% / 3.84% | 1.13 |
| W8 small-M `[1024,5120]`, T512 | 77.056 | 69.673 / 69.673 | 11.09; 484 / 484 | 33.63% / 6.84% | 0.38 |
| W8 general `[34816,5120]`, T512 | 1,015.040 | 179.831 / 179.831 | 815.54; 224 / 224 | 77.94% / 33.05% | 6.40 |
| dual W8 pair `2x[1024,5120]`, T1024 | 152.864 | 140.483 / 140.483 | 24.70; 869 / 869 | 56.56% / 6.22% | 0.75 |
| Q5 CTA-collective Add `[5120,6144]`, T1024 | 335.200 | 192.197 / 192.197 | 46.86; 1,375 / 1,375 | 85.40% / 5.64% | 1.88 |
| Q4 folded SwiGLU `[34816,5120]`, T129 | 521.568 | 88.177 / 174.988 | 138.78; 331 / 658 | 74.99% / 10.71% | 3.20 |
| dedicated Q5 T1 `[6144,5120]` | 19.744 | 3.187 / 3.187 | 20.68; 3.04 / 3.04 | 47.20% / 50.85% | 0.75 |
| dedicated Q6 head T1 `[248320,5120]` | 638.240 | 3.984 / 3.984 | about 1,010; 2.52 / 2.52 | 69.79% / 76.81% | 30.43 |
| Q5 direct split4 `[6144,5120]`, T6 | 30.336 | 12.444 / 12.444 | 20.73; 18.2 / 18.2 | 62.80% / 39.16% | 3.61 |
| Q4 TT4 `[6144,5120]`, T1 | 27.904 | 2.255 / 2.255 | 16.74; 3.76 / 3.76 | 47.12% / 34.62% | 1.13 |
| Vision Q4 BN128 `[3456,1152]`, T32768 | 1,232.130 | 211.763 / 211.763 | 287.76; 907 / 907 | 90.65% / 9.29% | 40.66 |

The same reports show distinct resource regimes rather than one tunable loop:

| Policy family | Registers/thread | Static shared | Achieved / theoretical occupancy |
|---|---:|---:|---:|
| Q5 BN64 / BN128 | 102 / 154 | 30.98 / 47.62 KiB | 8.33/25% / 15.31/16.67% |
| W8 small-M / general | 83 / 119 | 40.45 / 47.10 KiB | 16.64/33.33% / 32.28/33.33% |
| pair / Add / folded SwiGLU | 104 / 154 / 107 | 43.01 / 47.62 / 46.59 KiB | 25.04/33.33% / 16.45/16.67% / 31.60/33.33% |
| Q5 T1 / Q6 head T1 / Q5 direct / Q4 TT4 | 39 / 37 / 48 / 56 | 32.77 / 4.10 / 1.12 / 9.73 KiB | 69.45/100% / 89.90/100% / 74.52/83.33% / 61.72/66.67% |
| Vision Q4 BN128 | 152 | 46.59 KiB | 16.58/16.67% |

No sampled kernel spills. Current same-process synthetic ceilings are about 219.6--220.5 TFLOP/s
and 1,510.916 GB/s, a ridge near 145--146 FLOP/B. The older E003 197.458 TFLOP/s probe belongs to
a different clock epoch and is not mixed with this table. Physical conclusions are:

- Vision Q4 T32768 reaches 211.763 TFLOP/s, 96.38% of its same-process compute probe, with 90.65%
  SM SOL. This is the clearest sampled policy that reaches its relevant compute roofline.
- CTA-collective Add reaches 192.197 TFLOP/s and 85.40% SM SOL; the collective finalizer does not
  materially destroy its contraction throughput.
- W8 general reaches 179.831 TFLOP/s and near-theoretical occupancy, but reads 815.54 MB from DRAM,
  3.54x the minimum logical bytes, because its approximately 189 MB weight cannot remain in L2
  across four T tiles. Useful-byte roofline reporting would conceal this.
- Folded SwiGLU T129 executes almost 1.984x its useful contraction FLOP because BN128 pads T to
  256. Its 174.988 executed TFLOP/s becomes only 88.177 useful TFLOP/s; this is direct evidence for
  a measured semantic-Op route table rather than an always-fuse rule.
- BN64, BN128, W8 small-M, and pair launch only 0.31/1.13/0.38/0.75 waves. Their low SM percentages
  are partial-wave limits, not DRAM roofs. T1 Q6 is instead bandwidth-side at 76.81% DRAM SOL;
  small T1 views also pay latency/under-wave costs. Q5 direct T6 has no tensor instructions and
  must not be assessed against a tensor-core roofline.

These twelve samples establish a qualification method and representative physical limits, not a
claim that all 24 supports or 83 base routes reach roofline.

E050 performs the final source-to-binary closure rather than stopping at passing route tests. The
last review found four forms of residual duplicate authority:

1. W8 launch selection still spelled one structural decision as `weight.n <= 1024`, even though
   admission was exact;
2. the structural test mirrored that same heuristic instead of the admitted exact small-M view;
3. `--all-targets` carried a second hand-maintained list instead of deriving its rows from
   `SupportSpec`;
4. the unregistered Q4/Q5 `[7168,5120]` and unreachable dual-Q5 `[6144,5120]` tuned translation
   units were no longer dispatched but still linked.

The final W8 resolver names the one exact small-M view `[1024,5120,5120]` and maps the other eight
current W8 views explicitly to the general lifecycle; a constexpr failure makes a future admitted
view impossible to omit silently. The independent structural expectation now identifies that same
exact view instead of repeating an N threshold. The benchmark converts the same 24 `SupportSpec`
rows into CLI qtype/shape rows. The four obsolete source/header files containing the 7168 and dead
dual-Q5 kernels are removed from the build and repository. These changes do not move a registered
route boundary or alter a retained CUDA kernel body.

The first default build exposed one integration failure that target-specific builds had missed:
the CPU-only planner benchmark directly compiles `linear_plan.cpp`, whose internal Op header
includes CUDA types, but its target lacked the CUDA header search path. The target now receives
`${CUDAToolkit_INCLUDE_DIRS}` without linking CUDA. `ldd` confirms that the resulting executable
depends only on libc/libm/libstdc++/libgcc; it still does not initialize or link `cudart`. The
corrected default build configured successfully and built the remaining 61 targets.

Final exact-worktree verification is:

```bash
cmake --build build -j

./build/tests/ninfer_linear_plan_test
./build/tests/ninfer_linear_fused_plan_test
./build/tests/ninfer_bench_support_test
./build/bench/ninfer_linear_plan_bench

/usr/bin/time -v ./build/tests/ninfer_linear_test
NINFER_LINEAR_TEST_FUTURE35_ONLY=1 \
  /usr/bin/time -v ./build/tests/ninfer_linear_test
NINFER_LINEAR_TEST_PREFILL_FUSIONS_ONLY=1 \
  /usr/bin/time -v ./build/tests/ninfer_linear_test
NINFER_LINEAR_TEST_REGISTERED_GAPS_ONLY=1 \
  /usr/bin/time -v ./build/tests/ninfer_linear_test

./build/bench/ninfer_linear_op_bench \
  --all-targets --t-sweep 1 --warmup 1 --repeat 2 \
  --stream-ceiling-gbs 1510.916
```

Both structural plan binaries report `OK`; the benchmark support test exits zero. The complete
Release numerical suite reports `OK linear correctness` in 42.75 s with 11,146,956 KiB maximum
RSS. Future-shape, prefill-fusion, and registered-gap modes pass in 1.39/2.48/3.31 s with
634,860/991,332/1,188,488 KiB maximum RSS. The support-derived benchmark emits exactly 24 data
rows and executes every admitted view at T1. That one-warmup run is only a reachability smoke test;
its cold-start outlier is not added to any performance conclusion. All changed C++/CUDA files pass
`clang-format --dry-run --Werror`, and `git diff --check` passes.

The final Release linked checkpoint is:

| Item | Final value | Change from pre-cleanup checkpoint |
|---|---:|---:|
| `ninfer_linear_op_bench` | 13,358,080 B | -123,392 B (-0.92%) |
| `ninfer_linear_test` | 13,074,816 B | -132,160 B (-1.00%) |
| benchmark `.text` | 581,490 B | -4,528 B (-0.77%) |
| benchmark `.nv_fatbin` | 5,029,176 B | -34,992 B (-0.69%) |
| benchmark `__nv_relfatbin` | 7,091,944 B | -79,592 B (-1.11%) |
| full benchmark SASS | 706,207 lines | -4,783 lines (-0.67%) |
| true demangled Linear entries | 85 | -3 |

The older mangled-name regular expression reports 89 entries because it also matches four
benchmark helpers whose anonymous-namespace symbol contains `linear_op_bench`; demangling gives
the authoritative 85. Symbols containing `7168` or `linear_rowsplit_gemv_q5_dual` are zero. The
three removed CUDA entries are exactly Q4 `[7168,5120]`, Q5 `[7168,5120]`, and dual-Q5
`[6144,5120]`; they account for 4,768 SASS instruction lines.

Across all 85 final Linear entries, registers span 31--166, static shared memory 0--48,640 B,
constant0 928--1,108 B, and every entry has zero stack and zero local memory. Relative to E012, the
raw entry count remains 89, but its composition changes by `+4` named TT1 and `+4` CTA-collective
Add instances, offset by five obsolete Q5 N=5120 direct-split2 instances and the three dead tuned
entries. Relative to E023, the final binary has eight fewer semantic instances and 19,880 fewer
SASS lines. All 27 retained Small-T functions preserve exact resources and instruction streams;
all 82 same-name Linear entries shared with E023 preserve exact resources. The exact-view W8
mapping and support-derived host inventory generate no CUDA entry.

For descriptive provenance, the final benchmark and test binaries have SHA-256
`bedf9cf022e99b7c54fa14db1e6830a87a239f6108f7fb04d82f44be50ff1ef9` and
`d704701e797b14a33457d05e2183ba04c8dfd5a353675b927dfb42ed6ba80b43`; the full benchmark SASS
stream is `90968069136c6d18eed028131a1397165bce567cf31702dbf85c6e04c2a3fc67`. These hashes describe
this checkpoint and are not product validity requirements.

## 10. Revised conclusions and residual work

E051 changes ownership, not the measured execution facts.

The retained conclusions are:

- support is a finite set of exact physical problems; future shapes add measured rows rather than
  weakening admission;
- route boundaries are geometry-local and may be non-monotonic;
- each semantic Op needs one finite route/workspace plan and one fixed execution closure;
- complete mainloops are shared only when staging lifetime, synchronization, warp ownership, and
  output ownership match;
- codecs, decode atoms, MMA fragments, topology helpers, and narrow finalizers may be shared below
  that boundary when codegen evidence permits;
- fusion is supported through independent semantic Ops and finite complete policies, not a
  universal runtime epilogue;
- materialized composition is a legitimate winning policy, and capacity is the maximum scratch
  across every reachable route;
- roofline qualification is per fixed policy and wave/tail class. The demonstrated 32K Vision
  result does not qualify every route.

The superseded conclusions are:

- target owns Linear route selection;
- callers should pass `LinearExecutionProfile`;
- target/profile and kernel catalog form the semantic execution contract;
- a grid-legal terminal route can be admitted beyond its winner evidence;
- a fused executor may call public auto-dispatch Linear and silently inherit later Base route
  changes.

The revised ownership is:

```text
target owns which physical calls are reachable
semantic Op owns exact support, route resolution, workspace, and fixed execution
```

The 24 current base problems, 83 prototype routes, fused route surfaces, kernel resources, SASS,
correctness results, NSYS attribution, and NCU reports remain the starting evidence for a clean
Op-owned implementation. They must be rechecked against the qualified token envelope rather than
copied as a target profile.

Remaining work is:

- implement the corrected Op-owned architecture from the clean product baseline;
- revalidate the admitted token envelope against real Text, Vision, and MTP reachability;
- preserve or improve the measured route winners with unchanged semantic APIs;
- optimize and physically qualify product-dominant T1/Small-T, W8 reread, partial-wave, and
  tail-heavy policies;
- add a high-T independent Vision numerical oracle before changing those routes;
- design optimized dense BF16 and grouped-expert lifecycles only when required;
- admit future 35B physical problems only through exact numerical, winner, physical, and product
  gates.

## 11. Q4 三模板族重启、资格与 Roofline 实验（E052--E055）

### 11.1 实现边界与环境

本轮只实现纯 `Q4G64_F16S / RowSplit / Kpad==K` Linear candidate，production
`linear()` dispatch 保持不变。新增三个 kernel 模板族：

```text
q4_rowsplit_gemv_kernel
q4_rowsplit_gemm_simt_kernel
q4_rowsplit_gemm_mma_kernel
```

构建和运行环境：

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| compute capability | 12.0，编译目标 `sm_120a` |
| CUDA runtime / driver | 13.1 / 13.1，benchmark 数值 `13010/13010` |
| build | CMake `Release`，`-O3 -DNDEBUG` |
| candidate binary | 当前源码重新构建的 `build/bench/ninfer_linear_op_bench` |
| raw CSV | `profiles/bench/q4-linear-template-20260716/` |

构建、数值与资源命令：

```bash
cmake --build build -j 12 \
  --target ninfer_q4_linear_candidate_test ninfer_linear_op_bench

./build/tests/ninfer_q4_linear_candidate_test

/usr/local/cuda/bin/cuobjdump --dump-resource-usage \
  build/src/libninfer_ops.a
```

benchmark 的 fixed Q4 路径接受任意 measurement-only `Rows/K/Cols`，记录 canonical
schedule identity、variant、build、GPU 和 CUDA。CSV 明确区分：

- `useful_*`；
- 只考虑 column tile 权重重放的 `weight_replay_lower_bound_*`；
- Q4 `auto` 或 fixed candidate 解析到 MMA schedule 时才有意义的
  `executed_tflops/executed_tc_pct`；
- physical bytes、occupancy 和 SOL 留给 NCU。

因此结构 screen 中传入的 `--stream-ceiling-gbs 1000` 只是固定输出上下文，不构成
roofline 结论；本节只使用 cold latency 比较候选。

### 11.2 数值门

oracle 固定为：

```text
BF16-rounded X
test-side packed Q4 RowSplit dequant
FP64 CPU W @ X
```

覆盖：

- 资格阶段的三个 GEMV schedule；cutover 后保留的 R4/W1 与 R1/W8 均在 K5120
  独立 oracle 下复验，其中 R4/W1 还保留 K512 partial-tile 覆盖；
- SIMT C4/C8 的 Full、Predicated、K1152 partial stage；
- SIMT K3072 的三 stage ring-buffer 复用；
- MMA C64/C128 的 Full、Predicated；
- MMA K384 的多 stage 复用；
- illegal Full、lifecycle variant mismatch、Kpad mismatch、静态 GEMV K mismatch 和未知
  ScheduleId rejection。

最终 `ninfer_q4_linear_candidate_test` 报告：

```text
OK Q4 Linear fixed candidates
```

GEMV/SIMT `rel_l2` 约为 `1.54e-3..1.81e-3`；MMA 为
`2.08e-3..2.11e-3`，低于 `linear_tc` 的 `4e-3` normwise 上限。

### 11.3 最终 linked resources

以下是 linked archive 的 `cuobjdump` 数值；所有 entry 均为 0 stack、0 local、无 spill：

| schedule / variant | registers | linked static shared |
|---|---:|---:|
| GEMV R4/W1 shared | 40 | 3,200 B，另加运行时 `2*K` dynamic shared |
| GEMV R4/W1 direct | 40 | 3,200 B |
| GEMV R1/W8 static-groups80 | 36 | 5,152 B |
| SIMT R8/C4 Full / Predicated | 102 / 78 | 9,728 B |
| SIMT R8/C8 Full / Predicated | 128 / 96 | 9,728 B |
| MMA R64/C64 Full / Predicated | 98 / 101 | 29,952 B |
| MMA R64/C128 Full / Predicated | 150 / 152 | 46,592 B |

MMA 的 source-level array 总量比 linked shared 少 1,024 B；资格记录采用 linked
resource，而不是只引用模板公式。

### 11.4 Matched structural screen

共同命令形态：

```bash
./build/bench/ninfer_linear_op_bench \
  --rows N --k K --qtype Q4 \
  --candidate CANDIDATE --t-sweep T \
  --warmup 5 --repeat 20 \
  --stream-ceiling-gbs 1000 \
  --csv-out profiles/bench/q4-linear-template-20260716/NAME.csv
```

结果：

| matched point | legacy cold median | new cold median | 结论 |
|---|---:|---:|---|
| split-K GEMV `[4096,5120] C1` | 11.520 us | 11.520 us | 等价 |
| warp-row GEMV `[34816,5120] C1` | 70.912 us | 70.880 us | 等价 |
| SIMT C4 `[6144,5120] C4` | 32.032 us | 27.904 us | 新实现快 12.9% |
| SIMT C8 `[34816,5120] C8` | 204.064 us | 134.112 us | 新实现快 34.3% |
| MMA C64 `[4096,5120] C64` | 105.760 us | 105.728 us | practical tie |
| MMA C128 `[3456,1152] C128` | 29.920 us | 29.952 us | practical tie |

这仍是 structural screen；最终 route seam 和小于 3% 的差异按三进程
8-warmup/40-cold 规则确认。

### 11.5 Split-K 失败序列与架构修正

首版 R1/W8 把 warp-row 的 PackedWord8/SharedPair32 内层映射强加给 split-K，结果为：

| 版本 | cold median | resources | 判断 |
|---|---:|---|---|
| runtime PackedWord8 | 15.616 us | 48 reg / 5,408 B | 比旧 winner 慢 35.2%，拒绝 |
| runtime PackedByte2 pair loop | 19.488 us | 37 reg / 5,152 B | 资源恢复但 ILP 更差，拒绝 |
| runtime ownership + active-groups-10 unroll | 17.408 us | 39 reg / 5,152 B | 仍慢 51.1%，拒绝 |
| `StaticGroupsPerRow=80` | 11.520 us | 36 reg / 5,152 B | 与旧 winner 相同，接受 |

最终 schedule 保持 runtime Rows，但将 K5120 的 80 个 group 编译期均分到 8 个 warp：

```text
10 groups / warp
5 scale pairs / warp
PackedByte2 + ScalarInteger
SyncVector16 code
Scalar16Shuffle scale
```

最终 SASS hot mix 与旧实现对应：20 FFMA、10 X global loads、10 code shared loads、10
shuffle。结论不是“所有 K 都应静态化”，而是：

```text
默认 StaticGroupsPerRow=0
只有 runtime K 稳定回退超过 1% 且静态值可跨 Rows 复用时，才增加一个静态 schedule
```

R1/W8 因此只接受 K5120；R4/W1、SIMT 和 MMA 继续使用 runtime K。这个修正保留三个
kernel lifecycle，不增加应用场景 kernel，也不把 exact Rows 变成模板参数。

### 11.6 三进程 route 资格结论

同一 session 重新测得：

```text
copy ceiling = 1512.483 GB/s
BF16 MMA probe = 220.246 TFLOP/s
```

每个确认点运行三个独立进程，每进程 8 warmup、40 cold samples；第二个进程反转候选
顺序。matched median-of-process-medians：

| point | legacy | new | 结论 |
|---|---:|---:|---|
| R4/W1 `[34816,5120] C1` | 70.944 us | 70.912 us | tie |
| R1/W8 `[4096,5120] C1` | 11.520 us | 11.520 us | tie |
| SIMT C4 `[6144,5120] C4` | 32.032 us | 27.904 us | new -12.9% |
| SIMT C8 `[34816,5120] C8` | 204.064 us | 134.400 us | new -34.1% |
| MMA C64 `[4096,5120] C64` | 107.008 us | 105.760 us | new -1.2% |
| MMA C128 `[3456,1152] C128` | 29.952 us | 29.952 us | tie |

最终 current-product route：

| exact problem | admitted Cols | schedule |
|---|---|---|
| `[1024,5120,5120]` | 1 | R1/W8 static-groups80 |
|  | 2--15 | SIMT C4 |
|  | 16 | SIMT C8 |
| `[4096,5120,5120]` | 1 | R1/W8 static-groups80 |
|  | 2--4 | SIMT C4 |
|  | 5--16 | SIMT C8 |
| `[6144,5120,5120]` | 1 | R1/W8 static-groups80 |
|  | 2--7 | SIMT C4 |
|  | 8--16 | SIMT C8 |
| `[34816,5120,5120]` | 2--4 | SIMT C4 |
|  | 5--16 | SIMT C8 |
| `[131072,5120,5120]` | 1 | R4/W1 direct runtime-groups |
| `[3456,1152,1152]`, `C%4==0` | 4--36 | SIMT C4 |
|  | 40--320 | MMA C64 |
|  | 324--131072 | MMA C128 |
| `[4304,1152,1152]`, `C%4==0` | 4 | SIMT C4 |
|  | 8 | SIMT C8 |
|  | 12 | SIMT C4 |
|  | 16--24 | SIMT C8 |
|  | 28--320 | MMA C64 |
|  | 324--131072 | MMA C128 |

Vision 表中的范围只包含 4 的倍数。`[4304,1152] C20` 的 C4/C8 三进程 median 是
25.856/25.888 us，按 practical tie 选择 C8 以避免在 C16--24 中制造无收益的额外
边界。`[3456,1152] C40` 的 C4/C64/C128 是 29.952/29.952/29.984 us，选择 C64 以连接
后续明确的 C64 winner。两个 Vision shape 在 C320/324 共同切换：

| point | C64 | C128 |
|---|---:|---:|
| `[3456,1152] C320` | 31.648 us | 34.048 us |
| `[3456,1152] C324` | 34.048 us | 34.048 us |
| `[4304,1152] C320` | 34.048 us | 35.936 us |
| `[4304,1152] C324` | 36.096 us | 36.064 us |

Full variant 不能删除。单进程 full/predicated 代表点：

| schedule | Full | Predicated |
|---|---:|---:|
| SIMT C4 `[6144,5120] C4` | 约 27.4 us | 32.7 us |
| SIMT C8 `[34816,5120] C8` | 约 134.3 us | 165.2 us |
| MMA C64 `[3456,1152] C64` | 约 27.9 us | 31.8 us |
| MMA C128 `[3456,1152] C128` | 约 29.9 us | 31.9 us |

### 11.7 Future K2048 transfer screen

`[131072,2048]` 仍为 measurement-only，不进入 admission。5/20 screen：

| Cols | winner | cold median |
|---:|---|---:|
| 1 | R4/W1 shared/direct practical tie | 105.632 / 105.664 us |
| 8 | SIMT C8 | 206.144 us |
| 9 | MMA C64 | 294.176 us |
| 64 | MMA C64 | 285.472 us |
| 65 | MMA C128 | 377.856 us |

因此新增该相似 shape 的成本是数值资格、route 数据和 admission，而不是第四个 kernel
族。当前产品不需要该 shape，所以不增加 support row。

### 11.8 NCU 物理证据

技能 preflight 通过；NCU 版本 2025.4.1.0。每个报告使用精确 kernel regex、
`--launch-skip 0 --launch-count 1`。先用 `--set basic` 证明匹配，再用：

```text
--set detailed
explicit traffic metrics
explicit warp-stall metrics
```

报告位于：

```text
profiles/ncu/q4-linear-template-20260716/
```

NCU duration 只用于 profiler 内部诊断，route timing 仍采用无 profiler 的 CUDA event
结果。

| topology / point | SM SOL | DRAM SOL | achieved occupancy | regs / source static shared | physical traffic 摘要 | 主要 stall |
|---|---:|---:|---:|---|---|---|
| R1/W8 `[4096,5120] C1` | 66.86% | 52.77% | 86.85% | 36 / 4.13 KiB | DRAM read 11.16 MB，匹配一次 Q4 payload | long scoreboard、not selected、MIO、barrier |
| R4/W1 direct `[131072,5120] C1` | 52.40% | 75.18% | 75.85% | 44 / 2.18 KiB | DRAM read 356.55 MB，匹配 code+scale；19.28 waves | long scoreboard 明显占主导 |
| SIMT C4 `[34816,5120] C4` | 67.63% | 50.92% | 31.60% | 102 / 8.70 KiB | DRAM read 94.76 MB，匹配一次 payload | not selected、long scoreboard、math throttle |
| SIMT C8 `[34816,5120] C8` | 69.95% | 31.94% | 32.13% | 128 / 8.70 KiB | DRAM read 94.82 MB，匹配一次 payload | not selected、math throttle、long scoreboard |
| MMA C64 `[3456,1152] C320` | 36.17% | 3.91% | 13.24% | 98 / 28.93 KiB | 0.53 waves；622,080 tensor instructions | wait、math throttle、long scoreboard、barrier |
| MMA C128 `[3456,1152] C32768` | 90.66% | 9.12% | 16.59% | 150 / 45.57 KiB | 63,700,992 tensor instructions；DRAM read/write 77.63/189.54 MB | wait、math throttle、short scoreboard |

解释：

- R4 direct 是明确的 DRAM-side 饱和 kernel；实际 DRAM read 与 356.5 MB Q4 code+scale
  payload 对齐，达到 75.18% DRAM SOL。
- SIMT C4/C8 不是单一 copy-roof kernel；二者同时有约 68--70% SM SOL，C8 的 DRAM SOL
  更低，说明寄存器/issue/compute 与 cache reuse 共同限制。
- MMA C64 的低 SOL 来自 0.53 waves 的 partial-wave launch，按最低延迟判断，不要求峰值
  百分比。
- 饱和 MMA C128 达到 90.66% SM SOL，满足本文约 85--90% BF16 MMA roofline 目标。
- 六个 topology 的 local/shared spill metric 都为 0。

### 11.9 Production 原子切换与真实产品验收（E056）

#### 11.9.1 实现变化

原子 change set 同时完成：

1. 新增 `q4_rowsplit_plan.{h,cpp}`：
   - 7 个 exact support；
   - 21 条 support-local route；
   - 编译期证明 support 唯一、route span 连续、无空洞并完整覆盖 admission；
   - GEMV 派生 `None`，tiled schedule 优先合法 `Full`，否则 `Predicated`。
2. `linear()` 保持原始语义签名，在 Q4 metadata、contiguous 和对齐验证后直接进入
   Q4 planner；caller 不传 target、profile、role、schedule 或 variant。
3. production 使用独立 `q4_rowsplit_launch_fixed(Q4Plan, ...)`；benchmark/test 的
   candidate 入口只包装同一个闭合 fixed launch，不成为第二个 route owner。
4. 删除未获 route 的 R4/W1 shared 实例；最终为：

   ```text
   2 GEMV entries
   2 SIMT schedules x Full/Predicated
   2 MMA schedules x Full/Predicated
   = 10 production CUDA entries
   ```

5. 删除旧 pure-Q4：
   - application-named GateUp、Attention parent、GDN Q/K、LM-head plain GEMV；
   - generic Small-T 中的 `Q4Smallt` 与 Q4 case；
   - generic low-bit MMA 中的 Q4 case/short rule；
   - 四个 legacy benchmark selector；
   - 旧 Q4 policy ID 和 resolver 分支。
6. 保留 Q5/Q6/W8、Q5 residual、grouped input、folded SwiGLU、`Q4Codec` 和共享 MMA
   header 基座。
7. source audit 证明 `Phase::Decode` 没有调用者后，删除两个死分支和枚举值；packed
   Q4/Q5 parent 仍作为 artifact row-view storage 保留。

#### 11.9.2 构建与数值/route 验收

命令：

```bash
cmake --build build -j 12

./build/tests/ninfer_q4_linear_plan_test
./build/tests/ninfer_q4_linear_candidate_test
./build/tests/ninfer_q4_linear_dispatch_test
NINFER_LINEAR_TEST_PREFILL_FUSIONS_ONLY=1 \
  ./build/tests/ninfer_linear_test
./build/tests/ninfer_linear_test
```

结果：

```text
full build                              PASS
Q4 production plan                     OK
Q4 fixed candidates                    OK
Q4 public dispatch                     OK
prefill grouped/fused correctness      OK
full Linear correctness                OK
```

最终 linked archive 的 `cuobjdump --dump-resource-usage` 只有 10 个唯一 Q4 pure-Linear
kernel symbols；`nm -C` 和 source `rg` 均找不到被删除的 application-named Q4
launcher、`Q4Smallt` 或 R4/W1 shared entry。10 个 entry 继续保持 E055 记录的 zero
stack/local/spill 资源合同。

`q4_linear_plan_test` 覆盖完整 Vision step=4 域以及所有 Text/Vision route seam 和
rejection。`q4_linear_dispatch_test` 对 7 个 support 复用 deterministic packed weight，
在 21 条 route 的首末点去重后运行 34 个 GPU case；每个 case 同时断言 plan identity，
再比较 public `linear()` 与 expected fixed candidate 的全部 BF16 words。六个 schedule
和 `None/Full/Predicated` 均被实际执行。Public rejection 包括：

```text
[7168,5120] C1
[34816,5120] C1/C17
[131072,2048] C1
Vision C5
```

独立 fixed-candidate suite 还以 test-side Q4 dequant + FP64 `W @ X` 验证生产
R4/W1 的 K5120 五个完整 group tile；因此 draft-head 路径的数学正确性不依赖
public/fixed 同实现逐 word 对照。

real artifact gate：

```bash
NINFER_QWEN3_6_27B_WEIGHTS=out/qwen3_6_27b_rtx5090.ninfer \
  ./build/tests/ninfer_qwen3_6_27b_prefix_real_test
```

输出 `ok`。该测试通过一个真实 Engine 同时覆盖 Text、Vision、MTP、prefix/state 和
当前 artifact binding。

#### 11.9.3 Matched end-to-end old/new

为了避免用不同时间的绝对吞吐判断回退，建立 detached old worktree：

```bash
git worktree add --detach /tmp/ninfer-q4-old-eb9257e eb9257e
cmake -S /tmp/ninfer-q4-old-eb9257e \
      -B /tmp/ninfer-q4-old-eb9257e/build \
      -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ninfer-q4-old-eb9257e/build \
      --target ninfer_bench -j "$(nproc)"
```

Text 命令：

```bash
./build/bench/ninfer_bench \
  --weights out/qwen3_6_27b_rtx5090.ninfer \
  -pg 128,32 -r 3 --warmup 1 \
  --prefill-chunk 128 --output table
```

即时 new/old/new bracket：

| build | prefill tok/s | decode output tok/s |
|---|---:|---:|
| new | 1633.09 ± 5.27 | 78.61 ± 0.06 |
| old `eb9257e` | 1633.87 ± 5.63 | 77.90 ± 0.19 |
| new | 1635.07 ± 3.88 | 78.55 ± 0.03 |

因此稳定 bracket 中 Text prefill practical tie，decode 无回退。更早一次 new
1583.46/76.22 和 old 1677.06/79.68 发生在 GPU 功耗/频率漂移段；随后的即时 bracket
同时收敛，不能把那两个孤立绝对值归因给代码。

optimized draft-head MTP 命令：

```bash
./build/bench/ninfer_bench \
  --weights out/qwen3_6_27b_rtx5090.ninfer \
  -pg 32,32 -r 3 --warmup 1 \
  --prefill-chunk 128 \
  --mtp-draft-tokens 5 --lm-head-draft \
  --output table
```

| build | prefill tok/s | decode output tok/s | acceptance | rounds/fallbacks |
|---|---:|---:|---:|---:|
| new | 394.43 ± 1.06 | 120.77 ± 0.26 | 0.5 | 24/12 |
| old `eb9257e` | 394.95 ± 1.94 | 108.33 ± 0.12 | 0.5 | 24/12 |
| new | 396.13 ± 1.27 | 120.82 ± 0.38 | 0.5 | 24/12 |

MTP 语义轨迹不变，新实现 decode output throughput 提高约 11.5%。本轮没有追加新的
NSYS attribution，因此不把全部增益强行归因给单个 kernel；实际发生变化的 Q4 路径
包括 MTP final Q projection 和 optimized draft head，它们现在都使用 exact qualified
schedule。
