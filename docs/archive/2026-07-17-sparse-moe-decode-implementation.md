# SparseMoe Decode Implementation Plan

> Status: completed on 2026-07-17 and archived after decode qualification. The complete 35B target
> remains future work.

This plan turned the accepted [`SparseMoe` design decisions](../plans/2026-07-17-sparse-moe-design-log.md)
into an executable first implementation phase. It covers only ordinary decode with `T=1`. Small-T
and prefill keep the same closed Op identity but remain outside this plan until their private
execution designs are discussed separately.

The governing mathematical contract is
[`qwen3.6-35b-a3b-architecture.md`](../qwen3.6-35b-a3b-architecture.md), the persistent codec and bank
contract is [`qwen3.6-35b-a3b-ninfer-artifact.md`](../qwen3.6-35b-a3b-ninfer-artifact.md), and the
coverage inventory is
[`qwen3.6-35b-a3b-operator-inventory.md`](../qwen3.6-35b-a3b-operator-inventory.md). This plan does
not replace any of those authorities.

## 1. Outcome and boundaries

The phase is complete: the repository contains one repository-internal `SparseMoe` Op whose
decode route:

- accepts the exact 35B `T=1` input, residual, and five registered weight roles;
- executes the accepted D1-D4 mathematical responsibilities with exactly four launches;
- supports main Text Q4 routed gate/up with Q5 or Q6 routed down, plus the MTP W8/W8 routed route;
- uses the artifact's direct expert-bank addressing without selected-weight gathering or repacking;
- finishes the required `AddResidual` epilogue in D4;
- is CUDA Graph safe with caller-owned, stable workspace;
- passes the complete logical formula against one independent naive FP32/FP64 oracle; and
- has retained RTX 5090 evidence that every accepted sub-kernel has reached the applicable
  fixed-work, payload/decode, or candidate-envelope roofline for its exact domain.

This phase does not:

- add Small-T or prefill kernels, placeholders, dispatch branches, tests, or benchmark rows;
- register the 35B Engine target or change the current product target;
- expose router scores, selected ids, route weights, expert jobs, activations, or partials as Ops;
- implement L15-L19 or M1-M3 as independent target-callable entry points;
- add generic sparse-MoE abstractions, model-family APIs, capacity/drop policies, or all-expert
  fallbacks;
- claim end-to-end 35B model speed before the target Program exists; or
- require a production arithmetic path to reproduce a Python, unfused, or BF16-staged reference.

The work proceeds sub-kernel by sub-kernel, but correctness remains complete-Op correctness. A
simple complete D1-D4 baseline is established first so that optimizing one private kernel never
requires turning its handoff into an observable stage golden.

## 2. Fixed semantic and storage domain

For normalized BF16 `X[2048,1]` and BF16 destination/residual `R[2048,1]`, the Op computes:

```text
scores = router_shared_gate * X

I = top8(scores[0:256])
alpha[e] = exp(scores[e]) / sum(j in I, exp(scores[j])), e in I
shared_scale = sigmoid(scores[256])

act_e = SwiGLU(routed_gate_up[e] * X), e in I
act_s = SwiGLU(shared_gate_up * X)

Y = sum(e in I, alpha[e] * routed_down[e] * act_e)
    + shared_scale * shared_down * act_s

R' = R + Y
```

At an exact top-8 boundary tie, lower expert id wins. Every selected weight stays associated with
its expert id. There is no capacity limit, token drop, stochastic routing, auxiliary-loss term, or
logical expert permutation.

The exact decode storage combinations are:

| Role | Main Text | MTP |
|---|---|---|
| router/shared score | BF16 `[257,2048]` | BF16 `[257,2048]` |
| routed gate/up | Q4 `[256*1024,2048]` | W8 `[256*1024,2048]` |
| routed down | Q5, or Q6 in layers 34/38/39, `[256*2048,512]` | W8 `[256*2048,512]` |
| shared gate/up | W8 `[1024,2048]` | W8 `[1024,2048]` |
| shared down | W8 `[2048,512]` | W8 `[2048,512]` |

Expert `e` directly selects rows `e*1024:(e+1)*1024` in routed gate/up and
`e*2048:(e+1)*2048` in routed down. Gate rows precede up rows within every 1024-row expert view.

## 3. Numerical execution rule

Public BF16 input, BF16 destination, persistent weight representation, exact codec decode, top-8
tie behavior, and final BF16 writeback are semantic. Every D1-D4 handoff is private.

Consequently:

- D1 may hand FP32 scores directly to D2;
- D3 may hand its natural FP32 SwiGLU results directly to D4;
- no cast is inserted merely to reproduce an unfused BF16 materialization;
- private workspace dtype, reduction association, scale placement, and instruction operands remain
  implementation profiles; and
- a candidate is checked from the represented public inputs against the complete high-precision
  oracle, not against another production route.

The first decode profile uses FP32 scores, FP32 route weights/shared scale, and FP32 slot-major
activations because those are the natural producer results and the complete handoff is below
20 KiB. This is an implementation starting point, not a new Op constraint. A lower-precision
handoff may be considered only if the completed D3+D4 sequence demonstrates a material benefit and
qualifies independently against the same complete oracle.

## 4. Repository and component architecture

The implementation stays in the existing `ninfer_ops` library with explicit CMake source
registration.

```text
include/ninfer/ops/
    one SparseMoe semantic contract

src/ops/wrapper/
    one validation, workspace, and dispatch wrapper

src/ops/sparse_moe/
    private shared plan/workspace machinery
    decode implementation subtree

tests/ops/
    one complete SparseMoe numerical test and independent support code

bench/ops/
    one SparseMoe benchmark with complete-Op and stage scopes
```

No Small-T or prefill directory is created during this phase. Exact source filenames,
translation-unit splits, and helper-header depth are chosen when the implementation makes those
boundaries concrete.

### 4.1 Public repository-internal contract

The semantic header contains only:

- `SparseMoeWeights`, aggregating `router_shared_gate`, `routed_gate_up`, `routed_down`,
  `shared_gate_up`, and `shared_down`;
- `SparseMoeEpilogue`, currently admitting only `AddResidual`;
- a caller-capacity workspace query; and
- `sparse_moe(...)`.

`x` and all five weight views are read-only. `destination` is the sole observable mutation: its
incoming BF16 values are the residual and its outgoing BF16 values are `residual + SparseMoe(x)`.
It must not overlap `x`, weight storage, or workspace; workspace must not overlap any public
operand. Execution is enqueued on the supplied stream without a host synchronization.

The first workspace query admits only `max_tokens=1`. Its shape should remain the same kind of
capacity query used by existing fixed-domain Ops; it must not require a target role, layer id,
decode flag, or externally selected kernel route.

The wrapper validates exact tensor shapes, dtypes, contiguity, weight shapes, codecs, row-split
planes, group sizes, scale type, pointer alignment, non-overlap, epilogue, and workspace capacity.
It rejects `T!=1` in this phase. It does not contain CUDA mathematics.

### 4.2 Private decode plan

Private code may use finite enums for D1, D2, D3, and D4 candidates and one closed decode plan that
records:

- the four selected schedules;
- the private activation profile;
- required workspace bytes.

Codec specializations resolve directly from the already validated registered weight views rather
than being duplicated in the plan.

The experimental plan temporarily recorded whether D1+D2 was fused. The accepted plan removed
that field together with the rejected fusion route.

Production dispatch resolves only the accepted exact route. Tests and benchmarks may resolve a
fixed candidate so one kernel can be varied while the other three use the correctness baseline or
previously accepted schedules. There is no string-driven production registry and no generic
fallback.

### 4.3 Workspace views and lifetimes

The workspace layout is defined once in the SparseMoe subtree and shared by the query, executor,
tests, and benchmark. Bench code must not reproduce offsets independently.

The decode lifetimes are:

```text
route metadata:
    ids[8]                 I32
    alpha[8]               FP32
    shared_scale           FP32
    live from D2 through D4

phase scratch union:
    D1 scores or split partials
    D3/D4 act[9][512]
```

The separate D1 route writes FP32 scores; after D2 consumes them, D3 may reuse the same capacity for
the activation handoff. The initial D3/D4 activation view is FP32 `[9,512]`, slot-major and aligned
for vector access. A workspace scope is acquired and restored inside one Op call. Allocation and
addresses remain deterministic under CUDA Graph capture.

The evaluated draining-fusion candidate additionally allocated one I32 completion counter. That
candidate was rejected because caller-owned transient workspace has no defined initial state; the
accepted workspace therefore contains no counter or persistent cross-call state.

## 5. Correctness foundation

### 5.1 Correct the target reference and establish the one Op oracle

The current Python MoE target reference explicitly converts normalized route weights to BF16 and
performs several private BF16 materializations. That behavior may describe a historical execution
profile, but it is not the mathematical oracle required by this Op.

Before any optimized candidate is accepted:

1. correct the active Python target reference so its SparseMoe mathematics uses FP32/FP64 from
   represented inputs and exact decoded weights instead of prescribing those private BF16
   boundaries;
2. remove test expectations that prescribe BF16 route weights or BF16 private activation/down
   boundaries; and
3. establish one independent C++ test oracle for the complete Op, without calling production CUDA
   helpers or copying their reduction order.

The C++ oracle may reuse test-only exact row-split payload decoding, but it must not use a production
decode atom or materialize all 256 expert banks. It decodes only the eight selected expert spans.
The Python target reference remains supplementary target-parity evidence; it is not a second Op
oracle and its output is not the C++ test expected value.

For each case, the C++ oracle starts from the represented BF16 input and residual, decodes every
used signed code with its stored FP16 scale into `double`, evaluates the complete router,
selection, SwiGLU, down, merge, and residual formula naively in `double`, and applies the declared
BF16 output conversion. The test compares the production BF16 destination with named tolerances
for the Q4+Q5, Q4+Q6, and W8+W8 implementation profiles. Those criteria are fixed from the
registered domain before candidate tuning; they are not loosened to match a production route or
derived from candidate-to-candidate parity.

### 5.2 Correctness-first complete decode baseline

Before D1 optimization begins, implement a simple, structurally conforming four-launch route:

- one D1 projection launch;
- one exact D2 selection launch;
- one selected-routed-plus-shared D3 launch; and
- one selected-routed-plus-shared D4 launch with final `AddResidual`.

The baseline may use straightforward synchronous loads and unoptimized CTA geometry, but it must:

- preserve natural FP32 private results;
- use direct device expert addressing;
- avoid one launch per expert and selected-weight copies;
- support all three registered codec combinations; and
- be correct under CUDA Graph replay.

It is a test/candidate control, not a performance-qualified production fallback. After all four
stages qualify, remove baseline variants that no longer provide a meaningful benchmark control or
correctness purpose.

### 5.3 Permanent numerical test

The permanent test calls a complete decode route and compares only the observable BF16 destination
with the complete oracle. While one stage is under development, its candidate is combined with
baseline or already accepted implementations of the other stages.

Required cases are limited to realistic regressions:

- main Q4 gate/up plus Q5 down;
- main Q4 gate/up plus Q6 down;
- MTP W8 gate/up plus W8 down;
- selected expert ids spanning bank boundaries, including 0 and 255;
- an exact top-8 boundary tie whose lower-id choice changes the final output;
- a nonzero residual proving the in-place `AddResidual` effect;
- CUDA Graph capture and replay; and
- workspace-query/execution agreement at the returned capacity, plus essential invalid
  shape/format rejection.

The test does not compare score bits, route-weight bits, activation bits, workspace contents,
candidate-to-candidate parity, or a production route to the Python implementation. It does not add
near-tie, cancellation, or other artificial numerical-sensitivity cases without a reproduced bug.

### 5.4 Packed expert fixture

Tests and benchmarks allocate the complete physical address span required by the 256-expert
strides, but populate only the eight selected expert spans and required shared/router weights. This
checks direct addressing without constructing a multi-gigabyte host FP32 expert tensor. The oracle
decodes only populated selected rows.

## 6. Benchmark and profiler foundation

One benchmark executable covers the complete Op and private stages. Its conceptual controls are:

```text
scope:      full | d1 | d2 | d3 | d4
candidate:  production | one fixed private candidate
codec:      q4-q5 | q4-q6 | w8-w8
cache:      pipeline-cold | warm
profile:    timed matrix | one selected launch
```

Stage timing reproduces the actual producer/consumer cache relationship:

| Timed scope | Per-sample sequence |
|---|---|
| D1 | flush, then time D1 |
| D2 | flush, run D1 outside the event, then time D2 |
| D3 | flush, run D1-D2 outside the event, then time D3 |
| D4 | flush, run D1-D3 outside the event, then time D4 |
| full | flush, then time the complete route |

There is no flush between an untimed producer and the timed consumer. Thus the timed stage's
persistent weights are cold while its freshly produced handoff is hot, matching the intended
pipeline. Warm mode is diagnostic rather than the primary decode result.

Each row reports candidate identity, codec, grid/block geometry, cold median/min/p95, warm timing,
exact code/high/scale payload, useful and executed work, workspace bytes, build mode, GPU/toolchain,
and the matched control result. CSV output is retained for candidate comparisons.

### 6.1 Applicable rooflines

`1792 GB/s` nominal bandwidth is not a universal gate. Qualification uses the limiting resource of
the exact stage:

- D1: legal fixed-shape candidate lower envelope and a same-grid fixed-work/payload control;
- D2: single-CTA/warp launch floor and legal candidate lower envelope;
- D1+D2: direct sequence comparison against the two accepted separate launches;
- D3: same selected expert addresses, code/scale planes, and launch topology in a cold payload
  control, plus warm decode/issue evidence;
- D4: same K=512 codec payload, output-row tiling, producer-hot activation, merge, and epilogue
  envelope; and
- full: the accepted stage composition and launch count.

Nsight Compute is used only after timing identifies the winning candidate or an unresolved gap.
Relevant attribution is actual DRAM/L2 traffic, global transaction efficiency, SM/memory SOL,
waves, eligible warps, long-scoreboard stalls, registers, occupancy, and local/shared spilling.
There is no predeclared percentage that substitutes for this evidence.

## 7. Exact decode work model

| Stage/profile | Persistent payload | Useful dot work | Initial limiting hypothesis |
|---|---:|---:|---|
| D1 BF16 | 1,052,672 B | 526,336 FMA | launch/fixed work |
| D2 | about 1 KiB score input | 256-way top-8 plus 9 transcendentals | one-CTA launch floor |
| D3 main Q4+W8 | 10.625 MiB | 18,874,368 FMA | encoded payload/decode/scoreboard |
| D3 MTP W8+W8 | 19.125 MiB | 18,874,368 FMA | encoded payload |
| D4 Q5+W8 | 6.3125 MiB | 9,437,184 FMA | payload/decode/launch |
| D4 Q6+W8 | 7.3125 MiB | 9,437,184 FMA | payload/decode |
| D4 W8+W8 | 9.5625 MiB | 9,437,184 FMA | encoded payload |

At the approximately 1.50 TB/s same-machine cold-copy ceiling observed in the
[`35B Linear roofline report`](optimization-era/bench/qwen3.6-35b-linear-roofline.md),
ideal payload times are about 7.4 us for main D3, 13.4 us for MTP D3, 4.4/5.1 us for Q5/Q6 D4, and
6.7 us for MTP D4. These are planning lower bounds, not acceptance thresholds; online decode,
finite grid size, synchronization, and launch latency remain part of the exact operation.

## 8. D1: router and shared-score projection

### 8.1 Primary architecture

```text
grid                 257 CTAs
block                8 warps / 256 threads
CTA ownership        one output row
warp ownership       one contiguous K=256 slice
lane ownership       one uint4 = eight BF16 values
accumulator/output   FP32
```

Since `2048/(8*32)=8`, every thread performs one aligned 16-byte weight load, one aligned 16-byte X
load, and eight FP32 FMAs. Each warp reduces its K-slice, eight lane-0 partials are reduced inside
the CTA, and one FP32 score is stored. A CTA does not stage X because it consumes each X element
only once; X reuse across the 257 CTAs is left to cache.

This grid exposes 257 independent blocks rather than grouping rows into 33-65 blocks on a 170-SM
GPU.

### 8.2 Candidate progression

1. implement the 8-warp row-CTA candidate;
2. compare a 4-warp row-CTA neighbor to determine whether less synchronization beats lower K
   parallelism;
3. inspect the matched fixed-work control and NCU only if the winner has unexplained headroom; and
4. consider a 16-warp or cooperative BF16-MMA split-K candidate only when that evidence justifies
   the added reduction/workspace complexity.

Do not begin with tensor-core padding or a single CTA computing all 257 rows. D1 qualifies when the
complete Op is correct, the winner sits on the legal candidate/control lower envelope, Graph replay
works, and profiler evidence shows no removable structural limitation or spill.

## 9. D2: top-8, selected normalization, and shared sigmoid

### 9.1 Primary architecture

The primary candidate is one warp and one CTA:

1. lane `l` loads ids `l+32*j`, `j=0..7`;
2. each lane sorts its eight register candidates by `(score descending, id ascending)`;
3. eight warp winner reductions merge the 32 local lists into the global top-8;
4. lanes 0-7 evaluate shifted exponentials for only the selected logits;
5. a warp reduction produces the normalization denominator;
6. the eight associated `(id,alpha)` pairs are written; and
7. lane 0 computes the independent shared sigmoid.

The result is private `I32 ids[8]`, `FP32 alpha[8]`, and one FP32 shared scale. The kernel does not
materialize 256 softmax probabilities. Route-slot order remains private; an expert-id reorder is
considered only if D3/D4 measurements demonstrate value.

The existing finite-FP32 ordering-key logic used by sampling can become a narrow common comparator
now that a second Op needs the same value/lower-id ordering. SparseMoe must not depend on the full
sampling pipeline.

### 9.2 Candidate progression

The secondary candidate is a 256-thread hierarchical top-8: eight warp-local top-8 sets followed
by a final merge. A full 256-item block sort is a correctness/control implementation, not the
presumed production winner. D2 qualifies against the single-launch lower envelope and exact
selection behavior, not against DRAM throughput.

## 10. Optional D1+D2 launch fusion

D1 and D2 qualify separately before fusion is attempted. The credible fusion preserves D1's
distributed row parallelism:

1. launch the 257 D1 row CTAs;
2. each CTA stores its FP32 score, executes the required global visibility fence, and increments an
   Op-private completion counter;
3. the final CTA to increment the counter executes D2 using the complete score array;
4. it writes route metadata and resets the counter before exit; and
5. same-stream kernel completion protects the next Graph replay.

This threadblock-draining candidate saves one launch but not the approximately 1 KiB score handoff.
It is compared as a complete `D1+D2` sequence against the two accepted launches. Atomic contention,
memory ordering, counter reset, repeated Graph replay, and the last-CTA tail are part of the
qualification. A one-CTA 257-dot fusion and an always-resident cooperative grid are not baseline
candidates.

Outcome: the experiment reduced the Q4 front from about 11.3 us to 9.2 us, but required its
completion counter to be zero before the first call. Initializing caller-owned transient workspace
would add the launch/API work the fusion was meant to remove, and W8+W8 had no complete-route gain.
Fusion was rejected and its counter, kernel, and benchmark route were removed. Production uses
separate D1 and D2 launches.

## 11. D3: selected routed and shared gate/up SwiGLU

### 11.1 Common mapping

One CTA owns one intermediate coordinate `j`, giving 512 CTAs. The CTA stages BF16 `X[2048]` once:
256 threads each copy one aligned `uint4` into 4 KiB of shared memory. Expert ids address the
persistent routed bank directly.

The logical work is eight routed gate/up pairs and one shared gate/up pair. Every dot accumulates
in FP32; `SiLU(gate)*up` remains FP32 for the D4 handoff. Main routed Q4 and MTP routed W8 are
compile-time codec specializations, while shared gate/up is always W8.

### 11.2 Documented nine-warp control

```text
warp 0..7   one complete routed gate/up pair each
warp 8      one complete shared W8 gate/up pair
J tile      1
```

This is the direct existing baseline. Its likely imbalance is that one Q4 routed warp reads about
2176 B while the shared W8 warp reads about 4352 B, leaving the shared warp as a possible CTA tail.

### 11.3 Primary eight-warp balanced candidate

```text
block       8 warps / 256 threads
warp r      complete routed pair r
            plus shared gate/up K slice [256*r,256*(r+1))
```

For main Q4, each warp receives approximately 2176 B of routed work plus 544 B of shared W8 work.
Each warp produces its routed activation and two shared partials; one CTA reduction completes the
shared gate/up pair and shared activation. The same decomposition remains balanced for the MTP
W8-routed specialization.

K=2048 contains two K=1024 slabs. The first candidate uses aligned direct/vector loads or a shallow
two-stage pipeline. It does not inherit a deeper pipeline from an unrelated K merely because that
code exists. Q4 and W8 staging storage is sized by its own codec rather than allocating every warp
for the larger W8 case.

Outcome: the nine-warp mapping won for both routed codecs. The balanced candidate measured
20.480 us versus 18.432 us for Q4 and 26.624 us versus 22.528 us for W8. W8 D3 was improved by
giving all 32 lanes one value each; the same mapping did not benefit D4.

### 11.4 Candidate progression and gate

1. retain nine-warp, `J_TILE=1` as the matched control;
2. implement eight-warp balanced, `J_TILE=1` as the primary candidate;
3. use timing/NCU to decide whether direct loads or a two-stage pipeline wins; and
4. try `J_TILE=2` only if the `J_TILE=1` winner remains latency-bound and 256 blocks retain adequate
   occupancy. `J_TILE=4` is not an initial candidate because only 128 blocks would underfill 170 SMs.

D3 qualifies for both Q4+W8 and W8+W8. Its compute architecture may qualify before the FP32/BF16
handoff question is finally closed; the handoff is selected only with the completed D4 sequence.

## 12. D4: down projections, merge, and AddResidual

### 12.1 Required specialization

D4 uses exact compile-time routed codecs:

- Q5: eight G64 groups per K=512 row, 336 encoded bytes;
- Q6: eight G64 groups per K=512 row, 400 encoded bytes; and
- W8: sixteen G32 groups per K=512 row, 544 encoded bytes.

Shared down is W8. The hot loop does not branch on codec, and the K=512 row is never sent through a
generic K=1024 loop or scalar tail. Initial candidates use aligned direct, branch-free loads because
the complete rows are short; a pipeline is added only if measurement identifies memory-dependency
latency that the row tile cannot hide.

### 12.2 Nine-warp, one-row control

```text
block           9 warps
RowsPerCTA      1
warp 0..7       one routed path each
warp 8          shared path
grid            2048 CTAs
```

This is the documented direct baseline and preserves one final BF16 store without public expert
outputs.

### 12.3 Primary eight-warp, four-row candidate

```text
block           8 warps / 256 threads
RowsPerCTA      4
grid            512 CTAs
warp r          four adjacent rows for routed path r
                plus the same four rows of shared K slice [64*r,64*(r+1))
```

For each activation element, a warp updates four row accumulators. Persistent weight payload is
unchanged, activation L1 reads are reduced by roughly four, and 512 blocks retain about three block
waves across 170 SMs. Each warp produces four routed contributions and four shared partials. One
CTA reduction applies route weights, combines shared partials and shared scale, reads the BF16
residual, performs the final FP32 additions, and writes BF16 once.

The initial profile applies route/shared scales after each dot. Pre-scaling the FP32 D3 activation
is a later D3+D4 sequence candidate, not a separate semantic route.

Outcome: the nine-warp one-row mapping won for Q5, Q6, and W8. R2 did not improve the direct
control, and balanced R4 reduced grid parallelism and regressed every codec. R2 was removed; R4 is
retained only as a benchmark control.

### 12.4 Candidate progression and gate

1. measure nine-warp `RowsPerCTA=1` control;
2. implement eight-warp `RowsPerCTA=4` primary candidate;
3. if the result is ambiguous, use nine-warp R4 to isolate row-tiling benefit and eight-warp R1/R2
   to isolate shared-split and register/occupancy effects; and
4. consider R8 only if R4 remains latency-bound, because R8 leaves 256 blocks and raises the
   accumulator register footprint.

Do not implement the complete Cartesian product of warp counts, row tiles, activation dtypes,
scale positions, and codecs. Each additional candidate must answer a live measurement question.
D4 qualifies only after Q5, Q6, and W8 routed variants pass the complete oracle and their matched
performance envelopes.

## 13. Ordered implementation phases

### Phase 0: framework and correctness backbone

Deliver:

- semantic contract, wrapper, `T=1` admission, and five-role weight aggregate;
- private decode plan and lifetime-aware workspace;
- corrected high-precision Python target reference and the one independent C++ Op oracle;
- complete four-launch correctness baseline for all codec combinations;
- complete-Op test, CUDA Graph replay, and packed expert fixture;
- one stage/full benchmark with pipeline-cold timing and candidate selection; and
- explicit CMake registration in `ninfer_ops`, tests, and bench.

Exit when the baseline complete destination passes the oracle for all three codec combinations and
the benchmark isolates each of the four launches without changing the real handoff cache state.

### Phase 1: D1 qualification

Deliver and compare the 8-warp and necessary neighboring row-CTA candidates. Run each as
`candidate D1 + baseline D2-D4`. Retain a concise fixed-work report.

Exit when D1 has one accepted candidate, complete correctness and Graph replay pass, and matched
timing/profiler evidence explains its floor.

### Phase 2: D2 qualification

Deliver the one-warp register merge and only the control/secondary candidate needed to establish
the lower envelope. Run each with accepted D1 and baseline D3-D4.

Exit when exact top-8/tie behavior is protected through final output and D2 reaches its launch-floor
envelope.

### Phase 3: D1+D2 fusion decision

Implement the draining fusion only after separate D1 and D2 are accepted. Compare fused versus
separate sequences in the complete Op, including repeated CUDA Graph replay.

Exit with one explicit route decision: retain four launches or accept the fused first launch. Do
not keep both as unqualified production alternatives.

### Phase 4: D3 qualification

Implement main Q4+W8 and MTP W8+W8 using the nine-warp control and eight-warp balanced candidate.
Run each with the accepted front of the pipeline and baseline D4.

Exit when both codec profiles are correct, the selected candidate reaches the matched
payload/decode envelope without spill or structural tail, and the initial FP32 activation handoff
is recorded.

### Phase 5: D4 qualification

Implement exact Q5, Q6, and W8 K=512 codec specializations. Compare the documented R1 control with
the R4 primary candidate, expanding only where attribution requires it.

Exit when all three variants pass the complete oracle, direct final merge/AddResidual is protected,
and matched timing/NCU evidence explains the selected row tile and warp decomposition.

### Phase 6: complete decode cutover

Run D3+D4 sequence comparisons needed to confirm the activation representation and scale placement,
then resolve one production decode plan. Measure the complete Op under cold selected weights and
CUDA Graph replay. Remove obsolete non-control candidates.

Exit when:

- one production route covers Q4+Q5, Q4+Q6, and W8+W8;
- no unqualified fallback is reachable;
- complete correctness and relevant tests pass;
- all retained sub-kernel and full-Op claims have adequate RTX 5090 evidence;
- active artifact/inventory/design documents reflect the implemented result; and
- Small-T/prefill remain explicitly unsupported rather than silently using decode repeatedly.

## 14. Stage qualification record

The implementation work should maintain a compact table in this plan or the retained benchmark
report as stages complete:

| Stage | Accepted candidate | Correctness | Performance evidence | Status |
|---|---|---|---|---|
| D1 | 257 row CTAs, 8 warps | complete Op oracle and Graph replay | 5.216 us versus 5.344 us payload control | qualified |
| D2 | one-warp register top-8 | exact lower-id tie through final output | 6.176 us versus 14.368 us serial control | qualified |
| D1+D2 fusion | rejected; separate launches | candidate correct; initial counter contract invalid | about 9.2 us fused versus 11.3 us separate for Q4; no W8 full-route gain | closed |
| D3 Q4/W8 | nine path warps | Q4+Q5 and Q4+Q6 complete oracle | 18.432 us; payload 12.288 us; balanced 20.480 us; NCU-attributed | qualified |
| D3 W8/W8 | nine path warps, 32-lane W8 dot | W8+W8 complete oracle | 22.528 us; payload 16.384 us; balanced 26.624 us; no local spill | qualified |
| D4 Q5/W8 | nine path warps, R1 | Q4+Q5 complete oracle | 12.288 us, equal to payload control | qualified |
| D4 Q6/W8 | nine path warps, R1 | Q4+Q6 complete oracle | 14.336 us, equal to payload control | qualified |
| D4 W8/W8 | nine path warps, R1 | W8+W8 complete oracle | 16.416 us versus 16.384 us payload control | qualified |
| complete decode | four launches, FP32 private activation | all three profiles, residual, workspace, Graph | 40.960/43.008/49.120 us for Q4+Q5/Q4+Q6/W8+W8 | qualified |

The retained RTX 5090 methodology, candidate comparisons, and profiler attribution are in the
[`SparseMoe decode qualification report`](optimization-era/bench/qwen3.6-35b-sparse-moe-decode-roofline.md).

Qualifying L15-L19 private work does not by itself make M4 or the 35B target supported. Inventory
status changes must distinguish a qualified private domain from the completed closed Op and the
still-unregistered target.

## 15. Documentation and lifecycle

During implementation:

- record accepted candidate decisions and material deviations in the SparseMoe design log;
- update the artifact decode execution section only after measurement selects a topology;
- update inventory support rows only after their exact domains have evidence;
- retain concise benchmark/profiler reports only for claims that remain active; and
- do not turn raw experiment logs or discarded candidates into permanent authorities.

Stable API, numerical, launch, workspace, and support-status results were folded into the active
authorities. This completed plan is archived according to
[`docs/archive/README.md`](README.md); the active design log remains open for the later
Small-T and prefill discussions.
