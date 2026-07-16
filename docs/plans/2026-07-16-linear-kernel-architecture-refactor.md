# Linear Kernel Architecture

> Status: recommended architecture and accepted atomic implementation plan. The current dirty
> worktree is an evidence-producing prototype, not the product implementation baseline. It will be
> preserved for provenance, while the product cut-over is reimplemented from the last clean code
> baseline and integrated only as one complete architecture state. Per-policy roofline optimization
> follows that ownership cut-over.
>
> Scope: the registered `qwen3_6_27b_rtx5090` target. Future Qwen3.6-35B shapes are used only to
> test the design pressure; this document does not register 35B support.
>
> Evidence: the complete commands, rejected prototypes, measurements, profiler reports, and
> qualification limits are kept in the companion
> [experiment report](2026-07-16-linear-kernel-architecture-experiment-log.md). They are evidence
> for this design, not part of the design itself.

## 1. Architecture decision

NInfer Linear is a finite, target-optimized execution system, not a general GEMM framework. Its
execution chain is:

```text
target package statically binds TargetLinearProfile + compatible KernelCatalog
  -> target schedule
  -> semantic Op contract
  -> exact physical-problem admission
  -> Op-local planner
  -> one named execution policy
  -> one fixed launcher
  -> one complete-lifecycle kernel or explicit materialized composition
```

The corresponding implementation boundary is:

```text
Semantic Op
  owns mathematics, rounding, aliasing, and workspace semantics

Target profile
  owns exact checkpoint/GPU problems and their measured semantic-Op routes

Op planner and kernel catalog
  own resolution mechanics, policy capabilities, workspace semantics, and fixed launch closure

Execution policy
  names the complete schedule and its fixed launcher

Kernel lifecycle family
  owns staging, synchronization, warp roles, accumulation, and reduction

Closed device primitives
  provide encoding, path-specific decode, MMA, indexing, and finalization atoms
```

Five invariants define the architecture:

1. A quantized call is legal only when its effective physical weight view is explicitly
   registered by the active target profile. There is no arbitrary-shape fallback or cross-target
   shape union.
2. The semantic Op planner is the only routing authority. After planning, no launcher performs a
   second format, tile, or schedule selection.
3. A policy is a named whole execution bundle, not a runtime or compile-time Cartesian product of
   independently selectable format, tile, pipeline, and epilogue axes.
4. A reusable mainloop ends where producer lifetime, synchronization, warp ownership, or final
   reduction ownership changes. Reuse below that boundary is deliberately narrow.
5. Fusion is an Op semantic and policy decision. A typed finalizer is extensible, but there is no
   arbitrary runtime epilogue mechanism.

This model keeps exact specializations first-class. Generality is useful only when it preserves
the generated kernel and the measured winner.

## 2. Contracts and ownership

### 2.1 Semantic Ops are the public implementation units

Base Linear, LinearAdd, LinearSwiGLU, `linear_pair`, grouped projection, and GDN projection are
different semantic Ops. They can share kernels or primitives, but they do not share route
economics or workspace contracts merely because they contain a contraction.

The base contract remains conceptually:

```text
x   : contiguous BF16 [K,T]
w   : bound weight with logical shape [N,K]
out : contiguous BF16 [N,T]

out[n,t] = BF16(sum_k dequantize(w)[n,k] * float(x[k,t]))
```

Each fused Op additionally owns its exact BF16/FP32 rounding sequence. This matters because
`Linear -> BF16 -> Add`, the two independently rounded SwiGLU projections, and a dual-output pair
are not interchangeable with an arbitrary function applied to an FP32 accumulator.

The target package owns its immutable checkpoint/GPU profile: exact views, reachable token domain,
and measured route selection for each semantic Op. The target schedule owns which calls are made,
CUDA Graph lifetime, and arena reservation. The Op implementation owns the executable policy
catalog and semantic workspace rules. Target code must not communicate policy through layer names,
caller roles, or tensor names.

### 2.2 Admission, reachability, and qualification are separate

The design distinguishes three facts:

- **admission:** an exact physical problem is legal for an Op;
- **reachability:** a target schedule can call that problem at a particular `T`;
- **qualification:** a specific policy at that problem and `T` has correctness and performance
  evidence.

Conflating them either admits unused shapes or makes kernel planning depend on target call-site
knowledge. Current 27B admission is an exact finite inventory in one statically selected target
profile. Reachability remains target-owned, and qualification remains an evidence record attached
to policy points.

## 3. Problem and planning model

### 3.1 Closed physical identity

The planning identity is conceptually:

```cpp
enum class EncodingId {
    Q4G64RowSplitF16Scale,
    Q5G64RowSplitF16Scale,
    Q6G64RowSplitF16Scale,
    W8G32RowSplitF16Scale,
    DenseBF16,
    DenseFP32,
};

struct PhysicalProblem {
    EncodingId encoding;
    int32_t n;
    int32_t k;
    int32_t padded_k;
};
```

`EncodingId` is closed. Numeric traits and physical plane-layout traits may be separate compile-time
implementation details, but they are not independent runtime axes and arbitrary pairings do not
create new legal formats.

Admission uses the effective bound view, not its parent tensor identity. Alignment and required
planes are validated before planning. Current quantized product calls must bind their registered
views directly; kernels do not repack weights or allocate memory.

### 3.2 Kernel catalog and target profile are separate authorities

The Op-side catalog is a closed executable vocabulary. Conceptually, each typed policy descriptor
contains:

```cpp
struct PolicyDescriptor {
    PolicyId id;
    HardwareId hardware;
    LifecycleId lifecycle;
    CapabilitySignature capabilities;
    // Fixed launch closure and semantic workspace traits are statically associated with id.
};
```

This is not a runtime registry or function-pointer table. It may compile to traits and exhaustive
switches. Its purpose is to prove that a `PolicyId` is compatible with the selected hardware,
encoding, lifecycle topology, finalization form, and fixed launcher.

The target-side profile is immutable constexpr data selected by the target package:

```cpp
struct Route {
    TokenRange tokens;       // closed, contiguous interval
    PolicyId policy;         // final policy, not a hint
};

struct TargetOpProfile {
    PhysicalProblem problem;
    span<const Route> routes;
};

struct TargetLinearProfile {
    HardwareId hardware;
    span<const TargetOpProfile> linear;
    span<const TargetAddProfile> add;
    span<const TargetSwiGluProfile> swiglu;
    span<const TargetPairProfile> pair;
};
```

Base, Add, SwiGLU, and pair use typed profile records rather than one universal runtime schema.
The schematic structure above only shows the ownership relationship.

This separation is necessary even while only one product target exists. A future checkpoint can
reuse a kernel catalog without adding its shapes to the current target, and two GPUs can choose
different policies for the same physical problem. The target package statically passes its profile
to the Op planner, either directly or through a small target-owned execution context. There is no
string lookup, dynamic registration, or global “all known shapes” table.

### 3.3 One admitted problem, one complete route description

Each admitted target problem owns one explicit semantic-Op route profile:

```cpp
TargetOpProfile{physical_problem, complete_route_span}
```

The concrete code may store compact crossover facts and derive the route span, but every new
problem must provide a complete profile. There are no format-wide default thresholds from which a
new shape can silently inherit a route.

Joint target-profile/catalog validation proves:

- exactly one admission row exists;
- token intervals are ordered, contiguous, and non-overlapping over the legal envelope;
- every route names a policy owned by that semantic Op and compatible with the target hardware;
- every named policy's lifecycle supports its encoding, topology, and finalization form;
- every named policy has a fixed launch closure.

The planner returns a final execution plan:

```cpp
struct ExecutionPlan {
    PolicyId policy;
    WorkspacePlan workspace;
};
```

The generic Op planner resolves a typed target profile and then obtains semantic workspace facts
from the selected catalog policy. `WorkspacePlan` is part of the same route decision. Exact
scratch is used for one launch; range-capacity scratch is the maximum of every route reachable
within the reserved token range. This preserves stable CUDA Graph addresses even when the fastest
policy is non-monotonic in `T`.

### 3.4 Why finite tables are the right abstraction

The supported target set is intentionally small, while the best schedule depends on geometry,
tile waves, tails, encoding, and semantic topology. A global heuristic compresses the code but
does not compress the real decision surface; it merely hides it.

A finite target profile gives:

- exact product scope and fail-closed behavior;
- reviewable, reproducible routing;
- local extension for a new measured shape;
- no runtime autotuning or caller-role heuristics;
- freedom to retain an exact kernel when it wins;
- no accidental admission or route sharing between checkpoint/GPU targets.

The number of route rows is data complexity, not device-code complexity. Only named, reachable
policies are instantiated.

## 4. Execution policies and kernel lifecycle families

### 4.1 Policy is the optimization unit

A production policy names the complete measured bundle:

```text
encoding and geometry constraints
+ token/CTA ownership
+ CTA and warp topology
+ tile shape
+ producer and pipeline lifecycle
+ accumulator topology
+ finalization scope
+ fixed launcher
```

These fields are useful for reasoning and templates, but they are not freely composable API axes.
A policy exists only when one concrete bundle is implemented and measured. The executor switches
on the final `PolicyId` and launches it; it must not contain an `auto` path.

This design avoids two opposite failures:

- one giant configurable kernel whose unused options still affect registers, shared memory, or
  scheduling;
- unrelated copy-pasted kernels that cannot share proven zero-cost device mechanics.

### 4.2 Complete-lifecycle mainloop boundary

A mainloop owns the complete K-loop lifecycle:

- encoded-data and scale staging;
- shared-memory layout and lifetime;
- asynchronous copy and wait distance;
- prefetch and fragment buffering;
- CTA and warp roles;
- synchronization points;
- accumulator ownership and final K reduction.

If these properties differ, the implementations are different lifecycle families even when both
eventually execute MMA instructions. The recommended families are:

| Family | Purpose and boundary |
|---|---|
| exact T1/direct paths | geometry-specific row or split-K ownership for important decode/verify work |
| row-streaming Small-T | shared compile-time token bundles with path-specific quantized decode |
| split-low4 MMA | Q4/Q5/Q6 two-plane production and its own scale/stage lifetime |
| W8 MMA | one code plane, persistent scale staging, and distinct small-M/general warp topologies |
| dense | a future optimized dense lifecycle, separate from the correctness reference path |
| grouped expert | expert/job mapping and grouped ownership, separate from plain Linear |

In particular, split-low4 and W8 must not be forced behind a producer-hook mainloop. Their stage
count, scale lifetime, waits, prefetch, shared footprint, and residency tradeoffs are materially
different. A hook system capable of describing both would become a scheduling DSL and would put
the cost of optional state into hot kernels.

### 4.3 Narrow reusable substrate

Reuse is accepted at stable, compile-time seams:

- closed encoding and address facts;
- lifecycle-specific decode atoms;
- swizzles and fixed MMA atoms;
- accumulator indexing;
- typed lane-local or collective finalization helpers.

There are no runtime codec branches, device callbacks, virtual dispatch, or generic producer
objects in an inner loop. A new shared abstraction is justified only after at least two real
policy consumers demonstrate the same lifecycle and matched code generation, resources, and
performance. Otherwise duplication at the lifecycle boundary is cheaper and clearer.

## 5. Fused operations and epilogue extensibility

### 5.1 The abstraction seam

Fused behavior is described by four compile-time concepts:

```text
CtaProblemMap
  which physical input/output jobs a CTA owns before entering the K loop

AccumulatorTopology
  one accumulator set, paired halves, or multiple output sets

FinalizationScope
  lane-local or CTA-collective ownership after complete reduction

Finalizer
  the exact semantic rounding, terminal operation, and store
```

These are narrow implementation concepts, not a public epilogue framework. Each lifecycle has a
compile-time capability declaration describing the encodings, accumulator topologies,
finalization scopes, synchronization/store ownership, and CTA maps it can carry. A named policy
fixes one combination and is valid only when the lifecycle declares that combination supported.
Users cannot assemble arbitrary combinations at runtime, and the build does not instantiate their
Cartesian product.

If a fused behavior cannot run after the lifecycle's complete K reduction, needs unsupported
shared-memory or synchronization lifetime, changes producer/warp ownership, or crosses CTA
ownership, it is not an epilogue extension. It becomes a new lifecycle policy or an explicit
materialized composition. This capability check is the hard boundary that prevents typed
finalization from growing into a generic callback framework.

### 5.2 Three first-class execution forms

Every fused Op can choose among three forms per route:

1. **terminal finalizer:** reuse a lifecycle and apply a typed operation after the complete
   contraction reduction;
2. **topology-aware fused kernel:** change accumulator, CTA, or output ownership when fusion
   genuinely crosses the finalizer seam;
3. **materialized composition:** execute the base contraction, preserve its public rounding
   boundary, then run a separate semantic kernel.

Materialization is not a fallback failure. It is a first-class policy because fusion can reduce
launches yet lose on stores, tile tails, waves, registers, or occupancy. The Op-local planner
selects the measured winner rather than assuming fused is faster.

The current semantic examples establish the boundary:

- LinearAdd can use a lane-local or CTA-collective finalizer, but must preserve the projection's
  BF16 round before residual addition.
- LinearSwiGLU needs paired accumulator halves and two independent projection rounds; at some token
  ranges materialization wins over the folded topology.
- `linear_pair` changes output-job topology and owns routes independent of two base Linear calls.
- grouped projection changes the CTA problem map and therefore remains a typed topology, not
  baggage carried by every plain kernel.

### 5.3 Cost of adding a fused Op

A new fused Op pays only for what it uses:

1. define its mathematical and rounding contract;
2. define its typed accumulator/finalization topology;
3. instantiate a small set of named policies plus materialized competitors;
4. add its own route and workspace plan;
5. qualify the semantic seams and measured winners.

There is no global register or shared-memory tax on unrelated Linear kernels. The tradeoff is
intentional compile-time policy instances and an Op-local route table. This is a better cost model
for NInfer than a universal epilogue API whose flexibility would be paid by every kernel.

## 6. Complexity control and extension budget

The architecture controls complexity through closed vocabularies and local change surfaces:

| Change | Expected architectural impact |
|---|---|
| new shape using an existing lifecycle | one target-local measurement entry, then one exact target profile and measured routes |
| new encoding | one closed encoding registration, path decode support, and only the policies that consume it |
| new fused Op | one semantic contract, Op-local planner/workspace, and a finite set of named fused/materialized policies |
| new schedule winner | one policy and fixed launcher; existing profiles change only where evidence selects it |
| new lifecycle | a deliberate kernel-family implementation and qualification, without rewriting unrelated families |
| new target | one statically selected target profile; no shape union or role heuristics added to the Op planner |

The following are rejected regardless of apparent convenience:

- arbitrary quantized shape fallback;
- format-by-tile-by-pipeline-by-finalizer registration matrices;
- caller names or model roles in kernel routing;
- runtime epilogue descriptors or device function pointers;
- hidden allocation, weight repacking, or synchronization in kernels;
- generic source abstractions justified only by fewer lines of code;
- retaining unused policies or shapes for hypothetical compatibility.

The practical complexity test is local reasoning: adding one measured shape or fused Op must not
require understanding or recompiling an open-ended framework. If an extension changes the
lifecycle, that cost is made explicit as a new family instead of hidden inside optional branches.

## 7. Roofline strategy

### 7.1 What the architecture can guarantee

An architecture cannot guarantee roofline performance by construction. It can guarantee that the
performance unit is explicit, fixed, measurable, and replaceable. The relevant unit is:

```text
(semantic Op, exact physical problem, token range, PolicyId)
```

Roofline qualification is therefore per policy point or homogeneous route class, not one global
claim for “Linear.” Exact shape policies remain valid outcomes when they are needed to reach the
machine limit.

### 7.2 Required performance model

Every important policy records:

- useful contraction FLOP;
- actually executed FLOP including tile padding and tails;
- minimum logical bytes and measured DRAM/L2 traffic;
- CTA waves, partial-wave behavior, occupancy, registers, and shared memory;
- its product-time attribution.

This separates compute-limited full waves, bandwidth-limited T1, insufficient-parallelism cases,
weight rereads, and padded-tail loss. Useful TFLOP/s or a logical-byte estimate alone is not a
roofline conclusion.

### 7.3 Qualification ladder

Policies advance through four gates:

1. **semantic correctness:** an independent oracle at every decode, accumulation, and rounding
   seam;
2. **winner evidence:** matched candidate timing at real shapes, route boundaries, tails, and
   reversals;
3. **physical explanation:** NCU resources, traffic, waves, stalls, and executed work for selected
   fixed kernels;
4. **product value:** NSYS and real Engine evidence for routes with material end-to-end weight.

Optimization proceeds in product-attribution order. For each dominant policy, either the measured
resource roof is reached or the remaining limit is explained and competing schedules fail to
improve it. Kernel-count reduction and source uniformity are not performance objectives.

After the target-profile ownership cut-over, the architecture stage completes admission, fixed
routing, fusion seams, and the measurement method. The next stage optimizes current 27B policy
families in this order:

1. Text/MTP T1 and Small-T policies that dominate decode and verification;
2. full-wave W8 policies with physical weight rereads;
3. partial-wave and tail-heavy low4, pair, Add, and SwiGLU policies;
4. reachable high-T Vision routes.

## 8. Future 35B pressure test

Future support is evaluated by extension locality, not by pretending that arbitrary shapes already
work.

The experiments show that the existing closed quantized codecs and lifecycle vocabulary can
numerically execute the sampled non-grouped 35B geometries. They also show that routing is
shape-local and sometimes non-monotonic as CTA waves change. Consequently, 35B quantized support
cannot be inferred from one N/K threshold; it needs exact profiles and measured routes.

Dense BF16 and grouped experts are different conclusions:

- the current dense path is a correctness reference, not a competitive lifecycle;
- grouped experts change job mapping and ownership, so they require a separate lifecycle and
  planner work.

The admission workflow for a future target is:

1. inventory real effective views, encodings, alignments, reachable token sets, and fused Ops;
2. add them to a measurement-only vocabulary;
3. verify independent numerical oracles and benchmark existing fixed policies;
4. add a new policy or lifecycle when the current vocabulary does not win;
5. only then add exact product profiles, route closure, workspace, and target reachability.

Existing 27B profiles and kernels do not change merely because a future shape is introduced. A
35B target receives its own profile and selects from the compatible catalog. This is the intended
meaning of extensibility: bounded local additions with evidence, not universal shape support.
Qwen3.6-35B remains outside the current product contract.

## 9. Source ownership

```text
include/ninfer/ops/
  semantic Op contracts and public-internal workspace/alignment behavior

src/ops/linear/plan/
  generic typed resolvers, policy catalogs/capabilities, workspace semantics, and validation

src/ops/linear/codec/
  closed encoding registrations and path-specific decode atoms

src/ops/linear/gemv/ and gemm/
  complete lifecycle families and narrow shared device primitives

src/ops/linear/linear.cpp and src/ops/wrapper/
  validation, plan consumption, fixed launch closure, and explicit compositions

src/targets/qwen3_6_27b_rtx5090/
  exact Linear-family problem/route profiles, reachability, graph/state policy, and reservation

bench/ops/ and tests/ops/
  measurement-only candidates, independent numerics, and policy qualification
```

Source filenames are not architectural boundaries. They should be renamed only when real
ownership work touches them; cosmetic uniformity does not justify churn.

## 10. Evidence that determined the design

The experiment report records the full proof. The architecture depends on these conclusions:

| Hypothesis | Result used by this design |
|---|---|
| one generic quantized fallback | rejected: it admitted irrelevant shapes and concealed tuning gaps |
| one global token threshold | rejected: winners change with encoding, shape, waves, tails, and semantic topology |
| one configurable MMA mainloop | rejected: split-low4 and W8 have materially different lifecycles and residency constraints |
| closed codec/decode substrate | accepted: useful source reuse was obtained without hot-path code-generation cost |
| one universal T1 policy | rejected: lower-resource variants were not universal winners |
| arbitrary fused epilogue | rejected: Add, SwiGLU, pair, and grouped projection cross different ownership and rounding seams |
| finite typed finalization/topology | accepted: it supports direct, collective, paired, and materialized winners without taxing unrelated kernels |
| generic future-shape heuristic | rejected: sampled future quantized routes were shape-local; dense/grouped work needs new lifecycles |
| per-policy roofline model | accepted: NCU exposed compute roofs, bandwidth limits, under-wave work, tails, and physical rereads hidden by aggregate metrics |

These conclusions are architectural; exact current route thresholds and benchmark values remain in
the experiment report because they are tunable evidence, not stable design principles.

## 11. Prototype status and product acceptance

The current dirty worktree proves the important architectural hypotheses, but it is deliberately
classified as an experimental prototype rather than the final product implementation. It contains:

- exact current-27B quantized admission and measured route data;
- Op-local fixed-policy planning experiments for base, Add, SwiGLU, and pair;
- closed codec/decode reuse and distinct low4/W8 lifecycle evidence;
- typed direct/collective finalization, paired topology, and materialized composition;
- measurement-only future shapes with no accidental product admission;
- the benchmark, correctness, NSYS, and NCU evidence recorded in the companion report.

The prototype still places target problem and route facts in Op-global planners and was developed
incrementally across the old ownership boundary. Continuing to repair that tree would make it hard
to distinguish the intended architecture from transitional structure. The product implementation
therefore starts from the last clean code baseline. The prototype is retained as reproducible
evidence and a behavioral/performance oracle, not used as the cut-over base and not bulk
cherry-picked into the new implementation.

The product architecture is accepted when:

- every registered product call resolves exactly one target-owned support profile and one fixed
  Op-owned policy;
- no unregistered quantized call, global target union, or caller-role heuristic reaches a
  production kernel;
- every fused policy preserves its Op's mathematical rounding and range-capacity workspace
  contract;
- every target route is jointly validated against the selected hardware catalog and a fixed
  launcher;
- the new implementation preserves the prototype's qualified route manifest, numerical behavior,
  kernel resources, and matched product performance;
- future support enters only through the measurement-first admission workflow.

## 12. Atomic implementation plan

### 12.1 Repository states and provenance

Implementation uses three explicit repository states:

```text
clean code baseline
  last committed product code before the experimental Linear work

archived experiment
  the complete prototype worktree, experiment report, and raw profiler provenance

clean implementation
  a fresh implementation of this architecture based on the clean code baseline
```

The final architecture document, experiment report, and raw `profiles/bench`, `profiles/nsys`, and
`profiles/ncu` evidence are retained. Prototype product code, tests, and benchmark tools are
captured on an archive branch for comparison, but they are not the source base of the clean
implementation. Project instructions and unrelated user changes are preserved independently and
are never absorbed into the Linear archive or reset as part of the cut-over.

### 12.2 Meaning of an atomic cut-over

The integrated product tree may contain either the complete old architecture or the complete new
architecture. It must never retain a mixed product state. Development may proceed in an isolated
branch or worktree, but integration occurs only after all callers and authorities have switched.

The final cut-over has none of the following:

- profile-less and profile-aware quantized Linear APIs living together;
- some semantic Ops using target profiles while others use Op-global routes;
- a new planner falling back to the old planner or an automatic launcher;
- two support tables, compatibility aliases, feature flags, or transitional dispatch;
- partially migrated Text, Vision, MTP, workspace-reservation, benchmark, or test callers.

Intermediate implementation commits may be used on the isolated branch when they are complete,
reviewable internal boundaries. Before integration they are squashed or arranged so that the
product-facing cut-over itself is one complete commit. No half-migrated commit is merged.

### 12.3 Minimal profile and catalog contract

The Op layer defines non-owning typed profile views; the target owns their immutable backing data.
The schematic aggregate is:

```cpp
enum class LinearHardwareId { Sm120a };

struct LinearExecutionProfile {
    LinearHardwareId hardware;
    std::span<const LinearProblemProfile> linear;
    std::span<const LinearAddProblemProfile> add;
    std::span<const LinearSwiGluProblemProfile> swiglu;
    std::span<const LinearPairProblemProfile> pair;
};
```

Each problem-profile type has its own typed policy and route record. This aggregate is an execution
context, not a universal route schema. The registered target instantiates one constexpr profile;
a future target instantiates another without modifying the current target's support domain.

The catalog remains static and closed. It is implemented with typed policy enums, constexpr
capability traits, compile-time validation, semantic workspace functions, and exhaustive fixed
launch switches. It is not a runtime registry, virtual interface, function-pointer table, or
template Cartesian product.

All quantized Linear-family calls explicitly receive the selected execution profile, including
workspace-capacity queries and materialized compositions. There is no default profile, target
lookup by string, profile stored in `Weight`, or mutable global selection.

### 12.4 Compile-time joint validation

The target profile and catalog are accepted together with compile-time checks proving:

1. every effective physical problem appears exactly once in its semantic-Op profile;
2. every problem owns a nonempty route span starting at one and ending at the target token ceiling;
3. route token intervals are ordered, contiguous, and non-overlapping;
4. every route names a policy from the correct semantic Op;
5. the selected hardware catalog accepts that problem/policy pair;
6. every named policy has one fixed launch closure and valid workspace semantics;
7. materialized Add and SwiGLU routes reference a base problem admitted by the same target profile;
8. measurement-only future problems are absent from the registered 27B profile.

### 12.5 Isolated implementation sequence

The clean implementation is built in these internal work packages, then exposed as one cut-over:

1. define physical-problem, typed profile, execution-plan, workspace-plan, and catalog-capability
   contracts without wiring product callers;
2. re-establish the qualified fixed kernel vocabulary and narrow codec/finalizer/topology seams,
   using the prototype only as evidence and a comparison oracle;
3. instantiate the complete target-owned 27B base, Add, SwiGLU, and pair profiles from the measured
   route manifest;
4. switch every Text, Vision, MTP, wrapper, workspace-layout, benchmark, and test caller to explicit
   profile injection in the same product-facing change;
5. delete the old caller-role regimes, Op-global product routes, arbitrary quantized fallback,
   hidden second dispatch, unreachable Decode shapes, and all transition code;
6. validate structure, route identity, independent numerics, real-artifact behavior, matched
   microbenchmarks, and representative profiler resources before integration.

The qualified manifest expected at cut-over is 24 base problems and 83 base routes, two Add
problems and eight routes, one SwiGLU problem and seven routes, and one pair problem with three
routes. These counts protect the current target contract; they do not become a generic framework
limit.

### 12.6 Verification and commit gates

The atomic implementation is not accepted until all of the following hold:

- source ownership scans find no 27B exact route data in `src/ops`, no CUDA launcher in the target,
  no profile-less quantized API, and no old regime/fallback authority;
- compile-time profile/catalog validation and route-manifest tests pass;
- registered and rejection tests, independent FP64/BF16 operator oracles, Add/SwiGLU rounding,
  pair, Vision, MTP, token-ceiling, alignment, and range-capacity workspace checks pass;
- the real 27B artifact completes the affected Text, Vision, and MTP product routes;
- same-session comparison against the archived prototype preserves fixed policy selection, kernel
  resources/SASS where the abstraction must be zero-cost, planner cost, representative
  microbenchmarks, and end-to-end latency within measurement noise.

Reasonable commits are made only at durable boundaries: the architecture/provenance baseline, the
complete archived prototype, and the complete verified product cut-over. Private implementation
checkpoints may exist during development, but no partially migrated state is presented as the
product architecture.

This cut-over deliberately excludes 35B product registration, optimized dense and grouped-expert
lifecycles, runtime autotuning, source-tree cosmetic rewrites, and completion of every Stage-B
roofline optimization. Per-policy roofline optimization remains the next performance stage, not an
unresolved architecture decision.

The detailed experiment report remains the evidence ledger; this document is the stable
recommended design and supersedes the original draft.
