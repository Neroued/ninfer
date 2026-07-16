# Linear Kernel Architecture

> Status: implementation in progress. Q4, Q5, and Q6 pure Linear now use independent
> format-local backends; W8 and dense remain on compatibility paths until their own migrations.
>
> Correction: the earlier target-owned `LinearExecutionProfile` design is rejected. Passing a
> target/profile into Linear-family Ops leaks dispatch ownership across the semantic Op boundary.
> The original Op interfaces remain unchanged; exact admission, planning, workspace semantics,
> and fixed-launch selection are entirely internal to each Op.
>
> Scope: the current `qwen3_6_27b_rtx5090` product and its RTX 5090 execution backend. Future
> Qwen3.6-35B shapes are design-pressure inputs only; this document does not register or advertise
> a 35B product target.
>
> Evidence: commands, failed designs, route measurements, numerical checks, NSYS attribution, NCU
> reports, and qualification limits remain in
> [the experiment report](2026-07-16-linear-kernel-architecture-experiment-log.md). The archived
> prototype is evidence and a comparison oracle, not an implementation base.

## 1. Architecture decision

NInfer Linear is a finite, hardware-qualified semantic Op, not a target-configured dispatch
service and not a general GEMM framework.

The public repository-internal execution interface remains:

```cpp
void linear(const Tensor& x,
            const Weight& w,
            Tensor& out,
            WorkspaceArena& ws,
            cudaStream_t stream);
```

The caller expresses only the semantic operation and provides its operands, transient arena, and
stream. It does not provide a target, profile, policy, candidate, shape family, hardware selector,
or epilogue descriptor.

The complete execution chain is:

```text
target schedule
  -> linear(original semantic inputs)
  -> validate common tensor semantics and physical weight facts
  -> switch on the weight format
  -> q4 / q5 / q6 / w8 / bf16 format backend
  -> normalize one format-local physical problem
  -> format-local RTX 5090 route resolver
  -> one format-local final execution plan
  -> one fixed launcher or one explicit fixed composition
```

The same rule applies independently to `linear_add`, `linear_swiglu`, `linear_pair`, grouped
attention input projection, and grouped GDN input projection.

Seven invariants define the architecture:

1. **The semantic API is target-independent.** No Linear-family execution or workspace API accepts
   target-owned dispatch data.
2. **The Op owns its complete support domain.** A quantized call is legal only if the Op's internal
   hardware backend contains an exact physical-problem route for it.
3. **The selected format backend is the sole routing authority.** After resolution, no launcher
   performs a second format, shape, tile, or schedule choice.
4. **Unknown production problems fail closed.** There is no arbitrary quantized fallback, shape
   similarity rule, global token threshold, or caller-role heuristic.
5. **A policy is a complete execution bundle.** It names a lifecycle, topology, finalization form,
   workspace behavior, and fixed launcher rather than a Cartesian product of independent knobs.
6. **Fusion belongs to the fused semantic Op.** Base Linear does not expose an arbitrary epilogue
   interface.
7. **Admission implies qualified routing over a declared envelope.** An unmeasured conservative
   tail is not presented as the best production route.

The planner guarantees the best qualified production policy among the evaluated viable candidates
for an admitted physical problem and token range. It cannot prove that no undiscovered kernel
could ever be faster; that remains an optimization process rather than an API concern.

## 2. Semantic boundary and ownership

### 2.1 The caller does not participate in dispatch

The original Linear contract is the correct abstraction boundary:

```text
x   : contiguous BF16 [K,T]
w   : bound physical weight with logical shape [N,K]
out : contiguous BF16 [N,T]

out[n,t] = BF16(sum_k dequantize(w)[n,k] * float(x[k,t]))
```

`Weight` already contains physical storage facts such as qtype, layout, dimensions, padded
dimensions, group size, scale dtype, and plane pointers. Linear may inspect and validate those
facts because they are part of the operand. It must not require the caller to attach execution
policy.

In particular, `Weight` must not gain:

- a target identity;
- a `PolicyId`;
- a route or crossover table;
- a launcher pointer;
- workspace requirements;
- an opaque target profile.

A `Weight` may be sliced into effective row views, reused by different semantic Ops, and executed
at different token counts. A cached policy would be stale or semantically ambiguous under those
operations.

### 2.2 Ownership split

The correct ownership boundary is:

| Owner | Responsibilities |
|---|---|
| `src/targets/<target>` | checkpoint inventory, storage binding, effective weight views, schedules, reachability, graph/state policy, arena reservation, product integration |
| semantic Op | mathematics, rounding, aliasing, exact supported physical problems, route resolution, workspace semantics, fixed execution closure |
| format backend inside the semantic Op | format-specific storage/decode semantics, RTX 5090 route winners, legality capabilities, kernel resources, and fixed launcher implementations |
| core/runtime | tensors, arenas, device/stream lifetime, graph capture, and public Engine behavior |
| benchmark/test tools | candidate forcing, measurement-only future problems, independent numerical oracles, and qualification evidence |

The target decides **which Linear calls exist**. Linear decides **whether each physical call is
supported and exactly how it executes**.

Target code must not contain:

- Linear policy enums;
- route intervals or crossover thresholds;
- kernel launcher selection;
- profile injection;
- model-role strings used to influence Linear dispatch.

Op code must not require a model target name to choose a kernel.

### 2.3 Admission, reachability, qualification, and product support

Four separate facts must not be conflated:

- **Op admission:** the hardware-specific Op backend supports an exact physical problem over a
  declared token envelope.
- **target reachability:** a target schedule can produce that problem and call the Op at a token
  count.
- **qualification:** the selected route has numerical, performance, and when necessary physical
  profiler evidence.
- **product support:** an Engine target binds the required tensors and exposes the behavior through
  the product.

This distinction resolves the earlier cross-target concern.

If a future 35B target needs a new exact Linear problem, that problem is added to Linear's supported
physical domain after qualification. Once added, the Op can execute that physical problem
regardless of which model produced it. This is correct: identical physical operands on the same
hardware should not choose different kernels merely because they came from different targets.

If two apparently identical calls measure different winners, the design must first find the
missing physical variable—encoding, padded layout, alignment, token range, execution mode, or
hardware class. Target identity is not an acceptable substitute for an incomplete problem key.

Product isolation remains target-owned: a target that does not bind or schedule a problem does not
support that model path, even if an internal Op is capable of executing the same physical shape.

## 3. Exact physical problem and internal planning

### 3.1 Format-local physical identity

Linear does not normalize every encoding into one cross-format planner key. After common semantic
validation it dispatches exactly once on `Weight::qtype`. Each format backend then defines the
smallest physical problem needed by that encoding:

```cpp
struct Q5Problem {
    int32_t rows;
    int32_t k;
    int32_t padded_k;
    int32_t cols;
};

struct Q6Problem {
    int32_t rows;
    int32_t k;
    int32_t padded_k;
    int32_t cols;
};
```

These types are intentionally separate. Equal fields do not imply shared admission, schedules,
decode atoms, kernel bodies, or route tables. W8 and dense will receive the same ownership shape
when their compatibility paths migrate.

The real implementation may include additional physical facts only when they alter legality or
the generated execution, for example a required alignment class or storage encoding revision.

The key must not contain:

- model or target names;
- layer roles such as MLP, attention, or head;
- Text/Vision/MTP phase;
- tensor binding names;
- caller-selected backend names.

Policy names may mention exact geometry when the kernel is geometry-specialized, but routing still
uses physical facts rather than caller roles.

### 3.2 Format-owned immutable route catalogs

Each RTX 5090 format backend owns its own closed compile-time catalog:

```cpp
struct Q5ColsSet {
    int32_t first;
    int32_t last;
    int32_t step;
};

struct Q5RouteSpec {
    Q5ColsSet cols;
    Q5ScheduleId schedule;
};

struct Q5SupportSpec {
    int32_t rows;
    int32_t k;
    int32_t padded_k;
    Q5ColsSet admitted_cols;
    // span into the Q5 route table
};
```

The catalog maps an exact effective weight view and token set to a final format-local schedule. It
is:

- immutable;
- compiled into the Op;
- hardware-specific;
- finite and reviewable;
- free of runtime registration and initialization order;
- free of virtual calls, function-pointer registries, and string lookup.

It is not a target profile and not a cross-format registry. It describes the physical execution
capability of one weight format on the current hardware backend.

The preferred lookup is a compact constexpr table or nested switch inside that format directory. A
global `LinearPolicyId`, hash table, mutable cache, or target-selected registry is unnecessary.

### 3.3 One resolution, one final plan

The internal execution sequence is:

```cpp
void linear(const Tensor& x, const Weight& w, Tensor& out,
            WorkspaceArena& ws, cudaStream_t stream) {
    validate_semantics_and_storage(x, w, out);
    switch (w.qtype) {
    case QType::Q4G64_F16S:
        return q4_rowsplit_dispatch(x, w, out, stream);
    case QType::Q5G64_F16S:
        return q5_rowsplit_dispatch(x, w, out, stream);
    case QType::Q6G64_F16S:
        return q6_rowsplit_dispatch(x, w, out, stream);
    // W8 and dense use compatibility paths until their format migrations.
    }
}
```

Each backend has its own plan type:

```cpp
struct Q5Plan {
    Q5ScheduleId schedule;
    Q5KernelVariant variant;
};
```

The format-local fixed launcher uses an exhaustive switch. Each case launches one fixed
implementation. It may validate assumptions, but it may not choose another tile size, token tile,
mainloop, split factor, or fallback kernel.

The following old structures are rejected:

- `ShapeFamily`;
- `LinearRegime`;
- one global Small-T/Large-T threshold;
- automatic launchers that reclassify after the planner;
- generic quantized fallback;
- caller-role specialization as route authority.

### 3.4 Policy legality and route selection are separate internal facts

Two Op-owned authorities remain distinct:

1. the **route catalog** records which qualified policy wins for an exact problem and token range;
2. the **policy capability catalog** records whether a fixed policy is physically legal for a
   format, geometry, token range, alignment contract, and hardware backend.

Compile-time validation proves:

- every support problem is unique;
- every route interval is ordered, contiguous, and non-overlapping;
- every admitted interval lies within the declared supported envelope;
- every route names a policy owned by the correct semantic Op;
- the policy capability accepts that exact problem and range;
- every policy has one fixed execution closure;
- workspace semantics exist for every policy;
- no `Default`, `Auto`, `Fallback`, or `Unknown` policy can appear in production routes.

Capability does not imply admission. A geometry-generic TT4 or MMA kernel may be physically capable
of running many shapes, while the production resolver admits only shapes that have completed the
measurement and correctness workflow.

### 3.5 Supported token envelope

Each exact physical problem declares only the token envelope for which production routing is
qualified.

The previous proposal used the CUDA grid limit as a universal terminal route even where winner
measurements stopped much earlier. That is rejected. Grid legality proves that a launch can run; it
does not prove that Linear selected the best qualified kernel.

For each admitted problem:

- the supported envelope must cover the current product's declared reachable domain;
- candidate comparison must include route boundaries, tile-wave changes, important tails, and
  representative high-token points;
- the final route may use a structurally stable policy over a broad interval only when evidence
  justifies that conclusion;
- outside the qualified envelope, the Op rejects the call.

If the product needs a larger prefill or Vision token range, qualification expands before the
public target limit expands. The product limit must not silently outrun the Op's qualified domain.

### 3.6 Hardware selection remains internal

The current product supports exactly RTX 5090 and builds CUDA for `sm_120a`; target preflight
already rejects other devices. Therefore the current Linear implementation may compile and link
one RTX 5090 backend without adding hardware to the API or querying device properties on every
call.

A future additional GPU backend must remain an Op/runtime-internal concern and requires separate
design review. It must not reintroduce target profiles into semantic calls. Possible future
mechanisms include separate product builds or an immutable device backend chosen once by core, but
neither is current scope.

### 3.7 Dense reference code is not a production fallback

The requirement that Linear choose a qualified best kernel is incompatible with silently routing
arbitrary dense shapes to a slow reference implementation.

Generic dense BF16/FP32 implementations may remain as:

- internal numerical oracles;
- benchmark candidates;
- diagnostic tools.

They do not make arbitrary dense shapes production-supported. A dense production problem must
receive the same exact admission and qualification treatment as a quantized problem. Current
optimized dense and future grouped-expert lifecycles remain separate future work.

## 4. Execution policy and kernel lifecycle

### 4.1 Policy is the optimization unit

A `PolicyId` names a complete execution bundle:

```text
physical encoding assumptions
CTA problem mapping
token and row ownership
staging lifecycle
decode path
MMA or CUDA-core accumulation
reduction topology
final BF16 boundary
workspace behavior
fixed launcher
```

A policy is not assembled dynamically from independent format, tile, pipeline, split-K, and
epilogue choices.

Examples of legitimate policy identities include:

- row-streaming TT4;
- row-streaming TT8;
- Q5 direct split2;
- Q5 direct split4;
- Q4 BF16 MMA BN64 or BN128;
- Q5 BF16 MMA BN64 or BN128;
- Q6 BF16 MMA BN64 or BN128;
- W8G32 small-M MMA;
- W8G32 general MMA;
- an exact T1 warp-per-row specialization;
- an exact T1 split-K specialization.

Multiple exact support problems may select the same policy. Exact dispatch does not imply one
kernel per shape.

### 4.2 Complete-lifecycle reuse boundary

A reusable mainloop ends when any of the following changes materially:

- producer lifetime;
- asynchronous staging protocol;
- synchronization schedule;
- warp ownership;
- accumulator topology;
- final reduction ownership;
- output mapping.

The experiments support separate complete lifecycles for:

- Q4 row-streaming, SIMT, and MMA;
- Q5 GEMV, direct Small-T, SIMT, and MMA;
- Q6 SIMT and MMA;
- W8G32 MMA;
- ownership-specific T1;
- paired or grouped output topologies.

Q5 and Q6 deliberately do not share a configurable kernel body, schedule type, route table, or
codec abstraction. Their optimization trajectories must remain independent. Trying to force these
lifecycles through one configurable loop would create a scheduling DSL, enlarge template
combinations, and risk generated-code regressions.

### 4.3 Narrow reusable substrate

Reuse is restricted below the lifecycle boundary:

- core memory, warp, and MMA instruction primitives;
- neutral swizzles and address helpers;
- small mechanics-only configuration helpers used by fused kernels;
- format-local storage/decode atoms reused only by consumers of the same format;
- alignment and launch-contract checks that contain no format policy.

These primitives must remain closed and compile-time. They are accepted only when representative
resource usage, SASS, and matched timing show that the hot policy is preserved.

Source reuse is not a goal by itself. An exact implementation remains first-class whenever it is
the measured winner or the clearer ownership boundary.

## 5. Fused semantic Ops and epilogue extensibility

### 5.1 Base Linear remains semantically closed

Base `linear()` does not accept:

- an epilogue enum;
- an auxiliary pointer descriptor;
- a callback or function pointer;
- a target-selected finalizer;
- a template argument exposed to callers.

LinearAdd, LinearSwiGLU, LinearPair, grouped attention input, and grouped GDN input are independent
semantic Ops because they have different rounding, topology, aliasing, and workspace contracts.

Each owns its own internal:

- exact physical-problem key;
- immutable route catalog;
- typed policy enum;
- workspace resolver;
- exhaustive fixed executor.

### 5.2 Typed internal finalizers

Narrow compile-time finalizers are still useful implementation seams. Examples include:

```text
StoreBf16
ResidualAddAfterProjectionRound
CtaCollectiveResidualAfterProjectionRound
SwiGluAfterTwoProjectionRounds
```

They are internal pieces of complete policies, not runtime extensions.

The semantic rounding boundaries are mandatory:

- LinearAdd must preserve its declared BF16 projection boundary before the residual update;
- SwiGLU must preserve the required independently rounded projection halves;
- Pair owns two weight streams and two outputs, not merely one accumulator epilogue;
- grouped projection changes the CTA problem map and output ownership.

The finalizer seam may be shared only where those semantics and ownership facts match.

### 5.3 Three first-class execution forms

A fused semantic Op may select one of three complete forms:

1. **terminal finalizer:** an otherwise compatible mainloop finishes through a typed finalizer;
2. **topology-aware fused kernel:** the Op requires paired/grouped accumulators or a different CTA
   problem map;
3. **materialized composition:** fixed internal subplans execute into explicit scratch and a
   second semantic step completes the Op.

Materialized composition is not an inferior fallback. It is a first-class policy when measurement
selects it.

### 5.4 Materialized composition must not redispatch

The fused planner resolves a complete composed plan once. Its executor must not call public
`linear()` and trigger a second automatic route decision.

For example:

```cpp
struct LinearAddPlan {
    LinearAddPolicyId policy;
    std::optional<LinearPolicyId> materialized_base_policy;
    WorkspacePlan workspace;
};
```

The exact representation may differ, but the invariant is:

- the fused route fixes or resolves its required base subplan during the fused planning step;
- execution invokes the fixed base launcher directly;
- changing a Base Linear route cannot silently change a qualified fused route;
- compile-time validation proves the composed subpolicy is legal for the same physical problem.

This preserves one routing authority per semantic Op while allowing controlled reuse.

### 5.5 Cost of adding a fused capability

Adding a fused Op or fused policy requires:

- one explicit semantic and rounding contract;
- one physical problem definition;
- one finite route catalog;
- one or more complete policies;
- exact and range-capacity workspace rules;
- independent numerical verification;
- candidate timing and end-to-end evidence;
- resource/SASS inspection when a supposedly zero-cost seam is introduced.

The cost stays inside the Op. Target callers continue to use the original semantic interface.

## 6. Workspace and CUDA Graph behavior

### 6.1 Exact workspace belongs to the resolved plan

Every policy defines exact scratch for one call. Workspace availability must not influence route
selection:

```text
resolve best qualified policy
  -> compute exact workspace
  -> require caller arena capacity
  -> execute or fail
```

The implementation must not select a slower policy because the caller provided less scratch.

Base Linear policies should remain externally workspace-free when that preserves the winner and
simplifies graph stability. This is a preference, not permission to hide allocation. If a future
winning Base policy requires scratch, a target-independent capacity API must be designed
explicitly.

### 6.2 Capacity queries use the same route authority

Target layout planning may ask a semantic Op for capacity using semantic dimensions, physical
weight facts where necessary, and `max_tokens`. This is an execution-resource query, not dispatch
configuration.

For a route-dependent fused Op:

```text
capacity(max_tokens)
  = max(exact_workspace(route(T))) for every admitted T in [1,max_tokens]
```

The query must not inspect only the route at `T=max_tokens`, because winning policies can be
non-monotonic.

Materialized workspace includes:

- intermediate BF16/FP32 tensors;
- alignment padding;
- any fixed subplan scratch;
- the maximum simultaneous lifetime, not a sum of sequentially reusable allocations.

Execution and capacity queries must share the same Op-owned route facts so they cannot drift.

### 6.3 CUDA Graph stability

Planning is deterministic host work derived from immutable inputs and a compiled catalog.

During CUDA Graph capture:

- the same physical problem and token count resolve the same policy;
- workspace addresses come from the pre-reserved arena;
- no lazy initialization, autotuning, allocation, or mutable registration occurs.

Replay executes the captured kernel nodes and does not repeat host dispatch. This removes any
motivation to pollute `Weight` with cached policies.

## 7. Extending the supported domain

### 7.1 Adding a similar shape

Shape similarity proposes candidates; it never decides production routing.

For a new physical problem:

```text
construct exact benchmark problem
  -> enumerate physically compatible existing policies
  -> verify numerical legality
  -> compare candidates at relevant T and wave/tail points
  -> select route intervals
  -> add exact Op-owned support rows
```

If an existing lifecycle wins, the change adds only exact routes pointing to that policy. If no
existing policy is adequate, the change adds a new complete policy or exact specialization.

Until that process completes, the new shape is unsupported.

### 7.2 Future 35B workflow

Introducing 35B later follows this sequence:

1. inventory effective physical Linear-family problems from the target's real bound views;
2. expose those problems only in measurement tools;
3. test existing codecs and policies as candidates;
4. add missing dense or grouped-expert lifecycles where required;
5. qualify exact token routes on RTX 5090;
6. extend the Op-owned support catalogs;
7. implement the 35B target using unchanged semantic Op interfaces;
8. validate the real artifact and product schedules.

Current 27B callers and public interfaces do not change.

The existing experiments already show that future quantized problems can often reuse current
closed policies, but route boundaries remain geometry-local and sometimes non-monotonic. Dense
BF16 and grouped experts still require separate lifecycle work.

### 7.3 Extension budget

The intended cost is:

| Change | Required architecture work |
|---|---|
| new exact shape using a qualified lifecycle | one exact support row plus measured token routes |
| new route winner for existing shapes | one policy addition or route update, with affected evidence |
| new fused semantic Op | semantic contract, planner, policies, workspace, numerical and performance evidence |
| new physical encoding | codec/addressing contract plus compatible policy qualification |
| new hardware | separate Op-owned hardware backend and a dedicated design review |
| new product target | target binding/schedules plus coverage proof; no Linear API changes |

This is intentionally not constant-cost arbitrary extensibility. New execution behavior pays for
the facts it introduces, without forcing unrelated callers or kernels into a broader framework.

## 8. Complexity control and rejected alternatives

### 8.1 Accepted abstractions

- exact physical signatures;
- finite immutable support and route tables;
- typed complete policies;
- exhaustive fixed launch switches;
- hardware-local capability traits;
- Op-local planners and workspace plans;
- narrow closed codec, decode, MMA, topology, and finalizer primitives;
- internal measurement-only candidate launchers that cannot enter product resolution.

### 8.2 Rejected abstractions

| Rejected design | Reason |
|---|---|
| target/profile parameter on Linear-family Ops | leaks dispatch ownership and burdens every caller |
| profile hidden in Engine, Program, WorkspaceArena, global state, or thread-local state | preserves the same dependency while making it implicit |
| policy or target identity stored in `Weight` | stale across T, views, semantic Ops, and hardware; contaminates storage contracts |
| target-owned route tables | makes target code choose Op implementation |
| target-keyed duplicate routes | identical physical problems should share one hardware route |
| ShapeFamily or caller-role dispatch | names models rather than kernel-relevant facts |
| global Small-T/Large-T threshold | measured crossovers are geometry-local |
| unknown quantized fallback | silently expands support without qualification |
| arbitrary dense reference fallback | contradicts best-qualified production dispatch |
| launcher-side auto selection | creates a second routing authority |
| runtime registry or function-pointer table | unnecessary dynamism and initialization complexity |
| runtime autotuning | nondeterministic startup/capture behavior and weak reproducibility |
| universal configurable mainloop | hides materially different lifecycles in a scheduling DSL |
| arbitrary runtime epilogue | cannot preserve all semantic rounding and topology contracts |
| route chosen from available workspace | makes correctness/performance depend on arena pressure |
| grid-legal but unqualified terminal tail | launch legality is not best-kernel evidence |

### 8.3 Code-size discipline

Finite exact policies may increase compiled code, but uncontrolled template cross-products are not
accepted.

Every new policy must justify:

- which admitted problem/range selects it;
- why an existing policy is insufficient;
- generated entry count and binary impact;
- registers, shared memory, stack/local memory, and representative SASS;
- matched timing and numerical evidence.

Unused tuned translation units and unreachable caller-role kernels are deleted when the real target
inventory proves they have no execution path.

## 9. Roofline and performance qualification

### 9.1 What architecture can and cannot guarantee

The architecture can guarantee:

- one observable final policy per admitted problem and token count;
- no hidden post-plan dispatch;
- stable fixed kernels for measurement;
- isolated policy changes;
- explicit workspace and fusion semantics;
- exact rejection outside the qualified domain.

It cannot guarantee roofline merely by introducing abstractions or route tables. Roofline is a
measured property of a fixed policy at a geometry, token range, wave count, and tail class.

### 9.2 Performance models by regime

Qualification uses the appropriate model:

- **T1 and row-streaming Small-T:** useful and physical weight traffic, launch floor, occupancy,
  issue efficiency, and redundant rereads;
- **Q5 direct:** split ownership, extra-plane decode, reduction cost, and output traffic;
- **split-low4 MMA:** dequantization throughput, tensor-pipe utilization, wave occupancy, and
  BN64/BN128 tails;
- **W8 MMA:** code/scale staging, fragment consumption, two-CTA residency, and partial waves;
- **fused policies:** avoided launches and memory traffic versus registers, shared memory,
  collective stores, and topology overhead;
- **grouped policies:** CTA problem-map utilization and output tails;
- **end-to-end:** route frequency, CUDA Graph composition, launch gaps, and interaction with other
  Ops.

A useful-byte ratio above 100% is evidence that the accounting model is incomplete, not proof that
hardware limits were exceeded.

### 9.3 Qualification ladder

Each policy or route change follows:

1. **semantic oracle:** exact formats/transformations or independent FP64/BF16 numerical criteria;
2. **winner selection:** matched candidate measurements at real shapes and relevant token points;
3. **generated code:** resource and SASS comparison for abstraction-sensitive changes;
4. **physical diagnosis:** NCU only after the kernel and question are identified;
5. **product attribution:** NSYS and real-artifact benchmarks for important route families;
6. **admission update:** only after the evidence selects a production route.

Same-session matched comparison is required for performance claims. Profiler replay duration is
diagnostic and does not replace the normal benchmark timing.

### 9.4 Stage A and Stage B

The architecture cut-over is Stage A:

- correct ownership;
- exact internal admission;
- one final plan;
- fixed launch closure;
- preserved qualified winners;
- no caller API change.

Stage B iteratively pushes high-impact policies toward their attainable roofline:

- product-dominant T1 and Small-T;
- W8 reread and partial-wave behavior;
- tail-heavy Vision routes;
- dense and grouped-expert lifecycles when those products are admitted.

Stage B does not require another architecture rewrite.

## 10. Source and build ownership

The intended source shape is:

```text
include/ninfer/ops/
  linear.h
  linear_add.h
  linear_swiglu.h
  linear_pair.h
  grouped projection semantic headers

src/ops/linear/
  q4/
    storage/decode atoms
    GEMV/SIMT/MMA kernel families
    schedule capability, exact routes, fixed launch closure
  q5/
    storage/decode atoms
    GEMV/SIMT/MMA and residual kernel families
    schedule capability, exact routes, fixed launch closure
  q6/
    storage/decode atoms
    SIMT/MMA kernel families
    schedule capability, exact routes, fixed launch closure
  w8/                       # migration target; currently compatibility code
  bf16/                     # migration target; currently dense reference code
  common/                   # mechanics only; no cross-format route or kernel body
  gemv/ and gemm/           # transitional W8/dense/fused files, removed as migrations finish
  reference/                 # internal oracle/measurement code, not product fallback
  linear.cpp

src/ops/wrapper/
  semantic fused/grouped Op implementations

src/targets/qwen3_6_27b_rtx5090/
  bound storage views
  schedules and reachability
  workspace reservation through semantic capacity APIs
  real-artifact integration
```

Exact route data and policy selection live in `src/ops`, because they are execution behavior of the
Op. Target code contains no parallel manifest of policy choices.

The target may retain or derive an inventory test proving that every live bound view and reachable
token domain is accepted by the Op. That test is coverage evidence, not a second routing
authority.

Measurement tools may expose:

- exact production resolver mode;
- forced candidate mode;
- future physical problems not admitted by product resolution;
- policy identity and route boundaries.

Forced candidates call internal fixed launchers and are unavailable through semantic product APIs.

## 11. Current evidence and revised conclusions

The archived experiment remains valuable. Its measurements support:

- a 24-problem current 27B base Linear inventory;
- geometry-local route boundaries rather than one global regime threshold;
- 83 candidate-selected base route intervals in the prototype manifest;
- independent Q4, Q5, and Q6 format-local plans and kernel lifecycles;
- distinct TT4, TT8, Q5 direct, W8 MMA, and exact T1 policies;
- two LinearAdd problems with nontrivial fused/materialized/collective routes;
- one non-monotonic LinearSwiGLU route surface;
- one independently routed W8 LinearPair problem;
- mechanics-only cross-format reuse plus same-format decode reuse by fused Ops;
- removal of unreachable 7168 and dead dual-Q5 tuned execution paths;
- Linear dominance in whole-inference attribution;
- representative per-policy resource and NCU evidence.

Those are execution facts, not proof of the rejected target-profile boundary.

The following earlier conclusions are superseded:

- route data must be target-owned;
- Linear-family APIs should receive an execution profile;
- target/profile and kernel catalog require a public-facing joint contract;
- all routes should extend to a CUDA grid ceiling regardless of qualification;
- materialized fused execution may call public auto-dispatch Linear during execution.

The revised interpretation is:

- the measured 24/83 manifest is the starting RTX 5090 Op support catalog;
- exact route ownership belongs to Linear, not the 27B target;
- fused Ops receive their own Op-owned catalogs;
- the qualified envelope must be reviewed against real product reachability before admission;
- future 35B rows extend the same physical capability catalog after measurement.

The archived prototype branch and raw profiler outputs are preserved only for comparison. Its
profile-injection implementation must not be cherry-picked into the product.

## 12. Atomic implementation plan

### 12.1 Repository states and provenance

The implementation uses three explicit states:

```text
clean product baseline
  original Linear-family APIs and current product behavior

archived experiment
  measured prototype, raw reports, and failed target-profile architecture

clean Op-owned implementation
  fresh internal planner/catalog cut-over with unchanged callers
```

The failed uncommitted profile-injection implementation has been discarded. The archive remains
available as behavioral, numerical, kernel, and performance evidence.

### 12.2 Meaning of an atomic cut-over

The product-facing API remains unchanged throughout. Atomicity concerns internal routing authority:
the integrated tree contains either the complete old internal dispatch or the complete new
Op-owned exact dispatch.

The final tree must not contain:

- `LinearExecutionProfile` or equivalent target dispatch objects;
- profile-aware overloads;
- target route tables;
- old `ShapeFamily`/`LinearRegime` authority beside exact routes;
- generic quantized fallback;
- planner plus launcher-side auto selection;
- fused executors that redispatch through public Linear;
- old and new workspace semantics simultaneously.

Because callers do not change, no Text/Vision/MTP profile migration is required.

### 12.3 Implementation work packages

The clean implementation proceeds in these bounded packages:

1. define internal physical-problem, token-range, execution-plan, workspace-plan, and complete
   policy contracts;
2. define the RTX 5090 capability catalog and compile-time closure checks;
3. selectively re-establish qualified fixed kernels and narrow codec/finalizer/topology seams from
   the archive evidence;
4. reconstruct the exact Base Linear support/routes inside the Op;
5. implement independent Add, SwiGLU, Pair, and grouped semantic planners;
6. make every executor consume one final plan and call only fixed launchers;
7. remove ShapeFamily/regime/fallback/second-dispatch and unreachable tuned kernels;
8. align capacity queries with the same non-monotonic route facts;
9. update structural, numerical, benchmark, and real-artifact verification;
10. complete matched performance and binary/profiler closure before the product cut-over commit.

The prototype's 24 base problems, 83 base routes, two Add problems/eight routes, one SwiGLU
problem/seven routes, and one Pair problem/three routes are evidence inputs. They are not copied
blindly: supported token envelopes and every route claimed as best must be revalidated under the
revised admission rule.

### 12.4 Structural acceptance

Compile-time and host tests must prove:

- exact physical problems are unique;
- route intervals are ordered, contiguous, non-overlapping, and within qualified envelopes;
- every policy is legal for its exact problem/range;
- every policy has one fixed launcher and workspace definition;
- no production route names Auto/Default/Fallback;
- fused materialized routes contain fixed valid subplans;
- all live 27B effective views are covered;
- dead/unreachable parent storage views are not accidentally admitted;
- measurement-only 35B problems remain absent until separately qualified.

Source scans must find:

- no target/profile parameter in Linear-family APIs;
- no target-owned policy or route facts;
- no target/layer/phase-driven Linear routing;
- no arbitrary quantized or dense production fallback;
- no launcher-side second dispatch.

### 12.5 Numerical and product acceptance

Verification must include:

- independent dequantization and FP64/BF16 operator oracles at real shapes;
- every route boundary and representative tail/wave point;
- exact fused rounding for Add, SwiGLU, Pair, and grouped projections;
- alignment and physical-view validation;
- non-monotonic range-capacity workspace checks;
- rejection outside exact problem and token envelopes;
- real 27B Text, Vision, MTP, prefix reuse, and CUDA Graph routes.

Final-token plausibility is not operator verification.

### 12.6 Performance acceptance

The cut-over must preserve or improve:

- final policy selection relative to revalidated experiment winners;
- representative kernel resources and SASS where an abstraction claims zero cost;
- planner cost at the real call mix;
- matched per-Op latency;
- real-artifact prefill, decode, Vision, and MTP behavior within same-session noise.

NSYS is used for whole-inference attribution. NCU is used only for identified fixed kernels and
specific resource/roofline questions.

No claim is made that all supported routes already reach roofline. The cut-over must make each
route independently measurable and preserve the known winners; Stage B performs further policy
optimization.

### 12.7 Commit boundaries

Reasonable durable commits are:

1. this corrected architecture and provenance record;
2. the already preserved experiment archive;
3. one complete verified product cut-over.

Private implementation checkpoints may exist on an isolated branch, but no partially migrated
internal architecture is presented as the product state.

## 13. Final recommendation

The recommended architecture is:

```text
unchanged semantic Op API
  + Op-owned exact physical support
  + Op-owned hardware-specific routes
  + complete typed policies
  + one final plan
  + fixed launch closure
  + independent fused semantic planners
  + qualification-bounded admission
```

The defining ownership statement is:

> The target owns which Linear calls exist. The Linear-family Op owns the entire decision about
> whether a physical call is supported and which qualified kernel executes it.

This keeps the caller simple, preserves exact specialization, supports measured reuse for future
shapes, prevents target semantics from leaking into Ops, and provides a stable foundation for
per-policy roofline work.
