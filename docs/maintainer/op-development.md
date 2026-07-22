# NInfer Op Development

This is the maintainer guide for NInfer Ops and their CUDA implementations. Repository-wide scope
and verification principles remain in [`AGENTS.md`](../../AGENTS.md).

An **Op** is a semantic execution contract. A CUDA **kernel** is one implementation, or one stage
of an implementation, of an Op. Keeping those two concepts separate allows NInfer to specialize
aggressively for exact shapes and devices without assigning mathematical code to whichever model
first uses it.

## 1. Op definition

An Op is a host-callable, semantically closed computation in the engine execution layer. Given all
explicit inputs, weights, state, and semantic parameters, it completely defines its outputs and
state changes without depending on model identity or the schedule that invokes it:

```text
(outputs, new_state) = F(inputs, weights, old_state, semantic_parameters)
```

Workspace, CUDA stream, and device facts are execution resources. They may select how the Op is
implemented, but they do not change what the Op means.

A component is an Op only when it has all of these properties:

1. **Logical effect.** It computes or changes a logical tensor value or an explicit local state
   value. The effect can be stated as a formula, index mapping, algorithm, probability process, or
   state transition.
2. **Semantic closure.** Its contract states every observable output and mutation, including
   in-place updates, cache writes, counters, statistics, valid output regions, alias restrictions,
   and observable cast or rounding boundaries.
3. **Explicit dependencies.** Everything that affects the result arrives through tensor/weight
   views, explicit state references, semantic parameters, layout/format metadata, or explicit
   stochastic inputs.
4. **Schedule independence.** It performs one complete local transformation without deciding model
   call order, request policy, sequence frontier, commit/rollback, or state lifetime.
5. **Implementation independence.** Its semantic interface does not expose grids, blocks, warps,
   tiles, launch counts, CUDA symbols, implementation filenames, or model-labelled backend paths.
6. **Independent invocation.** It has a meaningful host-callable boundary and can be verified from
   its explicit inputs, outputs, and state without running a complete model schedule.

One caller, one supported shape, one registered tensor format, one optimized token extent, one device
implementation, mutable state, or a fused formula do not disqualify an Op. Reuse count and support
breadth describe the current implementation domain; they do not determine ownership.

### 1.1 Token extent is semantic input, not a request phase

When an Op contract defines an axis as the Text/MTP token extent `T`, that axis accepts any positive
value representable by the tensor and available storage unless the Op has an explicit semantic
capacity such as a cache envelope. The default prefill chunk size of 1024 is a primary performance
target, not an admission bound. `T=1`, other small values, and chunks around or beyond 1024 are
values of the same Op contract.

Do not infer that rule merely because an implementation stores work in matrix columns. Vision raw
patches `P`, merged Vision tokens `V=P/4`, MTP proposal count `K`, vocabulary rows, head widths, and
other axes keep the finite geometry or capacity declared by their own contracts. In particular,
the registered Vision Linear problems admit the valid frontend `P` or `V` domain; those columns are
not Text token `T`.

Names such as decode, Small-T, and prefill may describe private benchmarks or implementation
routes. They must not appear as semantic variants, admission domains, separate target-callable
entry points, or API limits. A kernel may internally select a schedule or split one call into
several launches to satisfy CUDA grid/resource limits; that decomposition is invisible to the
caller and does not narrow T.

### 1.2 Stateful, fused, and stochastic Ops

A stateful Op defines both its produced value and its state transition:

```text
output    = G(old_state, input)
new_state = F(old_state, input)
```

The contract includes every location written by the call and whether newly written state is
visible to the current computation. Later cursor movement, transaction commit, rollback, or state
instance selection remains schedule policy.

A fused computation is an Op when the complete fused result and all effects form one closed
contract. The contract states the full composition and any observable intermediate cast or
rounding seam. A tile decoder, partial reduction, epilogue fragment, or warp primitive remains an
implementation detail when it is only one step beneath a complete host-callable transformation.

#### Oracle precision and implementation freedom

Every floating-point Op has one independent naive FP32/FP64 mathematical oracle. The oracle starts
from the values represented by the public inputs, evaluates the complete logical formula without
production tiling or staging, and applies the declared observable output/state formats. For packed
weights it decodes each signed code with the exact stored scale before evaluating the formula.
Exact transforms and codecs use one independent exact oracle instead.

The oracle defines correctness; it does not freeze the arithmetic path of a kernel. Private
accumulators, instruction operand types, staging casts, reduction association, workspace dtypes,
and kernel boundaries are implementation choices unless the value is independently observable in
the Op contract. A fused implementation does not inherit a former unfused tensor's BF16 rounding
merely because that tensor once existed, but it is also not required to keep every private value in
FP32. Each route may use its natural numerical path and is qualified directly against the same
oracle with an appropriate tolerance. An implementation profile or another GPU path is never a
second oracle.

For a stochastic Op, all semantic random inputs are explicit. This includes the RNG state or seed,
logical position, purpose/domain separation, and draw index required by the algorithm. The contract
defines the probability process and state transition. It promises a particular backend bitstream
or sampled sequence only when that is deliberately part of the semantics.

### 1.3 Execution envelopes

An Op may accept an explicit host execution envelope when device-resident semantic inputs cannot
be read on the host without synchronization but do affect launch capacity or concrete kernel
selection. The envelope is an execution-resource promise, not a second mathematical input:

- device tensors still define the exact result and state transition;
- the caller proves the actual device value lies inside the declared finite interval;
- a larger valid interval may change grid capacity or implementation choice, but not the logical
  output or effects;
- CUDA Graph capture records one launch valid for the complete interval and replays without reading
  a target lifecycle cursor;
- target-private graph tiers may choose intervals, while the Op owns interval-aware finite dispatch.

GQA is the current concrete use: device `positions` define the causal mask, while
`GqaExecutionEnvelope[min_visible_keys,max_visible_keys]` bounds the visible-key window for
split capacity and INT8 implementation selection. The physical cache view contains no frontier or
graph identity.

### 1.4 Logical assignment versus raw transfer

Classification follows the logical interface, not the CUDA primitive used underneath:

- a typed scalar assignment whose contract is `destination' = source` is an Op;
- an interface expressed as addresses, byte count, and transfer direction is a core/host transfer;
- a copy with cast, transpose, concat, scatter, remap, or another logical index transformation is
  the corresponding Op.

A semantic assignment accepts typed logical views and states its dtype, shape, and aliasing rules.
Its implementation may use a device-to-device copy. That implementation choice does not turn the
contract into a raw transfer. Conversely, a raw upload or copy does not become an Op merely because
the caller assigns meaning to the destination afterward.

## 2. What is not an Op

### 2.1 Program and schedule

Program, target schedule, and runtime policy own:

- model layer topology and call order;
- selection of weight roles and state instances;
- prompt chunking and multimodal span interpretation;
- proposal, verification, and generated-token transactions;
- prefix/frontier/commit/rollback policy;
- cache and recurrent-state lifetime;
- CUDA Graph variants and stable graph addresses;
- interpretation and publication of product statistics.

A host helper that derives operands, chooses call time, or composes existing Ops remains schedule
code even when it has only one caller and can be unit tested.

### 2.2 L0 storage and execution mechanisms

Core owns storage and execution mechanisms rather than mathematical contracts:

- `Tensor` and `Weight` views;
- dtype/device facts and checked layout builders;
- arenas and `WorkspaceArena`;
- physical KV-cache containers and views;
- CUDA Graph lifetime;
- raw host/device and device/device transfers.

Physical cache containers bind planes and produce checked per-layer views; they contain no logical
sequence cursor. Published prefix lengths, rewind/reset decisions, transaction publication, and
cache/state alignment are target Program values and policy. An Op may consume a core type or
explicit view without owning that object's lifetime.

### 2.3 Implementation details

These are parts of an Op implementation, not separate Ops:

- wrapper validation and dispatch;
- launchers and CUDA entry points;
- `__global__` and `__device__` functions;
- tensor-format codecs and implementation plans;
- partial reductions and staging kernels;
- workspace layout/sizing helpers;
- narrow math, memory, warp, and MMA primitives.

An implementation-private helper may exist below an Op, but target and product code must not invoke
it directly.

### 2.4 Artifact, frontend, product, and tooling

Artifact framing/binding/materialization, converter recipes, tokenizer and chat-template handling,
media acquisition/decoding, request translation, serving transport, diagnostics, and profiling
tools are not Ops. They prepare, move, observe, or present data rather than define device execution
semantics.

## 3. Admission and ownership

Classify every proposed target-callable computation in this order:

1. Does it change a logical tensor or explicit state value? If not, it is infrastructure,
   validation, planning, transfer, or another non-Op mechanism.
2. Can the complete transformation be defined from explicit arguments and metadata? If not, it
   belongs to target/product schedule.
3. Does it decide call order, lifecycle, frontier, commit, rollback, or request policy? If so, it
   belongs to Program/schedule/runtime.
4. Is it only a partial step beneath another complete contract? If so, it is an implementation
   detail of that enclosing Op.
5. Otherwise it is an Op.

A useful review question is:

> If the model name and call site disappear, can the transformation still be described and
> verified completely?

If the answer still requires phrases such as “at this model phase”, “for this weight role”, or “at
the current request frontier”, the proposed boundary is not semantically closed.

Every Op contract and every implementation of that contract belong to the central Op layer. There
is no target-facing private Op category. A target directly invokes only:

- contracts from `include/ninfer/ops/`;
- L0/core storage and transfer mechanisms;
- its own host-side schedule composition.

If a target needs to call a new device transformation, either admit it as an Op or express it by
composing existing Ops and core mechanisms. Exact shape and hardware specialization remain private
implementations of the admitted Op; they do not move under the target.

## 4. Contract headers

Repository-internal Op contracts live in:

```text
include/ninfer/ops/<family>.h
namespace ninfer::ops
```

They are execution-layer interfaces, not public product ABI. A header may group closely related
operations or overloads. Do not create an empty family header merely to force one function per
file.

Each semantic Op, or closely related overload group, has one authoritative contract comment. Use
the applicable fields below:

```cpp
/**
 * Op: <semantic operation name>
 *
 * Math / indexing:
 *   <complete formula, algorithm, probability process, or index mapping>
 *
 * Logical shapes:
 *   <symbols, semantic axes, valid ranges, and layout interpretation; identify T versus P/V/etc.>
 *
 * Supported domain:
 *   <dtype, numeric format, storage layout, shape, and semantic variants>
 *
 * Numeric:
 *   <logical decode, epsilon, observable casts/state formats, masking, ties, and output criterion>
 *
 * Effects:
 *   <outputs, old/new state, in-place writes, valid regions, and alias rules>
 *
 * Workspace:
 *   <caller-owned transient scratch requirement or none>
 */
```

The comment describes behavior shared by every valid implementation. It must not freeze:

- reduction association or thread execution order;
- warp/CTA/tile decomposition;
- launch count or kernel symbol;
- a backend approximation instruction;
- private accumulator, operand-staging, or workspace dtype;
- an implementation-only intermediate cast or rounding point; or
- bitwise equality unless the semantic format requires it.

A contract may cite the authoritative tensor-format or layout document instead of duplicating a
complete registered encoding. It must still state which format and interpretation the Op accepts.

Formula quality is judged by whether an independent implementation and oracle can be written from
the contract. Names such as “fused projection” or “attention path” are not formulas. For a fused Op,
write the whole composition. For a stateful Op, write old and new state. For indexing, define every
axis mapping and valid region. For stochastic behavior, define ordering, filtering, normalization,
random inputs, and state effects.

Helpers covered by the same contract, such as workspace sizing queries, do not repeat the formula.

## 5. Source organization

The normal responsibility chain is:

```text
include/ninfer/ops/<family>.h      semantic contract
                |
                v
src/ops/wrapper/<family>.cpp      validation, workspace scope, finite dispatch
                |
                v
src/ops/launcher/<family>.*       private CUDA launch policy
                |
                v
src/ops/kernel/<family>*.cuh      device implementation
```

Shared implementation facilities live in:

```text
src/ops/common/                   narrow CUDA primitives
src/ops/linear/                   linear codec, plan, reference, GEMV, and GEMM paths
```

This is a responsibility model, not a requirement for four files per Op. Related Ops may share a
launcher, a wrapper may implement a fallback by composing other Ops, and a small operation does not
need empty source layers.

### 5.1 Wrapper

`src/ops/wrapper/` owns:

1. semantic dtype, rank, shape, layout, and alignment validation;
2. scalar, configuration, and explicit state validation;
3. transient workspace scope creation;
4. private implementation selection from semantic variant, format, layout, numerical shape, extent,
   state dtype, and device capability, without turning a tuning threshold into an admission bound;
5. invocation of the selected launcher or a composed fallback.

A wrapper must not dispatch on target key, artifact tensor name, source layer role, Program phase,
or arbitrary registry strings. It does not own persistent state, allocate hidden device memory,
capture graphs, or choose model call order.

### 5.2 Launcher

`src/ops/launcher/` owns private launch declarations, grid/block/shared-memory policy, template
instantiation, and launch-error handling. Launcher headers are implementation-private. Contract
headers, targets, and product code never include them.

### 5.3 Kernel

`src/ops/kernel/` owns `__global__` functions and Op-local reusable `__device__` computation. A
kernel may encode exact shape, tensor format, SM capability, tiling, padding, and alignment
assumptions. Every assumption must correspond to a wrapper/launcher predicate and must not be
inferred from model identity.

### 5.4 Common

`src/ops/common/` contains narrow zero-cost arithmetic, memory, warp, and MMA primitives used by Op
implementations. It is not a second semantic catalog. Common helpers do not expose target-callable
transformations, and targets never include this directory directly.

### 5.5 Linear

Linear keeps a dedicated subtree because registered formats and execution regimes produce a larger
implementation matrix:

```text
src/ops/linear/
├── linear.cpp
├── codec/
├── plan/
├── reference/
├── gemv/
└── gemm/
```

- `linear.cpp` integrates contract validation and dispatch;
- `codec/` implements registered device decode rules;
- `plan/` performs finite format/shape/device classification and private token-extent tuning;
- `reference/` provides supported generic or dense CUDA paths;
- `gemv/` contains single-token and small-token implementations;
- `gemm/` contains larger-token tensor-core and fused implementations.

This subtree is not a generic backend framework. Do not introduce an Op base class, runtime
registry, universal plan interface, plugin discovery, string dispatch, or graph IR. Select directly
from the finite formats, shapes, and devices that NInfer actually supports. Keep Text/MTP token T
independent of benchmark or schedule phases, and preserve separately declared domains such as
Vision P/V.

### 5.6 Implementation comments

Launcher and kernel files reference the semantic contract instead of copying it. Record the match
predicate and implementation assumptions in a compact form:

```text
Implements: include/ninfer/ops/<family>.h
Match: <device, dtype/format, layout, numerical shape, private token-extent route>
Algorithm assumptions: <tiling, padding, alignment, staging, or launch requirements>
```

Codecs, plan helpers, and common primitives document their implementation invariant and enclosing
Op rather than inventing a separate Op formula.

## 6. State, weights, and workspace

Ops receive non-owning execution views. They may inspect only facts required for numerical
execution, such as dtype/format, storage layout, logical and padded shape, payload planes,
quantization geometry, and alignment.

Ops must not receive artifact object names, converter source fields, model weight roles, recipe
information, or provenance. Artifact binding and target loading translate those concepts into the
explicit execution views consumed by Ops.

Caller-owned `WorkspaceArena&` or an explicit caller-owned scratch view is the workspace boundary.
An Op may suballocate transient scratch within its call scope, but it must not call `cudaMalloc`,
retain a workspace pointer, or own the arena. A sizing query and its execution path must share one
private layout definition so Program can plan memory from the same requirements the wrapper uses.

Stateful Ops receive explicit state containers or views. Program owns instances and lifetime;
core owns target-neutral physical mechanisms; the Op owns only the documented local reads, writes,
and outputs of one call.

## 7. Naming and dependency rules

### 7.1 Naming

- Name an Op after its mathematical transformation or explicit state transition, not its first
  model, layer role, schedule phase, or CUDA strategy.
- Use `ninfer::ops` for semantic entry points. Use `ninfer::ops::detail` only for implementation
  material that must cross private translation-unit boundaries.
- Name implementation specializations by real match facts such as dtype, format, shape, token
  regime, SM capability, or algorithm. A model role is not a dispatch fact.
- Keep the term `kernel` for CUDA implementation concepts: `__global__` functions, launch policy,
  occupancy, registers, shared memory, and profiler records.
- Prefer one family name across contract, wrapper, launcher, tests, and benchmark unless several
  contracts intentionally share an implementation family.

### 7.2 Dependencies

The dependency direction is:

```text
core <- ops <- target <- runtime/engine product route
```

Artifact and loading code may materialize execution views for a target, but the Op layer does not
depend on artifact provenance or target binding concepts.

Enforce these rules in code and build ownership:

- contract headers include only required L0 types and CUDA host types;
- `src/ops/**` does not include target, Program, schedule, product, or artifact-provenance headers;
- target schedule includes contract headers, never launcher, kernel, common, codec, or plan files;
- `ninfer_ops` does not link a target;
- core and artifact do not link Ops or targets;
- source lists remain explicit so every implementation has one build/link owner.

## 8. Correctness tests

Semantic Op tests live under `tests/ops/` and link `ninfer_ops` plus the required L0 libraries, not
a target package. They verify the contract independently of model call order.

Add or change a test when it protects a supported observable risk, for example:

- a mathematical result against an independent FP32/FP64 or exact oracle;
- registered format decode or layout/index mapping;
- an observable cast, rounding, masking, tie, or fusion boundary;
- every documented state mutation and valid output region;
- a real supported dispatch regime or an edge case that exposes changed indexing;
- a reproduced numerical, state, or memory bug.

Use exact comparison for exact transformations and contract-appropriate numerical or behavioral
criteria for floating-point and stochastic work. A stochastic test validates the probability/state
contract or controlled semantic inputs; it does not require sampled text to remain identical unless
the contract promises that result.

Every floating-point Op uses the one naive FP32/FP64 oracle defined in Section 1.2. Give each
implementation profile an explicit named tolerance when its rounding or quantization error differs.
Do not copy query quantization, staging casts, reduction trees, or another implementation's output
into the oracle merely to make parity pass. Conversely, do not turn the oracle's evaluation
precision into a required kernel evaluation order. State the qualification domain honestly: a
tolerance established for registered shapes, tested token extents, the conformance matrix, and
target-representative activations is not a universal error theorem for arbitrary unbounded or
adversarial tensors. Each semantically complete entry point is checked directly against the oracle;
pairwise implementation parity is supplementary evidence.

GQA applies this rule concretely. BF16 and INT8-G64 A1/A3 share an FP64 ideal attention oracle over
BF16 Q and logical cache values (BF16 or FP32-decoded INT8-G64). The target's INT8 Q8-G64 compute
profile remains an intentional optimized implementation and uses the named
`Tolerance::attention_int8()` rather than `Tolerance::attention_bf16()`; it is not folded into a
second reference. Exact INT8 cache code and scale validation remains a separate codec check.

### 8.1 Op qualification standard

One semantic Op, or one closely related overload group, owns one identifiable qualification suite.
The suite calls every semantically complete public entry directly. Tests of a private candidate,
plan, dispatch decision, CUDA Graph replay, or pairwise implementation parity may protect their own
risks, but they do not replace public Op qualification against the independent oracle.

#### Oracle

- A floating-point Op uses one test-owned, naive FP64 oracle. It starts from the values represented
  by the public inputs, exact-decodes packed weights from their stored signed codes and scales, and
  applies only the observable output or state casts declared by the contract.
- An exact transform, codec, copy, or index mapping uses an independent bit- or byte-exact oracle.
- A fused Op oracle evaluates the complete fused formula rather than calling the production Ops that
  could be composed to approximate it. A stateful Op oracle computes both the output and new state.
- A production kernel, another GPU route, a target reference implementation, or generated model
  output is supplementary evidence, never the oracle.

#### Coverage

The suite maintains a finite conformance matrix derived from the supported contract. It covers:

- every public entry, registered dtype/format/layout, semantic mode, state form, allowed alias form,
  and real geometry used by either registered target;
- every production route compiled into `ninfer_ops` and reachable through the public wrapper on the
  supported `sm_120a` device;
- `T=1` and, for an unbounded positive Text/MTP extent, each valid route boundary `b` at `b-1`, `b`,
  and `b+1`, plus a representative interior value for every route. Finite axes such as Vision
  geometry, cache capacity, and MTP proposal count keep their own declared boundaries;
- target-representative inputs and the operation-specific risks that can change the result, such as
  zero or near-zero values, ties, cancellation, saturation, quantization endpoints, and tile or
  group tails; and
- every output and documented state mutation, along with untouched regions and input preservation
  where the contract promises them.

Do not build a redundant Cartesian product when one case proves several dimensions. More random
seeds do not substitute for a missing semantic or route boundary. When a full FP64 comparison of a
registered large output is impractical, combine deterministic structure-aware sampling with a
full-output write, guard, or analytic invariant; sampling alone does not qualify the route.

#### Acceptance criteria

- Exact outputs and exact regions compare every observable bit or byte.
- Each approximate output or state uses one named criterion for its arithmetic profile. Tests do not
  define per-case tolerance literals or runtime overrides.
- A floating-point criterion states its non-finite policy. Unexpected NaN or infinity always fails.
  Elementwise work uses a pointwise bound; reductions, GEMM, and attention use a normwise bound plus
  a finite gross pointwise-error cap so cancellation does not require strict allclose and isolated
  corruption cannot hide in a global norm.
- Different routes use different criteria only when their real arithmetic or quantization profiles
  differ. Widening a criterion requires a numerical reason and requalification of the complete
  affected matrix; a failing implementation alone is not such a reason.

An Op is qualified only when every public entry, finite semantic variant, registered geometry, and
reachable production route maps to a direct oracle case, and every observable output and effect is
checked. Unsupported models, devices, formats, and hypothetical failure modes do not enlarge this
matrix.

Keep schedule composition, state-lifetime, and end-to-end behavior in target/product integration
tests. Do not add tests for private filenames, namespace strings, source paths, trivial wrappers,
coverage, retired compatibility, or hypothetical behavior.

## 9. Performance workflow

Op microbenchmarks live under `bench/ops/` and link the Op layer directly. A useful result records
the semantic operation, exact shape, format/layout, workload token extent, device/toolchain, warmup and
measurement method, cache conditions, and actual selected implementation. A microbenchmark is an
implementation measurement, not a correctness oracle or proof of end-to-end improvement.

For a performance change:

1. establish correctness with the affected Op test;
2. measure the relevant Op and workload token extent;
3. check the affected product route when the change can influence end-to-end inference;
4. use Nsight Systems when whole-request attribution or launch gaps matter;
5. use Nsight Compute only after a specific kernel question has been identified.

Collect only the profiler evidence needed to answer the concrete question. Preserve enough context
to interpret a result; fixed repository or artifact hashes are not required unless a separate
contract calls for them.

Full inference exposes persistent registered NVTX ranges in the `ninfer` domain. The outer
`generate` range is the stable capture trigger; its direct children are `prefill` and `decode`.
Prefill further identifies chunks, full-attention/GDN layers and mixers, while decode identifies
ordinary or MTP rounds and their submit/wait portions. Sparse MoE calls use
`sparse_moe.prefill`, `sparse_moe.small_t`, or `sparse_moe.decode`. Numeric payloads carry the
token extent for chunk/MoE ranges, the layer index for layer/mixer ranges, and the execution
frontier for decode-round ranges.

Use graph granularity for representative end-to-end timing:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-graph-trace=graph --capture-range=nvtx \
  --nvtx-capture='generate@ninfer' --capture-range-end=stop \
  --output profiles/nsys/<name> <application> <arguments...>
```

Use a separate `--cuda-graph-trace=node` capture when individual graph-node kernel ranking is
needed. Node tracing can perturb graph launch and request timing, so do not use that run for the
end-to-end latency claim.

## 10. Adding or changing an Op

Use this sequence for a proposed device transformation:

1. **Classify the boundary.** Apply Section 3. Do not create an Op for a schedule decision, raw
   transfer, container lifecycle operation, or partial implementation helper.
2. **Choose the semantic unit.** Extend an existing family when the new entry is a closely related
   overload or variant. Create a new family only for a distinct closed transformation.
3. **Write the contract first.** Define formula/indexing, logical shapes, supported domain,
   numerical behavior, effects, alias rules, and workspace before selecting a CUDA organization.
4. **Define finite support.** List the registered dtypes, layouts, numerical shapes, semantic axis
   ranges, state forms, and devices actually implemented. Keep every positive Text/MTP T admitted;
   list its measured counts only as correctness/performance evidence or private route thresholds.
   Preserve genuine finite domains such as Vision geometry, cache capacity, or proposal count.
5. **Place implementation code.** Put validation/dispatch in `wrapper`, launch policy in `launcher`,
   device code in `kernel`, and only genuinely narrow reusable primitives in `common`. Use the
   dedicated linear subtree for linear format/plan/GEMV/GEMM work.
6. **Make resource ownership explicit.** The caller supplies outputs, state views, workspace, and
   stream. Keep persistent state and schedule policy outside the Op.
7. **Add necessary evidence.** Add or update the smallest independent Op qualification suite that
   satisfies Section 8.1 for the changed contract or production route. Add a benchmark only when the
   operation or implementation needs isolated performance measurement.
8. **Integrate through the contract.** Target code includes only the semantic header and supplies
   explicit operands. It must not select or include a private backend path.

Changing an existing Op contract requires updating the authoritative comment, every affected
implementation, callers whose assumptions changed, and the tests that protect the changed
observable behavior. Adding a faster implementation under an unchanged contract normally changes
only private dispatch/implementation code and the evidence relevant to that path.

## 11. Review checklist

- Is the proposed boundary a complete logical transformation rather than a schedule step or kernel
  fragment?
- Can its formula and effects be understood without a model name, layer role, or Program phase?
- Are all inputs, state, random inputs, valid regions, and mutations explicit?
- Does the contract state observable numeric and fusion boundaries without freezing CUDA strategy?
- Does it leave private precision, staging, reduction, workspace representation, and intermediate
  rounding to the implementation while keeping every route accountable to one oracle?
- Does wrapper dispatch keep positive Text/MTP T admitted, use it only for private implementation
  selection, and preserve the declared ranges of other semantic axes?
- Are launcher, kernel, common, codec, and plan headers invisible to targets?
- Are workspace and state lifetime owned by the caller?
- Is a raw transfer or cursor mechanism being incorrectly promoted into an Op?
- Is a target-callable device transformation being incorrectly hidden as target-private code?
- Does the independent qualification suite cover every affected entry and reachable route with the
  correct oracle, effects checks, and named acceptance criterion?
- If performance changed, was the relevant Op measured and the affected product route checked?
- Does each source and symbol have one clear build/link owner?
