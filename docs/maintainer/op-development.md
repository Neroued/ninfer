# NInfer Op Development Rules

This document defines the repository-wide rules for admitting, specifying, owning, implementing,
qualifying, and measuring NInfer Ops. An **Op** is a semantic execution contract. A CUDA
**kernel** is one implementation, or one stage of an implementation, of an Op.

Repository-wide product scope and engineering policies are defined in [`AGENTS.md`](../../AGENTS.md).
This document owns Op qualification and performance-evidence rules. When it and a concrete Op
contract differ about that Op's represented inputs, formula, supported domain, or observable effects,
the contract in `include/ninfer/ops/` is the concrete authority and the inconsistency must be resolved.

## 1. Scope and authorities

This file owns only rules that apply across Op families:

- Op admission and semantic boundaries;
- contract content and implementation freedom;
- Op, wrapper, launcher, kernel, core, and model ownership;
- state, workspace, naming, and dependency rules;
- Op-level qualification and performance evidence.

The remaining authorities are:

- `include/ninfer/ops/<family>.h` — the formula, represented inputs, supported domain, outputs,
  state effects, alias rules, and workspace contract of a concrete Op;
- [`tests/README.md`](../../tests/README.md) — test organization, commands, and common reporting
  mechanisms;
- [`bench/README.md`](../../bench/README.md) — benchmark executables, command lines, fixtures, and
  measurement behavior;
- family-specific maintainer references — current family design that cannot be expressed in its
  contract header.

Do not add current source inventories, target geometry tables, private route thresholds, one-time
migration steps, profiler command lines, or test-tool output schemas here. A family reference may
specialize these rules but must not redefine them. Remove a migration document after its target
state becomes the implementation and active contract.

### 1.1 Design intent

The Op layer should be predictable to read and pleasant to extend. A maintainer should be able to
find the mathematical contract, implementation selector, tuned shape policy, and reusable kernel
without reconstructing the history of the model that first needed them.

- **Give every computation a clear home.** Organize source by computational family and algorithm
  subfamily. Related independent Ops can live together; sharing a model call sequence alone does
  not establish a family.
- **Separate three decisions.** An Op boundary defines observable mathematics and effects; a
  family groups related maintenance; a kernel library owns reusable computation. Neither a
  directory nor a CUDA launch count determines the public boundary.
- **Keep small work small.** A simple Op may fit in one CUDA source. Add files and layers when
  they separate meaningful responsibilities, shared implementations, or compilation units.
- **Make implementation selection easy to find.** Keep tuned choices in a selector or shape
  policy. Launchers apply the chosen implementation's resource and tail rules. Do not scatter
  another set of tuning decisions through the call chain.
- **Share computation, keep tuning private.** GEMM bodies, output mappings and epilogues can be
  shared. Each Op owns its selected schedules, shape policies and tuning cutoffs, including when
  their current values happen to match another Op. Optimizing one Op must not silently retune its
  siblings.
- **Keep interfaces about use.** Explain the formula, represented operands, effects and required
  execution resources. Keep tuning history, exact shape inventories and test implementation out
  of public comments. Short comments that state a necessary invariant are preferable to repeated
  explanations of what a function or file is named.
- **Preserve useful specialization.** Explicit formats, geometries and algorithms are welcome.
  A new use belongs in an existing semantic family when its actual computation fits. Do not add
  model-labelled copies, universal configuration objects, virtual backends or generic Op graphs.

For example, a row-segmented matrix projection belongs to Linear even when its outputs are called
query, gate or index key by a model. Projection fused with convolution and history publication has
additional semantics and belongs with that algorithm. Several such Ops can form a GDN or QSA
subfamily without becoming one model-layer execution entry.

## 2. Op admission and semantic boundary

An Op is a host-callable, semantically closed computation in the engine execution layer:

```text
(outputs, new_state) = F(inputs, weights, old_state, semantic_parameters)
```

Workspace, CUDA stream, and device facts are execution resources. They may select the
implementation but do not change the result or state transition.

A component is an Op only when all of the following hold:

1. **Logical effect.** It computes or changes a logical tensor value or an explicit local state.
2. **Semantic closure.** Its explicit inputs and metadata completely determine every observable
   output, mutation, valid region, alias restriction, and semantic cast or rounding boundary.
3. **Schedule independence.** It does not decide model call order, request policy, sequence
   frontier, transaction commit or rollback, or state lifetime.
4. **Implementation independence.** Its semantic interface does not expose grids, blocks, warps,
   tiles, launch counts, CUDA symbols, implementation filenames, or model-labelled backend paths.
5. **Independent invocation.** It has a meaningful host-callable boundary and can be qualified
   without running a complete model schedule.

One caller, one registered shape or format, one optimized token extent, one device implementation,
mutable state, or a fused formula does not disqualify an Op. Reuse count and support breadth do not
determine ownership.

Classify a proposed model-callable computation in this order:

1. If it has no logical tensor or state effect, it is infrastructure, validation, planning, or raw
   transfer rather than an Op.
2. If its complete transformation cannot be defined from explicit arguments and metadata, it
   belongs to model or product schedule.
3. If it decides call order, lifecycle, frontier, commit, rollback, or request policy, it belongs
   to Program, schedule, or runtime.
4. If it is only a partial step beneath a complete transformation, it is a private implementation
   detail of the enclosing Op.
5. Otherwise, admit it as an Op.

A useful admission test is:

> If the model name and call site disappear, can the transformation still be described and
> qualified completely?

If the answer still requires a model phase, weight role, or current request frontier, the proposed
boundary is not semantically closed.

### 2.1 Boundaries outside the Op layer

Program, model schedule, and runtime policy own model topology, call order, weight-role and state
instance selection, prompt chunking, multimodal spans, generated-token transactions,
prefix/frontier/commit/rollback policy, persistent state lifetime, CUDA Graph variants, and
publication of product statistics.

Core owns model-independent storage and execution mechanisms: tensor and weight views, checked
layouts, arenas, physical cache containers, CUDA Graph lifetime, and raw host/device or
device/device transfers. A physical container may produce a checked view, but it does not acquire a
logical sequence cursor or transaction policy.

Artifact, frontend, product, and tooling code own framing and materialization, converter recipes,
tokenization and templates, media handling, request translation, transport, diagnostics, and
profiling control. These activities prepare, move, observe, or present data; they do not define a
device transformation.

Wrapper validation, launchers, CUDA entry points, codecs, implementation plans, partial reductions,
staging kernels, workspace layout helpers, and device primitives are parts of an Op implementation,
not independently model-callable Ops.

Every model-callable device transformation belongs to the central Op layer. A model implementation invokes a
contract from `include/ninfer/ops/`, composes existing Ops and core mechanisms, or keeps host-side
schedule composition in the model implementation.

### 2.2 Semantic extents and private routes

When a contract defines an axis as the Text/MTP token extent `T`, it admits every positive value
representable by its views and available storage unless the contract declares a semantic capacity.
Decode, latency-sensitive/hot-interval, and prefill may name private workload or benchmark
regimes. They are not semantic variants, separate model-callable entries, or compute mechanisms.

Other axes retain the finite geometry or capacity declared by their own contracts. A matrix column
does not become Text/MTP `T` merely because an implementation uses the same physical layout.

A wrapper may select a route or split a call into several launches for a valid extent. Private
dispatch thresholds and CUDA grid or resource limits do not narrow the supported semantic domain.

### 2.3 Stateful, fused, and stochastic Ops

A stateful Op defines both its produced value and its state transition:

```text
output    = G(old_state, input)
new_state = F(old_state, input)
```

Its contract identifies every location read or written and whether a newly written value is
visible within the call. Cursor movement, instance selection, transaction publication, commit, and
rollback remain schedule policy.

A fused computation is an Op when its complete composition and all effects form one closed
contract. Only observable intermediate casts or rounding seams are semantic. Tile decoders,
partial reductions, epilogue fragments, and warp primitives remain private implementation details.

A stochastic Op receives every semantic random input explicitly, including the state or seed,
logical position, domain separation, and draw index required by its algorithm. It promises a
backend bitstream or sampled sequence only when the contract deliberately makes that observable.

### 2.4 Execution envelopes

An Op may accept a host execution envelope when a device-resident semantic value cannot be read on
the host without synchronization but affects launch capacity or finite route selection. The
envelope is an execution promise, not a second mathematical input:

- device tensors still determine the exact output and state transition;
- the caller proves that the device value lies inside the declared interval;
- changing to another valid envelope may change capacity or implementation choice, not semantics;
- a captured launch is valid for the complete declared envelope.

Lifecycle cursors and graph-tier policy remain outside the Op. The Op owns only validation,
capacity, and finite dispatch for the explicit envelope it accepts.

### 2.5 Logical assignment and raw transfer

Classification follows the logical interface rather than the CUDA primitive used underneath. A
typed assignment or a copy with cast, transpose, concat, scatter, remap, or another logical index
mapping is the corresponding Op. An interface expressed only as addresses, byte count, and transfer
direction is a core or host transfer.

## 3. Contract headers

Repository-internal contracts live in:

```text
include/ninfer/ops/<family>.h
namespace ninfer::ops
```

They are execution-layer interfaces, not a public product ABI. A header may group closely related
operations or overloads; do not create empty families to force one function per file.

Each semantic Op or closely related overload group has one authoritative comment. Its length
follows the contract's complexity; there is no mandatory list of section headings. For example:

```cpp
/**
 * x[i] = x[i] + y[i].
 *
 * x/y are same-shaped, contiguous, non-overlapping BF16 tensors.
 * Updates x in place and preserves y. Empty tensors perform no device work.
 */
void residual_add(const Tensor& y, Tensor& x, CudaExecutionView execution);
```

The contract must be sufficient to write an independent implementation and oracle. A name such as
“fused projection” or “attention path” is not a formula. A fused contract states the complete
composition; a stateful contract states old and new state; an indexing contract defines every axis
mapping and valid region; a stochastic contract defines its probability process and state effects.

A contract may cite the authoritative tensor-format or layout reference instead of duplicating a
registered encoding, but it still identifies the accepted format and interpretation.

Keep necessary shape relationships, finite semantic capacities, dtype/layout/alignment, alias
rules, observable numerical boundaries, valid regions, state effects and resource requirements.
State when support is limited to registered weight geometries; the implementation's shape catalog
owns the exact geometry list. Kernel choices for particular shapes and extents are private.
Qualification recipes and oracle implementation belong in tests and the qualification rules below,
not in each interface comment. Do not repeat a tensor's description in both prose and parameter
tags, or repeat a complete formula for a capacity query or closely related overload.

The contract describes behavior shared by every implementation. It must not freeze:

- reduction association or thread execution order;
- warp, CTA, tile, or launch decomposition;
- kernel symbols or implementation filenames;
- backend approximation instructions;
- private accumulator, staging, operand, or workspace dtypes;
- private intermediate casts or rounding points; or
- bitwise equality unless an exact semantic format requires it.

Workspace sizing helpers and related overloads refer to the same contract rather than repeating
its formula.

## 4. Implementation organization

Organize implementations under `src/ops/<family>/` and meaningful algorithm subdirectories.
Entries, selectors, launchers and kernels belong to their semantic family; do not create
global `wrapper/`, `launcher/`, or `kernel/` directories.

### 4.1 Responsibilities and selection

| Part | Responsibility |
|---|---|
| Semantic entry | Validate and interpret operands; scope caller workspace; invoke the implementation. |
| Selector | Choose an implementation from format, geometry, extent, semantic variant, envelope and device facts. |
| Shape policy | Own a particular geometry's tuned instances and selection boundaries. |
| Launcher | Bind arguments, instantiate templates, calculate grid/block/shared memory, handle launch limits and errors. |
| Kernel | Compute values and state effects, move data, synchronize and write outputs. |

These are responsibilities, not a requirement for five files. Use `select_*` for implementation
selection, `*_dispatch` for format/geometry routing, and `launch_*` for submitting a selected
implementation. A single-stage selector can return a typed launcher. A multi-stage selector can
return a small family-private Plan containing the selected stages and resource facts. Pass resolved
choices to allocation and launch instead of repeating the same selection downstream. Capacity
queries and execution must consume the same selection and allocation rules.

Runtime selection depends on actual operands and execution facts, never target keys, artifact names,
converter fields, model roles, Program phases or arbitrary registry strings. A launcher may
specialize full tiles versus tails and obey CUDA grid limits; performance policy belongs in the
selector. Device-resident per-row lengths stay Device inputs under the Op's capture contract.

### 4.2 Implementation examples

- **Compact Op:** [elementwise](../../src/ops/elementwise/sources.cmake) keeps each small Op's
  entry, selection, launch and private kernels together in one `.cu`. Use an anonymous namespace
  for local implementation. Extract a `.cuh` only when there is an actual template-sharing or
  source-size reason; do not create forwarding headers and empty plan types.
- **Templated family:** [RMSNorm](../../src/ops/normalization/rmsnorm/sources.cmake) has semantic entries,
  operand validation, one dispatch owner, launch bindings and shared kernel bodies. Ordinary,
  offset and gated mathematical forms share appropriate reductions and epilogues. Tuned dimension
  and workload cutoffs live in dispatch, not in the launch bindings.
- **Large family:** [Linear](../../src/ops/linear/sources.cmake) keeps computation and template
  launch support in `kernels/<format>/`. Ordinary Linear owns `projection/`; LinearSplit owns
  [split/](../../src/ops/linear/split/sources.cmake). Each has its own entry, format dispatch, shape
  policies and launch bindings. Their contraction templates are shared, while selected schedules
  and cutoffs remain private even when identical. Distinct fused Ops follow the same ownership
  rule. Activation preparation and reduction stages remain private to the Op.
- **Multi-stage Op:** [Sampling](../../src/ops/sampling/sources.cmake) gives sampling and acceptance
  their own entries and plans while sharing distribution kernels. [MoE](../../src/ops/moe/sources.cmake)
  gives routing, routed SwiGLU and complete routed/shared MoE separate semantic entries and
  private shape policies. The complete Op composes shared kernel mechanisms without requiring
  intermediate public calls or outputs. Grouping, scratch maps and stage launchers stay private.
- **Algorithm subfamily:** GDN controls, fused input convolution, recurrence and replay belong
  together within [LinearAttention](../../src/ops/linear_attention/sources.cmake). QSA preparation,
  indexing, sparse attention and tail handling form a [SoftmaxAttention subfamily](../../src/ops/softmax_attention/qsa/sources.cmake).
  Shared views connect independent Op interfaces. Model call order and persistent-state ownership
  remain outside these subtrees; pure projections belong to Linear.

New weight geometries extend the corresponding projection catalog and shape policy. A new
contraction pipeline extends the format's kernel library. A new mathematical fusion first gets an
explicit semantic boundary and then reuses suitable kernels at their output boundary. Do not make
different Ops share a tuning table or include a sibling's private shape policy.

### 4.3 Shared computation and compilation

Private launch headers, selectors and kernel templates are not included by models, contract
headers, product code, mathematical qualification tests or public benchmarks. The owning suite
may use the contract-specified execution as an exact-equivalence reference described in §6.1.
Every kernel assumption has a matching admission or launch predicate. An Op never infers it from
model identity.

`src/ops/common/` owns the shared implementation foundation:

- `numeric/`: scalar arithmetic, explicit approximation profiles, represented-value conversions
  and vector types;
- `cuda/`: memory instructions, reductions and ordering, MMA, asynchronous pipelines and
  shared-memory storage;
- `quant/`: packed-format decoding, encoding and explicit scale/rounding profiles;
- `host/`: operand checks, represented-weight validation and launch resource configuration;
- `cpu/`: ISA-specific vector conversions, reductions and packed-weight decoding; included only
  by compilation units built for that ISA.

Keep small companion facilities together: vector types with conversions, cache policy and swizzles
with memory operations, and asynchronous copy with its synchronization. Split a header for a real
dependency boundary, independent use or substantial implementation size, not for each helper.

These facilities depend only on core, CUDA and other shared facilities, never on an Op family.
Include the specific facility directly. Do not add a family-local forwarding `math` or `utils`
header, duplicate a generic primitive, or put selected schedules and tuning cutoffs in this layer.
Core remains the authority for storage types, layout contracts and codebooks. Device codecs must
keep format interpretation separate from matrix-tile or paged-cache addressing. Quantization
profiles with different scale bounds, divisors or rounding boundaries remain explicit.

Family kernel libraries own reusable algorithms, tile assembly, output mappings and epilogues.
When several routes share a computation, place it in that library rather than making one route
depend on a sibling's private implementation.
For example, an E4M3 conversion belongs to shared numeric/format support; a GEMM pipeline belongs
to Linear. Sharing either does not introduce a model-callable private backend or an intermediate
numerical boundary. The consuming Op chooses its arithmetic profile and tuned implementation.

Preserve per-shape compilation units and explicit template ownership. File organization must not
silently change RDC mode, architecture, compiler precision options or device-link ownership.

### 4.4 Execution-target ownership

The build selects one native execution target; see [Build system](build-system.md). Keep the
mathematical interface and caller-owned resources independent of that selection. A small portable
Op stays in its existing source. Split only where target-specific dispatch, instantiation or launch
requirements need independent ownership; not every family needs platform subdirectories.

Within an affected Op, `rtx5090/` and `dgx_spark/` own their private shape catalogs, schedules and
launch bindings. CMake includes the selected source manifest. Shared kernel templates, codecs,
operand contracts and mathematical oracles remain outside those directories when useful to both.
Each target owns its tuning independently. The compiled Op defines its supported inputs;
workspace queries and execution use the same admission rules and report unsupported calls there.

Keep a compatible implementation shared until a target actually needs independent selection or
launch policy. Creating target directories must not remove an existing supported route. Workspace
layout consumes the selected implementation's scratch extents; public interfaces remain independent
of platform selection.

For small compile-time differences, explicitly include `core/platform.h` and use its target or
capability macros according to the fact being tested. Keep substantial selectors in target-owned
sources rather than scattering preprocessor branches across kernels. Hardware reference values
for benchmarks live in `bench/common/roofline.h`; executed arithmetic and tuning decisions remain
with the affected Op/benchmark. A missing hardware reference leaves the derived utilization empty.

## 5. State, workspace, naming, and dependencies

### 5.1 State and execution views

Ops receive non-owning execution views. They inspect only facts required for execution, such as
dtype and numeric format, storage layout, logical and padded shape, payload planes, quantization
geometry, and alignment.

Ops do not receive artifact object names, converter source fields, model weight roles, recipes, or
provenance. Artifact binding and model loading translate those concepts into explicit execution
views.

A weight preparation entry resolves weight views and activation permissions into operands for a
supported native overload, checking its parent geometry, format, layout and shape. See
[`weight_input.h`](../../include/ninfer/ops/weight_input.h). Model execution composes the prepared
Ops; resource queries and execution check their applicable contracts.

Program owns persistent state instances and lifetime; core owns model-independent physical
containers; an Op owns only the documented reads, writes, outputs, and state transition of one
call.

CUDA entries receive the `CudaExecutionView` propagated from the owning `DeviceContext`.
Validate operand residency with the family's existing shape/layout checks, including active
optional planes and workspace. Thin overloads delegate to that complete validation. Private
launchers may receive the extracted stream; a composing Op forwards the execution view.
Raw device pointers in explicit CUDA descriptors belong to that same execution device and remain
the caller's lifetime responsibility. No entry infers execution from an operand or queries CUDA
pointer attributes on the launch path.

CPU entries receive `CpuExecutionView` from the owning fixed `CpuExecutor` and borrow CPU tensors,
native weight banks and caller scratch until synchronous return. Select ISA implementations using
the bound workers' common features. Keep advanced ISA flags local to kernel compilation units;
the caller-side entry and selection remain baseline C++. Workers neither allocate storage nor
call CUDA. Program owns CPU/GPU handoff and overlap, not the Op or executor.

### 5.2 Workspace

Caller-owned `MemoryArena&` or an explicit caller-owned scratch view is the transient workspace
boundary. An Op may suballocate within its call scope, but it must not allocate hidden device
memory, retain a scratch pointer, or own the arena. The arena borrows its backing allocation;
its capacity and cursor do not confer ownership or completion.
For CUDA entries, even a zero-capacity arena carries the execution device. Zero scratch means
no allocation, not an unlocated workspace. Absent optional tensors and explicitly supported
zero-work inputs follow the individual Op's contract and need no backing.

An entry with statically zero scratch omits both the arena argument and a sizing query. A
scratch-capable entry accepts caller-owned workspace and exposes one authoritative capacity query
for its fixed host-visible profile and inclusive extent interval.

The query returns the minimum high-water capacity sufficient for every legal extent in that
interval. An exact-call query uses equal interval endpoints. Invalid profiles and intervals throw;
zero is reserved for a legal route that requires no caller-owned global scratch.

Capacity queries consume host-visible geometry and explicit hardware facts such as SM count.
They do not require a stream or a synthetic execution view. Planned tensors have no residency;
binding to a real backing span supplies it.

Account for route transitions and nonmonotone regions when computing the interval maximum. An
upper-endpoint query suffices only for a region whose allocation is known to be nondecreasing.

Capacity is measured from a 256-byte-aligned Op-local cursor, includes internal alignment gaps, and
excludes trailing padding. The sizing query and execution share the same route facts and allocation
recipe. A composed Op reports its complete public scratch requirement rather than exposing private
child requirements.

Scratch regions may share storage when their device lifetimes do not overlap. Keep the last-reader
and synchronization requirements beside the layout and affected launch or kernel code; preserve
them when separating stages or compilation units.

Ending the host workspace scope does not end device use. Kernels that consume a suballocation must
be submitted on the declared stream before the scope is released, and reuse relies on stream
ordering.

### 5.3 Naming

- Name an Op after its mathematical transformation or explicit state transition, not its first
  model, layer role, schedule phase, or CUDA strategy.
- Use `ninfer::ops` for semantic entries. Use `ninfer::ops::detail` only for private material that
  must cross translation-unit boundaries.
- Name an implementation specialization by real match facts such as format, geometry, extent,
  device capability, or algorithm.
- Reserve `kernel` for CUDA implementation concepts.
- Prefer one family name across contract, implementation, tests, and benchmarks unless several
  contracts intentionally share an implementation family.

### 5.4 Dependencies

The dependency direction is:

```text
core <- ops <- model execution <- runtime/engine product route
```

Artifact and loading code may materialize execution views for a model, but the Op layer does not
depend on artifact provenance or model binding concepts.

Enforce the boundary in code and build ownership:

- contract headers include only required core, CUDA host, and peer semantic contract types;
- `src/ops/**` does not include model, Program, schedule, product, or artifact-provenance headers;
- model execution includes contract headers, never private launcher, kernel, common, codec, or plan
  headers;
- `ninfer_ops` does not link a model implementation;
- core and artifact do not link Ops or models;
- explicit source lists give every implementation one build and link owner.

The owning family registers its implementation in `src/ops/<family>/sources.cmake`; large families
delegate to explicit local manifests for their kernel library and semantic implementations.
Ordinary sources contribute to `ninfer_ops`; NVFP4 sources requiring non-RDC compilation contribute
to `ninfer_ops_non_rdc` from their owning manifests. Shape files retain their `.cpp` or `.cu`
language and separate translation units. See [Build system](build-system.md) for the target policy.

## 6. Qualification

Semantic Op tests live under `tests/ops/` and invoke the public contract independently of model call
order. They link the Op layer and required core libraries rather than a model implementation. Commands and
common reporting behavior belong in [`tests/README.md`](../../tests/README.md).

A suite is owned by a semantic Op or closely related overloads. Group executable entry points
by useful family boundaries; split files for substantial fixtures, oracles or case catalogues,
not for every helper. Keep one non-overlapping default CTest set. Metadata-only case declaration
and filtering precede device initialization and fixture construction. Cases have stable IDs and
selection-independent input recipes; a stateful or Graph sequence is one selectable case.

### 6.1 Oracle

Choose the reference and comparison from the observable promise of the Op. A tensor's floating-point
dtype alone does not determine its acceptance test. Identify the outputs, state effects, explicit
representation boundaries, and the realistic failure that the check must detect before selecting
an oracle.

For an Op whose promise is a mathematical computation, use one independent naive FP32/FP64 oracle over the logical
values represented by its public inputs. Test-owned fixture code independently decodes packed
values before invoking the oracle. The oracle evaluates the complete formula at high precision and
retains that result. It does not reproduce a production route's staging casts, activation
quantization, reduction tree, workspace dtype, public output rounding, or another implementation's
output.

Exact transforms and codecs use an independent exact oracle. A fused oracle evaluates the complete
fused formula instead of composing production Ops. A stateful oracle computes both output and new
state. For such mathematical contracts, another GPU route, generated model output, or pairwise
implementation parity is supplementary evidence, not the mathematical oracle.

When the contract instead requires exact equivalence to a specified execution, that execution is
the reference. ReplaySSM record/fold must preserve the specified snapshot or recurrent outputs and
committed state bit for bit. Compare the same initial state, physical block, inputs, and arithmetic
policy; for each committed prefix, select the corresponding snapshot from that same block. Re-running a
shorter projection can choose different arithmetic and is not an equivalent reference. Check raw
record copies, untouched state, invalid tails, and zero-commit effects according to their contracts.
A separate FP64 recurrence does not establish this equivalence and is not required for its acceptance.
If the snapshot computation itself changes, validate its mathematical contract separately.

The oracle determines correctness but does not prescribe production arithmetic. Private precision,
instruction operands, reduction association, staging, workspace representation, and kernel
decomposition remain implementation choices unless the contract makes an intermediate value
observable.
Unless the contract requires an intermediate representation boundary, a fused implementation must
not retain extra low-precision intermediate rounding to reproduce an unfused route or achieve
bitwise parity.

### 6.2 Conformance domain

One semantic Op or closely related overload group owns one identifiable qualification suite. Its
finite conformance matrix covers:

- every public entry and registered semantic dtype, format, layout, mode, state form, and real
  geometry affected by the contract;
- every reachable production route through the public wrapper, at its real match boundaries;
- representative inputs and semantic or route boundaries that can change results;
- every output and documented mutation, including untouched regions and preserved inputs when
  promised.

For an unbounded positive Text/MTP `T`, qualify `T=1`, each private route boundary at `b-1`, `b`,
and `b+1` where valid, and a representative interior extent for each route. Other axes use their
declared finite domains and relevant boundaries. Do not build a redundant Cartesian product when
one case establishes several dimensions.

Prune redundant sweeps by the independent risk they cover. Keep numerical stress histories,
state/alias/stride/page boundaries, and each compiled CPU arithmetic route. Cheap finite-domain
metadata checks can remain exhaustive. Reuse represented fixtures and parallelize independent
oracle outputs without changing their accumulation order. Large independent-output computations
may sample fixed output coordinates while evaluating their complete formula and retaining full
checks of exact effects, valid writes, untouched regions and guards. Do not truncate a recurrent
history to reduce oracle cost.

When CUDA Graph capture/replay is part of the public execution contract, qualify the captured
public Op and its observable effects against the same oracle, including changed inputs or metadata
at the captured addresses.

Keep schedule composition, persistent-state lifetime, and end-to-end behavior in model or product
integration tests. Private candidates, plan choices, launcher symbols, filenames, and source
organization are not qualification subjects.

### 6.3 Acceptance criteria

Exact outputs and regions compare every observable bit or byte. Each approximate output or state
uses a suite-owned named criterion for its public numerical policy or representation; tests do not
define per-case tolerance overrides. For Linear-family Ops, `A16Only`, `AllowA8`, and `AllowA4`
select the A16, A8, and A4 criteria respectively, regardless of which arithmetic the implementation
chooses. A permissive policy retains its criterion when a route uses higher precision. Input
recipes and sampling are separate from that choice; tests must not inspect selectors or infer
precision from T to select a criterion.

A floating-point criterion states its non-finite policy. Elementwise work uses a pointwise bound.
Reductions and matrix or attention computations use a normwise bound with a finite gross
pointwise-error cap so cancellation does not require strict elementwise agreement and isolated
corruption cannot hide in an aggregate norm.

Different public numerical policies or represented formats may require different criteria.
Widening a criterion requires a numerical reason and requalification
of its complete affected domain; one failing implementation is not sufficient justification.

## 7. Performance evidence

An Op microbenchmark measures the public semantic operation at an exact shape, format, layout,
extent, device/toolchain, and stated cache and timing condition. It is implementation evidence, not
a correctness oracle or proof of end-to-end improvement. Commands and executable-specific behavior
belong in [`bench/README.md`](../../bench/README.md).

A long-lived Op benchmark calls the public contract and, when applicable, its public workspace
capacity query. It does not include private implementation headers, call private launchers, expose
candidate or kernel forcing, or duplicate candidate legality and production dispatch tables.

The Gated DeltaNet benchmark is an explicit exception: alongside public Op measurements, it may
call the selected production `prepare` and `recurrence` stage launchers with their private
workspace layout, or force recurrent and chunked routes at matching prefill extents to measure
their crossover. It may also measure the chunked route with pre-normalized BF16 Q/K. These
controls qualify the production prefill decomposition and do not change decode or ReplaySSM;
their command-line interface is defined in [`bench/README.md`](../../bench/README.md).

Candidate comparison is task-local development work. A temporary sweep may call private launchers
and encode the exact overlapping candidate domains needed for a decision. Measure candidates under
the same inputs, cache, device, and timing conditions; put the selected winner or crossover in
production dispatch; verify final correctness and performance through the public Op; then delete
the losing candidates, temporary controls, and comparison-only entry points. The selected
production implementation is not temporary merely because it originated in the sweep.

Construct that sweep as one candidate-by-extent matrix whenever the candidate domain is known at
compile time. Compile the complete decision set together and collect every relevant extent in one
run; do not emulate a sweep by repeatedly editing one template instance, rebuilding, and timing a
few points. A second measurement pass is warranted only when the first result is invalid or
inconclusive, or when it identifies a materially new kernel family whose result can change the
decision. Changing one knob at a time after the decision domain is already known is not additional
evidence.

### 7.1 Route-development transaction

Develop a new optimized route as one vertical transaction around a representative registered
problem. First establish the public admission, validation, workspace query, independent oracle
case, and public benchmark point needed to exercise that problem.

Keep execution mechanisms and workload regions as independent dimensions. SIMT, Tensor Core MMA,
and other instruction or decomposition choices describe how a kernel computes. A single-token
point, a latency-sensitive interval, and a throughput anchor describe where an implementation is
measured. Do not treat a range label such as "small-T" as a compute mechanism, assume that MMA is
restricted to large extents, or require one kernel family per workload region.

Choose the order of mechanism exploration from the live performance question rather than a fixed
smallest-to-largest sequence. In particular, it can be useful to establish an accelerator route at
the primary throughput anchor before exhaustively tuning the latency-sensitive interval. Its
measured lower-extent behavior then bounds where further non-accelerator optimization is useful.
This ordering does not determine the eventual crossover: the latency-sensitive sweep later
compares every relevant mechanism, and an MMA route may or may not enter that interval.

When a low-precision MMA mechanism requires a private activation representation, activation
quantization belongs to that mechanism rather than to the workload region where it happens to win.
On-chip quantization inside the contraction and a separately launched materialization consumed by
the contraction are distinct execution decompositions. Hold the quantization formula, scale
granularity, and scale representation constant when attributing a result to that decomposition. If
one decomposition requires a different arithmetic profile, qualify it separately against the
oracle and report the comparison as a profile-plus-decomposition decision. Treat multiple
decompositions as candidates only while the choice remains a live performance question. A
qualified complete public route that reaches the relevant hardware roofline within measurement
uncertainty can close that question without implementing another decomposition; never require an
alternative whose only possible benefit would be to exceed the roofline. Otherwise compare the
complete launch/workspace traffic of the plausible alternatives, and do not select a route from
contraction-only timing.

Each kernel family may expose compile-time schedule parameters that distinguish concrete,
plausible candidates. Instantiate only the small overlapping candidate set needed to answer a live
decision; do not create a Cartesian product of speculative knobs. Add another family or parameter
only when evidence shows that the existing candidates cannot cover a relevant part of the
workload. Once dispatch is selected, retain the winning instances and parameters and remove losing
candidate instances and temporary tuning entry points. Reusable kernel families and schedule
parameters with a clear purpose and development validation may remain as tuning assets even when
current production dispatch does not select them. Qualify new configuration/epilogue combinations
during development; permanent tests protect public behavior and production routes, not an unused
configuration-by-epilogue matrix.

Derive latency-sensitive extents and bulk anchors from the active workload and the Op's actual
input semantics. Review pointwise behavior and material route seams; select a small set of useful
implementations, accepting justified tile/CTA-wave steps. A permissive activation policy does not
identify the arithmetic actually executed, so use the selected implementation's precision when
interpreting throughput against a hardware peak.

[Linear tuning and performance reports](linear-tuning.md) defines the Linear-specific T ranges,
priority points, specialization tradeoffs, and final report format.

Choose the development surface from the requested complete Op. A related simple Op can help
isolate shared computation when that answers a live design question, but completing a separate
simple-Op tuning campaign is not a prerequisite for a fused Op. Reuse suitable kernel bodies at
their output boundary when useful, and evaluate plausible routes through the complete public fused
Op. Its epilogue, post work, workspace traffic, outputs, and state effects determine its selected
route and completion evidence; an isolated contraction result cannot substitute for them.

Before timing, qualify each candidate arithmetic profile against the independent oracle. After
encoding the selected instances and boundaries in production dispatch, requalify boundary and
interior cases and remeasure the latency curve or throughput anchor through the public Op. The
temporary sweep and its private entry points are then removed as described above.

An Op-scoped performance claim ends at the public Op boundary. Exact formats, layouts, shapes, and
extents can be constructed directly by the Op benchmark; they do not authorize loading a model
artifact or invoking a model, Program, Engine, or whole-round benchmark. Product-route evidence is
required only when the requested deliverable explicitly makes an end-to-end claim and includes
that product route in scope.

For a performance change:

1. establish correctness for the affected contract or route;
2. measure the relevant Op and workload extent;
3. when an explicitly scoped claim is end-to-end, measure the authorized public product route;
4. for that end-to-end work, use whole-inference profiling only when attribution is unresolved;
5. use kernel profiling only after a specific kernel-level question is identified and its answer
   can change the Op implementation decision.

Preserve only the context needed to interpret the result, as required by `AGENTS.md`.

## 8. Change checklist

For a new or changed device transformation, apply the relevant contract checks below. They do not
require separate artifacts or a fixed execution order, and unchanged contracts need not be
rewritten:

1. classify the complete semantic boundary and reject schedule decisions, raw transfers,
   container lifecycle operations, and partial implementation helpers;
2. extend an existing family for a closely related overload or variant, and create a new family
   only for a distinct closed transformation;
3. define formula or indexing, logical shapes, supported domain, numeric boundaries, effects,
   aliasing, state, randomness, and workspace before selecting CUDA organization;
4. keep Text/MTP `T` semantic and positive unless the contract declares a capacity, while preserving
   the declared domains of other axes;
5. place the entry, selector, launch bindings and kernels with their owning family; use the compact,
   templated or large-family organization appropriate to the computation;
6. keep persistent state, graph lifecycle, and schedule policy outside the Op;
7. qualify every affected public entry and reachable production profile directly against the
   independent oracle;
8. measure the Op when performance changes; measure a product route only for an explicitly scoped
   end-to-end claim;
9. integrate models only through semantic contract headers and explicit operands;
10. give every source and symbol one clear build and link owner.

A contract change updates the authoritative comment, affected implementations, callers whose
assumptions changed, and tests protecting the changed behavior. A faster private implementation
under an unchanged contract changes private dispatch and implementation code plus the evidence
needed for that route; it does not create another semantic entry.
