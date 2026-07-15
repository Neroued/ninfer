# NInfer Core Engine Architecture

> Status: accepted and implemented for the registered `qwen3_6_27b_rtx5090` target.
>
> Authority: this document defines the NInfer core engine boundary, exact-target registration
> and dispatch, load-time construction, memory ownership, checkpoint frontend boundary,
> single-request program contract, decode-round transaction, sequence-state invariants, and the
> division among common infrastructure, central Ops, and target-private schedule/state policy. It
> also defines the repository/source ownership, internal header visibility, target-package layout,
> and build dependency direction that enforce those boundaries. It does not define model
> mathematics, persistent tensor numeric semantics, `.ninfer` binary framing, model-specific object
> inventories or layouts, conversion recipes, serving protocols,
> or future concurrent scheduling.
>
> Project purpose comes from [`ninfer-project-positioning.md`](ninfer-project-positioning.md).
> Persistent numeric formats come from [`ninfer-tensor-formats.md`](ninfer-tensor-formats.md), and
> artifact framing and metadata come from
> [`ninfer-container-format.md`](ninfer-container-format.md). Exact Qwen3.6 checkpoint mathematics
> remain in [`qwen3.6-27b-architecture.md`](qwen3.6-27b-architecture.md) and
> [`qwen3.6-35b-a3b-architecture.md`](qwen3.6-35b-a3b-architecture.md). The concise implemented-system
> overview is [`design.md`](design.md); this document is the detailed engine contract.

The native `.ninfer` converter, Python artifact and reference layers, narrow C++ reader, 27B
binder, exact target package, `Program`, common generation controller, public `Engine`, CLI, server,
and benchmark route implement this boundary for the registered 27B target.

## 1. Decision

NInfer is built around **compiled exact-target programs**, not a generic model `forward()` API.
The unit selected at load time is:

```text
exact checkpoint semantics + selected GPU implementation
```

Every supported pair supplies one statically compiled target package. That package owns its typed
weights, native input representation, persistent sequence state, memory formulas, Text/Vision/MTP
schedule, CUDA Graphs, prefix-reuse policy, and decode-round implementation. Common engine code owns
only the mechanisms and lifecycle that are genuinely identical across selected targets.

The package schedule composes repository-internal Op contracts. Mathematical CUDA implementations,
including exact-shape and selected-device specializations, remain centrally owned by the Op layer;
their specialization does not create target-private operators.

The stable boundary has three principal parts:

1. a **target package** turns a validated `.ninfer` directory into one immutable, typed loaded
   product for the actual GPU;
2. a **checkpoint frontend** turns product-level input into one fully owning, target-typed prepared
   prompt and supplies output-decoding semantics;
3. a **single-request program** owns all mutable GPU and host state for one resident sequence and
   exposes begin, transactional decode-round resolution, finish, and invalidation operations.

The common generation controller is compiled around this contract. It owns output limits, stop
conditions, cancellation, output preview/commit ordering, and publication. It does not own model caches, recurrent
state, speculative details, model positions, or model-specific memory arithmetic.

The central shape is:

```text
Engine
├── DeviceContext
├── artifact::Reader + binder/materializer
├── LoadedProduct                         loaded-Engine lifetime, immutable after construction
│   ├── exact-checkpoint Frontend
│   ├── typed persistent resources
│   ├── typed device weights
│   └── exact checkpoint × GPU implementation
├── Program                               exactly one resident/active sequence
│   ├── target Text/MTP caches
│   ├── target recurrent state
│   ├── sampling state and stable outputs
│   ├── graph-stable buffers and graphs
│   ├── target workspaces
│   └── target prefix ledger/checkpoints
├── RequestMemory                         grow/reset only before target execution
└── GenerationController                  one active request
    ├── output budget and stop policy
    ├── cancellation
    ├── output preview/commit
    └── publication/streaming
```

This is deliberately neither a plugin ABI nor a dynamic graph runtime. Adding a target means adding
compiled code and an explicit registry entry, then proving that target. It does not mean teaching a
generic runtime how to infer an execution graph from artifact metadata.

## 2. Scope and non-goals

### 2.1 Current execution scope

This architecture covers:

- one process-local loaded product;
- one selected GPU;
- one `Program` with at most one active request;
- an optionally reusable resident sequence between successive local requests;
- exact-checkpoint native Text, Vision, MTP, recurrent, and sampling behavior as applicable;
- synchronous GPU ownership even when transport and preparation run on other CPU threads;
- one current optimized path for each registered exact checkpoint and GPU pair.

The design must fit both the existing Qwen3.6-27B path and the planned Qwen3.6-35B-A3B path without
putting either checkpoint's dimensions or phase vocabulary into the common API.

### 2.2 Explicit non-goals

This document does not reserve interfaces for:

- continuous batching, request admission, preemption, or fairness;
- paged multi-request KV allocation or a global prefix cache;
- multi-GPU, distributed execution, CPU offload, or model sharding;
- arbitrary Hugging Face configuration ingestion;
- third-party runtime plugins or binary compatibility for target packages;
- an operator graph IR, public phase enum, or string-selected model schedule;
- load-time weight quantization, persistent repacking, or generic dense fallback;
- multiple user-selectable quality or kernel profiles for one target;
- stable project-owned source or binary compatibility.

Future continuous batching is an important possible project phase, but it is a different scheduling
architecture that must be designed for real high-performance batching. It must re-prove, per
sequence, exact-target ownership, accepted-token/publication consistency, and the
logical-versus-materialized relation at every resumable/reusable boundary. It may replace the
singleton `Program`/`GeneratedRound` API, add internal states, and hold work for many sequences at once.
That future does not justify adding request IDs, batch slots, page tables, schedulable phases, or
concurrency locks to the current single-request contract.

## 3. Terms

### 3.1 Exact checkpoint

An **exact checkpoint** is the model-semantic identity selected by `.ninfer` `model_id`. Dimensions,
layer topology, modality behavior, token domain, MTP alignment, cache semantics, and native behavior
are fixed checkpoint facts. Two family members with different dimensions are different checkpoints.

### 3.2 Target package

A **target package** is one compiled implementation of an exact checkpoint, selected GPU, and
complete persistent storage-consumer profile. It is a C++ type satisfying the internal contract in
this document. Its name/profile is a source-level identity derived from the artifact's existing
object signatures, not a new artifact field and not a public ABI.

One checkpoint may eventually have more than one target package only because different selected
GPUs are different optimization targets. For one exact checkpoint and one selected GPU, exactly one
current complete storage-consumer profile exists. Changing that profile replaces the old
project-owned package and artifact route; it does not retain a compatibility or quality alternative.
Each package accepts exactly one complete role-to-format/layout/encoding profile; file offsets/order
may vary as the container permits. No Cartesian product of known models, formats, layouts, and GPUs
is implied.

### 3.3 Loaded product

A **loaded product** is a successfully validated immutable subobject of a target package. It owns
typed weights and persistent resources for the loaded lifetime of its `Engine`. It contains no
active request state and is not independently published: the Engine can reach it only after the
complete `TargetInstance` is ready.

### 3.4 Prepared prompt

A **prepared prompt** is one fully owning CPU-side input produced by the loaded product's frontend.
Its concrete type is checkpoint-private. It may contain token IDs, checkpoint-specific position
data, preprocessed media, modality placement, prompt boundaries, and other facts required to begin
that exact target.

### 3.5 Program

A **program** is the non-movable execution object for one loaded product and one configured context
capacity. It owns every mutable host/device object whose meaning depends on the resident sequence.
At most one request is active and at most one decode round is unresolved.

### 3.6 Logical sequence and execution frontier

The **logical sequence** is the target-accepted token sequence: prompt tokens followed by generated
tokens that the controller has committed. It includes terminating tokens even when their text is
withheld from publication. The **execution frontier** is the prefix for which target
model state has been processed. A sampled next token may be logically present without yet being
processed by the target model. Section 11 defines the exact relation.

### 3.7 Decode round

A **decode round** is one target operation that returns one or more target-licensed next tokens. It
may be an ordinary one-token step or a target-private speculative MTP operation. Its provisional
device mutations are not reusable until the controller resolves the returned round.

## 4. Stable boundaries and ownership

### 4.1 Common engine owns mechanisms

Common engine code owns:

- selected-device discovery, stream ownership, and device error policy;
- strict `.ninfer` framing/JSON/range parsing;
- temporary artifact name indexing and checked object descriptors;
- generic host/file reads and device-copy execution from a target load plan;
- checked byte arithmetic, alignment, arenas, and layout-building primitives;
- target registry lookup by `(model_id, actual device)`;
- opaque public envelopes for prepared prompts and loaded implementation state;
- pure request option values and target-independent result summaries;
- the single-request state-machine obligations;
- output-budget, stop, cancellation, decoding, and publication control;
- common diagnostics and measurement plumbing whose semantics do not expose target internals.

### 4.2 Target package owns semantics and policy

Each target package owns:

- the complete expected persistent object and resource inventory;
- accepted shape, format, layout, encoding, and actual-GPU combinations;
- typed weight roles, fused views, tied bindings, and device descriptors;
- checkpoint token domains and output-row validity;
- tokenizer/template/media preparation and checkpoint-native positions;
- all persistent and transient memory formulas for that target;
- Text, Vision, MoE, recurrent, MTP, sampling, and final-head schedule;
- physical KV/recurrent state types, cursors, snapshots, and commit rules;
- prefix-reuse matching, checkpoints, restoration, and invalidation;
- CUDA Graph choice, graph-stable bindings, warmup, and recapture policy;
- decode-round maximum yield and context-tail fallback;
- target-specific counters and detailed diagnostics.

### 4.3 Generation controller owns user-visible policy

The common generation controller owns:

- requested output-token limit and the target-approved effective limit;
- caller-provided and frontend-default stop token/string handling;
- cancellation at defined round boundaries;
- non-mutating preview of decoded text from a returned token batch;
- resolving a decode round before publishing corresponding output;
- publication to the CLI, local server, or another product adapter;
- final user-visible token/text accounting.

Stop strings are not model execution state. MTP acceptance is not output policy. Keeping them in
their respective owners prevents a model program from calling transport code and prevents the
controller from modifying speculative caches.

### 4.4 Ownership rule

An owner belongs to the lowest layer that can state its complete invariant. Checkpoint topology,
weight-role binding, operand/view selection, model positions, recurrent/MTP instance lifetime,
Vision composition, prefix policy, and graph capture are target-private. Bytes, alignment,
device/storage lifetime, and committed-output mechanisms belong to lower common layers.

A host-callable tensor or explicit local-state transformation is classified separately by the Op
rule: if its complete values and effects can be defined from explicit arguments without schedule
context, its contract and every implementation are centrally owned. One caller, an exact model
shape, or one selected GPU does not require a second-target reuse proof. Similar schedule or state
objects still require matching real invariants before extracting a shared family abstraction.

## 5. Closed target registry and dispatch

### 5.1 Registry key

After generic artifact validation, the registry follows this closed sequence:

```text
artifact.model_id -> compiled package alternative -> package preflight on the actual device
```

The artifact does not carry a GPU selector. The registry matches `model_id` to one explicitly
compiled package, then that package checks the observed device and options. Its binder consumes the
complete directory and validates every canonical name, kind, shape, format/layout or resource
encoding before payload-sized allocation. Offsets and object-array order are not package keys beyond
the generic container geometry rules.

The current registry has one alternative. Adding another target extends the same explicit selection
sequence; it does not introduce a separate runtime profile object or a profile field in JSON. An
unknown model, wrong GPU, unavailable device capability, or unsupported storage signature is a load
error before payload-sized allocation.

### 5.2 Internal static package contract

The package contract is compile-time and intentionally coarse. The following is the normative
shape; concrete helper names may vary only where the same ownership and call ordering remain clear:

```cpp
struct SomeExactCheckpointRtx5090 {
    static constexpr std::string_view model_id;

    struct LoadPlan;
    struct LoadedModel;
    struct Frontend;
    struct PreparedPrompt;
    struct OutputSession;
    struct SequencePlan;
    struct RequestPlan;
    class Program;

    static void preflight(DeviceContext&, const EngineOptions&);
    static LoadPlan plan_load(artifact::Binder&);

    static std::unique_ptr<LoadedModel> construct_loaded_model(
        LoadPlan&&,
        artifact::MaterializedArtifact&&);

    static Frontend make_frontend(const LoadedModel&);

    static SequencePlan plan_sequence(
        const LoadedModel&,
        DeviceContext&,
        const EngineOptions&);

    static std::unique_ptr<Program> create_program(
        const LoadedModel&,
        SequencePlan&&,
        DeviceContext&);
};
```

The nested type names are abbreviated in this conceptual listing; they are not permission for the
physical `package.h` to contain only forward declarations. Every target type that composition code
must create, store, destroy, or call while instantiating `LoadedProduct<T>` and the whole generation
loop is a complete contract-facing facade in that header. Its method signatures and object lifetime
are visible, while target-private weights, state, schedules, and graph/lifecycle policy remain
behind an opaque implementation owned by the target library. Central Op implementation selection
remains behind the Op contracts. Section 19 defines the corresponding
export/implementation split.

`preflight` performs cheap target-owned device and option checks. `plan_load` consumes and validates
the complete artifact directory. It does not receive feature or quality toggles: one package
represents the one current complete product route. `construct_loaded_model` may run only after every
selected object is structurally validated and materialized. It allocates the final heap-stable
`LoadedModel`, moves the backing into that object, and forms typed views only after every subobject
they may reference is at its final address. It must
not return a by-value self-referential aggregate or move an already-bound object into the returned
allocation. `plan_sequence` computes all target-state and stable-workspace requirements against the
post-load device capacity without mutating the loaded model. `create_program` performs the one-time
allocation/binding/capture setup for a single stable-address program.

The package contract is not a virtual base class. It does not contain `forward`, `prefill`,
`run_vision`, `propose`, or `verify` calls exposed to common code.

### 5.3 Whole-loop dispatch

NInfer uses a closed internal sum type behind the public `Engine` PIMPL, conceptually:

```cpp
template<class Target>
struct LoadedProduct {
    std::unique_ptr<typename Target::LoadedModel> model;
    typename Target::Frontend frontend;

    explicit LoadedProduct(
        std::unique_ptr<typename Target::LoadedModel>&& stable_model)
        : model(std::move(stable_model)),
          frontend(Target::make_frontend(*model)) {}
};

template<class Target>
struct TargetInstance {
    std::unique_ptr<LoadedProduct<Target>> loaded;
    RequestMemory request_memory;
    std::unique_ptr<typename Target::Program> program;
};

using ActiveTarget = std::variant<
    std::unique_ptr<TargetInstance<TargetA>>,
    std::unique_ptr<TargetInstance<TargetB>>>;
```

Each `TargetInstance<T>` contains an immutable heap-stable `LoadedProduct<T>`, separate reusable
request memory, and its one `T::Program` in that declaration order. C++ reverse destruction therefore
destroys Program first, request memory second, and loaded
resources last. `LoadedModel` is itself a stable heap object: target binding forms no reference to a
movable owner or descriptor subobject, and `LoadedProduct` constructs `frontend` from the final
`*model` only after installing that owner. The program and frontend may then refer to immutable
loaded resources without depending on aggregate moves. `Engine::Impl` similarly declares
`DeviceContext` before `ActiveTarget`, so the complete target dies before its device/stream owner.

Every compiled exact target is an alternative of the same outer closed sum type; there is no inner
layout tag or string dispatch in `LoadedModel`/`Program`. A package's accepted storage signature is
part of its binder contract, not a second runtime alternative inside the package.

The registry constructs one variant alternative. Generation performs one `std::visit` around the
complete target-templated generation loop. It does not visit or make a virtual call once per layer
or token. This gives a single explicit list of compiled targets, complete types inside each
implementation, and no model-shaped public headers.

A coarse generated function table would be acceptable only if it preserves the same one-time
dispatch and typed target-local loop. A per-operator or per-phase polymorphic ABI is not acceptable.

### 5.4 Registry evolution

Adding a package requires, in one coherent change:

1. registering its exact `model_id` semantics under the container registry process;
2. defining its accepted persistent inventories and layouts outside this document;
3. implementing the package contract and exact device check;
4. proving model/block/end-to-end correctness and quality for its chosen formats;
5. proving memory capacity, prefill, decode, native modalities, and MTP where applicable;
6. adding it to the single closed dispatch registry.

Recognition of a related family config is not support and must not be used as a fallback.

## 6. Artifact loading and target construction

### 6.1 Load pipeline

The load pipeline has this dependency order:

```text
read and validate the v1 object directory
  -> select one compiled package by model_id and preflight the actual device/options
  -> target plan_load consumes the complete object directory
  -> validate inventory, roles, shapes, formats, layouts, encodings, consumers, and cheap
     target-specific payload invariants
  -> compute host/device materialization and target-private binding plans
  -> allocate and materialize persistent objects
  -> construct LoadedModel at its final heap address, move in backing, then form typed bindings
  -> construct LoadedProduct around that stable model and construct Frontend from retained resources
  -> plan sequence memory against post-load device capacity
  -> construct RequestMemory
  -> allocate TargetInstance at its final address with loaded/request memory
  -> construct and install Program as the last member; warm/capture required graphs
  -> release directory/name index and make the complete TargetInstance active
```

An active target is complete: its loaded product, request memory, and Program already exist. The
Program is never constructed as an external local and later assembled around owners it already
references; its `LoadedModel`, request memory, target instance, and device context are at their
final addresses first. Ordinary RAII owns temporary construction state.

### 6.2 `artifact::Binder`

`artifact::Binder` is a cold-path checked view over the parsed directory. It provides lookup by
canonical name, exact kind/shape/storage validation, checked view construction, and consumption
tracking. It does not interpret layer schedules.

The target package enumerates its expected roles, consumes each named entry, and rejects every
missing or unconsumed object. Array order is never a role key. The binder and target together enforce
the container requirements; the engine does not maintain a second compiled offset or order table.

The binder exists only during construction. Artifact strings do not select behavior after loading.

### 6.3 `LoadPlan` and materialization

`LoadPlan` is a target-generated, move-only construction object. Its private implementation owns two
parts:

```cpp
struct TargetPrivateArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};
```

The facade exposes only `LoadPlan::materialization()`. `artifact::MaterializationPlan` is the value
contract the generic materializer executes: object count, device-arena capacity, device object
handles with final offsets/byte lengths/alignment, and host-retained resource handles. `BindingPlan`
is target-private and maps the consumed object handles to typed roles used after materialization.

Neither part contains raw pointers/references into `artifact::Binder`, a JSON DOM/string, or a
temporary name index. Stable numeric object handles may refer to the reader while construction is in
progress, but all such handles are consumed before the reader is released.

It is not stored in the artifact, not serialized, and not retained as a runtime execution recipe.

The generic materializer consumes `artifact::MaterializationPlan` and returns
`artifact::MaterializedArtifact`, which owns the final device arena and retained raw resource bytes.
It allocates each planned device region at its declared final arena offset, copies tensor payloads,
and retains selected resources on the host. It never changes a persistent tensor representation,
decodes target resources, guesses a layout, chooses a kernel, or synthesizes a missing object. File
offsets are not device offsets. Construction keeps staging and destination storage alive for the
operations that use them; the exact read/copy implementation is not an architectural protocol.

### 6.4 Typed loaded bindings

`LoadedModel` contains named C++ fields or structured arrays that express actual model roles. Hot
execution reaches weights by these fixed bindings, not by strings or directory indexes. Repeated
layers may be represented with fixed-size arrays or package-owned generated structures, but their
element types remain semantic—for example, a GDN layer binding need not pretend to be the same type
as a full-attention layer binding.

`construct_loaded_model` directly constructs the final heap object, moves the ownership carried by
`MaterializedArtifact` into it, and only then forms typed views. The resulting `LoadedProduct`
transitively owns every byte and parsed resource reachable from its bindings. A view may point into
a move-stable allocation owned by the model or into one of the model's final subobjects; it may not
point into a source owner or descriptor array that was later moved. No published binding may refer
to `LoadPlan`, `artifact::Binder`, JSON, the name index, the file mapping/descriptor, or load-only
staging storage.

One stored object may supply multiple checked model views or tied roles. Such aliasing is created by
compiled target code after validation, never by overlapping artifact spans.

### 6.5 Persistent resources

Tokenizer data and other required resources become immutable, owned host objects during
construction. They remain owned by `LoadedProduct`; callers receive references or frontend
services, not a one-shot ownership transfer. Runtime inference does not reopen the artifact or
source model.

## 7. Memory planning and lifetime classes

### 7.1 One definition for size and binding

Every target uses a checked layout builder to define each project-owned backing region in one pass:

```cpp
LayoutBuilder layout;
auto kv       = layout.add("text-kv", text_kv_bytes, text_kv_alignment);
auto recurrent = layout.add("recurrent", recurrent_bytes, recurrent_alignment);
auto io       = layout.add("stable-io", stable_io_bytes, stable_io_alignment);
auto total    = layout.finish();
```

The returned handles are later bound against the allocated base. The same layout definition drives
both total bytes and offsets. A target must not maintain one approximate `default_*_bytes()` formula
and a separate allocation order that can drift.

All element counts, products, roundups, and sums are checked before allocation. Common code supplies
the arithmetic and arena mechanism; the target supplies every model-dependent dimension and
lifetime decision.

CUDA-driver graph-executable allocations are not fake entries in this layout because NInfer does
not control their offsets. `SequencePlan` reserves an explicit target/toolchain allowance, then
Program construction captures graphs and verifies actual free/used memory before target activation.
A capture that exceeds the planned capacity fails target construction; it does not silently shrink
context after activation.

### 7.2 Lifetime classes

NInfer distinguishes five lifetimes:

| Lifetime | Owner | Examples |
|---|---|---|
| loaded immutable | `LoadedProduct` | weights, tokenizer/resource tables, constant descriptors |
| sequence persistent | `Program` | KV, GDN/recurrent state, logical ledger, prefix checkpoints |
| graph stable | `Program` | graph inputs/outputs, scalar controls, host mirrors, captured workspaces |
| request-active values | `Program` storage | sampling RNG/counters and active-request controls, reset at begin |
| request transient | `RequestMemory` region | media staging and uncaptured prefill scratch |

Memory may be physically coalesced only when lifetime, alignment, destruction, and address-stability
requirements remain correct. The owner does not change merely because two regions share an arena.

### 7.3 Sequence planning

`plan_sequence` validates a requested context against exact target limits and actual remaining device
memory, then returns the one complete plan used by `create_program`. Common code must not calculate
KV bytes, GDN slots, MTP snapshots, MoE workspace, Vision workspace, or head buffers.

The plan identifies:

- effective materialized execution capacity `C_exec` and target-native limits;
- persistent and graph-stable layouts;
- uncaptured scratch requirements and allowed growth policy;
- feature dependencies required by the one current target route;
- graph shapes and stable addresses that must exist before capture;
- target-private prefix-checkpoint capacity.

If the one supported target route does not fit, construction fails. It does not silently disable a
native component, reduce quality, offload, or choose a generic fallback.

`SequencePlan` exposes `C_exec` as a bounded common diagnostic while retaining every model-specific
position/context rule privately. It is the maximum number of logical tokens whose target execution
state can be materialized in this Program, not a universal RoPE position or media-axis formula. It
is strictly below `UINT32_MAX` so the possible `C_exec + 1` frontier length is representable.

### 7.4 Request planning before mutation

Every request is planned before resident sequence state is changed:

```cpp
auto plan = program.plan_request(prompt, execution_options);
const auto summary = plan.summary();
request_memory.ensure(summary.transient_bytes, summary.transient_alignment);
auto first = program.begin(
    std::move(prompt), std::move(plan), request_memory.region());
```

`RequestPlan` is target-private, move-only, and self-contained. It may record an intended
prefix-reuse path, required media workspace, effective output capacity, or other precomputed
decisions as owning values and checked indices. It may not borrow from `PreparedPrompt`,
`RequestMemory`, a temporary descriptor, or mutable Program storage; `begin` receives the real
owners directly. In the current serialized one-request route, planning and `begin` run under the
same Engine generation lock, so no common plan-version protocol is required.

Every target-private plan provides a const `summary()` returning the common
`RequestPlanSummary`. The controller copies that bounded value before moving the plan into `begin`;
it never recomputes or substitutes the requested output limit afterward.

Request-memory growth occurs before `begin`. A Vision allocation or large prefill must never move or
invalidate graph-stable addresses. On every successful return or recoverable exception, `begin`
must complete or safely cancel every asynchronous read/use of the moved `PreparedPrompt` and
`TransientRegion` before returning or propagating. Program stores no pointer/view into either input,
and no CUDA Graph captures the transient region. The prompt may be destroyed and the region may be
reset/reused immediately after either outcome. A device failure that prevents safe quiescence
follows Section 12.2 instead of returning a recoverable exception. Output decoder preview scratch lives in
its separate owning CPU session.

### 7.5 Stable model-execution storage

After a request begins, ordinary decode and MTP model execution perform no filesystem access,
artifact-name lookup, device allocation, or graph-address change. Returned token storage is a fixed
Program-owned host buffer, and model-private workspaces are planned or bounded before execution. The
round protocol has no explicit per-round transaction PIMPL or candidate-lattice allocation. This is
not a blanket claim that request-local output decoding and publication perform no string/vector
allocation.

## 8. Checkpoint frontend and prepared prompts

### 8.1 Frontend boundary

The frontend belongs to the loaded exact checkpoint. It owns:

- tokenizer and detokenizer behavior;
- chat template and generation-prompt construction;
- checkpoint-native default stop tokens/strings;
- checkpoint-specific media validation and preprocessing over owning media;
- placeholder expansion and modality/token alignment;
- checkpoint-native position construction;
- assistant-content or other prefix-checkpoint hints;
- validation of tokenizer-addressable input IDs;
- creation of a request-local output preview/decoder session.

Protocol schemas and network fetching remain outside the target. The frontend receives owning
product input/media and produces checkpoint-native data.

Each target frontend supplies the preparation entry points required by that checkpoint and returns
its `T::PreparedPrompt`. Raw frontend argument types may differ with native capabilities and are
specified by the target/product input document, not forced into one core feature bag. The stable
core postcondition is the owning prepared-prompt contract below. Every `T::Frontend` also provides:

```cpp
OutputSession make_output_session(
    const T::PreparedPrompt&,
    const StopPolicy& stop,
    const OutputOptions& output) const;
```

The returned session is owning and request-local.

### 8.2 Concrete prompt types remain private

There is no universal `ProcessedInput` struct. A Qwen3.6 prepared prompt may contain data such as:

```cpp
struct QwenPreparedPrompt {
    std::vector<std::int32_t> token_ids;
    QwenPositionStorage positions;
    std::vector<QwenVisionItem> vision_items;
    OwnedPatchStorage patches;
    std::optional<std::uint32_t> assistant_content_boundary;
    PromptIdentity identity;
};
```

These members are illustrative target-private types, not a common ABI. Another checkpoint can have
different modalities, positions, prompt identity, or no prefix hint at all.

Common code sees a move-only opaque envelope conceptually carrying:

- a small common `PromptSummary`, including prompt token count;
- one alternative of the closed variant of opaque target-typed prepared values.

The closed prepared-value variant's alternative is the target tag; there is no second integer,
load-instance identity, or string dispatch field that can drift from it. A prepared value may be
consumed by another Engine instance of the same exact target package because it owns its complete
checkpoint-native input. A different target alternative is rejected by one
`get_if<T::PreparedPrompt>` before `plan_request` and sequence mutation. The receiving Program still
validates the token domain, prompt geometry, metadata, and its configured capacity.

### 8.3 Ownership and concurrency of preparation

A prepared prompt owns all CPU data required by `begin`. It contains no spans or references into a
request object, no device pointers, no serving lock, and no transport callback. It is immutable after
construction except for being moved into `begin`.

The immutable frontend may prepare a prompt outside the GPU execution critical section. Its shared
state is thread-safe and immutable; mutable tokenizer/decoder state is request-local. This allows
tokenization and media preprocessing to overlap other CPU work without claiming concurrent model
execution. `begin` consumes the prompt; it may release large media storage only after all
asynchronous reads from that storage are complete or safely cancelled. The same guarantee holds
when `begin` exits by a recoverable exception.

### 8.4 Output preview and commit

The frontend creates a request-local `OutputSession` containing the resolved frontend-default plus
caller stop policy and decoder state. The session owns all state it needs after the prepared prompt
is moved into `begin`; it never borrows prompt storage.

The session uses a compact API equivalent to:

```cpp
class OutputSession {
public:
    OutputDecision preview(
        std::span<const TokenId> tokens,
        std::uint32_t budget_remaining,
        FinishReason limit_reason);
    OutputDecision preview_terminal(FinishReason reason);
    PublishedOutput commit_preview() noexcept;
};
```

`preview` may inspect UTF-8 boundaries, tokenizer state, stop tokens/strings, and the current budget,
but it does not mutate committed decoder state and does not invoke a user callback. `OutputSession`
owns reusable request-local scratch for one prospective next `DecoderState` and the selected owning
output deltas. It scans the returned token batch and retains only the one real outcome rather than
constructing a lattice of Continue, budget, stop-token, and stop-string candidates.

`OutputDecision` contains only the exact accepted logical-token count and `FinishReason`; `None`
means Continue. After Program resolves the same count, `commit_preview()` swaps the preview state
into the committed decoder and transfers the selected deltas. It is `noexcept`.
`preview_terminal` prepares a zero-token terminal flush for cancellation at a controller boundary.
A preview must be committed before another preview is made.

The decoder therefore advances by exactly the accepted logical-token prefix even when published
text differs. This is required because one MTP round can return several tokens and a stop string may
end inside that batch.

The preview maps a textual stop back to an exact token prefix. If the stop begins or ends inside one
token's decoded byte contribution, that token is still included in the model-committed prefix while
the resolved stop policy may withhold the stop bytes from published text. A stop spanning token
boundaries retains enough uncommitted decoder tail to identify the same exact token count. Byte
trimming and token-state commit are therefore related but not falsely treated as the same index.

On Continue, the selected preview publishes only bytes that cannot still become the prefix of a
cross-round stop string and retains the ambiguous tail. On Finish, it flushes that tail except for
bytes intentionally withheld by the selected stop policy. The same accepted token count can
therefore produce different safe publication at Continue and Finish; `FinishReason::None` versus a
terminal reason records that distinction.

`PublishedOutput` is an ordered owning sequence of channel-tagged byte deltas, at minimum
`Reasoning` and `Content`. Checkpoint-native thinking markers, channel transitions, incremental UTF-8,
and stop matching are handled while previewing. A resolved stop string declares its matching channel
domain; ordinary caller stop strings default to Content unless the product surface explicitly asks
for another domain. Serving-specific tool parsing remains outside this decoder contract.

Publication consumes that ownership through a contract equivalent to
`publish(OutputSink&, PublishedOutput&&)`. The sink may consume bytes synchronously or move/copy them
into storage it owns, but it may never retain a borrowed view into the argument. Normal return is
the controller's publication completion point: every delta has been accepted into sink-owned
lifetime. A synchronous exception means acceptance failed, possibly after an externally visible
prefix; the batch is not retried and the generation call fails. `GenerationGuard` then prevents the
Program sequence from being reused. An asynchronous adapter may define successful
ownership-transferring enqueue as acceptance, in which case a later socket/client failure belongs to
that adapter and is not retroactively reported as this generation's publication failure.

The round protocol has no separate staged-candidate object, candidate ID/lattice, or decoder commit
plan, and therefore no explicit per-round transaction PIMPL allocation. This is not a blanket claim
that token decoding, strings, vectors, or publication are allocation-free. The observable
requirements—preview without committed-state mutation or publication, commit the same exact prefix
as Program, and preserve byte-correct streaming—are mandatory.

## 9. Common request values and summaries

### 9.1 Pure host-side options

Common request options are plain owning values. They do not contain device pointers, tensors, model
positions, cache slots, or kernel-launch fields.

```cpp
struct SamplingParameters {
    float temperature = 0.0f;
    std::int32_t top_k = 0;
    float top_p = 1.0f;
    float min_p = 0.0f;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    std::uint64_t seed = 0;
};

struct ExecutionOptions {
    SamplingParameters sampling;
    std::uint32_t requested_output_tokens = 0; // includes the first post-prefill token
    bool allow_prefix_reuse = true;
};

struct OutputOptions {
    bool raw = false;
    bool preserve_special_tokens = false;
};

struct RequestOptions {
    ExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
};

enum class FinishReason : std::uint8_t {
    None,
    OutputLimit,
    ContextCapacity,
    StopToken,
    StopString,
    Cancelled,
};

struct OutputDecision {
    std::uint32_t accepted_tokens;
    FinishReason finish_reason; // None means Continue

    bool finished() const noexcept { return finish_reason != FinishReason::None; }
};

struct GeneratedRound {
    std::span<const TokenId> tokens;
};
```

Only `ExecutionOptions` reaches `Program::plan_request`. The target validates the supported sampling
domain and translates those values into its own stable device controls. Op-facing sampling
structures remain internal to the target/Op boundary. Stop strings, output channels, and caller
publication policy never enter the model program.

`execution.requested_output_tokens == 0` performs no model execution. It is not represented by a
prefill that produces and then hides a token.

The frontend merges checkpoint defaults with `stop` once when it creates `OutputSession`.
The resulting `StopPolicy` is immutable for the request and specifies deterministic declaration
order, string channel domains, and whether matched markers are published. Program token-domain
facts are used by the frontend/controller to validate every resolved stop token before `begin`;
the policy itself is not passed to the Program.

### 9.2 Plans and observable summaries

The detailed `RequestPlan` remains target-private. Common code receives a bounded summary:

```cpp
struct RequestPlanSummary {
    std::uint32_t prompt_tokens;
    std::uint32_t reusable_prompt_tokens;
    std::uint32_t requested_output_tokens;
    std::uint32_t effective_output_tokens;
    FinishReason effective_limit_reason; // OutputLimit or ContextCapacity
    std::size_t transient_bytes;
    std::size_t transient_alignment;
};

struct BeginSummary {
    std::uint32_t prompt_tokens;
    std::uint32_t reused_prompt_tokens;
};

struct GenerationSummary {
    std::optional<BeginSummary> begin; // absent when no begin occurred
    FinishReason finish_reason;
};

struct BeginResult {
    BeginSummary summary;
    GeneratedRound round;
};
```

`begin` returns `BeginResult`, containing `BeginSummary` and a one-token `GeneratedRound`. The first
output is therefore previewed and resolved through the same contract as every later batch; it is not
accepted merely because prefill succeeded. `GeneratedRound` contains no owner, callback, generation
counter, destructor action, or allocation; Program privately remains Pending until resolution.

Summary invariants are `reusable_prompt_tokens <= prompt_tokens` and
`effective_output_tokens <= requested_output_tokens`. For a nonzero request,
`effective_limit_reason` is `OutputLimit` exactly when the effective count equals the requested
count; it is `ContextCapacity` when the target reduces that count, including reduction to zero.
Equality deliberately reports `OutputLimit` even if the request and target capacity happen to end
at the same token. Unsupported sampling or input is an error, not another silent reduction reason.
`transient_alignment` is a positive power of two supported by `RequestMemory`; zero transient bytes
use alignment one. `BeginSummary` reports the actual successful path and must agree with the plan or
be more conservative about reuse.

The target computes effective capacity. Common code does not assume a formula such as
`max_context - prompt_tokens + 1`: different recurrent, multimodal, speculative, or position
semantics may impose different boundaries.

The controller constructs its request-local generation budget from `effective_output_tokens` and
`effective_limit_reason`, not from the caller's larger request. Committing the first round consumes
one unit. Every later `RoundBudget` is derived from the remaining effective count, and reaching zero
is a terminal decision carrying that recorded reason and resolving the current round before return.
If a nonzero request has zero target-approved capacity, the controller does not call `begin` and
returns the same `ContextCapacity` reason.

Preview selection proves `accepted_tokens <= remaining` before Program mutation.
`GenerationBudget::commit` and `round_budget` are then `noexcept`; a violated bound is an internal
fatal invariant, not a recoverable exception after model/decoder commit.

The output matrix row count, tokenizer-addressable ID count, valid sampled-token domain, and stop-ID
domain are distinct checkpoint facts. A common `vocab_size` field must not collapse them.

## 10. `Program` contract

### 10.1 Required coarse operations

For a package `T`, its program provides the following target-typed operations to the templated
controller:

```cpp
class T::Program {
public:
    ~Program() noexcept;
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&) = delete;
    Program& operator=(Program&&) = delete;

    RequestPlan plan_request(
        const PreparedPrompt&,
        const ExecutionOptions&) const;

    BeginResult begin(
        PreparedPrompt&&,
        RequestPlan&&,
        TransientRegion);

    GeneratedRound decode_round(RoundBudget);

    void resolve_pending(std::uint32_t accepted_tokens, bool terminal);
    void finish_active();
    void abort_request() noexcept;

    std::uint32_t materialized_tokens() const noexcept; // narrow diagnostic query
};
```

The lifecycle state and `E`/`S` counters are target-private. Common code receives only the token
view and calls the coarse resolution operations; it does not inspect or mutate target state. The
narrow `materialized_tokens()` query exists for target diagnostics rather than controller policy.
Section 11 defines the required Active and Resident relationships.

`T::RequestPlan` is required to expose `const RequestPlanSummary& summary() const noexcept`. No
other part of its representation is common.

The exact target may have any number of private methods. The common controller cannot call them.
In particular, `target_verify`, `mtp_propose`, `run_vision`, cache rewind, recurrent-slot selection,
and graph launch are not common operations.

### 10.2 `begin`

`begin` is one atomic target-level operation from common code's perspective. Depending on the plan,
it may:

- fully reset sequence state;
- append to an exact reusable prefix;
- restore a target-private prefix checkpoint and continue;
- run native Vision and merger work;
- scatter composed embeddings and checkpoint-native positions;
- prefill Text state in chunks and retain every target hidden column required by MTP;
- install request sampling controls;
- sample the first target token from target logits;
- complete shifted MTP alignment/prefill using that sampled token and establish the first proposal.

On success, it returns `BeginResult` containing exactly one target-licensed token and privately
enters Pending. All prompt input reads are complete before control returns. The prepared prompt,
prefill state, and sampled token are not yet a reusable logical sequence: the controller must preview
the token and call `resolve_pending` with the same Continue/Finish rules as every later round.

Continuing with the begin token establishes `Active` with `E = P` and `S = P + 1`. Terminal
resolution establishes a Resident result when the target can represent the exact prefix, otherwise
the target may make it non-reusable. `abort_request()` invalidates unresolved provisional work. There
is no first-token exception to exact-prefix resolution.

On a host-side validation or logical execution failure, `begin` publishes no new resident identity
and leaves the program either at its unchanged pre-begin resident state when the target can prove no
mutation, or `Invalid`. It must never claim a partially updated prefix as reusable.

### 10.3 `RoundBudget`

Before each round, the controller supplies a pure bound:

```cpp
struct RoundBudget {
    std::uint32_t generated_tokens_remaining;
};
```

The target must not return more tokens than the budget. If its speculative graph can yield a wider
batch than the remaining output budget or safe context tail, it uses a target-private narrower path,
normally ordinary decode. A program never relies on the controller silently dropping a suffix.

### 10.4 Stable token return storage

`GeneratedRound::tokens` is a synchronous span over Program-owned fixed-capacity host memory. Its
contents remain valid until the current controller iteration resolves that round. It must not be
stored past `resolve_pending` or `abort_request`. A target may use pinned host memory and one
synchronized count/token transfer per round. It does not allocate and return `std::vector<int>` on
every decode step.

Only target-licensed output tokens appear in the span. Unverified proposals, rejected drafts, padded
slots, and model-internal row indexes never escape as candidate output.

### 10.5 Operation and destruction postconditions

The common controller relies on these exact failure guarantees:

- `plan_request` is read-only; success or exception leaves Program lifecycle and sequence unchanged;
- `begin` follows Section 10.2, never exposes a partial new identity, and before propagating any
  recoverable exception it quiesces or safely cancels all asynchronous access to its moved
  `PreparedPrompt` and `TransientRegion` inputs;
- `decode_round` enters target-private Pending only when all licensed output is ready; any recoverable
  exception after mutation but before returning a round leaves Program `Invalid` and leaves no
  untracked asynchronous access;
- `resolve_pending` accepts `1 <= accepted_tokens <= produced`; Continue additionally requires the
  complete returned batch and establishes Active, while terminal resolution establishes Resident or
  a non-reusable state for that exact prefix;
- `finish_active` changes Active to Resident; a precondition violation changes no state;
- `abort_request` makes any Active, Pending, or Resident sequence non-reusable and is `noexcept`;
- process-fatal device faults follow Section 12.2 instead of pretending to satisfy recovery.

`abort_request()` is a host-reachability operation only. It makes device state unreachable for reuse;
it does not launch, synchronize, allocate, or claim rollback. A later begin performs the target's
ordered full device reset. `Program::~Program()` is `noexcept`. Engine teardown quiesces its target
stream while DeviceContext is still alive, then destroys Program, RequestMemory, LoadedProduct, and
finally DeviceContext in the order established in Section 5.3.

## 11. Sequence state and frontier invariants

### 11.1 Counters

At every resolved state, define:

- `E`: number of logical sequence tokens whose target-model execution state is committed;
- `S`: number of tokens in the committed logical sequence;
- `P`: prompt token count for the current logical request where relevant.

For the Program's planned materialized capacity `C_exec`:

```text
0 <= E <= C_exec
E <= S <= E + 1
S <= C_exec + 1
```

`decode_round` may launch only when `E < C_exec`; a returned round of `n` tokens must be resolvable
with `E + n <= C_exec`. A prompt may occupy `P == C_exec` and still yield one unprocessed frontier
token when the target's own position rules allow it, but no subsequent decode may launch. The target
still computes effective output capacity; common code does not derive it from these bounds.

`E` is the materialized/processed target prefix, not the number of tokens accepted for user-visible
output. The distinction is normal rather than exceptional: a generation step processes the old
frontier and creates a new frontier.

In `Active` ready state:

```text
S = E + 1
```

The last logical token is the licensed next token at the execution frontier. It is visible to the
controller but is consumed by the next target step.

After a fresh begin round whose first token is committed for Continue:

```text
E = P
S = P + 1
```

After an ordinary decode consumes the old frontier token and returns a new one:

```text
E' = E + 1
S' = S + 1 = E' + 1
```

After a fully committed MTP round returns `n >= 1` licensed tokens:

```text
E' = E + n
S' = S + n = E' + 1
```

Such a round materializes the old frontier plus the first `n - 1` returned tokens and leaves the
last returned token as the new frontier. Thus accepting all `n` logical outputs still does not mean
all `n` have already been written as target KV/recurrent positions.

The target may execute provisional work beyond these values while privately Pending, but that state
is neither reusable nor exposed to common code as committed progress.

### 11.2 Resident state after finish

After a request finishes, the program may keep a reusable `Resident` prefix with:

```text
E <= S <= E + 1
```

`S == E + 1` means the final accepted logical token remains unprocessed but is part of the exact
sequence. `S == E` means target-private finalization processed/canonicalized through the last
accepted logical token. Both are legal; prefix reuse must know which form it holds.

For a decode round starting from Active `(E, S = E + 1)` and terminally accepting exactly `m`
returned tokens:

```text
S_f = S + m = E + m + 1

frontier Resident:   E_f = E + m       and S_f = E_f + 1
canonical Resident:  E_f = E + m + 1   and S_f = E_f
Invalid:              no logical identity; reported E_f = S_f = 0
```

For a begin round accepting its one token, the corresponding Resident forms are `(E_f, S_f) =
(P, P + 1)` or `(P + 1, P + 1)`. A target may leave physical candidate bytes after either form, but
no cursor, checkpoint, proposal, or identity may reach any suffix beyond `S_f`.

No cache cursor, recurrent slot, MTP cache, host token mirror, position state, or prefix checkpoint
beyond the committed logical sequence may be reachable after finish.

### 11.3 Serialized request planning

The current Engine holds one generation lock across `plan_request`, `begin`, all rounds, and finish.
There is therefore no common sequence-generation counter, plan token, or round-handle lifetime
protocol. Prefix eligibility is re-established by the target from its current resident ledger during
that serialized operation. Future continuous batching must define its own sequence-slot identity and
lifetime rules rather than preserving an unused single-request generation counter.

### 11.4 Prefix reuse

Prefix reuse is a target-private optimization over exact prepared-prompt identity. Token IDs alone
are insufficient when media content, position construction, templates, or other composed embeddings
can differ.

A target may reuse only:

- an execution frontier it knows how to continue exactly; or
- an explicit target-owned checkpoint whose full KV/recurrent/MTP/position state is consistent.

The target must atomically derive any requested boundary from the incoming prepared prompt and the
resident ledger. Common code does not call a separate mutable `set_boundary()` before prefill.

If matching, restore, or continuation cannot be proven, the target performs a full reset. A failed
reuse attempt cannot fall through with partially rewound state.

## 12. Program state machine

### 12.1 States

The required target-private lifecycle is:

```text
Empty ─────────────── begin success ──────────────> Pending
Resident ──────────── begin success ──────────────> Pending
Invalid ── full-reset begin success ──────────────> Pending

Active ────────────── decode_round ───────────────> Pending
Pending ── resolve_pending(full, Continue) ───────> Active
Pending ── resolve_pending(prefix, terminal) ─────> Resident or Invalid

Active ────────────── finish_active ──────────────> Resident
Active/Pending/Resident ── abort_request ─────────> Invalid
```

Only one pending round may exist. Pending is a target-private lifecycle state, not a common handle.
`decode_round` is illegal outside `Active`; `begin` is illegal while active or pending;
`finish_active` is illegal while a round is pending.

`Invalid` means host logic must not select the current device state for reuse. A later `begin` may
recover only through a target-defined full reset. It does not imply that arbitrary CUDA failures are
recoverable.

### 12.2 CUDA fault policy

The current process-fatal CUDA policy is compatible with this state machine: a fatal device error
terminates instead of pretending the program can be reset. `Invalid` covers host cancellation,
unresolved provisional work, and logical failures for which device state still exists but is not
trusted.

If NInfer later adopts recoverable device faults, it must add a distinct `Faulted`/reload contract
that proves stream and allocation validity. It must not overload `Invalid` with unsupported CUDA
recovery.

## 13. Generated-token round resolution

### 13.1 `GeneratedRound`

`begin` returns a one-token `GeneratedRound` inside `BeginResult`; `decode_round` returns the same
small value directly:

```cpp
struct GeneratedRound {
    std::span<const TokenId> tokens;
};
```

It is a synchronous view into Program-owned stable host storage. It never escapes the current
controller iteration and has no owner pointer, callback table, generation counter, destructor
action, or dynamic allocation. Program itself records that exactly one round is Pending.

The controller resolves that state through:

```cpp
void Program::resolve_pending(std::uint32_t accepted_tokens, bool terminal);
```

Exactly two successful outcomes are permitted:

- Continue accepts the complete returned batch, normalizes the model execution frontier according
  to Section 11, and establishes Active for another round;
- terminal resolution adopts exactly `1 <= accepted_tokens <= tokens.size()` logical tokens and
  finishes at that prefix. The target establishes Resident when it can represent that exact prefix,
  otherwise it makes the sequence non-reusable.

An unresolved or failed request is handled by the request-level `GenerationGuard`, which calls
`abort_request()`; there is no independent round-handle destructor protocol.

### 13.2 Why arbitrary continuation is forbidden

The interface intentionally forbids continuing from a strict prefix of a returned batch and has no
universal `rollback()`.

A speculative target may prepare the next proposal state only for its complete accepted outcome.
An intermediate returned token can be target-licensed for output while lacking the MTP, recurrent,
or cache state needed to continue as if the suffix had never happened. Similarly, pre-round backup
state may not exist for every mutable component. Claiming arbitrary continuation would either be
false or force expensive universal snapshots into every target.

The only partial-prefix action is terminal. The target can canonicalize exactly the accepted logical
prefix, select a verified per-token recurrent snapshot, rewind a position-addressed cache, or retain
an allowed final frontier token without promising continuation through an unsupported intermediate
proposal state. If it cannot construct and prove that exact resident state, it still accepts the
licensed output prefix for this result but makes the sequence non-reusable.

### 13.3 Provisional state

While Program is Pending, target-private state may include:

- verification KV entries and recurrent snapshots;
- accepted-count and target-token buffers;
- draft-model KV and autoregressive hidden state;
- sampler probability/RNG/occurrence deltas for the candidate logical outputs;
- a next-round proposal prepared for the complete outcome;
- graph control scalars and provisional cursor values.

None of it is common API. Full Continue normalizes every component to the complete returned logical
batch, leaving its last token as the new unprocessed frontier. Terminal resolution either normalizes
the exact accepted prefix and makes all later state unreachable, or invalidates reuse.
`abort_request()` invalidates the entire resident identity without claiming rollback.

Sampler storage is physically Program-resident but logically active-request state. Full Continue
commits exactly the returned logical tokens. Finish retires sampler state; the next begin always
initializes it from the new `ExecutionOptions`. A target may therefore avoid repairing suffix-only
penalty counters on terminal partial finish, but it may never continue with or expose those counters
as active/resident semantics.

### 13.4 Stop and cancellation resolution

For a returned batch, the controller:

1. asks `OutputSession::preview` to decode into scratch and select one `OutputDecision` without
   mutating committed decoder state;
2. calls `Program::resolve_pending(m, terminal)` with the same exact accepted token count `m`;
3. commits the decoder preview through `commit_preview()`;
4. charges `m` against the generated-token budget;
5. only then publishes the owning output deltas.

`m` counts model-logical generated tokens and is always at least one for a returned round. It
includes a terminating EOS/stop token and the token containing a stop-string byte boundary, even if
that token contributes zero published bytes. Published byte count and accepted token count are not
interchangeable.

Preview selection is deterministic and scans token boundaries in order. At one boundary, a matching
stop string wins over a stop token and the budget limit; among stop strings, earliest decoded byte
cut wins and equal cuts use resolved `StopPolicy` declaration order. Stop token wins over the budget
limit at the same boundary. The budget limit is considered only for the complete returned batch and
uses the `effective_limit_reason` fixed by RequestPlan; `OutputLimit` versus `ContextCapacity` is not
reconstructed from counters.

Cancellation is sampled at controller boundaries. Before `begin`, cancellation returns with no model
work. Between committed rounds, the controller calls `preview_terminal(Cancelled)`,
`finish_active()`, commits and publishes the decoder flush, and retains the coherent Resident prefix.
If cancellation becomes visible after `begin` or `decode_round` produced provisional work, the
controller previews only a terminal flush, calls `abort_request()`, commits and publishes that flush,
and accepts none of the new round tokens. A signal arriving after resolution is observed at the next
boundary. This is the single current policy, not a per-target choice.

If preview or `publish` throws, the model state is not rolled back. The controller reports the
failure, and the still-armed `GenerationGuard` calls `abort_request()` so the generated sequence is
not offered for prefix reuse. Bytes already accepted by a partially failing sink cannot be
retracted; the batch is not retried.

## 14. Common generation loop

The target-templated controller follows this shape:

```cpp
template<class Target>
GenerationSummary run_one(
    TargetInstance<Target>& target,
    typename Target::PreparedPrompt&& prompt,
    RequestOptions options,
    const CancellationView& cancellation,
    OutputSink& sink) {

    if (cancellation.requested()) {
        return cancelled_without_execution();
    }
    if (options.execution.requested_output_tokens == 0) {
        return empty_summary(prompt);
    }

    auto output = target.loaded->frontend.make_output_session(
        prompt, options.stop, options.output);
    auto plan = target.program->plan_request(prompt, options.execution);
    const auto plan_summary = plan.summary();
    if (plan_summary.effective_output_tokens == 0) {
        return no_capacity_summary(plan_summary);
    }
    target.request_memory.ensure(
        plan_summary.transient_bytes, plan_summary.transient_alignment);

    GenerationBudget budget(
        plan_summary.effective_output_tokens,
        plan_summary.effective_limit_reason);

    GenerationGuard guard(*target.program); // initially disarmed
    std::optional<BeginSummary> begin_summary;

    const auto resolve_and_publish = [&](GeneratedRound round)
        -> std::optional<GenerationSummary> {
        if (cancellation.requested()) {
            output.preview_terminal(FinishReason::Cancelled);
            target.program->abort_request();
            auto deltas = output.commit_preview();
            publish(sink, std::move(deltas));
            return finish_summary(*begin_summary, FinishReason::Cancelled);
        }

        const auto decision = output.preview(
            round.tokens, budget.remaining(), budget.limit_reason());
        if (!decision.finished()) {
            require(decision.accepted_tokens == round.tokens.size());
        }
        target.program->resolve_pending(
            decision.accepted_tokens, decision.finished());
        auto deltas = output.commit_preview();
        budget.commit(decision.accepted_tokens);
        publish(sink, std::move(deltas));

        if (decision.finished()) {
            return finish_summary(*begin_summary, decision.finish_reason);
        }
        return std::nullopt;
    };

    if (cancellation.requested()) {
        return cancelled_without_execution();
    }
    auto first = target.program->begin(
        std::move(prompt), std::move(plan), target.request_memory.region());
    begin_summary = first.summary;
    guard.arm();
    if (auto done = resolve_and_publish(first.round)) {
        guard.complete();
        return *done;
    }

    while (budget.remaining() != 0) {
        if (cancellation.requested()) {
            output.preview_terminal(FinishReason::Cancelled);
            target.program->finish_active();
            auto deltas = output.commit_preview();
            publish(sink, std::move(deltas));
            guard.complete();
            return finish_summary(*begin_summary, FinishReason::Cancelled);
        }

        auto round = target.program->decode_round(budget.round_budget());
        if (auto done = resolve_and_publish(round)) {
            guard.complete();
            return *done;
        }
    }

    unreachable_budget_state();
}
```

This pseudocode fixes ownership and ordering, not exact public names. The begin round and every later
round use the same resolver. Their only structural difference is the begin summary and the work the
target performed before returning the token view.

`GenerationGuard` is an internal scope guard, not a public request object. It is constructed
disarmed before `begin` so begin failure preserves that operation's own postcondition, then armed
only after a first round exists. Until `complete()`, its destructor calls `abort_request()` and makes
any Pending, Active, or just-finished Resident sequence non-reusable. The action is host-only and
`noexcept`; the next `begin` performs the target's real full reset. The guard remains armed through
decoder commit and publication, so output that failed during preview or publication is never offered
for prefix reuse.

The loop has no branch on Text versus Vision, dense versus MoE, attention versus GDN, ordinary
versus MTP, or exact checkpoint identity. Those branches occur within the selected package.

## 15. CUDA Graph and hot-path rules

### 15.1 Address stability

`Program` is heap-owned through `std::unique_ptr` and is non-movable. Every device and pinned-host
address referenced by a CUDA Graph remains stable for the program lifetime. Updating request values
means copying into stable control buffers, not replacing pointers captured by a graph.

Graph ownership, warmup, capture eligibility, recapture, and tail fallback are target-private.
Common code only invokes `begin` and `decode_round`.

### 15.2 Graph completeness

A target may use separate graphs for ordinary decode, MTP round, verification shapes, or other exact
paths. It must ensure that graph outputs and all mutable state are normalized by the round-resolution
contract. A graph launch is an implementation detail, not proof that a logical transaction is
committed.

### 15.3 Host/device synchronization

The target minimizes synchronization but never exposes an unread or partially copied result. A
round should normally copy its count and licensed tokens through stable memory with one completion
boundary. Partial finalization may require a rare schedule-selected scalar Op repair or
synchronization. The target owns the affected view and call time; the scalar Op owns the explicit
value transition. That cost is preferable to an incorrect reusable prefix.

### 15.4 Hot-path exclusions

Prefill/decode/MTP hot paths contain no:

- JSON parse or artifact string lookup;
- filesystem operation;
- runtime persistent repack or quantization;
- target registry lookup;
- per-layer polymorphic dispatch;
- per-round device allocation or explicit transaction-PIMPL/candidate-lattice allocation;
- hidden synchronization caused by movable or temporary graph buffers;
- serving callback from inside model execution.

Request-local output decoding and publication may still grow ordinary strings/vectors; this rule is
about stable model execution and the removed per-round transaction objects, not a universal
host-allocation guarantee.

## 16. Concrete fit: Qwen3.6-27B

This section records how the registered 27B package realizes the contract. Its concrete types remain
target-private and do not become common engine API.

### 16.1 Loaded product

The 27B target's `LoadedModel` owns typed bindings for:

- Text embedding, 64 hybrid decoder layers, final norm, full head, and any selected optimized head;
- one-layer MTP weights and its shared Text embedding/head semantics;
- Vision transformer, patch merger, and required frontend resources;
- exact format/layout descriptors already validated during binding.

Tokenizer, template, and other checkpoint resources remain owned by the loaded target product for
the same lifetime as the bound artifact.

### 16.2 Program state

The 27B `Program` owns:

- 16 full-attention Text KV caches and the MTP KV cache;
- 48 GDN convolution histories, FP32 recurrent matrices, and required snapshot slots;
- logical token identity, exact prefix-reuse ledger, and assistant-boundary checkpoint;
- position/rope controls and multimodal continuation state;
- sampling controls, occurrence counters, licensed output buffer, and MTP counters;
- ordinary-decode and MTP-round graphs and every graph-stable tensor;
- prefill, verification, proposal, head, and Vision workspaces with their lifetimes separated.

They are one ownership unit because only the 27B target can state their cross-component invariant.

### 16.3 Fresh and reused begin

For text, `begin` chooses full reset, append continuation, or an exact boundary restore based on the
prepared prompt and current resident ledger. Boundary selection and GDN snapshot scheduling are one target
operation; there is no out-of-band setter whose value can be cleared by a later reset.

For multimodal input, one `begin` covers:

```text
Vision encode
  -> merger output
  -> Text composed-embedding scatter and three-axis positions
  -> Text prefill and required final-normalized hidden columns
  -> first target sample from Text logits
  -> MTP shifted composed-embedding/hidden/position alignment and causal prefill
  -> first MTP proposal
```

Vision scratch can be released only after both Text and MTP consumers no longer read merger-derived
columns. The resident identity includes media/composed-embedding identity, so a text-token match
alone never authorizes multimodal reuse.

The implemented path carries Vision-merger composed columns into shifted MTP inputs instead of
reconstructing image/video placeholder columns through raw token embedding lookup. The native Python
reference implements the same composed-embedding lookahead and remains the independent oracle.

### 16.4 Decode round

Ordinary decode returns one licensed token. An MTP round privately performs target verification,
acceptance, GDN slot selection, MTP proposal preparation, and next-round state construction. It
returns only `[accepted target-licensed drafts..., target correction/bonus]` through the fixed output
buffer.

A batch is returned as one `GeneratedRound` and resolved once through `resolve_pending`, which
prevents the controller from exposing one token at a time while silently retaining an unresolved
speculative suffix.

The current 27B target deliberately makes a terminal strict prefix of an MTP batch non-reusable. It
accepts that logical output prefix for the user-visible result, but calls the target-private invalid
path instead of implementing KV/GDN/MTP repair for an uncommon terminal event. It never promises
that generation can continue from an arbitrary intermediate MTP proposal state.

For the 27B MTP result `[accepted drafts..., correction/bonus]`, let `n = accepted_drafts + 1`.
The required resolved states are:

| Resolution | Materialized `E_f` | Logical `S_f` | 27B state consequence |
|---|---:|---:|---|
| full Continue of `n` | `E + n` | `E + n + 1` | final correction/bonus is the new frontier |
| terminal `m < n` | not exposed | not exposed | `Invalid`; exact output prefix is returned but no sequence identity is reusable |
| terminal `m = n`, Resident | `E + n` | `E + n + 1` | final correction/bonus remains the frontier |

Advertising the Resident form requires agreement among the host KV cursor, device Text
position, decode RoPE position (`rope_pos = text_pos + rope_delta` for this target), GDN initial slot,
logical token mirror, MTP KV/carry/proposal
reachability, graph control scalars, and any still-active sampler state. Repairing only the host KV
cursor and GDN slot is insufficient.

Occurrence-count and RNG buffers are Program-resident storage but request-active values. They are
initialized at every successful begin and retired at Finish. A partial terminal result need not make
them continuable, because partial Continue is forbidden and the next request reinitializes them; it
must not expose their suffix-contaminated values as active state. A full Continue must account for
exactly all `n` accepted logical outputs.

The 27B target has the same split token domain as 35B: 248320 output rows but only 248077
tokenizer-addressable IDs. The implementation applies the valid-sample/decode domain to the first target
sample, ordinary decode, every MTP target correction/bonus, and any optimized draft-head remap.
Validation against 248320 matrix rows alone is not sufficient.

### 16.5 Public product fit

The public API exposes owning input/result values and the model-neutral operations
`Engine::prepare`, `Engine::prepare_tokens`, `Engine::generate`, `Engine::count_tokens`, and summary
queries. Target phase methods, checkpoint memory formulas, caches, and Op-facing sampling state
remain internal. CLI, server, and product benchmarks use this same public route.

## 17. Required fit: Qwen3.6-35B-A3B

The planned 35B-A3B checkpoint is the second concrete design constraint that prevents the common
contract from becoming renamed 27B bookkeeping. This section is a required fit analysis, not a
claim that the target is implemented.

### 17.1 Different loaded structure

The exact checkpoint has 40 Text layers: 30 GDN and 10 full-attention layers, with sparse MoE in
every layer. Each routed-expert bank is rank three, routes among 256 experts, selects eight, and is
combined with a gated shared expert. Its one-layer MTP block also contains full attention and MoE.

Those weight roles, expert descriptors, and router buffers are 35B target types. Routed-expert
mathematics must be expressed as central Ops; their exact shapes may have dedicated kernels beneath
those contracts. The common loader only materializes the validated regions in its load plan; it
does not gain an `experts` field or a generic layer graph.

### 17.2 Different state and workspace

The 35B program owns ten Text KV caches, thirty GDN state sets, one MTP KV cache, MoE routing and
grouping workspaces, and its own graph-stable controls. Workspace peaks and reuse opportunities differ
from dense 27B and are calculated by the 35B package, not a common formula parameterized by layer
counts.

### 17.3 Composed MTP input

The checkpoint reuses the target embedding for MTP, but a multimodal placeholder's shifted MTP
input must use the same composed embedding column produced by Vision merger replacement—not a fresh
raw token-table lookup. The 35B target therefore carries a private alignment value conceptually like:

```cpp
struct MtpAlignedBatch {
    ComposedEmbeddingView next_token_embeddings;
    HiddenView target_hidden;
    QwenPositionView positions;
};
```

This is internal schedule data. Putting token IDs alone into a common MTP API would be incorrect for
the multimodal checkpoint.

For prompt states `h_0..h_n` and positions `p_0..p_n`, Text first samples `x_(n+1)`. MTP then
causally prefills the complete alignment:

```text
composed token inputs = [embed/composed(x_1), ..., embed/composed(x_n),
                         embed/composed(x_(n+1))]
target hidden inputs  = [h_0, ..., h_n]
position inputs       = [p_0, ..., p_n]
```

Tokens shift left; hidden states and positions do not. The final MTP column cannot execute before
the target first token exists. Chunked Text prefill must either retain the complete required hidden
columns or stream an exactly aligned MTP prefix with the one-column carry; running MTP only at the
last prompt column is invalid. When any shifted source column is an image/video placeholder, its
composed merger embedding—not the raw placeholder row—is used.

The prepared-prompt and begin contracts also cover the checkpoint's native image and video paths:
multiple media items, temporal video slices/timestamps, placeholder expansion, three-axis positions,
token types, `rope_delta`, and isolation/alignment across media and text. These remain 35B-private
fields, but image, video, multi-item, and cross-chunk cases are required parity gates rather than one
generic “Vision works” check.

### 17.4 Token domains

The 35B output matrices have 248320 rows while the tokenizer exposes 248077 addressable IDs. The
target validates input IDs, masks or rejects nonsampleable output rows as required, and returns only
valid `TokenId` values. Common code neither allocates penalty state from a guessed tokenizer size nor
treats all head rows as decodable tokens.

### 17.5 Same stable controller contract

Despite different internal layers, both targets can:

- consume a target-typed prepared prompt;
- plan memory and effective output capacity before mutation;
- return one first-token `GeneratedRound` from `begin`;
- return only licensed tokens in a bounded decode round;
- commit the full round, finish at an accepted logical prefix, or invalidate;
- maintain the same `E`/`S` frontier invariant;
- expose target-independent generation summaries.

That is the common interface the 35B implementation must demonstrate. No lower-level phase needs to
be public.

## 18. Execution layers below the target package

Target-private scheduling does not make mathematical computation target-owned. NInfer separates the
target schedule, central Op contracts, and their CUDA implementations by responsibility rather than
by reuse count.

### 18.1 L0 mechanisms

Core and artifact mechanisms include:

- checked arithmetic, tensors/views, device/stream/event ownership;
- immutable and mutable arenas plus `LayoutBuilder`;
- strict container parsing, object binding, and materialization;
- generic direct/packed storage descriptors from registered formats/layouts;
- CUDA Graph lifetime wrappers;
- raw host/device transfers and pinned stable host buffers;
- target-neutral physical KV-cache containers and cursor mechanisms;
- diagnostic counters and timing/NVTX mechanisms.

Program owns each cache/state instance and all advance, rewind, reset, prefix, commit, and rollback
policy. An Op may consume a core container or explicit view without acquiring its lifetime. A
materially different future state representation may use another core type or remain target-owned;
this does not require a universal cache hierarchy.

### 18.2 Central Op layer

An Op is a host-callable, semantically closed tensor or explicit local-state transformation. Given
explicit inputs, weights, state, and semantic parameters, its full outputs and mutations are defined
without model identity or schedule context. Every Op contract and every implementation of that
contract belong to the central Op layer.

One caller, one exact shape, one registered format, one device, fusion, or mutable explicit state do
not make an Op target-private. An exact-shape CUDA kernel is an implementation selected by the Op
wrapper; another target may later use the same contract and implementation without moving code.

An Op does not own model layer order, operand/view selection, prompt chunking, MTP round
orchestration, cache cursor policy, prefix checkpoints, or graph lifetime. Those remain in the
target Program and schedule. The complete definition, contract-comment format, source layering,
and admission procedure are authoritative in [`op-development.md`](op-development.md).

### 18.3 Boundary decision

A target-callable device transformation is admitted as an Op when its complete value/state effect
can be described and verified from explicit arguments. If it still requires a model phase, weight
role, request frontier, commit decision, or instance-lifetime policy, it is schedule composition
rather than an Op. If it is only a tile, partial reduction, codec, launcher, or device helper under a
complete transformation, it remains private implementation material below that Op.

There is no target-facing private Op category and no requirement to prove use by a second target.
This central ownership does not introduce a family base class, model graph, dynamic Op registry, or
plugin interface.

## 19. Repository, source, and build organization

The ownership contract above must be visible in the source tree and enforced by the build graph.
The implemented split places common mechanisms, the public product API, and each exact target in
separate source and link ownership domains.

### 19.1 Physical ownership decision

NInfer has no top-level mutable runtime model layer. The physical target composition unit is one
complete target package:

```text
exact checkpoint + selected GPU + one current storage-consumer profile
```

Its source key uses `<checkpoint_key>_<device_key>`, for example
`qwen3_6_27b_rtx5090`. The source key is not an artifact field and is never inferred from a directory
at runtime. The package still declares and validates the registered `.ninfer` `model_id`, actual
device, and complete object signatures through compiled code.

There is no runtime or target-package family base directory, `qwen/common`, `BaseTarget`, or shared
checkpoint implementation created in advance. A later target starts as another complete sibling
for binding, frontend, Program, schedule, and state policy. Narrow family-common conversion tooling
does not create a shared runtime schedule. Every device transformation remains classified by the Op
boundary in Section 18.

### 19.2 Canonical repository tree

The implemented source tree follows this ownership shape:

```text
include/
└── ninfer/
    ├── engine.h                         public Engine PIMPL API
    ├── types.h                          public owning host values/opaque prepared prompt
    └── ops/                             repository-internal semantic Op contracts; not public API

src/
├── core/                                device/stream, tensor/view, layout, arena, graph/KV/transfer
├── artifact/                            .ninfer reader, binder, layouts, materializer
├── ops/
│   ├── wrapper/                         contract validation, workspace scope, finite dispatch
│   ├── launcher/                        private CUDA launch policy
│   ├── kernel/                          __global__/__device__ implementations
│   ├── common/                          narrow CUDA implementation primitives
│   └── linear/                          codec/plan/reference/GEMV/GEMM implementation family
├── text/                                checkpoint-neutral Unicode primitives
├── media/
│   └── decode/                          neutral image/video decode over owning bytes
├── product/
│   ├── media_acquire/                   path, URL, and data-URI acquisition for products
│   └── prompt_input/                    shared product message-input adapter
├── runtime/
│   ├── contract/                        target/controller values and lifetime contracts
│   ├── generation/                      budget, stop/cancel, round resolution, publication
│   └── engine/                          public PIMPL implementation and request memory
├── targets/
│   ├── registry.h / registry.cpp         one explicit cold target registry
│   └── qwen3_6_27b_rtx5090/
│       ├── CMakeLists.txt                explicit target-private source list and dependencies
│       ├── export/ninfer/targets/qwen3_6_27b_rtx5090/
│       │   └── package.h                 narrow composition facade
│       └── impl/
│           ├── package.cpp               facade definitions and private construction
│           ├── config.h                  exact semantic constants and token domains
│           ├── load/                     typed immutable bindings
│           ├── frontend/                 prompt/output, tokenizer, template, media preprocessing
│           ├── program/                  plans, layouts, prefix/state, graphs, Program owner
│           ├── state/                    target-specific recurrent/state representations
│           ├── schedule/                 private Text, Vision, and MTP execution
│           └── diagnostic/               explicitly linked target diagnostics
└── serve/                               protocol schemas, translation, streaming, transport

apps/
├── cli/                                 local `ninfer` product
└── serve/                               `ninfer-serve` product

tools/
├── artifact/                            generic .ninfer read/write/layout/inspection machinery
├── convert/common/                      generic checkpoint-reading/quantization helpers
├── convert/qwen3_6/common/               narrow Qwen3.6-family conversion leaves
├── convert/qwen3_6_27b_rtx5090/         source mapping and conversion recipe
├── convert/qwen3_6_35b_a3b_rtx5090/     accepted future-target conversion recipe
├── reference/qwen3_6_27b_rtx5090/       artifact-native Text/Vision/MTP Python reference
├── parity/qwen3_6_27b_rtx5090/          independent target/reference diagnostics
├── qwen3_6_27b_dump/                    C++ target activation dump
├── bench/                               corpus generation and performance orchestration
├── smoke/                               resident-server product smoke client
└── freq_corpus/                         registered draft-head frequency input and records

bench/                                   public Engine benchmark plus operator microbenchmarks

tests/                                   component, operator, frontend, target, and product tests
```

The runtime structure is implemented only for the 27B target. Conversion tooling may precede a
runtime registration when it implements an accepted complete artifact contract, as the 35B-A3B
converter does; this does not create an Engine target or product route. A later runtime target is
added as a complete sibling and an explicit registry/build entry, never as an empty placeholder.
The top-level roots, one-directory-per-exact-target rule, physical `export/` versus `impl/` split,
target responsibility partitions, package facade choke point, and dependency directions are
normative. Family common tooling contains only identity-free leaves: exact targets may depend on it,
but may not import sibling targets. Files may split or merge inside one owner when their invariant
and dependencies remain the same.

### 19.3 Inside one target package

Every target package keeps these responsibilities distinguishable even if a small implementation
uses fewer translation units:

| Area | Owns | Must not own |
|---|---|---|
| `export/.../package.h` | the package tag and narrow facade types used by the registry/controller | public product API, target implementation includes, protocol behavior, or a second execution owner |
| `impl/package.cpp` | facade definitions and private construction bridge | a second public/composition surface |
| `impl/config.h` | exact compiled semantic facts needed by this package, including dimensions, topology, token domains, and native limits | artifact offsets, conversion provenance, request state, user policy |
| `impl/load/` | target-private binding plan, typed immutable weight/resource structures, and direct final-address `LoadedModel` construction behind the facade `LoadPlan` | sequence state, generation policy, runtime lookup after publication |
| `impl/frontend/` | owning `PreparedPrompt`, tokenizer/template, checkpoint media/position preparation, `OutputSession`, and decoder state | network/file acquisition, CUDA state, Program or graph ownership |
| `impl/program/` | `SequencePlan`, `RequestPlan`, layouts, the sole mutable `Program`, core KV instances, recurrent/MTP/sampler state, prefix ledger, graph ownership, and round resolution | user callbacks, protocol schemas, artifact name lookup |
| `impl/state/` | target-specific recurrent/state representations bound by Program layouts | common runtime policy, mathematical Op implementations, request publication, or a second state owner |
| `impl/schedule/` | target-private definitions of Program's fixed Text/Vision/MTP/MoE execution methods | mathematical Op implementations, another long-lived execution owner, or an API callable by product code |

`export/package.h` is an internal build interface, not a public product API. It is the registry
and whole-loop controller's target composition point. Target implementation types remain beneath the
target directory and are not reachable from public product headers. Once a Program call enters
`impl/`, its fixed layer schedule uses direct target-private state and calls repository-internal Op
contracts with explicit execution views and semantic parameters.

Files under `impl/schedule/` are physical splits of private Program behavior. They do not introduce
another object that retains Program-owned state. A non-owning helper either remains a private Program
method or takes an explicit short-lived context. The only long-lived mutable sequence owner is
`Program`; the only long-lived immutable target owner is `LoadedProduct`/`LoadedModel`.

Later target directories need not contain identical files. For example, another target may require
private MoE scheduling and state that the current target does not. Its routed-expert computation is
still a central Op; empty target symmetry and family-shaped base classes are not introduced in
advance.

### 19.4 Dependency direction

In the following graph, `A -> B` means A may depend on B:

```text
serve -> public include/ninfer API + media acquisition
apps -> public include/ninfer API + product prompt input
product prompt input -> media acquisition
media acquisition -> public owning media values + platform/network facilities
runtime engine -> target composition + runtime generation + core
target registry -> generation controller + each target export/package.h
target package -> runtime contract + artifact + ops + core + neutral text/media-decode primitives
runtime generation -> runtime contract + public host values
media decode -> host codec mechanisms
artifact -> core
ops -> core
core -> standard library + selected platform/toolchain APIs
```

The following restrictions make that graph enforceable:

- public product headers do not include CUDA, tensor, artifact, Op, kernel, or target-private
  types; repository-internal `include/ninfer/ops/` contracts may include required L0/CUDA host types;
- `core`, `artifact`, `ops`, neutral `text`, and `media/decode` never include a target header;
- a target never includes `Engine`, `GenerationController`, serving/transport code, or another
  target, and never links product media acquisition;
- the explicit registry is the common runtime composition point for concrete target exports;
- Program may consume the target-private prepared-prompt data contract, but it does not call
  `Frontend` or `OutputSession` services; frontend implementation has no dependency on Program
  implementation;
- an Op implementation or launcher never imports a target/Program merely to obtain dimensions or
  state; the target passes explicit views and semantic parameters, while exact shape remains a valid
  wrapper dispatch predicate;
- Qwen tokenizer rules, chat templates, placeholder expansion, patch construction, MRoPE positions,
  and output channel markers remain in the target frontend. Only genuinely checkpoint-neutral
  Unicode and decode-over-owning-bytes mechanics live in `src/text` or `src/media/decode`;
- remote URL/file/data acquisition remains in `src/product/media_acquire`; protocol translation and
  transport callbacks remain in serve or apps. A target frontend receives owning media input and
  never performs acquisition or transport work.

There is one deliberate composition edge: `registry.h` and `registry.cpp` assemble the closed
`ActiveTarget` variant. They see package exports but contain no checkpoint mathematics.

The include and namespace are derived only from the compiled source key, for example:

```cpp
#include <ninfer/targets/qwen3_6_35b_a3b_rtx5090/package.h>

namespace ninfer::targets::qwen3_6_35b_a3b_rtx5090 { /* facade declarations */ }
```

The directory key, include path, namespace suffix, CMake target suffix, and registry tag use the same
snake-case identifier. It is a build identity, never a runtime-discovered path or artifact field.

### 19.5 Public and internal header rules

Only `include/ninfer/engine.h` and `include/ninfer/types.h` form the public product header surface. `Engine`
uses PIMPL, so adding or replacing a target does not place a CUDA, target, artifact, or Op header in
the public dependency graph. Public types are owning host values, explicitly bounded host views, or
opaque move-only handles. `PreparedPrompt`'s public envelope is opaque; concrete prepared values,
`LoadedModel`, `Program`, target checkpoint facts, tensors, and device sampling controls remain
internal.

For engine source and build identities, lowercase `ninfer` is fixed here as the public
include-directory name and C++ root namespace, and as the stem of the internal component targets in
Section 19.7. The repository directory and user-facing executables use their NInfer identities; the
source tree and executable `--help` define their exact current surfaces.

Each target's `export/` directory is a scoped internal composition interface and is not part of the
public product header surface.
Its `impl/` directory remains private to that exact target; neither path is part of the user-facing
C++ API.

Repository-internal mathematical Op headers live under `include/ninfer/ops/`; they are development
contracts, not product ABI or public product headers. Op benchmarks and numerical tests gain access by
linking `ninfer_ops` and its internal include path.

Production headers are not reshaped by testing macros. Diagnostics that require file output,
reference taps, or instrumented schedules live in a separate tool/test target or use an explicit
internal diagnostic seam. They do not add `#ifdef TESTING` members to a production target type.

### 19.6 Op source organization

The semantic contract and implementation layers are physically distinct:

```text
include/ninfer/ops/<family>.h      mathematical/state-transition contract
src/ops/wrapper/<family>.cpp      validation, workspace scope, finite dispatch
src/ops/launcher/<family>.*       private CUDA launch policy
src/ops/kernel/<family>*.cuh      __global__/__device__ implementation
src/ops/common/                   narrow CUDA implementation primitives
src/ops/linear/                   codec/plan/reference/GEMV/GEMM implementation family
```

The exact file count may vary, but these ownership rules are fixed:

- every contract comment defines the complete formula/indexing, logical shapes, supported domain,
  numerical behavior, effects/aliases, and workspace requirement;
- wrappers validate semantic inputs and select a finite implementation from format, layout,
  numerical shape, token regime, state dtype, and device facts;
- launchers own grid/block/shared-memory policy and CUDA entry invocation;
- kernels and common/linear helpers are private implementation material and are never included by a
  target schedule;
- every implementation receives explicit storage/state/workspace/stream arguments and never owns
  allocation, CUDA Graphs, sequence cursors, or commit policy;
- exact-shape, fused, or selected-device implementations stay in `src/ops`; only their dispatch
  predicate is specialized.

The word kernel remains correct for CUDA implementation code and profiling. It does not name the
host-callable semantic layer. [`op-development.md`](op-development.md) is authoritative for formula
comments, admission, naming, layering, testing, and performance workflow.

### 19.7 Build graph and closed registration

Directories are backed by separate CMake targets and explicit source lists. The implemented split is:

```text
ninfer_core
ninfer_artifact                    -> ninfer_core
ninfer_ops                         -> ninfer_core
ninfer_text                        -> host Unicode mechanism
ninfer_media_decode                -> host codec mechanism
ninfer_media_acquire               -> host network/filesystem mechanism
ninfer_product_prompt_input        -> media_acquire
ninfer_qwen3_6_27b_rtx5090         -> artifact + ops + text + media_decode
ninfer_engine                      -> explicit target + artifact + core
ninfer_serve                       -> engine + media_acquire
apps/ninfer                        -> engine + product_prompt_input
apps/ninfer-serve                  -> serve
```

Each exact target is an independently buildable static target with an explicit list of `.cpp/.cu`
sources. The final engine statically links the selected closed set. A target addition or replacement
changes both its explicit build entry and registry variant in the same coherent change.

`GLOB`, `GLOB_RECURSE`, directory scanning, static-constructor registration, shared-library plugin
discovery, and fallback-to-family registration are forbidden. They can silently change the product
target set or hide dependency violations. The explicit registry is the sole common construction
point that names the complete target set.

Converter/reference tooling is not linked into runtime target libraries. A converter's source
mapping, quantization/layout recipe, and reference configuration live under its own target-keyed
tool directory. Agreement is demonstrated by emitted artifacts, the independently compiled binder,
and parity evidence—not by importing one universal target configuration into both converter and
runtime.

These boundaries are verified through component compilation/linking and real behavior checks. They
must not be enforced by unit tests that scan source text for forbidden include names.

### 19.8 Placement and extraction rule

Place a new value or function by the complete invariant it must maintain:

1. bytes, ranges, alignment, device/stream lifetime, or generic materialization belong to
   `core`/`artifact`;
2. a complete tensor or explicit local-state transformation independent of checkpoint schedule is
   an Op and belongs to `include/ninfer/ops` plus `src/ops`; no cross-target-use proof is required;
3. checkpoint dimensions/topology, operand/view selection, Vision composition, GDN/MoE state
   lifetime, MTP orchestration, prefix repair, exact graph, and selected-device schedule policy
   belong to the exact target package;
4. stop/output-budget/cancellation policy, decoder preview/commit ordering, and publication belong to
   runtime generation; checkpoint token-to-text/channel preview and its mutable decoder state remain
   in the target frontend's `OutputSession`;
5. schemas, network/filesystem acquisition, and transport behavior belong to
   serve/product code and are not target dependencies.

Small duplication of schedules and state policy between targets is preferred to a false base class.
This does not permit duplicate target-owned Op implementations. A narrow shared helper is named
after its invariant; catch-all `common.h`, `utils.h`, generic `model.h`, global `ModelConfig`, or
phase-flag-driven family schedules are not acceptable destinations.

Files split because ownership, lifetime, dependency direction, compilation language, or independent
verification changes—not merely because a line threshold was crossed. A helper used by one
translation unit remains in its anonymous namespace instead of creating another shared header.

### 19.9 Implemented ownership summary

| Responsibility | Owner |
|---|---|
| dimensions, layer topology, token domains, and native limits | exact target `impl/config.h` |
| artifact inventory matching and immutable typed weight/resource bindings | target `impl/load/` over generic `artifact` mechanisms |
| tokenizer, template, media preprocessing, owning prepared prompt, and output decoding | target `impl/frontend/` |
| KV/GDN/MTP state, stable graph buffers, sampling state, prefix ledger, and request plans | target `Program` under `impl/program/` |
| fixed Text, Vision, and MTP execution order | target-private `impl/schedule/` |
| output budget, stopping, cancellation, exact-prefix publication ordering, and common summaries | runtime generation and public `Engine` implementation |
| device/tensor/layout/arena/graph/KV/transfer primitives | `ninfer_core` |
| semantic Op contracts and all mathematical/local-state implementations | `include/ninfer/ops`, `ninfer_ops` |
| checkpoint-neutral Unicode and image/video decoding | `ninfer_text` and `ninfer_media_decode` |
| product path/URL/data acquisition | `ninfer_media_acquire` |
| HTTP schemas, request translation, streaming, and transport | `ninfer_serve` and `apps/serve` |
| local prompt acquisition and presentation | `apps/cli` |
| conversion, Python reference, parity, and activation diagnostics | target-keyed `tools/` components |

Common `Engine` contains no Qwen dimension, GDN/MTP/MoE/Vision field, target cache formula, or
target-private setter protocol. Immutable binding is `LoadedModel`, mutable execution is `Program`,
input/output semantics are `Frontend`, and fixed computations are Program-private schedules.

## 20. Design provenance from reference engines

The boundary was informed by local vLLM, llama.cpp, and SGLang source study. That research provided
useful evidence for processed-frontier tracking, stable graph inputs, target-owned composite state,
explicit model implementations, accepted-length-aware stop handling, and separating generic loading
mechanisms from model-specific binding. External file paths and commit snapshots are intentionally
not part of this active contract: the sections above and the current NInfer source tree are the
authority.

The resulting decisions are:

| Concern | NInfer decision |
|---|---|
| stable CUDA Graph inputs | adopt program-lifetime stable device and host buffers |
| processed versus logical progress | adopt explicit frontier invariants |
| speculative rejection | require target normalization before reuse/publication |
| recurrent plus KV commit | keep one target-owned transaction |
| accepted-batch stop matching | stage text and resolve the exact accepted token prefix |
| explicit model implementations | adopt a closed compiled target registry |
| generic materialize, target-specific bind | adopt separate artifact and target-load components |
| source/build discovery | require explicit package source lists and registry composition |
| continuous batching scheduler | defer; outside current one-request scope |
| paged multi-request cache | defer; no current owner or invariant requires it |
| generic model graph/forward mode | reject from the common contract |
| runtime family/config compatibility | reject; targets are exact checkpoints |
| plugin/model ABI | reject; packages are compiled with the engine |

## 21. Rejected designs

### 21.1 Generic `IModel::forward`

A universal forward call either exposes a lowest-common-denominator tensor bundle or accumulates
phase flags for Text, Vision, recurrent state, MoE, verification, and proposal. It cannot express
complete sequence transactions without importing model bookkeeping into the caller. NInfer instead
dispatches once around the whole exact-target loop.

### 21.2 Public phase enum or graph IR

`Prefill`, `Decode`, `Verify`, `Draft`, `Vision`, and `MoE` are not a stable common sequence. A phase
enum would become a stringly/dynamically driven model program and force controller knowledge of
checkpoint schedules. They remain private methods and graph choices.

### 21.3 Container-driven execution

The `.ninfer` JSON describes persistent object identities and locations. Its canonical object names
are deliberately the inputs used by the compiled binder. What it does not define is the model-role
mapping, fixed-slot table, view/alias instructions, execution recipes, graph nodes, kernel names,
hardware policy, or a binding program. Adding those to teach runtime execution would duplicate
compiled model contracts and weaken validation. Load plans are generated by compiled code after
parsing; they are not artifact metadata.

### 21.4 Common model-state structs

A public struct containing KV, GDN state, positions, MTP slots, MoE routes, or Vision data would be a
Qwen-specific union disguised as reuse. `Program` owns a target-private state aggregate and exposes
only lifecycle operations and summaries.

### 21.5 Model-specific prefill overloads

Overloads such as text IDs, `ProcessedInput`, cached text, and future MoE/multimodal variants grow the
wrong common surface. A checkpoint frontend produces one target-typed prompt; `begin` implements all
valid paths behind one contract.

### 21.6 Public one-token phase API plus a hidden speculative queue

Returning one token while retaining the rest of a speculative batch hides unresolved state and makes
mid-batch stops ambiguous. The controller receives the complete licensed batch as `GeneratedRound`
and resolves Program once with the selected exact prefix.

### 21.7 Arbitrary round rollback

Universal rollback requires snapshots of every KV, recurrent, draft, sampler, and graph control
state. It is expensive and not guaranteed by current algorithms. NInfer requires full commit,
prefix-and-finish, or invalidation only.

### 21.8 Engine-owned memory formulas

Layer counts and cache dimensions in a common Engine-owned cache formula duplicate model config
outside the target and drift from allocation order. Target layout plans are the single definition.

### 21.9 Runtime name lookup or persistent repacking

Strings and artifact geometry are construction-only. Runtime lookup, repacking, or dense fallback
would add cold decisions and memory work to every target while allowing artifacts that do not match
the optimized program. Bind once to typed fields and execute directly.

### 21.10 Premature concurrent scheduler structures

Adding batch slots, page ownership, request maps, preemption state, or locks now would create an
untested architecture for a workload not yet implemented. Future continuous batching must begin
from measured scheduling and memory requirements and preserve the per-sequence correctness
principles above; it need not preserve this singleton API or its exact internal state set.

### 21.11 Split immutable and mutable runtime ownership

An object that binds immutable weights while borrowing Engine-owned cache, recurrent state,
workspace, step buffers, and policy controls is neither a loaded product nor a Program. It creates
setter protocols and splits one sequence invariant across two owners. Binding belongs to
`LoadedModel`, mutable execution belongs to Program, frontend semantics belong to Frontend, and
schedule definitions are private Program implementation.

### 21.12 Monolithic glob-built runtime library

A recursive source glob and one library containing core, artifact, Ops, targets, frontend, and
runtime allow every layer to include every other layer. Directory names then communicate style but
enforce no architecture; adding a file can also silently change the shipped target set. NInfer uses
explicit component/target source lists and one explicit closed registry link.

## 22. Normative invariants

An implementation conforming to this architecture satisfies all of the following:

1. **Exact selection:** one artifact and actual device select one explicit compiled package or fail.
2. **No artifact program:** JSON names objects but cannot define role mapping, slot/view instructions,
   model execution, kernels, or hardware policy.
3. **Complete binding:** every artifact object is consumed exactly once as an inventory entry, every
   required role is satisfied, and every unexpected object is rejected before target activation.
4. **Cold-load boundary:** an active `TargetInstance` is complete; artifact names and file access
   leave the inference path after construction.
5. **Typed hot path:** execution reaches persistent resources through target-typed bindings.
6. **Single owner:** one non-movable program owns all mutable state for the one resident sequence.
7. **Single active request:** no second begin and no second unresolved round are possible.
8. **Plan before mutation:** request capacity and transient memory are established before `begin`.
9. **Stable graph addresses:** graph-referenced host/device storage does not move during program life.
10. **Owning prompt:** asynchronous execution never reads caller-owned request or media memory.
11. **Target-private phases:** common code cannot invoke model subphases or manipulate caches/state.
12. **Frontier correctness:** every resolved Active state satisfies `S = E + 1`; Resident satisfies
    `E <= S <= E + 1`, and all states obey the planned `C_exec` bounds.
13. **Licensed output:** every returned token is valid target output, never an unverified proposal.
14. **One round resolution:** Program resolves a returned round once as full Continue or terminal
    exact prefix; failure makes the request non-reusable.
15. **No false rollback:** `abort_request` never claims restoration; arbitrary partial continuation is absent.
16. **Accepted-prefix reuse:** no state beyond the accepted logical sequence is reachable after finish.
17. **Preview before publish:** stop/output decisions and model resolution precede user callbacks.
18. **Exact decoder commit:** tokenizer state advances by the same token prefix committed to Program.
19. **Target capacity:** common code does not derive model context/output formulas or token domains.
20. **Stable model execution:** steady decode and MTP perform no device allocation or file I/O, use
    Program-owned token/workspace storage, and introduce no per-round transaction PIMPL or candidate
    lattice. This does not prohibit ordinary output string/vector growth.
21. **No silent degradation:** unsupported layout, hardware, capacity, or native component fails
    instead of choosing an unregistered fallback.
22. **Physical target ownership:** every checkpoint binding, topology, schedule, state-lifetime, and
    graph-policy invariant lives in its exact target package; numerical shape/GPU specialization of
    an Op remains centrally owned, and no parallel mutable target owner borrows Program state.
23. **Acyclic dependencies:** lower mechanisms and ordinary runtime files cannot include concrete
    target internals; target packages cannot include Engine/controller/serve or one another.
24. **One composition choke point:** concrete target types enter common construction/execution only
    through the explicit registry, `ActiveTarget`, and target `export/package.h` facade.
25. **Explicit product build:** source lists and the registered target set are explicit; recursive
    globbing, auto-registration, plugin discovery, and family fallback cannot change the product.
26. **Opaque public API:** public headers expose no CUDA, tensor, artifact, Op, kernel, or target
    type.
27. **Acquisition stays above target execution:** exact target execution consumes owning media and
    does not perform URL/filesystem acquisition.

## 23. Architectural verification obligations

Implementation of this decision is complete only when the following are demonstrated for every
registered target:

### 23.1 Construction

- missing, unexpected, wrong-shape, wrong-format/layout, and wrong-GPU artifacts are rejected before
  target execution;
- the active target contains its loaded resources, frontend, request memory, and Program;
- typed bindings cover the exact complete selected inventory;
- every typed view still names its intended backing after construction, proving that no bound
  self/subobject reference was invalidated by a move;
- planned bytes and actual bound regions come from the same checked layout definition.

### 23.2 Prompt and memory lifetime

- text and native multimodal prepared prompts own every asynchronous input;
- prepared prompts for the same exact target alternative may cross Engine instances; a different
  target alternative is rejected before planning;
- moving the prepared prompt and request plan into `begin` cannot invalidate a plan-borrowed
  pointer because RequestPlan contains no such borrow;
- large/dynamic prefill workspace cannot move graph-stable buffers;
- successful and recoverably failing `begin` paths leave no asynchronous access to destroyed prompt
  storage or reusable `TransientRegion` memory;
- maximum planned context and representative media inputs fit or fail deterministically.

### 23.3 State transitions

- fresh prefill, ordinary decode, full MTP acceptance, partial acceptance, correction, and context-tail
  fallback satisfy the `E`/`S` invariant;
- every launch and result satisfies `E < C_exec` and `E + n <= C_exec`;
- stop at every returned MTP position satisfies the exact frontier/canonical equations in Section
  11.2 and leaves that accepted logical prefix reusable or explicitly Invalid;
- cancellation before and after a round never advertises provisional state;
- an exception with provisional or committed-but-unpublished work leaves `GenerationGuard` armed and
  makes the Program sequence non-reusable without CUDA rollback;
- prefix append, explicit checkpoint restore, and full reset preserve KV/recurrent/MTP consistency;
- cross-round stop strings, UTF-8 tails, reasoning/content transitions, token stops, output limits,
  and cancellation keep Program token count, decoder state, and published bytes consistent;
- request/effective limit equality and truncation report `OutputLimit`/`ContextCapacity` exactly as
  specified, including zero-capacity requests;
- synchronous, queued-owning, and partially failing output sinks satisfy the ownership/completion
  contract without retaining borrowed bytes or making a failed publication sequence reusable.

### 23.4 End-to-end target fit

- the registered 27B target executes Text, image/video multimodal input, composed MTP inputs,
  sampling, and prefix reuse through only the coarse controller contract;
- output row padding and tokenizer-addressable ID boundaries cannot emit undecodable IDs;
- greedy and non-greedy MTP verification are checked against their mathematical oracle, including
  proposal/target `q/p` acceptance and correction required to preserve the target distribution;
- useful decode/prefill/context results are measured through the complete product path.

A future 35B-A3B target must, before registration, demonstrate that its GDN/full-attention/MoE,
image/video and multiple-media `rope_delta`, Vision/MTP alignment, and prefix state fit this boundary
without adding phase or expert fields to common APIs. This is an admission requirement and design-fit
claim, not evidence that a 35B runtime target is already implemented.

### 23.5 Source and build boundaries

- public API consumers compile without `src/`, CUDA, or target include directories;
- core, artifact, Ops, text, media decode, media acquisition, exact target, Engine, and serving
  are explicit CMake targets with declared source lists and link directions;
- each exact target builds as one independently named static target and does not depend on Engine or
  serving;
- the explicit registry names and constructs the complete registered target set;
- the registered 27B target executes through the public Engine without exposing checkpoint branches
  to CLI, serving, or benchmark code;
- converter/reference tools remain absent from runtime link dependencies;
- adding an unlisted source file or target directory cannot change the built product until the
  explicit CMake source list and registry are deliberately updated.

These checks protect numerical behavior, binary contracts, GPU lifetime, and observable output.
They must not be replaced with tests that scan source files for preferred class names or call order.

## 24. Decision boundary for later work

This document fixes the implemented engine seam:

```text
container object directory
  -> compiled exact-target binder/load plan
  -> immutable loaded product
  -> one non-movable target program
  -> target-licensed `GeneratedRound` views and exact-prefix resolution
  -> common output controller
```

It leaves these later decisions open:

- additional model, layout, and resource-encoding registrations beyond the initial identities
  governed by the container, storage-layout, and model-artifact specifications;
- conversion specifications and object inventories for additional models beyond the registered
  Qwen3.6-27B target;
- target-private schedule/state leaf-file splitting, central Op kernel optimizations, and target
  graph/workspace optimizations within the fixed ownership boundaries;
- the future continuous-batching scheduler and its memory policy.

Those decisions may refine implementation beneath this seam. They must not reintroduce a generic
model graph, move checkpoint state into common controller code, weaken round resolution, or make
artifact metadata executable.
