# NInfer Core Engine Architecture

> Status: accepted on 2026-07-14, pending implementation.
>
> Authority: this document defines the future NInfer core engine boundary, exact-target registration
> and dispatch, load-time construction, memory ownership, checkpoint frontend boundary,
> single-request program contract, decode-round transaction, sequence-state invariants, and the
> division between common infrastructure and target-private implementation. It also defines the
> future repository/source ownership, internal header visibility, target-package layout, and build
> dependency direction that enforce those boundaries. It does not define the currently delivered
> engine implementation, model mathematics, persistent tensor numeric semantics, `.ninfer` binary
> framing, model-specific object inventories or layouts, conversion recipes, serving protocols,
> repository migration steps, or future concurrent scheduling.
>
> Project purpose comes from [`ninfer-project-positioning.md`](ninfer-project-positioning.md).
> Persistent numeric formats come from [`ninfer-tensor-formats.md`](ninfer-tensor-formats.md), and
> artifact framing and metadata come from
> [`ninfer-container-format.md`](ninfer-container-format.md). Exact Qwen3.6 checkpoint mathematics
> remain in [`qwen3.6-27b-architecture.md`](qwen3.6-27b-architecture.md) and
> [`qwen3.6-35b-a3b-architecture.md`](qwen3.6-35b-a3b-architecture.md). The currently delivered NInfer
> engine remains governed by [`design.md`](design.md) until the corresponding migration is complete.

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

The stable boundary has three principal parts:

1. a **target package** turns a validated `.ninfer` directory into one immutable, typed loaded
   product for the actual GPU;
2. a **checkpoint frontend** turns product-level input into one fully owning, target-typed prepared
   prompt and supplies output-decoding semantics;
3. a **single-request program** owns all mutable GPU and host state for one resident sequence and
   exposes begin, transactional decode-round resolution, finish, and invalidation operations.

The common generation controller is compiled around this contract. It owns output limits, stop
conditions, cancellation, output staging, and publication. It does not own model caches, recurrent
state, speculative details, model positions, or model-specific memory arithmetic.

The central shape is:

```text
Engine
├── DeviceContext
├── ArtifactReader + Materializer
├── LoadedProduct                         loaded-Engine lifetime, immutable after construction
│   ├── exact-checkpoint Frontend
│   ├── typed persistent resources
│   ├── typed device weights
│   └── exact checkpoint × GPU implementation
├── SingleRequestProgram                  exactly one resident/active sequence
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
    ├── transactional output decoder
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
- one `SingleRequestProgram` with at most one active request;
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
- stable project-owned source or binary compatibility during the migration.

Future continuous batching is an important possible project phase, but it is a different scheduling
architecture that must be designed for real high-performance batching. It must re-prove, per
sequence, exact-target ownership, accepted-token/publication consistency, and the
logical-versus-materialized relation at every resumable/reusable boundary. It may replace the
singleton `Program`/`PendingRound` API, add internal states, and hold work for many sequences at once.
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
- transactional staging of decoded text from a returned token batch;
- resolving a decode round before publishing corresponding output;
- publication to the CLI, local server, or another product adapter;
- final user-visible token/text accounting.

Stop strings are not model execution state. MTP acceptance is not output policy. Keeping them in
their respective owners prevents a model program from calling transport code and prevents the
controller from modifying speculative caches.

### 4.4 Ownership rule

An object belongs to the lowest layer that can state its complete invariant. If an invariant names
Qwen positions, GDN slots, MoE experts, MTP drafts, Vision merger columns, vocabulary padding, or a
specific graph capture, the object is target-private. If an invariant is only about bytes,
alignment, one device, one active request, or committed output, it may be common.

Reuse occurs only after two real targets demonstrate identical semantics. Similar spelling or
similar tensor rank is insufficient.

## 5. Closed target registry and dispatch

### 5.1 Registry key

After generic artifact validation, NInfer first selects one closed target selector with:

```text
(artifact.model_id, actual selected device identity and capabilities)
```

The artifact does not carry a GPU selector. The device is observed locally, and the selected target
selector matches the complete canonical object signature set against its compiled storage profiles.
Exactly one match selects the concrete package type. Zero or multiple matches are hard errors before
payload-sized allocation.

The storage-profile match uses existing container fields—canonical name/kind/shape/format/layout or
resource encoding—and never invents a profile field in JSON. Offsets, byte order in the object array,
and other nonsemantic file placement do not affect the match. The selected package then performs the
full binder validation and device-consumer check.

An unknown model, wrong GPU, unavailable device capability, ambiguous profile, or unsupported
storage combination is a load error before payload-sized allocation.

### 5.2 Internal static package contract

The package contract is compile-time and intentionally coarse. The following is the normative
shape; concrete helper names may vary only where the same ownership and call ordering remain clear:

```cpp
struct SomeExactCheckpointRtx5090 {
    static constexpr std::string_view model_id;
    static constexpr DeviceClass device_class;
    static constexpr StorageProfile storage_profile;

    struct LoadPlan;
    struct BindingPlan;
    struct LoadedModel;
    struct Frontend;
    struct PreparedPrompt;
    struct OutputSession;
    struct SequencePlan;
    struct RequestPlan;
    class Program;

    static LoadPlan plan_load(ArtifactBinder&, const DeviceInfo&);

    static std::unique_ptr<LoadedModel> construct_loaded_model(
        BindingPlan&&,
        MaterializedArtifact&&,
        DeviceContext&);

    static Frontend make_frontend(const LoadedModel&);

    static SequencePlan plan_sequence(
        const LoadedModel&,
        DeviceContext&,
        std::uint32_t requested_context);

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
are visible, while target-private weights, state, schedules, and kernel policy remain behind an
opaque implementation owned by the target library. Section 19 defines the corresponding
export/implementation split.

`plan_load` consumes and validates the complete artifact directory. It does not receive feature or
quality toggles: one package represents the one current complete product route.
`construct_loaded_model` may run only after every selected payload and resource is valid and
materialized. It allocates the final heap-stable `LoadedModel`, moves the backing into that object,
and forms typed views only after every subobject they may reference is at its final address. It must
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
    ProductCookie cookie;
    std::unique_ptr<LoadedProduct<Target>> loaded;
    RequestMemory request_memory;
    std::unique_ptr<typename Target::Program> program;
};

using ActiveTarget = std::variant<
    std::unique_ptr<TargetInstance<TargetA>>,
    std::unique_ptr<TargetInstance<TargetB>>>;
```

Each `TargetInstance<T>` contains its non-reused load-instance cookie, an immutable heap-stable
`LoadedProduct<T>`, separate reusable request memory, and its one `T::Program` in that declaration
order. C++ reverse destruction therefore destroys Program first, request memory second, and loaded
resources last. `LoadedModel` is itself a stable heap object: target binding forms no reference to a
movable owner or descriptor subobject, and `LoadedProduct` constructs `frontend` from the final
`*model` only after installing that owner. The program and frontend may then refer to immutable
loaded resources without depending on aggregate moves. `Engine::Impl` similarly declares
`DeviceContext` before `ActiveTarget`, so the complete target dies before its device/stream owner.

Every concrete storage profile is an alternative of the same outer closed sum type; there is no
inner layout tag or string dispatch in `LoadedModel`/`Program`. Shared source templates may generate
multiple alternatives without changing this rule.

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
  -> observe actual device and select one compiled target package
  -> target plan_load consumes the complete object directory
  -> validate inventory, roles, shapes, formats, layouts, encodings, and consumers
  -> compute host/device placement plans
  -> allocate and materialize persistent objects
  -> establish required payload-content and resource invariants
  -> allocate one never-reused ProductCookie for this construction attempt
  -> construct LoadedModel at its final heap address, move in backing, then form typed bindings
  -> construct LoadedProduct around that stable model and then construct Frontend from it
  -> plan sequence memory against post-load device capacity
  -> construct RequestMemory
  -> allocate TargetInstance at its final address with cookie/loaded/request memory
  -> construct and install Program as the last member; warm/capture required graphs
  -> release directory/name index and make the complete TargetInstance active
```

An active target is complete: its loaded product, request memory, and Program already exist. The
Program is never constructed as an external local and later assembled around owners it already
references; its `LoadedModel`, request memory, target instance, and device context are at their
final addresses first. Ordinary RAII owns temporary construction state. This architecture does not
require an immutable-file transaction, a special atomic-publication protocol, or failure-injection
machinery for the project-managed local artifact workflow.

### 6.2 `ArtifactBinder`

`ArtifactBinder` is a cold-path checked view over the parsed directory. It provides lookup by
canonical name, exact kind/shape/storage validation, checked view construction, and consumption
tracking. It does not interpret layer schedules.

The target package enumerates its expected roles, consumes each named entry, and rejects every
missing or unconsumed object. Array order is never a role key. The binder and target together enforce
the container requirements; the engine does not maintain a second compiled offset or order table.

The binder exists only during construction. Artifact strings do not select behavior after loading.

### 6.3 `LoadPlan` and materialization

`LoadPlan` is a target-generated, move-only construction object with two explicit parts:

```cpp
struct Target::LoadPlan {
    CommonMaterializationPlan materialization;
    Target::BindingPlan bindings;
};
```

`CommonMaterializationPlan` is a complete value contract the generic materializer can execute. It
contains checked artifact object handles, final backing regions/alignment, copy spans and legal
coalescing, resource decode/construction destinations, and payload validators. `BindingPlan` is the
target-private value mapping consumed objects to typed roles and checked views after materialization.

Neither part contains raw pointers/references into `ArtifactBinder`, a JSON DOM/string, or a
temporary name index. Stable numeric object handles may refer to the reader while construction is in
progress, but all such handles are consumed before the reader is released.

It is not stored in the artifact, not serialized, and not retained as a runtime execution recipe.

The generic materializer consumes `CommonMaterializationPlan` and returns `MaterializedArtifact`,
which owns every final device arena, retained pinned/host allocation, and decoded resource backing
needed by binding. It executes checked allocations, reads, and copies. It never changes a
persistent tensor representation, guesses a layout, chooses a kernel, or synthesizes a missing
object. File offsets are not device offsets. Validators that need payload bytes may stream before
allocation or operate on materialized host/device data. Construction keeps staging and destination
storage alive for the operations that use them; the exact read/copy implementation is not an
architectural protocol.

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
to `LoadPlan`, `ArtifactBinder`, JSON, the name index, the file mapping/descriptor, or load-only
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
owners directly. If any decision depends on resident state, the plan records the program's sequence
epoch. `begin` rejects a stale epoch rather than applying a plan to changed state.

Every target-private plan provides a const `summary()` returning the common
`RequestPlanSummary`. The controller copies that bounded value before moving the plan into `begin`;
it never recomputes or substitutes the requested output limit afterward.

Request-memory growth occurs before `begin`. A Vision allocation or large prefill must never move or
invalidate graph-stable addresses. On every successful return or recoverable exception, `begin`
must complete or safely cancel every asynchronous read/use of the moved `PreparedPrompt` and
`TransientRegion` before returning or propagating. Program stores no pointer/view into either input,
and no CUDA Graph captures the transient region. The prompt may be destroyed and the region may be
reset/reused immediately after either outcome. A device failure that prevents safe quiescence
follows Section 12.2 instead of returning a recoverable exception. Output decoder staging lives in
its separate owning CPU session.

### 7.5 No hidden inference-path allocation

After a request begins, ordinary decode and MTP rounds perform no heap allocation, filesystem access,
artifact-name lookup, device allocation, or graph-address change. Returned token storage is a fixed
program-owned host buffer. Model-private workspaces are planned or bounded before execution.

## 8. Checkpoint frontend and prepared prompts

### 8.1 Frontend boundary

The frontend belongs to the loaded exact checkpoint. It owns:

- tokenizer and detokenizer behavior;
- chat template and generation-prompt construction;
- checkpoint-native default stop tokens/strings;
- checkpoint-specific media validation and preprocessing over already authorized owning media;
- placeholder expansion and modality/token alignment;
- checkpoint-native position construction;
- assistant-content or other prefix-checkpoint hints;
- validation of tokenizer-addressable input IDs;
- creation of a transactional output decoder.

Protocol schemas and network fetching remain outside the target. The frontend receives already
authorized product input/media and produces checkpoint-native data.

Each target frontend supplies the preparation entry points required by that checkpoint and returns
its `T::PreparedPrompt`. Raw frontend argument types may differ with native capabilities and are
specified by the target/product input document, not forced into one core feature bag. The stable
core postcondition is the owning prepared-prompt contract below. Every `T::Frontend` also provides:

```cpp
OutputSession make_output_session(
    const T::PreparedPrompt&,
    const StopPolicy& caller_stop) const;
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

- a value-semantic `ProductCookie` generated uniquely for this load instance;
- a small common `PromptSummary`, including prompt token count;
- one alternative of the closed variant of opaque target-typed prepared values.

A prompt produced for one loaded product cannot be submitted to another, even if `model_id` happens
to match. `ProductCookie` is never a raw address: it is a checked, process-wide non-reused
load-generation value owned authoritatively by `TargetInstance`. Its allocator is safe across
concurrent Engine construction and never reissues a value within the process. The cookie is
allocated before that construction attempt's loaded objects, consumed even if construction later
fails, and published only as the cookie field of the complete ActiveTarget. Exhaustion fails
construction rather than wrapping. Engine preparation copies that field into the envelope;
submission compares it against the currently visited `target.cookie`.

The closed prepared-value variant's alternative is the target tag; there is no second integer or
string dispatch field that can drift from it. Inside the already selected `TargetInstance<T>`
visitor, common code checks the cookie and performs one `get_if<T::PreparedPrompt>`/move. There is no
second cross-product visit and no hot-path type lookup. Every check occurs before `plan_request` and
sequence mutation.

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

### 8.4 Output decoder transaction

The frontend creates a request-local output session containing the resolved frontend-default plus
caller stop policy and a decoder. The session owns all state it needs after the prepared prompt is
moved into `begin`; it never borrows prompt storage.

The decoder uses a staging API equivalent to:

```cpp
class OutputDecoder {
public:
    StagedText stage(std::span<const TokenId> tokens, const StopPolicy&) const;
    StagedText stage_terminal(FinishReason) const;
    DecoderCommitPlan prepare_commit(
        StagedText&&,
        const OutputResolution&) const;
    PublishedOutput commit(DecoderCommitPlan&&) noexcept;
};
```

`stage` may inspect UTF-8 boundaries, tokenizer state, and stop strings but does not mutate committed
decoder state and does not invoke a user callback. The staged object contains precomputed decoder
states and token-to-byte boundaries for the possible accepted-token prefixes. `stage_terminal`
stages a flush with no new token for cancellation or another terminal decision between rounds.

`StagedText` exposes bounded immutable candidate summaries to the controller: issued
`StagedChoiceId`, committed-token count, finish reason, channel domain, decoded byte-cut ordering,
and resolved stop declaration order. Decoder-private next states and byte buffers remain opaque.
This is enough for the deterministic selection in Section 13.4 without exposing tokenizer internals.

`OutputResolution` records the exact accepted logical-token count, Continue versus Finish, terminal
reason, and an opaque candidate selected from that exact `StagedText`. `prepare_commit` validates the
selection and constructs the complete next decoder state and owning output deltas without mutating
the decoder. It is the final fallible state-preparation step and runs before model resolution. After
the Program has resolved the same token count, decoder `commit` and generation-budget bookkeeping
perform only no-fail swaps/moves and checked arithmetic. Decoder `commit` is `noexcept` and
allocation-free. The later ownership-transferring `publish` call is the sole allowed synchronous
failure after state resolution, with the invalidation consequence defined below.

The decoder therefore advances by exactly the accepted logical-token prefix even when published
text differs. This is required because one MTP round can return several tokens and a stop string may
end inside that batch.

The staged result maps a textual stop back to an exact token prefix. If the stop begins or ends
inside one token's decoded byte contribution, that token is still included in the model-committed
prefix while the resolved stop policy may withhold the stop bytes from published text. A stop
spanning token boundaries retains enough uncommitted decoder tail to identify the same exact token
count. Byte trimming and token-state commit are therefore related but not falsely treated as the
same index.

On Continue, the commit plan publishes only bytes that cannot still become the prefix of a
cross-round stop string and retains the ambiguous tail. On Finish, it flushes that tail except for
bytes intentionally withheld by the selected stop policy. The same accepted token count can
therefore produce different safe publication at Continue and Finish; the continuation flag is
mandatory.

`PublishedOutput` is an ordered owning sequence of channel-tagged byte deltas, at minimum
`Reasoning` and `Content`. Checkpoint-native thinking markers, channel transitions, incremental UTF-8,
and stop matching are handled while staging. A resolved stop string declares its matching channel
domain; ordinary caller stop strings default to Content unless the product surface explicitly asks
for another domain. Serving-specific tool parsing remains outside this decoder contract.

Publication consumes that ownership through a contract equivalent to
`publish(OutputSink&, PublishedOutput&&)`. The sink may consume bytes synchronously or move/copy them
into storage it owns, but it may never retain a borrowed view into the argument. Normal return is
the controller's publication completion point: every delta has been accepted into sink-owned
lifetime. A synchronous exception means acceptance failed, possibly after an externally visible
prefix; the batch is not retried and the generation call fails. An asynchronous adapter may define
successful ownership-transferring enqueue as acceptance, in which case a later socket/client
failure belongs to that adapter and is not retroactively reported as this generation's publication
failure. This contract separates payload lifetime and controller completion from physical network
delivery.

The implementation may use a small checkpoint/copy of decoder state instead of literal objects with
these names. The observable requirements—stage without publication, commit an exact prefix, and
preserve byte-correct streaming—are mandatory.

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

struct RequestOptions {
    ExecutionOptions execution;
    StopPolicy caller_stop;
};

enum class RoundContinuation : std::uint8_t {
    Continue,
    Finish,
};

enum class FinishReason : std::uint8_t {
    None,
    OutputLimit,
    ContextCapacity,
    StopToken,
    StopString,
    Cancelled,
};

struct StagedChoiceId {
    std::uint32_t value; // opaque; meaningful only with the StagedText that issued it
};

struct OutputResolution {
    std::uint32_t committed_tokens; // 1..round size, or 0 only for stage_terminal()
    RoundContinuation continuation;
    FinishReason finish_reason;    // None iff continuation == Continue
    StagedChoiceId staged_choice;   // valid only for the exact StagedText being prepared
};
```

Only `ExecutionOptions` reaches `Program::plan_request`. The target validates the supported sampling
domain and translates those values into its own stable device controls. Kernel-facing sampling
structures remain private to the target/operator layer. Stop strings, output channels, and caller
publication policy never enter the model program.

`execution.requested_output_tokens == 0` performs no model execution. It is not represented by a
prefill that produces and then hides a token.

The frontend merges checkpoint defaults with `caller_stop` once when it creates `OutputSession`.
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
    std::uint32_t committed_output_tokens;
    FinishReason finish_reason;
    std::optional<FinishDisposition> sequence; // Resident/Invalid after a begun request
};
```

`begin` returns `BeginRound<Program>`, a move-only pair of `BeginSummary` and a one-token
`PendingRound<Program>`. The first output is therefore staged and resolved through the same contract
as every later batch; it is not accepted merely because prefill succeeded.

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

Resolution selection proves `committed_tokens <= remaining` before Program mutation.
`GenerationBudget::commit` and `round_budget` are then `noexcept`; a violated bound is an internal
fatal invariant, not a recoverable exception after model/decoder commit.

The output matrix row count, tokenizer-addressable ID count, valid sampled-token domain, and stop-ID
domain are distinct checkpoint facts. A common `vocab_size` field must not collapse them.

## 10. `SingleRequestProgram` contract

### 10.1 Required coarse operations

For a package `T`, its program provides the following target-typed operations to the templated
controller:

```cpp
enum class ProgramState : std::uint8_t {
    Empty,
    Active,
    PendingRound,
    Resident,
    Invalid,
};

struct SequenceSummary {
    ProgramState state;
    std::uint64_t epoch;
    std::uint32_t materialized_tokens; // E
    std::uint32_t logical_tokens;      // S
};

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

    BeginRound<Program> begin(
        PreparedPrompt&&,
        RequestPlan&&,
        TransientRegion);

    PendingRound<Program> decode_round(RoundBudget);

    void finish_active();
    void abort_active() noexcept;
    void clear_resident() noexcept;

    ProgramState state() const noexcept;
    SequenceSummary sequence_summary() const noexcept;
};
```

`SequenceSummary` is diagnostic committed state, not a mutation interface. `Empty`, `PendingRound`,
and `Invalid` report zero counts because none exposes a reusable resolved logical identity;
provisional progress is hidden. Section 11 defines valid Active and Resident count relationships.

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

On success, it returns `BeginRound<Program>` containing exactly one target-licensed token and enters
`PendingRound`. All prompt input reads are complete before control returns. The prepared prompt,
prefill state, and sampled token are not yet a reusable logical sequence: the controller must stage
the token and consume the handle with the same Continue/Finish/discard rules as every later round.

`commit_all` on the begin round establishes `Active` with `E = P` and `S = P + 1`.
`commit_prefix_and_finish(1)` establishes a terminal Resident or Invalid result. Destruction/discard
invalidates the candidate sequence. There is no first-token exception to transactional output.

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

`PendingRound::tokens()` is a span over program-owned fixed-capacity host memory. Its contents remain
valid until that round is resolved. A target may use pinned host memory and one synchronized count/
token transfer per round. It does not allocate and return `std::vector<int>` on every decode step.

Only target-licensed output tokens appear in the span. Unverified proposals, rejected drafts, padded
slots, and model-internal row indexes never escape as candidate output.

### 10.5 Operation and destruction postconditions

The common controller relies on these exact failure guarantees:

- `plan_request` is read-only; success or exception leaves Program state/epoch unchanged;
- `begin` follows Section 10.2, never exposes a partial new identity, and before propagating any
  recoverable exception it quiesces or safely cancels all asynchronous access to its moved
  `PreparedPrompt` and `TransientRegion` inputs;
- `decode_round` enters `PendingRound` only when all licensed output is ready; any recoverable
  exception after mutation but before returning a handle leaves Program `Invalid` and leaves no
  untracked asynchronous access;
- a resolution method marks its handle resolved only after its full Active/Resident/Invalid
  postcondition holds; if it throws, the still-unresolved handle destructor invalidates the owner;
- `finish_active` succeeds in Resident and leaves Invalid on a recoverable exception;
- process-fatal device faults follow Section 12.2 instead of pretending to satisfy recovery.

`abort_active()` and `clear_resident()` are `noexcept` host-reachability operations only. They advance
the epoch and make device state unreachable; they do not launch, synchronize, allocate, or attempt
rollback. A later begin performs the target's ordered full device reset.

`PendingRound::~PendingRound()` and `Program::~Program()` are `noexcept`. Moving a handle makes the
source inert. `tokens()` is valid only on the live unresolved handle with the matching owner/round
epoch; misuse is an internal programming error and must never touch a newer round. Engine teardown
quiesces its target stream while DeviceContext is still alive, then destroys Program, RequestMemory,
LoadedProduct, and finally DeviceContext in the order established in Section 5.3.

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

The target may execute provisional work beyond these values inside an unresolved `PendingRound`,
but that state is neither reusable nor observable through `SequenceSummary` as committed progress.

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

### 11.3 Sequence epoch

Every reset, invalidation, successful begin, round resolution, finish, or resident-prefix change
advances a monotonic host sequence epoch. Request plans and prefix matches that depend on resident
state record this epoch. An epoch is a local stale-plan/handle guard, not a public request ID and not
a concurrency scheduler.
The fixed-width counter is incremented with checked arithmetic. It never wraps: exhaustion makes the
current TargetInstance unusable and requires reconstruction.

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

The common lifecycle is:

```text
Empty ─────────────── begin success ──────────────> PendingRound
Resident ──────────── begin success ──────────────> PendingRound
Invalid ── full-reset begin success ──────────────> PendingRound

Active ────────────── decode_round ───────────────> PendingRound
PendingRound ──────── commit_all ─────────────────> Active
PendingRound ──────── commit_prefix_and_finish ───> Resident or Invalid
PendingRound ──────── discard / unresolved dtor ──> Invalid

Active ────────────── finish_active ──────────────> Resident
Active ────────────── abort_active ───────────────> Invalid
Resident/Invalid ──── clear_resident ─────────────> Empty
```

Only one pending round may exist. `decode_round` is illegal outside `Active`; `begin` is illegal
while active or pending; `finish_active` is illegal while a round is pending.

`Invalid` means host logic must not select the current device state for reuse. A later `begin` may
recover only through a target-defined full reset. It does not imply that arbitrary CUDA failures are
recoverable.

`FinishDisposition` is the small common result of terminal round resolution:

```cpp
enum class FinishDisposition {
    Resident, // exact accepted logical prefix is coherent and may be considered for reuse
    Invalid,  // output is valid, but no current sequence state may be reused
};
```

### 12.2 CUDA fault policy

The current process-fatal CUDA policy is compatible with this state machine: a fatal device error
terminates instead of pretending the program can be reset. `Invalid` covers host cancellation,
unresolved provisional work, and logical failures for which device state still exists but is not
trusted.

If NInfer later adopts recoverable device faults, it must add a distinct `Faulted`/reload contract
that proves stream and allocation validity. It must not overload `Invalid` with unsupported CUDA
recovery.

## 13. Transactional generated-token rounds

### 13.1 `PendingRound`

`begin` returns a one-token handle inside `BeginRound`; `decode_round` returns the same move-only RAII
handle directly. Every handle is bound to its owner program:

```cpp
template<class Owner>
class PendingRound {
public:
    PendingRound(const PendingRound&) = delete;
    PendingRound& operator=(const PendingRound&) = delete;
    PendingRound(PendingRound&&) noexcept;

    std::span<const TokenId> tokens() const noexcept;

    void commit_all() &&;
    FinishDisposition commit_prefix_and_finish(std::size_t count) &&;
    void discard() && noexcept;

    ~PendingRound() noexcept;
};

template<class Owner>
struct BeginRound {
    BeginSummary summary;
    PendingRound<Owner> round;
};
```

Exactly one resolution action is permitted:

- `commit_all()` commits the entire returned batch to the logical sequence, normalizes the model
  execution frontier according to Section 11, and makes the program ready for another round;
- `commit_prefix_and_finish(m)` adopts exactly `1 <= m <= tokens().size()` returned logical tokens
  and finishes the request at that prefix. It returns whether an exact reusable `Resident` state was
  constructed or the sequence had to become `Invalid`; published text may be empty or shorter;
- `discard()` makes the sequence `Invalid` without claiming rollback.

Resolution methods are rvalue-qualified so a handle is consumed syntactically as well as logically.
The handle carries its owner and round epoch; a moved-from or stale handle cannot resolve another
round. It never escapes the internal generation controller, and its owner program outlives it.

Destroying an unresolved handle is equivalent to `discard()`. The destructor performs no CUDA
launch, synchronization, allocation, or throwing operation; it only invalidates host reachability.
A debug build should additionally diagnose an unresolved destructor after invalidation.

### 13.2 Why arbitrary continuation is forbidden

The interface intentionally has no `commit(m)` that continues generation, no `commit(0)`, and no
`rollback()`.

A speculative target may prepare the next proposal state only for its complete accepted outcome.
An intermediate returned token can be target-licensed for output while lacking the MTP, recurrent,
or cache state needed to continue as if the suffix had never happened. Similarly, pre-round backup
state may not exist for every mutable component. Claiming arbitrary continuation would either be
false or force expensive universal snapshots into every target.

The only required partial action ends the request. The target can then canonicalize exactly the
accepted logical prefix, select a verified per-token recurrent snapshot, rewind a
position-addressed cache,
or retain an allowed final pending token without promising continuation through an unsupported
intermediate proposal state. If it cannot construct and prove that exact state, it still accepts the
licensed logical token prefix but returns `Invalid`; no prefix from that round is subsequently
reusable.

### 13.3 Provisional state

During `PendingRound`, target-private state may include:

- verification KV entries and recurrent snapshots;
- accepted-count and target-token buffers;
- draft-model KV and autoregressive hidden state;
- sampler probability/RNG/occurrence deltas for the candidate logical outputs;
- a next-round proposal prepared for the complete outcome;
- graph control scalars and provisional cursor values.

None of it is common API. `commit_all` normalizes every component to the full returned logical batch,
leaving its last token as the new unprocessed frontier. It does not claim that every returned token
has a corresponding materialized KV entry. `commit_prefix_and_finish` either normalizes exactly the
accepted logical prefix and makes all later state unreachable, or invalidates reuse. `discard`
invalidates the entire resident identity.

Sampler storage is physically Program-resident but logically Active-request state. `commit_all`
commits exactly the returned logical tokens. Finish retires sampler state; the next begin always
initializes it from the new `ExecutionOptions`. A target may therefore avoid repairing suffix-only
penalty counters on terminal partial finish, but it may never continue with or expose those counters
as active/resident semantics.

### 13.4 Stop and cancellation resolution

For a returned batch, the controller:

1. stages tokenizer/channel decoding and all stop candidates without mutation;
2. selects one `OutputResolution` containing the exact accepted logical-token prefix `m`;
3. prepares the no-fail decoder commit and owning channel deltas;
4. consumes the Program handle with `commit_all()` for Continue, or
   `commit_prefix_and_finish(m)` for Finish;
5. commits the decoder with the same resolution using its `noexcept` commit plan;
6. charges `m` against the generated-token budget;
7. only then publishes the prepared output deltas.

`m` counts model-logical generated tokens and is always at least one for a returned round. It
includes a terminating EOS/stop token and the token containing a stop-string byte boundary, even if
that token contributes zero published bytes. Published byte count and accepted token count are not
interchangeable.

Resolution selection is deterministic. It chooses the earliest accepted-token boundary among the
effective budget limit, stop token, and staged stop-string candidates. At the same token boundary,
the earliest decoded byte cut wins; therefore a string cut that would otherwise leak bytes wins over
a token-only finish. Equal string cuts use resolved `StopPolicy` declaration order. If no textual
cut wins, StopToken precedes the budget's recorded `effective_limit_reason` for the reported reason.
`OutputLimit` versus `ContextCapacity` is already fixed by the RequestPlan rule in Section 9.2,
including the equality case; `decide_resolution` never reconstructs it from counters.
`staged_choice` identifies the corresponding candidate inside that exact staged object; it is never
reused across stages.

Cancellation is sampled at controller boundaries. Before `begin`, cancellation returns with no model
work. Before a later round, the controller stages a terminal decoder flush, calls `finish_active`,
commits the decoder with `{0, Finish, Cancelled}`, publishes the flush, and retains the coherent
Resident result. If cancellation becomes visible while `begin` or `decode_round` is executing, the
returned round is discarded, the sequence becomes Invalid, and only the previously committed
decoder tail is terminally flushed; none of the new round tokens are accepted or published. A signal
arriving after resolution is observed at the next boundary. This is the single current policy, not a
per-target choice.

If `publish` throws after model commit, the model state is not rolled back. The controller reports
the publication failure, and `GenerationGuard` makes the generated sequence unreachable for reuse
even if a coherent state had existed immediately before publication. Bytes already accepted by a
partially failing sink cannot be retracted; the generation call still fails and the sequence is not
reused.

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
        prompt, options.caller_stop);
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

    GenerationGuard guard(*target.program, Disarmed);
    std::optional<BeginSummary> begin_summary;

    const auto resolve_and_publish = [&](auto&& pending)
        -> std::optional<GenerationSummary> {
        if (cancellation.requested()) {
            auto staged = output.decoder.stage_terminal(FinishReason::Cancelled);
            const auto resolution = output.cancelled_resolution(staged);
            auto decoder_commit = output.decoder.prepare_commit(
                std::move(staged), resolution);
            std::move(pending).discard();
            auto deltas = output.decoder.commit(std::move(decoder_commit));
            publish(sink, std::move(deltas));
            return finish_summary(
                *begin_summary, FinishDisposition::Invalid, resolution.finish_reason);
        }

        auto staged = output.decoder.stage(pending.tokens(), output.stop_policy);
        const auto resolution = decide_resolution(staged, pending.tokens(), budget);
        auto decoder_commit = output.decoder.prepare_commit(
            std::move(staged), resolution);

        std::optional<FinishDisposition> disposition;
        if (resolution.continuation == RoundContinuation::Continue) {
            require(resolution.committed_tokens == pending.tokens().size());
            std::move(pending).commit_all();
        } else {
            disposition = std::move(pending).commit_prefix_and_finish(
                resolution.committed_tokens);
        }

        auto deltas = output.decoder.commit(std::move(decoder_commit));
        budget.commit(resolution.committed_tokens);
        publish(sink, std::move(deltas));

        if (resolution.continuation == RoundContinuation::Finish) {
            return finish_summary(
                *begin_summary, *disposition, resolution.finish_reason);
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
    if (auto done = resolve_and_publish(std::move(first.round))) {
        guard.complete();
        return *done;
    }

    while (budget.remaining() != 0) {
        if (cancellation.requested()) {
            auto staged = output.decoder.stage_terminal(FinishReason::Cancelled);
            const auto resolution = output.cancelled_resolution(staged);
            auto decoder_commit = output.decoder.prepare_commit(
                std::move(staged), resolution);
            target.program->finish_active();
            auto deltas = output.decoder.commit(std::move(decoder_commit));
            publish(sink, std::move(deltas));
            guard.complete();
            return finish_summary(
                *begin_summary, FinishDisposition::Resident, FinishReason::Cancelled);
        }

        auto round = target.program->decode_round(budget.round_budget());
        if (auto done = resolve_and_publish(std::move(round))) {
            guard.complete();
            return *done;
        }
    }

    unreachable_budget_state();
}
```

This pseudocode fixes ownership and ordering, not exact public names. The begin round and every later
round use the same resolver. Their only structural difference is the begin summary and the work the
target performed before constructing the handle.

`GenerationGuard` is an internal scope guard, not a public request object. It is constructed
disarmed before `begin` so begin failure preserves that operation's own postcondition, then armed
only after a handle exists. Because the handle is constructed later, stack unwinding destroys and
invalidates it before the guard runs. Until `complete()`, the guard makes any remaining Active
sequence Invalid and clears any just-finished Resident identity. These guard actions are host-only
and `noexcept`; the next `begin` performs the target's real full reset. Consequently no generation
state containing output that failed to publish is offered for prefix reuse.

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
boundary. Partial finalization may require a rare target-private scalar repair or synchronization;
that cost is preferable to an incorrect reusable prefix.

### 15.4 Hot-path exclusions

Prefill/decode/MTP hot paths contain no:

- JSON parse or artifact string lookup;
- filesystem operation;
- runtime persistent repack or quantization;
- target registry lookup;
- per-layer polymorphic dispatch;
- per-round heap or device allocation;
- hidden synchronization caused by movable or temporary graph buffers;
- serving callback from inside model execution.

## 16. Concrete fit: Qwen3.6-27B

This section maps the existing implementation into the contract and identifies corrections required
before it can conform. It does not claim that the current implementation already satisfies the new
invariants, and it does not make existing implementation types part of the future engine contract.

### 16.1 Loaded product

The 27B target's `LoadedModel` owns typed bindings for:

- Text embedding, 64 hybrid decoder layers, final norm, full head, and any selected optimized head;
- one-layer MTP weights and its shared Text embedding/head semantics;
- Vision transformer, patch merger, and required frontend resources;
- exact format/layout descriptors already validated during binding.

The current `WeightStore`, `Qwen3_6_27B`, and `Qwen3_6_Vision` ownership is therefore reorganized
under one immutable target product. Tokenizer resources remain owned there instead of being removed
through `take_tokenizer_bundle()`.

### 16.2 Program state

The 27B `Program` owns:

- 16 full-attention Text KV caches and the MTP KV cache;
- 48 GDN convolution histories, FP32 recurrent matrices, and required snapshot slots;
- logical token identity, exact prefix-reuse ledger, and assistant-boundary checkpoint;
- position/rope controls and multimodal continuation state;
- sampling controls, occurrence counters, licensed output buffer, and MTP counters;
- ordinary-decode and MTP-round graphs and every graph-stable tensor;
- prefill, verification, proposal, head, and Vision workspaces with their lifetimes separated.

These objects currently spread across `Engine`, model cards, cache/state classes, and `StepState`.
They move together because only the 27B target can state their cross-component invariant.

### 16.3 Fresh and reused begin

For text, `begin` chooses full reset, append continuation, or an exact boundary restore based on the
prepared prompt and resident epoch. Boundary selection and GDN snapshot scheduling are one target
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

This requires a real correction, not only ownership movement. The current 27B MTP prefill rebuilds
shifted inputs through raw token embedding lookup; for an image/video placeholder it must instead
carry the corresponding Vision-merger composed column into MTP. Multimodal Text/MTP block parity is
a migration gate.

### 16.4 Decode round

Ordinary decode returns one licensed token. An MTP round privately performs target verification,
acceptance, GDN slot selection, MTP proposal preparation, and next-round state construction. It
returns only `[accepted target-licensed drafts..., target correction/bonus]` through the fixed output
buffer.

The existing host `pending_sampled_` queue disappears. A batch is inspected and resolved as one
`PendingRound`, which prevents the controller from exposing one token at a time while silently
retaining an unresolved speculative suffix.

If a stop occurs on an intermediate returned draft, `commit_prefix_and_finish` may preserve a
resident prefix only by rewinding/canonicalizing full-attention KV, selecting the matching GDN
snapshot, fixing Text/MTP positions and frontier controls, normalizing any sampler/occurrence state
that remains relevant, and making later draft and proposal state unreachable. If the 27B target
cannot prove all of those conditions, it invalidates the resident sequence. It never promises that
generation can continue from an arbitrary intermediate MTP proposal state.

For the 27B MTP result `[accepted drafts..., correction/bonus]`, let `n = accepted_drafts + 1`.
The required resolved states are:

| Resolution | Materialized `E_f` | Logical `S_f` | 27B state consequence |
|---|---:|---:|---|
| full Continue of `n` | `E + n` | `E + n + 1` | final correction/bonus is the new frontier |
| terminal `m < n`, Resident | `E + m + 1` | `E + m + 1` | state is canonical through the stopped draft; select GDN snapshot `m` |
| terminal `m = n`, Resident | `E + n` | `E + n + 1` | final correction/bonus remains the frontier |
| any terminal result not exactly repairable | `0` | `0` | `Invalid`; no prefix identity is reachable |

Advertising either Resident form also requires agreement among the host KV cursor, device Text
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
tokenizer-addressable IDs. Migration must apply the valid-sample/decode domain to the first target
sample, ordinary decode, every MTP target correction/bonus, and any optimized draft-head remap.
Validation against 248320 matrix rows alone is not sufficient.

### 16.5 Why the current public shape is not retained

The current `Engine::prefill(span)`, `Engine::prefill(ProcessedInput)`, `prefill_cached`,
`decode_step`, model-specific memory formulas, and kernel-facing `SamplingConfig` expose a mixture of
controller, target, and operator concerns. Adding 35B behind those signatures would either grow
overloads and conditionals or hide incorrect assumptions. The new boundary replaces that surface;
it does not add a compatibility wrapper around it.

## 17. Required fit: Qwen3.6-35B-A3B

The planned 35B-A3B checkpoint is the second concrete design constraint that prevents the common
contract from becoming renamed 27B bookkeeping. This section is a required fit analysis, not a
claim that the target is implemented.

### 17.1 Different loaded structure

The exact checkpoint has 40 Text layers: 30 GDN and 10 full-attention layers, with sparse MoE in
every layer. Each routed-expert bank is rank three, routes among 256 experts, selects eight, and is
combined with a gated shared expert. Its one-layer MTP block also contains full attention and MoE.

Those weight roles, expert descriptors, router buffers, and stage-specific kernels are 35B target
types. The common loader only materializes the validated regions in its load plan; it does not gain
an `experts` field or a generic layer graph.

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
- return one first-token `PendingRound` from `begin`;
- return only licensed tokens in a bounded decode round;
- commit the full round, finish at an accepted logical prefix, or invalidate;
- maintain the same `E`/`S` frontier invariant;
- expose target-independent generation summaries.

That is the common interface the 35B implementation must demonstrate. No lower-level phase needs to
be public.

## 18. Reuse below the target package

Target-private scheduling does not mean duplicating every primitive. Reuse is allowed at boundaries
with complete mathematical semantics.

### 18.1 Common L0 candidates

Common core mechanisms may include:

- checked arithmetic, tensors/views, device/stream/event ownership;
- immutable and mutable arenas plus `LayoutBuilder`;
- strict container parser, object binder, and materializer;
- generic direct/packed storage descriptors from registered formats/layouts;
- CUDA Graph lifetime wrappers;
- pinned stable host buffers;
- diagnostic counters and timing/NVTX mechanisms.

Generic KV or recurrent storage belongs here only if its API fully represents the semantics shared
by real targets. Otherwise the storage primitive can be reused while the cache/state object remains
target-private.

### 18.2 Common L1 candidates

Kernel/operator APIs may be shared when they state complete mathematics independent of one model
schedule—for example a registered packed GEMM, an exact RMSNorm variant, a full-attention operation,
or a routed-expert primitive. Hardware/kernel dispatch remains private behind that mathematical API.

An operator does not own model layer order, residual scheduling, MTP alignment, cache commit, or
prefix checkpoints. Those remain in the target package even when every kernel it calls is shared.

### 18.3 Promotion rule

A second target using similar code is a reason to compare invariants, not automatically to extract a
base class. Promote code only when:

1. both targets require the same observable semantics;
2. ownership and lifetime are identical;
3. the common API does not expose either target's private phase structure;
4. specialization and whole-program optimization remain possible;
5. the extracted component can be verified independently at its real shapes.

## 19. Repository, source, and build organization

The ownership contract above must be visible in the source tree and enforced by the build graph.
Moving the existing classes into new directories while retaining one monolithic library would not
solve the problem: the current 27B execution program is split across `Engine`, model cards, core
state objects, and nominally common kernels precisely because all of those files can depend on one
another without a physical boundary.

### 19.1 Physical ownership decision

NInfer has no top-level runtime `model` layer and no C++ `ModelCard` owner. A model card remains a
documentation/publishing term only. The physical implementation unit is one complete target
package:

```text
exact checkpoint + selected GPU + one current storage-consumer profile
```

Its source key uses `<checkpoint_key>_<device_key>`, for example
`qwen3_6_27b_rtx5090`. The source key is not an artifact field and is never inferred from a directory
at runtime. The package still declares and validates the registered `.ninfer` `model_id`, actual
device, and complete object signatures through compiled code.

There is no family base directory, `qwen/common`, `BaseTarget`, or shared checkpoint implementation
created in advance. A later target starts as another complete sibling. Code moves out only under the
promotion rule in Section 18.3 after the real invariants agree.

### 19.2 Canonical repository tree

The future source tree follows this ownership shape:

```text
include/
└── ninfer/                              installed product API only
    ├── engine.h                         Engine PIMPL and opaque handles
    └── ...                              owning host request/result value headers

src/
├── core/                                device/stream/event, tensor/view, arena/layout, graph RAII
├── artifact/                            .ninfer reader, JSON/ranges, binder primitives, materializer
├── kernels/                             proven shared mathematical operators, grouped by operator
│   └── <operator>/
├── text/                                checkpoint-neutral Unicode/text primitives only
├── media/
│   └── decode/                          neutral decode over already-owned, authorized media bytes
├── product/
│   └── media_acquire/                   URL/file authorization and acquisition for apps/serve only
├── runtime/
│   ├── contract/                        target/controller value and lifetime contracts
│   ├── generation/                      stop/budget/cancel, decoder transaction order, publication
│   └── engine/                          public PIMPL implementation and target lifetime
├── targets/
│   ├── registered_targets.h             one explicit closed target list
│   ├── active_target.h                  closed variant composition
│   ├── registry.cpp                     cold selection and complete construction
│   ├── dispatch.cpp                     whole-generation-loop visit/instantiation
│   ├── qwen3_6_27b_rtx5090/
│   │   ├── CMakeLists.txt
│   │   ├── export/
│   │   │   └── ninfer/targets/qwen3_6_27b_rtx5090/
│   │   │       └── package.h            complete narrow facade; sole composition include
│   │   └── impl/
│   │       ├── package.cpp
│   │       ├── checkpoint.h             exact semantic constants and token domains
│   │       ├── load/                    storage profile, plans, immutable LoadedModel/bindings
│   │       ├── frontend/                PreparedPrompt, tokenizer/template/media, OutputSession
│   │       ├── program/                 plans, memory/state, rounds, prefix, graphs, Program owner
│   │       ├── schedule/                private Text, Vision, and MTP schedule definitions
│   │       └── kernels/                 target-only wrappers, fusion, and CUDA implementation
│   └── qwen3_6_35b_a3b_rtx5090/
│       ├── CMakeLists.txt
│       ├── export/
│       │   └── ninfer/targets/qwen3_6_35b_a3b_rtx5090/
│       │       └── package.h
│       └── impl/
│           ├── package.cpp
│           ├── checkpoint.h
│           ├── load/
│           ├── frontend/
│           ├── program/
│           ├── schedule/                includes 35B-private MoE scheduling as required
│           └── kernels/
└── serve/                               protocol schemas, translation, streaming, transport

apps/
├── cli/
└── serve/

tools/
├── convert/common/                      generic container/format writing machinery
├── convert/<target_key>/                conversion-time source mapping and recipe
├── inspect/                             generic artifact inspection
└── parity/<target_key>/                 independent target/reference diagnostics

bench/
├── kernels/
└── targets/

tests/                                   grouped by observable risk, not a source-tree mirror
├── artifact/
├── kernels/
├── targets/
└── serve/
```

The top-level roots, one-directory-per-exact-target rule, physical `export/` versus `impl/` split,
target responsibility partitions, `package.h` choke point, and dependency directions are normative.
The shown leaf-header split is a canonical starting arrangement, not a requirement to create empty
files or keep a one-class-per-file shape. Files may split or merge inside one owner when their
invariant and dependencies remain the same. The final public API method and leaf-header split remains
a separate API decision; only the `include/ninfer/` public/private boundary is fixed here.

### 19.3 Inside one target package

Every target package keeps these responsibilities distinguishable even if a small implementation
uses fewer translation units:

| Area | Owns | Must not own |
|---|---|---|
| `export/.../package.h` | complete facade class definitions with declared methods and opaque ownership, sufficient to instantiate registry/construction/generation templates, plus the target tag and narrow static package contract | any `impl/` include, weight/state fields, schedule declarations, kernel policy, or large inline bodies |
| `impl/package.cpp` | facade special members and package construction entry definitions that bridge to the private implementation | a second public/composition surface or model execution owned by common runtime |
| `impl/checkpoint.h` | exact compiled semantic facts needed by this package, including dimensions, topology, token domains, and native limits | artifact offsets, conversion provenance, request state, user policy |
| `impl/load/` | `StorageProfile`, `LoadPlan`, `BindingPlan`, typed immutable weight/resource structures, and direct final-address `LoadedModel` construction | sequence state, generation policy, runtime lookup after publication |
| `impl/frontend/` | owning `PreparedPrompt`, tokenizer/template, checkpoint media/position preparation, `OutputSession`, and decoder state | network authorization/fetch policy, CUDA state, Program or graph ownership |
| `impl/program/` | `SequencePlan`, `RequestPlan`, layouts, the sole mutable `Program`, KV/recurrent/MTP/sampler state, prefix ledger, graph ownership, and round resolution | user callbacks, protocol schemas, artifact name lookup |
| `impl/schedule/` | target-private definitions of Program's fixed Text/Vision/MTP/MoE execution methods | another long-lived model/card owner or an API callable by common runtime |
| `impl/kernels/` | fused/fixed-shape helpers whose full contract is true only for this checkpoint and GPU | allocation, graph ownership, logical cursor/commit policy, serving behavior |

`package.h` is an internal build export, not an installed product API and not a forward-declaration
index. It must be sufficient, together with declared lower contract headers, to compile the closed
registry and instantiate the whole-loop controller. Contract-facing classes that need private data
use owning opaque implementations with out-of-line special members, or contain only common contract
values; `package.h` never includes a file below that target's `impl/`. This indirection exists only
at coarse construction, preparation, begin, round-resolution, and finish calls. Once a Program call
enters `impl/`, its fixed layer schedule and kernels use direct target-private types and calls.

`impl/schedule/text.cpp`, `vision.cpp`, `mtp.cpp`, or `moe.cpp` are physical splits of private Program
behavior. They do not introduce `TextModel`, `VisionCard`, `MtpModel`, or another object that retains
references to Program-owned state. A non-owning helper either remains a private Program method or
takes an explicit short-lived context. The only long-lived mutable sequence owner is `Program`; the
only long-lived immutable target owner is `LoadedProduct`/`LoadedModel`.

The two current target directories need not contain identical files. For example, 35B may require
private MoE scheduling and routed-expert kernels that 27B does not. Empty symmetry and family-shaped
base classes are forbidden because they suggest compatibility that has not been proved.

### 19.4 Dependency direction

In the following graph, `A -> B` means A may depend on B:

```text
serve / apps -> public include/ninfer API + product media acquisition
product media acquisition -> public owning media values + platform/network facilities
runtime engine -> target composition + runtime generation + core
target composition -> generation template + each target package.h
target package -> runtime contract + artifact + kernels + core + neutral text/media-decode primitives
runtime generation -> runtime contract + public host values
media decode -> host codec mechanisms
artifact -> core
kernels -> core
core -> standard library + selected platform/toolchain APIs
```

The following restrictions make that graph enforceable:

- public headers do not include CUDA, tensor, artifact, kernel, or target-private types;
- `core`, `artifact`, `kernels`, neutral `text`, and `media/decode` never include a target header;
- a target never includes `Engine`, `GenerationController`, serving/transport code, or another
  target;
- ordinary runtime files never include a concrete target export or implementation header; only the
  explicit composition files receive each target's `export/` include root and include its
  namespaced `package.h`;
- a package export transitively includes only standard, public host-value, and declared runtime
  contract headers. It cannot include or expose its target's `impl/`; composition therefore cannot
  bypass the facade to include Program state, loaded weights, schedules, or target kernels;
- Program's facade and implementation may consume the target's owning `PreparedPrompt` value but
  have no dependency on `Frontend` or `OutputDecoder`; frontend implementation has no dependency on
  Program implementation;
- a mathematical operator or launcher never imports a runner/Program merely to obtain dimensions or
  state; dimensions needed by a target-only operation keep that operation in the target package;
- Qwen tokenizer rules, chat templates, placeholder expansion, patch construction, MRoPE positions,
  and output channel markers remain in the target frontend. Only genuinely checkpoint-neutral
  Unicode and decoding of already-owned media live in `src/text` or `src/media/decode`;
- remote URL policy, filesystem authorization and acquisition, protocol translation, and transport
  callbacks remain in `src/product/media_acquire`, serve, or apps. A target has no include or link
  path to that acquisition component; its frontend receives authorized owning media input.

There is one deliberate composition edge, not a general exception: `active_target.h`,
`registered_targets.h`, `registry.cpp`, and `dispatch.cpp` assemble the closed variant. They may see
package exports, but they contain no checkpoint branch inside a generated-token loop and no model
mathematics.

The include and namespace are derived only from the compiled source key, for example:

```cpp
#include <ninfer/targets/qwen3_6_35b_a3b_rtx5090/package.h>

namespace ninfer::targets::qwen3_6_35b_a3b_rtx5090 { /* facade declarations */ }
```

The directory key, include path, namespace suffix, CMake target suffix, and registry tag use the same
snake-case identifier. It is a build identity, never a runtime-discovered path or artifact field.

### 19.5 Public and internal header rules

Only `include/ninfer/` is installed or exported as a public include directory. `Engine` uses PIMPL,
so adding or replacing a target does not place a CUDA, target, or artifact header in the public
dependency graph. Public types are owning host values, explicitly bounded host views, or opaque
move-only handles. `PreparedPrompt`'s public envelope is opaque; concrete prepared values,
`LoadedModel`, `Program`, target checkpoint facts, tensors, and device sampling controls remain
internal.

For future engine source and build identities, lowercase `ninfer` is fixed here as the public
include-directory name and C++ root namespace, and as the stem of the internal component targets in
Section 19.7. The repository directory, user-facing executable names, and environment-variable
prefix already use their implemented NInfer identities, but remain outside this architecture's
authority; the source tree and executable `--help` define their exact current surfaces. A future
distribution package or new service identifier must be named only if such a contract is introduced.

Each target's `export/` root is a scoped internal interface visible only to target composition; it is
not installed. Its `impl/` root is private to that exact target's CMake target. A standalone facade
compile using only `export/` plus declared lower contract include roots must succeed, which prevents
the facade from acquiring a transitive private-header dependency. The entire `src/` tree is never
added as a global include path.

Internal mathematical operator headers also remain under `src/kernels/`; they are project
development contracts, not product ABI. Benchmarks and numerical tests gain access by linking the
specific internal CMake target and its scoped include directory.

Production headers are not reshaped by testing macros. Diagnostics that require file output,
reference taps, or instrumented schedules live in a separate tool/test target or use an explicit
internal diagnostic seam. They do not add `#ifdef TESTING` members to a production target type.

### 19.6 Kernel source organization

Shared L1 code is grouped vertically by complete mathematical operator rather than by global
`wrapper/launcher/kernel` buckets:

```text
src/kernels/rmsnorm/
├── rmsnorm.h                 mathematical contract and host entry
├── rmsnorm.cpp               validation and finite implementation selection
├── rmsnorm_cuda.h            private launcher declaration
├── rmsnorm_rtx5090.cu        launch and device kernel for the selected implementation
└── rmsnorm_detail.cuh        CUDA-only device templates/primitives
```

The exact file count may be smaller, but the language/ownership rules are fixed:

- `.h` contains host-visible declarations and does not expose block/warp/tiling policy;
- `.cpp` owns host validation and dispatch and does not include device-only `.cuh` files;
- `.cu` owns CUDA launches and `__global__` definitions;
- `.cuh` is device-only implementation material included only from `.cu/.cuh`;
- a kernel receives explicit storage/workspace/stream arguments and never owns allocation, CUDA
  Graphs, sequence cursors, or commit state;
- exact-shape MTP round helpers, checkpoint scheduling fusion, and GPU policy used by only one real
  target stay under that target's `impl/kernels/`;
- an operator moves to `src/kernels/` only when Section 18.3 is satisfied. A similar function name
  or rank is not enough.

For 35B, a routed-expert mathematical primitive may become shared L1 after proof. The decision that
every layer routes, how shared and routed experts combine with residuals, and how MTP invokes its MoE
remains 35B target scheduling.

### 19.7 Build graph and closed registration

Directories are backed by separate CMake targets and explicit source lists. The minimum conceptual
split is:

```text
ninfer_core
ninfer_artifact                    -> ninfer_core
ninfer_kernels                     -> ninfer_core
ninfer_text / ninfer_media_decode  -> host-only lower mechanisms
ninfer_runtime_contract
ninfer_target_<checkpoint>_<gpu>   -> required lower components, never media_acquire
ninfer_target_registry             -> explicit target object libraries + generation
ninfer                              -> Engine/runtime + target registry
ninfer_product_media_acquire       -> public host media values + network/filesystem facilities
ninfer_serve / apps                -> public ninfer API + product media_acquire as required
```

Each exact target is an independently buildable static/object target with a local `CMakeLists.txt`,
an `export/` interface include root, a private `impl/` include root, an explicit CUDA
architecture/device requirement, and an explicit list of `.cpp/.cu` sources. Its exported include
interface contains only the namespaced `package.h`; the registry never receives `impl/`. The final
NInfer library statically links the selected closed set. A target addition or replacement changes
both its explicit build entry and `registered_targets.h` in the same coherent change.

`GLOB`, `GLOB_RECURSE`, directory scanning, static-constructor registration, shared-library plugin
discovery, and fallback-to-family registration are forbidden. They can silently change the product
target set or hide dependency violations. Common components must build without any target directory
on their include path; each target must build without another target's include path or the product
media-acquisition component. The final registry link is the first build step allowed to see all
package exports.

Converter/reference tooling is not linked into runtime target libraries. A converter's source
mapping, quantization/layout recipe, and reference configuration live under its own target-keyed
tool directory. Agreement is demonstrated by emitted artifacts, the independently compiled binder,
and parity evidence—not by importing one universal model-card/config source into both converter and
runtime.

These boundaries are verified through component compilation/linking and real behavior checks. They
must not be enforced by unit tests that scan source text for forbidden include names.

### 19.8 Placement and extraction rule

Place a new value or function by the complete invariant it must maintain:

1. bytes, ranges, alignment, device/stream lifetime, or generic materialization belong to
   `core`/`artifact`;
2. a complete mathematical operation independent of checkpoint schedule belongs to shared
   `kernels` only after real cross-target proof;
3. any invariant naming a checkpoint dimension, Vision composition, GDN/MoE state, MTP alignment,
   prefix repair, exact graph, or selected-GPU policy belongs to the exact target package;
4. stop/output-budget/cancellation policy, decoder transaction ordering, and publication belong to
   runtime generation; checkpoint token-to-text/channel staging and its mutable decoder state remain
   in the target frontend's `OutputSession`;
5. schemas, network/filesystem authorization and acquisition, and transport behavior belong to
   serve/product code and are not target dependencies.

Small duplication between 27B and 35B is preferred to a false base class. Extraction requires the
same semantics, ownership, lifetime, failure behavior, performance freedom, and independent oracle.
An extracted helper is named after that narrow invariant. Catch-all `common.h`, `utils.h`, `ops.h`,
generic `model.h`, global `ModelConfig`, or phase-flag-driven family schedules are not acceptable
destinations.

Files split because ownership, lifetime, dependency direction, compilation language, or independent
verification changes—not merely because a line threshold was crossed. A helper used by one
translation unit remains in its anonymous namespace instead of creating another shared header.

### 19.9 Current implementation ownership correction

The following is an ownership mapping, not a migration sequence and not a compatibility promise:

| Current responsibility | Future owner |
|---|---|
| `model/config.h` dimensions, layer indexes, and token domains | target `impl/checkpoint.h`; only facts compiled by that exact package |
| `FullLayerW`, `GdnLayerW`, `MtpW`, Vision weight structs, and `bind()` | target `impl/load/` and immutable `LoadedModel` |
| `StepState`, KV/GDN/MTP state, stable graph buffers, sampling counters | target `Program` and its one checked memory plan |
| Text layer execution in `Qwen3_6_27B` | target-private `impl/schedule/text.cpp` Program methods |
| `Qwen3_6_Vision` binding, workspace formula, and encode path | respectively target `impl/load/`, `impl/program/`, and `impl/schedule/vision.cpp` |
| MTP forward methods in the card plus `Engine::record_propose`, `record_decode_round`, and partial state repair | one target-private `impl/schedule/mtp.cpp` plus Program round transaction |
| Engine cache/workspace formulas and exact tensor allocation order | target `impl/program/` planning/layout definition |
| Engine prefix ledger, GDN slot setters, position repair, and graph choice | target Program state, prefix, round, and graph files |
| `model/processor`, Qwen tokenizer/template, placeholder/patch/MRoPE construction | target `impl/frontend/`; network acquisition and authorization move to the product adapter |
| duplicated text-runner/server UTF-8, reasoning-channel, stop-string, cancel, and publication logic | target OutputDecoder plus common GenerationController according to Sections 8 and 13 |
| q5090-specific `WeightStore` module flags and tokenizer transfer | generic artifact mechanism plus target `LoadPlan`/`LoadedModel` resource ownership |
| generic KV allocation pieces | core only when their API is model-neutral; cursor/commit/composite semantics stay in Program |
| `GdnState`, fixed-shape MTP/GQA/position/Vision helpers, and kernels importing model constants | target state/schedule/kernels until two targets prove a shared operator |
| `FileTap` and filesystem tensor dumps inside production model headers | parity/diagnostic tool targets |
| one `ninfer_core` built by recursive glob | explicit lower libraries, independent target object libraries, and closed registry link |

After this split, common `Engine` contains no Qwen dimension, GDN/MTP/MoE/Vision field, target cache
formula, or model-card setter protocol. The former card is not renamed: immutable binding becomes
`LoadedModel`, mutable execution becomes `Program`, input/output semantics become `Frontend`, and
fixed computations become Program-private schedule definitions.

## 20. Comparison with local reference engines

This architecture was checked against the local source snapshots below. They are design evidence,
not dependencies and not normative authorities:

| Project | Local snapshot used | Relevant evidence |
|---|---|---|
| vLLM | `~/vllm` at `93e3bc8f30b1e135b88a2ffc7f219e3428d39293` | computed-token frontier, accepted/rejected speculative normalization, stable input batches, graph buffers, multimodal preparation |
| llama.cpp | `~/llama.cpp-mainline` at `e920c523e3b8a0163fe498af5bf90df35ff51d25` | explicit architecture/model implementations, batch-local memory context and mutation boundary, hybrid attention/recurrent memory composition |
| SGLang | `~/sglang` at `be9791071a36c582f7db09adf60a2374736bb920` | accepted-length-aware stop handling, post-verification recurrent-state commit, stable graph inputs, cost of generalized forward-mode surfaces |

### 20.1 vLLM: frontier normalization and stable batches

In `vllm/v1/core/sched/scheduler.py`, requests track `num_computed_tokens` rather than requiring a
universal public prefill/decode model. After speculative output, rejected speculative positions are
removed from computed progress before later scheduling. A full prefix-cache hit also backs off one
token so the final token can be recomputed to produce the next sample. These are independent evidence
for an explicit processed frontier and for normalizing provisional work to the accepted logical
prefix.

The legacy `vllm/v1/worker/gpu_input_batch.py` and experimental V2
`vllm/v1/worker/gpu/input_batch.py` both separate stable GPU input buffers from round descriptions;
the runners also prepare multimodal inputs before model execution. NInfer adopts stable graph
bindings and explicit accepted progress. It does not adopt request maps, paged multi-sequence KV,
continuous-batching scheduler output, or the large general-purpose runner because those solve a
different workload.

The generic loader sources under `vllm/model_executor/model_loader/` leave final weight assignment
to the concrete model, while reusable layer APIs and CUDA code have their own trees. That supports
NInfer's generic materializer versus target binder split and vertical shared-operator directories.
NInfer does not adopt vLLM's model-class registry, hook/ABC surface, or generalized runner/model-state
protocol; its exact package and build entry are static.

### 20.2 llama.cpp: target-owned composite memory

`src/llama-memory.h` separates memory from a per-batch memory context and states that mutation occurs
through the context's `apply()` boundary. Hybrid contexts compose attention and recurrent memory.
`src/llama-context.cpp` initializes that context before processing a batch and handles application/
failure explicitly. This supports treating KV, recurrent state, and their mutation as one target
transaction instead of independent common cursors.

The explicit Qwen model sources under `src/models/`, including Qwen3.5 dense/MoE implementations,
bind model-specific tensors and build model-specific graphs. NInfer similarly uses an explicit closed
target registry. It does not adopt llama.cpp's broad architecture enum, runtime graph construction,
optional-tensor/config variants, generic multi-sequence batch API, or virtual memory hierarchy. An
exact NInfer package can express the same principle with concrete types and compile-time scheduling.

The same snapshot also shows the cost of a broad central composition surface: declarations collect
in `src/models/models.h`, architecture switching remains in `src/llama-model.cpp`, and model source
discovery uses a CMake glob. NInfer takes the explicit per-model implementation lesson but replaces
central special cases and implicit discovery with one package export, an explicit source list, and a
closed registry.

### 20.3 SGLang: batch stop positions and verified recurrent commit

`python/sglang/srt/managers/schedule_batch.py` carries `new_accepted_len` into stop-string/token
handling and records the exact finished prefix when a multi-token speculative result contains a
stop. `python/sglang/srt/speculative/eagle_worker_v2.py` commits Mamba/recurrent states after target
verification. In the examined flow, recurrent/KV state for the full accepted run is committed before
later CPU stop trimming; output `finished_len` does not prove that model state was normalized to that
shorter stop prefix. SGLang therefore separately demonstrates accepted-length-aware output trimming
and target-owned full-run recurrent commit. It is evidence that NInfer must keep these concerns
explicit, not evidence that a reusable partial-prefix Resident is automatically available.

Conversely, `python/sglang/srt/model_executor/forward_batch_info.py` exposes a large `ForwardMode`
space including extend, decode, target verify, split prefill, and draft modes, while its generalized
runner coordinates many scheduling and parallelism variants. Those modes are necessary for SGLang's
scope but would leak model/scheduler phases into NInfer's core contract. NInfer keeps them private
inside one exact target's `begin` and `decode_round`.

SGLang's `python/sglang/srt/models/qwen3_5.py`, generalized `model_runner.py`, and layer code that
imports runner state demonstrate another organizational failure mode for NInfer: a concrete model
file can absorb loading, forward modes, platform policy, graph state, and execution context while a
nominal lower layer depends back on the runner. The exact NInfer package is allowed to be specialized,
but immutable load, mutable Program, frontend, and kernel ownership remain distinct and the build DAG
forbids that reverse dependency.

### 20.4 Adopted and deliberately rejected lessons

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

### 21.6 One-token `decode_step()` plus a hidden queue

Returning one token while retaining the rest of a speculative batch hides unresolved state and makes
mid-batch stops ambiguous. The controller receives the complete licensed batch as `PendingRound` and
resolves it once.

### 21.7 Arbitrary round rollback

Universal rollback requires snapshots of every KV, recurrent, draft, sampler, and graph control
state. It is expensive and not guaranteed by current algorithms. NInfer requires full commit,
prefix-and-finish, or invalidation only.

### 21.8 Engine-owned memory formulas

Layer counts and cache dimensions in a common `Engine::default_cache_bytes()` reproduce model config
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

### 21.11 `ModelCard` as a runtime ownership layer

A card that binds immutable weights while borrowing Engine-owned cache, recurrent state, workspace,
step buffers, and policy controls is neither a loaded product nor a Program. It creates setter
protocols and splits one sequence invariant across two owners. NInfer removes that layer instead of
renaming it: binding belongs to `LoadedModel`, mutable execution belongs to Program, frontend
semantics belong to Frontend, and schedule definitions are private Program implementation.

### 21.12 Monolithic glob-built runtime library

A recursive source glob and one library containing core, artifact, kernels, targets, frontend, and
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
14. **One round transaction:** a round is fully committed, prefix-finished, or invalidated exactly
    once.
15. **No false rollback:** discard never claims restoration; arbitrary partial continuation is absent.
16. **Accepted-prefix reuse:** no state beyond the accepted logical sequence is reachable after finish.
17. **Stage before publish:** stop/output decisions and model resolution precede user callbacks.
18. **Exact decoder commit:** tokenizer state advances by the same token prefix committed to Program.
19. **Target capacity:** common code does not derive model context/output formulas or token domains.
20. **Allocation-free rounds:** steady decode and MTP perform no heap/device allocation or file I/O.
21. **No silent degradation:** unsupported layout, hardware, capacity, or native component fails
    instead of choosing an unregistered fallback.
22. **Physical target ownership:** every checkpoint/GPU invariant lives in its exact target package;
    no parallel ModelCard/model owner borrows Program state.
23. **Acyclic dependencies:** lower mechanisms and ordinary runtime files cannot include concrete
    target internals; target packages cannot include Engine/controller/serve or one another.
24. **One composition choke point:** concrete target headers enter common construction/execution only
    through the explicit registry/ActiveTarget/dispatch composition files and complete `package.h`
    facades whose transitive includes never enter target `impl/`.
25. **Explicit product build:** source lists and the registered target set are explicit; recursive
    globbing, auto-registration, plugin discovery, and family fallback cannot change the product.
26. **Opaque public API:** installed headers expose no CUDA, tensor, artifact, kernel, or target type.
27. **Acquisition stays above targets:** exact targets may use neutral decode over authorized owning
    media, but cannot include or link URL/filesystem acquisition or authorization policy.

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
- prepared-prompt product cookies reject cross-load and stale-envelope submission before planning;
- moving the prepared prompt and request plan into `begin` cannot invalidate a plan-borrowed
  pointer because RequestPlan contains no such borrow;
- large/dynamic prefill workspace cannot move graph-stable buffers;
- successful and recoverably failing `begin` paths leave no asynchronous access to destroyed prompt
  storage or reusable `TransientRegion` memory;
- request-plan epoch mismatch fails before mutation;
- maximum planned context and representative media inputs fit or fail deterministically.

### 23.3 State transitions

- fresh prefill, ordinary decode, full MTP acceptance, partial acceptance, correction, and context-tail
  fallback satisfy the `E`/`S` invariant;
- every launch and result satisfies `E < C_exec` and `E + n <= C_exec`;
- stop at every returned MTP position satisfies the exact frontier/canonical equations in Section
  11.2 and leaves that accepted logical prefix reusable or explicitly Invalid;
- cancellation before and after a round never advertises provisional state;
- an unresolved `PendingRound` destructor invalidates without CUDA work;
- prefix append, explicit checkpoint restore, and full reset preserve KV/recurrent/MTP consistency;
- cross-round stop strings, UTF-8 tails, reasoning/content transitions, token stops, output limits,
  and cancellation keep Program token count, decoder state, and published bytes consistent;
- request/effective limit equality and truncation report `OutputLimit`/`ContextCapacity` exactly as
  specified, including zero-capacity requests;
- synchronous, queued-owning, and partially failing output sinks satisfy the ownership/completion
  contract without retaining borrowed bytes or making a failed publication sequence reusable.

### 23.4 End-to-end target fit

- 27B Text, image/video multimodal, composed MTP inputs, sampling, and prefix reuse execute through
  only the coarse controller contract;
- 35B GDN/full-attention/MoE, image, video, multiple-media positions/`rope_delta`, Vision/MTP
  composed alignment, and prefix state fit without adding phase or expert fields to common APIs;
- output row padding and tokenizer-addressable ID boundaries cannot emit undecodable IDs;
- greedy and non-greedy MTP verification are checked against their mathematical oracle, including
  proposal/target `q/p` acceptance and correction required to preserve the target distribution;
- useful decode/prefill/context results are measured through the complete product path.

### 23.5 Source and build boundaries

- public API consumers compile without `src/`, CUDA, or target include directories;
- core, artifact, shared kernels, neutral text/media-decode, and runtime-contract targets each
  compile with only their declared lower-layer include/link dependencies;
- each namespaced `package.h` compiles from its `export/` root with only declared lower contract
  includes and no path to its target's `impl/`;
- each exact target package compiles independently with `impl/` private and without Engine/serve,
  media acquisition, or another target on its include/link path;
- the explicit registry/dispatch composition target links the complete registered set and is the
  only build boundary that sees every target `export/` interface; it sees no target `impl/` root;
- 27B and 35B execute through that final composition without placing checkpoint branches in common
  generated-token code;
- converter/reference tools remain absent from runtime link dependencies;
- adding an unlisted source file or target directory cannot change the built product until the
  explicit CMake source list and registry are deliberately updated.

These checks protect numerical behavior, binary contracts, GPU lifetime, and observable output.
They must not be replaced with tests that scan source files for preferred class names or call order.

## 24. Decision boundary for later work

This document fixes the engine seam needed before implementation planning:

```text
container object directory
  -> compiled exact-target binder/load plan
  -> immutable loaded product
  -> one non-movable target program
  -> transactional target-licensed-token rounds
  -> common output controller
```

It deliberately leaves these later decisions open:

- exact public API methods and the final split among leaf headers under `include/ninfer/`;
- target-private leaf-file splitting within the fixed ownership directories in Section 19;
- additional model, layout, and resource-encoding registrations beyond the initial identities
  governed by the container, storage-layout, and model-artifact specifications;
- conversion specifications and object inventories for additional models beyond the registered
  Qwen3.6-27B target;
- detailed common tensor/arena/materializer class implementations;
- exact CUDA kernels, graph partitions, and workspace schedules;
- CLI and serving API migration;
- migration sequencing and replacement of current legacy engine surfaces—not `.qus` loader compatibility, which
  the accepted container contract already excludes;
- the future continuous-batching scheduler and its memory policy.

Those decisions may refine implementation beneath this seam. They must not reintroduce a generic
model graph, move checkpoint state into common controller code, weaken round resolution, or make
artifact metadata executable.
