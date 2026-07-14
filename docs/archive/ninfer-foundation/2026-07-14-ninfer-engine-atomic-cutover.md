# NInfer C++ Engine Atomic Cutover Implementation Plan

Status: final candidate; gate pending at archival

> This is an implementation plan for one atomic product cutover. It does not change the accepted
> architecture, current runtime behavior, artifact contracts, or supported target before the final
> cutover is complete. The architectural authority remains
> [`../../ninfer-engine-architecture.md`](../../ninfer-engine-architecture.md).

## 1. Goal

Replace the current Qwen3.6-27B/RTX 5090 C++ product route in one coherent change:

```text
q5090 v4.2 .qus
  -> q5090 WeightStore
  -> Qwen3_6_27B / Qwen3_6_Vision model cards
  -> model-shaped Engine
```

with the accepted NInfer route:

```text
qwen3_6_27b_rtx5090.ninfer
  -> production ArtifactReader + ArtifactBinder + Materializer
  -> closed qwen3_6_27b_rtx5090 target package
  -> immutable LoadedModel + target Frontend + non-movable Program
  -> transactional generated-token rounds + common GenerationController
  -> Engine + CLI + server + benchmark
```

The final product must retain the complete current 27B capability set—Text, image/video Vision,
MTP, sampling, prefix reuse, CLI, OpenAI, Anthropic, streaming, and measurement—while removing the
entire executable `.qus` route and the C++ ownership model that made the engine inseparable from one
checkpoint.

This plan has one delivery, one final review, and one completion gate. The work has internal
dependencies, but it has no independently supported loader phase, target-package phase, Program
phase, frontend phase, or serving phase.

## 2. Why this must be one atomic cutover

The current `.qus` Engine is one complete vertical product. The implemented `.ninfer` toolchain is
another complete artifact/reference route, but it is not yet the C++ product. Publishing only part
of the bridge would create one of several false intermediate products:

- a production `.ninfer` loader feeding the old model-shaped Engine would preserve the coupling the
  migration exists to remove;
- a new `LoadedModel` beside the old `WeightStore` would create two weight owners and two binding
  contracts;
- a target `Program` behind the old `prefill()`/`decode_step()` interface would preserve the hidden
  token queue and make MTP stop/state semantics unprovable;
- a new Engine with the old `TextGenerationRunner` and serving loop would keep duplicate stop,
  cancellation, UTF-8, and publication behavior;
- moving source files before all product entry points switch would make the tree look modular while
  the actual ownership remained split across Engine, ModelCard, and common state.

Accordingly:

- implementation order is allowed, but intermediate order is not a product contract;
- no intermediate state is documented, advertised, maintained, or accepted as completion;
- no public flag selects old versus new Engine, loader, or controller;
- no `.qus` fallback, extension auto-detection, compatibility shim, or dual benchmark lane is added;
- the old route may be used read-only to record the pre-cutover baseline, but new code must not
  depend on it;
- the final review evaluates the complete vertical route and the final deletion together.

Commit mechanics are not an architectural concern. Whether implementation uses temporary local
checkpoints or a short commit series, the submitted result must describe one coherent product state
and must not leave transitional compatibility commits as maintained design boundaries.

## 3. Starting point

### 3.1 Already implemented and reused

- `.ninfer` v1 framing, seven persistent numeric formats, two tensor layouts, and
  `raw-bytes-v1` resources are specified and implemented.
- The real `out/qwen3_6_27b_rtx5090.ninfer` artifact contains 1166 tensors and six frontend
  resources and passes the source-checkpoint verifier.
- The generic Python reader/writer/inspector and the narrow C++ reader agree on the real artifact.
- The target-private Python reference consumes only `.ninfer` and implements complete
  Text/Vision/MTP generation, including multimodal composed MTP inputs.
- Existing CUDA kernels already implement the selected 27B/RTX 5090 execution strategies. This is
  an ownership and integration migration, not a kernel rewrite.
- The accepted Engine architecture already fixes target selection, loading, lifetime, Program,
  generated-token transaction, frontend, source-tree, and build-dependency boundaries.

### 3.2 Coupling that still exists in the C++ product

The current tree demonstrates why the cutover is necessary:

- `include/ninfer/runtime/engine.h` publicly includes CUDA/core/model/kernel types and directly owns
  `WeightStore`, `KVCache`, `GdnState`, `Qwen3_6_27B`, `Qwen3_6_Vision`, graph objects, and MTP
  buffers;
- `src/runtime/engine.cpp` computes 27B KV/GDN/MTP/workspace formulas from `model::kCfg`, performs
  target verification/proposal steps, chooses GDN slots, and repairs prefix state;
- `include/ninfer/model/` and `src/model/` split immutable bindings, mutable sequence execution,
  Vision, MTP, and diagnostics across long-lived model-card objects;
- `include/ninfer/core/weight_store*.h` and `src/core/weight_store*.cpp` combine `.qus` parsing,
  model/module identities, device materialization, frontend resources, and 27B-specific views;
- `src/text/text_runner.cpp` and `src/serve/generation_service.cpp` contain overlapping generated-
  token, cancellation, stop, channel, and publication logic;
- `src/CMakeLists.txt` recursively globs core, kernels, model, runtime, text, and media into one
  `ninfer_core` library, so ownership boundaries are not enforced by compilation;
- CLI, server, benchmark, tests, diagnostics, and active documentation still treat `.qus` as the
  only C++ Engine artifact.

### 3.3 Focused reading list

Implementation and review use the smallest authority set that covers this cutover:

- `docs/ninfer-project-positioning.md` for the exact-target, selected-GPU, single-request product
  purpose;
- `docs/ninfer-engine-architecture.md` for every target/Program/frontend/controller/lifetime/source
  boundary implemented here;
- `docs/ninfer-container-format.md`, `docs/ninfer-storage-layouts.md`, and
  `docs/ninfer-tensor-formats.md` for generic artifact bytes and descriptors;
- `docs/qwen3.6-27b-ninfer-artifact.md` for the complete target inventory, views, resources, and
  binding obligations;
- `docs/qwen3.6-27b-architecture.md` for Text/Vision/MTP mathematics and state semantics;
- `docs/design.md` and the current `include/ninfer/runtime/engine.h`, `src/runtime/engine.cpp`,
  `include/ninfer/model/`, `src/model/`, `src/text/`, and `src/serve/generation_service.cpp` for the
  implementation being replaced;
- `docs/serving.md` for current CLI/OpenAI/Anthropic observable behavior;
- `docs/kernel-development.md` and the affected operator headers/tests before moving shared or
  target-private CUDA code;
- `tools/reference/qwen3_6_27b_rtx5090/` and `tools/parity/qwen3_6_27b_rtx5090/` for the native
  correctness oracle and diagnostic comparisons.

The archived q5090 format and optimization reports may explain old choices, but they do not define
the new runtime. Local vLLM, llama.cpp, and SGLang comparisons were already distilled into the
accepted architecture and are not reopened unless implementation exposes a new concrete problem.

## 4. Scope

The atomic change includes all of the following:

1. make the existing C++ `.ninfer` reader a production artifact component;
2. add generic binder/materialization mechanisms without model execution semantics;
3. register exactly `qwen3.6-27b` on the actual RTX 5090 as the first compiled target package;
4. implement its complete storage profile, typed bindings, loaded resources, memory plans,
   Frontend, PreparedPrompt, OutputSession, Program, schedules, and target-private diagnostics;
5. move current Text, Vision, MTP, sampling, prefix, graph, and state behavior into that target;
6. implement the common generated-token controller and opaque public Engine boundary;
7. switch CLI, OpenAI/Anthropic server, benchmarks, and retained diagnostics to that route;
8. replace the monolithic build with explicit components and an explicit closed target registry;
9. migrate useful numerical/behavioral tests and delete tests that only protect `.qus`;
10. remove the current C++ `.qus` parser/loader, q5090 converter/tool route, model-card ownership,
    legacy public interfaces, aliases, and active dual-route documentation;
11. verify the complete product on the real `.ninfer` artifact and RTX 5090 before declaring the
    change complete.

## 5. Non-goals

- Do not implement Qwen3.6-35B-A3B, MoE kernels, another checkpoint, or another GPU.
- Do not implement continuous batching, request admission, preemption, paged multi-request KV,
  multi-GPU execution, offload, or distributed execution.
- Do not revise `.ninfer` framing, metadata, numeric formats, layouts, or the 27B artifact recipe
  unless implementation exposes a real contradiction; such a contradiction pauses the cutover for
  explicit discussion.
- Do not change the selected weight quantization, draft-head construction, persistent packing, or
  model mathematics.
- Do not redesign or optimize CUDA kernels merely because files move. A demonstrated regression or
  correctness defect may justify a focused fix inside the same target, not a general optimization
  campaign.
- Do not add a generic model graph, `IModel::forward`, family base class, plugin ABI, dynamic target
  discovery, runtime model configuration ingestion, or artifact-driven execution.
- Do not introduce public multi-target feature bags, checkpoint phase enums, generic GDN/MTP/MoE
  state, or speculative scheduler placeholders.
- Do not preserve project-owned C++ APIs, CLI/report field names, source paths, or `.qus` behavior
  for compatibility. The OpenAI and Anthropic protocol behavior currently advertised by the
  product remains a real external contract.
- Do not refactor the Python `.ninfer` reference except to correct an oracle defect discovered by
  C++ parity work.
- Do not add fixed hashes, clean-worktree gates, byte-identical conversion requirements, exact
  probabilistic-output comparison, attack models, fuzz matrices, atomic-publication machinery, or
  failure-injection suites.
- Do not expand tests for coverage, file layout, class names, getters, constructors, or transitional
  behavior.

## 6. Atomic final-state contract

The only accepted post-cutover product state is:

| Boundary | Final state |
|---|---|
| Artifact input | `.ninfer` only |
| Registered target | `qwen3_6_27b_rtx5090` only |
| Target selection | `(model_id, actual GPU, complete registered storage profile)` |
| Weight owner | immutable target `LoadedModel` over materialized `.ninfer` objects |
| Mutable sequence owner | one non-movable target `Program` |
| Frontend owner | immutable target `Frontend`; request-local prepared prompt and decoder |
| Generated-token policy | one common transactional controller |
| Runtime dispatch | one coarse closed-target dispatch around the whole generation loop |
| Product entries | CLI, server, and benchmark all use the same new Engine route |
| Legacy route | no executable `.qus` parser, loader, converter, Engine, or fallback |
| Source/build | explicit lower components, target package, registry, and product link |

The final tree must not contain an old Engine wrapped behind a new name. In particular:

- `LoadedModel` is not a renamed `WeightStore` with q5090 module semantics;
- `Program` is not a renamed `Qwen3_6_27B` card while Engine still owns its state;
- `Frontend` is not a one-shot tokenizer bundle transferred out of the weight store;
- the generation controller is not a wrapper around `TextGenerationRunner::run_tokens()`;
- target-private MTP/Vision/prefix calls are not exposed through common Engine methods;
- `.ninfer` offsets, JSON names, and layout tags leave the inference path after construction.

## 7. Final runtime and ownership flow

```text
Engine construction
  -> DeviceContext
  -> ArtifactReader parses generic directory
  -> registry selects qwen3_6_27b_rtx5090 from model_id + actual RTX 5090
  -> target plan_load consumes all 1172 objects through ArtifactBinder
  -> Materializer performs checked reads/copies into final backing
  -> target constructs heap-stable immutable LoadedModel and typed views
  -> target constructs immutable Frontend from owned resources
  -> target plans sequence capacity against remaining GPU memory
  -> RequestMemory and non-movable Program are constructed at stable addresses
  -> graph warmup/capture completes
  -> directory, name index, mmap/file ownership, and load staging are released

Request preparation, outside GPU execution critical section
  -> product layer acquires authorized owning media where applicable
  -> selected target Frontend renders/tokenizes/preprocesses
  -> opaque PreparedPrompt owns tokens, positions, patches/media data, prefix identity, cookie

One generation call
  -> reject a mismatched product cookie before target planning or sequence mutation
  -> Frontend creates an owning OutputSession from the still-live prompt + caller stop policy
  -> Program plans request without mutation
  -> RequestMemory grows before begin
  -> Program begin returns one PendingRound
  -> controller stages output and resolves the exact accepted token prefix
  -> Program and decoder commit the same prefix
  -> output is published only after both commits
  -> each later decode_round follows the same transaction
  -> finish leaves a coherent Resident prefix or explicit Invalid state
```

Destruction order is part of the implementation contract:

```text
Program -> RequestMemory -> Frontend/LoadedModel -> DeviceContext
```

Program-owned graph-stable addresses and target bindings must remain valid for their entire owner
lifetime. No typed view may refer to the reader JSON, name index, mmap descriptor, temporary load
plan, moved owner subobject, or staging buffer.

## 8. Interfaces that must land together

The architecture document remains authoritative for exact semantics. The cutover must realize the
following interfaces as one connected vertical slice.

### 8.1 Public product boundary

The installed `include/ninfer/` surface contains only opaque Engine/PreparedPrompt handles and
owning host request/result/configuration values. It contains no CUDA, artifact, tensor, kernel,
target, q5090, Qwen dimension, or model-state type.

The public flow must provide the semantic equivalents of:

```text
load one artifact on one selected device and requested context capacity
prepare an owning product input through the loaded target frontend
count/summarize a prepared prompt without model execution where required
generate from a matching prepared prompt with request options, cancellation, and output sink
read common load/memory/generation summaries and target diagnostics needed by product measurement
```

Exact method and leaf-header names are implementation details only if this ownership and flow remain
unambiguous. There is no public `prefill`, `decode_step`, `run_vision`, `target_verify`, cache
rewind, GDN slot, or one-token queue API.

Current product behaviors and operational controls remain functional through the cutover; they are
not promoted into target state or a generic model graph. The public owning host values are divided
once as follows:

| Value | Public owner and lifetime | Final handling |
|---|---|---|
| artifact path, device ordinal, maximum context | `EngineOptions`, construction-time | registry/target selection and target `plan_sequence` validate the request |
| prefill chunk, BF16/INT8 KV representation, CUDA Graph/eager choice | `EngineOptions`, construction-time | selected target translates them at `plan_sequence/create_program` into its private memory/graph plan |
| MTP draft window `0..5` and full/optimized proposal head | `EngineOptions::speculative`, construction-time | selected target translates them at `plan_sequence/create_program` into private weights/graphs; common runtime never branches on MTP |
| sampling, requested output tokens, prefix-reuse permission | common `ExecutionOptions`, request-time | target request plan validates/translates sampling and capacity; controller owns the output budget |
| caller stop policy, cancellation view, output sink | common `RequestOptions`/generation call, request-time | Frontend/GenerationController only; never Program execution state |
| target taps, forced schedule probes, internal counters | internal diagnostic/benchmark seams only | absent from the installed product API |

These are value semantics, not exposed CUDA or target types. An exact target may reject an option it
does not support; common runtime does not emulate it or infer a substitute. The current fields are
kept because CLI/server/benchmark already expose and use them, not as a promise that every future
target implements every choice. This task adds no new tuning mode and removes none of these existing
controls. `plan_load` is independent of them and always loads the registered complete product.
Graph-disabled eager execution remains the current diagnostic/fallback control, not a second quality
mode or separately optimized kernel profile.

Every prepared-prompt envelope carries a process-wide non-reused load-instance cookie. Cookie
allocation is safe across concurrent Engine construction, consumes a value even when construction
later fails, never uses an owner address as identity, and fails rather than wrapping. Generation
checks it before target planning or mutation.

### 8.2 Closed target package

The first package implements the accepted coarse contract:

```text
plan_load
construct_loaded_model
make_frontend
plan_sequence
create_program
```

Its exported `package.h` is the only composition include and is complete enough to instantiate the
registry and target-templated generation loop. Target implementation headers remain private.

The registry contains one explicit entry. It does not create empty 35B entries, family fallback,
directory scanning, static-constructor registration, or a plugin mechanism. The complete target is
selected once at load and visited once around the whole generation operation, never per layer or
per token.

Artifact `model_id` is a target-selection fact validated against the compiled package. Server
`--model-id` is only the protocol-facing advertised/matched alias for `/v1/models` and request
translation; it never selects a target and is never compared in place of the artifact identity.

### 8.3 Artifact construction boundary

The production artifact layer owns only generic mechanisms:

- v1 prefix/JSON/range parsing and registered format/layout/encoding values;
- temporary name indexing and checked object handles;
- exact layout-size and plane-geometry helpers;
- `ArtifactBinder` lookup, exact descriptor checks, checked view creation, consumption tracking;
- common allocation/read/copy plans and materialization;
- generic load progress and byte/memory accounting.

The 27B package owns the complete expected inventory, resource meanings, logical views, tied roles,
consumer descriptors, and actual-GPU validation. It constructs those expectations from compiled
target constants and canonical object-name patterns; it does not hard-code file offsets or object
array order and does not import converter Python code.

Materialization copies persistent bytes directly into their final registered representation. In
particular, `row-split-k128-v1` is consumed as its low/high/FP16-scale planes without dense decode,
re-quantization, or persistent repacking. One artifact object may form several target-checked views,
but artifact payload ranges never overlap.

### 8.4 Loaded model and frontend resources

`LoadedModel` owns all materialized device weights, retained host resources, and immutable typed
descriptors. It is constructed at its final heap address before its views are formed.

The six required frontend resources remain owned by the loaded product. The 27B frontend constructs
or validates its tokenizer, chat-template behavior, generation defaults, and image/video processing
rules from those resources and compiled exact-checkpoint semantics. It does not reopen the artifact,
read the BF16 checkpoint, receive a one-shot `Q5090TokenizerBundle`, or depend on Python at runtime.

The target frontend accepts only authorized owning media values. URL/file authorization and
acquisition remain in the product/serve layer; checkpoint-specific resizing, patch construction,
placeholder expansion, token typing, Vision position construction, MRoPE, and `rope_delta` remain
target-private.

### 8.5 Program and generated-token transaction

The target `Program` implements the accepted coarse operations:

```text
plan_request
begin
decode_round
finish_active
abort_active
clear_resident
state
sequence_summary
```

It is the sole owner of mutable sequence state: Text and MTP KV, GDN state/snapshots, position
state, logical ledger, prefix checkpoints, sampler state, stable token output, graph inputs/outputs,
workspaces, CUDA Graphs, and target counters.

The first token and every later ordinary/MTP result use the same move-only `PendingRound` contract.
Each round is resolved exactly once by `commit_all`, `commit_prefix_and_finish`, or `discard`. There
is no hidden `pending_sampled_` queue, arbitrary `commit(m)` followed by continuation, or claimed
rollback of state that was never snapshotted.

The common controller always orders a returned batch as follows:

1. stage decoder, UTF-8/channel state, stop candidates, and token-to-byte boundaries without
   mutation or publication;
2. choose the exact accepted logical-token prefix from stop token/string and the already effective
   output budget supplied by the target request plan;
3. finish all fallible decoder-commit preparation;
4. resolve Program with that same prefix;
5. commit decoder state without allocation or failure;
6. charge the same token count to the budget;
7. publish owning output deltas.

Context capacity is not recomputed at round resolution. `RequestPlan` has already converted the
requested limit into `effective_output_tokens` and an `OutputLimit`/`ContextCapacity` reason; the
controller consumes that result. Equality uses `OutputLimit`, as defined by the architecture.
`requested_output_tokens == 0` performs no model work, and a nonzero request for which the target
approves zero effective capacity does not call `begin`.

`RequestPlan` is an owning, self-contained target value. It borrows neither `PreparedPrompt`,
`RequestMemory`, nor mutable Program storage. Before moving the plan and prompt into `begin`, the
controller copies only the bounded common plan summary and grows RequestMemory from that summary.

Cancellation uses its separate terminal path rather than pretending that zero or some arbitrary
prefix of a returned round was normally committed. Cancellation before `begin` performs no model
work. If cancellation is observed after a round has executed but before it is resolved, the
controller stages a terminal decoder flush, prepares its commit, discards the pending round,
commits zero new tokens, publishes only the previously committed tail, and leaves the sequence
Invalid. Cancellation between resolved rounds calls `finish_active`, commits/publishes the terminal
decoder flush, and may retain an exact Resident prefix.

If synchronous publication throws after Program and decoder commit, the batch is neither retried
nor rolled back. Generation reports failure and its scope guard makes the resulting sequence
unavailable for prefix reuse, even if Program briefly held a coherent Resident state. Bytes already
accepted by the sink are not retractable.

CLI and server call this one controller. They do not contain separate generated-token loops.

### 8.6 State and memory invariants

The implementation must preserve the accepted `E`/`S` frontier model:

```text
Active:   S = E + 1
Resident: E <= S <= E + 1
```

Only target-licensed tokens leave a round. A returned MTP suffix is not automatically equivalent to
a fully materialized KV/recurrent suffix. Stop at any returned position must either canonicalize an
exact reusable prefix or mark the resident sequence Invalid.

Every target-owned region is defined once by one checked layout builder and then bound from the same
layout handles. The implementation distinguishes loaded immutable, sequence persistent, graph
stable, request-active, and request-transient lifetimes. RequestMemory grows before `begin`; no
decode/MTP round performs heap/device allocation, artifact lookup, file I/O, or runtime repacking.

Every reset, successful begin, round resolution, finish, and invalidation advances the sequence
epoch. Request plans and round handles reject stale use before mutation. Unresolved-round destruction
performs host invalidation only and launches no CUDA work.

`plan_request` is read-only on success or failure. Before a recoverable `begin` failure returns,
every asynchronous use of the moved prompt and transient region has completed or been safely
cancelled; no partial new resident identity is published. A recoverable `decode_round` failure after
mutation leaves Program Invalid. Device faults remain process-fatal under the current CUDA policy
rather than being reported as a successfully recoverable Program reset.

## 9. Complete 27B behavior to migrate

This is not a Text-only architecture demonstration. The single cutover is incomplete unless the
following current target behavior passes through the new route.

### 9.1 Text and state

- all 64 layers: 48 GDN and 16 GQA;
- current embedding, norms, attention/GDN, MLP, and full output-head schedules;
- native Q4/Q5/Q6/W8 packed consumers and current fused row views;
- chunked prefill and ordinary decode;
- BF16 and INT8 KV operation currently exposed by the product;
- FP32 recurrent state and convolution history;
- greedy and sampled generation with temperature, top-k, top-p, presence/frequency penalties, and
  seed behavior;
- exact distinction among 248320 stored output-head rows, 248077 tokenizer-addressable IDs, and the
  valid sample domain; ordinary sampling and every MTP proposal/correction/bonus must stay inside
  the decodable domain;
- configured context, effective output capacity, and near-context-tail behavior;
- eager/diagnostic execution where currently required and CUDA Graph steady-state execution.

### 9.2 Vision and multimodal Text

- the complete 27-layer Vision tower and merger;
- image, video, multiple items, and mixed image/video input;
- current media limits and observable input errors;
- patch construction, independent media attention segments, position interpolation, Vision RoPE,
  merger order, placeholder expansion, and embedding scatter;
- token types, three-axis MRoPE, and `rope_delta`;
- release of media/patch/transient Vision storage only after all asynchronous Vision/Text/MTP reads
  are complete;
- no token-only prefix reuse for a multimodal prompt whose media/composed identity differs.

### 9.3 MTP

- draft windows `k=1..5` where currently exposed;
- full-head and shortlisted draft-head proposal paths;
- greedy draft proposal, sequential target verification, accept/reject, rejection correction, and
  all-accepted bonus;
- target sampling distribution and occurrence-state treatment in greedy and non-greedy modes;
- fixed Program-owned licensed-token return storage;
- ordinary fallback near capacity or when a round cannot honor the controller budget;
- stop/output limit at every returned position;
- consistent Text KV, GDN snapshot/slot, MTP KV/carry/proposal, sampler, graph scalar, position, and
  logical-ledger state after Continue; terminal partial finish must make every suffix-only state
  unreachable, but may retire sampler/next-proposal state or return Invalid instead of repairing a
  state that could continue;
- multimodal shifted MTP input from the composed Vision embedding rather than a placeholder token
  embedding lookup.

The `.ninfer` Python reference is the correctness oracle for the last item and for model-level
storage/forward parity. The current fused C++ kernels remain the performance implementation.

### 9.4 Prefix reuse

- fresh full reset;
- exact append continuation;
- assistant-content-boundary checkpoint restore;
- coherent KV, recurrent, MTP, position, frontier, and logical-ledger restoration;
- target-owned prompt/media identity rather than token-only matching;
- full reset when exact reuse cannot be proved;
- one atomic target plan/restore operation, not an external mutable boundary setter followed by a
  separate prefill.

### 9.5 Product behavior

- raw prompt and structured text/image/video messages;
- current tokenizer, chat template, thinking prompt, and generation defaults;
- incremental UTF-8 decoding and reasoning/content channels;
- stop tokens, cross-round stop strings, output limit, context limit, and cancellation;
- CLI generation and metrics;
- OpenAI Chat Completions and Anthropic Messages, including streaming and count-tokens behavior;
- current best-effort tool-call rendering/parsing outside the target decoder;
- single-user one-active-request execution and current preparation/engine locking behavior.

## 10. One implementation unit with three coordinated workstreams

These are workstreams inside one atomic change, not separately accepted phases. None may be merged,
documented as current, or used to claim completion without the other two.

### Workstream A: exact-target construction and execution

1. Promote `src/artifact/reader.*` and `storage_layouts.cpp` into an explicit production artifact
   library.
2. Add generic binder, checked layout-builder, materialization-plan, materialized-backing, resource,
   and load-stat mechanisms.
3. Add the explicit target list, one closed active-target alternative, cold registry selection, and
   whole-loop dispatch composition.
4. Implement `qwen3_6_27b_rtx5090` load planning against the complete 1172-object inventory,
   including direct tensors, quantized plane views, fused rows, tied objects, GDN convolution view,
   draft ID map, Vision objects, and frontend resources.
5. Materialize directly into final loaded backing, construct `LoadedModel` at its final address, and
   form typed bindings only afterward.
6. Define target-private sequence and request layouts from one checked source of size/offset truth.
7. Move KV/GDN/MTP/sampling/graph/prefix ownership from Engine into one non-movable Program.
8. Move fixed Text, Vision, and MTP schedules into Program-private target implementation files while
   reusing existing operator and CUDA implementations.
9. Implement begin/round/finish resolution, frontier normalization, prefix restoration, and target
   diagnostics without exposing target phases to runtime.

### Workstream B: frontend, controller, and product entries

1. Move Qwen tokenizer/template/media-position behavior into the target Frontend and construct it
   from loaded native resources.
2. Split media acquisition/authorization from checkpoint-specific decode/preprocessing so target
   code receives owning authorized input and never links network policy.
3. Replace public `ProcessedInput`, tokenizer transfer, and model-card types with an opaque prepared-
   prompt envelope carrying a target alternative, product cookie, and common summary.
4. Implement request-local OutputSession/decoder staging and exact-prefix commit for UTF-8,
   reasoning/content, stop tokens, and cross-round stop strings.
5. Implement the common generation budget, cancellation policy, `PendingRound` resolver,
   publication order, and result summary once.
6. Replace the current model-shaped Engine with PIMPL ownership of DeviceContext and the closed
   ActiveTarget.
7. Switch CLI, server preparation/run/count-tokens, and streaming to the public prepare/generate
   route; translate wire requests into public owning input/`RequestOptions`, map target planning
   errors and finish reasons back to protocol results, and derive request logs only from public
   request/result summaries. Keep protocol/tool translation outside the target and delete every
   serve-side context-capacity formula.
8. Rebuild `ninfer_bench` over the public Engine so artifact selection, Frontend, controller, and
   Program are the same route used by CLI/server. Add only internal timing observation around that
   route; do not implement a second generation loop or restore `prefill()`/`decode_step()` publicly.
   Keep a separately named target-verify benchmark only for proposal/verification micro-measurement.
9. Migrate retained preprocess/Vision/parity diagnostics to the target/native route or delete them
   if they no longer serve a current numerical or product need.

### Workstream C: physical replacement, proof, and deletion

1. Replace recursive source globs with explicit component and target source lists.
2. Move every installed implementation header—core, artifact, kernel, model, runtime, text, media,
   and serve—out of `include/ninfer/`; retain only the owning product API there.
3. Organize shared kernels vertically by mathematical operator and keep target-only fixed-shape
   helpers under the target package; do not alter kernel algorithms merely to perform the move.
4. Make lower components and the target package compile independently with only their declared
   include/link dependencies; make registry composition the only boundary that sees target exports.
5. Migrate useful artifact, binder, kernel, target-state, frontend, controller, CLI/report, and
   protocol tests to the new owners.
6. Record the legacy `.qus` performance/memory baseline before deleting the route; measure the final
   `.ninfer` product independently after the cutover.
7. Delete every legacy source/API/tool/test/document listed in Section 13 as part of the same final
   tree.
8. Update all active documentation from the temporary dual-route description to the one native
   product route, then form the complete candidate for the review and single final gate in
   Section 16.

### Coordination points

Coordination is limited to these closed seams; none is a temporary compatibility adapter:

| Seam | Producer | Consumer | Fixed fact |
|---|---|---|---|
| materialized object handle/view | artifact mechanisms | target load/bind | byte geometry only; no model role in common code |
| sequence plan and Program construction | target package | Engine-owned RequestMemory/TargetInstance | common capacity summary plus opaque target owners; Program is installed last at a stable address |
| owning media value | product acquisition | target Frontend | owned bytes and source metadata needed for decoding; no deferred path/URL access in target code |
| prepared prompt envelope | target frontend/Engine preparation | target Program visit | owning value + exact load-instance cookie |
| OutputSession | target Frontend | common GenerationController | owning decoder/stop/channel transaction created inside generation before prompt move |
| pending licensed-token round | target Program | common controller | exact token span and one resolution action |
| common summaries/diagnostics | target/runtime | CLI/server/bench | observable values; no mutable state handle |

No other temporary cross-layer adapter is permitted. In particular, there is no adapter from
`.ninfer` objects to the old q5090 module catalog and no adapter from new Program rounds to the old
one-token runner.

## 11. Current-to-final ownership ledger

| Current implementation | Final owner/action |
|---|---|
| `include/ninfer/runtime/engine.h`, `src/runtime/engine.cpp` | replace with opaque public Engine plus runtime engine/registry ownership; move every checkpoint fact to target |
| `Engine::prefill`, `prefill_cached`, `decode_step`, `generate` | delete public seam; Program + common controller own the complete request |
| Engine cache/workspace formulas and allocation order | target `impl/program/` checked layouts |
| Engine logical token mirror, boundary snapshot, GDN slot and graph flags | target Program state/prefix/graph owners |
| Engine MTP proposal/verify/round repair | target `impl/schedule/mtp.cpp` and Program round transaction |
| `include/ninfer/model/config.h` | target `impl/checkpoint.h` |
| `FullLayerW`, `GdnLayerW`, `MtpW`, bind methods | target `impl/load/` typed immutable bindings |
| `Qwen3_6_27B`, `Qwen3_6_Vision` long-lived cards | remove; schedules become private Program definitions |
| `StepState`, fixed MTP/GQA/Vision helpers | target Program/state/kernels unless a proven mathematical operator is already shared |
| production `FileTap`/testing members | target parity/diagnostic seam, never testing macros in production types |
| q5090 `WeightStore`/parser/module catalog | generic artifact mechanisms plus target load plan; old code deleted |
| `Q5090TokenizerBundle` transfer | loaded resources retained by target Frontend |
| `model::Processor`, Qwen chat/tokenizer files | target frontend; neutral Unicode/media decode remains below |
| media URL/path/data acquisition mixed with processing | product media-acquisition layer, then owning target input |
| `TextGenerationRunner` generated-token loop | common GenerationController + target OutputSession |
| serving `StopSequences`/duplicate output control | common controller/output decoder; protocol/tool parsing remains serve-owned |
| serve `translate`, `request_log`, `serve_options`, and `GenerationService` old Qwen/kernel/runner values | private serve translation to public owning input/`EngineOptions`/`RequestOptions`, and logging from common summaries |
| serve `resolve_context_output_budget()` and `test_generation_budget.cpp` | remove old `max_context - prompt + 1` derivation; target `RequestPlan` owns capacity, controller owns budget, serve only maps public errors/reasons |
| all public implementation headers, including core/kernel/model/runtime/text/media/serve | scoped internal component/target/serve headers; installed include tree becomes product-only |
| wrapper/launcher/kernel global buckets | vertical shared operators or target-private kernels according to actual ownership |
| monolithic `ninfer_core` recursive glob | explicit core/artifact/kernels/text/media/runtime/target/registry/product targets |
| current CLI/server main files | thin product adapters over public Engine API |
| `ninfer_bench` direct `prefill`/`decode_step` loop | public Engine/controller route with internal timing observations; no second generation loop |
| `target_verify_bench` | retained target-private proposal/verification microbenchmark with native target vocabulary |
| `tools/bench/run_ninfer_bench_matrix.py` | migrate executable path, CLI options, schema/report fields, and ordinary/MTP case matrix together |

Moving a file is not sufficient. The old owner must cease to express the invariant after the move.

## 12. Verification strategy: one final gate

Verification is organized by risk, not by the implementation workstreams. There is one final gate
after all product entries have switched and all legacy code has been removed.

### 12.1 Permanent tests to retain or replace

Retain or migrate only tests that protect current observable risks:

- the small `.ninfer` fixture with all seven numeric formats and a resource;
- Python writer/reader and Python-written/C++-read agreement;
- direct words and row-split plane/code/scale geometry, including one format-neutral independent
  C++ pack/golden helper reused by real Q4/Q5/Q6/W8 kernel tests and checked against the native
  artifact layout/quantization implementation;
- 27B inventory, complete binding, representative fused/logical views, GDN convolution view, and
  draft ID map;
- existing independent CUDA operator/kernel numerical tests at real shapes;
- target Text block and Vision/MTP model-level parity through an internal diagnostic seam;
- Program/frontier behavior for ordinary, accepted, rejected, all-accepted, partial-stop, capacity-
  fallback, prefix-restore, and invalidation paths;
- controller exact-prefix behavior for stop token/string, UTF-8/channel staging, output limit, and
  cancellation using a small deterministic fake Program/decoder where GPU execution is irrelevant;
- one focused synchronous publish-failure case proving that committed output is not retried or
  rolled back and that the sequence is invalidated for later prefix reuse;
- real `.ninfer` load/bind/reader-release/second-construction behavior and the prefix/lifetime
  scenarios that cannot be established through an external product request;
- CLI/report and OpenAI/Anthropic schema/translation behavior.

Do not preserve old tests by wrapping old types. A q5090 parser/store test is deleted; a kernel test
whose oracle happens to use old q5090 naming is renamed/reduced to the native row-split mathematical
contract if it still protects real numerical risk.

Architecture is enforced by actual component compilation and link dependencies, not tests that scan
source text for class names, include paths, call order, or files.

### 12.2 Real artifact and target construction

The final route must demonstrate on the real artifact:

- generic directory parse and exact target/device selection;
- complete consumption of all 1172 objects;
- expected shapes, formats, layouts, resources, views, aliases, and consumer descriptors;
- direct final materialization without dense decode/repack or a second persistent weight copy;
- immutable bindings that remain valid after Reader/Binder/LoadPlan destruction;
- correct destruction order and a clean second Engine construction in the same process;
- no artifact JSON/name lookup, BF16-checkpoint access, or artifact-file access after activation;
  ordinary product media acquisition remains an explicit higher-layer operation.

### 12.3 Capability evidence ledger

Run one deliberately small but complete matrix. Each row has one primary proof; a real run is not
repeated merely to reproduce a branch that a deterministic Program/controller test proves better.

| Area | Deterministic/reference evidence | Real RTX 5090/product evidence |
|---|---|---|
| Text | native Python-reference versus C++ layer activation comparison for a fixed token-ID prefill and first decode transition | one greedy chat-template request and one normally sampled request with explicit sampling parameters |
| KV | target state/layout tests for both representations | one BF16 request and one INT8 request |
| MTP transaction | deterministic Program tests for reject, all-accepted bonus, every accepted-prefix/stop position, capacity fallback, terminal partial-state retirement, and full/draft-head dispatch | greedy `k=1` full-head and sampled primary `k=3` draft-head requests execute proposal/verification and report valid round/proposal/accepted statistics; no particular random accept/reject outcome is required |
| Vision | native Python-reference versus C++ preprocessing/Vision/Text activation comparison on a fixed multimodal message | one image, one video, and one mixed/multiple-media request |
| Multimodal MTP | composed shifted-embedding parity through the target diagnostic seam | at least one mixed-media request with MTP enabled and valid proposal/verification statistics |
| Prefix | deterministic target tests for fresh, exact append, assistant-boundary restore, mismatch reset, multimodal non-reuse, and invalidation after cancellation/publication failure | one real multi-turn exact-append request reports reuse and one mismatched request reports reset |
| Stops and limits | deterministic controller/Program tests for stop token, cross-round stop string, UTF-8/channel staging, output-limit equality, context tail, cancellation, and publish failure | ordinary CLI termination plus server streaming termination completes without duplicate/missing publication |
| Product protocols | schema/translation tests | CLI; server health/model listing/count-tokens; OpenAI non-streaming/streaming; Anthropic request |
| Measurement | report-schema tests | fixed ordinary/MTP benchmark reports, load/memory summary, Vision timings, and fixed long-context INT8 run |

The Python `.ninfer` reference supplies the model, preprocessing, and representative activation
oracles. Comparisons use named phases/tensors and documented tolerances; they do not require exact
final token equality, bit-identical floating activations, or byte-identical artifacts. The sampled
request proves that the sampling route operates with explicit parameters; its probabilistic token
stream is not an oracle.

### 12.4 Performance and memory acceptance

Before implementation removes the old route, record a local `.qus` baseline from the current
binary. After the final tree is complete, run the same workloads through the `.ninfer` binary. The
final binary never contains both routes.

Use the same RTX 5090, prompt/corpus, context, prefill chunk, KV representation, MTP setting, graph
setting, generation length, and warmup. Timed workloads use a three-run median; the fixed
long-context capacity smoke runs once on each route.

Only values both routes already expose form paired regression evidence:

- Text prefill at 512 and 2048 tokens;
- ordinary fixed-length decode;
- MTP decode with the primary draft window/head, comparing engine/output throughput, acceptance,
  rounds, and fallback counts;
- image and video total multimodal prefill latency: old monolithic prefill versus the sum of new
  Vision and Text-prefill timings;
- weight H2D/resident bytes, KV payload bytes, and workspace peak;
- successful completion of the fixed 32K-prompt/128-token INT8-KV run under a 65536-token configured
  context.

The final route additionally records load time, file-read bytes, host staging, peak allocated/
reserved GPU memory, split preprocessing/Vision/Text timings, and the target planner's approved
capacity with free GPU memory recorded immediately before construction. Old CLI logs may contain a
subset with different boundaries. Unless both sides expose the same semantic boundary, these values
remain descriptive diagnostics rather than falsely paired measurements.

A stable regression greater than roughly 5% in any primary latency/throughput metric triggers a
same-configuration rerun and investigation. A stable unexplained regression greater than roughly
10% blocks completion unless the user explicitly accepts the trade-off. Paired memory is reviewed
in the harmful direction only: increased weight resident/H2D, KV payload, or workspace-peak bytes,
or failure of the fixed long-context run. Unpaired planner/peak/staging values are sanity-reviewed
but have no invented old-side percentage. These are human review thresholds, not CI tests
and not a demand to discover an absolute maximum context. NSYS is used only when end-to-end timing
exposes a regression or launch gap; NCU is used only after a specific kernel becomes the
demonstrated cause.

The most likely migration regressions are hot-path name lookup, duplicate payload copies, decoded
weights left resident, changed streaming/head behavior, staging synchronization, unstable graph
addresses, or a lifetime change that prevents workspace reuse. Kernel microbenchmarks alone cannot
clear the final product gate.

## 13. Required legacy removal

The final tree removes the old route rather than deprecating it.

### 13.1 Legacy paths absent from the final tree

The following old paths are removed after their still-useful behavior has moved to the owner named
in Sections 10 and 11. They are not left uncompiled as architectural fossils:

- `include/ninfer/core/weight_store.h`
- `include/ninfer/core/weight_store_parser.h`
- `src/core/weight_store.cpp`
- `src/core/weight_store_parser.cpp`
- `include/ninfer/model/` and `src/model/` after load/frontend/Program/schedule/kernel ownership has
  moved into the exact target;
- `include/ninfer/runtime/engine.h` and `src/runtime/engine.cpp`, replaced by
  `include/ninfer/engine.h` plus the runtime Engine PIMPL;
- installed `include/ninfer/core/` and `include/ninfer/kernels/` implementation surfaces after their
  headers move to scoped internal component include roots;
- installed `include/ninfer/text/`; Qwen tokenizer/template/CLI/runner code moves to target,
  app, or runtime generation, while only neutral Unicode primitives remain under internal
  `src/text/`;
- `include/ninfer/media/source.h`; path/URL acquisition becomes product-private and the target sees
  an owning public media value containing acquired bytes rather than deferred path/URL access;
- installed `include/ninfer/serve/`; schemas, translation, logging, options, service, and transport
  interfaces remain private to `src/serve/`;
- `src/main.cpp` and `src/serve/main.cpp`, replaced by `apps/cli/` and `apps/serve/` entry points;
- the old placement of `include/ninfer/runtime/decode_graph.h` and
  `src/runtime/decode_graph.cpp`; generic CUDA Graph RAII moves to internal core/runtime primitives,
  while the target Program owns every graph instance;
- `tools/q5090/`
- `tools/q5090_convert/`
- `tests/fixtures/make_q5090_fixture.py`
- `tests/test_q5090_parser.cpp`
- `tests/test_weight_store.cpp`
- `tests/test_weight_store_real.cpp`
- `.qus`-only structural/snapshot tests and fixtures
- active `docs/q5090_packed_file_format_v4.md` because the historical copy already exists under
  `docs/archive/optimization-era/`

`tests/test_q5090_pack_golden.cpp` and `tests/kernels/q5090_pack.h` are not retained under old names,
but their independent mathematical function is retained: migrate them to a format-neutral
`row-split-k128-v1` test helper/golden, switch the Python side from the old q5090 converter to
`tools.artifact.layouts`/`tools.convert.common`, and keep the helper as the input builder used by
Q4/Q5/Q6/W8 kernel tests.

### 13.2 Remove old interfaces and fields

- `Engine::q5090_load_stats()` and every `Q5090*` public type;
- `Engine` ownership of `WeightStore`, model cards, target state, and target graphs;
- `Q5090TokenizerBundle` and one-shot resource transfer;
- `Tensor`/`Weight` fields whose only meaning is q5090 layout metadata, replacing them only with the
  exact native consumer descriptor needed by kernels;
- q5090 `ModuleKind`, `SourceKind`, fusion/catalog identities, and any other model role that leaked
  into core types;
- benchmark/report fields such as `q5090_h2d_bytes`, `q5090_resident_bytes`, and
  `q5090_resident_modules`, with no aliases;
- CLI/server `.qus` usage and old-format error/help text;
- old public `prefill`/`decode_step` compatibility wrappers;
- old model-card test-only entry points and `NINFER_MODEL_TESTING` production-shape changes.

### 13.3 Migrate according to value

- the useful parts of `tests/test_engine_real_file.cpp` and `test_model_bind.cpp` become the native
  real-artifact load/bind/reader-release/second-construction and prefix/lifetime tests;
- `test_engine_mtp_e2e.cpp` and `test_engine_vision_e2e.cpp` do not survive as a duplicate second
  product matrix: their numerical/state obligations move into deterministic Program tests, the
  Python/C++ target parity runner, and the real CLI matrix; `test_engine_memory_stats.cpp` becomes
  report-contract coverage plus the benchmark evidence rather than another full inference run;
- tokenizer/template/processor and text-runner tests move to target frontend/controller behavior;
- `test_model_blocks`, `test_vision_support`, and model-level numerical cases move under the target;
  `test_model_config` retains only a fact not already covered by binder/schedule tests, otherwise it
  is deleted; production `FileTap` and its old runtime test are replaced by the manifest-based
  diagnostic tool rather than retained in production types;
- useful CUDA kernel tests remain, with internal headers and native format vocabulary;
- `tools/parity/qwen3_6_27b_rtx5090/` remains the target-private diagnostic home;
- old `tools/parity/block_dump.cpp`, `layer_dump.cpp`, `q5090_structural_dump.h`,
  `block_parity.py`, and `hf_reference.py` are either replaced by a direct native target/reference
  seam or deleted;
- q5090 preprocess/Vision diagnostics move to target/frontend-native tools only if the resulting
  binaries still protect a current product or numerical need; otherwise remove the binaries and
  their documentation together.
- `tools/bench/run_ninfer_bench_matrix.py`, `bench/target_verify_bench.cpp`, CLI/benchmark support,
  report-schema tests, `include/ninfer/text/tokenizer.h`, `src/main.cpp`, and
  `src/serve/serve_options.cpp` are explicit downstream consumers; migrate their paths, vocabulary,
  fields, and assumptions rather than relying on a final search to discover them.
- `tools/parity/qwen3_6_27b_rtx5090/preprocess.py` drops its `.qus --engine-weights` interface and
  invokes the native target diagnostic; `activations.py` retains only its manifest-based comparison
  and removes the legacy flat-FileTap fallback.

Local historical `.qus` files under ignored `out/` and archived profiler reports are not runtime
compatibility and need not be deleted by source migration. No active command, test, or code path may
depend on them after cutover.

### 13.4 Active documentation update

At minimum update:

- `AGENTS.md`
- `README.md`
- `docs/README.md`
- `docs/design.md`
- `docs/ninfer-naming.md`
- `docs/ninfer-engine-architecture.md`
- `docs/kernel-development.md`
- `docs/serving.md`
- `docs/qwen3.6-27b-architecture.md`
- `docs/qwen3.6-27b-ninfer-artifact.md`
- `bench/README.md`
- `tests/README.md`
- relevant tool/component README files
- `eval/README.md`, active evaluation plans/configuration, and `tools/bench` defaults that contain
  current product launch or artifact paths

The final active documentation describes `.ninfer` as the only product artifact, the target-package
Engine architecture as implemented, and the Python reference as the correctness oracle. It no
longer describes a current/pending dual route. Historical archive content is not rewritten merely
because it mentions QUS, q5090, or `.qus`.

## 14. Main implementation risks

| Risk | Required control |
|---|---|
| native row-split planes do not match current kernel descriptors | one authoritative layout helper plus independent code/scale numerical oracle; no runtime repack |
| wrong 1172-object role/view binding | compiled complete binder, consumption tracking, representative source/reference parity |
| view points into a moved/temporary owner | final-address LoadedModel construction and explicit post-construction lifetime review |
| duplicate 17 GiB weight backing | materialization accounting and real load memory/H2D evidence |
| frontend behavior changes with six native resources | text/template/token-ID and image/video preparation parity against native Python reference/current product behavior |
| Program/controller disagree on an MTP token prefix | deterministic PendingRound exact-prefix/stop-position tests plus ordinary real MTP execution |
| prefix reuse restores KV but not GDN/MTP/position state | target-owned epoch/ledger/checkpoint tests and multi-turn real execution |
| CUDA Graph captures movable/transient addresses | one target memory plan, stable Program storage, capture before activation |
| server retains a second token/stop loop | CLI/server both call one controller; protocol layer owns only translation/publication/tool parsing |
| build still permits cross-layer includes | explicit component compilation and scoped include roots, not source-scanning tests |
| temporary old wrapper survives cutover | required deletion ledger and final active-source review |
| only ordinary Text is validated | complete capability matrix is part of the single completion gate |

## 15. Canonical evidence and final-gate commands

This plan fixes the final executable locations rather than leaving them to interpretation. The new
CMake graph emits:

```text
build-cutover/apps/ninfer
build-cutover/apps/ninfer-serve
build-cutover/bench/ninfer_bench
build-cutover/tools/ninfer-qwen3_6_27b-dump
```

The last executable is a target-private diagnostic, not a product entry. It emits only structured
`ninfer_activation_dump_v1` manifests and bounded summaries needed by the native Python-reference
comparators.

Set the common environment once. The three messages files are real local acceptance inputs selected
when executing the plan; they must respectively contain an image, a video, and multiple mixed
image/video items. They need not become repository fixtures.

```bash
cd /home/neroued/ninfer
ROOT=$PWD
PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
MODEL=/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16
NINFER_WEIGHTS=$ROOT/out/qwen3_6_27b_rtx5090.ninfer
LEGACY_WEIGHTS=$ROOT/out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus
CORPUS=$ROOT/bench/fixtures/bench_corpus.ids
OLD_BUILD=$ROOT/build
BUILD=$ROOT/build-cutover
OLD_CLI=$OLD_BUILD/src/ninfer
OLD_BENCH=$OLD_BUILD/bench/ninfer_bench
CLI=$BUILD/apps/ninfer
SERVER=$BUILD/apps/ninfer-serve
BENCH=$BUILD/bench/ninfer_bench
DUMP=$BUILD/tools/ninfer-qwen3_6_27b-dump
SERVER_MODEL=qwen3.6-27b
TEXT_IDS='248045,846,198,5834,248046,198'
IMAGE_MESSAGES=/absolute/path/to/image-messages.json
VIDEO_MESSAGES=/absolute/path/to/video-messages.json
MIXED_MESSAGES=/absolute/path/to/mixed-media-messages.json
```

### 15.1 Pre-cutover baseline evidence

This is the one read-only use of the legacy product. Capture it before deleting that route:

```bash
mkdir -p profiles/bench/engine-cutover/legacy
test -x "$OLD_CLI"
test -x "$OLD_BENCH"
test -f "$LEGACY_WEIGHTS"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv \
  > profiles/bench/engine-cutover/legacy/gpu-before.txt

"$OLD_BENCH" \
  --weights "$LEGACY_WEIGHTS" --corpus "$CORPUS" \
  --max-ctx 4096 --prefill-chunk 1024 --kv-dtype bf16 \
  -p 512,2048 -n 128 -pg 2048,128 -r 3 --warmup 1 \
  --output json \
  --output-file profiles/bench/engine-cutover/legacy/ordinary.json

"$OLD_BENCH" \
  --weights "$LEGACY_WEIGHTS" --corpus "$CORPUS" \
  --max-ctx 4096 --prefill-chunk 1024 --kv-dtype bf16 \
  -p 512,2048 -n 128 -pg 2048,128 -r 3 --warmup 1 \
  --mtp-draft-tokens 3 --lm-head-draft \
  --output json \
  --output-file profiles/bench/engine-cutover/legacy/mtp-k3-draft.json

nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv \
  > profiles/bench/engine-cutover/legacy/gpu-before-long-int8.txt
"$OLD_BENCH" \
  --weights "$LEGACY_WEIGHTS" --corpus "$CORPUS" \
  --max-ctx 65536 --prefill-chunk 1024 --kv-dtype int8 \
  -pg 32768,128 -r 1 --warmup 0 \
  --output json \
  --output-file profiles/bench/engine-cutover/legacy/long-int8.json

for run in 1 2 3; do
  "$OLD_CLI" "$LEGACY_WEIGHTS" --messages "$IMAGE_MESSAGES" \
    --max-context 4096 --prefill-chunk 1024 --kv-dtype bf16 \
    --max-new 8 --greedy \
    > "profiles/bench/engine-cutover/legacy/image-$run.out" \
    2> "profiles/bench/engine-cutover/legacy/image-$run.log"
  "$OLD_CLI" "$LEGACY_WEIGHTS" --messages "$VIDEO_MESSAGES" \
    --max-context 4096 --prefill-chunk 1024 --kv-dtype bf16 \
    --max-new 8 --greedy \
    > "profiles/bench/engine-cutover/legacy/video-$run.out" \
    2> "profiles/bench/engine-cutover/legacy/video-$run.log"
done
```

In all benchmark commands, absence of `--no-cuda-graph` deliberately means the graph-enabled route;
the JSON report must record that choice and the actual decode path. Record the command text,
repository revision and dirty summary, GPU/CUDA/toolchain, free memory, and emitted settings as
descriptive provenance. Do not require a clean worktree, artifact hash, or byte-reproducible result.

### 15.2 Fresh final build and permanent tests

The old `build/` directory is baseline-only. The one final candidate is configured into a path that
does not already exist, so deleted CMake targets cannot survive as stale binaries:

```bash
test ! -e "$BUILD"
cmake -S . -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build "$BUILD" -j

NINFER_QWEN3_6_27B_WEIGHTS="$NINFER_WEIGHTS" \
  ctest --test-dir "$BUILD" --output-on-failure

"$PYTHON" -m pytest -q tests/artifact tests/targets/qwen3_6_27b
"$PYTHON" -m py_compile \
  tools/parity/qwen3_6_27b_rtx5090/vision_mtp.py \
  tools/smoke/serve_contract.py
"$PYTHON" -m tools.artifact.inspect "$NINFER_WEIGHTS" --objects
test -x "$CLI"
test -x "$SERVER"
test -x "$BENCH"
test -x "$DUMP"
```

`ctest` includes the deterministic controller/Program branch matrix, protocol/report schemas, and
only the real-artifact cases whose internal ownership cannot be proved by product runs: complete
load/bind, reader release, second construction, prefix reuse/reset, and relevant lifetime state. It
does not repeat the full Text/Vision/MTP product matrix already owned by Sections 15.3–15.5. The
real-artifact cases must execute, not skip, when the environment variable above is present.

The converter, layout contract, and artifact bytes are non-goals, and the existing artifact already
passed its source-checkpoint verifier. Therefore the full BF16 verifier is not part of the normal
gate. Run it only if the submitted cutover actually changes converter code, a storage/layout
contract, or artifact payload construction:

```bash
"$PYTHON" -m tools.convert.qwen3_6_27b_rtx5090.verify \
  "$NINFER_WEIGHTS" --model "$MODEL"
```

### 15.3 Native C++ versus Python-reference parity

The final diagnostic accepts the same fixed token/message inputs and emits the same manifest format
as the Python reference. Both sides receive an explicit empty stop-ID set, so `--decode 2` must
execute the first decode transition rather than terminating on the prefill token:

```bash
mkdir -p profiles/parity/engine-cutover/text

"$PYTHON" -m tools.reference.qwen3_6_27b_rtx5090.cli \
  --weights "$NINFER_WEIGHTS" --ids "$TEXT_IDS" \
  --decode 2 --greedy --prefill-chunk 1024 --kv-dtype bf16 \
  --stop-ids '' \
  --activation-dump profiles/parity/engine-cutover/text/python \
  --dump-level layer

"$DUMP" --weights "$NINFER_WEIGHTS" --ids "$TEXT_IDS" \
  --decode 2 --greedy --prefill-chunk 1024 --kv-dtype bf16 \
  --stop-ids '' \
  --activation-dump profiles/parity/engine-cutover/text/cpp \
  --dump-level layer

"$PYTHON" -m tools.parity.qwen3_6_27b_rtx5090.activations \
  profiles/parity/engine-cutover/text/python \
  profiles/parity/engine-cutover/text/cpp \
  --reference-phase prefill --candidate-phase prefill \
  --json profiles/parity/engine-cutover/text/prefill.json

"$PYTHON" -m tools.parity.qwen3_6_27b_rtx5090.activations \
  profiles/parity/engine-cutover/text/python \
  profiles/parity/engine-cutover/text/cpp \
  --reference-phase decode --candidate-phase decode \
  --json profiles/parity/engine-cutover/text/decode.json
```

The manifest comparator becomes a real gate, not a report-only script. Missing tensors, unexpected
tensors, shape/size mismatch, non-finite values, absence of a requested phase, or a metric outside
the target parity package's reviewed per-tap tolerance table produces a nonzero exit. That table is
checked into the target parity package and documented in its README before the final candidate; it
uses fixed named/patterned tensor rules and is not fitted to the candidate output. Reports retain
max-absolute, RMS/relative-RMS, and cosine values for diagnosis.

Migrate the existing target parity tools into one sequential `vision_mtp` runner. It first releases
the Python model before starting the C++ diagnostic, then checks exact frontend IDs/types/positions/
`rope_delta`, representative Vision taps, the composed Text embedding, and a fixed `k=1` full-head
MTP schedule probe:

```bash
mkdir -p profiles/parity/engine-cutover/vision-mtp
"$PYTHON" -m tools.parity.qwen3_6_27b_rtx5090.vision_mtp \
  --weights "$NINFER_WEIGHTS" --cpp "$DUMP" \
  --messages "$MIXED_MESSAGES" \
  --prefill-chunk 1024 --kv-dtype bf16 \
  --mtp-draft-tokens 1 --proposal-head full \
  --work-dir profiles/parity/engine-cutover/vision-mtp \
  --output profiles/parity/engine-cutover/vision-mtp/report.json
```

This runner is a target-private numerical diagnostic, not another inference frontend. Its report
uses the same fail-on-missing/shape/non-finite/tolerance semantics. It does not compare final sampled
text or require the BF16 source checkpoint.

### 15.4 Real product acceptance

Exercise every current execution choice through the final CLI. All commands deliberately use the
graph route:

```bash
COMMON=(--max-context 4096 --prefill-chunk 1024)

"$CLI" "$NINFER_WEIGHTS" "${COMMON[@]}" --kv-dtype bf16 \
  --prompt "用三句话解释 prefill 和 decode 的区别。" \
  --max-new 64 --greedy

"$CLI" "$NINFER_WEIGHTS" "${COMMON[@]}" --kv-dtype bf16 \
  --prompt "写一个简短的 CUDA kernel 优化建议。" --max-new 64 \
  --temperature 0.6 --top-p 0.95 --top-k 20 \
  --presence-penalty 1.0 --frequency-penalty 0.0 --seed 42

"$CLI" "$NINFER_WEIGHTS" "${COMMON[@]}" --kv-dtype int8 \
  --prompt "解释 KV cache。" --max-new 32 --greedy

"$CLI" "$NINFER_WEIGHTS" "${COMMON[@]}" --kv-dtype bf16 \
  --prompt "列出三个数字。" --max-new 32 --greedy \
  --mtp-draft-tokens 1

"$CLI" "$NINFER_WEIGHTS" "${COMMON[@]}" --kv-dtype bf16 \
  --prompt "列出三个颜色。" --max-new 64 \
  --mtp-draft-tokens 3 --lm-head-draft \
  --temperature 0.6 --top-p 0.95 --top-k 20 \
  --presence-penalty 1.0 --frequency-penalty 0.0 --seed 43

"$CLI" "$NINFER_WEIGHTS" "${COMMON[@]}" --kv-dtype bf16 \
  --messages "$IMAGE_MESSAGES" --max-new 32 --greedy

"$CLI" "$NINFER_WEIGHTS" "${COMMON[@]}" --kv-dtype bf16 \
  --messages "$VIDEO_MESSAGES" --max-new 32 --greedy

"$CLI" "$NINFER_WEIGHTS" "${COMMON[@]}" --kv-dtype bf16 \
  --messages "$MIXED_MESSAGES" --max-new 32 --greedy \
  --mtp-draft-tokens 3 --lm-head-draft
```

The MTP runs need only show that proposal and verification executed and that tokens/statistics are
valid. Deterministic tests, not prompt hunting, prove reject/all-accepted/partial-stop branches.

Use one lightweight standard-library acceptance client for the HTTP contract. It waits for health,
checks model listing and count-tokens, sends OpenAI non-streaming and SSE-streaming requests
(including a stop string), sends one OpenAI request containing an embedded local data-URI image,
and sends an Anthropic request. It validates framing, usage/counts, finish reasons, media translation
through the target frontend, and absence of duplicate streamed bytes; it is not a load/concurrency
test and does not fetch a remote asset.

```bash
mkdir -p profiles/serve/engine-cutover
"$SERVER" "$NINFER_WEIGHTS" \
  --host 127.0.0.1 --port 18080 --model-id "$SERVER_MODEL" \
  --max-context 4096 --prefill-chunk 1024 --kv-dtype bf16 \
  --mtp-draft-tokens 3 --lm-head-draft --default-max-tokens 32 --greedy \
  > profiles/serve/engine-cutover/server.log 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

SMOKE_STATUS=0
"$PYTHON" -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 --model "$SERVER_MODEL" \
  || SMOKE_STATUS=$?

if kill -0 "$SERVER_PID" 2>/dev/null; then
  kill "$SERVER_PID"
else
  SMOKE_STATUS=1
fi
wait "$SERVER_PID" || true
trap - EXIT
test "$SMOKE_STATUS" -eq 0
```

### 15.5 Final performance and memory evidence

Run the exact benchmark semantics used for the baseline through the final public Engine route:

```bash
mkdir -p profiles/bench/engine-cutover/native
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv \
  > profiles/bench/engine-cutover/native/gpu-before.txt

"$BENCH" \
  --weights "$NINFER_WEIGHTS" --corpus "$CORPUS" \
  --max-ctx 4096 --prefill-chunk 1024 --kv-dtype bf16 \
  -p 512,2048 -n 128 -pg 2048,128 -r 3 --warmup 1 \
  --output json \
  --output-file profiles/bench/engine-cutover/native/ordinary.json

"$BENCH" \
  --weights "$NINFER_WEIGHTS" --corpus "$CORPUS" \
  --max-ctx 4096 --prefill-chunk 1024 --kv-dtype bf16 \
  -p 512,2048 -n 128 -pg 2048,128 -r 3 --warmup 1 \
  --mtp-draft-tokens 3 --lm-head-draft \
  --output json \
  --output-file profiles/bench/engine-cutover/native/mtp-k3-draft.json

nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv \
  > profiles/bench/engine-cutover/native/gpu-before-long-int8.txt
"$BENCH" \
  --weights "$NINFER_WEIGHTS" --corpus "$CORPUS" \
  --max-ctx 65536 --prefill-chunk 1024 --kv-dtype int8 \
  -pg 32768,128 -r 1 --warmup 0 \
  --output json \
  --output-file profiles/bench/engine-cutover/native/long-int8.json

for run in 1 2 3; do
  "$CLI" "$NINFER_WEIGHTS" --messages "$IMAGE_MESSAGES" \
    --max-context 4096 --prefill-chunk 1024 --kv-dtype bf16 \
    --max-new 8 --greedy \
    > "profiles/bench/engine-cutover/native/image-$run.out" \
    2> "profiles/bench/engine-cutover/native/image-$run.log"
  "$CLI" "$NINFER_WEIGHTS" --messages "$VIDEO_MESSAGES" \
    --max-context 4096 --prefill-chunk 1024 --kv-dtype bf16 \
    --max-new 8 --greedy \
    > "profiles/bench/engine-cutover/native/video-$run.out" \
    2> "profiles/bench/engine-cutover/native/video-$run.log"
done
```

The final CLI timing summary separates load, preprocessing, Vision, Text prefill, and decode. For
the paired image/video metric, compare the old CLI's monolithic multimodal prefill with the sum of
the new Vision and Text-prefill timings; retain the new split only as descriptive evidence. The
ordinary/MTP JSON keeps the old/new common fields—graph mode, fixed settings, H2D/resident weights,
KV payload, workspace peak, output/engine throughput, and MTP rounds/fallbacks/acceptance—under
native names so they can be compared. New load/read/staging/GPU-peak/planner fields are recorded but
not assigned an invented old value. The fixed long-context comparison is success at the same
configured capacity; only the new report additionally records its materialized/planner-approved
capacity. Apply Section 12.4 to common metrics only and do not turn the thresholds into a permanent
benchmark test.

### 15.6 Cleanup review and evidence coverage

By this point the plan has already moved to the archive as required by Section 16, so it cannot
create false cleanup matches:

```bash
git diff --check
rg -n 'q5090|Q5090|\.qus|tools/q5090' \
  --glob '!docs/archive/**' --glob '!profiles/**' --glob '!out/**' \
  --glob '!build*/**' \
  --glob '!tools/freq_corpus/fixtures/ranking/*.manifest.json' .
```

The `rg` command is a one-time review aid, not a permanent source-scanning test. Every remaining
active match must be an intentional historical/provenance note; no executable route, compatibility
alias, current command, public field, or normative active `.qus` contract may remain. The excluded
frequency-corpus manifests retain source-data provenance and are not runtime compatibility. Review
all changed active Markdown links and authority/status claims in the same pass; do not add a
source-scanning unit test for documentation shape.

The final gate has no uncovered capability claim:

| Required evidence | Gate source |
|---|---|
| artifact formats, complete 1172-object bind, layouts, row-split oracle | full CTest/Python suites and inspect in 15.2 |
| Program reject/all-accepted/partial stop/capacity/cancel/publish failure and prefix variants | deterministic and real-artifact CTests in 15.2 |
| Text block/activation parity | explicit Python/C++ comparisons in 15.3 |
| frontend, Vision, composed multimodal MTP parity | sequential target parity runner in 15.3 |
| greedy, sampled, BF16, INT8, full-head MTP, draft-head MTP, image/video/mixed media | CLI matrix in 15.4 |
| health/models/count-tokens/OpenAI stream and non-stream/Anthropic | server acceptance client in 15.4 plus schema CTests |
| product-path prefill/decode/MTP, total multimodal prefill, common memory fields, fixed long context | paired baseline/native evidence in 15.1 and 15.5 |
| unpaired load/staging/GPU-peak and planner-capacity diagnostics | final-route descriptive evidence in 15.5 |
| no legacy route and active docs agree | final diff, cleanup review, and independent review |

## 16. One final-candidate review and gate

The completion sequence below is one candidate-state protocol, not a set of delivery phases:

1. finish the entire implementation, legacy deletion, active documentation, and benchmark/parity
   tooling in one candidate diff;
2. set this plan's status to `final candidate; gate pending at archival`—which records the exact
   historical moment without claiming completion—move it to the archive path in Section 18, and
   update the archive index as part of that same candidate;
3. independently review the complete candidate from four perspectives:
   - artifact/binding: every object, plane, view, resource, owner, and materialization lifetime;
   - state/transaction: Program state machine, `E`/`S`, MTP partial finish, prefix reuse,
     decoder/model exact-prefix agreement, cancellation, publication failure, and destruction;
   - architecture/build: public opacity, option ownership, target ownership, dependency direction,
     explicit source lists, and absence of wrappers/false abstractions;
   - product/performance: Text/Vision/MTP/serve/bench completeness, native artifact only, memory,
     and stable end-to-end results;
4. resolve every finding and re-review the affected portions until the candidate diff is closed;
5. starting from an absent `build-cutover/`, run Sections 15.2 through 15.6 once as the single final
   gate and compare with the already-recorded Section 15.1 baseline;
6. make no code, build, test, plan, or active-document change after that successful gate.

Any required change after step 5 invalidates that candidate: review the changed portion and rerun
the gate from a fresh final build directory. There is no partial approval for loader, target,
Program, frontend, serving, or deletion work.

## 17. Completion criteria

This atomic task is complete only when all of the following are true in the same submitted final
state:

- the real `.ninfer` artifact constructs the only C++ Engine product route;
- the closed registry selects the exact 27B/RTX 5090 package and no other target is implied;
- all 1172 objects are completely bound and materialized without runtime repack or duplicate
  persistent weights;
- immutable loaded resources, target Frontend, PreparedPrompt, Program, RequestMemory, graphs, and
  destruction order satisfy the accepted lifetime contract;
- Text, Vision, MTP, sampling, context-tail behavior, and prefix reuse execute through the target
  Program;
- first token and later rounds use the same transactional controller and exact-prefix output
  resolution;
- CLI, OpenAI, Anthropic, streaming, count-tokens, benchmark, and retained diagnostics use the same
  new product route;
- current external protocol behavior and complete native model capabilities remain available;
- the final public headers expose no model, target, CUDA, tensor, kernel, artifact, or q5090 type;
- component and target build boundaries are explicit and independently compile;
- `.qus` parser/loader/converter/tools/tests/public fields/current commands are deleted with no
  fallback or compatibility path;
- necessary artifact, numerical, state, controller, product, and real-artifact tests pass;
- real RTX 5090 Text/image/video/mixed/MTP/prefix/serve acceptance passes;
- stable decode, prefill, Vision, load, memory, and context results are reviewed under Section 12.4;
- active documents describe one implemented `.ninfer` target-package Engine rather than a dual
  route;
- changed active Markdown links and authority/status claims are valid;
- the submitted diff is limited to this cutover, any unrelated user changes are preserved, and
  `git diff --check` passes.

The task is not complete merely because a `.ninfer` loader opens the file, a target directory
exists, Text greedy output looks plausible, or the old route still supplies missing product
features.

## 18. Plan lifecycle

While proposed/approved and implementation is active, this file remains under `docs/plans/`. When
the complete implementation and active documentation first form the final candidate—but before its
independent review and single final gate—set the historical status to `final candidate; gate pending
at archival` and move it to:

```text
docs/archive/ninfer-foundation/2026-07-14-ninfer-engine-atomic-cutover.md
```

Update the archive index in the same candidate. Stable behavior must already have been distilled
into the active architecture, system, serving, artifact, benchmark, and test documents before the
move. A successful gate does not rewrite that accurate archival-moment status; completion is the
accepted submitted state and its evidence, not a mutable claim inside an archived plan. If the plan
is abandoned instead, archive it with that status. The archived plan remains implementation
history, not a second source of current runtime truth.
