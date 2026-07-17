# NInfer System Design

This document describes the implemented C++ product boundary and repository ownership. Model
mathematics are defined by the model architecture documents. Persistent numeric formats, layouts,
container framing, and the Qwen3.6-27B object inventory are defined by their dedicated artifact
documents.

The Qwen3.6 family boundary recorded here is the accepted organization for registering the planned
35B-A3B target. It does not claim that the second C++ target has already landed.

## 1. Product scope

NInfer is a high-performance local inference engine for a small set of compiled exact
checkpoint/GPU targets. The currently registered product is:

```text
checkpoint: Qwen3.6-27B
device:     NVIDIA RTX 5090 (sm_120a)
target key: qwen3_6_27b_rtx5090
artifact:   qwen3_6_27b_rtx5090.ninfer
execution:  one resident sequence, one active request
```

The target implements Text, image/video Vision, MTP, BF16/INT8 KV, sampling, prefix reuse, eager
decode, and CUDA Graph decode. Another checkpoint or GPU becomes supported only through another
explicit target package and registry entry.

Continuous batching, request preemption, multi-GPU execution, offload, and distributed serving are
outside the current implementation.

## 2. System boundary

The product route is one vertical path:

```text
source BF16 checkpoint
  -> target-specific offline converter
  -> .ninfer artifact
  -> generic reader / binder / materializer
  -> selected compiled target package
       immutable LoadedModel
       shared immutable Qwen3.6 Frontend
       reusable request memory
       one mutable non-movable Program
  -> common generated-token controller
  -> opaque public Engine
       CLI / server / benchmark
```

There is no runtime interpretation of a model graph and no family-level execution fallback. The
artifact describes persistent objects; compiled target code defines their semantic binding and all
execution behavior.

## 3. Components and ownership

| Component | Source | Ownership |
|---|---|---|
| Base mechanisms | `src/core`, selected `src/runtime` primitives | device/stream, tensor views, checked layouts, arenas, graph wrapper, physical KV containers, raw transfers, public host-type definitions, request memory |
| Artifact | `src/artifact` | `.ninfer` framing and directory parsing, object matching, range/geometry checks, binding records, direct final materialization |
| Ops | `include/ninfer/ops`, `src/ops` | repository-internal semantic contracts and every wrapper/launcher/kernel implementation, including fused and exact-shape/device-specialized paths |
| Neutral text/media decode | `src/text`, `src/media/decode` | Unicode primitives and image/video decoding over owning bytes |
| Product media acquisition | `src/product/media_acquire` | local-path, HTTP(S), and data-URI acquisition into owning media values |
| Product prompt input | `src/product/prompt_input` | shared JSON/message parsing into owning public prompt values for product tools |
| Qwen3.6 family leaves | `src/targets/qwen3_6` | one tokenizer/template/media/MRoPE frontend, owning prepared prompt and output session, and passive common Vision geometry/binding definitions; no executable model graph or target identity |
| Exact target | `src/targets/qwen3_6_27b_rtx5090` and, when registered, `src/targets/qwen3_6_35b_a3b_rtx5090` | registered storage signature, exact load bindings, sequence/request plans, Program state, fixed Text/Vision/MTP schedules, Op/fusion choices, graph/lifecycle policy, target diagnostics |
| Registry | `src/targets/registry.*` | closed target selection and complete target construction |
| Product runtime | `src/runtime` | generated-token budget, round resolution, cancellation, publication, public Engine PIMPL, target lifetime |
| Serving | `src/serve` | OpenAI/Anthropic schemas, translation, streaming, usage, request logs, and HTTP transport |
| Apps | `apps/cli`, `apps/serve` | command parsing, product input acquisition, and executable entry points |
| Offline tools | `tools/artifact`, `tools/convert`, `tools/reference`, `tools/parity` | artifact tooling, target conversion, independent Python reference, and numerical diagnostics |

The central placement rule is simple:

- generic byte/device/lifetime mechanisms belong in core/artifact;
- every host-callable, semantically closed tensor or explicit local-state transformation is an Op,
  and its complete implementation belongs in the central Op layer even when only one target,
  numerical shape, or GPU currently uses it;
- tokenizer/template/output semantics, multimodal prompt construction, owning prepared values, and
  passive Vision definitions that are identical for the two Qwen3.6 checkpoints belong in the
  Qwen3.6 family leaves;
- exact checkpoint binding, fixed layer order, operand/view selection, Vision composition, MTP
  orchestration, Op selection and fusion, prefix repair, state lifetime, workspace, and graph
  policy belong to the exact target;
- stop/output-budget/cancellation/publication policy belongs to common runtime;
- schemas, URLs/files, protocol translation, and transport belong to product/serve code.

The complete Op boundary and implementation rules are defined by
[`op-development.md`](op-development.md). The family layer shares stable semantics and passive
types only. Duplication of target schedules and state policy is intentional.

### 3.1 Qwen3.6 family boundary

The 27B and 35B-A3B packages use the same `qwen3_6::Frontend`, `PreparedPrompt`, and
`OutputSession`. Both artifacts embed the same six frontend resources. The family frontend owns
tokenizer/chat-template interpretation, image/video preprocessing, placeholder expansion, MRoPE
construction, output decoding, and default-stop interpretation. The common Vision portion owns
shared preprocessing, backbone geometry, and immutable binding vocabulary; the exact target still
binds its target-width merger and decides how Vision is scheduled and fused with the rest of the
request.

This is compile-time code reuse, not family-level target selection. `src/targets/qwen3_6` has no
`Program`, layer loop, execution graph, CUDA Graph, mutable sequence state, target registry entry,
or fallback path. Both exact packages call their own Text/Vision/MTP schedules and may call different
Ops or differently fused Ops even where the high-level mathematics are related.

## 4. Public C++ API

The public product surface consists of:

```text
include/ninfer/engine.h
include/ninfer/types.h
```

`Engine` is a PIMPL owner. Public request/configuration/result types are host-owning values. The API
does not expose CUDA objects, tensor views, artifact descriptors, target classes, or mutable
sequence state.

The main operations are:

```cpp
Engine(EngineOptions);
PreparedPrompt prepare(PromptInput) const;
PreparedPrompt prepare_tokens(std::vector<TokenId>, bool) const;
std::uint32_t count_tokens(PromptInput) const;
GenerationResult generate(PreparedPrompt, RequestOptions, OutputSink*, const CancellationView&);
LoadSummary load_summary() const;
MemorySummary memory_summary() const;
```

`PreparedPrompt` is opaque and move-only. For the supported and planned Qwen3.6 routes it contains
one owning `qwen3_6::PreparedPrompt`; it has no exact-target alternative, target tag, provenance
field, or mismatch check. Preparation may run outside the GPU execution critical section.
`generate` serializes access to the single Program.

## 5. Load and target construction

Engine construction performs the complete load before publishing a usable object:

1. create the selected `DeviceContext` and observe the actual GPU;
2. parse the `.ninfer` prefix and embedded object directory;
3. select the compiled package by `model_id` and let it preflight the actual GPU and options;
4. let the selected target consume every required tensor and let the Qwen3.6 family binder consume
   the shared frontend/common-Vision resources through `artifact::Binder`, validating the complete
   registered storage signature;
5. materialize tensors directly into their final backing and retain required host resources;
6. construct heap-stable immutable `LoadedModel` bindings;
7. construct the shared Qwen3.6 Frontend from retained resources;
8. plan sequence capacity and request work from the configured context/KV/graph/MTP options;
9. construct reusable request memory and the non-movable Program at stable addresses;
10. finish target initialization and release reader directory/name/staging state.

The current target requires the exact registered model inventory and RTX 5090. Selection is cold
construction logic; the generation loop contains no model-name or layout-string dispatch.

`LoadSummary` reports the selected target, load/upload time, file and H2D bytes, staging peak, and
object counts.

## 6. Lifetime model

The long-lived product objects have one ownership order:

```text
DeviceContext
  -> LoadedModel and Qwen3.6 Frontend
  -> RequestMemory
  -> Program
```

Destruction is the reverse: Program, request memory, loaded resources, then device context. This
keeps graph inputs, target bindings, and stream-owned work valid until their final user is gone.

Memory is divided by lifetime:

| Lifetime | Owner | Examples |
|---|---|---|
| loaded immutable | `LoadedModel` | materialized weights, lookup tables, frontend resources |
| sequence persistent | `Program` | Text/MTP KV, GDN state, token ledger, prefix checkpoint |
| graph stable | `Program` | graph inputs/outputs, scalar controls, host mirrors |
| request active | `Program` | sampling counters/RNG and active-request controls |
| request transient | `RequestMemory` | Vision and other request-planned scratch |

`MemorySummary` exposes weights, sequence, workspace, KV payload, configured capacity, and storage
mode without exposing internal allocators.

## 7. Qwen3.6 family Frontend

The Qwen3.6 family Frontend owns the input and output semantics shared by both exact checkpoints:

- tokenizer and chat-template resources embedded in the artifact;
- message/tool rendering and thinking controls;
- image/video placeholder expansion, patch construction, token types, and three-axis positions;
- opaque owning family `PreparedPrompt` values;
- output token decoding, UTF-8 buffering, reasoning/content channel state, and model default stops.

Apps and serving acquire media into owning bytes. The family Frontend receives those bytes and
performs Qwen3.6 preprocessing. It has no HTTP or filesystem policy.

`Engine::prepare` returns a public opaque envelope containing the family prepared value, summary,
and preparation time. The value contains no exact-target identity. `Engine::prepare_tokens`
supplies the equivalent raw-token route
for parity and repeatable benchmarks. `count_tokens` uses the same Frontend rendering and
preprocessing rules without executing the model.

## 8. Program

The target `Program` is the sole mutable owner of sequence execution. It contains:

- Text and MTP KV caches;
- GDN recurrent/convolution state and slots;
- logical token ledger and execution frontier;
- sampling configuration, occurrence counters, and RNG state;
- prefix checkpoint and restoration state;
- prefill/decode/MTP/Vision workspaces;
- stable buffers and captured CUDA Graphs.

The common runtime sees only coarse target operations:

```text
plan_request(prepared prompt, execution options)
begin(prepared prompt, plan, transient region) -> BeginResult + first GeneratedRound
decode_round(round budget)                     -> GeneratedRound
resolve_pending(accepted count, terminal)
finish_active / abort_request
```

The fixed Text, Vision, and MTP functions under `impl/schedule` are private pieces of Program
execution. They do not introduce another long-lived sequence owner.

The target-private Program lifecycle moves among Empty, Resident, Active, Pending, and Invalid. At
most one generated round is unresolved. `GeneratedRound` is only a synchronous span over
Program-owned stable token storage; the pending lifecycle remains in Program rather than in an
owning RAII handle. Planning is read-only; allocation/growth happens before begin; model execution
reuses Program-owned addresses.

## 9. Generated-token resolution

The common controller owns the only product generation loop. The begin token and every later
ordinary or MTP round use the same resolution sequence:

1. Program returns a `GeneratedRound`, a synchronous view over target-licensed tokens while its
   target-private state remains Pending;
2. the target `OutputSession` previews decoding into reusable request-local scratch and returns one
   `OutputDecision` containing the exact accepted prefix and finish reason;
3. Program resolves that same accepted count through `resolve_pending`;
4. `OutputSession::commit_preview()` commits the selected decoder state, and the budget charges the
   same count;
5. output deltas are published only after those internal commits;
6. `GenerationGuard` calls `abort_request()` on any escaping exception so failed provisional or
   publication state cannot be reused.

This keeps KV/GDN/MTP state, logical tokens, decoded bytes, usage accounting, and streamed output on
one accepted-token boundary. The output decoder scans for one real outcome rather than constructing
a candidate lattice, and the round path has no separate staged/commit-plan transaction PIMPL. This
does not make a blanket claim that all string/vector work in output decoding is allocation free. CLI
and server do not implement their own token loops.

`GenerationResult` reports generated IDs, content/reasoning text, finish reason, reused prompt
tokens, phase timings, and speculative statistics.

## 10. Prefix reuse and context

Program owns prefix eligibility and restoration. Text prompts may reuse a resident execution
frontier or a target checkpoint boundary when identity, tokens, state, and MTP preparation agree.
Multimodal prompts currently take the fresh route. Callers may disable reuse through
`ExecutionOptions`.

The target request plan determines effective output capacity from the prepared prompt and Program
state. CLI/server code does not duplicate target context formulas. An Engine rejects a prepared
prompt that already exceeds its configured capacity; an output request may finish with
`ContextCapacity` when the target plan shortens it.

## 11. Text, Vision, and MTP execution

The target schedules preserve the model architecture document:

- text prefill is chunked at the configured multiple-of-128 chunk size;
- multimodal preparation runs Vision, merges visual tokens, injects embeddings, and then executes
  the composed text sequence;
- ordinary decode advances one licensed token;
- MTP prepares a proposal at the active frontier, verifies up to the selected draft window, and
  commits only the controller-approved prefix;
- the optimized proposal head is selected with `ProposalHead::Optimized` / `--lm-head-draft`;
- BF16 and INT8 group-64 KV are target-planned storage choices;
- CUDA Graph capture/replay is target-private and uses Program-lifetime stable addresses.

Detailed layer equations and tensor dimensions remain in
[`qwen3.6-27b-architecture.md`](qwen3.6-27b-architecture.md).

## 12. Product entry points

All product entry points use the public Engine:

- `apps/cli` translates command-line text/messages/media into `PromptInput`, prepares, generates,
  and prints deltas/summaries;
- `apps/serve` and `src/serve` translate OpenAI/Anthropic requests, prepare/count/generate, and map
  public summaries into protocol responses;
- `build/bench/ninfer_bench`, implemented under `bench/targets/qwen3_6_27b_rtx5090/`, uses
  `prepare_tokens` plus `generate` and reports public load, memory, timing, and speculative values.

The target-private `ninfer-qwen3_6_27b-dump` diagnostic links the target package directly for
bounded activation manifests. It is not a public Engine method.

## 13. Build graph

CMake uses explicit source lists and component targets:

```text
ninfer_core
ninfer_artifact
  -> ninfer_core
ninfer_ops
  -> ninfer_core
ninfer_text
ninfer_media_decode
ninfer_media_acquire
ninfer_product_prompt_input
  -> ninfer_media_acquire
ninfer_qwen3_6_family
  -> ninfer_artifact + ninfer_text + ninfer_media_decode
ninfer_qwen3_6_27b_rtx5090
  -> ninfer_qwen3_6_family + ninfer_artifact + ninfer_ops
ninfer_qwen3_6_35b_a3b_rtx5090                  # when registered
  -> ninfer_qwen3_6_family + ninfer_artifact + ninfer_ops
ninfer_engine
  -> explicit target packages + common runtime
ninfer_serve
  -> ninfer_engine + ninfer_media_acquire + protocol/transport
apps / benchmark / diagnostic
```

The registry is the composition boundary that sees exact target packages. The family library is an
ordinary inward dependency and cannot register or select a target. Lower components do not discover
target directories or import target semantics. Adding a target requires an explicit CMake entry and
explicit closed-registry entry.

## 14. Verification boundary

Permanent checks are organized by observable risk:

- `.ninfer` framing, numeric formats, layouts, resources, binding, and real target inventory;
- Op numerical/state-transition behavior at real supported shapes;
- family Frontend behavior, target Program state/prefix transactions, Text/Vision/MTP parity, and a real
  artifact smoke path;
- OpenAI/Anthropic schema and tool-call behavior;
- benchmark CLI/report contracts and real performance evidence.

Performance acceptance uses the product benchmark and profiler evidence for affected kernels. Unit
tests do not define throughput requirements or duplicate the generation loop.

## 15. Current limits

- one compiled checkpoint/GPU target;
- one active request and no continuous batching;
- one GPU;
- no runtime model graph, dynamic target discovery, or plugin ABI;
- no family-level execution selection or checkpoint fallback;
- no general offload or distributed execution.

These limits are product boundaries, not placeholders in the public API.
