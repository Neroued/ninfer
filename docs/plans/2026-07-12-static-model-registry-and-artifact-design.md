# Static Model Registry And Model-Neutral Artifact Design And Implementation Plan

> Status: proposed on 2026-07-12; implementation has not started.
>
> This document defines how the project will move from one fixed Qwen3.6-27B deployment to a finite
> registry of explicitly compiled model profiles while preserving its one-RTX-5090, one-user,
> static-schedule specialization. Qwen3.6-35B-A3B is the first added profile; future profiles may
> belong to a different architecture family. Until the corresponding implementation phase lands,
> current active documents
> remain authoritative: [`../design.md`](../design.md) and
> [`../qwen3.6-27b-architecture.md`](../qwen3.6-27b-architecture.md) describe the implemented model
> scope until the static-registry/Qwen35 work is complete, while
> [`../q5090_packed_file_format_v4.md`](../q5090_packed_file_format_v4.md) remains the artifact
> authority only until the atomic v5 cutover in Phase 2.

## 1. Decision

The project will support a finite, compiled registry of exact model profiles, including profiles
from different architecture families, rather than becoming a configuration-driven general-purpose
runtime.

The first delivered registry will contain:

- the existing dense Qwen3.6-27B profile;
- Qwen3.6-35B-A3B, a sparse MoE profile whose local BF16 source is
  `/home/neroued/models/llm/qwen/Qwen3.6-35B-A3B/base-hf-bf16`.

A future change may register a non-Qwen architecture such as Gemma 4, but this plan does not claim
Gemma execution support. Gemma is used here only as an architecture-neutrality probe for the packed
artifact boundary.

An artifact will identify one exact execution profile, checkpoint profile, packing target, and
packing profile. The Engine will
select one concrete implementation after CPU-side artifact parsing and schema validation. Inside
that implementation, model dimensions, weight structures, layer order, and performance-critical
dispatch will remain compile-time or otherwise fixed at load time. There will be no per-layer
virtual dispatch, name lookup, filesystem access, runtime graph construction, or weight repacking on
the inference hot path.

This changes the project's proposition from:

> one engine for one Qwen3.6-27B checkpoint shape

to:

> one RTX-5090-specialized engine for an explicit set of qualified, compiled model profiles

It does not change the single-user, single-active-sequence, single-GPU deployment target.

Only the boundaries that must survive a new architecture are generalized. The q5090 container,
storage blocks, tensor views, assets, and profile binding become model-neutral. Model schedules,
state machines, resource planners, numerical contracts, and CUDA specialization remain handwritten
per registered profile or per deliberately shared architecture implementation.

### 1.1 Purpose and expected value

This elevation changes the unit of support. Today the checkpoint, model schedule, artifact format,
runtime memory formulas, tools, and product identity are effectively one indivisible 27B object.
That maximizes initial optimization speed, but a second architecture would otherwise force either a
forked engine or a gradual spread of model-name conditionals and duplicated constants.

The static-registry design creates one deliberate extension seam at each real boundary:

- a model-neutral container ABI defines safe bytes, storage blocks, views, assets, and integrity;
- an exact compiled model profile defines mathematics, tensor semantics, and state geometry;
- an exact packing profile defines how that model is represented for the RTX 5090;
- one load-time Engine selection preserves static, separately tunable hot paths;
- shared L0/L1 facilities are reused only where their contracts are truly model-independent;
- frontends and evidence always derive identity from the loaded artifact.

Qwen3.6-35B-A3B is a useful forcing function because it changes hidden width, layer count, GQA/GDN
ratios, FFN topology, MTP topology, and Vision output width at once. Supporting it correctly proves
that the engine's intra-family boundaries are real rather than a renamed collection of 27B
constants. A structurally different model such as Gemma 4 tests a second property: adding a compiled
profile that reuses existing storage primitives must not require another container major revision or
new Qwen-specific enums in L0.

The intended long-term value is not a model zoo. It is the ability to qualify another explicit model
without redesigning container safety, storage ownership, asset handling, reporting, and serving
identity each time, while preserving profile-specific execution and CUDA performance.

## 2. Goal

Establish a model-neutral q5090 v5 artifact ABI and a finite compiled model registry, then add
Qwen3.6-35B-A3B as the first new Text, MTP, Vision, CLI, OpenAI, Anthropic, conversion, reference,
and benchmark target while preserving the existing 27B behavior and optimization model.

The completed system must:

- identify the loaded model from the artifact instead of a caller-selected model-type flag;
- reject every artifact that does not exactly match a compiled execution/checkpoint/packing tuple;
- parse the generic structure of an unknown model profile safely, but reject semantic loading before
  CUDA initialization;
- keep component, tensor, and asset identities profile-scoped rather than extending a global
  Qwen-specific source-role enum for every model;
- allow a newly compiled profile that uses existing codecs/layouts to reuse v5 without changing the
  core header, catalog records, structural parser, or WeightStore storage descriptors;
- retain straight-line, shape-specialized Text/MTP/Vision execution inside each profile;
- express dense and MoE FFNs as two explicit L2 model structures;
- store all routed experts in a direct-consumption q5090 layout with no runtime repack;
- execute routing and expert selection entirely on device, including CUDA Graph replay;
- provide dedicated decode, MTP/small-token, and prefill MoE paths;
- derive KV, GDN, step-buffer, workspace, and Vision memory from the selected profile;
- carry checkpoint-appropriate tokenizer, chat-template, processor, and model identity metadata;
- report execution/checkpoint profile, schema digests, packing target/profile, and artifact identity
  in serving and benchmark output;
- keep the existing external OpenAI and Anthropic protocol contracts coherent;
- qualify both profiles independently for numerical correctness, memory safety, observable behavior,
  and caller-visible performance.

### 2.1 Compatibility contract

“More compatible” has a narrow, testable meaning in this project:

- **Model-schema extensibility:** a new compiled profile can define new layer schedules, tensor
  meanings, component groups, tied/shared weights, and frontend assets without changing the core v5
  record layouts.
- **Storage reuse:** any profile whose tensors can be represented by the existing finite codec and
  layout registry uses the same v5 container and generic parser.
- **Safe inspection:** generic tools can validate offsets, sizes, shapes, storage formulas, views,
  CRCs, and assets even when the local binary does not know the model profile.
- **Exact loading:** inference still requires an exact compiled profile id, schema digest, packing
  profile, complete inventory, and typed binder. Unknown profiles never fall back to a similar
  architecture.
- **Explicit limits:** a genuinely new storage primitive, unsafe view rule, integrity contract, or
  catalog interpretation may require a new artifact revision. The format does not claim that every
  future tensor can be encoded without change.

This is compatibility across registered model schemas, not backward compatibility between arbitrary
engine versions. An older v5 engine may structurally inspect a newer profile that uses known storage
primitives, but it rejects loading because the profile is absent from its compiled registry.

## 3. Non-Goals

This work will not:

- accept arbitrary Hugging Face `config.json` files;
- support arbitrary Qwen3.5/Qwen3.6 variants merely because some dimensions happen to match;
- implement Gemma 4 or promise support for any non-Qwen checkpoint in this delivery;
- become a universal interchange format such as safetensors or GGUF;
- introduce a dynamic model graph, operator registry keyed by model strings, or per-layer virtual
  dispatch;
- let an artifact-provided graph, tensor name, JSON/TLV field, or unknown role determine operator
  order at runtime;
- accept an unknown codec, layout, view kind, or packing target by skipping validation;
- add batching, concurrent GPU sequences, multi-GPU execution, expert parallelism, tensor
  parallelism, or CPU expert offload;
- page experts from storage or host memory on demand;
- add runtime weight conversion, dequantized resident copies, or alternate weight loaders;
- preserve q5090 v4.2 compatibility after the v5 migration is complete;
- infer model semantics solely from tensor names, marketing names, or dimension coincidences;
- promise the model card's extended million-token operating point without separate RoPE, memory,
  numerical, and performance qualification;
- claim that a 3B active-parameter model needs only 3B parameters resident;
- make the runtime execute arbitrary Jinja chat templates or arbitrary processor configuration.

## 4. Current Facts

### 4.1 Existing system shape

The implemented engine is intentionally fixed to Qwen3.6-27B:

- one `ModelConfig` owns all Text, GDN, attention, MTP, and derived dimensions;
- `Qwen3_6_27B` owns the dense Text schedule and dense MTP schedule;
- `Qwen3_6_Vision` owns a fixed 27-layer tower whose merger emits width 5120;
- `Engine` duplicates the 27B geometry in cache, state, step-buffer, and workspace formulas;
- q5090 v4.2 combines a binary container, one exact model card, and one exact packing policy;
- the converter, verifier, Python reference, real-artifact tests, benchmark shapes, and serving
  default model id are all 27B-specific.

The current fixed dimensions are visible in [`../../include/qus/model/config.h`](../../include/qus/model/config.h).
The parser then requires those same values exactly in
[`../../src/core/weight_store_parser.cpp`](../../src/core/weight_store_parser.cpp), and the converter
has a second fixed copy in [`../../tools/q5090_convert/convert.py`](../../tools/q5090_convert/convert.py).

These checks are valuable: they prevent a structurally valid artifact from being interpreted with
the wrong model schedule. The static-registry design must replace them with exact profile validation rather
than simply removing them.

### 4.2 Profile comparison

The local BF16 checkpoints establish the following model facts:

| Field | Dense 27B | MoE 35B-A3B |
|---|---:|---:|
| HF architecture | `Qwen3_5ForConditionalGeneration` | `Qwen3_5MoeForConditionalGeneration` |
| HF model type | `qwen3_5` | `qwen3_5_moe` |
| Text hidden | 5120 | 2048 |
| decoder layers | 64 | 40 |
| GDN / full-attention layers | 48 / 16 | 30 / 10 |
| full-attention Q / KV heads | 24 / 4 | 16 / 2 |
| attention head dimension | 256 | 256 |
| Q heads per KV head | 6 | 8 |
| GDN QK / V heads | 16 / 48 | 16 / 32 |
| GDN head dimension | 128 | 128 |
| GDN V heads per QK head | 3 | 2 |
| GDN convolution width | 4 | 4 |
| dense FFN intermediate | 17408 | absent |
| routed experts / top-k | absent | 256 / 8 |
| expert intermediate | absent | 512 |
| shared expert intermediate | absent | 512 |
| vocabulary | 248320 | 248320 |
| MTP layers | 1 dense | 1 MoE |
| Vision depth / hidden / intermediate | 27 / 1152 / 4304 | 27 / 1152 / 4304 |
| Vision merger output | 5120 | 2048 |

Both profiles retain:

- three GDN layers followed by one full-attention layer in each four-layer group;
- attention head dimension 256 and partial rotary dimension 64;
- RoPE theta `1e7`, interleaved MRoPE sections `[11,11,10]`, and RMS epsilon `1e-6`;
- GDN key/value head dimension 128, width-4 causal convolution, and FP32 recurrent state;
- gated full-attention output, zero-centered ordinary/QK RMSNorm, and plain GDN gated RMSNorm;
- an untied embedding and full `lm_head`;
- the same Vision backbone geometry, but not the same Vision weights.

The explicit 35B facts come from
[`config.json`](</home/neroued/models/llm/qwen/Qwen3.6-35B-A3B/base-hf-bf16/config.json>) and the
safetensors index under the same directory. The marketing name is not a sufficient dispatch key:
both implementations still use upstream `qwen3_5*` architecture identifiers, and the existing 27B
source also contains checkpoint-specific fields. The converter must normalize source metadata into a
project-owned exact profile.

### 4.3 Real 35B tensor inventory

For every 35B Text layer, and for the one MTP layer, the FFN tensors are:

```text
mlp.experts.gate_up_proj                    [256,1024,2048]
mlp.experts.down_proj                       [256,2048,512]
mlp.gate.weight                             [256,2048]
mlp.shared_expert.gate_proj.weight          [512,2048]
mlp.shared_expert.up_proj.weight            [512,2048]
mlp.shared_expert.down_proj.weight          [2048,512]
mlp.shared_expert_gate.weight               [1,2048]
```

`experts.gate_up_proj` is already fused in source order. For each expert, rows `[0,512)` are the
SiLU gate and rows `[512,1024)` are the multiplicative up projection. Expert id `e` is the same
identity as router row `e`; no implicit expert permutation is allowed in the initial format.

Other derived 35B widths are:

```text
attention Q / KV width              4096 / 512
attention raw q_proj rows           8192 (query and output gate interleaved per head)
attention o_proj                    [2048,4096]
GDN Q / K / V / Z width             2048 / 2048 / 4096 / 4096
GDN convolution rows                8192
GDN in_a / in_b                     [32,2048]
GDN out_proj                        [2048,4096]
MTP fc                              [2048,4096]
Vision merger fc2                   [2048,4608]
```

The attention q-projection remains interleaved as `[query_256 | gate_256]` per head. The converter
must continue to perform the existing value-preserving de-interleave transform; a simple split of
the first and second halves is wrong.

### 4.4 Parameter and memory facts

Header-only inspection of the local BF16 files gives:

| Component | Dense 27B | MoE 35B-A3B |
|---|---:|---:|
| embedding | 1.271B | 0.509B |
| Text layers | 24.353B | 33.643B |
| final norm + full head | 1.271B | 0.509B |
| Vision | 0.461B | 0.447B |
| MTP | 0.425B | 0.845B |
| total | 27.781B | 35.952B |

The full 35B expert banks must remain resident because any expert can be selected by the next token.
The approximately 3B active count describes work and bytes touched per token, not residency.

At context 262144, payload-only BF16 state is approximately:

| State | Dense 27B | MoE 35B-A3B |
|---|---:|---:|
| Text KV | 16 GiB | 5 GiB |
| one-layer MTP KV | 1 GiB | 0.5 GiB |
| one GDN recurrent snapshot slot | 144 MiB | 60 MiB |
| one GDN convolution snapshot slot | 2.81 MiB | 1.41 MiB |

The 35B profile therefore has larger resident weights but materially smaller KV and recurrent state.
The final qualified context remains a measured runtime property, not a model-card inheritance.

### 4.5 Tokenizer, template, and processor facts

The two local checkpoints have the same base BPE vocabulary mapping and the same core multimodal and
thinking token ids, but their embedded asset bytes and chat templates are not identical. The 35B
source template is
[`chat_template.jinja`](</home/neroued/models/llm/qwen/Qwen3.6-35B-A3B/base-hf-bf16/chat_template.jinja>).

The existing q5090 artifact already embeds checkpoint-specific tokenizer assets, which is the right
ownership. Static-registry support must not replace that with one global tokenizer assumption.
Chat-template and processor behavior also need explicit profile identity even if the two initial
processor geometries happen to match.

### 4.6 Cross-architecture format probe: Gemma 4

Gemma 4 is not an implementation target in this plan. It is a concrete probe for whether v5 is
actually model-neutral. As of this design, Google's
[official overview](https://ai.google.dev/gemma/docs/core) describes multiple substantially
different Gemma 4 architectures: small PLE models, a dense model, a sparse MoE model, and a unified
multimodal model. The official
[Transformers configuration](https://huggingface.co/docs/transformers/model_doc/gemma4) and
[implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modular_gemma4.py)
expose format-relevant structures that are not Qwen3.6 assumptions:

- Per-Layer Embeddings pack token-dependent inputs for many decoder layers into one large table;
- local and global attention layers may use different head geometry;
- K and V projections may be the same weight, and consecutive layers may share KV weights;
- embeddings and output heads may be tied;
- the model family includes dense and MoE FFNs, final-logit soft-capping, and different norm/activation
  semantics;
- multimodal variants include different Vision structures, audio convolution/attention weights,
  position tables, biases, rank-zero scale/control tensors, rank-three/four tensors, integer ordering
  tables, and processor assets.

The artifact implication is not that v5 must encode a Gemma execution graph. It is the opposite:

| Qwen-shaped core assumption to remove | Model-neutral replacement |
|---|---|
| fixed Text/MTP/Vision module enum | profile-scoped component slots |
| global Qwen source-role enum | profile-scoped numeric tensor slots |
| `LayerRecord` with mixer/FFN kinds | topology exists only in compiled model code |
| one semantic use per physical tensor | compiled profile may bind one storage object to tied/shared uses |
| logical tensor shape equals quantized matrix shape | separate logical tensor and physical storage/view records |
| fixed tokenizer record kinds | profile-scoped asset slots with bounded generic byte records |

If a tiny Gemma-shaped contract fixture cannot be represented and structurally checked without
adding Qwen/Gemma branches to the L0 parser, the v5 boundary is still too model-specific.

## 5. Design Principles

The implementation follows these principles:

1. **Exact profiles, not arbitrary configs.** Source configs are normalized and then matched to a
   compiled allowlist.
2. **Model-neutral bytes, model-specific mathematics.** L0 understands storage, views, assets, and
   integrity; compiled L2 profiles understand tensor meaning and execution order.
3. **One load-time choice.** Dynamic selection stops at the concrete Engine implementation boundary.
4. **Static hot paths.** Concrete profiles retain fixed arrays, straight-line layer order, and finite
   shape dispatch.
5. **Semantic L1 APIs.** Public operators express routing or expert mathematics, not one CUDA launch
   strategy.
6. **No names as runtime semantics.** Source tensor names and friendly labels are converter/manifest
   diagnostics only; runtime binding uses profile-scoped numeric slots and generated typed tables.
7. **Finite storage primitives.** Codecs, layouts, view kinds, and packing targets remain a compiled,
   strictly validated registry; model neutrality does not mean accepting unknown storage schemes.
8. **No hidden residency.** All persistent allocations and all workspace requirements are planned
   before execution.
9. **Direct artifact consumption.** Quantized weights stay in their canonical q5090 representation
   through decode and prefill.
10. **Structural and semantic validation remain separate.** A safe container is not automatically a
   valid model.
11. **27B remains a first-class optimized profile.** Registry support is not permission to replace
   tuned 27B kernels with slower generic kernels.
12. **Every claim names its profile.** Artifacts, reports, server model ids, parity results, and
   performance evidence are never ambiguous between 27B and 35B-A3B.

## 6. Target Engine Architecture

### 6.1 Profile model

Introduce a project-owned profile registry with numeric identities, compile-time execution specs,
exact artifact schemas, and read-only runtime information:

```cpp
enum class ExecutionProfileId {
    Qwen36Dense27B,
    Qwen36Moe35BA3B,
};

struct Qwen36Dense27BSpec;
struct Qwen36Moe35BA3BSpec;

struct CompiledArtifactSchema;  // components, tensor slots, assets, storage expectations
struct ModelInfo;  // immutable runtime mirror for reporting/frontends
```

Each registry entry binds exactly one execution profile id to:

- a schema digest and generated `CompiledArtifactSchema`;
- the finite allowed checkpoint profiles, schema digests, and canonical model/asset identities;
- the finite allowed packing-profile ids and digests;
- CPU-only exact inventory/asset validators;
- a whole-engine factory for one handwritten concrete implementation;
- immutable `ModelInfo` and frontend capabilities.

The registry may use a constexpr table, switch, variant, or generated enum. It is not an artifact-
provided operator registry. Adding an entry changes the binary and is subject to full profile
qualification.

The initial stable semantic identities are distinct from display/serving names:

| Meaning | 27B | 35B-A3B |
|---|---|---|
| architecture family label | `qwen3_6` | `qwen3_6` |
| execution profile id | `qwen3_6_dense_27b_v1` | `qwen3_6_moe_35b_a3b_v1` |
| checkpoint profile id | `qwen3_6_27b_checkpoint_v1` | `qwen3_6_35b_a3b_checkpoint_v1` |
| packing target id | `q5090_sm120a_v1` | `q5090_sm120a_v1` |
| canonical serving id | `qwen3.6-27b` | `qwen3.6-35b-a3b` |

The family label is informational and is not interpreted by L0. Numeric profile ids are stable
project-owned identities. The container defines their field width and
namespace but does not enumerate model-specific values as core ABI semantics. A packing profile
identifies storage policy, not model mathematics, and cannot substitute for the execution profile id.

The two Qwen execution specs own the fields listed in the Goal and additionally provide exact
layer-index maps:

- `layer_kind[layer]`;
- `full_slot[layer]` or `NO_SLOT`;
- `gdn_slot[layer]` or `NO_SLOT`;
- `ffn_kind[layer]`;
- the independent MTP layer profile.

The slot maps are derived once from the exact layer table. KV and GDN code must not keep formulas
that assume every fourth layer forever.

The runtime `ModelInfo` and artifact inventory tables are generated from the same profile source;
they are not independently maintained copies. That source enumerates tensor/component/asset
contracts only. It does not describe a runtime operator graph; the C++ model schedule remains
explicit code.

### 6.2 Load sequence

The new load flow is:

```text
open stable artifact fd
  -> parse bounded v5 header/component/storage/view/asset catalogs without CUDA
  -> validate framing, references, structure, and arithmetic
  -> validate target and every present codec/layout/view through generic storage support
  -> resolve execution/checkpoint ids, schema digest, target, and packing profile in compiled registry
  -> reject an unknown profile for loading while retaining a safe structural diagnostic
  -> validate exact component/tensor-slot/asset inventory through the compiled profile
  -> resolve requested optional components
  -> build exact weight/cache/workspace residency plan
  -> initialize CUDA
  -> upload selected immutable storage objects
  -> construct concrete EngineImpl and bind typed weight tables
  -> initialize stable step/sampling/routing buffers
  -> allow prefill/decode/serve calls
```

Artifact identity selects the implementation. A command-line `--model-type` must not be required and
must not be able to override the artifact.

### 6.3 Engine type erasure

The public `Engine` retains one stable API and owns one concrete loaded implementation through a
whole-engine variant or type-erased pointer:

```text
Engine
  `-- EngineImpl<Qwen36Dense27BSpec>
  `-- EngineImpl<Qwen36Moe35BA3BSpec>
```

A future non-Qwen profile adds another whole Engine implementation or an explicitly shared family
implementation. It does not extend the Qwen layer skeleton with model-name branches.

Type erasure occurs around whole calls such as `prefill`, `decode_step`, memory statistics, and MTP
statistics. It must not occur around individual layers or operators. Concrete EngineImpl storage
must not move after CUDA Graph capture.

### 6.4 Qwen3.6 L2 Text structure

The following skeleton is shared only by the two registered Qwen3.6 profiles:

```text
embedding
for exact layer in profile:
  h = zero-centered RMSNorm(x)
  x = x + FullAttention(h) or GDN(h)
  h = zero-centered RMSNorm(x)
  x = x + DenseSwiGLU(h) or QwenSparseMoE(h)
final zero-centered RMSNorm
full lm_head and sampling
```

Use explicit weight structures:

```cpp
struct DenseFfnW;
struct MoeFfnW;

template<class FfnW> struct FullLayerW;
template<class FfnW> struct GdnLayerW;
template<class FfnW> struct MtpW;
```

Templates are an implementation option, not a requirement. Two concrete cards sharing common
attention/GDN helpers are acceptable if that is clearer. The required property is that a dense layer
does not carry a runtime MoE branch and a MoE layer does not pretend to be three ordinary dense
weights.

It is not the universal registry schedule. A future Gemma profile may own different residual order,
norms, attention types, tied weights, shared projections, or modality injection in separate explicit
code while still consuming the same model-neutral storage records.

### 6.5 Qwen3.6 MTP structure

MTP shares the stem and full-attention semantics across both profiles:

```text
normalized token embedding + normalized target hidden
  -> fc(concat)
  -> one full-attention decoder layer
  -> profile-specific dense or MoE FFN
  -> final norm
```

The 35B MTP source contains its own router, 256 expert banks, shared expert, and shared gate. Target
Text and MTP must therefore use the same MoE mathematical contract but distinct weights and caches.

The proposal head remains separate from target correctness. A 35B shortlisted head must be regenerated
with width 2048; the current width-5120 draft block cannot be reused.

### 6.6 Qwen3.6 Vision and multimodal processing

The two initial profiles share the Vision backbone geometry, so its block implementation can be
reused. The Vision profile still declares every dimension explicitly and statically checks the shared
implementation assumptions.

The merger output, workspace output tensor, scatter validation, and Text placeholder injection use
`vision.out_hidden == text.hidden`. They must not contain a literal 5120.

The processor receives a `MultimodalProfile` containing patch/merge geometry, normalization,
position behavior, and special token identities. Media safety budgets remain runtime policy, not
model geometry.

### 6.7 Resource planning ownership

Physical arenas remain owned by runtime/L0. Exact size formulas move to the components that know the
layout:

- `KVCache::required_bytes(...)` from the same arguments as its constructor;
- `GdnState::required_bytes(...)` from the same arguments as its constructor;
- profile-specific `StepStateLayout`;
- dense and MoE Text workspace plans by phase and token regime;
- profile-specific Vision workspace planning.

The planner returns an exact allocation sequence as well as a conservative capacity. Tests compare
planned capacity with observable arena high-water behavior; they do not lock private allocation call
order.

## 7. Model-Neutral q5090 v5 Artifact Contract

### 7.1 Version decision

Static-registry support requires a new major artifact revision because it changes:

- model identity and schema binding;
- the catalog from Qwen modules/source roles to generic components, storage objects, tensor views,
  and assets;
- logical-versus-physical tensor representation;
- shared/tied storage binding;
- the versioned storage-descriptor envelope needed by later matrix-array layouts;
- parser, loader, and runtime descriptors.

The implementation will move directly to q5090 v5, but specification promotion and runtime cutover
must be atomic. During Phase 1, v5 remains a candidate contract owned by this active plan; it must
not be described as the implemented or normative format while the runtime still consumes v4.2.
During the Phase 2 cutover:

- the candidate contract is promoted as `docs/q5090_packed_file_format_v5.md` and replaces the active
  specification in the same change that switches the converter, reader, verifier, C++ parser,
  loader, fixtures, and runtime to v5 only;
- the final active v4.2 specification is preserved as
  `docs/archive/optimization-era/q5090_packed_file_format_v4_2.md`, avoiding collision with the
  earlier archived `q5090_packed_file_format_v4.md` snapshot;
- the real 27B artifact is regenerated as v5 and becomes the canonical runtime input;
- v4 branches, aliases, fallback parsers, and compatibility tests are deleted.

The 35B artifact is then generated from that already-active v5 contract in Phase 4. No supported
revision may leave the active specification describing a format different from the one accepted by
the runtime.

The `q5090` name is retained as the project's RTX-5090 packed-artifact lineage, not as a Qwen model
namespace. Artifact names must not continue to encode only `w4g64`: codec and layout are
self-describing per storage object, and the packing mixture is profile-specific.

Proposed canonical artifact names are:

```text
out/qwen3_6_27b.q5090_mixed_v5.qus
out/qwen3_6_35b_a3b.q5090_mixed_v5.qus
```

### 7.2 Contract separation

v5 separates five contracts that v4.2 currently merges:

| Contract | Owns | Must not own |
|---|---|---|
| container ABI | bounded records, offsets, storage formulas, views, assets, alignment, CRC | model layers or tensor meaning |
| compiled model schema | exact component/tensor slots, shapes, aliases, capabilities, frontend interfaces | physical qtype policy or runtime graph data |
| checkpoint profile | canonical serving identity, exact asset/product contract, compiled handlers, checkpoint semantics | execution dispatch or storage layout |
| packing schema | storage-object grouping, codec/layout assignment, offline transforms | model mathematics |
| handwritten execution profile | operator order, state transitions, resource plans, numerical semantics | parsing arbitrary artifact structure |

The container does not carry a generic `ModelRecord`, `LayerRecord`, FFN union, attention enum, GDN
geometry, or Vision/MTP topology. Those fields would merely turn the binary format into a Qwen graph
schema and force another major revision for a different architecture.

### 7.3 Header and exact identity

The fixed header contains only framing fields:

```text
magic, ABI major/minor, endian marker, header size, file size
section-directory offset/count/entry size
required and optional feature bits
artifact UUID and directory/root integrity
reserved zero fields
```

Each fixed-size `SectionRecord` contains `section_kind`, `section_version`, critical/optional/
repeatable flags, record size/count, offset/bytes, alignment, and integrity. Core sections include
identity, components, storage objects, tensor views, assets, provenance, optional debug symbols, and
payload integrity. Unknown critical sections are rejected. An unknown optional section may be skipped only if
it is explicitly non-executable and no required record references it.

The required identity section contains:

```text
packing target id                         # exact RTX 5090 / sm_120a target
execution_profile_id                      # selects compiled mathematics and EngineImpl
model_schema_version
model_schema_sha256
checkpoint_profile_id                     # exact product/tokenizer/template identity
checkpoint_schema_sha256
packing_profile_id                        # project-owned u64 identity
packing_schema_sha256
artifact content identity
```

`model_schema_sha256` identifies the canonical compiled inventory contract, not source checkpoint
bytes. `packing_schema_sha256` identifies the exact storage policy. Hashes detect mismatches; they do
not replace bounds checks, exact inventory validation, CRCs, or an authenticity mechanism.

Canonical schema bytes specify endian, field order, slot sort order, integer widths, and float bit
encoding; they are never C++ struct memory or ad-hoc JSON text. The model schema digest covers the
execution-semantics revision, component dependencies/capabilities, tensor slots and logical
dtype/shapes, allowed sharing, and frontend handler interfaces. The checkpoint schema digest covers
canonical serving identity, required asset slots/encodings/content digests, special-token/product
semantics, and compiled handler profiles. The packing digest covers target, tensor-view to storage
mapping, codecs/layouts/view versions, quantization parameters, fusion, and offline transform policy.
Debug names, paths, payload offsets, and actual weight bytes are excluded from model/packing schemas;
asset content digests remain part of the checkpoint schema.

Numeric execution profile identity selects the compiled validator and whole-engine factory;
checkpoint identity selects the exact allowed asset/product contract and canonical serving id for
that execution profile. The artifact may mirror friendly names in an optional `DebugSymbol` section,
but removing that section cannot change loading, binding, or serving identity.

Normalized architecture fields may be mirrored in the manifest for inspection, but no arbitrary
JSON/TLV metadata controls runtime dimensions or operator order.

### 7.4 Generic catalog records

The catalog has five model-neutral record families:

| Record | Meaning |
|---|---|
| `ComponentRecord` | profile-scoped residency/capability group and its object/view/asset ranges |
| `StorageObjectRecord` | one immutable physical payload with codec, layout, storage shape, alignment, byte formula, offset, and CRC |
| `TensorViewRecord` | one profile-scoped semantic tensor slot mapped to a bounded logical view of a storage object |
| `AssetRecord` | one profile-scoped raw frontend/model asset with bounded bytes and integrity |
| `ProvenanceRecord` | repeatable typed digest for source config, index, checkpoint file, converter, or calibration evidence |

`ComponentRecord` carries a profile-scoped slot, required/optional/default-resident/separately-
loadable flags, and bounded dependency references. The compiled profile decides what a component
means and whether a requested capability is available.

`AssetRecord` carries a profile-scoped slot, bounded encoding id, component membership, bytes, and
digest. The encoding id selects a compiled handler; raw template/config bytes may be retained for
provenance, but the runtime never executes arbitrary Jinja, JSON, or code from the artifact.

`ProvenanceRecord` is repeatable and never participates in execution dispatch. It can identify one
or several source checkpoints, config/index files, converter builds, or quantization evidence without
assuming Hugging Face is the only source format.

`StorageObjectRecord` owns payload non-overlap. `TensorViewRecord` never owns bytes. Multiple views
may safely reference one immutable storage object; the structural parser checks containment and the
compiled schema decides whether overlap, tying, or sharing is semantically legal. This supports tied
embeddings, K=V reuse, cross-layer shared projections, and fused blocks without duplicating payload.

Logical and physical shape are distinct:

- a tensor view carries the profile-visible dtype/rank/dimensions;
- a storage object carries runtime-native storage rank/dimensions and codec/layout parameters;
- a finite `ViewKind` registry defines only mechanically verifiable mappings such as whole-object,
  element-count-preserving reshape, row-range, and matrix-array-range views;
- arbitrary strided views, executable transforms, and artifact-provided indexing programs are not
  allowed;
- offline transforms are selected by the compiled packing schema and are validated through the
  expected logical/view/storage triple.

v5 permits `0 <= logical_rank, storage_rank <= 8`. Rank zero has element count one and no dimensions;
all higher-rank element-count products use checked arithmetic and require nonzero dimensions. This
covers real scalar controls, Text, expert-bank, convolution, position-table, and multimodal weights
without treating a quantized linear's two-dimensional matrix view as a universal tensor shape.

### 7.5 Profile-scoped slot namespaces

Core v5 has no global `TEXT_CORE`, `MTP_DRAFT`, `VISION_ENCODER`, `SourceKind`, layer-kind, fusion-id,
tokenizer-kind, or model-specific tensor enum. Instead each compiled model schema owns dense numeric
namespaces:

```text
component_slot : u32
tensor_slot    : u32
asset_slot     : u32
```

The slot is meaningful only under `(execution_profile_id, model_schema_sha256)`. Friendly names and
source safetensors paths exist in generated converter tables, manifests, and diagnostics, but are
not required for binding. Adding a model therefore adds a schema namespace rather than extending an
L0 switch over all models ever supported.

The schema source may generate constexpr C++ tables and Python converter constants. It describes
inventory, shapes, allowed aliases, component presence, and assets only; it cannot describe
operators, control flow, cache transitions, or CUDA launch policy.

### 7.6 Qwen3.6 component and tensor schemas

The first two compiled schemas retain these Qwen-specific component slots:

- `TEXT_CORE` includes Text router, experts, and shared experts;
- `MTP_DRAFT` includes MTP router, experts, and shared experts;
- `VISION_ENCODER` remains independent;
- `LM_HEAD_DRAFT` remains optional.

These names belong to the Qwen profile namespace, not the container ABI. A future Gemma profile can
define different component slots without changing the structural parser.

Qwen presence semantics remain explicit:

- `TEXT_CORE` is mandatory and appears exactly once;
- `MTP_DRAFT` and `VISION_ENCODER` are selectively present artifact capabilities; when absent, the
  Engine disables speculative decoding or multimodal requests explicitly;
- `LM_HEAD_DRAFT` is optional, may appear only with `MTP_DRAFT`, and changes proposal quality only;
- both canonical Qwen release artifacts must include Text, MTP, and Vision, even though the Qwen
  schema permits Text-only diagnostic artifacts.

The Qwen schema defines tensor slots corresponding to router, expert gate/up and down banks, shared
expert projections, shared scalar gate, and all existing dense/attention/GDN/MTP/Vision weights.
Shared gate/up may map to row views of one fused storage object. Expert gate/up remains one bank
view because it is already one source tensor.

### 7.7 Finite storage primitive registry

The Phase 2 v5 cutover initially defines only the storage primitives required by the qualified 27B
artifact:

- raw contiguous BF16, FP16, FP32, I32, I64, and U8 storage, including rank-zero scalars;
- existing Q4G64, Q5G64, Q6G64, and W8G32 code/scale encodings;
- ordinary contiguous and row-split layouts.

The fixed descriptor envelope permits Phase 3 to add a matrix-array layout id/version as an explicit
v5 storage extension after real-shape evidence. The Phase 2 normative contract does not predeclare
that layout's byte formula.

Each codec/layout/view combination has a normative checked byte formula. Unknown ids or combinations
are rejected before allocation or upload. A future profile using this existing registry needs no
container change. A new codec or view interpretation is a storage-ABI change and must be specified,
tested, and versioned deliberately; model compatibility is never implemented by accepting an opaque
payload the runtime cannot validate.

### 7.8 Matrix-array layout for expert banks

Do not create one TensorEntry per expert. That would create more than twenty thousand entries for
the two routed banks in the 40 Text layers alone.

The preferred extension candidate is the model-neutral `MATRIX_ARRAY_ROW_SPLIT`, with one physical
`[A,N,K]` storage object and one ordinary row-split mini-block per array element. For the Qwen MoE
packing profile, `A == E` and array element `e` stores expert `e`:

```text
Kp = align_up(K, 128)
G  = Kp / group_size

nibble_a = N * G * nibble_bytes_per_group
high_a   = N * G * high_bytes_per_group
scale_a  = N * G * sizeof(fp16)

high_rel_a  = align_up(nibble_a, 256)
scale_rel_a = high_rel_a + align_up(high_a, 256)
matrix_payload = scale_rel_a + scale_a
matrix_stride  = align_up(matrix_payload, 256)

matrix_base(a) = object_base + a * matrix_stride
object_payload = A * matrix_stride
```

This provides:

- direct `array_index -> base + stride` addressing;
- zero-copy ordinary Weight views of one selected expert;
- reuse of existing Q4/Q5/Q6/W8 code-plane decoding inside an expert;
- sequential, bounded converter/verifier processing;
- no runtime repack or expert-id map;
- a bounded catalog with approximately 1024 blocks for the proposed full 35B artifact.

Before the layout id/version and byte formula become normative, a real-shape microbenchmark must
compare this matrix-local layout with the
simpler alternative of flattening `[A,N,K]` to `[A*N,K]` in global ROW_SPLIT planes. The Qwen35
decision is based on selected-expert decode and grouped-prefill evidence, not intuition alone.

Whichever layout wins, the Qwen35 packing schema forbids:

- one catalog entry per expert;
- runtime expert permutation or repacking;
- any frequency-based or other expert reordering in the Qwen35 packing profile;
- a bank binding whose array index differs from router row identity.

The core layout assigns no router or expert semantics to axis `A`. A future compiled profile may use
the same physical matrix-array primitive for a different semantic array, subject to its own exact
schema.

### 7.9 Qwen3.6-35B-A3B initial packing candidate

The initial 35B packing candidate is:

| Tensor class | Candidate qtype/layout |
|---|---|
| routed expert gate/up bank | Q4G64 expert bank |
| routed expert down bank | Q5G64 expert bank |
| shared gate/up | Q4G64 ordinary fused ROW_SPLIT |
| shared down | Q5G64 ROW_SPLIT |
| router | BF16 CONTIGUOUS |
| shared scalar gate | BF16 CONTIGUOUS |
| MTP linears and expert banks | W8G32 initially |
| embedding and full head | Q6G64 |
| optional draft head | Q4G64 shortlist |

Router and shared-gate quantization are not part of the initial scope. Router error can change the
discrete expert set and is more dangerous than ordinary projection error.

Using the real source shapes and this candidate policy gives an estimated packed payload:

| Payload class | Estimated size |
|---|---:|
| 40-layer MoE tensors | 17.29 GiB |
| complete TEXT_CORE | 18.78 GiB |
| VISION_ENCODER | 0.26 GiB |
| LM_HEAD_DRAFT | 0.13 GiB |
| W8 MTP_DRAFT | 0.84 GiB |
| complete artifact payload | 20.02 GiB |

These are deterministic format estimates, not an accepted quality, residency, or throughput claim.
The qtype assignment becomes normative only after quantization parity and end-to-end evaluation.

### 7.10 Converter architecture

Replace the converter's single global `EXPECTED` dictionary with exact source adapters. Source
normalization handles checkpoint packaging differences such as nested config fields, `dtype` versus
`torch_dtype`, upstream architecture names, tied/absent tensors, and shared source weights. It does
not create new runtime profiles automatically.

The generic writer receives model-neutral records from one exact source adapter and packing planner:

```text
source checkpoint
  -> exact SourceAdapter<ExecutionProfile, CheckpointProfile>
  -> normalized source identity and profile match
  -> ComponentSpec / TensorViewSpec / AssetSpec / ProvenanceSpec
  -> exact PackingPlan<PackingTarget, PackingProfile>
  -> StorageObjectSpec and bounded payload producers
  -> model-neutral v5 writer
```

The writer knows record framing, codecs, layouts, views, alignment, integrity, and transactional file
production. It does not know Text, attention, GDN, MTP, Vision, MoE, Gemma, or source tensor names.
The Qwen adapters remain explicit dense/MoE Text, MTP, Vision, and draft-head plans above that
boundary.

Each source adapter classifies checkpoint keys as:

- required and consumed into one or more tensor slots;
- recognized but intentionally dropped, with a profile-owned reason recorded in the manifest;
- forbidden/unknown, which fails conversion.

Source inventory is therefore not assumed to equal runtime semantic inventory. This handles
checkpoints that retain redundant tied/shared tensors or training/export auxiliaries without either
silently ignoring unknown weights or forcing unused source tensors into the artifact ABI.

Expert banks must be processed in bounded slices. The converter must not materialize an entire
multi-hundred-megabyte packed bank or all expanded quantization intermediates at once. With a
bank-local layout it can read one expert slice, quantize it, write its final region, update integrity
state, and release temporary memory.

The writer keeps the existing transactional behavior: bounded temporary output, complete
verification, fsync, and atomic rename. Neither converter nor tests choose artifacts by glob or
modification time.

### 7.11 Structural and framing validation

The L0 parser validates safety independently of model support:

- every offset/size/product uses checked unsigned arithmetic;
- record counts, section sizes, string bytes, ranks, and descriptor bytes have explicit caps;
- header and component/storage/view/asset/provenance sections are bounded and aligned exactly as
  specified;
- scalar rank zero is distinguished from malformed rank/dimension data; every present dimension is
  nonzero and representable;
- component membership and every object/view/asset reference resolves within its catalog;
- storage-object payloads are ordered, bounded, aligned, and non-overlapping;
- every tensor view is mechanically contained in its referenced immutable storage object;
- asset/provenance/string ranges and padding are bounded and valid;
- all reserved fields and unknown critical record/section versions are rejected;
- arithmetic remains safe even when `execution_profile_id` is unknown.

CRC detects accidental corruption; it is not an authentication mechanism. Safety cannot depend on an
attacker being unable to recompute CRC.

### 7.12 Storage-support validation

After safe framing and still before CUDA initialization, the generic storage registry validates:

- packing target support;
- codec, layout, view kind, and descriptor versions;
- legal dtype/codec/layout/view combinations;
- checked plane, stride, padding, alignment, and payload-size formulas;
- matrix-array count/stride and all contained matrix payloads;
- every present component, including a component the caller does not plan to make resident.

An offline inspector may report an unsupported execution profile after completing safe framing. Engine
loading rejects an unknown profile, packing target, codec, layout, view, or required extension; it
never guesses a fallback representation.

### 7.13 Exact compiled-profile validation

After structural parsing and before CUDA initialization, the selected profile validates:

- execution, model-schema, checkpoint, checkpoint-schema, packing-target, packing-profile, and
  packing-schema identities;
- every required/optional/conditional component slot and dependency;
- no duplicate, missing, or unknown tensor/asset slot;
- exact logical dtype/rank/shape for every tensor view;
- exact storage-object/view mapping, permitted aliasing, codec, layout, and parameters;
- exact required asset slots, encodings, sizes/caps, and handler profiles;
- checkpoint-derived canonical model/serving identity;
- a completely typed binding table before any Engine object is constructed.

The two Qwen validators additionally check:

- embed and full head shape `[vocab,H]`;
- dense FFN shapes for the dense profile;
- router `[E,H]`, gate/up bank `[E,2I,H]`, and down bank `[E,H,I]` for every MoE layer;
- shared gate/up/down and scalar-gate shapes;
- general safety requires `0 < top_k <= experts`, and the exact 35B profile additionally requires
  `experts == 256` and `top_k == 8`;
- expert-axis index equal to router row id, with no expert id map or permutation in the Qwen35
  packing profile;
- full-attention, GDN, control, norm, and fused-storage view shapes;
- MTP FFN kind and complete inventory;
- Vision merger output equal to Text hidden;
- draft head `K == H`, matching id-map length, and valid unique vocab ids;
- qtype and layout assignment equal to the selected packing profile.

This replaces the parser's current fixed-27B assertion and
`WeightStore::require_mtp_module_expectations()`. It does not weaken them.

### 7.14 Content and value verification

Accepted structure is not proof that weights are correct. Offline/full verification additionally
checks:

- catalog/root integrity and every storage object/asset digest or CRC;
- zero padding and complete payload coverage;
- deterministic quantizer/code-plane parity against an independent numerical implementation;
- logical tensor values against the exact checkpoint source where available;
- provenance records without assuming every source uses Hugging Face config/index files;
- profile-level block/layer/output parity through the handwritten reference implementation.

Runtime integrity policy may optimize which already validated resident payloads are rechecked, but
it never uses schema hashes as a substitute for content integrity.

### 7.15 Runtime descriptors and manifest

L0 materializes generic non-owning descriptors first:

```cpp
struct PackedStorageObject;
struct PackedTensorView;
struct MatrixArrayWeight;
```

The compiled Qwen35 binder then constructs its direct expert descriptor:

```cpp
struct ExpertWeightBank {
    const void* payload;
    QType qtype;
    int experts;
    int n;
    int k;
    size_t matrix_stride;
    // per-expert plane offsets and validated metadata
};
```

`expert(e)` may construct a zero-copy temporary `Weight` view or the MoE operator may consume the
matrix array directly. A future profile may build a different typed weights struct from the same
generic descriptors. No descriptor allocates or repacks device memory.

The manifest continues to mirror the binary and adds structured fields for:

- execution/model-schema/checkpoint/checkpoint-schema identity;
- packing target/profile/schema identity and quantizer contract;
- artifact content identity;
- component and storage-object resident bytes;
- asset inventory and generic provenance digests;
- optional profile-specific statistics extensions.

For Qwen35, the profile extension reports matrix-array/expert layout, expert-bank resident bytes,
active top-k bytes per token, and router/shared bytes. These are not mandatory fields for a dense or
non-MoE model.

A single `effective_text_bpw` is insufficient for MoE because resident bytes and active bytes differ
substantially.

### 7.16 Evolution rules

The following table defines the promised model-schema compatibility of v5:

| Change | Core container revision |
|---|---|
| add an exact execution/checkpoint profile using known storage primitives | none |
| allocate a new checkpoint profile/schema for an existing execution profile | none |
| add profile-scoped component, tensor, asset, or provenance slots | none |
| add a packing profile using known codec/layout/view combinations | none |
| add a new compiled codec/layout/view id within the fixed descriptor envelope | compatible v5 extension; old engines finish framing then reject at storage-support validation |
| add non-executable optional diagnostic metadata | compatible bounded extension |
| change an existing id's meaning or byte formula | new major revision |
| change record framing, reference safety, integrity interpretation, or executable semantics | new major revision |

An extension never implies plugin loading. New storage ids are implemented and tested in the source
tree, recorded by required feature bits, and rejected by binaries that do not compile them. Optional
metadata can be skipped only when no required record references it and it cannot affect execution,
binding, assets, integrity, or schema identity.

Execution, checkpoint, packing, codec, layout, and view ids are immutable once assigned. A normal
model/checkpoint revision allocates a new id and schema digest inside the same container ABI; it never
silently changes an existing id's contract.

### 7.17 Adding a future architecture

Adding a future model such as a qualified Gemma profile requires an explicit delivery that adds:

1. a handwritten `ExecutionProfile`/EngineImpl with its exact mathematics, state, and resource plan;
2. one or more finite checkpoint profiles with canonical model ids and compiled frontend handlers;
3. a source adapter with required/dropped/forbidden source inventory;
4. component/tensor/asset slots and canonical model-schema bytes;
5. an RTX-5090 packing profile and canonical packing-schema bytes;
6. any missing semantic L1 operators or finite real-shape kernel dispatch;
7. independent weight, activation, model, protocol, memory, and performance qualification.

If existing v5 storage primitives cover the weights, none of these steps changes the core header,
section directory, generic record layouts, or structural parser. If a new primitive is required, its
codec/layout/view extension is designed and validated independently; the artifact still does not
describe the new model's execution graph.

## 8. Qwen3.6-35B-A3B MoE Mathematical And Operator Contract

### 8.1 Reference semantics

For post-attention normalized token state `h`, the 35B inference contract is:

```text
router_logits = W_router h                       # [E]
router_probs  = softmax(router_logits, FP32)
(values, ids) = topk(router_probs, top_k)
routing_weight = values / sum(values)
routing_weight = cast_to(router_logits.dtype)

for each selected expert e:
  gate, up = split(W_gate_up[e] h, [I,I])
  y_e = W_down[e](SiLU(gate) * up)

y_routed = sum_in_defined_order(routing_weight[e] * y_e)

y_shared = W_shared_down(
    SiLU(W_shared_gate h) * W_shared_up h
)
shared_scale = sigmoid(W_shared_gate_score h)

output = y_routed + shared_scale * y_shared
```

There is no inference capacity factor, token dropping, stochastic routing, or auxiliary-loss term.
The shared expert is not routed expert 256; it always executes and has an independent scalar gate.

The numerical contract must explicitly freeze:

- the BF16 input rounding boundary;
- router projection accumulation and output dtype;
- FP32 softmax;
- top-k tie policy;
- post-top-k normalization and cast point;
- expert activation and projection accumulation precision;
- routed expert reduction order and output rounding;
- shared scalar sigmoid precision;
- final routed/shared addition order.

Although selected-logit softmax can be algebraically equivalent to full-softmax followed by
renormalization, it is not adopted until its real BF16/router parity is demonstrated.

### 8.2 L1 public surface

The public surface describes mathematical work. Two acceptable decompositions are:

```text
topk_softmax_router
routed_swiglu_experts
shared_swiglu_expert
weighted_expert_reduce
```

or one public `sparse_moe` contract with private token-regime launchers. The final choice should make
independent numerical oracles and workspace ownership clear without exposing block size, expert CTA
mapping, histogram implementation, or GEMM strategy.

The model layer owns that every decoder layer invokes this operator. L1 owns decode/small-T/prefill
dispatch, expert-bank addressing, and launch policy.

### 8.3 One-token decode

For `T == 1`, every layer selects exactly eight routed experts and one shared expert. The path must:

- compute router logits, top-k ids, and weights on device;
- keep ids and weights in stable Engine-owned or workspace addresses;
- launch a fixed topology independent of expert identity;
- process the eight selected expert gate/up matrices with sufficient combined parallelism;
- process expert down projections and weighted reduction without host intervention;
- execute or fuse the shared branch where numerically justified;
- consume expert-bank bytes directly.

It must not read expert ids back to the CPU or issue 24 host-selected projection launches per layer.
A likely private mapping is CTA work over `(route_slot, output_row_tile)`, but that is a kernel
decision rather than an API contract.

### 8.4 MTP and small-T

Autoregressive MTP proposal and target candidate-window verification normally use `T` between one
and six. This regime needs a separate selected-expert path rather than being forced through either
the scalar decoder or a large prefill grouped GEMM.

Small-T routing remains dynamic data inside fixed launch shapes. It must cover:

- autoregressive proposal steps;
- target candidate-window verification;
- eager execution;
- captured MTP round execution.

MTP prompt preparation is not a small-T path. It follows the ordinary prompt in bounded
`mtp_prefill_chunk` slices and must use the grouped/prefill MoE contract in the next section.

### 8.5 Prefill

For prefill, each token creates `top_k` token-expert assignments. The operator requires bounded
workspace for at least:

```text
topk_ids[T,top_k]
topk_weights[T,top_k]
expert_counts[E]
expert_offsets[E+1]
token/expert permutation[T*top_k]
reverse or reduction mapping[T*top_k]
expert intermediate storage or streamed tiles
routed output accumulation[T,H]
shared expert intermediates
```

The implementation groups assignments by expert and invokes grouped or persistent expert GEMM
without unconditional host launches for all 256 experts. It must handle:

- uniform expert distribution;
- all tokens concentrated into the same eight experts;
- empty experts;
- repeated execution across chunk boundaries;
- `T*top_k` and workspace-size overflow checks.

No expert capacity limit may silently drop assignments.

### 8.6 Existing operator expansion

In addition to MoE, current operators and dispatch need explicit 35B coverage for:

- hidden width 2048 norms and residual operations;
- GQA 16Q/2KV with head dimension 256;
- GDN 16 QK heads and 32 V heads, including V:QK mapping ratio two;
- GDN Q/K/V projection rows 8192 plus Z rows 4096 (12288 fused-dispatch total), convolution Q/K/V
  rows 8192, and separate `[32,2048]` A/B controls;
- low-bit linears with new `(N,K)` shapes;
- `lm_head[248320,2048]` and draft heads with `K=2048`;
- MTP attention and fc shapes;
- Vision merger `[2048,4608]` and visual scatter width 2048.

Existing tuned 27B shape branches stay in the finite dispatch plan. Correct 35B fallbacks are added
first; tuning follows evidence.

## 9. State, Workspace, And CUDA Graphs

The ownership rules are registry-wide; the concrete geometry below is specific to the two Qwen
profiles. A future execution profile supplies its own cache/state/workspace plan rather than adding
fields to the container or the Qwen state structures.

### 9.1 Persistent state

Profile-owned state geometry includes:

- exact full-attention layer count and slot map;
- KV heads and head dimension;
- exact GDN layer count and slot map;
- GDN convolution, QK, V, and recurrent dimensions;
- MTP KV geometry;
- MTP and turn-boundary GDN snapshot-slot count;
- vocab/hidden-sized stable step and sampling buffers;
- stable MoE routing ids, weights, and counters where graph capture requires them.

Prefix reuse and MTP commit semantics remain common. They use the profile's slot maps rather than
assuming 16 KV layers and 48 GDN layers.

### 9.2 Workspace planning

Dense and MoE workspace are different model contracts. The MoE plan covers decode, small-T,
prefill, and MTP peaks independently and takes the maximum only after exact allocation sequences are
known.

The default remains a function of prefill chunk, profile, and enabled features, not total context.
Workspace high-water reports must remain flat when prompt length grows at fixed prefill chunk.

### 9.3 CUDA Graph stability

Dynamic expert ids are graph data, not graph structure. Graph safety requires:

- stable expert-bank payload addresses;
- stable router/top-k/output buffers;
- fixed launch counts and maximum shapes per captured path;
- no host-side expert branching;
- no post-capture movement of EngineImpl, cards, arenas, or sampling state;
- graph reset before unloading or changing execution profile.

Ordinary decode and full MTP rounds retain separate graph objects. Eager and captured execution must
share the same mathematical schedule.

### 9.4 Memory qualification

The provisional 20.02 GiB artifact estimate leaves plausible space on a 32 GiB device because 35B
KV/GDN state is smaller than 27B, but this is not yet a supported memory point.

Qualification records, for both BF16 and INT8 KV where applicable:

- CUDA runtime and non-arena overhead;
- selected component/storage-object residency;
- context-dependent KV bytes;
- GDN snapshot slots;
- Text and Vision workspace capacity and peak;
- graph instantiation memory;
- tokenizer/media host memory where relevant;
- success or rejection of complete prefill/decode/MTP flows.

## 10. Python Reference And Diagnostics

The Python tooling splits into a generic artifact reader and profile-specific mathematical
references. The reader understands v5 records, storage primitives, views, assets, and integrity even
for an unknown execution profile. It never invents a model schedule.

The current mathematical reference has one global 27B card in
[`../../tools/q5090/ref/config.py`](../../tools/q5090/ref/config.py), and its Text/MTP schedules always
run dense FFNs. It gains the two exact Qwen profile identities while remaining independent model
code. A future non-Qwen profile receives its own explicit reference adapter rather than more branches
inside a supposed universal Qwen reference.

Required changes include:

- v5 component/storage/view/asset/provenance parsing;
- cross-language golden generation for profile and packing schema digests;
- dense and MoE reference FFNs;
- explicit FP32 router softmax and top-k normalization;
- profile-specific Text, MTP, state, and Vision geometry;
- expert-bank dequantization by selected expert;
- taps for router logits, selected ids/weights, routed output, shared output, and layer output;
- real-weight block and layer parity for both profiles;
- profile identity in diagnostic dumps.

The CUDA algorithm must not be copied into the Python oracle. The reference follows upstream model
mathematics and documented BF16 rounding boundaries.

## 11. Tokenizer, Templates, Serving, And Reports

### 11.1 Runtime assets

v5 continues to embed checkpoint-specific frontend assets through generic `AssetRecord` slots. The
container does not define tokenizer JSON, SentencePiece, chat template, image, video, audio, or
generation-config enums. Each checkpoint profile declares the exact required asset slots, encodings,
digests, size caps, and compiled handler profile.

The two Qwen checkpoint schemas require tokenizer/generation assets and record chat-template,
image/video processor, and special-token provenance. A future profile may require a different
tokenizer model or modality assets without changing the core asset record.

The runtime uses an explicit compiled template/processor profile. It does not execute arbitrary
Jinja or arbitrary JSON behavior. The raw assets make provenance and parity auditable.

### 11.2 Chat rendering

The existing C++ renderer contains checkpoint-specific system/developer and thinking-history
behavior. Registry work must compare each compiled template profile to its source Jinja on canonical
text, tools, thinking-preservation, system-role, image, video, and tool-result fixtures.

Protocol adapters may intentionally normalize external roles before rendering, for example folding
OpenAI `developer` or Anthropic system reminders. Such behavior remains a documented serving
contract and is not presented as literal source-Jinja behavior.

### 11.3 Model identity

The artifact's validated checkpoint profile selects the compiled canonical model id. Serving
behavior becomes:

- `/v1/models` lists exactly the loaded canonical id;
- `/v1/models/{id}` accepts exactly that canonical id;
- request `model` must equal the loaded canonical id;
- CLI load summaries print model and packing profile;
- no default `qwen3.6-27b` identity is silently used for a 35B artifact.

The initial registry surface has no serving aliases. The current project-owned `--model-id` override
is removed because it can make the server advertise an identity different from the loaded artifact.

Any change to OpenAI/Anthropic schema behavior is updated in schema tests and
[`../serving.md`](../serving.md) in the same phase.

### 11.4 Benchmark and profiler identity

`qus_bench` JSON/CSV/table output adds:

- execution profile id and model-schema digest;
- checkpoint profile/schema digest and canonical model identity;
- packing target, packing profile, and packing-schema digest;
- artifact format and content identity;
- component/storage residency and exact command/environment identity;
- optional profile-specific statistics.

The Qwen35 extension adds resident expert-bank bytes, active expert bytes per token, and MoE regime/
routing statistics. Dense or non-MoE reports do not emit meaningless zero-valued expert fields.

27B and 35B reports are distinct baselines. No performance statement compares reports that differ
in model, artifact, commit, dirty state, command, context, KV dtype, graph mode, or MTP mode without
making that difference explicit.

## 12. Ownership And Source Impact

| Area | Owner | Required change |
|---|---|---|
| execution/checkpoint registry | L2/runtime boundary | exact ids, schemas, ModelInfo, handlers, factory |
| q5090 v5 core parser | L0 | generic components/storage/views/assets/provenance and framing safety |
| storage registry | L0/L1 boundary | finite codec/layout/view descriptors and checked formulas |
| compiled profile validator | L2-facing artifact validation | exact slots, aliases, assets, packing before CUDA |
| generic Weight views | L0 | non-owning storage/view descriptors and typed binding input |
| Qwen expert bank | L1/L2 boundary | direct matrix-array descriptor and MoE consumption |
| converter and verifier | offline tools | exact source adapters, generic v5 writer, both Qwen artifacts |
| linear codec/dispatch | L1 | expert layout addressing and new finite shape families |
| router and MoE | L1 | mathematical APIs, decode/small-T/prefill launchers |
| Text/MTP schedules | L2 | dense/MoE FFN structures and concrete cards |
| cache/state/work planner | runtime + L0 sizing helpers | profile-derived exact allocation |
| Vision | L2 | profile output width and binding validation |
| processor | media/model boundary | explicit multimodal profile |
| tokenizer/template | text | checkpoint/profile-aware assets and renderer |
| OpenAI/Anthropic | serve | loaded model identity and template-visible behavior |
| Python reference/parity | tools | v5/expert/profile support and independent oracle |
| benchmarks/profilers | bench | MoE entrypoints, profile identity, new matrices |
| active documentation | docs | static-registry scope, both Qwen profiles, v5, serving and kernel workflow |

L1 must remove any dependency on the global model card. Operator wrappers validate relationships
among their arguments or receive explicit semantic dimensions; they do not include `model::kCfg`.

## 13. Shared Coordination Points

The coordination gates are staged. Items 1–6 freeze in Phase 1 before the Phase 2 core migration can
diverge. Items 7–9 freeze during Phase 3 before Qwen35 converter and kernel work can diverge. Item 10
freezes in Phase 4 only after real weight/activation/model quality evidence:

1. generic header and component/storage/view/asset/provenance record layouts;
2. execution/checkpoint/packing/target id allocation, checkpoint template/processor identity, and
   canonical schema-hash encoding;
3. codec/layout/view descriptor versions, feature bits, and unknown-id policy;
4. profile-scoped slot generation and cross-language schema tables;
5. Qwen27/Qwen35 component, tensor, asset, and dependency inventories;
6. core artifact/benchmark/report identity fields;
7. final Qwen35 matrix-array physical layout;
8. router and expert numerical rounding contract;
9. public MoE operator inputs/outputs and workspace ownership;
10. final 35B packing/qtype policy and packing-schema digest.

Converter, parser, Python reader, C++ loader, model binding, kernels, and tests must consume the same
frozen definitions. Temporary duplicate definitions used during development must be reconciled before
the phase is complete.

## 14. Dependencies And Delivery Sequence

The dependency chain is:

```text
exact model facts
  -> model-neutral core records, identity, and schema-hash contract
  -> v5 core ABI freeze
  -> generic parser/reader/writer and 27B migration
  -> matrix-array layout benchmark and Qwen35 packing extension
  -> bounded expert tooling and MoE numerical operators
  -> real 35B artifact
  -> 35B Text schedule and resource plan
  -> MTP/Vision/frontends
  -> graph/performance qualification
  -> active-doc replacement and plan archive
```

Kernel prototypes may use synthetic expert banks before the real artifact exists, but no final
kernel or converter contract may invent a different bank layout independently.

## 15. Implementation Phases

### Phase 0: Freeze the source facts and 27B baseline

**Behavioral boundary:** later migration and performance claims have an immutable source identity,
an approved numerical oracle, and a reproducible pre-change 27B baseline.

Tasks:

- record the config and safetensors-index hashes for both local BF16 checkpoints;
- record the current v4.2 artifact hash, manifest, converter identity, git commit, dirty state, GPU,
  CUDA, and complete commands;
- add a small diagnostic activation-tap harness around the existing model `FileTap` and independent
  HF reference, then capture current 27B mixer/FFN/layer/final-logit evidence;
- capture the current 27B weight/output parity evidence and the `smoke` and `core` benchmark matrices;
- freeze the 27B numerical tolerances and the allowed performance/memory regression thresholds
  before introducing the static-registry boundary;
- state which long-running real-artifact gates are mandatory at each later milestone.

Verification and evidence capture:

```bash
PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
MODEL27=/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16
MODEL35=/home/neroued/models/llm/qwen/Qwen3.6-35B-A3B/base-hf-bf16
WEIGHTS27_V4=out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus
BASELINE=profiles/bench/static-registry-baseline

test -d "$MODEL27" && test -d "$MODEL35" && test -s "$WEIGHTS27_V4"
mkdir -p "$BASELINE"
git rev-parse HEAD > "$BASELINE/git-commit.txt"
git status --short > "$BASELINE/git-status.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader \
  > "$BASELINE/gpu.txt"
nvcc --version > "$BASELINE/cuda-toolkit.txt"
nsys --version > "$BASELINE/nsys-version.txt"
ncu --version > "$BASELINE/ncu-version.txt"
sha256sum "$MODEL27/config.json" "$MODEL27/model.safetensors.index.json" \
  "$MODEL35/config.json" "$MODEL35/model.safetensors.index.json" \
  "$WEIGHTS27_V4" "$WEIGHTS27_V4.manifest.json" > "$BASELINE/input-sha256.txt"

$PYTHON -m tools.q5090_convert.verify "$WEIGHTS27_V4" --model "$MODEL27"
$PYTHON tools/parity/block_parity.py --weights "$WEIGHTS27_V4" --hf "$MODEL27"
cmake --build build -j --target qus_profile_tap
$PYTHON tools/parity/run_profile_parity.py --weights "$WEIGHTS27_V4" --hf "$MODEL27" \
  --cpp ./build/tests/qus_profile_tap --token-ids 1,2,3,4 --phases prefill,decode \
  --output-dir "$BASELINE/activation-parity"
./build/src/qus "$WEIGHTS27_V4" --prompt '你好' --max-context 256 --max-new 16 \
  --no-thinking --greedy --print-token-ids \
  > "$BASELINE/greedy.stdout" 2> "$BASELINE/greedy.stderr"
$PYTHON tools/bench/run_qus_bench_matrix.py --preset smoke --weights "$WEIGHTS27_V4" \
  --output-dir "$BASELINE/smoke"
$PYTHON tools/bench/run_qus_bench_matrix.py --preset core --weights "$WEIGHTS27_V4" \
  --output-dir "$BASELINE/core"
nsys profile --force-overwrite=true --stats=false --trace=cuda,nvtx,osrt --sample=none \
  -o "$BASELINE/qwen36-27b-v4-k3" ./build/bench/qus_bench --weights "$WEIGHTS27_V4" \
  -pg 2048,128 -r 1 --warmup 1 --mtp-draft-tokens 3
```

The `core` matrix and real-model parity are deliberate milestone work, not ordinary edit-loop tests.

Definition of done:

- every later before/after claim names this exact artifact and source identity;
- numerical, memory, latency, and throughput acceptance thresholds are recorded before migration;
- no archived profiler report is treated as the current baseline.

### Phase 1: Freeze the model-neutral v5 core and compiled schema contract

**Behavioral boundary:** the candidate container describes safe storage and exact profile identity
without describing or executing a Qwen, Gemma, dense, MoE, Text, Vision, or MTP graph.

Tasks:

- write a self-contained candidate v5 specification owned by this plan, without calling it current
  or normative before the Phase 2 cutover;
- freeze generic header, component, storage-object, tensor-view, asset, provenance, and string records;
- freeze execution/checkpoint/target/packing identity fields, id allocation, required feature bits,
  and canonical language-independent schema serialization/hashing;
- freeze finite dtype/codec/layout/view descriptor envelopes and unknown-id/version behavior;
- define the two Qwen execution/checkpoint schemas and exact component/tensor/asset inventories;
- define source-key required/recognized-dropped/forbidden policy;
- define structural, storage-support, exact-profile, and content-validation matrices without adding a
  second accepted parser to the current runtime;
- create a tiny test-only Gemma-shaped contract fixture that exercises rank-0 BF16, rank-1 I64,
  rank-3 contiguous tensors, rank-4 convolution, rank-5 logical-to-packed reshape views, tied/shared storage,
  absent semantic slots, non-Qwen components, and non-Qwen assets;
- define v5 manifest/report identity and evolution rules.

The Gemma-shaped fixture is a binary-contract oracle, not a registered production model and not an
inference test. It must not add Gemma enums or branches to L0.

Verification:

```bash
PYTHON=/home/neroued/miniconda3/envs/py311/bin/python

$PYTHON -m py_compile $(rg --files tools/q5090_convert tools/q5090 | rg '\.py$')
$PYTHON -m pytest tools/q5090_convert/tests/test_schema_contract.py \
  tools/q5090/tests/test_model_neutral_fixture.py
cmake --build build -j --target qus_schema_contract_test
ctest --test-dir build -R '^qus_schema_contract_test$' --output-on-failure
git diff --check
```

Definition of done:

- the candidate v5 contract is self-contained and implementable;
- no core record contains layer topology, Qwen dimensions, model-specific component/role enums, or
  executable metadata;
- canonical schema bytes and hashes are deterministic across Python and C++ specifications;
- parser safety can be established without trusting or recognizing an execution profile;
- profile validity and binding do not depend on tensor/component/asset names;
- the Gemma-shaped fixture requires no change to the core record ABI;
- matrix-array/expert layout remains a storage extension decision for Phase 3 and does not define the
  core container.

### Phase 2: Migrate the tooling and 27B artifact

**Behavioral boundary:** the existing 27B model runs exclusively from a v5 artifact through the new
profile boundary with unchanged model behavior.

Tasks:

- implement the generic v5 serializer, reader, verifier, manifest, and schema-hash generator;
- convert tensor-plan globals into a Qwen27 source adapter, compiled artifact schema, and packing plan;
- convert parser/loader/WeightStore to generic component/storage/view/asset records and v5 only;
- implement framing, storage-support, exact-profile, and content validators with the frozen malformed
  matrix;
- generate identical profile/packing slot tables and schema digests for Python and C++;
- implement the Engine profile factory with only the dense profile enabled initially;
- resolve every numeric tensor/asset slot into typed Qwen27 bindings before Engine construction;
- move cache/work/step planning behind the dense profile;
- regenerate the 27B v5 artifact;
- promote the candidate v5 specification and archive v4 in the same finishing change that makes v5
  the only accepted format; update every active link at that cutover;
- remove v4 runtime/converter branches and retired fixtures;
- update `AGENTS.md` and every real-test, MTP/Vision E2E, benchmark-runner, tool, and documentation
  consumer of the canonical 27B artifact path in that same cutover;
- replace the misleading weight-only `block_parity.py` name with a v5 weight-parity tool and port the
  Phase 0 C++/Python activation-tap driver to profile-aware v5 block, layer, and final-logit parity;
- make real-file tests accept an explicit artifact/profile and register a `profile-27b` CTest label;
- replace fixture tests' ambient `python3` subprocesses with the canonical Python selected by CMake.

The migration tests also include a test-only unknown execution profile using the Phase 1 foreign
fixture. Generic inspection must complete structural checks, while Engine loading must return an
explicit unsupported-profile error before CUDA initialization or H2D transfer.

Verification:

```bash
PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
MODEL=/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16
WEIGHTS=out/qwen3_6_27b.q5090_mixed_v5.qus

mkdir -p profiles/bench
$PYTHON -m py_compile $(rg --files tools/q5090_convert tools/q5090 tools/parity | rg '\.py$')
$PYTHON -m pytest tools/q5090_convert/tests tools/q5090/tests
$PYTHON -m tools.q5090_convert.convert --model "$MODEL" --tokenizer "$MODEL" --out "$WEIGHTS"
test -s "$WEIGHTS" && test -s "$WEIGHTS.manifest.json"
$PYTHON -m tools.q5090_convert.verify "$WEIGHTS" --quick
$PYTHON -m tools.q5090_convert.verify "$WEIGHTS" --model "$MODEL"
$PYTHON tools/parity/weight_parity.py --weights "$WEIGHTS" --hf "$MODEL"

cmake --build build -j --target qus_q5090_parser_test qus_q5090_pack_golden_test \
  qus_weight_store_test qus_weight_store_real_file_test qus_artifact_profile_gate_test \
  qus_model_bind_test qus_model_blocks_test qus_engine_real_file_test qus_profile_tap qus_bench
ctest --test-dir build -R \
  'qus_(q5090_parser|q5090_pack_golden|weight_store|weight_store_real_file|artifact_profile_gate|model_bind|model_blocks|engine_real_file)_test' \
  --output-on-failure
ctest --test-dir build -L profile-27b --output-on-failure
$PYTHON tools/parity/run_profile_parity.py --profile qwen3_6_dense_27b_v1 \
  --weights "$WEIGHTS" --hf "$MODEL" --cpp ./build/tests/qus_profile_tap \
  --token-ids 1,2,3,4 --phases prefill,decode \
  --output-dir profiles/bench/qwen36-27b-v5-parity

cmake --build build -j
ctest --test-dir build --output-on-failure
$PYTHON tools/bench/run_qus_bench_matrix.py --preset core --weights "$WEIGHTS" \
  --output-dir profiles/bench/qwen36-27b-v5-core

rg -n --glob '!docs/archive/**' \
  '(q5090[_ -]?v4|format_v4|qwen3_6_27b\.q5090_w4g64_mixed_v4_2)' \
  AGENTS.md README.md docs bench tests tools
```

The generic parser/storage suites cover section/reference overflow, rank zero through the fixed cap,
storage/view containment, shared/overlapping immutable views, asset bounds, descriptor versions,
unknown codec/layout/view ids, padding, and CRC. The test-only foreign profile is structurally
inspectable but not loadable. The artifact-profile gate separately covers execution/checkpoint/
packing ids and schema-digest mismatches. The Qwen profile suites cover
missing/duplicate/unknown slots, wrong layer-owned shapes, top-k zero/overflow, wrong component
dependencies, and Vision/draft hidden mismatch. Matrix-array malformed cases enter in Phase 3 with
the layout extension.

Definition of done:

- no executable or tool accepts v4;
- the active format specification is v5, the v4 specification is archived, and no active link calls
  v4 current;
- an artifact with the optional `DebugSymbol` section removed still binds, loads, and reports the
  same canonical identity;
- unknown profiles and unsupported storage primitives fail before CUDA/H2D with no fallback;
- the core parser and generic WeightStore do not interpret Qwen component or tensor semantics;
- every Qwen27 binding is complete, duplicate-free, numeric, and typed before Engine construction;
- all canonical 27B consumers use the explicit v5 path and a full `ctest` has exercised the cutover;
- 27B block/model/end-to-end behavior matches the approved pre-migration oracle;
- 27B cache/workspace high-water behavior remains bounded by prefill chunk;
- the before/after `qus_bench` matrix shows no unexplained regression;
- report output names the exact execution/checkpoint/schema/target/packing tuple.

### Phase 3: Implement expert-bank tooling and MoE operators

**Behavioral boundary:** a synthetic or converted expert bank produces numerically correct Qwen MoE
outputs for decode, small-T, and prefill without host-selected routing.

Tasks:

- implement the candidate matrix-array layout inside the frozen v5 storage descriptor envelope;
- benchmark bank-local versus flat-global planes at the real gate-up/down shapes and freeze the
  layout id/version only after decode, small-T, and grouped-prefill evidence;
- add the selected layout id/version and normative byte formula to the active v5 specification in
  the same change that teaches reader/writer/parser/loader support;
- implement bounded expert-bank encode/decode/verification;
- implement generic `MatrixArrayWeight`, then the Qwen `ExpertWeightBank` typed view and addressing;
- add one risk-centered `qus_sparse_moe_test` contract suite with independent router, selected
  expert, shared expert, and reduction oracles;
- add shared expert and routed expert contracts;
- implement correct fallbacks for T=1, T=2..6, and prefill;
- add per-operator benchmarks for routing and expert execution;
- add real 35B attention/GDN/linear shapes to existing dispatch tests and benchmarks;
- run sanitizer coverage for first/last expert, skewed routing, and repeated workspace reuse.

Intended verification targets:

```bash
PYTHON=/home/neroued/miniconda3/envs/py311/bin/python

$PYTHON -m py_compile $(rg --files tools/q5090_convert tools/q5090 tools/parity | rg '\.py$')
$PYTHON -m pytest tools/q5090_convert/tests tools/q5090/tests
cmake --build build -j --target qus_sparse_moe_test qus_sparse_moe_bench \
  qus_expert_layout_bench qus_q5090_parser_test qus_q5090_pack_golden_test \
  qus_weight_store_test qus_linear_test \
  qus_rmsnorm_test qus_gqa_attention_test qus_gdn_gating_test qus_gdn_gating_proj_test \
  qus_gated_delta_rule_test qus_causal_conv1d_silu_test
ctest --test-dir build -R \
  'qus_(sparse_moe|q5090_parser|q5090_pack_golden|weight_store|linear|rmsnorm|gqa_attention|gdn_gating|gdn_gating_proj|gated_delta_rule|causal_conv1d_silu)_test' \
  --output-on-failure
./build/bench/qus_expert_layout_bench --layouts matrix-local,flat-global \
  --arrays 256 --rows 1024 --cols 2048 --qtype q4g64 \
  --tokens 1,2,6,128,1024 --routing uniform,skewed
./build/bench/qus_expert_layout_bench --layouts matrix-local,flat-global \
  --arrays 256 --rows 2048 --cols 512 --qtype q5g64 \
  --tokens 1,2,6,128,1024 --routing uniform,skewed
compute-sanitizer --tool memcheck --error-exitcode=99 ./build/tests/qus_sparse_moe_test
compute-sanitizer --tool racecheck --error-exitcode=99 ./build/tests/qus_sparse_moe_test
./build/bench/qus_sparse_moe_bench --profile qwen3_6_35b_a3b \
  --tokens 1,2,6,128,1024 --routing uniform,skewed
```

If grouped prefill introduces count/offset/permutation storage that can expose uninitialized reads,
the same suite also runs under `compute-sanitizer --tool initcheck`.

Numerical cases include:

- exact router ids/weights away from and near top-k boundaries;
- deterministic tie behavior;
- selected expert ids 0 and 255;
- all token assignments concentrated in eight experts;
- uniform and sparse expert histograms;
- independent gate/up/down and shared-gate oracle comparison;
- BF16 rounding and accumulation stress;
- repeated arena mark/rewind and graph-stable addresses.

Definition of done:

- the public contract contains no CUDA strategy fields;
- the selected matrix-array layout has real-shape evidence and does not alter any core v5 record;
- no hot path reads routing state to the host;
- every regime matches the independent oracle within its approved tolerance;
- sanitizer finds no OOB, stale workspace, or bank-stride error;
- benchmark output identifies layout, qtypes, T regime, and active bytes.

### Phase 4: Produce and bind the real 35B Text artifact

**Behavioral boundary:** 35B Text prefill and ordinary eager decode run from the real packed artifact
with correct KV/GDN state and no MTP or Vision dependency.

Tasks:

- implement the MoE Text tensor plan and apply the Phase 2 exact-profile validator to the complete
  real inventory;
- stream-convert the real 35B source;
- evaluate candidate expert/shared qtypes at weight, activation, layer, and short-generation levels;
  freeze the 35B packing profile id only after acceptance and regenerate the canonical artifact if
  the accepted policy differs from the candidate;
- implement the MoE model card and weight binding;
- implement profile-derived KV, GDN, StepState, and workspace planning;
- add 35B attention/GDN slot maps;
- implement Text prefill, greedy decode, sampling, and prefix reset/append behavior;
- establish block, layer, and short-generation parity.

Verification:

```bash
PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
MODEL=/home/neroued/models/llm/qwen/Qwen3.6-35B-A3B/base-hf-bf16
WEIGHTS=out/qwen3_6_35b_a3b.q5090_mixed_v5.qus

$PYTHON -m py_compile $(rg --files tools/q5090_convert tools/q5090 tools/parity | rg '\.py$')
$PYTHON -m pytest tools/q5090_convert/tests tools/q5090/tests
$PYTHON -m tools.q5090_convert.convert --model "$MODEL" --tokenizer "$MODEL" --out "$WEIGHTS"
test -s "$WEIGHTS" && test -s "$WEIGHTS.manifest.json"
$PYTHON -m tools.q5090_convert.verify "$WEIGHTS" --quick
$PYTHON -m tools.q5090_convert.verify "$WEIGHTS" --model "$MODEL"
$PYTHON tools/parity/weight_parity.py --weights "$WEIGHTS" --hf "$MODEL" --rows-per-chunk 256

cmake --build build -j --target qus_q5090_parser_test qus_q5090_pack_golden_test \
  qus_weight_store_test qus_weight_store_real_file_test qus_model_bind_test qus_model_blocks_test \
  qus_engine_real_file_test qus_runtime_file_tap_test qus_profile_tap qus_bench
ctest --test-dir build -R \
  'qus_(q5090_parser|q5090_pack_golden|weight_store|weight_store_real_file|model_bind|model_blocks|engine_real_file|runtime_file_tap)_test' \
  --output-on-failure
ctest --test-dir build -L profile-35b-a3b --output-on-failure
$PYTHON tools/parity/run_profile_parity.py --profile qwen3_6_moe_35b_a3b_v1 \
  --weights "$WEIGHTS" --hf "$MODEL" --cpp ./build/tests/qus_profile_tap \
  --token-ids 1,2,3,4 --phases prefill,decode \
  --output-dir profiles/bench/qwen36-35b-a3b-text-parity
./build/bench/qus_bench --weights "$WEIGHTS" -p 128,512,2048 -n 128 \
  -pg '2048,128' -r 5 --warmup 1 -o json \
  --output-file profiles/bench/qwen36-35b-a3b-text-eager.json --no-cuda-graph
```

Definition of done:

- the complete artifact passes structural, CRC, padding, plan, and value verification;
- every Text layer has exactly one valid router, two banks, one shared expert, and one shared gate;
- eager prefill/decode matches the approved Python/HF oracle at block, layer, and observable output
  levels;
- weight-only parity is reported separately and is never used as evidence for activation parity;
- the canonical artifact names a packing profile whose qtype policy has passed the accepted quality
  gates;
- cache/state/workspace planning matches actual high-water use;
- no 27B constant is used to interpret a 35B tensor or state slot.

### Phase 5: Add CUDA Graphs and qualify MoE performance

**Behavioral boundary:** ordinary 35B decode and Text prefill meet correctness and stable-address
requirements under their production execution paths.

Tasks:

- capture/replay ordinary MoE decode;
- optimize selected-expert decode only after NSYS attribution;
- optimize small-T and grouped prefill paths only from benchmark evidence;
- profile the frozen bank layout in complete inference and tune kernels without changing its v5
  addressing contract;
- validate BF16 and INT8 KV where supported;
- add stable NVTX ranges for routing, routed experts, shared expert, and reduction if useful to
  whole-inference attribution.

Verification:

```bash
WEIGHTS=out/qwen3_6_35b_a3b.q5090_mixed_v5.qus

cmake --build build -j --target qus_moe_graph_replay_test qus_engine_real_file_test qus qus_bench
ctest --test-dir build -R 'qus_(moe_graph_replay|engine_real_file)_test' --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode=99 \
  ./build/tests/qus_moe_graph_replay_test

./build/src/qus "$WEIGHTS" --prompt '你好' --max-context 256 --max-new 16 \
  --no-thinking --greedy --print-token-ids --no-cuda-graph \
  > /tmp/qwen36-35b-eager.out 2> /tmp/qwen36-35b-eager.err
./build/src/qus "$WEIGHTS" --prompt '你好' --max-context 256 --max-new 16 \
  --no-thinking --greedy --print-token-ids \
  > /tmp/qwen36-35b-graph.out 2> /tmp/qwen36-35b-graph.err
diff -u /tmp/qwen36-35b-eager.out /tmp/qwen36-35b-graph.out

./build/bench/qus_bench --weights "$WEIGHTS" -p 128,512,2048 -n 128 \
  -pg '2048,128' -r 5 --warmup 1 -o json \
  --output-file profiles/bench/qwen36-35b-a3b-graph.json
nsys profile --force-overwrite=true --stats=false --trace=cuda,nvtx,osrt --sample=none \
  --output profiles/nsys/qwen36-35b-a3b-decode \
  ./build/bench/qus_bench --weights "$WEIGHTS" -pg 2048,128 -r 1 --warmup 1
```

`qus_moe_graph_replay_test` captures once, then replays repeated inputs that force different and
highly skewed expert sets while checking stable addresses and eager/replay numerical equivalence.
The real CLI diff proves observable greedy equivalence for the packed 35B artifact; neither check is
replaced by the latency benchmark.

NCU commands are selected only after the NSYS report identifies a specific kernel. The exact kernel
shape and metric set are recorded with the resulting report rather than predeclared generically.

Definition of done:

- eager and graph outputs agree;
- expert identity changes do not trigger graph recapture or host branching;
- no capture-invalid allocation or address movement occurs;
- NSYS explains the caller-visible Text latency breakdown;
- any kernel performance claim has matching NCU and end-to-end `qus_bench` evidence.

### Phase 6: Add MoE MTP and Vision

**Behavioral boundary:** the complete 35B Text/MTP/Vision model works under eager and graph execution,
and speculative decoding preserves the target distribution.

Tasks:

- bind and execute the MoE MTP layer in prompt, proposal, and verify modes;
- route MTP prompt preparation through the grouped/prefill MoE path at bounded
  `mtp_prefill_chunk`, while proposal and verification use the small-T path;
- allocate MTP KV and GDN snapshot slots from the profile;
- capture/replay full MTP rounds;
- regenerate and evaluate a width-2048 shortlisted draft head;
- bind the 35B Vision weights and width-2048 merger;
- add one canonical small RGB Vision fixture shared by the C++ and Python activation-parity drivers;
- validate visual scatter and MRoPE continuation;
- preserve target correctness independently of draft-head quality.

Verification:

```bash
PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
MODEL=/home/neroued/models/llm/qwen/Qwen3.6-35B-A3B/base-hf-bf16
WEIGHTS=out/qwen3_6_35b_a3b.q5090_mixed_v5.qus

cmake --build build -j --target qus_engine_mtp_e2e_test qus_engine_vision_e2e_test \
  qus_mtp_round_test qus_mtp_prefill_moe_test qus_vision_support_test qus_profile_tap qus qus_bench
ctest --test-dir build -R \
  'qus_(engine_mtp_e2e|engine_vision_e2e|mtp_round|mtp_prefill_moe|vision_support)_test' \
  --output-on-failure
ctest --test-dir build -L profile-35b-a3b \
  -R 'engine_(mtp|vision)_e2e' --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode=99 ./build/tests/qus_engine_mtp_e2e_test
$PYTHON tools/parity/run_profile_parity.py --profile qwen3_6_moe_35b_a3b_v1 \
  --weights "$WEIGHTS" --hf "$MODEL" --cpp ./build/tests/qus_profile_tap \
  --token-ids 1,2,3,4 --phases mtp-prefill,mtp-propose,mtp-verify,vision \
  --vision-fixture tests/fixtures/media/qwen36_vision_reference.ppm \
  --output-dir profiles/bench/qwen36-35b-a3b-mtp-vision-parity

./build/src/qus "$WEIGHTS" --prompt '你好' --max-context 256 --max-new 16 \
  --no-thinking --greedy --print-token-ids --mtp-draft-tokens 3 --no-cuda-graph \
  > /tmp/qwen36-35b-mtp-eager.out 2> /tmp/qwen36-35b-mtp-eager.err
./build/src/qus "$WEIGHTS" --prompt '你好' --max-context 256 --max-new 16 \
  --no-thinking --greedy --print-token-ids --mtp-draft-tokens 3 \
  > /tmp/qwen36-35b-mtp-graph.out 2> /tmp/qwen36-35b-mtp-graph.err
diff -u /tmp/qwen36-35b-mtp-eager.out /tmp/qwen36-35b-mtp-graph.out

./build/bench/qus_bench --weights "$WEIGHTS" -p 512 -n 256 -r 5 --warmup 1 \
  --mtp-draft-tokens 5 -o json \
  --output-file profiles/bench/qwen36-35b-a3b-mtp-full-head.json
./build/bench/qus_bench --weights "$WEIGHTS" -p 512 -n 256 -r 5 --warmup 1 \
  --mtp-draft-tokens 5 --lm-head-draft -o json \
  --output-file profiles/bench/qwen36-35b-a3b-mtp-draft-head.json
```

Definition of done:

- MTP target verification and state-slot commit are correct for rejection at every candidate
  position and context-capacity fallback;
- MTP prompt preparation matches the reference across multiple chunk boundaries and never uses the
  small-T proposal path;
- sampled MTP preserves the target distribution;
- the draft head affects acceptance/performance only;
- Vision block/merger output matches the independent reference;
- visual embeddings are exactly width 2048 and decode continues with correct MRoPE state;
- sanitizer finds no state-slot, rewind, or multimodal workspace lifetime error.

### Phase 7: Complete frontends, serving, reports, and active docs

**Behavioral boundary:** both profiles are complete user-visible products under CLI, OpenAI, and
Anthropic surfaces, and active documentation describes only implemented behavior.

Tasks:

- implement and verify checkpoint-specific template/processor profiles;
- report the loaded canonical model id in CLI and HTTP model endpoints;
- remove the caller-selected `--model-id` override and reject every request/lookup identity other
  than the artifact's canonical id;
- update tokenizer/template/processor fixtures where they protect observable behavior;
- update benchmark report schema and downstream tests;
- add a stdlib-only loopback serving smoke driver that starts one server at a time and verifies
  model listing/lookup, wrong-model rejection, OpenAI, Anthropic, tools, thinking, and image cases;
- update README, design, registered-model references, kernel guide, serving guide, tools guides, and test
  guides;
- update `AGENTS.md` canonical paths for both supported real artifacts;
- archive this completed plan and its dated evidence.

Verification:

```bash
PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
WEIGHTS27=out/qwen3_6_27b.q5090_mixed_v5.qus
WEIGHTS35=out/qwen3_6_35b_a3b.q5090_mixed_v5.qus

cmake --build build -j --target qus qus-serve qus_qwen_chat_template_test \
  qus_qwen_text_tokenizer_test qus_qwen_text_runner_test qus_processor_test \
  qus_openai_schema_test qus_anthropic_schema_test qus_bench_support_test
ctest --test-dir build -R \
  'qus_(qwen_chat_template|qwen_text_tokenizer|qwen_text_runner|processor|openai_schema|anthropic_schema|bench_support)_test' \
  --output-on-failure
ctest --test-dir build --output-on-failure

./build/src/qus "$WEIGHTS27" --prompt '你好' --max-context 256 --max-new 16 \
  --no-thinking --greedy --print-token-ids > /tmp/qwen36-27b-cli.out
./build/src/qus "$WEIGHTS35" --prompt '你好' --max-context 256 --max-new 16 \
  --no-thinking --greedy --print-token-ids > /tmp/qwen36-35b-a3b-cli.out

$PYTHON tools/serving/profile_smoke.py --server ./build/src/qus-serve \
  --weights "$WEIGHTS27" --model qwen3.6-27b --port 18080 \
  --cases models,identity,openai,anthropic,tools,thinking,image \
  --image tests/fixtures/media/qwen36_vision_reference.ppm
$PYTHON tools/serving/profile_smoke.py --server ./build/src/qus-serve \
  --weights "$WEIGHTS35" --model qwen3.6-35b-a3b --port 18081 \
  --cases models,identity,openai,anthropic,tools,thinking,image \
  --image tests/fixtures/media/qwen36_vision_reference.ppm

rg -n --glob '!docs/archive/**' \
  '(q5090[_ -]?v4|format_v4|fixed Qwen3\.6-27B|only.*27B|single.*27B)' \
  AGENTS.md README.md docs bench tests tools

$PYTHON - <<'PY'
import re
import subprocess
import urllib.parse
from pathlib import Path

files = [
    Path(p)
    for p in subprocess.check_output(["git", "ls-files", "*.md"], text=True).splitlines()
    if not p.startswith("docs/archive/")
]
pattern = re.compile(r"!?\[[^\]\n]*\]\((<[^>\n]+>|[^)\s\n]+)")
missing = []
for source in files:
    text = re.sub(r"```.*?```", "", source.read_text(encoding="utf-8"), flags=re.S)
    for raw in pattern.findall(text):
        target = raw.strip("<>")
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        local = urllib.parse.unquote(target.split("#", 1)[0])
        if local and not (source.parent / local).exists():
            missing.append((source, target))
for source, target in missing:
    print(f"{source}: missing local link {target}")
raise SystemExit(bool(missing))
PY

git diff --check
```

Before running these commands, compare them with each newly built executable's current `--help` and
update this plan if the implemented option surface deliberately changed.

Definition of done:

- each artifact uses its own tokenizer/template/processor identity;
- CLI and server advertise the actual loaded model;
- OpenAI/Anthropic behavior and documentation agree;
- report consumers can distinguish execution/checkpoint/schema/target/packing tuples;
- all active documentation links resolve and no active document calls v4 or fixed-27B behavior
  current;
- the plan and evidence are archived.

## 16. Test And Evidence Matrix

Only tests that protect an allowed project risk are added.

| Risk | Required evidence |
|---|---|
| v5 framing/bounds/reference safety | model-neutral binary parser malformed-input tests |
| unknown execution profile | complete structural inspection followed by pre-CUDA load rejection |
| schema drift | cross-language canonical bytes/hash goldens and exact mismatch rejection |
| logical/physical tensor separation | rank-0..rank-cap, view containment, alias/tied/shared fixtures |
| unsupported storage primitive | explicit codec/layout/view rejection with no fallback or H2D |
| component/asset extensibility | non-Qwen slot fixture and exact required/optional dependency checks |
| source/runtime inventory difference | required/dropped/forbidden source-key adapter fixtures |
| expert bank code/scale addressing | bit-identical quantizer/decoder oracle at real shapes |
| profile mismatch | parser/schema rejection fixtures |
| router/top-k | FP32 mathematical oracle, boundary/tie stress |
| routed/shared MoE | BF16-rounded operator oracle for T=1, small-T, prefill |
| attention/GDN new geometry | existing operator oracles at 35B real shapes |
| layer schedule | block/layer parity and observable model output, not source scans |
| workspace/state lifetime | repeated execution, high-water reports, compute-sanitizer |
| CUDA Graph | eager/replay output equivalence and stable-address execution |
| MTP state/target correctness | rejection-position, capacity, sampled-distribution E2E cases |
| Vision width/MRoPE | real Vision/processor parity and multimodal E2E behavior |
| tokenizer/template | canonical rendered-token fixtures per profile |
| serving | OpenAI/Anthropic schema tests and real responses |
| performance | per-op bench, NSYS, targeted NCU, before/after `qus_bench` |

Tests must not scan source for preferred symbols, lock private class layout, preserve v4 compatibility,
or duplicate coverage merely because a second profile exists. Shared tests should be table-driven only
where the observable contract is genuinely shared; profile-specific numerical fixtures remain
explicit. In particular, registry work must not grow a model-config test into a list of trivial
constant assertions; exact dimensions are protected by artifact/profile schema rejection, real
weight binding, and numerical model execution. Any old constant-only coverage that protects no
observable risk is removed rather than replicated for 35B. The requirement that core records remain
model-neutral is enforced by ABI review plus the non-Qwen binary fixture, never by scanning source
files for forbidden words.

### 16.1 Real-artifact gate policy

The current real-file tests are not yet a multi-profile gate: they hard-code the 27B v4.2 path and return
success after printing `SKIP` when an artifact, GPU, or sufficient memory is absent. Phase 2 must
replace that behavior with an explicit registration contract:

- every real-file executable accepts `--weights <path>`, `--expect-execution-profile <id>`, and
  `--expect-checkpoint-profile <id>`;
- CMake registers separately named tests with `profile-27b` or `profile-35b-a3b` labels and explicit
  canonical artifact paths; no test chooses a file by glob, modification time, or `latest`;
- a developer without a large local prerequisite may still receive a clearly reported skip;
- a formal milestone/release gate first proves both artifacts exist and then treats every skip as a
  failure;
- fixture generators receive the canonical Python 3.11 executable from CMake instead of invoking an
  ambient `python3`.

A formal two-profile gate therefore includes:

```bash
WEIGHTS27=out/qwen3_6_27b.q5090_mixed_v5.qus
WEIGHTS35=out/qwen3_6_35b_a3b.q5090_mixed_v5.qus
LOGDIR=profiles/bench/static-registry-final

test -s "$WEIGHTS27" && test -s "$WEIGHTS27.manifest.json"
test -s "$WEIGHTS35" && test -s "$WEIGHTS35.manifest.json"
mkdir -p "$LOGDIR"
set -o pipefail
ctest --test-dir build -V -L 'profile-27b|profile-35b-a3b' | tee "$LOGDIR/ctest.log"

if rg -n 'SKIP:' "$LOGDIR/ctest.log"; then
  echo 'formal multi-profile gate contained a skipped test' >&2
  exit 1
fi
```

Ordinary `ctest` success is not evidence that either real model was loaded unless its verbose log
shows the requested profile and artifact and contains no skip.

## 17. Major Risks And Mitigations

### 17.1 Parser generalization accepts the wrong model

**Risk:** removing fixed 27B checks turns a shape mismatch into silent wrong-weight execution.

**Mitigation:** retain strict L0 structure checks and add a separate exact profile/inventory
validator before any CUDA allocation or upload.

### 17.2 The artifact becomes a graph language

**Risk:** generic Model/Layer records, names, TLVs, or view programs gradually determine operator
order and recreate a dynamic graph inside the file format.

**Mitigation:** core records describe data/storage only; profile slots are opaque; schema files cannot
encode control flow; the handwritten EngineImpl remains the only execution authority.

### 17.3 Schema hashes create false confidence

**Risk:** matching hashes conceal malformed inventory, content corruption, or incompatible bindings.

**Mitigation:** define canonical cross-language hash bytes, then still perform exact per-record schema,
storage, dependency, integrity, and value validation. Hashes are mismatch detectors, not trust.

### 17.4 Optional components hide unsupported storage

**Risk:** an unselected component contains an unknown codec or malformed view and becomes a latent
failure when a later request enables it.

**Mitigation:** validate every present component and storage object before accepting the artifact,
even when residency is optional; unknown storage never falls back or remains dormant.

### 17.5 Cross-architecture abstraction regresses 27B

**Risk:** runtime dimensions, virtual layer calls, or generic kernels increase launch gaps or reduce
specialization, or a Qwen-shaped shared skeleton forces a future model into wrong mathematics.

**Mitigation:** generalize only container/binding boundaries; dispatch once around a whole concrete
EngineImpl; retain 27B finite shape branches; require a before/after 27B NSYS/`qus_bench` gate.

### 17.6 Expert layout is frozen before evidence

**Risk:** a convenient binary layout creates scattered selected-expert loads or poor prefill GEMM.

**Mitigation:** benchmark matrix-local and flat-global candidates at the two real expert shapes before
the matrix-array layout id/version becomes normative.

### 17.7 Router quantization changes discrete expert identity

**Risk:** small linear error produces a different top-k set and large downstream divergence.

**Mitigation:** keep router and shared scalar gate BF16 initially; test top-k overlap and downstream
parity before considering any later quantization.

### 17.8 MTP is accidentally treated as dense

**Risk:** target Text works while proposal preparation uses the wrong MTP FFN.

**Mitigation:** MTP owns an independent profile and exact inventory; all target/MTP modes have taps
and parity evidence.

### 17.9 Dynamic routing breaks CUDA Graphs

**Risk:** host branching, allocation, or variable launch topology invalidates replay.

**Mitigation:** ids/counts are stable-address device data; launch maximum fixed shapes; compare eager
and replay under changing expert selections.

### 17.10 Prefill workspace is underestimated

**Risk:** `T*top_k` permutation/intermediate storage overflows or makes workspace depend on total
prompt length.

**Mitigation:** profile-owned checked sizing from prefill chunk; skew/uniform routing tests; arena
high-water and sanitizer evidence.

### 17.11 GDN state mapping is wrong

**Risk:** old 48-head or ratio-three assumptions corrupt 32-head ratio-two recurrence and snapshot
slots.

**Mitigation:** add real 35B GDN oracle shapes, exact slot maps, repeated prefill/decode, and
state-snapshot parity.

### 17.12 Vision silently emits width 5120

**Risk:** merger/scatter memory corruption or wrong multimodal embeddings.

**Mitigation:** profile schema asserts `vision.out_hidden == text.hidden`; bind and E2E tests cover
the real `[2048,4608]` merger.

### 17.13 Template or tokenizer is assumed global

**Risk:** the model receives a valid token sequence with the wrong conversation semantics.

**Mitigation:** artifact-specific assets, explicit template profile/hash, and per-profile rendered-id
fixtures.

### 17.14 32 GiB feasibility is overstated

**Risk:** packed weights fit in isolation while complete context/workspace/graph residency does not.

**Mitigation:** publish only measured full-process memory matrices and reject unsupported settings
before inference.

### 17.15 Performance optimization changes numerical order

**Risk:** fused routing or reduction changes top-k weights, accumulation, or MTP acceptance.

**Mitigation:** freeze the mathematical rounding contract first; rerun operator, layer, E2E, and
target-distribution gates after every relevant optimization.

## 18. Definition Of Done

The static-registry elevation is complete only when all of the following are true:

- the public project scope names both exact supported profiles and still excludes arbitrary models;
- only q5090 v5 is accepted and its active specification is self-contained;
- the core v5 ABI contains no Qwen/GDN/MTP/Vision/layer-topology field or global model-role enum;
- optional debug names and non-executable metadata can be removed without changing binding,
  execution, or serving identity;
- the Gemma-shaped contract fixture passes generic structural/storage checks and is rejected as an
  unsupported execution profile before CUDA/H2D;
- every accepted artifact matches exact execution/model-schema/checkpoint/checkpoint-schema/target/
  packing identities, and every present storage primitive is supported;
- adding a future compiled profile that uses existing storage primitives requires a new schema,
  source adapter, typed binder, concrete EngineImpl, and qualification, but no core v5 record/parser
  change;
- both real v5 artifacts pass structural, integrity, value, and exact inventory verification;
- the runtime selects a concrete profile from artifact identity before CUDA initialization;
- 27B retains approved numerical behavior and has no unexplained performance regression;
- 35B Text eager and graph paths pass block, layer, and observable-output parity;
- 35B MoE decode, small-T, and prefill paths have real-shape numerical and memory-safety evidence;
- 35B MTP preserves target correctness under eager/graph, greedy/sampled, and draft/full head modes;
- 35B Vision and multimodal continuation pass reference and E2E checks;
- cache/state/workspace plans match actual allocation peaks for both profiles;
- CLI, OpenAI, Anthropic, tokenizer, template, processor, and model identity behavior are coherent;
- benchmark/profiler reports name execution/checkpoint profile, schema digests, packing target/profile,
  artifact, command, commit, dirty state, GPU, and CUDA;
- NSYS exists for complete 35B inference and NCU exists for every kernel-specific performance
  claim;
- a separate ABI, numerical, lifetime, protocol, and performance review finds no unresolved issue;
- active docs describe implemented registry/Qwen-profile behavior and all completed plans/evidence
  are archived.

## 19. Final Review

This is a high-risk cross-cutting change and requires an independent final review after implementation.

The review is performed in five passes:

1. **ABI/security:** checked arithmetic, generic record/reference bounds, schema-hash encoding,
   storage/view formulas, asset/provenance regions, unknown-id rejection, CRC policy, absence of v4
   fallback, and confirmation that no executable graph semantics entered the container.
2. **Numerical/model:** norm offsets, q/gate de-interleave, GDN head mapping, router/top-k, expert and
   shared FFNs, MTP target distribution, Vision merger, and BF16 rounding.
3. **Lifetime/state:** arena marks, expert descriptors, KV/GDN slots, prefix rewind, graph addresses,
   multimodal workspace, and repeated request recovery.
4. **Protocol/product:** model identity, tokenizer/template behavior, CLI, OpenAI, Anthropic, tool
   history, thinking channels, and report schemas.
5. **Performance:** 27B regression, 35B phase attribution, expert-kernel evidence, MTP acceptance,
   memory matrices, and caller-visible throughput.

Findings are resolved before the final full gate. A near-budget or difficult result is not a reason
to waive a gate.

## 20. Documentation Lifecycle And Archive Destination

While implementation is active, this file remains under `docs/plans/` and is linked from working
changes only where useful. It does not redefine current behavior.

On completion:

- stable system ownership and supported scope are integrated into `docs/design.md`;
- common Qwen3.6 mathematics and both exact Qwen profiles replace the fixed-only Qwen model reference;
  registry/container ownership remains in `docs/design.md` and the active artifact specification;
- any later non-Qwen implementation receives an explicit active model reference linked from
  `docs/README.md`, not additions to a universal artifact-provided graph document;
- q5090 v5 remains the active contract established atomically with the Phase 2 runtime cutover;
- CLI/server behavior is integrated into `docs/serving.md`;
- kernel workflow changes are integrated into `docs/kernel-development.md`;
- README, tools, tests, and bench guides are updated;
- the retired final v4.2 specification remains archived at
  `docs/archive/optimization-era/q5090_packed_file_format_v4_2.md` without overwriting the existing
  historical v4 snapshot;
- this plan and dated implementation/performance evidence move to
  `docs/archive/static-registry-era/plans/`;
- `docs/archive/static-registry-era/README.md` is created to describe the multi-architecture static
  registry era, and
  `docs/archive/README.md` is updated in the same finishing change.

Abandoned or superseded work follows the same archive rule with an explicit historical status.

## 21. Focused Reading List

Before implementing a phase, read only the relevant current authorities plus this plan:

- [`../../AGENTS.md`](../../AGENTS.md) — project scope, compatibility, testing, CUDA, documentation,
  and verification policy;
- [`../design.md`](../design.md) — current ownership and runtime flows;
- [`../qwen3.6-27b-architecture.md`](../qwen3.6-27b-architecture.md) — implemented 27B mathematics,
  MTP, Vision, and state semantics;
- the local 35B `config.json`, safetensors index, and upstream Qwen3.5-MoE mathematical reference;
- the official [Gemma 4 overview](https://ai.google.dev/gemma/docs/core),
  [configuration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/configuration_gemma4.py),
  and [implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modular_gemma4.py)
  only when reviewing the Phase 1 architecture-neutrality fixture;
- [`../q5090_packed_file_format_v4.md`](../q5090_packed_file_format_v4.md) — current container,
  code-plane, segment, fusion, tokenizer, and validation contract to be replaced;
- [`../kernel-development.md`](../kernel-development.md) — L1 contract, oracle, benchmark, NSYS, and
  NCU rules;
- [`../serving.md`](../serving.md) — current CLI/OpenAI/Anthropic/template behavior;
- [`../../tools/q5090_convert/README.md`](../../tools/q5090_convert/README.md) — converter and verifier
  workflow;
- [`../../tools/q5090/README.md`](../../tools/q5090/README.md) — reference and diagnostics;
- [`../../tests/README.md`](../../tests/README.md) and [`../../bench/README.md`](../../bench/README.md)
  — allowed verification and report behavior.

Archived q5090 plans and optimization reports may explain earlier choices but are not current
requirements. Every stable requirement needed by the registry implementation is restated here or in
the active authority that replaces it.
